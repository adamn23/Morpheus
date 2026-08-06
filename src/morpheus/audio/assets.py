"""Cue assets: synthesis, registration, and provenance.

The trained/untrained distinction is the entire basis of the experiment's
control arm, so it cannot rest on a filename or an operator's memory. Every
asset is hashed on registration, and the hash is recorded with each cue that is
played. Whether a given night used the conditioned cue is therefore a fact
recoverable from the database, not an assertion (design.md §15.1).

The synthesised cues exist so that a matched untrained control can be generated
with the same spectral character as the trained cue but a different melody —
which is what makes the B arm a real control rather than just "some other noise".
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .player import load_wav, write_wav

DEFAULT_SAMPLERATE = 44100


@dataclass(frozen=True)
class CueAsset:
    id: Optional[int]
    name: str
    path: Path
    sha256: str
    trained: bool
    samplerate: int
    duration_s: float


def _bell(freq: float, duration_s: float, samplerate: int, decay: float = 4.0) -> np.ndarray:
    """A soft struck-bell tone: fundamental plus two inharmonic partials.

    Chosen over a pure sine because a bell-like timbre is distinctive enough to
    be recognisable inside a dream while remaining non-startling. Partials are
    deliberately quiet and the decay is fast, so the sound has no sustained
    edge to jolt against.
    """
    t = np.linspace(0.0, duration_s, int(samplerate * duration_s), endpoint=False, dtype=np.float32)
    envelope = np.exp(-decay * t).astype(np.float32)
    tone = np.sin(2 * np.pi * freq * t, dtype=np.float32)
    tone += 0.32 * np.sin(2 * np.pi * freq * 2.76 * t, dtype=np.float32)
    tone += 0.16 * np.sin(2 * np.pi * freq * 5.40 * t, dtype=np.float32)
    return (tone * envelope).astype(np.float32)


def synth_motif(
    notes: list[float],
    *,
    note_s: float = 0.45,
    gap_s: float = 0.10,
    samplerate: int = DEFAULT_SAMPLERATE,
) -> np.ndarray:
    """A short sequence of bell tones. Overlapping decay keeps it from sounding clipped."""
    step = int(samplerate * (note_s + gap_s))
    tail = int(samplerate * note_s)
    total = step * (len(notes) - 1) + tail
    out = np.zeros(total, dtype=np.float32)
    for index, freq in enumerate(notes):
        tone = _bell(freq, note_s, samplerate)
        start = index * step
        out[start:start + tone.size] += tone
    peak = float(np.max(np.abs(out)))
    return out / peak if peak > 1e-9 else out


# Two motifs with matched timbre, note count, duration and register, differing
# only in pitch contour. That matching is what makes the untrained cue a
# control for "a sound occurred" rather than a confound for "a different kind
# of sound occurred".
PRESETS: dict[str, list[float]] = {
    "trained-ascending": [523.25, 659.25, 783.99],   # C5 E5 G5
    "control-descending": [783.99, 659.25, 523.25],  # G5 E5 C5
    "trained-fifth": [587.33, 880.00, 587.33],       # D5 A5 D5
    "control-fourth": [880.00, 587.33, 880.00],      # A5 D5 A5
}


def generate_preset(name: str, samplerate: int = DEFAULT_SAMPLERATE) -> np.ndarray:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return synth_motif(PRESETS[name], samplerate=samplerate)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CueAssetRegistry:
    """Stores cue assets and their hashes."""

    def __init__(self, conn: sqlite3.Connection, asset_dir: Path) -> None:
        self._conn = conn
        self._dir = Path(asset_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def create_preset(self, preset: str, *, trained: bool, name: Optional[str] = None) -> CueAsset:
        samples = generate_preset(preset)
        path = self._dir / f"{name or preset}.wav"
        write_wav(path, samples, DEFAULT_SAMPLERATE)
        return self.register(path, trained=trained, name=name or preset)

    def register(self, path: Path, *, trained: bool, name: Optional[str] = None) -> CueAsset:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        samples, samplerate = load_wav(path)
        digest = sha256_file(path)
        duration = samples.size / samplerate if samplerate else 0.0
        label = name or path.stem

        existing = self._conn.execute(
            "SELECT id FROM cue_assets WHERE sha256 = ?", (digest,)
        ).fetchone()
        if existing:
            # Re-registering identical audio under a different trained flag
            # would make the experiment unanalysable: two cues indistinguishable
            # by content but labelled as different arms.
            row = self._conn.execute(
                "SELECT trained FROM cue_assets WHERE sha256 = ?", (digest,)
            ).fetchone()
            if bool(row["trained"]) != trained:
                raise ValueError(
                    f"audio with hash {digest[:12]}... is already registered with "
                    f"trained={bool(row['trained'])}. The same sound cannot serve as both "
                    f"the trained cue and the control."
                )
            return self.get(int(existing["id"]))

        cur = self._conn.execute(
            "INSERT INTO cue_assets (name, path, sha256, trained, samplerate, duration_s, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                label, str(path), digest, int(trained), samplerate, duration,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        return self.get(int(cur.lastrowid))

    def get(self, asset_id: int) -> CueAsset:
        row = self._conn.execute("SELECT * FROM cue_assets WHERE id = ?", (asset_id,)).fetchone()
        if row is None:
            raise KeyError(f"no cue asset with id {asset_id}")
        return self._row_to_asset(row)

    def list(self, trained: Optional[bool] = None) -> list[CueAsset]:
        if trained is None:
            rows = self._conn.execute("SELECT * FROM cue_assets ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM cue_assets WHERE trained = ? ORDER BY id", (int(trained),)
            ).fetchall()
        return [self._row_to_asset(r) for r in rows]

    def verify(self, asset: CueAsset) -> bool:
        """Confirm the file on disk still matches what was registered.

        Checked before a cue is played. If the audio has been edited or replaced
        since registration, the trained/untrained record is no longer meaningful
        and the night's data would be silently corrupted.
        """
        return asset.path.exists() and sha256_file(asset.path) == asset.sha256

    def load(self, asset: CueAsset) -> tuple[np.ndarray, int]:
        return load_wav(asset.path)

    @staticmethod
    def _row_to_asset(row: sqlite3.Row) -> CueAsset:
        return CueAsset(
            id=int(row["id"]),
            name=row["name"],
            path=Path(row["path"]),
            sha256=row["sha256"],
            trained=bool(row["trained"]),
            samplerate=int(row["samplerate"]),
            duration_s=float(row["duration_s"]),
        )
