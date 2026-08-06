"""Session lifecycle and batched persistence of 1 Hz feature frames."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid as uuid_mod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import MorpheusConfig, git_is_dirty, git_sha
from ..types import FeatureFrame, HealthCounters

_FRAME_COLUMNS = (
    "session_id", "t_mono", "t_utc", "n_frames",
    "signal_quality", "face_present", "eye_region_usable", "coverage_flag",
    "global_motion", "bed_motion", "face_motion",
    "yaw_proxy", "roll_deg", "interocular_px", "focus", "luminance_mean",
    "scene_change",
    "landmark_available", "pitch", "head_motion",
    "eye_flow_l", "eye_flow_r", "eye_flow_bilateral_corr",
    "lid_disp_l", "lid_disp_r", "resp_proxy",
)

_INSERT_FRAME = (
    f"INSERT OR REPLACE INTO frames_1hz ({', '.join(_FRAME_COLUMNS)}) "
    f"VALUES ({', '.join('?' * len(_FRAME_COLUMNS))})"
)


def _utc_iso(epoch: Optional[float] = None) -> str:
    ts = datetime.now(timezone.utc) if epoch is None else datetime.fromtimestamp(epoch, timezone.utc)
    return ts.isoformat(timespec="milliseconds")


class FeatureStore:
    """Writes sessions and feature frames.

    Frames are buffered and committed in batches to keep the capture loop free
    of per-second fsync latency. `flush()` is called on every batch boundary, on
    session end, and by the recorder's signal handler, so an abrupt stop loses
    at most `write_batch_size` seconds.
    """

    def __init__(self, conn: sqlite3.Connection, batch_size: int = 30) -> None:
        self._conn = conn
        self._batch_size = max(1, batch_size)
        self._buffer: list[tuple] = []
        self.session_id: Optional[int] = None
        self.session_uuid: Optional[str] = None

    # ---------------------------------------------------------------- setup

    def ensure_config_snapshot(self, config: MorpheusConfig, repo: Optional[Path] = None) -> int:
        fingerprint = config.fingerprint()
        row = self._conn.execute(
            "SELECT id FROM config_snapshots WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        if row:
            return int(row["id"])
        dirty = git_is_dirty(repo)
        cur = self._conn.execute(
            "INSERT INTO config_snapshots (created_at, fingerprint, config_json, git_sha, git_dirty) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                _utc_iso(),
                fingerprint,
                config.canonical_json(),
                git_sha(repo),
                None if dirty is None else int(dirty),
            ),
        )
        return int(cur.lastrowid)

    def ensure_device_profile(self, profile: dict[str, Any]) -> int:
        """Upsert a device profile keyed by a fingerprint of its fields."""
        import hashlib

        canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"), default=str)
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        row = self._conn.execute(
            "SELECT id FROM device_profiles WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
        if row:
            return int(row["id"])
        cols = [
            "camera_model", "backend", "width", "height", "fps", "fourcc",
            "manual_exposure", "ir_wavelength_nm", "mount_geometry", "audio_device",
        ]
        values = [profile.get(c) for c in cols]
        cur = self._conn.execute(
            f"INSERT INTO device_profiles (created_at, fingerprint, {', '.join(cols)}) "
            f"VALUES (?, ?, {', '.join('?' * len(cols))})",
            (_utc_iso(), fingerprint, *values),
        )
        return int(cur.lastrowid)

    # -------------------------------------------------------------- session

    def start_session(
        self,
        *,
        config: MorpheusConfig,
        device_profile: dict[str, Any],
        kind: str = "probe",
        notes: Optional[str] = None,
        repo: Optional[Path] = None,
        version: str = "0.1.0.dev0",
    ) -> int:
        config_id = self.ensure_config_snapshot(config, repo=repo)
        device_id = self.ensure_device_profile(device_profile)

        # night_index counts probe/recording nights so that analysis can absorb
        # practice effects, which are large in lucid-dream training.
        row = self._conn.execute("SELECT COALESCE(MAX(night_index), 0) FROM sessions").fetchone()
        night_index = int(row[0]) + 1

        self.session_uuid = str(uuid_mod.uuid4())
        cur = self._conn.execute(
            "INSERT INTO sessions (uuid, started_at_utc, started_at_mono, status, kind, "
            "night_index, device_profile_id, config_snapshot_id, morpheus_version, notes) "
            "VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?)",
            (
                self.session_uuid, _utc_iso(), time.monotonic(), kind,
                night_index, device_id, config_id, version, notes,
            ),
        )
        self.session_id = int(cur.lastrowid)
        self._conn.execute(
            "INSERT INTO session_health (session_id, updated_at) VALUES (?, ?)",
            (self.session_id, _utc_iso()),
        )
        return self.session_id

    def finish_session(self, status: str, health: Optional[HealthCounters] = None) -> None:
        if self.session_id is None:
            return
        self.flush()
        self._conn.execute(
            "UPDATE sessions SET ended_at_utc = ?, ended_at_mono = ?, status = ? WHERE id = ?",
            (_utc_iso(), time.monotonic(), status, self.session_id),
        )
        if health is not None:
            self.update_health(health)

    def update_health(self, health: HealthCounters, peak_rss_mb: Optional[float] = None) -> None:
        if self.session_id is None:
            return
        self._conn.execute(
            "UPDATE session_health SET frames_captured = ?, frames_dropped = ?, "
            "read_failures = ?, reconnects = ?, seconds_recorded = ?, capture_uptime = ?, "
            "clock_gaps_json = ?, peak_rss_mb = COALESCE(?, peak_rss_mb), updated_at = ? "
            "WHERE session_id = ?",
            (
                health.frames_captured, health.frames_dropped, health.read_failures,
                health.reconnects, health.seconds_recorded, health.capture_uptime,
                json.dumps(health.clock_gaps), peak_rss_mb, _utc_iso(), self.session_id,
            ),
        )

    # --------------------------------------------------------------- frames

    def append(self, ff: FeatureFrame) -> None:
        if self.session_id is None:
            raise RuntimeError("start_session() must be called before append()")
        self._buffer.append((
            self.session_id, ff.t_mono, _utc_iso(ff.t_utc), ff.n_frames,
            ff.signal_quality, ff.face_present, ff.eye_region_usable,
            ff.coverage_flag.value,
            ff.global_motion, ff.bed_motion, ff.face_motion,
            ff.yaw_proxy, ff.roll_deg, ff.interocular_px, ff.focus, ff.luminance_mean,
            ff.scene_change,
            ff.landmark_available, ff.pitch, ff.head_motion,
            ff.eye_flow_l, ff.eye_flow_r, ff.eye_flow_bilateral_corr,
            ff.lid_disp_l, ff.lid_disp_r, ff.resp_proxy,
        ))
        if len(self._buffer) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        self._conn.execute("BEGIN")
        try:
            self._conn.executemany(_INSERT_FRAME, self._buffer)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        self._buffer.clear()

    # ------------------------------------------------------------ lifecycle

    def __enter__(self) -> "FeatureStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.flush()
        finally:
            if self.session_id is not None:
                status = "completed" if exc_type is None else "aborted"
                self._conn.execute(
                    "UPDATE sessions SET ended_at_utc = COALESCE(ended_at_utc, ?), "
                    "status = CASE WHEN status = 'running' THEN ? ELSE status END WHERE id = ?",
                    (_utc_iso(), status, self.session_id),
                )
