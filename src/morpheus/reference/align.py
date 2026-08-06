"""Aligning Morpheus feature frames to reference epochs.

Two clocks that were never synchronised have to be joined to within a fraction
of a 30-second epoch, or the labels smear across state boundaries and depress
the measured AUC for reasons that have nothing to do with the camera. Getting
this wrong makes a working detector look broken.

The offset is estimated by cross-correlating gross body motion against the
reference's wake epochs. Wake is when people move, and it is the one state both
instruments observe through completely different physics — which is exactly what
makes it a usable synchronisation marker.

The estimate is reported with a confidence figure rather than applied silently.
If the correlation is weak the honest move is to say so and let the operator
supply a known offset, not to shift the data by a number nobody trusts.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

from .ingest import EPOCH_SECONDS, ReferenceEpoch
from .validate import FEATURE_COLUMNS


@dataclass
class AlignmentResult:
    offset_s: float
    correlation: float
    method: str
    searched_range_s: tuple[float, float] = (0.0, 0.0)

    @property
    def trustworthy(self) -> bool:
        """Whether the estimate is strong enough to use unattended.

        0.3 is a low bar, chosen because the two signals measure genuinely
        different things and a modest correlation is the realistic best case.
        Below it, the offset is a guess and should be supplied by hand.
        """
        return self.correlation >= 0.30


@dataclass
class AlignedDataset:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    feature_names: list[str]
    epochs_matched: int = 0
    epochs_dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)


def _to_utc_seconds(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def load_session_frames(
    conn: sqlite3.Connection, session_id: int, columns: Sequence[str] = FEATURE_COLUMNS
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_utc, feature matrix) for one session, NaN where unpopulated."""
    selected = ", ".join(columns)
    rows = conn.execute(
        f"SELECT t_utc, {selected} FROM frames_1hz WHERE session_id = ? ORDER BY t_mono",
        (session_id,),
    ).fetchall()
    if not rows:
        return np.empty(0), np.empty((0, len(columns)))

    times = np.array([_to_utc_seconds(r["t_utc"]) for r in rows], dtype=float)
    values = np.array(
        [[np.nan if r[c] is None else float(r[c]) for c in columns] for r in rows],
        dtype=float,
    )
    return times, values


def estimate_offset(
    frame_times: np.ndarray,
    motion: np.ndarray,
    epochs: Sequence[ReferenceEpoch],
    *,
    search_s: float = 900.0,
    step_s: float = 5.0,
) -> AlignmentResult:
    """Cross-correlate camera motion against reference wake epochs."""
    scored = [e for e in epochs if e.is_scored]
    if len(scored) < 20 or frame_times.size < 60:
        return AlignmentResult(0.0, 0.0, "insufficient data", (-search_s, search_s))

    epoch_times = np.array([e.t_utc for e in scored], dtype=float)
    is_wake = np.array([1.0 if e.stage == "W" else 0.0 for e in scored], dtype=float)
    if is_wake.std() < 1e-9:
        return AlignmentResult(0.0, 0.0, "reference has no wake epochs", (-search_s, search_s))

    finite = np.isfinite(motion)
    if finite.sum() < 60:
        return AlignmentResult(0.0, 0.0, "no usable motion signal", (-search_s, search_s))
    times, values = frame_times[finite], motion[finite]

    best_offset, best_corr = 0.0, -1.0
    for offset in np.arange(-search_s, search_s + step_s, step_s):
        # Mean camera motion inside each reference epoch, at this offset.
        binned = np.empty(epoch_times.size)
        for index, start in enumerate(epoch_times):
            mask = (times + offset >= start) & (times + offset < start + EPOCH_SECONDS)
            binned[index] = values[mask].mean() if mask.any() else np.nan
        usable = np.isfinite(binned)
        if usable.sum() < 20 or binned[usable].std() < 1e-12:
            continue
        corr = float(np.corrcoef(binned[usable], is_wake[usable])[0, 1])
        if np.isfinite(corr) and corr > best_corr:
            best_offset, best_corr = float(offset), corr

    return AlignmentResult(
        offset_s=best_offset,
        correlation=max(0.0, best_corr),
        method="motion vs wake cross-correlation",
        searched_range_s=(-search_s, search_s),
    )


def build_dataset(
    conn: sqlite3.Connection,
    session_epochs: dict[int, Sequence[ReferenceEpoch]],
    *,
    offsets: Optional[dict[int, float]] = None,
    columns: Sequence[str] = FEATURE_COLUMNS,
    require_sleep: bool = True,
) -> AlignedDataset:
    """Join camera features to reference labels, one row per 30-second epoch.

    Wake epochs are excluded by default. The question is whether the index
    separates REM from *other sleep*, and leaving wake in would inflate the AUC
    on the easiest possible contrast — a moving, sometimes-visible awake face
    against a still sleeping one. That would be a real effect and an irrelevant
    one.
    """
    offsets = offsets or {}
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    dropped: dict[str, int] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for session_id, epochs in session_epochs.items():
        times, values = load_session_frames(conn, session_id, columns)
        if times.size == 0:
            drop("session has no frames")
            continue
        offset = offsets.get(session_id, 0.0)
        shifted = times + offset

        for epoch in epochs:
            if not epoch.is_scored:
                drop("epoch unscored")
                continue
            if require_sleep and not epoch.is_sleep:
                drop("wake epoch excluded")
                continue

            mask = (shifted >= epoch.t_utc) & (shifted < epoch.t_utc + EPOCH_SECONDS)
            if mask.sum() < EPOCH_SECONDS * 0.5:
                drop("insufficient camera coverage in epoch")
                continue

            window = values[mask]
            with np.errstate(invalid="ignore"):
                summary = np.nanmean(window, axis=0)
            if not np.isfinite(summary).any():
                drop("all features missing in epoch")
                continue

            rows.append(summary)
            labels.append(1 if epoch.reference_scored_rem else 0)
            groups.append(session_id)

    if not rows:
        return AlignedDataset(
            features=np.empty((0, len(columns))),
            labels=np.empty(0, dtype=int),
            groups=np.empty(0, dtype=int),
            feature_names=list(columns),
            epochs_dropped=sum(dropped.values()),
            drop_reasons=dropped,
        )

    matrix = np.vstack(rows)

    # Drop feature columns that are entirely missing — in M0/M1 the eye-flow
    # columns are NULL by design, and passing all-NaN columns to the classifier
    # would fail rather than degrade.
    keep = [i for i in range(matrix.shape[1]) if np.isfinite(matrix[:, i]).any()]
    matrix = matrix[:, keep]
    names = [columns[i] for i in keep]

    # Median-impute what remains, so an occasional missing epoch does not
    # discard an otherwise usable row.
    for column in range(matrix.shape[1]):
        values_col = matrix[:, column]
        missing = ~np.isfinite(values_col)
        if missing.any():
            values_col[missing] = np.nanmedian(values_col[~missing]) if (~missing).any() else 0.0

    return AlignedDataset(
        features=matrix,
        labels=np.array(labels, dtype=int),
        groups=np.array(groups, dtype=int),
        feature_names=names,
        epochs_matched=len(labels),
        epochs_dropped=sum(dropped.values()),
        drop_reasons=dropped,
    )
