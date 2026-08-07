"""Building a calibration profile, and computing the H1 go/no-go.

The output that matters is `positive_control_auc`: how well the eye-flow index
separates deliberate closed-eye saccades from closed-eye stillness, in the same
recording session, on the same face, under the same lighting.

Design.md §22 fixes the M1 gate at **AUC >= 0.80 on this waking test**. That is a
deliberately soft bar for a deliberately easy task. The subject is awake,
cooperative, well-lit, frontal, close to the camera, holding their head still,
and making the largest eye movements they can manage. Every one of those
conditions is strictly better than what a sleeping subject offers. Missing 0.80
here does not mean "needs tuning" — it means the signal is not there to be
found, and the honest response is to stop.

Statistics are robust throughout: median and MAD rather than mean and standard
deviation, because a single blink or swallow produces an outlier that would drag
a mean-based baseline above the signal it is supposed to sit beneath.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np

from .protocol import SEGMENTS_BY_KEY, SegmentRole, posture_segments

# Design.md §22, M1 row. Pre-committed; not tunable from the CLI.
POSITIVE_CONTROL_AUC_PASS = 0.80
MIN_SAMPLES_PER_SEGMENT = 30

# Aggregation window for the positive control. One second, because the protocol
# asks for roughly one saccade per second, so each window contains an event and
# the window-level duty cycle approaches 100%. See window_maxima() for why this
# matters more than anything else in this module.
WINDOW_S = 1.0
MIN_WINDOWS = 10

SACCADE_SEGMENTS = ("slow_saccades", "fast_saccades")

# Below this, the "eyes closed and still" segment is genuinely a noise floor.
# Above it, coherent movement is contaminating the baseline and the comparison
# is not measuring what it claims to.
BASELINE_COHERENCE_MAX = 0.35


@dataclass
class SegmentStats:
    key: str
    role: str
    samples: int
    usable_samples: int
    eye_flow_median: Optional[float] = None
    eye_flow_mad: Optional[float] = None
    bilateral_median: Optional[float] = None
    face_visible_fraction: float = 0.0
    eye_usable_fraction: float = 0.0
    interocular_median: Optional[float] = None
    quality_median: Optional[float] = None

    @property
    def coverage_ok(self) -> bool:
        return self.usable_samples >= MIN_SAMPLES_PER_SEGMENT


@dataclass
class CalibrationProfile:
    created_at: str
    segments: dict[str, SegmentStats] = field(default_factory=dict)
    positive_control_auc: Optional[float] = None
    positive_control_detail: str = ""
    windows_positive: int = 0
    windows_baseline: int = 0
    # Second, independent measurement from MediaPipe's tracked lid contour.
    lid_auc: Optional[float] = None
    lid_detail: str = ""
    # V1 validity criterion; unmeasurable on the first two runs.
    baseline_coherence: Optional[float] = None
    head_turn_leakage: Optional[float] = None
    posture_visibility: dict[str, float] = field(default_factory=dict)
    baseline_median: Optional[float] = None
    baseline_mad: Optional[float] = None
    suggested_threshold: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    @property
    def v1_noise_floor_ok(self) -> Optional[bool]:
        """Is the baseline genuinely noise rather than coherent movement?"""
        if self.baseline_coherence is None:
            return None
        return self.baseline_coherence < BASELINE_COHERENCE_MAX

    @property
    def v2_registration_ok(self) -> Optional[bool]:
        """Does the index separate eye motion better than head motion?"""
        if self.head_turn_leakage is None or self.positive_control_auc is None:
            return None
        return self.head_turn_leakage < self.positive_control_auc

    @property
    def verdict(self) -> str:
        """PASS/FAIL only when the instrument is known to be working.

        A gate reading taken through a broken instrument is not a result, so
        the validity criteria are evaluated first and their failure produces NO
        VERDICT rather than a FAIL. Both earlier runs would have returned NO
        VERDICT under this rule.
        """
        if self.positive_control_auc is None:
            return "INSUFFICIENT DATA"
        if self.v1_noise_floor_ok is False or self.v2_registration_ok is False:
            return "NO VERDICT (instrument invalid)"
        if self.v1_noise_floor_ok is None:
            return "NO VERDICT (validity unmeasurable)"
        return "PASS" if self.positive_control_auc >= POSITIVE_CONTROL_AUC_PASS else "FAIL"

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"


def _auc(positive: Sequence[float], negative: Sequence[float]) -> Optional[float]:
    """Rank-based AUC via the Mann-Whitney U statistic.

    Equivalent to the probability that a randomly chosen positive sample exceeds
    a randomly chosen negative one. Distribution-free, which suits a signal
    whose shape is unknown and probably not Gaussian.
    """
    pos = np.asarray([v for v in positive if v is not None and np.isfinite(v)], dtype=float)
    neg = np.asarray([v for v in negative if v is not None and np.isfinite(v)], dtype=float)
    if pos.size < 10 or neg.size < 10:
        return None

    combined = np.concatenate([pos, neg])
    order = combined.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, combined.size + 1)
    # Average ranks within ties, or a signal that is constant-zero in both
    # groups would score 1.0 instead of the correct 0.5.
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]

    rank_sum = ranks[: pos.size].sum()
    u = rank_sum - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def window_maxima(
    samples: Sequence[dict], field: str, *, window_s: float = WINDOW_S
) -> list[float]:
    """Peak value of `field` within each window of the segment.

    This is the correction that invalidated the first two calibration runs.

    A saccade lasts 50-80 ms, which is one or two frames at 30 fps. Comparing
    *frames* labels every frame of a 30-second segment as positive when only
    about 5% of them contain the event, and the AUC of a perfect detector
    collapses toward chance: simulated ceilings are 0.514 at 3% duty, 0.524 at
    5%, 0.549 at 10%. Against a 0.80 gate the test could not be passed by any
    detector, working or not.

    Taking the maximum within a one-second window restores it. At one saccade
    per second every positive window contains an event, so duty cycle at the
    window level is ~100%. On simulated data a 4-sigma signal moves from 0.516
    frame-level to 0.962 windowed — same data, same signal, only the
    aggregation changed.

    Windows are cut on sample timestamps rather than counts, so a dropped frame
    shortens a window instead of shifting every window after it.
    """
    rows = [s for s in samples if s.get(field) is not None and s.get("t_mono") is not None]
    if not rows:
        return []

    out: list[float] = []
    start = rows[0]["t_mono"]
    peak = float("-inf")
    for row in rows:
        if row["t_mono"] - start >= window_s:
            if peak > float("-inf"):
                out.append(peak)
            start, peak = row["t_mono"], float("-inf")
        peak = max(peak, float(row[field]))
    if peak > float("-inf"):
        out.append(peak)
    return out


def _pooled_windows(
    collected: dict[str, list[dict]], keys: Sequence[str], field: str
) -> list[float]:
    values: list[float] = []
    for key in keys:
        values.extend(window_maxima(collected.get(key, []), field))
    return values


def summarise_segment(key: str, samples: Sequence[dict]) -> SegmentStats:
    segment = SEGMENTS_BY_KEY.get(key)
    role = segment.role.value if segment else "unknown"

    flows = [s["eye_flow"] for s in samples if s.get("eye_flow") is not None]
    bilateral = [s["bilateral"] for s in samples if s.get("bilateral") is not None]
    interocular = [s["interocular"] for s in samples if s.get("interocular") is not None]
    quality = [s["quality"] for s in samples if s.get("quality") is not None]

    median = float(np.median(flows)) if flows else None
    mad = float(np.median(np.abs(np.asarray(flows) - median))) if flows else None

    return SegmentStats(
        key=key,
        role=role,
        samples=len(samples),
        usable_samples=len(flows),
        eye_flow_median=median,
        eye_flow_mad=mad,
        bilateral_median=float(np.median(bilateral)) if bilateral else None,
        face_visible_fraction=(
            sum(1 for s in samples if s.get("face_present")) / len(samples) if samples else 0.0
        ),
        eye_usable_fraction=(len(flows) / len(samples)) if samples else 0.0,
        interocular_median=float(np.median(interocular)) if interocular else None,
        quality_median=float(np.median(quality)) if quality else None,
    )


def build_profile(collected: dict[str, list[dict]]) -> CalibrationProfile:
    """Turn per-segment samples into a profile and a verdict."""
    profile = CalibrationProfile(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    for key, samples in collected.items():
        profile.segments[key] = summarise_segment(key, samples)

    baseline_samples = [
        s["eye_flow"]
        for s in collected.get("eyes_closed_still", [])
        if s.get("eye_flow") is not None
    ]
    if baseline_samples:
        profile.baseline_median = float(np.median(baseline_samples))
        profile.baseline_mad = float(
            np.median(np.abs(np.asarray(baseline_samples) - profile.baseline_median))
        )
        # A personalised threshold in robust z units, matching the convention
        # the sensor-timing index uses at night.
        scale = max(profile.baseline_mad * 1.4826, 1e-9)
        profile.suggested_threshold = profile.baseline_median + 1.5 * scale

    # --- coherence, the V1 validity criterion -----------------------------
    # Computed but never surfaced before, which left V1 unmeasurable on both
    # earlier runs. Low baseline coherence means the floor is genuinely noise;
    # high means real coherent movement is leaking into the "still" condition.
    coherence = [
        s["coherence"]
        for s in collected.get("eyes_closed_still", [])
        if s.get("coherence") is not None
    ]
    profile.baseline_coherence = float(np.median(coherence)) if coherence else None

    # --- the go/no-go, on window maxima -----------------------------------
    positive = _pooled_windows(collected, SACCADE_SEGMENTS, "eye_flow")
    baseline_windows = window_maxima(collected.get("eyes_closed_still", []), "eye_flow")

    auc = _auc(positive, baseline_windows)
    profile.positive_control_auc = auc
    profile.windows_positive = len(positive)
    profile.windows_baseline = len(baseline_windows)
    if auc is None:
        shortfall = []
        if len(positive) < 10:
            shortfall.append(f"only {len(positive)} saccade windows")
        if len(baseline_windows) < 10:
            shortfall.append(f"only {len(baseline_windows)} baseline windows")
        profile.positive_control_detail = (
            "; ".join(shortfall) or "not enough usable windows"
        )
    else:
        profile.positive_control_detail = (
            f"{len(positive)} saccade vs {len(baseline_windows)} baseline "
            f"{WINDOW_S:.0f}s windows"
        )

    # --- second, independent measurement: lid-contour geometry ------------
    # MediaPipe tracks the eyelid contour itself, with its own smoothing across
    # the whole face mesh, so this is not subject to the ROI re-centring jitter
    # that afflicts the flow measure. Collected since M1 and never analysed.
    lid_positive = _pooled_windows(collected, SACCADE_SEGMENTS, "lid_disp")
    lid_baseline = window_maxima(collected.get("eyes_closed_still", []), "lid_disp")
    profile.lid_auc = _auc(lid_positive, lid_baseline)
    profile.lid_detail = (
        f"{len(lid_positive)} vs {len(lid_baseline)} windows"
        if lid_positive and lid_baseline
        else "no dense landmarks available"
    )

    # Head-turn leakage: how much a pure head movement looks like eye movement.
    # If it looks as much like a saccade as a saccade does, registration is not
    # working and any nightly signal would be head motion in disguise.
    turn = window_maxima(collected.get("head_turn", []), "eye_flow")
    if turn and baseline_windows:
        leak = _auc(turn, baseline_windows)
        profile.head_turn_leakage = leak
        if leak is not None and auc is not None and leak >= auc - 0.05:
            profile.notes.append(
                "Head turns separate from baseline about as strongly as eye movements do. "
                "The index is probably tracking head motion rather than eye motion, and "
                "a passing AUC above should not be trusted."
            )

    for segment in posture_segments():
        stats = profile.segments.get(segment.key)
        if stats:
            profile.posture_visibility[segment.key] = stats.eye_usable_fraction

    for key, stats in profile.segments.items():
        if not stats.coverage_ok:
            profile.notes.append(
                f"segment '{key}' produced only {stats.usable_samples} usable samples"
            )

    return profile


def save(
    conn: sqlite3.Connection,
    profile: CalibrationProfile,
    *,
    device_profile_id: Optional[int] = None,
    collected: Optional[dict[str, list[dict]]] = None,
) -> int:
    """Persist the profile, and the raw samples when supplied.

    Passing `collected` is strongly preferred. Without it a future failure can
    only be diagnosed by running another session, which is how the coherence
    data needed for validity criterion V1 came to be discarded twice.
    """
    cur = conn.execute(
        "INSERT INTO calibration_profiles (created_at, device_profile_id, "
        "positive_control_auc, head_turn_leakage, baseline_median, baseline_mad, "
        "suggested_threshold, passed, segments_json, posture_json, notes_json, "
        "baseline_coherence, lid_auc, windows_positive, windows_baseline, verdict) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            profile.created_at, device_profile_id,
            profile.positive_control_auc, profile.head_turn_leakage,
            profile.baseline_median, profile.baseline_mad, profile.suggested_threshold,
            int(profile.passed),
            json.dumps({k: vars(v) for k, v in profile.segments.items()}),
            json.dumps(profile.posture_visibility),
            json.dumps(profile.notes),
            profile.baseline_coherence, profile.lid_auc,
            profile.windows_positive, profile.windows_baseline, profile.verdict,
        ),
    )
    profile_id = int(cur.lastrowid)

    if collected:
        rows = [
            (
                profile_id, key, s.get("t_mono"), s.get("eye_flow"), s.get("coherence"),
                s.get("bilateral"), s.get("lid_disp"), s.get("interocular"),
                s.get("quality"), s.get("motion"),
                int(bool(s.get("face_present"))), s.get("coverage"),
            )
            for key, samples in collected.items()
            for s in samples
            if s.get("t_mono") is not None
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO calibration_samples (profile_id, segment_key, "
            "t_mono, eye_flow, coherence, bilateral, lid_disp, interocular, quality, "
            "motion, face_present, coverage) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
    return profile_id


def load_samples(conn: sqlite3.Connection, profile_id: int) -> dict[str, list[dict]]:
    """Re-load raw samples so a profile can be re-analysed without re-recording."""
    out: dict[str, list[dict]] = {}
    for row in conn.execute(
        "SELECT * FROM calibration_samples WHERE profile_id = ? ORDER BY segment_key, t_mono",
        (profile_id,),
    ):
        out.setdefault(row["segment_key"], []).append(dict(row))
    return out


def latest(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM calibration_profiles ORDER BY created_at DESC LIMIT 1"
    ).fetchone()


def format_profile(profile: CalibrationProfile) -> str:
    lines: list[str] = []
    add = lines.append

    add("Calibration profile")
    add("=" * 70)
    add(f"  created  {profile.created_at}")
    add("")

    add("Per-segment summary")
    add("-" * 70)
    add(f"  {'segment':<20} {'role':<10} {'face':>6} {'eye':>6} {'flow med':>10}")
    for key, stats in profile.segments.items():
        flow = f"{stats.eye_flow_median:.4f}" if stats.eye_flow_median is not None else "-"
        add(
            f"  {key:<20} {stats.role:<10} {stats.face_visible_fraction * 100:5.0f}% "
            f"{stats.eye_usable_fraction * 100:5.0f}% {flow:>10}"
        )
    add("")

    if profile.posture_visibility:
        add("Eye-region availability by sleep posture")
        add("-" * 70)
        for key, fraction in profile.posture_visibility.items():
            add(f"  {key:<20} {fraction * 100:5.1f}%")
        add("")

    if profile.baseline_median is not None:
        add("Baseline (eyes closed, still)")
        add("-" * 70)
        add(f"  median               {profile.baseline_median:.5f}")
        add(f"  MAD                  {profile.baseline_mad:.5f}")
        add(f"  suggested threshold  {profile.suggested_threshold:.5f}")
        add("")

    add("INSTRUMENT VALIDITY (checked before any verdict is read)")
    add("-" * 70)
    v1, v2 = profile.v1_noise_floor_ok, profile.v2_registration_ok
    coh = ("%.3f" % profile.baseline_coherence) if profile.baseline_coherence is not None else "-"
    add(f"  V1 noise floor    baseline coherence {coh} "
        f"(< {BASELINE_COHERENCE_MAX})  {_mark(v1)}")
    turn = ("%.3f" % profile.head_turn_leakage) if profile.head_turn_leakage is not None else "-"
    add(f"  V2 registration   head-turn AUC {turn} below saccade AUC  {_mark(v2)}")
    add("")

    add("POSITIVE CONTROL — deliberate closed-eye saccades vs stillness")
    add("=" * 70)
    add(f"  aggregation      {WINDOW_S:.0f}s window maxima, not per-frame")
    if profile.positive_control_auc is None:
        add(f"  no verdict: {profile.positive_control_detail}")
    else:
        add(f"  eye-flow AUC     {profile.positive_control_auc:.3f}  "
            f"({profile.positive_control_detail})")
        add(f"  pass threshold   {POSITIVE_CONTROL_AUC_PASS}")
    if profile.lid_auc is not None:
        add(f"  lid-contour AUC  {profile.lid_auc:.3f}  ({profile.lid_detail})")
        add("                   second, independent measurement from the mesh")
    else:
        add(f"  lid-contour AUC  -  ({profile.lid_detail})")
    add("")
    add(f"  verdict          {profile.verdict}")
    add("")

    for note in profile.notes:
        for line in _wrap(note, 66):
            add(f"  ! {line}")
    if profile.notes:
        add("")

    if profile.verdict.startswith("NO VERDICT"):
        add("  The instrument did not clear its validity checks, so the AUC above")
        add("  is not evidence either way. Fix the instrument, do not read the gate.")
    elif profile.verdict == "PASS":
        add("  Deliberate eye movements are detectable while awake. That is a")
        add("  necessary condition for H1, not a sufficient one — the sleeping")
        add("  case is harder in every respect, and still has to be validated")
        add("  against a reference before it may influence a single cue.")
    elif profile.verdict == "FAIL":
        add("  Large deliberate eye movements are NOT detectable under the easiest")
        add("  conditions this system will ever see: awake, still, frontal, well lit,")
        add("  close to the camera. A sleeping face at an unknown angle in the dark")
        add("  will not do better. Per design.md §23, the eye-movement branch ends")
        add("  here. Keep the camera as a motion guard and run the scheduled arm.")
        add("")
        add("  Do not tune thresholds and re-run until this passes. The threshold is")
        add("  pre-committed precisely so that it cannot be moved after seeing it.")
    return "\n".join(lines)


def _mark(ok: Optional[bool]) -> str:
    return "ok" if ok else ("FAIL" if ok is False else "unmeasurable")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
