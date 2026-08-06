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
    head_turn_leakage: Optional[float] = None
    posture_visibility: dict[str, float] = field(default_factory=dict)
    baseline_median: Optional[float] = None
    baseline_mad: Optional[float] = None
    suggested_threshold: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.positive_control_auc is None:
            return "INSUFFICIENT DATA"
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

    # --- the go/no-go -----------------------------------------------------
    positive: list[float] = []
    for key in ("slow_saccades", "fast_saccades"):
        positive.extend(
            s["eye_flow"] for s in collected.get(key, []) if s.get("eye_flow") is not None
        )

    auc = _auc(positive, baseline_samples)
    profile.positive_control_auc = auc
    if auc is None:
        shortfall = []
        if len(positive) < 10:
            shortfall.append(f"only {len(positive)} usable saccade samples")
        if len(baseline_samples) < 10:
            shortfall.append(f"only {len(baseline_samples)} usable baseline samples")
        profile.positive_control_detail = (
            "; ".join(shortfall) or "not enough usable samples"
        )
    else:
        profile.positive_control_detail = (
            f"{len(positive)} saccade vs {len(baseline_samples)} baseline samples"
        )

    # Head-turn leakage: how much a pure head movement looks like eye movement.
    # If it looks as much like a saccade as a saccade does, registration is not
    # working and any nightly signal would be head motion in disguise.
    turn = [
        s["eye_flow"] for s in collected.get("head_turn", []) if s.get("eye_flow") is not None
    ]
    if turn and baseline_samples:
        leak = _auc(turn, baseline_samples)
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


def save(conn: sqlite3.Connection, profile: CalibrationProfile, *, device_profile_id: Optional[int] = None) -> int:
    cur = conn.execute(
        "INSERT INTO calibration_profiles (created_at, device_profile_id, "
        "positive_control_auc, head_turn_leakage, baseline_median, baseline_mad, "
        "suggested_threshold, passed, segments_json, posture_json, notes_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            profile.created_at, device_profile_id,
            profile.positive_control_auc, profile.head_turn_leakage,
            profile.baseline_median, profile.baseline_mad, profile.suggested_threshold,
            int(profile.passed),
            json.dumps({k: vars(v) for k, v in profile.segments.items()}),
            json.dumps(profile.posture_visibility),
            json.dumps(profile.notes),
        ),
    )
    return int(cur.lastrowid)


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

    add("POSITIVE CONTROL — deliberate closed-eye saccades vs stillness")
    add("=" * 70)
    if profile.positive_control_auc is None:
        add(f"  no verdict: {profile.positive_control_detail}")
    else:
        add(f"  AUC              {profile.positive_control_auc:.3f}  "
            f"({profile.positive_control_detail})")
        add(f"  pass threshold   {POSITIVE_CONTROL_AUC_PASS}")
        if profile.head_turn_leakage is not None:
            add(f"  head-turn AUC    {profile.head_turn_leakage:.3f}  (should be well below)")
    add(f"  verdict          {profile.verdict}")
    add("")

    for note in profile.notes:
        for line in _wrap(note, 66):
            add(f"  ! {line}")
    if profile.notes:
        add("")

    if profile.verdict == "PASS":
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
