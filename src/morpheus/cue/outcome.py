"""Classifying what happened after a cue.

This is the closed part of the closed loop, and the component most exposed to
the honesty problem the naming rules exist for. The camera sees motion. Motion
after a cue is *consistent with* the cue having roused the sleeper, and is also
consistent with a spontaneous position change that would have happened anyway.
Nothing here can distinguish those, so nothing here claims to.

What the classifier does instead is bound the damage. `probable_arousal` backs
the volume off; `possible_awakening` stops the night. Both err toward stopping,
because a false positive costs one missed cue and a false negative costs sleep.

Spontaneous movement during sleep is common, so the false-positive rate on
`probable_arousal` will be substantial. That is an accepted cost, and it is
measurable: the design's M2 acceptance criterion tracks cue-attributed
awakenings, and comparing the post-cue motion rate against the baseline rate
during non-cue windows is what separates real cue effects from ordinary
restlessness (design.md §22, §23).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ..types import FeatureFrame
from .state import Outcome


@dataclass
class OutcomeThresholds:
    """Motion thresholds for classifying a post-cue window.

    Expressed as multiples of the pre-cue baseline rather than absolutes,
    because motion energy scales with camera distance, resolution, and how much
    of the frame the sleeper occupies. A fixed threshold would mean something
    different in every bedroom.
    """

    observe_window_s: float = 90.0
    baseline_window_s: float = 120.0

    arousal_ratio: float = 2.5      # post-cue motion vs pre-cue baseline
    arousal_absolute: float = 0.004  # floor, for when baseline is near zero

    awakening_ratio: float = 5.0
    awakening_absolute: float = 0.012
    # Sustained motion for this long is treated as leaving sleep rather than a
    # brief stir.
    awakening_sustained_s: float = 20.0

    min_quality: float = 0.25
    min_frames: int = 10


@dataclass
class OutcomeAssessment:
    outcome: Outcome
    motion_before: float
    motion_after: float
    motion_delta: float
    latency_to_motion_ms: Optional[float]
    quality_during: float
    coverage_during: float
    detail: str = ""


def classify_post_cue(
    *,
    cue_t_mono: float,
    before: Sequence[FeatureFrame],
    after: Sequence[FeatureFrame],
    thresholds: Optional[OutcomeThresholds] = None,
) -> OutcomeAssessment:
    """Classify the response to a cue from the surrounding feature frames."""
    th = thresholds or OutcomeThresholds()

    quality = _mean(f.signal_quality for f in after)
    coverage = _mean(f.face_present for f in after)

    if len(after) < th.min_frames or quality < th.min_quality:
        # Do not learn from a window we could not see. An `uncertain` outcome
        # leaves the adaptive layer untouched, which is the correct response to
        # missing information (design.md §12.4).
        return OutcomeAssessment(
            outcome=Outcome.UNCERTAIN,
            motion_before=_mean(f.global_motion for f in before),
            motion_after=_mean(f.global_motion for f in after),
            motion_delta=0.0,
            latency_to_motion_ms=None,
            quality_during=quality,
            coverage_during=coverage,
            detail=(
                f"insufficient signal: {len(after)} frames at quality {quality:.2f}"
            ),
        )

    baseline = _mean(f.global_motion for f in before)
    peak_after = max((f.global_motion for f in after), default=0.0)
    mean_after = _mean(f.global_motion for f in after)
    delta = mean_after - baseline

    arousal_level = max(baseline * th.arousal_ratio, th.arousal_absolute)
    awakening_level = max(baseline * th.awakening_ratio, th.awakening_absolute)

    sustained_s = _sustained_seconds(after, arousal_level)
    latency = _latency_ms(cue_t_mono, after, arousal_level)

    if peak_after >= awakening_level and sustained_s >= th.awakening_sustained_s:
        outcome = Outcome.POSSIBLE_AWAKENING
        detail = (
            f"peak {peak_after:.4f} >= {awakening_level:.4f} sustained {sustained_s:.0f}s"
        )
    elif peak_after >= arousal_level:
        outcome = Outcome.PROBABLE_AROUSAL
        detail = f"peak {peak_after:.4f} >= {arousal_level:.4f} (baseline {baseline:.4f})"
    else:
        outcome = Outcome.QUIET
        detail = f"peak {peak_after:.4f} below {arousal_level:.4f}"

    return OutcomeAssessment(
        outcome=outcome,
        motion_before=baseline,
        motion_after=mean_after,
        motion_delta=delta,
        latency_to_motion_ms=latency,
        quality_during=quality,
        coverage_during=coverage,
        detail=detail,
    )


def _sustained_seconds(after: Sequence[FeatureFrame], level: float) -> float:
    """Elapsed time above `level`, measured from timestamps.

    Counting frames instead would silently assume exactly 1 Hz. That happens to
    be true of the production aggregator, which is precisely what makes the
    assumption dangerous: it would hold in every real run and break only when
    the frame rate changed, in the component that decides whether to stop
    cueing for the night.
    """
    total = 0.0
    for previous, current in zip(after, after[1:]):
        if current.global_motion >= level:
            total += max(0.0, current.t_mono - previous.t_mono)
    return total


def _latency_ms(
    cue_t_mono: float, after: Sequence[FeatureFrame], level: float
) -> Optional[float]:
    for frame in after:
        if frame.global_motion >= level:
            return max(0.0, (frame.t_mono - cue_t_mono) * 1000.0)
    return None


def _mean(values) -> float:
    seq = list(values)
    return float(sum(seq) / len(seq)) if seq else 0.0
