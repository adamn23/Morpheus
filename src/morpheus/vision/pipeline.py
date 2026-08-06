"""Per-frame orchestration and aggregation to the 1 Hz persisted unit."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Optional

from ..config import MorpheusConfig
from ..types import (
    CoverageFlag,
    FeatureFrame,
    Frame,
    PresenceObservation,
    RawSample,
)
from .motion import MotionEstimator
from .presence import NO_DETECTOR, PresenceDetector, classify_coverage
from .quality import QualityAssessor


class VisionPipeline:
    """Frame in, RawSample out.

    Presence detection runs at `presence.detect_hz`, not per frame: it is by far
    the most expensive stage, and on a passively-cooled laptop the CPU budget is
    the binding constraint for an eight-hour run. Between detections the last
    observation is carried forward and marked stale.
    """

    def __init__(
        self,
        config: MorpheusConfig,
        *,
        presence_detector: Optional[PresenceDetector] = None,
    ) -> None:
        self._cfg = config
        self.quality = QualityAssessor(config.quality, config.motion.scene_change_downscale)
        # Injectable so integration tests can substitute a scripted detector.
        # YuNet does not fire on synthetic imagery, so the real detector can
        # only be exercised against live footage — which is a task for M0 setup
        # with the actual camera, not something the test suite can fake.
        self.presence = presence_detector or PresenceDetector(config.presence)
        self.motion = MotionEstimator(config.motion)
        self._last_presence: PresenceObservation = NO_DETECTOR
        self._last_detect_mono: Optional[float] = None
        self._detect_interval = 1.0 / max(0.1, config.presence.detect_hz)

    def process(self, frame: Frame) -> RawSample:
        quality = self.quality.assess(frame.image)

        due = (
            self._last_detect_mono is None
            or (frame.t_mono - self._last_detect_mono) >= self._detect_interval
        )
        # Skip detection outright on hopeless frames; it would fail anyway and
        # it is the expensive stage.
        if due and quality.score >= self._cfg.quality.min_score * 0.5:
            self._last_presence = self.presence.detect(frame.image)
            self._last_detect_mono = frame.t_mono
            fresh = True
        elif due:
            self._last_presence = (
                NO_DETECTOR if not self.presence.available else PresenceObservation(False)
            )
            self._last_detect_mono = frame.t_mono
            fresh = True
        else:
            fresh = False

        motion = self.motion.update(frame.image, self._last_presence)
        coverage = classify_coverage(
            self._last_presence,
            quality,
            (frame.height, frame.width),
            self._cfg.coverage,
        )

        return RawSample(
            seq=frame.seq,
            t_mono=frame.t_mono,
            t_utc=frame.t_utc,
            quality=quality,
            presence=self._last_presence,
            motion=motion,
            coverage=coverage,
            presence_fresh=fresh,
        )

    def reset(self) -> None:
        """Clear inter-frame state after a reconnect or a clock discontinuity."""
        self.quality.reset()
        self.motion.reset()
        self._last_presence = NO_DETECTOR
        self._last_detect_mono = None


class Aggregator:
    """Collapses RawSamples into one FeatureFrame per wall-clock second."""

    def __init__(self) -> None:
        self._bucket: Optional[int] = None
        self._samples: list[RawSample] = []

    def add(self, sample: RawSample) -> Optional[FeatureFrame]:
        bucket = int(sample.t_mono)
        completed: Optional[FeatureFrame] = None
        if self._bucket is None:
            self._bucket = bucket
        elif bucket != self._bucket:
            completed = self._emit()
            self._bucket = bucket
        self._samples.append(sample)
        return completed

    def flush(self) -> Optional[FeatureFrame]:
        """Emit the trailing partial second, if any."""
        return self._emit() if self._samples else None

    def _emit(self) -> Optional[FeatureFrame]:
        samples, self._samples = self._samples, []
        if not samples:
            return None

        # Fractions are computed over freshly-evaluated samples only. Counting
        # carried-forward detections would report the detector's cadence rather
        # than the fraction of the second the face was actually visible.
        fresh = [s for s in samples if s.presence_fresh] or samples

        face_hits = [s for s in fresh if s.presence.face_present]
        usable = [s for s in fresh if s.coverage is CoverageFlag.USABLE]

        return FeatureFrame(
            t_mono=float(samples[0].t_mono),
            t_utc=float(samples[0].t_utc),
            n_frames=len(samples),
            signal_quality=_mean(s.quality.score for s in samples),
            face_present=len(face_hits) / len(fresh),
            eye_region_usable=len(usable) / len(fresh),
            coverage_flag=Counter(s.coverage for s in fresh).most_common(1)[0][0],
            global_motion=_mean(s.motion.global_motion for s in samples),
            bed_motion=_mean(s.motion.bed_motion for s in samples),
            face_motion=_mean(s.motion.face_motion for s in samples),
            yaw_proxy=_mean_or_none(s.presence.yaw_proxy for s in face_hits),
            roll_deg=_mean_or_none(s.presence.roll_deg for s in face_hits),
            interocular_px=_mean_or_none(s.presence.interocular_px for s in face_hits),
            focus=_mean(s.quality.focus for s in samples),
            luminance_mean=_mean(s.quality.luminance_mean for s in samples),
            scene_change=max(s.quality.scene_change for s in samples),
        )


def _mean(values) -> float:
    seq = list(values)
    return float(statistics.fmean(seq)) if seq else 0.0


def _mean_or_none(values) -> Optional[float]:
    seq = list(values)
    return float(statistics.fmean(seq)) if seq else None
