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
from .eye_flow import EyeFlowConfig, EyeFlowExtractor, EyeFlowSample
from .landmarks import LandmarkSet, MediaPipeLandmarks, YuNetLandmarks
from .motion import MotionEstimator
from .presence import NO_DETECTOR, PresenceDetector, classify_coverage
from .quality import QualityAssessor, to_gray


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

        # --- M1, shadow mode -------------------------------------------------
        # These features are computed and logged but may not influence cue
        # timing: G9 is locked behind a passing H1 validation regardless of
        # whether this is enabled (design.md §8).
        self.eye_enabled = config.eye.enabled
        self.landmarks: Optional[MediaPipeLandmarks] = None
        self.fallback_landmarks: Optional[YuNetLandmarks] = None
        self.eye_flow: Optional[EyeFlowExtractor] = None
        self._last_landmarks: Optional[LandmarkSet] = None
        self._landmark_interval = 1.0 / max(0.1, config.eye.landmark_hz)
        self._last_landmark_mono: Optional[float] = None
        if self.eye_enabled:
            self.landmarks = MediaPipeLandmarks(
                config.eye.landmark_model,
                min_confidence=config.eye.min_landmark_confidence,
            )
            self.fallback_landmarks = YuNetLandmarks(self.presence)
            self.eye_flow = EyeFlowExtractor(
                EyeFlowConfig(
                    roi_scale=config.eye.roi_scale,
                    max_residual=config.eye.max_registration_residual,
                    equalize=config.eye.equalize_roi,
                )
            )

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

        eye = self._eye_features(frame, quality)
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
            eye=eye,
            landmarks_dense=self._last_landmarks.dense if self._last_landmarks else None,
            pitch=self._last_landmarks.pitch_deg if self._last_landmarks else None,
        )

    def _eye_features(self, frame: Frame, quality) -> Optional[EyeFlowSample]:
        """Landmarks plus eye-region flow. Returns None when unavailable.

        Landmarks run at their own cadence (default 10 Hz) but flow is computed
        every frame against the last known landmark set. That split matters:
        the mesh is the expensive part, while the signal of interest lives in
        frame-to-frame change and would be destroyed by subsampling it.
        """
        if not self.eye_enabled or self.eye_flow is None:
            return None
        if quality.score < self._cfg.quality.min_score * 0.5:
            self._last_landmarks = None
            self.eye_flow.reset()
            return None

        due = (
            self._last_landmark_mono is None
            or (frame.t_mono - self._last_landmark_mono) >= self._landmark_interval
        )
        if due:
            self._last_landmark_mono = frame.t_mono
            found = None
            if self.landmarks is not None and self.landmarks.available:
                found = self.landmarks.detect(frame.image, int(frame.t_mono * 1000))
            if found is None and self.fallback_landmarks is not None:
                # Degrade to five points rather than losing the ROI entirely.
                found = self.fallback_landmarks.from_observation(
                    self._last_presence, (frame.height, frame.width)
                )
            self._last_landmarks = found

        if self._last_landmarks is None:
            self.eye_flow.reset()
            return None
        return self.eye_flow.update(to_gray(frame.image), self._last_landmarks)

    def reset(self) -> None:
        """Clear inter-frame state after a reconnect or a clock discontinuity."""
        self.quality.reset()
        self.motion.reset()
        self._last_presence = NO_DETECTOR
        self._last_detect_mono = None
        self._last_landmarks = None
        self._last_landmark_mono = None
        if self.eye_flow is not None:
            self.eye_flow.reset()
        if self.landmarks is not None:
            self.landmarks.reset()


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
            # --- M1, shadow mode ---------------------------------------------
            # Aggregated only over samples where the measurement existed, and
            # left None when none did. An eye-flow column that reads 0.0 because
            # nothing could be seen is indistinguishable from one that reads 0.0
            # because the eye was still, and H1 would be validated against the
            # difference between those two things.
            landmark_available=_fraction(s.landmarks_dense for s in samples),
            pitch=_mean_or_none(s.pitch for s in samples if s.pitch is not None),
            eye_flow_l=_eye_mean(samples, "flow_left"),
            eye_flow_r=_eye_mean(samples, "flow_right"),
            eye_flow_bilateral_corr=_eye_mean(samples, "bilateral_corr"),
            lid_disp_l=_eye_mean(samples, "lid_disp_left"),
            lid_disp_r=_eye_mean(samples, "lid_disp_right"),
        )


def _eye_mean(samples, attribute: str) -> Optional[float]:
    """Mean of one eye-flow field over the samples that actually produced it."""
    values = [
        getattr(s.eye, attribute)
        for s in samples
        if s.eye is not None and getattr(s.eye, attribute, None) is not None
    ]
    return float(statistics.fmean(values)) if values else None


def _fraction(values) -> Optional[float]:
    """Fraction of non-None values that are truthy; None if all were None."""
    present = [v for v in values if v is not None]
    return (sum(1 for v in present if v) / len(present)) if present else None


def _mean(values) -> float:
    seq = list(values)
    return float(statistics.fmean(seq)) if seq else 0.0


def _mean_or_none(values) -> Optional[float]:
    seq = list(values)
    return float(statistics.fmean(seq)) if seq else None
