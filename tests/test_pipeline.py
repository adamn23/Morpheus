"""Vision pipeline and 1 Hz aggregation."""

from __future__ import annotations

import time

import numpy as np
import pytest

from morpheus.config import MorpheusConfig
from morpheus.types import CoverageFlag, Frame, PresenceObservation
from morpheus.vision.motion import MotionEstimator
from morpheus.vision.pipeline import Aggregator, VisionPipeline
from morpheus.vision.quality import QualityAssessor

from .conftest import ScriptedDetector, face_for_frame, gray_frame, visible_face


def frames_at(fps: float, count: int, image_fn) -> list[Frame]:
    base = time.time()
    return [
        Frame(seq=i, t_mono=i / fps, t_utc=base + i / fps, image=image_fn(i))
        for i in range(count)
    ]


# ------------------------------------------------------------------ aggregation


def test_aggregator_emits_one_frame_per_second(config: MorpheusConfig) -> None:
    detector = ScriptedDetector([visible_face()])
    pipeline = VisionPipeline(config, presence_detector=detector)
    agg = Aggregator()

    emitted = []
    for frame in frames_at(30.0, 95, lambda i: gray_frame(noise=3, seed=i)):
        out = agg.add(pipeline.process(frame))
        if out:
            emitted.append(out)
    tail = agg.flush()
    if tail:
        emitted.append(tail)

    assert len(emitted) == 4  # 3 full seconds + a 5-frame remainder
    assert [f.n_frames for f in emitted[:3]] == [30, 30, 30]
    assert emitted[-1].n_frames == 5


def test_face_fraction_counts_fresh_detections_only(config: MorpheusConfig) -> None:
    """Regression guard for a subtle and very plausible bug.

    Presence runs at 5 Hz while frames arrive at 30 Hz. If aggregation counted
    every sample, a continuously visible face would report as ~17% present —
    the detector's duty cycle rather than the face's. That would have silently
    understated the M0 decision-gate number by a factor of six.
    """
    config.presence.detect_hz = 5.0
    detector = ScriptedDetector([face_for_frame()])
    pipeline = VisionPipeline(config, presence_detector=detector)
    agg = Aggregator()

    emitted = []
    for frame in frames_at(30.0, 60, lambda i: gray_frame(noise=3, seed=i)):
        out = agg.add(pipeline.process(frame))
        if out:
            emitted.append(out)

    assert emitted, "expected at least one completed second"
    assert emitted[0].face_present == pytest.approx(1.0)
    assert emitted[0].eye_region_usable == pytest.approx(1.0)
    assert detector.calls <= 12, "detector ran more often than its configured cadence"


def test_absent_face_yields_zero_coverage(config: MorpheusConfig) -> None:
    pipeline = VisionPipeline(config, presence_detector=ScriptedDetector([]))
    agg = Aggregator()
    emitted = [
        out
        for frame in frames_at(30.0, 60, lambda i: gray_frame(noise=3, seed=i))
        if (out := agg.add(pipeline.process(frame)))
    ]
    assert emitted[0].eye_region_usable == 0.0
    assert emitted[0].coverage_flag is CoverageFlag.FACE_ABSENT


def test_pose_features_are_none_when_no_face(config: MorpheusConfig) -> None:
    """Missingness must stay missing. Never impute a pose that was not seen."""
    pipeline = VisionPipeline(config, presence_detector=ScriptedDetector([]))
    agg = Aggregator()
    emitted = [
        out
        for frame in frames_at(30.0, 40, lambda i: gray_frame(noise=3, seed=i))
        if (out := agg.add(pipeline.process(frame)))
    ]
    assert emitted[0].yaw_proxy is None
    assert emitted[0].interocular_px is None


def test_later_phase_columns_stay_unpopulated(config: MorpheusConfig) -> None:
    """M0 must not invent eye-flow values. Shadow-mode features arrive in M1."""
    pipeline = VisionPipeline(config, presence_detector=ScriptedDetector([face_for_frame()]))
    agg = Aggregator()
    emitted = [
        out
        for frame in frames_at(30.0, 40, lambda i: gray_frame(noise=3, seed=i))
        if (out := agg.add(pipeline.process(frame)))
    ]
    ff = emitted[0]
    assert ff.eye_flow_l is None and ff.eye_flow_r is None
    assert ff.eye_flow_bilateral_corr is None
    assert ff.landmark_available is None


def test_reset_clears_inter_frame_state(config: MorpheusConfig) -> None:
    pipeline = VisionPipeline(config, presence_detector=ScriptedDetector([visible_face()]))
    for frame in frames_at(30.0, 10, lambda i: gray_frame(noise=3, seed=i)):
        pipeline.process(frame)
    pipeline.reset()
    # After a reset the first frame has no predecessor, so motion must be zero
    # rather than a spurious spike against a pre-gap frame.
    sample = pipeline.process(frames_at(30.0, 1, lambda i: gray_frame(value=200))[0])
    assert sample.motion.global_motion == 0.0


# ---------------------------------------------------------------------- motion


def test_motion_zero_on_identical_frames(config: MorpheusConfig) -> None:
    est = MotionEstimator(config.motion)
    img = gray_frame()
    est.update(img)
    assert est.update(img).global_motion == pytest.approx(0.0, abs=1e-9)


def test_motion_responds_to_change(config: MorpheusConfig) -> None:
    est = MotionEstimator(config.motion)
    est.update(gray_frame(value=40))
    moved = est.update(gray_frame(value=140))
    assert moved.global_motion > 0.1


def test_motion_is_monotonic_in_magnitude(config: MorpheusConfig) -> None:
    """A bigger change must not report as less motion."""
    readings = []
    for delta in (10, 40, 90):
        est = MotionEstimator(config.motion)
        est.update(gray_frame(value=40))
        readings.append(est.update(gray_frame(value=40 + delta)).global_motion)
    assert readings == sorted(readings)


def test_motion_works_without_a_face(config: MorpheusConfig) -> None:
    """The gate-and-guard role must not depend on presence detection.

    This is the property that makes the recommended architecture viable for a
    side sleeper: motion is available even when the face never is.
    """
    est = MotionEstimator(config.motion)
    est.update(gray_frame(value=40), PresenceObservation(face_present=False))
    obs = est.update(gray_frame(value=90), PresenceObservation(face_present=False))
    assert obs.global_motion > 0.0
    assert obs.face_motion == 0.0


# --------------------------------------------------------------------- quality


def test_quality_low_for_black_frame(config: MorpheusConfig) -> None:
    assessor = QualityAssessor(config.quality)
    assert assessor.assess(np.zeros((240, 320, 3), np.uint8)).score < 0.1


def test_quality_low_for_blown_out_frame(config: MorpheusConfig) -> None:
    assessor = QualityAssessor(config.quality)
    assert assessor.assess(np.full((240, 320, 3), 255, np.uint8)).score < 0.1


def test_scene_change_high_when_frame_replaced(config: MorpheusConfig) -> None:
    """The camera being bumped must be distinguishable from a person moving."""
    assessor = QualityAssessor(config.quality)
    assessor.assess(gray_frame(value=40))
    rng = np.random.default_rng(7)
    disrupted = assessor.assess(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    assert disrupted.scene_change > 0.5


def test_scene_change_low_for_localised_movement(config: MorpheusConfig) -> None:
    import cv2

    assessor = QualityAssessor(config.quality)
    assessor.assess(gray_frame(value=40))
    img = gray_frame(value=40)
    cv2.rectangle(img, (10, 10), (40, 40), (200, 200, 200), -1)
    assert assessor.assess(img).scene_change < 0.2


def test_realistic_scene_clears_the_quality_gate(config: MorpheusConfig) -> None:
    """Regression guard for a scale mismatch that shipped and reached a user.

    quality.py measures Laplacian variance AFTER a 3x3 Gaussian blur, which
    suppresses sensor noise but also divides the scale by roughly 20x. The blur
    was added without rescaling quality.min_focus, so the floor was left
    calibrated for the unblurred scale. Every frame from a perfectly good
    camera scored ~0.13 and was discarded before face detection ran, which
    would have produced a zero-coverage night indistinguishable from the real
    finding M0 exists to make.

    None of the existing tests caught it: they asserted that bad frames fail,
    never that a good one passes. This asserts the other direction.
    """
    import cv2

    rng = np.random.default_rng(0)
    img = np.zeros((720, 1280), np.uint8)
    img[:] = np.linspace(60, 180, 1280).astype(np.uint8)
    for _ in range(40):
        x, y = int(rng.integers(0, 1200)), int(rng.integers(0, 680))
        cv2.rectangle(img, (x, y), (x + 60, y + 40), int(rng.integers(20, 230)), -1)
    img = np.clip(
        img.astype(np.int16) + rng.integers(-6, 7, img.shape), 0, 255
    ).astype(np.uint8)

    metrics = QualityAssessor(config.quality).assess(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))

    assert metrics.focus > config.quality.min_focus, (
        f"a well-exposed textured scene scored focus {metrics.focus:.1f} against a "
        f"floor of {config.quality.min_focus}; the floor and the blur are out of sync"
    )
    assert metrics.score >= config.quality.min_score, (
        f"a well-exposed textured scene scored {metrics.score:.3f}, below the "
        f"{config.quality.min_score} gate — real footage would be silently discarded"
    )


def test_blank_frame_still_fails_the_focus_floor(config: MorpheusConfig) -> None:
    """The floor must stay strict enough to reject a featureless frame."""
    flat = np.full((720, 1280, 3), 120, np.uint8)
    metrics = QualityAssessor(config.quality).assess(flat)
    assert metrics.focus < config.quality.min_focus
    assert metrics.score < config.quality.min_score
