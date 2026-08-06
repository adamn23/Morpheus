"""M1: eye ROI registration, flow, and bilateral correlation.

The real detector cannot be exercised on synthetic imagery, so these tests
attack the part that can be: the geometry. A synthetic "eye" is translated by a
known number of pixels, with and without a simultaneous head translation, and
the extractor is asked to tell them apart.

That is the crux of M1. An unregistered ROI reports head motion as eye motion,
which is what makes naive frame-differencing approaches to this problem
worthless. If registration works, the signal is at least measuring the right
thing; whether it is strong enough to matter is the separate question H1 asks.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from morpheus.vision.eye_flow import EyeFlowConfig, EyeFlowExtractor, _conjugate_correlation
from morpheus.vision.landmarks import LandmarkSet, YuNetLandmarks

from .conftest import visible_face


def scene(
    *,
    eye_dx: float = 0.0,
    head_dx: float = 0.0,
    size: tuple[int, int] = (480, 640),
    interocular: float = 120.0,
    seed: int = 0,
) -> np.ndarray:
    """A face-like frame with independently movable eyes and head.

    `head_dx` shifts everything; `eye_dx` shifts only the iris within each
    socket. Registration must remove the first and preserve the second.
    """
    height, width = size
    rng = np.random.default_rng(seed)
    img = np.full((height, width), 60, np.uint8)
    # Skin texture. Drawn before the head shift and rolled with it, because a
    # real head carries its texture along. Leaving the noise static in image
    # coordinates would make a pure head translation change the pixels around
    # the eye, which registration cannot remove and which no real head does.
    img = np.clip(img.astype(np.int16) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)

    cx, cy = width / 2, height / 2
    for sign in (-1, 1):
        socket_x = cx + sign * interocular / 2
        cv2.ellipse(img, (int(socket_x), int(cy)), (34, 20), 0, 0, 360, 150, -1)
        cv2.ellipse(img, (int(socket_x), int(cy)), (34, 20), 0, 0, 360, 90, 2)
        # The iris: the part that moves when the eye moves.
        cv2.circle(img, (int(socket_x + eye_dx), int(cy)), 11, 40, -1)
        cv2.circle(img, (int(socket_x + eye_dx), int(cy)), 4, 15, -1)

    if head_dx:
        img = np.roll(img, int(head_dx), axis=1)
    return img


def landmarks_for(head_dx: float = 0.0, *, interocular: float = 120.0,
                  size: tuple[int, int] = (480, 640)) -> LandmarkSet:
    height, width = size
    cx, cy = width / 2 + head_dx, height / 2
    right = (cx - interocular / 2, cy)
    left = (cx + interocular / 2, cy)
    return LandmarkSet(
        points=np.array([right, left], dtype=np.float32),
        dense=False,
        source="test",
        right_eye_centre=right,
        left_eye_centre=left,
        interocular_px=interocular,
        yaw_proxy=0.0,
        roll_deg=0.0,
    )


@pytest.fixture
def extractor() -> EyeFlowExtractor:
    return EyeFlowExtractor(EyeFlowConfig(roi_scale=0.28, equalize=False))


# ------------------------------------------------------------------- basics


def test_no_landmarks_yields_nothing(extractor: EyeFlowExtractor) -> None:
    sample = extractor.update(scene(), None)
    assert not sample.usable
    assert sample.flow_left is None


def test_first_frame_has_no_flow(extractor: EyeFlowExtractor) -> None:
    """Flow needs a predecessor; the opening frame must not invent one."""
    sample = extractor.update(scene(), landmarks_for())
    assert sample.flow_left is None and sample.flow_right is None


def test_static_eyes_produce_near_zero_flow(extractor: EyeFlowExtractor) -> None:
    frame = scene()
    marks = landmarks_for()
    extractor.update(frame, marks)
    sample = extractor.update(frame, marks)
    assert sample.flow_left == pytest.approx(0.0, abs=1e-3)
    assert sample.flow_right == pytest.approx(0.0, abs=1e-3)


def test_eye_movement_is_detected(extractor: EyeFlowExtractor) -> None:
    marks = landmarks_for()
    extractor.update(scene(eye_dx=0), marks)
    sample = extractor.update(scene(eye_dx=6), marks)
    assert sample.flow_left is not None and sample.flow_left > 0.05
    assert sample.flow_right is not None and sample.flow_right > 0.05


def test_flow_scales_with_displacement(extractor: EyeFlowExtractor) -> None:
    readings = []
    for dx in (2, 5, 9):
        ext = EyeFlowExtractor(EyeFlowConfig(roi_scale=0.28, equalize=False))
        marks = landmarks_for()
        ext.update(scene(eye_dx=0), marks)
        readings.append(ext.update(scene(eye_dx=dx), marks).flow_left)
    assert readings == sorted(readings)


# --------------------------------------------------- registration, the crux


def test_head_translation_is_registered_away() -> None:
    """The failure mode that makes naive approaches worthless.

    The head moves 8 px while the eyes stay fixed in their sockets. An
    unregistered ROI would report that as a large eye movement. Registration
    must suppress it to roughly the level of a static frame.
    """
    ext = EyeFlowExtractor(EyeFlowConfig(roi_scale=0.28, equalize=False))
    ext.update(scene(head_dx=0, eye_dx=0), landmarks_for(0))
    moved = ext.update(scene(head_dx=8, eye_dx=0), landmarks_for(8))

    ref = EyeFlowExtractor(EyeFlowConfig(roi_scale=0.28, equalize=False))
    ref.update(scene(eye_dx=0), landmarks_for())
    still = ref.update(scene(eye_dx=0), landmarks_for())

    eyes = EyeFlowExtractor(EyeFlowConfig(roi_scale=0.28, equalize=False))
    eyes.update(scene(eye_dx=0), landmarks_for())
    real = eyes.update(scene(eye_dx=6), landmarks_for())

    assert moved.flow_left is not None and still.flow_left is not None
    assert real.flow_left is not None

    # The claim worth asserting is the separation, not an absolute threshold:
    # an 8 px head translation must register as dramatically less eye movement
    # than a 6 px eye movement does. Without registration the two are
    # indistinguishable, which is precisely how this problem gets faked.
    assert moved.flow_left < real.flow_left / 100, (
        f"head translation ({moved.flow_left:.4f}) is not clearly separated from "
        f"genuine eye movement ({real.flow_left:.4f}); registration is not working"
    )
    assert moved.flow_left == pytest.approx(still.flow_left, abs=1e-3), (
        "a pure head translation should look like a static frame after registration"
    )


def test_eye_movement_survives_simultaneous_head_movement() -> None:
    """Registration must remove head motion without removing the signal."""
    ext = EyeFlowExtractor(EyeFlowConfig(roi_scale=0.28, equalize=False))
    ext.update(scene(head_dx=0, eye_dx=0), landmarks_for(0))
    both = ext.update(scene(head_dx=6, eye_dx=6), landmarks_for(6))
    assert both.flow_left is not None and both.flow_left > 0.03


def test_registration_residual_is_reported(extractor: EyeFlowExtractor) -> None:
    marks = landmarks_for()
    extractor.update(scene(), marks)
    sample = extractor.update(scene(), marks)
    assert sample.residual_left is not None
    assert 0.0 <= sample.residual_left < 0.05


def test_unregisterable_frames_are_dropped_not_reported() -> None:
    """When alignment fails, flow would be head motion. Better to report nothing."""
    ext = EyeFlowExtractor(EyeFlowConfig(roi_scale=0.28, equalize=False, max_residual=0.0001))
    marks = landmarks_for()
    ext.update(scene(eye_dx=0), marks)
    sample = ext.update(scene(eye_dx=9, seed=7), marks)
    assert sample.flow_left is None
    assert sample.residual_left is not None  # the reason is still recorded


def test_roi_partly_outside_frame_is_skipped(extractor: EyeFlowExtractor) -> None:
    marks = landmarks_for(head_dx=-320)  # pushes the right eye off the edge
    extractor.update(scene(), marks)
    sample = extractor.update(scene(), marks)
    assert sample.flow_right is None


# ------------------------------------------------- bilateral correlation


def test_conjugate_movement_correlates_positively() -> None:
    """Both eyes moving the same way is the signature of a real saccade."""
    left = np.zeros((10, 10, 2), np.float32)
    right = np.zeros((10, 10, 2), np.float32)
    left[..., 0] = 1.0
    right[..., 0] = 1.0
    assert _conjugate_correlation(left, right) == pytest.approx(1.0, abs=1e-6)


def test_opposing_movement_correlates_negatively() -> None:
    left = np.zeros((10, 10, 2), np.float32)
    right = np.zeros((10, 10, 2), np.float32)
    left[..., 0] = 1.0
    right[..., 0] = -1.0
    assert _conjugate_correlation(left, right) < -0.9


def test_one_eye_moving_alone_scores_low() -> None:
    """A strong movement in one eye and noise in the other is not conjugate.

    This is the case that separates a real eye movement from a local shadow or
    a stray reflection, and the magnitude weighting is what catches it.
    """
    left = np.zeros((10, 10, 2), np.float32)
    right = np.zeros((10, 10, 2), np.float32)
    left[..., 0] = 1.0
    right[..., 0] = 0.01
    assert abs(_conjugate_correlation(left, right)) < 0.05


def test_static_eyes_correlate_at_zero() -> None:
    zeros = np.zeros((10, 10, 2), np.float32)
    assert _conjugate_correlation(zeros, zeros) == 0.0


def test_bilateral_correlation_is_computed_on_real_rois(extractor: EyeFlowExtractor) -> None:
    marks = landmarks_for()
    extractor.update(scene(eye_dx=0), marks)
    sample = extractor.update(scene(eye_dx=6), marks)
    assert sample.bilateral_corr is not None
    # Both irises move the same direction in the image, so this should be
    # strongly positive.
    assert sample.bilateral_corr > 0.5


# ------------------------------------------------------ pipeline integration


def test_pipeline_populates_eye_columns_when_enabled(config) -> None:
    """The M1 columns must stop being NULL once the pipeline is wired."""
    from morpheus.types import Frame
    from morpheus.vision.pipeline import Aggregator, VisionPipeline

    from .conftest import ScriptedDetector

    config.eye.enabled = True
    config.eye.landmark_hz = 30.0
    # Force the five-point path: MediaPipe will not fire on synthetic imagery,
    # and this test is about the plumbing, not the mesh.
    config.eye.landmark_model = config.storage.data_dir / "absent.task"

    pipeline = VisionPipeline(
        config, presence_detector=ScriptedDetector([visible_face(centre=(320.0, 240.0))])
    )
    agg = Aggregator()

    emitted = []
    for i in range(40):
        image = cv2.cvtColor(scene(eye_dx=(i % 4) * 3), cv2.COLOR_GRAY2BGR)
        out = agg.add(pipeline.process(Frame(seq=i, t_mono=i / 30.0, t_utc=1.0 + i, image=image)))
        if out:
            emitted.append(out)

    assert emitted, "expected a completed second"
    frame = emitted[0]
    assert frame.eye_flow_l is not None, "eye flow still unpopulated"
    assert frame.eye_flow_bilateral_corr is not None
    # Five-point landmarks cannot give eyelid contours, so these stay None
    # rather than being approximated from points that do not exist.
    assert frame.lid_disp_l is None
    assert frame.landmark_available == 0.0  # present but not dense


def test_disabled_eye_tracking_leaves_columns_null(config) -> None:
    from morpheus.types import Frame
    from morpheus.vision.pipeline import Aggregator, VisionPipeline

    from .conftest import ScriptedDetector

    config.eye.enabled = False
    pipeline = VisionPipeline(config, presence_detector=ScriptedDetector([visible_face()]))
    agg = Aggregator()
    emitted = []
    for i in range(40):
        image = cv2.cvtColor(scene(), cv2.COLOR_GRAY2BGR)
        out = agg.add(pipeline.process(Frame(seq=i, t_mono=i / 30.0, t_utc=1.0 + i, image=image)))
        if out:
            emitted.append(out)
    assert emitted[0].eye_flow_l is None
    assert emitted[0].eye_flow_bilateral_corr is None
