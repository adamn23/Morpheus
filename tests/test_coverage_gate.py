"""Tests for the M0 decision gate: coverage classification and the yaw proxy.

These are the highest-stakes pure functions in M0. The number they produce
decides whether the eye-movement branch is developed or abandoned, so the logic
is tested exhaustively even though the code is short.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from morpheus.config import CoverageConfig
from morpheus.types import CoverageFlag, PresenceObservation, QualityMetrics
from morpheus.vision.presence import _yaw_proxy, classify_coverage

from .conftest import visible_face

FRAME = (720, 1280)  # height, width


def quality(score: float = 0.9) -> QualityMetrics:
    return QualityMetrics(
        luminance_mean=40.0,
        luminance_std=10.0,
        saturated_fraction=0.0,
        underexposed_fraction=0.1,
        focus=50.0,
        scene_change=0.0,
        score=score,
    )


def test_clear_frontal_face_is_usable() -> None:
    flag = classify_coverage(visible_face(), quality(), FRAME, CoverageConfig())
    assert flag is CoverageFlag.USABLE


def test_missing_detector_reported_distinctly() -> None:
    """Motion-only nights must not masquerade as "face absent" nights."""
    obs = PresenceObservation(face_present=False, detector_available=False)
    assert classify_coverage(obs, quality(), FRAME, CoverageConfig()) is CoverageFlag.NO_DETECTOR


def test_quality_floor_precedes_face_absence() -> None:
    """A black frame must not be recorded as evidence the face was gone.

    This ordering is the whole point of the reason breakdown: "could not see"
    and "was not there" lead to different remedies.
    """
    obs = PresenceObservation(face_present=False)
    flag = classify_coverage(obs, quality(score=0.05), FRAME, CoverageConfig())
    assert flag is CoverageFlag.QUALITY_TOO_LOW


def test_turned_head_is_pose_unsuitable() -> None:
    flag = classify_coverage(visible_face(yaw=0.6), quality(), FRAME, CoverageConfig())
    assert flag is CoverageFlag.POSE_UNSUITABLE


def test_insufficient_resolution_is_too_small() -> None:
    """Below the interocular floor there is no spatial basis for eyelid work."""
    flag = classify_coverage(visible_face(interocular=12.0), quality(), FRAME, CoverageConfig())
    assert flag is CoverageFlag.TOO_SMALL


def test_eye_outside_frame_detected() -> None:
    obs = visible_face(centre=(4.0, 360.0))
    assert classify_coverage(obs, quality(), FRAME, CoverageConfig()) is CoverageFlag.EYE_OUT_OF_FRAME


def test_low_confidence_treated_as_absent() -> None:
    flag = classify_coverage(visible_face(confidence=0.2), quality(), FRAME, CoverageConfig())
    assert flag is CoverageFlag.FACE_ABSENT


@pytest.mark.parametrize("yaw", [0.0, 0.1, 0.34])
def test_yaw_within_tolerance_stays_usable(yaw: float) -> None:
    assert classify_coverage(visible_face(yaw=yaw), quality(), FRAME, CoverageConfig()) is CoverageFlag.USABLE


# ------------------------------------------------------------------ yaw proxy


def test_yaw_proxy_zero_for_centred_nose() -> None:
    right, left = (600.0, 300.0), (680.0, 300.0)
    nose = (640.0, 340.0)  # centred laterally, offset vertically
    assert _yaw_proxy(right, left, nose, 80.0, 80.0, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_yaw_proxy_signs_are_opposite_for_opposite_turns() -> None:
    right, left = (600.0, 300.0), (680.0, 300.0)
    toward_left = _yaw_proxy(right, left, (670.0, 340.0), 80.0, 80.0, 0.0)
    toward_right = _yaw_proxy(right, left, (610.0, 340.0), 80.0, 80.0, 0.0)
    assert toward_left > 0 > toward_right


def test_yaw_proxy_is_roll_invariant() -> None:
    """A sleeper's head rolls constantly; the turn estimate must not care.

    Rotating the whole configuration about the eye midpoint should leave the
    proxy unchanged. If it did not, every posture statistic in the coverage
    report would be contaminated by roll.
    """
    right, left, nose = (600.0, 300.0), (680.0, 300.0), (665.0, 345.0)
    dx, dy = left[0] - right[0], left[1] - right[1]
    baseline = _yaw_proxy(right, left, nose, math.hypot(dx, dy), dx, dy)

    cx, cy = (right[0] + left[0]) / 2, (right[1] + left[1]) / 2
    for degrees in (15, 45, 90, -30):
        theta = math.radians(degrees)
        rot = lambda p: (  # noqa: E731
            cx + (p[0] - cx) * math.cos(theta) - (p[1] - cy) * math.sin(theta),
            cy + (p[0] - cx) * math.sin(theta) + (p[1] - cy) * math.cos(theta),
        )
        r2, l2, n2 = rot(right), rot(left), rot(nose)
        dx2, dy2 = l2[0] - r2[0], l2[1] - r2[1]
        rotated = _yaw_proxy(r2, l2, n2, math.hypot(dx2, dy2), dx2, dy2)
        assert rotated == pytest.approx(baseline, abs=1e-9)


def test_yaw_proxy_is_scale_invariant() -> None:
    """Camera distance is fixed but unknown; the proxy must be dimensionless."""
    right, left, nose = (600.0, 300.0), (680.0, 300.0), (665.0, 345.0)
    baseline = _yaw_proxy(right, left, nose, 80.0, 80.0, 0.0)
    scale = 2.5
    r2 = (right[0] * scale, right[1] * scale)
    l2 = (left[0] * scale, left[1] * scale)
    n2 = (nose[0] * scale, nose[1] * scale)
    dx2 = l2[0] - r2[0]
    scaled = _yaw_proxy(r2, l2, n2, abs(dx2), dx2, 0.0)
    assert scaled == pytest.approx(baseline, abs=1e-9)


def test_yaw_proxy_degenerate_input_does_not_explode() -> None:
    """Coincident eye landmarks must not divide by zero at 3am."""
    assert _yaw_proxy((10.0, 10.0), (10.0, 10.0), (10.0, 20.0), 0.0, 0.0, 0.0) == 0.0


@given(
    x=st.floats(-2000, 2000, allow_nan=False),
    y=st.floats(-2000, 2000, allow_nan=False),
    sep=st.floats(1.0, 500.0, allow_nan=False),
    nose_dx=st.floats(-500, 500, allow_nan=False),
    nose_dy=st.floats(-500, 500, allow_nan=False),
)
def test_yaw_proxy_is_translation_invariant(x, y, sep, nose_dx, nose_dy) -> None:
    """Where the head sits in frame must not change the turn estimate."""
    right, left = (x, y), (x + sep, y)
    nose = (x + sep / 2 + nose_dx, y + nose_dy)
    base = _yaw_proxy(right, left, nose, sep, sep, 0.0)

    shift = 137.5
    r2, l2 = (x + shift, y + shift), (x + sep + shift, y + shift)
    n2 = (nose[0] + shift, nose[1] + shift)
    moved = _yaw_proxy(r2, l2, n2, sep, sep, 0.0)
    assert moved == pytest.approx(base, rel=1e-6, abs=1e-6)
