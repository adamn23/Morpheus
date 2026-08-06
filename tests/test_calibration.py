"""Calibration profile and the M1 positive-control gate.

The gate must fail when the signal is absent. That direction matters more than
the passing one: a calibration that always passes would let the project spend
four months building on a detector that never worked, which is the exact
failure the design puts this test first to avoid.
"""

from __future__ import annotations

import numpy as np
import pytest

from morpheus.calibration.profile import (
    POSITIVE_CONTROL_AUC_PASS,
    _auc,
    build_profile,
    format_profile,
    latest,
)
from morpheus.calibration.profile import save as save_profile
from morpheus.calibration.protocol import (
    PROTOCOL,
    SEGMENTS_BY_KEY,
    SegmentRole,
    positive_controls,
    posture_segments,
    total_seconds,
)
from morpheus.calibration.runner import CalibrationRunner


def samples(values, *, face=True, quality=0.9, n_extra_none=0):
    rows = [
        {"eye_flow": v, "bilateral": 0.6, "face_present": face,
         "interocular": 120.0, "quality": quality}
        for v in values
    ]
    rows += [
        {"eye_flow": None, "bilateral": None, "face_present": False,
         "interocular": None, "quality": quality}
        for _ in range(n_extra_none)
    ]
    return rows


def collected(*, separation: float, seed: int = 0, n: int = 120) -> dict:
    """Baseline and saccade segments differing by `separation` standard deviations."""
    rng = np.random.default_rng(seed)
    baseline = rng.normal(1.0, 0.3, n).clip(0)
    saccades = rng.normal(1.0 + separation * 0.3, 0.3, n).clip(0)
    return {
        "eyes_closed_still": samples(baseline),
        "slow_saccades": samples(saccades[: n // 2]),
        "fast_saccades": samples(saccades[n // 2:]),
    }


# ------------------------------------------------------------------ protocol


def test_protocol_has_a_baseline_and_a_positive_control() -> None:
    roles = {s.role for s in PROTOCOL}
    assert SegmentRole.BASELINE in roles
    assert SegmentRole.POSITIVE_CONTROL in roles
    assert len(positive_controls()) >= 1


def test_positive_controls_contrast_against_the_baseline() -> None:
    """An effect size needs something to be relative to."""
    for segment in positive_controls():
        assert "eyes_closed_still" in segment.contrast_with


def test_posture_segments_cover_the_ways_people_sleep() -> None:
    keys = {s.key for s in posture_segments()}
    assert {"posture_supine", "posture_left", "posture_right", "posture_prone"} <= keys


def test_protocol_fits_in_a_sitting() -> None:
    assert 300 <= total_seconds() <= 1200  # 5-20 minutes


# ----------------------------------------------------------------------- AUC


def test_auc_is_half_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 500), rng.normal(0, 1, 500)
    assert _auc(a, b) == pytest.approx(0.5, abs=0.05)


def test_auc_is_one_for_perfect_separation() -> None:
    assert _auc([10, 11, 12] * 10, [1, 2, 3] * 10) == pytest.approx(1.0)


def test_auc_handles_ties_without_inflating() -> None:
    """A constant-zero signal in both groups must score 0.5, not 1.0.

    Without average-rank tie handling this returns 1.0, which would turn a
    completely dead detector into a passing calibration.
    """
    assert _auc([0.0] * 50, [0.0] * 50) == pytest.approx(0.5)


def test_auc_needs_enough_samples() -> None:
    assert _auc([1, 2], [3, 4]) is None


# ------------------------------------------------------------------- profile


def test_strong_signal_passes_the_gate() -> None:
    profile = build_profile(collected(separation=4.0))
    assert profile.positive_control_auc > POSITIVE_CONTROL_AUC_PASS
    assert profile.verdict == "PASS"


def test_absent_signal_fails_the_gate() -> None:
    """The direction that matters. A dead detector must not pass."""
    profile = build_profile(collected(separation=0.0))
    assert profile.positive_control_auc == pytest.approx(0.5, abs=0.12)
    assert profile.verdict == "FAIL"
    assert not profile.passed


def test_marginal_signal_fails_rather_than_squeaking_through() -> None:
    profile = build_profile(collected(separation=0.6))
    assert profile.verdict == "FAIL"


def test_threshold_is_the_precommitted_one() -> None:
    """Pinned so a disappointing AUC cannot renegotiate its own gate."""
    assert POSITIVE_CONTROL_AUC_PASS == 0.80


def test_baseline_uses_robust_statistics() -> None:
    """One swallow or blink must not drag the noise floor above the signal."""
    values = [1.0] * 100 + [500.0]  # one huge outlier
    profile = build_profile({"eyes_closed_still": samples(values)})
    assert profile.baseline_median == pytest.approx(1.0)
    assert profile.baseline_mad < 1.0


def test_head_turn_leakage_is_flagged() -> None:
    """If head turns separate as well as saccades do, the index tracks the head.

    A passing positive control means nothing in that case, and the profile has
    to say so rather than reporting a clean PASS.
    """
    data = collected(separation=4.0)
    # Head turns that look exactly like the saccades.
    data["head_turn"] = data["slow_saccades"]
    profile = build_profile(data)
    assert profile.head_turn_leakage is not None
    assert any("head motion" in note for note in profile.notes)


def test_no_head_turn_leakage_when_registration_works() -> None:
    data = collected(separation=4.0)
    data["head_turn"] = data["eyes_closed_still"]
    profile = build_profile(data)
    assert not any("head motion" in note for note in profile.notes)


def test_posture_visibility_is_recorded() -> None:
    data = collected(separation=3.0)
    data["posture_left"] = samples([1.0] * 10, n_extra_none=90)
    data["posture_supine"] = samples([1.0] * 90, n_extra_none=10)
    profile = build_profile(data)
    assert profile.posture_visibility["posture_left"] == pytest.approx(0.1)
    assert profile.posture_visibility["posture_supine"] == pytest.approx(0.9)


def test_insufficient_data_gives_no_verdict() -> None:
    profile = build_profile({"eyes_closed_still": samples([1.0] * 5)})
    assert profile.positive_control_auc is None
    assert profile.verdict == "INSUFFICIENT DATA"
    assert not profile.passed


def test_thin_segments_are_flagged() -> None:
    profile = build_profile(collected(separation=3.0, n=40))
    assert any("usable samples" in note for note in profile.notes)


def test_failure_text_forbids_tuning_and_retrying() -> None:
    """The report must say the quiet part, because the temptation is real."""
    text = format_profile(build_profile(collected(separation=0.0)))
    assert "FAIL" in text
    assert "Do not tune thresholds" in text


def test_profile_round_trips_through_the_database(conn) -> None:
    profile = build_profile(collected(separation=4.0))
    save_profile(conn, profile)
    row = latest(conn)
    assert row is not None
    assert bool(row["passed"]) is True
    assert row["positive_control_auc"] == pytest.approx(profile.positive_control_auc)


# -------------------------------------------------------------------- runner


def test_runner_requires_eye_tracking(config) -> None:
    """Calibration without the eye index would measure nothing."""
    config.eye.enabled = False
    with pytest.raises(ValueError, match="eye tracking"):
        CalibrationRunner(config, source=object())  # type: ignore[arg-type]


def test_runner_records_a_segment_from_video(config, make_video) -> None:
    from morpheus.capture.replay import FileReplaySource

    from .conftest import ScriptedDetector, visible_face

    config.eye.enabled = True
    config.eye.landmark_model = config.storage.data_dir / "absent.task"

    from morpheus.vision.pipeline import VisionPipeline

    pipeline = VisionPipeline(
        config, presence_detector=ScriptedDetector([visible_face(centre=(160.0, 120.0),
                                                                 interocular=40.0)])
    )
    source = FileReplaySource(make_video(frames=120, fps=30.0))
    source.open()

    runner = CalibrationRunner(config, source, pipeline=pipeline)
    ticks = iter([0.0] + [i * 0.1 for i in range(1, 400)])
    recording = runner.record_segment(
        SEGMENTS_BY_KEY["eyes_closed_still"], clock=lambda: next(ticks)
    )
    source.close()

    assert recording.key == "eyes_closed_still"
    assert len(recording.samples) > 10
    assert all("eye_flow" in row for row in recording.samples)


# ----------------------------------------------------- desk vs bed staging


def test_stages_partition_the_protocol() -> None:
    from morpheus.calibration.protocol import PROTOCOL, STAGES

    assert set(STAGES["signal"]) | set(STAGES["posture"]) == {s.key for s in PROTOCOL}
    assert not set(STAGES["signal"]) & set(STAGES["posture"])


def test_positive_control_is_a_desk_segment() -> None:
    """The H1 test must be runnable before any bedside mount exists.

    That is what lets the decisive question be answered on a laptop, today, for
    nothing — rather than waiting on hardware.
    """
    from morpheus.calibration.protocol import SegmentSetup, positive_controls

    for segment in positive_controls():
        assert segment.setup is SegmentSetup.DESK


def test_postures_are_bed_segments() -> None:
    """Posture visibility measured from a desk describes nothing.

    A first full run reported 98-99% availability for all four sleep postures,
    which is not credible for a side sleeper and was an artefact of lying down
    in front of a laptop.
    """
    from morpheus.calibration.protocol import SegmentSetup, posture_segments

    for segment in posture_segments():
        assert segment.setup is SegmentSetup.BED


def test_each_stage_is_short_enough_to_finish() -> None:
    from morpheus.calibration.protocol import stage_seconds

    for stage in ("signal", "posture"):
        assert 60 <= stage_seconds(stage) <= 600


def test_unknown_stage_is_rejected() -> None:
    from morpheus.calibration.protocol import segments_for

    with pytest.raises(KeyError):
        segments_for("nonsense")
