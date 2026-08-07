"""Frozen lid-geometry confirmation criteria.

These tests exist to prove the criteria behave correctly BEFORE the
confirmation data is recorded. Once that recording exists, neither the criteria
nor these tests may change — that is the whole point of freezing them.

The direction that matters most is rejection: the 0.909 that prompted this was
post-hoc, and the most likely explanation is still that it was an artefact. The
criteria must be able to say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from morpheus.calibration.confirmation import (
    CI_LOWER_BOUND_REQUIRED,
    MIN_HEAD_TURN_EXCLUSION,
    MOTION_GATE_PERCENTILE,
    PRIMARY_CHANNEL,
    confirm,
    format_result,
)

FPS = 30.0
QUIET = 0.0005      # measured body motion while holding still
MOVING = 0.0013     # measured during deliberate head turns


def seg(n_windows, *, level, spread=0.2, motion=QUIET, seed=0, t0=0.0):
    """A segment of `n_windows` seconds at a given lid-displacement level."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(int(n_windows * FPS)):
        rows.append({
            "t_mono": t0 + i / FPS,
            PRIMARY_CHANNEL: float(rng.normal(level, spread)),
            "motion": float(rng.normal(motion, motion * 0.1)),
        })
    return rows


def dataset(*, saccade_level, micro_level=1.0, micro_motion=QUIET, seed=0):
    return {
        "eyes_closed_still": seg(40, level=1.0, seed=seed),
        "slow_saccades": seg(40, level=saccade_level, seed=seed + 1),
        "fast_saccades": seg(40, level=saccade_level, seed=seed + 2),
        "head_turn": seg(25, level=6.0, motion=MOVING, seed=seed + 3),
        "micro_head_motion": seg(40, level=micro_level, motion=micro_motion, seed=seed + 4),
    }


# ------------------------------------------------------------ frozen values


def test_criteria_are_the_frozen_ones() -> None:
    """Pinned so an edit to the module is a visible test failure."""
    assert PRIMARY_CHANNEL == "lid_disp"
    assert CI_LOWER_BOUND_REQUIRED == 0.80
    assert MOTION_GATE_PERCENTILE == 90.0
    assert MIN_HEAD_TURN_EXCLUSION == 0.80


def test_bar_is_stricter_than_the_original_gate() -> None:
    """A post-hoc finding must clear a higher bar, not the same one.

    The original gate was a point estimate >= 0.80; this requires the CI lower
    bound >= 0.80, which is strictly harder.
    """
    from morpheus.calibration.profile import POSITIVE_CONTROL_AUC_PASS

    # Calibrated empirically: level 1.10 gives AUC ~0.835 with CI ~[0.75, 0.91].
    # Passes the old point-estimate gate, fails the confirmation.
    result = confirm(dataset(saccade_level=1.10))
    assert result.auc is not None and result.auc >= POSITIVE_CONTROL_AUC_PASS
    assert result.ci_low < CI_LOWER_BOUND_REQUIRED
    assert result.verdict == "NOT REPLICATED", (
        "a point estimate above 0.80 with a CI straddling it must not replicate"
    )


# --------------------------------------------------------------- replication


def test_strong_signal_replicates() -> None:
    result = confirm(dataset(saccade_level=1.20))   # AUC ~0.97
    assert result.validity_ok is True
    assert result.ci_low >= CI_LOWER_BOUND_REQUIRED
    assert result.verdict == "REPLICATED"


def test_absent_signal_does_not_replicate() -> None:
    """The most likely true outcome, and it must be reportable."""
    result = confirm(dataset(saccade_level=1.0))
    assert result.validity_ok is True
    assert result.auc == pytest.approx(0.5, abs=0.15)
    assert result.verdict == "NOT REPLICATED"


def test_marginal_signal_does_not_replicate() -> None:
    result = confirm(dataset(saccade_level=1.05))   # AUC ~0.71
    assert result.verdict == "NOT REPLICATED"


# ------------------------------------------------------------------ validity


def test_l1_gate_removes_large_head_movement() -> None:
    result = confirm(dataset(saccade_level=3.0))
    assert result.l1_head_turn_excluded >= MIN_HEAD_TURN_EXCLUSION
    assert result.l1_ok is True


def test_l1_fails_when_the_gate_lets_head_turns_through() -> None:
    """If large head movement is quiet enough to survive, the gate is useless."""
    data = dataset(saccade_level=3.0)
    data["head_turn"] = seg(25, level=6.0, motion=QUIET, seed=9)  # turns, but quiet
    result = confirm(data)
    assert result.l1_ok is False
    assert result.verdict == "NO VERDICT (instrument invalid)"


def test_l2_catches_small_head_movement_masquerading_as_eye_movement() -> None:
    """The confound the original protocol never tested.

    Large head turns are removed by the motion gate. Small, sleep-like ones are
    not — and if they produce the same lid displacement as a saccade, the
    channel is measuring the head inside the quiet condition too.
    """
    data = dataset(saccade_level=3.0, micro_level=3.5, micro_motion=QUIET)
    result = confirm(data)
    assert result.l2_micro_auc is not None
    assert result.l2_ok is False
    assert result.verdict == "NO VERDICT (instrument invalid)"
    assert any("tracking head motion" in n for n in result.notes)


def test_l2_passes_when_small_head_movement_is_inert() -> None:
    result = confirm(dataset(saccade_level=3.0, micro_level=1.0))
    assert result.l2_ok is True


def test_l2_satisfied_via_l1_when_micro_windows_are_gated_out() -> None:
    data = dataset(saccade_level=3.0)
    data["micro_head_motion"] = seg(40, level=5.0, motion=MOVING, seed=7)
    result = confirm(data)
    assert result.l2_micro_windows < 10
    assert result.l2_ok is True
    assert any("could not be measured directly" in n for n in result.notes)


# -------------------------------------------------------------- guard rails


def test_insufficient_data_gives_no_verdict() -> None:
    result = confirm({"eyes_closed_still": seg(5, level=1.0)})
    assert result.verdict == "INSUFFICIENT DATA"
    assert not result.replicated


def test_motion_gate_comes_from_the_baseline_only() -> None:
    """The threshold must not shift when the positive segments change.

    Deriving it from the control condition is what stops it being tuned, even
    unconsciously, by the contrast it is used to evaluate.
    """
    a = confirm(dataset(saccade_level=1.0, seed=3))
    b = confirm(dataset(saccade_level=6.0, seed=3))
    assert a.motion_threshold == pytest.approx(b.motion_threshold)


def test_report_states_the_outcome_plainly() -> None:
    text = format_result(confirm(dataset(saccade_level=1.0)))
    assert "NOT REPLICATED" in text
    assert "Do not adjust and retry" in text
    text = format_result(confirm(dataset(saccade_level=3.0)))
    assert "REPLICATED" in text
    assert "G9 stays locked" in text
