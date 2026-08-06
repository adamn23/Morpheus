"""Cue controller: state machine, gate stack, and outcome handling.

The controller is a deterministic step function driven by FeatureFrame
timestamps, which is what makes a full eight-hour night replayable in
milliseconds. These tests exploit that to simulate whole nights, including ones
where the sleeper thrashes around all night or the camera dies at 03:00.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from morpheus.cue.controller import ControllerConfig, CueController, GateConfig
from morpheus.cue.outcome import OutcomeThresholds, classify_post_cue
from morpheus.cue.policy import ScheduledPolicy
from morpheus.cue.safety import SafetyLimits, SafetySupervisor
from morpheus.cue.state import CueState, Gate, Outcome
from morpheus.types import CoverageFlag, FeatureFrame

HOUR = 3600.0


def frame(t: float, *, motion: float = 0.001, quality: float = 0.9, camera: bool = True) -> FeatureFrame:
    return FeatureFrame(
        t_mono=t,
        t_utc=1_700_000_000.0 + t,
        n_frames=30 if camera else 0,
        signal_quality=quality if camera else 0.0,
        face_present=0.8 if camera else 0.0,
        eye_region_usable=0.4 if camera else 0.0,
        coverage_flag=CoverageFlag.USABLE if camera else CoverageFlag.NO_DETECTOR,
        global_motion=motion,
        bed_motion=motion,
        face_motion=motion * 0.5,
    )


def build(**limit_kwargs) -> tuple[CueController, SafetySupervisor]:
    limits = SafetyLimits(
        min_delay_s=limit_kwargs.pop("min_delay_s", 2 * HOUR),
        max_cues_per_night=limit_kwargs.pop("max_cues_per_night", 4),
        max_cues_per_hour=limit_kwargs.pop("max_cues_per_hour", 2),
        min_cooldown_s=limit_kwargs.pop("min_cooldown_s", 900.0),
        **limit_kwargs,
    )
    sup = SafetySupervisor(limits=limits)
    ctl = CueController(sup, policy=ScheduledPolicy(), config=ControllerConfig())
    ctl.arm(sleep_onset_mono=0.0)
    return ctl, sup


def run_night(ctl: CueController, *, hours: float = 8.0, step_s: float = 10.0, **frame_kwargs):
    """Drive a night, playing any cue the controller asks for."""
    cues = []
    t = 0.0
    while t < hours * HOUR:
        f = frame(t, **frame_kwargs)
        command = ctl.step(f)
        if command is not None:
            cues.append((t, command))
            ctl.record_cue_played(t, success=True)
        t += step_s
    return cues


# ------------------------------------------------------------------ base flow


def test_no_cue_before_minimum_delay() -> None:
    ctl, _ = build(min_delay_s=6 * HOUR)
    cues = run_night(ctl, hours=5.5)
    assert cues == []
    assert ctl.state is CueState.SETTLING


def test_cues_fire_after_delay_and_respect_caps() -> None:
    ctl, _ = build(min_delay_s=1 * HOUR, max_cues_per_night=3, max_cues_per_hour=9, min_cooldown_s=600.0)
    cues = run_night(ctl, hours=8.0)

    assert len(cues) == 3
    assert all(t >= 1 * HOUR for t, _ in cues)
    for (t1, _), (t2, _) in zip(cues, cues[1:]):
        assert t2 - t1 >= 600.0


def test_clock_only_mode_still_cues() -> None:
    """No camera must not mean no cueing.

    The published protocol used no sensing at all, so refusing to cue without a
    camera would make Morpheus worse than the evidence it is built on.
    """
    ctl, _ = build(min_delay_s=1 * HOUR)
    cues = run_night(ctl, hours=4.0, camera=False)
    assert len(cues) >= 1


def test_camera_required_mode_suppresses_when_blind() -> None:
    limits = SafetyLimits(min_delay_s=1 * HOUR)
    ctl = CueController(
        SafetySupervisor(limits=limits),
        config=ControllerConfig(gates=GateConfig(require_camera=True)),
    )
    ctl.arm(0.0)
    assert run_night(ctl, hours=4.0, camera=False) == []


def test_high_body_motion_blocks_cues() -> None:
    """The camera's gate-and-guard role: do not cue into a moving sleeper."""
    ctl, _ = build(min_delay_s=1 * HOUR)
    assert run_night(ctl, hours=5.0, motion=0.05) == []


def test_low_quality_suspends_rather_than_cueing_blind() -> None:
    ctl, _ = build(min_delay_s=1 * HOUR)
    cues = run_night(ctl, hours=4.0, quality=0.05)
    assert cues == []
    assert ctl.state is CueState.SUSPENDED


def test_no_cue_condition_blocks_everything() -> None:
    """Arm C of the experiment: everything runs, nothing plays."""
    limits = SafetyLimits(min_delay_s=1 * HOUR)
    ctl = CueController(SafetySupervisor(limits=limits), condition_allows_cue=False)
    ctl.arm(0.0)
    cues = run_night(ctl, hours=6.0)
    assert cues == []


# ------------------------------------------------------------------- outcomes


def test_awakening_after_cue_halts_the_night() -> None:
    ctl, sup = build(min_delay_s=0.0, max_cues_per_night=9, min_cooldown_s=60.0)

    t = 0.0
    while t < 300.0:  # quiet baseline
        ctl.step(frame(t, motion=0.0005))
        t += 5.0
    command = ctl.step(frame(t, motion=0.0005))
    assert command is not None
    ctl.record_cue_played(t, success=True)

    # Sustained violent motion through the observation window.
    for _ in range(40):
        t += 5.0
        ctl.step(frame(t, motion=0.08))

    assert ctl.state is CueState.HALTED
    assert sup.halted
    assert run_night(ctl, hours=8.0) == [], "nothing may fire after a halt"


def test_quiet_outcome_allows_the_night_to_continue() -> None:
    ctl, _ = build(min_delay_s=0.0, max_cues_per_night=9, min_cooldown_s=60.0)
    t = 0.0
    while t < 300.0:
        ctl.step(frame(t, motion=0.0005))
        t += 5.0
    ctl.step(frame(t, motion=0.0005))
    ctl.record_cue_played(t, success=True)
    for _ in range(30):
        t += 5.0
        ctl.step(frame(t, motion=0.0005))

    assert ctl.state is not CueState.HALTED
    assert ctl.last_outcome == Outcome.QUIET.value


def test_failed_playback_does_not_consume_the_cue_budget() -> None:
    ctl, sup = build(min_delay_s=0.0)
    ctl.step(frame(0.0))
    ctl.record_cue_played(0.0, success=False)
    assert sup.cues_tonight == 0
    assert ctl.cue_count == 0
    assert ctl.state is CueState.COOLDOWN


# --------------------------------------------------------- outcome classifier


def test_uncertain_when_signal_collapsed() -> None:
    """Never learn from a window we could not see."""
    before = [frame(float(i), motion=0.001) for i in range(60)]
    after = [frame(60.0 + i, motion=0.05, quality=0.05) for i in range(30)]
    result = classify_post_cue(cue_t_mono=60.0, before=before, after=after)
    assert result.outcome is Outcome.UNCERTAIN


def test_uncertain_when_too_few_frames() -> None:
    before = [frame(float(i)) for i in range(60)]
    result = classify_post_cue(cue_t_mono=60.0, before=before, after=[frame(61.0)])
    assert result.outcome is Outcome.UNCERTAIN


def test_arousal_detected_relative_to_baseline() -> None:
    before = [frame(float(i), motion=0.001) for i in range(60)]
    after = [frame(60.0 + i, motion=0.02 if i == 5 else 0.001) for i in range(30)]
    result = classify_post_cue(cue_t_mono=60.0, before=before, after=after)
    assert result.outcome is Outcome.PROBABLE_AROUSAL
    assert result.latency_to_motion_ms == pytest.approx(5000.0)


def test_brief_stir_is_arousal_not_awakening() -> None:
    """A single twitch must not end the night; sustained motion should."""
    before = [frame(float(i), motion=0.001) for i in range(60)]
    after = [frame(60.0 + i, motion=0.05 if i in (3, 4) else 0.001) for i in range(60)]
    assert classify_post_cue(cue_t_mono=60.0, before=before, after=after).outcome is Outcome.PROBABLE_AROUSAL


# ------------------------------------------------------- policy volume ladder


def test_volume_steps_down_faster_than_up() -> None:
    """Asymmetric on purpose: waking the user is worse than a missed cue."""
    policy = ScheduledPolicy(start_gain=0.10, step_up=0.02, step_down=0.05)
    assert policy.propose_gain(cue_index=0, last_outcome=None) == pytest.approx(0.10)
    assert policy.propose_gain(cue_index=1, last_outcome="quiet") == pytest.approx(0.12)
    assert policy.propose_gain(cue_index=2, last_outcome="probable_arousal") == pytest.approx(0.07)


def test_uncertain_outcome_does_not_move_the_volume() -> None:
    policy = ScheduledPolicy(start_gain=0.10)
    policy.propose_gain(cue_index=0, last_outcome=None)
    assert policy.propose_gain(cue_index=1, last_outcome="uncertain") == pytest.approx(0.10)


def test_policy_gain_stays_within_its_own_band() -> None:
    policy = ScheduledPolicy(start_gain=0.10, min_gain=0.02, max_gain=0.20, step_up=0.05)
    policy.propose_gain(cue_index=0, last_outcome=None)
    for i in range(50):
        g = policy.propose_gain(cue_index=i, last_outcome="quiet")
        assert 0.02 <= g <= 0.20


# ------------------------------------------------------------ property tests


@given(
    motion=st.floats(0.0, 0.2, allow_nan=False),
    quality=st.floats(0.0, 1.0, allow_nan=False),
    delay_h=st.floats(0.0, 6.0, allow_nan=False),
    cap=st.integers(0, 8),
)
@settings(max_examples=120, deadline=None)
def test_invariants_hold_for_any_night(motion, quality, delay_h, cap) -> None:
    """Whatever the night looks like, these must never be violated."""
    ctl, sup = build(
        min_delay_s=delay_h * HOUR, max_cues_per_night=cap,
        max_cues_per_hour=max(1, cap), min_cooldown_s=600.0,
    )
    cues = run_night(ctl, hours=9.0, step_s=15.0, motion=motion, quality=quality)

    assert len(cues) <= cap
    assert all(t >= delay_h * HOUR for t, _ in cues)
    for (t1, _), (t2, _) in zip(cues, cues[1:]):
        assert t2 - t1 >= 600.0 - 1e-6
    assert all(c.gain <= sup.limits.max_gain + 1e-9 for _, c in cues)
    assert all(c.ramp_ms > 0 for _, c in cues)
    if sup.halted:
        halt_t = min((t for t, _ in cues), default=0.0)
        assert all(t >= halt_t for t, _ in cues)


@given(cap=st.integers(0, 6))
@settings(max_examples=40, deadline=None)
def test_zero_or_more_cap_is_always_respected(cap) -> None:
    ctl, _ = build(min_delay_s=0.0, max_cues_per_night=cap, max_cues_per_hour=99, min_cooldown_s=1.0)
    assert len(run_night(ctl, hours=8.0, step_s=30.0)) <= cap


def test_gate_snapshot_records_every_gate() -> None:
    """Stored with each cue so a night can be re-examined without re-running it."""
    ctl, _ = build(min_delay_s=0.0)
    cues = run_night(ctl, hours=1.0)
    assert cues
    snapshot = cues[0][1].gates
    assert set(snapshot.to_dict()) == {g.value for g in Gate}
