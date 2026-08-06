"""Safety invariants for the cue path.

These are the highest-stakes tests in the project. Everything else risks losing
data; this risks waking a sleeping person repeatedly, at volume, at 04:00.

The approach is adversarial rather than illustrative. Instead of checking that a
well-behaved policy produces well-behaved cues, the property tests drive the
supervisor with random and deliberately malicious inputs and assert the caps
hold regardless. A safety limit that only holds for callers that were trying to
respect it is not a safety limit.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from morpheus.audio.player import MIN_RAMP_MS, BufferSink, CuePlayer
from morpheus.audio.assets import generate_preset
from morpheus.cue.safety import Denial, SafetyLimits, SafetySupervisor

HOUR = 3600.0


def supervisor(**kwargs) -> SafetySupervisor:
    sup = SafetySupervisor(limits=SafetyLimits(**kwargs))
    sup.arm(sleep_onset_mono=0.0)
    return sup


def eligible_time(sup: SafetySupervisor) -> float:
    return sup.limits.min_delay_s + 1.0


# ------------------------------------------------------------- amplitude caps


def test_gain_above_ceiling_is_clamped_not_played() -> None:
    player = CuePlayer(BufferSink(), ceiling=0.3)
    rendered = player.render(generate_preset("trained-ascending"), gain=1.0, ramp_ms=1000)
    assert rendered.gain == pytest.approx(0.3)
    assert float(np.max(np.abs(rendered.samples))) <= 0.3 + 1e-6


@given(
    requested=st.floats(0.0, 50.0, allow_nan=False, allow_infinity=False),
    ceiling=st.floats(0.01, 1.0, allow_nan=False),
)
@settings(max_examples=150, deadline=None)
def test_output_never_exceeds_ceiling_for_any_request(requested: float, ceiling: float) -> None:
    """No gain request, however absurd, can produce audio above the ceiling."""
    player = CuePlayer(BufferSink(), ceiling=ceiling, samplerate=8000)
    waveform = np.sin(np.linspace(0, 40 * np.pi, 4000, dtype=np.float32))
    rendered = player.render(waveform, gain=requested, ramp_ms=50)
    assert float(np.max(np.abs(rendered.samples))) <= ceiling + 1e-6


def test_ceiling_must_be_positive_and_bounded() -> None:
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            CuePlayer(BufferSink(), ceiling=bad)


# -------------------------------------------------------------------- ramping


def test_every_cue_starts_from_silence() -> None:
    """Abrupt onset is the main way a cue wakes someone instead of being heard."""
    player = CuePlayer(BufferSink(), ceiling=0.3, samplerate=8000)
    waveform = np.ones(16000, dtype=np.float32)
    rendered = player.render(waveform, gain=0.3, ramp_ms=500)
    assert abs(float(rendered.samples[0])) < 1e-3


@given(ramp_ms=st.floats(0.0, 10_000.0, allow_nan=False))
@settings(max_examples=60, deadline=None)
def test_ramp_is_enforced_even_when_zero_is_requested(ramp_ms: float) -> None:
    player = CuePlayer(BufferSink(), ceiling=0.3, samplerate=8000)
    waveform = np.ones(24000, dtype=np.float32)
    rendered = player.render(waveform, gain=0.3, ramp_ms=ramp_ms, duration_ms=3000)

    assert rendered.ramp_ms >= MIN_RAMP_MS
    assert abs(float(rendered.samples[0])) < 1e-3
    # The opening must rise, not jump.
    head = rendered.samples[: max(2, int(8000 * MIN_RAMP_MS / 1000.0 / 4))]
    assert float(np.max(np.abs(head))) < 0.3


def test_ramp_survives_a_cue_shorter_than_the_ramp() -> None:
    """A short cue gets a proportionally shorter fade, never no fade."""
    player = CuePlayer(BufferSink(), ceiling=0.3, samplerate=8000)
    rendered = player.render(np.ones(400, dtype=np.float32), gain=0.3, ramp_ms=5000, duration_ms=50)
    assert abs(float(rendered.samples[0])) < 0.3
    assert rendered.samples.size == 400


# ----------------------------------------------------------------- hard caps


def test_no_cue_before_the_minimum_delay() -> None:
    sup = supervisor(min_delay_s=6 * HOUR)
    assert sup.authorize(5.9 * HOUR, 0.1).reason is Denial.BEFORE_MIN_DELAY
    assert sup.authorize(6.1 * HOUR, 0.1).allowed


def test_nightly_cap_is_absolute() -> None:
    sup = supervisor(min_delay_s=0.0, max_cues_per_night=3, max_cues_per_hour=99, min_cooldown_s=0.0)
    for i in range(3):
        auth = sup.authorize(float(i), 0.1)
        assert auth.allowed
        sup.record_cue(float(i))
    assert sup.authorize(100.0, 0.1).reason is Denial.NIGHTLY_CAP


def test_hourly_cap_limits_bursts() -> None:
    sup = supervisor(min_delay_s=0.0, max_cues_per_night=99, max_cues_per_hour=2, min_cooldown_s=0.0)
    sup.record_cue(0.0)
    sup.record_cue(10.0)
    assert sup.authorize(20.0, 0.1).reason is Denial.HOURLY_CAP
    # ...and lifts once the first cue ages out of the window.
    assert sup.authorize(3601.0, 0.1).allowed


def test_cooldown_blocks_immediate_repeat() -> None:
    sup = supervisor(min_delay_s=0.0, min_cooldown_s=1200.0)
    sup.record_cue(0.0)
    assert sup.authorize(600.0, 0.1).reason is Denial.COOLDOWN
    assert sup.authorize(1300.0, 0.1).allowed


def test_arousal_extends_the_cooldown() -> None:
    sup = supervisor(min_delay_s=0.0, min_cooldown_s=600.0, arousal_cooldown_multiplier=2.0)
    sup.record_cue(0.0)
    sup.record_arousal(10.0)
    assert sup.authorize(700.0, 0.1).reason is Denial.COOLDOWN
    assert sup.authorize(1300.0, 0.1).allowed


def test_awakening_ends_the_night_permanently() -> None:
    """The single most important safety behaviour in Morpheus."""
    sup = supervisor(min_delay_s=0.0)
    sup.record_cue(0.0)
    sup.record_awakening(60.0)

    assert sup.halted
    for t in (100.0, 3600.0, 7200.0, 30000.0):
        assert sup.authorize(t, 0.01).denied
    assert sup.time_until_eligible(10_000.0) is None


def test_halt_is_absorbing() -> None:
    sup = supervisor(min_delay_s=0.0)
    sup.halt("test")
    sup.record_arousal(10.0)  # must not resurrect the night
    assert sup.authorize(10_000.0, 0.05).reason is Denial.HALTED


def test_unarmed_supervisor_refuses_everything() -> None:
    sup = SafetySupervisor()
    assert sup.authorize(10 * HOUR, 0.1).reason is Denial.NOT_ARMED


def test_cues_stop_before_expected_wake() -> None:
    sup = SafetySupervisor(limits=SafetyLimits(min_delay_s=0.0, stop_before_wake_s=1800.0))
    sup.arm(sleep_onset_mono=0.0, expected_wake_mono=8 * HOUR)
    assert sup.authorize(7.0 * HOUR, 0.1).allowed
    assert sup.authorize(7.9 * HOUR, 0.1).reason is Denial.PAST_WINDOW


def test_excessive_gain_request_is_recorded_as_a_defect() -> None:
    """A policy asking for more than the ceiling is a bug worth surfacing."""
    sup = supervisor(min_delay_s=0.0, max_gain=0.3)
    auth = sup.authorize(10.0, 0.95)
    assert auth.allowed and auth.granted_gain == pytest.approx(0.3) and auth.clamped
    assert len(sup.violations) == 1
    assert "above ceiling" in sup.violations[0]


# ------------------------------------------------- adversarial property tests


@st.composite
def _hostile_actions(draw):
    """Arbitrary hostile actions, ordered by timestamp.

    The sort matters. Without it Hypothesis happily emits an awakening at t=0
    *after* a cue at t=7200, which no real clock can produce — monotonic time
    only moves forward. Testing against impossible histories produces failures
    that teach nothing.
    """
    actions = draw(
        st.lists(
            st.tuples(
                st.sampled_from(["authorize", "cue", "arousal", "awakening"]),
                st.floats(0.0, 12 * HOUR, allow_nan=False),
                st.floats(-10.0, 100.0, allow_nan=False),
            ),
            min_size=1,
            max_size=120,
        )
    )
    return sorted(actions, key=lambda a: a[1])


@given(actions=_hostile_actions())
@settings(max_examples=200, deadline=None)
def test_caps_hold_under_a_malicious_policy(actions) -> None:
    """Drive the supervisor with arbitrary, hostile input; the caps must hold.

    This simulates the worst plausible future: an adaptive policy that has
    learned to demand maximum volume continuously, at random times, ignoring
    every denial. Nothing it does may breach a cap.
    """
    limits = SafetyLimits(
        min_delay_s=2 * HOUR, max_cues_per_night=4, max_cues_per_hour=2,
        min_cooldown_s=900.0, max_gain=0.3,
    )
    sup = SafetySupervisor(limits=limits)
    sup.arm(sleep_onset_mono=0.0)

    granted: list[tuple[float, float]] = []
    awakening_step: int | None = None
    granted_steps: list[int] = []

    for step, (action, t, gain) in enumerate(actions):
        if action in ("authorize", "cue"):
            auth = sup.authorize(t, gain)
            if auth.allowed:
                granted.append((t, auth.granted_gain))
                granted_steps.append(step)
                sup.record_cue(t)
        elif action == "arousal":
            sup.record_arousal(t)
        elif action == "awakening":
            sup.record_awakening(t)
            if awakening_step is None:
                awakening_step = step

    assert len(granted) <= limits.max_cues_per_night
    for _, g in granted:
        assert limits.min_gain - 1e-9 <= g <= limits.max_gain + 1e-9
    for t, _ in granted:
        assert t >= limits.min_delay_s

    # No two granted cues closer together than the cooldown.
    for (t1, _), (t2, _) in zip(sorted(granted), sorted(granted)[1:]):
        assert t2 - t1 >= limits.min_cooldown_s - 1e-6

    # Nothing after an awakening. Checked in sequence order rather than by
    # timestamp: what matters is that no cue was granted once the supervisor had
    # been told about the awakening.
    if awakening_step is not None:
        assert all(s < awakening_step for s in granted_steps), (
            "a cue was granted after an awakening was recorded"
        )


@given(
    max_night=st.integers(0, 10),
    max_hour=st.integers(0, 5),
    cooldown=st.floats(0.0, 3600.0, allow_nan=False),
)
@settings(max_examples=100, deadline=None)
def test_nightly_cap_never_exceeded_for_any_limits(max_night, max_hour, cooldown) -> None:
    limits = SafetyLimits(
        min_delay_s=0.0, max_cues_per_night=max_night,
        max_cues_per_hour=max_hour, min_cooldown_s=cooldown,
    )
    sup = SafetySupervisor(limits=limits)
    sup.arm(sleep_onset_mono=0.0)

    played = 0
    for step in range(400):
        t = step * 120.0
        if sup.authorize(t, 0.1).allowed:
            sup.record_cue(t)
            played += 1
    assert played <= max_night
