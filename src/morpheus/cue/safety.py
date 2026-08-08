"""Hard safety limits, enforced independently of any policy.

This module exists because of a structural concern rather than a functional
one. The adaptive layer (M5) will eventually be choosing cue volume and timing
from outcome data. A learner optimising "cue incorporation" with no constraint
would discover that louder cues are noticed more often, right up until it is
waking the user every night. The design's answer is that safety limits live
outside the learner's action space entirely (design.md §12.5, §24).

So: the policy *proposes*, the supervisor *disposes*. Every cue passes through
`authorize()`, which knows nothing about policies, learning, or experiments.
It counts, it clocks, and it says no.

If the supervisor and a policy ever disagree, the supervisor wins and the
disagreement is recorded as a defect — a policy asking for something forbidden
is a bug worth seeing, not a condition to handle quietly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger("morpheus.safety")


class Denial(str, Enum):
    """Why a cue was refused. Recorded for every denied request."""

    OK = "ok"
    HALTED = "halted"
    BEFORE_MIN_DELAY = "before_min_delay"
    NIGHTLY_CAP = "nightly_cap_reached"
    HOURLY_CAP = "hourly_cap_reached"
    COOLDOWN = "within_cooldown"
    AWAKENING = "possible_awakening_this_night"
    NOT_ARMED = "not_armed"
    PAST_WINDOW = "past_permitted_window"


@dataclass
class SafetyLimits:
    """Hard caps. Not adjustable by any adaptive policy.

    Defaults are deliberately conservative. The failure mode that matters is
    waking the user, not missing a cue: a missed cue costs one opportunity out
    of hundreds, while a night of fragmented sleep costs data quality, wellbeing,
    and adherence to a study that needs many months of it.
    """

    # Gain ceiling as a fraction of full scale. The player clamps to this; it is
    # repeated here so a request can be *refused* rather than silently quietened.
    max_gain: float = 0.35
    min_gain: float = 0.02

    # No cueing before this much sleep has elapsed. The published protocol uses
    # roughly six hours, which also keeps cues away from the deep-sleep-dominant
    # early night.
    min_delay_s: float = 5.5 * 3600
    # Stop cueing this long before the expected wake time, so a cue cannot be
    # confused with, or collide with, the alarm.
    stop_before_wake_s: float = 45 * 60

    max_cues_per_night: int = 6
    max_cues_per_hour: int = 2
    min_cooldown_s: float = 20 * 60

    # Multiplier applied to the cooldown after a probable arousal.
    arousal_cooldown_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 < self.max_gain <= 1.0:
            raise ValueError("max_gain must be in (0, 1]")
        if not 0.0 <= self.min_gain <= self.max_gain:
            raise ValueError("min_gain must be within [0, max_gain]")
        if self.max_cues_per_night < 0 or self.max_cues_per_hour < 0:
            raise ValueError("caps must be non-negative")
        if self.min_cooldown_s < 0 or self.min_delay_s < 0:
            raise ValueError("durations must be non-negative")


@dataclass
class Authorization:
    """The supervisor's verdict on a single cue request."""

    allowed: bool
    reason: Denial
    granted_gain: float = 0.0
    clamped: bool = False

    @property
    def denied(self) -> bool:
        return not self.allowed


@dataclass
class SafetySupervisor:
    """Tracks the night and enforces the caps.

    Deliberately holds no reference to the controller, the policy, or the
    experiment. It cannot be talked into anything, because there is nothing to
    talk to.
    """

    limits: SafetyLimits = field(default_factory=SafetyLimits)

    _armed: bool = False
    _halted: bool = False
    _halt_reason: str = ""
    _sleep_onset_mono: Optional[float] = None
    _expected_wake_mono: Optional[float] = None
    _cue_times: list[float] = field(default_factory=list)
    _cooldown_until: Optional[float] = None
    _awakening_seen: bool = False
    _violations: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- lifecycle

    def begin_night(self) -> None:
        """Start a new night. Clears everything that is budgeted per night.

        Separated from `arm` because a night can legitimately be armed more than
        once — a WBTB wake re-arms with a new sleep onset — and the whole-night
        safety budget must not restart when it does.
        """
        self._cue_times.clear()
        self._halted = False
        self._halt_reason = ""
        self._awakening_seen = False
        self._violations.clear()

    def arm(
        self,
        sleep_onset_mono: float,
        expected_wake_mono: Optional[float] = None,
        *,
        new_night: bool = True,
    ) -> None:
        """Arm for cueing from a given sleep onset.

        `new_night=False` re-arms within the night, which is what a WBTB wake
        needs: the minimum-delay clock restarts from the moment you go back to
        sleep, but the nightly cue cap, the rolling-hour cap, and the terminal
        safety stops all carry over. Resetting those on re-arm would let a
        six-cue budget deliver twelve, and would let a re-arm silently revive a
        night that had halted on a possible awakening.
        """
        if new_night:
            self.begin_night()
        self._armed = True
        self._sleep_onset_mono = sleep_onset_mono
        self._expected_wake_mono = expected_wake_mono
        # Cooldown is genuinely arm-scoped: after a WBTB wake you have been up
        # for the best part of an hour, so a cooldown left over from before it
        # is measuring nothing.
        self._cooldown_until = None

    def halt(self, reason: str) -> None:
        """End cueing for the night. Terminal: nothing re-enters from here."""
        if not self._halted:
            log.warning("safety halt: %s", reason)
        self._halted = True
        self._halt_reason = reason

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def cues_tonight(self) -> int:
        return len(self._cue_times)

    @property
    def violations(self) -> list[str]:
        """Policy requests that breached a cap. Non-empty means a bug."""
        return list(self._violations)

    # ----------------------------------------------------------- the verdict

    def authorize(self, now_mono: float, requested_gain: float) -> Authorization:
        """Decide whether a cue may be played now, and at what gain.

        Order matters only for the reported reason, not the outcome: any single
        failing condition denies the request.
        """
        if self._halted:
            return Authorization(False, Denial.HALTED)
        if not self._armed or self._sleep_onset_mono is None:
            return Authorization(False, Denial.NOT_ARMED)
        if self._awakening_seen:
            return Authorization(False, Denial.AWAKENING)

        if now_mono - self._sleep_onset_mono < self.limits.min_delay_s:
            return Authorization(False, Denial.BEFORE_MIN_DELAY)

        if self._expected_wake_mono is not None:
            if now_mono > self._expected_wake_mono - self.limits.stop_before_wake_s:
                return Authorization(False, Denial.PAST_WINDOW)

        if len(self._cue_times) >= self.limits.max_cues_per_night:
            return Authorization(False, Denial.NIGHTLY_CAP)

        recent = [t for t in self._cue_times if now_mono - t < 3600.0]
        if len(recent) >= self.limits.max_cues_per_hour:
            return Authorization(False, Denial.HOURLY_CAP)

        if self._cooldown_until is not None and now_mono < self._cooldown_until:
            return Authorization(False, Denial.COOLDOWN)

        granted = min(max(float(requested_gain), self.limits.min_gain), self.limits.max_gain)
        clamped = abs(granted - float(requested_gain)) > 1e-9
        if float(requested_gain) > self.limits.max_gain:
            # A policy should never ask for this. Record it as a defect rather
            # than quietly clamping, so it surfaces in review.
            self._violations.append(
                f"policy requested gain {requested_gain:.3f} above ceiling "
                f"{self.limits.max_gain:.3f} at t={now_mono:.0f}"
            )
            log.error(
                "policy requested gain %.3f above ceiling %.3f — clamped, recorded as defect",
                requested_gain, self.limits.max_gain,
            )

        return Authorization(True, Denial.OK, granted_gain=granted, clamped=clamped)

    # -------------------------------------------------------------- feedback

    def record_cue(self, now_mono: float) -> None:
        """Called immediately after a cue is played. Starts the cooldown."""
        self._cue_times.append(now_mono)
        self._cooldown_until = now_mono + self.limits.min_cooldown_s

    def record_arousal(self, now_mono: float) -> None:
        """Probable arousal: extend the cooldown, but keep cueing available."""
        base = self._cooldown_until or now_mono
        extended = now_mono + self.limits.min_cooldown_s * self.limits.arousal_cooldown_multiplier
        self._cooldown_until = max(base, extended)

    def record_awakening(self, now_mono: float) -> None:
        """Possible awakening: no further cues tonight. Non-negotiable."""
        self._awakening_seen = True
        self.halt("possible awakening detected after a cue")

    def min_delay_remaining(self, now_mono: float) -> Optional[float]:
        """Seconds left on the minimum-delay clock alone.

        Deliberately separate from `time_until_eligible`, which folds in
        cooldown and the hourly cap. The controller needs to distinguish them:
        conflating the two made the state machine report SETTLING (still waiting
        to be allowed to start) when it was actually in COOLDOWN (started, now
        resting), which is the kind of mislabelling that makes an overnight log
        actively misleading.
        """
        if self._sleep_onset_mono is None:
            return None
        return max(0.0, self._sleep_onset_mono + self.limits.min_delay_s - now_mono)

    def time_until_eligible(self, now_mono: float) -> Optional[float]:
        """Seconds until a cue could next be permitted, or None if never tonight."""
        if self._halted or self._awakening_seen or not self._armed:
            return None
        if self._sleep_onset_mono is None:
            return None
        if len(self._cue_times) >= self.limits.max_cues_per_night:
            return None

        waits = [self._sleep_onset_mono + self.limits.min_delay_s - now_mono]
        if self._cooldown_until is not None:
            waits.append(self._cooldown_until - now_mono)
        recent = sorted(t for t in self._cue_times if now_mono - t < 3600.0)
        if len(recent) >= self.limits.max_cues_per_hour:
            waits.append(recent[0] + 3600.0 - now_mono)
        return max(0.0, max(waits))
