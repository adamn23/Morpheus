"""The overnight cue controller: state machine plus gate stack.

Deliberately structured as a deterministic step function. Time enters only via
`FeatureFrame.t_mono`, never from `time.monotonic()` inside this class, and
nothing here performs I/O. That makes a whole night replayable in milliseconds
and lets the property-based tests explore thousands of event orderings — which
matters, because the invariants this enforces are the ones that keep a sleeping
person from being woken repeatedly by a bug.

Layering, from least to most authority:

    Policy      proposes gain/timing, knows nothing about limits
    Gates       veto on context (motion, quality, caps, cooldown)
    Supervisor  enforces hard caps, cannot be overridden
    Controller  sequences the above and owns the state machine

A cue happens only if every layer agrees. Any layer can stop one alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..types import FeatureFrame
from .outcome import OutcomeAssessment, OutcomeThresholds, classify_post_cue
from .policy import Policy, ScheduledPolicy
from .safety import Authorization, SafetySupervisor
from .sensor_timing import (
    ActivityIndex,
    SensorTimingAuthorization,
    SensorTimingConfig,
    SensorTimingLocked,
)
from .state import CueCommand, CueState, Gate, GateResult, GateSnapshot, Outcome

log = logging.getLogger("morpheus.cue")


@dataclass
class GateConfig:
    """Thresholds for the context gates.

    `require_camera` decides what happens when the camera is unavailable. It
    defaults to False, and that is a considered choice: the published TLR
    protocol used no sensing at all, so refusing to cue without a camera would
    make Morpheus strictly worse than the evidence base it is built on. When
    the camera is absent the sensing gates record `unavailable` and pass, and
    the night is marked so analysis can separate sensed from unsensed cues.
    """

    min_signal_quality: float = 0.25
    max_body_motion: float = 0.006
    motion_window_s: float = 60.0
    arousal_lookback_s: float = 180.0
    require_camera: bool = False


@dataclass
class ControllerConfig:
    gates: GateConfig = field(default_factory=GateConfig)
    outcome: OutcomeThresholds = field(default_factory=OutcomeThresholds)
    # Ring buffer depth in seconds; must exceed the longest lookback used above.
    history_s: float = 400.0


@dataclass
class CueEvent:
    """Something the controller wants the caller to persist or act on."""

    kind: str
    t_mono: float
    payload: dict = field(default_factory=dict)


class CueController:
    """Sequences the night. Emits commands; never plays audio itself."""

    def __init__(
        self,
        supervisor: SafetySupervisor,
        *,
        policy: Optional[Policy] = None,
        config: Optional[ControllerConfig] = None,
        condition_allows_cue: bool = True,
        sensor_timing: Optional["SensorTimingConfig"] = None,
        authorization: Optional["SensorTimingAuthorization"] = None,
    ) -> None:
        self._supervisor = supervisor
        self._policy = policy or ScheduledPolicy()
        self._cfg = config or ControllerConfig()
        self._condition_allows_cue = condition_allows_cue

        # G9 is locked at construction, deliberately. This is a configuration
        # error, so it should surface before the night starts rather than
        # degrade silently at 04:00 — the opposite of the runtime rule. There is
        # no override flag: the person who would reach for one is precisely the
        # person the lock exists to protect (design.md §8).
        self._sensor_timing = None
        self._activity: Optional[ActivityIndex] = None
        if sensor_timing is not None:
            auth = authorization
            if auth is None or not auth.authorized:
                raise SensorTimingLocked(
                    (auth.reason if auth else "no authorization supplied")
                    + "\nSensor-timed cueing stays disabled until H1 passes. "
                    "Scheduled cueing is unaffected and remains the evidence-backed arm."
                )
            self._sensor_timing = sensor_timing
            self._activity = ActivityIndex(sensor_timing)
            log.info("sensor-timed cueing ENABLED — %s", auth.reason)

        self._state = CueState.IDLE
        self._history: list[FeatureFrame] = []
        self._last_arousal_mono: Optional[float] = None
        self._pending_cue_t: Optional[float] = None
        self._pending_cue_index: int = 0
        self._last_outcome: Optional[str] = None
        self._cue_count = 0
        self.events: list[CueEvent] = []
        # Seconds spent blocked by each gate. A night that produced no cues is
        # otherwise indistinguishable from a night where nothing was attempted,
        # and "why did nothing happen" is the first question anyone asks of an
        # overnight run.
        self.gate_blocks: dict[str, int] = {}

    # ---------------------------------------------------------------- state

    @property
    def state(self) -> CueState:
        return self._state

    @property
    def limits(self):
        """The supervisor's hard limits. Exposed so a WBTB re-arm can restart
        the minimum-delay clock; every other cap is the supervisor's alone."""
        return self._supervisor.limits

    @property
    def cue_count(self) -> int:
        return self._cue_count

    @property
    def last_outcome(self) -> Optional[str]:
        return self._last_outcome

    def arm(
        self,
        sleep_onset_mono: float,
        expected_wake_mono: Optional[float] = None,
        *,
        new_night: bool = True,
    ) -> None:
        """Arm for cueing. `new_night=False` re-arms mid-night after a WBTB wake,
        preserving the nightly cue budget and any terminal safety stop."""
        self._supervisor.arm(sleep_onset_mono, expected_wake_mono, new_night=new_night)
        self._state = CueState.SETTLING
        self._emit(
            "armed", sleep_onset_mono,
            {"expected_wake_mono": expected_wake_mono, "new_night": new_night},
        )

    def halt(self, reason: str, t_mono: float) -> None:
        self._supervisor.halt(reason)
        self._state = CueState.HALTED
        self._emit("halted", t_mono, {"reason": reason})

    def finish(self, t_mono: float) -> None:
        if self._state is not CueState.HALTED:
            self._state = CueState.MORNING
        self._emit("finished", t_mono, {"cues": self._cue_count})

    # ----------------------------------------------------------------- step

    def step(self, frame: FeatureFrame) -> Optional[CueCommand]:
        """Advance one second. Returns a cue to play, or None.

        The caller is responsible for actually playing a returned command and
        then calling `record_cue_played`. Splitting the decision from the action
        keeps this function pure and testable, and means a failure to play is
        recorded rather than silently losing the state transition.
        """
        self._remember(frame)

        if self._state in (CueState.IDLE, CueState.HALTED, CueState.MORNING):
            return None

        if self._detect_arousal(frame):
            self._last_arousal_mono = frame.t_mono
            self._supervisor.record_arousal(frame.t_mono)
            self._emit("probable_arousal", frame.t_mono, {"motion": frame.global_motion})

        if self._state is CueState.POST_CUE_OBSERVE:
            self._maybe_finish_observation(frame)
            return None

        gates = self._evaluate_gates(frame)
        for gate in gates.blocking:
            self.gate_blocks[gate.value] = self.gate_blocks.get(gate.value, 0) + 1
        self._state = CueState.MONITORING if gates.passed else self._resting_state(gates)

        if not gates.passed or not self._policy.should_propose(now_mono=frame.t_mono, gates=gates):
            return None

        requested = self._policy.propose_gain(
            cue_index=self._cue_count, last_outcome=self._last_outcome
        )
        auth: Authorization = self._supervisor.authorize(frame.t_mono, requested)
        if auth.denied:
            # The gates and the supervisor disagreed. Not an error — they check
            # overlapping but distinct things — but worth recording, because a
            # persistent disagreement means a gate is miscalibrated.
            self._emit(
                "cue_denied", frame.t_mono,
                {"reason": auth.reason.value, "requested_gain": requested},
            )
            return None

        self._state = CueState.CUEING
        return CueCommand(
            gain=auth.granted_gain,
            gain_requested=requested,
            ramp_ms=self._policy.propose_ramp_ms(),
            duration_ms=self._policy.propose_duration_ms(),
            repetition_index=self._cue_count,
            policy_version=self._policy.version,
            gates=gates,
            trigger="scheduled",
        )

    def record_cue_played(self, t_mono: float, *, success: bool = True) -> None:
        """Called after the caller has attempted playback."""
        if success:
            self._supervisor.record_cue(t_mono)
            self._cue_count += 1
            self._pending_cue_t = t_mono
            self._pending_cue_index = self._cue_count - 1
            self._state = CueState.POST_CUE_OBSERVE
            self._emit("cue_delivered", t_mono, {"index": self._pending_cue_index})
        else:
            # Playback failed. Do not consume a cue from the nightly budget, but
            # do back off rather than retrying immediately into a broken device.
            self._supervisor.record_arousal(t_mono)
            self._state = CueState.COOLDOWN
            self._emit("cue_failed", t_mono, {})

    # ----------------------------------------------------------------- gates

    def _evaluate_gates(self, frame: FeatureFrame) -> GateSnapshot:
        cfg = self._cfg.gates
        results: list[GateResult] = []
        sup = self._supervisor

        remaining = sup.min_delay_remaining(frame.t_mono)
        results.append(
            GateResult(
                Gate.G1_MIN_DELAY,
                passed=(remaining is not None and remaining <= 0.0),
                detail="elapsed" if remaining == 0.0
                else f"{remaining:.0f}s remaining" if remaining is not None
                else "not armed",
                value=remaining,
            )
        )

        # A zero-gain probe tells us which structural limit (if any) is blocking
        # without consuming or mutating anything.
        auth = sup.authorize(frame.t_mono, 0.0)
        results.append(
            GateResult(
                Gate.G2_PERMITTED_WINDOW,
                passed=auth.reason.value != "past_permitted_window",
                detail=auth.reason.value,
            )
        )
        results.append(
            GateResult(
                Gate.G6_COOLDOWN,
                passed=auth.reason.value != "within_cooldown",
                detail=auth.reason.value,
            )
        )
        results.append(
            GateResult(
                Gate.G7_NIGHTLY_CAP,
                passed=auth.reason.value not in ("nightly_cap_reached", "hourly_cap_reached"),
                detail=f"{sup.cues_tonight} cues so far",
                value=float(sup.cues_tonight),
            )
        )

        camera_live = frame.n_frames > 0 and frame.signal_quality > 0.0
        if camera_live:
            results.append(
                GateResult(
                    Gate.G3_SIGNAL_QUALITY,
                    passed=frame.signal_quality >= cfg.min_signal_quality,
                    detail=f"quality {frame.signal_quality:.2f}",
                    value=frame.signal_quality,
                )
            )
            motion = self._recent_motion(frame.t_mono, cfg.motion_window_s)
            results.append(
                GateResult(
                    Gate.G4_BODY_MOTION,
                    passed=motion <= cfg.max_body_motion,
                    detail=f"mean motion {motion:.5f} over {cfg.motion_window_s:.0f}s",
                    value=motion,
                )
            )
            quiet = (
                self._last_arousal_mono is None
                or (frame.t_mono - self._last_arousal_mono) > cfg.arousal_lookback_s
            )
            results.append(
                GateResult(Gate.G5_NO_RECENT_AROUSAL, passed=quiet, detail="" if quiet else "recent arousal")
            )
        else:
            passed = not cfg.require_camera
            for gate in (Gate.G3_SIGNAL_QUALITY, Gate.G4_BODY_MOTION, Gate.G5_NO_RECENT_AROUSAL):
                results.append(GateResult(gate, passed=passed, detail="camera unavailable"))

        results.append(
            GateResult(
                Gate.G8_EXPERIMENT,
                passed=self._condition_allows_cue,
                detail="condition permits" if self._condition_allows_cue else "no-cue condition",
            )
        )

        # G9 appears in the stack only when sensor timing is authorised. In
        # scheduled mode it is absent rather than auto-passing, so a gate
        # snapshot always shows honestly whether eye activity had any say.
        if self._sensor_timing is not None and self._activity is not None:
            burst = self._activity.update(
                frame.eye_flow_bilateral_corr, frame.eye_region_usable, frame.t_mono
            )
            results.append(
                GateResult(
                    Gate.G9_EYE_ACTIVITY,
                    passed=burst,
                    detail="sustained bilateral eye activity" if burst else "no qualifying burst",
                    value=frame.eye_flow_bilateral_corr,
                )
            )

        return GateSnapshot(results)

    # -------------------------------------------------------------- internals

    def _resting_state(self, gates: GateSnapshot) -> CueState:
        blocking = set(gates.blocking)
        if Gate.G1_MIN_DELAY in blocking:
            return CueState.SETTLING
        if Gate.G6_COOLDOWN in blocking:
            return CueState.COOLDOWN
        if {Gate.G3_SIGNAL_QUALITY} & blocking:
            return CueState.SUSPENDED
        return CueState.MONITORING

    def _detect_arousal(self, frame: FeatureFrame) -> bool:
        """Spontaneous arousal outside the post-cue window."""
        if self._state is CueState.POST_CUE_OBSERVE:
            return False
        baseline = self._recent_motion(frame.t_mono - 60.0, 120.0)
        level = max(baseline * self._cfg.outcome.arousal_ratio, self._cfg.outcome.arousal_absolute)
        return frame.global_motion >= level and baseline > 0.0

    def _maybe_finish_observation(self, frame: FeatureFrame) -> None:
        if self._pending_cue_t is None:
            self._state = CueState.COOLDOWN
            return
        elapsed = frame.t_mono - self._pending_cue_t
        if elapsed < self._cfg.outcome.observe_window_s:
            return

        assessment = self.assess_last_cue(frame.t_mono)
        self._last_outcome = assessment.outcome.value
        self._emit(
            "cue_outcome", frame.t_mono,
            {
                "index": self._pending_cue_index,
                "outcome": assessment.outcome.value,
                "detail": assessment.detail,
                "assessment": assessment,
            },
        )

        if assessment.outcome is Outcome.POSSIBLE_AWAKENING:
            self._supervisor.record_awakening(frame.t_mono)
            self._state = CueState.HALTED
            self._emit("halted", frame.t_mono, {"reason": "possible awakening after cue"})
        else:
            if assessment.outcome is Outcome.PROBABLE_AROUSAL:
                self._supervisor.record_arousal(frame.t_mono)
            self._state = CueState.COOLDOWN
        self._pending_cue_t = None

    def assess_last_cue(self, now_mono: float) -> OutcomeAssessment:
        cue_t = self._pending_cue_t or now_mono
        th = self._cfg.outcome
        before = [f for f in self._history if cue_t - th.baseline_window_s <= f.t_mono < cue_t]
        after = [f for f in self._history if cue_t <= f.t_mono <= cue_t + th.observe_window_s]
        return classify_post_cue(cue_t_mono=cue_t, before=before, after=after, thresholds=th)

    @property
    def sensor_timing_active(self) -> bool:
        return self._sensor_timing is not None

    def _remember(self, frame: FeatureFrame) -> None:
        self._history.append(frame)
        cutoff = frame.t_mono - self._cfg.history_s
        if self._history[0].t_mono < cutoff:
            self._history = [f for f in self._history if f.t_mono >= cutoff]

    def _recent_motion(self, until_mono: float, window_s: float) -> float:
        window = [
            f.global_motion
            for f in self._history
            if until_mono - window_s <= f.t_mono <= until_mono
        ]
        return float(sum(window) / len(window)) if window else 0.0

    def _emit(self, kind: str, t_mono: float, payload: dict) -> None:
        self.events.append(CueEvent(kind=kind, t_mono=t_mono, payload=payload))
