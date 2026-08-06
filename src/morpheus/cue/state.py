"""Cue controller states and the decisions it can emit."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class CueState(str, enum.Enum):
    """States of the overnight cue loop (design.md §12.1).

    `HALTED` is absorbing. Nothing re-enters the loop from it, which is what
    makes "stop cueing for the night" mean what it says.
    """

    IDLE = "idle"
    ARMED = "armed"
    SETTLING = "settling"            # asleep, but inside the minimum delay
    MONITORING = "monitoring"        # eligible window, waiting for gates
    CUEING = "cueing"                # a cue is being played
    POST_CUE_OBSERVE = "post_cue_observe"
    COOLDOWN = "cooldown"
    SUSPENDED = "suspended"          # signal lost; recoverable
    HALTED = "halted"                # terminal for the night
    MORNING = "morning"


class Gate(str, enum.Enum):
    """The gate stack. All must pass for a cue to be proposed (design.md §12.2).

    G9 is absent by design. Eye-movement activity may not influence cue timing
    until it has cleared the M3 validation gate; until then it is computed,
    logged, and ignored.
    """

    G1_MIN_DELAY = "g1_min_delay"
    G2_PERMITTED_WINDOW = "g2_permitted_window"
    G3_SIGNAL_QUALITY = "g3_signal_quality"
    G4_BODY_MOTION = "g4_body_motion"
    G5_NO_RECENT_AROUSAL = "g5_no_recent_arousal"
    G6_COOLDOWN = "g6_cooldown"
    G7_NIGHTLY_CAP = "g7_nightly_cap"
    G8_EXPERIMENT = "g8_experiment_condition"
    # Present in the enum, but only ever evaluated when sensor timing has been
    # authorised by a passing H1 validation. Absent from the stack otherwise.
    G9_EYE_ACTIVITY = "g9_eye_activity"


@dataclass
class GateResult:
    gate: Gate
    passed: bool
    detail: str = ""
    value: Optional[float] = None


@dataclass
class GateSnapshot:
    """The full gate evaluation at one instant, stored with every cue.

    Recording all gates rather than just the failing one means a night can be
    re-examined later to ask why cues did or did not fire, without re-running
    anything.
    """

    results: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def blocking(self) -> list[Gate]:
        return [r.gate for r in self.results if not r.passed]

    def to_dict(self) -> dict:
        return {
            r.gate.value: {"passed": r.passed, "detail": r.detail, "value": r.value}
            for r in self.results
        }


@dataclass
class CueCommand:
    """A decision to play a cue, already authorised by the supervisor."""

    gain: float
    gain_requested: float
    ramp_ms: float
    duration_ms: float
    repetition_index: int
    policy_version: str
    gates: GateSnapshot
    trigger: str = "scheduled"


class Outcome(str, enum.Enum):
    """Post-cue response classification (design.md §12.4)."""

    QUIET = "quiet"
    PROBABLE_AROUSAL = "probable_arousal"
    POSSIBLE_AWAKENING = "possible_awakening"
    UNCERTAIN = "uncertain"
