"""Cue policy: when to propose a cue, and how loud.

M2 ships a transparent heuristic and nothing else. That is a deliberate choice
rather than a placeholder (design.md §12.5): with zero nights of data, any
learner would be fitting noise, and a hand-written rule gives the adaptive layer
in M5 a legible baseline it has to beat.

The Policy protocol is the seam Thompson sampling will slot into later. Note
what it does *not* have access to: the safety limits. A policy proposes a gain;
the supervisor decides what is permitted. Keeping the caps outside this
interface is what stops a future learner from discovering that louder is better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from .state import GateSnapshot


class Policy(Protocol):
    """Proposes cue parameters. Has no authority to play anything."""

    @property
    def version(self) -> str: ...

    def propose_gain(self, *, cue_index: int, last_outcome: Optional[str]) -> float: ...

    def propose_ramp_ms(self) -> float: ...

    def propose_duration_ms(self) -> float: ...

    def should_propose(self, *, now_mono: float, gates: GateSnapshot) -> bool: ...


@dataclass
class ScheduledPolicy:
    """Fixed schedule with bounded, outcome-aware volume stepping.

    Cue timing follows the published TLR protocol: nothing before the minimum
    delay, then opportunities whenever the gates and the supervisor allow. The
    scheduler itself proposes at every eligible moment; all the actual timing
    restraint lives in the cooldown and the caps, which are enforced elsewhere
    and cannot be overridden here.

    Volume adapts within a narrow band using the previous cue's outcome. The
    step is asymmetric on purpose: it backs off faster than it advances, because
    the two errors are not equally costly. A cue that was too quiet wastes one
    opportunity out of hundreds; a cue that wakes the user costs sleep quality,
    a night of data, and some of the adherence a months-long study depends on.
    """

    start_gain: float = 0.08
    min_gain: float = 0.02
    max_gain: float = 0.30
    step_up: float = 0.02
    step_down: float = 0.05
    ramp_ms: float = 4000.0
    duration_ms: float = 9000.0

    _current_gain: Optional[float] = None

    @property
    def version(self) -> str:
        return (
            f"scheduled-v1(start={self.start_gain},up={self.step_up},"
            f"down={self.step_down},ramp={self.ramp_ms:.0f})"
        )

    def reset(self) -> None:
        self._current_gain = None

    def propose_gain(self, *, cue_index: int, last_outcome: Optional[str]) -> float:
        if self._current_gain is None:
            self._current_gain = self.start_gain
            return self._current_gain

        if last_outcome == "quiet":
            self._current_gain += self.step_up
        elif last_outcome == "probable_arousal":
            self._current_gain -= self.step_down
        # `uncertain` deliberately changes nothing. Learning from a window whose
        # signal quality collapsed would be learning from noise, and the
        # direction of that error is unknowable.

        self._current_gain = min(max(self._current_gain, self.min_gain), self.max_gain)
        return self._current_gain

    def propose_ramp_ms(self) -> float:
        return self.ramp_ms

    def propose_duration_ms(self) -> float:
        return self.duration_ms

    def should_propose(self, *, now_mono: float, gates: GateSnapshot) -> bool:
        return gates.passed
