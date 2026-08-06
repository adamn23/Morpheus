"""Thompson sampling over a small set of pre-declared cue arms.

Why a bandit and not something larger: the lifetime decision count of this
system is roughly a few cues a night for a few hundred nights — order 1,000
pulls. Deep RL needs four orders of magnitude more. Beta-Bernoulli Thompson
sampling is near-optimal at this scale, needs no tuning, survives restarts as
two numbers per arm, and can explain any decision it makes as "this arm's
posterior sampled highest tonight" (design.md §12.5).

The arm space is deliberately tiny and fully enumerated in advance: three gains
crossed with three delays. Every arm is constructed *inside* the safety limits,
and the supervisor still adjudicates every cue afterwards. The learner therefore
cannot reach an unsafe action even in principle — not because it is well
behaved, but because no such action exists in its action set.

Reward is deliberately not "was the cue heard". It is a composite that penalises
arousal, because an objective of pure salience has an obvious degenerate
solution: shout. The reward here treats waking the sleeper as a failure, which
makes the quiet-but-incorporated outcome the thing worth finding.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np

from .safety import SafetyLimits
from .state import GateSnapshot, Outcome

log = logging.getLogger("morpheus.adaptive")

# Nights of data below which the bandit defers to the heuristic. With fewer
# observations than arms, sampling is just an expensive random choice, and a
# random walk through cue volume is a worse experience than a fixed schedule.
COLD_START_PULLS = 12


@dataclass(frozen=True)
class CueArm:
    """One combination of cue parameters the bandit may choose."""

    gain: float
    delay_offset_s: float

    @property
    def key(self) -> str:
        return f"g{self.gain:.3f}/d{self.delay_offset_s:+.0f}"

    def __str__(self) -> str:
        return f"gain {self.gain:.2f}, delay {self.delay_offset_s / 60:+.0f} min"


def build_arms(limits: SafetyLimits, *, gains: Optional[Sequence[float]] = None) -> list[CueArm]:
    """Enumerate the action space, clipped into the safety envelope.

    Clipping happens here, at construction, so that an unsafe arm never exists
    for the learner to select. This is the structural half of the guarantee; the
    supervisor's per-cue check is the other half.
    """
    if gains is None:
        span = limits.max_gain - limits.min_gain
        gains = [
            limits.min_gain + span * fraction for fraction in (0.15, 0.35, 0.60)
        ]
    gains = [float(np.clip(g, limits.min_gain, limits.max_gain)) for g in gains]
    delays = (-1800.0, 0.0, 1800.0)
    return [CueArm(gain=g, delay_offset_s=d) for g in gains for d in delays]


def reward_for(outcome: Outcome, *, cue_heard: Optional[bool] = None,
               lucid: Optional[bool] = None) -> Optional[float]:
    """Map a night's result to a reward in [0, 1], or None to skip learning.

    Ordering, best to worst: a lucid night, then a quiet cue that was noticed,
    then a quiet cue, then an arousal, then an awakening. Waking the sleeper
    scores zero rather than merely low — the bandit should never trade sleep for
    salience, and the cheapest way to guarantee that is to make it worthless.

    `uncertain` returns None. Learning from a window whose signal collapsed
    would be learning from noise in an unknown direction.
    """
    if outcome is Outcome.UNCERTAIN:
        return None
    if outcome is Outcome.POSSIBLE_AWAKENING:
        return 0.0
    if outcome is Outcome.PROBABLE_AROUSAL:
        return 0.15

    # Quiet: the cue landed without disturbing sleep.
    if lucid:
        return 1.0
    if cue_heard:
        return 0.8
    return 0.5


@dataclass
class ThompsonPolicy:
    """Beta-Bernoulli Thompson sampling with a heuristic cold start.

    Posteriors persist in `policy_state`, so a restart mid-study does not reset
    what has been learned.
    """

    arms: list[CueArm]
    limits: SafetyLimits
    name: str = "thompson-v1"
    ramp_ms: float = 4000.0
    duration_ms: float = 9000.0
    fallback_gain: float = 0.08
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    _alpha: dict[str, float] = field(default_factory=dict)
    _beta: dict[str, float] = field(default_factory=dict)
    _pulls: dict[str, int] = field(default_factory=dict)
    _last_choice: Optional[CueArm] = None

    def __post_init__(self) -> None:
        for arm in self.arms:
            self._alpha.setdefault(arm.key, 1.0)
            self._beta.setdefault(arm.key, 1.0)
            self._pulls.setdefault(arm.key, 0)

    @property
    def version(self) -> str:
        return f"{self.name}({len(self.arms)} arms, {self.total_pulls} pulls)"

    @property
    def total_pulls(self) -> int:
        return sum(self._pulls.values())

    @property
    def cold(self) -> bool:
        return self.total_pulls < COLD_START_PULLS

    # ---------------------------------------------------------------- choose

    def select(self) -> CueArm:
        """Sample each arm's posterior; pick the highest draw.

        During cold start, cycle deterministically so every arm is tried before
        any is preferred. Thompson sampling would get there eventually, but on a
        budget of a few hundred pulls the wasted exploration is worth avoiding.
        """
        if self.cold:
            arm = self.arms[self.total_pulls % len(self.arms)]
        else:
            draws = {
                a.key: self.rng.beta(self._alpha[a.key], self._beta[a.key]) for a in self.arms
            }
            best = max(draws, key=draws.__getitem__)
            arm = next(a for a in self.arms if a.key == best)
        self._last_choice = arm
        return arm

    def propose_gain(self, *, cue_index: int, last_outcome: Optional[str]) -> float:
        arm = self.select()
        return float(np.clip(arm.gain, self.limits.min_gain, self.limits.max_gain))

    def propose_ramp_ms(self) -> float:
        return self.ramp_ms

    def propose_duration_ms(self) -> float:
        return self.duration_ms

    def should_propose(self, *, now_mono: float, gates: GateSnapshot) -> bool:
        return gates.passed

    @property
    def last_choice(self) -> Optional[CueArm]:
        return self._last_choice

    # ----------------------------------------------------------------- learn

    def update(self, arm: CueArm, reward: Optional[float]) -> None:
        """Fractional Beta update. `None` reward changes nothing."""
        if reward is None:
            return
        reward = float(np.clip(reward, 0.0, 1.0))
        self._alpha[arm.key] += reward
        self._beta[arm.key] += 1.0 - reward
        self._pulls[arm.key] += 1

    def posterior(self, arm: CueArm) -> tuple[float, float, int]:
        return self._alpha[arm.key], self._beta[arm.key], self._pulls[arm.key]

    def expected_value(self, arm: CueArm) -> float:
        a, b, _ = self.posterior(arm)
        return a / (a + b)

    def ranking(self) -> list[tuple[CueArm, float, int]]:
        ranked = [(a, self.expected_value(a), self._pulls[a.key]) for a in self.arms]
        ranked.sort(key=lambda item: -item[1])
        return ranked

    # ------------------------------------------------------------ persistence

    def load(self, conn: sqlite3.Connection) -> None:
        for row in conn.execute(
            "SELECT arm_key, successes, failures, pulls FROM policy_state WHERE policy_name = ?",
            (self.name,),
        ):
            key = row["arm_key"]
            if key in self._alpha:
                self._alpha[key] = 1.0 + float(row["successes"])
                self._beta[key] = 1.0 + float(row["failures"])
                self._pulls[key] = int(row["pulls"])

    def save(self, conn: sqlite3.Connection) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for arm in self.arms:
            conn.execute(
                "INSERT INTO policy_state (policy_name, arm_key, successes, failures, "
                "pulls, updated_at) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(policy_name, arm_key) DO UPDATE SET "
                "successes=excluded.successes, failures=excluded.failures, "
                "pulls=excluded.pulls, updated_at=excluded.updated_at",
                (
                    self.name, arm.key,
                    self._alpha[arm.key] - 1.0, self._beta[arm.key] - 1.0,
                    self._pulls[arm.key], now,
                ),
            )


def log_counterfactual(
    conn: sqlite3.Connection,
    *,
    cue_id: Optional[int],
    t_mono: float,
    chosen_policy: str,
    chosen_arm: str,
    baseline_policy: str,
    baseline_arm: str,
    context: Optional[dict] = None,
) -> None:
    """Record what the incumbent policy would have done.

    Without this there is no way to answer the only question that justifies the
    adaptive layer: did it beat the heuristic it replaced? Comparing the bandit
    against its own history cannot answer that, because the history is what the
    bandit chose.
    """
    import json

    conn.execute(
        "INSERT INTO counterfactuals (cue_id, t_mono, chosen_policy, chosen_arm, "
        "baseline_policy, baseline_arm, agreed, context_json) VALUES (?,?,?,?,?,?,?,?)",
        (
            cue_id, t_mono, chosen_policy, chosen_arm, baseline_policy, baseline_arm,
            int(chosen_arm == baseline_arm), json.dumps(context or {}),
        ),
    )


def format_ranking(policy: ThompsonPolicy) -> str:
    lines = [
        f"Adaptive policy — {policy.version}",
        "=" * 60,
    ]
    if policy.cold:
        lines.append(
            f"  Cold start: {policy.total_pulls}/{COLD_START_PULLS} pulls. Arms are "
            f"being cycled deterministically; no preference is being expressed yet."
        )
        lines.append("")
    lines.append(f"  {'arm':<28} {'E[reward]':>10} {'pulls':>7}")
    lines.append("  " + "-" * 47)
    for arm, value, pulls in policy.ranking():
        lines.append(f"  {str(arm):<28} {value:>10.3f} {pulls:>7}")
    lines.append("")
    lines.append("  Expected values with few pulls are mostly prior, not evidence.")
    return "\n".join(lines)
