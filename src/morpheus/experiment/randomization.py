"""Seeded block randomization for N-of-1 trials.

Two properties matter more here than they would in a group study.

**Reproducibility.** The entire assignment sequence is a pure function of
(seed, design). It is never stored as the source of truth and never drawn
incrementally from a mutable RNG — regenerate it from the seed and you get the
same nights, forever. That is what makes the analysis auditable by someone who
was not present, including yourself in six months.

**Balance over short runs.** At N-of-1 scale, simple randomization routinely
produces runs like AAAA that waste weeks. Block randomization guarantees each
arm appears equally often within every block, so a study stopped early is still
roughly balanced. The cost is predictability: the last entry in a block is
determined by the preceding ones, so a participant who is counting can infer it.
Since the participant here is also the developer, that is a real leak — it is
measured via the `guessed_condition` report field rather than waved away, and
block size is configurable so it can be traded off.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence


class Arm(str, Enum):
    """Experimental conditions (design.md §15.1).

    Training happens on every night including NO_CUE. Otherwise the arms differ
    in two variables at once and the design measures nothing.
    """

    TRAINED_CUE = "trained_cue"
    UNTRAINED_CUE = "untrained_cue"
    NO_CUE = "no_cue"

    @property
    def plays_audio(self) -> bool:
        return self is not Arm.NO_CUE


# Two-arm is the recommended starting design: it halves the calendar and still
# answers "does the intervention beat nothing". Three-arm additionally separates
# conditioning from mere sound, at roughly double the nights.
DESIGN_TWO_ARM: tuple[Arm, ...] = (Arm.TRAINED_CUE, Arm.NO_CUE)
DESIGN_THREE_ARM: tuple[Arm, ...] = (Arm.TRAINED_CUE, Arm.UNTRAINED_CUE, Arm.NO_CUE)

DESIGNS: dict[str, tuple[Arm, ...]] = {
    "two-arm": DESIGN_TWO_ARM,
    "three-arm": DESIGN_THREE_ARM,
}


@dataclass(frozen=True)
class Assignment:
    night_index: int
    arm: Arm
    block_index: int
    position_in_block: int


@dataclass
class RandomizationPlan:
    """A reproducible assignment sequence.

    `seed` is stored in the experiments table. Given it and the design, the
    whole sequence regenerates exactly — so the plan is derived, not persisted
    as truth, and cannot silently drift from what was pre-registered.
    """

    seed: int
    arms: tuple[Arm, ...]
    repeats_per_block: int = 2
    _cache: dict[int, list[Assignment]] = field(default_factory=dict, repr=False)

    @property
    def block_size(self) -> int:
        return len(self.arms) * self.repeats_per_block

    def _block(self, block_index: int) -> list[Assignment]:
        """Shuffle one block, seeded by (seed, block_index).

        Deriving each block from its own index rather than advancing a shared
        RNG means block 40 can be computed without generating blocks 0-39. That
        keeps `assignment_for(night)` O(1) and, more importantly, keeps the
        sequence stable if the plan is ever regenerated from a different
        starting point.
        """
        if block_index in self._cache:
            return self._cache[block_index]

        digest = hashlib.sha256(f"{self.seed}:{block_index}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        pool = list(self.arms) * self.repeats_per_block
        rng.shuffle(pool)

        start = block_index * self.block_size
        block = [
            Assignment(
                night_index=start + position + 1,
                arm=arm,
                block_index=block_index,
                position_in_block=position,
            )
            for position, arm in enumerate(pool)
        ]
        self._cache[block_index] = block
        return block

    def assignment_for(self, night_index: int) -> Assignment:
        """Assignment for a 1-based night index."""
        if night_index < 1:
            raise ValueError("night_index is 1-based")
        offset = night_index - 1
        block_index, position = divmod(offset, self.block_size)
        return self._block(block_index)[position]

    def sequence(self, nights: int) -> list[Assignment]:
        return [self.assignment_for(n) for n in range(1, nights + 1)]

    def counts(self, nights: int) -> dict[Arm, int]:
        tally = {arm: 0 for arm in self.arms}
        for assignment in self.sequence(nights):
            tally[assignment.arm] += 1
        return tally

    def fingerprint(self) -> str:
        """Identifies the plan in the pre-registration and the database."""
        spec = f"{self.seed}|{'.'.join(a.value for a in self.arms)}|{self.repeats_per_block}"
        return hashlib.sha256(spec.encode()).hexdigest()[:16]


def make_plan(
    *,
    seed: Optional[int] = None,
    design: str = "two-arm",
    repeats_per_block: int = 2,
) -> RandomizationPlan:
    if design not in DESIGNS:
        raise KeyError(f"unknown design {design!r}; available: {sorted(DESIGNS)}")
    if seed is None:
        seed = random.SystemRandom().randrange(2**31)
    return RandomizationPlan(
        seed=int(seed), arms=DESIGNS[design], repeats_per_block=repeats_per_block
    )


def imbalance(counts: dict[Arm, int]) -> int:
    """Largest minus smallest arm count. Bounded by block size at all times."""
    if not counts:
        return 0
    return max(counts.values()) - min(counts.values())


def required_nights_per_arm(
    baseline_rate: float, target_rate: float, *, power: float = 0.8, alpha: float = 0.05
) -> Optional[int]:
    """Rough nights-per-arm for a two-proportion comparison.

    A normal approximation, which is optimistic at these rates and sample
    sizes — treat it as a floor rather than an estimate. It exists to make the
    calendar cost visible before committing: at a 10% baseline and a 25% target
    it returns a number in the low hundreds, which is the single most important
    fact about this study design (design.md §15.4).
    """
    from math import sqrt

    from scipy.stats import norm  # local import; scipy is heavy to import

    if not (0 < baseline_rate < 1 and 0 < target_rate < 1):
        return None
    if abs(target_rate - baseline_rate) < 1e-9:
        return None

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    p_bar = (baseline_rate + target_rate) / 2
    numerator = (
        z_alpha * sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * sqrt(baseline_rate * (1 - baseline_rate) + target_rate * (1 - target_rate))
    ) ** 2
    return int(numerator / (target_rate - baseline_rate) ** 2 + 0.999)
