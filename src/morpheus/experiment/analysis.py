"""N-of-1 analysis: per-arm lucidity rates with honest uncertainty.

Beta-binomial rather than a frequentist test, for a specific reason. At this
sample size a p-value implies a precision the design does not have, and a
non-significant result would be read as "no effect" when it usually means "not
enough nights". A posterior distribution says the same thing without the
false comfort: wide intervals look wide.

The prior is Jeffreys (Beta(0.5, 0.5)) — weakly informative, standard for a
proportion, and it behaves sensibly at zero events, which matters when a
14-night arm may contain no lucid dreams at all.

What this deliberately does not do: adjust for covariates, drop outliers, or
try alternative specifications. All three are where an analyst who wants a
particular answer finds one. Covariates are reported alongside so they can be
inspected; the primary comparison stays fixed.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .randomization import Arm

PRIOR_A = 0.5
PRIOR_B = 0.5
POSTERIOR_DRAWS = 200_000


@dataclass
class ArmSummary:
    arm: Arm
    nights: int
    lucid: int
    rate: float
    ci_low: float
    ci_high: float

    @property
    def per_week(self) -> float:
        return self.rate * 7.0


@dataclass
class Comparison:
    treatment: Arm
    control: Arm
    prob_treatment_better: float
    diff_median: float
    diff_ci_low: float
    diff_ci_high: float

    @property
    def credible_interval_excludes_zero(self) -> bool:
        return self.diff_ci_low > 0 or self.diff_ci_high < 0


@dataclass
class AnalysisResult:
    experiment: str
    arms: list[ArmSummary] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    excluded_nights: int = 0
    exclusion_reasons: dict[str, int] = field(default_factory=dict)
    blinding: dict = field(default_factory=dict)
    prereg_intact: bool = True
    total_nights: int = 0
    guess_accuracy: Optional[float] = None


def _posterior(lucid: int, nights: int, rng: np.random.Generator) -> np.ndarray:
    return rng.beta(PRIOR_A + lucid, PRIOR_B + max(0, nights - lucid), POSTERIOR_DRAWS)


def summarise_arm(arm: Arm, lucid: int, nights: int, rng: np.random.Generator) -> ArmSummary:
    if nights == 0:
        return ArmSummary(arm, 0, 0, float("nan"), float("nan"), float("nan"))
    draws = _posterior(lucid, nights, rng)
    low, high = np.percentile(draws, [2.5, 97.5])
    return ArmSummary(
        arm=arm, nights=nights, lucid=lucid,
        rate=float(np.median(draws)), ci_low=float(low), ci_high=float(high),
    )


def compare(
    treatment: tuple[Arm, int, int],
    control: tuple[Arm, int, int],
    rng: np.random.Generator,
) -> Comparison:
    """Posterior comparison of two arms, each given as (arm, lucid, nights)."""
    t_arm, t_lucid, t_nights = treatment
    c_arm, c_lucid, c_nights = control
    t = _posterior(t_lucid, t_nights, rng)
    c = _posterior(c_lucid, c_nights, rng)
    diff = t - c
    low, high = np.percentile(diff, [2.5, 97.5])
    return Comparison(
        treatment=t_arm,
        control=c_arm,
        prob_treatment_better=float(np.mean(diff > 0)),
        diff_median=float(np.median(diff)),
        diff_ci_low=float(low),
        diff_ci_high=float(high),
    )


def analyse(
    conn: sqlite3.Connection,
    experiment_id: int,
    revealed: Sequence[tuple[str, Arm]],
    *,
    blinding: Optional[dict] = None,
    prereg_intact: bool = True,
    seed: int = 20260806,
) -> AnalysisResult:
    """Run the pre-registered analysis over revealed nights only.

    `revealed` comes from ExperimentStore.revealed_arms, so a night whose arm
    has not been legitimately unsealed is invisible here. That is intentional:
    it makes a mid-study peek impossible to launder through the analysis.
    """
    rng = np.random.default_rng(seed)
    row = conn.execute("SELECT name FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    name = row["name"] if row else str(experiment_id)

    result = AnalysisResult(experiment=name, blinding=blinding or {}, prereg_intact=prereg_intact)
    forced_dates = set()
    if blinding:
        forced_dates = set(blinding.get("forced_dates") or [])

    tally: dict[Arm, list[int]] = {}
    guesses: list[tuple[str, Arm]] = []

    for report_date, arm in revealed:
        report = conn.execute(
            "SELECT * FROM reports WHERE report_date = ?", (report_date,)
        ).fetchone()
        if report is None:
            result.excluded_nights += 1
            result.exclusion_reasons["no report"] = result.exclusion_reasons.get("no report", 0) + 1
            continue
        if report["lucid_binary"] is None:
            result.excluded_nights += 1
            result.exclusion_reasons["outcome not scored"] = (
                result.exclusion_reasons.get("outcome not scored", 0) + 1
            )
            continue

        lucid, nights = tally.setdefault(arm, [0, 0])
        tally[arm] = [lucid + int(bool(report["lucid_binary"])), nights + 1]
        result.total_nights += 1
        if report["guessed_condition"]:
            guesses.append((report["guessed_condition"], arm))

    for arm, (lucid, nights) in sorted(tally.items(), key=lambda kv: kv[0].value):
        result.arms.append(summarise_arm(arm, lucid, nights, rng))

    # Every non-control arm against the no-cue control, which is the
    # pre-registered comparison.
    if Arm.NO_CUE in tally:
        control = (Arm.NO_CUE, tally[Arm.NO_CUE][0], tally[Arm.NO_CUE][1])
        for arm, (lucid, nights) in tally.items():
            if arm is Arm.NO_CUE:
                continue
            result.comparisons.append(compare((arm, lucid, nights), control, rng))

    if guesses:
        # If the participant can name the arm better than chance, the blind is
        # not holding and the primary result is weakened regardless of its size.
        correct = sum(1 for guess, arm in guesses if guess.strip().lower() in arm.value)
        result.guess_accuracy = correct / len(guesses)

    return result


def format_result(result: AnalysisResult) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"N-of-1 analysis — {result.experiment}")
    add("=" * 70)
    if not result.prereg_intact:
        add("  !! PRE-REGISTRATION HASH MISMATCH")
        add("     The stored plan has been edited since it was registered. Any")
        add("     result below should be treated as exploratory.")
        add("")

    add(f"  nights analysed   {result.total_nights}")
    if result.excluded_nights:
        add(f"  excluded          {result.excluded_nights}")
        for reason, count in sorted(result.exclusion_reasons.items()):
            add(f"                      {reason}: {count}")
    add("")

    if not result.arms:
        add("  No revealed nights yet. Arms stay sealed until their morning")
        add("  report is submitted, so there is nothing to analyse.")
        return "\n".join(lines)

    add("Per-arm lucidity rate (posterior median, 95% CI)")
    add("-" * 70)
    add(f"  {'arm':<16} {'nights':>7} {'lucid':>6}  {'rate':>18}  per week")
    for arm in result.arms:
        if arm.nights == 0:
            continue
        ci = f"{arm.rate:.1%} [{arm.ci_low:.1%}, {arm.ci_high:.1%}]"
        add(f"  {arm.arm.value:<16} {arm.nights:>7} {arm.lucid:>6}  {ci:>18}  {arm.per_week:.2f}")
    add("")

    if result.comparisons:
        add("Comparison against no-cue control")
        add("-" * 70)
        for c in result.comparisons:
            add(f"  {c.treatment.value} vs {c.control.value}")
            add(f"    P(treatment better)   {c.prob_treatment_better:.3f}")
            add(
                f"    difference in rate    {c.diff_median:+.1%} "
                f"[{c.diff_ci_low:+.1%}, {c.diff_ci_high:+.1%}]"
            )
            if c.credible_interval_excludes_zero:
                add("    credible interval excludes zero")
            else:
                add("    credible interval includes zero — no reliable difference yet")
            add("")

    add("Blinding")
    add("-" * 70)
    b = result.blinding or {}
    add(f"  nights assigned   {b.get('nights_assigned', 0)}")
    add(f"  forced reveals    {b.get('forced_reveals', 0)}")
    add(f"  blocked attempts  {b.get('blocked_attempts', 0)}")
    if result.guess_accuracy is not None:
        add(f"  guess accuracy    {result.guess_accuracy:.1%}")
        if result.guess_accuracy > 0.7:
            add("    Above chance. The blind is not holding, and the primary")
            add("    result is weakened regardless of its magnitude.")
    add("")
    add("Intervals are wide because the sample is small. That is the honest")
    add("state of the evidence, not a presentation problem to be tightened.")
    return "\n".join(lines)
