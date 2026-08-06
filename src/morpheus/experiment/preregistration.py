"""Generating the pre-registration document.

A pre-registration is a commitment made before seeing data: what will be
measured, how it will be analysed, and what would count as a negative result.
Its value comes entirely from being fixed in advance — an analysis plan written
afterwards is a description of what happened to work.

That is a sharper problem here than in a group study. The participant, the
analyst, and the person who wants the answer to be yes are the same person, and
there are no co-authors to notice a moved goalpost. So the document is generated
from the actual experiment configuration, hashed, and the hash is checked before
any analysis runs.
"""

from __future__ import annotations

from datetime import date

from ..report.schema import PRIMARY_OUTCOME_DEFINITION
from .randomization import Arm, RandomizationPlan, required_nights_per_arm

TEMPLATE = """# Pre-registration — {name}

Generated {today}. Plan fingerprint `{fingerprint}`.

This document is hashed at creation. Analysis refuses to run if the stored text
no longer matches its hash, because a pre-registration that can be edited after
data exists is not a pre-registration.

## Hypothesis

{hypothesis}

## Design

N-of-1 randomized trial, {design}, single participant.

| Arm | Cue | Purpose |
|---|---|---|
{arm_rows}

Conditioning training is performed on **every** night, including no-cue nights.
Otherwise the arms would differ in two variables at once and the comparison
would measure nothing.

## Randomization

- Block randomization, block size {block_size} ({repeats} of each arm per block).
- Seed `{seed}`. The entire sequence is reproducible from the seed and design.
- Assignments are sealed at generation and unsealed only after the morning
  report for that date has been submitted.

Known limitation: with block randomization the final entry in a block is
determined by the preceding ones. A participant who tracks assignments can
sometimes infer it. This is why `guessed_condition` is collected on every
report — unblinding is measured rather than assumed away.

## Primary outcome

Binary, per night:

> {primary}

Fixed before data collection. Not revised mid-study under any circumstances.

## Secondary outcomes

- Lucidity confidence (0–4)
- Whether awareness occurred during the dream, asked separately
- Cue heard directly / incorporated indirectly / woke the participant
- Dreams recalled, vividness (1–5), sleep quality (1–5), awakenings

## Safety outcomes

- Cue-attributed awakening rate
- Weekly sleep-quality trend

Stopping rule: if sleep quality declines for seven consecutive nights, the study
halts automatically.

## Analysis plan

Primary: Bayesian beta-binomial comparison of per-night lucidity rate between
arms, reporting the posterior probability that the trained-cue arm exceeds
control, with 95% credible intervals on each arm's rate and on the difference.

Point estimates are reported with credible intervals throughout. p-values are
not reported: at this sample size they would imply a precision the design does
not have.

Covariates recorded and reported alongside, not adjusted for unless
pre-specified: night index (practice effects are large in lucid-dream training),
training adherence, and camera coverage where applicable.

Nights excluded from the primary analysis:

- Any night whose arm was force-revealed before its report
- Any night with no morning report
- Any night the participant flags as invalid at report time

## Sample size

Baseline rate assumed {baseline:.1%} per night; target {target:.1%}.

{power_note}

This is the dominant cost of the design. It is stated here so that stopping
early is recognised as a decision with consequences rather than a natural end.

## What would count as a negative result

- The credible interval on the difference includes zero after the planned
  number of nights.
- The trained-cue arm does not exceed control with posterior probability ≥ 0.90.

A negative result is reported as such. The trial is not extended in search of a
different answer, and the analysis is not re-specified after seeing the data.

## Deviations

Any change after `started_at` must be recorded here with its date and reason,
and invalidates the hash, which the analysis will report.
"""

_ARM_DESCRIPTIONS = {
    Arm.TRAINED_CUE: ("Conditioned cue", "Full intervention"),
    Arm.UNTRAINED_CUE: (
        "Acoustically matched, never trained",
        "Isolates conditioning from mere sound during sleep",
    ),
    Arm.NO_CUE: ("Nothing plays", "Isolates sound from training and expectancy"),
}

DEFAULT_HYPOTHESIS = (
    "Delivering a conditioned auditory cue during sleep, following pre-sleep "
    "conditioning that pairs the cue with a state of critical self-awareness, "
    "increases the proportion of nights on which a lucid dream is reported, "
    "relative to conditioning alone with no cue."
)


def generate(
    *,
    name: str,
    plan: RandomizationPlan,
    design: str,
    hypothesis: str = DEFAULT_HYPOTHESIS,
    baseline_rate: float = 0.10,
    target_rate: float = 0.25,
) -> str:
    arm_rows = "\n".join(
        f"| {arm.value} | {_ARM_DESCRIPTIONS[arm][0]} | {_ARM_DESCRIPTIONS[arm][1]} |"
        for arm in plan.arms
    )

    needed = required_nights_per_arm(baseline_rate, target_rate)
    if needed:
        total = needed * len(plan.arms)
        power_note = (
            f"A normal-approximation calculation gives roughly **{needed} nights per arm** "
            f"for 80% power at alpha 0.05 — about **{total} nights total**, or "
            f"**{total / 30.0:.0f} months** of nightly compliance. The approximation is "
            f"optimistic at these rates, so treat it as a floor."
        )
    else:
        power_note = "Sample size could not be estimated from the supplied rates."

    return TEMPLATE.format(
        name=name,
        today=date.today().isoformat(),
        fingerprint=plan.fingerprint(),
        hypothesis=hypothesis,
        design=design,
        arm_rows=arm_rows,
        block_size=plan.block_size,
        repeats=plan.repeats_per_block,
        seed=plan.seed,
        primary=PRIMARY_OUTCOME_DEFINITION,
        baseline=baseline_rate,
        target=target_rate,
        power_note=power_note,
    )
