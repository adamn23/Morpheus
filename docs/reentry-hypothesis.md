# The re-entry hypothesis — Morpheus reduced to one measurable question

**2026-08-12.** Supersedes the daily-lucidity projections in
`docs/daily-lucidity.md`, which rested on a statistical error.

---

## 0. Two retractions

**Lab percentages cannot be used as per-attempt probabilities.** "40% of
participants signal-verified within two nights" and "50% of participants in a
TLR nap protocol" are *participant-level* proportions — the fraction of people
achieving lucidity **at least once**. With strong between-person heterogeneity
(some people never do it, some do it readily), a 40% population figure is
consistent with a very wide range of per-attempt probabilities, and the two
cannot be separated without the underlying attempt counts. Substituting them
into `1−(1−p)ⁿ` conflates participant-level, session-level and attempt-level
rates. **Every projection built on that arithmetic is withdrawn.**

**REM rebound was overgeneralised.** Rebound follows *sustained selective REM
deprivation*. A brief arousal from mid- or late-REM is not deprivation. It may
shorten subsequent REM latency, do nothing, or fragment sleep and *reduce*
usable REM. **This becomes an experiment (S7), not a premise.**

---

## 1. The hypothesis

> **H₀ (falsifiable):** In machine-served REM → arousal → re-entry transitions,
> the probability of an LRLR-verified lucid dream with **short latency from REM
> re-entry** is `p_serve`, and Morpheus can serve `n` such transitions per night
> without degrading sleep below a pre-set floor.
>
> **Daily lucidity is credible if and only if the *measured* `p_serve` and `n`
> jointly satisfy `P(≥1 lucid/night) ≥ 0.8` under the *measured* clustering.**

`p_serve` is the unknown. Everything below exists to estimate it honestly. No
number for lucid nights per week appears in this document, because none has been
measured.

### What p_serve would have to be

Required per-attempt probability for `P(≥1) ≥ 0.8`, at measured `n`, under
independence and under moderate night-level clustering (ICC ≈ 0.3):

| n served/night | independent | clustered |
|---|---|---|
| 3 | 0.42 | ~0.52 |
| 4 | 0.33 | ~0.42 |
| 5 | 0.28 | ~0.36 |
| **6** | 0.24 | **~0.32** |
| 8 | 0.18 | ~0.25 |

**So the target to beat is roughly `p_serve ≥ 0.32` at `n = 6`.** That is a
concrete falsification threshold. If the measured value lands near 0.10, daily
lucidity via this route is arithmetically dead regardless of engineering.

---

## 2. What counts as one attempt

Without a prospective definition, `p_serve` is unmeasurable — you would count
only the transitions that looked promising and inflate it.

**A served re-entry attempt is declared when, and only when:**

1. The subject was in scored REM for ≥3 min immediately prior, **and**
2. an arousal is detected (§3), **and**
3. ≥20 min has elapsed since the previous attempt, **and**
4. the attempt is logged *at declaration*, before any outcome is known.

**Attempt declaration must be identical on served and unserved nights.** On
control nights the machine declares the attempt and does nothing. This is the
single most important methodological safeguard in the programme.

**Attempts are not independent.** Analysis uses a mixed-effects logistic model
with night as a random effect; the reported `p_serve` is the marginal
probability, and the ICC is estimated rather than assumed. The 20-minute
separation and the requirement of an intervening REM period reduce local
dependence but do not remove night-level clustering.

---

## 3. Detecting the trajectory

Four states, with the AASM-derived criteria:

| State | Criteria |
|---|---|
| **REM** | desynchronised EEG, **low chin EMG**, REMs present |
| **Arousal** | abrupt EEG frequency shift (alpha / theta / >16 Hz) lasting ≥3 s, preceded by ≥10 s stable sleep — **and, in REM specifically, a concurrent submental EMG increase** |
| **Awakening** | ≥15 s of wake-criteria epoch, or sustained EMG tone, or movement |
| **Re-entry** | return to sleep criteria; REM re-entry when REM criteria are met again |

**Consequence that changes the hardware plan: scoring an arousal *during REM*
requires concurrent chin EMG by definition.** Chin EMG is therefore not a
"V2 nice-to-have" — it is **definitional for the core hypothesis** and moves
into the minimum viable montage.

---

## 4. Distinguishing preserved awareness from ordinary recognition

This is the hardest measurement in the programme and the one that decides
whether the hypothesis is even about what it claims to be about.

**The discriminator is latency from REM re-entry to first LRLR.**

- **Preserved awareness** → the subject is already aware as the dream forms and
  can signal almost immediately. Expected latency: **seconds to tens of seconds.**
- **Ordinary recognition (DILD)** → awareness is recovered later, after the dream
  has developed. Expected latency: **minutes.**

So the outcome is not "did a lucid dream occur" but **"did an LRLR occur within
T seconds of REM re-entry,"** with T pre-registered (proposal: 60 s).

**Pre-registered prediction:** if the mechanism is real, the latency
distribution over served transitions is **bimodal**, with a short-latency mode
absent or much smaller in unserved transitions. A unimodal distribution matching
unserved transitions falsifies the *mechanism* even if total lucidity rises —
which would mean the servo is helping by some other route and the theory is
wrong.

Self-report ("I was aware the whole time") is recorded but never substitutes.

---

## 5. What the machine does during the transition

Two failure modes bracket the target: **falling asleep normally** (awareness
lost) and **waking up properly** (sleep lost). The controller servos between
them.

```
on declared attempt:
    silence all other stimulation; suppress any scheduled cue
    begin anchor at the lowest intensity that was perceptible in calibration
    while t < 90 s:
        A < lower bound  → raise anchor one step   (falling too deep)
        A > upper bound  → cut anchor to zero      (waking too far)
        movement or EMG rise → ABORT, refractory
    release anchor; monitor for re-entry; log outcome and latency
```

### The anchor

Requirements: perceptible without semantic processing, continuously adjustable,
minimally arousing, and *attendable without effort* — an exhausted person cannot
sustain deliberate attention.

**Proposal [S]: a faint rhythmic pulse locked to the subject's own respiration**,
delivered tactilely or aurally.

Rationale: the classical WILD anchor is breath awareness, and a machine cannot
make someone count. But a signal *locked to their own breathing* is an
externalised version of the anchor they would otherwise have to generate — it
feels endogenous rather than intrusive, requires no interpretation, and provides
a rhythm that attention tracks passively.

Alternatives to test against it: fixed-rate pulse, continuous faint tone,
heartbeat-locked pulse. **Modality and locking are experimental variables, not
assumptions.**

Intensity adapts against arousal depth, not against time.

---

## 6. The experimental programme

Ordered by what must be known before the next step is interpretable. **The
denominator comes first** — everything else is uninterpretable without it.

### S0 — How many transitions actually exist *(14 nights, passive)*

Instrument only. Count declared attempts per night using §3, with **no
intervention whatsoever**.

**Output:** `n̂`, the harvestable transition rate, and its variance.
**Kill criterion:** if `n̂ < 3`, the throughput argument fails at the first step
and no controller can rescue it.

This is the number the entire design rests on and it is currently **unknown** —
the "five awakenings a night" figure is self-report and includes non-REM
arousals.

### S1 — LRLR channel *(concurrent with S0)*

Waking calibration ≥30 rehearsals; overnight specificity on nights with no
attempt.
**Criterion:** <1 false positive per 10 nights.
**Kill:** no verified outcome ⇒ `p_serve` is unmeasurable ⇒ programme stops.

### S2 — Unassisted baseline *(20 nights)*

Before building any servo: on spontaneous awakenings the subject deliberately
stays still and holds awareness, unaided. Machine declares attempts and records.

**Output:** `p_manual`. **This is the number the machine must beat.** Skipping
it would make any later result uninterpretable — a servo that matches
unassisted DEILD has added nothing.

### S3 — Can the servo hold the band *(20 nights)*

Pure control engineering. Outcome is **band-holding, not lucidity**: fraction of
attempts where `A` stays in target for ≥60 s without movement or awakening.
**Criterion:** ≥50%. **Kill:** below that, the controller does not work and
`p_serve` cannot be estimated.

### S4 — Estimate p_serve *(40+ nights, randomised within night)*

Each declared attempt randomised to **served** or **unserved-declared**.
Primary outcome: **LRLR within 60 s of REM re-entry.** Mixed-effects logistic,
night as random effect.

**Output:** `p̂_serve` with a confidence interval, and the ICC.
**Decision:** compare against the §1 table at the measured `n̂`.

### S5 — Attempt order and sleep cost *(within S4)*

`p_serve` by attempt index (1st, 2nd, 3rd…). Sleep efficiency, total sleep time,
WASO, REM%, and morning sleep-quality score versus harvest count.
**Stopping rule applies with full force** — this is the experiment most likely
to damage sleep.

### S6 — Cascade *(after S4)*

Following an LRLR-verified lucid dream, is a served re-entry followed by another
verified lucid dream? **Output:** `p_cascade` by position. Currently supported
only by practitioner anecdote.

### S7 — Does harvesting change REM supply *(analysis of S4 data)*

Measure REM latency and REM duration **following served versus unserved
attempts**, within night. Directly tests the retracted rebound claim.
Three possible answers: shortens (supply grows), no effect (supply fixed),
lengthens (harvesting is self-limiting). **All three are informative and the
third would cap the whole architecture.**

---

## 7. What the programme costs

| Stage | Nights | Adds |
|---|---|---|
| S0 + S1 | 14 | headband + chin EMG |
| S2 | 20 | nothing |
| S3 | 20 | anchor hardware |
| S4 + S5 | 40+ | nothing |
| S6, S7 | analysis | nothing |

**~95 nights to a defensible `p̂_serve`.** Roughly four months, and the two
cheapest stages (S0, S2) can each terminate it.

Sample size: at `n̂ ≈ 4` served attempts/night, 40 nights yields ~80 served
attempts — enough to estimate `p_serve` to about ±0.10 at p≈0.3, which is
sufficient to distinguish "≥0.32, daily lucidity is on the table" from "≈0.10,
it is not." Tightening beyond that needs 100+ nights and is not required for the
decision.

---

## 8. What is now unknown, and stays unknown until measured

Stated plainly, because the previous document filled these in with borrowed
numbers:

- `n̂` — transitions harvestable per night. **Unknown.**
- `p_manual` — unassisted DEILD success. **Unknown.**
- `p_serve` — machine-served success. **Unknown. This is the hypothesis.**
- ICC — night-level clustering. **Unknown.**
- `p_cascade` — chaining. **Unknown, anecdote only.**
- Sleep cost per harvested attempt. **Unknown.**
- Whether served awakenings change REM supply. **Unknown, and previously
  asserted.**

Seven unknowns, all measurable, none measured. **No projection of lucid nights
per week is defensible until at least `n̂` and `p_serve` exist.**

---

## 9. The single sentence

> **Morpheus exists to determine whether closed-loop control of REM re-entry
> can preserve metacognitive awareness often enough — and often enough per
> night — that near-nightly lucid dreaming becomes reachable; and if the answer
> is yes, to exploit it.**

The design succeeds if it answers that question honestly, including when the
answer is no. `p̂_serve` with a confidence interval, measured against a
pre-registered threshold at a measured `n̂`, is the deliverable.

**First action is unchanged and still costs nothing: S2's manual baseline can
begin tonight**, because it needs no hardware — only that spontaneous
awakenings are used deliberately and recorded honestly. S0's instrumented
denominator begins when the headband arrives.
