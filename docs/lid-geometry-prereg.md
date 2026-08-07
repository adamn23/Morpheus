# Pre-registration — lid-geometry confirmation

**Written and committed 2026-08-07, before the confirmation data exists.**

The analysis is frozen in `src/morpheus/calibration/confirmation.py` and run by
`morpheus confirm`. There is no flag to adjust anything. The verdict is a
function of the data alone.

---

## The finding being tested

Session 3 of the waking calibration (2026-08-07) produced:

| Measure | AUC | 95% CI |
|---|---|---|
| eye-flow, all windows | 0.573 | [0.442, 0.700] |
| eye-flow, motion-gated | 0.605 | [0.453, 0.751] |
| lid-contour, all windows | 0.883 | [0.786, 0.961] |
| **lid-contour, motion-gated** | **0.909** | **[0.812, 0.983]** |

The optical-flow channel is dead across three sessions. The lid-contour channel
— computed since M1 and never analysed until now — is the only thing in this
project that has cleared the gate.

## Why it cannot be believed yet

Four variants were examined *after* the data were seen. The motion gate was not
pre-specified, and its threshold was chosen by looking at the data. n was 45
versus 22 windows, from one sitting, on one subject. None of those choices are
reflected in the confidence interval.

Separately, the channel is **pose-sensitive**: lid displacement responded to
deliberate head turns at AUC 0.988, *more* strongly than to saccades at 0.883.
It looked clean only because every head-turn window exceeded the motion gate and
was discarded. On its own the channel measures "the eye region changed", not
"the eyes moved".

## Hypothesis

Lid-contour displacement, aggregated as one-second window maxima and restricted
to low-body-motion windows, discriminates deliberate closed-eye saccades from
closed-eye stillness in a fresh recording.

## Primary outcome and pass rule

- **Measure:** `lid_disp`, one-second window maxima.
- **Comparison:** `slow_saccades` + `fast_saccades` versus `eyes_closed_still`.
- **Restriction:** windows with mean body motion below the gate.
- **Pass rule:** **lower bound of the 95% bootstrap CI ≥ 0.80.**

The original M1 gate was a *point estimate* of 0.80. This is stricter, on
purpose: a post-hoc finding should clear a higher bar than a pre-planned one.
Loosening was never available; tightening guards against the small sample that
produced the original number. Bootstrap is 3000 draws over windows, seed 20260807.

## Motion gate, fixed in advance

**The 90th percentile of window body-motion in the `eyes_closed_still` segment.**

Derived from the control condition only, so it cannot encode anything about the
saccade-versus-baseline contrast, and fixed here so it cannot be selected once
the answer is visible. A test asserts the threshold is unchanged when the
positive segments change.

## Validity criteria

Evaluated **before** the primary outcome is read. Failure yields NO VERDICT, not
FAIL — a gate reading taken through an unverified instrument is not evidence
either way.

- **L1 — the gate removes large head movement.** ≥ 80% of `head_turn` windows
  must be excluded by the motion gate. If large head movement is quiet enough to
  survive, the gate is not doing its job.
- **L2 — small head movement is not mistaken for eye movement.** Among
  `micro_head_motion` windows that survive the gate, AUC versus baseline must be
  **below** the saccade AUC. Evaluated only if ≥ 10 such windows survive;
  otherwise L1 covers it.

`micro_head_motion` is a new segment. The original protocol did not have it, and
its absence is why the pose confound went unnoticed: large turns are trivially
removed by a motion gate, whereas the small movements a sleeper actually makes
are not.

## Minimum data

≥ 20 saccade windows and ≥ 15 baseline windows after gating, or the result is
INSUFFICIENT DATA. Baseline and saccade segments were lengthened from 30 s to
45 s to widen the margin.

## Secondary, reported but not decisive

- `eye_flow` on the same windows, for the record.
- Ungated lid AUC, and the lid head-turn AUC.

## What each outcome means

| Verdict | Meaning |
|---|---|
| **REPLICATED** | The signal survived on unseen data under frozen criteria. First real evidence in the project. Still a *waking* result: the sleeping case is harder in every respect and G9 stays locked pending reference validation. |
| **NOT REPLICATED** | Instrument valid, signal absent. The 0.909 was an artefact of examining four variants post-hoc on one small session. **The camera eye-tracking branch ends.** |
| **NO VERDICT** | Validity failed. The AUC is not evidence either way. |

## Stopping rule

**One recording.** No adjustment of the detector, the criteria, the gate, or the
protocol on the basis of what this session shows. If it does not replicate, that
is the answer — the criteria were frozen precisely so that the negative outcome
counts for something.

If a genuinely different hypothesis emerges later, it requires its own
pre-registration and its own data. Not an edit to this one.

## Deviations

Any change after the first confirmation recording must be recorded here with its
date and reason, and invalidates the result.

*(none)*

---

# Outcome — 2026-08-07, session 4

**VERDICT: REPLICATED.** No deviations from the plan above.

| | Session 3 (post-hoc) | Session 4 (confirmation) |
|---|---|---|
| Lid AUC, motion-gated | 0.909 | **0.931** |
| 95% CI | [0.812, 0.983] | **[0.878, 0.972]** |
| Windows | 45 v 22 | 60 v 39 |

Pass rule was CI lower bound ≥ 0.80. Observed lower bound **0.878**.

## Validity

- **L1 — gate removes large head movement.** 24 of 25 head-turn windows excluded
  (96%, required ≥ 80%). Median motion during head turns was 0.00143 against a
  gate of 0.00068, so deliberate turns are cleanly separated by body motion
  alone. **Pass.**
- **L2 — small head movement is not mistaken for eye movement.** Micro-motion
  AUC 0.783 against saccade AUC 0.931. **Pass.**

## Breakdown

| Segment | Windows | Survived gate | Median motion | Gated lid AUC |
|---|---|---|---|---|
| eyes_closed_still | 44 | 39 | 0.000529 | — |
| slow_saccades | 45 | 28 | 0.000647 | 0.909 |
| fast_saccades | 45 | 32 | 0.000614 | 0.950 |
| micro_head_motion | 45 | 21 | 0.000691 | 0.783 |
| head_turn | 25 | 1 | 0.001433 | — |

## The caveat that matters, recorded because it passed rather than failed

**Micro head motion scored 0.783.** It cleared L2 with a margin of 0.148, but in
absolute terms small pillow-settling movements separate from stillness
substantially on their own. A meaningful part of what this channel measures is
head-driven; eye movement simply produces more of it.

That margin was measured on a cooperative awake subject making the largest
deliberate saccades they could, against gentle voluntary head movement. Asleep,
the terms move in the wrong direction: eye movements are smaller and
involuntary, while head movements are not obviously smaller. **The 0.148 margin
is the number most likely to close during sleep**, and it should be the primary
thing watched in the M4 reference validation.

Noted also: the saccade segments carried ~22% more body motion than the baseline
(0.00065 vs 0.00053), and only 28–32 of 45 saccade windows survived the gate
against 39 of 44 baseline windows. Making large deliberate eye movements without
any head movement is difficult. The gate handles it, but it is why the analysis
is restricted to gated windows rather than run on everything.

## What this establishes, and what it does not

**Establishes:** a camera-derived measure of eyelid contour geometry
discriminates deliberate closed-eye saccades from stillness, at 0.931, under
criteria frozen before the data existed, with the pose confound explicitly
tested and excluded. This is the first result in the project that is evidence
rather than a hypothesis.

**Does not establish:** anything about sleep. This is awake, cooperative,
frontal, ~120 px interocular, well lit, with maximal voluntary saccades. Sleep
is worse on every one of those axes, and the side-sleeping posture question is
untested pending a bed-mounted camera.

**G9 remains locked.** Sensor-timed cueing still requires M4 validation against
a reference device, and nothing here changes that.

## Consequences

The M1 eye-movement branch is alive and the hardware spend is justified. The
optical-flow channel remains dead — it scored 0.958 this session but failed its
own V2 with a head-turn AUC of 0.997, so it is more head-contaminated than ever.
Phase B's oblique-illumination work was aimed at rescuing flow; that priority is
now wrong, and any further work should target lid geometry.
