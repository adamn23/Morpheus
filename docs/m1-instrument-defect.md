# M1 instrument defect, and the criteria for re-running the gate

**Date:** 2026-08-06
**Status:** defect confirmed, fix pending re-test

## What happened

The first waking calibration returned:

| Measure | Value |
|---|---|
| Positive control AUC (saccades vs stillness) | **0.671** (threshold 0.80) |
| Head-turn AUC | **0.955** |
| Baseline flow median (eyes closed, still) | 0.2466 |
| Head turn / baseline | 5.2× |
| Saccade / baseline | 1.6× |

Verdict: FAIL, with a leakage warning.

## Why this is not a valid negative result

A test is only evidence about eye movement if the instrument measures eye
movement. Two facts say it did not:

1. **Head turns separated from baseline (0.955) far better than eye movements
   did (0.671).** The index responded ~5× more strongly to head motion than to
   the thing it exists to detect.
2. **The baseline was not a noise floor.** "Eyes closed, completely still"
   returned 0.2466, against 0.0004 for a static synthetic scene — 600× higher.
   A still face is not producing 600× the flow of a still image.

So this run measured head motion and sensor noise. It says nothing yet about
whether eye movement is detectable.

## The defect

`EyeFlowExtractor` reported `mean(|v|)` — the mean of per-pixel flow
*magnitudes*. Under that metric sensor noise adds constructively: every pixel
receives a random flow vector, and magnitudes never cancel.

Measured on a static synthetic scene with realistic sensor noise, where the true
motion is exactly zero:

| Sensor noise σ | Reported flow (`mean(|v|)`) |
|---|---|
| 0.0 | 0.0004 |
| 2.0 | 0.1102 |
| 4.0 | 0.2206 |

σ≈2–4 reproduces the observed 0.2466 baseline almost exactly. CLAHE roughly
doubled it again.

The correct measure is the **coherent** component, `|mean(v)|` — the magnitude
of the mean flow vector. Incoherent noise cancels; coherent motion survives.
Genuine conjugate eye movement is coherent by definition.

| Metric | Signal-to-noise (5 px eye movement, σ=2) |
|---|---|
| `mean(|v|)` (shipped) | 31× |
| `|mean(v)|` (fix) | **603×** |

This defect is demonstrable **independently of the calibration outcome**: it is
wrong to report 0.11 of flow on a scene containing zero motion, whatever the AUC
happens to be.

## Pre-committed criteria for the re-run

Declared before the fix is written, so the outcome cannot influence them.

**The pass threshold does not move. It stays at AUC ≥ 0.80.**

The re-run only counts as a valid test of H1 if the instrument first clears both
of these:

- **V1 — the noise floor is genuinely noise.** Baseline (eyes closed, still)
  median **coherence < 0.35**.

  > **Revised 2026-08-06, before the re-run.** V1 was first written as "flow
  > median < 0.05". That was an absolute on a scale with no units, chosen before
  > the fixed instrument had been measured, and it turns out to move with sensor
  > noise: the fixed extractor floors at 0.022 at sigma=2 but 0.066 at sigma=8,
  > so the same working instrument would pass or fail depending on the camera.
  >
  > Coherence is scale-free and separates cleanly — measured at 0.18-0.24 for
  > pure noise and 0.71-0.99 for real movement — so it tests the thing V1 was
  > reaching for. The revision is recorded here because it happened before any
  > re-run data existed. **The pass threshold (AUC >= 0.80) is untouched**, and
  > will not be revised at any point.

  Low baseline coherence means the still segment is noise, which is what a floor
  should be. High baseline coherence means real coherent movement is leaking
  into the "still" condition and contaminating the comparison.
- **V2 — registration.** Head-turn AUC must be **strictly below** the saccade
  AUC. If head motion still separates as well as eye motion, registration is not
  working on real footage regardless of what the synthetic tests show.

### Decision table

| V1 | V2 | Saccade AUC | Conclusion |
|---|---|---|---|
| pass | pass | ≥ 0.80 | H1 gate passed. Proceed to M1 shadow-mode nights. |
| pass | pass | < 0.80 | **Genuine FAIL.** Eye branch ends. Camera stays a motion guard. |
| fail | any | any | No verdict. Instrument still broken. |
| any | fail | any | No verdict. Measuring head motion, not eye motion. |

**One re-run.** If it fails on V1 or V2 again, that is not an invitation to keep
adjusting until something passes — it is evidence the approach cannot be made to
work with this hardware, which is itself the answer.

## Separately: the posture numbers from this run are void

All four postures reported 98–99% eye availability. That is not credible for a
side sleeper and is an artefact of running the whole protocol at a desk: "lie on
your left side" was performed in front of a laptop, not in bed.

The protocol conflated two questions needing two camera positions. Posture
segments only mean anything from the camera's real overnight mount. They are now
split into a separate stage and must be re-run with the IR camera in place.

---

# Outcome: the re-run, 2026-08-07

| Run | Baseline flow | Saccade AUC | Head-turn AUC |
|---|---|---|---|
| 1 (broken instrument) | 0.2466 | 0.671 | 0.955 |
| 2 (after the fix) | 0.2692 | **0.494** | 0.817 |

Against the criteria fixed above:

- **V1 — unmeasurable.** The coherence metric was added to `EyeFlowSample` but
  never plumbed into the calibration profile, so the criterion that depends on
  it could not be evaluated. An oversight, recorded rather than glossed.
- **V2 — failed.** Head-turn AUC (0.817) remains well above saccade AUC (0.494).
  The index still separates head motion far better than eye motion.
- **Gate — 0.494.** Chance.

## Why this reads as a genuine negative rather than more instrument trouble

**The baseline did not move.** The fix cut the synthetic noise floor five- to
tenfold; on real footage it changed 0.2466 to 0.2692, which is nothing.

That is the informative result. It means the floor was never sensor noise. It is
coherent motion in the eye region during deliberate stillness — micro head
movement, pulse, landmark jitter — and no amount of denoising removes something
that is not noise.

This matches the physics the design started from. Gross head movement is a large,
coherent, easily resolved signal, and it is still detected at 0.817.
Sub-millimetre lid-surface deformation from a cornea moving beneath a closed lid
is not resolvable at ~50 cm through a silicon sensor. The camera sees the head
and not the eyes.

The saccade AUC also *fell*, 0.671 to 0.494. The higher figure in run 1 was head
motion leaking into the measure; removing the leak removed the apparent signal
with it. Two independent runs now agree, and they agree with the physics.

## Decision

**H1 is not supported. Camera-based eye tracking ends here.**

Per the one-re-run rule fixed above, this is not an invitation to keep adjusting.
A further hypothesis does exist — landmark jitter below the 0.3 px no-warp
threshold, uncorrected, presenting as coherent flow — and it was deliberately not
pursued. It would have been the third adjustment after two failures against a
signal measuring at chance, which is the precise behaviour the pre-commitment
was written to prevent.

## What this does and does not invalidate

Ends: eye-movement detection from a camera. G9 stays permanently locked; the
lock in `cue/sensor_timing.py` needs no change, having done its job.

Survives, untouched: the conditioning protocol, the cue engine and its safety
supervisor, the morning report, the blinded N-of-1 harness, the adaptive layer,
and the reference-validation machinery, which is sensor-agnostic and will accept
EEG or a contact sensor without modification. The camera remains useful as a
gross-motion arousal guard, which is what design.md §8 recommended it for before
any of this was measured.

The cost of finding out was two calibration sessions of about three minutes each.
That was the point of putting the gate first.
