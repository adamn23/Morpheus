# Posture stage — plan

**Status:** planned 2026-08-07, not yet started. Blocked on hardware.

## Context

Lid-contour geometry replicated at AUC 0.931 [0.878, 0.972] under frozen
criteria (`docs/lid-geometry-prereg.md`). That is a *waking* result: awake,
cooperative, frontal, ~120 px interocular, well lit, maximal voluntary saccades.

The posture stage asks the two questions that stand between that and a usable
overnight sensor:

1. **Is the eye region visible at all** when lying in the postures actually
   slept in, from a camera that has to stay by the bed?
2. **Does the discrimination survive** the move to bed distance, an oblique
   angle, and infrared illumination?

Both must hold. A perfect signal visible 5% of the night cannot gate cues, and
80% visibility with no signal is equally useless.

---

## The two risks that decide this

### R1 — MediaPipe may not mesh an infrared face. Highest risk, cheapest test.

The entire live channel is MediaPipe's tracked eyelid contour. `landmarks.py`
converts a monochrome frame to grey RGB before handing it over, and MediaPipe is
trained on visible-light colour faces. Infrared skin reflectance looks
substantially different: veins and subsurface structure that are invisible in
colour become prominent, and all chrominance is gone.

**If no mesh appears under IR, `lid_disp` is never populated and the branch is
over regardless of everything else.** This is untestable without the hardware —
the built-in FaceTime camera has an IR-cut filter — so it becomes step 2, a
two-minute check before any protocol is run.

### R2 — geometry degrades with distance

Interocular pixels scale inversely with distance, and lid contour measures
sub-pixel change, so px/mm is the figure that matters rather than the 30 px
presence floor.

| Setup | Distance | Interocular | px/mm |
|---|---|---|---|
| Desk (session 4, measured) | 50 cm | 120 px | 1.9 |
| Bedside table, close | 60 cm | 100 px | 1.6 |
| Bedside table, typical | 80 cm | 75 px | 1.2 |
| Headboard / further | 110 cm | 55 px | 0.9 |

**Mount as close as tolerable.** 60 cm keeps ~85% of the resolution the
replication was measured at; 110 cm keeps under half. A narrower-FOV lens buys
the same thing without moving the camera nearer your head.

---

## Hardware

| Item | Spec | Note |
|---|---|---|
| USB IR camera | **manual exposure over UVC** | Verify in the spec sheet before ordering. Non-negotiable — `doctor` refuses to run without it, for good reason. |
| IR illuminator | **940 nm** | See below. |
| Mount | rigid, ~60 cm, aimed at the pillow | Rigidity matters more than the camera. Any nudge invalidates a night. |

### 850 nm versus 940 nm

850 nm gives roughly twice the silicon sensor response but emits a **faint
visible red glow**. 940 nm is invisible but dimmer.

**Recommendation: 940 nm.** A visible red source pointed at the head all night
is a sleep disturbance and an uncontrolled confound across conditions, and this
project cannot afford to degrade the sleep it is measuring. The sensitivity loss
is recoverable through exposure and gain, which are under manual control anyway.

If 940 nm proves too dim after step 1, 850 nm is the fallback — but then it must
be mounted oblique and low, never in the line of sight.

**Estimated cost: $50–70.**

---

## Sequence

### Step 1 — mount and verify (~10 min)

```
morpheus doctor
```

No `--allow-auto-exposure`. The calibration runs so far have all had AE enabled,
which is a known confound; from here on manual exposure is required. `doctor`
also reports the live quality distribution, which is what catches a mis-tuned
focus floor before it produces a zero-coverage night that looks like a finding.

### Step 2 — the R1 smoke test (~2 min)

Lie in position, run a short recording, and check whether **`landmark_available`
is non-zero** and `lid_disp` is being populated. If MediaPipe produces no mesh
under IR, stop here: nothing downstream can work, and no amount of protocol
fixes it.

*Needs a small tool — `morpheus doctor --landmarks` or equivalent — reporting
mesh availability and median interocular live. Not yet built.*

### Step 3 — posture positive control (~8 min)

**This is the design change that matters.** The existing BED segments only
measure *visibility*. That is necessary but not sufficient — it answers "can the
camera see your eye" and not "does the signal survive". So the positive control
is run **inside each posture**: lie as you sleep, and do the deliberate saccades
there.

Per posture — supine, left side, right side, prone:

| Segment | Duration |
|---|---|
| eyes closed, still (baseline) | 45 s |
| deliberate saccades | 45 s |

Plus once, in whichever posture scores best:

| Segment | Duration | Purpose |
|---|---|---|
| micro head motion | 45 s | L2 confound, re-measured in bed |
| head turn | 25 s | L1 gate efficacy |

The L2 re-measurement is important. Awake, small head movement scored 0.783
against saccades at 0.931 — a 0.148 margin. In bed the confound is more natural
and the margin may be narrower.

Total ~8 minutes. Requires new BED-stage segments; the current protocol has one
baseline-and-saccade pair, not one per posture.

### Step 4 — posture distribution (5–7 nights)

```
morpheus record --hours 8
```

Steps 1–3 measure signal *per posture*. This measures **how much of the night
you actually spend in each**, from `roll_deg` and the coverage flags. Without
it, per-posture numbers cannot be combined into anything meaningful.

### Step 5 — combine

**Expected effective coverage** = Σ over postures of
`P(time in posture) × coverage in posture × signal survives in posture`

---

## Pre-committed gates

To be frozen in a pre-registration before step 3, in the same form as
`docs/lid-geometry-prereg.md`.

| Gate | Requirement | Failure means |
|---|---|---|
| **R1** | mesh available in ≥ 50% of IR frames | Branch ends. No lid contour, no channel. |
| **P1** | eye region usable ≥ 25% of night, weighted by posture time | Camera cannot gate cues often enough to matter. |
| **P2** | lid AUC CI lower bound ≥ 0.80 in ≥ 1 posture | Signal did not survive the move to bed. |
| **P3** | expected effective coverage ≥ 25% | Below 15%, fall back to scheduled cueing plus the motion guard. |

P1 and P3 reuse the M0 thresholds fixed in `docs/design.md` §28. P2 reuses the
confirmation rule. **No threshold is new, and none moves.**

A plausible and important outcome is P2 passing in supine only. That would be a
real finding for a side sleeper: the signal exists but is unavailable most of the
night, which argues for the sensor moving to the face — a mask — rather than for
more camera work.

---

## Work required before step 3

1. Per-posture baseline/saccade segments in `calibration/protocol.py`.
2. A posture-aware confirmation analysis, structured like
   `calibration/confirmation.py`, evaluating each posture separately.
3. `morpheus doctor --landmarks` for the R1 smoke test.
4. Posture-time extraction from overnight `frames_1hz` for step 5.
5. The pre-registration, committed before any posture data exists.

Roughly a day. **None of it should be built until the hardware is in hand and R1
has passed** — if MediaPipe cannot mesh an IR face, all of it is wasted.

---

## Honest expectation

R1 is genuinely uncertain and I would not put it above 60/40. R2 is manageable
if the camera is mounted close.

The likeliest outcome remains that supine works and side-lying does not, given
that a camera cannot see through the back of a head. For someone who sleeps on
their side, that points at the mask rather than at a better camera — and the
software is already sensor-agnostic, so that pivot costs a driver rather than a
rewrite.

Meanwhile none of this blocks the protocol. The conditioning and scheduled
cueing remain the only components with published efficacy behind them, and they
still have zero nights collected.
