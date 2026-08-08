# Stages — what to do, in order, and what it is up against

The milestone numbering in `design.md` describes *code*. This describes *what
you do and what you learn*, which is not the same list and is the one that
matters when the project stops feeling legible.

**The single most important framing:** the camera is not the product. The
product is a conditioning protocol plus an experiment that can tell you whether
it worked *for you*. Every lucid-dream device ever sold skipped the second half.

---

## Stage 0 — Baseline. Done.

**Result:** 9 lucid nights in 59, **1.07 LD/week**, 5.2 dreams recalled/night.

That is a high baseline — the Northwestern participants started at 0.74/week.
It cuts both ways: a fixed multiplicative effect is easier to detect off a
higher base, but you have less headroom than a novice.

**Caveat that matters later:** 1.07/week is *not* the control-arm rate.
Conditioning runs on every night including `no_cue`, deliberately, so the arms
differ in one variable. The control arm will sit above 1.07 and the real
contrast will be smaller than any comparison against this number suggests.

**Compared to:** nothing. No consumer product asks you to establish a baseline,
which is why no consumer product can tell you whether it did anything.

---

## Stage 1 — Shakedown. Now. 2–3 nights.

**What you do**

```
morpheus train           # 7.8 min, last thing before sleep
morpheus night --hours 8 # lid open, plugged in
morpheus journal         # morning, before reading anything
```

**How it works.** Nothing for 5.5 h. Then the gate stack opens and a cue fires
at the first legal second, and at every legal second after, bounded by a 20-min
cooldown and 2/hour. About four cues, ~5:00 / 5:20 / 6:00 / 6:20 on an 11:30pm
start. Each is a 9 s tone at gain 0.08, fading in over 4 s. No camera, no
sensing — clock only.

**What you are proving:** the chain survives a real night. Machine stays awake,
audio fires, cues do not wake you, the report gets written. Nothing here is
data.

**Gate:** two consecutive nights with no `DEFECTIVE NIGHT`, full duration, cues
delivered, and you did not wake up because of one.

**Compared to:** at this stage Morpheus does roughly what **Remee** does — a
timer firing cues at a fixed hour, no sensing. Remee is a $95 mask with LEDs and
has existed for over a decade. You are not ahead yet.

---

## Stage 2 — The trial. 120 nights, ~4 months. **This is the product.**

**What you do**

```
morpheus experiment ...   # two-arm, block randomized, 60 nights/arm
morpheus prereg           # frozen before the first night
# then nightly, unchanged:
morpheus train ; morpheus night --hours 8
# morning:
morpheus journal ; morpheus reveal
```

**How it works.** Each night is sealed into `trained_cue` or `no_cue` before it
runs, and the assignment cannot be read until the morning report exists — the
blinding is enforced in code, not by discipline. Training happens on **both**
arms, so the only variable is whether a sound plays. At the end, a beta-binomial
posterior over the two rates.

**Power at your measured baseline**, under the pre-registered rule
`P(trained_cue better) > 0.95`:

| nights/arm | total | 2.85× (Northwestern) | 2.0× | 1.5× |
|---|---|---|---|---|
| 30 | 60 | 86% | 38% | 18% |
| **60** | **120** | **96%** | **66%** | **31%** |
| 90 | 180 | 100% | 78% | 35% |
| 120 | 240 | 100% | 90% | 46% |

60/arm is the commitment. Smaller only detects the optimistic case; chasing 1.5×
costs a year.

**Gate (design.md §22, "M2"):** ≥95% of cues fire as specified, cue-attributed
awakening rate <15%, and **no downward sleep-quality trend over two weeks** —
that last one is a stopping rule, not a metric. It halts the study.

**Compared to:**

| | Sensing | Cue | Tells you if it worked for *you* |
|---|---|---|---|
| Remee | none | timed LEDs | no |
| Lucid-dream apps (Awoken etc.) | none | reminders, journal | no |
| Northwestern TLR 2024 | none | scheduled audio | group average only |
| **Morpheus at Stage 2** | none | scheduled audio | **yes — randomized, blinded, pre-registered** |

This is the whole differentiator and it arrives with **zero extra hardware.**
Morpheus at Stage 2 delivers the same intervention as the published trial, and
adds the thing the published trial cannot give you: an answer about you
specifically. No device on the market does this, and it is not because it is
technically hard.

---

## Stage 3 — Camera. Optional. Hardware-gated. $50–70.

**Do not start this until Stage 2 is running.** It does not block anything.

**What you do:** buy a USB IR camera with manual exposure over UVC and a 940 nm
illuminator, mount rigidly at ~60 cm, then run the sequence in
`posture-stage-plan.md` — R1 smoke test, per-posture positive control, 5–7
nights of posture distribution.

**How it works.** MediaPipe tracks the eyelid contour; a saccade under a closed
lid deforms the lid where the cornea bulges, and that deformation is the signal.
It replicated at AUC **0.931 [0.878, 0.972]** awake, at a desk, frontal, well
lit. Everything in Stage 3 asks whether that survives being asleep.

**Gates:** R1 mesh in ≥50% of IR frames; P1 eye usable ≥25% of night; P2 lid AUC
lower bound ≥0.80 in at least one posture; P3 effective coverage ≥25%.

**Honest odds:** R1 alone is about 60/40 — MediaPipe is trained on visible-light
colour faces and may simply not mesh under IR. The likeliest overall outcome is
**supine passes and side-lying fails**, because a camera cannot see through the
back of a head and you sleep on your side.

**Compared to: the Nova Dreamer.** This is the honest comparison and it is not
flattering. The Nova Dreamer (LaBerge / Lucidity Institute, 1990s) was a sleep
mask with an IR emitter/detector pair sitting *millimetres from the closed lid*,
detecting eye movement and firing red LEDs. It had real REM detection thirty
years ago.

**Morpheus will not beat it at sensing.** Detecting the same movement from 60 cm
away, through room air, at an oblique angle, is strictly harder than doing it
from 2 mm away — that is geometry, not engineering effort. What Morpheus offers
instead is *contactless* (nothing on your face, which matters enormously for a
side sleeper) and *validated* (the Nova Dreamer's detection accuracy was never
published against EEG in any form you can check).

If Stage 3 fails, the finding is real and useful: the signal exists but is
unavailable most of the night, which argues for moving the sensor onto the face
— a mask — rather than for a better camera. The software is sensor-agnostic, so
that pivot costs a driver, not a rewrite.

---

## Stage 4 — Reference validation. Optional. ~$400. Gated on Stage 3.

**What you do:** wear a Muse S for 5–10 nights alongside the camera, and score
the camera index against the headband's staging on **held-out nights**.

**Gate (design.md §22, "M3"):** AUC ≥ 0.70 vs reference. **Below 0.65 kills the
CV cueing branch permanently.**

**The acquisition risk is you, not the device.** You are a side sleeper. If you
cannot tolerate a headband, this stage is unachievable and the design doc
already commits to the consequence: the camera stays in shadow mode forever and
Stage 5 never unlocks. That is an accepted outcome, not a failure — an
unvalidated detector driving cues is strictly worse than a clock.

**Compared to:** research rigs like **ZMax** (~€1500, research-grade EEG with
auditory stimulation). At Stage 4 Morpheus is doing what a sleep lab does, with
a $400 headband and a $60 camera, for one subject. Less accurate; vastly
cheaper; and unlike a lab, it runs for a year.

---

## Stage 5 — Sensor-timed cueing. Gated on Stage 4.

**What you do:** promote gate G9 (which already exists in the stack, permanently
inactive) and run scheduled-vs-sensor-timed as a second A/B.

**Gate:** P(sensor > scheduled) ≥ 0.90.

**Compared to: the Nova Dreamer, again — and this is where you could pass it.**
Not on detection quality, but on the loop being closed *and measured*. Nova
Dreamer detected REM and flashed. Nobody ever demonstrated that its timing beat
a clock. Stage 5 is a direct test of exactly that question, which as far as the
published record goes has never been answered for any consumer device.

**Realistic probability of reaching Stage 5:** low. It requires R1 to pass
(~60%), then P2 in a posture you actually sleep in (uncertain), then headband
tolerance (uncertain), then AUC ≥ 0.70 (uncertain). Multiply those and it is
maybe one in five. Plan accordingly — Stage 2 has to be worth doing on its own,
and it is.

---

## The one-line version

| Stage | Duration | Cost | Outcome |
|---|---|---|---|
| 0 Baseline | done | — | 1.07 LD/week |
| 1 Shakedown | 2–3 nights | — | the chain works |
| **2 Trial** | **4 months** | **$0** | **does cueing work for you** |
| 3 Camera | ~2 weeks | $60 | can a camera see your eyes asleep |
| 4 Reference | ~2 weeks | $400 | does the camera track REM |
| 5 Sensor timing | 4 months | — | does sensing beat a clock |

**Stages 3–5 are optional and probably will not all land. Stage 2 is the
project.** If you did Stage 2 and stopped, you would have something no product
on the market has: evidence about yourself.
