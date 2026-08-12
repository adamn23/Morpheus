# Morpheus — research plan

**2026-08-08.** Supersedes the brainstorming documents. One system, one build
order. Labels: **[E]** established, **[X]** extrapolation, **[S]** speculative.

---

## 1. What Morpheus is

> **Morpheus is a single-subject research instrument that does not assume the
> physiological target for lucidity is known.** It first raises the rate of
> lucid dreams by behavioural means and verifies them physiologically rather
> than by self-report; it then learns, from many nights of one nervous system,
> what reliably *precedes* those verified events; it then runs randomised
> perturbation experiments to find which of a small set of safe actuators
> causally move the brain along that pathway; and only then does it attempt
> closed-loop control. Each stage is falsifiable on its own and cheap enough to
> abandon. The output is not a device that makes you lucid — it is the first
> attempt to climb from *correlate* to *controllable cause* inside one brain.

### The loop

```
sleep physiology
   → state estimation           (REM? microstate? arousal depth?)
   → intervention opportunity   (is this a moment worth spending?)
   → intervention               (or deliberate no-op, randomised)
   → physiological response     (did the state move? did it arouse?)
   → LRLR-verified outcome      (physiology, not memory)
   → model update               (precursor model + actuator model)
   → improved intervention policy
```

### Why this is not NovaDreamer/TLR

Those systems are: **REM detected → cue → hope.** Three properties distinguish
this one, and all three are structural rather than incremental.

| | Conventional | Morpheus |
|---|---|---|
| Outcome | self-report, next morning | **LRLR, timestamped to the second** |
| Response to a cue | unobserved | measured within seconds, and used |
| Target | assumed (REM, or 40 Hz) | **discovered empirically per person** |
| Learning | none | across nights, over a year |
| Failure mode | invisible | flagged as a defective night |

The deepest difference: conventional systems **assume they know what to aim
at.** The one physiological target the field agreed on — frontal 40 Hz — does
not survive artifact scrutiny (§9). Morpheus treats the target as unknown, which
is the only honest position available.

---

## 2. Sensors — and what to leave out

| Sensor | Contributes | V1? | Later question |
|---|---|---|---|
| **EEG AF7/AF8** | arousal depth, staging, EOG via `AF7−AF8` | **yes** | everything |
| **Accelerometer** | movement veto, posture, artifact rejection | **yes** | transition control |
| **Dedicated EOG** (outer canthi, aux) | clean LRLR and REM density | no — V2 | precursor: REM density |
| **Chin EMG** (aux) | muscle atonia; separates REM from wake | no — V2 | arousal depth, staging accuracy |
| **ECG** (chest strap) | HR, HRV, R-peak timing | no — V4 | autonomic precursor |
| ~~Respiration belt~~ | — | **no** | **derivable** from ECG and accelerometer; a separate belt is one more thing to dislodge |
| ~~SpO2~~ | — | **no** | no research question in this plan needs it |
| ~~Temperature~~ | — | **no** | same |
| ~~fNIRS~~ | — | **no** | 4–6 s haemodynamic lag; useless for timing, no LD literature |

**Ruthlessness rule:** a sensor enters only when a *named* experiment requires
it. Three of the above are excluded permanently on that test.

---

## 3. A night, with decision logic

```
BEDTIME
  conditioning protocol; instrument donned; contact check
  if contact quality < threshold on any required channel → warn, log, continue

HOURS 0 – 5.5   PASSIVE
  record everything. No intervention of any kind.
  purpose: precursor-model training data, and the night's own baselines
           (REM saccade amplitude median, arousal-index distribution)

HOUR ~5.5       WBTB
  wait for detected REM offset within [5.0, 6.5] h; if none by 6.5 h, alarm anyway
  alarm (never the trained cue) → induction script → back to bed
  detect sleep onset; start the post-WBTB clock from there, not from the alarm

HOURS 6+        ACTIVE. Each second, in priority order:

  1. if LRLR detected (§5)
       → STOP all stimulation for 10 min
       → timestamp, log, mark night as verified-positive
       → play a single soft confirmation tone (once, quietly)
       → resume only if a further 10 min of REM occurs with no second LRLR

  2. if movement or EMG rise above veto threshold
       → suppress everything, 3 min refractory

  3. if arousal depth in TRANSITION BAND (§4)
       → hand control to the Transition Engineering servo

  4. if stable tonic REM and cue budget remains and cooldown expired
       → randomised assignment:
            • intervention  (whichever actuator is under test)
            • matched sham  (identical except for the active component)
            • deliberate no-op
       → observe 1–3 s response window, classify, log

  5. if stable REM and a system-identification slot is scheduled (V5+)
       → apply a randomised perturbation purely to measure Δx
       → this is not an induction attempt and is logged separately

  6. otherwise → observe

MORNING
  dream report BEFORE reading any output
  self-report lucidity recorded, but kept in a separate column from LRLR
  reveal experimental arm; nightly model update
```

**The ordering is the safety design.** LRLR beats everything; the movement veto
beats every intervention; interventions only run in the lowest-priority branch.

---

## 4. Transition Engineering — the algorithm

### Arousal depth

A scalar, recomputed at 4 Hz, each component z-scored against **that night's own
REM baseline**:

```
A(t) =  w1 · (alpha+beta power) / (delta+theta power)     frontal
      + w2 · chin EMG RMS                                  [V2+]
      + w3 · heart-rate acceleration                       [V4+]
      + w4 · movement energy
```

Initial weights equal; refit once labelled arousals exist.

### State classification

| State | Criteria |
|---|---|
| **Stable REM** | A < −0.5, sustained 30 s |
| **Cortical arousal** | abrupt A rise ≥ 3 s, **no EMG rise, no movement** |
| **Full awakening** | A > +1.5 sustained ≥ 15 s, *or* any movement, *or* sustained EMG tone |
| **Re-entry** | A falling for ≥ 60 s, REMs returning, EMG low |

The distinction that matters is **cortical arousal versus full awakening**, and
it is carried mostly by EMG and movement, not by EEG.

### The servo

Once a spontaneous cortical arousal is detected in or adjacent to REM:

```
TARGET BAND: A ∈ [0.0, +1.0]        "metacognition without motor activation"

while in band and t < 90 s:
    if A < 0.0    → raise stimulus intensity one step   (rate-limited)
    if A > +1.0   → cut stimulus to zero immediately
    if movement OR EMG rise → ABORT, 3 min refractory, log
release at 90 s or on exit; then silence and let REM pressure act
```

**Morpheus does not need to cause the arousal.** With ~5 spontaneous arousals a
night it can wait for one and then hold it — which is both safer and better
evidence, since the arousal is not confounded with the intervention that would
have produced it.

### The simplest version I can test first — and it needs no closed loop at all

**Do not build the servo first.** Build this:

> **20 nights, randomised:** on treatment nights an alarm fires once at a fixed
> late-night time. Wake, **do not move**, hold awareness for ~90 s, return to
> sleep. On control nights, undisturbed. Compare lucidity rate.

Cost: an alarm. If a fixed-time crude awakening does not beat undisturbed
nights, the premise behind the entire servo is wrong and no amount of control
engineering rescues it. **This is experiment E1 and it comes before any
hardware.**

---

## 5. Verified lucidity — the LRLR channel

Everything downstream depends on this. Self-report never substitutes for it when
training the precursor model.

### Signal

`AF7 − AF8` in V1; dedicated canthi EOG from V2. Bandpass **0.5–35 Hz** —
wide enough to preserve saccadic edges for template matching.

### The agreed signal

**Four alternating deflections** (L-R-L-R), deliberate and large, each 0.3–1.0 s,
whole sequence 1.5–4.0 s. Performed **twice, separated by ~10 s** on becoming
lucid — the repetition is the single strongest false-positive defence, because
spontaneous REM bursts rarely repeat a specific pattern on a specific delay.

### Detection

```
1. normalised cross-correlation against a subject-specific template
2. amplitude must exceed 3 × the night's own REM saccade median
3. reject if coincident with: accelerometer motion, EMG spike,
   or large common-mode deflection in (AF7+AF8)/2   [blink]
4. HIGH confidence  = two matching sequences within 30 s
   LOW confidence   = one sequence
5. only HIGH confidence counts as verified lucidity for model training
```

### Calibration and false-positive testing

**Waking calibration:** ≥30 rehearsals, eyes closed, across several head
positions, in a `morpheus calibrate` session — reusing the existing
positive-control machinery.

**Threshold selection:** from the precision-recall curve, not by hand.

**Overnight specificity:** run the detector over whole nights where no LRLR was
attempted. **Target: < 1 false positive per 10 nights.** A false "verified
lucid" is worse than a missed one — it poisons the label set that everything
downstream is trained on.

### Three-stage honesty about what is established

| Stage | Establishes | When |
|---|---|---|
| 1 | detects deliberate LRLR **awake** | immediately |
| 2 | overnight false-positive rate | a few nights |
| 3 | sensitivity to LRLR **during REM** | only once lucid dreams occur |

Stage 3 does not follow from stage 1 — amplitude, contact and head position all
differ in REM. Bootstrap it from nights where lucidity is self-reported *and*
signalling remembered.

---

## 6. Precursor discovery with scarce labels

At ~1.07 LD/week and ~65% signalling success, **300 nights ≈ 30 verified
events.** That supports **5–10 effective parameters**, not a neural network.
Design accordingly.

### Interpretable features only

REM density · tonic/phasic state · EEG band powers · spectral slope · autonomic
activation · HRV · respiration (derived) · chin EMG · proximity to last
micro-arousal · time within REM period · REM period number · time since WBTB.

~12 candidates, pre-registered, **with a pre-specified reduction to the top 5**
by univariate screening on a held-out subset.

### The design that makes 30 events usable

**Within-night matched controls.** For each verified event, draw control windows
**from the same night**, matched on REM period number and time-in-period. This
removes night-level confounds — sleep debt, alcohol, illness, room temperature —
which would otherwise dominate a 30-sample analysis. It is the single most
important methodological choice here.

**Leave-one-night-out** cross-validation throughout. Never random splits: epochs
within a night are heavily autocorrelated and random splits leak.

**Anomaly framing.** Ask "is the pre-lucid window an outlier in this person's own
REM distribution?" — a one-class problem that needs only positives.

**Pre-registered sealed holdout.** With 30 events and many possible analyses, the
garden of forking paths is a larger threat than any modelling choice.

### Correlate versus precursor — the decisive test

Compute the discriminability of lucid versus matched-control windows at
**−5 s, −30 s, −120 s** and plot it against horizon.

```
AUC
 |     ______                          smooth decay  → PRECURSOR
 |    /      \___                      the signal builds before the event
 |___/           \____
 +--------------------- horizon
 -120  -30  -5   0

AUC
 |                  |                  flat, then a jump at zero → CORRELATE
 |__________________|                  you are detecting lucidity in progress
 +--------------------- horizon
```

**This one curve separates level 1 from level 2**, and it is computable from
passively recorded nights with no intervention at all.

---

## 7. From "X predicts lucidity" to "X causes lucidity"

Four different claims, routinely conflated:

| Level | Claim | Requires |
|---|---|---|
| 1 Correlate | X differs in lucid REM | observation |
| 2 Precursor | X precedes lucidity | temporal ordering (§6) |
| 3 Causal contributor | manipulating X changes P(lucid) | **randomised intervention** |
| 4 Controllable causal contributor | we have an actuator that moves X safely | **system identification** |

### Step A — characterise the knobs before using them

**Do not assume any actuator works.** For each candidate, first measure
`u → Δx` with no lucidity outcome involved:

| Actuator | Status | First question |
|---|---|---|
| Auditory CLAS, phase-targeted | **[E]** modulates REM alpha/theta, both microstates, n=18 | does alpha-trough move *this person's* frontal rhythms vs alpha-rising as an acoustically identical control? |
| Conventional auditory cue | **[E]** | dose-response for incorporation vs arousal |
| Thermal (mild skin cooling) | **[X]** | does it shift arousal depth without waking? |
| Transition manipulation | see §4 | does the servo hold the band? |
| Vestibular | see §8 | **actuator not yet available** |

During stable REM, apply randomised perturbations in a fixed fraction of
windows. Each has a matched sham. Fit a local linear model **Δx = B·u**. The
perturbations are self-generated, so unlike the labels **this data accumulates
fast** — thousands of trials in months.

### Step B — the causal test

Once a precursor X is identified and an actuator that moves X is characterised:

> **Randomised, sham-controlled, within-night:** in matched REM windows, apply
> either the actuator (which moves X) or its sham (which does not). Outcome:
> LRLR-verified lucidity.
>
> If P(lucid | actuator) > P(lucid | sham), **X is a causal contributor** —
> level 3. If the actuator moves X but not lucidity, X is a precursor without
> causal force, and the pathway is a passenger, not a driver.

That second outcome is the likeliest one and it must be designed for, not
treated as failure. It is also the result almost no published study is capable
of producing.

---

## 8. Self-model destabilisation — retracted actuator, surviving hypothesis

### The hypothesis remains interesting

Cues attack the **world model**, which dreams rationalise effortlessly — that is
their default operation. The **self model** (self-location, embodiment,
perspective, agency) may be load-bearing in a way scene content is not. Nobody
has attacked it. Grounding is solid **[E]**: vestibular signals are constitutive
of bodily self-consciousness; out-of-body experience is a three-way
disembodiment tied to vestibular dysfunction; visual-vestibular stimulation
induces OBE-like illusions.

### But I was wrong about the hardware

I claimed a ~$20 bone-conduction transducer would do. **It will not.** Clinical
audiometric bone vibrators (B-71) are **already too weak at ≤500 Hz to elicit
VEMP reliably**; consistent vestibular activation needs a **B&K 4810 Minishaker
plus a power amplifier** — laboratory equipment, thousands of dollars. Consumer
bone-conduction audio transducers are designed for the speech band at
milliwatt levels and are orders of magnitude below vestibular threshold.

**Consequence: D has no cheap actuator.** GVS would work but is electrical and
excluded. Caloric stimulation is awkward and dizzying. Whole-body rocking is
mechanical and plausible but bulky and may itself wake the sleeper.

### What would have to be true, and how to test it

1. **Calibration:** any candidate transducer must be characterised with an
   accelerometer at the mastoid, in dB force level, against published VEMP
   thresholds. Without that measurement, "we applied vestibular stimulation" is
   an assumption.
2. **Awake demonstration:** the actuator must produce a measurable
   self-location or verticality distortion awake, eyes closed — e.g. a shift in
   subjective visual vertical, or the embodied/disembodied letter-tracing task
   from the GVS literature. **If it cannot perturb the self-model awake, it
   will not asleep.**
3. **REM falsification:** with a calibrated actuator, randomised sham-controlled
   REM stimulation; outcome LRLR-verified lucidity plus a self-model
   questionnaire in the dream report. Null result kills the hypothesis.

**Status: deferred to V5+, contingent on a calibrated actuator existing.** It is
the most interesting untested mechanism in this document and the least
accessible.

---

## 9. Build roadmap

### V1 — Behavioural, no new hardware

**Buy:** nothing required. Optionally sleep earbuds (~$100) for quieter,
consistent, partner-safe cue delivery.
**Build:** already built — WBTB, SSILD, gain titration, counts, phase switching.
**Experiment E1:** 40 nights randomised — crude transition engineering (fixed-time
still awakening) vs undisturbed. Plus gain titration to a level with
incorporation and no arousal.
**Criterion:** lucid rate on treatment nights exceeds control, or gain titration
finds a working level.
**If it fails:** transition engineering is dead as a premise; the servo is never
built; fall back to optimising scheduled cueing and reconsider whether the
programme is worth hardware at all.

### V2 — Verified outcomes

**Buy:** EEG headband. **Build:** `MuseSource`, contact-quality guard, clock
sync + one end-to-end latency calibration, LRLR detector.
**Experiment E2:** waking LRLR calibration; overnight false-positive rate.
**Criterion:** **< 1 false positive per 10 nights**, and LRLR detected on ≥1
self-reported lucid night.
**If it fails:** without verified events there are no labels; V4–V6 are
unreachable. The programme reduces to behavioural optimisation with self-report,
which is still worth running but is no longer novel.

### V3 — Transition Engineering, instrumented

**Build:** arousal-depth estimator, four-state classifier, the servo, movement veto.
**Experiment E3:** servo-held transitions vs matched spontaneous arousals left alone.
**Criterion:** the servo holds the band for ≥60 s without waking on ≥70% of
attempts, and lucid rate exceeds the un-serviced arm.
**If it fails:** keep crude V1 awakenings if those worked; abandon the servo.

### V4 — Precursor discovery

**Buy:** chest-strap ECG (~$90) for autonomic features. **Build:** feature
pipeline, within-night matched controls, AUC-vs-horizon analysis.
**Experiment E4:** the precursor curve (§6), on ≥20 verified events.
**Criterion:** smooth AUC decay with horizon, surviving leave-one-night-out on
sealed holdout nights.
**If it fails:** flat-then-jump means correlate-not-precursor. Steering has no
target. Publish the negative — it is a real and useful result — and stop at V3.

### V5 — System identification

**Build:** randomised perturbation scheduler; `Δx = B·u` estimation.
**Experiment E5:** characterise CLAS, conventional audio, thermal. Add vestibular
only if a calibrated actuator exists (§8).
**Criterion:** at least one actuator reliably moves the precursor feature X,
sham-controlled.
**If it fails:** X is real but not reachable with safe non-invasive actuators.
That is a publishable finding and an honest stopping point.

### V6 — Closed-loop steering

**Build:** model-predictive controller under arousal constraints.
**Experiment E6:** steering vs sham; outcome LRLR-verified lucidity.
**Criterion:** P(lucid | steering) > P(lucid | sham).
**If it fails:** the pathway was a passenger, not a driver.

**Note the shape:** V1 and V2 cost about $200 between them and either can end the
programme. The expensive stages are reached only by surviving the cheap ones.

---

## 10. The six questions

**1. What is genuinely novel?** The *ladder itself* — a design that can climb
from correlate to controllable cause in one nervous system. Specifically:
verified-outcome labelling at scale in a home setting; within-night matched
controls for a rare event; **system identification of actuator→state effects in
a sleeping brain**; and arousal depth as a servo-controlled variable. None of
these exist in this field.

**2. What is already established?** REM physiology and microstates; that tonic
and phasic REM differ oppositely in arousal threshold and auditory transmission;
that scheduled TLR raises lucidity at group level; that WBTB+MILD works; that
LRLR eye-signalling marks lucidity; that phase-targeted CLAS modulates REM
oscillations; that vestibular input is constitutive of bodily self-consciousness.

**3. What is speculative?** That a stable individual precursor exists at all;
that any safe non-invasive actuator can move it; that transition-held arousal
preserves metacognition into re-entry; that self-model perturbation triggers
lucidity. All four could be false independently.

**4. First real breakthrough?** **A smooth AUC-versus-horizon decay curve on
≥20 LRLR-verified events, surviving leave-one-night-out on sealed holdout
nights.** That would be the first demonstrated *precursor* — not correlate — of
lucidity in an individual, and it is the gate to everything after it.

**5. What would convince me it is failing?** Any of: LRLR false-positive rate
that cannot be driven below 1-in-10 nights; a flat AUC-versus-horizon curve
after 30 events; no actuator moving any candidate feature after V5; or — most
mundanely and most likely — **adherence collapsing before 100 nights.** The
programme's true failure mode is a person stopping, not a hypothesis being
wrong.

**6. What would a single REM period eventually look like?**

```
t+0:00   REM onset detected. Passive. Baselines established.
t+0:30   microstate: tonic. Precursor score 0.2 — low.
t+1:10   randomised system-ID slot: CLAS burst, alpha-trough, 6 s.
         Δx measured: frontal fast/slow ratio +0.3. Logged. No induction intent.
t+2:40   precursor score rises to 0.6. Above threshold.
         Controller selects the actuator with the largest identified
         effect on this feature, at the lowest effective intensity.
t+2:45   intervention begins. Response window opens.
t+2:48   response: state moved, no arousal. Continue.
t+3:20   precursor score 0.8. Hold.
t+4:05   spontaneous cortical arousal detected. Servo takes priority,
         holds arousal depth at +0.4 for 70 s, releases.
t+5:30   re-entry into REM confirmed.
t+6:12   LRLR DETECTED. High confidence, two sequences, 11 s apart.
         All stimulation stops. Timestamp recorded.
         Single soft confirmation tone. Nothing further for 10 minutes.
morning  dream report written before any output is read.
         This episode becomes training example #24.
```

That is the system. Everything before V4 is instrumentation for that loop;
everything after is the loop learning to close itself.

---

## 11. What to do this week

Nothing in this document requires a purchase.

1. **Run E1.** 40 nights, randomised, alarm clock. It tests the
   highest-probability architecture and it costs nothing.
2. **Titrate the gain** to a level with incorporation and no arousal.
3. **Keep journalling** to the pinned outcome definition.

The rate-limiting resource for the entire programme is **verified lucid events**,
and E1 is the cheapest thing that could increase them.
