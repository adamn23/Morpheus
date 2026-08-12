# Muse S Athena — full-potential programming plan

**Status:** design only, 2026-08-08. Nothing here is built. Gated on the free
headband-tolerance test.

Conditional on buying an Athena, this is what Morpheus should do with it, in
build order, with the algorithms named. Claims are labelled **[established]**,
**[supported, untested for LD]**, or **[novel hypothesis]** throughout.

---

## 1. What the hardware actually gives us

| Signal | Rate | Use |
|---|---|---|
| EEG AF7, AF8 (frontal) | 256 Hz | **EOG** — the whole novel programme |
| EEG TP9, TP10 (mastoid) | 256 Hz | reference, staging, arousal |
| PPG | 64 Hz | heart rate, HRV — autonomic |
| Accelerometer / gyro | 52 Hz | motion, posture, body position |
| fNIRS | slow | prefrontal haemodynamics; research curiosity only |

Streaming over LSL via [OpenMuse](https://github.com/DominiqueMakowski/OpenMuse).

### The one trick everything rests on

Horizontal eye movement is an electrical dipole — cornea positive, retina
negative. AF7 sits left, AF8 right, both above the eyes. So:

```
eog_h   = AF7 - AF8      # differential: eye movement adds, brain cancels
eeg_c   = (AF7 + AF8)/2  # common mode: brain adds, horizontal eye cancels
blinks  → common mode (vertical), so they show up in eeg_c, not eog_h
```

**`AF7 - AF8` is a clean horizontal EOG channel.** That single line is what
makes phasic-REM detection and eye-signal verification possible on a $520
consumer headband. Everything in §4 and §5 depends on it.

---

## 2. Build order

### A1 — Plumbing (~2 days)

`src/morpheus/capture/muse.py`: a `MuseSource` implementing the existing
`FrameSource` protocol. LSL inlet → ring buffer → 1 Hz `FeatureFrame`.

Map onto the **sensor-neutral fields that already exist** — the controller and
gate stack consume these and have never known where they came from:

| FeatureFrame field | From |
|---|---|
| `eye_flow_l` / `eye_flow_r` | EOG velocity, left/right deflection energy |
| `eye_flow_bilateral_corr` | phasic-burst confidence (§4.1) |
| `eye_region_usable` | electrode contact quality |
| `signal_quality` | impedance / artefact fraction |
| `global_motion` | accelerometer magnitude |
| `face_present` | redefined: headband worn and in contact |

Camera-shaped fields (`focus`, `luminance_mean`, `interocular_px`) stay `None`.
Add `hr`, `hrv_rmssd`, `stage_conf` in a new migration.

**Nothing in `cue/` changes.** That was the point of the seam.

### A2 — Contact quality and the silent-failure guard (~1 day)

The Muse's failure mode is an electrode losing skin contact mid-night: no
error, just noise. Same class as the suspend bug — a night that looks like data
and isn't.

Compute per-channel contact from line-noise power (50/60 Hz) and railing rate.
If `eye_region_usable` falls below threshold for a sustained period, mark the
night **DEFECTIVE** in the summary, exactly as truncation and suspend already
are. Reuse `NightSummary.defects()`.

### A3 — Sleep staging (~3 days)

Two tiers, because they answer different questions at different speeds.

**Offline, 30 s epochs — [established].** [YASA](https://raphaelvallat.com/yasa/)
on the raw export. Already supported: `reference/ingest.py` accepts YASA
hypnograms *today*. This gives REM mapping with zero live code, and it is the
first thing to do after the device arrives.

**Real-time, 30 s epochs.** Lightweight classifier on band powers from
`eeg_c` plus EOG density plus EMG proxy. Target: distinguish REM from
NREM/wake, not full 5-class staging. Validate against YASA on your own nights
before it is allowed to gate anything.

**Gate:** live REM detection must reach **AUC ≥ 0.70** against the offline
YASA scoring on held-out nights before G9 is promoted. That is design.md §22's
existing bar, reused unchanged.

### A4 — Phasic REM detector (~3 days)

The core of the novel work. Operates on `eog_h` at 256 Hz.

```
1. Bandpass 0.3–6 Hz            # REM band; removes drift and EMG
2. Differentiate → velocity
3. Threshold: |v| > k * MAD(v)  # per-night adaptive, robust to drift
4. Reject blinks: large common-mode in eeg_c at the same instant
5. Phasic if >= 2 qualifying movements within any 4 s window
```

Step 5 is the published definition of phasic REM. Step 4 matters more than it
looks — blinks are the dominant artefact and are common-mode, so the
differential channel already suppresses them; this catches the rest.

Latency budget: detection within **~200 ms** of the second movement, which is
well inside a phasic burst lasting seconds.

### A5 — LRLR eye-signal detection (~4 days)

**This is the single biggest upgrade available, and it is about measurement,
not induction.** Your primary outcome is currently self-report from someone who
wants the intervention to work. LaBerge's paradigm replaces that with physiology.

Agree a signal in advance: **left-right-left-right, deliberate and large**,
performed on becoming lucid.

```
template = 4 alternating deflections, ~0.5-1 s each, amplitude >> spontaneous REM
match    = normalised cross-correlation of eog_h against the template
accept   = correlation > threshold AND amplitude > 3x the night's REM median
           AND the pattern is not repeated within 10 s (spontaneous REMs cluster)
```

Threshold set on *waking* rehearsal data — you perform LRLR awake, eyes closed,
during the calibration session, exactly as `calibrate` already does for
saccades. That reuses the existing positive-control machinery.

Output: a timestamped, physiologically verified lucidity event. Store it beside
`lucid_binary`, never overwriting it — self-report and signal verification are
different measurements and both are worth having.

### A6 — Closed-loop control (~1 week)

Promote **G9**, gated on A3 passing. New gates alongside it:

- **G10 phasic** — cue only inside a detected phasic burst *[novel]*
- **G11 arousal-imminent** — suppress if an awakening looks imminent (§5.3)
- **G12 already-lucid** — stop cueing after an LRLR is detected. Cueing someone
  who is already lucid buys nothing and risks waking them.

The **arousal-aware controller finally works**: EEG-detected arousal produces
a real `probable_arousal` outcome, so the gain steps down and the cooldown
doubles automatically. That is design.md §6's novelty claim #2, currently inert
because every sensorless outcome is `uncertain`.

---

## 3. Things this fixes that are already built

| Component | Today | With Athena |
|---|---|---|
| Adaptive gain | never moves — outcomes always `uncertain` | steps down on real arousal |
| G4/G5 motion & arousal gates | log `unavailable`, always pass | actually gate |
| Sleep-quality stopping rule | 1–5 self-rating | measured fragmentation |
| `minutes_to_sleep_after_wbtb` | asked in the morning | **measured** (§5.2) |
| Primary outcome | self-report | self-report **+ LRLR verification** |

---

## 4. Novel directions worth building

### 4.1 Phasic-locked cueing — *[novel hypothesis]*

Cue at phasic-burst onset rather than anywhere in REM.

Grounds: lucid dreaming occurs in *activated* REM with phasic features
**[established]**; TMR time-locked to rapid eye movements improved consolidation
in non-LD work **[established]**; phasic/tonic dissociation in TMR is barely
studied. **No one appears to have tested phasic-locked TLR.**

The mechanism that makes it more than "more cues in REM": arousal threshold
varies with REM substate, so a phasic-locked cue should be **both more likely
incorporated and less likely to wake you at the same volume.** That directly
addresses the collision problem measured on night 1.

Falsifiable in N-of-1 as a Phase B arm: phasic-locked vs REM-anywhere, matched
cue, matched count.

### 4.2 REM-offset-triggered WBTB — *[supported, untested]*

Erlacher & Stumbrys woke participants **"from a subsequent REM period."** A
fixed 6.0 h alarm cannot do that; it fires wherever it fires. With staging, the
alarm can wait for a REM offset near the target hour and wake you at the
natural cycle boundary.

Combined with the fact that you already wake ~5×/night — many of those are
probably at cycle boundaries — this could make WBTB nearly free in sleep cost
while being *more* faithful to the protocol than the study it comes from.

Replaces `--wbtb-at 6.0` with `--wbtb-after 5.5 --wbtb-on rem-offset`.

### 4.3 Measured sleep-onset instead of a guessed delay — *[established method]*

`--post-wbtb-delay-min 25` is a guess. With staging, resume cueing a set
interval after **detected sleep onset**. Removes a made-up parameter and, as a
side effect, measures the ILDIS sleep-latency predictor objectively rather than
asking you to estimate it at 09:30.

### 4.4 Imminent-awakening suppression — *[novel hypothesis]*

You wake ~5×/night and those wakings cluster in the REM-dense hours where cues
fire. If alpha intrusion and rising EMG/HR predict a spontaneous awakening
seconds ahead, the controller can decline to cue into it.

This is the most directly useful novel idea for *you specifically*, because it
targets the exact failure you measured. Testable offline first: label
spontaneous awakenings in recorded nights, see whether the preceding 30 s is
separable. **No hardware risk — it is pure analysis of data you will already
have.**

### 4.5 Autonomic precursor gating — *[novel hypothesis, weaker]*

Autonomic change can precede the lucidity eye-signal by up to ~30 s
**[established]**. PPG gives HR and HRV. Gate cues on a rising-arousal-without-
awakening window.

Caveat stated plainly: the precursor is observed *before lucidity that already
occurred*, so using it predictively is an extrapolation. Cheap to test once
A5 gives you verified lucidity timestamps to regress against.

### 4.6 Two-way dream communication — *[established in lab, stretch goal]*

Konkoly et al. achieved real-time dialogue with lucid dreamers. With A5 in
place the ingredients exist: detect LRLR, present a question as a cue pattern,
listen for an LRLR-coded answer.

Not an induction technique, and it should not compete for build time with
anything above. But it is the most striking demonstration the hardware permits,
and it becomes nearly free once LRLR detection works.

### 4.7 Per-night parameter optimisation — *[established method, novel application]*

`cue/adaptive.py` already implements Thompson sampling over discrete arms with
bounded updates and policy versioning, and is dormant. With verified outcomes
it can optimise over a real arm space: gain × timing rule (phasic / REM-anywhere
/ clock) × cue count. Bounded by the same safety limits, which no policy can
touch.

### 4.8 fNIRS — *[speculative, log only]*

Prefrontal haemodynamics, seconds-scale. Too slow for cue timing, and there is
no published lucid-dream fNIRS work at all. **Do not build control logic on it.**
Record it, and once A5 provides verified lucidity timestamps, look retrospectively
at whether prefrontal activation precedes them on a minutes timescale. If it
does, that is a genuinely new finding. If not, nothing was spent.

---

## 5. Explicitly rejected

**Gamma-based lucidity detection.** The obvious idea, and
[J Neurosci 2025](https://www.jneurosci.org/content/45/20/e2237242025) rules it
out for this hardware: pooled across labs with a saccade-aware pipeline,
sensor-level lucid-vs-REM differences are **minimal**; real effects are
source-level (right parietal/TPJ beta, precuneus gamma), and four channels
cannot do source localisation. The classic Voss 40 Hz result is partly saccade
artefact.

**Use EOG for lucidity, EEG for staging.** That split is the design.

---

## 6. Sequencing against the trial

| When | What | Needs |
|---|---|---|
| Week 1 | A1, A2, offline REM mapping | nothing live |
| Week 2 | Retime WBTB and cue window from measured REM | A3 offline |
| Weeks 3–5 | A3 real-time, validate vs YASA | held-out nights |
| Weeks 4–6 | A5 LRLR — outcome upgrade | waking calibration |
| Weeks 6–10 | A4, A6 — phasic detection and closed loop | A3 gate passed |
| Ongoing | 4.4 offline analysis | recorded nights only |

**Phase A cueing continues throughout on the clock.** None of this blocks the
protocol, and the shadow-mode data accumulates during nights you are running
anyway.

---

## 7. Risks

- **Tolerance.** Untested, and behind-ear electrodes plus side-sleeping is the
  known bad case. Everything above is conditional on the beanie test.
- **Contact loss.** The most likely silent failure; A2 exists for it.
- **Battery.** 10 h capacity against 10 h nights is zero margin, and the tail
  is the REM-dense part. Use in blocks, not nightly.
- **Fabric and sensor degradation** over months of wear. Another reason for
  blocks.
- **Real-time staging may not reach AUC 0.70** on four channels. If it does not,
  G9 stays disabled and the offline value (REM mapping, verified lucidity,
  §4.4 analysis) still stands. That is a decent floor.
