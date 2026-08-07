# Morpheus — System Design Document v0.1

**Status:** Design for review. No code written.
**Repository:** `/Users/adam/Morpheus` — empty except a `LICENSE`, one commit (`077e789`), remote `github.com/adamn23/Morpheus`. Nothing to reuse; this is a greenfield design.

> **Licensing note (updated 2026-08-07).** The project was initially AGPL-3.0 and is now
> All Rights Reserved. Passages below written against the original licence have been
> corrected; the architecture and evidence review are unaffected.

---

## Context

You want a closed-loop lucid-dream induction system built from a camera, a computer, and a speaker. You asked me to be critical rather than accept the premise. After grounding the design in the current literature, **I am recommending an architecture materially different from the one you proposed**, for three reasons that surfaced during research:

1. **The evidence-backed part of this system is not the camera — it is the cue-conditioning protocol.** Targeted Lucidity Reactivation (TLR) has produced replicated effects using *purely scheduled* audio cues with **no sleep sensing at all**.
2. **A silicon camera physically cannot see eye movement through a closed eyelid.** The published work that achieves this uses short-wave infrared (0.9–1.7 μm, InGaAs sensors). Webcams and Pi NoIR cameras cut off around 1000 nm, and the eyelid is effectively opaque there. What remains observable is the much weaker *corneal bulge deforming the lid surface* — real, but small, and requiring a near-frontal, close, stable view.
3. **You are a side/stomach sleeper.** That is close to the worst case for any face-facing camera. Your eye region will be turned away, pillow-occluded, or both, for a large and currently unmeasured fraction of the night.

None of this means the project is not worth building. It means the camera must **earn its way into the control loop by beating a scheduled baseline**, rather than being assumed into it. That reframing is the core of this design, and it also happens to be what makes Morpheus a genuine contribution rather than a reimplementation of a 2013 Kickstarter.

---

## 1. Executive Summary

Morpheus is a local-first, single-user, research-grade N-of-1 platform for testing and delivering auditory lucid-dream cues.

**Recommended V1 architecture:** a *schedule-driven cue engine* implementing TLR, wrapped in a *blinded randomized N-of-1 experiment harness*, with a camera in a **gate-and-guard role** — not a REM detector. The camera's V1 job is to answer two questions it is actually competent to answer: *is the body quiescent enough that a cue is unlikely to land during wake?* and *did the cue just wake the user?* Eye-movement detection is developed in parallel as an explicitly gated **research track** that is forbidden from influencing cue timing until it is validated against a reference signal.

**Why this ordering:** the scheduled arm is simultaneously (a) the only component with published efficacy, (b) the necessary control condition for any sensing claim, and (c) shippable in weeks. Building camera-driven cueing first would produce a system whose central claim is untestable.

**Immediate first milestone (M0):** a *feasibility probe* — an overnight recorder that persists derived features only (never video) for 5–7 nights and measures whether your face and eye region are visible at all. This costs ~1–2 weeks and can invalidate or confirm the camera premise before a single line of detection logic is written.

**The honest headline:** the probability that a consumer camera meaningfully improves cue timing over a good schedule, for a side sleeper, is in my estimation **low — perhaps 15–25%**. The probability that Morpheus becomes a genuinely useful and novel *research instrument* regardless of that outcome is high. Design accordingly: make the null result cheap and publishable.

---

## 2. Product Definition

Morpheus is a locally-hosted application composed of:

- **A conditioning trainer** — a pre-sleep / WBTB guided protocol that binds a unique audio cue to a state of critical self-awareness.
- **An overnight cue engine** — a headless daemon that decides when, whether, and how loudly to play the conditioned cue, subject to hard safety constraints.
- **A sensing layer** — a camera pipeline producing a timestamped stream of derived, non-reconstructable features (visibility, pose, motion energy, eye-region activity, signal quality).
- **A morning report system** — structured dream recall capture with lucidity, cue-perception, and sleep-quality items.
- **An experiment harness** — reproducible, seeded, blinded randomization across cue conditions with condition concealment until after report submission.
- **An analysis layer** — N-of-1 statistics, adaptive policy fitting, and dashboards that report uncertainty rather than hiding it.

**Explicit non-goals.** Morpheus does not perform sleep staging. It does not diagnose. It does not confirm REM. It does not guarantee lucid dreams. It does not use electrical stimulation, and it makes no pharmacological recommendations of any kind.

---

## 3. User Goal

A single technically sophisticated user (you) wants to increase personal lucid-dream frequency **and know whether the increase is real**. The second half is the harder and more interesting requirement, and it is the one most existing products abandon.

Success is therefore two-sided:
- **Product success:** measurably more lucid dreams than a matched control condition, without degrading sleep.
- **Epistemic success:** a defensible answer either way, including a clean negative result.

A design that can only produce good news is a failed design.

---

## 4. Scientific and Technical Assumptions

Separated by evidential status, as requested. This section is the spine of the document; everything downstream inherits its confidence levels.

### 4.1 Proven / well-supported

| Claim | Support |
|---|---|
| Lucid dreams occur predominantly in REM sleep. | Long-established sleep science. |
| External sensory stimuli can be incorporated into dream content without waking the sleeper. | Basis of all cueing devices since DreamLight. |
| Pairing a cue with pre-sleep critical-awareness training, then re-presenting that cue in sleep (TLR), increases lucid dreaming. | Carr et al.; ~50% of participants lucid in a single lab nap. |
| **TLR works at home with purely scheduled cueing and no REM detection.** Frequency rose 0.74 → 2.11 lucid dreams/week (Exp. 1, n=19); Exp. 2 (n=50, 7 nights) included untrained-cue and no-cue controls. | Northwestern app study, *Consciousness and Cognition* 2024. |
| Sleep intervals later in the night contain proportionally more and longer REM. | Standard sleep architecture. |
| WBTB and MILD raise lucid-dream rates. | Replicated behavioural literature. |
| Consumer forehead EEG can stage REM with substantial-to-near-perfect agreement vs PSG (reported κ ≈ 0.85 for REM). | Independent and vendor-adjacent validations; treat vendor-funded figures with caution. |
| Camera-based sleep staging tops out near **73% accuracy, κ ≈ 0.61** for 4-class staging — and derives that from **rPPG heart rate, respiration, and gross body motion**, not from eye movement. | HealthBed study; SleepVST; NIR-video staging work, 2023–2025. |
| Seeing the pupil through a closed eyelid requires **short-wave infrared (~0.9–1.7 μm)** imaging. | *Communications Medicine* 2024, SWIR closed-eye pupillometry. |

### 4.2 Plausible engineering assumptions (untested here, testable)

- Corneal bulge motion under the closed lid produces a *geometrically detectable* deformation at ≥720p, ≥30 fps, from ~40–70 cm, at near-frontal pose. **Plausible; unquantified for this user.**
- Gross body-motion quiescence is a usable *negative* gate — sustained stillness makes "currently awake and moving" unlikely, even though it does not imply REM. **High confidence.**
- Post-cue motion onset within a short window is a usable proxy for probable arousal. **Moderate-to-high confidence; will have false positives from spontaneous movement.**
- Face landmarkers (MediaPipe) degrade substantially on closed eyes, IR-illuminated monochrome imagery, extreme yaw/roll, and partial occlusion — but *face-present/absent* and *motion energy* remain robust. **High confidence; documented weakness of landmarkers on closed/occluded eyes.**
- rPPG respiration extraction from a NIR camera is feasible on exposed skin but will fail under blankets and at extreme angles. **Moderate.**

### 4.3 Speculative hypotheses (Morpheus's actual research questions)

- **H1:** A camera-derived eye-movement index discriminates REM from NREM better than chance for a side sleeper. *(Falsifiable; the core gamble.)*
- **H2:** Cue timing locked to detected eye-movement bursts outperforms scheduled cueing on lucidity rate. *(The product thesis. Cannot be tested until H1 passes.)*
- **H3:** Arousal-aware adaptive volume increases cue incorporation while reducing awakenings, relative to fixed volume.
- **H4:** Personalized thresholds outperform population defaults enough to justify a calibration burden.
- **H5:** Journal-derived personal dream signs, injected into the conditioning script, increase lucidity beyond generic training.

Treat H2–H5 as sequenced, not parallel. Each requires the previous to hold.

### 4.4 The assumptions that would sink the project

Ranked by (probability of being false) × (cost if false):

1. **The eye region is visible enough of the night to matter.** Side/stomach sleeping puts this in serious doubt. *Mitigation: M0 measures it before anything is built on it.*
2. **The eyelid-bulge signal exceeds the noise floor of a silicon camera at bedroom distances.** *Mitigation: M0 measures SNR against deliberate waking eye movements.*
3. **A reference standard is obtainable.** Without EEG/EOG ground truth, no camera claim is falsifiable. Note a wrinkle specific to you: **forehead-and-ear EEG headbands are uncomfortable for side sleepers**, since behind-ear electrodes press into the pillow. Reference-signal acquisition is itself a risk.
4. **Enough nights.** At a ~0.74/week baseline, detecting even a doubling requires on the order of **100+ nights per arm**. This is a months-long commitment, not a weekend. See §22.

---

## 5. Existing Systems and Overlap

| System | What it does | Morpheus overlap |
|---|---|---|
| **NovaDreamer / DreamLight** (Lucidity Institute) | Mask, IR eyelid-movement sensing, light cues. The only mask line with real experimental backing. | Morpheus reproduces the *concept* contactlessly. NovaDreamer's sensor sits millimetres from the lid; Morpheus's sits a metre away. This distance gap is the whole difficulty. |
| **Remee** | Blind timed light cues, no sensing. | Morpheus's scheduled arm is a strictly better version (audio, conditioned, adaptive, measured). |
| **Aurora (iWinks)** | Open-source-ish EEG headband, REM detection, cues. Shipped with an explicit "REM-detection algorithm is not yet perfect" disclaimer. | Direct conceptual overlap; Morpheus is the contactless variant. |
| **ZMax (Hypnodyne)** | Research-grade wearable, raw data, real-time API. | The credible reference-standard option. |
| **Dreamento** | **Open-source Python dream-engineering toolbox** for real-time EEG monitoring, stimulation, and analysis on ZMax. | *Closest prior art to Morpheus's software ambition.* Study it before designing; do not duplicate its EEG path if you go hybrid. |
| **Lucid Scribe (webcam plugin)** | Monitors webcam/security-camera streams for rapid eye movements via **inter-frame difference**. | **Direct prior art for your exact premise.** Naive frame-differencing; no validation, no arousal handling, no experiment harness. |
| **INSPEC** | Night-vision smart camera detecting REM eye movements, triggering audio/visual cues. | **Direct prior art.** Confirms the idea is not new. |
| **Northwestern TLR app** (2024) | Scheduled TLR cueing, 6-hour delay, volume ramp, controlled trial. | **This is Morpheus's baseline arm.** Authors named the absence of REM detection as their key limitation and the obvious next step. |
| **Muse S / Oura / Apple Watch** | Consumer sleep staging. | Reference-signal candidates; Muse alone has credible REM κ. |

**Honest summary:** camera-based REM detection for lucid dreaming has been attempted at least twice (Lucid Scribe, INSPEC). Neither published validation. Morpheus is not first to the idea. It can be first to *test it properly*.

---

## 6. Potential Novel Contribution

Stated conservatively. Most of the feature list is reimplementation; these four are defensible:

1. **A validated (or credibly falsified) contactless eye-movement index.** No one has published whether a consumer camera can do this. A rigorous negative result — with SNR figures, visibility statistics, and a reference comparison — is a real contribution and is the *likely* outcome.
2. **An arousal-aware closed-loop cue controller with published stopping rules.** Existing devices cue and hope. Morpheus observes the response, attributes probable arousal, and adapts volume/cooldown under transparent, inspectable policy. Sleep-preservation as a first-class optimization target is not standard in this space.
3. **A reusable open-source blinded N-of-1 harness for dream research.** Seeded randomization, condition concealment until post-report, pre-registered analysis, exportable. This is arguably the most valuable artifact Morpheus could produce, and it is entirely independent of whether the camera works.
4. **Direct within-subject comparison of sensor-timed vs. scheduled TLR** — precisely the open question the Northwestern authors flagged. Morpheus is positioned to answer it, at N=1, honestly.

**Not novel:** TLR itself, cue conditioning, WBTB, dream journalling, adaptive volume as a concept, camera-based sleep monitoring, Bayesian personalization.

---

## 7. Architecture Options

Scored 1–5 (5 = best). "Cue-timing value" is the only column that ultimately matters.

| Option | Signal quality | Dream-event detection | Comfort | Cost | Impl. difficulty | Raw data | Real-time | Privacy | False positives | **Cue-timing value** |
|---|---|---|---|---|---|---|---|---|---|---|
| **Normal webcam (visible light)** | 1 | 1 | 5 | 5 | 3 | 5 | 5 | 2 | Very high | **1** — needs light; light disturbs sleep. Effectively disqualified. |
| **IR-illuminated USB camera (850/940 nm)** | 2–3 | 2 | 5 | 5 | 3 | 5 | 5 | 3 | High | **2** |
| **Pi NoIR + IR illuminator** | 2–3 | 2 | 5 | 4 | 2 | 5 | 4 | 4 | High | **2** — same optics as above, better as an appliance. |
| **Phone camera** | 3 | 2 | 5 | 5 | 2 | 2 | 3 | 2 | High | **2** — good sensor, bad platform: thermals, backgrounding, OS audio policy, charging heat. |
| **Consumer EEG (Muse S)** | 4 | 4 | 2 (poor for side sleepers) | 2 | 3 | 4 | 4 | 4 | Low | **4** |
| **Research EEG/EOG (ZMax)** | 5 | 5 | 2 | 1 | 3 | 5 | 5 | 4 | Very low | **5** |
| **HR / motion wearable** | 2 | 2 | 4 | 3 | 4 | 2–3 | 2 (vendor-lagged) | 3 | Moderate | **2** |
| **Hybrid camera + EEG** | 5 | 5 | 2 | 2 | 1 | 4 | 4 | 3 | Low | **5** — best signal, worst build cost. |
| **Scheduled audio only** | n/a | n/a | 5 | 5 | 5 | n/a | 5 | 5 | n/a | **3 — and it is the only option with published efficacy.** |

### Reading the table

The result that should reorganize your thinking is the **bottom row**. Scheduled-only scores a 3 on the column that matters while scoring 5 on cost, comfort, difficulty, and privacy. Every camera option scores *below it* on cue-timing value. A camera is not free — it costs weeks of engineering and buys a signal that is, on current evidence, weaker than a clock.

Three specific disqualifications worth stating plainly:

- **Visible-light webcam is out.** It requires illuminating the sleeper's face. Light suppresses melatonin and fragments sleep. You cannot study sleep by ruining it.
- **Silicon NIR cannot see the pupil.** At 850/940 nm the eyelid is opaque; the published closed-eye pupillometry uses 0.9–1.7 μm SWIR on InGaAs sensors, which cost thousands. Anyone claiming a webcam "sees REM" is measuring lid-surface deformation or frame noise — not eye position.
- **Consumer wearables (Oura/Apple/Fitbit) are out for closed loop.** Their staging is post-hoc, vendor-smoothed, and unavailable in real time.

**Side-sleeping penalty.** Every camera row above should be discounted further for your case. A ceiling- or headboard-mounted camera sees a near-frontal face only when you are supine. If that is 25% of the night, the effective duty cycle of any eye-region feature is 25% *before* accounting for pillow occlusion and blanket coverage.

---

## 8. Recommended Architecture

**A staged hybrid, with the camera in a subordinate role and a hard validation gate before it may influence cueing.**

```
V1  Scheduler-driven TLR  +  camera as gate-and-guard  +  N-of-1 harness
        │
        ├─ cue timing         ← adaptive schedule (evidence-backed)
        ├─ cue suppression    ← camera: gross motion / probable wake
        ├─ post-cue response  ← camera: probable arousal → stop / reduce
        └─ eye-movement index ← computed, logged, SHADOW MODE ONLY
                                 (never touches the cue decision)

        ══ VALIDATION GATE ══  reference EEG/EOG; H1 must pass (§22)

V2  Eye-movement index promoted to a cue-timing input, A/B'd against V1
V3  Full hybrid: EEG-derived stage prior fused with camera features
```

### Why this and not camera-first

1. **The scheduler is the control condition.** You cannot claim the camera improves timing without a scheduled arm to compare against. Building it first is not a compromise — it is a methodological requirement.
2. **Shadow mode makes the gamble cheap.** The eye-movement index is developed, logged, and evaluated from night one. It simply is not trusted. If M0/M1 show the signal is absent, you delete one module and still have a working, evidence-backed system.
3. **The camera's gate-and-guard role does not depend on H1.** Gross motion detection works at any pose, in the dark, from any angle, through blankets, with trivial CV. It is the highest-value-per-unit-effort component in the entire system, and it is *safety-relevant* — it is what stops Morpheus from repeatedly blasting a tone at an already-awake user.
4. **Side sleeping.** Betting the architecture on frontal face visibility, for a stomach sleeper, would be reckless.

### Stack decisions, with justification

| Choice | Verdict | Reasoning |
|---|---|---|
| **Python 3.12+** | Yes | Only ecosystem with OpenCV + MediaPipe + SciPy + sklearn + audio in one process. |
| **FastAPI + Uvicorn** | Yes | Local control plane, WebSocket telemetry, typed via Pydantic. |
| **OpenCV** | Yes | Capture, ROI, optical flow (Farnebäck / Lucas–Kanade). |
| **MediaPipe Face Landmarker** | Yes, **with a fallback** | 478 landmarks, real-time, free. But it degrades on closed eyes and IR monochrome, and has no support guarantees. Wrap it behind a `LandmarkProvider` interface so it can be swapped or bypassed; the motion pipeline must work with **zero** landmarks. |
| **NumPy / SciPy / pandas** | Yes | Ring buffers, Welch PSD, bandpass filtering, offline analysis. |
| **scikit-learn** | Yes | Logistic regression, HMM-adjacent utilities, calibration curves. Right complexity for ~100 nights of data. |
| **PyTorch** | **No, until V2+** | You will have no labels and no reference standard. Deep learning on an unvalidated target with n≈100 nights is how you produce a model that predicts your pillow. Admit it only after H1 passes *and* you hold reference-labelled data. |
| **SQLite (WAL)** | Yes | Single-user, local-first, ACID, zero ops, trivially backed up and exported. |
| **PostgreSQL** | **No** | Solves multi-user concurrency and network access — neither of which exists here. Pure operational cost. |
| **React + TypeScript + Vite** | Yes | Journal, calibration wizard, dashboards. |
| **WebSockets** | Yes — **observation only** | Live telemetry to the UI. **The cue path must never traverse the network.** |
| **Docker** | **No for V1** | Docker Desktop on macOS cannot pass through a USB camera or the audio device. Hard blocker. Use `uv`/venv natively; reserve a container for the offline replay/analysis harness only. |
| **Audio: `sounddevice`/PortAudio, in-process** | Yes | Deterministic, low-latency, no browser tab, no OS notification path, no dependency on a UI being open at 04:00. |

### The single most important structural rule

**The overnight cue engine is a headless daemon. The web UI is a read-only observer that can be closed, crashed, or absent without affecting the night.** A browser tab is not a life-support system. Every device in §5 that failed in the field failed at 4am, unattended.

### macOS-specific reality (your chosen platform)

You chose the Mac laptop by the bed. That is fine for development, but these are not theoretical problems and must be handled in M0:

- **System sleep will kill the run.** Requires `caffeinate -dimsu` or an `IOPMAssertion`; verify empirically across a full night before trusting any data.
- **Clamshell mode suspends** unless on external power with a display/dongle attached. Practically: lid open, screen brightness zero, `Shift-Ctrl-Eject` display sleep.
- **Fan noise and screen glow** are themselves sleep disturbances, and an *uncontrolled confounder* across conditions. Position the machine away from the head; measure whether fans spin up under sustained CV load (they will — budget CPU accordingly, see §10).
- **Camera and microphone permissions** must be granted to the daemon's binary, not to Terminal, or the daemon fails silently on relaunch.
- **Monotonic clock only.** Use `time.monotonic()` for all intervals; wall-clock for logging. DST transitions must not corrupt a night.

Plan a Raspberry Pi 5 migration path (§26) once the software stabilizes — it removes all five issues above. Keep hardware access behind an interface so this is a config change, not a rewrite.

---

## 9. End-to-End System Flow

```
EVENING
  └─ Conditioning session (5–12 min): cue playback ⇄ critical-awareness
     rehearsal, dream-sign visualization, MILD intention.
     → writes training_session row; cue asset hash recorded.

PRE-SLEEP
  └─ Experiment harness draws tonight's condition from a seeded PRNG.
     Condition is written ENCRYPTED-AT-REST and NOT displayed.
  └─ Camera framing check: live preview, ROI confirmation, IR exposure,
     focus, signal-quality floor. Refuses to arm if quality < threshold.

NIGHT (daemon, headless)
  IDLE → ARMED → SETTLING → MONITORING ⇄ CANDIDATE → CUEING
                                      → POST_CUE_OBSERVE → COOLDOWN
                                      → (ABORT_AROUSAL) → MONITORING
  Per frame (~15–30 fps):
    capture → quality assessment → face/pose → ROI extraction
    → motion energy + eye-region flow → feature ring buffer
    → 1 Hz feature aggregation → SQLite (features only; NO video)
  Cue decision at 1 Hz against the gate stack (§12).
  Hard caps enforced independently of all policy (§24).

MORNING
  └─ Report captured BEFORE condition is revealed. Narrative, lucidity
     (binary + confidence + "knew I was dreaming" separately), cue
     perception (heard / indirect / woke me / unsure), recall count,
     vividness, sleep quality, notes.
  └─ Condition unblinded and joined. Nightly summary generated.
  └─ Adaptive layer updates posteriors (§12.4). Nothing auto-changes
     outside pre-declared bounds.

WEEKLY
  └─ N-of-1 analysis refresh; stopping rules evaluated (§22, §23).
```

---

## 10. Computer-Vision Pipeline

Design principle: **graceful degradation**. Each stage may fail; the pipeline must emit a usable, honestly-labelled quality score rather than a confident wrong answer.

```
[0] Capture          MJPG/YUY2, fixed manual exposure + gain + focus.
                     Auto-exposure MUST be disabled — AE hunting creates
                     brightness oscillations indistinguishable from motion.
                     Target 30 fps @ 1280×720. Timestamp at capture.

[1] Quality gate     Mean/σ luminance, saturation %, Laplacian focus,
                     frame-drop rate, global scene-change detector
                     (camera bumped/moved). → signal_quality ∈ [0,1].

[2] Presence         Lightweight face detector at 2–5 Hz (not per frame).
                     Outputs: face_present, bbox, detector confidence.
                     Fallback when absent: whole-frame motion only.

[3] Landmarks        MediaPipe Face Landmarker at 5–10 Hz on the face
                     crop. Outputs 478 pts, yaw/pitch/roll, per-eye
                     landmark confidence.
                     EXPECTED TO FAIL on closed eyes / heavy yaw / IR
                     monochrome. Failure is logged as a feature, not an
                     error — landmark_availability is itself informative.

[4] ROI tracking     Per-eye ROI derived from landmarks when available;
                     otherwise from a Lucas–Kanade / KCF tracker seeded
                     at the last good landmark set, with drift detection.

[5] Stabilization    Register successive eye ROIs (ECC or phase
                     correlation) to remove head translation.
                     CRITICAL: unregistered head motion masquerades as
                     eye motion. This is the primary false-positive
                     mechanism, and the reason naive frame-differencing
                     (cf. Lucid Scribe) is untrustworthy.

[6] Features @30 Hz  Per-eye:
                       - dense optical flow magnitude & dominant direction
                         within the registered ROI
                       - residual energy after global-motion subtraction
                       - lid-contour displacement (when landmarks avail.)
                       - bilateral correlation between L and R eye flow
                         ← the key specificity feature: true ocular motion
                           is conjugate; noise and lighting are not
                     Global:
                       - whole-body motion energy (frame difference,
                         torso/bed region)
                       - head pose velocity
                       - respiration proxy (chest/shoulder ROI, 0.1–0.5 Hz
                         bandpass) — best-effort, often unavailable

[7] Aggregation @1Hz Windowed statistics (1 s, 10 s, 30 s, 5 min):
                     mean, σ, 90th pct, burst count, quality-weighted.
                     → persisted to SQLite. Raw frames discarded.
```

**Compute budget.** Steps 3–6 at full rate on a laptop CPU will spin fans. Mitigations: run landmarks at 5–10 Hz not 30; process only the eye ROI at full rate; downscale globally to 720p; use `cv2.setNumThreads` deliberately; measure and log CPU/thermals as part of M0 acceptance. **If the machine audibly cycles fans, the experiment is confounded.**

**The bilateral-correlation feature deserves emphasis.** It is the single best defence against false positives. Genuine eye movements are conjugate — both eyes move together in the same direction. Shadows, IR flicker, blanket motion, breathing-induced head sway, and sensor noise are not. Requiring *correlated bilateral* activity should dramatically improve specificity, at the cost of requiring both eyes visible — which for a side sleeper is a significant cost. Log both unilateral and bilateral variants; let the data decide.

---

## 11. Detection and Classification Strategy

### Naming discipline (enforced in code, not just docs)

The event taxonomy is a closed enum. Any string outside it fails a unit test.

| Permitted label | Definition |
|---|---|
| `probable_eye_movement_burst` | Registered bilateral eye-ROI flow exceeding a personalized threshold for ≥ N ms at acceptable quality. |
| `possible_dream_activity` | A burst pattern co-occurring with body quiescence in a plausible time window. Explicitly a *guess*. |
| `probable_arousal` | Sharp increase in body-motion energy and/or head-pose velocity. |
| `possible_awakening` | Sustained arousal + posture change + (optionally) user-confirmed. |
| `cue_delivered_during_detected_activity` | Factual record; asserts nothing about sleep stage. |
| `signal_unavailable` | Face absent, quality below floor, or landmarks unavailable. |

**Forbidden in code, UI, database, and logs:** `rem_detected`, `is_rem`, `dreaming`, `sleep_stage`, `confirmed_*`. A CI lint rule greps for these. This is not pedantry — the naming is what stops you from fooling yourself six months in.

### Staged detection maturity

- **Stage A (M0–M1): descriptive only.** No classification. Emit features and thresholded bursts. Establish personal distributions. Goal: characterize the noise floor, not detect anything.
- **Stage B (M2): personalized thresholding.** Robust z-scores against a rolling personal baseline (median/MAD, not mean/σ — sleep features are heavy-tailed). Hysteresis: enter burst above θ_hi, exit below θ_lo. Minimum duration and refractory period. Fully transparent, fully inspectable, no training data required.
- **Stage C (M3, requires reference data): supervised discrimination.** Logistic regression / gradient boosting on windowed features, labelled by the reference device. Report **AUC with confidence intervals and a calibration curve**, on *held-out nights* — never held-out windows, which leak catastrophically through temporal autocorrelation.
- **Stage D (V2+): temporal modelling.** A 2–3 state HMM or semi-Markov model over feature emissions to exploit the fact that sleep states persist for tens of minutes. This is where real gains live, and it is cheap relative to deep learning. Reserve PyTorch for a sequence model only if Stage C clears AUC ≥ 0.75 and you hold ≥ 30 reference-labelled nights.

### Handling missingness honestly

For a side sleeper, `signal_unavailable` will be the *modal* state. The system must:
- report **effective monitoring coverage** (% of night with usable eye-region signal) on every dashboard, prominently;
- never impute across gaps;
- treat "no burst detected" during unavailability as **unknown**, not as **absent** — this distinction propagates into every downstream statistic.

---

## 12. Closed-Loop Cue Controller

### 12.1 State machine

```
 IDLE ──arm()──► ARMED ──lights out / user confirms──► SETTLING
                                                          │ (min 90 min,
                                                          │  configurable)
                                                          ▼
                                    ┌──────────────► MONITORING ◄────────┐
                                    │                    │               │
                                    │        gate stack passes           │
                                    │                    ▼               │
                                    │              CANDIDATE             │
                                    │       (activity accumulating,      │
                                    │        persistence timer)          │
                                    │            │           │           │
                                    │      decays│           │confirmed  │
                                    │            └──────────►│           │
                                    │                        ▼           │
                                    │                     CUEING         │
                                    │              (ramped playback)     │
                                    │                        │           │
                                    │                        ▼           │
                                    │            POST_CUE_OBSERVE (60–120s)
                                    │              │                │    │
                                    │      quiet   │                │ arousal
                                    │              ▼                ▼    │
                                    └────────── COOLDOWN      ABORT_AROUSAL
                                                   │                │    │
                                                   └────────────────┴────┘
                                                                        │
 MORNING ◄── wake detected / alarm / user ends session ◄─────────────────┘
```

Additional transitions: any state → `SUSPENDED` on quality collapse or camera loss; any state → `HALTED` on hard-cap breach or user abort. `HALTED` is terminal for the night and cannot be re-entered by policy.

### 12.2 The gate stack (all must pass; evaluated at 1 Hz)

```
G1  time_since_sleep_onset      ≥ min_delay      (default 5.5–6 h, per TLR)
G2  within_permitted_window     (e.g. not within 45 min of alarm)
G3  signal_quality              ≥ q_min
G4  body_motion_energy          ≤ m_max over trailing 60 s   ← camera guard
G5  no probable_arousal         in trailing 3 min
G6  cooldown elapsed            since last cue
G7  cues_tonight                < nightly_cap
G8  experiment condition        permits a cue tonight
G9  [V2 ONLY, gated on H1] eye_activity ≥ θ_personal for ≥ persist_ms
```

**In V1, G9 is absent and G4/G5 do all the sensing work.** The camera's contribution is *veto power*, not initiation. This is deliberate: a false negative from a veto costs one missed cue; a false positive from an initiator costs a woken user and a corrupted trial.

### 12.3 Cue delivery

- Ramped amplitude envelope (silence → target over 3–8 s). Abrupt onset is the primary awakening mechanism.
- Short duration (5–15 s), optional repetition with spacing, all under the nightly cap.
- Volume as a **calibrated relative scale** anchored during setup, with a **hard ceiling** the adaptive layer cannot exceed. Note honestly: without an SPL meter, absolute dBA at the pillow is unknown; the ceiling is set by user judgement during a waking loudness-calibration step and is conservative by default.
- **Digital gain only, never OS volume** — OS volume is global, racy, and can be changed by other processes.
- Every cue writes an immutable record before audio starts (so a crash mid-cue is still attributable).

### 12.4 Post-cue response and adaptation

Observation window 60–120 s. Outcome classified as:

| Outcome | Signal | Action |
|---|---|---|
| `quiet` | No motion increase | Normal cooldown; volume may increase one step next time (bounded). |
| `probable_arousal` | Motion spike, sub-threshold duration | Extend cooldown ×2; reduce volume one step. |
| `possible_awakening` | Sustained motion + posture change | **Stop cueing for the night.** Do not retry. |
| `uncertain` | Quality collapsed during window | No adaptation. Do not learn from unknowns. |

### 12.5 Adaptive policy by maturity stage

| Stage | Method | Why |
|---|---|---|
| **V1** | **Transparent heuristics + bounded step rules.** Hand-written, versioned, fully inspectable. | You have zero data. Any learner would be fitting noise. Heuristics also give the adaptive layer a legible baseline to beat. |
| **V1.5** | **Bayesian hierarchical logistic regression** over nightly outcomes; **Thompson sampling** across a *small discrete arm set* (3 volumes × 3 delays = 9 arms). | Handles the ~1–3 decisions/night, ~100-night data budget. Thompson sampling is near-optimal, trivially implementable with Beta/Normal conjugates, and explains itself. |
| **V2** | **Contextual bandit** with a handful of context features (time since onset, coverage, recent arousal history) via LinUCB or Bayesian logistic TS. | Only once you have evidence that context actually modulates response. Adding context prematurely fractures an already tiny dataset. |
| **V2+** | **HMM/HSMM** for state segmentation feeding the bandit's context. | Exploits temporal structure. Cheap, interpretable. |
| **Never (for this data scale)** | Deep RL. | Total lifetime decision count is ~300. Deep RL needs 10⁴–10⁶. It would be theatre. |

**Non-negotiable constraint on all adaptation:** every learned parameter is clipped to a pre-declared safe range, and the *safety* gates (G1–G7, hard caps) are **outside the learner's action space entirely**. A bandit must never be able to learn "louder is better" past the ceiling, or "skip the cooldown."

---

## 13. Calibration Design

Two distinct calibrations. Conflating them is a common design error.

### 13.1 Waking calibration (~15 min, supervised, one-off per setup change)

Guided capture with explicit prompts, each 20–30 s:

| Segment | Purpose |
|---|---|
| Eyes closed, still, supine | Noise floor. The most important segment. |
| Deliberate slow L↔R eye movement, eyes closed | **Positive control** — the closest available proxy for the target signal. If this is undetectable, H1 is dead and M1 can stop early. |
| Deliberate fast L↔R saccades, eyes closed | Upper bound on detectable amplitude. |
| Blinks | Confound to reject. |
| Small facial movements | Confound. |
| Slow head turn | Tests ROI stabilization. |
| Roll to left side / right side / prone | **Measures the visibility cliff for your actual sleep postures.** |
| Sit up / leave frame / return | Presence-detector behaviour. |
| Partial occlusion (blanket, arm, hair over face) | Robustness. |
| Two IR illumination levels | Exposure sensitivity. |

Outputs a `calibration_profile`: per-feature median/MAD baselines, positive-control effect size (**this is the go/no-go number**), per-posture visibility fractions, quality-floor thresholds, ROI geometry.

### 13.2 Sleep-baseline calibration (5–7 nights, unsupervised, no cueing)

Establishes overnight distributions under real conditions: nightly coverage, posture time-budget, feature quantiles by hour, spontaneous motion rate (the false-positive base rate for the arousal detector), and camera-drift frequency.

Personalized thresholds are set from **sleep** quantiles, never waking ones. Waking calibration answers *can this be seen at all*; sleep baseline answers *what is normal for this bedroom*.

### 13.3 Audio calibration (waking, ~3 min)

Ascending-limits loudness: user identifies faintest audible level at the pillow, comfortable level, and sets a hard ceiling. Default operating range is deliberately near the *low* end — the failure mode you care about is waking, not inaudibility.

---

## 14. Lucidity Training Design

The evidence-backed core. Do not let the CV work starve this.

**Cue asset.** A short (2–4 s), distinctive, non-startling, low-frequency-weighted sound. Registered with a content hash so that "trained cue" vs "untrained control cue" is verifiable in the data rather than asserted.

**Conditioning session** (evening and/or WBTB, 5–12 min, guided by the UI with audio narration so eyes can stay closed):

1. Play the cue at moderate waking volume.
2. **Critical-awareness prompt:** "How did you get to where you are right now?" — trace the last few minutes explicitly.
3. Memory review — the discontinuity test that actually discriminates dreams from waking.
4. Scan for impossible or incongruent details.
5. A physical reality check (nose-pinch breathing / text re-read), performed genuinely, not perfunctorily.
6. Replay the cue while *imagining hearing it inside a dream* and responding with recognition.
7. Rehearse the moment of becoming lucid, first-person, in vivid detail.
8. **Personal dream signs** (H5): the top recurring motifs mined from the journal, presented for visualization.
9. MILD intention statement, held into sleep onset.

**Instrumentation.** Log session duration, completion, per-step dwell time, self-rated engagement, and time-to-lights-out. Training adherence is a covariate in every analysis, and is the most likely explanation for a positive result that has nothing to do with the camera. Measure it or you cannot rule it out.

---

## 15. Experiment Design

### 15.1 Conditions

| Arm | Cue | Purpose |
|---|---|---|
| **A — Trained cue** | Conditioned cue, delivered per policy | Full intervention |
| **B — Untrained cue** | Acoustically matched, never trained | Isolates *conditioning* from mere sound-during-sleep |
| **C — No cue** | Nothing plays; everything else identical | Isolates *sound* from training + expectancy |

Training occurs on **all** nights, including C. Otherwise the arms differ in two variables at once and the design measures nothing.

### 15.2 Randomization and blinding

- Seeded PRNG, seed stored, **assignment sequence fully reproducible** and regenerable from the seed + night index.
- **Block randomization** (blocks of 6: 2A/2B/2C) — guarantees balance over short runs, which matters enormously at N-of-1 scale.
- Condition stored encrypted at rest; API refuses to serve it until `report.submitted_at IS NOT NULL`. Enforced at the DB/query layer, not in the UI.
- **Realistic limits of blinding, stated up front:** you are the developer, the participant, and the analyst. You will sometimes infer the condition — from waking mid-cue, from a hunch, from reading a log. Therefore: (a) collect an explicit *"what condition do you think last night was?"* item, so unblinding can be **measured** rather than assumed away; (b) pre-register the analysis before data collection begins; (c) automate analysis so it runs identically regardless of your expectations. This is the weakest methodological point in the entire project and pretending otherwise would be dishonest.

### 15.3 Outcomes

- **Primary:** binary lucid dream (yes/no) per night, by a pre-specified definition — *"At some point during a dream I was aware that I was dreaming."* Fixed in writing before night 1; never revised mid-study.
- **Secondary:** lucidity confidence (0–4); cue heard directly / incorporated indirectly / woke me; dreams recalled; vividness; sleep quality; awakenings.
- **Safety:** cue-attributed awakening rate; weekly sleep-quality trend.

### 15.4 Analysis

Pre-registered. Primary: Bayesian logistic mixed model on nightly binary outcome with condition as a fixed effect and night index as a covariate (to absorb practice effects, which are large and confounding in lucid-dream training). Report posterior odds ratios with credible intervals — **not p-values**, which are near-meaningless at this sample size. Randomization tests as a secondary, assumption-light check.

**Sample size, honestly.** With a ~0.10/night baseline and a target of ~0.25/night, detecting the difference with reasonable confidence needs roughly **100+ nights per arm** — i.e. **~9–12 months of nightly compliance for a three-arm design.** This is the largest single risk to the project, larger than any technical risk in this document. Consider dropping to two arms (A vs C) to halve it, accepting that you then cannot separate conditioning from sound.

---

## 16. Data Model

SQLite, WAL mode. All timestamps UTC ISO-8601 with an explicit monotonic offset column for intra-night intervals.

```sql
sessions(id, started_at, ended_at, device_profile_id, calibration_profile_id,
         config_snapshot_id, experiment_id, night_index, status, notes)

frames_1hz(session_id, t_mono, t_utc,
           signal_quality, face_present, landmark_available,
           yaw, pitch, roll, head_motion, body_motion,
           eye_flow_l, eye_flow_r, eye_flow_bilateral_corr,
           lid_disp_l, lid_disp_r, resp_proxy, coverage_flag)
           -- ~28,800 rows/night. Trivial for SQLite. NO PIXELS EVER.

events(id, session_id, t_utc, kind /* CLOSED ENUM, §11 */,
       confidence, duration_ms, features_json, detector_version)

cues(id, session_id, t_utc, cue_asset_id, gain, ramp_ms, duration_ms,
     repetition_index, policy_version, gate_snapshot_json,
     scheduled_or_triggered)

cue_outcomes(cue_id, window_s, outcome /* quiet|probable_arousal|
             possible_awakening|uncertain */, motion_delta,
             latency_to_motion_ms, quality_during_window)

reports(session_id, submitted_at, narrative_encrypted, lucid_binary,
        lucid_confidence, knew_was_dreaming, cue_heard,
        cue_indirect, cue_woke_me, dreams_recalled, vividness,
        sleep_quality, guessed_condition, notes_encrypted)

experiments(id, name, seed, design_json, arms_json, preregistration_md,
            started_at, ended_at)

assignments(experiment_id, night_index, condition_encrypted,
            revealed_at, block_index)

calibration_profiles(id, created_at, device_profile_id,
                     baselines_json, thresholds_json,
                     positive_control_effect_size,
                     posture_visibility_json)

training_sessions(id, session_id, kind /* evening|wbtb */, started_at,
                  completed, steps_json, engagement_rating)

cue_assets(id, path, sha256, trained /* bool */, created_at, spectral_json)

device_profiles(id, camera_model, resolution, fps, ir_wavelength_nm,
                mount_geometry, audio_device, created_at)

config_snapshots(id, created_at, config_json, git_sha)

model_versions(id, kind, trained_at, params_json, metrics_json,
               training_data_hash)
```

**Design commitments:** every derived artifact references the `config_snapshot` and `detector_version`/`policy_version` that produced it, so a mid-study parameter change is visible in analysis rather than silently poisoning it. Narrative and condition fields are encrypted at rest.

---

## 17. Backend Architecture

Two processes, deliberately separated by criticality:

```
morpheus-daemon (critical path — must never depend on the UI)
 ├─ CaptureService      camera I/O, fixed exposure, frame timestamps
 ├─ VisionPipeline      §10 stages, emits 1 Hz FeatureFrame
 ├─ FeatureStore        ring buffer + batched SQLite writes
 ├─ Detector            burst / arousal detection, versioned
 ├─ CueController       state machine, gate stack, hard caps
 ├─ AudioPlayer         in-process PortAudio, ramped envelope
 ├─ SafetySupervisor    independent watchdog; can HALT the controller
 └─ TelemetryPublisher  best-effort WebSocket fan-out (droppable)

morpheus-api (FastAPI — non-critical)
 ├─ REST: sessions, reports, calibration, experiments, config, export
 ├─ WS:   /ws/telemetry (read-only)
 └─ Static: React bundle
```

**Key interfaces** (stable seams for swapping hardware and algorithms):

```python
class FrameSource(Protocol):
    def read(self) -> Frame | None: ...          # webcam | Pi | file replay
class LandmarkProvider(Protocol):
    def landmarks(self, frame) -> LandmarkSet | None: ...   # MediaPipe | other | None
class FeatureExtractor(Protocol):
    def update(self, frame, lm) -> FeatureFrame: ...
class Detector(Protocol):
    def step(self, ff: FeatureFrame) -> list[Event]: ...
class Policy(Protocol):
    def decide(self, state, ff, history) -> CueDecision: ...  # heuristic | bandit
class AudioSink(Protocol):
    def play(self, asset, gain, ramp_ms) -> CueRecord: ...
class ReferenceSource(Protocol):
    def epochs(self) -> Iterable[ReferenceEpoch]: ...       # Muse | ZMax | manual
```

`FrameSource` accepting a **recorded file** is not optional — it is what makes the entire system testable offline and lets you replay a night against a new detector. Build it in M0.

**SafetySupervisor** runs independently of the policy and enforces §24 caps. If the policy and the supervisor disagree, the supervisor wins and the disagreement is logged as a defect.

---

## 18. Frontend Architecture

React + TypeScript + Vite. Four surfaces, in build order:

1. **Journal / Morning Report** — the highest-value screen and the first one built, because it is required by *every* phase including a camera-free one. Fast, keyboard-driven, offline-capable, autosaving. Narrative capture must be frictionless at 06:00 or your primary outcome data will be garbage.
2. **Calibration Wizard** — guided segments with live preview, per-segment quality feedback, and a clear pass/fail on the positive-control test.
3. **Live Monitor** — read-only telemetry: quality, coverage, posture, motion, event stream, gate-stack state. Explicitly labelled *"observation only — closing this page does not affect the night."*
4. **Analysis Dashboard** — per-night timelines, coverage statistics (prominent), cue/outcome overlays, and the N-of-1 comparison with credible intervals. **Condition-blind by default**; unblinding requires an explicit action that is itself logged.

Design rule: every number displayed carries its uncertainty and its coverage. A lucidity rate computed over 40% coverage nights must *say so on the same line*.

---

## 19. Real-Time Communication

- **WebSocket at 1–2 Hz**, JSON, one `telemetry` message type carrying the aggregated feature frame + state. Lossy by design: if no client is connected, messages are dropped, not queued.
- **Backpressure:** bounded queue, drop-oldest. A slow browser must never stall the vision pipeline.
- **Control commands** (arm, disarm, abort) go over REST with explicit confirmation, not over the telemetry socket.
- **Abort is also available out-of-band** — a keyboard interrupt on the daemon and a physical speaker mute. Never make stopping the system depend on software you wrote at 2am.

---

## 20. Privacy and Local Storage

Sleep video of a person in their bedroom is among the most sensitive data a hobby project can generate. Treat it accordingly.

- **No raw video is persisted, ever, by default.** Frames are processed in memory and discarded. This is enforced structurally: no code path in the daemon writes image data, and a test asserts it.
- **Calibration clips** are the sole exception: explicitly consented, retained only if the user opts in, encrypted, auto-expiring (default 30 days), with a one-click purge.
- **All data local.** No cloud, no telemetry, no crash reporting, no model API calls containing dream narratives. The proprietary licence reinforces the intent, but the guarantee is structural rather than legal: the code contains no network egress path for recorded data.
- **Encryption at rest** for narratives and condition assignments; SQLite file on an encrypted volume (FileVault is a prerequisite, documented in setup).
- **Bed partners:** you sleep alone, so this is moot today — but the camera must refuse to run if it detects a second person during framing check, or at minimum surface an explicit consent prompt. Recording another person while they sleep, without consent, is not a design detail.
- **Export and delete** are first-class: full JSON/CSV export, and a genuine hard-delete (VACUUM, not tombstones).
- **Dream narratives are psychologically sensitive.** They can contain trauma, sexual content, and information about third parties. Never route them through an external LLM or any network service without an explicit, per-use, informed decision.

---

## 21. Testing Strategy

| Layer | Approach |
|---|---|
| **Unit** | Feature math against synthetic signals with known ground truth; threshold hysteresis; MAD/quantile estimators under heavy tails. |
| **State machine** | **Property-based tests (Hypothesis)** over random event sequences asserting invariants: cue count never exceeds cap; no cue during cooldown; `possible_awakening` always terminates cueing; `HALTED` is absorbing. This is where the highest-value tests live. |
| **Replay** | `FrameSource` from recorded video/feature logs. Every detector change is re-run over the full night corpus. **The regression corpus is the project's most valuable asset** — protect it. |
| **Synthetic nights** | Generated feature streams with injected bursts and arousals at known times → measures detector sensitivity/specificity/latency without needing sleep. |
| **Fault injection** | Camera unplugged, USB reset, exposure change, frame drops, disk full, clock jump/DST, audio device disappearing, process kill mid-cue. **All must fail safe (silent), never fail loud.** |
| **Soak** | 8-hour unattended runs before any night with cueing enabled. Memory, thread, and FD leak checks. A leak that OOMs at hour 6 is invisible in a 10-minute test. |
| **Safety** | Dedicated suite asserting hard caps hold under adversarial policy outputs (including a deliberately malicious mock policy that demands maximum volume continuously). |
| **Analysis** | Randomization/blinding logic tested for reproducibility from seed; a test asserts the API *cannot* return an unrevealed condition. |

**Testing rule with teeth:** no cueing is enabled on a real night until the soak test and the full safety suite pass on that exact commit. The daemon refuses to arm if `git` is dirty relative to the last passing test run.

---

## 22. Validation Plan

Every phase has an explicit, numeric, pre-committed go/no-go. These numbers are set *now*, before data exists, precisely so they cannot be rationalized later.

| Phase | Question | Metric | Pass |
|---|---|---|---|
| **M0** | Does the rig survive a night, and can the camera see anything? | Uptime; frames captured; % of night `face_present`; % with usable eye ROI; fan/thermal events | ≥ 95% uptime, zero crashes, **eye-region coverage ≥ 25%** |
| **M1** | Is the target signal above the noise floor? | Positive-control (deliberate closed-eye saccades) detection: **d′ or AUC vs. eyes-closed-still** | **AUC ≥ 0.80 waking.** If the *deliberate, supervised, awake* case fails, H1 is dead. |
| **M2** | Does the scheduled TLR baseline work, safely? | Cue delivery reliability; cue-attributed awakening rate; sleep-quality trend | ≥ 95% cues fire as specified; **awakening rate < 15%**; no downward sleep-quality trend over 2 weeks |
| **M3** | **Does the camera index track REM?** (H1) | AUC vs. reference EEG staging, on **held-out nights** | **AUC ≥ 0.70** → proceed to V2. **< 0.65** → kill the CV cueing branch. |
| **M4** | Does sensor timing beat scheduled timing? (H2) | Within-subject A/B, lucidity rate | Posterior P(sensor > scheduled) ≥ 0.90 |
| **M5** | Does the whole system beat no-cue? | 3-arm N-of-1, ≥ 100 nights/arm | Credible interval on OR(A vs C) excluding 1 |

**On M3:** it requires a reference device. Given your side-sleeping, headband comfort is a genuine acquisition risk — budget for the possibility that you cannot tolerate wearing one, in which case **M3 is unachievable and the CV branch must be permanently confined to shadow mode.** That is an acceptable outcome; an unvalidated detector driving cues is not.

---

## 23. Failure Modes

### Technical

| Failure | Detection | Response |
|---|---|---|
| Camera lost / unplugged | Frame timeout | Suspend cueing, log, attempt reconnect, never cue blind |
| Auto-exposure re-enabled by OS | Luminance variance spike | Flag night as degraded, exclude from analysis |
| Camera bumped mid-night | Global scene-change detector | Invalidate ROI, re-acquire, mark discontinuity |
| Head motion mistaken for eye motion | Bilateral-correlation check; registration residual | Require conjugate motion; discard high-residual windows |
| Machine sleeps | Monotonic-clock gap | Abort night, mark invalid |
| Audio device changes | Device enumeration check pre-cue | Refuse to cue |
| Crash mid-cue | Cue record written pre-playback | Recoverable and attributable |
| Silent detector regression | Replay corpus in CI | Block the change |

### Project-level stop/pivot conditions

State these now so that stopping is a *planned outcome*, not a failure of nerve:

1. **M1 fails (AUC < 0.80 on deliberate waking saccades).** → Delete the eye-movement branch. Keep camera as motion guard. Morpheus becomes an adaptive scheduled-TLR research platform. *This is still a good product.*
2. **M0 coverage < 15%.** → Camera cannot gate reliably either. Consider a second camera angle, or drop the camera entirely and pivot to schedule + wearable motion.
3. **M3 AUC < 0.65.** → Publish the negative result. Freeze CV in shadow mode. Do not "tune until it works" — that is how you fit noise.
4. **Cue-attributed awakening rate > 15% at minimum audible volume.** → Cueing is net-harmful. Stop and reconsider modality.
5. **Sleep quality declines for 7 consecutive nights.** → **Automatic hard stop.** Non-negotiable. The system should enforce this itself, not wait for judgement.
6. **Compliance below ~70% of nights by week 6.** → The study will never reach power. Reduce to two arms or redefine as an engineering project with no efficacy claim.
7. **Positive results appear only when you know the condition.** → Blinding failed. Discard the efficacy claim.

---

## 24. Safety Constraints

Hard-coded, enforced by `SafetySupervisor`, **outside the reach of any adaptive policy**:

- **Absolute volume ceiling**, user-set during waking calibration, never exceedable.
- **Mandatory amplitude ramp** — no cue may begin above silence.
- **Nightly cue cap** (default ≤ 6), and a **per-hour cap** (default ≤ 2).
- **Minimum cooldown** (default ≥ 20 min), extended on any arousal.
- **No cueing before minimum sleep delay** (default ≥ 5.5 h) — protects the deep-sleep-dominant early night.
- **Immediate cessation for the night** on `possible_awakening`.
- **Physical override:** the speaker's own volume/power is the final authority, documented in setup.
- **Weekly sleep-quality trend monitor** with automatic suspension.
- **Exclusion criteria, documented and self-assessed before starting:** diagnosed sleep disorder (apnea, narcolepsy, parasomnia, REM behaviour disorder), epilepsy or photosensitivity, psychiatric conditions in which sleep fragmentation or dissociative experience carries risk, current sleep deprivation, safety-critical occupation. Lucid-dream induction deliberately fragments sleep architecture; that is not free.
- **No electrical stimulation. No pharmacology. No supplements.** Out of scope, permanently.
- Morpheus displays a persistent, non-dismissible statement that it does not measure sleep stages and cannot confirm REM.

---

## 25. Development Roadmap

Effort in solo-developer hours, assuming ~10–15 h/week.

| Phase | Scope | Effort | Calendar |
|---|---|---|---|
| **M0 — Feasibility probe** | Capture service, quality metrics, presence detection, 1 Hz feature logging, SQLite schema, replay `FrameSource`, macOS power handling, overnight soak. **No cueing, no detection.** | **30–45 h** | 2–3 wks |
| **M1 — Calibration + signal characterization** | Calibration wizard, waking protocol, positive-control test, ROI stabilization, bilateral flow, threshold estimation, coverage analytics. **Go/no-go on H1.** | **40–60 h** | 3–4 wks |
| **M2 — Scheduled TLR + safety + journal** | Conditioning trainer, cue assets, audio player, state machine, gate stack, SafetySupervisor, morning report, property tests, soak. **First real cueing nights.** | **60–80 h** | 4–6 wks |
| **M3 — Experiment harness** | Seeded block randomization, encrypted assignment, blinding enforcement, pre-registration doc, analysis pipeline, dashboards. | **40–55 h** | 3–4 wks |
| **M4 — Reference integration & CV validation** | Reference ingest (Muse/ZMax), epoch alignment, supervised classifier, held-out-night AUC, calibration curves. **Go/no-go on H1 proper.** | **50–70 h** | 4–6 wks |
| **M5 — Adaptive layer** | Thompson sampling over discrete arms, bounded parameter updates, policy versioning, counterfactual logging. | **35–50 h** | 3–4 wks |
| **M6 — Sensor-timed cueing (gated on M4)** | G9 promotion, A/B vs. scheduled, HMM temporal smoothing. | **40–60 h** | 4–5 wks |
| **M7 — Longitudinal study** | ~100 nights/arm, analysis, write-up. Mostly *waiting*, not coding. | **20 h + 9–12 months** | — |

**Total to a defensible answer: ~300–420 engineering hours plus roughly a year of nights.** The nights, not the code, are the schedule.

---

## 26. Hardware and Cost Estimate

**Start here (~$60), because M0/M1 can invalidate everything downstream:**

| Item | Cost | Notes |
|---|---|---|
| USB IR camera w/ manual exposure + 850 nm illuminator | $35–60 | Manual exposure control is **mandatory**, not a nice-to-have. Verify before buying. |
| Adjustable mount / gooseneck arm | $15–25 | Rigidity matters more than the camera. |
| Bedside speaker (likely owned) | $0–40 | Any small powered speaker with a physical volume knob. |
| Mac laptop | owned | — |
| **M0/M1 subtotal** | **~$50–125** | |

**Buy only if M1 passes:**

| Item | Cost | Notes |
|---|---|---|
| Muse S (reference EEG) | ~$350–400 | Credible REM κ. **Comfort risk for side sleepers** — behind-ear electrodes against the pillow. |
| Raspberry Pi 5 + NoIR cam + PSU + case | ~$120–160 | Removes every macOS failure mode in §8. Recommended once software stabilizes. |
| Second camera (alternate angle) | $35–60 | Only if M0 shows coverage is posture-limited rather than fundamentally absent. |

**Only for a serious validation effort:**

| Item | Cost | Notes |
|---|---|---|
| Hypnodyne ZMax | ~$1,000–1,800 | Real EOG-adjacent channels, raw real-time API, Dreamento-compatible. The correct instrument if you commit to M4. |
| SWIR camera | $3,000+ | The *only* way to actually image the pupil through a lid. Almost certainly out of scope — noted so the cheap options are not mistaken for it. |

**Recommended spend path given "whatever the evidence justifies": ~$60 now. ~$400 more only after M1 passes. ZMax only if you commit to publishing.**

---

## 27. Prioritized Backlog

**P0 — required for any version to be meaningful**
1. SQLite schema + migrations + config snapshotting
2. `FrameSource` abstraction incl. **file replay** (testability foundation)
3. Capture service with fixed exposure and frame timestamping
4. Signal-quality metrics + coverage accounting
5. Presence detection + gross body-motion energy
6. 1 Hz feature aggregation and persistence
7. macOS power-management handling + overnight soak harness
8. Morning report UI + schema *(needed by every phase)*
9. Conditioning trainer + cue asset registry with hashing
10. Audio player with ramped envelope and hard ceiling
11. Cue controller state machine + gate stack
12. SafetySupervisor + hard caps + property-based tests
13. Seeded block randomization + blinding enforcement at the query layer
14. Pre-registered analysis document *(write before night 1)*

**P1 — the research programme**
15. Calibration wizard + positive-control test
16. Eye ROI tracking + stabilization + bilateral flow *(shadow mode)*
17. Coverage & posture analytics dashboard
18. Reference-device ingest + epoch alignment
19. Supervised classifier + held-out-night evaluation
20. N-of-1 Bayesian analysis + dashboards
21. Replay regression corpus in CI

**P2 — adaptation and depth**
22. Thompson sampling over discrete arms
23. Arousal-aware volume/cooldown adaptation
24. Journal-mined personal dream signs (H5)
25. HMM temporal smoothing
26. Raspberry Pi deployment target

**P3 — deferred, justified only by evidence**
27. rPPG respiration/heart-rate extraction
28. Sequence models (PyTorch) — *only* after M4 passes with ≥ 30 labelled nights
29. Multi-camera fusion
30. Contextual bandits

---

## 28. First Milestone Specification — M0: Overnight Feasibility Probe

**Thesis: spend two weeks finding out whether the camera can see your face at all, before spending four months building on the assumption that it can.**

**Explicitly out of scope:** cueing, audio, detection, thresholds, classification, adaptation, experiments.

**Deliverables**
- `morpheus-daemon` skeleton: `CaptureService` → `VisionPipeline` (quality + presence + motion only) → `FeatureStore`.
- `FrameSource` protocol with `WebcamSource` and `FileReplaySource`.
- Fixed-exposure camera configuration with verification-on-start (refuse to run if auto-exposure cannot be disabled).
- Quality metrics: luminance stats, focus, saturation, frame-drop rate, scene-change detection.
- Presence: face detected / not; bbox; coarse yaw estimate; eye-region-usable flag.
- Motion: whole-frame and bed-region energy at 1 Hz.
- SQLite `sessions` + `frames_1hz` with migrations.
- macOS sleep prevention, verified across a full night.
- CLI: `morpheus record --hours 8`, `morpheus report <session_id>`.
- Overnight soak test + a fault-injection test unplugging the camera mid-run.
- A one-page coverage report: uptime, `face_present` %, eye-region-usable %, per-posture breakdown, thermal/fan events.

**Acceptance criteria (5–7 consecutive nights)**
- Zero crashes; ≥ 95% frame-capture uptime.
- Memory flat over 8 hours.
- No macOS sleep interruptions.
- Coverage report generated automatically.
- **Decision gate: eye-region-usable coverage ≥ 25% of the night.** Below 15% → the camera cannot support eye-based detection for you; proceed to M2 (scheduled TLR) with the camera as a motion guard only, and mark H1 unreachable.

**Effort: 30–45 hours.**

**Why this is the right first move.** It is the cheapest possible test of the assumption most likely to be false, it builds infrastructure (schema, replay, soak, capture) that every later phase needs regardless of outcome, and it produces a real number — your personal eye-region coverage — that no amount of reasoning from this document can substitute for.

---

## 29. Essential Questions

Four were resolved: you sleep alone, are a **side/stomach sleeper**, will buy hardware the evidence justifies, prioritize **research rigor**, and will start on a **Mac laptop**. Remaining, in priority order:

1. **Will you actually wear a head-mounted EEG reference?** M3/M4 depend entirely on it, and side sleeping makes headbands uncomfortable. If the answer is no, say so now — the CV branch is then permanently shadow-mode and the roadmap shortens considerably.
2. **Can you commit to nightly morning reports for 6–12 months?** Below ~70% compliance the efficacy study is unreachable. If not, we should redefine the goal as an engineering/feasibility project with no efficacy claim, which is entirely respectable.
3. **Two arms or three?** Three (trained / untrained / no-cue) answers more but roughly doubles the calendar. My recommendation: **start with two (A vs C)**, add arm B only if A beats C.
4. **Is publication a goal?** It changes pre-registration rigor and possibly the hardware tier (ZMax vs Muse). Worth deciding early, since pre-registration must precede night 1.
5. **What is your current lucid-dream baseline?** If it is already high, effects are harder to detect; if zero, expect a long ramp. Two weeks of journal-only baseline before M2 would be well spent, and can run concurrently with M0/M1 coding.

---

## 30. Final Recommendation

**Build Morpheus. Do not build the system you described.**

The proposal treats the camera as the centre and the protocol as a feature. The evidence inverts that: the protocol has replicated efficacy with *no sensing whatsoever*, while camera-based eye-movement detection through closed lids is unvalidated, physically constrained by silicon sensor cutoff, and — for a side/stomach sleeper — likely to have low duty cycle. Existing attempts (Lucid Scribe, INSPEC) never published validation, which is itself informative.

So: make the scheduled TLR engine plus the blinded N-of-1 harness the product, make the camera a motion guard and a *shadow-mode research subject*, and put a hard validation gate between the camera and the cue decision. Under this design:

- If the camera works, you have the first validated contactless lucid-dream cueing system, with the control arm already built to prove it.
- If the camera does not work, you have a rigorous adaptive TLR platform, an open-source N-of-1 harness that dream researchers can reuse, and a documented negative result on a question two products have quietly ducked.

Both outcomes are worth the year. Only one of them is available if you build the camera first.

**Start with M0. Two weeks, ~$60, and one number — your eye-region coverage — that decides the shape of everything after it.**

---

### Sources

- [Provoking lucid dreams at home with sensory cues paired with pre-sleep cognitive training (Consciousness & Cognition, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S1053810024001260) · [PDF](https://faculty.wcas.northwestern.edu/paller/C&C24.pdf) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/39278157/) · [summary](https://www.psypost.org/lucid-dreaming-app-triples-users-awareness-in-dreams-study-finds/)
- [Investigating dreams by strategically presenting sounds during REM sleep (Neuropsychologia, 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0028393225001642)
- [Touchless short-wave infrared imaging for pupillometry and gaze estimation in closed eyes (Communications Medicine, 2024)](https://www.nature.com/articles/s43856-024-00572-1)
- [Contactless Camera-Based Sleep Staging: The HealthBed Study (Bioengineering, 2023)](https://www.mdpi.com/2306-5354/10/1/109)
- [SleepVST: Sleep Staging from Near-Infrared Video Signals (2024)](https://arxiv.org/pdf/2404.03831) · [Deep Learning-Enabled Sleep Staging from NIR Video (2023)](https://arxiv.org/abs/2306.03711) · [Video-PSG (2024)](https://pubmed.ncbi.nlm.nih.gov/39405136/)
- [Portable Devices to Induce Lucid Dreams—Are They Reliable? (Frontiers in Neuroscience, 2019)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6517539/)
- [Assessing a portable EEG sleep monitor against level-1 polysomnography](https://pmc.ncbi.nlm.nih.gov/articles/PMC12782022/)
- [Dreamento: an open-source dream engineering toolbox for sleep EEG wearables (SoftwareX, 2023)](https://www.sciencedirect.com/science/article/pii/S2352711023002911)
- [awesome-lucid-dreams (prior-art index)](https://github.com/IAmCoder/awesome-lucid-dreams)
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
