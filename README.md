# Morpheus

A local-first, single-user research platform for testing whether auditory cues
can increase lucid-dream frequency and, just as importantly, for finding out
honestly when they cannot.

**Status: M0-M6 built, partly gated.** Conditioning, overnight cueing under hard
safety limits, morning reports, a blinded randomized N-of-1 harness with
pre-registration, an adaptive bandit, calibration, M1 eye tracking, and
reference validation are all built.

Two milestones are built but **cannot run yet**. M4 needs a reference EEG device
you may not own; M6 (sensor-timed cueing) is hard-locked behind M4 passing and
raises at construction without it. That lock has no override see
`cue/sensor_timing.py` for why.

Morpheus still does not detect REM, and does not claim to.

The full design, including the evidence review that shaped it, is in
[`docs/design.md`](docs/design.md). Read §1–§8 before the code; the architecture
is deliberately not the obvious one and the reasoning matters.

---

## What Morpheus does not do

Stated first, because the honesty is the point of the project:

- It **does not measure sleep stages** and cannot confirm REM sleep. A silicon
 camera cannot see the pupil through a closed eyelid the published work that
 achieves this uses short-wave infrared (0.9–1.7 μm) on InGaAs sensors.
- It **does not guarantee** lucid dreams, or claim that any observed movement
 indicates dreaming.
- It **never records video.** Frames are processed in memory and discarded;
 only derived per-second features reach the disk.
- It uses **no electrical stimulation** and makes **no pharmacological
 recommendations**. Both are permanently out of scope.

The naming rules behind the first point are enforced by
[`tests/test_naming_discipline.py`](tests/test_naming_discipline.py), which
fails the build if an identifier asserts something the system cannot establish.

## Why the camera is not the centre

The evidence-backed component of this system is the cue-conditioning protocol,
not the sensing. Targeted Lucidity Reactivation has raised lucid-dream frequency
in controlled trials using *purely scheduled* audio cues with no sleep sensing
at all. Meanwhile, camera-based sleep staging tops out near 73% accuracy, and
gets there from cardiorespiratory signals and gross motion not eye movement.

So Morpheus inverts the obvious design. The camera's first job is **gate and
guard**: suppress a cue while the body is moving, and detect probable arousal
after one. Both work at any pose, in the dark, under blankets. Eye-movement
detection is developed in parallel but runs in **shadow mode**, forbidden from
influencing cue timing until it clears a pre-committed validation gate against a
reference signal. See `docs/design.md` §7–§8.

## M0: the feasibility probe

M0 exists to answer one question cheaply, before four months of work assume the
answer: **what fraction of the night is this sleeper's eye region actually
visible?**

The decision gate was fixed in the design document before any data existed:

| Coverage | Verdict | Consequence |
|---|---|---|
| ≥ 25% | PASS | Develop the eye-movement branch in M1, in shadow mode |
| 15–25% | MARGINAL | Collect more nights. Do not tune the thresholds |
| < 15% | FAIL | Abandon the eye branch; camera becomes a motion guard only |

## M1: the positive-control gate

`morpheus calibrate` runs a second, harder gate. It compares **deliberate
closed-eye eye movements against closed-eye stillness**, in one session, and
requires **AUC ≥ 0.80**.

That bar is deliberately soft for a deliberately easy task. You are awake,
cooperative, well lit, frontal, close to the camera, holding your head still,
and making the largest eye movements you can. Every one of those conditions is
better than what a sleeping face offers. Missing 0.80 does not mean the
thresholds need tuning it means the signal is not there, and the branch ends.

The profile also reports **head-turn leakage**. If head turns separate from
baseline about as well as saccades do, the index is tracking your head rather
than your eyes, and a passing AUC means nothing.

A FAIL is a real result, not a setback. It costs two weeks instead of four
months, and it is the outcome the design considers most likely for a side
sleeper.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/morpheus setup-models  # fetches YuNet, verified by SHA-256
```

Camera and microphone permissions on macOS attach to the *binary*, not the
terminal that launched it. If the daemon fails to open the camera after a
relaunch, that is usually why.

## Use

```bash
# Journalling start here, tonight. No hardware needed.
morpheus journal      # this morning's dream report
morpheus baseline      # lucid-dream rate over recent reports

# Cueing
morpheus cues --add-preset trained-ascending --trained
morpheus cues --add-preset control-descending --no-trained
morpheus cues --preview 1   # hear it at safe gain while awake
morpheus train       # conditioning protocol, before sleep or at WBTB
morpheus night --dry-run   # decide and log cues, play no audio
morpheus night       # a real cueing night

# Camera probe and calibration (M0-M1)
morpheus doctor      # verify the rig before trusting it with a night
morpheus calibrate     # guided ~15 min session; the H1 positive control
morpheus calibrate --audio   # ascending-limits loudness calibration
morpheus record --hours 8   # features only, never video
morpheus report      # coverage analysis and decision-gate verdict
morpheus sessions

# Experiment (M3)
morpheus experiment --create pilot --design two-arm
morpheus prereg --experiment 1 # required before collection can start
morpheus experiment --start 1
morpheus reveal      # last night's arm, only after the report exists
morpheus analyse      # pre-registered Bayesian comparison

# Validation and adaptation (M4-M6)
morpheus validate hypnogram.csv --session 12 # H1 go/no-go
morpheus adaptive          # bandit posteriors
```

### The order that matters

1. **Journal for two weeks first.** You need a pre-intervention baseline, and
 dream recall itself improves with practice. Without this the N-of-1
 comparison later has nothing to compare against.
2. **Condition the cue** (`morpheus train`) this is the part with published
 efficacy. The sound is inert without it.
3. **Run `--dry-run` first**, to see when cues *would* fire before any sound is
 made in your bedroom.

`morpheus night` defaults are deliberately conservative: no cue for the first
5.5 hours, six per night maximum, two per hour, twenty-minute cooldown, a
mandatory fade-in from silence, and a hard volume ceiling. It stops cueing for
the night entirely on a possible awakening.

`doctor` is not optional before the first overnight run. It checks three things
that fail silently otherwise:

- **manual exposure**, by property read-back *and* by an empirical luminance
 drift probe, because OpenCV's read-back can report success on a backend that
 changed nothing. Auto-exposure hunting in a dark room appears as motion in
 every downstream feature;
- **the quality distribution** against the configured floor, because the quality
 gate also gates face detection a mis-tuned floor produces a zero-coverage
 night that looks exactly like a genuine finding;
- **the sleep assertion**, because a laptop that suspends at 02:00 leaves an
 invisible hole rather than a short night.

### Hardware

M0 needs roughly $60: a USB IR camera **with manual exposure control** (verify
before buying the built-in FaceTime camera does not have it), an 850 nm
illuminator, a rigid mount, and a bedside speaker. Nothing else should be bought
until M1 passes. Full costing in `docs/design.md` §26.

## Development

```bash
.venv/bin/pytest -q

# Soak harness required before the first cueing night, and worth running
# before the first overnight probe. Loops a clip for the full duration and
# reports peak RSS, so a leak that only surfaces at hour six is visible here
# rather than at 04:00.
morpheus record --replay clip.mp4 --loop --hours 8
```

The suite covers the coverage gate exhaustively (it produces the number that
decides the project's direction), the aggregation arithmetic, and fault
injection for every overnight failure mode: camera loss, USB reset, clock gaps
from system suspend, and mid-run crashes. The invariant is always the same 
**fail quiet, never loud.**

One thing the tests deliberately cannot cover: YuNet does not fire on synthetic
imagery, so face detection itself is exercised through an injected scripted
detector. The real detector must be validated against live footage during M0
setup. `morpheus doctor` is where that happens.

### Layout

```
src/morpheus/
 types.py   value types; the closed EventKind enum
 config.py   typed config + snapshotting (provenance for every artifact)
 capture/   FrameSource protocol, live camera, file replay
 vision/   quality, presence, motion, 1 Hz aggregation
 store/    SQLite schema, migrations, batched writes
 runtime/   sleep prevention, the record loop, the cueing night
 analysis/   the M0 coverage report
 audio/    cue synthesis, hashed asset registry, ramped playback
 cue/    safety supervisor, gate stack, state machine, policy
 training/   the conditioning protocol
 report/   morning report schema and store
```

### How the cue path is layered

From least to most authority a cue happens only if every layer agrees, and any
one of them can stop it alone:

| Layer | Role |
|---|---|
| `cue/policy.py` | Proposes gain and timing. Knows nothing about limits. |
| `cue/controller.py` | Gate stack: motion, quality, caps, cooldown, condition. |
| `cue/safety.py` | Hard caps. Cannot be overridden by anything. |
| `audio/player.py` | Clamps to the ceiling and forces a fade-in, in the buffer. |

That separation is structural, not stylistic. When the adaptive layer arrives in
M5 it slots in at the `Policy` seam, where it has no route to the safety limits 
so a learner optimising for "cue noticed" cannot discover that louder is better.

`FileReplaySource` matters more than its size suggests: it is what lets a
detector change in M4 be re-run against every night recorded since M0. The
corpus of recorded nights is the project's most valuable asset, and the seam
that makes it reusable had to exist before the first night was recorded.

## Privacy

Sleep footage of a person in their bedroom is about as sensitive as hobby-project
data gets.

- No raw video is persisted. A test asserts the daemon contains no image-writing
 call, and another asserts a full run leaves no media files behind.
- All data stays local. No cloud, no telemetry, no crash reporting. Dream
 narratives (from M2) must never be routed through an external service.
- Run on an encrypted volume. FileVault is a documented prerequisite.
- The database and any recorded media are gitignored.
