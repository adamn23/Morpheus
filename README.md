# Morpheus
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-experimental-orange)
![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red)

Morpheus is a local, single-user research tool for testing whether audio cues
played during sleep can make lucid dreams more frequent. It runs on a laptop
with a USB infrared camera and a bedside speaker.

The cue protocol is based on Targeted Lucidity Reactivation (TLR): a sound is
paired with a short self-awareness exercise before sleep, then replayed later
in the night. TLR has worked in lab studies with purely scheduled cues and no
sleep sensing at all, so the schedule is the baseline here. The camera has to
prove it can do better before it is allowed to affect cue timing.

## Status

Milestones M0 to M6 are implemented. M4 validates the camera against a
reference sleep tracker (an EEG headband), and M6, sensor-timed cueing, is
locked until M4 passes. `cue/sensor_timing.py` raises on construction without
a passing validation result, and there is no override.

## Limitations

- It does **not** detect REM or any sleep stage. A normal camera can't see the
  pupil through a closed eyelid; the published work that does this uses
  short-wave infrared (0.9-1.7 μm) InGaAs sensors. What a webcam can pick up is
  the eyelid surface moving.
- It does not claim that any movement means someone is dreaming.
- It **never records video**. Frames are processed in memory and only
  per-second features are written to disk.
- No electrical stimulation and no drug recommendations. Both are out of scope.

`tests/test_naming_discipline.py` fails the build if a variable or column name
claims something the system can't measure, such as `is_rem` or `sleep_stage`.

## Design

Camera-based sleep staging reaches about 73% accuracy at best, and mostly from
breathing and body motion rather than eye movement. So the camera's main job
here is to gate and guard:

- hold back a cue while the body is moving
- detect a likely awakening after a cue, and stop cueing for the night

Both work in the dark, in any sleeping position, under blankets. Eye-movement
detection runs in shadow mode: it is computed and logged, but it can't
influence cue timing until it passes a threshold that was set in advance.

### Decision gates

Every stage has its pass threshold fixed before any data is collected, so a
result can't be reinterpreted after the fact.

**M0: is the eye region visible at all?** An overnight recording measures what
fraction of the night the face and eyes are in view.

| Coverage | Verdict | Next step |
|---|---|---|
| ≥ 25% | PASS | Build the eye-movement branch, in shadow mode |
| 15-25% | MARGINAL | Collect more nights without touching the thresholds |
| < 15% | FAIL | Drop the eye branch; the camera only guards against motion |

**M1: can it see deliberate eye movements while awake?** `morpheus calibrate`
compares closed-eye eye movements with closed-eye stillness and needs
**AUC ≥ 0.80**. The test is meant to be easy (awake, still, well lit, close to
the camera), so failing it means the signal isn't there. It also measures
head-turn leakage: if head turns separate from baseline as well as eye
movements do, the feature is tracking the head and a passing AUC means nothing.

## Results so far

These are all from awake calibration sessions, not from sleep.

- **Optical flow didn't work.** The first eye-flow index scored AUC 0.671
  against the 0.80 bar, and separated head turns (0.955) better than eye
  movements. Part of that was a bug: averaging per-pixel flow magnitudes meant
  sensor noise never cancelled out. Using the magnitude of the mean flow vector
  instead raised signal-to-noise from 31x to 603x, but the re-run scored 0.494
  with head turns still at 0.817, so I dropped that approach.
- **Eyelid shape did.** A lid-contour geometry feature scored 0.909 in one
  session. Since I found it after looking at that data, I froze the analysis
  and the pass rule (lower bound of the 95% CI ≥ 0.80) in
  `calibration/confirmation.py` before recording a new session. The new
  session scored **0.931 [0.878, 0.972]**, so it replicated. It's still a
  waking result, and a sleeping face at an angle in the dark is much harder.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/morpheus setup-models   # face detection + landmark models, SHA-256 checked
```

On macOS, camera and microphone permissions belong to the Python binary, not
the terminal you launched it from. If the camera stops opening after a
relaunch, check that first.

## Usage

```bash
# Dream journal (no hardware needed)
morpheus journal                  # log this morning's dreams
morpheus import-journal notes/    # import an existing journal (previews first)
morpheus review                   # score imported entries with no lucidity tag
morpheus baseline                 # lucid-dream rate over recent reports
morpheus dream-signs              # recurring motifs across reports

# Cues and nights
morpheus cues --add-preset trained-ascending --trained
morpheus cues --add-preset control-descending --no-trained
morpheus cues --preview 1         # play a cue at a safe volume while awake
morpheus train                    # conditioning session, before sleep
morpheus night --dry-run          # log when cues would play, no sound
morpheus night                    # a real cueing night
morpheus phase                    # show or switch tuning (A) / measuring (B)

# Camera
morpheus doctor                   # check the rig before a night
morpheus calibrate                # ~15 min guided waking calibration (M1)
morpheus calibrate --audio        # loudness calibration, ascending limits
morpheus confirm                  # check a session against the frozen lid criteria
morpheus record --hours 8         # overnight feature recording, no video
morpheus report                   # coverage report and M0 verdict
morpheus sessions

# Experiment (M3)
morpheus experiment --create pilot --design two-arm
morpheus prereg --experiment 1    # required before collection can start
morpheus experiment --start 1
morpheus reveal                   # last night's arm, only after the morning report
morpheus analyse                  # pre-registered Bayesian comparison

# Validation and adaptive cueing (M4-M6)
morpheus validate hypnogram.csv --session 12
morpheus adaptive                 # bandit posteriors
```

### Suggested order

1. **Keep the journal for two weeks first.** That gives a baseline to compare
   against later, and dream recall itself improves with practice.
2. **Condition the cue** with `morpheus train`. This is the part with
   published evidence behind it; the sound does nothing on its own.
3. **Do a `--dry-run` night** before any real one.

### Safety defaults

`morpheus night` is conservative out of the box:

- no cues in the first 5.5 hours, or in the 45 minutes before the alarm
- at most 6 cues a night and 2 an hour, 20 minutes apart
- every cue fades in from silence and is clamped to a hard volume ceiling
- any sign of waking up stops cueing for the rest of the night
- if sleep quality drops for 7 nights in a row, cueing stops until it recovers

Run `morpheus doctor` before the first night. It checks three things that
otherwise fail silently:

- **Manual exposure.** It reads the setting back and also measures brightness
  drift, because OpenCV can report success on a backend that ignored it.
  Auto-exposure hunting in a dark room looks like motion.
- **Image quality** against the configured floor. Face detection is gated on
  quality, so a bad floor gives a zero-coverage night that looks like a real
  finding.
- **System sleep.** A laptop that suspends at 2am leaves a hole in the data
  rather than a short night.

### Hardware

M0 needs about $60 of parts: a USB IR camera with manual exposure control (the
built-in FaceTime camera doesn't have it), an 850 nm illuminator, a rigid mount
and a bedside speaker.

## Development

```bash
.venv/bin/pytest -q

# Soak test: loop a clip for 8 hours and report peak memory, to catch slow leaks
morpheus record --replay clip.mp4 --loop --hours 8
```

The tests cover the coverage gate in detail, the per-second aggregation, and
fault injection for the overnight failure modes: camera loss, USB resets, clock
gaps from system sleep and crashes mid-run. In every case it should fail quietly.

YuNet doesn't fire on synthetic images, so the tests drive face detection with
a scripted fake detector. The real detector has to be checked on live footage,
which is what `morpheus doctor` is for.

### Layout

```
src/morpheus/
  types.py        shared value types, including the EventKind enum
  config.py       typed config, snapshotted for every recording
  capture/        live camera and file-replay frame sources
  vision/         quality, face presence, motion, eye and lid features
  calibration/    waking calibration and the confirmation test
  store/          SQLite schema, migrations and writers
  runtime/        sleep prevention, the recorder, the cueing night
  analysis/       coverage report and dream-sign mining
  audio/          cue synthesis, asset registry, playback
  cue/            safety limits, controller, policies, bandit
  training/       conditioning protocol
  report/         morning reports, journal import, sleep-quality monitor
  experiment/     randomization, blinding, pre-registration, analysis
  reference/      reference-device import and validation
```

### Cue path

A cue only plays if every layer allows it, and any one of them can block it:

| Layer | Role |
|---|---|
| `cue/policy.py` | Proposes timing and volume. Knows nothing about limits. |
| `cue/controller.py` | Gates: motion, quality, caps, cooldown, condition. |
| `cue/safety.py` | Hard caps. Nothing can override them. |
| `audio/player.py` | Clamps to the ceiling and forces a fade-in in the buffer. |

The adaptive bandit (M5) plugs in at the policy layer, so it has no way to
reach the safety limits. A learner rewarded for "cue noticed" would otherwise
work out that louder is better.

## Privacy

- No raw video is saved. One test checks that the recorder has no
  image-writing calls, and another checks that a full run leaves no media
  files behind.
- Everything stays local: no cloud, telemetry or crash reporting. Dream
  reports are never sent to an external service.
- Run it on an encrypted disk (FileVault).
- The database and any media files are gitignored.

## License

All rights reserved. See [LICENSE](LICENSE).
