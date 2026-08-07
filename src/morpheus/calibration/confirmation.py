"""Frozen confirmation analysis for the lid-geometry finding.

**This module is committed before the confirmation data exists and must not be
edited afterwards.** Everything in it — the primary measure, the motion gate,
the validity checks, the pass rule — is fixed here so that the verdict is a
function of the data alone. If a future change is genuinely warranted, it
belongs in a new module with a new pre-registration, not in an edit to this one.

## Why a confirmation is needed

On 2026-08-07, session 3 of the waking calibration produced:

| Measure | AUC | 95% CI |
|---|---|---|
| eye-flow, all windows | 0.573 | [0.442, 0.700] |
| eye-flow, motion-gated | 0.605 | [0.453, 0.751] |
| lid-contour, all windows | 0.883 | [0.786, 0.961] |
| **lid-contour, motion-gated** | **0.909** | **[0.812, 0.983]** |

The last line is the only thing in this project that has ever cleared the gate.
It is also **post-hoc**: four variants were examined after the data were seen,
the motion gate was not pre-specified, and its threshold was chosen by looking.
None of that is reflected in the confidence interval. A finding produced that
way has to survive on data it has not seen before it means anything.

## Why the bar is higher than the original gate

The original M1 gate was a point estimate of AUC >= 0.80. This confirmation
requires **the lower bound of the 95% bootstrap CI >= 0.80**, which is strictly
harder. That is deliberate: a post-hoc finding should clear a higher bar than a
pre-planned one, not the same bar. Loosening was never an option; tightening
guards against the small sample that produced the original number.

## The pose confound, stated plainly

Lid displacement responded to deliberate head turns at AUC 0.988 — *more*
strongly than to saccades at 0.883. On its own the channel measures "the eye
region changed", not "the eyes moved". It looked clean only because every
head-turn window exceeded the motion gate and was discarded.

That is defensible operationally, since the cue controller already suppresses
cues during body motion (gate G4), so the system never acts while the head is
moving. But it means the confirmation must show two separate things: that the
gate really does remove large head movement, and that *small, sleep-like* head
movement does not masquerade as eye movement. The protocol therefore adds a
`micro_head_motion` segment, which the original did not have.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .profile import _auc

# ---------------------------------------------------------------------------
# FROZEN PARAMETERS. Do not edit after 2026-08-07.
# ---------------------------------------------------------------------------

PRIMARY_CHANNEL = "lid_disp"
WINDOW_S = 1.0

#: Pass rule. Stricter than the original 0.80 point estimate, on purpose.
CI_LOWER_BOUND_REQUIRED = 0.80
BOOTSTRAP_DRAWS = 3000
BOOTSTRAP_SEED = 20260807

#: Motion gate. Defined as a percentile of window motion in the BASELINE
#: segment only. Using the control condition to define "quiet" cannot leak
#: information about the saccade-versus-baseline contrast, and fixing the
#: percentile here stops it being chosen once the answer is visible.
MOTION_GATE_PERCENTILE = 90.0

#: Minimum surviving windows. Below this the interval is too wide to mean
#: anything and the result is INSUFFICIENT DATA rather than a verdict.
MIN_POSITIVE_WINDOWS = 20
MIN_BASELINE_WINDOWS = 15

#: L1 — the motion gate must actually remove large head movement.
MIN_HEAD_TURN_EXCLUSION = 0.80

#: L2 — small, sleep-like head movement must not look like eye movement.
#: Evaluated only if enough micro-motion windows survive the gate; if the gate
#: removes them too, L2 is satisfied through L1.
MIN_MICRO_WINDOWS_FOR_L2 = 10

BASELINE_SEGMENT = "eyes_closed_still"
SACCADE_SEGMENTS = ("slow_saccades", "fast_saccades")
HEAD_TURN_SEGMENT = "head_turn"
MICRO_MOTION_SEGMENT = "micro_head_motion"


@dataclass
class Window:
    peak: float
    motion: float


@dataclass
class ConfirmationResult:
    auc: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    motion_threshold: Optional[float] = None

    n_positive: int = 0
    n_baseline: int = 0

    l1_head_turn_excluded: Optional[float] = None
    l2_micro_auc: Optional[float] = None
    l2_micro_windows: int = 0

    insufficient: str = ""
    notes: list[str] = field(default_factory=list)

    # --- the frozen decision rules ---------------------------------------

    @property
    def l1_ok(self) -> Optional[bool]:
        if self.l1_head_turn_excluded is None:
            return None
        return self.l1_head_turn_excluded >= MIN_HEAD_TURN_EXCLUSION

    @property
    def l2_ok(self) -> Optional[bool]:
        """Small head movement must not separate as well as eye movement.

        If the gate removed the micro-motion windows too, there is no residual
        confound to measure and L1 already covers it.
        """
        if self.l2_micro_windows < MIN_MICRO_WINDOWS_FOR_L2:
            return True
        if self.l2_micro_auc is None or self.auc is None:
            return None
        return self.l2_micro_auc < self.auc

    @property
    def validity_ok(self) -> Optional[bool]:
        checks = (self.l1_ok, self.l2_ok)
        if any(c is False for c in checks):
            return False
        if any(c is None for c in checks):
            return None
        return True

    @property
    def replicated(self) -> bool:
        return (
            self.validity_ok is True
            and self.ci_low is not None
            and self.ci_low >= CI_LOWER_BOUND_REQUIRED
        )

    @property
    def verdict(self) -> str:
        if self.insufficient:
            return "INSUFFICIENT DATA"
        if self.validity_ok is False:
            return "NO VERDICT (instrument invalid)"
        if self.validity_ok is None:
            return "NO VERDICT (validity unmeasurable)"
        return "REPLICATED" if self.replicated else "NOT REPLICATED"


def _windows(samples: Sequence[dict], field_name: str) -> list[Window]:
    """One-second window maxima, paired with mean body motion in that window."""
    rows = [
        s for s in samples
        if s.get(field_name) is not None
        and s.get("motion") is not None
        and s.get("t_mono") is not None
    ]
    if not rows:
        return []
    out: list[Window] = []
    start = rows[0]["t_mono"]
    peak = float("-inf")
    motion: list[float] = []
    for row in rows:
        if row["t_mono"] - start >= WINDOW_S:
            if motion:
                out.append(Window(peak, float(np.mean(motion))))
            start, peak, motion = row["t_mono"], float("-inf"), []
        peak = max(peak, float(row[field_name]))
        motion.append(float(row["motion"]))
    if motion:
        out.append(Window(peak, float(np.mean(motion))))
    return out


def _bootstrap_ci(pos: Sequence[float], neg: Sequence[float]) -> tuple[Optional[float], Optional[float]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        a = _auc(
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        )
        if a is not None:
            values.append(a)
    if len(values) < BOOTSTRAP_DRAWS // 4:
        return (None, None)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def confirm(collected: dict[str, list[dict]]) -> ConfirmationResult:
    """Evaluate the frozen criteria against a fresh recording."""
    result = ConfirmationResult()

    baseline = _windows(collected.get(BASELINE_SEGMENT, []), PRIMARY_CHANNEL)
    if len(baseline) < MIN_BASELINE_WINDOWS:
        result.insufficient = (
            f"only {len(baseline)} baseline windows; need {MIN_BASELINE_WINDOWS}"
        )
        return result

    # Motion gate, from the baseline segment alone.
    threshold = float(np.percentile([w.motion for w in baseline], MOTION_GATE_PERCENTILE))
    result.motion_threshold = threshold

    def gated(segment: str) -> list[float]:
        return [w.peak for w in _windows(collected.get(segment, []), PRIMARY_CHANNEL)
                if w.motion < threshold]

    quiet_baseline = [w.peak for w in baseline if w.motion < threshold]
    quiet_positive: list[float] = []
    for segment in SACCADE_SEGMENTS:
        quiet_positive.extend(gated(segment))

    result.n_positive = len(quiet_positive)
    result.n_baseline = len(quiet_baseline)
    if result.n_positive < MIN_POSITIVE_WINDOWS or result.n_baseline < MIN_BASELINE_WINDOWS:
        result.insufficient = (
            f"after gating: {result.n_positive} positive, {result.n_baseline} baseline "
            f"windows; need {MIN_POSITIVE_WINDOWS} and {MIN_BASELINE_WINDOWS}"
        )
        return result

    result.auc = _auc(quiet_positive, quiet_baseline)
    result.ci_low, result.ci_high = _bootstrap_ci(quiet_positive, quiet_baseline)

    # L1: does the gate remove large head movement?
    turns = _windows(collected.get(HEAD_TURN_SEGMENT, []), PRIMARY_CHANNEL)
    if turns:
        excluded = sum(1 for w in turns if w.motion >= threshold)
        result.l1_head_turn_excluded = excluded / len(turns)

    # L2: does small, sleep-like head movement look like eye movement?
    micro = gated(MICRO_MOTION_SEGMENT)
    result.l2_micro_windows = len(micro)
    if len(micro) >= MIN_MICRO_WINDOWS_FOR_L2:
        result.l2_micro_auc = _auc(micro, quiet_baseline)
        if result.l2_micro_auc is not None and result.auc is not None:
            if result.l2_micro_auc >= result.auc:
                result.notes.append(
                    "Small head movements separate from baseline at least as well as "
                    "eye movements do. Within the quiet condition the channel is still "
                    "tracking head motion, and the AUC above is not evidence of eye "
                    "movement."
                )
    else:
        result.notes.append(
            f"only {len(micro)} micro-motion windows survived the gate, so the residual "
            f"pose confound could not be measured directly; L1 covers it instead."
        )
    return result


def format_result(result: ConfirmationResult, *, reference_auc: float = 0.909) -> str:
    lines: list[str] = []
    add = lines.append
    add("Lid-geometry confirmation — frozen criteria")
    add("=" * 70)
    add(f"  primary channel   {PRIMARY_CHANNEL}, {WINDOW_S:.0f}s window maxima")
    add(f"  pass rule         95% CI lower bound >= {CI_LOWER_BOUND_REQUIRED}")
    add(f"  motion gate       p{MOTION_GATE_PERCENTILE:.0f} of baseline window motion")
    add("")

    if result.insufficient:
        add(f"  INSUFFICIENT DATA: {result.insufficient}")
        return "\n".join(lines)

    add("Result")
    add("-" * 70)
    add(f"  motion threshold  {result.motion_threshold:.6f}")
    add(f"  windows           {result.n_positive} positive / {result.n_baseline} baseline")
    add(f"  AUC               {result.auc:.3f}  [{result.ci_low:.3f}, {result.ci_high:.3f}]")
    add(f"  original finding  {reference_auc:.3f}  (post-hoc, session 3)")
    add("")

    add("Validity")
    add("-" * 70)
    l1 = result.l1_head_turn_excluded
    add(f"  L1 gate removes large head movement   "
        f"{('%.0f%%' % (l1 * 100)) if l1 is not None else '-':>6}  "
        f"(>= {MIN_HEAD_TURN_EXCLUSION:.0%})  {_mark(result.l1_ok)}")
    if result.l2_micro_auc is not None:
        add(f"  L2 small head movement vs saccades    "
            f"{result.l2_micro_auc:.3f}  (< {result.auc:.3f})  {_mark(result.l2_ok)}")
    else:
        add(f"  L2 small head movement                "
            f"{result.l2_micro_windows} windows survived gate  {_mark(result.l2_ok)}")
    add("")

    for note in result.notes:
        for line in _wrap(note, 66):
            add(f"  ! {line}")
    if result.notes:
        add("")

    add(f"VERDICT: {result.verdict}")
    add("=" * 70)
    if result.verdict == "REPLICATED":
        add("  The lid-contour signal survived on data it had not seen, under")
        add("  criteria fixed before the recording. This is the first result in")
        add("  the project that is evidence rather than a hypothesis.")
        add("")
        add("  It remains a WAKING result. The sleeping case is harder in every")
        add("  respect and still requires validation against a reference device")
        add("  before it may influence a single cue. G9 stays locked.")
    elif result.verdict == "NOT REPLICATED":
        add("  The instrument was valid and the signal did not survive. The 0.909")
        add("  was most likely an artefact of examining four variants after the")
        add("  fact on a single small session.")
        add("")
        add("  This ends the camera eye-tracking branch. Do not adjust and retry:")
        add("  the criteria were frozen precisely so that this outcome counts.")
    else:
        add("  No verdict. The instrument did not clear its validity checks, so")
        add("  the AUC above is not evidence either way.")
    return "\n".join(lines)


def load_and_confirm(conn: sqlite3.Connection, profile_id: Optional[int] = None) -> ConfirmationResult:
    from .profile import load_samples

    if profile_id is None:
        row = conn.execute("SELECT MAX(id) FROM calibration_profiles").fetchone()
        if row is None or row[0] is None:
            raise ValueError("no calibration profiles recorded")
        profile_id = int(row[0])
    return confirm(load_samples(conn, profile_id))


def _mark(ok: Optional[bool]) -> str:
    return "ok" if ok else ("FAIL" if ok is False else "unmeasurable")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
