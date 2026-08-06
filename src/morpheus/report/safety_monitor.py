"""Sleep-quality monitoring and the automatic stopping rule.

Design.md §23 failure condition 5 and §24 both commit to this: if sleep quality
declines for seven consecutive nights, the study halts. The generated
pre-registration states it as a stopping rule to any reader.

It was missing until an audit caught it, which made the pre-registration a
document that promised a safety behaviour the code did not implement. That is
worse than not having the rule at all — a stated guarantee nobody checks is how
a participant ends up trusting a protection that was never there.

The rule is enforced before a night arms rather than during it. Stopping a study
is a decision made awake, with the numbers in front of you, not something to
discover at 04:00.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Optional, Sequence

# Design.md §23. Pre-committed; not tuned against observed data.
DECLINE_NIGHTS = 7
MIN_HISTORY = 10


@dataclass
class SleepQualityCheck:
    nights_examined: int
    recent_mean: Optional[float]
    baseline_mean: Optional[float]
    consecutive_declines: int
    should_halt: bool
    reason: str = ""
    trend: list[tuple[str, int]] = field(default_factory=list)

    @property
    def warning(self) -> bool:
        """Two-thirds of the way to a halt. Worth surfacing before it triggers."""
        return not self.should_halt and self.consecutive_declines >= DECLINE_NIGHTS - 2


def _consecutive_declines(scores: Sequence[int]) -> int:
    """Length of the current run of non-increasing nights, oldest to newest.

    Non-increasing rather than strictly decreasing: a plateau at a low score is
    the same clinical picture as a slow slide, and requiring a strict decrease
    every single night would make the rule almost impossible to trigger on
    integer 1-5 data.
    """
    if len(scores) < 2:
        return 0
    run = 0
    for previous, current in zip(scores, scores[1:]):
        if current <= previous:
            run += 1
        else:
            run = 0
    return run


def check(conn: sqlite3.Connection, *, window: int = 30) -> SleepQualityCheck:
    """Evaluate the stopping rule against recent morning reports."""
    rows = conn.execute(
        "SELECT report_date, sleep_quality FROM reports "
        "WHERE sleep_quality IS NOT NULL ORDER BY report_date DESC LIMIT ?",
        (window,),
    ).fetchall()
    rows = list(reversed(rows))  # oldest first

    trend = [(r["report_date"], int(r["sleep_quality"])) for r in rows]
    scores = [score for _, score in trend]

    if len(scores) < MIN_HISTORY:
        return SleepQualityCheck(
            nights_examined=len(scores),
            recent_mean=None,
            baseline_mean=None,
            consecutive_declines=0,
            should_halt=False,
            reason=f"only {len(scores)} scored nights; need {MIN_HISTORY} to judge a trend",
            trend=trend,
        )

    declines = _consecutive_declines(scores)
    recent = scores[-DECLINE_NIGHTS:]
    baseline = scores[:-DECLINE_NIGHTS] or scores
    recent_mean = sum(recent) / len(recent)
    baseline_mean = sum(baseline) / len(baseline)

    should_halt = declines >= DECLINE_NIGHTS
    reason = ""
    if should_halt:
        reason = (
            f"sleep quality has not improved for {declines} consecutive nights "
            f"(recent mean {recent_mean:.1f} vs baseline {baseline_mean:.1f}). "
            f"Per design.md §23, cueing stops until it recovers."
        )

    return SleepQualityCheck(
        nights_examined=len(scores),
        recent_mean=recent_mean,
        baseline_mean=baseline_mean,
        consecutive_declines=declines,
        should_halt=should_halt,
        reason=reason,
        trend=trend,
    )


def format_check(result: SleepQualityCheck) -> str:
    lines = ["Sleep-quality stopping rule", "-" * 60]
    if result.recent_mean is None:
        lines.append(f"  {result.reason}")
        return "\n".join(lines)

    lines.append(f"  nights examined       {result.nights_examined}")
    lines.append(f"  recent mean           {result.recent_mean:.2f}")
    lines.append(f"  baseline mean         {result.baseline_mean:.2f}")
    lines.append(f"  consecutive declines  {result.consecutive_declines} of {DECLINE_NIGHTS}")
    if result.should_halt:
        lines.append("")
        lines.append("  HALT — " + result.reason)
    elif result.warning:
        lines.append("")
        lines.append(
            f"  Approaching the stopping rule. Two more non-improving nights and "
            f"cueing halts automatically."
        )
    return "\n".join(lines)
