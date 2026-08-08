"""The morning report: the primary outcome measure.

Everything else in Morpheus exists to influence the `lucid_binary` field of this
table. It is worth stating that plainly, because it is easy to spend months on a
vision pipeline and treat the questionnaire as an afterthought — when the
questionnaire *is* the measurement, and the vision pipeline is a hypothesis
about how to move it.

Two design commitments follow from that:

  1. The primary outcome is fixed in wording before data collection starts and
     is never revised mid-study. A definition that drifts is not a measurement.
  2. `guessed_condition` is collected on every report. The user is developer,
     participant and analyst simultaneously, so blinding will sometimes fail.
     Recording the guess makes unblinding measurable rather than something to be
     assumed away (design.md §15.2).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

# Fixed before night one. Changing this invalidates comparability with every
# report already collected, so it is pinned here and asserted by a test.
PRIMARY_OUTCOME_DEFINITION = (
    "At some point during a dream, I was aware that I was dreaming."
)


@dataclass
class MorningReport:
    report_date: str
    narrative: Optional[str] = None

    # Primary outcome
    lucid_binary: Optional[bool] = None
    lucid_confidence: Optional[int] = None      # 0-4
    knew_was_dreaming: Optional[bool] = None    # asked separately on purpose

    # Cue perception. Kept distinct: a cue can be heard, woven into the dream
    # without being recognised, or simply wake the person — three very different
    # results that a single "did you hear it" question would collapse.
    cue_heard: Optional[bool] = None
    cue_indirect: Optional[bool] = None
    cue_woke_me: Optional[bool] = None

    # The same three questions as counts, which is what gain titration actually
    # needs. Six cues a night collapsed into one bit cannot distinguish "one cue
    # was slightly loud" from "every cue woke me". The booleans above are kept
    # because the imported journal has them and nothing else.
    #
    # `cues_incorporated_count` is the one to maximise: a cue that turns up in
    # the dream was processed *while asleep*, which is the whole objective.
    # `cues_heard_count` is ambiguous on its own — hearing a cue may only mean
    # you were already awake.
    cues_heard_count: Optional[int] = None
    cues_woke_count: Optional[int] = None
    cues_incorporated_count: Optional[int] = None

    #: Minutes from finishing the WBTB script to falling back asleep.
    minutes_to_sleep_after_wbtb: Optional[float] = None

    dreams_recalled: Optional[int] = None
    vividness: Optional[int] = None             # 1-5
    sleep_quality: Optional[int] = None         # 1-5
    awakenings: Optional[int] = None

    guessed_condition: Optional[str] = None
    notes: Optional[str] = None
    session_id: Optional[int] = None

    def validate(self) -> list[str]:
        """Range checks. Returns problems rather than raising.

        A report submitted at 06:00 with one field out of range should be
        corrected, not discarded — the narrative is irreplaceable.
        """
        problems: list[str] = []
        for name, lo, hi in (
            ("lucid_confidence", 0, 4),
            ("vividness", 1, 5),
            ("sleep_quality", 1, 5),
        ):
            value = getattr(self, name)
            if value is not None and not (lo <= value <= hi):
                problems.append(f"{name} must be {lo}-{hi}, got {value}")
        for name in ("dreams_recalled", "awakenings"):
            value = getattr(self, name)
            if value is not None and value < 0:
                problems.append(f"{name} must be >= 0, got {value}")
        if self.lucid_binary and self.dreams_recalled == 0:
            problems.append("lucid dream reported but dreams_recalled is 0")
        return problems


class ReportStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def submit(self, report: MorningReport) -> int:
        problems = report.validate()
        if problems:
            raise ValueError("; ".join(problems))

        self._conn.execute(
            """
            INSERT INTO reports (
                session_id, report_date, submitted_at, narrative,
                lucid_binary, lucid_confidence, knew_was_dreaming,
                cue_heard, cue_indirect, cue_woke_me,
                cues_heard_count, cues_woke_count, cues_incorporated_count,
                minutes_to_sleep_after_wbtb,
                dreams_recalled, vividness, sleep_quality, awakenings,
                guessed_condition, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(report_date) DO UPDATE SET
                session_id=excluded.session_id,
                submitted_at=excluded.submitted_at,
                narrative=excluded.narrative,
                lucid_binary=excluded.lucid_binary,
                lucid_confidence=excluded.lucid_confidence,
                knew_was_dreaming=excluded.knew_was_dreaming,
                cue_heard=excluded.cue_heard,
                cue_indirect=excluded.cue_indirect,
                cue_woke_me=excluded.cue_woke_me,
                cues_heard_count=excluded.cues_heard_count,
                cues_woke_count=excluded.cues_woke_count,
                cues_incorporated_count=excluded.cues_incorporated_count,
                minutes_to_sleep_after_wbtb=excluded.minutes_to_sleep_after_wbtb,
                dreams_recalled=excluded.dreams_recalled,
                vividness=excluded.vividness,
                sleep_quality=excluded.sleep_quality,
                awakenings=excluded.awakenings,
                guessed_condition=excluded.guessed_condition,
                notes=excluded.notes
            """,
            (
                report.session_id, report.report_date,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                report.narrative,
                _b(report.lucid_binary), report.lucid_confidence, _b(report.knew_was_dreaming),
                _b(report.cue_heard), _b(report.cue_indirect), _b(report.cue_woke_me),
                report.cues_heard_count, report.cues_woke_count,
                report.cues_incorporated_count, report.minutes_to_sleep_after_wbtb,
                report.dreams_recalled, report.vividness, report.sleep_quality,
                report.awakenings, report.guessed_condition, report.notes,
            ),
        )
        row = self._conn.execute(
            "SELECT id FROM reports WHERE report_date = ?", (report.report_date,)
        ).fetchone()
        return int(row["id"])

    def get(self, report_date: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM reports WHERE report_date = ?", (report_date,)
        ).fetchone()

    def recent(self, limit: int = 30) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM reports ORDER BY report_date DESC LIMIT ?", (limit,)
        ).fetchall()

    def baseline_stats(self, limit: int = 60) -> dict:
        """Lucid-dream rate and recall over recent reports.

        This is the number the whole project is trying to move, and the reason
        the design insists on two weeks of journal-only baseline before any
        cueing begins.
        """
        rows = self.recent(limit)
        if not rows:
            return {"nights": 0}
        lucid = [r["lucid_binary"] for r in rows if r["lucid_binary"] is not None]
        recalled = [r["dreams_recalled"] for r in rows if r["dreams_recalled"] is not None]
        quality = [r["sleep_quality"] for r in rows if r["sleep_quality"] is not None]
        return {
            "nights": len(rows),
            "nights_scored": len(lucid),
            "lucid_nights": sum(lucid),
            "lucid_rate_per_night": (sum(lucid) / len(lucid)) if lucid else None,
            "lucid_per_week": (sum(lucid) / len(lucid) * 7) if lucid else None,
            "mean_dreams_recalled": (sum(recalled) / len(recalled)) if recalled else None,
            "mean_sleep_quality": (sum(quality) / len(quality)) if quality else None,
        }


def today_str() -> str:
    return date.today().isoformat()


def _b(value: Optional[bool]) -> Optional[int]:
    return None if value is None else int(bool(value))
