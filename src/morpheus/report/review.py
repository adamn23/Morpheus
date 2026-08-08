"""Scoring imported journal entries that carry no lucidity tag.

A journal kept in prose rather than with tags imports with `lucid_binary` unset,
which leaves the baseline rate unmeasurable. This walks the unscored entries so
they can be coded against the pinned outcome definition.

**The regex suggests; the human decides.** Prose is genuinely ambiguous — "I
dreamt I was flying and knew it was impossible" may or may not have involved
awareness of dreaming, and only the person who had the dream can say. Automatic
classification would fabricate the project's primary outcome, so the phrase
matching exists solely to sort entries by how likely they are to need a careful
read.

On bias: the person coding is also the person hoping the intervention works,
which is unavoidable at N-of-1. It is worth knowing that the bias runs
*conservative*. These are pre-intervention nights, so over-coding them as lucid
raises the baseline and makes any later effect look smaller. Erring generous
here is the safe direction.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Optional

# Phrases that suggest awareness *within* the dream. Deliberately broad: the
# cost of a false suggestion is a few seconds of reading, while a missed one is
# a silently mis-scored night.
SUGGESTIVE = [
    # This journal abbreviates lucid dream to "ld", but uses it three ways, so
    # the label matters more than the match. "Had an ld", "2nd Ld?" and "Ld #5?"
    # are claims. "Ld: location, people" is a dream-sign note about what could
    # have been noticed and is NOT a claim — auto-scoring those would inflate the
    # baseline. Bare mentions are often explicitly negative ("related to LD, but
    # probably not"). Word boundaries keep old/world/could/held from matching.
    (re.compile(
        r"\bhad an? ld\b"                          # "Had an ld!!???"
        r"|^\s*\d+(?:st|nd|rd|th)?\s*ld\b(?!\s*:)"  # "2nd Ld?"
        r"|^\s*ld\s*#\s*\d+",                      # "Ld #5?"
        re.I | re.M), "LD CLAIM"),
    (re.compile(r"^\s*ld\s*:", re.I | re.M), "ld: dream-sign note (not a claim)"),
    (re.compile(r"\bld\b", re.I), "mentions ld"),
    (re.compile(r"\blucid", re.I), "says lucid"),
    (re.compile(r"\b(realis|realiz)\w*\s+(that\s+)?(i|it)\b", re.I), "realised"),
    (re.compile(r"\bknew\s+(that\s+)?(i\s+was\s+)?dream", re.I), "knew dreaming"),
    (re.compile(r"\baware\s+(that\s+)?(i\s+was\s+)?dream", re.I), "aware dreaming"),
    (re.compile(r"\breality\s*check", re.I), "reality check"),
    (re.compile(r"\bi\s+was\s+dreaming\b", re.I), "was dreaming"),
    (re.compile(r"\bwoke\s+up\s+in\s+(the|my)\s+dream", re.I), "woke in dream"),
    (re.compile(r"\bcontrol\w*\s+the\s+dream", re.I), "controlled dream"),
    (re.compile(r"\bfalse\s+awaken", re.I), "false awakening"),
]


@dataclass
class Candidate:
    report_date: str
    narrative: str
    hints: list[str]

    @property
    def suggestive(self) -> bool:
        return bool(self.hints)


def unscored(conn: sqlite3.Connection, limit: Optional[int] = None) -> list[Candidate]:
    """Imported entries with no lucidity value, suggestive ones first.

    Ordering by suggestiveness is a convenience, not a shortcut: every entry
    still has to be answered, and the ordering does not change what is stored.
    """
    sql = (
        "SELECT report_date, narrative FROM reports "
        "WHERE lucid_binary IS NULL AND narrative IS NOT NULL AND narrative != '' "
        "ORDER BY report_date"
    )
    rows = conn.execute(sql).fetchall()

    out: list[Candidate] = []
    for row in rows:
        text = row["narrative"]
        hints = [label for pattern, label in SUGGESTIVE if pattern.search(text)]
        out.append(Candidate(row["report_date"], text, hints))

    # Strongest evidence first: an explicit claim, then anything else
    # suggestive, then the rest. Ordering only changes the order of the
    # questions, never their answers.
    def rank(candidate: Candidate) -> tuple:
        return (
            0 if "LD CLAIM" in candidate.hints else (1 if candidate.suggestive else 2),
            candidate.report_date,
        )

    out.sort(key=rank)
    return out[:limit] if limit else out


def score(conn: sqlite3.Connection, report_date: str, lucid: bool) -> None:
    conn.execute(
        "UPDATE reports SET lucid_binary = ? WHERE report_date = ?",
        (int(lucid), report_date),
    )


def progress(conn: sqlite3.Connection) -> dict:
    total = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE narrative IS NOT NULL AND narrative != ''"
    ).fetchone()[0]
    scored = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE lucid_binary IS NOT NULL"
    ).fetchone()[0]
    lucid = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE lucid_binary = 1"
    ).fetchone()[0]
    return {
        "total": total,
        "scored": scored,
        "remaining": max(0, total - scored),
        "lucid": lucid,
    }
