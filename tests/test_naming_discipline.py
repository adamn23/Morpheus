"""Enforces the labelling rules from design.md §11.

Morpheus does not measure sleep stages and cannot confirm REM. That constraint
is easy to hold in a design document and easy to lose in code, where a variable
called `is_rem` costs nothing to type and quietly converts a hedge into a claim.

This test is the enforcement mechanism. It is not pedantry: the naming is what
stops the developer — who is also the participant and the analyst — from
gradually believing the system does something it does not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from morpheus.types import EventKind

SRC = Path(__file__).resolve().parent.parent / "src" / "morpheus"

# Identifier-shaped uses only. The words may appear in prose (this file, and the
# comments explaining why they are banned), so the patterns require the
# snake/camel forms a variable or column would actually take.
FORBIDDEN = {
    r"\brem_detected\b": "asserts REM, which Morpheus cannot establish",
    r"\bis_rem\b": "asserts REM",
    r"\brem_state\b": "asserts REM",
    r"\bin_rem\b": "asserts REM",
    r"\bsleep_stage\b": "Morpheus does not stage sleep",
    r"\bsleep_phase\b": "Morpheus does not stage sleep",
    r"\bis_dreaming\b": "dreaming is not observable from outside",
    r"\bdream_detected\b": "dreaming is not observable from outside",
    r"\bconfirmed_rem\b": "nothing is confirmed without reference data",
    r"\bconfirmed_lucid\b": "lucidity comes from self-report, never from sensing",
}

ALLOWED_FILES = {"test_naming_discipline.py"}

# Prose that explains the ban necessarily contains the banned words. An explicit
# opt-out marker keeps that legal while making every exemption visible in review
# — which is the point. Silent exemptions would defeat the whole mechanism.
ESCAPE = "naming-lint: allow"


def _source_files() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if p.name not in ALLOWED_FILES]


@pytest.mark.parametrize("pattern,reason", list(FORBIDDEN.items()))
def test_forbidden_vocabulary_absent(pattern: str, reason: str) -> None:
    regex = re.compile(pattern)
    offenders: list[str] = []
    for path in _source_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if ESCAPE in line:
                continue
            if regex.search(line):
                offenders.append(f"{path.relative_to(SRC)}:{lineno}: {line.strip()}")
    assert not offenders, (
        f"forbidden identifier matching {pattern!r} ({reason}).\n"
        "Use one of the hedged EventKind labels instead.\n" + "\n".join(offenders)
    )


#: Events that record what *Morpheus did*, not what it inferred about the
#: sleeper. These are exempt from the hedging rule for a specific reason: the
#: rule exists because an inference about someone else's sleep must never be
#: stated as fact, and "we played an alarm" is not an inference. Hedging it
#: would be false modesty and would blur the line the rule is protecting.
#: Membership here is the reviewed decision — anything making a claim about the
#: sleeper belongs above, hedged.
PROTOCOL_ACTIONS = {"wbtb_wake", "wbtb_resume"}


def test_event_kind_is_closed_and_hedged() -> None:
    """Every observational label must hedge. Adding a bare claim should fail here."""
    hedges = ("probable_", "possible_", "cue_", "signal_")
    for kind in EventKind:
        if kind.value in PROTOCOL_ACTIONS:
            continue
        assert kind.value.startswith(hedges), (
            f"{kind.value!r} does not hedge. Every observational EventKind must be "
            "qualified: Morpheus reports what it observed, not what it concluded. "
            "If this is an action Morpheus took rather than an inference about the "
            "sleeper, add it to PROTOCOL_ACTIONS — deliberately."
        )


def test_protocol_actions_claim_nothing_about_sleep() -> None:
    """The exemption must not become a loophole.

    An action label may say what the system did. The moment one implies a sleep
    state, the hedging rule has been routed around rather than scoped.
    """
    banned = ("rem", "asleep", "dreaming", "lucid", "stage", "sleeping")
    for value in PROTOCOL_ACTIONS:
        assert value in {k.value for k in EventKind}, f"{value} is not an EventKind"
        assert not any(word in value for word in banned), (
            f"{value!r} is exempt from hedging because it describes an action, "
            "but it names a sleep state. Those cannot both be true."
        )


def test_event_kind_membership_is_deliberate() -> None:
    """Pin the enum. Growing it should be a reviewed decision, not a drive-by."""
    assert {k.value for k in EventKind} == {
        "probable_eye_movement_burst",
        "possible_dream_activity",
        "probable_arousal",
        "possible_awakening",
        "cue_delivered_during_detected_activity",
        "signal_unavailable",
        # Protocol actions, added 2026-08-08 for WBTB support. See
        # PROTOCOL_ACTIONS above for why these do not hedge.
        "wbtb_wake",
        "wbtb_resume",
    }


def test_no_video_persistence_in_daemon() -> None:
    """Structural guarantee behind the privacy claim in design.md §20.

    No code path may write image data. This asserts the absence of the obvious
    mechanisms; it is a tripwire against accidental reintroduction, not a proof.
    The one legitimate VideoWriter lives in the test fixtures, outside src/.
    """
    offenders: list[str] = []
    writers = re.compile(r"\b(VideoWriter|imwrite)\b")
    for path in _source_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if writers.search(line):
                offenders.append(f"{path.relative_to(SRC)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "image-writing call found in the daemon. Morpheus persists derived "
        "features only (design.md §20).\n" + "\n".join(offenders)
    )


def test_every_package_has_a_tracked_init() -> None:
    """Catch the packaging miss that has now happened twice.

    An editable install resolves missing __init__.py via namespace packages, so
    the whole test suite passes while a real wheel silently omits the package.
    It only surfaces on a fresh clone or a genuine install — i.e. for someone
    else, later.
    """
    import subprocess

    repo = SRC.parent.parent
    missing: list[str] = []
    for directory in sorted(p for p in SRC.rglob("*") if p.is_dir()):
        if directory.name in {"__pycache__"} or directory.name.endswith(".egg-info"):
            continue
        if not any(directory.glob("*.py")):
            continue
        init = directory / "__init__.py"
        if not init.exists():
            missing.append(f"{init.relative_to(repo)} does not exist")
            continue
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(init.relative_to(repo))],
            cwd=repo, capture_output=True,
        )
        if tracked.returncode != 0:
            missing.append(f"{init.relative_to(repo)} exists but is untracked")

    assert not missing, "packaging would omit these:\n  " + "\n  ".join(missing)
