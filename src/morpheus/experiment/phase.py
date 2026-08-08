"""Protocol phase: optimise first, then measure.

Two phases, deliberately separated, because they want opposite things.

**Phase A — MAX LD.** No randomisation. Run the strongest protocol available
every night and tune it: gain, WBTB timing, induction script. Tuning is exactly
what makes these nights useless for causal inference — the intervention changes
under you — and that is fine, because inference is not what Phase A is for.

**Phase B — VALIDATE.** Everything frozen except the cue condition, which the
blinded harness randomises. This is the only phase that can estimate a causal
cue effect.

The failure this module exists to prevent is drifting from A into B without
noticing: continuing to nudge the gain "just a little" after randomisation has
started. That silently turns the trial into a comparison between two arms whose
protocol differed in ways nobody recorded, and it is undetectable afterwards.
So entering Phase B requires the tunable parameters to be *declared frozen*,
and the values are recorded at the moment of freezing so a later drift is
visible rather than inferred.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

#: Parameters that must be frozen before Phase B. Each is something that would
#: otherwise vary between arms and compete with the cue for credit.
FROZEN_PARAMETERS = (
    "gain",
    "wbtb_at",
    "wbtb_awake_min",
    "post_wbtb_delay_min",
    "wbtb_kind",
)

#: Nights of Phase A required before Phase B may begin. Not a statistical
#: threshold — it is the minimum needed to have titrated gain at all, given the
#: search moves at most one step per two nights.
MIN_PHASE_A_NIGHTS = 14


@dataclass
class PhaseState:
    phase: str = "A"
    frozen: dict = field(default_factory=dict)
    frozen_at: Optional[str] = None

    @property
    def is_validating(self) -> bool:
        return self.phase == "B"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def current(conn: sqlite3.Connection) -> PhaseState:
    row = conn.execute(
        "SELECT value FROM app_state WHERE key = 'protocol_phase'"
    ).fetchone()
    if row is None or not row["value"]:
        return PhaseState()
    data = json.loads(row["value"])
    return PhaseState(
        phase=data.get("phase", "A"),
        frozen=data.get("frozen", {}),
        frozen_at=data.get("frozen_at"),
    )


def _save(conn: sqlite3.Connection, state: PhaseState) -> None:
    conn.execute(
        "INSERT INTO app_state (key, value) VALUES ('protocol_phase', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (json.dumps({
            "phase": state.phase,
            "frozen": state.frozen,
            "frozen_at": state.frozen_at,
        }),),
    )
    conn.commit()


def phase_a_nights(conn: sqlite3.Connection) -> int:
    """Cueing nights recorded so far. Used only for the readiness check."""
    return conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE kind = 'cue_night' "
        "AND status IN ('completed', 'stopped_by_user')"
    ).fetchone()[0]


def readiness(conn: sqlite3.Connection) -> list[str]:
    """Reasons Phase B should not start yet. Empty means ready.

    Advisory on every point except the frozen parameters, which `enter_b`
    enforces. Judgement about whether a protocol has settled belongs to the
    person running it; what cannot be left to judgement is whether the
    parameters were written down.
    """
    problems: list[str] = []
    nights = phase_a_nights(conn)
    if nights < MIN_PHASE_A_NIGHTS:
        problems.append(
            f"only {nights} Phase A nights; {MIN_PHASE_A_NIGHTS} is the minimum "
            f"for the gain search to have moved at all"
        )

    scored = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE cues_woke_count IS NOT NULL"
    ).fetchone()[0]
    if scored < 5:
        problems.append(
            f"only {scored} nights with cue counts recorded; gain cannot have "
            f"been titrated against evidence"
        )
    else:
        # The two-sided condition. Zero arousal with zero incorporation means
        # the cue may simply be inaudible, and a trial run on inaudible cues
        # returns a confident null that means nothing at all.
        row = conn.execute(
            "SELECT SUM(COALESCE(cues_incorporated_count, 0)) AS inc, "
            "       SUM(COALESCE(cues_woke_count, 0)) AS woke "
            "FROM reports WHERE cues_woke_count IS NOT NULL"
        ).fetchone()
        if (row["inc"] or 0) == 0:
            problems.append(
                "no cue has ever been reported inside a dream. The gain may be "
                "below the perceptual threshold, and a trial at that level would "
                "produce a confident null that means nothing"
            )
    return problems


def enter_b(conn: sqlite3.Connection, frozen: dict, *, force: bool = False) -> PhaseState:
    """Freeze the protocol and enter Phase B.

    `frozen` must name every parameter in FROZEN_PARAMETERS. Refusing an
    incomplete freeze is the whole point: an unnamed parameter is one nobody
    committed to holding still.
    """
    missing = [p for p in FROZEN_PARAMETERS if p not in frozen]
    if missing:
        raise ValueError(
            f"cannot enter Phase B without freezing: {', '.join(missing)}. "
            f"A parameter left out is one nobody committed to holding constant, "
            f"and it will compete with the cue for credit."
        )
    if not force:
        problems = readiness(conn)
        if problems:
            raise ValueError(
                "not ready for Phase B:\n  - " + "\n  - ".join(problems)
            )
    state = PhaseState(phase="B", frozen=dict(frozen), frozen_at=_utc())
    _save(conn, state)
    return state


def return_to_a(conn: sqlite3.Connection) -> PhaseState:
    """Drop back to Phase A. Keeps the frozen record for the audit trail."""
    state = current(conn)
    state.phase = "A"
    _save(conn, state)
    return state


def drift(conn: sqlite3.Connection, observed: dict) -> list[str]:
    """Parameters differing from what was frozen. Empty in a clean Phase B night.

    Reported rather than blocked. A night that deviates is still a night; what
    matters is that the deviation is on the record instead of being discovered
    — or not — during analysis months later.
    """
    state = current(conn)
    if not state.is_validating:
        return []
    out = []
    for name, frozen_value in state.frozen.items():
        if name in observed and observed[name] != frozen_value:
            out.append(f"{name}: frozen {frozen_value!r}, tonight {observed[name]!r}")
    return out
