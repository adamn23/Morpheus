"""Persistence for cues, outcomes, events, and training sessions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from ..cue.outcome import OutcomeAssessment
from ..cue.state import CueCommand
from ..types import EventKind


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class CueStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def begin_cue(
        self,
        session_id: int,
        command: CueCommand,
        t_mono: float,
        *,
        asset_id: Optional[int],
        asset_sha256: Optional[str],
    ) -> int:
        """Write the cue record BEFORE audio starts.

        If the process dies during playback, the record still exists with
        played=0, so the night's analysis can distinguish "we tried and
        something broke" from "no cue was attempted" (design.md §12.3).
        """
        cur = self._conn.execute(
            """
            INSERT INTO cues (
                session_id, t_mono, t_utc, cue_asset_id, asset_sha256,
                gain, gain_requested, ramp_ms, duration_ms, repetition_index,
                policy_version, gate_snapshot_json, trigger, played
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0)
            """,
            (
                session_id, t_mono, _utc(), asset_id, asset_sha256,
                command.gain, command.gain_requested, command.ramp_ms,
                command.duration_ms, command.repetition_index,
                command.policy_version, json.dumps(command.gates.to_dict()),
                command.trigger,
            ),
        )
        return int(cur.lastrowid)

    def complete_cue(self, cue_id: int, *, played: bool, error: Optional[str] = None) -> None:
        self._conn.execute(
            "UPDATE cues SET played = ?, error = ? WHERE id = ?",
            (int(played), error, cue_id),
        )

    def record_outcome(self, cue_id: int, window_s: float, a: OutcomeAssessment) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cue_outcomes (
                cue_id, window_s, outcome, motion_before, motion_after,
                motion_delta, latency_to_motion_ms, quality_during, coverage_during
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                cue_id, window_s, a.outcome.value, a.motion_before, a.motion_after,
                a.motion_delta, a.latency_to_motion_ms, a.quality_during, a.coverage_during,
            ),
        )

    def record_event(
        self,
        session_id: int,
        t_mono: float,
        kind: EventKind,
        *,
        confidence: Optional[float] = None,
        duration_ms: Optional[float] = None,
        features: Optional[dict] = None,
        detector_version: Optional[str] = None,
    ) -> None:
        # EventKind is a closed enum (design.md §11); accepting a bare string
        # here would route around the naming discipline the whole project rests on.
        if not isinstance(kind, EventKind):
            raise TypeError(f"kind must be an EventKind, got {type(kind).__name__}")
        self._conn.execute(
            "INSERT INTO events (session_id, t_mono, t_utc, kind, confidence, "
            "duration_ms, features_json, detector_version) VALUES (?,?,?,?,?,?,?,?)",
            (
                session_id, t_mono, _utc(), kind.value, confidence, duration_ms,
                json.dumps(features) if features else None, detector_version,
            ),
        )

    def cues_for_session(self, session_id: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT c.*, o.outcome, o.motion_delta FROM cues c "
            "LEFT JOIN cue_outcomes o ON o.cue_id = c.id "
            "WHERE c.session_id = ? ORDER BY c.t_mono",
            (session_id,),
        ).fetchall()

    # ------------------------------------------------------------- training

    def start_training(
        self, kind: str, *, cue_asset_id: Optional[int], session_id: Optional[int] = None
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO training_sessions (session_id, cue_asset_id, kind, started_at) "
            "VALUES (?,?,?,?)",
            (session_id, cue_asset_id, kind, _utc()),
        )
        return int(cur.lastrowid)

    def finish_training(
        self,
        training_id: int,
        *,
        completed: bool,
        duration_s: float,
        steps: dict,
        engagement_rating: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            "UPDATE training_sessions SET completed_at = ?, completed = ?, duration_s = ?, "
            "steps_json = ?, engagement_rating = ?, notes = ? WHERE id = ?",
            (
                _utc(), int(completed), duration_s, json.dumps(steps),
                engagement_rating, notes, training_id,
            ),
        )

    def latest_training(self, within_hours: float = 24.0) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM training_sessions WHERE completed = 1 "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
