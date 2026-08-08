"""Phase A optimises the protocol; Phase B measures it.

The mistake this guards against is drifting from one into the other without
noticing — continuing to nudge the gain "just a little" after randomisation has
started. That turns the trial into a comparison between arms whose protocol
differed in unrecorded ways, and it is undetectable after the fact.
"""

from __future__ import annotations

import pytest

from morpheus.experiment import phase as ph
from morpheus.store.db import open_db


@pytest.fixture
def conn(tmp_path):
    c = open_db(tmp_path / "m.db")
    yield c
    c.close()


def _frozen(**overrides):
    base = {p: "x" for p in ph.FROZEN_PARAMETERS}
    base.update(overrides)
    return base


def _nights(conn, n, *, incorporated=1, woke=0):
    for i in range(n):
        conn.execute(
            "INSERT INTO sessions (uuid, started_at_utc, started_at_mono, status, kind) "
            "VALUES (?,?,?, 'completed', 'cue_night')",
            (f"u{i}", "2026-08-01T00:00:00Z", float(i)),
        )
        conn.execute(
            "INSERT INTO reports (report_date, submitted_at, cues_woke_count, "
            "cues_incorporated_count) VALUES (?,?,?,?)",
            (f"2026-06-{i + 1:02d}", "2026-08-01T00:00:00Z", woke, incorporated),
        )
    conn.commit()


class TestDefaultIsPhaseA:
    def test_starts_in_a(self, conn):
        assert ph.current(conn).phase == "A"
        assert not ph.current(conn).is_validating


class TestEnteringB:
    def test_refuses_an_incomplete_freeze(self, conn):
        _nights(conn, ph.MIN_PHASE_A_NIGHTS)
        partial = _frozen()
        partial.pop("gain")
        with pytest.raises(ValueError, match="gain"):
            ph.enter_b(conn, partial)

    def test_refuses_before_enough_nights(self, conn):
        _nights(conn, 3)
        with pytest.raises(ValueError, match="Phase A nights"):
            ph.enter_b(conn, _frozen())

    def test_refuses_when_no_cue_was_ever_incorporated(self, conn):
        """Zero arousal with zero incorporation may just mean the cue is
        inaudible. Running a trial at that level yields a confident null that
        means nothing — the exact failure two-sided titration exists to avoid."""
        _nights(conn, ph.MIN_PHASE_A_NIGHTS, incorporated=0, woke=0)
        with pytest.raises(ValueError, match="below the perceptual threshold"):
            ph.enter_b(conn, _frozen())

    def test_enters_when_ready(self, conn):
        _nights(conn, ph.MIN_PHASE_A_NIGHTS, incorporated=2, woke=0)
        state = ph.enter_b(conn, _frozen(gain="0.03"))
        assert state.is_validating
        assert state.frozen["gain"] == "0.03"
        assert state.frozen_at
        assert ph.current(conn).is_validating

    def test_force_overrides_readiness_but_not_completeness(self, conn):
        _nights(conn, 1)
        assert ph.enter_b(conn, _frozen(), force=True).is_validating
        partial = _frozen()
        partial.pop("wbtb_at")
        with pytest.raises(ValueError, match="wbtb_at"):
            ph.enter_b(conn, partial, force=True)


class TestDriftDetection:
    def test_no_drift_in_phase_a(self, conn):
        # Tuning is the point of Phase A; nothing to report.
        assert ph.drift(conn, {"gain": "0.99"}) == []

    def test_reports_a_changed_parameter_in_phase_b(self, conn):
        _nights(conn, ph.MIN_PHASE_A_NIGHTS, incorporated=1)
        ph.enter_b(conn, _frozen(gain="0.03"))
        found = ph.drift(conn, {"gain": "0.08"})
        assert len(found) == 1 and "gain" in found[0]
        assert "0.03" in found[0] and "0.08" in found[0]

    def test_silent_when_the_night_matches(self, conn):
        _nights(conn, ph.MIN_PHASE_A_NIGHTS, incorporated=1)
        ph.enter_b(conn, _frozen(gain="0.03"))
        assert ph.drift(conn, {"gain": "0.03"}) == []


class TestReturningToA:
    def test_keeps_the_frozen_record(self, conn):
        _nights(conn, ph.MIN_PHASE_A_NIGHTS, incorporated=1)
        ph.enter_b(conn, _frozen(gain="0.03"))
        state = ph.return_to_a(conn)
        assert state.phase == "A"
        # The audit trail survives: what was frozen, and when, stays readable.
        assert ph.current(conn).frozen["gain"] == "0.03"
