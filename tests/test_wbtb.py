"""WBTB support: the induction script, the alarm, and the counts it needs.

The safety half of WBTB — that a mid-night re-arm cannot restart the nightly
cue budget — lives in `test_safety.py`, next to the supervisor it constrains.
"""

from __future__ import annotations

import pytest

from morpheus.audio.assets import PRESETS, WAKE_ONLY_PRESETS
from morpheus.report.schema import MorningReport, ReportStore
from morpheus.training.protocol import (
    MAX_POST_WAKE_SCRIPT_S,
    POST_WAKE_KINDS,
    TRAINING_KINDS,
    protocol_for,
    total_seconds,
)


class TestInductionScripts:
    def test_every_declared_kind_builds(self):
        for kind in TRAINING_KINDS:
            assert protocol_for(kind), f"{kind} produced no steps"

    def test_post_wake_scripts_stay_short(self):
        """Time on the script is time not spent falling back asleep, and sleep
        latency after the technique is one of only two replicated predictors of
        success. A script that sprawls works against the thing that helps."""
        for kind in POST_WAKE_KINDS:
            total = total_seconds(kind)
            assert total <= MAX_POST_WAKE_SCRIPT_S, (
                f"{kind} runs {total}s, over the {MAX_POST_WAKE_SCRIPT_S}s post-wake cap"
            )

    def test_ssild_keeps_the_cue_binding(self):
        """Without the cue steps this is plain SSILD, not a TLR protocol, and
        the cue it is supposed to be conditioning goes unpaired."""
        steps = protocol_for("ssild")
        assert sum(s.plays_cue for s in steps) >= 2

    def test_ssild_cycles_all_three_senses_quick_then_slow(self):
        keys = [s.key for s in protocol_for("ssild")]
        for sense in ("sight", "hearing", "touch"):
            assert any(k.startswith("ssild_quick") and k.endswith(sense) for k in keys)
            assert any(k.startswith("ssild_slow") and k.endswith(sense) for k in keys)
        # Quick cycles must come first; the technique depends on the ordering.
        first_slow = next(i for i, k in enumerate(keys) if k.startswith("ssild_slow"))
        last_quick = max(i for i, k in enumerate(keys) if k.startswith("ssild_quick"))
        assert last_quick < first_slow

    def test_unknown_kind_falls_back_rather_than_crashing(self):
        # At 05:00 a typo should not take the night down.
        assert protocol_for("nonsense") == protocol_for("evening")


class TestWakeAlarmIsNotACue:
    def test_alarm_preset_exists_and_is_wake_only(self):
        assert "wbtb-alarm" in PRESETS
        assert "wbtb-alarm" in WAKE_ONLY_PRESETS

    def test_alarm_is_acoustically_unlike_every_cue(self):
        """If the alarm resembled the cue it would counter-condition it to mean
        'get up' — the one failure the protocol cannot absorb."""
        alarm = PRESETS["wbtb-alarm"]
        for name, notes in PRESETS.items():
            if name in WAKE_ONLY_PRESETS:
                continue
            assert len(alarm) != len(notes), f"alarm has the same shape as {name}"
        assert min(alarm) > max(PRESETS["trained-ascending"]), (
            "alarm must sit above the cue register"
        )


class TestTitrationFields:
    """Gain titration is two-sided: minimise arousal *and* keep evidence the cue
    was processed. Zero awakenings with zero incorporation means the cue may
    simply be inaudible, which would make a whole trial meaningless."""

    def test_counts_round_trip(self, tmp_path):
        from morpheus.store.db import open_db

        conn = open_db(tmp_path / "m.db")
        store = ReportStore(conn)
        store.submit(MorningReport(
            report_date="2026-08-09",
            lucid_binary=False,
            cues_heard_count=3,
            cues_incorporated_count=1,
            cues_woke_count=2,
            minutes_to_sleep_after_wbtb=12.5,
            dreams_recalled=4,
        ))
        row = conn.execute(
            "SELECT * FROM reports WHERE report_date = '2026-08-09'"
        ).fetchone()
        assert row["cues_heard_count"] == 3
        assert row["cues_incorporated_count"] == 1
        assert row["cues_woke_count"] == 2
        assert row["minutes_to_sleep_after_wbtb"] == pytest.approx(12.5)
        conn.close()

    def test_wbtb_latency_stays_null_when_not_asked(self, tmp_path):
        # A night with no WBTB must leave the field null, not zero: zero would
        # read as "fell asleep instantly" and bias the predictor.
        from morpheus.store.db import open_db

        conn = open_db(tmp_path / "m.db")
        ReportStore(conn).submit(MorningReport(report_date="2026-08-10", lucid_binary=False))
        row = conn.execute(
            "SELECT minutes_to_sleep_after_wbtb FROM reports WHERE report_date='2026-08-10'"
        ).fetchone()
        assert row[0] is None
        conn.close()
