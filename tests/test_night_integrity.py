"""A night that failed must not look like a night that declined to cue.

Both produce zero cues, no exception, and a session row. The difference is that
one is evidence and the other is an artefact, and only the second must be kept
out of the analysis. These tests pin the two ways a run silently loses time:
stopping early, and the machine suspending underneath it.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time

import pytest

from morpheus.config import MorpheusConfig
from morpheus.runtime.night import SUSPEND_DETECT_S, NightSummary
from morpheus.runtime.power import SleepPreventer


def _summary(**kwargs) -> NightSummary:
    base = dict(
        session_id=1, status="completed", duration_s=8 * 3600.0,
        cues_played=0, cues_failed=0, intended_s=8 * 3600.0,
        wall_elapsed_s=8 * 3600.0, suspended_s=0.0,
    )
    base.update(kwargs)
    return NightSummary(**base)


class TestTruncation:
    def test_full_run_is_clean(self):
        assert _summary().defects() == []

    def test_short_run_is_flagged(self):
        s = _summary(duration_s=1.5 * 3600.0)
        assert s.truncated
        # The message must say the cueing window may not have opened: at the
        # 5.5 h default floor, a 1.5 h run *could not* have cued.
        assert "cueing window" in s.defects()[0]

    def test_marginal_shortfall_is_tolerated(self):
        # Shutdown takes a moment; a run is not defective for ending 30 s early.
        assert not _summary(duration_s=8 * 3600.0 - 30).truncated

    def test_stopped_by_user_still_reports_the_shortfall(self):
        # An honest early stop is still not a full night of cueing opportunity.
        assert _summary(status="stopped_by_user", duration_s=3600.0).truncated


class TestSuspendDetection:
    def test_wall_clock_divergence_is_the_signal(self):
        # macOS monotonic does not advance while suspended, so the loop sees a
        # short run while the wall clock sees a full night. That divergence is
        # the only observable, and it must be reported as lost time.
        s = _summary(duration_s=5 * 3600.0, wall_elapsed_s=8 * 3600.0,
                     suspended_s=3 * 3600.0)
        assert s.suspended
        assert any("suspended" in d for d in s.defects())

    def test_clock_skew_below_threshold_is_ignored(self):
        assert not _summary(suspended_s=SUSPEND_DETECT_S - 1).suspended

    def test_both_defects_reported_together(self):
        s = _summary(duration_s=2 * 3600.0, wall_elapsed_s=7 * 3600.0,
                     suspended_s=5 * 3600.0)
        assert len(s.defects()) == 2


class TestSleepAssertionLeavesTheDisplayAlone:
    def test_no_display_flags(self, monkeypatch):
        """-d holds the screen on and -u turns it on. Either puts a lit screen
        at the pillow all night, which is the sleep disturbance this project
        exists to avoid causing."""
        seen: list[list[str]] = []

        class FakeProc:
            pid = 999
            _alive = True

            def poll(self):
                return None if self._alive else 0

            def terminate(self):
                self._alive = False

            def wait(self, timeout=None):
                return 0

        monkeypatch.setattr("morpheus.runtime.power.platform.system", lambda: "Darwin")
        monkeypatch.setattr("morpheus.runtime.power.shutil.which", lambda _: "/usr/bin/caffeinate")
        monkeypatch.setattr(
            subprocess, "Popen", lambda argv, **kw: (seen.append(argv), FakeProc())[1]
        )

        with SleepPreventer() as sleeper:
            assert sleeper.active

        flags = seen[0][1]
        assert "d" not in flags, f"caffeinate {flags} keeps the display lit"
        assert "u" not in flags, f"caffeinate {flags} turns the display on"
        assert "i" in flags and "s" in flags, "must still prevent system sleep"


@pytest.mark.skipif(shutil.which("morpheus") is None, reason="CLI not installed")
class TestInterruptFinalizesTheSession:
    """Ctrl-C is what the CLI tells the user to press to end a night.

    Before this was wired, the default handler raised KeyboardInterrupt — a
    BaseException, invisible to the runner's `except Exception` — and the run
    exited with the session row still at status='running', no end time, no
    summary and no defect flag. The night vanished silently, which is the exact
    failure this module exists to make impossible. Only a subprocess can test
    it: the bug lived in signal disposition, which is process-global.
    """

    def test_sigint_produces_a_summary_and_a_defect(self, tmp_path):
        # Point the subprocess at a throwaway data dir. Without this the suite
        # writes a junk cue_night session into the real research database on
        # every run, which is a worse contamination than the bug under test.
        config = MorpheusConfig()
        config.storage.data_dir = tmp_path
        config_file = tmp_path / "config.json"
        config_file.write_text(config.model_dump_json())

        registered = subprocess.run(
            ["morpheus", "cues", "--add-preset", "trained-ascending", "--trained",
             "--config", str(config_file)],
            capture_output=True, text=True, timeout=60,
        )
        assert registered.returncode == 0, registered.stdout + registered.stderr

        proc = subprocess.Popen(
            ["morpheus", "night", "--hours", "8", "--dry-run",
             "--config", str(config_file)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, preexec_fn=os.setsid,
        )
        try:
            # A fresh database has no conditioning session, so `night` asks
            # whether to run an unconditioned cue anyway. Say yes: this test is
            # about shutdown, not about the training guard.
            assert proc.stdin is not None
            proc.stdin.write("y\n")
            proc.stdin.flush()
            time.sleep(4)
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            out, _ = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=10)

        assert proc.returncode == 0, f"unclean exit:\n{out}"
        assert "night finished" in out, f"no summary printed:\n{out}"
        # A four-second night must not be mistaken for a night that declined
        # to cue.
        assert "DEFECTIVE NIGHT" in out, f"truncation not flagged:\n{out}"
