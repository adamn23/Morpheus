"""Recorder integration and fault injection.

Every failure mode here is one that will actually happen during an overnight
run: a USB camera resetting, a laptop suspending, a disk filling, a process
being killed. The invariant under test is always the same — Morpheus stops
cleanly and writes down what happened. It never dies loudly, and it never
silently produces data that looks fine.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import pytest

from morpheus.capture.replay import FileReplaySource
from morpheus.capture.source import FrameSourceError
from morpheus.config import MorpheusConfig
from morpheus.runtime.recorder import Recorder
from morpheus.store.feature_store import FeatureStore
from morpheus.types import Frame

from .conftest import ScriptedDetector, face_for_frame, gray_frame


class FakeSource:
    """A scriptable FrameSource for fault injection.

    `fail_at` maps a frame index to how many consecutive read failures to
    inject there; `gap_at` injects a monotonic-clock jump, standing in for the
    machine suspending mid-run.
    """

    def __init__(
        self,
        total: int = 120,
        fps: float = 30.0,
        fail_at: Optional[dict[int, int]] = None,
        gap_at: Optional[dict[int, float]] = None,
        reconnectable: bool = True,
        fatal_after_failures: bool = False,
    ) -> None:
        self.total = total
        self.fps = fps
        self.fail_at = fail_at or {}
        self.gap_at = gap_at or {}
        self.reconnectable = reconnectable
        self.fatal_after_failures = fatal_after_failures
        self.index = 0
        self.opened = False
        self.closed = False
        self.reconnect_calls = 0
        self._pending_failures = 0
        self._clock = 0.0
        self._t0 = time.time()

    def open(self) -> None:
        self.opened = True

    def read(self) -> Optional[Frame]:
        if self._pending_failures > 0:
            self._pending_failures -= 1
            return None
        if self.index in self.fail_at:
            self._pending_failures = self.fail_at.pop(self.index) - 1
            return None
        if self.index >= self.total:
            return None
        self._clock += self.gap_at.pop(self.index, 0.0) + 1.0 / self.fps
        frame = Frame(
            seq=self.index,
            t_mono=self._clock,
            t_utc=self._t0 + self._clock,
            image=gray_frame(noise=3, seed=self.index),
        )
        self.index += 1
        return frame

    def reconnect(self) -> bool:
        self.reconnect_calls += 1
        if not self.reconnectable:
            return False
        self._pending_failures = 0
        return True

    def close(self) -> None:
        self.closed = True

    @property
    def exhausted(self) -> bool:
        return self.index >= self.total and self._pending_failures == 0

    def device_profile(self) -> dict[str, Any]:
        return {"camera_model": "fake", "width": 320, "height": 240, "fps": self.fps}


def build(config: MorpheusConfig, conn, source) -> Recorder:
    store = FeatureStore(conn, batch_size=5)
    recorder = Recorder(config, source, store)
    recorder._pipeline.presence = ScriptedDetector([face_for_frame()])  # noqa: SLF001
    return recorder


# ------------------------------------------------------------------- happy path


def test_records_features_and_completes(config: MorpheusConfig, conn) -> None:
    source = FakeSource(total=150)
    summary = build(config, conn, source).run(max_hours=1.0)

    assert summary.status == "completed"
    assert summary.health.frames_captured == 150
    assert summary.health.seconds_recorded >= 4
    assert source.closed, "the source must be released even on the happy path"

    rows = conn.execute("SELECT COUNT(*) FROM frames_1hz").fetchone()[0]
    assert rows == summary.health.seconds_recorded


def test_no_video_is_written_anywhere(config: MorpheusConfig, conn, tmp_path: Path) -> None:
    """The privacy claim, verified behaviourally rather than by inspection."""
    build(config, conn, FakeSource(total=90)).run(max_hours=1.0)
    media = [
        p for p in tmp_path.rglob("*")
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".png", ".jpg", ".jpeg", ".mkv"}
    ]
    assert media == [], f"recorder wrote image data: {media}"


# ---------------------------------------------------------------- fault injection


def test_transient_read_failures_are_survived(config: MorpheusConfig, conn) -> None:
    source = FakeSource(total=120, fail_at={30: 5, 70: 3})
    summary = build(config, conn, source).run(max_hours=1.0)

    assert summary.status == "completed"
    assert summary.health.read_failures == 8
    assert summary.health.frames_captured == 120, "no frames should be lost to a brief stall"
    assert source.reconnect_calls == 0, "a short stall must not trigger a reconnect"


def test_sustained_failure_triggers_reconnect(config: MorpheusConfig, conn) -> None:
    source = FakeSource(total=120, fail_at={40: 20})
    summary = build(config, conn, source).run(max_hours=1.0)

    assert source.reconnect_calls >= 1
    assert summary.health.reconnects >= 1
    assert summary.status == "completed"


def test_unrecoverable_camera_aborts_cleanly(config: MorpheusConfig, conn) -> None:
    """A dead camera ends the night with a diagnosis, not a traceback."""
    config.camera.max_reconnect_attempts = 3
    config.camera.reconnect_backoff_s = 0.0
    source = FakeSource(total=500, fail_at={20: 400}, reconnectable=False)
    summary = build(config, conn, source).run(max_hours=1.0)

    assert summary.status == "aborted_camera_lost"
    assert any("reconnect attempts" in note for note in summary.notes)
    assert source.closed
    row = conn.execute("SELECT status FROM sessions").fetchone()
    assert row["status"] == "aborted_camera_lost"


def test_clock_gap_is_detected_and_recorded(config: MorpheusConfig, conn) -> None:
    """A suspended laptop must leave a visible hole, not a seamless fiction.

    monotonic() does not advance across system sleep on macOS, so without this
    detection the recorder would resume and keep numbering seconds as though
    nothing had happened.
    """
    source = FakeSource(total=120, gap_at={60: 900.0})
    summary = build(config, conn, source).run(max_hours=1.0)

    assert len(summary.health.clock_gaps) == 1
    start, end = summary.health.clock_gaps[0]
    assert end - start == pytest.approx(900.0, abs=1.0)


def test_gap_time_excluded_from_dropped_frame_estimate(config: MorpheusConfig, conn) -> None:
    """One suspend must not make a healthy run look catastrophic."""
    source = FakeSource(total=120, gap_at={60: 3600.0})
    summary = build(config, conn, source).run(max_hours=2.0)
    assert summary.health.capture_uptime > 0.5


def test_stop_request_ends_run_cleanly(config: MorpheusConfig, conn) -> None:
    source = FakeSource(total=100_000)
    recorder = build(config, conn, source)

    original = recorder._pipeline.process  # noqa: SLF001

    def process_and_stop(frame):
        result = original(frame)
        if frame.seq >= 90:
            recorder.request_stop()
        return result

    recorder._pipeline.process = process_and_stop  # noqa: SLF001
    summary = recorder.run(max_hours=1.0)

    assert summary.status == "stopped_by_user"
    assert conn.execute("SELECT COUNT(*) FROM frames_1hz").fetchone()[0] >= 2


def test_max_duration_is_respected(config: MorpheusConfig, conn) -> None:
    source = FakeSource(total=10_000_000)
    summary = build(config, conn, source).run(max_hours=1.0 / 3600.0)  # one second
    assert summary.duration_s < 30
    assert any("limit" in note for note in summary.notes)


def test_unexpected_error_does_not_escape(config: MorpheusConfig, conn) -> None:
    """An overnight run must never die loudly. Diagnose, record, stop."""
    source = FakeSource(total=200)
    recorder = build(config, conn, source)

    def explode(frame):
        raise RuntimeError("synthetic pipeline failure")

    recorder._pipeline.process = explode  # noqa: SLF001
    summary = recorder.run(max_hours=1.0)

    assert summary.status == "aborted_error"
    assert any("synthetic pipeline failure" in note for note in summary.notes)
    assert source.closed


# ----------------------------------------------------------------------- replay


def test_replay_source_round_trip(config: MorpheusConfig, conn, make_video) -> None:
    """The seam that makes every future detector change re-testable."""
    path = make_video(frames=90, fps=30.0, motion=True)
    source = FileReplaySource(path)
    summary = build(config, conn, source).run(max_hours=1.0)

    assert summary.status == "completed"
    assert summary.health.frames_captured == 90
    assert any("exhausted" in note for note in summary.notes)
    assert conn.execute("SELECT COUNT(*) FROM frames_1hz").fetchone()[0] >= 3


def test_replay_missing_file_raises_clearly(tmp_path: Path) -> None:
    source = FileReplaySource(tmp_path / "nope.mp4")
    with pytest.raises(FrameSourceError, match="not found"):
        source.open()


def test_replay_timestamps_are_monotonic(make_video) -> None:
    source = FileReplaySource(make_video(frames=40, fps=25.0))
    source.open()
    stamps = []
    while (frame := source.read()) is not None:
        stamps.append(frame.t_mono)
    source.close()

    assert stamps == sorted(stamps)
    assert stamps[1] - stamps[0] == pytest.approx(1 / 25.0, rel=1e-6)
    assert source.exhausted


def test_replay_handles_bogus_frame_rate(make_video, monkeypatch) -> None:
    """A malformed container must not produce a divide-by-zero timestamp."""
    import cv2

    source = FileReplaySource(make_video(frames=10))
    original = cv2.VideoCapture.get

    def fake_get(self, prop):
        return 0.0 if prop == cv2.CAP_PROP_FPS else original(self, prop)

    monkeypatch.setattr(cv2.VideoCapture, "get", fake_get)
    source.open()
    first, second = source.read(), source.read()
    source.close()

    assert first is not None and second is not None
    # Timestamps are anchored to time.monotonic(), so only the spacing is
    # meaningful. A reported rate of zero must fall back to 30 fps rather than
    # dividing by it.
    assert second.t_mono - first.t_mono == pytest.approx(1 / 30.0, rel=1e-6)


# ------------------------------------------------------- camera open retry


class _FakeCapture:
    """Stands in for cv2.VideoCapture, failing to open N times first."""

    instances: list["_FakeCapture"] = []

    def __init__(self, device, fail_times: list[int]) -> None:
        self.device = device
        self.released = False
        self._open = fail_times[0] <= 0
        fail_times[0] -= 1
        _FakeCapture.instances.append(self)

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the cv2 API
        return self._open

    def getBackendName(self) -> str:  # noqa: N802 - mirrors the cv2 API
        return "fake"

    def release(self) -> None:
        self.released = True

    def set(self, *_: object) -> bool:
        return False

    def get(self, *_: object) -> float:
        return 0.0

    def read(self):
        return False, None


@pytest.fixture
def fake_capture(monkeypatch):
    """Patch cv2.VideoCapture to fail a configurable number of opens."""
    import cv2

    def install(fail_count: int):
        remaining = [fail_count]
        _FakeCapture.instances = []
        monkeypatch.setattr(
            cv2, "VideoCapture", lambda device, *a, **k: _FakeCapture(device, remaining)
        )
        return _FakeCapture.instances

    return install


def test_camera_open_retries_through_permission_dialog(config: MorpheusConfig, fake_capture) -> None:
    """macOS raises its consent dialog asynchronously and fails the open.

    Without a retry window the first run of any freshly-built binary looks like
    a hardware fault, and the user has to run every command twice. Regression
    guard for exactly that.
    """
    from morpheus.capture.webcam import WebcamSource

    config.camera.open_retry_attempts = 4
    config.camera.open_retry_delay_s = 0.01
    config.camera.warmup_frames = 0
    instances = fake_capture(fail_count=2)

    source = WebcamSource(config.camera)
    source.open()

    assert len(instances) == 3, "should have retried twice before succeeding"
    assert all(c.released for c in instances[:-1]), "failed handles must be released"


def test_camera_open_gives_up_with_actionable_error(config: MorpheusConfig, fake_capture) -> None:
    from morpheus.capture.webcam import WebcamSource

    config.camera.open_retry_attempts = 3
    config.camera.open_retry_delay_s = 0.01
    fake_capture(fail_count=99)

    with pytest.raises(FrameSourceError) as excinfo:
        WebcamSource(config.camera).open()

    message = str(excinfo.value)
    assert "3 attempts" in message
    assert "permission" in message.lower()
