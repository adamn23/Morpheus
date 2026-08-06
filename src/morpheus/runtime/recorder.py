"""The overnight record loop.

M0 records and nothing else: no detection, no thresholds, no audio. The only
question it exists to answer is whether the rig survives a night and whether the
camera can see anything worth analysing (design.md §28).

Failure philosophy throughout: fail *quiet*, never loud. Every error path here
ends in stopping cleanly and writing down what happened. From M2 the same rule
governs the cue path, where the stakes are higher — a bug that wakes the sleeper
is worse than a bug that loses a night.
"""

from __future__ import annotations

import logging
import platform
import resource
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..capture.source import FrameSource, FrameSourceError
from ..capture.webcam import WebcamSource
from ..config import MorpheusConfig
from ..store.feature_store import FeatureStore
from ..types import HealthCounters
from ..vision.pipeline import Aggregator, VisionPipeline

log = logging.getLogger("morpheus.recorder")

# A monotonic jump larger than this means the loop did not merely stutter:
# the process was suspended, the machine slept, or the scheduler starved us.
# Inter-frame state is meaningless across such a gap and must be discarded.
CLOCK_GAP_THRESHOLD_S = 5.0


@dataclass
class RecordingSummary:
    session_id: Optional[int]
    session_uuid: Optional[str]
    status: str
    duration_s: float
    health: HealthCounters
    peak_rss_mb: float
    exposure_detail: str = ""
    detector_status: str = ""
    sleep_assertion: str = ""
    notes: list[str] = field(default_factory=list)


class Recorder:
    def __init__(
        self,
        config: MorpheusConfig,
        source: FrameSource,
        store: FeatureStore,
        *,
        repo: Optional[Path] = None,
    ) -> None:
        self._cfg = config
        self._source = source
        self._store = store
        self._repo = repo
        self._pipeline = VisionPipeline(config)
        self._aggregator = Aggregator()
        self._health = HealthCounters()
        self._stop = False
        self._notes: list[str] = []

    def request_stop(self, *_: Any) -> None:
        """Idempotent, signal-safe. Sets a flag; the loop exits at the top."""
        self._stop = True

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.request_stop)
            except (ValueError, OSError):
                # Not on the main thread; the caller drives stopping instead.
                pass

    def run(self, *, max_hours: Optional[float] = None, kind: str = "probe") -> RecordingSummary:
        max_seconds = (max_hours if max_hours is not None else self._cfg.recorder.max_hours) * 3600.0
        started_mono = time.monotonic()
        status = "completed"

        self._source.open()
        session_id = self._store.start_session(
            config=self._cfg,
            device_profile=self._source.device_profile(),
            kind=kind,
            repo=self._repo,
        )
        log.info("session %s started (uuid=%s)", session_id, self._store.session_uuid)

        consecutive_failures = 0
        last_t_mono: Optional[float] = None
        last_log = started_mono

        try:
            while not self._stop:
                if time.monotonic() - started_mono >= max_seconds:
                    self._notes.append(f"reached configured limit of {max_seconds / 3600:.2f} h")
                    break

                frame = self._source.read()

                if frame is None:
                    if self._source.exhausted:
                        self._notes.append("frame source exhausted")
                        break
                    consecutive_failures += 1
                    self._health.read_failures += 1
                    if not self._handle_read_failure(consecutive_failures):
                        status = "aborted_camera_lost"
                        break
                    if consecutive_failures == 0:
                        last_t_mono = None
                    continue

                consecutive_failures = 0
                self._health.frames_captured += 1

                if last_t_mono is not None and (frame.t_mono - last_t_mono) > CLOCK_GAP_THRESHOLD_S:
                    gap = (last_t_mono, frame.t_mono)
                    self._health.clock_gaps.append(gap)
                    log.warning(
                        "monotonic gap of %.1f s — process suspended or machine slept; "
                        "discarding inter-frame state",
                        gap[1] - gap[0],
                    )
                    self._pipeline.reset()
                    self._aggregator.flush()
                last_t_mono = frame.t_mono

                sample = self._pipeline.process(frame)
                completed = self._aggregator.add(sample)
                if completed is not None:
                    self._store.append(completed)
                    self._health.seconds_recorded += 1

                now = time.monotonic()
                if now - last_log >= self._cfg.recorder.log_interval_s:
                    self._log_progress(now - started_mono)
                    self._store.update_health(self._health, _peak_rss_mb())
                    last_log = now

        except FrameSourceError as exc:
            status = "aborted_camera_error"
            self._notes.append(str(exc))
            log.error("camera error: %s", exc)
        except Exception as exc:  # noqa: BLE001 - an overnight run must not die loudly
            status = "aborted_error"
            self._notes.append(f"{type(exc).__name__}: {exc}")
            log.exception("unexpected error; stopping cleanly")
        finally:
            tail = self._aggregator.flush()
            if tail is not None:
                self._store.append(tail)
                self._health.seconds_recorded += 1
            self._finalise_health(started_mono)
            self._store.finish_session(status, self._health)
            self._store.update_health(self._health, _peak_rss_mb())
            self._source.close()

        if self._stop and status == "completed":
            status = "stopped_by_user"

        return RecordingSummary(
            session_id=session_id,
            session_uuid=self._store.session_uuid,
            status=status,
            duration_s=time.monotonic() - started_mono,
            health=self._health,
            peak_rss_mb=_peak_rss_mb(),
            exposure_detail=_exposure_detail(self._source),
            detector_status=self._pipeline.presence.status,
            notes=self._notes,
        )

    # ------------------------------------------------------------- internals

    def _handle_read_failure(self, consecutive: int) -> bool:
        """Return False when the source should be considered permanently lost."""
        if consecutive < 15:
            time.sleep(0.05)
            return True

        reconnect = getattr(self._source, "reconnect", None)
        if reconnect is None:
            self._notes.append("source does not support reconnection")
            return False

        attempts = self._cfg.camera.max_reconnect_attempts
        log.warning("%d consecutive read failures; attempting reconnect", consecutive)
        for attempt in range(1, attempts + 1):
            if self._stop:
                return False
            if reconnect():
                self._health.reconnects += 1
                self._pipeline.reset()
                log.info("camera reconnected on attempt %d", attempt)
                return True
            log.warning("reconnect attempt %d/%d failed", attempt, attempts)
        self._notes.append(f"camera did not return after {attempts} reconnect attempts")
        return False

    def _finalise_health(self, started_mono: float) -> None:
        elapsed = max(1e-6, time.monotonic() - started_mono)
        # Frames lost to clock gaps were never expected, so subtract that time
        # before estimating how many frames we should have seen. Without this,
        # a single suspend makes an otherwise healthy run look catastrophic.
        gap_time = sum(end - start for start, end in self._health.clock_gaps)
        active = max(1e-6, elapsed - gap_time)
        expected = int(active * self._cfg.camera.fps)
        self._health.frames_dropped = max(0, expected - self._health.frames_captured)

    def _log_progress(self, elapsed: float) -> None:
        log.info(
            "%.2f h | %d s recorded | uptime %.1f%% | %d read failures | %d reconnects | %.0f MB",
            elapsed / 3600.0,
            self._health.seconds_recorded,
            100.0 * self._health.capture_uptime,
            self._health.read_failures,
            self._health.reconnects,
            _peak_rss_mb(),
        )


def _exposure_detail(source: FrameSource) -> str:
    if isinstance(source, WebcamSource):
        status = source.exposure_status
        state = "confirmed manual" if status.manual_confirmed else "NOT confirmed"
        return f"{state} ({status.detail})"
    return "n/a (not a live camera)"


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return float(usage) / divisor
