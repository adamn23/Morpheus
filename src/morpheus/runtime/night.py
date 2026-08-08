"""The overnight cueing run.

Composes the M0 recorder with the M2 cue controller. Two modes:

  * **with camera** — features drive the sensing gates, so a cue is suppressed
    while the body is moving and the response to each cue is observed.
  * **clock only** — no camera at all. Feature frames are synthesised empty and
    the sensing gates record `unavailable`.

Clock-only is a first-class mode, not a fallback. The published TLR protocol
used no sensing whatsoever, so this is the arm with actual evidence behind it,
and the camera has to earn its way in by beating it (design.md §8). It also
means a broken camera degrades the night rather than ending it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..audio.assets import CueAsset, CueAssetRegistry
from ..audio.player import AudioError, CuePlayer
from ..capture.source import FrameSource
from ..config import MorpheusConfig
from ..cue.controller import CueController
from ..cue.state import CueState, Outcome
from ..store.cue_store import CueStore
from ..store.feature_store import FeatureStore
from ..types import CoverageFlag, EventKind, FeatureFrame, HealthCounters
from ..vision.pipeline import Aggregator, VisionPipeline

log = logging.getLogger("morpheus.night")

_OUTCOME_EVENTS = {
    Outcome.PROBABLE_AROUSAL.value: EventKind.PROBABLE_AROUSAL,
    Outcome.POSSIBLE_AWAKENING.value: EventKind.POSSIBLE_AWAKENING,
}

# Divergence between wall clock and monotonic clock above which the machine is
# taken to have been suspended. On macOS `time.monotonic()` is CLOCK_UPTIME_RAW,
# which does not advance during system sleep, so a suspend leaves *no* monotonic
# gap — `runtime.recorder`'s gap detector compares monotonic against monotonic
# and is structurally blind to it. Wall-versus-monotonic is the only signal that
# sees it. Thirty seconds is far above ordinary NTP correction and far below any
# suspend worth reporting.
SUSPEND_DETECT_S = 30.0

# Fraction of the intended run below which the night is reported as truncated.
_TRUNCATION_RATIO = 0.95


@dataclass
class NightSummary:
    session_id: Optional[int]
    status: str
    duration_s: float
    cues_played: int
    cues_failed: int
    outcomes: dict[str, int] = field(default_factory=dict)
    halted_reason: str = ""
    final_state: str = ""
    camera_used: bool = False
    health: HealthCounters = field(default_factory=HealthCounters)
    gate_blocks: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    #: What the run was asked for, so "no cues fired" can be distinguished from
    #: "the run never reached the hours in which cues were permitted".
    intended_s: float = 0.0
    #: Wall-clock span from arm to finish. Exceeds `duration_s` by the time the
    #: machine spent suspended.
    wall_elapsed_s: float = 0.0
    #: Estimated time suspended. Non-zero means the sleep assertion was lost.
    suspended_s: float = 0.0

    @property
    def truncated(self) -> bool:
        return self.duration_s < self.intended_s * _TRUNCATION_RATIO

    @property
    def suspended(self) -> bool:
        return self.suspended_s >= SUSPEND_DETECT_S

    def defects(self) -> list[str]:
        """Reasons this night should not be analysed as a clean observation.

        A night that stopped early or slept through its cueing window produces
        zero cues and no error, which is indistinguishable from a night the
        policy legitimately declined to cue. Left unflagged it enters the
        analysis as a real control-like observation and biases the result
        toward no effect. Naming the defect is what keeps it out.
        """
        out: list[str] = []
        if self.truncated:
            out.append(
                f"ran {self.duration_s / 3600:.2f} h of an intended "
                f"{self.intended_s / 3600:.2f} h — cueing window may never have opened"
            )
        if self.suspended:
            out.append(
                f"machine suspended for about {self.suspended_s / 60:.0f} min "
                f"— the sleep assertion was lost; elapsed time is not sleep time"
            )
        return out


class NightRunner:
    def __init__(
        self,
        config: MorpheusConfig,
        *,
        controller: CueController,
        player: CuePlayer,
        asset: CueAsset,
        registry: CueAssetRegistry,
        feature_store: FeatureStore,
        cue_store: CueStore,
        source: Optional[FrameSource] = None,
        dry_run: bool = False,
    ) -> None:
        self._cfg = config
        self._controller = controller
        self._player = player
        self._asset = asset
        self._registry = registry
        self._features = feature_store
        self._cues = cue_store
        self._source = source
        self._dry_run = dry_run

        self._pipeline = VisionPipeline(config) if source is not None else None
        self._aggregator = Aggregator() if source is not None else None
        self._health = HealthCounters()
        self._stop = False
        self._notes: list[str] = []
        self._cue_ids: dict[int, int] = {}  # repetition_index -> cues.id
        self._cues_played = 0
        self._cues_failed = 0
        self._outcomes: dict[str, int] = {}

    def request_stop(self, *_: object) -> None:
        self._stop = True

    def run(self, *, hours: float, session_id: int) -> NightSummary:
        # Verify the asset before a single cue can fire. If the audio on disk no
        # longer matches what was registered, the trained/untrained label is
        # meaningless and the night's data would be quietly unanalysable.
        if not self._registry.verify(self._asset):
            raise AudioError(
                f"cue asset '{self._asset.name}' does not match its registered hash. "
                f"The file has changed since registration, so its trained/untrained "
                f"status can no longer be trusted. Re-register it before cueing."
            )

        started = time.monotonic()
        deadline = started + hours * 3600.0
        # Wall clock is tracked alongside monotonic for two reasons: it is the
        # only way to see a suspend (see SUSPEND_DETECT_S), and it bounds the
        # run in real time. Without the wall deadline a machine that slept for
        # three hours would keep cueing three hours past the intended wake —
        # into the morning, with the user awake and out of bed, since
        # `stop_before_wake` is reckoned in monotonic time too.
        started_wall = time.time()
        wall_deadline = started_wall + hours * 3600.0
        status = "completed"

        if self._source is not None:
            self._source.open()

        self._controller.arm(sleep_onset_mono=started, expected_wake_mono=deadline)
        log.info(
            "night armed: %s cue '%s', %s",
            "DRY RUN — no audio" if self._dry_run else "audio armed",
            self._asset.name,
            "camera active" if self._source else "clock only (no camera)",
        )

        try:
            while not self._stop and time.monotonic() < deadline and time.time() < wall_deadline:
                frame = self._next_frame()
                if frame is None:
                    if self._source is not None and self._source.exhausted:
                        self._notes.append("frame source exhausted")
                        break
                    continue

                self._features.append(frame)
                command = self._controller.step(frame)
                if command is not None:
                    self._deliver(session_id, command, frame)

                self._drain_events(session_id)

                if self._controller.state is CueState.HALTED:
                    self._notes.append("controller halted; monitoring stopped")
                    break
        except AudioError as exc:
            status = "aborted_audio_error"
            self._notes.append(str(exc))
            log.error("audio error: %s", exc)
        except Exception as exc:  # noqa: BLE001 - an overnight run must not die loudly
            status = "aborted_error"
            self._notes.append(f"{type(exc).__name__}: {exc}")
            log.exception("unexpected error; stopping cleanly")
        finally:
            if self._aggregator is not None:
                tail = self._aggregator.flush()
                if tail is not None:
                    self._features.append(tail)
            self._controller.finish(time.monotonic())
            self._drain_events(session_id)
            self._features.flush()
            if self._source is not None:
                self._source.close()

        if self._stop and status == "completed":
            status = "stopped_by_user"

        elapsed = time.monotonic() - started
        wall_elapsed = time.time() - started_wall
        # Clamp: wall clock can step backwards over an NTP correction, and a
        # negative suspend is meaningless.
        suspended = max(0.0, wall_elapsed - elapsed)
        if suspended >= SUSPEND_DETECT_S:
            log.warning(
                "machine appears to have been suspended for %.0f s during the run; "
                "the sleep assertion was lost", suspended,
            )
            self._notes.append(f"suspended ~{suspended / 60:.0f} min mid-run")

        return NightSummary(
            session_id=session_id,
            status=status,
            duration_s=elapsed,
            intended_s=hours * 3600.0,
            wall_elapsed_s=wall_elapsed,
            suspended_s=suspended,
            cues_played=self._cues_played,
            cues_failed=self._cues_failed,
            outcomes=dict(self._outcomes),
            halted_reason=self._controller._supervisor.halt_reason,  # noqa: SLF001
            final_state=self._controller.state.value,
            camera_used=self._source is not None,
            health=self._health,
            gate_blocks=dict(self._controller.gate_blocks),
            notes=self._notes,
        )

    # ------------------------------------------------------------- internals

    def _next_frame(self) -> Optional[FeatureFrame]:
        """One aggregated second, from the camera or from the clock."""
        if self._source is None or self._pipeline is None or self._aggregator is None:
            time.sleep(1.0)
            now = time.monotonic()
            # n_frames=0 is the signal to the gate stack that no camera data
            # exists, so the sensing gates report `unavailable` rather than
            # silently passing on zeroed values.
            return FeatureFrame(
                t_mono=now, t_utc=time.time(), n_frames=0,
                signal_quality=0.0, face_present=0.0, eye_region_usable=0.0,
                coverage_flag=CoverageFlag.NO_DETECTOR,
                global_motion=0.0, bed_motion=0.0, face_motion=0.0,
            )

        raw = self._source.read()
        if raw is None:
            self._health.read_failures += 1
            return None
        self._health.frames_captured += 1
        sample = self._pipeline.process(raw)
        completed = self._aggregator.add(sample)
        if completed is not None:
            self._health.seconds_recorded += 1
        return completed

    def _deliver(self, session_id: int, command, frame: FeatureFrame) -> None:
        cue_id = self._cues.begin_cue(
            session_id, command, frame.t_mono,
            asset_id=self._asset.id, asset_sha256=self._asset.sha256,
        )
        self._cue_ids[command.repetition_index] = cue_id

        if self._dry_run:
            log.info(
                "DRY RUN: would play '%s' at gain %.3f (ramp %.0f ms)",
                self._asset.name, command.gain, command.ramp_ms,
            )
            self._cues.complete_cue(cue_id, played=False, error="dry_run")
            self._controller.record_cue_played(frame.t_mono, success=True)
            self._cues_played += 1
            return

        try:
            waveform, _ = self._registry.load(self._asset)
            rendered = self._player.render(
                waveform, gain=command.gain, ramp_ms=command.ramp_ms,
                duration_ms=command.duration_ms,
            )
            self._player.play(rendered, blocking=True)
        except (AudioError, OSError, ValueError) as exc:
            log.error("cue playback failed: %s", exc)
            self._cues.complete_cue(cue_id, played=False, error=str(exc))
            self._controller.record_cue_played(frame.t_mono, success=False)
            self._cues_failed += 1
            self._cues.record_event(
                session_id, frame.t_mono, EventKind.SIGNAL_UNAVAILABLE,
                features={"audio_error": str(exc)},
            )
            return

        self._cues.complete_cue(cue_id, played=True)
        self._controller.record_cue_played(frame.t_mono, success=True)
        self._cues_played += 1
        self._cues.record_event(
            session_id, frame.t_mono,
            EventKind.CUE_DELIVERED_DURING_DETECTED_ACTIVITY,
            features={"gain": command.gain, "asset": self._asset.name},
        )

    def _drain_events(self, session_id: int) -> None:
        """Persist controller events, then clear them."""
        for event in self._controller.events:
            if event.kind == "cue_outcome":
                index = event.payload.get("index")
                assessment = event.payload.get("assessment")
                cue_id = self._cue_ids.get(index)
                if cue_id is not None and assessment is not None:
                    self._cues.record_outcome(
                        cue_id, self._controller._cfg.outcome.observe_window_s, assessment  # noqa: SLF001
                    )
                name = event.payload.get("outcome", "unknown")
                self._outcomes[name] = self._outcomes.get(name, 0) + 1
                kind = _OUTCOME_EVENTS.get(name)
                if kind is not None:
                    self._cues.record_event(session_id, event.t_mono, kind)
            elif event.kind == "probable_arousal":
                self._cues.record_event(
                    session_id, event.t_mono, EventKind.PROBABLE_AROUSAL,
                    features=event.payload,
                )
        self._controller.events.clear()
