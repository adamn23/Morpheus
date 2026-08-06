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
            while not self._stop and time.monotonic() < deadline:
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

        return NightSummary(
            session_id=session_id,
            status=status,
            duration_s=time.monotonic() - started,
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
