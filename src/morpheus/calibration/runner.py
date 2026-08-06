"""Running the guided calibration session.

Kept separate from the CLI so the whole procedure can be driven from recorded
video in tests, without a camera, a person, or fifteen minutes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..capture.source import FrameSource
from ..config import MorpheusConfig
from ..vision.pipeline import VisionPipeline
from .protocol import PROTOCOL, Segment

log = logging.getLogger("morpheus.calibration")


@dataclass
class SegmentRecording:
    key: str
    samples: list[dict] = field(default_factory=list)
    aborted: bool = False


class CalibrationRunner:
    """Captures one segment at a time, extracting per-frame eye features.

    Samples are kept at frame rate rather than aggregated to 1 Hz. The positive
    control compares distributions, and collapsing thirty frames into one mean
    would discard most of the evidence and leave far too few points to compute a
    meaningful AUC from a thirty-second segment.
    """

    def __init__(
        self,
        config: MorpheusConfig,
        source: FrameSource,
        *,
        pipeline: Optional[VisionPipeline] = None,
    ) -> None:
        if not config.eye.enabled:
            raise ValueError(
                "calibration requires eye tracking to be enabled; the positive "
                "control measures the eye-flow index and there is nothing to "
                "measure without it"
            )
        self._cfg = config
        self._source = source
        self._pipeline = pipeline or VisionPipeline(config)

    def record_segment(
        self,
        segment: Segment,
        *,
        on_progress: Optional[Callable[[float], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> SegmentRecording:
        """Capture one segment. Inter-frame state is reset at the boundary."""
        self._pipeline.reset()
        recording = SegmentRecording(key=segment.key)
        deadline = clock() + segment.seconds

        while clock() < deadline:
            frame = self._source.read()
            if frame is None:
                if self._source.exhausted:
                    break
                continue
            sample = self._pipeline.process(frame)
            recording.samples.append(_to_row(sample))
            if on_progress is not None:
                remaining = max(0.0, deadline - clock())
                on_progress(1.0 - remaining / max(1e-6, segment.seconds))

        return recording

    def run_all(
        self,
        *,
        before_segment: Optional[Callable[[Segment], bool]] = None,
        on_progress: Optional[Callable[[float], None]] = None,
        segments=PROTOCOL,
    ) -> dict[str, list[dict]]:
        """Run every segment. `before_segment` returning False skips one."""
        collected: dict[str, list[dict]] = {}
        for segment in segments:
            if before_segment is not None and not before_segment(segment):
                continue
            recording = self.record_segment(segment, on_progress=on_progress)
            collected[segment.key] = recording.samples
            log.info("segment %s: %d samples", segment.key, len(recording.samples))
        return collected


def _to_row(sample) -> dict:
    """Flatten a RawSample into the fields the profile builder consumes.

    Eye flow is the mean of whichever eyes were measurable. Averaging is right
    here rather than requiring both: during the posture segments one eye is
    frequently occluded, and discarding those frames would throw away exactly
    the data the posture question needs.
    """
    eye = sample.eye
    flow = None
    if eye is not None:
        values = [v for v in (eye.flow_left, eye.flow_right) if v is not None]
        flow = sum(values) / len(values) if values else None

    return {
        "t_mono": sample.t_mono,
        "eye_flow": flow,
        "bilateral": getattr(eye, "bilateral_corr", None) if eye else None,
        "lid_disp": (
            getattr(eye, "lid_disp_left", None) or getattr(eye, "lid_disp_right", None)
            if eye
            else None
        ),
        "face_present": bool(sample.presence.face_present),
        "interocular": sample.presence.interocular_px or None,
        "quality": sample.quality.score,
        "coverage": sample.coverage.value,
        "motion": sample.motion.global_motion,
    }
