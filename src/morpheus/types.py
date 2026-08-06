"""Core value types shared across the daemon.

Naming discipline (design.md §11) is enforced here rather than by convention:
`EventKind` is a closed enum, and `tests/test_naming_discipline.py` fails the
build if forbidden vocabulary appears anywhere in the source tree — identifiers
asserting a stage or a confirmed state. Morpheus does not measure sleep stages,
and the type system should make it awkward to pretend otherwise. See that test
for the banned list and for the escape marker used by prose that must quote it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class EventKind(str, enum.Enum):
    """The complete set of things Morpheus is permitted to claim.

    Every label is deliberately hedged. None of them assert a sleep stage.
    Adding a member is a design decision, not an implementation detail.
    """

    PROBABLE_EYE_MOVEMENT_BURST = "probable_eye_movement_burst"
    POSSIBLE_DREAM_ACTIVITY = "possible_dream_activity"
    PROBABLE_AROUSAL = "probable_arousal"
    POSSIBLE_AWAKENING = "possible_awakening"
    CUE_DELIVERED_DURING_DETECTED_ACTIVITY = "cue_delivered_during_detected_activity"
    SIGNAL_UNAVAILABLE = "signal_unavailable"


class CoverageFlag(str, enum.Enum):
    """Why a given second is or is not usable for eye-region analysis.

    This is the M0 decision-gate instrumentation. Distinguishing *why* coverage
    was lost is what turns a disappointing number into an actionable one: a
    night lost to `FACE_ABSENT` suggests remounting the camera, whereas one lost
    to `POSE_UNSUITABLE` suggests the posture itself is the blocker.
    """

    USABLE = "usable"
    FACE_ABSENT = "face_absent"
    POSE_UNSUITABLE = "pose_unsuitable"  # face seen, turned too far away
    TOO_SMALL = "too_small"  # face seen, insufficient interocular resolution
    EYE_OUT_OF_FRAME = "eye_out_of_frame"
    QUALITY_TOO_LOW = "quality_too_low"  # dark, blurred, saturated, or unstable
    NO_DETECTOR = "no_detector"  # presence model unavailable; motion only


@dataclass(slots=True)
class Frame:
    """One captured image plus the clocks needed to reason about it.

    `t_mono` is the authority for all interval arithmetic; `t_utc` exists only
    for human-readable logging and joining against morning reports. Mixing the
    two is how a DST transition silently corrupts a night (design.md §8).
    """

    seq: int
    t_mono: float
    t_utc: float
    image: np.ndarray

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass(slots=True)
class QualityMetrics:
    """Per-frame assessment of whether the image is worth analysing at all."""

    luminance_mean: float
    luminance_std: float
    saturated_fraction: float  # proportion of pixels at/near 255
    underexposed_fraction: float  # proportion at/near 0
    focus: float  # variance of Laplacian
    scene_change: float  # 0..1, high => camera bumped or moved
    score: float  # composite signal quality in [0, 1]


@dataclass(slots=True)
class PresenceObservation:
    """Output of the face detector for a single frame.

    Pose is estimated from YuNet's five landmarks rather than a full landmark
    mesh. It is a *proxy*, accurate enough to bucket postures and nothing more;
    `yaw_proxy` is a normalised offset, not degrees.
    """

    face_present: bool
    confidence: float = 0.0
    bbox: Optional[tuple[int, int, int, int]] = None  # x, y, w, h
    right_eye: Optional[tuple[float, float]] = None
    left_eye: Optional[tuple[float, float]] = None
    interocular_px: float = 0.0
    yaw_proxy: float = 0.0  # signed, ~0 frontal, |.|>0.35 substantially turned
    roll_deg: float = 0.0
    detector_available: bool = True


@dataclass(slots=True)
class MotionObservation:
    """Gross movement energy. Works at any pose, in the dark, under blankets.

    This is the camera's highest-value M0 output: it underpins the gate-and-guard
    role (design.md §12.2 G4/G5), which — unlike eye tracking — does not depend
    on the face being visible.
    """

    global_motion: float  # whole-frame mean absolute difference, normalised
    bed_motion: float  # same, restricted to the configured bed region
    face_motion: float  # same, restricted to the face bbox when present


@dataclass(slots=True)
class RawSample:
    """Per-frame observations, buffered in memory and never persisted.

    Persisting 30 Hz samples would multiply storage 30x for no analytic gain;
    aggregation to 1 Hz happens before anything touches disk.
    """

    seq: int
    t_mono: float
    t_utc: float
    quality: QualityMetrics
    presence: PresenceObservation
    motion: MotionObservation
    coverage: CoverageFlag
    # Presence runs at ~5 Hz while frames arrive at ~30 Hz, so most samples
    # carry a stale detection forward. Aggregation must count only the fresh
    # ones, or `face_present` becomes a fraction of the detector's cadence
    # rather than a fraction of the second.
    presence_fresh: bool = True
    # M1 shadow-mode measurements. `eye` is None whenever the eye region could
    # not be measured, which is the common case and is not an error.
    eye: object = None
    landmarks_dense: Optional[bool] = None
    pitch: Optional[float] = None


@dataclass(slots=True)
class FeatureFrame:
    """One second of aggregated features. This is the persisted unit.

    Fields that later phases populate (eye flow, landmark availability, head
    pose from a full mesh, respiration) are declared here and left None in M0.
    The schema is stable from the outset so that a night recorded during M0
    remains joinable against one recorded during M4.
    """

    t_mono: float
    t_utc: float
    n_frames: int  # frames actually contributing to this second

    # --- populated in M0 ---
    signal_quality: float
    face_present: float  # fraction of the second with a face detected
    eye_region_usable: float  # fraction of the second usable for eye analysis
    coverage_flag: CoverageFlag  # modal flag for the second
    global_motion: float
    bed_motion: float
    face_motion: float
    yaw_proxy: Optional[float] = None
    roll_deg: Optional[float] = None
    interocular_px: Optional[float] = None
    focus: Optional[float] = None
    luminance_mean: Optional[float] = None
    scene_change: Optional[float] = None

    # --- reserved for later phases; deliberately unpopulated in M0 ---
    landmark_available: Optional[float] = None  # M1
    pitch: Optional[float] = None  # M1
    head_motion: Optional[float] = None  # M1
    eye_flow_l: Optional[float] = None  # M1, shadow mode
    eye_flow_r: Optional[float] = None  # M1, shadow mode
    eye_flow_bilateral_corr: Optional[float] = None  # M1, shadow mode
    lid_disp_l: Optional[float] = None  # M1, shadow mode
    lid_disp_r: Optional[float] = None  # M1, shadow mode
    resp_proxy: Optional[float] = None  # P3, best-effort


@dataclass(slots=True)
class HealthCounters:
    """Uptime accounting for the M0 acceptance criteria (design.md §28).

    A run that quietly drops a third of its frames looks identical to a healthy
    one in the feature table. These counters are what make that visible.
    """

    frames_captured: int = 0
    frames_dropped: int = 0
    read_failures: int = 0
    reconnects: int = 0
    seconds_recorded: int = 0
    clock_gaps: list[tuple[float, float]] = field(default_factory=list)

    @property
    def capture_uptime(self) -> float:
        """Fraction of expected frames actually captured, in [0, 1]."""
        expected = self.frames_captured + self.frames_dropped
        return self.frames_captured / expected if expected else 0.0
