"""Facial landmarks, behind an interface that tolerates their absence.

The design calls for MediaPipe wrapped so it can be swapped or bypassed, with
the motion pipeline working at zero landmarks (design.md §8). That is not
defensive coding for its own sake: landmark detection is expected to fail on
closed eyes, IR monochrome, and the heavy head roll of a side sleeper, which is
most of the night this system exists to observe.

So there are two providers. MediaPipe gives 478 points and real eyelid contours
when it works. YuNet gives five points and always works, because it is already
required for M0 presence detection. The pipeline degrades from one to the other
to nothing, and records which it had — `landmark_availability` is itself a
feature, not an error condition.

MediaPipe runs in VIDEO mode rather than IMAGE mode. It tracks between frames
instead of re-detecting each time, which is both faster and steadier — and
steadiness matters more than raw accuracy here, since the signal of interest is
a sub-pixel change measured across consecutive frames.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np

from ..types import PresenceObservation

log = logging.getLogger("morpheus.landmarks")

FACE_LANDMARKER_FILENAME = "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

# MediaPipe Face Mesh indices. The eyelid rings are what M1 actually needs: the
# corneal bulge deforms the lid surface, so lid contour geometry is the closest
# observable proxy for eye movement behind a closed lid.
LEFT_EYE_RING = (
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    398, 384, 385, 386, 387, 388, 466,
)
RIGHT_EYE_RING = (
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
)
LEFT_EYE_UPPER = (386, 385, 387)
LEFT_EYE_LOWER = (374, 380, 373)
RIGHT_EYE_UPPER = (159, 158, 160)
RIGHT_EYE_LOWER = (145, 153, 144)
LEFT_EYE_CORNERS = (362, 263)
RIGHT_EYE_CORNERS = (33, 133)
NOSE_TIP = 1


@dataclass
class LandmarkSet:
    """Landmarks for one frame, in pixel coordinates.

    `dense` is True for a full mesh and False for the five-point fallback. Code
    downstream must check it before asking for eyelid geometry, which only the
    mesh can provide.
    """

    points: np.ndarray            # (N, 2) pixel coordinates
    dense: bool
    source: str
    right_eye_centre: tuple[float, float]
    left_eye_centre: tuple[float, float]
    interocular_px: float
    yaw_proxy: float
    roll_deg: float
    pitch_deg: Optional[float] = None
    confidence: float = 1.0

    def ring(self, indices: Sequence[int]) -> Optional[np.ndarray]:
        if not self.dense or self.points.shape[0] <= max(indices):
            return None
        return self.points[list(indices)]


class LandmarkProvider(Protocol):
    @property
    def available(self) -> bool: ...

    @property
    def name(self) -> str: ...

    def detect(self, image: np.ndarray, t_ms: int) -> Optional[LandmarkSet]: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class MediaPipeLandmarks:
    """478-point mesh via MediaPipe Tasks, in stateful VIDEO mode.

    Expected to fail often in the target conditions. That is handled by
    returning None rather than raising, so a night degrades to the five-point
    fallback rather than ending.
    """

    def __init__(self, model_path: Path, *, min_confidence: float = 0.3) -> None:
        self._path = Path(model_path)
        self._landmarker = None
        self._available = False
        self._status = "not initialised"
        self._min_confidence = min_confidence
        self._last_t_ms = -1
        self._initialised = False

    def _ensure(self) -> None:
        """Load the model on first use.

        Construction is deliberately cheap. The 3.7 MB model takes a second or
        two to load, and VisionPipeline is constructed in plenty of places that
        never see a frame — every test that checks aggregation arithmetic, for
        one. Paying that cost eagerly turned the suite from 19 seconds into
        minutes before this was made lazy.
        """
        if self._initialised:
            return
        self._initialised = True
        self._init(self._min_confidence)

    def _init(self, min_confidence: float) -> None:
        if not self._path.exists():
            self._status = (
                f"model not found at {self._path}; run `morpheus setup-models`. "
                f"Falling back to five-point landmarks."
            )
            return
        try:
            from mediapipe.tasks import python as mpy
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            self._status = f"mediapipe not installed ({exc}); using five-point fallback"
            return

        try:
            options = vision.FaceLandmarkerOptions(
                base_options=mpy.BaseOptions(model_asset_path=str(self._path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=min_confidence,
                min_face_presence_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:  # noqa: BLE001 - any init failure means fallback
            self._status = f"could not create landmarker: {exc}"
            return

        self._available = True
        self._status = "ok"

    @property
    def available(self) -> bool:
        self._ensure()
        return self._available

    @property
    def name(self) -> str:
        return "mediapipe-478"

    @property
    def status(self) -> str:
        self._ensure()
        return self._status

    def detect(self, image: np.ndarray, t_ms: int) -> Optional[LandmarkSet]:
        self._ensure()
        if not self._available or self._landmarker is None:
            return None

        import mediapipe as mp

        # VIDEO mode requires strictly increasing timestamps. A repeated or
        # out-of-order value raises, which would take down the night for a
        # clock stutter, so it is clamped instead.
        if t_ms <= self._last_t_ms:
            t_ms = self._last_t_ms + 1
        self._last_t_ms = t_ms

        rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        try:
            result = self._landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
                t_ms,
            )
        except Exception:  # noqa: BLE001 - a bad frame must not end the night
            return None

        if not result.face_landmarks:
            return None

        height, width = image.shape[:2]
        points = np.array(
            [[lm.x * width, lm.y * height] for lm in result.face_landmarks[0]], dtype=np.float32
        )
        return _build_set(points, dense=True, source=self.name, shape=(height, width))

    def reset(self) -> None:
        self._last_t_ms = -1

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # noqa: BLE001
                pass
            self._landmarker = None


class YuNetLandmarks:
    """Five-point fallback, derived from the M0 presence detector.

    Enough to locate the eyes and estimate pose, which keeps ROI tracking alive
    when the mesh fails. Not enough for eyelid contours, so `dense` is False and
    lid-displacement features go unpopulated rather than being invented.
    """

    def __init__(self, presence_detector) -> None:
        self._detector = presence_detector

    @property
    def available(self) -> bool:
        return bool(getattr(self._detector, "available", False))

    @property
    def name(self) -> str:
        return "yunet-5"

    def from_observation(
        self, observation: PresenceObservation, shape: tuple[int, int]
    ) -> Optional[LandmarkSet]:
        if not observation.face_present or observation.right_eye is None or observation.left_eye is None:
            return None
        points = np.array([observation.right_eye, observation.left_eye], dtype=np.float32)
        return LandmarkSet(
            points=points,
            dense=False,
            source=self.name,
            right_eye_centre=observation.right_eye,
            left_eye_centre=observation.left_eye,
            interocular_px=observation.interocular_px,
            yaw_proxy=observation.yaw_proxy,
            roll_deg=observation.roll_deg,
            confidence=observation.confidence,
        )

    def detect(self, image: np.ndarray, t_ms: int) -> Optional[LandmarkSet]:
        observation = self._detector.detect(image)
        return self.from_observation(observation, image.shape[:2])

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


def _build_set(
    points: np.ndarray, *, dense: bool, source: str, shape: tuple[int, int]
) -> LandmarkSet:
    right = _centroid(points, RIGHT_EYE_CORNERS)
    left = _centroid(points, LEFT_EYE_CORNERS)
    dx, dy = left[0] - right[0], left[1] - right[1]
    interocular = math.hypot(dx, dy)
    roll = math.degrees(math.atan2(dy, dx))

    nose = tuple(points[NOSE_TIP]) if points.shape[0] > NOSE_TIP else ((right[0] + left[0]) / 2, right[1])
    yaw = _yaw_from(right, left, nose, interocular, dx, dy)
    pitch = _pitch_from(points, right, left, interocular) if dense else None

    return LandmarkSet(
        points=points,
        dense=dense,
        source=source,
        right_eye_centre=right,
        left_eye_centre=left,
        interocular_px=interocular,
        yaw_proxy=yaw,
        roll_deg=roll,
        pitch_deg=pitch,
    )


def _centroid(points: np.ndarray, indices: Sequence[int]) -> tuple[float, float]:
    subset = points[list(indices)]
    return (float(subset[:, 0].mean()), float(subset[:, 1].mean()))


def _yaw_from(right, left, nose, interocular, dx, dy) -> float:
    """Same roll-invariant construction as the M0 presence estimator."""
    if interocular < 1e-6:
        return 0.0
    mid = ((right[0] + left[0]) * 0.5, (right[1] + left[1]) * 0.5)
    ux, uy = dx / interocular, dy / interocular
    return float(((nose[0] - mid[0]) * ux + (nose[1] - mid[1]) * uy) / interocular)


def _pitch_from(points: np.ndarray, right, left, interocular: float) -> Optional[float]:
    """Nose-tip vertical offset from the eye line, normalised by eye spacing.

    A proxy in the same spirit as the yaw one: dimensionless, roll-corrected
    only to first order, and adequate for bucketing posture rather than for
    geometry.
    """
    if interocular < 1e-6 or points.shape[0] <= NOSE_TIP:
        return None
    mid_y = (right[1] + left[1]) * 0.5
    return float((points[NOSE_TIP][1] - mid_y) / interocular)


def eye_aspect_ratio(landmarks: LandmarkSet, *, left: bool) -> Optional[float]:
    """Vertical opening over horizontal width for one eye.

    Near zero with the lid closed, which is the normal state all night. Its
    *variation* while closed is the interesting part — that is lid surface
    movement, the only thing a silicon sensor can see of an eye behind a lid.
    """
    if not landmarks.dense:
        return None
    upper = landmarks.ring(LEFT_EYE_UPPER if left else RIGHT_EYE_UPPER)
    lower = landmarks.ring(LEFT_EYE_LOWER if left else RIGHT_EYE_LOWER)
    corners = landmarks.ring(LEFT_EYE_CORNERS if left else RIGHT_EYE_CORNERS)
    if upper is None or lower is None or corners is None:
        return None
    width = float(np.linalg.norm(corners[0] - corners[1]))
    if width < 1e-6:
        return None
    height = float(np.mean(np.linalg.norm(upper - lower, axis=1)))
    return height / width
