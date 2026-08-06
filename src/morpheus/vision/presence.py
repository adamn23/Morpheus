"""Face presence and coarse pose, via YuNet.

M0 uses YuNet's five landmarks (both eyes, nose tip, both mouth corners) rather
than a full 478-point mesh. That is a deliberate scope choice: M0 asks only
"can the eye region be seen at all", and a five-point answer is enough to
localise the eyes, estimate a roll-invariant yaw proxy, and measure interocular
resolution. MediaPipe's mesh arrives in M1, when the question becomes "can the
eyelid be measured", and it is expected to degrade badly on closed eyes and IR
monochrome (design.md §4.2).

If the model file is absent the detector reports itself unavailable rather than
raising. Motion features do not depend on it, and a night of motion-only data is
worth more than no night at all.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import CoverageConfig, PresenceConfig
from ..types import CoverageFlag, PresenceObservation, QualityMetrics

ABSENT = PresenceObservation(face_present=False)
NO_DETECTOR = PresenceObservation(face_present=False, detector_available=False)


class PresenceDetector:
    def __init__(self, config: PresenceConfig) -> None:
        self._cfg = config
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._input_size: Optional[tuple[int, int]] = None
        self._available = False
        self._reason = "not initialised"
        self._init_detector()

    def _init_detector(self) -> None:
        path = Path(self._cfg.model_path)
        if not path.exists():
            self._reason = (
                f"model not found at {path}; run `morpheus setup-models` to fetch it. "
                "Recording will continue with motion features only."
            )
            return
        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(path), "", (320, 320),
                self._cfg.score_threshold, self._cfg.nms_threshold, self._cfg.top_k,
            )
        except cv2.error as exc:
            self._reason = f"could not load YuNet model: {exc}"
            return
        self._available = True
        self._reason = "ok"

    @property
    def available(self) -> bool:
        return self._available

    @property
    def status(self) -> str:
        return self._reason

    def detect(self, image: np.ndarray) -> PresenceObservation:
        if not self._available or self._detector is None:
            return NO_DETECTOR

        scale = self._cfg.input_scale
        if scale != 1.0:
            work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            work = image
        # YuNet needs 3-channel input; IR cameras commonly deliver mono.
        if work.ndim == 2:
            work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)

        size = (work.shape[1], work.shape[0])
        if size != self._input_size:
            self._detector.setInputSize(size)
            self._input_size = size

        try:
            _, faces = self._detector.detect(work)
        except cv2.error:
            return ABSENT
        if faces is None or len(faces) == 0:
            return ABSENT

        # Largest face by area. A bed partner would break this assumption, but
        # the recorder refuses multi-person scenes at framing check (design.md §20).
        face = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        inv = 1.0 / scale
        return self._observe(face, inv)

    def _observe(self, face: np.ndarray, inv_scale: float) -> PresenceObservation:
        x, y, w, h = (float(v) * inv_scale for v in face[0:4])
        right_eye = (float(face[4]) * inv_scale, float(face[5]) * inv_scale)
        left_eye = (float(face[6]) * inv_scale, float(face[7]) * inv_scale)
        nose = (float(face[8]) * inv_scale, float(face[9]) * inv_scale)
        score = float(face[14])

        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        interocular = math.hypot(dx, dy)
        roll_deg = math.degrees(math.atan2(dy, dx))

        yaw_proxy = _yaw_proxy(right_eye, left_eye, nose, interocular, dx, dy)

        return PresenceObservation(
            face_present=True,
            confidence=score,
            bbox=(int(x), int(y), int(w), int(h)),
            right_eye=right_eye,
            left_eye=left_eye,
            interocular_px=interocular,
            yaw_proxy=yaw_proxy,
            roll_deg=roll_deg,
            detector_available=True,
        )


def _yaw_proxy(
    right_eye: tuple[float, float],
    left_eye: tuple[float, float],
    nose: tuple[float, float],
    interocular: float,
    dx: float,
    dy: float,
) -> float:
    """Signed, roll-invariant head-turn proxy. ~0 frontal, grows when turned.

    The nose tip sits midway between the eyes on a frontal face. As the head
    turns, it projects toward the nearer eye. Measuring that offset *along the
    interocular axis* and normalising by interocular distance makes the result
    invariant to both head roll and distance from the camera — which matters,
    because a sleeper's head rolls constantly and the camera distance is fixed
    but unknown.

    This is a proxy, not degrees. It saturates at large yaw (once the far eye is
    occluded YuNet stops returning a usable landmark at all), which is fine: by
    then the pose is already disqualifying.
    """
    if interocular < 1e-6:
        return 0.0
    eye_mid = ((right_eye[0] + left_eye[0]) * 0.5, (right_eye[1] + left_eye[1]) * 0.5)
    ux, uy = dx / interocular, dy / interocular
    offset = (nose[0] - eye_mid[0]) * ux + (nose[1] - eye_mid[1]) * uy
    return float(offset / interocular)


def classify_coverage(
    presence: PresenceObservation,
    quality: QualityMetrics,
    frame_shape: tuple[int, int],
    config: CoverageConfig,
) -> CoverageFlag:
    """Decide whether this frame could support eye-region analysis.

    This function computes the M0 decision gate. The ordering of the checks is
    meaningful: the returned flag is the *first* disqualifying reason, and the
    distribution of reasons across a night is what tells you whether to remount
    the camera, buy a better sensor, or abandon the eye-tracking branch
    entirely (design.md §23).
    """
    if not presence.detector_available:
        return CoverageFlag.NO_DETECTOR
    if quality.score < config.min_quality_score:
        return CoverageFlag.QUALITY_TOO_LOW
    if not presence.face_present:
        return CoverageFlag.FACE_ABSENT
    if presence.confidence < config.min_detector_confidence:
        return CoverageFlag.FACE_ABSENT
    if abs(presence.yaw_proxy) > config.max_abs_yaw_proxy:
        return CoverageFlag.POSE_UNSUITABLE
    if presence.interocular_px < config.min_interocular_px:
        return CoverageFlag.TOO_SMALL

    height, width = frame_shape[0], frame_shape[1]
    margin = config.frame_edge_margin_px
    for point in (presence.right_eye, presence.left_eye):
        if point is None:
            return CoverageFlag.EYE_OUT_OF_FRAME
        px, py = point
        if not (margin <= px <= width - margin and margin <= py <= height - margin):
            return CoverageFlag.EYE_OUT_OF_FRAME

    return CoverageFlag.USABLE
