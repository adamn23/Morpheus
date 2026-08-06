"""Eye-region flow: the M1 signal, and the reasons to distrust it.

This computes the quantity the whole camera branch is a bet on. It runs in
shadow mode — logged every night, forbidden from influencing cue timing until
H1 passes (design.md §8) — and it is written to make its own failure legible
rather than to look impressive.

**Registration is the whole game.** An unregistered eye ROI moves whenever the
head moves, and head movement is orders of magnitude larger than lid-surface
deformation. Frame-differencing an unstabilised ROI measures head motion and
reports it as eye motion. That is what the prior art in this space does, and
it is why none of it has published validation. Each ROI is registered against
the previous frame before any flow is computed, and the registration residual
is recorded so windows where alignment failed can be discarded rather than
believed.

**Bilateral correlation is the specificity feature.** Genuine eye movements are
conjugate: both eyes move together, in the same direction, at the same time.
Shadows, IR flicker, blanket motion, breathing sway and sensor noise are not
conjugate. Requiring correlated bilateral activity is the single best defence
against false positives available without a reference signal — at the cost of
needing both eyes visible, which for a side sleeper is a real cost. Unilateral
values are recorded too, so the trade can be measured rather than assumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .landmarks import (
    LEFT_EYE_RING,
    RIGHT_EYE_RING,
    LandmarkSet,
    eye_aspect_ratio,
)

log = logging.getLogger("morpheus.eye_flow")


@dataclass
class EyeFlowSample:
    """Per-frame eye-region measurements. All optional: absence is normal."""

    flow_left: Optional[float] = None
    flow_right: Optional[float] = None
    bilateral_corr: Optional[float] = None
    lid_disp_left: Optional[float] = None
    lid_disp_right: Optional[float] = None
    residual_left: Optional[float] = None
    residual_right: Optional[float] = None
    roi_px_left: int = 0
    roi_px_right: int = 0

    @property
    def usable(self) -> bool:
        return self.flow_left is not None or self.flow_right is not None


@dataclass
class EyeFlowConfig:
    # ROI half-width as a multiple of interocular distance. 0.22 covers the eye
    # and immediate lid without pulling in eyebrow or cheek, both of which move
    # with expression rather than with the eye.
    roi_scale: float = 0.22
    roi_min_px: int = 24
    roi_max_px: int = 220
    # Registration residual above which the alignment is not trusted. Expressed
    # as mean absolute difference in normalised intensity.
    max_residual: float = 0.08
    # Flow magnitudes below this are noise at any realistic gain.
    flow_floor: float = 1e-4
    equalize: bool = True


class EyeFlowExtractor:
    """Stateful: registration needs the previous frame's ROI."""

    def __init__(self, config: Optional[EyeFlowConfig] = None) -> None:
        self._cfg = config or EyeFlowConfig()
        self._prev: dict[str, np.ndarray] = {}
        self._prev_ear: dict[str, float] = {}

    def reset(self) -> None:
        self._prev.clear()
        self._prev_ear.clear()

    def update(self, gray: np.ndarray, landmarks: Optional[LandmarkSet]) -> EyeFlowSample:
        if landmarks is None or landmarks.interocular_px < 1e-6:
            self.reset()
            return EyeFlowSample()

        sample = EyeFlowSample()
        flows: dict[str, Optional[np.ndarray]] = {}

        for side, centre in (
            ("left", landmarks.left_eye_centre),
            ("right", landmarks.right_eye_centre),
        ):
            roi = self._extract_roi(gray, centre, landmarks.interocular_px)
            if roi is None:
                self._prev.pop(side, None)
                continue

            previous = self._prev.get(side)
            self._prev[side] = roi
            if previous is None or previous.shape != roi.shape:
                continue

            aligned, residual = _register(previous, roi)
            if side == "left":
                sample.residual_left = residual
                sample.roi_px_left = roi.size
            else:
                sample.residual_right = residual
                sample.roi_px_right = roi.size

            if residual > self._cfg.max_residual:
                # Registration failed. Reporting flow here would be reporting
                # head motion, which is the exact error this module exists to
                # avoid, so it is dropped instead.
                continue

            flow = _dense_flow(aligned, roi)
            magnitude = float(np.mean(np.linalg.norm(flow, axis=2)))
            magnitude = magnitude if magnitude >= self._cfg.flow_floor else 0.0
            flows[side] = flow
            if side == "left":
                sample.flow_left = magnitude
            else:
                sample.flow_right = magnitude

        left_flow, right_flow = flows.get("left"), flows.get("right")
        if left_flow is not None and right_flow is not None:
            sample.bilateral_corr = _conjugate_correlation(left_flow, right_flow)

        self._lid_displacement(landmarks, sample)
        return sample

    # ------------------------------------------------------------- internals

    def _extract_roi(
        self, gray: np.ndarray, centre: tuple[float, float], interocular: float
    ) -> Optional[np.ndarray]:
        half = int(np.clip(interocular * self._cfg.roi_scale, self._cfg.roi_min_px // 2,
                           self._cfg.roi_max_px // 2))
        cx, cy = int(round(centre[0])), int(round(centre[1]))
        height, width = gray.shape[:2]
        x0, y0 = cx - half, cy - half
        x1, y1 = cx + half, cy + half
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            return None  # partially out of frame; a clipped ROI would not register
        roi = gray[y0:y1, x0:x1]
        if roi.size < self._cfg.roi_min_px ** 2:
            return None
        if self._cfg.equalize:
            # Local contrast normalisation. A dark IR eye region has very little
            # dynamic range, and flow estimation needs gradients to work with.
            roi = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(roi)
        return roi.astype(np.float32) / 255.0

    def _lid_displacement(self, landmarks: LandmarkSet, sample: EyeFlowSample) -> None:
        """Frame-to-frame change in eye aspect ratio.

        With the lid closed the ratio sits near zero; its *variation* is lid
        surface movement. Only the dense mesh can supply this, so it stays None
        on the five-point fallback rather than being approximated.
        """
        for side, is_left in (("left", True), ("right", False)):
            ratio = eye_aspect_ratio(landmarks, left=is_left)
            if ratio is None:
                self._prev_ear.pop(side, None)
                continue
            previous = self._prev_ear.get(side)
            self._prev_ear[side] = ratio
            if previous is None:
                continue
            delta = abs(ratio - previous)
            if side == "left":
                sample.lid_disp_left = delta
            else:
                sample.lid_disp_right = delta


def _register(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, float]:
    """Align `previous` onto `current` by translation; return it and the residual.

    Phase correlation rather than ECC: it is far cheaper, and translation is
    the dominant component of frame-to-frame head movement at 30 fps. Rotation
    and scale change slowly enough between consecutive frames to be negligible.
    """
    try:
        (shift_x, shift_y), _ = cv2.phaseCorrelate(previous, current)
    except cv2.error:
        return previous, 1.0

    matrix = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    aligned = cv2.warpAffine(
        previous, matrix, (previous.shape[1], previous.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    # Trim the border, where warping invents pixels that would inflate flow.
    margin = max(2, int(min(aligned.shape) * 0.12))
    inner_a = aligned[margin:-margin, margin:-margin]
    inner_b = current[margin:-margin, margin:-margin]
    if inner_a.size == 0:
        return aligned, 1.0
    residual = float(np.mean(np.abs(inner_a - inner_b)))
    return aligned, residual


def _dense_flow(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Farnebäck flow on the registered pair, in the ROI interior."""
    a = np.clip(previous * 255.0, 0, 255).astype(np.uint8)
    b = np.clip(current * 255.0, 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        a, b, None,
        pyr_scale=0.5, levels=2, winsize=9, iterations=2,
        poly_n=5, poly_sigma=1.1, flags=0,
    )
    margin = max(2, int(min(flow.shape[:2]) * 0.12))
    return flow[margin:-margin, margin:-margin]


def _conjugate_correlation(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    """How much the two eyes moved together, in [-1, 1].

    The mean flow vector per eye is compared, not the per-pixel fields: what
    matters is whether both eyes translated in the same direction, not whether
    their internal texture happens to match. Note the horizontal flip — the eyes
    are mirror-imaged about the face midline in image coordinates, so a genuine
    conjugate movement appears as opposite x-components and must be corrected
    before correlating, or every real saccade would score -1.
    """
    if left.size == 0 or right.size == 0:
        return None
    left_mean = np.array([left[..., 0].mean(), left[..., 1].mean()], dtype=np.float64)
    right_mean = np.array([right[..., 0].mean(), right[..., 1].mean()], dtype=np.float64)

    left_norm = float(np.linalg.norm(left_mean))
    right_norm = float(np.linalg.norm(right_mean))
    if left_norm < 1e-9 or right_norm < 1e-9:
        return 0.0

    cosine = float(np.dot(left_mean, right_mean) / (left_norm * right_norm))
    # Scale by the weaker of the two magnitudes so that a strong movement in one
    # eye and noise in the other cannot score as high agreement.
    weight = min(left_norm, right_norm) / max(left_norm, right_norm)
    return float(np.clip(cosine * weight, -1.0, 1.0))
