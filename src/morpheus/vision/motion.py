"""Gross body-motion energy.

This is the highest value-per-unit-effort component in M0, and the one part of
the camera pipeline that does not depend on the face being visible. It works at
any pose, in the dark, under blankets, from any angle — which is precisely why
the recommended architecture gives the camera a gate-and-guard role built on
motion rather than an eye-tracking role built on hope (design.md §8).

From M2 these values feed gates G4 and G5: suppress a cue while the body is
moving, and detect probable arousal after one.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..config import MotionConfig
from ..types import MotionObservation, PresenceObservation
from .quality import to_gray


class MotionEstimator:
    """Frame-differencing motion energy, globally and within regions.

    Deliberately simple. Optical flow would give direction as well as
    magnitude, but for the gate-and-guard role only magnitude matters, and
    frame differencing costs a fraction of the CPU — which is the binding
    constraint on a passively-cooled laptop running eight hours (design.md §10).
    """

    def __init__(self, config: MotionConfig) -> None:
        self._cfg = config
        self._prev: Optional[np.ndarray] = None

    def update(
        self,
        image: np.ndarray,
        presence: Optional[PresenceObservation] = None,
    ) -> MotionObservation:
        gray = to_gray(image)
        scale = max(1, self._cfg.downscale)
        small = cv2.resize(
            gray,
            (max(1, gray.shape[1] // scale), max(1, gray.shape[0] // scale)),
            interpolation=cv2.INTER_AREA,
        )
        ksize = self._cfg.blur_ksize
        if ksize and ksize >= 3:
            # Blur before differencing: sensor noise in a dark IR frame is
            # spatially uncorrelated and would otherwise dominate the signal.
            small = cv2.GaussianBlur(small, (ksize | 1, ksize | 1), 0)

        if self._prev is None or self._prev.shape != small.shape:
            self._prev = small
            return MotionObservation(0.0, 0.0, 0.0)

        diff = cv2.absdiff(small, self._prev)
        self._prev = small

        global_motion = float(np.mean(diff)) / 255.0
        bed_motion = self._region_mean(diff, self._bed_rect(small.shape))
        face_motion = self._region_mean(diff, self._face_rect(presence, scale, small.shape))

        return MotionObservation(
            global_motion=global_motion,
            bed_motion=bed_motion if bed_motion is not None else global_motion,
            face_motion=face_motion if face_motion is not None else 0.0,
        )

    def _bed_rect(self, shape: tuple[int, ...]) -> Optional[tuple[int, int, int, int]]:
        region = self._cfg.bed_region
        if region is None:
            return None
        h, w = shape[0], shape[1]
        rx, ry, rw, rh = region
        return (int(rx * w), int(ry * h), int(rw * w), int(rh * h))

    @staticmethod
    def _face_rect(
        presence: Optional[PresenceObservation],
        scale: int,
        shape: tuple[int, ...],
    ) -> Optional[tuple[int, int, int, int]]:
        if presence is None or not presence.face_present or presence.bbox is None:
            return None
        x, y, w, h = presence.bbox
        return (x // scale, y // scale, max(1, w // scale), max(1, h // scale))

    @staticmethod
    def _region_mean(
        diff: np.ndarray, rect: Optional[tuple[int, int, int, int]]
    ) -> Optional[float]:
        if rect is None:
            return None
        h, w = diff.shape[:2]
        x, y, rw, rh = rect
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + rw), min(h, y + rh)
        if x1 <= x0 or y1 <= y0:
            return None
        return float(np.mean(diff[y0:y1, x0:x1])) / 255.0

    def reset(self) -> None:
        self._prev = None
