"""Per-frame signal quality, and detection of the camera being moved.

Quality is not a nicety. For a side sleeper, `signal_unavailable` will be the
modal state of the night (design.md §11), and the difference between "no eye
movement was detected" and "nothing could be seen" is the difference between a
finding and an artefact. This module is what keeps those apart.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..config import QualityConfig
from ..types import QualityMetrics


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class QualityAssessor:
    """Stateful because scene-change detection needs the previous frame."""

    def __init__(self, config: QualityConfig, scene_downscale: int = 8) -> None:
        self._cfg = config
        self._scene_downscale = max(1, scene_downscale)
        self._prev_small: Optional[np.ndarray] = None

    def assess(self, image: np.ndarray) -> QualityMetrics:
        gray = to_gray(image)

        luminance_mean = float(np.mean(gray))
        luminance_std = float(np.std(gray))
        total = gray.size
        saturated = float(np.count_nonzero(gray >= 250) / total)
        underexposed = float(np.count_nonzero(gray <= 5) / total)

        # Variance of the Laplacian: the standard cheap focus proxy.
        #
        # Known limitation, relevant precisely in our operating conditions: this
        # measures high-frequency energy, and sensor noise is high-frequency.
        # A very dark, high-gain IR frame therefore scores as *well focused*
        # when it is in fact useless. Blurring first suppresses the noise floor
        # without removing genuine edges, which narrows the gap but does not
        # close it. Estimating and subtracting the noise floor is an M1 job,
        # once real overnight footage exists to characterise it against.
        focus = float(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F).var())

        scene_change = self._scene_change(gray)

        return QualityMetrics(
            luminance_mean=luminance_mean,
            luminance_std=luminance_std,
            saturated_fraction=saturated,
            underexposed_fraction=underexposed,
            focus=focus,
            scene_change=scene_change,
            score=self._score(
                luminance_mean, saturated, underexposed, focus, scene_change
            ),
        )

    def _scene_change(self, gray: np.ndarray) -> float:
        """Fraction of the frame that changed since the last frame.

        A person moving changes part of the frame; the camera being bumped
        changes nearly all of it at once. Reporting the *fraction of changed
        area* rather than raw difference energy is what lets a caller tell
        those apart with a single threshold.
        """
        small = cv2.resize(
            gray,
            (
                max(1, gray.shape[1] // self._scene_downscale),
                max(1, gray.shape[0] // self._scene_downscale),
            ),
            interpolation=cv2.INTER_AREA,
        )
        if self._prev_small is None or self._prev_small.shape != small.shape:
            self._prev_small = small
            return 0.0
        diff = cv2.absdiff(small, self._prev_small)
        self._prev_small = small
        return float(np.count_nonzero(diff > 12) / diff.size)

    def _score(
        self,
        luminance: float,
        saturated: float,
        underexposed: float,
        focus: float,
        scene_change: float,
    ) -> float:
        """Composite quality in [0, 1].

        A product of soft penalties rather than a weighted sum: any single
        disqualifying condition (pitch black, blown out, camera flying through
        the air) should drive the score to zero regardless of how good the
        others look. A sum would let three mediocre terms outvote one fatal one.
        """
        cfg = self._cfg
        terms = [
            _ramp(luminance, cfg.min_luminance, cfg.min_luminance * 2.5),
            1.0 - _ramp(luminance, cfg.max_luminance - 20, cfg.max_luminance),
            1.0 - _ramp(saturated, cfg.max_saturated_fraction * 0.5, cfg.max_saturated_fraction),
            1.0 - _ramp(
                underexposed,
                cfg.max_underexposed_fraction * 0.7,
                cfg.max_underexposed_fraction,
            ),
            _ramp(focus, cfg.min_focus * 0.5, cfg.min_focus * 2.0),
            1.0 - _ramp(scene_change, cfg.scene_change_threshold, min(1.0, cfg.scene_change_threshold * 2)),
        ]
        score = 1.0
        for term in terms:
            score *= float(np.clip(term, 0.0, 1.0))
        return float(np.clip(score, 0.0, 1.0))

    def reset(self) -> None:
        """Forget the previous frame, e.g. after a reconnect."""
        self._prev_small = None


def _ramp(value: float, low: float, high: float) -> float:
    """Linear 0->1 ramp between low and high, clamped outside."""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))
