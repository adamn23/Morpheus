"""Live camera capture with exposure verification and reconnection.

The exposure handling here is the fussiest part of M0 and deserves the space.
OpenCV's `set(CAP_PROP_AUTO_EXPOSURE, ...)` is advisory: several backends accept
the call, return success, and change nothing. macOS/AVFoundation in particular
frequently exposes no manual exposure control at all for the built-in camera.

So we do two things rather than trusting one:

  1. try both UVC conventions and read the property back, and
  2. offer an *empirical* stability probe that watches mean luminance on a
     static scene, because auto-exposure hunting in a dark room shows up as a
     slow drift that no property read-back would reveal.

A run with AE silently active does not produce degraded data; it produces
brightness oscillations that every motion feature will faithfully report as
movement. That is fictional data, and it is worse than no data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from ..config import CameraConfig
from ..types import Frame
from .source import FrameSource, FrameSourceError

# UVC exposes manual mode as 1 and aperture-priority (auto) as 3. V4L2 maps
# these onto 0.25 / 0.75. Backends disagree, so we try both and verify.
_MANUAL_EXPOSURE_CANDIDATES = (0.25, 1.0, 0.0)


@dataclass
class ExposureStatus:
    """What we were actually able to establish about exposure control."""

    requested_manual: bool
    applied_value: Optional[float]
    readback_value: Optional[float]
    controllable: bool
    detail: str

    @property
    def manual_confirmed(self) -> bool:
        return self.requested_manual and self.controllable


class WebcamSource(FrameSource):
    def __init__(self, config: CameraConfig) -> None:
        self._cfg = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._seq = 0
        self._exposure = ExposureStatus(False, None, None, False, "not attempted")
        self._backend_name = "unknown"
        self._reconnects = 0

    # ----------------------------------------------------------------- open

    def open(self) -> None:
        cap = self._open_capture()
        self._cap = cap
        self._configure(cap)
        self._warmup(cap)

    def _open_capture(self) -> cv2.VideoCapture:
        cfg = self._cfg
        device: Any = cfg.device
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise FrameSourceError(
                f"could not open camera {device!r}. On macOS, camera access must be "
                f"granted to the binary running this process (System Settings > "
                f"Privacy & Security > Camera), not to the terminal that launched it."
            )
        try:
            self._backend_name = cap.getBackendName()
        except cv2.error:
            self._backend_name = "unknown"
        return cap

    def _configure(self, cap: cv2.VideoCapture) -> None:
        cfg = self._cfg
        # FOURCC before resolution: some UVC cameras only offer the higher
        # resolutions under MJPG, and setting size first silently clamps.
        if cfg.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)

        if cfg.autofocus is not None:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0 if not cfg.autofocus else 1)

        self._exposure = self._apply_manual_exposure(cap)
        if cfg.require_manual_exposure and not self._exposure.manual_confirmed:
            cap.release()
            raise FrameSourceError(
                "manual exposure could not be confirmed on this camera "
                f"({self._exposure.detail}). Auto-exposure hunting in a dark room is "
                "indistinguishable from motion in every downstream feature, so the "
                "recorder refuses to run. Use a UVC camera with exposure control, or "
                "pass --allow-auto-exposure for daylight development only."
            )

        if cfg.gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, cfg.gain)

    def _apply_manual_exposure(self, cap: cv2.VideoCapture) -> ExposureStatus:
        if not self._cfg.require_manual_exposure and self._cfg.exposure is None:
            return ExposureStatus(False, None, None, False, "manual exposure not requested")

        auto_before = _safe_get(cap, cv2.CAP_PROP_AUTO_EXPOSURE)
        applied: Optional[float] = None
        readback: Optional[float] = None
        controllable = False

        for candidate in _MANUAL_EXPOSURE_CANDIDATES:
            if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, candidate):
                continue
            value = _safe_get(cap, cv2.CAP_PROP_AUTO_EXPOSURE)
            if value is not None and _approx(value, candidate):
                applied, readback, controllable = candidate, value, True
                break

        if controllable and self._cfg.exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, self._cfg.exposure)

        detail = (
            f"auto_exposure readback {readback} after setting {applied}"
            if controllable
            else f"no candidate value took effect (property reads {auto_before})"
        )
        return ExposureStatus(True, applied, readback, controllable, detail)

    def _warmup(self, cap: cv2.VideoCapture) -> None:
        """Discard the first frames; sensors and AGC take time to settle."""
        for _ in range(max(0, self._cfg.warmup_frames)):
            cap.read()

    # ----------------------------------------------------------------- read

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            raise FrameSourceError("read() before open()")
        ok, image = self._cap.read()
        # Timestamp immediately after the read returns. This is still not the
        # true exposure instant — there is unmeasured driver latency — but it is
        # consistently biased, which is what interval arithmetic needs.
        t_mono, t_utc = time.monotonic(), time.time()
        if not ok or image is None:
            return None
        self._seq += 1
        return Frame(seq=self._seq, t_mono=t_mono, t_utc=t_utc, image=image)

    def reconnect(self) -> bool:
        """Attempt to reopen after a run of read failures (e.g. USB reset)."""
        self.close()
        time.sleep(self._cfg.reconnect_backoff_s)
        try:
            self.open()
        except FrameSourceError:
            return False
        self._reconnects += 1
        return True

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def exhausted(self) -> bool:
        return False  # a live camera is never exhausted, only broken

    @property
    def reconnects(self) -> int:
        return self._reconnects

    @property
    def exposure_status(self) -> ExposureStatus:
        return self._exposure

    # -------------------------------------------------------------- profile

    def device_profile(self) -> dict[str, Any]:
        cap = self._cap
        return {
            "camera_model": str(self._cfg.device),
            "backend": self._backend_name,
            "width": int(_safe_get(cap, cv2.CAP_PROP_FRAME_WIDTH) or self._cfg.width),
            "height": int(_safe_get(cap, cv2.CAP_PROP_FRAME_HEIGHT) or self._cfg.height),
            "fps": float(_safe_get(cap, cv2.CAP_PROP_FPS) or self._cfg.fps),
            "fourcc": self._cfg.fourcc,
            "manual_exposure": int(self._exposure.manual_confirmed),
            "ir_wavelength_nm": None,
            "mount_geometry": None,
            "audio_device": None,
        }

    # ------------------------------------------------------- empirical probe

    def probe_exposure_stability(self, seconds: float = 3.0) -> dict[str, float]:
        """Watch mean luminance on a static scene to catch AE hunting.

        Property read-back can lie; a drifting mean cannot. Reported as a
        coefficient of variation, so it is comparable across lighting levels.
        Interpretation is left to the caller (`morpheus doctor`) rather than
        baked in here, because the acceptable threshold depends on whether the
        scene really was static.
        """
        if self._cap is None:
            raise FrameSourceError("probe_exposure_stability() before open()")
        means: list[float] = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            frame = self.read()
            if frame is None:
                continue
            gray = _to_gray(frame.image)
            means.append(float(np.mean(gray)))
        if len(means) < 2:
            return {"samples": float(len(means)), "cv": float("nan")}
        arr = np.asarray(means, dtype=np.float64)
        mean = float(arr.mean())
        return {
            "samples": float(arr.size),
            "mean_luminance": mean,
            "std_luminance": float(arr.std()),
            "cv": float(arr.std() / mean) if mean > 1e-6 else float("nan"),
            "range": float(arr.max() - arr.min()),
        }


def _safe_get(cap: Optional[cv2.VideoCapture], prop: int) -> Optional[float]:
    if cap is None:
        return None
    try:
        value = cap.get(prop)
    except cv2.error:
        return None
    return None if value in (0.0, -1.0) and prop == cv2.CAP_PROP_AUTO_EXPOSURE else value


def _approx(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
