"""Cue rendering and playback.

This is the part of Morpheus that can actually harm the user. Everything else
either records or decides; this module makes a sound next to a sleeping person's
head. It is written accordingly.

Three invariants are enforced *here*, in the rendering path, rather than being
left to the caller's discipline:

  1. Output amplitude never exceeds the configured ceiling. Not "should not" —
     the buffer is clipped after gain is applied, so a caller that asks for
     more than the ceiling gets the ceiling.
  2. Every cue ramps up from silence. Abrupt onset is the primary mechanism by
     which a cue wakes someone instead of being incorporated into a dream.
  3. OS volume is never touched. It is global, racy, other processes change it,
     and a cue whose loudness depends on it is not reproducible. All level
     control is digital gain on our own buffer.

The AudioSink seam exists so the rendered waveform can be asserted on
numerically in tests, without hardware and without anyone hearing anything.
"""

from __future__ import annotations

import logging
import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

log = logging.getLogger("morpheus.audio")

# Absolute floor on the ramp. Even if a caller asks for zero, a cue is given
# this much fade-in. Measured in milliseconds.
MIN_RAMP_MS = 250.0


class AudioError(RuntimeError):
    """Raised when audio cannot be rendered or played as specified."""


class AudioSink(Protocol):
    """Somewhere rendered audio goes."""

    def play(self, samples: np.ndarray, samplerate: int, blocking: bool = True) -> None: ...

    def is_available(self) -> bool: ...

    def describe(self) -> str: ...


@dataclass
class RenderedCue:
    """A cue buffer plus the exact parameters that produced it.

    Recorded before playback begins, so that a crash mid-cue still leaves an
    attributable record of what was played (design.md §12.3).
    """

    samples: np.ndarray
    samplerate: int
    gain: float
    ramp_ms: float
    duration_ms: float
    peak: float
    asset_id: Optional[int] = None
    asset_sha256: Optional[str] = None


class BufferSink:
    """Captures audio instead of playing it. Used by tests and dry runs."""

    def __init__(self) -> None:
        self.played: list[tuple[np.ndarray, int]] = []

    def play(self, samples: np.ndarray, samplerate: int, blocking: bool = True) -> None:
        self.played.append((np.array(samples, copy=True), samplerate))

    def is_available(self) -> bool:
        return True

    def describe(self) -> str:
        return "buffer (no audible output)"

    @property
    def last(self) -> Optional[np.ndarray]:
        return self.played[-1][0] if self.played else None


class SoundDeviceSink:
    """Real output via PortAudio.

    Imported lazily so the rest of Morpheus — and the whole test suite — works
    on a machine with no audio stack at all.
    """

    def __init__(self, device: Optional[int | str] = None) -> None:
        self._device = device
        self._sd = None

    def _backend(self):
        if self._sd is None:
            try:
                import sounddevice as sd
            except (ImportError, OSError) as exc:  # OSError: PortAudio missing
                raise AudioError(f"audio backend unavailable: {exc}") from exc
            self._sd = sd
        return self._sd

    def is_available(self) -> bool:
        try:
            sd = self._backend()
            sd.query_devices(self._device, kind="output")
        except Exception:
            return False
        return True

    def describe(self) -> str:
        try:
            info = self._backend().query_devices(self._device, kind="output")
        except Exception as exc:
            return f"unavailable ({exc})"
        return f"{info['name']} @ {info['default_samplerate']:.0f} Hz"

    def play(self, samples: np.ndarray, samplerate: int, blocking: bool = True) -> None:
        sd = self._backend()
        # Re-check the device immediately before playing. A Bluetooth speaker
        # that disappeared during the night must fail the cue, not raise
        # somewhere deep in the callback thread.
        try:
            sd.query_devices(self._device, kind="output")
        except Exception as exc:
            raise AudioError(f"output device unavailable at play time: {exc}") from exc
        sd.play(samples, samplerate=samplerate, device=self._device, blocking=blocking)


class CuePlayer:
    """Renders and plays cues under a hard amplitude ceiling.

    The ceiling is set once at construction from calibrated configuration and
    is not settable afterwards. Adaptive policy adjusts `gain`; it has no route
    to `ceiling`, which is the point (design.md §12.5).
    """

    def __init__(
        self,
        sink: AudioSink,
        *,
        ceiling: float,
        samplerate: int = 44100,
        min_ramp_ms: float = MIN_RAMP_MS,
    ) -> None:
        if not 0.0 < ceiling <= 1.0:
            raise ValueError(f"ceiling must be in (0, 1], got {ceiling}")
        self._sink = sink
        self._ceiling = float(ceiling)
        self._samplerate = int(samplerate)
        self._min_ramp_ms = float(min_ramp_ms)

    @property
    def ceiling(self) -> float:
        return self._ceiling

    @property
    def samplerate(self) -> int:
        return self._samplerate

    def render(
        self,
        waveform: np.ndarray,
        *,
        gain: float,
        ramp_ms: float,
        duration_ms: Optional[float] = None,
    ) -> RenderedCue:
        """Apply gain, ramp, and the ceiling. Never raises on a loud request.

        A caller asking for gain above the ceiling gets the ceiling and a
        warning, rather than an exception. This is deliberate: at 04:00 the
        correct response to an out-of-range request is a quiet cue, not an
        unhandled error that takes the daemon down mid-night.
        """
        if waveform.ndim != 1:
            raise ValueError("waveform must be mono 1-D")
        if waveform.size == 0:
            raise ValueError("waveform is empty")

        requested = float(gain)
        effective = min(max(requested, 0.0), self._ceiling)
        if requested > self._ceiling:
            log.warning(
                "requested gain %.3f exceeds ceiling %.3f; clamping",
                requested, self._ceiling,
            )

        samples = np.asarray(waveform, dtype=np.float32).copy()

        if duration_ms is not None:
            wanted = int(self._samplerate * duration_ms / 1000.0)
            if wanted <= 0:
                raise ValueError("duration_ms must be positive")
            if wanted < samples.size:
                samples = samples[:wanted]
            elif wanted > samples.size:
                reps = math.ceil(wanted / samples.size)
                samples = np.tile(samples, reps)[:wanted]

        # Normalise the source so that `gain` means the same loudness across
        # different cue assets. Without this, "gain 0.3" would be quiet for one
        # sound and startling for another, and the adaptive layer would be
        # learning per-asset quirks rather than the user's response.
        peak = float(np.max(np.abs(samples)))
        if peak > 1e-9:
            samples /= peak

        samples *= effective
        samples = self._apply_envelope(samples, ramp_ms)
        # Belt and braces: clip after everything, so no combination of inputs
        # can produce a sample above the ceiling.
        np.clip(samples, -self._ceiling, self._ceiling, out=samples)

        return RenderedCue(
            samples=samples,
            samplerate=self._samplerate,
            gain=effective,
            ramp_ms=max(ramp_ms, self._min_ramp_ms),
            duration_ms=1000.0 * samples.size / self._samplerate,
            peak=float(np.max(np.abs(samples))) if samples.size else 0.0,
        )

    def _apply_envelope(self, samples: np.ndarray, ramp_ms: float) -> np.ndarray:
        """Cosine fade in and out. The fade-in is not optional."""
        ramp_ms = max(float(ramp_ms), self._min_ramp_ms)
        n = samples.size
        ramp = int(self._samplerate * ramp_ms / 1000.0)
        # A very short cue cannot have a full ramp at both ends; give each half
        # the buffer rather than silently skipping the fade-in.
        ramp = max(1, min(ramp, n // 2))

        envelope = np.ones(n, dtype=np.float32)
        rise = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ramp, dtype=np.float32)))
        envelope[:ramp] = rise
        envelope[n - ramp:] = rise[::-1]
        return samples * envelope

    def play(self, rendered: RenderedCue, blocking: bool = True) -> None:
        if not self._sink.is_available():
            raise AudioError(f"audio sink unavailable: {self._sink.describe()}")
        self._sink.play(rendered.samples, rendered.samplerate, blocking=blocking)


# ------------------------------------------------------------------ WAV I/O


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV into mono float32 in [-1, 1].

    Uses the standard library rather than soundfile: one less dependency, and
    one less native library to fail at 04:00.
    """
    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        samplerate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise AudioError(f"unsupported WAV sample width: {width * 8} bit ({path})")

    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:  # 8-bit PCM is unsigned, centred on 128
        data = (data - 128.0) / 128.0
    else:
        data /= float(2 ** (8 * width - 1))

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, samplerate


def write_wav(path: Path, samples: np.ndarray, samplerate: int) -> None:
    """Write mono float32 in [-1, 1] as 16-bit PCM."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(samplerate))
        handle.writeframes(pcm.tobytes())
