"""Shared fixtures: synthetic imagery, scripted detectors, temp databases."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

from morpheus.config import MorpheusConfig
from morpheus.store.db import open_db
from morpheus.store.feature_store import FeatureStore
from morpheus.types import PresenceObservation


@pytest.fixture
def config(tmp_path: Path) -> MorpheusConfig:
    cfg = MorpheusConfig()
    cfg.storage.data_dir = tmp_path
    cfg.camera.require_manual_exposure = False
    return cfg


@pytest.fixture
def conn(config: MorpheusConfig):
    connection = open_db(config.storage.db_path)
    yield connection
    connection.close()


@pytest.fixture
def store(conn, config: MorpheusConfig) -> FeatureStore:
    return FeatureStore(conn, batch_size=config.storage.write_batch_size)


class ScriptedDetector:
    """A PresenceDetector stand-in returning pre-programmed observations.

    YuNet does not fire on synthetic imagery, so integration tests cannot use
    the real detector. This substitutes for it at the same seam the production
    code uses, which keeps the tests honest about what they do and do not
    cover: everything downstream of detection is exercised; detection itself
    must be validated against live footage during M0 setup.
    """

    def __init__(self, observations: Optional[list[PresenceObservation]] = None) -> None:
        self._observations = observations or []
        self._index = 0
        self.available = True
        self.status = "scripted"
        self.calls = 0

    def detect(self, image: np.ndarray) -> PresenceObservation:
        self.calls += 1
        if not self._observations:
            return PresenceObservation(face_present=False)
        obs = self._observations[min(self._index, len(self._observations) - 1)]
        self._index += 1
        return obs


def visible_face(
    *,
    interocular: float = 60.0,
    yaw: float = 0.0,
    confidence: float = 0.9,
    centre: tuple[float, float] = (640.0, 360.0),
) -> PresenceObservation:
    """A PresenceObservation describing a clearly visible, frontal face."""
    cx, cy = centre
    half = interocular / 2.0
    return PresenceObservation(
        face_present=True,
        confidence=confidence,
        bbox=(int(cx - interocular), int(cy - interocular), int(interocular * 2), int(interocular * 2.4)),
        right_eye=(cx - half, cy),
        left_eye=(cx + half, cy),
        interocular_px=interocular,
        yaw_proxy=yaw,
        roll_deg=0.0,
        detector_available=True,
    )


def face_for_frame(width: int = 320, height: int = 240) -> PresenceObservation:
    """A visible face scaled and centred to fit the given frame.

    `visible_face()` defaults to a 1280x720 layout; using it against a small
    test frame puts the eye landmarks outside the image and trips the
    EYE_OUT_OF_FRAME check, which looks like a coverage bug but is the gate
    working correctly.
    """
    interocular = max(32.0, min(width, height) / 5.0)
    return visible_face(interocular=interocular, centre=(width / 2.0, height / 2.0))


@pytest.fixture
def scripted_detector():
    return ScriptedDetector


@pytest.fixture
def make_video(tmp_path: Path):
    """Write a synthetic video file and return its path.

    Used to exercise FileReplaySource, which is the seam that makes every
    later detector change re-testable against recorded nights.
    """

    def _make(
        name: str = "clip.mp4",
        frames: int = 90,
        size: tuple[int, int] = (320, 240),
        fps: float = 30.0,
        motion: bool = False,
        luminance: int = 40,
    ) -> Path:
        path = tmp_path / name
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )
        assert writer.isOpened(), "could not open VideoWriter"
        rng = np.random.default_rng(1234)
        for i in range(frames):
            img = np.full((size[1], size[0], 3), luminance, np.uint8)
            img += rng.integers(0, 6, img.shape, dtype=np.uint8)
            if motion:
                x = int((i / max(1, frames - 1)) * (size[0] - 40))
                cv2.rectangle(img, (x, 100), (x + 40, 140), (200, 200, 200), -1)
            writer.write(img)
        writer.release()
        return path

    return _make


def gray_frame(
    width: int = 320,
    height: int = 240,
    value: int = 40,
    noise: int = 0,
    seed: int = 0,
    textured: bool = True,
) -> np.ndarray:
    """A dim frame standing in for an IR view of a bedroom.

    `textured` is on by default and matters more than it looks: a perfectly
    flat field has no high-frequency content, so the focus metric correctly
    scores it as unusable and the quality gate rejects the frame before
    anything else runs. Real scenes have edges — bedding, a headboard, a face —
    so tests that want to exercise the downstream pipeline need some structure
    here, or they end up asserting against `QUALITY_TOO_LOW` by accident.
    """
    img = np.full((height, width, 3), value, np.uint8)
    if textured:
        step = max(8, width // 10)
        for x in range(0, width, step):
            cv2.line(img, (x, 0), (x, height), (int(value * 1.8), int(value * 1.8), int(value * 1.8)), 1)
        for y in range(0, height, step):
            cv2.line(img, (0, y), (width, y), (int(value * 0.5),) * 3, 1)
        cv2.ellipse(
            img, (width // 2, height // 2), (width // 5, height // 4), 0, 0, 360,
            (int(value * 2.2),) * 3, -1,
        )
    if noise:
        rng = np.random.default_rng(seed)
        img = np.clip(
            img.astype(np.int16) + rng.integers(-noise, noise + 1, img.shape), 0, 255
        ).astype(np.uint8)
    return img
