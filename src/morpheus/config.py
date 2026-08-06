"""Typed configuration, and the snapshotting that makes analysis honest.

Every persisted artifact references the `config_snapshots` row that produced it
(design.md §16). Without that, a threshold tweaked in week three silently
reinterprets the first two weeks of data, and nobody notices until the analysis
disagrees with itself.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

DEFAULT_DATA_DIR = Path("data")
DEFAULT_MODEL_DIR = Path("models")
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


class CameraConfig(BaseModel):
    """Capture settings.

    `require_manual_exposure` defaults to True and is not a formality. macOS
    auto-exposure hunts continuously in a dark room, producing brightness
    oscillations that are indistinguishable from motion in every downstream
    feature. A run with AE active is not degraded data; it is fictional data.
    Set it False only for daylight development against the built-in webcam.
    """

    device: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: Optional[str] = "MJPG"
    require_manual_exposure: bool = True
    exposure: Optional[float] = None  # backend-specific; None = leave as-is
    gain: Optional[float] = None
    autofocus: Optional[bool] = False
    warmup_frames: int = 15  # discard; sensors take time to settle
    # macOS raises its camera-permission dialog asynchronously and fails the
    # open immediately, so the first run of a new binary always fails without
    # a retry window. Also covers a USB camera still enumerating.
    open_retry_attempts: int = 6
    open_retry_delay_s: float = 2.0
    read_timeout_s: float = 2.0
    reconnect_backoff_s: float = 2.0
    max_reconnect_attempts: int = 30


class QualityConfig(BaseModel):
    """Thresholds for deciding an image is worth analysing."""

    min_luminance: float = 8.0  # below this the frame is effectively black
    max_luminance: float = 245.0
    max_saturated_fraction: float = 0.15
    max_underexposed_fraction: float = 0.85
    min_focus: float = 12.0  # variance of Laplacian
    scene_change_threshold: float = 0.35  # above => treat as camera moved
    min_score: float = 0.35  # composite floor for "usable"


class PresenceConfig(BaseModel):
    """Face detection. Runs at `detect_hz`, not per frame (design.md §10)."""

    model_path: Path = DEFAULT_MODEL_DIR / YUNET_FILENAME
    score_threshold: float = 0.6
    nms_threshold: float = 0.3
    top_k: int = 500
    detect_hz: float = 5.0
    input_scale: float = 0.5  # detect on a downscaled copy for speed


class MotionConfig(BaseModel):
    """Gross motion energy settings.

    `bed_region` is a normalised (x, y, w, h) rectangle. Left None it covers the
    whole frame, which is the safe default but includes anything else that moves
    in shot.
    """

    downscale: int = 4  # process motion at 1/N resolution; ample for gross motion
    blur_ksize: int = 5
    bed_region: Optional[tuple[float, float, float, float]] = None
    scene_change_downscale: int = 8


class CoverageConfig(BaseModel):
    """Definition of "eye region usable" — the M0 decision gate.

    These numbers determine the single output that decides the project's shape,
    so they are declared up front rather than tuned after seeing the answer.

    `min_interocular_px` is the load-bearing one. Lid-surface deformation from a
    moving cornea is a sub-millimetre geometric signal; below roughly 30 px
    between the eyes there is simply not enough spatial resolution for it to
    survive, regardless of how good the detector is.
    """

    min_interocular_px: float = 30.0
    max_abs_yaw_proxy: float = 0.35
    frame_edge_margin_px: int = 8
    min_quality_score: float = 0.35
    min_detector_confidence: float = 0.6


class StorageConfig(BaseModel):
    data_dir: Path = DEFAULT_DATA_DIR
    db_filename: str = "morpheus.db"
    write_batch_size: int = 30  # FeatureFrames per transaction (~30 s)

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename


class RecorderConfig(BaseModel):
    """Overnight run behaviour."""

    max_hours: float = 9.0
    prevent_system_sleep: bool = True
    log_interval_s: float = 300.0
    # M0 records features only. This flag exists so that the *absence* of video
    # writing is an explicit, auditable decision rather than an oversight.
    persist_video: bool = False


class MorpheusConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    presence: PresenceConfig = Field(default_factory=PresenceConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    recorder: RecorderConfig = Field(default_factory=RecorderConfig)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "MorpheusConfig":
        if path is None or not Path(path).exists():
            return cls()
        return cls.model_validate_json(Path(path).read_text())

    def canonical_json(self) -> str:
        """Stable serialisation, so the hash changes only when values change."""
        return json.dumps(
            json.loads(self.model_dump_json()), sort_keys=True, separators=(",", ":")
        )

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def git_sha(repo: Optional[Path] = None) -> Optional[str]:
    """Current commit, or None outside a repo. Recorded with every snapshot."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo) if repo else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_is_dirty(repo: Optional[Path] = None) -> Optional[bool]:
    """Whether the working tree has uncommitted changes.

    M0 only records this. From M2 the daemon refuses to arm a cueing night on a
    dirty tree (design.md §21), because a night whose code cannot be recovered
    is a night that cannot be analysed.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo) if repo else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip()) if out.returncode == 0 else None
