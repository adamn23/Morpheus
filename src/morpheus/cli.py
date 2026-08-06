"""Morpheus command line.

M0 surface: `doctor`, `setup-models`, `record`, `report`, `sessions`.
No cueing commands exist yet, deliberately — audio arrives in M2, after the
safety supervisor and the property-based tests that constrain it.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import typer

from .analysis.coverage import analyse_session, format_report, list_sessions
from .capture.replay import FileReplaySource
from .capture.source import FrameSourceError
from .capture.webcam import WebcamSource
from .config import (
    YUNET_SHA256,
    YUNET_URL,
    MorpheusConfig,
    git_is_dirty,
    git_sha,
)
from .runtime.power import SleepPreventer
from .runtime.recorder import Recorder
from .store.db import connect, open_db
from .store.feature_store import FeatureStore

app = typer.Typer(
    add_completion=False,
    help="Morpheus — local-first N-of-1 research platform for lucid-dream cueing.",
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # OpenCV 5 emits a backend warning per detector construction; it is benign
    # and would otherwise interleave with eight hours of progress logs.
    try:
        import cv2

        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except (ImportError, AttributeError):
        pass


def _load_config(path: Optional[Path]) -> MorpheusConfig:
    return MorpheusConfig.load(path)


# ---------------------------------------------------------------- setup-models


@app.command("setup-models")
def setup_models(
    force: bool = typer.Option(False, help="Re-download even if the file exists."),
) -> None:
    """Fetch the YuNet face-detection model.

    Weights are not vendored into the repository: they are third-party
    artefacts with their own licence, and pinning by hash makes the provenance
    explicit and verifiable.
    """
    _setup_logging(False)
    config = MorpheusConfig()
    dest = Path(config.presence.model_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest == YUNET_SHA256:
            typer.echo(f"model already present and verified: {dest}")
            raise typer.Exit(0)
        typer.echo(f"model present but hash mismatch ({digest[:12]}...); re-downloading")

    typer.echo(f"downloading {YUNET_URL}")
    try:
        with urllib.request.urlopen(YUNET_URL, timeout=60) as response:
            payload = response.read()
    except OSError as exc:
        typer.secho(f"download failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    digest = hashlib.sha256(payload).hexdigest()
    if digest != YUNET_SHA256:
        typer.secho(
            f"hash mismatch: expected {YUNET_SHA256[:12]}..., got {digest[:12]}...\n"
            "Refusing to install. Verify the source before proceeding.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    dest.write_bytes(payload)
    typer.secho(f"installed {dest} ({len(payload):,} bytes, sha256 verified)", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------- doctor


@app.command()
def doctor(
    device: Optional[str] = typer.Option(None, help="Camera device index or path."),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Config JSON."),
    allow_auto_exposure: bool = typer.Option(
        False, help="Permit auto-exposure. Daylight development only."
    ),
    probe_seconds: float = typer.Option(3.0, help="Duration of the exposure stability probe."),
    write_config: Optional[Path] = typer.Option(
        None,
        "--write-config",
        help="Write a config file with the focus floor calibrated to this camera.",
    ),
) -> None:
    """Check the rig before trusting it with a night.

    Exposure gets two separate checks because the cheap one is unreliable:
    OpenCV's property read-back can report success on a backend that changed
    nothing. The stability probe watches mean luminance on what should be a
    static scene, and a drifting mean is auto-exposure hunting regardless of
    what the property says.
    """
    _setup_logging(False)
    config = _load_config(config_path)
    if device is not None:
        config.camera.device = int(device) if device.isdigit() else device
    if allow_auto_exposure:
        config.camera.require_manual_exposure = False

    ok = True
    typer.echo("Morpheus doctor")
    typer.echo("=" * 68)

    typer.echo(f"  git sha            {git_sha() or 'not a repository'}")
    dirty = git_is_dirty()
    typer.echo(f"  working tree       {'DIRTY' if dirty else 'clean' if dirty is not None else 'unknown'}")
    typer.echo(f"  config fingerprint {config.fingerprint()[:16]}")

    model = Path(config.presence.model_path)
    if model.exists():
        typer.echo(f"  face model         present ({model})")
    else:
        typer.secho(
            f"  face model         MISSING — run `morpheus setup-models`\n"
            f"                     (recording still works, motion features only)",
            fg=typer.colors.YELLOW,
        )

    with SleepPreventer(enabled=config.recorder.prevent_system_sleep) as sleeper:
        typer.echo(f"  sleep assertion    {sleeper.status}")
        if config.recorder.prevent_system_sleep and not sleeper.active:
            typer.secho("                     the machine may suspend mid-run", fg=typer.colors.YELLOW)
            ok = False

    source = WebcamSource(config.camera)
    try:
        source.open()
    except FrameSourceError as exc:
        typer.secho(f"  camera             FAILED\n                     {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    profile = source.device_profile()
    typer.echo(
        f"  camera             {profile['width']}x{profile['height']} @ {profile['fps']:.0f} fps "
        f"via {profile['backend']}"
    )

    status = source.exposure_status
    if status.manual_confirmed:
        typer.secho(f"  manual exposure    confirmed ({status.detail})", fg=typer.colors.GREEN)
    else:
        colour = typer.colors.YELLOW if allow_auto_exposure else typer.colors.RED
        typer.secho(f"  manual exposure    NOT confirmed ({status.detail})", fg=colour)
        ok = ok and allow_auto_exposure

    typer.echo(f"  exposure probe     sampling {probe_seconds:.0f}s — keep the scene still")
    probe = source.probe_exposure_stability(probe_seconds)
    cv = probe.get("cv", float("nan"))
    typer.echo(
        f"                     {probe.get('samples', 0):.0f} frames, "
        f"mean luminance {probe.get('mean_luminance', float('nan')):.1f}, "
        f"CV {cv:.4f}, range {probe.get('range', float('nan')):.1f}"
    )
    if cv == cv and cv > 0.02:  # NaN-safe
        typer.secho(
            "                     luminance is drifting on a static scene — this is the\n"
            "                     signature of auto-exposure hunting, and it will appear\n"
            "                     as motion in every downstream feature.",
            fg=typer.colors.RED,
        )
        ok = False

    ok = _check_quality_calibration(source, config) and ok
    if write_config is not None:
        _write_calibrated_config(source, config, write_config)
    source.close()

    db_path = config.storage.db_path
    conn = open_db(db_path)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    typer.echo(f"  database           {db_path} (schema v{version})")

    typer.echo("=" * 68)
    if ok:
        typer.secho("  ready to record", fg=typer.colors.GREEN)
    else:
        typer.secho("  issues above should be resolved before an overnight run", fg=typer.colors.YELLOW)
    raise typer.Exit(0 if ok else 2)


# ---------------------------------------------------------------------- record


@app.command()
def record(
    hours: float = typer.Option(8.0, help="Maximum run length."),
    device: Optional[str] = typer.Option(None, help="Camera device index or path."),
    replay: Optional[Path] = typer.Option(None, help="Replay a video file instead of a camera."),
    loop: bool = typer.Option(
        False,
        help="Loop the replay file. With --hours, this is the soak harness: "
        "an 8-hour run that verifies memory stays flat before a real night.",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Config JSON."),
    allow_auto_exposure: bool = typer.Option(
        False, help="Permit auto-exposure. Daylight development only."
    ),
    notes: Optional[str] = typer.Option(None, help="Free-text note stored with the session."),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Record a session. Persists derived features only — never video.

    M0 does no detection and plays no audio. It answers one question: can this
    camera, in this room, see this sleeper's eye region at all?
    """
    _setup_logging(verbose)
    config = _load_config(config_path)
    if device is not None:
        config.camera.device = int(device) if device.isdigit() else device
    if allow_auto_exposure:
        config.camera.require_manual_exposure = False

    if replay is not None:
        source = FileReplaySource(replay, loop=loop)
    else:
        if loop:
            typer.secho("--loop only applies to --replay; ignoring", fg=typer.colors.YELLOW)
        source = WebcamSource(config.camera)

    conn = open_db(config.storage.db_path)
    store = FeatureStore(conn, batch_size=config.storage.write_batch_size)
    recorder = Recorder(config, source, store, repo=Path.cwd())
    recorder.install_signal_handlers()

    prevent = config.recorder.prevent_system_sleep and replay is None
    with SleepPreventer(enabled=prevent) as sleeper:
        if prevent and not sleeper.active:
            typer.secho(f"warning: {sleeper.status}", fg=typer.colors.YELLOW)
        typer.echo(f"recording up to {hours:.1f} h — Ctrl-C to stop cleanly")
        try:
            summary = recorder.run(max_hours=hours)
        except FrameSourceError as exc:
            typer.secho(f"could not start: {exc}", fg=typer.colors.RED)
            conn.close()
            raise typer.Exit(1)
        summary.sleep_assertion = sleeper.status

    conn.close()

    typer.echo("")
    typer.echo(f"session {summary.session_id} finished: {summary.status}")
    typer.echo(f"  duration        {summary.duration_s / 3600:.2f} h")
    typer.echo(f"  seconds stored  {summary.health.seconds_recorded:,}")
    typer.echo(f"  capture uptime  {summary.health.capture_uptime * 100:.1f}%")
    typer.echo(f"  peak RSS        {summary.peak_rss_mb:.0f} MB")
    typer.echo(f"  exposure        {summary.exposure_detail}")
    typer.echo(f"  detector        {summary.detector_status}")
    if summary.health.clock_gaps:
        typer.secho(f"  clock gaps      {len(summary.health.clock_gaps)}", fg=typer.colors.YELLOW)
    for note in summary.notes:
        typer.echo(f"  note            {note}")
    typer.echo("")
    typer.echo(f"run `morpheus report {summary.session_id}` for the coverage analysis")


# ---------------------------------------------------------------------- report


@app.command()
def report(
    session_id: Optional[int] = typer.Argument(None, help="Session id; defaults to the latest."),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Config JSON."),
) -> None:
    """Print the coverage report and the M0 decision-gate verdict."""
    _setup_logging(False)
    config = _load_config(config_path)
    if not config.storage.db_path.exists():
        typer.secho(f"no database at {config.storage.db_path}", fg=typer.colors.RED)
        raise typer.Exit(1)

    conn = connect(config.storage.db_path, read_only=True)
    if session_id is None:
        row = conn.execute("SELECT MAX(id) FROM sessions").fetchone()
        if row is None or row[0] is None:
            typer.secho("no sessions recorded yet", fg=typer.colors.RED)
            raise typer.Exit(1)
        session_id = int(row[0])

    try:
        result = analyse_session(conn, session_id)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)
    finally:
        conn.close()

    typer.echo(format_report(result))


@app.command()
def sessions(
    limit: int = typer.Option(20, help="How many sessions to list."),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Config JSON."),
) -> None:
    """List recorded sessions."""
    _setup_logging(False)
    config = _load_config(config_path)
    if not config.storage.db_path.exists():
        typer.secho(f"no database at {config.storage.db_path}", fg=typer.colors.RED)
        raise typer.Exit(1)

    conn = connect(config.storage.db_path, read_only=True)
    rows = list_sessions(conn, limit)
    conn.close()

    if not rows:
        typer.echo("no sessions recorded yet")
        return

    typer.echo(f"{'id':>4}  {'night':>5}  {'started':<26} {'status':<18} {'secs':>7} {'face':>6} {'eye':>6}")
    typer.echo("-" * 84)
    for r in rows:
        face = f"{(r['face_frac'] or 0) * 100:5.1f}%"
        eye = f"{(r['eye_frac'] or 0) * 100:5.1f}%"
        typer.echo(
            f"{r['id']:>4}  {r['night_index'] or 0:>5}  {r['started_at_utc']:<26} "
            f"{r['status']:<18} {r['seconds'] or 0:>7,} {face:>6} {eye:>6}"
        )


def _check_quality_calibration(source: WebcamSource, config: MorpheusConfig) -> bool:
    """Report the live quality distribution against the configured thresholds.

    This exists because of a failure mode that would be near-invisible in the
    coverage report. The quality gate also gates face detection: frames below
    the floor are skipped before the detector runs. If `min_focus` is tuned for
    daylight and the real IR footage is dimmer and flatter, every frame is
    rejected, no face is ever detected, and the night reports as zero coverage —
    which looks exactly like the finding M0 exists to make, while actually being
    a misconfiguration. Measuring the distribution before the first overnight
    run is what keeps those two apart.
    """
    from .vision.quality import QualityAssessor

    assessor = QualityAssessor(config.quality)
    scores: list[float] = []
    focuses: list[float] = []
    for _ in range(60):
        frame = source.read()
        if frame is None:
            continue
        metrics = assessor.assess(frame.image)
        scores.append(metrics.score)
        focuses.append(metrics.focus)

    if not scores:
        typer.secho("  quality probe      no frames captured", fg=typer.colors.RED)
        return False

    scores.sort()
    focuses.sort()
    median = scores[len(scores) // 2]
    median_focus = focuses[len(focuses) // 2]
    passing = sum(1 for s in scores if s >= config.quality.min_score) / len(scores)

    typer.echo(
        f"  quality probe      median score {median:.3f}, "
        f"median focus {median_focus:.1f} "
        f"(floor {config.quality.min_focus:.1f})"
    )
    typer.echo(f"                     {passing * 100:.0f}% of frames clear the quality gate")

    if passing >= 0.5:
        return True

    # Recommend a floor at a quarter of the observed median: low enough to pass
    # this scene comfortably and a dimmer one later, high enough to still reject
    # a blank or black frame, which reads near zero.
    suggested = max(0.5, round(median_focus / 4.0, 1))
    typer.secho(
        "                     Most frames would be discarded before face detection\n"
        "                     even runs, and the night would report as zero coverage\n"
        "                     for the wrong reason rather than as a real finding.",
        fg=typer.colors.RED,
    )
    typer.echo(
        f"                     Suggested quality.min_focus for this camera: {suggested}\n"
        f"                     Apply it with:  morpheus doctor --write-config morpheus.json\n"
        f"                     then pass --config morpheus.json to record and report.\n"
        f"                     Change it now, deliberately — not after seeing a\n"
        f"                     disappointing coverage number."
    )
    return False


def _write_calibrated_config(
    source: WebcamSource, config: MorpheusConfig, path: Path
) -> None:
    """Persist a config with the focus floor calibrated to this camera.

    Absolute Laplacian variance does not port across cameras or lighting, so a
    population default is always a guess. This records a measured one.

    It is still only an interim measure: the probe runs on a lit scene with the
    user awake and present, and an IR bedroom at 03:00 is a very different
    image. The sleep-baseline calibration in M1 is what sets these thresholds
    from overnight quantiles (design.md §13.2), which is the only version of
    this that is actually principled.
    """
    from .vision.quality import QualityAssessor

    assessor = QualityAssessor(config.quality)
    focuses = []
    for _ in range(90):
        frame = source.read()
        if frame is not None:
            focuses.append(assessor.assess(frame.image).focus)

    if not focuses:
        typer.secho(f"  config             no frames captured; not written", fg=typer.colors.RED)
        return

    focuses.sort()
    median_focus = focuses[len(focuses) // 2]
    config.quality.min_focus = max(0.5, round(median_focus / 4.0, 1))

    path.write_text(config.model_dump_json(indent=2))
    typer.secho(
        f"  config             wrote {path} with quality.min_focus="
        f"{config.quality.min_focus} (median focus {median_focus:.1f})",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"                     use it with:  --config {path}")


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()
