"""M2 commands: cues, train, night, journal.

Kept in its own module so the M0 probe surface stays readable. Registered onto
the main Typer app in cli.py.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import typer

from .audio.assets import PRESETS, CueAssetRegistry
from .audio.player import AudioError, BufferSink, CuePlayer, SoundDeviceSink, load_wav
from .capture.webcam import WebcamSource
from .config import MorpheusConfig
from .cue.controller import CueController, ControllerConfig
from .cue.policy import ScheduledPolicy
from .cue.safety import SafetyLimits, SafetySupervisor
from .report.schema import PRIMARY_OUTCOME_DEFINITION, MorningReport, ReportStore, today_str
from .runtime.night import NightRunner
from .runtime.power import SleepPreventer
from .store.cue_store import CueStore
from .store.db import connect, open_db
from .store.feature_store import FeatureStore
from .training.protocol import StepKind, protocol_for, total_seconds

log = logging.getLogger("morpheus.cli")


def register(app: typer.Typer) -> None:
    app.command("cues")(cues)
    app.command("train")(train)
    app.command("night")(night)
    app.command("journal")(journal)
    app.command("baseline")(baseline)


def _registry(config: MorpheusConfig, conn) -> CueAssetRegistry:
    return CueAssetRegistry(conn, config.storage.data_dir / "cues")


# ------------------------------------------------------------------------ cues


def cues(
    add_preset: Optional[str] = typer.Option(None, help=f"Create a preset cue: {', '.join(PRESETS)}"),
    add_file: Optional[Path] = typer.Option(None, help="Register an existing WAV file."),
    trained: bool = typer.Option(
        True, help="Is this the conditioned cue? False registers it as an untrained control."
    ),
    name: Optional[str] = typer.Option(None, help="Name for the asset."),
    preview: Optional[int] = typer.Option(None, help="Play a registered cue by id, at safe gain."),
    gain: float = typer.Option(0.15, help="Preview gain."),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Manage cue audio.

    Every asset is hashed on registration. Whether a night used the trained cue
    or the control is then a fact in the database rather than something you have
    to remember correctly months later.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = MorpheusConfig.load(config_path)
    conn = open_db(config.storage.db_path)
    registry = _registry(config, conn)

    if add_preset:
        asset = registry.create_preset(add_preset, trained=trained, name=name)
        label = "TRAINED" if asset.trained else "control"
        typer.secho(f"registered #{asset.id} {asset.name} [{label}] {asset.sha256[:12]}...", fg=typer.colors.GREEN)
    if add_file:
        asset = registry.register(add_file, trained=trained, name=name)
        label = "TRAINED" if asset.trained else "control"
        typer.secho(f"registered #{asset.id} {asset.name} [{label}] {asset.sha256[:12]}...", fg=typer.colors.GREEN)

    if preview is not None:
        asset = registry.get(preview)
        waveform, _ = load_wav(asset.path)
        player = CuePlayer(SoundDeviceSink(), ceiling=config_ceiling(config))
        rendered = player.render(waveform, gain=gain, ramp_ms=1500, duration_ms=None)
        typer.echo(f"playing '{asset.name}' at gain {rendered.gain:.3f} (ceiling {player.ceiling})")
        try:
            player.play(rendered)
        except AudioError as exc:
            typer.secho(f"playback failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(1)

    assets = registry.list()
    conn.close()
    if not assets:
        typer.echo("no cue assets yet. Create one with:")
        typer.echo("  morpheus cues --add-preset trained-ascending --trained")
        typer.echo("  morpheus cues --add-preset control-descending --no-trained")
        return

    typer.echo(f"\n{'id':>3}  {'name':<24} {'kind':<9} {'dur':>6}  sha256")
    typer.echo("-" * 72)
    for a in assets:
        kind = "TRAINED" if a.trained else "control"
        ok = "" if registry.verify(a) else "  <-- FILE CHANGED SINCE REGISTRATION"
        typer.echo(f"{a.id:>3}  {a.name:<24} {kind:<9} {a.duration_s:>5.1f}s  {a.sha256[:12]}...{ok}")


def config_ceiling(config: MorpheusConfig) -> float:
    return SafetyLimits().max_gain


# ----------------------------------------------------------------------- train


def train(
    kind: str = typer.Option("evening", help="'evening' or 'wbtb'."),
    cue_id: Optional[int] = typer.Option(None, help="Cue asset id; defaults to the trained cue."),
    gain: float = typer.Option(0.2, help="Playback gain during training (you are awake)."),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Run the conditioning protocol.

    This is the part of Morpheus with published efficacy behind it. The cue is
    inert without it — the sound works because it has been bound to a state of
    critical self-awareness beforehand.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    config = MorpheusConfig.load(config_path)
    conn = open_db(config.storage.db_path)
    registry = _registry(config, conn)
    store = CueStore(conn)

    trained_assets = registry.list(trained=True)
    if cue_id is not None:
        asset = registry.get(cue_id)
    elif trained_assets:
        asset = trained_assets[0]
    else:
        typer.secho(
            "no trained cue registered. Create one first:\n"
            "  morpheus cues --add-preset trained-ascending --trained",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    if not registry.verify(asset):
        typer.secho(f"cue file for '{asset.name}' no longer matches its hash", fg=typer.colors.RED)
        raise typer.Exit(1)

    steps = protocol_for(kind)
    player = CuePlayer(SoundDeviceSink(), ceiling=SafetyLimits().max_gain)
    waveform, _ = load_wav(asset.path)

    typer.echo(f"\nMorpheus conditioning — {kind}")
    typer.echo(f"cue: {asset.name}   approx {total_seconds(kind) // 60} min   Ctrl-C to abort")
    typer.echo("=" * 68)

    training_id = store.start_training(kind, cue_asset_id=asset.id)
    started = time.monotonic()
    completed_steps: dict[str, bool] = {}
    captured: dict[str, str] = {}
    finished = True

    try:
        for index, step in enumerate(steps, start=1):
            typer.echo("")
            typer.secho(f"[{index}/{len(steps)}] {step.title}", fg=typer.colors.CYAN, bold=True)
            for line in _wrap(step.body, 66):
                typer.echo(f"  {line}")

            if step.plays_cue:
                try:
                    rendered = player.render(waveform, gain=gain, ramp_ms=800, duration_ms=None)
                    player.play(rendered)
                except AudioError as exc:
                    typer.secho(f"  (cue playback failed: {exc})", fg=typer.colors.YELLOW)

            if step.kind is StepKind.INPUT and step.capture:
                captured[step.capture] = typer.prompt("  >", default="", show_default=False)
            else:
                typer.echo(f"  ({step.seconds}s — press Enter when done)")
                input()
            completed_steps[step.key] = True
    except KeyboardInterrupt:
        finished = False
        typer.echo("\n\naborted")

    duration = time.monotonic() - started
    rating = None
    if finished:
        typer.echo("")
        rating = typer.prompt("How engaged were you, 1-5?", type=int, default=3)

    store.finish_training(
        training_id,
        completed=finished,
        duration_s=duration,
        steps={**completed_steps, **{f"captured_{k}": v for k, v in captured.items()}},
        engagement_rating=rating,
    )
    conn.close()

    if finished:
        typer.secho(f"\nconditioning recorded ({duration / 60:.1f} min).", fg=typer.colors.GREEN)
        typer.echo("Hold the intention as you fall asleep. Repeat it, meaning it each time.")


# ----------------------------------------------------------------------- night


def night(
    hours: float = typer.Option(8.0, help="Maximum run length."),
    cue_id: Optional[int] = typer.Option(None, help="Cue asset id; defaults to the trained cue."),
    camera: bool = typer.Option(
        False, help="Use the camera for motion gating and arousal detection."
    ),
    dry_run: bool = typer.Option(
        False, help="Decide and log cues but play no audio. Use this for the first night."
    ),
    delay_hours: Optional[float] = typer.Option(
        None, help="Override the minimum delay before the first cue."
    ),
    max_cues: Optional[int] = typer.Option(None, help="Override the nightly cue cap."),
    max_gain: Optional[float] = typer.Option(None, help="Override the hard volume ceiling."),
    stop_before_wake_min: Optional[float] = typer.Option(
        None, help="Minutes before the end of the run after which no cue may fire."
    ),
    cooldown_min: Optional[float] = typer.Option(None, help="Override the minimum cooldown."),
    allow_auto_exposure: bool = typer.Option(False, help="Daylight development only."),
    config_path: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Run a cueing night.

    Defaults are conservative: six-hour delay, six cues maximum, twenty-minute
    cooldown, and a hard volume ceiling no policy can exceed. Run --dry-run
    first to see when cues *would* fire without any sound being made.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    config = MorpheusConfig.load(config_path)
    if allow_auto_exposure:
        config.camera.require_manual_exposure = False

    conn = open_db(config.storage.db_path)
    registry = _registry(config, conn)
    trained_assets = registry.list(trained=True)
    if cue_id is not None:
        asset = registry.get(cue_id)
    elif trained_assets:
        asset = trained_assets[0]
    else:
        typer.secho("no trained cue registered; run `morpheus cues --add-preset trained-ascending --trained`", fg=typer.colors.RED)
        raise typer.Exit(1)

    recent_training = CueStore(conn).latest_training()
    if recent_training is None:
        typer.secho(
            "No completed conditioning session on record.\n"
            "An unconditioned cue is just a noise in the night — the published effect "
            "comes from the pairing. Run `morpheus train` first.",
            fg=typer.colors.YELLOW,
        )
        if not typer.confirm("Continue anyway?", default=False):
            raise typer.Exit(1)

    limits = SafetyLimits()
    if delay_hours is not None:
        limits.min_delay_s = delay_hours * 3600
    if max_cues is not None:
        limits.max_cues_per_night = max_cues
    if max_gain is not None:
        limits.max_gain = max_gain
    if stop_before_wake_min is not None:
        limits.stop_before_wake_s = stop_before_wake_min * 60
    if cooldown_min is not None:
        limits.min_cooldown_s = cooldown_min * 60
    limits.__post_init__()  # re-validate after overrides

    supervisor = SafetySupervisor(limits=limits)
    controller = CueController(supervisor, policy=ScheduledPolicy(), config=ControllerConfig())
    sink = BufferSink() if dry_run else SoundDeviceSink()
    player = CuePlayer(sink, ceiling=limits.max_gain)

    source = None
    if camera:
        source = WebcamSource(config.camera)

    feature_store = FeatureStore(conn, batch_size=config.storage.write_batch_size)
    session_id = feature_store.start_session(
        config=config,
        device_profile=source.device_profile() if source else {"camera_model": "none"},
        kind="cue_night",
        notes=f"cue={asset.name} dry_run={dry_run} camera={camera}",
        repo=Path.cwd(),
    )

    runner = NightRunner(
        config, controller=controller, player=player, asset=asset, registry=registry,
        feature_store=feature_store, cue_store=CueStore(conn), source=source, dry_run=dry_run,
    )

    typer.echo("")
    typer.echo(f"  cue            {asset.name} ({'TRAINED' if asset.trained else 'CONTROL'})")
    typer.echo(f"  mode           {'DRY RUN — no audio' if dry_run else 'AUDIO ARMED'}")
    typer.echo(f"  sensing        {'camera' if camera else 'clock only'}")
    typer.echo(f"  first cue      after {limits.min_delay_s / 3600:.1f} h")
    typer.echo(f"  caps           {limits.max_cues_per_night}/night, {limits.max_cues_per_hour}/hour, "
               f"{limits.min_cooldown_s / 60:.0f} min cooldown")
    typer.echo(f"  volume ceiling {limits.max_gain:.2f} (hard, not adjustable by policy)")
    typer.echo(f"  stop           Ctrl-C, or turn the speaker off")

    # Pre-flight: can a cue fire at all tonight? The window is bounded at both
    # ends — nothing before min_delay, nothing within stop_before_wake of the
    # deadline — and it is entirely possible to configure a run where those
    # overlap. Discovering that from a silent zero-cue summary the next morning
    # wastes a night; saying so now costs nothing.
    usable_window = hours * 3600 - limits.min_delay_s - limits.stop_before_wake_s
    if usable_window <= 0:
        typer.echo("")
        typer.secho(
            f"  No cue can fire with these settings. A {hours:.1f} h night minus a "
            f"{limits.min_delay_s / 3600:.1f} h delay minus the "
            f"{limits.stop_before_wake_s / 60:.0f} min pre-wake guard leaves no window.\n"
            f"  Increase --hours, or lower --delay-hours.",
            fg=typer.colors.RED,
        )
        if not typer.confirm("  Run anyway?", default=False):
            raise typer.Exit(1)
    else:
        typer.echo(f"  cue window     {usable_window / 3600:.1f} h wide")
    typer.echo("")

    with SleepPreventer(enabled=config.recorder.prevent_system_sleep) as sleeper:
        if not sleeper.active:
            typer.secho(f"warning: {sleeper.status}", fg=typer.colors.YELLOW)
        try:
            summary = runner.run(hours=hours, session_id=session_id)
        except AudioError as exc:
            typer.secho(f"could not start: {exc}", fg=typer.colors.RED)
            conn.close()
            raise typer.Exit(1)

    feature_store.finish_session(summary.status, summary.health)
    conn.close()

    typer.echo("")
    typer.echo(f"night finished: {summary.status}  (final state: {summary.final_state})")
    typer.echo(f"  duration       {summary.duration_s / 3600:.2f} h")
    typer.echo(f"  cues played    {summary.cues_played}")
    if summary.cues_failed:
        typer.secho(f"  cues failed    {summary.cues_failed}", fg=typer.colors.YELLOW)
    for name, count in sorted(summary.outcomes.items()):
        typer.echo(f"  outcome        {name}: {count}")
    if summary.cues_played == 0 and summary.gate_blocks:
        typer.echo("")
        typer.echo("  No cues fired. Seconds blocked by each gate:")
        for gate, seconds in sorted(summary.gate_blocks.items(), key=lambda kv: -kv[1]):
            typer.echo(f"    {gate:<26} {seconds:>7,}s")
    if summary.halted_reason:
        typer.secho(f"  halted         {summary.halted_reason}", fg=typer.colors.YELLOW)
    for note in summary.notes:
        typer.echo(f"  note           {note}")
    typer.echo("")
    typer.echo("Write your dream report before you look at anything else: morpheus journal")


# --------------------------------------------------------------------- journal


def journal(
    for_date: Optional[str] = typer.Option(None, "--date", help="YYYY-MM-DD; defaults to today."),
    yesterday: bool = typer.Option(False, help="Report for yesterday's date."),
    show: bool = typer.Option(False, help="Show recent reports instead of writing one."),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Record this morning's dream report.

    Do this before checking messages or reading anything. Dream recall decays
    within minutes of waking, and this field is the primary outcome the entire
    project is built to move.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    config = MorpheusConfig.load(config_path)
    conn = open_db(config.storage.db_path)
    store = ReportStore(conn)

    if show:
        rows = store.recent(14)
        if not rows:
            typer.echo("no reports yet")
        else:
            typer.echo(f"{'date':<12} {'lucid':<6} {'conf':<5} {'dreams':<7} {'vivid':<6} {'sleep':<6}")
            typer.echo("-" * 50)
            for r in rows:
                lucid = "-" if r["lucid_binary"] is None else ("YES" if r["lucid_binary"] else "no")
                typer.echo(
                    f"{r['report_date']:<12} {lucid:<6} {_s(r['lucid_confidence']):<5} "
                    f"{_s(r['dreams_recalled']):<7} {_s(r['vividness']):<6} {_s(r['sleep_quality']):<6}"
                )
        conn.close()
        return

    target = for_date or ((date.today() - timedelta(days=1)).isoformat() if yesterday else today_str())

    typer.echo(f"\nMorning report — {target}")
    typer.echo("=" * 68)
    typer.echo("Write freely. Fragments are fine and are worth more than nothing.\n")

    typer.echo("Dream narrative (end with a single '.' on its own line):")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == ".":
            break
        lines.append(line)
    narrative = "\n".join(lines).strip() or None

    typer.echo("")
    typer.secho(f'Primary outcome: "{PRIMARY_OUTCOME_DEFINITION}"', bold=True)
    lucid = typer.confirm("  Was that true last night?", default=False)

    report = MorningReport(
        report_date=target,
        narrative=narrative,
        lucid_binary=lucid,
        dreams_recalled=typer.prompt("Dreams recalled", type=int, default=0),
        vividness=typer.prompt("Vividness 1-5", type=int, default=3),
        sleep_quality=typer.prompt("Sleep quality 1-5", type=int, default=3),
        awakenings=typer.prompt("Times you woke up", type=int, default=0),
    )
    if lucid:
        report.lucid_confidence = typer.prompt("How sure are you, 0-4", type=int, default=2)
        report.knew_was_dreaming = typer.confirm(
            "Did you know you were dreaming *during* the dream?", default=True
        )

    typer.echo("")
    report.cue_heard = typer.confirm("Did you hear the cue?", default=False)
    report.cue_indirect = typer.confirm("Did a similar sound appear in the dream?", default=False)
    report.cue_woke_me = typer.confirm("Did a sound wake you?", default=False)
    report.guessed_condition = typer.prompt(
        "Guess last night's condition (trained/control/none/unsure)", default="unsure"
    )
    report.notes = typer.prompt("Notes", default="", show_default=False) or None

    problems = report.validate()
    if problems:
        typer.secho("  " + "; ".join(problems), fg=typer.colors.RED)
        conn.close()
        raise typer.Exit(1)

    store.submit(report)
    conn.close()
    typer.secho(f"\nrecorded for {target}.", fg=typer.colors.GREEN)


def baseline(config_path: Optional[Path] = typer.Option(None, "--config")) -> None:
    """Show the lucid-dream rate over recent reports."""
    config = MorpheusConfig.load(config_path)
    if not config.storage.db_path.exists():
        typer.secho("no database yet", fg=typer.colors.RED)
        raise typer.Exit(1)
    conn = connect(config.storage.db_path, read_only=True)
    stats = ReportStore(conn).baseline_stats()
    conn.close()

    if not stats.get("nights"):
        typer.echo("no reports yet. Start tonight: morpheus journal")
        return

    typer.echo(f"\nBaseline over {stats['nights']} reports")
    typer.echo("-" * 44)
    typer.echo(f"  nights scored        {stats['nights_scored']}")
    typer.echo(f"  lucid nights         {stats['lucid_nights']}")
    if stats["lucid_rate_per_night"] is not None:
        typer.echo(f"  lucid rate           {stats['lucid_rate_per_night'] * 100:.1f}% of nights")
        typer.echo(f"                       {stats['lucid_per_week']:.2f} per week")
    if stats["mean_dreams_recalled"] is not None:
        typer.echo(f"  mean dreams recalled {stats['mean_dreams_recalled']:.2f}")
    if stats["mean_sleep_quality"] is not None:
        typer.echo(f"  mean sleep quality   {stats['mean_sleep_quality']:.2f}")

    if stats["nights"] < 14:
        typer.echo("")
        typer.echo(f"  {14 - stats['nights']} more nights for a usable pre-intervention baseline.")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def _s(value) -> str:
    return "-" if value is None else str(value)
