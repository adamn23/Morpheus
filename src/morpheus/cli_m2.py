"""M2 commands: cues, train, night, journal.

Kept in its own module so the M0 probe surface stays readable. Registered onto
the main Typer app in cli.py.
"""

from __future__ import annotations

import logging
import signal
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import typer

from .audio.assets import PRESETS, CueAsset, CueAssetRegistry
from .audio.player import AudioError, BufferSink, CuePlayer, SoundDeviceSink, load_wav
from .capture.webcam import WebcamSource
from .config import MorpheusConfig
from .cue.controller import CueController, ControllerConfig
from .cue.policy import ScheduledPolicy
from .cue.safety import SafetyLimits, SafetySupervisor
from .report.safety_monitor import check as check_sleep_quality
from .report.safety_monitor import format_check
from .report.schema import PRIMARY_OUTCOME_DEFINITION, MorningReport, ReportStore, today_str
from .runtime.night import NightRunner, WbtbPlan
from .runtime.power import SleepPreventer
from .store.cue_store import CueStore
from .store.db import connect, open_db
from .store.feature_store import FeatureStore
from .training.protocol import (
    TRAINING_KINDS,
    StepKind,
    protocol_for,
    total_seconds,
)
from .types import EventKind, HealthCounters

log = logging.getLogger("morpheus.cli")

#: Ceiling for the WBTB alarm. Far above the cue ceiling on purpose: the alarm's
#: job is the opposite of a cue's. It is still a ceiling, not a volume — the
#: requested alarm gain sits below it.
ALARM_CEILING = 0.95

#: Name of the preset registered as the wake alarm.
WBTB_ALARM_PRESET = "wbtb-alarm"


def _wbtb_alarm(registry: CueAssetRegistry) -> CueAsset:
    """Find or create the wake alarm, and refuse to hand back a trained cue.

    Registered untrained, because it is not a cue and must never be selectable
    as one: waking someone with their conditioned sound would teach them it
    means 'get up', undoing the conditioning the protocol depends on.
    """
    for existing in registry.list():
        if existing.name == WBTB_ALARM_PRESET:
            if existing.trained:
                raise typer.BadParameter(
                    f"asset '{WBTB_ALARM_PRESET}' is registered as TRAINED. "
                    f"Re-register it with --no-trained before running WBTB."
                )
            return existing
    return registry.create_preset(WBTB_ALARM_PRESET, trained=False)


def _resolve_arm(conn, config, registry, asset, today: str, session_id: Optional[int] = None):
    """Bind tonight's sealed arm to the cue that will actually play.

    Returns `(asset, plays_audio, assignment_id)`. When no experiment is running
    this is a no-op and the caller's chosen asset stands — which is Phase A, the
    unrandomised phase.

    The arm is read through `arm_for_running_night`, which audits the read as
    machine-made and does not count it against the blind. **Nothing about the
    arm may be printed.** The user learns it from `morpheus reveal`, after the
    morning report exists, and that ordering is the blinding.
    """
    from .experiment.assignments import ExperimentStore
    from .experiment.randomization import Arm

    store = ExperimentStore(conn, config.storage.data_dir)
    experiment = store.active()
    if experiment is None:
        return asset, True, None

    assignment_id = store.assign_night(experiment, today, session_id=session_id)
    arm = store.arm_for_running_night(assignment_id)

    if arm is Arm.NO_CUE:
        return asset, False, assignment_id
    if arm is Arm.UNTRAINED_CUE:
        # The matched twin: same timbre, note count, duration and register,
        # differing only in contour — and never conditioned. If the pairing is
        # missing this raises rather than substituting something unmatched.
        return registry.matched_control_for(asset), True, assignment_id
    return asset, True, assignment_id


def _last_night(conn) -> dict:
    """What the most recent cueing session did, for the morning questions.

    Used to ask about WBTB only on nights that had one, and to tell the user how
    many cues actually fired — without revealing the experimental arm, which the
    blinding depends on staying sealed until after the report.
    """
    row = conn.execute(
        "SELECT id FROM sessions WHERE kind = 'cue_night' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return {"cues": 0, "wbtb": False}
    session_id = row["id"]
    cues = conn.execute(
        "SELECT COUNT(*) FROM cues WHERE session_id = ? AND played = 1", (session_id,)
    ).fetchone()[0]
    wbtb = conn.execute(
        "SELECT COUNT(*) FROM events WHERE session_id = ? AND kind = ?",
        (session_id, EventKind.WBTB_WAKE.value),
    ).fetchone()[0]
    return {"cues": cues, "wbtb": bool(wbtb)}


def _wait_for_enter(timeout_s: float) -> bool:
    """Block for Enter, but never past `timeout_s`. True if Enter arrived.

    A bare `input()` here would be a trap: falling asleep mid-script at 05:00 is
    a completely ordinary thing to do, and it would hang the night forever with
    no cues and no error. Every wait in the induction script is bounded.
    """
    if timeout_s <= 0:
        return False
    try:
        import select
        import sys

        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if not ready:
            return False
        return sys.stdin.readline() != ""
    except (OSError, ValueError, ImportError):
        # No selectable stdin (piped, or a platform without it). Fall back to a
        # plain read; EOF is handled by the caller.
        try:
            input()
            return True
        except (EOFError, KeyboardInterrupt):
            return False


def _wbtb_script(kind: str, awake_min: float):
    """Return a callable that walks the induction script at the WBTB wake.

    Kept short by design: sleep latency after the technique is one of only two
    replicated predictors of success (Aspy 2017), so a long script works against
    the thing that most reliably helps. The user may finish early with Enter —
    going back to sleep promptly matters more than completing every step.
    """
    def run(_awake_min: float) -> None:
        steps = protocol_for(kind)
        deadline = time.monotonic() + awake_min * 60.0
        typer.echo("")
        typer.secho(
            f"  WBTB — {kind}, {total_seconds(kind) / 60:.0f} min of script, "
            f"~{awake_min:.0f} min awake at most.",
            bold=True,
        )
        typer.echo("  Stay in bed if you can. Enter to advance; it advances on its own")
        typer.echo("  if you do not, and ends by itself at the time limit.")
        typer.echo("")
        try:
            for index, step in enumerate(steps, start=1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    typer.echo("\n  time limit reached — back to sleep.")
                    break
                typer.secho(f"  [{index}/{len(steps)}] {step.title}", bold=True)
                typer.echo(f"      {step.body}")
                # Each step self-advances after its own duration, so the script
                # completes even if you drift off holding the intention — which
                # is the intended way to finish it, not a failure.
                _wait_for_enter(min(float(step.seconds), max(0.0, deadline - time.monotonic())))
        except KeyboardInterrupt:
            typer.echo("\n  script ended early — going back to sleep is the priority.")
        typer.echo("  Back to sleep. Cues resume shortly.\n")

    return run


def register(app: typer.Typer) -> None:
    app.command("cues")(cues)
    app.command("train")(train)
    app.command("night")(night)
    app.command("journal")(journal)
    app.command("baseline")(baseline)
    app.command("import-journal")(import_journal)
    app.command("dream-signs")(dream_signs)
    app.command("calibrate")(calibrate)
    app.command("confirm")(confirm_cmd)
    app.command("review")(review_cmd)


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
    kind: str = typer.Option(
        "evening",
        help="'evening', 'wbtb', or 'ssild'. SSILD and MILD were similarly "
             "effective in ILDIS (n=355) and the hybrid showed no advantage, so "
             "treat these as alternatives to compare, not a stack.",
    ),
    cue_id: Optional[int] = typer.Option(None, help="Cue asset id; defaults to the trained cue."),
    gain: float = typer.Option(0.2, help="Playback gain during training (you are awake)."),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Run the conditioning protocol.

    This is the part of Morpheus with published efficacy behind it. The cue is
    inert without it — the sound works because it has been bound to a state of
    critical self-awareness beforehand.
    """
    if kind not in TRAINING_KINDS:
        typer.secho(
            f"unknown kind {kind!r}; choose from {', '.join(sorted(TRAINING_KINDS))}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
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
    gain: Optional[float] = typer.Option(
        None, help="Cue volume, 0.02-0.35. Lower this if a cue woke you; the value "
                   "is recorded per cue, unlike the OS volume slider."
    ),
    wbtb_at: Optional[float] = typer.Option(
        None, help="Hours after start to wake for WBTB. Omit for no WBTB."
    ),
    wbtb_awake_min: float = typer.Option(
        45.0, help="Minutes awake during WBTB. 60 beat 30 in the one lab test "
                   "(Erlacher & Stumbrys 2020); 45 hedges against nightly cost."
    ),
    post_wbtb_delay_min: float = typer.Option(
        25.0, help="Minutes after returning to sleep before cues may resume."
    ),
    wbtb_kind: str = typer.Option(
        "wbtb", help="Induction script at the WBTB wake: 'wbtb' (MILD) or 'ssild'."
    ),
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

    # If an experiment is running, tonight's sealed arm decides which cue plays
    # and whether one plays at all. From here on the asset is blinded: nothing
    # downstream may print its name or its trained flag.
    asset, condition_allows_cue, assignment_id = _resolve_arm(
        conn, config, registry, asset, today_str()
    )
    blinded = assignment_id is not None

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

    # Design.md §23 stopping rule, enforced before arming rather than during
    # the night. Halting a study is a decision made awake with the numbers in
    # front of you, not something to discover at 04:00.
    quality = check_sleep_quality(conn)
    if quality.should_halt and not dry_run:
        typer.secho("\n" + format_check(quality), fg=typer.colors.RED)
        typer.secho(
            "\nCueing is halted by the pre-registered sleep-quality rule. This is not\n"
            "advisory: the pre-registration commits to it, and overriding it makes that\n"
            "document false. Take some nights off cueing and keep journalling.",
            fg=typer.colors.RED,
        )
        conn.close()
        raise typer.Exit(2)
    if quality.warning:
        typer.secho("\n" + format_check(quality), fg=typer.colors.YELLOW)

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
    # Volume belongs in the database, not in the OS volume slider. The slider is
    # global, unrecorded, and changed by anything that plays a sound, so a night
    # tuned with it is not reproducible and not comparable to any other night.
    policy = ScheduledPolicy()
    if gain is not None:
        if not limits.min_gain <= gain <= limits.max_gain:
            typer.secho(
                f"gain must be between {limits.min_gain} and {limits.max_gain}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
        # Both ends: `start_gain` is where it begins, `max_gain` stops the policy
        # stepping back above the level you just asked for.
        policy.start_gain = gain
        policy.max_gain = min(policy.max_gain, gain)
    controller = CueController(
        supervisor, policy=policy, config=ControllerConfig(),
        condition_allows_cue=condition_allows_cue,
    )
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
        # The asset name is the arm on a blinded night. It is recorded
        # against the cue rows, which are sealed reading, never in session
        # notes, which `morpheus sessions` prints.
        notes=(
            f"blinded assignment={assignment_id} dry_run={dry_run} camera={camera}"
            if blinded else
            f"cue={asset.name} dry_run={dry_run} camera={camera}"
        ),
        repo=Path.cwd(),
    )

    wbtb_plan = None
    alarm_asset = None
    alarm_player = None
    if wbtb_at is not None:
        if wbtb_kind not in TRAINING_KINDS:
            typer.secho(
                f"unknown --wbtb-kind {wbtb_kind!r}; choose from "
                f"{', '.join(sorted(TRAINING_KINDS))}", fg=typer.colors.RED,
            )
            conn.close()
            raise typer.Exit(1)
        alarm_asset = _wbtb_alarm(registry)
        # A separate player, because the cue player's ceiling is the *cue*
        # safety ceiling and an alarm bound by it would not wake anyone.
        alarm_player = CuePlayer(sink, ceiling=ALARM_CEILING)
        wbtb_plan = WbtbPlan(
            at_h=wbtb_at,
            awake_min=wbtb_awake_min,
            post_delay_min=post_wbtb_delay_min,
            prompt=_wbtb_script(wbtb_kind, wbtb_awake_min),
        )

    runner = NightRunner(
        config, controller=controller, player=player, asset=asset, registry=registry,
        feature_store=feature_store, cue_store=CueStore(conn), source=source, dry_run=dry_run,
        wbtb=wbtb_plan, alarm=alarm_asset, alarm_player=alarm_player,
    )

    typer.echo("")
    if blinded:
        typer.echo("  cue            [blinded — sealed until after tomorrow's report]")
    else:
        typer.echo(f"  cue            {asset.name} ({'TRAINED' if asset.trained else 'CONTROL'})")
    typer.echo(f"  mode           {'DRY RUN — no audio' if dry_run else 'AUDIO ARMED'}")
    typer.echo(f"  sensing        {'camera' if camera else 'clock only'}")
    typer.echo(f"  first cue      after {limits.min_delay_s / 3600:.1f} h")
    typer.echo(f"  caps           {limits.max_cues_per_night}/night, {limits.max_cues_per_hour}/hour, "
               f"{limits.min_cooldown_s / 60:.0f} min cooldown")
    typer.echo(f"  cue gain       {policy.start_gain:.3f}"
               f"{' (overridden)' if gain is not None else ''}")
    typer.echo(f"  volume ceiling {limits.max_gain:.2f} (hard, not adjustable by policy)")
    typer.echo(f"  stop           Ctrl-C, or turn the speaker off")

    # Pre-flight: can a cue fire at all tonight? The window is bounded at both
    # ends — nothing before min_delay, nothing within stop_before_wake of the
    # deadline — and it is entirely possible to configure a run where those
    # overlap. Discovering that from a silent zero-cue summary the next morning
    # wastes a night; saying so now costs nothing.
    if wbtb_plan is not None:
        # On a WBTB night the pre-WBTB delay is irrelevant to the window: no cue
        # can fire before the wake anyway, and afterwards the delay clock is
        # replaced by --post-wbtb-delay-min. Measuring against --delay-hours here
        # would report "no window" for a perfectly workable night.
        consumed = (
            wbtb_plan.at_s + wbtb_plan.awake_min * 60 + wbtb_plan.post_delay_min * 60
        )
        usable_window = hours * 3600 - consumed - limits.stop_before_wake_s
        window_advice = (
            f"  A {hours:.1f} h night minus a {wbtb_plan.at_h:.1f} h WBTB wake, "
            f"{wbtb_plan.awake_min:.0f} min awake, {wbtb_plan.post_delay_min:.0f} min "
            f"settling and the {limits.stop_before_wake_s / 60:.0f} min pre-wake guard "
            f"leaves no window.\n  Increase --hours, or wake earlier with --wbtb-at."
        )
    else:
        usable_window = hours * 3600 - limits.min_delay_s - limits.stop_before_wake_s
        window_advice = (
            f"  A {hours:.1f} h night minus a {limits.min_delay_s / 3600:.1f} h delay "
            f"minus the {limits.stop_before_wake_s / 60:.0f} min pre-wake guard leaves "
            f"no window.\n  Increase --hours, or lower --delay-hours."
        )

    if usable_window <= 0:
        typer.echo("")
        typer.secho(
            "  No cue can fire with these settings.\n" + window_advice,
            fg=typer.colors.RED,
        )
        if not typer.confirm("  Run anyway?", default=False):
            raise typer.Exit(1)
    else:
        typer.echo(f"  cue window     {usable_window / 3600:.1f} h wide")
    typer.echo("")

    # Ctrl-C is documented above as the way to stop a night, so it has to end
    # the run through the normal path. Left to the default handler it raises
    # KeyboardInterrupt — a BaseException, so the runner's `except Exception`
    # does not see it — and the session row stays at status='running' with no
    # end time, no summary, and no defect flag. That is precisely the silent
    # artefact the defect reporting exists to prevent. Ask the loop to stop
    # instead; a second signal falls through to the default and hard-exits.
    previous: dict[int, object] = {}

    def _on_signal(signum: int, _frame: object) -> None:
        typer.echo("\n  stopping cleanly — finishing the session record...")
        signal.signal(signum, previous.get(signum, signal.SIG_DFL))  # type: ignore[arg-type]
        runner.request_stop()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _on_signal)

    try:
        with SleepPreventer(enabled=config.recorder.prevent_system_sleep) as sleeper:
            if not sleeper.active:
                typer.secho(f"warning: {sleeper.status}", fg=typer.colors.YELLOW)
            try:
                summary = runner.run(hours=hours, session_id=session_id)
            except AudioError as exc:
                typer.secho(f"could not start: {exc}", fg=typer.colors.RED)
                feature_store.finish_session("aborted_audio_error", HealthCounters())
                conn.close()
                raise typer.Exit(1)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)  # type: ignore[arg-type]

    feature_store.finish_session(summary.status, summary.health)
    conn.close()

    typer.echo("")
    typer.echo(f"night finished: {summary.status}  (final state: {summary.final_state})")
    typer.echo(
        f"  duration       {summary.duration_s / 3600:.2f} h "
        f"of {summary.intended_s / 3600:.2f} h intended"
    )
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

    defects = summary.defects()
    if defects:
        typer.echo("")
        typer.secho("  DEFECTIVE NIGHT — do not analyse as a clean observation:",
                    fg=typer.colors.RED, bold=True)
        for defect in defects:
            typer.secho(f"    - {defect}", fg=typer.colors.RED)
        typer.echo("")
        typer.echo("  Still write the report; note the defect in it so the night can be")
        typer.echo("  excluded on a documented rule rather than on how the numbers look.")

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
    last = _last_night(conn)
    if last["cues"]:
        typer.echo(f"  ({last['cues']} cues were delivered last night.)")

    # Counts, not just yes/no. Gain titration needs to tell "one cue was
    # slightly loud" from "every cue woke me", and a boolean spanning six cues
    # cannot. The booleans are still recorded, derived from the counts.
    report.cues_heard_count = typer.prompt(
        "How many cues do you remember hearing?", type=int, default=0
    )
    report.cues_incorporated_count = typer.prompt(
        "How many turned up *inside* a dream (a similar sound, not waking you)?",
        type=int, default=0,
    )
    report.cues_woke_count = typer.prompt(
        "How many woke you up?", type=int, default=0
    )
    report.cue_heard = report.cues_heard_count > 0
    report.cue_indirect = report.cues_incorporated_count > 0
    report.cue_woke_me = report.cues_woke_count > 0

    if last["wbtb"]:
        # Only asked when the night actually had a WBTB, so the field stays
        # null rather than zero on nights where the question is meaningless.
        report.minutes_to_sleep_after_wbtb = typer.prompt(
            "Roughly how many minutes to fall back asleep after WBTB?",
            type=float, default=15.0,
        )

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

    if not stats["nights_scored"]:
        # Printing "lucid nights 0" here would read as "you had no lucid
        # dreams" when the truth is that nothing has been scored. Absence of
        # data must not be reported as a finding.
        typer.secho(
            "\n  No entries have been scored for lucidity yet, so there is no\n"
            "  baseline rate — this is NOT a rate of zero.\n\n"
            "  Imported journals carry no lucidity value unless they were tagged.\n"
            "  Score them with:  morpheus review",
            fg=typer.colors.YELLOW,
        )
        if stats["mean_dreams_recalled"] is not None:
            typer.echo(f"\n  mean dreams recalled {stats['mean_dreams_recalled']:.2f}"
                       f"   (derived from paragraph counts)")
        return

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

    conn2 = connect(config.storage.db_path, read_only=True)
    quality = check_sleep_quality(conn2)
    conn2.close()
    typer.echo("")
    typer.echo(format_check(quality))


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


# ------------------------------------------------------------- import-journal


def import_journal(
    path: Path = typer.Argument(..., help="File or directory exported from your notes app."),
    commit: bool = typer.Option(
        False, "--commit", help="Actually write. Without this, only a preview is shown."
    ),
    overwrite: bool = typer.Option(
        False, help="Replace existing reports on the same dates."
    ),
    sample: int = typer.Option(3, help="How many parsed entries to show in the preview."),
    year: Optional[int] = typer.Option(
        None,
        help="Year the journal STARTS in. Required for prose journals that write "
        "dates inline as 'June 30:' without a year. The year rolls forward "
        "automatically whenever the month goes backwards.",
    ),
    untagged: str = typer.Option(
        "unscored",
        help="What an entry with no lucidity marker means: 'unscored' (excluded "
        "from the rate) or 'not-lucid' (counted as a non-lucid night).",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Import an existing dream journal.

    Preview-first by design. Parsing freeform notes is guesswork — dates live in
    filenames or headings, lucidity is tagged however you happened to tag it —
    and a journal you cannot regenerate is the wrong thing to mangle silently.
    Nothing is written until you pass --commit.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from .report.importer import format_preview, scan

    config = MorpheusConfig.load(config_path)
    try:
        preview = scan(path, base_year=year)
    except FileNotFoundError:
        typer.secho(f"not found: {path}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.echo(format_preview(preview, limit=sample))

    if not preview.usable:
        typer.secho("\nnothing importable was found", fg=typer.colors.RED)
        if year is None:
            typer.echo(
                "\nIf your journal is continuous prose with inline dates like\n"
                "  June 30: Basically I was...\n"
                "then it has no year for the parser to read. Pass the year the\n"
                "journal starts in:\n"
                f"  morpheus import-journal {path} --year 2025"
            )
        raise typer.Exit(1)

    if not commit:
        typer.echo("")
        typer.secho("Preview only — nothing written. Re-run with --commit to import.",
                    fg=typer.colors.YELLOW)
        return

    conn = open_db(config.storage.db_path)
    store = ReportStore(conn)

    if untagged not in ("unscored", "not-lucid"):
        typer.secho("--untagged must be 'unscored' or 'not-lucid'", fg=typer.colors.RED)
        raise typer.Exit(1)

    written = skipped = 0
    for entry in preview.usable:
        assert entry.entry_date
        if store.get(entry.entry_date) is not None and not overwrite:
            skipped += 1
            continue
        lucid = entry.lucid
        if lucid is None and untagged == "not-lucid":
            lucid = False
        store.submit(
            MorningReport(
                report_date=entry.entry_date,
                narrative=entry.narrative,
                lucid_binary=lucid,
                dreams_recalled=entry.dreams_recalled,
                # Deliberately left unset: these entries predate the protocol,
                # so inventing recall counts or vividness scores for them would
                # manufacture data that was never collected.
                notes="imported from prior journal",
            )
        )
        written += 1

    stats = ReportStore(conn).baseline_stats(limit=10_000)
    conn.close()

    typer.secho(f"\nimported {written} entries", fg=typer.colors.GREEN)
    if skipped:
        typer.echo(f"skipped {skipped} dates that already had reports (use --overwrite to replace)")
    if stats.get("lucid_rate_per_night") is not None:
        typer.echo(
            f"baseline now {stats['lucid_rate_per_night'] * 100:.1f}% of nights "
            f"({stats['lucid_per_week']:.2f}/week) over {stats['nights_scored']} scored nights"
        )
    typer.echo("")
    typer.echo("Treat these as a prior, not as protocol data: they were written")
    typer.echo("before the outcome wording was fixed, so they are not strictly")
    typer.echo("comparable with nights collected from here on.")


def dream_signs(
    min_nights: int = typer.Option(3, help="Minimum separate nights for a motif to count."),
    top: int = typer.Option(20, help="How many motifs to show."),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Find recurring motifs across your dream narratives.

    Feeds the dream-sign step of the conditioning protocol, which is much
    stronger when it names things that actually recur in your dreams than when
    it asks you to think of one.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from .analysis.dream_signs import extract, format_signs

    config = MorpheusConfig.load(config_path)
    if not config.storage.db_path.exists():
        typer.secho("no database yet", fg=typer.colors.RED)
        raise typer.Exit(1)

    conn = connect(config.storage.db_path, read_only=True)
    rows = conn.execute(
        "SELECT narrative FROM reports WHERE narrative IS NOT NULL AND narrative != ''"
    ).fetchall()
    conn.close()

    narratives = [r["narrative"] for r in rows]
    if not narratives:
        typer.echo("no narratives recorded yet")
        return

    signs = extract(narratives, min_nights=min_nights, top_n=top)
    typer.echo(format_signs(signs, len(narratives)))


# ------------------------------------------------------------------ calibrate


def calibrate(
    audio: bool = typer.Option(False, help="Run audio loudness calibration instead."),
    show: bool = typer.Option(False, help="Show the most recent calibration profile."),
    device: Optional[str] = typer.Option(None, help="Camera device index or path."),
    allow_auto_exposure: bool = typer.Option(False, help="Daylight development only."),
    stage: str = typer.Option(
        "signal",
        help="'signal' (desk, the H1 test), 'posture' (bed, needs the overnight "
        "mount), or 'all'.",
    ),
    segments: Optional[str] = typer.Option(
        None, help="Comma-separated segment keys, overriding --stage."
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Guided waking calibration, including the H1 positive-control test.

    The important segment is deliberate closed-eye eye movement versus closed-eye
    stillness. If the camera cannot separate those two while you are awake,
    cooperative, well lit and holding still, it will not separate anything
    subtler from a sleeping face in the dark. That is the cheapest available
    test of the whole premise, and it takes about fifteen minutes.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from .calibration.profile import build_profile, format_profile, latest
    from .calibration.profile import save as save_profile
    from .calibration.protocol import (
        SEGMENTS_BY_KEY,
        STAGE_GUIDANCE,
        STAGES,
        segments_for,
        stage_seconds,
    )
    from .calibration.runner import CalibrationRunner

    config = MorpheusConfig.load(config_path)
    conn = open_db(config.storage.db_path)

    if show:
        row = latest(conn)
        if row is None:
            conn.close()
            typer.echo("no calibration profile recorded yet")
            raise typer.Exit(0)

        def _fmt(value, spec=".3f"):
            return format(value, spec) if value is not None else "-"

        n_samples = conn.execute(
            "SELECT COUNT(*) FROM calibration_samples WHERE profile_id = ?", (row["id"],)
        ).fetchone()[0]
        conn.close()
        typer.echo(f"created            {row['created_at']}")
        typer.echo(f"verdict            {row['verdict'] or '(predates verdict recording)'}")
        typer.echo(f"eye-flow AUC       {_fmt(row['positive_control_auc'])}")
        typer.echo(f"lid-contour AUC    {_fmt(row['lid_auc'])}")
        typer.echo(f"head-turn AUC      {_fmt(row['head_turn_leakage'])}")
        typer.echo(f"baseline coherence {_fmt(row['baseline_coherence'])}   (V1: < 0.35)")
        typer.echo(f"windows            {row['windows_positive'] or '-'} positive / "
                   f"{row['windows_baseline'] or '-'} baseline")
        typer.echo(f"raw samples kept   {n_samples:,}")
        if not n_samples:
            typer.secho(
                "  This profile has no raw samples, so it cannot be re-analysed\n"
                "  without recording again. Runs from before Phase A are like this.",
                fg=typer.colors.YELLOW,
            )
        raise typer.Exit(0)

    if audio:
        _calibrate_audio(conn, config)
        conn.close()
        raise typer.Exit(0)

    if device is not None:
        config.camera.device = int(device) if device.isdigit() else device
    if allow_auto_exposure:
        config.camera.require_manual_exposure = False
    config.eye.enabled = True

    if segments:
        keys = [k.strip() for k in segments.split(",") if k.strip()]
        unknown = [k for k in keys if k not in SEGMENTS_BY_KEY]
        if unknown:
            typer.secho(f"unknown segments: {unknown}", fg=typer.colors.RED)
            raise typer.Exit(1)
        chosen = tuple(SEGMENTS_BY_KEY[k] for k in keys)
        duration = sum(s.seconds for s in chosen)
    else:
        if stage not in STAGES:
            typer.secho(f"unknown stage {stage!r}; try: {', '.join(STAGES)}", fg=typer.colors.RED)
            raise typer.Exit(1)
        chosen = segments_for(stage)
        duration = stage_seconds(stage)

    source = WebcamSource(config.camera)
    try:
        source.open()
    except AudioError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        typer.secho(f"could not open camera: {exc}", fg=typer.colors.RED)
        conn.close()
        raise typer.Exit(1)

    runner = CalibrationRunner(config, source)
    typer.echo(f"\nMorpheus calibration — stage '{stage}', {len(chosen)} segments, "
               f"about {duration // 60}m{duration % 60:02d}s")
    typer.echo("=" * 68)
    for line in _wrap(STAGE_GUIDANCE.get(stage, ""), 66):
        typer.echo(f"  {line}")
    typer.echo("")
    if stage == "posture":
        typer.secho(
            "  These segments only mean anything from the camera's real overnight\n"
            "  mount. Running them from a desk measures a situation that will never\n"
            "  occur, and will report implausibly good visibility.",
            fg=typer.colors.YELLOW,
        )
        if not typer.confirm("  Is the camera in its final overnight position?", default=False):
            typer.echo("  Mount it first, then run this again.")
            raise typer.Exit(1)

    def before(segment) -> bool:
        typer.echo("")
        typer.secho(f"  {segment.title}  ({segment.seconds}s)", fg=typer.colors.CYAN, bold=True)
        for line in _wrap(segment.instruction, 64):
            typer.echo(f"    {line}")
        typer.echo("    press Enter when you are in position, then hold")
        input()
        typer.echo("    recording...")
        return True

    try:
        collected = runner.run_all(before_segment=before, segments=chosen)
    except KeyboardInterrupt:
        typer.echo("\naborted")
        source.close()
        conn.close()
        raise typer.Exit(1)
    source.close()

    profile = build_profile(collected)
    typer.echo("")
    typer.echo(format_profile(profile))

    save_profile(conn, profile, collected=collected)
    conn.close()
    typer.secho("\nprofile saved", fg=typer.colors.GREEN)


def _calibrate_audio(conn, config: MorpheusConfig) -> None:
    """Ascending-limits loudness calibration (design.md §13.3).

    Absolute SPL at the pillow is unknowable without a meter, so this anchors
    digital gain to the user's own judgement. Ascending order matters: starting
    loud and coming down biases the faintest-audible estimate upward, because
    you already know what you are listening for.
    """
    from datetime import datetime, timezone

    from .audio.assets import CueAssetRegistry
    from .audio.player import load_wav

    registry = CueAssetRegistry(conn, config.storage.data_dir / "cues")
    trained = registry.list(trained=True)
    if not trained:
        typer.secho("register a cue first: morpheus cues --add-preset trained-ascending", fg=typer.colors.RED)
        return
    asset = trained[0]
    waveform, _ = load_wav(asset.path)

    limits = SafetyLimits()
    player = CuePlayer(SoundDeviceSink(), ceiling=limits.max_gain)

    typer.echo("\nAudio calibration")
    typer.echo("=" * 68)
    typer.echo("  Set your speaker where it will sit overnight, at the volume you")
    typer.echo("  will leave it at. Lie in your normal sleeping position. Everything")
    typer.echo("  below is relative to that physical setup — change the speaker or")
    typer.echo("  its knob afterwards and this calibration is void.\n")
    input("  press Enter when ready")

    faintest = None
    for gain in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.14, 0.18, 0.24, 0.30]:
        if gain > limits.max_gain:
            break
        player.play(player.render(waveform, gain=gain, ramp_ms=1200, duration_ms=None))
        if typer.confirm(f"  gain {gain:.2f} — could you hear it?", default=False):
            faintest = gain
            break

    if faintest is None:
        typer.secho(
            "\n  Not audible even at the ceiling. Turn the speaker up, or move it\n"
            "  closer, and run this again. Do NOT raise the software ceiling to\n"
            "  compensate — it exists to bound what a bug can do.",
            fg=typer.colors.RED,
        )
        return

    typer.echo("")
    comfortable = faintest
    for gain in [g for g in [0.05, 0.08, 0.12, 0.16, 0.20, 0.26, 0.32] if g > faintest]:
        if gain > limits.max_gain:
            break
        player.play(player.render(waveform, gain=gain, ramp_ms=1200, duration_ms=None))
        if not typer.confirm(f"  gain {gain:.2f} — still comfortable to sleep through?", default=True):
            break
        comfortable = gain

    ceiling = min(limits.max_gain, round(comfortable * 1.2, 3))
    conn.execute(
        "INSERT INTO audio_calibrations (created_at, cue_asset_id, faintest_gain, "
        "comfortable_gain, ceiling_gain, output_device) VALUES (?,?,?,?,?,?)",
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            asset.id, faintest, comfortable, ceiling, SoundDeviceSink().describe(),
        ),
    )
    typer.echo("")
    typer.secho(f"  faintest audible   {faintest:.3f}", fg=typer.colors.GREEN)
    typer.secho(f"  comfortable        {comfortable:.3f}", fg=typer.colors.GREEN)
    typer.secho(f"  suggested ceiling  {ceiling:.3f}", fg=typer.colors.GREEN)
    typer.echo("")
    typer.echo("  Cueing starts near the faintest level and adapts upward only on")
    typer.echo("  quiet outcomes. Waking you is the failure mode that matters, so")
    typer.echo("  the starting point is deliberately close to inaudible.")


# -------------------------------------------------------------------- confirm


def confirm_cmd(
    profile_id: Optional[int] = typer.Option(
        None, "--profile", help="Calibration profile id; defaults to the most recent."
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Evaluate a calibration against the FROZEN lid-geometry criteria.

    The criteria live in calibration/confirmation.py and were committed before
    the confirmation data existed. This command only reads them; there is no
    flag to adjust anything, which is the point.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from .calibration.confirmation import format_result, load_and_confirm

    config = MorpheusConfig.load(config_path)
    if not config.storage.db_path.exists():
        typer.secho("no database yet", fg=typer.colors.RED)
        raise typer.Exit(1)

    conn = connect(config.storage.db_path, read_only=True)
    try:
        result = load_and_confirm(conn, profile_id)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)
    finally:
        conn.close()

    typer.echo(format_result(result))


# --------------------------------------------------------------------- review


def review_cmd(
    limit: Optional[int] = typer.Option(None, help="Stop after this many entries."),
    suggestive_only: bool = typer.Option(
        False, help="Only entries whose wording hints at lucidity."
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Score imported journal entries that carry no lucidity tag.

    A journal kept in prose imports unscored, which leaves the baseline rate
    unmeasurable. This walks those entries so you can code them against the
    pinned outcome definition.

    The phrase matching only sorts and highlights; it never decides. Prose is
    ambiguous and only you know what happened, so classifying automatically
    would be fabricating the project's primary outcome.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from .report.review import evidence_snippet, progress, score, unscored

    config = MorpheusConfig.load(config_path)
    if not config.storage.db_path.exists():
        typer.secho("no database yet — import your journal first", fg=typer.colors.RED)
        raise typer.Exit(1)

    conn = open_db(config.storage.db_path)
    candidates = unscored(conn, limit=limit)
    if suggestive_only:
        candidates = [c for c in candidates if c.suggestive]

    if not candidates:
        stats = progress(conn)
        conn.close()
        typer.secho("nothing left to score", fg=typer.colors.GREEN)
        typer.echo(f"  {stats['scored']}/{stats['total']} entries scored, "
                   f"{stats['lucid']} lucid")
        return

    typer.echo(f"\n{len(candidates)} entries to score")
    typer.echo("=" * 68)
    typer.secho(f'Question for each: "{PRIMARY_OUTCOME_DEFINITION}"', bold=True)
    typer.echo("")
    typer.echo("  y = yes   n = no   s = skip   q = stop and save")
    typer.echo("")
    typer.echo("  These are pre-intervention nights, so over-calling lucidity raises")
    typer.echo("  your baseline and makes any later effect look smaller. If you are")
    typer.echo("  genuinely unsure, erring generous is the safe direction.")

    done = 0
    for index, candidate in enumerate(candidates, start=1):
        typer.echo("")
        typer.echo("-" * 68)
        dreams = candidate.dreams
        header = f"[{index}/{len(candidates)}]  {candidate.report_date}   {len(dreams)} dream(s)"
        typer.secho(header, fg=typer.colors.CYAN, bold=True)

        relevant = candidate.relevant_dreams()
        show_full = False

        while True:
            if show_full:
                typer.echo("")
                for number, dream in enumerate(dreams, start=1):
                    typer.secho(f"  --- dream {number}/{len(dreams)} ---", fg=typer.colors.BLUE)
                    for line in _wrap(dream, 66):
                        typer.echo(f"    {line}")
                    typer.echo("")
            elif relevant:
                typer.echo("")
                for number, dream, hints in relevant:
                    typer.secho(
                        f"  dream {number}/{len(dreams)} — {', '.join(hints)}",
                        fg=typer.colors.YELLOW,
                    )
                    for line in _wrap(evidence_snippet(dream), 66):
                        typer.echo(f"    {line}")
                    typer.echo("")
                typer.echo("  (showing only the dreams that matched; 'f' for the full night)")
            else:
                typer.echo("")
                typer.echo("  no lucidity wording in any dream this night")
                for line in _wrap(dreams[0], 66)[:4]:
                    typer.echo(f"    {line}")
                typer.echo("    ...")
                typer.echo("")
                typer.echo("  ('f' for the full night)")

            typer.echo("")
            answer = ""
            while answer not in ("y", "n", "s", "q", "f"):
                answer = typer.prompt(
                    "  were you aware you were dreaming in ANY dream that night? "
                    "[y/n/s/q, f=full]"
                ).strip().lower()[:1]
            if answer == "f" and not show_full:
                show_full = True
                continue
            break

        if answer == "q":
            break
        if answer == "s":
            continue
        score(conn, candidate.report_date, answer == "y")
        done += 1

    stats = progress(conn)
    conn.close()
    typer.echo("")
    typer.secho(f"scored {done} this session", fg=typer.colors.GREEN)
    typer.echo(f"  {stats['scored']}/{stats['total']} total, {stats['lucid']} lucid, "
               f"{stats['remaining']} remaining")
    if stats["remaining"]:
        typer.echo("  run `morpheus review` again to continue")
    else:
        typer.echo("  run `morpheus baseline` to see your pre-intervention rate")
