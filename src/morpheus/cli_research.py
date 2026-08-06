"""M3-M6 commands: experiment, prereg, analyse, validate, adaptive."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import typer

from .config import MorpheusConfig
from .experiment.analysis import analyse, format_result
from .experiment.assignments import ExperimentStore
from .experiment.blinding import BlindingError
from .experiment.preregistration import generate as generate_prereg
from .experiment.randomization import DESIGNS, imbalance, required_nights_per_arm
from .store.db import connect, open_db

log = logging.getLogger("morpheus.research")


def register(app: typer.Typer) -> None:
    app.command("experiment")(experiment)
    app.command("prereg")(prereg)
    app.command("reveal")(reveal)
    app.command("analyse")(analyse_cmd)
    app.command("validate")(validate_cmd)
    app.command("adaptive")(adaptive_cmd)


# ------------------------------------------------------------------ experiment


def experiment(
    create: Optional[str] = typer.Option(None, help="Create an experiment with this name."),
    design: str = typer.Option("two-arm", help=f"One of: {', '.join(DESIGNS)}"),
    seed: Optional[int] = typer.Option(None, help="Randomization seed. Omit for a random one."),
    start: Optional[int] = typer.Option(None, help="Start collection for this experiment id."),
    show_plan: Optional[int] = typer.Option(None, help="Show the assignment plan (UNBLINDS IT)."),
    nights: int = typer.Option(30, help="How many nights of plan to show."),
    audit: bool = typer.Option(False, help="Show the reveal audit log."),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Create and inspect N-of-1 experiments."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    config = MorpheusConfig.load(config_path)
    conn = open_db(config.storage.db_path)
    store = ExperimentStore(conn, config.storage.data_dir)

    if create:
        record = store.create(create, design=design, seed=seed)
        typer.secho(f"created experiment #{record.id} '{record.name}'", fg=typer.colors.GREEN)
        typer.echo(f"  design      {record.design} (seed {record.seed})")
        typer.echo(f"  fingerprint {record.plan_fingerprint}")
        typer.echo("")
        typer.echo("Next: write the pre-registration, then start collection:")
        typer.echo(f"  morpheus prereg --experiment {record.id}")
        typer.echo(f"  morpheus experiment --start {record.id}")

    if start is not None:
        try:
            store.start(start)
        except ValueError as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(1)
        typer.secho(f"experiment {start} is now running", fg=typer.colors.GREEN)

    if show_plan is not None:
        # Deliberately awkward to reach and loudly labelled: seeing future
        # assignments unblinds every night that has not happened yet.
        record = store.get(show_plan)
        typer.secho(
            "\nThis reveals FUTURE assignments and unblinds the rest of the study.\n"
            "It exists for auditing a completed experiment, not for planning.",
            fg=typer.colors.RED,
        )
        if typer.confirm("Show anyway?", default=False):
            plan = record.plan()
            for assignment in plan.sequence(nights):
                typer.echo(f"  night {assignment.night_index:>3}  {assignment.arm.value}")
            counts = plan.counts(nights)
            typer.echo(f"\n  counts over {nights} nights: "
                       + ", ".join(f"{a.value}={n}" for a, n in counts.items()))
            typer.echo(f"  max imbalance: {imbalance(counts)}")

    if audit:
        rows = store.audit_log()
        if not rows:
            typer.echo("\nno reveals recorded")
        else:
            typer.echo(f"\n{'when':<22} {'ok':<4} {'reason'}")
            typer.echo("-" * 70)
            for row in rows[:40]:
                flag = "yes" if row["legitimate"] else "NO"
                typer.echo(f"{row['revealed_at']:<22} {flag:<4} {row['reason']}")

    experiments = store.list()
    conn.close()
    if experiments and not (create or start or show_plan is not None or audit):
        typer.echo(f"\n{'id':>3}  {'name':<24} {'design':<12} {'status':<10} seed")
        typer.echo("-" * 64)
        for record in experiments:
            typer.echo(
                f"{record.id:>3}  {record.name:<24} {record.design:<12} "
                f"{record.status:<10} {record.seed}"
            )


# ---------------------------------------------------------------------- prereg


def prereg(
    experiment_id: int = typer.Option(..., "--experiment", help="Experiment id."),
    baseline: float = typer.Option(0.10, help="Assumed baseline lucid rate per night."),
    target: float = typer.Option(0.25, help="Target rate you would want to detect."),
    out: Optional[Path] = typer.Option(None, help="Also write the document here."),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Generate and store the pre-registration.

    Required before an experiment can start. Deciding the analysis after seeing
    the data is the failure the harness exists to prevent, and with a single
    participant there are no co-authors to notice it happening.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    config = MorpheusConfig.load(config_path)
    conn = open_db(config.storage.db_path)
    store = ExperimentStore(conn, config.storage.data_dir)
    record = store.get(experiment_id)

    document = generate_prereg(
        name=record.name, plan=record.plan(), design=record.design,
        baseline_rate=baseline, target_rate=target,
    )
    import hashlib

    conn.execute(
        "UPDATE experiments SET preregistration = ?, prereg_sha256 = ? WHERE id = ?",
        (document, hashlib.sha256(document.encode()).hexdigest(), experiment_id),
    )
    conn.close()

    if out:
        Path(out).write_text(document)
        typer.secho(f"written to {out}", fg=typer.colors.GREEN)
    typer.echo(document)

    needed = required_nights_per_arm(baseline, target)
    if needed:
        typer.echo("")
        typer.secho(
            f"Note: roughly {needed} nights per arm. That is the real cost of this "
            f"design, and it is a floor rather than an estimate.",
            fg=typer.colors.YELLOW,
        )


# ---------------------------------------------------------------------- reveal


def reveal(
    for_date: Optional[str] = typer.Option(None, "--date", help="YYYY-MM-DD; defaults to today."),
    force: bool = typer.Option(
        False, help="Reveal without a morning report. Recorded as unblinded."
    ),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Reveal last night's condition, after the morning report exists."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    config = MorpheusConfig.load(config_path)
    conn = open_db(config.storage.db_path)
    store = ExperimentStore(conn, config.storage.data_dir)

    record = store.active()
    if record is None:
        typer.secho("no running experiment", fg=typer.colors.RED)
        raise typer.Exit(1)

    target = for_date or date.today().isoformat()
    row = store.assignment_for_date(record.id, target)
    if row is None:
        typer.secho(f"no assignment for {target}", fg=typer.colors.RED)
        raise typer.Exit(1)

    try:
        arm = store.reveal(int(row["id"]), force=force)
    except BlindingError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        conn.close()
        raise typer.Exit(1)
    conn.close()

    typer.echo("")
    typer.secho(f"  {target}: {arm.value}", bold=True)
    if force:
        typer.secho(
            "  Force-revealed without a report. This night is unblinded and will be\n"
            "  excluded from the primary analysis.",
            fg=typer.colors.RED,
        )


# --------------------------------------------------------------------- analyse


def analyse_cmd(
    experiment_id: Optional[int] = typer.Option(None, "--experiment", help="Defaults to active."),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Run the pre-registered analysis over revealed nights."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    config = MorpheusConfig.load(config_path)
    conn = open_db(config.storage.db_path)
    store = ExperimentStore(conn, config.storage.data_dir)

    record = store.get(experiment_id) if experiment_id else store.active()
    if record is None:
        typer.secho("no experiment found", fg=typer.colors.RED)
        raise typer.Exit(1)

    result = analyse(
        conn, record.id, store.revealed_arms(record.id),
        blinding=store.blinding_integrity(record.id),
        prereg_intact=store.prereg_intact(record.id),
    )
    conn.close()
    typer.echo(format_result(result))


# -------------------------------------------------------------------- validate


def validate_cmd(
    reference: Path = typer.Argument(..., help="Hypnogram CSV from the reference device."),
    session_id: int = typer.Option(..., "--session", help="Morpheus session to align against."),
    offset_s: Optional[float] = typer.Option(
        None, help="Known clock offset. Omit to estimate it from motion."
    ),
    commit: bool = typer.Option(False, "--commit", help="Record the verdict in the database."),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Validate the camera index against a reference device (H1).

    This is the go/no-go for sensor-timed cueing. G9 reads the recorded verdict
    and stays disabled without a passing one.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from .reference.align import build_dataset, estimate_offset, load_session_frames
    from .reference.ingest import load_hypnogram_csv, summarise
    from .reference.validate import format_result as format_validation
    from .reference.validate import record as record_validation
    from .reference.validate import validate

    config = MorpheusConfig.load(config_path)
    conn = open_db(config.storage.db_path)

    epochs = load_hypnogram_csv(reference)
    info = summarise(epochs)
    typer.echo(f"reference: {info['epochs']} epochs over {info['hours']:.1f} h, "
               f"{info['rem_epochs']} scored REM")

    if offset_s is None:
        times, values = load_session_frames(conn, session_id, ("global_motion",))
        alignment = estimate_offset(times, values[:, 0] if values.size else values, epochs)
        typer.echo(
            f"clock offset: {alignment.offset_s:+.0f}s "
            f"(correlation {alignment.correlation:.2f}, {alignment.method})"
        )
        if not alignment.trustworthy:
            typer.secho(
                "  Weak alignment. Pass --offset-s with a known value rather than\n"
                "  trusting this; a mis-aligned join smears labels across state\n"
                "  boundaries and depresses AUC for reasons unrelated to the camera.",
                fg=typer.colors.YELLOW,
            )
        offset_s = alignment.offset_s

    dataset = build_dataset(conn, {session_id: epochs}, offsets={session_id: offset_s})
    typer.echo(f"aligned: {dataset.epochs_matched} epochs, {dataset.epochs_dropped} dropped")
    for reason, count in sorted(dataset.drop_reasons.items(), key=lambda kv: -kv[1]):
        typer.echo(f"    {reason}: {count}")

    result = validate(
        dataset.features, dataset.labels, dataset.groups,
        feature_names=dataset.feature_names,
        reference_source=reference.name,
    )
    typer.echo("")
    typer.echo(format_validation(result))

    if commit and not result.insufficient:
        record_validation(conn, result)
        typer.secho("\nverdict recorded", fg=typer.colors.GREEN)
    elif commit:
        typer.secho("\nnot recorded — insufficient data for a verdict", fg=typer.colors.YELLOW)
    conn.close()


# -------------------------------------------------------------------- adaptive


def adaptive_cmd(
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Show the adaptive policy's current posteriors."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from .cue.adaptive import ThompsonPolicy, build_arms, format_ranking
    from .cue.safety import SafetyLimits

    config = MorpheusConfig.load(config_path)
    if not config.storage.db_path.exists():
        typer.secho("no database yet", fg=typer.colors.RED)
        raise typer.Exit(1)

    conn = connect(config.storage.db_path, read_only=True)
    limits = SafetyLimits()
    policy = ThompsonPolicy(arms=build_arms(limits), limits=limits)
    policy.load(conn)

    counterfactuals = conn.execute(
        "SELECT COUNT(*) n, SUM(agreed) agreed FROM counterfactuals"
    ).fetchone()
    conn.close()

    typer.echo(format_ranking(policy))
    if counterfactuals and counterfactuals["n"]:
        agreed = counterfactuals["agreed"] or 0
        total = counterfactuals["n"]
        typer.echo(f"  counterfactuals   {total} logged, agreed with heuristic "
                   f"{agreed}/{total} ({agreed / total:.0%})")
        typer.echo("  Agreement near 100% means the bandit has not yet diverged from")
        typer.echo("  the policy it is meant to beat, so it has not been tested.")
