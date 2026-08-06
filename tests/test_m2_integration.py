"""Cue assets, morning reports, and a full night end to end."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from morpheus.audio.assets import (
    PRESETS,
    CueAssetRegistry,
    generate_preset,
    sha256_file,
)
from morpheus.audio.player import BufferSink, CuePlayer, load_wav, write_wav
from morpheus.capture.replay import FileReplaySource
from morpheus.config import MorpheusConfig
from morpheus.cue.controller import ControllerConfig, CueController, GateConfig
from morpheus.cue.policy import ScheduledPolicy
from morpheus.cue.safety import SafetyLimits, SafetySupervisor
from morpheus.report.schema import (
    PRIMARY_OUTCOME_DEFINITION,
    MorningReport,
    ReportStore,
)
from morpheus.runtime.night import NightRunner
from morpheus.store.cue_store import CueStore
from morpheus.store.feature_store import FeatureStore
from morpheus.types import EventKind


# ----------------------------------------------------------------- cue assets


@pytest.fixture
def registry(conn, tmp_path: Path) -> CueAssetRegistry:
    return CueAssetRegistry(conn, tmp_path / "cues")


def test_presets_are_matched_pairs() -> None:
    """Trained and control cues must differ in contour, not character.

    If the control were a different *kind* of sound, arm B would be a confound
    for "an unfamiliar noise occurred" rather than a control for "a sound
    occurred" (design.md §15.1).
    """
    trained = generate_preset("trained-ascending")
    control = generate_preset("control-descending")
    assert trained.size == control.size
    assert not np.allclose(trained, control)
    assert PRESETS["trained-ascending"] == list(reversed(PRESETS["control-descending"]))


def test_register_preset_records_hash(registry: CueAssetRegistry) -> None:
    asset = registry.create_preset("trained-ascending", trained=True)
    assert asset.trained
    assert len(asset.sha256) == 64
    assert registry.verify(asset)


def test_same_audio_cannot_be_both_trained_and_control(registry: CueAssetRegistry) -> None:
    """The control arm's integrity depends on this.

    Registering identical audio under both labels would make the experiment
    unanalysable: two arms indistinguishable by content but recorded as
    different conditions.
    """
    registry.create_preset("trained-ascending", trained=True)
    path = registry._dir / "duplicate.wav"  # noqa: SLF001
    write_wav(path, generate_preset("trained-ascending"), 44100)
    with pytest.raises(ValueError, match="cannot serve as both"):
        registry.register(path, trained=False, name="duplicate")


def test_tampered_file_fails_verification(registry: CueAssetRegistry) -> None:
    """A cue edited after registration invalidates the night's provenance."""
    asset = registry.create_preset("trained-ascending", trained=True)
    write_wav(asset.path, generate_preset("control-descending"), 44100)
    assert not registry.verify(asset)


def test_wav_round_trip_preserves_signal(tmp_path: Path) -> None:
    original = generate_preset("trained-fifth")
    path = tmp_path / "rt.wav"
    write_wav(path, original, 44100)
    loaded, samplerate = load_wav(path)
    assert samplerate == 44100
    assert loaded.size == original.size
    assert float(np.max(np.abs(loaded - original))) < 1e-3


def test_hash_is_stable_across_reads(tmp_path: Path) -> None:
    path = tmp_path / "stable.wav"
    write_wav(path, generate_preset("trained-ascending"), 44100)
    assert sha256_file(path) == sha256_file(path)


# -------------------------------------------------------------- morning report


def test_primary_outcome_definition_is_pinned() -> None:
    """Fixed before night one. Changing it breaks comparability with all prior
    reports, so it is asserted rather than left to discipline."""
    assert PRIMARY_OUTCOME_DEFINITION == (
        "At some point during a dream, I was aware that I was dreaming."
    )


def test_report_round_trip(conn) -> None:
    store = ReportStore(conn)
    store.submit(
        MorningReport(
            report_date="2026-08-06", narrative="I was in a library that kept rearranging.",
            lucid_binary=True, lucid_confidence=3, knew_was_dreaming=True,
            cue_heard=False, cue_indirect=True, cue_woke_me=False,
            dreams_recalled=2, vividness=4, sleep_quality=3, awakenings=1,
            guessed_condition="trained",
        )
    )
    row = store.get("2026-08-06")
    assert row["lucid_binary"] == 1
    assert row["cue_indirect"] == 1
    assert "library" in row["narrative"]


def test_resubmitting_a_date_updates_rather_than_duplicates(conn) -> None:
    """Recall improves over the first few minutes; amending must be easy."""
    store = ReportStore(conn)
    store.submit(MorningReport(report_date="2026-08-06", dreams_recalled=1))
    store.submit(MorningReport(report_date="2026-08-06", dreams_recalled=3, narrative="more detail"))
    assert len(store.recent()) == 1
    assert store.get("2026-08-06")["dreams_recalled"] == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lucid_confidence": 9},
        {"vividness": 0},
        {"sleep_quality": 7},
        {"dreams_recalled": -1},
        {"lucid_binary": True, "dreams_recalled": 0},
    ],
)
def test_invalid_reports_are_rejected(conn, kwargs) -> None:
    store = ReportStore(conn)
    with pytest.raises(ValueError):
        store.submit(MorningReport(report_date="2026-08-06", **kwargs))


def test_baseline_stats(conn) -> None:
    store = ReportStore(conn)
    for day in range(1, 15):
        store.submit(
            MorningReport(
                report_date=f"2026-08-{day:02d}",
                lucid_binary=(day % 7 == 0),
                dreams_recalled=2, sleep_quality=4,
            )
        )
    stats = store.baseline_stats()
    assert stats["nights"] == 14
    assert stats["lucid_nights"] == 2
    assert stats["lucid_rate_per_night"] == pytest.approx(2 / 14)
    assert stats["lucid_per_week"] == pytest.approx(1.0)


# ------------------------------------------------------------ event integrity


def test_event_kind_must_be_the_enum(conn, config: MorpheusConfig) -> None:
    """A bare string here would route around the whole naming discipline."""
    store = FeatureStore(conn)
    session_id = store.start_session(config=config, device_profile={"camera_model": "t"})
    cues = CueStore(conn)
    with pytest.raises(TypeError):
        cues.record_event(session_id, 1.0, "rem detected")  # type: ignore[arg-type]
    cues.record_event(session_id, 1.0, EventKind.PROBABLE_AROUSAL)
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


# -------------------------------------------------------------- full night


def test_full_night_end_to_end(conn, config: MorpheusConfig, registry, make_video) -> None:
    """Source -> vision -> features -> controller -> cue -> outcome -> database.

    Runs against a replay clip so the whole path executes in seconds, with a
    BufferSink standing in for the speaker so nothing is audible.
    """
    asset = registry.create_preset("trained-ascending", trained=True)
    clip = make_video(frames=30 * 40, fps=30.0, size=(320, 240))

    limits = SafetyLimits(min_delay_s=0.0, max_cues_per_night=2, max_cues_per_hour=9, min_cooldown_s=5.0)
    supervisor = SafetySupervisor(limits=limits)
    controller = CueController(
        supervisor,
        policy=ScheduledPolicy(),
        # The synthetic clip has no face and modest noise; relax the sensing
        # gates so this exercises the cue path rather than the vision gates,
        # which have their own tests.
        config=ControllerConfig(gates=GateConfig(min_signal_quality=0.0, max_body_motion=1.0)),
    )
    sink = BufferSink()
    player = CuePlayer(sink, ceiling=limits.max_gain)

    features = FeatureStore(conn, batch_size=5)
    session_id = features.start_session(
        config=config, device_profile={"camera_model": "replay"}, kind="cue_night"
    )
    cue_store = CueStore(conn)

    runner = NightRunner(
        config, controller=controller, player=player, asset=asset, registry=registry,
        feature_store=features, cue_store=cue_store, source=FileReplaySource(clip),
        dry_run=False,
    )
    summary = runner.run(hours=1.0, session_id=session_id)
    features.finish_session(summary.status, summary.health)

    assert summary.status == "completed"
    assert summary.cues_played >= 1
    assert summary.cues_failed == 0
    assert summary.cues_played <= limits.max_cues_per_night

    rows = cue_store.cues_for_session(session_id)
    assert len(rows) == summary.cues_played
    for row in rows:
        assert row["played"] == 1
        assert row["asset_sha256"] == asset.sha256
        assert row["gain"] <= limits.max_gain + 1e-9
        assert row["ramp_ms"] > 0
        assert row["gate_snapshot_json"]

    # Audio actually reached the sink, under the ceiling, starting from silence.
    assert len(sink.played) == summary.cues_played
    for buffer, _ in sink.played:
        assert float(np.max(np.abs(buffer))) <= limits.max_gain + 1e-6
        assert abs(float(buffer[0])) < 1e-3

    assert conn.execute("SELECT COUNT(*) FROM frames_1hz WHERE session_id=?", (session_id,)).fetchone()[0] > 0


def test_night_refuses_a_tampered_cue_asset(conn, config: MorpheusConfig, registry) -> None:
    """A changed cue file means the trained/control label is no longer trustworthy."""
    from morpheus.audio.player import AudioError

    asset = registry.create_preset("trained-ascending", trained=True)
    write_wav(asset.path, generate_preset("control-descending"), 44100)

    supervisor = SafetySupervisor(limits=SafetyLimits(min_delay_s=0.0))
    runner = NightRunner(
        config,
        controller=CueController(supervisor),
        player=CuePlayer(BufferSink(), ceiling=0.3),
        asset=asset, registry=registry,
        feature_store=FeatureStore(conn), cue_store=CueStore(conn), source=None,
    )
    with pytest.raises(AudioError, match="registered hash"):
        runner.run(hours=0.001, session_id=1)


def test_dry_run_plays_no_audio(conn, config: MorpheusConfig, registry, make_video) -> None:
    """The recommended first night: decide and log cues, make no sound."""
    asset = registry.create_preset("trained-ascending", trained=True)
    clip = make_video(frames=30 * 20, fps=30.0, size=(320, 240))

    supervisor = SafetySupervisor(
        limits=SafetyLimits(min_delay_s=0.0, max_cues_per_night=2, min_cooldown_s=3.0)
    )
    controller = CueController(
        supervisor,
        config=ControllerConfig(gates=GateConfig(min_signal_quality=0.0, max_body_motion=1.0)),
    )
    sink = BufferSink()
    features = FeatureStore(conn, batch_size=5)
    session_id = features.start_session(
        config=config, device_profile={"camera_model": "replay"}, kind="cue_night"
    )

    runner = NightRunner(
        config, controller=controller, player=CuePlayer(sink, ceiling=0.3),
        asset=asset, registry=registry, feature_store=features,
        cue_store=CueStore(conn), source=FileReplaySource(clip), dry_run=True,
    )
    summary = runner.run(hours=1.0, session_id=session_id)

    assert summary.cues_played >= 1
    assert sink.played == [], "dry run must produce no audio"
    rows = CueStore(conn).cues_for_session(session_id)
    assert all(row["played"] == 0 and row["error"] == "dry_run" for row in rows)
