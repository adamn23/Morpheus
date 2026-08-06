"""Session lifecycle, batching, and provenance."""

from __future__ import annotations

import json
import time

import pytest

from morpheus.config import MorpheusConfig
from morpheus.store.db import migrate, open_db
from morpheus.store.feature_store import FeatureStore
from morpheus.types import CoverageFlag, FeatureFrame, HealthCounters


def feature(t: float = 0.0, **kwargs) -> FeatureFrame:
    defaults = dict(
        t_mono=t,
        t_utc=time.time() + t,
        n_frames=30,
        signal_quality=0.8,
        face_present=1.0,
        eye_region_usable=0.5,
        coverage_flag=CoverageFlag.USABLE,
        global_motion=0.01,
        bed_motion=0.01,
        face_motion=0.005,
    )
    defaults.update(kwargs)
    return FeatureFrame(**defaults)


def test_migrations_are_idempotent(config: MorpheusConfig) -> None:
    conn = open_db(config.storage.db_path)
    first = conn.execute("PRAGMA user_version").fetchone()[0]
    assert migrate(conn) == first
    assert migrate(conn) == first
    conn.close()


def test_batching_defers_writes_until_threshold(conn, config: MorpheusConfig) -> None:
    store = FeatureStore(conn, batch_size=5)
    store.start_session(config=config, device_profile={"camera_model": "t"})
    for i in range(4):
        store.append(feature(float(i)))
    assert conn.execute("SELECT COUNT(*) FROM frames_1hz").fetchone()[0] == 0
    store.append(feature(4.0))
    assert conn.execute("SELECT COUNT(*) FROM frames_1hz").fetchone()[0] == 5


def test_flush_persists_partial_batch(conn, config: MorpheusConfig) -> None:
    """An abrupt stop must lose at most one batch, never the whole night."""
    store = FeatureStore(conn, batch_size=100)
    store.start_session(config=config, device_profile={"camera_model": "t"})
    for i in range(7):
        store.append(feature(float(i)))
    store.flush()
    assert conn.execute("SELECT COUNT(*) FROM frames_1hz").fetchone()[0] == 7


def test_append_before_session_raises(conn) -> None:
    store = FeatureStore(conn)
    with pytest.raises(RuntimeError, match="start_session"):
        store.append(feature())


def test_config_snapshot_is_deduplicated(conn, config: MorpheusConfig) -> None:
    store = FeatureStore(conn)
    first = store.ensure_config_snapshot(config)
    assert store.ensure_config_snapshot(config) == first
    assert conn.execute("SELECT COUNT(*) FROM config_snapshots").fetchone()[0] == 1


def test_changed_config_creates_a_new_snapshot(conn, config: MorpheusConfig) -> None:
    """Provenance guarantee behind design.md §16.

    A threshold tweaked mid-study must be visible in analysis, not silently
    reinterpret every night recorded before it.
    """
    store = FeatureStore(conn)
    first = store.ensure_config_snapshot(config)
    config.coverage.min_interocular_px = 45.0
    assert store.ensure_config_snapshot(config) != first


def test_session_records_provenance(conn, config: MorpheusConfig) -> None:
    store = FeatureStore(conn, batch_size=1)
    session_id = store.start_session(config=config, device_profile={"camera_model": "cam"})
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    assert row["config_snapshot_id"] is not None
    assert row["device_profile_id"] is not None
    assert row["status"] == "running"
    assert row["night_index"] == 1


def test_night_index_increments_across_sessions(conn, config: MorpheusConfig) -> None:
    indices = []
    for _ in range(3):
        store = FeatureStore(conn)
        store.start_session(config=config, device_profile={"camera_model": "cam"})
        indices.append(
            conn.execute(
                "SELECT night_index FROM sessions WHERE id = ?", (store.session_id,)
            ).fetchone()[0]
        )
    assert indices == [1, 2, 3]


def test_finish_session_writes_health(conn, config: MorpheusConfig) -> None:
    store = FeatureStore(conn)
    store.start_session(config=config, device_profile={"camera_model": "cam"})
    health = HealthCounters(
        frames_captured=900, frames_dropped=100, read_failures=3,
        reconnects=1, seconds_recorded=30, clock_gaps=[(10.0, 25.0)],
    )
    store.finish_session("completed", health)
    row = conn.execute(
        "SELECT * FROM session_health WHERE session_id = ?", (store.session_id,)
    ).fetchone()
    assert row["frames_captured"] == 900
    assert row["capture_uptime"] == pytest.approx(0.9)
    assert json.loads(row["clock_gaps_json"]) == [[10.0, 25.0]]


def test_context_manager_marks_abort_on_exception(conn, config: MorpheusConfig) -> None:
    with pytest.raises(ValueError):
        with FeatureStore(conn) as store:
            store.start_session(config=config, device_profile={"camera_model": "cam"})
            store.append(feature())
            raise ValueError("boom")
    row = conn.execute("SELECT status FROM sessions").fetchone()
    assert row["status"] == "aborted"
    # Buffered frames are still flushed: a crashed night is partial, not lost.
    assert conn.execute("SELECT COUNT(*) FROM frames_1hz").fetchone()[0] == 1


def test_later_phase_columns_persist_as_null(conn, config: MorpheusConfig) -> None:
    store = FeatureStore(conn, batch_size=1)
    store.start_session(config=config, device_profile={"camera_model": "cam"})
    store.append(feature())
    row = conn.execute("SELECT eye_flow_l, eye_flow_bilateral_corr, resp_proxy FROM frames_1hz").fetchone()
    assert row["eye_flow_l"] is None
    assert row["eye_flow_bilateral_corr"] is None
    assert row["resp_proxy"] is None


def test_capture_uptime_handles_zero_frames() -> None:
    assert HealthCounters().capture_uptime == 0.0
