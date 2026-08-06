"""M4-M6: reference validation, the adaptive layer, and the G9 lock.

The single most important test in this file is
`test_g9_refuses_without_validation`. Everything else is quality; that one is
integrity. Sensor-timed cueing that runs on an unvalidated index would produce
data that looks like evidence about cue timing while being evidence about
nothing, and it would do so convincingly.
"""

from __future__ import annotations

import numpy as np
import pytest

from morpheus.cue.adaptive import (
    COLD_START_PULLS,
    ThompsonPolicy,
    build_arms,
    log_counterfactual,
    reward_for,
)
from morpheus.cue.controller import CueController
from morpheus.cue.safety import SafetyLimits, SafetySupervisor
from morpheus.cue.sensor_timing import (
    ActivityIndex,
    SensorTimingAuthorization,
    SensorTimingConfig,
    SensorTimingLocked,
    StickyTwoStateHMM,
)
from morpheus.cue.state import Gate, Outcome
from morpheus.reference.ingest import (
    ReferenceEpoch,
    load_hypnogram_csv,
    normalise_stage,
    parse_timestamp,
)
from morpheus.reference.validate import (
    AUC_KILL,
    AUC_PASS,
    latest_passing,
    record,
    validate,
)

FEATURES = ["eye_flow_bilateral_corr", "n1", "n2", "n3", "n4"]


def synthetic(signal: float, *, nights: int = 8, epochs: int = 120, seed: int = 3):
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for night in range(nights):
        rem = rng.random(epochs) < 0.22
        informative = rng.normal(0, 1, epochs) + signal * rem
        for i in range(epochs):
            X.append([informative[i], *rng.normal(0, 1, 4)])
            y.append(int(rem[i]))
            g.append(night)
    return np.array(X), np.array(y), np.array(g)


# ------------------------------------------------------------ reference I/O


@pytest.mark.parametrize(
    "raw,expected",
    [("REM", "R"), ("r", "R"), ("5", "R"), ("Wake", "W"), ("N3", "N3"),
     ("s4", "N3"), ("artifact", "UNKNOWN"), ("", "UNKNOWN")],
)
def test_stage_normalisation(raw: str, expected: str) -> None:
    assert normalise_stage(raw) == expected


def test_timestamp_formats() -> None:
    assert parse_timestamp("2026-08-06T03:15:00") is not None
    assert parse_timestamp("1785000000") == 1785000000.0
    # Millisecond epochs appear in exports without being labelled as such.
    assert parse_timestamp("1785000000000") == 1785000000.0
    assert parse_timestamp("not a time") is None


def test_hypnogram_csv_round_trip(tmp_path) -> None:
    path = tmp_path / "hyp.csv"
    path.write_text("timestamp,stage\n1785000000,W\n1785000030,N2\n1785000060,REM\n")
    epochs = load_hypnogram_csv(path)
    assert [e.stage for e in epochs] == ["W", "N2", "R"]
    assert epochs[2].reference_scored_rem and epochs[2].is_sleep
    assert not epochs[0].is_sleep


def test_headerless_hypnogram_needs_a_start_time(tmp_path) -> None:
    path = tmp_path / "stages.csv"
    path.write_text("W\nN1\nN2\nREM\n")
    epochs = load_hypnogram_csv(path, start_utc=1785000000.0)
    assert len(epochs) == 4
    assert epochs[3].t_utc == 1785000000.0 + 90.0


# --------------------------------------------------------------- validation


def test_strong_signal_passes() -> None:
    X, y, g = synthetic(1.6)
    result = validate(X, y, g, feature_names=FEATURES)
    assert result.auc > AUC_PASS
    assert result.verdict == "PASS"


def test_pure_noise_fails() -> None:
    X, y, g = synthetic(0.0)
    result = validate(X, y, g, feature_names=FEATURES)
    assert result.auc < AUC_PASS
    assert result.verdict in {"FAIL", "INCONCLUSIVE"}


def test_night_grouping_prevents_leakage() -> None:
    """The methodological point the whole validation rests on.

    Consecutive epochs are heavily autocorrelated. Splitting by window instead
    of by night lets a model memorise per-night quirks and report an AUC near
    1.0 while being useless on an unseen night. Here a feature encodes the night
    index perfectly and carries no real signal; grouping must deny the shortcut.
    """
    X, y, g = synthetic(0.0, seed=9)
    for night in range(8):
        X[g == night, 1] += night * 10.0
    result = validate(X, y, g, feature_names=FEATURES)
    assert result.auc < 0.6, f"night identity leaked into the score (AUC {result.auc:.3f})"


def test_insufficient_data_yields_no_verdict() -> None:
    X, y, g = synthetic(1.5, nights=2, epochs=30)
    result = validate(X, y, g, feature_names=FEATURES)
    assert result.insufficient
    assert result.verdict == "INSUFFICIENT DATA"
    assert not result.passed


def test_thresholds_are_the_precommitted_ones() -> None:
    """Pinned so a disappointing number cannot renegotiate the gate."""
    assert AUC_PASS == 0.70
    assert AUC_KILL == 0.65


def test_bootstrap_ci_is_reported(conn) -> None:
    X, y, g = synthetic(1.4)
    result = validate(X, y, g, feature_names=FEATURES)
    low, high = result.auc_ci
    assert low is not None and low < result.auc < high


# ------------------------------------------------------------- the G9 lock


def test_g9_refuses_without_validation() -> None:
    """Sensor-timed cueing must be impossible to enable unvalidated."""
    supervisor = SafetySupervisor(limits=SafetyLimits())
    with pytest.raises(SensorTimingLocked, match="H1"):
        CueController(supervisor, sensor_timing=SensorTimingConfig())


def test_g9_refuses_with_a_failing_authorization() -> None:
    supervisor = SafetySupervisor(limits=SafetyLimits())
    failing = SensorTimingAuthorization(False, "AUC 0.51, below threshold")
    with pytest.raises(SensorTimingLocked):
        CueController(supervisor, sensor_timing=SensorTimingConfig(), authorization=failing)


def test_authorization_denied_on_an_empty_database(conn) -> None:
    auth = SensorTimingAuthorization.from_db(conn)
    assert not auth.authorized
    assert "no passing H1 validation" in auth.reason


def test_authorization_granted_only_after_a_passing_record(conn) -> None:
    X, y, g = synthetic(0.0)
    record(conn, validate(X, y, g, feature_names=FEATURES))
    assert latest_passing(conn) is None
    assert not SensorTimingAuthorization.from_db(conn).authorized

    X, y, g = synthetic(1.8)
    record(conn, validate(X, y, g, feature_names=FEATURES))
    assert latest_passing(conn) is not None
    assert SensorTimingAuthorization.from_db(conn).authorized


def test_g9_absent_from_the_gate_stack_in_scheduled_mode() -> None:
    """Not auto-passing — absent. A gate snapshot should never suggest eye
    activity had a say when it did not."""
    controller = CueController(SafetySupervisor(limits=SafetyLimits(min_delay_s=0.0)))
    controller.arm(0.0)
    assert not controller.sensor_timing_active

    from tests.test_cue_controller import frame

    controller.step(frame(1.0))
    snapshot = controller._evaluate_gates(frame(2.0))  # noqa: SLF001
    assert Gate.G9_EYE_ACTIVITY.value not in snapshot.to_dict()


def test_g9_present_once_authorized() -> None:
    from tests.test_cue_controller import frame

    controller = CueController(
        SafetySupervisor(limits=SafetyLimits(min_delay_s=0.0)),
        sensor_timing=SensorTimingConfig(),
        authorization=SensorTimingAuthorization.unlocked_for_testing(),
    )
    controller.arm(0.0)
    snapshot = controller._evaluate_gates(frame(1.0))  # noqa: SLF001
    assert Gate.G9_EYE_ACTIVITY.value in snapshot.to_dict()


# ----------------------------------------------------------- activity index


def test_activity_index_needs_a_baseline() -> None:
    index = ActivityIndex(SensorTimingConfig())
    for t in range(60):
        assert index.update(5.0, coverage=1.0, t_mono=float(t)) is False


def test_activity_index_requires_persistence() -> None:
    """A single spike is not a burst."""
    config = SensorTimingConfig(min_persistence_s=20.0)
    index = ActivityIndex(config)
    for t in range(200):
        index.update(0.1 + 0.01 * (t % 3), coverage=1.0, t_mono=float(t))
    assert index.update(10.0, coverage=1.0, t_mono=200.0) is False


def test_activity_index_fires_on_a_sustained_burst() -> None:
    config = SensorTimingConfig(min_persistence_s=10.0)
    index = ActivityIndex(config)
    for t in range(200):
        index.update(0.1, coverage=1.0, t_mono=float(t))
    fired = [index.update(9.0, coverage=1.0, t_mono=200.0 + t) for t in range(20)]
    assert any(fired)


def test_activity_index_ignores_low_coverage() -> None:
    """A value built from three visible seconds out of thirty is not an estimate."""
    config = SensorTimingConfig(min_coverage=0.5)
    index = ActivityIndex(config)
    for t in range(200):
        index.update(0.1, coverage=1.0, t_mono=float(t))
    assert index.update(50.0, coverage=0.1, t_mono=201.0) is False


# ----------------------------------------------------------------- the HMM


def test_hmm_smooths_a_noisy_two_state_signal() -> None:
    rng = np.random.default_rng(0)
    truth = np.concatenate([np.zeros(300), np.ones(300), np.zeros(300)])
    noisy = truth + rng.normal(0, 0.9, truth.size)
    posterior = StickyTwoStateHMM(0.98).posterior(noisy)

    assert np.nanmean(posterior[300:600]) > np.nanmean(posterior[:300])
    # Smoothing should cut the state-flip rate well below the raw threshold.
    raw_flips = np.sum(np.diff((noisy > noisy.mean()).astype(int)) != 0)
    smooth_flips = np.sum(np.diff((posterior > 0.5).astype(int)) != 0)
    assert smooth_flips < raw_flips / 5


def test_hmm_degrades_gracefully_on_short_or_flat_input() -> None:
    hmm = StickyTwoStateHMM()
    assert np.all(np.isnan(hmm.posterior([1.0, 2.0])))
    assert np.all(np.isnan(hmm.posterior([1.0] * 200)))


# ------------------------------------------------------------- adaptive M5


def test_all_arms_are_inside_the_safety_envelope() -> None:
    """The structural half of the guarantee: no unsafe arm exists to pick."""
    limits = SafetyLimits(max_gain=0.25, min_gain=0.05)
    for arm in build_arms(limits):
        assert limits.min_gain <= arm.gain <= limits.max_gain


def test_reward_penalises_waking_hardest() -> None:
    """An objective of pure salience has an obvious degenerate solution: shout."""
    assert reward_for(Outcome.POSSIBLE_AWAKENING) == 0.0
    assert reward_for(Outcome.PROBABLE_AROUSAL) < reward_for(Outcome.QUIET)
    assert reward_for(Outcome.QUIET, lucid=True) > reward_for(Outcome.QUIET, cue_heard=True)
    assert reward_for(Outcome.UNCERTAIN) is None


def test_bandit_converges_on_the_better_arm() -> None:
    limits = SafetyLimits()
    arms = build_arms(limits)
    best = arms[4]
    rng = np.random.default_rng(0)
    policy = ThompsonPolicy(arms=arms, limits=limits, rng=rng)

    for _ in range(500):
        arm = policy.select()
        p = 0.7 if arm.key == best.key else 0.2
        policy.update(arm, 1.0 if rng.random() < p else 0.0)

    assert policy.ranking()[0][0].key == best.key


def test_cold_start_cycles_every_arm_before_preferring_one() -> None:
    limits = SafetyLimits()
    arms = build_arms(limits)
    policy = ThompsonPolicy(arms=arms, limits=limits)
    chosen = set()
    for _ in range(len(arms)):
        arm = policy.select()
        chosen.add(arm.key)
        policy.update(arm, 0.5)
    assert len(chosen) == len(arms)


def test_uncertain_outcome_does_not_update_the_posterior() -> None:
    limits = SafetyLimits()
    policy = ThompsonPolicy(arms=build_arms(limits), limits=limits)
    arm = policy.arms[0]
    before = policy.posterior(arm)
    policy.update(arm, reward_for(Outcome.UNCERTAIN))
    assert policy.posterior(arm) == before


def test_posteriors_survive_a_restart(conn) -> None:
    limits = SafetyLimits()
    arms = build_arms(limits)
    first = ThompsonPolicy(arms=arms, limits=limits)
    for _ in range(30):
        first.update(first.select(), 1.0)
    first.save(conn)

    second = ThompsonPolicy(arms=arms, limits=limits)
    second.load(conn)
    assert second.total_pulls == first.total_pulls
    for arm in arms:
        assert second.posterior(arm) == first.posterior(arm)


def test_bandit_gain_never_exceeds_the_ceiling() -> None:
    limits = SafetyLimits(max_gain=0.2)
    policy = ThompsonPolicy(arms=build_arms(limits), limits=limits)
    for i in range(200):
        assert policy.propose_gain(cue_index=i, last_outcome="quiet") <= limits.max_gain


def test_counterfactual_logging(conn) -> None:
    """Without this there is no way to tell whether the bandit beat the
    heuristic it replaced."""
    log_counterfactual(
        conn, cue_id=None, t_mono=1.0, chosen_policy="thompson-v1", chosen_arm="g0.1/d+0",
        baseline_policy="scheduled-v1", baseline_arm="g0.08/d+0",
    )
    row = conn.execute("SELECT * FROM counterfactuals").fetchone()
    assert row["agreed"] == 0
    assert row["baseline_policy"] == "scheduled-v1"
