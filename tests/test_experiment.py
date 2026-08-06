"""M3: randomization, blinding, pre-registration, and analysis.

The blinding tests matter most. Everything else here is arithmetic that would
show up as a wrong number; blinding failures show up as a *convincing* number,
which is worse.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from morpheus.experiment.analysis import analyse, summarise_arm
from morpheus.experiment.assignments import ExperimentStore
from morpheus.experiment.blinding import BlindingError, seal, unseal
from morpheus.experiment.preregistration import generate as generate_prereg
from morpheus.experiment.randomization import (
    Arm,
    imbalance,
    make_plan,
    required_nights_per_arm,
)
from morpheus.report.schema import MorningReport, ReportStore


@pytest.fixture
def store(conn, tmp_path):
    return ExperimentStore(conn, tmp_path)


# ------------------------------------------------------------ randomization


def test_plan_is_reproducible_from_seed() -> None:
    a = make_plan(seed=4242, design="three-arm")
    b = make_plan(seed=4242, design="three-arm")
    assert [x.arm for x in a.sequence(60)] == [x.arm for x in b.sequence(60)]
    assert a.fingerprint() == b.fingerprint()


def test_different_seeds_differ() -> None:
    a = make_plan(seed=1, design="three-arm").sequence(30)
    b = make_plan(seed=2, design="three-arm").sequence(30)
    assert [x.arm for x in a] != [x.arm for x in b]


def test_blocks_are_exactly_balanced() -> None:
    plan = make_plan(seed=7, design="three-arm", repeats_per_block=2)
    counts = plan.counts(plan.block_size * 5)
    assert len(set(counts.values())) == 1


@given(seed=st.integers(0, 2**31 - 1), nights=st.integers(1, 200))
@settings(max_examples=80, deadline=None)
def test_imbalance_is_bounded_by_block_size(seed: int, nights: int) -> None:
    """The property that makes block randomization worth its predictability cost.

    Simple randomization can drift arbitrarily far; at N-of-1 scale that wastes
    weeks. Imbalance here can never exceed the block size, whatever the seed or
    the stopping point.
    """
    plan = make_plan(seed=seed, design="three-arm")
    assert imbalance(plan.counts(nights)) <= plan.block_size


def test_assignment_is_o1_and_consistent() -> None:
    """assignment_for(n) must not depend on which nights were asked for first."""
    plan = make_plan(seed=11, design="two-arm")
    direct = plan.assignment_for(97).arm
    sequential = plan.sequence(97)[-1].arm
    assert direct == sequential


def test_power_calculation_is_sobering() -> None:
    """The calendar cost is the dominant fact about this design."""
    assert required_nights_per_arm(0.10, 0.25) >= 80
    assert required_nights_per_arm(0.10, 0.12) > 1000  # tiny effects are unreachable
    assert required_nights_per_arm(0.1, 0.1) is None


# ------------------------------------------------------------------ sealing


def test_seal_round_trip() -> None:
    key = b"k" * 32
    assert unseal(key, "n1", seal(key, "n1", "trained_cue")) == "trained_cue"


def test_sealed_value_does_not_leak_plaintext() -> None:
    key = b"k" * 32
    sealed = seal(key, "n1", "trained_cue")
    assert "trained" not in sealed and "cue" not in sealed


def test_same_arm_seals_differently_per_night() -> None:
    """A repeated ciphertext would let arms be matched up by eye."""
    key = b"k" * 32
    assert seal(key, "exp1/night1", "no_cue") != seal(key, "exp1/night2", "no_cue")


# ----------------------------------------------------------------- blinding


def test_reveal_blocked_until_report_exists(store, conn) -> None:
    """The core integrity rule of the whole harness."""
    exp = store.create("t", design="two-arm", seed=1, preregistration="x")
    store.start(exp.id)
    aid = store.assign_night(exp, "2026-08-06")

    with pytest.raises(BlindingError, match="no morning report"):
        store.reveal(aid)

    ReportStore(conn).submit(MorningReport(report_date="2026-08-06", lucid_binary=False))
    assert store.reveal(aid) in set(Arm)


def test_blocked_attempt_is_audited(store, conn) -> None:
    exp = store.create("t", design="two-arm", seed=1, preregistration="x")
    store.start(exp.id)
    aid = store.assign_night(exp, "2026-08-06")
    with pytest.raises(BlindingError):
        store.reveal(aid)

    integrity = store.blinding_integrity(exp.id)
    assert integrity["blocked_attempts"] == 1
    assert integrity["forced_reveals"] == 0


def test_forced_reveal_is_recorded_as_unblinded(store, conn) -> None:
    """The escape hatch must leave a mark.

    A hard block would eventually be worked around by editing the database,
    which leaves no trace at all. Recording the breach beats pretending it
    cannot happen.
    """
    exp = store.create("t", design="two-arm", seed=1, preregistration="x")
    store.start(exp.id)
    aid = store.assign_night(exp, "2026-08-06")
    store.reveal(aid, force=True)

    integrity = store.blinding_integrity(exp.id)
    assert integrity["forced_reveals"] == 1


def test_daemon_read_does_not_count_as_unblinding(store) -> None:
    """The cue engine must know the arm; that is not a breach of the blind."""
    exp = store.create("t", design="two-arm", seed=1, preregistration="x")
    store.start(exp.id)
    aid = store.assign_night(exp, "2026-08-06")
    store.arm_for_running_night(aid)
    assert store.blinding_integrity(exp.id)["forced_reveals"] == 0


def test_unrevealed_nights_are_invisible_to_analysis(store, conn) -> None:
    """Stops a mid-study peek being laundered through the analysis command."""
    exp = store.create("t", design="two-arm", seed=1, preregistration="x")
    store.start(exp.id)
    reports = ReportStore(conn)
    for day in range(1, 6):
        d = f"2026-08-0{day}"
        store.assign_night(exp, d)
        reports.submit(MorningReport(report_date=d, lucid_binary=True, dreams_recalled=1))

    assert store.revealed_arms(exp.id) == []
    assert analyse(conn, exp.id, store.revealed_arms(exp.id)).total_nights == 0


def test_experiment_cannot_start_without_preregistration(store) -> None:
    exp = store.create("t", design="two-arm", seed=1)
    with pytest.raises(ValueError, match="pre-registration"):
        store.start(exp.id)


def test_edited_preregistration_is_detected(store, conn) -> None:
    """A pre-registration that can be edited after the fact is not one."""
    exp = store.create("t", design="two-arm", seed=1, preregistration="original plan")
    assert store.prereg_intact(exp.id)
    conn.execute("UPDATE experiments SET preregistration = ? WHERE id = ?", ("edited", exp.id))
    assert not store.prereg_intact(exp.id)


def test_assignment_is_idempotent_per_date(store) -> None:
    """Re-arming the same night must not consume a new assignment."""
    exp = store.create("t", design="two-arm", seed=1, preregistration="x")
    store.start(exp.id)
    first = store.assign_night(exp, "2026-08-06")
    assert store.assign_night(exp, "2026-08-06") == first


# ----------------------------------------------------------------- analysis


def test_posterior_interval_widens_with_less_data() -> None:
    rng = np.random.default_rng(0)
    few = summarise_arm(Arm.NO_CUE, 2, 10, rng)
    many = summarise_arm(Arm.NO_CUE, 20, 100, rng)
    assert (few.ci_high - few.ci_low) > (many.ci_high - many.ci_low)


def test_analysis_recovers_a_known_effect(store, conn) -> None:
    exp = store.create("sim", design="two-arm", seed=99, preregistration="x")
    store.start(exp.id)
    reports = ReportStore(conn)
    rng = np.random.default_rng(0)
    truth = {Arm.TRAINED_CUE: 0.35, Arm.NO_CUE: 0.05}

    start = date(2026, 1, 1)
    for n in range(120):
        day = (start + timedelta(days=n)).isoformat()
        aid = store.assign_night(exp, day)
        arm = store.arm_for_running_night(aid)
        reports.submit(
            MorningReport(
                report_date=day,
                lucid_binary=bool(rng.random() < truth[arm]),
                dreams_recalled=1,
            )
        )
        store.reveal(aid)

    result = analyse(conn, exp.id, store.revealed_arms(exp.id))
    assert result.total_nights == 120
    comparison = result.comparisons[0]
    assert comparison.prob_treatment_better > 0.95
    assert comparison.diff_ci_low > 0


def test_analysis_finds_nothing_when_there_is_nothing(store, conn) -> None:
    """A null effect must not produce a confident answer."""
    exp = store.create("null", design="two-arm", seed=5, preregistration="x")
    store.start(exp.id)
    reports = ReportStore(conn)
    rng = np.random.default_rng(1)

    start = date(2026, 1, 1)
    for n in range(100):
        day = (start + timedelta(days=n)).isoformat()
        aid = store.assign_night(exp, day)
        store.arm_for_running_night(aid)
        reports.submit(
            MorningReport(report_date=day, lucid_binary=bool(rng.random() < 0.15), dreams_recalled=1)
        )
        store.reveal(aid)

    comparison = analyse(conn, exp.id, store.revealed_arms(exp.id)).comparisons[0]
    assert not comparison.credible_interval_excludes_zero


def test_credible_intervals_are_calibrated() -> None:
    """A 95% interval should contain the truth about 95% of the time.

    Worth checking directly: an interval that is quietly too narrow would make
    every result look more certain than it is, which is the exact failure this
    project is built to avoid.
    """
    rng = np.random.default_rng(11)
    true_rate = 0.2
    covered = 0
    trials = 300
    for _ in range(trials):
        nights = 60
        lucid = int(rng.binomial(nights, true_rate))
        summary = summarise_arm(Arm.TRAINED_CUE, lucid, nights, rng)
        if summary.ci_low <= true_rate <= summary.ci_high:
            covered += 1
    assert 0.90 <= covered / trials <= 1.0


def test_unscored_reports_are_excluded_with_a_reason(store, conn) -> None:
    exp = store.create("t", design="two-arm", seed=1, preregistration="x")
    store.start(exp.id)
    reports = ReportStore(conn)
    for day in range(1, 5):
        d = f"2026-08-0{day}"
        aid = store.assign_night(exp, d)
        reports.submit(MorningReport(report_date=d, lucid_binary=None))
        store.reveal(aid)

    result = analyse(conn, exp.id, store.revealed_arms(exp.id))
    assert result.total_nights == 0
    assert result.exclusion_reasons["outcome not scored"] == 4


# --------------------------------------------------------- pre-registration


def test_preregistration_states_the_negative_result() -> None:
    """A plan that cannot fail is not a plan."""
    plan = make_plan(seed=1, design="two-arm")
    document = generate_prereg(name="t", plan=plan, design="two-arm")
    assert "negative result" in document.lower()
    assert "At some point during a dream" in document
    assert str(plan.seed) in document
