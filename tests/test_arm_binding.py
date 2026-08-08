"""Binding a sealed arm to the cue that actually plays.

`Arm.UNTRAINED_CUE` existed in the enum and in the pre-registration text for
months, but nothing connected it to an audio file: `night` took `--cue-id` or
grabbed the first trained asset regardless of condition. An experiment run in
that state would have played the *trained* cue on every night of both arms and
produced a clean, confident, entirely meaningless null.
"""

from __future__ import annotations

import pytest

from morpheus.audio.assets import MATCHED_CONTROL, PRESETS, CueAssetRegistry
from morpheus.cli_m2 import _resolve_arm
from morpheus.config import MorpheusConfig
from morpheus.experiment.assignments import ExperimentStore
from morpheus.experiment.randomization import DESIGNS, Arm
from morpheus.store.db import open_db


@pytest.fixture
def setup(tmp_path):
    config = MorpheusConfig()
    config.storage.data_dir = tmp_path
    conn = open_db(config.storage.db_path)
    registry = CueAssetRegistry(conn, tmp_path / "cues")
    trained = registry.create_preset("trained-ascending", trained=True)
    yield config, conn, registry, trained
    conn.close()


class TestMatchedPairing:
    def test_the_pair_in_use_is_an_exact_multiset_match(self):
        """Identical tones in reverse order — same spectral content, same total
        energy. That equality is what makes the untrained arm a control for
        'a sound occurred' rather than a confound for 'a different sound'."""
        trained = PRESETS["trained-ascending"]
        control = PRESETS[MATCHED_CONTROL["trained-ascending"]]
        assert sorted(trained) == sorted(control)
        assert trained != control, "the control must differ in contour"

    def test_every_pairing_shares_length_and_register(self):
        for trained_name, control_name in MATCHED_CONTROL.items():
            trained, control = PRESETS[trained_name], PRESETS[control_name]
            assert len(trained) == len(control), trained_name
            assert set(trained) == set(control), trained_name

    def test_unmatched_cue_refuses_rather_than_substituting(self, tmp_path, setup):
        """A homemade cue has no twin. Substituting an unmatched sound would
        turn the control arm into a confound, invisibly, mid-trial."""
        _, _, registry, _ = setup
        from morpheus.audio.assets import synth_motif
        from morpheus.audio.player import write_wav

        path = tmp_path / "homemade.wav"
        write_wav(path, synth_motif([440.0, 466.16, 493.88, 523.25]), 44100)
        orphan = registry.register(path, trained=True, name="homemade")
        assert orphan.name == "homemade"
        with pytest.raises(KeyError, match="no acoustically matched control"):
            registry.matched_control_for(orphan)


class TestArmSelectsTheAsset:
    def test_no_experiment_is_a_no_op(self, setup):
        config, conn, registry, trained = setup
        asset, plays, assignment = _resolve_arm(
            conn, config, registry, trained, "2026-08-09"
        )
        assert asset.name == trained.name and plays and assignment is None

    def test_each_arm_maps_to_the_right_cue(self, setup):
        config, conn, registry, trained = setup
        store = ExperimentStore(conn, config.storage.data_dir)

        seen: dict[Arm, tuple[str, bool]] = {}
        # Walk enough nights that a two-arm-matched block randomisation has
        # certainly produced both arms.
        for day in range(1, 9):
            experiment = store.create(
                f"e{day}", design="two-arm-matched", preregistration="frozen for test"
            )
            store.start(experiment.id)
            date = f"2026-08-{day:02d}"
            asset, plays, assignment_id = _resolve_arm(
                conn, config, registry, trained, date
            )
            arm = store.arm_for_running_night(assignment_id)
            seen[arm] = (asset.name, plays)
            conn.execute("UPDATE experiments SET status='done' WHERE id=?", (experiment.id,))

        assert Arm.TRAINED_CUE in seen and Arm.UNTRAINED_CUE in seen, seen
        assert seen[Arm.TRAINED_CUE] == ("trained-ascending", True)
        assert seen[Arm.UNTRAINED_CUE] == ("control-descending", True)

    def test_no_cue_arm_suppresses_audio_without_changing_the_asset(self, setup):
        config, conn, registry, trained = setup
        store = ExperimentStore(conn, config.storage.data_dir)
        experiment = store.create(
            "silent", design="two-arm", preregistration="frozen for test"
        )
        store.start(experiment.id)

        for day in range(1, 9):
            _, plays, assignment_id = _resolve_arm(
                conn, config, registry, trained, f"2026-09-{day:02d}"
            )
            if store.arm_for_running_night(assignment_id) is Arm.NO_CUE:
                assert plays is False
                return
        pytest.fail("two-arm design never produced a NO_CUE night in 8 tries")


class TestMatchedDesignIsAvailable:
    def test_registered_and_has_no_silence_arm(self):
        arms = DESIGNS["two-arm-matched"]
        assert set(arms) == {Arm.TRAINED_CUE, Arm.UNTRAINED_CUE}
        assert Arm.NO_CUE not in arms, (
            "a silence arm varies whether a sound occurred as well as whether it "
            "was conditioned, which is the confound this design removes"
        )
