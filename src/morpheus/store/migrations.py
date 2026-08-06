"""Versioned schema migrations, applied via PRAGMA user_version.

Only the tables M0 actually writes are created here. The remaining tables from
design.md §16 (cues, reports, experiments, assignments, ...) arrive in their own
migrations as those phases land — creating twelve empty tables now would be
speculation dressed up as planning.

The `frames_1hz` columns are the exception: the later-phase feature columns are
declared from the outset and left NULL, so that a night recorded in M0 stays
joinable against one recorded in M4 without a backfill.
"""

from __future__ import annotations

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        -- Configuration in force when a session ran. Referenced by every
        -- session so that a mid-study parameter change is visible in analysis
        -- rather than silently reinterpreting earlier nights.
        CREATE TABLE config_snapshots (
            id           INTEGER PRIMARY KEY,
            created_at   TEXT    NOT NULL,
            fingerprint  TEXT    NOT NULL UNIQUE,
            config_json  TEXT    NOT NULL,
            git_sha      TEXT,
            git_dirty    INTEGER
        );

        -- Physical setup. A camera remount changes the meaning of every
        -- geometric feature, so it must be attributable.
        CREATE TABLE device_profiles (
            id               INTEGER PRIMARY KEY,
            created_at       TEXT    NOT NULL,
            fingerprint      TEXT    NOT NULL UNIQUE,
            camera_model     TEXT,
            backend          TEXT,
            width            INTEGER,
            height           INTEGER,
            fps              REAL,
            fourcc           TEXT,
            manual_exposure  INTEGER,
            ir_wavelength_nm INTEGER,
            mount_geometry   TEXT,
            audio_device     TEXT
        );

        CREATE TABLE sessions (
            id                 INTEGER PRIMARY KEY,
            uuid               TEXT    NOT NULL UNIQUE,
            started_at_utc     TEXT    NOT NULL,
            started_at_mono    REAL    NOT NULL,
            ended_at_utc       TEXT,
            ended_at_mono      REAL,
            status             TEXT    NOT NULL,
            kind               TEXT    NOT NULL DEFAULT 'probe',
            night_index        INTEGER,
            device_profile_id  INTEGER REFERENCES device_profiles(id),
            config_snapshot_id INTEGER REFERENCES config_snapshots(id),
            morpheus_version   TEXT,
            notes              TEXT
        );

        -- One row per second. ~28,800 rows/night; trivial for SQLite.
        -- NO PIXEL DATA IS EVER STORED HERE OR ANYWHERE ELSE (design.md §20).
        CREATE TABLE frames_1hz (
            session_id             INTEGER NOT NULL REFERENCES sessions(id),
            t_mono                 REAL    NOT NULL,
            t_utc                  TEXT    NOT NULL,
            n_frames               INTEGER NOT NULL,

            -- populated in M0
            signal_quality         REAL,
            face_present           REAL,   -- fraction of the second
            eye_region_usable      REAL,   -- fraction of the second
            coverage_flag          TEXT,
            global_motion          REAL,
            bed_motion             REAL,
            face_motion            REAL,
            yaw_proxy              REAL,
            roll_deg               REAL,
            interocular_px         REAL,
            focus                  REAL,
            luminance_mean         REAL,
            scene_change           REAL,

            -- declared now, populated from M1 onward
            landmark_available     REAL,
            pitch                  REAL,
            head_motion            REAL,
            eye_flow_l             REAL,
            eye_flow_r             REAL,
            eye_flow_bilateral_corr REAL,
            lid_disp_l             REAL,
            lid_disp_r             REAL,
            resp_proxy             REAL,

            PRIMARY KEY (session_id, t_mono)
        ) WITHOUT ROWID;

        CREATE INDEX idx_frames_session_utc ON frames_1hz(session_id, t_utc);

        -- Uptime accounting. A run that quietly drops a third of its frames
        -- looks identical to a healthy one in frames_1hz; this is what makes
        -- the difference visible to the M0 acceptance criteria.
        CREATE TABLE session_health (
            session_id      INTEGER PRIMARY KEY REFERENCES sessions(id),
            frames_captured INTEGER NOT NULL DEFAULT 0,
            frames_dropped  INTEGER NOT NULL DEFAULT 0,
            read_failures   INTEGER NOT NULL DEFAULT 0,
            reconnects      INTEGER NOT NULL DEFAULT 0,
            seconds_recorded INTEGER NOT NULL DEFAULT 0,
            capture_uptime  REAL,
            clock_gaps_json TEXT,
            peak_rss_mb     REAL,
            updated_at      TEXT
        );
        """,
    ),
    (
        2,
        """
        -- M2: cueing, conditioning, and the morning report.

        -- Cue audio, hashed. The trained/untrained distinction is the basis of
        -- the experiment's control arm, so it must be a recoverable fact rather
        -- than an operator's assertion (design.md §15.1).
        CREATE TABLE cue_assets (
            id         INTEGER PRIMARY KEY,
            name       TEXT    NOT NULL,
            path       TEXT    NOT NULL,
            sha256     TEXT    NOT NULL UNIQUE,
            trained    INTEGER NOT NULL,
            samplerate INTEGER NOT NULL,
            duration_s REAL    NOT NULL,
            created_at TEXT    NOT NULL
        );

        -- Hedged observations. The kind column is constrained to the closed
        -- EventKind enum; nothing here asserts a sleep stage.
        CREATE TABLE events (
            id               INTEGER PRIMARY KEY,
            session_id       INTEGER NOT NULL REFERENCES sessions(id),
            t_mono           REAL    NOT NULL,
            t_utc            TEXT    NOT NULL,
            kind             TEXT    NOT NULL,
            confidence       REAL,
            duration_ms      REAL,
            features_json    TEXT,
            detector_version TEXT
        );
        CREATE INDEX idx_events_session ON events(session_id, t_mono);

        -- One row per cue, written BEFORE audio starts so that a crash
        -- mid-playback still leaves an attributable record (design.md §12.3).
        CREATE TABLE cues (
            id                INTEGER PRIMARY KEY,
            session_id        INTEGER NOT NULL REFERENCES sessions(id),
            t_mono            REAL    NOT NULL,
            t_utc             TEXT    NOT NULL,
            cue_asset_id      INTEGER REFERENCES cue_assets(id),
            asset_sha256      TEXT,
            gain              REAL    NOT NULL,
            gain_requested    REAL,
            ramp_ms           REAL    NOT NULL,
            duration_ms       REAL    NOT NULL,
            repetition_index  INTEGER NOT NULL DEFAULT 0,
            policy_version    TEXT,
            gate_snapshot_json TEXT,
            trigger           TEXT,
            played            INTEGER NOT NULL DEFAULT 0,
            error             TEXT
        );
        CREATE INDEX idx_cues_session ON cues(session_id, t_mono);

        -- What happened in the observation window after a cue.
        CREATE TABLE cue_outcomes (
            cue_id               INTEGER PRIMARY KEY REFERENCES cues(id),
            window_s             REAL    NOT NULL,
            outcome              TEXT    NOT NULL,
            motion_before        REAL,
            motion_after         REAL,
            motion_delta         REAL,
            latency_to_motion_ms REAL,
            quality_during       REAL,
            coverage_during      REAL
        );

        -- Pre-sleep / WBTB conditioning. Adherence is a covariate in every
        -- analysis and the most likely alternative explanation for a positive
        -- result, so it is measured rather than assumed (design.md §14).
        CREATE TABLE training_sessions (
            id                INTEGER PRIMARY KEY,
            session_id        INTEGER REFERENCES sessions(id),
            cue_asset_id      INTEGER REFERENCES cue_assets(id),
            kind              TEXT    NOT NULL,
            started_at        TEXT    NOT NULL,
            completed_at      TEXT,
            completed         INTEGER NOT NULL DEFAULT 0,
            duration_s        REAL,
            steps_json        TEXT,
            engagement_rating INTEGER,
            notes             TEXT
        );

        -- Morning report. Narrative is the primary outcome's raw material and
        -- is treated as sensitive throughout (design.md §20).
        CREATE TABLE reports (
            id                INTEGER PRIMARY KEY,
            session_id        INTEGER REFERENCES sessions(id),
            report_date       TEXT    NOT NULL UNIQUE,
            submitted_at      TEXT    NOT NULL,
            narrative         TEXT,
            lucid_binary      INTEGER,
            lucid_confidence  INTEGER,
            knew_was_dreaming INTEGER,
            cue_heard         INTEGER,
            cue_indirect      INTEGER,
            cue_woke_me       INTEGER,
            dreams_recalled   INTEGER,
            vividness         INTEGER,
            sleep_quality     INTEGER,
            awakenings        INTEGER,
            guessed_condition TEXT,
            notes             TEXT
        );
        CREATE INDEX idx_reports_session ON reports(session_id);
        """,
    ),
    (
        3,
        """
        -- M3-M6: experiments, validation, and the adaptive layer.

        CREATE TABLE experiments (
            id                INTEGER PRIMARY KEY,
            name              TEXT    NOT NULL UNIQUE,
            seed              INTEGER NOT NULL,
            design            TEXT    NOT NULL,
            repeats_per_block INTEGER NOT NULL,
            plan_fingerprint  TEXT    NOT NULL,
            preregistration   TEXT,
            prereg_sha256     TEXT,
            created_at        TEXT    NOT NULL,
            started_at        TEXT,
            ended_at          TEXT,
            status            TEXT    NOT NULL DEFAULT 'draft'
        );

        -- One row per night. The arm is stored obfuscated so that browsing the
        -- database does not reveal it; see experiment/blinding.py for why that
        -- is deliberately not called encryption.
        CREATE TABLE assignments (
            id             INTEGER PRIMARY KEY,
            experiment_id  INTEGER NOT NULL REFERENCES experiments(id),
            night_index    INTEGER NOT NULL,
            assigned_date  TEXT,
            arm_sealed     TEXT    NOT NULL,
            block_index    INTEGER NOT NULL,
            session_id     INTEGER REFERENCES sessions(id),
            revealed_at    TEXT,
            UNIQUE (experiment_id, night_index)
        );
        CREATE UNIQUE INDEX idx_assignment_date
            ON assignments(experiment_id, assigned_date)
            WHERE assigned_date IS NOT NULL;

        -- Every unsealing, including illegitimate ones. Blinding cannot be
        -- enforced against the person holding the machine, so the design goal
        -- is that breaking it is *visible* rather than impossible.
        CREATE TABLE reveal_audit (
            id            INTEGER PRIMARY KEY,
            assignment_id INTEGER NOT NULL REFERENCES assignments(id),
            revealed_at   TEXT    NOT NULL,
            reason        TEXT    NOT NULL,
            legitimate    INTEGER NOT NULL,
            report_exists INTEGER NOT NULL
        );

        -- M4: the go/no-go record for H1. G9 (eye-movement-timed cueing) reads
        -- this table and refuses to activate without a passing row.
        CREATE TABLE validation_results (
            id                INTEGER PRIMARY KEY,
            created_at        TEXT    NOT NULL,
            hypothesis        TEXT    NOT NULL,
            reference_source  TEXT    NOT NULL,
            nights_used       INTEGER NOT NULL,
            held_out_nights   INTEGER NOT NULL,
            auc               REAL,
            auc_ci_low        REAL,
            auc_ci_high       REAL,
            brier             REAL,
            threshold         REAL    NOT NULL,
            passed            INTEGER NOT NULL,
            model_version     TEXT,
            metrics_json      TEXT,
            notes             TEXT
        );

        -- M5: persisted bandit posteriors, so learning survives a restart.
        CREATE TABLE policy_state (
            id            INTEGER PRIMARY KEY,
            policy_name   TEXT    NOT NULL,
            arm_key       TEXT    NOT NULL,
            successes     REAL    NOT NULL DEFAULT 0,
            failures      REAL    NOT NULL DEFAULT 0,
            pulls         INTEGER NOT NULL DEFAULT 0,
            updated_at    TEXT    NOT NULL,
            UNIQUE (policy_name, arm_key)
        );

        -- M5: what the incumbent policy would have chosen, recorded alongside
        -- what was actually chosen. Without this there is no way to tell
        -- whether the bandit beat the heuristic it replaced.
        CREATE TABLE counterfactuals (
            id             INTEGER PRIMARY KEY,
            cue_id         INTEGER REFERENCES cues(id),
            t_mono         REAL    NOT NULL,
            chosen_policy  TEXT    NOT NULL,
            chosen_arm     TEXT    NOT NULL,
            baseline_arm   TEXT,
            baseline_policy TEXT,
            agreed         INTEGER,
            context_json   TEXT
        );
        """,
    ),
    (
        4,
        """
        -- Waking calibration (design.md §13.1). The positive_control_auc column
        -- is the M1 go/no-go: deliberate closed-eye saccades versus closed-eye
        -- stillness, measured in one session on one face under one lighting.
        CREATE TABLE calibration_profiles (
            id                   INTEGER PRIMARY KEY,
            created_at           TEXT    NOT NULL,
            device_profile_id    INTEGER REFERENCES device_profiles(id),
            positive_control_auc REAL,
            head_turn_leakage    REAL,
            baseline_median      REAL,
            baseline_mad         REAL,
            suggested_threshold  REAL,
            passed               INTEGER NOT NULL DEFAULT 0,
            segments_json        TEXT,
            posture_json         TEXT,
            notes_json           TEXT
        );

        -- Audio loudness calibration (design.md §13.3), by ascending limits.
        -- Absolute SPL at the pillow is unknown without a meter, so these are
        -- digital gains anchored to the user's own judgement.
        CREATE TABLE audio_calibrations (
            id             INTEGER PRIMARY KEY,
            created_at     TEXT    NOT NULL,
            cue_asset_id   INTEGER REFERENCES cue_assets(id),
            faintest_gain  REAL,
            comfortable_gain REAL,
            ceiling_gain   REAL    NOT NULL,
            output_device  TEXT,
            notes          TEXT
        );
        """,
    ),
]

SCHEMA_VERSION = max(v for v, _ in MIGRATIONS)
