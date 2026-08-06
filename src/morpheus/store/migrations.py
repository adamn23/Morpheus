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
]

SCHEMA_VERSION = max(v for v, _ in MIGRATIONS)
