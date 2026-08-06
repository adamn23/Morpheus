"""Experiment lifecycle and the blinding gate.

The rule this module exists to enforce: **an arm cannot be revealed for a night
until that night's morning report has been submitted.** Everything else is
bookkeeping.

That ordering is what makes the report an observation rather than a
rationalisation. Knowing you were in the trained-cue arm before writing down
whether you were lucid contaminates the primary outcome in a way no amount of
later analysis can undo.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .blinding import BlindingError, load_or_create_key, seal, unseal
from .randomization import Arm, RandomizationPlan, make_plan

log = logging.getLogger("morpheus.experiment")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ExperimentRecord:
    id: int
    name: str
    seed: int
    design: str
    repeats_per_block: int
    plan_fingerprint: str
    status: str

    def plan(self) -> RandomizationPlan:
        return make_plan(
            seed=self.seed, design=self.design, repeats_per_block=self.repeats_per_block
        )


class ExperimentStore:
    def __init__(self, conn: sqlite3.Connection, data_dir: Path) -> None:
        self._conn = conn
        self._key = load_or_create_key(Path(data_dir))

    # ------------------------------------------------------------ lifecycle

    def create(
        self,
        name: str,
        *,
        design: str = "two-arm",
        seed: Optional[int] = None,
        repeats_per_block: int = 2,
        preregistration: Optional[str] = None,
    ) -> ExperimentRecord:
        plan = make_plan(seed=seed, design=design, repeats_per_block=repeats_per_block)
        prereg_hash = (
            hashlib.sha256(preregistration.encode()).hexdigest() if preregistration else None
        )
        cur = self._conn.execute(
            "INSERT INTO experiments (name, seed, design, repeats_per_block, "
            "plan_fingerprint, preregistration, prereg_sha256, created_at, status) "
            "VALUES (?,?,?,?,?,?,?,?,'draft')",
            (
                name, plan.seed, design, repeats_per_block, plan.fingerprint(),
                preregistration, prereg_hash, _utc(),
            ),
        )
        return self.get(int(cur.lastrowid))

    def start(self, experiment_id: int) -> None:
        """Begin collection. Freezes the pre-registration.

        A pre-registration that can be edited after data exists is not a
        pre-registration, so the hash recorded at creation is what later
        analysis is checked against.
        """
        row = self._conn.execute(
            "SELECT preregistration, prereg_sha256 FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        if not row["preregistration"]:
            raise ValueError(
                "cannot start an experiment without a pre-registration. "
                "Run `morpheus prereg` first — deciding the analysis after seeing "
                "the data is the failure mode the whole harness exists to prevent."
            )
        self._conn.execute(
            "UPDATE experiments SET status = 'running', started_at = ? WHERE id = ?",
            (_utc(), experiment_id),
        )

    def get(self, experiment_id: int) -> ExperimentRecord:
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no experiment {experiment_id}")
        return self._to_record(row)

    def active(self) -> Optional[ExperimentRecord]:
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE status = 'running' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return self._to_record(row) if row else None

    def list(self) -> list[ExperimentRecord]:
        return [
            self._to_record(r)
            for r in self._conn.execute("SELECT * FROM experiments ORDER BY id")
        ]

    def prereg_intact(self, experiment_id: int) -> bool:
        row = self._conn.execute(
            "SELECT preregistration, prereg_sha256 FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None or not row["preregistration"]:
            return False
        current = hashlib.sha256(row["preregistration"].encode()).hexdigest()
        return current == row["prereg_sha256"]

    # ----------------------------------------------------------- assignment

    def assign_night(
        self, experiment: ExperimentRecord, assigned_date: str, *, session_id: Optional[int] = None
    ) -> int:
        """Seal tonight's arm. Returns the assignment id, never the arm.

        Deliberately does not return the condition: a caller that wants it must
        go through `arm_for_running_night`, which exists separately so that the
        one legitimate blind read is easy to find and audit.
        """
        existing = self._conn.execute(
            "SELECT id FROM assignments WHERE experiment_id = ? AND assigned_date = ?",
            (experiment.id, assigned_date),
        ).fetchone()
        if existing:
            return int(existing["id"])

        row = self._conn.execute(
            "SELECT COALESCE(MAX(night_index), 0) AS n FROM assignments WHERE experiment_id = ?",
            (experiment.id,),
        ).fetchone()
        night_index = int(row["n"]) + 1

        assignment = experiment.plan().assignment_for(night_index)
        sealed = seal(self._key, self._nonce(experiment.id, night_index), assignment.arm.value)

        cur = self._conn.execute(
            "INSERT INTO assignments (experiment_id, night_index, assigned_date, "
            "arm_sealed, block_index, session_id) VALUES (?,?,?,?,?,?)",
            (experiment.id, night_index, assigned_date, sealed, assignment.block_index, session_id),
        )
        return int(cur.lastrowid)

    def arm_for_running_night(self, assignment_id: int) -> Arm:
        """Unseal for the daemon, without marking the night as revealed.

        The cue engine must know whether to play audio. That read is legitimate
        and is not an unblinding, because nothing is shown to the user — so it
        is audited as machine-read rather than counted against the blind.
        """
        row = self._require(assignment_id)
        arm = Arm(unseal(self._key, self._nonce(row["experiment_id"], row["night_index"]), row["arm_sealed"]))
        self._audit(assignment_id, "daemon read for cue decision", legitimate=True, report_exists=False)
        return arm

    def reveal(self, assignment_id: int, *, force: bool = False) -> Arm:
        """Unseal for the user. Refuses until the morning report exists.

        `force` exists because a hard block would eventually be worked around by
        editing the database, which would leave no trace at all. An escape hatch
        that records itself is better than a wall that gets climbed silently.
        """
        row = self._require(assignment_id)
        report_exists = self._report_exists(row["assigned_date"])

        if not report_exists and not force:
            self._audit(
                assignment_id, "blocked: no morning report yet",
                legitimate=False, report_exists=False,
            )
            raise BlindingError(
                f"no morning report for {row['assigned_date']}. The arm stays sealed until "
                f"the report is submitted — knowing the condition first contaminates the "
                f"primary outcome. Run `morpheus journal` first."
            )

        arm = Arm(unseal(self._key, self._nonce(row["experiment_id"], row["night_index"]), row["arm_sealed"]))
        self._audit(
            assignment_id,
            "revealed after report" if report_exists else "FORCED reveal before report",
            legitimate=report_exists,
            report_exists=report_exists,
        )
        self._conn.execute(
            "UPDATE assignments SET revealed_at = COALESCE(revealed_at, ?) WHERE id = ?",
            (_utc(), assignment_id),
        )
        if not report_exists:
            log.warning(
                "assignment %s was force-revealed before its report — this night is "
                "unblinded and should be excluded from the primary analysis",
                assignment_id,
            )
        return arm

    def assignment_for_date(self, experiment_id: int, assigned_date: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM assignments WHERE experiment_id = ? AND assigned_date = ?",
            (experiment_id, assigned_date),
        ).fetchone()

    def revealed_arms(self, experiment_id: int) -> list[tuple[str, Arm]]:
        """Dates and arms for nights that have been legitimately revealed.

        This is the analysis input. Unrevealed nights are absent by design — the
        analysis cannot see an arm the participant has not yet earned the right
        to see, which stops a mid-study peek disguised as "just running numbers".
        """
        rows = self._conn.execute(
            "SELECT * FROM assignments WHERE experiment_id = ? AND revealed_at IS NOT NULL "
            "ORDER BY night_index",
            (experiment_id,),
        ).fetchall()
        out = []
        for row in rows:
            arm = Arm(unseal(self._key, self._nonce(row["experiment_id"], row["night_index"]), row["arm_sealed"]))
            out.append((row["assigned_date"], arm))
        return out

    def audit_log(self, experiment_id: Optional[int] = None) -> list[sqlite3.Row]:
        if experiment_id is None:
            return self._conn.execute(
                "SELECT * FROM reveal_audit ORDER BY id DESC"
            ).fetchall()
        return self._conn.execute(
            "SELECT r.* FROM reveal_audit r JOIN assignments a ON a.id = r.assignment_id "
            "WHERE a.experiment_id = ? ORDER BY r.id DESC",
            (experiment_id,),
        ).fetchall()

    def blinding_integrity(self, experiment_id: int) -> dict:
        """How often the blind was broken early. Reported with every analysis."""
        rows = self.audit_log(experiment_id)
        forced = [r for r in rows if not r["legitimate"] and r["reason"].startswith("FORCED")]
        blocked = [r for r in rows if r["reason"].startswith("blocked")]
        assigned = self._conn.execute(
            "SELECT COUNT(*) FROM assignments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()[0]
        return {
            "nights_assigned": assigned,
            "forced_reveals": len(forced),
            "blocked_attempts": len(blocked),
            "forced_dates": [r["revealed_at"] for r in forced],
        }

    # ------------------------------------------------------------ internals

    @staticmethod
    def _nonce(experiment_id: int, night_index: int) -> str:
        return f"exp{experiment_id}/night{night_index}"

    def _require(self, assignment_id: int) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM assignments WHERE id = ?", (assignment_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no assignment {assignment_id}")
        return row

    def _report_exists(self, assigned_date: Optional[str]) -> bool:
        if not assigned_date:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM reports WHERE report_date = ? AND submitted_at IS NOT NULL",
            (assigned_date,),
        ).fetchone()
        return row is not None

    def _audit(
        self, assignment_id: int, reason: str, *, legitimate: bool, report_exists: bool
    ) -> None:
        self._conn.execute(
            "INSERT INTO reveal_audit (assignment_id, revealed_at, reason, legitimate, "
            "report_exists) VALUES (?,?,?,?,?)",
            (assignment_id, _utc(), reason, int(legitimate), int(report_exists)),
        )

    @staticmethod
    def _to_record(row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            id=int(row["id"]),
            name=row["name"],
            seed=int(row["seed"]),
            design=row["design"],
            repeats_per_block=int(row["repeats_per_block"]),
            plan_fingerprint=row["plan_fingerprint"],
            status=row["status"],
        )
