"""The M0 coverage report.

This produces the single number that decides the shape of the whole project:
what fraction of a night the eye region is actually available for analysis
(design.md §28). The thresholds it judges against were fixed in the design
document before any data existed, precisely so they cannot be renegotiated
after seeing the answer.

The breakdown by *reason* matters as much as the headline. A night lost to
`face_absent` argues for remounting the camera; one lost to `pose_unsuitable`
argues that the posture itself is the blocker and no amount of remounting will
help; one lost to `too_small` argues for a longer lens or a closer mount.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

# Pre-committed decision gate (design.md §28). Do not tune these to taste.
GATE_PASS = 0.25
GATE_FAIL = 0.15


@dataclass
class CoverageReport:
    session_id: int
    uuid: str
    started_at: str
    ended_at: Optional[str]
    status: str
    kind: str
    night_index: Optional[int]

    seconds_recorded: int
    wall_duration_s: Optional[float]
    frames_captured: int
    frames_dropped: int
    capture_uptime: Optional[float]
    read_failures: int
    reconnects: int
    clock_gaps: list[tuple[float, float]]
    peak_rss_mb: Optional[float]

    face_present_frac: float
    eye_usable_frac: float
    mean_quality: float
    mean_global_motion: float
    median_interocular_px: Optional[float]

    flag_seconds: dict[str, int] = field(default_factory=dict)
    posture_seconds: dict[str, int] = field(default_factory=dict)
    scene_change_events: int = 0

    @property
    def verdict(self) -> str:
        if self.eye_usable_frac >= GATE_PASS:
            return "PASS"
        if self.eye_usable_frac < GATE_FAIL:
            return "FAIL"
        return "MARGINAL"

    @property
    def verdict_detail(self) -> str:
        if self.verdict == "PASS":
            return (
                "Eye-region coverage clears the gate. The eye-movement branch is worth "
                "developing in M1 — in shadow mode, with no influence on cue timing."
            )
        if self.verdict == "FAIL":
            return (
                "Eye-region coverage is below the abandon threshold. Per design.md §23 "
                "failure condition 2, the camera cannot support eye-based detection for "
                "this sleeper. Proceed to M2 (scheduled TLR) with the camera as a motion "
                "guard only, and record H1 as unreachable. This is a valid result, not a "
                "setback: it was measured in two weeks rather than assumed for four months."
            )
        return (
            "Coverage sits between the abandon and proceed thresholds. Do not tune the "
            "thresholds. Collect more nights, and read the reason breakdown below: it "
            "says whether remounting the camera could plausibly move the number."
        )


def analyse_session(conn: sqlite3.Connection, session_id: int) -> CoverageReport:
    session = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if session is None:
        raise ValueError(f"no session with id {session_id}")
    health = conn.execute(
        "SELECT * FROM session_health WHERE session_id = ?", (session_id,)
    ).fetchone()

    agg = conn.execute(
        """
        SELECT COUNT(*)                AS seconds,
               AVG(face_present)       AS face_frac,
               AVG(eye_region_usable)  AS eye_frac,
               AVG(signal_quality)     AS quality,
               AVG(global_motion)      AS motion,
               MIN(t_mono)             AS t_first,
               MAX(t_mono)             AS t_last
        FROM frames_1hz WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()

    flags = {
        row["coverage_flag"]: row["n"]
        for row in conn.execute(
            "SELECT coverage_flag, COUNT(*) AS n FROM frames_1hz "
            "WHERE session_id = ? GROUP BY coverage_flag ORDER BY n DESC",
            (session_id,),
        )
    }

    # Posture buckets over seconds where a face was seen at all. The thresholds
    # mirror CoverageConfig.max_abs_yaw_proxy so the buckets and the gate agree.
    posture = {
        row["bucket"]: row["n"]
        for row in conn.execute(
            """
            SELECT CASE
                     WHEN yaw_proxy IS NULL       THEN 'no_face'
                     WHEN ABS(yaw_proxy) <= 0.15  THEN 'frontal'
                     WHEN ABS(yaw_proxy) <= 0.35  THEN 'moderate_turn'
                     ELSE 'turned_away'
                   END AS bucket,
                   COUNT(*) AS n
            FROM frames_1hz WHERE session_id = ? GROUP BY bucket ORDER BY n DESC
            """,
            (session_id,),
        )
    }

    interocular = conn.execute(
        "SELECT interocular_px FROM frames_1hz "
        "WHERE session_id = ? AND interocular_px IS NOT NULL ORDER BY interocular_px",
        (session_id,),
    ).fetchall()
    median_io = (
        float(interocular[len(interocular) // 2]["interocular_px"]) if interocular else None
    )

    scene_changes = conn.execute(
        "SELECT COUNT(*) FROM frames_1hz WHERE session_id = ? AND scene_change > 0.35",
        (session_id,),
    ).fetchone()[0]

    gaps: list[tuple[float, float]] = []
    if health is not None and health["clock_gaps_json"]:
        gaps = [tuple(g) for g in json.loads(health["clock_gaps_json"])]

    wall = None
    if agg["t_first"] is not None and agg["t_last"] is not None:
        wall = float(agg["t_last"]) - float(agg["t_first"])

    return CoverageReport(
        session_id=session_id,
        uuid=session["uuid"],
        started_at=session["started_at_utc"],
        ended_at=session["ended_at_utc"],
        status=session["status"],
        kind=session["kind"],
        night_index=session["night_index"],
        seconds_recorded=int(agg["seconds"] or 0),
        wall_duration_s=wall,
        frames_captured=int(health["frames_captured"]) if health else 0,
        frames_dropped=int(health["frames_dropped"]) if health else 0,
        capture_uptime=float(health["capture_uptime"]) if health and health["capture_uptime"] is not None else None,
        read_failures=int(health["read_failures"]) if health else 0,
        reconnects=int(health["reconnects"]) if health else 0,
        clock_gaps=gaps,
        peak_rss_mb=float(health["peak_rss_mb"]) if health and health["peak_rss_mb"] is not None else None,
        face_present_frac=float(agg["face_frac"] or 0.0),
        eye_usable_frac=float(agg["eye_frac"] or 0.0),
        mean_quality=float(agg["quality"] or 0.0),
        mean_global_motion=float(agg["motion"] or 0.0),
        median_interocular_px=median_io,
        flag_seconds=flags,
        posture_seconds=posture,
        scene_change_events=int(scene_changes),
    )


def format_report(r: CoverageReport) -> str:
    """Plain-text rendering, intended to be read at 07:00 without coffee."""
    lines: list[str] = []
    add = lines.append

    add(f"Morpheus M0 coverage report — session {r.session_id} (night {r.night_index})")
    add("=" * 72)
    add(f"  uuid          {r.uuid}")
    add(f"  started       {r.started_at}")
    add(f"  ended         {r.ended_at or '(still running or aborted)'}")
    add(f"  status        {r.status}")
    add("")

    add("Run health")
    add("-" * 72)
    hours = (r.wall_duration_s or 0) / 3600.0
    add(f"  recorded          {r.seconds_recorded:,} s  ({hours:.2f} h)")
    add(f"  frames captured   {r.frames_captured:,}")
    add(f"  frames dropped    {r.frames_dropped:,} (estimated)")
    if r.capture_uptime is not None:
        flag = "" if r.capture_uptime >= 0.95 else "   <-- below the 95% M0 criterion"
        add(f"  capture uptime    {r.capture_uptime * 100:.1f}%{flag}")
    add(f"  read failures     {r.read_failures}")
    add(f"  reconnects        {r.reconnects}")
    if r.peak_rss_mb is not None:
        add(f"  peak RSS          {r.peak_rss_mb:.0f} MB")
    if r.clock_gaps:
        add(f"  clock gaps        {len(r.clock_gaps)}   <-- machine slept or process stalled")
        for start, end in r.clock_gaps[:5]:
            add(f"                      {end - start:.1f} s gap")
    else:
        add("  clock gaps        none")
    if r.scene_change_events:
        add(f"  scene changes     {r.scene_change_events} s   <-- camera possibly bumped")
    add("")

    add("Signal")
    add("-" * 72)
    add(f"  mean quality      {r.mean_quality:.3f}")
    add(f"  mean motion       {r.mean_global_motion:.5f}")
    add(f"  face present      {r.face_present_frac * 100:.1f}% of recorded time")
    if r.median_interocular_px is not None:
        add(f"  median interocular {r.median_interocular_px:.1f} px")
    add("")

    add("Coverage by reason (seconds)")
    add("-" * 72)
    total = max(1, r.seconds_recorded)
    for flag, n in sorted(r.flag_seconds.items(), key=lambda kv: -kv[1]):
        add(f"  {flag:<20} {n:>8,}  {n / total * 100:5.1f}%")
    add("")

    add("Posture (seconds)")
    add("-" * 72)
    for bucket, n in sorted(r.posture_seconds.items(), key=lambda kv: -kv[1]):
        add(f"  {bucket:<20} {n:>8,}  {n / total * 100:5.1f}%")
    add("")

    add("DECISION GATE — eye-region usable coverage")
    add("=" * 72)
    add(f"  measured          {r.eye_usable_frac * 100:.1f}%")
    add(f"  pass threshold    {GATE_PASS * 100:.0f}%       abandon threshold {GATE_FAIL * 100:.0f}%")
    add(f"  verdict           {r.verdict}")
    add("")
    for line in _wrap(r.verdict_detail, 70):
        add(f"  {line}")
    add("")
    add("  Reminder: this measures visibility, not eye movement. A passing score")
    add("  means the eye region can be seen, not that eyelid motion is detectable")
    add("  through it. That is the separate M1 question, and it is the harder one.")
    return "\n".join(lines)


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


def list_sessions(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT s.id, s.uuid, s.night_index, s.started_at_utc, s.status, s.kind,
               COUNT(f.t_mono)        AS seconds,
               AVG(f.eye_region_usable) AS eye_frac,
               AVG(f.face_present)      AS face_frac
        FROM sessions s LEFT JOIN frames_1hz f ON f.session_id = s.id
        GROUP BY s.id ORDER BY s.id DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
