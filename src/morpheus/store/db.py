"""SQLite connection management and migration application.

WAL mode is not an optimisation here, it is a durability requirement: the
recorder writes continuously for eight hours unattended, and a rollback-journal
crash mid-transaction on an overnight run costs a night of data that cannot be
re-collected.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

from .migrations import MIGRATIONS


def connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the database with the pragmas Morpheus depends on."""
    db_path = Path(db_path)
    if not read_only:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    if read_only:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(str(db_path), isolation_level=None)

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not read_only:
        conn.execute("PRAGMA journal_mode = WAL")
        # NORMAL is the right trade here: with WAL it is crash-safe against
        # process death (the case we actually face overnight) and only risks
        # the last transactions on OS-level power loss.
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations. Returns the resulting schema version."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for version, sql in sorted(MIGRATIONS):
        if version <= current:
            continue
        # The transaction must live inside the script: executescript() commits
        # any pending transaction before it runs, so an outer BEGIN would be
        # discarded and a failed migration would leave a half-built schema.
        # PRAGMA takes no parameter binding, hence the interpolation of an int.
        script = f"BEGIN;\n{sql}\nPRAGMA user_version = {int(version)};\nCOMMIT;"
        try:
            conn.executescript(script)
        except Exception:
            conn.executescript("ROLLBACK;")
            raise
        current = version
    return current


def open_db(db_path: Path) -> sqlite3.Connection:
    """Connect and bring the schema up to date. The normal entry point."""
    conn = connect(db_path)
    migrate(conn)
    return conn


def iter_rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> Iterator[sqlite3.Row]:
    cur = conn.execute(sql, params)
    try:
        while True:
            rows = cur.fetchmany(512)
            if not rows:
                return
            yield from rows
    finally:
        cur.close()
