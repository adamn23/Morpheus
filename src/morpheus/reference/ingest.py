"""Loading reference sleep staging from an external device.

The reference is what turns "the camera sees something" into a claim that can be
checked. Without it, any eye-movement detector is unfalsifiable — which is why
G9 (eye-movement-timed cueing) is hard-locked behind a passing validation record
and why this module exists before that one.

Supports hypnogram CSVs, which is the common denominator across devices: Muse
via MindMonitor, ZMax, YASA output, and hand-scored files all export or convert
to (timestamp, stage). EDF is deliberately not parsed here — it would pull in a
heavy dependency to handle a format every one of those tools can already export
out of.

Stage labels are normalised to a small vocabulary and then collapsed to the only
distinction Morpheus needs: REM versus other. Note the naming: this is the *one*
place the word REM is legitimate, because here it refers to a measurement made
by an instrument qualified to make it, not to an inference of ours.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

EPOCH_SECONDS = 30.0

# Every spelling seen across MindMonitor, YASA, ZMax and manual scoring.
_STAGE_ALIASES: dict[str, str] = {
    "w": "W", "wake": "W", "awake": "W", "0": "W",
    "n1": "N1", "1": "N1", "s1": "N1", "light": "N1",
    "n2": "N2", "2": "N2", "s2": "N2",
    "n3": "N3", "3": "N3", "s3": "N3", "s4": "N3", "4": "N3",
    "sws": "N3", "deep": "N3",
    "r": "R", "rem": "R", "5": "R",
    "?": "UNKNOWN", "": "UNKNOWN", "unknown": "UNKNOWN",
    "art": "UNKNOWN", "artifact": "UNKNOWN", "movement": "UNKNOWN",
}

_TIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
)


@dataclass(frozen=True)
class ReferenceEpoch:
    t_utc: float          # epoch start, seconds since the Unix epoch
    stage: str            # W | N1 | N2 | N3 | R | UNKNOWN
    source: str = ""

    @property
    def reference_scored_rem(self) -> bool:
        """True when the *reference device* scored this epoch as REM.

        Named for what it is: an external instrument's measurement, which
        Morpheus is entitled to record but not to make. The verbosity is the
        point — it stays distinguishable from anything Morpheus concluded, and
        it does not need an exemption from the naming lint.
        """
        return self.stage == "R"

    @property
    def is_scored(self) -> bool:
        return self.stage != "UNKNOWN"

    @property
    def is_sleep(self) -> bool:
        return self.stage in {"N1", "N2", "N3", "R"}


class ReferenceError(RuntimeError):
    pass


def normalise_stage(raw: str) -> str:
    return _STAGE_ALIASES.get(str(raw).strip().lower(), "UNKNOWN")


def parse_timestamp(raw: str) -> Optional[float]:
    text = str(raw).strip()
    if not text:
        return None
    # Bare numbers are already Unix seconds (or milliseconds, which some
    # exporters emit without saying so).
    try:
        value = float(text)
        return value / 1000.0 if value > 1e11 else value
    except ValueError:
        pass
    for fmt in _TIME_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return None


def _find_column(header: list[str], candidates: Iterable[str]) -> Optional[int]:
    lowered = [h.strip().lower() for h in header]
    for candidate in candidates:
        if candidate in lowered:
            return lowered.index(candidate)
    for index, name in enumerate(lowered):
        if any(candidate in name for candidate in candidates):
            return index
    return None


def load_hypnogram_csv(
    path: Path,
    *,
    epoch_seconds: float = EPOCH_SECONDS,
    start_utc: Optional[float] = None,
) -> list[ReferenceEpoch]:
    """Read a hypnogram CSV into epochs.

    Accepts either an explicit timestamp column or a bare stage column, in which
    case epochs are assumed contiguous from `start_utc` — the shape YASA and
    several scoring tools produce.
    """
    path = Path(path)
    if not path.exists():
        raise ReferenceError(f"reference file not found: {path}")

    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ReferenceError(f"reference file is empty: {path}")

    header, data = rows[0], rows[1:]
    time_col = _find_column(header, ("timestamp", "time", "onset", "start", "datetime", "epoch"))
    # naming-lint: allow — these are column names in other tools' exports.
    stage_col = _find_column(
        header, ("stage", "sleep_stage", "label", "hypnogram", "score")  # naming-lint: allow
    )

    if stage_col is None:
        # Headerless single-column hypnogram.
        if len(header) == 1:
            data = rows
            stage_col, time_col = 0, None
        else:
            raise ReferenceError(
                f"could not find a stage column in {path.name}. "
                f"Columns seen: {header}"
            )

    epochs: list[ReferenceEpoch] = []
    fallback = start_utc
    for index, row in enumerate(data):
        if not row or stage_col >= len(row):
            continue
        stage = normalise_stage(row[stage_col])

        t_utc: Optional[float] = None
        if time_col is not None and time_col < len(row):
            t_utc = parse_timestamp(row[time_col])
        if t_utc is None:
            if fallback is None:
                raise ReferenceError(
                    f"{path.name} has no usable timestamps. Pass start_utc so epochs "
                    f"can be placed on the clock, or export with a timestamp column."
                )
            t_utc = fallback + index * epoch_seconds

        epochs.append(ReferenceEpoch(t_utc=t_utc, stage=stage, source=path.name))

    if not epochs:
        raise ReferenceError(f"no epochs parsed from {path}")
    epochs.sort(key=lambda e: e.t_utc)
    return epochs


def summarise(epochs: list[ReferenceEpoch]) -> dict:
    scored = [e for e in epochs if e.is_scored]
    counts: dict[str, int] = {}
    for epoch in epochs:
        counts[epoch.stage] = counts.get(epoch.stage, 0) + 1
    rem = sum(1 for e in scored if e.reference_scored_rem)
    sleep = sum(1 for e in scored if e.is_sleep)
    span = (epochs[-1].t_utc - epochs[0].t_utc) / 3600.0 if len(epochs) > 1 else 0.0
    return {
        "epochs": len(epochs),
        "scored": len(scored),
        "hours": span,
        "stage_counts": counts,
        "rem_epochs": rem,
        "rem_fraction_of_sleep": (rem / sleep) if sleep else None,
    }
