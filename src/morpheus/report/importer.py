"""Import an existing dream journal from a notes-app export.

Sixty days of prior journalling is worth more than the next two weeks of
collection: it is a pre-intervention baseline that already exists, and it cannot
be re-created if it is mangled on the way in.

So the design is preview-first. Parsing someone's freeform notes is inherently
guesswork — dates live in filenames or headings or frontmatter, and lucidity is
tagged however that person happened to tag it. The importer detects candidates,
shows exactly what it found, and refuses to write anything until asked. A silent
mis-parse of a journal that cannot be regenerated is the failure worth designing
against.

One honest limitation, worth stating where it will be read: entries written
before this project existed were not written against the pinned outcome wording
(report/schema.py). Coding them retrospectively introduces measurement
inconsistency. The bias runs conservative — over-coding past entries as lucid
inflates the baseline and makes any later intervention look *worse* — so an
imperfect import is still useful, but the imported nights should be treated as a
prior, not as data collected under the protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text"}

# Date forms seen in notes-app exports, in descending order of confidence.
_DATE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("iso", re.compile(r"(?<!\d)(\d{4})[-_./](\d{1,2})[-_./](\d{1,2})(?!\d)")),
    ("dmy", re.compile(r"(?<!\d)(\d{1,2})[-_./](\d{1,2})[-_./](\d{4})(?!\d)")),
    (
        "long",
        re.compile(
            r"\b(\d{1,2})?\s*(January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\w*\.?\s+(\d{1,2})?[,\s]*(\d{4})\b",
            re.IGNORECASE,
        ),
    ),
]

_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"], start=1
    )
}

# Ways people mark a lucid dream. All are searched; the preview reports which
# actually matched, so an unusual convention shows up as "nothing found" rather
# than as a quietly empty baseline.
LUCID_PATTERNS: dict[str, re.Pattern] = {
    "hashtag": re.compile(r"#lucid\b", re.IGNORECASE),
    "bracket": re.compile(r"[\[(]\s*lucid\s*[\])]", re.IGNORECASE),
    "field_yes": re.compile(r"^\s*lucid\s*[:=]\s*(yes|true|y|1|x)\b", re.IGNORECASE | re.MULTILINE),
    "field_no": re.compile(r"^\s*lucid\s*[:=]\s*(no|false|n|0)\b", re.IGNORECASE | re.MULTILINE),
    "standalone": re.compile(r"^\s*lucid[!.]?\s*$", re.IGNORECASE | re.MULTILINE),
    "checkbox": re.compile(r"^\s*[-*]\s*\[x\]\s*lucid\b", re.IGNORECASE | re.MULTILINE),
}
_POSITIVE = ("hashtag", "bracket", "field_yes", "standalone", "checkbox")

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

#: Inline date headers: "June 30:", "Jul 1 -", "December 25:".
#: A running journal kept as continuous prose marks days this way rather than
#: with Markdown headings, and almost never writes the year — it is obvious to
#: the author and invisible to a parser.
_INLINE_DATE = re.compile(
    r"^[ \t]*"
    # Optional weekday prefix: "Tuesday Feb. 4 2025 - ..."
    r"(?:(?:Mon|Tues?|Wed(?:nes)?|Thur?s?|Fri|Sat(?:ur)?|Sun)(?:day)?\.?,?[ \t]+)?"
    r"(January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)"
    r"\.?[ \t]+(\d{1,2})(?:st|nd|rd|th)?"
    # Optional explicit year, with or without a comma before it.
    r"[ \t]*,?[ \t]*(\d{4})?"
    r"[ \t]*[:\-\u2013]",
    re.IGNORECASE | re.MULTILINE,
)

#: A backwards month step this large is taken as a real year boundary
#: (December to January is -11). Anything smaller is treated as an
#: out-of-order entry or a typo, because in a journal written most days a
#: genuine twelve-month gap between consecutive entries is far less likely
#: than a slip of the pen — and the cost of guessing wrong is every
#: subsequent entry landing a year out.
_ROLLOVER_MIN_DECREASE = 6


@dataclass
class ParsedEntry:
    entry_date: Optional[str]
    narrative: str
    lucid: Optional[bool]
    lucid_evidence: list[str] = field(default_factory=list)
    date_source: str = ""
    source_file: str = ""
    dreams_recalled: Optional[int] = None

    @property
    def usable(self) -> bool:
        return self.entry_date is not None and bool(self.narrative.strip())


@dataclass
class ImportPreview:
    entries: list[ParsedEntry] = field(default_factory=list)
    undated: list[ParsedEntry] = field(default_factory=list)
    duplicates: dict[str, int] = field(default_factory=dict)
    files_scanned: int = 0
    pattern_hits: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> list[ParsedEntry]:
        return [e for e in self.entries if e.usable]

    @property
    def lucid_count(self) -> int:
        return sum(1 for e in self.usable if e.lucid)

    @property
    def date_range(self) -> Optional[tuple[str, str]]:
        dates = sorted(e.entry_date for e in self.usable if e.entry_date)
        return (dates[0], dates[-1]) if dates else None

    @property
    def lucid_rate(self) -> Optional[float]:
        scored = [e for e in self.usable if e.lucid is not None]
        return (sum(1 for e in scored if e.lucid) / len(scored)) if scored else None


def _normalise_date(raw: tuple, kind: str) -> Optional[str]:
    try:
        if kind == "iso":
            y, m, d = int(raw[0]), int(raw[1]), int(raw[2])
        elif kind == "dmy":
            first, second, y = int(raw[0]), int(raw[1]), int(raw[2])
            # Ambiguous by nature. Prefer D/M when the first field cannot be a
            # month, otherwise fall back to M/D and accept the uncertainty —
            # the preview shows parsed dates so a systematic error is visible.
            d, m = (first, second) if first > 12 else (second, first)
        else:
            # "6 August 2026" or "August 6, 2026" — the day sits on either side
            # of the month name, so whichever group matched is the day.
            day = raw[0] or raw[2]
            month = _month_lookup(raw[1])
            if not day or month is None:
                return None
            d, m, y = int(day), month, int(raw[3])
        return date(y, m, d).isoformat()
    except (ValueError, KeyError, IndexError):
        return None


def _find_date(text: str) -> tuple[Optional[str], str]:
    for kind, pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            iso = _normalise_date(match.groups(), kind)
            if iso:
                return iso, kind
    return None, ""


def _month_lookup(name: str) -> Optional[int]:
    key = name.lower()
    for full, index in _MONTHS.items():
        if full.startswith(key[:3]):
            return index
    return None


def detect_lucid(text: str) -> tuple[Optional[bool], list[str]]:
    """Return (lucid, evidence). None means no marker of either kind was found.

    An explicit negative marker wins over a positive one: someone who writes
    "Lucid: no" in a template every night, and "#lucid" only when it happened,
    would otherwise be scored lucid on every entry that contains the template.
    """
    evidence = [name for name, pattern in LUCID_PATTERNS.items() if pattern.search(text)]
    if "field_no" in evidence and not any(
        p in evidence for p in ("hashtag", "bracket", "standalone", "checkbox")
    ):
        return False, evidence
    if any(p in evidence for p in _POSITIVE):
        return True, evidence
    if "field_no" in evidence:
        return False, evidence
    return None, evidence


def parse_text(text: str, *, fallback_date: Optional[str] = None, source: str = "") -> ParsedEntry:
    frontmatter = _FRONTMATTER.match(text)
    body = text[frontmatter.end():] if frontmatter else text

    entry_date, source_kind = (None, "")
    if frontmatter:
        entry_date, source_kind = _find_date(frontmatter.group(1))
        if entry_date:
            source_kind = f"frontmatter/{source_kind}"

    if not entry_date:
        head = "\n".join(body.splitlines()[:3])
        entry_date, source_kind = _find_date(head)
        if entry_date:
            source_kind = f"heading/{source_kind}"

    if not entry_date and fallback_date:
        entry_date, source_kind = fallback_date, "filename"

    lucid, evidence = detect_lucid(text)
    return ParsedEntry(
        entry_date=entry_date,
        narrative=body.strip(),
        lucid=lucid,
        lucid_evidence=evidence,
        date_source=source_kind,
        source_file=source,
    )


def split_dated_prose(
    text: str, base_year: int
) -> tuple[list[tuple[str, str]], list[str]]:
    """Split a continuous prose journal on inline date headers.

    Returns ((iso_date, body) pairs, warnings). Anything before the first
    header — a title, a "Prev: June 10-29" note — is discarded.

    Handles both shapes seen in real journals: a bare "June 30:" with the year
    implied, and a fuller "Tuesday Feb. 4 2025 -" with the year written out. An
    explicit year always wins; otherwise the year carries forward and rolls
    over when the month drops far enough to look like a real December-January
    boundary.

    Small backwards steps are treated as typos rather than year boundaries, and
    both they and genuine rollovers are reported. A real journal contained
    "June 7:" between "July 6:" and "July 8:", which under a naive rule shifted
    thirteen entries into the following year — silently, because every date
    after it was individually plausible.
    """
    matches = list(_INLINE_DATE.finditer(text))
    if len(matches) < 2:
        return [], []

    out: list[tuple[str, str]] = []
    warnings: list[str] = []
    year = base_year
    previous_month = 0

    for index, match in enumerate(matches):
        month = _month_lookup(match.group(1))
        day = int(match.group(2))
        explicit_year = int(match.group(3)) if match.group(3) else None
        if month is None:
            continue

        if explicit_year is not None:
            candidate_year = explicit_year
        elif previous_month and month < previous_month:
            decrease = previous_month - month
            if decrease >= _ROLLOVER_MIN_DECREASE:
                candidate_year = year + 1
                warnings.append(
                    f"year rolled to {candidate_year} at "
                    f"{match.group(0).strip()} (month went back {decrease})"
                )
            else:
                candidate_year = year
                warnings.append(
                    f"out of order: {match.group(0).strip()} follows month "
                    f"{previous_month:02d}. Kept in {year} as a likely typo — "
                    f"check it."
                )
        else:
            candidate_year = year

        # Validate before committing state. An unparseable "February 31" must
        # not advance anything, or one typo shifts every later entry.
        try:
            iso = date(candidate_year, month, day).isoformat()
        except ValueError:
            warnings.append(f"unparseable date skipped: {match.group(0).strip()}")
            continue
        year, previous_month = candidate_year, month

        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if not body:
            continue

        # Consecutive headers with the same date are two dreams from one night
        # written under repeated headings rather than as paragraphs. Merging
        # them keeps both; report_date is unique, so leaving them separate
        # would silently drop the first.
        if out and out[-1][0] == iso:
            out[-1] = (iso, out[-1][1] + "\n\n" + body)
        else:
            out.append((iso, body))
    return out, warnings


#: Lines that mark a boundary between dreams rather than being a dream.
#: A journal that writes "Wake up" between two dreams of the same night would
#: otherwise have that line counted as a third dream.
_SEPARATOR_LINE = re.compile(
    r"^\W*(wake up|woke up|awake|awoke|back to sleep|fell back asleep|"
    r"second dream|next dream|later|---+|\*\*\*+)\W*$",
    re.IGNORECASE,
)


def count_dreams(body: str) -> int:
    """Separate dreams in one night's entry, counted as paragraphs.

    A night with three recalled dreams is written as three paragraphs under one
    date. That is worth capturing: dreams_recalled is a secondary outcome, and
    deriving it costs nothing where the alternative is leaving it null.

    Paragraphs that are only a separator are excluded. A real journal used a
    bare "Wake up" line between dreams fifteen times, and counting those as
    dreams inflated the total by fifteen.
    """
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", body)
        if p.strip() and not _SEPARATOR_LINE.match(p.strip())
    ]
    return max(1, len(paragraphs))


def _split_combined(text: str) -> list[str]:
    """Split a single file into entries on dated Markdown headings."""
    lines = text.splitlines()
    starts = [
        i for i, line in enumerate(lines)
        if re.match(r"^#{1,3}\s", line) and _find_date(line)[0]
    ]
    if len(starts) < 2:
        return [text]
    chunks = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        chunks.append("\n".join(lines[start:end]))
    return chunks


def scan(path: Path, *, base_year: Optional[int] = None) -> ImportPreview:
    """Parse a file or directory into a preview. Writes nothing.

    `base_year` enables the prose-journal path, where dates are inline and the
    year is never written.
    """
    path = Path(path)
    preview = ImportPreview()

    files: list[Path]
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in TEXT_SUFFIXES)
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(path)

    raw_entries: list[ParsedEntry] = []
    for file in files:
        preview.files_scanned += 1
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            continue

        filename_date, _ = _find_date(file.stem)

        # A running prose journal marks days inline ("June 30:") rather than
        # with headings, and omits the year. Detected first, because the
        # heading-based splitter cannot see those dates at all and would import
        # the entire journal as one undated blob.
        prose, prose_warnings = (
            split_dated_prose(text, base_year) if base_year else ([], [])
        )
        preview.warnings.extend(f"{file.name}: {w}" for w in prose_warnings)
        if prose:
            for iso, body in prose:
                lucid, evidence = detect_lucid(body)
                raw_entries.append(
                    ParsedEntry(
                        entry_date=iso,
                        narrative=body,
                        lucid=lucid,
                        lucid_evidence=evidence,
                        date_source="inline prose header",
                        source_file=file.name,
                        dreams_recalled=count_dreams(body),
                    )
                )
            continue

        # A file containing several dated headings holds several entries — the
        # common shape when a notes app exports a whole journal as one document.
        # A per-day export instead gets its date from the filename.
        chunks = _split_combined(text)
        if len(chunks) == 1:
            raw_entries.append(parse_text(text, fallback_date=filename_date, source=file.name))
        else:
            for chunk in chunks:
                raw_entries.append(
                    parse_text(chunk, fallback_date=filename_date, source=file.name)
                )

    for entry in raw_entries:
        for name in entry.lucid_evidence:
            preview.pattern_hits[name] = preview.pattern_hits.get(name, 0) + 1
        (preview.entries if entry.usable else preview.undated).append(entry)

    seen: dict[str, int] = {}
    for entry in preview.usable:
        assert entry.entry_date
        seen[entry.entry_date] = seen.get(entry.entry_date, 0) + 1
    preview.duplicates = {d: n for d, n in seen.items() if n > 1}
    return preview


def format_preview(preview: ImportPreview, limit: int = 3) -> str:
    lines: list[str] = []
    add = lines.append

    add("Journal import preview")
    add("=" * 68)
    add(f"  files scanned     {preview.files_scanned}")
    add(f"  entries parsed    {len(preview.usable)}")
    if preview.undated:
        add(f"  skipped (no date) {len(preview.undated)}")
    span = preview.date_range
    if span:
        add(f"  date range        {span[0]} to {span[1]}")
    add("")

    add("Lucidity")
    add("-" * 68)
    if preview.pattern_hits:
        for name, count in sorted(preview.pattern_hits.items(), key=lambda kv: -kv[1]):
            add(f"  {name:<14} matched {count} entries")
    else:
        add("  no lucidity markers matched any known pattern")
        add("  (entries will import with lucidity unset)")
    scored = [e for e in preview.usable if e.lucid is not None]
    unscored = len(preview.usable) - len(scored)
    add(f"  marked lucid      {preview.lucid_count}")
    add(f"  scored entries    {len(scored)} of {len(preview.usable)}")
    if preview.lucid_rate is not None:
        add(f"  baseline rate     {preview.lucid_rate * 100:.1f}% of scored nights "
            f"({preview.lucid_rate * 7:.2f}/week)")
    add("")

    if unscored:
        # This ambiguity changes the baseline substantially and only the author
        # of the journal can resolve it, so it is surfaced rather than guessed.
        both = len(preview.usable)
        as_lucid = preview.lucid_count / both
        add(f"  {unscored} entries carry no lucidity marker either way. That is")
        add("  ambiguous, and it moves the baseline a long way:")
        add(f"    if untagged means NOT lucid : {as_lucid * 100:.1f}% "
            f"({as_lucid * 7:.2f}/week)")
        add(f"    if untagged means unknown   : {preview.lucid_rate * 100:.1f}% "
            f"({preview.lucid_rate * 7:.2f}/week)" if preview.lucid_rate is not None else "")
        add("  If you only ever tagged the lucid ones, pass --untagged not-lucid.")
        add("  If you sometimes forgot to record either way, leave the default,")
        add("  which imports them unscored and excludes them from the rate.")
        add("")

    if preview.warnings:
        add("Warnings — check these before importing")
        add("-" * 68)
        for warning in preview.warnings[:20]:
            for line in _wrap(warning, 66):
                add(f"  ! {line}")
        if len(preview.warnings) > 20:
            add(f"  ... and {len(preview.warnings) - 20} more")
        add("")

    if preview.duplicates:
        add("Duplicate dates (later entries would overwrite earlier ones)")
        add("-" * 68)
        for day, count in sorted(preview.duplicates.items())[:10]:
            add(f"  {day}  x{count}")
        add("")

    add(f"Sample of {min(limit, len(preview.usable))} parsed entries")
    add("-" * 68)
    for entry in preview.usable[:limit]:
        flag = "LUCID" if entry.lucid else ("not lucid" if entry.lucid is False else "unscored")
        add(f"  {entry.entry_date}  [{flag}]  via {entry.date_source}  ({entry.source_file})")
        snippet = " ".join(entry.narrative.split())[:100]
        add(f"    {snippet}{'...' if len(snippet) == 100 else ''}")
    add("")
    add("Check the dates and the lucid counts against what you know is true")
    add("before importing. A mis-parse of a journal you cannot regenerate is")
    add("the one failure worth being slow about.")
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
