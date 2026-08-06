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


@dataclass
class ParsedEntry:
    entry_date: Optional[str]
    narrative: str
    lucid: Optional[bool]
    lucid_evidence: list[str] = field(default_factory=list)
    date_source: str = ""
    source_file: str = ""

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


def scan(path: Path) -> ImportPreview:
    """Parse a file or directory into a preview. Writes nothing."""
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
