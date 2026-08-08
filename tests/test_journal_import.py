"""Journal import and dream-sign mining.

The importer handles data that cannot be regenerated, so these tests lean
heavily on the failure directions: wrong dates, missed tags, and silent
mis-scoring. A parser that quietly gets 60 nights slightly wrong is worse than
one that refuses to run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from morpheus.analysis.dream_signs import extract
from morpheus.report.importer import detect_lucid, parse_text, scan
from morpheus.report.schema import MorningReport, ReportStore


def write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n")
    return path


# ------------------------------------------------------------- date handling


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("2026-06-01.md", "2026-06-01"),
        ("2026_06_01.md", "2026-06-01"),
        ("2026.06.01.txt", "2026-06-01"),
    ],
)
def test_date_from_filename(tmp_path: Path, filename: str, expected: str) -> None:
    write(tmp_path, filename, "A dream about nothing in particular.")
    preview = scan(tmp_path)
    assert [e.entry_date for e in preview.usable] == [expected]


def test_date_from_frontmatter_wins_over_filename(tmp_path: Path) -> None:
    write(tmp_path, "untitled-3.md", "---\ndate: 2026-06-09\n---\nA dream.")
    preview = scan(tmp_path)
    assert preview.usable[0].entry_date == "2026-06-09"
    assert "frontmatter" in preview.usable[0].date_source


def test_long_form_dates_parse(tmp_path: Path) -> None:
    write(tmp_path, "combined.md", "## June 6, 2026\nA dream.\n\n## 7 July 2026\nAnother.")
    dates = sorted(e.entry_date for e in scan(tmp_path).usable)
    assert dates == ["2026-06-06", "2026-07-07"]


def test_undated_entries_are_reported_not_guessed(tmp_path: Path) -> None:
    """Inventing a date would silently misplace an entry in the timeline."""
    write(tmp_path, "notes.md", "Something happened but I did not write when.")
    preview = scan(tmp_path)
    assert preview.usable == []
    assert len(preview.undated) == 1


def test_duplicate_dates_are_surfaced(tmp_path: Path) -> None:
    write(tmp_path, "a/2026-06-01.md", "First.")
    write(tmp_path, "b/2026-06-01.md", "Second.")
    assert scan(tmp_path).duplicates == {"2026-06-01": 2}


# ---------------------------------------------------------- lucidity tagging


@pytest.mark.parametrize(
    "text",
    ["#lucid\nI knew.", "[lucid] I knew.", "Lucid: yes\nI knew.",
     "lucid\nI knew.", "- [x] lucid\nI knew."],
)
def test_positive_lucid_markers(text: str) -> None:
    lucid, _ = detect_lucid(text)
    assert lucid is True


@pytest.mark.parametrize("text", ["Lucid: no\nOrdinary dream.", "lucid = false\nNope."])
def test_negative_lucid_markers(text: str) -> None:
    lucid, _ = detect_lucid(text)
    assert lucid is False


def test_absent_marker_is_none_not_false() -> None:
    """Unknown must stay unknown.

    Scoring an unmarked entry as not-lucid would deflate the baseline, making
    any later intervention look better than it is. Whether untagged means
    not-lucid is the user's call, made explicitly at import time.
    """
    lucid, evidence = detect_lucid("A dream with no annotation at all.")
    assert lucid is None
    assert evidence == []


def test_positive_tag_beats_template_negative() -> None:
    """A template with 'Lucid: no' plus a '#lucid' tag means it happened.

    Someone whose nightly template contains the negative field, and who adds a
    hashtag on the nights it occurred, would otherwise have every lucid night
    scored as non-lucid.
    """
    lucid, _ = detect_lucid("Lucid: no\n...actually #lucid, I realised near the end.")
    assert lucid is True


def test_lucid_rate_uses_scored_entries_only(tmp_path: Path) -> None:
    write(tmp_path, "2026-06-01.md", "#lucid\nOne.")
    write(tmp_path, "2026-06-02.md", "Lucid: no\nTwo.")
    write(tmp_path, "2026-06-03.md", "No marker at all.")
    preview = scan(tmp_path)
    assert preview.lucid_count == 1
    assert preview.lucid_rate == pytest.approx(0.5)  # 1 of 2 scored, not 1 of 3


# ------------------------------------------------------------------ splitting


def test_combined_file_splits_on_dated_headings(tmp_path: Path) -> None:
    write(
        tmp_path, "journal.md",
        "## 2026-06-01\nFirst dream.\n\n## 2026-06-02\n#lucid\nSecond dream.\n\n"
        "## 2026-06-03\nThird dream.",
    )
    preview = scan(tmp_path)
    assert len(preview.usable) == 3
    assert preview.lucid_count == 1
    assert "Second" in [e for e in preview.usable if e.lucid][0].narrative


def test_single_entry_file_is_not_split(tmp_path: Path) -> None:
    write(tmp_path, "2026-06-01.md", "# A title\nOne dream, several. Sentences. Here.")
    assert len(scan(tmp_path).usable) == 1


def test_narrative_excludes_frontmatter(tmp_path: Path) -> None:
    write(tmp_path, "2026-06-01.md", "---\ndate: 2026-06-01\ntags: x\n---\nThe dream itself.")
    assert scan(tmp_path).usable[0].narrative == "The dream itself."


def test_empty_and_binary_files_are_skipped(tmp_path: Path) -> None:
    write(tmp_path, "2026-06-01.md", "A real entry.")
    (tmp_path / "empty.md").write_text("")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
    preview = scan(tmp_path)
    assert len(preview.usable) == 1


# ---------------------------------------------------------------- dream signs


def test_recurring_motifs_are_found() -> None:
    narratives = [
        "Walking through the old house. The staircase went sideways.",
        "The old house again, and my brother was there.",
        "Back at the old house. Brother on the staircase.",
        "A beach, completely unrelated.",
    ]
    terms = [s.term for s in extract(narratives, min_nights=3, top_n=10)]
    assert "old house" in terms
    assert "beach" not in terms


def test_bigrams_do_not_cross_sentence_boundaries() -> None:
    """'...the old house. The staircase...' must not yield 'house staircase'."""
    narratives = ["The old house. The staircase creaked."] * 4
    terms = [s.term for s in extract(narratives, min_nights=3, top_n=20)]
    assert "house staircase" not in terms
    assert "old house" in terms


def test_lucidity_tag_is_not_mined_as_a_motif() -> None:
    """The tag lives in the narrative; without filtering it ranks near the top."""
    narratives = ["#lucid I was in a garden."] * 5
    terms = [s.term for s in extract(narratives, min_nights=3, top_n=20)]
    assert "lucid" not in terms
    assert "garden" in terms


def test_ranking_is_by_nights_not_raw_mentions() -> None:
    """One vivid dream repeating a word is not a recurring motif."""
    narratives = [
        "Tiger tiger tiger tiger tiger tiger.",
        "A quiet room.",
        "A quiet room.",
        "A quiet room.",
    ]
    signs = extract(narratives, min_nights=3, top_n=5)
    assert signs and signs[0].term in {"quiet room", "quiet", "room"}
    assert all(s.term != "tiger" for s in signs)


def test_no_narratives_returns_empty() -> None:
    assert extract([], min_nights=2) == []
    assert extract(["", "   "], min_nights=1) == []


# --------------------------------- scoring prose entries that carry no tag


def test_suggestive_phrases_are_detected() -> None:
    """A journal kept in prose has no tags, so wording is the only handle."""
    from morpheus.report.review import SUGGESTIVE

    def hints(text):
        return [label for pattern, label in SUGGESTIVE if pattern.search(text)]

    assert hints("Suddenly I realised I was dreaming.")
    assert hints("I knew I was dreaming and stayed calm.")
    assert hints("did a reality check, nose pinch worked")
    assert hints("became fully lucid in the corridor")
    assert hints("a false awakening, then I woke properly")
    assert not hints("A long dream about a train station.")


def test_unscored_returns_only_unscored_entries(conn) -> None:
    from morpheus.report.review import unscored

    store = ReportStore(conn)
    store.submit(MorningReport(report_date="2026-06-01", narrative="A plain dream."))
    store.submit(MorningReport(report_date="2026-06-02", narrative="I realised I was dreaming.",
                               lucid_binary=True, dreams_recalled=1))
    store.submit(MorningReport(report_date="2026-06-03", narrative="Another plain one."))

    dates = [c.report_date for c in unscored(conn)]
    assert "2026-06-02" not in dates
    assert set(dates) == {"2026-06-01", "2026-06-03"}


def test_suggestive_entries_are_ordered_first(conn) -> None:
    """Ordering is a convenience. Every entry still has to be answered."""
    from morpheus.report.review import unscored

    store = ReportStore(conn)
    store.submit(MorningReport(report_date="2026-06-01", narrative="A plain dream."))
    store.submit(MorningReport(report_date="2026-06-02", narrative="Then I knew I was dreaming."))

    candidates = unscored(conn)
    assert candidates[0].report_date == "2026-06-02"
    assert candidates[0].suggestive
    assert not candidates[1].suggestive


def test_scoring_is_never_automatic(conn) -> None:
    """The matcher must not write anything. Only an explicit call scores.

    Classifying prose automatically would fabricate the project's primary
    outcome, which is the one number everything else exists to move.
    """
    from morpheus.report.review import score, unscored

    store = ReportStore(conn)
    store.submit(MorningReport(report_date="2026-06-01",
                               narrative="I realised I was dreaming and it held."))
    assert unscored(conn)[0].suggestive
    assert conn.execute("SELECT lucid_binary FROM reports").fetchone()[0] is None

    score(conn, "2026-06-01", True)
    assert conn.execute("SELECT lucid_binary FROM reports").fetchone()[0] == 1
    assert unscored(conn) == []


def test_progress_counts(conn) -> None:
    from morpheus.report.review import progress, score

    store = ReportStore(conn)
    for day in range(1, 6):
        store.submit(MorningReport(report_date=f"2026-06-0{day}", narrative=f"dream {day}"))
    score(conn, "2026-06-01", True)
    score(conn, "2026-06-02", False)

    stats = progress(conn)
    assert stats == {"total": 5, "scored": 2, "remaining": 3, "lucid": 1}


# ------------------------------- continuous prose journals with inline dates


PROSE = """Dream Journal Part 2

Prev: June 10-29

June 30: Basically I dont remember much of this one, something about a game.

Basically me and a friend are on a raft at night.

Basically I am in a parking lot.

July 1: Basically I am in some sort of house, it feels very small.

Basically im standing on a road.

July 2: Basically im in toronto at the eaton centre.
"""


def test_prose_journal_needs_a_year(tmp_path: Path) -> None:
    """Inline dates like 'June 30:' carry no year, so scan must be told one.

    Without it the whole journal imports as a single undated blob, which is
    what happened on the first attempt at a real one.
    """
    write(tmp_path, "journal.md", PROSE)
    assert scan(tmp_path).usable == []
    assert len(scan(tmp_path, base_year=2025).usable) == 3


def test_prose_journal_splits_on_inline_dates(tmp_path: Path) -> None:
    write(tmp_path, "journal.md", PROSE)
    entries = scan(tmp_path, base_year=2025).usable
    assert [e.entry_date for e in entries] == ["2025-06-30", "2025-07-01", "2025-07-02"]
    assert all(e.date_source == "inline prose header" for e in entries)


def test_prose_preamble_is_discarded(tmp_path: Path) -> None:
    """The title and a 'Prev: June 10-29' note are not a dream."""
    write(tmp_path, "journal.md", PROSE)
    entries = scan(tmp_path, base_year=2025).usable
    assert not any("Dream Journal Part 2" in e.narrative for e in entries)
    assert not any("Prev:" in e.narrative for e in entries)


def test_paragraphs_become_the_recalled_dream_count(tmp_path: Path) -> None:
    """Several dreams in one night are written as several paragraphs."""
    write(tmp_path, "journal.md", PROSE)
    entries = {e.entry_date: e for e in scan(tmp_path, base_year=2025).usable}
    assert entries["2025-06-30"].dreams_recalled == 3
    assert entries["2025-07-01"].dreams_recalled == 2
    assert entries["2025-07-02"].dreams_recalled == 1


def test_year_rolls_forward_when_the_month_goes_backwards() -> None:
    """A journal running June to February spans two calendar years silently."""
    from morpheus.report.importer import split_dated_prose

    text = "Dec 20: winter entry.\n\nJan 3: new year entry.\n\nFeb 1: later.\n"
    entries, _ = split_dated_prose(text, 2025)
    assert [d for d, _ in entries] == ["2025-12-20", "2026-01-03", "2026-02-01"]


def test_abbreviated_months_and_dash_separators() -> None:
    from morpheus.report.importer import split_dated_prose

    text = "Jun 30 - first entry.\n\nJul 1: second entry.\n\nSept 4 – third.\n"
    entries, _ = split_dated_prose(text, 2025)
    assert [d for d, _ in entries] == ["2025-06-30", "2025-07-01", "2025-09-04"]


def test_impossible_dates_are_skipped_not_guessed() -> None:
    from morpheus.report.importer import split_dated_prose

    text = "June 30: real.\n\nFebruary 31: impossible.\n\nJuly 2: real again.\n"
    entries, warnings = split_dated_prose(text, 2025)
    assert [d for d, _ in entries] == ["2025-06-30", "2025-07-02"]
    assert any("unparseable" in w for w in warnings)


def test_a_single_inline_date_is_not_treated_as_prose() -> None:
    """One match is more likely a sentence than a journal structure."""
    from morpheus.report.importer import split_dated_prose

    assert split_dated_prose("I met her on June 30: it was raining.", 2025)[0] == []


def test_prose_dates_do_not_break_tagged_journals(tmp_path: Path) -> None:
    """The heading-based path must still work when a year IS present."""
    write(tmp_path, "j.md", "## 2026-06-01\n#lucid\nOne.\n\n## 2026-06-02\nTwo.")
    preview = scan(tmp_path, base_year=2025)
    assert [e.entry_date for e in preview.usable] == ["2026-06-01", "2026-06-02"]
    assert preview.lucid_count == 1


def test_weekday_prefix_and_explicit_year(tmp_path: Path) -> None:
    """A real journal switched format partway through the same file.

    The earlier section was 'Tuesday Feb. 4 2025 - ...' with a weekday prefix
    and the year written out; the later section was a bare 'June 10:'. The
    original pattern required the month at line start with no year, so eleven
    entries were silently dropped.
    """
    from morpheus.report.importer import split_dated_prose

    text = (
        "Tuesday Feb. 4 2025 - first.\n\n"
        "Wednesday February 6, 2025: second.\n\n"
        "Feb. 10th, 2025 - third.\n\n"
        "June 10: later section, no year.\n"
    )
    entries, _ = split_dated_prose(text, 2026)
    assert [d for d, _ in entries] == [
        "2025-02-04", "2025-02-06", "2025-02-10", "2025-06-10",
    ]


def test_explicit_year_overrides_the_carried_year() -> None:
    from morpheus.report.importer import split_dated_prose

    entries, _ = split_dated_prose("Mar 1, 2024: a.\n\nApr 2, 2024: b.\n", 2026)
    assert [d for d, _ in entries] == ["2024-03-01", "2024-04-02"]


def test_small_backwards_step_is_a_typo_not_a_year_boundary() -> None:
    """The failure that would have misdated thirteen real entries.

    'June 7:' sat between 'July 6:' and 'July 8:' — plainly a slip for July 7.
    A naive rule rolled the year and shifted everything after it into 2027,
    silently, because each following date was individually plausible.
    """
    from morpheus.report.importer import split_dated_prose

    entries, warnings = split_dated_prose("July 6: a.\n\nJune 7: b.\n\nJuly 8: c.\n", 2026)
    assert [d for d, _ in entries] == ["2026-07-06", "2026-06-07", "2026-07-08"]
    assert any("out of order" in w for w in warnings)


def test_december_to_january_still_rolls_over() -> None:
    from morpheus.report.importer import split_dated_prose

    entries, warnings = split_dated_prose("Dec 28: a.\n\nJan 2: b.\n", 2026)
    assert [d for d, _ in entries] == ["2026-12-28", "2027-01-02"]
    assert any("rolled to 2027" in w for w in warnings)


def test_repeated_date_headers_merge_into_one_night() -> None:
    """Two dreams written under two identical headers, not as paragraphs.

    report_date is unique, so leaving them separate silently drops the first.
    """
    from morpheus.report.importer import count_dreams, split_dated_prose

    entries, _ = split_dated_prose(
        "June 10: first dream.\n\nJune 10: second dream.\n\nJune 11: next night.\n", 2026
    )
    assert [d for d, _ in entries] == ["2026-06-10", "2026-06-11"]
    assert count_dreams(entries[0][1]) == 2


def test_baseline_stats_distinguish_unscored_from_zero(conn) -> None:
    """An unscored journal must not read as a zero lucid rate.

    Imported entries carry no lucidity value unless they were tagged, so a
    freshly imported journal has nothing scored. Reporting that as a rate of
    zero would be presenting absent data as a finding — and it would be the
    baseline the whole N-of-1 comparison is measured against.
    """
    store = ReportStore(conn)
    for day in range(1, 11):
        store.submit(MorningReport(report_date=f"2026-06-{day:02d}", narrative="a dream"))

    stats = store.baseline_stats()
    assert stats["nights"] == 10
    assert stats["nights_scored"] == 0
    assert stats["lucid_rate_per_night"] is None, "must be None, never 0.0"
    assert stats["lucid_per_week"] is None


def test_separator_lines_are_not_counted_as_dreams() -> None:
    """A real journal wrote a bare 'Wake up' between two dreams of one night.

    Counting that line as a third dream inflated the total by fifteen across
    the journal.
    """
    from morpheus.report.importer import count_dreams

    assert count_dreams("First dream.\n\nWake up\n\nSecond dream.") == 2
    assert count_dreams("One.\n\nwoke up\n\nTwo.\n\n---\n\nThree.") == 3
    assert count_dreams("One.\n\nTwo.\n\nThree.") == 3
    assert count_dreams("Only one dream here.") == 1


def test_an_entry_of_only_separators_still_counts_as_one() -> None:
    """Never return zero: the night had an entry, so it had a dream."""
    from morpheus.report.importer import count_dreams

    assert count_dreams("Wake up\n\nwoke up") == 1
