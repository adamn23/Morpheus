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
