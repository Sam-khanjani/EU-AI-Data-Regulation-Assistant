"""Parsing a EUR-Lex PDF via its own bookmark outline.

The unit tests below cover the rules that clean and slice the text. The class at the bottom
runs against the real downloaded PDF and is the one that matters most: it asserts the parse
still produces the structure this act is known to have, which is what makes an
outline-driven parser trustworthy rather than merely plausible.
"""

from __future__ import annotations

import pytest

from euaia.config import settings
from euaia.ingest.document import ParsedUnit
from euaia.ingest.pdf_outline import classify, clean_title
from euaia.ingest.pdf_parser import (
    _MARKER_LINE,
    _RUNNING_HEADER,
    MAX_MISSING_FRACTION,
    SOFT_HYPHEN,
    PdfParseError,
    _join,
    _Line,
    _locate,
    _split_paragraphs,
    parse,
)

PDF_NAME = "02024R1689-20260727.ENG.pdf"

# Parsing the real 151-page PDF takes about 15 seconds, so the classes below share one parse
# rather than each building their own.
_needs_pdf = pytest.mark.skipif(
    not (settings.raw_data_dir / PDF_NAME).exists(),
    reason="cached PDF not present; run `python -m euaia.ingest.pipeline --fetch-pdf` first",
)


@pytest.fixture(scope="module")
def doc():
    return parse((settings.raw_data_dir / PDF_NAME).read_bytes())


class TestTitleCleaning:
    def test_no_break_space_becomes_a_space(self):
        # 'Regulation\x92(EU)' must not collapse to 'Regulation(EU)'.
        assert clean_title("Regulation\x92(EU) 2024/1689") == "Regulation (EU) 2024/1689"

    def test_non_breaking_hyphen_artefact_is_dropped(self):
        assert clean_title("\x84high-\x84risk AI systems") == "high-risk AI systems"

    def test_whitespace_is_collapsed(self):
        assert clean_title("CHAPTER   I\n\nGENERAL  PROVISIONS") == "CHAPTER I GENERAL PROVISIONS"


class TestClassify:
    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("Article 6 Classification rules", ("article", "6", "Classification rules")),
            ("Article 4a Processing of data", ("article", "4a", "Processing of data")),
            ("CHAPTER III HIGH-RISK AI SYSTEMS", ("chapter", "III", "HIGH-RISK AI SYSTEMS")),
            ("SECTION 1 Classification", ("section", "1", "Classification")),
            ("ANNEX III High-risk AI systems", ("annex", "III", "High-risk AI systems")),
            ("ANNEX XIV", ("annex", "XIV", None)),
            ("Amended by:", ("other", None, None)),
        ],
    )
    def test_splits_number_from_heading(self, title, expected):
        assert classify(title) == expected

    def test_trailing_apostrophe_is_stripped(self):
        # A few bookmark titles carry one: "Article 1 Subject matter'".
        assert classify("Article 1 Subject matter'")[2] == "Subject matter"


class TestNoiseRemoval:
    @pytest.mark.parametrize("line", ["▼B", "▼M1", "► M1", "▼ B"])
    def test_marker_only_lines_are_recognised(self, line):
        assert _MARKER_LINE.match(line)

    @pytest.mark.parametrize("line", ["▼B is prohibited", "1. The ▼M1 provision"])
    def test_text_containing_a_marker_is_not_a_marker_line(self, line):
        assert not _MARKER_LINE.match(line)

    def test_running_header_is_recognised(self):
        assert _RUNNING_HEADER.match("02024R1689 — EN — 27.07.2026 — 001.001 — 61")

    def test_body_text_is_not_mistaken_for_a_header(self):
        assert not _RUNNING_HEADER.match("1. An AI system shall be considered high-risk")


class TestHyphenationRepair:
    def test_word_split_across_lines_is_rejoined(self):
        # This is the case verify/normalize.py cannot fix: it discards the soft hyphen and
        # then collapses the newline to a space, giving 'cumu lative'.
        lines = [_Line(f"the cumu{SOFT_HYPHEN}", 1), _Line("lative amount", 1)]
        assert _join(lines) == "the cumulative amount"

    def test_ordinary_lines_keep_their_break(self):
        lines = [_Line("Article 6", 1), _Line("Classification rules", 1)]
        assert _join(lines) == "Article 6\nClassification rules"


class TestParagraphSplitting:
    @staticmethod
    def _article() -> ParsedUnit:
        return ParsedUnit(
            unit_type="article",
            unit_path="CH_I/ART_6",
            text="",
            ordinal=1,
            unit_number="6",
        )

    def test_numbered_paragraphs_become_units(self):
        body = [
            _Line("Article 6", 17),
            _Line("Classification rules", 17),
            _Line("1. An AI system shall be considered high-risk where", 17),
            _Line("both conditions are fulfilled.", 17),
            _Line("2. Paragraph 1 shall not apply where the system", 18),
        ]
        counter = iter(range(100, 200))
        units = _split_paragraphs(body, self._article(), lambda: next(counter))

        assert [u.unit_path for u in units] == ["CH_I/ART_6/PAR_1", "CH_I/ART_6/PAR_2"]
        assert [u.unit_number for u in units] == ["6(1)", "6(2)"]
        assert units[0].parent_path == "CH_I/ART_6"
        # The paragraph's own text runs to the start of the next one, not to the line end.
        assert "both conditions are fulfilled." in units[0].text
        assert "Paragraph 1" not in units[0].text

    def test_page_is_recorded_from_the_first_line(self):
        body = [_Line("1. First", 17), _Line("2. Second", 18)]
        counter = iter(range(100, 200))
        units = _split_paragraphs(body, self._article(), lambda: next(counter))
        assert [u.page for u in units] == [17, 18]

    def test_article_without_numbered_paragraphs_yields_none(self):
        # The article unit already carries the full text.
        body = [_Line("Article 4", 11), _Line("Providers shall ensure literacy.", 11)]
        counter = iter(range(100, 200))
        assert _split_paragraphs(body, self._article(), lambda: next(counter)) == []

    def test_lettered_points_do_not_start_a_paragraph(self):
        body = [
            _Line("1. The following apply:", 17),
            _Line("(a) the first condition", 17),
            _Line("(b) the second condition", 17),
        ]
        counter = iter(range(100, 200))
        units = _split_paragraphs(body, self._article(), lambda: next(counter))
        assert len(units) == 1
        assert "(b) the second condition" in units[0].text


class TestHeadingLocation:
    @staticmethod
    def _entry(number: str, page: int, unit_type: str = "article"):
        from euaia.ingest.pdf_outline import OutlineEntry

        return OutlineEntry(
            level=3,
            unit_type=unit_type,
            unit_number=number,
            heading="x",
            page=page,
            title=f"Article {number} x",
        )

    def test_finds_the_heading_on_the_declared_page(self):
        lines = [_Line("preamble", 16), _Line("Article 6", 17), _Line("body", 17)]
        assert _locate(lines, self._entry("6", 17), 0) == 1

    def test_allows_one_page_of_slack_for_a_boundary_heading(self):
        # The outline's page and the text layer's page can disagree by one when a heading
        # sits at the very top or bottom of a page.
        lines = [_Line("body", 17), _Line("Article 6", 18)]
        assert _locate(lines, self._entry("6", 17), 0) == 1

    def test_does_not_search_far_past_the_declared_page(self):
        lines = [_Line("body", 17), _Line("Article 6", 25)]
        assert _locate(lines, self._entry("6", 17), 0) is None

    def test_the_scan_stays_monotonic(self):
        # `after` prevents a heading matching before the previous unit's, which would slice
        # the bodies in the wrong order.
        lines = [_Line("Article 6", 17), _Line("Article 6", 17)]
        assert _locate(lines, self._entry("6", 17), 1) == 1


@_needs_pdf
class TestRealDocument:
    """Runs only when the real PDF has been downloaded.

    The expected counts were established during development by comparing against EUR-Lex's
    own structured edition of the same act. That comparison is not repeated here -- these are
    plain regression values now, and their job is to fail if a future document or a change to
    the parser moves them.
    """

    def test_expected_structure_counts(self, doc):
        counts = {
            kind: len(doc.by_type(kind)) for kind in ("article", "chapter", "section", "annex")
        }
        assert counts == {"article": 119, "chapter": 13, "section": 16, "annex": 14}

    def test_expected_paragraph_count(self, doc):
        # EUR-Lex's structured edition yields 553. The one difference is Article 10(5),
        # which this PDF omits because the amendment replaced it -- see
        # TestKnownSourceDisagreement below.
        assert len(doc.by_type("paragraph")) == 552

    def test_paths_use_the_shared_scheme(self, doc):
        by_number = {a.unit_number: a for a in doc.by_type("article")}
        assert by_number["6"].unit_path == "CH_III/SEC_1/ART_6"
        assert by_number["5"].unit_path == "CH_II/ART_5"
        assert by_number["4a"].unit_path == "CH_I/ART_4a"

    def test_headings_survive_the_codepage(self, doc):
        by_number = {a.unit_number: a for a in doc.by_type("article")}
        assert by_number["5"].heading == "Prohibited AI practices"
        assert by_number["6"].heading == "Classification rules for high-risk AI systems"

    def test_amendment_inserted_articles_are_present(self, doc):
        numbers = {a.unit_number for a in doc.by_type("article")}
        assert {"4a", "60a", "75a", "75b", "75c", "75d"} <= numbers

    def test_pages_are_recorded_and_ordered(self, doc):
        articles = doc.by_type("article")
        pages = [a.page for a in articles]
        assert all(p is not None for p in pages), "every article must resolve to a page"
        assert pages == sorted(pages), "articles must appear in page order"
        assert 1 <= min(pages) and max(pages) <= 151

    def test_consolidation_markers_never_reach_the_text(self, doc):
        # These are editorial annotations, not legal text. A quote containing one would be
        # unverifiable against any other edition of the act.
        offenders = [u.unit_path for u in doc.units if any(c in u.text for c in "▲▼►◄")]
        assert not offenders

    def test_running_headers_never_reach_the_text(self, doc):
        offenders = [u.unit_path for u in doc.units if "— EN —" in u.text]
        assert not offenders

    def test_hyphenation_is_repaired(self, doc):
        # A soft hyphen left in the text means a word was never rejoined.
        offenders = [u.unit_path for u in doc.units if SOFT_HYPHEN in u.text]
        assert not offenders

    def test_consolidation_date_is_read_from_the_header(self, doc):
        assert doc.consolidation_date == "2026-07-27"

    def test_every_article_has_text(self, doc):
        empty = [a.unit_path for a in doc.by_type("article") if len(a.text) < 40]
        assert not empty

    def test_article_numbering_has_no_gaps(self, doc):
        # The check that catches a layout change without knowing what changed: legal
        # numbering is sequential, so a missing article is self-evident.
        plain = sorted(
            int(a.unit_number) for a in doc.by_type("article") if a.unit_number.isdigit()
        )
        assert plain == list(range(1, 114)), "articles 1..113 must all be present"


@_needs_pdf
class TestPartialParseIsRefused:
    """A parse that loses most of the document must fail, not return what it found.

    The caller activates whatever it is handed, so a partial *parse* would otherwise become
    a complete-looking *ingest* -- a corpus quietly missing provisions, answering as though
    they did not exist. That directly contradicts the pipeline's guarantee that a partially
    ingested version is never queryable.
    """

    @staticmethod
    def _pdf() -> bytes:
        return (settings.raw_data_dir / PDF_NAME).read_bytes()

    def test_losing_every_heading_raises(self, monkeypatch):
        monkeypatch.setattr("euaia.ingest.pdf_parser._locate", lambda *a, **k: None)
        with pytest.raises(PdfParseError, match="did not match a heading line"):
            parse(self._pdf())

    def test_the_error_names_the_scale_of_the_loss(self, monkeypatch):
        monkeypatch.setattr("euaia.ingest.pdf_parser._locate", lambda *a, **k: None)
        with pytest.raises(PdfParseError) as exc:
            parse(self._pdf())
        message = str(exc.value)
        assert "162 of 162" in message
        assert "layout has probably changed" in message

    def test_a_few_unmatched_headings_are_tolerated(self, monkeypatch):
        # Below the threshold the parse proceeds: the outline carries a handful of entries
        # whose heading is typeset unusually, and losing one should not fail an ingest.
        real = _locate
        calls = {"n": 0}

        def flaky(lines, entry, after):
            calls["n"] += 1
            return None if calls["n"] == 1 else real(lines, entry, after)

        monkeypatch.setattr("euaia.ingest.pdf_parser._locate", flaky)
        doc = parse(self._pdf())
        assert len(doc.by_type("article")) == 119

    def test_the_threshold_is_strict(self):
        # 2% of 162 outline entries is ~3. If this is ever loosened, the guard stops being
        # the thing that catches a layout change.
        assert MAX_MISSING_FRACTION <= 0.05


@_needs_pdf
class TestKnownSourceDisagreement:
    """The PDF and EUR-Lex's structured edition genuinely disagree about one provision.

    Article 10(5) was replaced by the new Article 4a. The consolidated PDF reflects the
    deletion -- Article 10's numbering runs 4 then 6 -- while the structured edition still
    carried the old paragraph. This is pinned as a test so the difference stays a known,
    documented fact rather than resurfacing later as a suspected parser bug.
    """

    def test_article_10_has_no_paragraph_5(self, doc):
        paths = {u.unit_path for u in doc.by_type("paragraph")}
        assert "CH_III/SEC_2/ART_10/PAR_4" in paths
        assert "CH_III/SEC_2/ART_10/PAR_6" in paths
        assert "CH_III/SEC_2/ART_10/PAR_5" not in paths

    def test_the_subject_matter_now_lives_in_article_4a(self, doc):
        article_4a = next(a for a in doc.by_type("article") if a.unit_number == "4a")
        assert "bias detection and correction" in article_4a.text


@_needs_pdf
class TestKnownTextLayerDamage:
    """Superscripts do not survive PDF text extraction, and the corpus inherits the damage.

    Article 51 sets the systemic-risk threshold for a general-purpose AI model at 10^25
    floating-point operations. The PDF's text layer renders the exponent as separate
    characters, so extraction yields '102 5'.

    Nothing in the parser can fix this -- the information is not in the file. It is pinned
    here rather than left implicit because of how it fails: it raises nothing, it reads like
    a number, and ``verify/citations.py`` will happily mark a quote containing it as verified,
    because the quote genuinely does appear in the ingested text. The verifier is working
    correctly; the source is wrong. If a future parser or library version recovers the real
    exponent, this test fails and that is the signal to update it.
    """

    def test_the_flop_threshold_is_damaged(self, doc):
        article_51 = next(a for a in doc.by_type("article") if a.unit_number == "51")
        assert "greater than 102 5" in article_51.text, (
            "expected the known-bad extraction; if this now reads 10^25 or similar, the "
            "text layer or the library has improved -- update the test and the warning in "
            "pdf_parser's module docstring"
        )
        assert "10²⁵" not in article_51.text
