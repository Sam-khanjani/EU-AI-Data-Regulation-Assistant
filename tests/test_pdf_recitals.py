"""Reading recitals out of an as-adopted act's PDF.

The unit tests cover the size rule that separates a recital from a footnote. The class at
the bottom runs against the real 144-page PDF: 180 recitals, numbered 1..180, no gaps.
"""

from __future__ import annotations

import pytest

from euaia.config import settings
from euaia.ingest.pdf_parser import PdfParseError
from euaia.ingest.pdf_recitals import (
    _EMPTY_REF,
    _NOISE,
    BODY_SIZE,
    MARKER_SIZE,
    MIN_RECITALS,
    _classify,
    parse_recitals,
)

PDF_NAME = "32024R1689.ENG.pdf"

_needs_pdf = pytest.mark.skipif(
    not (settings.raw_data_dir / PDF_NAME).exists(),
    reason="cached as-adopted PDF not present; run `--fetch-pdf` first",
)


def _words(*pairs: tuple[str, float]) -> list[dict]:
    return [{"text": t, "size": s, "x0": i * 10} for i, (t, s) in enumerate(pairs)]


class TestRecitalVersusFootnote:
    def test_a_recital_line_is_a_small_number_then_body_prose(self):
        # The whole trick: [8.5, 9.6, 9.6, ...] -- a hanging number introducing prose.
        words = _words(("(27)", MARKER_SIZE), ("While", BODY_SIZE), ("the", BODY_SIZE))
        assert _classify(words) == ("27", "While the")

    def test_a_footnote_line_is_uniformly_small_and_yields_nothing(self):
        # [8.5, 8.5, 8.5, ...]. Same '(1)' token, different line -- and this is the case that
        # a sequence-only rule gets wrong, taking a footnote as the next recital.
        words = _words(("(1)", MARKER_SIZE), ("OJ", MARKER_SIZE), ("C", MARKER_SIZE))
        assert _classify(words) == (None, "")

    def test_a_continuation_line_is_body_with_no_number(self):
        words = _words(("risk-based", BODY_SIZE), ("approach", BODY_SIZE))
        assert _classify(words) == (None, "risk-based approach")

    def test_a_number_at_body_size_does_not_start_a_recital(self):
        words = _words(("(27)", BODY_SIZE), ("While", BODY_SIZE))
        assert _classify(words)[0] is None


class TestNoise:
    @pytest.mark.parametrize(
        "line",
        [
            "OJ L, 12.7.2024",
            "ELI: http://data.europa.eu/eli/reg/2024/1689/oj",
            "2/144 ELI: http://data.europa.eu/eli/reg/2024/1689/oj",
        ],
    )
    def test_running_header_and_footer_are_dropped(self, line):
        assert _NOISE.match(line)

    def test_provision_text_is_not_mistaken_for_furniture(self):
        assert not _NOISE.match("(27) While the risk-based approach is the basis")

    def test_stranded_footnote_brackets_are_removed(self):
        # A footnote reference's digit is a superscript at another size, so filtering to body
        # size strands its brackets. EUR-Lex's structured edition drops the marker entirely.
        assert _EMPTY_REF.sub("", "the Union ( ) and the Member States") == (
            "the Union and the Member States"
        )


class TestFailsLoudly:
    def test_a_pdf_without_the_enacting_formula_is_refused(self):
        # Without that boundary there is no way to tell preamble from operative text.
        with pytest.raises(PdfParseError, match="HAVE ADOPTED THIS REGULATION"):
            parse_recitals(_MINIMAL_PDF)


@_needs_pdf
class TestRealDocument:
    @pytest.fixture(scope="class")
    @classmethod
    def doc(cls):
        return parse_recitals((settings.raw_data_dir / PDF_NAME).read_bytes())

    def test_finds_all_180_recitals(self, doc):
        assert len(doc.units) == 180

    def test_numbering_runs_1_to_180_with_no_gaps(self, doc):
        # The check that makes the size rule safe to depend on: if the typography changes,
        # this run breaks rather than the reader silently returning a fragment.
        assert [u.unit_number for u in doc.units] == [str(n) for n in range(1, 181)]

    def test_every_unit_is_a_recital(self, doc):
        assert {u.unit_type for u in doc.units} == {"recital"}

    def test_paths_match_the_shared_scheme(self, doc):
        assert doc.units[26].unit_path == "RCT_27"

    def test_recitals_lead_with_their_number(self, doc):
        assert doc.units[26].text.startswith("(27)\n")

    def test_text_is_the_real_provision(self, doc):
        assert "risk-based approach" in doc.units[26].text
        assert "seven non-binding ethical principles" in doc.units[26].text

    def test_pages_are_recorded_and_ordered(self, doc):
        pages = [u.page for u in doc.units]
        assert all(p is not None for p in pages)
        assert pages == sorted(pages)

    def test_footnote_text_never_leaks_in(self, doc):
        # 'OJ C 517, 22.12.2021, p. 56.' is a footnote on page 1, numbered (1) exactly like
        # recital 1. It must not appear in any recital.
        offenders = [u.unit_number for u in doc.units if "OJ C 517" in u.text]
        assert not offenders

    def test_running_furniture_never_leaks_in(self, doc):
        offenders = [u.unit_number for u in doc.units if "ELI:" in u.text or "OJ L," in u.text]
        assert not offenders

    def test_no_stranded_footnote_brackets(self, doc):
        offenders = [u.unit_number for u in doc.units if "( )" in u.text]
        assert not offenders

    def test_operative_text_is_excluded(self, doc):
        # Everything after the enacting formula is articles, not preamble.
        joined = " ".join(u.text for u in doc.units)
        assert "Subject matter" not in joined
        assert "shall be prohibited" not in joined

    def test_the_minimum_is_a_real_floor(self):
        assert MIN_RECITALS >= 20


# A one-page PDF with no enacting formula, built inline so the test needs no fixture file.
_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
    b"trailer<</Root 1 0 R>>\n"
)
