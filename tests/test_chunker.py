"""Chunker tests.

Packing sibling paragraphs is the behaviour that matters here. It exists for two reasons --
a 72-token paragraph has almost no standalone meaning, and the free-tier embedding quota is
counted per item -- but it must not buy either at the cost of citation precision. A chunk
that straddled two articles could not be cited, and a quote split across a chunk boundary
could not be verified.
"""

from __future__ import annotations

import pytest

from euaia.config import settings
from euaia.ingest.chunker import build_breadcrumb, chunk_document, count_tokens
from euaia.ingest.document import ParsedDocument, ParsedUnit

TITLE = "Regulation (EU) 2024/1689 (AI Act)"


def unit(unit_type, number, path, text, ordinal, parent=None, heading=None):
    return ParsedUnit(
        unit_type=unit_type,
        unit_number=number,
        unit_path=path,
        text=text,
        ordinal=ordinal,
        heading=heading,
        parent_path=parent,
    )


def words(n: int, tag: str = "provision") -> str:
    return " ".join([tag] * n)


@pytest.fixture
def doc():
    """Two articles under a chapter: one with short paragraphs, one with a huge paragraph."""
    units = [
        unit("chapter", "III", "CH_III", "CHAPTER III HIGH-RISK", 1),
        unit(
            "article", "6", "CH_III/ART_6", "Article 6\nClassification rules", 2,
            parent="CH_III", heading="Classification rules for high-risk AI systems",
        ),
        unit("paragraph", "6(1)", "CH_III/ART_6/PAR_1", words(40, "alpha"), 3, parent="CH_III/ART_6"),
        unit("paragraph", "6(2)", "CH_III/ART_6/PAR_2", words(40, "beta"), 4, parent="CH_III/ART_6"),
        unit("paragraph", "6(3)", "CH_III/ART_6/PAR_3", words(40, "gamma"), 5, parent="CH_III/ART_6"),
        unit(
            "article", "5", "CH_III/ART_5", "Article 5\nProhibited", 6,
            parent="CH_III", heading="Prohibited AI practices",
        ),
        unit("paragraph", "5(1)", "CH_III/ART_5/PAR_1", words(2000, "delta"), 7, parent="CH_III/ART_5"),
        unit("annex", "III", "ANX_III", "ANNEX III\n" + words(50, "annexword"), 8),
        unit("recital", "27", "RCT_27", words(60, "recitalword"), 9),
    ]
    return ParsedDocument(units=units)


class TestPacking:
    def test_short_sibling_paragraphs_are_packed_together(self, doc):
        drafts = chunk_document(doc, TITLE)
        art6 = [d for d in drafts if d.unit_path == "CH_III/ART_6"]
        assert len(art6) == 1, "three 40-word paragraphs should share one chunk"
        for tag in ("alpha", "beta", "gamma"):
            assert tag in art6[0].body

    def test_packed_chunks_attach_to_the_article_not_the_paragraph(self, doc):
        drafts = chunk_document(doc, TITLE)
        paths = {d.unit_path for d in drafts}
        assert "CH_III/ART_6" in paths
        assert not any(p.startswith("CH_III/ART_6/PAR_") for p in paths), (
            "a packed chunk spans several paragraphs, so it must cite the article"
        )

    def test_packing_never_crosses_an_article_boundary(self, doc):
        # A chunk spanning two articles could not be cited precisely.
        for draft in chunk_document(doc, TITLE):
            assert "delta" not in draft.body or "alpha" not in draft.body

    def test_an_oversized_paragraph_is_split_rather_than_dropped(self, doc):
        drafts = chunk_document(doc, TITLE)
        art5 = [d for d in drafts if d.unit_path == "CH_III/ART_5"]
        assert len(art5) > 1, "a 2000-word paragraph must be split"
        assert all(d.token_count <= settings.chunk_max_tokens for d in art5)

    def test_every_chunk_fits_the_embedding_input_limit(self, doc):
        for draft in chunk_document(doc, TITLE):
            assert draft.token_count < settings.embed_input_token_limit

    def test_annexes_and_recitals_are_still_chunked(self, doc):
        paths = {d.unit_path for d in chunk_document(doc, TITLE)}
        assert "ANX_III" in paths and "RCT_27" in paths

    def test_chapters_are_not_embedded(self, doc):
        # A chapter's text is just its title; embedding it adds a near-duplicate.
        paths = {d.unit_path for d in chunk_document(doc, TITLE)}
        assert "CH_III" not in paths


class TestUnitTypeFilter:
    def test_recitals_only_excludes_operative_text(self, doc):
        drafts = chunk_document(doc, TITLE, frozenset({"recital"}))
        assert {d.unit_path for d in drafts} == {"RCT_27"}

    def test_filtering_out_paragraphs_falls_back_to_article_text(self, doc):
        drafts = chunk_document(doc, TITLE, frozenset({"article"}))
        art6 = [d for d in drafts if d.unit_path == "CH_III/ART_6"]
        assert art6 and "Classification rules" in art6[0].body


class TestBreadcrumbs:
    def test_breadcrumb_carries_the_structural_location(self, doc):
        by_path = {u.unit_path: u for u in doc.units}
        crumb = build_breadcrumb(TITLE, by_path["CH_III/ART_6"], by_path)
        assert TITLE in crumb
        assert "Chapter III" in crumb
        assert "Article 6" in crumb
        assert "Classification rules for high-risk AI systems" in crumb

    def test_breadcrumb_is_included_in_the_embedded_text(self, doc):
        drafts = chunk_document(doc, TITLE)
        art6 = next(d for d in drafts if d.unit_path == "CH_III/ART_6")
        assert art6.text.startswith(TITLE)
        assert art6.body in art6.text

    def test_body_excludes_the_breadcrumb(self, doc):
        # Quotes are verified against unit text, never the breadcrumb we synthesised.
        drafts = chunk_document(doc, TITLE)
        art6 = next(d for d in drafts if d.unit_path == "CH_III/ART_6")
        assert not art6.body.startswith(TITLE)


class TestTokenCounting:
    def test_counts_grow_with_text(self):
        assert count_tokens(words(100)) > count_tokens(words(10))

    def test_empty_text(self):
        assert count_tokens("") == 0
