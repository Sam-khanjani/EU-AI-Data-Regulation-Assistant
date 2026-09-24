"""Codes of practice: parsed into the units a reader cites, and read by their structure.

The parser tests run against the real PDFs when they have been downloaded
(``python -m euaia.ingest.ec_documents``) and skip otherwise, like the AI Act's.
"""

from __future__ import annotations

from functools import cache

import pytest

from euaia.api.service import AnswerClaim, AnswerView
from euaia.chat import views
from euaia.config import settings
from euaia.ingest.chunker import chunk_document
from euaia.ingest.pdf import ParsedDocument, ParsedUnit, parse_code
from euaia.retrieval.hybrid import code_parts

CODES = settings.raw_data_dir / "AI_Act" / "Codes_of_Practice"


@cache
def parsed(name: str) -> dict[str, ParsedUnit]:
    path = CODES / name
    if not path.exists():
        pytest.skip("codes not downloaded; run `python -m euaia.ingest.ec_documents`")
    return {u.unit_number: u for u in parse_code(path.read_bytes()).units}


AI_CONTENT = "08_Transparency_Code_AI_Content.pdf"


class TestCodeUnits:
    def test_parts_are_numbered_and_sections_told_apart(self):
        units = parsed(AI_CONTENT)
        # Both sections have a Commitment 1; unprefixed they would be indistinguishable.
        assert {"S1 Commitment 1", "S2 Commitment 1", "S1 Measure 1.1",
                "S1 Sub-measure 1.1.2", "S2 Measure 1.1"} <= set(units)

    def test_each_lettered_recital_is_its_own_unit_with_its_title(self):
        units = parsed(AI_CONTENT)
        assert [n for n in units if n.startswith("S1 Recital ")] == [
            f"S1 Recital {letter}" for letter in "abcdefg"
        ]
        assert units["S1 Recital b"].heading == "Technical solutions for marking"
        assert units["S1 Recitals"].text.strip() == "Recitals\nWhereas:"

    def test_glossary_terms_are_units(self):
        units = parsed(AI_CONTENT)
        assert units["S1 Glossary: Watermark"].text.startswith("Watermark\n")
        assert "Term Definition" not in units["S1 Glossary"].text

    def test_quoted_act_text_is_split_out_and_labelled(self):
        units = parsed(AI_CONTENT)
        quoted = units["S1 Commitment 1, quoted AI Act text"]
        assert quoted.heading == "Quoted from Article 50(2) and recitals 133 and 135 AI Act"
        assert quoted.text.startswith("2. Providers of AI systems")
        assert "not the code's own words" in quoted.context
        # The voluntary commitment no longer carries the Act's "shall ensure".
        assert "shall ensure" not in units["S1 Commitment 1"].text

    def test_a_codes_own_numbered_paragraph_is_not_mistaken_for_quoted_act_text(self):
        units = parsed("06_GPAI_Code_Copyright.pdf")
        assert "Commitment 1, quoted AI Act text" not in units
        assert "implements Article 53(1)(c) AI Act" in units["Measure 1.1"].context

    def test_context_names_the_path_what_it_implements_and_whether_optional(self):
        context = parsed(AI_CONTENT)["S1 Sub-measure 1.1.3"].context
        assert "Commitment 1: Marking of AI-generated or Manipulated Content" in context
        assert "Measure 1.1: Machine-readable marking techniques" in context
        assert "implements Article 50(2)" in context
        assert "optional measure" in context

    def test_page_numbers_do_not_leak_into_the_text(self):
        assert "\n7\n" not in parsed(AI_CONTENT)["S1 Commitment 1"].text


class TestChunking:
    def test_the_context_is_the_breadcrumb_and_heading_only_containers_are_skipped(self):
        parent = ParsedUnit("section", "SEC_1", "Recitals\nWhereas:", 1, "Recitals")
        child = ParsedUnit("recital", "SEC_1/REC_a", "a) Trust: " + "word " * 80, 2,
                           "Recital a", parent_path="SEC_1", context="Recitals › Recital a")
        drafts = chunk_document(ParsedDocument([parent, child]), "Code", None)
        assert [d.unit_path for d in drafts] == ["SEC_1/REC_a"]
        assert drafts[0].text.startswith("Code - Recitals › Recital a\n")


class TestNamedParts:
    def test_parts_are_read_from_the_question(self):
        assert code_parts("What are the measures under Commitment 1 and sub-measure 1.1.2?") == [
            "Commitment 1", "Sub-measure 1.1.2"
        ]


class TestCodeOnlyAnswers:
    def _view(self, *bases: str) -> AnswerView:
        return AnswerView(
            question="q", verdict="answered",
            claims=[AnswerClaim(text=f"claim {b}", basis=b) for b in bases],
        )

    def test_an_answer_resting_only_on_codes_says_so_whatever_the_wording(self):
        assert views.CODE_ONLY_NOTE in views.answer_markdown(self._view("code", "code"))

    def test_an_answer_with_law_in_it_does_not(self):
        assert views.CODE_ONLY_NOTE not in views.answer_markdown(self._view("law", "code"))
