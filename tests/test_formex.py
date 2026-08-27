"""Formex parser tests.

The fixtures below are trimmed from the real AI Act Formex and deliberately reproduce the
quirks that broke earlier versions of the parser:

* ``STI.ART`` wrapping its text in ``<P>`` (original act) vs. direct text (consolidated)
* non-breaking space in ``TI.ART`` -- ``'Article\\xa04a'``
* ``<?PAGE NO='3'?>`` processing instructions interleaved with content
* ``<NOTE>`` footnotes that must stay out of citable text

An opt-in integration test at the bottom runs against the real cached document when one is
present, so the fast unit tests never depend on the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from lxml import etree

from euaia.config import settings
from euaia.ingest.formex import parse, serialize_text

CONSOLIDATED = b"""<?xml version="1.0" encoding="UTF-8"?>
<CONS.ACT>
  <INFO.CONSLEG CONSLEG.REF="2024R1689" CONSLEG.DATE="20260715" START.DATE="20260727"/>
  <CONS.DOC>
    <ENACTING.TERMS>
      <DIVISION>
        <TITLE><TI>CHAPTER II</TI><STI>PROHIBITED AI PRACTICES</STI></TITLE>
        <ARTICLE IDENTIFIER="005">
          <TI.ART>Article 5</TI.ART>
          <STI.ART>Prohibited AI practices'</STI.ART>
          <PARAG IDENTIFIER="005.001">
            <NO.PARAG>1.</NO.PARAG>
            <ALINEA>
              <P>The following AI practices shall be prohibited:</P>
              <LIST TYPE="alpha">
                <ITEM><NP><NO.P>(a)</NO.P><TXT>subliminal techniques;</TXT></NP></ITEM>
                <?PAGE NO='3'?>
                <ITEM><NP><NO.P>(b)</NO.P><TXT>exploitation of vulnerabilities;</TXT></NP></ITEM>
              </LIST>
            </ALINEA>
          </PARAG>
          <PARAG IDENTIFIER="005.002">
            <NO.PARAG>2.</NO.PARAG>
            <ALINEA>This paragraph applies without prejudice.<NOTE>See OJ L 1.</NOTE></ALINEA>
          </PARAG>
        </ARTICLE>
      </DIVISION>
      <DIVISION>
        <TITLE><TI>CHAPTER III</TI><STI>HIGH-RISK AI SYSTEMS</STI></TITLE>
        <DIVISION>
          <TITLE><TI>SECTION 1</TI><STI>Classification</STI></TITLE>
          <ARTICLE IDENTIFIER="004A">
            <TI.ART>Article\xc2\xa04a</TI.ART>
            <STI.ART>AI literacy</STI.ART>
            <PARAG IDENTIFIER="004A.001">
              <NO.PARAG>1.</NO.PARAG>
              <ALINEA>Providers shall ensure a sufficient level of AI literacy.</ALINEA>
            </PARAG>
          </ARTICLE>
        </DIVISION>
      </DIVISION>
    </ENACTING.TERMS>
    <CONS.ANNEX>
      <TITLE><TI>ANNEX III</TI><STI>High-risk AI systems referred to in Article 6(2)</STI></TITLE>
      <CONTENTS><P>Biometrics, in so far as their use is permitted.</P></CONTENTS>
    </CONS.ANNEX>
  </CONS.DOC>
</CONS.ACT>
"""

ORIGINAL = b"""<?xml version="1.0" encoding="UTF-8"?>
<ACT>
  <PREAMBLE>
    <GR.CONSID>
      <CONSID><NP><NO.P>(1)</NO.P><TXT>The purpose of this Regulation is to improve.</TXT></NP></CONSID>
      <CONSID><NP><NO.P>(2)</NO.P><TXT>AI systems can be easily deployed.</TXT></NP></CONSID>
    </GR.CONSID>
  </PREAMBLE>
  <ENACTING.TERMS>
    <DIVISION>
      <TITLE><TI>CHAPTER II</TI><STI>PROHIBITED AI PRACTICES</STI></TITLE>
      <ARTICLE IDENTIFIER="005">
        <TI.ART>Article 5</TI.ART>
        <STI.ART><P>Prohibited AI practices</P></STI.ART>
        <PARAG IDENTIFIER="005.001">
          <NO.PARAG>1.</NO.PARAG>
          <ALINEA><P>The following AI practices shall be prohibited:</P></ALINEA>
        </PARAG>
      </ARTICLE>
    </DIVISION>
  </ENACTING.TERMS>
</ACT>
"""


@pytest.fixture(scope="module")
def consolidated():
    return parse(CONSOLIDATED)


@pytest.fixture(scope="module")
def original():
    return parse(ORIGINAL)


class TestSerializeText:
    def test_block_elements_are_separated(self):
        # xpath("string()") would give 'Article 5Prohibited AI practices1.The following...'
        el = etree.fromstring(CONSOLIDATED).find(".//ARTICLE")
        text = serialize_text(el)
        assert text.startswith("Article 5\nProhibited AI practices'\n1.")
        assert "Article 5Prohibited" not in text

    def test_processing_instructions_are_skipped(self):
        el = etree.fromstring(CONSOLIDATED).find(".//ARTICLE")
        text = serialize_text(el)
        assert "PAGE" not in text
        assert "subliminal techniques;" in text

    def test_footnotes_are_excluded(self):
        el = etree.fromstring(CONSOLIDATED).find(".//ARTICLE")
        text = serialize_text(el)
        assert "This paragraph applies without prejudice." in text
        assert "See OJ L 1." not in text

    def test_list_items_keep_their_labels(self):
        el = etree.fromstring(CONSOLIDATED).find(".//ARTICLE")
        text = serialize_text(el)
        assert "(a)" in text and "(b)" in text


class TestConsolidatedAct:
    def test_root_and_conslegs_metadata(self, consolidated):
        assert consolidated.root_tag == "CONS.ACT"
        assert consolidated.consolidation_date == "20260715"
        assert consolidated.start_date == "20260727"

    def test_articles_found(self, consolidated):
        numbers = [a.unit_number for a in consolidated.by_type("article")]
        assert numbers == ["5", "4a"]

    def test_nbsp_in_title_does_not_break_number_extraction(self, consolidated):
        art = next(a for a in consolidated.by_type("article") if a.unit_number == "4a")
        assert art.heading == "AI literacy"

    def test_direct_text_heading_strips_trailing_apostrophe(self, consolidated):
        art = next(a for a in consolidated.by_type("article") if a.unit_number == "5")
        assert art.heading == "Prohibited AI practices"

    def test_chapter_and_section_parenting(self, consolidated):
        art5 = next(a for a in consolidated.by_type("article") if a.unit_number == "5")
        art4a = next(a for a in consolidated.by_type("article") if a.unit_number == "4a")
        assert art5.unit_path == "CH_II/ART_5"
        assert art5.parent_path == "CH_II"
        # Nested DIVISION: chapter > section > article.
        assert art4a.unit_path == "CH_III/SEC_1/ART_4a"
        assert art4a.parent_path == "CH_III/SEC_1"

    def test_paragraphs_are_numbered_within_their_article(self, consolidated):
        paras = [
            p for p in consolidated.by_type("paragraph") if p.parent_path == "CH_II/ART_5"
        ]
        assert [p.unit_number for p in paras] == ["5(1)", "5(2)"]
        assert [p.unit_path for p in paras] == ["CH_II/ART_5/PAR_1", "CH_II/ART_5/PAR_2"]

    def test_paragraphs_of_a_nested_article(self, consolidated):
        paras = [
            p for p in consolidated.by_type("paragraph") if p.parent_path == "CH_III/SEC_1/ART_4a"
        ]
        assert [p.unit_number for p in paras] == ["4a(1)"]

    def test_annex_parsed_with_number_and_heading(self, consolidated):
        annexes = consolidated.by_type("annex")
        assert len(annexes) == 1
        assert annexes[0].unit_number == "III"
        assert annexes[0].heading == "High-risk AI systems referred to in Article 6(2)"

    def test_consolidated_act_has_no_recitals(self, consolidated):
        # Consolidation does not restate recitals; they stay with the original act.
        assert consolidated.by_type("recital") == []


class TestOriginalAct:
    def test_root_tag(self, original):
        assert original.root_tag == "ACT"

    def test_recitals_parsed_with_numbers(self, original):
        recitals = original.by_type("recital")
        assert [r.unit_number for r in recitals] == ["1", "2"]
        assert "improve" in recitals[0].text

    def test_wrapped_sti_art_heading_is_found(self, original):
        # The regression: <STI.ART><P>...</P></STI.ART> used to yield None.
        art = next(a for a in original.by_type("article") if a.unit_number == "5")
        assert art.heading == "Prohibited AI practices"


class TestFailureModes:
    def test_empty_document_raises(self):
        with pytest.raises(ValueError, match="no structural units"):
            parse(b"<ACT><ENACTING.TERMS/></ACT>")


@pytest.mark.skipif(
    not (settings.raw_data_dir / "consolidated_02024R1689-20260727.xml").exists(),
    reason="cached real document not present; run the ingestion pipeline first",
)
class TestRealDocument:
    """Runs only when the real Formex has been downloaded. Guards against silent drift."""

    @pytest.fixture(scope="class")
    @classmethod
    def doc(cls):
        path: Path = settings.raw_data_dir / "consolidated_02024R1689-20260727.xml"
        return parse(path.read_bytes())

    def test_expected_structure_counts(self, doc):
        assert len(doc.by_type("article")) == 119  # 113 original + inserted (4a, ...)
        assert len(doc.by_type("chapter")) == 13
        assert len(doc.by_type("annex")) == 14

    def test_article_numbers_are_unique_and_complete(self, doc):
        numbers = [a.unit_number for a in doc.by_type("article")]
        assert all(numbers), "every article must have a number"
        assert len(numbers) == len(set(numbers)), "article numbers must be unique"

    def test_key_provisions_resolve(self, doc):
        by_number = {a.unit_number: a for a in doc.by_type("article")}
        assert by_number["5"].heading == "Prohibited AI practices"
        assert by_number["6"].unit_path == "CH_III/SEC_1/ART_6"
        assert "Transparency obligations" in by_number["50"].heading
        # Amendments applied: 1a/1b exist only in the consolidated text.
        assert "1a." in by_number["6"].text
