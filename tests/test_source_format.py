"""Per-source parser dispatch.

Both sources are PDFs, but of different documents needing different readers: the consolidated
act through its bookmark outline, the as-adopted act through its preamble. Which reader runs
is decided by ``SourceSpec.parser`` rather than by inspecting the bytes, so a source can never
quietly be read by the wrong one -- the failure that would produce is a corpus of *unamended*
article text presented as though it were in force.
"""

from __future__ import annotations

import dataclasses

import pytest

from euaia.ingest import sources
from euaia.ingest.cellar import Manifestation
from euaia.ingest.pipeline import _PARSERS, parse_content


def _manifestation(content: bytes = b"%PDF-1.7") -> Manifestation:
    return Manifestation(content=content, fmt="pdf", sha256="0" * 64, filename="x.pdf")


class TestSourceDeclarations:
    def test_every_source_declares_a_known_parser(self):
        for spec in sources.ALL_SOURCES:
            assert spec.parser in _PARSERS, spec.key

    def test_the_consolidated_act_is_read_through_its_outline(self):
        assert sources.get("eu-ai-act").parser == "pdf_outline"

    def test_the_recitals_are_read_from_the_preamble(self):
        # The as-adopted PDF carries only 14 bookmarks (annexes), so there is no outline to
        # read the 180 recitals out of.
        assert sources.get("eu-ai-act-recitals").parser == "pdf_preamble"

    def test_the_two_sources_use_different_readers(self):
        assert sources.get("eu-ai-act").parser != sources.get("eu-ai-act-recitals").parser

    def test_the_recitals_source_stays_restricted_to_recitals(self):
        # The as-adopted act still contains an unamended Article 6; indexing it would let the
        # assistant cite superseded wording as if in force.
        assert sources.get("eu-ai-act-recitals").unit_types == frozenset({"recital"})


class TestParseDispatch:
    def test_an_unknown_parser_names_the_source_and_the_known_readers(self):
        spec = dataclasses.replace(sources.get("eu-ai-act"), parser="docx")
        with pytest.raises(ValueError, match="no parser named 'docx'") as exc:
            parse_content(spec, _manifestation())
        assert "eu-ai-act" in str(exc.value)
        assert "pdf_outline" in str(exc.value) and "pdf_preamble" in str(exc.value)

    def test_dispatch_follows_the_spec_not_the_bytes(self):
        # Keying on the manifestation would make the bytes the authority, re-opening the
        # "wrong reader silently ran" hole the pipeline's format assertion closes.
        called: list[str] = []
        spec = sources.get("eu-ai-act-recitals")
        original = _PARSERS[spec.parser]
        _PARSERS[spec.parser] = lambda b: called.append("preamble")  # type: ignore[assignment]
        try:
            parse_content(spec, _manifestation())
        finally:
            _PARSERS[spec.parser] = original
        assert called == ["preamble"]
