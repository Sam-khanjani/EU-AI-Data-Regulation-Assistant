"""Sources are not interchangeable: the authority tier, from the spec to the rendered answer.

The failure these guard against is a single one, at four points along the path: a voluntary
code of practice being read as though it stated a legal requirement. It is the most likely
wrong answer this corpus can produce, because a code is longer, more concrete and more
quotable than the provision it implements, so relevance ranking favours it.
"""

from __future__ import annotations

from dataclasses import replace

from euaia.api.service import AnswerClaim
from euaia.chat import views
from euaia.graph import prompts
from euaia.ingest import sources
from euaia.retrieval.hybrid import (
    AUTHORITY_ORDER,
    Candidate,
    RetrievedUnit,
    by_authority,
    citation_label,
    fit_token_budget,
)
from euaia.retrieval.rerank import LabelledChunk, _keep_with_law
from euaia.verify.citations import VerifiedClaim, VerifiedQuote


def unit(unit_id: int, authority: str, source_key: str = "k") -> RetrievedUnit:
    return RetrievedUnit(
        unit_id=unit_id, unit_path=f"U{unit_id}", unit_type="section",
        unit_number=str(unit_id), heading=None, citation_label=f"U{unit_id}",
        text="body", document_version_id=1, version_label="v", source_key=source_key,
        authority=authority, deeplink=None, score=1.0,
    )


def quote(authority: str) -> VerifiedQuote:
    return VerifiedQuote(
        evidence_label="E1", unit_id=1, unit_path="p", citation_label="c",
        quote="q", method="exact", score=100.0, authority=authority,
    )


def claim(text: str, basis: str) -> AnswerClaim:
    return AnswerClaim(text=text, citations=[], basis=basis)


class TestSourceSpecs:
    def test_only_the_act_is_binding_law(self):
        law = {s.key for s in sources.ALL_SOURCES if s.authority == "law"}
        assert law == {"eu-ai-act", "eu-ai-act-recitals"}

    def test_codes_of_practice_are_marked_voluntary(self):
        codes = {s.key for s in sources.ALL_SOURCES if s.authority == "code"}
        assert codes == {
            "ec-gpai-code-transparency",
            "ec-gpai-code-copyright",
            "ec-gpai-code-safety",
            "ec-transparency-code-ai-content",
        }

    def test_the_qa_pages_are_guidance_not_a_tier_of_their_own(self):
        assert sources.get("ec-gpai-qa").authority == "guidance"

    def test_draft_status_reaches_the_version_label(self):
        """The label is what the answer model sees, so this is how a draft announces itself."""
        assert sources.get("ec-high-risk-guidelines-annex-iii").fixed_version_label == (
            "draft 2026-05-19"
        )

    def test_every_commission_source_declares_a_file_and_no_celex(self):
        for spec in sources.COMMISSION_SOURCES:
            assert spec.local_file and spec.celex_base is None, spec.key


class TestEvidenceOrdering:
    def test_binding_law_is_read_first(self):
        ordered = by_authority([unit(1, "code"), unit(2, "guidance"), unit(3, "law")])
        assert [u.unit_id for u in ordered] == [3, 2, 1]

    def test_rank_order_survives_within_a_tier(self):
        """Authority decides the tier, relevance still decides the order inside it."""
        ordered = by_authority([unit(1, "law"), unit(2, "law"), unit(3, "law")])
        assert [u.unit_id for u in ordered] == [1, 2, 3]

    def test_nothing_is_dropped(self):
        units = [unit(i, tier) for i, tier in enumerate(["code", "law", "guidance", "code"])]
        assert len(by_authority(units)) == len(units)

    def test_an_unknown_tier_sorts_last_rather_than_first(self):
        """A source whose tier we do not recognise must not displace the Act."""
        ordered = by_authority([unit(1, "invented"), unit(2, "law")])
        assert [u.unit_id for u in ordered] == [2, 1]

    def test_order_matches_the_declared_ranking(self):
        assert AUTHORITY_ORDER == ("law", "guidance", "code")


class TestClaimBasisIsComputed:
    """The model never states a claim's basis -- it is derived from what verified."""

    def test_a_claim_quoting_the_act_is_grounded_in_law(self):
        assert VerifiedClaim(text="t", quotes=[quote("law")]).basis == "law"

    def test_a_claim_quoting_only_a_code_is_not(self):
        assert VerifiedClaim(text="t", quotes=[quote("code")]).basis == "code"

    def test_the_strongest_quote_decides(self):
        mixed = VerifiedClaim(text="t", quotes=[quote("code"), quote("law")])
        assert mixed.basis == "law"

    def test_guidance_outranks_a_code(self):
        mixed = VerifiedClaim(text="t", quotes=[quote("code"), quote("guidance")])
        assert mixed.basis == "guidance"


class TestEvidenceBlocksDeclareWhatTheyAre:
    def test_a_code_block_says_it_is_voluntary(self):
        rendered = prompts.format_evidence([unit(1, "code")])
        assert "VOLUNTARY CODE OF PRACTICE (not binding)" in rendered

    def test_a_law_block_says_it_is_binding(self):
        assert "BINDING LAW" in prompts.format_evidence([unit(1, "law")])

    def test_the_version_label_travels_with_it(self):
        block = prompts.format_evidence([unit(1, "guidance")])
        assert "COMMISSION GUIDANCE (not binding); v" in block

    def test_the_answer_prompt_forbids_stating_guidance_as_a_requirement(self):
        system = prompts.ANSWER_SYSTEM
        assert "BINDING LAW" in system and "VOLUNTARY CODE OF PRACTICE" in system


class TestCitationsNameTheirDocument:
    def test_an_article_of_the_act_needs_no_document_name(self):
        assert citation_label("article", "6", "CH_III/ART_6", "eu-ai-act") == "Article 6"

    def test_a_commission_section_is_useless_without_one(self):
        label = citation_label("section", "2.1", "SEC_2/SEC_2.1", "ec-gpai-guidelines")
        assert label == "GPAI Guidelines 2.1"

    def test_a_document_with_no_sections_is_cited_whole(self):
        assert citation_label("section", None, "SEC_1", "ec-gpai-qa") == "GPAI QA"

    def test_an_unknown_source_falls_back_to_the_old_behaviour(self):
        assert citation_label("article", "6", "ART_6", "not-registered") == "Article 6"


class TestAnswerRendering:
    def test_mixed_bases_are_grouped_and_headed(self):
        parts = views._grouped_claims(
            [claim("practical", "code"), claim("required", "law"), claim("explains", "guidance")]
        )
        assert parts == [
            "**Legal requirement**", "required",
            "**Commission guidance**", "explains",
            "**Practical implementation**", "practical",
        ]

    def test_an_answer_wholly_from_the_act_gets_no_headings(self):
        """Most answers are law-only; heading every one adds ceremony that means nothing."""
        parts = views._grouped_claims([claim("a", "law"), claim("b", "law")])
        assert parts == ["a", "b"]


class TestBindingTextKeepsASlot:
    """Ordering evidence by authority can only order what survived reranking."""

    def _passing(self, tiers: list[str]) -> list[LabelledChunk]:
        """Candidates already sorted by score, as `rerank` hands them over."""
        return [
            LabelledChunk(
                label=f"C{index}",
                candidate=Candidate(
                    chunk_id=index, unit_id=index, unit_path=f"ART_{index}",
                    unit_type="article", unit_number=str(index), heading=None,
                    document_version_id=1, version_label="v", source_key="k",
                    authority=tier, text="t", chunk_text="t",
                ),
                score=1.0 - index / 100,
            )
            for index, tier in enumerate(tiers)
        ]

    def test_law_is_promoted_when_guidance_sweeps_the_top(self):
        kept = _keep_with_law(self._passing(["code", "guidance", "code", "guidance", "law"]), 4)
        assert [c.authority for c in kept] == ["code", "guidance", "code", "law"]

    def test_it_displaces_the_weakest_not_the_strongest(self):
        kept = _keep_with_law(self._passing(["guidance", "guidance", "guidance", "law"]), 3)
        assert [c.unit_path for c in kept] == ["ART_0", "ART_1", "ART_3"]

    def test_nothing_changes_when_law_already_ranked(self):
        passing = self._passing(["law", "guidance", "code", "code", "law"])
        assert [c.authority for c in _keep_with_law(passing, 4)] == [
            "law", "guidance", "code", "code",
        ]

    def test_no_binding_text_retrieved_at_all_keeps_the_top_scorers(self):
        """A question the Act genuinely does not address must still be answerable."""
        kept = _keep_with_law(self._passing(["guidance", "code", "guidance"]), 4)
        assert len(kept) == 3

    def test_it_is_one_slot_not_a_quota(self):
        kept = _keep_with_law(self._passing(["guidance", "code", "code", "code", "law", "law"]), 4)
        assert sum(1 for c in kept if c.authority == "law") == 1


class TestOversizedUnitsKeepTheFirstSlot:
    """Who receives the over-budget exemption is decided by the ordering, not by the budget.

    Article 5 is 3,714 tokens against a 2,400 budget and is the entire answer to "which
    practices are prohibited?". The Commission's Q&A pages have no heading structure and are
    single ~5,700-token units that answer nothing in particular. Both are over budget; only
    one should be able to take the whole of it, and authority ordering is what decides which.
    """

    def _unit(self, unit_id: int, tokens: int, authority: str = "guidance") -> RetrievedUnit:
        return replace(unit(unit_id, authority), text="word " * tokens)

    def test_a_long_provision_is_kept_whole_rather_than_traded_for_short_ones(self):
        units = [self._unit(1, 3000, "law"), self._unit(2, 100, "law"), self._unit(3, 100)]
        kept = fit_token_budget(units, budget=600)
        assert [u.unit_id for u in kept] == [1]

    def test_a_shapeless_guidance_blob_does_not_get_that_slot_when_law_was_retrieved(self):
        """Ordered by authority, the blob is never first, so it is simply skipped."""
        units = by_authority([self._unit(1, 4000), self._unit(2, 100, "law")])
        kept = fit_token_budget(units, budget=600)
        assert [u.unit_id for u in kept] == [2]

    def test_an_empty_pool_stays_empty(self):
        assert fit_token_budget([], budget=600) == []


class TestAuthorityOrdersTheBudgetNotJustTheReading:
    """Regression: the two steps compose in one order only.

    Reranking reserves a slot for binding text but appends it last. Spending the token budget
    in relevance order therefore dropped the Act *after* promoting it -- measured on "What
    must providers do about AI-generated content?", where Article 50 was promoted over a
    higher-scoring code section and then evicted, leaving guidance and a voluntary code as
    the whole evidence set.
    """

    def _units(self) -> list[RetrievedUnit]:
        # As reranking hands them over: relevance order, binding text promoted last, and the
        # provision is longer than the guidance discussing it.
        return [
            replace(unit(1, "code"), text="word " * 300),
            replace(unit(2, "guidance"), text="word " * 300),
            replace(unit(3, "law"), text="word " * 500),
        ]

    # Sized so the provision plus one shorter unit fit, and all three do not: ~726 tokens
    # for the law unit and ~448 for each of the others.
    BUDGET = 1200

    def test_ordering_before_the_budget_keeps_the_law(self):
        kept = fit_token_budget(by_authority(self._units()), budget=self.BUDGET)
        assert [u.authority for u in kept] == ["law", "guidance"]

    def test_ordering_after_the_budget_loses_it(self):
        """The bug, pinned: same inputs, same budget, binding text gone."""
        kept = by_authority(fit_token_budget(self._units(), budget=self.BUDGET))
        assert "law" not in [u.authority for u in kept]
