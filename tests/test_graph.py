"""Every path through the answer graph, with a scripted model and stubbed search.

Quote verification is real: a draft only survives if its quotes are in the evidence.
"""

from __future__ import annotations

import pytest

from euaia.graph import nodes, prompts
from euaia.llm.groq_client import Completion, SchemaValidationFailed
from euaia.llm.schemas import ASSESSMENT_SCHEMA
from euaia.retrieval.hybrid import RetrievedUnit
from euaia.retrieval.rerank import RerankResult

ART50 = ("Providers of AI systems shall ensure that the outputs of the AI system are marked in "
         "a machine-readable format and detectable as artificially generated or manipulated.")
ART3 = ("'deep fake' means AI-generated or manipulated image, audio or video content that "
        "resembles existing persons, objects, places, entities or events.")
GOOD, BAD = "outputs of the AI system are marked in a machine-readable format", "invented words"
DEFINITION = "AI-generated or manipulated image, audio or video content"


def unit(text: str, uid: int, label: str) -> RetrievedUnit:
    return RetrievedUnit(
        unit_id=uid, unit_path=f"ART_{uid}", unit_type="article", unit_number=str(uid),
        heading=None, citation_label=label, text=text, document_version_id=7,
        version_label="v", source_key="act", authority="law", deeplink=None, score=1.0,
        matched_chunk_ids=[uid],
    )


UNITS = {"deep fakes": unit(ART3, 3, "Article 3"), "default": unit(ART50, 50, "Article 50")}


def answer(*quotes: str, label: str = "E1") -> dict:
    return {"answerable": True, "abstain_reason": None, "summary": "Short.",
            "unanswered_aspects": [],
            "claims": [{"text": f"claim {q[:10]}", "supporting_quotes": [
                {"evidence_label": label, "quote": q}]} for q in quotes]}


COMPLETE = {"complete": True, "kind": "none", "question": "", "sides": []}


class Model:
    """Answers each prompt from its script; a draft of None overruns the schema."""

    def __init__(self, drafts=(), analysis=None, review=COMPLETE):
        self.drafts = list(drafts)
        self.analysis = {"intent": "lookup", "search_queries": [], "referenced_articles": [],
                         "referenced_annexes": [], "reply": "", "legal_test": "none",
                         "sides": []} | (analysis or {})
        self.review = review
        self.calls: list[dict] = []

    def structured(self, **call) -> Completion:
        self.calls.append(call)
        data = {
            prompts.ANALYSIS_SYSTEM: self.analysis,
            prompts.REVIEW_SYSTEM: self.review,
            prompts.WRAPUP_SYSTEM: {"summary": "Both parts, together.", "unanswered_aspects": []},
        }.get(call["system"])
        if data is None:
            data = self.drafts.pop(0)
        if data is None:
            raise SchemaValidationFailed("output_parse_failed")
        return Completion(data=data, model=call["model"])

    def asked(self, system: str) -> list[dict]:
        return [c for c in self.calls if c["system"] == system]


class Embedder:
    def embed_query(self, query):
        return [0.0]


@pytest.fixture
def searches(monkeypatch):
    """Stub search: each query finds one unit; scores and outlines are per test."""
    seen: list[dict] = []
    scores = {"default": 0.9}

    def retrieve(session, **search):
        seen.append(search)
        return [search["query"]]

    def rerank(query, candidates):
        score = scores.get(query, scores["default"])
        return RerankResult(kept=list(candidates) if score else [], best_score=score)

    def structural_search(session, articles, annexes, parts):
        seen.append({"named": articles + annexes})
        return ["legal test"]

    monkeypatch.setattr(nodes, "retrieve", retrieve)
    monkeypatch.setattr(nodes, "structural_search", structural_search)
    monkeypatch.setattr(nodes, "rerank", rerank)
    monkeypatch.setattr(nodes, "expand_to_units", lambda session, kept: [
        UNITS.get(kept[0], UNITS["default"])])
    monkeypatch.setattr(nodes, "with_structure", lambda session, units, budget: units)
    monkeypatch.setattr(nodes, "outline", lambda session, u: [unit("Article 8 X\nArticle 9 Y",
                                                                   99, "Section 2 (outline)")])
    return seen, scores


def run(model: Model, question: str = "What must providers do under Article 50?"):
    state = nodes.run_pipeline(question, None, model, Embedder())
    return state, [p.step for p in state.progress]


class TestOneRound:
    @pytest.mark.parametrize(("drafts", "verdict", "retries"), [
        ([answer(GOOD)], "answered", []),
        ([None, answer(GOOD)], "answered", ["retrying"]),
        ([answer(BAD), answer(GOOD)], "answered", ["repairing"]),
        ([answer(BAD), answer(BAD)], "abstained", ["repairing"]),
        ([None, None], "abstained", ["retrying"]),
    ])
    def test_drafts_are_retried_repaired_and_gated(self, searches, drafts, verdict, retries):
        state, steps = run(Model(drafts))
        assert state.verdict == verdict
        assert [s for s in steps if s in ("retrying", "repairing")] == retries

    def test_an_overrun_twice_says_so(self, searches):
        state, _ = run(Model([None, None]))
        assert state.abstain_reason == nodes.OVERRAN_REASON

    def test_a_complete_answer_is_reviewed_once_and_not_wrapped_up(self, searches):
        model = Model([answer(GOOD)])
        state, steps = run(model)
        assert steps[-2:] == ["reviewing", "done"]
        assert len(model.asked(prompts.REVIEW_SYSTEM)) == 1
        assert not model.asked(prompts.WRAPUP_SYSTEM)
        assert [c.text for c in state.claims] == ["claim outputs of"]
        assert (state.document_version_ids, state.retrieved_chunk_ids) == ([7], [50])

    def test_nothing_relevant_abstains_before_drafting(self, searches):
        searches[1]["default"] = 0.0
        model = Model()
        state, steps = run(model)
        assert state.abstain_reason == nodes.NO_EVIDENCE_REASON
        assert "generating" not in steps and not model.asked(prompts.REVIEW_SYSTEM)

    def test_an_abstained_answer_is_not_reviewed(self, searches):
        model = Model([answer(BAD), answer(BAD)])
        run(model)
        assert not model.asked(prompts.REVIEW_SYSTEM)


class TestPathsByQuestionType:
    def test_a_comparison_searches_each_side_in_parallel(self, searches):
        seen, _ = searches
        model = Model([answer(GOOD, label="E1") | {"claims": answer(GOOD)["claims"]
                                                   + answer(DEFINITION, label="E2")["claims"]}],
                      analysis={"intent": "comparison", "sides": ["watermarks", "deep fakes"]})
        state, steps = run(model, "How do watermarks differ from deep fakes?")
        assert sorted(s["query"] for s in seen) == ["deep fakes", "watermarks"]
        assert steps.count("retrieving") == 2
        assert {u.citation_label for u in state.evidence} == {"Article 3", "Article 50"}
        assert state.verdict == "answered" and len(state.claims) == 2

    def test_an_applicability_question_searches_its_legal_test(self, searches):
        seen, _ = searches
        criterion = {"criterion": "Marked outputs", "status": "needs_user_input",
                     "explanation": "Depends.", "supporting_quotes": [
                         {"evidence_label": "E1", "quote": GOOD}]}
        model = Model([{"framing": "It depends.", "criteria": [criterion],
                        "follow_up_questions": [], "unanswered_aspects": []}],
                      analysis={"intent": "applicability", "legal_test": "high_risk",
                                "referenced_articles": ["50"]})
        state, _ = run(model, "Is my chatbot high-risk?")
        # The question's own search, and the legal test's provisions searched on their own.
        assert {"named": ["6", "III"]} in seen
        assert [s["articles"] for s in seen if "articles" in s] == [["50"]]
        assert model.calls[1]["response_format"] is ASSESSMENT_SCHEMA
        assert [c["criterion"] for c in state.criteria] == ["Marked outputs"]
        assert state.claims == []  # criteria are shown as criteria, not again as claims
        assert not model.asked(prompts.REVIEW_SYSTEM)  # its open points are the user's facts

    def test_an_overview_reads_the_outline_of_the_whole_group(self, searches):
        model = Model([answer(GOOD)], analysis={"intent": "overview"})
        state, _ = run(model, "What are all the requirements for high-risk AI systems?")
        assert [u.citation_label for u in state.evidence] == ["Article 50", "Section 2 (outline)"]
        assert "Article 8 X" in model.calls[1]["user"]


class TestReviewAndWrapUp:
    FOLLOW_UP = {"complete": False, "kind": "lookup", "question": "What is a deep fake?",
                 "sides": []}

    def test_a_missing_part_is_answered_in_a_second_round_and_wrapped_up(
        self, searches, monkeypatch
    ):
        seen, _ = searches
        monkeypatch.setitem(UNITS, "What is a deep fake?", UNITS["deep fakes"])
        model = Model([answer(GOOD), answer(DEFINITION)], review=self.FOLLOW_UP)
        state, steps = run(model)
        assert [s["query"] for s in seen][1] == "What is a deep fake?"
        assert [c.text for c in state.claims] == ["claim outputs of", "claim AI-generat"]
        assert state.summary == "Both parts, together."
        assert "extending" in steps and steps[-2:] == ["wrapping_up", "done"]
        assert len(model.asked(prompts.REVIEW_SYSTEM)) == 1  # max_rounds: no third round
        assert state.report.quotes_total == 2
        assert {u.citation_label for u in state.evidence} == {"Article 3", "Article 50"}

    def test_a_follow_up_that_finds_nothing_keeps_the_first_answer(self, searches):
        _, scores = searches
        scores["What is a deep fake?"] = 0.0
        model = Model([answer(GOOD)], review=self.FOLLOW_UP)
        state, steps = run(model)
        assert state.verdict == "answered" and state.summary == "Short."
        assert [c.text for c in state.claims] == ["claim outputs of"]
        assert not model.asked(prompts.WRAPUP_SYSTEM) and steps[-1] == "done"
