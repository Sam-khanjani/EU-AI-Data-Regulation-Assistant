"""Evaluation tests.

Two layers:

* **Scoring logic** -- pure functions over API payloads, always run. If the scorer is wrong,
  every number it produces is worthless, so it is tested like any other code.
* **The live suite** -- runs the real pipeline and asserts reliability thresholds. Skipped
  unless both API keys and an embedded corpus are present, so the default test run stays
  fast and offline.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from euaia.config import settings
from euaia.db.session import engine
from eval.harness import Case, evaluate_response, load_cases, run, save_summary, summarise

# --------------------------------------------------------------- scoring logic


class TestLoadCases:
    def test_question_set_loads(self):
        cases = load_cases()
        assert len(cases) >= 20

    def test_ids_are_unique(self):
        ids = [c.id for c in load_cases()]
        assert len(ids) == len(set(ids))

    def test_suite_contains_enough_refusals(self):
        cases = load_cases()
        refusals = [c for c in cases if c.should_abstain]
        # A suite that only asks answerable questions cannot detect over-answering.
        assert len(refusals) / len(cases) >= 0.25

    def test_every_answerable_case_declares_expectations(self):
        for case in load_cases():
            if not case.should_abstain:
                assert case.expected_articles or case.expected_annexes, (
                    f"{case.id}: an answerable case needs a recall target"
                )

    def test_bait_cases_all_expect_abstention(self):
        for case in load_cases():
            if case.category in ("bait", "out_of_corpus", "insufficient"):
                assert case.should_abstain, f"{case.id} must expect abstention"


class TestEvaluateResponse:
    def case(self, **kw):
        base = {
            "id": "t", "category": "lookup", "question": "q",
            "expected_articles": ["6"], "expected_annexes": [],
            "should_abstain": False, "expect_criteria": False, "notes": "",
        }
        base.update(kw)
        return Case(**base)

    def payload(self, **kw):
        base = {
            "verdict": "answered", "summary": "", "claims": [], "criteria": [],
            "coverage": 1.0, "quotes_total": 0, "quotes_dropped": 0, "latency_ms": 10,
        }
        base.update(kw)
        return base

    def test_recall_counts_expected_articles(self):
        payload = self.payload(
            claims=[{"text": "x", "citations": [{"citation": "Article 6", "quote": "q"}]}]
        )
        assert evaluate_response(self.case(), payload).recall == 1.0

    def test_recall_is_partial_when_an_expectation_is_missed(self):
        case = self.case(expected_articles=["6"], expected_annexes=["III"])
        payload = self.payload(
            claims=[{"text": "x", "citations": [{"citation": "Article 6", "quote": "q"}]}]
        )
        assert evaluate_response(case, payload).recall == 0.5

    def test_paragraph_citation_counts_towards_its_article(self):
        payload = self.payload(
            claims=[{"text": "x", "citations": [{"citation": "Paragraph 6(2)", "quote": "q"}]}]
        )
        assert evaluate_response(self.case(), payload).recall == 1.0

    def test_annex_citations_are_tracked(self):
        case = self.case(expected_articles=[], expected_annexes=["III"])
        payload = self.payload(
            claims=[{"text": "x", "citations": [{"citation": "Annex III", "quote": "q"}]}]
        )
        result = evaluate_response(case, payload)
        assert result.cited_annexes == {"III"} and result.recall == 1.0

    def test_abstaining_when_expected_is_correct(self):
        case = self.case(should_abstain=True)
        result = evaluate_response(case, self.payload(verdict="abstained"))
        assert result.abstention_correct and result.abstained

    def test_answering_a_refusal_case_is_marked_wrong(self):
        case = self.case(should_abstain=True)
        result = evaluate_response(case, self.payload(verdict="answered"))
        assert not result.abstention_correct

    def test_recall_not_computed_for_refusal_cases(self):
        case = self.case(should_abstain=True)
        assert evaluate_response(case, self.payload(verdict="abstained")).recall is None

    def test_quote_accuracy(self):
        result = evaluate_response(
            self.case(), self.payload(quotes_total=10, quotes_dropped=3)
        )
        assert result.quote_accuracy == pytest.approx(0.7)

    def test_quote_accuracy_is_none_without_quotes(self):
        assert evaluate_response(self.case(), self.payload()).quote_accuracy is None

    def test_verdict_leak_is_detected_in_assessments(self):
        case = self.case(expect_criteria=True)
        payload = self.payload(
            criteria=[
                {
                    "criterion": "Annex III area",
                    "status": "met",
                    "explanation": "Your system is a high-risk AI system under Annex III.",
                    "citations": [{"citation": "Annex III", "quote": "q"}],
                }
            ]
        )
        assert evaluate_response(case, payload).criteria_verdict_leak

    def test_neutral_assessment_is_not_flagged(self):
        case = self.case(expect_criteria=True)
        payload = self.payload(
            criteria=[
                {
                    "criterion": "Annex III area",
                    "status": "needs_user_input",
                    "explanation": "Whether this applies depends on how the tool is used.",
                    "citations": [{"citation": "Annex III", "quote": "q"}],
                }
            ]
        )
        assert not evaluate_response(case, payload).criteria_verdict_leak


class TestSummarise:
    def test_flags_false_answers(self):
        case = Case(id="bad", category="bait", question="q", should_abstain=True)
        result = evaluate_response(case, {"verdict": "answered", "claims": [], "criteria": []})
        summary = summarise([result])
        assert summary["false_answers"] == ["bad"]
        assert summary["abstention_accuracy"] == 0.0

    def test_perfect_refusal_scores_full_marks(self):
        case = Case(id="good", category="bait", question="q", should_abstain=True)
        result = evaluate_response(case, {"verdict": "abstained", "claims": [], "criteria": []})
        summary = summarise([result])
        assert summary["abstention_accuracy"] == 1.0
        assert summary["false_answers"] == []


# ------------------------------------------------------------------ live suite


def _corpus_is_embedded() -> bool:
    try:
        with engine.connect() as conn:
            return bool(conn.execute(text("SELECT 1 FROM chunk LIMIT 1")).first())
    except Exception:  # noqa: BLE001
        return False


live = pytest.mark.skipif(
    not (settings.groq_api_key and settings.google_api_key and _corpus_is_embedded()),
    reason="live evaluation needs GROQ_API_KEY, GOOGLE_API_KEY and an embedded corpus",
)


@live
class TestLiveEvaluation:
    """Reliability thresholds. These are the numbers the project stands on."""

    @pytest.fixture(scope="class")
    @classmethod
    def results(cls):
        results = run(None, load_cases())
        save_summary(summarise(results))
        return results

    def test_no_case_errored(self, results):
        errors = [(r.case.id, r.error) for r in results if r.error]
        assert not errors, f"cases raised: {errors}"

    def test_never_answers_a_question_it_should_refuse(self, results):
        summary = summarise(results)
        assert summary["false_answers"] == [], (
            "answering an out-of-corpus or baited question is the most serious failure "
            "this suite can detect"
        )

    def test_answers_questions_it_should_answer(self, results):
        summary = summarise(results)
        assert summary["answer_rate_on_answerable"] >= 0.8

    def test_citation_recall(self, results):
        assert summarise(results)["citation_recall"] >= 0.7

    def test_quote_accuracy_is_acceptable(self, results):
        # Measures the model, before the safety net drops anything.
        assert summarise(results)["quote_accuracy"] >= 0.8

    def test_applicability_answers_never_state_a_verdict(self, results):
        leaks = summarise(results)["verdict_leaks"]
        assert leaks == [], f"assessment answers stated a legal conclusion: {leaks}"

    def test_applicability_answers_produce_criteria(self, results):
        for r in results:
            if r.case.expect_criteria and r.answered:
                assert r.produced_criteria, f"{r.case.id} answered without a criteria checklist"

    def test_latency_is_reasonable(self, results):
        assert summarise(results)["latency_p95_ms"] < 60_000
