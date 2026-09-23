"""The hosted reranking provider, offline: the HTTP call is replaced by a fake."""

from __future__ import annotations

import httpx
import pytest

from euaia.config import Settings, settings
from euaia.retrieval import rerank as rr
from euaia.retrieval.hybrid import Candidate


def candidate(i: int, text: str) -> Candidate:
    return Candidate(
        chunk_id=i, unit_id=i, unit_path=f"ART_{i}", unit_type="article", unit_number=str(i),
        heading=None, document_version_id=1, version_label="v", source_key="eu-ai-act",
        authority="law", text=text, chunk_text=text,
    )


@pytest.fixture(autouse=True)
def openrouter(monkeypatch):
    monkeypatch.setattr(Settings, "rerank_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(rr.time, "sleep", lambda _s: None)


def fake_post(monkeypatch, *responses):
    calls = []

    def post(url, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json})
        item = responses[min(len(calls), len(responses)) - 1]
        if isinstance(item, Exception):
            raise item
        status, body = item
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(rr.httpx, "post", post)
    return calls


def results(*scores):
    # OpenRouter returns results sorted by score, not in input order.
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    return {"results": [{"index": i, "relevance_score": s} for i, s in ranked]}


class TestScoring:
    def test_scores_come_back_in_input_order(self, monkeypatch):
        fake_post(monkeypatch, (200, results(0.1, 0.9, 0.5)))
        assert rr.score_passages("q", ["a", "b", "c"]) == [0.1, 0.9, 0.5]

    def test_one_request_carries_every_passage_and_the_configured_model(self, monkeypatch):
        calls = fake_post(monkeypatch, (200, results(0.1, 0.2)))
        rr.score_passages("Which AI practices are prohibited?", ["a", "b"])
        assert len(calls) == 1
        assert calls[0]["url"] == "https://openrouter.ai/api/v1/rerank"
        assert calls[0]["headers"] == {"Authorization": "Bearer test-key"}
        assert calls[0]["json"] == {
            "model": settings.openrouter_rerank_model,
            "query": "Which AI practices are prohibited?",
            "documents": ["a", "b"],
        }

    def test_rerank_keeps_the_best_candidates_by_the_hosted_scores(self, monkeypatch):
        fake_post(monkeypatch, (200, results(0.01, 0.95, 0.40, 0.80, 0.02)))
        pool = [candidate(i, f"text {i}") for i in range(5)]
        result = rr.rerank("q", pool, keep=2, min_score=0.005)
        assert [c.chunk_id for c in result.kept] == [1, 3]
        assert result.best_score == 0.95

    def test_the_local_model_is_never_loaded(self, monkeypatch):
        fake_post(monkeypatch, (200, results(0.3)))

        def boom(*_a, **_k):
            raise AssertionError("local model loaded while provider is openrouter")

        monkeypatch.setattr(rr, "load_model", boom)
        rr.score_passages("q", ["a"])
        rr.warm()


class TestConfiguredInCodeOnly:
    def test_environment_variables_cannot_change_the_reranker(self, monkeypatch):
        # The reranker decides what every answer is grounded in, so it is chosen in
        # config.py -- under version control -- and not per machine through .env.
        monkeypatch.setenv("RERANK_PROVIDER", "local")
        monkeypatch.setenv("RERANK_MODEL", "some/other-model")
        monkeypatch.setenv("OPENROUTER_RERANK_MODEL", "some/other-hosted-model")
        fresh = Settings(_env_file=None)
        assert fresh.rerank_provider == Settings.rerank_provider
        assert fresh.rerank_model == Settings.rerank_model
        assert fresh.openrouter_rerank_model == Settings.openrouter_rerank_model
        assert not {"rerank_provider", "rerank_model", "openrouter_rerank_model"} & set(
            Settings.model_fields
        )

    def test_a_model_can_be_passed_without_touching_the_configuration(self, monkeypatch):
        calls = fake_post(monkeypatch, (200, results(0.5)))
        rr.score_passages("q", ["a"], provider="openrouter", model="other/reranker")
        assert calls[0]["json"]["model"] == "other/reranker"


class TestFailures:
    def test_a_missing_key_is_reported_before_any_request(self, monkeypatch):
        monkeypatch.setattr(settings, "openrouter_api_key", "")
        calls = fake_post(monkeypatch, (200, results(0.1)))
        with pytest.raises(rr.RerankerUnavailable, match="OPENROUTER_API_KEY"):
            rr.score_passages("q", ["a"])
        assert calls == []

    def test_transient_errors_are_retried(self, monkeypatch):
        calls = fake_post(
            monkeypatch,
            httpx.ConnectError("down"),
            (503, {"error": "busy"}),
            (200, results(0.7)),
        )
        assert rr.score_passages("q", ["a"]) == [0.7]
        assert len(calls) == 3

    def test_a_rate_limit_that_persists_is_raised_not_swallowed(self, monkeypatch):
        calls = fake_post(monkeypatch, (429, {"error": "free-models-per-day"}))
        with pytest.raises(rr.RerankerUnavailable, match="429"):
            rr.score_passages("q", ["a"])
        assert len(calls) == 3

    def test_a_bad_request_is_not_retried(self, monkeypatch):
        calls = fake_post(monkeypatch, (401, {"error": "invalid key"}))
        with pytest.raises(rr.RerankerUnavailable, match="401"):
            rr.score_passages("q", ["a"])
        assert len(calls) == 1

    def test_a_passage_left_unscored_is_an_error_not_a_zero(self, monkeypatch):
        fake_post(monkeypatch, (200, {"results": [{"index": 0, "relevance_score": 0.9}]}))
        with pytest.raises(rr.RerankerUnavailable, match="scored 1 of 2"):
            rr.score_passages("q", ["a", "b"])
