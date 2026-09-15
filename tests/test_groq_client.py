"""Groq client behaviour that the pipeline depends on.

No network: a fake SDK client stands in, so these assert how we *handle* the API rather
than what the API does.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from euaia.llm.groq_client import (
    GroqClient,
    LLMError,
    SchemaValidationFailed,
    Usage,
    _retry_after_seconds,
)
from euaia.llm.ratelimit import Limits

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "t",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
    },
}


def _response(content: str, finish_reason: str = "stop", prompt=10, completion=5):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason
            )
        ],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


class FakeCompletions:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = 0
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        result = self.behaviour(self.calls) if callable(self.behaviour) else self.behaviour
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def client(monkeypatch):
    def build(behaviour):
        c = GroqClient(
            api_key="test-key",
            # Generous limits so the rate limiter never interferes with these tests.
            limits=Limits(requests_per_minute=1000, tokens_per_minute=1_000_000),
        )
        fake = FakeCompletions(behaviour)
        c._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
        return c, fake

    return build


class TestStructuredCall:
    def test_parses_the_response(self, client):
        c, _ = client(_response(json.dumps({"ok": True})))
        result = c.structured(system="s", user="u", response_format=SCHEMA)
        assert result.data == {"ok": True}

    def test_reports_token_usage(self, client):
        c, _ = client(_response(json.dumps({"ok": True}), prompt=120, completion=34))
        result = c.structured(system="s", user="u", response_format=SCHEMA)
        assert result.usage.prompt_tokens == 120
        assert result.usage.completion_tokens == 34
        assert result.usage.total_tokens == 154

    def test_sends_strict_schema_through_unchanged(self, client):
        # The whole reason for using the raw SDK: this flag must reach the API intact.
        c, fake = client(_response(json.dumps({"ok": True})))
        c.structured(system="s", user="u", response_format=SCHEMA)
        assert fake.last_kwargs["response_format"]["json_schema"]["strict"] is True

    def test_passes_reasoning_effort_only_when_set(self, client):
        c, fake = client(_response(json.dumps({"ok": True})))
        c.structured(system="s", user="u", response_format=SCHEMA)
        assert "reasoning_effort" not in fake.last_kwargs

        c2, fake2 = client(_response(json.dumps({"ok": True})))
        c2.structured(system="s", user="u", response_format=SCHEMA, reasoning_effort="low")
        assert fake2.last_kwargs["reasoning_effort"] == "low"


class TestTruncation:
    def test_a_truncated_answer_raises_rather_than_parsing_a_fragment(self, client):
        # Losing claims silently would be worse than failing: the user would see a
        # confidently partial answer with no sign anything was missing.
        c, _ = client(_response('{"ok": tr', finish_reason="length"))
        with pytest.raises(LLMError, match="truncated"):
            c.structured(system="s", user="u", response_format=SCHEMA)


class TestSchemaRejection:
    def _bad_request(self):
        exc = Exception(
            "Error code: 400 - {'error': {'code': 'json_validate_failed', "
            "'message': 'Generated JSON does not match the expected schema'}}"
        )
        exc.__class__ = type("BadRequestError", (Exception,), {})
        return exc

    def test_json_validate_failed_becomes_a_typed_error(self, client, monkeypatch):
        from euaia.llm import groq_client as mod

        class FakeBadRequest(Exception):
            pass

        monkeypatch.setattr(mod, "BadRequestError", FakeBadRequest)
        err = FakeBadRequest(
            "400 - {'error': {'code': 'json_validate_failed', 'message': "
            "'missing properties: unanswered_aspects'}}"
        )
        c, _ = client(err)
        with pytest.raises(SchemaValidationFailed, match="did not satisfy the schema"):
            c.structured(system="s", user="u", response_format=SCHEMA)

    def test_a_deterministic_rejection_is_not_retried(self, client, monkeypatch):
        from euaia.llm import groq_client as mod

        class FakeBadRequest(Exception):
            pass

        monkeypatch.setattr(mod, "BadRequestError", FakeBadRequest)
        err = FakeBadRequest("400 json_validate_failed missing properties")
        c, fake = client(err)
        with pytest.raises(SchemaValidationFailed):
            c.structured(system="s", user="u", response_format=SCHEMA)
        assert fake.calls == 1, "retrying an identical request just fails identically"

    def test_text_instead_of_json_is_also_a_rejection_and_not_retried(self, client, monkeypatch):
        # Seen live: the small model wrote its reasoning where the object belonged, and four
        # identical retries at temperature 0 failed identically before the error surfaced.
        from euaia.llm import groq_client as mod

        class FakeBadRequest(Exception):
            pass

        monkeypatch.setattr(mod, "BadRequestError", FakeBadRequest)
        err = FakeBadRequest(
            "400 - {'error': {'code': 'output_parse_failed', 'failed_generation': 'Need to'}}"
        )
        c, fake = client(err)
        with pytest.raises(SchemaValidationFailed):
            c.structured(system="s", user="u", response_format=SCHEMA)
        assert fake.calls == 1

    def test_other_bad_requests_propagate(self, client, monkeypatch):
        from euaia.llm import groq_client as mod

        class FakeBadRequest(Exception):
            pass

        monkeypatch.setattr(mod, "BadRequestError", FakeBadRequest)
        c, _ = client(FakeBadRequest("400 - model not found"))
        with pytest.raises(FakeBadRequest):
            c.structured(system="s", user="u", response_format=SCHEMA)


class TestRetryAfter:
    def test_reads_the_retry_after_header(self):
        exc = SimpleNamespace(response=SimpleNamespace(headers={"retry-after": "12"}))
        assert _retry_after_seconds(exc) == 12.0

    def test_strips_a_trailing_unit(self):
        exc = SimpleNamespace(
            response=SimpleNamespace(headers={"x-ratelimit-reset-tokens": "7.5s"})
        )
        assert _retry_after_seconds(exc) == 7.5

    def test_falls_back_when_no_header_is_present(self):
        exc = SimpleNamespace(response=SimpleNamespace(headers={}))
        assert _retry_after_seconds(exc, default=20.0) == 20.0

    def test_survives_a_response_without_headers(self):
        assert _retry_after_seconds(SimpleNamespace(), default=9.0) == 9.0


class TestUsageAggregation:
    def test_adds_across_pipeline_stages(self):
        total = Usage()
        total.add(Usage(prompt_tokens=100, completion_tokens=20, calls=1, latency_ms=300))
        total.add(Usage(prompt_tokens=50, completion_tokens=10, calls=1, waited_ms=5000))
        assert total.total_tokens == 180
        assert total.calls == 2
        assert total.waited_ms == 5000


class TestConfiguration:
    def test_missing_key_is_an_actionable_error(self, monkeypatch):
        # An empty argument falls back to settings, so clear that too.
        from euaia.llm import groq_client as mod

        monkeypatch.setattr(mod.settings, "groq_api_key", "")
        with pytest.raises(LLMError, match="GROQ_API_KEY"):
            GroqClient(api_key="")
