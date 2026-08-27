"""Strict JSON schemas for every model call.

Groq's strict mode uses constrained decoding, so a response is *guaranteed* to match the
schema. That turns the schema into an enforcement mechanism rather than a request: it is
structurally impossible for the answering model to emit a claim without a
``supporting_quotes`` array, because the decoder will not produce tokens that leave the
schema.

Strict mode imposes rules that these schemas must satisfy:

* every property listed in ``required`` -- there are no optional fields
* ``additionalProperties: false`` on every object
* optional values expressed as a nullable union, ``{"type": ["string", "null"]}``
* streaming and tool use unavailable (which is why answers are rendered whole)

What the schema cannot do is make the *contents* true. A model can still write invented
words into a quote field. That is what ``euaia.verify.citations`` exists to catch.
"""

from __future__ import annotations

from typing import Any

INTENTS = ("lookup", "applicability", "comparison", "out_of_scope")


def _schema(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a JSON schema in Groq's strict ``response_format`` envelope."""
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


# --------------------------------------------------------------------- analysis

QUERY_ANALYSIS_SCHEMA = _schema(
    "query_analysis",
    {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "intent",
            "reasoning",
            "search_queries",
            "referenced_articles",
            "referenced_annexes",
        ],
        "properties": {
            "intent": {
                "type": "string",
                "enum": list(INTENTS),
                "description": (
                    "lookup: asks what a provision says. "
                    "applicability: asks whether rules apply to the user's own system. "
                    "comparison: asks how provisions or versions differ. "
                    "out_of_scope: not answerable from the EU AI Act corpus."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence justifying the intent label.",
            },
            "search_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "One to four retrieval queries covering the question, using the "
                    "Regulation's own vocabulary."
                ),
            },
            "referenced_articles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Article numbers named explicitly, e.g. ['6', '50'].",
            },
            "referenced_annexes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Annex numbers named explicitly, e.g. ['III'].",
            },
        },
    },
)


# --------------------------------------------------------------------- rerank

RERANK_SCHEMA = _schema(
    "rerank",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["rankings"],
        "properties": {
            "rankings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["label", "score"],
                    "properties": {
                        "label": {"type": "string"},
                        "score": {
                            "type": "integer",
                            "description": (
                                "0-10. 0 = irrelevant, 10 = directly and completely answers "
                                "the question."
                            ),
                        },
                    },
                },
            }
        },
    },
)


# --------------------------------------------------------------------- answer

ANSWER_SCHEMA = _schema(
    "grounded_answer",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["answerable", "summary", "claims", "unanswered_aspects", "abstain_reason"],
        "properties": {
            "answerable": {
                "type": "boolean",
                "description": "False if the evidence does not support an answer.",
            },
            "summary": {
                "type": "string",
                "description": (
                    "One or two sentences framing the answer. Must not introduce any fact "
                    "that is not also stated in a claim."
                ),
            },
            "claims": {
                "type": "array",
                "description": (
                    "Each substantive statement, separately supported. A claim with no "
                    "verifiable quote will be discarded before the user sees it."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "supporting_quotes"],
                    "properties": {
                        "text": {"type": "string"},
                        "supporting_quotes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["evidence_label", "quote"],
                                "properties": {
                                    "evidence_label": {
                                        "type": "string",
                                        "description": "Exactly as labelled, e.g. 'E3'.",
                                    },
                                    "quote": {
                                        "type": "string",
                                        "description": (
                                            "A span copied CHARACTER FOR CHARACTER from that "
                                            "evidence block. It is checked against the source "
                                            "text; anything reworded is discarded."
                                        ),
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "unanswered_aspects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Parts of the question the evidence does not cover.",
            },
            "abstain_reason": {
                "type": ["string", "null"],
                "description": "Why no answer is possible, when answerable is false.",
            },
        },
    },
)


# ------------------------------------------------------- structured self-assessment

ASSESSMENT_SCHEMA = _schema(
    "criteria_assessment",
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["framing", "criteria", "follow_up_questions", "unanswered_aspects"],
        "properties": {
            "framing": {
                "type": "string",
                "description": (
                    "What the Regulation makes this determination depend on. States no "
                    "conclusion about the user's system."
                ),
            },
            "criteria": {
                "type": "array",
                "description": "The test the Regulation actually sets out, broken into criteria.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["criterion", "status", "explanation", "supporting_quotes"],
                    "properties": {
                        "criterion": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["met", "not_met", "needs_user_input"],
                            "description": (
                                "Use 'needs_user_input' unless the user stated a fact that "
                                "settles it. Never guess."
                            ),
                        },
                        "explanation": {"type": "string"},
                        "supporting_quotes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["evidence_label", "quote"],
                                "properties": {
                                    "evidence_label": {"type": "string"},
                                    "quote": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
            "follow_up_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific questions whose answers would resolve open criteria.",
            },
            "unanswered_aspects": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
)
