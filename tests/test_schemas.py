"""Every schema must satisfy Groq's strict-mode rules.

Strict mode rejects a schema at request time with a 400, which is a slow and confusing way
to discover a missing ``additionalProperties: false`` on a nested object. These tests
enforce the rules locally.

The rules (from Groq's structured-outputs documentation):

* every object sets ``additionalProperties: false``
* every property of an object appears in its ``required`` array -- there are no optional
  fields; optionality is expressed as a nullable union such as ``{"type": ["string", "null"]}``
"""

from __future__ import annotations

from typing import Any

import pytest

from euaia.llm import schemas

ALL_SCHEMAS = {
    "query_analysis": schemas.QUERY_ANALYSIS_SCHEMA,
    "rerank": schemas.RERANK_SCHEMA,
    "grounded_answer": schemas.ANSWER_SCHEMA,
    "criteria_assessment": schemas.ASSESSMENT_SCHEMA,
}


def walk_objects(node: Any, path: str = "$"):
    """Yield ``(path, object_schema)`` for every object node in a JSON schema."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield path, node
        for key, value in node.items():
            yield from walk_objects(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from walk_objects(value, f"{path}[{i}]")


@pytest.mark.parametrize("name", sorted(ALL_SCHEMAS))
class TestStrictModeCompliance:
    def test_envelope_is_correct(self, name):
        envelope = ALL_SCHEMAS[name]
        assert envelope["type"] == "json_schema"
        assert envelope["json_schema"]["strict"] is True
        assert envelope["json_schema"]["name"] == name

    def test_every_object_forbids_additional_properties(self, name):
        root = ALL_SCHEMAS[name]["json_schema"]["schema"]
        for path, obj in walk_objects(root):
            assert obj.get("additionalProperties") is False, (
                f"{name}: object at {path} must set additionalProperties: false"
            )

    def test_every_property_is_required(self, name):
        root = ALL_SCHEMAS[name]["json_schema"]["schema"]
        for path, obj in walk_objects(root):
            properties = set(obj.get("properties", {}))
            required = set(obj.get("required", []))
            missing = properties - required
            assert not missing, (
                f"{name}: object at {path} has optional properties {sorted(missing)}; "
                "strict mode requires all of them in 'required' (use a nullable union "
                "for optional values)"
            )

    def test_required_names_all_exist(self, name):
        root = ALL_SCHEMAS[name]["json_schema"]["schema"]
        for path, obj in walk_objects(root):
            properties = set(obj.get("properties", {}))
            phantom = set(obj.get("required", [])) - properties
            assert not phantom, f"{name}: {path} requires undefined properties {sorted(phantom)}"


class TestAnswerSchemaEnforcesGrounding:
    """The answer schema is the enforcement mechanism, so assert its shape explicitly."""

    def test_claims_cannot_omit_supporting_quotes(self):
        claim = ALL_SCHEMAS["grounded_answer"]["json_schema"]["schema"]["properties"]["claims"][
            "items"
        ]
        assert "supporting_quotes" in claim["required"], (
            "a claim must be structurally unable to exist without a quotes array"
        )

    def test_each_quote_must_name_its_evidence_block(self):
        quote = ALL_SCHEMAS["grounded_answer"]["json_schema"]["schema"]["properties"]["claims"][
            "items"
        ]["properties"]["supporting_quotes"]["items"]
        assert set(quote["required"]) == {"evidence_label", "quote"}

    def test_abstain_reason_is_nullable_not_optional(self):
        prop = ALL_SCHEMAS["grounded_answer"]["json_schema"]["schema"]["properties"][
            "abstain_reason"
        ]
        assert prop["type"] == ["string", "null"]


class TestAssessmentSchemaRefusesVerdicts:
    def test_status_enum_offers_no_overall_verdict(self):
        criterion = ALL_SCHEMAS["criteria_assessment"]["json_schema"]["schema"]["properties"][
            "criteria"
        ]["items"]
        statuses = set(criterion["properties"]["status"]["enum"])
        assert statuses == {"met", "not_met", "needs_user_input"}

    def test_there_is_no_top_level_conclusion_field(self):
        root = ALL_SCHEMAS["criteria_assessment"]["json_schema"]["schema"]
        # The schema must not give the model anywhere to state "your system is high-risk".
        forbidden = {"verdict", "conclusion", "is_high_risk", "determination", "answer"}
        assert not (forbidden & set(root["properties"]))


class TestIntents:
    def test_analysis_enum_matches_declared_intents(self):
        prop = ALL_SCHEMAS["query_analysis"]["json_schema"]["schema"]["properties"]["intent"]
        assert set(prop["enum"]) == set(schemas.INTENTS)
