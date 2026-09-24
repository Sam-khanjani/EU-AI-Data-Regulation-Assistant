"""The intent classifier routes; it must not veto the corpus.

`out_of_scope` ends a request immediately -- nothing is retrieved and no answer is
attempted. That makes a wrong call expensive: the evaluation caught
*"What does the AI Act say about AI literacy?"* being refused in 181ms, without retrieval,
even though AI literacy is Article 4.

The prompt now describes what the Act covers and tells the model to be reluctant to refuse.
This deterministic override is the second layer: if the user names the Act or one of its
provisions, the question is about the Act, whatever the classifier decided.

A false positive here is cheap by design. The override forces retrieval, not an answer -- if
nothing relevant is found the sufficiency gate still abstains. So it costs a search, never
correctness.
"""

from __future__ import annotations

import pytest

from euaia.graph.nodes import _corrected_intent


class TestOverrideRescuesAnsweredQuestions:
    @pytest.mark.parametrize(
        "question",
        [
            "What does the AI Act say about AI literacy?",
            "What does the Artificial Intelligence Act require of deployers?",
            "What does Article 50 require?",
            "Does Article 6 apply to my system?",
            "What is listed in Annex III?",
            "What does Recital 27 say?",
            "What penalties does Regulation (EU) 2024/1689 impose?",
            "What does this Regulation say about sandboxes?",
        ],
    )
    def test_naming_the_act_overrides_a_refusal(self, question):
        assert _corrected_intent("out_of_scope", question) == "lookup"

    def test_a_greeting_that_names_the_act_is_a_question(self):
        # Small talk must never be how a real question goes unanswered.
        assert _corrected_intent("greeting", "Hi! What does Article 5 say?") == "lookup"


class TestGenuineRefusalsSurvive:
    @pytest.mark.parametrize(
        "question",
        [
            "What is the lawful basis for processing personal data under the GDPR?",
            "When must an organisation appoint a Data Protection Officer?",
            "How do I fix a segmentation fault in my C++ program?",
            "How many companies have been fined so far?",
            "What are the best open-source language models?",
        ],
    )
    def test_questions_about_something_else_are_still_refused(self, question):
        assert _corrected_intent("out_of_scope", question) == "out_of_scope"


class TestOtherIntentsAreUntouched:
    @pytest.mark.parametrize("intent", ["lookup", "applicability", "comparison"])
    def test_only_out_of_scope_is_overridden(self, intent):
        # The override exists to prevent wrongful refusal, nothing else.
        assert _corrected_intent(intent, "What does Article 6 say?") == intent


class TestKnownFalsePositive:
    def test_another_jurisdictions_ai_act_is_also_rescued(self):
        # "Colorado AI Act" contains "AI Act", so the override fires. That is an accepted
        # trade: it forces a search, not an answer, and retrieval finds nothing relevant,
        # so the sufficiency gate abstains anyway. Refusing a real question is the worse
        # failure, so the override is deliberately biased towards looking.
        assert _corrected_intent("out_of_scope", "What does the Colorado AI Act require?") == (
            "lookup"
        )


class TestPatternHygiene:
    def test_the_pattern_contains_no_control_characters(self):
        # A previous edit wrote \b as literal backspace bytes, silently disabling every
        # word boundary and matching nothing at all.
        from euaia.graph.nodes import _NAMES_THE_ACT

        assert not any(ord(c) < 9 for c in _NAMES_THE_ACT.pattern)

    def test_word_boundaries_are_real(self):
        from euaia.graph.nodes import _NAMES_THE_ACT

        assert _NAMES_THE_ACT.search("the AI Act applies")
        assert not _NAMES_THE_ACT.search("thai actor")
