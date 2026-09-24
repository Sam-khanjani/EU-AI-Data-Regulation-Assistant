"""The chat interface: sign-in, follow-up questions, and how an answer is shown.

Offline. Nothing here starts Chainlit: :mod:`euaia.chat.views` is kept free of it for this
reason, and the follow-up rewriter is driven by a fake model client.
"""

from __future__ import annotations

import pytest

from euaia.api.service import AnswerClaim, AnswerView, Citation
from euaia.chat import views
from euaia.graph import nodes, prompts
from euaia.graph.state import Progress, QueryState, Turn
from euaia.llm.groq_client import MIN_OUTPUT_TOKENS, Completion, SchemaValidationFailed
from euaia.llm.schemas import FOLLOWUP_SCHEMA

ART5_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02024R1689-20260727#art_5"


def cite(label="Article 5", quote="the placing on the market", url=ART5_URL):
    return Citation(
        citation_label=label, quote=quote, deeplink=url,
        version_label="consolidated 2026-07-27", method="exact",
    )


def answered(**overrides) -> AnswerView:
    view = AnswerView(
        question="Which AI practices are prohibited?",
        verdict="answered",
        summary="The Act bans a short list of AI practices.",
        claims=[
            AnswerClaim("Manipulative techniques are banned.", [cite()]),
            AnswerClaim("Social scoring is banned.", [cite(), cite("Recital 31", "social scoring")]),
        ],
        quotes_total=3,
        coverage=1.0,
        latency_ms=12_345,
        sources=[{"version_label": "consolidated 2026-07-27", "version_id": "1"}],
        progress=[{"step": "retrieving", "detail": "searching"}, {"step": "done", "detail": "answered"}],
        query_log_id=42,
    )
    for key, value in overrides.items():
        setattr(view, key, value)
    return view


class TestSignIn:
    USERS = "alice:s3cret, bob:hunter2"

    def test_a_listed_user_with_the_right_password_gets_in(self):
        assert views.check_login("alice", "s3cret", self.USERS)
        assert views.check_login("bob", "hunter2", self.USERS)

    @pytest.mark.parametrize(
        "username,password",
        [("alice", "hunter2"), ("bob", "s3cret"), ("carol", "s3cret"), ("alice", ""), ("", "")],
    )
    def test_anything_else_is_refused(self, username, password):
        assert not views.check_login(username, password, self.USERS)

    def test_an_unset_user_list_refuses_everyone(self):
        # compare_digest("", "") is True, so blank entries must never become a match.
        assert not views.check_login("", "", "")
        assert not views.check_login("", "", ":,alice:,:x")

    def test_a_password_may_contain_a_colon(self):
        assert views.check_login("alice", "a:b", "alice:a:b")


class TestAnswerReadsAsProse:
    def test_each_claim_ends_with_links_to_its_provisions(self):
        text = views.answer_markdown(answered())
        assert text.startswith("The Act bans a short list of AI practices.")
        assert (
            f'Manipulative techniques are banned. [Art. 5](<{ART5_URL}> "the placing on the market")'
            in text
        )

    def test_a_provision_cited_twice_in_one_claim_gets_one_link(self):
        view = answered(claims=[AnswerClaim("Banned.", [cite(), cite(quote="another span")])])
        assert views.answer_markdown(view).count("[Art. 5]") == 1

    def test_quotes_are_safe_inside_a_link_title(self):
        link = views.citation_links([cite(quote='the "provider"\n  shall ensure')])
        assert '"the \\"provider\\" shall ensure"' in link

    def test_a_citation_without_a_link_is_still_shown(self):
        assert views.citation_links([cite(url=None)]) == "`Art. 5`"

    @pytest.mark.parametrize(
        "label,short",
        [("Article 50", "Art. 50"), ("Recital 27", "Rec. 27"), ("Annex III", "Annex III")],
    )
    def test_labels_are_shortened_like_footnotes(self, label, short):
        assert views.short_label(label) == short

    def test_follow_ups_and_gaps_are_listed(self):
        text = views.answer_markdown(
            answered(follow_up_questions=["Who deploys it?"], unanswered_aspects=["Fines"])
        )
        assert "- Who deploys it?" in text and "- Fines" in text

    def test_a_partial_answer_says_so_first(self):
        assert views.answer_markdown(answered(verdict="partial")).startswith("> Part of the draft")

    def test_an_abstention_explains_itself(self):
        view = AnswerView(
            question="What does Article 99a say?", verdict="abstained",
            abstain_reason="Nothing in the corpus addresses it.", claims_dropped=2,
        )
        text = views.answer_markdown(view)
        assert text.startswith("**I can't answer that from the AI Act.** Nothing in the corpus")
        assert "2 of its statements could not be verified" in text

    @pytest.mark.parametrize("intent", ["out_of_scope", "greeting"])
    def test_small_talk_and_off_topic_get_the_friendly_reply_alone(self, intent):
        view = AnswerView(
            question="hello", verdict="abstained", intent=intent,
            abstain_reason="Hi! What would you like to know?",
        )
        assert views.answer_markdown(view) == "Hi! What would you like to know?"

    def test_an_applicability_answer_gives_criteria_never_a_verdict(self):
        view = answered(
            intent="applicability",
            claims=[],
            criteria=[
                {
                    "criterion": "Listed in Annex III",
                    "status": "needs_user_input",
                    "explanation": "Recruitment tools are listed.",
                    "citations": [cite("Annex III", "recruitment or selection")],
                }
            ],
        )
        text = views.answer_markdown(view)
        assert "**1. Listed in Annex III** · ❔ Needs your input" in text
        assert "Recruitment tools are listed. [Annex III]" in text
        assert "depends on facts only you have" in text


class TestVerificationStep:
    def test_shows_the_checks_and_every_quote(self):
        text = views.verification_markdown(answered())
        assert "3 of 3 quotes found word for word" in text
        assert "audit record #42" in text
        assert "> the placing on the market" in text
        assert f"[Article 5](<{ART5_URL}>)" in text

    def test_names_the_rewritten_question(self):
        view = answered(asked="what about deployers?")
        assert views.verification_markdown(view).startswith(
            "**Answered as:** Which AI practices are prohibited?"
        )

    def test_title_counts_verified_quotes(self):
        assert views.step_title() == "the EU AI Act"
        assert views.step_title(answered(quotes_dropped=1)) == "the EU AI Act · 2 quotes verified"

    def test_progress_uses_readable_step_names(self):
        text = views.progress_markdown([Progress("verifying", "checking"), Progress("done")])
        assert text == "- **Checking every quote against the source** · checking\n- **Finished**"


class TestConversationSurvivesAReload:
    def test_history_is_rebuilt_from_saved_replies(self):
        view = answered()
        steps = [
            {"type": "user_message", "output": "Which AI practices are prohibited?"},
            {"type": "tool", "output": "..."},
            {"type": "assistant_message", "metadata": views.message_metadata(view)},
            {"type": "assistant_message", "metadata": {}},  # e.g. an error message
        ]
        assert views.history_from_steps(steps) == [Turn(view.question, views.recap(view))]

    def test_recap_is_plain_and_bounded(self):
        long = answered(summary="word " * 400)
        assert len(views.recap(long)) <= 700
        assert "[Art." not in views.recap(answered())


class FakeClient:
    def __init__(self, standalone: str):
        self.standalone = standalone
        self.calls: list[dict] = []

    def structured(self, **kwargs) -> Completion:
        self.calls.append(kwargs)
        return Completion(data={"standalone_question": self.standalone}, model=kwargs["model"])


HISTORY = [Turn("What are the obligations of providers of high-risk AI systems?", "Providers must...")]


class TestFollowUpRewriting:
    def test_a_follow_up_is_answered_as_a_standalone_question(self):
        client = FakeClient("What are the obligations of deployers of high-risk AI systems?")
        state = nodes.rewrite_followup(QueryState("what about deployers?"), HISTORY, client)
        assert state.question == "What are the obligations of deployers of high-risk AI systems?"
        assert state.asked == "what about deployers?"

    def test_the_rewriter_sees_the_conversation_and_uses_its_schema(self):
        client = FakeClient("x")
        nodes.rewrite_followup(QueryState("what about deployers?"), HISTORY, client)
        call = client.calls[0]
        assert call["response_format"] is FOLLOWUP_SCHEMA
        assert call["system"] == prompts.FOLLOWUP_SYSTEM
        assert call["user"] == (
            "CONVERSATION\n\nUser: What are the obligations of providers of high-risk AI "
            "systems?\nAssistant: Providers must...\n\nLATEST MESSAGE\nwhat about deployers?"
        )

    def test_the_rewrite_asks_for_an_allowance_the_real_client_accepts(self):
        # The real client refuses anything under MIN_OUTPUT_TOKENS outright; a fake does not,
        # which is how a 512-token allowance once passed here and failed every follow-up.
        client = FakeClient("x")
        nodes.rewrite_followup(QueryState("what about deployers?"), HISTORY, client)
        assert client.calls[0]["max_completion_tokens"] >= MIN_OUTPUT_TOKENS

    @pytest.mark.parametrize("returned", ["Which AI practices are prohibited?", "   "])
    def test_an_unchanged_or_empty_rewrite_keeps_the_question(self, returned):
        state = nodes.rewrite_followup(
            QueryState("Which AI practices are prohibited?"), HISTORY, FakeClient(returned)
        )
        assert state.question == "Which AI practices are prohibited?"
        assert state.asked == ""

    def test_a_rejected_rewrite_falls_back_to_the_previous_question_as_context(self):
        class RejectingClient(FakeClient):
            def structured(self, **kwargs):
                raise SchemaValidationFailed("output_parse_failed")

        state = nodes.rewrite_followup(
            QueryState("what about deployers?"), HISTORY, RejectingClient("unused")
        )
        assert state.question == (
            "What are the obligations of providers of high-risk AI systems? what about deployers?"
        )
        assert state.asked == "what about deployers?"

    def test_the_first_question_of_a_conversation_costs_no_rewrite(self, monkeypatch):
        client = FakeClient("should not be used")

        def stop(state, _client):
            raise StopIteration

        monkeypatch.setattr(nodes, "analyse", stop)
        with pytest.raises(StopIteration):
            nodes.run_pipeline("Which AI practices are prohibited?", None, client, None)
        assert client.calls == []


class TestProgressIsReportedLive:
    def test_each_step_reaches_the_listener_as_it_happens(self):
        heard: list[Progress] = []
        state = QueryState("q", on_progress=heard.append)
        state.note("retrieving", "searching")
        state.note("done")
        assert heard == state.progress == [Progress("retrieving", "searching"), Progress("done")]
