"""The answer pipeline, as a LangGraph ``StateGraph`` (drawn in the README).

Every node takes the state and the runtime context (:class:`Deps`) and returns only the
fields it changed; every decision is an edge:

* a follow-up is first rewritten to stand alone, so everything after it sees one question;
* small talk and out-of-scope questions stop before anything is searched;
* **sufficiency**, before generation: if nothing retrieved scores above the relevance
  threshold, we abstain without calling the answer model -- asking it to ground an answer in
  evidence that does not address the question is how systems produce confident nonsense;
* a draft that overruns the response schema is asked for once more with fewer claims;
* **repair**: rejected quotes are named back to the model once, and it is asked to quote
  again. We never adjust a quote on the model's behalf -- see ``euaia.verify.citations``;
* **coverage**, in finalise: how much of what the model said survived checking decides
  between answering, degrading to partial, and abstaining.

Progress is streamed as each node starts (LangGraph's ``custom`` stream), so the chat shows
the work live.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from sqlalchemy.orm import Session

from euaia.config import settings
from euaia.graph import prompts
from euaia.graph.state import Progress, QueryState, Turn
from euaia.ingest.embeddings import Embedder
from euaia.llm.groq_client import GroqClient, SchemaValidationFailed
from euaia.llm.ratelimit import Limits, estimate_tokens
from euaia.llm.schemas import (
    ANSWER_SCHEMA,
    ASSESSMENT_SCHEMA,
    FOLLOWUP_SCHEMA,
    QUERY_ANALYSIS_SCHEMA,
)
from euaia.retrieval.hybrid import (
    RetrievedUnit,
    by_authority,
    expand_to_units,
    fit_token_budget,
    label_units,
    retrieve,
    with_structure,
)
from euaia.retrieval.rerank import rerank
from euaia.verify.citations import Evidence, decide_verdict, verify_answer

log = logging.getLogger(__name__)

HELP_OFFER = (
    "I can help with questions about the EU AI Act and the Commission's guidelines, codes of "
    "practice and Q&A on it, such as prohibited practices, high-risk systems or "
    "transparency duties. What would you like to know?"
)
"""Fixed text, never model-written, so no reply can steer the chat off its subject."""
OUT_OF_SCOPE_REASON = f"That's outside what I can help with. {HELP_OFFER}"
_MAX_GREETING_CHARS = 200
"""The model's own words in a greeting reply are one short sentence; anything longer is
dropped rather than shown, so small talk cannot grow into a general-purpose answer."""
NO_EVIDENCE_REASON = (
    "The indexed corpus does not contain provisions that address this question closely "
    "enough to answer it."
)
UNSUPPORTED_REASON = (
    "A draft answer was produced but too little of it could be verified against the "
    "regulation text, so it has been withheld."
)
OVERRAN_REASON = (
    "The answer could not be produced within the response limits available on this plan."
)
_NO_SEARCH = ("out_of_scope", "greeting")

Update = dict[str, Any]
"""What a node returns: only the state fields it changed."""


@dataclass(slots=True)
class Deps:
    """The graph's runtime context: what nodes use but never pass along in the state."""

    session: Session
    client: GroqClient
    embedder: Embedder


def _note(runtime: Runtime[Deps], step: str, detail: str = "") -> list[Progress]:
    """Stream a step to whoever is watching as it starts, and return it for the state."""
    progress = Progress(step=step, detail=detail)
    runtime.stream_writer(progress)
    return [progress]


def _ask_small_model(runtime: Runtime[Deps], system: str, user: str, schema: dict):
    """The cheap model, for routing steps. Short answers, but the model reasons first, and the
    client refuses an allowance below MIN_OUTPUT_TOKENS."""
    return runtime.context.client.structured(
        system=system, user=user, response_format=schema, model=settings.groq_small_model,
        max_completion_tokens=1024, reasoning_effort="low",
    )


# ------------------------------------------------------------------------ nodes


def rewrite_followup(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Turn a follow-up such as "what about deployers?" into a question that stands alone.

    Everything downstream sees one question and no conversation, so the rewritten question
    is answered from the evidence exactly as a typed one would be. What the user typed is
    kept in ``asked``.
    """
    update: Update = {"progress": _note(runtime, "rewriting", "reading the conversation so far")}
    try:
        completion = _ask_small_model(
            runtime, prompts.FOLLOWUP_SYSTEM,
            prompts.followup_prompt(state.history, state.question), FOLLOWUP_SCHEMA,
        )
    except SchemaValidationFailed:
        # A failed rewrite should cost some precision, not the answer: search with the
        # previous question alongside the new message.
        log.warning("Follow-up rewrite was rejected; searching with the previous question too")
        standalone = f"{state.history[-1].question} {state.question}"
    else:
        update["usage"] = completion.usage
        standalone = str(completion.data.get("standalone_question") or "").strip()

    if standalone and standalone != state.question:
        update |= {"asked": state.question, "question": standalone}
        update["progress"] += _note(runtime, "rewritten", standalone)
    return update


def analyse(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Classify intent, expand the query, and pick up any provision the user named."""
    progress = _note(runtime, "analysing", "classifying the question")
    completion = _ask_small_model(
        runtime, prompts.ANALYSIS_SYSTEM, state.question, QUERY_ANALYSIS_SCHEMA
    )
    data = completion.data
    update: Update = {
        "progress": progress,
        "usage": completion.usage,
        "intent": _corrected_intent(data.get("intent", "lookup"), state.question),
        "search_queries": [q for q in data.get("search_queries", []) if q.strip()],
        **{
            key: [a.strip() for a in data.get(key, []) if a.strip()]
            for key in ("referenced_articles", "referenced_annexes")
        },
    }
    if update["intent"] == "greeting":
        reply = str(data.get("reply") or "").strip()
        update["abstain_reason"] = (
            f"{reply} {HELP_OFFER}" if 0 < len(reply) <= _MAX_GREETING_CHARS else HELP_OFFER
        )
    log.debug("analysis: %s", update)
    return update


# A question naming the Act, or a provision of it, is about the Act. The classifier gets to
# *route*, not to veto the corpus: "What does the AI Act say about AI literacy?" was once
# classified out_of_scope and refused without retrieval, though AI literacy is Article 4.
_NAMES_THE_ACT = re.compile(
    r"\bai act\b"
    r"|\bartificial intelligence act\b"
    r"|\b2024/1689\b"
    r"|\bthis regulation\b"
    r"|\barticle\s+\d"
    r"|\bannex\s+[IVXLC]+\b"
    r"|\brecital\s+\d",
    re.IGNORECASE,
)


def _corrected_intent(intent: str, question: str) -> str:
    """Override a no-search intent when the question explicitly names the Act.

    Covers ``greeting`` too: "hi! what does Article 5 say?" is a question, and small talk
    must never be how a real question goes unanswered.
    """
    if intent in _NO_SEARCH and _NAMES_THE_ACT.search(question):
        log.info("Overriding %s: the question names the Act or one of its provisions", intent)
        return "lookup"
    return intent


def embed_query(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Embed the question for the dense leg of retrieval."""
    return {
        "progress": _note(runtime, "retrieving", "searching the indexed regulation"),
        "query_embedding": runtime.context.embedder.embed_query(state.question),
    }


def retrieve_evidence(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Run the three retrieval legs, at chunk granularity."""
    return {"candidates": retrieve(
        runtime.context.session,
        query=state.question,
        embedding=state.query_embedding,
        search_queries=state.search_queries,
        articles=state.referenced_articles,
        annexes=state.referenced_annexes,
    )}


def rerank_evidence(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Score the ~99-token chunks, before expansion: cheaper, and a truer measure."""
    progress = _note(runtime, "reranking", f"scoring {len(state.candidates)} candidate passages")
    result = rerank(state.question, state.candidates)
    return {
        "progress": progress,
        "usage": result.usage,
        "best_rerank_score": result.best_score,
        "kept": result.kept,
    }


def expand_evidence(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Expand the kept chunks to whole provisions and fit them to the answer prompt."""
    if not state.kept:
        return {"evidence": []}
    progress = _note(runtime, "expanding", f"reading {len(state.kept)} provisions in full")
    session = runtime.context.session
    # Authority orders the units *before* the budget is spent: in relevance order, a long
    # binding provision like Article 50 was promoted and then dropped for want of room.
    units = fit_token_budget(
        by_authority(expand_to_units(session, state.kept)), settings.evidence_token_budget
    )
    # Code units are read inside their commitment, with what is left of the same budget.
    spent = sum(estimate_tokens(u.text) for u in units)
    units = with_structure(session, units, max(0, settings.evidence_token_budget - spent))
    return {"progress": progress, "evidence": _fit_to_prompt_budget(label_units(units), state)}


def generate(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Produce a structured, quote-carrying answer under a strict schema.

    Groq returns 400 when the response misses a field, which in practice means the model ran
    long and closed the object early: recoverable, by asking for fewer claims.
    """
    progress = _note(runtime, "generating", "drafting an answer from the retrieved provisions")
    max_claims = settings.max_claims_retry if state.shortened else settings.max_claims
    system, user = _answer_prompts(state, state.evidence, max_claims, state.rejected or None)
    try:
        completion = runtime.context.client.structured(
            system=system,
            user=user,
            response_format=ASSESSMENT_SCHEMA if state.intent == "applicability" else ANSWER_SCHEMA,
            model=settings.groq_model,
            # Evidence + output must fit the free tier's 8k tokens/minute.
            max_completion_tokens=settings.answer_max_tokens,
            # Low: this is selection and verbatim copying, and deliberation competes with the
            # answer for the minute budget.
            reasoning_effort="low",
        )
    except SchemaValidationFailed:
        log.warning("Answer overran the schema (shortened=%s)", state.shortened)
        return {"progress": progress, "overran": True, "raw_answer": None}
    return {
        "progress": progress,
        "usage": completion.usage,
        "overran": False,
        "raw_answer": completion.data,
    }


def shorten(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """After an overrun, ask for the answer again with fewer claims."""
    detail = "the first draft overran; asking for a shorter answer"
    return {"progress": _note(runtime, "retrying", detail), "shortened": True}


def verify(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Check every quote against the unit text shown to the model as its evidence block."""
    data = state.raw_answer or {}
    if state.intent == "applicability":
        # Quotes per criterion rather than per claim; the grounding requirement is the same.
        claims = [
            {"text": _criterion_text(c), "supporting_quotes": c.get("supporting_quotes", [])}
            for c in data.get("criteria", [])
        ]
    else:
        claims = data.get("claims", [])
    names = [f.name for f in fields(Evidence)]
    evidence = [Evidence(**{n: getattr(u, n) for n in names}) for u in state.evidence]
    return {
        "progress": _note(runtime, "verifying", "checking every quote against the regulation text"),
        "report": verify_answer(claims, evidence),
    }


def repair(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Name the rejected quotes back to the model, for one more draft at full length."""
    return {
        "progress": _note(runtime, "repairing", "asking the model to quote the source exactly"),
        "repair_attempted": True,
        "shortened": False,
        "rejected": state.report.unsupported_quote_texts(),
    }


def finalise(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Apply the coverage gate and assemble what the user actually sees."""
    update = _outcome(state)
    return update | {"progress": _note(runtime, "done", update["verdict"])}


def abstain(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Stop without an answer: small talk, out of scope, or nothing relevant found."""
    detail, reason = {
        # A greeting's reply was set in `analyse`.
        "greeting": ("small talk", state.abstain_reason),
        "out_of_scope": ("outside the indexed corpus", OUT_OF_SCOPE_REASON),
    }.get(state.intent, ("no sufficiently relevant provisions found", NO_EVIDENCE_REASON))
    return {
        "progress": _note(runtime, "abstaining", detail),
        "verdict": "abstained",
        "abstain_reason": reason,
    }


# ------------------------------------------------------------------------ the graph


def build_graph():
    """Wire the nodes and decisions into a compiled graph."""
    graph = StateGraph(QueryState, context_schema=Deps)
    for node in (rewrite_followup, analyse, generate, shorten, verify, repair, finalise, abstain):
        graph.add_node(node)
    graph.add_sequence([embed_query, retrieve_evidence, rerank_evidence, expand_evidence])

    def route(source: str, decide: Callable[[QueryState], bool], yes: str, no: str) -> None:
        graph.add_conditional_edges(source, lambda s: yes if decide(s) else no, [yes, no])

    # A first question costs no rewrite.
    route(START, lambda s: bool(s.history), "rewrite_followup", "analyse")
    graph.add_edge("rewrite_followup", "analyse")
    route("analyse", lambda s: s.intent in _NO_SEARCH, "abstain", "embed_query")
    route(
        "expand_evidence",
        lambda s: bool(s.evidence) and s.best_rerank_score >= settings.rerank_min_score,
        "generate", "abstain",
    )
    # One shorter retry; a second overrun is verified as empty and abstains.
    route("generate", lambda s: s.overran and not s.shortened, "shorten", "verify")
    graph.add_edge("shorten", "generate")
    # One repair, only if something was rejected and something might be salvageable.
    route(
        "verify",
        lambda s: not s.repair_attempted and bool(s.report.rejected)
        and s.report.coverage < settings.coverage_answer_threshold,
        "repair", "finalise",
    )
    graph.add_edge("repair", "generate")
    graph.add_edge("finalise", END)
    graph.add_edge("abstain", END)
    return graph.compile()


GRAPH = build_graph()


def run_pipeline(
    question: str,
    session: Session,
    client: GroqClient,
    embedder: Embedder,
    history: Sequence[Turn] = (),
    on_progress: Callable[[Progress], None] | None = None,
) -> QueryState:
    """Run the graph for one question; ``on_progress`` hears about each step as it starts."""
    started = time.perf_counter()
    initial = QueryState(question=question, history=list(history[-settings.followup_turns :]))
    final: dict[str, Any] = {}
    for mode, chunk in GRAPH.stream(
        initial, context=Deps(session, client, embedder), stream_mode=["custom", "values"]
    ):
        if mode == "values":
            final = chunk
        elif on_progress is not None:
            on_progress(chunk)
    state = QueryState(**final)
    state.latency_ms = int((time.perf_counter() - started) * 1000)
    return state


# ------------------------------------------------------------------------ helpers


def _answer_prompts(
    state: QueryState,
    evidence: list[RetrievedUnit],
    max_claims: int,
    rejected: list[str] | None = None,
) -> tuple[str, str]:
    """System and user prompt for the answer call, or the criteria assessment's."""
    assessment = state.intent == "applicability"
    system = (
        prompts.ASSESSMENT_SYSTEM if assessment
        else prompts.ANSWER_SYSTEM.format(max_claims=max_claims)
    )
    user = prompts.user_prompt(
        state.question, prompts.format_evidence(evidence), rejected, assessment=assessment
    )
    return system, user


def _fit_to_prompt_budget(
    evidence: list[RetrievedUnit], state: QueryState
) -> list[RetrievedUnit]:
    """Trim evidence until the *actual* assembled prompt fits the minute budget.

    A static estimate was once wrong by ~1,200 tokens of headers and separators, which made
    every answer call impossible to issue. Room is left for the repair note, and the budget
    is read from Limits so it cannot drift from what the rate limiter enforces.
    """
    usable = (
        Limits(tokens_per_minute=settings.groq_tpm).usable_tpm
        - settings.answer_max_tokens
        - estimate_tokens(prompts.REPAIR_NOTE)
    )

    def fits(units: list[RetrievedUnit]) -> bool:
        return estimate_tokens(*_answer_prompts(state, units, settings.max_claims)) <= usable

    kept = list(evidence)
    while len(kept) > 1 and not fits(kept):
        kept = kept[:-1]  # drop the lowest-ranked provision first
    if kept and not fits(kept):
        # One provision too large on its own (Article 5 is ~3,000 tokens). Truncating what
        # the model *sees* is safe: quotes are still verified against the full text.
        kept = [_truncate(kept[0], fits)]
    if len(kept) < len(evidence):
        log.info("Trimmed evidence from %d to %d units", len(evidence), len(kept))
    # Relabel so the model is given a gap-free E1..En.
    return [replace(unit, label=f"E{i}") for i, unit in enumerate(kept, start=1)]


def _truncate(
    unit: RetrievedUnit, fits: Callable[[list[RetrievedUnit]], bool]
) -> RetrievedUnit:
    """Shorten one unit's displayed text until the prompt fits, on a line boundary."""
    lines = unit.text.splitlines()
    while len(lines) > 1:
        lines = lines[: max(1, int(len(lines) * 0.8))]
        shortened = replace(unit, text="\n".join(lines))
        if fits([shortened]):
            log.warning("Truncated %s to fit the token budget", unit.citation_label)
            return shortened
    return unit


def _criterion_text(criterion: dict) -> str:
    """A criterion as a claim -- also the key that matches verified claims back to it."""
    return f"{criterion.get('criterion', '')} - {criterion.get('explanation', '')}".strip(" -")


def _outcome(state: QueryState) -> Update:
    """The verdict and what is shown with it."""
    data, report = state.raw_answer or {}, state.report
    if report is None or not report.claims:
        reason = OVERRAN_REASON if state.overran else data.get("abstain_reason")
        return {"verdict": "abstained", "abstain_reason": reason or UNSUPPORTED_REASON}
    # A model that declared itself unable to answer is believed, even if some quote verified.
    if state.intent != "applicability" and data.get("answerable") is False:
        reason = data.get("abstain_reason") or NO_EVIDENCE_REASON
        return {"verdict": "abstained", "abstain_reason": reason}
    verdict = decide_verdict(report)
    if verdict == "abstained":
        return {"verdict": verdict, "abstain_reason": UNSUPPORTED_REASON}

    update: Update = {
        "verdict": verdict,
        "summary": (data.get("framing") or data.get("summary") or "").strip(),
        "unanswered_aspects": list(data.get("unanswered_aspects") or []),
        "follow_up_questions": list(data.get("follow_up_questions") or []),
    }
    if state.intent == "applicability":
        # Only criteria whose quotes survived verification.
        surviving = {c.text: c for c in report.claims}
        update["criteria"] = [
            {
                "criterion": c.get("criterion", ""),
                "status": c.get("status", "needs_user_input"),
                "explanation": c.get("explanation", ""),
                "quotes": surviving[key].quotes,
            }
            for c in data.get("criteria", [])
            if (key := _criterion_text(c)) in surviving
        ]
    return update
