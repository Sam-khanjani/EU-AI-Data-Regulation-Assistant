"""The answer pipeline, as a LangGraph ``StateGraph`` (drawn in ``graph/README.md``).

Every node takes the state and the runtime context (:class:`Deps`) and returns only the
fields it changed; every decision is an edge:

* a follow-up is first rewritten to stand alone, so everything after it sees one question;
* small talk and out-of-scope questions stop before anything is searched;
* **plan** shapes the search to the kind of question: a comparison searches each side in
  parallel (LangGraph ``Send``), an applicability question adds the provisions of the legal
  test that decides it, an overview adds the outline of the whole group it asks about;
* **sufficiency**, before generation: if nothing retrieved scores above the relevance
  threshold, we abstain without calling the answer model -- asking it to ground an answer in
  evidence that does not address the question is how systems produce confident nonsense;
* a draft that overruns the response schema is asked for once more with fewer claims;
* **repair**: rejected quotes are named back to the model once, and it is asked to quote
  again. We never adjust a quote on the model's behalf -- see ``euaia.verify.citations``;
* **coverage**: how much of what the model said survived checking decides between
  answering, degrading to partial, and abstaining;
* **review**: a small model reads the verified answer and may send one follow-up round --
  a missing side of a comparison, or a term the answer depends on -- through the same
  search, draft and verification; **finalise** then writes one summary over all the rounds.
  Every claim shown is still a claim whose quotes were verified.

Progress is streamed as each node starts (LangGraph's ``custom`` stream), so the chat shows
the work live.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from sqlalchemy.orm import Session

from euaia.config import settings
from euaia.graph import prompts
from euaia.graph.state import (
    Finding,
    Progress,
    QueryState,
    Research,
    Round,
    SearchTask,
    Turn,
)
from euaia.ingest.embeddings import Embedder
from euaia.llm.groq_client import GroqClient, SchemaValidationFailed
from euaia.llm.ratelimit import Limits, estimate_tokens
from euaia.llm.schemas import (
    ANSWER_SCHEMA,
    ASSESSMENT_SCHEMA,
    FOLLOWUP_SCHEMA,
    QUERY_ANALYSIS_SCHEMA,
    REVIEW_SCHEMA,
    WRAPUP_SCHEMA,
)
from euaia.retrieval.hybrid import (
    RetrievedUnit,
    by_authority,
    expand_to_units,
    fit_token_budget,
    label_units,
    outline,
    retrieve,
    structural_search,
    with_structure,
)
from euaia.retrieval.rerank import rerank
from euaia.verify.citations import Evidence, VerificationReport, decide_verdict, verify_answer

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

LEGAL_TESTS = {
    "high_risk": (["6"], ["III"]),
    "prohibited": (["5"], []),
    "transparency": (["50"], []),
    "scope": (["2"], []),
    "gpai": (["51", "53", "55"], []),
    "definition": (["3"], []),
}
"""The provisions that set each test an applicability question turns on. Searched as if
the user had named them, so the checklist rests on the test itself -- Article 6 *and*
Annex III for high-risk -- rather than on whatever happened to rank well."""

Update = dict[str, Any]
"""What a node returns: only the state fields it changed."""


@dataclass(slots=True)
class Deps:
    """The graph's runtime context: what nodes use but never pass along in the state."""

    session: Session
    client: GroqClient
    embedder: Embedder
    lock: threading.Lock = field(default_factory=threading.Lock)
    """Parallel searches share one database session, which is not thread-safe."""


def _note(runtime: Runtime[Deps], step: str, detail: str = "") -> list[Progress]:
    """Stream a step to whoever is watching as it starts, and return it for the state."""
    progress = Progress(step=step, detail=detail)
    runtime.stream_writer(progress)
    return [progress]


def _ask_small_model(runtime: Runtime[Deps], system: str, user: str, schema: dict):
    """The cheap model, for routing and review. Short answers, but the model reasons first,
    and the client refuses an allowance below MIN_OUTPUT_TOKENS."""
    return runtime.context.client.structured(
        system=system, user=user, response_format=schema, model=settings.groq_small_model,
        max_completion_tokens=1024, reasoning_effort="low",
    )


# ------------------------------------------------------------------------ understand


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
    """Classify the question and gather what its path needs, in one call."""
    progress = _note(runtime, "analysing", "classifying the question")
    completion = _ask_small_model(
        runtime, prompts.ANALYSIS_SYSTEM, state.question, QUERY_ANALYSIS_SCHEMA
    )
    data = completion.data
    intent = _corrected_intent(data.get("intent", "lookup"), state.question)
    update: Update = {
        "progress": progress,
        "usage": completion.usage,
        "intent": intent,
        "search_queries": [q for q in data.get("search_queries", []) if q.strip()],
        **{
            key: [a.strip() for a in data.get(key, []) if a.strip()]
            for key in ("referenced_articles", "referenced_annexes")
        },
        "legal_test": data.get("legal_test") or "none",
        "sides": [s.strip() for s in data.get("sides", []) if s.strip()],
    }
    if intent == "greeting":
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


# ------------------------------------------------------------------------ search


_NAMED_ARTICLES = re.compile(r"\bArticles?\s+(\d+[a-z]?(?:\s*(?:,|and|or|to)\s*\d+[a-z]?)*)", re.I)
"""Articles a follow-up names, "Articles 53 and 54" included: round one has the analyser's."""


def plan(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Start a round: decide its question and its searches, one per side of a comparison."""
    number = state.round + 1
    if state.next_step:  # a follow-up the review asked for
        focus, intent = state.next_step["question"], state.next_step["kind"]
        sides = state.next_step["sides"]
        named = [n for g in _NAMED_ARTICLES.findall(focus) for n in re.findall(r"\d+[a-z]?", g)]
        tasks = [SearchTask(query=focus, label=f"for: {focus}", articles=named, round=number)]
    else:
        focus, intent, sides = state.question, state.intent, state.sides
        tasks = [SearchTask(
            query=focus,
            search_queries=state.search_queries,
            articles=state.referenced_articles,
            annexes=state.referenced_annexes,
            round=number,
        )]
        if intent == "applicability" and state.legal_test in LEGAL_TESTS:
            # The test's own provisions, ranked on their own so nothing crowds them out:
            # for a CV-screening tool, Annex III's employment point is the whole answer.
            articles, annexes = LEGAL_TESTS[state.legal_test]
            names = [f"Article {a}" for a in articles] + [f"Annex {a}" for a in annexes]
            tasks.append(SearchTask(
                query=focus, label=f"the legal test in {', '.join(names)}", articles=articles,
                annexes=annexes, round=number, side=1, named_only=True,
            ))
    if intent == "comparison" and len(sides) > 1:
        tasks = [
            SearchTask(query=side, label=f"for: {side}", round=number, side=i)
            for i, side in enumerate(sides[:3])
        ]
    return {
        "round": number, "focus": focus, "focus_intent": intent, "tasks": tasks,
        "next_step": None,
        # A fresh draft for the new round.
        "evidence": [], "raw_answer": None, "report": None, "rejected": [],
        "repair_attempted": False, "shortened": False, "overran": False,
    }


def embed_query(task: SearchTask, runtime: Runtime[Deps]) -> Update:
    """Embed the search's query for the dense leg of retrieval."""
    detail = f"searching {task.label}" if task.label else "searching the indexed regulation"
    progress = _note(runtime, "retrieving", detail)
    if task.named_only:  # a lookup by number needs no vector
        return {"progress": progress}
    return {
        "progress": progress,
        "query_embedding": runtime.context.embedder.embed_query(task.query),
    }


def retrieve_evidence(task: SearchTask, runtime: Runtime[Deps]) -> Update:
    """Run the three retrieval legs, at chunk granularity; or only the named provisions."""
    with runtime.context.lock:
        if task.named_only:
            return {"candidates": structural_search(
                runtime.context.session, task.articles, task.annexes, parts=[]
            )}
        return {"candidates": retrieve(
            runtime.context.session,
            query=task.query,
            embedding=task.query_embedding,
            search_queries=task.search_queries,
            articles=task.articles,
            annexes=task.annexes,
        )}


def rerank_evidence(task: SearchTask, runtime: Runtime[Deps]) -> Update:
    """Score the ~99-token chunks, before expansion: cheaper, and a truer measure."""
    progress = _note(runtime, "reranking", f"scoring {len(task.candidates)} candidate passages")
    result = rerank(task.query, task.candidates)
    return {
        "progress": progress,
        "usage": result.usage,
        "best_score": result.best_score,
        "kept": result.kept,
    }


def expand_evidence(task: SearchTask, runtime: Runtime[Deps]) -> Update:
    """Expand the kept chunks to whole provisions, and hand them back as this search's find."""
    units: list[RetrievedUnit] = []
    progress: list[Progress] = []
    if task.kept:
        progress = _note(runtime, "expanding", f"reading {len(task.kept)} provisions in full")
        with runtime.context.lock:
            units = expand_to_units(runtime.context.session, task.kept)
    return {
        "progress": progress,
        "found": [Finding(task.round, task.side, units, task.best_score)],
    }


def collect(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Merge this round's searches into one evidence set that fits the answer prompt."""
    found = sorted((f for f in state.found if f.round == state.round), key=lambda f: f.side)
    budget = settings.evidence_token_budget
    # Authority orders the units *before* the budget is spent: in relevance order, a long
    # binding provision like Article 50 was promoted and then dropped for want of room.
    # Each side of a comparison spends its own share, so neither crowds the other out.
    share = budget // max(1, len(found))
    units = by_authority(list({  # one copy of a provision two searches both found
        u.unit_id: u for f in found for u in fit_token_budget(by_authority(f.units), share)
    }.values()))
    session = runtime.context.session
    # Code units are read inside their commitment, with what is left of the same budget.
    spent = sum(estimate_tokens(u.text) for u in units)
    units = with_structure(session, units, max(0, budget - spent))
    if state.focus_intent == "overview" and units:
        units = [units[0], *outline(session, units[0]), *units[1:]]
    return {
        "best_rerank_score": max((f.best_score for f in found), default=0.0),
        "evidence": _fit_to_prompt_budget(label_units(units), state),
    }


# ------------------------------------------------------------------------ answer


def generate(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Produce a structured, quote-carrying answer under a strict schema.

    Groq returns 400 when the response misses a field, which in practice means the model ran
    long and closed the object early: recoverable, by asking for fewer claims.
    """
    progress = _note(runtime, "generating", "drafting an answer from the retrieved provisions")
    max_claims = settings.max_claims_retry if state.shortened else settings.max_claims
    system, user = _answer_prompts(state, state.evidence, max_claims, state.rejected or None)
    assessment = state.focus_intent == "applicability"
    try:
        completion = runtime.context.client.structured(
            system=system,
            user=user,
            response_format=ASSESSMENT_SCHEMA if assessment else ANSWER_SCHEMA,
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
    if state.focus_intent == "applicability":
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


# ------------------------------------------------------------------------ review and finish


def review(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Keep this round's verified answer, then ask whether a part of the question is missing.

    At most ``max_rounds`` rounds. The follow-up goes through the same search, drafting and
    verification as the question itself, so it can only add verified claims.
    """
    this = Round(
        state.focus, state.focus_intent, _outcome(state), state.report, state.evidence,
        state.raw_answer,
    )
    update: Update = {"rounds": [this]}
    # A checklist's open points are facts about the user's system, which no search supplies.
    if (
        this.outcome["verdict"] == "abstained" or state.round >= settings.max_rounds
        or state.focus_intent == "applicability"
    ):
        return update
    update["progress"] = _note(runtime, "reviewing", "checking the answer covers the question")
    so_far = prompts.answer_so_far([*state.rounds, this])
    try:
        completion = _ask_small_model(
            runtime, prompts.REVIEW_SYSTEM,
            f"QUESTION\n{state.question}\n\nANSWER SO FAR\n\n{so_far}", REVIEW_SCHEMA,
        )
    except SchemaValidationFailed:
        log.warning("Review was rejected; answering with what was verified")
        return update
    data = completion.data
    update["usage"] = completion.usage
    question = str(data.get("question") or "").strip()
    if data.get("complete") is False and data.get("kind") in ("lookup", "comparison") and (
        question and question != state.focus
    ):
        sides = [s.strip() for s in data.get("sides", []) if s.strip()]
        update["next_step"] = {"kind": data["kind"], "question": question, "sides": sides}
        update["progress"] += _note(runtime, "extending", question)
    return update


def finalise(state: QueryState, runtime: Runtime[Deps]) -> Update:
    """Assemble the rounds into what the user sees: one answer, every claim verified."""
    first, *more = state.rounds
    kept = [first]
    if first.outcome["verdict"] != "abstained":
        kept += [r for r in more if r.outcome["verdict"] != "abstained"]
    reports = [r.report for r in kept if r.report is not None]
    update: Update = dict(first.outcome) | {
        "claims": [
            claim for r in kept if r.intent != "applicability" and r.report
            for claim in r.report.claims
        ],
        "report": first.report if len(kept) == 1 else _merged(reports),
        "evidence": [u for r in kept for u in r.evidence],
        "raw_answer": (
            first.raw_answer if len(state.rounds) == 1
            else {"rounds": [r.raw_answer for r in state.rounds]}
        ),
    }
    progress: list[Progress] = []
    if len(kept) > 1:
        if any(r.outcome["verdict"] == "partial" for r in kept):
            update["verdict"] = "partial"
        progress = _note(runtime, "wrapping_up", "bringing the parts into one answer")
        try:
            completion = _ask_small_model(
                runtime, prompts.WRAPUP_SYSTEM,
                f"QUESTION\n{state.question}\n\nTHE PARTS\n\n{prompts.answer_so_far(kept)}",
                WRAPUP_SCHEMA,
            )
        except SchemaValidationFailed:
            log.warning("Wrap-up was rejected; keeping the first part's summary")
        else:
            update["usage"] = completion.usage
            update["summary"] = str(completion.data.get("summary") or "").strip() or first.outcome[
                "summary"
            ]
            update["unanswered_aspects"] = list(completion.data.get("unanswered_aspects") or [])
    return update | {"progress": progress + _note(runtime, "done", update["verdict"])}


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


def _after_collect(state: QueryState) -> str:
    """Gate before generation: is there anything worth grounding an answer in? A follow-up
    round that finds nothing is dropped, not the answer it follows."""
    if state.evidence and state.best_rerank_score >= settings.rerank_min_score:
        return "generate"
    return "finalise" if state.rounds else "abstain"


def build_graph():
    """Wire the nodes and decisions into a compiled graph, with search as a subgraph."""
    research = StateGraph(SearchTask, context_schema=Deps, output_schema=Research)
    research.add_sequence([embed_query, retrieve_evidence, rerank_evidence, expand_evidence])
    research.add_edge(START, "embed_query")

    graph = StateGraph(QueryState, context_schema=Deps)
    for node in (
        rewrite_followup, analyse, plan, collect, generate, shorten, verify, repair, review,
        finalise, abstain,
    ):
        graph.add_node(node)
    graph.add_node("research", research.compile())

    def route(source: str, decide: Callable[[QueryState], bool], yes: str, no: str) -> None:
        graph.add_conditional_edges(source, lambda s: yes if decide(s) else no, [yes, no])

    # A first question costs no rewrite.
    route(START, lambda s: bool(s.history), "rewrite_followup", "analyse")
    graph.add_edge("rewrite_followup", "analyse")
    route("analyse", lambda s: s.intent in _NO_SEARCH, "abstain", "plan")
    # One search per task, in parallel; collect runs once they have all finished.
    graph.add_conditional_edges(
        "plan", lambda s: [Send("research", task) for task in s.tasks], ["research"]
    )
    graph.add_edge("research", "collect")
    graph.add_conditional_edges("collect", _after_collect, ["generate", "finalise", "abstain"])
    # One shorter retry; a second overrun is verified as empty and abstains.
    route("generate", lambda s: s.overran and not s.shortened, "shorten", "verify")
    graph.add_edge("shorten", "generate")
    # One repair, only if something was rejected and something might be salvageable.
    route(
        "verify",
        lambda s: not s.repair_attempted and bool(s.report.rejected)
        and s.report.coverage < settings.coverage_answer_threshold,
        "repair", "review",
    )
    graph.add_edge("repair", "generate")
    route("review", lambda s: s.next_step is not None, "plan", "finalise")
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
    for namespace, mode, chunk in GRAPH.stream(
        initial,
        context=Deps(session, client, embedder),
        stream_mode=["custom", "values"],
        subgraphs=True,  # progress from inside the searches too
        config={"recursion_limit": 60},  # two rounds, each with its retries
    ):
        if mode == "custom":
            if on_progress is not None:
                on_progress(chunk)
        elif not namespace:
            final = chunk
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
    """System and user prompt for this round's answer call, or the criteria assessment's."""
    assessment = state.focus_intent == "applicability"
    system = (
        prompts.ASSESSMENT_SYSTEM if assessment
        else prompts.ANSWER_SYSTEM.format(max_claims=max_claims)
    )
    user = prompts.user_prompt(
        state.focus, prompts.format_evidence(evidence), rejected, assessment=assessment
    )
    if state.round > 1:
        user += prompts.PART_NOTE
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


def _merged(reports: list[VerificationReport]) -> VerificationReport:
    """One report over several rounds, for the answer's totals."""
    return VerificationReport(
        claims=[c for r in reports for c in r.claims],
        dropped_claims=[c for r in reports for c in r.dropped_claims],
        rejected=[q for r in reports for q in r.rejected],
        quotes_total=sum(r.quotes_total for r in reports),
        quotes_exact=sum(r.quotes_exact for r in reports),
        quotes_elided=sum(r.quotes_elided for r in reports),
    )


def _outcome(state: QueryState) -> Update:
    """This round's verdict and what is shown with it."""
    data, report = state.raw_answer or {}, state.report
    if report is None or not report.claims:
        reason = OVERRAN_REASON if state.overran else data.get("abstain_reason")
        return {"verdict": "abstained", "abstain_reason": reason or UNSUPPORTED_REASON}
    # A model that declared itself unable to answer is believed, even if some quote verified.
    if state.focus_intent != "applicability" and data.get("answerable") is False:
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
    if state.focus_intent == "applicability":
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
