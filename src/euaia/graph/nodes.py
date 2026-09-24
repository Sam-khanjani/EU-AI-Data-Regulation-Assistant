"""Pipeline nodes.

    analyse -> retrieve -> rerank -> sufficiency gate -> generate -> verify -> repair?
                                          |                                      |
                                          +------------- abstain <---------------+

In a conversation, a follow-up is first rewritten to stand alone (:func:`rewrite_followup`),
so everything above still sees a single question.

Two gates decide whether anything is said at all:

**Sufficiency** runs before generation. If nothing retrieved scores above the relevance
threshold, we abstain without calling the answer model -- there is no point asking it to
ground an answer in evidence that does not address the question, and doing so is how systems
end up producing confident nonsense from irrelevant context.

**Coverage** runs after verification. It measures how much of what the model said actually
survived checking, and decides between answering, degrading to partial, and abstaining.

Between them sits at most one repair round trip: rejected quotes are named back to the model
and it is asked to quote again. Note what repair is *not* -- we never adjust a quote on the
model's behalf to make it match. See ``euaia.verify.citations`` for why.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import replace

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


def rewrite_followup(state: QueryState, history: Sequence[Turn], client: GroqClient) -> QueryState:
    """Turn a follow-up such as "what about deployers?" into a question that stands alone.

    Everything downstream -- retrieval, the answer prompt, the audit row -- sees one question
    and no conversation, which keeps the grounding argument unchanged: the rewritten question
    is answered from the evidence exactly as a typed one would be. The rewrite only decides
    *what* is asked. What the user typed is kept in ``state.asked``.
    """
    state.note("rewriting", "reading the conversation so far")
    try:
        completion = client.structured(
            system=prompts.FOLLOWUP_SYSTEM,
            user=prompts.followup_prompt(history, state.question),
            response_format=FOLLOWUP_SCHEMA,
            model=settings.groq_small_model,
            # A short answer, but the model reasons before writing it, and the client refuses
            # any allowance below MIN_OUTPUT_TOKENS.
            max_completion_tokens=1024,
            reasoning_effort="low",
        )
    except SchemaValidationFailed:
        # The rewrite only steers the search; whatever is asked is still answered from
        # verified evidence. So a failed rewrite should cost some precision, not the answer:
        # search with the previous question alongside the new message.
        log.warning("Follow-up rewrite was rejected; searching with the previous question too")
        standalone = f"{history[-1].question} {state.question}"
    else:
        state.usage.add(completion.usage)
        standalone = str(completion.data.get("standalone_question") or "").strip()

    if standalone and standalone != state.question:
        state.asked, state.question = state.question, standalone
        state.note("rewritten", standalone)
    return state


def analyse(state: QueryState, client: GroqClient) -> QueryState:
    """Classify intent, expand the query, and pick up any provision the user named."""
    state.note("analysing", "classifying the question")
    completion = client.structured(
        system=prompts.ANALYSIS_SYSTEM,
        user=state.question,
        response_format=QUERY_ANALYSIS_SCHEMA,
        model=settings.groq_small_model,
        max_completion_tokens=1024,
        reasoning_effort="low",
    )
    state.usage.add(completion.usage)

    data = completion.data
    state.intent = _corrected_intent(data.get("intent", "lookup"), state.question)
    state.search_queries = [q for q in data.get("search_queries", []) if q.strip()]
    state.referenced_articles = [
        a.strip() for a in data.get("referenced_articles", []) if a.strip()
    ]
    state.referenced_annexes = [a.strip() for a in data.get("referenced_annexes", []) if a.strip()]
    if state.intent == "greeting":
        reply = str(data.get("reply") or "").strip()
        state.abstain_reason = (
            f"{reply} {HELP_OFFER}" if 0 < len(reply) <= _MAX_GREETING_CHARS else HELP_OFFER
        )

    log.debug(
        "intent=%s queries=%s articles=%s annexes=%s",
        state.intent, state.search_queries, state.referenced_articles, state.referenced_annexes,
    )
    return state


# A question naming the Act, or a provision of it, is about the Act. The classifier is a
# cheap heuristic and gets to *route*, not to veto the corpus: "What does the AI Act say
# about AI literacy?" was classified out_of_scope and refused in 181ms, without retrieval,
# even though AI literacy is Article 4. This overrides that specific failure deterministically.
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
    if intent in ("out_of_scope", "greeting") and _NAMES_THE_ACT.search(question):
        log.info("Overriding %s: the question names the Act or one of its provisions", intent)
        return "lookup"
    return intent


def retrieve_evidence(
    state: QueryState, session: Session, embedder: Embedder
) -> QueryState:
    """Embed the question and run the three retrieval legs, at chunk granularity."""
    state.note("retrieving", "searching the indexed regulation")
    embedding = embedder.embed_query(state.question)
    state.candidates = retrieve(
        session,
        query=state.question,
        embedding=embedding,
        search_queries=state.search_queries,
        articles=state.referenced_articles,
        annexes=state.referenced_annexes,
    )
    log.debug("retrieved %d candidate chunks", len(state.candidates))
    return state


def rerank_evidence(state: QueryState, session: Session) -> QueryState:
    """Score the chunks, then expand the survivors to whole articles.

    Expansion happens *after* scoring so the reranker sees ~99-token passages rather than
    ~530-token articles -- both cheaper and a truer measure of relevance.

    Scoring is local (:mod:`euaia.retrieval.rerank`), so this node spends no API tokens and
    takes no rate-limiter budget. It no longer needs the Groq client.
    """
    state.note("reranking", f"scoring {len(state.candidates)} candidate passages")
    result = rerank(state.question, state.candidates)
    state.usage.add(result.usage)
    state.best_rerank_score = result.best_score

    if not result.kept:
        state.evidence = []
        return state

    state.note("expanding", f"reading {len(result.kept)} provisions in full")
    units = expand_to_units(session, result.kept)
    # Authority orders the units *before* the budget is spent, not after. Ordering afterwards
    # looks equivalent and is not: reranking reserves a slot for binding text but appends it
    # last, so spending the budget in relevance order let a long provision like Article 50 be
    # promoted into the evidence and then dropped for want of room -- leaving an answer about
    # legal obligations resting entirely on guidance and a voluntary code.
    units = fit_token_budget(by_authority(units), settings.evidence_token_budget)
    state.retrieved = units
    state.evidence = _fit_to_prompt_budget(label_units(units), state)
    return state


def _answer_prompts(
    state: QueryState,
    evidence: list[RetrievedUnit],
    max_claims: int,
    rejected: list[str] | None = None,
) -> tuple[str, str]:
    """System and user prompt for the answer call, or the criteria assessment's."""
    assessment = state.intent == "applicability"
    if assessment:
        system = prompts.ASSESSMENT_SYSTEM
    else:
        system = prompts.ANSWER_SYSTEM.format(max_claims=max_claims)
    user = prompts.user_prompt(
        state.question, prompts.format_evidence(evidence), rejected, assessment=assessment
    )
    return system, user


def _fit_to_prompt_budget(
    evidence: list[RetrievedUnit], state: QueryState
) -> list[RetrievedUnit]:
    """Trim evidence until the assembled prompt fits the minute budget.

    Sized against the *actual* prompt rather than an estimate: an earlier version budgeted
    statically and was wrong by about 1,200 tokens once evidence headers and separators were
    counted, which made every answer call impossible to issue.

    Runs before the sufficiency gate so that a question whose evidence cannot fit abstains
    without spending an answer call on a prompt containing nothing.
    """
    # Leave room for the repair note, which is only added on a retry.
    # Sized against the budget the rate limiter will actually enforce, not the provider's
    # raw ceiling, and read from Limits so the two cannot drift apart.
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
        # A single provision too large to fit at all -- Article 5 is ~3,000 tokens on its
        # own. Truncating what the model *sees* is safe: quotes are still verified against
        # the unit's full text, so a quote copied from the visible part still passes and
        # the model cannot quote what it was not shown.
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


def evidence_is_sufficient(state: QueryState) -> bool:
    """Gate before generation: is there anything worth grounding an answer in?"""
    if state.intent == "out_of_scope":
        state.abstain_reason = OUT_OF_SCOPE_REASON
        return False
    if not state.evidence:
        state.abstain_reason = NO_EVIDENCE_REASON
        return False
    if state.best_rerank_score < settings.rerank_min_score:
        state.abstain_reason = NO_EVIDENCE_REASON
        return False
    return True


def _evidence_records(state: QueryState) -> list[Evidence]:
    """Bridge retrieval output into what the verifier checks against.

    ``text`` is the unit's own text -- the same string shown to the model as its evidence
    block -- so a faithfully copied quote always verifies, and a breadcrumb we synthesised
    never can.
    """
    return [
        Evidence(
            label=unit.label,
            unit_id=unit.unit_id,
            unit_path=unit.unit_path,
            citation_label=unit.citation_label,
            text=unit.text,
            deeplink=unit.deeplink,
            document_version_id=unit.document_version_id,
            version_label=unit.version_label,
            authority=unit.authority,
        )
        for unit in state.evidence
    ]


def generate(
    state: QueryState, client: GroqClient, rejected: list[str] | None = None
) -> QueryState:
    """Produce a structured, quote-carrying answer under a strict schema.

    A schema rejection is recoverable, not fatal. Groq validates the response and returns
    400 when a field is missing, which in practice means the model ran long and closed the
    object early. Asking again for fewer claims usually fits; if it still does not, the
    caller abstains rather than presenting a partial answer as a whole one.
    """
    try:
        return _generate_once(state, client, rejected, settings.max_claims)
    except SchemaValidationFailed:
        log.warning("Answer overran the schema; retrying with fewer claims")
        state.note("retrying", "the first draft overran; asking for a shorter answer")
        try:
            return _generate_once(state, client, rejected, settings.max_claims_retry)
        except SchemaValidationFailed:
            log.warning("Answer overran again; abstaining")
            state.raw_answer = None
            state.abstain_reason = (
                "The answer could not be produced within the response limits available on "
                "this plan."
            )
            return state


def _generate_once(
    state: QueryState,
    client: GroqClient,
    rejected: list[str] | None,
    max_claims: int,
) -> QueryState:
    state.note("generating", "drafting an answer from the retrieved provisions")
    # Evidence was already sized to the budget in rerank_evidence.
    system, user = _answer_prompts(state, state.evidence, max_claims, rejected)

    completion = client.structured(
        system=system,
        user=user,
        response_format=ASSESSMENT_SCHEMA if state.intent == "applicability" else ANSWER_SCHEMA,
        model=settings.groq_model,
        # Sized so evidence + output stays inside the free tier's 8k tokens/minute.
        # The client raises rather than silently truncating if it does not.
        max_completion_tokens=settings.answer_max_tokens,
        # Low, deliberately. This task is selection and verbatim copying, not reasoning:
        # the provisions have already been retrieved and ranked. Higher effort spends
        # completion tokens on deliberation that competes with the answer itself for the
        # minute budget, and overrunning gets the whole response rejected.
        reasoning_effort="low",
    )
    state.usage.add(completion.usage)
    state.raw_answer = completion.data
    return state


def _claims_from_answer(state: QueryState) -> list[dict]:
    """Normalise both answer shapes into the claim list the verifier expects.

    A criteria assessment carries its quotes per criterion rather than per claim, but the
    grounding requirement is identical, so both go through the same verifier.
    """
    data = state.raw_answer or {}
    if state.intent == "applicability":
        return [
            {
                "text": _criterion_text(c),
                "supporting_quotes": c.get("supporting_quotes", []),
            }
            for c in data.get("criteria", [])
        ]
    return data.get("claims", [])


def _criterion_text(criterion: dict) -> str:
    """A criterion as a claim -- also the key that matches verified claims back to it."""
    return f"{criterion.get('criterion', '')} - {criterion.get('explanation', '')}".strip(" -")


def verify(state: QueryState) -> QueryState:
    """Check every quote against the source it names."""
    state.note("verifying", "checking every quote against the regulation text")
    state.report = verify_answer(_claims_from_answer(state), _evidence_records(state))
    return state


def needs_repair(state: QueryState) -> bool:
    """One retry, only if something was rejected and something might be salvageable."""
    if state.repair_attempted or state.report is None:
        return False
    if not state.report.rejected:
        return False
    return state.report.coverage < settings.coverage_answer_threshold


def finalise(state: QueryState) -> QueryState:
    """Apply the coverage gate and assemble what the user actually sees."""
    data = state.raw_answer or {}
    report = state.report

    if report is None or not report.claims:
        state.verdict = "abstained"
        state.abstain_reason = state.abstain_reason or (
            data.get("abstain_reason") or UNSUPPORTED_REASON
        )
        return state

    # A model that declared itself unable to answer is believed, even if some quote verified.
    if state.intent != "applicability" and data.get("answerable") is False:
        state.verdict = "abstained"
        state.abstain_reason = data.get("abstain_reason") or NO_EVIDENCE_REASON
        return state

    state.verdict = decide_verdict(report)
    if state.verdict == "abstained":
        state.abstain_reason = UNSUPPORTED_REASON
        return state

    state.summary = (data.get("framing") or data.get("summary") or "").strip()
    state.unanswered_aspects = list(data.get("unanswered_aspects") or [])
    state.follow_up_questions = list(data.get("follow_up_questions") or [])
    if state.intent == "applicability":
        state.criteria = _verified_criteria(state)
    return state


def _verified_criteria(state: QueryState) -> list[dict]:
    """Keep only assessment criteria whose quotes survived verification."""
    if state.report is None:
        return []
    surviving = {c.text: c for c in state.report.claims}
    out = []
    for criterion in (state.raw_answer or {}).get("criteria", []):
        verified = surviving.get(_criterion_text(criterion))
        if verified is None:
            continue
        out.append(
            {
                "criterion": criterion.get("criterion", ""),
                "status": criterion.get("status", "needs_user_input"),
                "explanation": criterion.get("explanation", ""),
                "quotes": verified.quotes,
            }
        )
    return out


def run_pipeline(
    question: str,
    session: Session,
    client: GroqClient,
    embedder: Embedder,
    history: Sequence[Turn] = (),
    on_progress: Callable[[Progress], None] | None = None,
) -> QueryState:
    """Execute the whole graph for one question.

    ``history`` is the conversation so far, oldest first; with none, the question is taken
    as typed. ``on_progress`` hears about each step as it starts.
    """
    started = time.perf_counter()
    state = QueryState(question=question, on_progress=on_progress)

    if history:
        rewrite_followup(state, history[-settings.followup_turns :], client)
    analyse(state, client)
    if state.intent in ("out_of_scope", "greeting"):
        # Nothing is searched and nothing is generated: a greeting's reply is the model's
        # one short sentence plus fixed text, set in `analyse`.
        greeting = state.intent == "greeting"
        state.note("abstaining", "small talk" if greeting else "outside the indexed corpus")
        state.verdict = "abstained"
        state.abstain_reason = state.abstain_reason if greeting else OUT_OF_SCOPE_REASON
        state.latency_ms = int((time.perf_counter() - started) * 1000)
        return state

    retrieve_evidence(state, session, embedder)
    rerank_evidence(state, session)

    if not evidence_is_sufficient(state):
        state.note("abstaining", "no sufficiently relevant provisions found")
        state.verdict = "abstained"
        state.latency_ms = int((time.perf_counter() - started) * 1000)
        return state

    generate(state, client)
    verify(state)

    if needs_repair(state):
        state.note("repairing", "asking the model to quote the source exactly")
        state.repair_attempted = True
        generate(state, client, rejected=state.report.unsupported_quote_texts())
        verify(state)

    finalise(state)
    state.latency_ms = int((time.perf_counter() - started) * 1000)
    state.note("done", state.verdict)
    return state
