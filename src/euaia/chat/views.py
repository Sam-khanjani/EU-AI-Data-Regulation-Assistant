"""What the chat shows and accepts, kept free of Chainlit so it can be tested without it.

An answer reads as prose: a short summary, then one paragraph per verified claim, each
ending in small links to the provisions that support it. Every link is to a quote that was
found word for word in the regulation text -- a claim that could not be verified never
reaches this module. Hovering a link shows the quote; clicking it opens the provision on
EUR-Lex. How the answer was reached (search, ranking, verification) goes in a collapsible
step above it rather than in the answer itself.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from euaia.api.service import AnswerClaim, AnswerView, Citation
from euaia.graph.state import Progress, Turn

EXAMPLE_QUESTIONS = [
    ("Prohibited practices", "Which AI practices are prohibited?"),
    ("High-risk requirements", "What requirements apply to high-risk AI systems?"),
    ("Transparency", "What must an AI system tell users under Article 50?"),
    ("Deployer obligations", "What are the obligations of a deployer of a high-risk AI system?"),
    ("Is my system high-risk?", "Is my CV-screening tool a high-risk AI system?"),
]

STEP_LABELS = {
    "rewriting": "Reading the conversation",
    "rewritten": "Taking the follow-up to mean",
    "analysing": "Understanding the question",
    "retrieving": "Searching the regulation",
    "reranking": "Ranking the passages found",
    "expanding": "Reading the provisions in full",
    "generating": "Drafting the answer",
    "retrying": "Shortening the draft",
    "verifying": "Checking every quote against the source",
    "repairing": "Asking for exact quotes",
    "abstaining": "Stopping without an answer",
    "done": "Finished",
}

STATUS_LABELS = {
    "met": "✅ Met",
    "not_met": "❌ Not met",
    "needs_user_input": "❔ Needs your input",
}

_SHORT_LABELS = (("Article ", "Art. "), ("Recital ", "Rec. "), ("Paragraph ", "Para. "))
_RECAP_CHARS = 700


# ------------------------------------------------------------------------ sign-in


def check_login(username: str, password: str, users: str) -> bool:
    """Match a sign-in against ``CHAT_USERS``, written as ``name:password,name:password``.

    An empty setting refuses everyone. Every entry is compared, with ``compare_digest``, so
    neither a wrong name nor a wrong password can be told apart by timing.
    """
    matched = False
    for entry in users.split(","):
        name, sep, secret = entry.strip().partition(":")
        if not (sep and name and secret):
            continue
        name_ok = secrets.compare_digest(username.encode("utf-8"), name.encode("utf-8"))
        password_ok = secrets.compare_digest(password.encode("utf-8"), secret.encode("utf-8"))
        matched |= name_ok and password_ok
    return matched


# ------------------------------------------------------------------------ the answer


BASIS_HEADINGS = (
    ("law", "Legal requirement"),
    ("guidance", "Commission guidance"),
    ("code", "Practical implementation"),
)
"""Headings for the three kinds of document a claim can rest on, strongest first.

Shown because the difference changes what the reader should do. "The Act requires this" and
"a voluntary code suggests this" are not the same statement, and rendering them as one
undifferentiated list of claims invites reading the second as the first.
"""


def _grouped_claims(claims: list[AnswerClaim]) -> list[str]:
    """Claims under a heading naming what they stand on, in order of authority.

    A heading is only written when the answer actually draws on more than one kind of
    document -- most answers come wholly from the Act, and labelling those "Legal
    requirement" would add ceremony to every reply to distinguish it from nothing.
    """
    grouped = {name: [c for c in claims if c.basis == name] for name, _ in BASIS_HEADINGS}
    used = [(name, heading) for name, heading in BASIS_HEADINGS if grouped[name]]
    if len(used) < 2:
        return [_with_citations(claim.text, claim.citations) for claim in claims]

    parts: list[str] = []
    for name, heading in used:
        parts.append(f"**{heading}**")
        parts.extend(_with_citations(c.text, c.citations) for c in grouped[name])
    return parts


def answer_markdown(view: AnswerView) -> str:
    """The reply, as Markdown."""
    if view.verdict == "abstained":
        parts = [f"**I can't answer that from the AI Act.** {view.abstain_reason or ''}".strip()]
        if view.claims_dropped:
            parts.append(
                f"A draft was written, but {view.claims_dropped} of its statements could not "
                "be verified against the regulation text, so it has been withheld."
            )
        return "\n\n".join(parts)

    parts = []
    if view.verdict == "partial":
        parts.append(
            "> Part of the draft could not be verified against the regulation text and was "
            "withheld. What follows is only what the text supports."
        )
    if view.summary:
        parts.append(view.summary)

    if view.intent == "applicability" and view.criteria:
        parts.append(
            "> This sets out the test the Regulation applies. Whether your system meets it "
            "depends on facts only you have."
        )
        for number, criterion in enumerate(view.criteria, start=1):
            status = STATUS_LABELS.get(criterion["status"], criterion["status"])
            head = f"**{number}. {criterion['criterion']}** · {status}"
            body = _with_citations(criterion["explanation"], criterion["citations"])
            parts.append(f"{head}  \n{body}")
    else:
        parts.extend(_grouped_claims(view.claims))

    if view.follow_up_questions:
        parts.append("**To take this further, you would need to answer:**")
        parts.append(_bullets(view.follow_up_questions))
    if view.unanswered_aspects:
        parts.append("**Not covered by the provisions found:**")
        parts.append(_bullets(view.unanswered_aspects))
    return "\n\n".join(parts)


def citation_links(citations: Iterable[Citation]) -> str:
    """Small links to the supporting provisions, one per provision, quote on hover."""
    links: list[str] = []
    seen: set[str] = set()
    for citation in citations:
        label = short_label(citation.citation_label)
        if label in seen:
            continue
        seen.add(label)
        if citation.deeplink:
            links.append(f'[{label}](<{citation.deeplink}> "{_link_title(citation.quote)}")')
        else:
            links.append(f"`{label}`")
    return " ".join(links)


def short_label(label: str) -> str:
    """'Article 5' -> 'Art. 5', so a link reads as a footnote rather than a sentence."""
    for long, short in _SHORT_LABELS:
        if label.startswith(long):
            return short + label[len(long) :]
    return label


def _with_citations(text: str, citations: Iterable[Citation]) -> str:
    links = citation_links(citations)
    return f"{text.strip()} {links}".strip()


def _link_title(quote: str) -> str:
    text = " ".join(quote.split())
    if len(text) > 300:
        text = text[:297].rstrip() + "..."
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


# ------------------------------------------------------------------------ the working step


def step_title(view: AnswerView | None = None) -> str:
    """Shown after Chainlit's "Checking" / "Checked" prefix."""
    if view is None or not view.quotes_total:
        return "the EU AI Act"
    verified = view.quotes_total - view.quotes_dropped
    noun = "quote" if verified == 1 else "quotes"
    return f"the EU AI Act · {verified} {noun} verified"


def progress_markdown(progress: Sequence[Progress]) -> str:
    """The pipeline so far, one line per step."""
    return "\n".join(
        f"- **{STEP_LABELS.get(p.step, p.step)}** · {p.detail}" if p.detail
        else f"- **{STEP_LABELS.get(p.step, p.step)}**"
        for p in progress
    )


def verification_markdown(view: AnswerView) -> str:
    """Everything behind the answer: the steps, the checks, and the quoted text itself."""
    parts = []
    if view.asked:
        parts.append(f"**Answered as:** {view.question}")
    parts.append(progress_markdown([Progress(p["step"], p["detail"]) for p in view.progress]))

    facts = []
    if view.quotes_total:
        verified = view.quotes_total - view.quotes_dropped
        facts.append(f"{verified} of {view.quotes_total} quotes found word for word")
        facts.append(f"{view.coverage:.0%} of statements supported")
    if view.claims_dropped:
        facts.append(f"{view.claims_dropped} unsupported statements withheld")
    versions = "; ".join(s["version_label"] for s in view.sources if s.get("version_label"))
    if versions:
        facts.append(f"text version: {versions}")
    facts.append(f"{view.latency_ms / 1000:.1f} s")
    if view.query_log_id:
        facts.append(f"audit record #{view.query_log_id}")
    parts.append("**Verification:** " + " · ".join(facts))

    quotes = _quoted_text(view)
    if quotes:
        parts.append("**Quoted text**\n\n" + quotes)
    return "\n\n".join(parts)


def _quoted_text(view: AnswerView) -> str:
    citations = [c for claim in view.claims for c in claim.citations]
    citations += [c for criterion in view.criteria for c in criterion["citations"]]
    blocks = []
    for citation in citations:
        source = (
            f"[{citation.citation_label}](<{citation.deeplink}>)"
            if citation.deeplink
            else citation.citation_label
        )
        quote = " ".join(citation.quote.split())
        blocks.append(f"> {quote}\n>\n> — {source}")
    return "\n\n".join(blocks)


# ------------------------------------------------------------------------ conversation


def recap(view: AnswerView) -> str:
    """A plain-text reminder of an answer, for rewriting the next follow-up."""
    if view.verdict == "abstained":
        text = f"No answer given. {view.abstain_reason or ''}"
    else:
        pieces = [view.summary]
        pieces += [claim.text for claim in view.claims]
        pieces += [f"{c['criterion']}: {c['explanation']}" for c in view.criteria]
        text = " ".join(p.strip() for p in pieces if p and p.strip())
    text = " ".join(text.split())
    return text if len(text) <= _RECAP_CHARS else text[: _RECAP_CHARS - 3].rstrip() + "..."


def message_metadata(view: AnswerView) -> dict[str, Any]:
    """Stored with the reply, so a resumed conversation can still follow up on it."""
    return {"question": view.question, "recap": recap(view), "verdict": view.verdict}


def history_from_steps(steps: Iterable[Mapping[str, Any]]) -> list[Turn]:
    """Rebuild the conversation from a saved thread's messages, oldest first."""
    turns = []
    for step in steps:
        metadata = step.get("metadata") or {}
        if step.get("type") == "assistant_message" and metadata.get("question"):
            turns.append(Turn(metadata["question"], metadata.get("recap", "")))
    return turns
