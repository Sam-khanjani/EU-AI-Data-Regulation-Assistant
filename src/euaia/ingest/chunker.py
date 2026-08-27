"""Turn parsed legal units into embeddable chunks.

Four rules drive the design:

**Pack sibling paragraphs; never cross an article.** A whole article often exceeds the
Gemini embedding model's 2,048-token input limit, but one paragraph is usually too small to
mean anything -- two thirds of the AI Act's paragraphs are under 150 tokens, and
"2. Paragraph 1 shall not apply where..." embeds to noise. So paragraphs of the same article
are packed to roughly ``chunk_target_tokens`` and the chunk is attached to that article.
Packing stops at the article boundary, so every chunk still maps to exactly one citable
provision.

**Breadcrumbs go into the embedded text.** A bare provision carries no signal that it lives
in the high-risk chapter. Prefixing 'Regulation (EU) 2024/1689 - Chapter III, Section 1,
Article 6 - Classification rules for high-risk AI systems' puts that context into the
vector, which is what lets a query like "how are high-risk systems classified" find it.

**The breadcrumb never reaches citation verification.** Quotes are checked against the
*structural unit's* text, not the chunk text, so a model cannot "quote" a breadcrumb we
synthesised and have it pass as source text.

**Chunk boundaries do not constrain citation.** Because verification targets the unit rather
than the chunk, a quote spanning a chunk boundary still verifies. That is what makes it safe
to split a pathologically long provision mid-sentence as a last resort -- and splitting is
necessary, since anything over the embedding input limit is rejected outright.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import tiktoken

from euaia.config import settings
from euaia.ingest.formex import ParsedDocument, ParsedUnit

log = logging.getLogger(__name__)

# Unit types worth embedding. Chapters and sections are headings only -- their substance
# lives in the articles beneath them, so embedding them would just add near-duplicates.
_EMBEDDABLE = frozenset({"paragraph", "article", "recital", "annex"})


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """A chunk before it acquires an embedding or a database id."""

    unit_path: str
    """Path of the structural unit this chunk came from."""

    text: str
    """Breadcrumb + body. This is what gets embedded."""

    body: str
    """Body only, without the breadcrumb."""

    token_count: int
    part: int = 0
    """0-based index when one unit had to be split across several chunks."""

    total_parts: int = 1


class TokenCounter:
    """Approximate token counting.

    Gemini's tokenizer is not public, so this uses a GPT tokenizer as a stand-in. That is
    fine because we only need a *conservative* bound: the configured ceiling
    (``chunk_max_tokens``) sits well below the model's real 2,048-token limit, leaving
    headroom for tokenizer disagreement. The embedder still handles an over-length
    rejection from the API defensively.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


def build_breadcrumb(doc_title: str, unit: ParsedUnit, by_path: dict[str, ParsedUnit]) -> str:
    """Human-readable location of a unit, e.g.

    ``Regulation (EU) 2024/1689 - Chapter III, Section 1, Article 6 - Classification rules
    for high-risk AI systems``
    """
    trail: list[str] = []
    for ancestor in _ancestors(unit, by_path):
        label = _unit_label(ancestor)
        if label:
            trail.append(label)

    own = _unit_label(unit)
    if own:
        trail.append(own)

    location = ", ".join(trail)
    heading = unit.heading or _nearest_heading(unit, by_path)

    parts = [doc_title]
    if location:
        parts.append(location)
    if heading:
        parts.append(heading)
    return " - ".join(parts)


def _ancestors(unit: ParsedUnit, by_path: dict[str, ParsedUnit]) -> list[ParsedUnit]:
    chain: list[ParsedUnit] = []
    path = unit.parent_path
    seen: set[str] = set()
    while path and path in by_path and path not in seen:
        seen.add(path)
        parent = by_path[path]
        chain.append(parent)
        path = parent.parent_path
    return list(reversed(chain))


def _unit_label(unit: ParsedUnit) -> str:
    pretty = {
        "chapter": "Chapter",
        "section": "Section",
        "article": "Article",
        "annex": "Annex",
        "recital": "Recital",
        "paragraph": "Paragraph",
    }.get(unit.unit_type)
    if not pretty:
        return ""
    if unit.unit_type == "paragraph":
        # unit_number is already qualified, e.g. '6(2)'; the article appears via ancestors.
        return f"paragraph {unit.unit_number}" if unit.unit_number else "paragraph"
    return f"{pretty} {unit.unit_number}" if unit.unit_number else pretty


def _nearest_heading(unit: ParsedUnit, by_path: dict[str, ParsedUnit]) -> str | None:
    """A paragraph has no heading of its own; borrow its article's."""
    for ancestor in reversed(_ancestors(unit, by_path)):
        if ancestor.heading:
            return ancestor.heading
    return None


def chunk_document(
    doc: ParsedDocument,
    doc_title: str,
    counter: TokenCounter | None = None,
    include_unit_types: frozenset[str] | None = None,
) -> list[ChunkDraft]:
    """Produce embeddable chunks for a parsed document, in document order.

    Sibling paragraphs are **packed together** into chunks of roughly
    ``chunk_target_tokens`` and attached to their parent article, rather than embedded one
    per paragraph. Two reasons, and the quality one came first:

    * A 72-token paragraph reading "2. Paragraph 1 shall not apply where..." carries almost
      no standalone meaning, and its embedding is correspondingly useless. Two thirds of the
      AI Act's paragraphs are under 150 tokens. Packing them restores enough context for the
      vector to mean something.
    * It cuts the corpus from 599 chunks to about 270. The Gemini free tier allows 1,000
      embed items per day *including* query embeddings, so the difference is between a
      corpus that can be re-indexed comfortably and one that cannot.

    Packing never crosses an article boundary, so every chunk still maps to exactly one
    citable provision -- and because the chunk is attached to the article, quotes are
    verified against the article's full text, of which the chunk is a subset.

    ``include_unit_types`` narrows what gets indexed. That is a correctness requirement, not
    an optimisation: the original act and the consolidated act both contain Article 6, but
    only the consolidated one has amendments applied. Indexing both would let the system
    cite superseded wording as if it were in force, so the original act contributes recitals
    only (consolidation drops those) and the consolidated act supplies all operative text.
    """
    counter = counter or TokenCounter()
    allowed = _EMBEDDABLE if include_unit_types is None else (_EMBEDDABLE & include_unit_types)
    by_path = {u.unit_path: u for u in doc.units}

    paragraphs: dict[str, list[ParsedUnit]] = {}
    for unit in doc.units:
        if unit.unit_type == "paragraph" and unit.parent_path:
            paragraphs.setdefault(unit.parent_path, []).append(unit)

    drafts: list[ChunkDraft] = []
    for unit in doc.units:
        if unit.unit_type == "paragraph":
            continue  # emitted as part of its parent article
        if unit.unit_type not in allowed:
            continue

        breadcrumb = build_breadcrumb(doc_title, unit, by_path)
        children = paragraphs.get(unit.unit_path, []) if unit.unit_type == "article" else []

        if children and "paragraph" in allowed:
            bodies = _pack([p.text for p in children], breadcrumb, counter)
        else:
            bodies = _split_body(unit.text, breadcrumb, counter)

        for i, body in enumerate(bodies):
            text = f"{breadcrumb}\n\n{body}"
            drafts.append(
                ChunkDraft(
                    unit_path=unit.unit_path,
                    text=text,
                    body=body,
                    token_count=counter.count(text),
                    part=i,
                    total_parts=len(bodies),
                )
            )

    oversize = [d for d in drafts if d.token_count > settings.chunk_max_tokens]
    if oversize:
        log.warning(
            "%d chunks exceed chunk_max_tokens=%d (largest %d tokens)",
            len(oversize),
            settings.chunk_max_tokens,
            max(d.token_count for d in oversize),
        )
    log.info("Built %d chunks from %d units", len(drafts), len(doc.units))
    return drafts


def _pack(parts: list[str], breadcrumb: str, counter: TokenCounter) -> list[str]:
    """Pack sibling paragraphs into chunks of roughly ``chunk_target_tokens``.

    Paragraphs are kept whole and in order: a chunk boundary always falls between two
    paragraphs, never inside one. That matters for citation -- a quote split across two
    chunks could not be verified against either.

    A single paragraph larger than the budget is emitted alone and split by
    :func:`_split_body`, which breaks on line boundaries rather than mid-sentence.
    """
    budget = settings.chunk_max_tokens - counter.count(breadcrumb) - 8
    if budget <= 0:
        raise ValueError(f"breadcrumb alone exceeds the chunk budget: {breadcrumb!r}")
    target = min(settings.chunk_target_tokens, budget)

    out: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for part in parts:
        cost = counter.count(part)
        if cost > budget:
            if current:
                out.append("\n".join(current))
                current, current_tokens = [], 0
            out.extend(_split_body(part, breadcrumb, counter))
            continue

        if current and current_tokens + cost > target:
            out.append("\n".join(current))
            current, current_tokens = [], 0

        current.append(part)
        current_tokens += cost

    if current:
        out.append("\n".join(current))
    return out or [""]


_SENTENCE_END = re.compile(r"(?<=[.;:])\s+")


def _split_long_line(line: str, budget: int, counter: TokenCounter) -> list[str]:
    """Split one over-long line, preferring sentence boundaries, then words.

    Splitting inside a sentence is acceptable here in a way it would not be elsewhere,
    because a chunk boundary does not constrain citation: quotes are verified against the
    *structural unit's* full text, not against the chunk. A quote spanning a chunk boundary
    therefore still verifies. What a chunk boundary affects is only which passage retrieval
    surfaces.

    Leaving the line whole is not an option -- anything over the embedding model's
    2,048-token input limit is rejected outright, losing the provision entirely.
    """
    pieces = _SENTENCE_END.split(line)
    out: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for piece in pieces:
        cost = counter.count(piece)
        if cost > budget:
            if current:
                out.append(" ".join(current))
                current, current_tokens = [], 0
            out.extend(_split_words(piece, budget, counter))
            continue
        if current and current_tokens + cost > budget:
            out.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(piece)
        current_tokens += cost

    if current:
        out.append(" ".join(current))
    return out or [line]


def _split_words(text: str, budget: int, counter: TokenCounter) -> list[str]:
    """Last resort: pack words up to the budget. Only reached by pathological input."""
    out: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in text.split():
        cost = max(1, counter.count(word))
        if current and current_tokens + cost > budget:
            out.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += cost
    if current:
        out.append(" ".join(current))
    return out or [text]


def _split_body(body: str, breadcrumb: str, counter: TokenCounter) -> list[str]:
    """Split an over-long unit on line boundaries.

    Our serialiser emits one line per block element, so line boundaries fall between
    numbered points rather than mid-sentence. A quote can therefore never be split in half,
    which keeps citation verification able to find it.
    """
    budget = settings.chunk_max_tokens - counter.count(breadcrumb) - 8  # separator slack
    if budget <= 0:
        raise ValueError(f"breadcrumb alone exceeds the chunk budget: {breadcrumb!r}")

    if counter.count(body) <= budget:
        return [body]

    target = min(settings.chunk_target_tokens, budget)
    out: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for line in body.split("\n"):
        line_tokens = counter.count(line)

        # A single line longer than the whole budget. Our serialiser emits one line per
        # block element, so this means one very long unbroken provision.
        if line_tokens > budget:
            if current:
                out.append("\n".join(current))
                current, current_tokens = [], 0
            out.extend(_split_long_line(line, budget, counter))
            continue

        if current and current_tokens + line_tokens > target:
            out.append("\n".join(current))
            current, current_tokens = [], 0

        current.append(line)
        current_tokens += line_tokens

    if current:
        out.append("\n".join(current))
    return out
