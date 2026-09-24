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
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import tiktoken

from euaia.config import settings
from euaia.ingest.pdf import ParsedDocument, ParsedUnit

log = logging.getLogger(__name__)

# Unit types worth embedding. What a `section` means depends on the document: in the Act it
# is a heading whose substance lives in the articles beneath it, so embedding it would add
# near-duplicates -- which is why the Act names its own types explicitly. In the Commission's
# guidelines and codes there are no articles and the section *is* the text.
_EMBEDDABLE = frozenset({"paragraph", "article", "recital", "annex", "section"})

# Tokens reserved between the breadcrumb and the body.
_SEPARATOR_TOKENS = 8

# Below this, a unit with children is only a heading over them, and embedding it adds a
# near-empty vector that matches any question sharing its title words. Its children carry
# its heading in their own breadcrumbs, and it is still stored as their parent.
_CONTAINER_TOKENS = 60


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


@cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate token count.

    Gemini's tokenizer is not public, so this uses a GPT tokenizer as a stand-in. That is
    fine because we only need a *conservative* bound: the configured ceiling
    (``chunk_max_tokens``) sits well below the model's real 2,048-token limit, leaving
    headroom for tokenizer disagreement. The embedder still handles an over-length
    rejection from the API defensively.
    """
    return len(_encoding().encode(text, disallowed_special=()))


def build_breadcrumb(doc_title: str, unit: ParsedUnit, by_path: dict[str, ParsedUnit]) -> str:
    """Human-readable location of a unit, e.g.

    ``Regulation (EU) 2024/1689 - Chapter III, Section 1, Article 6 - Classification rules
    for high-risk AI systems``

    A unit whose reader supplied a ``context`` (the codes of practice) uses that instead:
    their structure is named in words -- Commitment, Measure -- that the numbering-based
    trail below cannot express.
    """
    if unit.context:
        return f"{doc_title} - {unit.context}"
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
    allowed = _EMBEDDABLE if include_unit_types is None else (_EMBEDDABLE & include_unit_types)
    by_path = {u.unit_path: u for u in doc.units}

    paragraphs: dict[str, list[ParsedUnit]] = {}
    for unit in doc.units:
        if unit.unit_type == "paragraph" and unit.parent_path:
            paragraphs.setdefault(unit.parent_path, []).append(unit)
    parents = {u.parent_path for u in doc.units if u.parent_path}

    drafts: list[ChunkDraft] = []
    for unit in doc.units:
        if unit.unit_type == "paragraph":
            continue  # emitted as part of its parent article
        if unit.unit_type not in allowed:
            continue

        breadcrumb = build_breadcrumb(doc_title, unit, by_path)
        children = paragraphs.get(unit.unit_path, []) if unit.unit_type == "article" else []

        if children and "paragraph" in allowed:
            bodies = _pack([p.text for p in children], breadcrumb)
        elif not children and unit.unit_path in parents and (
            count_tokens(unit.text) < _CONTAINER_TOKENS
        ):
            continue  # a heading over its children ("Section 1", "Recitals: Whereas:")
        else:
            bodies = _split_body(unit.text, breadcrumb)

        for body in bodies:
            text = f"{breadcrumb}\n\n{body}"
            drafts.append(
                ChunkDraft(
                    unit_path=unit.unit_path,
                    text=text,
                    body=body,
                    token_count=count_tokens(text),
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


def _greedy_pack(
    pieces: list[str],
    *,
    limit: int,
    target: int,
    joiner: str,
    split_oversize: Callable[[str], list[str]],
) -> list[str]:
    """Join pieces, in order, into chunks of at most ``target`` tokens.

    The one packing loop behind every splitter below. Pieces accumulate until the next one
    would take the chunk past ``target``. A piece over ``limit`` on its own closes the chunk
    in progress and is handed to ``split_oversize`` for a finer split.
    """
    out: list[str] = []
    current: list[str] = []
    used = 0
    for piece in pieces:
        tokens = count_tokens(piece)
        if tokens > limit:
            if current:
                out.append(joiner.join(current))
                current, used = [], 0
            out.extend(split_oversize(piece))
            continue
        if current and used + tokens > target:
            out.append(joiner.join(current))
            current, used = [], 0
        current.append(piece)
        used += tokens
    if current:
        out.append(joiner.join(current))
    return out


def _body_budget(breadcrumb: str) -> int:
    """Tokens left for a chunk's body once its breadcrumb and separator are counted."""
    budget = settings.chunk_max_tokens - count_tokens(breadcrumb) - _SEPARATOR_TOKENS
    if budget <= 0:
        raise ValueError(f"breadcrumb alone exceeds the chunk budget: {breadcrumb!r}")
    return budget


def _pack(parts: list[str], breadcrumb: str) -> list[str]:
    """Pack sibling paragraphs into chunks of roughly ``chunk_target_tokens``.

    Paragraphs are kept whole and in order: a chunk boundary always falls between two
    paragraphs, never inside one. That matters for citation -- a quote split across two
    chunks could not be verified against either.

    A single paragraph larger than the budget is emitted alone and split by
    :func:`_split_body`, which breaks on line boundaries rather than mid-sentence.
    """
    budget = _body_budget(breadcrumb)
    packed = _greedy_pack(
        parts,
        limit=budget,
        target=min(settings.chunk_target_tokens, budget),
        joiner="\n",
        split_oversize=lambda part: _split_body(part, breadcrumb),
    )
    return packed or [""]


def _split_body(body: str, breadcrumb: str) -> list[str]:
    """Split an over-long unit on line boundaries.

    Lines here are *typeset* lines, since that is what the PDF readers emit, so a split can
    land mid-sentence.

    That cannot break citation: verification runs against ``structural_unit.text``, not
    against the chunk, and ``normalize_text`` collapses all whitespace before matching. The
    cost is retrieval quality -- a chunk cut mid-sentence is a slightly worse thing for the
    reranker to score -- not correctness.
    """
    budget = _body_budget(breadcrumb)
    if count_tokens(body) <= budget:
        return [body]
    # A single line longer than the whole budget -- an unusually long typeset line, or a
    # provision set as one unbroken run -- is split further by sentence.
    return _greedy_pack(
        body.split("\n"),
        limit=budget,
        target=min(settings.chunk_target_tokens, budget),
        joiner="\n",
        split_oversize=lambda line: _split_long_line(line, budget),
    )


_SENTENCE_END = re.compile(r"(?<=[.;:])\s+")


def _split_long_line(line: str, budget: int) -> list[str]:
    """Split one over-long line, preferring sentence boundaries, then words.

    Splitting inside a sentence is acceptable here in a way it would not be elsewhere,
    because a chunk boundary does not constrain citation: quotes are verified against the
    *structural unit's* full text, not against the chunk. A quote spanning a chunk boundary
    therefore still verifies. What a chunk boundary affects is only which passage retrieval
    surfaces.

    Leaving the line whole is not an option -- anything over the embedding model's
    2,048-token input limit is rejected outright, losing the provision entirely.
    """
    pieces = _greedy_pack(
        _SENTENCE_END.split(line),
        limit=budget,
        target=budget,
        joiner=" ",
        split_oversize=lambda sentence: _split_words(sentence, budget),
    )
    return pieces or [line]


def _split_words(text: str, budget: int) -> list[str]:
    """Last resort: pack words up to the budget. Only reached by pathological input.

    A single word over the budget becomes a chunk of its own.
    """
    words = _greedy_pack(
        text.split(), limit=budget, target=budget, joiner=" ", split_oversize=lambda w: [w]
    )
    return words or [text]
