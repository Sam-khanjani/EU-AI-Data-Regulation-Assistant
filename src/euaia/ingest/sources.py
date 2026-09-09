"""The document sources we track.

The AI Act is registered as *two* sources, which is a deliberate modelling decision rather
than a workaround:

``eu-ai-act``
    The consolidated act. Operative text -- articles, paragraphs, annexes -- with all
    amendments applied. This is what changes over time, and what the change detector polls.

``eu-ai-act-recitals``
    The original as-adopted act, ingested for its 180 recitals only. Consolidation does not
    restate recitals, so they exist nowhere else; and legally they belong to the act as
    adopted, so they do not move when the operative text is amended.

The split matters for correctness, not tidiness. Both documents contain an "Article 6", but
only the consolidated one carries the amendments. Indexing both would let the assistant cite
superseded wording as if it were in force, so ``unit_types`` restricts the recitals source
to recitals.

Both are read from PDF, but from *different documents needing different readers*, which is
why ``parser`` is stated explicitly on both rather than defaulted -- see the field's docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Parser = Literal["pdf_outline", "pdf_preamble"]
"""Declared locally rather than imported from ``euaia.db.models``: this module is pure data
seeded into the ``source`` table, and should not drag SQLAlchemy in behind it."""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Static definition of a tracked source, seeded into the ``source`` table."""

    key: str
    title: str
    publisher: str
    source_type: str
    celex_base: str | None = None
    eli_uri: str | None = None
    landing_url: str | None = None
    check_interval_hours: int = 24

    use_consolidated: bool = True
    """Resolve the newest consolidated version rather than the as-adopted act."""

    parser: Parser = "pdf_outline"
    """Which reader turns this source's PDF into units.

    A property of the source rather than something sniffed from the bytes: getting it wrong
    would index the unamended text, so it is declared once, here.

    ``pdf_outline``
        Structure from the PDF's own bookmark outline. The consolidated act carries 165
        bookmarks naming every chapter, section, article and annex -- publisher-authored
        structure, so nothing has to be inferred from fonts.
    ``pdf_preamble``
        Recitals from the preamble text. The as-adopted act carries only 14 bookmarks
        (annexes alone), so there is no outline to read; the reader works from the enacting
        formula, the typographic size of the hanging number, and the 1..N run instead. See
        :mod:`euaia.ingest.pdf_recitals`.
    """

    unit_types: frozenset[str] | None = None
    """Restrict which parsed unit types get indexed. ``None`` means all embeddable types."""

    doc_title: str = ""
    """Prefix used in chunk breadcrumbs."""

    notes: str = ""


AI_ACT = SourceSpec(
    key="eu-ai-act",
    title="Regulation (EU) 2024/1689 (Artificial Intelligence Act)",
    publisher="Publications Office of the European Union",
    source_type="eurlex",
    celex_base="32024R1689",
    eli_uri="http://data.europa.eu/eli/reg/2024/1689/oj",
    landing_url="https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32024R1689",
    use_consolidated=True,
    parser="pdf_outline",
    unit_types=None,
    doc_title="Regulation (EU) 2024/1689 (AI Act)",
    notes="Consolidated operative text. Re-consolidated whenever the Act is amended.",
)

AI_ACT_RECITALS = SourceSpec(
    key="eu-ai-act-recitals",
    title="Regulation (EU) 2024/1689 (AI Act) - Recitals",
    publisher="Publications Office of the European Union",
    source_type="eurlex",
    celex_base="32024R1689",
    eli_uri="http://data.europa.eu/eli/reg/2024/1689/oj",
    landing_url="https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32024R1689",
    use_consolidated=False,
    parser="pdf_preamble",
    unit_types=frozenset({"recital"}),
    doc_title="Regulation (EU) 2024/1689 (AI Act)",
    notes="Recitals from the act as adopted; consolidation omits them.",
)

ALL_SOURCES: tuple[SourceSpec, ...] = (AI_ACT, AI_ACT_RECITALS)
BY_KEY: dict[str, SourceSpec] = {s.key: s for s in ALL_SOURCES}


def get(key: str) -> SourceSpec:
    try:
        return BY_KEY[key]
    except KeyError:
        known = ", ".join(sorted(BY_KEY))
        raise KeyError(f"unknown source {key!r}; known sources: {known}") from None
