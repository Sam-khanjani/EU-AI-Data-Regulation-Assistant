"""The parsed-document contract shared by every source format.

These types are what a parser produces and what the chunker and pipeline consume. They live
here rather than beside any one parser because there is more than one: ``pdf_parser.parse``
reads the consolidated act through its bookmark outline, ``pdf_recitals.parse_recitals`` reads
the as-adopted act's preamble, and both must hand back the same shape for the rest of
ingestion to stay reader-agnostic.

Deliberately dependency-free -- no pdfplumber, no database layer. A parser importing this
module should not thereby acquire another parser's dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ParsedUnit:
    """One node of the legal tree, ready to become a ``structural_unit`` row."""

    unit_type: str
    unit_path: str
    text: str
    ordinal: int
    unit_number: str | None = None
    heading: str | None = None
    parent_path: str | None = None
    page: int | None = None
    """1-based page the unit starts on, where the reader could establish it."""
    children: list[ParsedUnit] = field(default_factory=list)


@dataclass(slots=True)
class ParsedDocument:
    units: list[ParsedUnit]
    """Flat list in document order; hierarchy is expressed by ``parent_path``."""

    root_tag: str
    consolidation_date: str | None = None
    start_date: str | None = None

    def by_type(self, unit_type: str) -> list[ParsedUnit]:
        return [u for u in self.units if u.unit_type == unit_type]


class OrdinalCounter:
    """Monotonic document-order ordinal.

    Shared by both parsers rather than reimplemented in each. Two independent counters is
    exactly the kind of duplication that drifts -- one starting at 0, the other at 1 -- into a
    document order that is subtly wrong and that nothing directly tests.
    """

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n
