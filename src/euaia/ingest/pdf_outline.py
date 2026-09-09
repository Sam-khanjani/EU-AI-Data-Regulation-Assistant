"""Read the structure EUR-Lex declares inside its own PDFs.

Every EUR-Lex PDF carries a bookmark outline -- the navigation tree a PDF reader shows in
its sidebar. That outline is **publisher-authored data, not typesetting**, and it names every
chapter, section, article and annex in the act, nested exactly as the act nests them.

Why that matters here: the alternative is inferring structure from fonts and margins, which
is a guess that a stylesheet change silently invalidates. Reading the outline asks the
publisher what the document contains instead. Checked against EUR-Lex's own structured
(Formex) edition of the same act during development, the outline was exact on every count:

    articles 119, sections 16, annexes 14, chapters 13

including ``Article 4a`` and the other amendment-inserted articles that font heuristics are
most likely to miss.

Two quirks of the real files are handled here:

* **Titles arrive mis-decoded.** ``'Regulation\\x92(EU)'`` and ``'\\x84high-\\x84risk'`` are
  EUR-Lex's non-breaking space and non-breaking hyphen surviving a legacy codepage. Left
  alone they would corrupt every heading.
* **Destinations are indirect.** A bookmark points at a *named* destination; the name resolves
  through the catalog's ``/Names /Dests`` tree to a page object, and only then to a page
  number. That indirection is why a naive ``dest[0]`` lookup returns nothing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdftypes import PDFObjRef, resolve1

log = logging.getLogger(__name__)

# EUR-Lex writes NBSP and non-breaking hyphens that reach us through a legacy codepage.
# Mapping them back is not cosmetic: 'Regulation\x92(EU)' must not become 'Regulation(EU)'.
_TITLE_FIXUPS = {
    "\x92": " ",  # no-break space: 'Regulation\x92(EU)'
    "\x84": "",  # non-breaking hyphen artefact: '\x84high-\x84risk'
    "\x93": "",
    "\x94": "",
}

_ARTICLE = re.compile(r"^Article\s+(\d+[a-z]?)\b\s*(.*)$", re.DOTALL)
_CHAPTER = re.compile(r"^CHAPTER\s+([IVXLC]+)\b\s*(.*)$", re.DOTALL)
_SECTION = re.compile(r"^SECTION\s+([IVXLC]+|\d+)\b\s*(.*)$", re.DOTALL)
_ANNEX = re.compile(r"^ANNEX\s+([IVXLC]+|\d+)\b\s*(.*)$", re.DOTALL)

# Ordered: ANNEX before SECTION so 'ANNEX I' is not mistaken for anything else, and ARTICLE
# first because it is by far the most common.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("article", _ARTICLE),
    ("chapter", _CHAPTER),
    ("annex", _ANNEX),
    ("section", _SECTION),
)


@dataclass(frozen=True, slots=True)
class OutlineEntry:
    """One bookmark: a unit the publisher says exists, and where it starts."""

    level: int
    """Nesting depth as published. Chapter 2, section 3, article 3 or 4 -- see
    :func:`read_outline` for why depth is not a reliable parent signal on its own."""

    unit_type: str
    """``article`` | ``chapter`` | ``section`` | ``annex`` | ``other``."""

    unit_number: str | None
    """Official numbering: ``'6'``, ``'4a'``, ``'III'``."""

    heading: str | None
    """Title without the number, e.g. ``'Classification rules for high-risk AI systems'``."""

    page: int | None
    """1-based page the unit starts on. ``None`` when the destination cannot be resolved."""

    title: str
    """The cleaned bookmark text, kept for diagnostics."""


def clean_title(raw: str) -> str:
    """Undo the codepage damage and collapse whitespace."""
    for bad, good in _TITLE_FIXUPS.items():
        raw = raw.replace(bad, good)
    return " ".join(raw.split())


def classify(title: str) -> tuple[str, str | None, str | None]:
    """Split a bookmark title into (unit_type, number, heading)."""
    for unit_type, pattern in _PATTERNS:
        match = pattern.match(title)
        if match:
            number, heading = match.group(1), match.group(2).strip()
            # A trailing apostrophe shows up on a few titles ("Article 1 Subject matter'").
            heading = heading.rstrip("'\"").strip()
            return unit_type, number, heading or None
    return "other", None, None


def _collect_named_destinations(doc: PDFDocument) -> dict[bytes, object]:
    """Flatten the catalog's ``/Names /Dests`` name tree into ``name -> destination``.

    The tree is stored as interior ``/Kids`` nodes with leaf ``/Names`` arrays laid out as
    ``[name1, dest1, name2, dest2, ...]``. This document splits its 165 destinations across
    nine leaves, which is why they have to be gathered rather than read from one place.
    """
    catalog = doc.catalog
    names_root = resolve1(catalog.get("Names"))
    if not isinstance(names_root, dict):
        return {}
    dests = resolve1(names_root.get("Dests"))
    if not isinstance(dests, dict):
        return {}

    out: dict[bytes, object] = {}

    def walk(node: object) -> None:
        node = resolve1(node)
        if not isinstance(node, dict):
            return
        pairs = resolve1(node.get("Names"))
        if isinstance(pairs, list):
            for i in range(0, len(pairs) - 1, 2):
                key = pairs[i]
                if isinstance(key, bytes):
                    out[key] = pairs[i + 1]
        kids = resolve1(node.get("Kids"))
        if isinstance(kids, list):
            for kid in kids:
                walk(kid)

    walk(dests)
    return out


def _page_index(doc: PDFDocument) -> dict[int, int]:
    """Map each page's object id to its 1-based page number."""
    from pdfminer.pdfpage import PDFPage

    return {
        page.pageid: number
        for number, page in enumerate(PDFPage.create_pages(doc), start=1)
        if isinstance(page.pageid, int)
    }


def _resolve_page(dest: object, named: dict[bytes, object], pages: dict[int, int]) -> int | None:
    """Follow a bookmark destination to a page number, or ``None`` if it does not resolve."""
    # A named destination is a lookup away from the real thing.
    if isinstance(dest, bytes):
        dest = named.get(dest)
    dest = resolve1(dest)

    # Named destinations wrap the array in a /D entry.
    if isinstance(dest, dict):
        dest = resolve1(dest.get("D"))

    if not isinstance(dest, list) or not dest:
        return None

    target = dest[0]
    objid = target.objid if isinstance(target, PDFObjRef) else target
    return pages.get(objid) if isinstance(objid, int) else None


def read_outline(doc: PDFDocument) -> list[OutlineEntry]:
    """Read every bookmark in document order.

    ``level`` is reported as published rather than normalised. Depth alone does not give the
    parent -- articles sit at depth 3 under a chapter but depth 4 under a section -- so
    callers should build parentage from the running chapter/section, using depth only to
    confirm it.
    """
    try:
        raw = list(doc.get_outlines())
    except Exception as exc:  # noqa: BLE001 -- pdfminer raises bare exceptions for "no outline"
        log.warning("PDF has no readable outline: %r", exc)
        return []

    named = _collect_named_destinations(doc)
    pages = _page_index(doc)

    entries: list[OutlineEntry] = []
    for level, title, dest, action, _se in raw:
        cleaned = clean_title(title or "")
        unit_type, number, heading = classify(cleaned)

        # Most bookmarks carry `dest`; a few PDFs use a /GoTo action instead.
        target = dest
        if target is None and action is not None:
            act = resolve1(action)
            if isinstance(act, dict):
                target = act.get("D")

        entries.append(
            OutlineEntry(
                level=level,
                unit_type=unit_type,
                unit_number=number,
                heading=heading,
                page=_resolve_page(target, named, pages),
                title=cleaned,
            )
        )

    unresolved = sum(1 for e in entries if e.page is None)
    if unresolved:
        log.warning("%d of %d bookmarks did not resolve to a page", unresolved, len(entries))
    return entries
