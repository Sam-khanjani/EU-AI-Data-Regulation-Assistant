"""Read the corpus' PDFs into the units the rest of ingestion works with.

Each kind of document is published differently, so there is a reader per kind:

* :func:`parse_consolidated` reads the consolidated act -- chapters, sections, articles,
  paragraphs and annexes -- from the bookmark outline its PDF carries.
* :func:`parse_recitals` reads the recitals from the as-adopted act's preamble, which has no
  outline to read.
* :func:`parse_sections` reads the Commission's guidelines, codes of practice and Q&A, which
  are not legislation and have no articles -- only numbered sections, and sometimes not even
  those.

All return a :class:`ParsedDocument`, so the chunker and the pipeline never need to know which
one ran. The sections below follow that order: the shared shape, the outline, the consolidated
reader, the recitals reader, the section reader.
"""

from __future__ import annotations

import io
import logging
import re
from bisect import bisect_left
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from itertools import count

import pdfplumber
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import PDFObjRef, resolve1

log = logging.getLogger(__name__)


# -------------------------------------------------------------- parsed document

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
    context: str | None = None
    """Where the unit sits and what it is, for documents whose numbering alone cannot say
    (see :func:`parse_code`). Embedded with the chunk and shown to the answer model; never
    part of ``text``, so it can never be quoted as source."""


@dataclass(slots=True)
class ParsedDocument:
    units: list[ParsedUnit]
    """Flat list in document order; hierarchy is expressed by ``parent_path``."""

    consolidation_date: str | None = None

    def by_type(self, unit_type: str) -> list[ParsedUnit]:
        return [u for u in self.units if u.unit_type == unit_type]


# ------------------------------------------------------------- bookmark outline
# Read the structure EUR-Lex declares inside its own PDFs.
#
# Every EUR-Lex PDF carries a bookmark outline -- the navigation tree a PDF reader shows in
# its sidebar. That outline is **publisher-authored data, not typesetting**, and it names every
# chapter, section, article and annex in the act, nested exactly as the act nests them.
#
# Why that matters here: the alternative is inferring structure from fonts and margins, which
# is a guess that a stylesheet change silently invalidates. Reading the outline asks the
# publisher what the document contains instead. Checked against EUR-Lex's own structured
# (Formex) edition of the same act during development, the outline was exact on every count:
#
#     articles 119, sections 16, annexes 14, chapters 13
#
# including ``Article 4a`` and the other amendment-inserted articles that font heuristics are
# most likely to miss.
#
# Two quirks of the real files are handled here:
#
# * **Titles arrive mis-decoded.** ``'Regulation\x92(EU)'`` and ``'\x84high-\x84risk'`` are
#   EUR-Lex's non-breaking space and non-breaking hyphen surviving a legacy codepage. Left
#   alone they would corrupt every heading.
# * **Destinations are indirect.** A bookmark points at a *named* destination; the name resolves
#   through the catalog's ``/Names /Dests`` tree to a page object, and only then to a page
#   number. That indirection is why a naive ``dest[0]`` lookup returns nothing.

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


# ------------------------------------------------------------- consolidated act
# Parse a consolidated EUR-Lex act from its PDF.
#
# The design rule here is **read, don't infer**. Structure comes from the PDF's own bookmark
# outline (read in the section above), which the publisher authors deliberately and
# which reproduced EUR-Lex's own structured (Formex) edition exactly when the two were
# compared during development -- 119 articles, 16 sections, 14 annexes, 13 chapters.
# Nothing in this reader decides what a heading is by looking at fonts or margins; a stylesheet
# change at EUR-Lex would leave it working.
#
# That leaves three jobs, in descending order of confidence:
#
# **Skeleton** -- taken wholesale from the outline, including the page each unit starts on.
#
# **Bodies** -- sliced out of the page text between one unit's heading and the next. Because the
# boundaries are known in advance, this is a slice rather than a search.
#
# **Paragraphs** -- the one thing the outline does not carry, split on the ``1.`` / ``2.``
# numbering. That numbering is drafting convention mandated for EU legislation rather than
# typesetting: provisions have to cite each other as "Article 6(2)", so the numbers cannot
# quietly disappear the way a font can.
#
# Three artefacts of the real files are repaired on the way through:
#
# * ``▼B`` / ``▼M1`` consolidation markers -- editorial annotations showing which amendment
#   produced a passage. 237 of them, on their own lines. Not legal text, so they are dropped.
# * The running page header (``02024R1689 — EN — 27.07.2026 — 001.001 — 61``).
# * Words hyphenated across a line break (``cumu\xad\nlative``). These are rejoined *here*
#   rather than in ``euaia.verify.normalize``, because that module discards soft hyphens and
#   then collapses whitespace -- correct for Formex, where the pattern is ``sub\xadject`` with
#   no line break, but it would turn ``cumu\xad\nlative`` into ``cumu lative``.
#
# **Known limitation: superscripts are lost by the PDF text layer.** Article 51's systemic
# risk threshold of 10^25 FLOP extracts as ``'102 5'``. Nothing here can recover it -- the
# information is absent from the file. It does not raise an error and it reads like a
# plausible number. ``verify/citations.py`` cannot catch it either: a quote containing
# ``102 5`` verifies, because it genuinely appears in the ingested text. The verifier is
# right and the source is wrong. ``TestKnownTextLayerDamage`` pins it so a future fix is
# noticed; treat numeric thresholds read from a PDF as suspect until checked by hand.

SOFT_HYPHEN = "­"

# '02024R1689 — EN — 27.07.2026 — 001.001 — 61'
_RUNNING_HEADER = re.compile(r"^\S+\s+—\s+[A-Z]{2}\s+—\s+[\d.]+\s+—\s+[\d.]+\s+—\s+\d+\s*$")
_HEADER_DATE = re.compile(r"—\s+[A-Z]{2}\s+—\s+(\d{2}\.\d{2}\.\d{4})\s+—")

# A line that is nothing but a consolidation marker: '▼B', '▼M1', '►M1'.
_MARKER_LINE = re.compile(r"^[▲▼►◄]\s*[BM]\d*\s*$")
# The marker glyphs themselves, for the handful that appear inline.
_MARKER_CHARS = re.compile(r"[▲▼►◄]")

# '1.', '2.', '3a.' at the start of a line -- a numbered paragraph of an article.
_PARAGRAPH_START = re.compile(r"^(\d+[a-z]?)\.\s+\S")

# Units the outline carries that we turn into structural units. 'other' entries are the
# document's own title blocks and the "Amended by:" list, which are not citable provisions.
_KEEP = frozenset({"chapter", "section", "article", "annex"})

_PATH_PREFIX = {"chapter": "CH", "section": "SEC", "article": "ART", "annex": "ANX"}

# How much of the outline may fail to match a heading before the parse is called a failure.
# Not zero: the outline carries a handful of entries whose heading is typeset unusually, and
# the real document currently matches every one. Not lenient either -- see `parse_consolidated`.
MAX_MISSING_FRACTION = 0.02
MAX_REPORTED_MISSING = 5


@dataclass(frozen=True, slots=True)
class _Line:
    text: str
    page: int


class PdfParseError(RuntimeError):
    """The PDF could not be parsed into a usable unit tree."""


def _clean_lines(pdf: pdfplumber.PDF) -> tuple[list[_Line], str | None]:
    """Page text as lines, with page attribution and the known noise removed.

    Returns the consolidation date alongside, because the running header is the only place it
    appears and this is the one pass that sees headers before discarding them.
    """
    lines: list[_Line] = []
    consolidation_date: str | None = None

    for page in pdf.pages:
        for raw in (page.extract_text() or "").split("\n"):
            text = raw.strip()
            if not text or _MARKER_LINE.match(text):
                continue
            if _RUNNING_HEADER.match(text):
                if consolidation_date is None and (found := _HEADER_DATE.search(text)):
                    day, month, year = found.group(1).split(".")
                    consolidation_date = f"{year}-{month}-{day}"
                continue
            text = _MARKER_CHARS.sub("", text).strip()
            if text:
                lines.append(_Line(text=text, page=page.page_number))

    return lines, consolidation_date


def _join(lines: list[_Line]) -> str:
    """Join lines into a body, repairing words broken across a line break.

    ``'cumu\\xad'`` followed by ``'lative'`` is one word. Joining without the separator is the
    only place this can be fixed -- once the lines are concatenated with newlines, the
    downstream normaliser cannot tell the break from ordinary whitespace.
    """
    out: list[str] = []
    for line in lines:
        if out and out[-1].endswith(SOFT_HYPHEN):
            out[-1] = out[-1][: -len(SOFT_HYPHEN)] + line.text
        else:
            out.append(line.text)
    return "\n".join(out)


def _label(entry: OutlineEntry) -> str:
    """How the unit's heading appears as a line in the body text."""
    word = {"article": "Article", "chapter": "CHAPTER", "section": "SECTION", "annex": "ANNEX"}[
        entry.unit_type
    ]
    return f"{word} {entry.unit_number}"


def _locate(lines: list[_Line], entry: OutlineEntry, after: int) -> int | None:
    """Index of the line where this unit's heading sits.

    The outline already says which page to look on, so this is a bounded scan rather than a
    document-wide search: start at the declared page and allow one page of slack for a unit
    whose heading sits at a page boundary. ``after`` keeps the scan monotonic so a heading is
    never matched before the previous unit's.
    """
    if entry.page is None:
        return None
    label = _label(entry).casefold()
    limit = entry.page + 1

    for index in range(after, len(lines)):
        line = lines[index]
        if line.page < entry.page:
            continue
        if line.page > limit:
            break
        if line.text.casefold() == label:
            return index
    return None


def _split_paragraphs(
    body: list[_Line], article: ParsedUnit, ordinals: Iterator[int]
) -> list[ParsedUnit]:
    """Numbered paragraphs of an article: the granularity we embed at.

    An article with no numbered paragraphs (short
    articles are a single unnumbered block) yields nothing, because the article unit already
    carries the whole text.
    """
    starts = [i for i, line in enumerate(body) if _PARAGRAPH_START.match(line.text)]
    if not starts:
        return []

    out: list[ParsedUnit] = []
    bounds = [*starts, len(body)]
    for i, start in enumerate(starts):
        chunk = body[start : bounds[i + 1]]
        match = _PARAGRAPH_START.match(chunk[0].text)
        assert match is not None  # `starts` was built from the same predicate
        number = match.group(1)
        text = _join(chunk)
        if not text:
            continue
        out.append(
            ParsedUnit(
                unit_type="paragraph",
                unit_number=f"{article.unit_number}({number})" if article.unit_number else number,
                unit_path=f"{article.unit_path}/PAR_{number}",
                heading=None,
                text=text,
                ordinal=next(ordinals),
                parent_path=article.unit_path,
                page=chunk[0].page,
            )
        )
    return out


def parse_consolidated(pdf_bytes: bytes) -> ParsedDocument:
    """Parse a EUR-Lex PDF into a flat, ordered list of citable units."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        lines, consolidation_date = _clean_lines(pdf)

    document = PDFDocument(PDFParser(io.BytesIO(pdf_bytes)))
    outline = [e for e in read_outline(document) if e.unit_type in _KEEP]
    if not outline:
        raise PdfParseError(
            "PDF has no usable bookmark outline; structure cannot be read from this file"
        )

    # Locate every heading first, so each unit's body can be sliced at the next one.
    located: list[tuple[OutlineEntry, int]] = []
    missing: list[str] = []
    cursor = 0
    for entry in outline:
        index = _locate(lines, entry, cursor)
        if index is None:
            missing.append(f"{entry.title} (page {entry.page})")
            continue
        located.append((entry, index))
        cursor = index + 1

    # A few unlocated headings are tolerable; many mean the layout moved out from under the
    # page-bounded scan. Failing here matters because the caller activates whatever it gets:
    # a partial *parse* would otherwise become a complete-looking *ingest*, and a corpus
    # quietly missing twenty articles would answer questions as though they did not exist.
    if missing:
        log.warning("%d outline entries did not match a heading line", len(missing))
        for title in missing[:MAX_REPORTED_MISSING]:
            log.warning("  unmatched: %s", title)
    if len(missing) > MAX_MISSING_FRACTION * len(outline):
        raise PdfParseError(
            f"{len(missing)} of {len(outline)} outline entries did not match a heading line "
            f"in the page text (limit {MAX_MISSING_FRACTION:.0%}); the document layout has "
            f"probably changed. First unmatched: {'; '.join(missing[:MAX_REPORTED_MISSING])}"
        )
    if not located:
        raise PdfParseError("outline present but no heading matched the page text")

    ordinals = count(1)
    units: list[ParsedUnit] = []
    # Path context by outline depth: an article sits under a section where one exists and
    # under the chapter otherwise, and the outline's own nesting already says which.
    stack: list[tuple[int, ParsedUnit]] = []

    for position, (entry, index) in enumerate(located):
        end = located[position + 1][1] if position + 1 < len(located) else len(lines)
        body = lines[index:end]

        while stack and stack[-1][0] >= entry.level:
            stack.pop()
        # Annexes stand alone: they are not part of any chapter.
        parent = stack[-1][1] if stack and entry.unit_type != "annex" else None

        prefix = f"{_PATH_PREFIX[entry.unit_type]}_{entry.unit_number}"
        unit = ParsedUnit(
            unit_type=entry.unit_type,
            unit_number=entry.unit_number,
            unit_path=f"{parent.unit_path}/{prefix}" if parent else prefix,
            heading=entry.heading,
            text=_join(body),
            ordinal=next(ordinals),
            parent_path=parent.unit_path if parent else None,
            page=entry.page,
        )
        units.append(unit)
        stack.append((entry.level, unit))

        if entry.unit_type == "article":
            units.extend(_split_paragraphs(body, unit, ordinals))

    log.info(
        "Parsed PDF: %d units (%d articles, %d paragraphs, %d annexes)",
        len(units),
        sum(1 for u in units if u.unit_type == "article"),
        sum(1 for u in units if u.unit_type == "paragraph"),
        sum(1 for u in units if u.unit_type == "annex"),
    )
    return ParsedDocument(units=units, consolidation_date=consolidation_date)


# --------------------------------------------------------------------- recitals
# Read the recitals out of an as-adopted EU act's PDF.
#
# Recitals are the numbered ``(1) (2) (3)`` paragraphs of the preamble -- the reasoning behind
# the rules. They are not binding, but courts read them to establish what an article means, so
# an assistant that can only quote articles cannot answer "why".
#
# They need their own reader, separate from the consolidated reader above, because the
# document they live in is published differently. The consolidated act carries a 165-entry
# bookmark outline naming every article; the as-adopted act carries **14** bookmarks, all
# annexes. There is no outline to read recitals from, so structure has to come from the text.
#
# Three signals do it, and the order matters:
#
# 1. **The region.** Recitals sit between the preamble and the enacting formula
#    ``HAVE ADOPTED THIS REGULATION``. That phrase is part of the law, not the layout.
# 2. **The size.** A recital's number is set on its own line at 9pt while the body runs at
#    10pt; footnote prose is also 9pt. Keeping body-size lines therefore drops footnote text
#    wholesale, which matters because footnotes are numbered ``(1) (2) (3)`` too and would
#    otherwise be indistinguishable from recitals.
# 3. **The sequence.** Recitals run 1..N with no gaps. Any ``(N)`` that does not continue the
#    run is not the next recital.
#
# **Why 3 is not optional, and why 2 alone is not enough.** An earlier attempt used the
# sequence rule by itself and returned exactly 180 recitals numbered 1..180 -- the right count,
# and wrong: it had taken a footnote reading *"(4) Position of the European Parliament of 13
# March 2024"* as recital 4, and rejected the real one on the next page. A correct total is not
# evidence of correct content.
#
# That pairing is also what makes a size-based rule safe to rely on. If EUR-Lex changes the
# typography, the size filter stops matching, the run fails to reach its expected length, and
# ``parse_recitals`` raises rather than returning a plausible-looking fragment of the
# preamble. The size is a filter; the law's own numbering is the proof.

# Point sizes in the as-adopted PDF. The recital *number* is set smaller than the prose it
# introduces, and footnotes are set at the number's size -- which is what lets one rule
# separate body text from footnote text.
RECITAL_MARKER_SIZE = 8.5
RECITAL_BODY_SIZE = 9.6

# The enacting formula. Everything after it is operative text, not preamble.
ENACTING_FORMULA = "HAVE ADOPTED THIS REGULATION"

# A recital number. It hangs at the start of the first body line rather than sitting on a
# line of its own, so this matches the leading token, not the whole line.
_RECITAL_MARKER = re.compile(r"^\((\d{1,3})\)$")

# Running header ('OJ L, 12.7.2024') and footer ('2/144  ELI: http://data.europa.eu/...').
_NOISE = re.compile(r"^(OJ\s+[LC],|ELI:|\d+/\d+\s+ELI:|https?://data\.europa\.eu/)")

# Lines within this many points of the page edge are furniture, never provision text. Only a
# fallback: _NOISE catches the real header and footer by what they say.
_MARGIN = 40

# A preamble that yields fewer than this is not a preamble we understood.
MIN_RECITALS = 20

# What a footnote reference leaves behind. In the body its digit is set as a superscript at a
# smaller size, so filtering to body size removes the digit and strands its brackets: '( )'.
# EUR-Lex's structured edition drops footnote markers outright, so removing these is what
# made the two agree during development rather than a cosmetic tidy-up.
_EMPTY_REF = re.compile(r"\s*\(\s*\)")


def _word_lines(page: pdfplumber.page.Page) -> list[list[dict]]:
    """Group a page's words into lines, dropping header and footer."""
    rows: dict[int, list[dict]] = {}
    for word in page.extract_words(extra_attrs=["size"]):
        top = round(word["top"])
        if top < _MARGIN or top > page.height - _MARGIN:
            continue
        # Words on one visual line vary by a point or two; bucket them together.
        key = next((k for k in rows if abs(k - top) <= 2), top)
        rows.setdefault(key, []).append(word)

    out: list[list[dict]] = []
    for top in sorted(rows):
        words = sorted(rows[top], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in words).strip()
        if not text or _NOISE.match(text):
            continue
        out.append(words)
    return out


def _recital_line(words: list[dict]) -> tuple[str | None, str]:
    """Split a line into (recital number if it starts one, body text).

    The size pattern is the whole trick. A recital's first line reads
    ``[8.5, 9.6, 9.6, ...]`` -- a small hanging number followed by prose -- while a footnote
    reads ``[8.5, 8.5, 8.5, ...]`` all the way across. Same ``(N)`` token, different line.
    """
    sizes = [round(w["size"], 1) for w in words]

    marker = _RECITAL_MARKER.match(words[0]["text"])
    if marker and sizes[0] == RECITAL_MARKER_SIZE and RECITAL_BODY_SIZE in sizes[1:]:
        rest = " ".join(w["text"] for w in words[1:] if round(w["size"], 1) == RECITAL_BODY_SIZE)
        return marker.group(1), rest.strip()

    # Continuation of a recital. Footnote lines carry no body-size words and fall through.
    if RECITAL_BODY_SIZE in sizes:
        return None, " ".join(
            w["text"] for w in words if round(w["size"], 1) == RECITAL_BODY_SIZE
        ).strip()
    return None, ""


def parse_recitals(pdf_bytes: bytes, *, expected: int | None = None) -> ParsedDocument:
    """Parse an as-adopted act's preamble into recital units.

    ``expected`` pins the count when the caller knows it. Leave it unset to accept whatever
    complete 1..N run the document yields; the no-gaps requirement still applies either way.
    """
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        body: list[_Line] = []
        cuts: list[tuple[int, str, int]] = []
        found_end = False

        for number, page in enumerate(pdf.pages, start=1):
            for words in _word_lines(page):
                text = " ".join(w["text"] for w in words)
                if ENACTING_FORMULA in text:
                    found_end = True
                    break
                marker, prose = _recital_line(words)
                if marker is not None:
                    cuts.append((len(body), marker, number))
                if prose:
                    body.append(_Line(text=prose, page=number))
            if found_end:
                break

    if not found_end:
        raise PdfParseError(
            f"no {ENACTING_FORMULA!r} found; this PDF does not look like an as-adopted act, "
            "and without that boundary recitals cannot be told from operative text"
        )

    # Only a (N) that continues the run starts the next recital.
    kept: list[tuple[int, str, int]] = []
    want = 1
    for index, number, page in cuts:
        if int(number) == want:
            kept.append((index, number, page))
            want += 1

    if len(kept) < MIN_RECITALS:
        raise PdfParseError(
            f"found only {len(kept)} recitals in the preamble (minimum {MIN_RECITALS}); "
            f"the {RECITAL_MARKER_SIZE}pt/{RECITAL_BODY_SIZE}pt typography this reader depends "
            "on has probably changed"
        )
    if expected is not None and len(kept) != expected:
        raise PdfParseError(
            f"expected {expected} recitals, found {len(kept)} running 1..{kept[-1][1]}"
        )

    ordinals = count(1)
    units: list[ParsedUnit] = []
    for position, (index, number, page) in enumerate(kept):
        end = kept[position + 1][0] if position + 1 < len(kept) else len(body)
        text = _EMPTY_REF.sub("", _join(body[index:end]))
        if not text:
            continue
        units.append(
            ParsedUnit(
                unit_type="recital",
                unit_number=number,
                unit_path=f"RCT_{number}",
                heading=None,
                # The number leads the text, matching how a recital is quoted and cited.
                text=f"({number})\n{text}",
                ordinal=next(ordinals),
                page=page,
            )
        )

    log.info("Parsed PDF preamble: %d recitals (1..%s)", len(units), units[-1].unit_number)
    return ParsedDocument(units=units)


# --------------------------------------------------------------------- sections
# Read the Commission's guidelines, codes of practice and Q&A. These are not legislation and
# are not shaped like it: no articles or annexes, just decimal-numbered sections ("2.3.1."),
# or Commitments and Measures in the codes. So everything becomes a `section` unit and the
# hierarchy comes from the document's own outline nesting rather than from parsed numbering.

_SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
# Table-of-contents rows ("2.5. Interplay with ... ....... 9") match the heading pattern too.
_TOC_LINE = re.compile(r"\.{4,}\s*\d+\s*$|\s\d{1,3}$")
_MAX_HEADING_CHARS = 120
_NEEDLE_CHARS = 40
MIN_SECTIONS = 5
_MIN_PAGE_SPAN = 0.5
"""How the heading scan is judged to have failed, after which the document is kept whole
rather than sliced on whatever few lines happened to match: too few headings, or headings
that do not reach across the document."""


def _plain_lines(pdf: pdfplumber.PDF) -> list[_Line]:
    """Page text as lines. No noise filtering: these are not EUR-Lex typeset documents."""
    return [
        _Line(text=stripped, page=page.page_number)
        for page in pdf.pages
        for raw in (page.extract_text() or "").split("\n")
        if (stripped := raw.strip())
    ]


def _collapse(text: str) -> str:
    return " ".join(text.split()).casefold()


def _numbered(title: str) -> tuple[str | None, str]:
    """``'2.3.1. Classification'`` -> ``('2.3.1', 'Classification')``; else ``(None, title)``."""
    if match := _SECTION_NUMBER.match(title):
        return match.group(1), match.group(2).strip()
    return None, title


def _as_tuple(number: str) -> tuple[int, ...]:
    return tuple(int(part) for part in number.split("."))


def _headings_from_text(lines: list[_Line]) -> list[OutlineEntry]:
    """Numbered headings read from the body, for documents published without an outline.

    Depth comes from the numbering itself -- ``2.3.1`` is three levels down -- which is more
    reliable here than in the Act, because these documents number their headings consistently
    and the numbering is the only structure they have.

    Two things are rejected, both of which produced nonsense before they were: a contents
    page, whose rows carry the same numbering as the real headings, and numbered *list* items,
    which are everywhere in the Q&A. A contents row is dropped by keeping the last occurrence
    of each number, since the contents always precede the body; a list item is dropped by
    requiring the text to read like a title.
    """
    candidates: dict[str, tuple[int, OutlineEntry]] = {}
    for index, line in enumerate(lines):
        if len(line.text) > _MAX_HEADING_CHARS or _TOC_LINE.search(line.text):
            continue
        number, heading = _numbered(line.text)
        # A heading opens with a capital and does not run on into the next clause.
        if not number or not heading[:1].isupper() or heading.endswith((";", ",")):
            continue
        candidates[number] = (
            index,
            OutlineEntry(
                level=number.count(".") + 1,
                unit_type="section",
                unit_number=number,
                heading=heading,
                page=line.page,
                title=line.text,
            ),
        )
    # Re-sorted by where the *kept* occurrence sits: a dict preserves first-insertion order,
    # which is the contents page, so without this every body heading would arrive in contents
    # order and the monotonic filter would discard almost all of them.
    ordered = [entry for _, entry in sorted(candidates.values(), key=lambda pair: pair[0])]
    return _monotonic(ordered)


def _monotonic(entries: list[OutlineEntry]) -> list[OutlineEntry]:
    """The largest subset whose numbering only increases, in document order.

    Real headings ascend; a numbered list inside a section, or a contents row that escaped
    the filter above, breaks the run. Taking the *longest* increasing subsequence rather than
    scanning greedily from the first entry matters because the first entry is the likeliest
    to be the stray one -- a single leaked contents row at the front would otherwise discard
    every genuine heading numbered below it.
    """
    numbers = [_as_tuple(e.unit_number) for e in entries if e.unit_number]
    if not numbers:
        return []

    tails: list[int] = []  # index, per run length, of the smallest number ending such a run
    previous = [-1] * len(numbers)
    for index, number in enumerate(numbers):
        position = bisect_left([numbers[t] for t in tails], number)
        if position:
            previous[index] = tails[position - 1]
        if position == len(tails):
            tails.append(index)
        else:
            tails[position] = index

    run: list[OutlineEntry] = []
    cursor = tails[-1]
    while cursor != -1:
        run.append(entries[cursor])
        cursor = previous[cursor]
    return run[::-1]


def _section_entries(lines: list[_Line], document: PDFDocument) -> list[OutlineEntry]:
    """The document's own outline where it has one, otherwise headings read from the text.

    Outline titles are re-split here so a published ``2.1.`` becomes the unit's number rather
    than being replaced by its position: the numbering is what a reader would cite.
    """
    outline = [entry for entry in read_outline(document) if entry.title]
    if len(outline) >= MIN_SECTIONS:
        return [
            replace(entry, unit_number=number, heading=heading)
            for entry in outline
            for number, heading in [_numbered(entry.title)]
        ]
    return _headings_from_text(lines)


def _locate_heading(lines: list[_Line], entry: OutlineEntry, after: int) -> int | None:
    """Index of the line carrying this heading, scanned forward from ``after``.

    Matched on the heading text rather than on a synthesised label: unlike ``Article 6``,
    these headings have no predictable form. A bookmark title and its typeset line can differ
    in trailing punctuation and can wrap, so the comparison is prefix-based in both
    directions.
    """
    needle = _collapse(entry.title)[:_NEEDLE_CHARS]
    if not needle:
        return None
    limit = entry.page + 1 if entry.page else None

    for index in range(after, len(lines)):
        line = lines[index]
        if entry.page and line.page < entry.page:
            continue
        if limit and line.page > limit:
            break
        text = _collapse(line.text)
        if text.startswith(needle) or (text and needle.startswith(text)):
            return index
    return None


def parse_sections(pdf_bytes: bytes) -> ParsedDocument:
    """Parse a Commission publication into ``section`` units.

    Falls back to one unit for the whole document when too few headings are found. That is a
    real outcome rather than an error: the Q&A pages are rendered from HTML and carry no
    heading structure at all, and a single correctly-attributed unit is better than slicing
    on whatever lines happened to match. The chunker splits it either way.
    """
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        lines = _plain_lines(pdf)
    if not lines:
        raise PdfParseError("PDF has no extractable text")

    entries = _section_entries(lines, PDFDocument(PDFParser(io.BytesIO(pdf_bytes))))

    located: list[tuple[OutlineEntry, int]] = []
    cursor = 0
    for entry in entries:
        index = _locate_heading(lines, entry, cursor)
        if index is None:
            continue
        located.append((entry, index))
        cursor = index + 1

    # Headings that cluster on a couple of pages are a numbered list inside one section, not
    # the document's structure. Slicing on those would attribute most of the text to whatever
    # fragment happened to precede it, so the document is kept whole instead.
    pages = lines[-1].page
    spanned = located[-1][0].page and located[0][0].page
    if spanned and (located[-1][0].page - located[0][0].page) < _MIN_PAGE_SPAN * pages:
        located = []

    if len(located) < MIN_SECTIONS:
        log.info("No usable heading structure (%d found); keeping document whole", len(located))
        return ParsedDocument(
            units=[
                ParsedUnit(
                    unit_type="section",
                    unit_number=None,
                    unit_path="SEC_1",
                    heading=None,
                    text=_join(lines),
                    ordinal=1,
                    parent_path=None,
                    page=lines[0].page,
                )
            ]
        )

    ordinals = count(1)
    units: list[ParsedUnit] = []
    stack: list[tuple[int, ParsedUnit]] = []
    seen: set[str] = set()

    for position, (entry, index) in enumerate(located):
        end = located[position + 1][1] if position + 1 < len(located) else len(lines)

        while stack and stack[-1][0] >= entry.level:
            stack.pop()
        parent = stack[-1][1] if stack else None

        # Numbering is the stable part of a heading; where there is none (the codes name
        # their units "Commitment 1"), position stands in so the path is still unique.
        number = entry.unit_number or str(position + 1)
        prefix = f"SEC_{number}"
        path = f"{parent.unit_path}/{prefix}" if parent else prefix
        while path in seen:  # repeated numbering across parts of one document
            prefix += "_"
            path = f"{parent.unit_path}/{prefix}" if parent else prefix
        seen.add(path)

        unit = ParsedUnit(
            unit_type="section",
            unit_number=entry.unit_number,
            unit_path=path,
            heading=entry.heading or entry.title,
            text=_join(lines[index:end]),
            ordinal=next(ordinals),
            parent_path=parent.unit_path if parent else None,
            page=entry.page,
        )
        units.append(unit)
        stack.append((entry.level, unit))

    log.info("Parsed PDF: %d sections", len(units))
    return ParsedDocument(units=units)


# ------------------------------------------------------------------- codes of practice
# The codes are read as sections first, then given the structure a reader would cite. Their
# outline names every part ("Commitment 1", "Measure 1.1", "Sub-measure 1.1.2") but the
# numbering is in words, so the section reader leaves it unparsed; and three parts hide units
# inside one block of text: the lettered recitals, the glossary table, and the Act's own text
# that code 08 quotes at the head of each commitment.

_CODE_PART = re.compile(
    r"^(Section|Commitment|Measure|Sub-measure|Appendix|Annex)\s+(\d+(?:\.\d+)*)[.:]?\s*(.*)$"
)
# "a) " in code 08, "(a) " in the GPAI codes. A title ends in ":" (08) or "." (safety code).
_RECITAL_ITEM = re.compile(r"^\(?([a-z])\)\s+")
_RECITAL_TITLE = re.compile(r"^(.{3,100}?)[:.]\s")
_LEGAL_TEXT = re.compile(r"^LEGAL TEXT\b:?\s*(.*)$")
_QUOTE_START = re.compile(r"^(\d+\.\s|Article \d)")
"""What the Act's text looks like when a code reproduces it: "2. Providers ... shall" or
"Article 3(64) AI Act: ...". The copyright code's "(1) In order to demonstrate ..." is the
code's own wording and deliberately does not match."""
_OWN_WORDS = re.compile(
    r"^(In order to|To fulfil|To effectively|Signatories|ADDITIONAL LEGAL TEXT)"
)
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
_MAX_NUMBER_CHARS = 120


def parse_code(pdf_bytes: bytes) -> ParsedDocument:
    """Parse a code of practice into the units a reader would cite, each with its context.

    Every unit gets a number ("S1 Measure 1.1" -- sections prefix it only where a code has
    more than one, as "Commitment 1" then exists twice) and a ``context`` line built from the
    code itself: its path of headings, the Act provisions its commitment implements (the
    ``LEGAL TEXT:`` line), and whether a measure is optional.
    """
    doc = parse_sections(pdf_bytes)
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        units = _split_code_units(doc.units, pdf)
    _describe_code_units(units)
    for ordinal, unit in enumerate(units, start=1):
        unit.ordinal = ordinal
    log.info("Parsed code of practice: %d units", len(units))
    return ParsedDocument(units=units)


def _split_code_units(units: list[ParsedUnit], pdf: pdfplumber.PDF) -> list[ParsedUnit]:
    """Number every unit, and split out recitals, glossary terms and quoted Act text."""
    out: list[ParsedUnit] = []
    for position, unit in enumerate(units):
        lines = [line for line in unit.text.split("\n") if not _PAGE_NUMBER.match(line.strip())]
        unit.text = "\n".join(lines)
        if match := _CODE_PART.match(unit.heading or ""):
            kind, number, title = match.groups()
            unit.unit_number, unit.heading = f"{kind} {number}", title or None
        else:  # "Objectives", "Recitals", "Glossary", ...: the name is the number
            unit.unit_number, unit.heading = (unit.heading or "")[:_MAX_NUMBER_CHARS], None

        if unit.unit_number == "Recitals":
            children = _split_recitals(unit, lines)
        elif unit.unit_number == "Glossary":
            end = units[position + 1].page if position + 1 < len(units) else len(pdf.pages)
            children = _split_glossary(unit, lines, pdf.pages[(unit.page or 1) - 1 : end])
        else:
            children = _split_legal_text(unit, lines)
        out.append(unit)
        out.extend(children)
    return out


def _child(parent: ParsedUnit, suffix: str, unit_type: str, number: str,
           heading: str | None, lines: list[str]) -> ParsedUnit:
    return ParsedUnit(
        unit_type=unit_type,
        unit_path=f"{parent.unit_path}/{suffix}",
        text="\n".join(lines),
        ordinal=0,
        unit_number=number[:_MAX_NUMBER_CHARS],
        heading=heading,
        parent_path=parent.unit_path,
        page=parent.page,
    )


def _split_recitals(unit: ParsedUnit, lines: list[str]) -> list[ParsedUnit]:
    """One unit per lettered recital, "a) Trust in the information ecosystem: ...".

    Letters must run a, b, c, ... in order, so a lettered list *inside* a recital cannot
    start a new one.
    """
    starts, expected = [], "a"
    for index, line in enumerate(lines):
        if (match := _RECITAL_ITEM.match(line)) and match.group(1) == expected:
            starts.append(index)
            expected = chr(ord(expected) + 1)
    if len(starts) < 2:
        return []

    children = []
    for n, start in enumerate(starts):
        body = lines[start : starts[n + 1] if n + 1 < len(starts) else len(lines)]
        match = _RECITAL_ITEM.match(body[0])
        letter = match.group(1)
        # A short opening phrase ended by ":" or "." is the recital's title; a recital that
        # opens with a full sentence ("The Signatories recognise ...") has none.
        found = _RECITAL_TITLE.match(" ".join(body)[match.end():])
        title = found.group(1).strip() if found else None
        if title and (len(title.split()) > 14 or title.startswith(("The ", "This "))):
            title = None
        children.append(_child(unit, f"REC_{letter}", "recital", f"Recital {letter}", title, body))
    unit.text = "\n".join(lines[: starts[0]])
    return children


def _split_glossary(unit: ParsedUnit, lines: list[str], pages) -> list[ParsedUnit]:
    """One unit per term, read from the glossary's two-column table.

    A row with two filled cells is a term; a one-cell row continues the previous definition
    across a page break. Tables with no two-cell row (a figure's label) are not glossary.
    """
    terms: list[list[str]] = []
    for page in pages:
        for table in page.extract_tables():
            current = None
            for row in table:
                # Cells wrap inside the table: rejoin "model-\nindependent" as one word.
                cells = [re.sub(r"(\w)- (\w)", r"\1-\2", " ".join(c.split()))
                         for c in row if c and c.strip()]
                if len(cells) == 2 and cells != ["Term", "Definition"]:
                    current = cells
                    terms.append(current)
                elif len(cells) == 1 and current:
                    current[1] += " " + cells[0]
    header = next((i for i, line in enumerate(lines) if _collapse(line) == "term definition"), None)
    if len(terms) < 2 or header is None:
        return []

    unit.text = "\n".join(lines[:header])
    return [
        _child(unit, f"TERM_{n}", "section", f"Glossary: {re.sub('[‘’]', '', term)}", None,
               [term, definition])
        for n, (term, definition) in enumerate(terms, start=1)
    ]


def _split_legal_text(unit: ParsedUnit, lines: list[str]) -> list[ParsedUnit]:
    """Move the Act's own text, where a code reproduces it, into a unit labelled as a quote.

    Code 08 opens each commitment with the paragraph of Article 50 it implements, verbatim
    ("Providers ... shall ensure ..."). Left inside the commitment, that obligation reads as
    the voluntary code's own words. The quote runs from the ``LEGAL TEXT`` line to where the
    code speaks for itself again.
    """
    at = next((i for i, line in enumerate(lines) if _LEGAL_TEXT.match(line)), None)
    if at is None:
        return []
    end = next((i for i in range(at + 1, len(lines)) if _OWN_WORDS.match(lines[i])), len(lines))
    quoted = lines[at + 1 : end]
    if not quoted or not _QUOTE_START.match(quoted[0]):
        return []

    refs = _LEGAL_TEXT.match(lines[at]).group(1).strip() or "the AI Act"
    unit.text = "\n".join(lines[: at + 1] + lines[end:])
    return [_child(unit, "LEGAL", "section", f"{unit.unit_number}, quoted AI Act text",
                   f"Quoted from {refs}", quoted)]


def _describe_code_units(units: list[ParsedUnit]) -> None:
    """Fill in ``context``, then prefix numbers with their section where a code has two."""
    by_path = {u.unit_path: u for u in units}

    def chain(unit: ParsedUnit) -> list[ParsedUnit]:
        path, out = unit.unit_path, []
        while path in by_path:
            out.insert(0, by_path[path])
            path = by_path[path].parent_path
        return out

    implements = {}
    for unit in units:
        for line in unit.text.split("\n"):
            if (match := _LEGAL_TEXT.match(line)) and match.group(1).strip():
                implements[unit.unit_path] = match.group(1).strip()

    for unit in units:
        path = chain(unit)
        parts = [" › ".join(
            u.heading if u.unit_path.endswith("/LEGAL")
            else f"{u.unit_number}: {u.heading}" if u.heading else u.unit_number
            for u in path
        )]
        if refs := next((implements[u.unit_path] for u in reversed(path)
                         if u.unit_path in implements), None):
            parts.append(f"implements {refs}")
        if unit.unit_number.split()[0] in ("Measure", "Sub-measure"):
            optional = any("(optional)" in (u.heading or "") for u in path)
            parts.append(
                "optional measure" if optional else "mandatory for signatories of the code"
            )
        if unit.unit_type == "recital":
            parts.append("recital: context for the commitments, not a commitment itself")
        if unit.unit_path.endswith("/LEGAL"):
            parts.append("AI Act text reproduced in the code, not the code's own words")
        unit.context = " | ".join(parts)

    sections = [u for u in units if u.unit_number.startswith("Section ") and not u.parent_path]
    if len(sections) > 1:
        for unit in units:
            top = chain(unit)[0]
            if top is not unit and top.unit_number.startswith("Section "):
                unit.unit_number = f"S{top.unit_number.split()[1]} {unit.unit_number}"
