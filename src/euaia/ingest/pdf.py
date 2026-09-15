"""Read EUR-Lex PDFs into the legal units the rest of ingestion works with.

The two documents are published differently, so there are two readers:

* :func:`parse_consolidated` reads the consolidated act -- chapters, sections, articles,
  paragraphs and annexes -- from the bookmark outline its PDF carries.
* :func:`parse_recitals` reads the recitals from the as-adopted act's preamble, which has no
  outline to read.

Both return a :class:`ParsedDocument`, so the chunker and the pipeline never need to know
which one ran. The sections below follow that order: the shared shape, the outline, the
consolidated reader, the recitals reader.
"""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
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
