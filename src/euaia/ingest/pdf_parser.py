"""Parse a consolidated EUR-Lex act from its PDF.

The design rule here is **read, don't infer**. Structure comes from the PDF's own bookmark
outline (see :mod:`euaia.ingest.pdf_outline`), which the publisher authors deliberately and
which reproduced EUR-Lex's own structured (Formex) edition exactly when the two were
compared during development -- 119 articles, 16 sections, 14 annexes, 13 chapters.
Nothing in this module decides what a heading is by looking at fonts or margins; a stylesheet
change at EUR-Lex would leave it working.

That leaves three jobs, in descending order of confidence:

**Skeleton** -- taken wholesale from the outline, including the page each unit starts on.

**Bodies** -- sliced out of the page text between one unit's heading and the next. Because the
boundaries are known in advance, this is a slice rather than a search.

**Paragraphs** -- the one thing the outline does not carry, split on the ``1.`` / ``2.``
numbering. That numbering is drafting convention mandated for EU legislation rather than
typesetting: provisions have to cite each other as "Article 6(2)", so the numbers cannot
quietly disappear the way a font can.

Three artefacts of the real files are repaired on the way through:

* ``▼B`` / ``▼M1`` consolidation markers -- editorial annotations showing which amendment
  produced a passage. 237 of them, on their own lines. Not legal text, so they are dropped.
* The running page header (``02024R1689 — EN — 27.07.2026 — 001.001 — 61``).
* Words hyphenated across a line break (``cumu\\xad\\nlative``). These are rejoined *here*
  rather than in :mod:`euaia.verify.normalize`, because that module discards soft hyphens and
  then collapses whitespace -- correct for Formex, where the pattern is ``sub\\xadject`` with
  no line break, but it would turn ``cumu\\xad\\nlative`` into ``cumu lative``.

.. warning::
   **Known limitation: superscripts are lost by the PDF text layer.** Article 51's systemic
   risk threshold of 10^25 FLOP extracts as ``'102 5'``. Nothing here can recover it -- the
   information is absent from the file. It does not raise an error and it reads like a
   plausible number. ``verify/citations.py`` cannot catch it either: a quote containing
   ``102 5`` verifies, because it genuinely appears in the ingested text. The verifier is
   right and the source is wrong. ``TestKnownTextLayerDamage`` pins it so a future fix is
   noticed; treat numeric thresholds read from a PDF as suspect until checked by hand.
"""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import count

import pdfplumber
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfparser import PDFParser

from euaia.ingest.document import ParsedDocument, ParsedUnit
from euaia.ingest.pdf_outline import OutlineEntry, read_outline

log = logging.getLogger(__name__)

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
# the real document currently matches every one. Not lenient either -- see `parse`.
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


def parse(pdf_bytes: bytes) -> ParsedDocument:
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
