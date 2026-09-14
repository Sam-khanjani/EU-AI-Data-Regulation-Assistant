"""Read the recitals out of an as-adopted EU act's PDF.

Recitals are the numbered ``(1) (2) (3)`` paragraphs of the preamble -- the reasoning behind
the rules. They are not binding, but courts read them to establish what an article means, so
an assistant that can only quote articles cannot answer "why".

They need their own reader, separate from :mod:`euaia.ingest.pdf_parser`, because the
document they live in is published differently. The consolidated act carries a 165-entry
bookmark outline naming every article; the as-adopted act carries **14** bookmarks, all
annexes. There is no outline to read recitals from, so structure has to come from the text.

Three signals do it, and the order matters:

1. **The region.** Recitals sit between the preamble and the enacting formula
   ``HAVE ADOPTED THIS REGULATION``. That phrase is part of the law, not the layout.
2. **The size.** A recital's number is set on its own line at 9pt while the body runs at
   10pt; footnote prose is also 9pt. Keeping body-size lines therefore drops footnote text
   wholesale, which matters because footnotes are numbered ``(1) (2) (3)`` too and would
   otherwise be indistinguishable from recitals.
3. **The sequence.** Recitals run 1..N with no gaps. Any ``(N)`` that does not continue the
   run is not the next recital.

**Why 3 is not optional, and why 2 alone is not enough.** An earlier attempt used the
sequence rule by itself and returned exactly 180 recitals numbered 1..180 -- the right count,
and wrong: it had taken a footnote reading *"(4) Position of the European Parliament of 13
March 2024"* as recital 4, and rejected the real one on the next page. A correct total is not
evidence of correct content.

That pairing is also what makes a size-based rule safe to rely on. If EUR-Lex changes the
typography, the size filter stops matching, the run fails to reach its expected length, and
:func:`parse_recitals` raises rather than returning a plausible-looking fragment of the
preamble. The size is a filter; the law's own numbering is the proof.
"""

from __future__ import annotations

import io
import logging
import re
from itertools import count

import pdfplumber

from euaia.ingest.document import ParsedDocument, ParsedUnit

# _Line and _join describe PDF text generally rather than anything about the outline; they
# live in pdf_parser because that module needed them first.
from euaia.ingest.pdf_parser import PdfParseError, _join, _Line

log = logging.getLogger(__name__)

# Point sizes in the as-adopted PDF. The recital *number* is set smaller than the prose it
# introduces, and footnotes are set at the number's size -- which is what lets one rule
# separate body text from footnote text.
MARKER_SIZE = 8.5
BODY_SIZE = 9.6

# The enacting formula. Everything after it is operative text, not preamble.
ENACTING_FORMULA = "HAVE ADOPTED THIS REGULATION"

# A recital number. It hangs at the start of the first body line rather than sitting on a
# line of its own, so this matches the leading token, not the whole line.
_MARKER = re.compile(r"^\((\d{1,3})\)$")

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


def _lines(page: pdfplumber.page.Page) -> list[list[dict]]:
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


def _classify(words: list[dict]) -> tuple[str | None, str]:
    """Split a line into (recital number if it starts one, body text).

    The size pattern is the whole trick. A recital's first line reads
    ``[8.5, 9.6, 9.6, ...]`` -- a small hanging number followed by prose -- while a footnote
    reads ``[8.5, 8.5, 8.5, ...]`` all the way across. Same ``(N)`` token, different line.
    """
    sizes = [round(w["size"], 1) for w in words]

    marker = _MARKER.match(words[0]["text"])
    if marker and sizes[0] == MARKER_SIZE and BODY_SIZE in sizes[1:]:
        rest = " ".join(w["text"] for w in words[1:] if round(w["size"], 1) == BODY_SIZE)
        return marker.group(1), rest.strip()

    # Continuation of a recital. Footnote lines carry no body-size words and fall through.
    if BODY_SIZE in sizes:
        return None, " ".join(
            w["text"] for w in words if round(w["size"], 1) == BODY_SIZE
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
            for words in _lines(page):
                text = " ".join(w["text"] for w in words)
                if ENACTING_FORMULA in text:
                    found_end = True
                    break
                marker, prose = _classify(words)
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
            f"the {MARKER_SIZE}pt/{BODY_SIZE}pt typography this reader depends on has "
            "probably changed"
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
