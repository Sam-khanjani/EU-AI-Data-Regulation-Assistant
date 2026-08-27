"""Parse Formex XML into a tree of citable legal units.

This is deliberately *not* a generic XML-to-text loader. Generic loaders flatten the
document to prose, which destroys the one thing citation depends on: knowing that a given
sentence lives in Article 6(2) and not somewhere else. Everything here exists to preserve
official numbering.

Two Formex root shapes are handled, because the AI Act needs both:

``ACT``       the original as-adopted act. Carries the 180 recitals (``CONSID``), which
              consolidation drops. Articles here are the *unamended* text.
``CONS.ACT``  a consolidated act. Carries the operative text *with* amendments applied
              (Article 6 gains paragraphs 1a and 1b, for instance) and the annexes as
              ``CONS.ANNEX``, but no recitals.

Text extraction quirks this handles, all observed in the real document:

* ``xpath("string()")`` concatenates without separators, yielding
  ``'Article 6Classification rules...'``. We serialise block elements with line breaks.
* ``<?PAGE NO='3'?>`` processing instructions appear mid-sentence and must be skipped.
* Article titles mix ordinary and non-breaking spaces (``'Article\\xa04a'``), so numbers
  are extracted from normalised text.
* ``<NOTE>`` footnotes are excluded from citable body text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from lxml import etree

from euaia.verify.normalize import normalize_text

log = logging.getLogger(__name__)

# Elements that start and end a line when serialised to text.
_BLOCK_TAGS = frozenset(
    {
        "P", "ALINEA", "TXT", "NP", "ITEM", "PARAG", "LIST",
        "TI.ART", "STI.ART", "TI", "STI", "TITLE", "CONSID",
        "ARTICLE", "DIVISION", "CELL", "ROW", "TBL", "CONTENTS",
        "CONS.ANNEX", "ANNEX", "GR.SEQ",
    }
)

# Excluded from citable text: footnotes and their markers are not part of the provision.
_SKIP_TAGS = frozenset({"NOTE"})

_ARTICLE_NO = re.compile(r"Article\s+(\S+)", re.IGNORECASE)
_ANNEX_NO = re.compile(r"ANNEX\s+([IVXLC]+|\d+)", re.IGNORECASE)
_CHAPTER_NO = re.compile(r"CHAPTER\s+([IVXLC]+|\d+)", re.IGNORECASE)
_SECTION_NO = re.compile(r"SECTION\s+([IVXLC]+|\d+)", re.IGNORECASE)
_RECITAL_NO = re.compile(r"^\(?(\d+)\)?$")


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


def serialize_text(el: etree._Element) -> str:
    """Serialise an element to readable text, preserving block structure.

    Line breaks between block elements are what keep '1.' separate from the paragraph it
    numbers, and what stop an article title from fusing into its first sentence.
    """
    parts: list[str] = []

    def walk(node: etree._Element) -> None:
        tag = node.tag
        if not isinstance(tag, str):
            return  # comment or processing instruction, e.g. <?PAGE NO='3'?>
        if tag in _SKIP_TAGS:
            return

        is_block = tag in _BLOCK_TAGS
        if is_block and parts and not parts[-1].endswith("\n"):
            parts.append("\n")
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        if is_block:
            parts.append("\n")

    walk(el)
    lines = (line.strip() for line in "".join(parts).split("\n"))
    return "\n".join(line for line in lines if line)


def _child_text(el: etree._Element, tag: str) -> str:
    """Normalised single-line text of a child element, whatever its inner shape.

    Formex is inconsistent about wrapping: the consolidated act writes
    ``<STI.ART>Subject matter</STI.ART>`` while the original act writes
    ``<STI.ART><P>Prohibited AI practices</P></STI.ART>``. ``findtext`` returns only the
    direct text node, so it silently yields '' for the wrapped form -- which is how every
    article heading in the original act came back empty. Serialising the subtree handles
    both.
    """
    child = el.find(tag)
    if child is None:
        return ""
    return normalize_text(serialize_text(child).replace("\n", " "))


def parse(xml: bytes) -> ParsedDocument:
    """Parse a Formex document into a flat, ordered list of citable units."""
    root = etree.fromstring(xml)
    root_tag = str(root.tag)

    units: list[ParsedUnit] = []
    counter = _Counter()

    consolidation_date = start_date = None
    info = root.find("INFO.CONSLEG")
    if info is not None:
        consolidation_date = info.get("CONSLEG.DATE")
        start_date = info.get("START.DATE")

    _parse_recitals(root, units, counter)
    _parse_divisions_and_articles(root, units, counter)
    _parse_annexes(root, units, counter)

    if not units:
        raise ValueError(f"Formex root <{root_tag}> yielded no structural units")

    log.info(
        "Parsed <%s>: %d units (%d articles, %d recitals, %d annexes)",
        root_tag,
        len(units),
        sum(1 for u in units if u.unit_type == "article"),
        sum(1 for u in units if u.unit_type == "recital"),
        sum(1 for u in units if u.unit_type == "annex"),
    )
    return ParsedDocument(
        units=units,
        root_tag=root_tag,
        consolidation_date=consolidation_date,
        start_date=start_date,
    )


class _Counter:
    """Monotonic document-order ordinal."""

    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


def _parse_recitals(root: etree._Element, units: list[ParsedUnit], counter: _Counter) -> None:
    """Recitals live in ``GR.CONSID`` as ``CONSID`` elements. Present only in ``ACT``."""
    for consid in root.iter("CONSID"):
        number = _recital_number(consid)
        text = serialize_text(consid)
        if not text:
            continue
        path = f"RCT_{number}" if number else f"RCT_{counter.next()}"
        units.append(
            ParsedUnit(
                unit_type="recital",
                unit_number=number,
                unit_path=path,
                heading=None,
                text=text,
                ordinal=counter.next(),
            )
        )


def _recital_number(consid: etree._Element) -> str | None:
    no_p = consid.find("NP/NO.P")
    if no_p is None:
        no_p = consid.find(".//NO.P")
    if no_p is None:
        return None
    raw = normalize_text(serialize_text(no_p).replace("\n", " "))
    match = _RECITAL_NO.match(raw)
    return match.group(1) if match else None


def _parse_divisions_and_articles(
    root: etree._Element, units: list[ParsedUnit], counter: _Counter
) -> None:
    """Walk DIVISION nesting to build chapter/section context, then articles within it.

    Articles sit either directly under a chapter DIVISION or one level deeper under a
    section DIVISION -- both shapes occur in the AI Act.
    """
    enacting = root.find(".//ENACTING.TERMS")
    if enacting is None:
        enacting = root

    def walk(node: etree._Element, ancestors: list[ParsedUnit]) -> None:
        for child in node:
            if not isinstance(child.tag, str):
                continue
            if child.tag == "DIVISION":
                division = _make_division(child, ancestors, counter)
                if division is not None:
                    units.append(division)
                    walk(child, [*ancestors, division])
                else:
                    walk(child, ancestors)
            elif child.tag == "ARTICLE":
                article = _make_article(child, ancestors, counter)
                if article is not None:
                    units.append(article)
                    units.extend(_make_paragraphs(child, article, counter))
            else:
                walk(child, ancestors)

    walk(enacting, [])


def _division_title(div: etree._Element) -> str:
    title = div.find("TITLE")
    return serialize_text(title).replace("\n", " ").strip() if title is not None else ""


def _make_division(
    div: etree._Element, ancestors: list[ParsedUnit], counter: _Counter
) -> ParsedUnit | None:
    title = _division_title(div)
    if not title:
        return None
    norm = normalize_text(title)

    if match := _CHAPTER_NO.search(norm):
        unit_type, number = "chapter", match.group(1)
        prefix = f"CH_{number}"
    elif match := _SECTION_NO.search(norm):
        unit_type, number = "section", match.group(1)
        prefix = f"SEC_{number}"
    else:
        return None

    parent = ancestors[-1] if ancestors else None
    path = f"{parent.unit_path}/{prefix}" if parent else prefix
    heading = _strip_leading_label(norm, unit_type, number)
    return ParsedUnit(
        unit_type=unit_type,
        unit_number=number,
        unit_path=path,
        heading=heading or None,
        # A chapter's own text is just its title; article text lives on the articles.
        text=title,
        ordinal=counter.next(),
        parent_path=parent.unit_path if parent else None,
    )


def _strip_leading_label(norm_title: str, unit_type: str, number: str) -> str:
    label = f"{unit_type.upper()} {number}"
    stripped = norm_title[len(label) :] if norm_title.upper().startswith(label) else norm_title
    return stripped.strip(" .-—:")


def _make_article(
    art: etree._Element, ancestors: list[ParsedUnit], counter: _Counter
) -> ParsedUnit | None:
    title = _child_text(art, "TI.ART")
    # Trailing apostrophes occur in the consolidated text, e.g. "Subject matter'".
    heading = _child_text(art, "STI.ART").strip().rstrip("'")
    match = _ARTICLE_NO.search(title)
    # IDENTIFIER ("004A") is the reliable fallback when the title is malformed.
    number = match.group(1) if match else (art.get("IDENTIFIER") or "").lstrip("0") or None

    text = serialize_text(art)
    if not text:
        return None

    parent = ancestors[-1] if ancestors else None
    path = f"{parent.unit_path}/ART_{number}" if parent else f"ART_{number}"
    return ParsedUnit(
        unit_type="article",
        unit_number=number,
        unit_path=path,
        heading=heading or None,
        text=text,
        ordinal=counter.next(),
        parent_path=parent.unit_path if parent else None,
    )


def _make_paragraphs(
    art: etree._Element, article: ParsedUnit, counter: _Counter
) -> list[ParsedUnit]:
    """Numbered paragraphs of an article -- the granularity we embed at.

    Articles with no PARAG children (short articles are a single ALINEA) yield nothing;
    the article-level unit already carries their full text.
    """
    out: list[ParsedUnit] = []
    for parag in art.findall("PARAG"):
        number = _child_text(parag, "NO.PARAG").strip().rstrip(".")
        text = serialize_text(parag)
        if not text:
            continue
        suffix = number or str(len(out) + 1)
        out.append(
            ParsedUnit(
                unit_type="paragraph",
                unit_number=f"{article.unit_number}({suffix})" if article.unit_number else suffix,
                unit_path=f"{article.unit_path}/PAR_{suffix}",
                heading=None,
                text=text,
                ordinal=counter.next(),
                parent_path=article.unit_path,
            )
        )
    return out


def _parse_annexes(root: etree._Element, units: list[ParsedUnit], counter: _Counter) -> None:
    """Annexes: ``CONS.ANNEX`` in consolidated acts, ``ANNEX`` in original ones."""
    for tag in ("CONS.ANNEX", "ANNEX"):
        for annex in root.iter(tag):
            title_el = annex.find("TITLE")
            title = (
                serialize_text(title_el).replace("\n", " ").strip()
                if title_el is not None
                else ""
            )
            norm = normalize_text(title)
            match = _ANNEX_NO.search(norm)
            number = match.group(1) if match else None

            text = serialize_text(annex)
            if not text:
                continue
            path = f"ANX_{number}" if number else f"ANX_{counter.next()}"
            units.append(
                ParsedUnit(
                    unit_type="annex",
                    unit_number=number,
                    unit_path=path,
                    heading=_strip_leading_label(norm, "annex", number or "") or None,
                    text=text,
                    ordinal=counter.next(),
                )
            )
