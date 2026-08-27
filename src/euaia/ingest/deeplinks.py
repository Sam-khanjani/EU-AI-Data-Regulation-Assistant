"""Build EUR-Lex deep links for citable units.

The anchor scheme was read off the real rendered document rather than guessed:
``art_6``, ``art_4a`` (inserted articles get anchors too), ``anx_III``, ``rct_27``. The
consolidated act contains exactly 119 ``art_`` anchors, matching the 119 articles the
Formex parser finds -- so a link built from ``unit_number`` resolves for every article.

There is no paragraph-level anchor in EUR-Lex's rendering, so a paragraph links to its
parent article; the exact paragraph text is shown in our own UI alongside the link.

Note that eur-lex.europa.eu applies bot mitigation (HTTP 202 with an empty body) to
non-browser clients. The links are for humans to click, and work in a browser; automated
retrieval goes through CELLAR instead.
"""

from __future__ import annotations

from urllib.parse import quote

EURLEX_BASE = "https://eur-lex.europa.eu/legal-content/EN/TXT/"


def anchor_for(unit_type: str, unit_number: str | None, unit_path: str = "") -> str | None:
    """Anchor fragment for a unit, or ``None`` when the unit has no addressable anchor."""
    if unit_type == "article" and unit_number:
        return f"art_{unit_number}"
    if unit_type == "annex" and unit_number:
        return f"anx_{unit_number}"
    if unit_type == "recital" and unit_number:
        return f"rct_{unit_number}"
    if unit_type == "paragraph":
        # No paragraph anchors exist; fall back to the parent article.
        article = _article_from_path(unit_path)
        return f"art_{article}" if article else None
    if unit_type == "chapter" and unit_number:
        return f"cpt_{unit_number}"
    return None


def _article_from_path(unit_path: str) -> str | None:
    """``CH_III/SEC_1/ART_6/PAR_2`` -> ``6``."""
    for segment in unit_path.split("/"):
        if segment.startswith("ART_"):
            return segment[4:]
    return None


def build(celex: str, unit_type: str, unit_number: str | None, unit_path: str = "") -> str:
    """Full EUR-Lex URL for a unit, anchored where possible."""
    url = f"{EURLEX_BASE}?uri=CELEX%3A{quote(celex)}"
    fragment = anchor_for(unit_type, unit_number, unit_path)
    return f"{url}#{fragment}" if fragment else url
