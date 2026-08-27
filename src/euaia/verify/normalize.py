"""Text normalisation with an exact offset map back to the original.

Why this exists: citation verification asks "does this quote appear verbatim in the source?".
Formex-derived legal text is full of characters that make a naive answer *wrong*:

    'Article\\xa06'          non-breaking space instead of a space
    'the \\u2018provider\\u2019'   curly quotes where a model will emit straight ones
    'high\\u2011risk'             non-breaking hyphen
    'sub\\xadject'               soft hyphen left over from typesetting
    'point (a)\\n        and'    XML indentation inside a sentence

Matching on raw text rejects *correct* quotes because of these, which silently destroys the
coverage metric -- the system would abstain on good answers and we would blame the model.

So we match on a normalised copy, and keep an offset map so the quote we display and the
deep link we emit still point at the original characters.

Both sides of a comparison must go through this same function; the transformations are
deliberately consistent rather than strictly Unicode-canonical (NFKC is applied per
character to keep the offset map exact, which can differ from whole-string NFKC where
combining marks are involved -- irrelevant as long as both sides are treated identically).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# Dropped outright: they are typesetting artefacts with no textual meaning.
_DISCARD = {
    "­",  # soft hyphen
    "​",  # zero-width space
    "‌",  # zero-width non-joiner
    "‍",  # zero-width joiner
    "﻿",  # BOM / zero-width no-break space
}

# Folded to an ASCII equivalent so a model's straight quote matches the source's curly one.
_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "«": '"', "»": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
}


@dataclass(frozen=True, slots=True)
class Normalized:
    """Normalised text plus the mapping needed to get back to the original."""

    text: str
    """The normalised string. Compare against this."""

    offsets: tuple[int, ...]
    """``offsets[i]`` is the index in the original string that ``text[i]`` came from."""

    original: str
    """The untouched source string."""

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a ``[start, end)`` span in normalised space back to the original.

        The end bound is inclusive-of-the-last-character's source position plus one, so a
        slice of the original always covers at least the matched characters.
        """
        if not self.offsets:
            return (0, 0)
        if start >= len(self.offsets):
            return (len(self.original), len(self.original))
        first = self.offsets[start]
        last_idx = min(end, len(self.offsets)) - 1
        if last_idx < start:
            return (first, first)
        last = self.offsets[last_idx]
        return (first, last + 1)

    def original_slice(self, start: int, end: int) -> str:
        """The original text underlying a normalised span -- what we show the user."""
        a, b = self.original_span(start, end)
        return self.original[a:b]


def normalize(text: str) -> Normalized:
    """Fold away the differences that make correct quotes fail an exact match.

    Applies, in order: discard zero-width/soft-hyphen characters, fold quote, dash and
    exotic-space variants to ASCII, per-character NFKC, then collapse every run of
    whitespace to a single space and strip the ends.
    """
    chars: list[str] = []
    offsets: list[int] = []
    pending_space_src: int | None = None

    for i, ch in enumerate(text):
        if ch in _DISCARD:
            continue

        folded = _FOLD.get(ch)
        if folded is None:
            folded = unicodedata.normalize("NFKC", ch)
            if not folded:  # NFKC can erase a character entirely
                continue

        if folded.isspace():
            # Remember that whitespace occurred; emit at most one space, and only if
            # some non-space character follows. This drops leading/trailing space too.
            if pending_space_src is None:
                pending_space_src = i
            continue

        if pending_space_src is not None:
            if chars:  # never emit a leading space
                chars.append(" ")
                offsets.append(pending_space_src)
            pending_space_src = None

        for out_ch in folded:
            chars.append(out_ch)
            offsets.append(i)

    return Normalized(text="".join(chars), offsets=tuple(offsets), original=text)


def normalize_text(text: str) -> str:
    """Normalised string only, for callers that do not need the offset map."""
    return normalize(text).text
