"""Citation verification -- the part that makes the system trustworthy rather than plausible.

The answering model runs under a strict JSON schema, so every claim it emits must carry a
``supporting_quotes`` array or the response is rejected. That constrains the *shape*
(imperfectly -- Groq validates after generation as well as constraining during it, and does
return 400 for a response that slips through). It constrains nothing about the *content*: a
model can put invented words in the quote field and satisfy the schema completely. This
module closes that gap, and it is the part the trustworthiness claim actually rests on.

A quote is accepted only if it actually appears in the source unit it names. That check is
deterministic and adversarially sound in the direction that matters -- a fabricated sentence
cannot be found in text that does not contain it. There is no model in this loop and no
judgement call.

Three tiers, in order:

1. **exact** -- the normalised quote is a substring of the normalised source. Almost all
   good quotes land here.
2. **elided** -- the model marked an omission with an ellipsis; each segment either side
   must still match exactly, in order and without overlapping.
3. **rejected** -- everything else. The claim it supported is dropped and never shown.

**Why there is no fuzzy matching.** An earlier version of this module re-anchored near
misses by fuzzy alignment above a 92% similarity threshold. Testing showed that admits
exactly the failures it most needs to catch: against source reading *"shall be
prohibited"*, the quote *"shall **not** be prohibited"* scores ~96%, and *"**low**-risk
where **none** of the following conditions"* scores similarly against *"high-risk where
**both** of the following conditions"*. Both would have been accepted, silently
re-anchored onto the real text, and displayed to the user with a verified badge attached
to a claim the regulation contradicts. Lexical similarity and legal meaning are close to
uncorrelated across negation, so a near miss is never repaired by guessing. It is sent
back to the model to quote again, and dropped if it fails twice.

A fuzzy score is still *computed* for rejected quotes and recorded as a diagnostic -- a
high near-miss score is useful signal when tuning prompts, and carries no risk as long as
nothing is accepted on account of it.

Two further safeguards:

*Minimum quote length.* Without it, a model could "support" any claim with the word "the",
which appears in every provision. Short quotes are rejected outright.

*The source always wins.* Even for an exact match we display the original slice recovered
through the offset map, so what the user sees is the regulation's own characters -- curly
quotes, non-breaking spaces and all -- not a normalised approximation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from euaia.config import settings
from euaia.verify.normalize import normalize

log = logging.getLogger(__name__)

# A quote shorter than this cannot meaningfully support a claim; requiring both a character
# and a word floor stops both "the" and "shall be".
MIN_QUOTE_CHARS = 24
MIN_QUOTE_WORDS = 4

# Ways a model may signal that it omitted text from the middle of a quote.
ELLIPSIS_MARKERS = ("[...]", "[…]", "...", "…")


@dataclass(frozen=True, slots=True)
class Evidence:
    """One retrieved unit, as offered to the model and as checked against afterwards."""

    label: str
    """Short handle the model cites, e.g. 'E3'. Never a database id."""

    unit_id: int
    unit_path: str
    citation_label: str
    """Human citation, e.g. 'Article 6(2)'."""

    text: str
    """The unit's own text. Quotes are checked against *this*, not against the chunk text,
    so a model cannot pass off a breadcrumb we synthesised as source material."""

    deeplink: str | None = None
    document_version_id: int | None = None
    version_label: str | None = None


@dataclass(frozen=True, slots=True)
class VerifiedQuote:
    evidence_label: str
    unit_id: int
    unit_path: str
    citation_label: str
    quote: str
    """The original source text, recovered through the offset map."""

    method: str
    """'exact' or 'elided'."""

    score: float
    deeplink: str | None = None
    document_version_id: int | None = None


@dataclass(frozen=True, slots=True)
class RejectedQuote:
    evidence_label: str
    quote: str
    reason: str
    best_score: float = 0.0


@dataclass(slots=True)
class VerifiedClaim:
    text: str
    quotes: list[VerifiedQuote] = field(default_factory=list)

    @property
    def is_supported(self) -> bool:
        return bool(self.quotes)


@dataclass(slots=True)
class VerificationReport:
    """Outcome of checking one model answer."""

    claims: list[VerifiedClaim] = field(default_factory=list)
    """Claims that survived, in the model's original order."""

    dropped_claims: list[str] = field(default_factory=list)
    rejected: list[RejectedQuote] = field(default_factory=list)

    quotes_total: int = 0
    quotes_exact: int = 0
    quotes_elided: int = 0

    @property
    def quotes_dropped(self) -> int:
        return len(self.rejected)

    @property
    def claims_total(self) -> int:
        return len(self.claims) + len(self.dropped_claims)

    @property
    def coverage(self) -> float:
        """Fraction of claims that kept at least one verified quote."""
        if self.claims_total == 0:
            return 0.0
        return len(self.claims) / self.claims_total

    @property
    def quote_accuracy(self) -> float:
        """Share of the model's quotes that verified. The honest quality signal.

        Measured before any re-ask, so it reflects how reliably the model quotes source
        text rather than how well the retry loop papers over it.
        """
        if self.quotes_total == 0:
            return 0.0
        return (self.quotes_exact + self.quotes_elided) / self.quotes_total

    def unsupported_quote_texts(self) -> list[str]:
        """Rejected quotes, to name back to the model on its one retry."""
        return [r.quote for r in self.rejected]

    def near_misses(self, threshold: float = 85.0) -> list[RejectedQuote]:
        """Rejected quotes that were lexically close to the source.

        Diagnostic only. A high score here means the model paraphrased rather than
        invented -- a prompting problem, not a grounding failure. Nothing is ever accepted
        on the strength of this score.
        """
        return [r for r in self.rejected if r.best_score >= threshold]


def _trim_boundary_punctuation(text: str) -> str:
    """Drop leading/trailing punctuation, which never changes meaning at a quote boundary."""
    return text.strip(" \t\n\"'`.,;:-()[]")


def _split_elisions(needle: str) -> list[str]:
    """Split a quote on ellipsis markers into the segments that must each match."""
    for marker in ELLIPSIS_MARKERS:
        if marker in needle:
            parts = [_trim_boundary_punctuation(p) for p in needle.split(marker)]
            return [p for p in parts if p]
    return [needle]


def verify_quote(quote: str, evidence: Evidence) -> VerifiedQuote | RejectedQuote:
    """Check a single quote against the unit it claims to come from.

    Matching is exact (after normalisation), optionally across an explicit ellipsis. There
    is deliberately no similarity threshold -- see the module docstring.
    """
    needle = _trim_boundary_punctuation(normalize(quote).text)

    if len(needle) < MIN_QUOTE_CHARS or len(needle.split()) < MIN_QUOTE_WORDS:
        return RejectedQuote(
            evidence_label=evidence.label,
            quote=quote,
            reason=(
                f"quote too short to support a claim "
                f"({len(needle)} chars, {len(needle.split())} words)"
            ),
        )

    haystack = normalize(evidence.text)
    segments = _split_elisions(needle)

    # Walk the segments in order, each starting after the previous one ended, so an
    # elided quote cannot be satisfied by text appearing in the wrong sequence.
    start: int | None = None
    cursor = 0
    for segment in segments:
        found = haystack.text.find(segment, cursor)
        if found == -1:
            return _reject(quote, evidence, needle, haystack.text)
        if start is None:
            start = found
        cursor = found + len(segment)

    if start is None:
        return _reject(quote, evidence, needle, haystack.text)

    return VerifiedQuote(
        evidence_label=evidence.label,
        unit_id=evidence.unit_id,
        unit_path=evidence.unit_path,
        citation_label=evidence.citation_label,
        # The source span wins: we display the regulation's characters, never the model's.
        quote=haystack.original_slice(start, cursor),
        method="exact" if len(segments) == 1 else "elided",
        score=100.0,
        deeplink=evidence.deeplink,
        document_version_id=evidence.document_version_id,
    )


def _reject(quote: str, evidence: Evidence, needle: str, haystack: str) -> RejectedQuote:
    """Reject, recording the best fuzzy score purely as a diagnostic.

    The score never gates acceptance. It is here so the eval suite can distinguish "the
    model paraphrased" (high score) from "the model invented a provision" (low score),
    which are different prompting problems.
    """
    alignment = fuzz.partial_ratio_alignment(needle, haystack)
    return RejectedQuote(
        evidence_label=evidence.label,
        quote=quote,
        reason="quote does not appear verbatim in the cited source",
        best_score=float(alignment.score) if alignment is not None else 0.0,
    )


def verify_answer(claims: list[dict], evidence: list[Evidence]) -> VerificationReport:
    """Verify every quote in a model answer and report what survived.

    ``claims`` is the ``claims`` array from the model's structured output: each entry has
    ``text`` and ``supporting_quotes`` (a list of ``{evidence_label, quote}``).
    """
    by_label = {e.label: e for e in evidence}
    report = VerificationReport()

    for claim in claims:
        text = (claim.get("text") or "").strip()
        if not text:
            continue

        verified: list[VerifiedQuote] = []
        for raw in claim.get("supporting_quotes") or []:
            label = (raw.get("evidence_label") or "").strip()
            quote = (raw.get("quote") or "").strip()
            if not quote:
                continue

            report.quotes_total += 1
            source = by_label.get(label)
            if source is None:
                # The model cited a label we never supplied -- a hallucinated source.
                report.rejected.append(
                    RejectedQuote(
                        evidence_label=label,
                        quote=quote,
                        reason=f"unknown evidence label {label!r}",
                    )
                )
                continue

            outcome = verify_quote(quote, source)
            if isinstance(outcome, VerifiedQuote):
                verified.append(outcome)
                if outcome.method == "exact":
                    report.quotes_exact += 1
                else:
                    report.quotes_elided += 1
            else:
                report.rejected.append(outcome)

        if verified:
            report.claims.append(VerifiedClaim(text=text, quotes=verified))
        else:
            log.info("Dropping unsupported claim: %s", text[:120])
            report.dropped_claims.append(text)

    return report


def decide_verdict(report: VerificationReport) -> str:
    """Map coverage onto the answer / partial / abstain decision.

    Thresholds live in settings so they can be tuned against the eval suite rather than
    guessed.
    """
    if not report.claims:
        return "abstained"
    coverage = report.coverage
    if coverage >= settings.coverage_answer_threshold:
        return "answered"
    if coverage >= settings.coverage_partial_threshold:
        return "partial"
    return "abstained"
