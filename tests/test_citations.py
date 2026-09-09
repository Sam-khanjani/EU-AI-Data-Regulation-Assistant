"""Citation verification tests.

These are the tests that back the project's central claim, so they are written
adversarially: the interesting cases are the ones where a model tries to pass off text the
regulation does not contain.

Source text below is the real Article 6(1) and Article 5(1)(a) wording, including the
non-breaking hyphen and line wrapping that appear in the source text.
"""

from __future__ import annotations

import pytest

from euaia.verify.citations import (
    Evidence,
    RejectedQuote,
    VerifiedQuote,
    decide_verdict,
    verify_answer,
    verify_quote,
)

ART_6_TEXT = (
    "Article 6\n"
    "Classification rules for high-risk AI systems\n"
    "1.\n"
    "Irrespective of whether an AI system is placed on the market or put into service\n"
    "        independently of the products referred to in points (a) and (b), that AI system\n"
    "        shall be considered to be high‑risk where both of the following conditions are\n"
    "        fulfilled:\n"
    "(a)\n"
    "the AI system is intended to be used as a safety component of a product, or the AI\n"
    "        system is itself a product, covered by the Union harmonisation legislation listed\n"
    "        in Annex I;"
)

ART_5_TEXT = (
    "Article 5\n"
    "Prohibited AI practices\n"
    "1.\n"
    "The following AI practices shall be prohibited:\n"
    "(a)\n"
    "the placing on the market, the putting into service or the use of an AI system that\n"
    "        deploys subliminal techniques beyond a person’s consciousness or purposefully\n"
    "        manipulative or deceptive techniques;"
)


def ev(label: str, text: str, unit_id: int = 1, citation: str = "Article 6") -> Evidence:
    return Evidence(
        label=label,
        unit_id=unit_id,
        unit_path="CH_III/SEC_1/ART_6",
        citation_label=citation,
        text=text,
        deeplink="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A02024R1689-20260727#art_6",
        document_version_id=1,
    )


class TestExactVerification:
    def test_verbatim_quote_verifies(self):
        result = verify_quote(
            "The following AI practices shall be prohibited:", ev("E1", ART_5_TEXT)
        )
        assert isinstance(result, VerifiedQuote)
        assert result.method == "exact"
        assert result.score == 100.0

    def test_quote_across_xml_line_breaks_verifies(self):
        # The source wraps this across three indented lines; a model emits one line.
        quote = (
            "that AI system shall be considered to be high-risk where both of the "
            "following conditions are fulfilled:"
        )
        result = verify_quote(quote, ev("E1", ART_6_TEXT))
        assert isinstance(result, VerifiedQuote)
        assert result.method == "exact"

    def test_straight_hyphen_matches_non_breaking_hyphen(self):
        result = verify_quote(
            "shall be considered to be high-risk where both", ev("E1", ART_6_TEXT)
        )
        assert isinstance(result, VerifiedQuote)

    def test_straight_apostrophe_matches_curly(self):
        result = verify_quote(
            "subliminal techniques beyond a person's consciousness", ev("E1", ART_5_TEXT)
        )
        assert isinstance(result, VerifiedQuote)

    def test_displayed_quote_is_the_source_text_not_the_model_wording(self):
        # Model sends a straight apostrophe; the regulation uses a curly one. We must show
        # the regulation's characters.
        result = verify_quote(
            "subliminal techniques beyond a person's consciousness", ev("E1", ART_5_TEXT)
        )
        assert isinstance(result, VerifiedQuote)
        assert "’" in result.quote, "should display the source's curly apostrophe"


class TestFabricationIsRejected:
    def test_invented_sentence_is_rejected(self):
        result = verify_quote(
            "AI systems used for recruitment are exempt from all requirements under this Regulation.",
            ev("E1", ART_6_TEXT),
        )
        assert isinstance(result, RejectedQuote)
        assert "does not appear" in result.reason

    def test_plausible_but_absent_legal_phrasing_is_rejected(self):
        # Reads like the Act, is not in this provision. The dangerous failure mode.
        result = verify_quote(
            "that AI system shall be considered to be low-risk where none of the following "
            "conditions are fulfilled:",
            ev("E1", ART_6_TEXT),
        )
        assert isinstance(result, RejectedQuote)

    def test_negated_quote_is_rejected(self):
        result = verify_quote(
            "The following AI practices shall not be prohibited:", ev("E1", ART_5_TEXT)
        )
        assert isinstance(result, RejectedQuote)

    def test_quote_from_a_different_provision_is_rejected(self):
        # Real Article 5 text, but cited against Article 6.
        result = verify_quote(
            "The following AI practices shall be prohibited:", ev("E1", ART_6_TEXT)
        )
        assert isinstance(result, RejectedQuote)


class TestMinimumLength:
    @pytest.mark.parametrize("quote", ["the", "shall be", "AI system", "prohibited:"])
    def test_short_quotes_cannot_support_a_claim(self, quote):
        # Without this floor, 'the' would "verify" against every provision in the corpus.
        result = verify_quote(quote, ev("E1", ART_5_TEXT))
        assert isinstance(result, RejectedQuote)
        assert "too short" in result.reason


class TestNoFuzzyAcceptance:
    """Regression guard for the defect that motivated removing fuzzy matching.

    A 92%-similarity threshold accepted all three of these. Each one, if accepted, would
    have attached a 'verified' badge to a statement the regulation contradicts.
    """

    @pytest.mark.parametrize(
        ("quote", "why"),
        [
            (
                "The following AI practices shall not be prohibited:",
                "inserted negation",
            ),
            (
                "that AI system shall be considered to be low-risk where none of the "
                "following conditions are fulfilled:",
                "inverted risk tier and quantifier",
            ),
            (
                "Irrespective of whether an AI system is placed on market or put into service "
                "independently of the products referred to in points (a) and (b)",
                "dropped word - a misquote of a legal provision is still a misquote",
            ),
        ],
    )
    def test_near_misses_are_rejected_not_repaired(self, quote, why):
        source = ART_5_TEXT if "prohibited" in quote else ART_6_TEXT
        result = verify_quote(quote, ev("E1", source))
        assert isinstance(result, RejectedQuote), f"must reject: {why}"

    def test_near_miss_is_still_reported_as_a_diagnostic(self):
        # We record how close it was, to tell paraphrase apart from invention -- but the
        # score never gates acceptance.
        result = verify_quote(
            "The following AI practices shall not be prohibited:", ev("E1", ART_5_TEXT)
        )
        assert isinstance(result, RejectedQuote)
        assert result.best_score > 85.0, "a negation is lexically near-identical"


class TestElision:
    def test_explicit_ellipsis_matches_both_segments(self):
        quote = (
            "Irrespective of whether an AI system is placed on the market [...] "
            "that AI system shall be considered to be high-risk"
        )
        result = verify_quote(quote, ev("E1", ART_6_TEXT))
        assert isinstance(result, VerifiedQuote)
        assert result.method == "elided"

    def test_elided_segments_must_appear_in_order(self):
        # Same two segments, reversed. Must not verify.
        quote = (
            "that AI system shall be considered to be high-risk [...] "
            "Irrespective of whether an AI system is placed on the market"
        )
        assert isinstance(verify_quote(quote, ev("E1", ART_6_TEXT)), RejectedQuote)

    def test_elision_cannot_smuggle_in_absent_text(self):
        quote = "Irrespective of whether an AI system is placed on the market [...] is exempt from this Regulation"
        assert isinstance(verify_quote(quote, ev("E1", ART_6_TEXT)), RejectedQuote)


class TestBoundaryPunctuation:
    def test_trailing_period_the_source_lacks_is_tolerated(self):
        result = verify_quote(
            "The following AI practices shall be prohibited.", ev("E1", ART_5_TEXT)
        )
        assert isinstance(result, VerifiedQuote)

    def test_surrounding_quote_marks_are_tolerated(self):
        result = verify_quote(
            '"The following AI practices shall be prohibited:"', ev("E1", ART_5_TEXT)
        )
        assert isinstance(result, VerifiedQuote)


class TestVerifyAnswer:
    def build(self):
        return [ev("E1", ART_6_TEXT, unit_id=1), ev("E2", ART_5_TEXT, unit_id=2, citation="Article 5")]

    def test_fully_supported_answer(self):
        claims = [
            {
                "text": "An AI system is high-risk when both conditions in Article 6(1) are met.",
                "supporting_quotes": [
                    {
                        "evidence_label": "E1",
                        "quote": "shall be considered to be high-risk where both of the following conditions are fulfilled:",
                    }
                ],
            }
        ]
        report = verify_answer(claims, self.build())
        assert report.coverage == 1.0
        assert report.quotes_exact == 1
        assert report.quotes_dropped == 0
        assert decide_verdict(report) == "answered"

    def test_unsupported_claim_is_dropped_not_shown(self):
        claims = [
            {
                "text": "Supported claim.",
                "supporting_quotes": [
                    {"evidence_label": "E2", "quote": "The following AI practices shall be prohibited:"}
                ],
            },
            {
                "text": "Recruitment AI is exempt from the Regulation.",
                "supporting_quotes": [
                    {"evidence_label": "E1", "quote": "recruitment systems are wholly exempt from this Regulation"}
                ],
            },
        ]
        report = verify_answer(claims, self.build())
        assert [c.text for c in report.claims] == ["Supported claim."]
        assert report.dropped_claims == ["Recruitment AI is exempt from the Regulation."]
        assert report.coverage == 0.5

    def test_citing_an_evidence_label_we_never_supplied_is_rejected(self):
        claims = [
            {
                "text": "Invented source.",
                "supporting_quotes": [
                    {"evidence_label": "E99", "quote": "some text that was never provided to the model"}
                ],
            }
        ]
        report = verify_answer(claims, self.build())
        assert report.claims == []
        assert "unknown evidence label" in report.rejected[0].reason
        assert decide_verdict(report) == "abstained"

    def test_claim_with_no_quotes_at_all_is_dropped(self):
        claims = [{"text": "Bare assertion.", "supporting_quotes": []}]
        report = verify_answer(claims, self.build())
        assert report.claims == []
        assert report.dropped_claims == ["Bare assertion."]

    def test_metrics_are_reported(self):
        claims = [
            {
                "text": "One verifies, one is a misquote.",
                "supporting_quotes": [
                    {"evidence_label": "E2", "quote": "The following AI practices shall be prohibited:"},
                    {
                        "evidence_label": "E1",
                        "quote": "Irrespective of whether an AI system is placed on market or put into service independently of the products",
                    },
                ],
            }
        ]
        report = verify_answer(claims, self.build())
        assert report.quotes_total == 2
        assert report.quotes_exact == 1
        assert report.quotes_dropped == 1
        assert report.quote_accuracy == 0.5

    def test_rejected_quotes_are_available_for_the_repair_round_trip(self):
        claims = [
            {
                "text": "Bad.",
                "supporting_quotes": [
                    {"evidence_label": "E1", "quote": "this sentence is nowhere in the regulation at all"}
                ],
            }
        ]
        report = verify_answer(claims, self.build())
        assert report.unsupported_quote_texts() == [
            "this sentence is nowhere in the regulation at all"
        ]


class TestVerdict:
    def test_empty_report_abstains(self):
        from euaia.verify.citations import VerificationReport

        assert decide_verdict(VerificationReport()) == "abstained"
