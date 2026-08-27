"""Normalisation must fold real Formex artefacts without losing the way back.

Every case here is taken from, or modelled on, text actually present in the AI Act Formex:
non-breaking spaces in article titles ('Article\\xa04a'), curly quotes, non-breaking
hyphens in 'high-risk', and XML indentation inside sentences.
"""

from __future__ import annotations

import pytest

from euaia.verify.normalize import normalize, normalize_text


class TestFolding:
    def test_non_breaking_space_becomes_space(self):
        # Straight from the consolidated act: TI.ART of Article 4a.
        assert normalize_text("Article\xa04a") == "Article 4a"

    def test_curly_quotes_fold_to_straight(self):
        assert normalize_text("the ‘provider’ shall") == "the 'provider' shall"
        assert normalize_text("“AI system” means") == '"AI system" means'

    def test_dash_variants_fold_to_hyphen(self):
        for dash in ("‐", "‑", "‒", "–", "—", "―"):
            assert normalize_text(f"high{dash}risk") == "high-risk"

    def test_soft_hyphen_is_dropped(self):
        assert normalize_text("sub­ject") == "subject"

    def test_zero_width_characters_dropped(self):
        assert normalize_text("AI​system﻿") == "AIsystem"

    def test_whitespace_runs_collapse(self):
        assert normalize_text("point (a)\n        and point (b)") == "point (a) and point (b)"

    def test_leading_and_trailing_whitespace_stripped(self):
        assert normalize_text("\n   Article 6   \n") == "Article 6"

    def test_ellipsis_expands(self):
        assert normalize_text("shall…") == "shall..."

    def test_empty_and_whitespace_only(self):
        assert normalize_text("") == ""
        assert normalize_text("   \n\t  ") == ""


class TestOffsetMap:
    def test_offsets_point_at_original_characters(self):
        n = normalize("Article\xa06")
        assert n.text == "Article 6"
        # Every normalised char must map to a real index in the original.
        assert all(0 <= o < len(n.original) for o in n.offsets)
        assert len(n.offsets) == len(n.text)

    def test_round_trip_recovers_original_substring(self):
        raw = "the ‘provider’ shall ensure"
        n = normalize(raw)
        idx = n.text.index("'provider'")
        recovered = n.original_slice(idx, idx + len("'provider'"))
        # The original had curly quotes; we get them back, not the folded form.
        assert recovered == "‘provider’"

    def test_round_trip_across_collapsed_whitespace(self):
        raw = "point (a)\n        and point (b)"
        n = normalize(raw)
        idx = n.text.index("and")
        assert n.original_slice(idx, idx + 3) == "and"

    def test_round_trip_across_nbsp(self):
        raw = "see Article\xa06 of this Regulation"
        n = normalize(raw)
        idx = n.text.index("Article 6")
        assert n.original_slice(idx, idx + len("Article 6")) == "Article\xa06"

    def test_span_of_whole_text(self):
        raw = "  High-risk AI systems  "
        n = normalize(raw)
        assert n.original_slice(0, len(n.text)) == "High-risk AI systems"

    @pytest.mark.parametrize(
        "raw",
        [
            "Article\xa06",
            "the ‘provider’",
            "high‑risk",
            "a\n   b\n   c",
            "sub­ject to Article 5",
            "",
        ],
    )
    def test_offsets_are_monotonic(self, raw):
        # A non-decreasing map is what makes original_span meaningful.
        n = normalize(raw)
        assert list(n.offsets) == sorted(n.offsets)


class TestIdempotence:
    @pytest.mark.parametrize(
        "raw",
        [
            "Article\xa06",
            "the ‘provider’ shall",
            "point (a)\n   and (b)",
            "high‑risk AI",
        ],
    )
    def test_normalising_twice_changes_nothing(self, raw):
        once = normalize_text(raw)
        assert normalize_text(once) == once


class TestMatchingBehaviour:
    """The property the verifier actually depends on."""

    def test_model_style_quote_matches_formex_source(self):
        # Source as it appears in Formex, quote as a model would plausibly emit it.
        source = (
            "that AI system shall be considered to be high‑risk where both of the\n"
            "        following conditions are fulfilled:"
        )
        quote = "that AI system shall be considered to be high-risk where both of the following conditions are fulfilled:"
        assert normalize_text(quote) in normalize_text(source)

    def test_unrelated_quote_does_not_match(self):
        source = "AI systems shall be considered to be high-risk."
        quote = "AI systems are exempt from this Regulation."
        assert normalize_text(quote) not in normalize_text(source)
