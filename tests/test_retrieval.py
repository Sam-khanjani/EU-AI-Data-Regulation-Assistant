"""Retrieval tests against a real Postgres, using a seeded fixture corpus.

Embeddings here are deterministic hand-built vectors, not model output. That is deliberate:
it tests the retrieval *logic* -- the SQL, the fusion, the paragraph-to-article expansion,
the active-version filter -- without depending on an embedding provider or a network call.
Whether the real embeddings are any good is a question for the eval suite, not for these.

The fixture writes under its own source key and tears itself down, so it never disturbs an
ingested corpus sharing the same database.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import text

from euaia.config import settings
from euaia.db.models import Chunk, DocumentVersion, Source, StructuralUnit
from euaia.db.session import SessionLocal, engine
from euaia.retrieval.hybrid import (
    RetrievedUnit,
    citation_label,
    dense_search,
    expand_to_units,
    fit_token_budget,
    fulltext_search,
    reciprocal_rank_fusion,
    retrieve,
    structural_search,
)

TEST_SOURCE_KEY = "pytest-fixture-corpus"


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(), reason="Postgres not reachable; run `docker compose up -d db`"
)


def _vec(*leading: float) -> list[float]:
    """A vector whose first components are given and the rest zero."""
    v = [0.0] * settings.embed_dim
    for i, value in enumerate(leading):
        v[i] = value
    return v


@pytest.fixture(scope="module")
def corpus():
    """Seed a miniature two-version corpus and tear it down afterwards."""
    session = SessionLocal()
    try:
        _purge(session)

        source = Source(
            key=TEST_SOURCE_KEY,
            title="Fixture Regulation",
            publisher="pytest",
            source_type="eurlex",
            celex_base="39999R0001",
        )
        session.add(source)
        session.flush()

        # An older version that must never be retrieved.
        stale = DocumentVersion(
            source_id=source.id,
            version_label="superseded 2020-01-01",
            celex="09999R0001-20200101",
            content_sha256="0" * 64,
            format="pdf",
            status="superseded",
            doc_date=dt.date(2020, 1, 1),
        )
        live = DocumentVersion(
            source_id=source.id,
            version_label="consolidated 2026-07-27",
            celex="09999R0001-20260727",
            content_sha256="1" * 64,
            format="pdf",
            status="active",
            doc_date=dt.date(2026, 7, 27),
        )
        session.add_all([stale, live])
        session.flush()

        chapter = _unit(session, live, "chapter", "III", "CH_III", "HIGH-RISK AI SYSTEMS", 1)
        art6 = _unit(
            session, live, "article", "6", "CH_III/ART_6",
            "Classification rules for high-risk AI systems", 2, parent=chapter,
            body="Article 6\nClassification rules for high-risk AI systems",
        )
        par1 = _unit(
            session, live, "paragraph", "6(1)", "CH_III/ART_6/PAR_1", None, 3, parent=art6,
            body="1.\nthat AI system shall be considered to be high-risk where both of the "
                 "following conditions are fulfilled",
        )
        par2 = _unit(
            session, live, "paragraph", "6(2)", "CH_III/ART_6/PAR_2", None, 4, parent=art6,
            body="2.\nAI systems referred to in Annex III shall be considered to be high-risk.",
        )
        art50 = _unit(
            session, live, "article", "50", "CH_IV/ART_50",
            "Transparency obligations", 5,
            body="Article 50\nTransparency obligations\nProviders shall ensure that AI "
                 "systems intended to interact directly with natural persons are designed "
                 "so that users are informed they are interacting with an AI system.",
        )
        annex3 = _unit(
            session, live, "annex", "III", "ANX_III",
            "High-risk AI systems referred to in Article 6(2)", 6,
            body="ANNEX III\nBiometrics, employment, and access to essential services.",
        )
        # Same provision in the stale version: must be filtered out by status.
        stale_art6 = _unit(
            session, stale, "article", "6", "CH_III/ART_6", "Old classification rules", 1,
            body="Article 6\nOld classification rules that were repealed and must never "
                 "be cited as if in force.",
        )

        _chunk(session, live, art6, "high-risk classification conditions", _vec(1.0))
        _chunk(session, live, par1, "considered to be high-risk where both conditions", _vec(0.9, 0.1))
        _chunk(session, live, par2, "Annex III high-risk systems", _vec(0.8, 0.2))
        _chunk(session, live, art50, "transparency obligations chatbot interact", _vec(0.0, 1.0))
        _chunk(session, live, annex3, "biometrics employment essential services", _vec(0.0, 0.0, 1.0))
        _chunk(session, stale, stale_art6, "old repealed classification rules", _vec(1.0))

        session.commit()
        yield {
            "art6": art6.id, "par1": par1.id, "par2": par2.id,
            "art50": art50.id, "annex3": annex3.id, "stale_art6": stale_art6.id,
            "live_version": live.id, "stale_version": stale.id,
        }
    finally:
        _purge(session)
        session.commit()
        session.close()


def _purge(session) -> None:
    """Remove the fixture corpus. Raw SQL so ORM cascades cannot double-delete."""
    session.execute(
        text("DELETE FROM source WHERE key = :key"), {"key": TEST_SOURCE_KEY}
    )
    session.flush()


def _unit(session, version, unit_type, number, path, heading, ordinal, parent=None, body=None):
    unit = StructuralUnit(
        document_version_id=version.id,
        parent_id=parent.id if parent else None,
        unit_type=unit_type,
        unit_number=number,
        unit_path=path,
        heading=heading,
        text=body or heading or path,
        text_normalized=(body or heading or path).lower(),
        ordinal=ordinal,
        eurlex_deeplink=f"https://example.test/#{unit_type}_{number}",
    )
    session.add(unit)
    session.flush()
    return unit


def _chunk(session, version, unit, body, embedding):
    chunk = Chunk(
        structural_unit_id=unit.id,
        document_version_id=version.id,
        text=body,
        token_count=len(body.split()),
        embedding=embedding,
        embed_model="fixture",
    )
    session.add(chunk)
    session.flush()
    return chunk


@pytest.fixture
def session():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


class TestDenseSearch:
    def test_returns_nearest_first(self, corpus, session):
        hits = dense_search(session, _vec(1.0), limit=5, source_keys=[TEST_SOURCE_KEY])
        assert hits, "dense search returned nothing"
        assert hits[0].unit_id == corpus["art6"]

    def test_a_different_direction_ranks_differently(self, corpus, session):
        hits = dense_search(session, _vec(0.0, 1.0), limit=5, source_keys=[TEST_SOURCE_KEY])
        assert hits[0].unit_id == corpus["art50"]

    def test_superseded_versions_are_never_returned(self, corpus, session):
        # The stale Article 6 shares an identical embedding with the live one.
        hits = dense_search(session, _vec(1.0), limit=10, source_keys=[TEST_SOURCE_KEY])
        assert corpus["stale_art6"] not in {h.unit_id for h in hits}


class TestFulltextSearch:
    def test_finds_exact_legal_tokens(self, corpus, session):
        hits = fulltext_search(session, "Annex III", limit=5, source_keys=[TEST_SOURCE_KEY])
        assert corpus["par2"] in {h.unit_id for h in hits}

    def test_finds_domain_vocabulary(self, corpus, session):
        hits = fulltext_search(session, "transparency obligations", limit=5, source_keys=[TEST_SOURCE_KEY])
        assert corpus["art50"] in {h.unit_id for h in hits}

    def test_superseded_versions_are_excluded(self, corpus, session):
        hits = fulltext_search(session, "repealed classification", limit=10, source_keys=[TEST_SOURCE_KEY])
        assert corpus["stale_art6"] not in {h.unit_id for h in hits}

    def test_no_match_returns_empty(self, corpus, session):
        assert fulltext_search(session, "zzzznonexistenttoken", limit=5, source_keys=[TEST_SOURCE_KEY]) == []


class TestStructuralSearch:
    def test_named_article_is_found(self, corpus, session):
        hits = structural_search(session, ["50"], [], source_keys=[TEST_SOURCE_KEY])
        assert corpus["art50"] in {h.unit_id for h in hits}

    def test_named_annex_is_found(self, corpus, session):
        hits = structural_search(session, [], ["III"], source_keys=[TEST_SOURCE_KEY])
        assert corpus["annex3"] in {h.unit_id for h in hits}

    def test_article_lookup_also_returns_its_paragraphs(self, corpus, session):
        found = {h.unit_id for h in structural_search(session, ["6"], [], source_keys=[TEST_SOURCE_KEY])}
        assert corpus["par1"] in found and corpus["par2"] in found

    def test_annex_lookup_is_case_insensitive(self, corpus, session):
        assert structural_search(session, [], ["iii"], source_keys=[TEST_SOURCE_KEY])

    def test_nothing_named_returns_nothing(self, corpus, session):
        assert structural_search(session, [], [], source_keys=[TEST_SOURCE_KEY]) == []

    def test_superseded_article_is_not_returned(self, corpus, session):
        hits = structural_search(session, ["6"], [], source_keys=[TEST_SOURCE_KEY])
        assert corpus["stale_art6"] not in {h.unit_id for h in hits}


class TestFusion:
    def test_hit_in_both_legs_outranks_hit_in_one(self, corpus, session):
        dense = dense_search(session, _vec(0.0, 1.0), limit=5, source_keys=[TEST_SOURCE_KEY])
        lexical = fulltext_search(session, "transparency obligations", limit=5, source_keys=[TEST_SOURCE_KEY])
        fused = reciprocal_rank_fusion({"dense": dense, "fulltext": lexical})
        assert fused[0].unit_id == corpus["art50"]
        assert set(fused[0].legs) == {"dense", "fulltext"}

    def test_weighting_raises_the_weighted_legs_contribution(self, corpus, session):
        legs = {
            "dense": dense_search(session, _vec(1.0), limit=5, source_keys=[TEST_SOURCE_KEY]),
            "structural": structural_search(session, ["50"], [], source_keys=[TEST_SOURCE_KEY]),
        }
        unweighted = {c.unit_id: c.score for c in reciprocal_rank_fusion(legs)}
        weighted = {
            c.unit_id: c.score
            for c in reciprocal_rank_fusion(legs, weights={"structural": 10.0})
        }
        # Article 50 is the structural hit; Article 6 is dense-only and must be unmoved.
        assert weighted[corpus["art50"]] > unweighted[corpus["art50"]]
        assert weighted[corpus["art6"]] == pytest.approx(unweighted[corpus["art6"]])

    def test_scores_are_descending(self, corpus, session):
        fused = reciprocal_rank_fusion(
            {"a": dense_search(session, _vec(1.0), limit=5, source_keys=[TEST_SOURCE_KEY]),
             "b": fulltext_search(session, "high-risk", limit=5, source_keys=[TEST_SOURCE_KEY])}
        )
        assert [c.score for c in fused] == sorted((c.score for c in fused), reverse=True)


class TestExpansion:
    def test_paragraphs_collapse_into_their_article(self, corpus, session):
        candidates = structural_search(session, ["6"], [], source_keys=[TEST_SOURCE_KEY])
        units = expand_to_units(session, candidates)
        ids = [u.unit_id for u in units]
        assert corpus["art6"] in ids
        assert corpus["par1"] not in ids and corpus["par2"] not in ids

    def test_expanded_unit_carries_full_article_text_and_link(self, corpus, session):
        units = expand_to_units(session, structural_search(session, ["6"], [], source_keys=[TEST_SOURCE_KEY]))
        art6 = next(u for u in units if u.unit_id == corpus["art6"])
        assert art6.citation_label == "Article 6"
        assert art6.deeplink == "https://example.test/#article_6"
        assert "Classification rules" in art6.text

    def test_multiple_matching_paragraphs_merge_their_chunk_ids(self, corpus, session):
        units = expand_to_units(session, structural_search(session, ["6"], [], source_keys=[TEST_SOURCE_KEY]))
        art6 = next(u for u in units if u.unit_id == corpus["art6"])
        assert len(art6.matched_chunk_ids) >= 2

    def test_annex_is_not_promoted(self, corpus, session):
        units = expand_to_units(session, structural_search(session, [], ["III"], source_keys=[TEST_SOURCE_KEY]))
        assert [u.unit_id for u in units] == [corpus["annex3"]]


class TestRetrieveEndToEnd:
    def test_named_article_wins_over_a_merely_similar_one(self, corpus, session):
        candidates = retrieve(
            session,
            query="what does Article 50 require",
            embedding=_vec(1.0),  # points at Article 6
            articles=["50"],
            annexes=[],
            source_keys=[TEST_SOURCE_KEY],
        )
        assert candidates[0].unit_id == corpus["art50"], (
            "an explicitly named provision must outrank embedding similarity"
        )

    def test_returns_chunk_candidates_not_expanded_units(self, corpus, session):
        # Expansion happens after reranking, so paragraphs must still be separate here.
        candidates = retrieve(session, query="high-risk", embedding=_vec(1.0), source_keys=[TEST_SOURCE_KEY])
        assert candidates
        found = {c.unit_id for c in candidates}
        assert corpus["par1"] in found or corpus["par2"] in found

    def test_candidates_carry_chunk_text_and_provenance(self, corpus, session):
        for candidate in retrieve(session, query="high-risk", embedding=_vec(1.0), source_keys=[TEST_SOURCE_KEY]):
            assert candidate.chunk_text, "reranking needs the chunk text"
            assert candidate.text, "verification needs the unit text"
            assert candidate.version_label == "consolidated 2026-07-27"
            assert candidate.document_version_id == corpus["live_version"]


class TestTokenBudget:
    """Free-tier tokens-per-minute is the scarcest resource; the budget enforces it."""

    def _unit(self, unit_id: int, body: str) -> RetrievedUnit:
        return RetrievedUnit(
            unit_id=unit_id, unit_path=f"ART_{unit_id}", unit_type="article",
            unit_number=str(unit_id), heading=None, citation_label=f"Article {unit_id}",
            text=body, document_version_id=1, version_label="v", source_key="k",
            deeplink=None, score=1.0,
        )

    def test_stops_once_the_budget_is_spent(self):
        units = [self._unit(i, "word " * 400) for i in range(6)]  # ~550 tokens each
        kept = fit_token_budget(units, budget=1200)
        assert 1 <= len(kept) < len(units)

    def test_preserves_rank_order(self):
        units = [self._unit(i, "word " * 100) for i in range(5)]
        kept = fit_token_budget(units, budget=400)
        assert [u.unit_id for u in kept] == sorted(u.unit_id for u in kept)

    def test_keeps_one_oversized_unit_rather_than_returning_nothing(self):
        # Abstaining because the single relevant provision is long would be worse than
        # spending the budget on it.
        units = [self._unit(1, "word " * 5000)]
        assert len(fit_token_budget(units, budget=100)) == 1

    def test_everything_fits_under_a_generous_budget(self):
        units = [self._unit(i, "short text here") for i in range(4)]
        assert len(fit_token_budget(units, budget=100_000)) == 4

    def test_empty_input(self):
        assert fit_token_budget([], budget=1000) == []


class TestCitationLabel:
    @pytest.mark.parametrize(
        ("unit_type", "number", "expected"),
        [
            ("article", "6", "Article 6"),
            ("article", "4a", "Article 4a"),
            ("annex", "III", "Annex III"),
            ("recital", "27", "Recital 27"),
            ("paragraph", "6(2)", "Paragraph 6(2)"),
        ],
    )
    def test_labels(self, unit_type, number, expected):
        assert citation_label(unit_type, number, "X") == expected

    def test_falls_back_to_path_when_unnumbered(self):
        assert citation_label("article", None, "CH_I/ART_X") == "CH_I/ART_X"
