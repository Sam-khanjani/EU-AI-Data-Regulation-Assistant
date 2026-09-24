"""Embedding cache tests.

The cache is what makes this project usable on a free tier that allows 1,000 embed items per
day against a 779-chunk corpus. It has to be right about three things: never call the API
for text it already has, never lose work when a run fails part way, and never hand back a
vector produced by a different model.

A fake embedder counts calls, so these run offline and assert exactly what would have been
spent.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from euaia.config import settings
from euaia.db.session import SessionLocal, engine
from euaia.ingest import embeddings
from euaia.ingest.embeddings import (
    QuotaExhausted,
    embed_with_cache,
    plan,
    text_hash,
)


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

MARKER = "pytest-cache-fixture"


class FakeEmbedder:
    """Counts what a real embedder would have spent."""

    def __init__(self, model: str = "fake-embed-001", fail_after: int | None = None):
        self.model = model
        self.items_embedded = 0
        self.calls = 0
        self.fail_after = fail_after

    def embed_documents(self, texts, batch_size: int = 16):
        self.calls += 1
        if self.fail_after is not None and self.items_embedded >= self.fail_after:
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED. You exceeded your current quota"
            )
        self.items_embedded += len(texts)
        # Deterministic, distinguishable vectors.
        out = []
        for t in texts:
            v = [0.0] * settings.embed_dim
            v[0] = float(len(t) % 97) / 97.0
            v[1] = float(sum(map(ord, t[:8])) % 89) / 89.0
            out.append(v)
        return out


@pytest.fixture(autouse=True)
def no_real_waits(monkeypatch):
    """Quota rejections now pause before giving up; record the pauses instead of sleeping."""
    pauses: list[float] = []
    monkeypatch.setattr(embeddings.time, "sleep", pauses.append)
    return pauses


@pytest.fixture(autouse=True)
def clean_cache():
    """Remove only the fixture's own entries, leaving a real corpus cache intact."""

    def purge():
        with SessionLocal() as s:
            s.execute(
                text("DELETE FROM embedding_cache WHERE model LIKE 'fake-%'")
            )
            s.commit()

    purge()
    yield
    purge()


def texts(n: int, marker: str = MARKER) -> list[str]:
    return [f"{marker} chunk number {i} with some legal-sounding text" for i in range(n)]


class TestReuse:
    def test_first_run_embeds_everything(self):
        fake = FakeEmbedder()
        vectors, stats = embed_with_cache(texts(10), fake)
        assert len(vectors) == 10
        assert stats.computed == 10 and stats.reused == 0
        assert fake.items_embedded == 10

    def test_second_run_spends_nothing(self):
        embed_with_cache(texts(10), FakeEmbedder())
        fake = FakeEmbedder()
        vectors, stats = embed_with_cache(texts(10), fake)
        assert fake.items_embedded == 0, "re-embedding identical text wastes daily quota"
        assert stats.reused == 10 and stats.computed == 0
        assert len(vectors) == 10

    def test_reused_vectors_match_within_halfvec_precision(self):
        # The cache column is halfvec (fp16), same as chunk.embedding, so a round trip
        # loses a little precision. Both paths end up storing the same fp16 value, and
        # cosine similarity is unaffected at this magnitude.
        first, _ = embed_with_cache(texts(4), FakeEmbedder())
        second, _ = embed_with_cache(texts(4), FakeEmbedder())
        for a, b in zip(first, second, strict=True):
            assert a == pytest.approx(b, abs=1e-3)

    def test_only_changed_text_costs_anything(self):
        # The amendment case: most provisions are byte-identical, a few changed.
        embed_with_cache(texts(10), FakeEmbedder())
        changed = texts(10)
        changed[3] = "amended provision with entirely new wording"
        changed[7] = "another amended provision"

        fake = FakeEmbedder()
        _, stats = embed_with_cache(changed, fake)
        assert fake.items_embedded == 2, "a re-consolidation must only pay for what moved"
        assert stats.reused == 8

    def test_duplicate_text_within_one_run_costs_once(self):
        fake = FakeEmbedder()
        repeated = ["identical chunk text " + MARKER] * 5
        vectors, stats = embed_with_cache(repeated, fake)
        assert fake.items_embedded == 1
        assert len(vectors) == 5
        assert all(v == vectors[0] for v in vectors)
        assert stats.computed == 1

    def test_order_is_preserved(self):
        items = texts(6)
        vectors, _ = embed_with_cache(items, FakeEmbedder())
        # Re-request in a different order and confirm the mapping follows the input.
        shuffled = list(reversed(items))
        reordered, _ = embed_with_cache(shuffled, FakeEmbedder())
        for a, b in zip(reordered, reversed(vectors), strict=True):
            assert a == pytest.approx(b, abs=1e-3)


class TestModelIsolation:
    def test_a_different_model_does_not_reuse_vectors(self):
        embed_with_cache(texts(5), FakeEmbedder(model="fake-embed-001"))
        other = FakeEmbedder(model="fake-embed-002")
        _, stats = embed_with_cache(texts(5), other)
        assert other.items_embedded == 5, (
            "vectors from a different model live in a different space and must not be reused"
        )
        assert stats.reused == 0


class TestResumability:
    def test_work_before_a_quota_failure_survives(self):
        # Fail after 32 items: two batches of 16 succeed and are committed.
        fake = FakeEmbedder(fail_after=32)
        with pytest.raises(QuotaExhausted):
            embed_with_cache(texts(80), fake)

        # A later run reuses what the failed run managed to compute.
        resumed = FakeEmbedder()
        _, stats = embed_with_cache(texts(80), resumed)
        assert stats.reused == 32, "committed batches must survive the failure"
        assert resumed.items_embedded == 48

    def test_quota_error_is_reported_not_raised_raw(self):
        fake = FakeEmbedder(fail_after=0)
        with pytest.raises(QuotaExhausted, match="daily embedding quota is used up"):
            embed_with_cache(texts(20), fake)


class TestPerMinuteWindow:
    """A 429 without a retry hint is usually the per-minute window, not the daily quota."""

    def test_an_unhinted_rejection_is_waited_out_and_the_run_completes(self, no_real_waits):
        fake = FakeEmbedder(fail_after=16)
        original = fake.embed_documents

        def recovers_after_one_window(batch, batch_size=16):
            if no_real_waits:  # the window has passed once we have paused
                fake.fail_after = None
            return original(batch, batch_size)

        fake.embed_documents = recovers_after_one_window
        vectors, stats = embed_with_cache(texts(40), fake)
        assert len(vectors) == 40 and stats.computed == 40
        assert no_real_waits == [embeddings.UNHINTED_QUOTA_WAIT_SECONDS + 1]

    def test_waiting_that_brings_no_progress_gives_up(self, no_real_waits):
        """The daily quota really is spent: stop after a bounded number of pauses."""
        with pytest.raises(QuotaExhausted):
            embed_with_cache(texts(20), FakeEmbedder(fail_after=0))
        assert len(no_real_waits) == embeddings.MAX_STALLED_WAITS

    def test_a_long_hinted_delay_is_not_waited_out(self, no_real_waits):
        fake = FakeEmbedder(fail_after=0)
        fake.embed_documents = lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 7200s.")
        )
        with pytest.raises(QuotaExhausted):
            embed_with_cache(texts(5), fake)
        assert no_real_waits == []


class TestSpendCap:
    def test_cap_stops_before_spending_anything(self):
        fake = FakeEmbedder()
        with pytest.raises(QuotaExhausted, match="max-embeddings"):
            embed_with_cache(texts(50), fake, max_new=10)
        assert fake.items_embedded == 0, "the cap must refuse up front, not part way"

    def test_cap_allows_a_run_that_fits(self):
        fake = FakeEmbedder()
        _, stats = embed_with_cache(texts(10), fake, max_new=10)
        assert stats.computed == 10

    def test_cap_counts_only_new_work(self):
        embed_with_cache(texts(40), FakeEmbedder())
        fake = FakeEmbedder()
        # 40 cached + 5 new: a cap of 5 must be enough.
        items = texts(40) + ["brand new chunk " + MARKER + str(i) for i in range(5)]
        _, stats = embed_with_cache(items, fake, max_new=5)
        assert stats.reused == 40 and stats.computed == 5


class TestPlan:
    def test_reports_cost_without_calling_the_api(self):
        fake = FakeEmbedder()
        estimate = plan(texts(12), fake)
        assert estimate.total == 12 and estimate.to_embed == 12 and estimate.cached == 0
        assert fake.items_embedded == 0, "estimating must not spend quota"

    def test_reflects_cached_entries(self):
        embed_with_cache(texts(12), FakeEmbedder())
        estimate = plan(texts(12), FakeEmbedder())
        assert estimate.cached == 12 and estimate.to_embed == 0


class TestHashing:
    def test_identical_text_hashes_identically(self):
        assert text_hash("Article 6") == text_hash("Article 6")

    def test_whitespace_difference_is_a_different_key(self):
        # The cache is content-addressed on the exact embedded string, deliberately: a
        # different string produces a different vector.
        assert text_hash("Article 6") != text_hash("Article  6")


class TestQuotaDetection:
    @pytest.mark.parametrize(
        "message",
        [
            "429 RESOURCE_EXHAUSTED. {'error': {'code': 429}}",
            "You exceeded your current quota, please check your plan",
        ],
    )
    def test_recognises_provider_quota_errors(self, message):
        assert embeddings.is_quota_error(RuntimeError(message))

    def test_other_errors_are_not_mistaken_for_quota(self):
        assert not embeddings.is_quota_error(RuntimeError("connection reset"))
