"""End-to-end pipeline tests against the real answering model.

These run the whole graph -- analyse, rerank, generate, verify, finalise -- against Groq,
using **real AI Act text** parsed from the cached PDF. Only two things are stubbed:

* **Query embeddings.** The Gemini free tier allows 1,000 items per day and the corpus
  needs 779, so spending quota on tests would make ingestion impossible. Retrieval here
  leans on the full-text and structural legs, which need no vectors.
* **Chunk vectors**, for the same reason.

Everything that matters for trustworthiness is real: the model genuinely writes the quotes,
and the verifier genuinely checks them against the regulation's own words. A hallucinated
quote fails here exactly as it would in production.

Skipped without ``GROQ_API_KEY``. Each test costs Groq tokens against a 200,000/day budget,
so the suite is deliberately small.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text as sql_text

from euaia.config import settings
from euaia.db.models import Chunk, DocumentVersion, Source, StructuralUnit
from euaia.db.session import SessionLocal, engine
from euaia.ingest import deeplinks
from euaia.ingest.pdf_parser import parse
from euaia.verify.normalize import normalize_text

CORPUS_KEY = "pytest-live-corpus"
CELEX = "02024R1689-20260727"
RAW = settings.raw_data_dir / f"{CELEX}.ENG.pdf"

# A small but genuinely representative slice: a prohibition, the high-risk test, the
# transparency duty, and the annex the high-risk test refers to.
WANTED_ARTICLES = {"5", "6", "50", "26"}
WANTED_ANNEXES = {"III"}


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not (settings.groq_api_key and _db_available() and RAW.exists()),
    reason="needs GROQ_API_KEY, Postgres, and the cached AI Act PDF",
)


class StubEmbedder:
    """Stands in for Gemini so tests cost no embedding quota."""

    model = "stub-embedder"

    def embed_query(self, text: str) -> list[float]:
        return _vec(text)

    def embed_documents(self, texts, batch_size: int = 16):
        return [_vec(t) for t in texts]


def _vec(text: str) -> list[float]:
    v = [0.0] * settings.embed_dim
    v[0] = (len(text) % 101) / 101.0
    v[1] = (sum(map(ord, text[:16])) % 97) / 97.0
    return v


@pytest.fixture(scope="module")
def corpus():
    """Load real Article 5/6/26/50 and Annex III text under an isolated source key."""
    doc = parse(RAW.read_bytes())
    session = SessionLocal()
    try:
        _purge(session)
        source = Source(
            key=CORPUS_KEY,
            title="Regulation (EU) 2024/1689 (AI Act) - live test slice",
            publisher="pytest",
            source_type="eurlex",
            celex_base="32024R1689",
        )
        session.add(source)
        session.flush()

        version = DocumentVersion(
            source_id=source.id,
            version_label="consolidated 2026-07-27",
            celex=CELEX,
            content_sha256="f" * 64,
            format="pdf",
            status="active",
        )
        session.add(version)
        session.flush()

        wanted = [
            u
            for u in doc.units
            if (u.unit_type == "article" and u.unit_number in WANTED_ARTICLES)
            or (u.unit_type == "annex" and u.unit_number in WANTED_ANNEXES)
        ]
        assert wanted, "expected provisions not found in the cached PDF"

        for unit in wanted:
            row = StructuralUnit(
                document_version_id=version.id,
                unit_type=unit.unit_type,
                unit_number=unit.unit_number,
                unit_path=unit.unit_path,
                heading=unit.heading,
                text=unit.text,
                text_normalized=normalize_text(unit.text),
                ordinal=unit.ordinal,
                eurlex_deeplink=deeplinks.build(
                    CELEX, unit.unit_type, unit.unit_number, unit.unit_path
                ),
            )
            session.add(row)
            session.flush()
            session.add(
                Chunk(
                    structural_unit_id=row.id,
                    document_version_id=version.id,
                    text=unit.text[:4000],
                    token_count=len(unit.text) // 4,
                    embedding=_vec(unit.text),
                    embed_model="stub-embedder",
                )
            )
        session.commit()
        yield version.id
    finally:
        _purge(session)
        session.commit()
        session.close()


def _purge(session) -> None:
    session.execute(sql_text("DELETE FROM source WHERE key = :k"), {"k": CORPUS_KEY})
    session.flush()


@pytest.fixture(scope="module")
def ask(corpus):
    """Run a question through the real pipeline.

    Module-scoped so each expensive answer is produced once: every call spends Groq
    tokens against a 200,000/day budget.
    """
    from euaia.api import service
    from euaia.llm.groq_client import GroqClient

    client = GroqClient()
    embedder = StubEmbedder()

    def _ask(question: str):
        session = SessionLocal()
        try:
            return service.ask(question, session, client, embedder)
        finally:
            session.close()

    return _ask


@pytest.fixture(scope="module")
def prohibited_answer(ask):
    return ask("Which AI practices are prohibited?")


@pytest.fixture(scope="module")
def cv_assessment(ask):
    return ask(
        "I built a tool that screens CVs and ranks job candidates for our HR team. "
        "Is it a high-risk AI system?"
    )


class TestGroundedAnswer:
    @pytest.fixture(autouse=True)
    def _answer(self, prohibited_answer):
        self.answer = prohibited_answer

    def test_it_answers(self):
        assert self.answer.verdict in ("answered", "partial"), self.answer.abstain_reason

    def test_every_displayed_quote_was_verified(self):
        # The central guarantee: nothing reaches the user unverified.
        assert self.answer.quotes_total > 0
        citations = [c for claim in self.answer.claims for c in claim.citations]
        assert citations, "an answered response must carry citations"
        for citation in citations:
            assert citation.method in ("exact", "elided")

    def test_quotes_appear_verbatim_in_the_regulation(self):
        """Re-check independently of the verifier, against the database."""
        with SessionLocal() as session:
            corpus_text = normalize_text(
                " ".join(
                    session.scalars(
                        sql_text(
                            "SELECT su.text FROM structural_unit su "
                            "JOIN document_version dv ON dv.id = su.document_version_id "
                            "JOIN source s ON s.id = dv.source_id WHERE s.key = :k"
                        ),
                        {"k": CORPUS_KEY},
                    ).all()
                )
            )
        for claim in self.answer.claims:
            for citation in claim.citations:
                assert normalize_text(citation.quote) in corpus_text, (
                    f"displayed quote is not in the regulation: {citation.quote[:80]!r}"
                )

    def test_it_cites_article_5(self):
        cited = {c.citation_label for claim in self.answer.claims for c in claim.citations}
        assert any("5" in label for label in cited), f"expected Article 5, got {cited}"

    def test_citations_carry_working_links_and_version(self):
        for claim in self.answer.claims:
            for citation in claim.citations:
                assert citation.deeplink and citation.deeplink.startswith("https://")
                assert citation.version_label == "consolidated 2026-07-27"

    def test_the_attempt_is_logged(self):
        assert self.answer.query_log_id is not None
        with SessionLocal() as session:
            row = session.execute(
                sql_text(
                    "SELECT verdict, document_version_ids, prompt_version "
                    "FROM query_log WHERE id = :i"
                ),
                {"i": self.answer.query_log_id},
            ).one()
        assert row.verdict == self.answer.verdict
        assert row.document_version_ids, "provenance must record the versions read"


class TestAbstention:
    def test_out_of_corpus_question_is_refused(self, ask):
        answer = ask("What is the lawful basis for processing personal data under the GDPR?")
        assert answer.verdict == "abstained"
        assert answer.abstain_reason

    def test_invented_provision_is_refused(self, ask):
        # Article 99a does not exist. Retrieval will still return something.
        answer = ask("What does Article 99a of the AI Act require?")
        assert answer.verdict == "abstained", (
            f"invented a provision: {[c.text for c in answer.claims]}"
        )


class TestStructuredSelfAssessment:
    @pytest.fixture(autouse=True)
    def _assessment(self, cv_assessment):
        self.assessment = cv_assessment

    def test_classified_as_applicability(self):
        assert self.assessment.intent == "applicability"

    def test_returns_criteria_rather_than_a_verdict(self):
        if self.assessment.verdict == "abstained":
            pytest.skip(f"abstained: {self.assessment.abstain_reason}")
        assert self.assessment.criteria, "an applicability answer must lay out criteria"

    def test_states_no_legal_conclusion(self):
        blob = " ".join(
            [self.assessment.summary] + [c["explanation"] for c in self.assessment.criteria]
        ).lower()
        for phrase in ("your system is high-risk", "your tool is high-risk", "is prohibited"):
            assert phrase not in blob, f"stated a verdict: {phrase!r}"

    def test_open_criteria_ask_the_user_for_facts(self):
        if not self.assessment.criteria:
            pytest.skip("no criteria produced")
        statuses = {c["status"] for c in self.assessment.criteria}
        assert statuses <= {"met", "not_met", "needs_user_input"}
