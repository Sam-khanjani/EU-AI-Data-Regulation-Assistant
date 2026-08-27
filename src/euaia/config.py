"""Application settings, loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Credentials ---
    groq_api_key: str = ""
    google_api_key: str = ""

    # --- Database ---
    # No default: must be set in .env. Keeps the repo free of any embedded credential,
    # even a throwaway local one, so there is nothing for secret scanners to (correctly
    # or not) flag.
    database_url: str = ""

    # --- Models ---
    # Answer generation. Must be a Groq model supporting strict JSON schema mode.
    groq_model: str = "openai/gpt-oss-120b"
    # Query analysis + reranking: cheaper, also strict-capable.
    groq_small_model: str = "openai/gpt-oss-20b"

    embed_model: str = "gemini-embedding-001"
    # gemini-embedding-001 defaults to 3072. Anything below 3072 on *-001* requires
    # manual L2 re-normalisation, so we keep full width and store as halfvec(3072).
    embed_dim: int = 3072
    # Hard ceiling from the Gemini embedding API.
    embed_input_token_limit: int = 2048

    # --- Chunking ---
    chunk_target_tokens: int = 500
    chunk_max_tokens: int = 1500  # kept well under embed_input_token_limit

    # --- Free-tier rate limits (Groq, per model) ---
    # Tokens-per-minute is the binding constraint: a single answer call spends thousands,
    # so the pipeline is sized around this number rather than around latency.
    groq_rpm: int = 30
    groq_tpm: int = 8_000
    groq_tpd: int = 200_000

    # --- Retrieval ---
    retrieve_candidates: int = 20
    rerank_keep: int = 4
    # Reranker score (0-10) below which a candidate is treated as irrelevant.
    rerank_min_score: float = 4.0

    # --- Token budgets ---
    # Reranking scores CHUNKS (packed to chunk_target_tokens, ~500 each), not the articles
    # they belong to (~530 avg, 3369 max). Scoring expanded articles would put a single
    # rerank call at ~16k tokens -- twice the free-tier minute budget -- and it is also the
    # wrong thing to score: relevance belongs to the passage that matched.
    #
    # This must stay large enough to hold most of retrieve_candidates chunks at roughly
    # chunk_target_tokens each, or candidates below the cutoff are silently dropped before
    # the reranker ever sees them -- discovered when a chunk-packing rewrite quadrupled
    # average chunk size (~99 -> ~400 tokens) without this being re-tuned to match, which
    # left only ~6 of 20 fused candidates reaching the reranker. 3,600 is the largest value
    # that still fits tests/test_budgets.py::TestDailyBudget's full-evaluation-run ceiling.
    rerank_token_budget: int = 3_600
    # Total evidence handed to the answer model, after expanding survivors to articles.
    evidence_token_budget: int = 2_400
    # Output allowance for the answer call. Prompt + evidence + this must fit inside the
    # rate limiter's *usable* budget (groq_tpm x headroom, i.e. 6,800 of 8,000), not the
    # raw limit -- otherwise the request can never be issued at all.
    # tests/test_budgets.py enforces the arithmetic.
    answer_max_tokens: int = 2_200
    # Rough allowance for the system prompt and question wrapped around the evidence.
    prompt_overhead_tokens: int = 700
    # A paragraph is not promoted into its parent article if the article exceeds this.
    # Article 5 alone is ~1,900 tokens; expanding into it crowds out every other
    # provision and spends the minute budget on text the question did not ask about.
    max_expand_tokens: int = 1_200
    # Cap on claims requested, to keep the structured answer inside answer_max_tokens.
    max_claims: int = 4
    # Fallback cap used when the model overruns and the response is rejected.
    max_claims_retry: int = 2

    # --- Answer gates ---
    # Fraction of claims that must carry a verified quote to answer outright.
    coverage_answer_threshold: float = 0.8
    # Below this we abstain entirely rather than degrade to a partial answer.
    coverage_partial_threshold: float = 0.4

    # --- CELLAR / EUR-Lex ---
    cellar_sparql_endpoint: str = "https://publications.europa.eu/webapi/rdf/sparql"
    cellar_resource_base: str = "https://publications.europa.eu/resource/cellar"
    # CELLAR asks for a descriptive UA and fewer than 5 concurrent requests.
    cellar_user_agent: str = (
        "euaia-research-assistant/0.1 (EU AI Act RAG prototype; contact: local dev)"
    )
    cellar_max_concurrency: int = 4
    cellar_timeout_seconds: float = 60.0

    # --- Paths ---
    raw_data_dir: Path = REPO_ROOT / "data" / "raw"

    # --- Prompt versioning (recorded in query_log for auditability) ---
    prompt_version: str = "v1"


settings = Settings()
