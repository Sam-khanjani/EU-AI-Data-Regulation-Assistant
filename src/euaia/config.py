"""Application settings, loaded from environment / .env.

Almost everything here can be overridden from ``.env``. The exceptions are declared as
``ClassVar`` -- which reranker scores the evidence -- and are changed by editing this file:
they decide what the assistant answers from, so they live in version control rather than
drifting per machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

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
    # Only needed when rerank_provider = "openrouter".
    openrouter_api_key: str = ""

    # --- Admin auth ---
    # Gates /status and /status/check (corpus contents, change detection -- not meant for
    # every visitor). No default password: an empty value must refuse every request rather
    # than silently accepting an empty one, so admin/api/main.py checks for that explicitly
    # rather than relying on a default here.
    admin_username: str = "admin"
    admin_password: str = ""

    # --- Database ---
    # No default: must be set in .env. Keeps the repo free of any embedded credential,
    # even a throwaway local one, so there is nothing for secret scanners to (correctly
    # or not) flag.
    database_url: str = ""

    # --- Models ---
    # Answer generation. Must be a Groq model supporting strict JSON schema mode.
    groq_model: str = "openai/gpt-oss-120b"
    # Query analysis. Reranking used to share this model; it is local now.
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

    # --- Reranking ---
    # Set here only: ClassVar keeps these out of .env and environment variables, so the
    # choice of reranker is changed by editing this file (and rebuilding the Docker image).
    #
    # Where candidates are scored. "local" runs the cross-encoder `rerank_model` below on
    # this machine: no quota, no network, ~30s per question on CPU. "openrouter" sends the
    # same (question, passage) pairs to `openrouter_rerank_model` in one request: about a
    # second, but a network dependency with its own quota -- the free tier allows 50 requests
    # a day across all free models, and every question needs one. On failure a question
    # fails; there is no fallback to the other provider. Measured side by side in README.md
    # ("Hosted reranking") with eval/rerankers.py.
    rerank_provider: ClassVar[Literal["local", "openrouter"]] = "openrouter"
    openrouter_rerank_model: ClassVar[str] = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_timeout_seconds: float = 60.0

    # --- Local reranking (cross-encoder, no API quota) ---
    # Apache-2.0. Downloaded from the Hugging Face Hub on first use and cached under
    # HF_HOME; see "Reranker model" in README.md. The default is ~2.3 GB at fp32.
    #
    # Scored against eval/questions.yaml (20 questions with known articles/annexes, 20
    # candidates each, identical pools per model). 12 CPU threads, fp32, max_length 512:
    #
    #   model                                P@1    R@4    nDCG@4   ms/pair   20 candidates
    #   ms-marco-MiniLM-L-6-v2         23M   0.95   0.912  0.915        42       ~0.8 s
    #   bge-reranker-v2-m3            568M   0.95   0.921  0.922      1481        ~30 s
    #   bge-reranker-base             278M   0.95   0.846  0.864       442       ~8.8 s
    #
    # Those numbers make the models look interchangeable. They are NOT, and the benchmark
    # is what is misleading: it scored only `eu-ai-act` chunks, so recitals -- which are
    # short, quote-like, and lexically near-identical to a question -- never appeared as
    # distractors. In the real pool they dominate, and the two models handle them very
    # differently. For "Which AI practices are prohibited?" over the real 20 candidates:
    #
    #   MiniLM   1. Recital 45  2. Recital 28  3. Article 5  4. Article 5
    #   v2-m3    1. Article 5   2. Article 5   3. Recital 28 4. Recital 31
    #
    # MiniLM prefers the short recital that echoes the question's wording. Because
    # fit_token_budget then spends the evidence budget on those tiny units first, Article 5
    # (~3,000 tokens) no longer fits and is dropped -- the answer model is handed two
    # recitals saying prohibitions "should not be affected", cannot list a single
    # prohibited practice, and abstains. Three live tests fail exactly this way.
    #
    # So v2-m3 stays the default despite costing 35x the compute: it buys correctness on
    # the most basic question in the evaluation set, not a point of nDCG.
    # bge-reranker-base is dominated on every axis and is recorded only as measured.
    # Used when rerank_provider = "local". Set here only, like rerank_provider.
    rerank_model: ClassVar[str] = "BAAI/bge-reranker-v2-m3"
    # Cap on (query + passage) tokens per pair.
    #
    # v2-m3 accepts up to 8,194 positions, but the window is what costs: it measured 68s at
    # 1,536 against ~30s at 512 on the same 20 candidates. 512 buys back most of that.
    # Note this is a HARD CEILING for ms-marco-MiniLM-L-6-v2 -- that model is BERT-based
    # with max_position_embeddings=512, so raising this while it is selected will fail at
    # inference.
    #
    # About 27% of chunks are longer than 512 tokens and have their tail truncated *for
    # scoring only*. The answer model is still shown the full text of whatever survives, so
    # truncation here can cost ranking accuracy but can never make a quote unverifiable.
    rerank_max_length: int = 512
    rerank_batch_size: int = 8
    # Normalised (sigmoid) score below which a candidate is treated as irrelevant.
    # DELIBERATELY PERMISSIVE AND NOT YET CALIBRATED -- see the warning in
    # euaia.retrieval.rerank.rerank, which records the measurements showing that no single
    # absolute cutoff separates in-corpus from out-of-corpus questions on this corpus.
    # Correctness is protected downstream by citation verification and the coverage gate.
    rerank_min_score: float = 0.005

    # --- Token budgets ---
    # Reranking no longer appears here. It ran on Groq, where its prompt had to be capped at
    # 3,600 tokens to fit the minute budget (~7 of 20 candidates) and stalled the limiter for
    # ~58s per call. It is now a local cross-encoder scoring each pair separately, so it
    # consumes no tokens. Note the candidate cap cost little in practice -- see the measured
    # note in euaia.retrieval.rerank; the win here is quota and latency, not ranking.
    #
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
    # Answer rounds per question: the question itself, plus one follow-up the review may ask
    # for. Each round is a full answer call, so a second one waits out the minute budget.
    max_rounds: int = 2

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
    cellar_timeout_seconds: float = 60.0

    # --- Paths ---
    raw_data_dir: Path = REPO_ROOT / "data" / "raw"

    # --- Prompt versioning (recorded in query_log for auditability) ---
    # v2: fuller, plain-English answers, and follow-up questions rewritten to stand alone.
    # v3: authority tiers in the answer prompt; a greeting intent in the classifier.
    # v4: codes of practice -- context lines in evidence, measures named, parts covered.
    # v5: paths per question type (overview, comparison sides, legal tests), review, wrap-up.
    prompt_version: str = "v5"

    # --- Chat interface (python -m euaia.chat) ---
    # Who may sign in: comma-separated name:password pairs, e.g. "alice:s3cret,bob:hunter2".
    # Empty refuses every sign-in, for the same reason ADMIN_PASSWORD does. Chainlit also
    # needs CHAINLIT_AUTH_SECRET to sign its session tokens; see .env.example.
    chat_users: str = ""
    # Earlier turns shown to the follow-up rewriter. Each costs tokens on every follow-up.
    followup_turns: int = 3
    # Where the admin dashboard links to for the chat.
    chat_url: str = "http://127.0.0.1:8001"


settings = Settings()
