# Trustworthy EU AI Act Assistant

A RAG assistant that answers questions about the EU AI Act using official EUR-Lex sources,
where every claim in an answer is backed by a citation that's **mechanically verified**
against the source text — not just asked for in a prompt.

The goal isn't the chatbot itself. It's demonstrating that an LLM system can be made reliable
in a regulated domain: answers are grounded in authoritative text, the system abstains when
evidence is thin, every answer records which version of the law produced it, and the corpus
can be safely re-ingested when the regulation changes.

## How it works

1. **Ingest** — the EU AI Act is pulled from EUR-Lex (CELLAR API, Formex XML), parsed into
   its actual legal structure (articles, paragraphs, annexes, recitals), chunked, and
   embedded.
2. **Retrieve** — a hybrid search (vector similarity + full-text + direct article lookup)
   finds candidate provisions, which are reranked for relevance.
3. **Generate** — the answering model is required to return each claim together with the
   exact quotes it's based on, via structured output.
4. **Verify** — every quote is checked to appear **verbatim** in the source provision it
   claims to come from. Quotes that don't match are dropped; claims with no surviving quote
   are never shown. If too few claims survive, the system abstains instead of guessing.

For "is my system high-risk?"-type questions, the assistant never states a verdict — it
returns a checklist of the relevant criteria, cited and quoted, with the questions needed to
resolve it. The determination depends on facts only the user has.

## Stack

| Layer | Choice |
|---|---|
| Source | EUR-Lex CELLAR REST + SPARQL, Formex XML |
| Store | Postgres 17 + pgvector |
| Embeddings | Gemini `gemini-embedding-001` |
| Generation | Groq `openai/gpt-oss-120b`, structured output |
| Analysis / rerank | Groq `openai/gpt-oss-20b` |
| Orchestration | LangGraph |
| API / UI | FastAPI + Jinja2 + HTMX |

Both providers are used on free tiers, which are tight enough to shape the design directly —
chunking, caching, and prompt sizing are all built to fit inside them. Details on that, and on
why quote-matching is exact rather than fuzzy, are in the code's module docstrings
(`src/euaia/verify/citations.py`, `src/euaia/ingest/embedding_cache.py`).

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
cp .env.example .env      # then fill in GROQ_API_KEY and GOOGLE_API_KEY
uv sync --extra dev
docker compose up -d db
uv run alembic upgrade head
```

Ingest the corpus (add `--skip-embeddings` to parse and store structure without an API key):

```bash
uv run python -m euaia.ingest.pipeline
```

This registers two sources: `eu-ai-act`, the latest consolidated act (operative text with
amendments applied), and `eu-ai-act-recitals`, the act as adopted (for its recitals, which
consolidation doesn't restate).

Run the app, then open <http://localhost:8000>:

```bash
uv run uvicorn euaia.api.main:app --reload
```

## Testing and evaluation

```bash
uv run pytest -q                # fast, offline
uv run python -m eval.harness   # scored against a seeded question set
```

## Layout

```
src/euaia/
  ingest/     CELLAR client, Formex parser, chunker, embedder, pipeline
  retrieval/  hybrid dense + full-text + structural lookup, reranker
  graph/      pipeline nodes, state, prompts
  llm/        Groq client and structured-output schemas
  verify/     text normalisation and citation verification
  api/ web/   FastAPI routes and server-rendered UI
eval/         question set, scoring harness
```

## Versioning

Re-ingesting never overwrites a prior version — a new `document_version` row is written and
the old one is marked superseded, with at most one active version per source at any time.
Every answer records which document version it was generated from. `/status` shows what's
currently indexed.

## Status

Milestone 1 (this vertical slice) is complete and running end-to-end against the live corpus.
Next up: surfacing change detection on `/status` (polling EUR-Lex for a newer consolidated
version — the underlying lookup already exists), and closing gaps found in evaluation around
citation recall and applicability-question handling.
