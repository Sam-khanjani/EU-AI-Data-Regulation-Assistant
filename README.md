# Trustworthy EU AI Act Assistant

A RAG assistant that answers questions about the EU AI Act using official EUR-Lex sources,
where every claim in an answer is backed by a citation that's **mechanically verified**
against the source text — not just asked for in a prompt.

The goal isn't the chatbot itself. It's demonstrating that an LLM system can be made reliable
in a regulated domain: answers are grounded in authoritative text, the system abstains when
evidence is thin, every answer records which version of the law produced it, and the corpus
can be safely re-ingested when the regulation changes.

## How it works

1. **Ingest** — the EU AI Act is pulled from EUR-Lex (CELLAR API) as PDF and parsed into its
   actual legal structure (articles, paragraphs, annexes, recitals), chunked, and embedded.
   The consolidated text is read from the bookmark outline EUR-Lex authors inside the file,
   rather than by guessing headings from fonts. The recitals come from the as-adopted act,
   whose PDF has no such outline — they are recovered from the preamble instead, and the
   parse is only accepted if the recital numbering forms a complete 1..N run.
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
| Source | EUR-Lex CELLAR REST + SPARQL, PDF (bookmark outline + preamble) |
| Store | Postgres 17 + pgvector |
| Embeddings | Gemini `gemini-embedding-001` |
| Generation | Groq `openai/gpt-oss-120b`, structured output |
| Query analysis | Groq `openai/gpt-oss-20b` |
| Reranking | `BAAI/bge-reranker-v2-m3` cross-encoder, **run locally** |
| Orchestration | LangGraph |
| API / UI | FastAPI + Jinja2 + HTMX |

Both providers are used on free tiers, which are tight enough to shape the design directly —
chunking, caching, and prompt sizing are all built to fit inside them. Details on that, and on
why quote-matching is exact rather than fuzzy, are in the code's module docstrings
(`src/euaia/verify/citations.py`, `src/euaia/ingest/embedding_cache.py`).

Reranking runs locally rather than through an API. An LLM reranker scores every candidate
inside a single prompt, so its cost is the sum of all candidates — 20 chunks is ~9,000 tokens
against a free-tier ceiling of 8,000 per minute, which made a full candidate set impossible to
score and forced the rate limiter to idle ~58s before each call. A cross-encoder scores each
`(question, passage)` pair independently: no shared budget, no per-minute ceiling, no tokens.

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

## Running with Docker

The whole stack — database, migrations, web app — runs under Docker Compose, with no local
Python needed.

```bash
cp .env.example .env      # fill in GROQ_API_KEY, GOOGLE_API_KEY, POSTGRES_PASSWORD, ADMIN_PASSWORD
docker compose up -d --build
```

Then open <http://127.0.0.1:8000>. On first start the app downloads the reranker weights
(~2.3 GB) before it begins serving; follow it with `docker compose logs -f app`. Later starts
reuse them from a volume.

Ingest the corpus once (and again whenever you want to pick up a new consolidated version):

```bash
docker compose --profile tools run --rm ingest
docker compose --profile tools run --rm ingest --skip-embeddings   # structure only, no API key
```

| Service | Role | Lifetime |
|---|---|---|
| `db` | Postgres 17 + pgvector, published on `127.0.0.1:5433` | long-running |
| `migrate` | `alembic upgrade head` | runs once per `up`, then exits |
| `app` | web UI and API on `127.0.0.1:8000` | long-running, waits for `migrate` |
| `ingest` | fetch, parse, chunk, embed | on demand, `tools` profile only |

Data lives in three named volumes: `euaia-pgdata` (database), `euaia-models` (reranker
weights), `euaia-raw` (downloaded PDFs). `docker compose down` keeps them;
`docker compose down -v` deletes all three, including the embedded corpus.

Notes:

- **`DATABASE_URL` in `.env` is ignored inside containers.** It points at `127.0.0.1:5433`,
  the host side of the port mapping, which a container cannot reach. Compose builds the
  in-network URL (`db:5432`) from `POSTGRES_PASSWORD` instead, so that value must be URL-safe.
  Keep `DATABASE_URL` for running tools on the host.
- **`.env` never enters the image** — it is excluded by `.dockerignore` and supplied at run
  time.
- **The image uses CPU-only PyTorch.** On Linux, PyPI's torch pulls the full CUDA stack;
  `pyproject.toml` routes it to the CPU wheel index instead, keeping several GB of unused GPU
  libraries out of the image.
- Everything is published on `127.0.0.1` only. Nothing is reachable from other machines.

## Reranker model

The reranker runs locally. Its weights are **not in this repository** — they download from the
Hugging Face Hub the first time they are needed and are then cached on disk. Nothing here
needs a Hugging Face account or token; the default model is public and Apache-2.0.

| | |
|---|---|
| Model | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| Licence | Apache-2.0 (see `NOTICE.md`) |
| Download | ~2.3 GB (fp32), once |
| Cache | `$HF_HOME`, else `~/.cache/huggingface` (Windows: `C:\Users\<you>\.cache\huggingface`) |

### Pre-fetching

The first load is a multi-gigabyte download. Do it deliberately rather than discovering it on
a user's first question:

```bash
uv run python -c "from euaia.retrieval.rerank import load_model; load_model()"
```

The app also pre-loads the model at startup (`lifespan` in `api/main.py`), so `uvicorn` is
ready before it accepts traffic. If the download fails, startup still succeeds and the error
surfaces on the first question instead.

### Choosing a different model

Any `sentence-transformers` cross-encoder works. Set `RERANK_MODEL` (or `rerank_model`
in `.env`) — no code change:

```bash
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

**Scored against `eval/questions.yaml`.** All 29 questions, candidate pools built by
`hybrid.retrieve()` so they match what the pipeline actually reranks — recitals included.
Identical pools for every model. CPU only, 12 threads, fp32:

| Model | Params | Download | P@1 | R@4 | nDCG@4 | Evidence survival | 20 candidates |
|---|---|---|---|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` *(default)* | 568M | 2.3 GB | **0.450** | **0.633** | **0.560** | **90%** | ~30 s |
| `BAAI/bge-reranker-base` | 278M | 1.1 GB | 0.400 | 0.550 | 0.475 | 80% | ~8.8 s |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 23M | 92 MB | 0.350 | 0.583 | 0.500 | 75% | **~0.8 s** |

**Evidence survival** is the metric that matters, and it is not a ranking metric. After
reranking, survivors are collapsed to units and truncated to `evidence_token_budget`; a short
passage ranked highly can crowd out the article that actually answers. Evidence survival asks
whether the expected provision still reached the answering model.

Recitals are why the models differ. They are ~32% of real candidates, they are short, and they
echo a question's wording almost verbatim — so a similarity model ranks them above the articles
they merely describe. On *"Which AI practices are prohibited?"*:

```
MiniLM   1. Recital 45   2. Recital 28   3. Article 5   4. Article 5   -> evidence: 2 recitals
v2-m3    1. Article 5    2. Article 5    3. Recital 28  4. Recital 31  -> evidence: Article 5
```

Under MiniLM the two recitals consume the evidence budget, Article 5 (~3,000 tokens) no longer
fits, and the assistant abstains for want of any provision listing a prohibited practice.
`tests/test_pipeline_live.py` fails on exactly this, so **re-run it after changing
`RERANK_MODEL`** — ranking metrics alone will not catch this class of regression.

Note that scoring all 20 candidates is worth less than it looks: restricted to the top 8 by
retrieval rank, evidence survival is unchanged (90% for v2-m3). RRF ordering already puts the
answer near the top. The gains from running locally are zero token cost and no limiter stall,
not better ranking.

⚠️ `rerank_max_length` is 512. v2-m3 accepts up to 8,194, but the window is what costs
(68 s at 1,536 vs ~30 s at 512). It is a *hard* ceiling for MiniLM, which is BERT-based and
will fail at inference above 512.

**The score is not an abstention signal.** Across all 29 cases the score ranges of answerable
and `should_abstain` questions overlap completely — `nonexistent-article` scores 0.9945 under
MiniLM while the genuine `chatbot-applicability` question scores 0.0002. Rank survives where
score does not (those questions still have MRR 1.0). Use the reranker to order evidence, never
to decide whether to answer; citation verification and the coverage gate do that.

### Offline / air-gapped

Pre-fetch on a connected machine, copy the cache directory across, and pin it:

```bash
export HF_HOME=/path/to/cache
export HF_HUB_OFFLINE=1
```

`.gitignore` already excludes `*.safetensors`, `models/` and `.cache/`, so a cache placed
inside the project cannot be committed by accident.

## Testing and evaluation

```bash
uv run pytest -q                # fast, offline
uv run python -m eval.harness   # scored against a seeded question set
```

## Layout

```
src/euaia/
  ingest/     CELLAR client, PDF readers, chunker, embedder, pipeline
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
