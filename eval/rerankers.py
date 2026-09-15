"""Compare rerankers on identical candidate pools drawn from the evaluation set.

    python -m eval.rerankers collect      analyse + retrieve every case once -> eval_pools.json
    python -m eval.rerankers score        score each pool with each reranker -> eval_rerank_scores.json
    python -m eval.rerankers report       ranking metrics per reranker

Rerankers are only comparable on the *same* candidates, so pools are collected once and
frozen. Collecting costs one small-model analysis call and one query embedding per case;
scoring spends no Groq or Gemini quota at all. Hosted scores are cached with the local ones,
so re-running ``report`` -- or ``score`` after adding a reranker -- never repeats a request.
That matters on OpenRouter's free tier, which allows 50 requests a day.

Relevance is judged from the evaluation set: a chunk is relevant when it belongs to one of
the case's ``expected_articles`` or ``expected_annexes``. Recitals are never counted as
relevant, and are reported separately, because crowding them into the top four is the
failure that made MiniLM abstain on "Which AI practices are prohibited?" (see
``rerank_model`` in ``euaia.config``).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from eval.harness import Case, load_cases

REPO_ROOT = Path(__file__).resolve().parent.parent
POOLS_PATH = REPO_ROOT / "eval_pools.json"
SCORES_PATH = REPO_ROOT / "eval_rerank_scores.json"

BASELINE = "local:BAAI/bge-reranker-v2-m3"
RERANKERS = [
    BASELINE,
    "local:BAAI/bge-reranker-base",
    "local:cross-encoder/ms-marco-MiniLM-L-6-v2",
    "openrouter:nvidia/llama-nemotron-rerank-vl-1b-v2:free",
]

_ARTICLE = re.compile(r"ART_([0-9]+[a-z]*)")
_ANNEX = re.compile(r"ANX_([IVXLC]+)")


# ------------------------------------------------------------------------ collect


def collect(cases: list[Case]) -> dict[str, Any]:
    """Run the pipeline's own analysis and retrieval for every case, and freeze the result."""
    from euaia.db.session import SessionLocal
    from euaia.graph.nodes import analyse, retrieve_evidence
    from euaia.graph.state import QueryState
    from euaia.ingest.embeddings import Embedder
    from euaia.llm.groq_client import GroqClient

    client, embedder = GroqClient(), Embedder()
    pools: dict[str, Any] = {}
    for case in cases:
        state = analyse(QueryState(question=case.question), client)
        pool: dict[str, Any] = {"intent": state.intent, "candidates": []}
        if state.intent != "out_of_scope":
            with SessionLocal() as session:
                retrieve_evidence(state, session, embedder)
            pool["candidates"] = [
                {
                    "chunk_id": c.chunk_id,
                    "unit_path": c.unit_path,
                    "source_key": c.source_key,
                    "chunk_text": c.chunk_text,
                }
                for c in state.candidates
            ]
        pools[case.id] = pool
        print(f"  {case.id:34s} {state.intent:14s} {len(pool['candidates']):3d} candidates")
    return pools


# ------------------------------------------------------------------------ score


def score_pool(name: str, question: str, passages: list[str]) -> list[float]:
    from euaia.retrieval import rerank as rr

    provider, model = name.split(":", 1)
    return rr.score_passages(question, passages, provider=provider, model=model)


def score(cases: list[Case], pools: dict[str, Any], names: list[str], scores: dict) -> None:
    """Score every pool that production would actually rerank, caching as it goes."""
    from euaia.config import settings

    for name in names:
        done = scores.setdefault(name, {})
        for case in cases:
            candidates = pools[case.id]["candidates"]
            if case.id in done or len(candidates) <= settings.rerank_keep:
                continue  # cached, or production skips reranking a pool this small
            started = time.perf_counter()
            values = score_pool(name, case.question, [c["chunk_text"] for c in candidates])
            done[case.id] = {
                "scores": values,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
            SCORES_PATH.write_text(json.dumps(scores, indent=1), encoding="utf-8")
            print(f"  {name:56s} {case.id:34s} {done[case.id]['latency_ms']:>7,} ms")


# ------------------------------------------------------------------------ report


def _provision(candidate: dict[str, Any]) -> str | None:
    if candidate["source_key"] != "eu-ai-act":
        return None
    if match := _ARTICLE.search(candidate["unit_path"]):
        return f"art:{match.group(1)}"
    if match := _ANNEX.search(candidate["unit_path"]):
        return f"anx:{match.group(1)}"
    return None


def case_metrics(case: Case, pool: dict, entry: dict, keep: int, min_score: float) -> dict:
    candidates = pool["candidates"]
    expected = {f"art:{a}" for a in case.expected_articles} | {
        f"anx:{a}" for a in case.expected_annexes
    }
    order = sorted(range(len(candidates)), key=lambda i: -entry["scores"][i])
    ranked = [candidates[i] for i in order]
    kept = [candidates[i] for i in order if entry["scores"][i] >= min_score][:keep]
    relevant = [_provision(c) in expected for c in ranked]

    dcg = sum(1 / math.log2(rank + 2) for rank, hit in enumerate(relevant[:keep]) if hit)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(keep, sum(relevant))))
    first_hit = next((rank for rank, hit in enumerate(relevant) if hit), None)
    in_pool = expected & {_provision(c) for c in candidates}
    return {
        "p_at_1": float(relevant[0]) if relevant else 0.0,
        "recall_at_k": len(expected & {_provision(c) for c in kept}) / len(expected) if expected else 0.0,
        "pool_recall": len(in_pool) / len(expected) if expected else 0.0,
        "ndcg_at_k": dcg / ideal if ideal else 0.0,
        "mrr": 1 / (first_hit + 1) if first_hit is not None else 0.0,
        "recitals_kept": sum(1 for c in kept if c["source_key"] == "eu-ai-act-recitals"),
        "best_score": max(entry["scores"]),
        "kept_ids": [c["chunk_id"] for c in kept],
    }


def report(cases: list[Case], pools: dict, scores: dict, names: list[str]) -> dict[str, Any]:
    from euaia.config import settings

    keep, min_score = settings.rerank_keep, settings.rerank_min_score
    summary: dict[str, Any] = {}
    for name in names:
        rows = {
            case.id: case_metrics(case, pools[case.id], scores[name][case.id], keep, min_score)
            for case in cases
            if case.id in scores.get(name, {})
        }
        answerable = [c for c in cases if c.id in rows and not c.should_abstain]
        refusable = [c for c in cases if c.id in rows and c.should_abstain]
        latencies = [scores[name][c.id]["latency_ms"] for c in cases if c.id in rows]
        base = scores.get(BASELINE, {})

        def mean(key: str, group: list[Case] = answerable, rows: dict = rows) -> float:
            return statistics.fmean(rows[c.id][key] for c in group) if group else 0.0

        summary[name] = {
            "pools_scored": len(rows),
            "p_at_1": mean("p_at_1"),
            f"recall_at_{keep}": mean("recall_at_k"),
            f"ndcg_at_{keep}": mean("ndcg_at_k"),
            "mrr": mean("mrr"),
            "pool_recall_ceiling": mean("pool_recall"),
            "recitals_in_evidence": mean("recitals_kept"),
            "missed_all_expected": [c.id for c in answerable if rows[c.id]["recall_at_k"] == 0],
            "below_min_score": [c.id for c in answerable if rows[c.id]["best_score"] < min_score],
            "lowest_answerable_best": min((rows[c.id]["best_score"] for c in answerable), default=0),
            "highest_abstain_best": max((rows[c.id]["best_score"] for c in refusable), default=0),
            "latency_p50_ms": int(statistics.median(latencies)) if latencies else 0,
            "latency_mean_ms": int(statistics.fmean(latencies)) if latencies else 0,
            "evidence_differs_from_baseline": [
                c.id
                for c in cases
                if c.id in rows and c.id in base
                and set(rows[c.id]["kept_ids"])
                != set(case_metrics(c, pools[c.id], base[c.id], keep, min_score)["kept_ids"])
            ],
            "per_case": rows,
        }
    return summary


def format_report(summary: dict[str, Any]) -> str:
    keys = [k for k in next(iter(summary.values())) if k not in ("per_case",)]
    lines = []
    for key in keys:
        cells = []
        for values in summary.values():
            value = values[key]
            if isinstance(value, list):
                cells.append(", ".join(value) or "-")
            elif isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(f"{value:,}")
        lines.append(f"{key:32s} | " + " | ".join(f"{cell:>24s}" for cell in cells))
    header = f"{'':32s} | " + " | ".join(f"{n.split('/')[-1][:24]:>24s}" for n in summary)
    return "\n".join([header, "-" * len(header), *lines])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare rerankers on frozen candidate pools.")
    parser.add_argument("command", choices=["collect", "score", "report"])
    parser.add_argument("--reranker", action="append", help="limit to these (default: all)")
    parser.add_argument("--json", type=Path, help="write the report here")
    args = parser.parse_args(argv)

    cases = load_cases()
    names = args.reranker or RERANKERS
    if args.command == "collect":
        POOLS_PATH.write_text(json.dumps(collect(cases), indent=1), encoding="utf-8")
        print(f"wrote {POOLS_PATH.name}")
        return 0

    pools = json.loads(POOLS_PATH.read_text(encoding="utf-8"))
    scores = json.loads(SCORES_PATH.read_text(encoding="utf-8")) if SCORES_PATH.exists() else {}
    if args.command == "score":
        score(cases, pools, names, scores)
        return 0

    summary = report(cases, pools, scores, [n for n in names if n in scores])
    print(format_report(summary))
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
