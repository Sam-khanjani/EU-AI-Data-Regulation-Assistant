"""The numbers behind the admin dashboard's Monitor page, read from ``query_log``.

Built from the audit rows the app already writes, not from Langfuse: the dashboard works
with tracing off, and Langfuse is where one question is taken apart (each row links to its
trace). Computed in Python over the period's rows -- a few thousand at most on a free tier.
"""

from __future__ import annotations

import datetime as dt
import statistics
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from euaia import observability
from euaia.config import settings
from euaia.db.models import EvalRun, QueryLog

OPENROUTER_FREE_PER_DAY = 50
"""OpenRouter's free models share 50 requests a day; every search is one rerank request."""
NO_SEARCH = ("greeting", "out_of_scope")


def dashboard(session: Session, days: int) -> dict[str, Any]:
    now = dt.datetime.now(dt.UTC)
    rows = session.scalars(
        select(QueryLog)
        .where(QueryLog.asked_at >= now - dt.timedelta(days=days))
        .order_by(QueryLog.asked_at)
    ).all()
    today = [r for r in rows if r.asked_at.date() == now.date()]
    return {
        "days": days,
        "kpis": _kpis(rows),
        "quotas": _quotas(today),
        "daily": _daily(rows),
        "steps": _steps(rows),
        "intents": _intents(rows),
        "evals": _evals(session),
        "recent": [_recent(r) for r in reversed(rows[-25:])],
        "langfuse": settings.langfuse_enabled and settings.langfuse_public_url,
    }


def _run(row: QueryLog) -> dict[str, Any]:
    return row.run or {}


def _tokens(row: QueryLog) -> tuple[int, int]:
    counts = _run(row).get("tokens", {}).values()
    return sum(c["input"] for c in counts), sum(c["output"] for c in counts)


def _pct(values: list[float], q: float) -> float:
    return sorted(values)[max(0, int(len(values) * q) - 1)] if values else 0


def _kpis(rows: list[QueryLog]) -> dict[str, Any]:
    searched = [r for r in rows if r.intent not in NO_SEARCH]
    answered = [r for r in searched if r.verdict in ("answered", "partial")]
    quotes = sum(r.quotes_total or 0 for r in rows)
    dropped = sum(r.quotes_dropped or 0 for r in rows)
    latencies = [r.latency_ms for r in searched if r.latency_ms]
    waits = [_run(r).get("waited_ms", 0) for r in searched]
    tokens = [_tokens(r) for r in rows if r.run]
    reviewed = [r for r in searched if _run(r).get("rounds")]
    return {
        "questions": len(rows),
        "small_talk": len(rows) - len(searched),
        "answer_rate": len(answered) / len(searched) if searched else None,
        "partial": sum(1 for r in searched if r.verdict == "partial"),
        "quote_accuracy": (quotes - dropped) / quotes if quotes else None,
        "coverage": statistics.fmean(c) if (c := [r.citation_coverage for r in answered
                                                   if r.citation_coverage is not None]) else None,
        "latency_p50": _pct(latencies, 0.5),
        "latency_p95": _pct(latencies, 0.95),
        "wait_mean": int(statistics.fmean(waits)) if waits else 0,
        "tokens_in": int(statistics.fmean(t[0] for t in tokens)) if tokens else 0,
        "tokens_out": int(statistics.fmean(t[1] for t in tokens)) if tokens else 0,
        "second_round": (
            sum(1 for r in reviewed if _run(r)["rounds"] > 1) / len(reviewed) if reviewed else None
        ),
        "repaired": sum(1 for r in rows if r.quotes_repaired),
    }


def _quotas(today: list[QueryLog]) -> list[dict[str, Any]]:
    """What today has spent of each free-tier allowance: what runs out first."""
    per_model: Counter[str] = Counter()
    for row in today:
        for model, c in _run(row).get("tokens", {}).items():
            per_model[model] += c["input"] + c["output"]
    quotas = [
        {"name": f"Groq {model.split('/')[-1]} tokens", "used": used, "limit": settings.groq_tpd}
        for model, used in sorted(per_model.items())
    ]
    if settings.rerank_provider == "openrouter":
        searches = sum(_run(r).get("searches", 0) for r in today)
        quotas.append(
            {"name": "OpenRouter rerank requests", "used": searches,
             "limit": OPENROUTER_FREE_PER_DAY}
        )
    for quota in quotas:
        quota["share"] = min(1.0, quota["used"] / quota["limit"])
    return quotas


def _daily(rows: list[QueryLog]) -> dict[str, Any]:
    """Per-day series for the charts."""
    days: dict[str, list[QueryLog]] = defaultdict(list)
    for row in rows:
        days[row.asked_at.date().isoformat()].append(row)
    labels = sorted(days)
    models = sorted({m for r in rows for m in _run(r).get("tokens", {})})

    def per_day(fn):
        return [fn(days[day]) for day in labels]

    def quote_acc(day):
        total = sum(r.quotes_total or 0 for r in day)
        dropped = sum(r.quotes_dropped or 0 for r in day)
        return round((total - dropped) / total, 3) if total else None

    def latencies(day, q):
        values = [r.latency_ms / 1000 for r in day if r.latency_ms and r.intent not in NO_SEARCH]
        return round(_pct(values, q), 1) if values else None

    return {
        "labels": labels,
        "verdicts": {
            v: per_day(lambda d, v=v: sum(1 for r in d if r.verdict == v))
            for v in ("answered", "partial", "abstained")
        },
        "tokens": {
            f"{m.split('/')[-1]} {kind}": per_day(
                lambda d, m=m, kind=kind: sum(
                    _run(r).get("tokens", {}).get(m, {}).get(kind, 0) for r in d
                )
            )
            for m in models for kind in ("input", "output")
        },
        "quote_accuracy": per_day(quote_acc),
        "coverage": per_day(lambda d: round(statistics.fmean(c), 3) if (
            c := [r.citation_coverage for r in d if r.citation_coverage is not None]) else None),
        "latency_p50": per_day(lambda d: latencies(d, 0.5)),
        "latency_p95": per_day(lambda d: latencies(d, 0.95)),
        "waiting": per_day(lambda d: round(statistics.fmean(
            [_run(r).get("waited_ms", 0) / 1000 for r in d]), 1) if d else None),
    }


def _steps(rows: list[QueryLog]) -> list[dict[str, Any]]:
    """Mean time per graph node, slowest first: where the seconds go."""
    times: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        for step in _run(row).get("steps", []):
            times[step["node"]].append(step["ms"])
    means = [
        {"node": node, "ms": int(statistics.fmean(ms)), "runs": len(ms)}
        for node, ms in times.items()
    ]
    return sorted(means, key=lambda s: s["ms"], reverse=True)


def _intents(rows: list[QueryLog]) -> list[dict[str, Any]]:
    """Each kind of question: how many, and how they ended."""
    by_intent: dict[str, list[QueryLog]] = defaultdict(list)
    for row in rows:
        by_intent[row.intent or "unknown"].append(row)
    out = []
    for intent, group in sorted(by_intent.items(), key=lambda kv: -len(kv[1])):
        latencies = [r.latency_ms / 1000 for r in group if r.latency_ms]
        out.append({
            "intent": intent,
            "count": len(group),
            "verdicts": Counter(r.verdict for r in group),
            "rounds": round(statistics.fmean([_run(r).get("rounds", 0) for r in group]), 2),
            "latency": round(statistics.median(latencies), 1) if latencies else None,
        })
    return out


def _evals(session: Session) -> list[dict[str, Any]]:
    """The latest evaluation runs: accuracy against known answers, by prompt version."""
    runs = session.scalars(select(EvalRun).order_by(EvalRun.ran_at.desc()).limit(10)).all()
    return [
        {"ran_at": r.ran_at, "prompt_version": r.prompt_version, "cases": r.cases, **r.summary}
        for r in runs
    ]


def _recent(row: QueryLog) -> dict[str, Any]:
    run = _run(row)
    tokens_in, tokens_out = _tokens(row)
    return {
        "id": row.id,
        "asked_at": row.asked_at,
        "question": row.question,
        "intent": row.intent,
        "verdict": row.verdict,
        "rounds": run.get("rounds"),
        "quotes": f"{(row.quotes_total or 0) - (row.quotes_dropped or 0)}/{row.quotes_total or 0}",
        "tokens": f"{tokens_in:,} / {tokens_out:,}" if run else "",
        "latency": (row.latency_ms or 0) / 1000,
        "waited": run.get("waited_ms", 0) / 1000,
        "path": " → ".join(
            "search" if s["node"] == "rerank_evidence" else s["node"]
            for s in run.get("steps", []) if s["node"] not in _QUIET_NODES
        ),
        "trace_url": observability.trace_url(run.get("trace_id")),
    }


_QUIET_NODES = ("embed_query", "retrieve_evidence", "expand_evidence")
"""Inside every search; the path shows each search once, as "search"."""
