"""Evaluation harness: run the question set and measure reliability.

The metrics are chosen to make the system's central claim falsifiable rather than
flattering.

``citation_validity`` is 100% by construction -- unverified quotes are dropped before
display -- so reporting it alone would be self-congratulatory. The metric that carries
information is ``quote_accuracy``: of the quotes the model *tried* to use, how many actually
appeared in the source. That measures the model, not the safety net.

``abstention_accuracy`` is weighted equally with answer quality. A system that answers
everything scores zero here, which is the intended pressure: in a regulated domain, refusing
is a correct output.

**Free-tier budget.** Groq allows 200,000 tokens per day. A full pass over this question set
costs roughly 130,000 of them, so it fits once per day and not much more. The harness reports
token spend, and `--budget` stops the run before it blows the daily allowance rather than
collecting a wall of 429s halfway through.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.yaml"


@dataclass(slots=True)
class Case:
    id: str
    category: str
    question: str
    expected_articles: list[str] = field(default_factory=list)
    expected_annexes: list[str] = field(default_factory=list)
    should_abstain: bool = False
    expect_criteria: bool = False
    notes: str = ""


@dataclass(slots=True)
class CaseResult:
    case: Case
    verdict: str
    answered: bool
    abstained: bool
    abstention_correct: bool
    cited_articles: set[str] = field(default_factory=set)
    cited_annexes: set[str] = field(default_factory=set)
    recall: float | None = None
    quotes_total: int = 0
    quotes_dropped: int = 0
    coverage: float = 0.0
    produced_criteria: bool = False
    criteria_verdict_leak: bool = False
    latency_ms: int = 0
    tokens: int = 0
    waited_ms: int = 0
    error: str | None = None

    @property
    def quote_accuracy(self) -> float | None:
        if self.quotes_total == 0:
            return None
        return (self.quotes_total - self.quotes_dropped) / self.quotes_total


def load_cases(path: Path = QUESTIONS_PATH) -> list[Case]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Case(
            id=item["id"],
            category=item["category"],
            question=item["question"].strip(),
            expected_articles=[str(a) for a in item.get("expected_articles", [])],
            expected_annexes=[str(a) for a in item.get("expected_annexes", [])],
            should_abstain=bool(item.get("should_abstain", False)),
            expect_criteria=bool(item.get("expect_criteria", False)),
            notes=item.get("notes", ""),
        )
        for item in raw
    ]


# Phrases that would amount to deciding the user's legal position for them. An
# applicability answer containing one has broken the design rule, however well cited.
_VERDICT_LEAKS = (
    "your system is high-risk",
    "your system is a high-risk",
    "is a high-risk ai system",
    "your tool is high-risk",
    "you are a provider",
    "you must comply",
    "this is prohibited",
    "your system is not high-risk",
    "is not high-risk",
    "you do not need to",
)


def evaluate_response(case: Case, payload: dict[str, Any]) -> CaseResult:
    """Score one answer. Pure function over the JSON API response, so it is testable."""
    verdict = payload.get("verdict", "abstained")
    abstained = verdict == "abstained"
    answered = not abstained

    citations = [
        c
        for claim in payload.get("claims", [])
        for c in claim.get("citations", [])
    ] + [
        c
        for crit in payload.get("criteria", [])
        for c in crit.get("citations", [])
    ]

    cited_articles = set()
    cited_annexes = set()
    for citation in citations:
        label = (citation.get("citation") or "").strip()
        if label.startswith("Article "):
            cited_articles.add(label.removeprefix("Article ").strip())
        elif label.startswith("Annex "):
            cited_annexes.add(label.removeprefix("Annex ").strip())
        elif label.startswith("Paragraph "):
            number = label.removeprefix("Paragraph ").strip()
            cited_articles.add(number.split("(")[0])

    recall: float | None = None
    expected = set(case.expected_articles) | {f"anx:{a}" for a in case.expected_annexes}
    if expected and not case.should_abstain:
        found = cited_articles | {f"anx:{a}" for a in cited_annexes}
        recall = len(expected & found) / len(expected)

    criteria = payload.get("criteria", [])
    blob = " ".join(
        [payload.get("summary", "")]
        + [c.get("text", "") for c in payload.get("claims", [])]
        + [c.get("explanation", "") for c in criteria]
    ).lower()
    leak = case.expect_criteria and any(phrase in blob for phrase in _VERDICT_LEAKS)

    return CaseResult(
        case=case,
        verdict=verdict,
        answered=answered,
        abstained=abstained,
        abstention_correct=(abstained == case.should_abstain),
        cited_articles=cited_articles,
        cited_annexes=cited_annexes,
        recall=recall,
        quotes_total=payload.get("quotes_total", 0),
        quotes_dropped=payload.get("quotes_dropped", 0),
        coverage=payload.get("coverage", 0.0),
        tokens=payload.get("tokens", 0),
        waited_ms=payload.get("waited_ms", 0),
        produced_criteria=bool(criteria),
        criteria_verdict_leak=leak,
        latency_ms=payload.get("latency_ms", 0),
    )


def summarise(results: list[CaseResult]) -> dict[str, Any]:
    ok = [r for r in results if r.error is None]
    answerable = [r for r in ok if not r.case.should_abstain]
    refusable = [r for r in ok if r.case.should_abstain]
    with_recall = [r for r in answerable if r.recall is not None]
    with_quotes = [r for r in ok if r.quote_accuracy is not None]
    assessments = [r for r in ok if r.case.expect_criteria]
    latencies = [r.latency_ms for r in ok if r.latency_ms]

    def mean(values):
        return statistics.fmean(values) if values else 0.0

    return {
        "cases": len(results),
        "errors": sum(1 for r in results if r.error),
        "answer_rate_on_answerable": mean([1.0 if r.answered else 0.0 for r in answerable]),
        "abstention_accuracy": mean([1.0 if r.abstention_correct else 0.0 for r in ok]),
        "correct_refusals": f"{sum(1 for r in refusable if r.abstained)}/{len(refusable)}",
        "false_answers": [r.case.id for r in refusable if r.answered],
        "citation_recall": mean([r.recall for r in with_recall]),
        "quote_accuracy": mean([r.quote_accuracy for r in with_quotes]),
        "quotes_total": sum(r.quotes_total for r in ok),
        "quotes_dropped": sum(r.quotes_dropped for r in ok),
        "mean_coverage": mean([r.coverage for r in ok if r.answered]),
        "criteria_produced": f"{sum(1 for r in assessments if r.produced_criteria)}/{len(assessments)}",
        "verdict_leaks": [r.case.id for r in assessments if r.criteria_verdict_leak],
        "tokens_spent": sum(r.tokens for r in ok),
        "tokens_per_case": int(mean([r.tokens for r in ok])) if ok else 0,
        "rate_limit_wait_s": int(sum(r.waited_ms for r in ok) / 1000),
        "latency_p50_ms": int(statistics.median(latencies)) if latencies else 0,
        "latency_p95_ms": (
            int(sorted(latencies)[int(len(latencies) * 0.95) - 1]) if len(latencies) >= 2 else 0
        ),
    }


def format_report(results: list[CaseResult], summary: dict[str, Any]) -> str:
    lines = ["", "=" * 78, "EVALUATION", "=" * 78, ""]
    lines.append(f"{'id':32s} {'category':14s} {'verdict':10s} {'recall':>7s} {'quotes':>10s}")
    lines.append("-" * 78)
    for r in results:
        if r.error:
            lines.append(f"{r.case.id:32s} {r.case.category:14s} ERROR: {r.error[:30]}")
            continue
        mark = " " if r.abstention_correct else "!"
        recall = "-" if r.recall is None else f"{r.recall:.0%}"
        quotes = f"{r.quotes_total - r.quotes_dropped}/{r.quotes_total}"
        lines.append(
            f"{mark}{r.case.id:31s} {r.case.category:14s} {r.verdict:10s} {recall:>7s} {quotes:>10s}"
        )

    lines += ["", "-" * 78, "SUMMARY", "-" * 78]
    for key, value in summary.items():
        if isinstance(value, float):
            value = f"{value:.1%}" if value <= 1.0 else f"{value:.2f}"
        lines.append(f"  {key:32s} {value}")
    lines.append("")
    return "\n".join(lines)


def run(
    base_url: str | None,
    cases: list[Case],
    verbose: bool = False,
    budget: int | None = None,
) -> list[CaseResult]:
    """Run every case, either in-process or against a running server.

    ``budget`` caps total tokens. Stopping early with partial results beats exhausting the
    daily free-tier allowance and reporting a page of rate-limit errors as if they were
    reliability failures.
    """
    results: list[CaseResult] = []
    spent = 0

    if base_url:
        import httpx

        client = httpx.Client(timeout=180.0)

        def answer(question: str) -> dict[str, Any]:
            resp = client.post(f"{base_url.rstrip('/')}/api/ask", json={"question": question})
            resp.raise_for_status()
            return resp.json()
    else:
        from euaia.api import service
        from euaia.db.session import SessionLocal
        from euaia.ingest.embeddings import Embedder
        from euaia.llm.groq_client import GroqClient

        client_llm = GroqClient()
        embedder = Embedder()

        def answer(question: str) -> dict[str, Any]:
            session = SessionLocal()
            try:
                view = service.ask(question, session, client_llm, embedder)
                return service.answer_payload(view)
            finally:
                session.close()

    for case in cases:
        if budget is not None and spent >= budget:
            print(
                f"  stopping: token budget {budget:,} reached after {len(results)} cases",
                file=sys.stderr,
            )
            break
        started = time.perf_counter()
        try:
            payload = answer(case.question)
            result = evaluate_response(case, payload)
        except Exception as exc:  # noqa: BLE001
            result = CaseResult(
                case=case, verdict="error", answered=False, abstained=False,
                abstention_correct=False, error=f"{type(exc).__name__}: {exc}",
            )
        if not result.latency_ms:
            result.latency_ms = int((time.perf_counter() - started) * 1000)
        spent += result.tokens
        results.append(result)
        if verbose:
            print(
                f"  {case.id:32s} -> {result.verdict:10s} {result.tokens:>6,} tok "
                f"(total {spent:,})",
                file=sys.stderr,
            )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the EU AI Act assistant.")
    parser.add_argument("--base-url", help="evaluate a running server instead of in-process")
    parser.add_argument("--only", action="append", help="run only these case ids")
    parser.add_argument("--category", action="append", help="run only these categories")
    parser.add_argument("--json", type=Path, help="write the full result set here")
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="stop once this many tokens have been spent (Groq free tier: 200000/day)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    cases = load_cases()
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
    if args.category:
        cases = [c for c in cases if c.category in set(args.category)]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2

    results = run(args.base_url, cases, verbose=args.verbose, budget=args.budget)
    summary = summarise(results)
    print(format_report(results, summary))

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "cases": [
                        {
                            "id": r.case.id,
                            "category": r.case.category,
                            "verdict": r.verdict,
                            "abstention_correct": r.abstention_correct,
                            "recall": r.recall,
                            "quote_accuracy": r.quote_accuracy,
                            "cited_articles": sorted(r.cited_articles),
                            "cited_annexes": sorted(r.cited_annexes),
                            "latency_ms": r.latency_ms,
                            "tokens": r.tokens,
                            "error": r.error,
                        }
                        for r in results
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
