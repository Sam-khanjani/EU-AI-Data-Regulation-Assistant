"""FastAPI application: chat UI, corpus status, and a JSON API.

Server-rendered with Jinja and HTMX. There is no build step and no Node dependency, which
suits a project whose interesting parts are ingestion and verification rather than the
front end.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi import status as http_status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from euaia.api import service
from euaia.config import settings
from euaia.db.session import DatabaseUnavailable, db_session
from euaia.ingest.embedder import EmbeddingError
from euaia.llm.groq_client import LLMError

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="EU AI Act Assistant", version="0.1.0")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))


@app.exception_handler(DatabaseUnavailable)
async def database_unavailable_handler(request: Request, exc: DatabaseUnavailable):
    """Show a clear message instead of a connection-pool stack trace.

    Also catches the raw sqlalchemy.exc.OperationalError case: db_session() converts it
    to DatabaseUnavailable, but that conversion only happens where a session is actually
    opened -- this handler is the safety net for the response either way.
    """
    return HTMLResponse(f"<h1>503 Service Unavailable</h1><p>{exc}</p>", status_code=503)


def get_session() -> Iterator[Session]:
    with db_session() as session:
        yield session


# FastAPI's modern dependency style: keeps Depends() out of argument defaults.
Db = Annotated[Session, Depends(get_session)]

_admin_security = HTTPBasic()


def require_admin(credentials: Annotated[HTTPBasicCredentials, Depends(_admin_security)]) -> None:
    """Gate admin-only routes behind HTTP Basic Auth.

    An unset ADMIN_PASSWORD must refuse every request, not just wrong ones --
    ``secrets.compare_digest("", "")`` is True, so a blank configured password would
    otherwise let anyone in with a blank password field. ``compare_digest`` throughout
    rather than ``==`` so a wrong guess cannot be timed to find out which character failed.
    """
    valid_username = secrets.compare_digest(
        credentials.username.encode("utf-8"), settings.admin_username.encode("utf-8")
    )
    valid_password = bool(settings.admin_password) and secrets.compare_digest(
        credentials.password.encode("utf-8"), settings.admin_password.encode("utf-8")
    )
    if not (valid_username and valid_password):
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )


Admin = Depends(require_admin)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, session: Db):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "sources": service.corpus_status(session),
            "config_ok": bool(settings.groq_api_key and settings.google_api_key),
            "examples": EXAMPLE_QUESTIONS,
        },
    )


@app.post("/ask", response_class=HTMLResponse)
def ask(
    request: Request,
    session: Db,
    question: Annotated[str, Form()],
):
    """HTMX endpoint: returns the answer fragment."""
    question = question.strip()
    if not question:
        return templates.TemplateResponse(
            request=request, name="partials/error.html",
            context={"message": "Please enter a question."},
        )

    try:
        view = service.ask(question, session)
    except (LLMError, EmbeddingError) as exc:
        return templates.TemplateResponse(
            request=request, name="partials/error.html", context={"message": str(exc)}
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Unhandled error answering question")
        return templates.TemplateResponse(
            request=request, name="partials/error.html",
            context={"message": f"Unexpected error: {exc}"},
        )

    return templates.TemplateResponse(
        request=request, name="partials/answer.html", context={"a": view}
    )


@app.post("/api/ask")
def api_ask(payload: dict, session: Db):
    """JSON API, used by the evaluation harness."""
    question = (payload.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)

    view = service.ask(question, session)
    return {
        "question": view.question,
        "verdict": view.verdict,
        "summary": view.summary,
        "claims": [
            {
                "text": claim.text,
                "citations": [
                    {
                        "citation": c.citation_label,
                        "quote": c.quote,
                        "url": c.deeplink,
                        "version": c.version_label,
                    }
                    for c in claim.citations
                ],
            }
            for claim in view.claims
        ],
        "criteria": [
            {
                "criterion": c["criterion"],
                "status": c["status"],
                "explanation": c["explanation"],
                "citations": [
                    {"citation": q.citation_label, "quote": q.quote, "url": q.deeplink}
                    for q in c["citations"]
                ],
            }
            for c in view.criteria
        ],
        "abstain_reason": view.abstain_reason,
        "unanswered_aspects": view.unanswered_aspects,
        "follow_up_questions": view.follow_up_questions,
        "coverage": view.coverage,
        "quotes_total": view.quotes_total,
        "quotes_dropped": view.quotes_dropped,
        "latency_ms": view.latency_ms,
        "tokens": view.tokens,
        "waited_ms": view.waited_ms,
        "sources": view.sources,
        "query_log_id": view.query_log_id,
    }


@app.get("/status", response_class=HTMLResponse, dependencies=[Admin])
def status(request: Request, session: Db):
    """Corpus status and change detection -- the admin dashboard. Requires admin auth."""
    return templates.TemplateResponse(
        request=request,
        name="status.html",
        context={"sources": service.corpus_status(session)},
    )


@app.post("/status/check", response_class=HTMLResponse, dependencies=[Admin])
def status_check(request: Request, session: Db):
    """HTMX endpoint: poll CELLAR for each source and re-render the corpus table. Requires
    admin auth.

    Per-source failures (CELLAR unreachable, SPARQL timeout) are already caught inside
    ``check_now`` and recorded as a check outcome of ``error`` -- this except is only for
    something more fundamental (e.g. no network at all), and keeps the table on screen with
    an inline banner rather than replacing it with a bare error, which would also strand the
    retry button.
    """
    error = None
    try:
        service.check_now(session)
    except Exception as exc:  # noqa: BLE001
        log.exception("Change detection failed")
        error = str(exc)

    return templates.TemplateResponse(
        request=request,
        name="partials/corpus_status.html",
        context={"sources": service.corpus_status(session), "check_error": error},
    )


@app.get("/healthz")
def healthz(session: Db):
    sources = service.corpus_status(session)
    return {
        "ok": True,
        "sources": len(sources),
        "chunks": sum(s["chunks"] for s in sources),
        "groq_key": bool(settings.groq_api_key),
        "google_key": bool(settings.google_api_key),
    }


EXAMPLE_QUESTIONS = [
    "Which AI practices are prohibited?",
    "What requirements apply to high-risk AI systems?",
    "What must an AI system tell users under Article 50?",
    "What are the obligations of a deployer of a high-risk AI system?",
    "Is my CV-screening tool a high-risk AI system?",
]
