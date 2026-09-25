"""FastAPI application: the admin dashboard, and a JSON API.

The chat itself is a separate Chainlit app (``python -m euaia.chat``, port 8001). What stays
here is for operators and tooling: corpus status and change detection behind admin auth,
``/api/ask`` for the evaluation harness, and ``/healthz`` for Docker.

Server-rendered with Jinja and HTMX, with no build step and no Node dependency.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi import status as http_status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from euaia.api import monitoring, service
from euaia.config import settings
from euaia.db.session import DatabaseUnavailable, db_session
from euaia.ingest.pipeline import check_all
from euaia.ingest.sources import ALL_SOURCES

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


# The reranker is not warmed here, unlike in the chat: this app answers questions only on
# /api/ask, so its gigabytes of weights load on the first such call rather than into every
# dashboard process.
app = FastAPI(title="EU AI Act Assistant", version="0.1.0")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
templates.env.globals["chat_url"] = settings.chat_url


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


@app.get("/")
def index():
    """There is no public page here any more; the chat runs on its own port."""
    return RedirectResponse("/status")


@app.post("/api/ask")
def api_ask(payload: dict, session: Db):
    """JSON API, used by the evaluation harness."""
    question = (payload.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)

    return service.answer_payload(service.ask(question, session))


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
    ``check_all`` and recorded as a check outcome of ``error`` -- this except is only for
    something more fundamental (e.g. no network at all), and keeps the table on screen with
    an inline banner rather than replacing it with a bare error, which would also strand the
    retry button.
    """
    error = None
    try:
        check_all(session, list(ALL_SOURCES))
    except Exception as exc:  # noqa: BLE001
        log.exception("Change detection failed")
        error = str(exc)

    return templates.TemplateResponse(
        request=request,
        name="partials/corpus_status.html",
        context={"sources": service.corpus_status(session), "check_error": error},
    )


@app.get("/monitor", response_class=HTMLResponse, dependencies=[Admin])
def monitor(request: Request, days: int = 7):
    """How the assistant is doing: tokens, quality, latency, quotas. Requires admin auth."""
    return templates.TemplateResponse(
        request=request, name="monitor.html", context={"days": _period(days)}
    )


@app.get("/monitor/stats", response_class=HTMLResponse, dependencies=[Admin])
def monitor_stats(request: Request, session: Db, days: int = 7):
    """HTMX endpoint: the Monitor page's panels, re-rendered every few seconds."""
    return templates.TemplateResponse(
        request=request,
        name="partials/monitor_stats.html",
        context=monitoring.dashboard(session, _period(days)),
    )


def _period(days: int) -> int:
    return days if days in (1, 7, 30) else 7


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
