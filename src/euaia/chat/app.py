"""The chat interface, built on Chainlit. Start it with ``python -m euaia.chat``.

This module only connects Chainlit to the application: sign-in, saved conversations, and
running a question through :func:`euaia.api.service.ask`. What the answer looks like is
decided in :mod:`euaia.chat.views`.

The pipeline is synchronous and takes several seconds, so it runs in a worker thread. Each
step it reports is passed back to the event loop and shown live in a collapsible step above
the answer.
"""

from __future__ import annotations

import asyncio
import logging

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from sqlalchemy.engine import make_url

from euaia.api import service
from euaia.chat import views
from euaia.config import settings
from euaia.db.session import DatabaseUnavailable, db_session
from euaia.graph.state import Progress, Turn
from euaia.ingest.embeddings import EmbeddingError
from euaia.llm.groq_client import LLMError
from euaia.retrieval import rerank as reranker

log = logging.getLogger(__name__)

# Chainlit's tables (users, threads, steps, ...) live in their own schema, created by the
# chat_history migration, so they never mix with the corpus tables.
CHAT_SCHEMA = "chat"


@cl.on_app_startup
def load_reranker() -> None:
    """Load the reranker weights before the first question, not during it."""
    reranker.warm()


@cl.data_layer
def data_layer() -> SQLAlchemyDataLayer:
    """Saved conversations, in the same database as the corpus.

    Through asyncpg rather than the psycopg driver the rest of the application uses: this
    layer is async, and psycopg refuses to run async on the event loop Chainlit gets on
    Windows.
    """
    url = make_url(settings.database_url).set(drivername="postgresql+asyncpg")
    return SQLAlchemyDataLayer(
        conninfo=url.render_as_string(hide_password=False),
        connect_args={"server_settings": {"search_path": CHAT_SCHEMA}},
    )


@cl.password_auth_callback
def sign_in(username: str, password: str) -> cl.User | None:
    if views.check_login(username, password, settings.chat_users):
        return cl.User(identifier=username, metadata={"provider": "credentials"})
    return None


@cl.set_starters
async def starters(user: cl.User | None = None, language: str | None = None) -> list[cl.Starter]:
    return [
        cl.Starter(label=label, message=question) for label, question in views.EXAMPLE_QUESTIONS
    ]


@cl.on_chat_start
async def start() -> None:
    cl.user_session.set("history", [])


@cl.on_chat_resume
async def resume(thread: ThreadDict) -> None:
    cl.user_session.set("history", views.history_from_steps(thread.get("steps", [])))


@cl.on_message
async def answer(message: cl.Message) -> None:
    question = message.content.strip()
    if not question:
        await cl.Message(content="Please type a question about the EU AI Act.").send()
        return

    history: list[Turn] = cl.user_session.get("history") or []

    async with cl.Step(name=views.step_title(), type="tool") as step:
        try:
            view = await _ask_showing_progress(question, history, step)
        except (LLMError, EmbeddingError, DatabaseUnavailable) as exc:
            step.output = f"Stopped: {exc}"
            await cl.Message(content=f"**Something went wrong.** {exc}").send()
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("Unhandled error answering question")
            step.output = f"Stopped: {exc}"
            await cl.Message(content=f"**Unexpected error.** {exc}").send()
            return
        step.name = views.step_title(view)
        step.output = views.verification_markdown(view)

    await cl.Message(
        content=views.answer_markdown(view), metadata=views.message_metadata(view)
    ).send()

    history.append(Turn(view.question, views.recap(view)))
    cl.user_session.set("history", history[-settings.followup_turns :])


async def _ask_showing_progress(
    question: str, history: list[Turn], step: cl.Step
) -> service.AnswerView:
    """Run the pipeline in a thread, updating ``step`` as each stage starts."""
    loop = asyncio.get_running_loop()
    updates: asyncio.Queue[Progress] = asyncio.Queue()

    def report(progress: Progress) -> None:
        loop.call_soon_threadsafe(updates.put_nowait, progress)

    def run() -> service.AnswerView:
        with db_session() as session:
            return service.ask(question, session, history=history, on_progress=report)

    task = asyncio.create_task(asyncio.to_thread(run))
    seen: list[Progress] = []
    while True:
        finished = task.done()
        fresh = False
        while not updates.empty():
            seen.append(updates.get_nowait())
            fresh = True
        if fresh:
            step.output = views.progress_markdown(seen)
            await step.update()
        if finished:
            return task.result()
        await asyncio.sleep(0.2)
