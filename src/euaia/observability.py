"""Tracing to Langfuse: one trace per question, the graph's nodes through Langfuse's official
LangChain/LangGraph callback handler, a generation per model call (prompt, answer, input and
output tokens), and the answer's quality as scores.

Off unless ``LANGFUSE_ENABLED=true``; everything here is then a no-op. When on, it only
sends: spans leave in a background thread, so a Langfuse that is down costs a log line,
never an answer. Run one locally with ``docker compose --profile monitoring up -d``.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from typing import Any

from euaia.config import settings

log = logging.getLogger(__name__)

PROJECT_ID = "euaia"
"""Created by docker-compose.yml on Langfuse's first start (LANGFUSE_INIT_PROJECT_ID)."""


@functools.cache
def client():
    """The Langfuse client, or None when tracing is off."""
    if not settings.langfuse_enabled:
        return None
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        mask=_mask,
    )


def graph_callbacks() -> list:
    """Langfuse's LangGraph handler, for the graph run's config; none when tracing is off.

    It records every node, subgraph and parallel branch with its input and output state,
    under whatever trace is current -- the question's.
    """
    if client() is None:
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler(public_key=settings.langfuse_public_key)]


_PROMPT_CHARS = 20_000
"""Prompts and answers stay readable in full: they are what a trace is opened for."""
_STATE_CHARS, _STATE_ITEMS, _NUMBERS = 300, 12, 20


def _mask(*, data: Any, _state: bool = False) -> Any:
    """Keep traces small: the handler records each node's whole state, which carries a
    3,072-number query embedding and the full text of every passage searched. Anything
    inside a state object is cut to a glimpse; prompts and answers are left readable."""
    if is_dataclass(data) and not isinstance(data, type):
        data, _state = {f.name: getattr(data, f.name) for f in fields(data)}, True
    if isinstance(data, str):
        limit = _STATE_CHARS if _state else _PROMPT_CHARS
        return data if len(data) <= limit else f"{data[:limit]}... [{len(data):,} chars]"
    if isinstance(data, list | tuple):
        if len(data) > _NUMBERS and all(isinstance(x, float) for x in data[:_NUMBERS]):
            return f"[{len(data):,} numbers]"
        cut = _state or (bool(data) and is_dataclass(data[0]))  # passages, units, findings
        items = [_mask(data=x, _state=_state) for x in data[:_STATE_ITEMS if cut else None]]
        return items + [f"... {len(data) - len(items)} more"] if len(items) < len(data) else items
    if isinstance(data, dict):
        return {k: _mask(data=v, _state=_state) for k, v in data.items()}
    return data


def observe(name: str, as_type: str = "span", **attributes: Any):
    """A context manager recording ``name`` under whatever is being traced; None when off."""
    langfuse = client()
    if langfuse is None:
        return nullcontext()
    return langfuse.start_as_current_observation(name=name, as_type=as_type, **attributes)


def current_trace_id() -> str | None:
    langfuse = client()
    return langfuse.get_current_trace_id() if langfuse else None


def trace_url(trace_id: str | None) -> str | None:
    """Where a browser opens the trace; the app itself may reach Langfuse by another name."""
    if not trace_id:
        return None
    return f"{settings.langfuse_public_url}/project/{PROJECT_ID}/traces/{trace_id}"


def score(trace_id: str | None, scores: dict[str, float | str]) -> None:
    """Attach the answer's quality to its trace: numbers as numbers, labels as categories."""
    langfuse = client()
    if langfuse is None or not trace_id:
        return
    for name, value in scores.items():
        kind = "CATEGORICAL" if isinstance(value, str) else "NUMERIC"
        langfuse.create_score(trace_id=trace_id, name=name, value=value, data_type=kind)


def timed_node(node: Callable) -> Callable:
    """Wrap a graph node to time it into the state's ``timings``: the path each question
    took, with each step's milliseconds, for the admin dashboard -- which works with
    tracing off, so it cannot read them from Langfuse.

    ``functools.wraps`` keeps the node's signature visible, which is how LangGraph decides to
    pass it the runtime context.
    """

    @functools.wraps(node)
    def run(state, *args, **kwargs):
        started = time.perf_counter()
        update = node(state, *args, **kwargs)
        ms = int((time.perf_counter() - started) * 1000)
        return {**update, "timings": [(node.__name__, ms)]}

    return run


def flush() -> None:
    """Send what is buffered; for short-lived processes such as the evaluation harness."""
    langfuse = client()
    if langfuse is not None:
        langfuse.flush()
