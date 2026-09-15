# EU AI Act Assistant -- application image.
#
# One image serves four roles, chosen by the command: the admin dashboard (default), the
# chat, database migrations, and corpus ingestion. See docker-compose.yml.

FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    # Reranker weights live in a volume mounted here, so they survive rebuilds instead of
    # being downloaded again (~2.3 GB for the default model).
    HF_HOME=/models

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than the source, so a
# code edit rebuilds in seconds rather than reinstalling torch.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY alembic ./alembic
COPY alembic.ini LICENSE NOTICE.md ./

# Installed editable on purpose. config.py locates the repository root from its own path
# (REPO_ROOT = parents[2]) to find data/raw; installed into site-packages as a wheel, that
# path would point somewhere meaningless.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Run unprivileged. The volume mount points are created and chowned here so that fresh named
# volumes inherit the ownership and stay writable. Chainlit also writes into its app root at
# startup (translations, an uploads folder), so the chat folder must be writable too.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data/raw /models \
    && chown -R app:app /app/data /models /app/src/euaia/chat
USER app

EXPOSE 8000 8001

CMD ["uvicorn", "euaia.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
