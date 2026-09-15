"""chat history for the Chainlit interface

Revision ID: e3a9c5d17b42
Revises: c7d2e91b4a08
Create Date: 2026-09-15

Saved conversations for ``euaia.chat``. Chainlit's SQLAlchemy data layer
(``chainlit.data.sql_alchemy``) writes these tables with its own SQL and ships no migrations,
so their shape is dictated by it: the camelCase column names, and timestamps stored as ISO
text, are its choices, not ours.

They live in a separate ``chat`` schema -- the chat connects with ``search_path=chat`` -- so
names as generic as ``users`` and ``steps`` stay out of the corpus schema, and the models in
``euaia.db.models`` (which describe only the corpus) never see them.

None of this is the audit trail. Every answer is still recorded in ``query_log`` whether or
not a conversation is saved.
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = 'e3a9c5d17b42'
down_revision: str | None = 'c7d2e91b4a08'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS chat")
    op.execute(
        """
        CREATE TABLE chat.users (
            "id" UUID PRIMARY KEY,
            "identifier" TEXT NOT NULL UNIQUE,
            "metadata" JSONB NOT NULL,
            "createdAt" TEXT
        )
        """
    )
    op.execute(
        """
        CREATE TABLE chat.threads (
            "id" UUID PRIMARY KEY,
            "createdAt" TEXT,
            "name" TEXT,
            "userId" UUID REFERENCES chat.users ("id") ON DELETE CASCADE,
            "userIdentifier" TEXT,
            "tags" TEXT[],
            "metadata" JSONB
        )
        """
    )
    op.execute(
        """
        CREATE TABLE chat.steps (
            "id" UUID PRIMARY KEY,
            "name" TEXT NOT NULL,
            "type" TEXT NOT NULL,
            "threadId" UUID NOT NULL REFERENCES chat.threads ("id") ON DELETE CASCADE,
            "parentId" UUID,
            "streaming" BOOLEAN NOT NULL,
            "waitForAnswer" BOOLEAN,
            "isError" BOOLEAN,
            "metadata" JSONB,
            "tags" TEXT[],
            "input" TEXT,
            "output" TEXT,
            "createdAt" TEXT,
            "command" TEXT,
            "modes" JSONB,
            "start" TEXT,
            "end" TEXT,
            "generation" JSONB,
            "showInput" TEXT,
            "language" TEXT,
            "indent" INT,
            "defaultOpen" BOOLEAN,
            "autoCollapse" BOOLEAN
        )
        """
    )
    op.execute(
        """
        CREATE TABLE chat.elements (
            "id" UUID PRIMARY KEY,
            "threadId" UUID REFERENCES chat.threads ("id") ON DELETE CASCADE,
            "type" TEXT,
            "url" TEXT,
            "chainlitKey" TEXT,
            "name" TEXT NOT NULL,
            "display" TEXT,
            "objectKey" TEXT,
            "size" TEXT,
            "page" INT,
            "language" TEXT,
            "forId" UUID,
            "mime" TEXT,
            "props" JSONB,
            "autoPlay" BOOLEAN,
            "playerConfig" JSONB
        )
        """
    )
    op.execute(
        """
        CREATE TABLE chat.feedbacks (
            "id" UUID PRIMARY KEY,
            "forId" UUID NOT NULL,
            "threadId" UUID NOT NULL REFERENCES chat.threads ("id") ON DELETE CASCADE,
            "value" INT NOT NULL,
            "comment" TEXT
        )
        """
    )
    # The sidebar lists a user's threads and loads each thread's steps.
    op.execute('CREATE INDEX ix_chat_threads_user ON chat.threads ("userId")')
    op.execute('CREATE INDEX ix_chat_steps_thread ON chat.steps ("threadId")')


def downgrade() -> None:
    op.execute("DROP SCHEMA chat CASCADE")
