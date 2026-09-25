"""Monitoring: how each answer was produced, and evaluation runs over time.

``query_log.run`` holds the graph path with each node's time, tokens per model, rounds and
the Langfuse trace id -- one JSON column, so what is measured can grow without a migration.
``eval_run`` keeps each evaluation's summary, so accuracy can be followed across prompt
versions on the admin dashboard.

Revision ID: d4f8a2c6e913
Revises: b7d4e2a19c63
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "d4f8a2c6e913"
down_revision = "b7d4e2a19c63"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("query_log", sa.Column("run", postgresql.JSONB(), nullable=True))
    op.create_table(
        "eval_run",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("ran_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
                  nullable=False),
        sa.Column("prompt_version", sa.String(32)),
        sa.Column("cases", sa.Integer(), nullable=False),
        sa.Column("summary", postgresql.JSONB(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("eval_run")
    op.drop_column("query_log", "run")
