"""Structural unit context, and room for descriptive unit numbers.

The codes of practice are cited as "S1 Measure 1.1" or "Glossary: Detection mechanism for a
marking technique" rather than "6" or "III", which outgrows 32 characters. ``context`` holds
what the numbering alone cannot say -- the path of headings, the Act provisions a commitment
implements, whether a measure is optional -- for the chunk breadcrumb and the evidence header.

Revision ID: b7d4e2a19c63
Revises: a5e81c3f0d92
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "b7d4e2a19c63"
down_revision = "a5e81c3f0d92"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("structural_unit", "unit_number", type_=sa.String(128))
    op.add_column("structural_unit", sa.Column("context", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("structural_unit", "context")
    op.alter_column("structural_unit", "unit_number", type_=sa.String(32))
