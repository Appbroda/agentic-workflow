"""Persist the feature assistant conversation and the actions proposed inside it.

Chat is per feature and outlives a browser session, so it is durable like every other part of
a feature's record. Proposed actions are stored alongside the message that proposed them, and
carry their own status: a proposal is never executed by the assistant, only by a person
confirming it, and the record has to show which of those happened.

Revision ID: 20260825_0009
Revises: 20260817_0008
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260825_0009"
down_revision: str | None = "20260817_0008"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create the per-feature chat transcript."""
    op.create_table(
        "feature_chat_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        # Null unless the assistant proposed something. Stored as data rather than parsed out
        # of the message text, so confirming an action cannot depend on reading prose.
        sa.Column("proposed_action", sa.JSON(), nullable=True),
        sa.Column("action_status", sa.String(length=16), nullable=True),
        sa.Column("action_result", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_workflows.feature_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_feature_chat_messages_feature_id", "feature_chat_messages", ["feature_id"], unique=False
    )


def downgrade() -> None:
    """Drop the chat transcript."""
    op.drop_index("ix_feature_chat_messages_feature_id", table_name="feature_chat_messages")
    op.drop_table("feature_chat_messages")
