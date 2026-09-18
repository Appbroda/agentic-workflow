"""Make a workflow-changing action a durable record rather than a column on a message.

Confirming an action used to be tracked by one nullable string on the chat message that
proposed it: claimed by a conditional update, finished by another. An executor that died
between the two left the message reading ``executing`` forever, with nothing in the platform
able to move it and no way to tell whether the retry it authorized had actually happened.

The record here fixes that. It exists before anything runs, carries an identity that makes a
repeat a replay rather than a second effect, and is owned by a lease -- so an executor that
stops existing stops renewing, and recovery can tell.

Revision ID: 20260825_0011
Revises: 20260825_0010
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260825_0011"
down_revision: str | None = "20260825_0010"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create the durable action table and link confirmed proposals to it."""
    op.create_table(
        "feature_actions",
        sa.Column("action_id", sa.String(length=128), nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("repository_id", sa.String(length=128), nullable=True),
        sa.Column("action_type", sa.String(length=64), nullable=False),
        sa.Column("actor_id", sa.String(length=128), nullable=False),
        sa.Column("actor_display_name", sa.String(length=256), nullable=True),
        sa.Column("origin", sa.String(length=16), nullable=False),
        sa.Column("origin_message_id", sa.Integer(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("input_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("lease_owner", sa.String(length=256), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("external_operation_ids", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_workflows.feature_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("action_id"),
        # The database enforces one action per intent. Two concurrent confirmations both find
        # nothing and both insert; exactly one wins and the other reads the winner's row.
        # That is what makes a double-clicked Confirm button produce one effect.
        sa.UniqueConstraint("idempotency_key", name="uq_feature_actions_key"),
    )
    op.create_index("ix_feature_actions_feature_id", "feature_actions", ["feature_id"])
    op.create_index("ix_feature_actions_repository_id", "feature_actions", ["repository_id"])
    op.create_index("ix_feature_actions_action_type", "feature_actions", ["action_type"])
    op.create_index("ix_feature_actions_actor_id", "feature_actions", ["actor_id"])
    op.create_index("ix_feature_actions_status", "feature_actions", ["status"])
    # Recovery's only query: active actions whose lease has lapsed.
    op.create_index("ix_feature_actions_lease_expires_at", "feature_actions", ["lease_expires_at"])

    # Nullable, and deliberately not backfilled. A message written before durable actions
    # existed has no action to point at, and inventing one would claim knowledge of an
    # execution nobody recorded.
    op.add_column(
        "feature_chat_messages", sa.Column("action_id", sa.String(length=128), nullable=True)
    )
    op.create_index("ix_feature_chat_messages_action_id", "feature_chat_messages", ["action_id"])


def downgrade() -> None:
    """Remove the durable action record and its link from the transcript."""
    op.drop_index("ix_feature_chat_messages_action_id", table_name="feature_chat_messages")
    op.drop_column("feature_chat_messages", "action_id")
    for index in (
        "ix_feature_actions_lease_expires_at",
        "ix_feature_actions_status",
        "ix_feature_actions_actor_id",
        "ix_feature_actions_action_type",
        "ix_feature_actions_repository_id",
        "ix_feature_actions_feature_id",
    ):
        op.drop_index(index, table_name="feature_actions")
    op.drop_table("feature_actions")
