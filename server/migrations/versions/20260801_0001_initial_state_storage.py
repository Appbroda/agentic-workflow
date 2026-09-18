"""Create workflow state and artifact persistence tables.

Revision ID: 20260801_0001
Revises:
Create Date: 2026-08-01 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260801_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the initial tables required to persist workflow state."""
    workflow_status = sa.Enum(
        "pending",
        "running",
        "waiting_for_human",
        "review_rejected",
        "approved",
        "failed",
        "failed_requires_human",
        "cancelled",
        "completed",
        name="workflow_status",
        native_enum=False,
        create_constraint=True,
    )
    approval_state = sa.Enum(
        "not_required",
        "pending",
        "approved",
        "rejected",
        name="approval_state",
        native_enum=False,
        create_constraint=True,
    )
    log_level = sa.Enum(
        "debug",
        "info",
        "warning",
        "error",
        "critical",
        name="log_level",
        native_enum=False,
        create_constraint=True,
    )

    op.create_table(
        "workflows",
        sa.Column("workflow_id", sa.String(length=128), nullable=False),
        sa.Column("status", workflow_status, nullable=False),
        sa.Column("current_agent", sa.String(length=128), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("approval_state", approval_state, nullable=False),
        sa.Column("conversation_history", sa.JSON(), nullable=False),
        sa.Column("workspace_descriptor", sa.JSON(), nullable=False),
        sa.Column("checkpoints", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("workflow_id"),
    )
    op.create_index("ix_workflows_status", "workflows", ["status"], unique=False)

    op.create_table(
        "artifacts",
        sa.Column("artifact_id", sa.String(length=128), nullable=False),
        sa.Column("workflow_id", sa.String(length=128), nullable=False),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.String(length=32), nullable=False),
        sa.Column("producer", sa.String(length=128), nullable=False),
        sa.Column("validation_status", sa.String(length=16), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("produced_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.workflow_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("artifact_id"),
    )
    op.create_index("ix_artifacts_workflow_id", "artifacts", ["workflow_id"], unique=False)

    op.create_table(
        "execution_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("workflow_id", sa.String(length=128), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("level", log_level, nullable=False),
        sa.Column("source", sa.String(length=128), nullable=False),
        sa.Column("event", sa.String(length=128), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("context", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.workflow_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_execution_logs_workflow_id", "execution_logs", ["workflow_id"], unique=False
    )


def downgrade() -> None:
    """Remove the initial workflow persistence tables."""
    op.drop_index("ix_execution_logs_workflow_id", table_name="execution_logs")
    op.drop_table("execution_logs")
    op.drop_index("ix_artifacts_workflow_id", table_name="artifacts")
    op.drop_table("artifacts")
    op.drop_index("ix_workflows_status", table_name="workflows")
    op.drop_table("workflows")
