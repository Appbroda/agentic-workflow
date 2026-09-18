"""Add durable idempotency, lifecycle events, and repeatable artifact storage.

Revision ID: 20260802_0002
Revises: 20260801_0001
Create Date: 2026-08-02 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260802_0002"
down_revision: str | None = "20260801_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Make artifact IDs repeatable across workflows and persist lifecycle command metadata."""
    artifact_sequence = sa.Sequence("artifacts_id_seq")
    op.execute(sa.schema.CreateSequence(artifact_sequence))
    op.add_column(
        "artifacts",
        sa.Column(
            "id",
            sa.Integer(),
            server_default=sa.text("nextval('artifacts_id_seq')"),
            nullable=True,
        ),
    )
    op.execute("UPDATE artifacts SET id = nextval('artifacts_id_seq') WHERE id IS NULL")
    op.alter_column("artifacts", "id", nullable=False, server_default=None)
    op.drop_constraint("artifacts_pkey", "artifacts", type_="primary")
    op.create_primary_key("artifacts_pkey", "artifacts", ["id"])

    op.add_column("execution_logs", sa.Column("log_id", sa.String(length=128), nullable=True))
    op.execute("UPDATE execution_logs SET log_id = 'log-' || id WHERE log_id IS NULL")
    op.alter_column("execution_logs", "log_id", nullable=False)

    op.create_table(
        "workflow_requests",
        sa.Column("idempotency_key", sa.String(length=512), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column("workflow_id", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.workflow_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("idempotency_key"),
    )
    op.create_index(
        "ix_workflow_requests_workflow_id", "workflow_requests", ["workflow_id"], unique=False
    )
    op.create_table(
        "workflow_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("workflow_id", sa.String(length=128), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=128), nullable=False),
        sa.Column("event", sa.String(length=128), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["workflow_id"], ["workflows.workflow_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_workflow_events_workflow_id", "workflow_events", ["workflow_id"], unique=False
    )


def downgrade() -> None:
    """Remove durable control-plane records and restore the original artifact primary key."""
    op.drop_index("ix_workflow_events_workflow_id", table_name="workflow_events")
    op.drop_table("workflow_events")
    op.drop_index("ix_workflow_requests_workflow_id", table_name="workflow_requests")
    op.drop_table("workflow_requests")
    op.drop_column("execution_logs", "log_id")

    op.drop_constraint("artifacts_pkey", "artifacts", type_="primary")
    op.create_primary_key("artifacts_pkey", "artifacts", ["artifact_id"])
    op.drop_column("artifacts", "id")
    op.execute(sa.schema.DropSequence(sa.Sequence("artifacts_id_seq")))
