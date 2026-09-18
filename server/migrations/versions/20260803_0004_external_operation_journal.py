"""Add durable external-operation journal, attempts, and append-only events.

Revision ID: 20260803_0004
Revises: 20260802_0003
Create Date: 2026-08-03 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260803_0004"
down_revision: str | None = "20260802_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the credential-free operation journal before live side effects are enabled."""
    op.create_table(
        "external_operations",
        sa.Column("operation_id", sa.String(length=128), nullable=False),
        sa.Column("workflow_id", sa.String(length=128), nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=True),
        sa.Column("child_workflow_id", sa.String(length=256), nullable=True),
        sa.Column("repository_id", sa.String(length=128), nullable=True),
        sa.Column("operation_type", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("input_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("external_reference", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("result_payload", sa.JSON(), nullable=True),
        sa.Column("compensation_status", sa.String(length=64), nullable=True),
        sa.Column("safe_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("operation_id"),
        sa.UniqueConstraint("idempotency_key", name="uq_external_operations_key"),
    )
    for name, columns in (
        ("ix_external_operations_workflow_id", ["workflow_id"]),
        ("ix_external_operations_feature_id", ["feature_id"]),
        ("ix_external_operations_child_workflow_id", ["child_workflow_id"]),
        ("ix_external_operations_repository_id", ["repository_id"]),
        ("ix_external_operations_operation_type", ["operation_type"]),
        ("ix_external_operations_status", ["status"]),
    ):
        op.create_index(name, "external_operations", columns, unique=False)
    op.create_table(
        "external_operation_attempts",
        sa.Column("attempt_id", sa.String(length=128), nullable=False),
        sa.Column("operation_id", sa.String(length=128), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=128), nullable=True),
        sa.Column("model", sa.String(length=256), nullable=True),
        sa.Column("workspace_path", sa.Text(), nullable=True),
        sa.Column("task_plan_artifact_id", sa.String(length=256), nullable=True),
        sa.Column("contract_artifact_id", sa.String(length=256), nullable=True),
        sa.Column("process_id", sa.Integer(), nullable=True),
        sa.Column("external_run_id", sa.String(length=256), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("safe_metadata", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["operation_id"], ["external_operations.operation_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("attempt_id"),
    )
    op.create_index(
        "ix_external_operation_attempts_operation_id",
        "external_operation_attempts",
        ["operation_id"],
        unique=False,
    )
    op.create_table(
        "external_operation_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(length=128), nullable=False),
        sa.Column("operation_id", sa.String(length=128), nullable=False),
        sa.Column("previous_status", sa.String(length=64), nullable=True),
        sa.Column("new_status", sa.String(length=64), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("workflow_id", sa.String(length=128), nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=True),
        sa.Column("child_workflow_id", sa.String(length=256), nullable=True),
        sa.Column("repository_id", sa.String(length=128), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("safe_metadata", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["operation_id"], ["external_operations.operation_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id"),
    )
    for name, columns in (
        ("ix_external_operation_events_operation_id", ["operation_id"]),
        ("ix_external_operation_events_workflow_id", ["workflow_id"]),
        ("ix_external_operation_events_feature_id", ["feature_id"]),
    ):
        op.create_index(name, "external_operation_events", columns, unique=False)


def downgrade() -> None:
    """Remove journal children before their parent operation records."""
    for table, index in (
        ("external_operation_events", "ix_external_operation_events_feature_id"),
        ("external_operation_events", "ix_external_operation_events_workflow_id"),
        ("external_operation_events", "ix_external_operation_events_operation_id"),
        ("external_operation_attempts", "ix_external_operation_attempts_operation_id"),
        ("external_operations", "ix_external_operations_status"),
        ("external_operations", "ix_external_operations_operation_type"),
        ("external_operations", "ix_external_operations_repository_id"),
        ("external_operations", "ix_external_operations_child_workflow_id"),
        ("external_operations", "ix_external_operations_feature_id"),
        ("external_operations", "ix_external_operations_workflow_id"),
    ):
        op.drop_index(index, table_name=table)
    op.drop_table("external_operation_events")
    op.drop_table("external_operation_attempts")
    op.drop_table("external_operations")
