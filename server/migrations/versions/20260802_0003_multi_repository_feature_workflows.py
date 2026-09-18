"""Add durable parent feature workflows and independently queryable child workstreams.

Revision ID: 20260802_0003
Revises: 20260802_0002
Create Date: 2026-08-02 00:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260802_0003"
down_revision: str | None = "20260802_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create non-secret parent, child, contract, review, PR, event, and idempotency tables."""
    op.create_table(
        "feature_workflows",
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("execution_mode", sa.String(length=16), nullable=False),
        sa.Column("merge_strategy", sa.String(length=32), nullable=True),
        sa.Column("deployment_strategy", sa.String(length=32), nullable=True),
        sa.Column("state_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("feature_id"),
    )
    op.create_index("ix_feature_workflows_status", "feature_workflows", ["status"], unique=False)
    op.create_table(
        "feature_repository_specs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("repository_id", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("repository_url", sa.Text(), nullable=False),
        sa.Column("default_branch", sa.String(length=256), nullable=False),
        sa.Column("local_workspace_path", sa.Text(), nullable=True),
        sa.Column("required", sa.Boolean(), nullable=False),
        sa.Column("implementation_order", sa.Integer(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_workflows.feature_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_feature_repository_specs_feature_id",
        "feature_repository_specs",
        ["feature_id"],
        unique=False,
    )
    op.create_table(
        "feature_child_workflows",
        sa.Column("child_workflow_id", sa.String(length=256), nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("repository_id", sa.String(length=128), nullable=False),
        sa.Column("workstream_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("branch_name", sa.String(length=256), nullable=False),
        sa.Column("workspace_path", sa.Text(), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("code_completion_artifact_id", sa.String(length=256), nullable=True),
        sa.Column("review_artifact_id", sa.String(length=256), nullable=True),
        sa.Column("pull_request_artifact_id", sa.String(length=256), nullable=True),
        sa.Column("blocking_issues", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_workflows.feature_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("child_workflow_id"),
    )
    op.create_index(
        "ix_feature_child_workflows_feature_id",
        "feature_child_workflows",
        ["feature_id"],
        unique=False,
    )
    op.create_table(
        "feature_artifacts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("artifact_id", sa.String(length=256), nullable=False),
        sa.Column("artifact_type", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("produced_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_workflows.feature_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_feature_artifacts_feature_id", "feature_artifacts", ["feature_id"], unique=False
    )
    op.create_table(
        "feature_integration_contracts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("artifact_id", sa.String(length=256), nullable=False),
        sa.Column("contract_version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_workflows.feature_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_feature_integration_contracts_feature_id",
        "feature_integration_contracts",
        ["feature_id"],
        unique=False,
    )
    op.create_table(
        "feature_contract_change_requests",
        sa.Column("change_request_id", sa.String(length=256), nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("requested_by_repository_id", sa.String(length=128), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_workflows.feature_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("change_request_id"),
    )
    op.create_index(
        "ix_feature_contract_change_requests_feature_id",
        "feature_contract_change_requests",
        ["feature_id"],
        unique=False,
    )
    op.create_table(
        "feature_integration_reviews",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("artifact_id", sa.String(length=256), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_workflows.feature_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_feature_integration_reviews_feature_id",
        "feature_integration_reviews",
        ["feature_id"],
        unique=False,
    )
    op.create_table(
        "feature_pull_requests",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("repository_id", sa.String(length=128), nullable=False),
        sa.Column("artifact_id", sa.String(length=256), nullable=False),
        sa.Column("pull_request_url", sa.Text(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_workflows.feature_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_feature_pull_requests_feature_id", "feature_pull_requests", ["feature_id"], unique=False
    )
    op.create_table(
        "feature_workflow_requests",
        sa.Column("idempotency_key", sa.String(length=512), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_workflows.feature_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("idempotency_key"),
    )
    op.create_index(
        "ix_feature_workflow_requests_feature_id",
        "feature_workflow_requests",
        ["feature_id"],
        unique=False,
    )
    op.create_table(
        "feature_workflow_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=128), nullable=False),
        sa.Column("event", sa.String(length=128), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_workflows.feature_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_feature_workflow_events_feature_id",
        "feature_workflow_events",
        ["feature_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop Phase 11 tables in foreign-key dependency order."""
    for table, index in (
        ("feature_workflow_events", "ix_feature_workflow_events_feature_id"),
        ("feature_workflow_requests", "ix_feature_workflow_requests_feature_id"),
        ("feature_pull_requests", "ix_feature_pull_requests_feature_id"),
        ("feature_integration_reviews", "ix_feature_integration_reviews_feature_id"),
        ("feature_contract_change_requests", "ix_feature_contract_change_requests_feature_id"),
        ("feature_integration_contracts", "ix_feature_integration_contracts_feature_id"),
        ("feature_artifacts", "ix_feature_artifacts_feature_id"),
        ("feature_child_workflows", "ix_feature_child_workflows_feature_id"),
        ("feature_repository_specs", "ix_feature_repository_specs_feature_id"),
    ):
        op.drop_index(index, table_name=table)
        op.drop_table(table)
    op.drop_index("ix_feature_workflows_status", table_name="feature_workflows")
    op.drop_table("feature_workflows")
