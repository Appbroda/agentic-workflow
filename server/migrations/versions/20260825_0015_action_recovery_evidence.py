"""Bind external effects to actions and checkpoint committed domain results.

Revision ID: 20260825_0015
Revises: 20260825_0014
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260825_0015"
down_revision: str | None = "20260825_0014"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Add the evidence needed to recover one exact action without guessing by time."""
    # 0011/0012 used unconstrained strings while the ORM declared checked enums. Add the
    # missing production constraints without rewriting already valid historical rows.
    op.create_check_constraint(
        "feature_action_status",
        "feature_actions",
        "status IN ('proposed','confirmed','claimed','executing','succeeded','failed',"
        "'requires_reconciliation','cancelled')",
    )
    op.create_check_constraint(
        "feature_repository_repair_status",
        "feature_repository_repairs",
        "status IN ('proposed','approved','executing','succeeded','failed','rejected',"
        "'superseded')",
    )
    op.add_column(
        "feature_actions",
        sa.Column("domain_result_committed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("feature_actions", sa.Column("pending_result_summary", sa.Text(), nullable=True))
    op.add_column(
        "feature_actions", sa.Column("reconciled_by", sa.String(length=128), nullable=True)
    )
    op.add_column("feature_actions", sa.Column("reconciliation_reason", sa.Text(), nullable=True))
    op.add_column(
        "feature_actions",
        sa.Column("reconciled_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_feature_actions_reconciled_by", "feature_actions", ["reconciled_by"])
    op.create_table(
        "feature_action_external_operations",
        sa.Column("action_id", sa.String(length=128), nullable=False),
        sa.Column("operation_id", sa.String(length=128), nullable=False),
        sa.Column("linked_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["action_id"], ["feature_actions.action_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["operation_id"], ["external_operations.operation_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("action_id", "operation_id"),
    )
    op.create_index(
        "ix_feature_action_external_operations_operation_id",
        "feature_action_external_operations",
        ["operation_id"],
    )


def downgrade() -> None:
    """Remove action-specific recovery evidence."""
    op.drop_index(
        "ix_feature_action_external_operations_operation_id",
        table_name="feature_action_external_operations",
    )
    op.drop_table("feature_action_external_operations")
    op.drop_index("ix_feature_actions_reconciled_by", table_name="feature_actions")
    op.drop_column("feature_actions", "reconciled_at")
    op.drop_column("feature_actions", "reconciliation_reason")
    op.drop_column("feature_actions", "reconciled_by")
    op.drop_column("feature_actions", "pending_result_summary")
    op.drop_column("feature_actions", "domain_result_committed_at")
    op.drop_constraint(
        "feature_repository_repair_status", "feature_repository_repairs", type_="check"
    )
    op.drop_constraint("feature_action_status", "feature_actions", type_="check")
