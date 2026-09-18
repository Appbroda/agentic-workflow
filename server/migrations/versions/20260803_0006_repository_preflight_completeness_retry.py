"""Persist repository preflight, completion, and retry diagnostics.

Revision ID: 20260803_0006
Revises: 20260803_0005
Create Date: 2026-08-03 06:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260803_0006"
down_revision: str | None = "20260803_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add nullable/defaulted fields without invalidating existing workflow records."""
    with op.batch_alter_table("feature_child_workflows") as batch:
        batch.add_column(sa.Column("preflight_result", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("preflight_status", sa.String(length=64), nullable=True))
        batch.add_column(
            sa.Column("blocking_setup_issues", sa.JSON(), nullable=False, server_default="[]")
        )
        batch.add_column(sa.Column("selected_package_manager", sa.String(length=32), nullable=True))
        batch.add_column(
            sa.Column(
                "configured_validation_commands", sa.JSON(), nullable=False, server_default="[]"
            )
        )
        batch.add_column(sa.Column("test_availability", sa.String(length=64), nullable=True))
        batch.add_column(
            sa.Column("implementation_expectations", sa.JSON(), nullable=False, server_default="[]")
        )
        batch.add_column(
            sa.Column("production_files_changed", sa.JSON(), nullable=False, server_default="[]")
        )
        batch.add_column(
            sa.Column("test_files_changed", sa.JSON(), nullable=False, server_default="[]")
        )
        batch.add_column(
            sa.Column("configuration_files_changed", sa.JSON(), nullable=False, server_default="[]")
        )
        batch.add_column(
            sa.Column("requirements_implemented", sa.JSON(), nullable=False, server_default="[]")
        )
        batch.add_column(
            sa.Column(
                "requirements_not_implemented", sa.JSON(), nullable=False, server_default="[]"
            )
        )
        batch.add_column(sa.Column("failure_classification", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("retry_strategy", sa.JSON(), nullable=True))
        batch.add_column(
            sa.Column(
                "implementation_retry_count", sa.Integer(), nullable=False, server_default="0"
            )
        )
        batch.add_column(
            sa.Column("validation_retry_count", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(
            sa.Column(
                "repository_setup_retry_count", sa.Integer(), nullable=False, server_default="0"
            )
        )
        batch.add_column(
            sa.Column("integration_retry_count", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(sa.Column("meaningful_change", sa.Boolean(), nullable=True))
        batch.add_column(sa.Column("meaningful_change_reason", sa.Text(), nullable=True))
        batch.add_column(
            sa.Column("production_diff_fingerprint", sa.String(length=128), nullable=True)
        )
        batch.add_column(
            sa.Column("previous_attempt_fingerprint", sa.String(length=128), nullable=True)
        )


def downgrade() -> None:
    """Drop only derived diagnostic fields; immutable artifacts remain untouched."""
    columns = (
        "previous_attempt_fingerprint",
        "production_diff_fingerprint",
        "meaningful_change_reason",
        "meaningful_change",
        "integration_retry_count",
        "repository_setup_retry_count",
        "validation_retry_count",
        "implementation_retry_count",
        "retry_strategy",
        "failure_classification",
        "requirements_not_implemented",
        "requirements_implemented",
        "configuration_files_changed",
        "test_files_changed",
        "production_files_changed",
        "implementation_expectations",
        "test_availability",
        "configured_validation_commands",
        "selected_package_manager",
        "blocking_setup_issues",
        "preflight_status",
        "preflight_result",
    )
    with op.batch_alter_table("feature_child_workflows") as batch:
        for column in columns:
            batch.drop_column(column)
