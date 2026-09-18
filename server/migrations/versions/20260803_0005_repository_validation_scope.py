"""Persist repository-aware validation currentness and scoped workstream visibility.

Revision ID: 20260803_0005
Revises: 20260803_0004
Create Date: 2026-08-03 03:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260803_0005"
down_revision: str | None = "20260803_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add nullable/defaulted fields so all existing workflows remain readable."""
    with op.batch_alter_table("external_operations") as batch:
        batch.add_column(sa.Column("repository_revision", sa.String(length=128), nullable=True))
        batch.add_column(sa.Column("command_fingerprint", sa.String(length=64), nullable=True))
        batch.add_column(
            sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true())
        )
        batch.add_column(
            sa.Column("superseded_by_operation_id", sa.String(length=128), nullable=True)
        )
        batch.create_index(
            "ix_external_operations_repository_revision", ["repository_revision"], unique=False
        )
        batch.create_index("ix_external_operations_is_current", ["is_current"], unique=False)
    # Legacy rows have no content revision. Retain their audit history but never let
    # a post-Phase-14 reviewer reuse them as current validation evidence.
    op.execute(
        "UPDATE external_operations SET is_current = false WHERE repository_revision IS NULL"
    )
    with op.batch_alter_table("feature_child_workflows") as batch:
        batch.add_column(sa.Column("technology_profile", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("validation_plan", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("current_revision", sa.String(length=128), nullable=True))
        batch.add_column(
            sa.Column("current_validation_results", sa.JSON(), nullable=False, server_default="[]")
        )
        batch.add_column(
            sa.Column(
                "superseded_validation_count", sa.Integer(), nullable=False, server_default="0"
            )
        )
        batch.add_column(
            sa.Column("scoped_requirements", sa.JSON(), nullable=False, server_default="[]")
        )
        batch.add_column(
            sa.Column("out_of_scope_requirements", sa.JSON(), nullable=False, server_default="[]")
        )


def downgrade() -> None:
    """Remove derived diagnostics without modifying immutable artifact histories."""
    with op.batch_alter_table("feature_child_workflows") as batch:
        batch.drop_column("out_of_scope_requirements")
        batch.drop_column("scoped_requirements")
        batch.drop_column("superseded_validation_count")
        batch.drop_column("current_validation_results")
        batch.drop_column("current_revision")
        batch.drop_column("validation_plan")
        batch.drop_column("technology_profile")
    with op.batch_alter_table("external_operations") as batch:
        batch.drop_index("ix_external_operations_is_current")
        batch.drop_index("ix_external_operations_repository_revision")
        batch.drop_column("superseded_by_operation_id")
        batch.drop_column("is_current")
        batch.drop_column("command_fingerprint")
        batch.drop_column("repository_revision")
