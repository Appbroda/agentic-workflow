"""Index the repository repairs a person has to decide about.

A repository repair is a durable decision, like a contract change: the platform will not
alter somebody's checked-in setup on its own, so it writes down what is wrong and what it
proposes, and waits. The artifact history is the record; this table is the index, so asking
"what is waiting on me for this repository" does not mean deserializing every artifact a
completed feature ever produced.

Revision ID: 20260825_0012
Revises: 20260825_0011
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260825_0012"
down_revision: str | None = "20260825_0011"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create the repository repair decision index."""
    op.create_table(
        "feature_repository_repairs",
        sa.Column("repair_id", sa.String(length=128), nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("repository_id", sa.String(length=128), nullable=False),
        sa.Column("artifact_id", sa.String(length=256), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("failure_classification", sa.String(length=64), nullable=False),
        # What the diagnosis was written against. An approval is refused when the repository
        # has moved on, because the commands were chosen for a checkout that is gone.
        sa.Column("proposed_at_revision", sa.String(length=128), nullable=True),
        sa.Column("resulting_revision", sa.String(length=128), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_workflows.feature_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("repair_id"),
    )
    op.create_index(
        "ix_feature_repository_repairs_feature_id", "feature_repository_repairs", ["feature_id"]
    )
    op.create_index(
        "ix_feature_repository_repairs_repository_id",
        "feature_repository_repairs",
        ["repository_id"],
    )
    op.create_index(
        "ix_feature_repository_repairs_status", "feature_repository_repairs", ["status"]
    )


def downgrade() -> None:
    """Drop the repository repair index. The artifact history is unaffected."""
    for index in (
        "ix_feature_repository_repairs_status",
        "ix_feature_repository_repairs_repository_id",
        "ix_feature_repository_repairs_feature_id",
    ):
        op.drop_index(index, table_name="feature_repository_repairs")
    op.drop_table("feature_repository_repairs")
