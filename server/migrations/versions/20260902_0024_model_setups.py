"""Persist user-authored model setups, and the snapshot a custom feature pins at creation.

One new table: `model_setups`, a per-owner role map -- per role, the platform, model, effort
and output bound -- modelled on `repository_configurations`, with the same per-owner
uniqueness on the name a person chose.

Four new columns, mirroring `20260831_0020` (platform) and `20260831_0023` (tier):
`feature_workflows` and `feature_execution_queue` each gain `model_setup_snapshot` (the copy a
custom feature resolves through -- editing or deleting the setup never changes a running or
historical feature's meaning) and `model_setup_id` (provenance only, never resolved through).
Both are nullable and need no backfill: NULL means "this feature runs a tier", which is what
every existing row does.

Revision ID: 20260902_0024
Revises: 20260831_0023
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260902_0024"
down_revision: str | None = "20260831_0023"
branch_labels: str | None = None
depends_on: str | None = None

_TABLES = ("feature_workflows", "feature_execution_queue")


def upgrade() -> None:
    """Create the setups table and give features and queue entries their pinned snapshot."""
    op.create_table(
        "model_setups",
        sa.Column("setup_id", sa.String(length=128), primary_key=True),
        sa.Column("owner_id", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("roles", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("owner_id", "name", name="uq_model_setups_owner_name"),
    )
    op.create_index("ix_model_setups_owner_id", "model_setups", ["owner_id"])
    for table in _TABLES:
        op.add_column(table, sa.Column("model_setup_snapshot", sa.JSON(), nullable=True))
        op.add_column(table, sa.Column("model_setup_id", sa.String(length=128), nullable=True))


def downgrade() -> None:
    """Drop the setups and the snapshots, leaving every feature to resolve as a tier."""
    for table in reversed(_TABLES):
        op.drop_column(table, "model_setup_id")
        op.drop_column(table, "model_setup_snapshot")
    op.drop_index("ix_model_setups_owner_id", table_name="model_setups")
    op.drop_table("model_setups")
