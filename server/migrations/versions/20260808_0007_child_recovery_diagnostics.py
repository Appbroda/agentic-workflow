"""Persist child recovery boundaries and retry diagnostics.

Revision ID: 20260808_0007
Revises: 20260803_0006
Create Date: 2026-08-08 12:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260808_0007"
down_revision: str | None = "20260803_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Retain the fields needed to explain and safely resume a child attempt."""
    dialect_name = op.get_bind().dialect.name
    op.add_column(
        "workflows",
        sa.Column(
            "workflow_schema_version",
            sa.String(length=32),
            nullable=False,
            server_default="1.0",
        ),
    )
    op.add_column(
        "workflows",
        sa.Column(
            "created_by_build_revision",
            sa.String(length=64),
            nullable=False,
            server_default="legacy-unverified",
        ),
    )
    op.add_column(
        "workflows",
        sa.Column(
            "last_executor_build_revision",
            sa.String(length=64),
            nullable=False,
            server_default="legacy-unverified",
        ),
    )
    workflow_identity_columns = (
        "workflow_schema_version",
        "created_by_build_revision",
        "last_executor_build_revision",
    )
    if dialect_name == "sqlite":
        with op.batch_alter_table("workflows") as batch:
            for column_name in workflow_identity_columns:
                batch.alter_column(column_name, server_default=None)
    else:
        for column_name in workflow_identity_columns:
            op.alter_column("workflows", column_name, server_default=None)
    with op.batch_alter_table("feature_child_workflows") as batch:
        batch.add_column(sa.Column("checkpoint_boundary", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("layout_evidence", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("retry_refusal_reason", sa.Text(), nullable=True))
    # Existing snapshots predate runtime identity. Backfill them explicitly instead of
    # letting model defaults silently bless an unknown executor as the current release.
    if dialect_name == "postgresql":
        identity_defaults = (
            '\'{"workflow_schema_version": "1.0", '
            '"created_by_build_revision": "legacy-unverified", '
            '"last_executor_build_revision": "legacy-unverified"}\'::jsonb'
        )
        op.execute(
            sa.text(
                "UPDATE feature_workflows SET state_json = "
                f"({identity_defaults} || state_json::jsonb)::json"
            )
        )
    elif dialect_name == "sqlite":
        # json_patch applies the existing object last, matching PostgreSQL's `defaults ||
        # state` behavior so already-versioned snapshots are never overwritten.
        op.execute(
            sa.text(
                "UPDATE feature_workflows SET state_json = json_patch("
                '\'{"workflow_schema_version":"1.0",'
                '"created_by_build_revision":"legacy-unverified",'
                '"last_executor_build_revision":"legacy-unverified"}\', state_json)'
            )
        )
    else:
        msg = f"runtime identity migration does not support dialect {dialect_name!r}"
        raise RuntimeError(msg)


def downgrade() -> None:
    """Remove only the added derived recovery fields."""
    dialect_name = op.get_bind().dialect.name
    if dialect_name == "postgresql":
        op.execute(
            sa.text(
                "UPDATE feature_workflows SET state_json = "
                "(state_json::jsonb - 'workflow_schema_version' "
                "- 'created_by_build_revision' - 'last_executor_build_revision')::json"
            )
        )
    elif dialect_name == "sqlite":
        op.execute(
            sa.text(
                "UPDATE feature_workflows SET state_json = json_remove("
                "state_json, '$.workflow_schema_version', '$.created_by_build_revision', "
                "'$.last_executor_build_revision')"
            )
        )
    else:
        msg = f"runtime identity migration does not support dialect {dialect_name!r}"
        raise RuntimeError(msg)
    with op.batch_alter_table("feature_child_workflows") as batch:
        batch.drop_column("retry_refusal_reason")
        batch.drop_column("layout_evidence")
        batch.drop_column("checkpoint_boundary")
    if dialect_name == "sqlite":
        with op.batch_alter_table("workflows") as batch:
            batch.drop_column("last_executor_build_revision")
            batch.drop_column("created_by_build_revision")
            batch.drop_column("workflow_schema_version")
    else:
        op.drop_column("workflows", "last_executor_build_revision")
        op.drop_column("workflows", "created_by_build_revision")
        op.drop_column("workflows", "workflow_schema_version")
