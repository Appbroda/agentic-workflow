"""The table a submitted image's bytes live in.

One new table: `prd_attachments`. Postgres rather than a volume or an object store, for the
reason the model's docstring gives -- `workspace_data` is per-run scratch, and a submission's
evidence must outlive the run that read it.

No foreign key to `feature_workflows` and so no cascade, deliberately. Retiring a feature
purges attachment *content* and keeps the row, because the submitted PRD artifact quotes the
`attachment_id` and the `sha256`; a cascade would delete the only record that something was
submitted at all.

`content` is nullable from the start: null means purged, and a row whose content is gone
still answers metadata.

Revision ID: 20260908_0028
Revises: 20260907_0027
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260908_0028"
down_revision: str | None = "20260907_0027"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create the attachment table and the two indexes its only queries need."""
    op.create_table(
        "prd_attachments",
        sa.Column("attachment_id", sa.String(length=128), primary_key=True),
        sa.Column("owner_id", sa.String(length=128), nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=True),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("media_type", sa.String(length=64), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bound_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("purged_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Owner: the only reader of an unbound attachment, and how the caps are counted.
    op.create_index("ix_prd_attachments_owner_id", "prd_attachments", ["owner_id"])
    # Feature: how the product manager fetches a submission's images, and how retire finds
    # the content to purge.
    op.create_index("ix_prd_attachments_feature_id", "prd_attachments", ["feature_id"])


def downgrade() -> None:
    """Drop the table. The bytes go with it; the artifacts that named them do not."""
    op.drop_index("ix_prd_attachments_feature_id", table_name="prd_attachments")
    op.drop_index("ix_prd_attachments_owner_id", table_name="prd_attachments")
    op.drop_table("prd_attachments")
