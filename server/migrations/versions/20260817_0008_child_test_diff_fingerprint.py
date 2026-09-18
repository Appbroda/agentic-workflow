"""Persist the test-source fingerprint a retry decision needs.

Revision ID: 20260817_0008
Revises: 20260808_0007
Create Date: 2026-08-17 03:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260817_0008"
down_revision: str | None = "20260808_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Retain test-source bytes across attempts so a test-only fix is not read as a repeat."""
    op.add_column(
        "feature_child_workflows",
        sa.Column("test_diff_fingerprint", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "feature_child_workflows",
        sa.Column("previous_test_fingerprint", sa.String(length=128), nullable=True),
    )


def downgrade() -> None:
    """Drop the retained test-source fingerprints."""
    op.drop_column("feature_child_workflows", "previous_test_fingerprint")
    op.drop_column("feature_child_workflows", "test_diff_fingerprint")
