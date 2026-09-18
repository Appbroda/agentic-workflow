"""Record which authority demanded the attempt a workstream is on.

Integration-remediation retries drew as loops from each lane's own Review node, and their
records said "Review → Implementation" -- attributing to the repository reviewer a rework
that the Integration review had demanded. Observed live on AB-Feature-201: both children
finished, the integration review requested changes, and attempts 3/12 and 4/12 drew as
orange loops out of the wrong node.

Nothing persisted could answer it. The true fact is an in-flight parameter; the previous
attempt's failure classification fails in both directions -- an approved attempt carries
none, and a repository review with a `contract`-category finding sets `contract_mismatch`
too -- and `integration_retry_count` is cumulative, so it cannot say whether attempt K in
particular was a remediation. So the kind is stamped where those counters are stamped.

One additive nullable column and no backfill: `NULL` means a workstream retry, which is
what every attempt written before this column existed is, so old records keep drawing from
the lane loop exactly as they do today.

Revision ID: 20260903_0026
Revises: 20260902_0025
Create Date: 2026-09-03 09:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260903_0026"
down_revision: str | None = "20260902_0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the recorded attempt kind to each child workstream."""
    op.add_column(
        "feature_child_workflows",
        sa.Column("targeted_attempt_kind", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    """Drop the stamp. Every retry then draws from the lane's review loop again."""
    op.drop_column("feature_child_workflows", "targeted_attempt_kind")
