"""Each child workstream records which branch its checkout is created from.

A feature revision -- a person asking for changes to work that already completed and shipped a
pull request -- re-runs the repository's workstream on a new branch (`{old}-V2`) created from
the superseded branch's head, so the published work is the starting point rather than
regenerated. The branch a checkout builds from therefore stops being derivable: it is the
repository's configured default for an original run and the previous revision's branch for a
revision run, and only the child row can say which.

Nullable with no backfill. `NULL` means the repository's configured default, which is what
every child written before revisions existed actually did, so the base it decides needs no
data migration -- the `targeted_attempt_kind` precedent.

Revision ID: 20260912_0032
Revises: 20260909_0031
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260912_0032"
down_revision: str | None = "20260909_0031"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Add the nullable base-branch column to child workstream rows."""
    op.add_column(
        "feature_child_workflows",
        sa.Column("base_branch", sa.String(length=256), nullable=True),
    )


def downgrade() -> None:
    """Drop the base-branch column; NULL-equivalent behavior is the pre-revision default."""
    op.drop_column("feature_child_workflows", "base_branch")
