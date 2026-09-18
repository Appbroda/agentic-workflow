"""Record on the workstream that its plan was written blind.

Two columns, one fact: reconnaissance failed for this repository and the feature was
planned without its evidence. AB-Feature-173's admanager repository was planned blind
after a 37-minute ReadTimeout and the only record was a log line; the operator deciding
whether to trust that plan found out by asking why an artifact was missing. The flag
never clears -- the late reconnaissance probe restores evidence, not the judgment the
planner already exercised without it.

Revision ID: 20260831_0022
Revises: 20260831_0021
Create Date: 2026-08-31 15:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260831_0022"
down_revision: str | None = "20260831_0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the blind-planning flag and its sanitized reason to each child workstream."""
    op.add_column(
        "feature_child_workflows",
        sa.Column("planned_blind", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "feature_child_workflows",
        sa.Column("planned_blind_reason", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Drop the flag. A blind plan is then a log line again."""
    op.drop_column("feature_child_workflows", "planned_blind_reason")
    op.drop_column("feature_child_workflows", "planned_blind")
