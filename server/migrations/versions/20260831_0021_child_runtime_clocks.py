"""Record both runtime clocks on each child workstream.

Two columns, one fact each. ``runtime_wall_seconds`` is everything since the workstream's
loop started; ``runtime_charged_seconds`` excludes time spent inside external operations
that ended in a classified provider fault, retries and backoff included. The runtime
ceiling judges the charged clock -- AB-Feature-171's backend spent 54 of its 90 minutes
inside one terminally failing provider call and was ended by a ceiling that charged it for
the weather -- and recording both is what makes that difference legible afterwards.

Revision ID: 20260831_0021
Revises: 20260831_0020
Create Date: 2026-08-31 12:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260831_0021"
down_revision: str | None = "20260831_0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the wall and charged runtime clocks to each child workstream."""
    op.add_column(
        "feature_child_workflows",
        sa.Column("runtime_wall_seconds", sa.Float(), nullable=True),
    )
    op.add_column(
        "feature_child_workflows",
        sa.Column("runtime_charged_seconds", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Drop the clocks. A workstream is then only as legible as its journal timestamps."""
    op.drop_column("feature_child_workflows", "runtime_charged_seconds")
    op.drop_column("feature_child_workflows", "runtime_wall_seconds")
