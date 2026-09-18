"""Record attempts a person granted a stopped repository, and why.

The platform stops a repository when its retry budget is spent and refuses to reset that on
its own. An operator may override the stop, and the override has to survive in the record:
without these columns the only trace is a counter that has quietly passed the configured
limit, which reads as a platform bug rather than as somebody's decision.

Revision ID: 20260825_0010
Revises: 20260825_0009
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260825_0010"
down_revision: str | None = "20260825_0009"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Add the grant counter and the grant log to persisted child workstreams."""
    op.add_column(
        "feature_child_workflows",
        sa.Column(
            "granted_extra_attempts", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
    )
    op.add_column(
        "feature_child_workflows",
        sa.Column("retry_grants", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
    )


def downgrade() -> None:
    """Drop both columns; existing rows carry no grants to preserve."""
    op.drop_column("feature_child_workflows", "retry_grants")
    op.drop_column("feature_child_workflows", "granted_extra_attempts")
