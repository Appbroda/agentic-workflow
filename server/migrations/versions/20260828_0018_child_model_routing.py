"""Persist which configured model role a repository's next attempt uses.

Two columns, because they answer different questions. ``model_routing`` is the decision for the
next attempt -- mode, role, classification, model, why. ``model_routing_state`` is the per
finding record of which tiers have already been tried, which is what makes escalation
monotonic across a restart: without it, recovery would reclassify from the current review text
and could hand a finding that had reached the complex tier back to the cheap one.

Revision ID: 20260828_0018
Revises: 20260827_0017
Create Date: 2026-08-28 09:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260828_0018"
down_revision: str | None = "20260827_0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the routing decision and per-finding escalation record to each child workstream."""
    op.add_column(
        "feature_child_workflows",
        sa.Column("model_routing", sa.JSON(), nullable=True),
    )
    op.add_column(
        "feature_child_workflows",
        sa.Column("model_routing_state", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    """Drop the routing columns. A child then routes as an initial implementation again."""
    op.drop_column("feature_child_workflows", "model_routing_state")
    op.drop_column("feature_child_workflows", "model_routing")
