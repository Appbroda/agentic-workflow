"""Persist which performance tier each feature was submitted on.

The tier joins `agent_platform` everywhere the platform is already pinned, and for the same
reason: it is a property of a feature, chosen by the person submitting it and fixed for that
feature's life, so a feature submitted at Standard resolves its model roles at Standard across
every restart and resume -- whatever presets the deployment is configured with by the time it
resumes.

Two columns, mirroring `20260831_0020`: `feature_workflows` holds what the feature runs at,
and `feature_execution_queue` holds it too, because the worker that eventually runs a feature
is not the request that accepted it.

Existing rows are backfilled to `high` and the columns are NOT NULL. Every feature written
before this resolved against the unsuffixed model-role variables, which are the high tier --
so `high` is not a default standing in for the current configuration, it is what those
features actually ran at.

Revision ID: 20260831_0023
Revises: 20260831_0022
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260831_0023"
down_revision: str | None = "20260831_0022"
branch_labels: str | None = None
depends_on: str | None = None

# The same width as `agent_platform` beside it: a short closed vocabulary, not free text.
_TIER = sa.String(length=16)
# What every row that predates this column ran at.
_BACKFILL = "high"


def upgrade() -> None:
    """Give every feature, and every queue entry, the tier its model roles resolve at."""
    for table in ("feature_workflows", "feature_execution_queue"):
        op.add_column(
            table,
            # `server_default` rather than an application default: existing rows are
            # backfilled by this statement, and a NOT NULL column with no default cannot be
            # added to a table that already holds rows.
            sa.Column("performance_tier", _TIER, nullable=False, server_default=_BACKFILL),
        )


def downgrade() -> None:
    """Drop the tier, leaving every feature to resolve at the unsuffixed configuration."""
    for table in ("feature_execution_queue", "feature_workflows"):
        op.drop_column(table, "performance_tier")
