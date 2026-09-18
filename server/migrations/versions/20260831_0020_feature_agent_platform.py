"""Persist which model provider each feature was submitted on.

The provider used to be a deployment constant: whichever `OPENAI_*` variables were set decided
it for everything. It is now a property of a feature, chosen by the person submitting it and
fixed for that feature's life, so a feature submitted on Claude is planned, implemented,
reviewed and remediated on Claude across every restart -- whatever the deployment happens to
be configured with by the time it resumes.

Two columns, because they answer the question at two different moments. `feature_workflows`
holds what the feature is running on. `feature_execution_queue` holds it too, for the same
reason that table already carries `execution_mode`: the worker that eventually runs a feature
is not the request that accepted it, and it needs to know which provider's credentials to
resolve before it has loaded any feature state.

Existing rows are backfilled to `openai` and the columns are NOT NULL. Every feature written
before this ran on OpenAI, so that is not a default -- it is what actually happened. Leaving
NULL to mean "whatever the deployment is set to now" is the defect described at
`configs/model_roles.py` and `adapters/llm_adapter.py`: a historical record that resolves
against current configuration reports what the deployment happens to be at the moment of
reading, which for a completed run is simply wrong.

Revision ID: 20260831_0020
Revises: 20260830_0019
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260831_0020"
down_revision: str | None = "20260830_0019"
branch_labels: str | None = None
depends_on: str | None = None

# The same width as `execution_mode` beside it, and for the same reason: these are short
# closed vocabularies, not free text.
_PLATFORM = sa.String(length=16)
# What every row that predates this column ran on.
_BACKFILL = "openai"


def upgrade() -> None:
    """Give every feature, and every queue entry, the provider it runs on."""
    for table in ("feature_workflows", "feature_execution_queue"):
        op.add_column(
            table,
            # `server_default` rather than an application default: existing rows are
            # backfilled by this statement, and a NOT NULL column with no default cannot be
            # added to a table that already holds rows.
            sa.Column("agent_platform", _PLATFORM, nullable=False, server_default=_BACKFILL),
        )


def downgrade() -> None:
    """Drop the provider, leaving every feature to run on the deployment's own configuration."""
    for table in ("feature_execution_queue", "feature_workflows"):
        op.drop_column(table, "agent_platform")
