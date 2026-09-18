"""Bound a queue entry that now advances its feature one step at a time.

A claim used to mean "run this whole feature": one coroutine held it for a mean of 86 minutes
and up to 571, and everything it had not checkpointed died with the process. A claim now means
"advance this feature by one step and return", and the entry is re-queued for the next one.

That makes the loop between claims durable, and a durable loop needs a bound. `steps` counts
how many this run has completed and `max_steps` is the ceiling, so a step that goes on naming
itself stops instead of re-queueing for ever at the price of a provider call each time.

`steps` is deliberately not a record of *where* the feature is. That is derived from the
feature's own artifacts by `next_step`, and a column restating it would be the second source of
truth this platform has already been bitten by four times. This is a budget, like the attempt
count beside it.

Existing rows start at zero steps with the same ceiling as a new one, which is what they would
have had if they had been written by this revision.

Revision ID: 20260830_0019
Revises: 20260828_0018
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260830_0019"
down_revision: str | None = "20260828_0018"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Give every queue entry a completed-step count and a ceiling on it."""
    op.add_column(
        "feature_execution_queue",
        # `server_default` rather than an application default: existing rows are backfilled
        # by this statement, and a NOT NULL column with no default cannot be added to a table
        # that already holds rows.
        sa.Column("steps", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "feature_execution_queue",
        sa.Column("max_steps", sa.Integer(), nullable=False, server_default="200"),
    )


def downgrade() -> None:
    """Drop the step budget, leaving the queue as it was before stepping."""
    op.drop_column("feature_execution_queue", "max_steps")
    op.drop_column("feature_execution_queue", "steps")
