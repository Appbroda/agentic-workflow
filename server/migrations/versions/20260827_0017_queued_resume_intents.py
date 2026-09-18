"""Let the execution queue carry *why* a feature is being run, not only which one.

Starting a feature already went through the queue. Resuming one, and granting a stopped
workstream more attempts, did not: both ran the entire feature inside the HTTP request that
asked for them. AB-Feature-108 is what that costs. A clarification answer at 14:26 opened a
`POST /features/{id}/resume` that then ran contract planning, two workstreams and nine coding
attempts inside itself. Seventy minutes later one lease renewal failed, the request was
cancelled, and the run died mid-attempt -- with no terminal status, because the only thing
that knew the run existed was a socket.

A queue entry could not express "resume with these answers", so this adds the two columns
that let it: which operation to perform, and the payload that operation needs. `start` is the
default, so every existing row keeps meaning exactly what it meant.

`intent_payload` holds clarification answers and retry grants -- request arguments the caller
already sent in plaintext, and which are read back by the same platform. It deliberately never
holds provider credentials: the entry records *who* asked, and the worker resolves that
identity's stored credentials when it runs, so a secret is never at rest in this table.

Revision ID: 20260827_0017
Revises: 20260826_0016
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260827_0017"
down_revision: str | None = "20260826_0016"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Add the intent and its payload to every queue entry, defaulting to today's behaviour."""
    op.add_column(
        "feature_execution_queue",
        # `server_default` rather than an application default: existing rows are being
        # backfilled by this statement, and a NOT NULL column with no default cannot be added
        # to a table that already has rows.
        sa.Column(
            "intent",
            sa.String(length=32),
            nullable=False,
            server_default="start",
        ),
    )
    op.add_column(
        "feature_execution_queue",
        sa.Column(
            "intent_payload",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )


def downgrade() -> None:
    """Drop the intent columns, returning the queue to start-only entries."""
    op.drop_column("feature_execution_queue", "intent_payload")
    op.drop_column("feature_execution_queue", "intent")
