"""Whose workspace a feature is in.

One column on `feature_workflows`, which is the root of the graph. Thirteen tables carry a
`ForeignKey("feature_workflows.feature_id", ondelete="CASCADE")` and every route that reads
one of them takes a `{feature_id}` and loads the parent, so one column at the parent makes
all thirteen filterable. Adding an owner to each would be thirteen backfills, thirteen
chances to disagree with the parent, and no capability the parent's column does not give.

Nullable, then backfilled, then NOT NULL -- in that order, and deliberately not with a
`server_default`. A default would silently assign every *future* row to `platform-admin` if
some new code path forgot to set an owner, which is precisely the failure this whole change
exists to prevent. Afterwards the column has no default, so a missing owner is an integrity
error at insert time: loud, immediate, and visible in tests.

The backfill value is the literal `platform-admin`, which is the `actor_id` the deployment's
shared platform key has always resolved to and therefore the `owner_id` already on every
stored credential, repository configuration, model setup and attachment. Migration 0031 gives
that id a `platform_users` row. Nothing here rewrites an `owner_id` anywhere -- see 0031's
docstring for why rewriting one would be unrecoverable.

No foreign key from `owner_id` to `platform_users.user_id`, on purpose. Users are disabled
rather than deleted precisely so audit records keep resolving, so referential integrity buys
little; and an FK would make the order of 0030 and 0031 load-bearing and would fail outright
on any deployment whose `platform-admin` row does not exist yet. Ownership is checked in
application code against a resolved actor, which is a stronger guarantee here than the
database's.

Revision ID: 20260909_0030
Revises: 20260909_0029
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260909_0030"
down_revision: str | None = "20260909_0029"
branch_labels: str | None = None
depends_on: str | None = None

# The identity the shared platform key resolves to, and so the owner every pre-existing
# feature has always effectively had. Spelled out rather than imported: a migration must
# describe the schema as it was when it was written, and an import would let a later rename
# of the constant silently change what this revision did.
_EXISTING_OWNER = "platform-admin"


def upgrade() -> None:
    """Add the owner, give every existing feature to the administrator, then require it."""
    op.add_column(
        "feature_workflows",
        sa.Column("owner_id", sa.String(length=128), nullable=True),
    )
    op.execute(
        sa.text("UPDATE feature_workflows SET owner_id = :owner WHERE owner_id IS NULL").bindparams(
            owner=_EXISTING_OWNER
        )
    )
    op.alter_column(
        "feature_workflows",
        "owner_id",
        existing_type=sa.String(length=128),
        nullable=False,
    )
    # The dashboard's default query is "my features, newest first", and `list_features`
    # pages on `(created_at DESC, feature_id DESC)` -- the composite matches that ordering
    # so the owner filter and the keyset cursor are served by one index rather than by a
    # filter over a scan. The plain owner index serves the counts and the joins that do not
    # order.
    op.create_index("ix_feature_workflows_owner_id", "feature_workflows", ["owner_id"])
    op.create_index(
        "ix_feature_workflows_owner_created",
        "feature_workflows",
        ["owner_id", sa.text("created_at DESC"), sa.text("feature_id DESC")],
    )


def downgrade() -> None:
    """Drop the owner and both indexes. Every feature becomes the deployment's again."""
    op.drop_index("ix_feature_workflows_owner_created", table_name="feature_workflows")
    op.drop_index("ix_feature_workflows_owner_id", table_name="feature_workflows")
    op.drop_column("feature_workflows", "owner_id")
