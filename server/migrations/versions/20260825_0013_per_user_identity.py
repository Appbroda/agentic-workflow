"""Give the platform individual identities instead of one shared credential.

Every audit record the platform held answered the same question the same way: somebody with
the key did it. These tables let a request resolve to a person, so an action can record who
asked for it and an authorization check has something to check against.

The shared key keeps working. It is retained as an explicit administrative identity rather
than removed, because the operator console, the browser flows and existing deployments all
authenticate with it, and taking it away before tokens are actually issued would only mean
an outage.

Revision ID: 20260825_0013
Revises: 20260825_0012
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260825_0013"
down_revision: str | None = "20260825_0012"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create the identity directory and the tokens that prove membership of it."""
    op.create_table(
        "platform_users",
        sa.Column("user_id", sa.String(length=128), nullable=False),
        # Stable across renames, and what an identity provider would map onto later.
        sa.Column("subject", sa.String(length=256), nullable=False),
        sa.Column("display_name", sa.String(length=256), nullable=False),
        sa.Column("roles", sa.JSON(), nullable=False),
        # Disabled rather than deleted: an actor named in a year of audit records must still
        # resolve to somebody, and a deleted row would leave those records pointing nowhere.
        sa.Column("disabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
        sa.UniqueConstraint("subject", name="uq_platform_users_subject"),
    )

    op.create_table(
        "platform_api_tokens",
        sa.Column("token_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        # The digest, never the token. A copy of this database does not let somebody act as
        # its users, and the token cannot be shown again after it is issued.
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=256), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["platform_users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("token_id"),
        # Unique and indexed so authenticating a request is one indexed read. A scan would
        # make request time depend on where in the table the match sits.
        sa.UniqueConstraint("token_hash", name="uq_platform_api_tokens_hash"),
    )
    op.create_index("ix_platform_api_tokens_user_id", "platform_api_tokens", ["user_id"])


def downgrade() -> None:
    """Remove individual identity, leaving the shared platform key as the only credential."""
    op.drop_index("ix_platform_api_tokens_user_id", table_name="platform_api_tokens")
    op.drop_table("platform_api_tokens")
    op.drop_table("platform_users")
