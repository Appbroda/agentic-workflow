"""What a person needs in order to prove who they are with a password they chose.

Four columns on `platform_users` and one on `platform_api_tokens`. Every one of them is
nullable or defaulted, so this revision applies cleanly to a running deployment on the old
code: nothing reads them yet, and nothing inserts a row that would violate them.

`password_hash` is the encoded scrypt string, parameters and salt included, so raising the
work factor later is a rewrite of rows rather than a second migration. `NULL` means this
account cannot password-login at all -- which is the correct state for an account an
identity provider will assert, and for the administrator until the bootstrap sets it.

`kind` on a token distinguishes a login session from a long-lived automation credential.
It earns a column because two operations cannot be expressed without it: changing a password
revokes the user's *sessions* and must leave their API tokens working, and so must "log out
everywhere". Encoding it in `label` by convention would make a display string load-bearing.
`server_default='api'` so every token that already exists keeps the meaning it was issued
with -- a session default would have retroactively made every automation token expirable.

Revision ID: 20260909_0029
Revises: 20260908_0028
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260909_0029"
down_revision: str | None = "20260908_0028"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Add the password columns and the token kind, all safe on a live deployment."""
    op.add_column(
        "platform_users",
        sa.Column("password_hash", sa.String(length=512), nullable=True),
    )
    op.add_column(
        "platform_users",
        sa.Column("password_updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "platform_users",
        sa.Column(
            "must_change_password",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "platform_users",
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "platform_api_tokens",
        sa.Column(
            "kind",
            sa.String(length=16),
            nullable=False,
            server_default="api",
        ),
    )


def downgrade() -> None:
    """Drop the columns. Every stored password and the session/API distinction go with them.

    Rolling this back is not a way to undo login: the application must be rolled back first,
    because code that expects these columns cannot run without them. See
    `docs/AUTHENTICATION_AND_WORKSPACES.md` for the supported direction, which is to roll
    back the application and leave the schema in place.
    """
    op.drop_column("platform_api_tokens", "kind")
    op.drop_column("platform_users", "last_login_at")
    op.drop_column("platform_users", "must_change_password")
    op.drop_column("platform_users", "password_updated_at")
    op.drop_column("platform_users", "password_hash")
