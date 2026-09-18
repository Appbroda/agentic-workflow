"""Let a person's provider keys be kept, encrypted, instead of retyped every request.

Provider credentials were request-scoped and never persisted, which was a deliberate and
good decision: the platform held nothing worth stealing. The cost was that every action
needing a key asked for it again, so a feature could not be resumed without somebody present
with the key in hand.

This keeps the property that matters. The plaintext is never a column, never returned by any
endpoint and never logged; only a sealed blob and a four-character hint are stored, and the
seal is bound to its owner and provider so a row copied into another name does not open. A
request header still wins over anything stored, so nothing here changes how a request that
brings its own key behaves.

Revision ID: 20260825_0014
Revises: 20260825_0013
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260825_0014"
down_revision: str | None = "20260825_0013"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Create the encrypted provider-credential store."""
    op.create_table(
        "provider_credentials",
        sa.Column("credential_id", sa.String(length=128), nullable=False),
        sa.Column("owner_id", sa.String(length=128), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("nonce", sa.LargeBinary(), nullable=False),
        # Which key sealed this, so a rotation can re-seal without guessing.
        sa.Column("key_version", sa.String(length=32), nullable=False),
        # The last four characters only, so somebody can tell which of their own keys is
        # configured without the platform showing them a secret back.
        sa.Column("hint", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("credential_id"),
        # One credential per provider per owner. Replacing is an update, so an old key
        # cannot linger behind a new one and be resolved by accident.
        sa.UniqueConstraint("owner_id", "provider", name="uq_provider_credentials_owner_provider"),
    )
    op.create_index("ix_provider_credentials_owner_id", "provider_credentials", ["owner_id"])
    op.create_index("ix_provider_credentials_provider", "provider_credentials", ["provider"])


def downgrade() -> None:
    """Drop stored credentials, returning the platform to request-scoped keys only."""
    op.drop_index("ix_provider_credentials_provider", table_name="provider_credentials")
    op.drop_index("ix_provider_credentials_owner_id", table_name="provider_credentials")
    op.drop_table("provider_credentials")
