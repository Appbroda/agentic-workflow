"""Where designs come from: one configuration per deployment.

One new table and no columns on any existing one. That is worth stating explicitly, because
`test_migrations.py` checks that every mapped *table* has a migration and does not check
columns -- so a column added without one passes every test in this suite and fails in a
deployment, which builds its schema from Alembic alone. Nothing else in the 89- design-source
item adds a column, and if that changes the column belongs in this revision deliberately.

`design_source_configurations` is modelled field for field on `slack_workspace_configurations`
(revision 20260902_0025), including its at-most-one-enabled-row index: a partial unique index
over a constant expression, so two operators saving concurrently cannot both leave an enabled
row behind. A PostgreSQL feature, which is fine because these migrations are PostgreSQL-only
already -- revision 0001 uses CREATE SEQUENCE.

The design account's personal access token is deliberately not a column here. It lives in
`provider_credentials` under the `figma` provider for `token_owner_id`, sealed like every
other credential, and this table has no field a token could occupy.

Revision ID: 20260907_0027
Revises: 20260903_0026
Create Date: 2026-09-07 12:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260907_0027"
down_revision: str | None = "20260903_0026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the deployment's one design source configuration table."""
    op.create_table(
        "design_source_configurations",
        sa.Column("configuration_id", sa.String(length=128), primary_key=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("token_owner_id", sa.String(length=128), nullable=False),
        sa.Column("file_allowlist", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("status_reason", sa.Text(), nullable=True),
        sa.Column("updated_by", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    # At most one enabled configuration, decided by the database rather than by application
    # care, exactly as the Slack configuration decides it.
    op.execute(
        "CREATE UNIQUE INDEX uq_design_source_configurations_enabled "
        "ON design_source_configurations ((true)) WHERE enabled"
    )


def downgrade() -> None:
    """Drop the configuration. Citations are then refused for want of a design source."""
    op.execute("DROP INDEX IF EXISTS uq_design_source_configurations_enabled")
    op.drop_table("design_source_configurations")
