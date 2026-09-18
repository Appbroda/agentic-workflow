"""One Slack thread per feature: the anchor, the configuration, the links, the ledger.

Three columns on `feature_workflows` -- `slack_channel_id`, `slack_thread_ts`,
`slack_root_claimed_at` -- all nullable, because a feature with no Slack configuration has no
anchor and that is not a missing value. Columns rather than `state_json` fields so the
dispatcher can find features needing a root without hydrating artifacts, and so the anchor
cannot vanish at a state write that forgets to project it.

Three new tables:
- `slack_workspace_configurations`: where threads go. At most one enabled row, enforced by a
  partial unique index over a constant expression -- a PostgreSQL feature, which is fine
  because these migrations are PostgreSQL-only already (revision 0001 uses CREATE SEQUENCE).
- `slack_user_links`: each person's self-entered Slack member ID and opt-in scope.
- `slack_notifications`: the delivery ledger. `dedup_key` UNIQUE is the idempotency
  mechanism itself -- the claim is inserted before the send and a duplicate insert loses --
  so the constraint is the deliverable, not decoration.

Revision ID: 20260902_0025
Revises: 20260902_0024
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260902_0025"
down_revision: str | None = "20260902_0024"
branch_labels: str | None = None
depends_on: str | None = None

_ANCHOR_COLUMNS = ("slack_root_claimed_at", "slack_thread_ts", "slack_channel_id")


def upgrade() -> None:
    """Give features an anchor, and Slack delivery its configuration and ledger."""
    op.add_column(
        "feature_workflows", sa.Column("slack_channel_id", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "feature_workflows", sa.Column("slack_thread_ts", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "feature_workflows",
        sa.Column("slack_root_claimed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "slack_workspace_configurations",
        sa.Column("configuration_id", sa.String(length=128), primary_key=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=True),
        sa.Column("workspace_name", sa.String(length=256), nullable=True),
        sa.Column("channel_id", sa.String(length=64), nullable=False),
        sa.Column("channel_name", sa.String(length=256), nullable=True),
        sa.Column("token_owner_id", sa.String(length=128), nullable=False),
        sa.Column("verbosity", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("status_reason", sa.Text(), nullable=True),
        sa.Column("console_base_url", sa.String(length=512), nullable=True),
        sa.Column("updated_by", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    # At most one enabled configuration, decided by the database rather than by application
    # care: two workers saving concurrently cannot both leave an enabled row behind.
    op.execute(
        "CREATE UNIQUE INDEX uq_slack_workspace_configurations_enabled "
        "ON slack_workspace_configurations ((true)) WHERE enabled"
    )
    op.create_table(
        "slack_user_links",
        sa.Column("link_id", sa.String(length=128), primary_key=True),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("slack_user_id", sa.String(length=64), nullable=True),
        sa.Column("notify_scope", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", name="uq_slack_user_links_user"),
    )
    op.create_table(
        "slack_notifications",
        sa.Column("notification_id", sa.String(length=128), primary_key=True),
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("dedup_key", sa.String(length=512), nullable=False),
        sa.Column("entry_kind", sa.String(length=32), nullable=False),
        sa.Column("entry_record_id", sa.String(length=256), nullable=False),
        sa.Column("entry_emission", sa.Integer(), nullable=False),
        sa.Column("template", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("slack_ts", sa.String(length=64), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retry_after_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.UniqueConstraint("dedup_key", name="uq_slack_notifications_dedup_key"),
    )
    op.create_index("ix_slack_notifications_feature_id", "slack_notifications", ["feature_id"])
    op.create_index("ix_slack_notifications_dedup_key", "slack_notifications", ["dedup_key"])
    op.create_index("ix_slack_notifications_status", "slack_notifications", ["status"])


def downgrade() -> None:
    """Drop the ledger, the links, the configuration and the anchors, in that order."""
    op.drop_index("ix_slack_notifications_status", table_name="slack_notifications")
    op.drop_index("ix_slack_notifications_dedup_key", table_name="slack_notifications")
    op.drop_index("ix_slack_notifications_feature_id", table_name="slack_notifications")
    op.drop_table("slack_notifications")
    op.drop_table("slack_user_links")
    op.execute("DROP INDEX uq_slack_workspace_configurations_enabled")
    op.drop_table("slack_workspace_configurations")
    for column in _ANCHOR_COLUMNS:
        op.drop_column("feature_workflows", column)
