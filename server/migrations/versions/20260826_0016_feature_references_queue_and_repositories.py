"""Give features a name people can say, a durable queue, and saved repositories.

Three things a product refinement asked for, all of which needed persistence.

**References.** A feature's identity was its internal id -- `feature-9c1f...`, or whatever
string a caller happened to pass, which in the pilot runs was
`adunit-deactivate-live-086`. Neither is something a person quotes in a pull request. This
adds `AB-Feature-N`, allocated by the database rather than counted by the application: two
simultaneous submissions cannot receive the same number, because an insert cannot return the
same primary key twice.

Existing features are backfilled in creation order. That is deliberate and it is a one-way
decision, so it is worth stating why: a reference is display identity only -- nothing joins on
it, no artifact contains it, no pull request was ever titled with it -- and leaving historical
features without one would mean a console showing two kinds of identity forever. Ordering by
`created_at` makes the oldest feature `AB-Feature-1`, which is the only ordering that reads as
a history rather than as an accident. Nothing else about those features is touched: no status,
no artifact, no snapshot is rewritten.

**The queue.** Starting a feature used to mean waiting for planning to finish. The queue is
what lets acceptance be committed and answered on its own, with execution claimed afterwards
under a lease -- so a process dying mid-run leaves work another process picks up.

**Saved repositories.** So somebody stops retyping the same three URLs for every feature.

Revision ID: 20260826_0016
Revises: 20260825_0015
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "20260826_0016"
down_revision: str | None = "20260825_0015"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Add reference allocation, the execution queue, and saved repository configurations."""
    op.create_table(
        "feature_references",
        # The allocation itself. Autoincrement rather than a counter the application reads and
        # writes, because that read-then-write is exactly the race this table removes.
        sa.Column("reference_number", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("allocated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("reference_number"),
        sa.UniqueConstraint("feature_id", name="uq_feature_references_feature"),
        sqlite_autoincrement=True,
    )

    op.add_column("feature_workflows", sa.Column("reference", sa.String(length=64), nullable=True))
    op.add_column("feature_workflows", sa.Column("reference_number", sa.Integer(), nullable=True))
    op.create_index("ix_feature_workflows_reference", "feature_workflows", ["reference"])

    _backfill_references()

    op.create_table(
        "feature_execution_queue",
        sa.Column("feature_id", sa.String(length=128), nullable=False),
        sa.Column("execution_mode", sa.String(length=16), nullable=False),
        # Who asked. The dispatcher resolves this identity's stored provider credentials when
        # it runs, which is why no secret is stored here.
        sa.Column("requested_by", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("feature_id"),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_workflows.feature_id"], ondelete="CASCADE"
        ),
    )
    op.create_index("ix_feature_execution_queue_status", "feature_execution_queue", ["status"])
    op.create_index(
        "ix_feature_execution_queue_queued_at", "feature_execution_queue", ["queued_at"]
    )

    op.create_table(
        "repository_configurations",
        sa.Column("configuration_id", sa.String(length=128), nullable=False),
        sa.Column("owner_id", sa.String(length=128), nullable=False),
        # Derived from the URL, never asked for, recomputed whenever the URL changes.
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("repository_url", sa.Text(), nullable=False),
        sa.Column("default_branch", sa.String(length=256), nullable=False),
        # A label the person chose. It does not constrain what reconnaissance or the planner
        # conclude about the repository's actual role.
        sa.Column("repository_type", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("configuration_id"),
        sa.UniqueConstraint(
            "owner_id", "repository_url", name="uq_repository_configurations_owner_url"
        ),
    )
    op.create_index(
        "ix_repository_configurations_owner_id", "repository_configurations", ["owner_id"]
    )


def _backfill_references() -> None:
    """Assign `AB-Feature-N` to every existing feature, oldest first.

    Deterministic and idempotent: ordering is `(created_at, feature_id)`, so re-running this
    on the same data assigns the same numbers, and the allocation table is left holding every
    number handed out so the next new feature continues the sequence rather than colliding
    with a historical one.
    """
    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            "SELECT feature_id, created_at FROM feature_workflows "
            "ORDER BY created_at ASC, feature_id ASC"
        )
    ).fetchall()
    for index, row in enumerate(rows, start=1):
        connection.execute(
            sa.text(
                "INSERT INTO feature_references (reference_number, feature_id, allocated_at) "
                "VALUES (:number, :feature_id, :allocated_at)"
            ),
            {"number": index, "feature_id": row[0], "allocated_at": row[1]},
        )
        connection.execute(
            sa.text(
                "UPDATE feature_workflows SET reference = :reference, "
                "reference_number = :number WHERE feature_id = :feature_id"
            ),
            {
                "reference": f"AB-Feature-{index}",
                "number": index,
                "feature_id": row[0],
            },
        )
    if rows and connection.dialect.name == "postgresql":
        # The identity column has to be told where the backfill left off, or the next insert
        # starts at 1 and collides with the oldest feature's reference.
        connection.execute(
            sa.text(
                "SELECT setval(pg_get_serial_sequence('feature_references', "
                "'reference_number'), :value)"
            ),
            {"value": len(rows)},
        )


def downgrade() -> None:
    """Remove saved repositories, the queue, and feature references."""
    op.drop_index("ix_repository_configurations_owner_id", table_name="repository_configurations")
    op.drop_table("repository_configurations")
    op.drop_index("ix_feature_execution_queue_queued_at", table_name="feature_execution_queue")
    op.drop_index("ix_feature_execution_queue_status", table_name="feature_execution_queue")
    op.drop_table("feature_execution_queue")
    op.drop_index("ix_feature_workflows_reference", table_name="feature_workflows")
    op.drop_column("feature_workflows", "reference_number")
    op.drop_column("feature_workflows", "reference")
    op.drop_table("feature_references")
