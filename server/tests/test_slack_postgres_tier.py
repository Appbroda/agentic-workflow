"""The Slack guarantees only PostgreSQL can demonstrate: the constraints and the migration.

SQLite accepts the schema but cannot prove the race behaviour -- two workers inserting the
same `dedup_key`, two sweeps claiming one root, a second enabled configuration slipping past
the partial unique index. These are the mechanisms the whole item's idempotency rests on, so
they are exercised against the real database.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from state.enums import FeatureWorkflowStatus
from storage.db import Database
from storage.models import FeatureWorkflowModel, SlackWorkspaceConfigurationModel
from storage.slack_store import SlackNotificationStore
from tests.postgres_support import PostgresTier, database_url_for, run_migration

pytestmark = pytest.mark.postgres


async def _seed_feature(database: Database, feature_id: str) -> None:
    """One feature row for the anchor columns to live on."""
    now = datetime.now(UTC)
    async with database.session() as session:
        session.add(
            FeatureWorkflowModel(
                owner_id="platform-admin",
                feature_id=feature_id,
                status=FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
                title="Bulk add apps",
                reference="AB-Feature-184",
                execution_mode="mock",
                agent_platform="openai",
                performance_tier="high",
                state_json={},
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()


async def test_the_dedup_key_rejects_a_concurrent_duplicate_insert(
    postgres_database: Database,
) -> None:
    """T11: two workers claim the same entry at once; the database lets exactly one through."""
    await _seed_feature(postgres_database, "feature-184")
    store = SlackNotificationStore(postgres_database)

    async def claim() -> object:
        return await store.insert_claimed(
            feature_id="feature-184",
            dedup_key="feature-184|artifact|007_review.backend.json|0",
            entry_kind="artifact",
            entry_record_id="007_review.backend.json",
            entry_emission=0,
            template="artifact.review.approved",
        )

    first, second = await asyncio.gather(claim(), claim())
    outcomes = [first, second]
    assert sum(item is not None for item in outcomes) == 1, (
        "the unique constraint is the idempotency mechanism; both claims succeeding means "
        "the same message is sent twice"
    )


async def test_two_concurrent_root_claims_have_one_winner(postgres_database: Database) -> None:
    """T2(c): the conditional UPDATE decides the root, in the database, not in Python."""
    await _seed_feature(postgres_database, "feature-184")
    store = SlackNotificationStore(postgres_database)
    cutoff = datetime.now(UTC) - timedelta(seconds=300)

    first, second = await asyncio.gather(
        store.claim_root("feature-184", stale_claim_cutoff=cutoff),
        store.claim_root("feature-184", stale_claim_cutoff=cutoff),
    )
    assert sorted([first, second]) == [False, True]


async def test_a_second_enabled_configuration_is_rejected(postgres_database: Database) -> None:
    """T11: the partial unique index makes "at most one enabled" the database's decision."""
    now = datetime.now(UTC)

    def _row(configuration_id: str, *, enabled: bool) -> SlackWorkspaceConfigurationModel:
        return SlackWorkspaceConfigurationModel(
            configuration_id=configuration_id,
            enabled=enabled,
            channel_id="C12345",
            token_owner_id="platform-admin",
            verbosity="milestones",
            status="active" if enabled else "disabled",
            updated_by="platform-admin",
            created_at=now,
            updated_at=now,
        )

    async with postgres_database.session() as session:
        session.add(_row("config-1", enabled=True))
        await session.commit()
    # A disabled second row is fine; the index binds only enabled ones.
    async with postgres_database.session() as session:
        session.add(_row("config-2", enabled=False))
        await session.commit()
    with pytest.raises(Exception, match="uq_slack_workspace_configurations_enabled"):
        async with postgres_database.session() as session:
            session.add(_row("config-3", enabled=True))
            await session.commit()


async def test_the_migration_downgrades_and_re_upgrades_with_its_columns(
    postgres_tier: PostgresTier,
) -> None:
    """T12: `upgrade head` -> `downgrade` -> `upgrade head` against real PostgreSQL.

    `test_migrations.py` checks the chain and that every mapped *table* has a migration; the
    three `feature_workflows` columns are exactly what it cannot see, so they are verified
    here through `information_schema` after each direction.
    """
    name = await postgres_tier.create_database()
    url = database_url_for(name)
    database = Database(url)
    try:

        async def _feature_columns() -> set[str]:
            async with database.session() as session:
                rows = await session.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'feature_workflows'"
                    )
                )
                return {row.column_name for row in rows}

        async def _tables() -> set[str]:
            async with database.session() as session:
                rows = await session.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public'"
                    )
                )
                return {row.table_name for row in rows}

        slack_columns = {"slack_channel_id", "slack_thread_ts", "slack_root_claimed_at"}
        slack_tables = {
            "slack_workspace_configurations",
            "slack_user_links",
            "slack_notifications",
        }

        # The tier provisions its template at head, so this database starts fully upgraded.
        assert slack_columns <= await _feature_columns()
        assert slack_tables <= await _tables()

        await asyncio.to_thread(run_migration, url, "20260902_0024", downgrade=True)
        assert slack_columns.isdisjoint(await _feature_columns())
        assert slack_tables.isdisjoint(await _tables())

        await asyncio.to_thread(run_migration, url, "20260902_0025")
        assert slack_columns <= await _feature_columns()
        assert slack_tables <= await _tables()
    finally:
        await database.dispose()
        await postgres_tier.drop_database(name)
