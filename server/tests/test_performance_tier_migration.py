"""The tier columns `20260831_0023` adds, applied to a database that predates them.

Cloned from `test_agent_platform_migration.py` for the reason it exists: both columns land on
tables that already exist, so a missing migration passes every schema test in the suite and
fails only in a deployment, as an `UndefinedColumn` on the first query after the image has
been built and pushed. Only a real server applying the real migrations can catch that.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tests.postgres_support import PostgresTier, database_url_for, run_migration

# The revision this migration follows, which is the state a deployment is in before it runs.
_BEFORE = "20260831_0022"

# The two tables that gain a column, and what a row written before tiers existed ran at: the
# unsuffixed model-role variables those features resolved against are the high tier.
_TABLES = ("feature_workflows", "feature_execution_queue")
_BACKFILL = "high"


async def _columns(url: str, table: str) -> dict[str, str]:
    """Return column name to `is_nullable` for one table, read from the live schema."""
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            rows = await connection.execute(
                text(
                    "select column_name, is_nullable from information_schema.columns "
                    "where table_name = :table"
                ),
                {"table": table},
            )
            return {name: nullable for name, nullable in rows.all()}
    finally:
        await engine.dispose()


async def _scalar(url: str, statement: str) -> object:
    """Read one value from the migrated database."""
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            return (await connection.execute(text(statement))).scalar_one()
    finally:
        await engine.dispose()


async def _seed_pre_migration_rows(url: str) -> None:
    """Write the feature and queue entry a deployment already holds before this migration."""
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "insert into feature_workflows "
                    "(feature_id, status, title, execution_mode, agent_platform, state_json, "
                    " created_at, updated_at) "
                    "values ('pre-tier-feature', 'completed', 'Before the tier', "
                    " 'live', 'openai', '{}', now(), now())"
                )
            )
            await connection.execute(
                text(
                    "insert into feature_execution_queue "
                    "(feature_id, execution_mode, agent_platform, requested_by, intent, "
                    " intent_payload, status, attempt, max_attempts, queued_at) "
                    "values ('pre-tier-feature', 'live', 'openai', 'platform-admin', 'start', "
                    " '{}', 'succeeded', 1, 3, now())"
                )
            )
    finally:
        await engine.dispose()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_a_feature_written_before_tiers_reads_as_high_after_the_migration(
    postgres_tier: PostgresTier,
) -> None:
    """A historical record must report the cost basis it actually ran at.

    `high` is not a default standing in for current configuration: every pre-tier feature
    resolved against the unsuffixed variables, and those are the high tier. NOT NULL, so
    nothing downstream ever has to decide what an absent value means.
    """
    name = await postgres_tier.create_database()
    url = database_url_for(name)
    try:
        # The state a real deployment is in the moment before this migration runs.
        await asyncio.to_thread(run_migration, url, _BEFORE, downgrade=True)
        for table in _TABLES:
            assert "performance_tier" not in await _columns(url, table)
        await _seed_pre_migration_rows(url)

        await asyncio.to_thread(run_migration, url, "head")

        for table in _TABLES:
            found = await _columns(url, table)
            assert "performance_tier" in found, f"{table} gained no performance_tier column"
            assert found["performance_tier"] == "NO", f"{table}.performance_tier is nullable"
            assert (
                await _scalar(
                    url,
                    f"select performance_tier from {table} where feature_id = 'pre-tier-feature'",
                )
                == _BACKFILL
            )
    finally:
        await postgres_tier.drop_database(name)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_the_tier_columns_can_be_backed_out_and_applied_again(
    postgres_tier: PostgresTier,
) -> None:
    """A migration that cannot be rolled back is not one a running deployment can take."""
    name = await postgres_tier.create_database()
    url = database_url_for(name)
    try:
        await asyncio.to_thread(run_migration, url, _BEFORE, downgrade=True)
        await _seed_pre_migration_rows(url)
        await asyncio.to_thread(run_migration, url, "head")

        # To the revision below the tier migration, not "-1": pinning to one step from head
        # silently breaks the moment any later migration lands.
        await asyncio.to_thread(run_migration, url, _BEFORE, downgrade=True)
        for table in _TABLES:
            assert "performance_tier" not in await _columns(url, table)
        # The rows themselves survive: a rollback drops the tier, not the work.
        assert await _scalar(url, "select count(*) from feature_workflows") == 1

        # And it re-applies, which is the next thing a rolled-back deployment does.
        await asyncio.to_thread(run_migration, url, "head")
        assert (
            await _scalar(
                url,
                "select performance_tier from feature_workflows "
                "where feature_id = 'pre-tier-feature'",
            )
            == _BACKFILL
        )
    finally:
        await postgres_tier.drop_database(name)
