"""The setups table and snapshot columns `20260902_0024` adds, against a real database.

Cloned from `test_performance_tier_migration.py` for the reason it exists: the snapshot
columns land on tables that already exist, so a missing migration passes every schema test in
the suite -- `test_migrations.py` checks tables and never columns -- and fails only in a
deployment, as an `UndefinedColumn` on the first query after the image has been built and
pushed. Only a real server applying the real migrations can catch that. This is the spec's
T18: assert the columns explicitly.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tests.postgres_support import PostgresTier, database_url_for, run_migration

# The revision this migration follows, which is the state a deployment is in before it runs.
_BEFORE = "20260831_0023"

# The two tables that gain the snapshot columns. Nullable, no backfill: a NULL snapshot means
# "this feature runs a tier", which is what every existing row does.
_TABLES = ("feature_workflows", "feature_execution_queue")
_COLUMNS = ("model_setup_snapshot", "model_setup_id")


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
                    "(feature_id, status, title, execution_mode, agent_platform, "
                    " performance_tier, state_json, created_at, updated_at) "
                    "values ('pre-setup-feature', 'completed', 'Before setups', "
                    " 'live', 'openai', 'high', '{}', now(), now())"
                )
            )
            await connection.execute(
                text(
                    "insert into feature_execution_queue "
                    "(feature_id, execution_mode, agent_platform, performance_tier, "
                    " requested_by, intent, intent_payload, status, attempt, max_attempts, "
                    " queued_at) "
                    "values ('pre-setup-feature', 'live', 'openai', 'high', 'platform-admin', "
                    " 'start', '{}', 'succeeded', 1, 3, now())"
                )
            )
    finally:
        await engine.dispose()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_the_setups_table_and_snapshot_columns_exist_after_the_migration(
    postgres_tier: PostgresTier,
) -> None:
    """T18: `model_setups` and all four snapshot columns exist, asserted column by column.

    A feature written before setups keeps NULL in both columns -- it runs a tier, which is
    what NULL means -- so no backfill is asserted, only presence and nullability.
    """
    name = await postgres_tier.create_database()
    url = database_url_for(name)
    try:
        # The state a real deployment is in the moment before this migration runs.
        await asyncio.to_thread(run_migration, url, _BEFORE, downgrade=True)
        assert not await _columns(url, "model_setups")
        for table in _TABLES:
            found = await _columns(url, table)
            for column in _COLUMNS:
                assert column not in found
        await _seed_pre_migration_rows(url)

        await asyncio.to_thread(run_migration, url, "head")

        setups = await _columns(url, "model_setups")
        for column in ("setup_id", "owner_id", "name", "roles", "created_at", "updated_at"):
            assert column in setups, f"model_setups gained no {column} column"
            assert setups[column] == "NO", f"model_setups.{column} is nullable"
        for table in _TABLES:
            found = await _columns(url, table)
            for column in _COLUMNS:
                assert column in found, f"{table} gained no {column} column"
                assert found[column] == "YES", f"{table}.{column} is not nullable"
            assert (
                await _scalar(
                    url,
                    f"select model_setup_id is null from {table} "
                    "where feature_id = 'pre-setup-feature'",
                )
                is True
            )
    finally:
        await postgres_tier.drop_database(name)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_the_setup_columns_can_be_backed_out_and_applied_again(
    postgres_tier: PostgresTier,
) -> None:
    """A migration that cannot be rolled back is not one a running deployment can take."""
    name = await postgres_tier.create_database()
    url = database_url_for(name)
    try:
        await asyncio.to_thread(run_migration, url, _BEFORE, downgrade=True)
        await _seed_pre_migration_rows(url)
        await asyncio.to_thread(run_migration, url, "head")

        # To the revision below this migration, not "-1": pinning to one step from head
        # silently breaks the moment any later migration lands.
        await asyncio.to_thread(run_migration, url, _BEFORE, downgrade=True)
        assert not await _columns(url, "model_setups")
        for table in _TABLES:
            found = await _columns(url, table)
            for column in _COLUMNS:
                assert column not in found
        # The rows themselves survive: a rollback drops the columns, not the work.
        assert await _scalar(url, "select count(*) from feature_workflows") == 1

        # And it re-applies, which is the next thing a rolled-back deployment does.
        await asyncio.to_thread(run_migration, url, "head")
        assert "model_setup_id" in await _columns(url, "feature_workflows")
    finally:
        await postgres_tier.drop_database(name)
