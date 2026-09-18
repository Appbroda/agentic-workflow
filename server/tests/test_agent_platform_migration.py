"""The one thing `test_migrations.py` cannot check: a column added to an existing table.

That suite checks the chain, the downgrades, and that every *table* a model maps has a
migration creating it. Both columns `20260831_0020` adds are on tables that already exist, so
a missing migration passes every test in the suite and fails only in a deployment -- as an
`UndefinedColumn` on the first query, after the image has been built and pushed.

This runs in the `postgres` tier for the reason that tier exists: the suite builds its schema
with `create_all` from model metadata, and the deployment builds it from Alembic. Only a real
server applying the real migrations can tell the two apart.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tests.postgres_support import PostgresTier, database_url_for, run_migration

# The revision this migration follows, which is the state a deployment is in before it runs.
_BEFORE = "20260830_0019"

# The two tables that gain a column, and what a row written before the choice existed ran on.
_TABLES = ("feature_workflows", "feature_execution_queue")
_BACKFILL = "openai"


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
                    "(feature_id, status, title, execution_mode, state_json, "
                    " created_at, updated_at) "
                    "values ('pre-platform-feature', 'completed', 'Before the choice', "
                    " 'live', '{}', now(), now())"
                )
            )
            await connection.execute(
                text(
                    "insert into feature_execution_queue "
                    "(feature_id, execution_mode, requested_by, intent, intent_payload, "
                    " status, attempt, max_attempts, queued_at) "
                    "values ('pre-platform-feature', 'live', 'platform-admin', 'start', "
                    " '{}', 'succeeded', 1, 3, now())"
                )
            )
    finally:
        await engine.dispose()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_a_feature_written_before_the_choice_reads_as_openai_after_the_migration(
    postgres_tier: PostgresTier,
) -> None:
    """A historical record must report what it ran on, not what the deployment is set to now.

    Leaving `NULL` to mean "whatever is configured at the moment of reading" is the defect
    `configs/model_roles.py` and `adapters/llm_adapter.py` both already carry comments about:
    for a completed run it reports something that is simply wrong. So the columns are NOT NULL
    and every existing row is backfilled to the provider it actually used.
    """
    name = await postgres_tier.create_database()
    url = database_url_for(name)
    try:
        # The state a real deployment is in the moment before this migration runs.
        await asyncio.to_thread(run_migration, url, _BEFORE, downgrade=True)
        for table in _TABLES:
            assert "agent_platform" not in await _columns(url, table)
        await _seed_pre_migration_rows(url)

        await asyncio.to_thread(run_migration, url, "head")

        for table in _TABLES:
            found = await _columns(url, table)
            assert "agent_platform" in found, f"{table} gained no agent_platform column"
            # NOT NULL, so nothing downstream ever has to decide what an absent value means.
            assert found["agent_platform"] == "NO", f"{table}.agent_platform is nullable"
            assert (
                await _scalar(
                    url,
                    f"select agent_platform from {table} where feature_id = 'pre-platform-feature'",
                )
                == _BACKFILL
            )
    finally:
        await postgres_tier.drop_database(name)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_the_platform_columns_can_be_backed_out_and_applied_again(
    postgres_tier: PostgresTier,
) -> None:
    """A migration that cannot be rolled back is not one a running deployment can take."""
    name = await postgres_tier.create_database()
    url = database_url_for(name)
    try:
        await asyncio.to_thread(run_migration, url, _BEFORE, downgrade=True)
        await _seed_pre_migration_rows(url)
        await asyncio.to_thread(run_migration, url, "head")

        # To the revision below the platform migration, not "-1": pinning to one step from
        # head silently broke the moment any later migration landed, which says nothing
        # about whether the platform columns themselves can be backed out.
        await asyncio.to_thread(run_migration, url, _BEFORE, downgrade=True)
        for table in _TABLES:
            assert "agent_platform" not in await _columns(url, table)
        # The rows themselves survive: a rollback drops the choice, not the work.
        assert await _scalar(url, "select count(*) from feature_workflows") == 1

        # And it re-applies, which is the next thing a rolled-back deployment does.
        await asyncio.to_thread(run_migration, url, "head")
        assert (
            await _scalar(
                url,
                "select agent_platform from feature_workflows "
                "where feature_id = 'pre-platform-feature'",
            )
            == _BACKFILL
        )
    finally:
        await postgres_tier.drop_database(name)
