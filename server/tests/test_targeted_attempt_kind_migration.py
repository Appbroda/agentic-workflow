"""The column `20260903_0026` adds, applied to a database that predates it.

Cloned from `test_performance_tier_migration.py` for the reason that file exists: the column
lands on a table that already exists, so a missing or mis-ordered migration passes every
schema test in the suite and fails only in a deployment, as an `UndefinedColumn` on the first
query after the image has been built and pushed. Only a real server applying the real
migrations can catch that.

What is different here, and is the whole point of the column: there is no backfill. `NULL`
means a workstream retry, which is what every attempt written before this column existed is,
and Part D's compatibility rule turns on exactly that reading. A backfilled value would say
the platform knows something about those attempts that it does not.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from tests.postgres_support import PostgresTier, database_url_for, run_migration

# The revision this migration follows, which is the state a deployment is in before it runs.
_BEFORE = "20260902_0025"
_TABLE = "feature_child_workflows"
_COLUMN = "targeted_attempt_kind"


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
    """Write the feature and child workstream a deployment already holds before this runs."""
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "insert into feature_workflows "
                    "(feature_id, status, title, execution_mode, agent_platform, "
                    " performance_tier, state_json, created_at, updated_at) "
                    "values ('pre-stamp-feature', 'failed', 'Before the stamp', "
                    " 'live', 'anthropic', 'high', '{}', now(), now())"
                )
            )
            await connection.execute(
                text(
                    "insert into feature_child_workflows "
                    "(child_workflow_id, feature_id, repository_id, workstream_id, status, "
                    " branch_name, workspace_path, retry_count, blocking_issues, "
                    " current_validation_results, scoped_requirements, "
                    " out_of_scope_requirements, blocking_setup_issues, "
                    " configured_validation_commands, implementation_expectations, "
                    " production_files_changed, test_files_changed, "
                    " configuration_files_changed, requirements_implemented, "
                    " requirements_not_implemented, implementation_retry_count, "
                    " validation_retry_count, repository_setup_retry_count, "
                    " integration_retry_count, granted_extra_attempts, retry_grants) "
                    "values ('pre-stamp-feature:api', 'pre-stamp-feature', 'api', 'ws-api', "
                    " 'failed', 'ai/pre-stamp', '/w/api', 4, '[]', '[]', '[]', '[]', '[]', "
                    " '[]', '[]', '[]', '[]', '[]', '[]', '[]', 2, 0, 0, 2, 0, '[]')"
                )
            )
    finally:
        await engine.dispose()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_an_attempt_written_before_the_stamp_reads_as_a_workstream_retry(
    postgres_tier: PostgresTier,
) -> None:
    """Null, not backfilled, and nullable -- because absent has a meaning here.

    This workstream spent four attempts, two of them integration remediations, and nothing
    recorded which was which. The migration must not guess: `NULL` is read as a workstream
    retry, which is what the graph has always drawn these as, so old records keep attaching
    to the lane loop rather than moving or vanishing.
    """
    name = await postgres_tier.create_database()
    url = database_url_for(name)
    try:
        # The state a real deployment is in the moment before this migration runs.
        await asyncio.to_thread(run_migration, url, _BEFORE, downgrade=True)
        assert _COLUMN not in await _columns(url, _TABLE)
        await _seed_pre_migration_rows(url)

        await asyncio.to_thread(run_migration, url, "head")

        found = await _columns(url, _TABLE)
        assert _COLUMN in found, f"{_TABLE} gained no {_COLUMN} column"
        # Nullable, deliberately: a NOT NULL column would have forced a backfilled value,
        # and there is no true value to backfill with.
        assert found[_COLUMN] == "YES"
        assert (
            await _scalar(
                url,
                f"select {_COLUMN} from {_TABLE} where child_workflow_id = 'pre-stamp-feature:api'",
            )
            is None
        )
        # The counters it sits beside are untouched, so nothing about the existing attribution
        # changed under it.
        assert (
            await _scalar(
                url,
                "select integration_retry_count from feature_child_workflows "
                "where child_workflow_id = 'pre-stamp-feature:api'",
            )
            == 2
        )
    finally:
        await postgres_tier.drop_database(name)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_the_column_can_be_backed_out_and_applied_again(
    postgres_tier: PostgresTier,
) -> None:
    """A migration that cannot be rolled back is not one a running deployment can take."""
    name = await postgres_tier.create_database()
    url = database_url_for(name)
    try:
        await asyncio.to_thread(run_migration, url, _BEFORE, downgrade=True)
        await _seed_pre_migration_rows(url)
        await asyncio.to_thread(run_migration, url, "head")
        # A stamped attempt, as the platform writes one after this change.
        engine = create_async_engine(url)
        try:
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        f"update {_TABLE} set {_COLUMN} = 'integration_remediation' "
                        "where child_workflow_id = 'pre-stamp-feature:api'"
                    )
                )
        finally:
            await engine.dispose()

        # To the revision below this migration, not "-1": pinning one step from head
        # silently breaks the moment any later migration lands.
        await asyncio.to_thread(run_migration, url, _BEFORE, downgrade=True)
        assert _COLUMN not in await _columns(url, _TABLE)
        # The rows themselves survive: a rollback drops the stamp, not the workstream.
        assert await _scalar(url, f"select count(*) from {_TABLE}") == 1

        # And it re-applies, which is the next thing a rolled-back deployment does -- with
        # the stamp gone, which is correct: the rollback deleted the only record of it, and
        # the reading that follows is "workstream retry", exactly as before the column.
        await asyncio.to_thread(run_migration, url, "head")
        assert (
            await _scalar(
                url,
                f"select {_COLUMN} from {_TABLE} where child_workflow_id = 'pre-stamp-feature:api'",
            )
            is None
        )
    finally:
        await postgres_tier.drop_database(name)
