"""The three revisions, applied to a real PostgreSQL holding pre-migration rows.

Marked `postgres` because none of it can be checked anywhere else. Revision 0001 uses
`CREATE SEQUENCE` and dies on SQLite; `ALTER COLUMN ... SET NOT NULL` is a no-op there; and
the two asyncpg defects this file's first run found -- a `sa.func.now()` passed as a bind
parameter, and a JSON string bound to a `json` column without a cast -- are invisible on
SQLite because its driver accepts both.

The last assertion is the one no SQL can make. `owner_id` and `provider` are the AES-GCM
additional authenticated data of a stored credential, so a migration that rewrote either
would leave rows that satisfy every query in §7.6 and fail at the next `resolve`. The only
way to know is to open one.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import text

from services.secrets import (
    EncryptedDatabaseSecretStore,
    SecretStoreError,
    generate_encryption_key,
)
from storage.db import Database
from tests.postgres_support import PostgresTier, database_url_for, run_migration

pytestmark = pytest.mark.postgres

# The revision immediately before this change. Everything below starts here.
BEFORE = "20260908_0028"

# The identity the shared platform key resolves to, and so the owner every pre-existing row
# already has. Spelled out rather than imported for the reason the migrations spell it out:
# what is under test is what those revisions did, not what a constant currently says.
EXISTING_OWNER = "platform-admin"
ADMIN_SUBJECT = "akhilesh@appbroda.com"


async def migrate(url: str, revision: str, *, downgrade: bool = False) -> None:
    """Move one database to a revision, off the event loop.

    Alembic's `env.py` calls `asyncio.run`, which cannot run inside a running loop -- so
    every call from an async test goes through a thread. `postgres_support.run_migration`
    says so in its docstring; this is the one line that obeys it.
    """
    await asyncio.to_thread(run_migration, url, revision, downgrade=downgrade)


async def seed_pre_migration_rows(database: Database, *, encryption_key: str) -> dict[str, str]:
    """Write the rows a deployment holds before any of this was applied.

    A feature with no owner column to put an owner in; a queue entry for it that is *in
    flight*, which is what §7.6's fifth query is about; a second account that is not the
    administrator; a token issued before `kind` existed; and two provider credentials sealed
    by the real store under `platform-admin`.

    Returns the two secrets, so the decryption proof compares against the actual plaintext
    rather than against "something came back".
    """
    now = datetime.now(UTC)
    state = {
        "feature_id": "feature-before-workspaces",
        "workflow_id": "feature-before-workspaces",
        "status": "waiting_for_human",
        "title": "A feature submitted before workspaces existed",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    async with database.session() as session:
        await session.execute(
            text(
                "INSERT INTO feature_workflows (feature_id, status, title, reference,"
                " reference_number, execution_mode, agent_platform, performance_tier,"
                " state_json, created_at, updated_at) VALUES (:fid, 'waiting_for_human',"
                " :title, 'AB-Feature-1', 1, 'live', 'openai', 'high',"
                " CAST(:state AS json), :now, :now)"
            ),
            {
                "fid": state["feature_id"],
                "title": state["title"],
                "state": json.dumps(state),
                "now": now,
            },
        )
        await session.execute(
            text(
                "INSERT INTO feature_references (reference_number, feature_id, allocated_at)"
                " VALUES (1, :fid, :now)"
            ),
            {"fid": state["feature_id"], "now": now},
        )
        # Queued, not settled. Its `requested_by` is `platform-admin` because that is the
        # identity every submission through the shared key established, and §7.6's fifth
        # query proves the backfilled owner agrees with it -- which is the whole reason the
        # backfill value is this id and not a fresh one.
        await session.execute(
            text(
                "INSERT INTO feature_execution_queue (feature_id, execution_mode,"
                " agent_platform, performance_tier, requested_by, intent, intent_payload,"
                " status, attempt, max_attempts, steps, max_steps, queued_at)"
                " VALUES (:fid, 'live', 'openai', 'high', :owner, 'resume',"
                " CAST('{}' AS json), 'queued', 0, 3, 0, 200, :now)"
            ),
            {"fid": state["feature_id"], "owner": EXISTING_OWNER, "now": now},
        )
        await session.execute(
            text(
                "INSERT INTO platform_users (user_id, subject, display_name, roles, disabled,"
                " created_at, updated_at) VALUES ('user-before', 'someone@example.com',"
                " 'Someone', CAST('[\"operator\"]' AS json), false, :now, :now)"
            ),
            {"now": now},
        )
        await session.execute(
            text(
                "INSERT INTO platform_api_tokens (token_id, user_id, token_hash, label,"
                " created_at) VALUES ('token-before', 'user-before', :digest,"
                " 'an automation token', :now)"
            ),
            {"digest": secrets.token_hex(32), "now": now},
        )
        await session.commit()

    store = EncryptedDatabaseSecretStore(database, encryption_key=encryption_key)
    secrets_by_provider = {
        "github": f"gh-{secrets.token_hex(12)}",
        "openai": f"sk-{secrets.token_hex(12)}",
    }
    for provider, secret in secrets_by_provider.items():
        await store.put(owner_id=EXISTING_OWNER, provider=provider, secret=secret)
    return secrets_by_provider


async def scalar(database: Database, statement: str, **parameters: Any) -> Any:
    """Run one scalar query."""
    async with database.session() as session:
        return (await session.execute(text(statement), parameters)).scalar_one()


async def verification_queries(database: Database) -> dict[str, int]:
    """Return §7.6's six counts, by the names the report uses."""
    return {
        "features_with_no_owner": await scalar(
            database, "SELECT count(*) FROM feature_workflows WHERE owner_id IS NULL"
        ),
        "owners_with_no_account": await scalar(
            database,
            "SELECT count(*) FROM feature_workflows WHERE owner_id NOT IN"
            " (SELECT user_id FROM platform_users)",
        ),
        "administrator_accounts": await scalar(
            database,
            "SELECT count(*) FROM platform_users WHERE user_id = :owner",
            owner=EXISTING_OWNER,
        ),
        "credentials_not_the_administrators": await scalar(
            database,
            "SELECT count(*) FROM provider_credentials WHERE owner_id <> :owner",
            owner=EXISTING_OWNER,
        ),
        "in_flight_entries_not_naming_the_owner": await scalar(
            database,
            "SELECT count(*) FROM feature_execution_queue q"
            " JOIN feature_workflows f USING (feature_id)"
            " WHERE q.status IN ('queued','running') AND q.requested_by <> f.owner_id",
        ),
        "tokens_with_an_unknown_kind": await scalar(
            database, "SELECT count(*) FROM platform_api_tokens WHERE kind NOT IN ('api','session')"
        ),
    }


@pytest.fixture
async def migrated_from_before(postgres_tier: PostgresTier) -> Any:
    """Yield a database at revision 0028, holding pre-migration rows, and its secrets.

    Built by copying the head template and downgrading, rather than by migrating from
    scratch: the tier's template is the only migrated database this suite pays for, and the
    downgrade path is one of the things under test anyway.
    """
    name = await postgres_tier.create_database()
    url = database_url_for(name)
    await migrate(url, BEFORE, downgrade=True)
    key = generate_encryption_key()
    database = Database(url)
    try:
        yield database, url, key, await seed_pre_migration_rows(database, encryption_key=key)
    finally:
        await database.dispose()
        await postgres_tier.drop_database(name)


async def test_the_verification_queries_all_return_their_expected_values(
    migrated_from_before: Any,
) -> None:
    """§7.6, against a database that held rows before any of this existed."""
    database, url, _key, _secrets = migrated_from_before

    await migrate(url, "head")

    assert await verification_queries(database) == {
        "features_with_no_owner": 0,
        "owners_with_no_account": 0,
        "administrator_accounts": 1,
        "credentials_not_the_administrators": 0,
        "in_flight_entries_not_naming_the_owner": 0,
        "tokens_with_an_unknown_kind": 0,
    }


async def test_a_stored_credential_still_decrypts_after_the_migration(
    migrated_from_before: Any,
) -> None:
    """The assertion no SQL can make, and the one the whole ownership decision rests on.

    Rewriting `owner_id` would have made every stored credential permanently undecryptable,
    with the rows looking perfectly correct in every query above. This opens them.
    """
    database, url, key, expected = migrated_from_before

    await migrate(url, "head")

    store = EncryptedDatabaseSecretStore(database, encryption_key=key)
    for provider, secret in expected.items():
        assert await store.resolve(owner_id=EXISTING_OWNER, provider=provider) == secret
        descriptor = await store.describe(owner_id=EXISTING_OWNER, provider=provider)
        assert descriptor.configured is True
        assert descriptor.hint == secret[-4:]


async def test_a_credential_survives_a_downgrade_and_upgrade_round_trip(
    migrated_from_before: Any,
) -> None:
    """Rolling the schema back and forward must not disturb what it does not touch."""
    database, url, key, expected = migrated_from_before

    await migrate(url, "head")
    await migrate(url, BEFORE, downgrade=True)
    # The owner column is gone; the credentials, the features and the queue entry are not.
    assert await scalar(database, "SELECT count(*) FROM provider_credentials") == 2
    assert await scalar(database, "SELECT count(*) FROM feature_workflows") == 1
    assert await scalar(database, "SELECT count(*) FROM feature_execution_queue") == 1
    assert (
        await scalar(
            database,
            "SELECT count(*) FROM information_schema.columns WHERE"
            " table_name = 'feature_workflows' AND column_name = 'owner_id'",
        )
        == 0
    )
    await migrate(url, "head")

    assert await verification_queries(database) == {
        "features_with_no_owner": 0,
        "owners_with_no_account": 0,
        "administrator_accounts": 1,
        "credentials_not_the_administrators": 0,
        "in_flight_entries_not_naming_the_owner": 0,
        "tokens_with_an_unknown_kind": 0,
    }
    store = EncryptedDatabaseSecretStore(database, encryption_key=key)
    for provider, secret in expected.items():
        assert await store.resolve(owner_id=EXISTING_OWNER, provider=provider) == secret


async def test_the_owner_column_has_no_default_after_the_migration(
    migrated_from_before: Any,
) -> None:
    """A default would silently file every future ownerless insert under the administrator.

    Which is the exact failure this whole change exists to prevent, so the absence of one is
    asserted rather than assumed -- and asserted as an effect: an insert with no owner is
    refused by the database.
    """
    database, url, _key, _secrets = migrated_from_before

    await migrate(url, "head")

    column = await scalar(
        database,
        "SELECT is_nullable || '/' || coalesce(column_default, 'none') FROM"
        " information_schema.columns WHERE table_name = 'feature_workflows'"
        " AND column_name = 'owner_id'",
    )
    assert column == "NO/none"
    with pytest.raises(Exception, match="owner_id"):
        async with database.session() as session:
            await session.execute(
                text(
                    "INSERT INTO feature_workflows (feature_id, status, title,"
                    " execution_mode, agent_platform, performance_tier, state_json,"
                    " created_at, updated_at) VALUES ('feature-ownerless',"
                    " 'waiting_for_human', 'No owner', 'live', 'openai', 'high',"
                    " CAST('{}' AS json), now(), now())"
                )
            )
            await session.commit()


async def test_the_dashboard_indexes_exist(migrated_from_before: Any) -> None:
    """ "My features, newest first" is the default query, and it must not be a scan."""
    database, url, _key, _secrets = migrated_from_before

    await migrate(url, "head")

    names = await scalar(
        database,
        "SELECT string_agg(indexname, ' ' ORDER BY indexname) FROM pg_indexes"
        " WHERE tablename = 'feature_workflows' AND indexname LIKE '%owner%'",
    )
    assert names == "ix_feature_workflows_owner_created ix_feature_workflows_owner_id"


async def test_the_administrator_revision_is_idempotent(migrated_from_before: Any) -> None:
    """Re-running it must not create a second account for one person."""
    database, url, _key, _secrets = migrated_from_before

    await migrate(url, "head")
    # Rewind the recorded revision without undoing the row, which is what a re-run looks
    # like: an operator applying the chain twice, or a deploy retried after a partial
    # failure that had already inserted.
    await migrate(url, "20260909_0030", downgrade=True)
    await migrate(url, "head")

    assert await scalar(database, "SELECT count(*) FROM platform_users") == 2
    assert (
        await scalar(
            database,
            "SELECT count(*) FROM platform_users WHERE user_id = :owner",
            owner=EXISTING_OWNER,
        )
        == 1
    )


async def test_the_administrator_revision_refuses_a_conflicting_subject(
    migrated_from_before: Any,
) -> None:
    """One person must never end up with two accounts.

    Their credentials, repositories and features are filed under `platform-admin`; a second
    row for the same email would log them into an empty workspace and leave the real one
    reachable only by the shared platform key. So this fails, loudly, and creates nothing.
    """
    database, url, _key, _secrets = migrated_from_before

    await migrate(url, "20260909_0030")
    async with database.session() as session:
        await session.execute(
            text("UPDATE platform_users SET subject = :subject WHERE user_id = 'user-before'"),
            {"subject": ADMIN_SUBJECT},
        )
        await session.commit()

    with pytest.raises(RuntimeError, match="already belongs to user_id"):
        await migrate(url, "head")

    assert await scalar(database, "SELECT count(*) FROM platform_users") == 1


async def test_an_unopenable_credential_is_reported_rather_than_returned_empty(
    migrated_from_before: Any,
) -> None:
    """The failure mode the decryption proof is defending against, made visible.

    If a future change ever does rewrite `owner_id`, this is what it looks like from the
    application: not a missing credential, not an empty string, but a refusal naming the
    encryption key. Asserted here so the proof above is known to be capable of failing.
    """
    database, url, key, _secrets = migrated_from_before

    await migrate(url, "head")
    async with database.session() as session:
        await session.execute(text("UPDATE provider_credentials SET owner_id = 'somebody-else'"))
        await session.commit()

    store = EncryptedDatabaseSecretStore(database, encryption_key=key)
    with pytest.raises(SecretStoreError, match="re-entered"):
        await store.resolve(owner_id="somebody-else", provider="github")
