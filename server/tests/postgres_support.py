"""Provisioning for the durable test tier that runs on a real PostgreSQL server.

Every durable test in this suite used to run on SQLite, and the mechanisms that make the
platform safe against a second worker are the ones SQLite does not have.
`DatabaseFeatureExecutionQueue.claim` skips `FOR UPDATE SKIP LOCKED` on SQLite by an explicit
dialect check, and `_replace_state`'s `SELECT ... FOR UPDATE` is silently ignored there. Both
are the difference between one worker running a feature and two, and neither was ever executed
by a test.

Three properties this module exists to guarantee:

* **The schema comes from the migrations, not from `create_all`.** Migration drift is a real
  failure mode -- the deployment builds its schema from Alembic and the suite built its own
  from model metadata, so a model with no migration passed here and failed in production.
  The template database is migrated to head and its recorded revision is checked against the
  head the readiness probe reads.
* **Each test gets its own database.** Tests in this tier deliberately create contention;
  they must not contend with each other by accident.
* **An unavailable server is visible.** Locally the tier skips with a reason. In CI --
  anywhere `REQUIRE_POSTGRES_TESTS` is set -- an unavailable server is an error, because a
  tier that passes by skipping is worse than no tier.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

SERVER_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = SERVER_ROOT.parent
MIGRATIONS = SERVER_ROOT / "migrations"

# Set this to an SQLAlchemy async URL to point the tier at a specific server. It should name
# a maintenance database (`postgres`); the per-test databases are created beside it.
URL_ENVIRONMENT_VARIABLE = "POSTGRES_TEST_URL"

# Set this in CI. With it, an unreachable server fails the run instead of skipping it.
REQUIRE_ENVIRONMENT_VARIABLE = "REQUIRE_POSTGRES_TESTS"

# The maintenance database every PostgreSQL server has. Connected to only to issue
# `CREATE DATABASE` and `DROP DATABASE`; no test ever reads or writes it.
_MAINTENANCE_DATABASE = "postgres"


class PostgresUnavailable(RuntimeError):
    """No PostgreSQL server could be reached for the durable tier."""


def _dotenv_values(path: Path) -> dict[str, str]:
    """Read the `KEY=VALUE` pairs a deployment `.env` holds, and nothing cleverer.

    The settings model already reads this file for the same reason: the local loop should
    need no setup beyond the containers the repository ships with. This is a deliberately
    small reader -- it does not expand variables or interpret quotes beyond stripping them --
    because the only keys it is asked for are a host, a port and a password.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _connection_settings() -> dict[str, str]:
    """Resolve host, port and credentials from the environment, then the deployment `.env`."""
    defaults = {
        "POSTGRES_HOST": "127.0.0.1",
        "POSTGRES_PORT": "5432",
        "POSTGRES_USER": "platform",
        "POSTGRES_PASSWORD": "platform",
        "POSTGRES_DB": "ai_platform",
    }
    from_files: dict[str, str] = {}
    for candidate in (REPOSITORY_ROOT / ".env", SERVER_ROOT / ".env"):
        from_files.update(_dotenv_values(candidate))
    resolved = dict(defaults)
    for key in defaults:
        value = os.environ.get(key) or from_files.get(key)
        if value:
            resolved[key] = value
    return resolved


def maintenance_url() -> str:
    """Return the URL the tier connects to in order to create and drop its own databases."""
    configured = os.environ.get(URL_ENVIRONMENT_VARIABLE)
    if configured:
        return configured
    settings = _connection_settings()
    user = quote(settings["POSTGRES_USER"], safe="")
    password = quote(settings["POSTGRES_PASSWORD"], safe="")
    host = settings["POSTGRES_HOST"]
    port = settings["POSTGRES_PORT"]
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{_MAINTENANCE_DATABASE}"


def _protected_database_names() -> frozenset[str]:
    """Names this tier refuses to create, drop, or migrate.

    The local server the tier borrows is the same one the deployment uses. Read-only
    inspection of that database is fine and was how the audit was produced; creating a
    database beside it is fine; touching it is not.
    """
    settings = _connection_settings()
    return frozenset({settings["POSTGRES_DB"], _MAINTENANCE_DATABASE, "template0", "template1"})


def _require_disposable(name: str) -> str:
    """Refuse any database name that is not one this tier made up for itself."""
    if name in _protected_database_names() or not name.startswith("pytest_"):
        msg = f"refusing to manage a database this tier does not own: {name}"
        raise RuntimeError(msg)
    return name


def migration_head() -> str:
    """Read the packaged Alembic head, the same value the readiness probe compares against."""
    configuration = AlembicConfig()
    configuration.set_main_option("script_location", str(MIGRATIONS))
    head = ScriptDirectory.from_config(configuration).get_current_head()
    if head is None:
        msg = "Alembic migration head is not configured"
        raise RuntimeError(msg)
    return head


def database_url_for(name: str) -> str:
    """Compose the async URL for one of this tier's databases."""
    base, _, _ = maintenance_url().rpartition("/")
    return f"{base}/{name}"


async def _run_maintenance(statement: str) -> None:
    """Execute one `CREATE`/`DROP DATABASE`, which cannot run inside a transaction block."""
    engine = create_async_engine(maintenance_url(), isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            await connection.execute(text(statement))
    finally:
        await engine.dispose()


async def server_is_reachable() -> bool:
    """Answer whether a PostgreSQL server is actually there, without raising into a fixture."""
    engine = create_async_engine(maintenance_url(), isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            await connection.execute(text("select 1"))
    except Exception:
        return False
    else:
        return True
    finally:
        await engine.dispose()


def run_migration(url: str, revision: str, *, downgrade: bool = False) -> None:
    """Move one database to a named revision, exactly as the deployment's job does.

    Alembic's `env.py` calls `asyncio.run`, so this must not be invoked from a running loop.
    Callers reach it through `asyncio.to_thread`. `DATABASE_URL` is set around the call
    because `env.py` prefers it over the configured URL, and would otherwise migrate whatever
    that variable happened to name -- which on a developer's machine is the live database.

    A revision is a parameter rather than always `head` so a test can put a database into the
    state a real deployment is in *before* a migration, seed it, and then apply the one under
    test. `test_migrations.py` checks that every mapped table has a migration; it does not
    check columns, so a column added to an existing table needs exactly this.
    """
    configuration = AlembicConfig()
    configuration.set_main_option("script_location", str(MIGRATIONS))
    configuration.set_main_option("sqlalchemy.url", url)
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = url
    try:
        if downgrade:
            command.downgrade(configuration, revision)
        else:
            command.upgrade(configuration, revision)
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous


def _upgrade_to_head(url: str) -> None:
    """Apply every packaged migration, which is what provisioning the template means."""
    run_migration(url, "head")


@dataclass(slots=True)
class PostgresTier:
    """A migrated template database, and disposable copies of it for individual tests.

    Migrating once and copying is what keeps the tier affordable: `CREATE DATABASE ...
    TEMPLATE` is a file copy inside the server, while running eighteen migrations per test
    would cost seconds each. Every copy is still a database whose schema came from the
    migrations rather than from `Base.metadata.create_all`.
    """

    template_name: str
    head_revision: str

    @classmethod
    async def provision(cls) -> PostgresTier:
        """Create and migrate the template this session's databases are copied from."""
        head = migration_head()
        template = _require_disposable(f"pytest_template_{uuid.uuid4().hex[:12]}")
        await _run_maintenance(f'CREATE DATABASE "{template}"')
        try:
            url = database_url_for(template)
            await asyncio.to_thread(_upgrade_to_head, url)
            await _assert_revision_at_head(url, head)
        except BaseException:
            await _run_maintenance(f'DROP DATABASE IF EXISTS "{template}" WITH (FORCE)')
            raise
        return cls(template_name=template, head_revision=head)

    async def create_database(self) -> str:
        """Copy the migrated template into a database this test alone will use."""
        name = _require_disposable(f"pytest_{uuid.uuid4().hex[:16]}")
        await _run_maintenance(f'CREATE DATABASE "{name}" TEMPLATE "{self.template_name}"')
        return name

    async def drop_database(self, name: str) -> None:
        """Remove one test's database, forcing off connections a failed test left open."""
        owned = _require_disposable(name)
        await _run_maintenance(f'DROP DATABASE IF EXISTS "{owned}" WITH (FORCE)')

    async def dispose(self) -> None:
        """Remove the template once the session is over."""
        await self.drop_database(self.template_name)


async def _assert_revision_at_head(url: str, head: str) -> None:
    """Check the migrated database records the revision the deployment expects.

    This is the drift check the suite could not previously make. `test_migrations.py` proves
    the chain is well-formed and that every mapped table has a `create_table` somewhere; only
    actually applying the chain proves the result is the schema the models describe.
    """
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            recorded = await connection.scalar(text("select version_num from alembic_version"))
    finally:
        await engine.dispose()
    if recorded != head:
        msg = f"migrated database records revision {recorded!r}, expected head {head!r}"
        raise RuntimeError(msg)


def requires_postgres_in_this_environment() -> bool:
    """Report whether an unreachable server should fail the run rather than skip it."""
    return os.environ.get(REQUIRE_ENVIRONMENT_VARIABLE, "").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
    }


UNAVAILABLE_REASON = (
    "no PostgreSQL server for the durable tier. Start one with `docker compose up -d postgres`, "
    f"or point {URL_ENVIRONMENT_VARIABLE} at a maintenance database. CI sets "
    f"{REQUIRE_ENVIRONMENT_VARIABLE}=1, which turns this skip into a failure."
)


__all__ = [
    "REQUIRE_ENVIRONMENT_VARIABLE",
    "UNAVAILABLE_REASON",
    "URL_ENVIRONMENT_VARIABLE",
    "PostgresTier",
    "PostgresUnavailable",
    "database_url_for",
    "maintenance_url",
    "migration_head",
    "requires_postgres_in_this_environment",
    "server_is_reachable",
]
