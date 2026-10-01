"""Async SQLAlchemy database lifecycle utilities."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import boto3
from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import ConnectionPoolEntry, NullPool, Pool

from storage.models import Base


def attach_iam_auth(engine: AsyncEngine, database_url: str, region: str) -> None:
    """Replace the connection password with a fresh RDS/Aurora IAM auth token on every connect.

    A generated token is only valid for fifteen minutes, so it cannot be baked into the URL
    once at startup -- it has to be regenerated for each new physical connection the pool
    opens. ``do_connect`` fires exactly there, before every connect. It fires synchronously
    even on an async engine, because SQLAlchemy builds the DBAPI connect arguments
    synchronously and only awaits the connect call itself, so a plain (non-async) boto3 call
    here is safe.
    """
    url = make_url(database_url)
    if url.host is None or url.username is None:
        msg = "IAM-authenticated DATABASE_URL must include both a host and a username"
        raise ValueError(msg)
    host = url.host
    db_username = url.username
    client = boto3.client("rds", region_name=region)

    @event.listens_for(engine.sync_engine, "do_connect")
    def _inject_iam_token(
        dialect: Dialect,
        conn_rec: ConnectionPoolEntry,
        cargs: tuple[Any, ...],
        cparams: dict[str, Any],
    ) -> None:
        cparams["password"] = client.generate_db_auth_token(
            DBHostname=host,
            Port=url.port or 5432,
            DBUsername=db_username,
            Region=region,
        )
        cparams["ssl"] = "require"


def create_database_engine(
    database_url: str,
    *,
    echo: bool = False,
    pool_size: int | None = None,
    max_overflow: int | None = None,
    poolclass: type[Pool] | None = None,
    iam_auth_region: str | None = None,
) -> AsyncEngine:
    """Create a production-safe async SQLAlchemy engine for a configured database URL.

    Pool sizing is explicit because the default is too small for this workload and the
    symptom of exceeding it is not an error but a thirty-second wait, which surfaces
    somewhere else entirely -- as a lease that could not be renewed, for instance.

    SQLite has no pool to size (it is a single file behind a synchronous driver), so the
    arguments are only passed where they mean something.
    """
    options: dict[str, object] = {"echo": echo, "pool_pre_ping": True}
    if poolclass is not None:
        options["poolclass"] = poolclass
    elif not database_url.startswith("sqlite") and pool_size is not None:
        options["pool_size"] = pool_size
        options["max_overflow"] = 0 if max_overflow is None else max_overflow
    engine = create_async_engine(database_url, **options)
    if iam_auth_region is not None:
        attach_iam_auth(engine, database_url, iam_auth_region)
    return engine


class Database:
    """Own an async engine, session factory, and explicit schema lifecycle operations."""

    def __init__(
        self,
        database_url: str,
        *,
        echo: bool = False,
        pool_size: int | None = None,
        max_overflow: int | None = None,
        poolclass: type[Pool] | None = None,
        iam_auth_region: str | None = None,
    ) -> None:
        """Configure a database without opening a connection during application import."""
        self.engine = create_database_engine(
            database_url,
            echo=echo,
            pool_size=pool_size,
            max_overflow=max_overflow,
            poolclass=poolclass,
            iam_auth_region=iam_auth_region,
        )
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session and roll back automatically when its caller raises an error."""
        async with self.session_factory() as session:
            try:
                yield session
            except BaseException:
                await session.rollback()
                raise

    async def create_schema(self) -> None:
        """Create all mapped tables for isolated development and test databases."""
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def drop_schema(self) -> None:
        """Drop all mapped tables for isolated test-database cleanup."""
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.drop_all)

    async def dispose(self) -> None:
        """Close the engine and release its connection pool."""
        await self.engine.dispose()


class EphemeralDatabase(Database):
    """A private SQLite database that builds its schema the first time it is used.

    It exists so an isolated application can run the real durable control plane rather than a
    second implementation of the feature lifecycle. Creating the schema in the constructor is
    not available: ``create_app`` is synchronous and an isolated application has no lifespan
    to run setup in, so the only place left is the first session that needs it.

    Deliberately a file and not ``:memory:``. SQLAlchemy gives an in-memory SQLite URL a
    ``StaticPool``, which is one connection shared by every session -- and an application
    whose queue dispatcher works while a request is being served has two coroutines in the
    database at once, which that single connection cannot serve. The directory is removed
    when this object is collected, so nothing survives the process.

    ``NullPool`` for the same reason in reverse: an isolated application is created and
    dropped without anybody disposing it, and a pooled ``aiosqlite`` connection keeps a worker
    thread alive past the event loop it was opened on. Closing each connection with its
    session is what keeps a process that builds a thousand of these from accumulating a
    thousand threads that raise when their loop closes.
    """

    def __init__(self) -> None:
        """Configure a private on-disk engine without opening it."""
        self._directory = TemporaryDirectory(prefix="ai-platform-isolated-db-")
        super().__init__(
            f"sqlite+aiosqlite:///{Path(self._directory.name) / 'platform.db'}",
            poolclass=NullPool,
        )
        self._schema_ready = False

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session against a schema that is guaranteed to exist.

        No lock: ``create_all`` checks first, so two concurrent first sessions cannot produce
        two schemas or a partial one -- the second finds every table already there.
        """
        if not self._schema_ready:
            await self.create_schema()
            self._schema_ready = True
        async with self.session_factory() as session:
            try:
                yield session
            except BaseException:
                await session.rollback()
                raise
