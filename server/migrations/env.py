"""Alembic environment for the platform's asynchronous database engine."""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, async_engine_from_config

from storage.db import attach_iam_auth
from storage.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def database_url() -> str:
    """Prefer the runtime database URL while retaining the local Alembic default."""
    configured_url = config.get_main_option("sqlalchemy.url")
    if configured_url is None:
        msg = "DATABASE_URL or sqlalchemy.url must be configured for migrations"
        raise RuntimeError(msg)
    return os.environ.get("DATABASE_URL", configured_url)


def run_migrations_offline() -> None:
    """Run migrations without creating an async database connection."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Configure and execute migrations through a synchronous connection facade."""
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and execute migrations through SQLAlchemy's sync bridge."""
    configuration = config.get_section(config.config_ini_section, {})
    url = database_url()
    configuration["sqlalchemy.url"] = url
    connectable: AsyncEngine = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    iam_auth_region = os.environ.get("DATABASE_IAM_AUTH_REGION")
    if iam_auth_region:
        attach_iam_auth(connectable, url, iam_auth_region)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations using the configured asynchronous database engine."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
