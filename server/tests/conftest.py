"""Fixtures shared by the whole suite, and the provisioning for the `postgres` tier.

The default run is unchanged and still SQLite: `pyproject.toml` deselects `-m postgres`, so a
fast local loop costs nothing. `uv run pytest -m postgres` runs the durable tier, and CI runs
both. When the tier cannot reach a server it skips with a reason locally and fails outright
wherever `REQUIRE_POSTGRES_TESTS` is set, because a tier that passes by skipping certifies
nothing.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator

import pytest

from storage.db import Database
from tests.postgres_support import UNAVAILABLE_REASON as _UNAVAILABLE_REASON
from tests.postgres_support import (
    PostgresTier,
    database_url_for,
    requires_postgres_in_this_environment,
    server_is_reachable,
)
from tools import node_toolchain


@pytest.fixture(autouse=True)
def corepack_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin Corepack as available, because the worker image has it and a laptop may not.

    `package_manager_argv` asks whether Corepack is on PATH to decide whether a repository's
    declared `packageManager` version can be honoured. Left unpinned, every assertion about a
    declaring repository's commands would depend on the machine running the suite: green on a
    developer's laptop with Node installed, asserting the opposite in a container without it.
    Node 22 ships Corepack and the Dockerfile keeps it, so True is what production does.

    Tests for the fallback pin it False themselves -- see `without_corepack` in
    `test_repository_preflight_and_retry.py`.
    """
    monkeypatch.setattr(node_toolchain, "corepack_available", lambda: True)


@pytest.fixture(autouse=True)
async def finish_background_dispatches() -> AsyncIterator[None]:
    """Let an isolated application's one-shot dispatches unwind before the loop closes.

    `create_app` schedules each acceptance as its own background task, and a test that only
    checks the response it was given never awaits that task. Left pending, it is collected
    after its event loop has closed -- with a database session still open inside it, on a
    connection whose worker thread then raises into whichever test is running by then. Which
    is exactly the shape of failure a suite cannot debug: it lands somewhere else.

    A test that wants the result waits on `tests.support.settle`, which finds nothing left to
    do here. This only cleans up after the ones that deliberately do not.
    """
    yield
    pending = [
        task
        for task in asyncio.all_tasks()
        if task.get_name() == "feature-queue-dispatch" and not task.done()
    ]
    if not pending:
        return
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


@pytest.fixture(scope="session")
def postgres_tier() -> Iterator[PostgresTier]:
    """Migrate one template database per session, and hand out copies of it.

    Deliberately a synchronous fixture driving `asyncio.run`: it is set up before any test's
    event loop exists and torn down after the last one is gone, and every connection it opens
    is closed before it returns. That keeps it free of the loop-scope coupling a session-scoped
    async fixture would otherwise impose on every test in the suite.
    """
    if not asyncio.run(server_is_reachable()):
        if requires_postgres_in_this_environment():
            pytest.fail(_UNAVAILABLE_REASON, pytrace=False)
        pytest.skip(_UNAVAILABLE_REASON, allow_module_level=True)
    tier = asyncio.run(PostgresTier.provision())
    try:
        yield tier
    finally:
        asyncio.run(tier.dispose())


@pytest.fixture
async def postgres_database(postgres_tier: PostgresTier) -> AsyncIterator[Database]:
    """Yield a database of this test's own, with the schema the migrations produce.

    Isolated per test rather than per session: this tier's whole point is tests that create
    contention on purpose, and they must not contend with each other by accident.
    """
    name = await postgres_tier.create_database()
    database = Database(database_url_for(name))
    try:
        yield database
    finally:
        await database.dispose()
        await postgres_tier.drop_database(name)
