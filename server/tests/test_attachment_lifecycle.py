"""What happens to a submitted image's bytes afterwards (89-, Part H).

Two ends, and the difference between them is the point. An upload nobody submitted is
*deleted*, row and all, because it is not part of any record. An image that was submitted has
its content purged and its row kept, because the PRD artifact quotes its filename, size and
hash -- and a reader following that reference has to be told the bytes are gone rather than
that they never existed, which is why the fetch answers 410 and not 404.

The third thing here is the number. This is the first user-supplied blob in the database and
the first thing in it that grows without anybody deploying anything, so the sweep accounts for
it in a structured log line from day one; there is no operator storage surface to put it on.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import structlog
from httpx import ASGITransport, AsyncClient

from main import create_app
from services.recovery_service import RecoveryService
from storage.attachment_store import (
    UNBOUND_ATTACHMENT_TTL_SECONDS,
    DatabaseAttachmentStore,
    InMemoryAttachmentStore,
)
from storage.db import EphemeralDatabase
from storage.external_operation_store import ExternalOperationJournal
from storage.models import PRDAttachmentModel
from tests import attachment_fixtures as fixtures
from tests.support import settle

KEY = "attachment-lifecycle-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
OWNER = "platform-admin"


def client(app: Any) -> AsyncClient:
    """Return an HTTP client bound to the application under test."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


def payload(attachment_id: str) -> dict[str, Any]:
    """One mock submission carrying one image."""
    return {
        "feature_id": "feature-retire",
        "prd": {
            "title": "Login audit trail",
            "problem_statement": "Login fails with no explanation [image:login-error].",
            "attachments": [{"attachment_id": attachment_id, "marker": "login-error"}],
        },
        "repositories": [
            {
                "repository_url": "https://github.com/example/backend",
                "default_branch": "main",
                "required": True,
            }
        ],
        "execution_mode": "mock",
    }


# ------------------------------------------------------------------ unbound and forgotten


@pytest.mark.asyncio
async def test_the_periodic_pass_sweeps_forgotten_uploads_and_logs_the_table() -> None:
    """Somebody who uploads three images and closes the tab must not leave them forever."""
    store = DatabaseAttachmentStore(EphemeralDatabase())
    png = fixtures.content(fixtures.PNG)
    forgotten = await store.create(
        owner_id=OWNER, filename="forgotten.png", media_type="image/png", content=png
    )
    recent = await store.create(
        owner_id=OWNER, filename="recent.png", media_type="image/png", content=png
    )
    submitted = await store.create(
        owner_id=OWNER, filename="submitted.png", media_type="image/png", content=png
    )
    await store.bind([submitted.attachment_id], feature_id="feature-1")
    await _age(store, forgotten.attachment_id, seconds=UNBOUND_ATTACHMENT_TTL_SECONDS + 60)

    journal = ExternalOperationJournal(EphemeralDatabase())
    service = RecoveryService(journal, attachments=store)
    with _captured_logs() as entries:
        await service.recover_incomplete_operations()

    assert await store.get_metadata(forgotten.attachment_id) is None
    assert await store.get_metadata(recent.attachment_id) is not None
    assert await store.get_metadata(submitted.attachment_id) is not None
    # The accounting: bound and unbound bytes counted apart, and what was deleted.
    accounted = [item for item in entries if item.get("event") == "prd_attachment_storage"]
    assert accounted == [
        {
            "event": "prd_attachment_storage",
            "rows_purged": 1,
            "bound_bytes": len(png),
            "unbound_bytes": len(png),
            "log_level": "info",
        }
    ]


@pytest.mark.asyncio
async def test_the_accounting_logs_on_change_and_not_on_every_tick() -> None:
    """A table holding the same bytes is not news thirty seconds later.

    The sweep runs on the recovery loop's interval, so a line per tick would restate the same
    number a couple of thousand times a day. It fires when something was deleted or the
    (bound, unbound) pair moved, and stays quiet across a steady table -- the last line
    logged is always the current state.
    """
    store = DatabaseAttachmentStore(EphemeralDatabase())
    png = fixtures.content(fixtures.PNG)
    await store.create(owner_id=OWNER, filename="held.png", media_type="image/png", content=png)
    journal = ExternalOperationJournal(EphemeralDatabase())
    service = RecoveryService(journal, attachments=store)

    with _captured_logs() as entries:
        # First tick: the pair moves from nothing held to one unbound upload -- news.
        await service.recover_incomplete_operations()
        # Second and third ticks: nothing deleted, nothing moved -- silence.
        await service.recover_incomplete_operations()
        await service.recover_incomplete_operations()
        # The table changes again: a second upload arrives.
        await store.create(
            owner_id=OWNER, filename="second.png", media_type="image/png", content=png
        )
        await service.recover_incomplete_operations()

    accounted = [item for item in entries if item.get("event") == "prd_attachment_storage"]
    assert [item["unbound_bytes"] for item in accounted] == [len(png), 2 * len(png)]
    assert all(item["rows_purged"] == 0 for item in accounted)


@pytest.mark.asyncio
async def test_the_sweep_is_silent_when_the_table_is_empty() -> None:
    """Nothing held and nothing deleted is not news, and a line per tick would drown the log."""
    store = DatabaseAttachmentStore(EphemeralDatabase())
    journal = ExternalOperationJournal(EphemeralDatabase())

    with _captured_logs() as entries:
        await RecoveryService(journal, attachments=store).recover_incomplete_operations()

    assert [item for item in entries if item.get("event") == "prd_attachment_storage"] == []


@pytest.mark.asyncio
async def test_a_failing_sweep_does_not_stop_the_operation_recovery_it_runs_beside() -> None:
    """The isolation the two sweeps beside it already have, for the same reason."""

    class Broken:
        async def sweep_unbound_before(self, cutoff: datetime) -> Any:
            del cutoff
            msg = "the store is unreachable"
            raise RuntimeError(msg)

    journal = ExternalOperationJournal(EphemeralDatabase())
    service = RecoveryService(journal, attachments=Broken())

    with _captured_logs() as entries:
        summary = await service.recover_incomplete_operations()

    # The pass completed and reported, and the fault is logged as its own category.
    assert summary.scanned == 0
    failures = [item for item in entries if item.get("event") == "prd_attachment_sweep_failed"]
    assert failures and failures[0]["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_a_composition_with_no_attachment_store_sweeps_nothing() -> None:
    """The honest behaviour for a deployment that keeps no bytes."""
    journal = ExternalOperationJournal(EphemeralDatabase())
    with _captured_logs() as entries:
        await RecoveryService(journal).recover_incomplete_operations()
    assert [item for item in entries if "attachment" in str(item.get("event", ""))] == []


# ------------------------------------------------------------------------------- bound


@pytest.mark.asyncio
async def test_retiring_a_feature_purges_content_keeps_metadata_and_answers_410() -> None:
    """The record of what was submitted outlives the thing submitted, and says so."""
    app = create_app(platform_api_key=KEY, attachments=InMemoryAttachmentStore())
    store = app.state.attachments
    record = await store.create(
        owner_id=OWNER,
        filename="login-error.png",
        media_type="image/png",
        content=fixtures.content(fixtures.PNG),
    )

    async with client(app) as http:
        created = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "h1"},
            json=payload(record.attachment_id),
        )
        await settle(app)
        before = await http.get(f"/attachments/{record.attachment_id}", headers=AUTH)
        retired = await http.post(
            "/features/feature-retire/retire",
            headers=AUTH,
            json={"reason": "superseded", "operator": "an operator"},
        )
        after = await http.get(f"/attachments/{record.attachment_id}", headers=AUTH)
        artifacts = await http.get(
            "/features/feature-retire/artifacts",
            headers=AUTH,
            params={"artifact_type": "prd"},
        )
        artifact_id = artifacts.json()["artifacts"][-1]["artifact_id"]
        prd = await http.get(f"/features/feature-retire/artifacts/{artifact_id}", headers=AUTH)

    assert created.status_code == 201, created.text
    assert before.status_code == 200
    assert retired.status_code == 200, retired.text
    # Gone, and saying so: "gone" and "never was" are different answers to somebody
    # following a reference out of an artifact.
    assert after.status_code == 410
    # And the artifact still names the file, the size and the hash.
    recorded = prd.json()["payload"]["attachments"][0]
    assert recorded["filename"] == "login-error.png"
    assert recorded["byte_size"] == len(fixtures.content(fixtures.PNG))
    assert recorded["sha256"] == fixtures.digest(fixtures.PNG)
    metadata = await store.get_metadata(record.attachment_id)
    assert metadata is not None
    assert metadata.content_present is False
    assert metadata.purged_at is not None


@pytest.mark.asyncio
async def test_a_second_retire_of_the_same_feature_is_harmless() -> None:
    """Retiring twice is already idempotent; purging twice must be too."""
    app = create_app(platform_api_key=KEY, attachments=InMemoryAttachmentStore())
    record = await app.state.attachments.create(
        owner_id=OWNER,
        filename="login.png",
        media_type="image/png",
        content=fixtures.content(fixtures.PNG),
    )

    async with client(app) as http:
        await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "h2"},
            json=payload(record.attachment_id),
        )
        await settle(app)
        first = await http.post(
            "/features/feature-retire/retire",
            headers=AUTH,
            json={"reason": "superseded", "operator": "an operator"},
        )
        second = await http.post(
            "/features/feature-retire/retire",
            headers=AUTH,
            json={"reason": "superseded again", "operator": "an operator"},
        )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    metadata = await app.state.attachments.get_metadata(record.attachment_id)
    assert metadata is not None
    assert metadata.content_present is False


@pytest.mark.asyncio
async def test_a_purge_that_fails_does_not_fail_the_retirement() -> None:
    """An operator told a feature is retired must not read an error about its images."""
    from storage.feature_store import SqlAlchemyFeatureControlPlane

    class Broken(InMemoryAttachmentStore):
        async def purge_content_for_feature(self, feature_id: str) -> int:
            del feature_id
            msg = "the store is unreachable"
            raise RuntimeError(msg)

    database = EphemeralDatabase()
    store = Broken()
    plane = SqlAlchemyFeatureControlPlane(database, attachments=store)
    app = create_app(platform_api_key=KEY, attachments=store, feature_control_plane=plane)
    record = await store.create(
        owner_id=OWNER,
        filename="login.png",
        media_type="image/png",
        content=fixtures.content(fixtures.PNG),
    )

    async with client(app) as http:
        created = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "h3"},
            json=payload(record.attachment_id),
        )
        await settle(app)
        with _captured_logs() as entries:
            retired = await http.post(
                "/features/feature-retire/retire",
                headers=AUTH,
                json={"reason": "superseded", "operator": "an operator"},
            )

    assert created.status_code == 201, created.text
    assert retired.status_code == 200, retired.text
    failures = [
        item for item in entries if item.get("event") == "retired_feature_attachment_purge_failed"
    ]
    assert failures and failures[0]["error_type"] == "RuntimeError"
    # The bytes stay until somebody retires the feature again; nothing pretends otherwise.
    metadata = await store.get_metadata(record.attachment_id)
    assert metadata is not None
    assert metadata.content_present is True


async def _age(store: DatabaseAttachmentStore, attachment_id: str, *, seconds: float) -> None:
    """Move one row's clock back, which is the only way to test a time-based sweep."""
    async with store._database.session() as session:
        model = await session.get(PRDAttachmentModel, attachment_id)
        assert model is not None
        model.created_at = datetime.now(UTC) - timedelta(seconds=seconds)
        await session.commit()


class _captured_logs:
    """Collect structlog events for the duration of a block.

    The log line *is* the accounting -- there is no operator storage surface to assert
    against -- so it is asserted as a value rather than trusted to have been emitted.
    """

    def __init__(self) -> None:
        """Start with nothing captured."""
        self._capture = structlog.testing.LogCapture()

    def __enter__(self) -> list[MutableMapping[str, Any]]:
        """Swap in a capturing logger factory and hand back the list it fills."""
        self._previous = structlog.get_config()
        structlog.configure(
            processors=[self._capture],
            logger_factory=structlog.ReturnLoggerFactory(),
        )
        return self._capture.entries

    def __exit__(self, *exception: object) -> None:
        """Put the process's own configuration back."""
        structlog.configure(**self._previous)
