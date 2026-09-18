"""The durable mechanisms, executed against the database the deployment actually runs on.

Audit risk P1-8, first half. Every durable test in this suite ran on SQLite, and the things
that keep two workers from running the same feature are precisely the things SQLite does not
have. `DatabaseFeatureExecutionQueue.claim` skips `FOR UPDATE SKIP LOCKED` on SQLite by an
explicit dialect check. `_replace_state`'s `SELECT ... FOR UPDATE` is accepted and ignored
there. Foreign keys are not enforced there at all unless a pragma asks. None of that was ever
executed by a test, so none of it was ever proven to work.

Every test here is marked `postgres` and gets its own database, created from the real Alembic
migrations at head -- see `tests/postgres_support.py`. Two tests are deliberately *not*
marked: they run the same scenario on SQLite and assert the wrong answer it gives, so the
difference this tier exists for is written down rather than asserted about.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import HttpUrl
from sqlalchemy import select, text, update

from api.control_plane import RequestScopedCredentials
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import PullRequestArtifact
from services.feature_queue import (
    CONTINUE,
    QUEUED,
    RUNNING,
    START,
    SUCCEEDED,
    DatabaseFeatureExecutionQueue,
    FeatureQueueDispatcher,
)
from state.enums import ChildWorkflowStatus, FeatureActionStatus, FeatureWorkflowStatus
from state.external_operations import ExternalOperationStatus, ExternalOperationType
from state.feature_models import ChildWorkflowReference
from storage.action_store import DatabaseFeatureActionStore
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from storage.feature_store import ClaimDisposition, SqlAlchemyFeatureControlPlane
from storage.models import (
    ExternalOperationModel,
    FeatureActionModel,
    FeatureExecutionQueueModel,
    FeatureWorkflowModel,
)
from tests.postgres_support import migration_head
from tests.test_feature_api import feature_payload
from workflows.feature_workflow import FeatureWorkflowOrchestrator

pytestmark = pytest.mark.asyncio

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)


def _control_plane(database: Database) -> SqlAlchemyFeatureControlPlane:
    """Build the control plane the deployment builds, minus the live runner."""
    return SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())


async def _accept_feature(store: SqlAlchemyFeatureControlPlane, key: str) -> str:
    """Accept one feature through the real path, so its queue row and its feature row agree.

    Written through `start` rather than by inserting rows: the queue's foreign key to
    `feature_workflows` is enforced on PostgreSQL and silently ignored on SQLite, so a test
    that fabricated queue rows would be testing something the deployment cannot reach.
    """
    request = StartFeatureRequest.model_validate({**feature_payload(), "feature_id": f"f-{key}"})
    result = await store.start(
        request, idempotency_key=key, credentials=CREDENTIALS, owner_id="platform-admin"
    )
    return result.record.state.feature_id


async def _queue_row(database: Database, feature_id: str) -> FeatureExecutionQueueModel:
    """Read one queue entry as a later reader sees it."""
    async with database.session() as session:
        row = await session.get(FeatureExecutionQueueModel, feature_id)
        assert row is not None
        return row


@pytest.fixture
async def sqlite_database(tmp_path: Path) -> AsyncIterator[Database]:
    """The database the rest of the suite uses, for the two contrast tests below."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'contrast.db'}")
    await database.create_schema()
    try:
        yield database
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# The tier itself
# --------------------------------------------------------------------------------------


@pytest.mark.postgres
async def test_the_tier_runs_on_postgres_with_the_schema_the_migrations_produce(
    postgres_database: Database,
) -> None:
    """Without this, a misconfigured fixture would make the whole tier decorative.

    The revision check is the drift check the suite could not previously make.
    `test_migrations.py` proves the chain is well-formed and that every mapped table is
    created by some migration; only applying the chain proves the result is a schema the
    models can actually be used against.
    """
    assert postgres_database.engine.dialect.name == "postgresql"
    async with postgres_database.session() as session:
        recorded = await session.scalar(text("select version_num from alembic_version"))
        # A table the models describe, read through the schema the migrations built.
        await session.execute(select(FeatureWorkflowModel).limit(1))
    assert recorded == migration_head()


@pytest.mark.postgres
async def test_the_queues_foreign_key_to_its_feature_is_enforced(
    postgres_database: Database,
) -> None:
    """A queue row naming a feature that does not exist is work nothing can ever run.

    SQLite accepts it silently unless a pragma is set, and the suite never set one.
    """
    queue = DatabaseFeatureExecutionQueue(postgres_database)
    with pytest.raises(Exception) as raised:
        await queue.enqueue(
            feature_id="feature-that-was-never-created",
            execution_mode="mock",
            agent_platform="openai",
            requested_by="tier-test",
        )
    assert "foreign key" in str(raised.value).lower()


# --------------------------------------------------------------------------------------
# DatabaseFeatureExecutionQueue.claim -- the mechanism the audit named
# --------------------------------------------------------------------------------------


async def _claim_while_the_oldest_row_is_held(database: Database) -> tuple[str, str, str | None]:
    """Claim while another transaction holds a row lock on the oldest waiting entry.

    Returns the older feature, the newer one, and what a claim arriving during the lock got.
    This is the production shape of two workers draining one queue, made deterministic: the
    held lock stands in for the instant another worker is inside its own claim transaction.
    """
    store = _control_plane(database)
    older = await _accept_feature(store, "older")
    await asyncio.sleep(0.01)
    newer = await _accept_feature(store, "newer")
    async with database.session() as blocking:
        held = await blocking.scalar(
            select(FeatureExecutionQueueModel)
            .where(FeatureExecutionQueueModel.feature_id == older)
            .with_for_update()
        )
        assert held is not None
        claimed = await DatabaseFeatureExecutionQueue(database).claim(
            owner="second-worker", lease_seconds=60
        )
        await blocking.rollback()
    return older, newer, None if claimed is None else claimed.feature_id


@pytest.mark.postgres
async def test_a_queue_row_another_worker_holds_is_skipped_rather_than_claimed_twice(
    postgres_database: Database,
) -> None:
    """`SKIP LOCKED` is the whole reason two workers can drain one queue safely.

    This is the test the audit asked for: it fails on SQLite -- see the companion below --
    and passes here, which is what makes the tier worth its wall-clock.
    """
    older, newer, claimed = await _claim_while_the_oldest_row_is_held(postgres_database)

    assert claimed == newer, "the second worker should have moved past the held row"
    assert claimed != older
    # And the held row is untouched: nobody consumed an attempt on a feature they skipped.
    row = await _queue_row(postgres_database, older)
    assert (row.status, row.attempt, row.lease_owner) == (QUEUED, 0, None)


async def test_the_same_locked_queue_row_is_handed_straight_out_on_sqlite(
    sqlite_database: Database,
) -> None:
    """The contrast, written down rather than claimed.

    On SQLite `with_for_update` is accepted and ignored, and `claim` does not even ask for it.
    A second worker is handed the row the first is already inside -- the same feature run
    twice. Nothing about this is a defect in SQLite; it is the reason the durable mechanisms
    have to be exercised somewhere else.
    """
    older, _newer, claimed = await _claim_while_the_oldest_row_is_held(sqlite_database)

    assert claimed == older
    row = await _queue_row(sqlite_database, older)
    assert (row.status, row.attempt, row.lease_owner) == (RUNNING, 1, "second-worker")


@pytest.mark.postgres
async def test_four_workers_racing_for_one_entry_produce_exactly_one_claim(
    postgres_database: Database,
) -> None:
    """The property the queue exists for, asserted on the effect rather than on a call count."""
    store = _control_plane(postgres_database)
    feature_id = await _accept_feature(store, "contended")
    queue = DatabaseFeatureExecutionQueue(postgres_database)

    claims = await asyncio.gather(
        *(queue.claim(owner=f"worker-{index}", lease_seconds=60) for index in range(4))
    )

    won = [claim for claim in claims if claim is not None]
    assert len(won) == 1
    assert won[0].feature_id == feature_id
    assert won[0].attempt == 1
    row = await _queue_row(postgres_database, feature_id)
    assert row.status == RUNNING
    assert row.attempt == 1, "a losing worker must not consume an attempt it never ran"
    assert row.lease_owner in {f"worker-{index}" for index in range(4)}


@pytest.mark.postgres
async def test_a_lapsed_lease_is_redelivered_and_a_live_one_is_not(
    postgres_database: Database,
) -> None:
    """At-least-once delivery: a dead worker's claim comes back, a live worker's does not."""
    store = _control_plane(postgres_database)
    feature_id = await _accept_feature(store, "lapsing")
    queue = DatabaseFeatureExecutionQueue(postgres_database)
    first = await queue.claim(owner="worker-a", lease_seconds=600)
    assert first is not None

    assert await queue.claim(owner="worker-b", lease_seconds=600) is None

    async with postgres_database.session() as session:
        await session.execute(
            update(FeatureExecutionQueueModel)
            .where(FeatureExecutionQueueModel.feature_id == feature_id)
            .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        await session.commit()
    second = await queue.claim(owner="worker-b", lease_seconds=600)
    assert second is not None
    assert second.feature_id == feature_id
    assert second.attempt == 2, "redelivery spends an attempt, which is what bounds it"


@pytest.mark.postgres
async def test_renew_extends_only_the_holders_lease(postgres_database: Database) -> None:
    """A worker that lost its claim must not be able to keep it alive."""
    store = _control_plane(postgres_database)
    feature_id = await _accept_feature(store, "renewing")
    queue = DatabaseFeatureExecutionQueue(postgres_database)
    assert await queue.claim(owner="holder", lease_seconds=60) is not None
    before = (await _queue_row(postgres_database, feature_id)).lease_expires_at
    assert before is not None

    await queue.renew(feature_id, owner="not-the-holder", lease_seconds=6_000)
    unchanged = (await _queue_row(postgres_database, feature_id)).lease_expires_at
    await queue.renew(feature_id, owner="holder", lease_seconds=6_000)
    extended = (await _queue_row(postgres_database, feature_id)).lease_expires_at

    assert unchanged == before
    assert extended is not None and extended > before


@pytest.mark.postgres
async def test_release_returns_the_entry_and_defer_gives_back_the_attempt(
    postgres_database: Database,
) -> None:
    """The two ways a claim ends without finishing, and the difference between them.

    `defer` exists because a claim arriving while the work is genuinely in progress must not
    spend an attempt -- three of those in thirty seconds retired AB-Feature-104 while its
    first attempt was still working.
    """
    store = _control_plane(postgres_database)
    feature_id = await _accept_feature(store, "releasing")
    queue = DatabaseFeatureExecutionQueue(postgres_database)
    assert await queue.claim(owner="worker", lease_seconds=60) is not None

    await queue.release(feature_id, error="worker restarted")
    released = await _queue_row(postgres_database, feature_id)
    assert (released.status, released.attempt, released.lease_owner) == (QUEUED, 1, None)
    assert released.last_error == "worker restarted"

    assert await queue.claim(owner="worker", lease_seconds=60) is not None
    await queue.defer(feature_id, error="already executing")
    deferred = await _queue_row(postgres_database, feature_id)
    assert (deferred.status, deferred.attempt, deferred.lease_owner) == (QUEUED, 1, None)


@pytest.mark.postgres
async def test_requeue_reopens_a_settled_entry_with_a_fresh_budget_and_a_later_turn(
    postgres_database: Database,
) -> None:
    """One row per feature for its whole life, so a resume rewrites rather than duplicates."""
    store = _control_plane(postgres_database)
    feature_id = await _accept_feature(store, "requeued")
    queue = DatabaseFeatureExecutionQueue(postgres_database)
    assert await queue.claim(owner="worker", lease_seconds=60) is not None
    await queue.finish(feature_id, succeeded=True)
    finished = await _queue_row(postgres_database, feature_id)
    assert finished.status == SUCCEEDED
    queued_at_before = finished.queued_at

    await queue.requeue(
        feature_id=feature_id,
        execution_mode="mock",
        agent_platform="openai",
        intent="resume",
        payload={"answers": ["yes"]},
        requested_by="operator",
    )

    async with postgres_database.session() as session:
        rows = list(
            await session.scalars(
                select(FeatureExecutionQueueModel).where(
                    FeatureExecutionQueueModel.feature_id == feature_id
                )
            )
        )
    assert len(rows) == 1, "a feature has exactly one queue entry over its whole life"
    reopened = rows[0]
    assert (reopened.status, reopened.attempt, reopened.intent) == (QUEUED, 0, "resume")
    assert reopened.intent_payload == {"answers": ["yes"]}
    assert reopened.finished_at is None and reopened.started_at is None
    assert reopened.queued_at > queued_at_before, "a reopened entry takes its turn at the back"
    assert await queue.active_intent(feature_id) == "resume"


# --------------------------------------------------------------------------------------
# ExternalOperationJournal
# --------------------------------------------------------------------------------------


async def _create(journal: ExternalOperationJournal, key: str, **overrides: Any) -> Any:
    """Create one journal operation with the fields every caller supplies."""
    fields: dict[str, Any] = {
        "workflow_id": "workflow-tier",
        "feature_id": "feature-tier",
        "child_workflow_id": None,
        "repository_id": "backend",
        "operation_type": ExternalOperationType.PUSH_BRANCH,
        "idempotency_key": key,
        "input_fingerprint": "fingerprint-1",
        "max_attempts": 3,
        "safe_metadata": {"logical_step": "publish"},
    }
    fields.update(overrides)
    return await journal.create_operation(**fields)


@pytest.mark.postgres
async def test_concurrent_creates_with_one_idempotency_key_produce_one_operation(
    postgres_database: Database,
) -> None:
    """The uniqueness that stops a duplicated push is the database's, not a prior read's.

    Six concurrent callers all find nothing, all insert, and exactly one row survives. On
    SQLite the writers are serialised by the file lock, so the race the constraint exists for
    never actually happens.
    """
    journal = ExternalOperationJournal(postgres_database)

    created = await asyncio.gather(
        *(_create(journal, "one-key") for _ in range(6)), return_exceptions=True
    )

    raised = [item for item in created if isinstance(item, BaseException)]
    assert raised == [], f"a concurrent create raised instead of adopting the winner: {raised}"
    identifiers = {item.operation_id for item in created if not isinstance(item, BaseException)}
    assert len(identifiers) == 1
    async with postgres_database.session() as session:
        rows = list(
            await session.scalars(
                select(ExternalOperationModel).where(
                    ExternalOperationModel.idempotency_key == "one-key"
                )
            )
        )
    assert len(rows) == 1


@pytest.mark.postgres
async def test_only_one_sweep_claims_a_stale_operation(postgres_database: Database) -> None:
    """Two recovery passes seeing the same stale row must not both fence it.

    The claim is a conditional update on status and attempt together, which is a
    compare-and-set. Under real concurrency exactly one update reports a row.
    """
    journal = ExternalOperationJournal(postgres_database, stale_after_seconds=1.0)
    operation = await _create(journal, "stale-key")
    claimed = await journal.claim_operation(operation.operation_id)
    async with postgres_database.session() as session:
        long_ago = datetime.now(UTC) - timedelta(minutes=30)
        await session.execute(
            update(ExternalOperationModel)
            .where(ExternalOperationModel.operation_id == operation.operation_id)
            .values(heartbeat_at=long_ago, started_at=long_ago, updated_at=long_ago)
        )
        await session.commit()
    stale = await journal.get(operation.operation_id)
    assert stale.status is claimed.status

    outcomes = await asyncio.gather(*(journal.claim_stale_operation(stale) for _ in range(4)))

    assert sum(1 for won in outcomes if won) == 1
    settled = await journal.get(operation.operation_id)
    assert settled.status is ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE
    assert settled.error_code == "stale_recovery_claimed"


@pytest.mark.postgres
async def test_a_recorded_result_survives_the_round_trip_with_aware_timestamps(
    postgres_database: Database,
) -> None:
    """Timestamps come back timezone-aware, which SQLite's text columns cannot prove.

    Every staleness decision in the platform subtracts two of these. A naive value read back
    from a `timestamptz` column would raise on the subtraction, in a recovery sweep, in
    production.
    """
    journal = ExternalOperationJournal(postgres_database)
    operation = await _create(journal, "result-key")
    await journal.claim_operation(operation.operation_id)

    completed = await journal.record_result(
        operation.operation_id,
        external_reference="0" * 40,
        result_payload={"pushed": True},
    )

    assert completed.status is ExternalOperationStatus.SUCCEEDED
    assert completed.external_reference == "0" * 40
    assert completed.result_payload == {"pushed": True}
    for stamp in (completed.started_at, completed.heartbeat_at, completed.completed_at):
        assert stamp is not None and stamp.tzinfo is not None
    assert completed.completed_at is not None
    assert (datetime.now(UTC) - completed.completed_at) >= timedelta(0)
    events = await journal.events_for_operation(operation.operation_id)
    assert [event.new_status for event in events][-1] is ExternalOperationStatus.SUCCEEDED


# --------------------------------------------------------------------------------------
# SqlAlchemyFeatureControlPlane._replace_state and _merge_durable_progress
# --------------------------------------------------------------------------------------


async def _replace_state_while_the_feature_row_is_held(
    database: Database, *, timeout: float
) -> tuple[str, bool]:
    """Ask whether a state write waits for a row lock another transaction holds.

    Returns the feature and whether the write completed while the lock was held. This is the
    only way to exercise `_replace_state`'s `SELECT ... FOR UPDATE` from one process: the
    in-process lock above it is keyed by the database URL, so two control planes over the
    same database serialise on that instead and never reach the row lock at all.
    """
    store = _control_plane(database)
    feature_id = await _accept_feature(store, "locked-state")
    state = (await store.get_record(feature_id)).state.model_copy(deep=True)
    state.current_agent = "second-writer"
    async with database.session() as blocking:
        held = await blocking.scalar(
            select(FeatureWorkflowModel)
            .where(FeatureWorkflowModel.feature_id == feature_id)
            .with_for_update()
        )
        assert held is not None
        try:
            await asyncio.wait_for(
                store._replace_state(feature_id, state, "tier_test"),  # noqa: SLF001
                timeout=timeout,
            )
            completed = True
        except TimeoutError:
            completed = False
        await blocking.rollback()
    return feature_id, completed


@pytest.mark.postgres
async def test_a_state_write_waits_for_the_row_lock_another_writer_holds(
    postgres_database: Database,
) -> None:
    """Two control planes in different processes are serialised by the row, not by Redis.

    Redis is the first line of defence and it is not the last one. The comment above the
    `FOR UPDATE` says it serialises correctly configured PostgreSQL writers even when they
    are reached through different control-plane instances; until this tier existed, nothing
    had ever run that statement.
    """
    _feature_id, completed_while_locked = await _replace_state_while_the_feature_row_is_held(
        postgres_database, timeout=1.0
    )

    assert not completed_while_locked, "the write went through while another writer held the row"


async def test_the_same_state_write_is_not_delayed_at_all_on_sqlite(
    sqlite_database: Database,
) -> None:
    """The contrast: SQLite accepts `FOR UPDATE` and ignores it, so nothing is serialised."""
    _feature_id, completed_while_locked = await _replace_state_while_the_feature_row_is_held(
        sqlite_database, timeout=5.0
    )

    assert completed_while_locked


@pytest.mark.postgres
async def test_durable_progress_survives_a_write_that_did_not_know_about_it(
    postgres_database: Database,
) -> None:
    """The merge is what keeps a stale in-flight snapshot from erasing finished work.

    Asserted through the real JSON round trip, because the column is `JSONB` here and text on
    SQLite, and an artifact list is the thing that round trip has to preserve in order.
    """
    store = _control_plane(postgres_database)
    feature_id = await _accept_feature(store, "merging")
    fresh = (await store.get_record(feature_id)).state.model_copy(deep=True)

    advanced = fresh.model_copy(deep=True)
    advanced.child_workflows = {
        "backend": ChildWorkflowReference(
            child_workflow_id=f"{feature_id}:backend",
            repository_id="backend",
            workstream_id="ws-backend",
            status=ChildWorkflowStatus.APPROVED,
            branch_name="feature/audit",
            workspace_path=f"/workspaces/{feature_id}/backend",
            retry_count=1,
        )
    }
    advanced.artifacts = [
        *fresh.artifacts,
        PullRequestArtifact(
            schema_version="1.0",
            workflow_id=feature_id,
            artifact_id="artifact-tier",
            producer="publisher",
            timestamp=datetime.now(UTC),
            metadata={"source": "tier-test"},
            validation_status="valid",
            repository="example/backend",
            pull_request_number=7,
            url=HttpUrl("https://github.invalid/example/backend/pull/7"),
            title="Login audit trail",
            body="Opened by the durable tier test.",
            source_branch="feature/audit",
            target_branch="main",
            commit_sha="a" * 40,
            labels=["automated"],
            reviewers=[],
            state="open",
        ),
    ]
    await store._replace_state(feature_id, advanced, "workstream_approved")  # noqa: SLF001

    # A writer holding the snapshot from before that progress, as a slow coroutine does.
    stale = fresh.model_copy(deep=True)
    stale.current_agent = "late_writer"
    await store._replace_state(feature_id, stale, "late_write")  # noqa: SLF001

    record = await store.get_record(feature_id)
    assert record.state.child_workflows["backend"].status is ChildWorkflowStatus.APPROVED
    assert [item.artifact_id for item in record.state.artifacts][-1] == "artifact-tier"
    restored = record.state.artifacts[-1]
    assert isinstance(restored, PullRequestArtifact)
    assert (restored.pull_request_number, restored.commit_sha) == (7, "a" * 40)
    assert restored.timestamp.tzinfo is not None


# --------------------------------------------------------------------------------------
# DatabaseFeatureActionStore
# --------------------------------------------------------------------------------------


async def _confirmed_action(store: DatabaseFeatureActionStore, feature_id: str) -> str:
    """Record one confirmed action the way a Confirm button does."""
    action, _created = await store.create_or_get(
        feature_id=feature_id,
        repository_id="backend",
        action_type="RETRY_WORKSTREAM",
        actor_id="operator",
        actor_display_name="Operator",
        origin="rest",
        origin_message_id=None,
        payload={"reason": "flaky validation"},
        context_version="v1",
        max_attempts=1,
    )
    return action.action_id


@pytest.mark.postgres
async def test_a_double_clicked_confirmation_produces_one_action(
    postgres_database: Database,
) -> None:
    """Concurrent identical confirmations insert, race, and settle on one row."""
    control_plane = _control_plane(postgres_database)
    feature_id = await _accept_feature(control_plane, "actions")
    store = DatabaseFeatureActionStore(postgres_database)

    results = await asyncio.gather(
        *(_confirmed_action(store, feature_id) for _ in range(5)), return_exceptions=True
    )

    raised = [item for item in results if isinstance(item, BaseException)]
    assert raised == [], f"a concurrent confirmation raised: {raised}"
    assert len(set(results)) == 1
    assert len(await store.list_for_feature(feature_id)) == 1


@pytest.mark.postgres
async def test_only_one_worker_holds_an_action_lease_and_recovery_waits_for_it_to_lapse(
    postgres_database: Database,
) -> None:
    """The lease is what stops two workers running the same retry, and it is a row update."""
    control_plane = _control_plane(postgres_database)
    feature_id = await _accept_feature(control_plane, "action-lease")
    store = DatabaseFeatureActionStore(postgres_database, lease_seconds=600)
    action_id = await _confirmed_action(store, feature_id)

    claims = await asyncio.gather(
        *(store.claim(action_id, owner=f"worker-{index}") for index in range(4))
    )
    won = [claim for claim in claims if claim is not None]

    assert len(won) == 1
    assert won[0].status is FeatureActionStatus.CLAIMED
    assert won[0].attempt == 1
    # Nothing to recover while the lease is live, whatever a sweep thinks it saw.
    assert await store.list_expired() == []
    assert (
        await store.reconcile(
            action_id,
            status=FeatureActionStatus.FAILED,
            error_code="interrupted_before_completion",
            error_message="the executor stopped",
        )
        is None
    )

    async with postgres_database.session() as session:
        await session.execute(
            text(
                "update feature_actions set lease_expires_at = :expired "
                "where action_id = :action_id"
            ),
            {"expired": datetime.now(UTC) - timedelta(seconds=1), "action_id": action_id},
        )
        await session.commit()

    expired = await store.list_expired()
    assert [item.action_id for item in expired] == [action_id]
    reconciled = await store.reconcile(
        action_id,
        status=FeatureActionStatus.FAILED,
        error_code="interrupted_before_completion",
        error_message="the executor stopped",
    )
    assert reconciled is not None
    assert reconciled.status is FeatureActionStatus.FAILED
    assert reconciled.lease_owner is None and reconciled.lease_expires_at is None
    assert await store.renew_lease(action_id, owner=won[0].lease_owner or "") is False


# --------------------------------------------------------------------------------------
# reconcile_abandoned_runs
# --------------------------------------------------------------------------------------


@pytest.mark.postgres
async def test_an_abandoned_run_is_reconciled_and_a_leased_one_is_left_alone(
    postgres_database: Database,
) -> None:
    """The sweep's fence is a `NOT IN (subquery)`, and NULL semantics differ.

    A `NOT IN` whose subquery yields a NULL is empty on PostgreSQL -- so a single row with a
    null `feature_id` anywhere in the fence would silently make the sweep decide nothing at
    all. That is a live-database behaviour, not a model behaviour, which is why the sweep
    belongs in this tier. Task 28- made the fence one `UNION` of the three sources rather
    than three separate clauses, because the claim decision asks the same question about one
    feature and the two must not drift; a NULL in any branch now empties the whole fence, so
    this test matters more than it did.
    """
    store = _control_plane(postgres_database)
    abandoned = await _accept_feature(store, "abandoned")
    leased = await _accept_feature(store, "leased")
    stale = datetime.now(UTC) - timedelta(hours=2)
    for feature_id in (abandoned, leased):
        state = (await store.get_record(feature_id)).state.model_copy(deep=True)
        state.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
        state.child_workflows = {
            "backend": ChildWorkflowReference(
                child_workflow_id=f"{feature_id}:backend",
                repository_id="backend",
                workstream_id="ws-backend",
                status=ChildWorkflowStatus.RUNNING,
                branch_name="feature/audit",
                workspace_path=f"/workspaces/{feature_id}/backend",
                retry_count=0,
            )
        }
        state.updated_at = stale
        await store._replace_state(feature_id, state, "tier_setup")  # noqa: SLF001
        state_json = dict(state.model_dump(mode="json"))
        state_json["updated_at"] = stale.isoformat()
        async with postgres_database.session() as session:
            await session.execute(
                update(FeatureWorkflowModel)
                .where(FeatureWorkflowModel.feature_id == feature_id)
                .values(
                    status=FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS.value,
                    updated_at=stale,
                    state_json=state_json,
                )
            )
            # The abandoned one's entry is settled; the leased one is genuinely held.
            await session.execute(
                update(FeatureExecutionQueueModel)
                .where(FeatureExecutionQueueModel.feature_id == feature_id)
                .values(
                    status=RUNNING,
                    lease_owner="worker",
                    lease_expires_at=(
                        datetime.now(UTC) + timedelta(minutes=10) if feature_id == leased else stale
                    ),
                    started_at=stale,
                )
            )
            await session.commit()

    reconciled = await store.reconcile_abandoned_runs(stale_after_seconds=60)

    assert reconciled == [abandoned], "a live lease is not an abandoned run"
    decided = await store.get_record(abandoned)
    assert decided.state.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert decided.state.child_workflows["backend"].status is ChildWorkflowStatus.FAILED
    untouched = await store.get_record(leased)
    assert untouched.state.status is FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS


# --------------------------------------------------------------------------------------
# continuing a crashed run
# --------------------------------------------------------------------------------------


async def _crashed_run(store: SqlAlchemyFeatureControlPlane, database: Database, key: str) -> str:
    """Leave one feature in the shape a killed worker leaves: executing, claim lapsed."""
    feature_id = await _accept_feature(store, key)
    state = (await store.get_record(feature_id)).state.model_copy(deep=True)
    state.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
    state.child_workflows = {
        "backend": ChildWorkflowReference(
            child_workflow_id=f"{feature_id}:backend",
            repository_id="backend",
            workstream_id="ws-backend",
            status=ChildWorkflowStatus.RUNNING,
            branch_name="feature/audit",
            workspace_path=f"/workspaces/{feature_id}/backend",
            retry_count=1,
        )
    }
    await store._replace_state(feature_id, state, "child_workflows_started")  # noqa: SLF001
    lapsed = datetime.now(UTC) - timedelta(minutes=5)
    async with database.session() as session:
        await session.execute(
            update(FeatureExecutionQueueModel)
            .where(FeatureExecutionQueueModel.feature_id == feature_id)
            .values(
                status=RUNNING,
                attempt=1,
                lease_owner="worker-that-died",
                lease_expires_at=lapsed,
                started_at=lapsed,
            )
        )
        await session.commit()
    return feature_id


@pytest.mark.postgres
async def test_a_crashed_run_is_continued_by_exactly_one_of_four_racing_workers(
    postgres_database: Database,
) -> None:
    """A start claim on an executing feature now does work, so the race matters more.

    While the claim was dropped as unwanted, two workers arriving on a crashed feature both
    did nothing and the duplicate cost nothing. A claim that continues the run is a claim
    that codes, validates and publishes, so `FOR UPDATE SKIP LOCKED` -- which SQLite does not
    have and this tier exists to execute -- is what stops two of them doing it at once.
    """
    store = _control_plane(postgres_database)
    feature_id = await _crashed_run(store, postgres_database, "crashed")
    queue = DatabaseFeatureExecutionQueue(postgres_database)

    claims = await asyncio.gather(
        *(queue.claim(owner=f"worker-{index}", lease_seconds=60) for index in range(4))
    )

    won = [claim for claim in claims if claim is not None]
    assert len(won) == 1, "two workers would have continued the same crashed feature"
    assert (won[0].feature_id, won[0].intent, won[0].attempt) == (feature_id, START, 2)
    # And the claim that won is one this feature wants: the disposition is a continuation,
    # established from the evidence the sweep uses, not assumed from the status alone.
    assert (
        await store._claim_disposition(  # noqa: SLF001
            (await store.get_record(feature_id)).state, START
        )
        is ClaimDisposition.CONTINUE
    )


@pytest.mark.postgres
async def test_a_claim_holding_the_only_queue_row_still_reads_nothing_as_executing(
    postgres_database: Database,
) -> None:
    """The claim excludes its own row and nothing else, which is the whole subtlety.

    The queue's primary key is the feature, so a worker executing one sees exactly one queue
    row for it -- its own claim -- and taking that as proof somebody else is running would
    mean no crashed feature is ever continued. A live *action* lease is different evidence
    and must still stop it, and on PostgreSQL both are read through a `UNION` whose NULL
    semantics differ from SQLite's.
    """
    store = _control_plane(postgres_database)
    feature_id = await _crashed_run(store, postgres_database, "self-claim")
    queue = DatabaseFeatureExecutionQueue(postgres_database)
    claimed = await queue.claim(owner="worker-continuing", lease_seconds=60)
    assert claimed is not None

    state = (await store.get_record(feature_id)).state
    assert await store._claim_disposition(state, START) is ClaimDisposition.CONTINUE  # noqa: SLF001

    # A live action lease is somebody else executing, and it drops the claim instead.
    async with postgres_database.session() as session:
        session.add(
            FeatureActionModel(
                action_id=f"{feature_id}-live",
                feature_id=feature_id,
                repository_id=None,
                action_type="ANSWER_CLARIFICATION",
                actor_id="someone",
                origin="api",
                payload={},
                input_fingerprint="fingerprint",
                idempotency_key=f"{feature_id}-live",
                status="executing",
                attempt=1,
                max_attempts=1,
                lease_owner="live-worker",
                lease_expires_at=datetime.now(UTC) + timedelta(minutes=3),
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()

    assert await store._claim_disposition(state, START) is ClaimDisposition.DROP  # noqa: SLF001


# --------------------------------------------------------------------------------------
# step-based execution
# --------------------------------------------------------------------------------------


# Long enough to be a real grace period and short enough that a test does not wait it out.
_STALE = 0.01


async def _backdate(database: Database, feature_id: str) -> None:
    """Age one feature past the grace period, as the passage of time does."""
    async with database.session() as session:
        await session.execute(
            update(FeatureWorkflowModel)
            .where(FeatureWorkflowModel.feature_id == feature_id)
            .values(updated_at=datetime.now(UTC) - timedelta(minutes=5))
        )
        await session.commit()


class _StepRecordingOrchestrator(FeatureWorkflowOrchestrator):
    """The deterministic orchestrator, plus a record of every step it was asked to run."""

    def __init__(self) -> None:
        """Start with nothing run."""
        super().__init__()
        self.steps: list[str] = []

    async def _run_step(self, state: Any, step: Any, *, credentials: Any) -> Any:
        """Record this step, then run it exactly as the orchestrator would."""
        self.steps.append(f"{state.feature_id}:{step.value}")
        return await super()._run_step(state, step, credentials=credentials)


@pytest.mark.postgres
async def test_four_workers_draining_one_queue_run_each_step_exactly_once(
    postgres_database: Database,
) -> None:
    """A claim is one step, so the window two workers can collide in is entered constantly.

    A feature used to be claimed once and held for eighty-six minutes; it is now claimed once
    per step, which multiplies every race this tier exists for by the number of steps a
    feature takes. `FOR UPDATE SKIP LOCKED` -- skipped by an explicit dialect check on SQLite,
    and therefore never executed by the rest of the suite -- is what keeps that from meaning
    two workers coding the same repository.

    Running a step twice would not merely waste a call: the second one writes an artifact
    identifier the first one already used, which the immutable history refuses, and the
    feature ends on a platform error.
    """
    runner = _StepRecordingOrchestrator()
    store = SqlAlchemyFeatureControlPlane(postgres_database, mock_runner=runner)
    feature_id = await _accept_feature(store, "stepping-race")

    async def credentials_for(_owner: str) -> RequestScopedCredentials:
        return CREDENTIALS

    workers = [
        FeatureQueueDispatcher(queue=store.queue, executor=store, credentials_for=credentials_for)
        for _ in range(4)
    ]
    await asyncio.gather(*(worker.drain() for worker in workers))

    state = (await store.get_record(feature_id)).state
    assert state.status is FeatureWorkflowStatus.COMPLETED
    assert runner.steps == sorted(set(runner.steps), key=runner.steps.index), (
        f"a step ran more than once under four workers: {runner.steps}"
    )
    entry = await _queue_row(postgres_database, feature_id)
    assert entry.status == SUCCEEDED
    assert entry.steps == len(runner.steps) - 1


@pytest.mark.postgres
async def test_the_sweep_and_a_stepping_claim_still_cannot_both_take_one_feature(
    postgres_database: Database,
) -> None:
    """The fence between the sweep and a claim, re-asked at the rate stepping asks it.

    Both read the same evidence -- `_executing_feature_ids` -- and neither takes a lock the
    other waits on, so the two checks are coordinated only by what the rows say at the moment
    each of them looks. Task 28- tested both orderings at a microsecond window while a feature
    was claimed once; a feature is claimed once per step now, so that window is entered as
    many times as the feature has steps.

    The property is unchanged and still holds: whichever of them looks while the other holds a
    live lease sees it, so exactly one acts.
    """
    store = SqlAlchemyFeatureControlPlane(
        postgres_database, mock_runner=FeatureWorkflowOrchestrator()
    )
    feature_id = await _accept_feature(store, "stepping-fence")
    queue = DatabaseFeatureExecutionQueue(postgres_database)

    # A claim in flight between two steps: the entry is leased and the feature says it is
    # executing, which is exactly the shape an abandoned run also has.
    state = (await store.get_record(feature_id)).state.model_copy(deep=True)
    state.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
    await store._replace_state(feature_id, state, "child_workflows_started")  # noqa: SLF001
    await _backdate(postgres_database, feature_id)
    claimed = await queue.claim(owner="stepping-worker", lease_seconds=120)
    assert claimed is not None

    swept, disposition = await asyncio.gather(
        store.reconcile_abandoned_runs(stale_after_seconds=_STALE),
        store._claim_disposition(  # noqa: SLF001
            (await store.get_record(feature_id)).state, claimed.intent
        ),
    )

    assert swept == [], "the sweep tombstoned a feature a live claim was stepping"
    assert disposition is not ClaimDisposition.DROP
    assert (await store.get_record(feature_id)).state.status is (
        FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
    )

    # And the other ordering: once the claim's lease has lapsed, the sweep is the one that
    # acts and the next claim finds the feature already decided.
    await queue.advance_to_next_step(feature_id)
    async with postgres_database.session() as session:
        await session.execute(
            update(FeatureExecutionQueueModel)
            .where(FeatureExecutionQueueModel.feature_id == feature_id)
            .values(
                status=RUNNING,
                lease_owner="worker-that-died",
                lease_expires_at=datetime.now(UTC) - timedelta(minutes=5),
            )
        )
        await session.commit()

    await _backdate(postgres_database, feature_id)
    assert await store.reconcile_abandoned_runs(stale_after_seconds=_STALE) == [feature_id]
    decided = (await store.get_record(feature_id)).state
    assert decided.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert await store.feature_owes_another_step(feature_id, intent=CONTINUE) is False


@pytest.mark.postgres
async def test_the_retry_lineage_and_the_two_clocks_survive_the_postgres_round_trip(
    postgres_database: Database,
) -> None:
    """What the retry tasks record has to come back from the deployment's real schema.

    Three new durable facts ride existing JSON and float columns: a gate-rejected attempt's
    code completion, the attempt-inputs record saying what the retry ran against, and the
    two runtime clocks. A round trip through the migrated Postgres schema is what proves no
    column, serialiser or validator quietly drops them.
    """
    from agents.shared.contracts import create_artifact
    from artifacts.schemas import CodeCompletionArtifact

    store = _control_plane(postgres_database)
    feature_id = await _accept_feature(store, "retry-lineage")
    record = await store.get_record(feature_id)
    state = record.state.model_copy(deep=True)
    rejected = create_artifact(
        CodeCompletionArtifact,
        workflow_id=f"{feature_id}:backend",
        artifact_id="006_code_completion.backend.attempt-0.json",
        producer="engineer",
        metadata={
            "child_attempt": 0,
            "source_validation_rejected": True,
            "context_file_paths": ["src/index.js"],
        },
        payload={
            "completion_status": "failed",
            "summary": "The gate rejected this attempt.",
            "file_changes": [
                {
                    "path": "src/status.js",
                    "change_type": "added",
                    "description": "Updated by the configured coding executor.",
                }
            ],
            "validation_results": [],
            "test_coverage_percent": None,
            "remaining_work": [],
            "commit_sha": None,
        },
    )
    state.artifacts.append(rejected)
    state.child_workflows["backend"] = ChildWorkflowReference(
        child_workflow_id=f"{feature_id}:backend",
        repository_id="backend",
        workstream_id="backend",
        status=ChildWorkflowStatus.FAILED,
        branch_name="ai/retry-lineage/backend",
        workspace_path="/workspaces/retry-lineage/backend",
        retry_count=1,
        runtime_wall_seconds=5_400.0,
        runtime_charged_seconds=2_100.0,
    )
    await store._replace_state(feature_id, state, "tier_test")  # noqa: SLF001

    reloaded = (await store.get_record(feature_id)).state
    child = reloaded.child_workflows["backend"]
    assert child.runtime_wall_seconds == 5_400.0
    assert child.runtime_charged_seconds == 2_100.0
    persisted = next(
        artifact for artifact in reloaded.artifacts if isinstance(artifact, CodeCompletionArtifact)
    )
    assert persisted.completion_status == "failed"
    assert persisted.metadata["source_validation_rejected"] is True
    assert persisted.metadata["context_file_paths"] == ["src/index.js"]
    assert persisted.metadata["child_attempt"] == 0


@pytest.mark.postgres
async def test_the_planning_dark_records_survive_the_postgres_round_trip(
    postgres_database: Database,
) -> None:
    """Everything 41- makes durable has to come back from the deployment's real schema.

    Five new facts, four surfaces: the blind-planning flag and its reason on the child
    row's new columns; the blind map, the grounding-failure flag and the two planning
    clocks inside `state_json`; a journaled pre-coding model call read back through the
    per-feature listing the executions read uses; and the mid-run timeline event the new
    writer appends outside any state replacement. SQLite proves none of the column
    behaviour Postgres enforces, so the round trip here is the one that counts.
    """
    from services.cancellation import MockCancellationToken
    from services.execution_records import PLANNING_CALL_OPERATION_TYPES
    from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
    from state.external_operations import ExternalOperationStatus, ExternalOperationType
    from storage.external_operation_store import OperationResult

    store = _control_plane(postgres_database)
    feature_id = await _accept_feature(store, "planning-dark")
    state = (await store.get_record(feature_id)).state.model_copy(deep=True)
    state.planned_blind_repositories = {"backend": "the model provider call failed (ReadTimeout)"}
    state.clarification_grounding_failed = True
    state.planning_wall_seconds = 2_460.0
    state.planning_provider_fault_seconds = 2_220.0
    state.child_workflows["backend"] = ChildWorkflowReference(
        child_workflow_id=f"{feature_id}:backend",
        repository_id="backend",
        workstream_id="backend",
        status=ChildWorkflowStatus.PENDING,
        branch_name="ai/planning-dark/backend",
        workspace_path="/workspaces/planning-dark/backend",
        retry_count=0,
        planned_blind=True,
        planned_blind_reason="the model provider call failed (ReadTimeout)",
    )
    await store._replace_state(feature_id, state, "tier_test")  # noqa: SLF001
    await store.record_feature_event(
        feature_id,
        "repository_planned_blind",
        {"repository_id": "backend", "occurrence": "reconnaissance"},
    )

    journal = ExternalOperationJournal(postgres_database)
    executor = ExternalOperationExecutor(
        journal=journal,
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(workflow_id=feature_id, feature_id=feature_id),
    )

    async def call() -> tuple[str, OperationResult]:
        return "inspected", OperationResult()

    await executor.run(
        operation_type=ExternalOperationType.RUN_REPOSITORY_RECON,
        logical_step="repository_reconnaissance",
        safe_input={"stage": "repository_reconnaissance", "repository_id": "backend"},
        idempotency_input={"stage": "repository_reconnaissance", "call_nonce": "tier-test"},
        action=call,
        max_attempts=1,
        attempt_metadata={"stage": "repository_reconnaissance"},
    )

    reloaded = (await store.get_record(feature_id)).state
    assert reloaded.planned_blind_repositories == {
        "backend": "the model provider call failed (ReadTimeout)"
    }
    assert reloaded.clarification_grounding_failed is True
    assert reloaded.planning_wall_seconds == 2_460.0
    assert reloaded.planning_provider_fault_seconds == 2_220.0
    child = reloaded.child_workflows["backend"]
    assert child.planned_blind is True
    assert child.planned_blind_reason == "the model provider call failed (ReadTimeout)"

    rows = await journal.list_operations_for_feature(
        feature_id, operation_types=list(PLANNING_CALL_OPERATION_TYPES)
    )
    assert [row.operation_type for row in rows] == [ExternalOperationType.RUN_REPOSITORY_RECON]
    assert rows[0].status is ExternalOperationStatus.SUCCEEDED
    assert rows[0].started_at is not None and rows[0].heartbeat_at is not None
    assert rows[0].completed_at is not None

    events = await store.events_after(feature_id, after_id=None, limit=200)
    blind = [item for item in events if item.event == "repository_planned_blind"]
    assert len(blind) == 1 and blind[0].details["repository_id"] == "backend"
