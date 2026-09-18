"""Durable execution of the actions people ask a feature to perform.

These tests are about the window that used to be unprotected: an action is confirmed, its
execution begins, and the process dies. Before durable actions the only record was a string
column on a chat message, so afterwards nobody -- not a person, not the platform -- could tell
whether the retry that was authorized had happened.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from services.action_recovery import ActionRecoveryService
from services.cancellation import MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.feature_actions import (
    ActionLeaseLostError,
    FeatureActionService,
    action_context_version,
)
from state.enums import FeatureActionStatus, FeatureWorkflowStatus
from state.external_operations import ExternalOperationStatus, ExternalOperationType
from storage.action_store import (
    ActionConflictError,
    ActionInProgressError,
    DatabaseFeatureActionStore,
    InMemoryFeatureActionStore,
)
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal, OperationResult
from storage.models import FeatureWorkflowModel


def service(store: Any) -> FeatureActionService:
    """Build the action service over a store, with a recognisable build identity."""
    return FeatureActionService(store=store, build_revision="test-build")


async def submit(actions: FeatureActionService, **overrides: Any) -> Any:
    """Submit a retry intent, letting a test vary one part of it."""
    arguments: dict[str, Any] = {
        "feature_id": "feature-1",
        "action_type": "RETRY_WORKSTREAM",
        "actor_id": "someone",
        "repository_id": "backend",
        "payload": {"additional_attempts": 1, "reason": "read the blocking issues"},
        "context_version": "repository:backend:revision:abc:attempts:2:0",
    }
    arguments.update(overrides)
    return await actions.submit(**arguments)


@pytest.mark.asyncio
async def test_the_same_intent_submitted_twice_is_one_action() -> None:
    """A double-clicked Confirm must buy one retry, not two."""
    actions = service(InMemoryFeatureActionStore())

    first, created_first = await submit(actions)
    second, created_second = await submit(actions)

    assert created_first is True
    assert created_second is False
    assert first.action_id == second.action_id


@pytest.mark.asyncio
async def test_an_intent_decided_against_changed_state_is_a_new_action() -> None:
    """A grant made after the granted attempt was spent is a second decision, not a replay.

    Without this, somebody who granted an attempt, watched it fail and granted another would
    silently receive the first grant's recorded result and no second attempt at all.
    """
    actions = service(InMemoryFeatureActionStore())

    first, _ = await submit(actions)
    second, created = await submit(
        actions, context_version="repository:backend:revision:abc:attempts:3:1"
    )

    assert created is True
    assert first.action_id != second.action_id


@pytest.mark.asyncio
async def test_a_client_request_key_replays_after_workflow_state_changes() -> None:
    """A lost success response must not become a new action after the domain state advances."""
    actions = service(InMemoryFeatureActionStore())
    first, created = await submit(actions, request_idempotency_key="browser-request-1")
    assert created is True

    replay, replay_created = await submit(
        actions,
        request_idempotency_key="browser-request-1",
        context_version="repository:backend:revision:def:attempts:3:1",
    )

    assert replay_created is False
    assert replay.action_id == first.action_id


@pytest.mark.asyncio
async def test_a_client_request_key_cannot_be_reused_for_different_payload() -> None:
    """An opaque browser key is idempotency, not permission to alias two decisions."""
    actions = service(InMemoryFeatureActionStore())
    await submit(actions, request_idempotency_key="browser-request-1")

    with pytest.raises(ActionConflictError, match="different request payload"):
        await submit(
            actions,
            request_idempotency_key="browser-request-1",
            payload={"additional_attempts": 2, "reason": "a different grant"},
        )


@pytest.mark.asyncio
async def test_two_repositories_retried_with_identical_arguments_stay_separate() -> None:
    """Actions are scoped by repository id, never by role or by argument shape alone."""
    actions = service(InMemoryFeatureActionStore())

    backend, _ = await submit(actions, repository_id="backend")
    frontend, created = await submit(actions, repository_id="frontend")

    assert created is True
    assert backend.action_id != frontend.action_id


@pytest.mark.asyncio
async def test_a_successful_action_replays_instead_of_running_again() -> None:
    """The second confirmation of a completed action must not reach the control plane."""
    actions = service(InMemoryFeatureActionStore())
    action, _ = await submit(actions)
    runs: list[int] = []

    async def run() -> str:
        runs.append(1)
        return "granted one attempt"

    first = await actions.execute(action, run)
    second = await actions.execute(action, run)

    assert first.status is FeatureActionStatus.SUCCEEDED
    assert second.result_summary == "granted one attempt"
    assert len(runs) == 1


@pytest.mark.asyncio
async def test_a_failed_action_is_refused_rather_than_replayed() -> None:
    """A failure may have crossed an external checkpoint before it was reported.

    So the platform will not say it succeeded, and will not quietly try again. It refuses,
    which is the only answer it can support.
    """
    actions = service(InMemoryFeatureActionStore())
    action, _ = await submit(actions)

    async def failing() -> str:
        raise RuntimeError("the provider refused")

    with pytest.raises(RuntimeError):
        await actions.execute(action, failing)

    with pytest.raises(ActionConflictError, match="already failed"):
        await actions.execute(action, failing)


@pytest.mark.asyncio
async def test_an_action_someone_else_is_running_is_refused() -> None:
    """Two workers must not both carry out the same grant."""
    store = InMemoryFeatureActionStore()
    actions = service(store)
    action, _ = await submit(actions)
    assert await store.claim(action.action_id, owner="another-worker") is not None

    with pytest.raises(ActionInProgressError):
        await actions.execute(action, _unreachable)


@pytest.mark.asyncio
async def test_an_expired_lease_is_reclaimable_while_attempts_remain() -> None:
    """A crashed executor stops renewing, and its work must not stay owned forever."""
    store = InMemoryFeatureActionStore(lease_seconds=60)
    actions = service(store)
    action, _ = await submit(actions, max_attempts=2)
    claimed = await store.claim(action.action_id, owner="dead-worker")
    assert claimed is not None
    assert await store.claim(action.action_id, owner="live-worker") is None, (
        "a live lease must not be stealable"
    )

    # The worker died: no renewal, so the lease lapses.
    store._actions[action.action_id] = claimed.model_copy(  # noqa: SLF001 - a dead process
        update={"lease_expires_at": datetime.now(UTC) - timedelta(seconds=1)}
    )

    reclaimed = await store.claim(action.action_id, owner="live-worker")
    assert reclaimed is not None
    assert reclaimed.lease_owner == "live-worker"
    assert reclaimed.attempt == 2


@pytest.mark.asyncio
async def test_an_expired_lease_with_no_attempts_left_is_not_silently_retried() -> None:
    """The attempt budget is not reset by a crash. Recovery decides; it does not re-run."""
    store = InMemoryFeatureActionStore(lease_seconds=60)
    actions = service(store)
    action, _ = await submit(actions)
    claimed = await store.claim(action.action_id, owner="dead-worker")
    assert claimed is not None
    store._actions[action.action_id] = claimed.model_copy(  # noqa: SLF001 - a dead process
        update={"lease_expires_at": datetime.now(UTC) - timedelta(seconds=1)}
    )

    assert await store.claim(action.action_id, owner="live-worker") is None


class _FlakyLeaseStore(InMemoryFeatureActionStore):
    """Fail a fixed number of renewal calls the way a saturated pool does: by raising."""

    def __init__(self, *, lease_seconds: float, failures: int) -> None:
        super().__init__(lease_seconds=lease_seconds)
        self.remaining_failures = failures
        self.renewals = 0

    async def renew_lease(self, action_id: str, *, owner: str) -> bool:
        """Raise while failures remain, then behave normally."""
        self.renewals += 1
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            msg = "QueuePool limit of size 5 overflow 10 reached, connection timed out"
            raise TimeoutError(msg)
        return await super().renew_lease(action_id, owner=owner)


@pytest.mark.asyncio
async def test_a_transient_renewal_error_does_not_abandon_a_running_action() -> None:
    """The defect that ended AB-Feature-108 after seventy minutes and nine attempts.

    One renewal call failed -- the work it was guarding had the connection pool busy -- and
    the executor read a failed call as a lost lease, cancelled the run mid-attempt, and left
    the feature with no terminal status at all. A failed call is not a lost lease.
    """
    store = _FlakyLeaseStore(lease_seconds=1.0, failures=3)
    actions = service(store)
    action, _ = await submit(actions)

    async def work() -> str:
        await asyncio.sleep(0.6)
        return "finished under a lease that was never actually lost"

    result = await actions.execute(action, work)

    assert result.status is FeatureActionStatus.SUCCEEDED
    assert store.remaining_failures == 0, "the renewal must have been retried, not given up on"


@pytest.mark.asyncio
async def test_a_lease_taken_by_somebody_else_still_stops_the_work_at_once() -> None:
    """Tolerating failed calls must not tolerate an action another executor now owns.

    This is the one way the design could still produce a duplicate effect, so the action is
    cancelled rather than allowed to finish against a lease it no longer holds.
    """
    store = InMemoryFeatureActionStore(lease_seconds=0.3)
    actions = service(store)
    action, _ = await submit(actions)
    finished: list[str] = []

    async def slow() -> str:
        await asyncio.sleep(5)
        finished.append("completed anyway")
        return "done"

    async def steal() -> None:
        await asyncio.sleep(0.05)
        store._actions[action.action_id] = store._actions[action.action_id].model_copy(  # noqa: SLF001
            update={"lease_owner": "somebody-else"}
        )

    thief = asyncio.create_task(steal())
    with pytest.raises(ActionLeaseLostError):
        await actions.execute(action, slow)
    await thief

    assert finished == []


@pytest.mark.asyncio
async def test_renewal_gives_up_once_the_lease_has_objectively_expired() -> None:
    """Retrying is bounded by the lease, not unbounded: a real partition must still stop.

    Otherwise a worker that cannot reach the database at all would keep running while
    recovery, which can reach it, decides the action was abandoned and hands it on.
    """
    store = _FlakyLeaseStore(lease_seconds=0.4, failures=1_000)
    actions = service(store)
    action, _ = await submit(actions)
    finished: list[str] = []

    async def slow() -> str:
        await asyncio.sleep(5)
        finished.append("completed anyway")
        return "done"

    with pytest.raises(ActionLeaseLostError):
        await actions.execute(action, slow)

    assert finished == []
    assert store.renewals > 1, "the lease window must be used before giving up"


class _StubJournal:
    """Return a fixed set of operations, as the journal would after a crash."""

    def __init__(self, operations: list[Any]) -> None:
        self._operations = operations

    async def list_operations_for_action(self, action_id: str) -> list[Any]:
        """Return the exact recorded effects for the action a test does not vary."""
        del action_id
        return self._operations


class _Operation:
    """The two fields recovery reads off an operation."""

    def __init__(self, operation_type: ExternalOperationType, status: Any) -> None:
        self.operation_type = operation_type
        self.status = status
        self.operation_id = f"operation-{operation_type.value}"


async def _abandoned(store: InMemoryFeatureActionStore, actions: FeatureActionService) -> Any:
    """Produce an action in the state a crash leaves behind: claimed, lease expired."""
    action, _ = await submit(actions)
    claimed = await store.claim(action.action_id, owner="dead-worker")
    assert claimed is not None
    stranded = claimed.model_copy(
        update={
            "status": FeatureActionStatus.EXECUTING,
            "lease_expires_at": datetime.now(UTC) - timedelta(seconds=1),
        }
    )
    store._actions[action.action_id] = stranded  # noqa: SLF001 - simulating a dead process
    return stranded


@pytest.mark.asyncio
async def test_an_executing_action_without_a_domain_checkpoint_needs_reconciliation() -> None:
    """No external effects does not prove an internal workflow transaction did not commit."""
    store = InMemoryFeatureActionStore()
    actions = service(store)
    stranded = await _abandoned(store, actions)
    recovery = ActionRecoveryService(store, _StubJournal([]))

    summary = await recovery.recover_abandoned_actions()

    assert summary.unresolved == 1
    recovered = await store.get(stranded.action_id)
    assert recovered is not None
    assert recovered.status is FeatureActionStatus.REQUIRES_RECONCILIATION


@pytest.mark.asyncio
async def test_external_success_without_a_domain_checkpoint_is_not_reported_done() -> None:
    """Provider success alone does not prove that the workflow state transaction committed."""
    store = InMemoryFeatureActionStore()
    actions = service(store)
    stranded = await _abandoned(store, actions)
    recovery = ActionRecoveryService(
        store,
        _StubJournal(
            [
                _Operation(ExternalOperationType.PUSH_BRANCH, ExternalOperationStatus.SUCCEEDED),
                _Operation(
                    ExternalOperationType.CREATE_PULL_REQUEST, ExternalOperationStatus.SUCCEEDED
                ),
            ]
        ),
    )

    summary = await recovery.recover_abandoned_actions()

    assert summary.unresolved == 1
    recovered = await store.get(stranded.action_id)
    assert recovered is not None
    assert recovered.status is FeatureActionStatus.REQUIRES_RECONCILIATION


@pytest.mark.asyncio
async def test_a_committed_domain_checkpoint_recovers_as_succeeded() -> None:
    """A crash after the domain commit but before final status must not repeat the action."""
    store = InMemoryFeatureActionStore()
    actions = service(store)
    stranded = await _abandoned(store, actions)
    store._actions[stranded.action_id] = stranded.model_copy(  # noqa: SLF001 - crash window
        update={
            "domain_result_committed_at": datetime.now(UTC),
            "pending_result_summary": "retry committed",
        }
    )
    recovery = ActionRecoveryService(store, _StubJournal([]))

    summary = await recovery.recover_abandoned_actions()

    assert summary.completed == 1
    recovered = await store.get(stranded.action_id)
    assert recovered is not None
    assert recovered.status is FeatureActionStatus.SUCCEEDED
    assert recovered.result_summary == "retry committed"


@pytest.mark.asyncio
async def test_an_unconfirmed_external_effect_is_never_retried_automatically() -> None:
    """A push that may or may not have reached the provider is a question for a person."""
    store = InMemoryFeatureActionStore()
    actions = service(store)
    stranded = await _abandoned(store, actions)
    recovery = ActionRecoveryService(
        store,
        _StubJournal(
            [
                _Operation(
                    ExternalOperationType.PUSH_BRANCH,
                    ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
                )
            ]
        ),
    )

    await recovery.recover_abandoned_actions()

    recovered = await store.get(stranded.action_id)
    assert recovered is not None
    assert recovered.status is FeatureActionStatus.REQUIRES_RECONCILIATION
    # And it stays refused rather than quietly becoming available again.
    with pytest.raises(ActionConflictError, match="reconciled"):
        await actions.execute(recovered, _unreachable)


@pytest.mark.asyncio
async def test_an_administrator_can_close_an_uncertain_action_without_rerunning_it() -> None:
    """Manual reconciliation is an audit decision, never another execution attempt."""
    store = InMemoryFeatureActionStore()
    actions = service(store)
    stranded = await _abandoned(store, actions)
    await ActionRecoveryService(store, _StubJournal([])).recover_abandoned_actions()
    runs = 0

    reconciled = await actions.operator_reconcile(
        stranded.action_id,
        actor_id="admin-1",
        succeeded=False,
        reason="Verified that no branch, PR, or workflow state change exists.",
    )

    assert reconciled.status is FeatureActionStatus.FAILED
    assert reconciled.reconciled_by == "admin-1"
    assert reconciled.reconciled_at is not None
    assert reconciled.reconciliation_reason is not None
    assert reconciled.reconciliation_reason.startswith("Verified")
    assert runs == 0
    with pytest.raises(ActionConflictError, match="requiring reconciliation"):
        await actions.operator_reconcile(
            stranded.action_id,
            actor_id="admin-2",
            succeeded=True,
            reason="A conflicting second conclusion.",
        )


@pytest.mark.asyncio
async def test_a_half_finished_publication_requires_a_person() -> None:
    """The branch pushed and the pull request did not. What reached the provider is unknown."""
    store = InMemoryFeatureActionStore()
    actions = service(store)
    stranded = await _abandoned(store, actions)
    recovery = ActionRecoveryService(
        store,
        _StubJournal(
            [
                _Operation(ExternalOperationType.PUSH_BRANCH, ExternalOperationStatus.SUCCEEDED),
                _Operation(
                    ExternalOperationType.CREATE_PULL_REQUEST,
                    ExternalOperationStatus.FAILED_RETRYABLE,
                ),
            ]
        ),
    )

    await recovery.recover_abandoned_actions()

    recovered = await store.get(stranded.action_id)
    assert recovered is not None
    assert recovered.status is FeatureActionStatus.REQUIRES_RECONCILIATION


@pytest.mark.asyncio
async def test_workspace_only_work_is_recorded_as_failed_not_unresolved() -> None:
    """Its effects are inside a checkout the platform rebuilds, so nothing was left behind."""
    store = InMemoryFeatureActionStore()
    actions = service(store)
    stranded = await _abandoned(store, actions)
    recovery = ActionRecoveryService(
        store,
        _StubJournal(
            [
                _Operation(
                    ExternalOperationType.RUN_CODING_EXECUTOR, ExternalOperationStatus.SUCCEEDED
                ),
                _Operation(ExternalOperationType.RUN_TESTS, ExternalOperationStatus.CANCELLED),
            ]
        ),
    )

    summary = await recovery.recover_abandoned_actions()

    assert summary.failed == 1
    recovered = await store.get(stranded.action_id)
    assert recovered is not None
    assert recovered.status is FeatureActionStatus.FAILED


@pytest.mark.asyncio
async def test_cancellation_is_always_safe_to_ask_for_again() -> None:
    """Cancelling changes nothing outside the platform, and cancelling twice is cancelling."""
    store = InMemoryFeatureActionStore()
    actions = service(store)
    action, _ = await submit(
        actions,
        action_type="CANCEL_WORKFLOW",
        repository_id=None,
        payload={"reason": "no longer needed"},
        context_version=None,
    )
    claimed = await store.claim(action.action_id, owner="dead-worker")
    assert claimed is not None
    store._actions[action.action_id] = claimed.model_copy(  # noqa: SLF001
        update={
            "status": FeatureActionStatus.EXECUTING,
            "lease_expires_at": datetime.now(UTC) - timedelta(seconds=1),
        }
    )
    recovery = ActionRecoveryService(store, _StubJournal([]))

    await recovery.recover_abandoned_actions()

    recovered = await store.get(action.action_id)
    assert recovered is not None
    assert recovered.status is FeatureActionStatus.CONFIRMED
    assert recovered.attempt < recovered.max_attempts

    runs = 0

    async def cancel_again() -> str:
        nonlocal runs
        runs += 1
        return "cancelled"

    completed = await actions.execute(recovered, cancel_again)
    assert completed.status is FeatureActionStatus.SUCCEEDED
    assert runs == 1


@pytest.mark.asyncio
async def test_recovery_leaves_an_executor_that_came_back_alone() -> None:
    """A sweep must not decide the fate of work somebody is still doing."""
    store = InMemoryFeatureActionStore(lease_seconds=60)
    actions = service(store)
    stranded = await _abandoned(store, actions)
    recovery = ActionRecoveryService(store, _StubJournal([]))
    listed = await store.list_expired()
    assert [item.action_id for item in listed] == [stranded.action_id]

    # The worker renews between the sweep listing the action and deciding it.
    store._actions[stranded.action_id] = stranded.model_copy(  # noqa: SLF001
        update={"lease_expires_at": datetime.now(UTC) + timedelta(seconds=60)}
    )
    reconciled = await store.reconcile(stranded.action_id, status=FeatureActionStatus.CONFIRMED)

    assert reconciled is None
    del recovery


@pytest.mark.asyncio
async def test_the_durable_store_enforces_one_action_per_intent(tmp_path: Path) -> None:
    """The uniqueness that makes a repeat safe is the database's, not a prior read's."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'actions.db'}")
    await database.create_schema()
    try:
        async with database.session() as session:
            session.add(
                FeatureWorkflowModel(
                    owner_id="platform-admin",
                    feature_id="feature-1",
                    status=FeatureWorkflowStatus.PENDING,
                    title="A feature",
                    execution_mode="mock",
                    state_json={},
                )
            )
            await session.commit()
        store = DatabaseFeatureActionStore(database)
        actions = service(store)

        first, created_first = await submit(actions)
        second, created_second = await submit(actions)

        assert created_first is True
        assert created_second is False
        assert first.action_id == second.action_id
        assert len(await store.list_for_feature("feature-1")) == 1
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_claim_survives_being_read_back_from_the_database(tmp_path: Path) -> None:
    """Lease comparisons must work against a database that returns naive timestamps."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'lease.db'}")
    await database.create_schema()
    try:
        async with database.session() as session:
            session.add(
                FeatureWorkflowModel(
                    owner_id="platform-admin",
                    feature_id="feature-1",
                    status=FeatureWorkflowStatus.PENDING,
                    title="A feature",
                    execution_mode="mock",
                    state_json={},
                )
            )
            await session.commit()
        store = DatabaseFeatureActionStore(database)
        actions = service(store)
        action, _ = await submit(actions)

        claimed = await store.claim(action.action_id, owner="worker-1")
        assert claimed is not None
        assert claimed.lease_is_live(now=datetime.now(UTC))
        # A second worker cannot take it while the first still holds it.
        assert await store.claim(action.action_id, owner="worker-2") is None
        assert await store.renew_lease(action.action_id, owner="worker-1") is True
        assert await store.renew_lease(action.action_id, owner="worker-2") is False

        recorded = await store.record_success(
            action.action_id, owner="worker-1", result_summary="done"
        )
        assert recorded.status is FeatureActionStatus.SUCCEEDED
        assert recorded.lease_expires_at is None
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_database_reconciliation_is_attributed_and_compare_and_swapped(
    tmp_path: Path,
) -> None:
    """Only the first human decision may close a durable uncertain action."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'action-reconciliation.db'}")
    await database.create_schema()
    try:
        async with database.session() as session:
            session.add(
                FeatureWorkflowModel(
                    owner_id="platform-admin",
                    feature_id="feature-1",
                    status=FeatureWorkflowStatus.PENDING,
                    title="A feature",
                    execution_mode="mock",
                    state_json={},
                )
            )
            await session.commit()
        store = DatabaseFeatureActionStore(database)
        actions = service(store)
        action, _ = await submit(actions)
        claimed = await store.claim(action.action_id, owner="dead")
        assert claimed is not None
        async with database.session() as session:
            from sqlalchemy import update

            from storage.models import FeatureActionModel

            await session.execute(
                update(FeatureActionModel)
                .where(FeatureActionModel.action_id == action.action_id)
                .values(
                    status=FeatureActionStatus.REQUIRES_RECONCILIATION,
                    lease_owner=None,
                    lease_expires_at=None,
                )
            )
            await session.commit()

        decided = await actions.operator_reconcile(
            action.action_id,
            actor_id="admin-1",
            succeeded=True,
            reason="Workflow state and provider were checked and agree.",
        )

        assert decided.status is FeatureActionStatus.SUCCEEDED
        assert decided.reconciled_by == "admin-1"
        assert decided.reconciliation_reason is not None
        assert decided.reconciliation_reason.startswith("Workflow state")
        with pytest.raises(ActionConflictError):
            await actions.operator_reconcile(
                action.action_id,
                actor_id="admin-2",
                succeeded=False,
                reason="Late disagreement.",
            )
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_external_effect_evidence_is_linked_to_the_exact_action(tmp_path: Path) -> None:
    """A concurrent feature operation must never become evidence for the wrong action."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'action-evidence.db'}")
    await database.create_schema()
    try:
        async with database.session() as session:
            session.add(
                FeatureWorkflowModel(
                    owner_id="platform-admin",
                    feature_id="feature-1",
                    status=FeatureWorkflowStatus.PENDING,
                    title="A feature",
                    execution_mode="mock",
                    state_json={},
                )
            )
            await session.commit()
        journal = ExternalOperationJournal(database)
        actions = FeatureActionService(
            store=DatabaseFeatureActionStore(database),
            journal=journal,
            build_revision="test",
        )
        executor = ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(workflow_id="feature-1", feature_id="feature-1"),
        )

        async def effect() -> tuple[dict[str, bool], OperationResult]:
            return {"ok": True}, OperationResult(payload={"ok": True})

        # Same feature, but outside the durable action context.
        unrelated = await executor.run(
            operation_type=ExternalOperationType.RUN_TESTS,
            logical_step="unrelated",
            safe_input={"command_fingerprint": "unrelated"},
            action=effect,
        )
        action, _ = await submit(actions)

        async def run() -> str:
            await executor.run(
                operation_type=ExternalOperationType.RUN_TESTS,
                logical_step="owned",
                safe_input={"command_fingerprint": "owned"},
                action=effect,
            )
            return "workflow state committed"

        completed = await actions.execute(action, run)
        linked = await journal.list_operations_for_action(action.action_id)

        assert completed.status is FeatureActionStatus.SUCCEEDED
        assert [item.operation_id for item in linked] == completed.external_operation_ids
        assert unrelated.operation.operation_id not in completed.external_operation_ids
    finally:
        await database.dispose()


def test_cancellation_has_no_context_and_everything_else_does() -> None:
    """Only cancellation is idempotent whatever the feature has since done."""

    class _Child:
        current_revision = "abc123"
        retry_count = 2
        granted_extra_attempts = 0

    class _State:
        clarification_rounds = 1
        child_workflows = {"backend": _Child()}

    state = _State()

    assert action_context_version(state, "CANCEL_WORKFLOW", {}) is None
    assert action_context_version(state, "RESUME_WORKFLOW", {}) == "clarification:1"
    retry = action_context_version(state, "RETRY_WORKSTREAM", {"repository_id": "backend"})
    assert retry is not None
    assert "abc123" in retry
    assert "backend" in retry
    repair = action_context_version(state, "APPROVE_REPOSITORY_REPAIR", {"repair_id": "repair-9"})
    assert repair == "repair:repair-9"


@pytest.mark.asyncio
async def test_an_operation_the_platform_could_not_carry_out_is_not_recorded_as_committed(
    tmp_path: Path,
) -> None:
    """The action record must not say `committed` for work that never committed.

    This is the audit answer to "did this happen?", and it was wrong in the one case where
    the question matters: an operation whose runner failed unexpectedly marked the feature
    for attention and then returned the feature, which the durable action layer -- rightly
    unable to tell that apart from a normal return -- wrote down as a success. A caller
    relying on the platform's own action identity then replayed that success instead of
    trying again, so the repair could never be approved and nobody was told why.

    A resume no longer executes the feature inside the request, so what its action now
    commits to is arranging the resumption -- which genuinely happens. The invariant is
    therefore checked here on `approve_repair`, which still runs inline, and the queued
    counterpart is covered by
    `test_a_queued_mutation_that_fails_reaches_the_feature_not_the_caller`.
    """
    from httpx import ASGITransport, AsyncClient

    from main import create_app
    from storage.feature_store import SqlAlchemyFeatureControlPlane
    from tests.support import settle
    from tests.test_feature_state import BrokenAfterStartRunner, feature_payload

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'not-committed.db'}")
    await database.create_schema()
    try:
        control_plane = SqlAlchemyFeatureControlPlane(
            database, mock_runner=BrokenAfterStartRunner()
        )
        app = create_app(platform_api_key="action-truth-key", feature_control_plane=control_plane)
        headers = {"Authorization": "Bearer action-truth-key"}
        payload = dict(feature_payload())
        payload["feature_id"] = "not-committed"
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            started = await client.post("/features/start", headers=headers, json=payload)
            assert started.status_code == 201
            await settle(app)

            approved = await client.post(
                "/features/not-committed/repairs/repair-1/approve",
                headers=headers,
                json={"acknowledge_repository_change": True},
            )
            # Not 200. The caller asked for something the platform did not do.
            assert approved.status_code == 409
            assert "sk-secret-in-message" not in approved.text

            actions = await client.get("/features/not-committed/actions", headers=headers)
            recorded = actions.json()["actions"]
            assert len(recorded) == 1
            action = recorded[0]
            assert action["action_type"] == "APPROVE_REPOSITORY_REPAIR"
            assert action["status"] == "failed"
            assert action["result_summary"] is None
    finally:
        await database.drop_schema()
        await database.dispose()


async def _unreachable() -> str:
    """Fail the test if an action the platform should refuse is executed anyway."""
    raise AssertionError("this action must not have been executed")
