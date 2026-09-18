"""A refusal that attempted nothing is not a failure the ledger has to remember.

70-'s defect, found trying to grant AB-Feature-203's second repository an attempt. A durable
mutation is serialised by the workflow lock; while the first repository's grant ran, the
second one's request waited on that lock, timed out, and raised "workflow operation is
already in progress; retry shortly". The action ledger recorded that as `FAILED`, and from
then on the identity was refused permanently with "this action already failed and is not
repeated automatically" -- so the refusal's own advice to try again shortly became
impossible to follow.

The spec that described this defect had its mechanism wrong, and the correction is what
these tests pin. It said the lock conflict was raised "before the action is claimed, before
a lease is taken". It is not: `_execute_durable_mutation` claims the action and takes a
lease, and only then calls the domain method, which acquires the lock *inside* the lease
(`storage/feature_store.py:1061`). The four live rows on 203 all carry `attempt = 1` and a
`started_at`. So "was a lease ever taken" cannot tell the two cases apart -- it is true of
both -- and a fix built on it would have been a no-op that passed its own tests.

What does tell them apart is on the record too: every live side effect journals its intent
and links it to the running action *before* it is allowed to happen. An action with no
linked operation crossed no external checkpoint. That is the exact worry `execute`'s
docstring states, answered with evidence rather than with a tuple of exception classes -- so
a pre-flight refusal added years from now inherits the behaviour without being registered
anywhere.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from api.control_plane import WorkflowBusyError, WorkflowConflictError
from main import create_app
from services.cancellation import MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.feature_actions import FeatureActionService
from state.enums import FeatureActionStatus, FeatureWorkflowStatus
from state.external_operations import ExternalOperation, ExternalOperationType
from storage.action_store import (
    ActionConflictError,
    DatabaseFeatureActionStore,
    InMemoryFeatureActionStore,
)
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal, OperationResult
from storage.models import FeatureWorkflowModel
from tests.support import settle
from tests.test_feature_api import feature_payload

pytestmark = pytest.mark.asyncio

FEATURE = "feature-203"
# The identity of one grant. Held constant across every resubmission in this file, because a
# resend that changes the payload is a different intent and would prove nothing -- which is
# how the live 203 rows came to be four separate actions rather than one refused four times.
RETRY_PAYLOAD = {
    "additional_attempts": 1,
    "reason": "npm ci exceeded its install budget; disk reclaimed, retrying",
    "repository_id": "admanager_console-2.0",
}
CONTEXT = "repository:admanager_console-2.0:revision:abc123:attempts:2:0"


async def _database(tmp_path: Path, name: str) -> Database:
    """Open one test's own database with a feature row for the action to point at."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / f'{name}.db'}")
    await database.create_schema()
    async with database.session() as session:
        session.add(
            FeatureWorkflowModel(
                owner_id="platform-admin",
                feature_id=FEATURE,
                status=FeatureWorkflowStatus.PENDING,
                title="AB-Feature-203",
                execution_mode="mock",
                state_json={},
            )
        )
        await session.commit()
    return database


def _service(database: Database) -> tuple[FeatureActionService, ExternalOperationJournal]:
    """Build the action service over a real journal, as the application wires it."""
    journal = ExternalOperationJournal(database)
    return (
        FeatureActionService(
            store=DatabaseFeatureActionStore(database), journal=journal, build_revision="test"
        ),
        journal,
    )


async def _submit(actions: FeatureActionService) -> Any:
    """Submit the one grant identity this file is about."""
    action, _created = await actions.submit(
        feature_id=FEATURE,
        action_type="RETRY_WORKSTREAM",
        actor_id="akhilesh@appbroda.com",
        repository_id="admanager_console-2.0",
        payload=dict(RETRY_PAYLOAD),
        context_version=CONTEXT,
    )
    return action


class _EmptyJournal:
    """A journal that answers honestly that this action linked no external operation."""

    async def list_operations_for_action(self, action_id: str) -> list[ExternalOperation]:
        """Report no linked effects, which is the truth for every action in this file."""
        del action_id
        return []


def _lock_held() -> WorkflowConflictError:
    """The refusal the workflow lock raises, in its own words."""
    return WorkflowConflictError("workflow operation is already in progress; retry shortly")


async def test_t1_a_grant_refused_by_the_lock_runs_once_the_lock_clears(tmp_path: Path) -> None:
    """T1, the 203 replay: the second submission runs, and its effect lands."""
    database = await _database(tmp_path, "t1-lock-then-run")
    try:
        actions, _journal = _service(database)
        granted: list[int] = []

        async def refused() -> str:
            raise _lock_held()

        async def grants() -> str:
            # Stands in for the domain call: the grant that increments the repository's
            # `granted_extra_attempts` and queues the attempt.
            granted.append(1)
            return "RETRY_WORKSTREAM committed"

        first = await _submit(actions)
        with pytest.raises(WorkflowConflictError):
            await actions.execute(first, refused)

        # The same grant, resubmitted verbatim after the lock cleared.
        second = await _submit(actions)
        executed = await actions.execute(second, grants)

        # The effect, not the absence of a 409.
        assert granted == [1], "the grant never ran"
        assert executed.status is FeatureActionStatus.SUCCEEDED
        assert second.action_id == first.action_id, "the same intent must keep one identity"
    finally:
        await database.dispose()


async def test_t2_a_failure_after_an_external_effect_is_still_refused(tmp_path: Path) -> None:
    """T2: once a checkpoint is crossed, its extent is unknown and stays unrepeatable."""
    database = await _database(tmp_path, "t2-effect-then-fail")
    try:
        actions, journal = _service(database)
        action = await _submit(actions)

        executor = ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(workflow_id=FEATURE, feature_id=FEATURE),
        )

        async def effect() -> tuple[dict[str, bool], OperationResult]:
            return {"pushed": True}, OperationResult(payload={"pushed": True})

        async def pushes_then_dies() -> str:
            # A real journaled side effect, linked to this action because it runs inside it,
            # and then a failure. Nobody can say how much of what followed happened.
            await executor.run(
                operation_type=ExternalOperationType.PUSH_BRANCH,
                logical_step="push",
                safe_input={"command_fingerprint": "push"},
                action=effect,
            )
            msg = "the worker died after pushing"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError):
            await actions.execute(action, pushes_then_dies)

        stored = await actions.store.get(action.action_id)
        assert stored is not None
        assert stored.status is FeatureActionStatus.FAILED

        again = await _submit(actions)
        with pytest.raises(ActionConflictError, match="already failed and is not repeated"):
            await actions.execute(again, pushes_then_dies)
    finally:
        await database.dispose()


async def test_t3_reconciliation_is_untouched() -> None:
    """T3: an unconfirmed outcome keeps its own stronger refusal.

    Reached the way a crash reaches it: claimed, the lease lapsed, and the recovery sweep
    decided it could not say what happened. Deliberately distinct from this spec's case. An
    executor that died left nobody who can report where it got to, and no external operation
    does not prove an internal workflow transaction did not commit -- which is exactly what
    `test_an_executing_action_without_a_domain_checkpoint_needs_reconciliation` pins. Here
    the executor is alive and caught the exception, so it can say.
    """
    store = InMemoryFeatureActionStore()
    actions = FeatureActionService(store=store, journal=_EmptyJournal(), build_revision="test")
    action = await _submit(actions)
    claimed = await store.claim(action.action_id, owner="dead-worker")
    assert claimed is not None
    store._actions[action.action_id] = claimed.model_copy(  # noqa: SLF001 - a dead process
        update={
            "status": FeatureActionStatus.EXECUTING,
            "lease_expires_at": datetime.now(UTC) - timedelta(seconds=1),
        }
    )
    reconciling = await store.reconcile(
        action.action_id,
        status=FeatureActionStatus.REQUIRES_RECONCILIATION,
        error_code="executor_stopped",
        error_message="the executor stopped mid-flight",
    )
    assert reconciling is not None

    async def never_runs() -> str:
        msg = "reconciliation must be refused before anything runs"
        raise AssertionError(msg)

    with pytest.raises(ActionConflictError, match="must be reconciled"):
        await actions.execute(action, never_runs)


async def test_t4_a_succeeded_action_is_still_replayed(tmp_path: Path) -> None:
    """T4: one effect, both callers answered. The reason this ledger exists."""
    database = await _database(tmp_path, "t4-replay")
    try:
        actions, _journal = _service(database)
        runs: list[int] = []

        async def grants() -> str:
            runs.append(1)
            return "RETRY_WORKSTREAM committed"

        first = await actions.execute(await _submit(actions), grants)
        second = await actions.execute(await _submit(actions), grants)

        assert runs == [1], "a replay executed the domain operation a second time"
        assert first.status is FeatureActionStatus.SUCCEEDED
        assert second.status is FeatureActionStatus.SUCCEEDED
        assert second.action_id == first.action_id
        assert second.result_summary == first.result_summary
    finally:
        await database.dispose()


async def test_t6_after_a_refusal_the_ledger_holds_no_terminal_record(tmp_path: Path) -> None:
    """T6: assert the store directly. This is the property the whole change rests on.

    Asserting only that a later submission succeeds would pass on a fluke of timing, so this
    reads the row itself: not terminal, no failure sentence left on it, and claimable again.
    """
    database = await _database(tmp_path, "t6-no-memory")
    try:
        actions, _journal = _service(database)

        async def refused() -> str:
            raise _lock_held()

        action = await _submit(actions)
        with pytest.raises(WorkflowConflictError):
            await actions.execute(action, refused)

        stored = await actions.store.get(action.action_id)
        assert stored is not None
        assert stored.status is not FeatureActionStatus.FAILED
        assert stored.status is not FeatureActionStatus.CANCELLED
        assert stored.status is not FeatureActionStatus.REQUIRES_RECONCILIATION
        assert not stored.is_terminal, (
            "a refusal that attempted nothing is remembered as an outcome"
        )
        assert stored.status is FeatureActionStatus.CONFIRMED
        assert stored.error_message is None, "a failure sentence was left on a submittable action"
        assert stored.lease_expires_at is None
        # The attempt history is kept and the budget widened, exactly as recovery does: an
        # attempt really was made, and erasing it would make the ledger say something untrue.
        assert stored.attempt == 1
        assert stored.max_attempts == 2
        # No external operation was ever linked, which is the evidence the decision read.
        assert stored.external_operation_ids == []
    finally:
        await database.dispose()


async def test_a_refusal_repeated_many_times_never_poisons_the_identity(tmp_path: Path) -> None:
    """The live 203 loop: a well-behaved client resending while the lock stays held.

    Not in the spec's list. It is the shape that actually happened -- a caller retrying a
    transient conflict -- and one refusal being survivable is not the same claim as four
    being survivable, because each one widens the budget it is granted from.
    """
    database = await _database(tmp_path, "repeated-refusal")
    try:
        actions, _journal = _service(database)

        async def refused() -> str:
            raise _lock_held()

        async def grants() -> str:
            return "RETRY_WORKSTREAM committed"

        for _ in range(4):
            with pytest.raises(WorkflowConflictError):
                await actions.execute(await _submit(actions), refused)

        executed = await actions.execute(await _submit(actions), grants)

        assert executed.status is FeatureActionStatus.SUCCEEDED
    finally:
        await database.dispose()


async def test_without_a_journal_the_careful_answer_stands(tmp_path: Path) -> None:
    """Proof is required, not the absence of contradiction.

    A service with no journal cannot prove an action attempted nothing, so it must not claim
    it did. This is the guard that keeps the change from becoming "retry anything that
    failed before it got far enough to be linked to anything".
    """
    database = await _database(tmp_path, "no-journal")
    try:
        actions = FeatureActionService(
            store=DatabaseFeatureActionStore(database), journal=None, build_revision="test"
        )

        async def refused() -> str:
            raise _lock_held()

        action = await _submit(actions)
        with pytest.raises(WorkflowConflictError):
            await actions.execute(action, refused)

        stored = await actions.store.get(action.action_id)
        assert stored is not None
        assert stored.status is FeatureActionStatus.FAILED
    finally:
        await database.dispose()


async def test_t5_the_two_409s_are_told_apart_by_something_a_client_can_branch_on() -> None:
    """T5: assert what a client reads, not the prose it would otherwise have to parse.

    Both answers are 409 and both carry a sentence. Only one of them means "the same request
    works in a moment", and a client that cannot tell either gives up on work it should have
    waited a second for, or hammers work that has settled. 203's loop did the second.
    """
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "conflict-kinds"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        await settle(app)

        control_plane = app.state.feature_control_plane

        async def busy(*_args: Any, **_kwargs: Any) -> Any:
            raise WorkflowBusyError("workflow operation is already in progress; retry shortly")

        original = control_plane.retry_workstream
        control_plane.retry_workstream = busy
        try:
            transient = await client.post(
                "/features/feature-login/workstreams/backend/retry",
                headers=headers,
                json={
                    "additional_attempts": 1,
                    "requested_by": "akhilesh@appbroda.com",
                    "reason": "read the blocking issues",
                },
            )
        finally:
            control_plane.retry_workstream = original

        # The settled kind: the orchestrator refusing an operation that has reached an
        # outcome. Waiting changes nothing about it.
        settled = await client.post(
            "/features/feature-login/resume", headers=headers, json={"answers": []}
        )

    assert transient.status_code == 409
    assert settled.status_code == 409
    # Both are 409 with prose, so the prose is exactly what must not be the discriminator.
    assert transient.headers["X-Conflict-Reason"] == "workflow_busy"
    assert settled.headers["X-Conflict-Reason"] == "action_settled"
    assert transient.headers["Retry-After"] == "1"
    assert "Retry-After" not in settled.headers
