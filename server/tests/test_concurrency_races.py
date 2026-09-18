"""Ten races the plan created or inherited, produced on purpose and asserted durably.

Audit risk P1-8, third tier (C3). Every mechanism the preceding tasks added to stop two
workers colliding was verified by one worker calling it twice. That is coverage of the
mechanism's arithmetic, not of what happens when two things genuinely run at once -- and on
SQLite it could not be anything else, because `FOR UPDATE SKIP LOCKED` is skipped there by an
explicit dialect check and `SELECT ... FOR UPDATE` is accepted and ignored.

So every test here is marked `postgres`, runs against its own migrated database, and asserts
a **durable outcome** rather than a lock acquisition. "The lock was taken" is not a property;
"only one worker ran this step" is.

The rows are the audit's, in its order:

    1  two dispatchers draining one queue
    2  two workers claiming the same feature's step
    3  a resume arriving while a worker holds the feature
    4  a cancel arriving mid-step
    5  a cancel racing a granted retry
    6  a granted retry racing the abandoned-run sweep
    7  two state writers
    8  a duplicated submission with one idempotency key
    9  a step re-queue racing another feature's claim
   10  a lease lost mid-step, and the fence that stops the loser writing

Row 10 is the one with production history. AB-Feature-108's lease lapsed at 15:22, recovery
reconciled its action `interrupted_before_completion` at 15:36, and at 15:53 something still
belonging to that run wrote a ninth attempt into the feature. The fence that now refuses that
write has never been exercised under real contention, and task `29-` multiplied how often it
is entered by the number of steps a feature takes.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select, update

from api.control_plane import WorkflowConflictError
from api.feature_schemas import StartFeatureRequest
from services.action_context import clear_revoked_feature_actions
from services.feature_actions import ActionLeaseLostError, FeatureActionService
from services.feature_queue import (
    QUEUED,
    SUCCEEDED,
    DatabaseFeatureExecutionQueue,
    FeatureQueueDispatcher,
)
from state.enums import (
    CancellationLifecycleStatus,
    ChildWorkflowStatus,
    FeatureActionStatus,
    FeatureWorkflowStatus,
)
from state.feature_models import ChildWorkflowReference
from storage.action_store import DatabaseFeatureActionStore
from storage.db import Database
from storage.feature_store import (
    ClaimDisposition,
    FeatureRunRevokedError,
    SqlAlchemyFeatureControlPlane,
)
from storage.models import (
    FeatureActionModel,
    FeatureExecutionQueueModel,
    FeatureWorkflowModel,
)
from storage.workflow_lock import RedisWorkflowLock
from tests.concurrency_support import (
    CREDENTIALS,
    BlockingStepOrchestrator,
    SharedKeyStore,
    StepRecordingOrchestrator,
    credentials_for,
)
from tests.test_feature_api import feature_payload
from tests.test_feature_queue_and_references import ClarifyingRunner
from workflows.feature_workflow import FeatureWorkflowOrchestrator

pytestmark = [pytest.mark.asyncio, pytest.mark.postgres]

# Short enough that a second worker's refusal is immediate rather than a ten-second wait, and
# long enough that acquiring an uncontended lock never flakes.
_LOCK_TIMEOUT = 0.2


def _control_plane(
    database: Database,
    *,
    runner: FeatureWorkflowOrchestrator | None = None,
    keys: SharedKeyStore | None = None,
    lock_timeout: float = _LOCK_TIMEOUT,
    **options: Any,
) -> SqlAlchemyFeatureControlPlane:
    """Build the control plane the deployment builds, with a run lock that really locks."""
    return SqlAlchemyFeatureControlPlane(
        database,
        mock_runner=runner or FeatureWorkflowOrchestrator(),
        lock=(
            RedisWorkflowLock(keys, timeout_seconds=lock_timeout, lease_seconds=60)
            if keys is not None
            else None
        ),
        **options,
    )


async def _accept(store: SqlAlchemyFeatureControlPlane, key: str) -> str:
    """Accept one feature through the real path, so its queue row and feature row agree."""
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


def _dispatcher(store: SqlAlchemyFeatureControlPlane) -> FeatureQueueDispatcher:
    """Build one worker over this control plane's own queue."""
    return FeatureQueueDispatcher(
        queue=store.queue, executor=store, credentials_for=credentials_for
    )


# --------------------------------------------------------------------------------------
# 1. Two dispatchers draining one queue
# --------------------------------------------------------------------------------------


async def test_two_dispatchers_draining_one_queue_never_run_the_same_step_twice(
    postgres_database: Database,
) -> None:
    """The loser of a claim takes a different entry, or none -- never the same one.

    Asserted on the work rather than on the claim: running a step twice does not merely waste
    a call, it writes an artifact identifier the first run already used, which the immutable
    history refuses and which ends the feature on a platform error.
    """
    runner = StepRecordingOrchestrator()
    store = _control_plane(postgres_database, runner=runner)
    first = await _accept(store, "dispatch-a")
    second = await _accept(store, "dispatch-b")

    await asyncio.gather(*(_dispatcher(store).drain() for _ in range(2)))

    assert runner.duplicated() == [], f"a step ran twice under two dispatchers: {runner.steps}"
    for feature_id in (first, second):
        record = await store.get_record(feature_id)
        assert record.state.status is FeatureWorkflowStatus.COMPLETED
        entry = await _queue_row(postgres_database, feature_id)
        assert entry.status == SUCCEEDED
        ran = [item for item in runner.steps if item.startswith(feature_id)]
        assert entry.steps == len(ran) - 1


# --------------------------------------------------------------------------------------
# 2. Two workers claiming the same feature's step
# --------------------------------------------------------------------------------------


async def test_a_second_claim_on_a_running_feature_is_deferred_without_spending_an_attempt(
    postgres_database: Database,
) -> None:
    """The run lock is what turns a colliding claim into a deferral rather than a second run.

    Three of these in thirty seconds retired AB-Feature-104 while its first attempt was still
    working, because a claim that could not get in recorded a failed attempt. The attempt has
    to come back, which is why `defer` exists and why the assertion is on the queue row.
    """
    keys = SharedKeyStore()
    runner = BlockingStepOrchestrator()
    store = _control_plane(postgres_database, runner=runner, keys=keys)
    feature_id = await _accept(store, "busy")
    queue = DatabaseFeatureExecutionQueue(postgres_database)

    holder = asyncio.create_task(_dispatcher(store).run_once())
    await asyncio.wait_for(runner.entered.wait(), timeout=10)
    # The first worker is inside a step. Its lease is expired so a second worker can claim the
    # entry, which is exactly the shape a lease that lapsed under a long step produces.
    async with postgres_database.session() as session:
        await session.execute(
            update(FeatureExecutionQueueModel)
            .where(FeatureExecutionQueueModel.feature_id == feature_id)
            .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        await session.commit()
    assert await queue.claim(owner="second-worker", lease_seconds=60) is not None
    attempt_before = (await _queue_row(postgres_database, feature_id)).attempt

    with pytest.raises(Exception) as raised:
        await store.execute_queued(feature_id, credentials=CREDENTIALS, intent="start")

    assert type(raised.value).__name__ == "FeatureExecutionBusy"
    await queue.defer(feature_id, error="another worker holds this feature")
    deferred = await _queue_row(postgres_database, feature_id)
    assert deferred.attempt == attempt_before - 1, "a collision must not spend an attempt"
    assert deferred.status == QUEUED

    runner.release.set()
    await holder
    assert runner.duplicated() == [], "the deferred claim ran the step the holder was running"


# --------------------------------------------------------------------------------------
# 3. A resume arriving while a worker holds the feature
# --------------------------------------------------------------------------------------


async def test_a_resume_that_arrives_during_a_step_is_queued_and_applied_exactly_once(
    postgres_database: Database,
) -> None:
    """Answers travel with one claim, and four workers racing for it do not multiply them.

    Queued rather than run inline is the whole of AB-Feature-108's lesson: the resume that
    ran seventy minutes of work inside a request died with the socket. What this adds is the
    contention -- the entry a resume writes is one row, and every worker wants it.
    """
    runner = ClarifyingRunner()
    store = _control_plane(postgres_database, runner=runner)
    feature_id = await _accept(store, "resuming")
    await _dispatcher(store).drain()
    parked = (await store.get_record(feature_id)).state
    assert parked.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN

    await store.resume(feature_id, answers=[], credentials=CREDENTIALS)

    # Durable before anything runs: that is what makes it survive the request that asked.
    assert runner.resumed_with == []
    assert await store.queue.active_intent(feature_id) == "resume"

    await asyncio.gather(*(_dispatcher(store).drain() for _ in range(4)))

    assert runner.resumed_with == [[]], "the answers were applied more than once"
    assert (await store.get_record(feature_id)).state.status is FeatureWorkflowStatus.COMPLETED
    assert await store.queue.active_intent(feature_id) is None


# --------------------------------------------------------------------------------------
# 4. A cancel arriving mid-step
# --------------------------------------------------------------------------------------


async def test_a_cancel_during_a_step_is_never_erased_by_the_result_of_that_step(
    postgres_database: Database,
) -> None:
    """The step's writer holds a snapshot from before the cancel, and must not win with it.

    Termination is requested at the same time, and the signal is recorded here rather than
    asserted through a real child: killing a process at a boundary is task `27-`'s tier, and
    what this one owns is that the durable record cannot be walked backwards.
    """
    signalled: list[str] = []

    async def signal(feature_id: str) -> None:
        signalled.append(feature_id)

    runner = BlockingStepOrchestrator()
    store = _control_plane(postgres_database, runner=runner, cancellation_signaler=signal)
    feature_id = await _accept(store, "cancelling")

    worker = asyncio.create_task(_dispatcher(store).drain())
    await asyncio.wait_for(runner.entered.wait(), timeout=10)
    await store.cancel(
        feature_id, reason="an operator stopped this feature", requested_by="platform-admin"
    )
    cancelled_at = (await store.get_record(feature_id)).state.cancellation_requested_at
    runner.release.set()
    await worker

    record = await store.get_record(feature_id)
    assert signalled == [feature_id], "the running work was never asked to stop"
    assert record.state.cancellation_requested is True
    assert record.state.cancellation_requested_at == cancelled_at
    assert record.state.status in {
        FeatureWorkflowStatus.CANCELLING,
        FeatureWorkflowStatus.CANCELLED,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    }
    assert record.state.cancellation_status is not CancellationLifecycleStatus.NOT_REQUESTED


# --------------------------------------------------------------------------------------
# 5. A cancel racing a granted retry
# --------------------------------------------------------------------------------------


async def _stopped_workstream(store: SqlAlchemyFeatureControlPlane, feature_id: str) -> None:
    """Leave one repository stopped, which is the only state a grant may act on."""
    state = (await store.get_record(feature_id)).state.model_copy(deep=True)
    state.status = FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    state.child_workflows = {
        "backend": ChildWorkflowReference(
            child_workflow_id=f"{feature_id}:backend",
            repository_id="backend",
            workstream_id="ws-backend",
            status=ChildWorkflowStatus.FAILED,
            branch_name="feature/audit",
            workspace_path=f"/workspaces/{feature_id}/backend",
            retry_count=1,
        )
    }
    await store._replace_state(feature_id, state, "race_setup")  # noqa: SLF001


async def test_a_cancel_and_a_granted_retry_produce_one_terminal_state_not_two(
    postgres_database: Database,
) -> None:
    """A feature must not end both cancelled and retrying, whichever request arrives first.

    The grant is queued work now, so the two do not contend for one lock and cannot be
    ordered by one. What decides is the claim: a cancelled feature drops it, and a cancelling
    one runs into the cancellation the step itself raises.
    """
    store = _control_plane(postgres_database)
    feature_id = await _accept(store, "cancel-retry")
    await _dispatcher(store).drain()
    await _stopped_workstream(store, feature_id)

    grant, _cancelled = await asyncio.gather(
        store.retry_workstream(
            feature_id,
            "backend",
            additional_attempts=1,
            requested_by="operator",
            reason="flaky validation",
            credentials=CREDENTIALS,
        ),
        store.cancel(
            feature_id, reason="an operator stopped this feature", requested_by="platform-admin"
        ),
        return_exceptions=True,
    )
    if isinstance(grant, WorkflowConflictError):
        # The cancel got there first and the grant was refused synchronously, which is the
        # other legal ordering: nothing was queued, so nothing can be silently consumed.
        assert await store.queue.active_intent(feature_id) is None
    await _dispatcher(store).drain()

    record = await store.get_record(feature_id)
    assert record.state.cancellation_requested is True
    assert record.state.status in {
        FeatureWorkflowStatus.CANCELLING,
        FeatureWorkflowStatus.CANCELLED,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    }
    # The grant did not also take effect: the repository is not running another attempt.
    assert record.state.child_workflows["backend"].status is not ChildWorkflowStatus.RUNNING
    entry = await _queue_row(postgres_database, feature_id)
    assert entry.status in {SUCCEEDED, "failed"}, "the entry must settle rather than loop"


# --------------------------------------------------------------------------------------
# 6. A granted retry racing the abandoned-run sweep
# --------------------------------------------------------------------------------------


async def test_the_sweep_does_not_tombstone_a_granted_retry_a_worker_is_holding(
    postgres_database: Database,
) -> None:
    """A live claim is evidence of a worker; the sweep exists for the absence of one.

    Both read the same evidence and neither takes a lock the other waits on, so the only
    thing coordinating them is what the rows say at the moment each looks. Under a grant that
    is a queue entry a worker has just leased, and tombstoning it would end a feature
    somebody had explicitly asked to continue -- and would write a terminal state over the
    one the attempt is about to write.

    Both orderings are asserted, because only one of them is safe by accident. While the
    lease is live the sweep must decide nothing; once it has lapsed the sweep is the one
    entitled to decide, and the entry must then be settled rather than reopened.
    """
    store = _control_plane(postgres_database)
    feature_id = await _accept(store, "sweep-retry")
    # Run the feature first: a grant reads the planning artifacts, so a feature that never
    # planned has nothing to grant an attempt against.
    await _dispatcher(store).drain()
    await _stopped_workstream(store, feature_id)
    await store.retry_workstream(
        feature_id,
        "backend",
        additional_attempts=1,
        requested_by="operator",
        reason="flaky validation",
        credentials=CREDENTIALS,
    )
    # The shape a granted retry has once its claim has started work: the feature says it is
    # executing, the entry is leased, and the two are indistinguishable from an abandoned run
    # by anything except the lease.
    await _mark_executing(store, feature_id)
    queue = DatabaseFeatureExecutionQueue(postgres_database)
    claimed = await queue.claim(owner="grant-worker", lease_seconds=120)
    assert claimed is not None and claimed.intent == "retry_workstream"
    await _backdate(postgres_database, feature_id)

    swept, disposition = await asyncio.gather(
        store.reconcile_abandoned_runs(stale_after_seconds=0.01),
        store._claim_disposition(  # noqa: SLF001
            (await store.get_record(feature_id)).state, claimed.intent
        ),
    )

    assert swept == [], "the sweep tombstoned a feature a worker had claimed"
    assert disposition is not ClaimDisposition.DROP, "the grant this worker paid for was dropped"
    assert (await store.get_record(feature_id)).state.status is (
        FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
    )

    # The other ordering: the worker died, the lease lapsed, and the sweep is now the only
    # thing that can decide. Exactly one terminal state, and the entry does not reopen.
    async with postgres_database.session() as session:
        await session.execute(
            update(FeatureExecutionQueueModel)
            .where(FeatureExecutionQueueModel.feature_id == feature_id)
            .values(lease_expires_at=datetime.now(UTC) - timedelta(minutes=5))
        )
        await session.commit()
    await _backdate(postgres_database, feature_id)

    assert await store.reconcile_abandoned_runs(stale_after_seconds=0.01) == [feature_id]
    decided = (await store.get_record(feature_id)).state
    assert decided.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert decided.failure_summary is not None
    assert await store.reconcile_abandoned_runs(stale_after_seconds=0.01) == [], (
        "the sweep decided the same feature twice"
    )
    assert (await store.get_record(feature_id)).state.status is (
        FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    )


async def _mark_executing(store: SqlAlchemyFeatureControlPlane, feature_id: str) -> None:
    """Leave the feature in the executing shape a granted retry's first step produces."""
    state = (await store.get_record(feature_id)).state.model_copy(deep=True)
    state.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
    state.child_workflows = {
        "backend": state.child_workflows["backend"].model_copy(
            update={"status": ChildWorkflowStatus.RUNNING}
        )
    }
    await store._replace_state(feature_id, state, "child_workflows_started")  # noqa: SLF001


async def _backdate(database: Database, feature_id: str) -> None:
    """Age one feature past the grace period, as the passage of time does."""
    async with database.session() as session:
        await session.execute(
            update(FeatureWorkflowModel)
            .where(FeatureWorkflowModel.feature_id == feature_id)
            .values(updated_at=datetime.now(UTC) - timedelta(minutes=5))
        )
        await session.commit()


# --------------------------------------------------------------------------------------
# 7. Two state writers
# --------------------------------------------------------------------------------------


async def test_two_concurrent_state_writers_lose_neither_write_nor_any_artifact(
    postgres_database: Database,
) -> None:
    """Artifacts are append-only and their identifiers are immutable, under contention.

    Each writer holds a snapshot taken before the other's write, which is precisely the
    situation the merge exists for: a slow coroutine returning state that predates finished
    work must not erase it.
    """
    store = _control_plane(postgres_database)
    feature_id = await _accept(store, "two-writers")
    base = (await store.get_record(feature_id)).state
    before = {item.artifact_id for item in base.artifacts}

    async def write(suffix: str) -> None:
        state = (await store.get_record(feature_id)).state.model_copy(deep=True)
        state.current_agent = f"writer-{suffix}"
        state.child_workflows = {
            **state.child_workflows,
            suffix: ChildWorkflowReference(
                child_workflow_id=f"{feature_id}:{suffix}",
                repository_id=suffix,
                workstream_id=f"ws-{suffix}",
                status=ChildWorkflowStatus.APPROVED,
                branch_name=f"feature/{suffix}",
                workspace_path=f"/workspaces/{feature_id}/{suffix}",
                retry_count=0,
            ),
        }
        await store._replace_state(feature_id, state, f"write_{suffix}")  # noqa: SLF001

    await asyncio.gather(write("alpha"), write("beta"))

    record = await store.get_record(feature_id)
    assert {"alpha", "beta"} <= set(record.state.child_workflows)
    identifiers = [item.artifact_id for item in record.state.artifacts]
    assert len(identifiers) == len(set(identifiers)), "an artifact identifier was reused"
    assert before <= set(identifiers), "a concurrent write dropped an existing artifact"


# --------------------------------------------------------------------------------------
# 8. A duplicated submission with one idempotency key
# --------------------------------------------------------------------------------------


async def test_six_identical_submissions_produce_one_feature_and_one_queue_entry(
    postgres_database: Database,
) -> None:
    """One reference number is what a person sees, so two would be two features to them."""
    keys = SharedKeyStore()
    # The deployment's own acquisition budget, not the short one the rest of this module uses
    # to make a refusal immediate. Six duplicate submissions all get in and all replay; a
    # shorter budget would turn the last of them into "retry shortly", which is a legitimate
    # answer but a different property from the one this row is about.
    store = _control_plane(postgres_database, keys=keys, lock_timeout=10.0)
    request = StartFeatureRequest.model_validate({**feature_payload(), "feature_id": "f-once"})

    results = await asyncio.gather(
        *(
            store.start(
                request,
                idempotency_key="submitted-once",
                credentials=CREDENTIALS,
                owner_id="platform-admin",
            )
            for _ in range(6)
        ),
        return_exceptions=True,
    )

    raised = [item for item in results if isinstance(item, BaseException)]
    assert raised == [], f"a concurrent submission raised instead of replaying: {raised}"
    accepted = [item for item in results if not isinstance(item, BaseException)]
    assert len({item.record.state.feature_id for item in accepted}) == 1
    assert len([item for item in accepted if item.created]) == 1
    references = {item.record.state.reference for item in accepted}
    assert len(references) == 1, f"one submission produced more than one reference: {references}"
    async with postgres_database.session() as session:
        features = list(await session.scalars(select(FeatureWorkflowModel.feature_id)))
        entries = list(await session.scalars(select(FeatureExecutionQueueModel.feature_id)))
    assert len(features) == 1
    assert len(entries) == 1


# --------------------------------------------------------------------------------------
# 9. A step re-queue racing another feature's claim
# --------------------------------------------------------------------------------------


async def test_a_long_feature_re_queueing_itself_does_not_starve_the_one_behind_it(
    postgres_database: Database,
) -> None:
    """`advance_to_next_step` moves `queued_at` forward, so a step takes its turn at the back.

    Without that, a feature with forty steps holds every worker for its whole life and a
    one-step feature queued a second later waits for all of them. The property is fairness,
    and the evidence is the interleaving of the steps that actually ran.
    """
    runner = StepRecordingOrchestrator()
    store = _control_plane(postgres_database, runner=runner)
    long_feature = await _accept(store, "long")
    await asyncio.sleep(0.01)
    short_feature = await _accept(store, "short")

    await _dispatcher(store).drain()

    order = runner.steps
    assert runner.duplicated() == []
    first_short = next(index for index, item in enumerate(order) if item.startswith(short_feature))
    last_long = max(index for index, item in enumerate(order) if item.startswith(long_feature))
    assert first_short < last_long, (
        f"the second feature waited for the first to finish every step: {order}"
    )
    for feature_id in (long_feature, short_feature):
        assert (await store.get_record(feature_id)).state.status is FeatureWorkflowStatus.COMPLETED


# --------------------------------------------------------------------------------------
# 10. A lease lost mid-step, and the fence that stops the loser writing
# --------------------------------------------------------------------------------------


async def test_a_run_that_loses_its_claim_cannot_write_to_the_feature_afterwards(
    postgres_database: Database,
) -> None:
    """The revoked-claim fence, under the contention it was written for.

    AB-Feature-108: the lease lapsed at 15:22, recovery reconciled the action at 15:36, and
    at 15:53 the run that had already been declared dead wrote a ninth attempt into the
    feature. Cancellation is a request, and a coroutine blocked in a fifteen-minute test
    command unwinds when that command ends -- not when `cancel()` is called. The window is
    real, and this is it.

    Contention is produced rather than simulated: a second owner takes the action's lease
    through the store's own compare-and-set, which is what makes the first executor's
    renewal fail.
    """
    store = _control_plane(postgres_database)
    feature_id = await _accept(store, "revoked")
    actions = DatabaseFeatureActionStore(postgres_database, lease_seconds=1)
    service = FeatureActionService(store=actions)
    action, _created = await service.submit(
        feature_id=feature_id,
        action_type="RETRY_WORKSTREAM",
        actor_id="operator",
        payload={"reason": "flaky validation"},
        context_version="v1",
        # Two, so recovery can genuinely take the claim. One attempt would make the second
        # claim fail on budget rather than on ownership, and the contention would be a
        # fiction: nothing would ever have taken this executor's lease away.
        max_attempts=2,
    )
    before = (await store.get_record(feature_id)).state.current_agent
    started = asyncio.Event()
    late_write: list[str] = []

    async def run() -> str:
        """Work that outlives its own claim, and tries to write when it finally unwinds."""
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state = (await store.get_record(feature_id)).state.model_copy(deep=True)
            state.current_agent = "zombie"
            try:
                await store._replace_state(feature_id, state, "late_write")  # noqa: SLF001
            except FeatureRunRevokedError:
                late_write.append("refused")
            else:
                late_write.append("accepted")
            raise
        return "unreachable"

    try:
        execution = asyncio.create_task(service.execute(action, run))
        await asyncio.wait_for(started.wait(), timeout=10)
        # Another owner takes the lease, exactly as recovery does once it has lapsed.
        await _expire_action_lease(postgres_database, action.action_id)
        stolen = await actions.claim(action.action_id, owner="recovery-worker")
        assert stolen is not None, "the lease could not be taken, so nothing was contended"

        with pytest.raises(ActionLeaseLostError):
            await execution

        assert late_write == ["refused"], (
            "a run that had lost its claim wrote to the feature anyway; this is the "
            "AB-Feature-108 shape and the fence did not hold"
        )
        assert (await store.get_record(feature_id)).state.current_agent == before
        # The row is left claimed with a lapsed lease, which is the shape recovery looks for.
        current = await actions.get(action.action_id)
        assert current is not None
        assert current.status is not FeatureActionStatus.SUCCEEDED
    finally:
        clear_revoked_feature_actions()


async def _expire_action_lease(database: Database, action_id: str) -> None:
    """Age one action's lease past its expiry, as a partitioned executor does."""
    async with database.session() as session:
        await session.execute(
            update(FeatureActionModel)
            .where(FeatureActionModel.action_id == action_id)
            .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        await session.commit()
