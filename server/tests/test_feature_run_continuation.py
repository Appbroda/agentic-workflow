"""A feature whose executor died, picked back up by the next claim rather than tombstoned.

Audit risk P0-2, first half. Every checkpoint boundary was already written and nothing
consumed them: a start claim was valid only on a `pending` feature, so the claim that arrived
after a worker died was dropped as unwanted, the queue entry closed `succeeded`, and
forty-five minutes later the abandoned-run sweep wrote "start a fresh feature" over a run
whose whole state was sitting there. `_recover_from_checkpoint` existed, worked, and was
reachable only from a human pressing resume.

These tests drive the real dispatcher against a real durable store. The interrupted snapshots
are the orchestrator's own checkpoints, produced by running the accepted feature until it
reaches the window in question and then installing that boundary as the feature's persisted
state; the queue rows are set to what a killed worker leaves, a claim in `running` whose
lease has lapsed. `tests/test_crash_windows.py` boundary 7 produces the identical rows by
killing a real process, which is what proves this is the shape a dead worker leaves rather
than one these tests invented.

Mock execution reaches no repository, so clone, commit and push counts are not assertable
here; boundaries 1, 3 and 4 of the crash suite own those. What is assertable here is every
stage above them -- the product manager, reconnaissance, the planner, each child attempt, the
integration verdict and the pull request -- and each is counted on a double that holds state.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import update

from adapters.github_adapter import MockGitHubService
from api.control_plane import RequestScopedCredentials
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import (
    IntegrationContractArtifact,
    PullRequestArtifact,
    RepositoryExecutionPlanArtifact,
    TechnicalPRDArtifact,
)
from configs.settings import load_settings
from services.cancellation import MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.feature_queue import FAILED, QUEUED, RUNNING, SUCCEEDED
from services.journaled_github import JournaledGitHubService
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.external_operations import ExternalOperationStatus, ExternalOperationType
from state.feature_models import FeatureWorkflowSnapshot
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from storage.feature_store import SqlAlchemyFeatureControlPlane
from storage.models import (
    ExternalOperationModel,
    FeatureExecutionQueueModel,
    FeatureWorkflowModel,
)
from tests.support import drain_feature_queue
from tests.test_feature_api import feature_payload
from tests.test_feature_workflow import (
    CheckpointRecorder,
    CountingIntegrationReviewer,
    LoopProbeExecutor,
)
from workflows.feature_workflow import (
    DeterministicFeatureProductManager,
    FeatureWorkflowOrchestrator,
    GitHubPullRequestPublisher,
    NullRepositoryReconnaissance,
    resume_eligible_repository_ids,
)

pytestmark = pytest.mark.asyncio

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)
FEATURE = "feature-login"

# The script that makes the fan-out pass through the window this task is about. Both
# repositories are asked for changes and both eventually converge, but `frontend` needs one
# attempt more than `backend`. The two children run as concurrent coroutines with the same
# number of suspension points per attempt, so `backend`'s second attempt settles before
# `frontend`'s third: there is always a boundary where one repository is durably approved and
# the other is durably part-way through a numbered attempt. Scripting one repository to fail
# outright instead made that boundary depend on which coroutine the loop resumed first.
_INTERLEAVED = {
    "backend": ["changes_requested", "approved"],
    "frontend": ["changes_requested", "changes_requested", "approved"],
}


# --------------------------------------------------------------------------------------
# Scaffolding
# --------------------------------------------------------------------------------------


async def _store(
    tmp_path: Path, name: str, runner: FeatureWorkflowOrchestrator
) -> tuple[Database, SqlAlchemyFeatureControlPlane]:
    """A real durable control plane over a real database, as the deployment composes it."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / name}")
    await database.create_schema()
    return database, SqlAlchemyFeatureControlPlane(database, mock_runner=runner)


async def _accept(
    store: SqlAlchemyFeatureControlPlane, feature_id: str = FEATURE
) -> FeatureWorkflowSnapshot:
    """Accept one feature through the real path and return what acceptance persisted.

    The interrupted snapshots below are produced from *this* state rather than from a
    separately built one, because artifact identity is immutable: a second Product
    Requirement artifact with the same ID and a different timestamp is refused by the durable
    merge, correctly.
    """
    request = StartFeatureRequest.model_validate({**feature_payload(), "feature_id": feature_id})
    result = await store.start(
        request, idempotency_key=feature_id, credentials=CREDENTIALS, owner_id="platform-admin"
    )
    return result.record.state


async def _checkpoints(
    accepted: FeatureWorkflowSnapshot,
    *,
    orchestrator: FeatureWorkflowOrchestrator | None = None,
    recorder: CheckpointRecorder | None = None,
) -> CheckpointRecorder:
    """Run the accepted feature through to the end, keeping every boundary it wrote."""
    kept = recorder or CheckpointRecorder()
    runner = orchestrator or FeatureWorkflowOrchestrator(
        child_executor=cast(Any, LoopProbeExecutor(_INTERLEAVED)), checkpoint_writer=kept
    )
    await runner.start(accepted.model_copy(deep=True), credentials=CREDENTIALS)
    return kept


def _child(snapshot: FeatureWorkflowSnapshot, repository_id: str) -> Any:
    """One child if the run had reached it, or `None` at a boundary before it existed."""
    return snapshot.child_workflows.get(repository_id)


def _by_status(snapshot: FeatureWorkflowSnapshot, status: ChildWorkflowStatus) -> list[str]:
    """The repositories a snapshot records in one child status, in a stable order."""
    return sorted(
        repository_id
        for repository_id, child in snapshot.child_workflows.items()
        if child.status is status
    )


async def _mid_fan_out(accepted: FeatureWorkflowSnapshot) -> FeatureWorkflowSnapshot:
    """The boundary where one repository is approved and the other is mid-attempt.

    *Which* repository gets there first is not fixed -- the children are concurrent coroutines
    and the event loop decides -- so this asks for the shape rather than for named
    repositories. The shape itself is guaranteed: both converge, and whichever settles first
    leaves the other part-way through a numbered attempt.
    """
    recorder = await _checkpoints(accepted)
    return recorder.last(
        lambda snapshot, boundary, repository_id: (
            bool(_by_status(snapshot, ChildWorkflowStatus.APPROVED))
            and bool(_by_status(snapshot, ChildWorkflowStatus.RUNNING))
        )
    )


async def _both_running(accepted: FeatureWorkflowSnapshot) -> FeatureWorkflowSnapshot:
    """The boundary where neither repository has settled, so both are still schedulable."""
    recorder = await _checkpoints(accepted)
    return recorder.last(
        lambda snapshot, boundary, repository_id: (
            len(snapshot.child_workflows) == 2
            and all(
                child.status is ChildWorkflowStatus.RUNNING
                for child in snapshot.child_workflows.values()
            )
        )
    )


async def _install(
    store: SqlAlchemyFeatureControlPlane, state: FeatureWorkflowSnapshot, *, event: str = "test"
) -> None:
    """Persist a mid-flight snapshot as the feature's durable state."""
    await store._replace_state(state.feature_id, state, event)  # noqa: SLF001


async def _dead_claim(database: Database, *, attempt: int = 1) -> None:
    """Leave the queue row in the state a killed worker leaves: claimed, lease lapsed."""
    lapsed = datetime.now(UTC) - timedelta(minutes=5)
    async with database.session() as session:
        await session.execute(
            update(FeatureExecutionQueueModel)
            .where(FeatureExecutionQueueModel.feature_id == FEATURE)
            .values(
                status=RUNNING,
                attempt=attempt,
                lease_owner="worker-that-died",
                lease_expires_at=lapsed,
                started_at=lapsed,
            )
        )
        await session.commit()


async def _settle_queue_entry(database: Database) -> None:
    """Close the queue entry out, the way a spent or dropped claim leaves it."""
    async with database.session() as session:
        await session.execute(
            update(FeatureExecutionQueueModel)
            .where(FeatureExecutionQueueModel.feature_id == FEATURE)
            .values(status=SUCCEEDED, lease_owner=None, lease_expires_at=None)
        )
        await session.commit()


async def _hold_queue_entry(database: Database) -> None:
    """Give the queue entry a live claim, which is what a continuation in flight looks like."""
    async with database.session() as session:
        await session.execute(
            update(FeatureExecutionQueueModel)
            .where(FeatureExecutionQueueModel.feature_id == FEATURE)
            .values(
                status=RUNNING,
                attempt=2,
                lease_owner="worker-continuing-it",
                lease_expires_at=datetime.now(UTC) + timedelta(minutes=2),
            )
        )
        await session.commit()


async def _queue_row(database: Database) -> FeatureExecutionQueueModel:
    """Read one queue entry as a later reader sees it."""
    async with database.session() as session:
        row = await session.get(FeatureExecutionQueueModel, FEATURE)
        assert row is not None
        return row


async def _settled_by_the_queue(store: SqlAlchemyFeatureControlPlane, database: Database) -> bool:
    """Drain the queue and report whether this feature's entry was claimed and closed.

    What the drain's own return value used to say. It counted claims, and while a claim was a
    whole feature that number was 1 for every case here -- so `== 1` read as "the entry was
    claimed and not dropped". A claim is one step now and the count is however many steps the
    state each test installs still owes, which is not the property any of them are about. This
    asserts the property directly instead, and more exactly than the count did: the entry was
    claimable, something ran it, and it ended settled rather than left open.
    """
    if await drain_feature_queue(store) < 1:
        return False
    return (await _queue_row(database)).status in {SUCCEEDED, FAILED}


async def _backdate(database: Database, moment: datetime) -> None:
    """Move a feature's persisted timestamps back, so a sweep sees a stale run."""
    async with database.session() as session:
        model = await session.get(FeatureWorkflowModel, FEATURE)
        assert model is not None
        state_json = dict(model.state_json)
        state_json["updated_at"] = moment.isoformat()
        await session.execute(
            update(FeatureWorkflowModel)
            .where(FeatureWorkflowModel.feature_id == FEATURE)
            .values(updated_at=moment, state_json=state_json)
        )
        await session.commit()


async def _events(store: SqlAlchemyFeatureControlPlane) -> list[str]:
    """The names of everything recorded against this feature, in order."""
    return [event for *_, event, _details in await store.timeline(FEATURE)]


def _artifact_ids(state: FeatureWorkflowSnapshot, kind: type) -> list[str]:
    """The identity of every artifact of one kind, so reuse can be told from regeneration."""
    return [item.artifact_id for item in state.artifacts if isinstance(item, kind)]


class CountingProductManager:
    """The real deterministic product manager, counting how often it is asked."""

    def __init__(self) -> None:
        """Start with nothing asked of it."""
        self._delegate = DeterministicFeatureProductManager()
        self.calls = 0

    async def create_technical_prd(self, **kwargs: Any) -> Any:
        """Produce the same Technical PRD, and record that it was asked for."""
        self.calls += 1
        return await self._delegate.create_technical_prd(**kwargs)


class CountingReconnaissance:
    """The reconnaissance this mode uses, counting how often a checkout is inspected."""

    def __init__(self) -> None:
        """Start with nothing inspected."""
        self._delegate = NullRepositoryReconnaissance()
        self.calls = 0

    async def inspect(self, **kwargs: Any) -> Any:
        """Inspect exactly as the real one does, and record the call."""
        self.calls += 1
        return await self._delegate.inspect(**kwargs)


# --------------------------------------------------------------------------------------
# Continuation
# --------------------------------------------------------------------------------------


async def test_a_claim_on_a_feature_whose_executor_died_continues_it(tmp_path: Path) -> None:
    """The defect itself: this claim used to be dropped and the run tombstoned instead.

    The approved repository is not implemented a second time and the mid-attempt one picks up
    at the attempt its checkpoint holds, which is the difference between continuing a run and
    starting one over on top of the effects the first one already had.
    """
    executor = LoopProbeExecutor()
    database, store = await _store(
        tmp_path, "continued.db", FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))
    )
    try:
        interrupted = await _mid_fan_out(await _accept(store))
        (settled,) = _by_status(interrupted, ChildWorkflowStatus.APPROVED)
        (unfinished,) = _by_status(interrupted, ChildWorkflowStatus.RUNNING)
        await _install(store, interrupted)
        await _dead_claim(database)

        assert await _settled_by_the_queue(store, database)

        state = (await store.get_record(FEATURE)).state
        # The approved sibling was never asked again; only the unfinished repository ran.
        assert executor.attempts(settled) == 0
        assert executor.attempts(unfinished) == 1
        assert state.child_workflows[settled].status is ChildWorkflowStatus.COMPLETED
        # And it continued the persisted attempt rather than restarting the count. The probe
        # keys its revision on the attempt the durable child is on, so this is the difference
        # between resuming attempt N and handing the repository its whole budget again.
        persisted_attempt = interrupted.child_workflows[unfinished].retry_count
        assert persisted_attempt > 0
        assert executor.revisions[unfinished] == [f"{unfinished}-rev-{persisted_attempt}"]
        assert state.child_workflows[unfinished].retry_count >= persisted_attempt
        events = await _events(store)
        assert "feature_run_continued" in events
        assert "feature_run_abandoned" not in events
    finally:
        await database.drop_schema()
        await database.dispose()


async def test_a_continued_run_reuses_planning_instead_of_repeating_it(tmp_path: Path) -> None:
    """A feature that died after planning must not pay for planning twice.

    The product manager, reconnaissance and the planner each cost a model call and each
    produces an artifact the rest of the run is keyed on. Re-running them would also
    regenerate the contract the children were reviewed against.
    """
    product_manager = CountingProductManager()
    reconnaissance = CountingReconnaissance()
    database, store = await _store(
        tmp_path,
        "planning.db",
        FeatureWorkflowOrchestrator(
            product_manager=cast(Any, product_manager),
            reconnaissance=cast(Any, reconnaissance),
        ),
    )
    try:
        recorder = await _checkpoints(await _accept(store))
        interrupted = recorder.last(
            lambda snapshot, boundary, repository_id: (
                any(
                    isinstance(item, RepositoryExecutionPlanArtifact) for item in snapshot.artifacts
                )
                and bool(snapshot.child_workflows)
                and all(
                    child.status is ChildWorkflowStatus.PENDING
                    for child in snapshot.child_workflows.values()
                )
            )
        )
        await _install(store, interrupted)
        await _dead_claim(database)

        assert await _settled_by_the_queue(store, database)

        assert (product_manager.calls, reconnaissance.calls) == (0, 0)
        state = (await store.get_record(FEATURE)).state
        for kind in (
            TechnicalPRDArtifact,
            IntegrationContractArtifact,
            RepositoryExecutionPlanArtifact,
        ):
            assert _artifact_ids(state, kind) == _artifact_ids(interrupted, kind), (
                f"the continuation regenerated a {kind.__name__} that already existed"
            )

        # A control, because "zero calls" is also what a double nobody wired would report.
        # A second feature accepted on the same store goes through both of them once.
        await _accept(store, "feature-control")
        assert await _settled_by_the_queue(store, database)
        assert (product_manager.calls, reconnaissance.calls) == (1, 1)
    finally:
        await database.drop_schema()
        await database.dispose()


async def test_a_continued_run_reuses_the_integration_verdict_it_already_paid_for(
    tmp_path: Path,
) -> None:
    """A crash during integration review must not buy the same verdict twice.

    `_integration_review_input_signature` is what makes the persisted verdict reusable; this
    asserts the continuation actually routes it rather than asking the reviewer again, and
    that the cycle the interrupted run left uncharged is charged exactly once.
    """
    reviewer = CountingIntegrationReviewer()
    database, store = await _store(
        tmp_path,
        "verdict.db",
        FeatureWorkflowOrchestrator(
            integration_reviewer=cast(Any, reviewer),
            pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
        ),
    )
    try:
        shared = CheckpointRecorder()
        recorder = await _checkpoints(
            await _accept(store),
            orchestrator=FeatureWorkflowOrchestrator(
                integration_reviewer=cast(Any, CountingIntegrationReviewer()),
                pull_request_publisher=GitHubPullRequestPublisher(
                    github_service=MockGitHubService()
                ),
                checkpoint_writer=shared,
            ),
            recorder=shared,
        )
        interrupted = recorder.last(
            lambda snapshot, boundary, repository_id: (
                snapshot.integration_review_cycles == 0
                and any(item.artifact_type == "integration_review" for item in snapshot.artifacts)
            )
        )
        await _install(store, interrupted)
        await _dead_claim(database)

        assert await _settled_by_the_queue(store, database)

        assert reviewer.calls == 0, "the continuation bought the integration verdict again"
        assert (await store.get_record(FEATURE)).state.integration_review_cycles == 1
    finally:
        await database.drop_schema()
        await database.dispose()


async def test_a_continued_feature_opens_one_pull_request_per_repository(
    tmp_path: Path,
) -> None:
    """The effect a duplicated publication would produce, counted on the provider double.

    The double holds pull requests rather than counting calls, so this is a statement about
    what the provider ended up with -- which is the only version of "published once" that
    matters.
    """
    published = MockGitHubService()
    database, store = await _store(
        tmp_path,
        "publish.db",
        FeatureWorkflowOrchestrator(
            pull_request_publisher=GitHubPullRequestPublisher(github_service=published)
        ),
    )
    try:
        shared = CheckpointRecorder()
        recorder = await _checkpoints(
            await _accept(store),
            orchestrator=FeatureWorkflowOrchestrator(
                pull_request_publisher=GitHubPullRequestPublisher(
                    github_service=MockGitHubService()
                ),
                checkpoint_writer=shared,
            ),
            recorder=shared,
        )
        interrupted = recorder.last(
            lambda snapshot, boundary, repository_id: (
                len(snapshot.child_workflows) == 2
                and all(
                    child.status is ChildWorkflowStatus.APPROVED
                    for child in snapshot.child_workflows.values()
                )
                and not any(isinstance(item, PullRequestArtifact) for item in snapshot.artifacts)
            )
        )
        await _install(store, interrupted)
        await _dead_claim(database)

        assert await _settled_by_the_queue(store, database)

        state = (await store.get_record(FEATURE)).state
        repositories = {item.repository for item in published.pull_requests.values()}
        assert len(published.pull_requests) == len(repositories) == len(state.repository_specs)
        assert len(_artifact_ids(state, PullRequestArtifact)) == len(state.repository_specs)
    finally:
        await database.drop_schema()
        await database.dispose()


async def test_a_continued_feature_settles_the_deferred_effect_its_crash_left(
    tmp_path: Path,
) -> None:
    """The consequence task 27- surfaced, and the reason this task closes it.

    An interrupted push or pull request is left `awaiting_reconciliation` deliberately: the
    sweep holds no credentials and the next credentialed request through the same code path
    can prove what happened. Before this task no such request came by itself -- a crashed
    feature was tombstoned, so the deferred row waited for a human who might never arrive.
    A continued feature re-enters the publication path and reconciles its own operation.

    The pull requests here really exist in the provider double and the journal rows are in
    the durable shape the sweep leaves. Adoption -- not a second pull request -- is the claim.
    """
    provider = MockGitHubService()
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'deferred.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)

    def _publisher() -> GitHubPullRequestPublisher:
        """The publisher a deployment builds: journaled, over a state-holding double."""
        return GitHubPullRequestPublisher(
            github_service=JournaledGitHubService(
                provider,
                operation_executor=ExternalOperationExecutor(
                    journal=journal,
                    cancellation_token=MockCancellationToken(),
                    scope=ExternalOperationScope(workflow_id=FEATURE, feature_id=FEATURE),
                    heartbeat_seconds=3_600.0,
                ),
            )
        )

    try:
        store = SqlAlchemyFeatureControlPlane(
            database, mock_runner=FeatureWorkflowOrchestrator(pull_request_publisher=_publisher())
        )
        shared = CheckpointRecorder()
        recorder = await _checkpoints(
            await _accept(store),
            orchestrator=FeatureWorkflowOrchestrator(
                pull_request_publisher=_publisher(), checkpoint_writer=shared
            ),
            recorder=shared,
        )
        created = len(provider.pull_requests)
        assert created == 2, "the run this rewinds did not publish both repositories"

        interrupted = recorder.last(
            lambda snapshot, boundary, repository_id: (
                len(snapshot.child_workflows) == 2
                and all(
                    child.status is ChildWorkflowStatus.APPROVED
                    for child in snapshot.child_workflows.values()
                )
                and not any(isinstance(item, PullRequestArtifact) for item in snapshot.artifacts)
            )
        )
        # Put the journal back into the shape a crash between the provider call and the
        # journal write leaves, which is what the credential-free sweep then defers.
        async with database.session() as session:
            await session.execute(
                update(ExternalOperationModel)
                .where(
                    ExternalOperationModel.operation_type
                    == ExternalOperationType.CREATE_PULL_REQUEST
                )
                .values(
                    status=ExternalOperationStatus.AWAITING_RECONCILIATION,
                    error_code="awaiting_credentialed_reconciliation",
                    external_reference=None,
                    result_payload=None,
                )
            )
            await session.commit()
        assert len(await journal.list_unresolved_operations()) == 2
        # The consequence 27- surfaced: no sweep will ever look at these rows again.
        assert await journal.list_incomplete_operations() == []

        await _install(store, interrupted)
        await _dead_claim(database)

        assert await _settled_by_the_queue(store, database)

        assert len(provider.pull_requests) == created, "publication opened a second pull request"
        pull_requests = [
            item
            for item in await journal.list_operations_for_workflow(FEATURE)
            if item.operation_type is ExternalOperationType.CREATE_PULL_REQUEST
        ]
        assert len(pull_requests) == 2
        assert all(item.status is ExternalOperationStatus.SUCCEEDED for item in pull_requests)
        assert await journal.list_unresolved_operations() == []
    finally:
        await database.drop_schema()
        await database.dispose()


# --------------------------------------------------------------------------------------
# The attempt budget is the bound
# --------------------------------------------------------------------------------------


async def test_a_feature_that_crashes_past_its_attempt_budget_is_not_continued_again(
    tmp_path: Path,
) -> None:
    """Three crashes is a feature something is systematically wrong with.

    The queue entry's budget is the safety, and it is deliberately not raised to make
    continuation feel safer. What must not happen is a feature continued for ever by workers
    that each die on it.
    """
    executor = LoopProbeExecutor()
    database, store = await _store(
        tmp_path, "budget.db", FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))
    )
    try:
        interrupted = await _mid_fan_out(await _accept(store))
        await _install(store, interrupted)
        row = await _queue_row(database)
        await _dead_claim(database, attempt=row.max_attempts)

        assert await _settled_by_the_queue(store, database)

        assert executor.engineer_calls == {}, "a spent budget still bought an attempt"
        state = (await store.get_record(FEATURE)).state
        assert state.status is FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
        assert "feature_run_continued" not in await _events(store)
    finally:
        await database.drop_schema()
        await database.dispose()


async def test_the_terminal_record_says_it_was_continued_rather_than_to_start_again(
    tmp_path: Path,
) -> None:
    """The diagnostic this task deliberately rewrites.

    "Start a fresh feature rather than resuming this one" was correct while nothing continued
    a crashed run. Reaching this status now means the platform did continue it and it did not
    survive, and the record has to say that instead.
    """
    executor = LoopProbeExecutor()
    database, store = await _store(
        tmp_path, "exhausted.db", FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))
    )
    try:
        interrupted = await _mid_fan_out(await _accept(store))
        await _install(store, interrupted)
        await _dead_claim(database)
        assert await _settled_by_the_queue(store, database)
        assert "feature_run_continued" in await _events(store)

        # The continuation died too, leaving the same shape again: executing, stale, with a
        # settled entry no further claim will take.
        crashed = (await store.get_record(FEATURE)).state.model_copy(deep=True)
        crashed.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
        crashed.failure_summary = None
        await _install(store, crashed, event="test_recrash")
        await _settle_queue_entry(database)
        await _backdate(database, datetime.now(UTC) - timedelta(hours=2))

        assert await store.reconcile_abandoned_runs(stale_after_seconds=60) == [FEATURE]

        summary = (await store.get_record(FEATURE)).state.failure_summary
        assert summary is not None
        assert not any("Start a fresh feature" in item for item in summary.diagnostics)
        assert any("continued this run from its last checkpoint" in i for i in summary.diagnostics)
        assert any("did not survive" in item for item in summary.diagnostics)
        timeline = await store.timeline(FEATURE)
        abandoned = [details for *_, event, details in timeline if event == "feature_run_abandoned"]
        assert [item["continuations"] for item in abandoned] == [1]
    finally:
        await database.drop_schema()
        await database.dispose()


# --------------------------------------------------------------------------------------
# Resting states
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        FeatureWorkflowStatus.WAITING_FOR_HUMAN,
        FeatureWorkflowStatus.CONTRACT_READY,
        FeatureWorkflowStatus.CHANGES_REQUESTED,
        FeatureWorkflowStatus.READY_FOR_PULL_REQUESTS,
    ],
)
async def test_a_resting_feature_is_neither_continued_nor_tombstoned(
    tmp_path: Path, status: FeatureWorkflowStatus
) -> None:
    """Nothing is running and nothing is wrong; a person owns these.

    Continuing one would run work nobody asked for, and tombstoning one would end a feature
    that is waiting exactly as designed. The sweep already excluded them; the claim decision
    has to make the same distinction, from the same set.
    """
    executor = LoopProbeExecutor()
    database, store = await _store(
        tmp_path, "resting.db", FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))
    )
    try:
        resting = (await _mid_fan_out(await _accept(store))).model_copy(deep=True)
        resting.status = status
        await _install(store, resting)
        await _dead_claim(database)

        assert await _settled_by_the_queue(store, database)
        await _backdate(database, datetime.now(UTC) - timedelta(hours=2))
        assert await store.reconcile_abandoned_runs(stale_after_seconds=60) == []

        state = (await store.get_record(FEATURE)).state
        assert state.status is status
        assert executor.engineer_calls == {}
        events = await _events(store)
        assert "feature_run_continued" not in events
        assert "feature_run_abandoned" not in events
    finally:
        await database.drop_schema()
        await database.dispose()


# --------------------------------------------------------------------------------------
# The race with the abandoned-run sweep
# --------------------------------------------------------------------------------------


async def test_the_ordering_that_keeps_the_sweep_behind_a_re_claim() -> None:
    """The argument rests on two unrelated constants, so it is asserted rather than reasoned.

    A crashed worker's claim lapses after the queue lease and is re-claimed by the next
    worker; the sweep may not look at the feature until the grace period. 2700 against 900
    means a lapsed claim can be picked up twice over before the sweep is entitled to act.
    """
    settings = load_settings()
    assert settings.feature_run_stale_after_seconds > 2 * settings.feature_queue_lease_seconds, (
        "a crashed claim must be re-claimable before the abandoned-run sweep may act on it"
    )


async def test_the_sweep_leaves_a_feature_a_worker_has_claimed_alone(tmp_path: Path) -> None:
    """Two mechanisms now want the same feature, and only one of them may end it."""
    executor = LoopProbeExecutor()
    database, store = await _store(
        tmp_path, "race.db", FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))
    )
    try:
        await _install(store, await _mid_fan_out(await _accept(store)))
        await _backdate(database, datetime.now(UTC) - timedelta(hours=2))
        await _hold_queue_entry(database)

        assert await store.reconcile_abandoned_runs(stale_after_seconds=60) == []

        state = (await store.get_record(FEATURE)).state
        assert state.status is FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
        assert state.failure_summary is None
        assert "feature_run_abandoned" not in await _events(store)
    finally:
        await database.drop_schema()
        await database.dispose()


async def test_a_tombstoned_feature_is_not_continued_by_a_claim_that_arrives_after(
    tmp_path: Path,
) -> None:
    """The other order, and the other half of the fence.

    If the sweep gets there first the feature is no longer in an actively-running status, so
    the claim declines rather than reopening it. Between the two, exactly one terminal state
    is ever written.
    """
    executor = LoopProbeExecutor()
    database, store = await _store(
        tmp_path, "tombstone.db", FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))
    )
    try:
        await _install(store, await _mid_fan_out(await _accept(store)))
        await _settle_queue_entry(database)
        await _backdate(database, datetime.now(UTC) - timedelta(hours=2))
        assert await store.reconcile_abandoned_runs(stale_after_seconds=60) == [FEATURE]
        decided = (await store.get_record(FEATURE)).state.model_dump(mode="json")

        await _dead_claim(database)
        assert await _settled_by_the_queue(store, database)

        assert executor.engineer_calls == {}
        after = (await store.get_record(FEATURE)).state.model_dump(mode="json")
        assert after == decided, "a late claim rewrote a feature the sweep had already decided"
        assert (await _events(store)).count("feature_run_abandoned") == 1
    finally:
        await database.drop_schema()
        await database.dispose()


# --------------------------------------------------------------------------------------
# Per-child resume eligibility
# --------------------------------------------------------------------------------------


def _terminal_child(state: FeatureWorkflowSnapshot, repository_id: str) -> FeatureWorkflowSnapshot:
    """Give one repository the persisted terminal retry decision a stopped one carries."""
    updated = state.model_copy(deep=True)
    child = updated.child_workflows[repository_id]
    updated.child_workflows[repository_id] = child.model_copy(
        update={
            "status": ChildWorkflowStatus.FAILED,
            "retry_refusal_reason": "this workstream is not converging",
            "retry_count": updated.max_child_review_cycles - 1,
            "blocking_issues": [*child.blocking_issues, "the scoped requirement is unmet"],
        }
    )
    return updated


async def test_resume_runs_the_eligible_repositories_and_leaves_the_terminal_one_alone(
    tmp_path: Path,
) -> None:
    """One terminal repository used to make the whole feature unresumable.

    `_has_exhausted_child_retry_budget` asked the question feature-wide, so a feature where
    one repository had stopped could never resume the others: resume returned the state
    unchanged, with no event, and the queue recorded success.
    """
    executor = LoopProbeExecutor()
    database, store = await _store(
        tmp_path, "eligible.db", FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))
    )
    try:
        state = _terminal_child(await _both_running(await _accept(store)), "frontend")
        before = state.child_workflows["frontend"].model_copy(deep=True)
        assert resume_eligible_repository_ids(state) == {"backend"}
        await _install(store, state)
        await _dead_claim(database)

        assert await _settled_by_the_queue(store, database)

        assert executor.attempts("backend") == 1, "the eligible repository was not run"
        assert executor.attempts("frontend") == 0, "the terminal repository was run again"
        frontend = (await store.get_record(FEATURE)).state.child_workflows["frontend"]
        # Its status, its refusal and every counter behind that decision are untouched.
        assert frontend.status is before.status
        assert frontend.retry_refusal_reason == before.retry_refusal_reason
        assert frontend.retry_count == before.retry_count
        assert frontend.implementation_retry_count == before.implementation_retry_count
        assert frontend.validation_retry_count == before.validation_retry_count
        assert frontend.granted_extra_attempts == before.granted_extra_attempts
    finally:
        await database.drop_schema()
        await database.dispose()


async def test_a_feature_with_no_eligible_workstream_says_so_instead_of_closing_silently(
    tmp_path: Path,
) -> None:
    """A silent no-op the queue then marks `succeeded` is how a feature disappears.

    The feature genuinely must not move -- every repository has already reached a terminal
    retry decision and re-running them would repeat coding and validation to reach the same
    verdicts. What changes is that the timeline now says it was asked and declined.
    """
    executor = LoopProbeExecutor()
    database, store = await _store(
        tmp_path, "ineligible.db", FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))
    )
    try:
        both = await _both_running(await _accept(store))
        state = _terminal_child(_terminal_child(both, "frontend"), "backend")
        assert resume_eligible_repository_ids(state) == set()
        await _install(store, state)
        await _dead_claim(database)

        assert await _settled_by_the_queue(store, database)

        assert executor.engineer_calls == {}
        after = (await store.get_record(FEATURE)).state
        assert after.status is state.status
        assert {key: child.retry_count for key, child in after.child_workflows.items()} == {
            key: child.retry_count for key, child in state.child_workflows.items()
        }
        assert "feature_resume_found_no_eligible_workstreams" in await _events(store)
    finally:
        await database.drop_schema()
        await database.dispose()


# --------------------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------------------


async def test_a_feature_that_completes_normally_is_unaffected(tmp_path: Path) -> None:
    """One claim, one run, no continuation, and the statuses it always had."""
    executor = LoopProbeExecutor()
    database, store = await _store(
        tmp_path, "normal.db", FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))
    )
    try:
        await _accept(store)
        assert (await _queue_row(database)).status == QUEUED

        assert await _settled_by_the_queue(store, database)

        state = (await store.get_record(FEATURE)).state
        assert state.status is FeatureWorkflowStatus.COMPLETED
        assert executor.engineer_calls == {"backend": 1, "frontend": 1}
        events = await _events(store)
        assert "feature_run_continued" not in events
        assert "feature_run_abandoned" not in events
        row = await _queue_row(database)
        assert (row.status, row.attempt) == (SUCCEEDED, 1)
    finally:
        await database.drop_schema()
        await database.dispose()
