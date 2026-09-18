"""Backstops on how long work may run, and on accepting work that cannot fit.

Three limits, none of which existed. A live feature reached 571 minutes inside a single
coroutine, renewing its lease the whole way, so every sweep the platform had -- all of which
ask whether an executor has stopped *writing* -- reported it as healthy. Workspace capacity
was checked during execution, so three of fifty recent failures were features that were
accepted, planned, cloned, and then killed by a volume that was already full when they were
submitted.

Each of these is a backstop. The tests below therefore assert as hard on what they do *not*
do -- fire on a resting feature, touch a sibling workstream, interrupt an external effect --
as on what they do. A breaker that fires during normal operation is not a backstop; it is a
scheduling policy nobody chose.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import update

from adapters.llm_adapter import LLMAdapterError
from api.control_plane import RequestScopedCredentials, WorkflowConflictError
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.external_operations import ExternalOperationStatus, ExternalOperationType
from state.failure_diagnosis import FailureStage, FeatureFailureClassification
from state.feature_models import ChildWorkflowReference, ensure_feature_failure_summary
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from storage.feature_store import SqlAlchemyFeatureControlPlane
from storage.models import FeatureExecutionQueueModel, FeatureWorkflowModel
from tests.support import drain_feature_queue
from tests.test_feature_api import feature_payload
from tools.retry_strategy import TerminalCause
from workflows.feature_workflow import ChildExecution, FeatureWorkflowOrchestrator

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)

_SIX_HOURS = 21_600.0


async def _store(tmp_path: Path, name: str, **kwargs: Any) -> tuple[Database, Any]:
    """Build a real database and control plane, as the sweep runs against in production."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / name}")
    await database.create_schema()
    return database, SqlAlchemyFeatureControlPlane(
        database, mock_runner=FeatureWorkflowOrchestrator(), **kwargs
    )


async def _long_running_feature(
    store: Any,
    database: Database,
    *,
    status: FeatureWorkflowStatus = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
    claimed_minutes_ago: int = 600,
    still_writing: bool = True,
) -> str:
    """Leave one feature claimed long ago and, by default, still checkpointing normally.

    `still_writing` is what separates this from an abandoned run: `updated_at` is current,
    so the staleness sweep sees a feature making progress and correctly leaves it alone.
    That is precisely the case no limit covered.
    """
    result = await store.start(
        StartFeatureRequest.model_validate(feature_payload()),
        idempotency_key=f"runtime-{claimed_minutes_ago}-{status.value}",
        credentials=CREDENTIALS,
        owner_id="platform-admin",
    )
    feature_id = str(result.record.state.feature_id)
    await drain_feature_queue(store)

    state = (await store.get_record(feature_id)).state.model_copy(deep=True)
    state.status = status
    state.current_agent = "child_workflows"
    state.child_workflows = {
        "console-2.0": ChildWorkflowReference(
            child_workflow_id=f"{feature_id}:console",
            repository_id="console-2.0",
            workstream_id="ws-console",
            status=ChildWorkflowStatus.RUNNING,
            branch_name="feature/long",
            workspace_path=f"/workspaces/{feature_id}/console-2.0",
            retry_count=2,
        )
    }
    await store._replace_state(feature_id, state, "test_setup")  # noqa: SLF001

    claimed = datetime.now(UTC) - timedelta(minutes=claimed_minutes_ago)
    async with database.session() as session:
        model = await session.get(FeatureWorkflowModel, feature_id)
        assert model is not None
        state_json = dict(model.state_json)
        state_json["status"] = status.value
        await session.execute(
            update(FeatureWorkflowModel)
            .where(FeatureWorkflowModel.feature_id == feature_id)
            .values(
                status=status.value,
                state_json=state_json,
                updated_at=datetime.now(UTC) if still_writing else claimed,
            )
        )
        # The clock this breaker reads: when a worker claimed the feature, not when it was
        # submitted. A submission that queued for an hour is not charged for the wait.
        await session.execute(
            update(FeatureExecutionQueueModel)
            .where(FeatureExecutionQueueModel.feature_id == feature_id)
            .values(started_at=claimed, status="running")
        )
        await session.commit()
    return feature_id


@pytest.mark.asyncio
async def test_a_feature_past_its_runtime_ceiling_stops_and_says_which_ceiling(
    tmp_path: Path,
) -> None:
    """Ten hours of healthy progress is still ten hours, and nothing used to stop it."""
    database, store = await _store(tmp_path, "runtime.db")
    try:
        feature_id = await _long_running_feature(store, database)

        # The sweep that exists asks whether the executor stopped writing. This one has not.
        assert await store.reconcile_abandoned_runs(stale_after_seconds=2_700) == []

        stopped = await store.stop_overrunning_runs(runtime_limit_seconds=_SIX_HOURS)

        assert stopped == [feature_id]
        record = await store.get_record(feature_id)
        assert record.state.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        summary = record.state.failure_summary
        assert summary is not None
        assert summary.root_classification == (
            FeatureFailureClassification.FEATURE_RUNTIME_LIMIT_REACHED.value
        )
        assert summary.stage == FailureStage.FEATURE_RUNTIME.value
        assert summary.diagnostics
        # It says how long, against what ceiling, and that the run may have been fine.
        joined = " ".join(summary.diagnostics)
        assert "360-minute ceiling" in joined
        assert "console-2.0" in joined
        # A runtime ceiling is not the platform being broken and not the repository being
        # wrong, so it must not be retried automatically on either reading.
        assert summary.retryable is False
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_feature_waiting_for_a_person_is_never_stopped_by_the_clock(
    tmp_path: Path,
) -> None:
    """`waiting_for_human` is a resting state and can legitimately last days.

    The most important thing this breaker does not do. A clock that counted time spent
    waiting for an answer would kill exactly the features whose owners are still deciding.
    """
    database, store = await _store(tmp_path, "resting.db")
    try:
        feature_id = await _long_running_feature(
            store,
            database,
            status=FeatureWorkflowStatus.WAITING_FOR_HUMAN,
            claimed_minutes_ago=4_320,
        )

        assert await store.stop_overrunning_runs(runtime_limit_seconds=_SIX_HOURS) == []

        record = await store.get_record(feature_id)
        assert record.state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
        assert record.state.failure_summary is None
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_feature_inside_its_ceiling_is_left_alone(tmp_path: Path) -> None:
    """The backstop must be silent for every run that is merely long."""
    database, store = await _store(tmp_path, "inside.db")
    try:
        await _long_running_feature(store, database, claimed_minutes_ago=90)

        assert await store.stop_overrunning_runs(runtime_limit_seconds=_SIX_HOURS) == []
        # And a ceiling of zero is off entirely, whatever the elapsed time.
        assert await store.stop_overrunning_runs(runtime_limit_seconds=0) == []
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_the_runtime_ceiling_does_not_interrupt_an_external_operation(
    tmp_path: Path,
) -> None:
    """A breaker stops scheduling. It does not reach into a push that is already running.

    Interrupting one would produce exactly the unconfirmed remote effect the rest of this
    platform is built to avoid: a branch that may or may not exist, with nothing able to say
    which. The operation is left running and settles through the journal as it always does.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'effects.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    store = SqlAlchemyFeatureControlPlane(
        database, mock_runner=FeatureWorkflowOrchestrator(), operation_journal=journal
    )
    try:
        feature_id = await _long_running_feature(store, database)
        operation = await journal.create_operation(
            workflow_id=feature_id,
            feature_id=feature_id,
            child_workflow_id=f"{feature_id}:console",
            repository_id="console-2.0",
            operation_type=ExternalOperationType.PUSH_BRANCH,
            idempotency_key=f"{feature_id}:push",
            input_fingerprint=f"{feature_id}:push",
        )
        claimed = await journal.claim_operation(operation.operation_id)
        assert claimed.status is ExternalOperationStatus.RUNNING

        assert await store.stop_overrunning_runs(runtime_limit_seconds=_SIX_HOURS) == [feature_id]

        after = await journal.get(operation.operation_id)
        # Untouched: not cancelled, not failed, not stamped for manual review.
        assert after.status is ExternalOperationStatus.RUNNING
        assert after.error_code is None
    finally:
        await database.dispose()


# ---------------------------------------------------------------------------------------
# The repository ceiling
# ---------------------------------------------------------------------------------------


class _SlowWorkstreamExecutor:
    """Reject every attempt, and make one named repository take a very long time.

    Time is advanced by a fake monotonic clock rather than by sleeping: the ceiling is a
    wall-clock policy, and a test that actually waited ninety minutes to prove it is not a
    test anybody runs.
    """

    def __init__(self, slow_repository_id: str, *, seconds_per_attempt: float) -> None:
        """Charge `seconds_per_attempt` to the named repository and nothing to the rest."""
        from workflows.feature_workflow import MockChildWorkstreamExecutor

        self._inner = MockChildWorkstreamExecutor()
        self._slow = slow_repository_id
        self._seconds = seconds_per_attempt
        self.elapsed = 0.0
        self.attempts: dict[str, int] = {}

    def monotonic(self) -> float:
        """Stand in for `time.monotonic`, advancing only when the slow repository runs."""
        return self.elapsed

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Reject the attempt, charging time only to the repository under test."""
        repository_id = kwargs["repository"].repository_id
        self.attempts[repository_id] = self.attempts.get(repository_id, 0) + 1
        if repository_id == self._slow:
            self.elapsed += self._seconds
        execution = await self._inner.run(**kwargs)
        index = self.attempts[repository_id]
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    "pull_request_readiness": False,
                    "current_revision": f"{repository_id}-revision-{index}",
                    "production_files_changed": [f"src/module_{index}.py"],
                    "production_diff_fingerprint": f"{repository_id}-fingerprint-{index}",
                    "blocking_issues": [f"The reviewer asked for change {index}."],
                }
            ),
            code_completion=execution.code_completion,
            review=execution.review,
        )


@pytest.mark.asyncio
async def test_a_workstream_past_its_ceiling_stops_and_its_siblings_do_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One repository running too long is one repository's problem.

    The ceiling is checked between attempts and never before the first, so it can only ever
    withhold a *retry*: a backstop that could end a workstream before it had run once would
    be a scheduling policy rather than a limit.
    """
    del tmp_path
    executor = _SlowWorkstreamExecutor("backend", seconds_per_attempt=3_600.0)
    monkeypatch.setattr("workflows.feature_workflow.time.monotonic", executor.monotonic)
    state = _initial_feature_state(
        "feature-repository-ceiling", StartFeatureRequest.model_validate(feature_payload())
    )
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor),
        # Ninety minutes: the first attempt costs sixty and is never refused, the second
        # takes the workstream past the ceiling.
        repository_runtime_limit_seconds=5_400.0,
    )

    result = await orchestrator.start(state, credentials=CREDENTIALS)

    backend = result.child_workflows["backend"]
    assert backend.status is ChildWorkflowStatus.FAILED
    assert backend.failure_classification == (
        FeatureFailureClassification.REPOSITORY_RUNTIME_LIMIT_REACHED.value
    )
    stop = " ".join(backend.blocking_issues)
    assert "90-minute ceiling" in stop
    assert "repository_runtime_limit_seconds" in stop
    # Nothing was interrupted: the attempt that was running finished and reported.
    assert "Nothing was interrupted" in stop
    # The sibling is charged no time and keeps every attempt its own budget allows.
    assert executor.attempts["frontend"] > executor.attempts["backend"]

    # And what a person actually reads. The funnel `_replace_state` applies to every
    # snapshot it writes is what turns the child's classification into the feature's, so a
    # stop that only made sense at the workstream level would surface as `platform_defect`.
    diagnosed = ensure_feature_failure_summary(result)
    summary = diagnosed.failure_summary
    assert summary is not None
    assert summary.root_classification == (
        FeatureFailureClassification.REPOSITORY_RUNTIME_LIMIT_REACHED.value
    )
    assert summary.stage == FailureStage.CHILD_WORKFLOWS.value
    assert summary.repository_id == "backend"
    assert summary.diagnostics
    assert summary.retryable is False
    # The triage the stop recorded reaches the summary rather than being re-derived.
    assert summary.terminal_cause == TerminalCause.BUDGET_EXHAUSTED.value
    assert summary.next_action


@pytest.mark.asyncio
async def test_a_workstream_inside_its_ceiling_keeps_its_whole_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ceiling must not shorten any existing budget, which is the regression risk."""
    unbounded = _SlowWorkstreamExecutor("backend", seconds_per_attempt=1.0)
    bounded = _SlowWorkstreamExecutor("backend", seconds_per_attempt=1.0)
    monkeypatch.setattr("workflows.feature_workflow.time.monotonic", unbounded.monotonic)
    await FeatureWorkflowOrchestrator(child_executor=cast(Any, unbounded)).start(
        _initial_feature_state(
            "feature-unbounded", StartFeatureRequest.model_validate(feature_payload())
        ),
        credentials=CREDENTIALS,
    )
    monkeypatch.setattr("workflows.feature_workflow.time.monotonic", bounded.monotonic)
    await FeatureWorkflowOrchestrator(
        child_executor=cast(Any, bounded), repository_runtime_limit_seconds=5_400.0
    ).start(
        _initial_feature_state(
            "feature-bounded", StartFeatureRequest.model_validate(feature_payload())
        ),
        credentials=CREDENTIALS,
    )

    assert bounded.attempts == unbounded.attempts


# ---------------------------------------------------------------------------------------
# Workspace capacity, asked when the work is accepted
# ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_live_submission_is_refused_when_the_volume_cannot_house_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused at acceptance, with no feature row created.

    This matches the acceptance contract the concurrency quota beside it already follows:
    the refusal is raised before `_create_or_replay`, so there is no feature to explain and
    the caller is told in the response rather than in a record they have to go and find.
    """
    database, store = await _store(
        tmp_path,
        "capacity.db",
        workspace_root=tmp_path / "workspaces",
        workspace_minimum_free_bytes=2_147_483_648,
    )
    (tmp_path / "workspaces").mkdir()
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _path: shutil._ntuple_diskusage(100, 99, 1_048_576)
    )
    try:
        with pytest.raises(WorkflowConflictError) as refusal:
            await store.start(
                StartFeatureRequest.model_validate({**feature_payload(), "execution_mode": "live"}),
                idempotency_key="capacity-1",
                credentials=CREDENTIALS,
                owner_id="platform-admin",
            )

        message = str(refusal.value)
        # The preflight's own wording, so the refusal and the mid-run failure read the same.
        assert "Workspace capacity preflight failed" in message
        assert "1048576 bytes free" in message
        assert "expand the workspace volume" in message
        # Nothing was accepted, so nothing has to be explained afterwards.
        async with database.session() as session:
            assert await session.get(FeatureWorkflowModel, "feature-login") is None
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_live_submission_is_accepted_when_the_volume_has_room(tmp_path: Path) -> None:
    """The check must be silent whenever the volume is fine, which is almost always."""
    database, store = await _store(
        tmp_path,
        "roomy.db",
        workspace_root=tmp_path / "workspaces",
        workspace_minimum_free_bytes=1,
    )
    (tmp_path / "workspaces").mkdir()
    try:
        result = await store.start(
            StartFeatureRequest.model_validate({**feature_payload(), "execution_mode": "live"}),
            idempotency_key="capacity-2",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        assert result.created
    finally:
        await database.dispose()


class _ProviderStallThenWorkExecutor(_SlowWorkstreamExecutor):
    """One attempt stalls inside a provider call that then fails terminally.

    The stall advances the same fake clock the real attempts advance, so wall time grows
    while the classified fault contributes nothing to the charged clock.
    """

    def __init__(
        self, slow_repository_id: str, *, seconds_per_attempt: float, stall_seconds: float
    ) -> None:
        super().__init__(slow_repository_id, seconds_per_attempt=seconds_per_attempt)
        self._stall_seconds = stall_seconds
        self.stalled = False

    async def run(self, **kwargs: Any) -> ChildExecution:
        repository_id = kwargs["repository"].repository_id
        if (
            repository_id == self._slow
            and self.attempts.get(repository_id) == 1
            and not self.stalled
        ):
            self.stalled = True
            self.elapsed += self._stall_seconds
            msg = "the model provider stalled and then failed terminally"
            raise LLMAdapterError(msg)
        return await super().run(**kwargs)


@pytest.mark.asyncio
async def test_time_inside_a_classified_provider_fault_is_not_charged_to_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wall and charged clocks diverge on a fault, and the ceiling judges the charged one.

    AB-Feature-171's backend spent 54 of its 90 minutes inside one terminally failing
    `run_coding_executor` call and was then ended by the ceiling: the budget paid for the
    weather. Here the stall alone pushes wall time past the ceiling; under the old
    accounting the workstream would stop at the next check, and under the charged clock it
    demonstrably runs its next real attempt first.
    """
    executor = _ProviderStallThenWorkExecutor(
        "backend", seconds_per_attempt=3_600.0, stall_seconds=3_600.0
    )
    monkeypatch.setattr("workflows.feature_workflow.time.monotonic", executor.monotonic)
    monkeypatch.setattr("workflows.feature_workflow._INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)
    state = _initial_feature_state(
        "feature-provider-fault-clock", StartFeatureRequest.model_validate(feature_payload())
    )
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor),
        repository_runtime_limit_seconds=5_400.0,
    )

    result = await orchestrator.start(state, credentials=CREDENTIALS)

    backend = result.child_workflows["backend"]
    # The stall happened (wall passed the ceiling during it) and the loop still ran the
    # next real attempt: the charged clock, not the wall clock, is what the ceiling reads.
    assert executor.stalled
    assert executor.attempts["backend"] == 2, (
        "the attempt after the fault must run: its charged time was inside the ceiling"
    )
    assert backend.failure_classification == (
        FeatureFailureClassification.REPOSITORY_RUNTIME_LIMIT_REACHED.value
    )
    # Both clocks are recorded, and they disagree by exactly the fault's stall time.
    assert backend.runtime_wall_seconds is not None
    assert backend.runtime_charged_seconds is not None
    assert backend.runtime_wall_seconds > backend.runtime_charged_seconds
    assert backend.runtime_wall_seconds - backend.runtime_charged_seconds == 3_600.0
    stop = " ".join(backend.blocking_issues)
    assert "charged" in stop
    assert "not charged" in stop
