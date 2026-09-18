"""Features whose executor stopped without recording an outcome.

Everything else durable here is swept. External operations settle, action leases lapse and
are reconciled, queue claims expire and are re-delivered. A feature's own status was not,
and AB-Feature-108 is what that costs: its action was reconciled `interrupted_before_completion`
at 15:36 and the process was gone, but the feature claimed `running_child_workflows` -- with a
`running` workstream -- for over a day afterwards. Nothing was watching, and the only remedy
was to retire it by hand.

These tests are written against the persisted effect: what a later reader sees, and what the
sweep declines to touch.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import update

from api.control_plane import RequestScopedCredentials
from api.feature_schemas import StartFeatureRequest
from services.action_context import (
    bind_feature_action,
    clear_revoked_feature_actions,
    reset_feature_action,
    revoke_feature_action,
)
from services.feature_actions import ActionLeaseLostError, FeatureActionService
from services.recovery_service import RecoveryService
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.feature_models import ChildWorkflowReference
from storage.action_store import InMemoryFeatureActionStore
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from storage.feature_store import FeatureRunRevokedError, SqlAlchemyFeatureControlPlane
from storage.models import FeatureActionModel, FeatureWorkflowModel
from tests.support import drain_feature_queue
from tests.test_feature_api import feature_payload
from workflows.feature_workflow import FeatureWorkflowOrchestrator

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)


async def _abandoned_feature(
    store: SqlAlchemyFeatureControlPlane,
    database: Database,
    *,
    status: FeatureWorkflowStatus = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
    stale_minutes: int = 120,
    child_status: ChildWorkflowStatus | None = ChildWorkflowStatus.RUNNING,
) -> str:
    """Leave one feature in the state a dead executor leaves behind.

    Written straight to the row on purpose. This is the shape AB-Feature-108 was actually
    found in, and reproducing it through the orchestrator would require killing a process
    mid-attempt -- which is the one thing a test cannot do reliably.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    result = await store.start(
        request, idempotency_key="abandoned", credentials=CREDENTIALS, owner_id="platform-admin"
    )
    feature_id = result.record.state.feature_id
    await drain_feature_queue(store)

    state = (await store.get_record(feature_id)).state.model_copy(deep=True)
    state.status = status
    state.current_agent = "child_workflows"
    if child_status is not None:
        state.child_workflows = {
            "console-2.0": ChildWorkflowReference(
                child_workflow_id=f"{feature_id}:console",
                repository_id="console-2.0",
                workstream_id="ws-console",
                status=child_status,
                branch_name="feature/bulk-delete",
                workspace_path=f"/workspaces/{feature_id}/console-2.0",
                retry_count=9,
                validation_retry_count=6,
                failure_classification="validation_capacity_failure",
            )
        }
    stale = datetime.now(UTC) - timedelta(minutes=stale_minutes)
    state.updated_at = stale
    await store._replace_state(feature_id, state, "test_setup")  # noqa: SLF001
    # `_replace_state` stamps the row from the snapshot, but a merge may refresh it. Backdate
    # both copies so the sweep's own staleness check sees what a real abandonment looks like.
    async with database.session() as session:
        model = await session.get(FeatureWorkflowModel, feature_id)
        assert model is not None
        state_json = dict(model.state_json)
        state_json["status"] = status.value
        state_json["updated_at"] = stale.isoformat()
        await session.execute(
            update(FeatureWorkflowModel)
            .where(FeatureWorkflowModel.feature_id == feature_id)
            .values(status=status.value, updated_at=stale, state_json=state_json)
        )
        await session.commit()
    return feature_id


async def _store(tmp_path: Path, name: str) -> tuple[Database, SqlAlchemyFeatureControlPlane]:
    """Build a real database and control plane, as the sweep runs against in production."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / name}")
    await database.create_schema()
    return database, SqlAlchemyFeatureControlPlane(
        database, mock_runner=FeatureWorkflowOrchestrator()
    )


@pytest.mark.asyncio
async def test_a_feature_whose_executor_died_stops_claiming_to_be_running(
    tmp_path: Path,
) -> None:
    """The defect itself: -108 held `running_child_workflows` for a day with nothing running."""
    database, store = await _store(tmp_path, "abandoned.db")
    try:
        feature_id = await _abandoned_feature(store, database)

        reconciled = await store.reconcile_abandoned_runs(stale_after_seconds=60)

        assert reconciled == [feature_id]
        record = await store.get_record(feature_id)
        assert record.state.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        # The workstream stops claiming to run too. A feature that has reached a person
        # while one of its repositories still says `running` reads as work in progress.
        # Rehydrated children are keyed by repository, not by workstream.
        children = record.state.child_workflows
        assert children["console-2.0"].status is ChildWorkflowStatus.FAILED
        assert any(
            "stopped without recording an outcome" in issue
            for issue in children["console-2.0"].blocking_issues
        )
        # Only the unfinished one. A workstream that reached a verdict keeps it: -108's
        # second repository was `approved`, and losing that would discard real work.
        assert not any(
            child.status is ChildWorkflowStatus.FAILED
            for key, child in children.items()
            if key != "console-2.0"
        )
        # And it says why, rather than leaving somebody to infer it from artifacts. With a
        # stranded workstream the child's own diagnostics are the more specific answer and
        # are the ones the summary carries.
        summary = record.state.failure_summary
        assert summary is not None
        assert any("stopped without recording an outcome" in item for item in summary.diagnostics)
        # The run-level context reaches the timeline, naming what it was doing and what it
        # left behind -- which is what -108 offered nobody.
        timeline = await store.timeline(feature_id)
        abandoned = [details for *_, event, details in timeline if event == "feature_run_abandoned"]
        assert len(abandoned) == 1
        assert abandoned[0]["previous_status"] == "running_child_workflows"
        assert abandoned[0]["stranded_workstreams"] == ["console-2.0"]
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_run_abandoned_before_it_had_workstreams_still_explains_itself(
    tmp_path: Path,
) -> None:
    """A process that dies during planning leaves no child to speak for it.

    So the run-level diagnosis has to carry the whole answer: what it was doing, whether
    anything picked it back up, and where the effects it did record are.

    "Start a fresh feature rather than resuming this one" is what this asserted until task
    28-, and it was correct while nothing continued a crashed run. It is advice this platform
    no longer means: a worker continues one from its checkpoint, and reaching here is either
    the end of that or a run nothing got to in time. The diagnostic now says which.
    """
    database, store = await _store(tmp_path, "planning.db")
    try:
        feature_id = await _abandoned_feature(
            store,
            database,
            status=FeatureWorkflowStatus.PLANNING,
            child_status=None,
        )

        assert await store.reconcile_abandoned_runs(stale_after_seconds=60) == [feature_id]
        summary = (await store.get_record(feature_id)).state.failure_summary
        assert summary is not None
        assert any(
            "stopped without writing a terminal status" in item for item in summary.diagnostics
        )
        assert any("planning" in item for item in summary.diagnostics)
        assert not any("Start a fresh feature" in item for item in summary.diagnostics)
        assert any("nothing was retried" in item for item in summary.diagnostics)
        assert any("in the operation journal" in item for item in summary.diagnostics)
        timeline = await store.timeline(feature_id)
        abandoned = [details for *_, event, details in timeline if event == "feature_run_abandoned"]
        assert abandoned == [
            {"previous_status": "planning", "stranded_workstreams": [], "continuations": 0}
        ]
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_run_that_is_merely_slow_is_left_alone(tmp_path: Path) -> None:
    """A validation command is bounded at 900 seconds; the grace period must clear it.

    Ending a feature that is only between flushes would be a worse defect than the one
    being fixed, so the staleness threshold is the whole safety argument.
    """
    database, store = await _store(tmp_path, "slow.db")
    try:
        feature_id = await _abandoned_feature(store, database, stale_minutes=20)

        assert await store.reconcile_abandoned_runs(stale_after_seconds=2_700) == []
        record = await store.get_record(feature_id)
        assert record.state.status is FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_live_action_lease_protects_the_feature_it_owns(tmp_path: Path) -> None:
    """A long step that keeps renewing is alive by definition, however stale its state looks.

    This is the case the fix must not get wrong: -108's executor was killed *because* one
    renewal failed, and a sweep that ignored live leases would repeat that mistake from the
    other direction.
    """
    database, store = await _store(tmp_path, "leased.db")
    try:
        feature_id = await _abandoned_feature(store, database)
        async with database.session() as session:
            session.add(
                FeatureActionModel(
                    action_id="action-live",
                    feature_id=feature_id,
                    repository_id=None,
                    action_type="ANSWER_CLARIFICATION",
                    actor_id="someone",
                    origin="api",
                    payload={},
                    input_fingerprint="fingerprint",
                    idempotency_key="key",
                    status="executing",
                    attempt=1,
                    max_attempts=1,
                    lease_owner="live-worker",
                    lease_expires_at=datetime.now(UTC) + timedelta(minutes=3),
                    created_at=datetime.now(UTC),
                )
            )
            await session.commit()

        assert await store.reconcile_abandoned_runs(stale_after_seconds=60) == []
        record = await store.get_record(feature_id)
        assert record.state.status is FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_feature_waiting_for_a_person_is_never_swept(tmp_path: Path) -> None:
    """`waiting_for_human` is a resting state. Nothing is running and nothing is wrong."""
    database, store = await _store(tmp_path, "waiting.db")
    try:
        feature_id = await _abandoned_feature(
            store,
            database,
            status=FeatureWorkflowStatus.WAITING_FOR_HUMAN,
            child_status=None,
            stale_minutes=10_000,
        )

        assert await store.reconcile_abandoned_runs(stale_after_seconds=60) == []
        record = await store.get_record(feature_id)
        assert record.state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_finished_feature_is_not_reopened(tmp_path: Path) -> None:
    """The sweep only ever narrows: a completed feature must stay completed."""
    database, store = await _store(tmp_path, "completed.db")
    try:
        feature_id = await _abandoned_feature(
            store,
            database,
            status=FeatureWorkflowStatus.COMPLETED,
            child_status=ChildWorkflowStatus.APPROVED,
            stale_minutes=10_000,
        )

        assert await store.reconcile_abandoned_runs(stale_after_seconds=60) == []
        assert (await store.get_record(feature_id)).state.status is FeatureWorkflowStatus.COMPLETED
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_run_that_lost_its_claim_cannot_write_the_feature(tmp_path: Path) -> None:
    """The other half of -108: a revoked executor is refused, not merely asked to stop."""
    database, store = await _store(tmp_path, "revoked.db")
    try:
        feature_id = await _abandoned_feature(store, database, stale_minutes=1)
        before = (await store.get_record(feature_id)).state.model_copy(deep=True)

        token = bind_feature_action("action-revoked")
        try:
            revoke_feature_action("action-revoked")
            doomed = before.model_copy(deep=True)
            doomed.status = FeatureWorkflowStatus.COMPLETED
            with pytest.raises(FeatureRunRevokedError):
                await store._replace_state(feature_id, doomed, "child_workflow_failed")  # noqa: SLF001
        finally:
            reset_feature_action(token)
            clear_revoked_feature_actions()

        # Nothing landed. This is the assertion that matters: -108's ninth attempt did land,
        # seventeen minutes after recovery had decided the run was over.
        after = (await store.get_record(feature_id)).state
        assert after.status is before.status
        assert after.updated_at == before.updated_at
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_work_that_outlives_its_cancellation_is_fenced_out(tmp_path: Path) -> None:
    """A coroutine inside a fifteen-minute subprocess does not stop when cancel() is called.

    That window is what -108 wrote its ninth attempt through, so revocation has to happen
    when the lease is lost rather than when the work finally unwinds. The shield here stands
    in for the uncancellable subprocess.
    """
    database, store = await _store(tmp_path, "shielded.db")
    try:
        feature_id = await _abandoned_feature(store, database, stale_minutes=1)
        before = (await store.get_record(feature_id)).state.model_copy(deep=True)
        refused: list[str] = []

        action_store = InMemoryFeatureActionStore(lease_seconds=0.3)
        actions = FeatureActionService(store=action_store, build_revision="test-build")
        action, _ = await actions.submit(
            feature_id=feature_id,
            action_type="RETRY_WORKSTREAM",
            actor_id="someone",
            repository_id="console-2.0",
            payload={"additional_attempts": 1, "reason": "read the blocking issues"},
            context_version="v1",
        )

        async def keeps_going_after_being_cancelled() -> str:
            async def survive_and_write() -> None:
                # Outlasts the lease, the way a long validation command does.
                await asyncio.sleep(0.9)
                doomed = before.model_copy(deep=True)
                doomed.status = FeatureWorkflowStatus.COMPLETED
                try:
                    await store._replace_state(feature_id, doomed, "child_workflow_failed")  # noqa: SLF001
                except FeatureRunRevokedError:
                    refused.append("refused")

            await asyncio.shield(asyncio.create_task(survive_and_write()))
            return "done"

        async def steal() -> None:
            await asyncio.sleep(0.05)
            action_store._actions[action.action_id] = action_store._actions[  # noqa: SLF001
                action.action_id
            ].model_copy(update={"lease_owner": "somebody-else"})

        thief = asyncio.create_task(steal())
        with pytest.raises(ActionLeaseLostError):
            await actions.execute(action, keeps_going_after_being_cancelled)
        await thief
        # Let the shielded write attempt actually happen after the executor gave up.
        await asyncio.sleep(1.0)

        assert refused == ["refused"], "the late write must be refused, not silently applied"
        after = (await store.get_record(feature_id)).state
        assert after.status is before.status
    finally:
        clear_revoked_feature_actions()
        await database.drop_schema()
        await database.dispose()


class _RecordingFeatureRecovery:
    """Capture the grace period the recovery service passes through."""

    def __init__(self) -> None:
        self.calls: list[float] = []
        self.runtime_limit_calls: list[float] = []

    async def reconcile_abandoned_runs(
        self, *, stale_after_seconds: float, limit: int = 50
    ) -> list[str]:
        """Record the sweep and report one decided feature."""
        del limit
        self.calls.append(stale_after_seconds)
        return ["feature-abandoned"]

    async def stop_overrunning_runs(
        self, *, runtime_limit_seconds: float, limit: int = 50
    ) -> list[str]:
        """Record the runtime-ceiling sweep and report nothing stopped."""
        del limit
        self.runtime_limit_calls.append(runtime_limit_seconds)
        return []


class _FailingFeatureRecovery:
    """Fail the way a database blip would, to prove the other sweeps still run."""

    async def reconcile_abandoned_runs(
        self, *, stale_after_seconds: float, limit: int = 50
    ) -> list[str]:
        """Raise instead of answering."""
        del stale_after_seconds, limit
        msg = "connection reset"
        raise RuntimeError(msg)

    async def stop_overrunning_runs(
        self, *, runtime_limit_seconds: float, limit: int = 50
    ) -> list[str]:
        """Raise here too, so neither feature sweep can stop the ones before it."""
        del runtime_limit_seconds, limit
        msg = "connection reset"
        raise RuntimeError(msg)


class _RecordingActionRecovery:
    """Record that the action sweep ran, whatever the feature sweep does."""

    def __init__(self) -> None:
        self.swept = 0

    async def recover_abandoned_actions(self, *, limit: int = 100) -> Any:
        """Count one pass."""
        del limit
        self.swept += 1
        return None


@pytest.mark.asyncio
async def test_the_periodic_sweep_reaches_features_not_only_operations(
    tmp_path: Path,
) -> None:
    """The wiring is the fix. A sweep that exists but is never called changes nothing."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'sweep.db'}")
    await database.create_schema()
    try:
        features = _RecordingFeatureRecovery()
        service = RecoveryService(
            ExternalOperationJournal(database),
            workspace_root=tmp_path,
            feature_run_recovery=features,
            feature_run_stale_after_seconds=1_800.0,
        )

        await service.recover_incomplete_operations()

        assert features.calls == [1_800.0]
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_failing_feature_sweep_does_not_stop_action_recovery(tmp_path: Path) -> None:
    """Operation and action recovery guard against duplicated external effects.

    They are older and more important than this sweep, so a fault in the newest one must
    not take them down with it.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'isolated.db'}")
    await database.create_schema()
    try:
        actions = _RecordingActionRecovery()
        service = RecoveryService(
            ExternalOperationJournal(database),
            workspace_root=tmp_path,
            action_recovery=actions,
            feature_run_recovery=_FailingFeatureRecovery(),
        )

        summary = await service.recover_incomplete_operations()

        assert actions.swept == 1
        assert summary.scanned == 0
    finally:
        await database.drop_schema()
        await database.dispose()
