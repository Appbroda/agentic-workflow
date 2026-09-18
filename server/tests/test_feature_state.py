"""Durable parent and child state tests for multi-repository features."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest
from pydantic import ValidationError

import workflows.feature_workflow as feature_workflow_module
from adapters.llm_adapter import LLMAdapterError
from agents.planner.feature_planner import DeterministicFeaturePlanner
from agents.shared.contracts import AgentArtifactError
from api.control_plane import (
    FeatureOperationFailedError,
    RequestScopedCredentials,
    WorkflowConflictError,
)
from api.feature_control_plane import FeatureRecord
from api.feature_schemas import StartFeatureRequest
from api.schemas import ClarificationAnswer
from artifacts.schemas import FeatureCompletionArtifact
from state.enums import (
    CancellationLifecycleStatus,
    ChildWorkflowStatus,
    FeatureWorkflowStatus,
)
from state.external_operations import (
    ExternalOperationStatus,
    ExternalOperationType,
    WorkflowCheckpointBoundary,
)
from state.feature_models import FeatureWorkflowSnapshot
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from storage.feature_store import SqlAlchemyFeatureControlPlane
from storage.models import FeatureWorkflowModel
from tests.support import drain_feature_queue
from tests.test_feature_workflow import RecordingReconnaissance
from workflow_schema import LEGACY_UNVERIFIED_BUILD_REVISION, WORKFLOW_SCHEMA_VERSION
from workflows.feature_workflow import (
    ClarificationAnswerError,
    DeterministicFeatureProductManager,
    FeatureWorkflowOrchestrator,
)


class _AlwaysFailingRunner:
    """Base for mock runners that never produce a state and never resolve contracts.

    `advance_one_step` is what a queue claim calls now, so each subclass raises its own
    failure from there as well as from `start`. `begin_resume` is deliberately not a failure
    point: it settles what a resume request carries, reaches nothing, and every one of these
    doubles is about a runner that fails once it runs.
    """

    async def start(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise NotImplementedError

    async def advance_one_step(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise NotImplementedError

    async def begin_resume(
        self, state: FeatureWorkflowSnapshot, **_kwargs: object
    ) -> FeatureWorkflowSnapshot:
        """Accept the resume unchanged; these doubles fail in the step, not the request."""
        return state

    async def grant_and_run_one_retry(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise NotImplementedError

    async def answer_design_conflict(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise NotImplementedError

    async def resume(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise NotImplementedError

    async def retry_workstream(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise NotImplementedError

    async def publish_feature(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise NotImplementedError

    async def approve_contract_change(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise NotImplementedError

    async def reject_contract_change(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise NotImplementedError

    async def approve_repository_repair(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise NotImplementedError

    async def reject_repository_repair(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise NotImplementedError


class InvalidArtifactRunner(_AlwaysFailingRunner):
    """Simulate a planner response that remains invalid after its bounded repair.

    Names its stage, exactly as the real planner does. The control plane attributes a bare
    schema rejection from the model it rejected, and this error carries no model, so a double
    that stayed silent would be reported as unattributed -- which is the correct answer for
    an error that will not say who raised it, and the wrong one for this test.
    """

    async def start(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise AgentArtifactError("invalid model-generated execution plan", stage="feature_planner")

    async def advance_one_step(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise AgentArtifactError("invalid model-generated execution plan", stage="feature_planner")

    async def resume(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise AgentArtifactError("invalid model-generated execution plan", stage="feature_planner")


class FailingRuntimeRunner(_AlwaysFailingRunner):
    """Simulate an unexpected runtime failure after a feature has been accepted."""

    async def start(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("unexpected child executor failure sk-secret-in-message")

    async def advance_one_step(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("unexpected child executor failure sk-secret-in-message")

    async def resume(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("unexpected child executor failure sk-secret-in-message")


class TimingOutRuntimeRunner(_AlwaysFailingRunner):
    """Reproduce the adapter wrapper emitted by a live provider timeout."""

    @staticmethod
    def _failure() -> LLMAdapterError:
        diagnostic = "the model provider call failed (APITimeoutError)"
        return LLMAdapterError(
            diagnostic,
            diagnostics=(diagnostic,),
            failure_classification="APITimeoutError",
        )

    async def start(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise self._failure()

    async def advance_one_step(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise self._failure()

    async def resume(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise self._failure()


class ProductManagerThatTimesOut:
    """Fail before the first checkpoint a set number of times, then let recovery complete.

    The count is a parameter because a single fault no longer stops a feature: the stages
    before any repository runs retry a provider fault in process, and the queue then spends
    the attempts it reserved. Reaching the stopped state this test is about now means
    outlasting both, which is the point -- that is what a provider being genuinely down
    looks like, as opposed to one dropped connection.
    """

    def __init__(self, *, failures: int) -> None:
        self.calls = 0
        self._remaining = failures
        self._delegate = DeterministicFeatureProductManager()

    async def create_technical_prd(
        self, *, feature_id: str, prd: Any, design_snapshot: Any = None
    ) -> Any:
        """Reproduce a transient provider fault, then succeed once the budget is spent."""
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise TimingOutRuntimeRunner._failure()
        return await self._delegate.create_technical_prd(
            feature_id=feature_id, prd=prd, design_snapshot=design_snapshot
        )

    async def reconcile_requirements(self, **kwargs: Any) -> Any:
        """Reconcile as the deterministic composition does: this test asks nothing of a human."""
        return await self._delegate.reconcile_requirements(**kwargs)


class DiagnosticPlannerRunner(_AlwaysFailingRunner):
    """Simulate a planner that rejected both attempts and explained why."""

    async def start(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise AgentArtifactError(
            "feature planner could not produce a valid plan after one repair attempt",
            diagnostics=[
                "first attempt rejected: workstream 'ws-web' assigns unknown technical "
                "requirements: ['nfr-observability']",
                "repair attempt rejected: repository execution plan contains a workstream "
                "dependency cycle",
            ],
            stage="feature_planner",
        )

    async def advance_one_step(self, *_args: object, **_kwargs: object) -> NoReturn:
        await self.start()

    async def resume(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise NotImplementedError


class WaitingForHumanRunner:
    """Park the feature awaiting clarification, then reject mismatched answers."""

    async def start(
        self, state: FeatureWorkflowSnapshot, *, credentials: RequestScopedCredentials
    ) -> FeatureWorkflowSnapshot:
        """Leave the feature waiting for the human answers it asked for."""
        del credentials
        return state.model_copy(
            update={
                "status": FeatureWorkflowStatus.WAITING_FOR_HUMAN,
                "current_agent": "human_clarification",
            }
        )

    async def advance_one_step(
        self, state: FeatureWorkflowSnapshot, *, credentials: RequestScopedCredentials
    ) -> FeatureWorkflowSnapshot:
        """One step of this runner is the whole of it: park and wait."""
        return await self.start(state, credentials=credentials)

    async def begin_resume(self, *_args: object, **_kwargs: object) -> NoReturn:
        """Refuse answers that do not match the pending questions."""
        raise ClarificationAnswerError("clarification answers must match unresolved question IDs")

    async def grant_and_run_one_retry(self, *_args: object, **_kwargs: object) -> NoReturn:
        """Unused by this runner."""
        raise NotImplementedError

    async def answer_design_conflict(self, *_args: object, **_kwargs: object) -> NoReturn:
        """Unused by this runner."""
        raise NotImplementedError

    async def resume(self, *_args: object, **_kwargs: object) -> NoReturn:
        """Refuse answers that do not match the pending questions."""
        raise ClarificationAnswerError("clarification answers must match unresolved question IDs")

    async def retry_workstream(self, *_args: object, **_kwargs: object) -> NoReturn:
        """Unused by this runner."""
        raise NotImplementedError

    async def publish_feature(self, *_args: object, **_kwargs: object) -> NoReturn:
        """Unused by this runner."""
        raise NotImplementedError

    async def approve_contract_change(self, *_args: object, **_kwargs: object) -> NoReturn:
        """Unused by this runner."""
        raise NotImplementedError

    async def reject_contract_change(self, *_args: object, **_kwargs: object) -> NoReturn:
        """Unused by this runner."""
        raise NotImplementedError

    async def approve_repository_repair(self, *_args: object, **_kwargs: object) -> NoReturn:
        """Unused by this runner."""
        raise NotImplementedError

    async def reject_repository_repair(self, *_args: object, **_kwargs: object) -> NoReturn:
        """Unused by this runner."""
        raise NotImplementedError


class PausingCancellationFeatureStore(SqlAlchemyFeatureControlPlane):
    """Hold a stale cancellation snapshot while a newer checkpoint commits."""

    def __init__(self, database: Database) -> None:
        super().__init__(database)
        self.cancellation_write_started = asyncio.Event()
        self.allow_cancellation_write = asyncio.Event()

    async def _replace_state(
        self,
        feature_id: str,
        state: FeatureWorkflowSnapshot,
        event: str,
        details: dict[str, Any] | None = None,
        *,
        checkpoint_boundary: WorkflowCheckpointBoundary | None = None,
        checkpoint_repository_id: str | None = None,
        preserve_persisted_progress: bool = False,
    ) -> None:
        """Pause only the initial cancellation write, then delegate normally."""
        if event == "feature_cancellation_requested":
            self.cancellation_write_started.set()
            await self.allow_cancellation_write.wait()
        await super()._replace_state(
            feature_id,
            state,
            event,
            details,
            checkpoint_boundary=checkpoint_boundary,
            checkpoint_repository_id=checkpoint_repository_id,
            preserve_persisted_progress=preserve_persisted_progress,
        )


@pytest.mark.asyncio
async def test_feature_state_and_idempotency_survive_a_new_control_plane(tmp_path: Path) -> None:
    """Parent artifacts and child status survive a process-style control-plane rehydrate."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'feature.db'}")
    await database.create_schema()
    try:
        request = StartFeatureRequest.model_validate_json(json.dumps(feature_payload()))
        first_store = SqlAlchemyFeatureControlPlane(
            database, mock_runner=FeatureWorkflowOrchestrator()
        )
        created = await first_store.start(
            request,
            idempotency_key="feature-state-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        # Acceptance and execution are separate durable steps, so the run happens here rather
        # than inside `start`. This is the same dispatcher the deployment uses.
        await drain_feature_queue(first_store)
        second_store = SqlAlchemyFeatureControlPlane(
            database, mock_runner=FeatureWorkflowOrchestrator()
        )
        replay = await second_store.start(
            request,
            idempotency_key="feature-state-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        restored = await second_store.get_record("feature-state")
        workstreams = await second_store.workstreams("feature-state")

        assert created.created is True
        assert replay.created is False
        assert restored.state.status.value == "completed"
        assert restored.state.workflow_schema_version == WORKFLOW_SCHEMA_VERSION
        assert (
            restored.state.created_by_build_revision == restored.state.last_executor_build_revision
        )
        completion = next(
            item for item in restored.state.artifacts if isinstance(item, FeatureCompletionArtifact)
        )
        assert completion.metadata["executor_build_revision"] == (
            restored.state.last_executor_build_revision
        )
        assert completion.metadata["workflow_schema_version"] == WORKFLOW_SCHEMA_VERSION
        assert [item.repository_id for item in workstreams] == ["backend", "frontend"]
        assert {item.status.value for item in workstreams} == {"completed"}
        assert "request-openai-token" not in json.dumps(restored.state.model_dump(mode="json"))

        incompatible = restored.state.model_dump(mode="python")
        incompatible["workflow_schema_version"] = "0.9"
        with pytest.raises(ValidationError, match="incompatible with executor schema"):
            FeatureWorkflowSnapshot.model_validate(incompatible)

        legacy = restored.state.model_dump(mode="python")
        for field in (
            "workflow_schema_version",
            "created_by_build_revision",
            "last_executor_build_revision",
        ):
            legacy.pop(field)
        with pytest.raises(ValidationError, match="Field required"):
            FeatureWorkflowSnapshot.model_validate(legacy)
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_migrated_feature_is_audit_readable_but_all_mutations_fail_closed(
    tmp_path: Path,
) -> None:
    """A legacy identity may be inspected but cannot resume or rewrite durable state."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'legacy-feature.db'}")
    await database.create_schema()
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    try:
        request = StartFeatureRequest.model_validate(feature_payload()).model_copy(
            update={"feature_id": "legacy-feature"}
        )
        store = SqlAlchemyFeatureControlPlane(database)
        await store.start(
            request,
            idempotency_key="legacy-feature-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        async with database.session() as session:
            model = await session.get(FeatureWorkflowModel, "legacy-feature")
            assert model is not None
            state_json = dict(model.state_json)
            state_json["created_by_build_revision"] = LEGACY_UNVERIFIED_BUILD_REVISION
            state_json["last_executor_build_revision"] = LEGACY_UNVERIFIED_BUILD_REVISION
            model.state_json = state_json
            await session.commit()

        before = await store.get_record("legacy-feature")
        before_events = await store.timeline("legacy-feature")
        assert before.state.created_by_build_revision == LEGACY_UNVERIFIED_BUILD_REVISION

        with pytest.raises(WorkflowConflictError, match="audit-only.*legacy-unverified"):
            await store.resume("legacy-feature", answers=[], credentials=credentials)
        with pytest.raises(WorkflowConflictError, match="audit-only.*legacy-unverified"):
            await store.cancel("legacy-feature", requested_by="platform-admin")
        with pytest.raises(WorkflowConflictError, match="audit-only.*legacy-unverified"):
            await store.checkpoint(before.state, WorkflowCheckpointBoundary.BEFORE_CODING)

        after = await store.get_record("legacy-feature")
        assert after.state == before.state
        assert after.updated_at == before.updated_at
        assert await store.timeline("legacy-feature") == before_events

        incompatible = before.state.model_copy(update={"workflow_schema_version": "0.9"})
        with pytest.raises(WorkflowConflictError, match="schema '0.9' is incompatible"):
            await store.checkpoint(incompatible, WorkflowCheckpointBoundary.BEFORE_CODING)
        missing_creator = before.state.model_copy(update={"created_by_build_revision": ""})
        with pytest.raises(WorkflowConflictError, match="missing its creator"):
            await store.checkpoint(missing_creator, WorkflowCheckpointBoundary.BEFORE_CODING)
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_child_recovery_diagnostics_survive_a_process_restart(tmp_path: Path) -> None:
    """Recovery must not erase the precise boundary, checkout evidence, or retry refusal."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'child-recovery.db'}")
    await database.create_schema()
    try:
        request = StartFeatureRequest.model_validate_json(json.dumps(feature_payload()))
        first_store = SqlAlchemyFeatureControlPlane(
            database, mock_runner=FeatureWorkflowOrchestrator()
        )
        await first_store.start(
            request,
            idempotency_key="child-recovery-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        await drain_feature_queue(first_store)
        state = (await first_store.get_record("feature-state")).state.model_copy(deep=True)
        child = state.child_workflows["backend"]
        state.child_workflows["backend"] = child.model_copy(
            update={
                "checkpoint_boundary": WorkflowCheckpointBoundary.BEFORE_CODING,
                "layout_evidence": {
                    "checkout_root": child.workspace_path,
                    "manifest_paths": ["pyproject.toml"],
                },
                "retry_refusal_reason": "same_failure_without_meaningful_change",
            }
        )
        await first_store.checkpoint(state, WorkflowCheckpointBoundary.BEFORE_CODING, "backend")

        restored = await SqlAlchemyFeatureControlPlane(
            database, mock_runner=FeatureWorkflowOrchestrator()
        ).get_record("feature-state")
        restored_child = restored.state.child_workflows["backend"]

        assert restored_child.checkpoint_boundary is WorkflowCheckpointBoundary.BEFORE_CODING
        assert restored_child.layout_evidence == {
            "checkout_root": child.workspace_path,
            "manifest_paths": ["pyproject.toml"],
        }
        assert restored_child.retry_refusal_reason == "same_failure_without_meaningful_change"
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_repository_checkpoints_merge_stale_sibling_snapshots_and_artifacts(
    tmp_path: Path,
) -> None:
    """Parallel workstreams must not last-write-win away a sibling result or handoff."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'checkpoint-merge.db'}")
    await database.create_schema()
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    try:
        store = SqlAlchemyFeatureControlPlane(database)
        initial = await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="checkpoint-merge-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        base = await FeatureWorkflowOrchestrator().start(
            initial.record.state, credentials=credentials
        )
        await store.checkpoint(base, WorkflowCheckpointBoundary.AFTER_VALIDATION)

        backend = base.model_copy(deep=True)
        backend.child_workflows["backend"] = backend.child_workflows["backend"].model_copy(
            update={"retry_refusal_reason": "backend-checkpoint"}
        )
        frontend = base.model_copy(deep=True)
        frontend.child_workflows["frontend"] = frontend.child_workflows["frontend"].model_copy(
            update={"retry_refusal_reason": "frontend-checkpoint"}
        )
        completion = next(
            item for item in base.artifacts if isinstance(item, FeatureCompletionArtifact)
        )
        backend.artifacts.append(
            completion.model_copy(update={"artifact_id": "checkpoint.backend.json"})
        )
        frontend.artifacts.append(
            completion.model_copy(update={"artifact_id": "checkpoint.frontend.json"})
        )

        # Separate control-plane objects model concurrent local API instances. Production
        # additionally owns the same state-write key through Redis and the parent row lock.
        backend_store = SqlAlchemyFeatureControlPlane(database)
        frontend_store = SqlAlchemyFeatureControlPlane(database)
        await asyncio.gather(
            backend_store.checkpoint(backend, WorkflowCheckpointBoundary.BEFORE_CODING, "backend"),
            frontend_store.checkpoint(
                frontend, WorkflowCheckpointBoundary.AFTER_VALIDATION, "frontend"
            ),
        )

        restored = (await store.get_record("feature-state")).state
        assert restored.child_workflows["backend"].retry_refusal_reason == "backend-checkpoint"
        assert restored.child_workflows["frontend"].retry_refusal_reason == "frontend-checkpoint"
        assert restored.checkpoint_boundaries["backend"] is WorkflowCheckpointBoundary.BEFORE_CODING
        assert (
            restored.checkpoint_boundaries["frontend"]
            is WorkflowCheckpointBoundary.AFTER_VALIDATION
        )
        assert {item.artifact_id for item in restored.artifacts} >= {
            "checkpoint.backend.json",
            "checkpoint.frontend.json",
        }
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_stale_cancellation_snapshot_preserves_a_concurrent_checkpoint(
    tmp_path: Path,
) -> None:
    """Cancellation lifecycle fields may win, but newer child and artifact progress must remain."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'cancel-checkpoint-race.db'}")
    await database.create_schema()
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    try:
        store = PausingCancellationFeatureStore(database)
        initial = await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="cancel-checkpoint-race-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        progressed = await FeatureWorkflowOrchestrator().start(
            initial.record.state, credentials=credentials
        )

        cancellation = asyncio.create_task(
            store.cancel("feature-state", reason="operator stop", requested_by="platform-admin")
        )
        await store.cancellation_write_started.wait()
        await store.checkpoint(progressed, WorkflowCheckpointBoundary.AFTER_VALIDATION)
        store.allow_cancellation_write.set()
        cancelled = await cancellation

        assert cancelled.state.status is FeatureWorkflowStatus.CANCELLED
        assert set(cancelled.state.child_workflows) == {"backend", "frontend"}
        assert {child.status.value for child in cancelled.state.child_workflows.values()} == {
            "completed"
        }
        assert any(
            isinstance(item, FeatureCompletionArtifact) for item in cancelled.state.artifacts
        )
        assert (
            cancelled.state.checkpoint_boundaries["parent"]
            is WorkflowCheckpointBoundary.AFTER_VALIDATION
        )
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_unknown_operation_without_reference_requires_cleanup_by_operation_id(
    tmp_path: Path,
) -> None:
    """Unknown provider state can never disappear merely because no resource URL was captured."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'unknown-cleanup.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    try:
        request = StartFeatureRequest.model_validate(feature_payload()).model_copy(
            update={"feature_id": "unknown-cleanup"}
        )
        store = SqlAlchemyFeatureControlPlane(database, operation_journal=journal)
        started = await store.start(
            request,
            idempotency_key="unknown-cleanup-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        operation = await journal.create_operation(
            workflow_id="unknown-cleanup",
            feature_id="unknown-cleanup",
            child_workflow_id="unknown-cleanup:backend",
            repository_id="backend",
            operation_type=ExternalOperationType.PUSH_BRANCH,
            idempotency_key="unknown-cleanup:push",
            input_fingerprint="unknown-cleanup-push",
        )
        await journal.record_failure(
            operation.operation_id,
            status=ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
            error_code="provider_timeout",
            error_message="provider state requires reconciliation",
        )

        cancelled = await store.cancel("unknown-cleanup", requested_by="platform-admin")

        assert cancelled.state.status is FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS
        assert (
            cancelled.state.cancellation_status
            is CancellationLifecycleStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS
        )
        assert len(cancelled.state.cleanup_requirements) == 1
        cleanup = cancelled.state.cleanup_requirements[0]
        assert cleanup.external_reference == operation.operation_id
        assert cleanup.repository_id == "backend"
        assert "manual reconciliation" in cleanup.reason
        assert "do not assume the side effect is absent" in cleanup.recommended_action

        # A runner response already in flight at cancellation time cannot resurrect the
        # terminal state or erase the operation ID needed for reconciliation.
        await store.checkpoint(started.record.state, WorkflowCheckpointBoundary.AFTER_VALIDATION)
        restored = (await store.get_record("unknown-cleanup")).state
        assert restored.status is FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS
        assert restored.cleanup_requirements == cancelled.state.cleanup_requirements
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_invalid_model_artifact_returns_controlled_feature_state_not_an_exception(
    tmp_path: Path,
) -> None:
    """Invalid model planning is visible as human action, never an Internal Server Error."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'invalid-artifact.db'}")
    await database.create_schema()
    try:
        request = StartFeatureRequest.model_validate(feature_payload())
        request = request.model_copy(update={"feature_id": "invalid-artifact"})
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=InvalidArtifactRunner())

        result = await store.start(
            request,
            idempotency_key="invalid-artifact-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        executed = await store.get_record("invalid-artifact")

        assert result.created is True
        # Queued when the caller was answered, and recorded as needing a human once the
        # worker found the artifact unusable. Both are true in turn; neither is a 500.
        assert result.record.state.status is FeatureWorkflowStatus.PENDING
        assert executed.state.status.value == "failed_requires_human"
        assert executed.state.current_agent == "feature_planner"
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_unexpected_runtime_failure_returns_controlled_feature_state_not_a_500(
    tmp_path: Path,
) -> None:
    """Execution faults are durable workflow outcomes rather than uncaught HTTP failures."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'runtime-failure.db'}")
    await database.create_schema()
    try:
        request = StartFeatureRequest.model_validate(feature_payload())
        request = request.model_copy(update={"feature_id": "runtime-failure"})
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FailingRuntimeRunner())

        result = await store.start(
            request,
            idempotency_key="runtime-failure-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        executed = await store.get_record("runtime-failure")

        assert result.created is True
        assert executed.state.status.value == "failed_requires_human"
        assert executed.state.current_agent == "feature_runtime"
        summary = executed.state.failure_summary
        assert summary is not None
        assert summary.stage == "feature_runtime"
        # A `RuntimeError` nothing anticipated is this platform's defect, and is recorded as
        # one. Before, the type name was the classification, which reads as a finding about
        # the repository and sent a person to look at their own code.
        assert summary.root_classification == "platform_defect"
        assert summary.retryable is False
        assert any("defect in the platform" in item for item in summary.diagnostics)
        # An arbitrary exception's text can carry credentials and must never be persisted.
        failure = _failure_event(executed, "feature_execution_failed")
        assert failure == {"error_type": "RuntimeError"}
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_provider_timeout_is_recorded_as_worth_resuming(tmp_path: Path) -> None:
    """A provider that did not answer says nothing about whether the work was wrong.

    Feature -055 died on one `APITimeoutError` after the PRD artifact and nothing else,
    because every failure was recorded as terminal. The journal already replays completed
    effects, so this costs one resume rather than a whole feature and a human decision.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'provider-timeout.db'}")
    await database.create_schema()
    try:
        request = StartFeatureRequest.model_validate(feature_payload())
        request = request.model_copy(update={"feature_id": "provider-timeout"})
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=TimingOutRuntimeRunner())

        await store.start(
            request,
            idempotency_key="provider-timeout-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        result = await store.get_record("provider-timeout")

        summary = result.state.failure_summary
        assert summary is not None
        # The provider SDK's exception name is what makes this recognisable as a timeout, but
        # it is not a domain classification: it is normalised to the one the platform owns,
        # and the retryable verdict it used to drive by substring match is preserved.
        assert summary.root_classification == "provider_unavailable"
        assert summary.retryable is True
        # The operator must be told the recovery action, not sent to read diagnostics that
        # a timeout never produces.
        assert "resume" in summary.next_action.lower()
        # And the feature must actually be in a state that accepts one. Recording it as
        # failed contradicted its own instruction: resume rejects that status, so run-3's
        # timeout looked terminal to every caller until an empty-answer recovery was found.
        assert result.state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
        failure = _failure_event(result, "feature_execution_failed")
        assert failure == {
            "error_type": "LLMAdapterError",
            "failure_classification": "APITimeoutError",
            "diagnostics": ["the model provider call failed (APITimeoutError)"],
        }
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_pre_prd_provider_timeout_can_resume_from_the_original_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resumable must mean executable even when the failed call produced no Technical PRD."""
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'provider-timeout-resume.db'}")
    await database.create_schema()
    try:
        request = StartFeatureRequest.model_validate(feature_payload()).model_copy(
            update={"feature_id": "provider-timeout-resume"}
        )
        # Enough to outlast the in-process allowance on every one of the queue's attempts,
        # so the feature really does reach a person. Fewer and the platform recovers by
        # itself, which is the behaviour the two tests above this one cover.
        product_manager = ProductManagerThatTimesOut(failures=9)
        store = SqlAlchemyFeatureControlPlane(
            database,
            mock_runner=FeatureWorkflowOrchestrator(product_manager=product_manager),
        )
        credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)

        await store.start(
            request,
            idempotency_key="provider-timeout-resume-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        stopped = await store.get_record("provider-timeout-resume")
        assert stopped.state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
        assert stopped.state.failure_summary is not None
        assert stopped.state.failure_summary.retryable is True

        await store.resume("provider-timeout-resume", answers=[], credentials=credentials)
        await drain_feature_queue(store)

        recovered = await store.get_record("provider-timeout-resume")
        assert recovered.state.status is FeatureWorkflowStatus.COMPLETED
        assert recovered.state.failure_summary is None
        # Nine failed calls -- three queued attempts, each retrying the provider twice --
        # and then the tenth, on the operator's resume, which produced the Technical PRD.
        assert product_manager.calls == 10
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_wrong_clarification_answers_leave_the_feature_answerable(tmp_path: Path) -> None:
    """A caller mistake must not strand a feature that is still waiting for input.

    Feature -034 was lost this way: one resume with the wrong answers marked it failed, and
    the corrected resume was then refused because the feature was no longer waiting.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'clarification.db'}")
    await database.create_schema()
    try:
        request = StartFeatureRequest.model_validate(feature_payload())
        request = request.model_copy(update={"feature_id": "clarification-guard"})
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=WaitingForHumanRunner())
        credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
        await store.start(
            request,
            idempotency_key="clarification-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)

        with pytest.raises(WorkflowConflictError):
            await store.resume(
                "clarification-guard",
                answers=[ClarificationAnswer(question_id="Q-unknown", answer="no")],
                credentials=credentials,
            )

        record = await store.get_record("clarification-guard")
        assert record.state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_planning_failure_records_why_the_plan_was_rejected(tmp_path: Path) -> None:
    """No plan artifact is written on this path, so the event is the only explanation."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'planner-diagnostics.db'}")
    await database.create_schema()
    try:
        request = StartFeatureRequest.model_validate(feature_payload())
        request = request.model_copy(update={"feature_id": "planner-diagnostics"})
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=DiagnosticPlannerRunner())

        await store.start(
            request,
            idempotency_key="planner-diagnostics-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        result = await store.get_record("planner-diagnostics")

        failure = _failure_event(result, "feature_planning_failed")
        assert failure["error_type"] == "AgentArtifactError"
        diagnostics = failure["diagnostics"]
        assert isinstance(diagnostics, list)
        assert "nfr-observability" in diagnostics[0]
        assert "dependency cycle" in diagnostics[1]
    finally:
        await database.drop_schema()
        await database.dispose()


class BrokenAfterStartRunner(FeatureWorkflowOrchestrator):
    """A platform that starts a feature and then cannot carry anything else out.

    Deliberately not a refusal. `FeatureWorkflowError` already reaches the caller as a
    conflict; what was untested is the *unexpected* failure -- a full workspace volume, an
    unconfigured runtime -- which is the shape that used to be answered as a success.

    The step surface is broken alongside the feature-scoped one, because a queued mutation
    reaches the runner through `begin_resume` and `grant_and_run_one_retry` now. `start` and
    the ordinary step are left working: this is a platform that got the feature going and
    then could not carry anything else out, which is the situation being reproduced.
    """

    async def resume(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("workspace volume is full: /workspaces sk-secret-in-message")

    async def begin_resume(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("workspace volume is full: /workspaces sk-secret-in-message")

    async def retry_workstream(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("workspace volume is full: /workspaces sk-secret-in-message")

    async def grant_and_run_one_retry(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("workspace volume is full: /workspaces sk-secret-in-message")

    async def approve_repository_repair(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("workspace volume is full: /workspaces sk-secret-in-message")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["repair"])
async def test_an_operation_that_did_not_happen_is_not_reported_as_success(
    tmp_path: Path, operation: str
) -> None:
    """A mutation the platform could not carry out must refuse, not return the feature.

    Every mutation now runs inside a durable action, and that action's result is the audit
    answer to "did this happen?". Returning the unchanged feature was indistinguishable from
    success from the outside, so the action recorded `committed` for an operation that had
    committed nothing -- and a caller relying on the platform's own action identity replayed
    that false success on every later attempt instead of trying again.

    Marking the feature as needing attention is still right and still happens; what must not
    happen is answering the caller as though the work was done.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / f'not-done-{operation}.db'}")
    await database.create_schema()
    try:
        credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
        request = StartFeatureRequest.model_validate(feature_payload()).model_copy(
            update={"feature_id": f"not-done-{operation}"}
        )
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=BrokenAfterStartRunner())
        await store.start(
            request,
            idempotency_key=f"not-done-{operation}",
            credentials=credentials,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        # A stopped feature holding a stopped repository, which is the only state in which
        # either of these requests is something a person would make. The mock start completes
        # every repository, so without this the request under test is made against a finished
        # feature and what fails is the request rather than the runner.
        #
        # This used to be built for the retry case alone, and the repair case therefore
        # approved a repair on a `completed` feature. That mattered once task 30- made a
        # completed feature absorbing: the failure handler had been rewriting the completion
        # as `failed_requires_human`, and the assertion at the end of this test was reading
        # that rewrite. The precondition is now the same for both, so the assertion reads what
        # it always meant to -- a failure that is durably recorded on a feature that had not
        # already finished.
        state = (await store.get_record(f"not-done-{operation}")).state
        state.status = FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        child = state.child_workflows["backend"]
        state.child_workflows["backend"] = child.model_copy(
            update={"status": ChildWorkflowStatus.FAILED}
        )
        await store._replace_state(  # noqa: SLF001 - building the precondition, not the behaviour
            f"not-done-{operation}", state, "feature_failed_requires_human"
        )

        with pytest.raises(FeatureOperationFailedError) as raised:
            if operation == "resume":
                await store.resume(f"not-done-{operation}", answers=[], credentials=credentials)
            elif operation == "retry":
                await store.retry_workstream(
                    f"not-done-{operation}",
                    "backend",
                    additional_attempts=1,
                    requested_by="akhilesh",
                    reason="The runner was fixed out of band.",
                    credentials=credentials,
                )
            else:
                await store.approve_repair(
                    f"not-done-{operation}",
                    repair_id="repair-1",
                    actor_id="akhilesh",
                    credentials=credentials,
                )

        # The refusal quotes the platform, never the fault: an arbitrary exception's text can
        # carry a credential, and this string reaches an HTTP client and a chat transcript.
        message = str(raised.value)
        assert "sk-secret-in-message" not in message
        assert "/workspaces" not in message
        assert "Nothing was carried out" in message

        # The diagnosis is still durable. Refusing the caller must not cost the record of why.
        record = await store.get_record(f"not-done-{operation}")
        assert record.state.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["resume", "retry"])
async def test_a_queued_mutation_that_fails_reaches_the_feature_not_the_caller(
    tmp_path: Path, operation: str
) -> None:
    """Resuming and retrying are queued now, so their failures surface on the feature.

    They used to run the whole feature inside the caller's request and could therefore
    refuse synchronously. That is exactly what cost AB-Feature-108 seventy minutes of work:
    the run existed only for as long as a socket did. The invariant that mattered survives
    the move -- a failure is still durably recorded, still names no secret, and still leaves
    the feature needing a person -- but it is now read off the feature rather than raised at
    whoever happened to be holding the connection.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / f'queued-{operation}.db'}")
    await database.create_schema()
    try:
        credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
        feature_id = f"queued-{operation}"
        request = StartFeatureRequest.model_validate(feature_payload()).model_copy(
            update={"feature_id": feature_id}
        )
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=BrokenAfterStartRunner())
        await store.start(
            request, idempotency_key=feature_id, credentials=credentials, owner_id="platform-admin"
        )
        await drain_feature_queue(store)
        # The mock start completes the feature, and a completed feature is neither resumable
        # nor retryable. Put it where a person would actually find one of these requests: at
        # a failure, with a stopped repository, so the fault under test is the runner's.
        state = (await store.get_record(feature_id)).state
        state.status = FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        child = state.child_workflows["backend"]
        state.child_workflows["backend"] = child.model_copy(
            update={"status": ChildWorkflowStatus.FAILED}
        )
        await store._replace_state(  # noqa: SLF001 - building the precondition, not the behaviour
            feature_id, state, "feature_failed_requires_human"
        )

        # Accepted, not carried out. The caller is told what to watch, not what happened.
        if operation == "resume":
            await store.resume(feature_id, answers=[], credentials=credentials)
        else:
            await store.retry_workstream(
                feature_id,
                "backend",
                additional_attempts=1,
                requested_by="akhilesh",
                reason="The runner was fixed out of band.",
                credentials=credentials,
            )
        # The work is durable and claimable rather than tied to the request that asked.
        assert await store.queue.pending_count() == 1
        await drain_feature_queue(store)

        # The diagnosis is still durable, and still quotes the platform rather than the fault:
        # an arbitrary exception's text can carry a credential.
        record = await store.get_record(feature_id)
        assert record.state.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        summary = record.state.failure_summary
        assert summary is not None
        rendered = " ".join(summary.diagnostics)
        assert "sk-secret-in-message" not in rendered
        assert "/workspaces" not in rendered
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_resuming_a_finished_feature_is_refused_rather_than_silently_dropped(
    tmp_path: Path,
) -> None:
    """Queueing work a worker will discard is worse than refusing it.

    The caller would be answered as though something had been arranged, and nothing would
    ever happen or be recorded. So the checks a queued resume cannot make later are all
    still made here, before the entry is written.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'finished.db'}")
    await database.create_schema()
    try:
        credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
        request = StartFeatureRequest.model_validate(feature_payload()).model_copy(
            update={"feature_id": "finished-feature"}
        )
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        await store.start(
            request,
            idempotency_key="finished-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        assert (
            await store.get_record("finished-feature")
        ).state.status is FeatureWorkflowStatus.COMPLETED

        with pytest.raises(WorkflowConflictError, match="cannot be resumed"):
            await store.resume("finished-feature", answers=[], credentials=credentials)

        assert await store.queue.pending_count() == 0
    finally:
        await database.drop_schema()
        await database.dispose()


def _failure_event(record: FeatureRecord, event: str) -> dict[str, object]:
    """Return the recorded details of one lifecycle failure event."""
    events = [item for item in record.lifecycle_events if item.event == event]
    assert len(events) == 1, f"expected exactly one {event} event"
    return dict(events[0].details)


@pytest.mark.asyncio
async def test_the_persisted_json_does_not_shadow_artifacts_or_child_workflows(
    tmp_path: Path,
) -> None:
    """The two fields with their own tables are not copied into the snapshot column.

    `_state_from_model` overwrites both from `feature_artifacts` and
    `feature_child_workflows` on every load, so the JSON copy was written and never read --
    while being the copy a person sees when they open the row in psql, and the reason a
    completed feature's snapshot ran to hundreds of kilobytes.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'shadow.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        await store.start(
            StartFeatureRequest.model_validate_json(json.dumps(feature_payload())),
            idempotency_key="shadow-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)

        loaded = (await store.get_record("feature-state")).state
        async with database.session() as session:
            model = await session.get(FeatureWorkflowModel, "feature-state")
            assert model is not None
            persisted = dict(model.state_json)

        assert "artifacts" not in persisted
        assert "child_workflows" not in persisted
        # Not merely absent from the column: still present on the snapshot the platform
        # actually works with, rebuilt from the typed rows that own them.
        assert loaded.artifacts
        assert loaded.child_workflows
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_snapshot_round_trips_unchanged_through_the_durable_store(
    tmp_path: Path,
) -> None:
    """Load, save, load produces an identical snapshot once the shadow copies are gone."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'round-trip.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        await store.start(
            StartFeatureRequest.model_validate_json(json.dumps(feature_payload())),
            idempotency_key="round-trip-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)

        # Save once first, so the comparison is between two loads of a snapshot that has
        # already been through the writer -- not between a fresh run and its first save,
        # which a checkpoint boundary legitimately changes.
        await store.checkpoint(
            (await store.get_record("feature-state")).state.model_copy(deep=True),
            WorkflowCheckpointBoundary.BEFORE_CODING,
            "backend",
        )
        first = (await store.get_record("feature-state")).state
        await store.checkpoint(
            first.model_copy(deep=True), WorkflowCheckpointBoundary.BEFORE_CODING, "backend"
        )
        second = (
            await SqlAlchemyFeatureControlPlane(
                database, mock_runner=FeatureWorkflowOrchestrator()
            ).get_record("feature-state")
        ).state

        # Compared as artifacts and children, not only as a blob: those are exactly the two
        # the JSON no longer carries, so a lost one would otherwise pass unnoticed.
        assert [item.artifact_id for item in second.artifacts] == [
            item.artifact_id for item in first.artifacts
        ]
        assert second.child_workflows == first.child_workflows
        assert second.model_dump(mode="json") == first.model_dump(mode="json")
    finally:
        await database.drop_schema()
        await database.dispose()


def test_nothing_reads_the_shadow_copies_out_of_the_persisted_json() -> None:
    """A grep guard, because the exclusion is only safe while no reader expects them.

    Subscripting `state_json` with either name would read a field that is now absent from
    every row written after this change, and would do it silently -- `dict.get` returns
    ``None`` rather than raising.
    """
    root = Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.parts[0] in {".venv", "tests"}:
            continue
        source = path.read_text(encoding="utf-8")
        if "state_json" not in source:
            continue
        for number, line in enumerate(source.splitlines(), start=1):
            if "state_json" not in line:
                continue
            if any(f'state_json["{field}"]' in line for field in ("artifacts", "child_workflows")):
                offenders.append(f"{relative}:{number}")
            if any(
                f'state_json.get("{field}"' in line for field in ("artifacts", "child_workflows")
            ):
                offenders.append(f"{relative}:{number}")

    assert offenders == [], f"persisted JSON no longer carries these: {offenders}"


def feature_payload() -> dict[str, object]:
    """Return a complete mock feature request without any provider credential fields."""
    return {
        "feature_id": "feature-state",
        "prd": {
            "title": "Feature state persistence",
            "problem_statement": "Operators need durable multi-repository feature state.",
            "goals": ["Persist child workflow state."],
            "user_stories": [
                {
                    "story_id": "story-state",
                    "persona": "Operator",
                    "need": "Recover a feature after restart",
                    "benefit": "I can continue safely",
                    "acceptance_criteria": ["Child status is retained."],
                }
            ],
            "requirements": [
                {
                    "requirement_id": "requirement-state",
                    "description": "Persist feature parent and child state.",
                    "priority": "must",
                    "acceptance_criteria": ["A new control plane reads the same feature."],
                    "dependencies": [],
                }
            ],
            "constraints": ["No provider tokens in data."],
            "out_of_scope": ["Automatic merge."],
            "stakeholders": ["Platform"],
        },
        "repositories": [
            {
                "repository_id": "backend",
                "name": "Backend",
                "role": "backend",
                "repository_url": "https://github.com/example/backend.git",
                "default_branch": "main",
            },
            {
                "repository_id": "frontend",
                "name": "Frontend",
                "role": "frontend",
                "repository_url": "https://github.com/example/frontend.git",
                "default_branch": "main",
            },
        ],
    }


@pytest.mark.asyncio
async def test_an_operator_can_retire_an_audit_only_feature(tmp_path: Path) -> None:
    """The one write a legacy snapshot must accept is somebody closing it out.

    Resume, cancel and checkpoint all refuse an audit-only snapshot, correctly, because its
    recorded build identity is not evidence of what produced it. That left the two workflows
    the audit named stuck in a running status with no path to resolve them but editing rows by
    hand. Retirement executes nothing and discards nothing; it records a decision and its owner.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'retire-feature.db'}")
    await database.create_schema()
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    try:
        request = StartFeatureRequest.model_validate(feature_payload()).model_copy(
            update={"feature_id": "stale-feature"}
        )
        store = SqlAlchemyFeatureControlPlane(database)
        await store.start(
            request, idempotency_key="stale-001", credentials=credentials, owner_id="platform-admin"
        )
        async with database.session() as session:
            model = await session.get(FeatureWorkflowModel, "stale-feature")
            assert model is not None
            state_json = dict(model.state_json)
            state_json["created_by_build_revision"] = LEGACY_UNVERIFIED_BUILD_REVISION
            state_json["last_executor_build_revision"] = LEGACY_UNVERIFIED_BUILD_REVISION
            model.state_json = state_json
            model.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
            await session.commit()

        artifacts_before = await store.artifacts("stale-feature")

        with pytest.raises(WorkflowConflictError, match="audit-only.*legacy-unverified"):
            await store.cancel("stale-feature", requested_by="platform-admin")

        record = await store.retire(
            "stale-feature", reason="stale historical workflow", operator="akhilesh"
        )

        assert record.state.status is FeatureWorkflowStatus.CANCELLED
        # Nothing was executed and nothing was discarded.
        assert await store.artifacts("stale-feature") == artifacts_before
        events = await store.timeline("stale-feature")
        retired = [item for item in events if item[3] == "feature_retired_by_operator"]
        assert len(retired) == 1
        assert retired[0][4]["operator"] == "akhilesh"
        assert retired[0][4]["previous_status"] == "running_child_workflows"
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_retiring_a_feature_requires_an_operator_and_a_reason(tmp_path: Path) -> None:
    """An administrative close-out without an owner is an unattributed state change."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'retire-args.db'}")
    await database.create_schema()
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    try:
        request = StartFeatureRequest.model_validate(feature_payload()).model_copy(
            update={"feature_id": "needs-owner"}
        )
        store = SqlAlchemyFeatureControlPlane(database)
        await store.start(
            request,
            idempotency_key="needs-owner-001",
            credentials=credentials,
            owner_id="platform-admin",
        )

        with pytest.raises(WorkflowConflictError, match="operator and a reason"):
            await store.retire("needs-owner", reason="   ", operator="akhilesh")
        with pytest.raises(WorkflowConflictError, match="operator and a reason"):
            await store.retire("needs-owner", reason="stale", operator="")
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_live_start_is_refused_once_the_concurrency_quota_is_reached(
    tmp_path: Path,
) -> None:
    """Nothing bounded how many live features could run at once.

    Each one clones every repository it touches and installs their dependency trees, so an
    unbounded burst meets the disk or the provider before it meets any policy this platform
    chose. `waiting_for_human` counts against the quota because it settles nothing: it holds
    its checkout open until somebody answers.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'quota.db'}")
    await database.create_schema()
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    try:
        store = SqlAlchemyFeatureControlPlane(database, max_concurrent_live_features=1)
        payload = feature_payload()
        payload["execution_mode"] = "live"
        first = StartFeatureRequest.model_validate(payload).model_copy(
            update={"feature_id": "quota-one"}
        )
        await store.start(
            first, idempotency_key="quota-one", credentials=credentials, owner_id="platform-admin"
        )

        second = StartFeatureRequest.model_validate(payload).model_copy(
            update={"feature_id": "quota-two"}
        )
        with pytest.raises(WorkflowConflictError, match="live feature quota reached"):
            await store.start(
                second,
                idempotency_key="quota-two",
                credentials=credentials,
                owner_id="platform-admin",
            )

        # Settling the first one frees the slot.
        await store.retire("quota-one", reason="freeing the slot", operator="akhilesh")
        result = await store.start(
            second, idempotency_key="quota-two", credentials=credentials, owner_id="platform-admin"
        )
        assert result.created is True
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_retry_grant_is_refused_before_a_runner_is_ever_reached(tmp_path: Path) -> None:
    """A caller who named the wrong repository must not break the feature they named.

    These refusals read persisted state and nothing else, so they have to be decided before
    the runner is built. Behind it they were unreachable without provider credentials: the
    error fell through to the generic failure handler, which marked the feature
    `failed_requires_human` and answered 200. The request that broke the feature looked
    exactly like the one that fixed it.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'retry-guard.db'}")
    await database.create_schema()
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    try:
        request = StartFeatureRequest.model_validate(feature_payload()).model_copy(
            update={"feature_id": "retry-guard"}
        )
        # No runner at all: reaching one would already be the bug.
        store = SqlAlchemyFeatureControlPlane(database)
        await store.start(
            request,
            idempotency_key="retry-guard-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        before = await store.get_record("retry-guard")

        with pytest.raises(WorkflowConflictError, match="not part of this feature"):
            await store.retry_workstream(
                "retry-guard",
                "mobile",
                additional_attempts=1,
                requested_by="akhilesh",
                reason="Typo in the repository id.",
                credentials=credentials,
            )

        after = await store.get_record("retry-guard")
        # The feature is exactly as it was. A rejected request changes nothing.
        assert after.state.status is before.state.status
        assert after.state.current_agent == before.state.current_agent
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_cancelling_settles_a_workstream_that_was_mid_attempt(tmp_path: Path) -> None:
    """AB-Feature-107 was cancelled through this path and its console still read `running`.

    A day later the row still said so. Cancellation finalizes the feature and never looks at
    the children, so nothing would ever schedule that repository again and nothing would ever
    change the row: an operator, the features list and the workflow graph all read work that
    had stopped as work in flight.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'cancel-settles-children.db'}")
    await database.create_schema()
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    try:
        store = SqlAlchemyFeatureControlPlane(database)
        initial = await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="cancel-settles-children-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        progressed = await FeatureWorkflowOrchestrator().start(
            initial.record.state, credentials=credentials
        )
        mid_flight = progressed.model_copy(deep=True)
        # Wound back to the moment AB-Feature-107 was cancelled at: the feature still running,
        # one repository finished, the other holding an attempt that will never report.
        mid_flight.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
        backend = mid_flight.child_workflows["backend"]
        mid_flight.child_workflows["backend"] = backend.model_copy(
            update={"status": ChildWorkflowStatus.RUNNING}
        )
        await store.checkpoint(mid_flight, WorkflowCheckpointBoundary.BEFORE_CODING)

        cancelled = await store.cancel(
            "feature-state", reason="operator stop", requested_by="platform-admin"
        )

        assert cancelled.state.status is FeatureWorkflowStatus.CANCELLED
        assert not [
            repository_id
            for repository_id, child in cancelled.state.child_workflows.items()
            if child.status in {ChildWorkflowStatus.RUNNING, ChildWorkflowStatus.PENDING}
        ], {key: item.status for key, item in cancelled.state.child_workflows.items()}
        # The sibling that finished keeps what it earned, which is what cancellation promises.
        assert cancelled.state.child_workflows["frontend"].status is ChildWorkflowStatus.COMPLETED
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_failed_feature_settles_a_workstream_left_mid_attempt(tmp_path: Path) -> None:
    """A sibling that raises hard enough to escape the fan-out leaves the other one open.

    AB-Feature-131 and -132 both hit a full disk. One workstream raised, the feature was
    recorded `failed_requires_human`, and the attempt still running beside it was never
    closed -- both sat at `running` under a terminal feature, unchanged eleven minutes later.
    That is the row AB-Feature-107 left behind after a cancellation, reached by failing
    instead, and the invariant is the same: a feature that has stopped deciding has no
    workstream still in flight.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'failed-settles-children.db'}")
    await database.create_schema()
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    try:
        store = SqlAlchemyFeatureControlPlane(database)
        initial = await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="failed-settles-children-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        progressed = await FeatureWorkflowOrchestrator().start(
            initial.record.state, credentials=credentials
        )
        # The moment the disk filled: one repository finished, the other mid-attempt, and the
        # feature about to be recorded terminal by the failure path.
        mid_flight = progressed.model_copy(deep=True)
        mid_flight.status = FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        backend = mid_flight.child_workflows["backend"]
        mid_flight.child_workflows["backend"] = backend.model_copy(
            update={"status": ChildWorkflowStatus.RUNNING}
        )
        await store.checkpoint(mid_flight, WorkflowCheckpointBoundary.AFTER_VALIDATION)

        settled = await store.get_record("feature-state")

        assert settled.state.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        assert not [
            repository_id
            for repository_id, child in settled.state.child_workflows.items()
            if child.status in {ChildWorkflowStatus.RUNNING, ChildWorkflowStatus.PENDING}
        ], {key: item.status for key, item in settled.state.child_workflows.items()}
        # Failed work reads as failed, not cancelled: the two say different things.
        assert settled.state.child_workflows["backend"].status is ChildWorkflowStatus.FAILED
        # And the sibling that finished keeps what it earned.
        assert settled.state.child_workflows["frontend"].status is ChildWorkflowStatus.COMPLETED
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_routing_decision_survives_a_process_restart(tmp_path: Path) -> None:
    """A decision already made for an attempt must not be re-decided after a crash.

    Re-deciding could read different evidence or changed deployment configuration and hand
    the resumed attempt a different model than the one it started under. The record here is
    deliberately one written by the retired ladder, so this also proves such a row still
    loads unchanged.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'model-routing.db'}")
    await database.create_schema()
    try:
        request = StartFeatureRequest.model_validate_json(json.dumps(feature_payload()))
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        await store.start(
            request,
            idempotency_key="model-routing-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        state = (await store.get_record("feature-state")).state.model_copy(deep=True)
        child = state.child_workflows["backend"]
        decision = {
            "execution_mode": "REVIEW_REMEDIATION",
            "role": "complex_fix",
            "model": "configured-complex-fix-model",
            "reasoning": "xhigh",
            "classification": "COMPLEX",
            "attempt": 2,
            "escalation_level": 1,
            "escalated": True,
            "previous_role": "fix",
            "finding_ids": ["REV-003"],
            "finding_fingerprints": ["sha256:abc"],
            "repository_revision": "abc123",
            "routing_input_fingerprint": "sha256:question",
            "input_fingerprint": "sha256:effective",
            "routing_reason": "REV-003 remained unresolved after 1 attempt(s) at fix",
        }
        state.child_workflows["backend"] = child.model_copy(update={"model_routing": decision})
        await store.checkpoint(state, WorkflowCheckpointBoundary.BEFORE_CODING, "backend")

        restored = await SqlAlchemyFeatureControlPlane(
            database, mock_runner=FeatureWorkflowOrchestrator()
        ).get_record("feature-state")

        restored_child = restored.state.child_workflows["backend"]
        assert restored_child.model_routing == decision
    finally:
        await database.drop_schema()
        await database.dispose()


class PlannerThatOutlastsOneQueuedAttempt:
    """Fault through a whole in-process allowance, then plan on the queue's next attempt."""

    def __init__(self, *, failures: int) -> None:
        """Count planning calls so the test can prove which attempt each one belonged to."""
        self.calls = 0
        self._remaining = failures
        self._delegate = DeterministicFeaturePlanner()

    async def plan(self, **kwargs: Any) -> Any:
        """Drop the connection the way a long maximum-effort planning call really does."""
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            diagnostic = "the model provider call failed (APIConnectionError)"
            raise LLMAdapterError(
                diagnostic, diagnostics=(diagnostic,), failure_classification="APIConnectionError"
            )
        return await self._delegate.plan(**kwargs)


@pytest.mark.asyncio
async def test_an_answered_resume_is_retried_by_the_queue_without_asking_the_author_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole AB-Feature-168 sequence, end to end, with nobody pressing resume twice.

    Its author answered the clarification questions, planning ran for twenty-three minutes,
    the provider dropped the connection, and the feature came straight back asking to be
    resumed -- three times in one morning, with two of its three queued attempts unspent
    every time.

    Two things have to hold for the retry to be worth anything. The queue must spend the
    attempt rather than close the entry `succeeded`; and the retried resume must continue
    from the checkpoint the accepted answers were written to, instead of re-submitting
    answers the first attempt already applied -- which is refused, because by then the
    feature is no longer waiting for anybody.
    """
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'answered-resume-retry.db'}")
    await database.create_schema()
    try:
        request = StartFeatureRequest.model_validate(feature_payload()).model_copy(
            update={"feature_id": "answered-resume-retry"}
        )
        # Three faults: one whole in-process allowance, so the first queued attempt is
        # genuinely spent and the second is the one that plans.
        planner = PlannerThatOutlastsOneQueuedAttempt(failures=3)
        reconnaissance = RecordingReconnaissance(
            contradicted_premises={
                "backend": [
                    {
                        "premise": "Authentication can be applied per route.",
                        "contradicted_by": "Authentication is applied once, globally.",
                        "evidence_paths": ["server/config/express.js"],
                        "question": "Rely on the global middleware, or add per-route auth?",
                    }
                ]
            }
        )
        # Checkpoints are persisted, as the control plane arranges for a live runner. Without
        # them nothing is written mid-run, so the feature is still recorded
        # `waiting_for_human` when the planner faults and the retry is handed a state that
        # never left the clarification gate. That is the one condition under which
        # re-sending the answers happens to work, and it is not the condition
        # AB-Feature-168 was in: its `feature_checkpoint_before_clone` is in the timeline.
        control_plane: list[SqlAlchemyFeatureControlPlane] = []

        async def checkpoint(
            state: FeatureWorkflowSnapshot,
            boundary: WorkflowCheckpointBoundary,
            repository_id: str | None = None,
        ) -> None:
            await control_plane[0].checkpoint(state, boundary, repository_id)

        runner = FeatureWorkflowOrchestrator(
            reconnaissance=cast(Any, reconnaissance),
            planner=cast(Any, planner),
            checkpoint_writer=checkpoint,
        )
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=runner)
        control_plane.append(store)
        credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)

        await store.start(
            request,
            idempotency_key="answered-resume-retry-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        asked = await store.get_record("answered-resume-retry")
        assert asked.state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
        technical_prd = next(
            artifact
            for artifact in reversed(asked.state.artifacts)
            if artifact.artifact_type == "technical_prd"
        )
        question_ids = [item.question_id for item in technical_prd.unresolved_questions]
        assert question_ids, "the checkout must have raised something to answer"

        await store.resume(
            "answered-resume-retry",
            answers=[
                ClarificationAnswer(question_id=item, answer="Use the global middleware.")
                for item in question_ids
            ],
            credentials=credentials,
        )
        await drain_feature_queue(store)

        recovered = await store.get_record("answered-resume-retry")
        assert recovered.state.status is FeatureWorkflowStatus.COMPLETED, (
            "the queue's own attempt must finish this without a second human resume"
        )
        assert recovered.state.failure_summary is None
        # Three faults on the first queued attempt, and the fourth call -- the second
        # attempt's -- produced the plan.
        assert planner.calls == 4
    finally:
        await database.drop_schema()
        await database.dispose()
