"""The pre-coding model calls leave journal rows, and decide nothing from them.

AB-Feature-173 spent 06:10 to 06:48 inside one reconnaissance call and the database said
nothing at all: the queue lease proved the process was alive, not that the call was, and
the stages before the child loop journaled none of their model calls. These tests pin the
fix from 41- Part A: every pre-coding model call -- product manager, per-repository
reconnaissance, clarification grounding, planner -- writes one operation row with a start,
a heartbeat, an end, and a sanitized error identity on failure.

Equally load-bearing: the rows are observability, not recovery semantics. The stages resume
from their artifacts, so a fresh call writes a fresh row and no result is ever replayed
from one. The crash tier proves the recovery half; here the assertions are about what one
run records and that recording changes nothing about what the run does.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
import structlog

import workflows.feature_workflow as feature_workflow_module
from adapters.llm_adapter import ImageInput, LLMAdapterError, LLMResponse
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from configs.settings import load_settings
from services.cancellation import MockCancellationToken
from services.execution_records import feature_executions
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.feature_runtime import LiveRepositoryReconnaissance
from state.enums import FeatureWorkflowStatus
from state.external_operations import (
    ExternalOperation,
    ExternalOperationStatus,
    ExternalOperationType,
)
from state.feature_models import FeatureWorkflowSnapshot
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from tests.test_attachment_lifecycle import _captured_logs
from tests.test_clarification_suggestions import _GLOBAL_AUTH_PREMISE, GroundingReconnaissance
from tests.test_feature_workflow import (
    ProviderFaultThenPlanPlanner,
    feature_payload,
)
from tests.test_repository_reconnaissance import build_repository, response_payload
from tools.llm_call_record import record_llm_call
from workflows.feature_workflow import (
    DeterministicFeatureProductManager,
    FeatureWorkflowOrchestrator,
)

pytestmark = pytest.mark.asyncio

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)

_PLANNING_TYPES = (
    ExternalOperationType.RUN_PRODUCT_MANAGER,
    ExternalOperationType.RUN_REPOSITORY_RECON,
    ExternalOperationType.RUN_CLARIFICATION_GROUNDING,
    ExternalOperationType.RUN_FEATURE_PLANNER,
)


async def _journal(tmp_path: Path) -> tuple[Database, ExternalOperationJournal]:
    """A durable journal of this test's own, the shape the deployment composes."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'journal.db'}")
    await database.create_schema()
    return database, ExternalOperationJournal(database)


def _factory(journal: ExternalOperationJournal) -> Any:
    """Build executors the way the live composition does: scoped to the feature they serve."""

    def build(state: FeatureWorkflowSnapshot) -> ExternalOperationExecutor:
        return ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(
                workflow_id=state.workflow_id, feature_id=state.feature_id
            ),
        )

    return build


async def _rows(
    journal: ExternalOperationJournal, feature_id: str, kind: ExternalOperationType
) -> list[ExternalOperation]:
    """This feature's journal rows of one type, in creation order."""
    return await journal.list_operations_for_feature(feature_id, operation_types=[kind])


async def test_the_product_manager_and_planner_calls_each_leave_one_completed_row(
    tmp_path: Path,
) -> None:
    """One run, one row per pre-coding call, each with stage, timestamps, and a heartbeat."""
    database, journal = await _journal(tmp_path)
    try:
        state = _initial_feature_state(
            "feature-journaled-planning", StartFeatureRequest.model_validate(feature_payload())
        )
        orchestrator = FeatureWorkflowOrchestrator(operation_executor_factory=_factory(journal))

        result = await orchestrator.start(state, credentials=CREDENTIALS)

        assert result.status is FeatureWorkflowStatus.COMPLETED
        for kind, stage in (
            (ExternalOperationType.RUN_PRODUCT_MANAGER, "product_manager"),
            (ExternalOperationType.RUN_FEATURE_PLANNER, "feature_planner"),
        ):
            rows = await _rows(journal, result.feature_id, kind)
            assert len(rows) == 1, f"expected exactly one {kind.value} row"
            row = rows[0]
            assert row.status is ExternalOperationStatus.SUCCEEDED
            assert row.safe_metadata["logical_step"] == stage
            # The observability the dark stretch had none of: a start, a liveness stamp,
            # and an end -- all real measurements, not derived ones.
            assert row.started_at is not None
            assert row.heartbeat_at is not None
            assert row.completed_at is not None
            assert row.completed_at >= row.started_at
    finally:
        await database.dispose()


async def test_the_grounding_call_is_journaled_with_how_many_questions_it_was_asked(
    tmp_path: Path,
) -> None:
    """The answer-suggestion call is a model call like the others, and leaves the same row."""
    database, journal = await _journal(tmp_path)
    try:
        state = _initial_feature_state(
            "feature-journaled-grounding", StartFeatureRequest.model_validate(feature_payload())
        )
        reconnaissance = GroundingReconnaissance(
            contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]}, answers=[]
        )
        orchestrator = FeatureWorkflowOrchestrator(
            reconnaissance=cast(Any, reconnaissance),
            operation_executor_factory=_factory(journal),
        )

        result = await orchestrator.start(state, credentials=CREDENTIALS)

        assert result.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
        rows = await _rows(
            journal, result.feature_id, ExternalOperationType.RUN_CLARIFICATION_GROUNDING
        )
        assert len(rows) == 1
        assert rows[0].status is ExternalOperationStatus.SUCCEEDED
        assert rows[0].safe_metadata["logical_step"] == "clarification_grounding"
        assert rows[0].safe_metadata["question_count"] == len(reconnaissance.asked)
        assert reconnaissance.asked, "the grounding call must actually have been made"
    finally:
        await database.dispose()


async def test_a_deterministic_planning_failure_writes_one_terminal_row_with_its_identity(
    tmp_path: Path,
) -> None:
    """A 400 fails once, and its single row carries the fault identity -- never provider text."""

    class DeterministicallyRefusedProductManager:
        calls = 0

        async def create_technical_prd(self, **_kwargs: Any) -> Any:
            type(self).calls += 1
            msg = "the model provider call failed (BadRequestError)"
            raise LLMAdapterError(msg, diagnostics=(msg,), failure_classification="BadRequestError")

    database, journal = await _journal(tmp_path)
    try:
        state = _initial_feature_state(
            "feature-journaled-400", StartFeatureRequest.model_validate(feature_payload())
        )
        orchestrator = FeatureWorkflowOrchestrator(
            product_manager=cast(Any, DeterministicallyRefusedProductManager()),
            operation_executor_factory=_factory(journal),
        )

        with pytest.raises(LLMAdapterError):
            await orchestrator.start(state, credentials=CREDENTIALS)

        assert DeterministicallyRefusedProductManager.calls == 1, (
            "a deterministic provider answer must not earn the call again"
        )
        rows = await _rows(journal, state.feature_id, ExternalOperationType.RUN_PRODUCT_MANAGER)
        assert len(rows) == 1
        row = rows[0]
        assert row.status is ExternalOperationStatus.FAILED_TERMINAL
        assert row.error_code == "run_product_manager_failed"
        assert row.error_message is not None
        assert "LLMAdapterError/BadRequestError" in row.error_message
    finally:
        await database.dispose()


async def test_each_retried_planning_call_writes_its_own_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three calls, three rows: what the feature spent on faults is countable afterwards."""
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)
    database, journal = await _journal(tmp_path)
    try:
        state = _initial_feature_state(
            "feature-journaled-retries", StartFeatureRequest.model_validate(feature_payload())
        )
        planner = ProviderFaultThenPlanPlanner(faults=2)
        orchestrator = FeatureWorkflowOrchestrator(
            planner=cast(Any, planner), operation_executor_factory=_factory(journal)
        )

        result = await orchestrator.start(state, credentials=CREDENTIALS)

        assert result.status is FeatureWorkflowStatus.COMPLETED
        assert planner.calls == 3
        rows = await _rows(journal, result.feature_id, ExternalOperationType.RUN_FEATURE_PLANNER)
        assert [row.status for row in rows] == [
            ExternalOperationStatus.FAILED_TERMINAL,
            ExternalOperationStatus.FAILED_TERMINAL,
            ExternalOperationStatus.SUCCEEDED,
        ]
    finally:
        await database.dispose()


async def test_the_integration_review_call_leaves_one_completed_row(tmp_path: Path) -> None:
    """The cross-repository seam call is journaled exactly like the other pre-coding calls.

    AB-Feature-174 hung inside this call with no journal row and no heartbeat -- this pins the
    fix: a completed run of the default (mock-approving) composition leaves one
    ``run_integration_review`` row with a start, a heartbeat, and an end.
    """
    database, journal = await _journal(tmp_path)
    try:
        state = _initial_feature_state(
            "feature-journaled-integration-review",
            StartFeatureRequest.model_validate(feature_payload()),
        )
        orchestrator = FeatureWorkflowOrchestrator(operation_executor_factory=_factory(journal))

        result = await orchestrator.start(state, credentials=CREDENTIALS)

        assert result.status is FeatureWorkflowStatus.COMPLETED
        rows = await _rows(journal, result.feature_id, ExternalOperationType.RUN_INTEGRATION_REVIEW)
        assert len(rows) == 1
        row = rows[0]
        assert row.status is ExternalOperationStatus.SUCCEEDED
        assert row.safe_metadata["logical_step"] == "integration_review"
        assert row.started_at is not None
        assert row.heartbeat_at is not None
        assert row.completed_at is not None
        assert row.completed_at >= row.started_at
    finally:
        await database.dispose()


async def test_a_failed_integration_review_call_still_leaves_its_row(tmp_path: Path) -> None:
    """A deterministic refusal fails once, and its row carries the fault identity, not raw text."""

    class _RefusedIntegrationReviewer:
        calls = 0

        async def review(self, **_kwargs: Any) -> Any:
            type(self).calls += 1
            msg = "the model provider call failed (BadRequestError)"
            raise LLMAdapterError(msg, diagnostics=(msg,), failure_classification="BadRequestError")

    database, journal = await _journal(tmp_path)
    try:
        state = _initial_feature_state(
            "feature-journaled-integration-review-failure",
            StartFeatureRequest.model_validate(feature_payload()),
        )
        orchestrator = FeatureWorkflowOrchestrator(
            integration_reviewer=cast(Any, _RefusedIntegrationReviewer()),
            operation_executor_factory=_factory(journal),
        )

        with pytest.raises(LLMAdapterError):
            await orchestrator.start(state, credentials=CREDENTIALS)

        assert _RefusedIntegrationReviewer.calls == 1, (
            "a deterministic provider answer must not earn the call again"
        )
        rows = await _rows(journal, state.feature_id, ExternalOperationType.RUN_INTEGRATION_REVIEW)
        assert len(rows) == 1
        row = rows[0]
        assert row.status is ExternalOperationStatus.FAILED_TERMINAL
        assert row.error_code == "run_integration_review_failed"
        assert row.error_message is not None
        assert "LLMAdapterError/BadRequestError" in row.error_message
    finally:
        await database.dispose()


async def test_the_integration_review_stage_logs_its_start_completion_and_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live log file shows the stage running and its outcome, not only the DB journal.

    AB-Feature-174's hang and AB-Feature-171's silent early-exit were both invisible from the
    log file alone. This pins the part of the fix that writes to the log directly: a start line,
    a completion line, and the review's verdict.

    A fresh logger is substituted for the module's own before capturing: ``structlog``'s
    production configuration (``main.configure_structlog``, which every test process loads as
    an import side effect) caches a logger's resolved processors the first time it is ever
    called, and earlier tests in this file are that first call for
    ``workflows.feature_workflow``'s own logger -- so capturing through the module's already-
    cached instance would see nothing. A never-used proxy has no cached resolution to ignore.
    """
    monkeypatch.setattr(feature_workflow_module, "_LOGGER", structlog.get_logger("test.capture"))
    database, journal = await _journal(tmp_path)
    try:
        state = _initial_feature_state(
            "feature-journaled-integration-review-logs",
            StartFeatureRequest.model_validate(feature_payload()),
        )
        orchestrator = FeatureWorkflowOrchestrator(operation_executor_factory=_factory(journal))

        with _captured_logs() as entries:
            result = await orchestrator.start(state, credentials=CREDENTIALS)

        assert result.status is FeatureWorkflowStatus.COMPLETED
        by_event = {
            (entry["event"], entry.get("stage")): entry
            for entry in entries
            if entry["event"].startswith("feature_stage_")
            or entry["event"] == ("integration_review_verdict")
        }
        started = by_event[("feature_stage_started", "integration_review")]
        assert started["feature_id"] == result.feature_id
        completed = by_event[("feature_stage_completed", "integration_review")]
        assert completed["feature_id"] == result.feature_id
        verdict = by_event[("integration_review_verdict", None)]
        assert verdict["feature_id"] == result.feature_id
        assert verdict["review_status"] == "approved"
    finally:
        await database.dispose()


def _pushed_origin(tmp_path: Path) -> Path:
    """A real bare remote holding the reconnaissance fixture repository."""
    source = tmp_path / "recon-source"
    source.mkdir()
    build_repository(source)
    subprocess.run(("git", "add", "-A"), cwd=source, check=True, capture_output=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.email=recon@example.com",
            "-c",
            "user.name=Recon Suite",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "fixture",
        ),
        cwd=source,
        check=True,
        capture_output=True,
    )
    remote = tmp_path / "recon-remote.git"
    subprocess.run(
        ("git", "init", "--bare", "--initial-branch=main", str(remote)),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "push", str(remote), "HEAD:main"), cwd=source, check=True, capture_output=True
    )
    return remote


class _ScriptedReconClient:
    """Return one queued payload per model call, or raise what the script says to raise."""

    def __init__(self, *payloads: dict[str, Any] | Exception) -> None:
        self._payloads = list(payloads)
        self.calls = 0

    @property
    def vision_capable(self) -> bool:
        """No image reaches this double, so the boundary answers False."""
        return False

    async def respond(
        self,
        *,
        instructions: str,
        input_text: str,
        images: Sequence[ImageInput] = (),
    ) -> LLMResponse:
        del instructions, input_text
        self.calls += 1
        head = self._payloads.pop(0)
        if isinstance(head, Exception):
            raise head
        # The adapter contract: every real client records its response's model facts at this
        # seam, which is how the journal row wrapping the call can name what answered it.
        record_llm_call(model="recon-test-model", provider="test-provider", reasoning_effort="high")
        return LLMResponse(
            response_id=f"recon-response-{self.calls}",
            model="recon-test-model",
            output_text=json.dumps(head),
            input_tokens=None,
            output_tokens=None,
            provider="test-provider",
            reasoning_effort="high",
        )


def _single_repository_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> FeatureWorkflowSnapshot:
    """One backend repository whose token-free clone URL resolves to a real local origin.

    The Git adapter deliberately accepts only the askpass credential contract in its
    environment, so the URL is mapped at the module seam the live path already goes
    through, not by injecting Git configuration.
    """
    payload = feature_payload()
    payload["feature_id"] = "feature-journaled-recon"
    payload["repositories"] = [cast(list[Any], payload["repositories"])[0]]
    state = _initial_feature_state(
        "feature-journaled-recon", StartFeatureRequest.model_validate(payload)
    )
    remote = _pushed_origin(tmp_path)
    import services.feature_runtime as feature_runtime_module

    monkeypatch.setattr(feature_runtime_module, "_git_source_url", lambda _url: str(remote))
    return state


async def test_the_live_reconnaissance_call_is_journaled_per_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The call whose 37 invisible minutes started this task now has a row of its own."""
    database, journal = await _journal(tmp_path)
    try:
        state = _single_repository_state(tmp_path, monkeypatch)
        client = _ScriptedReconClient(response_payload())
        reconnaissance = LiveRepositoryReconnaissance(
            settings=load_settings(workspace_root=tmp_path / "workspaces"),
            git_environment={},
            recon_client=cast(Any, client),
            journal=journal,
            cancellation_token=MockCancellationToken(),
        )

        report = await reconnaissance.inspect(
            feature=state,
            repositories=state.repository_specs,
            technical_prd=_technical_prd(state),
            credentials=CREDENTIALS,
        )

        assert len(report.artifacts) == 1 and client.calls == 1
        assert report.blind == []
        rows = await _rows(journal, state.feature_id, ExternalOperationType.RUN_REPOSITORY_RECON)
        assert len(rows) == 1
        row = rows[0]
        assert row.status is ExternalOperationStatus.SUCCEEDED
        assert row.repository_id == "backend"
        assert row.safe_metadata["logical_step"] == "repository_reconnaissance"
        assert row.started_at is not None and row.heartbeat_at is not None
        assert row.completed_at is not None
        # The row names the model that answered it, from the client's own record -- never
        # from configuration. Same block shape an artifact carries, so one reader serves both.
        assert row.result_payload is not None
        assert row.result_payload["execution"] == {
            "model": "recon-test-model",
            "provider": "test-provider",
            "reasoning_effort": "high",
        }
        # And the executions read serves it: the planning_call record resolves its model.
        records = feature_executions(state, operations=rows)
        recon_calls = [item for item in records if item.execution_id.startswith("planning_call:")]
        assert len(recon_calls) == 1
        assert recon_calls[0].model == "recon-test-model"
        assert recon_calls[0].provider == "test-provider"
        assert recon_calls[0].reasoning_effort == "high"
        assert recon_calls[0].model_resolved is True
    finally:
        await database.dispose()


async def test_reconnaissance_succeeds_when_workspace_root_is_reached_through_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlinked `workspace_root` (a `/tmp` that is really `/private/tmp`) must not blind it.

    `_require_workspace_child` resolves the root it is given but not the candidate path it is
    checking, so a candidate built by joining an *unresolved* `workspace_root` never appears
    in its own resolved root's parents once a symlink sits between them -- failing for every
    repository, on every feature, regardless of what that repository contains. That is what
    this deployment's `_inspect_one` did before it resolved its own candidate first.
    """
    database, journal = await _journal(tmp_path)
    try:
        real_root = tmp_path / "real-workspaces"
        real_root.mkdir()
        linked_root = tmp_path / "linked-workspaces"
        linked_root.symlink_to(real_root, target_is_directory=True)

        state = _single_repository_state(tmp_path, monkeypatch)
        client = _ScriptedReconClient(response_payload())
        reconnaissance = LiveRepositoryReconnaissance(
            settings=load_settings(workspace_root=linked_root),
            git_environment={},
            recon_client=cast(Any, client),
            journal=journal,
            cancellation_token=MockCancellationToken(),
        )

        report = await reconnaissance.inspect(
            feature=state,
            repositories=state.repository_specs,
            technical_prd=_technical_prd(state),
            credentials=CREDENTIALS,
        )

        assert report.blind == []
        assert len(report.artifacts) == 1
    finally:
        await database.dispose()


async def test_a_failed_reconnaissance_call_fails_soft_and_still_leaves_its_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fail-soft is untouched -- and the row is why nobody needs `docker logs` anymore."""
    database, journal = await _journal(tmp_path)
    try:
        state = _single_repository_state(tmp_path, monkeypatch)
        fault = LLMAdapterError(
            "the model provider call failed (BadRequestError)",
            diagnostics=("the model provider call failed (BadRequestError)",),
            failure_classification="BadRequestError",
        )
        reconnaissance = LiveRepositoryReconnaissance(
            settings=load_settings(workspace_root=tmp_path / "workspaces"),
            git_environment={},
            recon_client=cast(Any, _ScriptedReconClient(fault, fault)),
            journal=journal,
            cancellation_token=MockCancellationToken(),
        )

        report = await reconnaissance.inspect(
            feature=state,
            repositories=state.repository_specs,
            technical_prd=_technical_prd(state),
            credentials=CREDENTIALS,
        )

        assert report.artifacts == [], "one repository's fault is its own; the feature continues"
        assert [item.repository_id for item in report.blind] == ["backend"]
        assert report.blind[0].error_type == "LLMAdapterError"
        rows = await _rows(journal, state.feature_id, ExternalOperationType.RUN_REPOSITORY_RECON)
        assert len(rows) == 1
        assert rows[0].status is ExternalOperationStatus.FAILED_TERMINAL
        assert rows[0].error_code == "run_repository_recon_failed"
        assert rows[0].error_message is not None
        assert "LLMAdapterError/BadRequestError" in rows[0].error_message
    finally:
        await database.dispose()


def _technical_prd(state: FeatureWorkflowSnapshot) -> Any:
    """The PRD the reconnaissance stage reads the checkout against."""
    from tests.test_repository_reconnaissance import technical_prd

    prd = technical_prd()
    return prd.model_copy(update={"workflow_id": state.feature_id})


async def test_the_executions_read_renders_the_journal_rows(tmp_path: Path) -> None:
    """`GET /features/{id}/executions` shows the calls, running ones included."""
    database, journal = await _journal(tmp_path)
    try:
        state = _initial_feature_state(
            "feature-journaled-read", StartFeatureRequest.model_validate(feature_payload())
        )
        orchestrator = FeatureWorkflowOrchestrator(operation_executor_factory=_factory(journal))
        result = await orchestrator.start(state, credentials=CREDENTIALS)

        operations = await journal.list_operations_for_feature(result.feature_id)
        records = feature_executions(result, operations=operations)

        planning_calls = [
            item for item in records if item.execution_id.startswith("planning_call:")
        ]
        # Product manager, planner, and -- since this deterministic composition's mock
        # reviewers approve every repository -- the integration review that follows them:
        # one record per journaled pre-coding-shaped call.
        assert len(planning_calls) == 3, "one record per journaled pre-coding-shaped call"
        by_handler = {item.handler: item for item in planning_calls}
        manager = by_handler["Product manager"]
        assert manager.status.value == "completed"
        assert manager.started_at is not None and manager.heartbeat_at is not None
        assert manager.completed_at is not None
        assert manager.duration_seconds is not None and manager.duration_seconds >= 0.0
        assert manager.agent_type == "product_manager"
        # The deterministic composition reaches no provider and records no model, and the
        # record says exactly that rather than reconstructing one from configuration.
        assert manager.model is None
        assert manager.model_resolved is False
        planner = by_handler["Technical planner"]
        assert planner.failure_classification is None
        integration_reviewer = by_handler["Integration reviewer"]
        assert integration_reviewer.status.value == "completed"
        assert integration_reviewer.started_at is not None
        assert integration_reviewer.heartbeat_at is not None
        # The artifact-derived stage rows are untouched beside them.
        assert any(item.execution_id.startswith("technical_prd:") for item in records)
    finally:
        await database.dispose()


class _RecordingProductManager(DeterministicFeatureProductManager):
    """The deterministic mock, plus the one thing a live adapter does: record its model."""

    async def create_technical_prd(
        self, *, feature_id: str, prd: Any, design_snapshot: Any = None
    ) -> Any:
        record_llm_call(model="pm-test-model", provider="test-provider", reasoning_effort="max")
        return await super().create_technical_prd(
            feature_id=feature_id, prd=prd, design_snapshot=design_snapshot
        )


async def test_a_journaled_planning_call_persists_the_model_its_client_recorded(
    tmp_path: Path,
) -> None:
    """The row carries what the client recorded; a call that recorded nothing carries nothing.

    The second half is as load-bearing as the first: the planner here reaches no provider, and
    its row must not inherit the product manager's model from a stale record. The reset before
    every journaled call, and the task-local variable, are what these assertions pin.
    """
    database, journal = await _journal(tmp_path)
    try:
        state = _initial_feature_state(
            "feature-recorded-model", StartFeatureRequest.model_validate(feature_payload())
        )
        orchestrator = FeatureWorkflowOrchestrator(
            product_manager=_RecordingProductManager(),
            operation_executor_factory=_factory(journal),
        )
        result = await orchestrator.start(state, credentials=CREDENTIALS)
        assert result.status is FeatureWorkflowStatus.COMPLETED

        pm_rows = await _rows(journal, result.feature_id, ExternalOperationType.RUN_PRODUCT_MANAGER)
        assert len(pm_rows) == 1
        assert pm_rows[0].result_payload is not None
        assert pm_rows[0].result_payload["execution"] == {
            "model": "pm-test-model",
            "provider": "test-provider",
            "reasoning_effort": "max",
        }
        planner_rows = await _rows(
            journal, result.feature_id, ExternalOperationType.RUN_FEATURE_PLANNER
        )
        assert len(planner_rows) == 1
        # The journal normalizes an absent payload to an empty one; what matters is that no
        # execution block was written -- not the product manager's, not anybody's.
        assert "execution" not in (planner_rows[0].result_payload or {})

        operations = await journal.list_operations_for_feature(result.feature_id)
        records = feature_executions(result, operations=operations)
        by_handler = {
            item.handler: item for item in records if item.execution_id.startswith("planning_call:")
        }
        assert by_handler["Product manager"].model == "pm-test-model"
        assert by_handler["Product manager"].provider == "test-provider"
        assert by_handler["Product manager"].model_resolved is True
        assert by_handler["Technical planner"].model is None
        assert by_handler["Technical planner"].model_resolved is False
    finally:
        await database.dispose()


async def test_journaling_changes_nothing_about_what_the_feature_does(tmp_path: Path) -> None:
    """The same feature, with and without the rows, ends in the same durable place."""
    database, journal = await _journal(tmp_path)
    try:
        request = StartFeatureRequest.model_validate(feature_payload())
        journaled = await FeatureWorkflowOrchestrator(
            operation_executor_factory=_factory(journal)
        ).start(_initial_feature_state("feature-with-rows", request), credentials=CREDENTIALS)
        unjournaled = await FeatureWorkflowOrchestrator().start(
            _initial_feature_state("feature-sans-rows", request), credentials=CREDENTIALS
        )

        def visible(state: FeatureWorkflowSnapshot) -> tuple[Any, ...]:
            # Sorted, because parallel workstreams append their artifacts in whichever
            # order the loop scheduled them -- that varies between any two runs and has
            # nothing to do with journaling.
            return (
                state.status,
                sorted(item.artifact_id for item in state.artifacts),
                {key: value.status for key, value in state.child_workflows.items()},
                state.checkpoint_boundaries,
                state.transition_reason,
            )

        assert visible(journaled) == visible(unjournaled)
    finally:
        await database.dispose()


async def test_the_design_fetch_leaves_one_row_and_a_citation_free_feature_leaves_none(
    tmp_path: Path,
) -> None:
    """The design fetch joins the pre-coding block, and only for a feature that cited one.

    Both halves matter. A hung fetch has to be a row with a widening
    `started_at`->`heartbeat_at` gap rather than silence -- which is the whole reason that
    block exists -- and Safety rule 3 says a feature that cites no design must produce no new
    operation row at all, which is asserted here against the real journal rather than argued.
    """
    database, journal = await _journal(tmp_path)
    try:
        cited = feature_payload()
        cited["feature_id"] = "feature-journaled-design"
        cited["prd"] = {
            **cast("dict[str, Any]", cited["prd"]),
            "design_references": [
                {
                    "url": (
                        "https://www.figma.com/design/VGULlnz44R0Ooe4FZKDxlhh4/"
                        "Untitled?node-id=10-11"
                    ),
                    "label": "the clock",
                }
            ],
        }
        orchestrator = FeatureWorkflowOrchestrator(operation_executor_factory=_factory(journal))

        with_design = await orchestrator.start(
            _initial_feature_state(
                "feature-journaled-design", StartFeatureRequest.model_validate(cited)
            ),
            credentials=CREDENTIALS,
        )
        plain = await orchestrator.start(
            _initial_feature_state(
                "feature-journaled-plain",
                StartFeatureRequest.model_validate(
                    {**feature_payload(), "feature_id": "feature-journaled-plain"}
                ),
            ),
            credentials=CREDENTIALS,
        )

        rows = await _rows(
            journal, with_design.feature_id, ExternalOperationType.FETCH_DESIGN_REFERENCE
        )
        # The snapshot resolution is one row, once, before the request is analysed. The design
        # reads that happen *per repository* as each workstream starts carry the same operation
        # type -- same provider, same outbound call -- and are told apart by `logical_step`.
        snapshot_rows = [
            item for item in rows if item.safe_metadata["logical_step"] == "design_snapshot"
        ]
        assert len(snapshot_rows) == 1
        row = snapshot_rows[0]
        assert row.status is ExternalOperationStatus.SUCCEEDED
        # One detail row per repository whose workstream was assigned a frame, and each names
        # the repository it read for -- so a hung read says which workstream is waiting.
        detail_rows = [
            item for item in rows if item.safe_metadata["logical_step"] == "design_detail"
        ]
        assert detail_rows, "a workstream assigned a frame journals its own design read"
        assert all(item.status is ExternalOperationStatus.SUCCEEDED for item in detail_rows)
        read_for = [str(item.safe_metadata["repository_id"]) for item in detail_rows]
        assert all(read_for)
        assert all(int(cast("int", item.safe_metadata["frames"])) >= 1 for item in detail_rows)
        assert len(set(read_for)) == len(detail_rows), (
            "one design read per repository, not one per attempt"
        )
        assert "figd" not in json.dumps([item.safe_metadata for item in detail_rows])
        assert row.started_at is not None
        assert row.heartbeat_at is not None
        assert row.completed_at is not None
        # File keys and a count, never a URL somebody pasted and never a credential.
        assert row.safe_metadata["file_keys"] == ["VGULlnz44R0Ooe4FZKDxlhh4"]
        assert row.safe_metadata["citations"] == 1
        assert "figd" not in json.dumps(row.safe_metadata)
        assert "figma.com" not in json.dumps(row.safe_metadata)

        assert (
            await _rows(journal, plain.feature_id, ExternalOperationType.FETCH_DESIGN_REFERENCE)
            == []
        ), "a feature that cited no design journals nothing new"
        # And the rows a citation-free feature *does* write are the ones it always wrote --
        # the two a deterministic composition reaches, since it has no reconnaissance and no
        # grounding call to make.
        for kind in (
            ExternalOperationType.RUN_PRODUCT_MANAGER,
            ExternalOperationType.RUN_FEATURE_PLANNER,
        ):
            assert len(await _rows(journal, plain.feature_id, kind)) == 1, kind.value
            assert len(await _rows(journal, with_design.feature_id, kind)) == 1, kind.value
    finally:
        await database.dispose()


async def test_the_design_fetch_renders_before_the_product_manager_in_the_read_model(
    tmp_path: Path,
) -> None:
    """The stage vocabulary says where it sits, and it sits before the request is analysed.

    A citation-free feature renders exactly the stages it always did, because the record is
    derived from the journal row and there is no row.
    """
    database, journal = await _journal(tmp_path)
    try:
        cited = feature_payload()
        cited["feature_id"] = "feature-design-stage"
        cited["prd"] = {
            **cast("dict[str, Any]", cited["prd"]),
            "design_references": [
                {
                    "url": (
                        "https://www.figma.com/design/VGULlnz44R0Ooe4FZKDxlhh4/"
                        "Untitled?node-id=10-11"
                    )
                }
            ],
        }
        orchestrator = FeatureWorkflowOrchestrator(operation_executor_factory=_factory(journal))
        result = await orchestrator.start(
            _initial_feature_state(
                "feature-design-stage", StartFeatureRequest.model_validate(cited)
            ),
            credentials=CREDENTIALS,
        )

        rows = await journal.list_operations_for_feature(result.feature_id)
        records = feature_executions(result, operations=rows)
        design = [item for item in records if item.to_stage.value == "design_snapshot"]

        assert len(design) == 1, (
            "one design_snapshot node, however many per-repository design reads happened: the "
            "detail reads share the operation type and are excluded from this read model"
        )
        assert design[0].from_stage.value == "request"
        assert design[0].handler == "Design resolver"
        assert design[0].agent_type == "design_snapshot"
        # And the per-repository reads are journaled but deliberately not drawn here -- they
        # are accounted for by their attempt artifacts, like every other child-loop operation.
        assert [
            item
            for item in rows
            if (item.safe_metadata or {}).get("logical_step") == "design_detail"
        ], "the detail reads happened"
        assert not [item for item in records if item.agent_type == "design_detail"]
    finally:
        await database.dispose()
