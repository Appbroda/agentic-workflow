"""Focused durable-operation, recovery, cancellation, and idempotency tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import update

from adapters.git_adapter import GitAdapterError
from adapters.github_adapter import MockGitHubService
from adapters.interruptible_git import (
    InterruptibleGitService,
    reviewed_content_fingerprint,
)
from adapters.llm_adapter import (
    ImageInput,
    LLMAdapterError,
    LLMResponse,
    MockCodingExecutor,
    ResponsesCodingExecutor,
)
from agents.engineer.publisher import ApprovedChangePublisher
from agents.shared.contracts import create_artifact
from artifacts.schemas import CodeCompletionArtifact, ReviewArtifact
from services.cancellation import CancellationRequested, MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.journaled_github import JournaledGitHubService
from services.process_runner import AsyncioProcessRunner, ProcessResult
from services.recovery_service import RecoveryService
from state.external_operations import ExternalOperationStatus, ExternalOperationType
from state.models import WorkspaceDescriptor, create_initial_agent_state
from storage.db import Database
from storage.external_operation_store import (
    ExternalOperationError,
    ExternalOperationJournal,
    OperationInProgressError,
    OperationLeaseLostError,
    OperationResult,
    OperationTransitionConflictError,
    deterministic_operation_key,
    fingerprint_operation_input,
)
from storage.models import ExternalOperationModel


@pytest.mark.asyncio
async def test_operation_journal_commits_intent_results_and_append_only_events(
    tmp_path: Path,
) -> None:
    """A duplicate deterministic key reuses the prior successful external operation."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'journal.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    first = await journal.create_operation(
        workflow_id="workflow-journal",
        feature_id=None,
        child_workflow_id=None,
        repository_id="platform",
        operation_type=ExternalOperationType.CREATE_PULL_REQUEST,
        idempotency_key="workflow-journal:platform:create-pr:one",
        input_fingerprint="fingerprint",
        safe_metadata={"repository": "example/platform"},
    )
    replay = await journal.create_operation(
        workflow_id="workflow-journal",
        feature_id=None,
        child_workflow_id=None,
        repository_id="platform",
        operation_type=ExternalOperationType.CREATE_PULL_REQUEST,
        idempotency_key="workflow-journal:platform:create-pr:one",
        input_fingerprint="fingerprint",
        safe_metadata={"repository": "example/platform"},
    )
    await journal.transition_status(
        first.operation_id, ExternalOperationStatus.STARTING, event_type="operation_starting"
    )
    await journal.record_attempt(first.operation_id, provider="github")
    completed = await journal.record_result(
        first.operation_id,
        external_reference="https://github.example/pull/1",
        result_payload={"number": 1},
    )

    assert replay.operation_id == first.operation_id
    assert completed.status is ExternalOperationStatus.SUCCEEDED
    assert [
        event.new_status for event in await journal.events_for_operation(first.operation_id)
    ] == [
        ExternalOperationStatus.PENDING,
        ExternalOperationStatus.STARTING,
        ExternalOperationStatus.RUNNING,
        ExternalOperationStatus.SUCCEEDED,
    ]
    with pytest.raises(ExternalOperationError, match="credential-like"):
        await journal.create_operation(
            workflow_id="workflow-journal",
            feature_id=None,
            child_workflow_id=None,
            repository_id=None,
            operation_type=ExternalOperationType.RUN_TESTS,
            idempotency_key="unsafe",
            input_fingerprint="unsafe",
            safe_metadata={"api_token": "must-not-persist"},
        )


@pytest.mark.asyncio
async def test_coding_replay_rejects_bytes_written_by_a_later_operation(tmp_path: Path) -> None:
    """A path match cannot attribute operation B's content to completed operation A."""

    class StaticCodingClient:
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
            return LLMResponse(
                response_id="coding-a",
                model="test-model",
                output_text=json.dumps(
                    {
                        "summary": "Write version A.",
                        "files": [{"path": "src/value.py", "content": "VALUE = 'A'\n"}],
                    }
                ),
                input_tokens=1,
                output_tokens=1,
            )

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'coding-replay.db'}")
    await database.create_schema()
    executor = ExternalOperationExecutor(
        journal=ExternalOperationJournal(database),
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(workflow_id="coding-replay"),
    )
    coding = ResponsesCodingExecutor(StaticCodingClient())
    try:
        await coding.execute(
            workspace_root=tmp_path,
            instructions="Write the value.",
            input_text="version A",
            operation_executor=executor,
        )
        (tmp_path / "src" / "value.py").write_text("VALUE = 'B'\n", encoding="utf-8")

        with pytest.raises(LLMAdapterError, match="persisted fingerprints"):
            await coding.execute(
                workspace_root=tmp_path,
                instructions="Write the value.",
                input_text="version A",
                operation_executor=executor,
            )
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_validation_journal_marks_prior_revision_superseded_and_selects_current_only(
    tmp_path: Path,
) -> None:
    """A later checkout revision cannot reuse earlier validation evidence."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'validation-journal.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    first = await journal.create_operation(
        workflow_id="feature-validation",
        feature_id="feature-validation",
        child_workflow_id="feature-validation:backend",
        repository_id="backend",
        operation_type=ExternalOperationType.RUN_LINTER,
        idempotency_key="validation:backend:lint:revision-a",
        input_fingerprint="input-a",
        repository_revision="revision-a",
        command_fingerprint="lint-command",
    )
    await journal.transition_status(
        first.operation_id, ExternalOperationStatus.STARTING, event_type="operation_starting"
    )
    await journal.record_attempt(first.operation_id)
    await journal.record_result(
        first.operation_id, external_reference=None, result_payload={"status": "failed"}
    )
    second = await journal.create_operation(
        workflow_id="feature-validation",
        feature_id="feature-validation",
        child_workflow_id="feature-validation:backend",
        repository_id="backend",
        operation_type=ExternalOperationType.RUN_LINTER,
        idempotency_key="validation:backend:lint:revision-b",
        input_fingerprint="input-b",
        repository_revision="revision-b",
        command_fingerprint="lint-command",
    )

    superseded = await journal.get(first.operation_id)
    assert not superseded.is_current
    assert superseded.superseded_by_operation_id == second.operation_id
    assert (
        await journal.get_current_validation_results(
            workflow_id="feature-validation",
            repository_id="backend",
            repository_revision="revision-a",
        )
        == []
    )

    await journal.transition_status(
        second.operation_id, ExternalOperationStatus.STARTING, event_type="operation_starting"
    )
    await journal.record_attempt(second.operation_id)
    await journal.record_result(
        second.operation_id, external_reference=None, result_payload={"status": "passed"}
    )
    current = await journal.get_current_validation_results(
        workflow_id="feature-validation", repository_id="backend", repository_revision="revision-b"
    )
    assert [item.operation_id for item in current] == [second.operation_id]
    await database.dispose()


@pytest.mark.asyncio
async def test_a_stale_remote_operation_becomes_unknown_without_a_blind_retry(
    tmp_path: Path,
) -> None:
    """A push may have reached the remote, so it stays unreconciled and blocks readiness."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'recovery.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    operation = await journal.create_operation(
        workflow_id="workflow-recovery",
        feature_id=None,
        child_workflow_id=None,
        repository_id=None,
        operation_type=ExternalOperationType.PUSH_BRANCH,
        idempotency_key="workflow-recovery:push",
        input_fingerprint="push",
    )
    await journal.transition_status(
        operation.operation_id, ExternalOperationStatus.STARTING, event_type="operation_starting"
    )
    await journal.record_attempt(operation.operation_id, provider="git")
    await asyncio.sleep(0.01)

    summary = await RecoveryService(journal).recover_incomplete_operations()
    recovered = await journal.get(operation.operation_id)

    assert summary.has_unresolved_critical_operations
    assert recovered.status is ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE
    await database.dispose()


@pytest.mark.asyncio
async def test_an_interrupted_workspace_operation_does_not_block_the_next_feature(
    tmp_path: Path,
) -> None:
    """An interrupted local step must not leave the runtime refusing every later request.

    Recovery had no rule for most operation types, so anything that was not a clone or a
    commit was parked as UNKNOWN. Readiness counts unknown operations, so one interrupted
    coding call or test run took the whole service down until the row was edited by hand.
    Nothing about these operations lives outside a workspace the platform rebuilds.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'workspace-recovery.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    for operation_type in (
        ExternalOperationType.RUN_CODING_EXECUTOR,
        ExternalOperationType.WRITE_FILE_CHANGES,
        ExternalOperationType.RUN_TESTS,
    ):
        operation = await journal.create_operation(
            workflow_id=f"workflow-{operation_type.value}",
            feature_id=None,
            child_workflow_id=None,
            repository_id=None,
            operation_type=operation_type,
            idempotency_key=f"workflow-{operation_type.value}:key",
            input_fingerprint=operation_type.value,
        )
        await journal.transition_status(
            operation.operation_id,
            ExternalOperationStatus.STARTING,
            event_type="operation_starting",
        )
        await journal.record_attempt(operation.operation_id, provider="platform")
    await asyncio.sleep(0.01)

    summary = await RecoveryService(journal).recover_incomplete_operations()

    assert summary.scanned == 3
    assert not summary.has_unresolved_critical_operations
    # The readiness gate reads this count, so a later feature is accepted.
    assert await journal.unresolved_critical_count() == 0
    await database.dispose()


@pytest.mark.asyncio
async def test_an_interrupted_local_operation_with_budget_is_reclaimed_once(
    tmp_path: Path,
) -> None:
    """A dead retry claim remains replayable under its stable idempotency key."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'local-retry-recovery.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    safe_input = {"input": "stable-coding-input"}
    fingerprint = fingerprint_operation_input(safe_input)
    logical_step = "stable-coding-step"
    operation = await journal.create_operation(
        workflow_id="workflow-local-retry",
        feature_id="feature-local-retry",
        child_workflow_id="feature-local-retry:backend",
        repository_id="backend",
        operation_type=ExternalOperationType.RUN_CODING_EXECUTOR,
        idempotency_key=deterministic_operation_key(
            workflow_id="workflow-local-retry",
            feature_id="feature-local-retry",
            child_workflow_id="feature-local-retry:backend",
            repository_id="backend",
            operation_type=ExternalOperationType.RUN_CODING_EXECUTOR,
            logical_step=logical_step,
            input_fingerprint=fingerprint,
        ),
        input_fingerprint=fingerprint,
        max_attempts=3,
    )
    await journal.claim_operation(operation.operation_id)
    await asyncio.sleep(0.01)

    summary = await RecoveryService(journal).recover_incomplete_operations()
    retryable = await journal.get(operation.operation_id)
    calls = 0

    async def action() -> tuple[str, OperationResult]:
        nonlocal calls
        calls += 1
        return "recovered", OperationResult(payload={"result": "recovered"})

    executor = ExternalOperationExecutor(
        journal=journal,
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(
            workflow_id="workflow-local-retry",
            feature_id="feature-local-retry",
            child_workflow_id="feature-local-retry:backend",
            repository_id="backend",
        ),
    )
    replay = await executor.run(
        operation_type=ExternalOperationType.RUN_CODING_EXECUTOR,
        logical_step=logical_step,
        safe_input=safe_input,
        action=action,
        max_attempts=3,
    )

    assert summary.reconciled == 1
    assert retryable.status is ExternalOperationStatus.FAILED_RETRYABLE
    assert replay.value == "recovered"
    assert calls == 1
    assert replay.operation.operation_id == operation.operation_id
    assert (await journal.get(replay.operation.operation_id)).attempt == 2
    await database.dispose()


@pytest.mark.asyncio
async def test_startup_recovery_does_not_take_over_a_fresh_live_operation(tmp_path: Path) -> None:
    """A second process cannot mark another worker's current heartbeat as uncertain."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'fresh-recovery.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=60)
    operation = await journal.create_operation(
        workflow_id="workflow-live-owner",
        feature_id=None,
        child_workflow_id=None,
        repository_id=None,
        operation_type=ExternalOperationType.RUN_CODING_EXECUTOR,
        idempotency_key="workflow-live-owner:coding",
        input_fingerprint="coding",
    )
    await journal.transition_status(
        operation.operation_id,
        ExternalOperationStatus.STARTING,
        event_type="operation_starting",
    )
    await journal.record_attempt(operation.operation_id, provider="openai_responses")

    summary = await RecoveryService(journal).recover_incomplete_operations()
    untouched = await journal.get(operation.operation_id)

    assert summary.scanned == 0
    assert untouched.status is ExternalOperationStatus.RUNNING
    await database.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "active_status",
    [
        ExternalOperationStatus.STARTING,
        ExternalOperationStatus.RUNNING,
        ExternalOperationStatus.CANCELLATION_REQUESTED,
    ],
)
async def test_periodic_recovery_sweeps_an_operation_after_its_fresh_lease_expires(
    tmp_path: Path, active_status: ExternalOperationStatus
) -> None:
    """Every startup-skipped active state is fenced later, and the loop cancels cleanly."""

    class ObservingJournal(ExternalOperationJournal):
        def __init__(self, database: Database) -> None:
            super().__init__(database, stale_after_seconds=60)
            self.fresh_checked = asyncio.Event()
            self.stale_recovered = asyncio.Event()

        async def claim_stale_operation(self, operation: Any) -> bool:
            claimed = await super().claim_stale_operation(operation)
            if not claimed:
                self.fresh_checked.set()
            return claimed

        async def record_recovery_failure(self, operation_id: str, **kwargs: Any) -> Any:
            recovered = await super().record_recovery_failure(operation_id, **kwargs)
            self.stale_recovered.set()
            return recovered

    database = Database(f"sqlite+aiosqlite:///{tmp_path / f'periodic-{active_status.value}.db'}")
    await database.create_schema()
    journal = ObservingJournal(database)
    operation = await journal.create_operation(
        workflow_id="workflow-periodic-recovery",
        feature_id=None,
        child_workflow_id=None,
        repository_id=None,
        operation_type=ExternalOperationType.PUSH_BRANCH,
        idempotency_key=f"workflow-periodic-recovery:{active_status.value}:push",
        input_fingerprint="push",
    )
    await journal.transition_status(
        operation.operation_id,
        ExternalOperationStatus.STARTING,
        event_type="operation_starting",
    )
    if active_status is not ExternalOperationStatus.STARTING:
        await journal.record_attempt(operation.operation_id)
    if active_status is ExternalOperationStatus.CANCELLATION_REQUESTED:
        await journal.transition_status(
            operation.operation_id,
            ExternalOperationStatus.CANCELLATION_REQUESTED,
            event_type="cancellation_requested",
        )
    recovery = RecoveryService(journal)
    sweep_task = asyncio.create_task(recovery.run_periodic_recovery(interval_seconds=0.001))
    try:
        await asyncio.wait_for(journal.fresh_checked.wait(), timeout=1)
        fresh = await journal.get(operation.operation_id)
        assert fresh.status is active_status

        async with database.session() as session:
            await session.execute(
                update(ExternalOperationModel)
                .where(ExternalOperationModel.operation_id == operation.operation_id)
                .values(heartbeat_at=datetime.now(UTC) - timedelta(minutes=2))
            )
            await session.commit()

        await asyncio.wait_for(journal.stale_recovered.wait(), timeout=1)
        recovered = await journal.get(operation.operation_id)
        # The STARTING case never spent an attempt, so a credentialed request can still
        # reconcile it and the sweep leaves it able to. The other two spent the operation's
        # only attempt, so there is nothing left to defer to and a person has to look.
        assert recovered.status is (
            ExternalOperationStatus.AWAITING_RECONCILIATION
            if active_status is ExternalOperationStatus.STARTING
            else ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE
        )
        assert recovery.last_summary is not None
        assert recovery.last_summary.scanned == 1
    finally:
        sweep_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sweep_task
        await database.dispose()


@pytest.mark.asyncio
async def test_recovery_stale_claim_loses_to_a_concurrent_heartbeat(tmp_path: Path) -> None:
    """A heartbeat between startup's list and claim prevents an atomic stale takeover."""

    class ClaimBarrierJournal(ExternalOperationJournal):
        def __init__(self, database: Database) -> None:
            super().__init__(database, stale_after_seconds=60)
            self.claim_started = asyncio.Event()
            self.release_claim = asyncio.Event()

        async def claim_stale_operation(self, operation: Any) -> bool:
            self.claim_started.set()
            await self.release_claim.wait()
            return await super().claim_stale_operation(operation)

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'recovery-heartbeat-race.db'}")
    await database.create_schema()
    journal = ClaimBarrierJournal(database)
    operation = await journal.create_operation(
        workflow_id="workflow-recovery-race",
        feature_id=None,
        child_workflow_id=None,
        repository_id=None,
        operation_type=ExternalOperationType.RUN_CODING_EXECUTOR,
        idempotency_key="workflow-recovery-race:coding",
        input_fingerprint="coding",
    )
    await journal.transition_status(
        operation.operation_id,
        ExternalOperationStatus.STARTING,
        event_type="operation_starting",
    )
    await journal.record_attempt(operation.operation_id)
    async with database.session() as session:
        await session.execute(
            update(ExternalOperationModel)
            .where(ExternalOperationModel.operation_id == operation.operation_id)
            .values(heartbeat_at=datetime.now(UTC) - timedelta(minutes=2))
        )
        await session.commit()
    assert await journal.unresolved_critical_count() == 1

    recovery_task = asyncio.create_task(RecoveryService(journal).recover_incomplete_operations())
    await journal.claim_started.wait()
    await journal.record_heartbeat(operation.operation_id)
    journal.release_claim.set()
    summary = await recovery_task

    current = await journal.get(operation.operation_id)
    assert summary.scanned == 0
    assert current.status is ExternalOperationStatus.RUNNING
    assert await journal.unresolved_critical_count() == 0
    await database.dispose()


@pytest.mark.asyncio
async def test_stale_recovery_claim_fences_original_worker_writes(tmp_path: Path) -> None:
    """Once recovery wins, the old action cannot heartbeat, succeed, or fail the row."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'recovery-fence.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    operation = await journal.create_operation(
        workflow_id="workflow-recovery-fence",
        feature_id=None,
        child_workflow_id=None,
        repository_id=None,
        operation_type=ExternalOperationType.PUSH_BRANCH,
        idempotency_key="workflow-recovery-fence:push",
        input_fingerprint="push",
    )
    await journal.transition_status(
        operation.operation_id,
        ExternalOperationStatus.STARTING,
        event_type="operation_starting",
    )
    await journal.record_attempt(operation.operation_id)
    await asyncio.sleep(0.01)
    stale = await journal.get(operation.operation_id)

    assert await journal.claim_stale_operation(stale)
    with pytest.raises(OperationLeaseLostError, match="lease is no longer active"):
        await journal.record_heartbeat(operation.operation_id)
    with pytest.raises(OperationLeaseLostError, match="fenced"):
        await journal.record_result(
            operation.operation_id,
            external_reference="old-worker-result",
            result_payload={"completed": True},
        )
    with pytest.raises(OperationLeaseLostError, match="fenced"):
        await journal.record_failure(
            operation.operation_id,
            status=ExternalOperationStatus.FAILED_TERMINAL,
            error_code="old_worker",
            error_message="old worker tried to overwrite recovery",
        )
    fenced = await journal.get(operation.operation_id)
    assert fenced.status is ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE
    assert fenced.error_code == "stale_recovery_claimed"

    reconciled = await journal.record_reconciled_result(
        operation.operation_id,
        external_reference="verified-result",
        result_payload={"reconciled": True},
    )
    assert reconciled.status is ExternalOperationStatus.SUCCEEDED
    await database.dispose()


@pytest.mark.asyncio
async def test_persisted_operation_errors_redact_literal_environment_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Literal deployment secrets are removed even when output omits secret-shaped labels."""
    secrets = {
        "PLATFORM_API_KEY": "platform-value-927461",
        "DATABASE_URL": "postgresql://runtime-value-38517/db",
        "REDIS_URL": "redis://runtime-value-64928/0",
        "GITHUB_TOKEN": "provider-value-573904",
    }
    for key, value in secrets.items():
        monkeypatch.setenv(key, value)
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'redacted-errors.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    operation = await journal.create_operation(
        workflow_id="workflow-redacted-error",
        feature_id=None,
        child_workflow_id=None,
        repository_id=None,
        operation_type=ExternalOperationType.RUN_TESTS,
        idempotency_key="workflow-redacted-error:test",
        input_fingerprint="test",
    )
    persisted = await journal.record_failure(
        operation.operation_id,
        status=ExternalOperationStatus.FAILED_TERMINAL,
        error_code="SubprocessError",
        error_message="connections failed: " + " | ".join(secrets.values()),
    )

    assert persisted.error_message is not None
    assert "connections failed" in persisted.error_message
    assert persisted.error_message.count("[REDACTED]") == len(secrets)
    assert all(value not in persisted.error_message for value in secrets.values())
    await database.dispose()


@pytest.mark.asyncio
async def test_interruptible_process_runner_terminates_active_subprocess_on_cancellation(
    tmp_path: Path,
) -> None:
    """Cancellation ends a local process group and returns a typed cancelled result."""
    token = MockCancellationToken()
    task = asyncio.create_task(
        AsyncioProcessRunner(termination_grace_seconds=0.1).run(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            tmp_path,
            timeout_seconds=60,
            cancellation_token=token,
        )
    )
    await asyncio.sleep(0.1)
    token.cancel()
    result = await task

    assert result.cancelled
    assert result.return_code is None
    assert not result.succeeded


@pytest.mark.asyncio
async def test_cancelled_coding_executor_never_writes_workspace_files(tmp_path: Path) -> None:
    """A cancellation signal before coding prevents all downstream workspace mutation."""
    token = MockCancellationToken(cancelled=True)
    executor = MockCodingExecutor(file_updates={"generated.py": "VALUE = 1\n"})

    with pytest.raises(CancellationRequested):
        await executor.execute(
            workspace_root=tmp_path,
            instructions="Implement the generated file.",
            input_text="Do not run after cancellation.",
            cancellation_token=token,
        )
    assert not (tmp_path / "generated.py").exists()


@pytest.mark.asyncio
async def test_journaled_github_pr_creation_reuses_prior_success_after_state_crash(
    tmp_path: Path,
) -> None:
    """A second execution reuses the journaled PR result rather than creating another PR."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'github.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    operations = ExternalOperationExecutor(
        journal=journal,
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(
            workflow_id="workflow-pr", feature_id="feature-pr", repository_id="platform"
        ),
    )
    provider = MockGitHubService()
    service = JournaledGitHubService(provider, operation_executor=operations)
    first = await service.create_pull_request(
        "example/platform",
        title="[AI] durable PR",
        body="Feature ID: feature-pr",
        source_branch="ai/feature-pr",
        target_branch="main",
        draft=True,
        expected_head_sha="approved-commit-sha",
    )
    replay = await service.create_pull_request(
        "example/platform",
        title="[AI] durable PR",
        body="Feature ID: feature-pr",
        source_branch="ai/feature-pr",
        target_branch="main",
        draft=True,
        expected_head_sha="approved-commit-sha",
    )

    assert first.number == replay.number
    assert first.head_sha == replay.head_sha == "approved-commit-sha"
    assert len(provider.pull_requests) == 1
    await database.dispose()


@pytest.mark.asyncio
async def test_stale_pr_creation_is_reconciled_with_fresh_request_scoped_access(
    tmp_path: Path,
) -> None:
    """A crash after GitHub creates a PR finds it instead of creating another one."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'github-recovery.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    provider = MockGitHubService()
    repository = "example/platform"
    title = "[AI] recover PR"
    branch = "ai/recover"
    body = "Feature ID: recover"
    provider.create_pull_request(
        repository,
        title=title,
        body=body,
        source_branch=branch,
        target_branch="main",
        draft=True,
    )
    safe_input = {
        "repository": repository,
        "head_branch": branch,
        "base_branch": "main",
        "title": title,
        "body_fingerprint": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "draft": True,
    }
    fingerprint = fingerprint_operation_input(safe_input)
    operation = await journal.create_operation(
        workflow_id="workflow-pr-recovery",
        feature_id="feature-pr-recovery",
        child_workflow_id=None,
        repository_id="platform",
        operation_type=ExternalOperationType.CREATE_PULL_REQUEST,
        idempotency_key=deterministic_operation_key(
            workflow_id="workflow-pr-recovery",
            feature_id="feature-pr-recovery",
            child_workflow_id=None,
            repository_id="platform",
            operation_type=ExternalOperationType.CREATE_PULL_REQUEST,
            logical_step="create_pr:ai/recover:main",
            input_fingerprint=fingerprint,
        ),
        input_fingerprint=fingerprint,
        safe_metadata={"logical_step": "create_pr:ai/recover:main", **safe_input},
    )
    await journal.transition_status(
        operation.operation_id, ExternalOperationStatus.STARTING, event_type="operation_starting"
    )
    await journal.record_attempt(operation.operation_id, provider="github")
    await asyncio.sleep(0.01)
    operations = ExternalOperationExecutor(
        journal=journal,
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(
            workflow_id="workflow-pr-recovery",
            feature_id="feature-pr-recovery",
            repository_id="platform",
        ),
    )
    service = JournaledGitHubService(provider, operation_executor=operations)

    recovered = await service.create_pull_request(
        repository,
        title=title,
        body=body,
        source_branch=branch,
        target_branch="main",
        draft=True,
    )

    assert recovered.number == 1
    assert len(provider.pull_requests) == 1
    assert (await journal.get(operation.operation_id)).status is ExternalOperationStatus.SUCCEEDED
    await database.dispose()


@pytest.mark.asyncio
async def test_executor_never_repeats_an_active_operation_from_another_worker(
    tmp_path: Path,
) -> None:
    """A second worker receives a safe in-progress conflict while the first owns the operation."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'concurrency.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    scope = ExternalOperationScope(workflow_id="workflow-lock")
    first = ExternalOperationExecutor(
        journal=journal, cancellation_token=MockCancellationToken(), scope=scope
    )
    second = ExternalOperationExecutor(
        journal=journal, cancellation_token=MockCancellationToken(), scope=scope
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def action() -> tuple[str, OperationResult]:
        started.set()
        await release.wait()
        return "done", OperationResult(payload={"done": True})

    first_task = asyncio.create_task(
        first.run(
            operation_type=ExternalOperationType.RUN_TESTS,
            logical_step="same",
            safe_input={"workspace_path": str(tmp_path)},
            action=action,
        )
    )
    await started.wait()
    with pytest.raises(Exception, match="already active"):
        await second.run(
            operation_type=ExternalOperationType.RUN_TESTS,
            logical_step="same",
            safe_input={"workspace_path": str(tmp_path)},
            action=action,
        )
    release.set()
    assert (await first_task).operation.status is ExternalOperationStatus.SUCCEEDED
    await database.dispose()


@pytest.mark.asyncio
async def test_cancellation_during_atomic_attempt_claim_finishes_cancelled(tmp_path: Path) -> None:
    """Cancellation cannot strand the old claim-to-attempt STARTING window."""

    class AttemptClaimBarrierJournal(ExternalOperationJournal):
        def __init__(self, database: Database) -> None:
            super().__init__(database)
            self.claim_committed = asyncio.Event()
            self.release_claim = asyncio.Event()
            self.claimed_operation_id: str | None = None

        async def claim_operation(self, operation_id: str, **kwargs: Any) -> Any:
            operation = await super().claim_operation(operation_id, **kwargs)
            self.claimed_operation_id = operation.operation_id
            self.claim_committed.set()
            await self.release_claim.wait()
            return operation

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'cancel-atomic-claim.db'}")
    await database.create_schema()
    journal = AttemptClaimBarrierJournal(database)
    token = MockCancellationToken()
    executor = ExternalOperationExecutor(
        journal=journal,
        cancellation_token=token,
        scope=ExternalOperationScope(workflow_id="workflow-cancel-atomic-claim"),
    )
    action_calls = 0

    async def action() -> tuple[str, OperationResult]:
        nonlocal action_calls
        action_calls += 1
        return "unexpected", OperationResult()

    execution = asyncio.create_task(
        executor.run(
            operation_type=ExternalOperationType.RUN_TESTS,
            logical_step="cancel-during-claim",
            safe_input={"workspace_path": str(tmp_path)},
            action=action,
        )
    )
    await journal.claim_committed.wait()
    assert journal.claimed_operation_id is not None
    await journal.transition_status(
        journal.claimed_operation_id,
        ExternalOperationStatus.CANCELLATION_REQUESTED,
        event_type="cancellation_requested",
    )
    token.cancel()
    journal.release_claim.set()

    with pytest.raises(CancellationRequested):
        await execution
    operation = (await journal.list_operations_for_workflow("workflow-cancel-atomic-claim"))[0]
    assert action_calls == 0
    assert operation.status is ExternalOperationStatus.CANCELLED
    assert operation.attempt == 1
    assert await journal.list_incomplete_operations() == []
    assert [
        event.new_status for event in await journal.events_for_operation(operation.operation_id)
    ] == [
        ExternalOperationStatus.PENDING,
        ExternalOperationStatus.STARTING,
        ExternalOperationStatus.RUNNING,
        ExternalOperationStatus.CANCELLATION_REQUESTED,
        ExternalOperationStatus.CANCELLED,
    ]
    await database.dispose()


@pytest.mark.asyncio
async def test_provider_exception_text_is_never_persisted(tmp_path: Path) -> None:
    """Generic provider failures retain a platform category, never request-scoped text."""
    secrets = (
        "ghp_exampleProviderCredential123",
        "gho_exampleOAuthCredential456",
        "sk-example-openai-credential-789",
        "Authorization: Bearer opaque-request-value",
    )
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'provider-error.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    executor = ExternalOperationExecutor(
        journal=journal,
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(workflow_id="workflow-provider-error"),
    )

    async def action() -> tuple[str, OperationResult]:
        raise RuntimeError("request failed: " + " | ".join(secrets))

    with pytest.raises(RuntimeError, match="request failed"):
        await executor.run(
            operation_type=ExternalOperationType.CREATE_PULL_REQUEST,
            logical_step="provider-error",
            safe_input={"repository": "example/platform"},
            action=action,
        )

    operation = (await journal.list_operations_for_workflow("workflow-provider-error"))[0]
    assert operation.status is ExternalOperationStatus.FAILED_TERMINAL
    assert operation.error_code == "create_pull_request_failed"
    # The type name is a platform-owned symbol and is deliberately kept: AB-Feature-174's
    # three identical coding-call failures were three empty rows without it. The provider's
    # message text -- where the secrets live -- still never persists, which the byte scan
    # below is the real assertion for.
    assert operation.error_message == (
        "external operation failed before a confirmed result (RuntimeError)"
    )
    await database.dispose()
    durable_bytes = b"".join(
        path.read_bytes() for path in tmp_path.glob("provider-error.db*") if path.is_file()
    )
    assert all(secret.encode() not in durable_bytes for secret in secrets)


@pytest.mark.asyncio
async def test_terminal_operation_is_preserved_and_never_invoked_again(tmp_path: Path) -> None:
    """A deterministic terminal failure cannot be reset to STARTING by a replay."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'terminal-operation.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    executor = ExternalOperationExecutor(
        journal=journal,
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(workflow_id="workflow-terminal"),
    )
    calls = 0

    async def fail() -> tuple[str, OperationResult]:
        nonlocal calls
        calls += 1
        raise RuntimeError("controlled provider failure")

    with pytest.raises(RuntimeError, match="controlled provider failure"):
        await executor.run(
            operation_type=ExternalOperationType.RUN_TESTS,
            logical_step="terminal",
            safe_input={"workspace_path": str(tmp_path)},
            action=fail,
        )
    operation = (await journal.list_operations_for_workflow("workflow-terminal"))[0]
    event_count = len(await journal.events_for_operation(operation.operation_id))
    assert operation.status is ExternalOperationStatus.FAILED_TERMINAL

    with pytest.raises(ExternalOperationError, match="terminal failure"):
        await executor.run(
            operation_type=ExternalOperationType.RUN_TESTS,
            logical_step="terminal",
            safe_input={"workspace_path": str(tmp_path)},
            action=fail,
        )

    preserved = await journal.get(operation.operation_id)
    assert calls == 1
    assert preserved.status is ExternalOperationStatus.FAILED_TERMINAL
    assert len(await journal.events_for_operation(operation.operation_id)) == event_count
    await database.dispose()


@pytest.mark.asyncio
async def test_concurrent_success_wins_over_stale_cancellation_transition(tmp_path: Path) -> None:
    """A cancellation snapshot cannot overwrite success committed while it was paused."""

    class CancellationBarrierJournal(ExternalOperationJournal):
        def __init__(self, database: Database) -> None:
            super().__init__(database)
            self.cancellation_entered = asyncio.Event()
            self.release_cancellation = asyncio.Event()

        async def transition_status(
            self, operation_id: str, status: ExternalOperationStatus, **kwargs: Any
        ) -> Any:
            if status is ExternalOperationStatus.CANCELLATION_REQUESTED:
                self.cancellation_entered.set()
                await self.release_cancellation.wait()
            return await super().transition_status(operation_id, status, **kwargs)

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'success-cancel-race.db'}")
    await database.create_schema()
    journal = CancellationBarrierJournal(database)
    operation = await journal.create_operation(
        workflow_id="workflow-success-cancel-race",
        feature_id=None,
        child_workflow_id=None,
        repository_id=None,
        operation_type=ExternalOperationType.PUSH_BRANCH,
        idempotency_key="workflow-success-cancel-race:push",
        input_fingerprint="push",
    )
    await journal.transition_status(
        operation.operation_id,
        ExternalOperationStatus.STARTING,
        event_type="operation_starting",
    )
    await journal.record_attempt(operation.operation_id)

    cancellation = asyncio.create_task(
        journal.transition_status(
            operation.operation_id,
            ExternalOperationStatus.CANCELLATION_REQUESTED,
            event_type="cancellation_requested",
        )
    )
    await journal.cancellation_entered.wait()
    succeeded = await journal.record_result(
        operation.operation_id,
        external_reference="remote-sha",
        result_payload={"pushed": True},
    )
    journal.release_cancellation.set()

    with pytest.raises(OperationTransitionConflictError, match="succeeded.*cancellation_requested"):
        await cancellation
    current = await journal.get(operation.operation_id)
    assert succeeded.status is ExternalOperationStatus.SUCCEEDED
    assert current.status is ExternalOperationStatus.SUCCEEDED
    assert ExternalOperationStatus.CANCELLATION_REQUESTED not in {
        event.new_status for event in await journal.events_for_operation(operation.operation_id)
    }
    await database.dispose()


@pytest.mark.asyncio
async def test_manual_transition_cannot_reopen_a_terminal_operation(tmp_path: Path) -> None:
    """Neither STARTING nor cancellation may be written after terminal failure."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'terminal-transition.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    operation = await journal.create_operation(
        workflow_id="workflow-terminal-transition",
        feature_id=None,
        child_workflow_id=None,
        repository_id=None,
        operation_type=ExternalOperationType.RUN_TESTS,
        idempotency_key="workflow-terminal-transition:test",
        input_fingerprint="test",
    )
    await journal.transition_status(
        operation.operation_id,
        ExternalOperationStatus.STARTING,
        event_type="operation_starting",
    )
    await journal.record_attempt(operation.operation_id)
    terminal = await journal.record_failure(
        operation.operation_id,
        status=ExternalOperationStatus.FAILED_TERMINAL,
        error_code="deterministic_failure",
        error_message="deterministic failure",
    )
    event_count = len(await journal.events_for_operation(operation.operation_id))

    for requested in (
        ExternalOperationStatus.STARTING,
        ExternalOperationStatus.CANCELLATION_REQUESTED,
    ):
        with pytest.raises(OperationTransitionConflictError, match="failed_terminal"):
            await journal.transition_status(
                operation.operation_id,
                requested,
                event_type="illegal_terminal_reopen",
            )

    preserved = await journal.get(operation.operation_id)
    assert terminal.status is ExternalOperationStatus.FAILED_TERMINAL
    assert preserved.status is ExternalOperationStatus.FAILED_TERMINAL
    assert preserved.attempt == terminal.attempt
    assert len(await journal.events_for_operation(operation.operation_id)) == event_count
    await database.dispose()


@pytest.mark.asyncio
async def test_simultaneous_pending_callers_atomically_claim_one_action(tmp_path: Path) -> None:
    """Both workers can observe one PENDING intent, but only the CAS winner executes."""

    class CreateBarrierJournal(ExternalOperationJournal):
        def __init__(self, database: Database) -> None:
            super().__init__(database)
            self._created = 0
            self._both_created = asyncio.Event()

        async def create_operation(self, **kwargs: Any) -> Any:
            operation = await super().create_operation(**kwargs)
            self._created += 1
            if self._created == 2:
                self._both_created.set()
            await self._both_created.wait()
            return operation

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'atomic-claim.db'}")
    await database.create_schema()
    journal = CreateBarrierJournal(database)
    scope = ExternalOperationScope(workflow_id="workflow-atomic-claim")
    executors = [
        ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=scope,
        )
        for _ in range(2)
    ]
    action_started = asyncio.Event()
    release_action = asyncio.Event()
    calls = 0

    async def action() -> tuple[str, OperationResult]:
        nonlocal calls
        calls += 1
        action_started.set()
        await release_action.wait()
        return "done", OperationResult(payload={"done": True})

    tasks = [
        asyncio.create_task(
            executor.run(
                operation_type=ExternalOperationType.RUN_TESTS,
                logical_step="same-pending-intent",
                safe_input={"workspace_path": str(tmp_path)},
                action=action,
            )
        )
        for executor in executors
    ]
    await action_started.wait()
    await asyncio.sleep(0)
    release_action.set()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    assert calls == 1
    assert sum(isinstance(item, OperationInProgressError) for item in outcomes) == 1
    assert sum(not isinstance(item, BaseException) for item in outcomes) == 1
    await database.dispose()


@pytest.mark.asyncio
async def test_heartbeat_loss_cancels_action_and_records_unknown_state(tmp_path: Path) -> None:
    """A failed durable heartbeat aborts work instead of silently losing its lease."""

    class FailingHeartbeatJournal(ExternalOperationJournal):
        async def record_heartbeat(self, operation_id: str) -> Any:
            del operation_id
            raise RuntimeError("database heartbeat unavailable")

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'heartbeat-loss.db'}")
    await database.create_schema()
    journal = FailingHeartbeatJournal(database)
    executor = ExternalOperationExecutor(
        journal=journal,
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(workflow_id="workflow-heartbeat-loss"),
        heartbeat_seconds=0.001,
    )
    action_cancelled = asyncio.Event()

    async def action() -> tuple[str, OperationResult]:
        try:
            await asyncio.Event().wait()
        finally:
            action_cancelled.set()
        return "unreachable", OperationResult()

    with pytest.raises(OperationLeaseLostError, match="heartbeat could not be renewed"):
        await executor.run(
            operation_type=ExternalOperationType.RUN_CODING_EXECUTOR,
            logical_step="lease-loss",
            safe_input={"workspace_path": str(tmp_path)},
            action=action,
        )

    operation = (await journal.list_operations_for_workflow("workflow-heartbeat-loss"))[0]
    assert action_cancelled.is_set()
    assert operation.status is ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE
    assert operation.error_code == "operation_lease_lost"
    await database.dispose()


@pytest.mark.asyncio
async def test_completed_clone_is_reused_when_the_graph_checkpoint_was_not_written(
    tmp_path: Path,
) -> None:
    """A completed clone is safe to reuse even though its destination now exists."""
    remote = tmp_path / "remote.git"
    source = tmp_path / "source"
    target = tmp_path / "target"
    _git("init", "--bare", str(remote))
    _git("init", str(source))
    _git("-C", str(source), "config", "user.email", "tests@example.invalid")
    _git("-C", str(source), "config", "user.name", "Test User")
    (source / "README.md").write_text("# source\n", encoding="utf-8")
    _git("-C", str(source), "add", "README.md")
    _git("-C", str(source), "commit", "-m", "initial")
    _git("-C", str(source), "remote", "add", "origin", str(remote))
    _git("-C", str(source), "push", "origin", "HEAD:main")

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'clone.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    operations = ExternalOperationExecutor(
        journal=journal,
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(workflow_id="workflow-clone"),
    )
    service = InterruptibleGitService(
        cancellation_token=MockCancellationToken(),
        operation_executor=operations,
        workspace_root=tmp_path,
    )

    first = await service.clone(str(remote), target)
    replay = await service.clone(str(remote), target)

    assert first == target
    assert replay == target
    assert (target / ".git").is_dir()
    recorded = await journal.list_operations_for_workflow("workflow-clone")
    assert recorded[0].max_attempts == 3
    await database.dispose()


def test_interruptible_git_supplies_a_non_secret_default_commit_identity() -> None:
    """Worker commits must not depend on a mutable container-global Git configuration."""
    service = InterruptibleGitService(cancellation_token=MockCancellationToken())

    environment = service._environment_with_os()

    assert environment["GIT_AUTHOR_NAME"]
    assert environment["GIT_COMMITTER_NAME"] == environment["GIT_AUTHOR_NAME"]
    assert environment["GIT_AUTHOR_EMAIL"] == "automation@localhost.invalid"
    assert environment["GIT_COMMITTER_EMAIL"] == environment["GIT_AUTHOR_EMAIL"]


@pytest.mark.asyncio
async def test_recovery_identifies_commit_created_before_its_sha_was_persisted(
    tmp_path: Path,
) -> None:
    """The pre-effect parent/message evidence prevents a duplicate commit after a crash."""
    repository = tmp_path / "repository"
    _git("init", str(repository))
    _git("-C", str(repository), "config", "user.email", "tests@example.invalid")
    _git("-C", str(repository), "config", "user.name", "Test User")
    tracked = repository / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    _git("-C", str(repository), "add", "tracked.txt")
    _git("-C", str(repository), "commit", "-m", "initial")
    parent_sha = _git_output("-C", str(repository), "rev-parse", "HEAD")
    tracked.write_text("after\n", encoding="utf-8")
    _git("-C", str(repository), "add", "tracked.txt")
    _git("-C", str(repository), "commit", "-m", "durable change")
    created_sha = _git_output("-C", str(repository), "rev-parse", "HEAD")

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'commit.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    operation = await journal.create_operation(
        workflow_id="workflow-commit",
        feature_id=None,
        child_workflow_id=None,
        repository_id="repository",
        operation_type=ExternalOperationType.CREATE_COMMIT,
        idempotency_key="workflow-commit:commit",
        input_fingerprint="commit",
        safe_metadata={
            "workspace_path": str(repository),
            "parent_head_sha": parent_sha,
            "commit_message": "durable change",
        },
    )
    await journal.transition_status(
        operation.operation_id, ExternalOperationStatus.STARTING, event_type="operation_starting"
    )
    await journal.record_attempt(operation.operation_id, provider="git")
    await asyncio.sleep(0.01)

    summary = await RecoveryService(journal).recover_incomplete_operations()
    recovered = await journal.get(operation.operation_id)

    assert summary.reconciled == 1
    assert recovered.status is ExternalOperationStatus.SUCCEEDED
    assert recovered.external_reference == created_sha
    assert recovered.result_payload == {
        "recovered": True,
        "commit_sha": created_sha,
        "recovery_method": "local_commit_present",
    }
    await database.dispose()


@pytest.mark.asyncio
async def test_commit_call_reconciles_a_crash_after_git_changed_head(tmp_path: Path) -> None:
    """Reviewed content is a stable key even after commit makes the working diff empty."""
    repository = tmp_path / "repository"
    _git("init", str(repository))
    _git("-C", str(repository), "config", "user.email", "tests@example.invalid")
    _git("-C", str(repository), "config", "user.name", "Test User")
    tracked = repository / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    _git("-C", str(repository), "add", "tracked.txt")
    _git("-C", str(repository), "commit", "-m", "initial")
    parent_sha = _git_output("-C", str(repository), "rev-parse", "HEAD")
    tracked.write_text("reviewed\n", encoding="utf-8")

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'commit-reconcile.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    scope = ExternalOperationScope(
        workflow_id="workflow-commit-reconcile",
        feature_id="workflow-commit-reconcile",
        child_workflow_id="workflow-commit-reconcile:repository",
        repository_id="repository",
    )
    service = InterruptibleGitService(
        cancellation_token=MockCancellationToken(),
        operation_executor=ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=scope,
        ),
    )
    files = ["tracked.txt"]
    message = "workflow publication"
    reviewed_fingerprint = reviewed_content_fingerprint(repository, files)
    expected_tree_fingerprint = await service._working_commit_tree_fingerprint(repository, files)
    idempotency_input = {
        "workspace_path": str(repository.resolve()),
        "expected_staged_files": files,
        "commit_message": message,
        "reviewed_content_fingerprint": reviewed_fingerprint,
    }
    fingerprint = fingerprint_operation_input(idempotency_input)
    operation = await journal.create_operation(
        workflow_id=scope.workflow_id,
        feature_id=scope.feature_id,
        child_workflow_id=scope.child_workflow_id,
        repository_id=scope.repository_id,
        operation_type=ExternalOperationType.CREATE_COMMIT,
        idempotency_key=deterministic_operation_key(
            workflow_id=scope.workflow_id,
            feature_id=scope.feature_id,
            child_workflow_id=scope.child_workflow_id,
            repository_id=scope.repository_id,
            operation_type=ExternalOperationType.CREATE_COMMIT,
            logical_step="create_commit",
            input_fingerprint=fingerprint,
        ),
        input_fingerprint=fingerprint,
        safe_metadata={
            "logical_step": "create_commit",
            **idempotency_input,
            "parent_head_sha": parent_sha,
            "working_tree_fingerprint": "pre-crash-diff",
            "expected_tree_fingerprint": expected_tree_fingerprint,
        },
    )
    await journal.transition_status(
        operation.operation_id, ExternalOperationStatus.STARTING, event_type="operation_starting"
    )
    await journal.record_attempt(operation.operation_id, provider="git")
    _git("-C", str(repository), "add", "--", *files)
    _git("-C", str(repository), "commit", "-m", message)
    created_sha = _git_output("-C", str(repository), "rev-parse", "HEAD")
    await asyncio.sleep(0.01)

    recovered_sha = await service.commit(
        repository,
        message,
        files=files,
        expected_content_fingerprint=reviewed_fingerprint,
    )
    recovered = await journal.get(operation.operation_id)

    assert recovered_sha == created_sha
    assert recovered.status is ExternalOperationStatus.SUCCEEDED
    assert recovered.result_payload and recovered.result_payload["recovered"] is True
    assert _git_output("-C", str(repository), "rev-list", "--count", f"{parent_sha}..HEAD") == "1"
    await database.dispose()


@pytest.mark.asyncio
async def test_legacy_publisher_reuses_commit_and_push_after_lost_graph_checkpoint(
    tmp_path: Path,
) -> None:
    """Re-entering the legacy publisher with pre-publication state has no duplicate effects."""
    remote = tmp_path / "remote.git"
    repository = tmp_path / "repository"
    _git("init", "--bare", str(remote))
    _git("init", "--initial-branch=main", str(repository))
    _git("-C", str(repository), "config", "user.email", "tests@example.invalid")
    _git("-C", str(repository), "config", "user.name", "Test User")
    tracked = repository / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    _git("-C", str(repository), "add", "tracked.txt")
    _git("-C", str(repository), "commit", "-m", "initial")
    _git("-C", str(repository), "remote", "add", "origin", str(remote))
    _git("-C", str(repository), "push", "origin", "main")
    branch = "ai/legacy-publication"
    _git("-C", str(repository), "switch", "-c", branch)
    tracked.write_text("reviewed\n", encoding="utf-8")

    completion = create_artifact(
        CodeCompletionArtifact,
        workflow_id="legacy-publication",
        artifact_id="006_code_completion.json",
        producer="engineer",
        payload={
            "completion_status": "completed",
            "summary": "Publish reviewed bytes.",
            "file_changes": [
                {
                    "path": "tracked.txt",
                    "change_type": "modified",
                    "description": "Reviewed implementation.",
                }
            ],
            "validation_results": [],
            "test_coverage_percent": None,
            "remaining_work": [],
            "commit_sha": None,
        },
        metadata={},
    )
    review = create_artifact(
        ReviewArtifact,
        workflow_id="legacy-publication",
        artifact_id="007_review.json",
        producer="reviewer",
        payload={
            "verdict": "approved",
            "summary": "The exact workspace diff is approved.",
            "requirement_checks": [],
            "findings": [],
            "architecture_assessment": "The change is scoped.",
            "security_assessment": "No security issue was found.",
            "test_coverage_assessment": "Validation evidence is sufficient.",
        },
        metadata={},
    )
    state = create_initial_agent_state(
        workflow_id="legacy-publication",
        workspace_descriptor=WorkspaceDescriptor(
            workspace_id="legacy-publication",
            root_path=str(repository),
            source_repo_url=str(remote),
            default_branch="main",
            working_branch=branch,
        ),
    )
    state["artifacts"] = [completion, review]
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'legacy-publication.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    git_service = InterruptibleGitService(
        cancellation_token=MockCancellationToken(),
        operation_executor=ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(workflow_id="legacy-publication"),
        ),
    )
    publisher = ApprovedChangePublisher(git_service=git_service)

    first = await publisher.run(state)
    # Simulate losing the publisher-node checkpoint by replaying the original state.
    replay = await publisher.run(state)
    first_completion = first["artifacts"][0]
    replay_completion = replay["artifacts"][0]
    recorded = await journal.list_operations_for_workflow("legacy-publication")

    assert first_completion.commit_sha == replay_completion.commit_sha
    assert _git_output("-C", str(repository), "rev-list", "--count", "main..HEAD") == "1"
    assert (
        _git_output("-C", str(repository), "ls-remote", "origin", f"refs/heads/{branch}").split()[0]
        == first_completion.commit_sha
    )
    assert [item.operation_type for item in recorded].count(
        ExternalOperationType.CREATE_COMMIT
    ) == 1
    assert [item.operation_type for item in recorded].count(ExternalOperationType.PUSH_BRANCH) == 1
    await database.dispose()


@pytest.mark.asyncio
async def test_stale_push_is_reconciled_from_the_remote_branch_before_retry(tmp_path: Path) -> None:
    """A remote SHA match turns an interrupted push into a reusable success."""
    remote = tmp_path / "push-remote.git"
    repository = tmp_path / "push-source"
    _git("init", "--bare", str(remote))
    _git("init", str(repository))
    _git("-C", str(repository), "config", "user.email", "tests@example.invalid")
    _git("-C", str(repository), "config", "user.name", "Test User")
    tracked = repository / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    _git("-C", str(repository), "add", "tracked.txt")
    _git("-C", str(repository), "commit", "-m", "initial")
    _git("-C", str(repository), "switch", "-c", "ai/recover-push")
    _git("-C", str(repository), "remote", "add", "origin", str(remote))
    _git("-C", str(repository), "push", "-u", "origin", "ai/recover-push")
    remote_before = _git_output("-C", str(repository), "rev-parse", "HEAD")
    tracked.write_text("after\n", encoding="utf-8")
    _git("-C", str(repository), "add", "tracked.txt")
    _git("-C", str(repository), "commit", "-m", "push change")
    local_sha = _git_output("-C", str(repository), "rev-parse", "HEAD")
    _git("-C", str(repository), "push", "origin", "ai/recover-push:ai/recover-push")

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'push.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    idempotency_input = {
        "workspace_path": str(repository),
        "remote": "origin",
        "branch": "ai/recover-push",
        "expected_local_sha": local_sha,
    }
    fingerprint = fingerprint_operation_input(idempotency_input)
    operation = await journal.create_operation(
        workflow_id="workflow-push",
        feature_id=None,
        child_workflow_id=None,
        repository_id=None,
        operation_type=ExternalOperationType.PUSH_BRANCH,
        idempotency_key=deterministic_operation_key(
            workflow_id="workflow-push",
            feature_id=None,
            child_workflow_id=None,
            repository_id=None,
            operation_type=ExternalOperationType.PUSH_BRANCH,
            logical_step="push_branch",
            input_fingerprint=fingerprint,
        ),
        input_fingerprint=fingerprint,
        safe_metadata={
            "logical_step": "push_branch",
            **idempotency_input,
            "expected_remote_before_sha": remote_before,
        },
    )
    await journal.transition_status(
        operation.operation_id, ExternalOperationStatus.STARTING, event_type="operation_starting"
    )
    await journal.record_attempt(operation.operation_id, provider="git")
    await asyncio.sleep(0.01)
    service = InterruptibleGitService(
        default_branch="main",
        cancellation_token=MockCancellationToken(),
        operation_executor=ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(workflow_id="workflow-push"),
        ),
    )

    result = await service.push(repository, "ai/recover-push")

    assert result.branch == "ai/recover-push"
    recovered = await journal.get(operation.operation_id)
    assert recovered.status is ExternalOperationStatus.SUCCEEDED
    assert recovered.external_reference == local_sha
    await database.dispose()


@pytest.mark.asyncio
async def test_successful_push_exit_requires_the_remote_to_match_local_head(tmp_path: Path) -> None:
    """Exit zero is insufficient when a concurrent update leaves the branch at another SHA."""

    class MismatchedRemoteRunner:
        def __init__(self) -> None:
            self.remote_reads = 0

        async def run(self, command: Any, *_args: Any, **_kwargs: Any) -> ProcessResult:
            argv = tuple(command)
            stdout = ""
            if argv == ("git", "rev-parse", "HEAD"):
                stdout = "local-reviewed-sha\n"
            elif argv[:2] == ("git", "ls-remote"):
                self.remote_reads += 1
                sha = "remote-before" if self.remote_reads == 1 else "remote-after-other-sha"
                stdout = f"{sha}\trefs/heads/ai/mismatch\n"
            return ProcessResult(
                command=argv,
                return_code=0,
                stdout=stdout,
                stderr="",
                duration_seconds=0.01,
            )

    service = InterruptibleGitService(
        default_branch="main",
        cancellation_token=MockCancellationToken(),
        process_runner=MismatchedRemoteRunner(),
    )

    with pytest.raises(GitAdapterError, match="remote branch does not point"):
        await service.push(tmp_path, "ai/mismatch")


class _LatePushFailureRunner:
    """Place the reviewed commit on the remote and then report the push as failed."""

    def __init__(self, *, remote_matches: bool) -> None:
        """Choose whether the failed push actually left the commit on the remote branch."""
        self._remote_matches = remote_matches
        self.remote_reads = 0
        self.pushes = 0

    async def run(self, command: Any, *_args: Any, **_kwargs: Any) -> ProcessResult:
        """Answer the adapter's Git calls, failing every push after its remote effect."""
        argv = tuple(command)
        stdout = ""
        return_code = 0
        if argv == ("git", "rev-parse", "HEAD"):
            stdout = "local-reviewed-sha\n"
        elif argv[:2] == ("git", "ls-remote"):
            self.remote_reads += 1
            landed = self._remote_matches and self.remote_reads > 1
            sha = "local-reviewed-sha" if landed else "remote-before"
            stdout = f"{sha}\trefs/heads/ai/late-failure\n"
        elif "push" in argv:
            self.pushes += 1
            return_code = 1
        return ProcessResult(
            command=argv,
            return_code=return_code,
            stdout=stdout,
            stderr="error: failed to write tracking ref" if return_code else "",
            duration_seconds=0.01,
        )


@pytest.mark.asyncio
async def test_a_push_that_placed_the_reviewed_commit_is_not_failed_for_its_exit_code(
    tmp_path: Path,
) -> None:
    """A push is judged by where the remote branch points, not by what the command reported.

    A push that lands the commit and then fails -- a flaky post-receive hook, a connection
    dropped while writing the tracking ref -- has done its whole job. Failing it cost the
    repository its pull request for a commit that was on the remote the entire time.
    """
    runner = _LatePushFailureRunner(remote_matches=True)
    service = InterruptibleGitService(
        default_branch="main",
        cancellation_token=MockCancellationToken(),
        process_runner=runner,
    )

    result = await service.push(tmp_path, "ai/late-failure")

    assert result.branch == "ai/late-failure"
    assert runner.pushes == 1


@pytest.mark.asyncio
async def test_a_push_that_did_not_reach_the_remote_still_fails(tmp_path: Path) -> None:
    """Accepting the effect must not accept a push whose commit never reached the remote."""
    service = InterruptibleGitService(
        default_branch="main",
        cancellation_token=MockCancellationToken(),
        process_runner=_LatePushFailureRunner(remote_matches=False),
    )

    with pytest.raises(GitAdapterError, match="git push failed"):
        await service.push(tmp_path, "ai/late-failure")


@pytest.mark.asyncio
async def test_a_failed_push_stays_replayable_rather_than_terminal(tmp_path: Path) -> None:
    """One dropped push must not deny a reviewed repository its branch for the whole feature.

    A terminal operation refuses to run again, so without a replay budget a single transient
    push failure meant the branch never reached the remote and the repository never got a
    pull request -- not on this run, and not on any resume of it either.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'push-retry.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    service = InterruptibleGitService(
        default_branch="main",
        cancellation_token=MockCancellationToken(),
        process_runner=_LatePushFailureRunner(remote_matches=False),
        operation_executor=ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(workflow_id="workflow-push-retry"),
        ),
    )

    with pytest.raises(GitAdapterError):
        await service.push(tmp_path, "ai/late-failure")

    operations = await journal.list_operations_for_workflow("workflow-push-retry")
    push_operations = [
        item for item in operations if item.operation_type is ExternalOperationType.PUSH_BRANCH
    ]
    assert [item.status for item in push_operations] == [
        ExternalOperationStatus.FAILED_RETRYABLE
    ], "a failed push must remain replayable so a resume can put the branch on the remote"
    await database.dispose()


@pytest.mark.asyncio
async def test_a_failed_branch_creation_stays_replayable_rather_than_terminal(
    tmp_path: Path,
) -> None:
    """Everything a repository publishes hangs off its branch, so creating it must be retryable."""

    class FailingBranchRunner:
        """Refuse to create the branch, the way a momentarily held lock file does."""

        async def run(self, command: Any, *_args: Any, **_kwargs: Any) -> ProcessResult:
            """Report every Git call as failed without leaving a branch behind."""
            return ProcessResult(
                command=tuple(command),
                return_code=1,
                stdout="",
                stderr="fatal: Unable to create index.lock: File exists",
                duration_seconds=0.01,
            )

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'branch-retry.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    service = InterruptibleGitService(
        default_branch="main",
        cancellation_token=MockCancellationToken(),
        process_runner=FailingBranchRunner(),
        operation_executor=ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(workflow_id="workflow-branch-retry"),
        ),
    )

    with pytest.raises(GitAdapterError):
        await service.create_branch(tmp_path, "ai/branch-retry", base_branch="main")

    operations = await journal.list_operations_for_workflow("workflow-branch-retry")
    branch_operations = [
        item for item in operations if item.operation_type is ExternalOperationType.CREATE_BRANCH
    ]
    assert [item.status for item in branch_operations] == [
        ExternalOperationStatus.FAILED_RETRYABLE
    ], "a failed branch creation must remain replayable so a resume can still publish"
    await database.dispose()


def _git(*args: str) -> None:
    """Execute a local Git setup command with test-friendly diagnostic output."""
    subprocess.run(("git", *args), check=True, capture_output=True, text=True)


def _git_output(*args: str) -> str:
    """Return one local Git metadata value for recovery assertions."""
    return subprocess.run(("git", *args), check=True, capture_output=True, text=True).stdout.strip()


@pytest.mark.asyncio
async def test_a_failed_pull_request_create_can_be_attempted_again(tmp_path: Path) -> None:
    """A provider blip must not journal the pull request as impossible to create ever again.

    A terminal operation refuses to run again, so the publisher's own retry and every later
    resume hit that refusal instead of the provider, and a reviewed, pushed branch could
    never get its pull request.
    """

    class BlipOnceService(MockGitHubService):
        """Fail the first create the way a provider blip does, then behave normally."""

        def __init__(self) -> None:
            """Start with the one failure this provider is going to report."""
            super().__init__()
            self.blipped = False

        def create_pull_request(self, repository: str, **kwargs: Any) -> Any:
            """Reject only the first attempt, leaving nothing behind on the provider."""
            if not self.blipped:
                self.blipped = True
                msg = "simulated provider blip"
                raise RuntimeError(msg)
            return super().create_pull_request(repository, **kwargs)

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'github-retry.db'}")
    await database.create_schema()
    service = JournaledGitHubService(
        BlipOnceService(),
        operation_executor=ExternalOperationExecutor(
            journal=ExternalOperationJournal(database),
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(
                workflow_id="workflow-pr-retry",
                feature_id="feature-pr-retry",
                repository_id="platform",
            ),
        ),
    )
    arguments: dict[str, Any] = {
        "title": "[AI] retry",
        "body": "Feature ID: feature-pr-retry",
        "source_branch": "ai/feature-pr-retry",
        "target_branch": "main",
        "draft": True,
    }

    with pytest.raises(RuntimeError):
        await service.create_pull_request("example/platform", **arguments)
    created = await service.create_pull_request("example/platform", **arguments)

    assert created.repository == "example/platform"
    assert created.source_branch == "ai/feature-pr-retry"
    await database.dispose()


@pytest.mark.asyncio
async def test_pull_request_creation_keeps_attempts_left_for_a_resume(tmp_path: Path) -> None:
    """A provider outage must not spend the whole replay budget inside one run.

    The publisher retries in place, and each of its attempts spends one journal attempt. An
    equal budget would be gone within seconds and leave a resume nothing to retry with.
    """

    class AlwaysFailingService(MockGitHubService):
        """Refuse every create the way a provider outage does."""

        def create_pull_request(self, repository: str, **kwargs: Any) -> Any:
            """Fail without leaving a pull request behind on the provider."""
            msg = "simulated provider outage"
            raise RuntimeError(msg)

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'github-budget.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    service = JournaledGitHubService(
        AlwaysFailingService(),
        operation_executor=ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(
                workflow_id="workflow-pr-budget",
                feature_id="feature-pr-budget",
                repository_id="platform",
            ),
        ),
    )
    # What GitHubPullRequestPublisher spends in one run before giving the repository up.
    in_run_attempts = 3

    for _ in range(in_run_attempts):
        with pytest.raises(Exception, match="outage"):
            await service.create_pull_request(
                "example/platform",
                title="[AI] budget",
                body="Feature ID: feature-pr-budget",
                source_branch="ai/feature-pr-budget",
                target_branch="main",
                draft=True,
            )

    operations = await journal.list_operations_for_workflow("workflow-pr-budget")
    assert [item.status for item in operations] == [ExternalOperationStatus.FAILED_RETRYABLE], (
        "a resume must still be able to attempt the pull request after an in-run outage"
    )
    await database.dispose()


@pytest.mark.asyncio
async def test_the_journaled_service_can_fetch_a_pull_request_back(tmp_path: Path) -> None:
    """Completion has to confirm a pull request exists, so the lookup must be reachable.

    Feature -052 created both pull requests and was still reported as failed, because the
    completion guard called a method the journaled wrapper did not expose and the
    AttributeError was recorded as 'the pull request could not be verified'.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'github-lookup.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    operations = ExternalOperationExecutor(
        journal=journal,
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(
            workflow_id="workflow-lookup", feature_id="feature-lookup", repository_id="platform"
        ),
    )
    service = JournaledGitHubService(MockGitHubService(), operation_executor=operations)
    created = await service.create_pull_request(
        "example/platform",
        title="[AI] lookup",
        body="Feature ID: feature-lookup",
        source_branch="ai/feature-lookup",
        target_branch="main",
        draft=True,
    )

    found = await service.find_pull_request(
        "example/platform",
        source_branch="ai/feature-lookup",
        target_branch="main",
        title="[AI] lookup",
    )

    assert found is not None
    assert found.number == created.number
    assert found.url == created.url
    await database.dispose()


@pytest.mark.asyncio
async def test_an_interrupted_coding_call_can_be_attempted_again(tmp_path: Path) -> None:
    """Coding gates everything downstream, so one interruption must not be terminal.

    The journal defaults to a single attempt, which records the first failure as terminal and
    refuses the operation on this run and on every resume. -076's backend was interrupted by
    an API restart mid-attempt and the feature then sat in `running_child_workflows` for hours
    with nothing able to move it: both of its coding operations were terminal at attempt 1 of
    1. Recovery already classifies this operation as workspace-local, so an interrupted one
    left nothing outside the checkout to reconcile.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'coding-replay.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    executor = ExternalOperationExecutor(
        journal=journal,
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(workflow_id="workflow-coding-replay"),
    )
    attempts = 0

    async def failing_once() -> tuple[str, OperationResult]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            msg = "interrupted mid-attempt"
            raise LLMAdapterError(msg)
        return "second", OperationResult(external_reference="resp_second")

    with pytest.raises(LLMAdapterError):
        await executor.run(
            operation_type=ExternalOperationType.RUN_CODING_EXECUTOR,
            logical_step="coding_and_file_changes",
            safe_input={"workspace_path": str(tmp_path)},
            action=failing_once,
            max_attempts=3,
        )

    replay = await executor.run(
        operation_type=ExternalOperationType.RUN_CODING_EXECUTOR,
        logical_step="coding_and_file_changes",
        safe_input={"workspace_path": str(tmp_path)},
        action=failing_once,
        max_attempts=3,
    )

    assert attempts == 2
    assert replay.value == "second"
    recorded = await journal.list_operations_for_workflow("workflow-coding-replay")
    assert recorded[0].max_attempts == 3
    await database.dispose()
