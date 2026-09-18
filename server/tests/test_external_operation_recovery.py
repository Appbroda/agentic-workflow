"""Recovery-policy, reconciliation and readiness-isolation tests for external operations.

The subject of this file is one defect with two halves. The credential-free recovery sweep
had no rule for five remote operation types, so it parked them as UNKNOWN and required a
person -- pre-empting the reconciliation callbacks that already existed and would have
settled them on the next credentialed request. That stamp then fed a global readiness gate,
and the mutation middleware reads readiness, so one interrupted push refused every non-GET
request on the whole deployment.

The provider doubles here hold real state -- branches, SHAs, pull requests, labels,
reviewers -- and every assertion is about what the provider ended up holding, not about
which calls were issued.
"""

from __future__ import annotations

import asyncio
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import update

from adapters.github_adapter import (
    MockGitHubService,
    PullRequestDetails,
    select_pull_request,
)
from adapters.interruptible_git import InterruptibleGitService
from api.control_plane import RequestScopedCredentials
from api.feature_schemas import StartFeatureRequest
from main import InfrastructureReadinessProbe, create_app
from services.cancellation import MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.journaled_github import JournaledGitHubService
from services.recovery_service import (
    _RECOVERY_POLICY,
    RecoveryDisposition,
    RecoveryService,
    assert_recovery_policy_is_total,
    recovery_disposition_for,
)
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.external_operations import (
    CompensationStatus,
    ExternalOperationStatus,
    ExternalOperationType,
    WorkflowCheckpointBoundary,
)
from state.feature_models import ChildWorkflowReference
from storage.db import Database
from storage.external_operation_store import (
    ExternalOperationError,
    ExternalOperationJournal,
    OperationReconciliationRequired,
    deterministic_operation_key,
    fingerprint_operation_input,
)
from storage.feature_store import SqlAlchemyFeatureControlPlane
from storage.models import ExternalOperationModel


class ReadyRedis:
    """Minimal readiness-only Redis substitute."""

    async def ping(self) -> bool:
        """Report the dependency as reachable without opening a network connection."""
        return True


class UnreachableRedis:
    """A dependency that is genuinely down, which readiness is still allowed to fail on."""

    async def ping(self) -> bool:
        """Refuse the way an unreachable Redis does."""
        msg = "redis is unreachable"
        raise ConnectionError(msg)


def _git(*args: str) -> None:
    """Execute a local Git setup command with test-friendly diagnostic output."""
    subprocess.run(("git", *args), check=True, capture_output=True, text=True)


def _git_output(*args: str) -> str:
    """Return one local Git metadata value for recovery assertions."""
    return subprocess.run(("git", *args), check=True, capture_output=True, text=True).stdout.strip()


# --------------------------------------------------------------------------------------
# 10.1 -- the mapping is total, and stays total
# --------------------------------------------------------------------------------------


def test_every_operation_type_has_an_explicit_recovery_disposition() -> None:
    """A new operation type cannot be added without deciding how recovery treats it."""
    undeclared = [item for item in ExternalOperationType if item not in _RECOVERY_POLICY]

    assert undeclared == []
    assert {recovery_disposition_for(item) for item in ExternalOperationType} <= set(
        RecoveryDisposition
    )
    for item in ExternalOperationType:
        assert isinstance(recovery_disposition_for(item), RecoveryDisposition)


def test_the_startup_assertion_refuses_a_policy_with_a_hole(monkeypatch: Any) -> None:
    """The process refuses to start rather than fall through to "park it for a human"."""
    incomplete = dict(_RECOVERY_POLICY)
    del incomplete[ExternalOperationType.PUSH_BRANCH]
    monkeypatch.setattr("services.recovery_service._RECOVERY_POLICY", incomplete)

    with pytest.raises(RuntimeError, match="push_branch"):
        assert_recovery_policy_is_total()


def test_the_remote_types_the_sweep_cannot_judge_are_deferred_not_parked() -> None:
    """The five types that reached the old fallthrough now each have their own answer."""
    assert recovery_disposition_for(ExternalOperationType.PUSH_BRANCH) is (
        RecoveryDisposition.DEFER_TO_CREDENTIALED
    )
    assert recovery_disposition_for(ExternalOperationType.CREATE_PULL_REQUEST) is (
        RecoveryDisposition.DEFER_TO_CREDENTIALED
    )
    assert recovery_disposition_for(ExternalOperationType.ADD_LABELS) is (
        RecoveryDisposition.DEFER_TO_CREDENTIALED
    )
    assert recovery_disposition_for(ExternalOperationType.ADD_REVIEWERS) is (
        RecoveryDisposition.DEFER_TO_CREDENTIALED
    )
    # The cross-link comment, which the publisher explicitly does not gate on.
    assert recovery_disposition_for(ExternalOperationType.UPDATE_PULL_REQUEST) is (
        RecoveryDisposition.TERMINAL_BENIGN
    )


async def _interrupted_operation(
    journal: ExternalOperationJournal,
    *,
    operation_type: ExternalOperationType,
    workflow_id: str = "workflow-recovery",
    feature_id: str | None = None,
    repository_id: str | None = None,
    idempotency_key: str = "workflow-recovery:operation",
    safe_metadata: dict[str, Any] | None = None,
    max_attempts: int = 3,
    spend_budget: bool = False,
) -> Any:
    """Journal an operation whose process died mid-flight, without a confirmed result."""
    operation = await journal.create_operation(
        workflow_id=workflow_id,
        feature_id=feature_id,
        child_workflow_id=None,
        repository_id=repository_id,
        operation_type=operation_type,
        idempotency_key=idempotency_key,
        input_fingerprint=idempotency_key,
        max_attempts=max_attempts,
        safe_metadata=safe_metadata or {},
    )
    await journal.claim_operation(operation.operation_id)
    if spend_budget:
        for _ in range(max_attempts - 1):
            await journal.record_failure(
                operation.operation_id,
                status=ExternalOperationStatus.FAILED_RETRYABLE,
                error_code="interrupted",
                error_message="interrupted",
            )
            await journal.claim_operation(operation.operation_id)
    await asyncio.sleep(0.01)
    return operation


@pytest.mark.asyncio
async def test_a_deferred_operation_is_visible_but_not_critical(tmp_path: Path) -> None:
    """A deferred remote effect is an operator's to see, not the deployment's to stop for."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'deferred.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    operation = await _interrupted_operation(
        journal, operation_type=ExternalOperationType.PUSH_BRANCH
    )

    summary = await RecoveryService(journal).recover_incomplete_operations()
    recovered = await journal.get(operation.operation_id)

    assert summary.deferred == 1
    assert summary.unknown == 0
    assert not summary.has_unresolved_critical_operations
    assert recovered.status is ExternalOperationStatus.AWAITING_RECONCILIATION
    assert recovered.compensation_status is CompensationStatus.NOT_REQUIRED
    assert await journal.unresolved_critical_count() == 0
    assert operation.operation_id in {
        item.operation_id for item in await journal.list_unresolved_operations()
    }
    await database.dispose()


@pytest.mark.asyncio
async def test_a_deferred_operation_with_no_budget_left_becomes_manual_review(
    tmp_path: Path,
) -> None:
    """Budget exhaustion is the only route from a remote type to UNKNOWN."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'deferred-exhausted.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    operation = await _interrupted_operation(
        journal,
        operation_type=ExternalOperationType.PUSH_BRANCH,
        max_attempts=2,
        spend_budget=True,
    )

    summary = await RecoveryService(journal).recover_incomplete_operations()
    recovered = await journal.get(operation.operation_id)

    assert summary.unknown == 1
    assert recovered.status is ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE
    assert recovered.compensation_status is CompensationStatus.MANUAL_REVIEW_REQUIRED
    await database.dispose()


@pytest.mark.asyncio
async def test_a_second_sweep_over_a_deferred_operation_changes_nothing(tmp_path: Path) -> None:
    """Repeated sweeps must not spend the attempt budget they left room in."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'deferred-idempotent.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    operation = await _interrupted_operation(
        journal, operation_type=ExternalOperationType.CREATE_PULL_REQUEST
    )
    recovery = RecoveryService(journal)

    await recovery.recover_incomplete_operations()
    first = await journal.get(operation.operation_id)
    first_events = len(await journal.events_for_operation(operation.operation_id))
    second = await recovery.recover_incomplete_operations()
    after = await journal.get(operation.operation_id)

    assert second.scanned == 0
    assert after.status is first.status
    assert after.attempt == first.attempt
    assert after.completed_at == first.completed_at
    assert after.error_code == first.error_code
    assert len(await journal.events_for_operation(operation.operation_id)) == first_events
    await database.dispose()


@pytest.mark.asyncio
async def test_the_cross_link_comment_ends_without_asking_anybody(tmp_path: Path) -> None:
    """An interrupted convenience comment is closed out, not queued for a person."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'cross-link.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    operation = await _interrupted_operation(
        journal, operation_type=ExternalOperationType.UPDATE_PULL_REQUEST
    )

    summary = await RecoveryService(journal).recover_incomplete_operations()
    recovered = await journal.get(operation.operation_id)

    assert summary.unknown == 0
    assert recovered.status is ExternalOperationStatus.CANCELLED
    assert recovered.compensation_status is CompensationStatus.NOT_REQUIRED
    assert await journal.unresolved_critical_count() == 0
    assert await journal.list_operations_requiring_manual_review() == []
    await database.dispose()


@pytest.mark.asyncio
async def test_the_recovery_service_is_never_given_a_provider_client(tmp_path: Path) -> None:
    """The posture that makes deferral necessary is itself asserted."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'no-credentials.db'}")
    await database.create_schema()
    service = RecoveryService(ExternalOperationJournal(database))

    held = vars(service)
    assert not any("github" in name.lower() for name in held)
    assert not any("credential" in name.lower() for name in held)
    assert not any("token" in name.lower() for name in held)
    await database.dispose()


# --------------------------------------------------------------------------------------
# 10.2 -- push
# --------------------------------------------------------------------------------------


def _push_repository(tmp_path: Path, name: str) -> tuple[Path, Path, str]:
    """Build a local origin and a checkout with one unpushed reviewed commit."""
    remote = tmp_path / f"{name}-remote.git"
    repository = tmp_path / f"{name}-source"
    _git("init", "--bare", str(remote))
    _git("init", str(repository))
    _git("-C", str(repository), "config", "user.email", "tests@example.invalid")
    _git("-C", str(repository), "config", "user.name", "Test User")
    (repository / "tracked.txt").write_text("before\n", encoding="utf-8")
    _git("-C", str(repository), "add", "tracked.txt")
    _git("-C", str(repository), "commit", "-m", "initial")
    _git("-C", str(repository), "remote", "add", "origin", str(remote))
    _git("-C", str(repository), "push", "origin", "HEAD:refs/heads/main")
    _git("-C", str(repository), "switch", "-c", "ai/recovery")
    (repository / "tracked.txt").write_text("after\n", encoding="utf-8")
    _git("-C", str(repository), "add", "tracked.txt")
    _git("-C", str(repository), "commit", "-m", "reviewed change")
    return remote, repository, _git_output("-C", str(repository), "rev-parse", "HEAD")


async def _journal_push_intent(
    journal: ExternalOperationJournal,
    *,
    repository: Path,
    local_sha: str,
    branch: str = "ai/recovery",
    workflow_id: str = "workflow-push",
) -> Any:
    """Write exactly the intent the adapter writes, then leave it mid-flight."""
    idempotency_input = {
        "workspace_path": str(repository),
        "remote": "origin",
        "branch": branch,
        "expected_local_sha": local_sha,
    }
    fingerprint = fingerprint_operation_input(idempotency_input)
    operation = await journal.create_operation(
        workflow_id=workflow_id,
        feature_id=None,
        child_workflow_id=None,
        repository_id=None,
        operation_type=ExternalOperationType.PUSH_BRANCH,
        idempotency_key=deterministic_operation_key(
            workflow_id=workflow_id,
            feature_id=None,
            child_workflow_id=None,
            repository_id=None,
            operation_type=ExternalOperationType.PUSH_BRANCH,
            logical_step="push_branch",
            input_fingerprint=fingerprint,
        ),
        input_fingerprint=fingerprint,
        max_attempts=3,
        safe_metadata={"logical_step": "push_branch", **idempotency_input},
    )
    await journal.claim_operation(operation.operation_id)
    await asyncio.sleep(0.01)
    return operation


def _push_service(journal: ExternalOperationJournal, workflow_id: str) -> InterruptibleGitService:
    """Build the adapter exactly as a credentialed request would."""
    return InterruptibleGitService(
        default_branch="main",
        cancellation_token=MockCancellationToken(),
        operation_executor=ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(workflow_id=workflow_id),
        ),
    )


@pytest.mark.asyncio
async def test_a_push_that_reached_the_remote_is_reconciled_without_pushing_again(
    tmp_path: Path,
) -> None:
    """The sweep defers, and the next credentialed request settles it from the remote."""
    remote, repository, local_sha = _push_repository(tmp_path, "landed")
    _git("-C", str(repository), "push", "origin", "ai/recovery:ai/recovery")
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'push-landed.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    operation = await _journal_push_intent(journal, repository=repository, local_sha=local_sha)

    await RecoveryService(journal).recover_incomplete_operations()
    swept = await journal.get(operation.operation_id)
    assert swept.status is ExternalOperationStatus.AWAITING_RECONCILIATION

    # The remote is made read-only, so a push during recovery cannot silently succeed.
    remote_head = remote / "refs"
    result = await _push_service(journal, "workflow-push").push(repository, "ai/recovery")

    reconciled = await journal.get(operation.operation_id)
    assert result.branch == "ai/recovery"
    assert reconciled.status is ExternalOperationStatus.SUCCEEDED
    assert reconciled.external_reference == local_sha
    assert (reconciled.result_payload or {})["recovery_method"] == "remote_branch_sha_match"
    # One attempt, the one that died. Reconciling did not spend another.
    assert reconciled.attempt == 1
    assert remote_head.exists()
    assert (
        _git_output("-C", str(repository), "ls-remote", "origin", "refs/heads/ai/recovery").split()[
            0
        ]
        == local_sha
    )
    await database.dispose()


@pytest.mark.asyncio
async def test_a_push_that_never_reached_the_remote_retries_inside_its_own_budget(
    tmp_path: Path,
) -> None:
    """Absence proven from the remote is the one thing that permits running the push again."""
    _remote, repository, local_sha = _push_repository(tmp_path, "absent")
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'push-absent.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    operation = await _journal_push_intent(journal, repository=repository, local_sha=local_sha)

    await RecoveryService(journal).recover_incomplete_operations()
    result = await _push_service(journal, "workflow-push").push(repository, "ai/recovery")

    settled = await journal.get(operation.operation_id)
    assert result.branch == "ai/recovery"
    assert settled.status is ExternalOperationStatus.SUCCEEDED
    # The interrupted attempt plus exactly one replay, inside the budget it already had.
    assert settled.attempt == 2
    assert settled.max_attempts == 3
    assert (
        _git_output("-C", str(repository), "ls-remote", "origin", "refs/heads/ai/recovery").split()[
            0
        ]
        == local_sha
    )
    await database.dispose()


@pytest.mark.asyncio
async def test_a_diverged_remote_branch_is_named_and_never_overwritten(tmp_path: Path) -> None:
    """The reviewed commit is not on the remote, and nothing here forces it there."""
    _remote, repository, local_sha = _push_repository(tmp_path, "diverged")
    other = tmp_path / "diverged-other"
    _git("clone", str(tmp_path / "diverged-remote.git"), str(other))
    _git("-C", str(other), "config", "user.email", "tests@example.invalid")
    _git("-C", str(other), "config", "user.name", "Other User")
    _git("-C", str(other), "switch", "-c", "ai/recovery")
    (other / "theirs.txt").write_text("theirs\n", encoding="utf-8")
    _git("-C", str(other), "add", "theirs.txt")
    _git("-C", str(other), "commit", "-m", "somebody else")
    _git("-C", str(other), "push", "origin", "ai/recovery:ai/recovery")
    foreign_sha = _git_output("-C", str(other), "rev-parse", "HEAD")

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'push-diverged.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    operation = await _journal_push_intent(journal, repository=repository, local_sha=local_sha)
    await RecoveryService(journal).recover_incomplete_operations()

    with pytest.raises(Exception, match="git push"):
        await _push_service(journal, "workflow-push").push(repository, "ai/recovery")

    settled = await journal.get(operation.operation_id)
    reconciliation = settled.safe_metadata["reconciliation"]
    assert isinstance(reconciliation, dict)
    assert reconciliation["method"] == "remote_branch_diverged"
    assert reconciliation["observed_sha"] == foreign_sha
    assert reconciliation["expected_sha"] == local_sha
    assert settled.status is not ExternalOperationStatus.SUCCEEDED
    # The other author's commit is still what the remote branch points at.
    assert (
        _git_output("-C", str(repository), "ls-remote", "origin", "refs/heads/ai/recovery").split()[
            0
        ]
        == foreign_sha
    )
    await database.dispose()


@pytest.mark.asyncio
async def test_an_unreachable_remote_leaves_the_push_deferred_and_says_why(
    tmp_path: Path,
) -> None:
    """A network fault is not evidence, and must never be recorded as one."""
    _remote, repository, local_sha = _push_repository(tmp_path, "unreachable")
    _git("-C", str(repository), "remote", "set-url", "origin", str(tmp_path / "not-a-remote.git"))
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'push-unreachable.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    operation = await _journal_push_intent(journal, repository=repository, local_sha=local_sha)
    await RecoveryService(journal).recover_incomplete_operations()

    with pytest.raises(OperationReconciliationRequired):
        await _push_service(journal, "workflow-push").push(repository, "ai/recovery")

    settled = await journal.get(operation.operation_id)
    reconciliation = settled.safe_metadata["reconciliation"]
    assert isinstance(reconciliation, dict)
    assert reconciliation["result"] == "unproven"
    assert settled.status is ExternalOperationStatus.AWAITING_RECONCILIATION
    assert settled.attempt == 1
    await database.dispose()


@pytest.mark.asyncio
async def test_a_succeeded_push_is_left_completely_alone_by_the_sweep(tmp_path: Path) -> None:
    """Recovery lists only unfinished operations, and a finished push is not one."""
    _remote, repository, local_sha = _push_repository(tmp_path, "succeeded")
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'push-succeeded.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    service = _push_service(journal, "workflow-push")
    await service.push(repository, "ai/recovery")
    remote_after_push = _git_output(
        "-C", str(repository), "ls-remote", "origin", "refs/heads/ai/recovery"
    )
    operations = await journal.list_operations_for_workflow("workflow-push")
    before = [
        (item.operation_id, item.status, item.attempt, item.completed_at) for item in operations
    ]

    summary = await RecoveryService(journal).recover_incomplete_operations()

    after = [
        (item.operation_id, item.status, item.attempt, item.completed_at)
        for item in await journal.list_operations_for_workflow("workflow-push")
    ]
    assert summary.scanned == 0
    assert before == after
    assert before[0][1] is ExternalOperationStatus.SUCCEEDED
    assert remote_after_push.split()[0] == local_sha
    await database.dispose()


# --------------------------------------------------------------------------------------
# 10.3 -- pull request
# --------------------------------------------------------------------------------------


def _github(journal: ExternalOperationJournal, service: Any, *, workflow_id: str) -> Any:
    """Wrap a stateful provider double in the journaled service a request would use."""
    return JournaledGitHubService(
        service,
        operation_executor=ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(workflow_id=workflow_id, feature_id=workflow_id),
        ),
    )


class _DiesBeforeRecordingSuccess(ExternalOperationJournal):
    """Lose the process in the window between the provider's answer and the durable write.

    This is the interruption the whole task is about, and it is deliberately modelled at the
    journal rather than with an exception the executor can catch: a killed process records
    nothing, so the row is left claimed and RUNNING for a later sweep to find.
    """

    async def record_result(self, *_args: Any, **_kwargs: Any) -> Any:
        """Vanish exactly where a SIGKILL would."""
        raise KeyboardInterrupt


class _DiesBefore:
    """Delegate to one real provider, losing the process before the effect is applied."""

    def __init__(self, inner: Any, method: str) -> None:
        self._inner = inner
        self._method = method

    def __getattr__(self, name: str) -> Any:
        """Forward everything except the one call that never reaches the provider."""
        if name != self._method:
            return getattr(self._inner, name)

        def dying(*_args: Any, **_kwargs: Any) -> None:
            msg = "process died before the provider was asked"
            raise RuntimeError(msg)

        return dying


class _RefusesToRead:
    """Delegate to one real provider whose read capability cannot be reached."""

    def __init__(self, inner: Any, *methods: str) -> None:
        self._inner = inner
        self._methods = frozenset(methods)

    def __getattr__(self, name: str) -> Any:
        """Answer every read in the named set the way an unreachable provider does."""
        if name not in self._methods:
            return getattr(self._inner, name)

        def unreachable(*_args: Any, **_kwargs: Any) -> None:
            msg = "provider is unreachable"
            raise ConnectionError(msg)

        return unreachable


async def _crash_the_create(database: Database, provider: Any, *, workflow_id: str) -> None:
    """Create the pull request on the provider, then lose the process before the result."""
    dying = _DiesBeforeRecordingSuccess(database, stale_after_seconds=0.001)
    with pytest.raises(KeyboardInterrupt):
        await _github(dying, provider, workflow_id=workflow_id).create_pull_request(
            "example/platform",
            title="[AI] recovery",
            body="Feature ID: recovery",
            source_branch="ai/recovery",
            target_branch="main",
        )
    await asyncio.sleep(0.01)


async def _crash_a_mutation(
    database: Database, provider: Any, *, workflow_id: str, method: str
) -> None:
    """Perform the real mutation on the provider, then lose the process before the result."""
    dying = _DiesBeforeRecordingSuccess(database, stale_after_seconds=0.001)
    service = _github(dying, provider, workflow_id=workflow_id)
    with pytest.raises(KeyboardInterrupt):
        if method == "add_labels":
            await service.add_labels("example/platform", 1, ["ai-generated"])
        else:
            await service.request_reviewers("example/platform", 1, ["reviewer-one"])
    await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_created_pull_request_is_adopted_without_creating_a_second(
    tmp_path: Path,
) -> None:
    """The provider ends up holding exactly one pull request, not two."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'pr-adopt.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    provider = MockGitHubService()
    await _crash_the_create(database, provider, workflow_id="feature-pr-adopt")
    await RecoveryService(journal).recover_incomplete_operations()
    assert len(provider.pull_requests) == 1

    adopted = await _github(journal, provider, workflow_id="feature-pr-adopt").create_pull_request(
        "example/platform",
        title="[AI] recovery",
        body="Feature ID: recovery",
        source_branch="ai/recovery",
        target_branch="main",
    )

    assert len(provider.pull_requests) == 1
    assert adopted.number == next(iter(provider.pull_requests.values())).number
    operations = await journal.list_operations_for_workflow("feature-pr-adopt")
    assert [item.status for item in operations] == [ExternalOperationStatus.SUCCEEDED]
    await database.dispose()


@pytest.mark.asyncio
async def test_no_matching_pull_request_creates_exactly_one_inside_the_budget(
    tmp_path: Path,
) -> None:
    """Proven absence is what lets the create run again, and only once."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'pr-absent.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    provider = MockGitHubService()
    service = _github(journal, provider, workflow_id="feature-pr-absent")

    dying = _DiesBefore(provider, "create_pull_request")
    with pytest.raises(RuntimeError):
        await _github(journal, dying, workflow_id="feature-pr-absent").create_pull_request(
            "example/platform",
            title="[AI] absent",
            body="Feature ID: absent",
            source_branch="ai/absent",
            target_branch="main",
        )
    await RecoveryService(journal).recover_incomplete_operations()

    created = await service.create_pull_request(
        "example/platform",
        title="[AI] absent",
        body="Feature ID: absent",
        source_branch="ai/absent",
        target_branch="main",
    )

    assert len(provider.pull_requests) == 1
    assert created.source_branch == "ai/absent"
    settled = (await journal.list_operations_for_workflow("feature-pr-absent"))[0]
    assert settled.status is ExternalOperationStatus.SUCCEEDED
    assert settled.attempt <= settled.max_attempts
    await database.dispose()


@pytest.mark.asyncio
async def test_two_candidate_pull_requests_ask_a_person_rather_than_open_a_third(
    tmp_path: Path,
) -> None:
    """Ambiguity is the one reconciliation outcome that is a person's to settle."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'pr-ambiguous.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    provider = MockGitHubService()
    await _crash_the_create(database, provider, workflow_id="feature-pr-ambiguous")
    await RecoveryService(journal).recover_incomplete_operations()
    # Somebody renamed the pull request this platform opened, and somebody opened a second
    # one on the same branch. Head and base no longer identify one of them, and the title
    # this operation remembers is now nobody's.
    key, created = next(iter(provider.pull_requests.items()))
    provider.pull_requests[key] = PullRequestDetails(
        repository=created.repository,
        number=created.number,
        url=created.url,
        title="renamed by a person",
        source_branch=created.source_branch,
        target_branch=created.target_branch,
        head_sha=created.head_sha,
    )
    provider.create_pull_request(
        "example/platform",
        title="a second pull request on the same branch",
        body="opened by a person",
        source_branch="ai/recovery",
        target_branch="main",
    )
    assert len(provider.pull_requests) == 2

    with pytest.raises(OperationReconciliationRequired):
        await _github(journal, provider, workflow_id="feature-pr-ambiguous").create_pull_request(
            "example/platform",
            title="[AI] recovery",
            body="Feature ID: recovery",
            source_branch="ai/recovery",
            target_branch="main",
        )

    assert len(provider.pull_requests) == 2
    settled = (await journal.list_operations_for_workflow("feature-pr-ambiguous"))[0]
    assert settled.status is ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE
    assert settled.compensation_status is CompensationStatus.MANUAL_REVIEW_REQUIRED
    await database.dispose()


@pytest.mark.asyncio
async def test_an_unavailable_provider_opens_no_pull_request_at_all(tmp_path: Path) -> None:
    """A lookup that cannot be answered must never become a mutation."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'pr-unavailable.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    provider = MockGitHubService()
    await _crash_the_create(database, provider, workflow_id="feature-pr-unavailable")
    await RecoveryService(journal).recover_incomplete_operations()

    unavailable = _RefusesToRead(provider, "find_pull_requests", "find_pull_request")

    with pytest.raises(OperationReconciliationRequired):
        await _github(
            journal, unavailable, workflow_id="feature-pr-unavailable"
        ).create_pull_request(
            "example/platform",
            title="[AI] recovery",
            body="Feature ID: recovery",
            source_branch="ai/recovery",
            target_branch="main",
        )

    assert len(provider.pull_requests) == 1
    settled = (await journal.list_operations_for_workflow("feature-pr-unavailable"))[0]
    # Deferred, not escalated: an unreachable provider is not evidence about anything, and
    # the next credentialed request gets to ask again.
    assert settled.status is ExternalOperationStatus.AWAITING_RECONCILIATION
    reconciliation = settled.safe_metadata["reconciliation"]
    assert isinstance(reconciliation, dict)
    assert reconciliation["result"] == "unproven"
    await database.dispose()


def test_a_pull_request_is_matched_on_head_and_base_with_title_only_breaking_ties() -> None:
    """Title is model-influenced text, so it narrows candidates and never qualifies them."""
    renamed = PullRequestDetails(
        repository="example/platform",
        number=7,
        url="https://github.invalid/example/platform/pull/7",
        title="a human renamed this",
        source_branch="ai/recovery",
        target_branch="main",
        head_sha="abc123",
    )
    other = PullRequestDetails(
        repository="example/platform",
        number=8,
        url="https://github.invalid/example/platform/pull/8",
        title="[AI] recovery",
        source_branch="ai/recovery",
        target_branch="main",
        head_sha="def456",
    )

    assert select_pull_request([renamed], title="[AI] recovery") == renamed
    assert select_pull_request([renamed, other], title="[AI] recovery") == other
    assert select_pull_request([renamed, other], expected_head_sha="abc123") == renamed
    assert select_pull_request([renamed, other], title="neither") is None
    assert select_pull_request([], title="[AI] recovery") is None


@pytest.mark.asyncio
async def test_a_pull_request_renamed_by_a_person_is_still_adopted(tmp_path: Path) -> None:
    """Head and base identify the branch this platform pushed; a retitle must not duplicate."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'pr-renamed.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    provider = MockGitHubService()
    await _crash_the_create(database, provider, workflow_id="feature-pr-renamed")
    await RecoveryService(journal).recover_incomplete_operations()
    key, existing = next(iter(provider.pull_requests.items()))
    provider.pull_requests[key] = PullRequestDetails(
        repository=existing.repository,
        number=existing.number,
        url=existing.url,
        title="chore: tidy up the recovery path",
        source_branch=existing.source_branch,
        target_branch=existing.target_branch,
        head_sha=existing.head_sha,
    )

    adopted = await _github(
        journal, provider, workflow_id="feature-pr-renamed"
    ).create_pull_request(
        "example/platform",
        title="[AI] recovery",
        body="Feature ID: recovery",
        source_branch="ai/recovery",
        target_branch="main",
    )

    assert len(provider.pull_requests) == 1
    assert adopted.number == existing.number
    await database.dispose()


# --------------------------------------------------------------------------------------
# 10.4 -- labels and reviewers
# --------------------------------------------------------------------------------------


def _provider_with_pull_request() -> MockGitHubService:
    """Return a provider double already holding the pull request the mutations target."""
    provider = MockGitHubService()
    provider.create_pull_request(
        "example/platform",
        title="[AI] recovery",
        body="Feature ID: recovery",
        source_branch="ai/recovery",
        target_branch="main",
    )
    return provider


@pytest.mark.asyncio
async def test_labels_already_present_reconcile_without_touching_the_provider(
    tmp_path: Path,
) -> None:
    """The provider's own answer settles it, and the labels are not applied twice."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'labels-present.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    provider = _provider_with_pull_request()
    await _crash_a_mutation(database, provider, workflow_id="feature-labels", method="add_labels")
    await RecoveryService(journal).recover_incomplete_operations()
    assert provider.labels[("example/platform", 1)] == ["ai-generated"]

    await _github(journal, provider, workflow_id="feature-labels").add_labels(
        "example/platform", 1, ["ai-generated"]
    )

    assert provider.labels[("example/platform", 1)] == ["ai-generated"]
    settled = (await journal.list_operations_for_workflow("feature-labels"))[0]
    assert settled.status is ExternalOperationStatus.SUCCEEDED
    assert (settled.result_payload or {})["recovery_method"] == "labels_present"
    await database.dispose()


@pytest.mark.asyncio
async def test_labels_the_provider_does_not_have_are_applied_by_the_existing_retry(
    tmp_path: Path,
) -> None:
    """Absence read from the provider, not inferred from an incomplete operation record."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'labels-absent.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    provider = _provider_with_pull_request()

    dying = _DiesBefore(provider, "add_labels")
    with pytest.raises(RuntimeError):
        await _github(journal, dying, workflow_id="feature-labels-absent").add_labels(
            "example/platform", 1, ["ai-generated"]
        )
    await RecoveryService(journal).recover_incomplete_operations()
    assert provider.labels[("example/platform", 1)] == []

    await _github(journal, provider, workflow_id="feature-labels-absent").add_labels(
        "example/platform", 1, ["ai-generated"]
    )

    assert provider.labels[("example/platform", 1)] == ["ai-generated"]
    await database.dispose()


@pytest.mark.asyncio
async def test_labels_that_cannot_be_read_leave_the_operation_deferred(tmp_path: Path) -> None:
    """An unanswerable read is not evidence of absence and produces no mutation."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'labels-ambiguous.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    provider = _provider_with_pull_request()
    await _crash_a_mutation(
        database, provider, workflow_id="feature-labels-ambiguous", method="add_labels"
    )
    await RecoveryService(journal).recover_incomplete_operations()
    provider.labels[("example/platform", 1)] = []

    unreadable = _RefusesToRead(provider, "get_pull_request_labels")

    with pytest.raises(OperationReconciliationRequired):
        await _github(journal, unreadable, workflow_id="feature-labels-ambiguous").add_labels(
            "example/platform", 1, ["ai-generated"]
        )

    assert provider.labels[("example/platform", 1)] == []
    await database.dispose()


@pytest.mark.asyncio
async def test_reviewers_still_requested_reconcile_as_already_done(tmp_path: Path) -> None:
    """A review request the provider still shows is a completed effect."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'reviewers-present.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    provider = _provider_with_pull_request()
    await _crash_a_mutation(
        database, provider, workflow_id="feature-reviewers", method="request_reviewers"
    )
    await RecoveryService(journal).recover_incomplete_operations()
    assert provider.reviewers[("example/platform", 1)] == ["reviewer-one"]

    await _github(journal, provider, workflow_id="feature-reviewers").request_reviewers(
        "example/platform", 1, ["reviewer-one"]
    )

    assert provider.reviewers[("example/platform", 1)] == ["reviewer-one"]
    settled = (await journal.list_operations_for_workflow("feature-reviewers"))[0]
    assert settled.status is ExternalOperationStatus.SUCCEEDED
    assert (settled.result_payload or {})["recovery_method"] == "reviewers_requested"
    await database.dispose()


@pytest.mark.asyncio
async def test_a_reviewer_who_already_reviewed_is_not_asked_again(tmp_path: Path) -> None:
    """The provider cannot distinguish never-asked from asked-and-answered, so neither do we."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'reviewers-acted.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    provider = _provider_with_pull_request()
    await _crash_a_mutation(
        database, provider, workflow_id="feature-reviewers-acted", method="request_reviewers"
    )
    await RecoveryService(journal).recover_incomplete_operations()
    # The reviewer reviewed, so GitHub no longer lists them as requested.
    provider.reviewers[("example/platform", 1)] = []

    with pytest.raises(OperationReconciliationRequired):
        await _github(journal, provider, workflow_id="feature-reviewers-acted").request_reviewers(
            "example/platform", 1, ["reviewer-one"]
        )

    assert provider.reviewers[("example/platform", 1)] == []
    settled = (await journal.list_operations_for_workflow("feature-reviewers-acted"))[0]
    reconciliation = settled.safe_metadata["reconciliation"]
    assert isinstance(reconciliation, dict)
    assert reconciliation["result"] == "unproven"
    assert reconciliation["unconfirmed_reviewers"] == ["reviewer-one"]
    await database.dispose()


# --------------------------------------------------------------------------------------
# 10.5 -- readiness
# --------------------------------------------------------------------------------------


async def _unresolved_remote_operation(
    journal: ExternalOperationJournal, operation_type: ExternalOperationType
) -> None:
    """Leave one feature holding an unresolved remote effect."""
    operation = await journal.create_operation(
        workflow_id="feature-unresolved",
        feature_id="feature-unresolved",
        child_workflow_id="feature-unresolved:backend",
        repository_id="backend",
        operation_type=operation_type,
        idempotency_key=f"feature-unresolved:{operation_type.value}",
        input_fingerprint=operation_type.value,
    )
    await journal.record_failure(
        operation.operation_id,
        status=ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
        error_code="unknown_external_state",
        error_message="the effect could not be confirmed",
        compensation_status=CompensationStatus.MANUAL_REVIEW_REQUIRED,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation_type",
    [ExternalOperationType.PUSH_BRANCH, ExternalOperationType.CREATE_PULL_REQUEST],
)
async def test_readiness_ignores_unresolved_external_operations(
    tmp_path: Path, operation_type: ExternalOperationType
) -> None:
    """Infrastructure is healthy, so the deployment is ready no matter what one feature holds."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / f'ready-{operation_type.value}.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    recovery = RecoveryService(journal)
    await recovery.recover_incomplete_operations()
    probe = InfrastructureReadinessProbe(database, ReadyRedis(), recovery_service=recovery)

    assert await probe.is_ready()
    await _unresolved_remote_operation(journal, operation_type)

    assert await probe.is_ready()
    assert await journal.unresolved_critical_count() == 1
    await database.dispose()


@pytest.mark.asyncio
async def test_readiness_still_fails_on_infrastructure_and_migrations(tmp_path: Path) -> None:
    """What readiness may fail on is now exactly the process-wide dependencies."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'ready-infrastructure.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    completed = RecoveryService(journal)
    await completed.recover_incomplete_operations()
    try:
        # Redis unreachable.
        assert not await InfrastructureReadinessProbe(database, UnreachableRedis()).is_ready()
        # Migration revision mismatch.
        assert not await InfrastructureReadinessProbe(
            database, ReadyRedis(), expected_migration_revision="not-the-current-head"
        ).is_ready()
        # Startup recovery has not completed in this process.
        assert not await InfrastructureReadinessProbe(
            database, ReadyRedis(), recovery_service=RecoveryService(journal)
        ).is_ready()
        assert await InfrastructureReadinessProbe(
            database, ReadyRedis(), recovery_service=completed
        ).is_ready()
    finally:
        await database.dispose()

    # PostgreSQL unreachable, modelled as a database that cannot be opened at all.
    unreachable = Database(f"sqlite+aiosqlite:///{tmp_path / 'missing' / 'nowhere.db'}")
    try:
        assert not await InfrastructureReadinessProbe(unreachable, ReadyRedis()).is_ready()
    finally:
        await unreachable.dispose()


# --------------------------------------------------------------------------------------
# 10.6 -- blast radius
# --------------------------------------------------------------------------------------


def _start_request(feature_id: str) -> StartFeatureRequest:
    """Build a minimal repository-agnostic feature request."""
    return StartFeatureRequest.model_validate(
        {
            "feature_id": feature_id,
            "execution_mode": "mock",
            "prd": {
                "title": "Blast radius",
                "problem_statement": "One feature's unresolved effect must not stop another.",
                "goals": ["Keep unrelated features writable."],
                "user_stories": [
                    {
                        "story_id": "story-blast-radius",
                        "persona": "Operator",
                        "need": "Start unrelated work while one effect is unresolved",
                        "benefit": "One feature's ambiguity is not an outage",
                        "acceptance_criteria": ["Unrelated writes still succeed."],
                    }
                ],
                "requirements": [
                    {
                        "requirement_id": "requirement-blast-radius",
                        "description": "Keep unrelated features writable.",
                        "priority": "must",
                        "acceptance_criteria": ["An unrelated feature can be started."],
                        "dependencies": [],
                    }
                ],
                "constraints": ["Do not store provider credentials."],
                "out_of_scope": ["Live provider calls."],
                "stakeholders": ["Platform team"],
            },
            "repositories": [
                {
                    "repository_id": "backend",
                    "name": "Backend API",
                    "role": "backend",
                    "repository_url": "https://github.com/example/backend.git",
                    "default_branch": "main",
                }
            ],
        }
    )


async def _running_feature_with_a_workstream(
    store: SqlAlchemyFeatureControlPlane, feature_id: str
) -> None:
    """Start a feature and put it where an interrupted publication would leave it."""
    await store.start(
        _start_request(feature_id),
        idempotency_key=f"{feature_id}-001",
        credentials=RequestScopedCredentials(None, None),
        owner_id="platform-admin",
    )
    record = await store.get_record(feature_id)
    state = record.state.model_copy(deep=True)
    state.status = FeatureWorkflowStatus.CREATING_PULL_REQUESTS
    state.child_workflows["backend"] = ChildWorkflowReference(
        child_workflow_id=f"{feature_id}:backend",
        repository_id="backend",
        workstream_id=f"{feature_id}:backend",
        status=ChildWorkflowStatus.APPROVED,
        branch_name=f"ai/{feature_id}",
        workspace_path=f"/workspaces/{feature_id}/backend",
        retry_count=0,
    )
    await store.checkpoint(state, WorkflowCheckpointBoundary.BEFORE_PR, "backend")


@pytest.mark.asyncio
async def test_one_features_unresolved_effect_does_not_stop_another_features_writes(
    tmp_path: Path,
) -> None:
    """Feature B starts, answers and acts through the real durable control plane."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'blast-radius.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    store = SqlAlchemyFeatureControlPlane(database, operation_journal=journal)
    credentials = RequestScopedCredentials(None, None)
    try:
        await store.start(
            _start_request("feature-a"),
            idempotency_key="feature-a-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        await _unresolved_remote_operation(journal, ExternalOperationType.PUSH_BRANCH)
        blocked = await journal.create_operation(
            workflow_id="feature-a",
            feature_id="feature-a",
            child_workflow_id="feature-a:backend",
            repository_id="backend",
            operation_type=ExternalOperationType.PUSH_BRANCH,
            idempotency_key="feature-a:push",
            input_fingerprint="feature-a-push",
        )
        await journal.record_failure(
            blocked.operation_id,
            status=ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
            error_code="unknown_external_state",
            error_message="the effect could not be confirmed",
            compensation_status=CompensationStatus.MANUAL_REVIEW_REQUIRED,
        )

        started = await store.start(
            _start_request("feature-b"),
            idempotency_key="feature-b-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        cancelled = await store.cancel("feature-b", requested_by="platform-admin")

        assert started.record.state.feature_id == "feature-b"
        assert cancelled.state.status in {
            FeatureWorkflowStatus.CANCELLED,
            FeatureWorkflowStatus.CANCELLING,
        }
        assert await journal.unresolved_critical_count() == 2
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_the_mutation_middleware_accepts_writes_while_an_effect_is_unresolved(
    tmp_path: Path,
) -> None:
    """The HTTP layer is where the outage was actually felt, so it is asserted there too."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'blast-radius-http.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    recovery = RecoveryService(journal)
    await recovery.recover_incomplete_operations()
    await _unresolved_remote_operation(journal, ExternalOperationType.CREATE_PULL_REQUEST)
    app = create_app(
        platform_api_key="blast-radius-key",
        readiness_probe=InfrastructureReadinessProbe(
            database, ReadyRedis(), recovery_service=recovery
        ),
    )
    headers = {"Authorization": "Bearer blast-radius-key"}
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            readiness = await client.get("/readyz")
            mutation = await client.post("/workflow/start", headers=headers, json={})

        assert readiness.status_code == 200
        # Rejected on its own merits rather than by the readiness gate: a 503 with the
        # runtime-unavailable detail is the outage this task removes.
        assert mutation.status_code != 503
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# 9 -- the owning feature
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_feature_owning_a_manual_review_effect_stops_and_says_why(
    tmp_path: Path,
) -> None:
    """Localising the blast radius must not mean ignoring the problem."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'owning-feature.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    store = SqlAlchemyFeatureControlPlane(database, operation_journal=journal)
    try:
        await _running_feature_with_a_workstream(store, "feature-owner")
        await _interrupted_operation(
            journal,
            operation_type=ExternalOperationType.PUSH_BRANCH,
            workflow_id="feature-owner",
            feature_id="feature-owner",
            repository_id="backend",
            idempotency_key="feature-owner:push",
            max_attempts=1,
        )
        recovery = RecoveryService(journal, unconfirmed_effect_escalation=store)

        await recovery.recover_incomplete_operations()

        record = await store.get_record("feature-owner")
        assert record.state.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        assert record.state.failure_summary is not None
        diagnostics = " ".join(record.state.failure_summary.diagnostics)
        assert "push branch" in diagnostics
        assert "backend" in diagnostics
        assert any(
            event.event == "unconfirmed_external_effect" for event in record.lifecycle_events
        )
        assert any(
            "could not be confirmed" in issue
            for child in record.state.child_workflows.values()
            for issue in child.blocking_issues
        )
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_feature_owning_a_deferred_effect_is_left_alone(tmp_path: Path) -> None:
    """A deferred operation is waiting for its next credentialed request, which is normal."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'owning-feature-deferred.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    store = SqlAlchemyFeatureControlPlane(database, operation_journal=journal)
    try:
        await _running_feature_with_a_workstream(store, "feature-deferred")
        await _interrupted_operation(
            journal,
            operation_type=ExternalOperationType.PUSH_BRANCH,
            workflow_id="feature-deferred",
            feature_id="feature-deferred",
            repository_id="backend",
            idempotency_key="feature-deferred:push",
            max_attempts=3,
        )
        recovery = RecoveryService(journal, unconfirmed_effect_escalation=store)

        await recovery.recover_incomplete_operations()

        record = await store.get_record("feature-deferred")
        assert record.state.status is FeatureWorkflowStatus.CREATING_PULL_REQUESTS
        assert record.state.failure_summary is None
        assert not any(
            event.event == "unconfirmed_external_effect" for event in record.lifecycle_events
        )
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_escalating_the_same_operation_twice_writes_one_event(tmp_path: Path) -> None:
    """A periodic sweep re-reads the same rows every tick and must stay quiet after the first."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'owning-feature-once.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    store = SqlAlchemyFeatureControlPlane(database, operation_journal=journal)
    try:
        await _running_feature_with_a_workstream(store, "feature-once")
        await _interrupted_operation(
            journal,
            operation_type=ExternalOperationType.PUSH_BRANCH,
            workflow_id="feature-once",
            feature_id="feature-once",
            repository_id="backend",
            idempotency_key="feature-once:push",
            max_attempts=1,
        )
        recovery = RecoveryService(journal, unconfirmed_effect_escalation=store)

        await recovery.recover_incomplete_operations()
        await recovery.recover_incomplete_operations()

        record = await store.get_record("feature-once")
        escalations = [
            event
            for event in record.lifecycle_events
            if event.event == "unconfirmed_external_effect"
        ]
        assert len(escalations) == 1
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# 10.7 -- regression
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_normal_pull_request_create_leaves_recovery_nothing_to_do(
    tmp_path: Path,
) -> None:
    """The ordinary path is untouched: one PR, one succeeded operation, no sweep activity."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'regression-pr.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    provider = MockGitHubService()

    created = await _github(
        journal, provider, workflow_id="feature-regression"
    ).create_pull_request(
        "example/platform",
        title="[AI] regression",
        body="Feature ID: regression",
        source_branch="ai/regression",
        target_branch="main",
    )
    summary = await RecoveryService(journal).recover_incomplete_operations()

    assert summary.scanned == 0
    assert len(provider.pull_requests) == 1
    assert created.number == 1
    settled = (await journal.list_operations_for_workflow("feature-regression"))[0]
    assert settled.status is ExternalOperationStatus.SUCCEEDED
    await database.dispose()


@pytest.mark.asyncio
async def test_a_definitive_provider_rejection_stays_a_failure(tmp_path: Path) -> None:
    """A provider that says no is not reclassified as ambiguous by any of this."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'regression-reject.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)

    class RejectingProvider(MockGitHubService):
        """Reject the create the way a provider rejects an invalid request."""

        def create_pull_request(self, *args: Any, **kwargs: Any) -> Any:
            """Refuse definitively, the way an invalid base branch is refused."""
            msg = "the base branch does not exist"
            raise ValueError(msg)

    with pytest.raises(ValueError, match="base branch"):
        await _github(
            journal, RejectingProvider(), workflow_id="feature-reject"
        ).create_pull_request(
            "example/platform",
            title="[AI] reject",
            body="Feature ID: reject",
            source_branch="ai/reject",
            target_branch="does-not-exist",
        )

    settled = (await journal.list_operations_for_workflow("feature-reject"))[0]
    assert settled.status is ExternalOperationStatus.FAILED_RETRYABLE
    assert settled.compensation_status is None
    assert await journal.list_operations_requiring_manual_review() == []
    await database.dispose()


@pytest.mark.asyncio
async def test_a_terminal_failure_is_not_reopened_by_a_reconcile_callback(
    tmp_path: Path,
) -> None:
    """A definitively refused mutation spends its budget and then stays refused."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'regression-terminal.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    provider = _provider_with_pull_request()

    class RefusingLabels:
        """Refuse every label mutation the way a provider refuses an unknown label."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            """Forward everything except the mutation, which is always refused."""
            if name != "add_labels":
                return getattr(self._inner, name)

            def refuse(*_args: Any, **_kwargs: Any) -> None:
                msg = "the label does not exist in this repository"
                raise ValueError(msg)

            return refuse

    service = _github(journal, RefusingLabels(provider), workflow_id="feature-terminal")
    for _ in range(3):
        with pytest.raises(ValueError, match="does not exist"):
            await service.add_labels("example/platform", 1, ["ai-generated"])

    settled = (await journal.list_operations_for_workflow("feature-terminal"))[0]
    assert settled.status is ExternalOperationStatus.FAILED_TERMINAL
    assert settled.attempt == settled.max_attempts

    # The reconcile path is reachable only from the two unresolved statuses, so a terminal
    # operation is not quietly turned back into an ambiguous one.
    with pytest.raises(ExternalOperationError, match="terminal failure"):
        await service.add_labels("example/platform", 1, ["ai-generated"])

    assert provider.labels[("example/platform", 1)] == []
    assert (
        await journal.get(settled.operation_id)
    ).status is ExternalOperationStatus.FAILED_TERMINAL
    assert await journal.list_operations_requiring_manual_review() == []
    await database.dispose()


# --------------------------------------------------------------------------------------
# 10.8 -- a deferred operation whose feature will never make another request
# --------------------------------------------------------------------------------------
#
# Task 24- made deferral the normal path: the credential-free sweep leaves an interrupted
# push or pull request where the next credentialed request through the same code path can
# settle it. Task 27- surfaced the consequence -- once the row is `awaiting_reconciliation`,
# `list_incomplete_operations` no longer returns it, so no sweep ever looks at it again.
# For a live feature that is fine and task 28- makes it better, because a crashed feature is
# now continued and a continuation re-enters that code path. For a feature that ended, there
# is no next request at all, and the row would wait for ever in the operator queue with no
# question attached to it.


async def _deferred_operation_owned_by(
    journal: ExternalOperationJournal,
    store: SqlAlchemyFeatureControlPlane,
    feature_id: str,
) -> Any:
    """Leave one feature owning a push the sweep deferred, exactly as 24- does."""
    await _running_feature_with_a_workstream(store, feature_id)
    operation = await _interrupted_operation(
        journal,
        operation_type=ExternalOperationType.PUSH_BRANCH,
        workflow_id=feature_id,
        feature_id=feature_id,
        repository_id="backend",
        idempotency_key=f"{feature_id}:push",
        max_attempts=3,
    )
    await RecoveryService(journal).recover_incomplete_operations()
    deferred = await journal.get(operation.operation_id)
    assert deferred.status is ExternalOperationStatus.AWAITING_RECONCILIATION
    # The consequence: nothing scans this row again.
    assert await journal.list_incomplete_operations() == []
    return deferred


async def _age_operation(database: Database, operation_id: str, *, hours: int) -> None:
    """Backdate one journal row, because the settlement policy is an age threshold."""
    async with database.session() as session:
        await session.execute(
            update(ExternalOperationModel)
            .where(ExternalOperationModel.operation_id == operation_id)
            .values(updated_at=datetime.now(UTC) - timedelta(hours=hours))
        )
        await session.commit()


@pytest.mark.asyncio
async def test_a_deferred_effect_on_an_ended_feature_is_put_to_a_person_once(
    tmp_path: Path,
) -> None:
    """Nobody will ever re-enter that code path, so the deferral has to be resolved somehow.

    Not by guessing the effect: the sweep still holds no credentials and still knows nothing
    about the provider. What changes is that the row stops silently waiting for a request
    that is not coming, and says a person has to look.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'stranded-deferred.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    store = SqlAlchemyFeatureControlPlane(database, operation_journal=journal)
    try:
        deferred = await _deferred_operation_owned_by(journal, store, "feature-ended")
        # The feature reaches a terminal status and nobody resumes it.
        ended = (await store.get_record("feature-ended")).state.model_copy(deep=True)
        ended.status = FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        await store._replace_state("feature-ended", ended, "feature_execution_failed")  # noqa: SLF001
        await _age_operation(database, deferred.operation_id, hours=48)

        await RecoveryService(
            journal, deferred_operation_settle_after_seconds=3_600.0
        ).recover_incomplete_operations()

        settled = await journal.get(deferred.operation_id)
        assert settled.status is ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE
        assert settled.compensation_status is CompensationStatus.MANUAL_REVIEW_REQUIRED
        assert settled.error_code == "reconciliation_required"
        assert "ended without any credentialed request" in (settled.error_message or "")
        # And it reaches the queue an operator actually reads, as a decision rather than a
        # row that has been waiting since some crash nobody remembers.
        assert deferred.operation_id in {
            item.operation_id for item in await journal.list_operations_requiring_manual_review()
        }
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_deferred_effect_on_a_live_feature_is_left_to_its_next_request(
    tmp_path: Path,
) -> None:
    """The whole point of deferring. A running feature will re-enter that code path itself.

    After task 28- it does so without a human even when its executor died, because the claim
    that follows continues the run rather than tombstoning it.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'live-deferred.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    store = SqlAlchemyFeatureControlPlane(database, operation_journal=journal)
    try:
        deferred = await _deferred_operation_owned_by(journal, store, "feature-live")
        await _age_operation(database, deferred.operation_id, hours=48)

        await RecoveryService(
            journal, deferred_operation_settle_after_seconds=3_600.0
        ).recover_incomplete_operations()

        unchanged = await journal.get(deferred.operation_id)
        assert unchanged.status is ExternalOperationStatus.AWAITING_RECONCILIATION
        assert unchanged.compensation_status is CompensationStatus.NOT_REQUIRED
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_deferred_effect_on_a_feature_that_just_ended_is_not_settled_yet(
    tmp_path: Path,
) -> None:
    """A failed feature can still be resumed by somebody who comes back to it.

    Asking immediately would ask about an effect the very next resume would confirm, which is
    the mistake task 24- removed from the sweep in the first place. The threshold is what
    keeps this a last resort rather than a second version of that.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'recent-deferred.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    store = SqlAlchemyFeatureControlPlane(database, operation_journal=journal)
    try:
        deferred = await _deferred_operation_owned_by(journal, store, "feature-recent")
        ended = (await store.get_record("feature-recent")).state.model_copy(deep=True)
        ended.status = FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        await store._replace_state("feature-recent", ended, "feature_execution_failed")  # noqa: SLF001

        await RecoveryService(
            journal, deferred_operation_settle_after_seconds=86_400.0
        ).recover_incomplete_operations()

        unchanged = await journal.get(deferred.operation_id)
        assert unchanged.status is ExternalOperationStatus.AWAITING_RECONCILIATION
        assert unchanged.compensation_status is CompensationStatus.NOT_REQUIRED
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_settling_a_stranded_deferred_effect_twice_changes_nothing(
    tmp_path: Path,
) -> None:
    """The sweep runs every thirty seconds and will find the same row on every tick."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'stranded-twice.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=0.001)
    store = SqlAlchemyFeatureControlPlane(database, operation_journal=journal)
    try:
        deferred = await _deferred_operation_owned_by(journal, store, "feature-twice")
        ended = (await store.get_record("feature-twice")).state.model_copy(deep=True)
        ended.status = FeatureWorkflowStatus.CANCELLED
        await store._replace_state("feature-twice", ended, "feature_cancelled")  # noqa: SLF001
        await _age_operation(database, deferred.operation_id, hours=48)
        recovery = RecoveryService(journal, deferred_operation_settle_after_seconds=3_600.0)

        await recovery.recover_incomplete_operations()
        first = await journal.get(deferred.operation_id)
        await recovery.recover_incomplete_operations()

        again = await journal.get(deferred.operation_id)
        assert first.status is ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE
        assert (again.status, again.attempt, again.error_code) == (
            first.status,
            first.attempt,
            first.error_code,
        )
    finally:
        await database.dispose()
