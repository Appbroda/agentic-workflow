"""Durability, locking, workspace, and readiness tests for the production runtime path."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text, update

from adapters.git_adapter import MockGitService
from agents.shared.contracts import safe_error_diagnostics
from api.control_plane import RequestScopedCredentials, WorkflowConflictError
from api.feature_schemas import StartFeatureRequest
from main import InfrastructureReadinessProbe, ReadinessProbe, _migration_head, create_app
from services.recovery_service import RecoveryService
from services.runtime import (
    RuntimeConfigurationError,
    WorkspaceProvisioner,
)
from state.enums import FeatureWorkflowStatus
from state.external_operations import ExternalOperationStatus, ExternalOperationType
from state.models import AgentState
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from storage.feature_store import SqlAlchemyFeatureControlPlane
from storage.models import ExternalOperationModel, FeatureWorkflowModel
from storage.workflow_lock import RedisWorkflowLock


class FakeRedis:
    """Minimal deterministic Redis substitute that evaluates the compare-and-delete release path."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def set(self, name: str, value: str, *, nx: bool, ex: int) -> bool | None:
        """Implement only NX set semantics used by the distributed lock."""
        assert nx is True
        assert ex > 0
        if name in self.values:
            return None
        self.values[name] = value
        return True

    async def eval(self, _script: str, numkeys: int, *keys_and_args: str) -> int:
        """Renew or delete only if the caller's token still owns the key."""
        assert numkeys == 1
        key, token, *_lease = keys_and_args
        if self.values.get(key) != token:
            return 0
        if len(keys_and_args) == 3:
            return 1
        del self.values[key]
        return 1

    async def exists(self, name: str) -> int:
        """Expose current lock ownership to cancellation handoff checks."""
        return int(name in self.values)


class LeaseLosingRedis(FakeRedis):
    """Simulate another worker taking ownership at the first renewal boundary."""

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> int:
        """Replace the token before renewal while retaining normal checked release behavior."""
        if len(keys_and_args) == 3:
            key, _token, _lease = keys_and_args
            self.values[key] = "replacement-owner"
            return 0
        return await super().eval(script, numkeys, *keys_and_args)


class ParentCheckingAsyncGit:
    """Async Git double that requires the clone process working directory to exist."""

    def __init__(self) -> None:
        self.clone_destination: Path | None = None

    async def clone(self, _source_url: str, destination: str | Path, **_kwargs: Any) -> Path:
        """Record a clone only after its parent directory is provisioned."""
        target = Path(destination)
        assert target.parent.is_dir()
        target.mkdir()
        self.clone_destination = target
        return target

    async def create_branch(
        self, _repository_path: str | Path, branch_name: str, **_kwargs: Any
    ) -> str:
        """Accept the branch step after a successful synthetic clone."""
        return branch_name


class UnavailableProbe(ReadinessProbe):
    """Simulate unavailable infrastructure without opening real database or Redis connections."""

    async def is_ready(self) -> bool:
        """Report an unavailable dependency state."""
        return False


class ReadyRedis:
    """Minimal readiness-only Redis substitute."""

    async def ping(self) -> bool:
        """Report the dependency as reachable without opening a network connection."""
        return True


@pytest.mark.asyncio
async def test_redis_lock_releases_only_its_own_key() -> None:
    """The lock uses token-checked deletion, not deletion of another caller's lock."""
    redis = FakeRedis()
    lock = RedisWorkflowLock(redis)

    async with lock.hold("workflow-1"):
        assert len(redis.values) == 1
    assert redis.values == {}


@pytest.mark.asyncio
async def test_redis_lock_aborts_the_protected_body_when_lease_ownership_is_lost() -> None:
    """A runner must stop before persisting side effects after another owner acquires its key."""
    redis = LeaseLosingRedis()
    lock = RedisWorkflowLock(redis, lease_seconds=1)
    body_completed = False

    with pytest.raises(WorkflowConflictError, match="ownership was lost"):
        async with asyncio.timeout(3):
            async with lock.hold("workflow-lost-lease"):
                await asyncio.Event().wait()
                body_completed = True

    assert not body_completed
    assert list(redis.values.values()) == ["replacement-owner"]


def test_workspace_provisioner_rejects_paths_outside_the_worker_volume(tmp_path: Path) -> None:
    """Callers cannot make a live workflow clone into arbitrary host-visible paths."""
    workspace_root = tmp_path / "workspaces"
    state = initial_state(tmp_path / "outside")

    with pytest.raises(RuntimeConfigurationError, match="configured workspace_root"):
        WorkspaceProvisioner(workspace_root).provision(state, git_service=MockGitService())


@pytest.mark.asyncio
async def test_workspace_provisioner_creates_a_nested_clone_parent(tmp_path: Path) -> None:
    """A feature workspace may be nested beneath the shared worker volume."""
    workspace_root = tmp_path / "workspaces"
    state = initial_state(workspace_root / "feature-1" / "backend")
    git_service = ParentCheckingAsyncGit()

    await WorkspaceProvisioner(workspace_root).provision_async(state, git_service=git_service)

    assert git_service.clone_destination == workspace_root / "feature-1" / "backend"


@pytest.mark.asyncio
async def test_readiness_endpoint_returns_503_when_dependencies_are_unavailable() -> None:
    """Load balancers receive an explicit unavailable response instead of a false-ready signal."""
    app = create_app(platform_api_key="test-key", readiness_probe=UnavailableProbe())

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


@pytest.mark.asyncio
async def test_production_host_policy_rejects_unconfigured_host_headers() -> None:
    """The live app does not trust an arbitrary Host header at the reverse-proxy boundary."""
    app = create_app(platform_api_key="test-key")
    app.state.allowed_hosts = ["api.example.test"]

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="https://untrusted.example.test"
    ) as client:
        response = await client.get("/healthz")

    assert response.status_code == 400
    assert response.json() == {"detail": "Host is not allowed."}


@pytest.mark.asyncio
async def test_readiness_requires_the_deployed_alembic_revision(tmp_path: Path) -> None:
    """A reachable database is not ready until the schema matches this application image."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'readiness.db'}")
    try:
        async with database.engine.begin() as connection:
            await connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32))"))
            await connection.execute(
                text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
                {"revision": _migration_head()},
            )
        ready = InfrastructureReadinessProbe(
            database, ReadyRedis(), expected_migration_revision=_migration_head()
        )
        stale = InfrastructureReadinessProbe(
            database, ReadyRedis(), expected_migration_revision="not-the-current-head"
        )

        assert await ready.is_ready()
        assert not await stale.is_ready()
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_unresolved_external_operations_do_not_degrade_readiness(tmp_path: Path) -> None:
    """One unresolved effect stays visible to an operator and refuses nobody else's writes.

    This assertion is the inverse of the one it replaces, deliberately. Readiness used to go
    unavailable while any operation was unresolved, and the mutation middleware reads
    readiness, so an interrupted push on one feature refused every non-GET request on the
    whole deployment until somebody edited the database.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'dynamic-readiness.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    recovery = RecoveryService(journal)
    try:
        await recovery.recover_incomplete_operations()
        probe = InfrastructureReadinessProbe(
            database,
            ReadyRedis(),
            recovery_service=recovery,
        )
        assert await probe.is_ready()
        operation = await journal.create_operation(
            workflow_id="dynamic-readiness",
            feature_id=None,
            child_workflow_id=None,
            repository_id=None,
            operation_type=ExternalOperationType.PUSH_BRANCH,
            idempotency_key="dynamic-readiness:push",
            input_fingerprint="push",
        )
        await journal.record_failure(
            operation.operation_id,
            status=ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
            error_code="reconcile",
            error_message="reconciliation required",
        )
        assert await probe.is_ready()
        # Still counted, and still listed: the operator metric is what surfaces it now.
        assert await journal.unresolved_critical_count() == 1
        assert [item.operation_id for item in await journal.list_unresolved_operations()] == [
            operation.operation_id
        ]

        await journal.record_reconciled_result(
            operation.operation_id,
            external_reference="verified-remote-sha",
            result_payload={"reconciled": True},
        )
        assert await probe.is_ready()
        assert await journal.unresolved_critical_count() == 0
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_stale_sweep_allows_a_cancelling_feature_owner_to_finalize_with_cleanup(
    tmp_path: Path,
) -> None:
    """Shared feature operations remain fenced and auditable without a permanent outage."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'feature-stale-cancel.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database, stale_after_seconds=60)
    recovery = RecoveryService(journal)
    store = SqlAlchemyFeatureControlPlane(database, operation_journal=journal)
    try:
        request = StartFeatureRequest.model_validate(
            {
                "feature_id": "feature-stale-cancel",
                "execution_mode": "live",
                "prd": feature_prd(),
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
        await store.start(
            request,
            idempotency_key="feature-stale-cancel-001",
            credentials=RequestScopedCredentials(None, None),
            owner_id="platform-admin",
        )
        operation = await journal.create_operation(
            workflow_id="feature-stale-cancel",
            feature_id="feature-stale-cancel",
            child_workflow_id="feature-stale-cancel:backend",
            repository_id="backend",
            operation_type=ExternalOperationType.PUSH_BRANCH,
            idempotency_key="feature-stale-cancel:push",
            input_fingerprint="feature-stale-cancel-push",
        )
        await journal.claim_operation(operation.operation_id)
        startup = await recovery.recover_incomplete_operations()
        probe = InfrastructureReadinessProbe(
            database,
            ReadyRedis(),
            recovery_service=recovery,
        )

        assert startup.scanned == 0
        cancelling = await store.cancel("feature-stale-cancel", requested_by="platform-admin")
        assert cancelling.state.status is FeatureWorkflowStatus.CANCELLING
        assert (await journal.get(operation.operation_id)).status is (
            ExternalOperationStatus.CANCELLATION_REQUESTED
        )
        # One feature's pending cancellation is not the deployment's problem.
        assert await probe.is_ready()
        assert await journal.unresolved_critical_count() == 1

        async with database.session() as session:
            await session.execute(
                update(ExternalOperationModel)
                .where(ExternalOperationModel.operation_id == operation.operation_id)
                .values(heartbeat_at=datetime.now(UTC) - timedelta(minutes=2))
            )
            await session.commit()
        swept = await recovery.recover_incomplete_operations()

        assert swept.unknown == 1
        assert (await journal.get(operation.operation_id)).status is (
            ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE
        )
        finalized = await store.cancel("feature-stale-cancel", requested_by="platform-admin")
        assert finalized.state.status is (
            FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS
        )
        assert [item.external_reference for item in finalized.state.cleanup_requirements] == [
            operation.operation_id
        ]
        assert await probe.is_ready()
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_terminal_feature_unknown_effect_remains_auditable_without_global_outage(
    tmp_path: Path,
) -> None:
    """Cleanup-fenced terminal uncertainty must not block unrelated workflows forever."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'terminal-unknown-ready.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    recovery = RecoveryService(journal)
    try:
        terminal_operation = await journal.create_operation(
            workflow_id="terminal-feature",
            feature_id="terminal-feature",
            child_workflow_id="terminal-feature:backend",
            repository_id="backend",
            operation_type=ExternalOperationType.PUSH_BRANCH,
            idempotency_key="terminal-feature:push",
            input_fingerprint="terminal-feature-push",
        )
        await journal.record_failure(
            terminal_operation.operation_id,
            status=ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
            error_code="provider_timeout",
            error_message="manual reconciliation required",
        )
        async with database.session() as session:
            session.add(
                FeatureWorkflowModel(
                    owner_id="platform-admin",
                    feature_id="terminal-feature",
                    status=FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
                    title="Terminal cleanup feature",
                    execution_mode="live",
                    merge_strategy=None,
                    deployment_strategy=None,
                    state_json={
                        "cleanup_requirements": [
                            {"external_reference": terminal_operation.operation_id}
                        ]
                    },
                )
            )
            await session.commit()

        await recovery.recover_incomplete_operations()
        probe = InfrastructureReadinessProbe(
            database,
            ReadyRedis(),
            recovery_service=recovery,
        )
        assert await journal.unresolved_critical_count() == 1
        assert await probe.is_ready()

        active_operation = await journal.create_operation(
            workflow_id="active-feature",
            feature_id="active-feature",
            child_workflow_id="active-feature:backend",
            repository_id="backend",
            operation_type=ExternalOperationType.PUSH_BRANCH,
            idempotency_key="active-feature:push",
            input_fingerprint="active-feature-push",
        )
        await journal.record_failure(
            active_operation.operation_id,
            status=ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
            error_code="provider_timeout",
            error_message="manual reconciliation required",
        )
        async with database.session() as session:
            session.add(
                FeatureWorkflowModel(
                    owner_id="platform-admin",
                    feature_id="active-feature",
                    status=FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
                    title="Active uncertain feature",
                    execution_mode="live",
                    merge_strategy=None,
                    deployment_strategy=None,
                    state_json={},
                )
            )
            await session.commit()

        # A second unresolved effect, this one on a live feature, and still not a reason to
        # refuse writes for every other feature on the deployment.
        assert await probe.is_ready()
        assert await journal.unresolved_critical_count() == 2
        await journal.record_reconciled_result(
            active_operation.operation_id,
            external_reference="verified-active-sha",
            result_payload={"reconciled": True},
        )
        assert await probe.is_ready()
        retained = await journal.get(terminal_operation.operation_id)
        assert retained.status is ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE
        assert retained.operation_id == terminal_operation.operation_id
    finally:
        await database.dispose()


def test_a_configuration_failure_records_its_cause_only_when_it_is_safe_to() -> None:
    """AB-Feature-111's console recorded six identical, contentless failures.

    Every attempt persisted exactly "The child workstream raised
    RuntimeConfigurationError and could not complete" -- the type name, for a failure whose
    cause the platform had already composed into a sentence. Which of eight raise sites
    fired was not recoverable from the database at all.

    Suppressing an arbitrary message stays right: it can carry a credential, a workspace
    path or subprocess output. So a raise site opts in explicitly, and this pins both
    halves of that decision.
    """
    from services.runtime import RuntimeConfigurationError, _recordable_configuration_error

    recordable = _recordable_configuration_error(
        "repository setup blocked live coding before the Engineer Agent ran "
        "(install_status=failed; issues=BASELINE_REQUIRED_VALIDATION_FAILED_0)"
    )
    # Composed from a constant, a status and platform-generated issue ids; no checkout text.
    assert safe_error_diagnostics(recordable) == (str(recordable),)

    # The default, for the sites that interpolate a resolved workspace path.
    withheld = RuntimeConfigurationError("workflow workspace already exists: /workspaces/abc")
    assert safe_error_diagnostics(withheld) == ()


def initial_state(workspace_path: Path) -> AgentState:
    """Return the smallest valid state required for workspace-boundary validation."""
    from state.models import WorkspaceDescriptor, create_initial_agent_state

    return create_initial_agent_state(
        workflow_id="workflow-workspace-1",
        workspace_descriptor=WorkspaceDescriptor(
            workspace_id="workspace-1",
            root_path=str(workspace_path),
            source_repo_url="https://github.com/example/platform.git",
            default_branch="main",
            working_branch="workflow/workspace-1",
        ),
    )


def feature_prd() -> dict[str, Any]:
    """Return one strict PRD submission without credential-shaped payload fields."""
    return {
        "title": "Durable feature",
        "problem_statement": "Feature records must survive a process restart.",
        "goals": ["Persist the feature."],
        "user_stories": [
            {
                "story_id": "story-durable",
                "persona": "Operator",
                "need": "Recover feature status",
                "benefit": "I can safely supervise automation",
                "acceptance_criteria": ["The state survives a restart."],
            }
        ],
        "requirements": [
            {
                "requirement_id": "requirement-durable",
                "description": "Persist feature lifecycle state.",
                "priority": "must",
                "acceptance_criteria": ["State is readable after a restart."],
                "dependencies": [],
            }
        ],
        "constraints": ["Do not store provider credentials."],
        "out_of_scope": ["Live provider calls."],
        "stakeholders": ["Platform team"],
    }
