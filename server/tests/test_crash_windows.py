"""Seven boundaries, each crossed by killing a real process, and what recovery then does.

Audit risk P1-8. The platform's whole defence against a duplicated side effect is the window
between an effect landing and the journal recording it -- and until now nothing in this suite
ever opened one. Every recovery path was verified by calling a recovery function directly,
in the test's own process, with a hand-built `ExternalOperation`. That is coverage of the
recovery code's arithmetic, not of the evidence a dying worker actually leaves behind.

Each test here starts real work in a child process against a real workspace, a real bare Git
remote, a real database and a state-holding provider double; kills that process with
`SIGKILL` at one named instant; and then runs recovery in this process, from services
constructed fresh out of the same database. The assertions are about the durable record and
the provider or filesystem, never about which method was called -- except boundary 2, where
"the coding model was not asked twice" is the property itself.

Every boundary also runs recovery **twice**. In production the sweep runs every thirty
seconds and will see the same operation many times; a recovery pass that is not idempotent
is a recovery pass that duplicates effects on its second tick.

`tests/crash_child.py` is the process that dies. `tests/crash_support.py` holds the doubles.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import update

from adapters.interruptible_git import InterruptibleGitService, reviewed_content_fingerprint
from adapters.llm_adapter import (
    AnthropicLLMClient,
    ImageInput,
    ResponsesCodingExecutor,
    completed_coding_operation_matches_workspace,
    llm_client_for,
)
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from configs.model_roles import AgentPlatform, ModelRole
from configs.settings import load_settings
from services.cancellation import MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.feature_queue import SUCCEEDED
from services.feature_runtime import LiveChildWorkstreamExecutor
from services.journaled_github import JournaledGitHubService
from services.process_runner import AsyncioProcessRunner
from services.recovery_service import RecoveryService
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.external_operations import (
    CompensationStatus,
    ExternalOperationStatus,
    ExternalOperationType,
    WorkflowCheckpointBoundary,
)
from state.failure_diagnosis import FailureStage, FeatureFailureClassification
from state.feature_models import ChildWorkflowReference
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from storage.feature_store import SqlAlchemyFeatureControlPlane
from storage.models import FeatureExecutionQueueModel
from tests import real_repository_support as real_support
from tests.crash_support import (
    CountingModelDouble,
    DyingModelDouble,
    PersistentGitHubDouble,
    RecordingRealProcessRunner,
    SucceedingProcessRunner,
    remove_index_lock_killer,
    run_crash_child,
)
from tests.fixtures import node_javascript_repository
from tests.support import drain_feature_queue
from tests.test_feature_api import feature_payload
from workflows.feature_workflow import FeatureStep, FeatureWorkflowOrchestrator

pytestmark = pytest.mark.asyncio

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)
WORKFLOW = "feature-crash"
REPOSITORY = "backend"
CHILD = "feature-crash:backend"
BRANCH = "ai/feature-crash/backend/status"

# The sweep only touches an operation whose liveness has lapsed. A boundary here takes
# milliseconds, so the recovering journal is told what "stale" means for these tests rather
# than the tests being told to wait five minutes for the production default.
_STALE_AFTER_SECONDS = 0.01


# --------------------------------------------------------------------------------------
# Shared scaffolding
# --------------------------------------------------------------------------------------


def _git(root: Path, *arguments: str) -> str:
    """Run one Git command against a real checkout and return its output."""
    completed = subprocess.run(
        ("git", *arguments), cwd=root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


async def _database(tmp_path: Path) -> Database:
    """A durable database on disk, because the process that writes it is about to die."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'crash.db'}")
    await database.create_schema()
    return database


def _origin(tmp_path: Path) -> tuple[Path, Path]:
    """Build a real bare remote with content, and return it beside its source checkout."""
    source = tmp_path / "source"
    node_javascript_repository(source)
    remote = tmp_path / "remote.git"
    # `--initial-branch` so the bare repository's HEAD names the branch the source actually
    # pushes. Without it a clone checks out nothing and every later assertion is about an
    # empty directory.
    subprocess.run(
        ("git", "init", "--bare", "--initial-branch=main", str(remote)),
        check=True,
        capture_output=True,
    )
    _git(source, "remote", "add", "origin", str(remote))
    _git(source, "push", "origin", "main")
    return remote, source


def _checkout(tmp_path: Path) -> tuple[Path, Path]:
    """A real feature-branch checkout wired to a real bare remote."""
    remote, _source = _origin(tmp_path)
    workspace = tmp_path / "workspaces" / WORKFLOW / REPOSITORY
    workspace.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(("git", "clone", str(remote), str(workspace)), check=True, capture_output=True)
    _git(workspace, "config", "user.email", "crash@example.com")
    _git(workspace, "config", "user.name", "Crash Suite")
    _git(workspace, "config", "commit.gpgsign", "false")
    _git(workspace, "checkout", "-b", BRANCH)
    return remote, workspace


def _scope_spec(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    """The identifiers and the database every crash child is given."""
    return {
        "database_url": f"sqlite+aiosqlite:///{tmp_path / 'crash.db'}",
        "workflow_id": WORKFLOW,
        "feature_id": WORKFLOW,
        "child_workflow_id": CHILD,
        "repository_id": REPOSITORY,
        "workspace_root": str(tmp_path / "workspaces"),
        **extra,
    }


def _recovery(database: Database, tmp_path: Path) -> RecoveryService:
    """The credential-free sweep, exactly as the deployment composes it."""
    return RecoveryService(
        ExternalOperationJournal(database, stale_after_seconds=_STALE_AFTER_SECONDS),
        workspace_root=tmp_path / "workspaces",
        process_runner=AsyncioProcessRunner(),
    )


async def _one_operation(database: Database, kind: ExternalOperationType) -> Any:
    """Read the single journal row of this type, failing loudly if there is more than one."""
    journal = ExternalOperationJournal(database)
    matching = [
        item
        for item in await journal.list_operations_for_workflow(WORKFLOW)
        if item.operation_type is kind
    ]
    assert len(matching) == 1, f"expected exactly one {kind.value} operation, found {len(matching)}"
    return matching[0]


def _durable_state(operation: Any) -> tuple[Any, ...]:
    """Everything about one operation a second recovery pass must not change."""
    return (
        operation.status,
        operation.attempt,
        operation.error_code,
        operation.external_reference,
        operation.compensation_status,
        json.dumps(operation.result_payload or {}, sort_keys=True),
    )


def _tree(root: Path) -> dict[str, str]:
    """Fingerprint every file under a directory, so a filesystem mutation cannot hide."""
    import hashlib

    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


def _executor(journal: ExternalOperationJournal) -> ExternalOperationExecutor:
    """The journaling executor a fresh, credentialed request builds after a crash."""
    return ExternalOperationExecutor(
        journal=journal,
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(
            workflow_id=WORKFLOW,
            feature_id=WORKFLOW,
            child_workflow_id=CHILD,
            repository_id=REPOSITORY,
        ),
        heartbeat_seconds=3_600.0,
    )


def _remote_sha(workspace: Path, branch: str) -> str | None:
    """Ask the real remote where a branch points, the way reconciliation does."""
    output = _git(workspace, "ls-remote", "origin", f"refs/heads/{branch}")
    return output.split("\t")[0] if output else None


# --------------------------------------------------------------------------------------
# Boundary 1 -- after clone, before journal success
# --------------------------------------------------------------------------------------


async def test_a_clone_killed_before_its_journal_entry_is_reused_not_repeated(
    tmp_path: Path,
) -> None:
    """The workspace is adopted from `origin` rather than cloned a second time.

    A re-clone is not merely wasteful here: the adapter refuses a destination that already
    exists, so repeating the effect would end the workstream. The sentinel file proves the
    directory that survives is the one the dead process made, not a replacement.
    """
    remote, _source = _origin(tmp_path)
    destination = tmp_path / "workspaces" / WORKFLOW / REPOSITORY
    destination.parent.mkdir(parents=True, exist_ok=True)
    database = await _database(tmp_path)
    try:
        run_crash_child(
            "clone",
            _scope_spec(tmp_path, source_url=str(remote), destination=str(destination)),
        )
        assert (destination / ".git").is_dir(), "the clone did not land before the kill"
        interrupted = await _one_operation(database, ExternalOperationType.CLONE_REPOSITORY)
        assert interrupted.status is ExternalOperationStatus.RUNNING
        sentinel = destination / "sentinel.txt"
        sentinel.write_text("written after the crash\n", encoding="utf-8")

        await _recovery(database, tmp_path).recover_incomplete_operations()

        reconciled = await _one_operation(database, ExternalOperationType.CLONE_REPOSITORY)
        assert reconciled.status is ExternalOperationStatus.SUCCEEDED
        assert reconciled.external_reference == str(destination)
        assert (reconciled.result_payload or {}).get("recovery_method") == "clone_present"

        # A second sweep must change nothing, and neither must the next credentialed call.
        before = _durable_state(reconciled), _tree(destination)
        await _recovery(database, tmp_path).recover_incomplete_operations()
        settled = await _one_operation(database, ExternalOperationType.CLONE_REPOSITORY)
        assert (_durable_state(settled), _tree(destination)) == before

        journal = ExternalOperationJournal(database)
        service = InterruptibleGitService(
            cancellation_token=MockCancellationToken(),
            operation_executor=_executor(journal),
            workspace_root=tmp_path / "workspaces",
        )
        assert await service.clone(str(remote), destination) == destination
        assert sentinel.read_text(encoding="utf-8") == "written after the crash\n"
        assert _tree(destination) == before[1]
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# Boundary 2 -- after the coding executor produced output, before review
# --------------------------------------------------------------------------------------


def _coding_input(attempt: int) -> str:
    """The Engineer input shape whose durable attempt identity keys the coding operation."""
    return json.dumps(
        {
            "task_plan": {
                "timestamp": datetime.now(UTC).isoformat(),
                "metadata": {"attempt": attempt, "repository_revision": "revision-before"},
            },
            "prior_review": None,
            "execution_context": {"repository_revision": "revision-before"},
        }
    )


_CODING_FILE = "src/controllers/status.js"
_CODING_FILE_CONTENT = "function statusController() {\n  return { status: 'ok' };\n}\n"
_CODING_RESPONSE: dict[str, Any] = {
    "summary": "Add the status controller.",
    "files": [{"path": _CODING_FILE, "content": _CODING_FILE_CONTENT}],
}


async def test_a_crash_after_the_engineer_wrote_its_files_never_calls_the_model_again(
    tmp_path: Path,
) -> None:
    """The receipt in the journal is what stops a second coding call, and it holds bytes.

    Boundary 2 is the one place a call count is the property under test rather than a proxy
    for one: an Engineer invoked twice for one durable attempt costs a model call, produces
    a different diff, and leaves the journal describing content the workspace no longer has.
    """
    _remote, workspace = _checkout(tmp_path)
    database = await _database(tmp_path)
    call_log = tmp_path / "model-calls.log"
    written = workspace / "src" / "controllers" / "status.js"
    try:
        run_crash_child(
            "coding",
            _scope_spec(
                tmp_path,
                workspace=str(workspace),
                instructions="Implement the status controller.",
                input_text=_coding_input(attempt=1),
                response=_CODING_RESPONSE,
                call_log=str(call_log),
            ),
        )
        model = CountingModelDouble(_CODING_RESPONSE, call_log)
        assert model.invocations == 1
        assert written.is_file(), "the coding output did not land before the kill"
        recorded = await _one_operation(database, ExternalOperationType.RUN_CODING_EXECUTOR)
        assert recorded.status is ExternalOperationStatus.SUCCEEDED

        # A fresh process resumes the same durable attempt. The prompt envelope carries a new
        # timestamp and the repository revision now includes the effect, exactly as it would
        # after a restart; neither is a new instruction.
        journal = ExternalOperationJournal(database)
        resumed = await ResponsesCodingExecutor(model).execute(
            workspace_root=workspace,
            instructions="Implement the status controller.",
            input_text=_coding_input(attempt=1),
            operation_executor=_executor(journal),
        )

        assert model.invocations == 1, "the Engineer was called again for a completed attempt"
        assert [path.as_posix() for path in resumed.modified_files] == ["src/controllers/status.js"]
        assert written.read_text(encoding="utf-8") == _CODING_FILE_CONTENT
        assert completed_coding_operation_matches_workspace(recorded, workspace)

        # And the executor's own recognition of a claimed attempt agrees.
        request = StartFeatureRequest.model_validate({**feature_payload(), "feature_id": WORKFLOW})
        feature = _initial_feature_state(WORKFLOW, request)
        repository = next(
            item for item in feature.repository_specs if item.repository_id == REPOSITORY
        )
        child = ChildWorkflowReference(
            child_workflow_id=CHILD,
            repository_id=REPOSITORY,
            workstream_id=REPOSITORY,
            status=ChildWorkflowStatus.RUNNING,
            branch_name=BRANCH,
            workspace_path=str(workspace),
            retry_count=1,
        )
        live = LiveChildWorkstreamExecutor(
            settings=load_settings(workspace_root=tmp_path / "workspaces"),
            git_environment={},
            engineer_client=model,
            reviewer_client=model,
            journal=journal,
            cancellation_token=MockCancellationToken(),
        )
        assert await live._completed_coding_output_matches(  # noqa: SLF001
            feature=feature, repository=repository, child=child, workspace=workspace
        )

        # Two recovery passes over a completed operation must do nothing at all.
        before = _durable_state(recorded), _tree(workspace), model.invocations
        for _ in range(2):
            await _recovery(database, tmp_path).recover_incomplete_operations()
        settled = await _one_operation(database, ExternalOperationType.RUN_CODING_EXECUTOR)
        assert (_durable_state(settled), _tree(workspace), model.invocations) == before
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# Boundary 3 -- after commit, before the SHA is written
# --------------------------------------------------------------------------------------


async def test_a_commit_killed_before_its_sha_was_recorded_is_identified_from_its_parent(
    tmp_path: Path,
) -> None:
    """The SHA is unknowable after the crash, so the journal identifies the commit by lineage.

    The parent SHA and the exact message are committed *before* the effect, which is what
    makes the commit findable afterwards without ever creating a second one.
    """
    _remote, workspace = _checkout(tmp_path)
    changed = workspace / "src" / "routes" / "status.js"
    changed.write_text("function statusRoute() {\n  return { status: 'ok' };\n}\n", "utf-8")
    parent = _git(workspace, "rev-parse", "HEAD")
    database = await _database(tmp_path)
    try:
        run_crash_child(
            "commit",
            _scope_spec(
                tmp_path,
                workspace=str(workspace),
                message="Serve the status payload",
                files=["src/routes/status.js"],
            ),
        )
        head = _git(workspace, "rev-parse", "HEAD")
        assert head != parent, "the commit did not land before the kill"
        assert _git(workspace, "rev-list", "--count", "HEAD") == "2"
        interrupted = await _one_operation(database, ExternalOperationType.CREATE_COMMIT)
        assert interrupted.status is ExternalOperationStatus.RUNNING
        assert interrupted.safe_metadata["parent_head_sha"] == parent
        assert interrupted.safe_metadata["commit_message"] == "Serve the status payload"

        await _recovery(database, tmp_path).recover_incomplete_operations()

        reconciled = await _one_operation(database, ExternalOperationType.CREATE_COMMIT)
        assert reconciled.status is ExternalOperationStatus.SUCCEEDED
        assert reconciled.external_reference == head
        assert (reconciled.result_payload or {}).get("recovery_method") == "local_commit_present"

        before = _durable_state(reconciled), _tree(workspace), head
        await _recovery(database, tmp_path).recover_incomplete_operations()
        settled = await _one_operation(database, ExternalOperationType.CREATE_COMMIT)
        assert (
            _durable_state(settled),
            _tree(workspace),
            _git(workspace, "rev-parse", "HEAD"),
        ) == before

        # The credentialed replay reuses the reconciled commit rather than making another.
        runner = RecordingRealProcessRunner()
        service = InterruptibleGitService(
            cancellation_token=MockCancellationToken(),
            operation_executor=_executor(ExternalOperationJournal(database)),
            workspace_root=tmp_path / "workspaces",
            process_runner=runner,
        )
        replayed = await service.commit(
            workspace,
            "Serve the status payload",
            files=["src/routes/status.js"],
            expected_content_fingerprint=reviewed_content_fingerprint(
                workspace, ["src/routes/status.js"]
            ),
        )
        assert replayed == head
        assert runner.ran("git", "commit") == 0, "a second commit was created"
        assert _git(workspace, "rev-list", "--count", "HEAD") == "2"
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# Boundary 4 -- after push, before journal success (task 24-'s contract)
# --------------------------------------------------------------------------------------


async def test_a_push_killed_before_its_journal_entry_is_deferred_and_later_reconciled(
    tmp_path: Path,
) -> None:
    """The sweep holds no credentials, so it must leave this where a caller can still settle it.

    Stamping `UNKNOWN_EXTERNAL_STATE` here is what task 24- removed: it asked a person about
    an effect the very next credentialed request could prove, and -- because readiness
    counted those rows -- it refused every unrelated write on the deployment while it waited.
    """
    _remote, workspace = _checkout(tmp_path)
    (workspace / "src" / "routes" / "status.js").write_text("// pushed\n", encoding="utf-8")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Serve the status payload")
    local = _git(workspace, "rev-parse", "HEAD")
    database = await _database(tmp_path)
    try:
        run_crash_child("push", _scope_spec(tmp_path, workspace=str(workspace), branch=BRANCH))
        assert _remote_sha(workspace, BRANCH) == local, "the push did not land before the kill"
        interrupted = await _one_operation(database, ExternalOperationType.PUSH_BRANCH)
        assert interrupted.status is ExternalOperationStatus.RUNNING

        summary = await _recovery(database, tmp_path).recover_incomplete_operations()

        deferred = await _one_operation(database, ExternalOperationType.PUSH_BRANCH)
        assert deferred.status is ExternalOperationStatus.AWAITING_RECONCILIATION
        assert deferred.compensation_status is not CompensationStatus.MANUAL_REVIEW_REQUIRED
        assert deferred.error_code == "awaiting_credentialed_reconciliation"
        assert deferred.compensation_status is CompensationStatus.NOT_REQUIRED
        assert (summary.deferred, summary.unknown, summary.failures) == (1, 0, 0)
        assert not summary.has_unresolved_critical_operations, (
            "one interrupted push must not make the whole platform unready"
        )

        before = _durable_state(deferred), _remote_sha(workspace, BRANCH)
        await _recovery(database, tmp_path).recover_incomplete_operations()
        settled = await _one_operation(database, ExternalOperationType.PUSH_BRANCH)
        assert (_durable_state(settled), _remote_sha(workspace, BRANCH)) == before

        # Now the credentialed request that re-enters the same code path.
        runner = RecordingRealProcessRunner()
        service = InterruptibleGitService(
            cancellation_token=MockCancellationToken(),
            operation_executor=_executor(ExternalOperationJournal(database)),
            workspace_root=tmp_path / "workspaces",
            process_runner=runner,
        )
        result = await service.push(workspace, BRANCH)

        assert result.summaries == ("reused",)
        assert runner.ran("git", "push") == 0, "the push was issued a second time"
        reconciled = await _one_operation(database, ExternalOperationType.PUSH_BRANCH)
        assert reconciled.status is ExternalOperationStatus.SUCCEEDED
        assert reconciled.external_reference == local
        payload = reconciled.result_payload or {}
        assert payload.get("recovery_method") == "remote_branch_sha_match"
        assert _remote_sha(workspace, BRANCH) == local

        # And the sweep that finds it again half a minute later leaves the settled row alone.
        after = _durable_state(reconciled)
        await _recovery(database, tmp_path).recover_incomplete_operations()
        settled_push = await _one_operation(database, ExternalOperationType.PUSH_BRANCH)
        assert _durable_state(settled_push) == after
        assert runner.ran("git", "push") == 0
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# Boundary 5 -- after PR create, before journal success (task 24-'s contract)
# --------------------------------------------------------------------------------------


async def test_a_pull_request_created_by_a_dead_process_is_adopted_not_opened_again(
    tmp_path: Path,
) -> None:
    """A second pull request for one branch is the worst outcome this journal exists to prevent."""
    database = await _database(tmp_path)
    provider_state = tmp_path / "github-state.json"
    arguments: dict[str, Any] = {
        "title": "Serve the status payload",
        "body": "Adds the status route.",
        "source_branch": BRANCH,
        "target_branch": "main",
    }
    try:
        run_crash_child(
            "pull_request",
            _scope_spec(
                tmp_path,
                provider_state=str(provider_state),
                repository="example/backend",
                **arguments,
            ),
        )
        provider = PersistentGitHubDouble(provider_state)
        assert provider.creations == 1
        assert len(provider.pull_requests) == 1
        interrupted = await _one_operation(database, ExternalOperationType.CREATE_PULL_REQUEST)
        assert interrupted.status is ExternalOperationStatus.RUNNING

        summary = await _recovery(database, tmp_path).recover_incomplete_operations()

        deferred = await _one_operation(database, ExternalOperationType.CREATE_PULL_REQUEST)
        assert deferred.status is ExternalOperationStatus.AWAITING_RECONCILIATION
        assert deferred.compensation_status is CompensationStatus.NOT_REQUIRED
        assert not summary.has_unresolved_critical_operations

        before = _durable_state(deferred), provider.creations, len(provider.pull_requests)
        await _recovery(database, tmp_path).recover_incomplete_operations()
        again = PersistentGitHubDouble(provider_state)
        settled = await _one_operation(database, ExternalOperationType.CREATE_PULL_REQUEST)
        assert (_durable_state(settled), again.creations, len(again.pull_requests)) == before

        service = JournaledGitHubService(
            again, operation_executor=_executor(ExternalOperationJournal(database))
        )
        adopted = await service.create_pull_request("example/backend", **arguments)

        assert adopted.number == 1
        assert again.creations == 1, "the provider was asked to open a second pull request"
        assert len(PersistentGitHubDouble(provider_state).pull_requests) == 1
        reconciled = await _one_operation(database, ExternalOperationType.CREATE_PULL_REQUEST)
        assert reconciled.status is ExternalOperationStatus.SUCCEEDED
        assert (reconciled.result_payload or {}).get(
            "recovery_method"
        ) == "existing_pull_request_adopted"

        # And the sweep that finds it again half a minute later leaves the settled row alone.
        after = _durable_state(reconciled)
        await _recovery(database, tmp_path).recover_incomplete_operations()
        settled_again = await _one_operation(database, ExternalOperationType.CREATE_PULL_REQUEST)
        assert _durable_state(settled_again) == after
        assert PersistentGitHubDouble(provider_state).creations == 1
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# Boundary 6 -- between AFTER_VALIDATION and the next BEFORE_CODING
# --------------------------------------------------------------------------------------


async def _accept(store: SqlAlchemyFeatureControlPlane, *, agent_platform: str = "openai") -> str:
    """Accept one feature through the real path so a crash child has something to claim."""
    request = StartFeatureRequest.model_validate(
        {**feature_payload(), "feature_id": WORKFLOW, "agent_platform": agent_platform}
    )
    result = await store.start(
        request, idempotency_key="crash", credentials=CREDENTIALS, owner_id="platform-admin"
    )
    return result.record.state.feature_id


def _child_reference(workspace: Path, *, attempt: int) -> dict[str, Any]:
    """The durable child a validated attempt leaves behind, at the boundary it stopped on."""
    return ChildWorkflowReference(
        child_workflow_id=CHILD,
        repository_id=REPOSITORY,
        workstream_id=REPOSITORY,
        status=ChildWorkflowStatus.RUNNING,
        branch_name=BRANCH,
        workspace_path=str(workspace),
        retry_count=attempt,
        checkpoint_boundary=WorkflowCheckpointBoundary.AFTER_VALIDATION,
    ).model_dump(mode="json")


async def test_a_process_killed_mid_git_leaves_a_lock_the_next_attempt_clears(
    tmp_path: Path,
) -> None:
    """The attempt counter survives the crash, and the next attempt is not killed by the lock.

    This platform terminates process groups -- on cancellation and on worker death -- so an
    attempt killed inside `git` leaves `.git/index.lock` behind, and the staging command at
    the top of the next attempt is the first Git write to meet it. `git add -A` exits 128 on
    that, which read as a hard workspace fault and ended the repository. Task 26- put
    `_clear_stale_index_lock` in front of it; this is the crash that proves it.

    The lock here is Git's own, written and abandoned by a real `git add` killed while it
    held it, through a real clean filter.
    """
    _remote, workspace = _checkout(tmp_path)
    database = await _database(tmp_path)
    store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
    settings = load_settings(workspace_root=tmp_path / "workspaces")
    try:
        await _accept(store)
        # What the killed attempt had written into the checkout and not committed.
        rejected = workspace / "src" / "controllers" / "half-done.js"
        rejected.parent.mkdir(parents=True, exist_ok=True)
        rejected.write_text("// the previous attempt's uncommitted work\n", encoding="utf-8")
        head_before = _git(workspace, "rev-parse", "HEAD")

        run_crash_child(
            "mid_git",
            _scope_spec(
                tmp_path,
                workspace=str(workspace),
                child=_child_reference(workspace, attempt=2),
                locked_path="src/controllers/half-done.js",
                call_log=str(tmp_path / "unused-model.log"),
            ),
        )

        assert (workspace / ".git" / "index.lock").is_file(), (
            "the kill did not happen while Git held the index lock"
        )
        persisted = (await store.get_record(WORKFLOW)).state.child_workflows[REPOSITORY]
        assert persisted.retry_count == 2, "the attempt counter is not the one that was written"
        assert persisted.checkpoint_boundary is WorkflowCheckpointBoundary.AFTER_VALIDATION

        # The filter the dying child installed was its crash vehicle, not repository content.
        # It has to come out before the recovering attempt runs, because it kills its own
        # process group -- and this process is the test suite.
        remove_index_lock_killer(workspace)
        live = LiveChildWorkstreamExecutor(
            settings=settings,
            git_environment={},
            engineer_client=CountingModelDouble({}, tmp_path / "unused-model.log"),
            reviewer_client=CountingModelDouble({}, tmp_path / "unused-model.log"),
            journal=ExternalOperationJournal(database),
            cancellation_token=MockCancellationToken(),
        )

        captured = await live._capture_and_reset(workspace)  # noqa: SLF001

        assert not (workspace / ".git" / "index.lock").exists()
        assert captured is not None and "half-done.js" in captured
        # The next attempt does not repeat the previous one's side effects: its uncommitted
        # work is gone and no second commit was made.
        assert not rejected.exists()
        assert _git(workspace, "status", "--porcelain") == ""
        assert _git(workspace, "rev-parse", "HEAD") == head_before
        assert (await store.get_record(WORKFLOW)).state.child_workflows[REPOSITORY].retry_count == 2

        # A repeated recovery pass over the same checkout changes nothing.
        before = _tree(workspace), _git(workspace, "rev-parse", "HEAD")
        assert await live._capture_and_reset(workspace) is None  # noqa: SLF001
        for _ in range(2):
            await _recovery(database, tmp_path).recover_incomplete_operations()
        assert (_tree(workspace), _git(workspace, "rev-parse", "HEAD")) == before
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# Boundary 7 -- mid-feature, executor gone entirely (tasks 25- and 28-'s contract)
# --------------------------------------------------------------------------------------


async def test_a_feature_whose_executor_vanished_is_continued_by_the_next_claim(
    tmp_path: Path,
) -> None:
    """AB-Feature-108's exact shape, produced by killing a worker rather than writing rows.

    Deliberately changed by task 28-. This test previously asserted that the sweep gave the
    feature a classified terminal state, and said in its own docstring that 28- would change
    it: a feature whose executor died should be picked back up and continued, not tombstoned.
    That is what it asserts now, and it still holds task 25-'s contract -- classification,
    stage and non-empty diagnostics -- on the terminal state that *does* result, once the
    continuation has spent the queue entry's attempts.

    The child process is a real worker: it claims the queue entry, declares the feature
    running with a durable child, and is `SIGKILL`ed. Everything after that is the platform's
    own dispatcher and sweep, run against the rows that worker actually left behind.
    """
    _remote, workspace = _checkout(tmp_path)
    database = await _database(tmp_path)
    store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
    try:
        # Submitted on the platform that is *not* what a client built from configuration
        # would choose, so the recovery assertion below can only pass by reading what this
        # feature persisted rather than what the deployment is set to.
        await _accept(store, agent_platform="anthropic")
        run_crash_child(
            "feature",
            _scope_spec(
                tmp_path,
                lease_seconds=1,
                children=[_child_reference(workspace, attempt=1)],
            ),
        )
        abandoned = (await store.get_record(WORKFLOW)).state
        assert abandoned.status is FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
        assert abandoned.child_workflows[REPOSITORY].status is ChildWorkflowStatus.RUNNING

        # Nothing is abandoned while its lease is live: the sweep waits for the claim the
        # dead worker took to lapse, which is the only evidence available that it is gone.
        assert await store.reconcile_abandoned_runs(stale_after_seconds=1) == []
        await asyncio.sleep(1.2)

        # The claim that follows a lapsed lease. Before task 28- it was dropped as unwanted
        # and the entry closed `succeeded`; it continues the run from its checkpoint. Draining
        # takes several claims rather than one because a claim is one step now -- what this
        # asserts is the outcome, which is a feature that was continued rather than
        # tombstoned, and the counted claims are how many steps that took.
        assert await drain_feature_queue(store) > 1

        continued = (await store.get_record(WORKFLOW)).state
        assert continued.status is FeatureWorkflowStatus.COMPLETED
        assert continued.child_workflows[REPOSITORY].retry_count == 1, (
            "the continuation reset the attempt the dead worker had persisted"
        )
        # The provider survives the restart. A recovered run that resolved its platform from
        # configuration would silently move a feature onto a different SDK mid-workstream and
        # make its own routing and execution records false, so this asserts the client the
        # recovered state actually builds -- with the deployment configured for both
        # platforms, and its own `openai` roles resolvable, so an implementation that read
        # configuration instead of the feature would build an OpenAI client and fail here.
        assert continued.agent_platform == "anthropic"
        _assert_recovered_run_builds_an_anthropic_client(continued)
        timeline = [event for *_, event, _details in await store.timeline(WORKFLOW)]
        assert "feature_run_continued" in timeline

        # The continuation had no artifacts to recover from, so it re-planned and finished.
        # A feature that keeps crashing instead spends the entry's attempts and reaches the
        # sweep, which is where 25-'s contract still has to hold. Put it back in the shape a
        # crash leaves, with the entry spent, and let the sweep decide it.
        crashed = continued.model_copy(deep=True)
        crashed.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
        crashed.failure_summary = None
        await store._replace_state(WORKFLOW, crashed, "child_workflows_started")  # noqa: SLF001
        async with database.session() as session:
            await session.execute(
                update(FeatureExecutionQueueModel)
                .where(FeatureExecutionQueueModel.feature_id == WORKFLOW)
                .values(status=SUCCEEDED, lease_owner=None, lease_expires_at=None)
            )
            await session.commit()
        await asyncio.sleep(1.2)

        decided = await store.reconcile_abandoned_runs(stale_after_seconds=1)

        assert decided == [WORKFLOW]
        state = (await store.get_record(WORKFLOW)).state
        assert state.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        assert not any(
            child.status is ChildWorkflowStatus.RUNNING for child in state.child_workflows.values()
        )
        summary = state.failure_summary
        assert summary is not None
        assert summary.root_classification == FeatureFailureClassification.EXECUTOR_STOPPED.value
        assert summary.stage == FailureStage.RUN_RECOVERY.value
        assert summary.diagnostics, "a terminal state reached by a crash said nothing about it"
        assert any(
            "stopped without writing a terminal status" in item for item in summary.diagnostics
        )
        # And it tells the truth about what happened to it, rather than the advice that was
        # correct while nothing continued a crashed run.
        assert not any("Start a fresh feature" in item for item in summary.diagnostics)
        assert any("continued this run" in item for item in summary.diagnostics)

        # A second sweep must not rewrite a feature it has already decided.
        recorded = state.model_dump(mode="json")
        assert await store.reconcile_abandoned_runs(stale_after_seconds=1) == []
        assert (await store.get_record(WORKFLOW)).state.model_dump(mode="json") == recorded
    finally:
        await database.dispose()


def _assert_recovered_run_builds_an_anthropic_client(state: Any) -> None:
    """Build the live orchestrator's model boundary from a recovered feature's own state."""
    settings = load_settings(
        openai_reasoning_model="gpt-5.6-sol",
        openai_coding_model="gpt-5.6-sol",
        openai_review_model="gpt-5.6-terra",
        openai_scoped_fix_model="gpt-5.3-codex",
        anthropic_reasoning_model="claude-opus-5",
        anthropic_coding_model="claude-opus-5",
        anthropic_review_model="claude-opus-5",
        anthropic_scoped_fix_model="claude-sonnet-5",
    )
    engineer = llm_client_for(
        AgentPlatform(state.agent_platform),
        settings,
        "engineer",
        api_key="sk-recovered",
        model_role=ModelRole.CODING,
    )
    assert isinstance(engineer, AnthropicLLMClient)
    assert engineer.provider == "anthropic"
    assert engineer.model == "claude-opus-5"


# --------------------------------------------------------------------------------------
# Boundary 8 -- a crash costs one step
# --------------------------------------------------------------------------------------


class _StepRecordingOrchestrator(FeatureWorkflowOrchestrator):
    """The surviving half of the crash child's recorder: the same log, no dying."""

    def __init__(self, log: Path) -> None:
        """Bind the log the killed process was already appending to."""
        super().__init__()
        self._log = log

    async def _run_step(self, state: Any, step: Any, *, credentials: Any) -> Any:
        """Append this step to the shared log before running it."""
        with self._log.open("a", encoding="utf-8") as handle:
            handle.write(f"{step.value}\n")
        return await super()._run_step(state, step, credentials=credentials)


def _steps(log: Path) -> list[str]:
    """Return every step both processes ran, in order."""
    return log.read_text(encoding="utf-8").split() if log.exists() else []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "die_after",
    [
        FeatureStep.EXECUTE_WORKSTREAMS.value,
        FeatureStep.INTEGRATION_REVIEW.value,
        FeatureStep.PUBLISH.value,
    ],
)
async def test_a_crash_costs_the_step_it_happened_in_and_no_earlier_one(
    tmp_path: Path, die_after: str
) -> None:
    """The point of stepping, killed rather than simulated.

    A worker claims the entry, advances the feature step by step, and is `SIGKILL`ed the
    moment one named step finishes -- before the claim records anything about it. Everything
    after that is the platform's own dispatcher against the rows that worker left.

    What must hold is that the second process does not repeat a step the first one finished.
    Before this change the unit of a claim was the feature, so a death here cost the product
    manager, reconnaissance, the planner and every repository attempt already made -- 86
    minutes on average, and up to 571.
    """
    database = await _database(tmp_path)
    log = tmp_path / "steps.log"
    store = SqlAlchemyFeatureControlPlane(database, mock_runner=_StepRecordingOrchestrator(log=log))
    try:
        await _accept(store)
        run_crash_child(
            "feature_step",
            _scope_spec(tmp_path, lease_seconds=120, step_log=str(log), die_after=die_after),
        )
        before = _steps(log)
        assert before, "the killed process recorded no step at all"
        assert before[-1] == die_after, "the child died somewhere other than its boundary"

        # The lease the dead worker holds is still live, so nothing may take the feature from
        # it yet -- the claim below is the deployment's own, once that lease is released.
        await _release_dead_claim(database)
        assert await drain_feature_queue(store) >= 1

        after = _steps(log)[len(before) :]
        assert after, "the crashed feature was never picked back up"
        finished = before[:-1]
        assert not (set(after) & set(finished)), (
            "the claim after the crash repeated a step the dead process had already finished: "
            f"ran {after} after {before}"
        )
        assert (await store.get_record(WORKFLOW)).state.status is FeatureWorkflowStatus.COMPLETED
    finally:
        await database.dispose()


async def _release_dead_claim(database: Database) -> None:
    """Expire the lease the killed worker was holding, as the passage of time does."""
    async with database.session() as session:
        await session.execute(
            update(FeatureExecutionQueueModel)
            .where(FeatureExecutionQueueModel.feature_id == WORKFLOW)
            .values(lease_expires_at=datetime.now(UTC) - timedelta(minutes=5))
        )
        await session.commit()


# --------------------------------------------------------------------------------------
# Boundary 6b -- the same crash artifact, met by the ordinary retry that now preserves
# --------------------------------------------------------------------------------------


async def test_the_preserving_retry_clears_a_crash_lock_without_destroying_the_attempt(
    tmp_path: Path,
) -> None:
    """The ordinary retry's capture meets a real crash's lock and keeps the work.

    The boundary above proves the reset path against `.git/index.lock`; this proves the
    path that replaced it for ordinary retries. The capture must clear the abandoned lock,
    record the previous attempt's diff, and leave the worktree exactly as the dead attempt
    left it -- files present, nothing staged -- because those files are now the lineage the
    retry edits rather than evidence to destroy. The dirty-workspace recovery reconciliation
    beside it is untouched: a completed coding receipt still short-circuits both paths.
    """
    _remote, workspace = _checkout(tmp_path)
    database = await _database(tmp_path)
    store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
    settings = load_settings(workspace_root=tmp_path / "workspaces")
    try:
        await _accept(store)
        rejected = workspace / "src" / "controllers" / "half-done.js"
        rejected.parent.mkdir(parents=True, exist_ok=True)
        rejected.write_text("// the previous attempt's uncommitted work\n", encoding="utf-8")
        head_before = _git(workspace, "rev-parse", "HEAD")

        run_crash_child(
            "mid_git",
            _scope_spec(
                tmp_path,
                workspace=str(workspace),
                child=_child_reference(workspace, attempt=2),
                locked_path="src/controllers/half-done.js",
                call_log=str(tmp_path / "unused-model.log"),
            ),
        )

        assert (workspace / ".git" / "index.lock").is_file(), (
            "the kill did not happen while Git held the index lock"
        )
        remove_index_lock_killer(workspace)
        live = LiveChildWorkstreamExecutor(
            settings=settings,
            git_environment={},
            engineer_client=CountingModelDouble({}, tmp_path / "unused-model.log"),
            reviewer_client=CountingModelDouble({}, tmp_path / "unused-model.log"),
            journal=ExternalOperationJournal(database),
            cancellation_token=MockCancellationToken(),
        )

        captured = await live._capture_attempt(workspace)  # noqa: SLF001

        assert not (workspace / ".git" / "index.lock").exists()
        assert captured is not None and "half-done.js" in captured
        # The preserved path's whole point: the dead attempt's work survives its own crash.
        assert rejected.is_file()
        assert _git(workspace, "rev-parse", "HEAD") == head_before
        # Nothing left staged: the commit adapter refuses pre-existing staged content, so
        # the capture must not manufacture that condition.
        assert _git(workspace, "diff", "--cached", "--name-only") == ""

        # Two more passes over the same checkout are idempotent, like every recovery here.
        before = _tree(workspace), _git(workspace, "rev-parse", "HEAD")
        for _ in range(2):
            repeat = await live._capture_attempt(workspace)  # noqa: SLF001
            assert repeat is not None and "half-done.js" in repeat
            await _recovery(database, tmp_path).recover_incomplete_operations()
        assert (_tree(workspace), _git(workspace, "rev-parse", "HEAD")) == before
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# Boundary 9 -- killed inside a pre-coding model call, its journal row still RUNNING
# --------------------------------------------------------------------------------------

_PLANNING_CALL_TYPES = (
    ExternalOperationType.RUN_PRODUCT_MANAGER,
    ExternalOperationType.RUN_REPOSITORY_RECON,
    ExternalOperationType.RUN_CLARIFICATION_GROUNDING,
    ExternalOperationType.RUN_FEATURE_PLANNER,
)


class _RepeatingReconClient:
    """Answer every reconnaissance call with the same fixed, valid response."""

    def __init__(self) -> None:
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
    ) -> Any:
        from adapters.llm_adapter import LLMResponse

        del instructions, input_text
        self.calls += 1
        # Every path here exists in the `node_javascript_repository` fixture the crash
        # origin is built from; the agent validates evidence paths against the checkout.
        payload = {
            "summary": "An Express-style service whose routes live under src/routes.",
            "source_areas": ["src"],
            "test_areas": ["test"],
            "conventions": [
                {
                    "convention_id": "route",
                    "kind": "route",
                    "description": "Routes are plain modules assembled in src/index.js.",
                    "evidence_paths": ["src/routes/status.js"],
                    "wiring_path": "src/index.js",
                }
            ],
            "shared_utilities": [],
            "contradicted_premises": [],
        }
        return LLMResponse(
            response_id="recon-resume-response",
            model="crash-suite-recon-model",
            output_text=json.dumps(payload),
            input_tokens=None,
            output_tokens=None,
            provider="test-provider",
            reasoning_effort="high",
        )


def _normalized_feature_state(dump: dict[str, Any], feature_id: str) -> str:
    """One feature's durable state as bytes, with identity and wall-clock scrubbed.

    Everything else -- statuses, artifact ids and payloads, child records, checkpoint
    boundaries, transition reasons -- must match to the byte between the run whose journal
    rows were kept and the run whose rows were removed. Time and the feature's own name are
    the only things two otherwise identical runs cannot share.
    """
    import re

    dump = dict(dump)
    # The planning clocks are wall-clock measurements, and parallel workstreams append
    # their artifacts in whichever order the loop scheduled them -- both vary between any
    # two runs of the same feature and say nothing about the journal rows.
    dump["planning_wall_seconds"] = "CLOCK" if dump.get("planning_wall_seconds") else None
    dump["planning_provider_fault_seconds"] = (
        "CLOCK" if dump.get("planning_provider_fault_seconds") is not None else None
    )

    # The attempt's own clocks -- in each result's metadata and folded onto the child
    # records -- are wall-clock measurements too: two identical runs cannot share them,
    # exactly like the planning clocks above.
    def scrub_clocks(record: Any) -> None:
        if not isinstance(record, dict):
            return
        for key in ("runtime_wall_seconds", "runtime_charged_seconds", "fault_seconds_excluded"):
            if record.get(key) is not None:
                record[key] = "CLOCK"

    children = dump.get("child_workflows")
    if isinstance(children, dict):
        for child in children.values():
            scrub_clocks(child)
    artifacts = dump.get("artifacts")
    if isinstance(artifacts, list):
        for item in artifacts:
            scrub_clocks(item.get("metadata") if isinstance(item, dict) else None)
        dump["artifacts"] = sorted(
            artifacts,
            key=lambda item: str(item.get("artifact_id", "")) if isinstance(item, dict) else "",
        )
    text = json.dumps(dump, sort_keys=True, default=str)
    text = text.replace(feature_id, "FEATURE")
    # The branch segment carries a short hash of the feature id, which the line above
    # cannot reach; it is identity, not behavior.
    text = re.sub(r"FEATURE-[0-9a-f]{6,}", "FEATURE", text)
    # Content fingerprints hash inputs that themselves carry the feature's name (routing
    # inputs, branch names), so no two differently named features can share one. The
    # content they fingerprint is compared directly by everything else in the dump.
    text = re.sub(r"sha256:[0-9a-f]{64}", "sha256:X", text)
    # The human reference is allocated by a database sequence, and the provider double
    # numbers pull requests across the whole test -- both are identity the two features
    # cannot share, not behavior.
    text = re.sub(r"AB-Feature-\d+", "REF", text)
    text = re.sub(r"pull/\d+", "pull/N", text)
    text = re.sub(r'_number": \d+', '_number": 0', text)
    return re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+00:00|Z)?", "TIME", text)


async def _accept_named(store: SqlAlchemyFeatureControlPlane, feature_id: str) -> None:
    """Accept one feature under its own id, through the real path."""
    request = StartFeatureRequest.model_validate({**feature_payload(), "feature_id": feature_id})
    await store.start(
        request,
        idempotency_key=f"crash-{feature_id}",
        credentials=CREDENTIALS,
        owner_id="platform-admin",
    )


async def _release_named_claim(database: Database, feature_id: str) -> None:
    """Expire the lease the killed worker holds on one feature."""
    async with database.session() as session:
        await session.execute(
            update(FeatureExecutionQueueModel)
            .where(FeatureExecutionQueueModel.feature_id == feature_id)
            .values(lease_expires_at=datetime.now(UTC) - timedelta(minutes=5))
        )
        await session.commit()


async def test_a_worker_killed_mid_recon_recovers_identically_with_and_without_the_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-coding journal rows are record, never a second reconciliation input.

    Two identical features are killed at the same instant -- inside the backend
    repository's reconnaissance model call, whose `run_repository_recon` row is claimed and
    RUNNING. One keeps its planning-call rows; the other has them deleted before recovery,
    which is exactly the deployment as it was before this task. Recovery and resume must
    then do the same thing to both: the sweep abandons the orphaned row and touches no
    feature state, the resumed step calls the model again rather than replaying anything
    out of a row, and the two features end in durable states identical to the byte once
    the feature's own name and wall-clock time are scrubbed.
    """
    from sqlalchemy import delete as sql_delete

    import services.feature_runtime as feature_runtime_module
    from storage.models import ExternalOperationModel

    database = await _database(tmp_path)
    remote, _source = _origin(tmp_path)
    kept, scrubbed = "feature-recon-kept", "feature-recon-scrubbed"
    journal = ExternalOperationJournal(database, stale_after_seconds=_STALE_AFTER_SECONDS)
    resume_client = _RepeatingReconClient()
    monkeypatch.setattr(feature_runtime_module, "_git_source_url", lambda _url: str(remote))
    resume_orchestrator = FeatureWorkflowOrchestrator(
        reconnaissance=feature_runtime_module.LiveRepositoryReconnaissance(
            settings=load_settings(workspace_root=tmp_path / "workspaces"),
            git_environment={},
            recon_client=resume_client,
            journal=journal,
            cancellation_token=MockCancellationToken(),
        ),
        workspace_root=tmp_path / "workspaces",
        operation_executor_factory=lambda state: ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(
                workflow_id=state.workflow_id, feature_id=state.feature_id
            ),
            heartbeat_seconds=3_600.0,
        ),
    )
    store = SqlAlchemyFeatureControlPlane(database, mock_runner=resume_orchestrator)
    try:
        for feature_id in (kept, scrubbed):
            await _accept_named(store, feature_id)
            call_log = tmp_path / f"{feature_id}-dead-model.log"
            run_crash_child(
                "recon",
                {
                    "database_url": f"sqlite+aiosqlite:///{tmp_path / 'crash.db'}",
                    "feature_id": feature_id,
                    "workflow_id": feature_id,
                    "workspace_root": str(tmp_path / "workspaces"),
                    "origin": str(remote),
                    "call_log": str(call_log),
                    "lease_seconds": 120,
                },
            )
            assert DyingModelDouble(call_log).invocations == 1, (
                "the dead process must have died inside its first reconnaissance call"
            )

        async def planning_rows(feature_id: str) -> list[Any]:
            return await journal.list_operations_for_feature(
                feature_id, operation_types=list(_PLANNING_CALL_TYPES)
            )

        orphan = next(
            row
            for row in await planning_rows(kept)
            if row.operation_type is ExternalOperationType.RUN_REPOSITORY_RECON
        )
        assert orphan.status is ExternalOperationStatus.RUNNING
        assert orphan.started_at is not None and orphan.heartbeat_at is not None

        # "Without the rows" is the deployment before this task: remove one feature's
        # planning-call rows entirely. Its clone rows stay -- those predate this task.
        async with database.session() as session:
            await session.execute(
                sql_delete(ExternalOperationModel).where(
                    ExternalOperationModel.feature_id == scrubbed,
                    ExternalOperationModel.operation_type.in_(_PLANNING_CALL_TYPES),
                )
            )
            await session.commit()
        assert await planning_rows(scrubbed) == []

        # Recovery sweeps twice, as the deployment's periodic sweep does. It abandons the
        # orphaned row in place and leaves every feature's durable state byte-identical.
        states_before = {
            feature_id: json.dumps(
                (await store.get_record(feature_id)).state.model_dump(mode="json"),
                sort_keys=True,
            )
            for feature_id in (kept, scrubbed)
        }
        await _recovery(database, tmp_path).recover_incomplete_operations()
        abandoned = await journal.get(orphan.operation_id)
        first_pass = _durable_state(abandoned)
        assert abandoned.status is ExternalOperationStatus.CANCELLED
        assert abandoned.error_code == "workspace_operation_abandoned"
        await _recovery(database, tmp_path).recover_incomplete_operations()
        assert _durable_state(await journal.get(orphan.operation_id)) == first_pass
        for feature_id in (kept, scrubbed):
            assert (
                json.dumps(
                    (await store.get_record(feature_id)).state.model_dump(mode="json"),
                    sort_keys=True,
                )
                == states_before[feature_id]
            ), "the recovery sweep must not touch feature state"

        for feature_id in (kept, scrubbed):
            await _release_named_claim(database, feature_id)
        assert await drain_feature_queue(store) >= 2

        # The resumed step asked the model again in both runs -- one call per feature, and
        # nothing was replayed out of the abandoned row.
        assert resume_client.calls == 2
        finals = {}
        for feature_id in (kept, scrubbed):
            state = (await store.get_record(feature_id)).state
            assert state.status is FeatureWorkflowStatus.COMPLETED
            finals[feature_id] = _normalized_feature_state(
                state.model_dump(mode="json"), feature_id
            )
        assert finals[kept] == finals[scrubbed], (
            "recovery with the journal rows present must reproduce recovery without them"
        )
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# Boundary 10 -- after commit and push, before the child result checkpoint
# --------------------------------------------------------------------------------------

_PUBLICATION_ENGINEER_RESPONSE: dict[str, Any] = {
    "summary": "Update the server status route.",
    "files": [
        {
            "path": "src/routes/status.js",
            "content": (
                "function statusRoute(req, res) {\n"
                "  res.json({ status: 'ok', crashed: false });\n"
                "}\n\nmodule.exports = { statusRoute };\n"
            ),
        },
        {
            "path": "test/status.test.js",
            "content": (
                "const { test } = require('node:test');\n"
                "const assert = require('node:assert');\n"
                "const { statusRoute } = require('../src/routes/status');\n\n"
                "test('status route responds', () => {\n"
                "  assert.ok(typeof statusRoute === 'function');\n"
                "});\n"
            ),
        },
    ],
}


async def test_a_publication_killed_before_its_result_checkpoint_recovers_without_the_model(
    tmp_path: Path,
) -> None:
    """The true crash window still recovers: no model call, one commit, one push.

    The Engineer wrote its files, the reviewer approved, and the real commit and real push
    both journaled SUCCEEDED before the worker was killed -- so the durable child state
    still says the attempt never finished, and the receipt in the CREATE_COMMIT row is the
    only record of the outcome. Recovery must complete the publication from that receipt
    without asking any model anything, and running it twice must not create a second
    commit or a second push.
    """
    remote, _source = _origin(tmp_path)
    workspace = tmp_path / "workspaces" / real_support.FEATURE_ID / real_support.REPOSITORY_ID
    workspace.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(("git", "clone", str(remote), str(workspace)), check=True, capture_output=True)
    _git(workspace, "config", "user.email", "crash@example.com")
    _git(workspace, "config", "user.name", "Crash Suite")
    _git(workspace, "config", "commit.gpgsign", "false")
    _git(workspace, "checkout", "-b", real_support.BRANCH)
    database = await _database(tmp_path)

    request = StartFeatureRequest.model_validate(real_support.feature_payload())
    feature = _initial_feature_state(real_support.FEATURE_ID, request)
    technical_prd = real_support.technical_prd_artifact()
    contract = real_support.contract_artifact()
    feature.artifacts.extend([technical_prd, contract])
    repository = next(
        item
        for item in feature.repository_specs
        if item.repository_id == real_support.REPOSITORY_ID
    )
    workstream = real_support.workstream_plan()
    child = ChildWorkflowReference(
        child_workflow_id=f"{real_support.FEATURE_ID}:{real_support.REPOSITORY_ID}",
        repository_id=real_support.REPOSITORY_ID,
        workstream_id=real_support.REPOSITORY_ID,
        status=ChildWorkflowStatus.RUNNING,
        branch_name=real_support.BRANCH,
        workspace_path=str(workspace),
        retry_count=1,
    )
    engineer_log = tmp_path / "engineer-calls.log"
    reviewer_log = tmp_path / "reviewer-calls.log"
    review_response = real_support.review_payload(verdict="approved")
    try:
        run_crash_child(
            "publication",
            {
                "database_url": f"sqlite+aiosqlite:///{tmp_path / 'crash.db'}",
                "workspace_root": str(tmp_path / "workspaces"),
                "repository_id": real_support.REPOSITORY_ID,
                "feature": feature.model_dump(mode="json"),
                "workstream": workstream.model_dump(mode="json"),
                "child": child.model_dump(mode="json"),
                "engineer_response": _PUBLICATION_ENGINEER_RESPONSE,
                "review_response": review_response,
                "engineer_log": str(engineer_log),
                "reviewer_log": str(reviewer_log),
            },
        )

        # What the dead worker actually left behind: one model call each, a committed and
        # pushed branch, and SUCCEEDED journal rows -- with no child result anywhere.
        assert CountingModelDouble(_PUBLICATION_ENGINEER_RESPONSE, engineer_log).invocations == 1
        assert CountingModelDouble(review_response, reviewer_log).invocations == 1
        journal = ExternalOperationJournal(database)
        operations = await journal.list_operations_for_workflow(feature.workflow_id)
        commits = [
            item
            for item in operations
            if item.operation_type is ExternalOperationType.CREATE_COMMIT
        ]
        pushes = [
            item for item in operations if item.operation_type is ExternalOperationType.PUSH_BRANCH
        ]
        assert len(commits) == 1 and commits[0].status is ExternalOperationStatus.SUCCEEDED
        assert bool(commits[0].safe_metadata.get("publication_receipt"))
        assert len(pushes) == 1 and pushes[0].status is ExternalOperationStatus.SUCCEEDED
        published_sha = _git(workspace, "rev-parse", "HEAD")
        assert _remote_sha(workspace, real_support.BRANCH) == published_sha

        engineer = CountingModelDouble(_PUBLICATION_ENGINEER_RESPONSE, engineer_log)
        reviewer = CountingModelDouble(review_response, reviewer_log)
        live = LiveChildWorkstreamExecutor(
            settings=load_settings(workspace_root=tmp_path / "workspaces"),
            git_environment={},
            engineer_client=engineer,
            reviewer_client=reviewer,
            journal=journal,
            cancellation_token=MockCancellationToken(),
            process_runner=SucceedingProcessRunner(),
        )
        arguments: dict[str, Any] = {
            "feature": feature,
            "repository": repository,
            "workstream": workstream,
            "child": child,
            "technical_prd": technical_prd,
            "contract": contract,
            "feedback": [],
            "credentials": CREDENTIALS,
        }

        recovered = await live.run(**arguments)

        assert recovered.result.status == "approved"
        assert recovered.code_completion is not None
        assert recovered.code_completion.metadata["publication_recovered"] is True
        assert engineer.invocations == 1, "recovery asked the coding model again"
        assert reviewer.invocations == 1, "recovery asked the reviewer again"

        # In production the sweep and the next claim can each meet this receipt again; a
        # second recovery pass must reuse the same commit and push, not repeat them.
        replay = await live.run(**arguments)
        assert replay.result.status == "approved"
        assert engineer.invocations == 1
        settled = await journal.list_operations_for_workflow(feature.workflow_id)
        kinds = [item.operation_type for item in settled]
        assert kinds.count(ExternalOperationType.RUN_CODING_EXECUTOR) == 1
        assert kinds.count(ExternalOperationType.WRITE_FILE_CHANGES) == 1
        assert kinds.count(ExternalOperationType.CREATE_COMMIT) == 1
        assert kinds.count(ExternalOperationType.PUSH_BRANCH) == 1
        assert _git(workspace, "rev-parse", "HEAD") == published_sha
        assert _remote_sha(workspace, real_support.BRANCH) == published_sha
    finally:
        await database.dispose()
