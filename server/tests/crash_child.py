"""The process the crash-window suite kills.

Run as ``python -m tests.crash_child <boundary> <json spec>``. Each entry point below drives
the platform's real adapters against a real workspace, a real bare Git remote, a real
durable database and a state-holding provider double -- then dies, without unwinding, at the
boundary it is named for.

This module exists because a mocked exception is not a crash. Every recovery path in the
platform was previously verified by calling a recovery function directly, in the same
process, with a hand-built `ExternalOperation`. Nothing proved that the durable evidence a
dying process actually leaves is the evidence recovery reads.

Nothing here is imported by the test process itself; it only ever runs as `__main__` in a
child. Keeping it in its own module is what lets the child be a real, separate process.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import services.feature_runtime as feature_runtime_module
from adapters.interruptible_git import InterruptibleGitService
from adapters.llm_adapter import ResponsesCodingExecutor
from api.control_plane import RequestScopedCredentials
from configs.settings import load_settings
from services.cancellation import MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.feature_queue import DatabaseFeatureExecutionQueue
from services.feature_runtime import LiveChildWorkstreamExecutor, LiveRepositoryReconnaissance
from services.journaled_github import JournaledGitHubService
from state.enums import FeatureWorkflowStatus
from state.external_operations import ExternalOperationType
from state.feature_models import ChildWorkflowReference
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from storage.feature_store import SqlAlchemyFeatureControlPlane
from tests.crash_support import (
    AFTER_JOURNAL_SUCCESS,
    BEFORE_JOURNAL_SUCCESS,
    CountingModelDouble,
    CrashingJournal,
    DyingModelDouble,
    PersistentGitHubDouble,
    die_now,
    install_index_lock_killer,
)
from workflows.feature_workflow import FeatureWorkflowOrchestrator

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)


def _scope(spec: dict[str, Any]) -> ExternalOperationScope:
    """The stable ownership identifiers every operation in one boundary shares."""
    return ExternalOperationScope(
        workflow_id=spec["workflow_id"],
        feature_id=spec.get("feature_id"),
        child_workflow_id=spec.get("child_workflow_id"),
        repository_id=spec.get("repository_id"),
    )


def _executor(journal: ExternalOperationJournal, spec: dict[str, Any]) -> ExternalOperationExecutor:
    """Build the real journaling executor the adapters are given in production."""
    return ExternalOperationExecutor(
        journal=journal,
        cancellation_token=MockCancellationToken(),
        scope=_scope(spec),
        # Long enough that a heartbeat never fires inside these short boundaries and
        # rewrites the liveness the recovering process is about to read.
        heartbeat_seconds=3_600.0,
    )


def _git(
    journal: ExternalOperationJournal, spec: dict[str, Any], *, workspace_root: Path
) -> InterruptibleGitService:
    """Build the real Git service, which shells out to real `git` throughout."""
    return InterruptibleGitService(
        default_branch=spec.get("default_branch", "main"),
        cancellation_token=MockCancellationToken(),
        operation_executor=_executor(journal, spec),
        workspace_root=workspace_root,
    )


async def _clone(spec: dict[str, Any]) -> None:
    """Boundary 1: clone the repository for real, then die before the journal records it."""
    database = Database(spec["database_url"])
    journal = CrashingJournal(
        database, crash_on=ExternalOperationType.CLONE_REPOSITORY, crash_at=BEFORE_JOURNAL_SUCCESS
    )
    service = _git(journal, spec, workspace_root=Path(spec["workspace_root"]))
    await service.clone(spec["source_url"], spec["destination"])
    raise AssertionError("the clone boundary returned without crashing")


async def _coding(spec: dict[str, Any]) -> None:
    """Boundary 2: write the Engineer's files, record the receipt, then die before review."""
    database = Database(spec["database_url"])
    journal = CrashingJournal(
        database,
        crash_on=ExternalOperationType.RUN_CODING_EXECUTOR,
        crash_at=AFTER_JOURNAL_SUCCESS,
    )
    executor = ResponsesCodingExecutor(
        CountingModelDouble(spec["response"], Path(spec["call_log"]))
    )
    await executor.execute(
        workspace_root=spec["workspace"],
        instructions=spec["instructions"],
        input_text=spec["input_text"],
        operation_executor=_executor(journal, spec),
    )
    raise AssertionError("the coding boundary returned without crashing")


async def _commit(spec: dict[str, Any]) -> None:
    """Boundary 3: create the commit for real, then die before its SHA is written down."""
    database = Database(spec["database_url"])
    journal = CrashingJournal(
        database, crash_on=ExternalOperationType.CREATE_COMMIT, crash_at=BEFORE_JOURNAL_SUCCESS
    )
    service = _git(journal, spec, workspace_root=Path(spec["workspace_root"]))
    await service.commit(spec["workspace"], spec["message"], files=spec["files"])
    raise AssertionError("the commit boundary returned without crashing")


async def _push(spec: dict[str, Any]) -> None:
    """Boundary 4: push to the real remote, then die before the journal records success."""
    database = Database(spec["database_url"])
    journal = CrashingJournal(
        database, crash_on=ExternalOperationType.PUSH_BRANCH, crash_at=BEFORE_JOURNAL_SUCCESS
    )
    service = _git(journal, spec, workspace_root=Path(spec["workspace_root"]))
    await service.push(spec["workspace"], spec["branch"])
    raise AssertionError("the push boundary returned without crashing")


async def _pull_request(spec: dict[str, Any]) -> None:
    """Boundary 5: open the pull request, then die inside the provider call that opened it.

    The crash is in the provider double rather than in the journal, because that is the
    shape this boundary actually has: the remote side effect is complete and the process
    that caused it never came back to write it down.
    """
    database = Database(spec["database_url"])
    journal = ExternalOperationJournal(database)
    service = JournaledGitHubService(
        PersistentGitHubDouble(Path(spec["provider_state"]), crash_after_create=True),
        operation_executor=_executor(journal, spec),
    )
    await service.create_pull_request(
        spec["repository"],
        title=spec["title"],
        body=spec["body"],
        source_branch=spec["source_branch"],
        target_branch=spec["target_branch"],
    )
    raise AssertionError("the pull-request boundary returned without crashing")


async def _publication(spec: dict[str, Any]) -> None:
    """Boundary 10: publish an approved child for real, then die after the push's journal write.

    The window `_recover_approved_publication` exists for, opened by a real kill: the
    Engineer wrote files, the reviewer approved, the commit and the push both landed and
    both journal rows say SUCCEEDED -- and the process died before the child result could
    be checkpointed. The durable child state the surviving test then runs recovery with
    still says the attempt never finished.
    """
    from artifacts.schemas import (
        IntegrationContractArtifact,
        RepositoryWorkstreamPlan,
        TechnicalPRDArtifact,
    )
    from state.feature_models import FeatureWorkflowSnapshot
    from tests.crash_support import SucceedingProcessRunner

    database = Database(spec["database_url"])
    journal = CrashingJournal(
        database, crash_on=ExternalOperationType.PUSH_BRANCH, crash_at=AFTER_JOURNAL_SUCCESS
    )
    # The same JSON-mode rehydration the feature store uses: the snapshot's models are
    # strict, so a python-mode validate rejects the JSON the spec necessarily carries.
    feature = FeatureWorkflowSnapshot.model_validate_json(json.dumps(spec["feature"]))
    workstream = RepositoryWorkstreamPlan.model_validate_json(json.dumps(spec["workstream"]))
    child = ChildWorkflowReference.model_validate(spec["child"])
    repository = next(
        item for item in feature.repository_specs if item.repository_id == spec["repository_id"]
    )
    technical_prd = next(
        item for item in feature.artifacts if isinstance(item, TechnicalPRDArtifact)
    )
    contract = next(
        item for item in feature.artifacts if isinstance(item, IntegrationContractArtifact)
    )
    executor = LiveChildWorkstreamExecutor(
        settings=load_settings(workspace_root=Path(spec["workspace_root"])),
        git_environment={},
        engineer_client=CountingModelDouble(spec["engineer_response"], Path(spec["engineer_log"])),
        reviewer_client=CountingModelDouble(spec["review_response"], Path(spec["reviewer_log"])),
        journal=journal,
        cancellation_token=MockCancellationToken(),
        process_runner=SucceedingProcessRunner(),
    )
    await executor.run(
        feature=feature,
        repository=repository,
        workstream=workstream,
        child=child,
        technical_prd=technical_prd,
        contract=contract,
        feedback=[],
        credentials=CREDENTIALS,
    )
    raise AssertionError("the publication boundary returned without crashing")


async def _mid_git(spec: dict[str, Any]) -> None:
    """Boundary 6: persist the attempt, then die inside `git add` holding the index lock.

    The reset the previous attempt runs is the platform's own `_capture_and_reset`, and the
    kill comes out of a real Git clean filter, so the `.git/index.lock` left behind is one
    Git wrote and abandoned. This is what terminating a process group does to a checkout.
    """
    database = Database(spec["database_url"])
    store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
    state = (await store.get_record(spec["feature_id"])).state.model_copy(deep=True)
    state.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
    state.child_workflows = {
        spec["repository_id"]: ChildWorkflowReference.model_validate(spec["child"])
    }
    await store._replace_state(  # noqa: SLF001
        spec["feature_id"], state, "child_validation_completed"
    )

    executor = LiveChildWorkstreamExecutor(
        settings=load_settings(workspace_root=Path(spec["workspace_root"])),
        git_environment={},
        engineer_client=CountingModelDouble({}, Path(spec["call_log"])),
        reviewer_client=CountingModelDouble({}, Path(spec["call_log"])),
        journal=ExternalOperationJournal(database),
        cancellation_token=MockCancellationToken(),
    )
    # Installed here rather than by the test, and naming this process: the filter kills its
    # own process group, and in the test process that group is the suite.
    install_index_lock_killer(
        Path(spec["workspace"]), path=spec["locked_path"], worker_pid=os.getpid()
    )
    await executor._capture_and_reset(Path(spec["workspace"]))  # noqa: SLF001
    raise AssertionError("the mid-git boundary returned without crashing")


async def _feature(spec: dict[str, Any]) -> None:
    """Boundary 7: claim the feature, declare it running, then vanish with no terminal status."""
    database = Database(spec["database_url"])
    store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
    queue = DatabaseFeatureExecutionQueue(database)
    claimed = await queue.claim(owner="crash-suite-worker", lease_seconds=spec["lease_seconds"])
    if claimed is None:
        raise AssertionError("the feature boundary found nothing to claim")
    state = (await store.get_record(spec["feature_id"])).state.model_copy(deep=True)
    state.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
    state.current_agent = "child_workflows"
    state.updated_at = datetime.now(UTC)
    state.child_workflows = {
        item["repository_id"]: ChildWorkflowReference.model_validate(item)
        for item in spec["children"]
    }
    await store._replace_state(spec["feature_id"], state, "child_workflows_started")  # noqa: SLF001
    die_now()


class _StepRecordingOrchestrator(FeatureWorkflowOrchestrator):
    """Record every step this process runs, and die when the named one finishes.

    The file is the point. Two processes run steps of the same feature -- the one that is
    killed and the one that claims the entry afterwards -- and the only way to assert what a
    crash cost is to read what each of them actually did. In-process counters die with the
    process, which is the whole difficulty a crash test exists to reproduce.
    """

    def __init__(self, *, log: Path, die_after: str) -> None:
        """Bind the shared log and the step whose completion ends this process."""
        super().__init__()
        self._log = log
        self._die_after = die_after

    async def _run_step(self, state: Any, step: Any, *, credentials: Any) -> Any:
        """Append this step to the shared log, run it, and die if it is the named one."""
        with self._log.open("a", encoding="utf-8") as handle:
            handle.write(f"{step.value}\n")
        result = await super()._run_step(state, step, credentials=credentials)
        if step.value == self._die_after:
            # After the step's own durable writes and before the claim records its outcome:
            # the window a deploy or an OOM kill actually opens.
            die_now()
        return result


async def _feature_step(spec: dict[str, Any]) -> None:
    """Boundary 8: advance the feature step by step, and vanish inside one named step."""
    database = Database(spec["database_url"])
    log = Path(spec["step_log"])
    store = SqlAlchemyFeatureControlPlane(
        database,
        mock_runner=_StepRecordingOrchestrator(log=log, die_after=spec["die_after"]),
    )
    queue = DatabaseFeatureExecutionQueue(database)
    while True:
        claimed = await queue.claim(owner="crash-suite-worker", lease_seconds=spec["lease_seconds"])
        if claimed is None:
            raise AssertionError("the step boundary ran out of entries before it crashed")
        await store.execute_queued(
            spec["feature_id"],
            credentials=CREDENTIALS,
            intent=claimed.intent,
            payload=claimed.payload,
        )
        if not await store.feature_owes_another_step(spec["feature_id"], intent=claimed.intent):
            raise AssertionError("the feature finished without reaching the crash boundary")
        await queue.advance_to_next_step(spec["feature_id"])


async def _recon(spec: dict[str, Any]) -> None:
    """Boundary 9: die inside a reconnaissance model call, its journal row still RUNNING.

    This is AB-Feature-173's dark stretch reproduced as a kill: the checkout was cloned,
    the model was asked, and the process holding the call vanished. What it leaves is a
    claimed `run_repository_recon` row with a start and a heartbeat and no end -- and a
    `run_product_manager` row from the step before, already succeeded. The surviving test
    proves recovery treats those rows as record, never as something to replay.
    """
    database = Database(spec["database_url"])
    journal = ExternalOperationJournal(database)
    # The payload's token-free GitHub URL resolves to the test's real bare origin, at the
    # same module seam the live path resolves clone URLs through.
    feature_runtime_module._git_source_url = (  # noqa: SLF001
        lambda repository_url: str(spec["origin"])  # noqa: ARG005
    )
    reconnaissance = LiveRepositoryReconnaissance(
        settings=load_settings(workspace_root=Path(spec["workspace_root"])),
        git_environment={},
        recon_client=DyingModelDouble(Path(spec["call_log"])),
        journal=journal,
        cancellation_token=MockCancellationToken(),
    )
    orchestrator = FeatureWorkflowOrchestrator(
        reconnaissance=reconnaissance,
        workspace_root=Path(spec["workspace_root"]),
        operation_executor_factory=lambda state: ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(
                workflow_id=state.workflow_id, feature_id=state.feature_id
            ),
            heartbeat_seconds=3_600.0,
        ),
    )
    store = SqlAlchemyFeatureControlPlane(database, mock_runner=orchestrator)
    queue = DatabaseFeatureExecutionQueue(database)
    while True:
        claimed = await queue.claim(owner="crash-suite-worker", lease_seconds=spec["lease_seconds"])
        if claimed is None:
            raise AssertionError("the recon boundary found nothing to claim")
        await store.execute_queued(
            spec["feature_id"],
            credentials=CREDENTIALS,
            intent=claimed.intent,
            payload=claimed.payload,
        )
        if not await store.feature_owes_another_step(spec["feature_id"], intent=claimed.intent):
            raise AssertionError("the feature finished without reaching the recon call")
        await queue.advance_to_next_step(spec["feature_id"])


_BOUNDARIES = {
    "clone": _clone,
    "coding": _coding,
    "commit": _commit,
    "push": _push,
    "pull_request": _pull_request,
    "publication": _publication,
    "mid_git": _mid_git,
    "feature": _feature,
    "feature_step": _feature_step,
    "recon": _recon,
}


def main() -> None:
    """Run one boundary, and never return normally if it did its job."""
    boundary, encoded = sys.argv[1], sys.argv[2]
    asyncio.run(_BOUNDARIES[boundary](json.loads(encoded)))


if __name__ == "__main__":
    main()
