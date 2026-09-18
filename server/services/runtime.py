"""Shared live-execution building blocks: workspaces, preflight, review, and Git access."""

from __future__ import annotations

import inspect
import os
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from adapters.git_adapter import GitService
from adapters.llm_adapter import LLMClient
from agents.product_manager.agent import AttachmentContentSource
from agents.reviewer.agent import ReviewerAgent
from artifacts.schemas import CodeCompletionArtifact
from configs.model_roles import ModelRole
from configs.settings import Settings
from prompts.prompt_loader import PromptLoader
from services.cancellation import CancellationToken
from services.external_operations import ExternalOperationExecutor
from services.process_runner import ProcessRunner
from state.models import AgentState
from tools.repository_preflight import RepositoryPreflight, RepositoryPreflightResult
from tools.resolved_issue_ledger import SettledQuestion
from tools.validation_tools import ValidationPlanBuilder, WorkspaceValidationTools


class WorkflowAgent(Protocol):
    """One artifact-producing agent, as the pieces below hand work to it."""

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Return the partial state update for one agent invocation."""


class RuntimeConfigurationError(RuntimeError):
    """Raised when a live workflow lacks a required request-scoped credential or workspace.

    `diagnostics` is how one raise site opts its own message into durable state. Recording
    only the type stays the default, because an arbitrary message can carry a credential, a
    workspace path or subprocess output -- and `safe_error_diagnostics` reads this attribute
    before it falls back to that default.

    A site that passes diagnostics is asserting something specific: that its text is composed
    from platform constants and platform-generated identifiers, and from nothing a checkout,
    a provider or the environment supplied. Use `_recordable_configuration_error` rather than
    setting it by hand, so the assertion is greppable.
    """

    # A class attribute, not an assignment in `__init__`. Three subclasses below this one
    # build their own diagnostics and set them *before* delegating here, so assigning
    # unconditionally would erase them -- which is what silently emptied
    # `WorkspaceCapacityError.diagnostics` and cost two retention tests their evidence.
    diagnostics: tuple[str, ...] = ()

    def __init__(self, message: str, *, diagnostics: Sequence[str] = ()) -> None:
        """Keep the message for logs, and record it durably only when a caller says it is safe."""
        super().__init__(message)
        if diagnostics:
            self.diagnostics = tuple(diagnostics)


def _recordable_configuration_error(message: str) -> RuntimeConfigurationError:
    """Build a configuration error whose message is safe to keep in durable state.

    AB-Feature-111's console failed six times in a row and every attempt recorded exactly
    "The child workstream raised RuntimeConfigurationError and could not complete." The type
    name alone, for a failure whose cause the platform had already composed into a sentence.
    Which of eight raise sites fired was not recoverable from the database at all.
    """
    return RuntimeConfigurationError(message, diagnostics=(message,))


def _changed_test_paths(state: AgentState) -> list[str]:
    """Return the test files the most recent completion in this state reported writing.

    The newest completion, not the first: a retry appends its own, and narrowing the run to
    an earlier attempt's files would test source that no longer exists.
    """
    for artifact in reversed(state.get("artifacts", [])):
        if isinstance(artifact, CodeCompletionArtifact):
            return list(artifact.test_files_changed)
    return []


class ProductionReviewer:
    """Bind workspace-local validation only when the reviewer executes inside a live graph."""

    def __init__(
        self,
        *,
        settings: Settings,
        llm_client: LLMClient,
        cancellation_token: CancellationToken | None = None,
        operation_executor: ExternalOperationExecutor | None = None,
        repository_id: str | None = None,
        process_runner: ProcessRunner | None = None,
        model_role: ModelRole | None = None,
        # The design questions a person has decided for this repository. The reviewer is
        # told which demands were overruled; the platform's verdict handling enforces the
        # demotion whatever the model does with the telling.
        settled_questions: Sequence[SettledQuestion] = (),
        # The attempt's pinned validation plan (83-). None keeps the tool's own derivation,
        # which is what every non-live caller and test double relies on.
        plan_builder: ValidationPlanBuilder | None = None,
        # Bytes-by-id, so the reviewer can be shown the frames it judges against rather than
        # only their coordinates. Optional: a composition without it judges from text, which
        # is what every non-live caller and test double does.
        attachments: AttachmentContentSource | None = None,
    ) -> None:
        """Retain immutable model policy while deriving the workspace from current graph state."""
        self._settings = settings
        self._llm_client = llm_client
        self._cancellation_token = cancellation_token
        self._operation_executor = operation_executor
        self._repository_id = repository_id
        self._process_runner = process_runner
        self._model_role = model_role
        self._settled_questions = tuple(settled_questions)
        self._plan_builder = plan_builder
        self._attachments = attachments
        self._prompt_loader = PromptLoader()

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Review the active workspace with the configured timeout and strict artifact boundary."""
        return await ReviewerAgent(
            prompt_loader=self._prompt_loader,
            llm_client=self._llm_client,
            attachments=self._attachments,
            validation_tool=WorkspaceValidationTools(
                state["workspace_descriptor"].root_path,
                repository_id=self._repository_id,
                plan_builder=self._plan_builder,
                cancellation_token=self._cancellation_token,
                operation_executor=self._operation_executor,
                process_runner=self._process_runner,
                # Read from the completion this review is about, so the repository's own test
                # runner can be asked about the suites the attempt just wrote before it is
                # asked about all of them.
                changed_test_paths=_changed_test_paths(state),
            ),
            validation_timeout_seconds=self._settings.validation_command_timeout_seconds,
            cancellation_token=self._cancellation_token,
            process_runner=self._process_runner,
            operation_executor=self._operation_executor,
            model_role=self._model_role,
            bounded_review_scope=self._settings.bounded_review_scope,
            settled_questions=self._settled_questions,
        ).run(state)


class _PreflightedEngineer:
    """Run deterministic checkout setup once before the live coding boundary."""

    def __init__(
        self,
        engineer: WorkflowAgent,
        *,
        settings: Settings,
        cancellation_token: CancellationToken,
        operation_executor: ExternalOperationExecutor | None,
        process_runner: ProcessRunner | None = None,
    ) -> None:
        self._engineer = engineer
        self._settings = settings
        self._cancellation_token = cancellation_token
        self._operation_executor = operation_executor
        self._process_runner = process_runner
        self._prepared_revision: str | None = None

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Bootstrap evidenced dependencies before invoking the coding model."""
        if self._prepared_revision is not None:
            return await self._engineer.run(state)
        descriptor = state["workspace_descriptor"]
        preflight = await RepositoryPreflight(
            default_timeout_seconds=float(self._settings.default_agent_timeout_seconds),
            dependency_install_timeout_seconds=float(
                self._settings.dependency_install_timeout_seconds
            ),
            process_runner=self._process_runner,
            cancellation_token=self._cancellation_token,
            operation_executor=self._operation_executor,
        ).run(descriptor.root_path, repository_id=descriptor.workspace_id)
        if preflight.validation_readiness == "blocked":
            raise _recordable_configuration_error(_repository_preflight_failure(preflight))
        # A review retry stays in the same compiled graph and checkout revision. Frozen
        # setup must not be re-run or charged to the Engineer retry budget. If a prior
        # attempt changed a manifest, the injected dependency synchronizer owns that change.
        self._prepared_revision = preflight.revision
        return await self._engineer.run(state)


def _repository_preflight_failure(preflight: RepositoryPreflightResult) -> str:
    """Return bounded platform-owned setup evidence without subprocess output."""
    issue_ids = sorted({issue.issue_id for issue in preflight.blocking_issues})
    issues = ",".join(issue_ids) if issue_ids else "UNKNOWN_REPOSITORY_SETUP_FAILURE"
    return (
        "repository setup blocked live coding before the Engineer Agent ran "
        f"(install_status={preflight.dependency_install_status}; issues={issues})"
    )


class WorkspaceProvisioner:
    """Clone a requested GitHub repository into the only workspace root exposed to the worker."""

    def __init__(self, workspace_root: Path) -> None:
        """Resolve the immutable parent directory allowed to hold workflow clones."""
        self._workspace_root = workspace_root.resolve(strict=False)

    def provision(self, state: AgentState, *, git_service: GitService) -> AgentState:
        """Clone the source and create its working branch before any coding agent can run."""
        descriptor = state["workspace_descriptor"]
        destination = Path(descriptor.root_path).resolve(strict=False)
        _require_workspace_within_root(destination, self._workspace_root)
        if destination.exists():
            msg = f"workflow workspace already exists: {destination}"
            raise RuntimeConfigurationError(msg)
        _require_github_https_source(descriptor.source_repo_url)
        self._workspace_root.mkdir(parents=True, exist_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        git_service.clone(_github_clone_url(descriptor.source_repo_url), destination)
        git_service.create_branch(
            destination,
            descriptor.working_branch,
            base_branch=descriptor.base_branch or descriptor.default_branch,
        )
        return state

    async def provision_async(self, state: AgentState, *, git_service: Any) -> AgentState:
        """Provision with an interruptible async Git adapter while preserving the sync test API."""
        descriptor = state["workspace_descriptor"]
        destination = Path(descriptor.root_path).resolve(strict=False)
        _require_workspace_within_root(destination, self._workspace_root)
        # A valid repository may already be present when the clone operation was
        # durably completed just before a worker crash.  The journalled adapter
        # verifies and reuses only that exact completed clone; other pre-existing
        # directories are rejected inside the clone action.
        if destination.exists() and not (destination / ".git").exists():
            msg = f"workflow workspace already exists: {destination}"
            raise RuntimeConfigurationError(msg)
        _require_github_https_source(descriptor.source_repo_url)
        self._workspace_root.mkdir(parents=True, exist_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        clone = git_service.clone(_github_clone_url(descriptor.source_repo_url), destination)
        if inspect.isawaitable(clone):
            await clone
        branch = git_service.create_branch(
            destination,
            descriptor.working_branch,
            # A revision's checkout builds from the superseded branch; the interruptible
            # clone above already checked that branch out, so this base always resolves.
            base_branch=descriptor.base_branch or descriptor.default_branch,
        )
        if inspect.isawaitable(branch):
            await branch
        return state


def _require_workspace_within_root(destination: Path, workspace_root: Path) -> None:
    """Reject paths outside the mounted worker volume before performing any Git operation."""
    if destination == workspace_root or workspace_root not in destination.parents:
        msg = f"workspace root must be a child of configured workspace_root: {workspace_root}"
        raise RuntimeConfigurationError(msg)


def _require_github_https_source(source_url: str) -> None:
    """Limit live cloning to safe, token-free GitHub HTTPS source URLs."""
    parsed = urlsplit(source_url)
    if parsed.scheme != "https" or parsed.hostname not in {"github.com", "www.github.com"}:
        msg = "live workflows require an HTTPS github.com source_repo_url"
        raise RuntimeConfigurationError(msg)
    if parsed.username or parsed.password or not parsed.path.endswith(".git"):
        msg = "source_repo_url must be a token-free GitHub repository URL ending in .git"
        raise RuntimeConfigurationError(msg)


def _github_clone_url(source_url: str) -> str:
    """Add GitHub's non-secret HTTPS username so askpass is prompted only for the password."""
    parsed = urlsplit(source_url)
    return urlunsplit(
        (parsed.scheme, f"x-access-token@{parsed.netloc}", parsed.path, parsed.query, "")
    )


@contextmanager
def _git_askpass_environment(token: str) -> Iterator[Mapping[str, str]]:
    """Create a mode-0700 askpass helper so Git can authenticate without putting tokens in URLs."""
    if not token.strip():
        msg = "GitHub token must not be empty"
        raise RuntimeConfigurationError(msg)
    descriptor, script_name = tempfile.mkstemp(prefix="ai-platform-git-askpass-", text=True)
    script_path = Path(script_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as script:
            script.write("#!/bin/sh\nprintf '%s\\n' \"$PLATFORM_GIT_TOKEN\"\n")
        script_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        yield {
            "GIT_ASKPASS": str(script_path),
            "GIT_TERMINAL_PROMPT": "0",
            "PLATFORM_GIT_TOKEN": token,
        }
    finally:
        script_path.unlink(missing_ok=True)


__all__ = [
    "ProductionReviewer",
    "RuntimeConfigurationError",
    "WorkflowAgent",
    "WorkspaceProvisioner",
]
