"""Live composition for request-scoped, multi-repository feature orchestration."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import re
import shutil
import stat
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import yaml
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.sql.elements import ColumnElement

from adapters.git_adapter import CredentialProvenance, EmptyCommitError, GitSafetyError
from adapters.github_adapter import PyGithubService
from adapters.interruptible_git import InterruptibleGitService, reviewed_content_fingerprint
from adapters.llm_adapter import (
    LLMAdapterError,
    LLMClient,
    ResponsesCodingExecutor,
    completed_coding_operation_matches_workspace,
    llm_client_for,
)
from agents.engineer.agent import EngineerAgent, repository_snapshot_budget
from agents.engineer.publisher import require_reviewed_workspace_match
from agents.integration_reviewer.agent import IntegrationReviewerAgent
from agents.planner.feature_planner import FeaturePlannerAgent
from agents.product_manager.agent import AttachmentContentSource, ProductManagerAgent
from agents.recon.agent import RepositoryReconAgent
from agents.shared.contracts import (
    ARTIFACT_FILENAMES,
    AgentArtifactError,
    attempt_artifact_id,
    attempt_number_from_qualified_id,
    create_artifact,
    safe_error_diagnostics,
)
from api.control_plane import RequestScopedCredentials
from api.schemas import ClarificationAnswer
from artifacts.schemas import (
    ChildWorkflowResultArtifact,
    CodeCompletionArtifact,
    ContractChangeRequestArtifact,
    DesignDetailArtifact,
    DesignSnapshotArtifact,
    IntegrationContractArtifact,
    OverruledReviewFinding,
    RepositoryReconnaissanceArtifact,
    RepositoryWorkstreamPlan,
    ReviewArtifact,
    ReviewFinding,
    ReviewFindingCounts,
    TaskPlanArtifact,
    TechnicalPRDArtifact,
)
from configs.model_roles import (
    AgentPlatform,
    ModelConfigurationError,
    ModelRole,
    PerformanceTier,
    normalize_context_window_tokens,
)
from configs.settings import Settings
from services.cancellation import (
    CancellationRequested,
    CancellationToken,
    CompositeCancellationToken,
    DatabaseCancellationToken,
    RedisCancellationToken,
)
from services.design_conflict import settled_questions
from services.design_resolution import DesignReferenceResolver
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.journaled_github import JournaledGitHubService
from services.process_runner import (
    AsyncioProcessRunner,
    ProcessRunner,
    redact_output,
    redact_source_credentials,
    repository_subprocess_environment,
    subprocess_result_code,
)
from services.repository_repair import approved_repair_for
from services.runtime import (
    ProductionReviewer,
    RuntimeConfigurationError,
    WorkspaceProvisioner,
    _git_askpass_environment,
)
from state.enums import ContractChangeRequestStatus, FeatureWorkflowStatus
from state.external_operations import (
    CONTRACT_PROJECTION_LOGICAL_STEP,
    ExternalOperation,
    ExternalOperationStatus,
    ExternalOperationType,
    child_attempt_of,
)
from state.failure_diagnosis import (
    DiagnosedFailure,
    FeatureFailureClassification,
    workspace_capacity_diagnostic,
)
from state.feature_models import ChildWorkflowReference, FeatureWorkflowSnapshot, RepositorySpec
from state.models import AgentState, WorkspaceDescriptor, create_initial_agent_state
from storage.attachment_store import DatabaseAttachmentStore
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal, OperationResult
from storage.models import FeatureWorkflowModel, RepositorySpecModel
from tools.assigned_file_conformance import RepositoryAssignedFileChecker
from tools.contract_tools import OpenAPIContractCodeGenerator
from tools.cross_repository_diffs import RepositoryChange, bounded_repository_change
from tools.dependency_sync import RepositoryToolSynchronizer
from tools.design_value_conformance import DesignValueChecker
from tools.file_tools import WorkspaceFileTools
from tools.implementation_completeness import (
    CategoryEvidenceSource,
    validate_implementation_completeness,
)
from tools.lineage_base import (
    CONTRACT_PROJECTION_PATH,
    LineageBaseRevision,
    branch_change_evidence,
)
from tools.lint_capabilities import RepositoryLintCapabilities
from tools.llm_call_record import last_llm_call_payload, reset_llm_call_record
from tools.model_routing import ModelRouter, ModelRoutingDecision
from tools.node_toolchain import package_manager_argv
from tools.reachability import RepositoryReachabilityChecker, unreachable_addition_issues
from tools.repository_layout import reconcile_workstream_layout
from tools.repository_preflight import (
    PreflightIssue,
    RepositoryHealthDisposition,
    RepositoryPreflight,
    RepositoryPreflightResult,
    classify_lint_failure,
)
from tools.repository_reconnaissance import inspect_repository_for_planning
from tools.requirement_reconciliation import AnsweredClarification, RequirementReconciliation
from tools.resolved_issue_ledger import SettledQuestion
from tools.retry_strategy import (
    FailureClassification,
    a_required_command_rejected_the_source,
    build_retry_plan,
    classify_failure,
)
from tools.review_fix_classification import finding_fingerprint, fingerprint_for_text
from tools.scoped_tests import RepositoryScopedTestRunner
from tools.self_review import CodingRoleSelfReviewer
from tools.source_formatting import RepositoryToolFormatter, SourceValidationError
from tools.technology_detection import (
    calculate_repository_revision,
    inspect_repository_technology,
)
from tools.typecheck import RepositoryTypecheckRunner
from tools.validation_tools import (
    DefaultValidationPlanBuilder,
    PinnedValidationPlanBuilder,
    RepositoryValidationPlan,
    ValidationPlanBuilder,
    ValidationResult,
    ValidationStatus,
    WorkspaceValidationTools,
    validation_failure_identity,
)
from workflows.feature_workflow import (
    BlindPlanningRecord,
    ChildExecution,
    ChildWorkstreamExecutor,
    FeatureEventWriter,
    FeatureWorkflowOrchestrator,
    GitHubPullRequestPublisher,
    ReconnaissanceReport,
    _child_result,
    _current_revision,
    _feature_branch_segment,
    feature_has_publishable_work,
    source_rejection_disposition,
)

# Generated from the approved contract and compared against afterwards, so the platform owns
# this path in every child workspace.
# Named in `tools.lineage_base`, where the branch-evidence readers that must exclude it
# already import from. Aliased rather than re-spelled so the two cannot drift apart.
# What counts as an image when deciding where this repository keeps them.
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif", ".avif"})

_CONTRACT_PROJECTION = CONTRACT_PROJECTION_PATH
# Untracked paths the reset must not remove, keyed by the runtime that produces them. Each
# is produced by a step whose result is journaled and therefore replayed rather than
# re-executed on a retry, so anything deleted here is gone for the rest of the workstream:
# the dependency tree, and the hook directory an installer generates. Everything else
# untracked belongs to the rejected attempt, or is disposable build output that inflates a
# project-wide lint until it times out.
#
# Keyed on what the checkout is detected to be, and never written down as a flat list of
# directory names. A flat list was the defect: only Node's tree was ever named, so a Python
# checkout lost `.venv` on its first retry, a Go checkout its `vendor/` and a Rust checkout
# its `target/` -- silently, permanently, and with every validation command afterwards
# running against a repository with no dependencies. The reasoning that put `node_modules`
# here applies identically to every other ecosystem; only Node had been written down.
#
# `.husky` is under the same rule rather than beside it. It is not a dependency directory --
# it is a hook bootstrap an installer generates -- but generically it is the same thing, the
# state a journaled setup step produced and will not produce again, and it belongs to the
# Node toolchain specifically. A Python checkout has no reason to keep it.
#
# Java is absent deliberately. Maven and Gradle resolve into the user's home, so a checkout
# holds none of their dependency tree, and the `target/` and `build/` directories they leave
# behind are the build output this clean exists to remove.
_INSTALLED_TREES_BY_RUNTIME: Mapping[str, tuple[str, ...]] = {
    "JavaScript": ("node_modules", ".husky"),
    "TypeScript": ("node_modules", ".husky"),
    "Python": (".venv",),
    "Go": ("vendor",),
    "Rust": ("target",),
}
# Enough to carry a feature-sized change without letting one runaway diff crowd out the
# task plan and repository snapshot the same prompt has to hold. Applied only at whole-file
# boundaries: a capture is either shown as complete files or the withheld files are named,
# never cut mid-hunk. The blind character slice this used to drive made every over-budget
# retry a regeneration from a fragment -- AB-Feature-170's capture was 50,477 characters, so
# five of its nine files (every remaining test file among them) were silently absent from
# the prompt that demanded their byte-identical reproduction.
_MAX_PREVIOUS_ATTEMPT_CHARACTERS = 24_000
# The reset triggers a retry may legitimately run under (§3.3). Recorded with the attempt,
# so a reset is always attributable to its cause.
_RESET_WHOLESALE_REWRITE = "wholesale_rewrite_rejected"
_RESET_WORKSPACE_UNUSABLE = "workspace_unusable"
# Where the previous attempt's diff names each file it touches.
_DIFF_FILE_HEADER = re.compile(r"^diff --git a/(?P<a>\S+) b/(?P<b>\S+)$", re.MULTILINE)
# A file has to lose real substance before a rewrite is worth reporting, and it has to lose
# far more than it gained: a genuine refactor trades lines roughly evenly, while a file
# answered by replacement keeps almost nothing. Both thresholds must be met.
_REWRITE_MINIMUM_DELETED_LINES = 40
_REWRITE_DELETED_TO_ADDED_RATIO = 3
_WORKSPACE_MAINTENANCE_LOCK = ".workspace-maintenance.lock"
_IN_PROCESS_WORKSPACE_MAINTENANCE_LOCK = threading.Lock()
_TERMINAL_WORKSPACE_STATUSES = (
    FeatureWorkflowStatus.COMPLETED,
    FeatureWorkflowStatus.CANCELLED,
)
_LOGGER = logging.getLogger(__name__)

# What a failing workspace Git command actually meant, keyed by a marker in its own stderr.
#
# Recognised rather than quoted. The diagnosed-failure contract forbids putting checkout
# text into a durable diagnostic -- Git's stderr names repository paths -- so the marker
# selects a sentence this platform composed, and the checkout's own words are never
# persisted. Everything here was reproduced against real Git before it was written down.
#
# The reason this table exists at all: AB-Feature-132 died on `git add` exiting 128, and the
# record kept only `WORKSPACE_COMMAND_FAILED_EXIT_128`. Recovering what that meant took a
# reproduction hunt across eight fixture repositories, months later, because nothing had
# written down which of the several fatal shapes it was.
_WORKSPACE_COMMAND_CAUSES: tuple[tuple[str, str], ...] = (
    (
        "index.lock",
        "Git refused because an index lock was already held in this workspace. That lock is "
        "left behind when a Git process is killed, which this platform does when it "
        "terminates a cancelled or abandoned attempt's process group.",
    ),
    (
        "not a git repository",
        "The workspace is no longer a Git repository. Its checkout was removed or replaced "
        "underneath the attempt that was using it.",
    ),
    (
        "does not have a commit checked out",
        "The workspace contains a nested Git repository with no commit on it, which Git "
        "cannot index. Something inside this checkout initialised a repository of its own.",
    ),
    (
        "ignored by one of your .gitignore",
        "Git refused to act on a path this repository gitignores. The command named an "
        "ignored path explicitly, which Git treats as a mistake rather than a filter.",
    ),
    (
        "did not match any files",
        "Git matched none of the paths the command named in this checkout.",
    ),
)


def _workspace_command_cause(stderr: str) -> str | None:
    """Name what a failing workspace command meant, in this platform's own words."""
    lowered = stderr.lower()
    return next((cause for marker, cause in _WORKSPACE_COMMAND_CAUSES if marker in lowered), None)


class RepositoryWorkspaceError(DiagnosedFailure, RuntimeConfigurationError):
    """A deterministic workspace maintenance command failed before safe coding can continue.

    One of the three exceptions that already followed the diagnosed-failure pattern
    informally. It now declares it, so "does this exception explain itself" is answered by
    the type rather than by two ``getattr`` calls at the boundary that reads it.
    """

    def __init__(self, *, stage: str, result: Any) -> None:
        command = " ".join(result.command)
        code = subprocess_result_code(
            return_code=result.return_code,
            timed_out=result.timed_out,
            cancelled=result.cancelled,
            prefix="workspace_command",
        )
        diagnostic = f"Workspace {stage} command `{command}` failed ({code})."
        cause = _workspace_command_cause(getattr(result, "stderr", "") or "")
        super().__init__(
            diagnostic,
            classification=FeatureFailureClassification.VALIDATION_CONFIGURATION_FAILURE,
            diagnostics=(diagnostic, *([cause] if cause else ())),
        )


class WorkspaceMaintenanceError(DiagnosedFailure, RuntimeConfigurationError):
    """Workspace cleanup failed without deleting an unverified path or hiding the error.

    The one of the three that carried diagnostics and no classification, so a cleanup failure
    reached a terminal record as whatever name happened to be available.
    """

    def __init__(self, diagnostic: str) -> None:
        super().__init__(
            diagnostic,
            classification=FeatureFailureClassification.WORKSPACE_MAINTENANCE_FAILURE,
            diagnostics=(diagnostic,),
        )


class WorkspaceCapacityError(DiagnosedFailure, RuntimeConfigurationError):
    """The worker volume cannot safely accept another clone or dependency installation."""

    def __init__(self, *, workspace_root: Path, free_bytes: int, required_bytes: int) -> None:
        # Shared with the acceptance-time refusal, which reaches the same volume from the
        # storage layer and must not describe the condition differently.
        diagnostic = workspace_capacity_diagnostic(
            workspace_root=str(workspace_root),
            free_bytes=free_bytes,
            required_bytes=required_bytes,
        )
        self.free_bytes = free_bytes
        self.required_bytes = required_bytes
        super().__init__(
            diagnostic,
            # A limit rather than a defect, and neither is a finding about the repository.
            # Three of the fifty failures measured for this task recorded the bare type name
            # `WorkspaceCapacityError` as their root classification instead.
            classification=FeatureFailureClassification.PLATFORM_CAPACITY_FAILURE,
            diagnostics=(diagnostic,),
        )


class AmbiguousPublicationReceiptError(DiagnosedFailure, RuntimeConfigurationError):
    """A durable publication receipt could not be validated after a possible commit.

    The deliberate fail-closed refusal in `_recover_approved_publication`: coding is never
    re-run while a commit may exist without a recorded result. Deliberate is the point.
    Raised as a bare `RuntimeConfigurationError`, this reached the generic triage as "the
    platform did not anticipate RuntimeConfigurationError", was classified a platform
    defect, and terminated the feature -- a refusal that exists to protect work destroying
    the run that contained it (AB-Feature-184). The refusal now states its classification
    and names the operation and branch a person should inspect.
    """

    def __init__(self, *, operation_id: str, branch_name: str) -> None:
        message = (
            "durable publication receipt is invalid; refusing to rerun coding after a "
            "possible commit"
        )
        super().__init__(
            message,
            classification=FeatureFailureClassification.UNCONFIRMED_EXTERNAL_EFFECT,
            diagnostics=(
                "Publication recovery refused to rerun coding: the durable publication "
                "receipt could not be validated, so an approved commit may exist without "
                "a recorded result.",
                f"Inspect the CREATE_COMMIT operation {operation_id} and the branch "
                f"{branch_name}; the commit may exist without a recorded result.",
            ),
        )


@dataclass(frozen=True, slots=True)
class _CategoryEvidence:
    """What the completeness gate judges category promises against, and where it came from.

    ``paths`` is ``None`` only on the last resort: the gate then reads the attempt's own
    declared buckets, which is the behaviour that made a targeted remediation unsatisfiable,
    so the source is carried onto the result rather than left to be inferred.
    """

    paths: tuple[str, ...] | None
    modified_paths: tuple[str, ...]
    source: CategoryEvidenceSource


@dataclass(frozen=True)
class WorkspaceMaintenanceResult:
    """Auditable outcome of one serialized terminal-workspace cleanup and capacity check."""

    removed_feature_ids: tuple[str, ...]
    skipped_symlink_feature_ids: tuple[str, ...]
    skipped_non_directory_feature_ids: tuple[str, ...]
    free_bytes: int


class LiveFeatureProductManager:
    """Adapt the existing Product Manager Agent to one parent PRD with no repository workspace."""

    def __init__(self, agent: ProductManagerAgent, workspace_root: Path) -> None:
        """Bind the agent and only a deterministic planning-only descriptor."""
        self._agent = agent
        self._workspace_root = workspace_root

    async def create_technical_prd(
        self,
        *,
        feature_id: str,
        prd: Any,
        design_snapshot: DesignSnapshotArtifact | None = None,
    ) -> TechnicalPRDArtifact:
        """Run PM exactly once at parent scope and return its normal technical-PRD artifact.

        The design snapshot goes into the state because that is where the agent looks for it:
        `design_snapshot_in_state` reads `state["artifacts"]`, and this list was `[prd]` alone
        until 2026-09-12. Every other role was wired -- the planner is handed the snapshot by
        its caller, and both child judges get it appended to their state -- so the product
        manager, whose whole task is deriving requirements from the frames, was the only role
        reading a citation it could not see. AB-Feature-228 proved it live: a 1,831-token
        prompt against 77,753 characters of resolved index, and a technical PRD that named the
        node id from the citation and not one element from inside it.

        No re-homing, unlike `_run_one_child`'s copies: this state's `workflow_id` *is* the
        feature id, which is what the snapshot already carries, so the agent's own
        same-workflow filter matches it as-is.
        """
        state = create_initial_agent_state(
            workflow_id=feature_id,
            workspace_descriptor=WorkspaceDescriptor(
                workspace_id=f"{feature_id}-planning",
                root_path=str(self._workspace_root / "planning" / _safe_segment(feature_id)),
                source_repo_url="https://github.com/feature-planning/placeholder.git",
                default_branch="main",
                working_branch=f"ai/{_safe_segment(feature_id)}/planning",
            ),
        )
        # Appended only when one was resolved, so a citation-free feature's prompt stays
        # byte-identical to what it was before designs existed at all.
        state["artifacts"] = [prd] if design_snapshot is None else [prd, design_snapshot]
        update = await self._agent.run(state)
        artifact = update["artifacts"][0]
        if not isinstance(artifact, TechnicalPRDArtifact):
            msg = "product manager did not return a technical PRD"
            raise RuntimeConfigurationError(msg)
        return artifact

    async def reconcile_requirements(
        self,
        *,
        feature_id: str,
        technical_prd: TechnicalPRDArtifact,
        answers: Sequence[AnsweredClarification],
    ) -> RequirementReconciliation:
        """Run the agent's reconciliation at parent scope, needing no workspace of its own."""
        return await self._agent.reconcile_requirements(
            workflow_id=feature_id, technical_prd=technical_prd, answers=answers
        )


class LiveRepositoryReconnaissance:
    """Clone every target repository and read it before the feature is planned against it."""

    def __init__(
        self,
        *,
        settings: Settings,
        git_environment: Mapping[str, str],
        recon_client: LLMClient,
        journal: ExternalOperationJournal,
        cancellation_token: CancellationToken,
        # Which stored credential `git_environment` carries, so a clone the remote refuses
        # names it. Optional: a composition that supplies no credential reports the generic
        # Git failure it always did.
        credential: CredentialProvenance | None = None,
    ) -> None:
        """Bind the read-only checkout boundary and the model that interprets it."""
        self._settings = settings
        self._git_environment = dict(git_environment)
        self._credential = credential
        self._agent = RepositoryReconAgent(prompt_loader=_prompt_loader(), llm_client=recon_client)
        self._journal = journal
        self._cancellation_token = cancellation_token

    async def inspect(
        self,
        *,
        feature: FeatureWorkflowSnapshot,
        repositories: Sequence[RepositorySpec],
        technical_prd: TechnicalPRDArtifact,
        credentials: RequestScopedCredentials,
    ) -> ReconnaissanceReport:
        """Return one artifact per readable repository, and name each one left unread."""
        del credentials
        artifacts: list[RepositoryReconnaissanceArtifact] = []
        blind: list[BlindPlanningRecord] = []
        for repository in repositories:
            await self._cancellation_token.raise_if_cancelled()
            try:
                artifacts.append(
                    await self._inspect_one(
                        feature=feature, repository=repository, technical_prd=technical_prd
                    )
                )
            except CancellationRequested:
                raise
            except Exception as error:  # noqa: BLE001 - one repository's fault is its own
                # A repository that cannot be read leaves the planner where it already was
                # for that repository: planning without evidence. Failing the feature here
                # would make reconnaissance a new way for a run to die before it starts.
                # In the message, not in `extra`. This module logs through stdlib logging
                # and the deployment renders those records with a plain formatter, which
                # drops `extra` entirely -- so this was one anonymous line saying that some
                # repository had been planned blind, naming neither which nor why. That is
                # precisely what somebody reading the log afterwards needs, and finding it
                # cost a database query and a guess.
                reason = "; ".join(safe_error_diagnostics(error)) or "no diagnostics"
                _LOGGER.warning(
                    "repository reconnaissance failed; planning %s blind: %s (%s) "
                    "[feature=%s stage=repository_reconnaissance agent=repository_recon]",
                    repository.repository_id,
                    type(error).__name__,
                    reason,
                    feature.feature_id,
                )
                blind.append(
                    BlindPlanningRecord(
                        repository_id=repository.repository_id,
                        error_type=type(error).__name__,
                        reason=reason,
                    )
                )
        return ReconnaissanceReport(artifacts=artifacts, blind=blind)

    async def suggest_answers(
        self,
        *,
        feature_id: str,
        questions: Sequence[Any],
        reconnaissance: Sequence[RepositoryReconnaissanceArtifact],
    ) -> list[Any]:
        """Answer the open questions the checkouts already settle, from what was recorded."""
        return await self._agent.suggest_answers(
            feature_id=feature_id, questions=questions, reconnaissance=reconnaissance
        )

    async def _inspect_one(
        self,
        *,
        feature: FeatureWorkflowSnapshot,
        repository: RepositorySpec,
        technical_prd: TechnicalPRDArtifact,
    ) -> RepositoryReconnaissanceArtifact:
        """Read one checkout in a disposable clone, then remove it."""
        # Deliberately not the child's workspace. The child's clone is journal-reused under
        # its own operation scope and carries the recovery path's assumptions about what
        # exists when; writing into it from the parent would couple reconnaissance to that
        # machinery for the sake of one avoided fetch. This clone is read-only and is
        # removed below, so it costs disk for the length of one model call.
        # Resolved before the guard below, not after: every other caller of
        # `_require_workspace_child` in this module resolves its candidate path first, and
        # this one silently didn't. `_require_workspace_child` resolves the *root* but not
        # the candidate, so an unresolved candidate under a symlinked `workspace_root` (a
        # `/tmp` that is actually `/private/tmp`, say) never appears in its own resolved
        # root's parents and the guard refuses every call, for every repository, regardless
        # of what that repository contains -- which is exactly what made every repository
        # reconnaissance in this deployment fail identically.
        workspace = (
            self._settings.workspace_root
            / _feature_branch_segment(feature.feature_id)
            / ".reconnaissance"
            / _safe_segment(repository.repository_id)
        ).resolve(strict=False)
        _require_workspace_child(workspace, self._settings.workspace_root)
        operation_executor = ExternalOperationExecutor(
            journal=self._journal,
            cancellation_token=self._cancellation_token,
            scope=ExternalOperationScope(
                workflow_id=feature.workflow_id,
                feature_id=feature.feature_id,
                repository_id=repository.repository_id,
            ),
        )
        git_service = InterruptibleGitService(
            default_branch=repository.default_branch,
            environment=self._git_environment,
            cancellation_token=self._cancellation_token,
            operation_executor=operation_executor,
            workspace_root=self._settings.workspace_root,
            credential=self._credential,
        )
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.parent.mkdir(parents=True, exist_ok=True)
        try:
            await git_service.clone(_git_source_url(str(repository.repository_url)), workspace)

            # One journal row per reconnaissance model call, in the shape the child loop's
            # calls have carried since the operation journal existed. Observability only:
            # reconnaissance resumes from its artifacts, so the idempotency input carries a
            # per-call nonce and no result is ever replayed from a row. AB-Feature-173's
            # 37-minute ReadTimeout was invisible while it passed because this call left no
            # record until it returned.
            async def inspect_once() -> tuple[RepositoryReconnaissanceArtifact, OperationResult]:
                # Reset-then-read, as at `_journaled_planning_call`: the row names the model
                # that answered this repository's inspection, read task-locally so concurrent
                # reconnaissance calls can never read each other's.
                reset_llm_call_record()
                artifact = await self._agent.inspect(
                    feature_id=feature.feature_id,
                    repository_id=repository.repository_id,
                    workspace_root=str(workspace),
                    technical_prd=technical_prd,
                )
                return artifact, OperationResult(payload=last_llm_call_payload())

            journaled = await operation_executor.run(
                operation_type=ExternalOperationType.RUN_REPOSITORY_RECON,
                logical_step="repository_reconnaissance",
                safe_input={
                    "stage": "repository_reconnaissance",
                    "repository_id": repository.repository_id,
                },
                idempotency_input={
                    "stage": "repository_reconnaissance",
                    "call_nonce": uuid4().hex,
                },
                action=inspect_once,
                max_attempts=1,
                attempt_metadata={"stage": "repository_reconnaissance"},
            )
            artifact = cast(RepositoryReconnaissanceArtifact, journaled.value)
            # Derived from this clone, while it still exists, and attached to the one artifact
            # the planner is given. The planner otherwise learns what a checkout contains from
            # prose -- "Playwright configuration present at the repository root" is a true
            # sentence about AB-Feature-225's checkout -- and writes a test requirement to run
            # it, which the reviewer then enforces against a plan that has no such command and
            # an engineer with no way to add one. Naming the runnable set here is what lets the
            # requirement be written against it in the first place.
            #
            # Best-effort on purpose: a checkout this builder cannot read leaves the list empty,
            # which the planner's clause treats as "not established" rather than as "nothing is
            # runnable". Reconnaissance must not fail over an advisory fact.
            try:
                plan = DefaultValidationPlanBuilder(
                    default_timeout_seconds=float(self._settings.validation_command_timeout_seconds)
                ).build_plan_sync(workspace, repository)
                artifact = artifact.model_copy(
                    update={
                        "runnable_validation_commands": [
                            " ".join(item.command) for item in plan.commands
                        ]
                    }
                )
            except (OSError, ValueError) as error:
                _LOGGER.warning(
                    "runnable validation commands could not be derived for %s (%s) "
                    "[agent=repository_recon]",
                    repository.repository_id,
                    type(error).__name__,
                )
            return artifact
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


class LiveCrossRepositoryDiffs:
    """Read each child's reviewed production source out of the workspace it was built in."""

    def __init__(self, *, settings: Settings) -> None:
        """Bind only the workspace policy; every other input comes from the results."""
        self._settings = settings

    async def collect(
        self, *, feature_id: str, child_results: Sequence[ChildWorkflowResultArtifact]
    ) -> list[RepositoryChange]:
        """Return the production files of every repository whose workspace can still be read."""
        del feature_id
        changes: list[RepositoryChange] = []
        for result in child_results:
            if not result.production_files_changed:
                continue
            workspace = Path(result.workspace_path).resolve(strict=False)
            try:
                _require_workspace_child(workspace, self._settings.workspace_root)
            except RuntimeConfigurationError:
                continue
            tools = WorkspaceFileTools(
                workspace, max_file_bytes=self._settings.max_workspace_file_bytes
            )
            files: list[tuple[str, str]] = []
            for path in result.production_files_changed:
                try:
                    # Redacted like every other model input: a changed production file is
                    # exactly where a newly added credential would be.
                    files.append((path, redact_source_credentials(tools.read_file(path))))
                except (OSError, UnicodeDecodeError, ValueError):
                    continue
            if not files:
                continue
            changes.append(
                bounded_repository_change(
                    repository_id=result.repository_id,
                    role=result.workstream_id,
                    contract_sections_implemented=result.contract_sections_implemented,
                    contract_sections_consumed=result.contract_sections_consumed,
                    files=files,
                )
            )
        return changes


class RepositoryRepairFailed(RuntimeError):
    """Raised when an approved repair's own command did not succeed.

    Distinguished from the attempt that follows it failing: a repair whose command never ran
    has not been tried, and reporting that as "the repair did not work" would send somebody
    to look at their repository for a problem that is the platform's.
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        """Record only the platform-owned classification of how the command ended."""
        super().__init__(message)
        self.code = code


class LiveChildWorkstreamExecutor(ChildWorkstreamExecutor):
    """Reuse the existing Engineer and Reviewer once per repository-specific child state."""

    def __init__(
        self,
        *,
        settings: Settings,
        git_environment: Mapping[str, str],
        # Bytes-by-id only, for placing the design's exported images into the checkout before
        # the Engineer runs. Optional: a composition without it places nothing and every
        # prompt is what it was before design assets existed.
        attachments: AttachmentContentSource | None = None,
        # Injected behind the protocol, not the concrete OpenAI client: the live child
        # path is only testable if a deterministic double can stand in for the provider.
        engineer_client: LLMClient,
        reviewer_client: LLMClient,
        journal: ExternalOperationJournal,
        cancellation_token: CancellationToken,
        # Preflight and validation both shell out. Injecting the runner lets tests drive
        # this path deterministically instead of depending on a package manager, a
        # network, and whatever happens to be installed on the host.
        process_runner: ProcessRunner | None = None,
        # Builds the Engineer's model boundary from the decision already persisted for this
        # attempt. One Engineer abstraction executes both initial and remediation work; the
        # factory makes recovery use the exact resolved model/effort rather than today's env.
        # Absent, every attempt uses the single injected client, as deterministic doubles do.
        engineer_client_factory: Callable[[ModelRoutingDecision | None], LLMClient] | None = None,
        # Builds the boundary the in-attempt source repair runs on: the SCOPED_FIX role's
        # model at this feature's pinned platform and tier. Returning None -- or omitting the
        # factory, as deterministic doubles do -- keeps the repair on the primary coding
        # executor, so no deployment loses the repair loop to an unresolvable role.
        scoped_fix_client_factory: Callable[[ModelRoutingDecision | None], LLMClient | None]
        | None = None,
        # Builds the boundary the one in-attempt self-review runs on: the CODING role at
        # this feature's pinned platform and tier -- never the REVIEW role, whose
        # independence belongs to the reviewer that judges this work afterwards. Returning
        # None, or omitting the factory as deterministic doubles do, runs no self-review
        # and `completed` keeps exactly its pre-Part-E meaning.
        self_review_client_factory: Callable[[ModelRoutingDecision | None], LLMClient | None]
        | None = None,
        # Writes `repository_planned_blind` again when the late reconnaissance probe also
        # fails, so the second failure is recorded the way the first was. Optional for the
        # same reason it is on the orchestrator.
        event_writer: FeatureEventWriter | None = None,
        # The boundary the Git adapter's own subprocesses run through. Deliberately separate
        # from `process_runner` above, because the two are answered differently by every
        # fixture in this repository: workspace maintenance is satisfied by an exit code,
        # while the Git adapter reads command *output* -- a double returning exit zero with
        # empty stdout can serve the first and not the second. Production supplies neither,
        # and both sides then build `AsyncioProcessRunner`.
        git_process_runner: ProcessRunner | None = None,
        # Which stored credential `git_environment` carries, by name and date. Optional for
        # the reason the reconnaissance boundary's is.
        credential: CredentialProvenance | None = None,
    ) -> None:
        """Keep provider clients request-scoped and workspace policy static."""
        self._settings = settings
        # Set by the wiring gate so the next attempt can be given a scoped repair. One per
        # unreachable module: reporting only the first cost a whole attempt per extra file,
        # because the engineer force-includes a repair's target in its snapshot and could
        # not quote back a file it was never shown. -106's backend spent both its attempts
        # that way, wiring the service it could see and not the validator it could not.
        self._wiring_repairs: list[dict[str, str]] = []
        # Added files the gate passed on textual evidence alone (83-). Recorded on the
        # attempt's result, never blocked on.
        self._textual_wirings: list[dict[str, str]] = []
        self._git_environment = dict(git_environment)
        self._attachments = attachments
        self._engineer_client = engineer_client
        self._engineer_client_factory = engineer_client_factory
        self._scoped_fix_client_factory = scoped_fix_client_factory
        self._self_review_client_factory = self_review_client_factory
        self._reviewer_client = reviewer_client
        self._journal = journal
        self._cancellation_token = cancellation_token
        self._process_runner = process_runner
        self._git_process_runner = git_process_runner
        self._event_writer = event_writer
        self._credential = credential
        self._provisioner = WorkspaceProvisioner(settings.workspace_root)
        self._contract_generator = OpenAPIContractCodeGenerator()

    async def run(
        self,
        *,
        feature: FeatureWorkflowSnapshot,
        repository: RepositorySpec,
        workstream: RepositoryWorkstreamPlan,
        child: ChildWorkflowReference,
        technical_prd: TechnicalPRDArtifact,
        contract: IntegrationContractArtifact,
        feedback: Sequence[str],
        credentials: RequestScopedCredentials,
    ) -> ChildExecution:
        """Provision one safe child clone, enforce its read-only contract, then review only it."""
        del credentials
        workspace = Path(child.workspace_path).resolve(strict=False)
        _require_workspace_child(workspace, self._settings.workspace_root)
        descriptor = WorkspaceDescriptor(
            workspace_id=repository.repository_id,
            root_path=str(workspace),
            source_repo_url=_git_source_url(str(repository.repository_url)),
            default_branch=repository.default_branch,
            working_branch=child.branch_name,
            # A revision's checkout builds on the superseded branch; `None` -- every child
            # before revisions existed -- means the repository default, as it always did.
            base_branch=child.base_branch,
        )
        state = create_initial_agent_state(
            workflow_id=child.child_workflow_id, workspace_descriptor=descriptor
        )
        state["artifacts"] = [
            technical_prd.model_copy(
                update={
                    "workflow_id": child.child_workflow_id,
                    "metadata": {
                        **technical_prd.metadata,
                        "parent_feature_id": feature.feature_id,
                    },
                }
            ),
            contract.model_copy(
                update={
                    "workflow_id": child.child_workflow_id,
                    "metadata": {
                        **contract.metadata,
                        "parent_feature_id": feature.feature_id,
                        "read_only": True,
                    },
                }
            ),
        ]
        # The resolved design, re-homed onto this child's workflow id the way the contract
        # above is: `design_snapshot_in_state` filters on it, for `_contract_in_state`'s
        # reason. Appended only when the feature has one, so a citation-free child's state --
        # and therefore both its judges' prompts -- is unchanged byte for byte.
        design_snapshot = _feature_design_snapshot(feature)
        if design_snapshot is not None:
            state["artifacts"].append(
                design_snapshot.model_copy(
                    update={
                        "workflow_id": child.child_workflow_id,
                        "metadata": {
                            **design_snapshot.metadata,
                            "parent_feature_id": feature.feature_id,
                            "read_only": True,
                        },
                    }
                )
            )
        # This repository's own frames at build fidelity, re-homed the same way and for the
        # same reason. Resolved by the parent before this loop's first attempt, so it is read
        # from the feature rather than produced here: one resolution per workstream, reused by
        # every attempt, which is what stops a design moving under a retry.
        #
        # Appended only when the parent resolved one -- a workstream the design does not apply
        # to leaves this out entirely, and its prompts are byte-identical to a feature that
        # cited nothing.
        design_detail = _feature_design_detail(feature, repository.repository_id)
        if design_detail is not None:
            state["artifacts"].append(
                design_detail.model_copy(
                    update={
                        "workflow_id": child.child_workflow_id,
                        "metadata": {
                            **design_detail.metadata,
                            "parent_feature_id": feature.feature_id,
                            "read_only": True,
                        },
                    }
                )
            )
        state["retry_count"] = child.retry_count
        # The lineage this retry builds on. `_prior_attempt_file_changes` has always existed
        # to merge earlier attempts' files into the next completion, and it reads exactly
        # this list -- but retry state never carried a prior completion, so the merge always
        # saw an empty set and the workspace reset had to compensate by destroying each
        # attempt. Loading them is what makes preserving the workspace safe: the union of
        # every attempt's files is what review and publication see, whoever touched what.
        state["artifacts"].extend(_prior_child_completions(feature, child))
        previous_review = _previous_child_review(feature, child.child_workflow_id)
        if previous_review is not None:
            state["artifacts"].append(previous_review)
        operation_executor = ExternalOperationExecutor(
            journal=self._journal,
            cancellation_token=self._cancellation_token,
            scope=ExternalOperationScope(
                workflow_id=feature.workflow_id,
                feature_id=feature.feature_id,
                child_workflow_id=child.child_workflow_id,
                repository_id=repository.repository_id,
                # One executor per call of this method, and this method runs one attempt. Every
                # row it journals -- clone, installs, the coding call and its write, the
                # validation commands, the commit and the push -- is therefore stamped with the
                # attempt it belongs to. Before the stamp existed, a reader had only
                # `clone_repository` to divide attempts by, and a retry that preserves its
                # workspace never re-clones: run 197 BE rendered attempts 0, 1 and 2 as one.
                child_attempt=child.retry_count,
            ),
        )
        git_service = InterruptibleGitService(
            default_branch=repository.default_branch,
            base_branch=child.base_branch,
            environment=self._git_environment,
            cancellation_token=self._cancellation_token,
            operation_executor=operation_executor,
            workspace_root=self._settings.workspace_root,
            credential=self._credential,
            process_runner=self._git_process_runner,
            # The commit triggers the repository's own pre-commit hook, so it needs the same
            # budget the platform gives that gate rather than Git's own brief default.
            commit_timeout_seconds=float(self._settings.commit_gate_timeout_seconds),
        )
        # Before anything reads the checkout. A retry provisions nothing, on the reasoning
        # that attempt 0 already did -- and AB-Feature-190's attempt-0 clones died on a
        # 40-second GitHub outage, so the operator-retry attempt met a workspace that was
        # never created. `_capture_attempt`'s first `git add` then raised `FileNotFoundError`
        # from the spawn itself, which nothing anticipated, so a diagnosable condition was
        # recorded as a platform defect. A retry with no usable checkout now provisions one
        # exactly as attempt 0 would.
        #
        # Ahead of the publication recovery below rather than after it, because the recovery
        # commits and pushes out of this same directory: with no checkout there it died the
        # same undiagnosed death. On a replacement checkout it still cannot succeed -- the
        # commit is bound to the reviewed content fingerprint, which a fresh clone does not
        # hold -- but it fails closed and says why.
        fresh_checkout = child.retry_count > 0 and not await self._usable_checkout(workspace)
        if fresh_checkout:
            await self._provision_replacement_checkout(
                state,
                git_service=git_service,
                feature=feature,
                repository=repository,
                child=child,
                workspace=workspace,
            )
        recovered_publication = await self._recover_approved_publication(
            git_service,
            feature=feature,
            repository=repository,
            workstream=workstream,
            child=child,
            workspace=workspace,
        )
        if recovered_publication is not None:
            return recovered_publication
        previous_attempt: str | None = None
        previous_attempt_withheld: tuple[str, ...] = ()
        workspace_preserved = False
        attempt_inputs: dict[str, Any] = {"workspace": "fresh_checkout"}
        if child.retry_count == 0:
            await self._provisioner.provision_async(state, git_service=git_service)
        # After the checkout exists and never before it. Writing them earlier creates the
        # destination directory, and the clone then refuses it with "workflow workspace
        # already exists" -- which is how AB-Feature-233 died before its first attempt,
        # having exported all six images successfully.
        await _place_design_assets(
            workspace,
            _feature_design_detail(feature, repository.repository_id),
            source=self._attachments,
            runner=self._process_runner,
            timeout=float(self._settings.default_agent_timeout_seconds),
            cancellation_token=self._cancellation_token,
        )
        completed_coding_output = await self._completed_coding_output_matches(
            feature=feature,
            repository=repository,
            child=child,
            workspace=workspace,
        )
        if fresh_checkout:
            # Nothing to capture, preserve or reset: this checkout was created moments ago
            # and holds no previous attempt at all. Recorded as the reset path's own trigger
            # so the record still says why this attempt started from a clean tree.
            attempt_inputs = {
                "workspace": "fresh_checkout",
                "reset_trigger": _RESET_WORKSPACE_UNUSABLE,
            }
        elif child.retry_count > 0 and not completed_coding_output:
            # An ordinary retry now runs against the workspace the previous attempt left,
            # so a one-line defect costs one edit rather than a re-implementation. What the
            # reset used to protect -- a retry that returns only its fix silently dropping
            # the implementation from the commit -- is protected by the recorded lineage
            # instead: the completion's `file_changes` are the union across attempts, and
            # publication stages that union.
            #
            # Two cases keep the reset, both explicit and both recorded: the retry policy
            # classified the previous attempt as not worth building on (the wholesale
            # rewrite: drift is exactly what must not be edited into shape), and a checkout
            # that cannot answer basic Git commands. On that path the retry receives the
            # capture bounded at whole-file boundaries, never a mid-hunk character slice,
            # with every withheld file named.
            reset_trigger = _required_reset_trigger(child)
            capture: str | None = None
            try:
                capture = await self._capture_attempt(workspace)
            except RepositoryWorkspaceError:
                reset_trigger = reset_trigger or _RESET_WORKSPACE_UNUSABLE
            if reset_trigger is None:
                workspace_preserved = True
                attempt_inputs = {
                    "workspace": "preserved",
                    "capture_characters": len(capture) if capture else 0,
                }
            else:
                previous_attempt, previous_attempt_withheld = _bounded_previous_attempt(capture)
                await self._reset_workspace_tree(workspace)
                attempt_inputs = {
                    "workspace": "reset",
                    "reset_trigger": reset_trigger,
                    "capture_characters": len(capture) if capture else 0,
                    "capture_withheld_paths": list(previous_attempt_withheld),
                }
        elif completed_coding_output:
            # A completed coding receipt for this exact durable attempt means these dirty
            # files are the Engineer output a dead worker had not yet reviewed. Resetting them
            # turns recovery into a second Engineer call and destroys the bytes the journal
            # needs in order to reconcile its first call.
            _LOGGER.info(
                "child_coding_output_recovered",
                extra={
                    "feature_id": feature.feature_id,
                    "repository_id": repository.repository_id,
                    "child_workflow_id": child.child_workflow_id,
                    "attempt": child.retry_count,
                },
            )
            attempt_inputs = {"workspace": "recovered_coding_output"}

        def recorded(execution: ChildExecution) -> ChildExecution:
            """Stamp what this attempt ran against onto its durable record (§2.4).

            Sizes and repository-relative paths only -- never diff content. Recorded on the
            result rather than logged, because "did this retry see the previous attempt"
            was reconstructable only from a reflog that a cleanup sweep deletes.
            """
            return ChildExecution(
                result=execution.result.model_copy(
                    update={
                        "metadata": {**execution.result.metadata, "attempt_inputs": attempt_inputs}
                    }
                ),
                code_completion=execution.code_completion,
                review=execution.review,
                contract_change_request=execution.contract_change_request,
            )

        # Written on every attempt: the projection is untracked, so the reset above removes it.
        expected_openapi = await self._write_contract_projection(
            workspace, contract, operation_executor=operation_executor
        )
        # The revision the Engineer is about to look at: after the reset that discarded the
        # previous attempt, and after the projection this platform writes into the checkout.
        # Held for the whole attempt so every exit below has something to report even if the
        # workspace becomes unreadable between here and there.
        attempt_start_revision = await self._attempt_revision(
            workspace, fallback=child.current_revision
        )
        # Plans are made before clone-time repository evidence exists. Reconcile only
        # stale areas with one exact checkout match; all other guesses remain evidence,
        # not completion constraints.
        workstream, layout = reconcile_workstream_layout(workstream, workspace)
        # The plan carries a timeout into every command it selects, so the budget has to
        # be set here rather than only at the call site: a caller that passed a larger
        # value still left the recorded plan holding the default, and a repository's real
        # test suite was killed at two minutes and reported as failing.
        derived_plan_builder = DefaultValidationPlanBuilder(
            default_timeout_seconds=float(self._settings.validation_command_timeout_seconds)
        )
        # The validation plan is a property of the workstream's baseline, derived once on its
        # first attempt -- before the Engineer has written anything -- and pinned for every
        # later attempt and every consumer within one attempt. Re-deriving from the tree the
        # agent mutates let AB-Feature-216's coder mint a required `pytest -q .` gate by adding
        # the repository's first `test_*.py`. A late derivation that would add a command is
        # logged as drift and changes nothing.
        pinned_plan = _pinned_validation_plan(child)
        if pinned_plan is not None:
            _log_validation_plan_drift(
                pinned_plan,
                derived_plan_builder.build_plan_sync(workspace, repository_spec=repository),
                feature_id=feature.feature_id,
                repository_id=repository.repository_id,
                attempt=child.retry_count,
            )
        plan_builder: ValidationPlanBuilder = (
            PinnedValidationPlanBuilder(pinned_plan)
            if pinned_plan is not None
            else derived_plan_builder
        )
        validation_tool = WorkspaceValidationTools(
            workspace,
            repository_id=repository.repository_id,
            repository_spec=repository,
            cancellation_token=self._cancellation_token,
            operation_executor=operation_executor,
            process_runner=self._process_runner,
            plan_builder=plan_builder,
        )
        validation_plan = validation_tool.validation_plan()
        # From here on every consumer -- the commit gate, the in-attempt runners, the reviewer's
        # authoritative run -- answers from this attempt's one plan, even on the first attempt,
        # whose later consumers would otherwise re-derive from a tree the Engineer has since
        # written into.
        plan_builder = PinnedValidationPlanBuilder(validation_plan)
        # Before preflight, because the whole point of a repair is that preflight should
        # then pass. Applied only when somebody authorized it: a proposal nobody decided
        # about must never touch a checkout.
        repair_applied = await self._apply_approved_repair(
            feature,
            repository_id=repository.repository_id,
            workspace=workspace,
            operation_executor=operation_executor,
        )
        baseline_lint_failed = False
        preflight = await RepositoryPreflight(
            default_timeout_seconds=float(self._settings.default_agent_timeout_seconds),
            # The install runs a package manager over the network, not a model call, and a
            # timeout there is terminal. Its own budget, so widening it cannot widen the
            # lint probe, the reachability check or the placement check beside it.
            dependency_install_timeout_seconds=float(
                self._settings.dependency_install_timeout_seconds
            ),
            cancellation_token=self._cancellation_token,
            operation_executor=operation_executor,
            process_runner=self._process_runner,
        ).run(
            workspace,
            repository_id=repository.repository_id,
            # An install into a preserved tree whose lockfiles have not moved reproduces the
            # tree it is already looking at. 218 ran eleven of them, ~395 s, on workspaces
            # this platform had itself marked preserved. The previous digest is read off the
            # previous attempt's own preflight record; absent or unreadable means install.
            workspace_preserved=workspace_preserved,
            previous_lockfile_digest=(
                child.preflight_result.get("dependency_lockfile_digest")
                if isinstance(child.preflight_result, dict)
                else None
            ),
        )
        if preflight.validation_readiness != "blocked":
            baseline_lint_results = await validation_tool.run_lint_checks_async(
                # The same project-wide command the commit gate runs, so it needs the same
                # budget. Borrowing the per-agent one made a slow lint look like a failure.
                timeout_seconds=float(self._settings.commit_gate_timeout_seconds)
            )
            # Findings the checkout already had are not this change's to fix. A timeout or a
            # cancellation proves nothing either way, and treating it as a failing baseline
            # tells the next attempt its own defect belongs to somebody else.
            #
            # Only the first attempt measures a baseline at all. A retry re-enters this method
            # with the previous attempt's files still in the workspace, so linting then reports
            # the engineer's own defect and would attribute it to the repository.
            baseline_lint_failed = child.retry_count == 0 and any(
                result.status is not None
                and result.status.value == "failed"
                and not result.timed_out
                and not result.cancelled
                for result in baseline_lint_results
            )
            baseline_configuration_failures = {
                "lint_configuration_error",
                "missing_dependency",
                "unsupported_runtime",
            }
            failed_lint_results = [
                result
                for result in baseline_lint_results
                if (
                    result.status is not None
                    and result.status.value == "failed"
                    and result.failure_classification in baseline_configuration_failures
                )
            ]
            baseline_validation_failures = await _baseline_validation_failures(
                validation_tool,
                timeout_seconds=float(self._settings.validation_command_timeout_seconds),
                measure=child.retry_count == 0,
            )
            preflight = preflight.model_copy(
                update={
                    "baseline_lint_configuration_failures": [
                        validation_failure_identity(result) for result in failed_lint_results
                    ],
                    "baseline_validation_failures": baseline_validation_failures,
                }
            )
            preflight = _block_failed_baseline_validation(preflight, baseline_validation_failures)
            preflight = _block_unrepairable_lint_configuration(
                preflight, failed_lint_results, workspace
            )
        child = child.model_copy(
            update={
                "preflight_result": {
                    **preflight.model_dump(mode="json"),
                    # What an approved repair actually ran in this checkout, kept beside the
                    # readiness it produced. Without it, a repair that succeeded and a repair
                    # that was never applied look identical afterwards.
                    **({"applied_repair_commands": repair_applied} if repair_applied else {}),
                },
                "preflight_status": preflight.validation_readiness,
                "blocking_setup_issues": [
                    item.model_dump(mode="json") for item in preflight.blocking_issues
                ],
                "selected_package_manager": preflight.package_manager,
                "layout_evidence": {
                    "source_directories": list(layout.source_directories),
                    "test_directories": list(layout.test_directories),
                    "remapped_areas": [list(item) for item in layout.remapped_areas],
                    "unresolved_areas": list(layout.unresolved_areas),
                    "ambiguous_areas": list(layout.ambiguous_areas),
                },
                "configured_validation_commands": [
                    item.model_dump(mode="json") for item in validation_plan.commands
                ],
                # The workstream's plan, pinned on the durable row. The first attempt writes
                # the baseline derivation; every retry writes the same pin back, so the row
                # survives the per-cycle update in the loop and a resume reads what the
                # workstream started with.
                "validation_plan": validation_plan.model_dump(mode="json"),
                "test_availability": (
                    "tests_configured"
                    if any(item.validation_type == "test" for item in validation_plan.commands)
                    else "TEST_COMMAND_NOT_CONFIGURED"
                ),
                "implementation_expectations": [
                    item.model_dump(mode="json") for item in workstream.implementation_expectations
                ],
            }
        )
        if preflight.validation_readiness == "blocked":
            failure = classify_failure(preflight=preflight)
            retry_plan = build_retry_plan(
                failure,
                expected_source_areas=[
                    area
                    for expectation in workstream.implementation_expectations
                    for area in expectation.expected_source_areas
                ],
                previous_findings=[item.description for item in preflight.blocking_issues],
                current_revision=preflight.revision,
            )
            child = child.model_copy(
                update={
                    "failure_classification": failure.value,
                    "retry_strategy": retry_plan.model_dump(mode="json"),
                    "repository_setup_retry_count": child.repository_setup_retry_count + 1,
                }
            )
            result = _child_result(
                feature=feature,
                repository=repository,
                workstream=workstream,
                child=child,
                code_completion=None,
                review=None,
                status="failed",
                blocking_issues=[item.description for item in preflight.blocking_issues],
                # Preflight measured this itself, against this checkout, and nothing has
                # touched the workspace since. Reported rather than re-measured so the
                # blocked record and the preflight result it carries cannot disagree.
                current_revision=preflight.revision or attempt_start_revision,
            )
            return recorded(ChildExecution(result=result))
        # Decided by the orchestrator before this attempt was started, and persisted with it.
        # Read rather than recomputed: the routing decision is durable state, and a second
        # opinion formed here could disagree with the one the retry policy actually authorized.
        routing = ModelRoutingDecision.from_persisted(child.model_routing)
        registry_paths = _registry_paths(feature, repository.repository_id)
        if child.planned_blind and not registry_paths:
            # The plan for this repository was written without reconnaissance evidence, but
            # by now the checkout exists and is already being read -- so the structural
            # probe reconnaissance would have run is run here, and its wiring files flow
            # where recon results already flow. Inside the attempt the policy already
            # granted, before the Engineer is called; a second failure leaves this
            # workstream exactly as blind as it is today, recorded again, and never fails
            # the attempt or spends a retry.
            registry_paths = await self._late_reconnaissance_paths(
                feature_id=feature.feature_id,
                repository_id=repository.repository_id,
                workspace=workspace,
            )
        task_plan = _child_task_plan(
            child=child,
            technical_prd=technical_prd,
            contract=contract,
            workstream=workstream,
            feedback=feedback,
            registry_paths=registry_paths,
            model_routing=routing,
        )
        state["artifacts"].insert(1, task_plan)
        scoped_fix_client = (
            self._scoped_fix_client_factory(routing)
            if self._scoped_fix_client_factory is not None
            else None
        )
        self_review_client = (
            self._self_review_client_factory(routing)
            if self._self_review_client_factory is not None
            else None
        )
        engineer = EngineerAgent(
            prompt_loader=_prompt_loader(),
            coding_executor=ResponsesCodingExecutor(
                self._engineer_client_for(routing),
                max_file_bytes=self._settings.max_workspace_file_bytes,
            ),
            # The in-attempt repair pass runs on the SCOPED_FIX role when it resolves --
            # the exact selection the tier presets configure. None falls back inside the
            # agent to the primary executor, so the repair loop is never lost to a role.
            scoped_fix_executor=(
                ResponsesCodingExecutor(
                    scoped_fix_client,
                    max_file_bytes=self._settings.max_workspace_file_bytes,
                )
                if scoped_fix_client is not None
                else None
            ),
            file_tools_factory=lambda workspace_root: WorkspaceFileTools(
                workspace_root, max_file_bytes=self._settings.max_workspace_file_bytes
            ),
            # No git service: nothing is committed until the reviewer has approved this
            # change. Committing inside the engineer published work that later stages went
            # on to reject, left branches on the remote that no pull request referenced, and
            # made a retry look like a documentation-only change because the implementation
            # it was meant to complete was already in HEAD and therefore invisible to it.
            git_service_factory=None,
            cancellation_token=self._cancellation_token,
            operation_executor=operation_executor,
            source_formatter=RepositoryToolFormatter(
                process_runner=self._process_runner,
                cancellation_token=self._cancellation_token,
                timeout_seconds=float(self._settings.default_agent_timeout_seconds),
                # Verify with the repository's own lint command, which is what its commit
                # hook runs. Agreeing with the gate matters more than being thorough.
                gate=_commit_gate_command(validation_plan),
                gate_timeout_seconds=float(self._settings.commit_gate_timeout_seconds),
                gate_already_failing=bool(baseline_lint_failed),
            ),
            # The change's own tests, narrowed to the test files the attempt writes and run
            # inside it. Unjournaled and non-authoritative by construction: the reviewer runs
            # the repository's configured command afterwards and that remains the verdict.
            scoped_test_runner=RepositoryScopedTestRunner(
                process_runner=self._process_runner,
                cancellation_token=self._cancellation_token,
                repository_id=repository.repository_id,
                timeout_seconds=float(self._settings.validation_command_timeout_seconds),
                # The attempt's pinned plan, so the in-attempt narrowing answers from the
                # same command set as the gate and the reviewer -- never from a derivation
                # over files this same attempt has just written.
                plan_builder=plan_builder,
            ),
            # The checkout's own typecheck, asked inside the attempt so a type error costs one
            # repair pass instead of a whole outer attempt. Unjournaled and non-authoritative
            # by construction, exactly like the narrowed test run above it: the reviewer runs
            # the repository's configured commands afterwards and that remains the verdict.
            typecheck_runner=RepositoryTypecheckRunner(
                process_runner=self._process_runner,
                cancellation_token=self._cancellation_token,
                repository_id=repository.repository_id,
                timeout_seconds=float(self._settings.validation_command_timeout_seconds),
                plan_builder=plan_builder,
            ),
            # The deterministic wiring inspection, asked inside the attempt so a module
            # nothing reaches costs one repair pass instead of a whole outer attempt. The
            # same implementation `_unreachable_addition_issues` runs below, over the same
            # checkout, deliberately: this one is fast feedback and that one is the
            # authority, and neither consults the other.
            reachability_checker=RepositoryReachabilityChecker(
                process_runner=self._process_runner or AsyncioProcessRunner(),
                cancellation_token=self._cancellation_token,
                timeout_seconds=float(self._settings.default_agent_timeout_seconds),
                excluded_paths=(_CONTRACT_PROJECTION,),
            ),
            # The deterministic placement check, and the only in-attempt inspection with no
            # counterpart below: nothing in the runtime asks this question, deliberately.
            # Whether code belongs in the file a plan named is a review judgement, and a gate
            # that refused a publication on it would be the platform overruling the reviewer.
            # Inside the attempt it is a free hint; outside it, it would be an authority.
            assigned_file_checker=RepositoryAssignedFileChecker(
                process_runner=self._process_runner or AsyncioProcessRunner(),
                cancellation_token=self._cancellation_token,
                timeout_seconds=float(self._settings.default_agent_timeout_seconds),
                excluded_paths=(_CONTRACT_PROJECTION,),
            ),
            # The one bounded self-review of the finished work, on the CODING role at this
            # feature's pinned tier. Gates `completion_status` and nothing further; the
            # commit gate below and the independent Reviewer still run whatever it says.
            self_reviewer=(
                CodingRoleSelfReviewer(self_review_client)
                if self_review_client is not None
                else None
            ),
            dependency_synchronizer=RepositoryToolSynchronizer(
                process_runner=self._process_runner,
                cancellation_token=self._cancellation_token,
            ),
            # The deterministic design-value inspection, and this workstream's own assigned
            # design at build fidelity for it to compare against. Read from the feature rather
            # than resolved here, like every other reader of the detail artifact: one
            # resolution per workstream, reused by every attempt.
            # Defaulted, never skipped: this executor is constructed without a process
            # runner, so a `None` guard here meant the design-value check never ran at all.
            design_value_checker=DesignValueChecker(
                process_runner=self._process_runner
                or AsyncioProcessRunner(output_limit_bytes=128 * 1024),
                cancellation_token=self._cancellation_token,
                timeout_seconds=float(self._settings.default_agent_timeout_seconds),
            ),
            design_contents=_assigned_design_contents(feature, repository.repository_id),
            # So the role that writes the CSS can see the frames it is building, not only
            # their coordinates.
            attachments=self._attachments,
            previous_attempt_diff=previous_attempt,
            previous_attempt_withheld_files=previous_attempt_withheld,
            previous_attempt_preserved=workspace_preserved,
            lint_capability_probe=RepositoryLintCapabilities(
                process_runner=self._process_runner,
                cancellation_token=self._cancellation_token,
                timeout_seconds=float(self._settings.default_agent_timeout_seconds),
            ),
            model_routing=routing,
            # Derived exactly once, here, where the persisted routing decision and the
            # deployment's declarations are both already in scope: the routed model resolves
            # to a declared context window, the window to a snapshot budget. No second
            # budget arithmetic exists anywhere downstream (the 61- law).
            snapshot_budget=repository_snapshot_budget(
                routing.model if routing is not None else None,
                normalize_context_window_tokens(self._settings.declared_context_window_tokens()),
            ),
        )
        try:
            engineer_update = await engineer.run(state)
        except EmptyCommitError:
            # Not a fault. The attempt reproduced the committed content exactly, which the
            # retry policy already refuses on its own terms; raising here abandoned a
            # workstream over an outcome it knows how to handle. The repair instruction makes
            # this more likely, not less, because it asks for a minimal change.
            return recorded(
                _no_change_execution(
                    feature=feature, repository=repository, workstream=workstream, child=child
                )
            )
        except SourceValidationError as error:
            # Handled here rather than in the orchestrator so the failure keeps the child
            # state this method already established. Letting it propagate discarded the
            # preflight result, layout evidence, and validation plan, leaving an operator
            # with diagnostics and no record of the checkout they describe.
            #
            # The completion the Engineer produced before the formatter rejected it is
            # attached rather than dropped: it is the only record of what the attempt was
            # shown and wrote, its status is `failed` and its readiness is false, so nothing
            # can select it for publication -- and the next retry's state carries it, which
            # is what lets that retry edit these files instead of regenerating them.
            rejected_completion = cast(CodeCompletionArtifact | None, error.code_completion)
            # Which failure this is -- the commit gate's own deterministic rejection, or a
            # self-review that judged the assignment unmet -- and the markers each carries,
            # decided by the one shared disposition so this handler and the orchestrator's
            # cannot classify the same attempt two ways.
            #
            # For the gate's own rejection: the repository's linter produced these, so
            # repeating them is the tool being consistent, not the model refusing to listen,
            # and `source_validation_rejected` is the fact the retry loop's pre-commit
            # allowance is actually about -- this attempt was stopped by the commit gate,
            # before it produced a code completion or committed anything. The loop used to
            # infer that from the classification plus an absent commit SHA, and neither
            # distinguishes this from an ordinary review rejection; sixty-two per cent of
            # the attempts in fifteen features took an allowance meant for this path.
            failure_classification, rejection_metadata = source_rejection_disposition(error)
            return recorded(
                ChildExecution(
                    result=_child_result(
                        feature=feature,
                        repository=repository,
                        workstream=workstream,
                        child=child,
                        code_completion=rejected_completion,
                        review=None,
                        status="failed",
                        blocking_issues=list(error.diagnostics),
                        current_revision=await self._attempt_revision(
                            workspace, fallback=attempt_start_revision
                        ),
                    ).model_copy(
                        update={
                            "failure_classification": failure_classification,
                            "metadata": dict(rejection_metadata),
                            # What the gate actually ran, where a check it ran produces a
                            # structured verdict -- today the change's own scoped tests.
                            # `_child_result` fills this from the review, and a gate-rejected
                            # attempt has no review, so the field was empty on every one of
                            # them. Empty is read as *unknown* by the coarse failure counter,
                            # which is correct and is why its walk-back broke here: 218's
                            # four consecutive failures on one suite were counted as two.
                            # A lint or typecheck rejection still records nothing, because
                            # neither produces a result this platform kept.
                            "current_validation_results": list(error.validation_results),
                            "preflight_result": child.preflight_result,
                            "technology_profile": validation_plan.technology_profile.model_dump(
                                mode="json"
                            ),
                            "validation_plan": validation_plan.model_dump(mode="json"),
                        }
                    ),
                    code_completion=rejected_completion,
                )
            )
        code_completion = cast(CodeCompletionArtifact, engineer_update["artifacts"][0])
        # What the workstream's branch holds, which is what the completion contract is about.
        # Judging a targeted remediation against the plan's whole contract using only its own
        # one-file diff is unsatisfiable by construction: 201's frontend was refused six times
        # for tests its own branch already carried.
        category_evidence = await self._category_evidence(
            workspace, feature=feature, repository=repository, child=child
        )
        completeness = validate_implementation_completeness(
            workstream,
            code_completion,
            branch_paths=category_evidence.paths,
            branch_modified_paths=category_evidence.modified_paths,
            branch_evidence_source=category_evidence.source,
        )
        # On the durable record of every exit below, alongside what this attempt ran against:
        # a remediation attempt judged from `attempt_declared_paths` has the pre-66 behaviour
        # however few findings it happens to carry, and that must be readable rather than
        # inferred.
        attempt_inputs = {
            **attempt_inputs,
            "category_evidence_source": completeness.category_evidence_source,
        }
        production_fingerprint = _reviewed_source_fingerprint(
            workspace,
            completeness.production_files_changed,
            completeness.production_diff_fingerprint,
        )
        test_fingerprint = _reviewed_source_fingerprint(
            workspace,
            completeness.test_files_changed,
            completeness.test_diff_fingerprint or "",
        )
        code_completion = code_completion.model_copy(
            update={
                "production_files_changed": completeness.production_files_changed,
                "test_files_changed": completeness.test_files_changed,
                "configuration_files_changed": completeness.configuration_files_changed,
                "requirements_implemented": completeness.requirements_implemented,
                "requirements_not_implemented": completeness.requirements_not_implemented,
                "implementation_expectations_satisfied": (
                    completeness.implementation_expectations_satisfied
                ),
            }
        )
        child = child.model_copy(
            update={
                "production_files_changed": completeness.production_files_changed,
                "test_files_changed": completeness.test_files_changed,
                "configuration_files_changed": completeness.configuration_files_changed,
                "requirements_implemented": completeness.requirements_implemented,
                "requirements_not_implemented": completeness.requirements_not_implemented,
                "production_diff_fingerprint": production_fingerprint,
                "test_diff_fingerprint": test_fingerprint,
            }
        )
        state["artifacts"].append(code_completion)
        self._wiring_repairs = []
        self._textual_wirings = []
        wholesale_rewrites = await self._wholesale_rewrite_issues(workspace)
        rewritten = [
            *wholesale_rewrites,
            *await self._unreachable_addition_issues(
                workspace, assigned_paths=tuple(workstream.expected_files_or_areas)
            ),
        ]
        if self._textual_wirings:
            # Onto the durable record via the attempt_inputs stamp every exit already gets:
            # the gate passed, and what satisfied it was text. The operator can read that
            # from the result instead of from a diff.
            attempt_inputs = {**attempt_inputs, "textual_wirings": list(self._textual_wirings)}
        if rewritten:
            # The gate knows the file and the symbol, so the next attempt is given that as a
            # scoped instruction instead of the whole task again. Re-running the full
            # implementation prompt regenerated every file and lost the one edit it was
            # asked for: chk-1 rebuilt the same four files three times over one line of JSX.
            retry_strategy = dict(child.retry_strategy or {})
            if self._wiring_repairs:
                # `wiring_repair` stays the first repair so anything reading a single scoped
                # instruction keeps working; `wiring_repairs` carries the rest.
                retry_strategy["wiring_repair"] = self._wiring_repairs[0]
                retry_strategy["wiring_repairs"] = list(self._wiring_repairs)
            if wholesale_rewrites:
                # A rewrite is drift, and drift is exactly what the next attempt must not
                # edit into shape: this marker is what sends that retry down the reset path,
                # and it is the one policy trigger that still discards a workspace.
                retry_strategy["workspace_reset_required"] = _RESET_WHOLESALE_REWRITE
            child = child.model_copy(
                update={
                    "failure_classification": FailureClassification.IMPLEMENTATION_MISSING.value,
                    "retry_strategy": retry_strategy or child.retry_strategy,
                }
            )
            result = _child_result(
                feature=feature,
                repository=repository,
                workstream=workstream,
                child=child,
                code_completion=code_completion,
                review=None,
                status="failed",
                blocking_issues=rewritten,
                current_revision=await self._attempt_revision(
                    workspace, fallback=attempt_start_revision
                ),
            )
            # Marked so the retry loop does not read this gate repeating itself as an
            # attempt that ignored its feedback: the sentence is fixed, the condition is not.
            metadata: dict[str, Any] = {**result.metadata, "deterministic_gate": True}
            if self._wiring_repairs:
                metadata["wiring_repair"] = self._wiring_repairs[0]
                metadata["wiring_repairs"] = list(self._wiring_repairs)
            if wholesale_rewrites:
                # The loop rebuilds the retry strategy from scratch; metadata is the channel
                # its carrier reads, so the marker has to travel here as well.
                metadata["workspace_reset_required"] = _RESET_WHOLESALE_REWRITE
            result = result.model_copy(update={"metadata": metadata})
            return recorded(ChildExecution(result=result, code_completion=code_completion))
        if not completeness.passed:
            failure = classify_failure(completeness=completeness)
            retry_plan = build_retry_plan(
                failure,
                expected_source_areas=[
                    area
                    for expectation in workstream.implementation_expectations
                    for area in expectation.expected_source_areas
                ],
                previous_findings=[item.description for item in completeness.findings],
                current_revision=preflight.revision,
            )
            child = child.model_copy(
                update={
                    "failure_classification": failure.value,
                    "retry_strategy": retry_plan.model_dump(mode="json"),
                }
            )
            result = _child_result(
                feature=feature,
                repository=repository,
                workstream=workstream,
                child=child,
                code_completion=code_completion,
                review=None,
                status="failed",
                blocking_issues=[item.description for item in completeness.findings],
                current_revision=await self._attempt_revision(
                    workspace, fallback=attempt_start_revision
                ),
            ).model_copy(
                # The completeness check is deterministic like the gates below it, so the
                # convergence rule must not read its steady wording as a model ignoring
                # feedback. ser-3 lost its backend to that at the second attempt, over a
                # missing documentation edit, with its frontend already approved.
                #
                # The evidence source travels with the rejection itself: this is the record a
                # 201-shaped repetition would be diagnosed from, and "which branch state was
                # this measured against" is the first question to ask of it.
                update={
                    "metadata": {
                        "deterministic_gate": True,
                        "category_evidence_source": completeness.category_evidence_source,
                    }
                }
            )
            return recorded(ChildExecution(result=result, code_completion=code_completion))
        if (
            expected_openapi is not None
            and WorkspaceFileTools(
                workspace, max_file_bytes=self._settings.max_workspace_file_bytes
            ).read_file("openapi.yaml")
            != expected_openapi
        ):
            change_request = _contract_change_request(
                feature=feature,
                repository=repository,
                child=child,
                contract=contract,
            )
            result = _child_result(
                feature=feature,
                repository=repository,
                workstream=workstream,
                child=child,
                code_completion=code_completion,
                review=None,
                status="waiting_for_contract_change",
                blocking_issues=["Generated openapi.yaml differs from the approved contract."],
                current_revision=await self._attempt_revision(
                    workspace, fallback=attempt_start_revision
                ),
            )
            return recorded(
                ChildExecution(
                    result=result,
                    code_completion=code_completion,
                    contract_change_request=change_request,
                )
            )
        partition = (
            _context_partition_scope(child.retry_strategy)
            if child.pending_context_partition
            else None
        )
        if partition is not None and partition.get("final") is not True:
            # An intermediate cluster of a context-partition wave (80-): the engineer ran
            # narrowed to its cluster and the deterministic gates above all passed, but the
            # review is deferred to the wave's last cluster -- the review judges the
            # workstream, not the delta (66-), and a per-cluster review would report every
            # not-yet-addressed cluster's findings as manufactured recurrences. The result
            # below is a control-flow signal to the wave and is never persisted as an
            # artifact; only the code completion enters the lineage, stamped by the wave.
            result = _child_result(
                feature=feature,
                repository=repository,
                workstream=workstream,
                child=child,
                code_completion=code_completion,
                review=None,
                status="failed",
                blocking_issues=[],
                current_revision=await self._attempt_revision(
                    workspace, fallback=attempt_start_revision
                ),
            )
            result = result.model_copy(
                update={
                    "metadata": {
                        **result.metadata,
                        "context_partition_intermediate": True,
                        "context_partition_index": partition.get("index"),
                    }
                }
            )
            return recorded(ChildExecution(result=result, code_completion=code_completion))
        # The design questions a person has decided about this repository, read from the same
        # lineage the Engineer's settled lines and the retry authority's exclusion read (73-).
        # Computed once here and used twice: the reviewer prompt is told what was overruled,
        # and the verdict handling below enforces it whatever the model does with the telling.
        settled = settled_questions(feature.artifacts, repository_id=repository.repository_id)
        review_update = await ProductionReviewer(
            settings=self._settings,
            llm_client=self._reviewer_client,
            cancellation_token=self._cancellation_token,
            operation_executor=operation_executor,
            repository_id=repository.repository_id,
            process_runner=self._process_runner,
            # Always the review role, whatever the Engineer was routed to. Role separation is
            # logical rather than a matter of which model is configured: the reviewer is the
            # only thing that may mark a finding resolved, and it never writes code.
            model_role=ModelRole.REVIEW,
            settled_questions=settled,
            # So the judge can see the design it is judging against. A reviewer holding
            # coordinates cannot answer "does it look like the design"; one holding the
            # picture can.
            attachments=self._attachments,
            # The attempt's pinned plan: the reviewer's authoritative run must judge with the
            # workstream's configured commands, not with a derivation over the checkout this
            # attempt has just mutated -- that derivation is how 216's coder minted a required
            # pytest gate the repository never configured.
            plan_builder=plan_builder,
        ).run(state)
        review = cast(ReviewArtifact, review_update["artifacts"][0])
        # Measured here, before publication, because this is the revision the verdict is
        # about. Taken after the commit it would describe a checkout the reviewer never saw:
        # the same content, but a different HEAD and a clean worktree, so an approval would
        # record a revision nothing was ever validated at.
        reviewed_revision = await self._attempt_revision(workspace, fallback=attempt_start_revision)
        remediation_scope: _RemediationReviewScope | None = None
        if previous_review is not None:
            prior_blocking = _blocking_findings(
                previous_review, bounded=self._settings.bounded_review_scope
            )
            remediation_scope = _RemediationReviewScope(
                prior_finding_ids=frozenset(item.finding_id for item in prior_blocking),
                prior_fingerprints=frozenset(finding_fingerprint(item) for item in prior_blocking),
                delta_paths=frozenset(
                    path
                    for path in (code_completion.metadata.get("attempt_changed_paths") or ())
                    if isinstance(path, str)
                ),
            )
        outcome = _resolve_review_outcome(
            review,
            bounded=self._settings.bounded_review_scope,
            remediation=remediation_scope,
            # What an approved result may legally carry, asked of the change rather than
            # assumed from the verdict: publishing without an approval must not produce a
            # result the artifact's own validator rejects.
            publishable=not code_completion.requirements_not_implemented
            and all(item.passed for item in code_completion.validation_results),
            settled=settled,
        )
        if outcome.overruled_findings:
            # The audit trace, on the review artifact itself, before anything records or
            # publishes it: which finding was demoted, matched on which fingerprint, under
            # whose decision, when. The findings list stays untouched -- the review still
            # says what the reviewer said; this metadata says what the platform did about
            # it and on whose authority. A journal replay reproduces the stamp, because it
            # is a pure function of the artifact and the settled questions.
            review = review.model_copy(
                update={
                    "metadata": {
                        **review.metadata,
                        "overruled_findings": [
                            item.model_dump(mode="json") for item in outcome.overruled_findings
                        ],
                    }
                }
            )
        accepted = review.verdict == "approved" or outcome.advisory_verdict is not None
        if accepted:
            try:
                code_completion = await self._publish_approved_change(
                    git_service,
                    workspace=workspace,
                    branch_name=child.branch_name,
                    summary=code_completion.summary,
                    code_completion=code_completion,
                    review=review,
                    workflow_id=child.child_workflow_id,
                    repository_id=repository.repository_id,
                    workstream_id=workstream.workstream_id,
                )
            except EmptyCommitError:
                # Approved, but there is nothing to publish: the files reproduce the
                # committed revision exactly. Refusable, not a fault.
                return recorded(
                    _no_change_execution(
                        feature=feature,
                        repository=repository,
                        workstream=workstream,
                        child=child,
                    )
                )
        result = _child_result(
            feature=feature,
            repository=repository,
            workstream=workstream,
            child=child,
            code_completion=code_completion,
            review=review,
            status="approved" if accepted else "failed",
            blocking_issues=outcome.blocking_issues,
            current_revision=reviewed_revision,
            advisory_findings=outcome.advisory_findings,
            review_finding_counts=outcome.counts,
            advisory_review_verdict=outcome.advisory_verdict,
            overruled_findings=outcome.overruled_findings,
        )
        if not accepted:
            if review.metadata.get("manual_review_required") is True:
                return recorded(
                    ChildExecution(
                        result=result.model_copy(
                            update={
                                "metadata": {
                                    **result.metadata,
                                    "retry_refusal_reason": "manual_review_required",
                                }
                            }
                        ),
                        code_completion=code_completion,
                        review=review,
                    )
                )
            classification = _review_failure_classification(review)
            retry_plan = build_retry_plan(
                classification,
                expected_source_areas=[
                    area
                    for expectation in workstream.implementation_expectations
                    for area in expectation.expected_source_areas
                ],
                previous_findings=result.blocking_issues,
                current_revision=result.current_revision,
                # A review rejection and a failing command share this classification, and the
                # advice for the two is opposite -- see `build_retry_plan`.
                source_verdict_available=a_required_command_rejected_the_source(
                    result.current_validation_results
                ),
            )
            result = result.model_copy(
                update={
                    "failure_classification": classification.value,
                    "retry_strategy": retry_plan.model_dump(mode="json"),
                }
            )
        return recorded(
            ChildExecution(result=result, code_completion=code_completion, review=review)
        )

    async def _category_evidence(
        self,
        workspace: Path,
        *,
        feature: FeatureWorkflowSnapshot,
        repository: RepositorySpec,
        child: ChildWorkflowReference,
    ) -> _CategoryEvidence:
        """Find the best available evidence for what this workstream's branch holds.

        Three sources, first one that answers wins, and the answer says which it was.

        1. The branch itself: committed work since the lineage base, plus the whole
           uncommitted worktree. This is the only source that can answer 201, whose test
           files live in a completion stamped `published_after_approval` -- the one thing
           every lineage reader on this platform deliberately excludes.
        2. The lineage's completions including the published ones, where the base did not
           resolve to a real branch point and a branch diff would therefore be measured
           against `HEAD`, which already contains the approved attempt's own commits.
        3. Nothing, which leaves the gate reading the attempt's own declared buckets exactly
           as it did before. Recorded and logged rather than fallen through silently: this is
           the pre-66 behaviour, and a run that regressed to it has to be legible as such.

        The contract projection is excluded because the platform writes it into every child
        workspace on every attempt, so it is permanently untracked there -- and it classifies
        as production source, which would satisfy a production promise nobody kept.
        """
        runner = self._process_runner or AsyncioProcessRunner()
        evidence = await branch_change_evidence(
            workspace=workspace,
            baseline=LineageBaseRevision(
                workspace=workspace,
                # The lineage of a revision child leaves its *base* branch, not the
                # repository default: measured against the default it would re-count the
                # whole superseded run's work as this attempt's changes.
                default_branch=child.base_branch or repository.default_branch,
                process_runner=runner,
                cancellation_token=self._cancellation_token,
            ),
            process_runner=runner,
            cancellation_token=self._cancellation_token,
            timeout_seconds=float(self._settings.default_agent_timeout_seconds),
            excluded_paths=(_CONTRACT_PROJECTION,),
        )
        if evidence is not None:
            return _CategoryEvidence(
                paths=evidence.paths,
                modified_paths=evidence.modified_paths,
                source="branch_diff",
            )
        completions = _prior_child_completions_including_published(feature, child)
        if completions:
            paths: list[str] = []
            modified: list[str] = []
            for completion in completions:
                for change in completion.file_changes:
                    if change.change_type == "deleted":
                        continue
                    paths.append(change.path)
                    if change.change_type == "modified":
                        modified.append(change.path)
                # The declared buckets as well as the file changes: a completion's own
                # bucketing is a claim about files it wrote, and a path named in only one of
                # the two places is still a file that attempt put on this branch.
                paths.extend(completion.production_files_changed)
                paths.extend(completion.test_files_changed)
                paths.extend(completion.configuration_files_changed)
            return _CategoryEvidence(
                paths=tuple(dict.fromkeys(paths)),
                modified_paths=tuple(dict.fromkeys(modified)),
                source="lineage_completions",
            )
        if child.retry_count > 0:
            _LOGGER.warning(
                "child_completeness_evidence_attempt_only",
                extra={
                    "feature_id": feature.feature_id,
                    "repository_id": repository.repository_id,
                    "child_workflow_id": child.child_workflow_id,
                    "attempt": child.retry_count,
                },
            )
        return _CategoryEvidence(paths=None, modified_paths=(), source="attempt_declared_paths")

    async def _attempt_revision(self, workspace: Path, *, fallback: str | None) -> str | None:
        """Measure the revision this checkout is at right now, and never lose a known one.

        Asked of the checkout directly rather than of review metadata. That is the whole
        change: `calculate_repository_revision` needs no review, no validation run and no
        model, so an attempt stopped at any of the gates before review can still say which
        revision it was stopped against. It fingerprints HEAD, the index, the worktree and
        untracked files, so it also distinguishes two attempts on the same commit.

        Failing to fingerprint is never allowed to decide an attempt. A revision is a fact
        recorded *about* an outcome; raising here would replace a real result -- an approval,
        or a diagnosis somebody needs -- with a Git error about how it was labelled. The
        fallback is the last revision this workstream is known to have been at, because a
        record that keeps saying an older revision is still readable, and one that says
        nothing is not.
        """
        try:
            revision = await calculate_repository_revision(workspace)
        except (OSError, ValueError) as error:
            _LOGGER.warning(
                "child_attempt_revision_unavailable",
                extra={"workspace": str(workspace), "error_type": type(error).__name__},
            )
            return fallback
        return revision.combined_fingerprint or fallback

    def _engineer_client_for(self, routing: ModelRoutingDecision | None) -> LLMClient:
        """Return the Engineer's model boundary for the decision persisted on this attempt.

        A child written before routing existed, or one whose decision could not be read, gets
        the coding role: that is what the Engineer has always used, and defaulting to a
        scoped repair would be inventing a classification nobody decided on. Historical tier
        names are mapped only here so an in-flight record from an older build remains runnable.
        """
        if self._engineer_client_factory is None:
            return self._engineer_client
        return self._engineer_client_factory(routing)

    async def _completed_coding_output_matches(
        self,
        *,
        feature: FeatureWorkflowSnapshot,
        repository: RepositorySpec,
        child: ChildWorkflowReference,
        workspace: Path,
    ) -> bool:
        """Find a completed Engineer operation for this claimed attempt and verify its bytes."""
        if child.pull_request_artifact_id is not None:
            # Parent persistence has already settled the publication produced by this coding
            # receipt. A later integration remediation is new work even if an old snapshot
            # reuses the same numeric counter; it must not be mistaken for crash recovery.
            return False
        for operation in reversed(
            await self._journal.list_operations_for_workflow(feature.workflow_id)
        ):
            if (
                operation.operation_type is ExternalOperationType.RUN_CODING_EXECUTOR
                and operation.status is ExternalOperationStatus.SUCCEEDED
                and operation.child_workflow_id == child.child_workflow_id
                and operation.repository_id == repository.repository_id
                and child_attempt_of(operation) == child.retry_count
            ):
                return completed_coding_operation_matches_workspace(operation, workspace)
        return False

    async def _remove_generated_output(self, workspace: Path) -> None:
        """Drop ignored build artifacts so the repository's own hook is not made slow by us."""
        runner = self._process_runner or AsyncioProcessRunner()
        timeout = float(self._settings.default_agent_timeout_seconds)
        result = await runner.run(
            ("git", "clean", "--force", "-d", "-X", "--", *_cleanable_paths(workspace)),
            workspace,
            timeout,
            self._cancellation_token,
            repository_subprocess_environment(),
        )
        _require_workspace_command(result, stage="generated-output cleanup")
        # A hook installer writes its bootstrap into an ignored directory, so a clean can
        # remove it however carefully it is excluded. Re-running the repository's own declared
        # setup restores whatever it owns, rather than the platform guessing at the paths.
        await self._restore_declared_setup(workspace, timeout)

    async def _restore_declared_setup(self, workspace: Path, timeout: float) -> None:
        """Re-run the checkout's own bootstrap script when it declares one."""
        manifest = workspace / "package.json"
        if not manifest.is_file():
            return
        try:
            scripts = json.loads(manifest.read_text(encoding="utf-8")).get("scripts", {})
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(scripts, dict) or "prepare" not in scripts:
            return
        runner = self._process_runner or AsyncioProcessRunner()
        manager = _declared_node_package_manager(workspace)
        # The declared version, matching the manager that installed this workspace.
        argv, _version = package_manager_argv(manager, workspace, workspace)
        result = await runner.run(
            (*argv, "run", "prepare"),
            workspace,
            timeout,
            self._cancellation_token,
            repository_subprocess_environment(),
        )
        _require_workspace_command(result, stage="declared setup restore")

    async def _recover_approved_publication(
        self,
        git_service: InterruptibleGitService,
        *,
        feature: FeatureWorkflowSnapshot,
        repository: RepositorySpec,
        workstream: RepositoryWorkstreamPlan,
        child: ChildWorkflowReference,
        workspace: Path,
    ) -> ChildExecution | None:
        """Finish a journaled approved publication before any coding is re-entered.

        The child result is checkpointed after this method returns. A worker can therefore
        die after commit or push while the durable child state still says ``before_coding``.
        The commit intent carries the validated artifact receipt needed to reconstruct that
        result; replaying the model against a now-committed checkout would create a new prompt
        and bypass ordinary operation reuse.

        That window is the whole point, and it closes as soon as the publication's outcome is
        recorded in durable child state -- the receipt's completion referenced by the child
        row or by a persisted child result. A receipt whose outcome durable state already
        holds is settled history, not work in doubt -- and treating it as recoverable made
        every later attempt on that repository a replay of the old publication rather than a
        new attempt. AB-Feature-111's console was approved and published at attempt 7;
        integration review then asked it for one precise fix, and each granted attempt died
        in under a second against evidence from the very publication that had already
        succeeded. A repository could be published, or corrected, but never both.

        The pull request used to be the only evidence of settledness, and it is recorded
        only after the integration review approves the feature -- so an integration review
        that requested changes routed the fix attempt straight back into this recovery
        against its own settled publication, and AB-Feature-184 died the same death at
        attempt 3 that -111 died at attempt 8 (see `_publication_outcome_checkpointed`).
        """
        if child.pull_request_artifact_id is not None:
            # A cheap sufficient condition: the pull request is recorded only after the
            # publication reached its terminal outcome. Not the only one -- the settled
            # check below closes the window for outcomes recorded before any pull request.
            return None
        recoverable_statuses = {
            ExternalOperationStatus.STARTING,
            ExternalOperationStatus.RUNNING,
            ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
            ExternalOperationStatus.SUCCEEDED,
        }
        candidate = None
        for operation in await self._journal.list_operations_for_workflow(feature.workflow_id):
            receipt = operation.safe_metadata.get("publication_receipt")
            if (
                operation.operation_type is ExternalOperationType.CREATE_COMMIT
                and operation.child_workflow_id == child.child_workflow_id
                and operation.repository_id == repository.repository_id
                and operation.status in recoverable_statuses
                and isinstance(receipt, dict)
                and receipt
            ):
                candidate = operation
        if candidate is None:
            return None

        receipt = candidate.safe_metadata["publication_receipt"]
        if not isinstance(receipt, dict):  # narrowed above; retain a fail-closed type boundary
            return None
        if _publication_outcome_checkpointed(
            feature, child=child, candidate=candidate, receipt=receipt
        ):
            # Settled history, not work in doubt: the ordinary attempt path takes over,
            # with the workspace preserved and the prior completions in its lineage.
            return None
        try:
            if (
                receipt.get("child_workflow_id") != child.child_workflow_id
                or receipt.get("repository_id") != repository.repository_id
                or receipt.get("workstream_id") != workstream.workstream_id
            ):
                raise ValueError("publication receipt ownership does not match this child")
            code_completion = CodeCompletionArtifact.model_validate_json(
                json.dumps(receipt["code_completion"])
            )
            review = ReviewArtifact.model_validate_json(json.dumps(receipt["review"]))
            expected_files = candidate.safe_metadata["expected_staged_files"]
            expected_fingerprint = candidate.safe_metadata["reviewed_content_fingerprint"]
            commit_message = candidate.safe_metadata["commit_message"]
            if not isinstance(expected_files, list) or not all(
                isinstance(path, str) for path in expected_files
            ):
                raise ValueError("publication receipt file paths are invalid")
            safe_expected_files = cast(list[str], expected_files)
            if (
                review.verdict != "approved"
                or code_completion.commit_sha is not None
                or sorted(change.path for change in code_completion.file_changes)
                != sorted(safe_expected_files)
                or not isinstance(expected_fingerprint, str)
                or not isinstance(commit_message, str)
            ):
                raise ValueError("publication receipt does not describe approved exact files")
            require_reviewed_workspace_match(
                review,
                reviewed_paths=safe_expected_files,
                content_fingerprint=expected_fingerprint,
                evidence_required=True,
            )
        except (AgentArtifactError, KeyError, TypeError, ValueError, ValidationError) as error:
            raise AmbiguousPublicationReceiptError(
                operation_id=candidate.operation_id,
                branch_name=child.branch_name,
            ) from error

        commit_sha = await git_service.commit(
            workspace,
            commit_message,
            files=safe_expected_files,
            expected_content_fingerprint=expected_fingerprint,
            publication_receipt=receipt,
        )
        await self._cancellation_token.raise_if_cancelled()
        await git_service.push(workspace, child.branch_name, expected_commit_sha=commit_sha)
        published = code_completion.model_copy(
            update={
                "commit_sha": commit_sha,
                "metadata": {
                    **code_completion.metadata,
                    "published_after_approval": True,
                    "publication_recovered": True,
                    "reviewed_content_fingerprint": expected_fingerprint,
                },
            }
        )
        return ChildExecution(
            result=_child_result(
                feature=feature,
                repository=repository,
                workstream=workstream,
                child=child,
                code_completion=published,
                review=review,
                status="approved",
                blocking_issues=[],
                # The one path where the review's own recorded revision is preferred. This
                # recovers a publication whose commit already exists, so measuring the
                # checkout now would report a post-commit state that no review ever saw.
                # The measurement is the fallback, for a review that recorded none.
                current_revision=(
                    _current_revision(review)
                    or await self._attempt_revision(workspace, fallback=child.current_revision)
                ),
            ),
            code_completion=published,
            review=review,
        )

    async def _publish_approved_change(
        self,
        git_service: InterruptibleGitService,
        *,
        workspace: Path,
        branch_name: str,
        summary: str,
        code_completion: CodeCompletionArtifact,
        review: ReviewArtifact,
        workflow_id: str,
        repository_id: str | None = None,
        workstream_id: str | None = None,
    ) -> CodeCompletionArtifact:
        """Commit and push only a change the reviewer has already approved.

        The commit is the last step of a successful workstream rather than a step on the way
        to one, so nothing reaches the remote that a later stage goes on to reject, and every
        attempt is judged on the whole change instead of on whatever an earlier attempt had
        not already committed.
        """
        paths = [change.path for change in code_completion.file_changes]
        # The validation build leaves generated output in the workspace, and the repository's
        # pre-commit hook lints the whole project. Twenty-four megabytes of bundles turned a
        # seconds-long hook into a ten-minute one, and the commit was killed part-way through.
        # None of it is committed, so removing it costs nothing and is not the model's work.
        await self._remove_generated_output(workspace)
        # Cleanup may invoke a repository-declared prepare script. Compare the bytes that now
        # enter Git with the exact paths and fingerprint ReviewerAgent recorded, immediately
        # before the adapter establishes its own pre-/post-hook integrity boundary.
        content_fingerprint = reviewed_content_fingerprint(workspace, paths)
        require_reviewed_workspace_match(
            review,
            reviewed_paths=paths,
            content_fingerprint=content_fingerprint,
            evidence_required=True,
        )
        publication_receipt: dict[str, object] = {}
        if repository_id is not None and workstream_id is not None:
            publication_receipt = {
                "receipt_version": 1,
                "child_workflow_id": workflow_id,
                "repository_id": repository_id,
                "workstream_id": workstream_id,
                "code_completion": code_completion.model_dump(mode="json"),
                "review": review.model_dump(mode="json"),
            }
        commit_sha = await git_service.commit(
            workspace,
            f"workflow {workflow_id}: {summary}",
            files=paths,
            expected_content_fingerprint=content_fingerprint,
            publication_receipt=publication_receipt,
        )
        await self._cancellation_token.raise_if_cancelled()
        await git_service.push(workspace, branch_name, expected_commit_sha=commit_sha)
        return code_completion.model_copy(
            update={
                "commit_sha": commit_sha,
                "metadata": {
                    **code_completion.metadata,
                    "published_after_approval": True,
                    "reviewed_content_fingerprint": content_fingerprint,
                },
            }
        )

    async def _reset_workspace(self, workspace: Path) -> str | None:
        """Capture the rejected attempt, then return the checkout to its committed state.

        The reset path, not the ordinary retry: an ordinary retry keeps the previous
        attempt's files and edits them in place. This runs only when the checkout is
        unusable or the retry policy classified the previous attempt as not worth building
        on, and the capture is then the only remaining record of what that attempt wrote.

        A commit an earlier attempt already pushed is kept: it is on the remote, and
        rewriting published history is worse than building the next attempt on top of it.
        Ignored paths are kept too, so an installed dependency tree is not thrown away.
        """
        captured = await self._capture_attempt(workspace)
        await self._reset_workspace_tree(workspace)
        return captured

    async def _wholesale_rewrite_issues(self, workspace: Path) -> list[str]:
        """Report existing files this attempt replaced instead of edited.

        The engineer returns whole file contents, so an attempt that does not read a file
        carefully can answer a small requirement by rewriting the file around it. One did
        exactly that to a homepage: it kept the tile it was asked to add, dropped the two
        hundred and seventy lines of tiles already there, and left a comment saying the rest
        would normally be rendered. Every gate passed, because each one asked whether the new
        behavior was present and none asked whether the old behavior survived.
        """
        runner = self._process_runner or AsyncioProcessRunner()
        timeout = float(self._settings.default_agent_timeout_seconds)
        captured = await runner.run(
            # Against HEAD so the comparison holds whether or not an earlier step staged the
            # work. The positive "." is required: a pathspec of exclusions alone matches nothing.
            ("git", "diff", "HEAD", "--numstat", "--", ".", f":(exclude){_CONTRACT_PROJECTION}"),
            workspace,
            timeout,
            self._cancellation_token,
            repository_subprocess_environment(),
        )
        if captured.return_code != 0:
            # Never block a change because the inspection itself could not run; the reviewer
            # and the repository's own validation still stand between this and a commit.
            return []
        issues: list[str] = []
        for line in captured.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit():
                # Binary files report "-" for both counts and carry no line evidence.
                continue
            added, deleted, path = int(fields[0]), int(fields[1]), fields[2]
            if deleted < _REWRITE_MINIMUM_DELETED_LINES:
                continue
            if deleted < _REWRITE_DELETED_TO_ADDED_RATIO * (added + 1):
                continue
            issues.append(
                f"{path} was rewritten rather than edited: this change removes {deleted} of "
                f"its existing lines and adds {added}. Preserve the behavior already in that "
                "file and add the required change alongside it, or explain in your summary "
                "why removing it is part of the requirement."
            )
        return issues

    async def _unreachable_addition_issues(
        self, workspace: Path, *, assigned_paths: tuple[str, ...] = ()
    ) -> list[str]:
        """Report new production code that nothing in the repository refers to.

        The authoritative wiring gate, and the reason it stays here: the Engineer now runs
        the same inspection inside its own attempt as fast feedback, and an agent that both
        writes the code and rules on whether it is reachable has no gate at all. So this asks
        again, from scratch, over the workspace as it finally stands. It consults nothing the
        Engineer decided and is not skipped by the Engineer's own check having passed.

        The inspection itself lives in ``tools.reachability`` so both layers ask the identical
        question of the identical checkout. What that module reports, why a convention-scanned
        directory is silent, and how a symbol is resolved are all documented there.

        ``assigned_paths`` is the plan's own ``expected_files_or_areas`` for this workstream:
        the only input in the whole inspection that knows what the change is *for*.
        """
        outcome = await unreachable_addition_issues(
            workspace,
            runner=self._process_runner or AsyncioProcessRunner(),
            timeout=float(self._settings.default_agent_timeout_seconds),
            cancellation_token=self._cancellation_token,
            assigned_paths=assigned_paths,
            excluded_paths=(_CONTRACT_PROJECTION,),
        )
        self._wiring_repairs = [dict(repair) for repair in outcome.repairs]
        self._textual_wirings = [dict(item) for item in outcome.textual_wirings]
        if self._textual_wirings:
            # Recorded, never blocked on: a path in a route table or a settings module is
            # legitimate wiring in many repositories. 216's string-literal "wiring" of
            # server/conftest.py is the shape this makes visible.
            _LOGGER.info(
                "wiring_reference_textual_only",
                extra={"wirings": self._textual_wirings},
            )
        return list(outcome.issues)

    async def _clear_stale_index_lock(self, workspace: Path) -> None:
        """Remove an index lock a killed attempt left behind, before the next Git write.

        Safe only because of where it is called. This runs at the top of a retry, before
        this attempt has issued a single Git command, and one workspace is worked on by one
        attempt at a time -- so a lock present here was left by a process that is already
        gone. It is not safe to call from anywhere that Git may be running concurrently, and
        it deliberately takes no timeout or force argument that would invite that.

        Never fatal in itself. If the lock cannot be removed, the staging command below
        fails and says so with Git's own words; guessing here would replace a precise error
        with a vaguer one.
        """
        lock = workspace / ".git" / "index.lock"
        try:
            existed = await asyncio.to_thread(_unlink_if_present, lock)
        except OSError as error:
            _LOGGER.warning(
                "child_index_lock_not_cleared",
                extra={"workspace": str(workspace), "error_type": type(error).__name__},
            )
            return
        if existed:
            _LOGGER.warning(
                "child_stale_index_lock_cleared",
                extra={"workspace": str(workspace)},
            )

    async def _late_reconnaissance_paths(
        self, *, feature_id: str, repository_id: str, workspace: Path
    ) -> list[str]:
        """Give a blind-planned repository its registry evidence back, from its own checkout.

        The structural half of reconnaissance only -- `inspect_repository_for_planning` is
        pure filesystem reading, no model call -- because `registry_paths` is precisely what
        the wiring gate's history says a blind repository's Engineer lacks: 24% of first
        attempts shipped unreachable code before 35- taught the prompt where things are
        registered. Bounded like every probe, and fail-soft like reconnaissance itself: a
        second failure records the repository blind again and returns nothing, and it must
        never fail the attempt or spend a retry.
        """
        try:
            evidence = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: inspect_repository_for_planning(workspace, repository_id=repository_id)
                ),
                timeout=float(self._settings.default_agent_timeout_seconds),
            )
        except (CancellationRequested, asyncio.CancelledError):
            raise
        except Exception as error:  # noqa: BLE001 - the probe informs; it never gates
            reason = "; ".join(safe_error_diagnostics(error)) or "no diagnostics"
            _LOGGER.warning(
                "repository late reconnaissance failed; %s stays blind: %s (%s) "
                "[feature=%s stage=late_reconnaissance agent=repository_recon]",
                repository_id,
                type(error).__name__,
                reason,
                feature_id,
            )
            if self._event_writer is not None:
                await self._event_writer(
                    feature_id,
                    "repository_planned_blind",
                    {
                        "repository_id": repository_id,
                        "error_type": type(error).__name__,
                        "reason": reason,
                        "occurrence": "late_reconnaissance",
                    },
                )
            return []
        paths = [module.path for module in evidence.wiring_files]
        _LOGGER.info(
            "repository late reconnaissance recovered %d registry path(s) for %s "
            "[feature=%s stage=late_reconnaissance]",
            len(paths),
            repository_id,
            feature_id,
        )
        return paths

    async def _usable_checkout(self, workspace: Path) -> bool:
        """Report whether this attempt has a checkout Git can still answer questions about.

        Two conditions, and the split between them is the point. The directory has to exist,
        which is checked here rather than asked of a subprocess: spawning a process with a
        working directory that is not there is what raised AB-Feature-190's
        `FileNotFoundError` out of `asyncio`'s own machinery, where no `except` in this file
        was waiting for it.

        Everything past that is Git's question, asked of Git through this composition's own
        runner -- `git rev-parse --git-dir`, the same "can this directory answer basic Git
        commands" test the reset path already keys off, asked before anything writes rather
        than after a staging command has failed. Deliberately not a filesystem probe for
        `.git`: this decision replaces a checkout, and a directory whose Git state is
        supplied by something other than a `.git` entry on this disk -- which every
        fixture-driven test in this repository is -- must not be destroyed by a guard
        looking for one.

        Read-only, for the same reason: a probe with side effects would make the decision to
        replace a checkout partly on its own behaviour.
        """
        if not workspace.is_dir():
            return False
        runner = self._process_runner or AsyncioProcessRunner()
        result = await runner.run(
            ("git", "rev-parse", "--git-dir"),
            workspace,
            float(self._settings.default_agent_timeout_seconds),
            self._cancellation_token,
            repository_subprocess_environment(),
        )
        return bool(result.succeeded)

    async def _provision_replacement_checkout(
        self,
        state: AgentState,
        *,
        git_service: InterruptibleGitService,
        feature: FeatureWorkflowSnapshot,
        repository: RepositorySpec,
        child: ChildWorkflowReference,
        workspace: Path,
    ) -> None:
        """Clone this attempt a checkout, exactly as attempt 0 would, and say so loudly.

        Logged at warning level with the attempt number, because a retry provisioning its
        own checkout means an earlier attempt of this workstream did not leave one -- a
        clone that never completed, or a directory removed underneath a live feature. Both
        are worth knowing about even though the attempt goes on to run normally.

        A directory that exists but holds no readable Git state is removed first. The
        provisioner refuses a destination that is already there unless it contains a `.git`
        -- the guard that lets a journal-reused clone recover a crashed worker -- and that
        refusal is right; what is being removed here is, by the check above, a directory
        with no recoverable Git state in it. The path is the child's own workspace and was
        confirmed to be under the configured root at the top of `run`.
        """
        _LOGGER.warning(
            "child_retry_workspace_absent",
            extra={
                "feature_id": feature.feature_id,
                "repository_id": repository.repository_id,
                "child_workflow_id": child.child_workflow_id,
                "attempt": child.retry_count,
                "workspace": str(workspace),
            },
        )
        _require_workspace_child(workspace, self._settings.workspace_root)
        if workspace.exists():
            await asyncio.to_thread(shutil.rmtree, workspace)
        await self._provisioner.provision_async(state, git_service=git_service)

    async def _capture_and_reset(self, workspace: Path) -> str | None:
        """Capture the previous attempt's diff, then restore the committed checkout."""
        return await self._reset_workspace(workspace)

    async def _capture_attempt(self, workspace: Path) -> str | None:
        """Capture the previous attempt's diff without touching the working tree.

        Staging and un-staging rather than one `git add` with an exclusion pathspec. The
        exclusion form failed in two ways this method can meet, both reproduced against real
        Git rather than reasoned about:

        `git add -A -- :(exclude)openapi.yaml` exits 1 in a repository that gitignores the
        contract projection. Naming a path in an `add` pathspec is what triggers Git's
        "the following paths are ignored" refusal, and excluding it counts as naming it --
        adding the positive `.` does not help. Every retry in such a repository ended the
        feature, and the repository was doing nothing wrong.

        The same command exits 128 -- fatal -- when a `.git/index.lock` is left behind. This
        platform manufactures exactly that: it terminates process groups on cancellation and
        on worker death, so an attempt killed mid-`git` leaves the lock, and this method's
        staging command is the first Git write the next attempt makes. That is cleared as a
        precondition below.

        Staging everything and then resetting the one path is equivalent for every shape
        tested -- a repository that tracks the projection, one that does not contain it, one
        that gitignores it, and a fresh checkout with nothing to stage -- and names no
        ignored path to Git in the process.
        """
        runner = self._process_runner or AsyncioProcessRunner()
        timeout = float(self._settings.default_agent_timeout_seconds)
        await self._clear_stale_index_lock(workspace)
        # Staged so that files the previous attempt created appear in the diff at all.
        staged = await runner.run(
            ("git", "add", "-A"),
            workspace,
            timeout,
            self._cancellation_token,
            repository_subprocess_environment(),
        )
        _require_workspace_command(staged, stage="retry staging")
        # The contract projection is unstaged again because the platform wrote it, not the
        # model, so it is not part of what the previous attempt should be shown having done.
        unstaged = await runner.run(
            ("git", "reset", "--quiet", "--", _CONTRACT_PROJECTION),
            workspace,
            timeout,
            self._cancellation_token,
            repository_subprocess_environment(),
        )
        _require_workspace_command(unstaged, stage="retry projection unstage")
        captured = await runner.run(
            ("git", "diff", "--cached"),
            workspace,
            timeout,
            self._cancellation_token,
            repository_subprocess_environment(),
        )
        _require_workspace_command(captured, stage="retry diff capture")
        # The capture staged everything so new files appear in the diff at all. Un-stage
        # again so the index leaves this method exactly as the attempt left it: the commit
        # adapter refuses pre-existing staged changes as a safety boundary, and a capture
        # on the preserved path must not manufacture that condition.
        # The pathspec form, deliberately: a bare `git reset` moves nothing but still writes
        # a `reset: moving to HEAD` reflog entry, and the reflog is the audit trail that says
        # whether an attempt was destroyed. A capture must leave no such fingerprint.
        unstaged_all = await runner.run(
            ("git", "reset", "--quiet", "--", "."),
            workspace,
            timeout,
            self._cancellation_token,
            repository_subprocess_environment(),
        )
        _require_workspace_command(unstaged_all, stage="retry capture unstage")
        # ProcessRunner implementations are injectable and therefore are not all guaranteed
        # to apply AsyncioProcessRunner's output policy. Redact again at the prompt boundary.
        # No character slice here: whatever bounding a prompt needs happens at whole-file
        # boundaries in `_bounded_previous_attempt`, never as a blind cut inside a hunk.
        return redact_source_credentials(redact_output(captured.stdout)).strip() or None

    async def _reset_workspace_tree(self, workspace: Path) -> None:
        """Return the checkout to its committed state, keeping installed toolchains."""
        runner = self._process_runner or AsyncioProcessRunner()
        timeout = float(self._settings.default_agent_timeout_seconds)
        for command in (
            ("git", "reset", "--hard", "HEAD"),
            # -x with explicit exceptions rather than omitting it. Build output is gitignored
            # too, so keeping every ignored path left a compiled tree behind: the next
            # attempt's project-wide lint then walked tens of megabytes of generated bundles
            # and timed out. What is preserved is whatever this checkout's detected runtime
            # installs into it -- expensive, reusable, and not reinstalled on a retry.
            ("git", "clean", "--force", "-d", "-x", "--", *_cleanable_paths(workspace)),
        ):
            await self._cancellation_token.raise_if_cancelled()
            result = await runner.run(
                command,
                workspace,
                timeout,
                self._cancellation_token,
                repository_subprocess_environment(),
            )
            _require_workspace_command(result, stage="retry workspace reset")

    async def _apply_approved_repair(
        self,
        feature: FeatureWorkflowSnapshot,
        *,
        repository_id: str,
        workspace: Path,
        operation_executor: ExternalOperationExecutor,
    ) -> list[str]:
        """Run an approved repository repair in this checkout, and say what it ran.

        Only a repair somebody authorized. A proposal is a diagnosis waiting for a decision,
        and applying one nobody made would be the platform changing a repository on its own --
        the exact thing the whole concept exists to prevent.

        Each command goes through the operation journal like every other side effect, so
        approving twice cannot install twice: the second attempt replays the recorded result
        instead of running the command again. The commands themselves were built by the
        platform from validated package names, never by a model, and run without a shell.
        """
        repair = approved_repair_for(feature.artifacts, repository_id)
        if repair is None or not repair.commands:
            return []
        applied: list[str] = []
        for index, step in enumerate(repair.commands):
            command = tuple(step.command)

            # Built here rather than defaulted at the call: the executor's runner is
            # optional, and a repair must run against a real one rather than silently doing
            # nothing when a caller left it out.
            runner = self._process_runner or AsyncioProcessRunner()

            async def run_step(
                command: tuple[str, ...] = command,
                runner: ProcessRunner = runner,
            ) -> tuple[str, OperationResult]:
                result = await runner.run(
                    command,
                    workspace,
                    timeout_seconds=float(self._settings.commit_gate_timeout_seconds),
                    cancellation_token=self._cancellation_token,
                    environment=repository_subprocess_environment(),
                )
                if not result.succeeded:
                    # Raised rather than returned so the journal records a failure. A repair
                    # whose command did not run is not a repair, and the attempt that follows
                    # would otherwise look like the repair simply not having worked.
                    msg = "an approved repository repair command did not succeed"
                    raise RepositoryRepairFailed(
                        msg,
                        # A platform-owned classification, not the tool's output: a package
                        # manager's stderr quotes registry URLs and repository paths.
                        code=subprocess_result_code(
                            return_code=result.return_code,
                            timed_out=result.timed_out,
                            cancelled=result.cancelled,
                            prefix="REPAIR",
                        ),
                    )
                return " ".join(command), OperationResult(payload={"command": list(command)})

            await operation_executor.run(
                operation_type=ExternalOperationType.INSTALL_DEPENDENCIES,
                logical_step=f"apply_repository_repair:{repair.repair_id}:{index}",
                safe_input={
                    "workspace_path": str(workspace),
                    "repair_id": repair.repair_id,
                    # The command is platform-built from validated names, so it is safe to
                    # record; it is also the only durable evidence of what was changed.
                    "command": list(command),
                    "repository_revision": repair.proposed_at_revision,
                },
                action=run_step,
            )
            applied.append(" ".join(command))
        _LOGGER.info(
            "applied an approved repository repair",
            extra={
                "feature_id": feature.feature_id,
                "repository_id": repository_id,
                "repair_id": repair.repair_id,
                "commands": len(applied),
            },
        )
        return applied

    async def _write_contract_projection(
        self,
        workspace: Path,
        contract: IntegrationContractArtifact,
        *,
        operation_executor: ExternalOperationExecutor,
    ) -> str | None:
        """Write a REST projection before coding so modifications become change requests."""
        content = await self._contract_projection(contract)
        if content is not None:

            async def write() -> tuple[str, OperationResult]:
                WorkspaceFileTools(
                    workspace, max_file_bytes=self._settings.max_workspace_file_bytes
                ).write_file("openapi.yaml", content)
                return content, OperationResult(payload={"path": "openapi.yaml"})

            await operation_executor.run(
                operation_type=ExternalOperationType.WRITE_FILE_CHANGES,
                logical_step=CONTRACT_PROJECTION_LOGICAL_STEP,
                safe_input={
                    "workspace_path": str(workspace),
                    "path": _CONTRACT_PROJECTION,
                    "contract_version": contract.contract_version,
                },
                action=write,
            )
            # Written again outside the journal because that write is idempotent: on a retry
            # it replays the recorded success without executing, so a projection the previous
            # attempt edited would stay edited and the next attempt would be accused of a
            # contract change it never made. The journal keeps the audit record; this keeps
            # the file itself true to the approved contract.
            WorkspaceFileTools(
                workspace, max_file_bytes=self._settings.max_workspace_file_bytes
            ).write_file(_CONTRACT_PROJECTION, content)
        return content

    async def _contract_projection(self, contract: IntegrationContractArtifact) -> str | None:
        """Return deterministic YAML when the approved contract exposes REST operations."""
        document = contract.openapi_document or await self._contract_generator.generate_openapi(
            contract
        )
        return yaml.safe_dump(document, sort_keys=True) if document is not None else None


class ProductionFeatureRunner:
    """Construct fresh live parent/child agents for each request-scoped feature operation."""

    def __init__(
        self,
        settings: Settings,
        *,
        database: Database | None = None,
        redis_client: Any | None = None,
        operation_journal: ExternalOperationJournal | None = None,
    ) -> None:
        """Retain only static deployment configuration; never retain provider tokens."""
        self._settings = settings
        self._database = database
        self._redis_client = redis_client
        self._operation_journal = operation_journal
        self._checkpoint_writer: Any | None = None
        self._event_writer: FeatureEventWriter | None = None
        # Bound after composition, like the two writers above, and for a stronger reason: the
        # design resolver opens the configured owner's `figma` credential, so it is built where
        # the secret store lives rather than here. That keeps the token out of
        # `RequestScopedCredentials` -- which is passed through every layer of a feature -- and
        # out of every reader that has no business holding one.
        self._design_resolver: DesignReferenceResolver | None = None

    def bind_checkpoint_writer(self, writer: Any) -> None:
        """Bind the durable feature-store checkpoint callback after composition is complete."""
        self._checkpoint_writer = writer

    def bind_event_writer(self, writer: FeatureEventWriter) -> None:
        """Bind the durable feature-store timeline writer after composition is complete."""
        self._event_writer = writer

    def bind_design_resolver(self, resolver: DesignReferenceResolver) -> None:
        """Bind the live design resolver after composition is complete.

        Absent by default, which is the honest state of a deployment with no secret store: the
        orchestrator then falls back to its deterministic resolver, and a citation that reaches
        it is refused at submission long before that -- Part C will not accept a citation
        against a design source this deployment has not configured.
        """
        self._design_resolver = resolver

    async def _maintain_workspaces_for(self, state: FeatureWorkflowSnapshot) -> None:
        """Reclaim what the database proves is retired, and refuse a claim the volume cannot hold.

        Every mutation this runner exposes goes through here before it builds anything,
        because every one of them may clone or install.

        One exception, and it belongs to the publication invariant rather than to capacity:
        a feature that still owes somebody a pull request is let past the floor. Reviewed work
        always reaches a pull request -- the rule every other failure in this platform honors
        -- and disk space was the one exception nobody had decided to make. Run 193 is what it
        cost: both repositories APPROVED, the 2 GB preflight failed at 1.84 GB free, and the
        feature flipped `failed_requires_human` with no salvage pull requests, before the
        orchestrator that would have published them was ever entered. A manual
        policy-compliant cleanup freed 12 GB and the very next claim completed the feature
        end to end.

        Letting it past is not pretending there is room, and it is not one uniform claim
        either -- 81- added a second kind of publishable work and the two need different
        things from the volume:

        * work that passed review is already committed and pushed, so opening its pull
          request needs no workspace at all; and
        * work whose review rejected it was never committed, so publishing it commits from
          the checkout that still holds it. That needs the workspace to exist -- not to have
          room, which is a different question -- and `require_publishable_workstream` is what
          refuses, with a sentence, once retention has reclaimed it.

        So the yield is keyed on "would anything be published if somebody asked", which is
        the question both kinds answer. Keying it on approved-and-unpublished work alone -- as
        it read until 81- -- refused a feature whose only publishable work was clean but
        rejected, for disk space, while offering that publication in the UI.

        Anything downstream that genuinely needs room fails inside the orchestrator, which
        honors the invariant. What is removed is only this refusal's ability to strand work
        that has nowhere else to go.
        """
        try:
            await _maintain_feature_workspaces(
                self._settings.workspace_root,
                database=self._database,
                keep=self._settings.workspace_retention_features,
                minimum_free_bytes=self._settings.workspace_minimum_free_bytes,
                failed_retention_hours=self._settings.workspace_failed_retention_hours,
                failed_retention_features=self._settings.workspace_failed_retention_features,
            )
        except WorkspaceCapacityError as error:
            if not feature_has_publishable_work(state):
                raise
            _LOGGER.warning(
                "workspace_capacity_yielded_to_publication",
                extra={
                    "feature_id": state.feature_id,
                    "free_bytes": error.free_bytes,
                    "required_bytes": error.required_bytes,
                },
            )

    async def start(
        self, state: FeatureWorkflowSnapshot, *, credentials: RequestScopedCredentials
    ) -> FeatureWorkflowSnapshot:
        """Run the live feature workflow with credentials that end when this request returns."""
        await self._maintain_workspaces_for(state)
        with self._orchestrator(state, credentials) as orchestrator:
            _apply_feature_limits(state, self._settings)
            return await orchestrator.start(state, credentials=credentials)

    async def advance_one_step(
        self, state: FeatureWorkflowSnapshot, *, credentials: RequestScopedCredentials
    ) -> FeatureWorkflowSnapshot:
        """Advance this feature by one step against freshly built live dependencies.

        The whole live composition -- workspaces, provider clients, the process runner -- is
        built for one step and torn down when it ends, exactly as it was for a whole feature.
        That is what makes a claim's credentials last no longer than the claim.
        """
        await self._maintain_workspaces_for(state)
        with self._orchestrator(state, credentials) as orchestrator:
            _apply_feature_limits(state, self._settings)
            return await orchestrator.advance_one_step(state, credentials=credentials)

    async def begin_resume(
        self, state: FeatureWorkflowSnapshot, *, answers: Sequence[ClarificationAnswer]
    ) -> FeatureWorkflowSnapshot:
        """Settle what a resume carries and refuses, with no workspace and no provider.

        Deliberately outside `_orchestrator`: nothing here clones, installs or calls a model,
        so building the live composition for it would maintain workspaces to write a
        clarification answer into an artifact.
        """
        orchestrator = FeatureWorkflowOrchestrator(checkpoint_writer=self._checkpoint_writer)
        _apply_feature_limits(state, self._settings)
        return await orchestrator.begin_resume(state, answers=answers)

    async def grant_and_run_one_retry(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repository_id: str,
        additional_attempts: int,
        requested_by: str,
        reason: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Record a granted retry and run the one repository it bought, then stop."""
        await self._maintain_workspaces_for(state)
        with self._orchestrator(state, credentials) as orchestrator:
            _apply_feature_limits(state, self._settings)
            return await orchestrator.grant_and_run_one_retry(
                state,
                repository_id=repository_id,
                additional_attempts=additional_attempts,
                requested_by=requested_by,
                reason=reason,
                credentials=credentials,
            )

    async def answer_design_conflict(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        conflict_id: str,
        verdict: str,
        decision: str,
        decided_by: str,
        additional_attempts: int,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Record a design decision and run the repository it unblocks, with live dependencies.

        The attempt a verdict authorises clones, codes, validates and pushes exactly as any
        other attempt does, so it needs the same workspace maintenance and the same live
        orchestrator a granted retry does.
        """
        await self._maintain_workspaces_for(state)
        with self._orchestrator(state, credentials) as orchestrator:
            _apply_feature_limits(state, self._settings)
            return await orchestrator.answer_design_conflict(
                state,
                conflict_id=conflict_id,
                verdict=verdict,
                decision=decision,
                decided_by=decided_by,
                additional_attempts=additional_attempts,
                credentials=credentials,
            )

    async def resume(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        answers: Sequence[ClarificationAnswer],
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Resume a parent pause with newly supplied tokens and freshly built live dependencies."""
        await self._maintain_workspaces_for(state)
        with self._orchestrator(state, credentials) as orchestrator:
            _apply_feature_limits(state, self._settings)
            return await orchestrator.resume(state, answers=answers, credentials=credentials)

    async def retry_workstream(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repository_id: str,
        additional_attempts: int,
        requested_by: str,
        reason: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Run one granted repository again with freshly built live dependencies.

        A granted attempt clones, codes, validates and pushes exactly as any other attempt
        does, so it needs the same workspace maintenance and the same live orchestrator.
        """
        await self._maintain_workspaces_for(state)
        with self._orchestrator(state, credentials) as orchestrator:
            _apply_feature_limits(state, self._settings)
            return await orchestrator.retry_workstream(
                state,
                repository_id=repository_id,
                additional_attempts=additional_attempts,
                requested_by=requested_by,
                reason=reason,
                credentials=credentials,
            )

    async def publish_feature(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        requested_by: str,
        reason: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Open what a person decided this feature should publish, with live dependencies.

        Workspaces are maintained first like every other mutation here, and for once the
        capacity yield inside `_maintain_workspaces_for` is exactly the case it was written
        for -- see its docstring for the two kinds of publishable work and what each needs
        from the volume.

        The checkout is the part with a deadline. Work whose review rejected it was never
        committed, so publishing it commits from the tree that still holds it, and a held
        feature is `failed_requires_human` -- bounded by
        `workspace_failed_retention_hours` (72 by default) *or*
        `workspace_failed_retention_features` (10), either being enough to reclaim. Not
        `workspace_retention_features`, which governs completed and cancelled features only.
        Roughly three days or ten failed features, then, and past that
        `require_publishable_workstream` refuses with a sentence saying so.
        """
        await self._maintain_workspaces_for(state)
        with self._orchestrator(state, credentials) as orchestrator:
            _apply_feature_limits(state, self._settings)
            return await orchestrator.publish_feature(
                state,
                requested_by=requested_by,
                reason=reason,
                credentials=credentials,
            )

    async def approve_contract_change(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        request_id: str,
        updated_contract: IntegrationContractArtifact,
        resolution: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Run only affected live children after an explicitly supplied contract revision."""
        await self._maintain_workspaces_for(state)
        with self._orchestrator(state, credentials) as orchestrator:
            _apply_feature_limits(state, self._settings)
            return await orchestrator.approve_contract_change(
                state,
                request_id=request_id,
                updated_contract=updated_contract,
                resolution=resolution,
                credentials=credentials,
            )

    async def reject_contract_change(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        request_id: str,
        resolution: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Record a human contract rejection with no network provider action."""
        with self._orchestrator(state, credentials) as orchestrator:
            return await orchestrator.reject_contract_change(
                state,
                request_id=request_id,
                resolution=resolution,
                credentials=credentials,
            )

    async def approve_repository_repair(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repair_id: str,
        actor_id: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Apply an approved repair, which runs a real attempt against a fresh checkout."""
        await self._maintain_workspaces_for(state)
        with self._orchestrator(state, credentials) as orchestrator:
            _apply_feature_limits(state, self._settings)
            return await orchestrator.approve_repository_repair(
                state,
                repair_id=repair_id,
                actor_id=actor_id,
                credentials=credentials,
            )

    async def reject_repository_repair(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repair_id: str,
        actor_id: str,
        reason: str,
    ) -> FeatureWorkflowSnapshot:
        """Record a declined repair. Nothing is cloned and no provider is reached."""
        with self._orchestrator(state, RequestScopedCredentials(None, None)) as orchestrator:
            return await orchestrator.reject_repository_repair(
                state,
                repair_id=repair_id,
                actor_id=actor_id,
                reason=reason,
            )

    def _orchestrator(
        self, state: FeatureWorkflowSnapshot, credentials: RequestScopedCredentials
    ) -> _LiveOrchestratorContext:
        """Return a context manager that removes the temporary Git askpass helper afterward."""
        if self._database is None or self._redis_client is None or self._operation_journal is None:
            msg = "live feature execution requires durable database, Redis, and operation journal"
            raise RuntimeConfigurationError(msg)
        database_token = DatabaseCancellationToken(
            self._database, workflow_id=state.workflow_id, feature_id=state.feature_id
        )
        redis_token = RedisCancellationToken(
            self._redis_client, scope="feature", identifier=state.feature_id
        )
        return _LiveOrchestratorContext(
            self._settings,
            credentials,
            state=state,
            cancellation_token=CompositeCancellationToken((database_token, redis_token)),
            operation_journal=self._operation_journal,
            checkpoint_writer=self._checkpoint_writer,
            event_writer=self._event_writer,
            # Where the product manager reads a submission's images. Built from the same
            # database the rest of this runner uses; the agent is handed a source narrowed
            # to `get_content`, so it can read bytes and do nothing else with the store.
            attachments=DatabaseAttachmentStore(self._database),
            design_resolver=self._design_resolver,
        )


# Which header carries each platform's key. Named rather than derived so the message a
# misconfigured deployment reads at three in the morning says exactly which of two credentials
# is missing, and so a header this platform reads is always greppable.
_PLATFORM_KEY_HEADERS: Mapping[AgentPlatform, str] = {
    AgentPlatform.OPENAI: "X-OpenAI-Api-Key",
    AgentPlatform.ANTHROPIC: "X-Anthropic-Api-Key",
}


def _platform_api_key(credentials: RequestScopedCredentials, platform: AgentPlatform) -> str | None:
    """Return the model credential one platform needs, and never the other platform's.

    A feature on Claude that happened to have an OpenAI key stored must fail as a missing
    Anthropic credential, not authenticate against the wrong provider.
    """
    if platform is AgentPlatform.ANTHROPIC:
        return credentials.anthropic_api_key
    return credentials.openai_api_key


class _LiveOrchestratorContext:
    """Own request-scoped LLM/GitHub clients and the temporary Git askpass environment."""

    def __init__(
        self,
        settings: Settings,
        credentials: RequestScopedCredentials,
        *,
        state: FeatureWorkflowSnapshot,
        cancellation_token: CancellationToken,
        operation_journal: ExternalOperationJournal,
        checkpoint_writer: Any | None,
        event_writer: FeatureEventWriter | None = None,
        attachments: AttachmentContentSource | None = None,
        design_resolver: DesignReferenceResolver | None = None,
    ) -> None:
        self._settings = settings
        self._credentials = credentials
        self._state = state
        self._cancellation_token = cancellation_token
        self._operation_journal = operation_journal
        self._checkpoint_writer = checkpoint_writer
        self._event_writer = event_writer
        self._attachments = attachments
        # Passed in rather than built here: it opens the configured owner's `figma`
        # credential, and this context holds request-scoped provider clients built from a
        # `RequestScopedCredentials` that deliberately carries no design token.
        self._design_resolver = design_resolver
        self._context: Any | None = None

    def __enter__(self) -> FeatureWorkflowOrchestrator:
        # The feature's own platform, never the deployment's. A recovered run reads what this
        # feature was submitted on, so a deployment that has since been reconfigured cannot
        # move a feature onto a provider it did not start on.
        platform = AgentPlatform(self._state.agent_platform)
        # And the feature's own tier, for the same reason. The tier-scoped settings view is
        # how the tier feeds every model boundary and the router without any of them learning
        # a new question: `HIGH` is the settings object itself, so a feature that predates
        # tiers runs through byte-for-byte the path it always did. A custom feature resolves
        # through its pinned snapshot instead -- the same kind of clone, whose role-addressed
        # service re-validates the snapshot and refuses honestly rather than clamping.
        tier = PerformanceTier(self._state.performance_tier)
        if tier is PerformanceTier.CUSTOM:
            snapshot = self._state.model_setup_snapshot
            if snapshot is None:
                msg = (
                    "this feature is pinned to a custom model setup but its snapshot is "
                    "missing from state; it cannot resolve a model"
                )
                raise RuntimeConfigurationError(msg)
            tier_settings = self._settings.for_model_setup(snapshot)
        else:
            tier_settings = self._settings.for_performance_tier(tier)
        service = tier_settings.model_configs
        github_token = self._credentials.github_token
        # Every platform this feature's roles resolve on needs its key before anything runs.
        # For a tier feature that is the one pinned platform, refused in the sentence it
        # always was; a mixed custom setup legitimately needs more than one, and the refusal
        # names the roles that pinned the missing one -- a platform name alone sends somebody
        # looking for which of four rows caused it. G4 makes this near-unreachable, but a
        # credential can be deleted between creation and execution.
        for needed in (
            service.configured_platforms() if tier is PerformanceTier.CUSTOM else (platform,)
        ):
            if _platform_api_key(self._credentials, needed):
                continue
            pinned = ", ".join(
                role.value for role in ModelRole if service.platform_for_role(role) is needed
            )
            detail = f" ({pinned} pinned to {needed.value} by this feature's model setup)"
            msg = (
                f"{_PLATFORM_KEY_HEADERS[needed]} is required for live feature execution "
                f"on {needed.value}" + (detail if pinned else "")
            )
            raise RuntimeConfigurationError(msg)
        if not github_token:
            msg = "X-GitHub-Token is required for live feature repository operations"
            raise RuntimeConfigurationError(msg)
        self._context = _git_askpass_environment(github_token)
        git_environment = self._context.__enter__()
        # Which credential that environment carries, and when it was stored -- never the
        # value. Both Git boundaries below get it so a remote that refuses the token names
        # it. Absent for a token supplied as a request header, which has no stored date.
        git_credential = CredentialProvenance(
            provider="github", stored_at=self._credentials.github_token_stored_at
        )

        def client(agent_name: str, **overrides: Any) -> LLMClient:
            """Build one model boundary on the platform that answers this agent's role.

            For a tier feature every role answers on the feature's one platform, and this is
            byte for byte the closure it always was. A custom setup pins each role to its own
            provider, so the construction site -- here, not the adapters -- resolves the
            agent's role, asks the setup which platform answers it, and passes that platform
            and that platform's key to the existing factory.
            """
            role = overrides.get("model_role") or tier_settings.agents[agent_name].model_role
            role_platform = (
                service.platform_for_role(role) if role is not None else None
            ) or platform
            return llm_client_for(
                role_platform,
                tier_settings,
                agent_name,
                api_key=_platform_api_key(self._credentials, role_platform),
                **overrides,
            )

        product_manager = ProductManagerAgent(
            prompt_loader=_prompt_loader(),
            llm_client=client("product_manager"),
            attachments=self._attachments,
        )
        planner = FeaturePlannerAgent(
            prompt_loader=_prompt_loader(),
            llm_client=client("planner"),
            contract_generator=OpenAPIContractCodeGenerator(),
        )
        child_executor = LiveChildWorkstreamExecutor(
            settings=self._settings,
            # Bytes-by-id only, for placing the design's exported images into the checkout.
            # The narrow protocol rather than the store: this executor has no business
            # listing, binding, deleting or purging anything.
            attachments=self._attachments,
            git_environment=git_environment,
            credential=git_credential,
            engineer_client=client("engineer", model_role=ModelRole.CODING),
            reviewer_client=client("reviewer", model_role=ModelRole.REVIEW),
            journal=self._operation_journal,
            cancellation_token=self._cancellation_token,
            event_writer=self._event_writer,
            # One boundary per persisted decision, built on demand. The engineer's operational
            # policy still comes from agent configuration; provider, model and effort come from
            # the durable routing snapshot so recovery cannot silently switch any of them.
            engineer_client_factory=lambda decision: llm_client_for(
                decision.platform if decision is not None else platform,
                tier_settings,
                "engineer",
                api_key=_platform_api_key(
                    self._credentials,
                    decision.platform if decision is not None else platform,
                ),
                model_role=(decision.role if decision is not None else ModelRole.CODING),
                resolved_model=decision.model if decision is not None else None,
                resolved_reasoning_effort=(decision.reasoning if decision is not None else None),
                resolved_model_variable=(decision.model_variable if decision is not None else None),
                resolved_routing_reason=(decision.routing_reason if decision is not None else None),
            ),
            # The in-attempt repair boundary: the SCOPED_FIX role at this feature's pinned
            # platform and tier, resolved fresh per attempt like the engineer's own client.
            # Not the persisted routing decision -- that names the attempt's coding role;
            # the repair is a different job with its own configured selection. A custom setup
            # may pin the repair role to a different provider than the attempt's, so the
            # role's own platform outranks the decision's.
            scoped_fix_client_factory=lambda decision: _scoped_fix_repair_client(
                service.platform_for_role(ModelRole.SCOPED_FIX)
                or (decision.platform if decision is not None else platform),
                tier_settings,
                api_key=_platform_api_key(
                    self._credentials,
                    service.platform_for_role(ModelRole.SCOPED_FIX)
                    or (decision.platform if decision is not None else platform),
                ),
            ),
            # The in-attempt self-review boundary: the CODING role at this feature's pinned
            # platform and tier, whatever role the attempt itself was routed to -- a scoped
            # remediation still reviews its finished work on the coding model, and no tier
            # escalates silently because a review is about to run. Never the REVIEW role;
            # that boundary belongs to the independent reviewer that judges this afterwards.
            self_review_client_factory=lambda decision: _self_review_client(
                service.platform_for_role(ModelRole.CODING)
                or (decision.platform if decision is not None else platform),
                tier_settings,
                api_key=_platform_api_key(
                    self._credentials,
                    service.platform_for_role(ModelRole.CODING)
                    or (decision.platform if decision is not None else platform),
                ),
            ),
        )
        parent_operations = ExternalOperationExecutor(
            journal=self._operation_journal,
            cancellation_token=self._cancellation_token,
            scope=ExternalOperationScope(
                workflow_id=self._state.workflow_id,
                feature_id=self._state.feature_id,
            ),
        )
        return FeatureWorkflowOrchestrator(
            product_manager=LiveFeatureProductManager(
                product_manager, self._settings.workspace_root
            ),
            # Absent on a deployment with no secret store, where the orchestrator's own
            # deterministic resolver stands in and no citation can have been accepted anyway.
            design_resolver=self._design_resolver,
            reconnaissance=LiveRepositoryReconnaissance(
                settings=self._settings,
                git_environment=git_environment,
                credential=git_credential,
                recon_client=client("recon"),
                journal=self._operation_journal,
                cancellation_token=self._cancellation_token,
            ),
            planner=planner,
            child_executor=child_executor,
            # The only reviewer in the platform that sees more than one repository at once.
            # Child reviews are scoped to their own repository by design, so without this the
            # seam between them is checked by nobody.
            integration_reviewer=IntegrationReviewerAgent(
                contract_generator=OpenAPIContractCodeGenerator(),
                llm_client=client("reviewer", model_role=ModelRole.REVIEW),
                diff_provider=LiveCrossRepositoryDiffs(settings=self._settings),
                prompt_loader=_prompt_loader(),
            ),
            pull_request_publisher=GitHubPullRequestPublisher(
                github_service=JournaledGitHubService(
                    PyGithubService(github_token), operation_executor=parent_operations
                ),
                draft_pull_requests=self._settings.github_default_draft_pull_requests,
                cancellation_token=self._cancellation_token,
            ),
            workspace_root=self._settings.workspace_root,
            cancellation_token=self._cancellation_token,
            checkpoint_writer=self._checkpoint_writer,
            event_writer=self._event_writer,
            # The parent scope already names this request's feature; the factory shape
            # exists for compositions where one orchestrator serves several features.
            operation_executor_factory=lambda _state: parent_operations,
            max_parallel_workstreams=self._settings.max_parallel_workstreams,
            # The tier feeds the router through the scoped view; the router's decisions are
            # unchanged -- it still chooses which role answers, never what a role costs.
            model_router=ModelRouter(tier_settings.model_configs, platform=platform),
            repository_runtime_limit_seconds=self._settings.repository_runtime_limit_seconds,
            # The Git boundary for the one path that commits outside a child attempt:
            # publishing a workstream whose checks all passed and whose review rejected it.
            # Built per repository because the adapter refuses to push a default branch and
            # has to know which one that is, and journaled through the parent scope so the
            # commit and the push appear in the agent-work drawer like any other operation.
            git_service_factory=lambda repository, child: InterruptibleGitService(
                default_branch=repository.default_branch,
                base_branch=child.base_branch,
                environment=git_environment,
                cancellation_token=self._cancellation_token,
                operation_executor=ExternalOperationExecutor(
                    journal=self._operation_journal,
                    cancellation_token=self._cancellation_token,
                    scope=ExternalOperationScope(
                        workflow_id=self._state.workflow_id,
                        feature_id=self._state.feature_id,
                        child_workflow_id=child.child_workflow_id,
                        repository_id=repository.repository_id,
                        child_attempt=child.retry_count,
                    ),
                ),
                workspace_root=self._settings.workspace_root,
                credential=git_credential,
                commit_timeout_seconds=float(self._settings.commit_gate_timeout_seconds),
            ),
        )

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._context is not None:
            self._context.__exit__(exc_type, exc_value, traceback)


def _self_review_client(
    platform: AgentPlatform, settings: Settings, *, api_key: str | None
) -> LLMClient | None:
    """Resolve the CODING role's model boundary for the in-attempt self-review, or nothing.

    Nothing rather than an exception, on the same contract as the scoped-fix client below:
    the review is best-effort, and a deployment whose coding role cannot resolve for it must
    degrade to no review -- `completed` meaning what it meant before Part E -- never to an
    attempt lost over the review's own configuration.
    """
    try:
        return llm_client_for(
            platform, settings, "engineer", api_key=api_key, model_role=ModelRole.CODING
        )
    except (LLMAdapterError, ModelConfigurationError):
        _LOGGER.warning(
            "self-review client could not be resolved for %s; no self-review runs and the "
            "attempt completes on the deterministic checks alone "
            "[outcome=self_review_unresolvable]",
            platform.value,
        )
        return None


def _scoped_fix_repair_client(
    platform: AgentPlatform, settings: Settings, *, api_key: str | None
) -> LLMClient | None:
    """Resolve the SCOPED_FIX role's model boundary for the in-attempt repair, or nothing.

    Nothing rather than an exception, because the repair is best-effort by contract: a
    platform or tier whose scoped-fix role cannot resolve -- a recovered feature on a since-
    unconfigured provider, a half-migrated deployment -- must degrade to the primary coding
    executor repairing, never to an attempt lost over the repair's own configuration.
    """
    try:
        return llm_client_for(
            platform, settings, "engineer", api_key=api_key, model_role=ModelRole.SCOPED_FIX
        )
    except (LLMAdapterError, ModelConfigurationError):
        _LOGGER.warning(
            "scoped-fix repair client could not be resolved for %s; the primary coding "
            "executor repairs instead [outcome=scoped_fix_unresolvable]",
            platform.value,
        )
        return None


def _attempt_retry_plan(child: ChildWorkflowReference) -> dict[str, Any] | None:
    """Return the retry plan this attempt's task plan should carry.

    A context-partition scope is honoured only while its wave is actually in flight
    (``pending_context_partition`` non-empty). An approved wave's final result carries the
    scope it ran under -- honest forensics -- and the recording write copies that strategy
    back onto the child, so without this gate a later re-entry (an integration remediation,
    an operator retry) would silently run narrowed to the last cluster's files.
    """
    plan = child.retry_strategy
    if (
        isinstance(plan, dict)
        and "context_partition" in plan
        and not (child.pending_context_partition)
    ):
        plan = {key: value for key, value in plan.items() if key != "context_partition"}
    return plan or (
        {
            "attempt": child.retry_count,
            "reason": "resume a pre-review child attempt",
        }
        if child.retry_count > 0
        else None
    )


def _context_partition_scope(retry_strategy: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the context-partition cluster scope stamped on this attempt's strategy (80-).

    The workflow's partition wave writes it into the durable ``retry_strategy`` before each
    scoped attempt, which is the one channel that already flows from child state through the
    task plan into the engineer's execution context. ``final`` says whether this cluster is
    the wave's last -- the only one the review runs after.
    """
    if not isinstance(retry_strategy, dict):
        return None
    partition = retry_strategy.get("context_partition")
    return partition if isinstance(partition, dict) else None


def _child_task_plan(
    *,
    child: ChildWorkflowReference,
    technical_prd: TechnicalPRDArtifact,
    contract: IntegrationContractArtifact,
    workstream: RepositoryWorkstreamPlan,
    feedback: Sequence[str],
    registry_paths: Sequence[str] = (),
    model_routing: ModelRoutingDecision | None = None,
) -> TaskPlanArtifact:
    """Translate one feature workstream into the existing Engineer Agent's task-plan artifact."""
    tasks = [
        {
            "task_id": task_id,
            "title": responsibility,
            "description": " ".join(
                [
                    responsibility,
                    f"Use approved contract version {contract.contract_version}.",
                    (
                        "Implement sections: "
                        f"{', '.join(workstream.contract_sections_implemented) or 'none'}."
                    ),
                    (
                        "Consume sections: "
                        f"{', '.join(workstream.contract_sections_consumed) or 'none'}."
                    ),
                    *feedback,
                ]
            ),
            "owner": "engineer",
            "priority": "high",
            "dependencies": [],
            "acceptance_criteria": workstream.acceptance_criteria,
            "estimated_effort_points": 1,
        }
        for task_id, responsibility in zip(
            workstream.task_ids, workstream.responsibilities, strict=False
        )
    ]
    if not tasks:
        msg = "repository workstream requires at least one task"
        raise RuntimeConfigurationError(msg)
    return create_artifact(
        TaskPlanArtifact,
        workflow_id=child.child_workflow_id,
        artifact_id=ARTIFACT_FILENAMES["task_plan"],
        producer="feature_planner",
        payload={
            "summary": (
                f"{workstream.workstream_id} workstream under contract {contract.contract_version}."
            ),
            "tasks": tasks,
            "milestones": [
                {
                    "milestone_id": "implementation",
                    "title": "Repository implementation",
                    "objective": "Complete the assigned contract-scoped workstream.",
                    "task_ids": [item["task_id"] for item in tasks],
                }
            ],
            "implementation_order": [item["task_id"] for item in tasks],
            "test_strategy": workstream.test_requirements,
            "risk_management_plan": [
                "Do not change the approved integration contract.",
                *feedback,
            ],
        },
        metadata={
            "source_artifact_ids": [technical_prd.artifact_id, contract.artifact_id],
            "contract_version": contract.contract_version,
            "workstream_id": workstream.workstream_id,
            "attempt": child.retry_count,
            # Distinguishes an unsettled post-Engineer crash from a completed publication
            # that integration review has legitimately sent back for another correction.
            "settled_pull_request_artifact_id": child.pull_request_artifact_id,
            "repository_revision": (
                child.preflight_result.get("revision")
                if isinstance(child.preflight_result, dict)
                else None
            ),
            "preflight_result": child.preflight_result,
            "registry_paths": list(registry_paths),
            # What this attempt is and which role is executing it. Carried in the plan so the
            # Engineer's own prompt states it -- a review remediation is a different job from
            # an initial implementation and has to be told so -- and so the journal's effective
            # input covers the classified task. The configured model is excluded, so changing
            # deployment configuration alone cannot evade a completed operation receipt.
            #
            # Deliberately not the whole decision. A task plan is prompt material, and the
            # configured model name and the fingerprints are the
            # platform's bookkeeping: putting the model's own identifier in front of it invites
            # a response that reasons about being the cheap choice instead of about the finding.
            # The complete decision is persisted on the code completion, the child and the
            # child result, none of which are sent to a model.
            "model_routing": _task_plan_routing(model_routing),
            "retry_plan": _attempt_retry_plan(child),
            **(
                {
                    "review_scope": {
                        "repository_id": workstream.repository_id,
                        "workstream_id": workstream.workstream_id,
                        "role": workstream.role,
                        "requirement_ids": workstream.requirement_ids,
                        "scoped_requirements": [
                            item.model_dump(mode="json") for item in workstream.scoped_requirements
                        ],
                        "out_of_scope_requirements": workstream.out_of_scope_requirements,
                        "shared_requirements": [
                            item.model_dump(mode="json") for item in workstream.shared_requirements
                        ],
                        "responsibilities": workstream.responsibilities,
                        "acceptance_criteria": workstream.acceptance_criteria,
                        "test_requirements": workstream.test_requirements,
                        "contract_sections_consumed": workstream.contract_sections_consumed,
                        "contract_sections_implemented": workstream.contract_sections_implemented,
                        "design_nodes": workstream.design_nodes,
                        "expected_files_or_areas": workstream.expected_files_or_areas,
                        "implementation_expectations": [
                            item.model_dump(mode="json")
                            for item in workstream.implementation_expectations
                        ],
                    }
                }
                if workstream.requirement_ids or workstream.scoped_requirements
                else {}
            ),
        },
    )


def _task_plan_routing(routing: ModelRoutingDecision | None) -> dict[str, Any] | None:
    """Reduce a routing decision to what belongs in an Engineer's own instructions.

    Enough that the job and its classified repair scope are unambiguous. Nothing about which
    model was configured, which is neither the Engineer's business nor part of the effective
    task identity used for operation reconciliation.
    """
    if routing is None:
        return None
    return {
        "execution_mode": routing.execution_mode.value,
        "role": routing.role.value,
        "classification": (
            routing.classification.value if routing.classification is not None else None
        ),
        "failure_classification": (
            routing.failure_classification.value
            if routing.failure_classification is not None
            else None
        ),
        "repair_scope": routing.repair_scope.value if routing.repair_scope is not None else None,
        "attempt": routing.attempt,
    }


def _reviewed_source_fingerprint(
    workspace: Path, changed_files: Sequence[str], fallback: str
) -> str:
    """Fingerprint the reviewed source bytes, not merely the paths that were touched.

    The completeness fallback hashes the sorted path list, so an attempt that rewrote the
    same files to answer a review finding produced the previous attempt's fingerprint and was
    read as a repeat. -067's frontend was ended by that rule while it was one missing import
    from passing lint, having changed a different file's contents on every attempt. The
    schema documents this field as covering path *and* file bytes; this makes it so.
    """
    if not changed_files:
        return fallback
    try:
        return reviewed_content_fingerprint(workspace, changed_files)
    except (GitSafetyError, OSError, ValueError):
        return fallback


def _registry_paths(feature: FeatureWorkflowSnapshot, repository_id: str) -> list[str]:
    """Return the files this repository registers new modules in, as reconnaissance found them.

    Reconnaissance already establishes this per convention, grounded in paths it actually
    read. Nothing showed it to the engineer, so the first attempt had to infer the registry
    from a snapshot ranked by how much a path's name looks like the task: `app.service.js`
    ranks for "bulk delete app" and `routes/adunits/adunit.route.js` does not, though the
    second is where this repository mounts validators. The gate then spent an attempt
    delivering the path that was known before coding began.
    """
    paths: list[str] = []
    for artifact in feature.artifacts:
        if (
            not isinstance(artifact, RepositoryReconnaissanceArtifact)
            or artifact.repository_id != repository_id
        ):
            continue
        for convention in artifact.conventions:
            if convention.wiring_path and convention.wiring_path not in paths:
                paths.append(convention.wiring_path)
        # A convention's `wiring_path` is optional and frequently absent: -106's backend
        # produced seven conventions and not one of them, so relying on it alone left the
        # repository that kept failing the wiring gate with no registry context whatsoever.
        # The reconnaissance tool also derives them structurally, most-referenced first.
        structural = artifact.metadata.get("wiring_files")
        if isinstance(structural, list):
            paths.extend(
                path for path in structural if isinstance(path, str) and path and path not in paths
            )
    return paths


def _required_reset_trigger(child: ChildWorkflowReference) -> str | None:
    """Return the recorded reason this retry must not build on the previous attempt.

    Today's one policy trigger is the wholesale-rewrite rejection: the gate caught drift,
    and drift is exactly what should not be edited into shape. The marker travels on the
    retry strategy because that is the record the next attempt is already handed.
    """
    strategy = child.retry_strategy if isinstance(child.retry_strategy, dict) else {}
    trigger = strategy.get("workspace_reset_required")
    return trigger if isinstance(trigger, str) and trigger else None


def _bounded_previous_attempt(capture: str | None) -> tuple[str | None, tuple[str, ...]]:
    """Bound a capture at whole-file boundaries, naming every file that was withheld.

    Never a mid-hunk character slice. Complete per-file diffs are included until the budget
    is spent; everything after that is withheld and named, so the prompt can bind its
    byte-identical instruction to exactly the files shown and say plainly what is missing.
    The first file is included even when it alone exceeds the budget, because an oversized
    complete file is still honest evidence and an empty capture is not.
    """
    if not capture:
        return None, ()
    headers = list(_DIFF_FILE_HEADER.finditer(capture))
    if not headers:
        # Not a per-file diff this platform recognises; showing a fragment of it would be
        # the blind slice again, so it is either whole or absent.
        if len(capture) <= _MAX_PREVIOUS_ATTEMPT_CHARACTERS:
            return capture, ()
        return None, ()
    included: list[str] = []
    withheld: list[str] = []
    used = 0
    for index, header in enumerate(headers):
        start = header.start()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(capture)
        segment = capture[start:end]
        path = header.group("b") or header.group("a")
        if (used + len(segment) <= _MAX_PREVIOUS_ATTEMPT_CHARACTERS or not included) and (
            not withheld
        ):
            included.append(segment)
            used += len(segment)
        else:
            withheld.append(path)
    return "".join(included).strip() or None, tuple(withheld)


def _publication_outcome_checkpointed(
    feature: FeatureWorkflowSnapshot,
    *,
    child: ChildWorkflowReference,
    candidate: ExternalOperation,
    receipt: dict[str, Any],
) -> bool:
    """Whether durable child state already records this publication receipt's outcome.

    The question the crash-window recovery actually needs answered: a worker that died
    after commit or push left a receipt whose outcome durable state does *not* hold, and
    that receipt must be recovered. A receipt whose outcome durable state *does* hold is
    settled history, and re-entering recovery for it replays the old publication instead of
    running the attempt that was asked for -- AB-Feature-184's integration-fix attempt died
    that way, because the pull request (the previous settledness evidence) is recorded only
    after integration review approves the whole feature.

    Answered from the receipt's completion identity where the ids match, and otherwise from
    the published completion itself: parent persistence renames a completion to its
    repository-and-attempt scoped id, so the receipt's original id can be referenced by
    nothing in durable state even though the outcome is fully recorded. The published
    completion carries a commit SHA and the exact reviewed-content fingerprint this
    receipt's operation journaled, which is the same content identity the evidence gate
    binds. A receipt this function cannot read is not settled -- the caller's validation
    then refuses, fail closed, rather than re-running coding after a possible commit.
    """
    completion_payload = receipt.get("code_completion")
    completion_id = (
        completion_payload.get("artifact_id") if isinstance(completion_payload, dict) else None
    )
    if not isinstance(completion_id, str) or not completion_id:
        return False
    if child.code_completion_artifact_id == completion_id:
        return True
    if any(
        isinstance(artifact, ChildWorkflowResultArtifact)
        and artifact.child_workflow_id == child.child_workflow_id
        and artifact.code_completion_artifact_id == completion_id
        for artifact in feature.artifacts
    ):
        return True
    fingerprint = candidate.safe_metadata.get("reviewed_content_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        return False
    return any(
        isinstance(artifact, CodeCompletionArtifact)
        and artifact.workflow_id == child.child_workflow_id
        and artifact.commit_sha is not None
        and artifact.metadata.get("published_after_approval") is True
        and artifact.metadata.get("reviewed_content_fingerprint") == fingerprint
        for artifact in feature.artifacts
    )


async def _repository_image_directory(
    workspace: Path,
    *,
    runner: ProcessRunner,
    timeout: float,
    cancellation_token: CancellationToken,
) -> str | None:
    """Return the directory this repository already keeps images in, or nothing.

    Decided from the checkout's own tracked files -- the directory holding the most committed
    images is where the next image belongs -- so no framework convention is encoded here. A
    Next.js application answers `public`; something else answers whatever it actually uses;
    a repository with no images at all answers nothing and the assets stay under `.design`.

    Read-only, and bounded to tracked files: `git ls-files` never reaches outside the
    checkout and never reports a build artefact that happens to be lying around.
    """
    try:
        listed = await runner.run(
            ("git", "ls-files"),
            cwd=workspace,
            timeout_seconds=timeout,
            cancellation_token=cancellation_token,
            environment=repository_subprocess_environment(),
        )
    except Exception:  # noqa: BLE001 - a best-effort inspection reports nothing on failure
        return None
    if listed.return_code != 0:
        return None
    counts: dict[str, int] = {}
    for line in listed.stdout.splitlines():
        name = line.strip()
        if not name or Path(name).suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        parent = Path(name).parent.as_posix()
        if parent in {"", "."}:
            continue
        counts[parent] = counts.get(parent, 0) + 1
    if not counts:
        return None
    # Most images wins; the shallowest path breaks a tie, because a repository's image root
    # is the directory its subdirectories hang off.
    return min(sorted(counts, key=lambda k: (-counts[k], k.count("/"), k)), key=lambda k: 0)


async def _place_design_assets(
    workspace: Path,
    detail: DesignDetailArtifact | None,
    *,
    source: AttachmentContentSource | None,
    runner: ProcessRunner | None,
    timeout: float,
    cancellation_token: CancellationToken,
) -> None:
    """Write the design's exported images into the checkout the attempt is about to change.

    Placed by the platform rather than asked for by the Engineer, because an agent cannot
    fetch bytes: the coding tools write text. Without this an image reaches it as
    `{"type": "IMAGE"}` beside a width and a height, and an empty box is the only honest thing
    it can draw -- which is what AB-Feature-231 drew for a 862x868 background and five
    thumbnails.

    The path is the artifact's own (`.design/assets/...`), resolved under the workspace and
    refused if it escapes: the value is platform-composed, so this should never bite, which is
    the reason to check it rather than the reason to skip it.

    Best-effort by design. A write that fails costs that one image and never the attempt --
    the rendering still names the box, and a change built without the artwork is worth more
    than no change at all.
    """
    if detail is None or not detail.assets or source is None:
        return
    # Where this repository already keeps images, read off its own tracked files rather than
    # assumed from a framework convention: the directory holding the most committed image
    # files is where the next image belongs. `public` in a Next.js app, `static` elsewhere,
    # and nothing encoded here either way.
    #
    # It has to be the platform's decision because the Engineer cannot make it: its tools
    # write text, so it cannot move a PNG. AB-Feature-236 was told to "move each file to
    # wherever this repository keeps static assets", correctly referenced
    # `/74046-28182-image-41.png`, and could not move the file -- so the reference pointed at
    # nothing.
    if not workspace.is_dir():
        # Never bring the workspace into existence. `mkdir(parents=True)` below would, and the
        # clone then refuses the destination with "workflow workspace already exists" -- which
        # is how AB-Feature-233 died before its first attempt, having exported all six images
        # successfully. The call site places these after provisioning; this makes the ordering
        # impossible to get wrong from anywhere.
        _LOGGER.warning(
            "design assets were not placed: the checkout does not exist yet "
            "[repository=%s outcome=design_assets_before_checkout]",
            detail.repository_id,
        )
        return
    # Defaulted here rather than treated as "skip", which is this codebase's convention for
    # an absent runner -- `ReviewerAgent` and the two in-attempt checkers all do
    # `process_runner or AsyncioProcessRunner(...)`. `LiveChildWorkstreamExecutor` is
    # constructed without one, so reading `None` as "do not detect" silently sent every
    # exported image to `.design/assets`, which nothing serves: AB-Feature-237 referenced six
    # public URLs that resolved to nothing and its self-review said so precisely.
    detect = runner or AsyncioProcessRunner(output_limit_bytes=128 * 1024)
    served = await _repository_image_directory(
        workspace, runner=detect, timeout=timeout, cancellation_token=cancellation_token
    )
    root = workspace.resolve()
    written: list[str] = []
    for asset in detail.assets:
        if not asset.attachment_id:
            # Nothing durable was stored, so there are no bytes to place on this attempt.
            continue
        placed = f"{served}/{Path(asset.workspace_path).name}" if served else asset.workspace_path
        target = (root / placed).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            _LOGGER.warning(
                "a design asset named a path outside the workspace and was not written "
                "[node=%s outcome=design_asset_escaped]",
                asset.node_id,
            )
            continue
        content = await source.get_content(asset.attachment_id)
        if not content:
            # Purged or never stored. The rendering still names the box and its size, so the
            # Engineer is not left inventing one; it simply has no artwork to put in it.
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        except OSError as error:
            _LOGGER.warning(
                "a design asset could not be written into the checkout: %s "
                "[node=%s outcome=design_asset_unwritten]",
                type(error).__name__,
                asset.node_id,
            )
            continue
        written.append(placed)
    await _commit_placed_assets(
        workspace, written, runner=detect, timeout=timeout, cancellation_token=cancellation_token
    )


async def _commit_placed_assets(
    workspace: Path,
    paths: Sequence[str],
    *,
    runner: ProcessRunner,
    timeout: float,
    cancellation_token: CancellationToken,
) -> None:
    """Commit the artwork the platform placed, because only the platform can.

    The attempt's commit stages exactly the reviewed paths and is bound to their content
    fingerprint -- a boundary that exists so nothing unreviewed can ride into a published
    branch, and one that correctly refuses these files: the Engineer did not write them and
    the Reviewer did not review their bytes.

    So the platform commits its own contribution. AB-Feature-237 raised
    https://github.com/Appbroda/DAM_admin/pull/211 referencing
    `/74046-28182-image-41.png` and five others, and every one of them 404s on the branch:
    the artwork was in the workspace, the code pointed at it correctly, and nothing ever
    committed it.

    Its own commit rather than the attempt's, so the reviewed-path boundary is untouched and
    the diff a reviewer reads still contains only what the Engineer wrote. Staged by explicit
    path -- never `git add -A` -- so nothing else in the tree can ride along.
    """
    if not paths:
        return
    for command in (
        ("git", "add", "--", *paths),
        ("git", "commit", "--quiet", "-m", "platform: place the design's exported artwork"),
    ):
        await cancellation_token.raise_if_cancelled()
        try:
            result = await runner.run(
                command,
                cwd=workspace,
                timeout_seconds=timeout,
                cancellation_token=cancellation_token,
                environment=repository_subprocess_environment(),
            )
        except Exception:  # noqa: BLE001 - best effort; the files are placed either way
            _LOGGER.warning(
                "the placed design artwork could not be committed "
                "[outcome=design_assets_uncommitted]"
            )
            return
        if result.return_code != 0:
            # Nothing to commit is the ordinary case on a retry that preserved its checkout:
            # the artwork is already committed and identical.
            _LOGGER.info(
                "placing design artwork made no commit "
                "[command=%s outcome=design_assets_unchanged]",
                command[1],
            )
            return


def _assigned_design_contents(
    feature: FeatureWorkflowSnapshot, repository_id: str
) -> tuple[Mapping[str, Any], ...]:
    """Return this repository's assigned frames at build fidelity, for the value check.

    The *detail* artifact and never the snapshot: the snapshot holds index records, which carry
    no paint at all, so checking a change against one would report every colour in the design as
    absent from every implementation. Empty when this workstream was assigned no frame, which
    makes the check return nothing.
    """
    detail = _feature_design_detail(feature, repository_id)
    if detail is None:
        return ()
    return tuple(node.content for node in detail.nodes)


def _feature_design_snapshot(feature: FeatureWorkflowSnapshot) -> DesignSnapshotArtifact | None:
    """Return the newest design snapshot this feature holds, or nothing.

    Newest rather than first, because a refresh is a new revision and the attempt being run
    now builds against the current one. The earlier revisions stay in the feature's artifacts
    and stay readable, which is what makes an old attempt's diff explainable.
    """
    for artifact in reversed(feature.artifacts):
        if isinstance(artifact, DesignSnapshotArtifact):
            return artifact
    return None


def _feature_design_detail(
    feature: FeatureWorkflowSnapshot, repository_id: str
) -> DesignDetailArtifact | None:
    """Return this repository's newest design detail, paired to the snapshot in play.

    Newest, like `_feature_design_snapshot` and for its reason. Filtered on the snapshot
    revision as well as the repository, because the two artifacts have to agree about which
    design an attempt built against: a detail resolved against an earlier snapshot revision
    describes a design this attempt is no longer being judged against, and quoting it would
    make the pairing unknowable exactly where 96- says it must stay checkable.
    """
    snapshot = _feature_design_snapshot(feature)
    if snapshot is None:
        return None
    for artifact in reversed(feature.artifacts):
        if (
            isinstance(artifact, DesignDetailArtifact)
            and artifact.repository_id == repository_id
            and artifact.snapshot_artifact_id == snapshot.artifact_id
        ):
            return artifact
    return None


def _prior_child_completions(
    feature: FeatureWorkflowSnapshot, child: ChildWorkflowReference
) -> list[CodeCompletionArtifact]:
    """Return this child's earlier attempts' completions, renamed for child-scoped lineage.

    Parent persistence namespaces a completion by repository --
    ``006_code_completion.<repository>.attempt-N.json`` -- which the child-scoped lineage
    matcher deliberately does not accept, so each is renamed back to the child-scoped form
    the Engineer originally gave it. Published completions are excluded because their files
    are already in HEAD; carrying them forward would re-stage settled history as this
    attempt's work. Only attempts before this one count: a completion recorded for this
    attempt number is crash-recovery evidence, and the reconciliation path owns it -- except
    a context-partition cluster's completion (80-), which legitimately shares this attempt
    number: the wave's later clusters and its final attempt build on it, and the final
    completion must merge its files or publication loses them.
    """
    completions: dict[tuple[int, int], CodeCompletionArtifact] = {}
    for artifact in feature.artifacts:
        if (
            not isinstance(artifact, CodeCompletionArtifact)
            or artifact.workflow_id != child.child_workflow_id
            or artifact.metadata.get("published_after_approval") is True
        ):
            continue
        attempt = artifact.metadata.get("child_attempt")
        if not isinstance(attempt, int):
            continue
        partition = artifact.metadata.get("context_partition_index")
        if isinstance(partition, int):
            if attempt > child.retry_count:
                continue
        elif attempt >= child.retry_count:
            continue
        completions[(attempt, partition if isinstance(partition, int) else -1)] = artifact
    return [
        completions[key].model_copy(
            update={
                "artifact_id": attempt_artifact_id(ARTIFACT_FILENAMES["code_completion"], key[0])
            }
        )
        for key in sorted(completions)
    ]


def _prior_child_completions_including_published(
    feature: FeatureWorkflowSnapshot, child: ChildWorkflowReference
) -> list[CodeCompletionArtifact]:
    """Return this child's earlier completions for evidence, published attempts included.

    Beside `_prior_child_completions` and deliberately not a loosening of it. That function
    feeds the *next attempt's* lineage -- the files publication will stage -- and excluding
    published completions is correct there for the reason it states: their files are already
    in `HEAD`, and carrying them forward would re-stage settled history as this attempt's
    work.

    This one answers a different question: what has this workstream's branch got on it. A
    published-after-approval completion is the strongest possible evidence for that, being
    the one kind whose files are certainly committed. Excluding it is precisely why a
    lineage-union reading of AB-Feature-201 would have found nothing -- its two test files
    live in exactly such a completion, and every reader on this platform skips it.

    Scoped to this child, as `_prior_child_completions` is: a sibling repository's completion
    describes a different checkout entirely.
    """
    completions: dict[tuple[int, int], CodeCompletionArtifact] = {}
    for artifact in feature.artifacts:
        if (
            not isinstance(artifact, CodeCompletionArtifact)
            or artifact.workflow_id != child.child_workflow_id
        ):
            continue
        attempt = artifact.metadata.get("child_attempt")
        if not isinstance(attempt, int):
            continue
        partition = artifact.metadata.get("context_partition_index")
        if isinstance(partition, int):
            # A context-partition cluster's completion (80-) legitimately shares this
            # attempt number; it is evidence about the branch exactly like a prior attempt's.
            if attempt > child.retry_count:
                continue
        elif attempt >= child.retry_count:
            continue
        completions[(attempt, partition if isinstance(partition, int) else -1)] = artifact
    return [completions[key] for key in sorted(completions)]


def _previous_child_review(
    feature: FeatureWorkflowSnapshot, child_workflow_id: str
) -> ReviewArtifact | None:
    """Find only the prior review belonging to this child, renamed for child-scoped lineage.

    Parent persistence namespaces a review by repository, exactly as it does a code completion
    -- `_prior_child_completions`'s own docstring names this -- which the child-scoped lineage
    matcher (`artifact_id_matches_lineage`, read by `_prior_reviews` in the reviewer agent)
    deliberately does not accept. Without renaming it back, the reviewer's own escalation and
    prior-findings mechanisms always see zero prior reviews for every repository, on every
    attempt after the first -- confirmed the root cause of Fix 5 (seam-evidence escalation)
    never firing in production. Mirrors the rename `_prior_child_completions` already performs.
    """
    for artifact in reversed(feature.artifacts):
        if isinstance(artifact, ReviewArtifact) and artifact.workflow_id == child_workflow_id:
            attempt = attempt_number_from_qualified_id(artifact.artifact_id)
            if attempt is None:
                return artifact
            return artifact.model_copy(
                update={"artifact_id": attempt_artifact_id(ARTIFACT_FILENAMES["review"], attempt)}
            )
    return None


@dataclass(frozen=True, slots=True)
class _RemediationReviewScope:
    """What a remediation review may block on: unresolved prior findings, and the delta.

    §1.5's other jaw: even a perfect remediation cannot land when round N+1 invents demands
    round N never made. The revision-tracking the child already carries defines "what
    changed" -- the attempt's own written paths -- and the prior review's blocking findings
    define what was already demanded. Everything else a remediation review wants to say is
    recorded as an advisory finding, visible to a person and to the pull request, blocking
    nothing.
    """

    prior_finding_ids: frozenset[str]
    prior_fingerprints: frozenset[str]
    delta_paths: frozenset[str]


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    """How one review's findings divide, and whether the work may be published anyway."""

    blocking_issues: list[str]
    advisory_findings: list[str]
    counts: ReviewFindingCounts | None
    # The declining verdict this platform accepted, where it accepted one. `None` on every
    # other path, including an ordinary approval.
    advisory_verdict: str | None
    # Blocking findings a person's `removal_holds` verdict took out of blocking authority
    # (73-). Every entry is an audit record -- which finding, matched on which fingerprint,
    # whose decision, when -- because a suppression that leaves no trace is a platform
    # quietly re-deciding what a person decided.
    overruled_findings: list[OverruledReviewFinding] = field(default_factory=list)


def _blocking_findings(review: ReviewArtifact, *, bounded: bool) -> list[ReviewFinding]:
    """Return the findings that stop this attempt and are handed to the next one.

    With ``bounded`` false this is the rule the platform has always applied: critical and
    high findings, and where a declining verdict raised none of those, every finding that is
    not `low`, falling back to every finding at all.

    With it true, two things change. A finding must name what it derives from -- or have
    been injected by this platform, which is a failed command or unreadable source rather
    than an opinion -- and the fallback to *every* finding is gone, because it contradicted
    the reserved meaning of `low` the branch above it depends on.
    """
    deterministic = set(review.metadata.get("deterministic_finding_ids") or ())

    def blocks(finding: ReviewFinding) -> bool:
        if not bounded:
            return True
        return finding.finding_id in deterministic or finding.names_its_scope()

    blocking = [
        item for item in review.findings if item.severity in {"critical", "high"} and blocks(item)
    ]
    if review.verdict == "approved" or blocking:
        return blocking
    # The verdict decides whether work is blocked; severity only ranks how the findings
    # read. A review that requested changes over medium-severity findings alone produced an
    # empty issue list, so the retry was handed a generic "did not satisfy its gate" note,
    # rewrote the same code, and spent the whole budget without ever being shown the defect.
    #
    # `low` is reserved, though. It is what the reviewer is told to record when a requirement
    # cannot be expressed in this runtime or is already satisfied by its execution model --
    # wording for the plan to correct, explicitly not a change to make. Passing those back as
    # blocking is how -068's backend was asked to add synchronization to a buffer whose own
    # review said the safety intent was already met. Prefer the findings that did ask for a
    # change.
    promoted = [item for item in review.findings if item.severity != "low" and blocks(item)]
    if bounded:
        return promoted
    return promoted or list(review.findings)


def _blocks_a_remediation(
    finding: ReviewFinding,
    remediation: _RemediationReviewScope,
    deterministic_ids: set[str],
) -> bool:
    """Say whether one finding is within a remediation review's blocking authority."""
    if finding.finding_id in deterministic_ids:
        return True
    if finding.finding_id in remediation.prior_finding_ids:
        return True
    # The fingerprint reduces a finding to the defect it names, so a reworded repeat of a
    # prior demand still counts as the prior demand rather than as a brand-new one.
    if finding_fingerprint(finding) in remediation.prior_fingerprints:
        return True
    return bool(finding.file_path) and finding.file_path in remediation.delta_paths


def _overruled_blocking_findings(
    blocking: Sequence[ReviewFinding],
    *,
    settled: Sequence[SettledQuestion],
    deterministic: Collection[str],
) -> tuple[list[ReviewFinding], list[OverruledReviewFinding]]:
    """Split off the blocking findings a person's ``removal_holds`` verdict has overruled.

    Identity is `fingerprint_for_text` of the finding's description -- the same function on
    the same class of text the recurrence stop matched on and the verdict artifact is keyed
    by (one fingerprint for both authorities was the 49- rule; this is the third authority
    reading the same key). Deliberately nothing looser: a wording that escapes
    `defect_identity` escapes the stop identically, and a wider reviewer-side match would
    suppress findings the person never saw, let alone ruled on.

    A platform-injected deterministic finding is never overruled, whatever its fingerprint.
    The verdict binds judgement, not measurement: a failing required command is a fact about
    this revision, and AB-Feature-204's conflict fingerprint names exactly such a sentence.
    Severity is deliberately not part of the match -- a demand re-raised at `critical` is
    still the demand the person overruled.
    """
    overruled_by = {item.fingerprint: item for item in settled if item.verdict == "removal_holds"}
    if not overruled_by:
        return list(blocking), []
    kept: list[ReviewFinding] = []
    overruled: list[OverruledReviewFinding] = []
    for finding in blocking:
        fingerprint = fingerprint_for_text(finding.description)
        question = overruled_by.get(fingerprint)
        if question is None or finding.finding_id in deterministic:
            kept.append(finding)
            continue
        overruled.append(
            OverruledReviewFinding(
                finding_id=finding.finding_id,
                description=finding.description,
                fingerprint=fingerprint,
                conflict_id=question.conflict_id,
                verdict="removal_holds",
                decision=question.decision,
                decided_by=question.decided_by,
                decided_at=question.decided_at,
            )
        )
    return kept, overruled


def _resolve_review_outcome(
    review: ReviewArtifact,
    *,
    bounded: bool,
    publishable: bool,
    remediation: _RemediationReviewScope | None = None,
    settled: Sequence[SettledQuestion] = (),
) -> ReviewOutcome:
    """Divide a review's findings into what blocks the work and what informs a person.

    Publication without an approving verdict is the whole trade-off, so it is fenced four
    ways. The review must have named nothing that blocks; it must have named *something*,
    because "a person judges this instead" is not an outcome you can give a review that
    raised no finding at all; it must not have asked for a human, since
    `manual_review_required` is a decision no retry and no publication may take from a
    person; and the change itself must be one an approved result may legally carry.

    ``settled`` is the design questions a person has already decided for this repository
    (73-). A blocking finding that re-raises a demand whose verdict was ``removal_holds`` is
    demoted to the overruled record rather than failing the attempt: the verdict already
    binds the Engineer (which must not implement the demand) and the retry authority (which
    may not stop on it), so leaving the review unbound built a pincer that burned the whole
    budget re-litigating an answered question. Everything outside the settled fingerprint
    keeps full blocking authority -- the verdict removes one question from the review's
    jurisdiction, it does not blind the review to anything else. And the demotion is NOT
    gated under ``bounded``: that flag is a temporary A/B measurement of finding
    traceability, and a human verdict is not an experimental arm.
    """
    deterministic = set(review.metadata.get("deterministic_finding_ids") or ())
    blocking = _blocking_findings(review, bounded=bounded)
    if bounded and remediation is not None:
        # A remediation review may not raise the bar the first round set. What it may still
        # block on: a previously reported blocking finding that remains unresolved, and a
        # defect in what changed since the previously reviewed revision -- including
        # regressions. A platform-injected deterministic finding always blocks, because a
        # failed required command ran against exactly this revision. Everything else stays
        # on the record as an advisory finding rather than costing a regeneration cycle.
        blocking = [
            item for item in blocking if _blocks_a_remediation(item, remediation, deterministic)
        ]
    blocking, overruled = _overruled_blocking_findings(
        blocking, settled=settled, deterministic=deterministic
    )
    overruled_ids = {item.finding_id for item in overruled}
    overruled_fingerprints = {item.fingerprint for item in overruled}
    manual_review_required = review.metadata.get("manual_review_required") is True
    may_publish = bool(review.findings) and publishable and not manual_review_required
    declined_without_blocking = review.verdict != "approved" and not blocking
    if not bounded:
        # The only acceptance the unbounded path knows is the one the verdict itself
        # creates: the review declined, and everything that blocked was a demand a person
        # overruled. The same four fences apply, because the verdict answers a design
        # question and nothing else -- it cannot stand in for validation, completeness, or
        # a human-decision limitation about content nobody may read.
        accepted = declined_without_blocking and may_publish and bool(overruled)
        if declined_without_blocking and not accepted and overruled:
            # The no-empty-rejection rule, minus the one demand a person ruled out. Refilled
            # from the unbounded list so the record says why the attempt failed -- but a
            # refill that re-admitted the overruled demand would hand the next attempt the
            # argument again through the back door.
            blocking = [
                item
                for item in _blocking_findings(review, bounded=False)
                if fingerprint_for_text(item.description) not in overruled_fingerprints
                or item.finding_id in deterministic
            ]
        return ReviewOutcome(
            blocking_issues=[item.description for item in blocking],
            # Everything the review said that this outcome is not already carrying, which on
            # acceptance is every finding that was not overruled -- nothing blocked, so the
            # two spellings agree -- and on a rejection is the findings the severity filter
            # left out of the work order.
            #
            # That second case used to be an empty list. `_blocking_findings` returns only the
            # criticals and highs whenever any exist, so a `medium` raised beside a `high` was
            # dropped by the filter and then dropped again here, and the result artifact
            # recorded no trace of it at all. AB-Feature-215's backend is the cost: its first
            # review named an authorization defect and a test-coverage defect, the remediation
            # was shown one of the two, and the one nobody was shown failed the next attempt.
            # A finding the platform has already written down is not free to withhold.
            advisory_findings=[
                item.description
                for item in review.findings
                if item.finding_id not in {entry.finding_id for entry in blocking}
                and item.finding_id not in overruled_ids
            ],
            counts=None,
            advisory_verdict=review.verdict if accepted else None,
            overruled_findings=overruled,
        )
    accepted = declined_without_blocking and may_publish
    if declined_without_blocking and not accepted:
        # Nothing blocks, yet the work cannot be published: an attempt with no reason at all
        # is what this task exists to stop producing, so the unbounded list is used to say
        # why rather than leaving the record empty. Where the review raised no finding at
        # all that list is empty too, which is the behaviour this path already had -- an
        # empty rejection is a defect in the review, and the repair above is what answers it.
        # Minus the overruled demand, for the reason above: the record of why an attempt
        # failed must not name the one thing a person ruled out.
        blocking = [
            item
            for item in _blocking_findings(review, bounded=False)
            if fingerprint_for_text(item.description) not in overruled_fingerprints
            or item.finding_id in deterministic
        ]
    blocking_ids = {item.finding_id for item in blocking}
    unbounded_ids = {item.finding_id for item in _blocking_findings(review, bounded=False)}
    advisory = [
        item
        for item in review.findings
        if item.finding_id not in blocking_ids and item.finding_id not in overruled_ids
    ]
    return ReviewOutcome(
        blocking_issues=[item.description for item in blocking],
        advisory_findings=[item.description for item in advisory],
        counts=ReviewFindingCounts(
            findings_total=len(review.findings),
            findings_blocking=len(blocking),
            findings_advisory=len(advisory),
            findings_untraceable=len(unbounded_ids - blocking_ids - overruled_ids),
            findings_overruled=len(overruled),
        ),
        advisory_verdict=review.verdict if accepted else None,
        overruled_findings=overruled,
    )


def _review_failure_classification(review: ReviewArtifact) -> FailureClassification:
    """Map structured reviewer findings to a retry budget without re-parsing process output."""
    finding_ids = {item.finding_id for item in review.findings}
    if "TEST_COMMAND_NOT_CONFIGURED" in finding_ids:
        return FailureClassification.TEST_INFRASTRUCTURE_MISSING
    if "LINT_CONFIGURATION_INVALID" in finding_ids:
        return FailureClassification.VALIDATION_CONFIGURATION_FAILURE
    # Ordered above the validation-failure branch, which it would otherwise fall into. This
    # is the classifier that ran for AB-Feature-108: it read a 900-second timeout and a Node
    # heap exhaustion as rejected source and handed both to the coding budget.
    if any(item.finding_category == "validation_capacity" for item in review.findings):
        return FailureClassification.VALIDATION_CAPACITY_FAILURE
    if any(item.finding_category == "validation_failure" for item in review.findings):
        return FailureClassification.VALIDATION_SOURCE_FAILURE
    if any(item.finding_category == "contract" for item in review.findings):
        return FailureClassification.CONTRACT_MISMATCH
    if any(item.finding_id == "TESTS_ONLY_CHANGE" for item in review.findings):
        return FailureClassification.IMPLEMENTATION_MISSING
    return FailureClassification.REVIEW_SCOPE_FAILURE


def _contract_change_request(
    *,
    feature: FeatureWorkflowSnapshot,
    repository: RepositorySpec,
    child: ChildWorkflowReference,
    contract: IntegrationContractArtifact,
) -> ContractChangeRequestArtifact:
    """Report a contract mutation attempt as a pause request instead of altering the contract."""
    return create_artifact(
        ContractChangeRequestArtifact,
        workflow_id=feature.feature_id,
        artifact_id=(
            f"013_contract_change_request.{repository.repository_id}.{child.retry_count}.json"
        ),
        producer="child_workflow",
        payload={
            "change_request_id": (
                f"{feature.feature_id}:{repository.repository_id}:{child.retry_count}"
            ),
            "feature_id": feature.feature_id,
            "current_contract_version": contract.contract_version,
            "requested_by_repository_id": repository.repository_id,
            "requested_changes": [
                "Update the integration contract to match the requested OpenAPI change."
            ],
            "reason": (
                "The child modified generated openapi.yaml, which is an immutable "
                "contract projection."
            ),
            "affected_workstreams": [repository.repository_id],
            "compatibility_impact": (
                "Unknown until a human provides an approved replacement contract."
            ),
            "migration_requirements": [],
            "status": ContractChangeRequestStatus.PENDING,
            "resolution": None,
            "new_contract_artifact_id": None,
        },
        metadata={
            "contract_artifact_id": contract.artifact_id,
            "child_workflow_id": child.child_workflow_id,
        },
    )


def _no_change_execution(
    *,
    feature: FeatureWorkflowSnapshot,
    repository: RepositorySpec,
    workstream: RepositoryWorkstreamPlan,
    child: ChildWorkflowReference,
) -> ChildExecution:
    """Report an attempt whose files already match the committed revision."""
    preflight_revision = (
        child.preflight_result.get("revision")
        if isinstance(child.preflight_result, dict)
        and isinstance(child.preflight_result.get("revision"), str)
        else None
    )
    return ChildExecution(
        result=_child_result(
            feature=feature,
            repository=repository,
            workstream=workstream,
            child=child,
            code_completion=None,
            review=None,
            status="failed",
            blocking_issues=[
                "This attempt produced no change: every file it returned was already "
                "identical to the committed revision, so there was nothing to commit."
            ],
        ).model_copy(
            update={
                "failure_classification": FailureClassification.IMPLEMENTATION_MISSING.value,
                "preflight_result": child.preflight_result,
                # No commit was created, so the exact preflight/durable revision remains the
                # current one. Returning null made parent persistence erase R1 precisely on
                # the no-progress path that most needs to say which revision was unchanged.
                "current_revision": preflight_revision or child.current_revision,
            }
        )
    )


async def _maintain_feature_workspaces(
    workspace_root: Path,
    *,
    database: Database | None,
    keep: int,
    minimum_free_bytes: int,
    failed_retention_hours: int = 0,
    failed_retention_features: int = 0,
) -> WorkspaceMaintenanceResult:
    """Prune only proven terminal workspaces, then enforce the worker-volume capacity floor.

    Two retention policies, one per kind of ended feature, because they are retained for
    different reasons and the bound that suits one does not suit the other.

    A finished feature -- completed or cancelled -- keeps its checkout only as a convenience:
    the newest ``keep`` of them stay so somebody looking at what just ran can still open it.

    A ``failed_requires_human`` feature keeps its checkout because that checkout is the only
    copy of what the attempt wrote. Two bounds apply to those, and either one is enough to
    reclaim: an age window, and a count. The age window alone is what run 193 met. Every
    retained workspace on that volume was from the same day, so none was old enough to
    reclaim, the volume reached 99% with 1.84 GB free, and the capacity preflight failed a
    feature whose repositories had both been approved. The count is what makes a day of
    failures bounded rather than only a week of them.

    Zero disables either bound, which is what a deployment that has configured neither gets
    and exactly the behaviour it had before each was added.
    """
    if keep < 0:
        msg = "workspace retention count cannot be negative"
        raise ValueError(msg)
    if failed_retention_features < 0:
        msg = "workspace failed-retention count cannot be negative"
        raise ValueError(msg)
    if minimum_free_bytes < 0:
        msg = "workspace minimum free bytes cannot be negative"
        raise ValueError(msg)
    reclaimable = (
        await _reclaimable_workspace_features(database)
        if database is not None
        else _ReclaimableFeatures((), ())
    )
    result = await asyncio.to_thread(
        _prune_terminal_feature_workspaces,
        workspace_root,
        doomed=_doomed_feature_workspaces(
            reclaimable,
            keep=keep,
            failed_retention_hours=failed_retention_hours,
            failed_retention_features=failed_retention_features,
        ),
    )
    if result.free_bytes < minimum_free_bytes:
        raise WorkspaceCapacityError(
            workspace_root=workspace_root,
            free_bytes=result.free_bytes,
            required_bytes=minimum_free_bytes,
        )
    return result


# Why one particular workspace was reclaimed. Recorded per deletion, because "the volume was
# cleaned" and "your failed feature's checkout was deleted to make room" are the same event
# and only the second one answers an operator who went looking for it.
_RECLAIM_FINISHED = "finished_beyond_retention_count"
_RECLAIM_FAILED_AGE = "failed_beyond_retention_window"
_RECLAIM_FAILED_COUNT = "failed_beyond_retention_count"


@dataclass(frozen=True, slots=True)
class _ReclaimableFeatures:
    """The two DB-confirmed candidate sets, each newest-first.

    Separate because their retention policies are, and ordered because both policies are
    "keep the newest N": the order is the policy's input, not a presentation detail.
    """

    finished: tuple[str, ...]
    failed: tuple[tuple[str, datetime], ...]


def _doomed_feature_workspaces(
    reclaimable: _ReclaimableFeatures,
    *,
    keep: int,
    failed_retention_hours: int,
    failed_retention_features: int,
) -> tuple[tuple[str, str], ...]:
    """Decide which candidate workspaces are reclaimed, and say why for each one.

    Pure, and separate from the deletion for that reason: this is the whole retention policy,
    and it is worth being able to read it without a filesystem or a lock in the way.
    """
    cutoff = (
        datetime.now(UTC) - timedelta(hours=failed_retention_hours)
        if failed_retention_hours > 0
        else None
    )
    doomed = [(feature_id, _RECLAIM_FINISHED) for feature_id in reclaimable.finished[keep:]]
    for index, (feature_id, updated_at) in enumerate(reclaimable.failed):
        # Count first, because it is the stronger statement about this workspace: over the
        # bound it goes whatever its age, and the age rule is what catches the ones under it.
        if failed_retention_features and index >= failed_retention_features:
            doomed.append((feature_id, _RECLAIM_FAILED_COUNT))
        elif cutoff is not None and updated_at < cutoff:
            doomed.append((feature_id, _RECLAIM_FAILED_AGE))
    return tuple(doomed)


async def _reclaimable_workspace_features(database: Database) -> _ReclaimableFeatures:
    """Return the DB-confirmed candidates whose top-level workspace is safe to retire.

    Two newest-first sets, because the two have different retention policies -- see
    ``_maintain_feature_workspaces``. Nothing here decides anything: what this establishes is
    only that the digest-derived directory is the platform's to remove at all.

    A terminal status alone is insufficient: a repository may point at an operator-supplied
    checkout. Requiring at least one persisted repository and rejecting every feature with a
    non-null local path makes the digest-derived directory the only path cleanup can consider.
    """
    has_repository = (
        select(RepositorySpecModel.id)
        .where(RepositorySpecModel.feature_id == FeatureWorkflowModel.feature_id)
        .exists()
    )
    has_user_workspace = (
        select(RepositorySpecModel.id)
        .where(
            RepositorySpecModel.feature_id == FeatureWorkflowModel.feature_id,
            RepositorySpecModel.local_workspace_path.is_not(None),
        )
        .exists()
    )

    def candidates(status: ColumnElement[bool]) -> Any:
        """Build one newest-first candidate query for a status predicate."""
        return (
            select(FeatureWorkflowModel.feature_id, FeatureWorkflowModel.updated_at)
            .where(
                FeatureWorkflowModel.execution_mode == "live",
                status,
                has_repository,
                ~has_user_workspace,
            )
            .order_by(
                FeatureWorkflowModel.updated_at.desc(), FeatureWorkflowModel.feature_id.desc()
            )
        )

    async with database.session() as session:
        finished = (
            await session.execute(
                candidates(FeatureWorkflowModel.status.in_(_TERMINAL_WORKSPACE_STATUSES))
            )
        ).all()
        failed = (
            await session.execute(
                candidates(
                    FeatureWorkflowModel.status == FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
                )
            )
        ).all()
    return _ReclaimableFeatures(
        finished=tuple(str(row[0]) for row in finished),
        failed=tuple((str(row[0]), _as_aware(row[1])) for row in failed),
    )


def _as_aware(value: datetime) -> datetime:
    """Read a persisted timestamp as UTC, whichever backend stored it.

    SQLite returns naive datetimes and PostgreSQL returns aware ones, and the retention
    window compares against `datetime.now(UTC)`. Comparing a naive value to it raises, which
    would turn a cleanup pass into the exception that stops a claim.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _prune_terminal_feature_workspaces(
    workspace_root: Path, *, doomed: Sequence[tuple[str, str]]
) -> WorkspaceMaintenanceResult:
    """Serialize deletion across workers and remove only exact digest-derived directories.

    Given the decision, never making it: ``doomed`` is the retention policy's output, and
    what happens here is the lock, the identity checks on each path, and the removal.
    """
    with _IN_PROCESS_WORKSPACE_MAINTENANCE_LOCK:
        _prepare_workspace_root(workspace_root)
        lock_descriptor = _open_workspace_maintenance_lock(workspace_root)
        try:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            except OSError as error:
                raise _workspace_maintenance_error(
                    action="acquire its maintenance lock", path=workspace_root, error=error
                ) from error
            candidates: list[tuple[str, str, Path]] = []
            skipped_symlinks: list[str] = []
            skipped_non_directories: list[str] = []
            for feature_id, reason in doomed:
                path = workspace_root / _feature_branch_segment(feature_id)
                try:
                    mode = path.lstat().st_mode
                except FileNotFoundError:
                    continue
                except OSError as error:
                    raise _workspace_maintenance_error(
                        action="inspect a terminal feature workspace", path=path, error=error
                    ) from error
                if stat.S_ISLNK(mode):
                    skipped_symlinks.append(feature_id)
                    continue
                if not stat.S_ISDIR(mode):
                    skipped_non_directories.append(feature_id)
                    continue
                candidates.append((feature_id, reason, path))

            removed: list[str] = []
            for feature_id, reason, path in candidates:
                # One line per deletion, at warning level, before it happens. A retained
                # failure workspace is the only copy of what that attempt wrote, so a batch
                # summary at info is not enough: an operator who goes looking for a checkout
                # that is gone has to be able to find the line that says this pass removed
                # it, which feature it belonged to, and which bound it was over.
                _LOGGER.warning(
                    "retained_feature_workspace_reclaimed",
                    extra={
                        "feature_id": feature_id,
                        "reason": reason,
                        "workspace": str(path),
                    },
                )
                try:
                    # ``rmtree`` is deliberately called without ``ignore_errors``. On supported
                    # platforms it uses descriptor-relative traversal to resist symlink swaps.
                    shutil.rmtree(path)
                except OSError as error:
                    raise _workspace_maintenance_error(
                        action="remove a DB-confirmed terminal feature workspace",
                        path=path,
                        error=error,
                    ) from error
                removed.append(feature_id)

            try:
                free_bytes = shutil.disk_usage(workspace_root).free
            except OSError as error:
                raise _workspace_maintenance_error(
                    action="measure free capacity", path=workspace_root, error=error
                ) from error
        finally:
            os.close(lock_descriptor)

    if skipped_symlinks or skipped_non_directories:
        _LOGGER.warning(
            "workspace cleanup skipped unsafe terminal paths",
            extra={
                "symlink_feature_ids": skipped_symlinks,
                "non_directory_feature_ids": skipped_non_directories,
            },
        )
    if removed:
        _LOGGER.info(
            "removed retained terminal feature workspaces",
            extra={"feature_ids": removed, "free_bytes": free_bytes},
        )
    return WorkspaceMaintenanceResult(
        removed_feature_ids=tuple(removed),
        skipped_symlink_feature_ids=tuple(skipped_symlinks),
        skipped_non_directory_feature_ids=tuple(skipped_non_directories),
        free_bytes=free_bytes,
    )


def _prepare_workspace_root(workspace_root: Path) -> None:
    """Create a real workspace root without accepting a symlink or non-directory target."""
    try:
        workspace_root.mkdir(parents=True, exist_ok=True)
        mode = workspace_root.lstat().st_mode
    except OSError as error:
        raise _workspace_maintenance_error(
            action="prepare the configured root", path=workspace_root, error=error
        ) from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        diagnostic = (
            f"Workspace maintenance refused configured root `{workspace_root}` because it is "
            "not a real directory. Configure a non-symlink worker-volume directory."
        )
        raise WorkspaceMaintenanceError(diagnostic)


def _open_workspace_maintenance_lock(workspace_root: Path) -> int:
    """Open the cross-process maintenance lock without following a substituted symlink."""
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    lock_path = workspace_root / _WORKSPACE_MAINTENANCE_LOCK
    try:
        return os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise _workspace_maintenance_error(
            action="open its maintenance lock", path=lock_path, error=error
        ) from error


def _workspace_maintenance_error(
    *, action: str, path: Path, error: OSError
) -> WorkspaceMaintenanceError:
    """Build one credential-free, operator-actionable filesystem failure."""
    errno = error.errno if error.errno is not None else "unknown"
    return WorkspaceMaintenanceError(
        f"Workspace maintenance could not {action} at `{path}` (errno {errno}). "
        "Check worker-volume permissions and filesystem health before retrying."
    )


def _preserved_on_reset(workspace: Path) -> frozenset[str]:
    """Return the entries this checkout's own detected toolchain produced and will not again.

    Asked of the checkout, not assumed. The platform already identifies a repository's stack
    from its manifests and sources, so the dependency directory is a property of what the
    detection found rather than a constant somebody remembered to extend.

    A checkout nothing classifies preserves nothing, and that is a decision rather than an
    oversight. Detection is what selects the dependency install in the first place, so a
    checkout it recognises no runtime in had no journaled install step and therefore holds no
    result a replay could fail to reproduce. Preserving more there would keep exactly the
    build output ``-x`` exists to remove -- the compiled tree that made a project-wide lint
    time out -- in exchange for protecting a tree nothing created. The cost of being wrong
    about it is a slower attempt, not a workstream that silently validates against nothing.
    """
    profile = inspect_repository_technology(workspace)
    return frozenset(
        entry
        for language in profile.languages
        for entry in _INSTALLED_TREES_BY_RUNTIME.get(language, ())
    )


def _cleanable_paths(workspace: Path) -> list[str]:
    """Return the top-level entries a clean may touch, naming them rather than excluding them.

    ``git clean --exclude`` does not protect a path: it adds the pattern to the ignore rules,
    which with ``-x`` or ``-X`` marks it for deletion instead. Every clean the platform ran
    was therefore destroying the dependency tree and the hook bootstrap it believed it was
    preserving, and the commit that followed failed because its pre-commit hook was gone.
    Excluding a gitignored directory by pathspec does not work either, because Git removes
    such a directory as a single unit, so the safe form is to name what may be cleaned.
    """
    preserved = {*_preserved_on_reset(workspace), ".git"}
    return sorted(entry.name for entry in workspace.iterdir() if entry.name not in preserved)


def _declared_node_package_manager(workspace: Path) -> str:
    """Select the checkout-declared manager for a root prepare script."""
    if (workspace / "pnpm-lock.yaml").is_file():
        return "pnpm"
    if (workspace / "yarn.lock").is_file():
        return "yarn"
    try:
        package = json.loads((workspace / "package.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        package = {}
    declared = package.get("packageManager") if isinstance(package, dict) else None
    if isinstance(declared, str):
        manager = declared.split("@", maxsplit=1)[0]
        if manager in {"npm", "pnpm", "yarn"}:
            return manager
    return "npm"


def _unlink_if_present(path: Path) -> bool:
    """Delete one path, reporting whether it was there. Blocking, so callers use a thread."""
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def _require_workspace_command(result: Any, *, stage: str) -> None:
    """Never continue with a partially reset or partially restored checkout."""
    if not result.succeeded:
        raise RepositoryWorkspaceError(stage=stage, result=result)


async def _baseline_validation_failures(
    validation_tool: WorkspaceValidationTools, *, timeout_seconds: float, measure: bool
) -> list[str]:
    """Record which repository-native commands already fail on the untouched checkout.

    Measured only on the first attempt, because a retry runs against a workspace that still
    holds the previous attempt's files and would attribute the engineer's own defect to the
    repository.
    """
    if not measure:
        return []
    # Stamped as the baseline it is, so the journal says whose work these commands measured.
    # Without it the rows are indistinguishable from the attempt's own validation, and 201's
    # backend attempt 0 -- stopped at coding by the self-review gate, its change never
    # validated -- showed three ticked validation rows that had measured the repository.
    results = await validation_tool.run_validations_async(
        timeout_seconds=timeout_seconds, purpose="baseline"
    )
    return [
        validation_failure_identity(result)
        for result in results
        if result.status is ValidationStatus.FAILED and not result.cancelled and result.required
    ]


def _block_failed_baseline_validation(
    preflight: RepositoryPreflightResult, failures: Sequence[str]
) -> RepositoryPreflightResult:
    """Stop before coding when the untouched checkout cannot pass a required command."""
    if not failures:
        return preflight
    issues = [
        PreflightIssue(
            issue_id=f"BASELINE_REQUIRED_VALIDATION_FAILED_{index}",
            category="baseline_validation",
            severity="high",
            description="The untouched checkout fails a required repository validation.",
            evidence=failure,
            recommended_action=(
                "Repair the repository's default branch or explicitly change its checked-in "
                "validation policy before requesting feature implementation."
            ),
            automatically_repairable=False,
        )
        for index, failure in enumerate(failures, start=1)
    ]
    return preflight.model_copy(
        update={
            "validation_readiness": "blocked",
            "blocking_issues": [*preflight.blocking_issues, *issues],
            "baseline_validation_failures": list(failures),
        }
    )


def _pinned_validation_plan(child: ChildWorkflowReference) -> RepositoryValidationPlan | None:
    """Return the plan this workstream already derived and persisted, where one exists.

    Read from the durable child row rather than any in-memory field, because a resume and the
    integration-fix re-entry start a fresh loop that has only the row. Anything that does not
    parse as a plan -- a legacy row, a schema drift -- answers None, and the caller derives
    fresh: a malformed pin must never fail an attempt.
    """
    if not isinstance(child.validation_plan, dict) or not child.validation_plan:
        return None
    try:
        return RepositoryValidationPlan.model_validate(child.validation_plan)
    except ValidationError:
        return None


def _log_validation_plan_drift(
    pinned: RepositoryValidationPlan,
    derived: RepositoryValidationPlan,
    *,
    feature_id: str,
    repository_id: str,
    attempt: int,
) -> None:
    """Record the commands a fresh derivation would add beyond the workstream's pinned plan.

    Logged once per attempt and never acted on: a command that appears mid-workstream exists
    because of something the workstream itself wrote, and adding it as required would let the
    agent author its own gate. The drift is still worth a record -- it is the signal that a
    repository legitimately gained tooling, which a person can encode in configuration.
    """
    pinned_commands = {tuple(item.command) for item in pinned.commands}
    added = [
        " ".join(item.command)
        for item in derived.commands
        if tuple(item.command) not in pinned_commands
    ]
    if not added:
        return
    _LOGGER.info(
        "validation_plan_drift",
        extra={
            "feature_id": feature_id,
            "repository_id": repository_id,
            "attempt": attempt,
            "commands": added,
        },
    )


def _commit_gate_command(plan: RepositoryValidationPlan) -> tuple[tuple[str, ...], str] | None:
    """Return the repository's own lint command and the directory it must run in.

    The directory is part of the command. A package inside a monorepo declares its lint
    script in its own manifest, and running that script from the checkout root either fails
    outright or lints a different scope than the commit hook does.
    """
    for command in plan.commands:
        if command.validation_type == "lint":
            return tuple(command.command), command.working_directory
    return None


def _block_unrepairable_lint_configuration(
    preflight: RepositoryPreflightResult,
    failed_lint_results: list[ValidationResult],
    workspace: Path,
) -> RepositoryPreflightResult:
    """Stop before coding when the repository's own lint setup cannot be safely repaired.

    Dependencies are already installed deterministically from the lockfile, so a shared
    config that still will not resolve is either undeclared or structurally broken. Coding
    against it produces lint failures the engineer cannot fix, which is how feature ``-007``
    burned its whole retry budget without ever touching production code.
    """
    unrepairable: list[PreflightIssue] = []
    for result in failed_lint_results:
        classification = classify_lint_failure(
            result_stdout=result.stdout, result_stderr=result.stderr, repository_root=workspace
        )
        if classification.disposition is not RepositoryHealthDisposition.REQUIRES_HUMAN:
            continue
        unrepairable.append(
            PreflightIssue(
                issue_id="LINT_CONFIGURATION_REQUIRES_HUMAN",
                category=(
                    "missing_dependency"
                    if classification.kind == "missing_dependency"
                    else "invalid_lint_configuration"
                ),
                severity="high",
                description=classification.description,
                evidence=(
                    f"Command {' '.join(result.command)} exited before linting source files; "
                    "the child transcript was not persisted. Classification code: "
                    f"{classification.kind.upper()}."
                ),
                recommended_action=(
                    "Declare the shared configuration as a repository dependency, or remove the "
                    "reference from the checked-in lint configuration, then restart the feature."
                ),
                automatically_repairable=False,
                # Carried through rather than left on the classification. These name the exact
                # packages that would not resolve, and a repair proposal has to be able to say
                # which ones without re-reading the sentence above.
                unresolved_references=list(classification.unresolved_references),
                undeclared_references=list(classification.undeclared_references),
            )
        )
    if not unrepairable:
        return preflight
    return preflight.model_copy(
        update={
            "validation_readiness": "blocked",
            "blocking_issues": [*preflight.blocking_issues, *unrepairable],
        }
    )


def _require_workspace_child(workspace: Path, workspace_root: Path) -> None:
    """Reject workspace traversal before clone and validation actions."""
    root = workspace_root.resolve(strict=False)
    if workspace == root or root not in workspace.parents:
        msg = "feature child workspace must be a child of configured workspace_root"
        raise RuntimeConfigurationError(msg)


def _git_source_url(repository_url: str) -> str:
    """Normalize a token-free GitHub repository URL to the safe live clone form."""
    if not repository_url.endswith(".git"):
        return f"{repository_url.rstrip('/')}.git"
    return repository_url


def _apply_feature_limits(state: FeatureWorkflowSnapshot, settings: Settings) -> None:
    """Apply versioned deployment policy on every start/resume without storing credentials."""
    state.max_clarification_rounds = settings.max_clarification_rounds
    state.max_child_review_cycles = settings.max_child_review_cycles
    state.max_implementation_retries = settings.max_implementation_retries
    state.max_validation_retries = settings.max_validation_retries
    state.max_repository_setup_retries = settings.max_repository_setup_retries
    state.max_integration_review_cycles = settings.max_integration_review_cycles
    state.max_contract_revision_cycles = settings.max_contract_revision_cycles


def _safe_segment(value: str) -> str:
    """Return a short path component for the non-repository parent planning descriptor."""
    return "".join(character if character.isalnum() else "-" for character in value.lower())[:48]


def _prompt_loader() -> Any:
    """Create a fresh prompt loader so no request data can outlive its invocation frame."""
    from prompts.prompt_loader import PromptLoader

    return PromptLoader()


__all__ = ["LiveChildWorkstreamExecutor", "ProductionFeatureRunner"]
