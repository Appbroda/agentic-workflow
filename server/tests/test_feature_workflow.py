"""Regression coverage for parent multi-repository coordination boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, cast

import pytest

import workflows.feature_workflow as feature_workflow_module
from adapters.git_adapter import GitAdapterError, GitSafetyError
from adapters.github_adapter import GitHubAdapterError, MockGitHubService, PullRequestDetails
from adapters.llm_adapter import LLMAdapterError
from agents.engineer.agent import RequiredContextRefusal
from agents.planner.feature_planner import DeterministicFeaturePlanner
from agents.shared.contracts import (
    FEATURE_ARTIFACT_FILENAMES,
    artifact_id_matches_lineage,
    create_artifact,
    safe_error_diagnostics,
)
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from api.schemas import ClarificationAnswer
from artifacts.schemas import (
    ChildWorkflowResultArtifact,
    CodeCompletionArtifact,
    ContractChangeRequestArtifact,
    FeatureCompletionArtifact,
    FileChange,
    IntegrationContractArtifact,
    IntegrationReviewArtifact,
    IntegrationReviewFinding,
    PullRequestArtifact,
    RepositoryExecutionPlanArtifact,
    RepositoryReconnaissanceArtifact,
    RepositoryWorkstreamPlan,
    ReviewArtifact,
    TechnicalPRDArtifact,
)
from services.cancellation import CancellationRequested, MockCancellationToken
from state.enums import (
    ChildWorkflowStatus,
    ContractChangeRequestStatus,
    FeatureWorkflowStatus,
    TargetedAttempt,
)
from state.external_operations import ExternalOperationType
from state.failure_diagnosis import FeatureFailureClassification
from state.feature_models import ChildWorkflowReference, FeatureWorkflowSnapshot
from storage.external_operation_store import (
    ExternalOperationError,
    OperationInProgressError,
    OperationLeaseLostError,
    OperationReconciliationRequired,
    OperationTransitionConflictError,
)
from tools.reachability import REACHABILITY_DIAGNOSTIC_PREFIX
from tools.requirement_reconciliation import RequirementReconciliation
from tools.retry_strategy import FailureClassification, decide_child_retry
from tools.scoped_tests import scoped_test_diagnostic
from tools.self_review import SELF_REVIEW_SUBSTANTIVE_OUTCOME
from tools.source_formatting import SourceValidationError
from tools.typecheck import TYPECHECK_DIAGNOSTIC_PREFIX
from tools.validation_tools import ValidationResult
from workflow_schema import LEGACY_UNVERIFIED_BUILD_REVISION
from workflows.feature_workflow import (
    ChildExecution,
    FeatureWorkflowError,
    FeatureWorkflowOrchestrator,
    GitHubPullRequestPublisher,
    MockChildWorkstreamExecutor,
    ReconnaissanceReport,
    WorkstreamPublicationClass,
    _append_artifacts,
    _causal_diagnostics,
    _commit_sha,
    _context_partition_clusters,
    _fault_backoff_seconds,
    _feature_branch_segment,
    _replace_artifact,
    _require_acyclic_workstreams,
    _resolve_target_workstream_ids,
    _retry_feedback,
    feature_is_at_rest,
    is_transient_provider_fault,
    publication_is_held,
    require_publishable_feature,
    require_publishable_workstream,
    source_rejection_disposition,
)


async def _publish_on_request(
    state: FeatureWorkflowSnapshot,
    *,
    orchestrator: FeatureWorkflowOrchestrator | None = None,
) -> FeatureWorkflowSnapshot:
    """Press `PUBLISH_FEATURE`, the way a person does, and return what it produced.

    A feature that did not land publishes nothing by itself, so every scenario that used to
    end with an automatic pull request now ends with the offer of one. This is the press --
    including the precondition the route checks first, so a scenario where the action is not
    actually advertised fails here rather than silently publishing from a button nobody
    would have been shown.
    """
    require_publishable_feature(state)
    publisher = orchestrator or FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService())
    )
    return await publisher.publish_feature(
        state,
        requested_by="an operator (via platform key)",
        reason="the sibling is not going to land and this work should be reviewed",
        credentials=_credentials(),
    )


def _published_repository_ids(state: FeatureWorkflowSnapshot) -> set[str]:
    """Return every repository this feature has recorded a pull request for."""
    return {
        repository_id
        for repository_id, child in state.child_workflows.items()
        if child.pull_request_artifact_id is not None
    }


class ContractChangeInOneSibling:
    """Return one change request while another concurrently completed child still has evidence."""

    def __init__(self) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self.calls: list[str] = []
        self._requested_change = False

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Delegate normal artifacts, then make only the backend request a contract revision."""
        execution = await self._delegate.run(**kwargs)
        repository = kwargs["repository"]
        self.calls.append(repository.repository_id)
        if repository.repository_id != "backend" or self._requested_change:
            return execution
        self._requested_change = True
        feature = kwargs["feature"]
        child = kwargs["child"]
        contract = kwargs["contract"]
        waiting_result = execution.result.model_copy(
            update={
                "status": "waiting_for_contract_change",
                "blocking_issues": ["The approved contract needs a human-owned revision."],
                "pull_request_readiness": False,
            }
        )
        request = create_artifact(
            ContractChangeRequestArtifact,
            workflow_id=feature.feature_id,
            artifact_id="013_contract_change_request.backend.0.json",
            producer="child_workflow",
            payload={
                "change_request_id": f"{feature.feature_id}:backend:0",
                "feature_id": feature.feature_id,
                "current_contract_version": contract.contract_version,
                "requested_by_repository_id": "backend",
                "requested_changes": ["Add a contract field."],
                "reason": "The backend requires an explicit contract revision.",
                "affected_workstreams": ["backend"],
                "compatibility_impact": "Requires parent approval.",
                "migration_requirements": [],
                "status": ContractChangeRequestStatus.PENDING,
                "resolution": None,
                "new_contract_artifact_id": None,
            },
            metadata={"child_workflow_id": child.child_workflow_id},
        )
        return ChildExecution(
            result=waiting_result,
            code_completion=execution.code_completion,
            review=execution.review,
            contract_change_request=request,
        )


class PermanentlyFailingPullRequestService(MockGitHubService):
    """Refuse one repository's pull request however many times it is attempted."""

    def __init__(self, repository: str) -> None:
        """Record the repository whose publication the provider will never accept."""
        super().__init__()
        self._repository = repository

    def create_pull_request(self, repository: str, **kwargs: Any) -> Any:
        """Fail one repository permanently while serving every other repository normally."""
        if repository == self._repository:
            raise RuntimeError("simulated provider failure")
        return super().create_pull_request(repository, **kwargs)


class HealablePullRequestService(MockGitHubService):
    """Reject one repository's pull request until the provider fault is cleared."""

    def __init__(self, repository: str) -> None:
        """Start unhealthy for the named repository so the first publication fails."""
        super().__init__()
        self._repository = repository
        self.healthy = False

    def create_pull_request(self, repository: str, **kwargs: Any) -> Any:
        """Fail the named repository until a test marks the provider healthy again."""
        if repository == self._repository and not self.healthy:
            raise RuntimeError("simulated provider failure")
        return super().create_pull_request(repository, **kwargs)


class TransientlyFailingPullRequestService(MockGitHubService):
    """Reject a repository's first publication attempts the way a provider blip does."""

    def __init__(self, repository: str, failures: int) -> None:
        """Record which repository blips and how many attempts fail before it succeeds."""
        super().__init__()
        self._repository = repository
        self._remaining_failures = failures

    def create_pull_request(self, repository: str, **kwargs: Any) -> Any:
        """Fail a bounded number of attempts, then behave exactly like a healthy provider."""
        if repository == self._repository and self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise RuntimeError("simulated transient provider failure")
        return super().create_pull_request(repository, **kwargs)


class RegressingOnReworkExecutor:
    """Approve every repository once, then fail the one integration review sends back."""

    def __init__(self, repository_id: str) -> None:
        """Record which repository fails when it is asked to rework already-approved code."""
        self._delegate = MockChildWorkstreamExecutor()
        self._repository_id = repository_id
        self._seen: set[str] = set()

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Return an approved result the first time and a failure for every rework after it."""
        execution = await self._delegate.run(**kwargs)
        repository_id = kwargs["repository"].repository_id
        first_attempt = repository_id not in self._seen
        self._seen.add(repository_id)
        if repository_id != self._repository_id or first_attempt:
            return execution
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    "blocking_issues": ["The rework regressed the approved implementation."],
                    "pull_request_readiness": False,
                }
            ),
            code_completion=execution.code_completion,
            review=execution.review,
        )


class RejectingIntegrationReviewer:
    """Never approve integration, so the feature has to resolve without that approval."""

    def __init__(self, *, responsible_repository_id: str | None) -> None:
        """Choose whether a finding names a repository the work can be sent back to."""
        self._responsible_repository_id = responsible_repository_id

    async def review(
        self,
        *,
        feature_id: str,
        contract: IntegrationContractArtifact,
        child_results: Sequence[ChildWorkflowResultArtifact],
        merge_order: Sequence[str],
    ) -> IntegrationReviewArtifact:
        """Return a blocking review whose findings never clear, however often it is asked."""
        findings = (
            []
            if self._responsible_repository_id is None
            else [
                {
                    "finding_id": "finding-integration-blocked",
                    "severity": "high",
                    "responsible_repository_id": self._responsible_repository_id,
                    "affected_repository_ids": [self._responsible_repository_id],
                    "contract_reference": "contract",
                    "description": "The contract is not satisfied across repositories.",
                    "evidence": "Simulated blocking integration finding.",
                    "recommended_fix": "Align the implementation with the contract.",
                }
            ]
        )
        return create_artifact(
            IntegrationReviewArtifact,
            workflow_id=feature_id,
            artifact_id=FEATURE_ARTIFACT_FILENAMES["integration_review"],
            producer="integration_reviewer",
            payload={
                "feature_id": feature_id,
                "contract_artifact_id": contract.artifact_id,
                "review_status": "changes_requested",
                "repository_results": [
                    {
                        "repository_id": item.repository_id,
                        "child_workflow_id": item.child_workflow_id,
                        "status": item.status,
                        "child_result_artifact_id": item.artifact_id,
                    }
                    for item in child_results
                ],
                "contract_checks": ["Simulated contract check."],
                "cross_repository_findings": findings,
                "compatibility_assessment": "Simulated cross-repository incompatibility.",
                "security_assessment": "No security assessment was performed.",
                "deployment_assessment": "No deployment assessment was performed.",
                "merge_order": list(merge_order),
                "required_fixes": ["Align the implementation with the contract."],
            },
            metadata={"source_artifact_ids": [contract.artifact_id]},
        )


class LostResponsePullRequestService(MockGitHubService):
    """Create the pull request and then raise, the way a lost provider response does."""

    def __init__(self, repository: str) -> None:
        """Record the repository whose first successful create reports itself as failed."""
        super().__init__()
        self._repository = repository
        self._reported_failure = False

    def create_pull_request(self, repository: str, **kwargs: Any) -> Any:
        """Land the pull request on the provider before reporting the call as failed."""
        details = super().create_pull_request(repository, **kwargs)
        if repository == self._repository and not self._reported_failure:
            self._reported_failure = True
            msg = "provider response was lost after the pull request was created"
            raise RuntimeError(msg)
        return details


class PublishOnlyPublisher:
    """Satisfy publication while deliberately omitting provider fetch-back support."""

    def __init__(self) -> None:
        self._delegate = GitHubPullRequestPublisher()

    async def publish(self, **kwargs: Any) -> list[PullRequestArtifact]:
        """Create artifacts without exposing the verification operation."""
        return await self._delegate.publish(**kwargs)


def _reconnaissance_artifact(
    feature_id: str, repository_id: str, *, contradicted_premises: list[Any] | None = None
) -> Any:
    """Return one grounded reconnaissance artifact for a repository."""
    return create_artifact(
        RepositoryReconnaissanceArtifact,
        workflow_id=feature_id,
        artifact_id=f"015_repository_reconnaissance.{repository_id}.json",
        producer="repository_recon",
        metadata={},
        payload={
            "feature_id": feature_id,
            "repository_id": repository_id,
            "repository_revision": f"revision-{repository_id}",
            "summary": f"The {repository_id} repository as it actually is.",
            "source_areas": [f"{repository_id}/src"],
            "test_areas": [f"{repository_id}/tests"],
            "conventions": [],
            "shared_utilities": [],
            "contradicted_premises": contradicted_premises or [],
        },
    )


class RecordingReconnaissance:
    """Report grounded evidence for every repository and record which were read."""

    def __init__(self, *, contradicted_premises: dict[str, list[Any]] | None = None) -> None:
        """Start with nothing inspected, optionally contradicting a premise per repository."""
        self.inspected: list[str] = []
        self._contradicted = contradicted_premises or {}

    async def inspect(self, **kwargs: Any) -> ReconnaissanceReport:
        """Return one artifact per repository, in submission order."""
        feature = kwargs["feature"]
        artifacts = []
        for repository in kwargs["repositories"]:
            self.inspected.append(repository.repository_id)
            artifacts.append(
                _reconnaissance_artifact(
                    feature.feature_id,
                    repository.repository_id,
                    contradicted_premises=self._contradicted.get(repository.repository_id),
                )
            )
        return ReconnaissanceReport(artifacts=artifacts)


class RecordingPlanner:
    """Plan deterministically while capturing the reconnaissance it was actually handed."""

    def __init__(self) -> None:
        """Delegate planning and record only what reached this boundary."""
        self._delegate = DeterministicFeaturePlanner()
        self.received: list[Any] = []

    async def plan(self, **kwargs: Any) -> Any:
        """Record the reconnaissance argument, then plan as the deterministic planner does."""
        self.received = list(kwargs.get("reconnaissance", ()))
        return await self._delegate.plan(**kwargs)


class UnexpectedProductManager:
    """Prove an audit-only snapshot is rejected before any agent can execute."""

    async def create_technical_prd(self, **_kwargs: Any) -> TechnicalPRDArtifact:
        """Fail the test if durable identity validation is ever bypassed."""
        raise AssertionError("product manager must not run for an audit-only snapshot")

    async def reconcile_requirements(self, **_kwargs: Any) -> RequirementReconciliation:
        """Fail the test for the same reason: no agent of this family may run either."""
        raise AssertionError("reconciliation must not run for an audit-only snapshot")


@pytest.mark.asyncio
async def test_feature_orchestrator_rejects_legacy_identity_before_agent_execution() -> None:
    """Storage guards are not a substitute for fencing side effects at the runner boundary."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-legacy-executor", request).model_copy(
        update={
            "created_by_build_revision": LEGACY_UNVERIFIED_BUILD_REVISION,
            "last_executor_build_revision": LEGACY_UNVERIFIED_BUILD_REVISION,
        }
    )
    orchestrator = FeatureWorkflowOrchestrator(product_manager=UnexpectedProductManager())

    with pytest.raises(FeatureWorkflowError, match="legacy-unverified"):
        await orchestrator.start(
            state,
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )


@pytest.mark.asyncio
async def test_contract_change_pause_retains_completed_parallel_sibling_results() -> None:
    """A parent pause cannot discard completed sibling handoffs and cause an unsafe rerun later."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-contract-pause", request)
    orchestrator = FeatureWorkflowOrchestrator(child_executor=ContractChangeInOneSibling())

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    child_results = [
        artifact
        for artifact in result.artifacts
        if isinstance(artifact, ChildWorkflowResultArtifact)
    ]
    assert result.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
    assert {artifact.repository_id for artifact in child_results} == {"backend", "frontend"}
    assert result.child_workflows["frontend"].status.value == "approved"
    assert result.child_workflows["backend"].status.value == "waiting_for_contract_change"


@pytest.mark.asyncio
async def test_contract_revision_after_a_child_commit_requires_fresh_branches() -> None:
    """A delta rerun cannot relabel committed v1 history as fully reviewed against v2."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-contract-revision", request)
    executor = ContractChangeInOneSibling()
    orchestrator = FeatureWorkflowOrchestrator(child_executor=executor)
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    paused = await orchestrator.start(state, credentials=credentials)
    change_request = next(
        item
        for item in paused.artifacts
        if isinstance(item, ContractChangeRequestArtifact) and item.status == "pending"
    )
    current = next(
        item for item in reversed(paused.artifacts) if isinstance(item, IntegrationContractArtifact)
    )
    revised = current.model_copy(
        update={
            "artifact_id": "009_integration_contract.v2.json",
            "contract_version": "2.0.0",
            "metadata": {**current.metadata, "supersedes": current.artifact_id},
        }
    )

    refused = await orchestrator.approve_contract_change(
        paused,
        request_id=change_request.change_request_id,
        updated_contract=revised,
        resolution="Approve the new shared response schema.",
        credentials=credentials,
    )

    assert refused.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert executor.calls.count("backend") == 1
    assert executor.calls.count("frontend") == 1
    assert refused.failure_summary is not None
    assert (
        refused.failure_summary.root_classification == "contract_revision_requires_fresh_branches"
    )
    assert not any(
        isinstance(item, IntegrationContractArtifact) and item.artifact_id == revised.artifact_id
        for item in refused.artifacts
    )
    latest_request = next(
        item
        for item in reversed(refused.artifacts)
        if isinstance(item, ContractChangeRequestArtifact)
    )
    assert latest_request.status == "rejected"


@pytest.mark.asyncio
async def test_uncommitted_contract_revision_reruns_every_sibling_against_new_contract() -> None:
    """Before any commit exists, all consumers can still be rerun against the exact revision."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-contract-revision-uncommitted", request)
    executor = ContractChangeInOneSibling()
    orchestrator = FeatureWorkflowOrchestrator(child_executor=executor)
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    paused = await orchestrator.start(state, credentials=credentials)
    paused.artifacts = [
        (
            item.model_copy(update={"commit_sha": None})
            if isinstance(item, CodeCompletionArtifact)
            else item.model_copy(update={"metadata": {**item.metadata, "commit_sha": None}})
            if isinstance(item, ChildWorkflowResultArtifact)
            else item
        )
        for item in paused.artifacts
    ]
    change_request = next(
        item
        for item in paused.artifacts
        if isinstance(item, ContractChangeRequestArtifact) and item.status == "pending"
    )
    current = next(
        item for item in reversed(paused.artifacts) if isinstance(item, IntegrationContractArtifact)
    )
    revised = current.model_copy(
        update={
            "artifact_id": "009_integration_contract.v2.uncommitted.json",
            "contract_version": "2.0.0",
            "metadata": {**current.metadata, "supersedes": current.artifact_id},
        }
    )

    completed = await orchestrator.approve_contract_change(
        paused,
        request_id=change_request.change_request_id,
        updated_contract=revised,
        resolution="Approve the new shared response schema before any commit.",
        credentials=credentials,
    )

    assert completed.status is FeatureWorkflowStatus.COMPLETED
    assert executor.calls.count("backend") == 2
    assert executor.calls.count("frontend") == 2
    latest_results: dict[str, ChildWorkflowResultArtifact] = {}
    for artifact in completed.artifacts:
        if isinstance(artifact, ChildWorkflowResultArtifact):
            latest_results[artifact.repository_id] = artifact
    assert set(latest_results) == {"backend", "frontend"}
    assert all(
        result.metadata["contract_artifact_id"] == revised.artifact_id
        and result.metadata["contract_version"] == revised.contract_version
        for result in latest_results.values()
    )


@pytest.mark.asyncio
async def test_publication_failure_does_not_withhold_a_later_repository_pull_request() -> None:
    """The first repository failing at the provider cannot deny a later one its attempt."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-partial-pr", request)
    # Backend publishes first, so a backend fault is the one that used to abort the loop
    # before frontend was ever attempted.
    github = PermanentlyFailingPullRequestService("example/backend")
    orchestrator = FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(
            github_service=github, retry_backoff_seconds=0.0
        )
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    pull_requests = [
        artifact for artifact in result.artifacts if isinstance(artifact, PullRequestArtifact)
    ]
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert [artifact.repository for artifact in pull_requests] == ["example/frontend"]
    assert (
        result.child_workflows["frontend"].pull_request_artifact_id == pull_requests[0].artifact_id
    )
    assert result.child_workflows["backend"].pull_request_artifact_id is None
    assert any(
        "Pull-request creation did not succeed" in issue
        for issue in result.child_workflows["backend"].blocking_issues
    )


@pytest.mark.asyncio
async def test_transient_provider_failure_still_produces_every_pull_request() -> None:
    """A provider blip during creation costs a retry, not the repository's pull request."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-transient-pr", request)
    github = TransientlyFailingPullRequestService("example/frontend", failures=2)
    orchestrator = FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(
            github_service=github, retry_backoff_seconds=0.0
        )
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    pull_requests = [
        artifact for artifact in result.artifacts if isinstance(artifact, PullRequestArtifact)
    ]
    assert result.status is FeatureWorkflowStatus.COMPLETED
    assert {artifact.repository for artifact in pull_requests} == {
        "example/backend",
        "example/frontend",
    }


@pytest.mark.asyncio
async def test_pull_request_that_landed_before_a_failure_is_adopted_not_duplicated() -> None:
    """A create whose response was lost is reconciled, so the retry never opens a second PR."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-lost-response-pr", request)
    github = LostResponsePullRequestService("example/frontend")
    orchestrator = FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(
            github_service=github, retry_backoff_seconds=0.0
        )
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    pull_requests = [
        artifact for artifact in result.artifacts if isinstance(artifact, PullRequestArtifact)
    ]
    assert result.status is FeatureWorkflowStatus.COMPLETED
    # One pull request exists on the provider for the repository whose response was lost,
    # and the artifact points at that one rather than at a duplicate opened by the retry.
    frontend_pull_requests = [
        details
        for details in github.pull_requests.values()
        if details.repository == "example/frontend"
    ]
    assert len(frontend_pull_requests) == 1
    published = next(item for item in pull_requests if item.repository == "example/frontend")
    assert published.pull_request_number == frontend_pull_requests[0].number


@pytest.mark.asyncio
async def test_unconverged_integration_review_still_publishes_reviewed_repositories() -> None:
    """Spending the integration cycle budget cannot strand work that passed its own review."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-integration-stuck", request)
    orchestrator = FeatureWorkflowOrchestrator(
        integration_reviewer=cast(
            Any, RejectingIntegrationReviewer(responsible_repository_id="backend")
        ),
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    pull_requests = [
        artifact for artifact in result.artifacts if isinstance(artifact, PullRequestArtifact)
    ]
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert {artifact.repository for artifact in pull_requests} == {
        "example/backend",
        "example/frontend",
    }
    # An integration review that never approved may not be reported as a completed feature.
    assert not any(isinstance(item, FeatureCompletionArtifact) for item in result.artifacts)
    # Nor may its pull requests tell the human reading them on GitHub that it did. This body
    # is the only account of the feature most reviewers will ever see.
    for artifact in pull_requests:
        assert "did NOT approve this feature." in artifact.body
        assert "integration gate passed" not in artifact.body


@pytest.mark.asyncio
async def test_a_published_pull_request_states_what_the_integration_gate_does_not_check() -> None:
    """Every PR body must carry the gate's coverage limit, not just the completion artifact.

    A reviewer approving one repository's PR has no reason to know that the seam between it
    and its sibling was checked by nobody, and the PR body is where they would look.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-pr-body-coverage", request)
    orchestrator = FeatureWorkflowOrchestrator()

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.COMPLETED
    pull_requests = [
        artifact for artifact in result.artifacts if isinstance(artifact, PullRequestArtifact)
    ]
    assert pull_requests
    for artifact in pull_requests:
        assert "the feature's integration gate approved." in artifact.body
        assert "behaviour across repositories is not reviewed by it" in artifact.body


@pytest.mark.asyncio
async def test_reviewed_commit_is_published_when_the_integration_rework_regresses() -> None:
    """A repository that regressed after review still publishes the commit that passed it.

    On request now rather than automatically -- the feature did not land, so nothing
    publishes itself -- but the commit that is published is unchanged: the reviewed one, from
    the branch that still holds it.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-rework-regressed", request)
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, RegressingOnReworkExecutor("backend")),
        integration_reviewer=cast(
            Any, RejectingIntegrationReviewer(responsible_repository_id="backend")
        ),
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
    )

    held = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )
    assert not _published_repository_ids(held), "a feature that did not land publishes nothing"
    result = await _publish_on_request(held, orchestrator=orchestrator)

    pull_requests = [
        artifact for artifact in result.artifacts if isinstance(artifact, PullRequestArtifact)
    ]
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    # The backend's branch still holds the commit that passed review, so it is published
    # rather than left on the remote attached to nothing.
    assert {artifact.repository for artifact in pull_requests} == {
        "example/backend",
        "example/frontend",
    }
    backend_pull_request = next(
        item for item in pull_requests if item.repository == "example/backend"
    )
    approved_results = [
        artifact
        for artifact in result.artifacts
        if isinstance(artifact, ChildWorkflowResultArtifact)
        and artifact.repository_id == "backend"
        and artifact.status == "approved"
    ]
    assert backend_pull_request.commit_sha == _commit_sha(approved_results[-1])
    # The regression is still on the record, reported by the gate that observed it rather
    # than inferred from the repository looking absent. It used to be phrased "regressed
    # after review" by the missing-required path, which only ran because a failed rework hid
    # the approval from the integration gate -- the defect AB-Feature-152 exposed.
    assert any(
        "regressed the approved implementation" in issue
        for issue in result.child_workflows["backend"].blocking_issues
    )
    assert not any(isinstance(item, FeatureCompletionArtifact) for item in result.artifacts)


@pytest.mark.asyncio
async def test_blocking_integration_finding_without_an_owner_still_publishes() -> None:
    """A blocking review naming no repository has nowhere to send work back to, so it publishes."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-integration-unowned", request)
    orchestrator = FeatureWorkflowOrchestrator(
        integration_reviewer=cast(
            Any, RejectingIntegrationReviewer(responsible_repository_id=None)
        ),
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    pull_requests = [
        artifact for artifact in result.artifacts if isinstance(artifact, PullRequestArtifact)
    ]
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert {artifact.repository for artifact in pull_requests} == {
        "example/backend",
        "example/frontend",
    }
    assert not any(isinstance(item, FeatureCompletionArtifact) for item in result.artifacts)


def _publication_scenario(scenario: str) -> tuple[Any, Any]:
    """Build the child executor and integration reviewer for one way a feature goes wrong."""
    rejecting = RejectingIntegrationReviewer(responsible_repository_id="backend")
    return {
        "everything succeeds": (None, None),
        "one repository never passes its own review": (FailOneRepositoryExecutor("backend"), None),
        "integration review never converges": (None, rejecting),
        "a blocking integration finding names no repository": (
            None,
            RejectingIntegrationReviewer(responsible_repository_id=None),
        ),
        "the integration rework regresses an approved repository": (
            RegressingOnReworkExecutor("backend"),
            rejecting,
        ),
    }[scenario]


@pytest.mark.parametrize(
    "scenario",
    [
        "everything succeeds",
        "one repository never passes its own review",
        "integration review never converges",
        "a blocking integration finding names no repository",
        "the integration rework regresses an approved repository",
    ],
)
@pytest.mark.asyncio
async def test_every_repository_that_passed_review_ends_with_a_pull_request(
    scenario: str,
) -> None:
    """The guarantee itself: reviewed work always reaches a human, by one of two routes.

    Each of these scenarios once ended with reviewed, pushed work and no pull request for it,
    by a different route. The route does not matter to the operator; the missing pull request
    does. That has not changed and this test still asserts it.

    What 81- changed is the delivery. A feature that fully landed publishes itself, exactly as
    before. A feature that did not publishes nothing until a person says so -- because a
    pull request whose sibling repository does not exist is not reviewable work -- and the
    offer to publish it has to be advertised on the feature and has to actually work. A
    scenario reaching neither a pull request nor a working offer is the original bug, so both
    halves are asserted here rather than the second being taken on trust.
    """
    executor, reviewer = _publication_scenario(scenario)
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state(f"feature-invariant-{abs(hash(scenario))}", request)
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor) if executor is not None else None,
        integration_reviewer=cast(Any, reviewer) if reviewer is not None else None,
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    reviewed_repository_ids = {
        artifact.repository_id
        for artifact in result.artifacts
        if isinstance(artifact, ChildWorkflowResultArtifact)
        and artifact.status == "approved"
        and artifact.pull_request_readiness
    }
    assert reviewed_repository_ids, f"{scenario} must still produce reviewed work to publish"
    if reviewed_repository_ids <= _published_repository_ids(result):
        # Published automatically, which is what a feature that landed still does.
        return
    # Otherwise the feature is holding publication, and it owes three things: rest, so the
    # queue does not churn on it; an advertised action; and pull requests when that action is
    # taken. Work that reaches none of them is exactly the defect this rule was written for.
    assert publication_is_held(result), (
        f"{scenario}: reviewed work is unpublished and the feature is not holding publication"
    )
    assert feature_is_at_rest(result), f"{scenario}: a held feature must be at rest"
    published = await _publish_on_request(result, orchestrator=orchestrator)
    assert reviewed_repository_ids <= _published_repository_ids(published), (
        f"{scenario}: reviewed work was left on the remote without a pull request"
    )


@pytest.mark.asyncio
async def test_publisher_without_fetch_back_cannot_mark_feature_complete() -> None:
    """The publisher interface cannot silently turn missing verification into approval."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-unverified-publisher", request)
    orchestrator = FeatureWorkflowOrchestrator(
        pull_request_publisher=cast(Any, PublishOnlyPublisher())
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert not any(isinstance(item, FeatureCompletionArtifact) for item in result.artifacts)
    assert all(
        any("cannot fetch provider state" in issue for issue in child.blocking_issues)
        for child in result.child_workflows.values()
    )


@pytest.mark.asyncio
async def test_partial_pull_request_resume_does_not_rerun_completed_children() -> None:
    """Resume must create only the missing PR, never another commit for a published child."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-partial-pr-resume", request)
    github = HealablePullRequestService("example/frontend")
    executor = FailOneRepositoryExecutor("never")
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=executor,
        pull_request_publisher=GitHubPullRequestPublisher(
            github_service=github, retry_backoff_seconds=0.0
        ),
    )
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    failed = await orchestrator.start(state, credentials=credentials)
    assert failed.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN

    github.healthy = True
    resumed = await orchestrator.resume(failed, answers=[], credentials=credentials)

    assert resumed.status is FeatureWorkflowStatus.COMPLETED
    # The already-published repository is neither re-implemented nor published a second time.
    assert executor.calls == ["backend", "frontend"]
    pull_requests = [
        artifact for artifact in resumed.artifacts if isinstance(artifact, PullRequestArtifact)
    ]
    assert {artifact.repository for artifact in pull_requests} == {
        "example/backend",
        "example/frontend",
    }
    assert len(github.pull_requests) == 2


class FailOneRepositoryExecutor:
    """Fail exactly one repository so sibling isolation can be observed directly."""

    def __init__(self, failing_repository_id: str) -> None:
        """Record which repository must fail and which repositories were executed."""
        self._delegate = MockChildWorkstreamExecutor()
        self._failing_repository_id = failing_repository_id
        self.calls: list[str] = []

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Return an approved result for every repository except the designated failure."""
        execution = await self._delegate.run(**kwargs)
        repository_id = kwargs["repository"].repository_id
        self.calls.append(repository_id)
        if repository_id != self._failing_repository_id:
            return execution
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    "blocking_issues": ["The repository review rejected this attempt."],
                    "pull_request_readiness": False,
                }
            ),
            code_completion=execution.code_completion,
            review=execution.review,
        )


class RewordsTheSameDefectExecutor:
    """Fail on one defect for ever, describing it differently on every attempt.

    What a coding model that cannot fix something actually does. AB-Feature-110's console
    reported the same missing table selection six times over, phrased anew each time.
    """

    def __init__(self, repository_id: str) -> None:
        """Record the repository to fail and count how many attempts it was given."""
        self._delegate = MockChildWorkstreamExecutor()
        self._repository_id = repository_id
        self.attempts = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Reject the attempt with the same defect, worded to defeat a string comparison."""
        execution = await self._delegate.run(**kwargs)
        if kwargs["repository"].repository_id != self._repository_id:
            return execution
        self.attempts += 1
        wording = [
            "TypeError: undefined is not iterable at app.route.js:117 while destructuring.",
            "At app.route.js:117 the destructuring raises TypeError: undefined is not iterable.",
            "Each failure is a TypeError: undefined is not iterable, at app.route.js:117.",
            "The route fails with TypeError: undefined is not iterable (app.route.js:117).",
        ][(self.attempts - 1) % 4]
        # Genuinely different production source every attempt, which is the case that
        # matters: the change is real, so the budget guard permits another attempt, and only
        # the diagnostics reveal that nothing is actually being fixed. The paths have to
        # differ, not merely their content, because without a workspace to read the mock
        # path fingerprints what was reported changed rather than what it now says.
        touched = [
            "server/routes/app.route.js",
            f"server/services/bulkDelete{self.attempts}.service.js",
        ]
        completion = execution.code_completion
        assert completion is not None
        completion = completion.model_copy(
            update={
                "file_changes": [
                    FileChange(
                        path=path,
                        change_type="modified",
                        description=f"rewrite {self.attempts}",
                    )
                    for path in touched
                ],
                "production_files_changed": touched,
                # No commit: a pre-commit rejection resets the workspace, which is the shape
                # AB-Feature-108's attempts actually had. The loop exempts that case from the
                # identical-source guard as long as the diagnostics differ -- and reworded
                # diagnostics always differ as strings, which is the hole being closed.
                "commit_sha": None,
            }
        )
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    "blocking_issues": [wording],
                    "pull_request_readiness": False,
                    "production_files_changed": touched,
                    "failure_classification": "validation_source_failure",
                    # The runtime rebinds the child's fingerprints from the workspace before
                    # building the result, so an attempt that wrote different source reports
                    # a different fingerprint. Stated here because the mock has no workspace
                    # to read: without it every attempt carried the previous attempt's value
                    # and was compared against itself, which reads as a reproduction of a
                    # change that is genuinely new -- the opposite of what this test is about.
                    "production_diff_fingerprint": f"production-rewrite-{self.attempts}",
                    "test_diff_fingerprint": "tests-unchanged",
                }
            ),
            code_completion=completion,
            review=execution.review,
        )


@pytest.mark.asyncio
async def test_a_workstream_stopped_for_not_converging_does_not_invite_another_attempt() -> None:
    """AB-Feature-110 ended by telling its reader "Attempt 7 may proceed with a changed strategy".

    Nothing was going to proceed: the loop had just stopped the workstream because its
    diagnostics came back unchanged. The refusal is recorded before that decision is taken,
    from a budget check that only ever answers "is there an attempt left", so a feature that
    stopped for a different reason kept an answer belonging to the question nobody asked.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-not-converging", request)
    # The budget AB-Feature-108 actually ran with. Generous on purpose: the point is that a
    # workstream which is not converging stops early rather than spending all eight.
    state.max_validation_retries = 8
    executor = RewordsTheSameDefectExecutor("backend")
    orchestrator = FeatureWorkflowOrchestrator(child_executor=executor)

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    child = result.child_workflows["backend"]
    assert child.status is ChildWorkflowStatus.FAILED
    # Stopped well short of the ceiling, because rewording one defect is not converging.
    assert executor.attempts < 8
    assert child.retry_refusal_reason is not None
    assert "not converging" in child.retry_refusal_reason
    # This is the field the durable failure summary publishes as `next_action`, which is what
    # -110 showed a reader beside a dead feature. Asserting it here asserts that.
    assert "may proceed" not in child.retry_refusal_reason


class ConcurrencyProbeExecutor:
    """Record the highest number of repositories implemented at the same moment."""

    def __init__(self) -> None:
        """Track live and peak concurrency across every child invocation."""
        self._delegate = MockChildWorkstreamExecutor()
        self._live = 0
        self.peak = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Hold the slot briefly so genuinely concurrent children overlap in time."""
        self._live += 1
        self.peak = max(self.peak, self._live)
        try:
            await asyncio.sleep(0.01)
            return await self._delegate.run(**kwargs)
        finally:
            self._live -= 1


@pytest.mark.asyncio
async def test_a_failed_repository_does_not_starve_independent_siblings() -> None:
    """One rejected repository must never prevent unrelated repositories from finishing.

    This is the exact production failure: the backend failed review and the frontend was
    left `pending` across ten consecutive live runs, so no pull request was ever created.
    """
    payload = feature_payload()
    payload["repositories"] = [
        _repository("backend", "backend"),
        _repository("frontend", "frontend"),
        _repository("worker", "service"),
    ]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-sibling-isolation", request)
    executor = FailOneRepositoryExecutor("backend")
    orchestrator = FeatureWorkflowOrchestrator(child_executor=executor)

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert set(executor.calls) == {"backend", "frontend", "worker"}
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.FAILED
    # Both siblings ran to a passing review and are offered for publication. The feature did
    # not land, so nothing opened on its own -- but the sibling that passed is not starved
    # either, which is the production failure this test is about.
    assert not _published_repository_ids(result)
    for repository_id in ("frontend", "worker"):
        assert (
            require_publishable_workstream(result, repository_id=repository_id)
            is WorkstreamPublicationClass.REVIEWED
        )
    published = await _publish_on_request(result)
    # Completed rather than merely approved: a sibling that passes review has its pull
    # request opened, even though the feature as a whole still needs a human.
    assert published.child_workflows["frontend"].status is ChildWorkflowStatus.COMPLETED
    assert published.child_workflows["worker"].status is ChildWorkflowStatus.COMPLETED
    assert published.child_workflows["backend"].status is ChildWorkflowStatus.FAILED


@pytest.mark.asyncio
async def test_empty_resume_does_not_replay_a_persisted_terminal_retry_refusal() -> None:
    """A restart cannot run an executor before consulting the prior terminal decision."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-terminal-retry", request)
    executor = FailOneRepositoryExecutor("backend")
    orchestrator = FeatureWorkflowOrchestrator(child_executor=executor)
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    failed = await orchestrator.start(state, credentials=credentials)
    calls_before_resume = list(executor.calls)

    resumed = await orchestrator.resume(failed, answers=[], credentials=credentials)

    assert resumed.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert resumed.child_workflows["backend"].retry_refusal_reason
    assert executor.calls == calls_before_resume


@pytest.mark.asyncio
async def test_optional_repository_failure_does_not_block_required_pull_request() -> None:
    """An optional child may fail visibly without making the required feature undeliverable."""
    payload = feature_payload()
    repositories = payload["repositories"]
    assert isinstance(repositories, list)
    repositories[1] = {**repositories[1], "required": False}
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-optional-failure", request)
    orchestrator = FeatureWorkflowOrchestrator(child_executor=FailOneRepositoryExecutor("frontend"))

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.COMPLETED
    assert result.child_workflows["frontend"].status is ChildWorkflowStatus.FAILED
    completion = next(
        artifact for artifact in result.artifacts if isinstance(artifact, FeatureCompletionArtifact)
    )
    assert len(completion.pull_request_urls) == 1
    assert any("Optional repository 'frontend'" in item for item in completion.known_limitations)


@pytest.mark.asyncio
async def test_only_dependents_are_blocked_when_a_shared_library_fails() -> None:
    """A real build-artifact failure blocks its consumers and nothing else."""
    payload = feature_payload()
    payload["repositories"] = [
        _repository("shared-sdk", "shared"),
        _repository("backend", "backend"),
        _repository("docs", "documentation_site", role="other"),
    ]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-blocked-dependent", request)
    executor = FailOneRepositoryExecutor("shared-sdk")
    orchestrator = FeatureWorkflowOrchestrator(child_executor=executor)

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    # Consumers of the failed library are reported as blocked rather than silently pending.
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.BLOCKED
    assert result.child_workflows["docs"].status is ChildWorkflowStatus.BLOCKED
    assert "shared-sdk" in result.child_workflows["backend"].blocking_issues[0]
    # The library exhausts its own retry budget; neither consumer is ever executed.
    assert set(executor.calls) == {"shared-sdk"}


@pytest.mark.asyncio
async def test_workstream_fan_out_respects_the_configured_concurrency_limit() -> None:
    """Unbounded fan-out would let one feature exhaust the worker for every other feature."""
    payload = feature_payload()
    payload["repositories"] = [_repository(f"service-{index}", "service") for index in range(5)]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-concurrency", request)
    executor = ConcurrencyProbeExecutor()
    orchestrator = FeatureWorkflowOrchestrator(child_executor=executor, max_parallel_workstreams=2)

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.COMPLETED
    assert executor.peak == 2


def test_a_workstream_dependency_cycle_is_rejected_before_any_repository_is_touched() -> None:
    """A cyclic graph must fail loudly instead of deadlocking the fan-out loop."""
    workstreams = {
        "a": _workstream_plan("a", ["b"]),
        "b": _workstream_plan("b", ["a"]),
    }

    with pytest.raises(FeatureWorkflowError, match="dependency cycle"):
        _require_acyclic_workstreams(workstreams)


class RaisingChildExecutor:
    """Raise an unexpected fault for one repository and succeed for the rest."""

    def __init__(self, failing_repository_id: str) -> None:
        """Record which repository raises and delegate the others."""
        self._delegate = MockChildWorkstreamExecutor()
        self._failing_repository_id = failing_repository_id

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Raise a non-cancellation error for the designated repository."""
        if kwargs["repository"].repository_id == self._failing_repository_id:
            msg = "git commit failed: command failed"
            raise RuntimeError(msg)
        return await self._delegate.run(**kwargs)


class ProviderFaultThenApproveExecutor:
    """Raise an infrastructure fault for the backend a fixed number of times, then approve."""

    def __init__(self, *, faults: int, error: type[Exception] = LLMAdapterError) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self._remaining = faults
        self._error = error
        self.attempts = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        if kwargs["repository"].repository_id != "backend":
            return await self._delegate.run(**kwargs)
        self.attempts += 1
        if self._remaining > 0:
            self._remaining -= 1
            msg = "infrastructure fault"
            raise self._error(msg)
        return await self._delegate.run(**kwargs)


class SourceValidationRetryExecutor:
    """Fail one backend attempt with safe lint output, then capture retry feedback."""

    def __init__(self) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self.feedback: list[list[str]] = []

    async def run(self, **kwargs: Any) -> ChildExecution:
        if kwargs["repository"].repository_id != "backend":
            return await self._delegate.run(**kwargs)
        self.feedback.append(list(kwargs["feedback"]))
        if len(self.feedback) == 1:
            raise SourceValidationError(
                ["Pre-commit source validation `eslint src/tile.js` exited with code 1"]
            )
        return await self._delegate.run(**kwargs)


class OneRepositoryFailsExecutor:
    """Approve the frontend through the real mock path and fail the backend outright."""

    def __init__(self) -> None:
        self._delegate = MockChildWorkstreamExecutor()

    async def run(self, **kwargs: Any) -> ChildExecution:
        execution = await self._delegate.run(**kwargs)
        if kwargs["repository"].repository_id != "backend":
            return execution
        failed = execution.result.model_copy(
            update={
                "status": "failed",
                "blocking_issues": ["The backend could not satisfy its scoped requirement."],
                "pull_request_readiness": False,
                "production_diff_fingerprint": "backend-unfinished",
            }
        )
        return ChildExecution(result=failed, code_completion=execution.code_completion)


@pytest.mark.asyncio
async def test_a_repository_that_passed_review_is_published_when_its_sibling_fails() -> None:
    """Reviewed work is published on its own rather than discarded with the feature.

    A single failed required repository returned before the publication step, so anything
    already approved went unpublished: r-2 finished and approved its frontend and opened
    nothing because the backend was still failing, and r-3 did the same with the sides
    reversed. Half a feature, reviewed and open for a human, is worth more than none of it.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("partial-publication", request)

    result = await FeatureWorkflowOrchestrator(child_executor=OneRepositoryFailsExecutor()).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    # Nothing publishes itself: the backend never landed, so a frontend pull request opened
    # unasked would be half a feature nobody chose to review.
    assert not _published_repository_ids(result)
    assert publication_is_held(result)
    opened = await _publish_on_request(result)

    published = [
        artifact.repository
        for artifact in opened.artifacts
        if isinstance(artifact, PullRequestArtifact)
    ]
    assert published == ["example/frontend"]
    assert opened.child_workflows["frontend"].pull_request_artifact_id is not None
    assert opened.child_workflows["backend"].pull_request_artifact_id is None
    # The feature is still unfinished, and says so.
    assert opened.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN


class RepeatedDeterministicGateExecutor:
    """Fail twice with one fixed gate sentence, then let the delegate succeed."""

    def __init__(self, *, deterministic: bool) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self._deterministic = deterministic
        self.attempts = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        if kwargs["repository"].repository_id != "backend":
            return await self._delegate.run(**kwargs)
        self.attempts += 1
        execution = await self._delegate.run(**kwargs)
        if self.attempts > 2:
            return execution
        metadata = dict(execution.result.metadata)
        if self._deterministic:
            metadata["deterministic_gate"] = True
        failed = execution.result.model_copy(
            update={
                "status": "failed",
                # Byte-identical on every attempt, which is what a deterministic check does.
                "blocking_issues": ["src/tile.js was added but no production file refers to it."],
                "pull_request_readiness": False,
                "metadata": metadata,
                # Real production change, distinct per attempt, so the loop-prevention
                # guard is satisfied and the convergence rule is the only thing under test.
                "production_files_changed": ["src/tile.js"],
                "production_diff_fingerprint": f"attempt-{self.attempts}",
            }
        )
        return ChildExecution(result=failed, code_completion=execution.code_completion)


@pytest.mark.asyncio
async def test_a_deterministic_gate_repeating_itself_still_gets_its_retries() -> None:
    """A fixed gate sentence is not evidence that the attempt ignored its feedback."""
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-deterministic-gate", request)
    executor = RepeatedDeterministicGateExecutor(deterministic=True)

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert executor.attempts == 3
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED


@pytest.mark.asyncio
async def test_a_model_repeating_its_own_findings_still_stops_early() -> None:
    """The convergence rule must keep working for the reviewer prose it was written for."""
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-non-converging", request)
    executor = RepeatedDeterministicGateExecutor(deterministic=False)

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert executor.attempts == 2
    assert result.child_workflows["backend"].status is not ChildWorkflowStatus.COMPLETED


# --------------------------------------------------------------------------------------
# 86- Part A: the marker names the check that spoke, not the gate it came from
# --------------------------------------------------------------------------------------

_LINT_REJECTION = (
    "Pre-commit source validation failed (tool=eslint): "
    "src/routes/status.js:14:7  error  'formatStatus' is not defined  no-undef"
)
_REACHABILITY_REJECTION = (
    f"{REACHABILITY_DIAGNOSTIC_PREFIX}: src/routes/status_formatter.js is not referred to "
    "by any production file in this repository."
)


def _scoped_test_rejection() -> str:
    """AB-Feature-218's backend attempt 2, as its own producer composed it.

    Built with `scoped_test_diagnostic` rather than as a literal, because the predicate that
    identifies these matches on the producer's constant and a hand-written sentence would let
    the two drift apart silently -- which is the whole complaint the prefix constant answers.
    """
    return scoped_test_diagnostic(
        ValidationResult(
            command=("npm", "run", "test", "--", "server/tests/bulk-create-apps.service.test.js"),
            return_code=1,
            stdout="",
            stderr=(
                "  * bulk create apps > rejects a row missing app_title\n"
                "    expected BULK_APP_VALIDATION_FAILED (400), received "
                "BULK_APP_TRANSACTION_FAILED (500)\n"
            ),
            timed_out=False,
            duration_seconds=64.0,
            validation_type="test",
            result_code="TEST_VALIDATION_FAILED_EXIT_1",
        )
    )


def test_a_lint_only_rejection_keeps_todays_exemption_unchanged() -> None:
    """The half that must not move: a linter repeating itself is the tool being consistent."""
    classification, metadata = source_rejection_disposition(
        SourceValidationError([_LINT_REJECTION])
    )

    assert classification == FailureClassification.VALIDATION_SOURCE_FAILURE.value
    assert metadata == {"deterministic_gate": True, "source_validation_rejected": True}


def test_218s_backend_attempt_2_is_not_a_deterministic_gate() -> None:
    """A test failing identically across attempts is not a tool being consistent.

    AB-Feature-218's backend attempts 2 and 3 were commit-gate rejections carrying
    `deterministic_gate: true`, and with it four of five convergence guards were off: the
    non-converging refusal, the resolved-issue ledger, the recurring-demand theme and the
    unchanged-failure run. The suite failed on attempts 2, 3, 4 and 5 on one defect and
    nothing stopped the workstream.

    `source_validation_rejected` stays true, because it is a fact about where the attempt was
    stopped -- before the commit, by the gate -- and that is what the pre-commit retry
    allowance is about. 83- established the two must not be re-conflated.
    """
    classification, metadata = source_rejection_disposition(
        SourceValidationError([_scoped_test_rejection()])
    )

    assert classification == FailureClassification.VALIDATION_SOURCE_FAILURE.value
    assert not metadata.get("deterministic_gate")
    assert metadata["source_validation_rejected"] is True


def test_a_lint_error_riding_along_does_not_exempt_a_failing_test() -> None:
    """A mixed rejection is not exempt, in either order the diagnostics arrive.

    The failing test is the part that says something about convergence, and treating the pair
    as exempt because a lint error rode along is how the exemption became universal.
    """
    scoped = _scoped_test_rejection()

    for diagnostics in (
        [_LINT_REJECTION, scoped],
        [scoped, _LINT_REJECTION],
        [_REACHABILITY_REJECTION, scoped],
    ):
        _classification, metadata = source_rejection_disposition(SourceValidationError(diagnostics))
        assert not metadata.get("deterministic_gate"), diagnostics
        assert metadata["source_validation_rejected"] is True


def test_a_self_review_rejection_carries_neither_marker() -> None:
    """Unchanged by Part A: every configured check accepted the source here."""
    error = SourceValidationError(["The self-review found the assignment unmet."])
    error.terminal_outcome = SELF_REVIEW_SUBSTANTIVE_OUTCOME

    classification, metadata = source_rejection_disposition(error)

    assert classification == FailureClassification.IMPLEMENTATION_MISSING.value
    assert not metadata.get("deterministic_gate")
    assert not metadata.get("source_validation_rejected")
    assert metadata["self_review_rejected"] is True


def test_the_052b_guard_a_reachability_sentence_repeated_is_still_exempt() -> None:
    """The -052b guard. Nothing in Part A may narrow the exemption for an inspection.

    Feature -052b lost both workstreams two attempts in because the reachability gate said the
    same true thing twice while the attempts around it were changing. A wiring finding is a
    deterministic inspection whose sentence is fixed for as long as the module is
    unreferenced; no model produced it, and no rewriting of a test makes it go away.

    Asserted twice over: the marker survives on its own, and it survives beside the formatting
    and typecheck sentences it shares the gate with. The lineage-level half of this guard --
    that such a workstream really does get its retries and reach its approval -- is
    `test_l3_the_052b_lineage_is_not_stopped` in `test_convergence_composition.py`.
    """
    _classification, metadata = source_rejection_disposition(
        SourceValidationError([_REACHABILITY_REJECTION])
    )
    assert metadata["deterministic_gate"] is True

    _classification, mixed = source_rejection_disposition(
        SourceValidationError(
            [_REACHABILITY_REJECTION, _LINT_REJECTION, f"{TYPECHECK_DIAGNOSTIC_PREFIX}: 3 errors."]
        )
    )
    assert mixed["deterministic_gate"] is True


class ReviewRetryThenApproveExecutor:
    """Return two complete engineer/reviewer attempts so lineage can be inspected."""

    def __init__(self) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self.attempts = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        self.attempts += 1
        execution = await self._delegate.run(**kwargs)
        if self.attempts > 1:
            return execution
        assert execution.review is not None
        rejected_review = execution.review.model_copy(update={"verdict": "changes_requested"})
        rejected_result = execution.result.model_copy(
            update={
                "status": "failed",
                "blocking_issues": ["The first review requested a targeted correction."],
                "pull_request_readiness": False,
                # The reviewer is responding to an actual production change. Without this,
                # the retry would have no changed implementation input and must be refused.
                "production_files_changed": ["src/login.py"],
                "production_diff_fingerprint": "first-reviewable-change",
            }
        )
        return ChildExecution(
            result=rejected_result,
            code_completion=execution.code_completion,
            review=rejected_review,
        )


class ManualReviewRequiredExecutor:
    """Return terminal incomplete evidence that must never re-enter the coding model."""

    def __init__(self) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self.attempts = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        self.attempts += 1
        execution = await self._delegate.run(**kwargs)
        assert execution.review is not None
        review = execution.review.model_copy(
            update={
                "verdict": "rejected",
                "metadata": {
                    **execution.review.metadata,
                    "manual_review_required": True,
                    "retryable": False,
                },
            }
        )
        result = execution.result.model_copy(
            update={
                "status": "failed",
                "blocking_issues": ["Workspace evidence exceeded the bounded review policy."],
                "pull_request_readiness": False,
                "metadata": {
                    **execution.result.metadata,
                    "retry_refusal_reason": "manual_review_required",
                },
            }
        )
        return ChildExecution(
            result=result,
            code_completion=execution.code_completion,
            review=review,
        )


class CommitThenFailExecutor:
    """Commit and push on the first attempt, then fail every later one.

    This is what live feature -021 did: an attempt reached the remote, a later attempt was
    rejected, and the result artifact described only the rejection.
    """

    def __init__(self) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self.attempts = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        self.attempts += 1
        execution = await self._delegate.run(**kwargs)
        failed = execution.result.model_copy(
            update={
                "status": "failed",
                "blocking_issues": ["`npm run lint` exited with code 1"],
            }
        )
        if self.attempts == 1:
            # Committed and pushed, then rejected by a later stage.
            return ChildExecution(result=failed, code_completion=execution.code_completion)
        return ChildExecution(
            result=failed.model_copy(update={"code_completion_artifact_id": None}),
            code_completion=None,
        )


_SCHEMA_REJECTION_SECRET = "github_pat_schema_rejection_secret"


class SchemaRejectingExecutor:
    """Fail one repository the way a rejected model response actually fails."""

    def __init__(self) -> None:
        self._delegate = MockChildWorkstreamExecutor()

    async def run(self, **kwargs: Any) -> ChildExecution:
        if kwargs["repository"].repository_id != "backend":
            return await self._delegate.run(**kwargs)
        ReviewArtifact.model_validate(
            {"workflow_id": "", _SCHEMA_REJECTION_SECRET: _SCHEMA_REJECTION_SECRET}
        )
        raise AssertionError("validation should have rejected the payload")


class UnverifiablePullRequestService(MockGitHubService):
    """Create pull requests normally, then be unable to find any of them again."""

    def find_pull_request(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return None


class MismatchedHeadPullRequestService(MockGitHubService):
    """Return a PR whose branch was advanced away from the reviewed commit."""

    def find_pull_request(self, *args: Any, **kwargs: Any) -> PullRequestDetails | None:
        found = super().find_pull_request(*args, **kwargs)
        if found is None:
            return None
        return PullRequestDetails(
            repository=found.repository,
            number=found.number,
            url=found.url,
            title=found.title,
            source_branch=found.source_branch,
            target_branch=found.target_branch,
            head_sha="different-remote-head",
        )


@pytest.mark.asyncio
async def test_a_feature_is_not_completed_until_its_pull_requests_are_verified() -> None:
    """A completion artifact asserts that pull requests exist, so that must be checked.

    Thirty live runs reached no pull request at all, and the completion path trusted the
    create call's own report; a publisher that failed silently would have been recorded as a
    finished feature carrying URLs nobody had ever fetched.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-unverified-pr", request)
    orchestrator = FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(
            github_service=UnverifiablePullRequestService()
        )
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert not any(isinstance(artifact, FeatureCompletionArtifact) for artifact in result.artifacts)
    issues = [
        item
        for child in result.child_workflows.values()
        for item in child.blocking_issues
        if "could not be fetched back" in item
    ]
    assert issues, result.child_workflows


@pytest.mark.asyncio
async def test_a_feature_is_not_completed_when_a_pull_request_head_changed() -> None:
    """Fetch-back must prove the PR serves the exact reviewed and published commit."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-mismatched-pr-head", request)
    orchestrator = FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(
            github_service=MismatchedHeadPullRequestService()
        )
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert not any(isinstance(artifact, FeatureCompletionArtifact) for artifact in result.artifacts)
    assert any(
        "not the approved commit" in issue
        for child in result.child_workflows.values()
        for issue in child.blocking_issues
    )


class CommentRejectingService(MockGitHubService):
    """Create pull requests normally, then refuse to add the cross-link comment."""

    def add_comment(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("comments are disabled for this repository")


class ForbiddenCommentService(MockGitHubService):
    """Refuse the cross-link the way the live adapter reports a refused comment."""

    diagnostics = (
        "pull-request comment refused (GithubException status=403) on "
        "POST /repos/example/backend/issues/1/comments",
        "likely cause: the credential may not write pull-request comments",
    )

    def add_comment(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise GitHubAdapterError(
            "pull-request comment could not be posted (status 403)",
            diagnostics=self.diagnostics,
        )


class RecordingLogger:
    """Capture the structured events the publisher emits, with no logging configured."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **fields: Any) -> None:
        """Record one structured warning."""
        self.events.append((event, fields))

    def info(self, event: str, **fields: Any) -> None:
        """Record nothing; only warnings are under test."""
        del event, fields


@pytest.mark.asyncio
async def test_a_skipped_cross_link_comment_logs_one_warning_naming_the_likely_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ten for ten of these comments have failed, and every record named only a type.

    Skipping the comment is correct -- the pull requests exist and the review approved
    them -- but the skip has to be legible: one warning per pull request, carrying the
    status and likely cause the adapter classified, and no stack trace.
    """
    recorder = RecordingLogger()
    monkeypatch.setattr(feature_workflow_module, "_LOGGER", recorder)
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-forbidden-cross-link", request)
    orchestrator = FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(github_service=ForbiddenCommentService())
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.COMPLETED
    skips = [
        fields for event, fields in recorder.events if event == "pull_request_cross_link_failed"
    ]
    assert len(skips) == 2
    for fields in skips:
        assert fields["error_type"] == "GitHubAdapterError"
        assert fields["diagnostics"] == list(ForbiddenCommentService.diagnostics)
        assert "traceback" not in fields


@pytest.mark.asyncio
async def test_the_cross_link_comment_is_built_from_the_published_pull_requests() -> None:
    """Nothing in the comment may come from an assumed repository or URL shape.

    The platform is pointed at whatever repositories a feature names, so the body has to
    read each published pull request's own repository and provider-returned URL.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-cross-link-body", request)
    service = MockGitHubService()
    orchestrator = FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(github_service=service)
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.COMPLETED
    published = [
        artifact for artifact in result.artifacts if isinstance(artifact, PullRequestArtifact)
    ]
    assert len(published) == 2
    assert service.comments, "the cross-link comment was never attempted"
    for artifact in published:
        posted = service.comments[(artifact.repository, artifact.pull_request_number)]
        assert len(posted) == 1
        for sibling in published:
            assert f"- {sibling.repository}: {sibling.url}" in posted[0]


@pytest.mark.asyncio
async def test_a_failed_cross_link_comment_does_not_fail_a_finished_feature() -> None:
    """Feature -051 created both pull requests, passed integration review, and was recorded
    as failed because a convenience comment could not be posted.

    The cross-link helps a human reader find sibling pull requests. Every pull request
    already exists by that point, so it cannot decide whether the feature succeeded.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-comment-failure", request)
    orchestrator = FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(github_service=CommentRejectingService())
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.COMPLETED
    completion = next(
        artifact for artifact in result.artifacts if isinstance(artifact, FeatureCompletionArtifact)
    )
    assert len(completion.pull_request_urls) == 2
    child_completion_ids = {
        artifact.artifact_id
        for artifact in result.artifacts
        if isinstance(artifact, CodeCompletionArtifact)
    }
    assert child_completion_ids == {
        "006_code_completion.backend.attempt-0.json",
        "006_code_completion.frontend.attempt-0.json",
    }


@pytest.mark.asyncio
async def test_a_feature_completes_when_every_pull_request_is_verifiable() -> None:
    """The guard must not block a genuinely finished feature."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-verified-pr", request)
    orchestrator = FeatureWorkflowOrchestrator()

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.COMPLETED
    completion = next(
        artifact for artifact in result.artifacts if isinstance(artifact, FeatureCompletionArtifact)
    )
    assert len(completion.pull_request_urls) == 2


@pytest.mark.asyncio
async def test_reconnaissance_reaches_the_planner_before_the_plan_is_frozen() -> None:
    """The plan must be made from what the checkouts contain, not from the repository URL.

    Everything the plan freezes -- scope, acceptance criteria, expected source areas, the
    contract -- was previously decided before anything had read a repository, which is how a
    requirement resting on a convention that does not exist became a workstream no attempt
    could finish.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-recon-order", request)
    reconnaissance = RecordingReconnaissance()
    planner = RecordingPlanner()
    orchestrator = FeatureWorkflowOrchestrator(
        reconnaissance=cast(Any, reconnaissance), planner=cast(Any, planner)
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert reconnaissance.inspected == ["backend", "frontend"]
    assert [item.repository_id for item in planner.received] == ["backend", "frontend"]
    # And the evidence is durable, so an operator can tell a grounded plan from a blind one.
    published = [
        artifact
        for artifact in result.artifacts
        if isinstance(artifact, RepositoryReconnaissanceArtifact)
    ]
    assert [item.repository_id for item in published] == ["backend", "frontend"]


@pytest.mark.asyncio
async def test_a_premise_the_checkout_contradicts_is_asked_before_anything_is_planned() -> None:
    """The question a human is asked must be the one only the repository could raise.

    Clarification used to run before anything had read a repository, so the answers were
    themselves assumptions -- "use the existing X" where no X existed. Those answers became
    requirements the reviewer enforced faithfully and no attempt could satisfy.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-premise-question", request)
    planner = RecordingPlanner()
    reconnaissance = RecordingReconnaissance(
        contradicted_premises={
            "backend": [
                {
                    "premise": "Authentication can be applied per route.",
                    "contradicted_by": "Authentication is applied once, globally.",
                    "evidence_paths": ["server/config/express.js"],
                    "question": "Rely on the global middleware, or introduce per-route auth?",
                }
            ]
        }
    )
    orchestrator = FeatureWorkflowOrchestrator(
        reconnaissance=cast(Any, reconnaissance), planner=cast(Any, planner)
    )

    waiting = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert waiting.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
    # Nothing was planned against the contradicted premise.
    assert planner.received == []
    technical_prd = next(
        artifact
        for artifact in reversed(waiting.artifacts)
        if isinstance(artifact, TechnicalPRDArtifact)
    )
    question = next(
        item for item in technical_prd.unresolved_questions if item.question_id == "recon-backend-1"
    )
    assert question.question == "Rely on the global middleware, or introduce per-route auth?"
    # The evidence travels with the question, so the human can go and check it.
    assert "server/config/express.js" in question.rationale
    assert "applied once, globally" in question.rationale


@pytest.mark.asyncio
async def test_answering_a_premise_question_lets_the_feature_plan_and_finish() -> None:
    """The new gate must be a question, not a dead end."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-premise-answered", request)
    reconnaissance = RecordingReconnaissance(
        contradicted_premises={
            "backend": [
                {
                    "premise": "Authentication can be applied per route.",
                    "contradicted_by": "Authentication is applied once, globally.",
                    "evidence_paths": ["server/config/express.js"],
                    "question": "Rely on the global middleware, or introduce per-route auth?",
                }
            ]
        }
    )
    orchestrator = FeatureWorkflowOrchestrator(reconnaissance=cast(Any, reconnaissance))
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)

    waiting = await orchestrator.start(state, credentials=credentials)
    result = await orchestrator.resume(
        waiting,
        answers=[
            ClarificationAnswer(
                question_id="recon-backend-1", answer="Rely on the global middleware."
            )
        ],
        credentials=credentials,
    )

    assert result.status is FeatureWorkflowStatus.COMPLETED
    # Answered once and not asked again: the checkouts were read before the question, so
    # re-reading them on resume would restate facts the feature already holds.
    assert reconnaissance.inspected == ["backend", "frontend"]
    technical_prd = next(
        artifact
        for artifact in reversed(result.artifacts)
        if isinstance(artifact, TechnicalPRDArtifact)
    )
    assert technical_prd.unresolved_questions == []
    assert technical_prd.metadata["clarification_answers"] == {
        "recon-backend-1": "Rely on the global middleware."
    }


@pytest.mark.asyncio
async def test_a_stopped_workstream_hands_its_operator_a_question() -> None:
    """A workstream that gives up must say what decision is now the human's.

    Blocking issues are what reach the parent, the API and the console, so the question is
    recorded there rather than only in metadata a person would have to go looking for.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-operator-question", request)
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, FailOneRepositoryExecutor("backend"))
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    backend = result.child_workflows["backend"]
    assert backend.status is ChildWorkflowStatus.FAILED
    question = next(item for item in backend.blocking_issues if item.endswith("?"))
    # This executor never writes production source, so the question is the one that fact
    # supports -- not a menu of three things the operator has to choose between.
    assert "no production source change at all" in question
    child_result = next(
        artifact
        for artifact in reversed(result.artifacts)
        if isinstance(artifact, ChildWorkflowResultArtifact) and artifact.repository_id == "backend"
    )
    assert child_result.metadata["terminal_cause"] == "never_implemented"
    assert child_result.metadata["terminal_evidence"]
    # The repository that succeeded is not an escalation and is asked nothing.
    assert result.child_workflows["frontend"].blocking_issues == []


@pytest.mark.asyncio
async def test_revising_a_revised_artifact_keeps_its_lineage_one_level_deep() -> None:
    """A revision of a revision must stay findable by the children that need it.

    Live feature -082 died here: reconnaissance revised the technical PRD to add its
    questions, answering them revised that revision, and the id became
    `002_technical_prd.revision-2.revision-3.json`. `artifact_id_matches_lineage` accepts one
    qualifier by design, so both repositories failed with "required artifact is missing:
    002_technical_prd.json" before either wrote a line. Nothing revised the technical PRD
    twice until reconnaissance started asking first.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-double-revision", request)
    original = create_artifact(
        TechnicalPRDArtifact,
        workflow_id=state.feature_id,
        artifact_id="002_technical_prd.json",
        producer="product_manager",
        metadata={},
        payload=_technical_prd_payload(),
    )
    state.artifacts.append(original)

    _replace_artifact(state, original, original)
    first = state.artifacts[-1]
    _replace_artifact(state, first, first)
    second = state.artifacts[-1]

    assert first.artifact_id == "002_technical_prd.revision-2.json"
    assert second.artifact_id == "002_technical_prd.revision-3.json"
    for artifact in (first, second):
        assert artifact_id_matches_lineage(artifact.artifact_id, "002_technical_prd.json")


@pytest.mark.asyncio
async def test_a_criterion_no_change_can_demonstrate_is_asked_about_before_any_code() -> None:
    """An unmeetable acceptance criterion is raised with its author, not silently dropped.

    The reviewer withholds these so an attempt is not failed for missing a measurement it
    cannot take. Withholding alone decided on the author's behalf and told them nothing, so a
    requirement someone wrote left the judgement without anyone knowing.
    """
    payload = feature_payload()
    prd = payload["prd"]
    assert isinstance(prd, dict)
    requirements = prd["requirements"]
    assert isinstance(requirements, list)
    requirements[0] = {
        **requirements[0],
        "requirement_id": "NFR-1",
        "acceptance_criteria": [
            "The p95 latency stays under 200ms in a staging environment.",
            "The endpoint performs only cheap read-only work.",
        ],
    }
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-unverifiable-criteria", request)
    orchestrator = FeatureWorkflowOrchestrator()

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
    technical_prd = next(
        artifact
        for artifact in reversed(result.artifacts)
        if isinstance(artifact, TechnicalPRDArtifact)
    )
    questions = {item.question_id: item for item in technical_prd.unresolved_questions}
    assert set(questions) == {"criteria-NFR-1-1"}
    # The criterion is quoted back, so the author can see exactly what was unmeetable -- and
    # the criterion that a diff *can* demonstrate is left alone.
    assert "p95 latency" in questions["criteria-NFR-1-1"].question
    assert "cheap read-only work" not in questions["criteria-NFR-1-1"].question


@pytest.mark.asyncio
async def test_answering_a_criteria_question_does_not_ask_it_again() -> None:
    """A question re-asked after it was answered is a feature that never leaves the gate.

    The criteria check reads the technical PRD, which still holds the original criterion after
    an answer is recorded, so a gate keyed on anything but 'already asked' would loop forever.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-criteria-answered", request)
    state.status = FeatureWorkflowStatus.WAITING_FOR_HUMAN
    prd_payload = _technical_prd_payload()
    prd_payload["unresolved_questions"] = [
        {
            "question_id": "criteria-NFR-1-1",
            "question": "Restate the latency criterion.",
            "rationale": "No diff can demonstrate a percentile.",
            "required": True,
        }
    ]
    prd_payload["non_functional_requirements"] = [
        {
            "requirement_id": "NFR-1",
            "description": "The status endpoint stays responsive under load.",
            "priority": "must",
            "acceptance_criteria": ["The p95 latency stays under 200ms."],
            "dependencies": [],
        }
    ]
    state.artifacts.append(
        create_artifact(
            TechnicalPRDArtifact,
            workflow_id=state.feature_id,
            artifact_id="002_technical_prd.json",
            producer="product_manager",
            metadata={},
            payload=prd_payload,
        )
    )
    orchestrator = FeatureWorkflowOrchestrator()

    result = await orchestrator.resume(
        state,
        answers=[
            ClarificationAnswer(
                question_id="criteria-NFR-1-1",
                answer="The handler does no blocking work; latency is measured separately.",
            )
        ],
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
    )

    assert result.status is FeatureWorkflowStatus.COMPLETED
    technical_prd = next(
        artifact
        for artifact in reversed(result.artifacts)
        if isinstance(artifact, TechnicalPRDArtifact)
    )
    assert technical_prd.unresolved_questions == []


@pytest.mark.asyncio
async def test_a_feature_that_already_read_its_repositories_does_not_read_them_again() -> None:
    """Answering a clarification re-enters planning; the checkouts have not moved since.

    Re-inspecting would spend one model call per repository restating facts the feature
    already recorded, and would let two plans in the same feature disagree about the same
    checkout.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-recon-reuse", request)
    state.status = FeatureWorkflowStatus.WAITING_FOR_HUMAN
    state.artifacts.append(
        create_artifact(
            TechnicalPRDArtifact,
            workflow_id=state.feature_id,
            artifact_id="002_technical_prd.json",
            producer="product_manager",
            metadata={},
            payload=_technical_prd_payload(),
        )
    )
    state.artifacts.append(_reconnaissance_artifact(state.feature_id, "backend"))
    reconnaissance = RecordingReconnaissance()
    orchestrator = FeatureWorkflowOrchestrator(reconnaissance=cast(Any, reconnaissance))

    result = await orchestrator.resume(
        state,
        answers=[ClarificationAnswer(question_id="q1", answer="Use the existing status route.")],
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
    )

    assert result.status is FeatureWorkflowStatus.COMPLETED
    assert reconnaissance.inspected == []


@pytest.mark.asyncio
async def test_a_completed_feature_states_that_nothing_reviewed_the_repository_seam() -> None:
    """The completion summary must not let an approved integration review imply more than it is.

    This is the one artifact a non-engineer reads, and it names an approved integration review.
    That approval is contract conformance decided from child result metadata, so a frontend
    calling the backend with the wrong field name reaches a completed feature unremarked.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-coverage-limits", request)
    orchestrator = FeatureWorkflowOrchestrator()

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.COMPLETED
    completion = next(
        artifact for artifact in result.artifacts if isinstance(artifact, FeatureCompletionArtifact)
    )
    assert (
        "The integration review checked contract conformance only: it did not review "
        "cross-repository behaviour, security or deployment."
    ) in completion.known_limitations


@pytest.mark.asyncio
async def test_a_schema_rejection_records_safe_types_without_payload_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Schema failures remain classifiable without leaking values through state or logs."""
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-schema-rejection", request)
    orchestrator = FeatureWorkflowOrchestrator(child_executor=SchemaRejectingExecutor())

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    issues = result.child_workflows["backend"].blocking_issues
    assert "ValidationError" in issues[0]
    assert any("[missing]" in item for item in issues[1:]), issues
    assert any("[extra_forbidden]" in item for item in issues[1:]), issues
    captured = capsys.readouterr()
    assert _SCHEMA_REJECTION_SECRET not in " ".join(issues)
    assert _SCHEMA_REJECTION_SECRET not in captured.out
    assert _SCHEMA_REJECTION_SECRET not in captured.err


@pytest.mark.asyncio
async def test_a_branch_pushed_by_an_earlier_attempt_is_reported_on_failure() -> None:
    """Work already on the remote must be named, not silently dropped from the record."""
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-stranded-branch", request)
    orchestrator = FeatureWorkflowOrchestrator(child_executor=CommitThenFailExecutor())

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    child = result.child_workflows["backend"]
    assert child.status is ChildWorkflowStatus.FAILED
    stranded = [item for item in child.blocking_issues if "pushed branch" in item]
    assert len(stranded) == 1, child.blocking_issues
    assert child.branch_name in stranded[0]
    # The status stays failed: a later attempt really was rejected.
    assert "still failed" in stranded[0]


@pytest.mark.asyncio
async def test_an_unexpected_child_fault_is_contained_to_that_repository() -> None:
    """A raising child must fail itself, not abort the whole feature.

    An empty commit on a no-op retry used to propagate out of the fan-out and take every
    unrelated repository down with it.
    """
    payload = feature_payload()
    payload["repositories"] = [
        _repository("backend", "backend"),
        _repository("frontend", "frontend"),
    ]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-child-fault", request)
    orchestrator = FeatureWorkflowOrchestrator(child_executor=RaisingChildExecutor("backend"))

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.FAILED
    # The frontend was not taken down with the raising sibling: it reached a passing review
    # and is offered for publication, which is what containment means here now that a feature
    # which did not land opens nothing on its own.
    published = await _publish_on_request(result)
    assert published.child_workflows["frontend"].status is ChildWorkflowStatus.COMPLETED
    # The error type is retained for triage; its message is never persisted.
    assert "RuntimeError" in result.child_workflows["backend"].blocking_issues[0]
    assert "git commit failed" not in result.child_workflows["backend"].blocking_issues[0]


@pytest.mark.asyncio
async def test_a_provider_fault_earns_the_attempt_again_instead_of_ending_the_workstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider error is not a defect in the repository, and must not spend the workstream.

    It escaped the child loop and was caught by the fan-out, so the repository was recorded
    failed having used none of its retries: -072's backend ended before it wrote a line, and
    -070's ended one review finding from approval.
    """
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)
    state = _initial_feature_state("feature-provider-fault", request)
    executor = ProviderFaultThenApproveExecutor(faults=3)

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    # Three consecutive failures is what -077's backend actually met, inside four minutes.
    assert executor.attempts == 4
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED


@pytest.mark.asyncio
async def test_a_retryable_operation_failure_earns_the_attempt_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AB-Feature-111's backend: `failed_retryable`, two attempts reserved, none ever spent.

    Its coding call failed and the operation journal recorded the failure as retryable with
    two of three attempts still available. The workstream ended anyway at retry_count 0,
    because the exception reached the loop as `ExternalOperationError` and only the adapter
    errors were named as faults. Whether a fault is worth another attempt is a property of
    the fault, not of which layer happened to wrap it.
    """
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)
    state = _initial_feature_state("feature-operation-fault", request)
    executor = ProviderFaultThenApproveExecutor(faults=2, error=ExternalOperationError)

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert executor.attempts == 3
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        OperationInProgressError,
        OperationLeaseLostError,
        OperationReconciliationRequired,
        OperationTransitionConflictError,
    ],
)
async def test_an_unconfirmed_external_effect_is_never_retried_by_the_child_loop(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    """The other half of the same decision, and the more important one.

    These four all mean the same thing: what happened outside this process could not be
    confirmed. Retrying is the platform guessing about an effect it deliberately refuses to
    guess about -- a second push, a second pull request -- which is the whole reason the
    operation journal exists. Widening the fault tuple must not widen this.
    """
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)
    state = _initial_feature_state(f"feature-unconfirmed-{error.__name__}", request)
    executor = ProviderFaultThenApproveExecutor(faults=1, error=error)

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    # Exactly one call: the loop did not try again behind a person's back.
    assert executor.attempts == 1
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.FAILED


class PrecommitRejectionExecutor:
    """Reject the backend at pre-commit twice with different diagnostics, then approve."""

    def __init__(self) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self.attempts = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        if kwargs["repository"].repository_id != "backend":
            return await self._delegate.run(**kwargs)
        self.attempts += 1
        if self.attempts <= 2:
            raise SourceValidationError(
                [f"Pre-commit source validation failed (attempt {self.attempts})"]
            )
        return await self._delegate.run(**kwargs)


class UncommittedLintRejectionExecutor:
    """Reject at the source gate while the workspace still holds an earlier attempt's files.

    The rejected attempt commits nothing, so its fingerprint is whatever the reset left
    behind -- identical to the attempt before it -- while the changed-file list is non-empty.
    """

    def __init__(self, *, rejections: int) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self._rejections = rejections
        self.attempts = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        if kwargs["repository"].repository_id != "backend":
            return await self._delegate.run(**kwargs)
        self.attempts += 1
        execution = await self._delegate.run(**kwargs)
        if self.attempts > self._rejections:
            return execution
        failed = execution.result.model_copy(
            update={
                "status": "failed",
                "blocking_issues": [
                    f"Pre-commit source validation failed (tool=eslint) at line {self.attempts}"
                ],
                "pull_request_readiness": False,
                "failure_classification": FailureClassification.VALIDATION_SOURCE_FAILURE.value,
                # What the commit gate records when it stops an attempt. The classification
                # beside it does not identify this path -- a reviewer reporting one failing
                # command earns the same one -- so the loop reads the marker instead, and a
                # fixture that omits it is not reproducing a pre-commit rejection at all.
                "metadata": {
                    **execution.result.metadata,
                    "deterministic_gate": True,
                    "source_validation_rejected": True,
                },
                "production_files_changed": ["src/routes/status.js"],
                "production_diff_fingerprint": "workspace-state-after-reset",
                "test_diff_fingerprint": "tests-after-reset",
            }
        )
        return ChildExecution(result=failed, code_completion=None)


@pytest.mark.asyncio
async def test_a_lint_rejection_holding_earlier_files_is_not_a_reproduction() -> None:
    """Nothing was committed, so the fingerprint is not evidence about what was written.

    The test asked whether any production file changed, but they accumulate in the workspace
    across attempts, so a rejected attempt still reports them. -079's backend was rejected by
    eslint while holding an earlier attempt's files, was counted as reproducing the previous
    change, and sat one rejection from being ended while it was narrowing its findings.
    """
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-uncommitted-lint", request)
    # The default budget of two would end the workstream before the repeat rule could be
    # what decides it, which is the thing under test here.
    state = state.model_copy(update={"max_validation_retries": 8})
    executor = UncommittedLintRejectionExecutor(rejections=3)

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert executor.attempts == 4
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED


@pytest.mark.asyncio
async def test_repeated_precommit_rejections_are_not_read_as_a_circle() -> None:
    """A rejected attempt resets the workspace, so its fingerprint is not what it wrote.

    -076's console produced two attempts whose fingerprints matched exactly while they failed
    at different gates, which cannot both be true of the same source. Counting that as a
    return to an earlier state would undo the allowance that lets a lint rejection be retried.
    """
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-precommit-repeat", request)
    executor = PrecommitRejectionExecutor()

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert executor.attempts == 3
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED


@pytest.mark.asyncio
async def test_a_failed_clone_earns_the_attempt_the_journal_reserved_for_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clone the journal marks retryable must not end the repository at attempt zero.

    -075's backend cloned, failed, and was recorded failed with `retry_count` 0 while its
    `clone_repository` operation sat at attempt 1 of 3, holding two attempts it never used.
    """
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)
    state = _initial_feature_state("feature-clone-fault", request)
    executor = ProviderFaultThenApproveExecutor(faults=1, error=GitAdapterError)

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert executor.attempts == 2
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED


def test_a_fault_with_no_model_spend_behind_it_waits_four_times_as_long() -> None:
    """The clone-stage tail is five minutes, not seventy-five seconds.

    54- Part 2 left this open: AB-Feature-190 died inside a forty-second GitHub outage, and
    the four provider-shaped retries the clone was given -- 5, 10, 20, 40 -- barely outlast
    one. A clone retry has nothing to protect. It spends no tokens, holds no attempt a model
    could have used, and the wait is already excluded from the runtime ceiling, so the same
    four retries wait 20, 40, 80, 160 instead.
    """
    provider = LLMAdapterError("provider fault")
    git = GitAdapterError("git clone failed")
    schedule = [1, 2, 3, 4]
    assert [_fault_backoff_seconds(provider, fault_count=n) for n in schedule] == [
        5.0,
        10.0,
        20.0,
        40.0,
    ]
    assert [_fault_backoff_seconds(git, fault_count=n) for n in schedule] == [
        20.0,
        40.0,
        80.0,
        160.0,
    ]
    # And through the wrapper a clone really arrives in, which is why the discriminator walks
    # the cause chain rather than reading the top-level type.
    wrapped = ExternalOperationError("clone failed before a confirmed result")
    wrapped.__cause__ = git
    assert _fault_backoff_seconds(wrapped, fault_count=1) == 20.0


@pytest.mark.asyncio
async def test_the_longer_clone_tail_is_served_in_cancellable_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled feature must not wait out a 160-second backoff to notice.

    Both fault loops check cancellation between attempts only, so lengthening the tail with a
    single `sleep` would have made a cancelled feature take almost three minutes to stop --
    worse than the tail this replaces, on the one axis that was not broken.
    """
    slept: list[float] = []

    async def record(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("asyncio.sleep", record)
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-clone-tail", request)
    executor = ProviderFaultThenApproveExecutor(faults=1, error=GitAdapterError)

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED
    assert sum(slept) == 20.0, "the first Git fault waits the whole twenty seconds"
    assert max(slept) <= feature_workflow_module._FAULT_BACKOFF_SLICE_SECONDS, (
        "no single sleep may outlast one slice, or a cancellation waits behind it"
    )


@pytest.mark.asyncio
async def test_a_provider_that_stays_down_reaches_a_human_instead_of_retrying_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The allowance is bounded, so an outage escalates rather than burning the deadline."""
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)
    state = _initial_feature_state("feature-provider-outage", request)
    executor = ProviderFaultThenApproveExecutor(faults=99)

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.FAILED
    assert "LLMAdapterError" in result.child_workflows["backend"].blocking_issues[0]


@pytest.mark.asyncio
async def test_precommit_source_failure_is_retried_with_its_lint_diagnostic() -> None:
    """Scoped lint output reaches the next coding attempt instead of failing at Husky."""
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-source-validation-retry", request)
    executor = SourceValidationRetryExecutor()
    orchestrator = FeatureWorkflowOrchestrator(child_executor=executor)

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.COMPLETED
    # The exact diagnostic reaches the next attempt, and reaches it marked as the thing to
    # fix rather than as one more line in an undifferentiated history.
    assert executor.feedback[0] == []
    [retry] = executor.feedback[1]
    assert retry.startswith("MUST FIX -- attempt 1 failed on this: ")
    assert retry.endswith("Pre-commit source validation `eslint src/tile.js` exited with code 1")
    assert result.child_workflows["backend"].validation_retry_count == 1


@pytest.mark.asyncio
async def test_manual_review_evidence_failure_does_not_retry_feature_coding() -> None:
    """An unreviewable bounded prompt stops after one attempt and records operator routing."""
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    state = _initial_feature_state(
        "feature-manual-source-review",
        StartFeatureRequest.model_validate(payload),
    )
    executor = ManualReviewRequiredExecutor()

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state,
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
    )

    assert executor.attempts == 1
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    child = result.child_workflows["backend"]
    assert child.retry_count == 0
    assert child.retry_refusal_reason == "manual_review_required"


@pytest.mark.asyncio
async def test_two_child_attempts_keep_distinct_artifacts_and_exact_references() -> None:
    """A later approval must not overwrite or point back to the rejected attempt."""
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-attempt-lineage", request)
    executor = ReviewRetryThenApproveExecutor()

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    completions = [
        item
        for item in result.artifacts
        if isinstance(item, CodeCompletionArtifact)
        and item.workflow_id == "feature-attempt-lineage:backend"
    ]
    reviews = [
        item
        for item in result.artifacts
        if isinstance(item, ReviewArtifact)
        and item.workflow_id == "feature-attempt-lineage:backend"
    ]
    child_results = [
        item
        for item in result.artifacts
        if isinstance(item, ChildWorkflowResultArtifact) and item.repository_id == "backend"
    ]

    assert executor.attempts == 2
    assert [item.artifact_id for item in completions] == [
        "006_code_completion.backend.attempt-0.json",
        "006_code_completion.backend.attempt-1.json",
    ]
    assert [item.artifact_id for item in reviews] == [
        "007_review.backend.attempt-0.json",
        "007_review.backend.attempt-1.json",
    ]
    assert [item.artifact_id for item in child_results] == [
        "011_child_workflow_result.backend.attempt-0.json",
        "011_child_workflow_result.backend.attempt-1.json",
    ]
    for attempt in range(2):
        assert completions[attempt].artifact_id in reviews[attempt].metadata["source_artifact_ids"]
        assert (
            child_results[attempt].code_completion_artifact_id == completions[attempt].artifact_id
        )
        assert child_results[attempt].review_artifact_id == reviews[attempt].artifact_id
        assert child_results[attempt].metadata["source_artifact_ids"] == [
            completions[attempt].artifact_id,
            reviews[attempt].artifact_id,
        ]
    integration_review = next(
        item for item in result.artifacts if isinstance(item, IntegrationReviewArtifact)
    )
    assert (
        integration_review.repository_results[0].child_result_artifact_id
        == child_results[1].artifact_id
    )
    feature_completion = next(
        item for item in result.artifacts if isinstance(item, FeatureCompletionArtifact)
    )
    assert feature_completion.child_workflow_results == [child_results[1].artifact_id]


@pytest.mark.asyncio
async def test_retry_decision_is_checkpointed_before_the_next_child_attempt() -> None:
    """A crash between attempts must reload the consumed budget and exact prior diagnostic."""
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-retry-checkpoint", request)
    executor = SourceValidationRetryExecutor()
    snapshots: list[Any] = []

    async def checkpoint(snapshot: Any, boundary: Any, repository_id: str | None) -> None:
        snapshots.append((snapshot, boundary, repository_id))

    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=executor, checkpoint_writer=checkpoint
    )

    await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    retry_snapshots = [
        snapshot
        for snapshot, boundary, repository_id in snapshots
        if repository_id == "backend"
        and boundary.value == "before_coding"
        and snapshot.child_workflows["backend"].retry_count == 1
    ]
    assert retry_snapshots
    persisted_child = retry_snapshots[0].child_workflows["backend"]
    assert persisted_child.validation_retry_count == 1
    assert "eslint src/tile.js" in persisted_child.blocking_issues[0]
    persisted_results = [
        item
        for item in retry_snapshots[0].artifacts
        if isinstance(item, ChildWorkflowResultArtifact) and item.repository_id == "backend"
    ]
    assert [item.artifact_id for item in persisted_results] == [
        "011_child_workflow_result.backend.attempt-0.json"
    ]


@pytest.mark.asyncio
async def test_the_last_allowed_clarification_answer_is_processed() -> None:
    """The bound limits extra rounds; it must not reject the valid answer at the boundary."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-clarification-bound", request)
    state.status = FeatureWorkflowStatus.WAITING_FOR_HUMAN
    state.max_clarification_rounds = 2
    state.clarification_rounds = 1
    technical_prd = create_artifact(
        TechnicalPRDArtifact,
        workflow_id=state.feature_id,
        artifact_id="002_technical_prd.json",
        producer="product_manager",
        metadata={},
        payload=_technical_prd_payload(),
    )
    state.artifacts.append(technical_prd)
    orchestrator = FeatureWorkflowOrchestrator()

    result = await orchestrator.resume(
        state,
        answers=[ClarificationAnswer(question_id="q1", answer="Use the existing status route.")],
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
    )

    assert result.status is FeatureWorkflowStatus.COMPLETED
    assert result.clarification_rounds == 2
    technical_revisions = [
        artifact for artifact in result.artifacts if isinstance(artifact, TechnicalPRDArtifact)
    ]
    assert len(technical_revisions) == 2
    assert technical_revisions[0].unresolved_questions
    resolved = technical_revisions[-1]
    assert resolved.metadata["supersedes"] == technical_revisions[0].artifact_id
    assert resolved.metadata["clarification_answers"] == {"q1": "Use the existing status route."}


def _technical_prd_payload() -> dict[str, object]:
    """Return a technical PRD carrying exactly one unresolved clarification question."""
    return {
        "title": "Contract pause",
        "solution_summary": "Expose server status through the shared contract.",
        "functional_requirements": [
            {
                "requirement_id": "requirement-contract-pause",
                "description": "Retain all completed parallel child results.",
                "priority": "must",
                "acceptance_criteria": ["Both child results are persisted before pausing."],
                "dependencies": [],
            }
        ],
        "non_functional_requirements": [],
        "data_requirements": [],
        "integration_requirements": [],
        "security_requirements": [],
        "assumptions": [],
        "unresolved_questions": [
            {
                "question_id": "q1",
                "question": "Which route should serve status?",
                "rationale": "The PRD does not name one.",
                "required": True,
            }
        ],
    }


def test_repository_targets_resolve_when_workstream_ids_are_different() -> None:
    """Integration findings are repository-owned even when planner workstream IDs differ."""
    backend = _workstream_plan("api-implementation", []).model_copy(
        update={"repository_id": "backend"}
    )
    plan = create_artifact(
        RepositoryExecutionPlanArtifact,
        workflow_id="feature-routing",
        artifact_id="010_repository_execution_plan.json",
        producer="feature_planner",
        metadata={},
        payload={
            "feature_id": "feature-routing",
            "contract_artifact_id": "009_integration_contract.json",
            "workstreams": [backend.model_dump(mode="json")],
            "execution_order": ["api-implementation"],
            "parallel_groups": [["api-implementation"]],
            "recommended_merge_order": ["api-implementation"],
            "integration_test_plan": ["Validate API compatibility."],
            "merge_strategy": "independent",
            "deployment_strategy": "independent",
            "feature_flag_strategy": [],
            "rollback_strategy": ["Revert the API pull request."],
        },
    )

    assert _resolve_target_workstream_ids(plan, {"backend"}) == {"api-implementation"}

    rejected_value = "github_pat_model_supplied_workstream"
    with pytest.raises(FeatureWorkflowError) as raised:
        _resolve_target_workstream_ids(plan, {rejected_value})
    assert rejected_value in str(raised.value)
    assert rejected_value not in " ".join(raised.value.diagnostics)
    # Written at the raise site now, rather than recovered by matching the message against a
    # table of substrings that no longer exists.
    assert raised.value.diagnostics == (
        "A targeted retry named a repository workstream this feature does not have.",
    )
    assert raised.value.failure_classification is FeatureFailureClassification.PLATFORM_DEFECT


def test_workspace_identity_does_not_collide_for_punctuation_variants() -> None:
    """Distinct durable feature IDs cannot normalize to the same workspace directory."""
    assert _feature_branch_segment("feature.a") != _feature_branch_segment("feature-a")


def test_feature_history_refuses_a_new_duplicate_artifact_identifier() -> None:
    """Append-only history cannot silently reuse an earlier attempt's identifier."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-duplicate-artifact", request)
    artifact = create_artifact(
        TechnicalPRDArtifact,
        workflow_id=state.feature_id,
        artifact_id="002_technical_prd.json",
        producer="product_manager",
        metadata={},
        payload=_technical_prd_payload(),
    )
    _append_artifacts(state, [artifact])

    with pytest.raises(FeatureWorkflowError, match="artifact identifier already exists"):
        _append_artifacts(state, [artifact])


def _repository(repository_id: str, name: str, *, role: str | None = None) -> dict[str, object]:
    """Build one repository specification for a dynamic-cardinality payload."""
    return {
        "repository_id": repository_id,
        "name": name,
        "role": role or name,
        "repository_url": f"https://github.com/example/{repository_id}.git",
        "default_branch": "main",
    }


def _workstream_plan(workstream_id: str, dependencies: list[str]) -> RepositoryWorkstreamPlan:
    """Build the minimum valid workstream needed to exercise cycle detection."""
    return RepositoryWorkstreamPlan.model_validate(
        {
            "workstream_id": workstream_id,
            "repository_id": workstream_id,
            "role": "service",
            "requirement_ids": [],
            "scoped_requirements": [],
            "out_of_scope_requirements": [],
            "shared_requirements": [],
            "responsibilities": ["Implement the assigned work."],
            "task_ids": [f"{workstream_id}-task"],
            "dependency_workstream_ids": dependencies,
            "contract_sections_consumed": [],
            "contract_sections_implemented": [],
            "acceptance_criteria": ["Review is approved."],
            "test_requirements": ["Run configured validation."],
            "documentation_requirements": [],
            "expected_files_or_areas": [],
            "required": True,
            "implementation_expectations": [],
        }
    )


def feature_payload() -> dict[str, object]:
    """Return two independent workstreams scheduled in the same parallel group."""
    return {
        "feature_id": "feature-contract-pause",
        "prd": {
            "title": "Contract pause",
            "problem_statement": "A child may require an explicit contract change.",
            "goals": ["Retain concurrent child outcomes."],
            "user_stories": [
                {
                    "story_id": "story-contract-pause",
                    "persona": "Operator",
                    "need": "Review child outcomes after a contract pause",
                    "benefit": "No successful sibling work is lost",
                    "acceptance_criteria": ["Both child results remain traceable."],
                }
            ],
            "requirements": [
                {
                    "requirement_id": "requirement-contract-pause",
                    "description": "Retain all completed parallel child results.",
                    "priority": "must",
                    "acceptance_criteria": ["Both child results are persisted before pausing."],
                    "dependencies": [],
                }
            ],
            "constraints": ["Use mock dependencies."],
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


def test_a_wrapped_failure_reports_what_actually_caused_it() -> None:
    """AB-Feature-105's backend, recorded as a type name for the thirty minutes it was dead.

    `ExternalOperationError` is raised around a journalled operation and explains nothing by
    itself. The `GitAdapterError` underneath held `git clone failed
    (GIT_CLONE_FAILED_EXIT_128)` -- the one sentence that identifies a wrong repository URL
    at a glance -- and it was thrown away.
    """
    cause = GitAdapterError(
        "clone failed", diagnostics=["git clone failed (GIT_CLONE_FAILED_EXIT_128)"]
    )
    wrapper = ExternalOperationError("operation could not be completed")
    wrapper.__cause__ = cause

    assert _causal_diagnostics(wrapper) == ("git clone failed (GIT_CLONE_FAILED_EXIT_128)",)
    # And an error that explains itself is still preferred over anything beneath it.
    assert _causal_diagnostics(cause) == ("git clone failed (GIT_CLONE_FAILED_EXIT_128)",)


def test_a_failure_nobody_can_explain_safely_still_reports_nothing() -> None:
    """The default is unchanged: an arbitrary message may carry credential-bearing detail."""
    cause = RuntimeError("connection refused for https://token@example.invalid/repo")
    wrapper = ExternalOperationError("operation could not be completed")
    wrapper.__cause__ = cause

    assert _causal_diagnostics(wrapper) == ()


def test_a_cycle_in_the_cause_chain_terminates() -> None:
    """A `raise ... from` loop must not hang the failure path that reports it."""
    first = ExternalOperationError("first")
    second = ExternalOperationError("second")
    first.__cause__ = second
    second.__cause__ = first

    assert _causal_diagnostics(first) == ()


def test_retry_feedback_leads_with_what_this_attempt_failed_on() -> None:
    """Each line says its own status, because nothing guarantees the order survives.

    These are spliced into every task's description. Merged into one deduplicated list --
    which is what this replaced -- -105's ninth attempt received "wire BulkDeleteApps into
    AppConfigModal" beside "your AppConfigModal wiring can never enable the button", with
    nothing saying which was still true.
    """
    lines = _retry_feedback(
        ["'BulkDeleteApps' is not defined"],
        unresolved_since={"'BulkDeleteApps' is not defined": 4},
        attempt=9,
        inherited=["Match the contract's response envelope."],
        superseded=["BulkDeleteApps was added but nothing refers to it."],
    )

    # The live defect first, and named as something the previous approach failed to fix.
    assert lines[0].startswith("MUST FIX")
    assert "4 attempts running" in lines[0]
    assert "'BulkDeleteApps' is not defined" in lines[0]
    # The stale one is still available, and marked as possibly already fixed.
    stale = next(item for item in lines if "nothing refers to it" in item)
    assert "may already be fixed" in stale
    assert lines.index(stale) > 0
    assert any("integration review" in item for item in lines)


def test_a_first_failure_is_not_described_as_a_repeat() -> None:
    """ "Reported 1 attempts running" would read as a stall on the very first rejection."""
    lines = _retry_feedback(
        ["npm run test exited 1"],
        unresolved_since={"npm run test exited 1": 1},
        attempt=1,
    )

    assert lines == ["MUST FIX -- attempt 1 failed on this: npm run test exited 1"]


def test_carried_findings_are_bounded() -> None:
    """An unbounded history is what diluted the live defect into a pile of contradictions."""
    lines = _retry_feedback(
        ["the current one"],
        unresolved_since={"the current one": 1},
        attempt=7,
        superseded=[f"old finding {index}" for index in range(20)],
    )

    assert sum("may already be fixed" in item for item in lines) == 4
    # The most recent of them, not the oldest: an ancient finding is the least likely to
    # still describe the workspace.
    assert any("old finding 19" in item for item in lines)
    assert not any("old finding 0:" in item for item in lines)


class OneFailsWhileTheOtherHolds:
    """One repository fails at once; the other blocks until it is released.

    The shape of AB-Feature-105: the backend was refused a clone in its first minute while
    the frontend went on retrying for half an hour.
    """

    def __init__(self) -> None:
        """Hold the slow repository until the test lets it finish."""
        self.release = asyncio.Event()
        self.fast_failed = asyncio.Event()
        self.holding = asyncio.Event()
        # The live snapshot the orchestrator is mutating, which is not the object the caller
        # handed to `start`.
        self.state: Any = None

    async def run(self, **kwargs: Any) -> Any:
        """Fail `backend` immediately and hold `frontend` open."""
        self.state = kwargs["feature"]
        repository = kwargs["repository"]
        if repository.repository_id == "backend":
            self.fast_failed.set()
            # A refused path, which is what -105's backend actually met, and which the fault
            # policy deliberately excludes: repeating a refusal only refuses again. This was
            # an `ExternalOperationError` until that became a retryable fault -- correctly,
            # since the journal reserves attempts for it -- at which point this executor
            # stopped failing fast and started describing a retry loop instead.
            msg = "clone path was refused"
            raise GitSafetyError(msg)
        self.holding.set()
        await self.release.wait()
        return await MockChildWorkstreamExecutor().run(**kwargs)


@pytest.mark.asyncio
async def test_a_repository_that_fails_first_is_recorded_before_its_sibling_finishes() -> None:
    """The half-hour lie: a dead repository displayed as running until the last one stops.

    `asyncio.gather` withheld every result until the slowest sibling returned, so
    AB-Feature-105's backend failed on a refused clone at 10:35 and said so at 11:06 -- one
    second before the feature itself failed. Somebody watched a dead workstream described as
    live for thirty minutes, which is worse than showing nothing.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-settles-early", request)
    executor = OneFailsWhileTheOtherHolds()
    orchestrator = FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))

    run = asyncio.create_task(
        orchestrator.start(
            state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
        )
    )
    try:
        await asyncio.wait_for(executor.fast_failed.wait(), timeout=10)
        await asyncio.wait_for(executor.holding.wait(), timeout=10)
        live = executor.state
        # Give the scheduler its turn to record the outcome it already has.
        for _ in range(200):
            if live.child_workflows["backend"].status is not ChildWorkflowStatus.RUNNING:
                break
            await asyncio.sleep(0.01)

        # Recorded while the sibling is provably still going.
        assert not executor.release.is_set()
        assert live.child_workflows["backend"].status is ChildWorkflowStatus.FAILED
        assert live.child_workflows["frontend"].status is ChildWorkflowStatus.RUNNING
        # And it says what happened, rather than naming the exception type alone.
        assert any(
            "GitSafetyError" in issue for issue in live.child_workflows["backend"].blocking_issues
        )
    finally:
        executor.release.set()
        await asyncio.wait_for(run, timeout=30)


class RefusesBackendRequiredContext:
    """Fail `backend` with the Engineer's pre-model-call refusal; run `frontend` normally."""

    async def run(self, **kwargs: Any) -> Any:
        if kwargs["repository"].repository_id == "backend":
            raise RequiredContextRefusal(
                [
                    "This attempt was refused before the model was called: the repository "
                    "snapshot dropped a file the attempt is ordered to work on, and an "
                    "attempt that cannot see its own target answers findings with guesses. "
                    "No retry was spent on this refusal.",
                    "Required file src/routes/orders.js is named by the plan or by a "
                    "blocking diagnostic, but was dropped from the snapshot: the required "
                    "set exceeded the snapshot's total character budget.",
                ]
            )
        return await MockChildWorkstreamExecutor().run(**kwargs)


@pytest.mark.asyncio
async def test_a_required_context_refusal_ends_the_workstream_with_its_budget_unspent() -> None:
    """Now-2's accounting half: the refusal is a decision, not a fault and not an attempt.

    D2's budget-sink shape: an attempt whose assigned file was dropped is deterministic for
    the same required set, so a refusal that consumed a retry would burn the whole budget
    refusing the same thing. The exception must escape the child loop the way GitSafetyError
    does -- recorded as a failed workstream carrying its own sentences and classification,
    with every retry counter still at zero and no "did not anticipate" triage.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-refused-context", request)
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, RefusesBackendRequiredContext())
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    backend = result.child_workflows["backend"]
    assert backend.status is ChildWorkflowStatus.FAILED
    # The refusal spent nothing: not the attempt counter, not any classified retry budget.
    assert backend.retry_count == 0
    assert backend.validation_retry_count == 0
    assert backend.implementation_retry_count == 0
    text = " ".join(backend.blocking_issues)
    assert "refused before the model was called" in text
    assert "src/routes/orders.js" in text
    assert "did not anticipate" not in text
    # T10 (80-): a first-attempt refusal stays terminal -- no partition without prior
    # blocking diagnostics, because attempt 0's ordered set cannot meaningfully overflow.
    assert backend.pending_context_partition is None
    assert backend.targeted_attempt_kind is None
    assert backend.context_partition_count == 0


# Two blocking findings naming two disjoint files -- 212's dropped frontend paths, verbatim
# from the preserved artifacts, used here as fixture data only (the platform stays
# repo-agnostic; these are strings a reviewer once wrote).
_FINDING_ALL_APPS = (
    "The apps table drops ownerless rows: src/pages/AllApps.js:42 renders undefined owners."
)
_FINDING_ADD_APP_MODAL = (
    "The add-app modal never validates input: src/components/apps/AddAppModal.js:17 "
    "submits the raw form body."
)
# The refusal sentences 212's attempt-1 results carry, verbatim.
_212_REFUSAL_SENTENCES = [
    "This attempt was refused before the model was called: the repository snapshot dropped "
    "a file the attempt is ordered to work on, and an attempt that cannot see its own "
    "target answers findings with guesses. No retry was spent on this refusal.",
    "Required file src/pages/AllApps.js is named by the plan or by a blocking diagnostic, "
    "but was dropped from the snapshot: the required set exceeded the snapshot's total "
    "character budget.",
    "Required file src/components/apps/AddAppModal.js is named by the plan or by a "
    "blocking diagnostic, but was dropped from the snapshot: the required set exceeded "
    "the snapshot's total character budget.",
]


class PartitionWaveExecutor:
    """Script 212's backend arc: reject attempt 0, refuse the granted retry, serve the wave.

    `frontend` approves immediately. For `backend`: attempt 0 is rejected with
    ``attempt0_findings``; the granted retry's unpartitioned run raises the 61- refusal;
    a scoped (partitioned) run returns an intermediate marker execution when non-final and
    follows ``final_verdicts`` when final; any later unpartitioned attempt also follows
    ``final_verdicts``. ``refuse_scoped`` makes every scoped run refuse too -- the floor.
    """

    def __init__(
        self,
        *,
        attempt0_findings: Sequence[str] = (_FINDING_ALL_APPS, _FINDING_ADD_APP_MODAL),
        final_verdicts: Sequence[str] = ("approved",),
        refuse_scoped: bool = False,
    ) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self._attempt0_findings = list(attempt0_findings)
        self._final_verdicts = list(final_verdicts)
        self._final_calls = 0
        self._refuse_scoped = refuse_scoped
        self.backend_calls: list[dict[str, Any]] = []

    def _rejected(
        self, execution: ChildExecution, findings: list[str], revision: str
    ) -> ChildExecution:
        review = execution.review
        assert review is not None
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    "pull_request_readiness": False,
                    "failure_classification": FailureClassification.REVIEW_SCOPE_FAILURE.value,
                    "blocking_issues": findings,
                    "current_revision": revision,
                    "production_files_changed": ["src/pages/AllApps.js"],
                    "production_diff_fingerprint": f"fingerprint-{revision}",
                }
            ),
            code_completion=execution.code_completion,
            review=review.model_copy(update={"verdict": "changes_requested"}),
        )

    async def run(self, **kwargs: Any) -> ChildExecution:
        child = kwargs["child"]
        repository_id = kwargs["repository"].repository_id
        execution = await self._delegate.run(**kwargs)
        if repository_id != "backend":
            return execution
        strategy = child.retry_strategy if isinstance(child.retry_strategy, dict) else {}
        partition = strategy.get("context_partition")
        partition = partition if isinstance(partition, dict) else None
        self.backend_calls.append(
            {
                "attempt": child.retry_count,
                "partition": partition,
                "feedback": list(kwargs["feedback"]),
                "kind": child.targeted_attempt_kind,
            }
        )
        if child.retry_count == 0 and partition is None:
            return self._rejected(execution, self._attempt0_findings, "backend-rev-0")
        if partition is None and child.retry_count == 1:
            raise RequiredContextRefusal(
                _212_REFUSAL_SENTENCES,
                dropped_paths=(
                    "src/pages/AllApps.js",
                    "src/components/apps/AddAppModal.js",
                ),
                budget_max_characters=160_000,
                budget_source="default",
            )
        if partition is not None and self._refuse_scoped:
            raise RequiredContextRefusal(
                _212_REFUSAL_SENTENCES[:2], dropped_paths=("src/pages/AllApps.js",)
            )
        if partition is not None and partition.get("final") is not True:
            return ChildExecution(
                result=execution.result.model_copy(
                    update={
                        "status": "failed",
                        "pull_request_readiness": False,
                        "metadata": {
                            **execution.result.metadata,
                            "context_partition_intermediate": True,
                            "context_partition_index": partition.get("index"),
                        },
                    }
                ),
                code_completion=execution.code_completion,
            )
        verdict = self._final_verdicts[min(self._final_calls, len(self._final_verdicts) - 1)]
        self._final_calls += 1
        if verdict == "approved":
            return execution
        return self._rejected(
            execution, [_FINDING_ADD_APP_MODAL], f"backend-rev-final-{self._final_calls}"
        )


def _backend_artifacts(state: FeatureWorkflowSnapshot, kind: type) -> list[Any]:
    """Return the backend child's parent-persisted artifacts of one type.

    Reviews and completions carry the child workflow id; child results are parent-scoped
    and carry the feature's, so those are matched on their own `repository_id`.
    """
    child_workflow_id = state.child_workflows["backend"].child_workflow_id
    return [
        artifact
        for artifact in state.artifacts
        if isinstance(artifact, kind)
        and (
            artifact.workflow_id == child_workflow_id
            or getattr(artifact, "repository_id", None) == "backend"
        )
    ]


@pytest.mark.asyncio
async def test_a_refused_retry_partitions_into_scoped_attempts_and_completes() -> None:
    """T6 (80-): the wave replaces the r1 death -- scoped attempts, one review, one retry.

    A retry whose ordered set exceeds the budget, with blocking diagnostics naming two
    disjoint files: two scoped attempts run sequentially on the same branch, each engineer
    call sees only its cluster's diagnostics and files, the wave's attempts are stamped
    `context_partition`, `retry_count` is incremented exactly once (by the grant the refused
    attempt already carried), and the review runs once, after the last cluster.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-context-partition", request)
    executor = PartitionWaveExecutor()
    orchestrator = FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))

    result = await orchestrator.start(state, credentials=_credentials())

    backend = result.child_workflows["backend"]
    assert backend.status in {ChildWorkflowStatus.APPROVED, ChildWorkflowStatus.COMPLETED}
    # One retry, spent once: the grant's increment, and nothing else.
    assert backend.retry_count == 1
    assert backend.context_partition_count == 1
    assert backend.targeted_attempt_kind is TargetedAttempt.CONTEXT_PARTITION
    assert backend.pending_context_partition is None

    attempt0, refused, first_cluster, final_cluster = executor.backend_calls
    assert attempt0["attempt"] == 0 and attempt0["partition"] is None
    assert refused["attempt"] == 1 and refused["partition"] is None
    # Each scoped attempt sees only its own cluster's diagnostics and files.
    assert first_cluster["partition"]["final"] is False
    assert first_cluster["partition"]["blocking_diagnostics"] == [_FINDING_ALL_APPS]
    assert "src/pages/AllApps.js" in first_cluster["partition"]["expected_files_or_areas"]
    assert first_cluster["feedback"] == [_FINDING_ALL_APPS]
    assert first_cluster["kind"] is TargetedAttempt.CONTEXT_PARTITION
    assert final_cluster["partition"]["final"] is True
    assert final_cluster["partition"]["blocking_diagnostics"] == [_FINDING_ADD_APP_MODAL]
    assert (
        "src/components/apps/AddAppModal.js"
        in final_cluster["partition"]["expected_files_or_areas"]
    )
    assert any(_FINDING_ADD_APP_MODAL in line for line in final_cluster["feedback"])
    assert not any(_FINDING_ALL_APPS in line for line in final_cluster["feedback"])
    # Both scoped attempts ran at the granted number: the wave numbered no new attempt.
    assert first_cluster["attempt"] == final_cluster["attempt"] == 1

    # One review cycle per wave: attempt 0's review and the wave's single final review.
    assert len(_backend_artifacts(result, ReviewArtifact)) == 2
    assert len(_backend_artifacts(result, ChildWorkflowResultArtifact)) == 2
    # The intermediate cluster persisted only its completion, partition-qualified.
    partition_completions = [
        artifact
        for artifact in _backend_artifacts(result, CodeCompletionArtifact)
        if isinstance(artifact.metadata.get("context_partition_index"), int)
    ]
    assert len(partition_completions) == 1
    assert "partition-0" in partition_completions[0].artifact_id
    assert partition_completions[0].metadata["child_attempt"] == 1
    # The sibling was never touched by the wave.
    assert result.child_workflows["frontend"].status in {
        ChildWorkflowStatus.APPROVED,
        ChildWorkflowStatus.COMPLETED,
    }


@pytest.mark.asyncio
async def test_a_refusal_whose_demands_share_one_file_stays_terminal() -> None:
    """T7 (80-): one cluster is nothing to split -- the honest floor stays terminal.

    All blocking diagnostics anchor to one file, so the partition would reproduce the
    refused attempt exactly. The workstream ends with the existing sentences and
    classification, and no retry counter moves beyond the grant the refusal already
    carried -- byte-compatible with the refusal-spends-nothing pin above.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-partition-floor", request)
    executor = PartitionWaveExecutor(
        attempt0_findings=[
            _FINDING_ALL_APPS,
            "The table header sorts wrong: src/pages/AllApps.js:7 compares display strings.",
        ]
    )
    orchestrator = FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))

    result = await orchestrator.start(state, credentials=_credentials())

    backend = result.child_workflows["backend"]
    assert backend.status is ChildWorkflowStatus.FAILED
    assert backend.retry_count == 1
    assert backend.pending_context_partition is None
    assert backend.context_partition_count == 0
    assert backend.targeted_attempt_kind is None
    text = " ".join(backend.blocking_issues)
    assert "refused before the model was called" in text
    assert "did not anticipate" not in text
    # No scoped attempt ever ran: every backend call was unpartitioned.
    assert all(call["partition"] is None for call in executor.backend_calls)


@pytest.mark.asyncio
async def test_the_wave_never_reconsults_the_retry_authority_between_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T8a (80-): the partition is execution shape, not a new grant.

    The refused attempt's grant was already decided; the wave spends it and asks nobody.
    The only retry-authority call in the whole arc is the one that granted the retry after
    attempt 0's rejection -- the one-retry-decision law, applied verbatim.
    """
    calls = 0
    real_decide = decide_child_retry

    def counting_decide(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real_decide(*args, **kwargs)

    monkeypatch.setattr(feature_workflow_module, "decide_child_retry", counting_decide)
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-partition-one-decision", request)
    orchestrator = FeatureWorkflowOrchestrator(child_executor=cast(Any, PartitionWaveExecutor()))

    result = await orchestrator.start(state, credentials=_credentials())

    assert result.child_workflows["backend"].status in {
        ChildWorkflowStatus.APPROVED,
        ChildWorkflowStatus.COMPLETED,
    }
    assert calls == 1


@pytest.mark.asyncio
async def test_a_mid_wave_crash_resumes_the_persisted_plan_without_redeciding() -> None:
    """T8b (80-): a partition survives the sweep like any other step.

    The wave persists its plan before the first scoped attempt and marks each cluster done
    as it lands, so the crash window between clusters is safe: a fresh orchestrator resumes
    the persisted plan through the ordinary claim path, runs exactly the clusters still
    pending, and never re-decides -- no re-clustering, no unpartitioned re-run, no second
    retry-authority call.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-partition-crash", request)
    recorder = CheckpointRecorder()
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, PartitionWaveExecutor()), checkpoint_writer=recorder
    )
    await orchestrator.start(state, credentials=_credentials())

    def mid_wave(snapshot: FeatureWorkflowSnapshot, boundary: Any, repository_id: Any) -> bool:
        if repository_id != "backend":
            return False
        plan = snapshot.child_workflows["backend"].pending_context_partition or []
        return (
            len(plan) == 2
            and plan[0].get("status") == "done"
            and plan[1].get("status") == "pending"
        )

    interrupted = recorder.last(mid_wave)
    assert interrupted.child_workflows["backend"].pending_context_partition is not None

    resumed_executor = PartitionWaveExecutor()
    recovered = FeatureWorkflowOrchestrator(child_executor=cast(Any, resumed_executor))
    result = await recovered.resume(interrupted, answers=[], credentials=_credentials())

    backend = result.child_workflows["backend"]
    assert backend.status in {ChildWorkflowStatus.APPROVED, ChildWorkflowStatus.COMPLETED}
    assert backend.retry_count == 1
    assert backend.pending_context_partition is None
    # The resumed run served only what was still pending: the final cluster, already
    # partitioned. It never re-ran attempt 0, never met the unpartitioned refusal, and
    # never rebuilt the plan.
    assert [call["partition"] is not None for call in resumed_executor.backend_calls] == [True]
    assert resumed_executor.backend_calls[0]["partition"]["final"] is True
    assert resumed_executor.backend_calls[0]["attempt"] == 1


@pytest.mark.asyncio
async def test_partitioned_findings_do_not_fake_recurrence() -> None:
    """T9 (80-): one review cycle per wave keeps the ledger's cycle count honest.

    A wave whose final review repeats a finding from before the partition contributes
    exactly one result and one review to the lineage -- the same number of eligible cycles
    an unpartitioned retry would -- so the theme counter and the resolved-issue ledger see
    no recurrences the partition itself manufactured.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-partition-ledger", request)
    executor = PartitionWaveExecutor(final_verdicts=("changes_requested", "approved"))
    orchestrator = FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))

    result = await orchestrator.start(state, credentials=_credentials())

    backend = result.child_workflows["backend"]
    assert backend.status in {ChildWorkflowStatus.APPROVED, ChildWorkflowStatus.COMPLETED}
    # attempt 0, the wave (attempt 1), and the ordinary post-wave retry (attempt 2):
    # exactly one result and one review per attempt, nothing per cluster.
    results = _backend_artifacts(result, ChildWorkflowResultArtifact)
    reviews = _backend_artifacts(result, ReviewArtifact)
    assert len(results) == 3
    assert len(reviews) == 3
    assert sorted(item.metadata.get("child_retry_count") for item in results) == [0, 1, 2]
    partition_completions = [
        artifact
        for artifact in _backend_artifacts(result, CodeCompletionArtifact)
        if isinstance(artifact.metadata.get("context_partition_index"), int)
    ]
    assert len(partition_completions) == 1


@pytest.mark.asyncio
async def test_a_scoped_attempt_that_itself_refuses_ends_the_workstream() -> None:
    """T6's floor clause (80- Part 3B.6): a refusing cluster is the honest end.

    A scoped attempt whose own file set still exceeds the window ends the workstream
    exactly as an unpartitioned refusal does -- same sentence, same classification, and the
    refusal itself spends nothing further.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-partition-scoped-floor", request)
    executor = PartitionWaveExecutor(refuse_scoped=True)
    orchestrator = FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))

    result = await orchestrator.start(state, credentials=_credentials())

    backend = result.child_workflows["backend"]
    assert backend.status is ChildWorkflowStatus.FAILED
    assert backend.retry_count == 1
    assert backend.pending_context_partition is None
    text = " ".join(backend.blocking_issues)
    assert "refused before the model was called" in text


def test_partition_clusters_merge_smallest_first_beyond_the_cap() -> None:
    """The wave is bounded: past four clusters the smallest merge, and no-file demands ride
    with the first cluster rather than forming a phantom one."""
    findings = [f"Demand {index}: src/module_{index}.js:1 is wrong." for index in range(6)] + [
        "A demand naming no file at all."
    ]
    child = ChildWorkflowReference(
        child_workflow_id="feature-partition-cap:backend",
        repository_id="backend",
        workstream_id="backend",
        branch_name="feature/partition-cap",
        workspace_path="/tmp/partition-cap",
        status=ChildWorkflowStatus.RUNNING,
        retry_count=1,
        blocking_issues=findings,
    )

    clusters = _context_partition_clusters(child)

    assert len(clusters) == 4
    assert sum(len(cluster["diagnostics"]) for cluster in clusters) == len(findings)
    # The no-file demand rides with the cluster of the first anchored diagnostic.
    assert "A demand naming no file at all." in " ".join(clusters[0]["diagnostics"])
    assert all(cluster["status"] == "pending" for cluster in clusters)


class LoopProbeExecutor:
    """Run exactly one Engineer and one Review per attempt, on a per-repository script.

    Stands in for `LiveChildWorkstreamExecutor`, which runs one Engineer Agent and one
    Reviewer Agent per call and returns. Counting its calls per repository is therefore
    counting Engineer and Review transitions, which is what the loop-safety assertions below
    need: asserting only the final state cannot tell a workstream that converged from one
    that cycled twenty times and then gave up.

    `script[repository_id]` holds the review verdict for each successive attempt and its last
    entry repeats for ever -- a reviewer that keeps asking for changes is the whole failure
    mode. `moves` decides whether each attempt produces a genuinely new revision and new
    production source, or resubmits byte-for-byte what the previous one did.
    """

    def __init__(
        self,
        script: dict[str, list[str]] | None = None,
        *,
        moves: bool = True,
        cap: int = 40,
    ) -> None:
        """Record the per-repository verdict script and whether attempts change anything."""
        self._delegate = MockChildWorkstreamExecutor()
        self._script = script or {}
        self._moves = moves
        self._cap = cap
        self.engineer_calls: dict[str, int] = {}
        self.reviewer_calls: dict[str, int] = {}
        self.revisions: dict[str, list[str]] = {}
        self.feedback: dict[str, list[list[str]]] = {}

    def attempts(self, repository_id: str) -> int:
        """Return how many Engineer/Review passes one repository was given."""
        return self.engineer_calls.get(repository_id, 0)

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Produce one scripted attempt, refusing to keep going past the safety cap."""
        repository_id = kwargs["repository"].repository_id
        attempt = self.engineer_calls.get(repository_id, 0)
        self.engineer_calls[repository_id] = attempt + 1
        self.feedback.setdefault(repository_id, []).append(list(kwargs["feedback"]))
        total = sum(self.engineer_calls.values())
        if total > self._cap:
            # The assertion the whole exercise exists for: a finite configured workflow may
            # never produce an unbounded number of Engineer/Review transitions.
            msg = f"unbounded engineer/review loop: {total} attempts across all repositories"
            raise AssertionError(msg)
        execution = await self._delegate.run(**kwargs)
        self.reviewer_calls[repository_id] = self.reviewer_calls.get(repository_id, 0) + 1
        # Keyed on the attempt the durable child state is on, not on this object's own call
        # count, so a recovered run continues the revision sequence rather than replaying
        # revisions the feature's history already holds.
        marker: Any = kwargs["child"].retry_count if self._moves else "frozen"
        revision = f"{repository_id}-rev-{marker}"
        self.revisions.setdefault(repository_id, []).append(revision)
        touched = [f"src/{repository_id}/module_{marker}.py"]
        verdicts = self._script.get(repository_id) or ["approved"]
        verdict = verdicts[min(attempt, len(verdicts) - 1)]
        completion = execution.code_completion
        assert completion is not None
        completion = completion.model_copy(
            update={
                "commit_sha": revision,
                "file_changes": [
                    FileChange(path=path, change_type="modified", description=f"attempt {attempt}")
                    for path in touched
                ],
                "production_files_changed": touched,
            }
        )
        review = execution.review
        assert review is not None
        # The live executor writes the fingerprint of what this attempt actually produced.
        # `_child_result` carries the child's previous one forward, so an executor that does
        # not override it reports the state it started from.
        shared: dict[str, Any] = {
            "current_revision": revision,
            "production_files_changed": touched,
            "production_diff_fingerprint": f"fingerprint-{repository_id}-{marker}",
        }
        if verdict == "approved":
            return ChildExecution(
                result=execution.result.model_copy(update=shared),
                code_completion=completion,
                review=review,
            )
        review = review.model_copy(
            update={"verdict": "changes_requested", "summary": f"Changes requested at {revision}."}
        )
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    **shared,
                    "status": "failed",
                    "pull_request_readiness": False,
                    "failure_classification": FailureClassification.REVIEW_SCOPE_FAILURE.value,
                    "blocking_issues": [
                        f"ValueError: the scoped requirement is unmet at src/{repository_id}/"
                        f"module_{marker}.py:17"
                    ],
                }
            ),
            code_completion=completion,
            review=review,
        )


class CountingIntegrationReviewer(RejectingIntegrationReviewer):
    """A never-approving integration reviewer that records every recomputation."""

    def __init__(self, *responsible_repository_ids: str) -> None:
        """Name every repository this reviewer holds responsible for its blocking findings."""
        super().__init__(responsible_repository_id=None)
        self._responsible = list(responsible_repository_ids)
        self.calls = 0

    async def review(self, **kwargs: Any) -> IntegrationReviewArtifact:
        """Return the same blocking verdict for every repository named, counting each call."""
        self.calls += 1
        artifact = await super().review(**kwargs)
        return artifact.model_copy(
            update={
                "cross_repository_findings": [
                    IntegrationReviewFinding(
                        finding_id=f"finding-{repository_id}",
                        severity="high",
                        responsible_repository_id=repository_id,
                        affected_repository_ids=[repository_id],
                        contract_reference="contract#apps",
                        description="The contract is not satisfied across repositories.",
                        evidence="Simulated blocking integration finding.",
                        recommended_fix=f"Align {repository_id} with the contract.",
                    )
                    for repository_id in self._responsible
                ]
            }
        )


class CheckpointRecorder:
    """Keep every durable snapshot the orchestrator wrote, so a crash can be resumed from one."""

    def __init__(self) -> None:
        """Start with no recorded boundaries."""
        self.writes: list[tuple[FeatureWorkflowSnapshot, Any, str | None]] = []

    async def __call__(
        self, state: FeatureWorkflowSnapshot, boundary: Any, repository_id: str | None
    ) -> None:
        """Record one already-deep-copied snapshot exactly as persistence would receive it."""
        self.writes.append((state, boundary, repository_id))

    def last(self, predicate: Any) -> FeatureWorkflowSnapshot:
        """Return the most recent recorded snapshot matching a crash window."""
        for state, boundary, repository_id in reversed(self.writes):
            if predicate(state, boundary, repository_id):
                return state
        msg = "no recorded checkpoint matched the requested crash window"
        raise AssertionError(msg)


def _credentials() -> RequestScopedCredentials:
    """Return the credential-free scope every mock-mode orchestration runs under."""
    return RequestScopedCredentials(openai_api_key=None, github_token=None)


def _loop_state(feature_id: str) -> Any:
    """Build one two-repository feature state for a convergence scenario."""
    return _initial_feature_state(feature_id, StartFeatureRequest.model_validate(feature_payload()))


def _integration_reviews(state: FeatureWorkflowSnapshot) -> list[IntegrationReviewArtifact]:
    """Return every durable integration review verdict in the order it was recorded."""
    return [
        artifact for artifact in state.artifacts if isinstance(artifact, IntegrationReviewArtifact)
    ]


@pytest.mark.asyncio
async def test_a_review_that_requests_changes_converges_in_exactly_two_attempts() -> None:
    """Test 1 -- normal convergence, asserted on call counts rather than the ending alone.

    One rejected attempt, one remediation that moves the revision, one approval. A fix that
    stopped the loop by refusing to retry at all would pass a final-state assertion and fail
    this one.
    """
    state = _loop_state("feature-loop-converges")
    executor = LoopProbeExecutor({"backend": ["changes_requested", "approved"]})
    orchestrator = FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))

    result = await orchestrator.start(state, credentials=_credentials())

    assert result.status is FeatureWorkflowStatus.COMPLETED
    assert executor.engineer_calls == {"backend": 2, "frontend": 1}
    assert executor.reviewer_calls == {"backend": 2, "frontend": 1}
    # The remediation was told what the review actually found, not a generic retry prompt.
    assert any("module_0.py:17" in line for line in executor.feedback["backend"][1])
    # And it reviewed the revision the remediation produced, not the one it replaced.
    backend = result.child_workflows["backend"]
    assert backend.current_revision == "backend-rev-1"
    assert backend.retry_count == 1


@pytest.mark.asyncio
async def test_a_remediation_that_changes_nothing_stops_instead_of_being_reviewed_again() -> None:
    """Test 2 -- no material progress ends the cycle and says so durably."""
    state = _loop_state("feature-loop-no-progress")
    executor = LoopProbeExecutor({"backend": ["changes_requested"]}, moves=False)
    orchestrator = FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))

    result = await orchestrator.start(state, credentials=_credentials())

    backend = result.child_workflows["backend"]
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert backend.status is ChildWorkflowStatus.FAILED
    # Review is not recomputed over and over on a repository that keeps resubmitting itself:
    # it stops well inside the configured ceiling rather than spending every cycle.
    assert executor.attempts("backend") == 2
    assert executor.reviewer_calls["backend"] == 2
    assert executor.attempts("backend") < result.max_child_review_cycles
    # The stop reason is persisted, and it is the reason this workstream actually stopped.
    assert backend.retry_refusal_reason is not None
    assert "not converging" in backend.retry_refusal_reason
    assert backend.meaningful_change_reason in {
        "attempt_reproduced_the_previous_production_change",
        "attempt_returned_to_an_earlier_submitted_state",
    }


@pytest.mark.asyncio
async def test_integration_review_does_not_send_the_same_input_back_to_the_engineer() -> None:
    """Test 3 -- duplicate Engineer input. This is AB-Feature-111's loop.

    Before the fix `changes_requested` was an unconditional edge back to the Engineer Agent,
    so a repository that had already satisfied the finding -- or could not act on it -- was
    recoded once per remaining integration cycle. With the cycle limit raised to twelve, the
    console ran twelve Engineer/Review passes against one frozen revision and one unchanging
    recommended fix, right past the five review cycles it was configured for.
    """
    state = _loop_state("feature-loop-duplicate-engineer")
    state.max_integration_review_cycles = 12
    executor = LoopProbeExecutor(moves=False)
    reviewer = CountingIntegrationReviewer("backend")
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor),
        integration_reviewer=cast(Any, reviewer),
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
    )

    result = await orchestrator.start(state, credentials=_credentials())

    backend = result.child_workflows["backend"]
    # One initial implementation and exactly one remediation: the second remediation would
    # have carried the identical revision, contract and finding.
    assert executor.attempts("backend") == 2
    assert executor.attempts("frontend") == 1
    # The configured review-cycle ceiling is respected by this path too, which is the ceiling
    # `require_retryable_workstream` already refuses an operator grant against.
    assert backend.retry_count + 1 <= result.max_child_review_cycles
    assert result.integration_review_cycles < result.max_integration_review_cycles
    assert backend.retry_refusal_reason is not None
    assert "identical inputs" in backend.retry_refusal_reason
    # Work that passed its own review is still published, and the feature still needs a human.
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert {
        artifact.repository
        for artifact in result.artifacts
        if isinstance(artifact, PullRequestArtifact)
    } == {"example/backend", "example/frontend"}
    assert not any(isinstance(item, FeatureCompletionArtifact) for item in result.artifacts)


@pytest.mark.asyncio
async def test_an_unchanged_integration_review_is_reused_rather_than_recomputed() -> None:
    """Test 4 -- duplicate Review input. Resuming must not ask the same question again."""
    state = _loop_state("feature-loop-duplicate-review")
    reviewer = CountingIntegrationReviewer()
    orchestrator = FeatureWorkflowOrchestrator(
        integration_reviewer=cast(Any, reviewer),
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
    )
    credentials = _credentials()

    first = await orchestrator.start(state, credentials=credentials)
    assert first.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert reviewer.calls == 1
    charged = first.integration_review_cycles

    resumed = await orchestrator.resume(first, answers=[], credentials=credentials)

    # Same repositories, same revisions, same contract, same merge order: the durable verdict
    # is routed again rather than a second provider call being spent to reproduce it.
    assert reviewer.calls == 1
    assert len(_integration_reviews(resumed)) == 1
    # And routing a reused verdict does not spend the cycle its own review already spent.
    assert resumed.integration_review_cycles == charged
    assert resumed.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN


@pytest.mark.asyncio
async def test_the_configured_review_cycle_maximum_stops_calling_the_engineer() -> None:
    """Test 5 -- every attempt is genuinely new work, and the ceiling still ends it."""
    state = _loop_state("feature-loop-max-cycles")
    # Generous enough that the classification budget is not what stops this; the point is
    # that the review-cycle ceiling does.
    state.max_implementation_retries = 9
    executor = LoopProbeExecutor({"backend": ["changes_requested"]})
    orchestrator = FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))

    result = await orchestrator.start(state, credentials=_credentials())

    backend = result.child_workflows["backend"]
    assert executor.attempts("backend") == result.max_child_review_cycles == 5
    assert executor.reviewer_calls["backend"] == 5
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert backend.status is ChildWorkflowStatus.FAILED
    # Attempts used, the maximum, the latest revision and the stop reason are all persisted.
    assert backend.retry_count == 4
    assert backend.implementation_retry_count == 4
    assert backend.current_revision == "backend-rev-4"
    assert backend.retry_refusal_reason is not None
    assert "child review cycle limit of 5" in backend.retry_refusal_reason
    assert backend.blocking_issues


@pytest.mark.asyncio
async def test_recovery_after_an_approved_attempt_does_not_run_the_engineer_again() -> None:
    """Test 6 -- crash after Engineer. The persisted revision is reviewed exactly once."""
    state = _loop_state("feature-loop-crash-after-engineer")
    recorder = CheckpointRecorder()
    reviewer = CountingIntegrationReviewer()
    executor = LoopProbeExecutor()
    crashed: list[FeatureWorkflowSnapshot] = []

    class CrashBeforeIntegrationReview(CountingIntegrationReviewer):
        """Die the way a worker does: after every child settled, before the seam is judged."""

        async def review(self, **kwargs: Any) -> IntegrationReviewArtifact:
            """Fail the first time, so the run ends between the children and the verdict."""
            if not crashed:
                crashed.append(state)
                msg = "simulated worker death before integration review"
                raise RuntimeError(msg)
            return await super().review(**kwargs)

    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor),
        integration_reviewer=cast(Any, CrashBeforeIntegrationReview()),
        checkpoint_writer=recorder,
    )
    credentials = _credentials()
    with pytest.raises(RuntimeError):
        await orchestrator.start(state, credentials=credentials)

    interrupted = recorder.last(
        lambda snapshot, boundary, repository_id: all(
            child.status is ChildWorkflowStatus.APPROVED
            for child in snapshot.child_workflows.values()
        )
    )
    assert not _integration_reviews(interrupted)
    engineer_calls_before = dict(executor.engineer_calls)

    recovered = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor), integration_reviewer=cast(Any, reviewer)
    )
    result = await recovered.resume(interrupted, answers=[], credentials=credentials)

    # No repository is implemented a second time, and the integration review runs once over
    # the revisions the interrupted run had already produced.
    assert executor.engineer_calls == engineer_calls_before
    assert reviewer.calls == 1
    assert len(_integration_reviews(result)) == 1


@pytest.mark.asyncio
async def test_recovery_after_a_persisted_verdict_routes_it_once_without_reviewing_again() -> None:
    """Test 7 -- crash after Review, before routing.

    The verdict is appended and checkpointed before its cycle is charged, so this is a real
    crash window: the durable snapshot holds a blocking integration review whose cycle has
    not been spent. Recovery must route that verdict exactly once, not ask for a new one.
    """
    state = _loop_state("feature-loop-crash-after-review")
    recorder = CheckpointRecorder()
    orchestrator = FeatureWorkflowOrchestrator(
        integration_reviewer=cast(Any, CountingIntegrationReviewer()),
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
        checkpoint_writer=recorder,
    )
    credentials = _credentials()
    await orchestrator.start(state, credentials=credentials)

    interrupted = recorder.last(
        lambda snapshot, boundary, repository_id: (
            bool(_integration_reviews(snapshot)) and snapshot.integration_review_cycles == 0
        )
    )
    assert len(_integration_reviews(interrupted)) == 1

    reviewer = CountingIntegrationReviewer()
    recovered = FeatureWorkflowOrchestrator(
        integration_reviewer=cast(Any, reviewer),
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
    )
    result = await recovered.resume(interrupted, answers=[], credentials=credentials)

    assert reviewer.calls == 0
    assert len(_integration_reviews(result)) == 1
    # Routed once: the cycle the crash left uncharged is charged now, and only now.
    assert result.integration_review_cycles == 1
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN


@pytest.mark.asyncio
async def test_recovery_mid_retry_resumes_the_persisted_attempt_rather_than_restarting_it() -> None:
    """Test 8 -- crash after a retry was claimed, before its work meant anything.

    The child loop persists its retry decision at `BEFORE_CODING` before the next model call
    precisely so this window is safe. Recovery must continue from the counters that snapshot
    holds; reading them as zero would hand the repository its whole budget again and repeat
    every coding and validation effect already spent.
    """
    state = _loop_state("feature-loop-crash-mid-retry")
    state.max_implementation_retries = 9
    recorder = CheckpointRecorder()
    executor = LoopProbeExecutor({"backend": ["changes_requested"]})
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor), checkpoint_writer=recorder
    )
    credentials = _credentials()
    await orchestrator.start(state, credentials=credentials)
    assert executor.attempts("backend") == 5

    claimed = recorder.last(
        lambda snapshot, boundary, repository_id: (
            repository_id == "backend" and snapshot.child_workflows["backend"].retry_count == 2
        )
    )
    assert claimed.child_workflows["backend"].implementation_retry_count == 2

    resumed_executor = LoopProbeExecutor({"backend": ["changes_requested"]})
    recovered = FeatureWorkflowOrchestrator(child_executor=cast(Any, resumed_executor))
    result = await recovered.resume(claimed, answers=[], credentials=credentials)

    backend = result.child_workflows["backend"]
    # Three attempts remain of the five this repository is allowed, and it takes exactly
    # those: the persisted counter is honoured rather than reset.
    assert resumed_executor.attempts("backend") == 3
    assert backend.retry_count == 4
    assert backend.implementation_retry_count == 4
    assert backend.status is ChildWorkflowStatus.FAILED
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    # The sibling had not durably settled when this snapshot was written, so recovery does
    # run it -- once. What it must never do is run it once per attempt the other repository
    # still has left.
    assert resumed_executor.attempts("frontend") == 1


@pytest.mark.asyncio
async def test_a_repository_retrying_does_not_rerun_a_sibling_that_already_succeeded() -> None:
    """Test 9 -- successful sibling isolation, asserted on every effect a rerun would have."""
    state = _loop_state("feature-loop-sibling-isolation")
    github = MockGitHubService()
    executor = LoopProbeExecutor(moves=False)
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor),
        integration_reviewer=cast(Any, CountingIntegrationReviewer("backend")),
        pull_request_publisher=GitHubPullRequestPublisher(github_service=github),
    )

    result = await orchestrator.start(state, credentials=_credentials())

    # The sibling is implemented, validated, reviewed and committed once, and its pull
    # request is created once, however many times the other repository is sent back.
    assert executor.attempts("frontend") == 1
    assert executor.reviewer_calls["frontend"] == 1
    assert executor.revisions["frontend"] == ["frontend-rev-frozen"]
    assert executor.attempts("backend") > 1
    assert len(github.pull_requests) == 2
    frontend = result.child_workflows["frontend"]
    assert frontend.retry_count == 0
    assert frontend.integration_retry_count == 0
    assert frontend.retry_refusal_reason is None


@pytest.mark.asyncio
async def test_two_repositories_reworking_at_once_keep_independent_counters() -> None:
    """Test 10 -- parallel repository isolation of counters, fingerprints and refusals.

    One repository moves its revision on every remediation and the other resubmits the same
    one. They must diverge: reaching a limit is a fact about one repository, and one hitting
    its cap must not stop or restart the other.
    """

    class OneMovesOneFrozenExecutor(LoopProbeExecutor):
        """Move the backend's revision on every attempt while the frontend never changes."""

        async def run(self, **kwargs: Any) -> ChildExecution:
            """Freeze only the frontend, so the two repositories are judged differently."""
            self._moves = kwargs["repository"].repository_id == "backend"
            return await super().run(**kwargs)

    state = _loop_state("feature-loop-parallel-counters")
    executor = OneMovesOneFrozenExecutor()
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor),
        integration_reviewer=cast(Any, CountingIntegrationReviewer("backend", "frontend")),
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
    )

    result = await orchestrator.start(state, credentials=_credentials())

    backend = result.child_workflows["backend"]
    frontend = result.child_workflows["frontend"]
    # The frozen repository is refused once its remediation input repeats; the moving one
    # keeps its own allowance, and neither counter is derived from the other.
    assert frontend.retry_refusal_reason is not None
    assert "identical inputs" in frontend.retry_refusal_reason
    assert backend.integration_retry_count > frontend.integration_retry_count
    assert executor.attempts("backend") > executor.attempts("frontend")
    assert backend.production_diff_fingerprint != frontend.production_diff_fingerprint
    # And a repository stopping does not restart or fail the sibling that was still working:
    # both stay bounded by the configured ceiling.
    for child in (backend, frontend):
        assert child.retry_count + 1 <= result.max_child_review_cycles


@pytest.mark.asyncio
async def test_a_verdict_stays_bound_to_the_revision_it_judged() -> None:
    """Test 11 -- stale review. A verdict on revision A is never read as a verdict on B."""
    state = _loop_state("feature-loop-stale-revision")
    executor = LoopProbeExecutor()
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor),
        integration_reviewer=cast(Any, CountingIntegrationReviewer("backend")),
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
    )

    result = await orchestrator.start(state, credentials=_credentials())

    reviews = _integration_reviews(result)
    assert len(reviews) >= 2
    first, second = reviews[0], reviews[1]
    # Each verdict names the revision it actually read, taken from the child result it cites
    # rather than from whatever the repository has since become.
    judged_first = feature_workflow_module._reviewed_revision(result, first, "backend")
    judged_second = feature_workflow_module._reviewed_revision(result, second, "backend")
    assert judged_first == "backend-rev-0"
    assert judged_second == "backend-rev-1"
    assert result.child_workflows["backend"].current_revision != judged_first
    # So the same findings against a different revision are different work, and the same
    # findings against the same revision are not.
    assert feature_workflow_module._remediation_signature(
        result, first, "backend"
    ) != feature_workflow_module._remediation_signature(result, second, "backend")
    assert feature_workflow_module._remediation_signature(
        result, first, "backend"
    ) == feature_workflow_module._remediation_signature(result, first, "backend")

    dangling = first.model_copy(
        update={
            "repository_results": [
                item.model_copy(
                    update={"child_result_artifact_id": "missing-result-for-revision-a"}
                )
                if item.repository_id == "backend"
                else item
                for item in first.repository_results
            ]
        }
    )
    assert feature_workflow_module._reviewed_revision(result, dangling, "backend") == ""
    assert not feature_workflow_module._integration_remediation_decision(
        result, dangling, "backend"
    ).should_retry


@pytest.mark.asyncio
async def test_remediation_fingerprint_tracks_stable_effective_inputs_only() -> None:
    """Validation/repair changes matter; finding order, case and punctuation do not."""
    state = _loop_state("feature-loop-remediation-fingerprint")
    result = await FeatureWorkflowOrchestrator(
        integration_reviewer=cast(Any, CountingIntegrationReviewer("backend")),
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
    ).start(state, credentials=_credentials())
    review = _integration_reviews(result)[0]
    reviewed_result = feature_workflow_module._reviewed_child_result(result, review, "backend")
    assert reviewed_result is not None

    passed = reviewed_result.model_copy(
        update={
            "artifact_id": "011_child_workflow_result.backend.validation-passed.json",
            "current_validation_results": [
                {
                    "name": "repository tests",
                    "passed": True,
                    "repository_revision": reviewed_result.current_revision,
                    "duration_seconds": 1.2,
                    "timestamp": "2026-08-27T10:00:00Z",
                },
                {
                    "name": "repository lint",
                    "passed": True,
                    "repository_revision": reviewed_result.current_revision,
                    "duration_seconds": 0.4,
                    "timestamp": "2026-08-27T10:00:01Z",
                },
            ],
            "retry_strategy": {"required_strategy_change": "inspect the contract"},
        }
    )
    failed = passed.model_copy(
        update={
            "artifact_id": "011_child_workflow_result.backend.validation-failed.json",
            "current_validation_results": [
                {
                    "name": "repository tests",
                    "passed": False,
                    "repository_revision": reviewed_result.current_revision,
                    "duration_seconds": 99.0,
                    "timestamp": "2026-08-27T11:00:00Z",
                }
            ],
        }
    )
    validation_reordered = passed.model_copy(
        update={
            "artifact_id": "011_child_workflow_result.backend.validation-reordered.json",
            "current_validation_results": [
                {
                    **item,
                    "duration_seconds": float(index + 20),
                    "timestamp": f"2026-08-27T12:00:0{index}Z",
                }
                for index, item in enumerate(reversed(passed.current_validation_results))
            ],
        }
    )

    def citing(
        artifact: ChildWorkflowResultArtifact, *, findings: list[IntegrationReviewFinding]
    ) -> IntegrationReviewArtifact:
        return review.model_copy(
            update={
                "repository_results": [
                    item.model_copy(update={"child_result_artifact_id": artifact.artifact_id})
                    if item.repository_id == "backend"
                    else item
                    for item in review.repository_results
                ],
                "cross_repository_findings": findings,
            }
        )

    [base_finding] = [
        item
        for item in review.cross_repository_findings
        if item.responsible_repository_id == "backend"
    ]
    first_finding = base_finding.model_copy(
        update={"finding_id": "stable-a", "recommended_fix": "Update client timeout."}
    )
    second_finding = base_finding.model_copy(
        update={"finding_id": "stable-b", "recommended_fix": "Handle empty responses!"}
    )
    reordered_findings = [
        second_finding.model_copy(
            update={"finding_id": "fresh-b", "recommended_fix": "handle EMPTY responses"}
        ),
        first_finding.model_copy(
            update={"finding_id": "fresh-a", "recommended_fix": "UPDATE client timeout"}
        ),
    ]
    result.artifacts.extend([passed, failed, validation_reordered])
    passed_review = citing(passed, findings=[first_finding, second_finding])
    reordered_review = citing(passed, findings=reordered_findings)
    failed_review = citing(failed, findings=reordered_findings)
    reordered_validation_review = citing(validation_reordered, findings=reordered_findings)

    assert feature_workflow_module._remediation_signature(
        result, passed_review, "backend"
    ) == feature_workflow_module._remediation_signature(result, reordered_review, "backend")
    assert feature_workflow_module._remediation_signature(
        result, passed_review, "backend"
    ) != feature_workflow_module._remediation_signature(result, failed_review, "backend")
    assert feature_workflow_module._remediation_signature(
        result, reordered_review, "backend"
    ) == feature_workflow_module._remediation_signature(
        result, reordered_validation_review, "backend"
    )


@pytest.mark.asyncio
async def test_cancelling_during_the_retry_loop_schedules_no_further_attempt() -> None:
    """Test 12 -- cancellation between an Engineer run and the next one ends the loop."""
    state = _loop_state("feature-loop-cancelled")
    token = MockCancellationToken()

    class CancelAfterTheFirstRejection(LoopProbeExecutor):
        """Request cancellation the way an operator does: while a retry is being prepared."""

        async def run(self, **kwargs: Any) -> ChildExecution:
            """Produce one rejected attempt, then cancel before the next one is scheduled."""
            execution = await super().run(**kwargs)
            if kwargs["repository"].repository_id == "backend":
                token.cancel()
            return execution

    executor = CancelAfterTheFirstRejection({"backend": ["changes_requested"]})
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor), cancellation_token=token
    )

    result = await orchestrator.start(state, credentials=_credentials())

    assert result.status is FeatureWorkflowStatus.CANCELLED
    assert result.cancellation_requested is True
    # No retry follows a cancellation, and none is left scheduled for a resume to pick up.
    assert executor.attempts("backend") == 1
    assert not any(isinstance(item, PullRequestArtifact) for item in result.artifacts)

    resumed = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor), cancellation_token=token
    )
    with pytest.raises(CancellationRequested):
        await resumed.resume(result, answers=[], credentials=_credentials())
    assert executor.attempts("backend") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("script", "moves", "responsible"),
    [
        ({"backend": ["changes_requested"]}, True, ()),
        ({"backend": ["changes_requested"]}, False, ()),
        ({}, False, ("backend",)),
        ({}, False, ("backend", "frontend")),
        ({"backend": ["changes_requested", "approved"]}, True, ("backend",)),
    ],
)
async def test_engineer_and_review_transitions_stay_bounded_by_the_configured_policy(
    script: dict[str, list[str]], moves: bool, responsible: tuple[str, ...]
) -> None:
    """The loop-safety invariant, over every way this cycle has been observed to run away.

    A finite configured workflow must never produce an unbounded number of Engineer/Review
    transitions for one repository. Stated once, here, rather than left implicit in each
    scenario above: every new routing rule has to keep satisfying it.
    """
    state = _loop_state(f"feature-loop-bounded-{len(script)}{int(moves)}{len(responsible)}")
    # Deliberately generous, so the bound being asserted is the per-repository review-cycle
    # ceiling and not some smaller limit that happens to stop the run first.
    state.max_integration_review_cycles = 12
    state.max_implementation_retries = 9
    executor = LoopProbeExecutor(script, moves=moves)
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor),
        integration_reviewer=(
            cast(Any, CountingIntegrationReviewer(*responsible)) if responsible else None
        ),
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
    )

    result = await orchestrator.start(state, credentials=_credentials())

    assert result.status in {
        FeatureWorkflowStatus.COMPLETED,
        FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
    }
    for repository_id in ("backend", "frontend"):
        attempts = executor.attempts(repository_id)
        assert attempts <= result.max_child_review_cycles, repository_id
        assert executor.reviewer_calls.get(repository_id, 0) == attempts
        assert (
            result.child_workflows[repository_id].retry_count + 1 <= result.max_child_review_cycles
        )


class ReviewRejectionNamingAFailedCommand:
    """Resubmit one production diff forever, rejected each time by a reviewer.

    The exact shape of AB-Feature-108's and -114's console. Nothing commits until review
    approves, so every attempt reports `commit_sha=None`; one of the reviewer's findings
    reports a failing command, so `_review_failure_classification` calls the whole attempt
    `validation_source_failure`. Between them those two facts were the entire test for "the
    commit gate rejected this before it produced anything", and an ordinary review rejection
    satisfied both.
    """

    def __init__(self, repository_id: str) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self._repository_id = repository_id
        self.attempts = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Return the identical production source to the identical rejection, every time."""
        execution = await self._delegate.run(**kwargs)
        if kwargs["repository"].repository_id != self._repository_id:
            return execution
        self.attempts += 1
        touched = ["src/pages/AllApps.js"]
        completion = execution.code_completion
        assert completion is not None
        completion = completion.model_copy(
            update={"production_files_changed": touched, "commit_sha": None}
        )
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    "blocking_issues": [
                        "npm run test -- src/pages/AllApps.test.js exited with code 1.",
                        f"The selection handler is still unwired (attempt {self.attempts}).",
                    ],
                    "pull_request_readiness": False,
                    "production_files_changed": touched,
                    "failure_classification": "validation_source_failure",
                    # Byte-identical source every attempt. No `source_validation_rejected`:
                    # the commit gate never ran, a reviewer did.
                    "production_diff_fingerprint": "04e11d7a-identical-every-attempt",
                    "test_diff_fingerprint": "tests-identical-every-attempt",
                }
            ),
            code_completion=completion,
            review=execution.review,
        )


@pytest.mark.asyncio
async def test_a_review_rejection_does_not_borrow_the_pre_commit_allowance() -> None:
    """Sixty-two per cent of recent attempts took an exemption written for a different path.

    The pre-commit allowance exists so a lint rejection -- which resets the workspace and
    therefore reports a fingerprint describing the state it started from -- is not read as an
    attempt that wrote nothing. It was granted on `validation_source_failure` plus an absent
    commit SHA, and neither identifies that path: 82 of the 93 code completions in the last
    fifteen features had no commit SHA, because nothing commits until review approves.

    So 72 of 117 attempts took it when 24 had earned it, and 26 of those 72 resubmitted the
    previous production diff byte for byte with every repeat rule suppressed.
    """
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-review-rejection", request)
    # Generous on purpose: the point is that the loop stops well short of the ceiling.
    state = state.model_copy(update={"max_validation_retries": 8, "max_child_review_cycles": 12})
    executor = ReviewRejectionNamingAFailedCommand("backend")

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    child = result.child_workflows["backend"]
    assert child.status is ChildWorkflowStatus.FAILED
    # The first submission and the two repeats the loop tolerates -- repeating once is a
    # model that did not read its diagnostic, twice is one that cannot act on it. Not the
    # eight the validation budget allows, and nowhere near the twelve review cycles that
    # AB-Feature-108's console actually spent on this exact shape.
    assert executor.attempts == 3
    assert child.retry_count + 1 < state.max_child_review_cycles
    assert child.meaningful_change is False
    # Both repeat rules see it now, and the more specific one names the record.
    assert child.meaningful_change_reason == "attempt_returned_to_an_earlier_submitted_state"


class AlternatesBetweenTwoProductionStates:
    """Cycle between two production diffs, with the classification alternating in step.

    AB-Feature-112's console, exactly. Production fingerprint `04e11d7a` was submitted on
    attempts 0, 2, 4, 7 and 9 -- five copies of one diff -- and the workstream still spent all
    twelve of its review cycles. The circle detector fired on three of them and was reset to
    zero by the attempt in between each time, because a `validation_source_failure` attempt
    took the pre-commit branch and cleared the count on its way through.
    """

    def __init__(self, repository_id: str) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self._repository_id = repository_id
        self.attempts = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Return state A on even attempts and state B on odd ones, forever."""
        execution = await self._delegate.run(**kwargs)
        if kwargs["repository"].repository_id != self._repository_id:
            return execution
        self.attempts += 1
        even = self.attempts % 2 == 1
        touched = ["src/pages/AllApps.js", "src/components/apps/BulkDeleteApps.js"]
        completion = execution.code_completion
        assert completion is not None
        completion = completion.model_copy(
            update={"production_files_changed": touched, "commit_sha": None}
        )
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    "blocking_issues": [
                        f"The bulk-delete button is still not reachable (attempt {self.attempts})."
                    ],
                    "pull_request_readiness": False,
                    "production_files_changed": touched,
                    "failure_classification": (
                        "implementation_missing" if even else "validation_source_failure"
                    ),
                    "production_diff_fingerprint": "state-a" if even else "state-b",
                    "test_diff_fingerprint": "tests-unchanged",
                }
            ),
            code_completion=completion,
            review=execution.review,
        )


@pytest.mark.asyncio
async def test_a_workstream_cycling_between_two_states_does_not_spend_every_review_cycle() -> None:
    """Returning to a state it already submitted is going in a circle, whoever rejected it.

    The circle detector was disabled whenever the attempt was classified
    `validation_source_failure`, and its counter was cleared by the same attempts, so a
    workstream alternating between two diffs could never accumulate the two consecutive
    repeats needed to stop it. AB-Feature-112 and -113 each burned all twelve review cycles
    that way and ended `failed_requires_human`.
    """
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-two-state-circle", request)
    state = state.model_copy(update={"max_validation_retries": 12, "max_child_review_cycles": 12})
    executor = AlternatesBetweenTwoProductionStates("backend")

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    child = result.child_workflows["backend"]
    assert child.status is ChildWorkflowStatus.FAILED
    # It has to notice the circle, which needs the third attempt to return to state A. It
    # must not need twelve.
    assert 3 <= executor.attempts <= 5, executor.attempts
    assert child.retry_count + 1 < state.max_child_review_cycles


@pytest.mark.asyncio
async def test_a_genuine_pre_commit_rejection_keeps_the_allowance_it_was_written_for() -> None:
    """Narrowing the allowance must not take it away from the path that earned it.

    A commit-gate rejection resets the workspace before anything is committed, so the
    fingerprint taken afterwards describes the state the attempt started from rather than the
    source it wrote. Reading that as no progress stranded rep-a's frontend three attempts into
    eight while its source was being rewritten between them.
    """
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-real-pre-commit", request)
    state = state.model_copy(update={"max_validation_retries": 8})
    executor = UncommittedLintRejectionExecutor(rejections=3)

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    # Three rejections and the attempt that clears them, none refused for repeating itself.
    assert executor.attempts == 4
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED


@pytest.mark.asyncio
async def test_cancelling_settles_a_repository_whose_turn_never_came() -> None:
    """A workstream waiting behind a dependency is `pending` when cancellation ends the loop.

    Nothing will schedule it now, so `pending` is not a turn that is coming -- it is a row
    nobody will ever update, and an operator, the features list and the workflow graph all
    read it as work still to do. AB-Feature-107 held one of these for a day, in `running`.
    """
    payload = feature_payload()
    payload["repositories"] = [
        _repository("shared-lib", "shared-lib", role="shared"),
        _repository("backend", "backend", role="backend"),
    ]
    state = _initial_feature_state(
        "feature-cancelled-before-its-turn", StartFeatureRequest.model_validate(payload)
    )
    token = MockCancellationToken()

    class CancelOnceTheDependencyIsBuilt:
        """Cancel while the repository that waits for the shared library has not started."""

        def __init__(self) -> None:
            self._delegate = MockChildWorkstreamExecutor()
            self.ran: list[str] = []

        async def run(self, **kwargs: Any) -> ChildExecution:
            repository_id = kwargs["repository"].repository_id
            self.ran.append(repository_id)
            execution = await self._delegate.run(**kwargs)
            token.cancel()
            return execution

    executor = CancelOnceTheDependencyIsBuilt()
    result = await FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor), cancellation_token=token
    ).start(state, credentials=_credentials())

    assert result.status is FeatureWorkflowStatus.CANCELLED
    # The dependent repository never ran, which is the whole point: cancellation schedules
    # no new work. Its row must still say something true about that.
    assert executor.ran == ["shared-lib"]
    assert result.child_workflows["backend"].status is not ChildWorkflowStatus.PENDING
    assert not [
        repository_id
        for repository_id, child in result.child_workflows.items()
        if child.status in {ChildWorkflowStatus.RUNNING, ChildWorkflowStatus.PENDING}
    ], {key: item.status for key, item in result.child_workflows.items()}


class CirclesBackThroughVariedStates:
    """Return to one state repeatedly, with a genuinely new state between each visit.

    AB-Feature-124's console, exactly: production fingerprint 04199e3a on attempts 1, 3 and
    5, with 53188603 and ac8cbe16 in between. It added an unreachable page, wired it, undid
    the wiring, wired it differently, undid it again -- arriving three times at a state it
    had already submitted, each arrival correctly detected and each one forgotten before the
    next, because the consecutive counter was cleared by the new attempt in between.
    """

    _STATES = ["state-a", "state-b", "state-a", "state-c", "state-a", "state-d", "state-a"]

    def __init__(self, repository_id: str) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self._repository_id = repository_id
        self.attempts = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Walk the recorded fingerprint sequence, rejected with a fresh finding each time."""
        execution = await self._delegate.run(**kwargs)
        if kwargs["repository"].repository_id != self._repository_id:
            return execution
        index = self.attempts
        self.attempts += 1
        fingerprint = self._STATES[min(index, len(self._STATES) - 1)]
        touched = ["src/pages/Home.js", "src/pages/HomeWithServiceStatus.js"]
        completion = execution.code_completion
        assert completion is not None
        completion = completion.model_copy(
            update={"production_files_changed": touched, "commit_sha": None}
        )
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    "blocking_issues": [
                        f"The status indicator is still not reachable (attempt {index})."
                    ],
                    "pull_request_readiness": False,
                    "production_files_changed": touched,
                    "failure_classification": "implementation_missing",
                    "production_diff_fingerprint": fingerprint,
                    "test_diff_fingerprint": "tests-unchanged",
                }
            ),
            code_completion=completion,
            review=execution.review,
        )


@pytest.mark.asyncio
async def test_returning_to_an_earlier_state_is_counted_even_with_new_work_in_between() -> None:
    """A circle with variety in it is still a circle, and the repeat counter could not see one.

    The revisit detector looks back at every earlier attempt, so it caught each return. What
    stopped it mattering was the bound: two *consecutive* repeats, cleared by any attempt that
    produced something new. AB-Feature-124's console alternated a genuinely new state with a
    return to an old one and ran to its ceiling while every single return was being detected
    and discarded.
    """
    payload = feature_payload()
    payload["repositories"] = [_repository("backend", "backend")]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-varied-circle", request)
    state = state.model_copy(
        update={"max_implementation_retries": 12, "max_child_review_cycles": 12}
    )
    executor = CirclesBackThroughVariedStates("backend")

    result = await FeatureWorkflowOrchestrator(child_executor=executor).start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    child = result.child_workflows["backend"]
    assert child.status is ChildWorkflowStatus.FAILED
    # a0 new, a1 new, a2 returns to a0's state (1st), a3 new, a4 returns (2nd) -> stop.
    # Not the twelve the ceiling would otherwise have allowed.
    assert executor.attempts == 5, executor.attempts
    assert child.retry_count + 1 < state.max_child_review_cycles
    assert child.meaningful_change is False
    assert child.meaningful_change_reason == "attempt_returned_to_an_earlier_submitted_state"


def test_an_approval_survives_a_later_failed_attempt_for_the_integration_gate() -> None:
    """The gate reads the revision publication reads, not whatever artifact came last.

    AB-Feature-152's console passed review at attempt 1. The integration gate asked for a
    change, attempt 2 failed, and the next gate pass reported "No child workflow result was
    produced for a contract owner" as a critical finding against it -- while both pull
    requests were being opened, one of them from the very approval the gate could no longer
    see. A false critical finding is worse than a missing one: it blocks a feature and sends
    its reader hunting for work that was never missing.
    """
    payload = feature_payload()
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state("feature-approval-survives", request)

    def result(repository_id: str, attempt: int, status: str) -> ChildWorkflowResultArtifact:
        return create_artifact(
            ChildWorkflowResultArtifact,
            workflow_id=state.feature_id,
            artifact_id=f"011_child_workflow_result.{repository_id}.attempt-{attempt}.json",
            producer="child_workflow",
            payload={
                "feature_id": state.feature_id,
                "parent_workflow_id": state.feature_id,
                "child_workflow_id": f"{state.feature_id}:{repository_id}",
                "repository_id": repository_id,
                "workstream_id": repository_id,
                "branch_name": f"ai/{state.feature_id}/{repository_id}/x",
                "workspace_path": f"/workspaces/{repository_id}",
                "status": status,
                "blocking_issues": [] if status == "approved" else ["the rework failed"],
                "pull_request_readiness": status == "approved",
                "contract_sections_consumed": [],
                "contract_sections_implemented": [],
                "code_completion_artifact_id": None,
                "review_artifact_id": None,
                "changed_files": [],
                "validation_results": [],
            },
            metadata={"child_retry_count": attempt},
        )

    # The console's own history: rejected, approved, then a failed integration remediation.
    state.artifacts.extend(
        [
            result("backend", 0, "failed"),
            result("backend", 1, "approved"),
            result("frontend", 0, "approved"),
            result("backend", 2, "failed"),
        ]
    )

    selected = feature_workflow_module._latest_approved_child_results(state)

    assert {item.repository_id for item in selected} == {"backend", "frontend"}
    backend = next(item for item in selected if item.repository_id == "backend")
    # The approval, not the failure that followed it -- the same revision publication uses.
    assert backend.artifact_id.endswith("attempt-1.json")
    assert backend.status == "approved"


class ProviderFaultThenPlanPlanner:
    """Fail with a provider fault a fixed number of times, then plan deterministically."""

    def __init__(self, *, faults: int) -> None:
        """Count every call so the test can prove the retries actually happened."""
        self._delegate = DeterministicFeaturePlanner()
        self._remaining = faults
        self.calls = 0

    async def plan(self, **kwargs: Any) -> Any:
        """Drop the connection the way a long maximum-effort planning call really does."""
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            msg = "the model provider call failed (APIConnectionError)"
            raise LLMAdapterError(msg, diagnostics=(msg,))
        return await self._delegate.plan(**kwargs)


@pytest.mark.asyncio
async def test_planning_survives_a_provider_fault_the_way_a_workstream_already_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dropped connection during planning says nothing about the feature being planned.

    The allowance existed only inside the child loop, so the identical fault was survivable
    once a repository was running and fatal in the stage before it. AB-Feature-168 spent a
    morning proving it: three planning runs, twenty to forty-five minutes each, every one
    ended by the provider and every one handed back to its author to press resume.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)
    state = _initial_feature_state("feature-planning-fault", request)
    planner = ProviderFaultThenPlanPlanner(faults=2)
    orchestrator = FeatureWorkflowOrchestrator(planner=cast(Any, planner))

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert planner.calls == 3, "the two faults must each have earned the call again"
    assert result.status is FeatureWorkflowStatus.COMPLETED
    # No trace of the faults in the record: they were the provider's, and the feature planned.
    assert result.failure_summary is None


@pytest.mark.asyncio
async def test_a_provider_that_stays_down_reaches_a_human_rather_than_retrying_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The allowance is bounded, because a provider that is down stays down.

    The fault is re-raised rather than swallowed, so the classification, the summary and the
    queue's own attempt budget all still see it -- this retries a call, it does not decide
    what a feature does when the provider is genuinely unavailable.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)
    state = _initial_feature_state("feature-planning-outage", request)
    planner = ProviderFaultThenPlanPlanner(faults=99)
    orchestrator = FeatureWorkflowOrchestrator(planner=cast(Any, planner))

    with pytest.raises(LLMAdapterError) as fault:
        await orchestrator.start(
            state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
        )

    assert planner.calls == 3, "one call plus its bounded allowance, and no more"
    # The original fault, not a wrapper. Everything downstream keys on it: the queue spends
    # an attempt, and the summary tells its reader the provider failed rather than the work.
    assert safe_error_diagnostics(fault.value) == (
        "the model provider call failed (APIConnectionError)",
    )
    assert is_transient_provider_fault(fault.value)


def test_a_deterministic_provider_answer_is_not_a_transient_fault() -> None:
    """A refusal, a truncation, or a 4xx returns identically on every retry.

    AB-Feature-174's backend spent three identical twelve-minute coding calls on one
    truncated response because the predicate keyed on the exception type alone;
    AB-Feature-172 spent nine calls on one 400 the same way.
    """
    for classification in (
        "response_truncated",
        "model_refusal",
        "BadRequestError",
        "AuthenticationError",
        "NotFoundError",
    ):
        direct = LLMAdapterError(
            "the model provider call failed",
            diagnostics=("the model provider call failed",),
            failure_classification=classification,
        )
        assert not is_transient_provider_fault(direct), classification
        # The coding call's adapter error reaches the child loop wrapped in
        # ExternalOperationError, so the classification must be found on the cause chain.
        wrapped = ExternalOperationError("external operation failed before a confirmed result")
        wrapped.__cause__ = direct
        assert not is_transient_provider_fault(wrapped), f"wrapped {classification}"


def test_a_transient_provider_fault_still_earns_the_attempt_again() -> None:
    """Timeouts, connection drops, and rate limits keep the existing allowance."""
    for classification in ("ReadTimeout", "APIConnectionError", "RateLimitError", None):
        error = LLMAdapterError(
            "the model provider call failed",
            diagnostics=("the model provider call failed",),
            failure_classification=classification,
        )
        assert is_transient_provider_fault(error), classification or "unclassified"


def test_a_lease_lost_on_a_workspace_local_operation_earns_the_attempt_again() -> None:
    """AB-Feature-167: the reviewer's own operation lost its lease and got zero retries.

    Nothing outside the workspace could have happened here -- `run_reviewer` is one of the
    operation types `services/recovery_service.py`'s own policy table already calls
    WORKSPACE_LOCAL, and its periodic sweep would have reached the identical conclusion,
    just too late to matter once the workflow had already given up.
    """
    error = OperationLeaseLostError("external operation heartbeat could not be renewed")
    error.operation_type = ExternalOperationType.RUN_REVIEWER
    assert is_transient_provider_fault(error)


@pytest.mark.parametrize(
    "operation_type",
    [
        ExternalOperationType.CREATE_PULL_REQUEST,  # DEFER_TO_CREDENTIALED
        ExternalOperationType.PUSH_BRANCH,  # DEFER_TO_CREDENTIALED
        ExternalOperationType.CLONE_REPOSITORY,  # LOCAL_RECONCILE -- settles from evidence
    ],
)
def test_a_lease_lost_on_an_external_or_reconcilable_operation_is_still_never_retried(
    operation_type: ExternalOperationType,
) -> None:
    """Widening the workspace-local case must not widen the guard around anything else."""
    error = OperationLeaseLostError("external operation heartbeat could not be renewed")
    error.operation_type = operation_type
    assert not is_transient_provider_fault(error)


def test_a_lease_lost_with_no_operation_type_recorded_is_still_never_retried() -> None:
    """Every pre-existing raise site that never learned about this field keeps today's answer."""
    assert not is_transient_provider_fault(OperationLeaseLostError("lease lost"))


class FixAttemptFailsUnrelatedExecutor:
    """Approve every first attempt, then fail the backend's routed fix attempt off-topic.

    The failure is deliberately *unrelated* to the repository -- a workspace fault, not a
    finding -- because that is AB-Feature-184's terminal shape: the integration fix attempt
    died on the platform, and the work both repositories had already passed review on was
    what still deserved publication.
    """

    def __init__(self) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self.backend_attempts = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        execution = await self._delegate.run(**kwargs)
        if kwargs["repository"].repository_id != "backend":
            return execution
        self.backend_attempts += 1
        if self.backend_attempts == 1:
            return execution
        failed = execution.result.model_copy(
            update={
                "status": "failed",
                "blocking_issues": [
                    "The workspace volume filled while this fix attempt ran. Nothing here "
                    "is a finding about this repository."
                ],
                "pull_request_readiness": False,
                # Distinct per attempt so the convergence guards judge only the failure.
                "production_diff_fingerprint": f"fix-attempt-{self.backend_attempts}",
            }
        )
        return ChildExecution(result=failed, code_completion=execution.code_completion)


@pytest.mark.asyncio
async def test_an_unrelated_fix_attempt_failure_still_publishes_the_approved_work() -> None:
    """The publication invariant across an integration fix attempt that dies off-topic.

    Both repositories pass their own reviews; the integration review asks the backend for
    one fix; the granted fix attempt fails for a reason that says nothing about the
    repository. The approved work -- the sibling's and the backend's own reviewed commit --
    must still reach pull requests, exactly as AB-Feature-184's salvage did at 05:22:38.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-fix-attempt-dies", request)
    executor = FixAttemptFailsUnrelatedExecutor()
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=executor,
        integration_reviewer=cast(
            Any, RejectingIntegrationReviewer(responsible_repository_id="backend")
        ),
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
    )

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert executor.backend_attempts >= 2, "the routed fix attempt must actually have run"
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert not _published_repository_ids(result), "a feature that did not land publishes nothing"
    result = await _publish_on_request(result, orchestrator=orchestrator)
    pull_requests = [
        artifact for artifact in result.artifacts if isinstance(artifact, PullRequestArtifact)
    ]
    assert {artifact.repository for artifact in pull_requests} == {
        "example/backend",
        "example/frontend",
    }
    assert result.child_workflows["frontend"].pull_request_artifact_id is not None
    assert result.child_workflows["backend"].pull_request_artifact_id is not None
