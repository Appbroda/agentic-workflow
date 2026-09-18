"""The single-repository ``/workflow/*`` surface, served by the feature path.

There is one orchestration engine. A single-repository workflow is a feature with one
repository, and this module is the only place that knows it -- it translates a start request
into a feature, and a feature's state back into the vocabulary ``/workflow/*`` has always
answered in.

The translation lives here rather than in the domain on purpose. ``WorkflowStatus`` and
``FeatureWorkflowStatus`` are different enums with overlapping members: the API contract is the
old one and the truth is the new one, and letting the old vocabulary back into the domain would
be reintroducing the second engine one enum at a time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from api.control_plane import (
    ClarificationValidationError,
    RequestScopedCredentials,
    WorkflowConflictError,
)
from api.feature_control_plane import FeatureControlPlane, FeatureRecord
from api.feature_schemas import StartFeatureRequest
from api.identity import PLATFORM_ADMIN_ID, WorkspaceScope
from api.schemas import ClarificationAnswer, StartWorkflowRequest
from artifacts.schemas import Artifact, TechnicalPRDArtifact
from state.enums import ApprovalState, FeatureWorkflowStatus, WorkflowStatus
from state.feature_models import FeatureWorkflowSnapshot
from state.models import ExecutionLogEntry

# What the caller's workspace descriptor is kept under. A single-repository request still
# names a workspace id, a checkout path and a working branch, and the response still reports
# the first of them -- but the feature path provisions its own workspace and names its own
# branches, so the other two are audit data rather than instructions.
WORKSPACE_DESCRIPTOR_METADATA_KEY = "single_repository_workspace_descriptor"

# `WorkflowStatus` predates feature workflows and has fewer members. Everything the feature
# path does between accepting the work and finishing it reads as `running` to a caller that
# only ever knew about one repository, which is what it always read as.
_STATUS_VOCABULARY: dict[FeatureWorkflowStatus, WorkflowStatus] = {
    FeatureWorkflowStatus.PENDING: WorkflowStatus.PENDING,
    FeatureWorkflowStatus.ANALYZING_PRD: WorkflowStatus.RUNNING,
    FeatureWorkflowStatus.WAITING_FOR_HUMAN: WorkflowStatus.WAITING_FOR_HUMAN,
    FeatureWorkflowStatus.INSPECTING_REPOSITORIES: WorkflowStatus.RUNNING,
    FeatureWorkflowStatus.PLANNING: WorkflowStatus.RUNNING,
    FeatureWorkflowStatus.CONTRACT_READY: WorkflowStatus.RUNNING,
    FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS: WorkflowStatus.RUNNING,
    FeatureWorkflowStatus.INTEGRATION_REVIEW: WorkflowStatus.RUNNING,
    # The one member with a real counterpart rather than a widening: a review that asked for
    # changes is exactly what `review_rejected` meant.
    FeatureWorkflowStatus.CHANGES_REQUESTED: WorkflowStatus.REVIEW_REJECTED,
    FeatureWorkflowStatus.READY_FOR_PULL_REQUESTS: WorkflowStatus.RUNNING,
    FeatureWorkflowStatus.CREATING_PULL_REQUESTS: WorkflowStatus.RUNNING,
    FeatureWorkflowStatus.COMPLETED: WorkflowStatus.COMPLETED,
    FeatureWorkflowStatus.FAILED: WorkflowStatus.FAILED,
    FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN: WorkflowStatus.FAILED_REQUIRES_HUMAN,
    FeatureWorkflowStatus.CANCELLING: WorkflowStatus.CANCELLING,
    FeatureWorkflowStatus.CANCELLED: WorkflowStatus.CANCELLED,
    FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS: (
        WorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS
    ),
}

_APPROVAL_STATE: dict[WorkflowStatus, ApprovalState] = {
    WorkflowStatus.WAITING_FOR_HUMAN: ApprovalState.PENDING,
    WorkflowStatus.REVIEW_REJECTED: ApprovalState.REJECTED,
    WorkflowStatus.FAILED: ApprovalState.REJECTED,
    WorkflowStatus.FAILED_REQUIRES_HUMAN: ApprovalState.REJECTED,
    WorkflowStatus.COMPLETED: ApprovalState.APPROVED,
}

# Cancelling a workflow that has already settled has always been a conflict rather than a
# no-op on this surface, and a caller must not be able to tell that the engine underneath
# changed. The feature path answers the same request idempotently, deliberately, so the
# refusal is preserved here rather than by changing what a feature does.
_UNCANCELLABLE = frozenset(
    {
        FeatureWorkflowStatus.COMPLETED,
        FeatureWorkflowStatus.CANCELLED,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    }
)

_UNRESUMABLE = frozenset(
    {
        FeatureWorkflowStatus.COMPLETED,
        FeatureWorkflowStatus.CANCELLED,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    }
)


@dataclass(frozen=True, slots=True)
class WorkflowRecord:
    """One workflow's public lifecycle snapshot, in the vocabulary the API answers in."""

    workflow_id: str
    status: WorkflowStatus
    current_agent: str | None
    confidence: float
    retry_count: int
    approval_state: ApprovalState
    workspace_id: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class StartWorkflowResult:
    """The record produced by a start operation and whether it created new work."""

    record: WorkflowRecord
    created: bool


class WorkflowControlPlane(Protocol):
    """The lifecycle and read surface ``/workflow/*`` is served from."""

    async def start(
        self,
        request: StartWorkflowRequest,
        *,
        idempotency_key: str | None,
        credentials: RequestScopedCredentials,
    ) -> StartWorkflowResult:
        """Create or replay a workflow start operation."""

    async def resume(
        self,
        workflow_id: str,
        *,
        answers: Sequence[ClarificationAnswer],
        credentials: RequestScopedCredentials,
    ) -> WorkflowRecord:
        """Resume a known paused workflow."""

    async def cancel(self, workflow_id: str) -> WorkflowRecord:
        """Cancel an active workflow."""

    async def get_record(self, workflow_id: str) -> WorkflowRecord:
        """Read a workflow snapshot."""

    async def artifacts(self, workflow_id: str) -> list[Artifact]:
        """Read workflow artifacts."""

    async def logs(self, workflow_id: str) -> list[ExecutionLogEntry]:
        """Read workflow logs."""

    async def timeline(
        self, workflow_id: str
    ) -> list[tuple[datetime, str, str, str, dict[str, Any]]]:
        """Read chronological workflow events."""


class FeatureBackedWorkflowControlPlane:
    """Serve ``/workflow/*`` from the feature control plane.

    ``execution_mode`` is bound at construction because the request schema has no field for
    it and must not grow one. It is the same distinction the two compositions always had: a
    deployment runs this surface against live providers, an isolated application runs it in
    mock mode and reaches nothing.

    **Deprecated, and confined to one workspace.** This surface predates both features and
    workspaces: it has no owner in its request schema, no owner in its response schema, and
    no notion of one anywhere in its vocabulary. It cannot be scoped to its caller, because
    it does not know who that is in any sense this platform can act on.

    So it is confined to `platform-admin`'s workspace instead. Every feature it creates is
    owned by that id, every feature it reads must be owned by that id, and the router that
    exposes it requires an administrator. Which is internally consistent: this surface
    operates on exactly the features it created, plus the ones migration 0030 backfilled to
    the same owner -- which is every workflow that existed before workspaces did.

    Left in place rather than removed: `docs/USER_GUIDE.md`, `docs/DEPLOYMENT.md` and the
    canary all use it, and break-glass access to a single-repository run is a real
    requirement. New work goes through `/features/*`, which is owner-scoped.
    """

    def __init__(
        self,
        features: FeatureControlPlane,
        *,
        execution_mode: str = "mock",
    ) -> None:
        """Bind the one control plane this surface delegates to, and the workspace it uses."""
        self._features = features
        self._execution_mode = execution_mode
        # Not `unscoped()`. An unscoped delegate would make this a fully open read path over
        # every workspace's features, reachable by anybody the router lets through -- which
        # is what it was before this was written down.
        self._scope = WorkspaceScope(owner_id=PLATFORM_ADMIN_ID)

    async def start(
        self,
        request: StartWorkflowRequest,
        *,
        idempotency_key: str | None,
        credentials: RequestScopedCredentials,
    ) -> StartWorkflowResult:
        """Create or replay a single-repository feature and answer in workflow vocabulary."""
        result = await self._features.start(
            _feature_request(request, execution_mode=self._execution_mode),
            idempotency_key=idempotency_key,
            credentials=credentials,
            owner_id=PLATFORM_ADMIN_ID,
        )
        return StartWorkflowResult(record=_record(result.record), created=result.created)

    async def resume(
        self,
        workflow_id: str,
        *,
        answers: Sequence[ClarificationAnswer],
        credentials: RequestScopedCredentials,
    ) -> WorkflowRecord:
        """Validate the clarification contract, then resume the feature underneath."""
        record = await self._features.get_record(workflow_id, scope=self._scope)
        _validate_resumable(record.state, answers)
        resumed = await self._features.resume(
            workflow_id, answers=answers, credentials=credentials, scope=self._scope
        )
        return _record(resumed)

    async def cancel(self, workflow_id: str) -> WorkflowRecord:
        """Cancel an active workflow, refusing one that has already settled."""
        record = await self._features.get_record(workflow_id, scope=self._scope)
        if record.state.status in _UNCANCELLABLE:
            msg = "completed or cancelled workflows cannot be cancelled again"
            raise WorkflowConflictError(msg)
        return _record(
            await self._features.cancel(
                workflow_id, requested_by=PLATFORM_ADMIN_ID, scope=self._scope
            )
        )

    async def get_record(self, workflow_id: str) -> WorkflowRecord:
        """Read one workflow snapshot."""
        return _record(await self._features.get_record(workflow_id, scope=self._scope))

    async def artifacts(self, workflow_id: str) -> list[Artifact]:
        """Return the artifact handoffs this workflow has produced, in emission order."""
        return await self._features.artifacts(workflow_id, scope=self._scope)

    async def logs(self, workflow_id: str) -> list[ExecutionLogEntry]:
        """Return structured execution logs.

        Always empty, and deliberately so: structured execution evidence is recorded against
        the feature's own artifacts and events, and inventing entries here would be this
        surface reporting something no engine writes.
        """
        await self._features.get_record(workflow_id, scope=self._scope)
        return []

    async def timeline(
        self, workflow_id: str
    ) -> list[tuple[datetime, str, str, str, dict[str, Any]]]:
        """Return chronological lifecycle and artifact events in workflow vocabulary."""
        return [
            (timestamp, event_type, source, _workflow_event(event), details)
            for timestamp, event_type, source, event, details in await self._features.timeline(
                workflow_id, scope=self._scope
            )
        ]


def _feature_request(request: StartWorkflowRequest, *, execution_mode: str) -> StartFeatureRequest:
    """Express one single-repository workflow request as a one-repository feature."""
    descriptor = request.workspace_descriptor
    return StartFeatureRequest.model_validate(
        {
            "feature_id": request.workflow_id,
            "prd": request.prd.model_dump(mode="json"),
            "repositories": [
                {
                    # Identity is derived from the URL by the request model, the same way it
                    # is for a feature submitted directly. The caller's workspace id is not
                    # used for it: `workspace_id` is a free-form string on this surface and a
                    # repository id is not, so borrowing it would turn an accepted request
                    # into a rejected one.
                    "repository_url": str(descriptor.source_repo_url),
                    "default_branch": descriptor.default_branch,
                    "required": True,
                    "metadata": {
                        WORKSPACE_DESCRIPTOR_METADATA_KEY: descriptor.model_dump(mode="json")
                    },
                }
            ],
            "execution_mode": execution_mode,
        }
    )


def _record(record: FeatureRecord) -> WorkflowRecord:
    """Project a feature snapshot onto the workflow response fields."""
    state = record.state
    status = _STATUS_VOCABULARY[state.status]
    child = next(iter(state.child_workflows.values()), None)
    return WorkflowRecord(
        workflow_id=state.workflow_id,
        status=status,
        current_agent=state.current_agent,
        # No engine has ever written one. `create_initial_agent_state` set it to zero and
        # nothing moved it, so reporting it as anything else would be inventing a number.
        confidence=0.0,
        retry_count=child.retry_count if child is not None else 0,
        approval_state=_APPROVAL_STATE.get(status, ApprovalState.NOT_REQUIRED),
        workspace_id=_workspace_id(state),
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _workspace_id(state: FeatureWorkflowSnapshot) -> str:
    """Return the workspace identity the caller supplied, or the repository's own."""
    for spec in state.repository_specs:
        descriptor = spec.metadata.get(WORKSPACE_DESCRIPTOR_METADATA_KEY)
        if isinstance(descriptor, dict):
            workspace_id = descriptor.get("workspace_id")
            if isinstance(workspace_id, str) and workspace_id:
                return workspace_id
        return spec.repository_id
    return state.feature_id


def _workflow_event(event: str) -> str:
    """Answer in the event vocabulary this surface has always used."""
    return f"workflow_{event.removeprefix('feature_')}" if event.startswith("feature_") else event


def _validate_resumable(
    state: FeatureWorkflowSnapshot, answers: Sequence[ClarificationAnswer]
) -> None:
    """Apply the clarification-answer contract this surface has always enforced."""
    if state.status in _UNRESUMABLE:
        msg = "cancelled or completed workflows cannot be resumed"
        raise WorkflowConflictError(msg)
    if state.status is not FeatureWorkflowStatus.WAITING_FOR_HUMAN:
        if answers:
            msg = "clarification answers are accepted only while waiting for human input"
            raise WorkflowConflictError(msg)
        return
    technical_prd = next(
        (item for item in reversed(state.artifacts) if isinstance(item, TechnicalPRDArtifact)),
        None,
    )
    if technical_prd is None:
        msg = "workflow is not awaiting technical-PRD clarification"
        raise WorkflowConflictError(msg)
    unresolved = {question.question_id for question in technical_prd.unresolved_questions}
    if not unresolved:
        msg = "workflow has no unresolved clarification questions"
        raise WorkflowConflictError(msg)
    if {answer.question_id for answer in answers} != unresolved:
        msg = "clarification answers must match unresolved question IDs exactly"
        raise ClarificationValidationError(msg)


__all__ = [
    "WORKSPACE_DESCRIPTOR_METADATA_KEY",
    "FeatureBackedWorkflowControlPlane",
    "StartWorkflowResult",
    "WorkflowControlPlane",
    "WorkflowRecord",
]
