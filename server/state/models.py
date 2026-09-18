"""Typed LangGraph workflow state and its structured supporting models."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from operator import add
from pathlib import Path
from typing import Annotated, Any, NotRequired, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypedDict

from artifacts.schemas import Artifact
from runtime_identity import load_runtime_identity
from state.enums import ApprovalState, ConversationEventType, LogLevel, WorkflowStatus

type NonEmptyString = Annotated[str, Field(min_length=1)]


class StateModel(BaseModel):
    """Base model that makes state data explicit and rejects undeclared fields."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class ConversationEntry(StateModel):
    """A structured workflow event, rather than an untyped chat message."""

    entry_id: NonEmptyString
    sender: NonEmptyString
    recipient: NonEmptyString | None
    event_type: ConversationEventType
    artifact_ids: list[NonEmptyString]
    question_ids: list[NonEmptyString]
    timestamp: datetime
    metadata: dict[str, Any]

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_include_timezone(cls, value: datetime) -> datetime:
        """Require timestamps that remain unambiguous in distributed workflows."""
        if value.tzinfo is None or value.utcoffset() is None:
            msg = "timestamp must be timezone-aware"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def references_must_be_unique(self) -> Self:
        """Keep references in a history event unambiguous."""
        _require_unique(self.artifact_ids, "artifact")
        _require_unique(self.question_ids, "question")
        return self


class WorkspaceDescriptor(StateModel):
    """The repository workspace that an engineering workflow is allowed to change."""

    workspace_id: NonEmptyString
    root_path: NonEmptyString
    source_repo_url: NonEmptyString
    default_branch: NonEmptyString
    working_branch: NonEmptyString
    # The branch the checkout is created from when it is not the default: a feature
    # revision builds on the superseded run's branch. `None` -- every workspace before
    # revisions existed -- means the default branch, so nothing changes for them.
    base_branch: str | None = None

    @field_validator("root_path")
    @classmethod
    def root_path_must_be_absolute(cls, value: str) -> str:
        """Prevent a resumed workflow from resolving its workspace relative to its process."""
        if not Path(value).is_absolute():
            msg = "root_path must be absolute"
            raise ValueError(msg)
        return value

    @field_validator("source_repo_url")
    @classmethod
    def source_repo_url_must_not_embed_credentials(cls, value: str) -> str:
        """Reject body-persisted credentials before a workspace descriptor is durable."""
        parsed = urlsplit(value)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            msg = "source_repo_url must not contain credentials, query parameters, or fragments"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def branches_must_differ(self) -> Self:
        """Ensure the engineer works on a non-default branch of its own."""
        if self.default_branch == self.working_branch:
            msg = "working_branch must differ from default_branch"
            raise ValueError(msg)
        if self.base_branch is not None and self.base_branch == self.working_branch:
            msg = "working_branch must differ from base_branch"
            raise ValueError(msg)
        return self


class ExecutionLogEntry(StateModel):
    """A structured execution event kept in the live LangGraph state."""

    log_id: NonEmptyString
    timestamp: datetime
    level: LogLevel
    source: NonEmptyString
    event: NonEmptyString
    message: NonEmptyString
    context: dict[str, Any]

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_include_timezone(cls, value: datetime) -> datetime:
        """Require timestamps that preserve ordering across system boundaries."""
        if value.tzinfo is None or value.utcoffset() is None:
            msg = "timestamp must be timezone-aware"
            raise ValueError(msg)
        return value


class WorkflowCheckpoint(StateModel):
    """A recovery point that records the workflow position and known artifacts."""

    checkpoint_id: NonEmptyString
    status: WorkflowStatus
    created_at: datetime
    artifact_ids: list[NonEmptyString]
    retry_count: Annotated[int, Field(ge=0)]

    @field_validator("created_at")
    @classmethod
    def created_at_must_include_timezone(cls, value: datetime) -> datetime:
        """Require an unambiguous recovery timestamp."""
        if value.tzinfo is None or value.utcoffset() is None:
            msg = "created_at must be timezone-aware"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def artifact_ids_must_be_unique(self) -> Self:
        """Ensure a checkpoint refers to each artifact only once."""
        _require_unique(self.artifact_ids, "checkpoint artifact")
        return self


class AgentState(TypedDict):
    """The complete LangGraph state for one workflow execution."""

    workflow_id: str
    workflow_schema_version: str
    created_by_build_revision: str
    last_executor_build_revision: str
    current_step: WorkflowStatus
    current_agent: str | None
    conversation_history: Annotated[list[ConversationEntry], add]
    artifacts: Annotated[list[Artifact], add]
    confidence: float
    retry_count: int
    approval_state: ApprovalState
    workspace_descriptor: WorkspaceDescriptor
    execution_logs: Annotated[list[ExecutionLogEntry], add]
    checkpoints: Annotated[list[WorkflowCheckpoint], add]
    cancellation_requested: NotRequired[bool]
    cancellation_requested_at: NotRequired[datetime]
    cancellation_reason: NotRequired[str | None]


def create_initial_agent_state(
    *, workflow_id: str, workspace_descriptor: WorkspaceDescriptor
) -> AgentState:
    """Create a valid initial state for a newly started LangGraph workflow."""
    if not workflow_id.strip():
        msg = "workflow_id must not be empty"
        raise ValueError(msg)
    runtime = load_runtime_identity()
    return {
        "workflow_id": workflow_id,
        "workflow_schema_version": runtime.workflow_schema_version,
        "created_by_build_revision": runtime.build_revision,
        "last_executor_build_revision": runtime.build_revision,
        "current_step": WorkflowStatus.PENDING,
        "current_agent": None,
        "conversation_history": [],
        "artifacts": [],
        "confidence": 0.0,
        "retry_count": 0,
        "approval_state": ApprovalState.NOT_REQUIRED,
        "workspace_descriptor": workspace_descriptor,
        "execution_logs": [],
        "checkpoints": [],
    }


def _require_unique(identifiers: Iterable[str], entity_name: str) -> None:
    """Reject duplicate identifiers in structured state collections."""
    values = list(identifiers)
    if len(values) != len(set(values)):
        msg = f"{entity_name} identifiers must be unique"
        raise ValueError(msg)
