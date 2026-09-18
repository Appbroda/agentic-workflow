"""Strict request and response schemas for the workflow HTTP control plane."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from api.prd_markers import MARKER_PATTERN, MAX_CAPTION_LENGTH, markers_in_all
from artifacts.design_references import DesignReference, duplicate_design_pairs
from artifacts.prd_evidence import without_empty_prd_evidence
from artifacts.schemas import Requirement, UserStory
from state.enums import ApprovalState, LogLevel, WorkflowStatus
from state.models import WorkspaceDescriptor

type NonEmptyString = str


class APIModel(BaseModel):
    """Reject unexpected request fields, including credentials supplied in JSON bodies."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class PRDAttachmentRef(APIModel):
    """One already-uploaded image, and the name the submission's prose calls it by.

    The bytes are not here and never are: they were uploaded to `POST /attachments` and this
    names the result. What a submission adds is the *marker* -- how a sentence points at an
    image -- and a caption, which is the one thing the uploader knows that the file does not
    say.
    """

    attachment_id: NonEmptyString = Field(min_length=1)
    # A slug, matched against the one regular expression the validator, the prompt renderer
    # and the interface all read. Three near-identical patterns is how a submission gets
    # accepted with a reference the prompt then renders as literal text.
    marker: NonEmptyString = Field(pattern=MARKER_PATTERN)
    caption: str = Field(default="", max_length=MAX_CAPTION_LENGTH)


class PRDSubmission(APIModel):
    """The user-provided PRD content used to create the initial workflow artifact.

    Only a title and a problem statement are required. The structured sections below are
    optional because the product-manager agent's job is to derive them: demanding a goal, a
    user story and a requirement -- each with its own acceptance criteria -- before the
    platform would accept a feature meant asking a person to do the analysis they came here
    to have done. Anything they do supply is carried through unchanged and takes precedence
    over anything derived from it.
    """

    title: NonEmptyString = Field(min_length=1)
    problem_statement: NonEmptyString = Field(min_length=1)
    goals: list[NonEmptyString] = Field(default_factory=list)
    user_stories: list[UserStory] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    constraints: list[NonEmptyString] = Field(default_factory=list)
    out_of_scope: list[NonEmptyString] = Field(default_factory=list)
    stakeholders: list[NonEmptyString] = Field(default_factory=list)
    # Images already uploaded, and the names this submission's prose calls them by. Empty for
    # every submission that predates the field and every one that describes rather than shows.
    attachments: list[PRDAttachmentRef] = Field(default_factory=list)
    # The designs this request cites, validated and normalized at this boundary and nowhere
    # else. Optional like every other structured section, and for a stronger reason: a feature
    # that cites no design must behave in every respect as it does today, which is why the
    # client omits the key entirely rather than sending an empty array.
    design_references: list[DesignReference] = Field(default_factory=list)

    def prose(self) -> list[str]:
        """Return every piece of this submission's text that may point at an image.

        The whole document, because a marker is allowed anywhere a person writes: the problem
        statement, a goal, a story's need or acceptance criterion, a requirement's
        description, a constraint. Restricting it to the problem statement would mean a
        reference written in a requirement is silently literal text.
        """
        texts = [self.title, self.problem_statement, *self.goals, *self.constraints]
        texts.extend(self.out_of_scope)
        texts.extend(self.stakeholders)
        for story in self.user_stories:
            texts.extend((story.persona, story.need, story.benefit, *story.acceptance_criteria))
        for requirement in self.requirements:
            texts.extend((requirement.description, *requirement.acceptance_criteria))
        return texts

    @model_validator(mode="after")
    def identifiers_must_be_unique(self) -> Self:
        """Reject ambiguous traceability before artifact construction starts the workflow."""
        story_ids = [item.story_id for item in self.user_stories]
        if len(story_ids) != len(set(story_ids)):
            raise ValueError("user story identifiers must be unique")
        requirement_ids = [item.requirement_id for item in self.requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("requirement identifiers must be unique")
        markers = [item.marker for item in self.attachments]
        if len(markers) != len(set(markers)):
            raise ValueError("attachment markers must be unique within a submission")
        attachment_ids = [item.attachment_id for item in self.attachments]
        if len(attachment_ids) != len(set(attachment_ids)):
            raise ValueError("an attachment may be referenced only once in a submission")
        duplicates = duplicate_design_pairs(self.design_references)
        if duplicates:
            msg = f"design citations must be unique; these are cited twice: {', '.join(duplicates)}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def every_referenced_marker_must_be_declared(self) -> Self:
        """Refuse a marker the prose points at and no attachment declares.

        A dangling pointer is the failure this codebase already refuses elsewhere, and the
        reason is the same: accepting it means the prompt renders `[image:login-error]` as
        four literal words, the model answers from the description, and nobody finds out. The
        message names the marker, because "one of your references is wrong" is not an
        actionable sentence in a page of prose.

        The converse is deliberately accepted. An attachment nothing references still reaches
        the model, in declaration order after the referenced ones: somebody who attached
        three screenshots without writing `[image:...]` three times has still shown the
        platform three screenshots.
        """
        declared = {item.marker for item in self.attachments}
        unknown = [item for item in markers_in_all(self.prose()) if item not in declared]
        if unknown:
            named = ", ".join(f"[image:{item}]" for item in dict.fromkeys(unknown))
            msg = f"this submission references images that are not attached: {named}"
            raise ValueError(msg)
        return self

    @model_serializer(mode="wrap")
    def omit_evidence_nobody_supplied(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """Serialize as this submission did before designs and attachments existed, when empty.

        `_feature_fingerprint` hashes this request's JSON, so an added key changes the derived
        idempotency key of every submission that cited nothing and attached nothing. Nothing
        breaks -- it is a fresh key, not a wrong one -- but "a feature that cites no design and
        shows no picture behaves exactly as it does today" should be true of the bytes, and
        this is one of the two places it is made true. The other is `PRDArtifact`, where the
        prompt reads it.
        """
        return without_empty_prd_evidence(dict(handler(self)))


class StartWorkflowRequest(APIModel):
    """Start input containing only workflow data; provider credentials are header-only."""

    workflow_id: NonEmptyString | None = Field(default=None, min_length=1)
    workspace_descriptor: WorkspaceDescriptor
    prd: PRDSubmission


class ClarificationAnswer(APIModel):
    """One required human answer keyed to a technical-PRD clarification question."""

    question_id: NonEmptyString = Field(min_length=1)
    answer: NonEmptyString = Field(min_length=1)


class ClarificationRequest(APIModel):
    """Resume with clarification answers, or an empty list to recover an interrupted run."""

    workflow_id: NonEmptyString = Field(min_length=1)
    answers: list[ClarificationAnswer] = Field(default_factory=list)

    @model_validator(mode="after")
    def question_identifiers_must_be_unique(self) -> Self:
        """Make every resume answer unambiguous before it reaches workflow orchestration."""
        question_ids = [answer.question_id for answer in self.answers]
        if len(question_ids) != len(set(question_ids)):
            msg = "clarification answer question_ids must be unique"
            raise ValueError(msg)
        return self


class CancelWorkflowRequest(APIModel):
    """Identify a workflow that should stop before its next agent transition."""

    workflow_id: NonEmptyString = Field(min_length=1)


class WorkflowResponse(APIModel):
    """The public lifecycle snapshot returned by workflow-control endpoints."""

    workflow_id: NonEmptyString
    status: WorkflowStatus
    current_agent: NonEmptyString | None
    confidence: float = Field(ge=0, le=1)
    retry_count: int = Field(ge=0)
    approval_state: ApprovalState
    workspace_id: NonEmptyString
    created_at: datetime
    updated_at: datetime


class StartWorkflowResponse(WorkflowResponse):
    """Return whether a start request created a workflow or replayed an idempotent result."""

    created: bool


class ArtifactResponse(APIModel):
    """A JSON-safe versioned artifact envelope returned by the control plane."""

    artifact_id: NonEmptyString
    artifact_type: NonEmptyString
    workflow_id: NonEmptyString
    schema_version: NonEmptyString
    producer: NonEmptyString
    timestamp: datetime
    metadata: dict[str, Any]
    validation_status: Literal["valid", "invalid", "pending"]
    payload: dict[str, Any]


class ArtifactsResponse(APIModel):
    """List the artifacts currently available for a workflow."""

    workflow_id: NonEmptyString
    artifacts: list[ArtifactResponse]


class LogResponse(APIModel):
    """A structured execution log suitable for API clients and dashboards."""

    log_id: NonEmptyString
    timestamp: datetime
    level: LogLevel
    source: NonEmptyString
    event: NonEmptyString
    message: NonEmptyString
    context: dict[str, Any]


class LogsResponse(APIModel):
    """List structured execution logs for one workflow."""

    workflow_id: NonEmptyString
    logs: list[LogResponse]


class TimelineEventResponse(APIModel):
    """One lifecycle, artifact, or execution-log event in chronological workflow order."""

    timestamp: datetime
    event_type: Literal["lifecycle", "artifact", "log"]
    source: NonEmptyString
    event: NonEmptyString
    details: dict[str, Any]


class TimelineResponse(APIModel):
    """The complete chronological workflow timeline."""

    workflow_id: NonEmptyString
    events: list[TimelineEventResponse]
