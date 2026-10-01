"""Strict, versioned schemas for artifacts exchanged by workflow agents."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import PurePath
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from artifacts.design_references import DesignReference, duplicate_design_pairs
from artifacts.prd_evidence import without_empty_prd_evidence

type NonEmptyString = Annotated[str, Field(min_length=1)]
type NonNegativeInteger = Annotated[int, Field(ge=0)]
type PositiveInteger = Annotated[int, Field(gt=0)]
type Percentage = Annotated[float, Field(ge=0, le=100)]

_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")

type ArtifactKind = Literal[
    "prd",
    "technical_prd",
    "architecture",
    "execution_graph",
    "task_plan",
    "code_completion",
    "review",
    "pull_request",
    "integration_contract",
    "repository_execution_plan",
    "child_workflow_result",
    "integration_review",
    "contract_change_request",
    "feature_completion",
    "repository_reconnaissance",
    "repository_repair_proposal",
]


class StrictSchema(BaseModel):
    """Base class that rejects coercion and unknown fields in artifact data."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class BaseArtifact(StrictSchema):
    """The immutable envelope shared by every workflow artifact."""

    model_config = ConfigDict(frozen=True)

    schema_version: NonEmptyString
    workflow_id: NonEmptyString
    artifact_id: NonEmptyString
    producer: NonEmptyString
    timestamp: datetime
    metadata: dict[str, Any]
    validation_status: Literal["valid", "invalid", "pending"]

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_include_timezone(cls, value: datetime) -> datetime:
        """Require a timezone so artifacts are unambiguous across deployments."""
        if value.tzinfo is None or value.utcoffset() is None:
            msg = "timestamp must be timezone-aware"
            raise ValueError(msg)
        return value


class UserStory(StrictSchema):
    """A user-centred requirement from the submitted product requirement document."""

    story_id: NonEmptyString
    persona: NonEmptyString
    need: NonEmptyString
    benefit: NonEmptyString
    acceptance_criteria: list[NonEmptyString] = Field(min_length=1)


class Requirement(StrictSchema):
    """A traceable requirement shared across PRD and technical planning artifacts."""

    requirement_id: NonEmptyString
    description: NonEmptyString
    priority: Literal["must", "should", "could", "wont"]
    acceptance_criteria: list[NonEmptyString] = Field(min_length=1)
    dependencies: list[NonEmptyString]


class ClarificationQuestion(StrictSchema):
    """A question that must be resolved before implementation can proceed.

    The suggestion fields exist because asking somebody to decide "does this endpoint use the
    existing admin middleware" is asking them to go and read the repository the platform has
    just finished reading. Where the platform already knows what the checkout does, it says
    so and prefills the answer; where it does not, all three fields stay empty and the
    question is asked plainly. They are never filled in by a client, and answering is still an
    explicit act -- a suggestion is a starting point, not a decision made on somebody's behalf.
    """

    question_id: NonEmptyString
    question: NonEmptyString
    rationale: NonEmptyString
    required: bool
    # Empty when the platform has no grounded answer to offer. Deliberately not "the model's
    # best guess": an ungrounded suggestion is worse than no suggestion, because it looks the
    # same as a grounded one and gets accepted.
    suggested_answer: str = ""
    # One short line saying where the suggestion came from -- "Suggested from repository
    # analysis of admanager-server". Never reasoning, never a model transcript.
    suggestion_source: str = ""
    suggestion_confidence: Literal["high", "medium", "low"] | None = None

    @model_validator(mode="after")
    def suggestion_must_say_where_it_came_from(self) -> Self:
        """Refuse a suggestion with no provenance.

        A prefilled answer whose origin is unstated is indistinguishable from one the author
        typed, which is how a guess becomes a requirement nobody agreed to.
        """
        if self.suggested_answer.strip() and not self.suggestion_source.strip():
            msg = "a suggested answer must record where it came from"
            raise ValueError(msg)
        return self


class PRDAttachment(StrictSchema):
    """The record of one image a submission attached -- what it was, not what it contained.

    No bytes, deliberately and permanently. This artifact's payload is pasted into the
    product-manager prompt, sent as that call's `input_text`, and returned whole by the
    artifacts API; a base64 blob here would be sent to the model as text, would be counted
    against nothing, would break the context budget the snapshot derivation computes, and
    would be echoed to every reader of the artifact list.

    `sha256` is what makes this a record rather than a pointer. The artifact says which bytes
    were submitted, and it keeps saying it after the content has been purged.
    """

    attachment_id: NonEmptyString
    marker: NonEmptyString
    caption: str = ""
    filename: NonEmptyString
    media_type: Literal["image/png", "image/jpeg", "image/webp"]
    byte_size: PositiveInteger
    sha256: NonEmptyString


class PRDArtifact(BaseArtifact):
    """The normalized representation of a submitted product requirement document.

    The structured sections are optional. This artifact is the record of what somebody
    actually submitted, and a submission of a title and a problem statement is a real
    submission -- turning it into goals, stories and traceable requirements is the
    product-manager agent's work, recorded separately in the technical PRD. Requiring them
    here would have meant either refusing the submission or writing them on the author's
    behalf and storing the result as though they had said it.
    """

    artifact_type: Literal["prd"] = "prd"
    title: NonEmptyString
    problem_statement: NonEmptyString
    goals: list[NonEmptyString] = Field(default_factory=list)
    user_stories: list[UserStory] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    constraints: list[NonEmptyString] = Field(default_factory=list)
    out_of_scope: list[NonEmptyString] = Field(default_factory=list)
    stakeholders: list[NonEmptyString] = Field(default_factory=list)
    # What was attached, in the order the images are supplied to the model: the ones the
    # prose references first, then the rest in declaration order. Empty for every submission
    # that describes a screen rather than showing one.
    attachments: list[PRDAttachment] = Field(default_factory=list)
    # The designs the author attached, as citations rather than as resolved content: this is
    # the record of what somebody submitted, and the resolution is a separate artifact.
    # `default_factory=list` for the reason every field added to a persisted artifact gets
    # one -- every PRD written before this item must still load.
    design_references: list[DesignReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def identifiers_must_be_unique(self) -> Self:
        """Keep requirements, stories, image markers and design citations addressable."""
        _require_unique_ids((story.story_id for story in self.user_stories), "user story")
        _require_unique_ids(
            (requirement.requirement_id for requirement in self.requirements), "requirement"
        )
        # A marker is how prose addresses an image. Two attachments answering to one marker
        # means `[image:login-error]` names both and points at neither.
        _require_unique_ids(
            (attachment.marker for attachment in self.attachments), "attachment marker"
        )
        _require_unique_design_references(self.design_references)
        return self

    @model_serializer(mode="wrap")
    def omit_evidence_nobody_supplied(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """Serialize as this artifact did before designs and attachments existed, when empty.

        `{{ prd }}` in the product-manager prompt is this whole artifact, so a field added
        here reaches every feature's first model call. Both 89- items forbid that for a feature
        that supplied nothing, and the honest way to keep the promise is to make the bytes the
        same rather than to argue that an empty array is harmless.
        """
        return without_empty_prd_evidence(dict(handler(self)))


class TechnicalPRDArtifact(BaseArtifact):
    """An implementation-ready technical interpretation of the product requirements."""

    artifact_type: Literal["technical_prd"] = "technical_prd"
    title: NonEmptyString
    solution_summary: NonEmptyString
    functional_requirements: list[Requirement] = Field(min_length=1)
    non_functional_requirements: list[Requirement]
    data_requirements: list[NonEmptyString]
    integration_requirements: list[NonEmptyString]
    security_requirements: list[NonEmptyString]
    assumptions: list[NonEmptyString]
    unresolved_questions: list[ClarificationQuestion]

    @model_validator(mode="after")
    def requirement_identifiers_must_be_unique(self) -> Self:
        """Avoid ambiguous traceability between functional and quality requirements."""
        requirement_ids = (
            requirement.requirement_id
            for requirement in [*self.functional_requirements, *self.non_functional_requirements]
        )
        _require_unique_ids(requirement_ids, "technical requirement")
        _require_unique_ids(
            (question.question_id for question in self.unresolved_questions),
            "clarification question",
        )
        return self


class ArchitectureComponent(StrictSchema):
    """A deployable or logical component in the target architecture."""

    component_id: NonEmptyString
    name: NonEmptyString
    responsibility: NonEmptyString
    technology: NonEmptyString
    dependencies: list[NonEmptyString]


class APIContract(StrictSchema):
    """A public or internal HTTP contract defined by the architecture."""

    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: NonEmptyString
    summary: NonEmptyString
    request_schema: NonEmptyString | None
    response_schema: NonEmptyString
    authentication_required: bool

    @field_validator("path")
    @classmethod
    def path_must_be_absolute(cls, value: str) -> str:
        """Require HTTP paths to be expressed relative to the service root."""
        if not value.startswith("/"):
            msg = "API contract paths must start with '/'"
            raise ValueError(msg)
        return value


class DataField(StrictSchema):
    """A field in a persisted or exchanged data entity."""

    name: NonEmptyString
    data_type: NonEmptyString
    nullable: bool
    description: NonEmptyString


class DataEntity(StrictSchema):
    """A data entity, including the fields required to represent it."""

    entity_name: NonEmptyString
    description: NonEmptyString
    fields: list[DataField] = Field(min_length=1)

    @model_validator(mode="after")
    def field_names_must_be_unique(self) -> Self:
        """Prevent ambiguous field definitions."""
        _require_unique_ids((field.name for field in self.fields), "data field")
        return self


class ArchitectureDecision(StrictSchema):
    """A recorded architecture decision and its consequences."""

    decision_id: NonEmptyString
    title: NonEmptyString
    decision: NonEmptyString
    rationale: NonEmptyString
    consequences: list[NonEmptyString]


class ArchitectureRisk(StrictSchema):
    """A technical risk and the strategy for addressing it."""

    risk_id: NonEmptyString
    description: NonEmptyString
    likelihood: Literal["low", "medium", "high"]
    impact: Literal["low", "medium", "high", "critical"]
    mitigation: NonEmptyString


class ArchitectureArtifact(BaseArtifact):
    """The target architecture, integration contracts, and implementation risks."""

    artifact_type: Literal["architecture"] = "architecture"
    system_overview: NonEmptyString
    technology_stack: dict[NonEmptyString, NonEmptyString] = Field(min_length=1)
    repository_structure: list[NonEmptyString] = Field(min_length=1)
    components: list[ArchitectureComponent] = Field(min_length=1)
    api_contracts: list[APIContract]
    data_entities: list[DataEntity]
    decisions: list[ArchitectureDecision] = Field(min_length=1)
    risks: list[ArchitectureRisk]
    deployment_strategy: NonEmptyString

    @model_validator(mode="after")
    def component_references_must_be_valid(self) -> Self:
        """Ensure component dependencies point to known components."""
        component_ids = {component.component_id for component in self.components}
        _require_unique_ids((component.component_id for component in self.components), "component")
        _require_unique_ids(
            (decision.decision_id for decision in self.decisions), "architecture decision"
        )
        _require_unique_ids((risk.risk_id for risk in self.risks), "architecture risk")

        for component in self.components:
            unknown_dependencies = set(component.dependencies) - component_ids
            if unknown_dependencies:
                msg = (
                    f"component '{component.component_id}' references unknown dependencies: "
                    f"{sorted(unknown_dependencies)}"
                )
                raise ValueError(msg)
            if component.component_id in component.dependencies:
                msg = f"component '{component.component_id}' cannot depend on itself"
                raise ValueError(msg)
        return self


class ExecutionNode(StrictSchema):
    """A single agent action in a workflow execution graph."""

    node_id: NonEmptyString
    agent: NonEmptyString
    action: NonEmptyString
    input_artifact_types: list[ArtifactKind]
    output_artifact_type: ArtifactKind
    requires_human_approval: bool


class ExecutionEdge(StrictSchema):
    """A conditional transition between two workflow actions."""

    source_node_id: NonEmptyString
    target_node_id: NonEmptyString
    condition: NonEmptyString


class ExecutionGraphArtifact(BaseArtifact):
    """The explicit graph that controls workflow execution and handoffs."""

    artifact_type: Literal["execution_graph"] = "execution_graph"
    entry_node_id: NonEmptyString
    terminal_node_ids: list[NonEmptyString] = Field(min_length=1)
    nodes: list[ExecutionNode] = Field(min_length=1)
    edges: list[ExecutionEdge]

    @model_validator(mode="after")
    def graph_references_must_be_valid(self) -> Self:
        """Reject dangling node references and duplicate graph edges."""
        node_ids = {node.node_id for node in self.nodes}
        _require_unique_ids((node.node_id for node in self.nodes), "execution node")
        if self.entry_node_id not in node_ids:
            msg = f"entry node '{self.entry_node_id}' does not exist"
            raise ValueError(msg)
        unknown_terminal_nodes = set(self.terminal_node_ids) - node_ids
        if unknown_terminal_nodes:
            msg = f"unknown terminal nodes: {sorted(unknown_terminal_nodes)}"
            raise ValueError(msg)
        _require_unique_ids(self.terminal_node_ids, "terminal node")

        edge_ids: list[tuple[str, str, str]] = []
        for edge in self.edges:
            if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
                msg = "execution edges must reference nodes declared in the graph"
                raise ValueError(msg)
            if edge.source_node_id == edge.target_node_id:
                msg = "execution edges cannot self-reference"
                raise ValueError(msg)
            edge_ids.append((edge.source_node_id, edge.target_node_id, edge.condition))
        if len(edge_ids) != len(set(edge_ids)):
            msg = "execution graph contains duplicate edges"
            raise ValueError(msg)
        return self


class PlannedTask(StrictSchema):
    """A traceable unit of implementation work."""

    task_id: NonEmptyString
    title: NonEmptyString
    description: NonEmptyString
    owner: NonEmptyString
    priority: Literal["critical", "high", "medium", "low"]
    dependencies: list[NonEmptyString]
    acceptance_criteria: list[NonEmptyString] = Field(min_length=1)
    estimated_effort_points: PositiveInteger


class Milestone(StrictSchema):
    """A delivery checkpoint containing a known set of planned tasks."""

    milestone_id: NonEmptyString
    title: NonEmptyString
    objective: NonEmptyString
    task_ids: list[NonEmptyString] = Field(min_length=1)


class TaskPlanArtifact(BaseArtifact):
    """The ordered, dependency-aware implementation plan."""

    artifact_type: Literal["task_plan"] = "task_plan"
    summary: NonEmptyString
    tasks: list[PlannedTask] = Field(min_length=1)
    milestones: list[Milestone] = Field(min_length=1)
    implementation_order: list[NonEmptyString] = Field(min_length=1)
    test_strategy: list[NonEmptyString] = Field(min_length=1)
    risk_management_plan: list[NonEmptyString]

    @model_validator(mode="after")
    def task_references_must_be_valid(self) -> Self:
        """Ensure task dependencies and delivery order form a complete valid plan."""
        task_ids = {task.task_id for task in self.tasks}
        _require_unique_ids((task.task_id for task in self.tasks), "task")
        _require_unique_ids((milestone.milestone_id for milestone in self.milestones), "milestone")
        _require_unique_ids(self.implementation_order, "implementation-order task")
        if set(self.implementation_order) != task_ids:
            msg = "implementation_order must contain every task exactly once"
            raise ValueError(msg)

        for task in self.tasks:
            unknown_dependencies = set(task.dependencies) - task_ids
            if unknown_dependencies:
                msg = (
                    f"task '{task.task_id}' references unknown dependencies: "
                    f"{sorted(unknown_dependencies)}"
                )
                raise ValueError(msg)
            if task.task_id in task.dependencies:
                msg = f"task '{task.task_id}' cannot depend on itself"
                raise ValueError(msg)

        milestone_task_ids: list[str] = []
        for milestone in self.milestones:
            unknown_task_ids = set(milestone.task_ids) - task_ids
            if unknown_task_ids:
                msg = (
                    f"milestone '{milestone.milestone_id}' references unknown tasks: "
                    f"{sorted(unknown_task_ids)}"
                )
                raise ValueError(msg)
            _require_unique_ids(milestone.task_ids, "milestone task")
            milestone_task_ids.extend(milestone.task_ids)
        if set(milestone_task_ids) != task_ids or len(milestone_task_ids) != len(task_ids):
            msg = "each task must belong to exactly one milestone"
            raise ValueError(msg)

        dependency_map = {task.task_id: task.dependencies for task in self.tasks}
        if _contains_dependency_cycle(dependency_map):
            msg = "task dependencies must not contain a cycle"
            raise ValueError(msg)
        return self


class FileChange(StrictSchema):
    """A file-level change made by the software engineering agent."""

    path: NonEmptyString
    change_type: Literal["added", "modified", "deleted", "renamed"]
    description: NonEmptyString

    @field_validator("path")
    @classmethod
    def path_must_stay_within_workspace(cls, value: str) -> str:
        """Reject absolute and traversal paths in implementation reports."""
        path = PurePath(value)
        if path.is_absolute() or ".." in path.parts:
            msg = "file change paths must be relative and must not traverse parent directories"
            raise ValueError(msg)
        return value


class ValidationResult(StrictSchema):
    """The result of one command or automated validation step."""

    name: NonEmptyString
    command: NonEmptyString
    passed: bool
    output_summary: NonEmptyString
    duration_seconds: Annotated[float, Field(ge=0)]


class RequirementImplementationEvidence(StrictSchema):
    """Concrete file and symbol evidence for one implemented requirement."""

    requirement_id: NonEmptyString
    files: list[NonEmptyString] = Field(min_length=1)
    symbols: list[NonEmptyString] = Field(default_factory=list)
    description: NonEmptyString
    validation_results: list[NonEmptyString] = Field(default_factory=list)


class CodeCompletionArtifact(BaseArtifact):
    """The engineer's traceable report of applied code changes and validation."""

    artifact_type: Literal["code_completion"] = "code_completion"
    completion_status: Literal["completed", "partially_completed", "failed"]
    summary: NonEmptyString
    file_changes: list[FileChange]
    validation_results: list[ValidationResult]
    test_coverage_percent: Percentage | None
    remaining_work: list[NonEmptyString]
    commit_sha: NonEmptyString | None
    production_files_changed: list[NonEmptyString] = Field(default_factory=list)
    test_files_changed: list[NonEmptyString] = Field(default_factory=list)
    configuration_files_changed: list[NonEmptyString] = Field(default_factory=list)
    requirements_implemented: list[NonEmptyString] = Field(default_factory=list)
    requirements_not_implemented: list[NonEmptyString] = Field(default_factory=list)
    implementation_expectations_satisfied: list[NonEmptyString] = Field(default_factory=list)
    requirement_implementation_evidence: list[RequirementImplementationEvidence] = Field(
        default_factory=list
    )
    # Calculated by the trusted workspace boundary from path + file bytes. The model cannot
    # decide whether a retry changed production behavior merely by repeating a file name.
    production_diff_fingerprint: NonEmptyString | None = None

    @model_validator(mode="after")
    def completed_work_must_have_no_open_failures(self) -> Self:
        """Make a completed report internally consistent."""
        if self.completion_status == "completed":
            if self.remaining_work:
                msg = "completed artifacts cannot report remaining work"
                raise ValueError(msg)
            if any(not result.passed for result in self.validation_results):
                msg = "completed artifacts cannot report failed validations"
                raise ValueError(msg)
        evidence_ids = {item.requirement_id for item in self.requirement_implementation_evidence}
        if not set(self.requirements_implemented).issubset(evidence_ids):
            msg = "implemented requirements require requirement-to-file evidence"
            raise ValueError(msg)
        if set(self.requirements_implemented) & set(self.requirements_not_implemented):
            msg = "a requirement cannot be both implemented and not implemented"
            raise ValueError(msg)
        return self


class ReviewFinding(StrictSchema):
    """A concrete issue reported by the quality-review agent."""

    finding_id: NonEmptyString
    severity: Literal["critical", "high", "medium", "low", "info"]
    title: NonEmptyString
    description: NonEmptyString
    recommendation: NonEmptyString
    file_path: NonEmptyString | None
    line_number: NonNegativeInteger | None
    # Repository review findings need an actionable ownership reference.  Optional
    # fields preserve older persisted artifacts while the validator below rejects
    # unscoped new implementation findings.
    repository_id: NonEmptyString | None = None
    requirement_id: NonEmptyString | None = None
    contract_reference: NonEmptyString | None = None
    responsibility: Literal["implements", "consumes", "validates", "documents"] | None = None
    validated_revision: NonEmptyString | None = None
    evidence: NonEmptyString | None = None
    recommended_fix: NonEmptyString | None = None
    finding_category: Literal[
        "requirement",
        "contract",
        "security",
        "code_quality",
        "repository_health",
        "validation_failure",
        # Distinct from `validation_failure` on purpose: a command that never completed
        # returned no verdict on the source, so the retry policy must not read it as
        # rejected code and spend a coding budget answering it.
        "validation_capacity",
    ] = "code_quality"

    @field_validator("file_path")
    @classmethod
    def finding_path_must_stay_within_workspace(cls, value: str | None) -> str | None:
        """Apply the same workspace-bound path policy to review findings."""
        if value is not None:
            FileChange(path=value, change_type="modified", description="review location")
        return value

    def names_its_scope(self) -> bool:
        """Say whether this finding names what it derives from.

        The four sources a bounded review may block on: a scoped requirement, a contract
        section, a failed required validation command, or an implementation expectation.
        The first two are named by the reference fields; the third is what the two
        validation categories mean, and the fourth is keyed by the requirement whose
        expectation it belongs to, so it is named by `requirement_id` as well.

        A finding that names none of them is not wrong -- the reviewer is a competent senior
        engineer and its observations are kept -- it is simply not something the requirement
        asked for, so it is published for a person rather than spent on a coding attempt.
        """
        return bool(
            self.requirement_id
            or self.contract_reference
            or self.finding_category in {"validation_failure", "validation_capacity"}
        )

    @model_validator(mode="after")
    def finding_must_be_scoped_or_a_repository_quality_exception(self) -> Self:
        """Keep implementation findings traceable without weakening security review."""
        exceptions = {
            "security",
            "code_quality",
            "repository_health",
            "validation_failure",
            "validation_capacity",
        }
        if self.finding_category not in exceptions:
            if not (self.requirement_id or self.contract_reference):
                msg = "review findings must reference a scoped requirement or contract"
                raise ValueError(msg)
            if not all(
                (self.repository_id, self.responsibility, self.validated_revision, self.evidence)
            ):
                msg = (
                    "scoped review findings require repository, responsibility, revision, "
                    "and evidence"
                )
                raise ValueError(msg)
        return self


class RequirementCheck(StrictSchema):
    """The review result for a requirement or acceptance criterion."""

    requirement_id: NonEmptyString
    passed: bool
    evidence: NonEmptyString


class ReviewArtifact(BaseArtifact):
    """A requirement, quality, security, and test review of an implementation."""

    artifact_type: Literal["review"] = "review"
    verdict: Literal["approved", "changes_requested", "rejected"]
    summary: NonEmptyString
    requirement_checks: list[RequirementCheck]
    findings: list[ReviewFinding]
    architecture_assessment: NonEmptyString
    security_assessment: NonEmptyString
    test_coverage_assessment: NonEmptyString

    @model_validator(mode="after")
    def approved_reviews_cannot_contain_blocking_findings(self) -> Self:
        """Enforce that high-severity defects block approval."""
        _require_unique_ids(
            (check.requirement_id for check in self.requirement_checks), "requirement check"
        )
        _require_unique_ids((finding.finding_id for finding in self.findings), "review finding")
        if self.verdict == "approved":
            blocking_findings = [
                finding.finding_id
                for finding in self.findings
                if finding.severity in {"critical", "high"}
            ]
            if blocking_findings:
                msg = f"approved reviews cannot contain blocking findings: {blocking_findings}"
                raise ValueError(msg)
        return self


class PullRequestArtifact(BaseArtifact):
    """The result of safely creating a GitHub pull request for a workflow."""

    artifact_type: Literal["pull_request"] = "pull_request"
    repository: NonEmptyString
    pull_request_number: PositiveInteger
    url: HttpUrl
    title: NonEmptyString
    body: NonEmptyString
    source_branch: NonEmptyString
    target_branch: NonEmptyString
    commit_sha: NonEmptyString
    labels: list[NonEmptyString]
    reviewers: list[NonEmptyString]
    state: Literal["open", "merged", "closed"]

    @model_validator(mode="after")
    def source_and_target_branches_must_differ(self) -> Self:
        """Prevent an unsafe pull request targeting its own source branch."""
        if self.source_branch == self.target_branch:
            msg = "source_branch and target_branch must differ"
            raise ValueError(msg)
        return self


class EndpointContract(StrictSchema):
    """A stable, machine-readable endpoint owned and consumed by named workstreams."""

    operation_id: NonEmptyString
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    path: NonEmptyString
    summary: NonEmptyString
    request_schema: dict[str, Any] | None
    response_schema: dict[str, Any]
    query_parameters: list[dict[str, Any]]
    path_parameters: list[dict[str, Any]]
    headers: list[dict[str, Any]]
    authentication_required: bool
    authorization_requirements: list[NonEmptyString]
    success_status_codes: list[Annotated[int, Field(ge=100, le=599)]] = Field(min_length=1)
    error_codes: list[NonEmptyString]
    version: NonEmptyString

    @field_validator("path")
    @classmethod
    def path_must_be_absolute(cls, value: str) -> str:
        """Ensure generated OpenAPI operations use a canonical path."""
        if not value.startswith("/"):
            msg = "integration contract paths must start with '/'"
            raise ValueError(msg)
        return value


class SharedSchemaDefinition(StrictSchema):
    """A named JSON Schema payload that can produce backend and frontend types."""

    name: NonEmptyString
    description: NonEmptyString
    json_schema: dict[str, Any]


class AuthenticationContract(StrictSchema):
    """The transport and token behavior shared by API providers and consumers."""

    scheme: NonEmptyString
    header_name: NonEmptyString | None
    description: NonEmptyString


class AuthorizationRule(StrictSchema):
    """A least-privilege rule referenced by endpoint authorization requirements."""

    rule_id: NonEmptyString
    description: NonEmptyString
    applies_to_operation_ids: list[NonEmptyString]


class ErrorContract(StrictSchema):
    """A stable error identifier that consumers can handle without guessing."""

    code: NonEmptyString
    status_code: Annotated[int, Field(ge=100, le=599)]
    description: NonEmptyString
    schema_definition: dict[str, Any] | None


class EventContract(StrictSchema):
    """An event schema for event-driven or mixed integration styles."""

    event_name: NonEmptyString
    version: NonEmptyString
    payload_schema: dict[str, Any]
    producer_workstream_id: NonEmptyString
    consumer_workstream_ids: list[NonEmptyString]


class EnvironmentVariableContract(StrictSchema):
    """A non-secret environment compatibility requirement across repositories."""

    name: NonEmptyString
    required: bool
    description: NonEmptyString
    owner_workstream_id: NonEmptyString


class CompatibilityPolicy(StrictSchema):
    """Compatibility, migration, and rollback requirements for contract consumers."""

    policy: NonEmptyString
    breaking_change_allowed: bool
    migration_requirements: list[NonEmptyString]
    rollback_requirements: list[NonEmptyString]


class IntegrationContractArtifact(BaseArtifact):
    """The immutable, approved source of truth shared by repository workstreams."""

    artifact_type: Literal["integration_contract"] = "integration_contract"
    feature_id: NonEmptyString
    contract_version: NonEmptyString
    status: Literal["draft", "approved", "superseded"]
    api_style: Literal["rest", "graphql", "grpc", "event_driven", "none", "mixed"]
    endpoints: list[EndpointContract]
    shared_schemas: list[SharedSchemaDefinition]
    authentication_contract: AuthenticationContract | None
    authorization_rules: list[AuthorizationRule]
    error_contracts: list[ErrorContract]
    event_contracts: list[EventContract]
    environment_variables: list[EnvironmentVariableContract]
    compatibility_policy: CompatibilityPolicy
    owning_workstreams: list[NonEmptyString] = Field(min_length=1)
    approved_at: datetime | None
    openapi_document: dict[str, Any] | None = None

    @field_validator("contract_version")
    @classmethod
    def contract_version_must_be_semantic(cls, value: str) -> str:
        """Use ordered immutable contract versions rather than arbitrary revision labels."""
        if not _SEMVER.fullmatch(value):
            msg = "contract_version must use MAJOR.MINOR.PATCH semantic versioning"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def contract_identifiers_must_be_stable(self) -> Self:
        """Reject duplicate operation, schema, rule, and error identifiers."""
        _require_unique_ids((endpoint.operation_id for endpoint in self.endpoints), "operation")
        _require_unique_ids((schema.name for schema in self.shared_schemas), "shared schema")
        _require_unique_ids(
            (rule.rule_id for rule in self.authorization_rules), "authorization rule"
        )
        _require_unique_ids((error.code for error in self.error_contracts), "error contract")
        _require_unique_ids(self.owning_workstreams, "owning workstream")
        if self.status == "approved" and self.approved_at is None:
            msg = "approved integration contracts require approved_at"
            raise ValueError(msg)
        return self


class ScopedRequirementReference(StrictSchema):
    """One repository's explicit relationship to a technical requirement."""

    requirement_id: NonEmptyString
    acceptance_criterion_ids: list[NonEmptyString] = Field(min_length=1)
    responsibility: Literal["implements", "consumes", "validates", "documents"]


class ImplementationExpectation(StrictSchema):
    """The production and test changes a scoped requirement is expected to produce."""

    requirement_id: NonEmptyString
    expected_change_categories: list[
        Literal[
            "production",
            "route",
            "controller",
            "service",
            "model",
            "migration",
            "frontend_component",
            "client",
            "test",
            "documentation",
            "configuration",
            # New code has to be reachable to be finished. Satisfied by modifying a file
            # that already existed, which is what registering a route or rendering a
            # component requires. Promised in code below rather than left to the planner,
            # because whether it is remembered decided whether a feature shipped.
            "integration",
        ]
    ] = Field(min_length=1)
    expected_source_areas: list[NonEmptyString] = Field(default_factory=list)
    tests_required: bool


class WorkstreamTaskDependency(StrictSchema):
    """One of a repository's tasks that cannot be started until others of its tasks are done.

    A different question from `dependency_workstream_ids` one level up. That one is about a
    published build artifact passing between two repositories; this one is about which of a
    single repository's own tasks has to exist before another can be written against it.
    Both `task_id` and every `depends_on` entry name a task declared in the same workstream's
    `task_ids`.
    """

    task_id: NonEmptyString
    # Accepted empty rather than refused. A model that lists every task and gives most of them
    # nothing to wait for is saying the ordinary and correct thing -- these are independent --
    # and rejecting that shape would spend a repair round trip on a true statement.
    depends_on: list[NonEmptyString] = Field(default_factory=list)


class RepositoryWorkstreamPlan(StrictSchema):
    """The contract-scoped implementation assignment for one repository."""

    workstream_id: NonEmptyString
    repository_id: NonEmptyString
    # A role describes this branch in the graph; repository/workstream IDs are its identity.
    # Open-ended labels keep persisted plans forward compatible with new repository kinds.
    role: Annotated[str, Field(min_length=1, max_length=32)]
    # Older persisted feature plans did not record requirement ownership. Keep them
    # readable; newly generated live plans must still be non-empty via planner validation.
    requirement_ids: list[NonEmptyString] = Field(default_factory=list)
    scoped_requirements: list[ScopedRequirementReference] = Field(default_factory=list)
    out_of_scope_requirements: list[NonEmptyString] = Field(default_factory=list)
    shared_requirements: list[ScopedRequirementReference] = Field(default_factory=list)
    responsibilities: list[NonEmptyString] = Field(min_length=1)
    task_ids: list[NonEmptyString] = Field(min_length=1)
    # Ordering among this workstream's own tasks. Defaulted empty because every plan persisted
    # before the field existed genuinely declared none, and because independent tasks -- the
    # ordinary case -- declare none either. Empty means what it has always meant: nothing here
    # waits for anything else here.
    task_dependencies: list[WorkstreamTaskDependency] = Field(default_factory=list)
    dependency_workstream_ids: list[NonEmptyString]
    contract_sections_consumed: list[NonEmptyString]
    contract_sections_implemented: list[NonEmptyString]
    # The frames of the resolved design snapshot this workstream is asked to build, which is
    # what `contract_sections_implemented` already is for the contract. `default_factory=list`
    # because older persisted plans did not record it, and because most workstreams have no
    # design at all -- an empty list means the design applies somewhere else, or nowhere.
    #
    # Assigned by the planner and never inferred. This platform does not know which repository
    # renders UI: `role` is a descriptive label ("Roles are descriptive labels, never
    # repository identity"), and using it to guess would encode exactly the assumption the
    # repo-agnostic rule forbids.
    design_nodes: list[NonEmptyString] = Field(default_factory=list)
    acceptance_criteria: list[NonEmptyString] = Field(min_length=1)
    test_requirements: list[NonEmptyString] = Field(min_length=1)
    documentation_requirements: list[NonEmptyString]
    expected_files_or_areas: list[NonEmptyString]
    required: bool
    implementation_expectations: list[ImplementationExpectation] = Field(default_factory=list)

    @model_validator(mode="after")
    def scoped_requirements_must_not_conflict(self) -> Self:
        """Prevent a child from being given mutually contradictory ownership."""
        scoped_ids = [item.requirement_id for item in self.scoped_requirements]
        shared_ids = [item.requirement_id for item in self.shared_requirements]
        _require_unique_ids(
            (f"{item.requirement_id}:{item.responsibility}" for item in self.scoped_requirements),
            "scoped requirement responsibility",
        )
        if set(scoped_ids) & set(self.out_of_scope_requirements):
            msg = "a requirement cannot be both scoped and out of scope"
            raise ValueError(msg)
        if set(shared_ids) & set(self.out_of_scope_requirements):
            msg = "a shared requirement cannot be out of scope"
            raise ValueError(msg)
        if (
            self.scoped_requirements
            and self.requirement_ids
            and set(scoped_ids) != set(self.requirement_ids)
        ):
            msg = "requirement_ids must mirror scoped_requirements for new workstreams"
            raise ValueError(msg)
        if not self.implementation_expectations:
            object.__setattr__(
                self,
                "implementation_expectations",
                _default_implementation_expectations(self),
            )
        # Applied to whatever the planner produced, not only to the default above, which is
        # used solely when it produced nothing. A live plan always carries its own
        # expectations, so promising integration in the default alone changed nothing at
        # all: p-1's workstreams arrived as production/test/documentation and its backend
        # still shipped a controller no running file referred to.
        object.__setattr__(
            self,
            "implementation_expectations",
            [_with_integration_promise(item) for item in self.implementation_expectations],
        )
        expectation_ids = {item.requirement_id for item in self.implementation_expectations}
        scoped_requirement_ids = set(scoped_ids)
        if not expectation_ids.issubset(scoped_requirement_ids):
            msg = "implementation expectations must reference scoped requirements"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def task_dependencies_must_reference_declared_tasks(self) -> Self:
        """Keep task ordering inside the set of tasks this workstream actually declares.

        Structural and nothing more. Unknown, self-referential and circular edges are removed
        by the planner before an artifact is built, so this is the invariant that says a
        persisted plan can never carry one -- not a gate a live plan is expected to fail.

        Deliberately no cycle check here. A cycle is refused where the graph becomes an
        instruction, by `TaskPlanArtifact`, which has refused one since it was written. Making
        the same refusal twice would mean the backstop could never be exercised.
        """
        declared = set(self.task_ids)
        _require_unique_ids((item.task_id for item in self.task_dependencies), "task dependency")
        for item in self.task_dependencies:
            if item.task_id not in declared:
                msg = (
                    f"task dependency '{item.task_id}' is not one of this workstream's "
                    "declared task_ids"
                )
                raise ValueError(msg)
            unknown = set(item.depends_on) - declared
            if unknown:
                msg = (
                    f"task '{item.task_id}' depends on tasks this workstream does not "
                    f"declare: {sorted(unknown)}"
                )
                raise ValueError(msg)
            if item.task_id in item.depends_on:
                msg = f"task '{item.task_id}' cannot depend on itself"
                raise ValueError(msg)
        return self


class RepositoryExecutionPlanArtifact(BaseArtifact):
    """The feature-level dependency graph and repository workstream plan."""

    artifact_type: Literal["repository_execution_plan"] = "repository_execution_plan"
    feature_id: NonEmptyString
    contract_artifact_id: NonEmptyString
    workstreams: list[RepositoryWorkstreamPlan] = Field(min_length=1)
    execution_order: list[NonEmptyString] = Field(min_length=1)
    parallel_groups: list[list[NonEmptyString]] = Field(min_length=1)
    # Operator sequencing intent is merge guidance, never an execution barrier: every
    # workstream still fans out in parallel once the contract is approved. Older
    # persisted plans predate the field and legitimately carry no recommendation.
    recommended_merge_order: list[NonEmptyString] = Field(default_factory=list)
    integration_test_plan: list[NonEmptyString] = Field(min_length=1)
    merge_strategy: Literal[
        "backend_first", "frontend_first", "simultaneous", "independent", "manual"
    ]
    deployment_strategy: Literal[
        "backend_first", "frontend_first", "simultaneous", "independent", "manual"
    ]
    feature_flag_strategy: list[NonEmptyString]
    rollback_strategy: list[NonEmptyString] = Field(min_length=1)

    @model_validator(mode="after")
    def workstream_references_must_be_valid(self) -> Self:
        """Ensure fan-out groups form a valid dependency-respecting execution schedule."""
        workstream_ids = [workstream.workstream_id for workstream in self.workstreams]
        _require_unique_ids(workstream_ids, "workstream")
        _require_unique_ids(
            (workstream.repository_id for workstream in self.workstreams), "repository"
        )
        if set(self.execution_order) != set(workstream_ids):
            msg = "execution_order must include every workstream exactly once"
            raise ValueError(msg)
        if any(not group for group in self.parallel_groups):
            msg = "parallel_groups cannot contain an empty group"
            raise ValueError(msg)
        parallel_ids = [item for group in self.parallel_groups for item in group]
        if set(parallel_ids) != set(workstream_ids) or len(parallel_ids) != len(workstream_ids):
            msg = "parallel_groups must include every workstream exactly once"
            raise ValueError(msg)
        if self.recommended_merge_order and (
            set(self.recommended_merge_order) != set(workstream_ids)
            or len(self.recommended_merge_order) != len(workstream_ids)
        ):
            msg = "recommended_merge_order must include every workstream exactly once"
            raise ValueError(msg)
        known_workstreams = set(workstream_ids)
        execution_index = {
            workstream_id: index for index, workstream_id in enumerate(self.execution_order)
        }
        group_index = {
            workstream_id: index
            for index, group in enumerate(self.parallel_groups)
            for workstream_id in group
        }
        for workstream in self.workstreams:
            unknown = set(workstream.dependency_workstream_ids) - known_workstreams
            if unknown or workstream.workstream_id in workstream.dependency_workstream_ids:
                msg = "workstream dependencies must reference other declared workstreams"
                raise ValueError(msg)
            for dependency_id in workstream.dependency_workstream_ids:
                if execution_index[dependency_id] >= execution_index[workstream.workstream_id]:
                    msg = "execution_order must place dependencies before their workstream"
                    raise ValueError(msg)
                if group_index[dependency_id] >= group_index[workstream.workstream_id]:
                    msg = "parallel_groups must place dependencies in an earlier group"
                    raise ValueError(msg)
        assignments: dict[str, set[str]] = {}
        for workstream in self.workstreams:
            for reference in [*workstream.scoped_requirements, *workstream.shared_requirements]:
                assignments.setdefault(reference.requirement_id, set()).add(
                    reference.responsibility
                )
        # A shared requirement may involve several repositories, including a
        # provider and consumer that each implement their side of one contract.
        # Every reference still records an explicit responsibility for review scope.
        for requirement_id, responsibilities in assignments.items():
            if not responsibilities:
                msg = f"requirement '{requirement_id}' lacks an explicit responsibility"
                raise ValueError(msg)
        return self


class ReviewFindingCounts(StrictSchema):
    """How one review's findings were divided between blocking work and advising a person.

    Recorded so the effect of bounding review scope is measured rather than assumed.
    ``findings_untraceable`` is the number that would have blocked under the unbounded rule
    and is advisory now -- the size of the behaviour change, per review.
    """

    findings_total: NonNegativeInteger
    findings_blocking: NonNegativeInteger
    findings_advisory: NonNegativeInteger
    findings_untraceable: NonNegativeInteger
    # How many blocking findings a person's removal_holds verdict demoted (73-). Defaulted,
    # because every count persisted before the field existed genuinely had none.
    findings_overruled: NonNegativeInteger = 0


class OverruledReviewFinding(StrictSchema):
    """One review finding demoted from blocking because a person overruled its demand.

    The audit trace the 73- demotion is required to leave: which finding, matched on which
    fingerprint, under which verdict, decided by whom and when. Recorded on the review
    artifact's metadata and carried here on the result so the pull-request body can show a
    reader that the finding was judged -- by a named person, not by the platform -- which is
    the opposite of an advisory finding "nobody has judged yet".
    """

    finding_id: NonEmptyString
    description: NonEmptyString
    # The ledger identity the demand was settled under: `fingerprint_for_text` of the
    # finding's description, the same key the stop matched on and the verdict is stored by.
    fingerprint: NonEmptyString
    # Plain strings, not NonEmptyString: the live verdict path always fills them, but a
    # record with a blank name must degrade to a blank field in an audit row, never to a
    # validation error that kills the attempt carrying it.
    conflict_id: str = ""
    verdict: Literal["removal_holds"] = "removal_holds"
    # The decision in the decider's own words, carried so a pull-request reader sees why the
    # finding does not block without chasing the conflict artifact it points at.
    decision: str = ""
    decided_by: str = ""
    decided_at: datetime | None = None


class ChildWorkflowResultArtifact(BaseArtifact):
    """The immutable result of one repository's isolated engineer and reviewer loop."""

    artifact_type: Literal["child_workflow_result"] = "child_workflow_result"
    feature_id: NonEmptyString
    parent_workflow_id: NonEmptyString
    child_workflow_id: NonEmptyString
    repository_id: NonEmptyString
    workstream_id: NonEmptyString
    branch_name: NonEmptyString
    workspace_path: NonEmptyString
    code_completion_artifact_id: NonEmptyString | None
    review_artifact_id: NonEmptyString | None
    changed_files: list[FileChange]
    validation_results: list[ValidationResult]
    status: Literal["approved", "failed", "waiting_for_contract_change", "cancelled"]
    blocking_issues: list[NonEmptyString]
    pull_request_readiness: bool
    contract_sections_consumed: list[NonEmptyString]
    contract_sections_implemented: list[NonEmptyString]
    technology_profile: dict[str, Any] | None = None
    validation_plan: dict[str, Any] | None = None
    current_revision: NonEmptyString | None = None
    current_validation_results: list[dict[str, Any]] = Field(default_factory=list)
    superseded_validation_count: NonNegativeInteger = 0
    scoped_requirements: list[ScopedRequirementReference] = Field(default_factory=list)
    out_of_scope_requirements: list[NonEmptyString] = Field(default_factory=list)
    preflight_result: dict[str, Any] | None = None
    layout_evidence: dict[str, Any] | None = None
    failure_classification: str | None = None
    retry_strategy: dict[str, Any] | None = None
    # Which configured model role produced this result, and why that role was selected. Held
    # on the result so the record of an attempt names the model that made it without having to
    # be joined against the child state, which has since moved on to the next attempt.
    model_routing: dict[str, Any] | None = None
    production_files_changed: list[NonEmptyString] = Field(default_factory=list)
    test_files_changed: list[NonEmptyString] = Field(default_factory=list)
    configuration_files_changed: list[NonEmptyString] = Field(default_factory=list)
    requirements_implemented: list[NonEmptyString] = Field(default_factory=list)
    requirements_not_implemented: list[NonEmptyString] = Field(default_factory=list)
    meaningful_change: bool | None = None
    meaningful_change_reason: NonEmptyString | None = None
    production_diff_fingerprint: NonEmptyString | None = None
    previous_attempt_fingerprint: NonEmptyString | None = None
    test_diff_fingerprint: NonEmptyString | None = None
    previous_test_fingerprint: NonEmptyString | None = None
    # Findings the review raised that named nothing this workstream was asked for. Kept here
    # so they reach the pull-request body: the observation is not discarded, it moves from
    # blocking a machine that will rewrite the file five times to informing a person.
    advisory_findings: list[NonEmptyString] = Field(default_factory=list)
    # Findings the review raised as blocking that a person's removal_holds verdict had
    # already overruled. Separate from `advisory_findings` on purpose: an advisory finding
    # awaits a person's judgement, an overruled one carries it -- name, verdict and
    # timestamp -- and presenting the second as the first would erase the decision.
    overruled_findings: list[OverruledReviewFinding] = Field(default_factory=list)
    review_finding_counts: ReviewFindingCounts | None = None
    # The verdict that was accepted without being an approval, where one was. Present only
    # when the review declined but every finding it raised was advisory, so a reader of this
    # result -- and of the pull request it produces -- can see that the reviewer did not say
    # yes and why the work was published anyway.
    advisory_review_verdict: Literal["changes_requested", "rejected"] | None = None

    @model_validator(mode="after")
    def approval_requires_complete_verified_work(self) -> Self:
        """Prevent an incomplete child result from reaching integration or PR publication."""
        if self.status == "approved" and self.requirements_not_implemented:
            msg = "approved child results cannot contain unimplemented requirements"
            raise ValueError(msg)
        if self.pull_request_readiness and self.status != "approved":
            msg = "pull_request_readiness requires an approved child result"
            raise ValueError(msg)
        if self.status == "approved" and any(
            not result.passed for result in self.validation_results
        ):
            msg = "approved child results cannot contain failed validations"
            raise ValueError(msg)
        return self


# What stops a coordinated publication. `medium` is in the set because this gate is the only
# reader of two repositories' changes at once, so the defects it alone can see -- a frontend
# gating an admin control on a permission its own backend does not accept, say -- are exactly
# the ones nobody else will catch, and they are rarely written up as `critical`.
#
# AB-Feature-215 shipped on that gap: this gate found the authorization mismatch, wrote the fix
# into `required_fixes`, approved anyway because the finding was `medium`, and opened both pull
# requests. The remediation machinery it should have used already exists and is fenced against
# looping -- `_integration_remediation_decision` refuses a repeat, and the cycle budget bounds
# the rest -- so the finding had a route home and was simply never given it.
#
# `low` and `info` stay out: `low` is reserved for wording a plan should correct rather than a
# change anyone should make, and promoting it would send repositories back to edit nothing.
INTEGRATION_BLOCKING_SEVERITIES = frozenset({"critical", "high", "medium"})


class IntegrationReviewFinding(StrictSchema):
    """A cross-repository compatibility finding routed to one responsible repository."""

    finding_id: NonEmptyString
    severity: Literal["critical", "high", "medium", "low", "info"]
    responsible_repository_id: NonEmptyString
    affected_repository_ids: list[NonEmptyString] = Field(min_length=1)
    contract_reference: NonEmptyString
    description: NonEmptyString
    evidence: NonEmptyString
    recommended_fix: NonEmptyString


class RepositoryResultSummary(StrictSchema):
    """A compact child status suitable for parent integration review artifacts."""

    repository_id: NonEmptyString
    child_workflow_id: NonEmptyString
    status: NonEmptyString
    child_result_artifact_id: NonEmptyString


class RepositoryConvention(StrictSchema):
    """One established way a repository already does a kind of work, with the files proving it."""

    convention_id: NonEmptyString
    kind: NonEmptyString
    description: NonEmptyString
    evidence_paths: list[NonEmptyString] = Field(min_length=1)
    # Where a new member of this kind has to be registered to be reachable at all. Null when
    # the repository has no such registry and members are picked up by convention instead.
    wiring_path: NonEmptyString | None


class ContradictedPremise(StrictSchema):
    """A thing the requirements assume exists that this checkout does not contain.

    The most expensive failure this platform has had is a requirement written against an
    assumed convention, enforced faithfully by a reviewer, that no attempt could satisfy
    because the convention was never there. Naming it before planning is the whole point of
    reconnaissance; the `question` is what a human should be asked instead.
    """

    premise: NonEmptyString
    contradicted_by: NonEmptyString
    evidence_paths: list[NonEmptyString] = Field(min_length=1)
    question: NonEmptyString


class RepositoryReconnaissanceArtifact(BaseArtifact):
    """What one checkout actually contains, established before the plan is allowed to assume."""

    artifact_type: Literal["repository_reconnaissance"] = "repository_reconnaissance"
    feature_id: NonEmptyString
    repository_id: NonEmptyString
    repository_revision: NonEmptyString
    summary: NonEmptyString
    source_areas: list[NonEmptyString]
    test_areas: list[NonEmptyString]
    conventions: list[RepositoryConvention]
    shared_utilities: list[NonEmptyString]
    contradicted_premises: list[ContradictedPremise]
    # Every command the platform will run against this checkout, derived from the checkout the
    # reconnaissance just read. Carried here because this is the only artifact produced while a
    # real clone exists and before any plan is written -- the validation plan proper is built
    # inside the child workflow, hours of wall clock later, and by then the requirement naming a
    # command nobody can run has already been issued and is binding on the review.
    #
    # Optional, and empty means only "not established": every artifact persisted before this
    # field existed reads back as empty, and the planner's clause is written so that empty asks
    # nothing rather than forbidding everything.
    runnable_validation_commands: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def conventions_are_uniquely_identified(self) -> Self:
        """Keep a convention referenceable by a plan without ambiguity."""
        _require_unique_ids((item.convention_id for item in self.conventions), "convention")
        return self


class IntegrationReviewArtifact(BaseArtifact):
    """The contract-backed decision to proceed to coordinated pull requests or fix a child."""

    artifact_type: Literal["integration_review"] = "integration_review"
    feature_id: NonEmptyString
    contract_artifact_id: NonEmptyString
    review_status: Literal["approved", "changes_requested", "failed", "failed_requires_human"]
    repository_results: list[RepositoryResultSummary]
    contract_checks: list[NonEmptyString]
    cross_repository_findings: list[IntegrationReviewFinding]
    compatibility_assessment: NonEmptyString
    security_assessment: NonEmptyString
    deployment_assessment: NonEmptyString
    merge_order: list[NonEmptyString]
    required_fixes: list[NonEmptyString]

    @model_validator(mode="after")
    def approved_review_cannot_have_blocking_findings(self) -> Self:
        """Make the PR gate impossible to approve with unresolved high-risk findings.

        Deliberately looser than `INTEGRATION_BLOCKING_SEVERITIES`, and it must stay that way.
        Stored artifacts are re-validated on every read -- `_artifact_from_model` calls
        `model_validate_json` on the persisted payload -- so a rule tightened here is applied
        retroactively to history that was written under the old one. Raising this bar to match
        the gate would make 10 of the 57 approved integration reviews already in the database
        unreadable, including AB-Feature-215's, and take the feature detail and logbook
        endpoints down with them for those features.

        So the bar lives in two places on purpose: this validator is the *read* contract and
        admits what the platform used to write, while `IntegrationReviewAgent.review` is the
        *write* contract and is where `medium` blocks. `required_fixes` non-empty on an approved
        review is likewise a historical shape that must stay readable -- the agent can no longer
        produce one, which `test_integration_reviewer` pins, but 10 rows already have it.
        """
        _require_unique_ids((item.finding_id for item in self.cross_repository_findings), "finding")
        if self.review_status == "approved" and any(
            item.severity in {"critical", "high"} for item in self.cross_repository_findings
        ):
            msg = "approved integration reviews cannot contain blocking findings"
            raise ValueError(msg)
        return self


class ContractChangeRequestArtifact(BaseArtifact):
    """A child-owned request to revise—not mutate—the approved integration contract."""

    artifact_type: Literal["contract_change_request"] = "contract_change_request"
    change_request_id: NonEmptyString
    feature_id: NonEmptyString
    current_contract_version: NonEmptyString
    requested_by_repository_id: NonEmptyString
    requested_changes: list[NonEmptyString] = Field(min_length=1)
    reason: NonEmptyString
    affected_workstreams: list[NonEmptyString] = Field(min_length=1)
    compatibility_impact: NonEmptyString
    migration_requirements: list[NonEmptyString]
    status: Literal["pending", "approved", "rejected", "superseded"]
    resolution: NonEmptyString | None
    new_contract_artifact_id: NonEmptyString | None


class RepositoryRepairCommand(StrictSchema):
    """One command a repair may run, with the reason it is needed."""

    command: list[NonEmptyString] = Field(min_length=1)
    working_directory: str | None = None
    purpose: NonEmptyString


class RepositoryRepairProposalArtifact(BaseArtifact):
    """A repository whose own checked-in setup stops it running its checks, and the fix.

    This is not a coding failure and must never be produced for one. It describes a
    repository that cannot validate anything on an untouched checkout -- a lint config whose
    shared plugin is undeclared, a package manager whose lockfile does not resolve, a script
    that is not there. No implementation written into such a repository could have been
    checked, which is why the platform stops rather than reporting work it could not verify,
    and why applying the fix is a decision a person makes about their repository rather than
    something an agent does on their behalf.
    """

    artifact_type: Literal["repository_repair_proposal"] = "repository_repair_proposal"
    repair_id: NonEmptyString
    feature_id: NonEmptyString
    repository_id: NonEmptyString
    child_workflow_id: str | None = None
    originating_stage: NonEmptyString
    failure_classification: NonEmptyString
    detected_problem: NonEmptyString
    evidence: list[NonEmptyString] = Field(default_factory=list)
    proposed_repair: NonEmptyString
    affected_files: list[NonEmptyString] = Field(default_factory=list)
    affected_dependencies: list[NonEmptyString] = Field(default_factory=list)
    commands: list[RepositoryRepairCommand] = Field(default_factory=list)
    expected_impact: NonEmptyString
    risk: Literal["low", "medium", "high"]
    # Whether applying this would change what the repository *does*, as opposed to what it
    # can check. A repair that changes behaviour is a code change wearing a repair's clothes,
    # and the console requires an explicit confirmation before one is approved.
    changes_source_logic: bool
    # The revision the diagnosis was written against. An approval is refused when the
    # repository has moved on since, because the problem described may no longer be there.
    proposed_at_revision: str | None = None
    status: Literal[
        "proposed", "approved", "executing", "succeeded", "failed", "rejected", "superseded"
    ]
    approved_by: str | None = None
    approved_at: datetime | None = None
    rejected_by: str | None = None
    rejection_reason: str | None = None
    execution_result: str | None = None
    resulting_revision: str | None = None

    @field_validator("approved_at")
    @classmethod
    def approval_timestamp_must_include_timezone(cls, value: datetime | None) -> datetime | None:
        """Keep an approval's time unambiguous in an audit read from another zone."""
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            msg = "approved_at must be timezone-aware"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def a_decision_must_carry_its_reason(self) -> Self:
        """Refuse a rejection with nothing recorded about why."""
        if self.status == "rejected" and not (self.rejection_reason or "").strip():
            msg = "a rejected repair must record why it was rejected"
            raise ValueError(msg)
        return self


class DesignConflictArtifact(BaseArtifact):
    """A design decision this workstream keeps reversing, put to a person as a question.

    Not a defect and not a capability limit. It is the situation where one review demanded a
    change, a later cycle of the same review stopped demanding it -- or the other review took
    the opposite position -- and it is being demanded again. Another attempt would re-argue the
    question rather than answer it, which is why the workstream stops instead of retrying.

    Written by the platform at the stop, and revised in place by appending a newer artifact
    with the same ``conflict_id`` once somebody answers it. Both positions are recorded as they
    were actually worded, because what a person arbitrates is two sentences about one defect.
    """

    artifact_type: Literal["design_conflict"] = "design_conflict"
    conflict_id: NonEmptyString
    feature_id: NonEmptyString
    repository_id: NonEmptyString
    child_workflow_id: NonEmptyString
    # Which shape of re-litigation this is. A `reversal` is the 49-B stop: demanded,
    # satisfied, demanded again -- it carries both positions. A `recurring_demand` (77-,
    # item 35) is a demand that was never satisfied but kept coming back reworded, dodging
    # every exact-text guard; it carries the wordings in its evidence and has no satisfied
    # position at all. Defaulted so every persisted conflict validates unchanged.
    kind: Literal["reversal", "recurring_demand"] = "reversal"
    # The identity both authorities are compared on, and the key the verdict is looked up by.
    # `fingerprint_for_text` throughout, exactly as the ledger that produced this uses. For a
    # recurring demand it is the theme's earliest wording -- an exact key of a real sentence,
    # stable across later rewordings; never a looser key (73-).
    fingerprint: NonEmptyString
    # The platform's own sentence to a person, and the lineage facts behind it. Both come from
    # the narrative that raised the stop, so this artifact and the stopped attempt's blocking
    # issue cannot say two different things.
    question: NonEmptyString
    evidence: list[NonEmptyString] = Field(default_factory=list)
    # Position one: what is being demanded now, and by which review.
    demanded_by: Literal["repository_review", "integration_review"]
    demand: NonEmptyString
    # Position two, reversals only: the same defect as it was worded when it was satisfied,
    # which review had asked for it, and what the attempt that stopped reporting it said it
    # had done. `grounds` is empty where the lineage holds no completion summary for that
    # cycle. A recurring demand has no second position -- nothing was ever satisfied.
    satisfied_under: Literal["repository_review", "integration_review"] | None = None
    satisfied_demand: NonEmptyString | None = None
    grounds: str = ""
    # How many times the demand has gone from made to absent, and how many attempts this
    # workstream had spent when the stop was recognised. Facts about the lineage. Zero
    # removals is the recurring-demand shape: the demand never went away.
    removals: int = Field(ge=0, default=1)
    attempts_spent: int = Field(ge=0, default=0)
    # Whether the two positions come from different reviews. Recorded rather than derived from
    # the two authority fields, so a reader and the narrative cannot disagree.
    cross_authority: bool = False
    status: Literal["open", "answered"] = "open"
    # The decision, once made. `requirement_holds` means the demand stands and the next attempt
    # must implement it; `removal_holds` means the review is overruled and the next attempt must
    # not. `decision` is the person's own words -- runs 193 and 194 showed that an answer which
    # encodes the implementation strategy is what precedes a delivery, so it travels verbatim.
    verdict: Literal["requirement_holds", "removal_holds"] | None = None
    decision: str = ""
    decided_by: str | None = None
    decided_at: datetime | None = None

    @field_validator("decided_at")
    @classmethod
    def decision_timestamp_must_include_timezone(cls, value: datetime | None) -> datetime | None:
        """Keep a decision's time unambiguous in an audit read from another zone."""
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            msg = "decided_at must be timezone-aware"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def each_kind_carries_exactly_its_own_positions(self) -> Self:
        """Keep the two shapes honest: a reversal proves satisfaction, a recurrence has none.

        A reversal without its satisfied position is an unanswerable question -- the pair is
        what a person arbitrates -- and a recurring demand carrying one would claim the
        workstream built something it never did.
        """
        if self.kind == "reversal":
            if self.satisfied_under is None or self.satisfied_demand is None:
                msg = "a reversal conflict must record the position that was satisfied"
                raise ValueError(msg)
            if self.removals < 1:
                msg = "a reversal conflict records at least one removal"
                raise ValueError(msg)
        else:
            if self.satisfied_under is not None or self.satisfied_demand is not None:
                msg = "a recurring-demand conflict has no satisfied position to record"
                raise ValueError(msg)
            if self.removals != 0:
                msg = "a recurring-demand conflict records no removals"
                raise ValueError(msg)
            if self.cross_authority:
                msg = "a recurring-demand conflict is a single review's own repetition"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def an_answered_conflict_must_carry_its_decision(self) -> Self:
        """Refuse an answered conflict that records no verdict, no words and nobody.

        An invariant the next attempt cannot relitigate is only as good as the record behind
        it: a verdict with no author and no reasoning is an instruction nobody can be held to,
        and it is spliced into a coding prompt.
        """
        if self.status != "answered":
            return self
        if self.verdict is None:
            msg = "an answered design conflict must record which position holds"
            raise ValueError(msg)
        if not self.decision.strip():
            msg = "an answered design conflict must record the decision in the decider's words"
            raise ValueError(msg)
        if not (self.decided_by or "").strip():
            msg = "an answered design conflict must record who decided it"
            raise ValueError(msg)
        return self


class DesignSourceFile(StrictSchema):
    """One design file a snapshot read, and the version the provider said it read.

    `file_version` is provenance and a pin. The console's preview endpoint renders on demand
    and passes this back to Figma, because Figma's images endpoint renders the *current* file
    state unless told otherwise -- so an unpinned re-render would show a design that has
    changed under a snapshot that must not change.
    """

    file_key: NonEmptyString
    file_name: str = ""
    file_version: str = ""


class DesignNodeRecord(StrictSchema):
    """One cited frame, rendered as text a person writing a component would want.

    `content` is a mapping rather than a further schema for `contract_sections`' reason: the
    shape inside it is the design tool's, not this repository's, and a schema that restated it
    would refuse a field Figma added rather than carry it. What *is* typed is everything a
    reader of this platform needs to act on it: which file and frame, what the author called
    it, the link they pasted, and how much of the bound it spent.
    """

    file_key: NonEmptyString
    node_id: NonEmptyString
    # What the author called this frame, or the frame's own name for a whole-file citation.
    label: str = ""
    # The link the person pasted, so the console links back to what they were looking at.
    source_url: str = ""
    # Which repositories the citation scoped this to. Empty means the plan decides -- this
    # platform does not know which repository renders UI, and `role` is a descriptive label
    # that must never be used to guess.
    applies_to: list[NonEmptyString] = Field(default_factory=list)
    content: dict[str, Any]
    characters: NonNegativeInteger = 0
    nodes_rendered: NonNegativeInteger = 0
    # How deep the rendering reached. Since the detail tier became a hybrid this is the depth
    # of the *index tail*, which for a real frame means "the leaves" -- so it no longer varies
    # and no longer tells a reader anything about what the budget bought.
    depth_rendered: NonNegativeInteger = 0
    # How deep *build* fidelity reached: paint, geometry and typography below this depth were
    # not rendered, only names and text. This is the number the character budget actually
    # varies, so it is the one a reader needs to answer "how much of this frame could the
    # Engineer implement from". Absent on records written before the hybrid existed.
    build_depth_rendered: NonNegativeInteger | None = None
    # A rendering of this frame as a picture, for the roles that build and judge it. Not an
    # asset the application uses and never written into the checkout -- it is the thing a
    # reviewer compares a diff against, and the thing an engineer can notice it has not
    # matched. Empty when the deployment stores no attachments or the render was refused, and
    # every prompt is then exactly what it was before.
    preview_attachment_id: str = ""


class DesignNodeCitation(StrictSchema):
    """One cited frame that is not in the snapshot, and which of the answers that was.

    Absent is a fact, not a gap -- `contract_sections`' rule. A citation Figma does not define
    means the plan is wrong or the design moved, and a citation the token could not read means
    the deployment cannot see it: both are things every judge should be shown rather than
    left to infer from a frame that is simply not there.
    """

    file_key: NonEmptyString
    node_id: NonEmptyString
    label: str = ""
    source_url: str = ""
    reason: NonEmptyString


class DesignNodeOmission(StrictSchema):
    """One cited frame that resolved but did not fit, with the size and the bound that bit.

    Never trimmed, always whole: a judge shown two thirds of a frame reads a layout that is
    missing children and finds a violation that is not there.
    """

    file_key: NonEmptyString
    node_id: NonEmptyString
    label: str = ""
    reason: NonEmptyString
    characters: NonNegativeInteger = 0


class DesignSnapshotArtifact(BaseArtifact):
    """The designs a feature cited, resolved once, as text.

    Written before the product manager runs, because the design is part of the *request*: the
    requirements a PM derives from a problem statement should be derived from the frames too,
    and a design that arrives after the requirements are written can only ever contradict
    them.

    Immutable like every artifact, and Safety rule 4 makes that load-bearing rather than
    incidental: attempt 2 of a workstream builds against the same snapshot attempt 1 did, and
    a review judges against the snapshot the attempt was given. A refresh is a new revision,
    never an edit -- so a design edited mid-feature can never silently change the work order
    and leave an unexplainable diff behind.

    **Names outrank values.** Every rendered node puts its style, component and component-set
    names before its resolved fills, because a hex code tells an engineer what to hardcode and
    a name (`color/surface/raised`, `Button/Primary/Hover`) tells it what the repository
    already has and must reuse. That is the difference between a change that matches the design
    system and one that matches the picture.
    """

    artifact_type: Literal["design_snapshot"] = "design_snapshot"
    feature_id: NonEmptyString
    resolved_at: datetime
    files: list[DesignSourceFile] = Field(default_factory=list)
    # Every cited frame, rendered for *deciding* rather than for building: structure, names and
    # text, with no layout, typography, size or paint. That is what the product manager and the
    # planner read, and neither is building -- one is deriving requirements from the frames and
    # the other is deciding which repository each frame belongs to.
    #
    # Complete by construction, which is the whole point. A frame too large to *build* is still
    # a frame the flow has, and hiding it here would hide it from the planner -- the role that
    # decides where it goes -- which is the failure 96- exists to fix. A tier-2 size failure
    # must never remove a frame from this list; it is reported on the `DesignDetailArtifact`
    # instead, against the workstream that could not fit it.
    nodes: list[DesignNodeRecord] = Field(default_factory=list)
    # The three ways a citation can fail to be in `nodes`, each populated independently. No
    # cap here is silent: an unreported one reads as "the whole design was considered".
    design_nodes_omitted: list[DesignNodeOmission] = Field(default_factory=list)
    design_nodes_absent: list[DesignNodeCitation] = Field(default_factory=list)
    design_nodes_unreachable: list[DesignNodeCitation] = Field(default_factory=list)
    # Which of the two sources the style names came from. Measured live on 2026-09-07: the
    # variables API answers 403 for a read-only personal access token and parts of it are
    # Enterprise-only, so this platform reads the names already on the nodes -- and says so,
    # rather than leaving a reader to assume the richer source was used.
    style_name_source: Literal["node_styles", "file_variables"] = "node_styles"
    bounds: dict[str, int] = Field(default_factory=dict)
    characters_selected: NonNegativeInteger = 0

    @model_validator(mode="after")
    def a_snapshot_accounts_for_every_citation_exactly_once(self) -> Self:
        """Refuse a snapshot that lists one frame as both resolved and not."""
        _refuse_a_frame_quoted_and_missing_at_once(
            self.nodes,
            omitted=self.design_nodes_omitted,
            absent=self.design_nodes_absent,
            unreachable=self.design_nodes_unreachable,
            subject="snapshot",
        )
        return self


def _refuse_a_frame_quoted_and_missing_at_once(
    nodes: list[DesignNodeRecord],
    *,
    omitted: list[DesignNodeOmission],
    absent: list[DesignNodeCitation],
    unreachable: list[DesignNodeCitation],
    subject: str,
) -> None:
    """Refuse a resolution that lists one frame as both resolved and not.

    The lists are what a judge reads to know what it was *not* shown, so a frame appearing in
    two of them would make that reading ambiguous in the one direction that matters: an
    engineer would see a frame quoted and also see it named as missing.

    Shared by the snapshot and the per-repository detail rather than written twice, because the
    two must agree about what an honest resolution looks like. Detail is where this matters
    most: it is the tier an engineer actually builds from, so a frame quoted there *and* named
    as omitted is the one ambiguity that reaches code.
    """
    resolved = {(item.file_key, item.node_id) for item in nodes}
    for name, entries in (
        ("design_nodes_omitted", omitted),
        ("design_nodes_absent", absent),
        ("design_nodes_unreachable", unreachable),
    ):
        clashing = sorted(
            f"{item.file_key}#{item.node_id}"
            for item in entries
            if (item.file_key, item.node_id) in resolved
        )
        if clashing:
            msg = f"{name} names frames this {subject} also quotes: {', '.join(clashing)}"
            raise ValueError(msg)


class DesignAsset(StrictSchema):
    """One image the design fills a node with, exported and placed in the workspace.

    Without this an image reaches the Engineer as `{"type": "IMAGE"}` beside a width and a
    height, and an empty box is the only honest thing it can draw -- which is exactly what
    AB-Feature-231 drew for a 862x868 background and five ad thumbnails. Figma stores the bytes
    behind an `imageRef` no REST caller can resolve, so the platform renders the node itself
    and writes the result into the checkout.

    `workspace_path` is deliberately neutral (`.design/assets/...`) and never a framework's
    convention: this platform does not know whether a repository serves static files from
    `public/`, `static/`, `assets/` or an import pipeline, and guessing would be the kind of
    encoded assumption that makes it stop being repository-agnostic. The Engineer is told the
    file is there and told to put it where *this* repository keeps such things.

    `node_id` is the node the bytes were rendered from, not the frame that was cited, so a
    reader can go back to the exact rectangle in Figma.
    """

    node_id: NonEmptyString
    name: str = ""
    media_type: NonEmptyString = "image/png"
    # Where the platform wrote it, relative to the repository root.
    workspace_path: NonEmptyString
    # The durable copy, so a resume places the same bytes without re-rendering and without
    # spending another Figma call against a rate limit measured in requests per month.
    attachment_id: str = ""
    width: NonNegativeInteger = 0
    height: NonNegativeInteger = 0
    bytes_written: NonNegativeInteger = 0


class DesignAssetOmission(StrictSchema):
    """One image the design fills a node with that this platform could not place.

    Reported rather than dropped, on the same rule as every other bound here: an unreported
    absence reads as "the design had no image there", and the Engineer would then be blamed for
    a box it was never given anything to fill.
    """

    node_id: NonEmptyString
    name: str = ""
    reason: NonEmptyString
    width: NonNegativeInteger = 0
    height: NonNegativeInteger = 0


class DesignDetailArtifact(BaseArtifact):
    """One repository's own frames, resolved at build fidelity when its workstream starts.

    The second of the two tiers 96- split one eager resolution into. The snapshot holds the
    **index** -- every cited frame rendered as structure, names and text, for the roles that
    decide which frame belongs where. This holds the **detail**: only the frames one
    workstream's approved plan assigns, rendered with their layout, typography and paint, which
    is what the Engineer, its self-review and the Reviewer build and judge against.

    A second artifact rather than a field on the snapshot, for two reasons that are both about
    when things are known. The snapshot is immutable and is written *before the planner runs*,
    so it cannot contain a per-repository resolution that does not exist yet. And a detail
    artifact per repository keeps what a reader loads proportional to the workstream rather
    than to the feature: a thirty-screen citation indexes once and details two screens.

    One per repository, which on this platform is one per workstream and one per child
    workflow: the execution plan refuses two workstreams naming the same repository
    (`RepositoryExecutionPlanArtifact.workstream_references_must_be_valid`) and the child id is
    keyed `feature_id:repository_id`. So `repository_id` is the identity here and
    `workstream_id` is recorded beside it as the plan's own name for the same thing.

    `snapshot_artifact_id` is load-bearing rather than provenance. The snapshot has revisions --
    a re-resolution is a new revision, never an edit (Safety rule 4) -- so "attempt 2 built
    against the same design attempt 1 did" is only checkable if each detail says which index
    revision it belongs to. A detail that omits it makes the pairing unknowable after a refresh.
    """

    artifact_type: Literal["design_detail"] = "design_detail"
    feature_id: NonEmptyString
    repository_id: NonEmptyString
    workstream_id: NonEmptyString
    # Which index revision this detail belongs to. See the class docstring: without it, a
    # snapshot refresh makes "which design did this attempt build against" unanswerable.
    snapshot_artifact_id: NonEmptyString
    resolved_at: datetime
    files: list[DesignSourceFile] = Field(default_factory=list)
    # Build fidelity, always. An index record here would be a rendering with no colours, no
    # type and no spacing handed to the role that writes the CSS, and the values it invented to
    # fill the gaps would pass every gate this platform has -- no command it runs checks a
    # colour. `design_content_is_buildable` is what enforces that; this comment says why.
    nodes: list[DesignNodeRecord] = Field(default_factory=list)
    # The three ways an assigned frame can fail to be here, each populated independently and
    # each carried to the prompts. Omit-don't-trim applies in this tier exactly as in the
    # snapshot: a frame that does not fit is named whole, never quoted in part.
    design_nodes_omitted: list[DesignNodeOmission] = Field(default_factory=list)
    design_nodes_absent: list[DesignNodeCitation] = Field(default_factory=list)
    design_nodes_unreachable: list[DesignNodeCitation] = Field(default_factory=list)
    bounds: dict[str, int] = Field(default_factory=dict)
    characters_selected: NonNegativeInteger = 0
    # The images this design fills nodes with, exported and written into the checkout, and the
    # ones that could not be. Empty for a design with no image fills, which leaves every
    # prompt byte-identical to what it was before assets existed.
    assets: list[DesignAsset] = Field(default_factory=list)
    assets_omitted: list[DesignAssetOmission] = Field(default_factory=list)

    @model_validator(mode="after")
    def a_detail_accounts_for_every_assigned_frame_exactly_once(self) -> Self:
        """Refuse a detail that lists one frame as both resolved and not."""
        _refuse_a_frame_quoted_and_missing_at_once(
            self.nodes,
            omitted=self.design_nodes_omitted,
            absent=self.design_nodes_absent,
            unreachable=self.design_nodes_unreachable,
            subject="resolution",
        )
        return self


class FeatureCompletionArtifact(BaseArtifact):
    """The auditable result and human merge/deployment recommendation for a feature."""

    artifact_type: Literal["feature_completion"] = "feature_completion"
    feature_id: NonEmptyString
    parent_workflow_id: NonEmptyString
    status: Literal["completed", "partial_failure", "failed", "cancelled"]
    technical_prd_artifact_id: NonEmptyString
    integration_contract_artifact_id: NonEmptyString
    repository_execution_plan_artifact_id: NonEmptyString
    child_workflow_results: list[NonEmptyString]
    integration_review_artifact_id: NonEmptyString
    pull_request_artifact_ids: list[NonEmptyString]
    pull_request_urls: list[HttpUrl]
    merge_strategy: Literal[
        "backend_first", "frontend_first", "simultaneous", "independent", "manual"
    ]
    merge_order: list[NonEmptyString]
    deployment_strategy: Literal[
        "backend_first", "frontend_first", "simultaneous", "independent", "manual"
    ]
    feature_flags: list[NonEmptyString]
    rollback_plan: list[NonEmptyString]
    known_limitations: list[NonEmptyString]
    completed_at: datetime


class SupersededPullRequest(StrictSchema):
    """The published work one revision replaces, for one repository.

    Snapshotted when the revision is requested rather than derived later: the child rows are
    about to be reset for the new run, and after that reset nothing else records which branch
    and pull request the person was looking at when they asked for changes.
    """

    repository_id: NonEmptyString
    branch_name: NonEmptyString
    pull_request_artifact_id: str | None
    pull_request_url: str | None
    pull_request_number: PositiveInteger | None


class FeatureRevisionRequestArtifact(BaseArtifact):
    """A person's post-completion request to change already-published work.

    The durable record of why a completed feature re-opened: who asked, what they asked for,
    which revision this is, and exactly which branches and pull requests it supersedes. The
    request text itself becomes the revision's requirement set (a synthesized technical PRD),
    so this artifact is the provenance rather than the working input.
    """

    artifact_type: Literal["feature_revision_request"] = "feature_revision_request"
    feature_id: NonEmptyString
    revision: PositiveInteger
    request_text: NonEmptyString
    requested_by: NonEmptyString
    superseded: list[SupersededPullRequest]
    requested_at: datetime


type Artifact = (
    PRDArtifact
    | TechnicalPRDArtifact
    | ArchitectureArtifact
    | ExecutionGraphArtifact
    | TaskPlanArtifact
    | CodeCompletionArtifact
    | ReviewArtifact
    | PullRequestArtifact
    | IntegrationContractArtifact
    | RepositoryExecutionPlanArtifact
    | ChildWorkflowResultArtifact
    | IntegrationReviewArtifact
    | ContractChangeRequestArtifact
    | FeatureCompletionArtifact
    | RepositoryReconnaissanceArtifact
    | RepositoryRepairProposalArtifact
    | DesignConflictArtifact
    | DesignSnapshotArtifact
    | DesignDetailArtifact
    | FeatureRevisionRequestArtifact
)


_INTEGRATION_REQUIRING_CATEGORIES = frozenset(
    {
        "production",
        "route",
        "controller",
        "service",
        "model",
        "migration",
        "frontend_component",
        "client",
    }
)


def _with_integration_promise(expectation: ImplementationExpectation) -> ImplementationExpectation:
    """Require a requirement that adds production code to also connect it.

    Documentation-only and test-only expectations are untouched: there is nothing for them
    to wire in.
    """
    categories = list(expectation.expected_change_categories)
    if "integration" in categories:
        return expectation
    if not _INTEGRATION_REQUIRING_CATEGORIES.intersection(categories):
        return expectation
    return expectation.model_copy(
        update={"expected_change_categories": [*categories, "integration"]}
    )


def _default_implementation_expectations(
    workstream: RepositoryWorkstreamPlan,
) -> list[ImplementationExpectation]:
    """Preserve old plans while giving new workstreams an explicit completion contract."""
    expectations: list[ImplementationExpectation] = []
    for reference in workstream.scoped_requirements:
        if reference.responsibility == "implements":
            # Repository links are supplied before a checkout exists.  A role such as
            # "backend" does not prove an MVC layout (or even a particular language),
            # so require a real production change without inventing paths or framework
            # concepts.  Checkout-derived evidence may later narrow an explicit area.
            categories = (
                ["configuration", "test"]
                if workstream.role == "infrastructure"
                else ["production", "test", "integration"]
            )
            areas = workstream.expected_files_or_areas
        elif reference.responsibility == "validates":
            categories = ["test"]
            areas = workstream.expected_files_or_areas
        elif reference.responsibility == "documents":
            categories = ["documentation"]
            areas = workstream.expected_files_or_areas
        else:
            # A consumer may be Python, Node, or another language.  Require an actual
            # production integration change rather than assuming a frontend client path.
            categories = ["production", "test", "integration"]
            areas = workstream.expected_files_or_areas
        expectations.append(
            ImplementationExpectation(
                requirement_id=reference.requirement_id,
                expected_change_categories=categories,  # type: ignore[arg-type]
                expected_source_areas=areas,
                tests_required="test" in categories,
            )
        )
    return expectations


def _require_unique_design_references(references: list[DesignReference]) -> None:
    """Refuse a submission that cites the same frame twice.

    The same reason repeated story and requirement ids are refused: the citation is ambiguous,
    and the ambiguity would be resolved silently -- one of the two labels would win and the
    other would vanish from the snapshot the engineer is judged against.
    """
    duplicates = duplicate_design_pairs(references)
    if duplicates:
        msg = f"design citations must be unique; these are cited twice: {', '.join(duplicates)}"
        raise ValueError(msg)


def _require_unique_ids(identifiers: Iterable[str], entity_name: str) -> None:
    """Raise a validation error when a collection contains duplicate identifiers."""
    values = list(identifiers)
    if len(values) != len(set(values)):
        msg = f"{entity_name} identifiers must be unique"
        raise ValueError(msg)


def _contains_dependency_cycle(dependency_map: dict[str, list[str]]) -> bool:
    """Return whether a directed task-dependency graph contains a cycle."""
    visited: set[str] = set()
    in_progress: set[str] = set()

    def visit(task_id: str) -> bool:
        if task_id in in_progress:
            return True
        if task_id in visited:
            return False

        in_progress.add(task_id)
        for dependency_id in dependency_map[task_id]:
            if visit(dependency_id):
                return True
        in_progress.remove(task_id)
        visited.add(task_id)
        return False

    return any(visit(task_id) for task_id in dependency_map)
