"""Strict request and response schemas for additive multi-repository feature endpoints."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from api.schemas import APIModel, ClarificationAnswer, PRDSubmission
from artifacts.schemas import (
    AuthenticationContract,
    AuthorizationRule,
    CompatibilityPolicy,
    EndpointContract,
    EnvironmentVariableContract,
    ErrorContract,
    EventContract,
    SharedSchemaDefinition,
)
from services.execution_records import ExecutionRecord
from state.enums import (
    CancellationLifecycleStatus,
    ChildWorkflowStatus,
    DeploymentStrategy,
    FeatureActionStatus,
    FeatureWorkflowStatus,
    MergeStrategy,
    RepositoryRepairStatus,
    WorkstreamRole,
)
from state.external_operations import ManualCleanupRequirement, WorkflowCheckpointBoundary
from state.feature_models import FeatureFailureSummary, RepositorySpec

# Read-model phases composed from a durable workflow checkpoint and its active queue intent.
# They are not persisted lifecycle states: a queued resume must leave the checkpoint intact so
# another worker can recover it, while the API must not keep telling a person that answers are
# required after their answers have already been accepted.
RESUMING_EFFECTIVE_STATUS = "resuming"
RETRYING_EFFECTIVE_STATUS = "retrying"
# An accepted design verdict, before its attempt has run. Reported apart from `retrying`
# because a reader watching a feature that stopped on a design question needs to see that the
# decision landed -- "retrying" reads as the platform having another go on its own.
DECIDING_EFFECTIVE_STATUS = "acting_on_decision"
# An accepted publication request, before its pull requests exist. Reported apart from the
# rest because pressing publish and seeing the feature keep saying `failed_requires_human` is
# indistinguishable from the press having done nothing at all.
PUBLISHING_EFFECTIVE_STATUS = "publishing"
# An accepted revision request, before its run has advanced. Reported apart from the rest
# because a person who just asked for changes to a completed feature has to see that the
# request landed -- `planning` alone reads like the platform re-planning of its own accord.
REVISING_EFFECTIVE_STATUS = "revising"


class StartFeatureRequest(APIModel):
    """Start one parent feature workflow without provider credentials in its JSON body."""

    feature_id: str | None = Field(default=None, min_length=1)
    prd: PRDSubmission
    repositories: list[RepositorySpec] = Field(min_length=1)
    execution_mode: Literal["mock", "live"] = "mock"
    # Which model provider runs this feature's agents, start to finish. Fixed here and never
    # re-resolved: there is no endpoint that changes a running feature's platform, and no code
    # path that reads it back out of configuration. Same rule as `execution_mode`.
    #
    # The default is `openai` rather than the interface's, deliberately. This is the API's
    # answer for a caller that names no platform -- scripts and older clients -- and every
    # feature that predates the field ran on OpenAI. The submission form states its own
    # default, which is the one a person is shown.
    agent_platform: Literal["openai", "anthropic"] = "openai"
    # How expensively that provider's model roles resolve, fixed here like the platform
    # beside it. The default is `high` rather than the interface's `medium`, deliberately: it
    # is the unsuffixed configuration every deployment already runs, so a caller that never
    # heard of tiers -- scripts, older clients -- gets exactly today's behaviour. The
    # submission form states its own default, which is the one a person is shown.
    #
    # `custom` is deliberately not acceptable here: a caller selects a custom setup by naming
    # `model_setup_id`, and the server derives the tier. Accepting the tier word without the
    # setup would be a selection that resolves nothing.
    performance_tier: Literal["low", "medium", "high", "ultra"] = "high"
    # A user-authored model setup to run this feature on, owned by the caller. The server
    # snapshots it at acceptance, validates the snapshot, and pins it for the feature's life;
    # the id itself is provenance afterwards, never resolved through.
    model_setup_id: str | None = Field(default=None, min_length=1)
    idempotency_key: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def a_setup_and_a_tier_are_two_answers_to_one_question(self) -> StartFeatureRequest:
        """Refuse a request that names both a setup and an explicit tier or platform.

        Not a precedence rule, deliberately: two answers to one question is exactly the shape
        that let AB-Feature-173 run somewhere nobody chose. The defaults do not count -- only
        a field the caller actually sent -- so a client that always includes the fields it
        knows about keeps working the day it learns about setups, as long as it sends one
        selection or the other.
        """
        if self.model_setup_id is None:
            return self
        explicit = self.model_fields_set
        if "performance_tier" in explicit:
            msg = (
                "model_setup_id and performance_tier are two answers to one question; "
                "send one selection or the other"
            )
            raise ValueError(msg)
        if "agent_platform" in explicit:
            msg = (
                "model_setup_id and agent_platform are two answers to one question; the "
                "setup pins each role's platform, and the feature's platform is derived "
                "from its coding role"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="before")
    @classmethod
    def derive_repository_identity(cls, data: Any) -> Any:
        """Fill in a repository's identity from its URL when the caller did not supply one.

        Identity stays what it always was: `RepositorySpec` still requires a stable
        `repository_id` and carries it through branches, child workflow ids and workspace
        paths. What changed is who has to type it. Asking somebody to invent an identifier,
        a display name and a role before they can describe a feature was three questions
        whose answers were already in the URL they had just pasted.

        Derivation happens here, at the request boundary, rather than in the domain model,
        because only here is it known which values the caller actually omitted -- which is
        what makes it possible to disambiguate two repositories that share a name.
        """
        if not isinstance(data, dict):
            return data
        repositories = data.get("repositories")
        if not isinstance(repositories, list):
            return data

        derived: list[Any] = []
        # Two owners can publish `api`. Derived ids fall back to `owner-repository` when the
        # plain name is claimed twice; an explicitly supplied id always wins and is never
        # rewritten, so a caller that names its repositories keeps the names it chose.
        claimed = {
            str(item["repository_id"])
            for item in repositories
            if isinstance(item, dict) and item.get("repository_id")
        }
        for item in repositories:
            if not isinstance(item, dict):
                derived.append(item)
                continue
            entry = dict(item)
            url = entry.get("repository_url")
            owner, repository = _split_repository_url(url if isinstance(url, str) else "")
            if repository:
                if not entry.get("name"):
                    entry["name"] = repository
                if not entry.get("repository_id"):
                    candidate = _identifier(repository)
                    if candidate in claimed and owner:
                        candidate = _identifier(f"{owner}-{repository}")
                    entry["repository_id"] = candidate
                    claimed.add(candidate)
            # A role is a descriptive label the planner also assigns for itself. Left
            # unstated it must not imply anything: `shared` is the one role that makes
            # siblings wait for this repository, so the default deliberately is not it.
            if not entry.get("role"):
                entry["role"] = WorkstreamRole.OTHER.value
            derived.append(entry)

        return {**data, "repositories": derived}

    @field_validator("feature_id")
    @classmethod
    def feature_id_must_be_safe(cls, value: str | None) -> str | None:
        """Keep caller-selected ids safe for routes, branches, and durable child identities."""
        if value is not None and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,126}", value):
            raise ValueError(
                "feature_id may contain only letters, numbers, '.', '_', and '-', must start "
                "with a letter or number, and must be at most 127 characters"
            )
        return value

    @model_validator(mode="after")
    def at_least_one_required_repository(self) -> StartFeatureRequest:
        """Avoid creating a feature that has no completion gate at all."""
        if not any(item.required for item in self.repositories):
            msg = "at least one repository must be required"
            raise ValueError(msg)
        repository_ids = [item.repository_id for item in self.repositories]
        if len(repository_ids) != len(set(repository_ids)):
            raise ValueError("repository_ids must be unique")
        return self


def _split_repository_url(url: str) -> tuple[str, str]:
    """Return ``(owner, repository)`` from a repository URL, or empty strings.

    Deliberately forgiving: this runs before the URL itself has been validated, and a URL
    this cannot read is not this function's error to raise. `RepositorySpec` rejects a bad
    URL immediately afterwards with a message about the URL.
    """
    path = urlsplit(url).path.removesuffix(".git")
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return "", ""
    if len(segments) == 1:
        return "", segments[0]
    return segments[-2], segments[-1]


def _identifier(value: str) -> str:
    """Reduce a repository name to the character set ids are allowed to use."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    # A leading character outside the allowed set would fail `RepositorySpec`'s own check,
    # which is the right place for that to be reported rather than here.
    return cleaned[:127]


class ResumeFeatureRequest(APIModel):
    """Resolve parent clarification, or use an empty list to recover an interrupted feature."""

    answers: list[ClarificationAnswer] = Field(default_factory=list)


class RetryWorkstreamRequest(APIModel):
    """An explicit grant of more attempts to one repository the platform stopped.

    Both fields are required and neither may be empty. The platform stopped this repository
    because its budget ran out, and an override with no stated author and no stated reason is
    indistinguishable, three weeks later, from the platform having simply changed its mind.
    """

    # Deliberately small. A grant is a decision to spend money on evidence a person has read;
    # buying twenty attempts at once is buying them without reading the next four.
    additional_attempts: int = Field(default=1, ge=1, le=5)
    requested_by: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2000)


class PublishFeatureRequest(APIModel):
    """A person's decision to open the pull requests a feature that did not land is holding.

    ``reason`` is required for the reason a retry grant's is: this overrides a decision the
    platform made on purpose, and for a repository whose review rejected it, the pull request
    it opens carries code nothing approved. An override with no stated reason is
    indistinguishable, three weeks later, from the platform having simply changed its mind.

    ``requested_by`` is a label the client may supply for the record. It is never the audit
    answer on its own -- the route composes that from the authenticated identity.
    """

    requested_by: str = Field(default="", max_length=200)
    reason: str = Field(min_length=1, max_length=2000)


class ReviseFeatureRequest(APIModel):
    """A person's request to change a completed feature's published work.

    ``request`` is the change itself, in the person's own words. It becomes the revision
    run's requirement set verbatim, so it is bounded the way everything spliced into a
    coding prompt is, and it must actually say something.

    ``requested_by`` is a label the client may supply for the record. It is never the audit
    answer on its own -- the route composes that from the authenticated identity.
    """

    request: str = Field(min_length=1, max_length=20_000)
    requested_by: str = Field(default="", max_length=200)


class AnswerDesignConflictRequest(APIModel):
    """A person's decision about which position holds in one re-litigated design decision.

    ``decision`` is required and is not a formality. It is handed to the next attempt verbatim
    as the invariant it may not argue, so it is the difference between an instruction an
    engineer can follow and a preference it will reverse -- runs 193 and 194 showed that a
    clarification answer which encodes the implementation strategy is what precedes a delivery.
    Bounded because it is spliced into a coding prompt.
    """

    verdict: Literal["requirement_holds", "removal_holds"]
    decision: str = Field(min_length=1, max_length=2000)
    # Zero by default, and that default is the point. 49- Part B stops the loop without
    # spending an attempt, so the workstream this answers usually still has its budget; topping
    # it up by reflex would make an answered question look like a refund of attempts the stop
    # declined to spend. Anything above zero is an ordinary grant and is recorded as one.
    additional_attempts: int = Field(default=0, ge=0, le=5)


class CancelFeatureRequest(APIModel):
    """Reserved explicit payload for clients that require a body on cancellation requests."""

    reason: str | None = Field(default=None, min_length=1)


class RetireFeatureRequest(APIModel):
    """An operator's recorded decision to stop tracking a feature that cannot progress."""

    operator: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class UnresolvedOperationResponse(APIModel):
    """One external operation awaiting an operator decision, without provider detail."""

    operation_id: str
    workflow_id: str
    feature_id: str | None
    repository_id: str | None
    operation_type: str
    status: str
    attempt: int = Field(ge=0)
    max_attempts: int = Field(ge=0)
    heartbeat_at: datetime | None


class UnresolvedOperationsResponse(APIModel):
    """Everything currently needing reconciliation."""

    operations: list[UnresolvedOperationResponse]


class WorkstreamOperationRepeatResponse(APIModel):
    """Why one row is one of several of its type inside a single attempt.

    Two ``run_linter`` rows in one attempt read as a bug; both were real. The journal already
    encoded why, in ``repository_revision`` and ``command_fingerprint`` -- two first-class
    columns on the operation -- and no served field carried it, so an honest re-run after an
    in-attempt repair looked like a duplicate.

    Server-computed and typed on purpose. The same information exists in ``logical_step``,
    which lives in ``safe_metadata`` and is never published; and a client-side count of rows
    could only produce an ordinal, which silently changes meaning when the response's row
    budget truncates the attempt.
    """

    # new_revision | different_command | same_step
    kind: str
    # The short form of whichever column distinguishes this row: the revision for a re-run,
    # the command fingerprint for a sibling command. Absent where neither column can say.
    detail: str | None = None


class WorkstreamOperationResponse(APIModel):
    """One journal row for a repository workstream, without provider detail.

    ``safe_metadata`` is never published as metadata: the journal screens what it stores, but
    this endpoint must not become a second, unscreened channel. ``child_attempt`` is the one
    field read out of it, and it is published as a typed integer rather than as a passthrough
    -- one known key the platform writes itself, validated on the way out, so the channel
    stays as narrow as the field. Nothing else from the metadata travels.

    The rest of the row shape answers "what is running, since when, and is it alive" --
    timestamps are returned raw and the client computes ages, so the response stays
    deterministic and cacheable.
    """

    operation_id: str
    operation_type: str
    # The operation type mapped to its human stage name (setup, coding, validation, review,
    # publication, planning), so a client renders the stage without owning the mapping.
    stage: str
    status: str
    attempt: int = Field(ge=0)
    max_attempts: int = Field(ge=0)
    # Which of the child's attempts this row belongs to, or null where the journal cannot say:
    # a row written before the stamp existed, or one that belongs to no attempt at all.
    # Distinct from ``attempt`` above, which counts this one operation's own retries.
    child_attempt: int | None = Field(default=None, ge=0)
    started_at: datetime | None
    heartbeat_at: datetime | None
    completed_at: datetime | None
    error_code: str | None
    # Present only where this attempt holds more than one row of this operation type, and
    # only where the row is not simply the first run of its command.
    repeat: WorkstreamOperationRepeatResponse | None = None
    # How many times this row's model call had to be issued again because the stream it
    # opened said nothing inside the first-event budget. Typed and server-read, like
    # ``child_attempt`` above and for the same reason -- one known key, validated on the way
    # out, never a passthrough of the row's payload.
    #
    # Null where no measurement exists: a row that made no model call, a plain-POST
    # transport, or a row written before the field. `0` is a measurement and is served as
    # one, because the question this answers is whether the budget is close to binding, and
    # "never close" and "never measured" are different answers to it.
    stream_reissues: int | None = Field(default=None, ge=0)


class WorkstreamAttemptResponse(APIModel):
    """Where one finished attempt ended, assembled from records that already exist.

    The journal says what ran and succeeded. It cannot say where an attempt *ended*, and
    AB-Feature-201's backend attempt 0 is what that costs: 57 minutes, every journaled row
    ticked, stopped by the implementation self-review before the reviewer was called, and
    drawn as a wall of green.

    Served only for attempts the child has moved past -- the in-flight attempt has no
    ending by definition -- and assembled once per attempt, because an ending is immutable
    once its attempt is over.
    """

    attempt: int = Field(ge=0)
    # The record kind that ended this attempt's own cycle: self_review, review_rejected,
    # validation_failed, refusal, fault, superseded, approved or published. Existing record
    # kinds, never a new vocabulary.
    ended_by: str
    # The stage the ending landed in, in the same vocabulary the rows above carry, so a
    # client marks a stage without deciding which one it was.
    stage: str
    # One platform-composed sentence quoting the record: the finding, the command, the
    # refusal's first sentence.
    detail: str | None = None
    # This attempt's own recorded ``attempt_inputs["workspace"]`` -- fresh_checkout,
    # preserved, reset or recovered_coding_output. The recorded value rather than a boolean:
    # `reset` and `fresh_checkout` are different reasons for the same absence.
    workspace: str | None = None
    # The implementation self-review's own outcome for this attempt, or null where the
    # attempt recorded no self-review at all -- which 201's backend attempt 4 did, and which
    # a surface must not render as "clean".
    self_review_outcome: str | None = None
    # How many files a self-review correction rewrote. Files, not correction rounds:
    # ``corrections_applied`` is a list of paths.
    self_review_corrected_files: int | None = Field(default=None, ge=0)
    # In-attempt repair passes. Usually absent, and `0` where present, so a surface that
    # badges it renders nothing for either.
    source_repair_passes: int | None = Field(default=None, ge=0)
    # Re-issues spent by this attempt's unjournaled in-attempt passes, when any of them
    # measured it. Null means unmeasured, `0` means every stream spoke on the first issue --
    # and the two are different answers to "is the first-event budget close to binding".
    stream_reissues: int | None = Field(default=None, ge=0)


class WorkstreamOperationsResponse(APIModel):
    """One repository's journal rows, newest first and bounded, with per-attempt endings."""

    feature_id: str
    repository_id: str
    operations: list[WorkstreamOperationResponse]
    # One entry per finished attempt, oldest first. Empty for a repository on its first
    # attempt, and empty for every attempt of a deployment that has not recorded the
    # artifacts this is read from -- both of which a client already has honest words for.
    attempts: list[WorkstreamAttemptResponse] = Field(default_factory=list)


class IntegrationContractRevision(APIModel):
    """Human-approved replacement contract contents; the server adds its immutable envelope."""

    contract_version: str = Field(min_length=1)
    api_style: Literal["rest", "graphql", "grpc", "event_driven", "none", "mixed"]
    endpoints: list[EndpointContract]
    shared_schemas: list[SharedSchemaDefinition]
    authentication_contract: AuthenticationContract | None
    authorization_rules: list[AuthorizationRule]
    error_contracts: list[ErrorContract]
    event_contracts: list[EventContract]
    environment_variables: list[EnvironmentVariableContract]
    compatibility_policy: CompatibilityPolicy
    owning_workstreams: list[str] = Field(min_length=1)

    @field_validator("contract_version")
    @classmethod
    def contract_version_must_be_semantic(cls, value: str) -> str:
        """Reject unordered revision strings before they can reach a contract approval gate."""
        if not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", value):
            msg = "contract_version must use MAJOR.MINOR.PATCH semantic versioning"
            raise ValueError(msg)
        return value


class ApproveContractChangeRequest(APIModel):
    """Require the human owner to provide the complete next immutable contract revision."""

    resolution: str = Field(min_length=1)
    updated_contract: IntegrationContractRevision


class RejectContractChangeRequest(APIModel):
    """Record why the proposed deviation cannot be accepted."""

    resolution: str = Field(min_length=1)


class RepositorySummaryResponse(APIModel):
    """What a repository is, for a client that must render it without knowing its role.

    The submitted `RepositorySpec` was persisted and never returned by any read endpoint, so
    a client could show a workstream's identifier but not its name, role or URL.
    """

    repository_id: str
    name: str
    role: str
    repository_url: str
    default_branch: str
    required: bool
    implementation_order: int | None = None


class PinnedModelSetupRoleResponse(APIModel):
    """One role of a pinned setup, exactly as it was snapshotted at submission."""

    role: str
    platform: str
    model: str
    reasoning_effort: str | None = None
    max_tokens: int | None = None


class PinnedModelSetupResponse(APIModel):
    """The setup a custom feature was pinned to, in full.

    "In full" is the requirement: a record naming only the setup id is a dangling pointer
    once the setup is edited. `setup_state` says what became of the setup since -- pinning
    survives either answer.
    """

    setup_id: str
    name: str
    roles: list[PinnedModelSetupRoleResponse]
    # `unchanged` | `edited` | `deleted`: whether the id still resolves to a setup whose role
    # map matches this snapshot. Presentation only; execution reads the snapshot regardless.
    setup_state: Literal["unchanged", "edited", "deleted"] = "unchanged"


class FeatureResponse(APIModel):
    """Public snapshot of the parent feature lifecycle without any credential fields."""

    feature_id: str
    workflow_id: str
    status: FeatureWorkflowStatus
    # Usually identical to `status`. While an asynchronous resume or retry owns the queue,
    # this is the truthful live phase and `status` remains the durable recovery checkpoint.
    effective_status: str
    title: str
    # The identity people use for this feature: `AB-Feature-42`. Allocated by the server at
    # creation and immutable afterwards. `None` only for a feature created before references
    # existed whose backfill has not run.
    reference: str | None = None
    # How many times a person has revised this feature after completion. 0 is the original
    # run, which is also what every feature from before revisions existed reads as.
    revision: int = Field(default=0, ge=0)
    current_agent: str | None
    repository_count: int = Field(ge=1)
    required_repository_count: int = Field(ge=1)
    repositories: list[RepositorySummaryResponse] = Field(default_factory=list)
    clarification_rounds: int = Field(ge=0)
    integration_review_cycles: int = Field(ge=0)
    # The budgets this feature is actually being run under, so a client can say "attempt 2
    # of 8" instead of "attempt 2". They are read from the feature's own state rather than
    # from current settings: a feature started last week keeps the limits it started with,
    # and reporting today's configuration against last week's counters would be wrong.
    max_clarification_rounds: int = Field(ge=1)
    max_integration_review_cycles: int = Field(ge=1)
    max_child_review_cycles: int = Field(ge=1)
    max_implementation_retries: int = Field(ge=0)
    max_validation_retries: int = Field(ge=0)
    max_repository_setup_retries: int = Field(ge=0)
    merge_strategy: MergeStrategy | None
    deployment_strategy: DeploymentStrategy | None
    execution_mode: Literal["mock", "live"]
    # Which model provider actually ran this feature. Read from the feature's own state rather
    # than from current configuration, for the same reason the budgets above are: a feature
    # started last week keeps what it started on.
    agent_platform: Literal["openai", "anthropic"] = "openai"
    # The tier beside it, for the same reason: a feature keeps the cost basis it started on,
    # and `high` is what every feature that predates the field actually ran at. `custom`
    # means `model_setup` below carries the pinned role map.
    performance_tier: Literal["low", "medium", "high", "ultra", "custom"] = "high"
    # The pinned setup of a custom feature: its four roles as snapshotted at submission, the
    # setup's id and name, and whether that setup still exists unchanged. G3 made visible: a
    # feature shows what it actually runs, however the setup has been edited since.
    model_setup: PinnedModelSetupResponse | None = None
    cancellation_status: CancellationLifecycleStatus = CancellationLifecycleStatus.NOT_REQUESTED
    cancellation_requested_at: datetime | None = None
    cancellation_reason: str | None = None
    cleanup_requirements: list[ManualCleanupRequirement] = Field(default_factory=list)
    # The pre-coding stages' two clocks: what their agent calls took, and how much of that
    # classified provider faults consumed. `None` where nothing was measured -- a feature
    # from before the clocks existed, or one that has not planned yet.
    planning_wall_seconds: float | None = Field(default=None, ge=0)
    planning_provider_fault_seconds: float | None = Field(default=None, ge=0)
    failure_summary: FeatureFailureSummary | None = None
    available_actions: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class StartFeatureResponse(FeatureResponse):
    """Feature start result, including idempotent replay state."""

    created: bool


class FeatureArtifactResponse(APIModel):
    """One feature artifact's public envelope and JSON-safe payload."""

    artifact_id: str
    artifact_type: str
    workflow_id: str
    schema_version: str
    producer: str
    timestamp: datetime
    metadata: dict[str, Any]
    validation_status: Literal["valid", "invalid", "pending"]
    payload: dict[str, Any]


class ClarificationQuestionResponse(APIModel):
    """One open question the feature is waiting on, with the answer field the client posts."""

    question_id: str
    question: str
    rationale: str
    required: bool
    # What the platform would answer, and where it read that. Empty when it has no grounded
    # answer -- a client must not invent one, and must not submit one without being told to.
    suggested_answer: str = ""
    suggestion_source: str = ""
    suggestion_confidence: str | None = None


# Which of the clarification surface's states this feature is in. AB-Feature-173's panel
# polled fifty times over thirty-five minutes reading "waiting for you" while the platform
# was investigating the questions itself; these are the three situations that rendering
# collapsed, plus the rest state, serialized rather than re-derived by the client.
ClarificationState = Literal[
    # No open questions, or a resume already owns them.
    "idle",
    # Unresolved questions exist and the platform is answering them itself (reconnaissance
    # and grounding in flight). The questions are its open items, explicitly not the user's.
    "investigating",
    # The platform decided it needs the human: today's answer form.
    "awaiting_answers",
    # awaiting_answers, plus the fact that grounding failed and the human is the fallback.
    "asked_after_grounding_failure",
    # A workstream stopped because a design decision is being re-litigated, and the question
    # is a person's to settle. A different kind of question from the four above -- it is about
    # one repository rather than the requirement, it arrives after coding rather than before
    # it, and the feature it belongs to has already stopped -- so it carries its own payload
    # (`design_conflicts`) rather than being folded into `questions`, whose contract is the
    # technical PRD's unresolved questions and whose answers reopen planning.
    "awaiting_design_verdict",
]


class DesignConflictPositionResponse(APIModel):
    """One side of a re-litigated design decision, in the words it was actually put in."""

    # Which review holds this position. Named as the review rather than the agent: what a
    # person arbitrates is which review is right, not which model wrote it.
    authority: Literal["repository_review", "integration_review"]
    # The demand itself, whole. A half-quoted requirement is not a position anybody can decide
    # between, which is the same capture discipline the invariant section follows.
    statement: str
    # What the attempt that stopped reporting this said it had done. Present only on the
    # satisfied side and only where the lineage holds a completion summary for that cycle --
    # an integration review records a verdict on someone else's work, never an account of it.
    grounds: str = ""


class DesignConflictResponse(APIModel):
    """A design decision this workstream keeps reversing, as an answerable question.

    Served whole because the decision cannot be made from the current demand alone: what makes
    it a question rather than a defect is that the same thing was required, delivered, and then
    let go, and only both positions show that.
    """

    conflict_id: str
    repository_id: str
    # Which shape of re-litigation this question is: a `reversal` carries both positions; a
    # `recurring_demand` (77-, item 35) was never satisfied, so it has no second position and
    # its evidence quotes every wording the review used.
    kind: Literal["reversal", "recurring_demand"] = "reversal"
    # The platform's own sentence to the reader, and the lineage facts behind it. Both come
    # from the stop, so this and the stopped attempt's blocking issue cannot disagree.
    question: str
    evidence: list[str] = Field(default_factory=list)
    demanded: DesignConflictPositionResponse
    # Absent for a recurring demand: nothing was ever satisfied, and rendering an empty
    # second position would claim the workstream built something it never did.
    satisfied: DesignConflictPositionResponse | None = None
    # Whether the two positions come from different reviews. Published rather than derived from
    # the two authority fields, so a client cannot render a same-review reversal as a
    # disagreement between reviewers -- they send a reader to two different decisions.
    cross_authority: bool = False
    removals: int = Field(ge=0)
    attempts_spent: int = Field(ge=0)
    # How many attempts this repository can still spend. Zero means answering the question also
    # requires granting one, and the client says so rather than letting the request be refused.
    attempts_remaining: int = Field(ge=0)
    # Whether the platform would accept a verdict right now. False for a question whose
    # repository has run out of review cycles or is no longer stopped: the decision is still
    # worth reading, and there is nothing for it to resume.
    answerable: bool = True


class ClarificationResponse(APIModel):
    """The questions a human must answer before this feature can continue.

    Served rather than derived on the client: the current technical PRD may be a revision
    (`002_technical_prd.revision-3.json`), and deciding which artifact is current is a
    server rule the client must not re-implement. The same goes for `clarification_state`:
    the client renders it, and never works it out from status plus question count.
    """

    feature_id: str
    awaiting_answers: bool
    clarification_state: ClarificationState = "idle"
    technical_prd_artifact_id: str | None
    clarification_rounds: int = Field(ge=0)
    max_clarification_rounds: int = Field(ge=0)
    questions: list[ClarificationQuestionResponse] = Field(default_factory=list)
    previous_answers: dict[str, str] = Field(default_factory=dict)
    # The design decisions waiting on a person, oldest first. Deliberately on this surface
    # rather than an endpoint of its own: this is the read a client already polls to find out
    # what is waiting on the user, and a second one would make "is anything waiting on me"
    # two questions with two answers that could disagree.
    design_conflicts: list[DesignConflictResponse] = Field(default_factory=list)


class FeatureArtifactsResponse(APIModel):
    """All parent and child handoff artifacts associated with a feature."""

    feature_id: str
    artifacts: list[FeatureArtifactResponse]


class WorkstreamResponse(APIModel):
    """Public lifecycle status for one independently persisted repository child."""

    repository_id: str
    # Carried alongside the identifier so a client renders a workstream without a second
    # lookup, and without inferring anything from the role.
    repository_name: str | None = None
    repository_role: str | None = None
    repository_url: str | None = None
    repository_required: bool | None = None
    child_workflow_id: str
    workstream_id: str
    status: ChildWorkflowStatus
    branch_name: str
    # The branch this workstream's checkout was created from; null means the repository
    # default. Set by a feature revision to the branch it superseded.
    base_branch: str | None = None
    workspace_path: str
    retry_count: int = Field(ge=0)
    code_completion_artifact_id: str | None
    review_artifact_id: str | None
    blocking_issues: list[str]
    pull_request_artifact_id: str | None
    checkpoint_boundary: WorkflowCheckpointBoundary | None = None
    technology_profile: dict[str, Any] | None = None
    validation_plan: dict[str, Any] | None = None
    current_revision: str | None = None
    current_validation_results: list[dict[str, Any]] = Field(default_factory=list)
    superseded_validation_count: int = Field(default=0, ge=0)
    scoped_requirements: list[dict[str, Any]] = Field(default_factory=list)
    out_of_scope_requirements: list[str] = Field(default_factory=list)
    preflight_status: str | None = None
    blocking_setup_issues: list[dict[str, Any]] = Field(default_factory=list)
    selected_package_manager: str | None = None
    layout_evidence: dict[str, Any] | None = None
    configured_validation_commands: list[dict[str, Any]] = Field(default_factory=list)
    test_availability: str | None = None
    implementation_expectations: list[dict[str, Any]] = Field(default_factory=list)
    production_files_changed: list[str] = Field(default_factory=list)
    test_files_changed: list[str] = Field(default_factory=list)
    configuration_files_changed: list[str] = Field(default_factory=list)
    requirements_implemented: list[str] = Field(default_factory=list)
    requirements_not_implemented: list[str] = Field(default_factory=list)
    failure_classification: str | None = None
    retry_strategy: dict[str, Any] | None = None
    # Which configured model role the next attempt uses, and why. Published because a reader
    # should see the classified execution decision, not infer it from a retry counter.
    model_routing: dict[str, Any] | None = None
    implementation_retry_count: int = Field(default=0, ge=0)
    validation_retry_count: int = Field(default=0, ge=0)
    repository_setup_retry_count: int = Field(default=0, ge=0)
    integration_retry_count: int = Field(default=0, ge=0)
    # Which authority demanded the attempt this repository is currently on:
    # `integration_remediation`, `operator_retry`, or null for a workstream retry -- which is
    # also what every attempt recorded before the platform stamped this reads as. The latest
    # attempt's and no other's, like the failure classification beside it. Published because
    # the read model publishes what the platform records; the graph draws each arrow's origin
    # from the execution records, which carry it per attempt.
    targeted_attempt_kind: str | None = None
    # The in-flight context-partition wave, when one is: the clusters a refused retry's
    # blocking demands were split into, and how many scoped attempts beyond the first have
    # run. Published because the read model publishes what the platform records.
    pending_context_partition: list[dict[str, Any]] | None = None
    context_partition_count: int = Field(default=0, ge=0)
    # An override of the platform's own stop is not something to discover from a counter that
    # no longer matches the configured limit, so the grants are published as themselves.
    granted_extra_attempts: int = Field(default=0, ge=0)
    retry_grants: list[dict[str, Any]] = Field(default_factory=list)
    meaningful_change: bool | None = None
    meaningful_change_reason: str | None = None
    production_diff_fingerprint: str | None = None
    previous_attempt_fingerprint: str | None = None
    test_diff_fingerprint: str | None = None
    previous_test_fingerprint: str | None = None
    retry_refusal_reason: str | None = None
    # The two clocks of the workstream's run: everything since its loop started, and that
    # time minus what was spent inside classified provider faults. The runtime ceiling
    # judges the charged clock, and the difference is what the provider cost this run.
    runtime_wall_seconds: float | None = Field(default=None, ge=0)
    runtime_charged_seconds: float | None = Field(default=None, ge=0)
    # Reconnaissance failed for this repository and its plan was written without checkout
    # evidence. An operator deciding whether to trust the plan reads this here instead of
    # `docker logs`; the reason is sanitized at the fail-soft that recorded it.
    planned_blind: bool = False
    planned_blind_reason: str | None = None
    available_actions: list[str] = Field(default_factory=list)
    # Whether this repository would be published if somebody pressed `PUBLISH_FEATURE`, and
    # under which of the two classes: `reviewed` for work that passed review and has no pull
    # request yet, `unreviewed` for work whose every required check passed and whose review
    # rejected it. `null` means it would not be, and `publication_refusal` says why -- so a
    # client can render "2 ready to open, 1 held for lint" rather than a bare button.
    #
    # Computed by the route from the workflow's own precondition, never dumped off the child
    # record: `WorkstreamResponse` forbids extra fields and is built by dumping the whole
    # `ChildWorkflowReference`, so a new field on that model would break this endpoint.
    publication_class: str | None = None
    publication_refusal: str | None = None


class WorkstreamsResponse(APIModel):
    """Feature child workstream list."""

    feature_id: str
    workstreams: list[WorkstreamResponse]


class PullRequestsResponse(APIModel):
    """Only the coordinated PR artifacts emitted after integration approval."""

    feature_id: str
    pull_requests: list[FeatureArtifactResponse]


class FeatureSummaryResponse(APIModel):
    """Enough of a feature to recognise and reopen it, without its artifacts."""

    feature_id: str
    workflow_id: str
    title: str
    reference: str | None = None
    status: FeatureWorkflowStatus
    execution_mode: str
    agent_platform: str = "openai"
    performance_tier: str = "high"
    created_at: datetime
    updated_at: datetime
    repository_count: int = Field(default=0, ge=0)
    pull_request_count: int = Field(default=0, ge=0)
    # Derived from status alone. `current_agent` is deliberately absent: it lives inside the
    # parent state blob, and reading it per row would make listing recent work cost more than
    # the work. A client that needs it opens the feature.
    human_action_required: bool = False
    dashboard_group: Literal["queued", "running", "waiting", "failed", "completed", "cancelled"]


class FeatureListResponse(APIModel):
    """One page of features, newest first, with the cursor that continues it."""

    features: list[FeatureSummaryResponse]
    next_cursor: str | None = None


class FeatureTimelineEventResponse(APIModel):
    """A safe, chronological parent lifecycle or artifact event."""

    timestamp: datetime
    event_type: Literal["lifecycle", "artifact"]
    source: str
    event: str
    details: dict[str, Any]


class FeatureEventResponse(APIModel):
    """One durable lifecycle event, carrying the cursor that continues after it."""

    id: int
    timestamp: datetime
    event_type: str
    source: str
    event: str
    details: dict[str, Any]


class FeatureEventsResponse(APIModel):
    """Events after a cursor, so a client polls a delta instead of a whole history.

    Read straight from the indexed event table. The timeline endpoint hydrates parent state,
    which for a completed feature carries every artifact, so polling that for liveness would
    make watching a feature more expensive than running it.
    """

    feature_id: str
    events: list[FeatureEventResponse]
    last_event_id: int | None


class LogbookRecordResponse(APIModel):
    """The durable row one logbook bubble was read from.

    A reference rather than a URL: which page shows an artifact or an event is the client's
    decision, and the server's job is to name the record so the link cannot point at
    something that is not the bubble's source.
    """

    kind: Literal["event", "artifact", "operation", "workstream"]
    id: str
    repository_id: str | None = None


class LogbookEntryResponse(APIModel):
    """One bubble of a feature's run, told from one record.

    ``template`` is the registry key the sentence came from -- published so a reader can tell
    two bubbles of the same kind apart, and so Slack delivery (which reuses the same registry)
    and this tab can be checked against each other. ``quote`` is the agent's own persisted
    prose, clipped at a whole word; ``quote_source`` names the field it was read from, and the
    full text is behind the record link.

    ``emission`` is the per-record ordinal: which of its record's bubbles this one is.
    ``(record.kind, record.id, emission)`` is the identity that is stable across
    recompositions -- ``sequence`` is a render-time index that any late-arriving record
    shifts -- and it is what the Slack delivery ledger keys on. A published contract, like
    the template keys.
    """

    sequence: int
    emission: int
    timestamp: datetime
    agent: str
    tone: Literal["done", "stopped", "attention", "working"]
    template: str
    text: str
    detail: str | None = None
    quote: str | None = None
    quote_source: str | None = None
    record: LogbookRecordResponse
    repository_id: str | None = None


class LogbookResponse(APIModel):
    """A feature's run as a chronological thread, oldest first and paginated.

    Composed at read time from durable records only: no logbook is stored, so a feature that
    finished before this endpoint existed tells its whole story the first time it is asked.
    """

    feature_id: str
    entries: list[LogbookEntryResponse]
    # The sequence to pass as `after` for the next page, or null at the end of the thread.
    next_cursor: int | None = None
    # Every chip the registry can attribute a bubble to, so a client renders the legend from
    # the server's vocabulary instead of holding a second copy of it.
    agents: list[str] = Field(default_factory=list)


class ProposedActionResponse(APIModel):
    """An action the assistant proposed and a person has not yet confirmed."""

    type: str
    arguments: dict[str, Any]
    summary: str


class ChatMessageResponse(APIModel):
    """One turn of a feature conversation."""

    id: int
    role: str
    content: str
    proposed_action: ProposedActionResponse | None = None
    action_status: str | None = None
    action_result: str | None = None
    # The durable action a confirmed proposal became, so a client that reloads mid-execution
    # can follow the action itself rather than guessing from a status string.
    action_id: str | None = None
    created_at: datetime | None = None


class FeatureActionResponse(APIModel):
    """One durable workflow-changing action and what became of it.

    Deliberately excludes the lease owner. It names a process and a build, which is operator
    detail rather than something a browser needs, and publishing it would invite a client to
    reason about ownership the server is responsible for.
    """

    action_id: str
    feature_id: str
    repository_id: str | None = None
    action_type: str
    actor_id: str
    actor_display_name: str | None = None
    origin: str
    origin_message_id: int | None = None
    status: FeatureActionStatus
    attempt: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    # Whether some executor still credibly owns this. Derived server-side because it is a
    # comparison against server time, and a client's clock is not the platform's.
    in_progress: bool = False
    result_summary: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    external_operation_ids: list[str] = Field(default_factory=list)
    reconciled_by: str | None = None
    reconciliation_reason: str | None = None
    reconciled_at: datetime | None = None


class ReconcileFeatureActionRequest(APIModel):
    """An administrator's explicit decision after checking an uncertain operation."""

    outcome: Literal["succeeded", "failed"]
    reason: str = Field(min_length=1, max_length=2000)


class FeatureActionsResponse(APIModel):
    """A feature's durable actions, newest first."""

    feature_id: str
    actions: list[FeatureActionResponse]


class RepositoryRepairCommandResponse(APIModel):
    """One command an approved repair may run, and why."""

    command: list[str]
    working_directory: str | None = None
    purpose: str


class RepositoryRepairResponse(APIModel):
    """A repository the platform will not change without being told to, and what it proposes."""

    repair_id: str
    feature_id: str
    repository_id: str
    originating_stage: str
    failure_classification: str
    detected_problem: str
    evidence: list[str] = Field(default_factory=list)
    proposed_repair: str
    affected_files: list[str] = Field(default_factory=list)
    affected_dependencies: list[str] = Field(default_factory=list)
    commands: list[RepositoryRepairCommandResponse] = Field(default_factory=list)
    expected_impact: str
    risk: Literal["low", "medium", "high"]
    changes_source_logic: bool
    proposed_at_revision: str | None = None
    status: RepositoryRepairStatus
    approved_by: str | None = None
    approved_at: datetime | None = None
    rejected_by: str | None = None
    rejection_reason: str | None = None
    execution_result: str | None = None
    resulting_revision: str | None = None
    # Whether the checkout has moved since the diagnosis was written. A stale proposal is
    # shown as stale rather than quietly offered for approval, because the problem it
    # describes may already be gone.
    stale: bool = False
    created_at: datetime | None = None


class RepositoryRepairsResponse(APIModel):
    """Every repair proposed for one feature, newest first."""

    feature_id: str
    repairs: list[RepositoryRepairResponse]


class ApproveRepairRequest(APIModel):
    """Authorize one repair to be applied to a repository."""

    # Required, and required to be true. A repair that changes files or dependencies is a
    # change to somebody's repository; the console asks before sending this, and the server
    # refuses without it rather than trusting that it did.
    acknowledge_repository_change: bool = False
    note: str | None = Field(default=None, max_length=2_000)


class RejectRepairRequest(APIModel):
    """Decline one repair, recording why so the workstream's stop stays explained."""

    reason: str = Field(min_length=1, max_length=2_000)


class ChatHistoryResponse(APIModel):
    """The conversation for one feature, oldest first."""

    feature_id: str
    messages: list[ChatMessageResponse]


class SendChatMessageRequest(APIModel):
    """One question about this feature."""

    message: str = Field(min_length=1, max_length=4000)


class FeatureTimelineResponse(APIModel):
    """Chronological parent feature history."""

    feature_id: str
    events: list[FeatureTimelineEventResponse]


class FeatureExecutionsResponse(APIModel):
    """Who performed each transition in this feature's execution.

    One response for the whole graph rather than one per arrow: a five-repository feature on
    its fourth attempt has well over a hundred transitions, and a request per edge would make
    watching a feature cost more than running it.

    Every record is derived from durable state. Nothing here is recomputed from current model
    configuration, so a completed transition keeps naming the model that actually ran it after
    a deployment changes what a role resolves to.
    """

    feature_id: str
    executions: list[ExecutionRecord]
