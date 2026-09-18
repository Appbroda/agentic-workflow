"""What actually performed each transition in a feature's execution, read from durable state.

A feature's graph has always been able to say *where* execution is. This says *who moved it*:
for every transition between two stages, which model or which deterministic handler performed
it, on which attempt, against which revision, and -- where the transition was a retry -- what
the previous attempt failed on and what the next one was asked to change.

Nothing here is a second source of truth. Every field is read from the records the workflow
already persists:

* the immutable artifact each agent wrote, whose metadata names the model that produced it;
* the per-attempt ``child_workflow_result`` artifact, which carries that attempt's status,
  blocking issues, failure classification, retry plan, revision and routing decision;
* the child workflow reference, which carries the *current* attempt and the routing decision
  already made for it;
* the feature snapshot, whose own budgets are the denominators any ratio is measured against.

Two rules run through all of it.

**Never claim a model executed something a model did not.** Validation is a subprocess, the
commit and the push are Git, the pull request is a GitHub API call, and the fan-out is this
orchestrator. Those transitions name their real handler. A model is named only where an
artifact records one.

**Never reconstruct a historical model from current configuration.** The model on a completed
transition is the one its own artifact recorded. A deployment that changes
``OPENAI_CODING_MODEL`` today does not change what yesterday's attempt ran on, and this module
has no access to the model configuration at all -- which is what makes that impossible rather
than merely unintended.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from artifacts.schemas import (
    Artifact,
    ChildWorkflowResultArtifact,
    CodeCompletionArtifact,
    IntegrationContractArtifact,
    IntegrationReviewArtifact,
    PullRequestArtifact,
    RepositoryExecutionPlanArtifact,
    RepositoryRepairProposalArtifact,
    ReviewArtifact,
    TechnicalPRDArtifact,
)
from state.enums import (
    ChildWorkflowStatus,
    FeatureWorkflowStatus,
    RepositoryRepairStatus,
    TargetedAttempt,
)
from state.external_operations import (
    ExternalOperation,
    ExternalOperationStatus,
    ExternalOperationType,
)
from state.feature_models import ChildWorkflowReference, FeatureWorkflowSnapshot
from state.models import StateModel


class ExecutionStage(StrEnum):
    """The stages a transition can run between.

    A server-owned vocabulary rather than the graph's own node identifiers: a client decides
    how it draws these, and the platform decides what they are. Adding a stage here is a
    published contract change; renaming a node in a browser is not.
    """

    REQUEST = "request"
    # Reached only by a feature that cited a design. Adding a member here is a published
    # contract change and is treated as one: a client that has never heard of it renders no
    # node for it, which is exactly what every feature that cites nothing already shows.
    DESIGN_SNAPSHOT = "design_snapshot"
    TECHNICAL_PRD = "technical_prd"
    INTEGRATION_CONTRACT = "integration_contract"
    EXECUTION_PLAN = "execution_plan"
    REPOSITORY = "repository"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    REVIEW = "review"
    INTEGRATION_REVIEW = "integration_review"
    PULL_REQUESTS = "pull_requests"


class ExecutionHandlerType(StrEnum):
    """What kind of thing performed a transition.

    The distinction exists so that a deterministic step is never dressed as an AI one. A
    ``DETERMINISTIC`` execution has a handler name and no model, and that is not missing
    information -- it is the information.
    """

    MODEL = "model"
    DETERMINISTIC = "deterministic"
    HUMAN = "human"


class ExecutionStatus(StrEnum):
    """How a transition is going, in the workflow's own vocabulary.

    Deliberately not a new state machine. Each value is derived from the feature status, the
    child workflow status and which artifacts exist -- the three things that already decide
    what the graph draws.
    """

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    NEEDS_HUMAN = "needs_human"
    CANCELLED = "cancelled"


class HumanAction(StrEnum):
    """Which human gate a transition is waiting at, where it is waiting at one."""

    CLARIFICATION = "clarification"
    REPOSITORY_REPAIR = "repository_repair"
    CONTRACT_CHANGE = "contract_change"
    RETRY_REFUSED = "retry_refused"


class ExecutionRecord(StateModel):
    """One execution transition, and everything durable state can say about it.

    Every optional field is optional because the record may genuinely not contain it, not
    because it is inconvenient to fill in. A reader may treat ``None`` as "the platform did not
    record this", and never as "the platform recorded nothing interesting".
    """

    # Stable across polls and distinct per attempt, because two attempts produce two
    # executions between the same pair of stages. Derived from the artifact or attempt it
    # describes, so it survives a restart and identifies the same execution afterwards.
    execution_id: str = Field(min_length=1)
    from_stage: ExecutionStage
    to_stage: ExecutionStage
    repository_id: str | None = None
    is_retry: bool = False

    handler_type: ExecutionHandlerType
    # Always populated: the short name of whatever performed this. For a model execution it is
    # the agent role -- "Engineer", "Code reviewer" -- because the role is what the platform
    # asked to do the work, and the model is a separate field on purpose.
    handler: str = Field(min_length=1)
    agent_type: str | None = None

    provider: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    model_role: str | None = None
    # The configuration key the role read, kept for the technical detail view. Never a value
    # from it, and never a credential.
    model_variable: str | None = None
    # False where the platform has not resolved a model for this transition yet. A client must
    # say so rather than predicting one.
    model_resolved: bool = False

    status: ExecutionStatus
    attempt: int | None = Field(default=None, ge=1)
    max_attempts: int | None = Field(default=None, ge=1)
    # The platform's own sentence for what the ratio counts, so a client never has to invent
    # one and two clients cannot invent different ones.
    attempt_meaning: str | None = None
    # The per-classification budget this attempt was actually granted against, where the
    # record names one. Separate from the attempt ratio because they are separate budgets.
    counter_label: str | None = None
    counter_value: int | None = Field(default=None, ge=0)
    counter_limit: int | None = Field(default=None, ge=0)

    failure_classification: str | None = None
    # Why the previous attempt did not pass, in the words the platform recorded -- a review
    # finding, a failed command, a blocking issue. Never a model's reasoning.
    failure_summary: str | None = None
    failure_severity: str | None = None
    # What the next attempt was asked to change. Read from the platform's own retry plan, which
    # is a different thing from the failure: a broken lint configuration fails as a lint error
    # and is remediated by repairing the repository.
    remediation_summary: str | None = None
    routing_reason: str | None = None

    review_verdict: str | None = None
    review_finding_count: int | None = Field(default=None, ge=0)
    # How that count divided, where the platform recorded the division. `untraceable` is the
    # number of findings that would have blocked under the unbounded rule and are advisory
    # now: the size of the behaviour change on this one review, not a rate over a run.
    review_findings_blocking: int | None = Field(default=None, ge=0)
    review_findings_advisory: int | None = Field(default=None, ge=0)
    review_findings_untraceable: int | None = Field(default=None, ge=0)
    validation_passed: int | None = Field(default=None, ge=0)
    validation_total: int | None = Field(default=None, ge=0)
    command: list[str] = Field(default_factory=list)
    exit_code: int | None = None

    revision_before: str | None = None
    revision_after: str | None = None
    contract_version: str | None = None
    # The effective identity of this execution, which is what the operation journal reconciles
    # against. Internal detail, published because an engineer reading a repeated attempt needs
    # to see whether the platform considered it the same question.
    input_fingerprint: str | None = None

    # Where to go for the evidence. The drawer routes; it does not rebuild these views.
    artifact_id: str | None = None
    code_completion_artifact_id: str | None = None
    review_artifact_id: str | None = None
    result_artifact_id: str | None = None
    pull_request_artifact_id: str | None = None
    repair_id: str | None = None
    previous_execution_id: str | None = None

    human_action: HumanAction | None = None
    human_requirement: str | None = None

    # The producing artifact's timestamp for artifact-derived records. Operation-derived
    # records (the journaled pre-coding calls) also carry a real start and last heartbeat,
    # because the journal measures both -- a running call with a widening
    # started_at->heartbeat_at gap is exactly what "is it moving?" reads.
    started_at: datetime | None = None
    heartbeat_at: datetime | None = None
    completed_at: datetime | None = None
    # Only where a real measurement exists: the validation commands time themselves.
    duration_seconds: float | None = Field(default=None, ge=0)
    execution_mode: str = Field(min_length=1)


# --------------------------------------------------------------------------- model metadata

# The canonical block agents now write. One key, one shape, one reader.
_EXECUTION_METADATA_KEY = "execution"

# What agents wrote before that block existed, in no particular order because each belongs to
# a different agent. Artifacts are immutable, so a feature that ran last week still has to be
# able to name its model -- reading these is how historical accuracy is kept without
# rewriting history.
_LEGACY_MODEL_KEYS = ("model", "coding_model", "seam_review_model")


class _ModelFacts(StateModel):
    """What one artifact's metadata says about the model that produced it."""

    provider: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    model_role: str | None = None
    model_variable: str | None = None
    agent_type: str | None = None
    routing_reason: str | None = None


def _text(value: object) -> str | None:
    """Return a non-empty string, or nothing. Anything else in the record is not an answer."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _mapping(value: object) -> Mapping[str, Any]:
    """Return a metadata sub-object, tolerating a record that does not carry one."""
    return value if isinstance(value, Mapping) else {}


def _model_facts(metadata: Mapping[str, Any]) -> _ModelFacts:
    """Read the model that produced an artifact from the artifact's own metadata."""
    block = _mapping(metadata.get(_EXECUTION_METADATA_KEY))
    if block:
        return _ModelFacts(
            provider=_text(block.get("provider")),
            model=_text(block.get("model")),
            reasoning_effort=_text(block.get("reasoning_effort")),
            model_role=_text(block.get("model_role")),
            model_variable=_text(block.get("model_variable")),
            agent_type=_text(block.get("agent_type")),
            routing_reason=_text(block.get("routing_reason")),
        )
    legacy = next(
        (_text(metadata.get(key)) for key in _LEGACY_MODEL_KEYS if metadata.get(key)), None
    )
    routing = _mapping(metadata.get("model_routing"))
    return _ModelFacts(
        model=legacy or _text(routing.get("model")),
        reasoning_effort=_text(routing.get("reasoning")),
        model_role=_text(metadata.get("model_role")) or _text(routing.get("role")),
        routing_reason=_text(routing.get("routing_reason")),
    )


def _routing_facts(routing: Mapping[str, Any]) -> _ModelFacts:
    """Read the model a routing decision selected, which is durable per attempt."""
    return _ModelFacts(
        model=_text(routing.get("model")),
        reasoning_effort=_text(routing.get("reasoning")),
        model_role=_text(routing.get("role")),
        model_variable=_text(routing.get("model_variable")),
        routing_reason=_text(routing.get("routing_reason")),
    )


def _merged(*facts: _ModelFacts) -> _ModelFacts:
    """Combine what several records say, preferring the first that answers each field."""
    return _ModelFacts(
        provider=next((item.provider for item in facts if item.provider), None),
        model=next((item.model for item in facts if item.model), None),
        reasoning_effort=next(
            (item.reasoning_effort for item in facts if item.reasoning_effort), None
        ),
        model_role=next((item.model_role for item in facts if item.model_role), None),
        model_variable=next((item.model_variable for item in facts if item.model_variable), None),
        agent_type=next((item.agent_type for item in facts if item.agent_type), None),
        routing_reason=next((item.routing_reason for item in facts if item.routing_reason), None),
    )


# --------------------------------------------------------------------------- status helpers

_FEATURE_STOPPED = frozenset(
    {
        FeatureWorkflowStatus.FAILED,
        FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
    }
)
_FEATURE_CANCELLED = frozenset(
    {
        FeatureWorkflowStatus.CANCELLING,
        FeatureWorkflowStatus.CANCELLED,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    }
)
_CHILD_DONE = frozenset({ChildWorkflowStatus.APPROVED, ChildWorkflowStatus.COMPLETED})
_CHILD_STOPPED = frozenset(
    {
        ChildWorkflowStatus.FAILED,
        ChildWorkflowStatus.REVIEW_REJECTED,
        ChildWorkflowStatus.CANCELLED,
    }
)

# Deterministic handlers, named once so no caller invents a different word for the same thing.
_ORCHESTRATOR = "Orchestrator"
_VALIDATOR = "Validator"
_GITHUB = "GitHub"
_HUMAN = "Human action"


class _Reader:
    """One feature's durable state, indexed the few ways this module reads it."""

    def __init__(self, state: FeatureWorkflowSnapshot) -> None:
        """Index the artifacts once; every execution below is a lookup rather than a scan."""
        self._state = state
        self._by_id: dict[str, Artifact] = {item.artifact_id: item for item in state.artifacts}
        self.stopped = state.status in _FEATURE_STOPPED
        self.cancelled = state.status in _FEATURE_CANCELLED
        self.execution_mode = state.execution_mode

    def latest(self, kind: type[Artifact]) -> Any:
        """Return the newest artifact of one type, which is the current one for that stage."""
        found = None
        for item in self._state.artifacts:
            if isinstance(item, kind):
                found = item
        return found

    def all_of(self, kind: type[Artifact]) -> list[Any]:
        """Return every artifact of one type, in the order the workflow recorded them."""
        return [item for item in self._state.artifacts if isinstance(item, kind)]

    def artifact(self, artifact_id: str | None) -> Artifact | None:
        """Resolve an artifact reference held on another record."""
        return self._by_id.get(artifact_id) if artifact_id else None

    def results_for(self, repository_id: str) -> list[ChildWorkflowResultArtifact]:
        """Return one repository's per-attempt results, oldest attempt first."""
        results = [
            item
            for item in self.all_of(ChildWorkflowResultArtifact)
            if item.repository_id == repository_id
        ]
        return sorted(results, key=lambda item: _result_attempt(item))


def _result_attempt(result: ChildWorkflowResultArtifact) -> int:
    """Return the zero-based attempt a result artifact describes."""
    recorded = result.metadata.get("child_retry_count")
    return recorded if isinstance(recorded, int) and not isinstance(recorded, bool) else 0


def _attempt_meaning(attempt: int, limit: int | None) -> str:
    """State what the attempt ratio counts, in the platform's own terms."""
    if limit is None:
        return f"Repository attempt {attempt}."
    return (
        f"Repository attempt {attempt} of a maximum of {limit} review cycles, which is the "
        "budget this feature is being run under."
    )


def _retry_meaning(attempt: int, limit: int | None) -> str:
    """State what a retry ratio counts, which is not the same sentence as an attempt's."""
    if limit is None:
        return f"Remediation attempt {attempt}."
    return (
        f"Remediation attempt {attempt} of a maximum of {limit}. Configured capacity is not "
        "permission to retry: an attempt runs only where the retry policy allowed one."
    )


def _blocking_summary(result: ChildWorkflowResultArtifact) -> str | None:
    """Return the first blocking issue, which is what stopped this attempt."""
    return next((_text(item) for item in result.blocking_issues if _text(item)), None)


def _review_failure(review: ReviewArtifact | None) -> tuple[str | None, str | None]:
    """Return the highest-severity blocking finding's title and severity.

    Read from the reviewer's structured findings rather than from its prose summary, and
    only the blocking severities: a low-severity remark recorded for the plan's author is not
    why the attempt was sent back.
    """
    if review is None:
        return None, None
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    blocking = [item for item in review.findings if item.severity in {"critical", "high"}]
    considered = blocking or list(review.findings)
    if not considered:
        return None, None
    worst = min(considered, key=lambda item: order.get(item.severity, 5))
    return worst.title, worst.severity


def _validation_counts(
    results: Sequence[Mapping[str, Any]],
) -> tuple[int, int, list[str], int | None, float | None, str | None]:
    """Reduce one attempt's validation results to what an edge and its drawer show."""
    total = len(results)
    passed = sum(1 for item in results if item.get("passed") is True)
    failed = next((item for item in results if item.get("passed") is not True), None)
    command: list[str] = []
    exit_code: int | None = None
    failed_type: str | None = None
    if failed is not None:
        raw = failed.get("command")
        if isinstance(raw, list):
            command = [str(item) for item in raw]
        elif isinstance(raw, str) and raw:
            command = raw.split()
        candidate = failed.get("exit_code")
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            exit_code = candidate
        # The gate that failed -- build, test, lint -- as the failing result itself recorded
        # it, and belonging to the same result the command and exit code above describe.
        failed_type = _text(failed.get("validation_type"))
    durations = [
        float(item["duration_seconds"])
        for item in results
        if isinstance(item.get("duration_seconds"), (int, float))
        and not isinstance(item.get("duration_seconds"), bool)
    ]
    return passed, total, command, exit_code, (sum(durations) if durations else None), failed_type


def _attempt_validation(result: ChildWorkflowResultArtifact) -> list[Mapping[str, Any]]:
    """Return the validation results belonging to one attempt.

    ``current_validation_results`` is the revision-bound set the attempt was actually judged
    on; ``validation_results`` is the typed handoff list. The first is preferred because it is
    the one the retry policy read.
    """
    if result.current_validation_results:
        return [item for item in result.current_validation_results if isinstance(item, Mapping)]
    return [item.model_dump(mode="json") for item in result.validation_results]


# --------------------------------------------------------------------------- the derivation


def feature_executions(
    state: FeatureWorkflowSnapshot,
    *,
    operations: Sequence[ExternalOperation] = (),
) -> list[ExecutionRecord]:
    """Return every execution transition this feature's durable state can account for.

    One pass over one already-loaded snapshot. There is no per-edge read and no per-attempt
    read: a client asks for this once and gets the whole graph's execution history, because
    the alternative -- a request per arrow -- is how a graph with five repositories on their
    fourth attempt would make watching a feature more expensive than running it.

    ``operations`` are this feature's journal rows, where the caller has them loaded. Only
    the pre-coding model calls are rendered from them: the artifact rows above already
    account for everything a stage persisted, but AB-Feature-173's forty-one dark minutes
    were spent entirely *between* artifacts, and the journal rows are the only record with
    a start time and a heartbeat while a call is still in flight.
    """
    reader = _Reader(state)
    records: list[ExecutionRecord] = []
    records.extend(_parent_stage_executions(state, reader))
    records.extend(_planning_call_executions(state, operations))
    for spec in sorted(state.repository_specs, key=lambda item: item.repository_id):
        child = state.child_workflows.get(spec.repository_id)
        records.extend(
            _repository_executions(state, reader, repository_id=spec.repository_id, child=child)
        )
    records.extend(_convergence_executions(state, reader))
    return records


# The journaled pre-coding model calls, and where each sits in the stage vocabulary. Every
# other operation type is deliberately not rendered here: child-loop operations are already
# accounted for by their attempt artifacts, and duplicating them would say the same thing
# twice with two statuses.
_PLANNING_CALL_STAGES: dict[ExternalOperationType, tuple[ExecutionStage, ExecutionStage, str]] = {
    # Before the product manager, because the design is part of the request. A citation-free
    # feature journals no such row, so it renders exactly the stages it always did.
    ExternalOperationType.FETCH_DESIGN_REFERENCE: (
        ExecutionStage.REQUEST,
        ExecutionStage.DESIGN_SNAPSHOT,
        "Design resolver",
    ),
    ExternalOperationType.RUN_PRODUCT_MANAGER: (
        ExecutionStage.REQUEST,
        ExecutionStage.TECHNICAL_PRD,
        "Product manager",
    ),
    ExternalOperationType.RUN_REPOSITORY_RECON: (
        ExecutionStage.TECHNICAL_PRD,
        ExecutionStage.INTEGRATION_CONTRACT,
        "Repository reconnaissance",
    ),
    ExternalOperationType.RUN_CLARIFICATION_GROUNDING: (
        ExecutionStage.TECHNICAL_PRD,
        ExecutionStage.INTEGRATION_CONTRACT,
        "Clarification grounding",
    ),
    ExternalOperationType.RUN_FEATURE_PLANNER: (
        ExecutionStage.TECHNICAL_PRD,
        ExecutionStage.INTEGRATION_CONTRACT,
        "Technical planner",
    ),
}

# What a caller loading journal rows for this read model should filter on.
PLANNING_CALL_OPERATION_TYPES: tuple[ExternalOperationType, ...] = tuple(_PLANNING_CALL_STAGES)

_OPERATION_EXECUTION_STATUS: dict[ExternalOperationStatus, ExecutionStatus] = {
    ExternalOperationStatus.PENDING: ExecutionStatus.QUEUED,
    ExternalOperationStatus.STARTING: ExecutionStatus.QUEUED,
    ExternalOperationStatus.RUNNING: ExecutionStatus.RUNNING,
    ExternalOperationStatus.CANCELLATION_REQUESTED: ExecutionStatus.RUNNING,
    ExternalOperationStatus.CANCELLED: ExecutionStatus.CANCELLED,
    ExternalOperationStatus.SUCCEEDED: ExecutionStatus.COMPLETED,
    ExternalOperationStatus.FAILED_RETRYABLE: ExecutionStatus.FAILED,
    ExternalOperationStatus.FAILED_TERMINAL: ExecutionStatus.FAILED,
    ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE: ExecutionStatus.BLOCKED,
    ExternalOperationStatus.AWAITING_RECONCILIATION: ExecutionStatus.BLOCKED,
}


# Logical steps that share a rendered type above but are not pre-coding calls, so they are not
# this read model's to draw. `design_detail` is one design read per repository, made as that
# repository's workstream starts: it carries `FETCH_DESIGN_REFERENCE` because it is the same
# outbound call to the same provider, and rendering it here would draw one `design_snapshot`
# node per repository -- three for a two-repository feature -- all claiming to be the single
# resolution that happens before the request is analysed.
#
# Excluded rather than given a stage of its own, on this module's own stated rule: per-repository
# operations are already accounted for by their attempt artifacts, and drawing them here would
# say the same thing twice with two statuses. The journal row still exists, which is what it is
# for -- a hung design read is a widening started_at->heartbeat_at gap either way.
_NON_PLANNING_LOGICAL_STEPS = frozenset({"design_detail"})


def _planning_call_executions(
    state: FeatureWorkflowSnapshot, operations: Sequence[ExternalOperation]
) -> list[ExecutionRecord]:
    """One record per journaled pre-coding model call, timestamps and heartbeat included."""
    records: list[ExecutionRecord] = []
    for operation in operations:
        stages = _PLANNING_CALL_STAGES.get(operation.operation_type)
        if stages is None:
            continue
        if (operation.safe_metadata or {}).get("logical_step") in _NON_PLANNING_LOGICAL_STEPS:
            continue
        from_stage, to_stage, handler = stages
        duration: float | None = None
        if operation.started_at is not None and operation.completed_at is not None:
            duration = max(0.0, (operation.completed_at - operation.started_at).total_seconds())
        metadata = operation.safe_metadata or {}
        stage_name = metadata.get("logical_step")
        # The model that answered this call, from the row's own result payload -- written by
        # the adapter at response time, in the same `execution` block artifacts carry, so one
        # reader serves both. Never reconstructed from configuration: a row that recorded
        # nothing (an older row, a mock run, a deterministic composition) stays unresolved and
        # the client says so.
        facts = _model_facts(operation.result_payload or {})
        records.append(
            ExecutionRecord(
                execution_id=f"planning_call:{operation.operation_id}",
                from_stage=from_stage,
                to_stage=to_stage,
                repository_id=operation.repository_id,
                handler_type=ExecutionHandlerType.MODEL,
                handler=handler,
                agent_type=stage_name if isinstance(stage_name, str) else handler,
                provider=facts.provider,
                model=facts.model,
                reasoning_effort=facts.reasoning_effort,
                model_role=facts.model_role,
                model_variable=facts.model_variable,
                model_resolved=facts.model is not None,
                routing_reason=facts.routing_reason,
                status=_OPERATION_EXECUTION_STATUS.get(operation.status, ExecutionStatus.RUNNING),
                attempt=operation.attempt or None,
                max_attempts=operation.max_attempts,
                failure_classification=operation.error_code,
                failure_summary=operation.error_message,
                input_fingerprint=operation.input_fingerprint,
                started_at=operation.started_at,
                heartbeat_at=operation.heartbeat_at,
                completed_at=operation.completed_at,
                duration_seconds=duration,
                execution_mode=state.execution_mode,
            )
        )
    return records


def _parent_stage_executions(
    state: FeatureWorkflowSnapshot, reader: _Reader
) -> list[ExecutionRecord]:
    """The stages before the fan-out: the technical PRD, the contract and the plan."""
    records: list[ExecutionRecord] = []
    technical_prd = reader.latest(TechnicalPRDArtifact)
    contract = reader.latest(IntegrationContractArtifact)
    plan = reader.latest(RepositoryExecutionPlanArtifact)

    records.append(
        _model_stage(
            reader,
            execution_id=(
                f"technical_prd:{technical_prd.artifact_id if technical_prd else 'pending'}"
            ),
            from_stage=ExecutionStage.REQUEST,
            to_stage=ExecutionStage.TECHNICAL_PRD,
            handler="Product manager",
            agent_type="Product manager",
            artifact=technical_prd,
            # Accepted and durably queued is not running: nothing has picked it up.
            reached=state.status is not FeatureWorkflowStatus.PENDING,
        )
    )

    clarification = _clarification_gate(state, technical_prd)
    records.append(
        clarification
        if clarification is not None
        else _model_stage(
            reader,
            execution_id=f"integration_contract:{contract.artifact_id if contract else 'pending'}",
            from_stage=ExecutionStage.TECHNICAL_PRD,
            to_stage=ExecutionStage.INTEGRATION_CONTRACT,
            handler="Technical planner",
            agent_type="Technical planner",
            artifact=contract,
            reached=technical_prd is not None,
            contract_version=contract.contract_version if contract is not None else None,
        )
    )
    records.append(
        _model_stage(
            reader,
            execution_id=f"execution_plan:{plan.artifact_id if plan else 'pending'}",
            from_stage=ExecutionStage.INTEGRATION_CONTRACT,
            to_stage=ExecutionStage.EXECUTION_PLAN,
            handler="Technical planner",
            agent_type="Technical planner",
            artifact=plan,
            reached=contract is not None,
        )
    )
    return records


def _clarification_gate(
    state: FeatureWorkflowSnapshot, technical_prd: TechnicalPRDArtifact | None
) -> ExecutionRecord | None:
    """The transition out of the product manager, where a person is holding it.

    Emitted only where both halves of the record say so: the feature is waiting for a human
    and the current technical PRD carries unresolved questions. No model is named, because
    answering a question is not something a model did.
    """
    if state.status is not FeatureWorkflowStatus.WAITING_FOR_HUMAN or technical_prd is None:
        return None
    questions = technical_prd.unresolved_questions
    if not questions:
        return None
    required = sum(1 for item in questions if item.required)
    return ExecutionRecord(
        execution_id=f"human:clarification:{technical_prd.artifact_id}",
        from_stage=ExecutionStage.TECHNICAL_PRD,
        to_stage=ExecutionStage.INTEGRATION_CONTRACT,
        handler_type=ExecutionHandlerType.HUMAN,
        handler=_HUMAN,
        status=ExecutionStatus.NEEDS_HUMAN,
        attempt=state.clarification_rounds + 1,
        max_attempts=state.max_clarification_rounds,
        attempt_meaning=(
            f"Clarification round {state.clarification_rounds + 1} of a maximum of "
            f"{state.max_clarification_rounds}."
        ),
        human_action=HumanAction.CLARIFICATION,
        human_requirement=(
            f"{len(questions)} question(s) about this feature must be answered before planning "
            f"can continue; {required} of them are required."
        ),
        artifact_id=technical_prd.artifact_id,
        execution_mode=state.execution_mode,
    )


def _model_stage(
    reader: _Reader,
    *,
    execution_id: str,
    from_stage: ExecutionStage,
    to_stage: ExecutionStage,
    handler: str,
    agent_type: str,
    artifact: Artifact | None,
    reached: bool,
    contract_version: str | None = None,
) -> ExecutionRecord:
    """One model-backed parent stage, from the artifact it produced."""
    facts = _model_facts(artifact.metadata) if artifact is not None else _ModelFacts()
    if artifact is not None:
        status = ExecutionStatus.COMPLETED
    elif reader.cancelled:
        status = ExecutionStatus.CANCELLED
    elif not reached:
        status = ExecutionStatus.PENDING
    elif reader.stopped:
        status = ExecutionStatus.FAILED
    else:
        status = ExecutionStatus.RUNNING
    handler_type, handler_name = _agent_handler(reader, facts, agent=facts.agent_type or handler)
    return ExecutionRecord(
        execution_id=execution_id,
        from_stage=from_stage,
        to_stage=to_stage,
        handler_type=handler_type,
        handler=handler_name,
        agent_type=facts.agent_type or agent_type,
        provider=facts.provider,
        model=facts.model,
        reasoning_effort=facts.reasoning_effort,
        model_role=facts.model_role,
        model_variable=facts.model_variable,
        model_resolved=facts.model is not None,
        routing_reason=facts.routing_reason,
        status=status,
        artifact_id=artifact.artifact_id if artifact is not None else None,
        contract_version=contract_version,
        completed_at=artifact.timestamp if artifact is not None else None,
        execution_mode=reader.execution_mode,
    )


# What performed an agent transition in a feature that reaches no provider at all.
_SIMULATION = "Simulation"


def _agent_handler(
    reader: _Reader, facts: _ModelFacts, *, agent: str
) -> tuple[ExecutionHandlerType, str]:
    """Decide whether a model, or nothing at all, performed an agent transition.

    A mock feature reaches no provider. Recording those transitions as model executions with
    an unknown model would put "model" on an arrow no model touched, which is the one thing
    this whole contract exists to prevent -- so a mock run's transitions are named as the
    deterministic simulation they were, and the agent role stays in ``agent_type``.
    """
    if facts.model is not None:
        return ExecutionHandlerType.MODEL, agent
    if reader.execution_mode == "mock":
        return ExecutionHandlerType.DETERMINISTIC, _SIMULATION
    return ExecutionHandlerType.MODEL, agent


def _repository_executions(
    state: FeatureWorkflowSnapshot,
    reader: _Reader,
    *,
    repository_id: str,
    child: ChildWorkflowReference | None,
) -> list[ExecutionRecord]:
    """Every execution inside one repository's lane, one set per attempt."""
    records: list[ExecutionRecord] = [
        _dispatch_execution(state, reader, repository_id=repository_id, child=child)
    ]
    if child is None:
        return records

    repair = _open_repair(reader, repository_id)
    if repair is not None:
        records.append(_repair_execution(state, reader, repository_id=repository_id, repair=repair))

    results = reader.results_for(repository_id)
    limit = state.max_child_review_cycles
    previous: ChildWorkflowResultArtifact | None = None
    for result in results:
        attempt = _result_attempt(result)
        records.extend(
            _attempt_executions(
                state,
                reader,
                repository_id=repository_id,
                result=result,
                previous=previous,
                attempt=attempt,
                limit=limit,
            )
        )
        previous = result

    # The attempt in flight. It has no result artifact yet -- that is written when it ends --
    # so its record comes from the child reference and the routing decision already made for
    # it. This is the only place a *future* model appears, and it appears because the backend
    # has already resolved it, never because a client guessed.
    recorded_attempts = {_result_attempt(item) for item in results}
    if child.retry_count not in recorded_attempts:
        records.extend(
            _live_attempt_executions(
                state,
                reader,
                repository_id=repository_id,
                child=child,
                previous=previous,
                limit=limit,
            )
        )

    # The retry that will not happen, and why. Emitted from the refusal itself rather than from
    # whether an attempt is in flight: the ordinary terminal case is a *completed* final attempt
    # whose successor the reliability logic declined, and keying this off an in-flight attempt
    # left exactly that case with no arrow at all. Added last so it is what the arrow reads,
    # since it is the most recent thing that happened on that transition.
    refused = _text(child.retry_refusal_reason)
    if previous is not None and refused is not None and _workstream_stopped(reader, child):
        records.append(
            _refusal_execution(
                state,
                repository_id=repository_id,
                child=child,
                previous=previous,
                previous_review=_previous_review(reader, previous),
                # Attempts 0 through `retry_count` inclusive have been spent, and the attempt
                # after them is the one that will not run.
                attempts_spent=child.retry_count + 1,
                limit=limit,
                refused=refused,
            )
        )

    if child.status is ChildWorkflowStatus.WAITING_FOR_CONTRACT_CHANGE:
        records.append(
            ExecutionRecord(
                execution_id=f"human:contract_change:{repository_id}",
                from_stage=ExecutionStage.REVIEW,
                to_stage=ExecutionStage.INTEGRATION_REVIEW,
                repository_id=repository_id,
                handler_type=ExecutionHandlerType.HUMAN,
                handler=_HUMAN,
                status=ExecutionStatus.NEEDS_HUMAN,
                human_action=HumanAction.CONTRACT_CHANGE,
                human_requirement=(
                    "This repository cannot implement its scope without changing the shared "
                    "contract, which is immutable. The proposed deviation needs a decision "
                    "before the feature can continue."
                ),
                failure_summary=next(
                    (_text(item) for item in child.blocking_issues if _text(item)), None
                ),
                revision_before=child.current_revision,
                execution_mode=state.execution_mode,
            )
        )
    return records


def _dispatch_execution(
    state: FeatureWorkflowSnapshot,
    reader: _Reader,
    *,
    repository_id: str,
    child: ChildWorkflowReference | None,
) -> ExecutionRecord:
    """The fan-out itself: this orchestrator starting one repository workstream."""
    if child is None:
        status = ExecutionStatus.CANCELLED if reader.cancelled else ExecutionStatus.PENDING
    elif child.status is ChildWorkflowStatus.PENDING:
        status = ExecutionStatus.QUEUED
    elif child.status is ChildWorkflowStatus.BLOCKED:
        status = ExecutionStatus.BLOCKED
    elif child.status is ChildWorkflowStatus.CANCELLED:
        status = ExecutionStatus.CANCELLED
    else:
        status = ExecutionStatus.COMPLETED
    return ExecutionRecord(
        execution_id=f"dispatch:{repository_id}",
        from_stage=ExecutionStage.EXECUTION_PLAN,
        to_stage=ExecutionStage.REPOSITORY,
        repository_id=repository_id,
        handler_type=ExecutionHandlerType.DETERMINISTIC,
        handler=_ORCHESTRATOR,
        status=status,
        human_requirement=(
            "; ".join(child.blocking_issues[:1])
            if child is not None and child.status is ChildWorkflowStatus.BLOCKED
            else None
        ),
        execution_mode=state.execution_mode,
    )


def _open_repair(reader: _Reader, repository_id: str) -> RepositoryRepairProposalArtifact | None:
    """Return the repair this repository is waiting on somebody to decide about."""
    proposals = [
        item
        for item in reader.all_of(RepositoryRepairProposalArtifact)
        if item.repository_id == repository_id
        and item.status == RepositoryRepairStatus.PROPOSED.value
    ]
    return proposals[-1] if proposals else None


def _repair_execution(
    state: FeatureWorkflowSnapshot,
    reader: _Reader,
    *,
    repository_id: str,
    repair: RepositoryRepairProposalArtifact,
) -> ExecutionRecord:
    """The transition into implementation, held by a repository the platform will not change."""
    del reader
    return ExecutionRecord(
        execution_id=f"human:repair:{repair.repair_id}",
        from_stage=ExecutionStage.REPOSITORY,
        to_stage=ExecutionStage.IMPLEMENTATION,
        repository_id=repository_id,
        handler_type=ExecutionHandlerType.HUMAN,
        handler=_HUMAN,
        status=ExecutionStatus.NEEDS_HUMAN,
        failure_classification=repair.failure_classification,
        failure_summary=repair.detected_problem,
        remediation_summary=repair.proposed_repair,
        repair_id=repair.repair_id,
        human_action=HumanAction.REPOSITORY_REPAIR,
        human_requirement=(
            "This repository cannot run its own checks on an untouched checkout. A repair has "
            "been proposed and will not be applied without an explicit decision."
        ),
        revision_before=repair.proposed_at_revision,
        artifact_id=repair.artifact_id,
        completed_at=repair.timestamp,
        execution_mode=state.execution_mode,
    )


def _attempt_executions(
    state: FeatureWorkflowSnapshot,
    reader: _Reader,
    *,
    repository_id: str,
    result: ChildWorkflowResultArtifact,
    previous: ChildWorkflowResultArtifact | None,
    attempt: int,
    limit: int | None,
) -> list[ExecutionRecord]:
    """The three executions of one completed attempt, plus the retry edge that started it."""
    completion = reader.artifact(result.code_completion_artifact_id)
    review = reader.artifact(result.review_artifact_id)
    completion_artifact = completion if isinstance(completion, CodeCompletionArtifact) else None
    review_artifact = review if isinstance(review, ReviewArtifact) else None
    routing = _mapping(result.model_routing)
    number = attempt + 1
    records: list[ExecutionRecord] = []

    if attempt > 0 and previous is not None:
        records.append(
            _retry_execution(
                state,
                reader,
                repository_id=repository_id,
                previous=previous,
                previous_review=_previous_review(reader, previous),
                attempt=number,
                limit=limit,
                routing=routing,
                completion=completion_artifact,
                status=ExecutionStatus.COMPLETED,
                revision_after=result.current_revision,
                # This attempt's own stamp, not the child's: the child has moved on, and
                # reading it here would draw every past retry from whichever authority
                # demanded the latest one.
                kind=_attempt_kind(result),
            )
        )

    facts = _merged(
        _model_facts(completion_artifact.metadata)
        if completion_artifact is not None
        else _ModelFacts(),
        _routing_facts(routing),
    )
    handler_type, handler_name = _agent_handler(reader, facts, agent="Engineer")
    records.append(
        ExecutionRecord(
            execution_id=f"implementation:{repository_id}:{attempt}",
            from_stage=ExecutionStage.REPOSITORY,
            to_stage=ExecutionStage.IMPLEMENTATION,
            repository_id=repository_id,
            handler_type=handler_type,
            handler=handler_name,
            agent_type="Engineer",
            provider=facts.provider,
            model=facts.model,
            reasoning_effort=facts.reasoning_effort,
            model_role=facts.model_role,
            model_variable=facts.model_variable,
            model_resolved=facts.model is not None,
            status=(
                ExecutionStatus.COMPLETED
                if completion_artifact is not None
                else ExecutionStatus.FAILED
            ),
            attempt=number,
            max_attempts=limit,
            attempt_meaning=_attempt_meaning(number, limit),
            routing_reason=facts.routing_reason or _text(routing.get("routing_reason")),
            input_fingerprint=_text(routing.get("input_fingerprint")),
            revision_before=previous.current_revision if previous is not None else None,
            revision_after=result.current_revision,
            contract_version=_text(result.metadata.get("contract_version")),
            artifact_id=result.code_completion_artifact_id,
            code_completion_artifact_id=result.code_completion_artifact_id,
            result_artifact_id=result.artifact_id,
            completed_at=(
                completion_artifact.timestamp
                if completion_artifact is not None
                else result.timestamp
            ),
            execution_mode=state.execution_mode,
        )
    )

    # Only where commands actually ran. An attempt rejected by the completeness gate before
    # validation started produced no subprocess execution, and publishing one -- as "pending",
    # on an attempt that finished weeks ago -- describes work that never happened.
    validation = _attempt_validation(result)
    if validation:
        passed, total, command, exit_code, duration, failed_type = _validation_counts(validation)
        records.append(
            ExecutionRecord(
                execution_id=f"validation:{repository_id}:{attempt}",
                from_stage=ExecutionStage.IMPLEMENTATION,
                to_stage=ExecutionStage.VALIDATION,
                repository_id=repository_id,
                handler_type=ExecutionHandlerType.DETERMINISTIC,
                handler=_VALIDATOR,
                status=(
                    ExecutionStatus.COMPLETED
                    if total > 0 and passed == total
                    else ExecutionStatus.FAILED
                    if total > 0
                    else ExecutionStatus.PENDING
                ),
                # On a deterministic validation record the failure class is the gate that
                # failed -- the failing result's own ``validation_type`` -- not the retry
                # policy's coarser classification, which belongs to the remediation records
                # that consume it. Only a failed run states one.
                failure_classification=failed_type if total > 0 and passed < total else None,
                attempt=number,
                max_attempts=limit,
                attempt_meaning=_attempt_meaning(number, limit),
                validation_passed=passed,
                validation_total=total,
                command=command,
                exit_code=exit_code,
                revision_after=result.current_revision,
                result_artifact_id=result.artifact_id,
                duration_seconds=duration,
                execution_mode=state.execution_mode,
            )
        )

    if review_artifact is not None:
        review_facts = _model_facts(review_artifact.metadata)
        title, severity = _review_failure(review_artifact)
        review_handler_type, review_handler = _agent_handler(
            reader, review_facts, agent="Code reviewer"
        )
        records.append(
            ExecutionRecord(
                execution_id=f"review:{repository_id}:{attempt}",
                from_stage=ExecutionStage.VALIDATION,
                to_stage=ExecutionStage.REVIEW,
                repository_id=repository_id,
                handler_type=review_handler_type,
                handler=review_handler,
                agent_type="Code reviewer",
                provider=review_facts.provider,
                model=review_facts.model,
                reasoning_effort=review_facts.reasoning_effort,
                model_role=review_facts.model_role,
                model_variable=review_facts.model_variable,
                model_resolved=review_facts.model is not None,
                routing_reason=review_facts.routing_reason,
                status=ExecutionStatus.COMPLETED,
                attempt=number,
                max_attempts=limit,
                attempt_meaning=_attempt_meaning(number, limit),
                review_verdict=review_artifact.verdict,
                review_finding_count=len(review_artifact.findings),
                review_findings_blocking=(
                    result.review_finding_counts.findings_blocking
                    if result.review_finding_counts is not None
                    else None
                ),
                review_findings_advisory=(
                    result.review_finding_counts.findings_advisory
                    if result.review_finding_counts is not None
                    else None
                ),
                review_findings_untraceable=(
                    result.review_finding_counts.findings_untraceable
                    if result.review_finding_counts is not None
                    else None
                ),
                failure_summary=title if review_artifact.verdict != "approved" else None,
                failure_severity=severity if review_artifact.verdict != "approved" else None,
                revision_after=result.current_revision,
                artifact_id=review_artifact.artifact_id,
                review_artifact_id=review_artifact.artifact_id,
                result_artifact_id=result.artifact_id,
                completed_at=review_artifact.timestamp,
                execution_mode=state.execution_mode,
            )
        )

    return records


def _remediation_origin(kind: str | None) -> tuple[ExecutionStage, bool]:
    """Which node a retry's arrow starts at, from the one stamp that records it.

    Integration remediation starts at the integration review, because that is the authority
    that demanded the rework. Everything else starts at the repository's own review, which is
    where every retry used to be drawn from -- including the remediations, which is the defect.

    ``None`` is a workstream retry and not "unknown". Every attempt recorded before the stamp
    existed reads that way, so an old record keeps attaching to the lane loop exactly as it
    does today rather than vanishing from the graph or drawing twice.
    """
    remediation = kind == str(TargetedAttempt.INTEGRATION_REMEDIATION)
    origin = ExecutionStage.INTEGRATION_REVIEW if remediation else ExecutionStage.REVIEW
    return origin, remediation


def _attempt_kind(result: ChildWorkflowResultArtifact) -> str | None:
    """The kind stamped onto one attempt's own immutable record, where it carries one."""
    return _text(result.metadata.get("targeted_attempt_kind"))


def _previous_review(
    reader: _Reader, previous: ChildWorkflowResultArtifact
) -> ReviewArtifact | None:
    """Return the review that rejected the attempt before this one."""
    candidate = reader.artifact(previous.review_artifact_id)
    return candidate if isinstance(candidate, ReviewArtifact) else None


def _retry_execution(
    state: FeatureWorkflowSnapshot,
    reader: _Reader,
    *,
    repository_id: str,
    previous: ChildWorkflowResultArtifact,
    previous_review: ReviewArtifact | None,
    attempt: int,
    limit: int | None,
    routing: Mapping[str, Any],
    completion: CodeCompletionArtifact | None,
    status: ExecutionStatus,
    revision_after: str | None,
    kind: str | None,
) -> ExecutionRecord:
    """One remediation transition: review sending a repository back to the engineer.

    The interesting half is not the arrow, it is the pair of sentences on it. *Why the previous
    attempt failed* comes from that attempt's own review finding or blocking issue. *What the
    next attempt has to change* comes from the platform's retry plan, which is a different
    answer -- a repository whose lint configuration is broken failed as a lint error and is
    remediated by repairing the repository, not by writing the file again.
    """
    plan = _mapping(previous.retry_strategy)
    title, severity = _review_failure(previous_review)
    facts = _merged(
        _model_facts(completion.metadata) if completion is not None else _ModelFacts(),
        _routing_facts(routing),
    )
    classification = _text(previous.failure_classification)
    counter_label, counter_value, counter_limit = _counter(state, previous, repository_id)
    handler_type, handler_name = _agent_handler(reader, facts, agent="Engineer")
    from_stage, _ = _remediation_origin(kind)
    return ExecutionRecord(
        execution_id=f"retry:{repository_id}:{attempt - 1}",
        from_stage=from_stage,
        to_stage=ExecutionStage.IMPLEMENTATION,
        repository_id=repository_id,
        is_retry=True,
        handler_type=handler_type,
        handler=handler_name,
        agent_type="Engineer",
        provider=facts.provider,
        model=facts.model,
        reasoning_effort=facts.reasoning_effort,
        model_role=facts.model_role,
        model_variable=facts.model_variable,
        model_resolved=facts.model is not None,
        status=status,
        attempt=attempt,
        max_attempts=limit,
        attempt_meaning=_retry_meaning(attempt, limit),
        counter_label=counter_label,
        counter_value=counter_value,
        counter_limit=counter_limit,
        failure_classification=classification,
        failure_summary=title or _blocking_summary(previous),
        failure_severity=severity,
        remediation_summary=_text(plan.get("required_strategy_change")),
        routing_reason=facts.routing_reason or _text(routing.get("routing_reason")),
        input_fingerprint=_text(routing.get("input_fingerprint")),
        review_verdict=previous_review.verdict if previous_review is not None else None,
        review_finding_count=(
            len(previous_review.findings) if previous_review is not None else None
        ),
        revision_before=previous.current_revision,
        revision_after=revision_after,
        review_artifact_id=previous.review_artifact_id,
        result_artifact_id=previous.artifact_id,
        previous_execution_id=f"review:{repository_id}:{_result_attempt(previous)}",
        completed_at=completion.timestamp if completion is not None else None,
        execution_mode=state.execution_mode,
    )


_COUNTER_LABELS = {
    "implementation_retry_count": "Implementation retries",
    "validation_retry_count": "Validation retries",
    "repository_setup_retry_count": "Repository setup retries",
    "integration_retry_count": "Integration retries",
}


def _counter(
    state: FeatureWorkflowSnapshot,
    result: ChildWorkflowResultArtifact,
    repository_id: str,
) -> tuple[str | None, int | None, int | None]:
    """Return the classification budget this retry was granted against, where one is named.

    Separate from the attempt ratio because the platform enforces both, against different
    limits. Reporting one as though it were the other is how "2 / 8" came to mean nothing in
    particular.
    """
    classification = _text(result.failure_classification)
    if classification is None:
        return None, None, None
    field = {
        "implementation_missing": "implementation_retry_count",
        "validation_source_failure": "validation_retry_count",
        "validation_capacity_failure": "validation_retry_count",
        "validation_configuration_failure": "repository_setup_retry_count",
        "dependency_installation_failure": "repository_setup_retry_count",
        "test_infrastructure_missing": "repository_setup_retry_count",
        "review_scope_failure": "implementation_retry_count",
        "contract_mismatch": "integration_retry_count",
    }.get(classification)
    if field is None:
        return None, None, None
    recorded = result.metadata.get(field)
    value = recorded if isinstance(recorded, int) and not isinstance(recorded, bool) else None
    child = state.child_workflows.get(repository_id)
    grant = child.granted_extra_attempts if child is not None else 0
    if field == "integration_retry_count":
        # The number this attempt is actually refused against, which is not the one this
        # function used to report. `_integration_remediation_decision` calls
        # `decide_child_retry` with `budget=state.max_integration_review_cycles` and no grant
        # -- a grant buys workstream attempts, not the parent's review cycles -- so printing
        # `max_child_review_cycles + grant` beside `integration_retry_count` put two
        # different numbers on one arrow.
        return _COUNTER_LABELS[field], value, state.max_integration_review_cycles
    limits = {
        "implementation_retry_count": state.max_implementation_retries,
        "validation_retry_count": state.max_validation_retries,
        "repository_setup_retry_count": state.max_repository_setup_retries,
    }
    return _COUNTER_LABELS[field], value, limits[field] + grant


def _live_attempt_executions(
    state: FeatureWorkflowSnapshot,
    reader: _Reader,
    *,
    repository_id: str,
    child: ChildWorkflowReference,
    previous: ChildWorkflowResultArtifact | None,
    limit: int | None,
) -> list[ExecutionRecord]:
    """The attempt currently in flight, queued, refused, or stopped without a result.

    ``model_routing`` here is the decision already taken for *this* attempt, so a queued retry
    can name the model that will run it. It is published only when its own recorded attempt
    number matches this one: a stale decision from a previous attempt is not a prediction about
    the next one, and publishing it as one would be exactly the guess this contract forbids.
    """
    attempt = child.retry_count
    number = attempt + 1
    routing = _mapping(child.model_routing)
    resolved_for_this_attempt = routing.get("attempt") == attempt
    facts = _routing_facts(routing) if resolved_for_this_attempt else _ModelFacts()
    records: list[ExecutionRecord] = []

    stopped = _workstream_stopped(reader, child)

    if reader.cancelled or child.status is ChildWorkflowStatus.CANCELLED:
        status = ExecutionStatus.CANCELLED
    elif stopped:
        status = ExecutionStatus.FAILED
    elif child.status is ChildWorkflowStatus.WAITING_FOR_CONTRACT_CHANGE:
        status = ExecutionStatus.BLOCKED
    elif child.status is ChildWorkflowStatus.PENDING:
        status = ExecutionStatus.QUEUED
    else:
        status = ExecutionStatus.RUNNING

    # The transition into this attempt, where this attempt is a retry. Emitted alongside the
    # implementation record below rather than instead of it, so an in-flight attempt has the
    # same pair of arrows a completed one does.
    if attempt > 0 and previous is not None:
        records.append(
            _retry_execution(
                state,
                reader,
                repository_id=repository_id,
                previous=previous,
                previous_review=_previous_review(reader, previous),
                attempt=number,
                limit=limit,
                routing=routing if resolved_for_this_attempt else {},
                completion=None,
                status=status,
                revision_after=None,
                # The attempt in flight is the one the child's stamp describes.
                kind=str(child.targeted_attempt_kind) if child.targeted_attempt_kind else None,
            )
        )

    records.append(
        ExecutionRecord(
            execution_id=f"implementation:{repository_id}:{attempt}",
            from_stage=ExecutionStage.REPOSITORY,
            to_stage=ExecutionStage.IMPLEMENTATION,
            repository_id=repository_id,
            handler_type=ExecutionHandlerType.MODEL,
            handler="Engineer",
            agent_type="Engineer",
            model=facts.model,
            reasoning_effort=facts.reasoning_effort,
            model_role=facts.model_role,
            model_resolved=facts.model is not None,
            status=status,
            attempt=number,
            max_attempts=limit,
            attempt_meaning=_attempt_meaning(number, limit),
            routing_reason=(
                _text(routing.get("routing_reason")) if resolved_for_this_attempt else None
            ),
            revision_before=child.current_revision,
            execution_mode=state.execution_mode,
        )
    )
    return records


def _workstream_stopped(reader: _Reader, child: ChildWorkflowReference) -> bool:
    """Whether this repository has stopped, by its own status or by the feature's."""
    return child.status in _CHILD_STOPPED or reader.stopped


def _refusal_execution(
    state: FeatureWorkflowSnapshot,
    *,
    repository_id: str,
    child: ChildWorkflowReference,
    previous: ChildWorkflowResultArtifact,
    previous_review: ReviewArtifact | None,
    attempts_spent: int,
    limit: int | None,
    refused: str,
) -> ExecutionRecord:
    """The retry that will not happen, and the platform's own reason for refusing it.

    Its origin is derived exactly as a granted retry's is. A refusal recorded while the child
    was in integration remediation is a refusal of the integration review's request, and
    drawing it from the lane's own reviewer misattributes it the same way.
    """
    title, severity = _review_failure(previous_review)
    facts = _routing_facts(_mapping(previous.model_routing))
    from_stage, _ = _remediation_origin(
        str(child.targeted_attempt_kind) if child.targeted_attempt_kind else None
    )
    return ExecutionRecord(
        execution_id=f"retry-refused:{repository_id}:{attempts_spent}",
        from_stage=from_stage,
        to_stage=ExecutionStage.IMPLEMENTATION,
        repository_id=repository_id,
        is_retry=True,
        handler_type=ExecutionHandlerType.HUMAN,
        handler=_HUMAN,
        status=ExecutionStatus.NEEDS_HUMAN,
        attempt=attempts_spent,
        max_attempts=limit,
        attempt_meaning=(
            f"{attempts_spent} attempt(s) were spent"
            + (f" of a maximum of {limit}" if limit is not None else "")
            + ". No further attempt is scheduled: remaining capacity is not permission to "
            "retry."
        ),
        failure_classification=_text(previous.failure_classification),
        failure_summary=title or _blocking_summary(previous),
        failure_severity=severity,
        remediation_summary=_text(
            _mapping(previous.retry_strategy).get("required_strategy_change")
        ),
        human_action=HumanAction.RETRY_REFUSED,
        human_requirement=refused,
        # The model that ran last, so a reader knows what was already tried. Never presented as
        # the model of a next attempt, because there is no next attempt.
        model=facts.model,
        reasoning_effort=facts.reasoning_effort,
        model_role=facts.model_role,
        model_resolved=False,
        review_verdict=previous_review.verdict if previous_review is not None else None,
        review_finding_count=len(previous_review.findings) if previous_review is not None else None,
        revision_before=previous.current_revision,
        review_artifact_id=previous.review_artifact_id,
        result_artifact_id=previous.artifact_id,
        previous_execution_id=f"review:{repository_id}:{_result_attempt(previous)}",
        execution_mode=state.execution_mode,
    )


def _convergence_executions(
    state: FeatureWorkflowSnapshot, reader: _Reader
) -> list[ExecutionRecord]:
    """Integration review, and the pull requests it releases."""
    records: list[ExecutionRecord] = []
    reviews = reader.all_of(IntegrationReviewArtifact)
    integration = reviews[-1] if reviews else None
    children = state.child_workflows.values()
    all_done = bool(children) and all(item.status in _CHILD_DONE for item in children)
    facts = _model_facts(integration.metadata) if integration is not None else _ModelFacts()
    if integration is not None:
        status = ExecutionStatus.COMPLETED
    elif reader.cancelled:
        status = ExecutionStatus.CANCELLED
    elif not all_done:
        status = ExecutionStatus.PENDING
    elif reader.stopped:
        status = ExecutionStatus.FAILED
    else:
        status = ExecutionStatus.RUNNING
    cycles = state.integration_review_cycles
    # The integration review is two things: a deterministic conformance check against the
    # approved contract, and -- where the changed source fit its budget -- a model reading the
    # seams between repositories. Only the second involves a model, and its artifact records
    # which. Where it did not run, the transition was the contract validator's, and saying
    # "model, not recorded" about it would describe an execution that never happened.
    if integration is not None and facts.model is None:
        validator = _text(integration.metadata.get("contract_validator"))
        handler_type = ExecutionHandlerType.DETERMINISTIC
        handler_name = "Contract validator" if validator else _ORCHESTRATOR
    else:
        handler_type, handler_name = _agent_handler(reader, facts, agent="Integration reviewer")
    records.append(
        ExecutionRecord(
            execution_id=(
                f"integration_review:{integration.artifact_id if integration else 'pending'}"
            ),
            from_stage=ExecutionStage.REVIEW,
            to_stage=ExecutionStage.INTEGRATION_REVIEW,
            handler_type=handler_type,
            handler=handler_name,
            agent_type="Integration reviewer",
            provider=facts.provider,
            model=facts.model,
            reasoning_effort=facts.reasoning_effort,
            model_role=facts.model_role,
            model_variable=facts.model_variable,
            model_resolved=facts.model is not None,
            routing_reason=facts.routing_reason,
            status=status,
            attempt=max(cycles, 1),
            max_attempts=state.max_integration_review_cycles,
            attempt_meaning=(
                f"Integration review cycle {max(cycles, 1)} of a maximum of "
                f"{state.max_integration_review_cycles}."
            ),
            review_verdict=integration.review_status if integration is not None else None,
            review_finding_count=(
                len(integration.cross_repository_findings) if integration is not None else None
            ),
            contract_version=_text(integration.metadata.get("contract_version"))
            if integration is not None
            else None,
            artifact_id=integration.artifact_id if integration is not None else None,
            completed_at=integration.timestamp if integration is not None else None,
            execution_mode=state.execution_mode,
        )
    )

    pull_requests = reader.all_of(PullRequestArtifact)
    if pull_requests:
        pr_status = ExecutionStatus.COMPLETED
    elif reader.cancelled:
        pr_status = ExecutionStatus.CANCELLED
    elif integration is None:
        pr_status = ExecutionStatus.PENDING
    elif reader.stopped:
        pr_status = ExecutionStatus.FAILED
    else:
        pr_status = ExecutionStatus.RUNNING
    records.append(
        ExecutionRecord(
            execution_id="pull_requests",
            from_stage=ExecutionStage.INTEGRATION_REVIEW,
            to_stage=ExecutionStage.PULL_REQUESTS,
            handler_type=ExecutionHandlerType.DETERMINISTIC,
            # The commit, the push and the pull request are provider API calls made through the
            # operation journal. Naming a model here would be a fabrication.
            handler=_GITHUB,
            status=pr_status,
            pull_request_artifact_id=pull_requests[-1].artifact_id if pull_requests else None,
            completed_at=pull_requests[-1].timestamp if pull_requests else None,
            execution_mode=state.execution_mode,
        )
    )
    return records


__all__ = [
    "ExecutionHandlerType",
    "ExecutionRecord",
    "ExecutionStage",
    "ExecutionStatus",
    "HumanAction",
    "feature_executions",
]
