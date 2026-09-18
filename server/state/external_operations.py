"""Durable, credential-free records for externally observable workflow operations."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, NamedTuple

from pydantic import Field, JsonValue, field_validator

from state.models import StateModel


class ExternalOperationType(StrEnum):
    """Every live side effect that must be journaled before it begins."""

    CLONE_REPOSITORY = "clone_repository"
    CREATE_BRANCH = "create_branch"
    RUN_CODING_EXECUTOR = "run_coding_executor"
    RUN_REVIEWER = "run_reviewer"
    WRITE_FILE_CHANGES = "write_file_changes"
    RUN_FORMATTER = "run_formatter"
    INSTALL_DEPENDENCIES = "install_dependencies"
    RUN_LINTER = "run_linter"
    RUN_TYPECHECK = "run_typecheck"
    RUN_TESTS = "run_tests"
    RUN_BUILD = "run_build"
    CREATE_COMMIT = "create_commit"
    PUSH_BRANCH = "push_branch"
    CREATE_PULL_REQUEST = "create_pull_request"
    UPDATE_PULL_REQUEST = "update_pull_request"
    # A feature revision closing the pull request it superseded, strictly after its own
    # replacement was created and read back. Its own member rather than a reuse of
    # UPDATE_PULL_REQUEST, which -- despite the name -- is the cross-link comment.
    CLOSE_PULL_REQUEST = "close_pull_request"
    ADD_LABELS = "add_labels"
    ADD_REVIEWERS = "add_reviewers"
    # The feature's pre-coding model calls. Journaled for observability: a hung
    # reconnaissance call is a row with a widening started_at->heartbeat_at gap instead of
    # forty minutes of nothing (AB-Feature-173). These rows record; they are never a
    # recovery input -- the pre-coding stages already resume from their artifacts, so every
    # call writes a fresh row and no result is ever replayed from one.
    RUN_PRODUCT_MANAGER = "run_product_manager"
    RUN_REPOSITORY_RECON = "run_repository_recon"
    RUN_CLARIFICATION_GROUNDING = "run_clarification_grounding"
    RUN_FEATURE_PLANNER = "run_feature_planner"
    # Inside that block, and for that block's reason. A design fetch is a live outbound call,
    # so a hung one has to be a row with a widening started_at->heartbeat_at gap rather than
    # silence -- and *it is never a recovery input*: the resolution step resumes from its
    # artifact like every other pre-coding stage, and no result is ever replayed from a row.
    # A feature that cites no design writes none of these at all.
    FETCH_DESIGN_REFERENCE = "fetch_design_reference"


class ExternalOperationStatus(StrEnum):
    """The durable lifecycle for one externally observable operation."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"
    UNKNOWN_EXTERNAL_STATE = "unknown_external_state"
    # A remote effect this process could not judge, which a credentialed caller still can.
    # Distinct from UNKNOWN_EXTERNAL_STATE, which asserts that nobody can judge it without a
    # person. The credential-free recovery sweep can only ever establish the former, and
    # stamping the latter pre-empted the reconciliation callback that would have settled it.
    AWAITING_RECONCILIATION = "awaiting_reconciliation"


class CompensationStatus(StrEnum):
    """Whether a human needs to account for an intentionally retained side effect."""

    NOT_REQUIRED = "not_required"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    ACKNOWLEDGED = "acknowledged"


class WorkflowCheckpointBoundary(StrEnum):
    """Stable recovery points emitted around live side-effect sequences."""

    BEFORE_CLONE = "before_clone"
    AFTER_CLONE = "after_clone"
    BEFORE_CODING = "before_coding"
    AFTER_CODING = "after_coding"
    BEFORE_VALIDATION = "before_validation"
    AFTER_VALIDATION = "after_validation"
    BEFORE_COMMIT = "before_commit"
    AFTER_COMMIT = "after_commit"
    BEFORE_PUSH = "before_push"
    AFTER_PUSH = "after_push"
    BEFORE_PR = "before_pr"
    AFTER_PR = "after_pr"


class ExternalOperation(StateModel):
    """The durable operation row represented without provider credentials or raw output."""

    operation_id: str = Field(min_length=1)
    workflow_id: str = Field(min_length=1)
    feature_id: str | None = None
    child_workflow_id: str | None = None
    repository_id: str | None = None
    operation_type: ExternalOperationType
    idempotency_key: str = Field(min_length=1)
    status: ExternalOperationStatus
    attempt: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    input_fingerprint: str = Field(min_length=1)
    repository_revision: str | None = None
    command_fingerprint: str | None = None
    is_current: bool = True
    superseded_by_operation_id: str | None = None
    external_reference: str | None = None
    started_at: datetime | None = None
    heartbeat_at: datetime | None = None
    completed_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    result_payload: dict[str, JsonValue] | None = None
    compensation_status: CompensationStatus | None = None
    safe_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("started_at", "heartbeat_at", "completed_at")
    @classmethod
    def timestamps_must_be_aware(cls, value: datetime | None) -> datetime | None:
        """Keep recovery ordering unambiguous across worker processes."""
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            msg = "operation timestamps must be timezone-aware"
            raise ValueError(msg)
        return value


class ExternalOperationAttempt(StateModel):
    """One durable execution attempt for an operation, including coding-provider evidence."""

    attempt_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    provider: str | None = None
    model: str | None = None
    workspace_path: str | None = None
    task_plan_artifact_id: str | None = None
    contract_artifact_id: str | None = None
    process_id: int | None = Field(default=None, ge=1)
    external_run_id: str | None = None
    status: ExternalOperationStatus
    started_at: datetime
    heartbeat_at: datetime | None = None
    completed_at: datetime | None = None
    safe_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ExternalOperationEvent(StateModel):
    """An append-only transition in the operation timeline."""

    event_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    previous_status: ExternalOperationStatus | None = None
    new_status: ExternalOperationStatus
    timestamp: datetime
    attempt: int = Field(ge=0)
    workflow_id: str = Field(min_length=1)
    feature_id: str | None = None
    child_workflow_id: str | None = None
    repository_id: str | None = None
    event_type: str = Field(min_length=1)
    safe_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ManualCleanupRequirement(StateModel):
    """A visible, non-destructive recommendation for a retained external side effect."""

    resource_type: str = Field(min_length=1)
    repository_id: str | None = None
    external_reference: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)


def safe_operation_metadata(value: dict[str, Any]) -> dict[str, JsonValue]:
    """Validate JSON-safe metadata before it can be persisted in an operation journal."""
    return dict(value)


# The one metadata key that names which child attempt an operation belongs to. Distinct from
# `ExternalOperation.attempt`, which counts this operation's own retries -- a coding call on
# its third provider attempt inside the child's first attempt is `attempt=3`,
# `child_attempt=0`, and reading one for the other merges attempts or splits them.
CHILD_ATTEMPT_METADATA_KEY = "child_attempt"

# The metadata key naming which logical step of the workflow journaled the row. Written by the
# executor for every operation, and the only thing that tells two callers of one operation type
# apart -- `apply_coding_updates` and `write_contract_projection` are both `write_file_changes`.
LOGICAL_STEP_METADATA_KEY = "logical_step"

# The payload key naming how many times a row's model call had to be issued again because the
# stream it opened never spoke. A *payload* key, not metadata: it is something the call
# reported about itself, journaled beside the model and the usage counts.
STREAM_REISSUES_PAYLOAD_KEY = "stream_reissues"
# The block a call journaled through `llm_call_record` nests its facts under. Spelled here
# rather than imported for the reason that module spells it itself: both ends stay leaves,
# and the vocabulary is the artifact vocabulary either way.
EXECUTION_PAYLOAD_KEY = "execution"

# The deterministic write the platform makes into the checkout before the Engineer is called at
# all: the approved contract's projection, so a later modification of it becomes a visible
# change request. Named here rather than spelled at each end, because the step name is what
# tells this write apart from the implementation write and a silent rename would put the
# implementation's words back on it.
CONTRACT_PROJECTION_LOGICAL_STEP = "write_contract_projection"

# The checks the platform runs on an untouched checkout, before the Engineer writes anything:
# `_baseline_validation_failures` proving the repository can pass its own required commands,
# and the baseline lint whose findings are not this change's to fix. They run the same
# commands through the same tool as the attempt's own validation, so without a name of their
# own the record cannot tell the two apart -- and 201's backend attempt 0, stopped by the
# self-review gate before it ever validated its change, showed three ticked validation rows
# that had measured the repository. Named here rather than spelled at each end, because the
# writer and the reader have to agree and a silent rename would put the attempt's words back
# on the repository's work.
BASELINE_VALIDATION_LOGICAL_STEP = "baseline_validation"
# What the attempt's own validation is stamped with, for the same reason.
CHANGE_VALIDATION_LOGICAL_STEP = "validation"


# The two ways an operation row ends without a result: it spent its attempts inside one child
# attempt, or it was cancelled. A caller in the *same* child attempt is refused a replay of
# either -- that is what these statuses exist to stop. A caller in a *later* child attempt is
# not repeating anything: it has not tried this work, and it gets its own row (69-).
REPLAY_REFUSED_STATUSES = frozenset(
    {
        ExternalOperationStatus.FAILED_TERMINAL,
        ExternalOperationStatus.CANCELLED,
    }
)


def child_attempt_of(operation: ExternalOperation) -> int | None:
    """Read the child attempt an operation was created under, or ``None`` if it is unstamped.

    ``None`` is the answer for every row written before the stamp existed, and for the rows
    that belong to no child attempt at all -- reconnaissance and the parent's publication
    calls are scoped to the feature, not to an attempt. A caller that cannot say which
    attempt a row belongs to must say so rather than assume the current one.

    Booleans are rejected even though Python counts them as integers: ``true`` is not an
    attempt identity, and comparing it against a counter of ``1`` would silently match.
    """
    value = operation.safe_metadata.get(CHILD_ATTEMPT_METADATA_KEY)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def stream_reissues_of(operation: ExternalOperation) -> int | None:
    """Read how many times this row's model call had to be issued again, or ``None``.

    ``None`` means no measurement exists: a row written before the field, a call whose
    transport was a plain POST, or an operation that made no model call at all. ``0`` means
    a stream that spoke on the first issue, and is deliberately a different answer -- the
    field exists to say whether the first-event budget is close to binding, and a surface
    that cannot tell "never close" from "never measured" cannot answer that.

    Read from ``result_payload`` rather than ``safe_metadata``: this is something the call
    reported about itself on the way out, which is where the adapters already journal the
    model, the provider and the usage counts.

    Booleans are rejected for the reason ``child_attempt_of`` rejects them: ``true`` is not
    a count, and it would otherwise render as one re-issue.
    """
    payload = operation.result_payload or {}
    value = payload.get(STREAM_REISSUES_PAYLOAD_KEY)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        # The execution block is the other shape a journaled call's payload takes -- the
        # planning calls record theirs through `llm_call_record`, which nests it. One reader
        # for both, because "which model answered, and did its stream stall" is one question
        # regardless of which seam journaled it.
        block = payload.get(EXECUTION_PAYLOAD_KEY)
        value = block.get(STREAM_REISSUES_PAYLOAD_KEY) if isinstance(block, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def logical_step_of(operation: ExternalOperation) -> str | None:
    """Read the workflow step an operation was journaled under, or ``None`` if it has none."""
    value = operation.safe_metadata.get(LOGICAL_STEP_METADATA_KEY)
    return value if isinstance(value, str) and value else None


def is_baseline_validation(operation: ExternalOperation) -> bool:
    """Whether this row measured the untouched checkout rather than the attempt's change.

    Read from the step the executor stamped, never inferred from position or timing: the two
    runs use the same commands and the same tool, and "it came before the coding call" is the
    kind of ordering guess the journal exists to replace.

    A row written before the step existed answers ``False``, which is today's attribution.
    Old rows keep it rather than being re-labelled from a heuristic that cannot tell a
    baseline run from a validation of a coding call that wrote nothing.
    """
    step = logical_step_of(operation)
    return step is not None and step.startswith(f"{BASELINE_VALIDATION_LOGICAL_STEP}:")


class OperationRepeat(NamedTuple):
    """Why one row is one of several of its type inside a single attempt.

    Two ``run_linter`` rows in one attempt read as a bug. Both were real: 201's backend
    attempt 3 ran the linter at the revision it started from and again after an in-attempt
    repair changed the tree. The journal already encoded why, in two first-class columns, and
    the drawer rendered neither -- so honest re-runs looked like duplicates.

    ``kind`` is one of three, and ``detail`` is the short form of whichever column
    distinguishes this row. Where no column can distinguish it, the answer is that this is
    another run of the same step -- never a fabricated reason, and never an ordinal. "2nd" is
    wrong on purpose: the response is bounded, so a count over served rows silently changes
    meaning when the window truncates the attempt. Relative wording has no such failure mode.
    """

    kind: str
    detail: str | None


# The same command at a new revision: a re-run after something changed the tree.
REPEAT_NEW_REVISION = "new_revision"
# A different command at the same revision: a scoped test beside the full suite (47-C).
REPEAT_DIFFERENT_COMMAND = "different_command"
# Another run of the same step, with nothing recorded that distinguishes it.
REPEAT_SAME_STEP = "same_step"

# Enough of a hash to tell two apart at a glance, and not enough to read as an identifier a
# person is meant to use. The whole value is on the row's own record either way.
_SHORT = 8


def operation_repeats(
    operations: Sequence[ExternalOperation],
) -> dict[str, OperationRepeat]:
    """Say why each repeated row repeats, keyed by operation id.

    Computed per attempt and per operation type, from ``repository_revision`` and
    ``command_fingerprint`` -- the two facts this needs, both already first-class columns on
    the operation. ``logical_step`` carries the same information in a parsed string, and it
    lives in ``safe_metadata``, which is never published: the channel stays as narrow as the
    one field read out of it. So these are read from the columns, and what is published is a
    typed field rather than anything parsed.

    Rows with no attempt stamp are not compared: an unstamped row belongs to no attempt, so
    two of them are not two runs of one step.
    """
    repeats: dict[str, OperationRepeat] = {}
    grouped: dict[tuple[int, str], list[ExternalOperation]] = {}
    for operation in operations:
        attempt = child_attempt_of(operation)
        if attempt is None:
            continue
        grouped.setdefault((attempt, str(operation.operation_type)), []).append(operation)
    for siblings in grouped.values():
        if len(siblings) < 2:
            continue
        for operation in siblings:
            repeat = _repeat_of(operation, siblings)
            if repeat is not None:
                repeats[operation.operation_id] = repeat
    return repeats


def _repeat_of(
    operation: ExternalOperation, siblings: Sequence[ExternalOperation]
) -> OperationRepeat | None:
    """Which of the three answers this row's own columns support.

    ``siblings`` arrive in the order the caller holds them, newest first, which is how the
    journal answers -- so the last of them is the earliest run, and only the ones after it
    are called re-runs. The first run of a command is not a repeat of itself.
    """
    revision = operation.repository_revision
    fingerprint = operation.command_fingerprint
    if fingerprint is not None and revision is not None:
        same_command = [
            item
            for item in siblings
            if item.command_fingerprint == fingerprint and item.repository_revision is not None
        ]
        if any(item.repository_revision != revision for item in same_command):
            earliest = same_command[-1]
            if earliest.operation_id != operation.operation_id:
                return OperationRepeat(REPEAT_NEW_REVISION, revision[:_SHORT])
            # The first run of this command in the attempt. It is not a re-run of anything,
            # so it carries no differentiator at all rather than a neutral one that would
            # read as an explanation.
            return None
        if any(
            item.repository_revision == revision and item.command_fingerprint != fingerprint
            for item in siblings
        ):
            return OperationRepeat(REPEAT_DIFFERENT_COMMAND, fingerprint[:_SHORT])
    # Coding and git rows carry flat steps and may carry neither column, so there is nothing
    # here that distinguishes them. Saying only that is the honest answer.
    return OperationRepeat(REPEAT_SAME_STEP, None)
