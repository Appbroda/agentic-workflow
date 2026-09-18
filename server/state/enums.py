"""Enumerations that define workflow lifecycle and structured state events."""

from enum import StrEnum


class WorkflowStatus(StrEnum):
    """The lifecycle state of an AI software engineering workflow."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting_for_human"
    REVIEW_REJECTED = "review_rejected"
    APPROVED = "approved"
    FAILED = "failed"
    FAILED_REQUIRES_HUMAN = "failed_requires_human"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS = "cancelled_with_external_side_effects"
    COMPLETED = "completed"


class ApprovalState(StrEnum):
    """The approval position of a workflow at its current step."""

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ConversationEventType(StrEnum):
    """The structured events retained in workflow conversation history."""

    ARTIFACT_PRODUCED = "artifact_produced"
    CLARIFICATION_REQUESTED = "clarification_requested"
    CLARIFICATION_ANSWERED = "clarification_answered"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RECORDED = "approval_recorded"
    STATUS_CHANGED = "status_changed"


class LogLevel(StrEnum):
    """Supported severity levels for persisted execution logs."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class FeatureWorkflowStatus(StrEnum):
    """Lifecycle states for a feature spanning one or more repositories."""

    PENDING = "pending"
    ANALYZING_PRD = "analyzing_prd"
    WAITING_FOR_HUMAN = "waiting_for_human"
    INSPECTING_REPOSITORIES = "inspecting_repositories"
    PLANNING = "planning"
    CONTRACT_READY = "contract_ready"
    RUNNING_CHILD_WORKFLOWS = "running_child_workflows"
    INTEGRATION_REVIEW = "integration_review"
    CHANGES_REQUESTED = "changes_requested"
    READY_FOR_PULL_REQUESTS = "ready_for_pull_requests"
    CREATING_PULL_REQUESTS = "creating_pull_requests"
    COMPLETED = "completed"
    FAILED = "failed"
    FAILED_REQUIRES_HUMAN = "failed_requires_human"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS = "cancelled_with_external_side_effects"


# Statuses from which this platform will not, of its own accord, make another credentialed
# request for a feature. A person may still resume a failed one, which is why anything read
# off this set has to be paired with an age threshold rather than acted on immediately.
# `waiting_for_human` and the other resting states are deliberately absent: those are waiting
# for exactly such a request, and it is expected rather than hypothetical.
TERMINAL_FEATURE_STATUSES = frozenset(
    {
        FeatureWorkflowStatus.COMPLETED,
        FeatureWorkflowStatus.FAILED,
        FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
        FeatureWorkflowStatus.CANCELLED,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    }
)

# Statuses in which the feature has stopped and is waiting on a person's decision before the
# platform will act again. This is what the Slack human-interaction ping keys on -- a named
# family, never a list inside a dispatcher, so a new resting state added elsewhere fails the
# exhaustiveness test until somebody classifies it rather than quietly pinging nobody.
# `FAILED_REQUIRES_HUMAN` is deliberately in both this set and the terminal set: it is a
# terminal outcome *and* a question addressed to a person, and the notification that needs a
# person keys on this one. `CHANGES_REQUESTED` is deliberately absent: the integration review
# asks for fixes and the platform makes them -- nobody is waiting on a person.
HUMAN_INTERACTION_FEATURE_STATUSES = frozenset(
    {
        FeatureWorkflowStatus.WAITING_FOR_HUMAN,
        FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
    }
)

# Everything else: the feature is the platform's to advance. Declared rather than computed so
# that adding a status forces a decision -- the exhaustiveness test asserts every
# `FeatureWorkflowStatus` member is in exactly one of the three families, counting
# `FAILED_REQUIRES_HUMAN`'s deliberate dual membership as its human-interaction placement.
IN_FLIGHT_FEATURE_STATUSES = frozenset(
    {
        FeatureWorkflowStatus.PENDING,
        FeatureWorkflowStatus.ANALYZING_PRD,
        FeatureWorkflowStatus.INSPECTING_REPOSITORIES,
        FeatureWorkflowStatus.PLANNING,
        FeatureWorkflowStatus.CONTRACT_READY,
        FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
        FeatureWorkflowStatus.INTEGRATION_REVIEW,
        FeatureWorkflowStatus.CHANGES_REQUESTED,
        FeatureWorkflowStatus.READY_FOR_PULL_REQUESTS,
        FeatureWorkflowStatus.CREATING_PULL_REQUESTS,
        FeatureWorkflowStatus.CANCELLING,
    }
)


class CancellationLifecycleStatus(StrEnum):
    """The user-visible progress of a cooperative cancellation request."""

    NOT_REQUESTED = "not_requested"
    CANCELLATION_REQUESTED = "cancellation_requested"
    CANCELLATION_IN_PROGRESS = "cancellation_in_progress"
    CANCELLED = "cancelled"
    CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS = "cancelled_with_external_side_effects"


class ChildWorkflowStatus(StrEnum):
    """The independently persisted state of one repository workstream."""

    PENDING = "pending"
    RUNNING = "running"
    # Distinct from PENDING: this workstream will never run in the current attempt because
    # a build-artifact dependency failed. Reporting it as pending hid starved repositories.
    BLOCKED = "blocked"
    WAITING_FOR_CONTRACT_CHANGE = "waiting_for_contract_change"
    REVIEW_REJECTED = "review_rejected"
    APPROVED = "approved"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class TargetedAttempt(StrEnum):
    """Why a re-run targets one repository, which decides whose budget it draws down.

    Both kinds increment ``retry_count``, because an attempt that reuses the previous number
    would collide with the earlier attempt's completion, review and result artifact IDs. They
    differ in what else they spend: remediation of a contract mismatch draws on the shared
    integration allowance, while an attempt a person granted draws on the grant they made.

    It lives here rather than beside the workflow that branches on it because it is now
    *recorded*: the attempt kind is stamped onto the child alongside the counters it is
    stamped with, and read back to say which authority demanded the attempt. One vocabulary,
    persisted once.
    """

    INTEGRATION_REMEDIATION = "integration_remediation"
    OPERATOR_RETRY = "operator_retry"
    # A retry the 61- required-context guard refused was re-run as a wave of scoped attempts,
    # each narrowed to one cluster of the blocking demands (80-). Execution shape, not a new
    # grant: the wave spends the one retry the refused attempt was already granted. Older
    # builds rehydrate this value as None through the tolerant validator, which reads as a
    # workstream retry -- the compatible answer.
    CONTEXT_PARTITION = "context_partition"


class WorkstreamRole(StrEnum):
    """The responsibility a repository has in a feature delivery."""

    FRONTEND = "frontend"
    BACKEND = "backend"
    SERVICE = "service"
    SHARED = "shared"
    INFRASTRUCTURE = "infrastructure"
    OTHER = "other"


class IntegrationReviewStatus(StrEnum):
    """The outcome of a cross-repository compatibility review."""

    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    FAILED = "failed"
    FAILED_REQUIRES_HUMAN = "failed_requires_human"


class MergeStrategy(StrEnum):
    """Human-recommended order for merging a coordinated PR set."""

    BACKEND_FIRST = "backend_first"
    FRONTEND_FIRST = "frontend_first"
    SIMULTANEOUS = "simultaneous"
    INDEPENDENT = "independent"
    MANUAL = "manual"


class DeploymentStrategy(StrEnum):
    """The delivery pattern recorded for a multi-repository feature."""

    BACKEND_FIRST = "backend_first"
    FRONTEND_FIRST = "frontend_first"
    SIMULTANEOUS = "simultaneous"
    INDEPENDENT = "independent"
    MANUAL = "manual"


class ContractChangeRequestStatus(StrEnum):
    """The explicit lifecycle for an immutable-contract deviation request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class FeatureActionStatus(StrEnum):
    """The durable lifecycle of one workflow-changing action a person asked for.

    Separate from the workflow's own status because the two answer different questions.
    A feature is ``running_child_workflows``; the action that started them either reached a
    confirmed outcome or did not, and after a crash only this record can say which.
    """

    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    CLAIMED = "claimed"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    # The executor died mid-flight and the evidence does not prove what happened. Never
    # replayed automatically: that is the whole point of recording it separately from FAILED.
    REQUIRES_RECONCILIATION = "requires_reconciliation"
    CANCELLED = "cancelled"


# Statuses in which an action has reached an outcome nobody needs to act on again. A repeated
# request carrying the same identity replays the recorded result instead of executing.
TERMINAL_ACTION_STATUSES = frozenset(
    {
        FeatureActionStatus.SUCCEEDED,
        FeatureActionStatus.FAILED,
        FeatureActionStatus.CANCELLED,
    }
)

# Statuses in which an executor claims to own the action. A lease that has expired in one of
# these is what recovery looks for.
ACTIVE_ACTION_STATUSES = frozenset({FeatureActionStatus.CLAIMED, FeatureActionStatus.EXECUTING})


class RepositoryRepairStatus(StrEnum):
    """The explicit lifecycle of one proposed repair to a repository's checked-in setup."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    # The repository moved on after the proposal was written, so the diagnosis it rests on
    # may no longer describe the checkout. Re-evaluated rather than applied.
    SUPERSEDED = "superseded"
