"""Durable SQLAlchemy control plane for multi-repository parent feature workflows."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, cast
from uuid import uuid4
from weakref import WeakValueDictionary

import structlog
from pydantic import ValidationError
from sqlalchemy import CompoundSelect, delete, func, select, tuple_, union
from sqlalchemy.ext.asyncio import AsyncSession

from agents.shared.contracts import AgentArtifactError, safe_error_diagnostics
from api.control_plane import (
    FeatureOperationFailedError,
    RequestScopedCredentials,
    TimelineEvent,
    WorkflowConflictError,
    WorkflowNotFoundError,
)
from api.feature_control_plane import (
    FeatureEvent,
    FeaturePage,
    FeatureRecord,
    FeatureStartResult,
    FeatureSummary,
    _artifact_events,
    _contract_revision_artifact,
    _event_for_status,
    _feature_fingerprint,
    _feature_idempotency_key,
    _initial_feature_state,
    decode_feature_cursor,
    encode_feature_cursor,
    feature_awaits_human,
    feature_dashboard_group,
    format_feature_reference,
)
from api.feature_schemas import IntegrationContractRevision, StartFeatureRequest
from api.identity import WorkspaceScope
from api.schemas import ClarificationAnswer
from artifacts.schemas import (
    ArchitectureArtifact,
    Artifact,
    BaseArtifact,
    ChildWorkflowResultArtifact,
    CodeCompletionArtifact,
    ContractChangeRequestArtifact,
    ExecutionGraphArtifact,
    FeatureCompletionArtifact,
    FeatureRevisionRequestArtifact,
    IntegrationContractArtifact,
    IntegrationReviewArtifact,
    PRDArtifact,
    PRDAttachment,
    PullRequestArtifact,
    RepositoryExecutionPlanArtifact,
    RepositoryRepairProposalArtifact,
    ReviewArtifact,
    TaskPlanArtifact,
    TechnicalPRDArtifact,
)
from services.action_context import current_feature_action_is_revoked
from services.cancellation import CancellationRequested
from services.design_conflict import (
    VERDICTS,
    DesignConflictNotFoundError,
    find_design_conflict,
)
from services.feature_queue import (
    ANSWER_DESIGN_VERDICT,
    CONTINUE,
    PUBLISH,
    QUEUED,
    RESUME,
    RETRY_WORKSTREAM,
    REVISE,
    START,
    DatabaseFeatureExecutionQueue,
    FeatureExecutionBusy,
    FeatureExecutionQueue,
    FeatureProviderFault,
    FeatureWorkspaceCapacityUnavailable,
)
from services.repository_repair import RepairNotFoundError
from state.enums import (
    CancellationLifecycleStatus,
    ChildWorkflowStatus,
    FeatureWorkflowStatus,
    RepositoryRepairStatus,
)
from state.external_operations import (
    ExternalOperationStatus,
    ExternalOperationType,
    ManualCleanupRequirement,
    WorkflowCheckpointBoundary,
)
from state.failure_diagnosis import (
    FailureStage,
    FeatureFailureClassification,
    classification_of,
    normalize_classification,
    workspace_capacity_diagnostic,
)
from state.feature_models import (
    ChildWorkflowReference,
    FeatureWorkflowSnapshot,
    ensure_feature_failure_summary,
)
from state.feature_transitions import transition_feature, transition_feature_state_json
from storage.attachment_store import AttachmentError, AttachmentStore
from storage.db import Database
from storage.external_operation_store import (
    RECOVERY_DEFECT_ERROR_CODE as _RECOVERY_DEFECT_ERROR_CODE,
)
from storage.external_operation_store import (
    ExternalOperationJournal,
    OperationTransitionConflictError,
)
from storage.models import (
    ChildWorkflowModel,
    ContractChangeRequestModel,
    FeatureActionModel,
    FeatureArtifactModel,
    FeatureExecutionQueueModel,
    FeaturePullRequestModel,
    FeatureReferenceModel,
    FeatureWorkflowEventModel,
    FeatureWorkflowModel,
    FeatureWorkflowRequestModel,
    IntegrationContractModel,
    IntegrationReviewModel,
    RepositoryRepairModel,
    RepositorySpecModel,
)
from storage.workflow_lock import NoopWorkflowLock, WorkflowLock
from workflow_schema import workflow_mutation_identity_error
from workflows.feature_workflow import (
    ClarificationAnswerError,
    FeatureResumeNotEligible,
    FeatureWorkflowError,
    FeatureWorkflowRunner,
    RepairSupersededError,
    begin_revision,
    exhausted_fault_diagnostics,
    feature_is_at_rest,
    feature_may_be_resumed,
    is_transient_provider_fault,
    next_step,
    require_answerable_design_conflict,
    require_publishable_feature,
    require_retryable_workstream,
    settle_unfinished_children,
    transient_fault_classification,
    validate_clarification_answers,
)

# Which event names a failure that happened while carrying out one queued intent. The event
# is what an operator reads first, so "resume failed" must not be reported as "execution
# failed" simply because both go through one code path now.
_FAILURE_EVENTS = {
    RESUME: "feature_resume_failed",
    RETRY_WORKSTREAM: "feature_retry_failed",
    ANSWER_DESIGN_VERDICT: "feature_design_verdict_failed",
    PUBLISH: "feature_publication_failed",
    REVISE: "feature_revision_failed",
    START: "feature_execution_failed",
}

# Which agent could not produce the artifact a schema rejected, by the name of the model it
# was building. A rejection arrives here as a bare `ValidationError` carrying no stage, and
# this path used to answer "feature_planner" for all of them -- so AB-Feature-120, which died
# in the product manager's first model call, was recorded as a planning failure and sent its
# reader to the one component that had not run. The rejected model's name is owned by this
# repository's schemas, so reading the stage off it discloses nothing.
_REJECTED_ARTIFACT_STAGES = {
    "TechnicalPRDArtifact": "product_manager",
    "ArchitectureArtifact": "feature_planner",
    "IntegrationContractArtifact": "feature_planner",
    "RepositoryExecutionPlanArtifact": "feature_planner",
}
_STAGE_FAILURE_EVENTS = {
    "product_manager": "feature_analysis_failed",
    "feature_planner": "feature_planning_failed",
}
# What an unattributable rejection is called. Deliberately not a stage name: claiming one the
# platform cannot establish is exactly what this mapping exists to stop.
_UNATTRIBUTED_REJECTION = ("feature_artifact_rejected", "feature_runtime")


def _rejected_stage(error: AgentArtifactError | ValidationError) -> tuple[str, str]:
    """Return the event name and agent for one rejected artifact, or an honest fallback.

    A raiser that knows which agent it is says so; a bare schema rejection is attributed
    from the name of the model it rejected. Anything else is reported as unattributed
    rather than assigned to a component that may never have run.
    """
    stage = getattr(error, "stage", None)
    if not (isinstance(stage, str) and stage):
        title = error.title if isinstance(error, ValidationError) else ""
        stage = _REJECTED_ARTIFACT_STAGES.get(title)
    if stage is None:
        return _UNATTRIBUTED_REJECTION
    return _STAGE_FAILURE_EVENTS.get(stage, _UNATTRIBUTED_REJECTION[0]), stage


# A failure record must stay small enough to sit beside the state snapshot in one row.
_MAX_DIAGNOSTICS = 5
_MAX_DIAGNOSTIC_CHARACTERS = 2_000
_LOCAL_STATE_WRITE_LOCKS: WeakValueDictionary[tuple[str, str], asyncio.Lock] = WeakValueDictionary()

# The qualifiers a revision adds to a pull-request artifact id after the repository id:
# `.v{n}` on a revision's own pull request, `.closed` on the durable record that a superseded
# one was closed. Stripped when projecting the row so `repository_id` stays the repository id.
_PULL_REQUEST_ID_QUALIFIERS = re.compile(r"(\.v\d+)?(\.closed)?$")

_FEATURE_ARTIFACT_MODELS: dict[str, type[BaseArtifact]] = {
    "prd": PRDArtifact,
    "technical_prd": TechnicalPRDArtifact,
    "architecture": ArchitectureArtifact,
    "execution_graph": ExecutionGraphArtifact,
    "task_plan": TaskPlanArtifact,
    "code_completion": CodeCompletionArtifact,
    "review": ReviewArtifact,
    "pull_request": PullRequestArtifact,
    "integration_contract": IntegrationContractArtifact,
    "repository_execution_plan": RepositoryExecutionPlanArtifact,
    "child_workflow_result": ChildWorkflowResultArtifact,
    "integration_review": IntegrationReviewArtifact,
    "contract_change_request": ContractChangeRequestArtifact,
    "feature_completion": FeatureCompletionArtifact,
    "feature_revision_request": FeatureRevisionRequestArtifact,
}


# A feature still occupies a worker, a workspace and provider quota until it settles, and
# `waiting_for_human` settles nothing: it holds its checkout open indefinitely.
# What the queue records as a failed run. `waiting_for_human` is deliberately absent: a
# feature paused on a clarification has not failed, and closing its entry as a failure would
# make an ordinary question look like a defect.
_TERMINAL_FAILURE_STATUSES = frozenset(
    {
        FeatureWorkflowStatus.FAILED,
        FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
    }
)


_QUOTA_SETTLED_STATUSES = (
    FeatureWorkflowStatus.COMPLETED,
    FeatureWorkflowStatus.CANCELLED,
    FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    FeatureWorkflowStatus.FAILED,
    FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
)

# The feature has finished deciding and will schedule nothing further, so no workstream of
# it can still be in flight. `CANCELLING` is deliberately absent: it is the one status that
# says work is still active. `COMPLETED` is absent too -- reaching it requires every
# workstream to have finished already, so there is nothing there to settle.
#
# The failure statuses are here because a workstream really can be left mid-attempt by one.
# When a sibling raises hard enough to escape the fan-out -- AB-Feature-131 and -132 hit a
# full disk -- the feature is recorded terminal and the attempt still running underneath it
# is never closed. Both sat at `running` under `failed_requires_human` with the feature
# untouched for eleven minutes, which is the same row AB-Feature-107 left behind after a
# cancellation, reached by a different door.
_CANCELLED_FEATURE_STATUSES = frozenset(
    {
        FeatureWorkflowStatus.CANCELLED,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    }
)
_SETTLED_FEATURE_STATUSES = _CANCELLED_FEATURE_STATUSES | frozenset(
    {
        FeatureWorkflowStatus.FAILED,
        FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
    }
)

_RETIRED_EQUIVALENT_STATUSES = frozenset(
    {
        FeatureWorkflowStatus.COMPLETED.value,
        FeatureWorkflowStatus.CANCELLED.value,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS.value,
    }
)

# What every method here does when nobody names a workspace: reach the whole deployment.
#
# That is correct for the callers who have no request and no actor -- the queue dispatcher,
# the recovery sweeps, the Slack dispatcher, and the tests that drive the lifecycle directly.
# It is wrong for a route, and a route cannot get it: `get_feature_control_plane` hands routes
# a `ScopedFeatureControlPlane`, which supplies its own scope on every call and ignores the
# absence of one. So the permissive default is reachable only by deliberately holding the
# unwrapped store, which the composition in `main.py` gives to workers and to nothing else.
_ANY_WORKSPACE = WorkspaceScope.unscoped()

# The statuses that assert a process is working on this feature right now. Deliberately not
# every non-terminal status: `pending` belongs to the queue, and `waiting_for_human`,
# `contract_ready`, `changes_requested` and `ready_for_pull_requests` are resting states
# where nothing running is expected and a sweep would be wrong to intervene.
_ACTIVELY_RUNNING_STATUSES = frozenset(
    {
        FeatureWorkflowStatus.ANALYZING_PRD.value,
        FeatureWorkflowStatus.INSPECTING_REPOSITORIES.value,
        FeatureWorkflowStatus.PLANNING.value,
        FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS.value,
        FeatureWorkflowStatus.INTEGRATION_REVIEW.value,
        FeatureWorkflowStatus.CREATING_PULL_REQUESTS.value,
        FeatureWorkflowStatus.CANCELLING.value,
    }
)

# The statuses a claim may continue a crashed run from. Every status that asserts something
# is executing, minus `cancelling`: a cancellation is already ending, it records its own
# cleanup requirements as it goes, and picking one back up as ordinary work would discard
# them. A cancellation whose executor died is left to the abandoned-run sweep.
_CONTINUABLE_STATUSES = _ACTIVELY_RUNNING_STATUSES - {FeatureWorkflowStatus.CANCELLING.value}

# The statuses from which an unconfirmed remote effect is still worth stopping. Cancelling
# is excluded: a cancellation already accounts for retained side effects through its cleanup
# requirements, and turning it into a failure would lose that.
_ESCALATABLE_STATUSES = frozenset(
    {
        FeatureWorkflowStatus.ANALYZING_PRD,
        FeatureWorkflowStatus.INSPECTING_REPOSITORIES,
        FeatureWorkflowStatus.PLANNING,
        FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
        FeatureWorkflowStatus.INTEGRATION_REVIEW,
        FeatureWorkflowStatus.CREATING_PULL_REQUESTS,
    }
)

# The child statuses an abandoned run leaves behind mid-flight.
_UNFINISHED_CHILD_STATUSES = frozenset(
    {
        ChildWorkflowStatus.PENDING,
        ChildWorkflowStatus.RUNNING,
        ChildWorkflowStatus.WAITING_FOR_CONTRACT_CHANGE,
    }
)

# The two snapshot fields that have their own tables. `_state_from_model` rebuilds both from
# `feature_artifacts` and `feature_child_workflows` on every load and throws away whatever the
# JSON said, so a copy inside `state_json` was written and never read -- while being the copy a
# person sees when they open the row in psql, and the reason a completed feature's snapshot
# reached hundreds of kilobytes.
_SHADOWED_STATE_FIELDS = frozenset({"artifacts", "child_workflows"})


def _persisted_state_json(state: FeatureWorkflowSnapshot) -> dict[str, Any]:
    """Serialize a snapshot without the two fields typed rows already own."""
    return state.model_dump(mode="json", exclude=set(_SHADOWED_STATE_FIELDS))


class ClaimDisposition(StrEnum):
    """What a worker should do with the queue entry it just claimed.

    Two answers were not enough. `START` was treated as valid only on `pending`, so a claim
    that arrived on a feature in `running_child_workflows` -- which is what a lapsed lease
    after a worker died looks like -- was dropped as unwanted, the entry closed `succeeded`,
    and forty-five minutes later the abandoned-run sweep tombstoned a feature whose whole
    checkpointed state was sitting there. The third answer is that one.
    """

    # The claim describes work this feature has not started yet: run the intent it names.
    RUN = "run"
    # The feature claims to be executing and nothing is: continue it from its checkpoint.
    CONTINUE = "continue"
    # The feature has moved past this claim, or is resting where a person owns it.
    DROP = "drop"


def _executing_feature_ids(
    now: datetime, *, excluding_queue_claim_on: str | None = None
) -> CompoundSelect[tuple[str]]:
    """Select every feature something is demonstrably executing right now.

    One definition with two readers, which is the point of extracting it. The abandoned-run
    sweep excludes these features from its scan; a claim that lands on an actively-running
    feature asks whether its own feature is in here before continuing the run. Two callers
    with two copies of this query would eventually disagree, and the disagreement would be
    either a tombstoned feature that was fine or two workers on one feature.

    The evidence is deliberately only what a dead process cannot fake: a live action lease, a
    live queue lease, or a queue entry still waiting to be claimed. None of them survive the
    process that took them.

    `excluding_queue_claim_on` names the feature whose queue entry the caller itself holds.
    The queue's primary key is the feature, so a worker executing one sees exactly one queue
    row for it -- its own claim -- and that row is not evidence that something *else* is
    running. Its own action lease is not excluded: an action lease belongs to a live executor
    and the worker asking this question has not started one yet.
    """
    leased_actions = (
        select(FeatureActionModel.feature_id)
        .where(FeatureActionModel.lease_expires_at.is_not(None))
        .where(FeatureActionModel.lease_expires_at > now)
    )
    leased_queue_entries = (
        select(FeatureExecutionQueueModel.feature_id)
        .where(FeatureExecutionQueueModel.lease_expires_at.is_not(None))
        .where(FeatureExecutionQueueModel.lease_expires_at > now)
    )
    # A queue entry still waiting to be claimed is not abandoned -- the dispatcher has not
    # started it yet, and its own attempt budget bounds it.
    waiting_queue_entries = select(FeatureExecutionQueueModel.feature_id).where(
        FeatureExecutionQueueModel.status == QUEUED
    )
    if excluding_queue_claim_on is not None:
        leased_queue_entries = leased_queue_entries.where(
            FeatureExecutionQueueModel.feature_id != excluding_queue_claim_on
        )
        waiting_queue_entries = waiting_queue_entries.where(
            FeatureExecutionQueueModel.feature_id != excluding_queue_claim_on
        )
    return union(leased_actions, leased_queue_entries, waiting_queue_entries)


class FeatureRunRevokedError(WorkflowConflictError):
    """A run that lost its durable claim tried to change the feature anyway.

    Subclasses `WorkflowConflictError` so existing handlers -- the API's conflict mapping,
    the recovery sweep's skip list -- already treat it correctly: the caller is out of date,
    not broken. It is a distinct type so a log or a test can tell this apart from an
    ordinary concurrent-write conflict, because the two need very different responses.
    """


class SqlAlchemyFeatureControlPlane:
    """Persist parent state, children, contracts, reviews, and partial PR sets across restarts."""

    def __init__(
        self,
        database: Database,
        *,
        mock_runner: FeatureWorkflowRunner | None = None,
        live_runner: FeatureWorkflowRunner | None = None,
        lock: WorkflowLock | None = None,
        operation_journal: ExternalOperationJournal | None = None,
        cancellation_signaler: Callable[[str], Awaitable[None]] | None = None,
        max_concurrent_live_features: int = 0,
        queue: FeatureExecutionQueue | None = None,
        on_queued: Callable[[str], None] | None = None,
        # The volume live features clone into, and the floor below which accepting another
        # one is accepting work that will die an hour later. Both absent means the check is
        # off, which is what every in-memory and test composition wants.
        workspace_root: Path | None = None,
        workspace_minimum_free_bytes: int = 0,
        # Where a submission's images live. Injected rather than constructed here so an
        # isolated application's in-memory store is the one acceptance binds against; a
        # composition with no attachment store accepts no submission that references one,
        # which the route has already refused by then.
        attachments: AttachmentStore | None = None,
    ) -> None:
        """Bind durable storage and mode-specific runners without retaining provider credentials."""
        self._database = database
        self._attachments = attachments
        # The queue is what makes "accepted" durable independently of "executed". It is
        # written in the same transaction as the feature, so a crash between the two cannot
        # leave a feature nothing will ever run.
        self._queue = queue or DatabaseFeatureExecutionQueue(database)
        self._on_queued = on_queued
        self._mock_runner = mock_runner
        self._live_runner = live_runner
        self._lock = lock or NoopWorkflowLock()
        # Redis serializes writers across API processes in production.  The redacted
        # database identity lets separate local control-plane instances share a weak lock
        # without retaining connection credentials or leaking one lock per completed feature.
        self._state_lock_scope = database.engine.url.render_as_string(hide_password=True)
        self._operation_journal = operation_journal
        self._cancellation_signaler = cancellation_signaler
        # Nothing bounded how many live features could run at once. Each one clones every
        # repository it touches, installs their dependency trees and holds provider quota, so
        # the first limit any of them meets is the disk or the provider rather than a policy
        # this platform chose. Zero keeps the previous unbounded behaviour.
        self._max_concurrent_live_features = max_concurrent_live_features
        self._workspace_root = workspace_root
        self._workspace_minimum_free_bytes = workspace_minimum_free_bytes
        self._bind_checkpoint_writer(self._live_runner)
        self._bind_event_writer(self._live_runner)

    @property
    def queue(self) -> FeatureExecutionQueue:
        """The queue this control plane writes accepted features to."""
        return self._queue

    async def start(
        self,
        request: StartFeatureRequest,
        *,
        idempotency_key: str | None,
        credentials: RequestScopedCredentials,
        owner_id: str,
        model_setup: dict[str, Any] | None = None,
        attachments: Sequence[PRDAttachment] | None = None,
    ) -> FeatureStartResult:
        """Persist the feature and queue its execution, then answer.

        This used to run the whole of planning before replying: the product manager read the
        PRD and reconnaissance cloned every repository while the caller waited, so a
        submission was invisible for minutes and lost outright if the request timed out. Now
        the feature, its reference and its queue entry commit together and the response goes
        back queued; a worker claims the entry afterwards.

        `credentials` is deliberately unused here and deliberately still accepted. Execution
        happens outside this request, so it resolves the requesting identity's stored
        credentials rather than carrying a header's value into a background task -- and the
        signature stays what every caller and the protocol already expect.

        `owner_id` is required, with no default. It becomes `feature_workflows.owner_id`,
        which decides who may ever see this feature, and the queue entry's `requested_by`,
        which decides whose stored credentials the worker resolves -- and for a submission
        those are the same identity by construction, because the person submitting is the
        person whose workspace it lands in. It replaced a `requested_by: str = "api"`
        default: an enqueue that took that default named an account that does not exist and
        resolved no credentials, which was invisible while every identity was shared.
        """
        del credentials
        fingerprint = _feature_fingerprint(request)
        key = _feature_idempotency_key(
            idempotency_key or request.idempotency_key, fingerprint, owner_id
        )
        async with self._lock.hold(f"feature:start:{key}"):
            if request.execution_mode == "live":
                await self._require_live_capacity()
                await self._require_workspace_capacity()
            result = await self._create_or_replay(
                request,
                key=key,
                fingerprint=fingerprint,
                owner_id=owner_id,
                model_setup=model_setup,
                attachments=attachments,
            )
        if result.created and self._on_queued is not None:
            self._on_queued(result.record.state.feature_id)
        return result

    async def execute_queued(
        self,
        feature_id: str,
        *,
        credentials: RequestScopedCredentials,
        intent: str = START,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Run one queued feature, recording every outcome against the feature itself.

        Lifted unchanged out of `start`, including its exception handling: the reason each of
        these branches exists has not changed by moving off the request thread. What has
        changed is that a caller is no longer waiting, so a raise here reaches the queue
        rather than an HTTP response -- and the queue decides whether to retry.

        `intent` is what lets resuming and granting a retry be queued work too. Both used to
        run the whole feature inside the HTTP request that asked for them, which is how
        AB-Feature-108 lost seventy minutes to a cancelled socket.
        """
        arguments = dict(payload or {})
        record = await self.get_record(feature_id)
        runner = self._runner_for(record.state.execution_mode)
        if runner is None:
            return
        if await self._claim_disposition(record.state, intent) is ClaimDisposition.DROP:
            # Checked before the lock as well as inside it. A claim that arrives on a feature
            # which has already moved on must return at once rather than spending ten seconds
            # failing to acquire a lock held by the run that moved it.
            return
        try:
            lock = self._lock.hold(f"feature:run:{feature_id}")
            await lock.__aenter__()
        except WorkflowConflictError as error:
            # Another worker is executing this feature right now. That is not a failure of
            # this feature and must not spend one of its attempts: three of these in thirty
            # seconds retired AB-Feature-104 while its first attempt was still working.
            raise FeatureExecutionBusy(str(error)) from error
        try:
            record = await self.get_record(feature_id)
            disposition = await self._claim_disposition(record.state, intent)
            if disposition is ClaimDisposition.DROP:
                # Something already moved this feature: a replayed claim, a cancellation, or
                # a resume that got there first. Running it again would rewind it.
                return
            if disposition is ClaimDisposition.CONTINUE:
                # Recorded before the work rather than after it, because the reason this
                # event exists is to explain a feature that goes on to die again. The status
                # is unchanged: what continues is the run, from the checkpoint it reached.
                await self._record_queued_refusal(
                    feature_id,
                    event="feature_run_continued",
                    reason=(
                        "The process executing this feature stopped without recording an "
                        f"outcome. It was left in {record.state.status.value} and is being "
                        "continued from its last checkpoint."
                    ),
                )
                record = await self.get_record(feature_id)
            try:
                updated = await self._run_intent(
                    runner,
                    record.state,
                    intent=intent,
                    arguments=arguments,
                    credentials=credentials,
                    continuation=disposition is ClaimDisposition.CONTINUE,
                )
            except CancellationRequested:
                await self._finish_cancellation(feature_id)
                return
            except ClarificationAnswerError as error:
                # The answers were checked synchronously when the resume was accepted, so
                # reaching here means the feature moved underneath the queued entry. It is
                # still answerable, so it must stay where it is rather than be failed.
                await self._record_queued_refusal(
                    feature_id, event="feature_resume_refused", reason=str(error)
                )
                return
            except FeatureResumeNotEligible as error:
                # Every workstream has already reached a terminal retry decision, so there is
                # nothing left for a resume to run. This used to return the state unchanged
                # and the entry closed `succeeded`: a feature that disappeared without one
                # line anywhere saying it had been asked and had declined.
                await self._record_queued_refusal(
                    feature_id,
                    event="feature_resume_found_no_eligible_workstreams",
                    reason=str(error),
                )
                return
            except (AgentArtifactError, ValidationError) as error:
                event, agent = _rejected_stage(error)
                await self._mark_failed_requires_human(
                    feature_id, event, error, current_agent=agent
                )
                return
            except Exception as error:
                if is_transient_provider_fault(error):
                    # Deliberately not recorded terminal here. This entry has attempts
                    # reserved for exactly this failure, and the dispatcher is what knows
                    # whether any are left. Marking the feature and returning normally --
                    # what this did -- closed the entry `succeeded` on attempt 1 of 3 and
                    # left the other two unspent, so every recovery had to be a person
                    # pressing resume. `fail_exhausted_fault` below records the same terminal
                    # state once the attempts really are gone.
                    raise FeatureProviderFault(error) from error
                if (
                    classification_of(error)
                    is FeatureFailureClassification.PLATFORM_CAPACITY_FAILURE
                ):
                    # The same reasoning, for the same reason, about the worker volume. This
                    # is the one condition that recorded the feature as needing a human
                    # *while a claim was still retrying it*: run 193 read
                    # `failed_requires_human` for an hour while the dispatcher went on
                    # claiming it, and a manual cleanup then let the very next claim finish
                    # the feature. The timeline gains the line that says what happened; the
                    # status stays where it is, because it is not true yet.
                    #
                    # Classified rather than typed, so this reads the same condition whether
                    # it was raised by the mid-run preflight or by the acceptance-time check
                    # that shares its wording, without the storage layer importing either.
                    await self._record_queued_refusal(
                        feature_id,
                        event="feature_workspace_capacity_unavailable",
                        reason=(
                            "The worker volume could not provide the workspace this step "
                            "needs. The feature keeps its status and this claim will be "
                            "retried; nothing about it has been decided."
                        ),
                    )
                    raise FeatureWorkspaceCapacityUnavailable(error) from error
                await self._mark_failed_requires_human(
                    feature_id,
                    _FAILURE_EVENTS.get(intent, "feature_execution_failed"),
                    error,
                    current_agent="feature_runtime",
                )
                return
            await self._replace_state(feature_id, updated, _event_for_status(updated.status))
        finally:
            await lock.__aexit__(None, None, None)

    async def _run_intent(
        self,
        runner: FeatureWorkflowRunner,
        state: FeatureWorkflowSnapshot,
        *,
        intent: str,
        arguments: dict[str, Any],
        credentials: RequestScopedCredentials,
        continuation: bool = False,
    ) -> FeatureWorkflowSnapshot:
        """Apply what this claim's intent carries, then advance the feature by one step.

        A claim used to mean "run this whole feature": it went to `start`, `resume` or
        `retry_workstream` and came back when the feature was finished, which for a live one
        meant a mean of 86 minutes and a worst case of 571 inside a single coroutine. Whatever
        that coroutine had not checkpointed died with it. Now it means "advance this feature by
        one step and return", and the dispatcher re-queues whatever is left.

        Every intent is therefore two things: what the *request* decided, applied here exactly
        once, and one step, which is the same step for all of them. `continuation` says this
        claim is picking a crashed run back up rather than starting the work its intent names;
        it settles as an empty-answer resume, which walks the persisted artifacts, rather than
        as a `start` that would re-run the product manager over artifacts that already exist.

        `continue` is the intent with nothing to apply: it is a claim this platform re-queued
        for itself between two steps of a run already in progress, so there is no request to
        settle and it goes straight to the step.
        """
        if intent == CONTINUE and not continuation:
            return await runner.advance_one_step(state, credentials=credentials)
        if intent == REVISE and not continuation:
            # Nothing to apply: the revision itself -- request artifact, replacement PRD,
            # reset children -- was applied synchronously by `revise_feature` before this
            # entry existed. The claim arrives on a `planning` feature and the evidence
            # already calls for the planning step, so this is `continue` under its true name.
            return await runner.advance_one_step(state, credentials=credentials)
        if intent == RETRY_WORKSTREAM and not continuation:
            # The one step that is not the step `next_step` would choose. A grant names a
            # repository and buys attempts for that repository, so the attempt it paid for
            # runs targeted; everything after it is an ordinary step.
            return await runner.grant_and_run_one_retry(
                state,
                repository_id=str(arguments["repository_id"]),
                additional_attempts=int(arguments["additional_attempts"]),
                requested_by=str(arguments["requested_by"]),
                reason=str(arguments["reason"]),
                credentials=credentials,
            )
        if intent == ANSWER_DESIGN_VERDICT and not continuation:
            # The same shape as a grant, and targeted for the same reason: a verdict is about
            # one repository's design question, so the attempt it authorises is that
            # repository's. What differs is that the verdict is recorded into the lineage
            # first, so the attempt runs with the decision already binding it.
            return await runner.answer_design_conflict(
                state,
                conflict_id=str(arguments["conflict_id"]),
                verdict=str(arguments["verdict"]),
                decision=str(arguments["decision"]),
                decided_by=str(arguments["decided_by"]),
                additional_attempts=int(arguments.get("additional_attempts", 0)),
                credentials=credentials,
            )
        if intent == PUBLISH and not continuation:
            # Not a step, and deliberately not followed by one. A person decided that a
            # feature which did not land should open what it has; that decision is carried out
            # here and the feature is left exactly as unfinished as it was.
            return await runner.publish_feature(
                state,
                requested_by=str(arguments["requested_by"]),
                reason=str(arguments.get("reason", "")),
                credentials=credentials,
            )
        if continuation or intent == RESUME:
            # Answers only travel with the first attempt that gets to apply them. A queued
            # resume is now run once per step -- and can be run more than once per step, since
            # a provider fault spends an attempt and a lapsed lease returns the claim -- and by
            # the second run the answers are already persisted into the Technical PRD and the
            # feature has left `waiting_for_human`. Sending them again is refused as "answers
            # require a feature waiting for human input", which would quietly end the retry the
            # attempt was spent on. What is actually left to do at that point is what an
            # empty-answer resume means: continue from the checkpoint they were written to.
            answered = not continuation and state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
            state = await runner.begin_resume(
                state,
                answers=[
                    ClarificationAnswer.model_validate(item)
                    for item in (arguments.get("answers", []) if answered else [])
                ],
            )
        return await runner.advance_one_step(state, credentials=credentials)

    async def _claim_disposition(
        self, state: FeatureWorkflowSnapshot, intent: str
    ) -> ClaimDisposition:
        """Decide what a claimed entry should do with the feature it names.

        Three cases, not two. A resume or a retry claim is valid on anything not already
        finished, because what it continues from is the feature's own persisted state; that
        is unchanged. What changes is the start claim, which used to be valid only on
        `pending` and therefore had exactly one answer for two very different situations:

        * the feature moved past this claim, or is resting somewhere a person owns it --
          `waiting_for_human`, `contract_ready`, `changes_requested`,
          `ready_for_pull_requests`. Dropping the claim is right, and still what happens.
        * the feature says it is executing and nothing is executing it. That is a worker that
          died, and dropping the claim is how AB-Feature-108 sat in `running_child_workflows`
          for a day: the entry closed `succeeded` with the checkpoints intact and unread.

        The second case is established rather than assumed, from the same evidence the
        abandoned-run sweep uses and through the same query.

        Both fences below now apply to *every* intent, not only to `start`. They used to be
        reachable only through a start claim, because that was the only claim that could
        arrive on a feature already executing -- resuming was something a person did to a
        feature at rest. A claim is one step now, so the ordinary claim on a mid-run feature
        carries `resume`: leaving the fences on `start` alone would have left the common case
        unguarded, with nothing bounding how many times a step that kills its process is
        allowed to kill the next one.
        """
        if state.status.value in _RETIRED_EQUIVALENT_STATUSES:
            return ClaimDisposition.DROP
        if intent == START and state.status is FeatureWorkflowStatus.PENDING:
            return ClaimDisposition.RUN
        if state.status.value not in _CONTINUABLE_STATUSES:
            # Resting somewhere a person owns it. A resume or a granted retry is exactly the
            # request that may act on that; a start claim has been overtaken and must not.
            return ClaimDisposition.DROP if intent == START else ClaimDisposition.RUN
        if not await self._nothing_is_executing(state.feature_id, holding_queue_claim=True):
            # Another worker really is running this. Dropping is right and the run lock would
            # have said the same thing, more expensively.
            return ClaimDisposition.DROP
        attempts = await self._claim_attempts(state.feature_id)
        if attempts is not None and attempts[0] > attempts[1]:
            # A feature that has crashed as many times as its entry allows is a feature
            # something is systematically wrong with. The budget is the bound on continuation
            # and is deliberately not raised to make continuing feel safer; the abandoned-run
            # sweep records the terminal state and says how many continuations it took.
            #
            # It counts crashes rather than claims, which is what makes it survive stepping:
            # `advance_to_next_step` returns the attempt whenever a step completes, so what is
            # left here is consecutive claims that recorded nothing. A forty-step feature
            # never approaches it; three deaths in one step reach it exactly as before.
            return ClaimDisposition.DROP
        return ClaimDisposition.CONTINUE if intent == START else ClaimDisposition.RUN

    async def _nothing_is_executing(self, feature_id: str, *, holding_queue_claim: bool) -> bool:
        """Answer whether any live claim on this feature survives the process that took it.

        `holding_queue_claim` says the caller is the worker holding this feature's queue
        entry. Its own claim is then excluded, because a worker cannot take its own lease as
        proof that somebody else is running the feature; what remains is an action lease a
        live executor renews. The sweep passes `False` and counts the queue lease, which for
        it is the whole point.
        """
        executing = _executing_feature_ids(
            datetime.now(UTC),
            excluding_queue_claim_on=feature_id if holding_queue_claim else None,
        ).subquery()
        async with self._database.session() as session:
            claimant = await session.scalar(
                select(executing.c.feature_id).where(executing.c.feature_id == feature_id).limit(1)
            )
        return claimant is None

    async def _record_queued_refusal(self, feature_id: str, *, event: str, reason: str) -> None:
        """Note something the queue decided, without moving the feature off its state.

        Used for a refusal -- work the claim declined to do -- and for a continuation, which
        is the opposite decision and needs the same treatment: the feature stays exactly where
        it is and the timeline gains a line saying why.
        """
        state = (await self.get_record(feature_id)).state.model_copy(deep=True)
        state.updated_at = datetime.now(UTC)
        await self._replace_state(feature_id, state, event, {"reason": reason})

    async def fail_exhausted_fault(
        self, feature_id: str, *, intent: str, error: Exception, attempts: int
    ) -> None:
        """Record a retryable fault as this feature's stop, once no queued attempt remains.

        Separate from `fail_queued` because that one refuses work the queue never ran and is
        gated on a still-`pending` feature. This one closes out work that did run, from
        whatever stage it reached, so it must not be gated that way -- a resume that failed
        in planning is not pending, and gating it there would leave the feature mid-status
        with no summary at all.

        Whichever fault ran out of attempts: a provider that did not answer, a Git remote that
        did not either, or a worker volume that could not provide a workspace. The
        classification comes off the error, so the record names the condition rather than the
        queue's route to it -- but it comes off `transient_fault_classification` and not off
        the exception's type name, which is what `classification_of` alone answered. No type
        name carries a transient marker, so every fault the queue ran out of attempts on was
        filed as a defect in this platform: `retryable` false, "inspect the persisted
        diagnostics and repository evidence" as the next action, and a logbook one-liner
        reading "the platform itself failed while running it" for weather.

        The attempts this entry spent go in the diagnostics rather than the classification. A
        service that did not answer is what the classification says, and it says the same thing
        whether one attempt failed or all of them did; that a person is now the only thing
        that will move this feature is a fact about this route alone.
        """
        await self._mark_failed_requires_human(
            feature_id,
            _FAILURE_EVENTS.get(intent, "feature_execution_failed"),
            error,
            current_agent="feature_runtime",
            error_type=transient_fault_classification(error).value,
            diagnostics_prefix=exhausted_fault_diagnostics(error, attempts=attempts),
        )

    async def stop_unprogressing_run(self, feature_id: str, *, steps: int) -> None:
        """Stop a feature that keeps asking for another step without ever finishing.

        The step ceiling's terminal record. `fail_queued` cannot write it: that one refuses
        work the queue never ran and is gated on a still-`pending` feature, deliberately, so a
        stale claim cannot overwrite a run that progressed. This feature progressed -- it took
        every step it was allowed -- so it needs the other kind of stop, the one
        `stop_overrunning_runs` writes when a feature runs out of time rather than steps.

        Silence here would be the defect task 25- exists to prevent: a feature that ended and
        cannot say what ended it.
        """
        record = await self.get_record(feature_id)
        state = record.state.model_copy(deep=True)
        previous_status = state.status.value
        transition_feature(
            state,
            FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
            reason=(
                f"This feature took the {steps} steps this deployment allows one run and "
                "still had work outstanding."
            ),
            agent="feature_queue",
        )
        state.updated_at = datetime.now(UTC)
        settle_unfinished_children(state, ChildWorkflowStatus.FAILED)
        state = ensure_feature_failure_summary(
            state,
            stage=FailureStage.FEATURE_QUEUE,
            classification=FeatureFailureClassification.FEATURE_RUNTIME_LIMIT_REACHED,
            diagnostics=[
                f"This feature took the {steps} steps this deployment allows one run and "
                "still had work outstanding, so it was stopped rather than kept going.",
                f"It was in {previous_status} when it was stopped. A feature that has not "
                "finished after that many steps is repeating one rather than progressing "
                "through them; read what the last one recorded before starting it again.",
                "Nothing was interrupted. Any push or pull request this run had already "
                "started is recorded in the operation journal and settles on its own.",
            ],
        )
        await self._replace_state(
            feature_id,
            state,
            "feature_step_budget_exhausted",
            {"previous_status": previous_status, "steps": steps},
        )

    async def fail_queued(
        self,
        feature_id: str,
        *,
        event: str,
        reason: str,
        error_type: str = FeatureFailureClassification.FEATURE_QUEUE_REFUSED.value,
    ) -> None:
        """Record that a queued feature will not be started, and say why in its own record.

        Used for refusals the queue can decide without running anything -- missing provider
        credentials, attempts exhausted. The feature is left resumable, because both of those
        are things somebody can fix.
        """
        record = await self.get_record(feature_id)
        if record.state.status is not FeatureWorkflowStatus.PENDING:
            # The feature got somewhere. The queue has no business overwriting a run that
            # progressed, and a stale claim reporting failure over a working feature is how a
            # perfectly good run becomes "waiting on you" with nothing to answer.
            return
        state = record.state.model_copy(deep=True)
        # Marked failed first so the summary is actually written: `ensure_feature_failure_summary`
        # only diagnoses a failed state, so setting `waiting_for_human` up front produced a
        # feature that said it was waiting on somebody and carried no explanation at all.
        transition_feature(
            state,
            FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
            reason=reason,
            agent="feature_queue",
        )
        state.updated_at = datetime.now(UTC)
        state = ensure_feature_failure_summary(
            state,
            stage=FailureStage.FEATURE_QUEUE,
            # The queue names the refusal it made; anything it cannot name is the platform's
            # own defect rather than a statement about this account's repositories.
            classification=normalize_classification(error_type),
            diagnostics=[reason],
        )
        # Then left answerable, which every one of these refusals is: a missing key can be
        # configured and a collision can be re-run.
        transition_feature(state, FeatureWorkflowStatus.WAITING_FOR_HUMAN, reason=reason)
        await self._replace_state(feature_id, state, event, {"reason": reason})

    async def feature_owes_another_step(self, feature_id: str, *, intent: str = CONTINUE) -> bool:
        """Report whether this feature has a next step the platform may take unasked.

        The question the dispatcher asks to decide between re-queueing the entry and closing
        it. Two authorities answer it, and both are needed: `next_step` says what the
        feature's evidence still calls for, and `feature_is_at_rest` says whether the platform
        may act on that without being asked. Collapsing them would either strand a feature
        that stopped for a person -- because its evidence still names work a resume could do
        -- or restart one, which is the same defect from the other side.

        `intent` is the one the claim just ran under, and it has to be. A `start` claim that
        arrives on a feature resting where a person owns it is dropped; asking this under any
        other intent would answer that the entry should be reopened, and reopening it under an
        intent that does not drop runs exactly the work the drop refused.
        """
        try:
            record = await self.get_record(feature_id)
        except WorkflowNotFoundError:
            # Retired or deleted between the step and this read. There is nothing left to
            # advance, and a queue entry is not the place to decide otherwise.
            return False
        if feature_is_at_rest(record.state) or next_step(record.state).step is None:
            return False
        if not feature_may_be_resumed(record.state):
            # Every workstream has spent its retry decision. The evidence still names a step
            # -- an integration review it never reached -- but no claim may run it, and the
            # claim that just declined already said so on the timeline.
            return False
        # A step the next claim would refuse is not a step this entry should be reopened for.
        # Asked through the same disposition that claim would use, so a spent crash budget or
        # a concurrent executor ends this entry here instead of re-queueing it into a claim
        # that drops it, and drops it again, for as long as anything keeps asking.
        return await self._claim_disposition(record.state, intent) is not ClaimDisposition.DROP

    async def feature_run_succeeded(self, feature_id: str) -> bool:
        """Report what the feature ended as, so the queue entry can record the same thing.

        `execute_queued` returns normally after marking a feature failed, so the dispatcher
        had nothing but "no exception escaped" to close the entry on and closed it
        `succeeded`. The queue therefore reported 73 successes against 95 features waiting on
        a human. Waiting is not failing -- a clarification pause is a working run -- so only
        the two terminal failure statuses answer no.
        """
        try:
            record = await self.get_record(feature_id)
        except WorkflowNotFoundError:
            # Retired or deleted between the run and this read. Nothing here is evidence the
            # run failed, and a queue entry is not the place to decide that it did.
            return True
        return record.state.status not in _TERMINAL_FAILURE_STATUSES

    async def resume(
        self,
        feature_id: str,
        *,
        answers: Sequence[ClarificationAnswer],
        credentials: RequestScopedCredentials,
        scope: WorkspaceScope = _ANY_WORKSPACE,
    ) -> FeatureRecord:
        """Queue a resumption of this feature and answer, rather than running it inline.

        This used to run the entire feature inside the caller's request, exactly as `start`
        once did. AB-Feature-108 is why it no longer does: answering a clarification opened a
        `POST /resume` that then ran contract planning, two workstreams and nine coding
        attempts inside itself for seventy minutes, until one lease renewal failed and the
        request was cancelled. The run died mid-attempt with no terminal status, because the
        only thing that knew it existed was a socket.

        Everything answerable synchronously still is: an audit-only snapshot, a missing
        runner, and answers that do not match the open questions are all refused here, with
        the feature untouched. What moves is the execution.

        `credentials` is accepted and deliberately unused, as in `start`: the work outlives
        this request, so the worker resolves the requesting identity's stored credentials
        instead of carrying a header's value into a background task.

        The identity on the queue entry is the feature's *owner*, read off the row, and never
        whoever pressed the button. `requested_by` used to be a parameter here for the acting
        actor; it is gone, because the only correct value is one this method can look up.
        """
        del credentials
        async with self._lock.hold(f"feature:run:{feature_id}"):
            record = await self.get_record(feature_id, scope=scope)
            _require_mutable_feature_identity(record.state)
            # Raises if this feature's mode has no runner configured, which must remain a
            # synchronous refusal: queueing work nothing can execute would strand it.
            self._required_runner(record.state.execution_mode)
            if record.state.status.value in _RETIRED_EQUIVALENT_STATUSES:
                # Refused here rather than dropped by the worker. A finished feature has
                # nothing to resume, and accepting the request would answer as though
                # something had been arranged while the claim was silently discarded.
                msg = f"a {record.state.status.value} feature cannot be resumed"
                raise WorkflowConflictError(msg)
            try:
                validate_clarification_answers(record.state, answers)
            except ClarificationAnswerError as error:
                # The caller sent the wrong answers. The feature is untouched and still
                # answerable, so it must stay where it is rather than being marked failed and
                # made permanently unresumable by a correctable mistake.
                raise WorkflowConflictError(str(error)) from error
            await self._queue.requeue(
                feature_id=feature_id,
                execution_mode=record.state.execution_mode,
                agent_platform=record.state.agent_platform,
                performance_tier=record.state.performance_tier,
                model_setup_snapshot=record.state.model_setup_snapshot,
                model_setup_id=record.state.model_setup_id,
                requested_by=await self.owner_of(feature_id),
                intent=RESUME,
                payload={"answers": [item.model_dump(mode="json") for item in answers]},
            )
        if self._on_queued is not None:
            self._on_queued(feature_id)
        return await self.get_record(feature_id, scope=scope)

    async def publish_feature(
        self,
        feature_id: str,
        *,
        requested_by: str,
        reason: str,
        credentials: RequestScopedCredentials,
        scope: WorkspaceScope = _ANY_WORKSPACE,
    ) -> FeatureRecord:
        """Accept a person's decision to publish a feature that did not land, and queue it.

        Queued, for the reason a retry grant is: this runs `git push` and a provider call per
        eligible repository, and doing that on a request thread is what left AB-Feature-108
        with no terminal status when the socket went away.

        The refusal is synchronous, and stays that way. A feature with nothing publishable is
        answered as a conflict here -- before an entry exists -- rather than accepted and
        failed on a worker minutes later with nobody watching.
        """
        del credentials
        async with self._lock.hold(f"feature:run:{feature_id}"):
            record = await self.get_record(feature_id, scope=scope)
            _require_mutable_feature_identity(record.state)
            try:
                # Persisted state and one stat per repository; no workspace maintenance and no
                # credentials, exactly like the retry grant's own precondition above.
                require_publishable_feature(record.state)
            except FeatureWorkflowError as error:
                raise WorkflowConflictError(str(error)) from error
            # Raises if this feature's mode has no runner configured. Kept synchronous: there
            # is no point queueing a publication nothing can carry out.
            self._required_runner(record.state.execution_mode)
            await self._queue.requeue(
                feature_id=feature_id,
                execution_mode=record.state.execution_mode,
                agent_platform=record.state.agent_platform,
                performance_tier=record.state.performance_tier,
                model_setup_snapshot=record.state.model_setup_snapshot,
                model_setup_id=record.state.model_setup_id,
                # The queue's identity is whose stored credentials the worker resolves, and
                # that is the feature's *owner* -- read off the row, not the actor who pressed
                # the button. An administrator granting a publication on somebody's feature
                # would otherwise push to that person's repository with the administrator's
                # GitHub token.
                #
                # `requested_by` here is the audit answer to who decided this, and it travels
                # in the payload where it always has. Crossing the two killed
                # AB-Feature-111's granted retry: a queue entry named an account that never
                # existed, and the poisoned value survived the code fix.
                requested_by=await self.owner_of(feature_id),
                intent=PUBLISH,
                payload={"requested_by": requested_by, "reason": reason},
            )
        if self._on_queued is not None:
            self._on_queued(feature_id)
        return await self.get_record(feature_id, scope=scope)

    async def revise_feature(
        self,
        feature_id: str,
        *,
        request: str,
        requested_by: str,
        credentials: RequestScopedCredentials,
        scope: WorkspaceScope = _ANY_WORKSPACE,
    ) -> FeatureRecord:
        """Apply a person's post-completion change request and queue the revision run.

        The apply is synchronous, like a cancellation's, and that is load-bearing: every
        claim gate and the abandoned-run sweep treat `completed` as untouchable, so the
        feature must leave `completed` inside this request -- under the run lock -- for the
        queued claim to arrive on a feature the ordinary machinery will advance. What is
        queued is only the run; `begin_revision` is a pure state mutation.

        Refusals stay synchronous for `publish_feature`'s reason: a feature that is not
        completed, or that never shipped a pull request, is answered as a conflict here
        rather than accepted and dropped by a worker minutes later with nobody watching.
        """
        del credentials
        async with self._lock.hold(f"feature:run:{feature_id}"):
            record = await self.get_record(feature_id, scope=scope)
            _require_mutable_feature_identity(record.state)
            if record.state.status is not FeatureWorkflowStatus.COMPLETED:
                msg = (
                    "only a completed feature can be revised; this one is "
                    f"{record.state.status.value}"
                )
                raise WorkflowConflictError(msg)
            if not any(
                child.pull_request_artifact_id is not None
                for child in record.state.child_workflows.values()
            ):
                msg = "this feature has no published pull request to revise"
                raise WorkflowConflictError(msg)
            if not request.strip():
                msg = "a revision request must say what should change"
                raise WorkflowConflictError(msg)
            # Raises if this feature's mode has no runner configured. Kept synchronous: there
            # is no point re-opening a feature nothing can run.
            self._required_runner(record.state.execution_mode)
            state = begin_revision(
                record.state.model_copy(deep=True),
                request_text=request.strip(),
                requested_by=requested_by,
            )
            await self._replace_state(
                feature_id,
                state,
                "feature_revision_requested",
                {"revision": state.revision, "requested_by": requested_by},
            )
            await self._queue.requeue(
                feature_id=feature_id,
                execution_mode=state.execution_mode,
                agent_platform=state.agent_platform,
                performance_tier=state.performance_tier,
                model_setup_snapshot=state.model_setup_snapshot,
                model_setup_id=state.model_setup_id,
                # The queue's identity is whose stored credentials the worker resolves --
                # the feature's owner, read off the row. `requested_by` is the audit answer
                # and travels in the payload, as on every other human-initiated intent.
                requested_by=await self.owner_of(feature_id),
                intent=REVISE,
                payload={"requested_by": requested_by, "revision": state.revision},
            )
        if self._on_queued is not None:
            self._on_queued(feature_id)
        return await self.get_record(feature_id, scope=scope)

    async def retry_workstream(
        self,
        feature_id: str,
        repository_id: str,
        *,
        additional_attempts: int,
        requested_by: str,
        reason: str,
        credentials: RequestScopedCredentials,
        scope: WorkspaceScope = _ANY_WORKSPACE,
    ) -> FeatureRecord:
        """Grant one stopped repository more attempts and queue the attempt.

        This takes the same run lock as start and resume, so a grant cannot be arranged while
        the feature is already executing one. Like resume, the attempt itself is queued rather
        than run inside this request: a retry does everything a first attempt does -- clone,
        install, code, validate, review -- and doing that on a request thread is what left
        AB-Feature-108 with no terminal status when the socket went away.
        """
        del credentials
        async with self._lock.hold(f"feature:run:{feature_id}"):
            record = await self.get_record(feature_id, scope=scope)
            _require_mutable_feature_identity(record.state)
            try:
                # Before the runner, deliberately. These checks read persisted state and
                # nothing else, so a caller who named the wrong repository is told so without
                # a workspace being maintained or a provider client being built. Behind the
                # runner they were unreachable without credentials, and the resulting error
                # reached the generic handler -- which marked the feature failed_requires_human
                # and answered 200.
                require_retryable_workstream(
                    record.state,
                    repository_id=repository_id,
                    additional_attempts=additional_attempts,
                )
            except FeatureWorkflowError as error:
                raise WorkflowConflictError(str(error)) from error
            # Raises if this feature's mode has no runner configured. Kept synchronous: there
            # is no point queueing a grant nothing can carry out.
            self._required_runner(record.state.execution_mode)
            await self._queue.requeue(
                feature_id=feature_id,
                execution_mode=record.state.execution_mode,
                agent_platform=record.state.agent_platform,
                performance_tier=record.state.performance_tier,
                # Deliberately not `requested_by`. The queue's identity is whose stored
                # credentials the worker resolves; `requested_by` is the audit answer to "who
                # overrode the platform's decision", composed by the route as a sentence --
                # "somebody (via platform key)". Passing one as the other sent the worker
                # looking for credentials belonging to a label, and AB-Feature-111's granted
                # retry died on "needs provider credentials configured for
                # claude-autonomous-session (via platform key)". The audit author still
                # travels, in the payload below.
                #
                # The identity is now read off the feature rather than left to the entry's
                # existing value. Under one shared identity those were the same thing; with
                # isolated workspaces they are not, and an administrator granting a retry on
                # somebody's feature must buy the attempt with *that person's* key.
                model_setup_snapshot=record.state.model_setup_snapshot,
                model_setup_id=record.state.model_setup_id,
                requested_by=await self.owner_of(feature_id),
                intent=RETRY_WORKSTREAM,
                payload={
                    "repository_id": repository_id,
                    "additional_attempts": additional_attempts,
                    "requested_by": requested_by,
                    "reason": reason,
                },
            )
        if self._on_queued is not None:
            self._on_queued(feature_id)
        return await self.get_record(feature_id, scope=scope)

    async def answer_design_conflict(
        self,
        feature_id: str,
        conflict_id: str,
        *,
        verdict: str,
        decision: str,
        decided_by: str,
        additional_attempts: int,
        credentials: RequestScopedCredentials,
        scope: WorkspaceScope = _ANY_WORKSPACE,
    ) -> FeatureRecord:
        """Accept a design decision and queue the attempt it authorises.

        Queued rather than executed here, exactly as a retry grant is. The attempt a verdict
        unblocks does everything a first attempt does -- clone, install, code, validate,
        review -- and doing that on a request thread is what left AB-Feature-108 with no
        terminal status when the socket went away. What arrives synchronously is the refusal:
        a verdict this feature cannot act on is answered as a conflict, before anything is
        queued and before the decision is recorded.

        The verdict itself is written into the feature's artifact lineage by the queued step
        rather than here. That keeps one writer for it -- the step that then runs the attempt
        bound by it -- so there is no window in which a decision is recorded and the run it
        authorised never happened.
        """
        del credentials
        async with self._lock.hold(f"feature:run:{feature_id}"):
            record = await self.get_record(feature_id, scope=scope)
            _require_mutable_feature_identity(record.state)
            if verdict not in VERDICTS:
                # The request's own shape first: a caller who sent a verdict this platform does
                # not have is told that, rather than told about a budget.
                msg = f"{verdict!r} is not a design verdict; expected one of {list(VERDICTS)}"
                raise WorkflowConflictError(msg)
            try:
                # Before the runner, for the reason the retry grant's checks are: these read
                # persisted state only, so a caller naming a decided or unknown question is
                # told so without a workspace being maintained or a provider client built.
                conflict = find_design_conflict(record.state.artifacts, conflict_id)
                require_answerable_design_conflict(
                    record.state,
                    conflict=conflict,
                    additional_attempts=additional_attempts,
                )
            except DesignConflictNotFoundError as error:
                raise WorkflowNotFoundError(str(error)) from error
            except FeatureWorkflowError as error:
                raise WorkflowConflictError(str(error)) from error
            # Raises if this feature's mode has no runner configured. Kept synchronous: there
            # is no point queueing a decision nothing can act on.
            self._required_runner(record.state.execution_mode)
            await self._queue.requeue(
                feature_id=feature_id,
                execution_mode=record.state.execution_mode,
                agent_platform=record.state.agent_platform,
                performance_tier=record.state.performance_tier,
                # The queue's identity is whose stored credentials the worker resolves, and
                # that is the feature's owner, read off the row. `decided_by` is the audit
                # answer to who decided the question, and travels in the payload. Crossing
                # the two is what killed AB-Feature-111's granted retry.
                model_setup_snapshot=record.state.model_setup_snapshot,
                model_setup_id=record.state.model_setup_id,
                requested_by=await self.owner_of(feature_id),
                intent=ANSWER_DESIGN_VERDICT,
                payload={
                    "conflict_id": conflict_id,
                    "repository_id": conflict.repository_id,
                    "verdict": verdict,
                    "decision": decision,
                    "decided_by": decided_by,
                    "additional_attempts": additional_attempts,
                },
            )
        if self._on_queued is not None:
            self._on_queued(feature_id)
        return await self.get_record(feature_id, scope=scope)

    async def cancel(
        self,
        feature_id: str,
        *,
        requested_by: str,
        reason: str | None = None,
        scope: WorkspaceScope = _ANY_WORKSPACE,
    ) -> FeatureRecord:
        """Record cancellation without destroying parent, child, or pull-request audit records.

        `requested_by` is required rather than defaulted to `"api"`. It is written into the
        cancellation reason and into `state.cancellation_requested_by`, so a default made
        every cancellation somebody could not be named for look like one nobody asked for.
        Nothing enqueues here, so this identity is purely the audit answer.
        """
        async with self._lock.hold(f"feature:cancel:{feature_id}"):
            record = await self.get_record(feature_id, scope=scope)
            _require_mutable_feature_identity(record.state)
            if record.state.status in {
                FeatureWorkflowStatus.COMPLETED,
                FeatureWorkflowStatus.CANCELLED,
                FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
            }:
                return record
            state = record.state.model_copy(deep=True)
            transition_feature(
                state,
                FeatureWorkflowStatus.CANCELLING,
                reason=(reason or f"{requested_by} asked for this feature to be cancelled."),
                agent="api",
            )
            state.cancellation_requested = True
            state.cancellation_status = CancellationLifecycleStatus.CANCELLATION_REQUESTED
            state.cancellation_requested_at = datetime.now(UTC)
            state.cancellation_requested_by = requested_by
            state.cancellation_reason = reason
            state.current_agent = "api"
            state.updated_at = datetime.now(UTC)
            await self._replace_state(
                feature_id,
                state,
                "feature_cancellation_requested",
                preserve_persisted_progress=True,
            )
            await self._request_operation_cancellation(feature_id)
            if self._cancellation_signaler is not None:
                await self._cancellation_signaler(feature_id)
            if self._operation_journal is None or not await self._has_active_operations(feature_id):
                await self._finish_cancellation(feature_id)
            return await self.get_record(feature_id, scope=scope)

    async def retire(
        self,
        feature_id: str,
        *,
        reason: str,
        operator: str,
        scope: WorkspaceScope = _ANY_WORKSPACE,
    ) -> FeatureRecord:
        """Record an operator's decision to stop tracking a feature that cannot progress.

        Deliberately not a lifecycle write, and deliberately not guarded by mutable identity.
        A snapshot migrated from before build provenance existed is audit-only: the platform
        refuses to resume or rewrite it, correctly, because its recorded identity is not
        evidence of what produced it. That left the two workflows the audit named stuck in
        `running_child_workflows` and `waiting_for_human` with no path to resolve them except
        editing rows by hand, which is what an operator API exists to avoid.

        Nothing here executes: no orchestration, no provider call, no repository touched. It
        writes a terminal status and an event naming who decided and why, and every artifact,
        child record, pull request and journal row is left exactly where it was.

        One exception, added with attachments and stated here because this docstring used to
        promise the opposite: the *content* of a submitted image is purged. The row is not --
        `purged_at` is set and `content` goes null -- so the PRD artifact that quoted the
        filename, the size and the sha256 still resolves, and a reader following that
        reference is told the bytes are gone rather than that they never existed. The
        exception exists because attachment content is the only thing a retired feature holds
        that grows without bound and that nobody will ever look at again; every other record
        here is audit evidence, which is exactly why it stays.
        """
        if not reason.strip() or not operator.strip():
            msg = "retiring a feature requires both an operator and a reason"
            raise WorkflowConflictError(msg)
        async with self._lock.hold(f"feature:retire:{feature_id}"):
            if self._operation_journal is not None and await self._has_active_operations(
                feature_id
            ):
                msg = (
                    "feature still has active external operations; cancel it instead so those "
                    "operations are given the chance to stop"
                )
                raise WorkflowConflictError(msg)
            async with self._database.session() as session:
                model = await self._require_model(session, feature_id, scope=scope)
                previous = str(model.status)
                if previous in _RETIRED_EQUIVALENT_STATUSES:
                    # Already terminal: there is no transition to write, and the operator
                    # asked for a retirement all the same. The purge runs anyway, because
                    # "this feature is finished with" is exactly what they said and the
                    # bytes are the one thing here that costs anything to keep. Idempotent,
                    # so a second retire of an already-purged feature does nothing.
                    settled = await self._record_from_model(session, model)
                    await self._purge_attachment_content(feature_id)
                    return settled
                now = datetime.now(UTC)
                state_json = transition_feature_state_json(
                    dict(model.state_json),
                    FeatureWorkflowStatus.CANCELLED,
                    reason=f"{operator} retired this feature: {reason}",
                    agent="operator",
                )
                state_json["updated_at"] = now.isoformat()
                # The column mirrors the snapshot rather than deciding anything: a read
                # rebuilds the record from `state_json`, so writing one alone retires nothing.
                model.status = FeatureWorkflowStatus(str(state_json["status"]))
                model.updated_at = now
                model.state_json = state_json
                self._add_event(
                    session,
                    feature_id,
                    "feature_retired_by_operator",
                    {"operator": operator, "reason": reason, "previous_status": previous},
                )
                await session.commit()
            # After the commit, not inside it. Retiring is the decision; purging the bytes is
            # a consequence of it, and a purge that failed must not leave a feature the
            # operator was told is retired still claiming to be running. The sweep is not a
            # backstop here -- these rows are bound -- so a failure is logged loudly and the
            # bytes stay until somebody retires the feature again.
            await self._purge_attachment_content(feature_id)
            return await self.get_record(feature_id, scope=scope)

    async def reconcile_abandoned_runs(
        self,
        *,
        stale_after_seconds: float,
        limit: int = 50,
    ) -> list[str]:
        """Give a terminal status to features whose executor died without writing one.

        Every other durable thing this platform runs is swept: external operations settle,
        action leases lapse and are reconciled. A feature's own status was not, and the gap
        is not theoretical. AB-Feature-108 sat in `running_child_workflows` with a `running`
        workstream for over a day: its action had already been reconciled
        `interrupted_before_completion`, the process was gone, and nothing existed whose job
        it was to notice that the feature it had been running still claimed to be running.
        An operator could only see a live feature that was making no progress, and the one
        remedy was to retire it by hand.

        Conservative on purpose, because a false positive kills work that is fine:
        - only statuses that assert something is executing right now,
        - only after a grace period longer than the slowest single step,
        - and only when no live action lease and no live queue lease claims the feature.

        Returns the features it decided, so a caller can log the count.
        """
        if stale_after_seconds <= 0:
            msg = "stale_after_seconds must be positive"
            raise ValueError(msg)
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=stale_after_seconds)
        async with self._database.session() as session:
            # A subquery, rather than a join: a feature is owned if *any* live lease names
            # it, and an outer join would multiply rows per action. The definition of "owned"
            # is shared with the claim decision, which asks the same question about one
            # feature -- see `_executing_feature_ids`.
            candidates = list(
                await session.scalars(
                    select(FeatureWorkflowModel.feature_id)
                    .where(FeatureWorkflowModel.status.in_(_ACTIVELY_RUNNING_STATUSES))
                    .where(FeatureWorkflowModel.updated_at < cutoff)
                    .where(
                        FeatureWorkflowModel.feature_id.not_in(
                            _executing_feature_ids(now).scalar_subquery()
                        )
                    )
                    .order_by(FeatureWorkflowModel.updated_at)
                    .limit(limit)
                )
            )
        reconciled: list[str] = []
        for feature_id in candidates:
            try:
                await self._reconcile_abandoned_run(feature_id, cutoff=cutoff)
            except (WorkflowNotFoundError, WorkflowConflictError):
                # Cancelled, retired or resumed between the scan and the write. The next
                # sweep re-reads the truth; nothing here is worth failing the pass for.
                continue
            reconciled.append(feature_id)
        return reconciled

    async def stop_overrunning_runs(
        self,
        *,
        runtime_limit_seconds: float,
        limit: int = 50,
    ) -> list[str]:
        """Stop features that have been executing longer than this deployment allows.

        A different question from `reconcile_abandoned_runs` above, and the two must not be
        collapsed. That one asks whether an executor has stopped *writing*; this one asks
        how long the feature has been running at all. A feature checkpointing every few
        minutes passes the first test indefinitely -- the longest run recorded here reached
        571 minutes, entirely inside one coroutine, renewing its lease the whole way.

        Clocked from the queue entry's `started_at`, which is when a worker claimed the
        feature, so a submission that waited an hour to be picked up is not charged for the
        wait. A feature with no queue entry -- an execution this platform never scheduled --
        is left alone rather than measured against a timestamp that means something else.

        Scoped to `_ESCALATABLE_STATUSES` rather than every actively-running status, for the
        same reason unconfirmed effects are: `cancelling` is already ending, and converting a
        cancellation in progress into a runtime failure would discard the cleanup
        requirements the cancellation is recording. Resting statuses are excluded by
        construction -- `waiting_for_human` can legitimately last days, and is not in this
        set at all.

        Stops scheduling only. Nothing here cancels an operation, kills a process or touches
        the operation journal; the executing process discovers it lost its claim the next
        time it writes.
        """
        if runtime_limit_seconds <= 0:
            return []
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=runtime_limit_seconds)
        async with self._database.session() as session:
            overrunning_claims = (
                select(FeatureExecutionQueueModel.feature_id)
                .where(FeatureExecutionQueueModel.started_at.is_not(None))
                .where(FeatureExecutionQueueModel.started_at < cutoff)
                .scalar_subquery()
            )
            candidates = list(
                await session.scalars(
                    select(FeatureWorkflowModel.feature_id)
                    .where(
                        FeatureWorkflowModel.status.in_(
                            {item.value for item in _ESCALATABLE_STATUSES}
                        )
                    )
                    .where(FeatureWorkflowModel.feature_id.in_(overrunning_claims))
                    .order_by(FeatureWorkflowModel.updated_at)
                    .limit(limit)
                )
            )
        stopped: list[str] = []
        for feature_id in candidates:
            try:
                if await self._stop_overrunning_run(
                    feature_id, limit_seconds=runtime_limit_seconds, cutoff=cutoff
                ):
                    stopped.append(feature_id)
            except (WorkflowNotFoundError, WorkflowConflictError):
                # Cancelled, retired or finished between the scan and the write. The next
                # sweep re-reads the truth; nothing here is worth failing the pass for.
                continue
        return stopped

    async def _stop_overrunning_run(
        self, feature_id: str, *, limit_seconds: float, cutoff: datetime
    ) -> bool:
        """Write one overrunning feature's terminal status under the ordinary state lock."""
        record = await self.get_record(feature_id)
        state = record.state.model_copy(deep=True)
        if state.status not in _ESCALATABLE_STATUSES:
            # Re-checked under the lock `_replace_state` takes: the scan above is
            # unsynchronized, so a feature that finished in between must not be stopped.
            return False
        started_at = await self._claim_started_at(feature_id)
        if started_at is None or started_at >= cutoff:
            # Re-claimed, or its entry finished and a new one began, between the scan and
            # here. The clock restarted with it.
            return False
        elapsed_minutes = round((datetime.now(UTC) - started_at).total_seconds() / 60)
        limit_minutes = round(limit_seconds / 60)
        previous_status = state.status.value
        unfinished = sorted(
            child.repository_id
            for child in state.child_workflows.values()
            if child.status in _UNFINISHED_CHILD_STATUSES
        )
        transition_feature(
            state,
            FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
            reason=(
                f"This feature executed for about {elapsed_minutes} minutes, past the "
                f"{limit_minutes} minutes this deployment allows one run."
            ),
            agent="feature_runtime",
        )
        state.updated_at = datetime.now(UTC)
        state = ensure_feature_failure_summary(
            state,
            stage=FailureStage.FEATURE_RUNTIME,
            classification=FeatureFailureClassification.FEATURE_RUNTIME_LIMIT_REACHED,
            diagnostics=[
                f"This feature executed for about {elapsed_minutes} minutes, past the "
                f"{limit_minutes}-minute ceiling this deployment sets for a single feature.",
                f"It was still in {previous_status} when it was stopped, so it may have been "
                "progressing; what ran out was time, not attempts.",
                *(
                    [f"Workstreams still running when it was stopped: {', '.join(unfinished)}."]
                    if unfinished
                    else []
                ),
                "Nothing was interrupted. Any push or pull request this run had already "
                "started is recorded in the operation journal and settles on its own; check "
                "there before starting the work again.",
            ],
        )
        await self._replace_state(
            feature_id,
            state,
            "feature_runtime_limit_reached",
            {
                "previous_status": previous_status,
                "elapsed_minutes": elapsed_minutes,
                "limit_minutes": limit_minutes,
            },
        )
        return True

    async def _claim_attempts(self, feature_id: str) -> tuple[int, int] | None:
        """Read what this feature's queue entry has spent and what it was given.

        `None` when the platform has no entry for this feature at all -- an execution it
        never scheduled -- because measuring such a feature against a budget it was never
        issued would stop a continuation for a reason that does not exist.
        """
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(
                        FeatureExecutionQueueModel.attempt,
                        FeatureExecutionQueueModel.max_attempts,
                    ).where(FeatureExecutionQueueModel.feature_id == feature_id)
                )
            ).first()
        return None if row is None else (int(row[0]), int(row[1]))

    async def _continuation_count(self, feature_id: str) -> int:
        """Count how many times a worker picked this run back up rather than ending it.

        Read off the timeline rather than off the queue entry's attempt counter: an attempt
        can also be spent on a provider fault, and a terminal record that says "continued
        three times" when two of those were something else is a diagnostic that misleads.
        """
        async with self._database.session() as session:
            count = await session.scalar(
                select(func.count())
                .select_from(FeatureWorkflowEventModel)
                .where(FeatureWorkflowEventModel.feature_id == feature_id)
                .where(FeatureWorkflowEventModel.event == "feature_run_continued")
            )
        return int(count or 0)

    async def _claim_started_at(self, feature_id: str) -> datetime | None:
        """Read when a worker claimed this feature, which is when its runtime clock starts."""
        async with self._database.session() as session:
            started_at = await session.scalar(
                select(FeatureExecutionQueueModel.started_at).where(
                    FeatureExecutionQueueModel.feature_id == feature_id
                )
            )
        if started_at is None:
            return None
        # SQLite hands back naive datetimes for timezone-aware columns.
        return started_at if started_at.tzinfo else started_at.replace(tzinfo=UTC)

    async def flag_unconfirmed_external_effect(
        self,
        feature_id: str,
        *,
        operation_type: str,
        repository_id: str | None,
        reason: str,
        error_code: str | None = None,
    ) -> bool:
        """Stop a running feature from publishing over an effect nobody has confirmed.

        Readiness no longer refuses the whole deployment when one operation is unresolved,
        which is right -- an interrupted push on one feature was never a reason to reject
        every other account's writes. The condition still has to land somewhere, and the
        feature that owns the operation is the only place it actually means anything.

        ``error_code`` says which of two very different things happened. ``recovery_error``
        is written when the recovery sweep's *own* code raised while reconciling: nothing is
        known about the provider, and the operation was stamped for manual review by a
        handler catching a defect in this repository. Escalating that as an unconfirmed
        effect asked the feature's owner to go and inspect their repository because the
        platform's sweep threw. It is recorded as ``PLATFORM_DEFECT`` and says so.

        Returns whether this call was the one that flagged it, so a repeated sweep over the
        same operation is silent rather than writing the same event again.
        """
        try:
            record = await self.get_record(feature_id)
        except WorkflowNotFoundError:
            return False
        if record.state.status not in _ESCALATABLE_STATUSES:
            # Already finished, already cancelled, or already waiting on a person. A
            # cancellation that retained side effects records them as cleanup requirements
            # and must not be reopened as a failure.
            return False
        state = record.state.model_copy(deep=True)
        previous_status = state.status.value
        readable_type = operation_type.replace("_", " ")
        platform_defect = error_code == _RECOVERY_DEFECT_ERROR_CODE
        if platform_defect:
            classification = FeatureFailureClassification.PLATFORM_DEFECT
            blocking_issue = (
                f"The platform's recovery sweep failed while reconciling a {readable_type} "
                "for this repository. This is a defect in the platform, not in this "
                "repository."
            )
            diagnostics = [
                f"The platform's own recovery pass raised while reconciling a "
                f"{readable_type}, so it recorded the operation as needing manual review.",
                *([f"Repository: {repository_id}."] if repository_id else []),
                "The platform failed here. Nothing is known about the provider either way, "
                "so the operation still has to be checked -- but the reason this feature "
                "stopped is ours to fix.",
            ]
        else:
            classification = FeatureFailureClassification.UNCONFIRMED_EXTERNAL_EFFECT
            blocking_issue = (
                f"A {readable_type} for this repository was interrupted and its effect on "
                "the provider could not be confirmed. Check the repository before this work "
                "is attempted again."
            )
            diagnostics = [
                f"An interrupted {readable_type} could not be confirmed against the provider.",
                *([f"Repository: {repository_id}."] if repository_id else []),
                reason,
                "Check the repository for a branch, pull request or comment this run may "
                "have left behind before starting the work again.",
            ]
        for key, child in list(state.child_workflows.items()):
            if repository_id is not None and child.repository_id != repository_id:
                continue
            state.child_workflows[key] = child.model_copy(
                update={"blocking_issues": [*child.blocking_issues, blocking_issue]}
            )
        transition_feature(
            state,
            FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
            reason=blocking_issue,
            agent="operation_recovery",
        )
        state.updated_at = datetime.now(UTC)
        state.failure_summary = None
        state = ensure_feature_failure_summary(
            state,
            stage=FailureStage.OPERATION_RECOVERY,
            classification=classification,
            diagnostics=diagnostics,
        )
        try:
            await self._replace_state(
                feature_id,
                state,
                "unconfirmed_external_effect",
                {
                    "previous_status": previous_status,
                    "operation_type": operation_type,
                    "repository_id": repository_id,
                },
            )
        except (WorkflowNotFoundError, WorkflowConflictError):
            # Resumed, cancelled or retired between the read and the write. The next sweep
            # re-reads the truth; nothing here is worth failing the pass for.
            return False
        return True

    async def _reconcile_abandoned_run(self, feature_id: str, *, cutoff: datetime) -> None:
        """Write one abandoned feature's terminal status under the ordinary state lock."""
        record = await self.get_record(feature_id)
        state = record.state.model_copy(deep=True)
        # Re-checked under the lock `_replace_state` takes: the scan above is unsynchronized,
        # so a run that resumed in between must not be declared dead by this pass.
        if state.status.value not in _ACTIVELY_RUNNING_STATUSES or state.updated_at >= cutoff:
            return
        # And re-checked against the same evidence the scan used, because a crashed feature
        # is now something a worker *wants*. The scan and this write are two statements apart
        # and a claim can land between them; tombstoning a feature a worker is continuing
        # would put two writers on one terminal status. The claim's own half of this fence is
        # `_claim_disposition`, which continues only an actively-running feature and so
        # declines the moment a status is written here.
        if not await self._nothing_is_executing(feature_id, holding_queue_claim=False):
            return
        stranded = [
            child.repository_id
            for child in state.child_workflows.values()
            if child.status in _UNFINISHED_CHILD_STATUSES
        ]
        for key, child in list(state.child_workflows.items()):
            if child.status not in _UNFINISHED_CHILD_STATUSES:
                continue
            state.child_workflows[key] = child.model_copy(
                update={
                    "status": ChildWorkflowStatus.FAILED,
                    "blocking_issues": [
                        *child.blocking_issues,
                        (
                            "The process running this workstream stopped without recording an "
                            "outcome. Whatever it had reached is in its artifacts; anything it "
                            "had not yet written is lost."
                        ),
                    ],
                }
            )
        previous_status = state.status.value
        transition_feature(
            state,
            FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
            reason=(
                f"The process executing this feature stopped in {previous_status} without "
                "recording an outcome, and nothing has claimed it since."
            ),
            agent="run_recovery",
        )
        state.updated_at = datetime.now(UTC)
        # "Start a fresh feature rather than resuming this one" was the right advice while
        # nothing continued a crashed run. It is not any more: a worker picks one back up from
        # its checkpoint, and this status is what is left after that ran out of attempts. The
        # diagnostic now says which of the two actually happened.
        continuations = await self._continuation_count(feature_id)
        state = ensure_feature_failure_summary(
            state,
            stage=FailureStage.RUN_RECOVERY,
            classification=FeatureFailureClassification.EXECUTOR_STOPPED,
            diagnostics=[
                "The executor holding this feature stopped without writing a terminal status.",
                f"It was last making progress in {previous_status}.",
                *(
                    [f"Workstreams left mid-flight: {', '.join(sorted(stranded))}."]
                    if stranded
                    else []
                ),
                # One sentence, not two: the summary keeps five diagnostics and the failing
                # repository's own blockers have to fit beside these.
                (
                    (
                        "The platform continued this run from its last checkpoint "
                        f"{continuations} more "
                        f"{'time' if continuations == 1 else 'times'} and it did not "
                        "survive, so its queue entry has no attempts left."
                        if continuations
                        else "No worker reached this run to continue it before its state "
                        "went stale, so nothing was retried and nothing was repeated."
                    )
                    + " Its workspace and artifacts are intact, and any remote effect it "
                    "recorded is in the operation journal; check there before this work is "
                    "started again."
                ),
            ],
        )
        await self._replace_state(
            feature_id,
            state,
            "feature_run_abandoned",
            {
                "previous_status": previous_status,
                "stranded_workstreams": sorted(stranded),
                "continuations": continuations,
            },
        )

    async def checkpoint(
        self,
        state: FeatureWorkflowSnapshot,
        boundary: WorkflowCheckpointBoundary,
        repository_id: str | None = None,
    ) -> None:
        """Persist a child or parent safe boundary without waiting for the whole live request."""
        _require_mutable_feature_identity(state)
        await self._replace_state(
            state.feature_id,
            state,
            f"feature_checkpoint_{boundary.value}",
            {"repository_id": repository_id} if repository_id is not None else {},
            checkpoint_boundary=boundary,
            checkpoint_repository_id=repository_id,
        )

    async def approve_contract_change(
        self,
        feature_id: str,
        *,
        request_id: str,
        revision: IntegrationContractRevision,
        resolution: str,
        credentials: RequestScopedCredentials,
        scope: WorkspaceScope = _ANY_WORKSPACE,
    ) -> FeatureRecord:
        """Apply a human-approved next contract revision and persist only its non-secret data."""
        async with self._lock.hold(f"feature:{feature_id}"):
            record = await self.get_record(feature_id, scope=scope)
            _require_mutable_feature_identity(record.state)
            runner = self._required_runner(record.state.execution_mode)
            contract = _contract_revision_artifact(record.state, revision)
            updated = await runner.approve_contract_change(
                record.state,
                request_id=request_id,
                updated_contract=contract,
                resolution=resolution,
                credentials=credentials,
            )
            await self._replace_state(feature_id, updated, _event_for_status(updated.status))
            return await self.get_record(feature_id, scope=scope)

    async def reject_contract_change(
        self,
        feature_id: str,
        *,
        request_id: str,
        resolution: str,
        credentials: RequestScopedCredentials,
        scope: WorkspaceScope = _ANY_WORKSPACE,
    ) -> FeatureRecord:
        """Persist an explicit human rejection and leave the failed feature fully auditable."""
        async with self._lock.hold(f"feature:{feature_id}"):
            record = await self.get_record(feature_id, scope=scope)
            _require_mutable_feature_identity(record.state)
            runner = self._required_runner(record.state.execution_mode)
            updated = await runner.reject_contract_change(
                record.state,
                request_id=request_id,
                resolution=resolution,
                credentials=credentials,
            )
            await self._replace_state(feature_id, updated, _event_for_status(updated.status))
            return await self.get_record(feature_id, scope=scope)

    async def approve_repair(
        self,
        feature_id: str,
        *,
        repair_id: str,
        actor_id: str,
        credentials: RequestScopedCredentials,
        scope: WorkspaceScope = _ANY_WORKSPACE,
    ) -> FeatureRecord:
        """Apply an approved repair and run the repository it was blocking.

        Held under the same run lock as start, resume and a retry grant. An approval starts a
        real attempt, and two of them beginning at once would have the repository being
        cloned and written twice.
        """
        async with self._lock.hold(f"feature:run:{feature_id}"):
            record = await self.get_record(feature_id, scope=scope)
            _require_mutable_feature_identity(record.state)
            runner = self._required_runner(record.state.execution_mode)
            try:
                updated = await runner.approve_repository_repair(
                    record.state,
                    repair_id=repair_id,
                    actor_id=actor_id,
                    credentials=credentials,
                )
            except RepairNotFoundError as error:
                raise WorkflowNotFoundError(str(error)) from error
            except CancellationRequested:
                await self._finish_cancellation(feature_id)
                return await self.get_record(feature_id, scope=scope)
            except RepairSupersededError as error:
                # The refusal carries a state change: this repair is now recorded as stale.
                # Persisting it before reporting the conflict is what stops the same
                # out-of-date proposal being offered for approval again.
                await self._replace_state(
                    feature_id,
                    error.state,
                    "repository_repair_superseded",
                    {"actor_id": actor_id, "repair_id": repair_id},
                )
                raise WorkflowConflictError(str(error)) from error
            except FeatureWorkflowError as error:
                # A stale, already-decided or unknown repair is a refusal about the request,
                # not a fault in the feature. Marking the feature failed for one would break
                # a repository that is still perfectly retryable.
                raise WorkflowConflictError(str(error)) from error
            except (AgentArtifactError, ValidationError) as error:
                await self._mark_failed_requires_human(
                    feature_id, "repository_repair_failed", error, current_agent="feature_runtime"
                )
                raise _operation_did_not_happen("apply this repair") from error
            except Exception as error:
                await self._mark_failed_requires_human(
                    feature_id, "repository_repair_failed", error, current_agent="feature_runtime"
                )
                raise _operation_did_not_happen("apply this repair") from error
            # Who approved it travels with the event, not only on the action record: a
            # timeline is where somebody looks first, and "a repair was applied" without
            # a name is the audit gap this work exists to close.
            await self._replace_state(
                feature_id,
                updated,
                "repository_repair_completed",
                {"actor_id": actor_id, "repair_id": repair_id},
            )
            return await self.get_record(feature_id, scope=scope)

    async def reject_repair(
        self,
        feature_id: str,
        *,
        repair_id: str,
        actor_id: str,
        reason: str,
        scope: WorkspaceScope = _ANY_WORKSPACE,
    ) -> FeatureRecord:
        """Record a declined repair. Nothing executes, so this takes only the state lock."""
        async with self._lock.hold(f"feature:{feature_id}"):
            record = await self.get_record(feature_id, scope=scope)
            _require_mutable_feature_identity(record.state)
            runner = self._required_runner(record.state.execution_mode)
            try:
                updated = await runner.reject_repository_repair(
                    record.state, repair_id=repair_id, actor_id=actor_id, reason=reason
                )
            except RepairNotFoundError as error:
                raise WorkflowNotFoundError(str(error)) from error
            except FeatureWorkflowError as error:
                raise WorkflowConflictError(str(error)) from error
            await self._replace_state(
                feature_id,
                updated,
                "repository_repair_rejected",
                {"actor_id": actor_id, "repair_id": repair_id},
            )
            return await self.get_record(feature_id, scope=scope)

    async def repairs(
        self, feature_id: str, *, scope: WorkspaceScope = _ANY_WORKSPACE
    ) -> list[RepositoryRepairProposalArtifact]:
        """Return each repair's current state, read from the indexed decision table.

        Read from the projection rather than from parent state on purpose: a completed
        feature's snapshot carries every artifact it produced, and asking "what is waiting on
        me for this repository" must not cost that.
        """
        statement = (
            select(RepositoryRepairModel)
            .where(RepositoryRepairModel.feature_id == feature_id)
            .order_by(RepositoryRepairModel.repository_id, RepositoryRepairModel.repair_id)
        )
        async with self._database.session() as session:
            await self._require_model(session, feature_id, scope=scope)
            rows = (await session.execute(statement)).scalars().all()
        return [
            RepositoryRepairProposalArtifact.model_validate_json(json.dumps(row.payload))
            for row in rows
        ]

    async def get_record(
        self, feature_id: str, *, scope: WorkspaceScope = _ANY_WORKSPACE
    ) -> FeatureRecord:
        """Hydrate parent state from durable generic and independently indexed child records.

        Every mutation on this class begins here, which is what makes one ownership check at
        `_require_model` cover all of them.
        """
        async with self._database.session() as session:
            model = await self._require_model(session, feature_id, scope=scope)
            return await self._record_from_model(session, model)

    async def artifacts(
        self, feature_id: str, *, scope: WorkspaceScope = _ANY_WORKSPACE
    ) -> list[Artifact]:
        """Read versioned parent and child artifacts in their persisted emission order."""
        return list((await self.get_record(feature_id, scope=scope)).state.artifacts)

    async def workstreams(
        self, feature_id: str, *, scope: WorkspaceScope = _ANY_WORKSPACE
    ) -> list[ChildWorkflowReference]:
        """Expose isolated workstream status rows in repository submission order."""
        record = await self.get_record(feature_id, scope=scope)
        return list(record.state.child_workflows.values())

    async def pull_requests(
        self, feature_id: str, *, scope: WorkspaceScope = _ANY_WORKSPACE
    ) -> list[PullRequestArtifact]:
        """Return PR artifacts so partial creation remains visible after a later failure.

        One entry per provider pull request, in its latest recorded state: a revision closes
        the pull request it superseded by appending a `.closed` copy of its artifact, and
        serving both would show the same pull request twice -- once falsely open. History is
        append-only and ordered, so the last artifact carrying a number is its current state.
        """
        latest: dict[tuple[str, int], PullRequestArtifact] = {}
        for item in await self.artifacts(feature_id, scope=scope):
            if isinstance(item, PullRequestArtifact):
                latest[(item.repository, item.pull_request_number)] = item
        return list(latest.values())

    async def events_after(
        self,
        feature_id: str,
        *,
        after_id: int | None,
        limit: int,
        scope: WorkspaceScope = _ANY_WORKSPACE,
    ) -> list[FeatureEvent]:
        """Read lifecycle events after a cursor without hydrating parent state.

        This is the query a client polls for liveness, so it touches only the event table,
        which is indexed on `feature_id`. Reading the record instead would deserialise every
        artifact the feature has produced on every poll.

        The ownership check is `require_visible` rather than the record load, for that same
        reason -- one indexed primary-key read against the same column, which keeps the poll
        cheap and still answers 404 for somebody else's feature.
        """
        statement = (
            select(FeatureWorkflowEventModel)
            .where(FeatureWorkflowEventModel.feature_id == feature_id)
            .order_by(FeatureWorkflowEventModel.id)
            .limit(limit)
        )
        if after_id is not None:
            statement = statement.where(FeatureWorkflowEventModel.id > after_id)
        async with self._database.session() as session:
            await self._require_model(session, feature_id, scope=scope)
            rows = (await session.execute(statement)).scalars().all()
        return [
            FeatureEvent(
                id=row.id,
                timestamp=_as_utc(row.timestamp),
                event_type=row.event_type,
                source=row.source,
                event=row.event,
                details=dict(row.details or {}),
            )
            for row in rows
        ]

    async def timeline(
        self, feature_id: str, *, scope: WorkspaceScope = _ANY_WORKSPACE
    ) -> list[tuple[datetime, str, str, str, dict[str, Any]]]:
        """Build a safe timeline from durable lifecycle and artifact records."""
        record = await self.get_record(feature_id, scope=scope)
        events: list[tuple[datetime, str, str, str, dict[str, Any]]] = [
            (item.timestamp, "lifecycle", "api", item.event, item.details)
            for item in record.lifecycle_events
        ]
        events.extend(
            (
                item.timestamp,
                "artifact",
                item.producer,
                item.artifact_type,
                {"artifact_id": item.artifact_id},
            )
            for item in record.state.artifacts
        )
        return sorted(events, key=lambda item: item[0])

    async def list_features(
        self, *, limit: int, cursor: str | None, scope: WorkspaceScope = _ANY_WORKSPACE
    ) -> FeaturePage:
        """Page features newest first, reading only indexed columns.

        Keyset rather than offset: features are written while somebody is paging, and an
        offset would silently skip or repeat rows as newer ones arrive. One extra row is
        fetched to decide whether a next page exists without a second count query.

        The owner filter and the cursor predicate are both in the same `WHERE`, which is
        what a keyset cursor requires: the cursor names a position in *this* caller's
        ordering, so narrowing after interpreting it would leave holes in every page. The
        composite index `(owner_id, created_at DESC, feature_id DESC)` matches this shape
        exactly.
        """
        statement = select(
            FeatureWorkflowModel.feature_id,
            FeatureWorkflowModel.title,
            FeatureWorkflowModel.reference,
            FeatureWorkflowModel.status,
            FeatureWorkflowModel.execution_mode,
            FeatureWorkflowModel.agent_platform,
            FeatureWorkflowModel.performance_tier,
            FeatureWorkflowModel.created_at,
            FeatureWorkflowModel.updated_at,
        ).order_by(
            FeatureWorkflowModel.created_at.desc(),
            FeatureWorkflowModel.feature_id.desc(),
        )
        if scope.applies():
            statement = statement.where(FeatureWorkflowModel.owner_id == scope.owner_id)
        if cursor is not None:
            created_at, feature_id = decode_feature_cursor(cursor)
            statement = statement.where(
                tuple_(FeatureWorkflowModel.created_at, FeatureWorkflowModel.feature_id)
                < (created_at, feature_id)
            )
        async with self._database.session() as session:
            rows = (await session.execute(statement.limit(limit + 1))).all()
            page = rows[:limit]
            identifiers = [row.feature_id for row in page]
            # Two grouped counts scoped to this page only. Both tables are indexed on
            # feature_id and neither touches `state_json`, so a listing stays cheap; the
            # alternative, hydrating parent state per row, is what this method exists to avoid.
            repository_counts = await _counts_by_feature(
                session, RepositorySpecModel.feature_id, identifiers
            )
            pull_request_counts = await _counts_by_feature(
                session, FeaturePullRequestModel.feature_id, identifiers
            )
        summaries = [
            FeatureSummary(
                feature_id=row.feature_id,
                # Parent feature workflows use the feature id as their workflow identity.
                # This is established when the state is created and is also the durable row key.
                workflow_id=row.feature_id,
                title=row.title,
                reference=row.reference,
                status=row.status,
                execution_mode=row.execution_mode,
                agent_platform=row.agent_platform,
                performance_tier=row.performance_tier,
                created_at=_as_utc(row.created_at),
                updated_at=_as_utc(row.updated_at),
                repository_count=repository_counts.get(row.feature_id, 0),
                pull_request_count=pull_request_counts.get(row.feature_id, 0),
                human_action_required=feature_awaits_human(row.status),
                dashboard_group=feature_dashboard_group(row.status),
            )
            for row in page
        ]
        return FeaturePage(
            features=summaries,
            next_cursor=encode_feature_cursor(summaries[-1]) if len(rows) > limit else None,
        )

    async def replayed_feature_id(
        self,
        request: StartFeatureRequest,
        *,
        idempotency_key: str | None,
        scope: WorkspaceScope = _ANY_WORKSPACE,
    ) -> str | None:
        """Return the feature an identical earlier request produced, without deciding anything.

        The same key and fingerprint derivation `start` uses, so the two cannot disagree
        about what "the same request" means. A key already used with a *different*
        fingerprint answers nothing here and is still the conflict `start` raises.

        Scoped, because `Idempotency-Key` is a client-chosen string. Without the owner filter
        somebody sending a header another person had used would be handed that person's
        feature id -- and the start route answers a replay by returning the feature, so the
        leak would be the whole record. A key that names a feature outside this workspace
        answers nothing here, and the submission proceeds as new work.
        """
        fingerprint = _feature_fingerprint(request)
        # The owner is part of the key, so a caller reusing somebody else's `Idempotency-Key`
        # -- or submitting an identical PRD, which needs no header at all -- derives a
        # different key and finds nothing. The scope check below is defence in depth.
        key = _feature_idempotency_key(
            idempotency_key or request.idempotency_key, fingerprint, scope.owner_id
        )
        async with self._database.session() as session:
            existing = await session.get(FeatureWorkflowRequestModel, key)
            if existing is None or existing.fingerprint != fingerprint:
                return None
            try:
                await self._require_model(session, str(existing.feature_id), scope=scope)
            except WorkflowNotFoundError:
                return None
        return str(existing.feature_id)

    async def _purge_attachment_content(self, feature_id: str) -> None:
        """Drop the bytes of a retired feature's images, keeping every record of them.

        Isolated: a store that cannot answer must not turn a completed retirement into an
        error the operator reads as "it did not work".
        """
        if self._attachments is None:
            return
        # Bound at the call site rather than as a module logger: this file has none, and one
        # log line is not a reason for every read in it to acquire a logger.
        logger = structlog.get_logger("storage.attachments")
        try:
            purged = await self._attachments.purge_content_for_feature(feature_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error(
                "retired_feature_attachment_purge_failed",
                feature_id=feature_id,
                error_type=type(error).__name__,
            )
            return
        if purged:
            logger.info(
                "retired_feature_attachment_content_purged",
                feature_id=feature_id,
                attachments=purged,
            )

    async def _bind_attachments(
        self,
        session: AsyncSession,
        attachments: Sequence[PRDAttachment] | None,
        *,
        feature_id: str,
    ) -> None:
        """Claim this submission's images for this feature, inside the acceptance transaction.

        Nothing to do for a submission with no images, which is every submission that
        predates this and most that follow it. A composition with no attachment store and a
        submission that names one is a wiring fault rather than a user error: the route
        resolves every reference against the same store before accepting anything, so
        reaching here without one means the two were given different stores.

        The replay arm above returns before this: a replayed request's attachments are
        already bound to this feature, because the transaction that bound them is the one
        that wrote the idempotency row being replayed.
        """
        if not attachments:
            return
        if self._attachments is None:
            msg = "this composition cannot bind attachments: no attachment store is configured"
            raise WorkflowConflictError(msg)
        try:
            await self._attachments.bind_within(
                session,
                [item.attachment_id for item in attachments],
                feature_id=feature_id,
            )
        except AttachmentError as error:
            # A conflict, in the vocabulary the start path already refuses in: an attachment
            # bound to another feature between the route's check and this transaction.
            raise WorkflowConflictError(str(error)) from error

    async def _create_or_replay(
        self,
        request: StartFeatureRequest,
        *,
        key: str,
        fingerprint: str,
        owner_id: str,
        model_setup: dict[str, Any] | None = None,
        attachments: Sequence[PRDAttachment] | None = None,
    ) -> FeatureStartResult:
        """Atomically persist the parent feature, its reference, and its queue entry.

        One transaction on purpose. The reference is the identity the caller is about to be
        told, and the queue entry is the promise that the work will happen; committing any of
        the three without the others would mean answering with something that is not yet true.

        The submission's attachments bind inside that same transaction, for the same reason.
        Binding earlier would claim somebody's evidence for a feature a conflict below meant
        never existed; binding later would publish a PRD artifact whose image references
        point at unbound rows the periodic sweep is entitled to delete. If anything here
        fails, nothing is bound and the attachments are still uploaded and still usable.
        """
        async with self._database.session() as session:
            existing = await session.get(FeatureWorkflowRequestModel, key)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    msg = "idempotency key was already used with a different feature request"
                    raise WorkflowConflictError(msg)
                model = await self._require_model(session, existing.feature_id)
                return FeatureStartResult(
                    record=await self._record_from_model(session, model), created=False
                )
            feature_id = request.feature_id or f"feature-{uuid4()}"
            if await session.get(FeatureWorkflowModel, feature_id) is not None:
                msg = f"feature already exists: {feature_id}"
                raise WorkflowConflictError(msg)
            # The database allocates the number. An application-side count would give two
            # simultaneous submissions the same answer, and this insert cannot.
            allocation = FeatureReferenceModel(feature_id=feature_id)
            session.add(allocation)
            await session.flush()
            reference = format_feature_reference(allocation.reference_number)
            state = _initial_feature_state(
                feature_id,
                request,
                reference=reference,
                model_setup=model_setup,
                attachments=attachments,
            )
            model = FeatureWorkflowModel(
                feature_id=feature_id,
                # Whose workspace this lands in, fixed here and never reassigned. Taken from
                # the submitting actor by the route; there is no path by which a client names
                # it.
                owner_id=owner_id,
                status=state.status,
                title=state.title,
                reference=reference,
                reference_number=allocation.reference_number,
                execution_mode=state.execution_mode,
                agent_platform=state.agent_platform,
                performance_tier=state.performance_tier,
                model_setup_snapshot=state.model_setup_snapshot,
                model_setup_id=state.model_setup_id,
                merge_strategy=None,
                deployment_strategy=None,
                state_json=_persisted_state_json(state),
                created_at=state.created_at,
                updated_at=state.updated_at,
            )
            session.add(model)
            # The derived repository, artifact, and event rows have independent foreign
            # keys rather than ORM relationships. Flush the parent explicitly so
            # PostgreSQL can satisfy those constraints regardless of unit-of-work
            # insertion ordering.
            await session.flush()
            session.add(
                FeatureWorkflowRequestModel(
                    idempotency_key=key,
                    fingerprint=fingerprint,
                    feature_id=feature_id,
                    created_at=state.created_at,
                )
            )
            self._add_state_children(session, state)
            await self._bind_attachments(session, attachments, feature_id=feature_id)
            self._add_event(session, feature_id, "feature_started", timestamp=state.created_at)
            self._add_event(
                session,
                feature_id,
                "feature_queued",
                {"reference": reference, "requested_by": owner_id},
                timestamp=state.created_at,
            )
            await self._queue.enqueue(
                feature_id=feature_id,
                execution_mode=state.execution_mode,
                agent_platform=state.agent_platform,
                performance_tier=state.performance_tier,
                model_setup_snapshot=state.model_setup_snapshot,
                model_setup_id=state.model_setup_id,
                requested_by=owner_id,
                session=session,
            )
            await session.commit()
            return FeatureStartResult(
                record=FeatureRecord(
                    state=state,
                    created_at=state.created_at,
                    updated_at=state.updated_at,
                    lifecycle_events=[
                        TimelineEvent(state.created_at, "feature_started", {}),
                        TimelineEvent(
                            state.created_at,
                            "feature_queued",
                            {"reference": reference, "requested_by": owner_id},
                        ),
                    ],
                ),
                created=True,
            )

    async def _replace_state(
        self,
        feature_id: str,
        state: FeatureWorkflowSnapshot,
        event: str,
        details: dict[str, Any] | None = None,
        *,
        checkpoint_boundary: WorkflowCheckpointBoundary | None = None,
        checkpoint_repository_id: str | None = None,
        preserve_persisted_progress: bool = False,
    ) -> None:
        """Merge and replace one durable snapshot behind local, Redis, and database locks."""
        if state.feature_id != feature_id:
            msg = "feature runner returned state for a different feature"
            raise WorkflowConflictError(msg)
        # Every mutation of a feature's durable state passes through here, which is why the
        # fence is here and not at each caller. An executor that lost its claim is not
        # entitled to write: recovery may already have reconciled its action and given the
        # feature a verdict, and a late write would silently overwrite that decision.
        # AB-Feature-108 did exactly this seventeen minutes after being declared abandoned.
        if current_feature_action_is_revoked():
            msg = (
                "this executor no longer holds the claim on this feature; its outcome has "
                "already been reconciled"
            )
            raise FeatureRunRevokedError(msg)
        _require_mutable_feature_identity(state)
        state = ensure_feature_failure_summary(state)
        lock_key = (self._state_lock_scope, feature_id)
        local_lock = _LOCAL_STATE_WRITE_LOCKS.get(lock_key)
        if local_lock is None:
            local_lock = asyncio.Lock()
            _LOCAL_STATE_WRITE_LOCKS[lock_key] = local_lock
        async with (
            local_lock,
            self._lock.hold(f"feature:state-write:{feature_id}"),
            self._database.session() as session,
        ):
            # The Redis lock is the cross-process first line of defence.  The row
            # lock also serializes correctly configured PostgreSQL writers even if
            # they are reached through different control-plane instances.
            model = await session.scalar(
                select(FeatureWorkflowModel)
                .where(FeatureWorkflowModel.feature_id == feature_id)
                .with_for_update()
            )
            if model is None:
                msg = f"feature not found: {feature_id}"
                raise WorkflowNotFoundError(msg)
            persisted = await self._state_from_model(session, model)
            state = _merge_durable_progress(
                persisted,
                state,
                checkpoint_boundary=checkpoint_boundary,
                checkpoint_repository_id=checkpoint_repository_id,
                preserve_persisted_progress=preserve_persisted_progress,
            )
            state = self._preserve_cancellation(persisted, state)
            # Every writer of child state comes through here, which is why the refusal
            # invariant is checked here and not at each of them.
            _reject_reintroduced_refusal(persisted, state)
            # After the merge, because the merge is what decides the children: a cancellation
            # snapshot keeps the newest durable progress, so a repository that finished in the
            # meantime survives and settling before this point would simply be overwritten.
            #
            # Once the feature is terminal nothing will schedule a repository again, so a
            # workstream still reading `running` is not in flight -- it is a row nobody will
            # ever update. AB-Feature-107 was cancelled with its console mid-attempt and that
            # row still said `running` a day later, to an operator, the features list and the
            # workflow graph alike; -131 and -132 reached the identical row by failing instead,
            # when a sibling raised hard enough to escape the fan-out and leave the attempt
            # beside it unclosed.
            #
            # Cancelled work is recorded as cancelled and failed work as failed, because the
            # two say different things to whoever reads the row next: one was stopped, the
            # other stopped. `settle_unfinished_children` rewrites neither finished work nor a
            # blocked workstream, so an approved repository still publishes and a starved one
            # still says it was starved.
            if state.status in _SETTLED_FEATURE_STATUSES:
                settle_unfinished_children(
                    state,
                    ChildWorkflowStatus.CANCELLED
                    if state.status in _CANCELLED_FEATURE_STATUSES
                    else ChildWorkflowStatus.FAILED,
                )
            existing_artifact_count = len(persisted.artifacts)
            self._apply_state(model, state)
            await self._delete_state_children(session, feature_id)
            self._add_state_children(session, state)
            self._add_event(session, feature_id, event, details)
            for artifact in state.artifacts[existing_artifact_count:]:
                for artifact_event, artifact_details in _artifact_events(artifact):
                    self._add_event(session, feature_id, artifact_event, artifact_details)
            await session.commit()

    def _preserve_cancellation(
        self, persisted: FeatureWorkflowSnapshot, state: FeatureWorkflowSnapshot
    ) -> FeatureWorkflowSnapshot:
        """Ensure a stale in-flight runner result can never erase a concurrent cancel request."""
        if not persisted.cancellation_requested or state.cancellation_requested:
            return state
        terminal_statuses = {
            FeatureWorkflowStatus.CANCELLED,
            FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
        }
        return state.model_copy(
            update={
                "cancellation_requested": True,
                "cancellation_status": persisted.cancellation_status,
                "cancellation_requested_at": persisted.cancellation_requested_at,
                "cancellation_requested_by": persisted.cancellation_requested_by,
                "cancellation_reason": persisted.cancellation_reason,
                "cancellation_current_operation_id": (persisted.cancellation_current_operation_id),
                "cancellation_completed_at": persisted.cancellation_completed_at,
                "cleanup_requirements": persisted.cleanup_requirements,
                "status": (
                    persisted.status
                    if persisted.status in terminal_statuses
                    else FeatureWorkflowStatus.CANCELLING
                ),
                "current_agent": "cancellation",
                "updated_at": datetime.now(UTC),
            }
        )

    async def _require_live_capacity(self) -> None:
        """Refuse a live start that would exceed the configured concurrent-feature quota."""
        if self._max_concurrent_live_features <= 0:
            return
        async with self._database.session() as session:
            running = await session.scalar(
                select(func.count())
                .select_from(FeatureWorkflowModel)
                .where(
                    FeatureWorkflowModel.execution_mode == "live",
                    FeatureWorkflowModel.status.notin_(_QUOTA_SETTLED_STATUSES),
                )
            )
        if int(running or 0) >= self._max_concurrent_live_features:
            msg = (
                f"live feature quota reached: {running} in flight, limit "
                f"{self._max_concurrent_live_features}. Wait for one to finish, or resolve a "
                "feature that is stuck through the retire endpoint."
            )
            raise WorkflowConflictError(msg)

    async def _require_workspace_capacity(self) -> None:
        """Refuse a live submission the worker volume cannot house, before accepting it.

        The same floor the capacity preflight enforces, asked an hour earlier. That preflight
        runs *during* execution, so a volume that was already full accepted the feature,
        planned it, cloned what it could and then killed it -- three of the fifty failures
        measured for this task ended that way, each after real work had been done and paid
        for. The condition was knowable at submission time in every one of them.

        Refused before any row exists, matching `_require_live_capacity` beside it and the
        acceptance contract generally: a rejected submission leaves no feature to explain,
        and the caller gets the reason in the response rather than in a record they have to
        go and find. The message is the preflight's own, so the two read identically.

        Deliberately not a cleanup trigger. Pruning terminal workspaces is the executing
        worker's job and needs the lock it takes; doing it inside a submission would put
        filesystem work on the request path for every feature.
        """
        root = self._workspace_root
        if root is None or self._workspace_minimum_free_bytes <= 0:
            return
        try:
            free_bytes = await asyncio.to_thread(lambda: shutil.disk_usage(root).free)
        except OSError:
            # An unreadable volume is not evidence that it is full, and refusing every
            # submission on a failed stat would be a worse failure than the one this guards.
            return
        if free_bytes >= self._workspace_minimum_free_bytes:
            return
        raise WorkflowConflictError(
            workspace_capacity_diagnostic(
                workspace_root=str(root),
                free_bytes=free_bytes,
                required_bytes=self._workspace_minimum_free_bytes,
            )
        )

    async def _has_active_operations(self, feature_id: str) -> bool:
        """Keep a request in progress until all live operations are terminal or reconciled."""
        assert self._operation_journal is not None
        return any(
            item.status.value in {"starting", "running", "cancellation_requested"}
            for item in await self._operation_journal.list_operations_for_workflow(feature_id)
        )

    async def _request_operation_cancellation(self, feature_id: str) -> None:
        """Make the requested stop visible in every active side-effect timeline."""
        if self._operation_journal is None:
            return
        active = {
            ExternalOperationStatus.STARTING,
            ExternalOperationStatus.RUNNING,
        }
        for operation in await self._operation_journal.list_operations_for_workflow(feature_id):
            if operation.status in active:
                try:
                    await self._operation_journal.transition_status(
                        operation.operation_id,
                        ExternalOperationStatus.CANCELLATION_REQUESTED,
                        event_type="cancellation_requested",
                    )
                except OperationTransitionConflictError:
                    # The operation completed or recovery fenced it after the active-row
                    # snapshot. Its newer durable state is authoritative.
                    continue

    async def _finish_cancellation(self, feature_id: str) -> None:
        """Finalize only after active work stops, retaining known side effects for review."""
        record = await self.get_record(feature_id)
        state = record.state.model_copy(deep=True)
        if self._operation_journal is not None and await self._has_active_operations(feature_id):
            transition_feature(
                state,
                FeatureWorkflowStatus.CANCELLING,
                reason=(
                    "This feature's cancellation is still waiting for external operations to stop."
                ),
                agent="cancellation",
            )
            state.cancellation_status = CancellationLifecycleStatus.CANCELLATION_IN_PROGRESS
        else:
            requirements = await self._cleanup_requirements(feature_id)
            state.cleanup_requirements = requirements
            if requirements:
                transition_feature(
                    state,
                    FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
                    reason=(
                        f"This feature was cancelled and left {len(requirements)} external "
                        "effect(s) for somebody to review."
                    ),
                    agent="cancellation",
                )
                state.cancellation_status = (
                    CancellationLifecycleStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS
                )
            else:
                transition_feature(
                    state,
                    FeatureWorkflowStatus.CANCELLED,
                    reason="This feature was cancelled and left nothing behind to clean up.",
                    agent="cancellation",
                )
                state.cancellation_status = CancellationLifecycleStatus.CANCELLED
            state.cancellation_completed_at = datetime.now(UTC)
        state.cancellation_requested = True
        state.updated_at = datetime.now(UTC)
        await self._replace_state(
            feature_id,
            state,
            "feature_cancellation_updated",
            preserve_persisted_progress=True,
        )

    async def _cleanup_requirements(self, feature_id: str) -> list[ManualCleanupRequirement]:
        """Expose retained commit/push/PR effects rather than silently deleting remote work."""
        if self._operation_journal is None:
            return []
        requirements: list[ManualCleanupRequirement] = []
        affected_types = {
            ExternalOperationType.CREATE_BRANCH,
            ExternalOperationType.CREATE_COMMIT,
            ExternalOperationType.PUSH_BRANCH,
            ExternalOperationType.CREATE_PULL_REQUEST,
        }
        for operation in await self._operation_journal.list_operations_for_workflow(feature_id):
            if operation.operation_type not in affected_types or operation.status not in {
                ExternalOperationStatus.SUCCEEDED,
                ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
                # A deferred remote effect is unconfirmed too. It is normally settled by the
                # next credentialed request, but a cancelled feature will never make one, so
                # cancellation is the point at which it becomes a person's question.
                ExternalOperationStatus.AWAITING_RECONCILIATION,
            }:
                continue
            unknown_without_reference = (
                operation.status
                in {
                    ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
                    ExternalOperationStatus.AWAITING_RECONCILIATION,
                }
                and operation.external_reference is None
            )
            reference = operation.external_reference or operation.operation_id
            requirements.append(
                ManualCleanupRequirement(
                    resource_type=operation.operation_type.value,
                    repository_id=operation.repository_id,
                    external_reference=reference,
                    reason=(
                        "external state is unknown; the journal operation ID is retained "
                        "for manual reconciliation"
                        if unknown_without_reference
                        else "feature cancellation retained an externally observable operation"
                    ),
                    recommended_action=(
                        "Reconcile the journal operation against the provider before any "
                        "cleanup; do not assume the side effect is absent."
                        if unknown_without_reference
                        else "Review the retained resource; do not delete it automatically."
                    ),
                )
            )
        return requirements

    def _bind_checkpoint_writer(self, runner: FeatureWorkflowRunner | None) -> None:
        """Use a narrow optional extension so existing mock runner public APIs remain unchanged."""
        binder = getattr(runner, "bind_checkpoint_writer", None)
        if binder is not None:
            binder(self.checkpoint)

    def _bind_event_writer(self, runner: FeatureWorkflowRunner | None) -> None:
        """Give the live runner the timeline writer, through the same optional extension."""
        binder = getattr(runner, "bind_event_writer", None)
        if binder is not None:
            binder(self.record_feature_event)

    async def record_feature_event(
        self, feature_id: str, event: str, details: dict[str, Any]
    ) -> None:
        """Append one timeline event mid-run, outside any state replacement.

        The checkpoint writer is the only other channel by which a running orchestrator
        writes something durable before its step ends, and it can only say that a boundary
        passed. `repository_planned_blind` is a fact about a repository an operator is
        deciding whether to trust, and it must be readable *while* the run that produced it
        is still going -- AB-Feature-173's operator asked twice and the record had nothing.
        """
        async with self._database.session() as session:
            self._add_event(session, feature_id, event, details)
            await session.commit()

    async def _mark_failed_requires_human(
        self,
        feature_id: str,
        event: str,
        error: Exception,
        *,
        current_agent: str,
        error_type: str | None = None,
        diagnostics_prefix: Sequence[str] = (),
    ) -> None:
        """Record a workflow failure as human-action state without exposing error text.

        A feature that has already finished being decided is left where it is. This is the
        generic handler every unexpected exception reaches, and it is the one path that never
        asked what the feature it was about to tombstone had reached: a repair approved
        against a completed feature, whose runner then raised, rewrote `completed` as
        `failed_requires_human` and threw the completion away. The caller still gets the same
        refusal -- its request genuinely did not happen -- and the timeline still gains the
        line saying so; what does not happen is a finished feature being reopened by a request
        that arrived after it. `_RETIRED_EQUIVALENT_STATUSES` is the same absorbing set
        `cancel`, `retire` and the claim disposition already read.

        `error_type` and `diagnostics_prefix` are for the one caller that knows more about the
        failure than the exception does. `fail_exhausted_fault` has established the fault is
        transient before it ever gets here -- that is what the queue spent its attempts on --
        and reading the type name back off the error throws that away. Both default to what
        every other caller already got.
        """
        state = (await self.get_record(feature_id)).state.model_copy(deep=True)
        if state.status.value in _RETIRED_EQUIVALENT_STATUSES:
            await self._record_queued_refusal(
                feature_id,
                event=event,
                reason=(
                    f"This feature was already {state.status.value} when {event} happened, so "
                    "the failure belongs to the request rather than to the feature."
                ),
            )
            return
        transition_feature(
            state,
            FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
            reason=f"{event} while {current_agent} was running this feature.",
            agent=current_agent,
        )
        state.updated_at = datetime.now(UTC)
        # This handler represents a new execution failure. Historical failures remain in the
        # lifecycle timeline; the snapshot's summary must describe this attempt rather than
        # blocking `ensure_feature_failure_summary` with a diagnosis from a previous run.
        state.failure_summary = None
        state = ensure_feature_failure_summary(
            state,
            # The agent this handler was told about is where the failure happened: callers
            # resolve it from the rejected artifact before they get here. Passed explicitly
            # rather than read back off `current_agent`, which by the time a summary is built
            # names whichever agent ran last rather than the one that failed.
            stage=current_agent,
            # The weaker form deliberately: a repository that failed its own validation keeps
            # that classification even though the exception that ended the run arrived here.
            # What this replaces is the type name, which said `DBAPIError` where the record
            # needed to say whose defect it was. Still the weak form for the exhaustion
            # caller too: a queue fault is what stopped the run, but a repository that
            # rejected its own source is the better answer to what went wrong.
            error_type=error_type if error_type is not None else classification_of(error).value,
            diagnostics=[*diagnostics_prefix, *safe_error_diagnostics(error)],
        )
        # The summary already decides whether the provider simply failed to answer, and says
        # so by setting `retryable` and telling the operator to resume. Leaving the feature
        # FAILED_REQUIRES_HUMAN contradicted its own instruction: resume rejects that status,
        # so the advice could only be followed through the empty-answer recovery path, and a
        # caller reading the status saw a dead feature. One OpenAI timeout during planning
        # ended run-3 that way, with nothing wrong but a slow provider.
        if state.failure_summary is not None and state.failure_summary.retryable:
            transition_feature(
                state,
                FeatureWorkflowStatus.WAITING_FOR_HUMAN,
                reason=(
                    "The provider failed to answer rather than the work being wrong, so this "
                    "feature is left resumable."
                ),
                agent="human_clarification",
            )
        await self._replace_state(feature_id, state, event, _failure_details(error))

    async def _record_from_model(self, session: Any, model: FeatureWorkflowModel) -> FeatureRecord:
        """Rehydrate strict parent state and restore separately indexed child references."""
        state = await self._state_from_model(session, model)
        events = list(
            await session.scalars(
                select(FeatureWorkflowEventModel)
                .where(FeatureWorkflowEventModel.feature_id == model.feature_id)
                .order_by(FeatureWorkflowEventModel.id)
            )
        )
        return FeatureRecord(
            state=state,
            created_at=_as_utc(model.created_at),
            updated_at=_as_utc(model.updated_at),
            lifecycle_events=[
                TimelineEvent(_as_utc(item.timestamp), item.event, item.details) for item in events
            ],
        )

    async def _state_from_model(
        self, session: Any, model: FeatureWorkflowModel
    ) -> FeatureWorkflowSnapshot:
        """Use generic artifact rows as the source of truth for immutable parent handoffs."""
        artifacts = list(
            await session.scalars(
                select(FeatureArtifactModel)
                .where(FeatureArtifactModel.feature_id == model.feature_id)
                .order_by(FeatureArtifactModel.id)
            )
        )
        children = list(
            await session.scalars(
                select(ChildWorkflowModel)
                .where(ChildWorkflowModel.feature_id == model.feature_id)
                .order_by(ChildWorkflowModel.repository_id)
            )
        )
        state_data = dict(model.state_json)
        # The column is authoritative. A snapshot written before references existed has no
        # `reference` inside its JSON, and the backfill wrote the column rather than rewriting
        # ninety-odd snapshots -- so the column is where the answer lives either way.
        if model.reference is not None:
            state_data["reference"] = model.reference
        state_data["artifacts"] = [item.payload for item in artifacts]
        state_data["child_workflows"] = {
            item.repository_id: ChildWorkflowReference(
                child_workflow_id=item.child_workflow_id,
                repository_id=item.repository_id,
                workstream_id=item.workstream_id,
                status=item.status,
                branch_name=item.branch_name,
                base_branch=item.base_branch,
                workspace_path=item.workspace_path,
                retry_count=item.retry_count,
                code_completion_artifact_id=item.code_completion_artifact_id,
                review_artifact_id=item.review_artifact_id,
                pull_request_artifact_id=item.pull_request_artifact_id,
                blocking_issues=item.blocking_issues,
                checkpoint_boundary=item.checkpoint_boundary,
                technology_profile=item.technology_profile,
                validation_plan=item.validation_plan,
                current_revision=item.current_revision,
                current_validation_results=item.current_validation_results,
                superseded_validation_count=item.superseded_validation_count,
                scoped_requirements=item.scoped_requirements,
                out_of_scope_requirements=item.out_of_scope_requirements,
                preflight_result=item.preflight_result,
                preflight_status=item.preflight_status,
                blocking_setup_issues=item.blocking_setup_issues,
                selected_package_manager=item.selected_package_manager,
                layout_evidence=item.layout_evidence,
                configured_validation_commands=item.configured_validation_commands,
                test_availability=item.test_availability,
                implementation_expectations=item.implementation_expectations,
                production_files_changed=item.production_files_changed,
                test_files_changed=item.test_files_changed,
                configuration_files_changed=item.configuration_files_changed,
                requirements_implemented=item.requirements_implemented,
                requirements_not_implemented=item.requirements_not_implemented,
                failure_classification=item.failure_classification,
                retry_strategy=item.retry_strategy,
                model_routing=item.model_routing,
                implementation_retry_count=item.implementation_retry_count,
                validation_retry_count=item.validation_retry_count,
                repository_setup_retry_count=item.repository_setup_retry_count,
                integration_retry_count=item.integration_retry_count,
                targeted_attempt_kind=item.targeted_attempt_kind,
                granted_extra_attempts=item.granted_extra_attempts,
                retry_grants=list(item.retry_grants or []),
                meaningful_change=item.meaningful_change,
                meaningful_change_reason=item.meaningful_change_reason,
                retry_refusal_reason=item.retry_refusal_reason,
                production_diff_fingerprint=item.production_diff_fingerprint,
                previous_attempt_fingerprint=item.previous_attempt_fingerprint,
                test_diff_fingerprint=item.test_diff_fingerprint,
                previous_test_fingerprint=item.previous_test_fingerprint,
                runtime_wall_seconds=item.runtime_wall_seconds,
                runtime_charged_seconds=item.runtime_charged_seconds,
                planned_blind=bool(item.planned_blind),
                planned_blind_reason=item.planned_blind_reason,
            )
            for item in children
        }
        state_data["child_workflows"] = {
            key: value.model_dump(mode="json")
            for key, value in state_data["child_workflows"].items()
        }
        return FeatureWorkflowSnapshot.model_validate_json(json.dumps(state_data))

    async def _delete_state_children(self, session: Any, feature_id: str) -> None:
        """Remove derived rows before replacing them with a transactionally consistent snapshot."""
        for model in (
            RepositorySpecModel,
            ChildWorkflowModel,
            FeatureArtifactModel,
            IntegrationContractModel,
            ContractChangeRequestModel,
            RepositoryRepairModel,
            IntegrationReviewModel,
            FeaturePullRequestModel,
        ):
            await session.execute(delete(model).where(model.feature_id == feature_id))

    def _add_state_children(self, session: Any, state: FeatureWorkflowSnapshot) -> None:
        """Write artifact, repository, child, contract, review, and PR indexes without secrets."""
        for repository in state.repository_specs:
            session.add(
                RepositorySpecModel(
                    feature_id=state.feature_id,
                    repository_id=repository.repository_id,
                    name=repository.name,
                    role=repository.role,
                    repository_url=str(repository.repository_url),
                    default_branch=repository.default_branch,
                    local_workspace_path=repository.local_workspace_path,
                    required=repository.required,
                    implementation_order=repository.implementation_order,
                    metadata_json=repository.metadata,
                )
            )
        for child in state.child_workflows.values():
            session.add(
                ChildWorkflowModel(
                    child_workflow_id=child.child_workflow_id,
                    feature_id=state.feature_id,
                    repository_id=child.repository_id,
                    workstream_id=child.workstream_id,
                    status=child.status,
                    branch_name=child.branch_name,
                    base_branch=child.base_branch,
                    workspace_path=child.workspace_path,
                    retry_count=child.retry_count,
                    code_completion_artifact_id=child.code_completion_artifact_id,
                    review_artifact_id=child.review_artifact_id,
                    pull_request_artifact_id=child.pull_request_artifact_id,
                    blocking_issues=child.blocking_issues,
                    checkpoint_boundary=(
                        child.checkpoint_boundary.value
                        if child.checkpoint_boundary is not None
                        else None
                    ),
                    technology_profile=child.technology_profile,
                    validation_plan=child.validation_plan,
                    current_revision=child.current_revision,
                    current_validation_results=child.current_validation_results,
                    superseded_validation_count=child.superseded_validation_count,
                    scoped_requirements=child.scoped_requirements,
                    out_of_scope_requirements=child.out_of_scope_requirements,
                    preflight_result=child.preflight_result,
                    preflight_status=child.preflight_status,
                    blocking_setup_issues=child.blocking_setup_issues,
                    selected_package_manager=child.selected_package_manager,
                    layout_evidence=child.layout_evidence,
                    configured_validation_commands=child.configured_validation_commands,
                    test_availability=child.test_availability,
                    implementation_expectations=child.implementation_expectations,
                    production_files_changed=child.production_files_changed,
                    test_files_changed=child.test_files_changed,
                    configuration_files_changed=child.configuration_files_changed,
                    requirements_implemented=child.requirements_implemented,
                    requirements_not_implemented=child.requirements_not_implemented,
                    failure_classification=child.failure_classification,
                    retry_strategy=child.retry_strategy,
                    model_routing=child.model_routing,
                    implementation_retry_count=child.implementation_retry_count,
                    validation_retry_count=child.validation_retry_count,
                    repository_setup_retry_count=child.repository_setup_retry_count,
                    integration_retry_count=child.integration_retry_count,
                    targeted_attempt_kind=(
                        str(child.targeted_attempt_kind)
                        if child.targeted_attempt_kind is not None
                        else None
                    ),
                    granted_extra_attempts=child.granted_extra_attempts,
                    retry_grants=list(child.retry_grants),
                    meaningful_change=child.meaningful_change,
                    meaningful_change_reason=child.meaningful_change_reason,
                    retry_refusal_reason=child.retry_refusal_reason,
                    production_diff_fingerprint=child.production_diff_fingerprint,
                    previous_attempt_fingerprint=child.previous_attempt_fingerprint,
                    test_diff_fingerprint=child.test_diff_fingerprint,
                    previous_test_fingerprint=child.previous_test_fingerprint,
                    runtime_wall_seconds=child.runtime_wall_seconds,
                    runtime_charged_seconds=child.runtime_charged_seconds,
                    planned_blind=child.planned_blind,
                    planned_blind_reason=child.planned_blind_reason,
                )
            )
        # Artifacts are append-only, and a decision on a durable request is expressed by
        # appending a revision carrying the same business identifier -- so a request that has
        # been approved is present twice. The tables below are keyed by that identifier, and
        # writing every artifact into them inserted two rows with one primary key: the first
        # contract-change approval to reach durable storage failed on an integrity error
        # rather than being applied. Only the newest artifact per identifier is projected.
        current_change_requests = _newest_by_key(
            state.artifacts, ContractChangeRequestArtifact, lambda item: item.change_request_id
        )
        current_repairs = _newest_by_key(
            state.artifacts, RepositoryRepairProposalArtifact, lambda item: item.repair_id
        )
        for artifact in state.artifacts:
            serialized = artifact.model_dump(mode="json")
            session.add(
                FeatureArtifactModel(
                    feature_id=state.feature_id,
                    artifact_id=artifact.artifact_id,
                    artifact_type=artifact.artifact_type,
                    payload=serialized,
                    produced_at=artifact.timestamp,
                )
            )
            if isinstance(artifact, IntegrationContractArtifact):
                session.add(
                    IntegrationContractModel(
                        feature_id=state.feature_id,
                        artifact_id=artifact.artifact_id,
                        contract_version=artifact.contract_version,
                        status=artifact.status,
                        payload=serialized,
                    )
                )
            elif isinstance(artifact, ContractChangeRequestArtifact):
                if current_change_requests.get(artifact.change_request_id) is artifact:
                    session.add(
                        ContractChangeRequestModel(
                            change_request_id=artifact.change_request_id,
                            feature_id=state.feature_id,
                            status=artifact.status,
                            requested_by_repository_id=artifact.requested_by_repository_id,
                            payload=serialized,
                        )
                    )
            elif isinstance(artifact, RepositoryRepairProposalArtifact):
                if current_repairs.get(artifact.repair_id) is artifact:
                    session.add(
                        RepositoryRepairModel(
                            repair_id=artifact.repair_id,
                            feature_id=state.feature_id,
                            repository_id=artifact.repository_id,
                            artifact_id=artifact.artifact_id,
                            status=RepositoryRepairStatus(artifact.status),
                            failure_classification=artifact.failure_classification,
                            proposed_at_revision=artifact.proposed_at_revision,
                            resulting_revision=artifact.resulting_revision,
                            payload=serialized,
                        )
                    )
            elif isinstance(artifact, IntegrationReviewArtifact):
                session.add(
                    IntegrationReviewModel(
                        feature_id=state.feature_id,
                        artifact_id=artifact.artifact_id,
                        status=artifact.review_status,
                        payload=serialized,
                    )
                )
            elif isinstance(artifact, PullRequestArtifact):
                # Strip the qualifiers a revision adds (`.v2`, and `.closed` on the durable
                # closed-state copy) so the row's repository id stays the repository id.
                repository_id = _PULL_REQUEST_ID_QUALIFIERS.sub(
                    "",
                    artifact.artifact_id.removeprefix("008_pull_request.").removesuffix(".json"),
                )
                session.add(
                    FeaturePullRequestModel(
                        feature_id=state.feature_id,
                        repository_id=repository_id,
                        artifact_id=artifact.artifact_id,
                        pull_request_url=str(artifact.url),
                        payload=serialized,
                    )
                )

    def _apply_state(self, model: FeatureWorkflowModel, state: FeatureWorkflowSnapshot) -> None:
        """Synchronize queryable summary fields and the full non-secret feature snapshot."""
        model.status = state.status
        model.title = state.title
        # Never reassigned: a reference is the identity somebody has already quoted in a pull
        # request and a chat message. It is only written here for a snapshot that predates
        # references and has just been backfilled.
        if model.reference is None and state.reference is not None:
            model.reference = state.reference
        model.execution_mode = state.execution_mode
        # Written on every flush like the mode beside it, though nothing ever changes it: the
        # platform is fixed at submission, and a write that could change it is the one thing
        # this column must not allow.
        model.agent_platform = state.agent_platform
        # Projected explicitly for the reason `40-` documented: a state field this method does
        # not name silently never persists, and the runtime clocks were lost to exactly that.
        model.performance_tier = state.performance_tier
        # The pinned setup, projected for the same reason -- and never rewritten to something
        # newer: the snapshot is fixed at creation and this write only carries it forward.
        model.model_setup_snapshot = state.model_setup_snapshot
        model.model_setup_id = state.model_setup_id
        model.merge_strategy = (
            state.merge_strategy.value if state.merge_strategy is not None else None
        )
        model.deployment_strategy = (
            state.deployment_strategy.value if state.deployment_strategy is not None else None
        )
        model.state_json = _persisted_state_json(state)
        model.updated_at = state.updated_at

    async def _require_model(
        self, session: Any, feature_id: str, *, scope: WorkspaceScope = _ANY_WORKSPACE
    ) -> FeatureWorkflowModel:
        """Load one parent row within a workspace, or raise the existing not-found error.

        This is where workspace isolation is enforced, and it is one function because every
        read and every mutation reaches a feature through it -- `get_record` calls it, and
        every mutation begins by calling `get_record`. A route cannot forget the check
        because no route performs it.

        The owner is in the `WHERE` rather than compared after loading. A row somebody else
        owns is therefore indistinguishable from a row that does not exist, all the way up:
        `WorkflowNotFoundError` is what the routes already answer `404`, so the caller sees
        the same bytes for "not yours" and "never existed" without any route deciding to.
        """
        statement = select(FeatureWorkflowModel).where(
            FeatureWorkflowModel.feature_id == feature_id
        )
        if scope.applies():
            statement = statement.where(FeatureWorkflowModel.owner_id == scope.owner_id)
        model = (await session.execute(statement)).scalar_one_or_none()
        if model is None:
            msg = f"feature not found: {feature_id}"
            raise WorkflowNotFoundError(msg)
        return cast(FeatureWorkflowModel, model)

    async def owner_of(self, feature_id: str) -> str:
        """Return whose workspace one feature is in, whoever is asking.

        Deliberately unscoped, and deliberately not a read of anything else. This answers
        "whose credentials must the worker resolve", which is a fact about the feature rather
        than about the caller: an administrator granting a retry on somebody's feature must
        put *that person* on the queue entry, and asking within the administrator's own scope
        would answer nothing.

        Callers must have already established that they may act on the feature -- every one
        of them has, through `get_record`. This is one indexed primary-key read, so it costs
        nothing beside the record they already loaded.
        """
        async with self._database.session() as session:
            owner = (
                await session.execute(
                    select(FeatureWorkflowModel.owner_id).where(
                        FeatureWorkflowModel.feature_id == feature_id
                    )
                )
            ).scalar_one_or_none()
        if owner is None:
            msg = f"feature not found: {feature_id}"
            raise WorkflowNotFoundError(msg)
        return str(owner)

    async def require_visible(
        self, feature_id: str, *, scope: WorkspaceScope = _ANY_WORKSPACE
    ) -> None:
        """Raise unless one feature is in this caller's workspace.

        For the reads that do not hydrate parent state -- the event poll, the chat
        transcript, the action list. Those exist precisely so a client can watch a feature
        without deserialising every artifact it has produced, and routing them through
        `get_record` to get the ownership check would undo that. This is the same predicate
        against the same column, as one indexed read.
        """
        async with self._database.session() as session:
            await self._require_model(session, feature_id, scope=scope)

    def _required_runner(self, execution_mode: str) -> FeatureWorkflowRunner:
        """Require deliberate configuration for the persisted mock or live execution mode."""
        runner = self._runner_for(execution_mode)
        if runner is None:
            msg = f"feature runner is not configured for execution_mode '{execution_mode}'"
            raise WorkflowConflictError(msg)
        return runner

    def _runner_for(self, execution_mode: str) -> FeatureWorkflowRunner | None:
        """Keep mock mode isolated from live provider credentials and network adapters."""
        return self._mock_runner if execution_mode == "mock" else self._live_runner

    def _add_event(
        self,
        session: Any,
        feature_id: str,
        event: str,
        details: dict[str, Any] | None = None,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Append a credential-free parent event suitable for observability and audit reads."""
        session.add(
            FeatureWorkflowEventModel(
                feature_id=feature_id,
                timestamp=timestamp or datetime.now(UTC),
                event_type="lifecycle",
                source="api",
                event=event,
                details=details or {},
            )
        )


def _merge_durable_progress(
    persisted: FeatureWorkflowSnapshot,
    incoming: FeatureWorkflowSnapshot,
    *,
    checkpoint_boundary: WorkflowCheckpointBoundary | None,
    checkpoint_repository_id: str | None,
    preserve_persisted_progress: bool,
) -> FeatureWorkflowSnapshot:
    """Merge append-only artifacts and repository-scoped checkpoints without lost updates."""
    if persisted.feature_id != incoming.feature_id:
        msg = "cannot merge feature snapshots with different feature IDs"
        raise WorkflowConflictError(msg)

    artifacts_by_id = {item.artifact_id: item for item in persisted.artifacts}
    merged_artifacts = list(persisted.artifacts)
    for artifact in incoming.artifacts:
        existing = artifacts_by_id.get(artifact.artifact_id)
        if existing is None:
            artifacts_by_id[artifact.artifact_id] = artifact
            merged_artifacts.append(artifact)
        elif existing.model_dump(mode="json") != artifact.model_dump(mode="json"):
            msg = (
                "artifact ID is immutable and already has different content: "
                f"{artifact.artifact_id}"
            )
            raise WorkflowConflictError(msg)

    if preserve_persisted_progress:
        # A cancellation snapshot can be stale by the time its write lock is acquired.
        # Start with the newest durable workflow progress and copy only cancellation-owned
        # fields from the request, so a just-finished child or contract revision survives.
        merged = persisted.model_copy(
            deep=True,
            update={
                "status": incoming.status,
                "cancellation_requested": incoming.cancellation_requested,
                "cancellation_status": incoming.cancellation_status,
                "cancellation_requested_at": incoming.cancellation_requested_at,
                "cancellation_requested_by": incoming.cancellation_requested_by,
                "cancellation_reason": incoming.cancellation_reason,
                "cancellation_current_operation_id": (incoming.cancellation_current_operation_id),
                "cancellation_completed_at": incoming.cancellation_completed_at,
                "cleanup_requirements": incoming.cleanup_requirements,
                "current_agent": incoming.current_agent,
                "updated_at": incoming.updated_at,
            },
        )
        merged.artifacts = merged_artifacts
        return merged

    merged_children = dict(persisted.child_workflows)
    if checkpoint_boundary is not None and checkpoint_repository_id is not None:
        try:
            checkpoint_child = incoming.child_workflows[checkpoint_repository_id]
        except KeyError as error:
            msg = f"repository checkpoint omitted its child workflow: {checkpoint_repository_id}"
            raise WorkflowConflictError(msg) from error
        merged_children[checkpoint_repository_id] = checkpoint_child.model_copy(
            update={"checkpoint_boundary": checkpoint_boundary}
        )
    else:
        # Parent/final writes may advance several children together, but omission is not
        # deletion: child rows are durable audit records once initialized.
        merged_children.update(incoming.child_workflows)

    if checkpoint_boundary is not None:
        merged_boundaries = dict(persisted.checkpoint_boundaries)
        merged_boundaries[checkpoint_repository_id or "parent"] = checkpoint_boundary
    else:
        merged_boundaries = {
            **persisted.checkpoint_boundaries,
            **incoming.checkpoint_boundaries,
        }

    return incoming.model_copy(
        deep=True,
        update={
            "artifacts": merged_artifacts,
            "child_workflows": merged_children,
            "checkpoint_boundaries": merged_boundaries,
        },
    )


def _newest_by_key[T](
    artifacts: Sequence[Artifact],
    artifact_class: type[T],
    key: Callable[[T], str],
) -> dict[str, T]:
    """Return the last artifact of one kind for each business identifier.

    Artifact history is append-only and ordered, so the last one carrying an identifier is
    the current state of that thing. Identity comparison at the call site keeps this honest:
    two revisions with identical content are still different artifacts.
    """
    current: dict[str, T] = {}
    for artifact in artifacts:
        if isinstance(artifact, artifact_class):
            current[key(artifact)] = artifact
    return current


def _feature_artifact_from_model(model: FeatureArtifactModel) -> Artifact:
    """Hydrate one strict artifact using its explicit persisted discriminant."""
    try:
        artifact_class = _FEATURE_ARTIFACT_MODELS[model.artifact_type]
    except KeyError as error:
        msg = f"unsupported persisted feature artifact type: {model.artifact_type}"
        raise WorkflowConflictError(msg) from error
    return cast(Artifact, artifact_class.model_validate_json(json.dumps(model.payload)))


def _require_mutable_feature_identity(state: FeatureWorkflowSnapshot) -> None:
    """Keep migrated audit snapshots readable while rejecting every lifecycle write."""
    error = workflow_mutation_identity_error(
        workflow_schema_version=state.workflow_schema_version,
        created_by_build_revision=state.created_by_build_revision,
        last_executor_build_revision=state.last_executor_build_revision,
    )
    if error is not None:
        raise WorkflowConflictError(f"feature snapshot is audit-only: {error}")


def _operation_did_not_happen(operation: str) -> FeatureOperationFailedError:
    """Say that a requested operation was not carried out, without quoting the fault.

    Reached only after the feature has been marked as needing attention, so the diagnosis is
    already durable and the caller is being pointed at it. The text is fixed rather than
    derived from the exception: the exception may be a provider response or a path inside a
    repository, and this string reaches an HTTP client, a chat transcript and the action's
    own error message.
    """
    return FeatureOperationFailedError(
        f"The platform could not {operation}. Nothing was carried out; the feature now "
        "records why it stopped, and asking again will not help until that is addressed."
    )


def _failure_details(error: Exception) -> dict[str, Any]:
    """Record the failure type always, and explanatory text only when declared safe.

    An arbitrary exception's text can carry credential-bearing detail, so it stays out of
    persisted state. An error that opts in by carrying ``diagnostics`` has built that text
    from data it validated itself, and without it a planning failure leaves the operator
    with a type name and no artifact to inspect.
    """
    details: dict[str, Any] = {"error_type": type(error).__name__}
    classification = _root_failure_classification(error)
    if classification != type(error).__name__:
        details["failure_classification"] = classification
    diagnostics = safe_error_diagnostics(error)
    if diagnostics:
        details["diagnostics"] = [
            item[:_MAX_DIAGNOSTIC_CHARACTERS] for item in diagnostics[:_MAX_DIAGNOSTICS]
        ]
    return details


# What a refusal must never say. `retry_refusal_reason` is the field an escalation reads to
# explain why no further attempt was scheduled; `decide_child_retry` also produces a reason
# for the *positive* verdict, and one caller wrote that one unconditionally. Six production
# workstreams therefore stopped with "Attempt N may proceed with a changed strategy" as the
# stated reason they stopped. Matched on the phrase rather than the exact sentence, because
# the verdict contract owns that wording and this check must not depend on it.
_PROCEEDING_REFUSAL_MARKER = "may proceed"


def refusal_states_work_may_proceed(reason: str | None) -> bool:
    """Return whether a stop reason contradicts itself by saying work may continue."""
    return reason is not None and _PROCEEDING_REFUSAL_MARKER in reason.lower()


def _reject_reintroduced_refusal(
    persisted: FeatureWorkflowSnapshot, state: FeatureWorkflowSnapshot
) -> None:
    """Refuse a write that gives a workstream a stop reason saying it may continue.

    The invariant: ``retry_refusal_reason`` describes a stop, never a permission. A path that
    writes a positive verdict there produces a durable record telling whoever now owns a dead
    feature that its next attempt may proceed, which is what AB-Feature-110 said.

    Enforced against what is already persisted rather than absolutely. Six workstreams carry
    such a reason from before this check existed, and raising on them would make those
    features unwritable -- a cancellation or a recovery sweep would fail on a diagnostic
    string. A value identical to the one already stored is therefore carried through
    untouched; anything a writer *introduces* is refused. No path can introduce one now, so
    nothing new can ever become "already persisted".
    """
    for repository_id, child in state.child_workflows.items():
        if not refusal_states_work_may_proceed(child.retry_refusal_reason):
            continue
        previous = persisted.child_workflows.get(repository_id)
        if previous is not None and previous.retry_refusal_reason == child.retry_refusal_reason:
            continue
        msg = (
            "a workstream's retry_refusal_reason must state why it stopped; "
            f"{repository_id!r} was given one saying an attempt may proceed"
        )
        raise WorkflowConflictError(msg)


def _root_failure_classification(error: Exception) -> str:
    """Retain a safe provider subtype declared by an adapter, or use the outer type."""
    declared = getattr(error, "failure_classification", None)
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    return type(error).__name__


def _as_utc(value: datetime) -> datetime:
    """Normalize SQLite's naive timestamp results without altering absolute time."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = ["SqlAlchemyFeatureControlPlane"]


async def _counts_by_feature(
    session: AsyncSession, column: Any, feature_ids: Sequence[str]
) -> dict[str, int]:
    """Count rows per feature for one page, or nothing when the page is empty."""
    if not feature_ids:
        return {}
    rows = (
        await session.execute(
            select(column, func.count()).where(column.in_(feature_ids)).group_by(column)
        )
    ).all()
    return {str(row[0]): int(row[1]) for row in rows}
