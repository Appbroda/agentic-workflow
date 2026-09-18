"""Feature-level control plane with idempotency and request-scoped provider credentials."""

from __future__ import annotations

import hashlib
from base64 import urlsafe_b64decode, urlsafe_b64encode
from binascii import Error as BinasciiError
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from agents.shared.contracts import ARTIFACT_FILENAMES, create_artifact
from api.control_plane import (
    RequestScopedCredentials,
    TimelineEvent,
    WorkflowConflictError,
)
from api.feature_schemas import IntegrationContractRevision, StartFeatureRequest
from api.identity import WorkspaceScope
from api.schemas import ClarificationAnswer
from artifacts.schemas import (
    Artifact,
    ChildWorkflowResultArtifact,
    ContractChangeRequestArtifact,
    FeatureCompletionArtifact,
    IntegrationContractArtifact,
    IntegrationReviewArtifact,
    PRDArtifact,
    PRDAttachment,
    PullRequestArtifact,
    RepositoryRepairProposalArtifact,
)
from runtime_identity import load_runtime_identity
from state.enums import FeatureWorkflowStatus
from state.failure_diagnosis import (
    FeatureFailureClassification,
)
from state.feature_models import FeatureWorkflowSnapshot

# The prefix of the identity people use for a feature. The number after it is allocated by
# the database, never by the caller and never by counting rows.
FEATURE_REFERENCE_PREFIX = "AB-Feature"


def format_feature_reference(number: int) -> str:
    """Return the canonical user-facing reference for one allocated number."""
    return f"{FEATURE_REFERENCE_PREFIX}-{number}"


class FeatureControlPlane(Protocol):
    """Persist additive parent-feature lifecycle data and never retain HTTP header credentials.

    Every method that names a feature accepts a `WorkspaceScope`, and the implementation
    applies it in the SQL `WHERE`: a feature outside the scope is `WorkflowNotFoundError`,
    which the routes already answer `404`. The scope is defaulted here so the callers with no
    request and no actor -- the queue dispatcher, the recovery sweeps, the Slack dispatcher --
    read the deployment, which is what they are for.

    Routes never hold this object. `get_feature_control_plane` hands them a
    `ScopedFeatureControlPlane`, which carries the request's scope and supplies it on every
    call whether the caller passed one or not. That is the boundary: a route added by somebody
    who never read this docstring still cannot reach another person's feature.
    """

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
        """Persist one parent feature, queue its execution durably, and return immediately.

        ``model_setup`` is the already-resolved snapshot of the caller's chosen setup --
        ``{"setup_id", "name", "roles"}`` -- validated by the route before anything is
        accepted. The control plane copies it onto the feature and its queue entry; nothing
        below this point resolves the setup id again (G3).

        ``attachments`` are the submission's images, already resolved against the attachment
        store by the route: existence, ownership and unboundness are settled before anything
        is accepted, and the records here carry the filename, size and hash the PRD artifact
        quotes. They are bound to the feature in the same transaction that writes it.

        ``owner_id`` is whose workspace the feature lands in, and whose stored credentials the
        worker will resolve. Required, because the two identities are the same only here and
        an enqueue with no named owner resolves nobody's keys.
        """

    async def replayed_feature_id(
        self,
        request: StartFeatureRequest,
        *,
        idempotency_key: str | None,
        scope: WorkspaceScope = ...,
    ) -> str | None:
        """Return the feature an identical earlier request already produced, if there is one.

        Asked *before* acceptance by the start route, for one reason: the preflight checks
        it runs are checks on a submission being made for the first time. Re-running them
        against a replay refuses a request that was already accepted -- an attachment bound
        to the very feature this replay is about looks exactly like one somebody else
        submitted. A replay is a request for the answer already given, so the checks whose
        subject is "this is new" are skipped and `start` returns that answer.

        Nothing here is a lock and nothing here decides anything: a differing fingerprint on
        the same key is still `start`'s conflict to raise, and two simultaneous first
        submissions are still serialized there.

        Scoped: `Idempotency-Key` is a client-chosen string, so without the owner filter a
        caller reusing somebody else's header would be handed that person's feature -- and a
        replay answers with the whole record.
        """

    async def execute_queued(
        self,
        feature_id: str,
        *,
        credentials: RequestScopedCredentials,
        intent: str = "start",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Run one already-persisted feature: start it, resume it, or retry a workstream."""

    async def fail_queued(
        self,
        feature_id: str,
        *,
        event: str,
        reason: str,
        error_type: str = FeatureFailureClassification.FEATURE_QUEUE_REFUSED.value,
    ) -> None:
        """Record that a queued feature could not be started, and why."""

    async def resume(
        self,
        feature_id: str,
        *,
        answers: Sequence[ClarificationAnswer],
        credentials: RequestScopedCredentials,
        scope: WorkspaceScope = ...,
    ) -> FeatureRecord:
        """Resume a parent clarification loop.

        No acting-identity argument. The queue entry names the feature's owner, read off the
        row inside, because that is whose credentials the resumed run must spend.
        """

    async def retry_workstream(
        self,
        feature_id: str,
        repository_id: str,
        *,
        additional_attempts: int,
        requested_by: str,
        reason: str,
        credentials: RequestScopedCredentials,
        scope: WorkspaceScope = ...,
    ) -> FeatureRecord:
        """Grant one stopped repository more attempts and run it again.

        ``requested_by`` is the audit sentence naming who granted it, and travels in the
        queue entry's payload. It is not the queue's identity -- see the implementation.
        """

    async def publish_feature(
        self,
        feature_id: str,
        *,
        requested_by: str,
        reason: str,
        credentials: RequestScopedCredentials,
        scope: WorkspaceScope = ...,
    ) -> FeatureRecord:
        """Open the pull requests a person decided a feature that did not land should have."""

    async def revise_feature(
        self,
        feature_id: str,
        *,
        request: str,
        requested_by: str,
        credentials: RequestScopedCredentials,
        scope: WorkspaceScope = ...,
    ) -> FeatureRecord:
        """Re-open a completed feature so a person's change request runs as its next revision.

        The new run modifies the published work on `-V{n}` branches and, once its own pull
        requests are read back from the provider, closes the superseded ones.
        """

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
        scope: WorkspaceScope = ...,
    ) -> FeatureRecord:
        """Record which position holds in one design conflict and queue the attempt it frees."""

    async def cancel(
        self,
        feature_id: str,
        *,
        requested_by: str,
        reason: str | None = None,
        scope: WorkspaceScope = ...,
    ) -> FeatureRecord:
        """Cancel future parent and child work without deleting audit artifacts."""

    async def retire(
        self,
        feature_id: str,
        *,
        reason: str,
        operator: str,
        scope: WorkspaceScope = ...,
    ) -> FeatureRecord:
        """Record an operator closing out a feature the platform will not execute again."""

    async def approve_contract_change(
        self,
        feature_id: str,
        *,
        request_id: str,
        revision: IntegrationContractRevision,
        resolution: str,
        credentials: RequestScopedCredentials,
        scope: WorkspaceScope = ...,
    ) -> FeatureRecord:
        """Apply one explicit human-approved immutable contract revision."""

    async def reject_contract_change(
        self,
        feature_id: str,
        *,
        request_id: str,
        resolution: str,
        credentials: RequestScopedCredentials,
        scope: WorkspaceScope = ...,
    ) -> FeatureRecord:
        """Reject one contract deviation request without guessing a code change."""

    async def approve_repair(
        self,
        feature_id: str,
        *,
        repair_id: str,
        actor_id: str,
        credentials: RequestScopedCredentials,
        scope: WorkspaceScope = ...,
    ) -> FeatureRecord:
        """Apply one approved repository repair and retry the repository it was blocking."""

    async def reject_repair(
        self,
        feature_id: str,
        *,
        repair_id: str,
        actor_id: str,
        reason: str,
        scope: WorkspaceScope = ...,
    ) -> FeatureRecord:
        """Record that a person declined one repository repair."""

    async def repairs(
        self, feature_id: str, *, scope: WorkspaceScope = ...
    ) -> list[RepositoryRepairProposalArtifact]:
        """Return the current state of every repair proposed for this feature."""

    async def get_record(self, feature_id: str, *, scope: WorkspaceScope = ...) -> FeatureRecord:
        """Return one durable parent state."""

    async def artifacts(self, feature_id: str, *, scope: WorkspaceScope = ...) -> list[Artifact]:
        """Return parent and child artifacts."""

    async def workstreams(self, feature_id: str, *, scope: WorkspaceScope = ...) -> list[Any]:
        """Return independently persisted child references."""

    async def pull_requests(
        self, feature_id: str, *, scope: WorkspaceScope = ...
    ) -> list[PullRequestArtifact]:
        """Return coordinated pull requests already created for the feature."""

    async def timeline(
        self, feature_id: str, *, scope: WorkspaceScope = ...
    ) -> list[tuple[datetime, str, str, str, dict[str, Any]]]:
        """Return a credential-free parent history."""

    async def list_features(
        self, *, limit: int, cursor: str | None, scope: WorkspaceScope = ...
    ) -> FeaturePage:
        """Return one page of feature summaries, newest first."""

    async def events_after(
        self,
        feature_id: str,
        *,
        after_id: int | None,
        limit: int,
        scope: WorkspaceScope = ...,
    ) -> list[FeatureEvent]:
        """Return durable lifecycle events after a cursor, cheaply."""

    async def require_visible(self, feature_id: str, *, scope: WorkspaceScope = ...) -> None:
        """Raise `WorkflowNotFoundError` unless one feature is in this caller's workspace.

        For the routes whose data lives outside the feature graph -- the chat transcript, the
        durable action records -- so they gate on the same predicate against the same column
        without hydrating parent state.
        """

    async def owner_of(self, feature_id: str) -> str:
        """Return whose workspace one feature is in, whoever is asking."""


class ScopedFeatureControlPlane:
    """One request's view of the control plane, narrowed to one workspace.

    This is the boundary that makes workspace isolation something a route cannot forget.
    `get_feature_control_plane` -- the dependency every feature route resolves -- returns one
    of these, so no route ever holds the unscoped store. Every method here injects this
    request's scope and *ignores* whatever the caller passed, which is deliberately stronger
    than a required parameter: a required parameter can be satisfied with
    `WorkspaceScope.unscoped()` by somebody in a hurry, and this cannot be.

    It is not where the filter lives. The filter is in the SQL `WHERE` inside the store, one
    predicate at `_require_model`; this only guarantees the store is always asked the narrow
    question. Two layers because they answer different risks: the `WHERE` means the answer
    cannot be wrong, and this means the question cannot be omitted.

    Reads pass the scope as built; **mutations pass `scope.for_mutation()`**, which drops
    `may_read_any`. That is where "an administrator may read any workspace and act in only
    their own" is implemented, and it is one line per method here rather than a rule each
    route remembers. See `WorkspaceScope.for_mutation` for why acting is not a read.

    Not a `dataclass` because it is a delegating facade rather than a value, and not
    `__getattr__`-based forwarding because that would silently pass a *new* store method
    through unscoped -- the whole failure this class exists to prevent. Each method is
    written out, so adding one to the store and forgetting it here is an
    `AttributeError` in a test rather than a leak in production.
    """

    __slots__ = ("_inner", "_scope")

    def __init__(self, inner: FeatureControlPlane, scope: WorkspaceScope) -> None:
        """Bind the deployment-wide control plane and the workspace this request may reach."""
        self._inner = inner
        self._scope = scope

    @property
    def scope(self) -> WorkspaceScope:
        """Return the workspace this view is narrowed to."""
        return self._scope

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
        """Create a feature in this request's workspace.

        `owner_id` is passed through rather than replaced with `self._scope.owner_id`, and
        that is not an oversight: the route takes it from `actor.actor_id`, which is the same
        value, and leaving it explicit keeps the store's signature honest about needing an
        owner. An administrator's scope reads every workspace but still creates in their own.
        """
        return await self._inner.start(
            request,
            idempotency_key=idempotency_key,
            credentials=credentials,
            owner_id=owner_id,
            model_setup=model_setup,
            attachments=attachments,
        )

    async def replayed_feature_id(
        self, request: StartFeatureRequest, *, idempotency_key: str | None
    ) -> str | None:
        """Return the feature an identical earlier request in this workspace produced."""
        return await self._inner.replayed_feature_id(
            request, idempotency_key=idempotency_key, scope=self._scope
        )

    async def resume(
        self,
        feature_id: str,
        *,
        answers: Sequence[ClarificationAnswer],
        credentials: RequestScopedCredentials,
    ) -> FeatureRecord:
        """Resume a feature in this workspace."""
        return await self._inner.resume(
            feature_id,
            answers=answers,
            credentials=credentials,
            scope=self._scope.for_mutation(),
        )

    async def retry_workstream(
        self,
        feature_id: str,
        repository_id: str,
        *,
        additional_attempts: int,
        requested_by: str,
        reason: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureRecord:
        """Grant a retry on a feature in this workspace."""
        return await self._inner.retry_workstream(
            feature_id,
            repository_id,
            additional_attempts=additional_attempts,
            requested_by=requested_by,
            reason=reason,
            credentials=credentials,
            scope=self._scope.for_mutation(),
        )

    async def publish_feature(
        self,
        feature_id: str,
        *,
        requested_by: str,
        reason: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureRecord:
        """Publish a feature in this workspace."""
        return await self._inner.publish_feature(
            feature_id,
            requested_by=requested_by,
            reason=reason,
            credentials=credentials,
            scope=self._scope.for_mutation(),
        )

    async def revise_feature(
        self,
        feature_id: str,
        *,
        request: str,
        requested_by: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureRecord:
        """Revise a completed feature in this workspace."""
        return await self._inner.revise_feature(
            feature_id,
            request=request,
            requested_by=requested_by,
            credentials=credentials,
            scope=self._scope.for_mutation(),
        )

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
    ) -> FeatureRecord:
        """Decide a design conflict on a feature in this workspace."""
        return await self._inner.answer_design_conflict(
            feature_id,
            conflict_id,
            verdict=verdict,
            decision=decision,
            decided_by=decided_by,
            additional_attempts=additional_attempts,
            credentials=credentials,
            scope=self._scope.for_mutation(),
        )

    async def cancel(
        self, feature_id: str, *, requested_by: str, reason: str | None = None
    ) -> FeatureRecord:
        """Cancel a feature in this workspace."""
        return await self._inner.cancel(
            feature_id,
            requested_by=requested_by,
            reason=reason,
            scope=self._scope.for_mutation(),
        )

    async def retire(self, feature_id: str, *, reason: str, operator: str) -> FeatureRecord:
        """Retire a feature in this workspace."""
        return await self._inner.retire(
            feature_id,
            reason=reason,
            operator=operator,
            scope=self._scope.for_mutation(),
        )

    async def approve_contract_change(
        self,
        feature_id: str,
        *,
        request_id: str,
        revision: IntegrationContractRevision,
        resolution: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureRecord:
        """Approve a contract revision on a feature in this workspace."""
        return await self._inner.approve_contract_change(
            feature_id,
            request_id=request_id,
            revision=revision,
            resolution=resolution,
            credentials=credentials,
            scope=self._scope.for_mutation(),
        )

    async def reject_contract_change(
        self,
        feature_id: str,
        *,
        request_id: str,
        resolution: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureRecord:
        """Reject a contract revision on a feature in this workspace."""
        return await self._inner.reject_contract_change(
            feature_id,
            request_id=request_id,
            resolution=resolution,
            credentials=credentials,
            scope=self._scope.for_mutation(),
        )

    async def approve_repair(
        self,
        feature_id: str,
        *,
        repair_id: str,
        actor_id: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureRecord:
        """Approve a repair on a feature in this workspace."""
        return await self._inner.approve_repair(
            feature_id,
            repair_id=repair_id,
            actor_id=actor_id,
            credentials=credentials,
            scope=self._scope.for_mutation(),
        )

    async def reject_repair(
        self, feature_id: str, *, repair_id: str, actor_id: str, reason: str
    ) -> FeatureRecord:
        """Reject a repair on a feature in this workspace."""
        return await self._inner.reject_repair(
            feature_id,
            repair_id=repair_id,
            actor_id=actor_id,
            reason=reason,
            scope=self._scope.for_mutation(),
        )

    async def repairs(self, feature_id: str) -> list[RepositoryRepairProposalArtifact]:
        """Return the repairs proposed for a feature in this workspace."""
        return await self._inner.repairs(feature_id, scope=self._scope)

    async def get_record(self, feature_id: str) -> FeatureRecord:
        """Return one feature in this workspace."""
        return await self._inner.get_record(feature_id, scope=self._scope)

    async def artifacts(self, feature_id: str) -> list[Artifact]:
        """Return the artifacts of a feature in this workspace."""
        return await self._inner.artifacts(feature_id, scope=self._scope)

    async def workstreams(self, feature_id: str) -> list[Any]:
        """Return the workstreams of a feature in this workspace."""
        return await self._inner.workstreams(feature_id, scope=self._scope)

    async def pull_requests(self, feature_id: str) -> list[PullRequestArtifact]:
        """Return the pull requests of a feature in this workspace."""
        return await self._inner.pull_requests(feature_id, scope=self._scope)

    async def timeline(
        self, feature_id: str
    ) -> list[tuple[datetime, str, str, str, dict[str, Any]]]:
        """Return the history of a feature in this workspace."""
        return await self._inner.timeline(feature_id, scope=self._scope)

    async def list_features(self, *, limit: int, cursor: str | None) -> FeaturePage:
        """Return one page of this workspace's features."""
        return await self._inner.list_features(limit=limit, cursor=cursor, scope=self._scope)

    async def events_after(
        self, feature_id: str, *, after_id: int | None, limit: int
    ) -> list[FeatureEvent]:
        """Return lifecycle events for a feature in this workspace."""
        return await self._inner.events_after(
            feature_id, after_id=after_id, limit=limit, scope=self._scope
        )

    async def require_visible(self, feature_id: str) -> None:
        """Raise unless one feature is in this workspace."""
        await self._inner.require_visible(feature_id, scope=self._scope)

    async def owner_of(self, feature_id: str) -> str:
        """Return whose workspace one feature is in."""
        return await self._inner.owner_of(feature_id)


@dataclass(frozen=True, slots=True)
class FeatureEvent:
    """One persisted lifecycle event and the identifier a client resumes from."""

    id: int
    timestamp: datetime
    event_type: str
    source: str
    event: str
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FeatureSummary:
    """Enough to find a feature again, without hydrating its artifacts.

    Deliberately built from indexed columns only. Reading full parent state for every row
    would make a list of recent work more expensive than the work itself.
    """

    feature_id: str
    workflow_id: str
    title: str
    status: FeatureWorkflowStatus
    execution_mode: str
    agent_platform: str
    created_at: datetime
    updated_at: datetime
    # The cost basis beside the platform. Defaulted for the reason the column backfill is:
    # every summary row that predates the field describes a feature that ran at `high`.
    performance_tier: str = "high"
    # The identity a person searches for and quotes. `None` only for a feature created before
    # references existed and not yet backfilled.
    reference: str | None = None
    # Counted from the two tables already indexed by feature_id. `current_agent` is not here
    # on purpose: it lives inside `state_json`, which for a completed feature carries every
    # artifact, so reading it per row would make listing recent work cost more than the work.
    repository_count: int = 0
    pull_request_count: int = 0
    human_action_required: bool = False
    dashboard_group: str = "running"


@dataclass(frozen=True, slots=True)
class FeaturePage:
    """One page of summaries and the cursor that continues it."""

    features: list[FeatureSummary]
    next_cursor: str | None


def encode_feature_cursor(summary: FeatureSummary) -> str:
    """Encode the keyset position after a row, so paging cannot skip or repeat on new writes."""
    raw = f"{summary.created_at.isoformat()}|{summary.feature_id}"
    return urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def decode_feature_cursor(cursor: str) -> tuple[datetime, str]:
    """Reject a malformed cursor rather than silently returning the first page again."""
    try:
        raw = urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
        timestamp, _, feature_id = raw.partition("|")
        parsed = datetime.fromisoformat(timestamp)
    except (ValueError, UnicodeDecodeError, BinasciiError) as error:
        msg = "pagination cursor is not valid"
        raise ValueError(msg) from error
    if not feature_id:
        msg = "pagination cursor is not valid"
        raise ValueError(msg)
    return parsed, feature_id


@dataclass(slots=True)
class FeatureRecord:
    """The mutable parent snapshot and append-only lifecycle events."""

    state: FeatureWorkflowSnapshot
    created_at: datetime
    updated_at: datetime
    lifecycle_events: list[TimelineEvent] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FeatureStartResult:
    """Feature start response carrying idempotent creation status."""

    record: FeatureRecord
    created: bool


def _initial_feature_state(
    feature_id: str,
    request: StartFeatureRequest,
    *,
    reference: str | None = None,
    model_setup: dict[str, Any] | None = None,
    attachments: Sequence[PRDAttachment] | None = None,
) -> FeatureWorkflowSnapshot:
    """Create a parent PRD artifact and no child state before the contract is approved.

    A feature pinned to a model setup records `custom` in the NOT NULL tier column -- writing
    `high` there would be a lie the cost records are built on -- and its `agent_platform` is
    the coding role's platform: the platform the implementation work runs on, and the value
    every existing reader of that column keeps working against. Everything credential-shaped
    asks the setup's platform *set* instead.
    """
    now = datetime.now(UTC)
    runtime = load_runtime_identity()
    agent_platform: Literal["openai", "anthropic"] = request.agent_platform
    performance_tier: Literal["low", "medium", "high", "ultra", "custom"] = request.performance_tier
    setup_id: str | None = None
    if model_setup is not None:
        setup_id = str(model_setup.get("setup_id") or "") or None
        performance_tier = "custom"
        coding = model_setup.get("roles", {}).get("coding", {})
        # The route validated the snapshot, so the coding platform is one of the two; the
        # fallback keeps a malformed snapshot from crashing acceptance rather than execution.
        agent_platform = "anthropic" if str(coding.get("platform")) == "anthropic" else "openai"
    payload = request.prd.model_dump(mode="python")
    # The submission carried references -- an id, a marker, a caption. The artifact carries
    # the *record*: the same three plus the filename, size and hash the store resolved, so
    # what was submitted stays readable after the bytes are gone. The route resolved them, in
    # the order the model will be shown them, and the submission's own reference shapes are
    # replaced rather than merged with.
    payload["attachments"] = [item.model_dump(mode="python") for item in (attachments or ())]
    prd = create_artifact(
        PRDArtifact,
        workflow_id=feature_id,
        artifact_id=ARTIFACT_FILENAMES["prd"],
        producer="api",
        payload=payload,
        metadata={"source": "api", "feature_id": feature_id},
    )
    return FeatureWorkflowSnapshot(
        feature_id=feature_id,
        workflow_id=feature_id,
        workflow_schema_version=runtime.workflow_schema_version,
        created_by_build_revision=runtime.build_revision,
        last_executor_build_revision=runtime.build_revision,
        status=FeatureWorkflowStatus.PENDING,
        title=request.prd.title,
        reference=reference,
        repository_specs=request.repositories,
        artifacts=[prd],
        max_integration_review_cycles=5,
        max_child_review_cycles=5,
        max_implementation_retries=4,
        max_validation_retries=2,
        max_repository_setup_retries=1,
        max_contract_revision_cycles=3,
        execution_mode=request.execution_mode,
        agent_platform=agent_platform,
        performance_tier=performance_tier,
        model_setup_snapshot=model_setup,
        model_setup_id=setup_id,
        created_at=now,
        updated_at=now,
    )


def _contract_revision_artifact(
    state: FeatureWorkflowSnapshot, revision: IntegrationContractRevision
) -> IntegrationContractArtifact:
    """Attach the immutable server-controlled envelope to a human-approved revision payload."""
    # Count-based rather than `contract_revision_cycles + 2`: the two are equal on every
    # feature that has never been revised (one original contract plus one per approved
    # change), and only the count stays collision-free once a feature revision's planning
    # pass has appended contracts of its own.
    next_number = (
        sum(1 for item in state.artifacts if isinstance(item, IntegrationContractArtifact)) + 1
    )
    return create_artifact(
        IntegrationContractArtifact,
        workflow_id=state.feature_id,
        artifact_id=f"009_integration_contract.v{next_number}.json",
        producer="human_contract_owner",
        payload={
            "feature_id": state.feature_id,
            "status": "approved",
            "approved_at": datetime.now(UTC),
            "openapi_document": None,
            **revision.model_dump(mode="python"),
        },
        metadata={
            "source": "feature contract-change approval",
            # A contract approved mid-revision belongs to that revision, or the `next_step`
            # plan gate would read it as stale and re-plan on every claim.
            **({"feature_revision": state.revision} if state.revision else {}),
        },
    )


def feature_awaits_human(status: FeatureWorkflowStatus) -> bool:
    """Report whether a feature is stopped on a person rather than on the platform.

    Derived from the status alone so a listing never pays for it. These are the two states in
    which nothing moves until somebody acts, which is what a dashboard has to make prominent.
    """
    return status in {
        FeatureWorkflowStatus.WAITING_FOR_HUMAN,
        FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
    }


def feature_dashboard_group(status: FeatureWorkflowStatus) -> str:
    """Return the server-owned dashboard category for one lifecycle state."""
    if status is FeatureWorkflowStatus.PENDING:
        # Accepted and durably queued, but nothing has run yet. Counted apart from "running"
        # because the distinction is the one somebody watching a fresh submission cares about:
        # whether the platform has picked it up.
        return "queued"
    if status in {
        FeatureWorkflowStatus.WAITING_FOR_HUMAN,
        FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    }:
        return "waiting"
    if status is FeatureWorkflowStatus.COMPLETED:
        return "completed"
    if status is FeatureWorkflowStatus.FAILED:
        return "failed"
    if status is FeatureWorkflowStatus.CANCELLED:
        return "cancelled"
    return "running"


def _feature_fingerprint(request: StartFeatureRequest) -> str:
    """Hash only body data and never provider headers for feature idempotency."""
    return hashlib.sha256(request.model_dump_json(exclude_none=True).encode("utf-8")).hexdigest()


def _feature_idempotency_key(idempotency_key: str | None, fingerprint: str, owner_id: str) -> str:
    """Use one explicit non-empty key or a safe deterministic payload fallback, per owner.

    The owner is part of the key, and has to be. Both halves of this were cross-workspace
    leaks without it:

    * `Idempotency-Key` is a string the client chooses. Two people choosing `retry-1` shared
      a key, and a replay is answered by returning the recorded feature -- so the second
      submitter was handed the first's feature through the ordinary success path.
    * the fallback is a hash of the request body alone, so two people submitting the *same*
      PRD collided with no header involved at all. That one needs no malice to happen.

    One consequence, stated because it is a real behaviour change: keys recorded before this
    was namespaced no longer match. A submission replayed across the deploy creates a new
    feature instead of returning the old one. That is the safe direction -- a duplicate
    feature is visible and retirable; a leaked one is not.
    """
    if idempotency_key is None:
        return f"payload:{owner_id}:{fingerprint}"
    if not idempotency_key.strip():
        msg = "Idempotency-Key must not be empty"
        raise WorkflowConflictError(msg)
    return f"header:{owner_id}:{idempotency_key}"


def _event_for_status(status: FeatureWorkflowStatus) -> str:
    """Map terminal and waiting statuses to the published feature observability event set."""
    return {
        FeatureWorkflowStatus.WAITING_FOR_HUMAN: "feature_waiting_for_human",
        FeatureWorkflowStatus.COMPLETED: "feature_completed",
        FeatureWorkflowStatus.FAILED: "feature_failed",
        FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN: "feature_failed",
        FeatureWorkflowStatus.CANCELLED: "feature_cancelled",
    }.get(status, "feature_executed")


def _artifact_events(artifact: Artifact) -> list[tuple[str, dict[str, Any]]]:
    """Translate immutable handoffs into the structured feature observability vocabulary."""
    details: dict[str, Any] = {"artifact_id": artifact.artifact_id}
    if isinstance(artifact, IntegrationContractArtifact):
        details["contract_version"] = artifact.contract_version
        events = [("contract_created", details)]
        if artifact.status == "approved":
            events.append(("contract_approved", details))
        return events
    if isinstance(artifact, ChildWorkflowResultArtifact):
        details.update(
            {
                "child_workflow_id": artifact.child_workflow_id,
                "repository_id": artifact.repository_id,
            }
        )
        completion_event = (
            "child_workflow_completed" if artifact.status == "approved" else "child_workflow_failed"
        )
        return [("child_workflow_started", details), (completion_event, details)]
    if isinstance(artifact, IntegrationReviewArtifact):
        details["review_status"] = artifact.review_status
        events = [
            ("integration_review_started", details),
            ("integration_review_completed", details),
        ]
        if artifact.review_status == "changes_requested":
            events.append(("repository_fix_requested", details))
        return events
    if isinstance(artifact, ContractChangeRequestArtifact):
        details["repository_id"] = artifact.requested_by_repository_id
        return [("contract_change_requested", details)]
    if isinstance(artifact, PullRequestArtifact):
        details["repository"] = artifact.repository
        return [("pull_request_created", details)]
    if isinstance(artifact, FeatureCompletionArtifact):
        return [("feature_completed", details)]
    return []


__all__ = [
    "FEATURE_REFERENCE_PREFIX",
    "FeatureControlPlane",
    "FeatureRecord",
    "FeatureStartResult",
    "ScopedFeatureControlPlane",
    "format_feature_reference",
]
