"""Execution of workflow-changing actions under a durable claim.

The platform runs an action inside the request that asked for it -- there is no worker queue,
and adding one would change how every existing mutation behaves. That is workable, but it
means the executor is a process that can die while holding work nobody else knows about. This
module is what makes that survivable: the intent is durable before anything runs, ownership is
a lease that stops being renewed when the process stops existing, and the outcome is written
against that same lease so a returning zombie cannot overwrite a reconciled decision.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

import structlog

from services.action_context import (
    bind_feature_action,
    reset_feature_action,
    revoke_feature_action,
)
from services.action_recovery import OperationEvidence
from state.enums import FeatureActionStatus
from state.feature_actions import FeatureAction
from storage.action_store import ActionConflictError, ActionInProgressError

# Renewed at a third of the lease, so two consecutive failures still leave a whole window
# before another executor could take the action.
_RENEWAL_FRACTION = 3.0
# After a failed renewal call, retried at a tenth of the lease instead. That fits several
# more attempts inside the window the lease still has left, which is the difference between
# riding out a brief database stall and abandoning a run that was making progress.
_RENEWAL_RETRY_FRACTION = 10.0


class FeatureActionStore(Protocol):
    """The durable contract both the database and in-memory action stores satisfy."""

    @property
    def lease_seconds(self) -> float:
        """Return how long one claim owns an action without renewal."""

    async def create_or_get(
        self,
        *,
        feature_id: str,
        repository_id: str | None,
        action_type: str,
        actor_id: str,
        actor_display_name: str | None,
        origin: str,
        origin_message_id: int | None,
        payload: dict[str, Any],
        context_version: str | None,
        request_idempotency_key: str | None = None,
        max_attempts: int = 1,
    ) -> tuple[FeatureAction, bool]:
        """Return the action for one intent, creating it only the first time."""

    async def get(self, action_id: str) -> FeatureAction | None:
        """Return one action by identifier."""

    async def list_for_feature(self, feature_id: str, *, limit: int = 100) -> list[FeatureAction]:
        """Return a feature's actions, newest first."""

    async def claim(self, action_id: str, *, owner: str) -> FeatureAction | None:
        """Take ownership when nothing live holds it."""

    async def mark_executing(self, action_id: str, *, owner: str) -> FeatureAction | None:
        """Record that external work has begun under this lease."""

    async def renew_lease(self, action_id: str, *, owner: str) -> bool:
        """Extend ownership, reporting whether it is still held."""

    async def record_success(
        self,
        action_id: str,
        *,
        owner: str,
        result_summary: str,
        external_operation_ids: Sequence[str] = (),
    ) -> FeatureAction:
        """Record a confirmed outcome and release the lease."""

    async def record_domain_result(
        self, action_id: str, *, owner: str, result_summary: str
    ) -> FeatureAction | None:
        """Checkpoint that the domain service returned after committing its state."""

    async def record_failure(
        self,
        action_id: str,
        *,
        owner: str,
        status: FeatureActionStatus,
        error_code: str,
        error_message: str,
    ) -> FeatureAction:
        """Record a failure and release the lease."""

    async def release_unattempted(
        self, action_id: str, *, owner: str, error_code: str
    ) -> FeatureAction | None:
        """Return an action that attempted no external effect to a submittable state."""

    async def list_expired(self, *, limit: int = 100) -> list[FeatureAction]:
        """Return actions whose executor stopped renewing."""

    async def reconcile(
        self,
        action_id: str,
        *,
        status: FeatureActionStatus,
        result_summary: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        external_operation_ids: Sequence[str] = (),
    ) -> FeatureAction | None:
        """Decide an abandoned action without holding its lease."""

    async def operator_reconcile(
        self,
        action_id: str,
        *,
        actor_id: str,
        succeeded: bool,
        reason: str,
    ) -> FeatureAction | None:
        """Record a person's evidence-based decision about an uncertain action."""


class ActionLeaseLostError(RuntimeError):
    """Raised when an executor can no longer prove it owns the action it is running."""


class FeatureActionService:
    """Submit an intent durably, then run it exactly once under a claim."""

    def __init__(
        self,
        *,
        store: FeatureActionStore,
        journal: OperationEvidence | None = None,
        build_revision: str = "local",
    ) -> None:
        """Bind the durable store, the effect evidence, and this build's identity."""
        self._store = store
        self._journal = journal
        self._build_revision = build_revision

    @property
    def store(self) -> FeatureActionStore:
        """Expose the store for read paths that do not execute anything."""
        return self._store

    def owner_id(self) -> str:
        """Return an owner identifier unique to this claim, in this process, on this build.

        The build revision is included because a lease held by a build that is no longer
        deployed is information: it says the executor cannot come back.
        """
        return f"{self._build_revision}:{uuid4()}"

    async def submit(
        self,
        *,
        feature_id: str,
        action_type: str,
        actor_id: str,
        payload: dict[str, Any],
        context_version: str | None,
        repository_id: str | None = None,
        actor_display_name: str | None = None,
        origin: str = "rest",
        origin_message_id: int | None = None,
        max_attempts: int = 1,
        request_idempotency_key: str | None = None,
    ) -> tuple[FeatureAction, bool]:
        """Record the intent before anything happens, returning the existing one on a repeat."""
        return await self._store.create_or_get(
            feature_id=feature_id,
            repository_id=repository_id,
            action_type=action_type,
            actor_id=actor_id,
            actor_display_name=actor_display_name,
            origin=origin,
            origin_message_id=origin_message_id,
            payload=payload,
            context_version=context_version,
            request_idempotency_key=request_idempotency_key,
            max_attempts=max_attempts,
        )

    async def execute(
        self,
        action: FeatureAction,
        run: Callable[[], Awaitable[str]],
    ) -> FeatureAction:
        """Run one action once, under a lease, and record what happened.

        An action that already succeeded is replayed rather than repeated -- that is what
        makes a double-submitted confirmation safe.

        One that already failed is refused instead of replayed, and the difference matters.
        A failure may have crossed an external checkpoint before it was reported, so the
        platform does not know how much of it happened; answering "done" would be a claim it
        cannot support, and answering "here, try again" would risk repeating whatever did
        land. Both cases end without a second effect; only the successful one can honestly
        report an outcome. An action awaiting reconciliation is refused for the same reason,
        more strongly.
        """
        current = await self._store.get(action.action_id) or action
        if current.status is FeatureActionStatus.SUCCEEDED:
            return current
        if current.status in {FeatureActionStatus.FAILED, FeatureActionStatus.CANCELLED}:
            msg = (
                f"this action already {current.status.value} and is not repeated "
                "automatically; review the feature's current state before asking again"
            )
            raise ActionConflictError(msg)
        if current.status is FeatureActionStatus.REQUIRES_RECONCILIATION:
            msg = (
                "this action was interrupted and its outcome is unconfirmed; it must be "
                "reconciled before it can be attempted again"
            )
            raise ActionConflictError(msg)

        owner = self.owner_id()
        claimed = await self._store.claim(current.action_id, owner=owner)
        if claimed is None:
            refreshed = await self._store.get(current.action_id)
            if refreshed is not None and refreshed.is_terminal:
                return refreshed
            if refreshed is not None and refreshed.lease_is_live(now=_now()):
                msg = "this action is already being carried out"
                raise ActionInProgressError(msg)
            msg = "this action has used every attempt it was given"
            raise ActionConflictError(msg)

        await self._store.mark_executing(claimed.action_id, owner=owner)
        try:
            token = bind_feature_action(claimed.action_id)
            try:
                result = await self._run_with_lease(claimed.action_id, owner, run)
            finally:
                reset_feature_action(token)
            checkpoint = await self._store.record_domain_result(
                claimed.action_id, owner=owner, result_summary=result
            )
            if checkpoint is None:
                msg = "the action lease was lost before its domain result could be checkpointed"
                raise ActionLeaseLostError(msg)
        except ActionLeaseLostError:
            # Deliberately not recorded here. Writing an outcome requires a live lease, and
            # this executor has just proved it does not have one -- the row stays claimed
            # with an expired lease, which is exactly the shape recovery looks for.
            structlog.get_logger("runtime.actions").warning(
                "feature_action_lease_lost",
                action_type=claimed.action_type,
                attempt=claimed.attempt,
            )
            raise
        except Exception as error:
            if await self._attempted_no_external_effect(claimed.action_id):
                # Nothing crossed an external checkpoint, so there is nothing for the next
                # reader to be careful about and nothing to refuse. The caller still gets the
                # same refusal it always got; what changes is that the identity stays
                # submittable, so a refusal whose own words are "retry shortly" can be.
                await self._store.release_unattempted(
                    claimed.action_id,
                    owner=owner,
                    error_code=f"{claimed.action_type.lower()}_not_attempted",
                )
                structlog.get_logger("runtime.actions").info(
                    "feature_action_released_unattempted",
                    action_type=claimed.action_type,
                    attempt=claimed.attempt,
                    error_type=type(error).__name__,
                )
                raise
            await self._store.record_failure(
                claimed.action_id,
                owner=owner,
                status=FeatureActionStatus.FAILED,
                error_code=f"{claimed.action_type.lower()}_failed",
                # The exception text may carry provider responses or repository paths. Keep
                # the platform-owned category, exactly as the request logger does.
                error_message=(
                    "The platform did not complete this action. Review the current feature "
                    "state before asking for it again."
                ),
            )
            structlog.get_logger("runtime.actions").warning(
                "feature_action_failed",
                action_type=claimed.action_type,
                attempt=claimed.attempt,
                error_type=type(error).__name__,
            )
            raise
        return await self._store.record_success(
            claimed.action_id,
            owner=owner,
            result_summary=result,
            external_operation_ids=await self._operation_ids(claimed.action_id),
        )

    async def _run_with_lease(
        self,
        action_id: str,
        owner: str,
        run: Callable[[], Awaitable[str]],
    ) -> str:
        """Abort the action if this executor stops being able to prove it owns it.

        Without this, a process that was partitioned from the database long enough for its
        lease to expire would carry on making changes while recovery decided the action was
        abandoned -- which is the one way this design could produce the duplicate effects it
        is meant to prevent.
        """

        # Wrapped rather than scheduled directly: the callable is typed as returning an
        # awaitable, and only a coroutine can be turned into a task.
        async def invoke() -> str:
            return await run()

        work: asyncio.Task[str] = asyncio.create_task(invoke())
        renewal = asyncio.create_task(self._renew(action_id, owner))
        try:
            done, _pending = await asyncio.wait(
                {work, renewal}, return_when=asyncio.FIRST_COMPLETED
            )
            if renewal in done:
                # Revoked before it is cancelled, and deliberately in that order. Cancelling
                # is a request: a coroutine blocked in a fifteen-minute test command unwinds
                # when that command ends, not when `cancel()` is called, and -108 wrote a
                # whole further attempt into the feature inside that window. Revoking first
                # means anything the dying run still tries to write is refused.
                revoke_feature_action(action_id)
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
                msg = "the action lease could not be renewed while it was running"
                raise ActionLeaseLostError(msg) from renewal.exception()
            return await work
        finally:
            work.cancel()
            renewal.cancel()
            await asyncio.gather(work, renewal, return_exceptions=True)

    async def _renew(self, action_id: str, owner: str) -> None:
        """Hold the lease open, returning only when ownership is genuinely lost.

        A renewal *call* that fails is not a lease that is lost. This loop used to treat
        the two as the same thing, so any transient database error -- a pool checkout that
        waited past its timeout while the work it is guarding held every other connection --
        ended the loop, and `_run_with_lease` read that as lost ownership and cancelled the
        run. AB-Feature-108 died that way after seventy minutes and nine coding attempts:
        one missed renewal, and a feature that was still working was abandoned mid-attempt.

        So a failure is retried, faster than the normal cadence, for as long as the lease
        could still be ours. Only two things end this loop: the store answering that
        somebody else owns the action, or enough wall-clock passing that the lease has
        objectively expired and another executor would be entitled to claim it.
        """
        lease_seconds = self._store.lease_seconds
        interval = lease_seconds / _RENEWAL_FRACTION
        retry_interval = lease_seconds / _RENEWAL_RETRY_FRACTION
        logger = structlog.get_logger("runtime.actions")
        last_renewed = time.monotonic()
        while True:
            await asyncio.sleep(interval)
            try:
                held = await self._store.renew_lease(action_id, owner=owner)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                unrenewed = time.monotonic() - last_renewed
                if unrenewed >= lease_seconds:
                    # The lease has expired whatever the database now says. Stopping is the
                    # honest answer: recovery may already have decided this action was
                    # abandoned, and continuing would be the zombie this design forbids.
                    logger.warning(
                        "feature_action_lease_renewal_abandoned",
                        error_type=type(error).__name__,
                        unrenewed_seconds=round(unrenewed, 1),
                    )
                    return
                logger.warning(
                    "feature_action_lease_renewal_retried",
                    error_type=type(error).__name__,
                    unrenewed_seconds=round(unrenewed, 1),
                )
                interval = retry_interval
                continue
            if not held:
                return
            last_renewed = time.monotonic()
            interval = lease_seconds / _RENEWAL_FRACTION

    async def _attempted_no_external_effect(self, action_id: str) -> bool:
        """Whether the platform can *prove* this action crossed no external checkpoint.

        The exact worry `execute`'s docstring states -- "a failure may have crossed an
        external checkpoint before it was reported, so the platform does not know how much of
        it happened" -- answered with the platform's own evidence rather than reasoned about.
        Every live side effect journals its intent and links it to the running action before
        it is allowed to happen, so an action with no linked operation attempted none. That
        is a fact on the record, not a list of exception classes: a pre-flight refusal added
        years from now inherits this without anybody remembering to register it.

        Deliberately not "was a lease taken". The lock that serialises durable mutations is
        acquired *inside* the domain call, which runs under the lease -- so every conflict it
        raises has a lease behind it, and a lease proves only that the platform started
        looking, never that it did anything.

        Proof is required, not absence of contradiction. No journal to consult means the
        question cannot be answered, and the answer is then the careful one: an effect nobody
        confirmed is never repeated on a guess. A committed domain result says the domain
        call returned having changed state, which settles it the other way.
        """
        if self._journal is None:
            return False
        current = await self._store.get(action_id)
        if current is None or current.domain_result_committed_at is not None:
            return False
        try:
            operations = await self._journal.list_operations_for_action(action_id)
        except Exception:
            # The same rule: a journal that cannot be read has not proved anything.
            return False
        return not operations

    async def _operation_ids(self, action_id: str) -> list[str]:
        """Collect the external effects this action caused, for the audit record.

        Best effort by design. Recovery never reads this field -- an executor that crashed
        never wrote it -- so failing to collect it must not fail an action that succeeded.
        """
        if self._journal is None:
            return []
        try:
            operations = await self._journal.list_operations_for_action(action_id)
        except Exception:  # pragma: no cover - audit enrichment must never fail the action
            return []
        return [str(item.operation_id) for item in operations]

    async def operator_reconcile(
        self,
        action_id: str,
        *,
        actor_id: str,
        succeeded: bool,
        reason: str,
    ) -> FeatureAction:
        """Close an uncertain action without repeating it or changing workflow state."""
        reconciled = await self._store.operator_reconcile(
            action_id,
            actor_id=actor_id,
            succeeded=succeeded,
            reason=reason,
        )
        if reconciled is None:
            msg = "only an action requiring reconciliation can be reconciled"
            raise ActionConflictError(msg)
        return reconciled


def action_context_version(state: Any, action_type: str, arguments: dict[str, Any]) -> str | None:
    """Return the state an action was decided against, for its identity.

    This is what makes a repeat a replay rather than a coincidence. Two confirmations of
    "retry the backend" are the same intent only if the repository has not moved and the
    previous attempt has not been spent; once either changes, the second is a new decision
    and must be allowed to execute. Returning ``None`` means the action is idempotent
    regardless of state, which is true of cancellation and of nothing else.
    """
    if action_type == "CANCEL_WORKFLOW":
        return None
    if action_type in {"RESUME_WORKFLOW", "ANSWER_CLARIFICATION"}:
        # A resume is bound to the clarification round it answers. Answering round two with
        # round one's identity would replay the first answer's result.
        return f"clarification:{getattr(state, 'clarification_rounds', 0)}"
    if action_type == "RETRY_WORKSTREAM":
        repository_id = str(arguments.get("repository_id", ""))
        child = getattr(state, "child_workflows", {}).get(repository_id)
        if child is None:
            return f"repository:{repository_id}:absent"
        # Both the code the repository is sitting on and how many attempts it has already
        # been given. A grant made after a failed granted attempt is a second decision.
        return (
            f"repository:{repository_id}"
            f":revision:{child.current_revision or 'unknown'}"
            f":attempts:{child.retry_count}:{child.granted_extra_attempts}"
        )
    if action_type == "ANSWER_DESIGN_VERDICT":
        # The question itself is the context, exactly as a repair proposal is: a conflict is
        # identified by the repository and the fingerprint of the demand being re-litigated, so
        # a second question is necessarily a different identifier and a different decision.
        # Deliberately not the retry key beside it -- that one includes the attempt counters,
        # which would make one operator confirming one decision twice look like two decisions
        # the moment the first attempt landed.
        return f"design-conflict:{arguments.get('conflict_id', '')}"
    if action_type in {"APPROVE_REPOSITORY_REPAIR", "REJECT_REPOSITORY_REPAIR"}:
        # The proposal itself is the context: a superseded proposal is replaced by a new one
        # with a new identifier, so approving that is necessarily a different action.
        return f"repair:{arguments.get('repair_id', '')}"
    if action_type in {"APPROVE_CONTRACT_CHANGE", "REJECT_CONTRACT_CHANGE"}:
        return f"contract-change:{arguments.get('request_id', '')}"
    if action_type == "RETIRE_FEATURE":
        return f"feature-status:{getattr(state, 'status', 'unknown')}"
    if action_type == "PUBLISH_FEATURE":
        # What would be published, and what it would be published from: every repository with
        # no pull request yet, and the revision each is sitting on. Sorted, so the ordering of
        # a dictionary cannot make one decision look like two.
        #
        # Deliberately not the attempt counters the retry key uses. Those move whenever an
        # attempt is spent, and an attempt that changed nothing publishable is not a second
        # decision about publishing. A retry that *did* move a repository changes its
        # revision, which is in here, so that case is allowed to execute again.
        children = getattr(state, "child_workflows", {})
        awaiting = sorted(
            f"{repository_id}:{getattr(child, 'current_revision', None) or 'unknown'}"
            for repository_id, child in children.items()
            if getattr(child, "pull_request_artifact_id", None) is None
        )
        return "publish:" + "|".join(awaiting)
    if action_type == "REVISE_FEATURE":
        # Bound to which revision the feature is on: asking for changes again after a
        # revision completes is a new decision, and a replay of the same request against the
        # same revision is the same one.
        return f"revision:{getattr(state, 'revision', 0)}"
    return None


def _now() -> datetime:
    """Return an aware timestamp for lease comparisons."""
    from datetime import UTC

    return datetime.now(UTC)


__all__ = [
    "ActionLeaseLostError",
    "FeatureActionService",
    "FeatureActionStore",
    "action_context_version",
]
