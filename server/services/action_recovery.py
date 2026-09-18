"""Deciding what happened to actions whose executor died holding them.

The rule this module exists to enforce is the one from the reliability requirements: *never
simply rerun every incomplete action*. An action found in an active status with an expired
lease could be any of six things, and only evidence tells them apart. Guessing "retry" would
duplicate a push; guessing "failed" would tell somebody their repository was never retried
when it was.

The evidence is the external-operation journal, which is written before each side effect
rather than after it -- so it survives exactly the crash that leaves an action stranded.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from state.enums import FeatureActionStatus
from state.external_operations import ExternalOperationStatus, ExternalOperationType
from state.feature_actions import FeatureAction

# Effects that reach somewhere this platform cannot see. If one of these is anything other
# than confirmed-succeeded, the action's outcome is genuinely unknown and a person has to
# look. Local effects are different: the next attempt rebuilds the workspace before it
# writes, so an interrupted one leaves nothing behind to duplicate.
REMOTE_EFFECT_OPERATIONS = frozenset(
    {
        ExternalOperationType.PUSH_BRANCH,
        ExternalOperationType.CREATE_PULL_REQUEST,
        ExternalOperationType.UPDATE_PULL_REQUEST,
        ExternalOperationType.CLOSE_PULL_REQUEST,
        ExternalOperationType.ADD_LABELS,
        ExternalOperationType.ADD_REVIEWERS,
    }
)

# Actions that change nothing outside this platform's own database. Re-running one is
# already idempotent at the control plane -- cancelling a cancelled feature returns it
# unchanged, rejecting a rejected repair is refused -- so an interrupted one is simply
# handed back rather than investigated.
STATE_ONLY_ACTIONS = frozenset({"CANCEL_WORKFLOW", "REJECT_REPOSITORY_REPAIR"})


@dataclass(frozen=True, slots=True)
class ActionVerdict:
    """What recovery concluded about one abandoned action, and why."""

    status: FeatureActionStatus
    finding: str
    detail: str


@dataclass(frozen=True, slots=True)
class ActionRecoverySummary:
    """A safe aggregate suitable for structured logging and readiness."""

    scanned: int
    retryable: int
    completed: int
    failed: int
    unresolved: int


class ActionReconciler(Protocol):
    """The part of the action store recovery needs, and nothing that executes work."""

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
        external_operation_ids: tuple[str, ...] | list[str] = (),
    ) -> FeatureAction | None:
        """Decide an abandoned action without holding its lease."""


class OperationEvidence(Protocol):
    """The one journal read recovery performs.

    Narrowed to a protocol rather than the journal itself because this service must not be
    able to change an operation. It reads what happened; the journal owns what happens.
    """

    async def list_operations_for_action(self, action_id: str) -> list[Any]:
        """Return the exact effects durably linked to one action."""


class ActionRecoveryService:
    """Classify abandoned actions from durable effect evidence, and never replay blindly."""

    def __init__(self, store: ActionReconciler, journal: OperationEvidence) -> None:
        """Bind the action store to reconcile and the journal that holds the evidence."""
        self._store = store
        self._journal = journal
        self._sweep_lock = asyncio.Lock()

    async def recover_abandoned_actions(self, *, limit: int = 100) -> ActionRecoverySummary:
        """Decide every action whose lease has lapsed, one serialized pass at a time."""
        async with self._sweep_lock:
            return await self._recover(limit=limit)

    async def _recover(self, *, limit: int) -> ActionRecoverySummary:
        """Apply one verdict per abandoned action."""
        logger = structlog.get_logger("runtime.actions")
        actions = await self._store.list_expired(limit=limit)
        retryable = completed = failed = unresolved = 0
        for action in actions:
            verdict = await self.classify(action)
            reconciled = await self._store.reconcile(
                action.action_id,
                status=verdict.status,
                result_summary=(
                    verdict.detail if verdict.status is FeatureActionStatus.SUCCEEDED else None
                ),
                error_code=(
                    None if verdict.status is FeatureActionStatus.SUCCEEDED else verdict.finding
                ),
                error_message=(
                    None if verdict.status is FeatureActionStatus.SUCCEEDED else verdict.detail
                ),
            )
            if reconciled is None:
                # The executor renewed between listing and writing, so it is alive after all
                # and still owns this. Leaving it alone is the correct outcome.
                continue
            if verdict.status is FeatureActionStatus.CONFIRMED:
                retryable += 1
            elif verdict.status is FeatureActionStatus.SUCCEEDED:
                completed += 1
            elif verdict.status is FeatureActionStatus.FAILED:
                failed += 1
            else:
                unresolved += 1
            logger.info(
                "feature_action_reconciled",
                action_type=action.action_type,
                attempt=action.attempt,
                finding=verdict.finding,
                outcome=verdict.status.value,
            )
        return ActionRecoverySummary(
            scanned=len(actions),
            retryable=retryable,
            completed=completed,
            failed=failed,
            unresolved=unresolved,
        )

    async def classify(self, action: FeatureAction) -> ActionVerdict:
        """Decide what happened to one abandoned action from its durable effect evidence."""
        if action.domain_result_committed_at is not None:
            return ActionVerdict(
                status=FeatureActionStatus.SUCCEEDED,
                finding="domain_result_committed",
                detail=action.pending_result_summary
                or "The domain operation committed before its executor was interrupted.",
            )
        if action.action_type in STATE_ONLY_ACTIONS:
            return ActionVerdict(
                status=FeatureActionStatus.CONFIRMED,
                finding="state_only_action_interrupted",
                detail=(
                    "This action changes nothing outside the platform and is safe to ask for again."
                ),
            )
        if action.status is FeatureActionStatus.CLAIMED:
            # Claimed but never marked executing. Nothing can have run.
            return ActionVerdict(
                status=FeatureActionStatus.CONFIRMED,
                finding="action_never_started",
                detail="The action was claimed but never began, and can be asked for again.",
            )

        operations = await self._journal.list_operations_for_action(action.action_id)
        if not operations:
            return ActionVerdict(
                status=FeatureActionStatus.REQUIRES_RECONCILIATION,
                finding="domain_result_unconfirmed",
                detail=(
                    "The action began but did not checkpoint a committed domain result. Its "
                    "current workflow state must be checked before it can be attempted again."
                ),
            )

        unknown = [
            item
            for item in operations
            if item.status is ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE
        ]
        if unknown:
            return ActionVerdict(
                status=FeatureActionStatus.REQUIRES_RECONCILIATION,
                finding="unconfirmed_external_effect",
                detail=(
                    f"{len(unknown)} operation(s) may have taken effect outside the platform "
                    "and could not be confirmed. Somebody has to check the repository or the "
                    "provider before this is attempted again."
                ),
            )

        if all(item.status is ExternalOperationStatus.SUCCEEDED for item in operations):
            return ActionVerdict(
                status=FeatureActionStatus.REQUIRES_RECONCILIATION,
                finding="effects_confirmed_domain_result_unconfirmed",
                detail=(
                    f"All {len(operations)} linked external effects completed, but the workflow "
                    "did not checkpoint its committed result. Check workflow state before "
                    "marking this action complete."
                ),
            )

        unsettled_remote = [
            item
            for item in operations
            if item.operation_type in REMOTE_EFFECT_OPERATIONS
            and item.status is not ExternalOperationStatus.SUCCEEDED
        ]
        if unsettled_remote:
            return ActionVerdict(
                status=FeatureActionStatus.REQUIRES_RECONCILIATION,
                finding="partial_remote_effect",
                detail=(
                    "This action was interrupted partway through publishing to a repository "
                    "provider. Check what reached the provider before attempting it again."
                ),
            )

        return ActionVerdict(
            status=FeatureActionStatus.FAILED,
            finding="interrupted_before_completion",
            detail=(
                "The action was interrupted before it finished. Its work happened inside a "
                "workspace the platform rebuilds, so nothing was left behind, but it did not "
                "complete."
            ),
        )


__all__ = [
    "REMOTE_EFFECT_OPERATIONS",
    "STATE_ONLY_ACTIONS",
    "ActionRecoveryService",
    "ActionRecoverySummary",
    "ActionVerdict",
    "OperationEvidence",
]
