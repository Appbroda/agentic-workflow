"""Startup reconciliation for durable but incomplete live external operations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, assert_never

import structlog

from services.cancellation import MockCancellationToken
from services.process_runner import AsyncioProcessRunner
from state.external_operations import (
    CompensationStatus,
    ExternalOperation,
    ExternalOperationStatus,
    ExternalOperationType,
)
from storage.attachment_store import UNBOUND_ATTACHMENT_TTL_SECONDS
from storage.external_operation_store import (
    RECOVERY_DEFECT_ERROR_CODE,
    ExternalOperationJournal,
)


class RecoveryDisposition(StrEnum):
    """What a credential-free sweep is entitled to conclude about an interrupted effect."""

    # The whole effect is inside a workspace this platform rebuilds before it writes again.
    WORKSPACE_LOCAL = "workspace_local"
    # Local evidence -- a checkout, a commit object -- is sufficient and authoritative.
    LOCAL_RECONCILE = "local_reconcile"
    # A remote effect this process has no credentials to judge, and an existing reconcile
    # callback can, on the next credentialed request that re-enters the same code path.
    DEFER_TO_CREDENTIALED = "defer_to_credentialed"
    # A remote effect whose truth no callback can establish. A person has to look.
    MANUAL_REVIEW = "manual_review"
    # A remote effect that is not a gate on anything, and whose unconfirmed state is not a
    # question worth putting to a person.
    TERMINAL_BENIGN = "terminal_benign"


# Total by construction: every ``ExternalOperationType`` appears exactly once. The three
# guards that keep it that way are the ``assert_never`` in ``_recover``, the startup
# assertion below, and the enumeration test in the suite -- a new member cannot be added
# without a disposition, and cannot fall through to "park it as UNKNOWN for a human", which
# is how one interrupted local test run used to refuse every later write on the deployment.
_RECOVERY_POLICY: dict[ExternalOperationType, RecoveryDisposition] = {
    ExternalOperationType.CLONE_REPOSITORY: RecoveryDisposition.LOCAL_RECONCILE,
    ExternalOperationType.CREATE_BRANCH: RecoveryDisposition.WORKSPACE_LOCAL,
    ExternalOperationType.RUN_CODING_EXECUTOR: RecoveryDisposition.WORKSPACE_LOCAL,
    ExternalOperationType.RUN_REVIEWER: RecoveryDisposition.WORKSPACE_LOCAL,
    ExternalOperationType.WRITE_FILE_CHANGES: RecoveryDisposition.WORKSPACE_LOCAL,
    ExternalOperationType.RUN_FORMATTER: RecoveryDisposition.WORKSPACE_LOCAL,
    ExternalOperationType.INSTALL_DEPENDENCIES: RecoveryDisposition.WORKSPACE_LOCAL,
    ExternalOperationType.RUN_LINTER: RecoveryDisposition.WORKSPACE_LOCAL,
    ExternalOperationType.RUN_TYPECHECK: RecoveryDisposition.WORKSPACE_LOCAL,
    ExternalOperationType.RUN_TESTS: RecoveryDisposition.WORKSPACE_LOCAL,
    ExternalOperationType.RUN_BUILD: RecoveryDisposition.WORKSPACE_LOCAL,
    ExternalOperationType.CREATE_COMMIT: RecoveryDisposition.LOCAL_RECONCILE,
    ExternalOperationType.PUSH_BRANCH: RecoveryDisposition.DEFER_TO_CREDENTIALED,
    ExternalOperationType.CREATE_PULL_REQUEST: RecoveryDisposition.DEFER_TO_CREDENTIALED,
    # Despite the name this is ``add_comment`` -- the pull-request cross-link, which the
    # publisher explicitly does not gate on. An unconfirmed one is not worth a person's
    # attention, and the operation is left terminal so no replay can duplicate the comment.
    ExternalOperationType.UPDATE_PULL_REQUEST: RecoveryDisposition.TERMINAL_BENIGN,
    # Closing a superseded pull request is naturally idempotent and provably reconcilable: a
    # credentialed pass reads the pull request's state and settles the row, exactly as it
    # settles an unconfirmed create.
    ExternalOperationType.CLOSE_PULL_REQUEST: RecoveryDisposition.DEFER_TO_CREDENTIALED,
    ExternalOperationType.ADD_LABELS: RecoveryDisposition.DEFER_TO_CREDENTIALED,
    ExternalOperationType.ADD_REVIEWERS: RecoveryDisposition.DEFER_TO_CREDENTIALED,
    # The pre-coding model calls have no effect outside this process: the stages that make
    # them resume from their artifacts, never from these rows. An interrupted one is
    # abandoned in place -- the resumed stage writes a fresh row for its fresh call.
    ExternalOperationType.RUN_PRODUCT_MANAGER: RecoveryDisposition.WORKSPACE_LOCAL,
    ExternalOperationType.RUN_REPOSITORY_RECON: RecoveryDisposition.WORKSPACE_LOCAL,
    ExternalOperationType.RUN_CLARIFICATION_GROUNDING: RecoveryDisposition.WORKSPACE_LOCAL,
    ExternalOperationType.RUN_FEATURE_PLANNER: RecoveryDisposition.WORKSPACE_LOCAL,
    # The design fetch is in that block for that block's reason, and the reasoning is if
    # anything cleaner: it is a *read*. An interrupted one left nothing behind anywhere -- no
    # file changed in Figma, nothing was created, nothing needs compensating -- and the
    # resolution step resumes from its artifact like every other pre-coding stage, writing a
    # fresh row for its fresh call. There is nothing for a person to look at.
    ExternalOperationType.FETCH_DESIGN_REFERENCE: RecoveryDisposition.WORKSPACE_LOCAL,
    # The cross-repository seam call has no effect outside this process either: the step
    # resumes from `_persisted_integration_review`'s artifact, never from this row.
    ExternalOperationType.RUN_INTEGRATION_REVIEW: RecoveryDisposition.WORKSPACE_LOCAL,
}


def recovery_disposition_for(operation_type: ExternalOperationType) -> RecoveryDisposition:
    """Read the declared disposition without exposing the table itself for mutation."""
    return _RECOVERY_POLICY[operation_type]


def assert_recovery_policy_is_total() -> None:
    """Refuse to start a process whose recovery policy cannot decide some operation type."""
    missing = sorted(item.value for item in ExternalOperationType if item not in _RECOVERY_POLICY)
    if missing:
        msg = f"external operation types have no recovery disposition: {', '.join(missing)}"
        raise RuntimeError(msg)


assert_recovery_policy_is_total()

# The wording persisted on every operation only a person can settle. Retained verbatim from
# the branch this policy replaced, so existing rows and this one read the same.
_UNCONFIRMED_EFFECT_REASON = (
    "operation may have completed externally and requires reconciliation before retry"
)

# How long a feature may claim to be running without writing anything before a sweep treats
# it as abandoned. Comfortably longer than the slowest legitimate step -- a repository's own
# test or build command is bounded at 900s, and a coding attempt around it takes minutes more
# -- because the cost of being wrong is killing work that was fine.
_DEFAULT_FEATURE_RUN_GRACE_SECONDS = 2_700.0

# How long a deferred remote effect on a terminal feature may wait for the credentialed
# request that will settle it, before a person is asked about it instead. Long by design --
# a day -- because a failed feature can still be resumed by somebody who comes back to it in
# the morning, and asking early would ask about an effect that was about to be confirmed.
_DEFAULT_DEFERRED_SETTLE_AFTER_SECONDS = 86_400.0


@dataclass(frozen=True, slots=True)
class RecoverySummary:
    """Safe startup-recovery aggregate suitable for readiness and structured logging."""

    scanned: int
    reconciled: int
    cancelled: int
    unknown: int
    failures: int
    deferred: int = 0

    @property
    def has_unresolved_critical_operations(self) -> bool:
        """Report only effects no credentialed request can still settle by itself.

        A deferred remote operation is deliberately excluded: it is waiting for its next
        credentialed request, which is the normal path, not a problem.
        """
        return self.unknown > 0 or self.failures > 0


class ActionRecovery(Protocol):
    """The action-level sweep this service runs after settling operation evidence."""

    async def recover_abandoned_actions(self, *, limit: int = 100) -> Any:
        """Decide every action whose executor stopped renewing its lease."""


class UnconfirmedEffectEscalation(Protocol):
    """The feature-level surface for an effect only a person can now account for."""

    async def flag_unconfirmed_external_effect(
        self,
        feature_id: str,
        *,
        operation_type: str,
        repository_id: str | None,
        error_code: str | None = None,
        reason: str,
    ) -> bool:
        """Stop a feature from continuing publication over an unconfirmed remote effect."""


class AttachmentSweep(Protocol):
    """The one method the attachment store owes the periodic pass."""

    async def sweep_unbound_before(self, cutoff: datetime) -> Any:
        """Delete unbound attachments older than the cutoff and report the table's size."""


class FeatureRunRecovery(Protocol):
    """The feature-level sweep, narrowed to the one method that decides abandoned runs."""

    async def reconcile_abandoned_runs(
        self, *, stale_after_seconds: float, limit: int = 50
    ) -> list[str]:
        """Give a terminal status to features whose executor died without writing one."""

    async def stop_overrunning_runs(
        self, *, runtime_limit_seconds: float, limit: int = 50
    ) -> list[str]:
        """Give a terminal status to features that have been executing for too long."""


class RecoveryService:
    """Settle interrupted operations from local evidence, and defer what only a caller can."""

    def __init__(
        self,
        journal: ExternalOperationJournal,
        *,
        workspace_root: Path | None = None,
        process_runner: AsyncioProcessRunner | None = None,
        action_recovery: ActionRecovery | None = None,
        feature_run_recovery: FeatureRunRecovery | None = None,
        unconfirmed_effect_escalation: UnconfirmedEffectEscalation | None = None,
        # Where a submission's images live. Swept by this loop rather than by a new
        # scheduler: a second periodic task is a second thing to start, supervise and
        # notice the death of, and this one already runs on the interval the sweep wants.
        attachments: AttachmentSweep | None = None,
        feature_run_stale_after_seconds: float = _DEFAULT_FEATURE_RUN_GRACE_SECONDS,
        # How long one feature may execute, however healthy it looks. Zero is off, which is
        # what a composition that only wants the abandoned-run sweep gets.
        feature_runtime_limit_seconds: float = 0.0,
        deferred_operation_settle_after_seconds: float = _DEFAULT_DEFERRED_SETTLE_AFTER_SECONDS,
    ) -> None:
        self._journal = journal
        self._workspace_root = workspace_root.resolve(strict=False) if workspace_root else None
        self._process_runner = process_runner or AsyncioProcessRunner()
        # Actions are reconciled by the same sweep, and deliberately after it: an action's
        # verdict is read off its operations, so those have to have settled first. Ordering
        # them in code rather than in a comment somewhere else is the point of composing here.
        self._action_recovery = action_recovery
        # And features after actions, for the same reason one step further out: a feature is
        # only abandoned if nothing holds a live action lease on it, so the action sweep has
        # to have released the leases of executors that are gone before this one looks.
        self._feature_run_recovery = feature_run_recovery
        # Driven off journal state rather than off what a given pass just wrote, so the
        # startup pass -- which runs before the feature control plane exists -- leaves
        # nothing stranded: the next periodic tick finds the same rows and escalates them.
        self._unconfirmed_effect_escalation = unconfirmed_effect_escalation
        # Absent in every composition that keeps no attachments, which sweeps nothing and is
        # the honest behaviour for a deployment with no bytes to forget.
        self._attachments = attachments
        # The last (bound_bytes, unbound_bytes) pair the sweep logged, so a steady table is
        # logged once and not on every thirty-second tick. Starts at zero held: the empty
        # table is the state nothing needs to hear about.
        self._attachment_bytes_logged: tuple[int, int] = (0, 0)
        if feature_run_stale_after_seconds <= 0:
            msg = "feature_run_stale_after_seconds must be positive"
            raise ValueError(msg)
        self._feature_run_stale_after_seconds = feature_run_stale_after_seconds
        if feature_runtime_limit_seconds < 0:
            msg = "feature_runtime_limit_seconds must not be negative"
            raise ValueError(msg)
        self._feature_runtime_limit_seconds = feature_runtime_limit_seconds
        if deferred_operation_settle_after_seconds <= 0:
            msg = "deferred_operation_settle_after_seconds must be positive"
            raise ValueError(msg)
        self._deferred_operation_settle_after_seconds = deferred_operation_settle_after_seconds
        self._completed = False
        self._last_summary: RecoverySummary | None = None
        self._sweep_lock = asyncio.Lock()

    def bind_feature_run_recovery(
        self,
        recovery: FeatureRunRecovery,
        *,
        stale_after_seconds: float | None = None,
        runtime_limit_seconds: float | None = None,
    ) -> None:
        """Attach the feature sweep after the control plane that performs it exists.

        Startup builds this service before the feature control plane, because the first
        operation-recovery pass has to finish before anything is allowed to execute. So the
        feature sweep is bound afterwards instead of injected, and the periodic pass picks
        it up on its next tick.
        """
        self._feature_run_recovery = recovery
        if stale_after_seconds is not None:
            if stale_after_seconds <= 0:
                msg = "stale_after_seconds must be positive"
                raise ValueError(msg)
            self._feature_run_stale_after_seconds = stale_after_seconds
        if runtime_limit_seconds is not None:
            if runtime_limit_seconds < 0:
                msg = "runtime_limit_seconds must not be negative"
                raise ValueError(msg)
            self._feature_runtime_limit_seconds = runtime_limit_seconds

    def bind_unconfirmed_effect_escalation(self, escalation: UnconfirmedEffectEscalation) -> None:
        """Attach the owning-feature surface, for the same startup-ordering reason."""
        self._unconfirmed_effect_escalation = escalation

    def bind_attachment_sweep(self, attachments: AttachmentSweep) -> None:
        """Attach the attachment store, for the same startup-ordering reason as above."""
        self._attachments = attachments

    @property
    def completed(self) -> bool:
        """Return whether the current process completed its bounded startup-recovery pass."""
        return self._completed

    @property
    def last_summary(self) -> RecoverySummary | None:
        """Return the aggregate without exposing individual repository or provider details."""
        return self._last_summary

    async def recover_incomplete_operations(self) -> RecoverySummary:
        """Resolve only proven local outcomes; never replay an uncertain external side effect."""
        async with self._sweep_lock:
            summary = await self._recover_incomplete_operations()
        if self._action_recovery is not None:
            # Outside the operation lock: this reads the evidence the pass above just
            # settled, and holding the lock across it would serialize two unrelated sweeps.
            await self._action_recovery.recover_abandoned_actions()
        # Before the escalation, so an operation this pass decides a person must look at is
        # put to the feature's owner on the same tick rather than the next one.
        await self._settle_stranded_deferred_operations()
        await self._escalate_unconfirmed_effects()
        await self._recover_abandoned_feature_runs()
        # After the abandoned sweep, so a feature whose executor is actually gone is
        # recorded as that rather than as one that merely ran long. The two conditions can
        # both hold on the same row, and "the process died" is the more useful of the two.
        await self._stop_overrunning_feature_runs()
        # Last, and touching nothing any of the above decided: forgotten uploads are the one
        # thing swept here that is not an interrupted effect.
        await self._sweep_unbound_attachments()
        return summary

    async def _sweep_unbound_attachments(self) -> None:
        """Delete uploads nobody submitted, and account for what the table holds.

        Isolated behind its own try, like the two sweeps above it and for the same reason: a
        fault here must not stop operations and actions from being reconciled.

        The log line is the accounting, and it is the whole of it. There is no operator
        storage surface to add a number to -- what exists is capacity diagnostics and
        structured logs -- and this is the first user-supplied blob in this database, the
        first thing in it that grows without anybody deploying anything. So the number has to
        exist somewhere from day one: "nothing was deleted and the table holds 140 MiB" is
        precisely the sentence an operator needs.

        Emitted when the sweep deleted something or the held bytes changed since the last
        line, and not otherwise: the accounting is news when the table moves, and a table
        holding the same 140 MiB does not become more held by being restated every thirty
        seconds. The last logged line is always the current state.
        """
        if self._attachments is None:
            return
        logger = structlog.get_logger("runtime.recovery")
        try:
            report = await self._attachments.sweep_unbound_before(
                datetime.now(UTC) - timedelta(seconds=UNBOUND_ATTACHMENT_TTL_SECONDS)
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error(
                "prd_attachment_sweep_failed",
                error_type=type(error).__name__,
            )
            return
        held = (report.bound_bytes, report.unbound_bytes)
        if report.rows_deleted or held != self._attachment_bytes_logged:
            logger.info(
                "prd_attachment_storage",
                rows_purged=report.rows_deleted,
                bound_bytes=report.bound_bytes,
                unbound_bytes=report.unbound_bytes,
            )
            self._attachment_bytes_logged = held

    async def _settle_stranded_deferred_operations(self) -> None:
        """Ask a person once about a deferred effect no credentialed request will reach.

        Deferring is the normal path and is what task 24- exists for: the sweep holds no
        credentials, and the next request that re-enters the same code path can prove what
        happened. Task 28- makes that request far more likely, because a crashed feature is
        now continued rather than tombstoned, and a continued feature re-enters the very code
        path that reconciles its own deferred push or pull request.

        What it cannot do is help a feature that ended. Once the feature is terminal there is
        no next request, `list_incomplete_operations` no longer returns the row, and the
        operation sits in the operator queue for ever with no question attached to it. After
        a configured age this converts it to the one conclusion that is honest about that:
        unknown, needs a person. Nothing here guesses the effect and nothing settles it
        silently.
        """
        logger = structlog.get_logger("runtime.recovery")
        try:
            stranded = await self._journal.list_deferred_operations_no_request_will_reach(
                older_than_seconds=self._deferred_operation_settle_after_seconds
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error("deferred_operation_scan_failed", error_type=type(error).__name__)
            return
        for operation in stranded:
            try:
                await self._require_manual_review(
                    operation,
                    "remote operation was interrupted and the feature that owns it ended "
                    "without any credentialed request confirming it",
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error(
                    "deferred_operation_settlement_failed",
                    operation_id=operation.operation_id,
                    operation_type=operation.operation_type.value,
                    error_type=type(error).__name__,
                )
                continue
            logger.warning(
                "deferred_operation_needs_manual_review",
                operation_id=operation.operation_id,
                operation_type=operation.operation_type.value,
                feature_id=operation.feature_id,
                repository_id=operation.repository_id,
            )

    async def _escalate_unconfirmed_effects(self) -> None:
        """Stop a feature continuing over an effect only a person can now account for.

        Localising the blast radius must not mean ignoring the problem. Readiness no longer
        refuses the whole deployment for one unresolved operation, so the feature that owns
        it is the only place left where the condition can be seen -- and it is the right one.
        """
        if self._unconfirmed_effect_escalation is None:
            return
        logger = structlog.get_logger("runtime.recovery")
        try:
            operations = await self._journal.list_operations_requiring_manual_review()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error("unconfirmed_effect_scan_failed", error_type=type(error).__name__)
            return
        escalation = self._unconfirmed_effect_escalation
        for operation in operations:
            if operation.feature_id is None:
                continue
            try:
                flagged = await escalation.flag_unconfirmed_external_effect(
                    operation.feature_id,
                    operation_type=operation.operation_type.value,
                    repository_id=operation.repository_id,
                    reason=operation.error_message or _UNCONFIRMED_EFFECT_REASON,
                    # Which kind of unresolved this is. The sweep's own generic handler
                    # below stamps `recovery_error` when *its* code raised: nothing was
                    # learned about the provider, and escalating that as an unconfirmed
                    # effect asked a feature's owner to inspect their repository because a
                    # defect here threw. The escalation classifies it as a platform defect.
                    error_code=operation.error_code,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error(
                    "unconfirmed_effect_escalation_failed",
                    operation_id=operation.operation_id,
                    operation_type=operation.operation_type.value,
                    error_type=type(error).__name__,
                )
                continue
            if flagged:
                logger.warning(
                    "feature_blocked_on_unconfirmed_external_effect",
                    operation_id=operation.operation_id,
                    operation_type=operation.operation_type.value,
                    feature_id=operation.feature_id,
                    repository_id=operation.repository_id,
                )

    async def _recover_abandoned_feature_runs(self) -> None:
        """Decide features whose executor is gone, without failing the operation sweep.

        Isolated behind its own try: this is the newest and outermost of the three sweeps,
        and a fault here must not stop operations and actions from being reconciled -- those
        are the ones that guard against duplicated external effects.
        """
        if self._feature_run_recovery is None:
            return
        logger = structlog.get_logger("runtime.recovery")
        try:
            reconciled = await self._feature_run_recovery.reconcile_abandoned_runs(
                stale_after_seconds=self._feature_run_stale_after_seconds
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error(
                "feature_run_recovery_sweep_failed",
                error_type=type(error).__name__,
            )
            return
        if reconciled:
            logger.warning(
                "feature_runs_reconciled_as_abandoned",
                count=len(reconciled),
                features=sorted(reconciled),
            )

    async def _stop_overrunning_feature_runs(self) -> None:
        """Stop features past their runtime ceiling, isolated like the sweep above it."""
        if self._feature_run_recovery is None or self._feature_runtime_limit_seconds <= 0:
            return
        logger = structlog.get_logger("runtime.recovery")
        try:
            stopped = await self._feature_run_recovery.stop_overrunning_runs(
                runtime_limit_seconds=self._feature_runtime_limit_seconds
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error(
                "feature_runtime_limit_sweep_failed",
                error_type=type(error).__name__,
            )
            return
        if stopped:
            logger.warning(
                "feature_runs_stopped_at_runtime_limit",
                count=len(stopped),
                features=sorted(stopped),
            )

    async def _recover_incomplete_operations(self) -> RecoverySummary:
        """Run one serialized stale-operation claim and reconciliation pass."""
        operations: list[ExternalOperation] = []
        for operation in await self._journal.list_incomplete_operations():
            if await self._journal.claim_stale_operation(operation):
                operations.append(operation)
        reconciled = 0
        cancelled = 0
        unknown = 0
        failures = 0
        deferred = 0
        for operation in operations:
            try:
                outcome = await self._recover(operation)
            except Exception as error:
                # This handler catches a defect in the recovery code itself, not a fact about
                # the provider. It is recorded as `RECOVERY_DEFECT_ERROR_CODE` so the
                # escalation that reads these rows can tell the two apart: without it, a
                # sweep that threw escalated somebody else's feature and asked them to go and
                # inspect their repository. The type is logged because nothing else keeps it.
                failures += 1
                logger = structlog.get_logger("runtime.recovery")
                logger.error(
                    "operation_recovery_raised",
                    operation_id=operation.operation_id,
                    operation_type=operation.operation_type.value,
                    error_type=type(error).__name__,
                )
                await self._journal.record_recovery_failure(
                    operation.operation_id,
                    status=ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
                    error_code=RECOVERY_DEFECT_ERROR_CODE,
                    error_message="the platform's recovery pass raised while reconciling",
                    compensation_status=CompensationStatus.MANUAL_REVIEW_REQUIRED,
                )
                continue
            if outcome == "reconciled":
                reconciled += 1
            elif outcome == "cancelled":
                cancelled += 1
            elif outcome == "deferred":
                deferred += 1
            else:
                unknown += 1
        summary = RecoverySummary(
            scanned=len(operations),
            reconciled=reconciled,
            cancelled=cancelled,
            unknown=unknown,
            failures=failures,
            deferred=deferred,
        )
        self._last_summary = summary
        self._completed = True
        return summary

    async def run_periodic_recovery(self, *, interval_seconds: float = 30.0) -> None:
        """Sweep operations that were fresh at startup after their heartbeat later expires."""
        if interval_seconds <= 0:
            msg = "recovery sweep interval must be positive"
            raise ValueError(msg)
        logger = structlog.get_logger("runtime.recovery")
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                summary = await self.recover_incomplete_operations()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # Provider responses and repository paths may be present in exception text.
                # Retain only the platform-owned category and retry on the next bounded tick.
                logger.error(
                    "external_operation_recovery_sweep_failed",
                    error_type=type(error).__name__,
                )
                continue
            if summary.scanned:
                logger.info(
                    "external_operation_recovery_sweep_completed",
                    scanned=summary.scanned,
                    reconciled=summary.reconciled,
                    cancelled=summary.cancelled,
                    unknown=summary.unknown,
                    failures=summary.failures,
                    deferred=summary.deferred,
                )

    async def _recover(self, operation: ExternalOperation) -> str:
        """Decide one interrupted operation through its declared disposition, never a fallthrough.

        Every status this writes is reached from ``_RECOVERY_POLICY``. There is no ``else``:
        the previous chain ended in one, and five remote types reached it only because nobody
        had written a branch for them. What it stamped -- UNKNOWN plus manual review -- was
        also the one thing that stops the reconciliation callback that would have settled them.
        """
        disposition = _RECOVERY_POLICY[operation.operation_type]
        match disposition:
            case RecoveryDisposition.LOCAL_RECONCILE:
                return await self._recover_from_local_evidence(operation)
            case RecoveryDisposition.WORKSPACE_LOCAL:
                return await self._abandon_workspace_local(operation)
            case RecoveryDisposition.DEFER_TO_CREDENTIALED:
                return await self._defer_to_credentialed(operation)
            case RecoveryDisposition.TERMINAL_BENIGN:
                return await self._abandon_benign(operation)
            case RecoveryDisposition.MANUAL_REVIEW:
                return await self._require_manual_review(operation, _UNCONFIRMED_EFFECT_REASON)
            case _ as unreachable:
                assert_never(unreachable)

    async def _recover_from_local_evidence(self, operation: ExternalOperation) -> str:
        """Settle a clone or a commit from the checkout, which is authoritative for both."""
        if operation.operation_type is ExternalOperationType.CLONE_REPOSITORY:
            if await self._clone_is_valid(operation):
                workspace_path = _metadata_string(operation, "workspace_path")
                await self._journal.record_reconciled_result(
                    operation.operation_id,
                    external_reference=workspace_path,
                    result_payload={"recovered": True, "recovery_method": "clone_present"},
                )
                return "reconciled"
            if operation.status is ExternalOperationStatus.CANCELLATION_REQUESTED:
                await self._cleanup_incomplete_clone(operation)
                await self._journal.record_recovery_failure(
                    operation.operation_id,
                    status=ExternalOperationStatus.CANCELLED,
                    error_code="cancelled_during_clone",
                    error_message="incomplete workflow-owned clone was stopped during recovery",
                )
                return "cancelled"
            return await self._require_manual_review(operation, _UNCONFIRMED_EFFECT_REASON)
        commit_sha = await self._commit_is_present(operation)
        if commit_sha is not None:
            await self._journal.record_reconciled_result(
                operation.operation_id,
                external_reference=commit_sha,
                result_payload={
                    "recovered": True,
                    "commit_sha": commit_sha,
                    "recovery_method": "local_commit_present",
                },
            )
            return "reconciled"
        # Deliberately unchanged: a checkout that cannot show the clone or the commit leaves
        # this sweep with no evidence at all, which is not the same as evidence of absence.
        return await self._require_manual_review(operation, _UNCONFIRMED_EFFECT_REASON)

    async def _abandon_workspace_local(self, operation: ExternalOperation) -> str:
        """End a workspace-owned effect without ever calling it externally unknown.

        Readiness used to count unknown operations, so parking a local one there took the
        whole runtime down until someone edited the database. There is nothing outside the
        workspace to reconcile.
        """
        if operation.attempt < operation.max_attempts:
            # Stable idempotency keys deliberately survive process restart. Leave a claimed
            # local operation replayable when its own bounded attempt budget has room;
            # cancelling it would make that same key permanently terminal and strand the
            # child at its persisted retry claim.
            await self._journal.record_recovery_failure(
                operation.operation_id,
                status=ExternalOperationStatus.FAILED_RETRYABLE,
                error_code="workspace_operation_interrupted",
                error_message=(
                    "workspace-local operation was interrupted before a result was recorded"
                ),
                compensation_status=CompensationStatus.NOT_REQUIRED,
            )
            return "reconciled"
        await self._journal.record_recovery_failure(
            operation.operation_id,
            status=ExternalOperationStatus.CANCELLED,
            error_code="workspace_operation_abandoned",
            error_message=(
                "workspace-local operation was interrupted after its replay budget was spent"
            ),
            compensation_status=CompensationStatus.NOT_REQUIRED,
        )
        return "cancelled"

    async def _defer_to_credentialed(self, operation: ExternalOperation) -> str:
        """Leave a remote effect exactly where a credentialed caller can still settle it.

        This sweep holds no provider credentials and never will, so it cannot tell whether
        the push landed or the pull request exists. The next request that re-enters the same
        code path can, using the reconcile callback that is already written for the type.
        Stamping UNKNOWN here took that chance away and, worse, was itself the thing that
        refused every unrelated write on the deployment.
        """
        if operation.attempt >= operation.max_attempts:
            # No credentialed request can reach it any more: a reconcile that proves the
            # effect absent has no attempt left to spend. That is genuinely a person's
            # problem, and is the only route from a remote type to UNKNOWN.
            return await self._require_manual_review(
                operation,
                "remote operation was interrupted after its replay budget was spent and its "
                "effect could not be confirmed",
            )
        await self._journal.record_recovery_failure(
            operation.operation_id,
            status=ExternalOperationStatus.AWAITING_RECONCILIATION,
            error_code="awaiting_credentialed_reconciliation",
            error_message=(
                "remote operation was interrupted; a credentialed request can still confirm it"
            ),
            compensation_status=CompensationStatus.NOT_REQUIRED,
        )
        return "deferred"

    async def _abandon_benign(self, operation: ExternalOperation) -> str:
        """End an effect that gates nothing, rather than spend a person's attention on it."""
        await self._journal.record_recovery_failure(
            operation.operation_id,
            status=ExternalOperationStatus.CANCELLED,
            error_code="benign_operation_abandoned",
            error_message=(
                "interrupted operation is not a gate on the feature and was not confirmed"
            ),
            compensation_status=CompensationStatus.NOT_REQUIRED,
        )
        return "cancelled"

    async def _require_manual_review(self, operation: ExternalOperation, reason: str) -> str:
        """Record the one conclusion that needs a person, with the reason attached."""
        await self._journal.record_recovery_failure(
            operation.operation_id,
            status=ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
            error_code="reconciliation_required",
            error_message=reason,
            compensation_status=CompensationStatus.MANUAL_REVIEW_REQUIRED,
        )
        return "unknown"

    async def _clone_is_valid(self, operation: ExternalOperation) -> bool:
        """Verify a prior clone's workspace and origin before treating it as reusable."""
        workspace_value = operation.safe_metadata.get("workspace_path")
        expected_origin = operation.safe_metadata.get("source_url")
        if not isinstance(workspace_value, str) or not isinstance(expected_origin, str):
            return False
        workspace = Path(workspace_value).resolve(strict=False)
        if not workspace.is_dir() or not (workspace / ".git").exists():
            return False
        result = await self._process_runner.run(
            ("git", "config", "--get", "remote.origin.url"),
            workspace,
            timeout_seconds=5,
            cancellation_token=MockCancellationToken(),
            environment=None,
        )
        return result.succeeded and result.stdout.strip() == expected_origin

    async def _commit_is_present(self, operation: ExternalOperation) -> str | None:
        """Identify a completed commit from its immutable pre-effect evidence.

        The SHA is unavailable if the worker dies immediately after `git commit`.
        The journal therefore records its parent and exact message before staging;
        those values safely identify the newly created non-merge commit locally.
        """
        workspace_value = operation.safe_metadata.get("workspace_path")
        commit_sha = operation.safe_metadata.get("commit_sha")
        if not isinstance(workspace_value, str):
            return None
        workspace = Path(workspace_value).resolve(strict=False)
        if not workspace.is_dir():
            return None
        if isinstance(commit_sha, str):
            result = await self._process_runner.run(
                ("git", "cat-file", "-e", f"{commit_sha}^{{commit}}"),
                workspace,
                timeout_seconds=5,
                cancellation_token=MockCancellationToken(),
                environment=None,
            )
            if result.succeeded:
                return commit_sha

        parent_sha = _metadata_string(operation, "parent_head_sha")
        message = _metadata_string(operation, "commit_message")
        if parent_sha is None or message is None:
            return None
        log = await self._process_runner.run(
            ("git", "log", "-n", "50", "--format=%H%x00%P%x00%B%x00"),
            workspace,
            timeout_seconds=5,
            cancellation_token=MockCancellationToken(),
            environment=None,
        )
        if not log.succeeded:
            return None
        fields = log.stdout.split("\x00")
        for index in range(0, len(fields) - 2, 3):
            candidate_sha, parents, candidate_message = fields[index : index + 3]
            if (
                candidate_sha
                and parents.split() == [parent_sha]
                and candidate_message.strip() == message.strip()
            ):
                return candidate_sha
        return None

    async def _cleanup_incomplete_clone(self, operation: ExternalOperation) -> None:
        """Delete only a configured incomplete workflow-owned clone directory."""
        if self._workspace_root is None:
            return
        workspace_value = operation.safe_metadata.get("workspace_path")
        if not isinstance(workspace_value, str):
            return
        workspace = Path(workspace_value).resolve(strict=False)
        if workspace == self._workspace_root or self._workspace_root not in workspace.parents:
            return
        if not workspace.exists() or (workspace / ".git").exists():
            return
        await asyncio.to_thread(_remove_empty_or_incomplete_workspace, workspace)


def _remove_empty_or_incomplete_workspace(workspace: Path) -> None:
    """Remove a known owned directory without following links or touching arbitrary paths."""
    for path in sorted(workspace.rglob("*"), reverse=True):
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            path.rmdir()
    workspace.rmdir()


def _metadata_string(operation: ExternalOperation, key: str) -> str | None:
    """Return a persisted string field without treating arbitrary JSON as an external reference."""
    value = operation.safe_metadata.get(key)
    return value if isinstance(value, str) else None


__all__ = [
    "RecoveryDisposition",
    "RecoveryService",
    "RecoverySummary",
    "UnconfirmedEffectEscalation",
    "assert_recovery_policy_is_total",
    "recovery_disposition_for",
]
