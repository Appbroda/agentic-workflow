"""Execution wrapper enforcing durable intent and idempotency around live side effects."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import structlog

from services.action_context import current_feature_action_id
from services.cancellation import CancellationRequested, CancellationToken
from state.external_operations import (
    CHILD_ATTEMPT_METADATA_KEY,
    LOGICAL_STEP_METADATA_KEY,
    REPLAY_REFUSED_STATUSES,
    CompensationStatus,
    ExternalOperation,
    ExternalOperationStatus,
    ExternalOperationType,
    child_attempt_of,
)
from storage.external_operation_store import (
    ExternalOperationError,
    ExternalOperationJournal,
    OperationInProgressError,
    OperationLeaseLostError,
    OperationReconciliationRequired,
    OperationReplayRefused,
    OperationResult,
    deterministic_operation_key,
    fingerprint_operation_input,
    successor_operation_key,
)

T = TypeVar("T")

_LOGGER = structlog.get_logger("runtime.reconciliation")


class UnknownExternalOperation(RuntimeError):
    """An adapter cannot tell whether its remote side effect completed after interruption."""

    def __init__(self, message: str, *, external_reference: str | None = None) -> None:
        super().__init__(message)
        self.external_reference = external_reference


@dataclass(frozen=True, slots=True)
class EffectAbsent:
    """Positive provider evidence that the intended side effect never took place.

    Distinct from "we could not tell". Only this outcome lets the existing retry mechanism
    run the action again, and only inside the attempt budget the operation already had.
    """

    method: str
    detail: str
    observations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EffectUnproven:
    """The provider could not be consulted, or answered something that settles nothing.

    The operation stays where it is. A later credentialed request gets another chance, and
    the recorded reason tells the next reader what was actually tried.
    """

    method: str
    detail: str
    observations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EffectAmbiguous:
    """The provider answered, and its answer cannot identify the intended effect.

    More than one candidate matched and none can be chosen. Guessing here is how a second
    pull request gets opened, so this is the one reconciliation outcome that asks a person.
    """

    method: str
    detail: str
    observations: dict[str, Any] = field(default_factory=dict)


# What a ``reconcile`` callback may conclude. ``None`` is retained as the unadorned form of
# ``EffectUnproven`` so callbacks that have nothing to record stay as short as they were.
ReconciliationOutcome = OperationResult | EffectAbsent | EffectUnproven | EffectAmbiguous | None

# The statuses a credentialed request may resolve through a reconcile callback rather than by
# replaying the side effect.
_RECONCILABLE_STATUSES = frozenset(
    {
        ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
        ExternalOperationStatus.AWAITING_RECONCILIATION,
    }
)


def _replay_refusal(operation: ExternalOperation) -> str:
    """Word the refusal the way each ended status has always been worded.

    Two sentences rather than one: they are what a workstream's blocking issues carry, and an
    operator reading "cancelled" wants to know it was cancelled rather than that something
    failed.
    """
    if operation.status is ExternalOperationStatus.CANCELLED:
        return f"cancelled operation cannot be replayed: {operation.operation_id}"
    return f"operation has a terminal failure and cannot be replayed: {operation.operation_id}"


@dataclass(frozen=True, slots=True)
class JournaledOperationResult:
    """The adapter result together with operation evidence for workflow checkpointing."""

    value: Any
    operation: ExternalOperation
    reused: bool


@dataclass(frozen=True, slots=True)
class ExternalOperationScope:
    """Stable non-secret identifiers included in every operation idempotency key.

    ``child_attempt`` is the exception: it is stamped onto every operation's metadata and is
    deliberately *not* part of the key. The key carries input hashes, and adding the attempt
    to it would give the same work a new identity on every retry -- which is what the reuse
    machinery exists to prevent. The stamp is a label on the row, not part of its identity.
    """

    workflow_id: str
    feature_id: str | None = None
    child_workflow_id: str | None = None
    repository_id: str | None = None
    # Which of the child's attempts this executor was built for, where it was built inside one.
    # `None` for the feature-level scopes -- reconnaissance and publication belong to no
    # attempt, and stamping them with a number would invent one.
    child_attempt: int | None = None


class ExternalOperationExecutor:
    """Run a callable only after journal intent and cancellation checks are durably committed."""

    def __init__(
        self,
        *,
        journal: ExternalOperationJournal,
        cancellation_token: CancellationToken,
        scope: ExternalOperationScope,
        heartbeat_seconds: float = 10.0,
    ) -> None:
        if heartbeat_seconds <= 0:
            msg = "heartbeat_seconds must be positive"
            raise ValueError(msg)
        self._journal = journal
        self._cancellation_token = cancellation_token
        self._scope = scope
        self._heartbeat_seconds = heartbeat_seconds

    @property
    def scope(self) -> ExternalOperationScope:
        """Expose the stable ownership identifiers for adapter construction."""
        return self._scope

    @property
    def cancellation_token(self) -> CancellationToken:
        """Expose the shared cooperative stop signal to long-running adapters."""
        return self._cancellation_token

    async def run(
        self,
        *,
        operation_type: ExternalOperationType,
        logical_step: str,
        safe_input: dict[str, Any],
        action: Callable[[], Awaitable[tuple[T, OperationResult]]],
        idempotency_input: dict[str, Any] | None = None,
        reconcile: Callable[[ExternalOperation], Awaitable[ReconciliationOutcome]] | None = None,
        max_attempts: int = 1,
        fault_admits_retry: Callable[[BaseException], bool] | None = None,
        attempt_metadata: dict[str, Any] | None = None,
    ) -> JournaledOperationResult:
        """Execute idempotently and never repeat active, stale, or ambiguous operations blindly."""
        fingerprint = fingerprint_operation_input(idempotency_input or safe_input)
        key = deterministic_operation_key(
            workflow_id=self._scope.workflow_id,
            feature_id=self._scope.feature_id,
            child_workflow_id=self._scope.child_workflow_id,
            repository_id=self._scope.repository_id,
            operation_type=operation_type,
            logical_step=logical_step,
            input_fingerprint=fingerprint,
        )
        open_operation = functools.partial(
            self._journal.create_operation,
            workflow_id=self._scope.workflow_id,
            feature_id=self._scope.feature_id,
            child_workflow_id=self._scope.child_workflow_id,
            repository_id=self._scope.repository_id,
            operation_type=operation_type,
            input_fingerprint=fingerprint,
            max_attempts=max_attempts,
            # The attempt stamp goes ahead of `safe_input` so a caller that already records
            # its own `child_attempt` keeps the value it recorded. The coding call is that
            # caller: it reads the attempt out of the task plan it was handed, which is the
            # value its completed-receipt lookup matches on. Both are `child.retry_count` by
            # construction; where they could ever disagree, the caller's is the one the
            # recovery path already reads, and one row must not carry two answers.
            safe_metadata={
                LOGICAL_STEP_METADATA_KEY: logical_step,
                **(
                    {CHILD_ATTEMPT_METADATA_KEY: self._scope.child_attempt}
                    if self._scope.child_attempt is not None
                    else {}
                ),
                **safe_input,
            },
            repository_revision=(
                safe_input.get("repository_revision")
                if isinstance(safe_input.get("repository_revision"), str)
                else None
            ),
            command_fingerprint=(
                safe_input.get("command_fingerprint")
                if isinstance(safe_input.get("command_fingerprint"), str)
                else None
            ),
        )
        operation = await open_operation(idempotency_key=key)
        resolved = await self._resolve_reusable(operation, reconcile)
        if isinstance(resolved, JournaledOperationResult):
            return resolved
        operation = resolved
        if operation.status in REPLAY_REFUSED_STATUSES:
            # A row that ended without a result. Whether that ends this caller too depends on
            # who is asking: the attempt that failed it is repeating itself and is refused,
            # and a later attempt has not tried this work at all. See `_successor_for`.
            successor = await self._successor_for(operation, key=key, open_operation=open_operation)
            if successor is None:
                raise OperationReplayRefused(_replay_refusal(operation))
            resolved = await self._resolve_reusable(successor, reconcile)
            if isinstance(resolved, JournaledOperationResult):
                return resolved
            operation = resolved
            if operation.status in REPLAY_REFUSED_STATUSES:
                # The successor for this attempt has itself already ended. Bounded here on
                # purpose: chasing a further successor would be this attempt replaying its
                # own terminal row, which is the one thing the status exists to stop.
                raise OperationReplayRefused(_replay_refusal(operation))
        await self._cancellation_token.raise_if_cancelled()
        operation = await self._journal.claim_operation(
            operation.operation_id,
            safe_metadata=attempt_metadata or {},
        )
        try:
            await self._cancellation_token.raise_if_cancelled()
            value, result = await self._run_with_heartbeat(operation.operation_id, action)
            completed = await self._journal.record_result(
                operation.operation_id,
                external_reference=result.external_reference,
                result_payload=result.payload,
            )
            return JournaledOperationResult(value=value, operation=completed, reused=False)
        except CancellationRequested:
            await self._journal.record_failure(
                operation.operation_id,
                status=ExternalOperationStatus.CANCELLED,
                error_code="cancelled",
                error_message="operation stopped before a confirmed external result",
            )
            raise
        except UnknownExternalOperation as error:
            await self._journal.record_failure(
                operation.operation_id,
                status=ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
                error_code="unknown_external_state",
                error_message="external operation outcome could not be confirmed",
                external_reference=error.external_reference,
                compensation_status=CompensationStatus.MANUAL_REVIEW_REQUIRED,
            )
            raise
        except OperationLeaseLostError as error:
            # Read back by `is_transient_provider_fault` (workflows/feature_workflow.py), so a
            # lease lost on a workspace-local operation can be told apart from one lost on an
            # operation with a real external effect, without this handler itself deciding that
            # -- the recovery sweep's own policy table already does, and this is the one
            # thing missing to let it be consulted from here too.
            error.operation_type = operation_type
            current = await self._journal.get(operation.operation_id)
            if current.status is ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE:
                raise
            try:
                await self._journal.record_failure(
                    operation.operation_id,
                    status=ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
                    error_code="operation_lease_lost",
                    error_message="external operation ownership could not be confirmed",
                    compensation_status=CompensationStatus.MANUAL_REVIEW_REQUIRED,
                )
            except Exception:
                msg = (
                    "external operation lease was lost and its unknown state could not be recorded"
                )
                raise OperationLeaseLostError(msg) from error
            raise
        except Exception as error:
            current = await self._journal.get(operation.operation_id)
            # A refused or deterministic answer returns identically on every retry, so a row
            # that calls it retryable tells the next reader a replay is worth making that the
            # caller's own driver has already refused -- two authorities, one question. The
            # caller's admission test decides; without one, the budget alone decides, as it
            # always has.
            admitted = fault_admits_retry is None or fault_admits_retry(error)
            failure_status = (
                ExternalOperationStatus.FAILED_RETRYABLE
                if admitted and current.attempt < current.max_attempts
                else ExternalOperationStatus.FAILED_TERMINAL
            )
            # The type name and the adapter's classification token are platform- or
            # library-owned symbols, never provider message text. Without them,
            # AB-Feature-174's three identical coding-call failures were three empty rows,
            # and the fault class had to be reconstructed by reading source code.
            fault_name = type(error).__name__
            classification = getattr(error, "failure_classification", None)
            if isinstance(classification, str) and classification:
                fault_name = f"{fault_name}/{classification}"
            # A caller that can tell its own failures apart may say so, and the row then says
            # which one happened rather than only that one did. Without this every failure of
            # an operation type shares a single `..._failed` code, and telling a dependency
            # install that ran out of budget from one whose lockfile does not resolve -- two
            # unrelated actions for whoever is paged -- costs a database session. The default
            # is what it has always been, so an operation that does not distinguish its
            # failures is unaffected.
            outcome = getattr(error, "operation_outcome", None)
            outcome_token = outcome if isinstance(outcome, str) and outcome else "failed"
            await self._journal.record_failure(
                operation.operation_id,
                status=failure_status,
                error_code=f"{operation_type.value}_{outcome_token}",
                error_message=(
                    f"external operation failed before a confirmed result ({fault_name})"
                ),
            )
            raise

    async def _run_with_heartbeat(
        self,
        operation_id: str,
        action: Callable[[], Awaitable[tuple[T, OperationResult]]],
    ) -> tuple[T, OperationResult]:
        """Abort the protected action if durable heartbeat renewal stops succeeding."""

        async def invoke_action() -> tuple[T, OperationResult]:
            return await action()

        action_task: asyncio.Task[tuple[T, OperationResult]] = asyncio.create_task(invoke_action())
        heartbeat_task = asyncio.create_task(self._heartbeat(operation_id))
        try:
            done, _pending = await asyncio.wait(
                {action_task, heartbeat_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if heartbeat_task in done:
                heartbeat_error = heartbeat_task.exception()
                action_task.cancel()
                await asyncio.gather(action_task, return_exceptions=True)
                msg = "external operation heartbeat could not be renewed during execution"
                if heartbeat_error is None:
                    raise OperationLeaseLostError(msg)
                raise OperationLeaseLostError(msg) from heartbeat_error
            return await action_task
        finally:
            action_task.cancel()
            heartbeat_task.cancel()
            await asyncio.gather(action_task, heartbeat_task, return_exceptions=True)

    async def _heartbeat(self, operation_id: str) -> None:
        """Persist liveness while an adapter waits on a subprocess or remote provider."""
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            await self._journal.record_heartbeat(operation_id)

    async def _resolve_reusable(
        self,
        operation: ExternalOperation,
        reconcile: Callable[[ExternalOperation], Awaitable[ReconciliationOutcome]] | None,
    ) -> JournaledOperationResult | ExternalOperation:
        """Reuse what this row already proves, or hand back the row to run under.

        Lifted out of `run` unchanged so it can be applied twice: once to the row found under
        the base key, and once to a successor row a later attempt opened. Returning a result
        means the work is already done and must not be repeated; returning an operation means
        this caller carries on with it.
        """
        action_id = current_feature_action_id()
        if action_id is not None:
            # Committed before the side effect. Recovery must never infer ownership from a
            # feature-wide time window because two actions can execute concurrently.
            await self._journal.link_operation_to_action(operation.operation_id, action_id)
        if operation.status is ExternalOperationStatus.SUCCEEDED:
            return JournaledOperationResult(
                value=operation.result_payload or {}, operation=operation, reused=True
            )
        if operation.status in _RECONCILABLE_STATUSES:
            reconciled = await self._reconcile_or_raise(operation, reconcile)
            if reconciled is not None:
                return reconciled
            # The provider proved the effect never landed. The operation was put back on its
            # existing retry budget, so continue into the ordinary claim below rather than
            # inventing a second retry mechanism for the recovery path.
            return await self._journal.get(operation.operation_id)
        if operation.status in {
            ExternalOperationStatus.STARTING,
            ExternalOperationStatus.RUNNING,
            ExternalOperationStatus.CANCELLATION_REQUESTED,
        }:
            if not await self._journal.claim_stale_operation(operation):
                raise OperationInProgressError(
                    f"operation is already active: {operation.operation_id}"
                )
            reconciled = await self._reconcile_or_raise(
                await self._journal.get(operation.operation_id), reconcile
            )
            if reconciled is not None:
                return reconciled
            return await self._journal.get(operation.operation_id)
        return operation

    async def _successor_for(
        self,
        ended: ExternalOperation,
        *,
        key: str,
        open_operation: Callable[..., Awaitable[ExternalOperation]],
    ) -> ExternalOperation | None:
        """Open this attempt's own row for work an earlier attempt ended, or refuse.

        A new attempt inherits confirmed effects, never terminal ones. The succeeded case
        never reaches here -- it is returned as reuse before any of this -- so nothing about
        the 23- rule is at stake: a preserved-workspace retry still finds its clone and its
        unchanged install under the base key and still does not repeat them.

        The decision is read from the ended row's own `child_attempt` stamp rather than from
        anything about the caller's history. Strictly newer starts fresh work; the same
        attempt is refused, which is exactly what `FAILED_TERMINAL` is for. `None` on either
        side is also a refusal: an unstamped row belongs to no attempt, and a scope with no
        attempt (reconnaissance, the parent's publication calls) is not one either, so
        neither can be called newer than the other without inventing the number.
        """
        caller_attempt = self._scope.child_attempt
        ended_attempt = child_attempt_of(ended)
        if caller_attempt is None or ended_attempt is None or caller_attempt <= ended_attempt:
            return None
        successor = await open_operation(
            idempotency_key=successor_operation_key(key, child_attempt=caller_attempt)
        )
        await self._journal.mark_superseded_by(
            ended.operation_id, successor_operation_id=successor.operation_id
        )
        _LOGGER.info(
            "external_operation_retried_under_new_attempt",
            operation_id=successor.operation_id,
            superseded_operation_id=ended.operation_id,
            operation_type=str(ended.operation_type),
            repository_id=self._scope.repository_id,
            child_attempt=caller_attempt,
            ended_attempt=ended_attempt,
            ended_status=str(ended.status),
        )
        return successor

    async def _reconcile_or_raise(
        self,
        operation: ExternalOperation,
        reconcile: Callable[[ExternalOperation], Awaitable[ReconciliationOutcome]] | None,
    ) -> JournaledOperationResult | None:
        """Use fresh request-scoped access to prove an uncertain side effect once.

        The callback is deliberately read-only and must not conclude more than the provider
        actually told it.  This lets a new request reconcile a prior push/PR without ever
        persisting its credentials for startup recovery.

        Returns the reconciled result when the effect is proven to have happened, and
        ``None`` when the provider proved it did not -- the caller then continues into the
        ordinary claim path under the operation's existing attempt budget.  Every other
        conclusion raises, because the alternative is repeating a side effect blindly.
        """
        if reconcile is None:
            raise OperationReconciliationRequired(
                f"operation requires reconciliation: {operation.operation_id}"
            )
        outcome = await reconcile(operation)
        if isinstance(outcome, OperationResult):
            completed = await self._journal.record_reconciled_result(
                operation.operation_id,
                external_reference=outcome.external_reference,
                result_payload=outcome.payload,
            )
            self._log_reconciliation(
                operation,
                method=str((outcome.payload or {}).get("recovery_method", "adapter_evidence")),
                result="reconciled",
            )
            return JournaledOperationResult(
                value=completed.result_payload or {}, operation=completed, reused=True
            )
        if isinstance(outcome, EffectAbsent):
            await self._reopen_absent_effect(operation, outcome)
            return None
        if isinstance(outcome, EffectAmbiguous):
            await self._journal.record_recovery_failure(
                operation.operation_id,
                status=ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
                error_code=f"{operation.operation_type.value}_reconciliation_ambiguous",
                error_message=outcome.detail,
                compensation_status=CompensationStatus.MANUAL_REVIEW_REQUIRED,
            )
            await self._journal.record_reconciliation_note(
                operation.operation_id,
                method=outcome.method,
                result="ambiguous",
                safe_metadata=outcome.observations,
            )
            self._log_reconciliation(operation, method=outcome.method, result="ambiguous")
            raise OperationReconciliationRequired(
                f"operation requires manual reconciliation: {operation.operation_id}"
            )
        if isinstance(outcome, EffectUnproven):
            await self._journal.record_reconciliation_note(
                operation.operation_id,
                method=outcome.method,
                result="unproven",
                safe_metadata=outcome.observations,
            )
            self._log_reconciliation(operation, method=outcome.method, result="unproven")
        else:
            self._log_reconciliation(operation, method="none", result="unproven")
        raise OperationReconciliationRequired(
            f"operation requires manual reconciliation: {operation.operation_id}"
        )

    async def _reopen_absent_effect(
        self, operation: ExternalOperation, outcome: EffectAbsent
    ) -> None:
        """Put a provably-unperformed effect back on its own budget, or end it honestly."""
        await self._journal.record_reconciliation_note(
            operation.operation_id,
            method=outcome.method,
            result="effect_absent",
            safe_metadata=outcome.observations,
        )
        current = await self._journal.get(operation.operation_id)
        exhausted = current.attempt >= current.max_attempts
        await self._journal.record_recovery_failure(
            operation.operation_id,
            status=(
                ExternalOperationStatus.FAILED_TERMINAL
                if exhausted
                else ExternalOperationStatus.FAILED_RETRYABLE
            ),
            error_code=f"{operation.operation_type.value}_effect_absent",
            error_message=outcome.detail,
            compensation_status=CompensationStatus.NOT_REQUIRED,
        )
        self._log_reconciliation(
            operation,
            method=outcome.method,
            result="effect_absent_exhausted" if exhausted else "effect_absent",
        )
        if exhausted:
            msg = (
                "external operation never reached the provider and its attempt budget is "
                f"spent: {operation.operation_id}"
            )
            raise ExternalOperationError(msg)

    def _log_reconciliation(
        self, operation: ExternalOperation, *, method: str, result: str
    ) -> None:
        """Emit the identifiers an operator needs, and never a provider message or token."""
        _LOGGER.info(
            "external_operation_reconciliation_attempted",
            operation_id=operation.operation_id,
            operation_type=operation.operation_type.value,
            feature_id=operation.feature_id,
            repository_id=operation.repository_id,
            reconciliation_method=method,
            reconciliation_outcome=result,
        )


__all__ = [
    "EffectAbsent",
    "EffectAmbiguous",
    "EffectUnproven",
    "ExternalOperationExecutor",
    "ExternalOperationScope",
    "JournaledOperationResult",
    "ReconciliationOutcome",
    "UnknownExternalOperation",
]
