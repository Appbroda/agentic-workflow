"""Durable journal repository for every live external operation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError

from state.enums import TERMINAL_FEATURE_STATUSES
from state.external_operations import (
    REPLAY_REFUSED_STATUSES,
    CompensationStatus,
    ExternalOperation,
    ExternalOperationAttempt,
    ExternalOperationEvent,
    ExternalOperationStatus,
    ExternalOperationType,
)
from state.failure_diagnosis import FeatureFailureClassification
from storage.db import Database
from storage.models import (
    ExternalOperationAttemptModel,
    ExternalOperationEventModel,
    ExternalOperationModel,
    FeatureActionExternalOperationModel,
    FeatureWorkflowModel,
)


class ExternalOperationError(RuntimeError):
    """Base error for a journal action that cannot safely be repeated."""


class OperationInProgressError(ExternalOperationError):
    """An operation is still active and another worker must not duplicate it."""


class OperationReconciliationRequired(ExternalOperationError):
    """An operation reached an ambiguous state and needs explicit reconciliation first."""


class OperationLeaseLostError(ExternalOperationError):
    """An operation could not renew its durable liveness marker while executing.

    ``operation_type`` is set after construction, once the catching handler
    (``services/external_operations.py``) knows which operation this happened to -- read back
    by ``is_transient_provider_fault`` (``workflows/feature_workflow.py``) to tell a lease lost
    on a workspace-local operation (nothing outside the workspace to reconcile, safe to retry)
    from one lost on an operation with a real external effect (not safe to retry blind). `None`
    until then, and for any caller that never sets it, which keeps every existing
    ``OperationLeaseLostError(msg)`` raise unchanged.
    """

    operation_type: ExternalOperationType | None = None


class OperationTransitionConflictError(ExternalOperationError):
    """A requested lifecycle transition lost ownership or is not legally monotonic."""


class OperationReplayRefused(ExternalOperationError):
    """A caller asked to repeat an operation whose row already ended, inside the same attempt.

    Deliberately not one of the unconfirmed-effect errors above. Those mean "something may
    have happened outside this process and nobody can say how much"; this one means the
    opposite -- nothing was attempted, and nothing will be, because the platform's own record
    already says this attempt is finished with this work. It returns identically on every
    retry, for ever, so it is a diagnosis rather than weather: `is_transient_provider_fault`
    answers False for it, and the workstream stops on the first occurrence with this sentence
    in its blocking issues instead of four backoffs later with "the provider did not answer".
    """

    # Reaching the workflow at all, after 69-, means one attempt asked twice for work it had
    # already ended -- so it is this platform's to fix, and not a finding about the target
    # repository. Declared rather than left to the child loop's fallback, which would file it
    # as "the platform did not anticipate OperationReplayRefused": it was anticipated, and
    # saying otherwise sends whoever reads the record looking for the wrong thing.
    failure_classification = FeatureFailureClassification.PLATFORM_DEFECT.value

    def __init__(self, message: str) -> None:
        """Carry the refusal as explicit diagnostics so the sentence reaches the record.

        `safe_error_diagnostics` records nothing from an arbitrary exception's text, which is
        right -- a message can carry credential-bearing detail. This one cannot: it is built
        here from a status and an operation id, both platform-owned. Without the opt-in the
        blocking issues name only the exception type, which is the failure mode that left
        four separate runs diagnosable by timing forensics alone.
        """
        super().__init__(message)
        self.diagnostics = (message,)


@dataclass(frozen=True, slots=True)
class OperationResult:
    """A non-secret result supplied by a side-effect adapter to the durable journal."""

    external_reference: str | None = None
    payload: dict[str, Any] | None = None


class ExternalOperationJournal:
    """Persist operation intent, attempts, transitions, and results in separate transactions."""

    def __init__(self, database: Database, *, stale_after_seconds: float = 300.0) -> None:
        """Keep stale detection bounded so startup cannot accidentally replay live work."""
        if stale_after_seconds <= 0:
            msg = "stale_after_seconds must be positive"
            raise ValueError(msg)
        self._database = database
        self._stale_after = timedelta(seconds=stale_after_seconds)

    async def create_operation(
        self,
        *,
        workflow_id: str,
        feature_id: str | None,
        child_workflow_id: str | None,
        repository_id: str | None,
        operation_type: ExternalOperationType,
        idempotency_key: str,
        input_fingerprint: str,
        repository_revision: str | None = None,
        command_fingerprint: str | None = None,
        max_attempts: int = 1,
        safe_metadata: dict[str, Any] | None = None,
    ) -> ExternalOperation:
        """Create and commit intent before an adapter is allowed to perform a side effect."""
        if max_attempts < 1:
            msg = "max_attempts must be at least one"
            raise ValueError(msg)
        metadata = _safe_metadata(safe_metadata or {})
        async with self._database.session() as session:
            existing = await session.scalar(
                select(ExternalOperationModel).where(
                    ExternalOperationModel.idempotency_key == idempotency_key
                )
            )
            if existing is not None:
                if existing.input_fingerprint != input_fingerprint:
                    msg = "external-operation idempotency key was reused with different input"
                    raise ExternalOperationError(msg) from None
                return _operation_from_model(existing)
            now = datetime.now(UTC)
            model = ExternalOperationModel(
                operation_id=f"operation-{uuid4()}",
                workflow_id=workflow_id,
                feature_id=feature_id,
                child_workflow_id=child_workflow_id,
                repository_id=repository_id,
                operation_type=operation_type,
                idempotency_key=idempotency_key,
                status=ExternalOperationStatus.PENDING,
                attempt=0,
                max_attempts=max_attempts,
                input_fingerprint=input_fingerprint,
                repository_revision=repository_revision,
                command_fingerprint=command_fingerprint,
                safe_metadata=metadata,
                created_at=now,
                updated_at=now,
            )
            if repository_revision is not None and operation_type in _VALIDATION_OPERATION_TYPES:
                previous = list(
                    await session.scalars(
                        select(ExternalOperationModel).where(
                            ExternalOperationModel.workflow_id == workflow_id,
                            ExternalOperationModel.repository_id == repository_id,
                            ExternalOperationModel.operation_type.in_(_VALIDATION_OPERATION_TYPES),
                            ExternalOperationModel.is_current.is_(True),
                            ExternalOperationModel.repository_revision.is_not(None),
                            ExternalOperationModel.repository_revision != repository_revision,
                        )
                    )
                )
                for prior in previous:
                    prior.is_current = False
                    prior.superseded_by_operation_id = model.operation_id
            session.add(model)
            self._add_event(
                session,
                model,
                previous_status=None,
                new_status=ExternalOperationStatus.PENDING,
                event_type="operation_created",
                timestamp=now,
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                existing = await session.scalar(
                    select(ExternalOperationModel).where(
                        ExternalOperationModel.idempotency_key == idempotency_key
                    )
                )
                if existing is None:
                    raise
                if existing.input_fingerprint != input_fingerprint:
                    msg = "external-operation idempotency key was reused with different input"
                    raise ExternalOperationError(msg) from None
                return _operation_from_model(existing)
            return _operation_from_model(model)

    async def mark_superseded_by(self, operation_id: str, *, successor_operation_id: str) -> None:
        """Record that a later child attempt started fresh work in place of an ended row.

        Only ever called for a row that ended `FAILED_TERMINAL` or `CANCELLED`. The ended row
        is left exactly as it is otherwise -- it is the durable history of the attempt that
        failed, and the drawer renders it per attempt (65-). What changes is that it stops
        being the current row for its key family, so a successor that goes on to succeed is
        the one the current-evidence reads return.

        Where several attempts each start a successor, the pointer names the newest: it is
        read to answer "what happened after this", and the newest is the answer. Every
        intermediate row keeps its own record and stays reachable by operation id.
        """
        async with self._database.session() as session:
            model = await _require_operation(session, operation_id)
            if model.status not in REPLAY_REFUSED_STATUSES:
                msg = (
                    "only an operation that ended terminally can be superseded: "
                    f"{operation_id} is {model.status}"
                )
                raise ExternalOperationError(msg)
            if model.superseded_by_operation_id == successor_operation_id:
                return
            model.is_current = False
            model.superseded_by_operation_id = successor_operation_id
            model.updated_at = datetime.now(UTC)
            self._add_event(
                session,
                model,
                previous_status=model.status,
                new_status=model.status,
                event_type="operation_superseded",
                timestamp=model.updated_at,
                safe_metadata={"successor_operation_id": successor_operation_id},
            )
            await session.commit()

    async def get_by_idempotency_key(self, idempotency_key: str) -> ExternalOperation | None:
        """Read one operation by its deterministic deduplication key."""
        async with self._database.session() as session:
            model = await session.scalar(
                select(ExternalOperationModel).where(
                    ExternalOperationModel.idempotency_key == idempotency_key
                )
            )
            return _operation_from_model(model) if model is not None else None

    async def get(self, operation_id: str) -> ExternalOperation:
        """Return one durable journal record or a safe missing-operation error."""
        async with self._database.session() as session:
            model = await session.get(ExternalOperationModel, operation_id)
            if model is None:
                msg = f"external operation not found: {operation_id}"
                raise ExternalOperationError(msg)
            return _operation_from_model(model)

    async def link_operation_to_action(self, operation_id: str, action_id: str) -> None:
        """Persist exact action evidence before the linked side effect can execute.

        The link is many-to-many because a later idempotent action may safely reuse an
        existing journal result. A feature-wide timestamp query cannot distinguish that
        from an unrelated concurrent action.
        """
        async with self._database.session() as session:
            existing = await session.get(
                FeatureActionExternalOperationModel,
                {"action_id": action_id, "operation_id": operation_id},
            )
            if existing is not None:
                return
            session.add(
                FeatureActionExternalOperationModel(
                    action_id=action_id,
                    operation_id=operation_id,
                    linked_at=datetime.now(UTC),
                )
            )
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                # Another executor may have linked the same replay concurrently. The
                # composite primary key makes that race harmless.
                existing = await session.get(
                    FeatureActionExternalOperationModel,
                    {"action_id": action_id, "operation_id": operation_id},
                )
                if existing is None:
                    raise

    async def transition_status(
        self,
        operation_id: str,
        status: ExternalOperationStatus,
        *,
        event_type: str,
        safe_metadata: dict[str, Any] | None = None,
        external_reference: str | None = None,
    ) -> ExternalOperation:
        """CAS one legal transition without reopening terminal or newer attempts."""
        metadata = _safe_metadata(safe_metadata or {})
        async with self._database.session() as session:
            model = await _require_operation(session, operation_id)
            prior = model.status
            allowed_from = _ALLOWED_MANUAL_TRANSITIONS.get(status, frozenset())
            if prior not in allowed_from:
                msg = f"external operation cannot transition from {prior.value} to {status.value}"
                raise OperationTransitionConflictError(msg)
            prior_attempt = model.attempt
            now = datetime.now(UTC)
            transitioned = await session.execute(
                update(ExternalOperationModel)
                .where(
                    ExternalOperationModel.operation_id == operation_id,
                    ExternalOperationModel.status == prior,
                    ExternalOperationModel.attempt == prior_attempt,
                )
                .values(
                    status=status,
                    heartbeat_at=now,
                    updated_at=now,
                    started_at=(
                        model.started_at or now
                        if status
                        in {
                            ExternalOperationStatus.STARTING,
                            ExternalOperationStatus.RUNNING,
                        }
                        else model.started_at
                    ),
                    completed_at=now if status in _TERMINAL_STATUSES else None,
                    external_reference=external_reference or model.external_reference,
                )
                .execution_options(synchronize_session=False)
            )
            if cast(CursorResult[Any], transitioned).rowcount != 1:
                msg = "external operation transition lost a concurrent status or attempt race"
                raise OperationTransitionConflictError(msg)
            await session.refresh(model)
            self._add_event(
                session,
                model,
                previous_status=prior,
                new_status=status,
                event_type=event_type,
                safe_metadata=metadata,
                timestamp=now,
            )
            await session.commit()
            return _operation_from_model(model)

    async def claim_operation(
        self,
        operation_id: str,
        *,
        safe_metadata: dict[str, Any] | None = None,
    ) -> ExternalOperation:
        """Atomically claim an operation together with its durable running attempt."""
        return await self.claim_attempt(operation_id, safe_metadata=safe_metadata)

    async def claim_attempt(
        self,
        operation_id: str,
        *,
        safe_metadata: dict[str, Any] | None = None,
    ) -> ExternalOperation:
        """Atomically claim an intent and persist its running attempt.

        No committed ``STARTING`` row is exposed between ownership acquisition and the
        attempt record. Cancellation therefore observes either an untouched intent or a
        complete RUNNING attempt that the executor can finish as CANCELLED.
        """
        metadata = _safe_metadata(safe_metadata or {})
        claimable = {
            ExternalOperationStatus.PENDING,
            ExternalOperationStatus.FAILED_RETRYABLE,
        }
        async with self._database.session() as session:
            operation = await _require_operation(session, operation_id)
            prior = operation.status
            prior_attempt = operation.attempt
            if prior not in claimable:
                if prior in {
                    ExternalOperationStatus.STARTING,
                    ExternalOperationStatus.RUNNING,
                    ExternalOperationStatus.CANCELLATION_REQUESTED,
                }:
                    raise OperationInProgressError(
                        f"operation is already active: {operation.operation_id}"
                    )
                msg = f"operation cannot be claimed from terminal status {prior.value}"
                raise ExternalOperationError(msg)
            if prior_attempt >= operation.max_attempts:
                msg = "external operation exhausted its configured retry budget"
                raise ExternalOperationError(msg)

            now = datetime.now(UTC)
            attempt_number = prior_attempt + 1
            claimed = await session.execute(
                update(ExternalOperationModel)
                .where(
                    ExternalOperationModel.operation_id == operation_id,
                    ExternalOperationModel.status == prior,
                    ExternalOperationModel.attempt == prior_attempt,
                )
                .values(
                    status=ExternalOperationStatus.RUNNING,
                    attempt=attempt_number,
                    heartbeat_at=now,
                    started_at=operation.started_at or now,
                    completed_at=None,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            if cast(CursorResult[Any], claimed).rowcount != 1:
                raise OperationInProgressError(
                    f"operation was claimed concurrently: {operation.operation_id}"
                )
            await session.refresh(operation)
            attempt = ExternalOperationAttemptModel(
                attempt_id=f"operation-attempt-{uuid4()}",
                operation_id=operation.operation_id,
                attempt=attempt_number,
                status=ExternalOperationStatus.RUNNING,
                started_at=now,
                heartbeat_at=now,
                safe_metadata=metadata,
            )
            session.add(attempt)
            self._add_event(
                session,
                operation,
                previous_status=prior,
                new_status=ExternalOperationStatus.STARTING,
                event_type="operation_starting",
                timestamp=now,
            )
            self._add_event(
                session,
                operation,
                previous_status=ExternalOperationStatus.STARTING,
                new_status=ExternalOperationStatus.RUNNING,
                event_type="attempt_started",
                safe_metadata={"attempt_id": attempt.attempt_id},
                timestamp=now,
            )
            await session.commit()
            return _operation_from_model(operation)

    async def claim_stale_operation(self, operation: ExternalOperation) -> bool:
        """Atomically fence one stale active operation for conservative recovery.

        The status and liveness predicate are evaluated by the same database update.
        A heartbeat committed after the caller listed the row therefore prevents this
        claim; if this update wins, later live-worker writes reject the UNKNOWN status.
        """
        active = {
            ExternalOperationStatus.STARTING,
            ExternalOperationStatus.RUNNING,
            ExternalOperationStatus.CANCELLATION_REQUESTED,
        }
        heartbeat = operation.heartbeat_at or operation.started_at
        now = datetime.now(UTC)
        if (
            operation.status not in active
            or heartbeat is None
            or now - heartbeat <= self._stale_after
        ):
            return False
        cutoff = now - self._stale_after
        async with self._database.session() as session:
            claimed = await session.execute(
                update(ExternalOperationModel)
                .where(
                    ExternalOperationModel.operation_id == operation.operation_id,
                    ExternalOperationModel.status == operation.status,
                    ExternalOperationModel.attempt == operation.attempt,
                    _stale_liveness_clause(cutoff),
                )
                .values(
                    status=ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
                    error_code="stale_recovery_claimed",
                    error_message=("recovery fenced a stale operation before reconciliation"),
                    compensation_status=CompensationStatus.MANUAL_REVIEW_REQUIRED,
                    heartbeat_at=now,
                    completed_at=now,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            if cast(CursorResult[Any], claimed).rowcount != 1:
                return False
            model = await _require_operation(session, operation.operation_id)
            await session.refresh(model)
            await self._finish_latest_attempt(
                session, model, ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE, now
            )
            self._add_event(
                session,
                model,
                previous_status=operation.status,
                new_status=ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
                event_type="stale_recovery_claimed",
                timestamp=now,
            )
            await session.commit()
            return True

    async def record_attempt(
        self,
        operation_id: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        workspace_path: str | None = None,
        task_plan_artifact_id: str | None = None,
        contract_artifact_id: str | None = None,
        process_id: int | None = None,
        external_run_id: str | None = None,
        safe_metadata: dict[str, Any] | None = None,
    ) -> ExternalOperationAttempt:
        """Persist an execution attempt before starting the underlying adapter action."""
        metadata = _safe_metadata(safe_metadata or {})
        async with self._database.session() as session:
            operation = await _require_operation(session, operation_id)
            if operation.status is not ExternalOperationStatus.STARTING:
                msg = (
                    "external operation attempt requires starting status, got "
                    f"{operation.status.value}"
                )
                raise ExternalOperationError(msg)
            if operation.attempt >= operation.max_attempts:
                msg = "external operation exhausted its configured retry budget"
                raise ExternalOperationError(msg)
            operation.attempt += 1
            now = datetime.now(UTC)
            operation.status = ExternalOperationStatus.RUNNING
            operation.started_at = operation.started_at or now
            operation.heartbeat_at = now
            operation.updated_at = now
            attempt = ExternalOperationAttemptModel(
                attempt_id=f"operation-attempt-{uuid4()}",
                operation_id=operation.operation_id,
                attempt=operation.attempt,
                provider=provider,
                model=model,
                workspace_path=workspace_path,
                task_plan_artifact_id=task_plan_artifact_id,
                contract_artifact_id=contract_artifact_id,
                process_id=process_id,
                external_run_id=external_run_id,
                status=ExternalOperationStatus.RUNNING,
                started_at=now,
                heartbeat_at=now,
                safe_metadata=metadata,
            )
            session.add(attempt)
            self._add_event(
                session,
                operation,
                previous_status=ExternalOperationStatus.STARTING,
                new_status=ExternalOperationStatus.RUNNING,
                event_type="attempt_started",
                safe_metadata={"attempt_id": attempt.attempt_id},
                timestamp=now,
            )
            await session.commit()
            return _attempt_from_model(attempt)

    async def record_heartbeat(self, operation_id: str) -> ExternalOperation:
        """Advance a non-secret liveness marker for long-running external work."""
        async with self._database.session() as session:
            now = datetime.now(UTC)
            renewed = await session.execute(
                update(ExternalOperationModel)
                .where(
                    ExternalOperationModel.operation_id == operation_id,
                    ExternalOperationModel.status.in_(
                        (
                            ExternalOperationStatus.STARTING,
                            ExternalOperationStatus.RUNNING,
                            ExternalOperationStatus.CANCELLATION_REQUESTED,
                        )
                    ),
                )
                .values(heartbeat_at=now, updated_at=now)
                .execution_options(synchronize_session=False)
            )
            if cast(CursorResult[Any], renewed).rowcount != 1:
                msg = "external operation heartbeat lease is no longer active"
                raise OperationLeaseLostError(msg)
            operation = await _require_operation(session, operation_id)
            await session.refresh(operation)
            latest = await session.scalar(
                select(ExternalOperationAttemptModel)
                .where(ExternalOperationAttemptModel.operation_id == operation_id)
                .order_by(ExternalOperationAttemptModel.attempt.desc())
            )
            if latest is not None:
                latest.heartbeat_at = now
            await session.commit()
            return _operation_from_model(operation)

    async def record_result(
        self,
        operation_id: str,
        *,
        external_reference: str | None,
        result_payload: dict[str, Any] | None,
    ) -> ExternalOperation:
        """Commit a successful result before any caller advances workflow state."""
        return await self._record_success(
            operation_id,
            external_reference=external_reference,
            result_payload=result_payload,
            allowed_statuses={
                ExternalOperationStatus.RUNNING,
                ExternalOperationStatus.CANCELLATION_REQUESTED,
            },
            conflict_is_lease_loss=True,
        )

    async def record_reconciled_result(
        self,
        operation_id: str,
        *,
        external_reference: str | None,
        result_payload: dict[str, Any] | None,
    ) -> ExternalOperation:
        """Commit a result proven after an UNKNOWN recovery fence was acquired."""
        return await self._record_success(
            operation_id,
            external_reference=external_reference,
            result_payload=result_payload,
            allowed_statuses=_RECOVERY_FENCED_STATUSES,
            conflict_is_lease_loss=False,
        )

    async def record_reconciliation_note(
        self,
        operation_id: str,
        *,
        method: str,
        result: str,
        safe_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append what a reconciliation attempt observed, without changing the status.

        A reconciliation that concludes nothing is exactly the case an operator later has to
        understand, and the only record of it used to be a log line. This keeps the reason
        with the operation, where the next reader is already looking.
        """
        observations = _safe_metadata(safe_metadata or {})
        async with self._database.session() as session:
            operation = await _require_operation(session, operation_id)
            now = datetime.now(UTC)
            operation.safe_metadata = {
                **operation.safe_metadata,
                "reconciliation": {
                    "method": method,
                    "result": result,
                    "observed_at": now.isoformat(),
                    **observations,
                },
            }
            operation.updated_at = now
            self._add_event(
                session,
                operation,
                previous_status=operation.status,
                new_status=operation.status,
                event_type="operation_reconciliation_attempted",
                safe_metadata={"method": method, "result": result, **observations},
                timestamp=now,
            )
            await session.commit()

    async def _record_success(
        self,
        operation_id: str,
        *,
        external_reference: str | None,
        result_payload: dict[str, Any] | None,
        allowed_statuses: set[ExternalOperationStatus],
        conflict_is_lease_loss: bool,
    ) -> ExternalOperation:
        """CAS one permitted status to success so a recovery fence cannot be overwritten."""
        payload = _safe_metadata(result_payload or {})
        async with self._database.session() as session:
            operation = await _require_operation(session, operation_id)
            prior = operation.status
            if prior not in allowed_statuses:
                msg = f"external operation result is fenced from status {prior.value}"
                if conflict_is_lease_loss:
                    raise OperationLeaseLostError(msg)
                raise ExternalOperationError(msg)
            now = datetime.now(UTC)
            completed = await session.execute(
                update(ExternalOperationModel)
                .where(
                    ExternalOperationModel.operation_id == operation_id,
                    ExternalOperationModel.status == prior,
                    ExternalOperationModel.attempt == operation.attempt,
                )
                .values(
                    status=ExternalOperationStatus.SUCCEEDED,
                    external_reference=external_reference or operation.external_reference,
                    result_payload=payload,
                    completed_at=now,
                    heartbeat_at=now,
                    error_code=None,
                    error_message=None,
                    compensation_status=None,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            if cast(CursorResult[Any], completed).rowcount != 1:
                msg = "external operation status changed before its result could be recorded"
                if conflict_is_lease_loss:
                    raise OperationLeaseLostError(msg)
                raise ExternalOperationError(msg)
            await session.refresh(operation)
            await self._finish_latest_attempt(
                session, operation, ExternalOperationStatus.SUCCEEDED, now
            )
            self._add_event(
                session,
                operation,
                previous_status=prior,
                new_status=ExternalOperationStatus.SUCCEEDED,
                event_type="operation_succeeded",
                timestamp=now,
            )
            await session.commit()
            return _operation_from_model(operation)

    async def record_failure(
        self,
        operation_id: str,
        *,
        status: ExternalOperationStatus,
        error_code: str,
        error_message: str,
        external_reference: str | None = None,
        compensation_status: CompensationStatus | None = None,
    ) -> ExternalOperation:
        """Record a bounded, credential-redacted terminal or retryable failure."""
        return await self._record_failure(
            operation_id,
            status=status,
            error_code=error_code,
            error_message=error_message,
            external_reference=external_reference,
            compensation_status=compensation_status,
            allowed_statuses={
                ExternalOperationStatus.PENDING,
                ExternalOperationStatus.STARTING,
                ExternalOperationStatus.RUNNING,
                ExternalOperationStatus.CANCELLATION_REQUESTED,
            },
            conflict_is_lease_loss=True,
        )

    async def record_recovery_failure(
        self,
        operation_id: str,
        *,
        status: ExternalOperationStatus,
        error_code: str,
        error_message: str,
        external_reference: str | None = None,
        compensation_status: CompensationStatus | None = None,
    ) -> ExternalOperation:
        """Finish recovery only while its unresolved fence remains authoritative."""
        if status not in {
            ExternalOperationStatus.CANCELLED,
            ExternalOperationStatus.FAILED_RETRYABLE,
            ExternalOperationStatus.FAILED_TERMINAL,
            ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
            ExternalOperationStatus.AWAITING_RECONCILIATION,
        }:
            msg = "record_recovery_failure requires a recovery failure status"
            raise ValueError(msg)
        return await self._record_failure(
            operation_id,
            status=status,
            error_code=error_code,
            error_message=error_message,
            external_reference=external_reference,
            compensation_status=compensation_status,
            allowed_statuses=_RECOVERY_FENCED_STATUSES,
            conflict_is_lease_loss=False,
        )

    async def _record_failure(
        self,
        operation_id: str,
        *,
        status: ExternalOperationStatus,
        error_code: str,
        error_message: str,
        external_reference: str | None,
        compensation_status: CompensationStatus | None,
        allowed_statuses: set[ExternalOperationStatus],
        conflict_is_lease_loss: bool,
    ) -> ExternalOperation:
        """CAS a failure transition while respecting live and recovery ownership fences."""
        if status not in {
            ExternalOperationStatus.CANCELLED,
            ExternalOperationStatus.FAILED_RETRYABLE,
            ExternalOperationStatus.FAILED_TERMINAL,
            ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
            ExternalOperationStatus.AWAITING_RECONCILIATION,
            ExternalOperationStatus.CANCELLATION_REQUESTED,
        }:
            msg = "record_failure requires a failure or cancellation status"
            raise ValueError(msg)
        async with self._database.session() as session:
            operation = await _require_operation(session, operation_id)
            prior = operation.status
            if prior not in allowed_statuses:
                msg = f"external operation failure is fenced from status {prior.value}"
                if conflict_is_lease_loss:
                    raise OperationLeaseLostError(msg)
                raise ExternalOperationError(msg)
            now = datetime.now(UTC)
            failed = await session.execute(
                update(ExternalOperationModel)
                .where(
                    ExternalOperationModel.operation_id == operation_id,
                    ExternalOperationModel.status == prior,
                    ExternalOperationModel.attempt == operation.attempt,
                )
                .values(
                    status=status,
                    error_code=error_code[:128],
                    error_message=_safe_error_message(error_message),
                    external_reference=external_reference or operation.external_reference,
                    compensation_status=compensation_status,
                    heartbeat_at=now,
                    completed_at=now if status in _TERMINAL_STATUSES else operation.completed_at,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            if cast(CursorResult[Any], failed).rowcount != 1:
                msg = "external operation status changed before its failure could be recorded"
                if conflict_is_lease_loss:
                    raise OperationLeaseLostError(msg)
                raise ExternalOperationError(msg)
            await session.refresh(operation)
            await self._finish_latest_attempt(session, operation, status, now)
            self._add_event(
                session,
                operation,
                previous_status=prior,
                new_status=status,
                event_type="operation_failed"
                if status != ExternalOperationStatus.CANCELLED
                else "operation_cancelled",
                timestamp=now,
            )
            await session.commit()
            return _operation_from_model(operation)

    async def list_incomplete_operations(self) -> list[ExternalOperation]:
        """List only externally uncertain operations that startup recovery must reconcile."""
        statuses = (
            ExternalOperationStatus.STARTING,
            ExternalOperationStatus.RUNNING,
            ExternalOperationStatus.CANCELLATION_REQUESTED,
        )
        async with self._database.session() as session:
            models = list(
                await session.scalars(
                    select(ExternalOperationModel)
                    .where(ExternalOperationModel.status.in_(statuses))
                    .order_by(ExternalOperationModel.created_at)
                )
            )
            return [_operation_from_model(model) for model in models]

    async def list_operations_for_workflow(self, workflow_id: str) -> list[ExternalOperation]:
        """Return all operation evidence for one recoverable parent or child workflow."""
        async with self._database.session() as session:
            models = list(
                await session.scalars(
                    select(ExternalOperationModel)
                    .where(ExternalOperationModel.workflow_id == workflow_id)
                    .order_by(ExternalOperationModel.created_at)
                )
            )
            return [_operation_from_model(model) for model in models]

    async def list_operations_for_feature_since(
        self, feature_id: str, since: datetime
    ) -> list[ExternalOperation]:
        """Return the effects one feature recorded at or after a time.

        This is how a durable action finds out what it actually did. It is a query rather
        than something the executor writes down as it goes, because the case that matters is
        the executor that died: whatever it meant to record, it did not. The journal is the
        only witness left, and it is written before each effect rather than after.
        """
        async with self._database.session() as session:
            models = list(
                await session.scalars(
                    select(ExternalOperationModel)
                    .where(
                        ExternalOperationModel.feature_id == feature_id,
                        ExternalOperationModel.created_at >= since,
                    )
                    .order_by(ExternalOperationModel.created_at)
                )
            )
            return [_operation_from_model(model) for model in models]

    async def list_operations_for_action(self, action_id: str) -> list[ExternalOperation]:
        """Return only effects durably associated with one feature action."""
        async with self._database.session() as session:
            models = list(
                await session.scalars(
                    select(ExternalOperationModel)
                    .join(
                        FeatureActionExternalOperationModel,
                        FeatureActionExternalOperationModel.operation_id
                        == ExternalOperationModel.operation_id,
                    )
                    .where(FeatureActionExternalOperationModel.action_id == action_id)
                    .order_by(FeatureActionExternalOperationModel.linked_at)
                )
            )
            return [_operation_from_model(model) for model in models]

    async def get_current_validation_results(
        self, *, workflow_id: str, repository_id: str, repository_revision: str
    ) -> list[ExternalOperation]:
        """Return only current revision-matching validation evidence for repository review."""
        async with self._database.session() as session:
            models = list(
                await session.scalars(
                    select(ExternalOperationModel)
                    .where(
                        ExternalOperationModel.workflow_id == workflow_id,
                        ExternalOperationModel.repository_id == repository_id,
                        ExternalOperationModel.operation_type.in_(_VALIDATION_OPERATION_TYPES),
                        ExternalOperationModel.repository_revision == repository_revision,
                        ExternalOperationModel.is_current.is_(True),
                        ExternalOperationModel.status == ExternalOperationStatus.SUCCEEDED,
                    )
                    .order_by(ExternalOperationModel.created_at)
                )
            )
            return [_operation_from_model(model) for model in models]

    async def is_stale(self, operation: ExternalOperation) -> bool:
        """Return whether an unfinished operation has exceeded the bounded heartbeat window."""
        heartbeat = operation.heartbeat_at or operation.started_at
        return heartbeat is not None and datetime.now(UTC) - heartbeat > self._stale_after

    async def list_unresolved_operations(self, *, limit: int = 100) -> list[ExternalOperation]:
        """Return the operations an operator has to decide about, newest first.

        The readiness gate publishes only a count, which tells an operator that something
        needs attention and nothing about what. Reconciling then meant querying the database
        directly, which is exactly what an operator API is supposed to replace.
        """
        cutoff = datetime.now(UTC) - self._stale_after
        async with self._database.session() as session:
            models = list(
                await session.scalars(
                    select(ExternalOperationModel)
                    .where(
                        or_(
                            ExternalOperationModel.status.in_(
                                (
                                    ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
                                    ExternalOperationStatus.AWAITING_RECONCILIATION,
                                    ExternalOperationStatus.CANCELLATION_REQUESTED,
                                    ExternalOperationStatus.FAILED_RETRYABLE,
                                )
                            ),
                            and_(
                                ExternalOperationModel.status.in_(
                                    (
                                        ExternalOperationStatus.STARTING,
                                        ExternalOperationStatus.RUNNING,
                                    )
                                ),
                                _stale_liveness_clause(cutoff),
                            ),
                        )
                    )
                    .order_by(ExternalOperationModel.created_at.desc())
                    .limit(limit)
                )
            )
            return [_operation_from_model(model) for model in models]

    async def list_operations_for_feature(
        self,
        feature_id: str,
        *,
        operation_types: Sequence[ExternalOperationType] | None = None,
        limit: int = 200,
    ) -> list[ExternalOperation]:
        """Return one feature's journal rows, oldest first, optionally filtered by type.

        The executions read model uses this to show the pre-coding model calls beside the
        artifact-derived stage rows. A read, nothing more: nothing decides anything from
        what it returns.
        """
        async with self._database.session() as session:
            query = (
                select(ExternalOperationModel)
                .where(ExternalOperationModel.feature_id == feature_id)
                .order_by(ExternalOperationModel.created_at.asc())
                .limit(limit)
            )
            if operation_types is not None:
                query = query.where(ExternalOperationModel.operation_type.in_(operation_types))
            models = list(await session.scalars(query))
            return [_operation_from_model(model) for model in models]

    async def list_operations_for_repository(
        self,
        feature_id: str,
        repository_id: str,
        *,
        limit: int = 50,
    ) -> list[ExternalOperation]:
        """Return one repository's journal rows, newest first and bounded.

        This is the read behind the workstream operations endpoint: the journal is the only
        record with a start time and a heartbeat while an operation is still in flight, and
        every operational question of the 185-194 verification cycle was answered by querying
        these rows by hand. Newest first because the reader's question is "what is happening
        now", and bounded because the endpoint's answer must stay one screen, not a history.
        """
        async with self._database.session() as session:
            models = list(
                await session.scalars(
                    select(ExternalOperationModel)
                    .where(
                        ExternalOperationModel.feature_id == feature_id,
                        ExternalOperationModel.repository_id == repository_id,
                    )
                    .order_by(ExternalOperationModel.created_at.desc())
                    .limit(limit)
                )
            )
            return [_operation_from_model(model) for model in models]

    async def list_deferred_operations_no_request_will_reach(
        self, *, older_than_seconds: float, limit: int = 100
    ) -> list[ExternalOperation]:
        """Return deferred remote effects whose feature will never make another request.

        Deferring an interrupted push or pull request is right: the next credentialed request
        that re-enters the same code path reconciles it, and that request is the normal path.
        It stops being right when there will be no next request. A feature that reached a
        terminal status and is never resumed leaves its operation deferred for ever, and
        because a deferred operation is no longer returned by `list_incomplete_operations`,
        no sweep looks at it again either -- it simply accumulates in the operator queue with
        nobody ever being asked a question about it.

        This finds those, and only after they have aged: a failed feature may still be
        resumed by a person, and asking about an effect somebody is about to confirm anyway
        would be asking too early. It establishes nothing about the provider and guesses
        nothing -- what it changes is that a person is asked once rather than never.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=older_than_seconds)
        async with self._database.session() as session:
            models = list(
                await session.scalars(
                    select(ExternalOperationModel)
                    .join(
                        FeatureWorkflowModel,
                        FeatureWorkflowModel.feature_id == ExternalOperationModel.feature_id,
                    )
                    .where(
                        ExternalOperationModel.status
                        == ExternalOperationStatus.AWAITING_RECONCILIATION,
                        ExternalOperationModel.updated_at < cutoff,
                        FeatureWorkflowModel.status.in_(
                            item.value for item in TERMINAL_FEATURE_STATUSES
                        ),
                    )
                    .order_by(ExternalOperationModel.created_at)
                    .limit(limit)
                )
            )
            return [_operation_from_model(model) for model in models]

    async def list_operations_requiring_manual_review(
        self, *, limit: int = 100
    ) -> list[ExternalOperation]:
        """Return the operations recovery decided nobody but a person can settle.

        Narrowed to the error codes recovery itself writes. The stale-claim fence briefly
        looks identical, and a concurrent sweep reading that window would otherwise fail a
        feature over an operation that was about to be classified as merely deferred.
        """
        async with self._database.session() as session:
            models = list(
                await session.scalars(
                    select(ExternalOperationModel)
                    .where(
                        ExternalOperationModel.status
                        == ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
                        ExternalOperationModel.compensation_status
                        == CompensationStatus.MANUAL_REVIEW_REQUIRED,
                        ExternalOperationModel.error_code.in_(_RECOVERY_MANUAL_REVIEW_CODES),
                    )
                    .order_by(ExternalOperationModel.created_at.desc())
                    .limit(limit)
                )
            )
            return [_operation_from_model(model) for model in models]

    async def unresolved_critical_count(self) -> int:
        """Count only the operations no credentialed request can still settle by itself."""
        cutoff = datetime.now(UTC) - self._stale_after
        async with self._database.session() as session:
            count = await session.scalar(
                select(func.count())
                .select_from(ExternalOperationModel)
                .where(
                    or_(
                        ExternalOperationModel.status.in_(
                            (
                                ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
                                ExternalOperationStatus.CANCELLATION_REQUESTED,
                            )
                        ),
                        and_(
                            ExternalOperationModel.status.in_(
                                (
                                    ExternalOperationStatus.STARTING,
                                    ExternalOperationStatus.RUNNING,
                                )
                            ),
                            _stale_liveness_clause(cutoff),
                        ),
                    )
                )
            )
            return int(count or 0)

    async def events_for_operation(self, operation_id: str) -> list[ExternalOperationEvent]:
        """Read the append-only timeline for diagnostics and tests."""
        async with self._database.session() as session:
            models = list(
                await session.scalars(
                    select(ExternalOperationEventModel)
                    .where(ExternalOperationEventModel.operation_id == operation_id)
                    .order_by(ExternalOperationEventModel.id)
                )
            )
            return [_event_from_model(model) for model in models]

    def _add_event(
        self,
        session: Any,
        operation: ExternalOperationModel,
        *,
        previous_status: ExternalOperationStatus | None,
        new_status: ExternalOperationStatus,
        event_type: str,
        safe_metadata: dict[str, Any] | None = None,
        timestamp: datetime,
    ) -> None:
        """Append a credential-free transition event in the caller's transaction."""
        session.add(
            ExternalOperationEventModel(
                event_id=f"operation-event-{uuid4()}",
                operation_id=operation.operation_id,
                previous_status=previous_status,
                new_status=new_status,
                timestamp=timestamp,
                attempt=operation.attempt,
                workflow_id=operation.workflow_id,
                feature_id=operation.feature_id,
                child_workflow_id=operation.child_workflow_id,
                repository_id=operation.repository_id,
                event_type=event_type,
                safe_metadata=_safe_metadata(safe_metadata or {}),
            )
        )

    async def _finish_latest_attempt(
        self,
        session: Any,
        operation: ExternalOperationModel,
        status: ExternalOperationStatus,
        now: datetime,
    ) -> None:
        """Synchronize the latest attempt with the durable parent-operation result."""
        latest = await session.scalar(
            select(ExternalOperationAttemptModel)
            .where(ExternalOperationAttemptModel.operation_id == operation.operation_id)
            .order_by(ExternalOperationAttemptModel.attempt.desc())
        )
        if latest is not None:
            latest.status = status
            latest.heartbeat_at = now
            if status in _TERMINAL_STATUSES:
                latest.completed_at = now


async def _require_operation(session: Any, operation_id: str) -> ExternalOperationModel:
    """Return a journal model in a transaction or raise a safe domain error."""
    operation = await session.get(ExternalOperationModel, operation_id)
    if operation is None:
        msg = f"external operation not found: {operation_id}"
        raise ExternalOperationError(msg)
    return cast(ExternalOperationModel, operation)


def deterministic_operation_key(
    *,
    workflow_id: str,
    repository_id: str | None,
    operation_type: ExternalOperationType,
    logical_step: str,
    input_fingerprint: str,
    feature_id: str | None = None,
    child_workflow_id: str | None = None,
) -> str:
    """Build a stable operation key without including raw request contents or credentials."""
    scope = feature_id or workflow_id
    child = child_workflow_id or "parent"
    repository = repository_id or "none"
    return f"{scope}:{child}:{repository}:{operation_type.value}:{logical_step}:{input_fingerprint}"


def successor_operation_key(idempotency_key: str, *, child_attempt: int) -> str:
    """Build the key a later attempt's fresh run of the same work is journaled under.

    Derived from the base key rather than computed with the attempt as key material, and the
    difference is the whole of the 23- rule. The base key is what every caller looks up
    first, so a *succeeded* row is still found and reused across attempts -- a preserved
    workspace still must not re-clone or re-install at an unchanged revision. Only after the
    base row is found to have ended terminally under an *older* attempt does anything reach
    this function, and then a distinct key is not a choice: `uq_external_operations_key` is
    unique, so the successor cannot share the ended row's key even though it is the same
    work. Two rows, two attempts, one lineage.

    Deterministic in the attempt, so re-entering the same attempt resolves to the same
    successor row and the within-attempt terminal rule still bites on it.
    """
    return f"{idempotency_key}#attempt-{child_attempt}"


def fingerprint_operation_input(value: dict[str, Any]) -> str:
    """Hash normalized safe input instead of retaining potentially sensitive raw inputs."""
    encoded = json.dumps(_safe_metadata(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


_TERMINAL_STATUSES = {
    ExternalOperationStatus.CANCELLED,
    ExternalOperationStatus.SUCCEEDED,
    ExternalOperationStatus.FAILED_RETRYABLE,
    ExternalOperationStatus.FAILED_TERMINAL,
    ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
    ExternalOperationStatus.AWAITING_RECONCILIATION,
}

# The resting statuses only recovery or a reconciling credentialed request may write from.
# A live worker's lease has already been fenced by the time an operation reaches one.
_RECOVERY_FENCED_STATUSES = {
    ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE,
    ExternalOperationStatus.AWAITING_RECONCILIATION,
}

# Written by the recovery sweep's generic exception handler: its own code raised while
# reconciling, so nothing is known about the provider and nothing about this repository is
# implicated. Named rather than repeated as a literal, because the escalation that reads it
# has to classify a defect here differently from an unconfirmed external effect.
RECOVERY_DEFECT_ERROR_CODE = "recovery_error"

# Written only by the recovery sweep, and only where it has established that no credentialed
# request can settle the operation either.
_RECOVERY_MANUAL_REVIEW_CODES = ("reconciliation_required", RECOVERY_DEFECT_ERROR_CODE)

_ALLOWED_MANUAL_TRANSITIONS = {
    # Retryable failures may restart only through ``claim_operation``, which also
    # enforces the retry budget. The generic transition API never reopens a terminal row.
    ExternalOperationStatus.STARTING: frozenset({ExternalOperationStatus.PENDING}),
    ExternalOperationStatus.CANCELLATION_REQUESTED: frozenset(
        {
            ExternalOperationStatus.STARTING,
            ExternalOperationStatus.RUNNING,
        }
    ),
}

_VALIDATION_OPERATION_TYPES = {
    ExternalOperationType.RUN_FORMATTER,
    ExternalOperationType.INSTALL_DEPENDENCIES,
    ExternalOperationType.RUN_LINTER,
    ExternalOperationType.RUN_TYPECHECK,
    ExternalOperationType.RUN_TESTS,
    ExternalOperationType.RUN_BUILD,
}

_SENSITIVE_ENVIRONMENT_KEYS = {
    "DATABASE_URL",
    "GITHUB_TOKEN",
    "OPENAI_API_KEY",
    "PLATFORM_GIT_TOKEN",
    "PLATFORM_API_KEY",
    "REDIS_URL",
}
_SENSITIVE_ENVIRONMENT_SUFFIXES = (
    "_ACCESS_KEY",
    "_API_KEY",
    "_AUTHORIZATION",
    "_CREDENTIAL",
    "_PASSWORD",
    "_PRIVATE_KEY",
    "_SECRET",
    "_TOKEN",
)


def _stale_liveness_clause(cutoff: datetime) -> Any:
    """Match only rows whose current durable heartbeat remains older than the cutoff."""
    return or_(
        and_(
            ExternalOperationModel.heartbeat_at.is_not(None),
            ExternalOperationModel.heartbeat_at <= cutoff,
        ),
        and_(
            ExternalOperationModel.heartbeat_at.is_(None),
            ExternalOperationModel.started_at.is_not(None),
            ExternalOperationModel.started_at <= cutoff,
        ),
    )


def _safe_error_message(message: str) -> str:
    """Keep errors operationally useful while excluding common credential-shaped values."""
    redacted = message.replace("\n", " ")
    for key, secret in os.environ.items():
        normalized_key = key.upper()
        sensitive_key = normalized_key in _SENSITIVE_ENVIRONMENT_KEYS or normalized_key.endswith(
            _SENSITIVE_ENVIRONMENT_SUFFIXES
        )
        if secret and sensitive_key and secret in redacted:
            if len(secret) < 4:
                return "provider or subprocess error redacted"
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = redacted[:512]
    for marker in ("token", "secret", "api_key", "authorization", "password"):
        if marker in redacted.lower():
            return "provider or subprocess error redacted"
    return redacted


def _safe_metadata(value: dict[str, Any]) -> dict[str, Any]:
    """Reject credential-like keys before JSON persistence and preserve only JSON values."""
    prohibited = ("token", "secret", "password", "authorization", "api_key", "credential")

    def validate(item: Any) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                if not isinstance(key, str) or any(marker in key.lower() for marker in prohibited):
                    msg = "credential-like values must not be stored in external operation metadata"
                    raise ExternalOperationError(msg)
                validate(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                validate(nested)

    validate(value)
    try:
        json.dumps(value)
    except (TypeError, ValueError) as error:
        msg = "external operation metadata must be JSON serializable"
        raise ExternalOperationError(msg) from error
    return dict(value)


def _operation_from_model(model: ExternalOperationModel) -> ExternalOperation:
    """Hydrate the strict domain model from a database row."""
    return ExternalOperation.model_validate(
        {
            "operation_id": model.operation_id,
            "workflow_id": model.workflow_id,
            "feature_id": model.feature_id,
            "child_workflow_id": model.child_workflow_id,
            "repository_id": model.repository_id,
            "operation_type": model.operation_type,
            "idempotency_key": model.idempotency_key,
            "status": model.status,
            "attempt": model.attempt,
            "max_attempts": model.max_attempts,
            "input_fingerprint": model.input_fingerprint,
            "repository_revision": model.repository_revision,
            "command_fingerprint": model.command_fingerprint,
            "is_current": model.is_current,
            "superseded_by_operation_id": model.superseded_by_operation_id,
            "external_reference": model.external_reference,
            "started_at": _as_utc(model.started_at),
            "heartbeat_at": _as_utc(model.heartbeat_at),
            "completed_at": _as_utc(model.completed_at),
            "error_code": model.error_code,
            "error_message": model.error_message,
            "result_payload": model.result_payload,
            "compensation_status": model.compensation_status,
            "safe_metadata": model.safe_metadata,
        }
    )


def _attempt_from_model(model: ExternalOperationAttemptModel) -> ExternalOperationAttempt:
    """Hydrate one durable attempt record."""
    return ExternalOperationAttempt.model_validate(
        {
            "attempt_id": model.attempt_id,
            "operation_id": model.operation_id,
            "attempt": model.attempt,
            "provider": model.provider,
            "model": model.model,
            "workspace_path": model.workspace_path,
            "task_plan_artifact_id": model.task_plan_artifact_id,
            "contract_artifact_id": model.contract_artifact_id,
            "process_id": model.process_id,
            "external_run_id": model.external_run_id,
            "status": model.status,
            "started_at": _as_utc(model.started_at),
            "heartbeat_at": _as_utc(model.heartbeat_at),
            "completed_at": _as_utc(model.completed_at),
            "safe_metadata": model.safe_metadata,
        }
    )


def _event_from_model(model: ExternalOperationEventModel) -> ExternalOperationEvent:
    """Hydrate one append-only operation event."""
    return ExternalOperationEvent.model_validate(
        {
            "event_id": model.event_id,
            "operation_id": model.operation_id,
            "previous_status": model.previous_status,
            "new_status": model.new_status,
            "timestamp": _as_utc(model.timestamp),
            "attempt": model.attempt,
            "workflow_id": model.workflow_id,
            "feature_id": model.feature_id,
            "child_workflow_id": model.child_workflow_id,
            "repository_id": model.repository_id,
            "event_type": model.event_type,
            "safe_metadata": model.safe_metadata,
        }
    )


def _as_utc(value: datetime | None) -> datetime | None:
    """Normalize SQLite timestamps while preserving their original absolute value."""
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = [
    "ExternalOperationError",
    "ExternalOperationJournal",
    "OperationInProgressError",
    "OperationLeaseLostError",
    "OperationReconciliationRequired",
    "OperationReplayRefused",
    "OperationResult",
    "OperationTransitionConflictError",
    "deterministic_operation_key",
    "fingerprint_operation_input",
    "successor_operation_key",
]
