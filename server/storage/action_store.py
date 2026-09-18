"""Durable and in-memory stores for workflow-changing actions.

Every state change here is a conditional UPDATE guarded on what the caller believed was
true -- the status it expected, the lease it holds. That is deliberate: two API processes may
receive the same confirmation at the same moment, and a read-then-write would let both
execute. The external-operation journal is built the same way for the same reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError

from state.enums import ACTIVE_ACTION_STATUSES, FeatureActionStatus
from state.feature_actions import FeatureAction, normalize_action_payload
from storage.db import Database
from storage.external_operation_store import fingerprint_operation_input
from storage.models import FeatureActionModel

# How long a claimed action stays owned without a heartbeat. Long enough that an ordinary
# slow step -- a live retry clones, installs, codes and validates -- never loses its lease to
# a scheduling hiccup, and short enough that a crashed executor is reconciled within a
# support conversation rather than a shift. The executor renews at a third of this.
DEFAULT_LEASE_SECONDS = 180.0


class ActionConflictError(RuntimeError):
    """Raised when an action cannot move as asked, because something else already moved it."""


class ActionInProgressError(ActionConflictError):
    """Raised when an action is genuinely being executed by a live owner right now."""


def deterministic_action_key(
    *,
    feature_id: str,
    repository_id: str | None,
    action_type: str,
    input_fingerprint: str,
) -> str:
    """Build the identity two requests must share to be the same action.

    Repository-scoped because two repositories may legitimately be retried with otherwise
    identical arguments, and those are two intents rather than a duplicate.
    """
    return f"{feature_id}:{repository_id or 'parent'}:{action_type}:{input_fingerprint}"


def request_action_key(
    *,
    feature_id: str,
    repository_id: str | None,
    action_type: str,
    request_idempotency_key: str,
) -> str:
    """Hash a client request key so an opaque caller value is never persisted verbatim."""
    digest = fingerprint_operation_input({"request_idempotency_key": request_idempotency_key})
    return f"{feature_id}:{repository_id or 'parent'}:{action_type}:request:{digest}"


def action_fingerprint(
    *,
    action_type: str,
    feature_id: str,
    repository_id: str | None,
    payload: dict[str, Any],
    context_version: str | None,
) -> str:
    """Hash the action together with the state it was decided against.

    ``context_version`` is what stops a stale confirmation replaying against a repository
    that has moved -- the current repository revision, the contract version, the repair
    proposal being approved. Without it the same button pressed before and after a change
    would be treated as one intent, which is how a retry could silently reuse a verdict that
    no longer applied.
    """
    return fingerprint_operation_input(
        {
            "action_type": action_type,
            "feature_id": feature_id,
            "repository_id": repository_id,
            "payload": normalize_action_payload(payload),
            "context_version": context_version,
        }
    )


class DatabaseFeatureActionStore:
    """Persist actions beside the feature they change."""

    def __init__(self, database: Database, *, lease_seconds: float = DEFAULT_LEASE_SECONDS) -> None:
        """Bind the shared database and the ownership window a claim buys."""
        if lease_seconds <= 0:
            msg = "lease_seconds must be positive"
            raise ValueError(msg)
        self._database = database
        self._lease = timedelta(seconds=lease_seconds)

    @property
    def lease_seconds(self) -> float:
        """Expose the configured lease window so an executor can pick a renewal interval."""
        return self._lease.total_seconds()

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
        """Return the action for this intent, creating it only the first time.

        The uniqueness of the idempotency key is enforced by the database, not by a prior
        read: two concurrent confirmations both find nothing, both insert, and exactly one
        wins. The loser reads the winner's row, which is the behaviour that makes a
        double-clicked Confirm button produce one effect.
        """
        fingerprint = action_fingerprint(
            action_type=action_type,
            feature_id=feature_id,
            repository_id=repository_id,
            payload=payload,
            context_version=context_version,
        )
        normalized_payload = normalize_action_payload(payload)
        key = (
            request_action_key(
                feature_id=feature_id,
                repository_id=repository_id,
                action_type=action_type,
                request_idempotency_key=request_idempotency_key,
            )
            if request_idempotency_key
            else deterministic_action_key(
                feature_id=feature_id,
                repository_id=repository_id,
                action_type=action_type,
                input_fingerprint=fingerprint,
            )
        )
        existing = await self.get_by_key(key)
        if existing is not None:
            _require_same_request(existing, payload=normalized_payload)
            return existing, False
        row = FeatureActionModel(
            action_id=f"action-{uuid4()}",
            feature_id=feature_id,
            repository_id=repository_id,
            action_type=action_type,
            actor_id=actor_id,
            actor_display_name=actor_display_name,
            origin=origin,
            origin_message_id=origin_message_id,
            payload=normalized_payload,
            input_fingerprint=fingerprint,
            idempotency_key=key,
            status=FeatureActionStatus.CONFIRMED,
            attempt=0,
            max_attempts=max_attempts,
            created_at=datetime.now(UTC),
            external_operation_ids=[],
        )
        try:
            async with self._database.session() as session:
                session.add(row)
                await session.commit()
                await session.refresh(row)
                return _as_action(row), True
        except IntegrityError:
            # Another request inserted the same identity between the read and this write.
            # Its row is the one that counts.
            raced = await self.get_by_key(key)
            if raced is None:  # pragma: no cover - only reachable if the row was then deleted
                msg = "action could not be created or read back"
                raise ActionConflictError(msg) from None
            _require_same_request(raced, payload=normalized_payload)
            return raced, False

    async def get(self, action_id: str) -> FeatureAction | None:
        """Return one action by identifier."""
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(FeatureActionModel).where(FeatureActionModel.action_id == action_id)
                )
            ).scalar_one_or_none()
        return None if row is None else _as_action(row)

    async def get_by_key(self, idempotency_key: str) -> FeatureAction | None:
        """Return the action already recorded for one intent, if any."""
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(FeatureActionModel).where(
                        FeatureActionModel.idempotency_key == idempotency_key
                    )
                )
            ).scalar_one_or_none()
        return None if row is None else _as_action(row)

    async def list_for_feature(self, feature_id: str, *, limit: int = 100) -> list[FeatureAction]:
        """Return this feature's actions, newest first, for an operator view."""
        statement = (
            select(FeatureActionModel)
            .where(FeatureActionModel.feature_id == feature_id)
            .order_by(FeatureActionModel.created_at.desc())
            .limit(limit)
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [_as_action(row) for row in rows]

    async def claim(self, action_id: str, *, owner: str) -> FeatureAction | None:
        """Take ownership of an action that nothing live currently owns.

        Claimable from CONFIRMED, from a failed attempt with budget left, and from an active
        status whose lease has expired -- the crashed-executor case. Not claimable while a
        lease is live, which is what stops two workers running the same retry.
        """
        now = datetime.now(UTC)
        expires = now + self._lease
        claimable_status = FeatureActionModel.status.in_(
            [
                FeatureActionStatus.CONFIRMED,
                FeatureActionStatus.CLAIMED,
                FeatureActionStatus.EXECUTING,
            ]
        )
        unowned = or_(
            FeatureActionModel.lease_expires_at.is_(None),
            FeatureActionModel.lease_expires_at <= now,
        )
        async with self._database.session() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(FeatureActionModel)
                    .where(
                        FeatureActionModel.action_id == action_id,
                        claimable_status,
                        unowned,
                        FeatureActionModel.attempt < FeatureActionModel.max_attempts,
                    )
                    .values(
                        status=FeatureActionStatus.CLAIMED,
                        lease_owner=owner,
                        lease_expires_at=expires,
                        heartbeat_at=now,
                        started_at=now,
                        attempt=FeatureActionModel.attempt + 1,
                    )
                ),
            )
            await session.commit()
            if result.rowcount != 1:
                return None
            return await self._read(session, action_id)

    async def mark_executing(self, action_id: str, *, owner: str) -> FeatureAction | None:
        """Record that the external work has actually begun, still under this lease."""
        return await self._owned_update(
            action_id,
            owner=owner,
            values={"status": FeatureActionStatus.EXECUTING, "heartbeat_at": datetime.now(UTC)},
            expected=[FeatureActionStatus.CLAIMED, FeatureActionStatus.EXECUTING],
        )

    async def renew_lease(self, action_id: str, *, owner: str) -> bool:
        """Extend ownership. Returning false means this executor no longer owns the action."""
        now = datetime.now(UTC)
        updated = await self._owned_update(
            action_id,
            owner=owner,
            values={"heartbeat_at": now, "lease_expires_at": now + self._lease},
            expected=list(ACTIVE_ACTION_STATUSES),
        )
        return updated is not None

    async def record_success(
        self,
        action_id: str,
        *,
        owner: str,
        result_summary: str,
        external_operation_ids: Sequence[str] = (),
    ) -> FeatureAction:
        """Record the outcome and release the lease, only if this executor still owns it."""
        action = await self._owned_update(
            action_id,
            owner=owner,
            values={
                "status": FeatureActionStatus.SUCCEEDED,
                "result_summary": result_summary,
                "external_operation_ids": list(external_operation_ids),
                "completed_at": datetime.now(UTC),
                "lease_owner": None,
                "lease_expires_at": None,
            },
            expected=list(ACTIVE_ACTION_STATUSES),
        )
        if action is None:
            msg = f"action ownership was lost before its result could be recorded: {action_id}"
            raise ActionConflictError(msg)
        return action

    async def record_domain_result(
        self, action_id: str, *, owner: str, result_summary: str
    ) -> FeatureAction | None:
        """Commit proof that the domain call returned before finalizing the action."""
        return await self._owned_update(
            action_id,
            owner=owner,
            values={
                "domain_result_committed_at": datetime.now(UTC),
                "pending_result_summary": result_summary,
            },
            expected=list(ACTIVE_ACTION_STATUSES),
        )

    async def record_failure(
        self,
        action_id: str,
        *,
        owner: str,
        status: FeatureActionStatus,
        error_code: str,
        error_message: str,
    ) -> FeatureAction:
        """Record a terminal or reconciliation-pending outcome and release the lease."""
        action = await self._owned_update(
            action_id,
            owner=owner,
            values={
                "status": status,
                "error_code": error_code,
                "error_message": error_message,
                "completed_at": datetime.now(UTC),
                "lease_owner": None,
                "lease_expires_at": None,
            },
            expected=list(ACTIVE_ACTION_STATUSES),
        )
        if action is None:
            msg = f"action ownership was lost before its failure could be recorded: {action_id}"
            raise ActionConflictError(msg)
        return action

    async def release_unattempted(
        self, action_id: str, *, owner: str, error_code: str
    ) -> FeatureAction | None:
        """Return an action that attempted no external effect to a submittable resting state.

        Not a terminal outcome, because nothing happened that anybody has to be careful
        about. The caller is answered with the same refusal it always got, and the identity
        stays submittable so that a refusal saying "retry shortly" can actually be retried
        shortly.

        The attempt count is kept and the budget is widened by one instead, exactly as
        recovery does for an abandoned intent: an attempt really was made, and erasing that
        would make the ledger say something untrue about how many times a person asked. What
        changes is only that this one is not held against the next.
        """
        return await self._owned_update(
            action_id,
            owner=owner,
            values={
                "status": FeatureActionStatus.CONFIRMED,
                "error_code": error_code,
                # No `error_message`: the row is not reporting an outcome, and a failure
                # sentence left on a confirmed action reads as one.
                "error_message": None,
                "started_at": None,
                "lease_owner": None,
                "lease_expires_at": None,
                "max_attempts": FeatureActionModel.max_attempts + 1,
            },
            expected=list(ACTIVE_ACTION_STATUSES),
        )

    async def list_expired(self, *, limit: int = 100) -> list[FeatureAction]:
        """Return actions whose executor stopped renewing, newest lease first.

        This is the whole input to recovery. An action is here because somebody claimed it
        and then stopped saying they were alive, which is what a crash looks like from
        outside the process.
        """
        now = datetime.now(UTC)
        statement = (
            select(FeatureActionModel)
            .where(
                FeatureActionModel.status.in_(list(ACTIVE_ACTION_STATUSES)),
                FeatureActionModel.lease_expires_at.is_not(None),
                FeatureActionModel.lease_expires_at <= now,
            )
            .order_by(FeatureActionModel.lease_expires_at)
            .limit(limit)
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [_as_action(row) for row in rows]

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
        """Move an abandoned action to a decided state, without holding its lease.

        Guarded on the lease still being expired so a recovery sweep can never overwrite an
        executor that came back and renewed in the meantime.
        """
        now = datetime.now(UTC)
        values: dict[str, Any] = {
            "status": status,
            "lease_owner": None,
            "lease_expires_at": None,
            "error_code": error_code,
            "error_message": error_message,
        }
        if result_summary is not None:
            values["result_summary"] = result_summary
        if external_operation_ids:
            values["external_operation_ids"] = list(external_operation_ids)
        if status is not FeatureActionStatus.CONFIRMED:
            values["completed_at"] = now
        else:
            # Handed back for another attempt: it has not started and owns nothing.
            values["started_at"] = None
            # Recovery consumes one attempt. Give precisely that abandoned intent one new
            # claim without erasing the attempt history.
            values["max_attempts"] = FeatureActionModel.max_attempts + 1
        async with self._database.session() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(FeatureActionModel)
                    .where(
                        FeatureActionModel.action_id == action_id,
                        FeatureActionModel.status.in_(list(ACTIVE_ACTION_STATUSES)),
                        FeatureActionModel.lease_expires_at.is_not(None),
                        FeatureActionModel.lease_expires_at <= now,
                    )
                    .values(**values)
                ),
            )
            await session.commit()
            if result.rowcount != 1:
                return None
            return await self._read(session, action_id)

    async def operator_reconcile(
        self,
        action_id: str,
        *,
        actor_id: str,
        succeeded: bool,
        reason: str,
    ) -> FeatureAction | None:
        """Atomically close an uncertain action after an administrator checks its effects."""
        now = datetime.now(UTC)
        target = FeatureActionStatus.SUCCEEDED if succeeded else FeatureActionStatus.FAILED
        values: dict[str, Any] = {
            "status": target,
            "completed_at": now,
            "reconciled_by": actor_id,
            "reconciliation_reason": reason,
            "reconciled_at": now,
            "error_code": None if succeeded else "operator_verified_not_completed",
            "error_message": None if succeeded else reason,
        }
        if succeeded:
            values["result_summary"] = reason
        async with self._database.session() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(FeatureActionModel)
                    .where(
                        FeatureActionModel.action_id == action_id,
                        FeatureActionModel.status == FeatureActionStatus.REQUIRES_RECONCILIATION,
                    )
                    .values(**values)
                ),
            )
            await session.commit()
            if result.rowcount != 1:
                return None
            return await self._read(session, action_id)

    async def _owned_update(
        self,
        action_id: str,
        *,
        owner: str,
        values: dict[str, Any],
        expected: Sequence[FeatureActionStatus],
    ) -> FeatureAction | None:
        """Apply an update only while this executor still holds a live lease."""
        now = datetime.now(UTC)
        async with self._database.session() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(FeatureActionModel)
                    .where(
                        FeatureActionModel.action_id == action_id,
                        FeatureActionModel.lease_owner == owner,
                        FeatureActionModel.lease_expires_at.is_not(None),
                        FeatureActionModel.lease_expires_at > now,
                        FeatureActionModel.status.in_(list(expected)),
                    )
                    .values(**values)
                ),
            )
            await session.commit()
            if result.rowcount != 1:
                return None
            return await self._read(session, action_id)

    async def _read(self, session: Any, action_id: str) -> FeatureAction:
        """Read the row back inside the caller's session after a conditional write."""
        row = (
            await session.execute(
                select(FeatureActionModel).where(FeatureActionModel.action_id == action_id)
            )
        ).scalar_one()
        return _as_action(row)


class InMemoryFeatureActionStore:
    """The same contract for mock mode and isolated tests, with no database."""

    def __init__(self, *, lease_seconds: float = DEFAULT_LEASE_SECONDS) -> None:
        """Start empty with the same ownership window the durable store uses."""
        if lease_seconds <= 0:
            msg = "lease_seconds must be positive"
            raise ValueError(msg)
        self._actions: dict[str, FeatureAction] = {}
        self._by_key: dict[str, str] = {}
        self._lease = timedelta(seconds=lease_seconds)

    @property
    def lease_seconds(self) -> float:
        """Expose the configured lease window."""
        return self._lease.total_seconds()

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
        """Create the action once per identity, exactly as the durable store does."""
        fingerprint = action_fingerprint(
            action_type=action_type,
            feature_id=feature_id,
            repository_id=repository_id,
            payload=payload,
            context_version=context_version,
        )
        normalized_payload = normalize_action_payload(payload)
        key = (
            request_action_key(
                feature_id=feature_id,
                repository_id=repository_id,
                action_type=action_type,
                request_idempotency_key=request_idempotency_key,
            )
            if request_idempotency_key
            else deterministic_action_key(
                feature_id=feature_id,
                repository_id=repository_id,
                action_type=action_type,
                input_fingerprint=fingerprint,
            )
        )
        existing_id = self._by_key.get(key)
        if existing_id is not None:
            existing = self._actions[existing_id]
            _require_same_request(existing, payload=normalized_payload)
            return existing, False
        action = FeatureAction(
            action_id=f"action-{uuid4()}",
            feature_id=feature_id,
            repository_id=repository_id,
            action_type=action_type,
            actor_id=actor_id,
            actor_display_name=actor_display_name,
            origin=origin,
            origin_message_id=origin_message_id,
            payload=normalized_payload,
            input_fingerprint=fingerprint,
            idempotency_key=key,
            status=FeatureActionStatus.CONFIRMED,
            max_attempts=max_attempts,
            created_at=datetime.now(UTC),
        )
        self._actions[action.action_id] = action
        self._by_key[key] = action.action_id
        return action, True

    async def get(self, action_id: str) -> FeatureAction | None:
        """Return one action by identifier."""
        return self._actions.get(action_id)

    async def get_by_key(self, idempotency_key: str) -> FeatureAction | None:
        """Return the action recorded for one intent."""
        action_id = self._by_key.get(idempotency_key)
        return None if action_id is None else self._actions[action_id]

    async def list_for_feature(self, feature_id: str, *, limit: int = 100) -> list[FeatureAction]:
        """Return this feature's actions, newest first."""
        selected = [item for item in self._actions.values() if item.feature_id == feature_id]
        selected.sort(key=lambda item: item.created_at, reverse=True)
        return selected[:limit]

    async def claim(self, action_id: str, *, owner: str) -> FeatureAction | None:
        """Take ownership when nothing live holds it."""
        action = self._actions.get(action_id)
        now = datetime.now(UTC)
        if action is None or action.attempt >= action.max_attempts:
            return None
        if action.status not in {
            FeatureActionStatus.CONFIRMED,
            FeatureActionStatus.CLAIMED,
            FeatureActionStatus.EXECUTING,
        }:
            return None
        if action.lease_is_live(now=now):
            return None
        claimed = action.model_copy(
            update={
                "status": FeatureActionStatus.CLAIMED,
                "lease_owner": owner,
                "lease_expires_at": now + self._lease,
                "heartbeat_at": now,
                "started_at": now,
                "attempt": action.attempt + 1,
            }
        )
        self._actions[action_id] = claimed
        return claimed

    async def mark_executing(self, action_id: str, *, owner: str) -> FeatureAction | None:
        """Record that external work has begun."""
        return self._owned_update(
            action_id,
            owner=owner,
            update={"status": FeatureActionStatus.EXECUTING, "heartbeat_at": datetime.now(UTC)},
        )

    async def renew_lease(self, action_id: str, *, owner: str) -> bool:
        """Extend ownership, reporting whether it is still held."""
        now = datetime.now(UTC)
        return (
            self._owned_update(
                action_id,
                owner=owner,
                update={"heartbeat_at": now, "lease_expires_at": now + self._lease},
            )
            is not None
        )

    async def record_success(
        self,
        action_id: str,
        *,
        owner: str,
        result_summary: str,
        external_operation_ids: Sequence[str] = (),
    ) -> FeatureAction:
        """Record the outcome and release the lease."""
        action = self._owned_update(
            action_id,
            owner=owner,
            update={
                "status": FeatureActionStatus.SUCCEEDED,
                "result_summary": result_summary,
                "external_operation_ids": list(external_operation_ids),
                "completed_at": datetime.now(UTC),
                "lease_owner": None,
                "lease_expires_at": None,
            },
        )
        if action is None:
            msg = f"action ownership was lost before its result could be recorded: {action_id}"
            raise ActionConflictError(msg)
        return action

    async def record_domain_result(
        self, action_id: str, *, owner: str, result_summary: str
    ) -> FeatureAction | None:
        """Commit proof that the domain call returned before finalizing the action."""
        return self._owned_update(
            action_id,
            owner=owner,
            update={
                "domain_result_committed_at": datetime.now(UTC),
                "pending_result_summary": result_summary,
            },
        )

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
        action = self._owned_update(
            action_id,
            owner=owner,
            update={
                "status": status,
                "error_code": error_code,
                "error_message": error_message,
                "completed_at": datetime.now(UTC),
                "lease_owner": None,
                "lease_expires_at": None,
            },
        )
        if action is None:
            msg = f"action ownership was lost before its failure could be recorded: {action_id}"
            raise ActionConflictError(msg)
        return action

    async def release_unattempted(
        self, action_id: str, *, owner: str, error_code: str
    ) -> FeatureAction | None:
        """Return an action that attempted no external effect to a submittable state."""
        current = self._actions.get(action_id)
        if current is None:
            return None
        return self._owned_update(
            action_id,
            owner=owner,
            update={
                "status": FeatureActionStatus.CONFIRMED,
                "error_code": error_code,
                "error_message": None,
                "started_at": None,
                "lease_owner": None,
                "lease_expires_at": None,
                "max_attempts": current.max_attempts + 1,
            },
        )

    async def list_expired(self, *, limit: int = 100) -> list[FeatureAction]:
        """Return actions whose lease has lapsed."""
        now = datetime.now(UTC)
        expired = [
            item
            for item in self._actions.values()
            if item.status in ACTIVE_ACTION_STATUSES and not item.lease_is_live(now=now)
        ]
        expired.sort(key=lambda item: item.lease_expires_at or item.created_at)
        return expired[:limit]

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
        """Decide an abandoned action, refusing if its executor renewed in the meantime."""
        action = self._actions.get(action_id)
        now = datetime.now(UTC)
        if action is None or action.status not in ACTIVE_ACTION_STATUSES:
            return None
        if action.lease_is_live(now=now) or action.lease_expires_at is None:
            return None
        values: dict[str, Any] = {
            "status": status,
            "lease_owner": None,
            "lease_expires_at": None,
            "error_code": error_code,
            "error_message": error_message,
        }
        if result_summary is not None:
            values["result_summary"] = result_summary
        if external_operation_ids:
            values["external_operation_ids"] = list(external_operation_ids)
        if status is FeatureActionStatus.CONFIRMED:
            values["started_at"] = None
            values["max_attempts"] = max(action.max_attempts, action.attempt + 1)
        else:
            values["completed_at"] = now
        reconciled = action.model_copy(update=values)
        self._actions[action_id] = reconciled
        return reconciled

    async def operator_reconcile(
        self,
        action_id: str,
        *,
        actor_id: str,
        succeeded: bool,
        reason: str,
    ) -> FeatureAction | None:
        """Close only an uncertain action; a repeat cannot overwrite the first decision."""
        action = self._actions.get(action_id)
        if action is None or action.status is not FeatureActionStatus.REQUIRES_RECONCILIATION:
            return None
        now = datetime.now(UTC)
        values: dict[str, Any] = {
            "status": FeatureActionStatus.SUCCEEDED if succeeded else FeatureActionStatus.FAILED,
            "completed_at": now,
            "reconciled_by": actor_id,
            "reconciliation_reason": reason,
            "reconciled_at": now,
            "error_code": None if succeeded else "operator_verified_not_completed",
            "error_message": None if succeeded else reason,
        }
        if succeeded:
            values["result_summary"] = reason
        reconciled = action.model_copy(update=values)
        self._actions[action_id] = reconciled
        return reconciled

    def _owned_update(
        self, action_id: str, *, owner: str, update: dict[str, Any]
    ) -> FeatureAction | None:
        """Apply an update only while this owner still holds a live lease."""
        action = self._actions.get(action_id)
        now = datetime.now(UTC)
        if action is None or action.lease_owner != owner or not action.lease_is_live(now=now):
            return None
        if action.status not in ACTIVE_ACTION_STATUSES:
            return None
        updated = action.model_copy(update=update)
        self._actions[action_id] = updated
        return updated


def _as_action(row: FeatureActionModel) -> FeatureAction:
    """Project a durable row onto the shape services and the API share."""
    return FeatureAction(
        action_id=row.action_id,
        feature_id=row.feature_id,
        repository_id=row.repository_id,
        action_type=row.action_type,
        actor_id=row.actor_id,
        actor_display_name=row.actor_display_name,
        origin=row.origin,
        origin_message_id=row.origin_message_id,
        payload=dict(row.payload or {}),
        input_fingerprint=row.input_fingerprint,
        idempotency_key=row.idempotency_key,
        status=row.status,
        attempt=row.attempt,
        max_attempts=row.max_attempts,
        lease_owner=row.lease_owner,
        lease_expires_at=_as_utc(row.lease_expires_at),
        heartbeat_at=_as_utc(row.heartbeat_at),
        created_at=_as_utc(row.created_at) or datetime.now(UTC),
        started_at=_as_utc(row.started_at),
        completed_at=_as_utc(row.completed_at),
        domain_result_committed_at=_as_utc(row.domain_result_committed_at),
        pending_result_summary=row.pending_result_summary,
        result_summary=row.result_summary,
        error_code=row.error_code,
        error_message=row.error_message,
        external_operation_ids=list(row.external_operation_ids or []),
        reconciled_by=row.reconciled_by,
        reconciliation_reason=row.reconciliation_reason,
        reconciled_at=_as_utc(row.reconciled_at),
    )


def _as_utc(value: datetime | None) -> datetime | None:
    """Treat a naive timestamp from a database without timezone support as UTC.

    SQLite returns naive datetimes even for ``DateTime(timezone=True)`` columns. Lease
    comparisons must not raise on a development database that behaves differently from the
    deployed one.
    """
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _require_same_request(action: FeatureAction, *, payload: dict[str, Any]) -> None:
    """Refuse a client key reused for different data while allowing state to advance."""
    if action.payload != payload:
        msg = "action idempotency key was reused with a different request payload"
        raise ActionConflictError(msg)


__all__ = [
    "DEFAULT_LEASE_SECONDS",
    "ActionConflictError",
    "ActionInProgressError",
    "DatabaseFeatureActionStore",
    "InMemoryFeatureActionStore",
    "action_fingerprint",
    "deterministic_action_key",
    "request_action_key",
]
