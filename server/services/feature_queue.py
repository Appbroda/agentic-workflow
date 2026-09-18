"""The durable queue between accepting a feature and executing it.

`POST /features/start` used to run the whole of planning before it answered: the product
manager read the PRD, reconnaissance cloned every repository, and only then did the caller
learn their feature existed. A submission was therefore invisible for minutes, indistinguishable
from one that had failed, and lost outright if the request timed out.

This module is what makes acceptance and execution two separate durable facts. The feature and
one queue row commit together, the API answers, and a worker claims the row afterwards. A
claim carries a lease, so a process dying mid-run leaves work another process picks up rather
than a feature stuck in whatever state it had reached -- which is the difference between a
queue and a background task.

Provider credentials are deliberately not part of a queue entry. It records who asked, and the
dispatcher resolves that identity's stored credentials when it runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import uuid4

import structlog
from sqlalchemy import or_, select

from api.control_plane import RequestScopedCredentials
from state.failure_diagnosis import FeatureFailureClassification, classification_of
from storage.db import Database
from storage.models import DEFAULT_FEATURE_STEP_LIMIT, FeatureExecutionQueueModel

_LOGGER = structlog.get_logger("feature.queue")


class FeatureExecutionBusy(RuntimeError):
    """Another worker is executing this feature right now.

    Not a failure. A claim that arrives while the work is genuinely in progress must go back
    on the queue *without* spending an attempt: counting it means three of these in thirty
    seconds retire a feature that was running perfectly well, which is what happened to
    AB-Feature-104 while its first attempt was still working.
    """


class FeatureProviderFault(RuntimeError):
    """One claimed attempt ended because the model provider failed, not because of the work.

    A failure, but a retryable one, and the entry has attempts reserved for exactly this.
    Those attempts were never spent: the executor caught the fault, recorded the feature as
    needing a human, and returned normally, so the dispatcher closed the entry `succeeded`
    at attempt 1 of 3. AB-Feature-168 was put back in front of its author three times that
    way, and each manual resume did what attempts 2 and 3 were already reserved to do.

    Carries the original error so the terminal record, once the attempts really are gone,
    still names what failed rather than this wrapper.
    """

    def __init__(self, cause: Exception) -> None:
        """Keep the provider's own fault addressable rather than flattening it to a string."""
        super().__init__(str(cause))
        self.cause = cause


class FeatureWorkspaceCapacityUnavailable(RuntimeError):
    """One claimed attempt could not start because the worker volume was too full.

    A condition of the worker, not a verdict on the feature, and the same shape as
    ``FeatureProviderFault`` for the same reason: the entry has attempts reserved for it and
    it very often clears between them.

    Run 193 is why this is its own type rather than an ordinary exception. The capacity
    preflight failed, the executor recorded the feature `failed_requires_human` and returned
    normally -- so the status said a person was needed while the dispatcher went on claiming
    the feature, over and over, for as long as the volume stayed full. A manual cleanup freed
    12 GB and the very next claim completed the feature end to end, which is exactly what a
    status of "needs a human" had been denying for the whole preceding hour. The diagnostic
    text was never the problem; the status was the lie.

    Carries the original error so the terminal record, once the attempts really are gone,
    still names the capacity condition and its measured numbers.
    """

    def __init__(self, cause: Exception) -> None:
        """Keep the capacity error itself addressable rather than flattening it to a string."""
        super().__init__(str(cause))
        self.cause = cause


QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"

# What a worker should do with the feature an entry names. `start` was the only one for as
# long as the queue only carried submissions; resuming and granting a retry ran inside the
# request instead, which is how AB-Feature-108 lost seventy minutes of work to one cancelled
# socket. An entry that cannot say why it exists cannot carry those.
START = "start"
RESUME = "resume"
RETRY_WORKSTREAM = "retry_workstream"
# A person deciding a design question the platform stopped on. Distinct from `retry_workstream`
# even though both run one targeted attempt: this one records a verdict first, and the attempt
# it runs is bound by that verdict. Sharing the retry intent would have made the decision an
# argument of a grant, and the commonest verdict grants nothing at all.
ANSWER_DESIGN_VERDICT = "answer_design_verdict"
# A person deciding that a feature which did not land should publish what it has. Its own
# intent because what it runs is not a step: it opens pull requests across every eligible
# repository, running `git push` and provider calls for each, and none of that may happen
# inside the HTTP request that accepted it -- the shape that lost AB-Feature-108 seventy
# minutes to one cancelled socket. The refusal stays synchronous; only the work is queued.
PUBLISH = "publish"
# A person asking for changes to a feature that already completed and shipped its pull
# requests. The revision itself -- the request artifact, the replacement technical PRD, the
# reset children on `-V{n}` branches -- is applied synchronously by the control plane before
# this entry exists, so the entry carries nothing to apply: like `continue`, it goes straight
# to the next step. Its own intent rather than `continue` because the read model reads exactly
# this field to say what a feature is doing, and "revising" is the true answer.
REVISE = "revise"
# The step after a step: what a claim re-queues its own entry as. Deliberately distinct from
# `resume`, which is a person asking. Reusing that one made every mid-run feature advertise
# itself as "resuming" for its whole life, because the public read model reads exactly this
# field to say what a feature is doing -- and it would have been true only of the first claim.
CONTINUE = "continue"

# The ceiling on how many steps one run may take. Defined once, beside the column that carries
# it, and re-exported here because the queue is what enforces it -- the number used to be the
# literal 200 in three separate places, which is the shape a limit takes just before the three
# of them start disagreeing. A queue may be constructed with a different one; every queue in
# this deployment uses this.
MAX_STEPS = DEFAULT_FEATURE_STEP_LIMIT


@dataclass(frozen=True, slots=True)
class QueuedFeature:
    """One claimed queue entry: what to run, for whom, why, and on which attempt."""

    feature_id: str
    execution_mode: str
    # Which provider's stored credential this entry needs. Carried on the row rather than
    # read from feature state, because the dispatcher checks it before it loads any.
    agent_platform: str
    requested_by: str
    attempt: int
    max_attempts: int
    intent: str = START
    payload: dict[str, Any] = field(default_factory=dict)
    # The cost basis beside the platform, carried for the same reason it is: the worker that
    # runs a feature is not the request that accepted it. `high` for an entry written before
    # the field existed, which is what those features actually ran at.
    performance_tier: str = "high"
    # The pinned role map of a custom feature, and its provenance id. On the row rather than
    # read from feature state because the dispatcher decides which platforms' credentials the
    # entry needs before it loads any state. None for a tier feature.
    model_setup_snapshot: dict[str, Any] | None = None
    model_setup_id: str | None = None
    # How many steps this run has already completed, and how many it may. A claim advances
    # the feature by one step, so this is what bounds the durable loop between claims.
    steps: int = 0
    max_steps: int = MAX_STEPS

    @property
    def attempts_exhausted(self) -> bool:
        """Whether this claim is the last one this entry will ever get.

        Counted in *consecutive failures*, not in claims. `advance_to_next_step` returns the
        attempt on every step that completes, so a forty-step feature reaches its fortieth
        step on attempt 1 -- and three crashes in the same step still exhaust the budget,
        which is what it was always bounding.
        """
        return self.attempt >= self.max_attempts

    @property
    def steps_exhausted(self) -> bool:
        """Whether this run has taken every step this deployment allows it."""
        return self.steps >= self.max_steps


class FeatureExecutionQueue(Protocol):
    """Durable hand-off from "accepted" to "running", with at-least-once delivery."""

    async def enqueue(
        self,
        *,
        feature_id: str,
        execution_mode: str,
        agent_platform: str,
        requested_by: str,
        session: Any = None,
        intent: str = START,
        payload: dict[str, Any] | None = None,
        performance_tier: str = "high",
        model_setup_snapshot: dict[str, Any] | None = None,
        model_setup_id: str | None = None,
    ) -> None:
        """Record that one feature is waiting to be executed."""

    async def requeue(
        self,
        *,
        feature_id: str,
        execution_mode: str,
        agent_platform: str,
        intent: str,
        payload: dict[str, Any] | None = None,
        requested_by: str | None = None,
        performance_tier: str = "high",
        model_setup_snapshot: dict[str, Any] | None = None,
        model_setup_id: str | None = None,
    ) -> None:
        """Queue further work on a feature whose previous entry has already finished.

        Distinct from `enqueue`, which ignores a feature it has already seen -- that is what
        makes a resubmitted submission idempotent. A resume is not a duplicate submission: it
        is new work on a feature that has one settled entry, and it must reset the attempt
        budget rather than inherit one already spent.

        `requested_by` defaults to whoever the existing entry names. That identity is what
        the worker resolves stored credentials against, so carrying it forward is what lets a
        resume run with the same credentials the submission did.
        """

    async def claim(self, *, owner: str, lease_seconds: int) -> QueuedFeature | None:
        """Take the oldest entry that is waiting, or whose previous claim has lapsed."""

    async def renew(self, feature_id: str, *, owner: str, lease_seconds: int) -> None:
        """Extend one claim while its work is still in progress."""

    async def finish(self, feature_id: str, *, succeeded: bool, error: str | None = None) -> None:
        """Close one entry out, whether or not the work it described succeeded."""

    async def release(self, feature_id: str, *, error: str | None = None) -> None:
        """Return one entry to the queue so a later claim retries it."""

    async def defer(self, feature_id: str, *, error: str | None = None) -> None:
        """Return one entry to the queue without spending the attempt it just used."""

    async def advance_to_next_step(
        self, feature_id: str, *, intent: str = CONTINUE, payload: dict[str, Any] | None = None
    ) -> None:
        """Re-queue one entry whose step succeeded, so a later claim runs the next one."""

    async def active_intent(self, feature_id: str) -> str | None:
        """Return the queued/running intent for one feature, or ``None`` once it settled."""

    async def pending_count(self) -> int:
        """Return how many entries are waiting or claimed, for readiness and metrics."""


class InMemoryFeatureExecutionQueue:
    """The same contract without a database, for isolated applications and mock mode."""

    def __init__(self, *, max_steps: int = MAX_STEPS) -> None:
        """Start empty; entries live only as long as the process that accepted them."""
        self._entries: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._lock = asyncio.Lock()
        self._max_steps = max_steps

    async def enqueue(
        self,
        *,
        feature_id: str,
        execution_mode: str,
        agent_platform: str,
        requested_by: str,
        session: Any = None,
        intent: str = START,
        payload: dict[str, Any] | None = None,
        performance_tier: str = "high",
        model_setup_snapshot: dict[str, Any] | None = None,
        model_setup_id: str | None = None,
    ) -> None:
        """Append one waiting entry, ignoring a duplicate submission of the same feature."""
        async with self._lock:
            if feature_id in self._entries:
                return
            self._entries[feature_id] = {
                "execution_mode": execution_mode,
                "agent_platform": agent_platform,
                "performance_tier": performance_tier,
                "model_setup_snapshot": model_setup_snapshot,
                "model_setup_id": model_setup_id,
                "requested_by": requested_by,
                "intent": intent,
                "intent_payload": dict(payload or {}),
                "status": QUEUED,
                "attempt": 0,
                "max_attempts": 3,
                "steps": 0,
                "max_steps": self._max_steps,
                "lease_expires_at": None,
                "lease_owner": None,
            }
            self._order.append(feature_id)

    async def requeue(
        self,
        *,
        feature_id: str,
        execution_mode: str,
        agent_platform: str,
        intent: str,
        payload: dict[str, Any] | None = None,
        requested_by: str | None = None,
        performance_tier: str = "high",
        model_setup_snapshot: dict[str, Any] | None = None,
        model_setup_id: str | None = None,
    ) -> None:
        """Replace any settled entry with fresh work, resetting the attempt budget."""
        async with self._lock:
            existing = self._entries.get(feature_id)
            owner = requested_by or (str(existing["requested_by"]) if existing else "api")
            self._entries[feature_id] = {
                "execution_mode": execution_mode,
                "agent_platform": agent_platform,
                "performance_tier": performance_tier,
                "model_setup_snapshot": model_setup_snapshot,
                "model_setup_id": model_setup_id,
                "requested_by": owner,
                "intent": intent,
                "intent_payload": dict(payload or {}),
                "status": QUEUED,
                "attempt": 0,
                "max_attempts": 3,
                "steps": 0,
                "max_steps": self._max_steps,
                "lease_expires_at": None,
                "lease_owner": None,
            }
            if feature_id not in self._order:
                self._order.append(feature_id)

    async def claim(self, *, owner: str, lease_seconds: int) -> QueuedFeature | None:
        """Claim in submission order, reclaiming anything whose lease has lapsed."""
        now = datetime.now(UTC)
        async with self._lock:
            for feature_id in self._order:
                entry = self._entries.get(feature_id)
                if entry is None:
                    continue
                lapsed = (
                    entry["status"] == RUNNING
                    and entry["lease_expires_at"] is not None
                    and entry["lease_expires_at"] <= now
                )
                if entry["status"] != QUEUED and not lapsed:
                    continue
                entry["status"] = RUNNING
                entry["attempt"] += 1
                entry["lease_owner"] = owner
                entry["lease_expires_at"] = now + timedelta(seconds=lease_seconds)
                return QueuedFeature(
                    feature_id=feature_id,
                    execution_mode=str(entry["execution_mode"]),
                    agent_platform=str(entry["agent_platform"]),
                    performance_tier=str(entry.get("performance_tier", "high")),
                    model_setup_snapshot=entry.get("model_setup_snapshot"),
                    model_setup_id=entry.get("model_setup_id"),
                    requested_by=str(entry["requested_by"]),
                    attempt=int(entry["attempt"]),
                    max_attempts=int(entry["max_attempts"]),
                    intent=str(entry.get("intent", START)),
                    payload=dict(entry.get("intent_payload") or {}),
                    steps=int(entry.get("steps", 0)),
                    max_steps=int(entry.get("max_steps", self._max_steps)),
                )
        return None

    async def renew(self, feature_id: str, *, owner: str, lease_seconds: int) -> None:
        """Extend a claim this owner still holds."""
        async with self._lock:
            entry = self._entries.get(feature_id)
            if entry is not None and entry["lease_owner"] == owner:
                entry["lease_expires_at"] = datetime.now(UTC) + timedelta(seconds=lease_seconds)

    async def finish(self, feature_id: str, *, succeeded: bool, error: str | None = None) -> None:
        """Mark one entry terminal and stop it being claimed again."""
        async with self._lock:
            entry = self._entries.get(feature_id)
            if entry is None:
                return
            entry["status"] = SUCCEEDED if succeeded else FAILED
            entry["lease_owner"] = None
            entry["lease_expires_at"] = None
            entry["last_error"] = error

    async def release(self, feature_id: str, *, error: str | None = None) -> None:
        """Return one entry to the queue for another attempt."""
        async with self._lock:
            entry = self._entries.get(feature_id)
            if entry is None:
                return
            entry["status"] = QUEUED
            entry["lease_owner"] = None
            entry["lease_expires_at"] = None
            entry["last_error"] = error

    async def defer(self, feature_id: str, *, error: str | None = None) -> None:
        """Return one entry to the queue, giving back the attempt this claim consumed."""
        async with self._lock:
            entry = self._entries.get(feature_id)
            if entry is None:
                return
            entry["status"] = QUEUED
            entry["attempt"] = max(0, int(entry["attempt"]) - 1)
            entry["lease_owner"] = None
            entry["lease_expires_at"] = None
            entry["last_error"] = error

    async def advance_to_next_step(
        self, feature_id: str, *, intent: str = CONTINUE, payload: dict[str, Any] | None = None
    ) -> None:
        """Return one entry to the queue with its step recorded and its attempt given back."""
        async with self._lock:
            entry = self._entries.get(feature_id)
            if entry is None:
                return
            entry["status"] = QUEUED
            entry["attempt"] = 0
            entry["steps"] = int(entry.get("steps", 0)) + 1
            entry["intent"] = intent
            entry["intent_payload"] = dict(payload or {})
            entry["lease_owner"] = None
            entry["lease_expires_at"] = None
            entry["last_error"] = None

    async def active_intent(self, feature_id: str) -> str | None:
        """Read one active hand-off without exposing its answers or retry arguments."""
        async with self._lock:
            entry = self._entries.get(feature_id)
            if entry is None or entry["status"] not in {QUEUED, RUNNING}:
                return None
            return str(entry.get("intent", START))

    async def pending_count(self) -> int:
        """Count entries that have not reached a terminal state."""
        async with self._lock:
            return sum(1 for item in self._entries.values() if item["status"] in {QUEUED, RUNNING})


class DatabaseFeatureExecutionQueue:
    """The durable queue: one row per accepted feature, claimed under a row lock."""

    def __init__(self, database: Database, *, max_steps: int = MAX_STEPS) -> None:
        """Bind the database this queue lives in, and the step ceiling it writes onto entries."""
        self._database = database
        self._max_steps = max_steps

    async def enqueue(
        self,
        *,
        feature_id: str,
        execution_mode: str,
        agent_platform: str,
        requested_by: str,
        session: Any = None,
        intent: str = START,
        payload: dict[str, Any] | None = None,
        performance_tier: str = "high",
        model_setup_snapshot: dict[str, Any] | None = None,
        model_setup_id: str | None = None,
    ) -> None:
        """Add one waiting entry.

        `session` is how this becomes one transaction with the feature it describes. Committed
        separately, a crash between the two writes would leave a feature nothing ever runs.
        """
        entry = FeatureExecutionQueueModel(
            feature_id=feature_id,
            execution_mode=execution_mode,
            agent_platform=agent_platform,
            performance_tier=performance_tier,
            model_setup_snapshot=model_setup_snapshot,
            model_setup_id=model_setup_id,
            requested_by=requested_by,
            intent=intent,
            intent_payload=dict(payload or {}),
            status=QUEUED,
            attempt=0,
            max_attempts=3,
            max_steps=self._max_steps,
            queued_at=datetime.now(UTC),
        )
        if session is not None:
            session.add(entry)
            return
        async with self._database.session() as own_session:
            own_session.add(entry)
            await own_session.commit()

    async def requeue(
        self,
        *,
        feature_id: str,
        execution_mode: str,
        agent_platform: str,
        intent: str,
        payload: dict[str, Any] | None = None,
        requested_by: str | None = None,
        performance_tier: str = "high",
        model_setup_snapshot: dict[str, Any] | None = None,
        model_setup_id: str | None = None,
    ) -> None:
        """Reopen the settled entry for this feature with fresh work and a fresh budget.

        The primary key is the feature, so a feature has exactly one entry over its whole
        life and this rewrites it rather than adding a second. `queued_at` moves forward so
        the reopened entry takes its turn behind whatever is already waiting instead of
        jumping to the front of a queue it joined hours ago.
        """
        now = datetime.now(UTC)
        async with self._database.session() as session:
            entry = await session.get(FeatureExecutionQueueModel, feature_id)
            if entry is None:
                session.add(
                    FeatureExecutionQueueModel(
                        feature_id=feature_id,
                        execution_mode=execution_mode,
                        agent_platform=agent_platform,
                        performance_tier=performance_tier,
                        model_setup_snapshot=model_setup_snapshot,
                        model_setup_id=model_setup_id,
                        requested_by=requested_by or "api",
                        intent=intent,
                        intent_payload=dict(payload or {}),
                        status=QUEUED,
                        attempt=0,
                        max_attempts=3,
                        max_steps=self._max_steps,
                        queued_at=now,
                    )
                )
                await session.commit()
                return
            entry.execution_mode = execution_mode
            entry.agent_platform = agent_platform
            entry.performance_tier = performance_tier
            entry.model_setup_snapshot = model_setup_snapshot
            entry.model_setup_id = model_setup_id
            entry.requested_by = requested_by or entry.requested_by
            entry.intent = intent
            entry.intent_payload = dict(payload or {})
            entry.status = QUEUED
            entry.attempt = 0
            # A person asked for this, so it is a new run: the step ceiling starts again, as
            # `started_at` below does. Carrying either forward would let one feature's history
            # refuse the work somebody just requested.
            entry.steps = 0
            entry.lease_owner = None
            entry.lease_expires_at = None
            entry.queued_at = now
            entry.started_at = None
            entry.finished_at = None
            entry.last_error = None
            await session.commit()

    async def claim(self, *, owner: str, lease_seconds: int) -> QueuedFeature | None:
        """Claim the oldest waiting or lapsed entry, under a row lock.

        `SELECT ... FOR UPDATE SKIP LOCKED` is what lets several workers drain the same queue
        without two of them running the same feature. SQLite has neither clause and does not
        need them: it serialises writers already.
        """
        now = datetime.now(UTC)
        statement = (
            select(FeatureExecutionQueueModel)
            .where(
                or_(
                    FeatureExecutionQueueModel.status == QUEUED,
                    (FeatureExecutionQueueModel.status == RUNNING)
                    & (FeatureExecutionQueueModel.lease_expires_at <= now),
                )
            )
            .order_by(FeatureExecutionQueueModel.queued_at, FeatureExecutionQueueModel.feature_id)
            .limit(1)
        )
        if self._database.engine.dialect.name != "sqlite":
            statement = statement.with_for_update(skip_locked=True)
        async with self._database.session() as session:
            entry = await session.scalar(statement)
            if entry is None:
                return None
            entry.status = RUNNING
            entry.attempt += 1
            entry.lease_owner = owner
            entry.lease_expires_at = now + timedelta(seconds=lease_seconds)
            entry.started_at = entry.started_at or now
            claimed = QueuedFeature(
                feature_id=entry.feature_id,
                execution_mode=entry.execution_mode,
                agent_platform=entry.agent_platform,
                performance_tier=entry.performance_tier,
                model_setup_snapshot=entry.model_setup_snapshot,
                model_setup_id=entry.model_setup_id,
                requested_by=entry.requested_by,
                attempt=entry.attempt,
                max_attempts=entry.max_attempts,
                intent=entry.intent,
                payload=dict(entry.intent_payload or {}),
                steps=entry.steps,
                max_steps=entry.max_steps,
            )
            await session.commit()
        return claimed

    async def renew(self, feature_id: str, *, owner: str, lease_seconds: int) -> None:
        """Extend a claim this owner still holds, and do nothing if it has been taken over."""
        async with self._database.session() as session:
            entry = await session.get(FeatureExecutionQueueModel, feature_id)
            if entry is None or entry.lease_owner != owner:
                return
            entry.lease_expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)
            await session.commit()

    async def finish(self, feature_id: str, *, succeeded: bool, error: str | None = None) -> None:
        """Mark one entry terminal so it is never claimed again."""
        async with self._database.session() as session:
            entry = await session.get(FeatureExecutionQueueModel, feature_id)
            if entry is None:
                return
            entry.status = SUCCEEDED if succeeded else FAILED
            entry.lease_owner = None
            entry.lease_expires_at = None
            entry.finished_at = datetime.now(UTC)
            entry.last_error = _short(error)
            await session.commit()

    async def release(self, feature_id: str, *, error: str | None = None) -> None:
        """Return one entry to the queue so a later claim retries it."""
        async with self._database.session() as session:
            entry = await session.get(FeatureExecutionQueueModel, feature_id)
            if entry is None:
                return
            entry.status = QUEUED
            entry.lease_owner = None
            entry.lease_expires_at = None
            entry.last_error = _short(error)
            await session.commit()

    async def defer(self, feature_id: str, *, error: str | None = None) -> None:
        """Return one entry to the queue, giving back the attempt this claim consumed."""
        async with self._database.session() as session:
            entry = await session.get(FeatureExecutionQueueModel, feature_id)
            if entry is None:
                return
            entry.status = QUEUED
            entry.attempt = max(0, entry.attempt - 1)
            entry.lease_owner = None
            entry.lease_expires_at = None
            entry.last_error = _short(error)
            await session.commit()

    async def advance_to_next_step(
        self, feature_id: str, *, intent: str = CONTINUE, payload: dict[str, Any] | None = None
    ) -> None:
        """Reopen this entry for the next step of a run that is already in progress.

        Deliberately not `requeue`, which exists for work a person newly asked for. Three
        things differ, and each of them is load-bearing:

        * the attempt count goes back to zero, because a step that completed is progress and
          progress must not spend a budget reserved for crashes;
        * `started_at` is *kept*, because the feature runtime ceiling is clocked from it. A
          re-queue that cleared it would restart that clock on every step and silently retire
          the six-hour backstop on a feature that never stops making small amounts of
          progress -- which is the exact failure it was added for;
        * `steps` goes up, which is what bounds the loop this method creates.

        `queued_at` moves forward, as it does for any re-queue, so a feature with forty steps
        takes its turn behind whatever is waiting instead of monopolising the workers.
        """
        async with self._database.session() as session:
            entry = await session.get(FeatureExecutionQueueModel, feature_id)
            if entry is None:
                return
            entry.status = QUEUED
            entry.attempt = 0
            entry.steps += 1
            entry.intent = intent
            entry.intent_payload = dict(payload or {})
            entry.lease_owner = None
            entry.lease_expires_at = None
            entry.queued_at = datetime.now(UTC)
            entry.last_error = None
            await session.commit()

    async def active_intent(self, feature_id: str) -> str | None:
        """Read one active hand-off without loading its potentially sensitive payload."""
        async with self._database.session() as session:
            intent = await session.scalar(
                select(FeatureExecutionQueueModel.intent).where(
                    FeatureExecutionQueueModel.feature_id == feature_id,
                    FeatureExecutionQueueModel.status.in_((QUEUED, RUNNING)),
                )
            )
            return str(intent) if intent is not None else None

    async def pending_count(self) -> int:
        """Count entries that have not reached a terminal state."""
        async with self._database.session() as session:
            rows = await session.scalars(
                select(FeatureExecutionQueueModel.feature_id).where(
                    FeatureExecutionQueueModel.status.in_((QUEUED, RUNNING))
                )
            )
            return len(list(rows))


class QueuedFeatureExecutor(Protocol):
    """What the dispatcher does with a claimed entry: run the feature it names."""

    async def execute_queued(
        self,
        feature_id: str,
        *,
        credentials: RequestScopedCredentials,
        intent: str = START,
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
        """Record that one queued feature cannot be executed, and why."""

    async def fail_exhausted_fault(
        self, feature_id: str, *, intent: str, error: Exception, attempts: int
    ) -> None:
        """Record a retryable fault as one feature's stop, once no attempt is left to spend.

        Whichever fault it was. Two kinds reach here -- the provider or the Git remote not
        answering, and the worker volume being too full -- and the record they need is the same
        one: the feature's terminal state, classified from the error itself so the diagnosis
        names the condition rather than this queue's opinion of it.

        `attempts` is how many this entry spent, which only the queue knows. It belongs in the
        record because it is the evidence that the platform behaved correctly and then gave up
        on weather, rather than never having tried.
        """

    async def stop_unprogressing_run(self, feature_id: str, *, steps: int) -> None:
        """Record that one feature took every step it was allowed and did not finish."""

    async def feature_owes_another_step(self, feature_id: str, *, intent: str = CONTINUE) -> bool:
        """Whether this feature has a next step a claim of this intent may take unasked."""

    async def feature_run_succeeded(self, feature_id: str) -> bool:
        """Whether the feature this entry just ran ended anywhere other than failed."""


@dataclass
class FeatureQueueDispatcher:
    """Workers that turn queued features into running ones.

    Deliberately a poller rather than a listener. A queue this size is drained in a query the
    database answers from an index, and a listener per worker would hold a connection open to
    buy latency nobody is waiting on -- the caller has already had their answer.
    """

    queue: FeatureExecutionQueue
    executor: QueuedFeatureExecutor
    credentials_for: Callable[[str], Awaitable[RequestScopedCredentials]]
    workers: int = 2
    lease_seconds: int = 120
    poll_seconds: float = 1.0
    # The floor on how often a claim is renewed, so a short lease cannot turn into a tight
    # loop against the database. A field rather than a literal because it is the one number
    # that decides whether renewal behaviour is observable in a test at all.
    min_renewal_seconds: float = 5.0
    _tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False, repr=False)
    _dispatches: set[asyncio.Task[None]] = field(default_factory=set, init=False, repr=False)
    _wake: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)

    def notify(self) -> None:
        """Tell the workers something was enqueued, so they do not wait out a poll interval."""
        self._wake.set()

    def schedule(self) -> None:
        """Run one claim now, in the background, without a standing worker fleet.

        This is how an application with no lifespan -- an isolated test app, or mock mode --
        still executes what it accepts. The task ends when the claim does, so nothing is left
        running after the work is finished.
        """
        task = asyncio.create_task(self._dispatch(), name="feature-queue-dispatch")
        self._dispatches.add(task)
        task.add_done_callback(self._dispatches.discard)

    async def wait_for_idle(self) -> None:
        """Wait until nothing is queued or in flight.

        The deterministic counterpart to `schedule`: a caller that wants to observe the
        result of what it submitted waits here rather than sleeping. Executing a feature can
        enqueue nothing further, so this terminates once the queue is empty.
        """
        while True:
            scheduled = list(self._dispatches)
            if scheduled:
                await asyncio.gather(*scheduled, return_exceptions=True)
            if await self.queue.pending_count() == 0:
                return
            if not await self.run_once():
                return

    async def _dispatch(self) -> None:
        """Drain what is claimable, keeping a failure from becoming an unretrieved exception.

        Draining rather than claiming once, because a claim is one step now. An application
        with no worker fleet -- an isolated test app, or mock mode -- would otherwise advance
        each feature by a single step per submission and stop with the work half done.
        """
        try:
            await self.drain()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - already recorded against the feature and the entry
            _LOGGER.exception("feature_queue_dispatch_error")

    def start(self) -> None:
        """Start the workers. Idempotent, so a second call does not double the fleet."""
        if self._tasks:
            return
        for index in range(self.workers):
            task = asyncio.create_task(self._work(), name=f"feature-queue-worker-{index}")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def stop(self) -> None:
        """Cancel the workers and wait for them, so shutdown does not leak a running claim."""
        tasks = [*self._tasks, *self._dispatches]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._dispatches.clear()

    async def drain(self) -> int:
        """Run every entry that is currently claimable, and return how many were run.

        This is the deterministic path: tests and mock mode use it instead of waiting for a
        poll, and it exercises exactly the code the workers do.
        """
        processed = 0
        while await self.run_once():
            processed += 1
        return processed

    async def run_once(self) -> bool:
        """Claim and execute at most one entry. Returns whether there was one."""
        owner = f"worker-{uuid4().hex[:12]}"
        entry = await self.queue.claim(owner=owner, lease_seconds=self.lease_seconds)
        if entry is None:
            return False
        renewal = asyncio.create_task(self._renew(entry.feature_id, owner))
        try:
            credentials = await self.credentials_for(entry.requested_by)
            missing = _missing_providers(
                entry.execution_mode,
                entry.agent_platform,
                credentials,
                model_setup_snapshot=entry.model_setup_snapshot,
            )
            if missing:
                # Not retried. A live feature cannot run without the keys its agents need,
                # and re-claiming it every two minutes would burn the queue rather than tell
                # anybody. The feature says so, and configuring the credentials and resuming
                # is the way forward.
                reason = (
                    "This feature runs against real repositories and needs provider "
                    f"credentials configured for {entry.requested_by}: "
                    f"{', '.join(missing)}. Configure them in Settings, then resume the "
                    "feature."
                )
                await self.executor.fail_queued(
                    entry.feature_id,
                    event="feature_credentials_missing",
                    reason=reason,
                    error_type=FeatureFailureClassification.PROVIDER_CREDENTIALS_MISSING.value,
                )
                await self.queue.finish(entry.feature_id, succeeded=False, error=reason)
                return True
            await self.executor.execute_queued(
                entry.feature_id,
                credentials=credentials,
                intent=entry.intent,
                payload=entry.payload,
            )
        except asyncio.CancelledError:
            # Shutdown, not failure. Leave the claim to lapse so another process runs it.
            raise
        except FeatureExecutionBusy:
            # The work is happening; this claim simply arrived on top of it. Put the entry
            # back with the attempt returned, so a lease that lapsed under a long run cannot
            # spend the feature's whole budget colliding with itself.
            _LOGGER.info(
                "queued_feature_already_executing",
                feature_id=entry.feature_id,
                attempt=entry.attempt,
            )
            await self.queue.defer(entry.feature_id, error="another worker holds this feature")
            return True
        except FeatureProviderFault as fault:
            # The provider failed, which says nothing about this feature. Spend an attempt --
            # that is what they are reserved for -- and only record the feature as stopped
            # once there is genuinely none left. Before this, the executor recorded the stop
            # itself and returned, so the entry closed `succeeded` with its remaining
            # attempts intact and a person had to press resume to do what the queue already
            # had budget to do.
            detail = f"{type(fault.cause).__name__} on attempt {entry.attempt}"
            _LOGGER.warning(
                "queued_feature_provider_fault",
                feature_id=entry.feature_id,
                attempt=entry.attempt,
                max_attempts=entry.max_attempts,
                intent=entry.intent,
                error_type=type(fault.cause).__name__,
            )
            if entry.attempts_exhausted:
                await self.executor.fail_exhausted_fault(
                    entry.feature_id,
                    intent=entry.intent,
                    error=fault.cause,
                    attempts=entry.attempt,
                )
                await self.queue.finish(entry.feature_id, succeeded=False, error=detail)
            else:
                await self.queue.release(entry.feature_id, error=detail)
            return True
        except FeatureWorkspaceCapacityUnavailable as unavailable:
            # The volume, not the feature. Handled exactly as a provider fault is, because
            # the two are the same kind of thing: an attempt this entry has budget for did
            # not start, and the condition may well be gone by the next one. Run 193's whole
            # defect was that this path recorded the feature as needing a human and returned
            # normally, so its status contradicted the loop that was still retrying it.
            detail = f"workspace capacity on attempt {entry.attempt}"
            _LOGGER.warning(
                "queued_feature_workspace_capacity_unavailable",
                feature_id=entry.feature_id,
                attempt=entry.attempt,
                max_attempts=entry.max_attempts,
                intent=entry.intent,
            )
            if entry.attempts_exhausted:
                await self.executor.fail_exhausted_fault(
                    entry.feature_id,
                    intent=entry.intent,
                    error=unavailable.cause,
                    attempts=entry.attempt,
                )
                await self.queue.finish(entry.feature_id, succeeded=False, error=detail)
            else:
                await self.queue.release(entry.feature_id, error=detail)
            return True
        except Exception as error:  # noqa: BLE001 - the entry decides what happens next
            _LOGGER.error(
                "queued_feature_execution_failed",
                feature_id=entry.feature_id,
                attempt=entry.attempt,
                error_type=type(error).__name__,
            )
            detail = f"{type(error).__name__} on attempt {entry.attempt}"
            if entry.attempts_exhausted:
                # The error type goes in the reason. Without it this said only "could not
                # start after 3 attempts", which is the one thing the reader already knew.
                await self.executor.fail_queued(
                    entry.feature_id,
                    event="feature_execution_failed",
                    reason=(
                        f"The platform could not start this feature after {entry.attempt} "
                        f"attempts. The last one ended in {type(error).__name__}."
                    ),
                    # The classification, not the type name. `execute_queued` raising
                    # `DBAPIError` says the platform failed; recording that type name as the
                    # feature's root classification presented it as a finding about the
                    # repository, and asked a person to decide about their own code.
                    error_type=classification_of(error).value,
                )
                await self.queue.finish(entry.feature_id, succeeded=False, error=detail)
            else:
                await self.queue.release(entry.feature_id, error=detail)
            return True
        finally:
            renewal.cancel()
            await asyncio.gather(renewal, return_exceptions=True)
        if await self.executor.feature_owes_another_step(entry.feature_id, intent=entry.intent):
            if entry.steps_exhausted:
                # The loop this dispatcher runs is durable, so something has to be able to
                # end it. A feature still naming a next step after this many of them is not
                # progressing towards one, and every further claim would spend another
                # provider call to establish that again.
                _LOGGER.warning(
                    "queued_feature_step_budget_exhausted",
                    feature_id=entry.feature_id,
                    steps=entry.steps,
                    max_steps=entry.max_steps,
                )
                await self.executor.stop_unprogressing_run(entry.feature_id, steps=entry.max_steps)
                await self.queue.finish(
                    entry.feature_id,
                    succeeded=False,
                    error=f"still owed a step after {entry.max_steps} of them",
                )
                return True
            # One step done and more owed. The entry goes back on the queue rather than being
            # closed, which is what makes a crash cost one step instead of a feature -- and it
            # goes back behind whatever is already waiting, so a forty-step feature cannot
            # hold both workers against a one-step feature queued after it.
            await self.queue.advance_to_next_step(entry.feature_id)
            return True
        # What the feature ended as, not merely that `execute_queued` returned. It returns
        # normally after recording a feature as failed, so the entry closed `succeeded` --
        # 73 succeeded entries against 95 features needing a human. A flag that reads as
        # success for a run that failed is worse than no flag: it is the number an operator
        # would check first.
        await self.queue.finish(
            entry.feature_id,
            succeeded=await self.executor.feature_run_succeeded(entry.feature_id),
        )
        return True

    async def _work(self) -> None:
        """Drain the queue, then wait for a nudge or the poll interval."""
        while True:
            try:
                worked = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a worker must outlive one bad claim
                _LOGGER.exception("feature_queue_worker_error")
                worked = False
            if worked:
                continue
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                continue

    async def _renew(self, feature_id: str, owner: str) -> None:
        """Hold a claim while its feature runs, riding out a database that blinks.

        A live feature takes minutes, and a lease long enough to cover the slowest one would
        also be how long a crash goes unnoticed. Renewing at a third of the lease keeps both
        numbers small.

        Every failure is retried rather than ending the loop. Now that resuming and retrying
        are queued, this claim is what proves a long run is still alive, and a loop that quit
        on one transient error would let the entry lapse under a run that was working -- the
        same defect that cost AB-Feature-108 seventy minutes on the action lease. Another
        worker claiming it is not a disaster (the run lock defers the duplicate) but it burns
        the entry's attempts for no reason.
        """
        interval = max(self.lease_seconds / 3, self.min_renewal_seconds)
        retry_interval = max(self.lease_seconds / 10, self.min_renewal_seconds / 5)
        while True:
            await asyncio.sleep(interval)
            try:
                await self.queue.renew(feature_id, owner=owner, lease_seconds=self.lease_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                _LOGGER.warning(
                    "feature_queue_lease_renewal_retried",
                    feature_id=feature_id,
                    error_type=type(error).__name__,
                )
                interval = retry_interval
                continue
            interval = max(self.lease_seconds / 3, self.min_renewal_seconds)


def _missing_providers(
    execution_mode: str,
    agent_platform: str,
    credentials: RequestScopedCredentials,
    *,
    model_setup_snapshot: dict[str, Any] | None = None,
) -> list[str]:
    """Return the credentials this entry needs and does not have.

    The model credentials are the ones this feature's platforms use and not the other's. A
    feature on Claude held up for a missing OpenAI key would be refused over a credential none
    of its agents would ever reach for. A custom feature needs every platform its pinned
    role map names -- a mixed setup legitimately needs both.
    """
    if execution_mode != "live":
        # Mock execution reaches no provider and no repository, so it needs nothing.
        return []
    platforms = [agent_platform]
    if model_setup_snapshot is not None:
        roles = model_setup_snapshot.get("roles")
        if isinstance(roles, dict):
            named = [
                str(entry.get("platform"))
                for entry in roles.values()
                if isinstance(entry, dict) and entry.get("platform")
            ]
            if named:
                platforms = list(dict.fromkeys(named))
    missing = []
    for platform in platforms:
        if platform == "anthropic":
            if not credentials.anthropic_api_key:
                missing.append("Anthropic")
        elif not credentials.openai_api_key:
            missing.append("OpenAI")
    if not credentials.github_token:
        missing.append("GitHub")
    return missing


def _short(value: str | None) -> str | None:
    """Keep a queue row small; the feature's own failure summary holds the detail."""
    return None if value is None else value[:500]


__all__ = [
    "ANSWER_DESIGN_VERDICT",
    "CONTINUE",
    "PUBLISH",
    "FAILED",
    "FeatureExecutionBusy",
    "QUEUED",
    "RUNNING",
    "SUCCEEDED",
    "DatabaseFeatureExecutionQueue",
    "FeatureExecutionQueue",
    "FeatureProviderFault",
    "FeatureQueueDispatcher",
    "InMemoryFeatureExecutionQueue",
    "QueuedFeature",
    "QueuedFeatureExecutor",
]
