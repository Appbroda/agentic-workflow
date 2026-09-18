"""Cooperative cancellation tokens backed by durable state and optional Redis signaling."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Sequence
from typing import Any, Protocol, cast

from state.enums import FeatureWorkflowStatus, WorkflowStatus
from storage.db import Database
from storage.models import FeatureWorkflowModel, WorkflowModel


class CancellationRequested(RuntimeError):
    """Raised at a safe boundary when a caller must stop further side effects."""


class CancellationToken(Protocol):
    """A restart-safe cooperative cancellation signal for long-running live operations."""

    async def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested by any configured source."""

    async def raise_if_cancelled(self) -> None:
        """Raise a typed cancellation exception at a safe side-effect boundary."""

    async def wait_cancelled(self) -> None:
        """Wait until cancellation becomes visible without returning spuriously."""


class MockCancellationToken:
    """Deterministic cancellation source for unit tests and isolated mock workflows."""

    def __init__(self, *, cancelled: bool = False) -> None:
        self._event = asyncio.Event()
        if cancelled:
            self._event.set()

    def cancel(self) -> None:
        """Signal cancellation immediately for a controlled test or local caller."""
        self._event.set()

    async def is_cancelled(self) -> bool:
        """Return the test event state."""
        return self._event.is_set()

    async def raise_if_cancelled(self) -> None:
        """Raise the shared typed cancellation error if the event is set."""
        if await self.is_cancelled():
            raise CancellationRequested("operation cancellation requested")

    async def wait_cancelled(self) -> None:
        """Wait for the test event to be signalled."""
        await self._event.wait()


class DatabaseCancellationToken:
    """Read durable cancellation state so a request survives API and worker restarts."""

    def __init__(
        self,
        database: Database,
        *,
        workflow_id: str,
        feature_id: str | None = None,
        poll_seconds: float = 0.1,
    ) -> None:
        if poll_seconds <= 0:
            msg = "poll_seconds must be positive"
            raise ValueError(msg)
        self._database = database
        self._workflow_id = workflow_id
        self._feature_id = feature_id
        self._poll_seconds = poll_seconds

    async def is_cancelled(self) -> bool:
        """Read the current durable state without retaining an ORM object across awaits."""
        async with self._database.session() as session:
            if self._feature_id is not None:
                feature = await session.get(FeatureWorkflowModel, self._feature_id)
                if feature is None:
                    return False
                return bool(feature.state_json.get("cancellation_requested")) or feature.status in {
                    FeatureWorkflowStatus.CANCELLING,
                    FeatureWorkflowStatus.CANCELLED,
                    FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
                }
            workflow = await session.get(WorkflowModel, self._workflow_id)
            if workflow is None:
                return False
            return workflow.status in {
                WorkflowStatus.CANCELLING,
                WorkflowStatus.CANCELLED,
                WorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
            }

    async def raise_if_cancelled(self) -> None:
        """Raise as soon as durable cancellation is visible."""
        if await self.is_cancelled():
            raise CancellationRequested("operation cancellation requested")

    async def wait_cancelled(self) -> None:
        """Poll durable state at a bounded interval when Redis notification is unavailable."""
        while not await self.is_cancelled():
            await asyncio.sleep(self._poll_seconds)


class RedisCancellationToken:
    """Use a short-lived Redis key for prompt in-process cancellation propagation."""

    def __init__(
        self,
        redis: Any,
        *,
        scope: str,
        identifier: str,
        namespace: str = "ai-platform:cancellation",
        poll_seconds: float = 0.05,
    ) -> None:
        if poll_seconds <= 0:
            msg = "poll_seconds must be positive"
            raise ValueError(msg)
        self._redis = redis
        self._key = f"{namespace}:{scope}:{identifier}"
        self._poll_seconds = poll_seconds

    @property
    def key(self) -> str:
        """Expose the non-secret signal name so the control plane can set it on cancellation."""
        return self._key

    async def is_cancelled(self) -> bool:
        """Read the optional signal; missing Redis methods simply provide no immediate signal."""
        getter = getattr(self._redis, "get", None)
        if getter is None:
            return False
        return await getter(self._key) is not None

    async def raise_if_cancelled(self) -> None:
        """Raise if the local worker sees the short-lived signal."""
        if await self.is_cancelled():
            raise CancellationRequested("operation cancellation requested")

    async def wait_cancelled(self) -> None:
        """Poll the Redis signal without requiring a long-lived pub/sub connection."""
        while not await self.is_cancelled():
            await asyncio.sleep(self._poll_seconds)


class CompositeCancellationToken:
    """Treat cancellation from any durable or immediate source as authoritative."""

    def __init__(self, tokens: Sequence[CancellationToken]) -> None:
        if not tokens:
            msg = "CompositeCancellationToken requires at least one token"
            raise ValueError(msg)
        self._tokens = tuple(tokens)

    async def is_cancelled(self) -> bool:
        """Query all sources so a database restart never loses a request signal."""
        return any(await asyncio.gather(*(token.is_cancelled() for token in self._tokens)))

    async def raise_if_cancelled(self) -> None:
        """Raise the shared signal when either durable or Redis state requests a stop."""
        if await self.is_cancelled():
            raise CancellationRequested("operation cancellation requested")

    async def wait_cancelled(self) -> None:
        """Return when the first configured cancellation source becomes visible."""
        tasks = [asyncio.create_task(token.wait_cancelled()) for token in self._tokens]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def signal_redis_cancellation(
    redis: Any,
    *,
    scope: str,
    identifier: str,
    ttl_seconds: int = 3600,
    namespace: str = "ai-platform:cancellation",
) -> None:
    """Publish a short-lived immediate signal after durable state was already committed."""
    setter = getattr(redis, "set", None)
    if setter is None:
        return
    await setter(f"{namespace}:{scope}:{identifier}", "requested", ex=ttl_seconds)


async def await_cancellable[T](awaitable: Awaitable[T], token: CancellationToken) -> T:
    """Stop waiting for a cancellable provider call and prevent downstream side effects."""
    await token.raise_if_cancelled()
    operation_task: asyncio.Future[T] = asyncio.ensure_future(awaitable)
    cancellation_task: asyncio.Task[None] = asyncio.create_task(token.wait_cancelled())
    try:
        done, _ = await asyncio.wait(
            {
                cast(asyncio.Future[object], operation_task),
                cast(asyncio.Future[object], cancellation_task),
            },
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation_task in done:
            return await operation_task
        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
        raise CancellationRequested("operation cancellation requested")
    finally:
        cancellation_task.cancel()
        await asyncio.gather(cancellation_task, return_exceptions=True)


__all__ = [
    "CancellationRequested",
    "CancellationToken",
    "CompositeCancellationToken",
    "DatabaseCancellationToken",
    "MockCancellationToken",
    "RedisCancellationToken",
    "signal_redis_cancellation",
    "await_cancellable",
]
