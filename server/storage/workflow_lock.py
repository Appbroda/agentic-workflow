"""The distributed lock that serializes one feature's lifecycle operations.

Redis-backed and ownership-checked: a holder proves its lease for as long as it runs, and a
holder that cannot prove it is stopped rather than allowed to keep writing. That is what makes
it safe for two API processes to accept work for the same feature at the same time.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from time import monotonic
from typing import Any, Protocol
from uuid import uuid4

from api.control_plane import WorkflowBusyError, WorkflowConflictError


class RedisClient(Protocol):
    """Minimal asynchronous Redis interface needed for distributed idempotency locks."""

    async def set(self, name: str, value: str, *, nx: bool, ex: int) -> bool | None:
        """Set a lock key only when it does not already exist."""

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> Any:
        """Run the ownership-checked lock release script."""

    async def exists(self, name: str) -> int:
        """Return whether a lock key currently has an owner."""


class WorkflowLock(Protocol):
    """Coordinate one workflow lifecycle operation across API processes."""

    def hold(self, key: str) -> AbstractAsyncContextManager[None]:
        """Yield while the caller owns a keyed distributed lock."""


class NoopWorkflowLock:
    """Use database uniqueness alone for isolated tests and single-process development."""

    @asynccontextmanager
    async def hold(self, _key: str) -> AsyncIterator[None]:
        """Yield immediately without acquiring an external lock."""
        yield


class RedisWorkflowLock:
    """Use ownership-checked Redis keys to serialize idempotent workflow commands."""

    _RELEASE_SCRIPT = (
        "if redis.call('get', KEYS[1]) == ARGV[1] then "
        "return redis.call('del', KEYS[1]) else return 0 end"
    )
    _RENEW_SCRIPT = (
        "if redis.call('get', KEYS[1]) == ARGV[1] then "
        "return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end"
    )

    def __init__(
        self,
        redis: RedisClient,
        *,
        namespace: str = "ai-platform:workflow-lock",
        timeout_seconds: float = 10.0,
        lease_seconds: int = 1800,
    ) -> None:
        """Configure finite acquisition and lease durations for workflow operations."""
        self._redis = redis
        self._namespace = namespace
        self._timeout_seconds = timeout_seconds
        self._lease_seconds = lease_seconds

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        """Acquire one lock or return a retryable conflict without deleting another owner's key."""
        token = str(uuid4())
        redis_key = f"{self._namespace}:{key}"
        deadline = monotonic() + self._timeout_seconds
        while True:
            acquired = await self._redis.set(redis_key, token, nx=True, ex=self._lease_seconds)
            if acquired:
                break
            if monotonic() >= deadline:
                msg = "workflow operation is already in progress; retry shortly"
                # The one conflict in this file that resending fixes: somebody else holds the
                # lock now and will not for ever. Every other conflict raised here is a lease
                # that was lost mid-operation, which waiting does not repair.
                raise WorkflowBusyError(msg)
            await asyncio.sleep(0.05)
        owner_task = asyncio.current_task()
        lease_failure: asyncio.Future[Exception] = asyncio.get_running_loop().create_future()

        async def renew_or_abort_owner() -> None:
            """Stop the protected task as soon as its lease can no longer be proven."""
            try:
                await self._renew(redis_key, token)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if not lease_failure.done():
                    lease_failure.set_result(error)
                if owner_task is not None:
                    owner_task.cancel()

        def lease_error() -> WorkflowConflictError:
            """Normalize Redis renewal faults to the lifecycle conflict exposed by the lock."""
            error = lease_failure.result()
            if isinstance(error, WorkflowConflictError):
                return WorkflowConflictError(str(error))
            return WorkflowConflictError(
                "workflow lock lease could not be renewed during execution"
            )

        try:
            renewal_task = asyncio.create_task(renew_or_abort_owner())
            try:
                try:
                    yield
                except asyncio.CancelledError:
                    if lease_failure.done():
                        error = lease_error()
                        raise error from lease_failure.result()
                    raise
                if lease_failure.done():
                    error = lease_error()
                    raise error from lease_failure.result()
            finally:
                renewal_task.cancel()
                await asyncio.gather(renewal_task, return_exceptions=True)
        finally:
            await self._redis.eval(self._RELEASE_SCRIPT, 1, redis_key, token)

    async def _renew(self, redis_key: str, token: str) -> None:
        """Renew a long-running owner lease without ever extending a stolen lock."""
        interval = max(1.0, self._lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            renewed = await self._redis.eval(
                self._RENEW_SCRIPT, 1, redis_key, token, str(self._lease_seconds)
            )
            if not renewed:
                msg = "workflow lock ownership was lost during execution"
                raise WorkflowConflictError(msg)

    async def is_held(self, key: str) -> bool:
        """Report a live run lease without acquiring or disturbing its ownership token."""
        return bool(await self._redis.exists(f"{self._namespace}:{key}"))


__all__ = [
    "NoopWorkflowLock",
    "RedisClient",
    "RedisWorkflowLock",
    "WorkflowLock",
]
