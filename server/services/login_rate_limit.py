"""How many times somebody may guess a password, and where that count is kept.

Two counters per attempt, not one: the source address and the email being guessed. An
attacker with a botnet defeats a per-address limit; an attacker spraying one password across
a leaked address list defeats a per-email limit. Neither alone is worth much.

The count lives in Redis, which this deployment already requires. An in-process counter would
be per-worker, and `FEATURE_QUEUE_WORKERS` is two by default -- so the real limit would be
the configured one times however many processes happen to be serving, which is not a limit.

Deliberately rate limiting and not account lockout. Lockout stops an attacker and hands
anybody who knows an email address a denial of service against that person, and there is no
existing pattern in this repository that decides otherwise. Recorded in
`docs/AUTHENTICATION_AND_WORKSPACES.md` as a decision rather than left implicit here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import structlog

_logger = structlog.get_logger("api.login_rate_limit")


class RateLimitBackend(Protocol):
    """The two operations a fixed-window counter needs."""

    async def incr(self, name: str) -> int:
        """Increment one counter and return its new value."""

    async def expire(self, name: str, seconds: int) -> Any:
        """Give one counter a lifetime, so the window rolls forward."""


@dataclass(frozen=True, slots=True)
class RateLimitVerdict:
    """Whether an attempt may proceed, and how long to wait if not."""

    allowed: bool
    retry_after_seconds: int


class LoginRateLimiter:
    """Count login attempts per source address and per email over a fixed window.

    A fixed window rather than a sliding one on purpose: a sliding window needs a sorted set
    per key and a trim on every read, and the thing being bounded here is a human-scale
    guessing rate. The worst case a fixed window allows is twice the configured attempts
    across a window boundary, which does not change what this defends against.
    """

    def __init__(
        self,
        backend: RateLimitBackend | None,
        *,
        attempts: int,
        window_seconds: int,
    ) -> None:
        """Bind the counter store and the budget.

        `backend` may be `None`, which disables the limit. That is the honest behaviour for
        an application assembled without Redis -- an isolated test app, or a mock deployment
        -- rather than refusing every login because a counter has nowhere to live.
        """
        self._backend = backend
        self._attempts = attempts
        self._window_seconds = window_seconds

    async def check(self, *, source: str | None, subject: str) -> RateLimitVerdict:
        """Count one attempt against both budgets and say whether it may proceed.

        Counted before the password is checked, so a refused attempt still costs the
        attacker part of their budget. A Redis failure allows the attempt: the alternative is
        that a broken counter locks every account out of the deployment, which is a worse
        outage than a briefly unbounded guessing rate, and it is logged so it is not silent.
        """
        if self._backend is None:
            return RateLimitVerdict(allowed=True, retry_after_seconds=0)
        keys = [f"login:subject:{subject}"]
        if source:
            keys.append(f"login:source:{source}")
        try:
            counts = [await self._count(key) for key in keys]
        except Exception:  # noqa: BLE001 - any backend failure is one decision
            # Never the reason a login is refused. Logged without the subject: an email
            # beside a login outcome in shared logs is exactly what §5.2 forbids.
            _logger.warning("login_rate_limit_unavailable")
            return RateLimitVerdict(allowed=True, retry_after_seconds=0)
        if any(count > self._attempts for count in counts):
            return RateLimitVerdict(allowed=False, retry_after_seconds=self._window_seconds)
        return RateLimitVerdict(allowed=True, retry_after_seconds=0)

    async def _count(self, key: str) -> int:
        """Increment one window counter, setting its lifetime on the first attempt."""
        count = int(await self._backend.incr(key))  # type: ignore[union-attr]
        if count == 1:
            # Only on the first increment. Refreshing the lifetime on every attempt would
            # turn a fixed window into a sliding one that never expires while an attacker
            # keeps trying, which locks the address out permanently.
            await self._backend.expire(key, self._window_seconds)  # type: ignore[union-attr]
        return count


__all__ = ["LoginRateLimiter", "RateLimitBackend", "RateLimitVerdict"]
