"""The three Slack Web API calls this platform makes, classified the platform's own way.

Three endpoints and no SDK: `chat.postMessage`, `auth.test`, `conversations.info`. `httpx`
is already pinned and in the lock, and an SDK would be one more dependency whose logging
would have to be screened for what it does with a token.

Every refusal is reclassified before it leaves this module, the way
`github_adapter._comment_failure` does it: the platform's own error code, the endpoint, the
HTTP status or Slack `error` string, and a cause stated as a possibility, never as a finding.
No stack trace, no provider message quoted, no token. The classification carries what the
dispatcher's failure table needs -- retryable or not, a `Retry-After` instant, and whether
the refusal means delivery as a whole must stop (a revoked token, an archived channel)
rather than one message having failed.
"""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

_SLACK_API_BASE = "https://slack.com/api"
_REQUEST_TIMEOUT_SECONDS = 10.0

# Slack `error` strings that mean the token itself is dead. Nothing retried under a dead
# token succeeds, and every retry logs another line that buries the one that says what to do.
_TOKEN_REVOKED_ERRORS = frozenset({"invalid_auth", "token_revoked", "account_inactive"})
# Slack `error` strings that mean the channel cannot be delivered to at all.
_CHANNEL_GONE_ERRORS = frozenset({"is_archived", "channel_not_found", "not_in_channel"})


class SlackFailureMode(StrEnum):
    """The dispatcher's three-mode failure table, plus the ordinary per-message refusal."""

    # Slack down, a 5xx, or a timeout: retry the message on a later sweep, bounded.
    TRANSPORT = "transport"
    # A 429: retry, but not before the instant Slack named.
    RATE_LIMITED = "rate_limited"
    # The token is revoked: all sending stops until an operator re-saves it.
    TOKEN_REVOKED = "token_revoked"
    # The channel is archived, deleted, or the bot was removed from it.
    CHANNEL_UNAVAILABLE = "channel_unavailable"
    # A refusal about this one message (too long, malformed blocks). Not retryable, and not
    # a reason to stop delivering everything else.
    MESSAGE_REFUSED = "message_refused"


@dataclass(frozen=True, slots=True)
class PostedMessage:
    """What a successful `chat.postMessage` proves: where the message is, and its ts."""

    channel: str
    ts: str


@dataclass(frozen=True, slots=True)
class SlackAuthIdentity:
    """The workspace `auth.test` says this token belongs to, for display and nothing else."""

    workspace_id: str
    workspace_name: str


@dataclass(frozen=True, slots=True)
class SlackChannelInfo:
    """What `conversations.info` says about a channel, bounded to what the console shows."""

    channel_id: str
    channel_name: str
    is_archived: bool


class SlackClientError(RuntimeError):
    """One classified Slack refusal, carrying only platform-owned diagnostics.

    ``error_code`` is the platform's classification (`slack_transport_503`,
    `slack_api_invalid_auth`), never a provider message. ``retry_after_seconds`` is set only
    for a rate limit that named one.
    """

    def __init__(
        self,
        message: str,
        *,
        mode: SlackFailureMode,
        error_code: str,
        endpoint: str,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.mode = mode
        self.error_code = error_code
        self.endpoint = endpoint
        self.retry_after_seconds = retry_after_seconds

    @property
    def retryable(self) -> bool:
        """Whether a later sweep may try this same message again."""
        return self.mode in (SlackFailureMode.TRANSPORT, SlackFailureMode.RATE_LIMITED)

    @property
    def disables_delivery(self) -> bool:
        """Whether this refusal means the configuration must degrade, not the message."""
        return self.mode in (SlackFailureMode.TOKEN_REVOKED, SlackFailureMode.CHANNEL_UNAVAILABLE)


class SlackClient:
    """The calls the dispatcher and the configuration check make, one signature each.

    A `Protocol` in spirit; a plain base class in practice so the fake in the suite and the
    real adapter here are checked against the same signatures by mypy without a structural
    cast at every injection site.
    """

    async def post_message(
        self,
        channel: str,
        text: str,
        *,
        blocks: list[dict[str, Any]] | None = None,
        thread_ts: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PostedMessage:
        """Post one message, threaded when ``thread_ts`` names a root."""
        raise NotImplementedError

    async def auth_test(self) -> SlackAuthIdentity:
        """Ask Slack whose token this is."""
        raise NotImplementedError

    async def channel_info(self, channel: str) -> SlackChannelInfo:
        """Read one channel's name and whether it can still be delivered to."""
        raise NotImplementedError


class HttpxSlackClient(SlackClient):
    """The real adapter. Construction takes a token and opens nothing.

    The HTTP client is created per call and the `httpx` import is deferred to the first
    call, for `github_adapter._create_pygithub_client`'s reason: a mocked or test run that
    never sends must never initialize network-capable code.
    """

    def __init__(self, token: str) -> None:
        self._token = token

    async def post_message(
        self,
        channel: str,
        text: str,
        *,
        blocks: list[dict[str, Any]] | None = None,
        thread_ts: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PostedMessage:
        """Post one message and return the proof of where it landed."""
        payload: dict[str, Any] = {"channel": channel, "text": text}
        if blocks is not None:
            payload["blocks"] = blocks
        if thread_ts is not None:
            payload["thread_ts"] = thread_ts
        if metadata is not None:
            payload["metadata"] = metadata
        body = await self._call("chat.postMessage", payload)
        message_ts = body.get("ts")
        landed_channel = body.get("channel")
        if not isinstance(message_ts, str) or not isinstance(landed_channel, str):
            raise _refusal(
                "chat.postMessage",
                mode=SlackFailureMode.MESSAGE_REFUSED,
                error_code="slack_response_missing_ts",
            )
        return PostedMessage(channel=landed_channel, ts=message_ts)

    async def auth_test(self) -> SlackAuthIdentity:
        """Ask Slack whose token this is, keeping only the workspace identity."""
        body = await self._call("auth.test", {})
        return SlackAuthIdentity(
            workspace_id=str(body.get("team_id") or ""),
            workspace_name=str(body.get("team") or ""),
        )

    async def channel_info(self, channel: str) -> SlackChannelInfo:
        """Read one channel's display facts."""
        body = await self._call("conversations.info", {"channel": channel})
        raw = body.get("channel")
        detail = raw if isinstance(raw, dict) else {}
        return SlackChannelInfo(
            channel_id=str(detail.get("id") or channel),
            channel_name=str(detail.get("name") or ""),
            is_archived=bool(detail.get("is_archived", False)),
        )

    async def _call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Make one Web API call and reclassify every way it can refuse."""
        httpx = importlib.import_module("httpx")
        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{_SLACK_API_BASE}/{method}",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "Content-Type": "application/json; charset=utf-8",
                    },
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Timeouts, DNS failures, connection resets. The type name is platform-safe;
            # the message may carry a URL or a proxy detail, so only the type is kept.
            raise _refusal(
                method,
                mode=SlackFailureMode.TRANSPORT,
                error_code=f"slack_transport_{type(error).__name__.lower()}",
            ) from error
        return _classify_response(method, response.status_code, response)


def _classify_response(method: str, status: int, response: Any) -> dict[str, Any]:
    """Turn one HTTP response into a body or a classified refusal, never a raw error."""
    if status == 429:
        retry_after = _retry_after_seconds(response)
        raise _refusal(
            method,
            mode=SlackFailureMode.RATE_LIMITED,
            error_code="slack_rate_limited",
            retry_after_seconds=retry_after,
        )
    if status >= 500:
        raise _refusal(
            method,
            mode=SlackFailureMode.TRANSPORT,
            error_code=f"slack_transport_{status}",
        )
    try:
        body = response.json()
    except ValueError as error:
        raise _refusal(
            method,
            mode=SlackFailureMode.TRANSPORT,
            error_code="slack_transport_unreadable_response",
        ) from error
    if not isinstance(body, dict):
        raise _refusal(
            method,
            mode=SlackFailureMode.TRANSPORT,
            error_code="slack_transport_unreadable_response",
        )
    if body.get("ok") is True:
        return body
    return _classify_api_error(method, body)


def _classify_api_error(method: str, body: dict[str, Any]) -> dict[str, Any]:
    """Classify a well-formed `ok: false` answer into the failure table's modes.

    The `error` string is Slack's own enumerated vocabulary -- `invalid_auth`, `is_archived`
    -- not free text, which is what makes recording it inside the platform's code safe where
    recording a provider *message* is not.
    """
    error_value = body.get("error")
    error_name = error_value if isinstance(error_value, str) and error_value else "unnamed_error"
    if error_name in _TOKEN_REVOKED_ERRORS:
        raise _refusal(
            method,
            mode=SlackFailureMode.TOKEN_REVOKED,
            error_code=f"slack_api_{error_name}",
        )
    if error_name in _CHANNEL_GONE_ERRORS:
        raise _refusal(
            method,
            mode=SlackFailureMode.CHANNEL_UNAVAILABLE,
            error_code=f"slack_api_{error_name}",
        )
    raise _refusal(
        method,
        mode=SlackFailureMode.MESSAGE_REFUSED,
        error_code=f"slack_api_{error_name}",
    )


def _retry_after_seconds(response: Any) -> float | None:
    """Read the instant Slack named, and never invent one it did not."""
    header = response.headers.get("Retry-After") if hasattr(response, "headers") else None
    if header is None:
        return None
    try:
        seconds = float(header)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _refusal(
    endpoint: str,
    *,
    mode: SlackFailureMode,
    error_code: str,
    retry_after_seconds: float | None = None,
) -> SlackClientError:
    """Build one classified refusal whose message carries nothing provider-authored."""
    return SlackClientError(
        f"slack call refused ({error_code}) on {endpoint}",
        mode=mode,
        error_code=error_code,
        endpoint=endpoint,
        retry_after_seconds=retry_after_seconds,
    )


__all__ = [
    "HttpxSlackClient",
    "PostedMessage",
    "SlackAuthIdentity",
    "SlackChannelInfo",
    "SlackClient",
    "SlackClientError",
    "SlackFailureMode",
]
