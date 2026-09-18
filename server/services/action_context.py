"""Request-local identity for external effects caused by a durable feature action.

The action id bound here is also the fence that stops a run which has lost its claim from
carrying on writing. AB-Feature-108 is why that fence exists: its lease lapsed at 15:22, the
action was reconciled `interrupted_before_completion` at 15:36, and at 15:53 something still
belonging to that run wrote a ninth attempt into the feature. Recovery had already decided
the run was dead; the run disagreed, and the database recorded whichever spoke last.

Lease-guarded writes in the action store already stop a zombie from overwriting its own
action's verdict. This closes the same hole for the feature state the run mutates on the way,
which no lease covered.
"""

from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar, Token

_CURRENT_FEATURE_ACTION_ID: ContextVar[str | None] = ContextVar(
    "current_feature_action_id", default=None
)

# Revocations are per-process and bounded: they exist to stop a coroutine this process is
# still running, and a process that restarts has no zombie left to fence. The cap keeps a
# long-lived API from accumulating one entry per lost lease forever; entries fall out oldest
# first, long after the work they refer to has been cancelled and unwound.
_MAX_REVOKED_ACTIONS = 1_024
_REVOKED_ACTION_IDS: OrderedDict[str, None] = OrderedDict()


def current_feature_action_id() -> str | None:
    """Return the action whose executor currently owns this async context."""
    return _CURRENT_FEATURE_ACTION_ID.get()


def bind_feature_action(action_id: str) -> Token[str | None]:
    """Associate subsequent journaled effects with one durable action."""
    return _CURRENT_FEATURE_ACTION_ID.set(action_id)


def reset_feature_action(token: Token[str | None]) -> None:
    """Restore the previous action association after execution completes."""
    _CURRENT_FEATURE_ACTION_ID.reset(token)


def revoke_feature_action(action_id: str) -> None:
    """Record that this action's executor may no longer change anything.

    Called when a lease is lost, before the work is cancelled. Cancellation is cooperative
    and a subprocess mid-flight can take minutes to unwind, so the window between "no longer
    owned" and "actually stopped" is real and was long enough to matter.
    """
    _REVOKED_ACTION_IDS[action_id] = None
    _REVOKED_ACTION_IDS.move_to_end(action_id)
    while len(_REVOKED_ACTION_IDS) > _MAX_REVOKED_ACTIONS:
        _REVOKED_ACTION_IDS.popitem(last=False)


def feature_action_is_revoked(action_id: str) -> bool:
    """Return whether one action's executor has been stripped of its claim."""
    return action_id in _REVOKED_ACTION_IDS


def current_feature_action_is_revoked() -> bool:
    """Return whether the action owning this context has lost its claim.

    False when no action is bound, which is the ordinary case for work that is not running
    under a durable action -- a mock run, a test, a direct control-plane call. Only an
    executor that was given a claim can have it taken away.
    """
    action_id = _CURRENT_FEATURE_ACTION_ID.get()
    return action_id is not None and feature_action_is_revoked(action_id)


def clear_revoked_feature_actions() -> None:
    """Forget every revocation. For tests, which must not leak state between cases."""
    _REVOKED_ACTION_IDS.clear()


__all__ = [
    "bind_feature_action",
    "clear_revoked_feature_actions",
    "current_feature_action_id",
    "current_feature_action_is_revoked",
    "feature_action_is_revoked",
    "reset_feature_action",
    "revoke_feature_action",
]
