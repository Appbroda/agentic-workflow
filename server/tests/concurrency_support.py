"""Doubles that hold real state, for the two-worker tier.

Audit risk P1-8, third tier (C3). Every mechanism this plan added to stop two workers
colliding was verified by one worker calling it twice. What a concurrency test needs is two
things running at once and a *durable* answer afterwards, and these are the parts that make
one:

* `SharedKeyStore` is the state behind `RedisWorkflowLock`. The lock itself is the production
  class -- its ownership token, its lease renewal and its compare-and-delete release all run
  as deployed -- and only the key/value store underneath is local. That matters because the
  run lock is what turns two claims on one feature into `FeatureExecutionBusy`, and a test
  substituting the lock would prove nothing about the branch that raises it.
* `BlockingStepOrchestrator` holds one step open until a test lets it finish. That is the
  only way to have a second worker arrive *during* a step rather than between two of them,
  which is where every race in the table actually lives.
* `StepRecordingOrchestrator` records which step ran for which feature. "Each step ran
  exactly once" is a statement about work, not about locks, and it is the assertion the whole
  tier is for.

Doubles hold state; they do not count calls, except where a call count *is* the property
under test. See overview section 4.4.
"""

from __future__ import annotations

import asyncio
from typing import Any

from api.control_plane import RequestScopedCredentials
from state.feature_models import FeatureWorkflowSnapshot
from workflows.feature_workflow import FeatureStep, FeatureWorkflowOrchestrator

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)


class SharedKeyStore:
    """The key/value semantics `RedisWorkflowLock` needs, held in this process.

    Only `set NX`, the two Lua scripts and `exists` are implemented, because those are the
    only operations the lock performs. Expiry is deliberately absent: a lease that lapses on
    wall-clock time would make every test in this tier a race against its own timings, and
    the lapse a test actually wants is written explicitly by releasing the key.
    """

    def __init__(self) -> None:
        """Start with nothing held."""
        self.values: dict[str, str] = {}

    async def set(self, name: str, value: str, *, nx: bool, ex: int) -> bool | None:
        """Take a key only when nobody holds it, which is what makes the lock a lock."""
        assert nx is True
        assert ex > 0
        if name in self.values:
            return None
        self.values[name] = value
        return True

    async def eval(self, _script: str, numkeys: int, *keys_and_args: str) -> int:
        """Renew or release, and only for the caller that still owns the key."""
        assert numkeys == 1
        key, token, *_lease = keys_and_args
        if self.values.get(key) != token:
            return 0
        if len(keys_and_args) == 3:
            return 1
        del self.values[key]
        return 1

    async def exists(self, name: str) -> int:
        """Report whether a key currently has an owner."""
        return int(name in self.values)


class StepRecordingOrchestrator(FeatureWorkflowOrchestrator):
    """The deterministic orchestrator, plus a record of every step it was asked to run."""

    def __init__(self) -> None:
        """Start with nothing run."""
        super().__init__()
        self.steps: list[str] = []

    async def _run_step(
        self, state: FeatureWorkflowSnapshot, step: FeatureStep, *, credentials: Any
    ) -> FeatureWorkflowSnapshot:
        """Record this step, then run it exactly as the orchestrator would."""
        self.steps.append(f"{state.feature_id}:{step.value}")
        return await super()._run_step(state, step, credentials=credentials)

    def duplicated(self) -> list[str]:
        """Return every step that ran more than once, which must always be empty."""
        return sorted({item for item in self.steps if self.steps.count(item) > 1})


class BlockingStepOrchestrator(StepRecordingOrchestrator):
    """Hold the first step open, so a second worker arrives while the first is inside one.

    Between two steps a feature has no owner and every race in the table is trivially safe.
    The window that matters is the one a step is executing in, and reaching it from a test
    means a step that does not return until the test says so.
    """

    def __init__(self) -> None:
        """Start with nothing running and nothing released."""
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self._blocked_once = False

    async def _run_step(
        self, state: FeatureWorkflowSnapshot, step: FeatureStep, *, credentials: Any
    ) -> FeatureWorkflowSnapshot:
        """Block inside the first step only, so the rest of the feature still finishes."""
        if not self._blocked_once:
            self._blocked_once = True
            self.entered.set()
            await self.release.wait()
        return await super()._run_step(state, step, credentials=credentials)


async def credentials_for(_owner: str | None) -> RequestScopedCredentials:
    """Resolve the credentials a dispatcher asks for, which this tier never needs."""
    return CREDENTIALS


__all__ = [
    "CREDENTIALS",
    "BlockingStepOrchestrator",
    "SharedKeyStore",
    "StepRecordingOrchestrator",
    "credentials_for",
]
