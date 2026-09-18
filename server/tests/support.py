"""Helpers for tests that submit a feature and then want to see what it became.

A feature is accepted and queued, and executed afterwards. A test that only wants the
end state therefore has to drain the queue, and doing that by sleeping would make every one of
these tests slow and occasionally wrong. These two helpers run exactly the code the deployment
runs -- the same dispatcher, the same claim, the same executor -- to completion, and return.
"""

from __future__ import annotations

from typing import Any

from api.control_plane import RequestScopedCredentials
from services.feature_queue import FeatureQueueDispatcher


async def drain_feature_queue(
    control_plane: Any, *, credentials: RequestScopedCredentials | None = None
) -> int:
    """Execute every queued feature this control plane holds, and return how many ran."""
    resolved = credentials or RequestScopedCredentials(openai_api_key=None, github_token=None)

    async def credentials_for(_owner_id: str) -> RequestScopedCredentials:
        return resolved

    dispatcher = FeatureQueueDispatcher(
        queue=control_plane.queue,
        executor=control_plane,
        credentials_for=credentials_for,
    )
    return await dispatcher.drain()


async def settle(app: Any) -> None:
    """Wait until an application has finished everything it has accepted.

    An isolated application dispatches each acceptance as its own background task, so this
    awaits those and then drains anything still queued.
    """
    await app.state.feature_dispatcher.wait_for_idle()
