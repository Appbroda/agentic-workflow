"""Capture a real operations payload for the client's agent-work tests.

Run from `server/`, against a database that holds a multi-attempt run:

    DATABASE_URL=postgresql+asyncpg://platform:platform@localhost:15432/ai_platform \
    CAPTURE_FEATURE_ID=feature-... CAPTURE_REPOSITORY_ID=... \
    uv run python -m tests.capture_operations_fixture

Writes `client/tests/fixtures/operations.<name>.json`. It is a genuine response from the
real application -- the same router, response model and serialization a deployment runs, over
the records a real run left -- because a hand-written fixture is how the web client silently
lost a dozen fields once. The rendering tests build rows inline; this pins the *shape* of the
per-attempt block, which is the part a hand-written file would get wrong.

Read-only: the app is composed with a mock runner it never calls, and the only request made
is a GET.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from httpx import ASGITransport, AsyncClient

from main import create_app
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from storage.feature_store import SqlAlchemyFeatureControlPlane
from workflows.feature_workflow import FeatureWorkflowOrchestrator

KEY = "operations-capture-key"
FIXTURES = Path(__file__).resolve().parents[2] / "client" / "tests" / "fixtures"


async def capture() -> None:
    """Read one repository's journal and per-attempt endings through the real endpoint."""
    feature_id = os.environ["CAPTURE_FEATURE_ID"]
    repository_id = os.environ["CAPTURE_REPOSITORY_ID"]
    name = os.environ.get("CAPTURE_NAME", "captured")
    database = Database(os.environ["DATABASE_URL"])
    journal = ExternalOperationJournal(database)
    app = create_app(
        platform_api_key=KEY,
        feature_control_plane=SqlAlchemyFeatureControlPlane(
            database,
            # Never invoked: this script issues one GET. Composed because the control plane
            # requires a runner, not because anything here runs a workflow.
            mock_runner=FeatureWorkflowOrchestrator(workspace_root=Path("/tmp")),
            operation_journal=journal,
        ),
        operation_journal=journal,
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as http:
        response = await http.get(
            f"/features/{feature_id}/workstreams/{repository_id}/operations",
            headers={"Authorization": f"Bearer {KEY}"},
            params={"limit": 200},
        )
        response.raise_for_status()
        _write(f"operations.{name}.json", response.json())


def _write(name: str, payload: dict[str, Any]) -> None:
    """Write one captured payload, stably formatted so a refresh diffs cleanly."""
    path = FIXTURES / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(
        f"wrote {path} ({len(payload.get('operations', []))} rows, "
        f"{len(payload.get('attempts', []))} attempts)"
    )


if __name__ == "__main__":
    asyncio.run(capture())
