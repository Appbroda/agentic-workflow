"""Capture a real design-snapshot artifact envelope for the client's rendering tests.

Run from `server/`:

    uv run python -m tests.capture_design_snapshot_fixture

Writes `client/tests/fixtures/design-snapshot.json` and `design-snapshot.omissions.json`. Both
are genuine responses from the real application: the real `/features/start` path, the real
resolution step, the shipped extraction over the live Figma payload committed at
`tests/fixtures/figma/figma_file_read.json`, and the real artifact endpoint's own response
model. The only thing stubbed is the network -- the adapter is replaced by a client that
answers out of that committed capture.

A hand-written fixture is how the web client silently lost a dozen fields once, and here it
would hide the one thing the renderer exists to check: that the field names the server
publishes and the field names the console reads are the same field names.

Two variants, because being honest about the second is the renderer's whole job:

* `design-snapshot.json` -- three frames quoted, nothing left out.
* `design-snapshot.omissions.json` -- one frame quoted and one omitted by the character bound,
  which is what a real snapshot looks like when a cap bites.

Read-only and offline.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

from httpx import ASGITransport, AsyncClient

from main import create_app
from services.design_resolution import FigmaDesignResolver
from services.feature_queue import DatabaseFeatureExecutionQueue
from services.secrets import InMemorySecretStore
from storage.db import EphemeralDatabase
from storage.design_source_store import InMemoryDesignSourceConfigurationDirectory
from storage.feature_store import SqlAlchemyFeatureControlPlane
from tests.support import settle
from tests.test_design_snapshot import (
    REAL_FILE_KEY,
    _ScriptedFigmaClient,
    captured_subtree,
    one_file,
)
from tests.test_feature_api import feature_payload
from workflows.feature_workflow import FeatureWorkflowOrchestrator

KEY = "design-snapshot-capture-key"
FIXTURES = Path(__file__).resolve().parents[2] / "client" / "tests" / "fixtures"

# The three real frames the committed capture contains, and the bound that splits the two
# largest: they render to 8,994 and 7,270 characters, so 8,000 omits exactly one.
_ALL_FRAMES = ("10:11", "12:53", "14:0")
_TWO_FRAMES = ("10:11", "12:53")
_BITING_BOUND = 8_000


def _app(node_ids: tuple[str, ...], *, node_character_bound: int | None) -> Any:
    """An isolated application whose resolution runs the shipped extraction, offline."""
    scripted = _ScriptedFigmaClient(
        nodes={REAL_FILE_KEY: one_file([captured_subtree(item) for item in node_ids])}
    )

    async def factory() -> Any:
        return scripted

    bounds: dict[str, Any] = (
        {} if node_character_bound is None else {"node_character_bound": node_character_bound}
    )
    isolated = EphemeralDatabase()
    return create_app(
        platform_api_key=KEY,
        secret_store=InMemorySecretStore(),
        design_source_configuration=InMemoryDesignSourceConfigurationDirectory(),
        feature_control_plane=SqlAlchemyFeatureControlPlane(
            isolated,
            mock_runner=FeatureWorkflowOrchestrator(
                design_resolver=FigmaDesignResolver(client_factory=factory, **bounds)
            ),
            queue=DatabaseFeatureExecutionQueue(isolated),
        ),
    )


async def _capture(name: str, node_ids: tuple[str, ...], bound: int | None) -> None:
    """Start one citing feature, let it resolve, and write what the endpoint returns."""
    app = _app(node_ids, node_character_bound=bound)
    headers = {"Authorization": f"Bearer {KEY}", "Idempotency-Key": f"capture-{name}"}
    payload = feature_payload()
    payload["feature_id"] = f"feature-{name}"
    payload["prd"] = {
        **cast("dict[str, Any]", payload["prd"]),
        "design_references": [
            {
                "url": (
                    f"https://www.figma.com/design/{REAL_FILE_KEY}/Untitled?node-id="
                    + ",".join(item.replace(":", "-") for item in node_ids)
                ),
                "label": "the clock",
            }
        ],
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as http:
        await http.put("/design-source", headers=headers, json={"enabled": True})
        started = await http.post("/features/start", headers=headers, json=payload)
        started.raise_for_status()
        feature_id = started.json()["feature_id"]
        await settle(app)
        response = await http.get(
            f"/features/{feature_id}/artifacts",
            headers=headers,
            params={"artifact_type": "design_snapshot"},
        )
    response.raise_for_status()
    artifacts = response.json()["artifacts"]
    if not artifacts:
        msg = f"{name}: the feature produced no design snapshot"
        raise RuntimeError(msg)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / f"{name}.json").write_text(
        json.dumps(artifacts[0], indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    payload_written = artifacts[0]["payload"]
    print(
        f"wrote {FIXTURES / f'{name}.json'}: "
        f"{len(payload_written['nodes'])} quoted, "
        f"{len(payload_written['design_nodes_omitted'])} omitted"
    )


async def capture() -> None:
    """Write both variants."""
    await _capture("design-snapshot", _ALL_FRAMES, None)
    await _capture("design-snapshot.omissions", _TWO_FRAMES, _BITING_BOUND)


if __name__ == "__main__":
    asyncio.run(capture())
