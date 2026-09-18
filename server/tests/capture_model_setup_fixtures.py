"""Capture real API payloads for the client's model-setup tests.

Run from `server/`:

    uv run python -m tests.capture_model_setup_fixtures

Writes JSON into `client/tests/fixtures/`. These are genuine responses from the real
application — the same routers, response models and serialization a deployment runs — not
hand-written fixtures, which dropped a dozen fields from this client before. Refresh by
re-running after changing the response models.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from httpx import ASGITransport, AsyncClient

from tests.support import settle
from tests.test_feature_api import feature_payload
from tests.test_model_setups import KEY, app_with, setup_payload, stocked_store

AUTH = {"Authorization": f"Bearer {KEY}"}
FIXTURES = Path(__file__).resolve().parents[2] / "client" / "tests" / "fixtures"


async def capture() -> None:
    """Author a setup, start a feature on it, and save the three payloads the client reads."""
    app = app_with(secret_store=await stocked_store("anthropic", "openai", "github"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as http:
        created = await http.post("/model-setups", headers=AUTH, json=setup_payload())
        created.raise_for_status()
        setup_id = created.json()["setup_id"]

        started = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "fixture-capture"},
            json={**feature_payload(), "model_setup_id": setup_id},
        )
        started.raise_for_status()
        feature_id = started.json()["feature_id"]
        await settle(app)
        # Edit the setup so the captured feature shows the edited-since sentence's data shape.
        edited = setup_payload()
        edited["roles"]["review"]["reasoning_effort"] = "medium"
        edited["roles"]["review"]["max_tokens"] = 32_000
        (await http.put(f"/model-setups/{setup_id}", headers=AUTH, json=edited)).raise_for_status()

        table = (await http.get("/model-configuration", headers=AUTH)).json()
        _write("model-configuration.custom-setup.json", table)
        _write("model-setups.mixed.json", (await http.get("/model-setups", headers=AUTH)).json())
        _write(
            "feature.pinned-model-setup.json",
            (await http.get(f"/features/{feature_id}", headers=AUTH)).json(),
        )


def _write(name: str, payload: dict[str, Any]) -> None:
    """Write one captured payload, stably formatted so a refresh diffs cleanly."""
    path = FIXTURES / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    asyncio.run(capture())
