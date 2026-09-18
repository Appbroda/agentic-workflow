"""AsyncClient coverage for authenticated, idempotent workflow control-plane endpoints.

The lifecycle these endpoints drive is exercised in `test_single_repository_workflow.py`,
against the one engine that runs it. What is here is the HTTP surface itself: who may call
it, what it refuses to parse, and that a repeated key replays rather than duplicates.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from main import create_app
from tests.support import settle


@pytest.mark.asyncio
async def test_workflow_routes_require_bearer_authentication() -> None:
    """Control-plane routes reject bad keys while health remains public."""
    app = create_app(platform_api_key="platform-test-key")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        health = await client.get("/healthz")
        missing = await client.get("/workflow/missing")
        invalid = await client.get(
            "/workflow/missing",
            headers={"Authorization": "Bearer incorrect"},
        )

    assert health.status_code == 200
    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert invalid.headers["www-authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_api_rejects_oversized_request_bodies_before_parsing_them() -> None:
    """A caller cannot force the control plane to parse an unbounded PRD payload."""
    app = create_app(platform_api_key="platform-test-key", max_request_body_bytes=1024)
    body = b'{"padding":"' + (b"x" * 2_000) + b'"}'

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/workflow/start",
            headers={
                "Authorization": "Bearer platform-test-key",
                "Content-Type": "application/json",
            },
            content=body,
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body exceeds the configured size limit."}


@pytest.mark.asyncio
async def test_start_is_idempotent_and_tokens_stay_out_of_request_models_and_artifacts() -> None:
    """A repeated key returns the original workflow and provider tokens remain request-scoped."""
    app = create_app(platform_api_key="platform-test-key")
    headers = {
        "Authorization": "Bearer platform-test-key",
        "Idempotency-Key": "idempotency-1",
        "X-OpenAI-Api-Key": "request-openai-token",
        "X-GitHub-Token": "request-github-token",
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        first = await client.post("/workflow/start", headers=headers, json=start_payload())
        replay = await client.post("/workflow/start", headers=headers, json=start_payload())
        await settle(app)
        artifacts = await client.get(
            "/workflow/workflow-api-1/artifacts",
            headers={"Authorization": "Bearer platform-test-key"},
        )
        timeline = await client.get(
            "/workflow/workflow-api-1/timeline",
            headers={"Authorization": "Bearer platform-test-key"},
        )
        token_in_body = await client.post(
            "/workflow/start",
            headers={"Authorization": "Bearer platform-test-key"},
            json={**start_payload(workflow_id="workflow-api-2"), "openai_api_key": "forbidden"},
        )

    assert first.status_code == 201
    assert first.json()["created"] is True
    assert replay.status_code == 200
    assert replay.json()["created"] is False
    assert replay.json()["workflow_id"] == first.json()["workflow_id"]
    assert artifacts.status_code == 200
    assert "request-openai-token" not in artifacts.text
    assert "request-github-token" not in artifacts.text
    assert timeline.status_code == 200
    # The vocabulary this surface has always answered in, whatever the engine underneath
    # names its own events.
    assert {event["event"] for event in timeline.json()["events"]} >= {
        "workflow_started",
        "prd",
        "technical_prd",
    }
    assert token_in_body.status_code == 422


@pytest.mark.asyncio
async def test_repository_credentials_are_rejected_before_workflow_persistence() -> None:
    """The single-repository start schema must reject URL userinfo before creating anything."""
    app = create_app(platform_api_key="platform-test-key")
    payload = start_payload(workflow_id="workflow-credential-url")
    payload["workspace_descriptor"]["source_repo_url"] = (
        "https://request-token@github.com/example/platform.git"
    )
    headers = {"Authorization": "Bearer platform-test-key"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        rejected = await client.post("/workflow/start", headers=headers, json=payload)
        audit_read = await client.get("/workflow/workflow-credential-url", headers=headers)

    assert rejected.status_code == 422
    assert audit_read.status_code == 404


def start_payload(*, workflow_id: str = "workflow-api-1") -> dict[str, Any]:
    """Return a valid API start payload containing no provider credential fields."""
    return {
        "workflow_id": workflow_id,
        "workspace_descriptor": {
            "workspace_id": "workspace-api-1",
            "root_path": "/tmp/workspace-api-1",
            "source_repo_url": "https://github.com/example/platform.git",
            "default_branch": "main",
            "working_branch": "workflow/workflow-api-1",
        },
        "prd": {
            "title": "API control plane",
            "problem_statement": "Clients need secure lifecycle control for engineering workflows.",
            "goals": ["Expose authenticated workflow state."],
            "user_stories": [
                {
                    "story_id": "story-1",
                    "persona": "Platform operator",
                    "need": "Start and inspect workflows",
                    "benefit": "I can supervise automation safely",
                    "acceptance_criteria": ["A workflow status response is returned."],
                }
            ],
            "requirements": [
                {
                    "requirement_id": "requirement-1",
                    "description": "The API must authenticate every control-plane request.",
                    "priority": "must",
                    "acceptance_criteria": ["Invalid tokens receive a 401 response."],
                    "dependencies": [],
                }
            ],
            "constraints": ["Credentials must remain header-only."],
            "out_of_scope": ["Provider token persistence."],
            "stakeholders": ["Platform team"],
        },
    }


@pytest.mark.asyncio
async def test_metrics_publishes_the_gauges_an_operator_watches() -> None:
    """Container logs rotate away, so the numbers that decide intervention must be readable.

    Without this, checking how many features are in flight against the quota, or how much the
    workspace volume has left, meant opening a psql session against production.
    """
    app = create_app(platform_api_key="metrics-key")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "platform_runtime_compatible" in body
    assert "# TYPE platform_runtime_compatible gauge" in body
