"""Coverage for finding a feature again without knowing its identifier."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import (
    FeatureSummary,
    decode_feature_cursor,
    encode_feature_cursor,
    feature_dashboard_group,
)
from api.feature_schemas import StartFeatureRequest
from main import create_app
from state.enums import FeatureWorkflowStatus
from storage.db import Database
from storage.feature_store import SqlAlchemyFeatureControlPlane
from tests.support import settle
from workflows.feature_workflow import FeatureWorkflowOrchestrator

KEY = "listing-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


def _payload(title: str) -> dict[str, Any]:
    """Return the smallest valid feature submission, with a distinguishing title."""
    return {
        "prd": {
            "title": title,
            "problem_statement": "Operators cannot tell whether the server is healthy.",
            "goals": ["Operators can see server health"],
            "user_stories": [
                {
                    "story_id": "story-1",
                    "persona": "Operator",
                    "need": "to see server status",
                    "benefit": "outages are noticed",
                    "acceptance_criteria": ["A status tile is shown"],
                }
            ],
            "requirements": [
                {
                    "requirement_id": "requirement-1",
                    "description": "Expose a health endpoint",
                    "priority": "must",
                    "acceptance_criteria": ["GET /status returns the state"],
                    "dependencies": [],
                }
            ],
            "constraints": [],
            "out_of_scope": [],
            "stakeholders": ["Platform"],
        },
        "repositories": [
            {
                "repository_id": "backend",
                "name": "Backend",
                "role": "backend",
                "repository_url": "https://github.com/example/backend.git",
                "default_branch": "main",
                "required": True,
            }
        ],
        "execution_mode": "mock",
    }


@pytest.fixture(name="client")
def client_fixture() -> TestClient:
    """Serve an isolated application, whose feature store is the one a deployment runs."""
    return TestClient(create_app(platform_api_key=KEY))


def test_listing_requires_the_platform_key(client: TestClient) -> None:
    """The list is feature data and must not be readable without authentication."""
    assert client.get("/features").status_code == 401


def test_features_are_listed_newest_first(client: TestClient) -> None:
    """A product manager who closed the tab has to be able to find their work again."""
    for title in ("First feature", "Second feature", "Third feature"):
        assert client.post("/features/start", headers=AUTH, json=_payload(title)).status_code == 201

    listed = client.get("/features", headers=AUTH).json()

    assert [item["title"] for item in listed["features"]] == [
        "Third feature",
        "Second feature",
        "First feature",
    ]
    assert listed["next_cursor"] is None


def test_the_page_cursor_walks_every_feature_exactly_once(client: TestClient) -> None:
    """Paging must not skip or repeat a feature, which is why it is keyset and not offset."""
    for index in range(5):
        client.post("/features/start", headers=AUTH, json=_payload(f"Feature {index}"))

    seen: list[str] = []
    cursor: str | None = None
    for _ in range(10):  # bounded so a broken cursor fails the test instead of hanging
        query = {"limit": 2} | ({"cursor": cursor} if cursor else {})
        page = client.get("/features", headers=AUTH, params=query).json()
        seen.extend(item["feature_id"] for item in page["features"])
        cursor = page["next_cursor"]
        if cursor is None:
            break

    assert cursor is None
    assert len(seen) == 5
    assert len(set(seen)) == 5


def test_a_malformed_cursor_is_rejected_rather_than_silently_restarting(
    client: TestClient,
) -> None:
    """Returning page one for an unreadable cursor would loop a client forever."""
    response = client.get("/features", headers=AUTH, params={"cursor": "not-a-cursor"})

    assert response.status_code == 400


def test_the_listing_limit_is_bounded(client: TestClient) -> None:
    """An unbounded page would become the most expensive call the API serves."""
    assert client.get("/features", headers=AUTH, params={"limit": 101}).status_code == 422
    assert client.get("/features", headers=AUTH, params={"limit": 0}).status_code == 422


def test_a_cursor_round_trips() -> None:
    """The cursor is an encoding of a keyset position, not an opaque server-side token."""
    summary = FeatureSummary(
        feature_id="feature-round-trip",
        workflow_id="feature-round-trip",
        title="Round trip",
        status=FeatureWorkflowStatus.COMPLETED,
        execution_mode="mock",
        agent_platform="openai",
        created_at=datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
        updated_at=datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
    )

    assert decode_feature_cursor(encode_feature_cursor(summary)) == (
        summary.created_at,
        summary.feature_id,
    )


def test_dashboard_categories_are_owned_by_the_server_and_keep_outcomes_separate() -> None:
    """Presentation tone cannot collapse failed and cancelled into one product category."""
    assert feature_dashboard_group(FeatureWorkflowStatus.WAITING_FOR_HUMAN) == "waiting"
    assert feature_dashboard_group(FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN) == "waiting"
    assert feature_dashboard_group(FeatureWorkflowStatus.FAILED) == "failed"
    assert feature_dashboard_group(FeatureWorkflowStatus.CANCELLED) == "cancelled"
    assert feature_dashboard_group(FeatureWorkflowStatus.COMPLETED) == "completed"
    assert feature_dashboard_group(FeatureWorkflowStatus.PLANNING) == "running"


@pytest.mark.asyncio
async def test_the_durable_store_pages_in_the_same_order(tmp_path: Path) -> None:
    """The in-memory and PostgreSQL-backed planes must not disagree about ordering."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'listing.db'}")
    await database.create_schema()
    plane = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    for index in range(3):
        await plane.start(
            StartFeatureRequest.model_validate(
                _payload(f"Durable {index}") | {"feature_id": f"feature-durable-{index}"}
            ),
            idempotency_key=f"key-{index}",
            credentials=credentials,
            owner_id="platform-admin",
        )

    first = await plane.list_features(limit=2, cursor=None)
    second = await plane.list_features(limit=2, cursor=first.next_cursor)

    assert [item.feature_id for item in first.features] == [
        "feature-durable-2",
        "feature-durable-1",
    ]
    assert first.next_cursor is not None
    assert [item.feature_id for item in second.features] == ["feature-durable-0"]
    assert second.next_cursor is None
    await database.dispose()


@pytest.mark.asyncio
async def test_paging_is_stable_while_new_features_are_written(tmp_path: Path) -> None:
    """An offset would shift every row when a feature is created mid-page; a keyset does not."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'stable.db'}")
    await database.create_schema()
    plane = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
    for index in range(4):
        await plane.start(
            StartFeatureRequest.model_validate(
                _payload(f"Stable {index}") | {"feature_id": f"feature-stable-{index}"}
            ),
            idempotency_key=f"stable-{index}",
            credentials=credentials,
            owner_id="platform-admin",
        )

    first = await plane.list_features(limit=2, cursor=None)
    # A newer feature arrives between the two page reads.
    await plane.start(
        StartFeatureRequest.model_validate(
            _payload("Arrived late") | {"feature_id": "feature-stable-late"}
        ),
        idempotency_key="stable-late",
        credentials=credentials,
        owner_id="platform-admin",
    )
    second = await plane.list_features(limit=2, cursor=first.next_cursor)

    # The second page continues from where the first ended rather than repeating a row that
    # the newer write pushed down, and the late arrival never appears behind the cursor.
    assert not {item.feature_id for item in first.features} & {
        item.feature_id for item in second.features
    }
    assert "feature-stable-late" not in {item.feature_id for item in second.features}
    assert [item.feature_id for item in second.features] == [
        "feature-stable-1",
        "feature-stable-0",
    ]
    await database.dispose()


@pytest.mark.asyncio
async def test_listing_carries_dashboard_counts_without_reading_parent_state() -> None:
    """A dashboard needs counts per row; hydrating parent state to get them is the trap.

    Hydrating a feature means reading every artifact it produced -- roughly 400 KB for a
    completed two-repository feature -- so counting from that per row would make listing
    recent work more expensive than the work. These come from two tables already indexed on
    `feature_id`.
    """
    app = create_app(platform_api_key="listing-key")
    headers = {"Authorization": "Bearer listing-key", "Idempotency-Key": "listing-counts"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        # The counts under test are of finished work, so the queued feature has to be run.
        await settle(app)
        listed = await client.get("/features", headers=headers)

    [summary] = [
        item for item in listed.json()["features"] if item["feature_id"] == "feature-login"
    ]
    assert summary["repository_count"] == 2
    assert summary["pull_request_count"] == 2
    # A completed feature is not waiting on anybody.
    assert summary["human_action_required"] is False
    assert summary["workflow_id"] == summary["feature_id"]
    assert summary["dashboard_group"] == "completed"
    # Deliberately not offered by the list: it is only reachable through the parent state blob.
    assert "current_agent" not in summary


def feature_payload() -> dict[str, Any]:
    """Reuse the shared two-repository payload the feature API tests submit."""
    from tests.test_feature_api import feature_payload as shared_payload

    return shared_payload()
