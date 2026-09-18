"""Authenticated API coverage for additive multi-repository parent feature workflows."""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from main import create_app
from tests.support import settle


@pytest.mark.asyncio
async def test_mock_feature_start_runs_two_repositories_and_returns_coordinated_prs() -> None:
    """One parent PRD produces two isolated child workstreams and cross-linked mock PR artifacts."""
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-001"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        started = await client.post("/features/start", headers=headers, json=feature_payload())
        # The response is the acceptance, not the outcome: the feature is queued and the work
        # happens on a worker. Draining runs that worker to completion.
        await settle(app)
        replay = await client.post("/features/start", headers=headers, json=feature_payload())
        completed = await client.get("/features/feature-login", headers=headers)
        workstreams = await client.get("/features/feature-login/workstreams", headers=headers)
        artifacts = await client.get("/features/feature-login/artifacts", headers=headers)
        pull_requests = await client.get("/features/feature-login/pull-requests", headers=headers)
        timeline = await client.get("/features/feature-login/timeline", headers=headers)

    assert started.status_code == 201, started.text
    # Queued, immediately, with the reference the server allocated. Nothing has been analysed.
    assert started.json()["status"] == "pending"
    assert started.json()["reference"] == "AB-Feature-1"
    assert completed.json()["status"] == "completed"
    assert completed.json()["reference"] == "AB-Feature-1"
    assert replay.status_code == 200
    assert replay.json()["created"] is False
    assert [item["repository_id"] for item in workstreams.json()["workstreams"]] == [
        "backend",
        "frontend",
    ]
    assert {item["status"] for item in workstreams.json()["workstreams"]} == {"completed"}
    artifact_types = [item["artifact_type"] for item in artifacts.json()["artifacts"]]
    # Parent and child artifacts keep their own real workflow identity; none is lost from the
    # public envelope.
    assert all(item["workflow_id"] for item in artifacts.json()["artifacts"])
    assert {
        "integration_contract",
        "repository_execution_plan",
        "child_workflow_result",
        "integration_review",
        "feature_completion",
    }.issubset(artifact_types)
    assert len(pull_requests.json()["pull_requests"]) == 2
    assert completed.json()["failure_summary"] is None
    assert {item["event"] for item in timeline.json()["events"]} >= {
        "feature_started",
        "feature_queued",
        "contract_created",
        "contract_approved",
        "child_workflow_started",
        "child_workflow_completed",
        "integration_review_completed",
        "pull_request_created",
        "feature_completed",
    }


@pytest.mark.asyncio
async def test_a_whitespace_idempotency_key_is_refused_as_a_409_not_a_500() -> None:
    """The refusal is an answer, and it must stay one wherever in the route it is raised.

    The replay lookup runs before the accepting transaction and derives the same idempotency
    key `start` does, so the same `WorkflowConflictError` -- a header that is nothing but
    whitespace -- now surfaces there first. Unmapped, it reached the client as a 500, which
    reads as a platform fault instead of a request the caller can fix.
    """
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "   "}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        refused = await client.post("/features/start", headers=headers, json=feature_payload())
        listed = await client.get("/features", headers={"Authorization": "Bearer feature-test-key"})

    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"] == "Idempotency-Key must not be empty"
    # Refused before anything was accepted: nothing was queued under a blank key.
    assert listed.json()["features"] == []


@pytest.mark.asyncio
async def test_rest_mutations_create_replayable_durable_actions() -> None:
    """Buttons and direct API calls must survive independently of their HTTP response."""
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-action"}
    body = {"operator": "A. Operator", "reason": "Close the completed audit record."}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        first = await client.post("/features/feature-login/retire", headers=headers, json=body)
        replay = await client.post("/features/feature-login/retire", headers=headers, json=body)
        actions = await client.get("/features/feature-login/actions", headers=headers)

    action_id = first.headers["x-feature-action-id"]
    assert replay.headers["x-feature-action-id"] == action_id
    matching = [item for item in actions.json()["actions"] if item["action_id"] == action_id]
    assert len(matching) == 1
    assert matching[0]["action_type"] == "RETIRE_FEATURE"
    assert matching[0]["origin"] == "rest"
    assert matching[0]["actor_id"] == "platform-admin"
    assert matching[0]["status"] == "succeeded"


@pytest.mark.asyncio
async def test_repository_credentials_are_rejected_before_feature_persistence() -> None:
    """URL userinfo in a request body must never reach even the mock durable state boundary."""
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key"}
    payload = feature_payload()
    payload["feature_id"] = "feature-credential-url"
    payload["repositories"][0]["repository_url"] = (
        "https://request-token@github.com/example/backend.git"
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        rejected = await client.post("/features/start", headers=headers, json=payload)
        audit_read = await client.get("/features/feature-credential-url", headers=headers)

    assert rejected.status_code == 422
    assert audit_read.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["feature_id", "repositories", "stories", "requirements"])
async def test_start_rejects_unsafe_or_ambiguous_identifiers_as_request_validation(
    invalid: str,
) -> None:
    """Invalid identity must be a 422 before a half-created workflow or artifact exists."""
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key"}
    payload = feature_payload()
    if invalid == "feature_id":
        payload["feature_id"] = "../unsafe"
    elif invalid == "repositories":
        payload["repositories"][1]["repository_id"] = payload["repositories"][0]["repository_id"]
    elif invalid == "stories":
        payload["prd"]["user_stories"].append(dict(payload["prd"]["user_stories"][0]))
    else:
        payload["prd"]["requirements"].append(dict(payload["prd"]["requirements"][0]))

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post("/features/start", headers=headers, json=payload)

    assert response.status_code == 422


def feature_payload() -> dict[str, Any]:
    """Return one frontend/backend parent request using no provider credentials in JSON."""
    return {
        "feature_id": "feature-login",
        "prd": {
            "title": "Login audit trail",
            "problem_statement": "Support staff need to trace failed login attempts.",
            "goals": ["Record failed attempts."],
            "user_stories": [
                {
                    "story_id": "story-login-audit",
                    "persona": "Support agent",
                    "need": "Inspect failed login attempts",
                    "benefit": "I can diagnose authentication issues",
                    "acceptance_criteria": ["An audit event is available for failed logins."],
                }
            ],
            "requirements": [
                {
                    "requirement_id": "requirement-login-audit",
                    "description": "Record failed authentication attempts.",
                    "priority": "must",
                    "acceptance_criteria": ["A failed login emits an audit event."],
                    "dependencies": [],
                }
            ],
            "constraints": ["Use mock providers."],
            "out_of_scope": ["Automatic merge."],
            "stakeholders": ["Support"],
        },
        "repositories": [
            {
                "repository_id": "backend",
                "name": "Backend API",
                "role": "backend",
                "repository_url": "https://github.com/example/backend.git",
                "default_branch": "main",
            },
            {
                "repository_id": "frontend",
                "name": "Frontend Web",
                "role": "frontend",
                "repository_url": "https://github.com/example/frontend.git",
                "default_branch": "main",
            },
        ],
    }


@pytest.mark.asyncio
async def test_reads_carry_the_repository_identity_a_client_must_render() -> None:
    """A workstream identifier alone cannot be shown to a person.

    `RepositorySpec` was accepted at submission and persisted, but no read endpoint returned
    it, so a client could name a workstream `backend` and say nothing about which repository
    that is. Roles travel as data because a client must not infer anything from them.
    """
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-identity"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        await settle(app)
        feature = await client.get("/features/feature-login", headers=headers)
        workstreams = await client.get("/features/feature-login/workstreams", headers=headers)

    repositories = {item["repository_id"]: item for item in feature.json()["repositories"]}
    assert set(repositories) == {"backend", "frontend"}
    assert repositories["backend"]["name"]
    assert repositories["backend"]["role"] == "backend"
    assert repositories["backend"]["repository_url"].startswith("https://")
    assert repositories["backend"]["required"] is True

    by_id = {item["repository_id"]: item for item in workstreams.json()["workstreams"]}
    assert by_id["frontend"]["repository_role"] == "frontend"
    assert by_id["frontend"]["repository_name"]
    assert by_id["frontend"]["repository_url"].startswith("https://")


@pytest.mark.asyncio
async def test_future_repository_role_remains_data_not_workflow_identity() -> None:
    """New role labels use the same generic workstream without a backend release."""
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "future-role"}
    payload = feature_payload()
    payload["feature_id"] = "feature-data-pipeline"
    payload["repositories"] = [
        {
            **payload["repositories"][0],
            "repository_id": "events-pipeline",
            "name": "Events Pipeline",
            "role": "data-pipeline",
        }
    ]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        started = await client.post("/features/start", headers=headers, json=payload)
        await settle(app)
        workstreams = await client.get(
            "/features/feature-data-pipeline/workstreams", headers=headers
        )

    assert started.status_code == 201, started.text
    assert started.json()["repositories"][0]["role"] == "data-pipeline"
    assert workstreams.json()["workstreams"][0]["repository_role"] == "data-pipeline"


@pytest.mark.asyncio
async def test_artifacts_can_be_listed_without_their_payloads_and_fetched_singly() -> None:
    """A completed two-repository feature returns roughly 400 KB when every payload is included.

    Listing envelopes and opening one on demand is the difference between a workspace that
    loads and one that downloads every artifact to show a heading.
    """
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-artifacts"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        await settle(app)
        listed = await client.get(
            "/features/feature-login/artifacts",
            headers=headers,
            params={"include_payload": "false"},
        )
        filtered = await client.get(
            "/features/feature-login/artifacts",
            headers=headers,
            params={"artifact_type": "integration_contract"},
        )
        contract_id = filtered.json()["artifacts"][0]["artifact_id"]
        single = await client.get(
            f"/features/feature-login/artifacts/{contract_id}", headers=headers
        )
        missing = await client.get(
            "/features/feature-login/artifacts/999_does_not_exist.json", headers=headers
        )

    assert listed.json()["artifacts"]
    assert all(item["payload"] == {} for item in listed.json()["artifacts"])
    # The envelope survives so a client can list and label what exists.
    assert all(item["artifact_type"] for item in listed.json()["artifacts"])
    assert {item["artifact_type"] for item in filtered.json()["artifacts"]} == {
        "integration_contract"
    }
    assert single.json()["artifact_id"] == contract_id
    assert single.json()["payload"]
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_open_clarification_questions_are_served_rather_than_derived() -> None:
    """Deciding which technical PRD is current is a server rule with a regression history.

    After reconnaissance adds its questions and an answer revises them, the current artifact
    is `002_technical_prd.revision-N.json`. A client re-implementing that lineage match is a
    second place for it to be wrong.
    """
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-clarify"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        await settle(app)
        clarification = await client.get("/features/feature-login/clarification", headers=headers)

    body = clarification.json()
    assert body["feature_id"] == "feature-login"
    # Mock mode asks nothing, so a client renders no action-required panel.
    assert body["awaiting_answers"] is False
    assert body["questions"] == []
    assert body["technical_prd_artifact_id"]
    assert body["max_clarification_rounds"] >= 1


@pytest.mark.asyncio
async def test_events_are_readable_as_a_delta_so_watching_stays_cheap() -> None:
    """Liveness polling must not hydrate parent state.

    The timeline endpoint merges lifecycle events with artifacts, which means loading every
    artifact the feature produced -- roughly 400 KB for a completed two-repository feature.
    Polling that every few seconds would make watching a feature cost more than running it, so
    the events endpoint reads only the indexed event table and resumes from a cursor.
    """
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-events"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        first = await client.get("/features/feature-login/events", headers=headers)
        cursor = first.json()["last_event_id"]
        delta = await client.get(
            "/features/feature-login/events", headers=headers, params={"after": cursor}
        )

    body = first.json()
    assert body["events"], "a started feature has lifecycle events"
    assert [item["id"] for item in body["events"]] == sorted(item["id"] for item in body["events"])
    assert body["last_event_id"] == body["events"][-1]["id"]
    # Nothing new since the cursor, so a poll returns an empty delta rather than the history.
    assert delta.json()["events"] == []
    assert delta.json()["last_event_id"] == cursor


@pytest.mark.asyncio
async def test_events_from_a_cursor_return_only_what_followed_it() -> None:
    """A reconnecting client must not replay what it already showed."""
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-cursor"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        everything = await client.get("/features/feature-login/events", headers=headers)
        events = everything.json()["events"]
        midpoint = events[0]["id"]
        after = await client.get(
            "/features/feature-login/events", headers=headers, params={"after": midpoint}
        )

    assert len(events) > 1
    assert [item["id"] for item in after.json()["events"]] == [
        item["id"] for item in events if item["id"] > midpoint
    ]


@pytest.mark.asyncio
async def test_refusing_an_operation_is_an_answer_not_a_server_error() -> None:
    """A completed feature cannot be resumed, and saying so is not a fault.

    The orchestrator's refusal reached the client as `500 Internal server error`, which reads
    as the platform breaking and tells an operator nothing about what to do instead.
    """
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-refusal"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        await settle(app)
        resumed = await client.post(
            "/features/feature-login/resume", headers=headers, json={"answers": []}
        )

    assert resumed.status_code == 409
    assert "cannot be resumed" in resumed.json()["detail"]


@pytest.mark.asyncio
async def test_the_read_model_says_whether_each_repository_would_be_published() -> None:
    """Publication eligibility travels as data, per repository, on the endpoint that lists them.

    `PUBLISH_FEATURE` is a feature-level decision -- one press publishes everything eligible
    -- so a bare button on the feature cannot say "2 ready to open, 1 held for lint". These
    two fields are what lets a client say it, and they are computed by the route from the
    workflow's own precondition rather than dumped off the child record: `WorkstreamResponse`
    forbids extra fields and is built by dumping the whole `ChildWorkflowReference`, so a new
    field on that model would break this endpoint and only this test would notice.
    """
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-publish"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        await settle(app)
        feature = await client.get("/features/feature-login", headers=headers)
        workstreams = await client.get("/features/feature-login/workstreams", headers=headers)
        published = await client.post(
            "/features/feature-login/publish",
            headers=headers,
            json={"reason": "open whatever is finished"},
        )
        unexplained = await client.post(
            "/features/feature-login/publish", headers=headers, json={"reason": ""}
        )

    assert workstreams.status_code == 200, workstreams.text
    for item in workstreams.json()["workstreams"]:
        # This feature landed and published itself, so nothing is awaiting a decision -- but
        # both fields must be present and honest rather than absent.
        assert item["publication_class"] is None
        assert "already has a pull request" in item["publication_refusal"]
    # And the feature does not advertise a decision nobody has to make.
    assert "PUBLISH_FEATURE" not in feature.json()["available_actions"]
    # Pressing it anyway is a refusal with a sentence, synchronously -- never a queued entry
    # that fails on a worker minutes later with nobody watching.
    assert published.status_code == 409, published.text
    assert "a completed feature cannot be published" in published.json()["detail"]
    # An override of the platform's own decision is not accepted without a stated reason.
    assert unexplained.status_code == 422


@pytest.mark.asyncio
async def test_a_retry_grant_states_who_asked_and_why_before_the_platform_will_run_it() -> None:
    """An override of the platform's own stop must be attributable.

    The endpoint exists to overrule a decision the platform made deliberately. Accepting one
    without an author or a reason would leave, three weeks later, an attempt count past the
    configured limit and no way to tell an operator's judgement from a platform defect.
    """
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-retry"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        await settle(app)
        anonymous = await client.post(
            "/features/feature-login/workstreams/backend/retry",
            headers=headers,
            json={"additional_attempts": 1, "reason": "Repaired the runner."},
        )
        unexplained = await client.post(
            "/features/feature-login/workstreams/backend/retry",
            headers=headers,
            json={"additional_attempts": 1, "requested_by": "akhilesh", "reason": ""},
        )
        greedy = await client.post(
            "/features/feature-login/workstreams/backend/retry",
            headers=headers,
            json={"additional_attempts": 50, "requested_by": "akhilesh", "reason": "Just run it."},
        )
        approved = await client.post(
            "/features/feature-login/workstreams/backend/retry",
            headers=headers,
            json={"additional_attempts": 1, "requested_by": "akhilesh", "reason": "Repaired it."},
        )
        unknown = await client.post(
            "/features/feature-login/workstreams/mobile/retry",
            headers=headers,
            json={"additional_attempts": 1, "requested_by": "akhilesh", "reason": "Typo."},
        )

    assert anonymous.status_code == 422
    assert unexplained.status_code == 422
    # A grant is a decision made on evidence somebody read; buying twenty at once is buying
    # them without reading the next four.
    assert greedy.status_code == 422
    # This mock feature ran to completion, and a completed feature has nothing to grant an
    # attempt to: its repositories were approved and their pull requests opened. The platform
    # refuses -- as a refusal with a reason, not as a fault.
    assert approved.status_code == 409
    assert "cannot retry a workstream" in approved.json()["detail"]
    assert unknown.status_code == 409
    # The refusals for a repository that is not stopped and for one that is not part of the
    # feature are exercised against a stopped feature in test_repository_preflight_and_retry.


@pytest.mark.asyncio
async def test_the_api_answers_under_the_prefix_a_browser_uses_and_at_its_own_paths() -> None:
    """A web client sharing this origin needs the API somewhere its own routes are not.

    The client's page for a feature is `/features/{id}` -- the same URL as this API's. Served
    together the API wins, so opening or refreshing a feature page returned JSON instead of the
    application. Found by loading the app in a browser; no test could have, because none of
    them is a browser navigating to a client route.

    Both mountings must work: existing clients and the deployment's health checks address the
    root paths and must not be moved.
    """
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-prefixed"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        at_root = await client.get("/features/feature-login", headers=headers)
        under_prefix = await client.get("/api/features/feature-login", headers=headers)
        vocabulary = await client.get("/api/console/status-vocabulary", headers=headers)
        unauthenticated = await client.get("/api/features/feature-login")

    assert at_root.status_code == 200
    assert under_prefix.status_code == 200
    assert under_prefix.json()["feature_id"] == at_root.json()["feature_id"]
    assert vocabulary.status_code == 200
    # The prefix is a second address for the same API, not a way around its authentication.
    assert unauthenticated.status_code == 401


@pytest.mark.asyncio
async def test_a_feature_starts_from_a_title_a_problem_and_a_repository_url() -> None:
    """The smallest honest submission is a title, a problem, and where to build it.

    Everything else the previous contract demanded -- a goal, a user story and a requirement,
    each with its own acceptance criteria, plus an identifier, a display name and a role per
    repository -- was analysis the platform exists to do, asked of the person requesting it.
    """
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-minimal"}
    payload = {
        "feature_id": "feature-minimal",
        "prd": {
            "title": "Deactivate ad units",
            "problem_statement": "Operators cannot deactivate an ad unit from the console.",
        },
        "repositories": [
            {
                "repository_url": "https://github.com/Appbroda/admanager_console-2.0",
                "default_branch": "master",
            }
        ],
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        started = await client.post("/features/start", headers=headers, json=payload)
        await settle(app)
        artifacts = await client.get("/features/feature-minimal/artifacts", headers=headers)

    assert started.status_code == 201, started.text
    repository = started.json()["repositories"][0]
    # Identity is derived from the URL and is still a real, stable repository id.
    assert repository["repository_id"] == "admanager_console-2.0"
    assert repository["name"] == "admanager_console-2.0"
    # A role nobody stated must not imply anything; `shared` is the one that would.
    assert repository["role"] == "other"
    assert repository["default_branch"] == "master"

    prd = next(item for item in artifacts.json()["artifacts"] if item["artifact_type"] == "prd")
    # The record of the submission is the submission: nothing was written on the author's
    # behalf and stored as though they had said it.
    assert prd["payload"]["requirements"] == []
    assert prd["payload"]["user_stories"] == []
    assert prd["payload"]["goals"] == []
    # The product manager is what turns that into something traceable.
    technical = next(
        item for item in artifacts.json()["artifacts"] if item["artifact_type"] == "technical_prd"
    )
    assert technical["payload"]["functional_requirements"]


@pytest.mark.asyncio
async def test_derived_repository_ids_stay_unique_across_owners() -> None:
    """Two owners can both publish `api`, and identity has to survive that."""
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-collide"}
    payload = {
        "feature_id": "feature-collide",
        "prd": {"title": "Two APIs", "problem_statement": "Both need the same change."},
        "repositories": [
            {"repository_url": "https://github.com/one/api", "default_branch": "main"},
            {"repository_url": "https://github.com/two/api", "default_branch": "main"},
        ],
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        started = await client.post("/features/start", headers=headers, json=payload)

    assert started.status_code == 201, started.text
    ids = [item["repository_id"] for item in started.json()["repositories"]]
    assert ids == ["api", "two-api"]


@pytest.mark.asyncio
async def test_an_explicit_repository_identity_is_never_rewritten() -> None:
    """Derivation fills gaps. It does not overrule somebody who named their repositories."""
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-explicit"}
    payload = {
        "feature_id": "feature-explicit",
        "prd": {"title": "Named", "problem_statement": "The caller chose these names."},
        "repositories": [
            {
                "repository_id": "sdk",
                "name": "Shared SDK",
                "role": "shared",
                "repository_url": "https://github.com/example/shared-contracts",
                "default_branch": "main",
            }
        ],
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        started = await client.post("/features/start", headers=headers, json=payload)

    assert started.status_code == 201, started.text
    repository = started.json()["repositories"][0]
    assert repository["repository_id"] == "sdk"
    assert repository["name"] == "Shared SDK"
    assert repository["role"] == "shared"


@pytest.mark.asyncio
async def test_a_feature_publishes_the_budgets_it_is_being_run_under() -> None:
    """ "Attempt 2" says nothing. "Attempt 2 of 8" says whether to wait or to intervene.

    The limits come from the feature's own state rather than from current settings: a feature
    started last week is still being run under the budgets it started with.
    """
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-limits"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        feature = await client.get("/features/feature-login", headers=headers)

    body = feature.json()
    for field in (
        "max_clarification_rounds",
        "max_integration_review_cycles",
        "max_child_review_cycles",
        "max_implementation_retries",
        "max_validation_retries",
        "max_repository_setup_retries",
    ):
        assert isinstance(body[field], int), field
    assert body["max_clarification_rounds"] >= body["clarification_rounds"]


@pytest.mark.asyncio
async def test_executions_are_served_once_for_the_whole_graph_and_carry_no_secrets() -> None:
    """A graph must cost one request, and what it publishes must be safe to put in a browser.

    A mock feature reaches no provider, so the assertion that matters most here is the one
    about what is *absent*: no arrow claims a model, because none ran.
    """
    app = create_app(platform_api_key="feature-test-key")
    headers = {"Authorization": "Bearer feature-test-key", "Idempotency-Key": "feature-exec"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        await settle(app)
        executions = await client.get("/features/feature-login/executions", headers=headers)
        missing = await client.get("/features/does-not-exist/executions", headers=headers)
        unauthenticated = await client.get("/features/feature-login/executions")

    assert executions.status_code == 200, executions.text
    assert missing.status_code == 404
    # The same authentication as every other read on this router; nothing about a graph makes
    # it public.
    assert unauthenticated.status_code == 401
    records = executions.json()["executions"]
    assert records, "a completed feature has transitions"
    assert len({item["execution_id"] for item in records}) == len(records)
    # Every repository the feature has, and the deterministic handlers that are not models.
    assert {item["repository_id"] for item in records} >= {"backend", "frontend"}
    handlers = {item["handler"] for item in records}
    assert {"Orchestrator", "GitHub"}.issubset(handlers)
    assert all(item["model"] is None for item in records)
    assert all(item["execution_mode"] == "mock" for item in records)
    body = executions.text
    for forbidden in ("Authorization", "Bearer", "api_key", "prompt_template", "jinja2"):
        assert forbidden not in body
