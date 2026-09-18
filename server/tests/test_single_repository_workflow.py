"""The `/workflow/*` surface, now that a single-repository workflow is a one-repository feature.

There is one engine. These are the properties a caller of the older surface is entitled to --
its request and response schemas, its status vocabulary, its clarification contract -- plus the
assertions rescued from the graph tests that the compiled LangGraph workflow used to carry.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from httpx import ASGITransport, AsyncClient

from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from api.schemas import StartWorkflowRequest
from api.workflow_control_plane import (
    WORKSPACE_DESCRIPTOR_METADATA_KEY,
    _feature_request,
)
from artifacts.schemas import PullRequestArtifact
from main import create_app
from tests.support import settle
from tests.test_clarification_suggestions import _GLOBAL_AUTH_PREMISE
from tests.test_feature_workflow import RecordingReconnaissance
from workflows.feature_workflow import FeatureWorkflowOrchestrator

KEY = "single-repository-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


def workflow_payload(*, workflow_id: str = "workflow-single-1") -> dict[str, Any]:
    """Return a valid single-repository start payload with no provider credential fields."""
    return {
        "workflow_id": workflow_id,
        "workspace_descriptor": {
            "workspace_id": "workspace single 1",
            "root_path": "/tmp/workspace-single-1",
            "source_repo_url": "https://github.com/example/platform.git",
            "default_branch": "main",
            "working_branch": f"workflow/{workflow_id}",
        },
        "prd": {
            "title": "Single repository control plane",
            "problem_statement": "One repository still needs the older start contract.",
            "goals": ["Keep the single-repository surface working."],
        },
    }


def feature_payload(*, feature_id: str) -> dict[str, Any]:
    """Return the same work submitted directly as a one-repository feature."""
    return {
        "feature_id": feature_id,
        "prd": {
            "title": "Single repository control plane",
            "problem_statement": "One repository still needs the older start contract.",
            "goals": ["Keep the single-repository surface working."],
        },
        "repositories": [
            {
                "repository_url": "https://github.com/example/platform.git",
                "default_branch": "main",
            }
        ],
        "execution_mode": "mock",
    }


@pytest.mark.asyncio
async def test_a_single_repository_request_is_a_one_repository_feature() -> None:
    """The same work through either door produces the same observable outcome.

    This is what retiring the second engine has to mean. Not that the old surface still
    answers -- that is only the schema -- but that what it answers about is the same work,
    done by the same orchestrator, ending in the same place with the same handoffs.
    """
    app = create_app(platform_api_key=KEY)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/workflow/start", headers=AUTH, json=workflow_payload())
        await client.post(
            "/features/start", headers=AUTH, json=feature_payload(feature_id="feature-single-1")
        )
        await settle(app)
        workflow = await client.get("/workflow/workflow-single-1", headers=AUTH)
        workflow_artifacts = await client.get("/workflow/workflow-single-1/artifacts", headers=AUTH)
        feature = await client.get("/features/feature-single-1", headers=AUTH)
        feature_artifacts = await client.get("/features/feature-single-1/artifacts", headers=AUTH)

    # The terminal status, in each surface's own vocabulary. They agree because they are the
    # same value: `completed` is a member of both enums.
    assert workflow.json()["status"] == "completed"
    assert feature.json()["status"] == "completed"

    def types(response: Any) -> list[str]:
        return sorted(item["artifact_type"] for item in response.json()["artifacts"])

    assert types(workflow_artifacts) == types(feature_artifacts)
    assert types(workflow_artifacts).count("pull_request") == 1


@pytest.mark.asyncio
async def test_the_response_reports_the_status_vocabulary_this_surface_has_always_used() -> None:
    """`FeatureWorkflowStatus` has seventeen members and `WorkflowStatus` eleven.

    A caller of this surface never knew about the feature vocabulary and must not start
    receiving it. The translation is at the API boundary, which is why the domain underneath
    is free to have as many statuses as it needs.
    """
    app = create_app(platform_api_key=KEY)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        accepted = await client.post("/workflow/start", headers=AUTH, json=workflow_payload())
        await settle(app)
        finished = await client.get("/workflow/workflow-single-1", headers=AUTH)

    # Queued but not yet claimed. `pending` means the same thing in both vocabularies.
    assert accepted.json()["status"] == "pending"
    assert accepted.json()["created"] is True
    assert finished.json()["status"] == "completed"
    assert finished.json()["approval_state"] == "approved"
    # The workspace identity the caller supplied comes back, including the spaces that make it
    # an illegal repository id -- the feature underneath derives its own from the URL.
    assert finished.json()["workspace_id"] == "workspace single 1"


@pytest.mark.asyncio
async def test_a_published_pull_request_names_the_branches_the_commit_was_made_on() -> None:
    """Rescued from the graph's GitHub node, which refused to complete without this.

    A pull request opened from the wrong branch, or into the wrong one, proposes work nobody
    reviewed. The graph checked it against the workspace descriptor; the feature path opens
    each repository's pull request from the branch its own workstream committed on, and this
    is the assertion that the two are the same branch.
    """
    app = create_app(platform_api_key=KEY)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/workflow/start", headers=AUTH, json=workflow_payload())
        await settle(app)
        artifacts = await client.get("/workflow/workflow-single-1/artifacts", headers=AUTH)
        # Read through the feature surface deliberately: the branch the pull request must
        # name is the one the workstream committed on, and that is where it is recorded.
        workstreams = await client.get("/features/workflow-single-1/workstreams", headers=AUTH)

    [pull_request] = [
        item for item in artifacts.json()["artifacts"] if item["artifact_type"] == "pull_request"
    ]
    [workstream] = workstreams.json()["workstreams"]
    assert pull_request["payload"]["source_branch"] == workstream["branch_name"]
    assert pull_request["payload"]["target_branch"] == "main"
    # And it references the commit that was reviewed, not merely some commit.
    completions = [
        item for item in artifacts.json()["artifacts"] if item["artifact_type"] == "code_completion"
    ]
    assert pull_request["payload"]["commit_sha"] in {
        item["payload"]["commit_sha"] for item in completions
    }


@pytest.mark.asyncio
async def test_answers_must_match_the_open_questions_before_a_workflow_resumes() -> None:
    """The clarification contract, which is part of what `/workflow/resume` promises.

    An answer keyed to a question nobody asked is a client bug, and reporting it as one is
    the difference between a caller fixing their payload and a caller retrying it forever.
    """
    app = create_app(platform_api_key=KEY)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/workflow/start", headers=AUTH, json=workflow_payload())
        await settle(app)
        unanswered = await client.post(
            "/workflow/resume",
            headers=AUTH,
            json={
                "workflow_id": "workflow-single-1",
                "answers": [{"question_id": "nobody-asked", "answer": "PostgreSQL"}],
            },
        )
        missing = await client.post(
            "/workflow/resume",
            headers=AUTH,
            json={"workflow_id": "does-not-exist", "answers": []},
        )

    # Mock mode asks nothing, so this workflow finished; answers submitted to a workflow that
    # is not waiting on a person are refused rather than silently discarded.
    assert unanswered.status_code == 409
    assert "cannot be resumed" in unanswered.json()["detail"]
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_a_clarification_pauses_the_workflow_and_an_answer_resumes_it() -> None:
    """Rescued from the graph's human-clarification interrupt.

    The pause is what makes a question worth asking: the workflow stops, reports the older
    vocabulary\'s `waiting_for_human`, refuses an answer keyed to a question nobody asked, and
    continues once the open question is actually answered.
    """
    app = create_app(platform_api_key=KEY)
    app.state.feature_control_plane._mock_runner = FeatureWorkflowOrchestrator(  # noqa: SLF001
        reconnaissance=cast(
            Any,
            RecordingReconnaissance(contradicted_premises={"platform": [_GLOBAL_AUTH_PREMISE]}),
        )
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/workflow/start", headers=AUTH, json=workflow_payload())
        await settle(app)
        paused = await client.get("/workflow/workflow-single-1", headers=AUTH)
        mismatched = await client.post(
            "/workflow/resume",
            headers=AUTH,
            json={
                "workflow_id": "workflow-single-1",
                "answers": [{"question_id": "nobody-asked", "answer": "Per-route auth."}],
            },
        )
        questions = await client.get("/features/workflow-single-1/clarification", headers=AUTH)
        answered = await client.post(
            "/workflow/resume",
            headers=AUTH,
            json={
                "workflow_id": "workflow-single-1",
                "answers": [
                    {"question_id": item["question_id"], "answer": "Use the global middleware."}
                    for item in questions.json()["questions"]
                ],
            },
        )
        await settle(app)
        resumed = await client.get("/workflow/workflow-single-1", headers=AUTH)

    assert paused.json()["status"] == "waiting_for_human"
    assert paused.json()["approval_state"] == "pending"
    # An answer to a question nobody asked is a client bug, reported as one rather than
    # silently discarded or accepted as though it resolved something.
    assert mismatched.status_code == 422
    assert answered.status_code == 200
    assert resumed.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_the_read_routes_answer_for_a_workflow_that_has_run() -> None:
    """The four read surfaces, in the shapes this contract has always returned.

    `logs` is empty and says so rather than 404ing: structured execution evidence is recorded
    against artifacts and events, and inventing log entries here would be this surface
    reporting something no engine writes.
    """
    app = create_app(platform_api_key=KEY)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/workflow/start", headers=AUTH, json=workflow_payload())
        await settle(app)
        workflow = await client.get("/workflow/workflow-single-1", headers=AUTH)
        logs = await client.get("/workflow/workflow-single-1/logs", headers=AUTH)
        timeline = await client.get("/workflow/workflow-single-1/timeline", headers=AUTH)
        missing = await client.get("/workflow/workflow-single-1/logs", headers=AUTH)

    assert workflow.json()["current_agent"]
    assert workflow.json()["retry_count"] >= 0
    assert logs.json() == {"workflow_id": "workflow-single-1", "logs": []}
    assert missing.status_code == 200
    events = {item["event"] for item in timeline.json()["events"]}
    assert "workflow_started" in events
    # Nothing in the feature vocabulary leaks out of this surface.
    assert not any(item.startswith("feature_") for item in events)


@pytest.mark.asyncio
async def test_cancelling_a_settled_workflow_is_still_a_conflict() -> None:
    """A caller must not be able to tell that the engine underneath changed.

    The feature path answers a repeated cancellation idempotently, deliberately. This surface
    has always refused it, so the refusal is preserved at the boundary rather than by changing
    what a feature does.
    """
    app = create_app(platform_api_key=KEY)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/workflow/start", headers=AUTH, json=workflow_payload())
        await settle(app)
        refused = await client.post(
            "/workflow/cancel", headers=AUTH, json={"workflow_id": "workflow-single-1"}
        )

    assert refused.status_code == 409
    assert "cannot be cancelled again" in refused.json()["detail"]


@pytest.mark.asyncio
async def test_a_running_workflow_can_still_be_cancelled_and_reads_back_cancelled() -> None:
    """The lifecycle this surface exists for, end to end and in its own vocabulary."""
    app = create_app(platform_api_key=KEY)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/workflow/start", headers=AUTH, json=workflow_payload())
        cancelled = await client.post(
            "/workflow/cancel", headers=AUTH, json={"workflow_id": "workflow-single-1"}
        )
        read_back = await client.get("/workflow/workflow-single-1", headers=AUTH)

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert read_back.json()["status"] == "cancelled"


def test_the_start_request_becomes_one_repository_and_keeps_the_workspace_it_named() -> None:
    """The mapping itself, without a server: one `RepositorySpec`, identity from the URL.

    The workspace id is deliberately not borrowed for the repository id. `workspace_id` is a
    free-form string on this surface and a repository id is not, so a request that has always
    been accepted must not start being rejected by a validator it never met.
    """
    request = StartWorkflowRequest.model_validate(workflow_payload())
    feature = _feature_request(request, execution_mode="mock")

    assert isinstance(feature, StartFeatureRequest)
    [repository] = feature.repositories
    assert repository.repository_id == "platform"
    assert str(repository.repository_url) == "https://github.com/example/platform.git"
    assert repository.default_branch == "main"
    assert repository.metadata[WORKSPACE_DESCRIPTOR_METADATA_KEY]["workspace_id"] == (
        "workspace single 1"
    )
    assert feature.prd.title == "Single repository control plane"

    # And the feature it becomes is a feature like any other.
    state = _initial_feature_state("workflow-single-1", feature)
    assert len(state.repository_specs) == 1
    assert not any(isinstance(item, PullRequestArtifact) for item in state.artifacts)
