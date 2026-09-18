"""A person asks for changes to a completed feature, and a V2 replaces what it shipped.

Three layers, in the order a defect would be found: the pure state mutation (`begin_revision`
turns a completed snapshot into a planning one on `-V{n}` branches), the control plane's
synchronous refusals (a feature that is not completed, or that shipped nothing, is answered as
a conflict before anything is queued), and the whole lifecycle through the real queue -- a
feature completes, is revised, completes again with a `.v2` pull request while the superseded
one is closed with a cross-link comment, and is revised again onto `-V3`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adapters.github_adapter import MockGitHubService
from api.control_plane import RequestScopedCredentials, WorkflowConflictError
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import (
    FeatureCompletionArtifact,
    FeatureRevisionRequestArtifact,
    PullRequestArtifact,
    TechnicalPRDArtifact,
)
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from storage.db import Database
from storage.feature_store import SqlAlchemyFeatureControlPlane
from tests.support import drain_feature_queue
from tests.test_feature_api import feature_payload
from workflows.feature_workflow import (
    FeatureStep,
    FeatureWorkflowOrchestrator,
    GitHubPullRequestPublisher,
    begin_revision,
    next_step,
)

pytestmark = pytest.mark.asyncio

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)


async def _completed_feature(
    tmp_path: Path, name: str
) -> tuple[Database, SqlAlchemyFeatureControlPlane, MockGitHubService, str]:
    """Run one feature to completion through the real queue, on a shared mock GitHub."""
    github = MockGitHubService()
    runner = FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(github_service=github)
    )
    database = Database(f"sqlite+aiosqlite:///{tmp_path / name}")
    await database.create_schema()
    store = SqlAlchemyFeatureControlPlane(database, mock_runner=runner)
    feature_id = f"feature-revision-{name.removesuffix('.db')}"
    request = StartFeatureRequest.model_validate({**feature_payload(), "feature_id": feature_id})
    await store.start(
        request, idempotency_key=feature_id, credentials=CREDENTIALS, owner_id="platform-admin"
    )
    await drain_feature_queue(store)
    state = (await store.get_record(feature_id)).state
    assert state.status is FeatureWorkflowStatus.COMPLETED
    return database, store, github, feature_id


# --------------------------------------------------------------------------------------
# 1 -- the pure mutation
# --------------------------------------------------------------------------------------


async def test_begin_revision_moves_a_completed_feature_back_to_planning(tmp_path: Path) -> None:
    """Branches, workspaces, counters and artifacts all say 'revision one' afterwards."""
    database, store, _github, feature_id = await _completed_feature(tmp_path, "pure.db")
    try:
        state = (await store.get_record(feature_id)).state.model_copy(deep=True)
        before = {
            repository_id: child.model_copy(deep=True)
            for repository_id, child in state.child_workflows.items()
        }

        revised = begin_revision(state, request_text="Round the corners.", requested_by="a person")

        assert revised.revision == 1
        assert revised.status is FeatureWorkflowStatus.PLANNING
        for repository_id, child in revised.child_workflows.items():
            previous = before[repository_id]
            assert child.branch_name == f"{previous.branch_name}-V2"
            assert child.base_branch == previous.branch_name
            assert child.workspace_path == f"{previous.workspace_path}-v2"
            assert child.status is ChildWorkflowStatus.PENDING
            assert child.retry_count == previous.retry_count + 1
            assert child.pull_request_artifact_id is None
            assert child.blocking_issues == []
        request_artifact = next(
            item for item in revised.artifacts if isinstance(item, FeatureRevisionRequestArtifact)
        )
        assert request_artifact.artifact_id == "019_feature_revision_request.v1.json"
        assert request_artifact.request_text == "Round the corners."
        superseded = {item.repository_id: item for item in request_artifact.superseded}
        for repository_id, previous in before.items():
            assert superseded[repository_id].branch_name == previous.branch_name
            assert (
                superseded[repository_id].pull_request_artifact_id
                == previous.pull_request_artifact_id
            )
        latest_prd = next(
            item for item in reversed(revised.artifacts) if isinstance(item, TechnicalPRDArtifact)
        )
        assert latest_prd.functional_requirements[0].description == "Round the corners."
        assert latest_prd.unresolved_questions == []
        # The evidence calls for planning next: the superseded run's plan reads as absent.
        decision = next_step(revised)
        assert decision.step is FeatureStep.PLAN
    finally:
        await database.dispose()


async def test_a_second_revision_supersedes_the_first_not_the_original(tmp_path: Path) -> None:
    """`-V3` replaces `-V2` on the original stem; the base is always the previous branch."""
    database, store, _github, feature_id = await _completed_feature(tmp_path, "twice.db")
    try:
        state = (await store.get_record(feature_id)).state.model_copy(deep=True)
        first = begin_revision(state, request_text="First change.", requested_by="a person")
        # Between revisions the feature completes again; the pure function needs only the
        # counters and names, so completing is simulated by revising the revised state.
        second = begin_revision(first, request_text="Second change.", requested_by="a person")

        assert second.revision == 2
        for child in second.child_workflows.values():
            assert child.branch_name.endswith("-V3")
            assert "-V2-V3" not in child.branch_name
            assert child.base_branch is not None
            assert child.base_branch.endswith("-V2")
            assert child.workspace_path.endswith("-v3")
            assert "-v2-v3" not in child.workspace_path
        request_ids = [
            item.artifact_id
            for item in second.artifacts
            if isinstance(item, FeatureRevisionRequestArtifact)
        ]
        assert request_ids == [
            "019_feature_revision_request.v1.json",
            "019_feature_revision_request.v2.json",
        ]
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# 2 -- the synchronous refusals
# --------------------------------------------------------------------------------------


async def test_only_a_completed_feature_with_pull_requests_can_be_revised(tmp_path: Path) -> None:
    """A mid-run feature, and an empty request, are both refused with the feature untouched."""
    github = MockGitHubService()
    runner = FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(github_service=github)
    )
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'refusals.db'}")
    await database.create_schema()
    store = SqlAlchemyFeatureControlPlane(database, mock_runner=runner)
    try:
        request = StartFeatureRequest.model_validate(
            {**feature_payload(), "feature_id": "feature-refuses-revision"}
        )
        await store.start(
            request,
            idempotency_key="feature-refuses-revision",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        # Still pending: nothing has run, so there is nothing published to revise.
        with pytest.raises(WorkflowConflictError, match="completed"):
            await store.revise_feature(
                "feature-refuses-revision",
                request="Change it.",
                requested_by="a person",
                credentials=CREDENTIALS,
            )
        await drain_feature_queue(store)
        # Completed now -- but a blank request says nothing and is refused before queueing.
        with pytest.raises(WorkflowConflictError, match="must say what should change"):
            await store.revise_feature(
                "feature-refuses-revision",
                request="   ",
                requested_by="a person",
                credentials=CREDENTIALS,
            )
        record = await store.get_record("feature-refuses-revision")
        assert record.state.status is FeatureWorkflowStatus.COMPLETED
        assert record.state.revision == 0
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# 3 -- the whole lifecycle, through the real queue
# --------------------------------------------------------------------------------------


async def test_a_revision_ships_a_v2_pull_request_and_closes_the_superseded_one(
    tmp_path: Path,
) -> None:
    """Complete, revise, complete again: new branch, new pull request, old one closed."""
    database, store, github, feature_id = await _completed_feature(tmp_path, "lifecycle.db")
    try:
        original = (await store.get_record(feature_id)).state
        original_children = {
            repository_id: child.model_copy(deep=True)
            for repository_id, child in original.child_workflows.items()
        }
        original_pull_requests = {
            child.pull_request_artifact_id for child in original_children.values()
        }
        assert original_pull_requests, "the completed feature published nothing to supersede"

        await store.revise_feature(
            feature_id,
            request="Use a dropdown instead of radio buttons.",
            requested_by="a person",
            credentials=CREDENTIALS,
        )
        mid = (await store.get_record(feature_id)).state
        assert mid.status is FeatureWorkflowStatus.PLANNING
        assert mid.revision == 1

        await drain_feature_queue(store)
        revised = (await store.get_record(feature_id)).state
        assert revised.status is FeatureWorkflowStatus.COMPLETED
        assert revised.revision == 1

        pull_requests_by_id = {
            item.artifact_id: item
            for item in revised.artifacts
            if isinstance(item, PullRequestArtifact)
        }
        for repository_id, child in revised.child_workflows.items():
            previous = original_children[repository_id]
            assert child.branch_name == f"{previous.branch_name}-V2"
            assert child.base_branch == previous.branch_name
            assert child.status is ChildWorkflowStatus.COMPLETED
            assert child.pull_request_artifact_id is not None
            assert child.pull_request_artifact_id.endswith(".v2.json")
            replacement = pull_requests_by_id[child.pull_request_artifact_id]
            assert replacement.source_branch == child.branch_name
            assert "[V2]" in replacement.title
            # The superseded pull request was closed on the provider, with a comment
            # pointing at its replacement.
            old = pull_requests_by_id[str(previous.pull_request_artifact_id)]
            key = (old.repository, old.pull_request_number)
            assert github.states[key] == "closed"
            assert any(
                f"Superseded by V2: {replacement.url}" in comment
                for comment in github.comments[key]
            )
            # And the closure is durable: a closed copy of the old artifact exists.
            marker = pull_requests_by_id[f"{old.artifact_id.removesuffix('.json')}.closed.json"]
            assert marker.state == "closed"
            # The replacement itself is untouched on the provider.
            assert github.states[(replacement.repository, replacement.pull_request_number)] == (
                "open"
            )

        completions = [
            item for item in revised.artifacts if isinstance(item, FeatureCompletionArtifact)
        ]
        assert [item.artifact_id for item in completions] == [
            "014_feature_completion.json",
            "014_feature_completion.v2.json",
        ]
        assert int(completions[-1].metadata.get("feature_revision", 0)) == 1

        # The read model serves each provider pull request once, in its latest state: the
        # superseded one closed, its replacement open.
        served = await store.pull_requests(feature_id)
        by_number = {(item.repository, item.pull_request_number): item for item in served}
        assert len(served) == len(by_number)
        for child in revised.child_workflows.values():
            replacement = pull_requests_by_id[str(child.pull_request_artifact_id)]
            assert by_number[(replacement.repository, replacement.pull_request_number)].state == (
                "open"
            )

        # And the feature can be revised again, onto -V3.
        await store.revise_feature(
            feature_id,
            request="Actually make it a combobox.",
            requested_by="a person",
            credentials=CREDENTIALS,
        )
        await drain_feature_queue(store)
        third = (await store.get_record(feature_id)).state
        assert third.status is FeatureWorkflowStatus.COMPLETED
        assert third.revision == 2
        for child in third.child_workflows.values():
            assert child.branch_name.endswith("-V3")
            assert "-V2-V3" not in child.branch_name
            assert child.pull_request_artifact_id is not None
            assert child.pull_request_artifact_id.endswith(".v3.json")
    finally:
        await database.dispose()
