"""Publication is a person's decision when the feature did not land (task 81-).

Two rules meet here and they pull in opposite directions, which is why they are tested
together. A feature that fully landed publishes itself, unchanged -- the regression fence for
the narrowing. A feature that did not publishes nothing until somebody says so, and when they
do it opens more than the automatic path ever could: the work that passed review, and the work
that is clean and was rejected on judgement, each labelled for what it is.

The half that keeps the first rule from being a regression is the held state being loud, so
the offer itself -- advertised, refused with a sentence, at rest rather than churning -- is
asserted here rather than taken on trust.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from adapters.github_adapter import MockGitHubService
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import FileChange, PullRequestArtifact
from services.cancellation import MockCancellationToken
from services.feature_actions import action_context_version
from state.enums import FeatureWorkflowStatus
from state.feature_models import FeatureWorkflowSnapshot
from tests.test_feature_workflow import (
    FailOneRepositoryExecutor,
    HealablePullRequestService,
    OneRepositoryFailsExecutor,
    _repository,
    feature_payload,
)
from workflows.feature_workflow import (
    ChildExecution,
    FeatureWorkflowError,
    FeatureWorkflowOrchestrator,
    GitHubPullRequestPublisher,
    MockChildWorkstreamExecutor,
    WorkstreamPublicationClass,
    feature_is_at_rest,
    next_step,
    publication_is_held,
    require_publishable_feature,
    require_publishable_workstream,
)

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)


# ------------------------------------------------------------------------------------------
# A -- automatic publication narrows to the complete feature
# ------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_partial_feature_publishes_nothing_automatically() -> None:
    """Half a feature is not reviewable work, so nobody is handed half a feature unasked.

    The frontend passes review and the backend does not. Before this task the frontend's pull
    request opened on its own, which is a trap for whoever finds it: its counterpart does not
    exist. Nothing opens now -- and nothing is lost either, which is the second assertion.
    """
    held = await _run("partial-publishes-nothing", child_executor=OneRepositoryFailsExecutor())

    assert held.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert not [item for item in held.artifacts if isinstance(item, PullRequestArtifact)]
    assert all(child.pull_request_artifact_id is None for child in held.child_workflows.values())
    # And the feature says so, in the record an operator reads first.
    assert publication_is_held(held)
    assert "did not land" in (held.transition_reason or "")
    # The frontend's reviewed work is still there to publish, and the action is offered.
    require_publishable_feature(held)
    assert require_publishable_workstream(held, repository_id="frontend") is (
        WorkstreamPublicationClass.REVIEWED
    )
    # The backend is not offered: its attempt neither passed review nor recorded a clean run.
    with pytest.raises(FeatureWorkflowError):
        require_publishable_workstream(held, repository_id="backend")

    published = await FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService())
    ).publish_feature(
        held,
        requested_by="an operator (via platform key)",
        reason="the backend is not going to land",
        credentials=CREDENTIALS,
    )

    assert published.child_workflows["frontend"].pull_request_artifact_id is not None
    assert published.child_workflows["backend"].pull_request_artifact_id is None
    # Still unfinished: publishing the half that works does not make the half that failed land.
    assert published.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN


@pytest.mark.asyncio
async def test_a_complete_feature_still_publishes_automatically() -> None:
    """The regression fence. Narrowing must not reach the feature that actually landed.

    Every repository passes review, the integration gate approves, and both pull requests
    open with nobody asked for anything -- exactly as before. If this ever needs a person, the
    narrowing has swallowed the ordinary case.
    """
    completed = await _run("complete-publishes-itself")

    assert completed.status is FeatureWorkflowStatus.COMPLETED
    assert not publication_is_held(completed)
    published = {
        item.repository for item in completed.artifacts if isinstance(item, PullRequestArtifact)
    }
    assert published == {"example/backend", "example/frontend"}
    # And there is nothing left for a person to decide.
    with pytest.raises(FeatureWorkflowError):
        require_publishable_feature(completed)


# ------------------------------------------------------------------------------------------
# B -- the held feature is allowed to rest
# ------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_held_feature_is_at_rest_so_the_queue_does_not_reclaim_it() -> None:
    """The churn fence. A held feature owes a step nobody may take, which is what rest means.

    `next_step` keeps answering `publish`, because publication genuinely has not happened.
    If `feature_is_at_rest` answered the same way it did before this task, the dispatcher
    would re-claim the feature, publication would decline again, and it would do that for as
    long as anything kept asking.
    """
    from workflows.feature_workflow import FeatureStep

    held = await _run("held-feature-rests", child_executor=OneRepositoryFailsExecutor())

    assert next_step(held).step is FeatureStep.PUBLISH
    assert feature_is_at_rest(held)


@pytest.mark.asyncio
async def test_a_feature_whose_publication_broke_is_not_treated_as_held() -> None:
    """Narrower than "every stopped feature rests", and this is the case that proves it.

    A feature whose repositories all landed and whose *publication* failed is waiting on
    nobody: its missing pull requests are a step's to open, and a person's resume is what
    re-enters that step. Collapsing the two would strand it on a transient provider fault.
    """
    from tests.test_feature_workflow import FailOneRepositoryExecutor, HealablePullRequestService

    github = HealablePullRequestService("example/frontend")
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=FailOneRepositoryExecutor("never"),
        pull_request_publisher=GitHubPullRequestPublisher(
            github_service=github, retry_backoff_seconds=0.0
        ),
    )
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("publication-broke", request)
    broken = await orchestrator.start(state, credentials=CREDENTIALS)

    assert broken.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert not publication_is_held(broken), "no repository failed its review; nothing is held"

    github.healthy = True
    resumed = await orchestrator.resume(broken, answers=[], credentials=CREDENTIALS)

    assert resumed.status is FeatureWorkflowStatus.COMPLETED


# ------------------------------------------------------------------------------------------
# D -- publishing a workstream that did not pass review
# ------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_rejected_workstream_is_committed_pushed_and_opened(tmp_path: Path) -> None:
    """Class 2, end to end over real Git: commit, push, draft pull request, labelled as such.

    A rejected attempt commits nothing, so this is not "adopt a branch" -- there is no commit
    and no remote branch until the action makes them. Judged by what Git actually holds
    afterwards rather than by the calls having returned.
    """
    workspace, remote, _seed_branch = _real_checkout(tmp_path, "backend")
    rejected = await _run(
        "publish-rejected",
        child_executor=_CleanButRejectedExecutor(
            repository_id="backend", workspace_path=str(workspace)
        ),
    )
    assert rejected.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert require_publishable_workstream(rejected, repository_id="backend") is (
        WorkstreamPublicationClass.UNREVIEWED
    )
    # The tree a rejected attempt leaves behind: on the workstream's own branch, holding the
    # edits, with nothing committed and nothing pushed. That last part is the precondition the
    # whole class rests on, so it is asserted rather than assumed.
    branch = rejected.child_workflows["backend"].branch_name
    _git(workspace, "checkout", "-b", branch)
    (workspace / "server").mkdir(parents=True, exist_ok=True)
    (workspace / "server" / "app.py").write_text("# the work the reviewer rejected\n")
    assert not _git(workspace, "ls-remote", remote, branch)

    github = MockGitHubService()
    published = await FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(github_service=github),
        git_service_factory=lambda _repository, _child: _git_service(branch="main"),
    ).publish_feature(
        rejected,
        requested_by="an operator (via platform key)",
        reason="the reviewer is wrong about this one",
        credentials=CREDENTIALS,
    )

    # 1. There is a commit, and it is the tree the last attempt left behind.
    head = _git(workspace, "rev-parse", "HEAD")
    assert head
    assert "server/app.py" in _git(workspace, "show", "--name-only", "--format=", "HEAD")
    # 2. The branch reached the remote, judged from the bare repository itself.
    assert head in _git(workspace, "ls-remote", remote, branch)
    # 3. A draft pull request exists, over that exact commit.
    artifact = next(
        item
        for item in published.artifacts
        if isinstance(item, PullRequestArtifact) and item.repository == "example/backend"
    )
    assert artifact.commit_sha == head
    assert "draft" in artifact.labels
    assert "review-rejected" in artifact.labels
    # 4. Its title says what it is, because a reviewer scanning a list reads titles.
    assert "[REVIEW REJECTED]" in artifact.title
    # 5. And its body states the rejection without hedging, and quotes what was asked for.
    assert "REJECTED" in artifact.body
    assert "A person published it anyway" in artifact.body
    assert "the reviewer wanted this written another way" in artifact.body
    assert published.child_workflows["backend"].pull_request_artifact_id == artifact.artifact_id


@pytest.mark.asyncio
async def test_a_workstream_that_died_in_lint_is_refused_with_its_reason() -> None:
    """The operator's own rule: nobody overrules a failing command by decision.

    A person may overrule a *judgement*. A required check that failed is not a judgement, and
    a workstream whose lint never passed is never offered -- with the sentence a disabled
    control shows, not a bare "no".
    """
    lint_failed = await _run(
        "publish-lint-failed",
        child_executor=_CleanButRejectedExecutor(
            repository_id="backend",
            workspace_path="/workspaces/publish-lint-failed/backend",
            validation=[
                {"name": "repository lint", "required": True, "passed": False},
                {"name": "repository tests", "required": True, "passed": True},
            ],
        ),
    )

    with pytest.raises(FeatureWorkflowError) as refusal:
        require_publishable_workstream(lint_failed, repository_id="backend")

    sentence = refusal.value.diagnostics[0]
    assert "did not pass every check it requires" in sentence
    assert "repository lint" in sentence
    assert "nobody overrules a failing command" in sentence


@pytest.mark.asyncio
async def test_a_swept_worktree_is_refused_with_its_reason(tmp_path: Path) -> None:
    """The action has a shelf life, and it has to say so rather than raise from inside Git.

    A held feature is `failed_requires_human`, so the bound is
    `workspace_failed_retention_hours` (72 by default) or
    `workspace_failed_retention_features` (10), either being enough to reclaim -- roughly
    three days or ten failed features. Deliberately not `workspace_retention_features`, which
    governs completed and cancelled features only and is the setting the obvious guess names.
    A rejected attempt lives only in its worktree, so a feature left past that bound has
    nothing left to commit -- which is a real answer about this feature, and must arrive as
    one. The sentence has to name the bound, because "gone" without "for how long" gives
    somebody deciding whether to hurry nothing to act on.
    """
    workspace, _remote, branch = _real_checkout(tmp_path, "backend")
    swept = await _run(
        "publish-swept",
        child_executor=_CleanButRejectedExecutor(
            repository_id="backend", workspace_path=str(workspace), branch_name=branch
        ),
    )
    assert require_publishable_workstream(swept, repository_id="backend") is (
        WorkstreamPublicationClass.UNREVIEWED
    )

    # The retention sweep, as it leaves the volume.
    subprocess.run(("rm", "-rf", str(workspace)), check=True)

    with pytest.raises(FeatureWorkflowError) as refusal:
        require_publishable_workstream(swept, repository_id="backend")

    sentence = refusal.value.diagnostics[0]
    assert "has been reclaimed" in sentence
    # The real bound, named -- and specifically not the five-feature retention that governs
    # completed features, which is the number this sentence used to imply.
    assert "about three days" in sentence
    assert "ten most recent features" in sentence


@pytest.mark.asyncio
async def test_a_workstream_that_recorded_no_validation_is_refused() -> None:
    """No evidence is not the same as clean, and must not be published as though it were.

    An attempt that never reached validation cannot show its checks passed. Publishing it
    would put work nothing has ever run behind a pull request that says every check passed.
    """
    unproven = await _run(
        "publish-no-evidence",
        child_executor=_CleanButRejectedExecutor(
            repository_id="backend",
            workspace_path="/workspaces/publish-no-evidence/backend",
            validation=[],
        ),
    )

    with pytest.raises(FeatureWorkflowError) as refusal:
        require_publishable_workstream(unproven, repository_id="backend")

    assert "recorded no validation at all" in refusal.value.diagnostics[0]


@pytest.mark.asyncio
async def test_a_pull_request_that_opened_is_recorded_even_when_a_sibling_fails() -> None:
    """One repository's provider fault must not make the record deny the other's pull request.

    `publish` opens each repository independently and reports the failures by raising --
    carrying the artifacts for the ones that did open. Letting that propagate out of the
    publication would skip the two lines that persist them, so a draft pull request would sit
    on GitHub with this feature's record saying no pull request exists. That is the exact
    shape the publication invariant was written to prevent, reached through the path built to
    honour it.
    """
    held = await _run("publish-partial", child_executor=OneRepositoryFailsExecutor())
    # One eligible repository -- the frontend, which passed review -- and the provider refuses
    # its create. `PartialPullRequestError` then carries an empty artifact list, which is the
    # shape that must still leave a readable record rather than an exception.
    assert require_publishable_workstream(held, repository_id="frontend") is (
        WorkstreamPublicationClass.REVIEWED
    )

    github = HealablePullRequestService("example/frontend")
    orchestrator = FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(
            github_service=github, retry_backoff_seconds=0.0
        )
    )
    partial = await orchestrator.publish_feature(
        held,
        requested_by="an operator (via platform key)",
        reason="open what is finished",
        credentials=CREDENTIALS,
    )

    # Nothing raised: a publication that partly succeeded is an outcome, not an exception.
    assert partial.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    # The frontend is the one that failed here, so nothing opened at all -- and the record
    # says which repository still needs one, in its own blocking issues.
    assert partial.child_workflows["frontend"].pull_request_artifact_id is None
    assert any(
        "Pull-request creation did not succeed" in issue
        for issue in partial.child_workflows["frontend"].blocking_issues
    )
    summary = partial.failure_summary
    assert summary is not None
    assert any("did not succeed for frontend" in item for item in summary.diagnostics)
    # And the summary tells the next reader what to do about it, rather than keeping the
    # hold's advice about work that has since been opened.
    assert any("Press publish again" in item for item in summary.diagnostics)


@pytest.mark.asyncio
async def test_a_partial_publication_records_what_opened_and_what_did_not() -> None:
    """The half that opened is on the record; the half that did not names its reason.

    A doubled publisher that opens the first repository and raises on the second, which is
    the only shape that reaches `PartialPullRequestError` with a non-empty artifact list.
    """
    held = await _run_three_repositories(
        "publish-half", child_executor=FailOneRepositoryExecutor("backend")
    )
    # Two eligible repositories, both reviewed, held because the required backend never
    # landed. Exactly the shape that reaches `PartialPullRequestError` carrying one artifact.
    for repository_id in ("frontend", "worker"):
        assert require_publishable_workstream(held, repository_id=repository_id) is (
            WorkstreamPublicationClass.REVIEWED
        )

    github = HealablePullRequestService("example/worker")
    orchestrator = FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(
            github_service=github, retry_backoff_seconds=0.0
        )
    )
    partial = await orchestrator.publish_feature(
        held,
        requested_by="an operator (via platform key)",
        reason="open what is finished",
        credentials=CREDENTIALS,
    )

    # The frontend opened, and state says so -- artifact appended, child marked.
    opened = [item for item in partial.artifacts if isinstance(item, PullRequestArtifact)]
    assert [item.repository for item in opened] == ["example/frontend"]
    assert partial.child_workflows["frontend"].pull_request_artifact_id == opened[0].artifact_id
    # The worker did not, and the outcome sentence names what actually opened rather than
    # what was asked for -- naming a repository whose create failed would be the same false
    # record this handler exists to prevent.
    assert partial.child_workflows["worker"].pull_request_artifact_id is None
    assert "example/frontend" in (partial.transition_reason or "")
    assert "did not succeed for worker" in (partial.transition_reason or "")
    assert "example/worker" not in (partial.transition_reason or "")
    assert any(
        "Pull-request creation did not succeed" in issue
        for issue in partial.child_workflows["worker"].blocking_issues
    )

    # A second press finishes the job rather than duplicating it: the frontend is refused
    # because it already has a pull request, and the worker now reaches one.
    github.healthy = True
    with pytest.raises(FeatureWorkflowError, match="already has a pull request"):
        require_publishable_workstream(partial, repository_id="frontend")
    finished = await orchestrator.publish_feature(
        partial,
        requested_by="an operator (via platform key)",
        reason="the provider is back",
        credentials=CREDENTIALS,
    )

    assert finished.child_workflows["worker"].pull_request_artifact_id is not None
    assert len(github.pull_requests) == 2, "the frontend must not be opened a second time"


@pytest.mark.asyncio
async def test_a_create_that_reached_the_provider_is_adopted_not_duplicated() -> None:
    """The orphan case: the create raised and the pull request exists anyway.

    This is what makes a retry of a partial publication safe. `_create_or_adopt` looks the
    pull request up before retrying, so a fault that happened *after* the provider committed
    is adopted rather than turned into a second pull request for the same branch -- which the
    provider would refuse forever.
    """
    held = await _run("publish-orphan", child_executor=OneRepositoryFailsExecutor())

    github = _CreatesThenRaises("example/frontend")
    orchestrator = FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(
            github_service=github, retry_backoff_seconds=0.0
        )
    )
    published = await orchestrator.publish_feature(
        held,
        requested_by="an operator (via platform key)",
        reason="open what is finished",
        credentials=CREDENTIALS,
    )

    # One create call, one pull request, and the record holds it: the raise did not turn a
    # committed provider effect into an orphan the platform denies.
    assert github.create_calls == 1
    assert len(github.pull_requests) == 1
    assert published.child_workflows["frontend"].pull_request_artifact_id is not None


# ------------------------------------------------------------------------------------------
# C -- the action's identity
# ------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_presses_are_one_decision_until_a_repository_moves() -> None:
    """A double-click is a replay; a press after a retry moved a repository is a new decision.

    The identity is what would be published and what it would be published from, so it must
    not move for a reason that changes neither -- the attempt counters the retry key uses do
    exactly that, and are deliberately absent.
    """
    held = await _run("publish-identity", child_executor=OneRepositoryFailsExecutor())

    first = action_context_version(held, "PUBLISH_FEATURE", {})
    assert first == action_context_version(held, "PUBLISH_FEATURE", {})
    assert first is not None and first.startswith("publish:")

    # Another spent attempt that moved nothing publishable is the same decision.
    spent = held.model_copy(deep=True)
    child = spent.child_workflows["backend"]
    spent.child_workflows["backend"] = child.model_copy(
        update={"retry_count": child.retry_count + 1, "granted_extra_attempts": 1}
    )
    assert action_context_version(spent, "PUBLISH_FEATURE", {}) == first

    # A retry that moved the repository's revision is a different one, and may execute.
    moved = held.model_copy(deep=True)
    moved.child_workflows["backend"] = moved.child_workflows["backend"].model_copy(
        update={"current_revision": "0f0f0f0f"}
    )
    assert action_context_version(moved, "PUBLISH_FEATURE", {}) != first

    # And so is the same feature after one repository has been published.
    opened = held.model_copy(deep=True)
    opened.child_workflows["frontend"] = opened.child_workflows["frontend"].model_copy(
        update={"pull_request_artifact_id": "008_pull_request.frontend.json"}
    )
    assert action_context_version(opened, "PUBLISH_FEATURE", {}) != first


@pytest.mark.asyncio
async def test_a_publication_is_accepted_and_queued_and_the_worker_runs_it(
    tmp_path: Path,
) -> None:
    """Not inside the request. A push and a provider call per repository run on a worker.

    AB-Feature-108 lost seventy minutes to one cancelled socket doing exactly this shape of
    work on a request thread. What arrives synchronously is the refusal; the work is a queue
    entry, and the durable path from that entry to the pull request is what this drives.
    """
    from storage.db import Database
    from storage.feature_store import SqlAlchemyFeatureControlPlane
    from tests.support import drain_feature_queue

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'publish.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(
            database,
            mock_runner=FeatureWorkflowOrchestrator(
                child_executor=cast(Any, OneRepositoryFailsExecutor()),
                pull_request_publisher=GitHubPullRequestPublisher(
                    github_service=MockGitHubService()
                ),
            ),
        )
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="queued-publish-001",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        held = (await store.get_record("feature-contract-pause")).state
        assert publication_is_held(held)
        assert all(item.pull_request_artifact_id is None for item in held.child_workflows.values())

        await store.publish_feature(
            "feature-contract-pause",
            requested_by="akhilesh (via platform key)",
            reason="the backend is not going to land",
            credentials=CREDENTIALS,
        )

        # Accepted, and nothing has run: the entry is the work.
        assert all(
            item.pull_request_artifact_id is None
            for item in (
                await store.get_record("feature-contract-pause")
            ).state.child_workflows.values()
        )
        claimed = await store.queue.claim(owner="worker", lease_seconds=60)
        assert claimed is not None
        assert claimed.intent == "publish"
        assert claimed.payload == {
            "requested_by": "akhilesh (via platform key)",
            "reason": "the backend is not going to land",
        }
        # The identity the worker resolves stored credentials against stays the submission's,
        # not the audit sentence: crossing the two killed AB-Feature-111's granted retry.
        assert claimed.requested_by != "akhilesh (via platform key)"
        await store.queue.release(claimed.feature_id)
        await drain_feature_queue(store)

        published = (await store.get_record("feature-contract-pause")).state
        assert published.child_workflows["frontend"].pull_request_artifact_id is not None
        assert published.child_workflows["backend"].pull_request_artifact_id is None
        assert published.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    finally:
        await database.dispose()


# ------------------------------------------------------------------------------------------
# Scaffolding
# ------------------------------------------------------------------------------------------


async def _run(feature_id: str, **orchestrator_kwargs: Any) -> FeatureWorkflowSnapshot:
    """Run one deterministic two-repository feature to rest and return its snapshot."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state(feature_id, request)
    orchestrator = FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
        **orchestrator_kwargs,
    )
    return await orchestrator.start(state, credentials=CREDENTIALS)


async def _run_three_repositories(
    feature_id: str, **orchestrator_kwargs: Any
) -> FeatureWorkflowSnapshot:
    """Run a three-repository feature, so two of them can be publishable at once.

    The two-repository payload cannot express a partial publication: holding the feature
    requires one required repository to have failed, which leaves exactly one publishable
    sibling and nothing for a provider fault to be partial about.
    """
    payload = feature_payload()
    payload["feature_id"] = feature_id
    payload["repositories"] = [
        _repository("backend", "Backend"),
        _repository("frontend", "Frontend"),
        _repository("worker", "Worker"),
    ]
    request = StartFeatureRequest.model_validate(payload)
    state = _initial_feature_state(feature_id, request)
    orchestrator = FeatureWorkflowOrchestrator(
        pull_request_publisher=GitHubPullRequestPublisher(github_service=MockGitHubService()),
        **orchestrator_kwargs,
    )
    return await orchestrator.start(state, credentials=CREDENTIALS)


def _git_service(*, branch: str) -> Any:
    """Build the real Git adapter, with no journal and no credential, for a local remote."""
    from adapters.interruptible_git import InterruptibleGitService

    return InterruptibleGitService(
        default_branch=branch, cancellation_token=MockCancellationToken()
    )


def _real_checkout(tmp_path: Path, repository_id: str) -> tuple[Path, str, str]:
    """Create one real checkout on a feature branch, with a bare remote behind it."""
    workspace = tmp_path / repository_id
    workspace.mkdir(parents=True)
    remote = tmp_path / f"{repository_id}.git"
    subprocess.run(("git", "init", "--bare", str(remote)), check=True, capture_output=True)
    _git(workspace, "init", "-b", "main")
    _git(workspace, "config", "user.email", "platform@example.com")
    _git(workspace, "config", "user.name", "Platform")
    (workspace / "README.md").write_text("# fixture\n")
    _git(workspace, "add", "README.md")
    _git(workspace, "commit", "-m", "initial")
    _git(workspace, "remote", "add", "origin", str(remote))
    _git(workspace, "push", "origin", "main")
    branch = f"ai/{repository_id}/rejected-work"
    _git(workspace, "checkout", "-b", branch)
    return workspace, str(remote), branch


def _git(root: Path, *arguments: str) -> str:
    """Run one git command against a fixture checkout and return its stdout."""
    finished = subprocess.run(
        ("git", *arguments), cwd=root, check=True, capture_output=True, timeout=30, text=True
    )
    return finished.stdout.strip()


class _CleanButRejectedExecutor:
    """Fail one repository's review while every check it requires passes.

    The shape class 2 exists for: lint green, tests green, and a reviewer who said no on
    judgement. Everything else about the result is the mock executor's own.
    """

    def __init__(
        self,
        *,
        repository_id: str,
        workspace_path: str,
        branch_name: str | None = None,
        validation: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record which repository is rejected, where its tree is, and what its checks said."""
        self._delegate = MockChildWorkstreamExecutor()
        self._repository_id = repository_id
        self._workspace_path = workspace_path
        self._branch_name = branch_name
        self._validation = (
            validation
            if validation is not None
            else [
                {"name": "repository lint", "required": True, "passed": True},
                {"name": "repository tests", "required": True, "passed": True},
            ]
        )

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Return an approved result for every repository except the rejected one."""
        execution = await self._delegate.run(**kwargs)
        if kwargs["repository"].repository_id != self._repository_id:
            return execution
        update: dict[str, Any] = {
            "status": "failed",
            "pull_request_readiness": False,
            "blocking_issues": ["the reviewer wanted this written another way"],
            "failure_classification": "review_scope_failure",
            "changed_files": [
                FileChange(
                    path="server/app.py",
                    change_type="modified",
                    description="the work the reviewer rejected",
                )
            ],
            "current_validation_results": self._validation,
            "workspace_path": self._workspace_path,
            "production_diff_fingerprint": "rejected-but-clean",
        }
        if self._branch_name is not None:
            update["branch_name"] = self._branch_name
        failed = execution.result.model_copy(update=update)
        return ChildExecution(result=failed, code_completion=execution.code_completion)


class _CreatesThenRaises(MockGitHubService):
    """Commit the pull request to the provider and then fail the call that made it.

    The window every retry-then-adopt path exists for: the effect landed and the caller was
    told it did not. A provider that has the pull request refuses the next create for the
    same branch forever, so retrying blindly turns a transient fault into a permanent one.
    """

    def __init__(self, repository: str) -> None:
        """Fail exactly the named repository's first create, after it has taken effect."""
        super().__init__()
        self._repository = repository
        self.create_calls = 0

    def create_pull_request(self, repository: str, **kwargs: Any) -> Any:
        """Record the pull request, then raise as a provider does when it answers late."""
        created = super().create_pull_request(repository, **kwargs)
        if repository != self._repository:
            return created
        self.create_calls += 1
        raise RuntimeError("simulated provider fault after the create took effect")
