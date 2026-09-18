"""Safe pull-request publication node with an explicit mock default provider."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

from adapters.github_adapter import MockGitHubService
from agents.shared.contracts import (
    ARTIFACT_FILENAMES,
    AgentArtifactError,
    artifact_json,
    artifact_update,
    create_artifact,
    require_artifact,
)
from artifacts.schemas import CodeCompletionArtifact, PullRequestArtifact, ReviewArtifact
from prompts.prompt_loader import PromptLoader
from services.cancellation import CancellationToken
from state.models import AgentState


class GitHubAgent:
    """Create a pull-request artifact through an explicitly configured GitHub service."""

    def __init__(
        self,
        *,
        prompt_loader: PromptLoader,
        github_service: Any | None = None,
        labels: Sequence[str] = ("automated",),
        reviewers: Sequence[str] = (),
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        """Use a no-network mock provider unless a production provider is explicitly injected."""
        self._prompt_loader = prompt_loader
        self._github_service: Any = github_service or MockGitHubService()
        self._labels = _distinct_nonempty(labels)
        self._reviewers = _distinct_nonempty(reviewers)
        self._cancellation_token = cancellation_token

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Create a pull request only after completed implementation and approved review."""
        code_completion = require_artifact(
            state,
            CodeCompletionArtifact,
            artifact_id=ARTIFACT_FILENAMES["code_completion"],
        )
        review = require_artifact(state, ReviewArtifact, artifact_id=ARTIFACT_FILENAMES["review"])
        if code_completion.completion_status != "completed":
            msg = "pull-request creation requires a completed code-completion artifact"
            raise AgentArtifactError(msg)
        if review.verdict != "approved":
            msg = "pull-request creation requires an approved review artifact"
            raise AgentArtifactError(msg)
        if code_completion.commit_sha is None:
            msg = "pull-request creation requires a commit SHA in 006_code_completion.json"
            raise AgentArtifactError(msg)

        workspace = state["workspace_descriptor"]
        repository = _repository_name(workspace.source_repo_url)
        title = f"Workflow {state['workflow_id']}: {code_completion.summary}"
        body = self._prompt_loader.render(
            "github/v1.jinja2",
            workflow_id=state["workflow_id"],
            code_completion=artifact_json(code_completion),
            review=artifact_json(review),
        )
        await self._raise_if_cancelled()
        pull_request: Any = await _await_if_needed(
            self._github_service.create_pull_request(
                repository,
                title=title,
                body=body,
                source_branch=workspace.working_branch,
                target_branch=workspace.default_branch,
                expected_head_sha=code_completion.commit_sha,
            )
        )
        finder = getattr(self._github_service, "find_pull_request", None)
        if finder is None:
            msg = "GitHub service cannot fetch a created pull request for verification"
            raise AgentArtifactError(msg)
        fetched: Any = await _await_if_needed(
            finder(
                repository,
                source_branch=workspace.working_branch,
                target_branch=workspace.default_branch,
                title=title,
            )
        )
        if fetched is None:
            msg = "created pull request could not be fetched back"
            raise AgentArtifactError(msg)
        if (
            fetched.repository != repository
            or fetched.number != pull_request.number
            or fetched.source_branch != workspace.working_branch
            or fetched.target_branch != workspace.default_branch
            or fetched.title != title
        ):
            msg = "fetched pull request does not match the created pull request"
            raise AgentArtifactError(msg)
        if fetched.head_sha != code_completion.commit_sha:
            msg = "fetched pull-request head does not match the approved commit"
            raise AgentArtifactError(msg)
        await self._raise_if_cancelled()
        await _await_if_needed(
            self._github_service.add_labels(repository, pull_request.number, self._labels)
        )
        await self._raise_if_cancelled()
        await _await_if_needed(
            self._github_service.request_reviewers(repository, pull_request.number, self._reviewers)
        )
        await self._raise_if_cancelled()
        await _await_if_needed(
            self._github_service.add_comment(
                repository,
                pull_request.number,
                f"Workflow {state['workflow_id']} review verdict: {review.verdict}.",
            )
        )
        pull_request_artifact = create_artifact(
            PullRequestArtifact,
            workflow_id=state["workflow_id"],
            artifact_id=ARTIFACT_FILENAMES["pull_request"],
            producer="github",
            payload={
                "repository": fetched.repository,
                "pull_request_number": fetched.number,
                "url": fetched.url,
                "title": fetched.title,
                "body": body,
                "source_branch": pull_request.source_branch,
                "target_branch": pull_request.target_branch,
                "commit_sha": code_completion.commit_sha,
                "labels": list(self._labels),
                "reviewers": list(self._reviewers),
                "state": "open",
            },
            metadata={
                "source_artifact_ids": [code_completion.artifact_id, review.artifact_id],
                "provider": type(self._github_service).__name__,
                "prompt_template": "github/v1.jinja2",
                "verified_by_fetch": True,
                "verified_head_sha": fetched.head_sha,
            },
        )
        return artifact_update("github", [pull_request_artifact])

    async def _raise_if_cancelled(self) -> None:
        """Prevent a cancellation request from beginning a later irreversible PR mutation."""
        if self._cancellation_token is not None:
            await self._cancellation_token.raise_if_cancelled()


async def github_node(
    state: AgentState,
    *,
    prompt_loader: PromptLoader,
    github_service: Any | None = None,
    labels: Sequence[str] = ("automated",),
    reviewers: Sequence[str] = (),
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """Run a GitHub node with a mock provider by default and live service only by injection."""
    return await GitHubAgent(
        prompt_loader=prompt_loader,
        github_service=github_service,
        labels=labels,
        reviewers=reviewers,
        cancellation_token=cancellation_token,
    ).run(state)


async def _await_if_needed(value: object) -> object:
    """Accept the existing synchronous mock adapter and the async journaled live adapter."""
    return await value if inspect.isawaitable(value) else value


def _repository_name(source_repo_url: str) -> str:
    """Extract ``owner/repository`` from HTTPS or SSH repository URLs without credentials."""
    parsed = urlparse(source_repo_url)
    if parsed.scheme:
        path = parsed.path
    elif ":" in source_repo_url:
        path = source_repo_url.rsplit(":", maxsplit=1)[1]
    else:
        path = source_repo_url
    parts = [part for part in path.strip("/").removesuffix(".git").split("/") if part]
    if len(parts) < 2:
        msg = "workspace source_repo_url must identify an owner and repository"
        raise AgentArtifactError(msg)
    return "/".join(parts[-2:])


def _distinct_nonempty(values: Sequence[str]) -> tuple[str, ...]:
    """Return configured labels or reviewers in deterministic insertion order."""
    return tuple(dict.fromkeys(value for value in values if value.strip()))
