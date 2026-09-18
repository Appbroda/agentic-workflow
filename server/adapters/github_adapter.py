"""GitHub service protocol with live PyGithub and deterministic mock implementations."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class GitHubAdapterError(RuntimeError):
    """Raised when a GitHub operation cannot be completed safely.

    Carries ``diagnostics`` only when the raiser built them from platform-owned operation
    names, the endpoint this adapter chose to call, and the provider's HTTP status. A
    provider message is never quoted: it can carry repository or credential detail.
    """

    def __init__(
        self,
        *args: object,
        diagnostics: Sequence[str] = (),
        failure_classification: str | None = None,
        provider_status: int | None = None,
    ) -> None:
        """Record safe diagnostics, a provider-owned failure category, and its HTTP status."""
        super().__init__(*args)
        self.diagnostics = tuple(diagnostics)
        self.failure_classification = (
            failure_classification.strip()
            if isinstance(failure_classification, str) and failure_classification.strip()
            else None
        )
        # The integer the provider reported, never its message. Carried so the platform's one
        # retryability predicate can tell an answer (a 4xx, identical on every retry) from
        # weather (a 5xx, or no answer at all) without parsing the classification token back
        # apart. ``None`` where the provider was never reached or reported nothing.
        self.provider_status = provider_status


@dataclass(frozen=True, slots=True)
class PullRequestDetails:
    """The stable pull-request data needed by workflow artifacts and later operations."""

    repository: str
    number: int
    url: str
    title: str
    source_branch: str
    target_branch: str
    head_sha: str | None = None


class GitHubService(Protocol):
    """Protocol for the GitHub operations required by the GitHub workflow agent."""

    def create_pull_request(
        self,
        repository: str,
        *,
        title: str,
        body: str,
        source_branch: str,
        target_branch: str,
        draft: bool = False,
        expected_head_sha: str | None = None,
    ) -> PullRequestDetails:
        """Create a pull request from a safe source branch to a target branch."""

    def add_labels(self, repository: str, pull_request_number: int, labels: Sequence[str]) -> None:
        """Attach labels to an existing pull request."""

    def request_reviewers(
        self, repository: str, pull_request_number: int, reviewers: Sequence[str]
    ) -> None:
        """Request reviews from GitHub users or teams."""

    def add_comment(self, repository: str, pull_request_number: int, body: str) -> None:
        """Add a workflow summary comment to an existing pull request."""

    def find_pull_request(
        self,
        repository: str,
        *,
        source_branch: str,
        target_branch: str,
        title: str,
    ) -> PullRequestDetails | None:
        """Find an existing workflow PR so an interrupted create is never duplicated."""

    def find_pull_requests(
        self,
        repository: str,
        *,
        source_branch: str,
        target_branch: str,
    ) -> Sequence[PullRequestDetails]:
        """Return every open PR on this head and base, so ambiguity can be seen not guessed."""

    def get_pull_request_labels(self, repository: str, pull_request_number: int) -> Sequence[str]:
        """Read the labels actually on a pull request, for reconciliation after interruption."""

    def get_requested_reviewers(self, repository: str, pull_request_number: int) -> Sequence[str]:
        """Read the reviewers still requested on a pull request."""

    def get_pull_request_comments(self, repository: str, pull_request_number: int) -> Sequence[str]:
        """Read the conversation comments on a pull request, so a failed post can be proven."""

    def get_pull_request_state(self, repository: str, pull_request_number: int) -> str:
        """Read whether a pull request is open, closed, or merged, for close reconciliation."""

    def close_pull_request(self, repository: str, pull_request_number: int) -> None:
        """Close a superseded pull request; closing one already closed or merged is a no-op."""


def select_pull_request(
    candidates: Sequence[PullRequestDetails],
    *,
    title: str | None = None,
    expected_head_sha: str | None = None,
) -> PullRequestDetails | None:
    """Choose at most one candidate from stable fields, and never by resemblance.

    Head and base already identify the branch this platform pushed, so they are the match.
    Title is model-influenced text and is only a tiebreaker among candidates that already
    share a head and a base; an expected head SHA, where one is known, outranks it. If more
    than one candidate survives, this returns nothing rather than picking one -- guessing
    here is how a repository ends up with a second pull request for the same branch.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    for narrowed in (
        [item for item in candidates if expected_head_sha and item.head_sha == expected_head_sha],
        [item for item in candidates if title is not None and item.title == title],
    ):
        if len(narrowed) == 1:
            return narrowed[0]
    return None


class MockGitHubService:
    """In-memory GitHub implementation for deterministic tests and local workflow runs."""

    def __init__(self) -> None:
        """Initialize the pull-request records managed by this mock service."""
        self._next_pull_request_number = 1
        self.pull_requests: dict[tuple[str, int], PullRequestDetails] = {}
        self.labels: dict[tuple[str, int], list[str]] = {}
        self.reviewers: dict[tuple[str, int], list[str]] = {}
        self.comments: dict[tuple[str, int], list[str]] = {}
        self.states: dict[tuple[str, int], str] = {}

    def create_pull_request(
        self,
        repository: str,
        *,
        title: str,
        body: str,
        source_branch: str,
        target_branch: str,
        draft: bool = False,
        expected_head_sha: str | None = None,
    ) -> PullRequestDetails:
        """Create and retain a local pull-request representation without network access."""
        _validate_pull_request_input(repository, title, body, source_branch, target_branch)
        if not isinstance(draft, bool):
            msg = "draft must be a boolean"
            raise GitHubAdapterError(msg)
        number = self._next_pull_request_number
        self._next_pull_request_number += 1
        pull_request = PullRequestDetails(
            repository=repository,
            number=number,
            url=f"https://github.invalid/{repository}/pull/{number}",
            title=title,
            source_branch=source_branch,
            target_branch=target_branch,
            head_sha=expected_head_sha,
        )
        key = (repository, number)
        self.pull_requests[key] = pull_request
        self.labels[key] = []
        self.reviewers[key] = []
        self.comments[key] = []
        self.states[key] = "open"
        return pull_request

    def add_labels(self, repository: str, pull_request_number: int, labels: Sequence[str]) -> None:
        """Record distinct labels on an existing mock pull request."""
        key = self._require_pull_request(repository, pull_request_number)
        self.labels[key] = _merge_distinct(self.labels[key], labels)

    def request_reviewers(
        self, repository: str, pull_request_number: int, reviewers: Sequence[str]
    ) -> None:
        """Record distinct review requests on an existing mock pull request."""
        key = self._require_pull_request(repository, pull_request_number)
        self.reviewers[key] = _merge_distinct(self.reviewers[key], reviewers)

    def add_comment(self, repository: str, pull_request_number: int, body: str) -> None:
        """Record a non-empty pull-request summary comment."""
        if not body.strip():
            msg = "pull-request comments must not be empty"
            raise GitHubAdapterError(msg)
        key = self._require_pull_request(repository, pull_request_number)
        self.comments[key].append(body)

    def find_pull_request(
        self,
        repository: str,
        *,
        source_branch: str,
        target_branch: str,
        title: str,
    ) -> PullRequestDetails | None:
        """Find an existing local pull request from its head and base, title as a tiebreaker."""
        return select_pull_request(
            self.find_pull_requests(
                repository, source_branch=source_branch, target_branch=target_branch
            ),
            title=title,
        )

    def find_pull_requests(
        self,
        repository: str,
        *,
        source_branch: str,
        target_branch: str,
    ) -> Sequence[PullRequestDetails]:
        """Return every open local pull request on this head and base, as the live adapter does."""
        return [
            details
            for details in self.pull_requests.values()
            if details.repository == repository
            and details.source_branch == source_branch
            and details.target_branch == target_branch
            and self.states.get((details.repository, details.number), "open") == "open"
        ]

    def get_pull_request_labels(self, repository: str, pull_request_number: int) -> Sequence[str]:
        """Return the labels recorded on an existing mock pull request."""
        key = self._require_pull_request(repository, pull_request_number)
        return list(self.labels[key])

    def get_requested_reviewers(self, repository: str, pull_request_number: int) -> Sequence[str]:
        """Return the review requests recorded on an existing mock pull request."""
        key = self._require_pull_request(repository, pull_request_number)
        return list(self.reviewers[key])

    def get_pull_request_comments(self, repository: str, pull_request_number: int) -> Sequence[str]:
        """Return the comment bodies recorded on an existing mock pull request."""
        key = self._require_pull_request(repository, pull_request_number)
        return list(self.comments[key])

    def get_pull_request_state(self, repository: str, pull_request_number: int) -> str:
        """Return the recorded state of an existing mock pull request."""
        key = self._require_pull_request(repository, pull_request_number)
        return self.states.get(key, "open")

    def close_pull_request(self, repository: str, pull_request_number: int) -> None:
        """Close an existing mock pull request; closing again, or when merged, is a no-op."""
        key = self._require_pull_request(repository, pull_request_number)
        if self.states.get(key, "open") == "open":
            self.states[key] = "closed"

    def _require_pull_request(self, repository: str, pull_request_number: int) -> tuple[str, int]:
        """Return an existing pull-request key or raise a useful adapter error."""
        key = (repository, pull_request_number)
        if key not in self.pull_requests:
            msg = f"pull request {pull_request_number} does not exist in {repository}"
            raise GitHubAdapterError(msg)
        return key


class PyGithubService:
    """Live GitHub implementation that delegates only to an injected PyGithub client."""

    def __init__(self, token: str, *, client: Any | None = None) -> None:
        """Configure a client from an injected token; tests can supply a mocked client."""
        if not token.strip():
            msg = "GitHub token must not be empty"
            raise GitHubAdapterError(msg)
        self._client = client if client is not None else _create_pygithub_client(token)

    def create_pull_request(
        self,
        repository: str,
        *,
        title: str,
        body: str,
        source_branch: str,
        target_branch: str,
        draft: bool = False,
        expected_head_sha: str | None = None,
    ) -> PullRequestDetails:
        """Create a pull request through PyGithub without storing credentials in workflow data."""
        _validate_pull_request_input(repository, title, body, source_branch, target_branch)
        arguments: dict[str, Any] = {
            "title": title,
            "body": body,
            "head": source_branch,
            "base": target_branch,
        }
        if draft:
            arguments["draft"] = True
        pull_request = self._client.get_repo(repository).create_pull(**arguments)
        return PullRequestDetails(
            repository=repository,
            number=int(pull_request.number),
            url=str(pull_request.html_url),
            title=str(pull_request.title),
            source_branch=source_branch,
            target_branch=target_branch,
            head_sha=(
                str(head_sha)
                if (head_sha := getattr(getattr(pull_request, "head", None), "sha", None))
                else None
            ),
        )

    def add_labels(self, repository: str, pull_request_number: int, labels: Sequence[str]) -> None:
        """Attach supplied labels to the selected pull request."""
        if labels:
            pull_request = self._client.get_repo(repository).get_pull(pull_request_number)
            pull_request.add_to_labels(*labels)

    def request_reviewers(
        self, repository: str, pull_request_number: int, reviewers: Sequence[str]
    ) -> None:
        """Request supplied reviewers for the selected pull request."""
        if reviewers:
            pull_request = self._client.get_repo(repository).get_pull(pull_request_number)
            pull_request.create_review_request(reviewers=list(reviewers))

    def add_comment(self, repository: str, pull_request_number: int, body: str) -> None:
        """Create a non-empty summary comment on the selected pull request.

        Reached through ``get_pull``, not ``get_repo(...).get_issue(...)``. Both post the
        comment to the same place -- GitHub serves pull-request conversation comments from
        the Issues API -- but the issue *read* is an extra call on an endpoint the Issues
        permission gates, and it is where every cross-link comment this platform has ever
        attempted died with a 403 before the write was even tried. Reading the pull request
        needs only the pull-request access this adapter already demonstrates by opening the
        pull request in the first place.

        The two remaining calls are wrapped separately so a refusal names the one that was
        actually refused. Blaming the write for a failed read is how this took ten features
        to diagnose.
        """
        if not body.strip():
            msg = "pull-request comments must not be empty"
            raise GitHubAdapterError(msg)
        try:
            pull_request = self._client.get_repo(repository).get_pull(pull_request_number)
        except Exception as error:  # noqa: BLE001 - reclassified into safe diagnostics below
            raise _comment_failure(
                error,
                endpoint=f"GET /repos/{repository}/pulls/{pull_request_number}",
                remedy=_PULL_REQUEST_READ_REMEDY,
            ) from error
        try:
            pull_request.create_issue_comment(body)
        except Exception as error:  # noqa: BLE001 - reclassified into safe diagnostics below
            raise _comment_failure(
                error,
                endpoint=f"POST /repos/{repository}/issues/{pull_request_number}/comments",
                remedy=_COMMENT_WRITE_REMEDY,
            ) from error

    def find_pull_request(
        self,
        repository: str,
        *,
        source_branch: str,
        target_branch: str,
        title: str,
    ) -> PullRequestDetails | None:
        """Search open PRs by head and base, with the title only as a tiebreaker."""
        return select_pull_request(
            self.find_pull_requests(
                repository, source_branch=source_branch, target_branch=target_branch
            ),
            title=title,
        )

    def find_pull_requests(
        self,
        repository: str,
        *,
        source_branch: str,
        target_branch: str,
    ) -> Sequence[PullRequestDetails]:
        """Return every open pull request on this head and base without deciding between them."""
        pulls = self._client.get_repo(repository).get_pulls(state="open", base=target_branch)
        matches: list[PullRequestDetails] = []
        for pull_request in pulls:
            if getattr(getattr(pull_request, "head", None), "ref", None) != source_branch:
                continue
            matches.append(
                PullRequestDetails(
                    repository=repository,
                    number=int(pull_request.number),
                    url=str(pull_request.html_url),
                    title=str(pull_request.title),
                    source_branch=source_branch,
                    target_branch=target_branch,
                    head_sha=(
                        str(head_sha)
                        if (head_sha := getattr(getattr(pull_request, "head", None), "sha", None))
                        else None
                    ),
                )
            )
        return matches

    def get_pull_request_labels(self, repository: str, pull_request_number: int) -> Sequence[str]:
        """Read the labels currently attached to a pull request."""
        pull_request = self._client.get_repo(repository).get_pull(pull_request_number)
        return [str(label.name) for label in pull_request.get_labels()]

    def get_requested_reviewers(self, repository: str, pull_request_number: int) -> Sequence[str]:
        """Read the users and teams whose review is still outstanding on a pull request."""
        pull_request = self._client.get_repo(repository).get_pull(pull_request_number)
        users, teams = pull_request.get_review_requests()
        return [str(user.login) for user in users] + [str(team.slug) for team in teams]

    def get_pull_request_comments(self, repository: str, pull_request_number: int) -> Sequence[str]:
        """Read the conversation comments on a pull request.

        Through ``get_pull`` for the same reason ``add_comment`` posts through it: GitHub
        serves these from the Issues API, but reading the pull request needs only the
        pull-request access this adapter already demonstrates by opening one at all.
        """
        pull_request = self._client.get_repo(repository).get_pull(pull_request_number)
        return [str(comment.body) for comment in pull_request.get_issue_comments()]

    def get_pull_request_state(self, repository: str, pull_request_number: int) -> str:
        """Read a pull request's lifecycle state, distinguishing merged from merely closed.

        GitHub reports a merged pull request's ``state`` as ``closed``; the ``merged`` flag is
        what tells the two apart, and a revision must never report the work it superseded as
        discarded when a person in fact merged it.
        """
        pull_request = self._client.get_repo(repository).get_pull(pull_request_number)
        if bool(getattr(pull_request, "merged", False)):
            return "merged"
        return str(pull_request.state)

    def close_pull_request(self, repository: str, pull_request_number: int) -> None:
        """Close a superseded pull request; one already closed or merged is left alone.

        The read and the write are wrapped separately, as in ``add_comment``, so a refusal
        names the call that was actually refused.
        """
        try:
            pull_request = self._client.get_repo(repository).get_pull(pull_request_number)
        except Exception as error:  # noqa: BLE001 - reclassified into safe diagnostics below
            raise _close_failure(
                error,
                endpoint=f"GET /repos/{repository}/pulls/{pull_request_number}",
                remedy=_PULL_REQUEST_READ_REMEDY,
            ) from error
        if str(pull_request.state) != "open":
            return
        try:
            pull_request.edit(state="closed")
        except Exception as error:  # noqa: BLE001 - reclassified into safe diagnostics below
            raise _close_failure(
                error,
                endpoint=f"PATCH /repos/{repository}/pulls/{pull_request_number}",
                remedy=_PULL_REQUEST_CLOSE_REMEDY,
            ) from error


def _provider_status(error: BaseException) -> int | None:
    """Return the provider's HTTP status when it reported one as an integer."""
    status = getattr(error, "status", None)
    if isinstance(status, bool) or not isinstance(status, int):
        return None
    return status


# What the grant probably is when the answer was a refusal, one per call the comment makes.
# GitHub's fine-grained permission table lists `POST /issues/{n}/comments` under both the
# `Issues` and the `Pull requests` repository permissions, so either write grant permits it;
# the pull-request read is the one this platform already demonstrates by opening PRs at all.
_COMMENT_WRITE_REMEDY = (
    "the credential may not write pull-request comments on this repository -- GitHub serves "
    "them from the Issues API, so a fine-grained token needs Pull requests: write "
    "(Issues: write also grants it)"
)
_PULL_REQUEST_READ_REMEDY = (
    "the credential may not read this pull request, which the same token opened -- a "
    "fine-grained token needs Pull requests: read on this repository"
)
_PULL_REQUEST_CLOSE_REMEDY = (
    "the credential may not edit pull requests on this repository -- a fine-grained token "
    "needs Pull requests: write to close one"
)


def _comment_failure(error: BaseException, *, endpoint: str, remedy: str) -> GitHubAdapterError:
    """Wrap a failed comment so the control plane learns the cause, not just the type.

    Every cross-link comment in this platform's history failed, and every record of it read
    "GithubException" with empty diagnostics -- a hundred-per-cent failure that could not be
    diagnosed from the journal because the one detail that identifies it, the HTTP status,
    was thrown away at the boundary. The status is an integer and the endpoint is this
    adapter's own choice, so both are safe to record where a provider message is not.

    A cause is stated as a possibility and never as a finding: the response says the call was
    refused, not why, and a wrong cause asserted as fact costs more than an honest "probably".
    """
    status = _provider_status(error)
    described_status = str(status) if status is not None else "unreported"
    diagnostics = [
        f"pull-request comment refused ({type(error).__name__} status={described_status}) "
        f"on {endpoint}"
    ]
    if status == 401:
        diagnostics.append(
            "likely cause: the credential was rejected outright (expired or revoked token); "
            "not confirmed by this response alone"
        )
    elif status == 403:
        diagnostics.append(f"likely cause: {remedy}; not confirmed by this response alone")
    elif status == 404:
        # GitHub answers 404 rather than 403 where admitting the resource exists would leak
        # its existence, so a missing read grant and a genuinely absent pull request are the
        # same answer here.
        diagnostics.append(
            "likely cause: the pull request is not visible to this credential, which GitHub "
            "also reports as 404 when read access is missing; not confirmed by this response "
            "alone"
        )
    msg = f"pull-request comment could not be posted (status {described_status})"
    return GitHubAdapterError(
        msg,
        diagnostics=diagnostics,
        failure_classification=(
            f"pull_request_comment_status_{status}"
            if status is not None
            else "pull_request_comment_failed"
        ),
        provider_status=status,
    )


def _close_failure(error: BaseException, *, endpoint: str, remedy: str) -> GitHubAdapterError:
    """Wrap a failed close so the record carries the endpoint and status, never the message.

    `_comment_failure`'s shape, for `_comment_failure`'s reason: the HTTP status is the one
    detail that identifies a refusal, it is an integer, and the endpoint is this adapter's
    own choice -- both are safe to record where a provider message is not.
    """
    status = _provider_status(error)
    described_status = str(status) if status is not None else "unreported"
    diagnostics = [
        f"pull-request close refused ({type(error).__name__} status={described_status}) "
        f"on {endpoint}"
    ]
    if status == 401:
        diagnostics.append(
            "likely cause: the credential was rejected outright (expired or revoked token); "
            "not confirmed by this response alone"
        )
    elif status in {403, 404}:
        # GitHub answers 404 rather than 403 where admitting the resource exists would leak
        # its existence, so a missing grant and an absent pull request read the same here.
        diagnostics.append(f"likely cause: {remedy}; not confirmed by this response alone")
    msg = f"pull request could not be closed (status {described_status})"
    return GitHubAdapterError(
        msg,
        diagnostics=diagnostics,
        failure_classification=(
            f"pull_request_close_status_{status}"
            if status is not None
            else "pull_request_close_failed"
        ),
        provider_status=status,
    )


def _create_pygithub_client(token: str) -> Any:
    """Create the PyGithub client lazily so mocks never initialize network-capable code."""
    github_module = importlib.import_module("github")
    authentication = github_module.Auth.Token(token)
    return github_module.Github(auth=authentication)


def _validate_pull_request_input(
    repository: str, title: str, body: str, source_branch: str, target_branch: str
) -> None:
    """Validate pull-request inputs shared by mock and live implementations."""
    if not repository.strip() or not title.strip() or not body.strip():
        msg = "repository, title, and body must not be empty"
        raise GitHubAdapterError(msg)
    if not source_branch.strip() or not target_branch.strip():
        msg = "source_branch and target_branch must not be empty"
        raise GitHubAdapterError(msg)
    if source_branch == target_branch:
        msg = "source_branch and target_branch must differ"
        raise GitHubAdapterError(msg)


def _merge_distinct(current_values: Sequence[str], new_values: Sequence[str]) -> list[str]:
    """Preserve order while appending only non-empty values not already recorded."""
    merged = list(current_values)
    for value in new_values:
        if value.strip() and value not in merged:
            merged.append(value)
    return merged


__all__ = [
    "GitHubAdapterError",
    "GitHubService",
    "MockGitHubService",
    "PullRequestDetails",
    "PyGithubService",
    "select_pull_request",
]
