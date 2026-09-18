"""Git service protocol with safe GitPython and deterministic mock implementations."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from services.process_runner import sanitized_subprocess_environment
from state.failure_diagnosis import FeatureFailureClassification
from tools.file_tools import (
    PathLike,
    carries_key_material,
    is_credential_shaped_path,
    read_key_material_scan_prefix,
)

_GIT_CREDENTIAL_ENVIRONMENT_KEYS = frozenset(
    {"GIT_ASKPASS", "GIT_TERMINAL_PROMPT", "PLATFORM_GIT_TOKEN"}
)


class GitSafetyError(PermissionError):
    """Raised when a requested Git operation violates platform branch safety rules."""


class GitAdapterError(RuntimeError):
    """Raised when a Git operation cannot be completed by the selected adapter.

    Carries ``diagnostics`` only when the raiser built them from platform-owned operation
    names, exit status, and result codes. Repository hook output is never trusted as safe.
    """

    def __init__(self, *args: object, diagnostics: Sequence[str] = ()) -> None:
        """Keep explicitly safe process evidence available to the control plane."""
        self.diagnostics = tuple(diagnostics)
        super().__init__(*args)


class EmptyCommitError(GitAdapterError):
    """Raised when the staged files match the committed revision exactly.

    Distinguished from other commit failures because it is not a fault: an attempt that
    reproduced the existing content simply made no change, which the retry policy already
    treats as a refusable outcome rather than a reason to abandon the workstream.
    """


class GitAuthenticationError(GitAdapterError):
    """The remote refused the credential, rather than failing to answer with it.

    Every other ``GitAdapterError`` is a fault: the remote was unreachable, or answered
    something a second attempt might get past. A refused credential is an *answer*, and it is
    the same answer every time -- so the two call for opposite handling, and until they were
    distinguished the platform gave a dead credential the treatment reserved for weather.

    AB-Feature-190 and -191 are what this class is for. The stored GitHub PAT, created
    2026-08-26, expired at 2026-09-02 00:00 UTC. Clones succeeded at 23:58 and failed at
    00:04; from then on every authenticated clone failed as an anonymous-looking
    ``GitAdapterError``, spent four fault retries per repository, and was reported as
    "The model provider did not answer" -- twice, killing two features before anybody could
    correlate the timestamps by hand and find the actual cause.

    It carries a classification, so the record says the credential is the problem, and the
    classification is deliberately not a retryable one: nothing about waiting fixes a
    credential a person has to replace.
    """

    failure_classification = FeatureFailureClassification.PROVIDER_CREDENTIALS_MISSING


class GitBranchNotFoundError(GitAdapterError):
    """The repository has no branch by the name this feature was configured to build from.

    A sibling of `GitAuthenticationError` and for its reason: every other `GitAdapterError`
    is weather, and this is an *answer*. It is the same answer on every attempt, so spending
    the fault allowance on it converts a one-line configuration mistake into "the Git remote
    did not answer" three backoffs later.

    AB-Feature-221 is what this class is for. Two repositories were configured to build from
    `DAM-EXP-Automation-branch`, which exists on both remotes; `git clone` was invoked without
    `--branch`, so it checked out each remote's HEAD -- `master` -- and created no local ref
    for the configured branch. `git rev-parse DAM-EXP-Automation-branch` then failed, because
    gitrevisions resolves a bare name through `refs/heads` and `refs/remotes/<name>` but never
    through `refs/remotes/origin/<name>`. Both workstreams spent three fault retries and 140
    seconds of backoff on a ref that was never going to appear, and the feature was recorded
    as a platform defect.

    The clone pins the branch now, so this is raised where the branch is actually looked for
    and names the branch that was not found. The classification is a workspace-preparation
    failure rather than a defect or a repository finding: the platform worked, the repository
    is fine, and what is wrong is one field somebody can correct in Settings.
    """

    failure_classification = FeatureFailureClassification.WORKSPACE_MAINTENANCE_FAILURE


@dataclass(frozen=True, slots=True)
class CredentialProvenance:
    """Which stored credential a remote Git operation authenticated with, and when.

    Never the secret, and never anything derived from it. The provider name and the date it
    was stored are the two facts that turn "authentication failed" into an instruction: go
    and look at *this* credential, which is *this* old. That is the whole difference between
    the diagnosis run 190 got and the one it needed.
    """

    provider: str
    stored_at: datetime | None = None

    def described(self) -> str:
        """Name this credential and its age in the one clause a diagnosis can carry."""
        if self.stored_at is None:
            return f"the stored {self.provider} credential"
        return f"the stored {self.provider} credential (stored {self.stored_at.date().isoformat()})"


# Markers in Git's own stderr that mean the remote refused this credential rather than
# failing to answer with it. Recognised, never quoted: the diagnosed-failure contract forbids
# putting checkout or remote text into a durable diagnostic, so a marker selects a sentence
# this platform composed -- the same discipline `_WORKSPACE_COMMAND_CAUSES` follows.
#
# Deliberately conservative, and the asymmetry is the reason. A missed refusal costs what run
# 190 cost: a few wasted fault retries and a misdirected diagnosis. A false positive costs a
# workstream that would have recovered on its own, stopped and handed to a person with a
# sentence about a credential that is fine. So every entry here is a phrase Git or a hosting
# provider emits only for an authentication or authorization decision, and anything else
# keeps today's behaviour.
_AUTHENTICATION_REFUSAL_MARKERS = (
    # Git's own message for an HTTP 401 on a credential-bearing remote.
    "authentication failed",
    "invalid username or password",
    "http basic: access denied",
    # The status line itself, when Git reports the transport's code rather than its meaning.
    "the requested url returned error: 401",
    "the requested url returned error: 403",
    # GitHub's own refusals: a retired authentication method, and a token whose scope does
    # not reach this repository. Both are decisions; neither changes on a retry.
    "support for password authentication was removed",
    "remote: permission to",
    "permission denied (publickey)",
)


def authentication_was_refused(stderr: str) -> bool:
    """Report whether a failing Git command's own output says the credential was refused."""
    lowered = stderr.lower()
    return any(marker in lowered for marker in _AUTHENTICATION_REFUSAL_MARKERS)


# Markers meaning the *ref* this operation was told to use is not there. Recognised and never
# quoted, on the same contract as the refusal markers above: a marker selects a sentence this
# platform composed, and Git's own words -- which name remote URLs -- do not travel.
#
# Every entry is a phrase Git emits only when a named ref could not be resolved, and the
# remedy for all of them is identical: somebody corrects the branch this repository is
# configured to build from. Nothing about waiting changes the answer.
_MISSING_BRANCH_MARKERS = (
    # `git clone --branch <name>` against a remote that has no such branch.
    "not found in upstream",
    "could not find remote branch",
    # `git switch -c <new> <start-point>` where the start point does not resolve. Reachable
    # only if a clone were ever made without pinning the branch; kept so the two paths
    # cannot disagree about what a missing base branch means.
    "invalid reference",
)


def branch_was_not_found(stderr: str) -> bool:
    """Report whether a failing Git command's own output says the named ref is absent."""
    lowered = stderr.lower()
    return any(marker in lowered for marker in _MISSING_BRANCH_MARKERS)


@dataclass(frozen=True, slots=True)
class GitPushResult:
    """The branch and remote outcome recorded after a successful push request."""

    remote: str
    branch: str
    summaries: tuple[str, ...]


class GitService(Protocol):
    """Protocol for repository lifecycle operations used by the GitHub workflow agent."""

    def clone(
        self,
        source_url: str,
        destination: PathLike,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> Path:
        """Clone a repository to the specified local destination."""

    def create_branch(
        self, repository_path: PathLike, branch_name: str, *, base_branch: str
    ) -> str:
        """Create a branch from a named base branch."""

    def commit(
        self,
        repository_path: PathLike,
        message: str,
        *,
        files: Sequence[PathLike] = (),
        expected_content_fingerprint: str | None = None,
    ) -> str:
        """Create a commit containing the exact reviewed content for selected files."""

    def push(
        self,
        repository_path: PathLike,
        branch_name: str,
        *,
        remote_name: str = "origin",
        force: bool = False,
        expected_commit_sha: str | None = None,
    ) -> GitPushResult:
        """Push an exact reviewed commit on a non-default branch without force semantics."""


@dataclass(slots=True)
class _MockRepository:
    """Mutable local state maintained by the in-memory Git implementation."""

    source_url: str
    branches: set[str]
    commits: list[str] = field(default_factory=list)


class MockGitService:
    """In-memory Git implementation for unit tests and deterministic workflow runs."""

    def __init__(self, *, default_branch: str = "main") -> None:
        """Create a mock service with the same branch safety policy as production."""
        self.default_branch = _validate_branch_name(default_branch)
        self.repositories: dict[Path, _MockRepository] = {}
        self.pushes: list[GitPushResult] = []

    def clone(
        self,
        source_url: str,
        destination: PathLike,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> Path:
        """Create an in-memory clone record without contacting a remote service."""
        del environment
        if not source_url.strip():
            msg = "source_url must not be empty"
            raise GitAdapterError(msg)
        destination_path = Path(destination).expanduser().resolve(strict=False)
        if destination_path in self.repositories:
            msg = f"repository already exists at {destination_path}"
            raise GitAdapterError(msg)
        destination_path.mkdir(parents=True, exist_ok=False)
        self.repositories[destination_path] = _MockRepository(
            source_url=source_url,
            branches={self.default_branch},
        )
        return destination_path

    def create_branch(
        self, repository_path: PathLike, branch_name: str, *, base_branch: str
    ) -> str:
        """Create a branch in the mock repository from an existing base branch."""
        repository = self._repository(repository_path)
        branch_name = _validate_branch_name(branch_name)
        if base_branch not in repository.branches:
            msg = f"base branch does not exist: {base_branch}"
            raise GitAdapterError(msg)
        if branch_name in repository.branches:
            msg = f"branch already exists: {branch_name}"
            raise GitAdapterError(msg)
        repository.branches.add(branch_name)
        return branch_name

    def commit(
        self,
        repository_path: PathLike,
        message: str,
        *,
        files: Sequence[PathLike] = (),
        expected_content_fingerprint: str | None = None,
    ) -> str:
        """Record a deterministic mock commit identifier after validating its message."""
        del expected_content_fingerprint
        repository = self._repository(repository_path)
        if not message.strip():
            msg = "commit message must not be empty"
            raise GitAdapterError(msg)
        _validate_commit_files(Path(repository_path), files)
        commit_id = f"mock-{len(repository.commits) + 1:08d}"
        repository.commits.append(commit_id)
        return commit_id

    def push(
        self,
        repository_path: PathLike,
        branch_name: str,
        *,
        remote_name: str = "origin",
        force: bool = False,
        expected_commit_sha: str | None = None,
    ) -> GitPushResult:
        """Record a safe mock push after applying the common branch policy."""
        repository = self._repository(repository_path)
        _validate_safe_push(branch_name, self.default_branch, force)
        if expected_commit_sha is not None and (
            not repository.commits or repository.commits[-1] != expected_commit_sha
        ):
            msg = "refusing to push a branch whose head is not the reviewed commit"
            raise GitSafetyError(msg)
        if branch_name not in repository.branches:
            msg = f"branch does not exist: {branch_name}"
            raise GitAdapterError(msg)
        result = GitPushResult(
            remote=remote_name,
            branch=branch_name,
            summaries=(f"[new branch] {branch_name} -> {branch_name}",),
        )
        self.pushes.append(result)
        return result

    def _repository(self, repository_path: PathLike) -> _MockRepository:
        """Return an existing mock repository by its resolved local path."""
        path = Path(repository_path).expanduser().resolve(strict=False)
        try:
            return self.repositories[path]
        except KeyError as error:
            msg = f"repository is not cloned by this service: {path}"
            raise GitAdapterError(msg) from error


class GitPythonService:
    """Legacy synchronous Git adapter with sanitized subprocess and hook environments."""

    def __init__(
        self,
        *,
        default_branch: str = "main",
        environment: Mapping[str, str] | None = None,
        repository_factory: Callable[[Path], Any] | None = None,
        repository_class: Any | None = None,
    ) -> None:
        """Configure GitPython lazily so standard unit tests can inject local mocks."""
        self.default_branch = _validate_branch_name(default_branch)
        self._environment = _validated_git_credentials(environment)
        self._repository_factory = repository_factory
        self._repository_class = repository_class

    @property
    def environment(self) -> Mapping[str, str]:
        """Return a copy of the ephemeral subprocess environment bound to this adapter."""
        return dict(self._environment)

    def clone(
        self,
        source_url: str,
        destination: PathLike,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> Path:
        """Clone through GitPython after confirming the destination is not already occupied."""
        if not source_url.strip():
            msg = "source_url must not be empty"
            raise GitAdapterError(msg)
        destination_path = Path(destination).expanduser().resolve(strict=False)
        if destination_path.exists():
            msg = f"clone destination already exists: {destination_path}"
            raise GitAdapterError(msg)
        repository_class = self._repository_class or _git_repository_class()
        repository_class.clone_from(
            source_url,
            destination_path,
            env=_gitpython_environment(
                {**self._environment, **_validated_git_credentials(environment)},
                remote=True,
            ),
        )
        return destination_path

    def create_branch(
        self, repository_path: PathLike, branch_name: str, *, base_branch: str
    ) -> str:
        """Create a local branch from the caller-selected existing base branch."""
        repository = self._repository(repository_path)
        branch_name = _validate_branch_name(branch_name)
        with _repository_git_environment(repository, _gitpython_environment()):
            existing_branches = {str(branch.name) for branch in repository.heads}
            if branch_name in existing_branches:
                msg = f"branch already exists: {branch_name}"
                raise GitAdapterError(msg)
            base_commit = repository.commit(base_branch)
            repository.create_head(branch_name, base_commit)
        return branch_name

    def commit(
        self,
        repository_path: PathLike,
        message: str,
        *,
        files: Sequence[PathLike] = (),
        expected_content_fingerprint: str | None = None,
    ) -> str:
        """Stage only validated intended files and create a local Git commit."""
        del expected_content_fingerprint
        if not message.strip():
            msg = "commit message must not be empty"
            raise GitAdapterError(msg)
        repository = self._repository(repository_path)
        repository_root = Path(repository_path).expanduser().resolve(strict=True)
        safe_files = _validate_commit_files(repository_root, files)
        with _repository_git_environment(repository, _gitpython_environment()):
            if repository.is_dirty(index=True, working_tree=False, untracked_files=False):
                msg = (
                    "repository has pre-existing staged changes; refusing to include "
                    "unintended files"
                )
                raise GitSafetyError(msg)
            repository.index.add(safe_files)
            if not repository.is_dirty(index=True, working_tree=False, untracked_files=False):
                msg = "the staged files are identical to the committed revision"
                raise EmptyCommitError(msg)
            commit = repository.index.commit(message)
        return str(commit.hexsha)

    def push(
        self,
        repository_path: PathLike,
        branch_name: str,
        *,
        remote_name: str = "origin",
        force: bool = False,
        expected_commit_sha: str | None = None,
    ) -> GitPushResult:
        """Push a safe feature branch and return all remote status summaries."""
        _validate_safe_push(branch_name, self.default_branch, force)
        repository = self._repository(repository_path)
        if expected_commit_sha is not None and str(repository.head.commit.hexsha) != str(
            expected_commit_sha
        ):
            msg = "refusing to push a branch whose head is not the reviewed commit"
            raise GitSafetyError(msg)
        remote = repository.remote(remote_name)
        remote.push(
            f"{branch_name}:{branch_name}",
            env=_gitpython_environment(self._environment, remote=True),
        )
        return GitPushResult(
            remote=remote_name,
            branch=branch_name,
            summaries=("GIT_PUSH_SUCCEEDED",),
        )

    def _repository(self, repository_path: PathLike) -> Any:
        """Open a local repository through an injected factory or the GitPython implementation."""
        path = Path(repository_path).expanduser().resolve(strict=True)
        if self._repository_factory is not None:
            return self._repository_factory(path)
        return _git_repository_class()(path)


def _validated_git_credentials(environment: Mapping[str, str] | None) -> dict[str, str]:
    """Accept only the request-scoped askpass fields used by live Git authentication."""
    supplied = dict(environment or {})
    unsupported = sorted(set(supplied) - _GIT_CREDENTIAL_ENVIRONMENT_KEYS)
    if unsupported:
        msg = f"Git credential environment contains unsupported keys: {unsupported}"
        raise GitSafetyError(msg)
    return {str(key): str(value) for key, value in supplied.items()}


def _gitpython_environment(
    credentials: Mapping[str, str] | None = None, *, remote: bool = False
) -> dict[str, str]:
    """Mask inherited process values before GitPython merges its inline environment."""
    # GitPython copies ``os.environ`` before applying ``env``. Supplying an empty value for
    # every inherited key is therefore necessary; merely passing the allowlist would leave
    # API keys and database URLs intact in hooks and helpers.
    environment = {key: "" for key in os.environ}
    environment.update(sanitized_subprocess_environment())
    environment.update(
        {
            "GIT_AUTHOR_NAME": "AI Workflow Platform",
            "GIT_AUTHOR_EMAIL": "automation@localhost.invalid",
            "GIT_COMMITTER_NAME": "AI Workflow Platform",
            "GIT_COMMITTER_EMAIL": "automation@localhost.invalid",
        }
    )
    environment.update(_validated_git_credentials(credentials))
    if remote:
        environment.update(
            {
                "GIT_CONFIG_COUNT": "2",
                "GIT_CONFIG_KEY_0": "credential.helper",
                "GIT_CONFIG_VALUE_0": "",
                "GIT_CONFIG_KEY_1": "core.hooksPath",
                "GIT_CONFIG_VALUE_1": os.devnull,
            }
        )
    return environment


def _repository_git_environment(repository: Any, environment: Mapping[str, str]) -> Any:
    """Apply the sanitized map to a real GitPython repository and tolerate test doubles."""
    git = getattr(repository, "git", None)
    custom_environment = getattr(git, "custom_environment", None)
    if callable(custom_environment):
        return custom_environment(**dict(environment))
    return nullcontext()


def _git_repository_class() -> Any:
    """Load GitPython lazily so mock-based tests never execute GitPython setup code."""
    git_module = importlib.import_module("git")
    return git_module.Repo


def _validate_safe_push(branch_name: str, default_branch: str, force: bool) -> None:
    """Enforce the platform's immutable-default-branch and no-force-push policy."""
    branch_name = _validate_branch_name(branch_name)
    if force:
        msg = "force pushes are prohibited"
        raise GitSafetyError(msg)
    if branch_name == default_branch:
        msg = f"pushing the default branch '{default_branch}' is prohibited"
        raise GitSafetyError(msg)


def _validate_branch_name(branch_name: str) -> str:
    """Reject ref names that Git interprets ambiguously or as an option/path escape."""
    if not branch_name.strip():
        msg = "branch name must not be empty"
        raise GitAdapterError(msg)
    prohibited_fragments = ("..", "@{", "\\", "?", "*", "[", "^", "~", ":")
    if (
        branch_name.startswith(("-", "/", "."))
        or branch_name.endswith(("/", ".", ".lock"))
        or "//" in branch_name
        or any(fragment in branch_name for fragment in prohibited_fragments)
        or any(
            not component or component.startswith(".") or component.endswith(".lock")
            for component in branch_name.split("/")
        )
    ):
        msg = "branch name is not a safe Git ref"
        raise GitSafetyError(msg)
    return branch_name


def _validate_commit_files(repository_root: Path, files: Sequence[PathLike]) -> list[str]:
    """Return safe repository-relative files and never permit broad or secret staging.

    This reads the staged files, which is a real change in character from the pure path
    validation it used to be, and it belongs here rather than at the callers. Three of them
    reach this function -- `GitPythonService.commit`, `MockGitService.commit` and
    `InterruptibleGitService.commit`, which is the one the live executor uses -- and
    "an autonomous commit never contains key material" has to hold for every one of them. A
    check placed at the callers is a guard on some paths, which reads as coverage without
    being it. This function already resolves each file against `repository_root`, so it is
    the one place that both shares the rule and has the file.
    """
    if not files:
        msg = "commit requires explicit intended file paths"
        raise GitSafetyError(msg)
    root = repository_root.expanduser().resolve(strict=False)
    safe_files: list[str] = []
    seen: set[str] = set()
    for file in files:
        raw_path = Path(file)
        if raw_path.is_absolute() or ".." in raw_path.parts:
            msg = "commit file paths must be repository-relative"
            raise GitSafetyError(msg)
        relative_path = raw_path.as_posix()
        if not relative_path or relative_path == "." or _is_sensitive_path(raw_path):
            msg = "committing Git metadata or secret-like files is prohibited"
            raise GitSafetyError(msg)
        resolved_path = (root / raw_path).resolve(strict=False)
        try:
            resolved_path.relative_to(root)
        except ValueError as error:
            msg = "commit file path escapes repository root"
            raise GitSafetyError(msg) from error
        _refuse_staged_key_material(relative_path, resolved_path)
        if relative_path not in seen:
            safe_files.append(relative_path)
            seen.add(relative_path)
    return safe_files


def _is_sensitive_path(path: Path) -> bool:
    """Identify paths that must never be added by autonomous repository commits.

    Two rules, deliberately not merged. `.git` metadata is not key material and has nothing
    to do with credentials: it is refused because an autonomous commit rewriting a
    repository's own Git state is wrong on its face, and that rule belongs to this path
    alone. The credential half defers to `is_credential_shaped_path` so that the commit gate
    and the context rule cannot answer differently about the same file.

    This used to carry its own copy, and it drifted in exactly the two ways the copy in
    engineer context did: it matched `.env` as a prefix, so `sendgrid.env` passed, and it
    judged by filename only, so a service-account key called
    `imls-1566290608647-c1739c763650.json` passed. `_refuse_staged_key_material` answers
    that second half from the bytes.
    """
    return ".git" in path.parts or is_credential_shaped_path(path)


def _refuse_staged_key_material(relative_path: str, staged_path: Path) -> None:
    """Refuse a staged file whose own bytes are key material, whatever it is called.

    A filename is half the question. `is_credential_shaped_path` cannot distinguish a live
    service-account key from `package.json`, and the cost of being wrong here is worse than
    it is in context selection: a credential sent to a provider goes to one party under a
    contract, while a credential committed to a branch lands in a pull request, in the
    repository's history, and in every clone of it.

    Reading is affordable because the commit stages exactly the files the Engineer reported
    writing -- a handful -- and each is read only to `read_key_material_scan_prefix`'s bound.

    Fail closed on a file that exists and cannot be read. At commit time that is a stranger
    event than in context selection, since the attempt has just claimed to have written the
    file, and the safe answer is to stop rather than to stage bytes nothing has inspected.

    Not being *text* is a different thing and is not a refusal. This gate briefly refused
    every binary a repository can contain -- an image, a font, an icon, a compiled asset --
    because the prefix read let `UnicodeDecodeError` escape and it was caught here. That
    blocked any feature adding an asset, and it said the file could not be read, which was
    untrue. The markers are ASCII, so a file that is not text simply carries none; the read
    is decode-tolerant now and only a genuine read failure arrives here.

    A path that does not exist is a staged deletion and has no bytes to leak. A symlink
    commits its target's *path*, not that target's contents, so reading through it would
    refuse a commit over bytes that are not being committed. Neither is read.
    """
    if staged_path.is_symlink() or not staged_path.exists():
        return
    try:
        head = read_key_material_scan_prefix(staged_path)
    except (OSError, ValueError) as error:
        msg = f"refusing to commit a staged file that cannot be read: {relative_path}"
        raise GitSafetyError(msg) from error
    if carries_key_material(head):
        msg = f"committing a file whose contents are key material is prohibited: {relative_path}"
        raise GitSafetyError(msg)
