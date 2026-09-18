"""Cancellation-aware Git CLI adapter with durable operation-journal integration."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from adapters.git_adapter import (
    CredentialProvenance,
    EmptyCommitError,
    GitAdapterError,
    GitAuthenticationError,
    GitBranchNotFoundError,
    GitPushResult,
    GitSafetyError,
    _validate_commit_files,
    authentication_was_refused,
    branch_was_not_found,
)
from services.cancellation import CancellationRequested, CancellationToken
from services.external_operations import (
    EffectAbsent,
    EffectUnproven,
    ExternalOperationExecutor,
    ReconciliationOutcome,
    UnknownExternalOperation,
)
from services.process_runner import (
    AsyncioProcessRunner,
    ProcessRunner,
    sanitized_subprocess_environment,
    subprocess_result_code,
)
from state.external_operations import ExternalOperationType
from storage.external_operation_store import OperationResult
from tools.file_tools import PathLike, resolve_workspace_root

# A commit is bounded by whatever its pre-commit hook does, not by Git itself.
_DEFAULT_COMMIT_TIMEOUT_SECONDS = 900.0
_REMOTE_GIT_ENVIRONMENT_KEYS = frozenset(
    {"GIT_ASKPASS", "GIT_TERMINAL_PROMPT", "PLATFORM_GIT_TOKEN"}
)


HASH_CHUNK_BYTES = 1024 * 1024


def streamed_file_digest(path: Path) -> bytes:
    """Hash one file's exact bytes without ever holding the file in memory.

    Named and exported because two callers need the same guarantee for the same reason: a
    file's identity must be computable regardless of its size. The publication binding below
    hashes every reviewed file this way, and the reviewer's evidence hashes a generated
    lockfile this way when the file is too large to be quoted -- a fact about the bytes is
    exactly what remains available when the bytes themselves are not.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.digest()


def reviewed_content_fingerprint(repository_path: PathLike, files: Sequence[PathLike]) -> str:
    """Hash exact reviewed workspace bytes, paths, and executable modes.

    The value deliberately survives a successful commit: unlike ``git diff``, it is the
    same before and after publication.  That makes it both an integrity assertion and a
    stable idempotency input when a worker dies after Git changed ``HEAD`` but before the
    workflow checkpoint was persisted.
    """
    workspace = resolve_workspace_root(repository_path)
    safe_files = sorted(_validate_commit_files(workspace, files))
    digest = hashlib.sha256()
    for relative_path in safe_files:
        path = workspace / relative_path
        _update_digest_field(digest, relative_path.encode("utf-8"))
        if path.is_symlink():
            _update_digest_field(digest, b"symlink")
            _update_digest_field(digest, os.readlink(path).encode("utf-8"))
            continue
        if not path.exists():
            _update_digest_field(digest, b"deleted")
            continue
        if not path.is_file():
            msg = f"reviewed commit path is not a regular file: {relative_path}"
            raise GitSafetyError(msg)
        mode = b"executable" if path.stat().st_mode & stat.S_IXUSR else b"regular"
        _update_digest_field(digest, mode)
        _update_digest_field(digest, streamed_file_digest(path))
    return digest.hexdigest()


class InterruptibleGitService:
    """Use `git` subprocesses so clone, commit, and push obey active cancellation signals."""

    def __init__(
        self,
        *,
        default_branch: str = "main",
        base_branch: str | None = None,
        environment: Mapping[str, str] | None = None,
        cancellation_token: CancellationToken,
        operation_executor: ExternalOperationExecutor | None = None,
        process_runner: ProcessRunner | None = None,
        workspace_root: Path | None = None,
        commit_timeout_seconds: float = _DEFAULT_COMMIT_TIMEOUT_SECONDS,
        credential: CredentialProvenance | None = None,
    ) -> None:
        self.default_branch = default_branch
        # What the clone checks out, when it is not the default: a feature revision builds
        # on the superseded run's branch. Deliberately a second field rather than an
        # overload of `default_branch`, which also drives the push guard -- collapsing the
        # two would permit pushing the repository's true default branch.
        self._base_branch = base_branch or default_branch
        self._environment = _credential_environment(environment)
        self._cancellation_token = cancellation_token
        self._operation_executor = operation_executor
        self._process_runner = process_runner or AsyncioProcessRunner()
        self._workspace_root = workspace_root.resolve(strict=False) if workspace_root else None
        self._commit_timeout = commit_timeout_seconds
        # Which stored credential `environment` above carries, as a name and a date and
        # nothing else. Optional because a deployment that authenticates some other way, and
        # every test that supplies no credential at all, still needs a working service; where
        # it is absent, a refused remote operation reports the generic failure it always did.
        self._credential = credential

    @property
    def environment(self) -> Mapping[str, str]:
        """Expose only the current request-scoped Git environment to adapter composition."""
        return dict(self._environment)

    async def clone(
        self,
        source_url: str,
        destination: PathLike,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> Path:
        """Clone through a process group and clean only an incomplete owned workspace on cancel."""
        if not source_url.strip():
            msg = "source_url must not be empty"
            raise GitAdapterError(msg)
        target = Path(destination).expanduser().resolve(strict=False)
        merged_environment = self._remote_environment(environment)

        # The branch this repository is configured to build from, pinned into the clone.
        #
        # `git clone` without it checks out whatever the *remote's* HEAD is and creates a
        # local ref for that branch alone; every other branch arrives only as
        # `refs/remotes/origin/<name>`. A bare name does not resolve through that --
        # gitrevisions tries `refs/heads/<name>` and `refs/remotes/<name>`, never
        # `refs/remotes/origin/<name>` -- so `create_branch`'s `rev-parse <base_branch>`
        # failed for every repository whose configured branch was not its default. That is
        # AB-Feature-221: two repositories configured to build from an integration branch,
        # three fault retries and 140 seconds of backoff each, recorded as a platform defect.
        #
        # Pinning it here rather than resolving `origin/<base>` at `create_branch` fixes the
        # branch this checkout *is*, not just the ref one command can find. Everything between
        # the clone and the first commit -- the preflight install, the baseline validation that
        # proves the platform can run the repository's own commands -- then runs against the
        # branch the change will actually be based on, instead of against a default branch
        # nobody asked about.
        base_branch = self._base_branch.strip()
        if not base_branch:
            msg = "a clone requires the branch this repository is configured to build from"
            raise GitAdapterError(msg)

        async def action() -> tuple[Path, OperationResult]:
            # Check only after the journal has had a chance to return a previously
            # confirmed clone.  A crash after clone but before the graph checkpoint
            # must reuse that clone rather than fail because its directory exists.
            if target.exists() and not await self._discard_spent_workspace(target):
                msg = f"clone destination already exists: {target}"
                raise GitAdapterError(msg)
            result = await self._process_runner.run(
                ("git", "clone", "--branch", base_branch, source_url, str(target)),
                target.parent,
                timeout_seconds=600,
                cancellation_token=self._cancellation_token,
                environment=merged_environment,
            )
            if result.cancelled:
                await self._cleanup_incomplete_clone(target)
                raise CancellationRequested("git clone was cancelled")
            _require_success(
                result.stderr,
                result.return_code,
                "git clone",
                credential=self._credential,
                branch=base_branch,
            )
            return target, OperationResult(
                external_reference=str(target),
                payload={
                    "workspace_path": str(target),
                    "source_url": source_url,
                    "base_branch": base_branch,
                },
            )

        return await self._run_or_execute(
            operation_type=ExternalOperationType.CLONE_REPOSITORY,
            logical_step="clone_repository",
            # The branch is part of what makes this clone the operation it is. Without it a
            # journalled clone taken on a different branch would be reused for a feature
            # configured to build from another one, and the reuse would look like success.
            safe_input={
                "workspace_path": str(target),
                "source_url": source_url,
                "base_branch": base_branch,
            },
            action=action,
            reused=lambda payload: _reused_clone(target, payload),
            # A clone is a read-only fetch into a local workspace.  A bounded
            # retry is safe because a completed clone is journal-reused and an
            # incomplete destination is rejected before another clone begins.
            max_attempts=3,
        )

    async def create_branch(
        self, repository_path: PathLike, branch_name: str, *, base_branch: str
    ) -> str:
        """Create or safely reuse an expected feature branch from the declared base branch."""
        workspace = resolve_workspace_root(repository_path)

        async def action() -> tuple[str, OperationResult]:
            existing = await self._git_output(workspace, "rev-parse", "--verify", branch_name)
            base_sha = await self._git_output(workspace, "rev-parse", base_branch)
            if existing is not None:
                ancestry = await self._process_runner.run(
                    ("git", "merge-base", "--is-ancestor", base_branch, branch_name),
                    workspace,
                    20,
                    self._cancellation_token,
                    self._local_environment(),
                )
                if not ancestry.succeeded:
                    msg = "existing branch does not contain the expected base branch"
                    raise GitSafetyError(msg)
                switch = await self._process_runner.run(
                    ("git", "switch", branch_name),
                    workspace,
                    30,
                    self._cancellation_token,
                    self._local_environment(),
                )
                _require_success(switch.stderr, switch.return_code, "git switch")
                return branch_name, OperationResult(
                    external_reference=existing,
                    payload={
                        "branch_name": branch_name,
                        "base_sha": base_sha,
                        "reused_branch": True,
                    },
                )
            await self._cancellation_token.raise_if_cancelled()
            result = await self._process_runner.run(
                ("git", "switch", "-c", branch_name, base_branch),
                workspace,
                30,
                self._cancellation_token,
                self._local_environment(),
            )
            if result.cancelled:
                raise CancellationRequested("git branch creation was cancelled")
            _require_success(result.stderr, result.return_code, "git branch creation")
            return branch_name, OperationResult(
                external_reference=await self._git_output_required(workspace, "rev-parse", "HEAD"),
                payload={"branch_name": branch_name, "base_sha": base_sha},
            )

        async def reconcile(_operation: object) -> OperationResult | None:
            existing = await self._git_output(workspace, "rev-parse", "--verify", branch_name)
            if existing is None:
                return None
            base_sha = await self._git_output(workspace, "rev-parse", base_branch)
            ancestry = await self._process_runner.run(
                ("git", "merge-base", "--is-ancestor", base_branch, branch_name),
                workspace,
                20,
                self._cancellation_token,
                self._local_environment(),
            )
            if base_sha is None or not ancestry.succeeded:
                return None
            return OperationResult(
                external_reference=existing,
                payload={"branch_name": branch_name, "base_sha": base_sha, "recovered": True},
            )

        created_branch = await self._run_or_execute(
            operation_type=ExternalOperationType.CREATE_BRANCH,
            logical_step="create_branch",
            safe_input={
                "workspace_path": str(workspace),
                "branch_name": branch_name,
                "base_branch": base_branch,
            },
            action=action,
            reused=lambda payload: str(payload.get("branch_name", branch_name)),
            reconcile=reconcile,
            # Everything a repository publishes hangs off its branch existing, and without a
            # replay budget one failed creation is journaled as terminal -- so a lock file
            # held for a moment ended the repository for the whole feature, resumes included.
            # The action verifies the branch first and reuses it, so replaying cannot create
            # a second one or move an existing one.
            max_attempts=3,
        )
        # A prior successful operation may be replayed after the process switched
        # away from the feature branch.  Re-establish that local, non-remote state
        # before the coding step runs.
        current_branch = await self._git_output(workspace, "branch", "--show-current")
        if current_branch != created_branch:
            switch = await self._process_runner.run(
                ("git", "switch", created_branch),
                workspace,
                30,
                self._cancellation_token,
                self._local_environment(),
            )
            if switch.cancelled:
                raise CancellationRequested("git branch checkout was cancelled")
            _require_success(switch.stderr, switch.return_code, "git switch")
        return created_branch

    async def commit(
        self,
        repository_path: PathLike,
        message: str,
        *,
        files: Sequence[PathLike] = (),
        expected_content_fingerprint: str | None = None,
        publication_receipt: Mapping[str, object] | None = None,
    ) -> str:
        """Stage explicit files and commit only while cancellation remains clear."""
        if not message.strip():
            msg = "commit message must not be empty"
            raise GitAdapterError(msg)
        workspace = resolve_workspace_root(repository_path)
        safe_files = sorted(_validate_commit_files(workspace, files))
        current_content_fingerprint = reviewed_content_fingerprint(workspace, safe_files)
        reviewed_fingerprint = expected_content_fingerprint or current_content_fingerprint
        durable_publication_receipt = dict(publication_receipt or {})
        if current_content_fingerprint != reviewed_fingerprint:
            msg = "reviewed file content changed after approval and before commit"
            raise GitSafetyError(msg)
        parent_sha = await self._git_output_required(workspace, "rev-parse", "HEAD")
        working_tree_fingerprint = await self._working_tree_fingerprint(workspace, safe_files)
        expected_tree_fingerprint = await self._working_commit_tree_fingerprint(
            workspace, safe_files
        )

        async def action() -> tuple[str, OperationResult]:
            await self._cancellation_token.raise_if_cancelled()
            staged = await self._process_runner.run(
                ("git", "diff", "--cached", "--quiet"),
                workspace,
                20,
                self._cancellation_token,
                self._local_environment(),
            )
            if staged.return_code not in {0, None}:
                msg = (
                    "repository has pre-existing staged changes; "
                    "refusing unintended commit contents"
                )
                raise GitSafetyError(msg)
            add = await self._process_runner.run(
                ("git", "add", "--", *safe_files),
                workspace,
                30,
                self._cancellation_token,
                self._local_environment(),
            )
            if add.cancelled:
                raise CancellationRequested("git staging was cancelled")
            _require_success(add.stderr, add.return_code, "git add")
            await self._cancellation_token.raise_if_cancelled()
            # An attempt that reproduces the committed content exactly has nothing to record.
            # Git exits non-zero for it, which read as a hard adapter fault and killed the
            # workstream; it is really the ordinary "this attempt changed nothing" outcome the
            # retry policy already knows how to refuse.
            unchanged = await self._process_runner.run(
                ("git", "diff", "--cached", "--quiet"),
                workspace,
                20,
                self._cancellation_token,
                self._local_environment(),
            )
            if unchanged.return_code == 0:
                msg = "the staged files are identical to the committed revision"
                raise EmptyCommitError(msg)
            staged_tree_fingerprint = await self._index_tree_fingerprint(workspace, safe_files)
            if staged_tree_fingerprint != expected_tree_fingerprint:
                msg = "staged Git content does not match the reviewed workspace content"
                raise GitSafetyError(msg)
            commit = await self._process_runner.run(
                ("git", "commit", "-m", message),
                workspace,
                # A commit runs the repository's pre-commit hook, which lints the whole
                # project. A minute was never enough for that: the hook was killed part-way
                # and the commit reported whatever it had printed so far, which was a
                # warning rather than the reason it failed.
                self._commit_timeout,
                self._cancellation_token,
                self._local_environment(),
            )
            if commit.cancelled:
                head = await self._git_output(workspace, "rev-parse", "HEAD")
                if head is not None and head != parent_sha:
                    raise UnknownExternalOperation(
                        "git commit may have completed after cancellation", external_reference=head
                    )
                raise CancellationRequested("git commit was cancelled")
            _require_success(commit.stderr, commit.return_code, "git commit")
            commit_sha = await self._git_output_required(workspace, "rev-parse", "HEAD")
            if not await self._commit_matches_review(
                workspace,
                commit_sha=commit_sha,
                parent_sha=parent_sha,
                expected_files=safe_files,
                expected_content_fingerprint=reviewed_fingerprint,
                expected_tree_fingerprint=expected_tree_fingerprint,
            ):
                raise UnknownExternalOperation(
                    "git hook or post-review process changed the approved commit contents",
                    external_reference=commit_sha,
                )
            return commit_sha, OperationResult(
                external_reference=commit_sha,
                payload={
                    "commit_sha": commit_sha,
                    "parent_head_sha": parent_sha,
                    "working_tree_fingerprint": working_tree_fingerprint,
                    "expected_staged_files": safe_files,
                    "commit_message": message,
                    "reviewed_content_fingerprint": reviewed_fingerprint,
                    "expected_tree_fingerprint": expected_tree_fingerprint,
                },
            )

        async def reconcile(operation: object) -> OperationResult | None:
            metadata = getattr(operation, "safe_metadata", {})
            expected_parent = metadata.get("parent_head_sha")
            expected_files = metadata.get("expected_staged_files")
            expected_reviewed = metadata.get("reviewed_content_fingerprint")
            expected_tree = metadata.get("expected_tree_fingerprint")
            if (
                not isinstance(expected_parent, str)
                or not isinstance(expected_files, list)
                or not all(isinstance(path, str) for path in expected_files)
                or not isinstance(expected_reviewed, str)
                or not isinstance(expected_tree, str)
            ):
                return None
            candidate = await self._git_output(workspace, "rev-parse", "HEAD")
            if candidate is None or not await self._commit_matches_review(
                workspace,
                commit_sha=candidate,
                parent_sha=expected_parent,
                expected_files=cast(list[str], expected_files),
                expected_content_fingerprint=expected_reviewed,
                expected_tree_fingerprint=expected_tree,
            ):
                return None
            return OperationResult(
                external_reference=candidate,
                payload={
                    "commit_sha": candidate,
                    "parent_head_sha": expected_parent,
                    "reviewed_content_fingerprint": expected_reviewed,
                    "expected_tree_fingerprint": expected_tree,
                    "recovered": True,
                },
            )

        commit_sha = await self._run_or_execute(
            operation_type=ExternalOperationType.CREATE_COMMIT,
            logical_step="create_commit",
            safe_input={
                "workspace_path": str(workspace),
                "parent_head_sha": parent_sha,
                "working_tree_fingerprint": working_tree_fingerprint,
                "expected_staged_files": safe_files,
                "commit_message": message,
                "reviewed_content_fingerprint": reviewed_fingerprint,
                "expected_tree_fingerprint": expected_tree_fingerprint,
                "publication_receipt": durable_publication_receipt,
            },
            idempotency_input={
                "workspace_path": str(workspace),
                "expected_staged_files": safe_files,
                "commit_message": message,
                "reviewed_content_fingerprint": reviewed_fingerprint,
            },
            action=action,
            reused=lambda payload: str(payload.get("commit_sha", "")),
            reconcile=reconcile,
        )
        verified_parent = await self._commit_parent(workspace, commit_sha) if commit_sha else ""
        if (
            not commit_sha
            or not verified_parent
            or not await self._commit_matches_review(
                workspace,
                commit_sha=commit_sha,
                parent_sha=verified_parent,
                expected_files=safe_files,
                expected_content_fingerprint=reviewed_fingerprint,
                expected_tree_fingerprint=expected_tree_fingerprint,
            )
        ):
            msg = "recovered commit does not match the exact reviewed workspace content"
            raise GitSafetyError(msg)
        return commit_sha

    async def push(
        self,
        repository_path: PathLike,
        branch_name: str,
        *,
        remote_name: str = "origin",
        force: bool = False,
        expected_commit_sha: str | None = None,
    ) -> GitPushResult:
        """Push once and mark ambiguous cancellation outcomes for reconciliation."""
        if force:
            msg = "force pushes are prohibited"
            raise GitSafetyError(msg)
        if branch_name == self.default_branch:
            msg = f"pushing the default branch '{self.default_branch}' is prohibited"
            raise GitSafetyError(msg)
        workspace = resolve_workspace_root(repository_path)
        local_sha = await self._git_output_required(workspace, "rev-parse", "HEAD")
        if expected_commit_sha is not None and local_sha != expected_commit_sha:
            msg = "refusing to push a branch whose head is not the reviewed commit"
            raise GitSafetyError(msg)
        remote_before = await self._remote_sha(workspace, remote_name, branch_name)

        async def action() -> tuple[GitPushResult, OperationResult]:
            await self._cancellation_token.raise_if_cancelled()
            result = await self._process_runner.run(
                (
                    "git",
                    "-c",
                    f"core.hooksPath={os.devnull}",
                    "-c",
                    "credential.helper=",
                    "push",
                    remote_name,
                    f"{branch_name}:{branch_name}",
                ),
                workspace,
                180,
                self._cancellation_token,
                self._remote_environment(),
            )
            remote_after = await self._remote_sha(workspace, remote_name, branch_name)
            if result.cancelled:
                if remote_after == local_sha:
                    return (
                        GitPushResult(
                            remote_name, branch_name, ("GIT_PUSH_COMPLETED_AFTER_CANCELLATION",)
                        ),
                        OperationResult(
                            external_reference=remote_after,
                            payload={
                                "remote_sha": remote_after,
                                "completed_after_cancellation": True,
                            },
                        ),
                    )
                raise UnknownExternalOperation(
                    "git push outcome is ambiguous after cancellation",
                    external_reference=remote_after,
                )
            # Judge the push by where the remote branch points, not by what the command's
            # exit code said. A push that placed the reviewed commit on the remote and then
            # reported a failure -- a flaky post-receive hook, a dropped connection while
            # writing the tracking ref -- has already done its whole job, and failing it here
            # cost the repository its pull request for a commit that was on the remote the
            # entire time. A push that did not place the commit still fails.
            if remote_after != local_sha:
                _require_success(
                    result.stderr, result.return_code, "git push", credential=self._credential
                )
                msg = (
                    "git push returned success but the remote branch does not point to "
                    "the approved local commit"
                )
                raise GitAdapterError(msg)
            return (
                GitPushResult(
                    remote_name,
                    branch_name,
                    (
                        subprocess_result_code(
                            return_code=result.return_code,
                            timed_out=result.timed_out,
                            cancelled=result.cancelled,
                            prefix="git_push",
                        ),
                    ),
                ),
                OperationResult(
                    external_reference=remote_after,
                    payload={"remote_sha": remote_after, "local_sha": local_sha},
                ),
            )

        async def reconcile(operation: object) -> ReconciliationOutcome:
            metadata = getattr(operation, "safe_metadata", {})
            expected_sha = metadata.get("expected_local_sha")
            if not isinstance(expected_sha, str):
                return EffectUnproven(
                    method="remote_branch_sha_match",
                    detail="the operation does not record which commit the push was to place",
                )
            reachable, remote_sha = await self._remote_branch_state(
                workspace, remote_name, branch_name
            )
            if not reachable:
                return EffectUnproven(
                    method="remote_branch_sha_match",
                    detail="the remote could not be asked where the branch points",
                )
            if remote_sha == expected_sha:
                return OperationResult(
                    external_reference=remote_sha,
                    payload={
                        "remote_sha": remote_sha,
                        "local_sha": expected_sha,
                        "recovered": True,
                        "recovery_method": "remote_branch_sha_match",
                    },
                )
            if remote_sha is None:
                return EffectAbsent(
                    method="remote_branch_sha_match",
                    detail="the remote has no such branch, so the push did not reach it",
                    observations={"expected_sha": expected_sha, "branch": branch_name},
                )
            # The branch is there and points somewhere else. The reviewed commit is not on
            # the remote, and a replay is still a plain push -- force is never used, so a
            # genuinely divergent remote refuses the replay instead of being overwritten.
            return EffectAbsent(
                method="remote_branch_diverged",
                detail=(
                    "the remote branch points at a different commit, so the reviewed commit "
                    "was not placed on it"
                ),
                observations={
                    "expected_sha": expected_sha,
                    "observed_sha": remote_sha,
                    "branch": branch_name,
                },
            )

        return await self._run_or_execute(
            operation_type=ExternalOperationType.PUSH_BRANCH,
            logical_step="push_branch",
            safe_input={
                "workspace_path": str(workspace),
                "remote": remote_name,
                "branch": branch_name,
                "expected_local_sha": local_sha,
                "expected_remote_before_sha": remote_before,
            },
            idempotency_input={
                "workspace_path": str(workspace),
                "remote": remote_name,
                "branch": branch_name,
                "expected_local_sha": local_sha,
            },
            action=action,
            reused=lambda _payload: GitPushResult(remote_name, branch_name, ("reused",)),
            reconcile=reconcile,
            # Without a replay budget a single failed push is journaled as terminal, and a
            # terminal operation refuses to run again -- so one dropped connection denied a
            # reviewed repository its branch on the remote, and therefore its pull request,
            # for the life of the feature. Resuming into it is safe: a completed push is
            # reused from the journal, the replay pushes the same reviewed commit to the
            # same branch, the remote SHA is checked against it, and force is prohibited.
            max_attempts=3,
        )

    async def _run_or_execute[T](
        self,
        *,
        operation_type: ExternalOperationType,
        logical_step: str,
        safe_input: dict[str, object],
        action: Callable[[], Awaitable[tuple[T, OperationResult]]],
        reused: Callable[[dict[str, Any]], T],
        idempotency_input: dict[str, object] | None = None,
        reconcile: Callable[[object], Awaitable[ReconciliationOutcome]] | None = None,
        max_attempts: int = 1,
    ) -> T:
        """Use journaling when configured while keeping the adapter useful in isolated tests."""
        if self._operation_executor is None:
            value, _ = await action()
            return value
        journaled = await self._operation_executor.run(
            operation_type=operation_type,
            logical_step=logical_step,
            safe_input=safe_input,
            action=action,
            idempotency_input=idempotency_input,
            reconcile=reconcile,
            max_attempts=max_attempts,
        )
        if journaled.reused:
            return reused(dict(journaled.operation.result_payload or {}))
        return cast(T, journaled.value)

    async def _git_output(
        self, workspace: Path, *args: str, credentialed: bool = False
    ) -> str | None:
        """Run a bounded read-only Git command and return stripped output only on success."""
        result = await self._process_runner.run(
            ("git", *args),
            workspace,
            20,
            self._cancellation_token,
            self._remote_environment() if credentialed else self._local_environment(),
        )
        return result.stdout.strip() if result.succeeded else None

    async def _git_output_required(self, workspace: Path, *args: str) -> str:
        """Return required Git metadata or fail before attempting a side effect."""
        output = await self._git_output(workspace, *args)
        if not output:
            msg = f"git {' '.join(args)} failed"
            raise GitAdapterError(msg)
        return output

    async def _remote_sha(self, workspace: Path, remote: str, branch: str) -> str | None:
        """Read remote branch state for durable push reconciliation without force semantics."""
        _reachable, sha = await self._remote_branch_state(workspace, remote, branch)
        return sha

    async def _remote_branch_state(
        self, workspace: Path, remote: str, branch: str
    ) -> tuple[bool, str | None]:
        """Separate "the remote says the branch is not there" from "the remote did not answer".

        Both look like an absent SHA, and treating them the same is how a network fault gets
        recorded as proof that a push never happened.
        """
        result = await self._process_runner.run(
            ("git", "ls-remote", remote, f"refs/heads/{branch}"),
            workspace,
            20,
            self._cancellation_token,
            self._remote_environment(),
        )
        if not result.succeeded:
            return False, None
        output = result.stdout.strip()
        return True, output.split()[0] if output else None

    async def _working_commit_tree_fingerprint(self, workspace: Path, files: Sequence[str]) -> str:
        """Fingerprint the blobs Git would stage without changing the index."""
        entries: list[tuple[str, str, str]] = []
        for relative_path in sorted(files):
            path = workspace / relative_path
            if not path.exists() and not path.is_symlink():
                entries.append((relative_path, "000000", "deleted"))
                continue
            blob = await self._git_output_required(
                workspace, "hash-object", f"--path={relative_path}", "--", relative_path
            )
            entries.append((relative_path, _git_file_mode(path), blob))
        return _tree_entries_fingerprint(entries)

    async def _index_tree_fingerprint(self, workspace: Path, files: Sequence[str]) -> str:
        """Fingerprint the exact selected index entries immediately before hooks run."""
        entries: list[tuple[str, str, str]] = []
        for relative_path in sorted(files):
            output = await self._git_output(workspace, "ls-files", "--stage", "--", relative_path)
            if not output:
                entries.append((relative_path, "000000", "deleted"))
                continue
            fields = output.splitlines()[0].split(maxsplit=3)
            if len(fields) < 3 or fields[2] != "0":
                msg = f"Git index is conflicted for reviewed path: {relative_path}"
                raise GitSafetyError(msg)
            entries.append((relative_path, fields[0], fields[1]))
        return _tree_entries_fingerprint(entries)

    async def _revision_tree_fingerprint(
        self, workspace: Path, revision: str, files: Sequence[str]
    ) -> str | None:
        """Fingerprint selected entries as stored in one immutable commit tree."""
        entries: list[tuple[str, str, str]] = []
        for relative_path in sorted(files):
            output = await self._git_output(workspace, "ls-tree", revision, "--", relative_path)
            if not output:
                entries.append((relative_path, "000000", "deleted"))
                continue
            metadata, _, recorded_path = output.partition("\t")
            fields = metadata.split()
            if len(fields) != 3 or recorded_path != relative_path:
                return None
            entries.append((relative_path, fields[0], fields[2]))
        return _tree_entries_fingerprint(entries)

    async def _commit_parent(self, workspace: Path, commit_sha: str) -> str:
        """Return the sole parent required for autonomous non-merge commits."""
        lineage = await self._git_output(workspace, "rev-list", "--parents", "-n", "1", commit_sha)
        fields = lineage.split() if lineage else []
        return fields[1] if len(fields) == 2 else ""

    async def _commit_matches_review(
        self,
        workspace: Path,
        *,
        commit_sha: str,
        parent_sha: str,
        expected_files: Sequence[str],
        expected_content_fingerprint: str,
        expected_tree_fingerprint: str,
    ) -> bool:
        """Prove HEAD contains only the reviewed paths and exact reviewed blobs."""
        head = await self._git_output(workspace, "rev-parse", "HEAD")
        if head != commit_sha or await self._commit_parent(workspace, commit_sha) != parent_sha:
            return False
        changed = await self._process_runner.run(
            (
                "git",
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                parent_sha,
                commit_sha,
                "--",
            ),
            workspace,
            20,
            self._cancellation_token,
            self._local_environment(),
        )
        changed_paths = set(changed.stdout.splitlines())
        if (
            not changed.succeeded
            or not changed_paths
            or not changed_paths.issubset(set(expected_files))
        ):
            return False
        if reviewed_content_fingerprint(workspace, expected_files) != expected_content_fingerprint:
            return False
        tree_fingerprint = await self._revision_tree_fingerprint(
            workspace, commit_sha, expected_files
        )
        return tree_fingerprint == expected_tree_fingerprint

    async def _working_tree_fingerprint(self, workspace: Path, files: Sequence[str]) -> str:
        """Hash intended local changes before staging for safe commit retry identification."""
        result = await self._process_runner.run(
            ("git", "diff", "--", *files),
            workspace,
            20,
            self._cancellation_token,
            self._local_environment(),
        )
        _require_success(result.stderr, result.return_code, "git diff")
        return hashlib.sha256(result.stdout.encode("utf-8")).hexdigest()

    def _local_environment(self) -> Mapping[str, str]:
        """Return a credential-free environment for local Git and repository hooks."""
        environment = sanitized_subprocess_environment()
        environment.setdefault("GIT_AUTHOR_NAME", "AI Workflow Platform")
        environment.setdefault("GIT_AUTHOR_EMAIL", "automation@localhost.invalid")
        environment.setdefault("GIT_COMMITTER_NAME", environment["GIT_AUTHOR_NAME"])
        environment.setdefault("GIT_COMMITTER_EMAIL", environment["GIT_AUTHOR_EMAIL"])
        return environment

    def _remote_environment(self, extra: Mapping[str, str] | None = None) -> Mapping[str, str]:
        """Expose request credentials only to hook-disabled Git remote operations."""
        credentials = {**self._environment, **_credential_environment(extra)}
        return {
            **self._local_environment(),
            **credentials,
            # Prevent a checkout-controlled credential helper from receiving the request
            # token. Git's fixed askpass helper remains the only credential boundary.
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "credential.helper",
            "GIT_CONFIG_VALUE_0": "",
        }

    def _environment_with_os(self) -> Mapping[str, str]:
        """Compatibility alias returning the credential-free local Git environment."""
        return self._local_environment()

    async def _discard_spent_workspace(self, target: Path) -> bool:
        """Remove a previous attempt's clone when it holds no work, so a retry can clone again.

        Reached only when the journal has *no* confirmed result for this exact clone -- a
        replay is served by `_reused_clone` and never arrives here -- so nothing durable
        points at what this removes.

        The case that made it necessary is a retry across the change that put the configured
        branch into the clone's journalled input. That input is what identifies the operation,
        so a feature whose clone confirmed before the change asks for a *different* clone
        afterwards; the journal rightly declines to replay, and the old attempt's directory
        was then a permanent `clone destination already exists`. Every feature on disk from
        before the change would have been unretryable, AB-Feature-221 among them -- the one
        this whole change exists to unblock.

        Re-cloning rather than adopting the directory is the conservative choice: the old
        checkout is on whatever branch the old operation used, and carries that attempt's
        stale refs and untracked build output. AB-Feature-218 is what inheriting untracked
        build output costs.

        Refuses to touch a workspace holding uncommitted or untracked files, or one Git
        cannot describe. An unfinished attempt is somebody's work, and the caller's error
        naming the path is a better outcome than deleting it.
        """
        if self._workspace_root is None:
            return False
        if target == self._workspace_root or self._workspace_root not in target.parents:
            return False
        if not (target / ".git").exists():
            return False
        # `None` is "git could not answer", which is not the same as "nothing to lose".
        status = await self._git_output(target, "status", "--porcelain")
        if status is None or status:
            return False
        shutil.rmtree(target)
        return True

    async def _cleanup_incomplete_clone(self, workspace: Path) -> None:
        """Remove only an owned clone path that does not contain valid Git metadata."""
        if self._workspace_root is None:
            return
        if workspace == self._workspace_root or self._workspace_root not in workspace.parents:
            return
        if not workspace.exists() or (workspace / ".git").exists():
            return
        for path in sorted(workspace.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        workspace.rmdir()


def _require_success(
    stderr: str,
    return_code: int | None,
    operation: str,
    *,
    credential: CredentialProvenance | None = None,
    branch: str | None = None,
) -> None:
    """Map a Git failure to platform-owned evidence without trusting hook output.

    ``credential`` is supplied only by the operations that authenticate against a remote --
    the clone and the push. For those, the output is read for an authentication refusal
    before it is discarded, and a refusal names the credential and its age instead of
    reporting a generic failure. Which credential is not derivable from anything else the
    caller can see, which is why it is passed in rather than looked up.

    ``branch`` is supplied by the operations that name a ref the caller configured. For
    those, a "no such ref" answer is separated from a fault before the output is discarded,
    because the two call for opposite handling: one is weather and the other is a field
    somebody corrects. The branch name travels because it came *from* this platform's own
    configuration -- it is not remote output, and naming it is the difference between a
    person fixing a setting and a person reading a stack trace.

    The output itself is still never persisted, on any branch. A marker in it selects a
    sentence this platform composed; Git's own words, which name remote URLs and repository
    paths, do not leave this function.
    """
    if return_code == 0:
        return
    code = subprocess_result_code(return_code=return_code, prefix=operation)
    refused = credential is not None and authentication_was_refused(stderr)
    missing_branch = branch is not None and branch_was_not_found(stderr)
    del stderr
    if missing_branch and branch is not None:
        # Checked before the credential, and the order matters: a refusal names a credential
        # to replace, and telling somebody to replace a working token because their branch
        # name has a typo is the more expensive wrong answer. The markers are disjoint, so
        # this only decides which is reported when Git somehow emits both.
        diagnostic = (
            f"{operation} found no branch named {branch!r} in this repository. That is the "
            "branch this repository is configured to build from, so nothing can be based on "
            f"it until the name is corrected ({code}). Repository hook output was not "
            "persisted."
        )
        raise GitBranchNotFoundError(diagnostic, diagnostics=[diagnostic])
    if refused and credential is not None:
        diagnostic = (
            f"{operation} was refused by the remote: {credential.described()} was not "
            "accepted, so it may have expired or had its access revoked. Replace it before "
            f"running this again ({code}). Repository hook output was not persisted."
        )
        raise GitAuthenticationError(diagnostic, diagnostics=[diagnostic])
    diagnostic = f"{operation} failed ({code}). Repository hook output was not persisted."
    raise GitAdapterError(diagnostic, diagnostics=[diagnostic])


def _credential_environment(
    environment: Mapping[str, str] | None,
) -> dict[str, str]:
    """Accept only the fixed askpass contract used by request-scoped Git operations."""
    supplied = dict(environment or {})
    unsupported = sorted(set(supplied) - _REMOTE_GIT_ENVIRONMENT_KEYS)
    if unsupported:
        msg = f"Git credential environment contains unsupported keys: {unsupported}"
        raise GitSafetyError(msg)
    return {str(key): str(value) for key, value in supplied.items()}


def _update_digest_field(digest: Any, value: bytes) -> None:
    """Add one unambiguous length-delimited field to a SHA-256 digest."""
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _git_file_mode(path: Path) -> str:
    """Return the Git tree mode corresponding to one reviewed workspace path."""
    if path.is_symlink():
        return "120000"
    return "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"


def _tree_entries_fingerprint(entries: Sequence[tuple[str, str, str]]) -> str:
    """Hash canonical path, mode, and blob triples without leaking source bytes."""
    digest = hashlib.sha256()
    for path, mode, blob in sorted(entries):
        _update_digest_field(digest, path.encode("utf-8"))
        _update_digest_field(digest, mode.encode("ascii"))
        _update_digest_field(digest, blob.encode("ascii"))
    return digest.hexdigest()


def _reused_clone(target: Path, _payload: dict[str, object]) -> Path:
    """Reject a replay when the durable completed clone no longer exists locally.

    Deliberately does not check which branch the workspace is on. Two reasons: the base
    branch is part of the clone's journalled input, so a reuse can only ever be a clone taken
    from the same branch; and a replay after a crash lands here once `create_branch` has
    already switched the workspace onto the feature branch, so requiring the base branch to
    be the current HEAD would reject exactly the recovery this reuse exists for.
    """
    if not target.is_dir() or not (target / ".git").exists():
        msg = "completed clone operation cannot be reused because workspace is unavailable"
        raise GitAdapterError(msg)
    return target


__all__ = ["InterruptibleGitService", "reviewed_content_fingerprint"]
