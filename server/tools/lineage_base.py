"""Where a workstream's branch left the default branch, and everything it changed since.

Two questions, one baseline. The reviewer asks the first to quote a large file's changed
regions against something other than ``HEAD``; the completeness gate asks the second to
decide whether the categories a plan promised exist in the *workstream* rather than in one
attempt's diff.

``LineageBaseRevision`` lived in the reviewer and was private to it. It is here because a
second copy of it is exactly where its ``is_branch_point`` fence would go missing -- that
property is the load-bearing half, and a re-spelling of ``merge-base`` that dropped it would
read as evidence while answering from ``HEAD``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from services.cancellation import CancellationToken
from services.process_runner import ProcessRunner, sanitized_subprocess_environment

_BASELINE_TIMEOUT_SECONDS = 30.0
# `git diff --name-status` against the base, so committed work and uncommitted tracked
# modifications arrive in one read: an attempt commits only after approval, so a branch
# mid-workstream carries approved attempts in `HEAD` and every failed attempt's edits in the
# worktree.
_STATUS_MODIFIED = frozenset({"M", "T", "R"})
_STATUS_DELETED = frozenset({"D"})

# The one path on a child branch that belongs to no attempt: the platform writes the contract
# projection into every child workspace on every attempt, so it is permanently untracked there
# and every reader of this module's evidence has to exclude it. Named here, where both readers
# already import from, rather than spelled twice.
CONTRACT_PROJECTION_PATH = "openapi.yaml"


class LineageBaseRevision:
    """The revision this review's change evidence is measured against, resolved once.

    Not ``HEAD``. An approved earlier attempt commits its work to the working branch, so by
    the time a later attempt in the same lineage is reviewed, ``HEAD`` already contains changes
    this review is still attesting to -- and a diff against it reports them as absent. The
    baseline is therefore where the working branch left the default branch, which is the union
    of everything the attempts on it have written.

    Repository-agnostic: the branch name comes from the workspace descriptor, and both the
    remote-tracking and local spellings are tried because a fresh clone has only the first and
    a locally-created checkout may have only the second. When neither resolves -- no
    repository, an unborn branch, a checkout with no shared ancestor -- the answer is ``HEAD``,
    which is exactly today's behaviour rather than a new failure mode.
    """

    def __init__(
        self,
        *,
        workspace: Path,
        default_branch: str,
        process_runner: ProcessRunner,
        cancellation_token: CancellationToken,
    ) -> None:
        """Record what the lookup needs without performing it."""
        self._workspace = workspace
        self._default_branch = default_branch
        self._process_runner = process_runner
        self._cancellation_token = cancellation_token
        self._resolved: str | None = None
        self._is_branch_point = False

    async def resolve(self) -> str:
        """Return the baseline revision, asking Git at most once per evidence build."""
        if self._resolved is not None:
            return self._resolved
        self._resolved = "HEAD"
        for reference in (f"origin/{self._default_branch}", self._default_branch):
            result = await self._process_runner.run(
                ("git", "merge-base", "HEAD", reference),
                self._workspace,
                _BASELINE_TIMEOUT_SECONDS,
                self._cancellation_token,
                sanitized_subprocess_environment(),
            )
            revision = result.stdout.strip()
            if result.succeeded and not result.output_truncated and revision:
                self._resolved = revision
                self._is_branch_point = True
                break
        return self._resolved

    @property
    def is_branch_point(self) -> bool:
        """Whether the resolved revision is the lineage's branch point rather than ``HEAD``.

        Load-bearing, and it is the fence around the one way this whole path could approve
        blind. "No hunks" from a diff against a real branch point means the file genuinely
        holds the baseline's bytes and there is nothing about it to attest to. The same answer
        from the ``HEAD`` fallback means only that no *uncommitted* change is left -- which is
        precisely the state run 186 was in -- so it must be read as unanswerable and refused,
        not as nothing having changed.
        """
        return self._is_branch_point


@dataclass(frozen=True, slots=True)
class BranchChangeEvidence:
    """Every path this workstream's branch changed, and which of them already existed.

    ``modified_paths`` is a subset of ``paths``: a file that was present at the baseline and
    was edited, which is the only thing that answers the ``integration`` category -- adding
    modules only leaves them unreachable.
    """

    paths: tuple[str, ...]
    modified_paths: tuple[str, ...]


async def branch_change_evidence(
    *,
    workspace: Path,
    baseline: LineageBaseRevision,
    process_runner: ProcessRunner,
    cancellation_token: CancellationToken,
    timeout_seconds: float = _BASELINE_TIMEOUT_SECONDS,
    excluded_paths: Sequence[str] = (),
) -> BranchChangeEvidence | None:
    """Report what this branch changed since its baseline, or ``None`` if Git cannot say.

    Two reads unioned, and the second is the one whose absence bites. ``git diff`` against the
    baseline covers committed work and tracked modifications together; ``git ls-files
    --others`` covers files an attempt created and has not committed, which on this platform is
    every file written by an attempt that has not yet been approved. Take only the committed
    side and the case that repeats is missed: attempt 1 writes the tests and fails the commit
    gate on lint, attempt 2 fixes one line -- the tests are in neither ``HEAD`` nor attempt 2's
    completion, only in the worktree.

    ``None`` means the question was not answered, never "nothing changed", so a caller can fall
    to its next evidence source and say which one it used. It is returned when the baseline did
    not resolve to a real branch point (``is_branch_point`` false: a diff against the ``HEAD``
    fallback would report an approved attempt's own committed work as absent), when either read
    failed, or when either read was truncated at the runner's output cap and the path list is
    therefore incomplete.

    Deletions are dropped: removing a test file is not evidence that the branch has tests.
    """
    base = await baseline.resolve()
    if not baseline.is_branch_point:
        return None
    excluded = set(excluded_paths)
    paths: list[str] = []
    modified: list[str] = []
    changed = await process_runner.run(
        ("git", "diff", "--name-status", base),
        workspace,
        timeout_seconds,
        cancellation_token,
        sanitized_subprocess_environment(),
    )
    if not changed.succeeded or changed.output_truncated:
        return None
    for line in changed.stdout.splitlines():
        fields = [field for field in line.split("\t") if field]
        if len(fields) < 2:
            continue
        status, path = fields[0][:1].upper(), fields[-1]
        if status in _STATUS_DELETED or path in excluded:
            continue
        paths.append(path)
        if status in _STATUS_MODIFIED:
            # A rename counts: the file existed at the baseline and this branch changed it,
            # which is what the integration category asks. A copy does not, because the file
            # it produced is new.
            modified.append(path)
    untracked = await process_runner.run(
        ("git", "ls-files", "--others", "--exclude-standard"),
        workspace,
        timeout_seconds,
        cancellation_token,
        sanitized_subprocess_environment(),
    )
    if not untracked.succeeded or untracked.output_truncated:
        return None
    for line in untracked.stdout.splitlines():
        path = line.strip()
        if path and path not in excluded:
            paths.append(path)
    return BranchChangeEvidence(
        paths=tuple(dict.fromkeys(paths)), modified_paths=tuple(dict.fromkeys(modified))
    )


__all__ = [
    "CONTRACT_PROJECTION_PATH",
    "BranchChangeEvidence",
    "LineageBaseRevision",
    "branch_change_evidence",
]
