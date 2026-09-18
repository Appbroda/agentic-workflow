"""Building from a branch that is not the repository's default one.

A repository spec carries `default_branch`, and it means "the branch this feature is based on
and whose PR it targets" -- not "whatever this repository's HEAD happens to be". Those two
were the same value in every repository the platform had been pointed at until
AB-Feature-221, whose two repositories were configured to build from an integration branch.

`git clone` without `--branch` checks out the *remote's* HEAD and creates a local ref for
that branch alone; every other branch arrives only as `refs/remotes/origin/<name>`. A bare
name does not resolve through that, because gitrevisions tries `refs/heads/<name>` and
`refs/remotes/<name>` and never `refs/remotes/origin/<name>`. So `create_branch`'s
`rev-parse <base_branch>` failed, three fault retries and 140 seconds of backoff went on a
ref that could not appear, and the feature was recorded as a platform defect.

Every test here runs real `git` against a real bare remote. A double would have agreed with
whatever the adapter did, and what was wrong was the adapter's own command line.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from adapters.git_adapter import GitAdapterError, GitBranchNotFoundError
from adapters.interruptible_git import InterruptibleGitService
from services.cancellation import MockCancellationToken


def _run(*command: str, cwd: Path) -> None:
    """Run one Git command, failing the test loudly with its output."""
    subprocess.run(command, cwd=cwd, check=True, capture_output=True, timeout=60)


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """A bare remote whose HEAD is `master` and which also carries an integration branch.

    Deliberately the shape AB-Feature-221 met: the branch the feature is configured to build
    from exists, is not HEAD, and carries a commit HEAD does not have -- so a clone that
    silently used HEAD would be visibly building on the wrong tree rather than merely on a
    differently-named one.
    """
    source = tmp_path / "source"
    source.mkdir()
    _run("git", "init", "--initial-branch=master", cwd=source)
    _run("git", "config", "user.email", "platform@example.com", cwd=source)
    _run("git", "config", "user.name", "Platform", cwd=source)
    (source / "on-master.txt").write_text("master\n", encoding="utf-8")
    _run("git", "add", "-A", cwd=source)
    _run("git", "commit", "-m", "master commit", cwd=source)
    _run("git", "switch", "-c", "DAM-EXP-Automation-branch", cwd=source)
    (source / "on-integration.txt").write_text("integration\n", encoding="utf-8")
    _run("git", "add", "-A", cwd=source)
    _run("git", "commit", "-m", "integration commit", cwd=source)
    _run("git", "switch", "master", cwd=source)

    bare = tmp_path / "remote.git"
    _run("git", "clone", "--bare", str(source), str(bare), cwd=tmp_path)
    _run("git", "symbolic-ref", "HEAD", "refs/heads/master", cwd=bare)
    return bare


def _service(default_branch: str, workspace_root: Path) -> InterruptibleGitService:
    """The adapter as the runtime composes it: the repository's configured branch."""
    return InterruptibleGitService(
        default_branch=default_branch,
        cancellation_token=MockCancellationToken(),
        workspace_root=workspace_root,
    )


async def test_a_clone_checks_out_the_configured_branch_not_the_remote_head(
    remote: Path, tmp_path: Path
) -> None:
    """The fix, stated as the property that was missing.

    Asserted on the checkout rather than on the command line: what matters is that the
    workspace is on the configured branch and carries its content, not that a particular flag
    was passed.
    """
    root = tmp_path / "workspaces"
    root.mkdir()
    service = _service("DAM-EXP-Automation-branch", root)

    await service.clone(str(remote), root / "repo")

    head = subprocess.run(
        ("git", "rev-parse", "--abbrev-ref", "HEAD"),
        cwd=root / "repo",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == "DAM-EXP-Automation-branch"
    # The integration branch's own commit is present, so this is the right tree and not just
    # the right name.
    assert (root / "repo" / "on-integration.txt").is_file()


async def test_the_configured_branch_resolves_by_bare_name_after_a_clone(
    remote: Path, tmp_path: Path
) -> None:
    """The exact command AB-Feature-221 died on, run against the fixed clone.

    `create_branch` resolves the base branch by bare name. Before the clone pinned it, this
    was `fatal: ambiguous argument 'DAM-EXP-Automation-branch': unknown revision`, because
    the ref existed only as `refs/remotes/origin/DAM-EXP-Automation-branch`.
    """
    root = tmp_path / "workspaces"
    root.mkdir()
    service = _service("DAM-EXP-Automation-branch", root)
    await service.clone(str(remote), root / "repo")

    resolved = subprocess.run(
        ("git", "rev-parse", "DAM-EXP-Automation-branch"),
        cwd=root / "repo",
        capture_output=True,
        text=True,
    )

    assert resolved.returncode == 0, resolved.stderr


async def test_a_feature_branch_is_created_from_the_configured_branch(
    remote: Path, tmp_path: Path
) -> None:
    """End to end: clone then branch, which is the sequence the workstream runs.

    The ancestry assertion is the one that matters. A feature branch cut from `master` would
    still exist and still be pushable -- it would simply be based on the wrong tree, and the
    pull request would show every difference between the two branches as this feature's work.
    """
    root = tmp_path / "workspaces"
    root.mkdir()
    service = _service("DAM-EXP-Automation-branch", root)
    workspace = root / "repo"
    await service.clone(str(remote), workspace)

    await service.create_branch(
        workspace, "feature-login-theme", base_branch="DAM-EXP-Automation-branch"
    )

    head = subprocess.run(
        ("git", "rev-parse", "--abbrev-ref", "HEAD"),
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == "feature-login-theme"
    ancestry = subprocess.run(
        ("git", "merge-base", "--is-ancestor", "DAM-EXP-Automation-branch", "HEAD"),
        cwd=workspace,
        capture_output=True,
    )
    assert ancestry.returncode == 0, "the feature branch is not based on the configured branch"

    # Cut from the integration tip and nothing else: freshly created with no commits of its
    # own, so its HEAD *is* that tip. Not asserted by excluding `origin/master` from the
    # ancestry -- the integration branch descends from master, so master is legitimately an
    # ancestor of anything cut from it, and asserting otherwise would be asserting a false
    # thing about Git.
    def sha(revision: str) -> str:
        return subprocess.run(
            ("git", "rev-parse", revision),
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    assert sha("HEAD") == sha("DAM-EXP-Automation-branch")
    assert sha("HEAD") != sha("origin/master")
    # The content proof, which is what a wrong base would actually cost: the integration
    # branch's own file is in the tree the coding agent is about to be handed.
    assert (workspace / "on-integration.txt").is_file()


async def test_the_default_branch_still_works_unchanged(remote: Path, tmp_path: Path) -> None:
    """The ordinary case, which is every repository configured before this existed."""
    root = tmp_path / "workspaces"
    root.mkdir()
    service = _service("master", root)

    await service.clone(str(remote), root / "repo")

    head = subprocess.run(
        ("git", "rev-parse", "--abbrev-ref", "HEAD"),
        cwd=root / "repo",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == "master"
    assert not (root / "repo" / "on-integration.txt").exists()


async def test_a_branch_that_does_not_exist_names_itself_and_is_not_weather(
    remote: Path, tmp_path: Path
) -> None:
    """A typo in Settings is an answer, not a fault to be retried.

    Two properties, and the second is why this has its own exception type: the message names
    the branch so somebody can go and correct it, and the class is `GitBranchNotFoundError`
    so the workstream's fault allowance is not spent waiting for a ref to appear.
    """
    root = tmp_path / "workspaces"
    root.mkdir()
    service = _service("DAM-EXP-Automation-brnach", root)  # transposed, as a person would

    with pytest.raises(GitBranchNotFoundError) as captured:
        await service.clone(str(remote), root / "repo")

    message = str(captured.value)
    assert "DAM-EXP-Automation-brnach" in message
    assert "configured to build from" in message
    # Git's own words name the remote URL, and they do not travel into a durable diagnostic.
    assert str(remote) not in message


def test_a_missing_branch_is_refused_the_fault_allowance() -> None:
    """The classification seam, asserted where the decision is made.

    `GitBranchNotFoundError` is a `GitAdapterError`, and every other `GitAdapterError` is
    weather that earns a retry. Without this branch in the predicate the new exception type
    would be correct and the behaviour unchanged -- three backoffs on a ref that cannot
    appear, which is what AB-Feature-221 spent.
    """
    from workflows.feature_workflow import is_transient_provider_fault

    missing = GitBranchNotFoundError("no such branch", diagnostics=["no such branch"])
    weather = GitAdapterError("git clone failed", diagnostics=["git clone failed"])

    assert is_transient_provider_fault(missing) is False
    assert is_transient_provider_fault(weather) is True


async def test_the_refusal_survives_the_journal(remote: Path, tmp_path: Path) -> None:
    """The production path: a journalled clone, and the predicate applied to what escapes it.

    Worth its own test because the journal is between the adapter and the workstream, and it
    is what turns some adapter failures into an `ExternalOperationError` carrying the original
    as its cause. This path re-raises unwrapped -- asserted below rather than assumed, since
    the reason the classifier *walks* the chain is that not every path does.
    """
    from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
    from storage.db import Database
    from storage.external_operation_store import ExternalOperationJournal
    from workflows.feature_workflow import is_transient_provider_fault

    root = tmp_path / "workspaces"
    root.mkdir()
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'journal.db'}")
    await database.create_schema()
    try:
        service = InterruptibleGitService(
            default_branch="no-such-branch",
            cancellation_token=MockCancellationToken(),
            workspace_root=root,
            operation_executor=ExternalOperationExecutor(
                journal=ExternalOperationJournal(database),
                cancellation_token=MockCancellationToken(),
                scope=ExternalOperationScope(
                    workflow_id="feature-branch-probe",
                    feature_id="feature-branch-probe",
                    child_workflow_id="feature-branch-probe:backend",
                    repository_id="backend",
                ),
            ),
        )

        with pytest.raises(Exception) as captured:  # noqa: PT011 - the wrap type is the subject
            await service.clone(str(remote), root / "repo")

        raised = captured.value
        # However the journal presents it, the branch is named and the allowance is refused.
        assert isinstance(raised, GitBranchNotFoundError)
        assert "no-such-branch" in str(raised)
        assert is_transient_provider_fault(raised) is False

        # And wrapped, which is how the failures that *are* wrapped arrive: the classifier
        # walks the cause chain, so a wrap must not restore the fault allowance. Constructed
        # rather than provoked, because which paths wrap is the journal's business and this
        # asserts the walk regardless of it.
        wrapped = RuntimeError("external operation failed before a confirmed result")
        wrapped.__cause__ = raised
        assert is_transient_provider_fault(wrapped) is False
    finally:
        await database.dispose()


async def test_an_empty_configured_branch_is_refused_before_any_clone(tmp_path: Path) -> None:
    """A blank branch would clone the remote's HEAD and look like it worked.

    Which is the same class of mistake as the empty-string settings default that locked the
    console out: a blank is not a value, and accepting one here would silently restore the
    behaviour this change removes.
    """
    root = tmp_path / "workspaces"
    root.mkdir()
    service = _service("   ", root)

    with pytest.raises(GitAdapterError, match="configured to build from"):
        await service.clone("https://github.com/example/repo", root / "repo")


async def test_a_retry_reclones_over_a_spent_workspace_from_the_old_key(
    remote: Path, tmp_path: Path
) -> None:
    """The upgrade path, which the change above would otherwise have broken.

    `base_branch` is part of the clone's journalled input, so a feature whose clone confirmed
    before this change asks for a *different* operation afterwards. The journal correctly
    declines to replay -- and the old attempt's directory was then a permanent
    `clone destination already exists`, making every feature already on disk unretryable.
    AB-Feature-221 is one of them, so this is the difference between the fix landing and the
    feature it exists for staying stuck.

    Set up as the live workspaces actually were: a clean checkout of the same repository, left
    on the remote's HEAD by the very bug being fixed.
    """
    root = tmp_path / "workspaces"
    root.mkdir()
    workspace = root / "repo"
    _run("git", "clone", str(remote), str(workspace), cwd=tmp_path)  # the old, branchless clone
    assert not (workspace / "on-integration.txt").exists()

    await _service("DAM-EXP-Automation-branch", root).clone(str(remote), workspace)

    head = subprocess.run(
        ("git", "rev-parse", "--abbrev-ref", "HEAD"),
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == "DAM-EXP-Automation-branch"
    assert (workspace / "on-integration.txt").is_file()


async def test_a_workspace_holding_work_is_never_discarded(remote: Path, tmp_path: Path) -> None:
    """The limit on the above: unfinished work outranks a convenient retry.

    Both an edit to a tracked file and an untracked file, because `--porcelain` reports them
    the same way and a check that caught only one would still delete somebody's attempt.
    """
    root = tmp_path / "workspaces"
    root.mkdir()

    # (subdirectory, file to write) -- an edit to a tracked file, then a brand new file.
    for filename, written in (("tracked", "on-master.txt"), ("untracked", "scratch.py")):
        workspace = root / filename
        _run("git", "clone", str(remote), str(workspace), cwd=tmp_path)
        (workspace / written).write_text("half-done\n", encoding="utf-8")

        with pytest.raises(GitAdapterError, match="already exists"):
            await _service("DAM-EXP-Automation-branch", root).clone(str(remote), workspace)

        # Still there, still theirs.
        assert workspace.is_dir()
        assert subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()


async def test_a_destination_that_is_not_a_checkout_is_still_refused(tmp_path: Path) -> None:
    """A directory Git knows nothing about is not a spent workspace, and is not cloned over.

    The pre-existing guard, kept: whatever put arbitrary files at a workspace path, deleting
    them is not this adapter's decision to make.
    """
    root = tmp_path / "workspaces"
    root.mkdir()
    occupied = root / "repo"
    occupied.mkdir()
    (occupied / "important.txt").write_text("not a clone\n", encoding="utf-8")

    with pytest.raises(GitAdapterError, match="already exists"):
        await _service("master", root).clone("https://github.com/example/repo", occupied)

    assert (occupied / "important.txt").is_file()
