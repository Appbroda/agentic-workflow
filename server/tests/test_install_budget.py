"""68-: dependency installation runs on its own clock, and says so when it runs out.

AB-Feature-203, 204 and 205 all ended in `failed_requires_human` on dependency installation
and nothing else, 205 with a warm cache and 37 GB free, so neither environmental excuse
applies. `npm ci` was being given the 120 seconds sized for an agent call while a successful
install on that machine had taken 303, and a timeout there is deliberately terminal -- so the
budget alone decided whether the run reached a coding call. These tests pin the budget to its
own setting, pin the four-plus other consumers of the shared one to their existing values, and
pin the two sentences an operator reads when the budget is exceeded anyway.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from configs.settings import Settings
from services.cancellation import MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.process_runner import ProcessResult
from state.external_operations import ExternalOperationStatus
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from tools import repository_preflight as preflight_module
from tools.lint_capabilities import RepositoryLintCapabilities
from tools.repository_preflight import (
    RepositoryPreflight,
    install_failure_is_transient,
)

_SERVER_ROOT = Path(__file__).resolve().parents[1]


def _configured(name: str) -> int:
    """Read one value out of the configuration the deployment actually loads."""
    text = (_SERVER_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    return int(yaml.safe_load(text)[name])


# The exact strings a non-zero `npm ci` exit has always produced. Written out rather than
# imported so that changing them has to change this test too, which is what T4 is for.
_EXIT_FAILURE_DESCRIPTION = "Deterministic dependency installation failed before coding began."
_EXIT_FAILURE_ACTION = (
    "Resolve the checked-in lockfile or registry/runtime issue, then start a fresh "
    "replacement workstream."
)
_EXIT_FAILURE_ERROR_MESSAGE = (
    "external operation failed before a confirmed result (_DependencyInstallationFailed)"
)
# The install as it is actually spawned, `--no-audit` and all. Written out rather than
# imported from the module under test for the same reason as the strings above: a test that
# reads the command back out of the code it is testing cannot fail when that command changes.
_INSTALL_COMMAND = ("npm", "ci", "--no-audit")


class _TimeoutRecordingRunner:
    """Record the timeout every subprocess is actually given, and answer as instructed."""

    def __init__(
        self,
        *,
        install_result: ProcessResult | None = None,
        node_version: str = "22.14.0",
    ) -> None:
        self.timeouts: list[tuple[tuple[str, ...], float]] = []
        self._install_result = install_result
        self._node_version = node_version

    async def run(
        self,
        command: Any,
        cwd: Path,
        timeout_seconds: float,
        *_args: Any,
        **_kwargs: Any,
    ) -> ProcessResult:
        recorded = tuple(str(part) for part in command)
        self.timeouts.append((recorded, float(timeout_seconds)))
        if len(recorded) == 2 and recorded[1] == "--version":
            return ProcessResult(
                command=recorded,
                return_code=0,
                stdout=f"{self._node_version}\n",
                stderr="",
                duration_seconds=0.01,
            )
        if self._install_result is not None:
            return self._install_result
        return ProcessResult(
            command=recorded,
            return_code=0,
            stdout="installed",
            stderr="",
            duration_seconds=0.01,
        )

    def timeout_for(self, executable: str, *arguments: str) -> float:
        """Return the single timeout handed to one command, failing on ambiguity."""
        wanted = (executable, *arguments)
        matching = [timeout for command, timeout in self.timeouts if command == wanted]
        assert matching, f"{wanted} was never run; ran {[c for c, _ in self.timeouts]}"
        assert len(set(matching)) == 1, f"{wanted} ran under several timeouts: {matching}"
        return matching[0]


def _timed_out_install(*, duration_seconds: float) -> ProcessResult:
    """A subprocess the runner killed at its budget: no exit code, `timed_out` set."""
    return ProcessResult(
        command=_INSTALL_COMMAND,
        return_code=None,
        stdout="",
        stderr="",
        duration_seconds=duration_seconds,
        timed_out=True,
    )


def _failed_install(*, return_code: int = 1) -> ProcessResult:
    """A subprocess that ran to completion and reported a real, non-zero answer."""
    return ProcessResult(
        command=_INSTALL_COMMAND,
        return_code=return_code,
        stdout="",
        stderr="npm ERR! Missing: left-pad@1.3.0 from lock file",
        duration_seconds=12.5,
    )


def _node_repository(root: Path, name: str, *, declares_engine: bool = False) -> Path:
    """One checkout whose only preflight work is a deterministic npm install.

    `declares_engine` is what makes the preflight probe `node --version` at all: without an
    `engines.node` range there is no declaration to check, so no probe runs and there is no
    second budget to compare the install's against.
    """
    repository = root / name
    repository.mkdir()
    engines = ',"engines":{"node":">=22 <23"}' if declares_engine else ""
    (repository / "package.json").write_text(f'{{"name":"{name}"{engines}}}', encoding="utf-8")
    (repository / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")
    return repository


def _install_issue(result: Any) -> Any:
    """The one blocking issue the install produced."""
    issues = [
        issue
        for issue in result.blocking_issues
        if issue.issue_id == "DEPENDENCY_INSTALLATION_FAILED"
    ]
    assert len(issues) == 1, [issue.issue_id for issue in result.blocking_issues]
    return issues[0]


async def _journaled_preflight(
    tmp_path: Path,
    *,
    runner: Any,
    workflow_id: str,
    dependency_install_timeout_seconds: float = 600.0,
) -> tuple[RepositoryPreflight, ExternalOperationJournal]:
    """A preflight whose install is journaled, so the operation row can be read back."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / f'{workflow_id}.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    preflight = RepositoryPreflight(
        default_timeout_seconds=120.0,
        dependency_install_timeout_seconds=dependency_install_timeout_seconds,
        process_runner=runner,
        cancellation_token=MockCancellationToken(),
        operation_executor=ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(workflow_id=workflow_id, repository_id="frontend"),
        ),
    )
    return preflight, journal


# --------------------------------------------------------------------------------------
# T1 -- the install runs on its own clock
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_install_subprocess_is_given_the_install_budget(tmp_path: Path) -> None:
    """`npm ci` runs under `dependency_install_timeout_seconds`, not the agent call's budget.

    The effect, not the reading: what killed 203 was the number handed to the process runner,
    so that is the number asserted. A test that only checked the setting had been loaded would
    have passed against the defect.
    """
    runner = _TimeoutRecordingRunner()

    result = await RepositoryPreflight(
        default_timeout_seconds=120.0,
        dependency_install_timeout_seconds=600.0,
        process_runner=runner,
        cancellation_token=MockCancellationToken(),
    ).run(_node_repository(tmp_path, "frontend"), repository_id="frontend")

    assert result.dependency_install_status == "installed"
    assert runner.timeout_for(*_INSTALL_COMMAND) == 600.0
    assert runner.timeout_for(*_INSTALL_COMMAND) != 120.0


@pytest.mark.asyncio
async def test_every_install_try_inside_one_attempt_gets_the_install_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget bounds each subprocess run, and `_install_with_retries` makes several.

    This is why an install that took 303s succeeded under a nominal 120s budget, and it is the
    reason the new number has to reach the loop body rather than the call around it. It is also
    what makes the product worth stating: three tries at 600s plus the backoffs is the real
    ceiling on one attempt, and raising the budget raises that too.
    """
    monkeypatch.setattr(preflight_module, "_INSTALL_FAULT_BACKOFF_SECONDS", 0.0)

    class _FlakeThenSucceed(_TimeoutRecordingRunner):
        def __init__(self) -> None:
            super().__init__()
            self._install_calls = 0

        async def run(
            self, command: Any, cwd: Path, timeout_seconds: float, *args: Any, **kwargs: Any
        ) -> ProcessResult:
            recorded = tuple(str(part) for part in command)
            if recorded == _INSTALL_COMMAND:
                self.timeouts.append((recorded, float(timeout_seconds)))
                self._install_calls += 1
                if self._install_calls <= preflight_module._ALLOWED_INSTALL_FAULTS:  # noqa: SLF001
                    return ProcessResult(
                        command=recorded,
                        return_code=1,
                        stdout="",
                        stderr="npm ERR! request to registry ETIMEDOUT",
                        duration_seconds=1.0,
                    )
                return ProcessResult(
                    command=recorded,
                    return_code=0,
                    stdout="installed",
                    stderr="",
                    duration_seconds=1.0,
                )
            return await super().run(command, cwd, timeout_seconds, *args, **kwargs)

    runner = _FlakeThenSucceed()

    result = await RepositoryPreflight(
        default_timeout_seconds=120.0,
        dependency_install_timeout_seconds=600.0,
        process_runner=runner,
        cancellation_token=MockCancellationToken(),
    ).run(_node_repository(tmp_path, "flaky"), repository_id="frontend")

    install_timeouts = [
        timeout for command, timeout in runner.timeouts if command == _INSTALL_COMMAND
    ]
    assert result.dependency_install_status == "installed"
    assert len(install_timeouts) == preflight_module._ALLOWED_INSTALL_FAULTS + 1  # noqa: SLF001
    assert set(install_timeouts) == {600.0}


# --------------------------------------------------------------------------------------
# T2 -- the other consumers are untouched
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_version_probe_keeps_its_own_narrowing(tmp_path: Path) -> None:
    """The version probe still narrows the shared budget and never sees the install's.

    The distinct values make the assertion discriminating: 20 for the shared budget, 600 for
    the install. A probe that had started inheriting the install budget would run for ten
    minutes asking `node` for a version string, which is the failure mode a blanket increase
    produces and this test refuses.
    """
    runner = _TimeoutRecordingRunner()

    await RepositoryPreflight(
        default_timeout_seconds=20.0,
        dependency_install_timeout_seconds=600.0,
        process_runner=runner,
        cancellation_token=MockCancellationToken(),
    ).run(_node_repository(tmp_path, "frontend", declares_engine=True), repository_id="frontend")

    assert runner.timeout_for("node", "--version") == min(20.0, 30.0)
    assert runner.timeout_for(*_INSTALL_COMMAND) == 600.0


@pytest.mark.asyncio
async def test_the_lint_capability_probe_still_runs_under_the_agent_budget(
    tmp_path: Path,
) -> None:
    """The probe `feature_runtime` builds is given `default_agent_timeout_seconds`, unchanged."""
    repository = _node_repository(tmp_path, "frontend")
    (repository / "src").mkdir()
    (repository / "src" / "app.js").write_text("export const a = 1;\n", encoding="utf-8")
    # The probe only runs a checkout-local binary, never one from the host PATH.
    binaries = repository / "node_modules" / ".bin"
    binaries.mkdir(parents=True)
    (binaries / "eslint").write_text("#!/bin/sh\n", encoding="utf-8")
    runner = _TimeoutRecordingRunner()

    await RepositoryLintCapabilities(
        process_runner=runner,
        cancellation_token=MockCancellationToken(),
        timeout_seconds=float(_configured("default_agent_timeout_seconds")),
    ).describe(repository, ["src/app.js"])

    assert runner.timeouts, "the lint capability probe ran no subprocess"
    assert {timeout for _command, timeout in runner.timeouts} == {120.0}


def test_the_shared_agent_budget_was_not_raised() -> None:
    """`default_agent_timeout_seconds` keeps its value; the install got a number of its own.

    Read from the deployed configuration and from the field's own default, because the two
    have to agree: a host that never overrides the setting takes the default, and a host that
    reads `config.yaml` takes the file. 600 in both, and 120 still meaning one agent call.
    """
    assert _configured("default_agent_timeout_seconds") == 120
    assert _configured("dependency_install_timeout_seconds") == 600
    field = Settings.model_fields["dependency_install_timeout_seconds"]
    assert field.default == 600


def test_only_the_install_reads_the_install_budget() -> None:
    """No consumer of the shared budget was moved onto the install's, in either direction.

    The spec's own list of "four other call sites" undercounts: `default_agent_timeout_seconds`
    is read at sixteen places outside its declaration -- the source formatter's verify
    command, the reachability, assigned-file and design-value checkers, the lint capability
    probe, the generated-output clean, and more. That makes the case against a blanket increase
    stronger rather than weaker, and it is why this is asserted by count: adding a consumer to
    either budget has to be a deliberate edit to this test.

    The fifteenth is the design-value inspection and the sixteenth is the design-asset
    placement, and both belong on the shared clock for the reason the two checkers beside them
    do: each is one `git` command over a workspace, not a model call and not an install.
    """
    consumers: dict[str, list[str]] = {}
    for module in ("services/feature_runtime.py", "services/runtime.py"):
        source = (_SERVER_ROOT / module).read_text(encoding="utf-8")
        consumers[module] = re.findall(r"_settings\.(\w*timeout\w*)", source)

    shared = [name for names in consumers.values() for name in names]
    assert shared.count("default_agent_timeout_seconds") == 16
    # Exactly the two `RepositoryPreflight` constructions, and nothing else.
    assert shared.count("dependency_install_timeout_seconds") == 2
    assert consumers["services/runtime.py"].count("dependency_install_timeout_seconds") == 1

    # The shared clock still bounds the version probe, narrowed as it always was, and the
    # install clock reaches exactly one subprocess call. Statements only, so that a mention in
    # a comment cannot satisfy or break either count.
    statements = [
        line.strip()
        for line in (_SERVER_ROOT / "tools/repository_preflight.py")
        .read_text(encoding="utf-8")
        .splitlines()
        if not line.strip().startswith("#")
    ]
    assert statements.count("min(self._timeout, 30.0),") == 1
    assert statements.count("self._install_timeout,") == 1


# --------------------------------------------------------------------------------------
# T3 -- a timeout is legible as a timeout
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_timed_out_install_names_the_budget_and_the_elapsed_time(
    tmp_path: Path,
) -> None:
    """The operation row and the blocking issue both say a budget was exceeded, with numbers.

    Answering "why is 203 failing in installing dependency" took a database session: the row
    said `install_dependencies_failed` and "failed before a confirmed result", which is equally
    true of a lockfile conflict, and only the child's `preflight_result.blocking_issues`
    carried `DEPENDENCY_INSTALLATION_TIMED_OUT`. Both now answer the question where it is
    asked.
    """
    runner = _TimeoutRecordingRunner(install_result=_timed_out_install(duration_seconds=604.0))
    preflight, journal = await _journaled_preflight(
        tmp_path, runner=runner, workflow_id="workflow-install-timeout"
    )

    result = await preflight.run(_node_repository(tmp_path, "frontend"), repository_id="frontend")

    operation = (await journal.list_operations_for_workflow("workflow-install-timeout"))[0]
    assert operation.error_code == "install_dependencies_timed_out"
    assert operation.error_code != "install_dependencies_failed"
    assert "DEPENDENCY_INSTALLATION_TIMED_OUT" in (operation.error_message or "")

    issue = _install_issue(result)
    assert issue.description == (
        "Deterministic dependency installation exceeded its budget before coding began."
    )
    assert "600s dependency install budget" in issue.recommended_action
    assert "after 604s" in issue.recommended_action
    # The advice 203 was actually given, and it was wrong: both lockfiles were valid and the
    # registry answered 200 in 0.73s from inside the container. The sentence now says so
    # rather than sending the reader to look at either of them.
    assert _EXIT_FAILURE_ACTION not in issue.recommended_action
    assert "does not implicate the lockfile or the registry" in issue.recommended_action
    assert "DEPENDENCY_INSTALLATION_TIMED_OUT" in issue.evidence


@pytest.mark.asyncio
async def test_an_unmeasured_elapsed_time_is_omitted_rather_than_reported_as_zero(
    tmp_path: Path,
) -> None:
    """A result reconstructed without a duration drops the clause instead of claiming 0s."""
    runner = _TimeoutRecordingRunner(install_result=_timed_out_install(duration_seconds=0.0))

    result = await RepositoryPreflight(
        default_timeout_seconds=120.0,
        dependency_install_timeout_seconds=600.0,
        process_runner=runner,
        cancellation_token=MockCancellationToken(),
    ).run(_node_repository(tmp_path, "frontend"), repository_id="frontend")

    issue = _install_issue(result)
    assert "600s dependency install budget" in issue.recommended_action
    assert "after" not in issue.recommended_action
    assert "after 0s" not in issue.recommended_action


@pytest.mark.asyncio
async def test_the_budget_in_the_sentence_is_the_budget_that_was_enforced(
    tmp_path: Path,
) -> None:
    """A host running a different budget says its own number, not the default."""
    runner = _TimeoutRecordingRunner(install_result=_timed_out_install(duration_seconds=905.0))

    result = await RepositoryPreflight(
        default_timeout_seconds=120.0,
        dependency_install_timeout_seconds=900.0,
        process_runner=runner,
        cancellation_token=MockCancellationToken(),
    ).run(_node_repository(tmp_path, "frontend"), repository_id="frontend")

    assert runner.timeout_for(*_INSTALL_COMMAND) == 900.0
    assert "900s dependency install budget after 905s" in _install_issue(result).recommended_action


# --------------------------------------------------------------------------------------
# T4 -- a failing install is still a failing install
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_non_zero_install_exit_keeps_todays_classification_byte_identical(
    tmp_path: Path,
) -> None:
    """A real answer from `npm ci` reads exactly as it did before the budget was split out.

    The strings are literals here on purpose: legibility was added for the timeout only, so
    anything that reads a non-zero-exit row does not have to learn a second spelling.
    """
    runner = _TimeoutRecordingRunner(install_result=_failed_install())
    preflight, journal = await _journaled_preflight(
        tmp_path, runner=runner, workflow_id="workflow-install-exit"
    )

    result = await preflight.run(_node_repository(tmp_path, "frontend"), repository_id="frontend")

    operation = (await journal.list_operations_for_workflow("workflow-install-exit"))[0]
    assert operation.error_code == "install_dependencies_failed"
    assert operation.error_message == _EXIT_FAILURE_ERROR_MESSAGE

    issue = _install_issue(result)
    assert result.dependency_install_status == "failed"
    assert issue.category == "dependency_configuration"
    assert issue.severity == "high"
    assert issue.description == _EXIT_FAILURE_DESCRIPTION
    assert issue.recommended_action == _EXIT_FAILURE_ACTION
    assert issue.evidence == (
        "Dependency manager `npm` did not complete successfully "
        "(DEPENDENCY_INSTALLATION_FAILED_EXIT_1)."
    )
    assert issue.automatically_repairable is False


# --------------------------------------------------------------------------------------
# T5 -- still terminal
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_timed_out_install_is_not_retried_and_still_stops_the_workstream(
    tmp_path: Path,
) -> None:
    """A budget the sandbox enforced is not a transient fault, and the fix is not a retry loop.

    Retrying spends the whole limit again to reach the same verdict; this spec buys a budget
    that fits the work and changes nothing about how many times it is asked.
    """
    runner = _TimeoutRecordingRunner(install_result=_timed_out_install(duration_seconds=604.0))

    result = await RepositoryPreflight(
        default_timeout_seconds=120.0,
        dependency_install_timeout_seconds=600.0,
        process_runner=runner,
        cancellation_token=MockCancellationToken(),
    ).run(_node_repository(tmp_path, "frontend"), repository_id="frontend")

    assert not install_failure_is_transient(_timed_out_install(duration_seconds=604.0))
    install_runs = [command for command, _timeout in runner.timeouts if command == _INSTALL_COMMAND]
    assert len(install_runs) == 1
    assert result.dependency_install_status == "failed"
    assert result.validation_readiness == "blocked"
    assert _install_issue(result).automatically_repairable is False


# --------------------------------------------------------------------------------------
# T6 -- the 203 replay
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_repositories_that_both_exceed_the_budget_report_the_budget(
    tmp_path: Path,
) -> None:
    """AB-Feature-203 exactly: both repositories over budget, neither one's lockfile at fault.

    203's two installs started ten seconds apart and ran to 378s and 274s. Both children
    ended `failed` with preflight `blocked`, and the recommended action pointed at a lockfile
    and a registry that were both fine. Same shape here, with the sentence corrected.
    """
    durations = {"AB-console-admin-2.0": 604.0, "admanager_console-2.0": 601.0}
    runners = {
        name: _TimeoutRecordingRunner(install_result=_timed_out_install(duration_seconds=duration))
        for name, duration in durations.items()
    }
    preflights = {}
    for name, runner in runners.items():
        preflight, journal = await _journaled_preflight(
            tmp_path, runner=runner, workflow_id=f"workflow-203-{name}"
        )
        preflights[name] = (preflight, journal)

    results = await asyncio.gather(
        *(
            preflights[name][0].run(_node_repository(tmp_path, name), repository_id=name)
            for name in durations
        )
    )

    for name, result in zip(durations, results, strict=True):
        assert result.dependency_install_status == "failed"
        assert result.validation_readiness == "blocked"
        issue = _install_issue(result)
        assert "600s dependency install budget" in issue.recommended_action
        assert f"after {durations[name]:.0f}s" in issue.recommended_action
        assert "Resolve the checked-in lockfile" not in issue.recommended_action
        journal = preflights[name][1]
        operation = (await journal.list_operations_for_workflow(f"workflow-203-{name}"))[0]
        assert operation.error_code == "install_dependencies_timed_out"
        assert operation.status in {
            ExternalOperationStatus.FAILED_RETRYABLE,
            ExternalOperationStatus.FAILED_TERMINAL,
        }


# --------------------------------------------------------------------------------------
# 85- F2: the install that need not run
# --------------------------------------------------------------------------------------


async def _preflight_over(
    repository: Path,
    *,
    runner: Any,
    workspace_preserved: bool,
    previous_lockfile_digest: str | None,
) -> Any:
    """Run an unjournaled preflight, which is enough to see whether the install spawned."""
    return await RepositoryPreflight(
        default_timeout_seconds=120.0,
        dependency_install_timeout_seconds=600.0,
        process_runner=runner,
        cancellation_token=MockCancellationToken(),
    ).run(
        repository,
        repository_id=repository.name,
        workspace_preserved=workspace_preserved,
        previous_lockfile_digest=previous_lockfile_digest,
    )


@pytest.mark.asyncio
async def test_a_preserved_workspace_with_an_unchanged_lockfile_does_not_reinstall(
    tmp_path: Path,
) -> None:
    """AB-Feature-218 ran `install_dependencies` on all eleven of its attempt boundaries.

    About 395 s, against workspaces the platform itself had marked
    `attempt_inputs.workspace: "preserved"`, with the lockfiles untouched throughout.
    Installing into a tree that already holds exactly those dependencies reproduces the tree
    it is looking at.
    """
    repository = _node_repository(tmp_path, "admanager")

    first = await _preflight_over(
        repository,
        runner=_TimeoutRecordingRunner(),
        workspace_preserved=False,
        previous_lockfile_digest=None,
    )

    assert first.dependency_install_status == "installed"
    assert first.dependency_lockfile_digest is not None

    runner = _TimeoutRecordingRunner()
    second = await _preflight_over(
        repository,
        runner=runner,
        workspace_preserved=True,
        previous_lockfile_digest=first.dependency_lockfile_digest,
    )

    # Recorded as its own status, so an operator can tell a skip from a run that did nothing.
    assert second.dependency_install_status == "skipped_unchanged_lockfile"
    assert second.dependency_lockfile_digest == first.dependency_lockfile_digest
    assert _INSTALL_COMMAND not in [command for command, _timeout in runner.timeouts]
    assert second.validation_readiness != "blocked"


@pytest.mark.asyncio
async def test_a_moved_lockfile_still_installs_on_a_preserved_workspace(tmp_path: Path) -> None:
    """The digest is over the bytes, because that is the only thing that answers the question."""
    repository = _node_repository(tmp_path, "admanager")
    first = await _preflight_over(
        repository,
        runner=_TimeoutRecordingRunner(),
        workspace_preserved=False,
        previous_lockfile_digest=None,
    )
    (repository / "package-lock.json").write_text(
        '{"lockfileVersion":3,"packages":{"node_modules/multer":{}}}', encoding="utf-8"
    )

    runner = _TimeoutRecordingRunner()
    second = await _preflight_over(
        repository,
        runner=runner,
        workspace_preserved=True,
        previous_lockfile_digest=first.dependency_lockfile_digest,
    )

    assert second.dependency_install_status == "installed"
    assert second.dependency_lockfile_digest != first.dependency_lockfile_digest
    assert _INSTALL_COMMAND in [command for command, _timeout in runner.timeouts]


@pytest.mark.asyncio
async def test_the_install_skip_fails_open_in_every_direction(tmp_path: Path) -> None:
    """A fresh checkout and a missing previous digest both install.

    The cost of a needless install is a few minutes; the cost of a skipped necessary one is
    an attempt that fails on a missing module, which is the confusion this platform exists
    to avoid. So every unanswerable case installs.
    """
    repository = _node_repository(tmp_path, "admanager")
    digest = (
        await _preflight_over(
            repository,
            runner=_TimeoutRecordingRunner(),
            workspace_preserved=False,
            previous_lockfile_digest=None,
        )
    ).dependency_lockfile_digest

    # A fresh or reset checkout: the tree holds nothing, whatever the digest says.
    fresh_runner = _TimeoutRecordingRunner()
    fresh = await _preflight_over(
        repository,
        runner=fresh_runner,
        workspace_preserved=False,
        previous_lockfile_digest=digest,
    )
    # Preserved, but the previous attempt recorded no digest at all.
    unknown_runner = _TimeoutRecordingRunner()
    unknown = await _preflight_over(
        repository,
        runner=unknown_runner,
        workspace_preserved=True,
        previous_lockfile_digest=None,
    )

    assert fresh.dependency_install_status == "installed"
    assert unknown.dependency_install_status == "installed"
    for runner in (fresh_runner, unknown_runner):
        assert _INSTALL_COMMAND in [command for command, _timeout in runner.timeouts]


@pytest.mark.asyncio
async def test_a_checkout_with_no_installer_answers_no_digest(tmp_path: Path) -> None:
    """No installer is nothing to skip, and `None` is what "cannot be answered" means."""
    repository = tmp_path / "docs-only"
    repository.mkdir()
    (repository / "README.md").write_text("# docs\n", encoding="utf-8")

    result = await _preflight_over(
        repository,
        runner=_TimeoutRecordingRunner(),
        workspace_preserved=True,
        previous_lockfile_digest=None,
    )

    assert result.dependency_lockfile_digest is None
    assert result.dependency_install_status == "not_applicable"
