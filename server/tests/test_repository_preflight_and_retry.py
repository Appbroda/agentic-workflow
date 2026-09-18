"""Regression tests for repository preflight, completion checks, and retry isolation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from agents.reviewer.agent import _apply_completion_findings, _apply_validation_findings
from agents.shared.contracts import create_artifact
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import (
    CodeCompletionArtifact,
    FileChange,
    IntegrationReviewArtifact,
    RepositoryConvention,
    RepositoryReconnaissanceArtifact,
    RepositoryWorkstreamPlan,
    RequirementImplementationEvidence,
)
from services.cancellation import MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.feature_runtime import (
    _block_failed_baseline_validation,
    _block_unrepairable_lint_configuration,
    _registry_paths,
    _reviewed_source_fingerprint,
)
from services.process_runner import ProcessResult
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.external_operations import ExternalOperationStatus
from state.feature_models import FeatureWorkflowSnapshot
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from tools import node_toolchain
from tools import repository_preflight as preflight_module
from tools.implementation_completeness import (
    meaningful_progress,
    validate_implementation_completeness,
)
from tools.repository_preflight import (
    RepositoryHealthDisposition,
    RepositoryPreflight,
    RepositoryPreflightResult,
    _node_version_satisfies,
    classify_lint_failure,
    install_failure_is_transient,
)
from tools.repository_reconnaissance import registers_modules
from tools.retry_strategy import (
    FailureClassification,
    TerminalCause,
    TerminalTriage,
    TreeResubmission,
    a_required_command_rejected_the_source,
    build_retry_plan,
    classify_failure,
    decide_child_retry,
    diagnostic_signature,
    retry_counter_field,
    triage_stopped_workstream,
)
from tools.validation_tools import (
    VALIDATION_RESOURCE_EXHAUSTED,
    VALIDATION_TIMED_OUT,
    DefaultValidationPlanBuilder,
    ValidationCommand,
    ValidationResult,
    ValidationStatus,
)
from workflows.feature_workflow import (
    ChildExecution,
    FeatureWorkflowError,
    FeatureWorkflowOrchestrator,
    _child_result,
)


class StubProcessRunner:
    """Mock deterministic installation without network or a real package manager."""

    def __init__(self, *, create_airbnb_config: bool = False) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.working_directories: list[Path] = []
        self._create_airbnb_config = create_airbnb_config

    async def run(self, command: Any, cwd: Path, *_args: Any, **_kwargs: Any) -> ProcessResult:
        self.calls.append(tuple(command))
        self.working_directories.append(cwd)
        if self._create_airbnb_config:
            (cwd / "node_modules" / "eslint-config-airbnb-base").mkdir(parents=True)
        return ProcessResult(
            command=tuple(command),
            return_code=0,
            stdout="installed",
            stderr="",
            duration_seconds=0.01,
        )


class RuntimeVersionProcessRunner(StubProcessRunner):
    """Report concrete worker tool versions while keeping installations deterministic.

    Models Corepack as well as the image's own binaries, because whether a declared version
    can be provisioned is what decides the outcome for a declaring repository. A double that
    answered only `npm --version` would report every such repository as unprovisionable and
    assert the opposite of production.

    `provisionable` is the set of `manager@version` strings Corepack can supply, defaulting to
    "any exactly-declared version", which is what a worker with registry access does. Pass an
    explicit set to model a worker that cannot reach one.
    """

    def __init__(self, versions: dict[str, str], *, provisionable: set[str] | None = None) -> None:
        super().__init__()
        self._versions = versions
        self._provisionable = provisionable

    def _corepack_version(self, command: tuple[str, ...]) -> str | None:
        """Resolve `corepack <manager>@<version> --version` the way Corepack would."""
        if len(command) != 3 or command[0] != "corepack" or command[2] != "--version":
            return None
        requested = command[1]
        if self._provisionable is not None and requested not in self._provisionable:
            return None
        return requested.partition("@")[2] or None

    async def run(self, command: Any, cwd: Path, *_args: Any, **_kwargs: Any) -> ProcessResult:
        executable = str(command[0])
        if executable == "corepack" and tuple(command)[-1:] == ("--version",):
            self.calls.append(tuple(command))
            provisioned = self._corepack_version(tuple(str(item) for item in command))
            return ProcessResult(
                command=tuple(command),
                return_code=0 if provisioned is not None else 1,
                stdout=f"{provisioned}\n" if provisioned is not None else "",
                stderr="" if provisioned is not None else "no such version",
                duration_seconds=0.01,
            )
        if tuple(command) == (executable, "--version"):
            self.calls.append(tuple(command))
            version = self._versions.get(executable)
            return ProcessResult(
                command=tuple(command),
                return_code=0 if version is not None else 127,
                stdout=f"{version}\n" if version is not None else "",
                stderr="",
                duration_seconds=0.01,
            )
        return await super().run(command, cwd, *_args, **_kwargs)


@pytest.fixture
def without_corepack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model a worker image that has no Corepack.

    The lockfile-compatibility judgement below is only reachable here. Where Corepack exists
    the declared version is provisioned and run, so there is no ambient version left to be
    compatible with -- and a test that asserted compatibility without saying this would be
    asserting a path production does not take.
    """
    monkeypatch.setattr(node_toolchain, "corepack_available", lambda: False)


@pytest.mark.parametrize(
    ("actual", "requirement", "expected"),
    [
        ("22.14.0", ">=22 <23", True),
        ("22.14.0", "^20.9.0 || >=22 <23", True),
        ("22.14.0", "20 - 22", True),
        ("22.14.0", "22.x", True),
        ("20.8.0", "^20.9.0", False),
        ("0.2.5", "^0.2.3", True),
        ("0.3.0", "^0.2.3", False),
        ("18.18.9", "~18.18.0", True),
        ("22.14.0", "lts/*", None),
        ("23.0.0-rc.1", ">=23", None),
    ],
)
def test_node_engine_ranges_are_evaluated_conservatively(
    actual: str, requirement: str, expected: bool | None
) -> None:
    """Common npm ranges are honored and unknown syntax is never guessed."""
    assert _node_version_satisfies(actual, requirement) is expected


@pytest.mark.asyncio
async def test_a_yarn_berry_declaration_runs_berry_on_a_yarn_one_image(tmp_path: Path) -> None:
    """A Yarn 4 declaration must never be executed by the worker's global Yarn 1.

    That property is unchanged; what changed is the resolution. It used to be satisfied by
    refusing to run the repository at all, which is a defensible answer for one repository and
    an impossible one for a platform -- the image pins a single Yarn, so every Berry
    repository was permanently unservable. Corepack provisions the declared version instead,
    so Berry runs as Berry and the image's Yarn 1 never touches it.
    """
    (tmp_path / "package.json").write_text(
        '{"name":"web","packageManager":"yarn@4.6.0"}', encoding="utf-8"
    )
    (tmp_path / "yarn.lock").write_text("# yarn lock\n", encoding="utf-8")
    runner = RuntimeVersionProcessRunner({"yarn": "1.22.22"})

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="web")

    assert result.dependency_install_status != "blocked_unsupported_runtime"
    assert not result.blocking_issues
    assert result.dependency_install_commands == [
        ["corepack", "yarn@4.6.0", "install", "--frozen-lockfile"]
    ]
    # Both recorded: what the image carries, and what this repository actually ran on.
    assert result.package_manager_versions == {"yarn": "1.22.22", "yarn (declared)": "4.6.0"}


@pytest.mark.asyncio
async def test_a_declared_version_corepack_cannot_supply_is_reported_plainly(
    tmp_path: Path,
) -> None:
    """A version that does not exist, or a worker that cannot reach the registry.

    Blocking is right here -- the commands are built to run through Corepack, so an
    unprovisionable version would otherwise fail later with no diagnosis -- but it must say
    which version could not be supplied rather than blaming the image's.
    """
    (tmp_path / "package.json").write_text(
        '{"name":"web","packageManager":"npm@99.99.99"}', encoding="utf-8"
    )
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")
    runner = RuntimeVersionProcessRunner({"npm": "10.9.8"}, provisionable=set())

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="web")

    issue = next(
        item
        for item in result.blocking_issues
        if item.issue_id == "PACKAGE_MANAGER_PROVISION_FAILED"
    )
    assert "npm@99.99.99" in issue.evidence
    assert "Corepack could not provision" in issue.evidence


@pytest.mark.asyncio
async def test_the_repository_that_killed_222_runs_the_version_it_declared(
    tmp_path: Path,
) -> None:
    """AB-Feature-222, in the shape the live worker met it, now working.

    `npm@9.8.1` declared, npm 10.9.8 in the image. The feature was stopped before a line of
    code was written, after 537 seconds of planning, and classified as an unrepairable
    repository defect -- for a repository whose `npm ci` adds 1100 packages and exits 0.

    The declaration is a Corepack provisioning directive, so it is honoured rather than
    argued with: 9.8.1 is provisioned and 9.8.1 is what installs and what runs every check.
    Nothing is left to be incompatible, which is why no warning is expected either.
    """
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "name": "admin",
                "packageManager": "npm@9.8.1",
                "engines": {"node": ">=20.0.0"},
                "scripts": {"lint": "eslint .", "test": "jest", "build": "next build"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text(
        json.dumps({"name": "admin", "lockfileVersion": 3, "packages": {}}), encoding="utf-8"
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "page.tsx").write_text("export default () => null\n", encoding="utf-8")
    runner = RuntimeVersionProcessRunner({"node": "22.23.2", "npm": "10.9.8"})

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="admin")

    # `blocked` is what every terminal repository-setup gate reads, and the install status was
    # a second independent gate on the same fact.
    assert result.validation_readiness == "ready"
    assert result.dependency_install_status == "installed"
    assert not result.blocking_issues
    assert result.dependency_install_commands == [["corepack", "npm@9.8.1", "ci", "--no-audit"]]
    assert result.package_manager_versions == {"npm": "10.9.8", "npm (declared)": "9.8.1"}


@pytest.mark.asyncio
async def test_without_corepack_a_readable_lockfile_is_still_not_a_defect(
    tmp_path: Path, without_corepack: None
) -> None:
    """The fallback for a worker that cannot provision anything.

    Here the image's npm is what will run, so the question is the narrower one of whether it
    can read the committed lockfile -- `lockfileVersion` 3, which every npm from 7 onwards
    reads. Reported, because the run is genuinely not using the pinned version, but not a
    reason to refuse a repository that installs.
    """
    (tmp_path / "package.json").write_text(
        json.dumps({"name": "admin", "packageManager": "npm@9.8.1", "scripts": {"lint": "x"}}),
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text(
        json.dumps({"name": "admin", "lockfileVersion": 3}), encoding="utf-8"
    )
    runner = RuntimeVersionProcessRunner({"node": "22.23.2", "npm": "10.9.8"})

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="admin")

    assert result.dependency_install_status == "installed"
    assert result.dependency_install_commands == [["npm", "ci", "--no-audit"]]
    warning = next(
        item for item in result.warnings if item.issue_id == "PACKAGE_MANAGER_VERSION_DIFFERS"
    )
    assert warning.severity == "low"
    assert "npm@9.8.1" in warning.evidence
    assert "lockfileVersion 3" in warning.evidence


@pytest.mark.asyncio
async def test_an_npm_too_old_for_the_committed_lockfile_still_blocks(
    tmp_path: Path, without_corepack: None
) -> None:
    """The limit on the above, and why the lockfile is read rather than assumed.

    npm 6 cannot read `lockfileVersion` 3. Downgrading every major-version difference to a
    warning would have let this install run and produce a tree the lockfile never described.
    """
    (tmp_path / "package.json").write_text(
        json.dumps({"name": "legacy", "packageManager": "npm@9.8.1"}), encoding="utf-8"
    )
    (tmp_path / "package-lock.json").write_text(
        json.dumps({"name": "legacy", "lockfileVersion": 3}), encoding="utf-8"
    )
    runner = RuntimeVersionProcessRunner({"node": "22.23.2", "npm": "6.14.18"})

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="legacy")

    assert result.dependency_install_status == "blocked_unsupported_runtime"
    issue = next(
        item
        for item in result.blocking_issues
        if item.issue_id == "PACKAGE_MANAGER_VERSION_UNSUPPORTED"
    )
    assert "lockfileVersion 3" in issue.evidence


@pytest.mark.asyncio
async def test_an_unreadable_lockfile_is_not_evidence_of_compatibility(
    tmp_path: Path, without_corepack: None
) -> None:
    """A lockfile the platform cannot parse proves nothing, so the major jump keeps blocking.

    The failure mode being avoided is a helper that returns "compatible" for every input it
    cannot understand, which would silently turn this whole check off.
    """
    (tmp_path / "package.json").write_text(
        json.dumps({"name": "web", "packageManager": "npm@9.8.1"}), encoding="utf-8"
    )
    (tmp_path / "package-lock.json").write_text("{ not json at all", encoding="utf-8")
    runner = RuntimeVersionProcessRunner({"node": "22.23.2", "npm": "10.9.8"})

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="web")

    assert [
        item
        for item in result.blocking_issues
        if item.issue_id == "PACKAGE_MANAGER_VERSION_UNSUPPORTED"
    ]


@pytest.mark.asyncio
async def test_a_patch_difference_never_blocks_any_manager(tmp_path: Path) -> None:
    """Same major, no lockfile reasoning needed: semver says the format did not change.

    Uses pnpm, whose lockfile this code deliberately does not parse, to show the same-major
    rule stands on its own rather than on npm's self-description.
    """
    (tmp_path / "package.json").write_text(
        json.dumps({"name": "web", "packageManager": "pnpm@9.1.0"}), encoding="utf-8"
    )
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")
    runner = RuntimeVersionProcessRunner({"node": "22.23.2", "pnpm": "9.15.4"})

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="web")

    assert result.dependency_install_status != "blocked_unsupported_runtime"
    assert not [
        item
        for item in result.blocking_issues
        if item.issue_id == "PACKAGE_MANAGER_VERSION_UNSUPPORTED"
    ]


@pytest.mark.asyncio
async def test_the_install_and_the_checks_both_run_the_declared_version(tmp_path: Path) -> None:
    """The declared version reaches the commands, not just a compatibility verdict.

    Both halves matter and they are separate code paths. Installing with the declared version
    and then running `npm run build` with the image's would be a difference nobody asked for
    and nobody would notice until it mattered.
    """
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "name": "admin",
                "packageManager": "npm@9.8.1",
                "scripts": {"lint": "eslint .", "test": "jest", "build": "next build"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text(
        json.dumps({"name": "admin", "lockfileVersion": 3}), encoding="utf-8"
    )
    runner = RuntimeVersionProcessRunner({"node": "22.23.2", "npm": "10.9.8"})

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="admin")
    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)

    assert result.dependency_install_commands == [["corepack", "npm@9.8.1", "ci", "--no-audit"]]
    assert [command.command for command in plan.commands] == [
        ["corepack", "npm@9.8.1", "run", "lint"],
        ["corepack", "npm@9.8.1", "run", "test"],
        ["corepack", "npm@9.8.1", "run", "build"],
    ]
    # The image's npm is nowhere in what will be executed.
    assert not [command for command in plan.commands if command.command[:1] == ["npm"]], (
        "a check would have run on the image's npm"
    )


@pytest.mark.asyncio
async def test_a_repository_declaring_nothing_still_runs_the_image_toolchain(
    tmp_path: Path,
) -> None:
    """Every repository the platform already builds declares nothing; none of them change."""
    (tmp_path / "package.json").write_text(
        json.dumps({"name": "web", "scripts": {"lint": "eslint ."}}), encoding="utf-8"
    )
    (tmp_path / "package-lock.json").write_text(
        json.dumps({"name": "web", "lockfileVersion": 3}), encoding="utf-8"
    )
    runner = RuntimeVersionProcessRunner({"node": "22.23.2", "npm": "10.9.8"})

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="web")
    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)

    assert result.dependency_install_commands == [["npm", "ci", "--no-audit"]]
    assert [command.command for command in plan.commands] == [["npm", "run", "lint"]]


class TimingOutVersionProcessRunner(StubProcessRunner):
    """Time out the version probe a fixed number of times, then answer normally."""

    def __init__(self, *, timeouts: int) -> None:
        super().__init__()
        self._remaining = timeouts

    async def run(self, command: Any, cwd: Path, *args: Any, **kwargs: Any) -> ProcessResult:
        if tuple(command) == ("node", "--version"):
            self.calls.append(tuple(command))
            if self._remaining > 0:
                self._remaining -= 1
                return ProcessResult(
                    command=tuple(command),
                    return_code=None,
                    stdout="",
                    stderr="",
                    duration_seconds=30.0,
                    timed_out=True,
                )
            return ProcessResult(
                command=tuple(command),
                return_code=0,
                stdout="v22.14.0\n",
                stderr="",
                duration_seconds=0.01,
            )
        return await super().run(command, cwd, *args, **kwargs)


@pytest.mark.asyncio
async def test_a_version_probe_that_times_out_is_asked_again(tmp_path: Path) -> None:
    """A loaded worker failing to answer says nothing about the repository's setup.

    run-2's backend was recorded as having an invalid checked-in setup -- terminal, so no
    coding attempt was ever made -- because `node --version` did not answer in time while
    another feature saturated the machine.
    """
    (tmp_path / "package.json").write_text(
        '{"name":"web","engines":{"node":">=22 <23"}}', encoding="utf-8"
    )
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")
    runner = TimingOutVersionProcessRunner(timeouts=2)

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="web")

    assert result.node_runtime_version == "22.14.0"
    assert not any(
        item.issue_id == "RUNTIME_VERSION_UNAVAILABLE" for item in result.blocking_issues
    )


@pytest.mark.asyncio
async def test_a_missing_runtime_binary_still_fails_on_its_first_answer(tmp_path: Path) -> None:
    """Retrying is only for a probe that did not answer; a real answer is trusted at once."""
    (tmp_path / "package.json").write_text(
        '{"name":"web","engines":{"node":">=22 <23"}}', encoding="utf-8"
    )
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")
    runner = RuntimeVersionProcessRunner({})

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="web")

    assert runner.calls.count(("node", "--version")) == 1
    assert any(item.issue_id == "RUNTIME_VERSION_UNAVAILABLE" for item in result.blocking_issues)


@pytest.mark.asyncio
async def test_matching_node_engine_and_pnpm_version_are_persisted_and_installed(
    tmp_path: Path,
) -> None:
    """Successful preflight records the exact toolchain evidence used for execution."""
    (tmp_path / "package.json").write_text(
        ('{"name":"web","packageManager":"pnpm@9.15.4","engines":{"node":"^20.9.0 || >=22 <23"}}'),
        encoding="utf-8",
    )
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")
    runner = RuntimeVersionProcessRunner({"node": "v22.14.0", "pnpm": "9.15.4"})

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="web")

    # The declared version is provisioned and used even though the image happens to carry the
    # same one. Uniform on purpose: which version runs is then a property of the repository
    # rather than of whichever worker picked the job up.
    assert runner.calls == [
        ("node", "--version"),
        ("pnpm", "--version"),
        ("corepack", "pnpm@9.15.4", "--version"),
        ("corepack", "pnpm@9.15.4", "install", "--frozen-lockfile"),
    ]
    assert result.validation_readiness == "ready_with_warnings"
    assert result.dependency_install_status == "installed"
    assert result.node_runtime_version == "22.14.0"
    assert result.package_manager_versions == {"pnpm": "9.15.4", "pnpm (declared)": "9.15.4"}


@pytest.mark.asyncio
async def test_incompatible_node_engine_blocks_before_dependency_install(tmp_path: Path) -> None:
    """A checked-in Node engine constraint is evaluated before any lifecycle scripts run."""
    (tmp_path / "package.json").write_text(
        '{"name":"legacy","engines":{"node":">=18 <20"}}', encoding="utf-8"
    )
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")
    runner = RuntimeVersionProcessRunner({"node": "v22.14.0"})

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="legacy")

    assert runner.calls == [("node", "--version")]
    assert result.dependency_install_status == "blocked_unsupported_runtime"
    issue = next(
        item
        for item in result.blocking_issues
        if item.issue_id == "NODE_RUNTIME_VERSION_UNSUPPORTED"
    )
    assert ">=18 <20" in issue.evidence
    assert "Node.js 22.14.0" in issue.evidence


@pytest.mark.asyncio
async def test_package_manager_declaration_without_lockfile_reports_both_guards(
    tmp_path: Path,
) -> None:
    """An exact manager declaration is not a substitute for a reproducible lockfile."""
    (tmp_path / "package.json").write_text(
        '{"name":"web","packageManager":"pnpm@9.15.4"}', encoding="utf-8"
    )
    runner = RuntimeVersionProcessRunner({"pnpm": "9.15.4"})

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="web")

    assert runner.calls == [("pnpm", "--version"), ("corepack", "pnpm@9.15.4", "--version")]
    assert result.dependency_install_status == "blocked_missing_lockfile"
    assert {item.issue_id for item in result.blocking_issues} == {"NODE_LOCKFILE_NOT_FOUND"}
    assert result.package_manager_versions == {
        "pnpm": "9.15.4",
        "pnpm (declared)": "9.15.4",
    }


@pytest.mark.asyncio
async def test_declared_manager_must_match_the_committed_lockfile(tmp_path: Path) -> None:
    """Manager selection cannot silently prefer a conflicting lockfile family."""
    (tmp_path / "package.json").write_text(
        '{"name":"web","packageManager":"yarn@1.22.22"}', encoding="utf-8"
    )
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")
    runner = RuntimeVersionProcessRunner({"yarn": "1.22.22"})

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="web")

    assert runner.calls == [("yarn", "--version"), ("corepack", "yarn@1.22.22", "--version")]
    assert result.dependency_install_status == "blocked_unsupported_runtime"
    mismatch = next(
        item
        for item in result.blocking_issues
        if item.issue_id == "PACKAGE_MANAGER_LOCKFILE_MISMATCH"
    )
    assert "selects `yarn`" in mismatch.evidence
    assert "selects `pnpm`" in mismatch.evidence


def test_failed_required_baseline_validation_blocks_before_coding() -> None:
    """An already-red checkout is diagnosed as repository health, not relabeled optional."""
    preflight = RepositoryPreflightResult(
        repository_id="backend",
        revision="revision-a",
        dependency_install_status="installed",
        validation_readiness="ready",
    )

    blocked = _block_failed_baseline_validation(
        preflight,
        ['{"command":["npm","test"],"validation_type":"test"}'],
    )

    assert blocked.validation_readiness == "blocked"
    assert blocked.baseline_validation_failures == [
        '{"command":["npm","test"],"validation_type":"test"}'
    ]
    assert blocked.blocking_issues[0].category == "baseline_validation"
    assert blocked.blocking_issues[0].automatically_repairable is False


@pytest.mark.asyncio
async def test_lint_configuration_is_not_a_language_specific_preflight_block(
    tmp_path: Path,
) -> None:
    """Manifest and lock evidence, not a named lint plugin, controls preflight readiness."""
    (tmp_path / "package.json").write_text(
        '{"scripts":{"lint":"eslint ."},"eslintConfig":{"extends":"airbnb-base"}}',
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")

    result = await RepositoryPreflight(process_runner=StubProcessRunner()).run(
        tmp_path, repository_id="backend"
    )
    lint = classify_lint_failure(
        result_stdout="",
        result_stderr=(
            "Failed to load config airbnb-base: Cannot find module eslint-config-airbnb-base"
        ),
        repository_root=tmp_path,
    )

    assert result.validation_readiness == "ready_with_warnings"
    assert result.blocking_issues == []
    assert lint.kind == "missing_dependency"
    # Undeclared: installing it would be the platform inventing a dependency change.
    assert lint.disposition is RepositoryHealthDisposition.REQUIRES_HUMAN
    assert "airbnb-base" in lint.undeclared_references


@pytest.mark.asyncio
async def test_declared_eslint_dependency_is_restored_by_deterministic_install(
    tmp_path: Path,
) -> None:
    """A declared, locked shared lint config is repaired only by npm ci before validation."""
    (tmp_path / "package.json").write_text(
        (
            '{"scripts":{"lint":"eslint ."},"eslintConfig":{"extends":"airbnb-base"},'
            '"devDependencies":{"eslint-config-airbnb-base":"1.0.0"}}'
        ),
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text(
        '{"lockfileVersion":3,"packages":{"node_modules/eslint-config-airbnb-base":{}}}',
        encoding="utf-8",
    )
    runner = StubProcessRunner(create_airbnb_config=True)

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="backend")
    lint = classify_lint_failure(
        result_stdout="",
        result_stderr=(
            "Failed to load config airbnb-base: Cannot find module eslint-config-airbnb-base"
        ),
        repository_root=tmp_path,
    )

    assert runner.calls == [("npm", "ci", "--no-audit")]
    assert result.dependency_install_status == "installed"
    assert result.validation_readiness != "blocked"
    # Declared in the manifest, so a deterministic reinstall is a safe bounded repair.
    assert lint.disposition is RepositoryHealthDisposition.AUTO_REPAIRABLE
    assert lint.undeclared_references == []


@pytest.mark.asyncio
async def test_the_npm_install_does_not_call_the_advisory_service(tmp_path: Path) -> None:
    """`npm ci` runs with `--no-audit`, so no install waits on npm's advisory endpoints.

    Measured on 2026-09-03, warm cache, idle machine, from npm's own timers:

        npm timing reify:build            Completed in     267ms
        npm timing auditReport:getReport  Completed in  421789ms

    The advisory POST had begun accepting connections and never answering, so every install
    spent minutes on a dead socket and then died on its budget -- AB-Feature-203, 204 and 205,
    all on dependency installation and nothing else. `--no-audit` put the same checkout back at
    20s. The flag is asserted on the spawned command rather than on a duration because the
    duration is the sandbox's to measure and the command is this platform's to choose.

    Only npm needs it: `pnpm` and `yarn` do not audit on install, and pinning all three here
    is what would notice if that stopped being true.
    """
    (tmp_path / "package.json").write_text('{"name":"web"}', encoding="utf-8")
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")
    runner = StubProcessRunner()

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="web")

    assert runner.calls == [("npm", "ci", "--no-audit")]
    assert result.dependency_install_command == ["npm", "ci", "--no-audit"]
    assert result.dependency_install_status == "installed"


@pytest.mark.asyncio
async def test_failed_journaled_install_retries_instead_of_replaying_false_success(
    tmp_path: Path,
) -> None:
    """A nonzero install is retryable journal evidence, never a cached successful result.

    The first failure is deliberately a checked-in lockfile problem rather than a registry
    one. A registry blip is answered inside the operation now (see the transient-retry test
    below), which would settle this install on its first `run` and leave nothing here to
    assert about the journal.
    """

    class FailOnceProcessRunner:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, command: Any, *_args: Any, **_kwargs: Any) -> ProcessResult:
            self.calls += 1
            return ProcessResult(
                command=tuple(command),
                return_code=1 if self.calls == 1 else 0,
                stdout="",
                stderr=(
                    "npm ERR! `npm ci` can only install with an existing package-lock.json"
                    if self.calls == 1
                    else ""
                ),
                duration_seconds=0.01,
            )

    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "package.json").write_text('{"name":"journal-retry"}', encoding="utf-8")
    (repository / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'install-journal.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    runner = FailOnceProcessRunner()
    preflight = RepositoryPreflight(
        process_runner=runner,
        cancellation_token=MockCancellationToken(),
        operation_executor=ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(
                workflow_id="workflow-install-retry", repository_id="backend"
            ),
        ),
    )

    first = await preflight.run(repository, repository_id="backend")
    operation = (await journal.list_operations_for_workflow("workflow-install-retry"))[0]
    assert first.dependency_install_status == "failed"
    assert operation.status is ExternalOperationStatus.FAILED_RETRYABLE
    assert operation.result_payload is None

    second = await preflight.run(repository, repository_id="backend")
    completed = await journal.get(operation.operation_id)
    assert second.dependency_install_status == "installed"
    assert runner.calls == 2
    assert completed.status is ExternalOperationStatus.SUCCEEDED
    assert completed.attempt == 2
    await database.dispose()


class _InstallFlakeRunner:
    """Fail the install with the wording a package manager uses, then behave."""

    def __init__(self, *, failures: int, stderr: str) -> None:
        self.calls = 0
        self._failures = failures
        self._stderr = stderr

    async def run(self, command: Any, *_args: Any, **_kwargs: Any) -> ProcessResult:
        self.calls += 1
        failing = self.calls <= self._failures
        return ProcessResult(
            command=tuple(command),
            return_code=1 if failing else 0,
            stdout="",
            stderr=self._stderr if failing else "",
            duration_seconds=0.01,
        )


def _node_repository(tmp_path: Path, name: str) -> Path:
    """One checkout whose only preflight work is a deterministic npm install."""
    repository = tmp_path / name
    repository.mkdir()
    (repository / "package.json").write_text(f'{{"name":"{name}"}}', encoding="utf-8")
    (repository / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")
    return repository


@pytest.mark.asyncio
async def test_a_registry_blip_is_asked_again_instead_of_costing_a_whole_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The install flake that killed 198's frontend is answered by asking the install again.

    Three `dependency_installation_failure`s in one batch, and one of them ended a workstream:
    the flake was charged to the repository-setup budget, which bought a fresh coding call
    against source nothing had changed, and the no-meaningful-change refusal then stopped it.
    The install is what should have been repeated, and repeating it costs no model spend --
    which is why it may wait far longer than a provider fault is allowed to.
    """
    monkeypatch.setattr(preflight_module, "_INSTALL_FAULT_BACKOFF_SECONDS", 0.0)
    runner = _InstallFlakeRunner(failures=2, stderr="npm ERR! network socket hang up")

    result = await RepositoryPreflight(
        process_runner=runner, cancellation_token=MockCancellationToken()
    ).run(_node_repository(tmp_path, "flaky"), repository_id="frontend")

    assert result.dependency_install_status == "installed"
    assert result.validation_readiness != "blocked"
    # Two absorbed failures and the install that finally worked, inside one attempt.
    assert runner.calls == 3


@pytest.mark.asyncio
async def test_a_lockfile_failure_is_not_asked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checked-in defect returns the same answer at any interval, so it is charged at once.

    The house rule from 54- Part 3: detect the transient shape conservatively and, when the
    evidence does not say one, keep today's behaviour. Two more identical installs would only
    delay the operator question this repository actually needs.
    """
    monkeypatch.setattr(preflight_module, "_INSTALL_FAULT_BACKOFF_SECONDS", 0.0)
    runner = _InstallFlakeRunner(
        failures=99, stderr="npm ERR! Missing: left-pad@1.3.0 from lock file"
    )

    result = await RepositoryPreflight(
        process_runner=runner, cancellation_token=MockCancellationToken()
    ).run(_node_repository(tmp_path, "broken-lock"), repository_id="frontend")

    assert result.dependency_install_status == "failed"
    assert runner.calls == 1


@pytest.mark.asyncio
async def test_a_registry_that_stays_down_is_charged_to_the_setup_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extra installs are bounded, so a dead registry still reaches the retry policy."""
    monkeypatch.setattr(preflight_module, "_INSTALL_FAULT_BACKOFF_SECONDS", 0.0)
    runner = _InstallFlakeRunner(failures=99, stderr="npm ERR! request to registry ETIMEDOUT")

    result = await RepositoryPreflight(
        process_runner=runner, cancellation_token=MockCancellationToken()
    ).run(_node_repository(tmp_path, "registry-down"), repository_id="frontend")

    assert result.dependency_install_status == "failed"
    assert runner.calls == 1 + preflight_module._ALLOWED_INSTALL_FAULTS


def test_an_install_that_ran_out_of_time_is_not_a_registry_blip() -> None:
    """The timeout is the sandbox's limit, and an install that does not fit will not start to.

    Deliberately excluded rather than forgotten: retrying spends the whole limit again to
    reach the same verdict, and the remedy is a larger limit or a smaller install.
    """
    timed_out = ProcessResult(
        command=("npm", "ci"),
        return_code=None,
        stdout="",
        stderr="npm ERR! network socket hang up",
        duration_seconds=120.0,
        timed_out=True,
    )
    assert not install_failure_is_transient(timed_out)


def test_multiple_failing_npm_commands_produce_distinct_finding_identifiers() -> None:
    """Every Node command shares one executable, so ids must not collide.

    Keying a finding on the executable alone made two failing npm scripts emit the same
    id, and the review artifact's uniqueness rule then rejected the whole review -- a
    crash for any repository with more than one npm script.
    """
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}
    failures = [
        _npm_validation_failure("lint", "eslint reported errors"),
        _npm_validation_failure("test", "one test failed"),
        _npm_validation_failure("build", "the bundle failed"),
    ]

    _apply_validation_findings(payload, tuple(failures), test_validation_required=False)

    identifiers = [item["finding_id"] for item in payload["findings"]]
    assert len(identifiers) == len(set(identifiers)) == 3
    assert payload["verdict"] == "changes_requested"


def test_a_review_rejection_with_no_failing_command_is_not_sent_to_fix_source() -> None:
    """AB-Feature-225 was told to inspect diagnostics that did not exist, three times.

    `validation_source_failure` is what a reviewer's own findings are called whenever one of
    them reports a failing command, so it arrives for a review rejection exactly as it does
    for a command that exited non-zero. The advice is written for the second. Given to the
    first, it sent three consecutive attempts looking for a failing diagnostic while format,
    lint and build were green -- and the only failing thing within reach was the checkout's own
    validation configuration, which is what all three of them rewrote.
    """
    plan = build_retry_plan(
        FailureClassification.VALIDATION_SOURCE_FAILURE,
        expected_source_areas=["src/app"],
        previous_findings=["The configured browser suite has no passing result."],
        current_revision="rev-1",
        source_verdict_available=False,
    )

    assert "no failing source diagnostic" in plan.required_strategy_change
    assert "leave the scripts alone" in plan.required_strategy_change
    assert "package" in plan.previous_approach_to_avoid
    assert "Inspect the exact failing source diagnostics" not in plan.required_strategy_change


def test_a_failing_command_still_sends_the_attempt_to_the_source() -> None:
    """The classification's other shape keeps the advice it was written for."""
    plan = build_retry_plan(
        FailureClassification.VALIDATION_SOURCE_FAILURE,
        expected_source_areas=["src/app"],
        previous_findings=["npm run test exited 1"],
        current_revision="rev-1",
        source_verdict_available=True,
    )

    assert "Inspect the exact failing source diagnostics" in plan.required_strategy_change


def test_only_a_required_command_counts_as_a_verdict_on_the_source() -> None:
    """An advisory command's rejection is not the source being judged."""
    assert not a_required_command_rejected_the_source(
        [
            {"status": "passed", "required": True},
            {"status": "no_tests_found", "required": True},
            {"status": "failed", "required": False},
        ]
    )
    assert a_required_command_rejected_the_source([{"status": "failed", "required": True}])


_UNIT_TEST_COMMAND = ValidationCommand(
    validation_type="test",
    command=["npm", "run", "test"],
    required=True,
    timeout_seconds=900.0,
    accepts_test_paths=True,
    collectible_suffixes=[".js", ".jsx", ".ts", ".tsx"],
)


def _empty_suite_result() -> ValidationResult:
    """The required test command reporting that it collected nothing."""
    return ValidationResult(
        command=("npm", "run", "test"),
        return_code=0,
        stdout="No tests found, exiting with code 0\n",
        stderr="",
        timed_out=False,
        duration_seconds=1.9,
        validation_type="test",
    )


def test_an_empty_suite_still_blocks_an_attempt_that_wrote_no_tests() -> None:
    """The gate's own case is unchanged: asked for tests, wrote none, has none."""
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}

    _apply_validation_findings(
        payload,
        (_empty_suite_result(),),
        test_validation_required=True,
        changed_test_paths=(),
        planned_commands=(_UNIT_TEST_COMMAND,),
    )

    assert payload["verdict"] == "changes_requested"
    assert [item["severity"] for item in payload["findings"]] == ["high"]


def test_an_empty_suite_still_blocks_when_the_runner_could_have_seen_the_tests() -> None:
    """A collectable suite that came back empty is the change's problem, not the plan's."""
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}

    _apply_validation_findings(
        payload,
        (_empty_suite_result(),),
        test_validation_required=True,
        changed_test_paths=("src/components/sign-in.test.tsx",),
        planned_commands=(_UNIT_TEST_COMMAND,),
    )

    assert payload["verdict"] == "changes_requested"
    assert [item["severity"] for item in payload["findings"]] == ["high"]


def test_an_empty_suite_does_not_block_tests_the_planned_runner_cannot_collect() -> None:
    """AB-Feature-225: the demand the attempt wrote tests for and could never satisfy.

    The workstream added browser suites into the very area its plan named. The one planned
    test command was the repository's unit runner, whose configuration excludes that area, so
    `no_tests_found` was the only answer it could return for any source whatsoever. The old
    rule forced `changes_requested` at `high` on that basis every round -- and the only action
    that would clear it was adding a unit test the feature had no use for, which the same
    review was separately telling the attempt not to do. Four attempts, nine of ten
    requirements passing, and no attempt count that reaches approval.
    """
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}

    _apply_validation_findings(
        payload,
        (_empty_suite_result(),),
        test_validation_required=True,
        changed_test_paths=("e2e/sign-in-theme.spec.ts",),
        planned_commands=(
            ValidationCommand(
                validation_type="test",
                command=["npm", "run", "test"],
                required=True,
                timeout_seconds=900.0,
                accepts_test_paths=True,
                # A runner that collects Python only, standing in for any runner whose own
                # configuration excludes what the attempt wrote.
                collectible_suffixes=[".py"],
            ),
        ),
    )

    assert payload["verdict"] == "approved"
    finding = payload["findings"][0]
    assert finding["severity"] == "low"
    assert "cannot collect" in finding["title"]
    assert "e2e/sign-in-theme.spec.ts" in finding["description"]
    # And it says whose defect it is, because the next attempt is handed this text.
    assert "defect in the assignment" in finding["recommendation"]


def test_the_narrowed_run_answers_what_the_suffix_table_only_guesses() -> None:
    """A runner can parse a file and still be configured not to collect it.

    The suffix table reads a runner's language, so a browser spec and a unit test in the same
    language are indistinguishable to it -- and AB-Feature-225's unit runner was configured to
    ignore the very directory its plan told the attempt to write into. The narrowed run put
    that question to the runner itself and got `no_tests_found`; the platform recorded the
    answer as a finding and drew no conclusion from it. It is the conclusion.
    """
    narrowed = ValidationResult(
        command=("npm", "run", "test", "--", "e2e/sign-in-theme.spec.ts"),
        return_code=0,
        stdout="No tests found, exiting with code 0\n",
        stderr="",
        timed_out=False,
        duration_seconds=1.2,
        validation_type="test",
        required=False,
    )
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}

    _apply_validation_findings(
        payload,
        (_empty_suite_result(), narrowed),
        test_validation_required=True,
        changed_test_paths=("e2e/sign-in-theme.spec.ts",),
        # The suffix table says this runner collects `.ts`, so the guess alone would block.
        planned_commands=(_UNIT_TEST_COMMAND,),
    )

    assert payload["verdict"] == "approved"
    assert any("cannot collect" in item["title"] for item in payload["findings"])


def test_a_required_empty_suite_is_not_evidence_about_particular_paths() -> None:
    """Only the narrowed run was handed these files; the full run speaks for the repository."""
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}

    _apply_validation_findings(
        payload,
        (_empty_suite_result(),),
        test_validation_required=True,
        changed_test_paths=("src/sign_in.test.ts",),
        planned_commands=(_UNIT_TEST_COMMAND,),
    )

    assert payload["verdict"] == "changes_requested"


def test_a_repository_with_no_test_command_at_all_always_blocks() -> None:
    """`not_configured` is the neighbouring status and the collectability narrowing must not
    reach it.

    It says this repository has no test command whatsoever, so nothing about the change was
    checked and the question "could the planned runner have collected these files" has no
    meaning -- there is no runner to ask. Narrowing it as though it were an empty suite let an
    approving review publish a change against a checkout that cannot validate anything, which
    the real-repository tier caught.
    """
    not_configured = ValidationResult(
        command=("npm", "run", "test"),
        return_code=None,
        stdout="",
        stderr="",
        timed_out=False,
        duration_seconds=0.0,
        validation_type="test",
        status=ValidationStatus.NOT_CONFIGURED,
        result_code="TEST_COMMAND_NOT_CONFIGURED",
    )
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}

    _apply_validation_findings(
        payload,
        (not_configured,),
        test_validation_required=True,
        # The attempt wrote tests, and no planned command could collect them -- the exact
        # combination that suppresses the gate for `no_tests_found`.
        changed_test_paths=("e2e/sign-in-theme.spec.ts",),
        planned_commands=(),
    )

    assert payload["verdict"] == "changes_requested"
    assert [item["severity"] for item in payload["findings"]] == ["high"]


def test_an_unrecognised_runner_keeps_the_empty_suite_gate() -> None:
    """Unknown collectible suffixes must not retire the gate by silence."""
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}

    _apply_validation_findings(
        payload,
        (_empty_suite_result(),),
        test_validation_required=True,
        changed_test_paths=("e2e/sign-in-theme.spec.ts",),
        planned_commands=(
            ValidationCommand(
                validation_type="test",
                command=["make", "test"],
                required=True,
                timeout_seconds=900.0,
                collectible_suffixes=[],
            ),
        ),
    )

    assert payload["verdict"] == "changes_requested"


def test_an_advisory_test_command_cannot_hold_a_change_it_cannot_run() -> None:
    """Only a required command carries a verdict, so only it can support this block."""
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}

    _apply_validation_findings(
        payload,
        (_empty_suite_result(),),
        test_validation_required=True,
        changed_test_paths=("e2e/sign-in-theme.spec.ts",),
        planned_commands=(
            ValidationCommand(
                validation_type="test",
                command=["npm", "run", "test"],
                required=False,
                timeout_seconds=900.0,
                collectible_suffixes=[],
            ),
        ),
    )

    assert payload["verdict"] == "approved"


def _npm_validation_failure(validation_type: str, detail: str) -> ValidationResult:
    """Build one failed npm validation result for the given validation category."""
    return ValidationResult(
        command=("npm", "run", validation_type),
        return_code=1,
        stdout="",
        stderr=detail,
        timed_out=False,
        duration_seconds=0.1,
        validation_type=validation_type,  # type: ignore[arg-type]
        repository_id="backend",
        status=ValidationStatus.FAILED,
        required=True,
    )


def test_retry_is_refused_outright_when_no_relevant_input_changed() -> None:
    """The single retry authority enforces the loop-prevention invariant directly."""
    unchanged = decide_child_retry(
        classification=FailureClassification.IMPLEMENTATION_MISSING,
        attempt_count=0,
        budget=4,
        retry_count=1,
        max_child_review_cycles=5,
        meaningful_change=False,
    )
    progressed = decide_child_retry(
        classification=FailureClassification.IMPLEMENTATION_MISSING,
        attempt_count=0,
        budget=4,
        retry_count=1,
        max_child_review_cycles=5,
        meaningful_change=True,
    )

    assert unchanged.should_retry is False
    assert "no meaningful production change" in unchanged.reason
    # Budget alone must not authorise a repeat; only a changed input does.
    assert progressed.should_retry is True


def test_every_refusal_states_the_decision_a_human_now_owns() -> None:
    """A stopped workstream must hand its operator a question, not only a counter.

    "The implementation_retry_count budget of 4 is exhausted" says what the platform
    observed. It does not say whether the requirement, the repository, or the budget is what
    should change -- and that is the decision now sitting with whoever reads it.
    """
    refusals = [
        decide_child_retry(
            classification=FailureClassification.VALIDATION_CONFIGURATION_FAILURE,
            attempt_count=0,
            budget=5,
            retry_count=0,
            max_child_review_cycles=5,
            meaningful_change=True,
        ),
        decide_child_retry(
            classification=FailureClassification.IMPLEMENTATION_MISSING,
            attempt_count=0,
            budget=4,
            retry_count=1,
            max_child_review_cycles=5,
            meaningful_change=False,
        ),
        decide_child_retry(
            classification=FailureClassification.IMPLEMENTATION_MISSING,
            attempt_count=4,
            budget=4,
            retry_count=1,
            max_child_review_cycles=9,
            meaningful_change=True,
        ),
        decide_child_retry(
            classification=FailureClassification.IMPLEMENTATION_MISSING,
            attempt_count=0,
            budget=9,
            retry_count=4,
            max_child_review_cycles=5,
            meaningful_change=True,
        ),
    ]

    assert [item.should_retry for item in refusals] == [False, False, False, False]
    for decision in refusals:
        assert decision.operator_question.endswith("?")
        # The reason describes the platform's own counters; the question must not just
        # restate them back at a human who cannot act on a counter.
        assert decision.operator_question != decision.reason


def test_triage_attributes_a_stopped_workstream_from_what_its_attempts_produced() -> None:
    """Three workstreams stop for the same refusal; the evidence says three different things.

    Handing an operator "the requirement, the repository, or the budget -- you decide" makes
    them reconstruct from artifacts what the attempts already recorded. A workstream that
    never wrote production source, one that wrote plenty and hit the same wall each time, and
    one that was converging and ran out are three different jobs for three different people.
    """

    def triage(*, written: bool, repeated: bool) -> TerminalTriage:
        """Vary only the evidence the attempts produced, holding the refusal constant."""
        return triage_stopped_workstream(
            classification=FailureClassification.IMPLEMENTATION_MISSING,
            attempts=4,
            production_source_written=written,
            diagnostics_repeated=repeated,
            fallback_question="Was this larger than its budget?",
        )

    never = triage(written=False, repeated=False)
    stuck = triage(written=True, repeated=True)
    ran_out = triage(written=True, repeated=False)

    assert never.cause is TerminalCause.NEVER_IMPLEMENTED
    assert "no production source change at all" in never.question
    assert stuck.cause is TerminalCause.BLOCKER_UNRESPONSIVE
    assert "outside the implementation's reach" in stuck.question
    # Where the evidence distinguishes nothing, it must not invent an attribution: sending
    # someone to repair a repository that was never broken is worse than asking openly.
    assert ran_out.cause is TerminalCause.BUDGET_EXHAUSTED
    assert ran_out.question == "Was this larger than its budget?"


def test_the_unresponsive_blocker_narrative_names_the_owner_the_evidence_establishes() -> None:
    """Three stuck workstreams, three owners; only a genuinely outside blocker blames outside.

    AB-Feature-176's backend died with "a defect outside the implementation's reach" about a
    syntax error in a file the feature itself wrote, and AB-Feature-177 got the same sentence
    for a capability limit. A person reading either record was sent to debug the wrong thing.
    """

    def stuck(**evidence: Any) -> TerminalTriage:
        """Hold the refusal constant; vary only the terminal evidence."""
        return triage_stopped_workstream(
            classification=FailureClassification.VALIDATION_SOURCE_FAILURE,
            attempts=4,
            production_source_written=True,
            diagnostics_repeated=True,
            fallback_question="Was this larger than its budget?",
            **evidence,
        )

    # 176's shape: the gate names one file, by absolute workspace path, and the attempt
    # lineage wrote it. The defect is the implementation's own.
    own_files = stuck(
        final_diagnostics=[
            "Pre-commit source validation failed (tool=npm; code=SOURCE_VALIDATION_FAILED).\n"
            "/workspaces/feature-1/backend/server/services/app/appBulkImport.service.js"
            ":179:23  error  Parsing error: Unexpected token ;"
        ],
        attempt_authored_paths=[
            "server/services/app/appBulkImport.service.js",
            "server/routes/app.route.js",
        ],
    )
    # 177's shape: a substantive review finding that outlived three attempts of changing
    # source. That is the tier's ceiling, not the repository's fault.
    capability = stuck(
        final_diagnostics=["The bulk path bypasses the existing create-app business flow."],
        attempt_authored_paths=["src/services/bulkImport.service.js"],
        surviving_finding="The bulk path bypasses the existing create-app business flow.",
        surviving_finding_attempts=3,
        tier_label="Economy",
    )
    # A blocker naming a file this feature never wrote is genuinely outside the
    # implementation's reach, and keeps today's sentence.
    pre_existing = stuck(
        final_diagnostics=["Cannot find module 'left-pad' required by src/vendor/legacy.js:10:1"],
        attempt_authored_paths=["src/routes/bulk.js"],
    )

    assert own_files.cause is TerminalCause.BLOCKER_UNRESPONSIVE
    assert "its own files" in own_files.question
    assert "appBulkImport.service.js" in " ".join(own_files.evidence)
    assert capability.cause is TerminalCause.BLOCKER_UNRESPONSIVE
    assert "capability limit" in capability.question
    assert "Economy" in capability.question
    assert "stronger tier" in capability.question
    assert pre_existing.cause is TerminalCause.BLOCKER_UNRESPONSIVE
    # Only the genuinely unreachable blocker keeps the outside-the-implementation sentence.
    assert "outside the implementation's reach" in pre_existing.question
    assert "outside the implementation's reach" not in own_files.question
    assert "outside the implementation's reach" not in capability.question


def test_a_capability_narrative_requires_the_finding_to_survive_three_attempts() -> None:
    """One or two survivals stay an open question; guessing capability too early misleads."""
    early = triage_stopped_workstream(
        classification=FailureClassification.VALIDATION_SOURCE_FAILURE,
        attempts=2,
        production_source_written=True,
        diagnostics_repeated=True,
        fallback_question="Was this larger than its budget?",
        surviving_finding="The bulk path bypasses the existing create-app business flow.",
        surviving_finding_attempts=2,
        tier_label="Economy",
    )

    assert "outside the implementation's reach" in early.question


def test_a_resubmission_stop_reports_the_tree_it_measured_not_the_diagnostics() -> None:
    """77- item 33, the 207 regression pin: the evidence states what the firing rule compared.

    AB-Feature-207's stop fired on `production_diff_fingerprint` equality while its
    diagnostics changed between attempts 3 and 4, and the fixed sentence "The blocking
    diagnostics did not change between attempts." misled the failure summary and a forensic
    pass. A tree comparison reports a tree, with both fingerprints on the record.
    """
    fingerprint = "sha256:aa11"
    triage = triage_stopped_workstream(
        classification=FailureClassification.IMPLEMENTATION_MISSING,
        attempts=5,
        production_source_written=True,
        diagnostics_repeated=False,
        resubmission=TreeResubmission(
            attempt=3,
            production_diff_fingerprint=fingerprint,
            matched_production_diff_fingerprint=fingerprint,
        ),
        fallback_question="Was this larger than its budget?",
    )

    assert triage.cause is TerminalCause.BLOCKER_UNRESPONSIVE
    assert "This attempt resubmitted the production tree of attempt 3." in triage.evidence
    assert not any("did not change between attempts" in line for line in triage.evidence)
    # The question must not send anyone to repair a repository over a circling attempt.
    assert "outside the implementation's reach" not in triage.question
    assert "circling" in triage.question
    assert triage.metadata == {
        "resubmitted_production_diff_fingerprint": fingerprint,
        "matched_production_diff_fingerprint": fingerprint,
        "resubmitted_attempt": "3",
    }


def test_a_diagnostic_repeat_keeps_its_measured_sentence_and_a_double_fact_states_both() -> None:
    """The diagnostics sentence survives exactly where a diagnostic comparison fired."""
    repeated_only = triage_stopped_workstream(
        classification=FailureClassification.VALIDATION_SOURCE_FAILURE,
        attempts=4,
        production_source_written=True,
        diagnostics_repeated=True,
        fallback_question="Was this larger than its budget?",
    )
    both = triage_stopped_workstream(
        classification=FailureClassification.VALIDATION_SOURCE_FAILURE,
        attempts=4,
        production_source_written=True,
        diagnostics_repeated=True,
        resubmission=TreeResubmission(attempt=1, production_diff_fingerprint="sha256:bb22"),
        fallback_question="Was this larger than its budget?",
    )

    assert "The blocking diagnostics did not change between attempts." in repeated_only.evidence
    assert not any("resubmitted the production tree" in line for line in repeated_only.evidence)
    assert repeated_only.metadata == {}
    assert "The blocking diagnostics did not change between attempts." in both.evidence
    assert "This attempt resubmitted the production tree of attempt 1." in both.evidence


def test_a_resubmission_without_a_matched_attempt_still_states_the_fact() -> None:
    """An immediate repeat judged before its predecessor's artifact landed keeps the sentence."""
    triage = triage_stopped_workstream(
        classification=FailureClassification.IMPLEMENTATION_MISSING,
        attempts=2,
        production_source_written=True,
        resubmission=TreeResubmission(production_diff_fingerprint="sha256:cc33"),
        fallback_question="Was this larger than its budget?",
    )

    assert "This attempt resubmitted the production tree of an earlier attempt." in triage.evidence
    assert triage.metadata == {"resubmitted_production_diff_fingerprint": "sha256:cc33"}


def test_a_resubmission_with_authored_paths_does_not_claim_the_same_defect() -> None:
    """The feature-authored branch's 'same defect' claim needs the diagnostic comparison.

    207's actual terminal record: authored paths, a resubmitted tree, and diagnostics that
    had changed -- yet the question said the implementation "repeatedly reintroduced the
    same defect". The files sentence stays; the unmeasured claim goes.
    """
    triage = triage_stopped_workstream(
        classification=FailureClassification.IMPLEMENTATION_MISSING,
        attempts=5,
        production_source_written=True,
        resubmission=TreeResubmission(attempt=3, production_diff_fingerprint="sha256:dd44"),
        final_diagnostics=["src/services/import.service.js:12:1 error Unexpected token ;"],
        attempt_authored_paths=["src/services/import.service.js"],
        fallback_question="Was this larger than its budget?",
    )

    assert triage.cause is TerminalCause.BLOCKER_UNRESPONSIVE
    assert "its own files" in triage.question
    assert "the same defect" not in triage.question
    assert "This attempt resubmitted the production tree of attempt 3." in triage.evidence
    assert not any("did not change between attempts" in line for line in triage.evidence)


def test_a_blocked_checkout_is_triaged_to_the_repository_however_it_stopped() -> None:
    """A checkout that cannot validate itself is never the requirement's fault."""
    triage = triage_stopped_workstream(
        classification=FailureClassification.VALIDATION_CONFIGURATION_FAILURE,
        attempts=1,
        production_source_written=False,
        diagnostics_repeated=False,
        fallback_question="Was this larger than its budget?",
    )

    assert triage.cause is TerminalCause.REPOSITORY_SETUP
    # Not NEVER_IMPLEMENTED: nothing was written because nothing was allowed to start.
    assert "untouched checkout" in triage.question


def _test_command_result(
    *,
    timed_out: bool = False,
    return_code: int | None = 1,
    stderr: str = "",
) -> ValidationResult:
    """Build one required test-command result the way a live run reports it."""
    return ValidationResult(
        command=("npm", "run", "test"),
        return_code=return_code,
        stdout="",
        stderr=stderr,
        timed_out=timed_out,
        duration_seconds=900.0 if timed_out else 12.0,
        validation_type="test",
        repository_id="AB-console-admin-2.0",
    )


def test_a_test_command_that_times_out_is_not_a_source_failure() -> None:
    """AB-Feature-108's last attempts: a 900s timeout charged to the coding budget.

    The command never reported a verdict on the source, so there was nothing for an
    engineer attempt to answer -- and six of them were spent answering it anyway.
    """
    timed_out = _test_command_result(timed_out=True, return_code=None)

    assert timed_out.failure_classification == VALIDATION_TIMED_OUT
    classification = classify_failure(validation_results=(timed_out,))
    assert classification is FailureClassification.VALIDATION_CAPACITY_FAILURE
    assert retry_counter_field(classification) != "validation_retry_count"
    assert retry_counter_field(classification) != "implementation_retry_count"


def test_a_test_command_that_exhausts_memory_is_not_a_source_failure() -> None:
    """The other half of -108's attempt 8: Node died on its heap limit, exit code and all."""
    exhausted = _test_command_result(
        return_code=134,
        stderr=(
            "FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory"
        ),
    )

    assert exhausted.failure_classification == VALIDATION_RESOURCE_EXHAUSTED
    assert (
        classify_failure(validation_results=(exhausted,))
        is FailureClassification.VALIDATION_CAPACITY_FAILURE
    )


def test_a_killed_validation_command_is_read_as_exhaustion_not_rejection() -> None:
    """The OOM killer leaves no output at all -- only the signal it killed the child with."""
    killed = _test_command_result(return_code=137)

    assert killed.failure_classification == VALIDATION_RESOURCE_EXHAUSTED


def test_a_command_that_completed_and_failed_is_still_a_source_failure() -> None:
    """The fix must not swallow the ordinary case: a real test failure still reaches coding."""
    failed = _test_command_result(
        return_code=1,
        stderr="FAIL src/pages/AllApps.test.js\n  TypeError: undefined is not iterable",
    )

    assert failed.failure_classification is None
    assert (
        classify_failure(validation_results=(failed,))
        is FailureClassification.VALIDATION_SOURCE_FAILURE
    )


def test_a_capacity_failure_exhausts_its_budget_without_a_production_diff() -> None:
    """The retry it is owed is to scope the command, which changes no application source."""
    permitted = decide_child_retry(
        classification=FailureClassification.VALIDATION_CAPACITY_FAILURE,
        attempt_count=0,
        budget=1,
        retry_count=0,
        max_child_review_cycles=5,
        # A scoped command is not a production change, and demanding one here would refuse
        # the single attempt that could make the command fit.
        meaningful_change=False,
    )
    assert permitted.should_retry is True

    exhausted = decide_child_retry(
        classification=FailureClassification.VALIDATION_CAPACITY_FAILURE,
        attempt_count=1,
        budget=1,
        retry_count=1,
        max_child_review_cycles=5,
        meaningful_change=False,
    )
    assert exhausted.should_retry is False
    # Not the generic budget question: nobody should be asked whether the requirement
    # should change because a test suite ran out of memory.
    assert "exhausted the machine's memory" in exhausted.operator_question
    assert "should the requirement" not in exhausted.operator_question


def test_a_capacity_failure_is_triaged_to_the_command_not_the_implementation() -> None:
    """Production source was written here, so 'nothing was ever built' is the wrong answer."""
    triage = triage_stopped_workstream(
        classification=FailureClassification.VALIDATION_CAPACITY_FAILURE,
        attempts=9,
        production_source_written=False,
        diagnostics_repeated=True,
        fallback_question="Was this larger than its budget?",
    )

    assert triage.cause is TerminalCause.VALIDATION_CAPACITY
    assert "never finished" in triage.question


def test_the_same_defect_worded_three_ways_has_one_signature() -> None:
    """-108's attempts 5, 6 and 7: one TypeError at one line, described differently each time.

    The loop compared the prose, saw three different strings, and allowed all three.
    """
    fifth = [
        "The required npm run test command exits 1 with three failed AllApps tests. The "
        "reported render failure is `TypeError: undefined is not iterable` at "
        "AllApps.js:117 while destructuring `const [{ loading }, refetch] = useAxios(...)`.",
        "npm run test exited with code 1.",
    ]
    sixth = [
        "The required `npm run test` command exits 1 with three failed AllApps tests. Each "
        "failure is caused by `TypeError: undefined is not iterable` at AllApps.js:117 "
        "while destructuring the return from useAxios for the main app-list request.",
        "npm run test exited with code 1.",
    ]
    seventh = [
        "The required npm run test command exits 1 with one failed suite and three failed "
        "AllApps bulk-deletion tests. The tests fail during AllApps rendering with "
        "TypeError: undefined is not iterable while destructuring the main useAxios result "
        "at AllApps.js:117.",
        "npm run test exited with code 1.",
    ]

    assert fifth != sixth != seventh
    assert diagnostic_signature(fifth) == diagnostic_signature(sixth)
    assert diagnostic_signature(sixth) == diagnostic_signature(seventh)
    assert "at:AllApps.js:117" in diagnostic_signature(fifth)
    assert "error:TypeError" in diagnostic_signature(fifth)


def test_different_defects_in_the_same_file_keep_different_signatures() -> None:
    """-108's attempts 3 and 4 were genuinely different problems and must stay so.

    Over-merging here would end a workstream that is working through a list of real
    defects one at a time, which is exactly what converging looks like.
    """
    lint = [
        "Pre-commit source validation failed (tool=eslint; "
        "code=SOURCE_VALIDATION_FAILED_EXIT_1).\n"
        "/workspaces/feature-1/AB-console-admin-2.0/src/pages/AllApps.test.js\n"
        "  51:47  error  Component definition is missing display name  react/display-name",
    ]
    mock_factory = [
        "npm run test exited 1. Jest rejects the axios-hooks jest.mock factory in "
        "AllApps.test.js because it captures deleteResponse from outer scope.",
        "npm run test exited with code 1.",
    ]

    assert diagnostic_signature(lint) != diagnostic_signature(mock_factory)
    # A workspace path must not be mistaken for a scoped linter rule.
    assert "rule:react/display-name" in diagnostic_signature(lint)
    assert not any(token.startswith("rule:src/") for token in diagnostic_signature(lint))


def test_a_signature_that_identifies_nothing_never_compares_equal() -> None:
    """Two unidentifiable reports are unknown, not proof that nothing changed."""
    assert diagnostic_signature(["something went wrong"]) == ()
    assert diagnostic_signature([]) == ()


def test_a_permitted_retry_asks_the_operator_nothing() -> None:
    """Work that may continue is not an escalation, and must not read like one."""
    decision = decide_child_retry(
        classification=FailureClassification.IMPLEMENTATION_MISSING,
        attempt_count=0,
        budget=4,
        retry_count=1,
        max_child_review_cycles=5,
        meaningful_change=True,
    )

    assert decision.should_retry is True
    assert decision.operator_question == ""


def test_structural_repository_defects_are_never_retried_by_coding_again() -> None:
    """A broken checked-in configuration cannot be fixed by another engineer attempt."""
    decision = decide_child_retry(
        classification=FailureClassification.VALIDATION_CONFIGURATION_FAILURE,
        attempt_count=0,
        budget=5,
        retry_count=0,
        max_child_review_cycles=5,
        meaningful_change=True,
    )

    assert decision.should_retry is False
    assert decision.counter_field == "repository_setup_retry_count"


def test_source_lint_violations_stay_the_feature_engineer_s_responsibility(tmp_path: Path) -> None:
    """A genuine code-quality failure must not be misreported as a repository defect."""
    (tmp_path / "package.json").write_text('{"scripts":{"lint":"eslint ."}}', encoding="utf-8")

    lint = classify_lint_failure(
        result_stdout="src/app.js\n  1:1  error  Unexpected var  no-var\n\n1 problem",
        result_stderr="",
        repository_root=tmp_path,
    )

    assert lint.kind == "lint_violation"
    assert lint.disposition is RepositoryHealthDisposition.REQUIRES_FEATURE_ENGINEER


@pytest.mark.asyncio
async def test_an_undeclared_shared_lint_config_blocks_before_any_coding_begins(
    tmp_path: Path,
) -> None:
    """The repository defect that exhausted feature -007's retries now stops the child first."""
    (tmp_path / "package.json").write_text(
        '{"scripts":{"lint":"eslint ."},"eslintConfig":{"extends":"airbnb-base"}}',
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")
    preflight = await RepositoryPreflight(process_runner=StubProcessRunner()).run(
        tmp_path, repository_id="backend"
    )
    lint_failure = ValidationResult(
        command=("npm", "run", "lint"),
        return_code=2,
        stdout="",
        stderr="Failed to load config airbnb-base: Cannot find module eslint-config-airbnb-base",
        timed_out=False,
        duration_seconds=0.1,
        validation_type="lint",
        repository_id="backend",
        status=ValidationStatus.FAILED,
        failure_classification="missing_dependency",
    )

    blocked = _block_unrepairable_lint_configuration(preflight, [lint_failure], tmp_path)

    assert preflight.validation_readiness != "blocked"
    assert blocked.validation_readiness == "blocked"
    assert blocked.blocking_issues[-1].issue_id == "LINT_CONFIGURATION_REQUIRES_HUMAN"
    assert blocked.blocking_issues[-1].automatically_repairable is False
    # Classified as repository setup, so it cannot consume the implementation retry budget.
    assert retry_counter_field(classify_failure(preflight=blocked)) != "implementation_retry_count"


@pytest.mark.asyncio
async def test_missing_test_command_is_honest_not_a_pass(tmp_path: Path) -> None:
    """No configured test script produces the public structural status, never a pass."""
    (tmp_path / "package.json").write_text('{"scripts":{"lint":"eslint ."}}', encoding="utf-8")
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")
    preflight = await RepositoryPreflight(process_runner=StubProcessRunner()).run(
        tmp_path, repository_id="backend"
    )
    result = ValidationResult(
        command=(),
        return_code=None,
        stdout="",
        stderr="",
        timed_out=False,
        duration_seconds=0.0,
        validation_type="test",
        status=ValidationStatus.NOT_CONFIGURED,
    )

    assert any(issue.issue_id == "TEST_COMMAND_NOT_CONFIGURED" for issue in preflight.warnings)
    assert result.result_code == "TEST_COMMAND_NOT_CONFIGURED"
    assert not result.succeeded


@pytest.mark.asyncio
async def test_preflight_bootstraps_each_frozen_dependency_tool_in_a_mixed_checkout(
    tmp_path: Path,
) -> None:
    """A Python and Node project select their own evidenced installers and directories."""
    (tmp_path / "api").mkdir()
    (tmp_path / "web").mkdir()
    (tmp_path / "api" / "pyproject.toml").write_text("[project]\nname='api'\n", encoding="utf-8")
    (tmp_path / "api" / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "web" / "package.json").write_text('{"name":"web"}', encoding="utf-8")
    (tmp_path / "web" / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")
    runner = StubProcessRunner()

    result = await RepositoryPreflight(process_runner=runner).run(tmp_path, repository_id="mixed")

    assert runner.calls == [("uv", "sync", "--frozen"), ("npm", "ci", "--no-audit")]
    assert runner.working_directories == [tmp_path / "api", tmp_path / "web"]
    assert result.package_managers == ["npm", "uv"]
    assert result.dependency_install_working_directories == ["api", "web"]
    assert result.validation_readiness != "blocked"


def test_a_planned_configuration_category_the_checkout_does_not_need_is_advisory() -> None:
    """Mounting an existing component needs no configuration file, so it cannot block."""
    workstream = RepositoryWorkstreamPlan.model_validate(
        {
            **_backend_workstream().model_dump(mode="json"),
            "role": "other",
            "implementation_expectations": [
                {
                    "requirement_id": "status-api",
                    # The planner guessed this category before the repository was checked out.
                    "expected_change_categories": ["production", "configuration", "test"],
                    "expected_source_areas": [],
                    "tests_required": True,
                }
            ],
        }
    )
    completion = _completion(
        [
            FileChange(path="src/components/Tiles.js", change_type="modified", description="mount"),
            FileChange(path="src/components/Tile.test.js", change_type="added", description="test"),
        ],
        implemented=True,
    )

    result = validate_implementation_completeness(workstream, completion)

    assert result.passed
    assert result.requirements_not_implemented == []


def test_generic_production_category_accepts_a_non_conventional_python_layout() -> None:
    """Completion needs production evidence, not route/controller naming conventions."""
    workstream = RepositoryWorkstreamPlan.model_validate(
        {
            **_backend_workstream().model_dump(mode="json"),
            "role": "other",
            "implementation_expectations": [
                {
                    "requirement_id": "status-api",
                    "expected_change_categories": ["production", "test"],
                    "expected_source_areas": [],
                    "tests_required": True,
                }
            ],
        }
    )
    completion = _completion(
        [
            FileChange(
                path="application/handlers/status.py",
                change_type="added",
                description="status handler",
            ),
            FileChange(
                path="application/app.py",
                change_type="modified",
                description="register the handler",
            ),
            FileChange(
                path="checks/test_status.py", change_type="added", description="status test"
            ),
        ],
        implemented=True,
    )

    result = validate_implementation_completeness(workstream, completion)

    assert result.passed


def test_tests_only_backend_change_fails_completion_and_complete_change_passes() -> None:
    """A route/controller workstream cannot be completed through test-file edits alone."""
    workstream = _backend_workstream()
    tests_only = _completion(
        [FileChange(path="tests/status.test.js", change_type="modified", description="test")]
    )
    complete = _completion(
        [
            FileChange(path="src/routes/status.js", change_type="added", description="route"),
            FileChange(
                path="src/controllers/status.js", change_type="added", description="controller"
            ),
            FileChange(path="src/index.js", change_type="modified", description="register"),
            FileChange(path="src/services/status.js", change_type="added", description="service"),
            FileChange(path="tests/status.test.js", change_type="added", description="test"),
        ],
        implemented=True,
    )

    only_result = validate_implementation_completeness(workstream, tests_only)
    complete_result = validate_implementation_completeness(workstream, complete)

    assert not only_result.passed
    assert {item.code for item in only_result.findings} >= {"tests_only_change"}
    assert not only_result.production_files_changed
    assert complete_result.passed
    assert complete_result.requirements_implemented == ["status-api"]


def test_explicit_source_categories_cannot_be_satisfied_by_an_unrelated_source_file() -> None:
    """A source edit plus tests is not proof that promised route/controller/service work exists."""
    completion = _completion(
        [
            FileChange(path="src/misc.js", change_type="added", description="unrelated source"),
            FileChange(path="tests/status.test.js", change_type="added", description="test"),
        ],
        implemented=True,
    )

    result = validate_implementation_completeness(_backend_workstream(), completion)

    assert not result.passed
    assert result.requirements_not_implemented == ["status-api"]
    assert any("controller" in item.description for item in result.findings)


def test_same_path_with_new_content_fingerprint_is_meaningful_retry_progress() -> None:
    """Remediation often changes bytes in the same file, which a path-only hash hid."""
    completion = _completion(
        [FileChange(path="src/routes/status.js", change_type="modified", description="route")],
        implemented=True,
    ).model_copy(update={"production_diff_fingerprint": "content-version-two"})
    result = validate_implementation_completeness(_backend_workstream(), completion)

    meaningful, reason = meaningful_progress(result, previous_fingerprint="content-version-one")

    assert meaningful
    assert reason == "new_or_material_production_source_implementation"


def test_adding_only_new_files_does_not_satisfy_the_integration_promise() -> None:
    """New code nothing running refers to is unfinished, and the plan now says so.

    Every clean run measured on this platform failed at the same step: the route was never
    registered or the tile never rendered. Detecting that afterwards took three heuristics
    and each had a hole, most recently a component placed in a brand-new directory where
    there were no neighbours to compare against. It is a promise the plan can simply make.
    """
    workstream = RepositoryWorkstreamPlan.model_validate(
        {
            **_backend_workstream().model_dump(mode="json"),
            "implementation_expectations": [
                {
                    "requirement_id": "status-api",
                    "expected_change_categories": ["production", "test", "integration"],
                    "expected_source_areas": [],
                    "tests_required": True,
                }
            ],
        }
    )
    added_only = _completion(
        [
            FileChange(path="src/routes/status.js", change_type="added", description="route"),
            FileChange(path="test/status.test.js", change_type="added", description="test"),
        ],
        implemented=True,
    )
    wired = _completion(
        [
            FileChange(path="src/routes/status.js", change_type="added", description="route"),
            FileChange(path="src/index.js", change_type="modified", description="register"),
            FileChange(path="test/status.test.js", change_type="added", description="test"),
        ],
        implemented=True,
    )

    unfinished = validate_implementation_completeness(workstream, added_only)
    finished = validate_implementation_completeness(workstream, wired)

    assert not unfinished.passed
    assert any("integration" in item.description for item in unfinished.findings)
    assert any("nothing that already runs" in item.description for item in unfinished.findings)
    # Editing the file that registers the new route is what completes the requirement.
    assert finished.passed


def test_a_second_identical_resubmission_is_what_ends_the_loop() -> None:
    """One repeat buys a further attempt; the second does not.

    r-2's backend had registered its route and was one documentation edit from approval when
    it resent the same five files, and the loop ended with six of its eight attempts unspent.
    ser-1 is the other side of the trade: ten identical attempts before its budget ran out.
    """
    completion = _completion(
        [
            FileChange(path="src/routes/status.js", change_type="added", description="route"),
            FileChange(path="src/index.js", change_type="modified", description="register"),
            FileChange(path="test/status.test.js", change_type="added", description="test"),
        ],
        implemented=True,
    ).model_copy(update={"production_diff_fingerprint": "identical-content"})
    result = validate_implementation_completeness(_backend_workstream(), completion)

    repeated, reason = meaningful_progress(result, previous_fingerprint="identical-content")

    # The reason the loop keys on is unchanged; the allowance lives in the retry loop, which
    # grants exactly one before this verdict is enforced.
    assert not repeated
    assert reason == "attempt_reproduced_the_previous_production_change"


def test_resending_the_previous_change_is_not_progress() -> None:
    """An identical resubmission must end the loop, not spend the rest of the budget.

    The test branch asked only whether tests and production files were both present, so an
    attempt that returned byte-identical files counted as progress for as long as it carried
    a test. ser-1 spent ten attempts in both repositories re-sending the same unwired change
    to the same gate before its budget ran out.
    """
    completion = _completion(
        [
            FileChange(path="src/routes/status.js", change_type="added", description="route"),
            FileChange(path="test/status.test.js", change_type="added", description="test"),
        ],
        implemented=True,
    ).model_copy(update={"production_diff_fingerprint": "identical-content"})
    result = validate_implementation_completeness(_backend_workstream(), completion)

    repeated, repeated_reason = meaningful_progress(
        result, previous_fingerprint="identical-content"
    )
    changed, changed_reason = meaningful_progress(result, previous_fingerprint="earlier-content")

    assert not repeated
    assert repeated_reason == "attempt_reproduced_the_previous_production_change"
    # A genuinely different change is still progress.
    assert changed
    assert changed_reason == "new_or_material_production_source_implementation"


def test_a_repeated_lint_rejection_still_counts_as_progress() -> None:
    """A linter repeats itself while a rule is broken; that is not the model stalling.

    The case is rep-a's frontend, stopped after three of its eight attempts because its
    eslint output did not change while the source behind it did. Written with a previous
    fingerprint on purpose: passing `None` describes the first rejection, not a repeat, and
    that gap is what let -105 spend eleven attempts here.
    """
    completion = _completion([])
    result = validate_implementation_completeness(_backend_workstream(), completion)

    allowed, reason = meaningful_progress(
        result,
        previous_fingerprint="a-different-attempt",
        source_rejected_before_commit=True,
        diagnostics_changed=False,
    )

    assert allowed
    assert reason == "source_rejected_before_commit_with_changed_input"


def test_the_same_source_rejected_with_the_same_diagnostic_is_not_progress() -> None:
    """AB-Feature-105: eleven attempts, one missing import, nothing ever changing.

    Byte-identical production source and byte-identical eslint output --
    `'BulkDeleteApps' is not defined` -- and every guard read it as progress because the
    pre-commit exemption was unconditional. Repeating an input that produced an identical
    output is the definition of not converging, whatever rejected it.
    """
    completion = _completion([])
    result = validate_implementation_completeness(_backend_workstream(), completion)

    refused, reason = meaningful_progress(
        result,
        previous_fingerprint=result.production_diff_fingerprint,
        source_rejected_before_commit=True,
        diagnostics_changed=False,
    )

    assert not refused
    assert reason != "source_rejected_before_commit_with_changed_input"


def test_rewriting_tests_against_an_unchanged_diagnostic_is_not_progress() -> None:
    """The second escape hatch -105 used once its validation budget was spent.

    Editing tests answers a reviewer that rejected an attempt over its tests. It does not
    answer a production defect: with the same production source and the same diagnostic,
    the broken thing is untouched and the churn merely buys another attempt.
    """
    completion = _completion([])
    result = validate_implementation_completeness(_backend_workstream(), completion)

    allowed, allowed_reason = meaningful_progress(
        result,
        previous_fingerprint=result.production_diff_fingerprint,
        previous_test_fingerprint="an-older-test-fingerprint",
        diagnostics_changed=True,
    )
    refused, refused_reason = meaningful_progress(
        result,
        previous_fingerprint=result.production_diff_fingerprint,
        previous_test_fingerprint="an-older-test-fingerprint",
        diagnostics_changed=False,
    )

    # A reviewer that said something new is still answered by editing the tests.
    assert allowed_reason == "new_test_source_for_unchanged_production_implementation"
    assert allowed
    assert not refused
    assert refused_reason == "attempt_reproduced_the_previous_production_change"


def test_a_change_the_commit_gate_rejected_may_still_be_retried() -> None:
    """A lint rejection commits nothing, which must not read as writing nothing."""
    # The pre-commit gate blocked the commit, so the attempt reports no changed files at all.
    completion = _completion([])
    result = validate_implementation_completeness(_backend_workstream(), completion)

    refused, refused_reason = meaningful_progress(result, previous_fingerprint=None)
    allowed, allowed_reason = meaningful_progress(
        result, previous_fingerprint=None, source_rejected_before_commit=True
    )

    assert not refused
    assert refused_reason == "no_material_change_detected"
    assert allowed
    assert allowed_reason == "source_rejected_before_commit_with_changed_input"


def test_reviewer_emits_backend_implementation_findings_for_tests_only_change() -> None:
    """Reviewer findings distinguish the missing backend source pieces from test evidence."""
    completion = _completion(
        [FileChange(path="tests/status.test.js", change_type="modified", description="test")]
    )
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}
    scope = {
        "repository_id": "backend",
        "role": "backend",
        "repository_revision": "revision-a",
        "scoped_requirements": [{"requirement_id": "status-api", "responsibility": "implements"}],
        "implementation_expectations": [
            {
                "requirement_id": "status-api",
                "expected_change_categories": ["route", "controller", "service", "test"],
                "expected_source_areas": ["src/routes", "src/controllers", "src/services"],
                "tests_required": True,
            }
        ],
    }

    _apply_completion_findings(payload, completion, scope)

    ids = {item["finding_id"] for item in payload["findings"]}
    assert payload["verdict"] == "changes_requested"
    assert ids >= {
        "BACKEND_ROUTE_NOT_IMPLEMENTED",
        "BACKEND_CONTROLLER_NOT_IMPLEMENTED",
        "TESTS_ONLY_CHANGE",
    }


def test_retry_plan_changes_strategy_and_keeps_setup_budget_separate() -> None:
    """Implementation retries target missing source rather than repeating tests."""
    plan = build_retry_plan(
        FailureClassification.IMPLEMENTATION_MISSING,
        expected_source_areas=["src/routes", "src/controllers"],
        previous_findings=["Tests-only change."],
        current_revision="revision-2",
    )

    assert "production" in plan.required_strategy_change
    assert plan.files_or_areas_to_inspect == ["src/routes", "src/controllers"]
    assert retry_counter_field(FailureClassification.DEPENDENCY_INSTALLATION_FAILURE) == (
        "repository_setup_retry_count"
    )
    assert retry_counter_field(FailureClassification.DEPENDENCY_INSTALLATION_FAILURE) != (
        "implementation_retry_count"
    )


def test_retry_plan_carries_the_findings_the_severity_filter_did_not_block_on() -> None:
    """A finding raised once must not have to be raised again to reach the model.

    AB-Feature-215's backend review named two defects, one `high` and one `medium`.
    `_blocking_findings` hands the remediation only the criticals and highs whenever any
    exist, which is the right work order -- but the medium used to be dropped outright, so the
    next attempt fixed the high, was failed by the medium, and spent a whole retry on a defect
    the platform had already written down and was holding.
    """
    plan = build_retry_plan(
        FailureClassification.REVIEW_SCOPE_FAILURE,
        expected_source_areas=["server/controllers"],
        previous_findings=["The route guard and the handler disagree on who may export."],
        current_revision="revision-2",
        advisory_findings=["The service test proves one retrieval page, not several."],
    )

    assert plan.advisory_findings == ["The service test proves one retrieval page, not several."]
    # Carried beside the work order, never folded into it: these are not why the attempt failed.
    assert "retrieval page" not in plan.root_cause


def test_a_finding_that_leads_the_work_order_is_not_also_offered_as_advisory() -> None:
    """The same sentence in both lists reads as two demands and invites duplicate edits."""
    repeated = "The route guard and the handler disagree on who may export."

    plan = build_retry_plan(
        FailureClassification.REVIEW_SCOPE_FAILURE,
        expected_source_areas=["server/controllers"],
        previous_findings=[repeated],
        current_revision="revision-2",
        advisory_findings=[repeated, "A second, genuinely unblocked remark."],
    )

    assert plan.advisory_findings == ["A second, genuinely unblocked remark."]


def test_a_retry_without_advisory_findings_carries_none() -> None:
    """The field is additive: a review that blocked on everything it raised adds nothing."""
    plan = build_retry_plan(
        FailureClassification.VALIDATION_SOURCE_FAILURE,
        expected_source_areas=["src"],
        previous_findings=["The change's own tests failed."],
        current_revision="revision-3",
    )

    assert plan.advisory_findings == []


@pytest.mark.asyncio
async def test_backend_setup_failure_does_not_starve_an_independent_sibling() -> None:
    """A setup-blocked backend must not stop an unrelated repository from being delivered.

    The frontend only consumes the approved contract, so it has no reason to wait. Leaving
    it `pending` is what produced ten live runs in which it never executed at all.
    """
    request = StartFeatureRequest.model_validate(_feature_payload())
    state = _initial_feature_state("preflight-retry", request)
    executor = ControlledChildExecutor()
    orchestrator = FeatureWorkflowOrchestrator(child_executor=executor)

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert set(executor.calls) == {"backend", "frontend"}
    assert result.child_workflows["frontend"].status is ChildWorkflowStatus.APPROVED
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.FAILED
    # A dependency install is often transient infrastructure, so it is bounded by the
    # repository-setup budget rather than the code-change guard, and stops at it.
    assert executor.calls.count("backend") == 2
    assert executor.calls.count("frontend") == 1
    assert result.child_workflows["backend"].repository_setup_retry_count == 1
    assert "repository_setup_retry_count" in (
        result.child_workflows["backend"].retry_refusal_reason or ""
    )


async def _stopped_backend(
    executor: Any,
) -> tuple[FeatureWorkflowOrchestrator, FeatureWorkflowSnapshot]:
    """Run a feature until the backend has stopped with its setup budget spent."""
    request = StartFeatureRequest.model_validate(_feature_payload())
    state = _initial_feature_state("granted-retry", request)
    orchestrator = FeatureWorkflowOrchestrator(child_executor=executor)
    stopped = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )
    assert stopped.child_workflows["backend"].status is ChildWorkflowStatus.FAILED
    return orchestrator, stopped


@pytest.mark.asyncio
async def test_an_operator_grant_buys_the_attempt_the_platform_refused() -> None:
    """The platform stops; a person who knows more may spend one more attempt.

    Ordinary resume deliberately will not do this. It refuses because repeating an attempt on
    unchanged inputs reaches the same place at the same cost, and that refusal is right for
    everything the platform can see on its own. What it cannot see is somebody having gone and
    fixed the repository, which is exactly the information the grant carries.
    """
    executor = RepairedAfterGrantExecutor()
    orchestrator, stopped = await _stopped_backend(executor)
    attempts_before = executor.calls.count("backend")

    resumed = await orchestrator.resume(
        stopped,
        answers=[],
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
    )
    # Resume alone changes nothing: the budget is spent and no new information arrived.
    assert executor.calls.count("backend") == attempts_before
    assert resumed.child_workflows["backend"].status is ChildWorkflowStatus.FAILED

    granted = await orchestrator.retry_workstream(
        stopped,
        repository_id="backend",
        additional_attempts=1,
        requested_by="akhilesh",
        reason="Installed the missing private registry token on the runner.",
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
    )

    assert executor.calls.count("backend") == attempts_before + 1
    child = granted.child_workflows["backend"]
    assert child.status is ChildWorkflowStatus.APPROVED
    # The override is in the record as an override, with who made it and why. A counter that
    # had quietly passed its configured limit would read as a platform defect instead.
    assert child.granted_extra_attempts == 1
    assert child.retry_grants[0]["granted_by"] == "akhilesh"
    assert "registry token" in child.retry_grants[0]["reason"]
    assert "repository_setup_retry_count" in child.retry_grants[0]["stopped_because"]
    assert child.retry_refusal_reason is None


@pytest.mark.asyncio
async def test_a_grant_raises_one_repository_ceiling_and_no_sibling() -> None:
    """An allowance made for a repository somebody read must not be inherited by its siblings."""
    executor = RepairedAfterGrantExecutor()
    orchestrator, stopped = await _stopped_backend(executor)
    before = stopped.max_repository_setup_retries

    granted = await orchestrator.retry_workstream(
        stopped,
        repository_id="backend",
        additional_attempts=2,
        requested_by="akhilesh",
        reason="Repaired the lockfile.",
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
    )

    # The feature's configured limits are untouched, so a repository nobody looked at is
    # still held to the budget it was given.
    assert granted.max_repository_setup_retries == before
    assert granted.child_workflows["frontend"].granted_extra_attempts == 0


@pytest.mark.asyncio
async def test_a_grant_is_refused_for_a_repository_that_has_nothing_to_retry() -> None:
    """Retrying an approved repository would discard work that already passed review."""
    executor = RepairedAfterGrantExecutor()
    orchestrator, stopped = await _stopped_backend(executor)
    assert stopped.child_workflows["frontend"].status is ChildWorkflowStatus.APPROVED
    credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)

    with pytest.raises(FeatureWorkflowError, match="nothing to retry"):
        await orchestrator.retry_workstream(
            stopped,
            repository_id="frontend",
            additional_attempts=1,
            requested_by="akhilesh",
            reason="Just in case.",
            credentials=credentials,
        )

    with pytest.raises(FeatureWorkflowError, match="not part of this feature"):
        await orchestrator.retry_workstream(
            stopped,
            repository_id="mobile",
            additional_attempts=1,
            requested_by="akhilesh",
            reason="Typo in the repository id.",
            credentials=credentials,
        )


@pytest.mark.asyncio
async def test_a_granted_attempt_is_not_charged_to_the_integration_allowance() -> None:
    """A person's grant must not spend the review cycles the contract loop needs.

    Targeted re-runs already existed for contract remediation, and they draw on the shared
    integration budget because that is whose loop asked for them. Reusing that path for an
    operator grant would take those cycles away from the parent silently.
    """
    executor = RepairedAfterGrantExecutor()
    orchestrator, stopped = await _stopped_backend(executor)
    before = stopped.child_workflows["backend"].integration_retry_count

    granted = await orchestrator.retry_workstream(
        stopped,
        repository_id="backend",
        additional_attempts=1,
        requested_by="akhilesh",
        reason="Repaired the runner.",
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
    )

    child = granted.child_workflows["backend"]
    assert child.integration_retry_count == before
    # retry_count still advances: reusing the previous number would collide with the earlier
    # attempt's completion, review and result artifact IDs.
    assert child.retry_count > stopped.child_workflows["backend"].retry_count


@pytest.mark.asyncio
async def test_a_repeated_tests_only_attempt_is_refused_immediately() -> None:
    """An attempt whose inputs did not change must not run at all.

    The loop-prevention invariant is absolute: previously an unchanged fingerprint still
    bought two further attempts, which is how a child burned its budget re-submitting
    identical work.
    """
    request = StartFeatureRequest.model_validate(_feature_payload())
    state = _initial_feature_state("tests-only-retries", request)
    executor = OnlyTestChangesExecutor()
    orchestrator = FeatureWorkflowOrchestrator(child_executor=executor)

    result = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    # Counted per repository: every workstream now fans out, so a shared total would
    # silently change whenever repository cardinality does. A changed prompt alone is not a
    # changed relevant input when the attempt produced no implementation to build on.
    assert executor.calls["backend"] == 1
    assert result.child_workflows["backend"].meaningful_change is False
    assert result.child_workflows["backend"].implementation_retry_count == 0
    assert "no meaningful production change" in (
        result.child_workflows["backend"].retry_refusal_reason or ""
    )


class AdvisoryCarryingExecutor:
    """Fail the backend once with a blocking and a non-blocking finding, then record the retry."""

    def __init__(self) -> None:
        self.retry_strategies: list[dict[str, Any]] = []

    async def run(self, **kwargs: Any) -> ChildExecution:
        repository_id = kwargs["repository"].repository_id
        child = kwargs["child"]
        common = {
            "feature": kwargs["feature"],
            "repository": kwargs["repository"],
            "workstream": kwargs["workstream"],
            "child": child,
            "code_completion": None,
            "review": None,
        }
        if repository_id != "backend":
            return ChildExecution(
                result=_child_result(**common, status="approved", blocking_issues=[])
            )
        if child.retry_count == 0:
            failed = _child_result(
                **common,
                status="failed",
                blocking_issues=["The route guard and the handler disagree on who may export."],
                advisory_findings=["The service test proves one retrieval page, not several."],
                current_revision="backend-rev-0",
            )
            return ChildExecution(
                result=failed.model_copy(
                    update={
                        "pull_request_readiness": False,
                        "failure_classification": (
                            FailureClassification.REVIEW_SCOPE_FAILURE.value
                        ),
                        "production_files_changed": ["server/controllers/adunit.controller.js"],
                        "production_diff_fingerprint": "fingerprint-backend-rev-0",
                    }
                )
            )
        self.retry_strategies.append(
            child.retry_strategy if isinstance(child.retry_strategy, dict) else {}
        )
        return ChildExecution(result=_child_result(**common, status="approved", blocking_issues=[]))


@pytest.mark.asyncio
async def test_the_retry_is_handed_the_findings_its_review_did_not_block_on() -> None:
    """The wiring, not the plan builder: what the result carried has to reach the next attempt.

    `_resolve_review_outcome` has always sorted a review's findings into blocking and
    advisory, and the advisory half has always been recorded on the result artifact. It just
    never went anywhere -- the retry read `blocking_issues` alone. So on AB-Feature-215 the
    platform held the medium finding, wrote it down, showed it to nobody, and blocked on it
    one attempt later.
    """
    request = StartFeatureRequest.model_validate(_feature_payload())
    state = _initial_feature_state("advisory-carry", request)
    executor = AdvisoryCarryingExecutor()
    orchestrator = FeatureWorkflowOrchestrator(child_executor=executor)

    await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert executor.retry_strategies, "the backend was never retried"
    assert executor.retry_strategies[0].get("advisory_findings") == [
        "The service test proves one retrieval page, not several."
    ]


class ControlledChildExecutor:
    """Block the backend on repository setup while the frontend completes normally."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(self, **kwargs: Any) -> ChildExecution:
        repository_id = kwargs["repository"].repository_id
        self.calls.append(repository_id)
        child = kwargs["child"]
        if repository_id != "backend":
            return ChildExecution(
                result=_child_result(
                    feature=kwargs["feature"],
                    repository=kwargs["repository"],
                    workstream=kwargs["workstream"],
                    child=child,
                    code_completion=None,
                    review=None,
                    status="approved",
                    blocking_issues=[],
                )
            )
        preflight = RepositoryPreflightResult(
            repository_id="backend",
            revision="revision-a",
            package_manager="npm",
            dependency_install_command=["npm", "ci"],
            dependency_install_status="failed",
            configured_scripts={"lint": "eslint ."},
            source_directories=["src"],
            test_directories=[],
            detected_entrypoints=[],
            detected_frameworks=["Express"],
            validation_readiness="blocked",
            blocking_issues=[],
            warnings=[],
        )
        result = _child_result(
            feature=kwargs["feature"],
            repository=kwargs["repository"],
            workstream=kwargs["workstream"],
            child=child,
            code_completion=None,
            review=None,
            status="failed",
            blocking_issues=["npm ci failed"],
        ).model_copy(
            update={
                "preflight_result": preflight.model_dump(mode="json"),
                "failure_classification": "dependency_installation_failure",
                "metadata": {
                    "child_retry_count": child.retry_count,
                    "repository_setup_retry_count": 1,
                },
            }
        )
        return ChildExecution(result=result)


class RepairedAfterGrantExecutor(ControlledChildExecutor):
    """Fail the backend until somebody grants an attempt, then let it pass.

    This is the case a grant exists for: the repository could not install its dependencies,
    a person fixed that outside the platform, and the next attempt has a reason to behave
    differently from the ones the platform refused to repeat.
    """

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Approve the backend once its child carries a grant."""
        child = kwargs["child"]
        if kwargs["repository"].repository_id == "backend" and child.granted_extra_attempts:
            self.calls.append("backend")
            return ChildExecution(
                result=_child_result(
                    feature=kwargs["feature"],
                    repository=kwargs["repository"],
                    workstream=kwargs["workstream"],
                    child=child,
                    code_completion=None,
                    review=None,
                    status="approved",
                    blocking_issues=[],
                )
            )
        return await super().run(**kwargs)


class OnlyTestChangesExecutor:
    """Return a test-only backend result to exercise bounded no-progress retries."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = defaultdict(int)
        self.retry_strategies: dict[str, list[dict[str, Any]]] = defaultdict(list)

    async def run(self, **kwargs: Any) -> ChildExecution:
        repository_id = kwargs["repository"].repository_id
        self.calls[repository_id] += 1
        child = kwargs["child"]
        self.retry_strategies[repository_id].append(child.retry_strategy or {})
        code_completion = _completion(
            [FileChange(path="tests/status.test.js", change_type="modified", description="test")]
        )
        result = _child_result(
            feature=kwargs["feature"],
            repository=kwargs["repository"],
            workstream=kwargs["workstream"],
            child=child,
            code_completion=code_completion,
            review=None,
            status="failed",
            blocking_issues=["Only tests changed."],
        )
        return ChildExecution(result=result, code_completion=code_completion)


def _backend_workstream() -> RepositoryWorkstreamPlan:
    return RepositoryWorkstreamPlan.model_validate(
        {
            "workstream_id": "backend",
            "repository_id": "backend",
            "role": "backend",
            "requirement_ids": ["status-api"],
            "scoped_requirements": [
                {
                    "requirement_id": "status-api",
                    "acceptance_criterion_ids": ["status-api:ac-1"],
                    "responsibility": "implements",
                }
            ],
            "out_of_scope_requirements": [],
            "shared_requirements": [],
            "responsibilities": ["Implement status API."],
            "task_ids": ["status-api-task"],
            "dependency_workstream_ids": [],
            "contract_sections_consumed": [],
            "contract_sections_implemented": [],
            "acceptance_criteria": ["Status API works."],
            "test_requirements": ["Run tests."],
            "documentation_requirements": [],
            "expected_files_or_areas": ["src/routes", "src/controllers", "src/services"],
            "required": True,
            "implementation_expectations": [
                {
                    "requirement_id": "status-api",
                    "expected_change_categories": ["route", "controller", "service", "test"],
                    "expected_source_areas": [
                        "src/routes",
                        "src/controllers",
                        "src/services",
                    ],
                    "tests_required": True,
                }
            ],
        }
    )


def _completion(changes: list[FileChange], *, implemented: bool = False) -> CodeCompletionArtifact:
    evidence = []
    if implemented:
        evidence = [
            RequirementImplementationEvidence(
                requirement_id="status-api",
                files=["src/routes/status.js", "src/controllers/status.js"],
                symbols=["status"],
                description="Status API implementation.",
                validation_results=[],
            )
        ]
    return create_artifact(
        CodeCompletionArtifact,
        workflow_id="feature:backend",
        artifact_id="006_code_completion.json",
        producer="engineer",
        metadata={},
        payload={
            "completion_status": "completed",
            "summary": "Changed repository files.",
            "file_changes": [item.model_dump(mode="json") for item in changes],
            "validation_results": [],
            "test_coverage_percent": None,
            "remaining_work": [],
            "commit_sha": None,
            "requirements_implemented": ["status-api"] if implemented else [],
            "requirements_not_implemented": [] if implemented else ["status-api"],
            "requirement_implementation_evidence": [
                item.model_dump(mode="json") for item in evidence
            ],
        },
    )


def _feature_payload() -> dict[str, object]:
    return {
        "prd": {
            "title": "Repository preflight",
            "problem_statement": "Repository bootstrap must block unsafe coding.",
            "goals": ["Stop setup retries safely."],
            "user_stories": [
                {
                    "story_id": "preflight",
                    "persona": "operator",
                    "need": "See setup failures",
                    "benefit": "Avoid duplicate retries",
                    "acceptance_criteria": ["Setup failure is exposed."],
                }
            ],
            "requirements": [
                {
                    "requirement_id": "status-api",
                    "description": "Implement one API status route.",
                    "priority": "must",
                    "acceptance_criteria": ["The status route responds."],
                    "dependencies": [],
                }
            ],
            "constraints": [],
            "out_of_scope": [],
            "stakeholders": ["platform"],
        },
        "repositories": [
            {
                "repository_id": "backend",
                "name": "Backend",
                "role": "backend",
                "repository_url": "https://github.com/example/backend.git",
                "default_branch": "main",
                "implementation_order": 0,
            },
            {
                "repository_id": "frontend",
                "name": "Frontend",
                "role": "frontend",
                "repository_url": "https://github.com/example/frontend.git",
                "default_branch": "main",
                "implementation_order": 1,
            },
        ],
    }


def test_a_full_volume_is_not_reported_as_a_repository_defect(tmp_path: Path) -> None:
    """Feature -038 blamed two healthy repositories for a full disk.

    Both installs failed with a bare package-manager exit code, and the workstreams were
    recorded as having a broken dependency configuration when nothing was wrong with them.
    """
    classification = classify_lint_failure(
        result_stdout="npm ERR! nospc ENOSPC: no space left on device, write",
        result_stderr="",
        repository_root=tmp_path,
    )

    assert classification.kind == "unsupported_runtime"
    assert "volume is full" in classification.description
    assert classification.disposition is RepositoryHealthDisposition.REQUIRES_HUMAN


def test_rewriting_the_same_files_is_not_read_as_the_previous_change(tmp_path: Path) -> None:
    """The same paths with different bytes must fingerprint differently.

    The fallback hashed the sorted path list, so answering a review finding in place produced
    the previous attempt's fingerprint. -067's frontend was ended by the identical-resubmission
    rule while it was one missing import from passing lint: attempt-3 failed on
    `no-unnecessary-act` in a test file and attempt-4 on `'ServerStatus' is not defined` in the
    router, demonstrably different source, identical path set.
    """
    workspace = tmp_path / "repo"
    (workspace / "src").mkdir(parents=True)
    paths = ["src/page.js", "src/router.js"]
    (workspace / "src/page.js").write_text("export const Page = () => null;\n", encoding="utf-8")
    (workspace / "src/router.js").write_text("const routes = [];\n", encoding="utf-8")

    first = _reviewed_source_fingerprint(workspace, paths, "path-only-fallback")

    # Exactly the edit a retry makes when it acts on a review finding: same files, new bytes.
    (workspace / "src/router.js").write_text(
        "import { Page } from './page';\nconst routes = [Page];\n", encoding="utf-8"
    )
    second = _reviewed_source_fingerprint(workspace, paths, "path-only-fallback")

    assert first != second
    assert second == _reviewed_source_fingerprint(workspace, paths, "path-only-fallback")


def test_an_unchanged_workspace_still_fingerprints_as_the_previous_change(tmp_path: Path) -> None:
    """Byte-identical files must still repeat, so the resubmission rule keeps working."""
    workspace = tmp_path / "repo"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src/page.js").write_text("export const Page = () => null;\n", encoding="utf-8")

    first = _reviewed_source_fingerprint(workspace, ["src/page.js"], "unused-fallback")
    second = _reviewed_source_fingerprint(workspace, ["src/page.js"], "unused-fallback")

    assert first == second


def test_a_change_with_no_production_files_keeps_the_reported_fingerprint(tmp_path: Path) -> None:
    """A tests-only attempt has no production bytes to hash, so the caller's value stands."""
    assert _reviewed_source_fingerprint(tmp_path, [], "reported-value") == "reported-value"


def test_rewriting_only_tests_answers_a_test_finding_and_counts_as_progress() -> None:
    """A review that rejects an attempt over its tests is answered by editing those tests.

    -068's backend ended holding two test-scoped findings -- one asking the mounted-endpoint
    test to exercise the configured capacity -- after it rewrote that test to drive a real
    request through the assembled application. Production bytes were correctly unchanged, and
    asking only about production read the attempt as a repeat of the one before it.
    """
    completion = _completion(
        [
            FileChange(path="src/routes/status.js", change_type="added", description="route"),
            FileChange(path="src/index.js", change_type="modified", description="register"),
            FileChange(path="test/status.test.js", change_type="added", description="test"),
        ],
        implemented=True,
    ).model_copy(update={"production_diff_fingerprint": "unchanged-production"})
    result = validate_implementation_completeness(_backend_workstream(), completion).model_copy(
        update={"test_diff_fingerprint": "rewritten-tests"}
    )

    meaningful, reason = meaningful_progress(
        result,
        previous_fingerprint="unchanged-production",
        previous_test_fingerprint="original-tests",
    )

    assert meaningful
    assert reason == "new_test_source_for_unchanged_production_implementation"


def test_resending_identical_production_and_test_source_is_still_not_progress() -> None:
    """The resubmission rule must still stop an attempt that changed nothing at all."""
    completion = _completion(
        [
            FileChange(path="src/routes/status.js", change_type="added", description="route"),
            FileChange(path="src/index.js", change_type="modified", description="register"),
            FileChange(path="test/status.test.js", change_type="added", description="test"),
        ],
        implemented=True,
    ).model_copy(update={"production_diff_fingerprint": "unchanged-production"})
    result = validate_implementation_completeness(_backend_workstream(), completion).model_copy(
        update={"test_diff_fingerprint": "unchanged-tests"}
    )

    meaningful, reason = meaningful_progress(
        result,
        previous_fingerprint="unchanged-production",
        previous_test_fingerprint="unchanged-tests",
    )

    assert not meaningful
    assert reason == "attempt_reproduced_the_previous_production_change"


@pytest.mark.parametrize(
    ("referrer", "registers"),
    [
        ("src/utils/routeUtils.js", True),
        ("src/router/PrivateRoutes.js", True),
        ("src/routes/index.js", True),
        ("src/App.js", True),
        ("src/components/Sidebar.js", True),
        ("src/apiUtils/activities.apiUtils.js", False),
        ("src/const.js", False),
        ("src/utils/roleUtils.js", False),
        ("server/services/health.service.js", False),
        # A leaf modal is not a registry. "app" identifies `App.js`, the React root; matching
        # it as one camel-case fragment made every file naming an app a registry, which in an
        # ad-management console is nearly all of them. AB-Feature-112's console was told
        # twelve times to mount bulk deletion in this modal and spent every cycle on it.
        ("src/components/apps/AddAppModal.js", False),
        ("src/components/apps/AppConfigModal.js", False),
        # Nor is the page that will consume it -- which is correct, and is why the gate must
        # say it found no registry rather than naming its best guess as one.
        ("src/pages/AllApps.js", False),
        # Still recognised: the job-describing names, and the directories a repository uses.
        ("src/utils/routeUtils.js", True),
        ("app/layout.tsx", True),
    ],
)
def test_only_a_module_registry_is_offered_as_the_wiring_example(
    referrer: str, registers: bool
) -> None:
    """The hint must point at where a module is mounted, not at one that consumes it.

    -069's console was told a page is wired in by `src/apiUtils/activities.apiUtils.js`,
    because that referrer came first. It rendered the new page inside an unrelated one, the
    reviewer rejected it, the attempt reverted, and the same hint sent it back. The registry
    it needed, `src/utils/routeUtils.js`, was in the same list further down.
    """
    assert registers_modules(referrer) is registers


def test_registry_paths_fall_back_to_structural_evidence() -> None:
    """A repository whose conventions name no wiring path still gets its registries.

    `wiring_path` is optional and the model frequently omits it: -106's backend produced
    seven conventions and not one wiring path, for the repository that then failed the wiring
    gate on three separate attempts. Reconnaissance also derives the same files structurally,
    so the engineer is never left with nothing to read.
    """
    conventions = [
        RepositoryConvention(
            convention_id="convention-1",
            kind="service",
            description="Services are registered in the app service.",
            evidence_paths=["server/services/app/app.galaxy.service.js"],
            wiring_path=None,
        )
    ]
    artifact = create_artifact(
        RepositoryReconnaissanceArtifact,
        workflow_id="feature-1",
        artifact_id="015_repository_reconnaissance.backend.json",
        producer="repository_recon",
        metadata={
            "wiring_files": ["server/config/express.js", "server/services/app/app.service.js"]
        },
        payload={
            "feature_id": "feature-1",
            "repository_id": "backend",
            "repository_revision": "revision-1",
            "summary": "The backend as it actually is.",
            "source_areas": ["server"],
            "test_areas": ["server/tests"],
            "conventions": conventions,
            "shared_utilities": [],
            "contradicted_premises": [],
        },
    )
    request = StartFeatureRequest.model_validate(_feature_payload())
    feature = _initial_feature_state("registry-paths", request)
    feature.artifacts.append(artifact)

    assert _registry_paths(feature, "backend") == [
        "server/config/express.js",
        "server/services/app/app.service.js",
    ]
    # Another repository's reconnaissance must never supply this one's registries.
    assert _registry_paths(feature, "frontend") == []


class RecordsGrantFeedbackExecutor(ControlledChildExecutor):
    """Approve once granted, and record the feedback each granted attempt was given."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        super().__init__()
        self.granted_feedback: list[list[str]] = []

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Capture the feedback the engineer would receive, then approve."""
        child = kwargs["child"]
        if kwargs["repository"].repository_id == "backend" and child.granted_extra_attempts:
            self.calls.append("backend")
            self.granted_feedback.append(list(kwargs["feedback"]))
            return ChildExecution(
                result=_child_result(
                    feature=kwargs["feature"],
                    repository=kwargs["repository"],
                    workstream=kwargs["workstream"],
                    child=child,
                    code_completion=None,
                    review=None,
                    status="approved",
                    blocking_issues=[],
                )
            )
        return await super().run(**kwargs)


@pytest.mark.asyncio
async def test_a_grant_carries_the_fix_integration_review_asked_for() -> None:
    """AB-Feature-111's console was granted attempts and told nothing about what was wrong.

    Integration review had named one precise cross-repository defect -- the UI submitted
    `app_id` where the API validates and queries `_id`. The automatic remediation path
    forwards that finding; a grant carried only the sentence a person typed for the audit
    record. So the engineer re-read the requirement, found the work already implemented and
    published, changed nothing, and failed as though it were not converging.

    A grant buys attempts. It does not get to decide what the attempt is for.
    """
    executor = RecordsGrantFeedbackExecutor()
    orchestrator, stopped = await _stopped_backend(executor)
    stopped.artifacts.append(
        create_artifact(
            IntegrationReviewArtifact,
            workflow_id=stopped.feature_id,
            artifact_id="012_integration_review.attempt-1.json",
            producer="integration_reviewer",
            payload={
                "feature_id": stopped.feature_id,
                "contract_artifact_id": "009_integration_contract.json",
                "review_status": "changes_requested",
                "repository_results": [],
                "contract_checks": ["contract conformance was checked"],
                "cross_repository_findings": [
                    {
                        "finding_id": "IR-1",
                        "severity": "high",
                        "responsible_repository_id": "backend",
                        "affected_repository_ids": ["backend"],
                        "contract_reference": "POST /apps/bulk-delete",
                        "description": "The console submits app_id where the API expects _id.",
                        "evidence": "The consumer sends app_id; the provider queries _id.",
                        "recommended_fix": "Submit the persisted _id rather than app_id.",
                    }
                ],
                "compatibility_assessment": "One identifier mismatch across the seam.",
                "security_assessment": "not_checked",
                "deployment_assessment": "not_checked",
                "merge_order": ["backend"],
                "required_fixes": ["Submit the persisted _id rather than app_id."],
            },
            metadata={},
        )
    )

    await orchestrator.retry_workstream(
        stopped,
        repository_id="backend",
        additional_attempts=1,
        requested_by="akhilesh",
        reason="Granting an attempt so the identifier fix can land.",
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
    )

    assert executor.granted_feedback, "the granted attempt must have run"
    assert any("persisted _id" in item for item in executor.granted_feedback[-1]), (
        executor.granted_feedback
    )


@pytest.mark.asyncio
async def test_a_grant_with_no_outstanding_integration_finding_adds_no_feedback() -> None:
    """The common case must stay unchanged: most grants follow an ordinary stop."""
    executor = RecordsGrantFeedbackExecutor()
    orchestrator, stopped = await _stopped_backend(executor)

    await orchestrator.retry_workstream(
        stopped,
        repository_id="backend",
        additional_attempts=1,
        requested_by="akhilesh",
        reason="Installed the missing private registry token on the runner.",
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
    )

    # A grant already carries the repository's own previous findings; what must not appear
    # is an integration instruction, because integration review asked for nothing here.
    assert executor.granted_feedback
    assert not any("persisted _id" in item for item in executor.granted_feedback[-1]), (
        executor.granted_feedback
    )


@pytest.mark.asyncio
async def test_a_grant_is_refused_when_no_review_cycle_remains() -> None:
    """An override the platform cannot honour must cost nothing.

    A grant raises this repository's classification budget. It does not raise the
    review-cycle ceiling, and `decide_child_retry` checks that too -- so a grant at the
    ceiling was accepted, spent a real clone, install and coding call, and was then refused
    before a second attempt. AB-Feature-111's console was at 12 of 12 and accepted three
    more, one of which ran.

    Refusing before anything is spent is this function's stated purpose.
    """
    executor = RepairedAfterGrantExecutor()
    orchestrator, stopped = await _stopped_backend(executor)
    stopped.max_child_review_cycles = 2
    stopped.child_workflows["backend"] = stopped.child_workflows["backend"].model_copy(
        update={"retry_count": 1}
    )
    attempts_before = executor.calls.count("backend")

    with pytest.raises(FeatureWorkflowError, match="review cycles"):
        await orchestrator.retry_workstream(
            stopped,
            repository_id="backend",
            additional_attempts=3,
            requested_by="akhilesh",
            reason="Buying attempts this feature cannot spend.",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )

    # Nothing ran, and the grant is not recorded as though it had been honoured.
    assert executor.calls.count("backend") == attempts_before
    assert stopped.child_workflows["backend"].granted_extra_attempts == 0
