"""Repository bootstrap checks performed before a coding executor is invoked.

The preflight is deliberately evidence based: it inspects checked-in manifests and
lockfiles, runs only deterministic dependency installation commands, and records
the result without treating a repository setup defect as an engineer mistake.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

import structlog
from pydantic import Field

from services.cancellation import CancellationRequested, CancellationToken, MockCancellationToken
from services.external_operations import ExternalOperationExecutor
from services.process_runner import (
    AsyncioProcessRunner,
    ProcessResult,
    ProcessRunner,
    repository_subprocess_environment,
    subprocess_result_code,
)
from state.external_operations import ExternalOperationType
from state.models import StateModel
from storage.external_operation_store import OperationResult, fingerprint_operation_input
from tools.file_tools import PathLike, resolve_workspace_root
from tools.node_toolchain import package_manager_argv
from tools.repository_layout import inspect_repository_layout
from tools.technology_detection import calculate_repository_revision, inspect_repository_technology

_LOGGER = structlog.get_logger(__name__)


class PreflightIssue(StateModel):
    """One actionable repository bootstrap observation."""

    issue_id: str = Field(min_length=1)
    category: Literal[
        "dependency_configuration",
        "missing_dependency",
        "invalid_lint_configuration",
        "missing_test_command",
        "missing_build_command",
        "source_structure",
        "repository_permissions",
        "package_manager_lockfile",
        "baseline_validation",
        "unsupported_runtime",
    ]
    severity: Literal["critical", "high", "medium", "low", "info"]
    description: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)
    automatically_repairable: bool
    # The packages a finding actually blames, kept as data rather than left inside the
    # sentence describing it. The classifier extracts these precisely; a reader downstream --
    # a repair proposal, for one -- would otherwise have to recover them from prose, and
    # scraping a sentence for package names is a guess dressed as a fact.
    unresolved_references: list[str] = Field(default_factory=list)
    undeclared_references: list[str] = Field(default_factory=list)


class RepositoryPreflightResult(StateModel):
    """Durable, credential-free repository readiness result."""

    repository_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    # ``package_manager`` is retained for existing API consumers.  The plural fields
    # describe mixed-language repositories without pretending one tool owns them all.
    package_manager: str | None = None
    package_managers: list[str] = Field(default_factory=list)
    dependency_install_command: list[str] | None = None
    dependency_install_commands: list[list[str]] = Field(default_factory=list)
    dependency_install_working_directories: list[str] = Field(default_factory=list)
    dependency_install_status: str = Field(min_length=1)
    # The digest of the lockfiles this checkout's installers resolve, so the NEXT attempt on
    # a preserved workspace can tell "nothing to install" from "not asked". `None` means the
    # question could not be answered, and an unanswerable digest always installs.
    dependency_lockfile_digest: str | None = None
    configured_scripts: dict[str, str] = Field(default_factory=dict)
    source_directories: list[str] = Field(default_factory=list)
    test_directories: list[str] = Field(default_factory=list)
    detected_entrypoints: list[str] = Field(default_factory=list)
    detected_frameworks: list[str] = Field(default_factory=list)
    validation_readiness: str = Field(min_length=1)
    blocking_issues: list[PreflightIssue] = Field(default_factory=list)
    warnings: list[PreflightIssue] = Field(default_factory=list)
    baseline_lint_configuration_failures: list[str] = Field(default_factory=list)
    baseline_validation_failures: list[str] = Field(default_factory=list)
    node_runtime_version: str | None = None
    package_manager_versions: dict[str, str] = Field(default_factory=dict)


class RepositoryHealthDisposition(StrEnum):
    """Who must act on a repository-health defect found before feature work starts."""

    AUTO_REPAIRABLE = "auto_repairable"
    REQUIRES_FEATURE_ENGINEER = "requires_feature_engineer"
    REQUIRES_HUMAN = "requires_human"
    UNSUPPORTED = "unsupported"


class _DependencyInstallationFailed(RuntimeError):
    """Carry a failed process result without persisting its raw output as an exception."""

    def __init__(self, result: ProcessResult) -> None:
        super().__init__("deterministic dependency installation exited unsuccessfully")
        self.result = result
        # Name the shape of the failure on the operation row, because that row is where the
        # failure is read. An operator asking why AB-Feature-203 died in dependency
        # installation found `install_dependencies_failed` and "external operation failed
        # before a confirmed result (_DependencyInstallationFailed)" -- true of a lockfile
        # conflict, a missing package and a budget overrun alike -- and had to open the child's
        # `preflight_result.blocking_issues` in a database session to learn which. A timeout
        # and a non-zero exit call for different actions, so they get different codes.
        #
        # Only the timeout is named. A non-zero exit keeps the code and the message it has
        # always had, so nothing that reads those rows has to learn a second spelling.
        if result.timed_out:
            self.operation_outcome = "timed_out"
            self.failure_classification = subprocess_result_code(
                return_code=result.return_code,
                timed_out=result.timed_out,
                cancelled=result.cancelled,
                prefix="dependency_installation",
            )


class LintFailureClassification(StateModel):
    """Explain whether a failed lint invocation actually reached source linting."""

    kind: Literal[
        "lint_violation",
        "lint_configuration_error",
        "missing_dependency",
        "unsupported_runtime",
    ]
    description: str = Field(min_length=1)
    # Which shared configs or plugins the linter could not resolve, and whether the
    # repository actually declares them. Declared-but-absent is a bootstrap problem the
    # platform may fix; undeclared is a checked-in defect only a human should decide on.
    unresolved_references: list[str] = Field(default_factory=list)
    undeclared_references: list[str] = Field(default_factory=list)
    disposition: RepositoryHealthDisposition = RepositoryHealthDisposition.REQUIRES_FEATURE_ENGINEER


class RepositoryPreflight:
    """Inspect and bootstrap one checked-out repository with bounded execution."""

    def __init__(
        self,
        *,
        default_timeout_seconds: float = 120.0,
        dependency_install_timeout_seconds: float = 600.0,
        process_runner: ProcessRunner | None = None,
        cancellation_token: CancellationToken | None = None,
        operation_executor: ExternalOperationExecutor | None = None,
    ) -> None:
        if default_timeout_seconds <= 0:
            msg = "default_timeout_seconds must be positive"
            raise ValueError(msg)
        if dependency_install_timeout_seconds <= 0:
            msg = "dependency_install_timeout_seconds must be positive"
            raise ValueError(msg)
        self._timeout = default_timeout_seconds
        # The install's own clock. Beside the shared one rather than derived from it, for the
        # same reason the version probe narrows to `min(self._timeout, 30.0)` at `_probe`: a
        # step that knows the shape of its own work should not inherit a budget sized for
        # something else. This one runs a package manager over the network; that one asks a
        # binary for its version string; the coding call in between is a model request. Three
        # jobs, three numbers.
        self._install_timeout = dependency_install_timeout_seconds
        self._process_runner = process_runner or AsyncioProcessRunner()
        self._cancellation_token = cancellation_token or MockCancellationToken()
        self._operation_executor = operation_executor

    async def run(
        self,
        repository_root: PathLike,
        *,
        repository_id: str | None = None,
        workspace_preserved: bool = False,
        previous_lockfile_digest: str | None = None,
    ) -> RepositoryPreflightResult:
        """Inspect and bootstrap only dependency tools evidenced by this checkout.

        ``workspace_preserved`` and ``previous_lockfile_digest`` together decide whether the
        install has anything to do. AB-Feature-218 ran `install_dependencies` on all eleven
        of its attempt boundaries -- about 395 s -- against workspaces the platform itself
        had marked `attempt_inputs.workspace: "preserved"`, with the lockfiles untouched
        throughout. Installing into a tree that already holds exactly those dependencies is
        the same install, repeated.

        Fails OPEN, deliberately and in every direction: a fresh or reset workspace, a
        digest that could not be computed now, and a previous digest that is missing or
        different all install. The cost of a needless install is a few minutes; the cost of
        a skipped necessary one is an attempt that fails on a missing module, which is
        exactly the class of confusion this platform is built to avoid.
        """
        root = resolve_workspace_root(repository_root)
        identifier = repository_id or root.name
        package = _package_json(root)
        scripts = _scripts(package)
        profile = inspect_repository_technology(root, repository_id=identifier)
        revision = await calculate_repository_revision(root)
        issues: list[PreflightIssue] = []
        warnings: list[PreflightIssue] = []
        node_projects = _manifest_roots(root, "package.json")
        installers = _dependency_installers(root)
        install_command = installers[0][1] if installers else None
        install_status = "not_applicable"
        lockfile_digest = _lockfile_digest(root, installers)

        (
            runtime_issues,
            node_runtime_version,
            package_manager_versions,
        ) = await self._validate_node_runtime(root, node_projects)
        issues.extend(runtime_issues)

        missing_node_locks = [
            project for project in node_projects if _node_install_root(project, root) is None
        ]
        if missing_node_locks:
            for project in missing_node_locks:
                relative = project.relative_to(root).as_posix() or "."
                issues.append(
                    _issue(
                        "NODE_LOCKFILE_NOT_FOUND",
                        "package_manager_lockfile",
                        "high",
                        (
                            "A declared Node project has no supported lockfile in its workspace "
                            "ancestry."
                        ),
                        (
                            f"Detected package.json at {relative} without package-lock.json, "
                            "pnpm-lock.yaml, or yarn.lock in that project or its repository root."
                        ),
                        "Commit the project's package-manager lockfile before live execution.",
                        False,
                    )
                )
        # Severity, not the mere presence of a runtime observation. A toolchain difference the
        # worker can honour is reported as a warning, and a warning that still refused to
        # install would leave the feature exactly as stuck as the blocking issue it replaced.
        if any(issue.severity in {"critical", "high"} for issue in runtime_issues):
            install_status = "blocked_unsupported_runtime"
        elif missing_node_locks:
            install_status = "blocked_missing_lockfile"
        elif (
            installers
            and lockfile_digest is not None
            and workspace_preserved
            and (lockfile_digest == previous_lockfile_digest)
            and _installed_trees_present(installers)
        ):
            # The tree already holds exactly these dependencies and nothing has asked for
            # different ones. Recorded as its own status rather than as `installed`, so an
            # operator reading the record can tell a skip from a run that did nothing.
            install_status = "skipped_unchanged_lockfile"
            _LOGGER.info(
                "the dependency install was skipped: this workspace was preserved and its "
                "lockfiles are unchanged since the previous attempt "
                "[outcome=dependency_install_skipped repository=%s]",
                identifier,
            )
        elif installers:
            install_status = "installed"
            for _installer, command, install_root in installers:
                current_status, install_issue = await self._install(
                    install_root, identifier, command, revision=revision.combined_fingerprint
                )
                if install_issue is not None:
                    issues.append(install_issue)
                    install_status = current_status
                    break

        if package:
            if "test" not in scripts and "test:ci" not in scripts:
                warnings.append(
                    _issue(
                        "TEST_COMMAND_NOT_CONFIGURED",
                        "missing_test_command",
                        "medium",
                        "No repository-native test command is configured.",
                        "package.json has no test or test:ci script.",
                        (
                            "Configure an existing test framework or explicitly plan test "
                            "infrastructure."
                        ),
                        False,
                    )
                )
            if "build" not in scripts:
                warnings.append(
                    _issue(
                        "BUILD_COMMAND_NOT_CONFIGURED",
                        "missing_build_command",
                        "low",
                        "No repository-native build command is configured.",
                        "package.json has no build script.",
                        "Document why a build is unnecessary or add the checked-in build command.",
                        False,
                    )
                )

        layout = inspect_repository_layout(root)
        source_directories = list(layout.source_directories)
        test_directories = list(layout.test_directories)
        if profile.primary_language != "Unknown" and not source_directories:
            warnings.append(
                _issue(
                    "SOURCE_DIRECTORY_NOT_DETECTED",
                    "source_structure",
                    "low",
                    "No non-root source directory was detected.",
                    "The checkout has no source file beneath a non-test directory.",
                    "Use checkout evidence to confirm whether source files live at the root.",
                    False,
                )
            )
        if not root.exists() or not root.is_dir():
            issues.append(
                _issue(
                    "REPOSITORY_WORKSPACE_UNAVAILABLE",
                    "repository_permissions",
                    "critical",
                    "The checked-out repository workspace is unavailable.",
                    f"Workspace path is not a readable directory: {root}",
                    "Restore workspace access before retrying the child workstream.",
                    False,
                )
            )

        blocking = [issue for issue in issues if issue.severity in {"critical", "high"}]
        readiness = "blocked" if blocking else "ready_with_warnings" if warnings else "ready"
        return RepositoryPreflightResult(
            repository_id=identifier,
            revision=revision.combined_fingerprint,
            package_manager=installers[0][0] if installers else None,
            package_managers=sorted({name for name, _command, _cwd in installers}),
            dependency_install_command=install_command,
            dependency_install_commands=[command for _name, command, _cwd in installers],
            dependency_install_working_directories=[
                install_root.relative_to(root).as_posix() or "."
                for _name, _command, install_root in installers
            ],
            dependency_install_status=install_status,
            dependency_lockfile_digest=lockfile_digest,
            configured_scripts=scripts,
            source_directories=source_directories,
            test_directories=test_directories,
            detected_entrypoints=_entrypoints(root, package),
            detected_frameworks=profile.frameworks,
            validation_readiness=readiness,
            blocking_issues=blocking,
            warnings=[*warnings, *[item for item in issues if item not in blocking]],
            node_runtime_version=node_runtime_version,
            package_manager_versions=package_manager_versions,
        )

    async def _validate_node_runtime(
        self, root: Path, node_projects: list[Path]
    ) -> tuple[list[PreflightIssue], str | None, dict[str, str]]:
        """Verify repository runtime declarations against the binaries that will execute them.

        A manager family is not enough evidence: invoking global Yarn 1 for a checkout that
        declares Yarn 4 can install a different dependency graph or reject the lockfile.  The
        same applies to a Node engine range.  Probe the sanitized worker PATH and stop before
        installation when its concrete versions cannot satisfy checked-in declarations.
        """
        issues: list[PreflightIssue] = []
        node_version: str | None = None
        manager_versions: dict[str, str] = {}

        engine_requirements: list[tuple[Path, str]] = []
        manager_declarations: dict[Path, str] = {}
        for project in node_projects:
            package = _package_json(project)
            engines = package.get("engines")
            if isinstance(engines, dict) and "node" in engines:
                requirement = engines["node"]
                if isinstance(requirement, str) and requirement.strip():
                    engine_requirements.append((project, requirement.strip()))
                else:
                    issues.append(
                        _runtime_issue(
                            "NODE_ENGINE_DECLARATION_INVALID",
                            project,
                            root,
                            "The package's engines.node declaration is not a non-empty string.",
                            "Declare engines.node as a supported semantic-version range.",
                        )
                    )
            declared = package.get("packageManager")
            if declared is not None:
                if isinstance(declared, str) and declared.strip():
                    manager_declarations[project] = declared.strip()
                else:
                    issues.append(
                        _runtime_issue(
                            "PACKAGE_MANAGER_DECLARATION_INVALID",
                            project,
                            root,
                            "The packageManager declaration is not a non-empty string.",
                            "Declare packageManager as an exact supported name@version value.",
                        )
                    )

        if engine_requirements:
            node_version, probe_issue = await self._probe_runtime_version(
                root, ["node", "--version"], runtime_name="Node.js"
            )
            if probe_issue is not None:
                issues.append(probe_issue)
            elif node_version is not None:
                for project, requirement in engine_requirements:
                    satisfies = _node_version_satisfies(node_version, requirement)
                    if satisfies is None:
                        issues.append(
                            _runtime_issue(
                                "NODE_ENGINE_RANGE_UNSUPPORTED",
                                project,
                                root,
                                (
                                    f"engines.node declares `{requirement}`, which the platform "
                                    "cannot evaluate safely."
                                ),
                                "Use a standard npm semantic-version range for engines.node.",
                            )
                        )
                    elif not satisfies:
                        issues.append(
                            _runtime_issue(
                                "NODE_RUNTIME_VERSION_UNSUPPORTED",
                                project,
                                root,
                                (
                                    f"engines.node requires `{requirement}`, but the worker "
                                    f"runtime is Node.js {node_version}."
                                ),
                                (
                                    "Run this repository on a worker image whose Node.js "
                                    "version satisfies engines.node."
                                ),
                            )
                        )

        parsed_declarations: dict[Path, tuple[str, str]] = {}
        for project, declaration in manager_declarations.items():
            parsed = _parse_package_manager_declaration(declaration)
            if parsed is None:
                issues.append(
                    _runtime_issue(
                        "PACKAGE_MANAGER_DECLARATION_INVALID",
                        project,
                        root,
                        (
                            f"packageManager declares `{declaration}`, which is not a supported "
                            "exact npm, pnpm, or Yarn version."
                        ),
                        "Declare packageManager as npm@x.y.z, pnpm@x.y.z, or yarn@x.y.z.",
                    )
                )
                continue
            parsed_declarations[project] = parsed

        for project in node_projects:
            inherited = _nearest_package_manager_declaration(project, root)
            selected = _node_install_root(project, root)
            if inherited is None or selected is None:
                continue
            declaration_root, declaration = inherited
            parsed = parsed_declarations.get(declaration_root)
            if parsed is None:
                continue
            declared_manager, _declared_version = parsed
            lock_manager, lock_root = selected
            if declared_manager != lock_manager:
                relative_lock = lock_root.relative_to(root).as_posix() or "."
                issues.append(
                    _runtime_issue(
                        "PACKAGE_MANAGER_LOCKFILE_MISMATCH",
                        project,
                        root,
                        (
                            f"packageManager selects `{declared_manager}`, but the nearest "
                            f"lockfile at `{relative_lock}` selects `{lock_manager}`."
                        ),
                        "Align packageManager with the committed lockfile before execution.",
                    )
                )

        declarations_by_manager: dict[str, list[tuple[Path, str]]] = {}
        for project, (manager, version) in parsed_declarations.items():
            declarations_by_manager.setdefault(manager, []).append((project, version))
        for manager, declarations in sorted(declarations_by_manager.items()):
            actual_version, probe_issue = await self._probe_runtime_version(
                root, [manager, "--version"], runtime_name=manager
            )
            if probe_issue is not None:
                issues.append(probe_issue)
                continue
            if actual_version is None:
                continue
            manager_versions[manager] = actual_version
            for project, declared_version in declarations:
                # Whether this project's commands will actually run through Corepack. Asked of
                # the same function that builds those commands, so the thing verified here is
                # the thing that runs -- checking the ambient version instead would verify a
                # binary no command was going to invoke.
                argv, _resolved = package_manager_argv(manager, project, root)
                if argv[0] == "corepack":
                    # Corepack provisions the exact declared version, which is what the
                    # declaration asks for and what makes two repositories pinning different
                    # versions both work. Probed rather than assumed, before the install
                    # depends on it, and the probe warms the cache the install then reuses.
                    provisioned, _ = await self._probe_runtime_version(
                        project,
                        [*argv, "--version"],
                        runtime_name=f"corepack {manager}",
                    )
                    if provisioned is not None and _normalized_exact_version(
                        provisioned
                    ) == _normalized_exact_version(declared_version):
                        manager_versions[f"{manager} (declared)"] = provisioned
                        continue
                    issues.append(
                        _runtime_issue(
                            "PACKAGE_MANAGER_PROVISION_FAILED",
                            project,
                            root,
                            (
                                f"packageManager declares `{manager}@{declared_version}` and "
                                "Corepack could not provision it on this worker."
                            ),
                            (
                                "Confirm the declared version exists in the registry and that "
                                "the worker can reach it, or declare a version that does."
                            ),
                        )
                    )
                    continue
                # No Corepack on this worker, so the image's manager is what will run and the
                # question becomes whether it can honour the committed lockfile.
                if _normalized_exact_version(actual_version) == _normalized_exact_version(
                    declared_version
                ):
                    continue
                selected = _node_install_root(project, root)
                lockfile_version = (
                    _committed_lockfile_version(selected[1]) if selected is not None else None
                )
                if _package_manager_honours_lockfile(
                    manager, declared_version, actual_version, lockfile_version
                ):
                    # Not a repository defect. `packageManager` is a Corepack provisioning
                    # directive -- "fetch this version" -- and a platform that does not enable
                    # Corepack is reporting its own choice as the repository's fault. The
                    # demand was also unsatisfiable as written: its recommended action was to
                    # use a worker image carrying the exact declared version, which no single
                    # image can do for two repositories that pin different versions.
                    #
                    # What determines whether the install is reproducible is the committed
                    # lockfile, not the version of the binary reading it. Where that can be
                    # shown, the difference is worth saying and not worth stopping for.
                    issues.append(
                        _issue(
                            "PACKAGE_MANAGER_VERSION_DIFFERS",
                            "unsupported_runtime",
                            "low",
                            "The worker's package manager is not the declared version.",
                            (
                                f"package.json at "
                                f"`{project.relative_to(root).as_posix() or '.'}`: "
                                f"packageManager declares `{manager}@{declared_version}` and "
                                f"the worker provides {manager} {actual_version}, which reads "
                                f"the committed lockfile format"
                                + (
                                    f" (lockfileVersion {lockfile_version})."
                                    if lockfile_version is not None
                                    else "."
                                )
                            ),
                            (
                                "No action required for this run. Enable Corepack on the worker "
                                "if the declared version must be used exactly."
                            ),
                            False,
                        )
                    )
                    continue
                issues.append(
                    _runtime_issue(
                        "PACKAGE_MANAGER_VERSION_UNSUPPORTED",
                        project,
                        root,
                        (
                            f"packageManager requires `{manager}@{declared_version}`, but "
                            f"the worker provides {manager} {actual_version}, which cannot be "
                            "shown to read the committed lockfile format"
                            + (
                                f" (lockfileVersion {lockfile_version})."
                                if lockfile_version is not None
                                else "."
                            )
                        ),
                        (
                            "Align the declared package-manager major version with the "
                            "committed lockfile, or enable Corepack on the worker so the "
                            "declared version is provisioned."
                        ),
                    )
                )
        return _deduplicate_issues(issues), node_version, manager_versions

    async def _probe_runtime_version(
        self, root: Path, command: list[str], *, runtime_name: str
    ) -> tuple[str | None, PreflightIssue | None]:
        """Read one tool version without persisting repository-controlled process output.

        A probe that times out is retried, because the answer it failed to give is not
        evidence about the repository. Feature run-2's backend was declared to have an
        invalid checked-in setup -- a terminal classification, so no coding attempt was
        ever made -- when `node --version` did not answer within the timeout on a loaded
        worker. Anything other than a timeout is a real answer and is trusted immediately:
        a missing binary should still fail on its first attempt.
        """
        result = None
        for attempt in range(_VERSION_PROBE_ATTEMPTS):
            result = await self._process_runner.run(
                command,
                root,
                min(self._timeout, 30.0),
                self._cancellation_token,
                repository_subprocess_environment(),
            )
            if result.cancelled:
                raise CancellationRequested(f"{runtime_name} version probe was cancelled")
            version = _extract_runtime_version(result.stdout) if result.succeeded else None
            if version is not None:
                return version, None
            if not result.timed_out or attempt == _VERSION_PROBE_ATTEMPTS - 1:
                break
        assert result is not None
        code = subprocess_result_code(
            return_code=result.return_code,
            timed_out=result.timed_out,
            cancelled=result.cancelled,
            prefix="runtime_version_probe",
        )
        return None, _issue(
            "RUNTIME_VERSION_UNAVAILABLE",
            "unsupported_runtime",
            "high",
            f"The worker could not verify its {runtime_name} version.",
            f"The fixed `{command[0]} --version` probe returned {code}.",
            f"Install a supported {runtime_name} binary on the worker before execution.",
            False,
        )

    async def _install(
        self, root: Path, repository_id: str, command: list[str], *, revision: str
    ) -> tuple[str, PreflightIssue | None]:
        """Journal an installation attempt, but retain its process result as evidence."""
        safe_input = {
            "workspace_path": str(root),
            "repository_id": repository_id,
            "repository_revision": revision,
            "command": command,
            "command_fingerprint": fingerprint_operation_input({"command": command}),
            # Version the idempotency contract so pre-fix rows that cached a nonzero
            # process result as SUCCEEDED can never be replayed as authoritative.
            "result_semantics": "exit_zero_required_v2",
        }

        async def action() -> tuple[ProcessResult, OperationResult]:
            result, absorbed = await self._install_with_retries(
                root, command, repository_id=repository_id
            )
            if result.cancelled:
                raise CancellationRequested("dependency installation was cancelled")
            if not result.succeeded:
                raise _DependencyInstallationFailed(result)
            return result, OperationResult(
                payload={
                    "return_code": result.return_code,
                    "timed_out": result.timed_out,
                    "cancelled": result.cancelled,
                    "stdout_summary": "",
                    "stderr_summary": "",
                    # How many registry blips this one operation absorbed. Zero for almost
                    # every install; without it, an operation that took three minutes and one
                    # that took twenty seconds are the same row.
                    "transient_faults_absorbed": absorbed,
                    "result_code": subprocess_result_code(
                        return_code=result.return_code,
                        timed_out=result.timed_out,
                        cancelled=result.cancelled,
                        prefix="dependency_installation",
                    ),
                }
            )

        if self._operation_executor is None:
            result, _ = await self._install_with_retries(root, command, repository_id=repository_id)
        else:
            try:
                journaled = await self._operation_executor.run(
                    operation_type=ExternalOperationType.INSTALL_DEPENDENCIES,
                    logical_step="repository_preflight:dependency_install",
                    safe_input=safe_input,
                    idempotency_input=safe_input,
                    action=action,
                    max_attempts=2,
                )
                if journaled.reused:
                    payload = journaled.operation.result_payload or {}
                    result = ProcessResult(
                        command=tuple(command),
                        return_code=_int_or_none(payload.get("return_code")),
                        stdout="",
                        stderr=str(payload.get("stderr_summary", "")),
                        duration_seconds=0.0,
                        timed_out=bool(payload.get("timed_out", False)),
                        cancelled=bool(payload.get("cancelled", False)),
                    )
                else:
                    result = journaled.value
            except _DependencyInstallationFailed as error:
                result = error.result
        if result.cancelled:
            raise CancellationRequested("dependency installation was cancelled")
        if result.succeeded:
            return "installed", None
        return (
            "failed",
            _issue(
                "DEPENDENCY_INSTALLATION_FAILED",
                "dependency_configuration",
                "high",
                self._install_failure_description(result),
                _process_evidence(command, result),
                self._install_failure_action(result),
                False,
            ),
        )

    def _install_failure_description(self, result: ProcessResult) -> str:
        """Say which kind of failure this was in the first sentence a person reads."""
        if not result.timed_out:
            return "Deterministic dependency installation failed before coding began."
        return "Deterministic dependency installation exceeded its budget before coding began."

    def _install_failure_action(self, result: ProcessResult) -> str:
        """Recommend the action this failure actually calls for, with its numbers in it.

        A timeout and a non-zero exit want opposite things. The standing advice -- resolve the
        lockfile or the registry -- was actively wrong for AB-Feature-203: the lockfiles were
        valid and the registry answered 200 in 0.73s from inside the container while the
        failure was being diagnosed. What had happened was that a cold `npm ci` did not fit in
        a budget sized for a model call, and no sentence anywhere said so. Naming the budget
        and the measured elapsed time is what makes the next one readable without a database
        session; without the numbers the reader cannot tell a budget that is slightly too
        small from an install that will never finish.
        """
        if not result.timed_out:
            return (
                "Resolve the checked-in lockfile or registry/runtime issue, then start a "
                "fresh replacement workstream."
            )
        # Zero only where the result was reconstructed from a journal payload rather than
        # measured. Claiming "after 0s" of an install that ran for ten minutes would be worse
        # than saying nothing, so the clause is dropped instead of guessed.
        elapsed = f" after {result.duration_seconds:.0f}s" if result.duration_seconds > 0.0 else ""
        return (
            f"Dependency installation exceeded the {self._install_timeout:g}s dependency "
            f"install budget{elapsed}. A timeout does not implicate the lockfile or the "
            "registry: warm this host's package manager cache, reduce concurrent installs, or "
            "raise `dependency_install_timeout_seconds`, then start a fresh replacement "
            "workstream."
        )

    async def _install_with_retries(
        self, root: Path, command: list[str], *, repository_id: str
    ) -> tuple[ProcessResult, int]:
        """Run the install command, asking again when the evidence names the registry.

        Returns the last result and how many transient failures were absorbed reaching it.

        Inside one journal attempt rather than around it, and that is the whole design. The
        journal's attempt budget is what stops an *unconfirmed external effect* from being
        repeated blindly; an install is a local subprocess whose outcome is never in doubt, so
        spending that budget on weather would push the operation to `failed_terminal` early
        and make a later child attempt's install unreplayable. One operation, one confirmed
        answer, and the absorbed retries reported rather than hidden -- loud in the log and
        counted on the operation's own result payload.
        """
        faults = 0
        while True:
            result = await self._process_runner.run(
                command,
                root,
                self._install_timeout,
                self._cancellation_token,
                repository_subprocess_environment(),
            )
            if result.succeeded or result.cancelled:
                return result, faults
            if faults >= _ALLOWED_INSTALL_FAULTS or not install_failure_is_transient(result):
                return result, faults
            faults += 1
            backoff = _INSTALL_FAULT_BACKOFF_SECONDS * (2 ** (faults - 1))
            _LOGGER.warning(
                "dependency_install_transient_fault_retried",
                repository_id=repository_id,
                command=list(command),
                fault_count=faults,
                backoff_seconds=round(backoff, 3),
                result_code=subprocess_result_code(
                    return_code=result.return_code,
                    timed_out=result.timed_out,
                    cancelled=result.cancelled,
                    prefix="dependency_installation",
                ),
            )
            await asyncio.sleep(backoff)
            await self._cancellation_token.raise_if_cancelled()


def classify_lint_failure(
    *, result_stdout: str, result_stderr: str, repository_root: PathLike
) -> LintFailureClassification:
    """Classify a linter early exit without misreporting it as a source violation."""
    output = f"{result_stdout}\n{result_stderr}".lower()
    if any(
        token in output
        for token in ("unsupported engine", "unsupported node", "requires node", "node version")
    ):
        return LintFailureClassification(
            kind="unsupported_runtime",
            description="Lint could not run because the configured runtime is unsupported.",
        )
    config_markers = (
        "failed to load config",
        "couldn't find the config",
        "cannot find config",
        "eslint couldn't find",
        "failed to load plugin",
    )
    # A full volume surfaces as an opaque package-manager exit code. Naming it stops the
    # repository being blamed for a dependency configuration problem it does not have.
    if any(
        token in output for token in ("enospc", "no space left on device", "disk quota exceeded")
    ):
        return LintFailureClassification(
            kind="unsupported_runtime",
            description=(
                "The workspace volume is full, so the package manager could not write. This is "
                "a platform capacity problem, not a defect in this repository."
            ),
            disposition=RepositoryHealthDisposition.REQUIRES_HUMAN,
        )
    dependency_markers = (
        "cannot find module",
        "module not found",
        "no module named",
        "could not resolve",
        "failed to resolve",
    )
    if any(token in output for token in config_markers):
        references = _unresolved_references(f"{result_stdout}\n{result_stderr}")
        declared = _declared_dependency_names(resolve_workspace_root(repository_root))
        undeclared = [
            reference
            for reference in references
            if not (_candidate_package_names(reference) & declared)
        ]
        if any(token in output for token in dependency_markers):
            if references and not undeclared:
                # Declared in the manifest but absent from the checkout: a deterministic
                # reinstall is a safe, bounded repair.
                return LintFailureClassification(
                    kind="missing_dependency",
                    description=(
                        "Lint exited before source checks because a declared ESLint dependency "
                        "is not installed."
                    ),
                    unresolved_references=references,
                    disposition=RepositoryHealthDisposition.AUTO_REPAIRABLE,
                )
            # Referenced but never declared. Adding it would be the platform guessing at a
            # dependency change the repository never asked for.
            return LintFailureClassification(
                kind="missing_dependency",
                description=(
                    "Lint exited before source checks because its configuration extends a "
                    "shared config the repository does not declare as a dependency."
                ),
                unresolved_references=references,
                undeclared_references=undeclared,
                disposition=RepositoryHealthDisposition.REQUIRES_HUMAN,
            )
        return LintFailureClassification(
            kind="lint_configuration_error",
            description=(
                "Lint exited before source checks because its checked-in configuration is invalid."
            ),
            unresolved_references=references,
            undeclared_references=undeclared,
            disposition=RepositoryHealthDisposition.REQUIRES_HUMAN,
        )
    return LintFailureClassification(
        kind="lint_violation",
        description="Lint completed and reported source-code violations.",
        disposition=RepositoryHealthDisposition.REQUIRES_FEATURE_ENGINEER,
    )


# Real linters report the same problem quoted, unquoted, or backticked, so the reference
# is captured with optional surrounding quotes rather than assuming one output style.
# A version probe is cheap and idempotent, so a timeout is worth asking again before
# concluding anything about the worker's toolchain.
_VERSION_PROBE_ATTEMPTS = 3
# How many extra installs a registry blip is worth, and how long to wait between them.
#
# A failed install is charged to the repository-setup budget, which buys a whole fresh
# attempt: a new coding call against unchanged source. The batch that motivated this had
# three `dependency_installation_failure`s, and one of them killed a workstream that way --
# install flake, identical retry, refused for no meaningful change. The install itself is
# what should have been asked again, and asking it costs nothing but the wait.
#
# The wait starts where the clone-stage tail starts (`_NO_MODEL_COST_FAULT_BACKOFF_MULTIPLIER`
# in `workflows.feature_workflow`, four times its five-second base) and doubles the same way:
# 20 then 40 seconds. Same reasoning, same numbers, two stages -- kept as its own constant
# because this module sits below that one in the import graph.
_ALLOWED_INSTALL_FAULTS = 2
_INSTALL_FAULT_BACKOFF_SECONDS = 20.0
# Registry and network conditions, in the wording the package managers this platform runs
# actually use. Conservative on purpose, and the house rule from 54- Part 3 applies: when the
# evidence does not say one of these, the failure keeps today's behaviour and is charged to
# the setup budget. A checked-in lockfile conflict or a failing postinstall script must not
# buy itself two more identical installs, and a manager whose transient wording is not here
# yet is no worse off than it is today.
#
# `timed_out` is deliberately not transient. The install timeout is the sandbox's limit, and
# an install that does not fit inside it will not start fitting; retrying only spends the
# limit again before reaching the same verdict.
_TRANSIENT_INSTALL_MARKERS = (
    # npm, yarn, pnpm
    "etimedout",
    "econnreset",
    "econnrefused",
    "enotfound",
    "eai_again",
    "socket hang up",
    "network timeout",
    "err_socket_timeout",
    "registry error",
    # pip, uv, poetry
    "readtimeouterror",
    "max retries exceeded",
    "temporary failure in name resolution",
    "connection reset by peer",
    "connection aborted",
    "httpsconnectionpool",
    # go, cargo, bundler, and the shapes an HTTP registry answers with under load
    "i/o timeout",
    "tls handshake timeout",
    "server misbehaving",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway time-out",
    "504 gateway timeout",
    "429 too many requests",
    "rate limit",
    "temporarily unavailable",
    "service unavailable",
)
_UNRESOLVED_REFERENCE_PATTERN = re.compile(
    r"(?:cannot find module|module not found|no module named|failed to load config"
    r"|couldn't find the config|cannot find config|failed to load plugin|could not resolve"
    r"|failed to resolve)"
    r"[:\s]+[\"'`]?(@?[\w.][\w./@-]*)[\"'`]?",
    re.IGNORECASE,
)
_DEPENDENCY_MANIFEST_SECTIONS = (
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "optionalDependencies",
)


def _unresolved_references(output: str) -> list[str]:
    """Extract the shared configs or plugins a linter reported it could not resolve."""
    seen: dict[str, None] = {}
    for match in _UNRESOLVED_REFERENCE_PATTERN.finditer(output):
        reference = match.group(1).strip()
        if reference:
            seen.setdefault(reference, None)
    return list(seen)


def _candidate_package_names(reference: str) -> set[str]:
    """Map an ESLint config reference to every package name that could satisfy it.

    ESLint resolves ``extends: "airbnb-base"`` to the ``eslint-config-airbnb-base``
    package, so a declaration check on the raw reference alone reports false positives.
    """
    reference = reference.strip()
    if not reference:
        return set()
    if reference.startswith(("./", "../", "/", ".")):
        return set()
    names = {reference}
    if reference.startswith("plugin:"):
        plugin = reference.removeprefix("plugin:").split("/", 1)[0]
        names.update({plugin, f"eslint-plugin-{plugin}"})
        return names
    # A scoped reference such as @acme/eslint-config resolves as written.
    if not reference.startswith("@") and not reference.startswith("eslint-config-"):
        names.add(f"eslint-config-{reference}")
    # Only the package itself is declared; a deep import path is not.
    names.add(reference.split("/", 1)[0])
    return names


def _declared_dependency_names(root: Path) -> set[str]:
    """Collect every dependency name the repository declares across its manifests."""
    declared: set[str] = set()
    for project in _manifest_roots(root, "package.json"):
        package = _package_json(project)
        for section in _DEPENDENCY_MANIFEST_SECTIONS:
            value = package.get(section)
            if isinstance(value, dict):
                declared.update(key for key in value if isinstance(key, str))
    return declared


def _package_json(root: Path) -> dict[str, object]:
    try:
        value = json.loads((root / "package.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _scripts(package: dict[str, object]) -> dict[str, str]:
    scripts = package.get("scripts")
    if not isinstance(scripts, dict):
        return {}
    return {
        key: value
        for key, value in scripts.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def _node_package_manager(root: Path) -> str | None:
    if (root / "pnpm-lock.yaml").is_file():
        return "pnpm"
    if (root / "yarn.lock").is_file():
        return "yarn"
    if (root / "package-lock.json").is_file():
        return "npm"
    return None


def _nearest_package_manager_declaration(
    project: Path, repository_root: Path
) -> tuple[Path, str] | None:
    """Resolve Corepack's nearest packageManager declaration within the checkout."""
    current = project
    while True:
        declared = _package_json(current).get("packageManager")
        if isinstance(declared, str) and declared.strip():
            return current, declared.strip()
        if current == repository_root:
            return None
        current = current.parent


_EXACT_SEMANTIC_VERSION = re.compile(
    r"^v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_PACKAGE_MANAGER_DECLARATION = re.compile(
    r"^(npm|pnpm|yarn)@(.+)$",
    re.IGNORECASE,
)


def _normalized_exact_version(value: str) -> str | None:
    """Normalize one exact semantic version while ignoring only build metadata."""
    match = _EXACT_SEMANTIC_VERSION.fullmatch(value.strip())
    if match is None:
        return None
    major, minor, patch, prerelease = match.groups()
    normalized = f"{major}.{minor}.{patch}"
    return f"{normalized}-{prerelease}" if prerelease is not None else normalized


def _semantic_version(value: str) -> tuple[int, int, int] | None:
    """Return the numeric release tuple used for stable Node engine comparison."""
    normalized = _normalized_exact_version(value)
    # The range evaluator intentionally handles stable worker runtimes only. Treating a
    # prerelease as its eventual stable release would incorrectly satisfy ranges such as
    # ``>=23`` under npm's prerelease semantics.
    if normalized is None or "-" in normalized:
        return None
    release = normalized.partition("-")[0]
    major, minor, patch = release.split(".")
    return int(major), int(minor), int(patch)


def _extract_runtime_version(output: str) -> str | None:
    """Accept only a single semantic-version line from a fixed version command."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    return _normalized_exact_version(lines[0])


def _committed_lockfile_version(install_root: Path) -> int | None:
    """Read `lockfileVersion` from a committed npm lockfile, or None if there is no answer.

    Only npm writes this field. A missing file, unreadable JSON, or a non-integer value all
    return None, which callers must treat as "unproven" rather than "compatible" -- the whole
    point of reading it is to avoid guessing.
    """
    lockfile = install_root / "package-lock.json"
    if not lockfile.is_file():
        return None
    try:
        parsed = json.loads(lockfile.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    version = parsed.get("lockfileVersion")
    return version if isinstance(version, int) and not isinstance(version, bool) else None


def _package_manager_honours_lockfile(
    manager: str, declared_version: str, actual_version: str, lockfile_version: int | None
) -> bool:
    """Decide whether the worker's package manager can honour the committed lockfile.

    This replaces a demand for exact version equality, which the platform could never
    satisfy for arbitrary repositories and which stopped AB-Feature-222 before a line of code
    was written: `npm@9.8.1` declared, npm 10.9.8 on the worker, `lockfileVersion` 3 -- a
    format every npm from 7 onwards reads and writes.

    Two things are provable and nothing else is guessed:

    * The same major version is compatible for all three managers. Under semver a patch or
      minor release does not change the lockfile format it emits.
    * Across majors, only npm states its own format compatibility in the file itself.
      `lockfileVersion` 1 belongs to npm 5 and 6; 2 and 3 are both read by every npm from 7
      onwards. So an npm at or above the major that introduced the committed format honours
      it, whichever exact version was declared.

    Anything else -- a Yarn 1 worker against a Berry declaration, a pnpm major jump, a
    lockfile that could not be read -- stays blocking. Those are real incompatibilities, and
    a lockfile format this function cannot parse is not evidence of anything.
    """
    declared = _semantic_version(declared_version)
    actual = _semantic_version(actual_version)
    if declared is None or actual is None:
        return False
    if declared[0] == actual[0]:
        return True
    if manager != "npm" or lockfile_version is None:
        return False
    minimum_major = 5 if lockfile_version == 1 else 7 if lockfile_version in {2, 3} else None
    return minimum_major is not None and actual[0] >= minimum_major


def _parse_package_manager_declaration(value: str) -> tuple[str, str] | None:
    """Parse the standard packageManager field without discarding its exact version."""
    match = _PACKAGE_MANAGER_DECLARATION.fullmatch(value.strip())
    if match is None:
        return None
    manager, raw_version = match.groups()
    version = _normalized_exact_version(raw_version)
    if version is None:
        return None
    return manager.lower(), version


def _node_version_satisfies(actual_value: str, requirement: str) -> bool | None:
    """Evaluate common npm engine ranges, returning None for syntax we cannot prove."""
    actual = _semantic_version(actual_value)
    if actual is None:
        return None
    alternatives = [item.strip() for item in requirement.split("||")]
    if not alternatives or any(not item for item in alternatives):
        return None
    branch_results: list[bool] = []
    for alternative in alternatives:
        result = _node_range_branch_matches(actual, alternative)
        if result is None:
            return None
        branch_results.append(result)
    return any(branch_results)


def _node_range_branch_matches(actual: tuple[int, int, int], requirement: str) -> bool | None:
    """Evaluate one whitespace-AND npm range branch."""
    hyphen = re.fullmatch(r"\s*(\S+)\s+-\s+(\S+)\s*", requirement)
    if hyphen is not None:
        lower = _partial_version_bounds(hyphen.group(1))
        upper = _partial_version_bounds(hyphen.group(2))
        if lower is None or upper is None:
            return None
        lower_bound, _lower_ceiling, _lower_complete = lower
        upper_bound, upper_ceiling, upper_complete = upper
        return actual >= lower_bound and (
            actual <= upper_bound
            if upper_complete
            else upper_ceiling is not None and actual < upper_ceiling
        )

    normalized = re.sub(r"([<>=~^]+)\s+(?=[vV\d*xX])", r"\1", requirement)
    tokens = normalized.replace(",", " ").split()
    if not tokens:
        return None
    matches: list[bool] = []
    for token in tokens:
        token_match = _node_comparator_matches(actual, token)
        if token_match is None:
            return None
        matches.append(token_match)
    return all(matches)


def _node_comparator_matches(actual: tuple[int, int, int], token: str) -> bool | None:
    """Evaluate one comparator, wildcard, caret, tilde, or partial exact version."""
    match = re.fullmatch(r"(>=|<=|>|<|=|\^|~)?(.*)", token.strip())
    if match is None:
        return None
    operator, raw_version = match.groups()
    bounds = _partial_version_bounds(raw_version)
    if bounds is None:
        return None
    lower, ceiling, complete = bounds
    operator = operator or "="
    if operator == "=":
        return actual == lower if complete else ceiling is not None and lower <= actual < ceiling
    if operator == ">=":
        return actual >= lower
    if operator == ">":
        return actual > lower if complete else ceiling is not None and actual >= ceiling
    if operator == "<=":
        return actual <= lower if complete else ceiling is not None and actual < ceiling
    if operator == "<":
        return actual < lower
    specified = _specified_version_parts(raw_version)
    if specified is None or specified == 0:
        return None
    if operator == "~":
        upper = (lower[0] + 1, 0, 0) if specified == 1 else (lower[0], lower[1] + 1, 0)
        return lower <= actual < upper
    if operator == "^":
        if specified == 1 or lower[0] > 0:
            upper = (lower[0] + 1, 0, 0)
        elif specified == 2 or lower[1] > 0:
            upper = (0, lower[1] + 1, 0)
        else:
            upper = (0, 0, lower[2] + 1)
        return lower <= actual < upper
    return None


def _specified_version_parts(raw_version: str) -> int | None:
    """Count consecutive numeric components in one partial semantic version."""
    value = raw_version.strip().removeprefix("v").removeprefix("V")
    if value in {"*", "x", "X"}:
        return 0
    parts = value.split(".")
    if not 1 <= len(parts) <= 3:
        return None
    specified = 0
    wildcard_seen = False
    for part in parts:
        if part in {"*", "x", "X"}:
            wildcard_seen = True
            continue
        if wildcard_seen or not part.isdigit() or (len(part) > 1 and part.startswith("0")):
            return None
        specified += 1
    return specified


def _partial_version_bounds(
    raw_version: str,
) -> tuple[tuple[int, int, int], tuple[int, int, int] | None, bool] | None:
    """Return inclusive lower, exclusive partial ceiling, and exactness."""
    specified = _specified_version_parts(raw_version)
    if specified is None:
        return None
    value = raw_version.strip().removeprefix("v").removeprefix("V")
    raw_parts = [] if specified == 0 else value.split(".")[:specified]
    parts = [int(item) for item in raw_parts]
    parts.extend([0] * (3 - len(parts)))
    lower = (parts[0], parts[1], parts[2])
    if specified == 3:
        return lower, None, True
    if specified == 0:
        return lower, (2**31, 0, 0), False
    if specified == 1:
        return lower, (parts[0] + 1, 0, 0), False
    return lower, (parts[0], parts[1] + 1, 0), False


def _runtime_issue(
    issue_id: str,
    project: Path,
    repository_root: Path,
    evidence: str,
    recommended_action: str,
) -> PreflightIssue:
    """Build a repository-scoped, non-repairable runtime compatibility issue."""
    relative = project.relative_to(repository_root).as_posix() or "."
    return _issue(
        issue_id,
        "unsupported_runtime",
        "high",
        "The declared Node toolchain is incompatible with this worker runtime.",
        f"package.json at `{relative}`: {evidence}",
        recommended_action,
        False,
    )


def _deduplicate_issues(issues: list[PreflightIssue]) -> list[PreflightIssue]:
    """Avoid repeating one inherited declaration conflict for every workspace package."""
    unique: dict[tuple[str, str], PreflightIssue] = {}
    for issue in issues:
        unique.setdefault((issue.issue_id, issue.evidence), issue)
    return list(unique.values())


def _install_command(
    manager: str | None, project_root: Path, repository_root: Path
) -> list[str] | None:
    """Install from the lockfile and nothing else -- no advisory lookup on the way.

    `npm ci` runs a security-advisory report before it returns, and that report is a live
    POST to the registry on every install, cache state irrelevant. On 2026-09-03 those two
    endpoints -- `/-/npm/v1/security/audits/quick` and `/-/npm/v1/security/advisories/bulk`
    -- began accepting connections and never answering, and `npm ci` sat on the dead socket
    until something gave up. npm's own timers on a warm cache said it plainly:

        npm timing reify:build            Completed in     267ms
        npm timing auditReport:getReport  Completed in  421789ms

    267 milliseconds of installing, seven minutes of waiting. `npm ci --no-audit` on the same
    checkout took 20s, which is the 25-58s band every install in the journal sat in before
    that day. The registry itself was healthy throughout (package metadata answered 200 in
    90ms, and a POST to an unrelated path 404'd in 0.3s), so nothing here is a fallback for a
    registry outage -- it is the removal of a call this platform never had a use for.

    Dropping it costs no verification. `npm ci` checks every tarball against the integrity
    hash in the committed lockfile whether or not the advisory report runs; the report is
    advice about published CVEs, addressed to a human who reads it, and no caller here reads
    it. A bootstrap for a sandboxed build is not the place to learn about them.

    `pnpm` and `yarn` do not audit on install, so only npm needs saying. Kept as a flag on
    the command rather than configuration: the command is what the journal records and what
    an operator replays by hand, and a behaviour that lives in an env var is one an operator
    reproducing the failure will not have.
    """
    arguments = {
        "npm": ["ci", "--no-audit"],
        "pnpm": ["install", "--frozen-lockfile"],
        "yarn": ["install", "--frozen-lockfile"],
    }
    if manager is None or manager not in arguments:
        return None
    # The repository's own declared version where it declares one, so two repositories
    # pinning different versions both install. `argv` is `[manager]` when it declares
    # nothing, which is the image's manager and the behaviour this platform always had.
    argv, _version = package_manager_argv(manager, project_root, repository_root)
    return [*argv, *arguments[manager]]


# The lockfiles an install reads, by the package manager that resolves them. Keyed by the
# same manager names `_dependency_installers` produces, so adding a package manager there
# and forgetting it here answers `None` -- which installs -- rather than answering wrongly.
_MANAGER_LOCKFILES = {
    "npm": ("package-lock.json",),
    "pnpm": ("pnpm-lock.yaml",),
    "yarn": ("yarn.lock",),
    "uv": ("uv.lock",),
}

# Where each manager's install actually lands, so the unchanged-lockfile skip below can tell
# "nothing has changed" from "nothing was ever installed". A lockfile digest says only that
# an install *would* be a repeat of the last one -- it says nothing about whether that install
# ever actually ran, and it is recorded even when one of the earlier `install_status` branches
# refused to run one (`blocked_unsupported_runtime`, `blocked_missing_lockfile`). A worker that
# was blocked on attempt 0 for the wrong Node version, then unblocked on attempt 1 with the
# lockfile untouched, saw this skip fire with no `node_modules` on disk anywhere -- the
# repository's checked-in dependency (already declared in its manifest) was never missing, it
# was simply never installed, and lint/test then failed on it in a way this platform
# misclassified as invalid repository configuration rather than as its own skipped step.
_MANAGER_INSTALLED_TREES = {
    "npm": "node_modules",
    "pnpm": "node_modules",
    "yarn": "node_modules",
    "uv": ".venv",
}


def _installed_trees_present(installers: Sequence[tuple[str, list[str], Path]]) -> bool:
    """Whether every installer's own installed tree already exists and holds something.

    Checked directly on disk rather than inferred from any recorded status, because the
    workspace this runs against is the one thing here that cannot lie: a preserved workspace
    whose lockfile digest matches is exactly the case an empty or absent tree must not be
    allowed to pass silently. Fails toward installing -- an unrecognised manager or a tree this
    call cannot read answers "not present", which routes to a real install rather than another
    skip.
    """
    for manager, _command, install_root in installers:
        tree_name = _MANAGER_INSTALLED_TREES.get(manager)
        if tree_name is None:
            return False
        tree = install_root / tree_name
        try:
            if not tree.is_dir() or not any(tree.iterdir()):
                return False
        except OSError:
            return False
    return True


def _lockfile_digest(root: Path, installers: Sequence[tuple[str, list[str], Path]]) -> str | None:
    """Digest the lockfiles these installers resolve, or ``None`` if that cannot be answered.

    Over the BYTES, not over mtimes or sizes: a checkout is restored, rewritten and reset
    between attempts, and only the content answers "would this install produce the same tree
    it produced last time".

    `None` for every unanswerable case -- no installer, an unrecognised manager, a lockfile
    that has gone missing or will not read -- because the only caller treats `None` as
    "install", and a digest that quietly stands for an unread file would skip an install
    this checkout needed.
    """
    if not installers:
        return None
    digest = hashlib.sha256()
    for manager, _command, install_root in installers:
        names = _MANAGER_LOCKFILES.get(manager)
        if names is None:
            return None
        for name in names:
            lockfile = install_root / name
            try:
                content = lockfile.read_bytes()
            except OSError:
                return None
            digest.update(install_root.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _dependency_installers(root: Path) -> list[tuple[str, list[str], Path]]:
    """Select only frozen, repository-declared dependency setup commands.

    A Python project without a frozen uv lock is intentionally not bootstrapped by
    guesswork.  Likewise, this does not create a lockfile, add a package manager, or
    run a package install merely because a source file has a familiar extension.
    """
    installers: list[tuple[str, list[str], Path]] = []
    for project in _manifest_roots(root, "package.json"):
        selected = _node_install_root(project, root)
        if selected is None:
            continue
        manager, install_root = selected
        command = _install_command(manager, install_root, root)
        if command is not None:
            installers.append((manager, command, install_root))
    for project in _manifest_roots(root, "pyproject.toml"):
        if (project / "uv.lock").is_file():
            installers.append(("uv", ["uv", "sync", "--frozen"], project))
    unique: dict[tuple[str, tuple[str, ...], Path], tuple[str, list[str], Path]] = {}
    for name, command, install_root in installers:
        unique[(name, tuple(command), install_root)] = (name, command, install_root)
    return [unique[key] for key in sorted(unique, key=lambda item: (str(item[2]), item[0]))]


def _node_install_root(project: Path, repository_root: Path) -> tuple[str, Path] | None:
    """Find the nearest checked-in Node lockfile without assuming a monorepo layout."""
    current = project
    while True:
        manager = _node_package_manager(current)
        if manager is not None:
            return manager, current
        if current == repository_root:
            return None
        current = current.parent


def _manifest_roots(root: Path, name: str) -> list[Path]:
    """Find project manifests while excluding generated dependency workspaces."""
    ignored = {".git", ".venv", "node_modules", "__pycache__"}
    return sorted(
        {
            path.parent
            for path in root.rglob(name)
            if path.is_file() and not any(part in ignored for part in path.relative_to(root).parts)
        }
    )


def _entrypoints(root: Path, package: dict[str, object]) -> list[str]:
    entries: list[str] = []
    for key in ("main", "module", "types"):
        value = package.get(key)
        if isinstance(value, str):
            entries.append(value)
    entries.extend(
        name
        for name in ("index.js", "index.ts", "src/index.js", "src/index.ts", "server.js")
        if (root / name).is_file()
    )
    return list(dict.fromkeys(entries))


def _issue(
    issue_id: str,
    category: Literal[
        "dependency_configuration",
        "missing_dependency",
        "invalid_lint_configuration",
        "missing_test_command",
        "missing_build_command",
        "source_structure",
        "repository_permissions",
        "package_manager_lockfile",
        "baseline_validation",
        "unsupported_runtime",
    ],
    severity: Literal["critical", "high", "medium", "low", "info"],
    description: str,
    evidence: str,
    recommended_action: str,
    automatically_repairable: bool,
) -> PreflightIssue:
    return PreflightIssue(
        issue_id=issue_id,
        category=category,
        severity=severity,
        description=description,
        evidence=evidence,
        recommended_action=recommended_action,
        automatically_repairable=automatically_repairable,
    )


def install_failure_is_transient(result: ProcessResult) -> bool:
    """Return whether a failed install says the registry or the network was the problem.

    Reads the process output as evidence and never publishes it: what crosses into durable
    state is still `_process_evidence`, which is composed from platform-owned fields alone.
    """
    if result.cancelled or result.timed_out:
        return False
    haystack = f"{result.stdout}\n{result.stderr}".lower()
    return any(marker in haystack for marker in _TRANSIENT_INSTALL_MARKERS)


def _process_evidence(command: list[str], result: ProcessResult) -> str:
    """Persist only platform-owned outcome fields, never package lifecycle output."""
    manager = command[0] if command else "unknown"
    code = subprocess_result_code(
        return_code=result.return_code,
        timed_out=result.timed_out,
        cancelled=result.cancelled,
        prefix="dependency_installation",
    )
    return f"Dependency manager `{manager}` did not complete successfully ({code})."


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


__all__ = [
    "LintFailureClassification",
    "PreflightIssue",
    "RepositoryPreflight",
    "RepositoryPreflightResult",
    "classify_lint_failure",
]
