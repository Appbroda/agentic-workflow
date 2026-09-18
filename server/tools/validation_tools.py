"""Repository-aware, revision-safe workspace validation.

Commands are constructed from versioned repository evidence.  The coding model never
supplies a shell command, and a journal entry is reusable only for the exact workspace
revision and validation configuration that produced it.
"""

from __future__ import annotations

import json
import math
import os
import re
import selectors
import signal
import subprocess
import time
from collections import deque
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast

from pydantic import Field

from services.cancellation import CancellationToken, MockCancellationToken
from services.external_operations import ExternalOperationExecutor
from services.process_runner import (
    AsyncioProcessRunner,
    ProcessRunner,
    redact_output,
    repository_subprocess_environment,
    subprocess_result_code,
)
from state.external_operations import (
    BASELINE_VALIDATION_LOGICAL_STEP,
    CHANGE_VALIDATION_LOGICAL_STEP,
    ExternalOperationType,
)
from state.feature_models import RepositorySpec
from state.models import StateModel
from storage.external_operation_store import OperationResult, fingerprint_operation_input
from tools.diagnostic_references import diagnostic_file_references
from tools.file_tools import (
    PathLike,
    WorkspacePathError,
    resolve_workspace_path,
    resolve_workspace_root,
)
from tools.node_toolchain import package_manager_argv
from tools.repository_environment import declared_validation_environment
from tools.repository_preflight import classify_lint_failure
from tools.technology_detection import (
    RepositoryRevision,
    RepositoryTechnologyProfile,
    calculate_repository_revision,
    calculate_repository_revision_sync,
    inspect_repository_technology,
)

type ValidationType = Literal["format", "lint", "typecheck", "test", "build", "custom"]

# Whose work a run measured. The same commands, through the same tool, on the same checkout:
# what differs is whether the Engineer had written anything yet, and that is a fact about the
# caller rather than about the command, so it travels as an argument and is recorded.
type ValidationPurpose = Literal["change", "baseline"]

_LOGICAL_STEP_FOR_PURPOSE: dict[ValidationPurpose, str] = {
    "change": CHANGE_VALIDATION_LOGICAL_STEP,
    "baseline": BASELINE_VALIDATION_LOGICAL_STEP,
}

# Enough for a test runner's failure list and summary without carrying a whole transcript
# into durable state.
_MAX_SUMMARY_CHARACTERS = 4_000

# How many workspace files a failing command's summary names explicitly. First-seen order,
# one entry per file; a pathological output cannot bloat the durable record past this.
_MAX_REFERENCED_WORKSPACE_FILES = 20

# Heads the reference list appended to a failing summary. AB-Feature-182's frontend spent
# three remediation attempts patching a test fixture because the tail-bounded summary had
# dropped the one stack frame naming the product file: Jest prints the throwing frame near
# the top and fills the tail with framework frames, so the tail alone said `AllApps.test.js`
# where the defect lived in `AllApps.js`. References are therefore resolved from the FULL
# output before any bounding, and appended at the end -- the position both this module's
# tail-bound and the reviewer's tail-bound excerpt preserve by construction.
_WORKSPACE_REFERENCES_HEADER = "[workspace files named by this output]"

# Package-manager conventions, not repository details -- the same standard
# `_EMPTY_TEST_SUITE_MARKERS` applies. A frame inside one of these is a dependency's
# internals, not a file any repair may edit.
_PACKAGE_STORE_SEGMENTS = frozenset(
    {"node_modules", "site-packages", "dist-packages", "vendor", ".venv", "venv"}
)


class ValidationStatus(StrEnum):
    """Semantically meaningful validation outcomes used by reviewers and APIs."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_CONFIGURED = "not_configured"
    NO_TESTS_FOUND = "no_tests_found"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class ValidationCommand(StateModel):
    """A fixed, non-shell validation invocation selected from repository evidence."""

    validation_type: ValidationType
    command: list[str] = Field(default_factory=list)
    working_directory: str = "."
    required: bool = True
    timeout_seconds: float = Field(gt=0)
    success_exit_codes: list[int] = Field(default_factory=lambda: [0])
    environment: dict[str, str] = Field(default_factory=dict)
    # Whether this command accepts trailing test paths, and what has to precede them. Read
    # off the checkout's own script rather than assumed: `npm run test` forwards nothing
    # without a `--`, and a script that chains two commands would give the arguments to
    # whichever ran last. Absent means "do not narrow this command", which is always safe.
    accepts_test_paths: bool = False
    test_path_separator: list[str] = Field(default_factory=list)
    # Which file suffixes this command's runner can collect when handed explicit paths, set
    # where the command is built and the script body naming the runner is in hand -- at
    # narrowing time `npm run test` names only the package manager. Empty means the runner is
    # unknown, and narrowing then passes every changed test path, which is today's behaviour
    # and also what every plan persisted before this field existed says.
    collectible_suffixes: list[str] = Field(default_factory=list)


class RepositoryValidationPlan(StateModel):
    """All commands applicable to a repository at a point in time."""

    repository_id: str = Field(min_length=1)
    technology_profile: RepositoryTechnologyProfile
    commands: list[ValidationCommand] = Field(default_factory=list)
    source: Literal["repository_scripts", "detected_defaults", "configured"]


class ValidationPlanBuilder(Protocol):
    """Build commands from a checkout and durable repository specification."""

    async def build_plan(
        self, repository_root: Path, repository_spec: RepositorySpec | None = None
    ) -> RepositoryValidationPlan:
        """Return only validated local commands; never execute repository code."""


class DefaultValidationPlanBuilder:
    """Deterministic command planner for JavaScript/TypeScript and Python repositories."""

    def __init__(self, *, default_timeout_seconds: float = 120.0) -> None:
        if default_timeout_seconds <= 0:
            msg = "default_timeout_seconds must be positive"
            raise ValueError(msg)
        self._timeout = default_timeout_seconds

    async def build_plan(
        self, repository_root: Path, repository_spec: RepositorySpec | None = None
    ) -> RepositoryValidationPlan:
        return self.build_plan_sync(repository_root, repository_spec=repository_spec)

    def build_plan_sync(
        self, repository_root: PathLike, repository_spec: RepositorySpec | None = None
    ) -> RepositoryValidationPlan:
        root = resolve_workspace_root(repository_root)
        repository_id = repository_spec.repository_id if repository_spec is not None else root.name
        profile = inspect_repository_technology(root, repository_id=repository_id)
        commands: list[ValidationCommand] = []
        source: Literal["repository_scripts", "detected_defaults", "configured"] = (
            "detected_defaults"
        )
        for package_root in _manifest_roots(root, "package.json"):
            package = _package_json(package_root)
            scripts = package.get("scripts") if isinstance(package.get("scripts"), dict) else {}
            if not isinstance(scripts, dict):
                continue
            manager = _node_package_manager(package_root, repository_root=root)
            if manager is None:
                continue
            source = "repository_scripts"
            commands.extend(
                _with_working_directory(
                    _node_commands(
                        package_root, manager, scripts, self._timeout, repository_root=root
                    ),
                    package_root.relative_to(root).as_posix() or ".",
                )
            )
        for python_root in _python_project_roots(root):
            python_commands = _python_commands(python_root, self._timeout)
            if not python_commands:
                continue
            commands.extend(
                _with_working_directory(
                    python_commands, python_root.relative_to(root).as_posix() or "."
                )
            )
            source = "configured" if source == "detected_defaults" else source
        return RepositoryValidationPlan(
            repository_id=repository_id,
            technology_profile=profile,
            commands=commands,
            source=source,
        )


class PinnedValidationPlanBuilder:
    """Serve one workstream's persisted plan instead of re-deriving from the working tree.

    The validation plan is a property of the repository at the workstream's baseline, not of
    whatever the current attempt has written into the checkout. Deriving per attempt let an
    agent mint its own required gate: AB-Feature-216's coder added the repository's first
    `test_*.py`, and every later derivation -- the commit gate's, the reviewer's, the scoped
    runner's -- planned a required `pytest -q .` for a repository that configures no Python
    test tooling at all. Every consumer of the plan is handed this builder for the life of the
    attempt, so they all answer from the same baseline set.
    """

    def __init__(self, plan: RepositoryValidationPlan) -> None:
        self._plan = plan

    async def build_plan(
        self, repository_root: Path, repository_spec: RepositorySpec | None = None
    ) -> RepositoryValidationPlan:
        del repository_root, repository_spec
        return self._plan

    def build_plan_sync(
        self, repository_root: PathLike, repository_spec: RepositorySpec | None = None
    ) -> RepositoryValidationPlan:
        del repository_root, repository_spec
        return self._plan


class MockValidationPlanBuilder:
    """Deterministic test double with an optional explicitly supplied plan."""

    def __init__(self, plan: RepositoryValidationPlan | None = None) -> None:
        self._plan = plan

    async def build_plan(
        self, repository_root: Path, repository_spec: RepositorySpec | None = None
    ) -> RepositoryValidationPlan:
        if self._plan is not None:
            return self._plan
        root = resolve_workspace_root(repository_root)
        repository_id = repository_spec.repository_id if repository_spec is not None else root.name
        return RepositoryValidationPlan(
            repository_id=repository_id,
            technology_profile=inspect_repository_technology(root, repository_id=repository_id),
            commands=[],
            source="detected_defaults",
        )


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Bounded result of one repository-native validation command.

    Existing call sites can continue to use ``return_code``, ``stdout``, and
    ``succeeded``.  The added fields make the result auditable and prevent a stale
    result from being treated as evidence for a later engineer revision.
    """

    command: tuple[str, ...]
    return_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    output_truncated: bool = False
    cancelled: bool = False
    validation_type: ValidationType = "custom"
    repository_id: str = "unknown"
    repository_revision: str = "legacy"
    working_tree_fingerprint: str = "legacy"
    status: ValidationStatus | None = None
    operation_id: str | None = None
    stdout_summary: str = ""
    stderr_summary: str = ""
    validated_revision: str | None = None
    current_revision: str | None = None
    is_current: bool = True
    superseded_by_revision: str | None = None
    required: bool = True
    failure_classification: str | None = None
    result_code: str | None = None

    def __post_init__(self) -> None:
        status = self.status
        if status is None:
            status = _status_for(
                validation_type=self.validation_type,
                return_code=self.return_code,
                timed_out=self.timed_out,
                cancelled=self.cancelled,
                command=self.command,
                output=f"{self.stdout}\n{self.stderr}",
            )
            object.__setattr__(self, "status", status)
        # A failing command's own report is the only thing that says what to change, and it
        # reaches the coding model as the reviewer's validation evidence. Feature -062 spent
        # four review cycles on "npm run test exited with code 1" while the output it was
        # denied named the defect and its line: a test mocking a module it had not mocked.
        #
        # A checkout does control this text, so it is bounded to a tail -- where test runners
        # print their failures and summary -- and passed through the same redaction as any
        # other subprocess output. A passing command's chatter has no diagnostic value and is
        # still discarded.
        #
        # A summary supplied at construction wins: the live-run path (`_result`) builds a
        # richer one there, where the workspace is in hand -- the bounded tail plus the
        # workspace files the FULL output names, which a plain tail can silently drop
        # (AB-Feature-182's Jest crash buried the throwing frame above 4,000 characters of
        # framework frames). This fallback covers construction paths with no workspace:
        # replayed journal payloads and test doubles.
        failed = _output_is_diagnostic(
            status=self.status,
            return_code=self.return_code,
            timed_out=self.timed_out,
            cancelled=self.cancelled,
        )
        if failed:
            if not self.stdout_summary:
                object.__setattr__(self, "stdout_summary", _bounded_summary(self.stdout))
            if not self.stderr_summary:
                object.__setattr__(self, "stderr_summary", _bounded_summary(self.stderr))
        else:
            object.__setattr__(self, "stdout_summary", "")
            object.__setattr__(self, "stderr_summary", "")
        # Set here rather than at one construction site so every path that builds a result --
        # a live run, a replayed journal payload, an injected test double -- carries the same
        # verdict about whether the command completed at all. A caller that already
        # classified the failure (lint bootstrap) keeps its own, more specific answer.
        if self.status is ValidationStatus.FAILED and self.failure_classification is None:
            capacity = _capacity_classification(
                timed_out=self.timed_out,
                return_code=self.return_code,
                output=f"{self.stdout}\n{self.stderr}",
            )
            if capacity is not None:
                object.__setattr__(self, "failure_classification", capacity)
        if self.validated_revision is None:
            object.__setattr__(self, "validated_revision", self.repository_revision)
        if self.current_revision is None:
            object.__setattr__(self, "current_revision", self.repository_revision)
        object.__setattr__(self, "result_code", _validation_result_code(self))

    @property
    def succeeded(self) -> bool:
        """Only a successful command is passing validation evidence."""
        return self.status is ValidationStatus.PASSED

    @property
    def exit_code(self) -> int | None:
        """Use the public model name while keeping the historic property compatible."""
        return self.return_code


class ValidationTool(Protocol):
    """Protocol retained for injected legacy test implementations."""

    def run_ruff(self, *, timeout_seconds: float) -> ValidationResult:
        """Run Ruff with an explicit timeout."""

    def run_pytest(self, *, timeout_seconds: float) -> ValidationResult:
        """Run pytest with an explicit timeout."""


class WorkspaceValidationTools:
    """Repository-aware validation bound to one isolated workspace."""

    def __init__(
        self,
        workspace_root: PathLike,
        *,
        repository_id: str | None = None,
        repository_spec: RepositorySpec | None = None,
        plan_builder: ValidationPlanBuilder | None = None,
        cancellation_token: CancellationToken | None = None,
        process_runner: ProcessRunner | None = None,
        operation_executor: ExternalOperationExecutor | None = None,
        changed_test_paths: Sequence[str] | None = None,
    ) -> None:
        self._changed_test_paths = [
            str(item) for item in (changed_test_paths or []) if str(item).strip()
        ]
        self.workspace_root = resolve_workspace_root(workspace_root)
        self._repository_id = repository_id or (
            repository_spec.repository_id
            if repository_spec is not None
            else self.workspace_root.name
        )
        self._repository_spec = repository_spec
        self._plan_builder = plan_builder or DefaultValidationPlanBuilder()
        self._cancellation_token = cancellation_token or MockCancellationToken()
        self._process_runner = process_runner or AsyncioProcessRunner()
        self._operation_executor = operation_executor

    def technology_profile(self) -> RepositoryTechnologyProfile:
        """Expose the deterministic profile for child-result/API diagnostics."""
        return inspect_repository_technology(self.workspace_root, repository_id=self._repository_id)

    def validation_plan(self) -> RepositoryValidationPlan:
        """Return the deterministic plan synchronously for non-live callers."""
        builder = self._plan_builder
        sync = getattr(builder, "build_plan_sync", None)
        if sync is not None:
            return cast(
                RepositoryValidationPlan,
                sync(self.workspace_root, repository_spec=self._repository_spec),
            )
        return DefaultValidationPlanBuilder().build_plan_sync(
            self.workspace_root, repository_spec=self._repository_spec
        )

    def run_ruff(self, *, timeout_seconds: float) -> ValidationResult:
        return run_ruff(self.workspace_root, timeout_seconds=timeout_seconds)

    def run_pytest(self, *, timeout_seconds: float) -> ValidationResult:
        return run_pytest(self.workspace_root, timeout_seconds=timeout_seconds)

    def run_validations(self, *, timeout_seconds: float) -> tuple[ValidationResult, ...]:
        plan = self.validation_plan()
        revision = calculate_repository_revision_sync(self.workspace_root)
        return _run_plan_sync(self.workspace_root, plan, revision, timeout_seconds=timeout_seconds)

    async def run_ruff_async(self, *, timeout_seconds: float) -> ValidationResult:
        return await self._run_command_async(
            ValidationCommand(
                validation_type="lint",
                command=["ruff", "check", "."],
                timeout_seconds=timeout_seconds,
            )
        )

    async def run_pytest_async(self, *, timeout_seconds: float) -> ValidationResult:
        return await self._run_command_async(
            ValidationCommand(
                validation_type="test",
                command=["pytest", "-q", "."],
                timeout_seconds=timeout_seconds,
            )
        )

    async def run_validations_async(
        self, *, timeout_seconds: float, purpose: ValidationPurpose = "change"
    ) -> tuple[ValidationResult, ...]:
        """Run the repository's own required commands and report what they say.

        ``purpose`` says whose work is being measured, and it is a parameter rather than a
        property of this object because one instance serves both callers: the platform asks
        it about the untouched checkout before coding, and the attempt's own validation asks
        it about the change afterwards. The default is the attempt's, so a caller that says
        nothing keeps today's meaning.
        """
        plan = await self._plan_builder.build_plan(
            self.workspace_root, repository_spec=self._repository_spec
        )
        if not plan.commands:
            revision = await calculate_repository_revision(self.workspace_root)
            return (_not_configured_result(plan, revision),)
        results: list[ValidationResult] = []
        for command in plan.commands:
            narrowed = (
                scoped_test_command(command, self._changed_test_paths)
                if command.validation_type == "test"
                else None
            )
            if narrowed is not None:
                scoped_result = await self._run_command_async(
                    narrowed, timeout_seconds=timeout_seconds, plan=plan, purpose=purpose
                )
                if scoped_result.status is ValidationStatus.FAILED:
                    # The change's own suites already reject it, so the whole-repository run
                    # can only agree, far more slowly. -108 paid fifteen minutes and an
                    # exhausted heap for that agreement on five separate attempts, and the
                    # answer it needed was in the file it had just written.
                    #
                    # Promoted to required: this is the repository's own runner rejecting the
                    # source, which is a real test failure however few files it was given.
                    results.append(replace(scoped_result, required=True))
                    continue
                # It passed, so it proves nothing on its own and the configured command still
                # decides. Kept in the results because a full run that then cannot complete
                # leaves this as the only evidence anybody has about the change itself.
                results.append(scoped_result)
            results.append(
                await self._run_command_async(
                    command, timeout_seconds=timeout_seconds, plan=plan, purpose=purpose
                )
            )
        if not any(command.validation_type == "test" for command in plan.commands):
            revision = await calculate_repository_revision(self.workspace_root)
            return (*results, _not_configured_result(plan, revision))
        return tuple(results)

    async def run_changed_tests_async(
        self, paths: Sequence[str], *, timeout_seconds: float | None = None
    ) -> tuple[ValidationResult, ...]:
        """Run only the narrowed form of each configured test command, and nothing else.

        The counterpart to `run_validations_async` for a caller that wants fast feedback
        rather than a verdict: the Engineer, asking the repository's own runner about the
        suites its attempt just wrote, before it declares itself done. A command that cannot
        be narrowed is skipped rather than run whole -- an in-attempt check that ran the
        entire repository suite would cost the fifteen minutes `scoped_test_command` exists
        to avoid, and the authoritative full run happens in the Reviewer regardless.
        """
        plan = await self._plan_builder.build_plan(
            self.workspace_root, repository_spec=self._repository_spec
        )
        results: list[ValidationResult] = []
        for command in plan.commands:
            if command.validation_type != "test":
                continue
            narrowed = scoped_test_command(command, paths)
            if narrowed is None:
                continue
            results.append(
                await self._run_command_async(narrowed, timeout_seconds=timeout_seconds, plan=plan)
            )
        return tuple(results)

    def focus_on_changed_tests(self, paths: Sequence[str]) -> None:
        """Narrow this repository's test command to the suites one attempt changed.

        Set after coding rather than at construction, because which suites an attempt
        touched is not known until it has written them.
        """
        self._changed_test_paths = [str(item) for item in paths if str(item).strip()]

    async def run_lint_checks_async(
        self, *, timeout_seconds: float
    ) -> tuple[ValidationResult, ...]:
        """Run only checkout-configured linters before coding to establish baseline setup.

        Baseline by definition, so the purpose is not a parameter here: this method exists to
        measure findings the checkout already had, which are not this change's to fix.
        """
        return await self._run_typed_commands_async(
            "lint", timeout_seconds=timeout_seconds, purpose="baseline"
        )

    async def run_typecheck_checks_async(
        self, *, timeout_seconds: float | None = None
    ) -> tuple[ValidationResult, ...]:
        """Run only the checkout's own typecheck commands, and nothing else.

        The Engineer's counterpart to `run_lint_checks_async`, asked *inside* the attempt.
        Only the typecheck-typed commands: lint is already covered by the source gate, and a
        whole-repository test run here would cost the minutes `scoped_test_command` exists to
        avoid. A repository that configures no typecheck command gets no results, which is a
        fact about the checkout and never an assertion that the types are sound.
        """
        return await self._run_typed_commands_async("typecheck", timeout_seconds=timeout_seconds)

    async def _run_typed_commands_async(
        self,
        validation_type: ValidationType,
        *,
        timeout_seconds: float | None,
        purpose: ValidationPurpose = "change",
    ) -> tuple[ValidationResult, ...]:
        """Run every planned command of one type, in the order the plan holds them."""
        plan = await self._plan_builder.build_plan(
            self.workspace_root, repository_spec=self._repository_spec
        )
        results: list[ValidationResult] = []
        for command in plan.commands:
            if command.validation_type == validation_type:
                results.append(
                    await self._run_command_async(
                        command, timeout_seconds=timeout_seconds, plan=plan, purpose=purpose
                    )
                )
        return tuple(results)

    async def _run_command_async(
        self,
        command: ValidationCommand,
        *,
        timeout_seconds: float | None = None,
        plan: RepositoryValidationPlan | None = None,
        purpose: ValidationPurpose = "change",
    ) -> ValidationResult:
        effective_timeout = timeout_seconds or command.timeout_seconds
        revision = await calculate_repository_revision(self.workspace_root)
        plan = plan or RepositoryValidationPlan(
            repository_id=self._repository_id,
            technology_profile=self.technology_profile(),
            commands=[command],
            source="configured",
        )
        command_fingerprint = fingerprint_operation_input(
            {
                "validation_type": command.validation_type,
                "command": command.command,
                "working_directory": command.working_directory,
                "success_exit_codes": command.success_exit_codes,
                "environment": command.environment,
                "plan_source": plan.source,
            }
        )
        workdir = _resolve_workdir(self.workspace_root, command.working_directory)
        safe_input = {
            "workspace_path": str(self.workspace_root),
            "repository_id": plan.repository_id,
            "repository_revision": revision.combined_fingerprint,
            "head_sha": revision.head_sha,
            "working_tree_fingerprint": revision.working_tree_fingerprint,
            "command_fingerprint": command_fingerprint,
            "validation_type": command.validation_type,
            "command": command.command,
            "working_directory": command.working_directory,
            "environment_fingerprint": fingerprint_operation_input(command.environment),
            "plan_source": plan.source,
            # What this run was for, in the idempotency input as well as the step. Deliberate:
            # a coding call that writes nothing leaves the revision unchanged, so without this
            # the attempt's own validation would reuse the baseline row and an attempt that
            # wrote nothing would be recorded as having validated a change. The cost is
            # running the suite twice in that one case, which is the honest trade.
            "purpose": purpose,
        }

        async def action() -> tuple[ValidationResult, OperationResult]:
            result = await _run_validation_command_async(
                workdir,
                tuple(command.command),
                timeout_seconds=effective_timeout,
                cancellation_token=self._cancellation_token,
                process_runner=self._process_runner,
                validation_type=command.validation_type,
                repository_id=plan.repository_id,
                revision=revision,
                success_exit_codes=tuple(command.success_exit_codes),
                environment=command.environment,
                required=command.required,
            )
            return result, OperationResult(payload=_result_payload(result))

        if self._operation_executor is None:
            result, _ = await action()
            return result
        journaled = await self._operation_executor.run(
            operation_type=_operation_type(command.validation_type),
            logical_step=(
                f"{_LOGICAL_STEP_FOR_PURPOSE[purpose]}:{command.validation_type}:"
                f"{revision.combined_fingerprint[:16]}:{command_fingerprint[:16]}"
            ),
            safe_input=safe_input,
            # Explicitly duplicate revision data in idempotency input: a stale command is
            # never eligible for reuse after coding changes, commits, or untracked edits.
            idempotency_input=safe_input,
            action=action,
        )
        if not journaled.reused:
            result = cast(ValidationResult, journaled.value)
            return _with_operation_id(result, journaled.operation.operation_id)
        return _result_from_payload(
            journaled.operation.result_payload or {},
            command=command,
            repository_id=plan.repository_id,
            revision=revision,
            operation_id=journaled.operation.operation_id,
        )


def run_ruff(workspace_root: PathLike, *, timeout_seconds: float) -> ValidationResult:
    """Legacy explicit Ruff helper; repository-native callers should use a plan."""
    revision = calculate_repository_revision_sync(workspace_root)
    return _run_validation_command(
        workspace_root,
        ("ruff", "check", "."),
        timeout_seconds=timeout_seconds,
        validation_type="lint",
        revision=revision,
    )


def run_pytest(workspace_root: PathLike, *, timeout_seconds: float) -> ValidationResult:
    """Legacy explicit pytest helper; exit code 5 is recorded as no-tests, not success."""
    revision = calculate_repository_revision_sync(workspace_root)
    return _run_validation_command(
        workspace_root,
        ("pytest", "-q", "."),
        timeout_seconds=timeout_seconds,
        validation_type="test",
        revision=revision,
    )


async def run_ruff_async(
    workspace_root: PathLike,
    *,
    timeout_seconds: float,
    cancellation_token: CancellationToken,
    process_runner: ProcessRunner | None = None,
) -> ValidationResult:
    revision = await calculate_repository_revision(workspace_root)
    return await _run_validation_command_async(
        workspace_root,
        ("ruff", "check", "."),
        timeout_seconds=timeout_seconds,
        cancellation_token=cancellation_token,
        process_runner=process_runner or AsyncioProcessRunner(),
        validation_type="lint",
        revision=revision,
    )


async def run_pytest_async(
    workspace_root: PathLike,
    *,
    timeout_seconds: float,
    cancellation_token: CancellationToken,
    process_runner: ProcessRunner | None = None,
) -> ValidationResult:
    revision = await calculate_repository_revision(workspace_root)
    return await _run_validation_command_async(
        workspace_root,
        ("pytest", "-q", "."),
        timeout_seconds=timeout_seconds,
        cancellation_token=cancellation_token,
        process_runner=process_runner or AsyncioProcessRunner(),
        validation_type="test",
        revision=revision,
    )


def _workspace_validation_commands(
    workspace_root: Path,
) -> tuple[tuple[tuple[str, ...], ExternalOperationType], ...]:
    """Compatibility helper exposing only real configured commands for older callers."""
    plan = DefaultValidationPlanBuilder().build_plan_sync(workspace_root)
    return tuple(
        (tuple(command.command), _operation_type(command.validation_type))
        for command in plan.commands
    )


def _run_plan_sync(
    root: Path,
    plan: RepositoryValidationPlan,
    revision: RepositoryRevision,
    *,
    timeout_seconds: float,
) -> tuple[ValidationResult, ...]:
    if not plan.commands:
        return (_not_configured_result(plan, revision),)
    results = tuple(
        _run_validation_command(
            _resolve_workdir(root, command.working_directory),
            tuple(command.command),
            timeout_seconds=timeout_seconds,
            validation_type=command.validation_type,
            repository_id=plan.repository_id,
            revision=revision,
            success_exit_codes=tuple(command.success_exit_codes),
            environment=command.environment,
            required=command.required,
        )
        for command in plan.commands
    )
    if not any(command.validation_type == "test" for command in plan.commands):
        return (*results, _not_configured_result(plan, revision))
    return results


# Test runners that take the files to run as trailing arguments. Every one of these is a
# framework convention rather than a repository detail, which is what keeps this detection
# repo-agnostic: the checkout's own `test` script names its runner, and the runner decides
# whether narrowing is possible.
_PATH_ARGUMENT_TEST_RUNNERS = (
    "react-scripts test",
    "craco test",
    "vitest",
    "jest",
    "mocha",
    "jasmine",
    "bun test",
    "node --test",
    "ava",
)

# A script that chains or pipes commands cannot be narrowed: trailing arguments would reach
# only whichever part ran last, silently testing something other than what was asked for.
_SCRIPT_COMPOSITION_OPERATORS = ("&&", "||", ";", "|", ">", "<", "&")

# What each runner above can collect when handed explicit file paths -- rows beside the runner
# list they mirror, framework conventions rather than repository details. AB-Feature-216's
# narrowed commands crossed languages both ways: `pytest -q` was handed three `.js` suites it
# cannot collect (a guaranteed rejection the agent then silenced by rewriting the repository's
# test collection), and `npm run test --` was handed a `.py` file.
_JS_RUNNER_SUFFIXES = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts")
_RUNNER_COLLECTIBLE_SUFFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("react-scripts test", _JS_RUNNER_SUFFIXES),
    ("craco test", _JS_RUNNER_SUFFIXES),
    ("vitest", _JS_RUNNER_SUFFIXES),
    ("jest", _JS_RUNNER_SUFFIXES),
    ("mocha", _JS_RUNNER_SUFFIXES),
    ("jasmine", _JS_RUNNER_SUFFIXES),
    ("bun test", _JS_RUNNER_SUFFIXES),
    ("node --test", _JS_RUNNER_SUFFIXES),
    ("ava", _JS_RUNNER_SUFFIXES),
    ("pytest", (".py",)),
)


def _collectible_suffixes(script_body: str) -> list[str]:
    """Read which suffixes the script's own runner can collect, or nothing when unknown."""
    lowered = script_body.lower()
    for runner, suffixes in _RUNNER_COLLECTIBLE_SUFFIXES:
        if runner in lowered:
            return list(suffixes)
    return []


# Flags by which a formatter says it will edit rather than report. `--check`, `--list-
# different` and `--dry-run` are the opposite declaration and win, because a script may
# reasonably carry both -- `prettier --write` in one branch and `--check` in another -- and
# the checking form is the one that produces a verdict.
_REWRITE_FLAGS = ("--write", "--fix", " -w ", "--in-place", "-i ")
_CHECK_FLAGS = ("--check", "--list-different", "-l ", "--dry-run", "--no-write")


def _rewrites_its_input(script_body: str) -> bool:
    """Return whether a checkout's own script edits the files it looks at."""
    lowered = f" {script_body.lower().strip()} "
    if any(flag in lowered for flag in _CHECK_FLAGS):
        return False
    return any(flag in lowered for flag in _REWRITE_FLAGS)


def _accepts_test_paths(script_body: str) -> bool:
    """Return whether a checkout's own test script can be narrowed to specific files."""
    lowered = script_body.lower()
    if any(operator in lowered for operator in _SCRIPT_COMPOSITION_OPERATORS):
        return False
    return any(runner in lowered for runner in _PATH_ARGUMENT_TEST_RUNNERS)


def scoped_test_command(
    command: ValidationCommand, paths: Sequence[str]
) -> ValidationCommand | None:
    """Narrow one test command to the suites an attempt actually changed.

    The point is feedback, not coverage. AB-Feature-108 spent five attempts on defects in its
    own new test file -- an eslint rule, a bad `jest.mock` factory, an unmocked hook -- and
    learned about each one only after fifteen minutes of running the entire repository suite,
    which then ran out of memory before reaching the file in question. Every one of those was
    visible in seconds from the file the attempt had just written.

    Returns None whenever narrowing would change what is being asked, which is the safe
    answer: an unrecognised runner, a composed script, or no changed test files at all.
    """
    if not command.accepts_test_paths or not paths:
        return None
    working_directory = PurePosixPath(command.working_directory or ".")
    suffixes = {suffix.lower() for suffix in command.collectible_suffixes}
    scoped: list[str] = []
    for path in paths:
        candidate = PurePosixPath(path)
        # A path the runner cannot collect is not narrowing, it is a guaranteed rejection:
        # 216's `pytest -q` was handed three `.js` suites, and the failed advisory run is
        # promoted to required. An empty suffix set means the runner is unknown; every path
        # passes, which is exactly today's behaviour.
        if suffixes and candidate.suffix.lower() not in suffixes:
            continue
        if working_directory in {PurePosixPath(), PurePosixPath(".")}:
            relative = candidate
        elif working_directory == candidate or working_directory in candidate.parents:
            relative = candidate.relative_to(working_directory)
        else:
            # A changed test belonging to a different package in the same repository. Its own
            # command will cover it; adding it here would run it from the wrong directory.
            continue
        scoped.append(relative.as_posix())
    if not scoped:
        return None
    # A trailing `.` is the command saying "everything under here", so narrowing replaces it
    # rather than adding to it. Left in place, `pytest -q . one_test.py` still runs the whole
    # suite and the narrowed command would be a slower version of the one it replaces.
    base = list(command.command)
    if base and base[-1] == ".":
        base.pop()
    return command.model_copy(
        update={
            "command": [*base, *command.test_path_separator, *sorted(set(scoped))],
            # Advisory. The full suite remains the authoritative gate, so a narrowed run that
            # fails is diagnostic detail rather than the verdict, and one that passes never
            # substitutes for the command the repository actually configured.
            "required": False,
        }
    )


def _node_commands(
    root: Path,
    manager: str,
    scripts: dict[object, object],
    timeout: float,
    *,
    repository_root: Path | None = None,
) -> list[ValidationCommand]:
    commands: list[ValidationCommand] = []
    # The version this project declared, so its checks run under the same manager that
    # installed its dependencies. Running `npm ci` as one version and `npm run build` as
    # another is a difference nobody asked for and nobody would see until it mattered.
    argv, _version = package_manager_argv(manager, root, repository_root or root)
    mapping: tuple[tuple[str, ValidationType], ...] = (
        ("format:check", "format"),
        ("format", "format"),
        ("lint", "lint"),
        ("typecheck", "typecheck"),
        ("test", "test"),
        ("test:ci", "test"),
        ("build", "build"),
    )
    seen_types: set[str] = set()
    for script, validation_type in mapping:
        if (
            script not in scripts
            or not isinstance(scripts[script], str)
            or validation_type in seen_types
        ):
            continue
        body = cast(str, scripts[script])
        # A script that rewrites the files it inspects is not a check, whatever it is named.
        # AB-console-admin-2.0's `format` is `prettier --write './**/*.{js,jsx,ts,tsx,css,md,
        # json}'`, and every live feature run against that repository planned it as a required
        # validation command. It cannot fail -- writing is how it resolves what it finds -- so
        # it was a repository-wide run that could never return a verdict, executed twice per
        # attempt. And it is a validation step with write access to files the feature never
        # touched: anything in the checkout that is not already Prettier-clean is rewritten
        # into the change under review. Those two repositories happened to be clean, so no
        # such churn was observed in the audited sample; the exposure is the point.
        #
        # Not skipped silently as a type: `format:check` is preferred above and a repository
        # that declares one still gets it. `seen_types` is only marked once a command is
        # actually taken, so rejecting a writer here does not consume the slot.
        #
        # Deliberately only formatting. A `lint` script written as `eslint . --fix` also
        # rewrites, but it still exits non-zero on every rule it cannot fix, so it is a gate
        # with a verdict and dropping it would remove a check rather than a no-op.
        if validation_type == "format" and _rewrites_its_input(body):
            continue
        seen_types.add(validation_type)
        commands.append(
            ValidationCommand(
                validation_type=validation_type,
                command=[*argv, "run", script],
                required=True,
                timeout_seconds=timeout,
                accepts_test_paths=validation_type == "test" and _accepts_test_paths(body),
                collectible_suffixes=(
                    _collectible_suffixes(body) if validation_type == "test" else []
                ),
                # `npm run` forwards nothing to the script without it. pnpm accepts it and
                # yarn treats it as the separator it looks like, so one form covers all three
                # rather than encoding a per-manager table that would drift.
                test_path_separator=["--"],
                # Only for tests. Every mainstream Node test runner treats CI as the signal
                # to run once and exit, and without it a watch-mode runner such as
                # `react-scripts test` never returns. The same variable changes what a build
                # means, though: `react-scripts build` promotes warnings to errors under CI,
                # so setting it there failed a build over pre-existing warnings the change
                # never introduced.
                environment={"CI": "true"} if validation_type == "test" else {},
            )
        )
    return commands


def _python_commands(root: Path, timeout: float) -> list[ValidationCommand]:
    text = "\n".join(
        _read_text(root / name).lower()
        for name in (
            "pyproject.toml",
            "requirements.txt",
            "setup.py",
            "setup.cfg",
            "tox.ini",
            "pytest.ini",
        )
    )
    files = [path.name for path in root.rglob("*.py") if ".venv" not in path.parts]
    runner = ["uv", "run", "--frozen"] if (root / "uv.lock").is_file() else []
    commands: list[ValidationCommand] = []
    if "ruff" in text:
        commands.append(
            ValidationCommand(
                validation_type="lint",
                command=[*runner, "ruff", "check", "."],
                timeout_seconds=timeout,
            )
        )
    if "mypy" in text or "pyright" in text:
        executable = "mypy" if "mypy" in text else "pyright"
        commands.append(
            ValidationCommand(
                validation_type="typecheck",
                command=[*runner, executable, "."],
                required=False,
                timeout_seconds=timeout,
            )
        )
    has_python_tests = any(name.startswith("test_") or name.endswith("_test.py") for name in files)
    if "pytest" in text or has_python_tests:
        commands.append(
            ValidationCommand(
                validation_type="test",
                command=[*runner, "pytest", "-q", "."],
                timeout_seconds=timeout,
                # pytest takes paths directly, and the trailing `.` it is given is replaced
                # by them rather than added to, so a narrowed run is exactly the files asked
                # for.
                accepts_test_paths=True,
                collectible_suffixes=[".py"],
            )
        )
    return commands


def _node_package_manager(root: Path, *, repository_root: Path | None = None) -> str | None:
    """Resolve the nearest declared manager before falling back to npm.

    Workspace packages commonly keep their only lockfile or ``packageManager`` declaration at
    the monorepo root. Treating each nested ``package.json`` as an independent npm project runs
    the wrong executable and can even create an unintended nested lockfile.
    """
    boundary = repository_root or root
    if root != boundary and boundary not in root.parents:
        msg = "Node package root must be within the repository root"
        raise ValueError(msg)
    current = root
    while True:
        manager = _node_package_manager_at(current)
        if manager is not None:
            return manager
        if current == boundary:
            break
        current = current.parent
    return "npm" if (root / "package.json").is_file() else None


def _node_package_manager_at(root: Path) -> str | None:
    """Read one directory's lockfile or standard packageManager declaration."""
    if (root / "pnpm-lock.yaml").is_file():
        return "pnpm"
    if (root / "yarn.lock").is_file():
        return "yarn"
    if (root / "package-lock.json").is_file():
        return "npm"
    declared = _package_json(root).get("packageManager")
    if isinstance(declared, str):
        manager = declared.partition("@")[0].strip().lower()
        if manager in {"npm", "pnpm", "yarn"}:
            return manager
    return None


def _manifest_roots(root: Path, name: str) -> list[Path]:
    """Return manifest directories in deterministic order, excluding dependency caches."""
    ignored = {".git", ".venv", "node_modules", "__pycache__"}
    return sorted(
        {
            path.parent
            for path in root.rglob(name)
            if path.is_file() and not any(part in ignored for part in path.relative_to(root).parts)
        }
    )


def _python_project_roots(root: Path) -> list[Path]:
    manifests = ["pyproject.toml", "setup.py", "setup.cfg", "tox.ini", "pytest.ini"]
    roots = {directory for name in manifests for directory in _manifest_roots(root, name)}
    if not roots and any(path.is_file() for path in root.rglob("*.py")):
        roots.add(root)
    return sorted(roots)


def _with_working_directory(
    commands: list[ValidationCommand], working_directory: str
) -> list[ValidationCommand]:
    return [item.model_copy(update={"working_directory": working_directory}) for item in commands]


def _package_json(root: Path) -> dict[str, object]:
    try:
        value = json.loads((root / "package.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _resolve_workdir(root: PathLike, relative: str) -> Path:
    workspace = resolve_workspace_root(root)
    resolved = (workspace / relative).resolve(strict=False)
    if resolved != workspace and workspace not in resolved.parents:
        msg = "validation working directory must stay within the workspace"
        raise ValueError(msg)
    return resolved


def _operation_type(validation_type: ValidationType) -> ExternalOperationType:
    return {
        "format": ExternalOperationType.RUN_FORMATTER,
        "lint": ExternalOperationType.RUN_LINTER,
        "typecheck": ExternalOperationType.RUN_TYPECHECK,
        "test": ExternalOperationType.RUN_TESTS,
        "build": ExternalOperationType.RUN_BUILD,
        "custom": ExternalOperationType.INSTALL_DEPENDENCIES,
    }[validation_type]


def _not_configured_result(
    plan: RepositoryValidationPlan, revision: RepositoryRevision
) -> ValidationResult:
    return ValidationResult(
        command=(),
        return_code=None,
        stdout="",
        stderr="",
        timed_out=False,
        duration_seconds=0.0,
        validation_type="test",
        repository_id=plan.repository_id,
        repository_revision=revision.combined_fingerprint,
        working_tree_fingerprint=revision.working_tree_fingerprint,
        status=ValidationStatus.NOT_CONFIGURED,
    )


def _run_validation_command(
    workspace_root: PathLike,
    command: tuple[str, ...],
    *,
    timeout_seconds: float,
    validation_type: ValidationType = "custom",
    repository_id: str | None = None,
    revision: RepositoryRevision | None = None,
    success_exit_codes: tuple[int, ...] = (0,),
    environment: dict[str, str] | None = None,
    required: bool = True,
) -> ValidationResult:
    _validate_timeout(timeout_seconds)
    root = resolve_workspace_root(workspace_root)
    revision = revision or calculate_repository_revision_sync(root)
    started_at = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=root,
        env=_validation_environment(environment, root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        start_new_session=True,
    )
    stdout, stderr, output_truncated, timed_out = _capture_process_output(
        process, timeout_seconds=timeout_seconds, output_limit=64 * 1024
    )
    return _result(
        command=command,
        return_code=None if timed_out else process.returncode,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
        timed_out=timed_out,
        duration_seconds=time.monotonic() - started_at,
        output_truncated=output_truncated,
        validation_type=validation_type,
        repository_id=repository_id or root.name,
        revision=revision,
        success_exit_codes=success_exit_codes,
        required=required,
        repository_root=root,
    )


async def _run_validation_command_async(
    workspace_root: PathLike,
    command: tuple[str, ...],
    *,
    timeout_seconds: float,
    cancellation_token: CancellationToken,
    process_runner: ProcessRunner,
    validation_type: ValidationType = "custom",
    repository_id: str | None = None,
    revision: RepositoryRevision | None = None,
    success_exit_codes: tuple[int, ...] = (0,),
    environment: dict[str, str] | None = None,
    required: bool = True,
) -> ValidationResult:
    _validate_timeout(timeout_seconds)
    root = resolve_workspace_root(workspace_root)
    revision = revision or await calculate_repository_revision(root)
    result = await process_runner.run(
        command,
        root,
        timeout_seconds,
        cancellation_token,
        _validation_environment(environment, root),
    )
    return _result(
        command=command,
        return_code=result.return_code,
        stdout=result.stdout,
        stderr=result.stderr,
        timed_out=result.timed_out,
        duration_seconds=result.duration_seconds,
        output_truncated=result.output_truncated,
        cancelled=result.cancelled,
        validation_type=validation_type,
        repository_id=repository_id or root.name,
        revision=revision,
        success_exit_codes=success_exit_codes,
        required=required,
        repository_root=root,
    )


def _result(
    *,
    command: tuple[str, ...],
    return_code: int | None,
    stdout: str,
    stderr: str,
    timed_out: bool,
    duration_seconds: float,
    validation_type: ValidationType,
    repository_id: str,
    revision: RepositoryRevision,
    success_exit_codes: tuple[int, ...],
    required: bool,
    repository_root: Path,
    output_truncated: bool = False,
    cancelled: bool = False,
) -> ValidationResult:
    status = _status_for(
        validation_type=validation_type,
        return_code=return_code,
        timed_out=timed_out,
        cancelled=cancelled,
        command=command,
        success_exit_codes=success_exit_codes,
        output=f"{stdout}\n{stderr}",
    )
    lint_classification = None
    if validation_type == "lint" and status is ValidationStatus.FAILED:
        # The real checkout is required: whether a missing shared config is declared in
        # the repository decides between an automatic reinstall and human intervention.
        lint_classification = classify_lint_failure(
            result_stdout=stdout, result_stderr=stderr, repository_root=repository_root
        ).kind
    # Summaries are built here, where the workspace is in hand, so the files the FULL
    # output names can be resolved and appended before the tail-bound discards them --
    # see `_summary_with_references`. `__post_init__` keeps its plain tail-bound as the
    # fallback for construction paths that hold no workspace (replay, test doubles).
    failed = _output_is_diagnostic(
        status=status,
        return_code=return_code,
        timed_out=timed_out,
        cancelled=cancelled,
        success_exit_codes=success_exit_codes,
    )
    return ValidationResult(
        stdout_summary=_summary_with_references(stdout, repository_root) if failed else "",
        stderr_summary=_summary_with_references(stderr, repository_root) if failed else "",
        command=command,
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        duration_seconds=duration_seconds,
        output_truncated=output_truncated,
        cancelled=cancelled,
        validation_type=validation_type,
        repository_id=repository_id,
        repository_revision=revision.combined_fingerprint,
        working_tree_fingerprint=revision.working_tree_fingerprint,
        status=status,
        required=required,
        failure_classification=lint_classification,
    )


# What a runner prints when it ran correctly and simply had nothing to run. These are
# framework conventions, not repository details: Jest emits the first (and react-scripts,
# Vitest and Bun wrap or copy it), so matching the sentence covers every checkout that uses
# them without naming any of them.
def _bounded_summary(output: str) -> str:
    """Return the tail of a failing command's output, redacted and length-bounded.

    The tail rather than the head: every runner the platform drives prints its failures and
    its summary last, so a head-anchored excerpt captures startup banners and cuts away the
    part that names the defect.
    """
    if not output.strip():
        return ""
    redacted = redact_output(output)
    if len(redacted) <= _MAX_SUMMARY_CHARACTERS:
        return redacted
    return f"[truncated]\n{redacted[-_MAX_SUMMARY_CHARACTERS:]}"


def _workspace_reference_lines(redacted_output: str, workspace_root: Path) -> list[str]:
    """Resolve the workspace files a command's FULL output names, as `path[:line]` lines.

    Run before any bounding, so a reference near the head of a long report survives the
    tail-bound that would otherwise discard it. Only tokens that resolve to a regular file
    inside the workspace survive -- same membership discipline the engineer's
    `_diagnostic_file_locations` applies -- and files under a package store are dropped: a
    dependency's internals are not files a repair may edit. First-seen order, one entry per
    file with the first line number the output attached to it, bounded.
    """
    references: list[str] = []
    seen: set[str] = set()
    for token, line in diagnostic_file_references([redacted_output]):
        try:
            resolved = resolve_workspace_path(workspace_root, token)
        except (WorkspacePathError, ValueError, OSError):
            continue
        if not resolved.is_file():
            continue
        relative = resolved.relative_to(workspace_root).as_posix()
        if relative in seen:
            continue
        if any(segment in _PACKAGE_STORE_SEGMENTS for segment in PurePosixPath(relative).parts):
            continue
        seen.add(relative)
        references.append(f"{relative}:{line}" if line is not None else relative)
        if len(references) >= _MAX_REFERENCED_WORKSPACE_FILES:
            break
    return references


def validation_summary(result: ValidationResult) -> dict[str, Any]:
    """Retain bounded, credential-free validation evidence in prompts and artifact metadata.

    The one writer of the shape `current_validation_results` is persisted in. It lived
    beside the Reviewer, which was the only thing that ran a repository's own commands
    against a finished attempt; the commit gate now runs some of them *inside* the attempt
    and its rejections have to reach the same durable field. Two writers of one shape would
    drift, and the direction they drift is a reader -- ``_failing_required_evidence`` --
    silently seeing half the record.
    """
    return {
        "name": result.command[0] if result.command else result.validation_type,
        "command": " ".join(result.command),
        "passed": result.succeeded,
        "validation_type": result.validation_type,
        "repository_id": result.repository_id,
        "repository_revision": result.repository_revision,
        "working_tree_fingerprint": result.working_tree_fingerprint,
        "status": result.status.value if result.status is not None else "failed",
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        # Whether the record is the whole story. Without it a reader cannot tell a command
        # that failed from one this platform cut off, and AB-Feature-168's Jest run looked
        # like the former while holding none of the thirteen failures it actually reported.
        "output_truncated": result.output_truncated,
        "duration_seconds": result.duration_seconds,
        "operation_id": result.operation_id,
        "is_current": result.is_current,
        "stdout_summary": result.stdout_summary,
        "stderr_summary": result.stderr_summary,
        "result_code": result.result_code,
        "failure_classification": result.failure_classification,
        "required": result.required,
    }


# The `:line` or `:line:col` a reference line carries. Anchored at the end because the path
# in front of it may itself contain colons on a checkout that allows them.
_REFERENCE_LINE_SUFFIX = re.compile(r":\d+(?::\d+)?$")


def workspace_references_in_summary(summary: str) -> list[str]:
    """Read back the workspace files a persisted failing summary names, without line numbers.

    The inverse of the section ``_summary_with_references`` appends, and it lives here so the
    header is written and matched in one place. The reader exists because a later attempt has
    no workspace and no output -- only the durable summary -- and comparing two attempts'
    failing file sets is what tells "the same defect, reported differently" from progress.

    Line numbers are dropped deliberately. A repair that changes the file moves every line in
    the report below it, and an identity that moves with the source cannot answer whether the
    source changed anything. The last occurrence of the header wins: a failing report is free
    to contain the header's own text, and the appended section is always last.
    """
    lines = summary.splitlines()
    if _WORKSPACE_REFERENCES_HEADER not in lines:
        return []
    start = len(lines) - lines[::-1].index(_WORKSPACE_REFERENCES_HEADER)
    files: list[str] = []
    for entry in lines[start:]:
        path = _REFERENCE_LINE_SUFFIX.sub("", entry.strip())
        if path and path not in files:
            files.append(path)
    return files


def _names_only_package_store_files(line: str) -> bool:
    """Return whether every file-shaped token on this line points into a package store."""
    tokens = diagnostic_file_references([line])
    if not tokens:
        return False
    return all(
        any(segment in _PACKAGE_STORE_SEGMENTS for segment in PurePosixPath(token).parts)
        for token, _line in tokens
    )


def _bounded_workspace_tail(redacted_output: str) -> str:
    """Bound a failing report to its tail, spending the budget on workspace lines.

    When the output exceeds the budget, lines whose only file tokens point into package
    stores are dropped first -- a Jest component crash carries dozens of framework frames
    between the failure and the suite summary, and they crowd the frames that name the
    defect out of a plain tail. Lines naming no file at all are kept: error messages and
    assertion diffs are the defect's description. If filtering empties the report, fall
    back to the unfiltered tail -- a noisy record beats an empty one.
    """
    if len(redacted_output) <= _MAX_SUMMARY_CHARACTERS:
        return redacted_output
    kept = "\n".join(
        line for line in redacted_output.splitlines() if not _names_only_package_store_files(line)
    )
    if not kept.strip():
        kept = redacted_output
    return f"[truncated]\n{kept[-_MAX_SUMMARY_CHARACTERS:]}"


def _summary_with_references(output: str, workspace_root: Path) -> str:
    """Build one failing stream's summary: bounded tail plus the files the full text names.

    The reference section sits at the END deliberately. This summary is re-truncated
    downstream (the reviewer's excerpt keeps a tail of it), and every bound in the path is
    tail-anchored -- so an end-appended section survives each of them by construction.
    """
    if not output.strip():
        return ""
    redacted = redact_output(output)
    references = _workspace_reference_lines(redacted, workspace_root)
    body = _bounded_workspace_tail(redacted)
    if not references:
        return body
    section = "\n".join([_WORKSPACE_REFERENCES_HEADER, *references])
    return f"{body}\n\n{section}" if body.strip() else section


_EMPTY_TEST_SUITE_MARKERS = (
    "no tests found",
    "no test files found",
)

# How a runner states the exit code its empty suite produced, printed beside the banner above:
# Jest writes `No tests found, exiting with code 1`, and the runners that wrap or copy its
# message carry the same sentence. Read only where the banner and the exit code disagree --
# see `_found_no_tests` -- so a runner that never prints it is unaffected on the paths where
# the exit code already settles the question.
_EXIT_CODE_DECLARATION = re.compile(r"exiting with code (\d+)")


# What a runtime prints when it ran out of memory rather than finding a defect. These are
# runtime conventions, not repository details: Node names its heap, CPython raises
# MemoryError, and a process the kernel's OOM killer took reports no output at all and only
# its signal. AB-Feature-108 spent six coding retries on the first of these because a
# command that cannot complete looked exactly like a command that found a bug.
_RESOURCE_EXHAUSTION_MARKERS = (
    "javascript heap out of memory",
    "heap out of memory",
    "allocation failed - javascript heap",
    "fatal error: reached heap limit",
    "out of memory",
    "cannot allocate memory",
    "memoryerror",
    "std::bad_alloc",
    "killed process",
)

# 128 + SIGKILL(9). A shell reports a killed child this way, and the OOM killer is the
# overwhelmingly common reason a validation command is killed rather than exiting.
_SIGKILL_RETURN_CODES = frozenset({137, -9})

VALIDATION_TIMED_OUT = "validation_timed_out"
VALIDATION_RESOURCE_EXHAUSTED = "validation_resource_exhausted"

# The classifications that mean "the required command could not run to completion here", as
# opposed to "the required command completed and rejected the source".
CAPACITY_FAILURE_CLASSIFICATIONS = frozenset({VALIDATION_TIMED_OUT, VALIDATION_RESOURCE_EXHAUSTED})


def rejects_the_source(result: ValidationResult) -> bool:
    """Say whether this run completed and blamed the change, rather than failing to run.

    The predicate every in-attempt check needs before it may hand a result to a repair loop,
    and the reason it is one function: a run that timed out or exhausted the machine says
    nothing about the change -- rewriting a file cannot make a command fit in memory -- so
    admitting one would spend a coding budget on a verdict that was never available. Two
    copies of that judgement would drift, and the direction they drift is a repair pass
    burned on a machine problem.
    """
    return (
        result.status is ValidationStatus.FAILED
        and result.failure_classification not in CAPACITY_FAILURE_CLASSIFICATIONS
    )


def _capacity_classification(
    *, timed_out: bool, return_code: int | None, output: str
) -> str | None:
    """Return why a command could not complete, or None if it completed and failed.

    A timeout and an exhausted heap are not source defects: rewriting the file under test
    cannot make the whole suite fit in memory or finish inside its limit. Distinguishing
    them here is what lets the retry policy stop instead of spending a coding budget on a
    command whose verdict was never available.
    """
    if timed_out:
        return VALIDATION_TIMED_OUT
    lowered = output.lower()
    if any(marker in lowered for marker in _RESOURCE_EXHAUSTION_MARKERS):
        return VALIDATION_RESOURCE_EXHAUSTED
    if return_code in _SIGKILL_RETURN_CODES:
        return VALIDATION_RESOURCE_EXHAUSTED
    return None


def _status_for(
    *,
    validation_type: ValidationType,
    return_code: int | None,
    timed_out: bool,
    cancelled: bool,
    command: tuple[str, ...],
    success_exit_codes: tuple[int, ...] = (0,),
    output: str = "",
) -> ValidationStatus:
    if cancelled:
        return ValidationStatus.CANCELLED
    if (
        validation_type == "test"
        and not timed_out
        and _found_no_tests(
            return_code=return_code,
            command=command,
            output=output,
            success_exit_codes=success_exit_codes,
        )
    ):
        return ValidationStatus.NO_TESTS_FOUND
    if timed_out or return_code not in success_exit_codes:
        return ValidationStatus.FAILED
    return ValidationStatus.PASSED


def _found_no_tests(
    *,
    return_code: int | None,
    command: tuple[str, ...],
    output: str,
    success_exit_codes: tuple[int, ...] = (0,),
) -> bool:
    """Return whether a test runner exited only because the repository has no tests.

    A repository that has not written tests yet is not a broken checkout, and blocking on one
    refuses to implement the very feature that would add them -- which is how feature -056
    died. pytest says this in its exit code; Jest and the runners built on it exit 1 like any
    other failure and say it only in their output, so the message has to be read.

    Reading the message is not enough on its own, though, because a repository's `test` script
    may drive more than one runner. AB-Feature-225's did: Jest ran first under
    `--passWithNoTests` and printed the empty-suite banner, a second runner then failed, and
    the whole command exited 1. The banner was in the output, so the command that failed was
    recorded as `no_tests_found` -- a structural non-failure whose output is then discarded,
    which is how the only account of the real failure was lost.

    So the banner has to account for the exit code before it may override it:

    * A command that **succeeded** and says it had nothing to run is unambiguous.
    * pytest says it in its exit code, which is the same statement in another form.
    * Otherwise the runner must say which code the empty suite produced -- Jest and everything
      that wraps or copies it print `exiting with code N` beside the banner -- and that code
      must be the one the process actually returned.

    A runner that exits non-zero, announces an empty suite and declares no code is now read as
    a failure rather than as emptiness. That is the safe direction of the two: a real failure
    reported as a failure keeps its output and costs a retry, while a real failure reported as
    emptiness blocks approval with nothing to act on.
    """
    if return_code == 5 and _invokes_pytest(command):
        return True
    lowered = output.lower()
    if not any(marker in lowered for marker in _EMPTY_TEST_SUITE_MARKERS):
        return False
    if return_code in success_exit_codes:
        return True
    return any(int(declared) == return_code for declared in _EXIT_CODE_DECLARATION.findall(lowered))


def _output_is_diagnostic(
    *,
    status: ValidationStatus | None,
    return_code: int | None,
    timed_out: bool,
    cancelled: bool,
    success_exit_codes: tuple[int, ...] = (0,),
) -> bool:
    """Say whether this run's output is the only account of why it did not simply pass.

    Keyed on the exit code as well as the status, because the status is a judgement about the
    exit code and the two can disagree. ``no_tests_found`` is the case that matters: it is
    reached from a non-zero exit by both of the runner conventions `_found_no_tests` reads, and
    treating it as benign discarded the output of every one of them. AB-Feature-225's attempt 3
    is the shape -- exit 1, status ``no_tests_found``, both summaries empty, and nothing
    anywhere in the record saying what had failed, including in the journalled operation.

    A command that exited cleanly still keeps nothing. Its chatter has no diagnostic value and
    a checkout controls that text.
    """
    if status in {ValidationStatus.FAILED, ValidationStatus.CANCELLED}:
        return True
    if timed_out or cancelled:
        return True
    return return_code is not None and return_code not in success_exit_codes


def _invokes_pytest(command: tuple[str, ...]) -> bool:
    """Recognize pytest behind fixed argument-vector wrappers such as uv or ``python -m``."""
    return any(Path(argument).name in {"pytest", "pytest.exe"} for argument in command)


def _with_operation_id(result: ValidationResult, operation_id: str) -> ValidationResult:
    return replace(result, operation_id=operation_id)


def _result_payload(result: ValidationResult) -> dict[str, object]:
    return {
        "validation_type": result.validation_type,
        "command": list(result.command),
        "repository_id": result.repository_id,
        "repository_revision": result.repository_revision,
        "working_tree_fingerprint": result.working_tree_fingerprint,
        "status": result.status.value
        if result.status is not None
        else ValidationStatus.FAILED.value,
        "exit_code": result.return_code,
        # Repository output is deliberately excluded. Exit/status/classification are
        # platform-owned, actionable evidence and cannot smuggle a secret into the journal.
        "stdout_summary": "",
        "stderr_summary": "",
        "duration_seconds": result.duration_seconds,
        "output_truncated": result.output_truncated,
        "cancelled": result.cancelled,
        "required": result.required,
        "failure_classification": result.failure_classification,
        "result_code": result.result_code,
    }


def validation_failure_identity(result: ValidationResult) -> str:
    """Return a stable identity for one validation failure without storing its output."""
    return json.dumps(
        {
            "validation_type": result.validation_type,
            "command": result.command,
            "failure_classification": result.failure_classification,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _result_from_payload(
    payload: dict[str, Any],
    *,
    command: ValidationCommand,
    repository_id: str,
    revision: RepositoryRevision,
    operation_id: str,
) -> ValidationResult:
    raw_status = payload.get("status", ValidationStatus.FAILED.value)
    try:
        status = ValidationStatus(cast(str, raw_status))
    except ValueError:
        status = ValidationStatus.FAILED
    raw_command = payload.get("command", command.command)
    command_parts = (
        tuple(item for item in raw_command if isinstance(item, str))
        if isinstance(raw_command, list)
        else tuple(command.command)
    )
    return ValidationResult(
        command=command_parts,
        return_code=_optional_return_code(payload.get("exit_code", payload.get("return_code"))),
        stdout="",
        stderr="",
        timed_out=False,
        duration_seconds=_duration_seconds(payload.get("duration_seconds")),
        output_truncated=bool(payload.get("output_truncated", False)),
        cancelled=bool(payload.get("cancelled", False)),
        validation_type=command.validation_type,
        repository_id=str(payload.get("repository_id", repository_id)),
        repository_revision=str(payload.get("repository_revision", revision.combined_fingerprint)),
        working_tree_fingerprint=str(
            payload.get("working_tree_fingerprint", revision.working_tree_fingerprint)
        ),
        status=status,
        operation_id=operation_id,
        # Ignore summaries from records produced before output became non-persistable. This
        # prevents replay from reintroducing an arbitrary historical child transcript.
        stdout_summary="",
        stderr_summary="",
        required=bool(payload.get("required", command.required)),
        failure_classification=(
            str(payload["failure_classification"])
            if isinstance(payload.get("failure_classification"), str)
            else None
        ),
        result_code=(
            str(payload["result_code"]) if isinstance(payload.get("result_code"), str) else None
        ),
    )


def _validation_environment(
    extra: dict[str, str] | None = None, root: Path | None = None
) -> dict[str, str]:
    """The environment one validation command runs in, including what the repository declares.

    `root` is optional only so the handful of callers that run a command with no repository
    context keep working; where it is known, the repository's committed example environment is
    what lets a check that needs configuration run at all.
    """
    return repository_subprocess_environment(
        extra, declared=declared_validation_environment(root) if root is not None else None
    )


def _capture_process_output(
    process: subprocess.Popen[bytes], *, timeout_seconds: float, output_limit: int
) -> tuple[bytes, bytes, bool, bool]:
    assert process.stdout is not None
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    output = {
        "stdout": _HeadAndTail(output_limit),
        "stderr": _HeadAndTail(output_limit),
    }
    timed_out = False
    deadline = time.monotonic() + timeout_seconds
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _terminate_process(process)
                break
            for key, _ in selector.select(timeout=min(remaining, 0.1)):
                stream = cast(Any, key.fileobj)
                chunk = stream.read1(8192)
                if not chunk:
                    selector.unregister(stream)
                    continue
                # Kept, not counted and abandoned. This used to stop reading at the limit and
                # kill the command, so a verbose failing run was cut off before it said what
                # failed: AB-Feature-168's Jest run reported thirteen failing tests and the
                # durable record held React warnings and no verdict. Every runner this
                # platform drives prints its summary last, so the end is the part worth
                # keeping -- and the deadline above, not the byte count, is what bounds a
                # command that will not stop.
                output[key.data].feed(chunk)
            if process.poll() is not None and not selector.get_map():
                break
        if process.poll() is None:
            _terminate_process(process)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            # SIGTERM was declined or ignored. Escalate rather than leave the child holding
            # the pipes this function is about to stop reading.
            _kill_process(process)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=2)
    finally:
        selector.close()
    output_truncated = any(stream.truncated for stream in output.values())
    return output["stdout"].collected(), output["stderr"].collected(), output_truncated, timed_out


class _HeadAndTail:
    """Bounded capture that keeps both ends of a stream and drops only its middle.

    Head-only capture loses the answer. A failing command explains itself at the end -- the
    assertion, the summary line, the count of what failed -- while the beginning is a banner
    and a list of what ran. Tail-only would lose the invocation and the first error, which is
    often the one that caused the rest.

    Memory is bounded by construction rather than by stopping the command, so a run that
    talks a lot still gets to finish and say how it went.
    """

    def __init__(self, limit: int) -> None:
        """Split the budget evenly: the opening of the run, and how it ended."""
        self._head_limit = max(1, limit // 2)
        self._head = bytearray()
        self._tail: deque[int] = deque(maxlen=max(1, limit - self._head_limit))
        self.truncated = False

    def feed(self, chunk: bytes) -> None:
        """Take one read, filling the head first and rolling the rest through the tail."""
        if len(self._head) < self._head_limit:
            room = self._head_limit - len(self._head)
            self._head.extend(chunk[:room])
            chunk = chunk[room:]
        if not chunk:
            return
        if len(self._tail) + len(chunk) > (self._tail.maxlen or 0):
            self.truncated = True
        self._tail.extend(chunk)

    def collected(self) -> bytes:
        """Return what was kept, marking the gap where the middle was dropped."""
        if not self.truncated:
            return bytes(self._head) + bytes(self._tail)
        return bytes(self._head) + b"\n[output truncated]\n" + bytes(self._tail)


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Stop a command without letting the attempt to stop it become the failure.

    Every path here is a race against a child that may be exiting on its own. `killpg`
    raises EPERM once the group is gone or is no longer ours to signal, and that escaped as
    a PermissionError from the timeout path -- so a repository whose test suite hung was
    reported as a crashed validation step rather than a timeout, losing the diagnostic the
    retry needed. Stopping is best effort; the caller's timeout is the real answer.
    """
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        return
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        pass
    with suppress(ProcessLookupError, PermissionError, OSError):
        process.terminate()


def _kill_process(process: subprocess.Popen[bytes]) -> None:
    """Send SIGKILL to a child that declined to terminate, ignoring every exit race."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
        return
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        pass
    with suppress(ProcessLookupError, PermissionError, OSError):
        process.kill()


def _validate_timeout(timeout_seconds: float) -> None:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        msg = "timeout_seconds must be finite and greater than zero"
        raise ValueError(msg)


def _validation_result_code(result: ValidationResult) -> str:
    """Return a stable, child-output-free diagnostic code for review and persistence."""
    if result.status is ValidationStatus.NOT_CONFIGURED:
        return "TEST_COMMAND_NOT_CONFIGURED"
    if result.status is ValidationStatus.NO_TESTS_FOUND:
        return "NO_TESTS_FOUND"
    return subprocess_result_code(
        return_code=result.return_code,
        timed_out=result.timed_out,
        cancelled=result.cancelled,
        prefix=f"{result.validation_type}_validation",
    )


def _optional_return_code(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _duration_seconds(value: object) -> float:
    return float(value) if isinstance(value, (float, int)) else 0.0


__all__ = [
    "CAPACITY_FAILURE_CLASSIFICATIONS",
    "VALIDATION_RESOURCE_EXHAUSTED",
    "VALIDATION_TIMED_OUT",
    "DefaultValidationPlanBuilder",
    "MockValidationPlanBuilder",
    "PinnedValidationPlanBuilder",
    "RepositoryValidationPlan",
    "ValidationCommand",
    "ValidationPlanBuilder",
    "ValidationResult",
    "ValidationStatus",
    "ValidationTool",
    "WorkspaceValidationTools",
    "rejects_the_source",
    "scoped_test_command",
    "validation_failure_identity",
    "validation_summary",
    "run_pytest",
    "run_pytest_async",
    "run_ruff",
    "run_ruff_async",
    "workspace_references_in_summary",
]
