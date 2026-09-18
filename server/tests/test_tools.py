"""Tests for workspace-bounded file, repository, and validation tools."""

import os
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from adapters.git_adapter import GitAdapterError, GitSafetyError, _validate_commit_files
from adapters.interruptible_git import _require_success
from agents.reviewer.agent import _sensitive_review_path
from agents.shared.contracts import create_artifact
from artifacts.schemas import CodeCompletionArtifact, RepositoryWorkstreamPlan
from services.feature_runtime import _log_validation_plan_drift, _pinned_validation_plan
from state.enums import ChildWorkflowStatus
from state.external_operations import ExternalOperationType
from state.feature_models import ChildWorkflowReference
from tools.file_tools import (
    PatchApplicationError,
    WorkspacePathError,
    carries_key_material,
    is_credential_shaped_path,
    is_model_safe_context_path,
    list_files,
    patch_file,
    read_file,
    write_file,
)
from tools.implementation_completeness import (
    classify_file_change,
    validate_implementation_completeness,
)
from tools.repo_tools import find_files, inspect_python_dependencies, scan_directory
from tools.technology_detection import calculate_repository_revision_sync
from tools.validation_tools import (
    _WORKSPACE_REFERENCES_HEADER,
    DefaultValidationPlanBuilder,
    PinnedValidationPlanBuilder,
    ValidationCommand,
    ValidationResult,
    ValidationStatus,
    WorkspaceValidationTools,
    _HeadAndTail,
    _result,
    _summary_with_references,
    _workspace_validation_commands,
    run_pytest,
    run_ruff,
    scoped_test_command,
)


def test_file_tools_read_write_patch_and_list_utf8_content(tmp_path: Path) -> None:
    """File operations preserve UTF-8 content and return workspace-relative paths."""
    written_path = write_file(tmp_path, "nested/note.txt", "café\nversion = 1\n")

    assert written_path == Path("nested/note.txt")
    assert read_file(tmp_path, written_path) == "café\nversion = 1\n"

    patched_path = patch_file(tmp_path, written_path, "version = 1", "version = 2")

    assert patched_path == written_path
    assert read_file(tmp_path, written_path) == "café\nversion = 2\n"
    assert list_files(tmp_path) == [Path("nested/note.txt")]


def test_file_tools_reject_traversal_and_symlink_escape(tmp_path: Path) -> None:
    """Paths outside the workspace, including through symlinks, are never accessible."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")
    (workspace / "outside-link").symlink_to(outside_file)

    with pytest.raises(WorkspacePathError, match="escapes workspace"):
        read_file(workspace, "../outside.txt")
    with pytest.raises(WorkspacePathError, match="escapes workspace"):
        write_file(workspace, "../outside.txt", "attempted overwrite")
    with pytest.raises(WorkspacePathError, match="escapes workspace"):
        read_file(workspace, "outside-link")
    with pytest.raises(WorkspacePathError, match="stay within"):
        list_files(workspace, pattern="../*")


def test_file_tools_enforce_a_configurable_byte_limit(tmp_path: Path) -> None:
    """Untrusted repository content cannot consume arbitrary worker memory or disk space."""
    with pytest.raises(ValueError, match="size limit"):
        write_file(tmp_path, "large.txt", "12345", max_file_bytes=4)

    write_file(tmp_path, "large.txt", "12345")
    with pytest.raises(ValueError, match="size limit"):
        read_file(tmp_path, "large.txt", max_file_bytes=4)


# Every path the two rules have to agree about, and one row per reason they might not.
# `sendgrid.env` and the three service-account names are the real AB-Feature-170 files,
# kept verbatim because a generic stand-in cannot prove the incident is closed.
_KEY_MATERIAL_PATHS = (
    ".env",
    ".env.production",
    "sendgrid.env",
    "config/prod.env",
    "id_rsa",
    "deploy/server.pem",
    "certs/client.p12",
    "secrets.yaml",
    "config/credentials.json",
)
_ORDINARY_PATHS = (
    ".env.example",
    ".env.sample",
    "secrets.example.yaml",
    "package.json",
    "server/validation/app.validation.js",
    "imls-1566290608647-c1739c763650.json",
    "leafy-respect-324311-dadb978a83e7.json",
    "shamir1997-7d237-128001e15e13.json",
    "src/secretRotation.ts",
)


@pytest.mark.parametrize("path", [*_KEY_MATERIAL_PATHS, *_ORDINARY_PATHS])
def test_the_reviewer_and_engineer_cannot_disagree_about_what_is_key_material(path: str) -> None:
    """One rule, enforced. Two copies drift, and the direction they drift is a leak.

    The reviewer keeps its own *policy* -- a credential-shaped file the change itself touched
    is evidence, so it is redacted rather than dropped -- but not its own answer. This fails
    the moment `_sensitive_review_path` grows an implementation again instead of delegating.
    """
    expected = is_credential_shaped_path(PurePosixPath(path))

    assert _sensitive_review_path(Path(path)) is expected
    assert is_model_safe_context_path(PurePosixPath(path)) is not expected


@pytest.mark.parametrize("path", [*_KEY_MATERIAL_PATHS, *_ORDINARY_PATHS])
def test_the_commit_gate_and_the_context_rule_cannot_disagree_about_key_material(
    path: str, tmp_path: Path
) -> None:
    """The third copy of the rule, held against the same corpus as the other two.

    This one is the worst place to drift. A credential sent to a provider goes to one party
    under a contract; a credential committed to a branch lands in a pull request, in the
    repository's history, and in every clone of it.

    Asserted on the commit being refused rather than on `_is_sensitive_path`, because the
    private predicate answering correctly is not what keeps a key out of a branch. The
    policies still differ -- context drops one file, the commit gate refuses the whole
    commit -- but the answer may not, and this fails the moment the gate grows its own
    implementation again instead of delegating.
    """
    is_key_material = is_credential_shaped_path(PurePosixPath(path))
    staged = tmp_path / path
    staged.parent.mkdir(parents=True, exist_ok=True)
    # Placeholder bytes throughout, so this row isolates the *path* answer. The three
    # service-account names among the ordinary paths are committable by name and refused by
    # content; the content half is asserted where the bytes are real.
    staged.write_text("PLACEHOLDER=value\n", encoding="utf-8")

    if is_key_material:
        with pytest.raises(GitSafetyError, match="secret-like"):
            _validate_commit_files(tmp_path, [path])
    else:
        assert _validate_commit_files(tmp_path, [path]) == [path]


def test_key_material_is_recognized_by_path_and_by_content_together() -> None:
    """The three service-account names above are why a path rule alone is not enough.

    They are the real files AB-Feature-170 sent, and nothing about them is decidable from
    the name -- which is exactly why the row above asserts they are model-safe *paths*. What
    withholds them is their bytes.
    """
    assert carries_key_material('{\n  "type": "service_account",\n  "project_id": "x"\n}\n')
    assert carries_key_material("-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n")
    assert carries_key_material("-----BEGIN OPENSSH PRIVATE KEY-----\nAAAA\n")
    assert carries_key_material("key = '-----BEGIN PGP PRIVATE KEY BLOCK-----'\n")

    assert not carries_key_material('{"name": "package", "type": "module"}\n')
    assert not carries_key_material("MAILER_API_KEY=''\n")
    # Bounded on purpose: an unbounded scan over every context candidate is a denial-of-
    # service surface, and a key at byte 400,000 of a minified bundle is not this rule's case.
    assert not carries_key_material(f"{'x' * 20_000}\n-----BEGIN PRIVATE KEY-----\n")


def test_patch_file_requires_an_unambiguous_expected_fragment(tmp_path: Path) -> None:
    """Text patches fail instead of making an accidental broad replacement."""
    write_file(tmp_path, "settings.txt", "enabled=true\nenabled=true\n")

    with pytest.raises(PatchApplicationError, match="expected 1 replacement"):
        patch_file(tmp_path, "settings.txt", "enabled=true", "enabled=false")


def test_repository_tools_scan_search_and_inspect_python_imports(tmp_path: Path) -> None:
    """Repository inspection is deterministic and excludes tooling internals by default."""
    write_file(
        tmp_path,
        "app/module.py",
        "import os\nfrom package.client import Client\nfrom .local import helper\n",
    )
    write_file(tmp_path, "app/local.py", "VALUE = 1\n")
    write_file(tmp_path, ".git/config", "[core]\n")

    scan = scan_directory(tmp_path)
    dependencies = inspect_python_dependencies(tmp_path)

    assert scan.files == (Path("app/local.py"), Path("app/module.py"))
    assert scan.directories == (Path("app"),)
    assert scan.total_size_bytes > 0
    assert find_files(tmp_path, ("**/*.py",)) == [Path("app/local.py"), Path("app/module.py")]
    assert dependencies[Path("app/module.py")] == (".local", "os", "package.client")


def test_validation_tools_run_workspace_local_ruff_and_pytest(tmp_path: Path) -> None:
    """Validation wrappers run fixed commands with captured output and explicit timeouts."""
    write_file(tmp_path, "module.py", "value = 1\n")
    write_file(
        tmp_path,
        "test_module.py",
        "def test_value() -> None:\n    assert 1 + 1 == 2\n",
    )

    ruff_result = run_ruff(tmp_path, timeout_seconds=10)
    pytest_result = run_pytest(tmp_path, timeout_seconds=10)

    assert ruff_result.command == ("ruff", "check", ".")
    assert ruff_result.succeeded
    assert not ruff_result.timed_out
    assert isinstance(ruff_result.stdout, str)
    assert isinstance(ruff_result.stderr, str)
    assert pytest_result.command == ("pytest", "-q", ".")
    assert pytest_result.succeeded
    assert "1 passed" in pytest_result.stdout


@pytest.mark.asyncio
async def test_validation_tools_run_only_configured_lint_for_baseline(tmp_path: Path) -> None:
    """Baseline lint execution returns concrete results rather than an async generator."""
    write_file(
        tmp_path,
        "package.json",
        '{"scripts":{"lint":"node -e \\"process.exit(0)\\""}}\n',
    )

    results = await WorkspaceValidationTools(tmp_path).run_lint_checks_async(timeout_seconds=10)

    assert len(results) == 1
    assert results[0].validation_type == "lint"
    assert results[0].succeeded


def test_validation_commands_use_npm_for_a_node_only_repository(tmp_path: Path) -> None:
    """A frontend repository is not incorrectly forced through Python validation."""
    write_file(tmp_path, "package.json", '{"scripts":{"lint":"eslint .","test":"vitest run"}}')
    write_file(tmp_path, "package-lock.json", '{"lockfileVersion":3}')

    commands = _workspace_validation_commands(tmp_path)

    assert commands == (
        (("npm", "run", "lint"), ExternalOperationType.RUN_LINTER),
        (("npm", "run", "test"), ExternalOperationType.RUN_TESTS),
    )


def test_validation_plan_uses_javascript_scripts_without_python_commands(tmp_path: Path) -> None:
    """A JavaScript backend uses exactly configured npm scripts, never Python defaults."""
    write_file(tmp_path, "src/server.js", "export const status = () => 'ok';\n")
    write_file(
        tmp_path,
        "package.json",
        '{"scripts":{"lint":"eslint .","test":"jest --runInBand","build":"node build.js"}}',
    )
    write_file(tmp_path, "package-lock.json", '{"lockfileVersion":3}')

    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)

    assert plan.technology_profile.primary_language == "JavaScript"
    assert [item.command for item in plan.commands] == [
        ["npm", "run", "lint"],
        ["npm", "run", "test"],
        ["npm", "run", "build"],
    ]
    assert all(item.command[0] not in {"ruff", "pytest"} for item in plan.commands)


def test_a_checkouts_own_runner_decides_whether_tests_can_be_narrowed(tmp_path: Path) -> None:
    """Read off the script the repository wrote, never assumed from the file layout.

    AB-Feature-108's console runs `react-scripts test` and its backend runs `jest --ci`.
    Both take the files to run as arguments, which is the only reason narrowing is possible
    at all -- and a repository whose script chains two commands must be left alone.
    """
    write_file(tmp_path, "src/App.jsx", "export const App = () => null;\n")
    write_file(tmp_path, "package.json", '{"scripts":{"test":"react-scripts test"}}')
    write_file(tmp_path, "package-lock.json", '{"lockfileVersion":3}')

    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)
    test_command = next(item for item in plan.commands if item.validation_type == "test")

    assert test_command.accepts_test_paths is True
    narrowed = scoped_test_command(test_command, ["src/pages/AllApps.test.js"])
    assert narrowed is not None
    # `npm run` forwards nothing to the script without the separator.
    assert narrowed.command == ["npm", "run", "test", "--", "src/pages/AllApps.test.js"]
    # Advisory: the repository's own full command still decides the verdict.
    assert narrowed.required is False


def test_a_composed_test_script_is_never_narrowed(tmp_path: Path) -> None:
    """Trailing arguments would reach only whichever half ran last.

    That would quietly run something other than what was asked for, which is worse than
    running everything. Refusing to narrow is always safe.
    """
    write_file(tmp_path, "src/App.jsx", "export const App = () => null;\n")
    write_file(
        tmp_path,
        "package.json",
        '{"scripts":{"test":"npm run test:unit && npm run test:e2e"}}',
    )
    write_file(tmp_path, "package-lock.json", '{"lockfileVersion":3}')

    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)
    test_command = next(item for item in plan.commands if item.validation_type == "test")

    assert test_command.accepts_test_paths is False
    assert scoped_test_command(test_command, ["src/pages/AllApps.test.js"]) is None


def test_narrowing_pytest_replaces_the_directory_it_was_given(tmp_path: Path) -> None:
    """`pytest -q . one_test.py` still runs everything, only more slowly."""
    write_file(tmp_path, "pyproject.toml", '[project]\nname="svc"\ndependencies=["pytest"]\n')
    write_file(tmp_path, "tests/test_login.py", "def test_ok() -> None:\n    assert True\n")

    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)
    test_command = next(item for item in plan.commands if item.validation_type == "test")
    narrowed = scoped_test_command(test_command, ["tests/test_login.py"])

    assert narrowed is not None
    assert "." not in narrowed.command
    assert narrowed.command[-1] == "tests/test_login.py"


def test_narrowing_hands_a_runner_only_paths_it_can_collect(tmp_path: Path) -> None:
    """216's cross-language appends, both ways: pytest got `.js` suites, jest got a `.py`.

    A path the runner cannot collect is not narrowing, it is a guaranteed rejection -- and a
    failed narrowed run is promoted to required, so the mismatch fails the attempt over an
    impossibility the change never created.
    """
    write_file(tmp_path, "pyproject.toml", '[project]\nname="svc"\ndependencies=["pytest"]\n')
    write_file(tmp_path, "server/test_export.py", "def test_ok() -> None:\n    assert True\n")
    write_file(tmp_path, "package.json", '{"scripts":{"test":"jest --ci"}}')
    write_file(tmp_path, "package-lock.json", '{"lockfileVersion":3}')

    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)
    pytest_command = next(item for item in plan.commands if item.command[-2:] == ["-q", "."])
    jest_command = next(item for item in plan.commands if item.command[:2] == ["npm", "run"])
    changed = [
        "server/test_export.py",
        "tests/export.controller.test.js",
        "tests/export.route.test.js",
    ]

    narrowed_pytest = scoped_test_command(pytest_command, changed)
    narrowed_jest = scoped_test_command(jest_command, changed)
    js_only_pytest = scoped_test_command(pytest_command, changed[1:])

    assert narrowed_pytest is not None
    assert narrowed_pytest.command[-1] == "server/test_export.py"
    assert not any(item.endswith(".js") for item in narrowed_pytest.command)
    assert narrowed_jest is not None
    assert not any(item.endswith(".py") for item in narrowed_jest.command)
    # Nothing the runner can collect means no narrowed command at all -- the documented
    # "narrowing would change what is being asked" answer, never a command that must fail.
    assert js_only_pytest is None


def test_an_unknown_runner_keeps_every_changed_test_path() -> None:
    """No suffix table entry means no opinion: today's behaviour, and every stored plan's."""
    command = ValidationCommand(
        validation_type="test",
        command=["make", "check"],
        timeout_seconds=60.0,
        accepts_test_paths=True,
    )

    narrowed = scoped_test_command(command, ["a.test.js", "b_test.py"])

    assert narrowed is not None
    assert narrowed.command[-2:] == ["a.test.js", "b_test.py"]


def test_the_workstreams_plan_is_pinned_at_its_baseline(tmp_path: Path) -> None:
    """A file the agent writes must not mint a validation command (83-, AB-Feature-216).

    The pilot repository's shape: JavaScript tooling, a population of `.py` utility files,
    no `test_*.py` and no Python manifest. Adding the repository's first `test_*.py` makes a
    fresh derivation plan a required `pytest -q .`; the pinned builder keeps answering with
    the baseline plan, and the persisted round-trip is what a retry and a resume read.
    """
    write_file(tmp_path, "src/server.js", "export const status = () => 'ok';\n")
    write_file(tmp_path, "src/util/report.py", "def report():\n    return 'ok'\n")
    write_file(tmp_path, "package.json", '{"scripts":{"lint":"eslint .","test":"jest --ci"}}')
    write_file(tmp_path, "package-lock.json", '{"lockfileVersion":3}')

    baseline = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)
    assert all(item.command[0] != "pytest" for item in baseline.commands)

    # The agent's own test file appears, exactly as 216's attempt 0 wrote one.
    write_file(tmp_path, "src/util/test_report.py", "def test_report():\n    assert True\n")
    drifted = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)
    assert any(item.command[0] == "pytest" and item.required for item in drifted.commands)

    pinned = PinnedValidationPlanBuilder(baseline)
    assert pinned.build_plan_sync(tmp_path) is baseline

    # The persisted round-trip: the child row's JSON parses back to the same command set,
    # and anything malformed answers None so the caller derives fresh instead of failing.
    child = _child_reference(validation_plan=baseline.model_dump(mode="json"))
    restored = _pinned_validation_plan(child)
    assert restored is not None
    assert [item.command for item in restored.commands] == [
        item.command for item in baseline.commands
    ]
    assert _pinned_validation_plan(_child_reference(validation_plan=None)) is None
    assert _pinned_validation_plan(_child_reference(validation_plan={"commands": "wrong"})) is None


def test_plan_drift_is_logged_and_never_becomes_a_command(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A late derivation's extra command is a record, not a gate."""
    write_file(tmp_path, "src/server.js", "export const status = () => 'ok';\n")
    write_file(tmp_path, "package.json", '{"scripts":{"lint":"eslint ."}}')
    write_file(tmp_path, "package-lock.json", '{"lockfileVersion":3}')
    baseline = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)
    write_file(tmp_path, "src/test_minted.py", "def test_minted():\n    assert True\n")
    drifted = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)

    with caplog.at_level("INFO"):
        _log_validation_plan_drift(
            baseline, drifted, feature_id="feature-x", repository_id="repo-x", attempt=1
        )
        _log_validation_plan_drift(
            baseline, baseline, feature_id="feature-x", repository_id="repo-x", attempt=1
        )

    drift_records = [item for item in caplog.records if item.message == "validation_plan_drift"]
    assert len(drift_records) == 1
    assert any("pytest" in command for command in drift_records[0].commands)  # type: ignore[attr-defined]


def _child_reference(*, validation_plan: dict[str, object] | None) -> ChildWorkflowReference:
    """Build the minimal durable child row the pin is read from."""
    return ChildWorkflowReference(
        child_workflow_id="feature-x:repo-x",
        repository_id="repo-x",
        workstream_id="ws-x",
        status=ChildWorkflowStatus.RUNNING,
        branch_name="ai/feature-x",
        workspace_path="/workspaces/feature-x/repo-x",
        retry_count=1,
        validation_plan=validation_plan,
    )


def test_narrowing_ignores_tests_belonging_to_another_package(tmp_path: Path) -> None:
    """Each package's own command covers its own tests, and runs from its own directory.

    Passing a sibling package's path would ask one runner to load a file it cannot resolve.
    """
    command = ValidationCommand(
        validation_type="test",
        command=["npm", "run", "test"],
        working_directory="packages/web",
        timeout_seconds=900,
        accepts_test_paths=True,
        test_path_separator=["--"],
    )

    narrowed = scoped_test_command(
        command, ["packages/web/src/App.test.js", "packages/api/src/routes.test.js"]
    )

    assert narrowed is not None
    # Rebased onto the command's own working directory, and the foreign package dropped.
    assert narrowed.command == ["npm", "run", "test", "--", "src/App.test.js"]


def test_an_attempt_that_changed_no_tests_runs_the_configured_command_unchanged() -> None:
    """Narrowing exists to speed up feedback, not to reduce what is checked."""
    command = ValidationCommand(
        validation_type="test",
        command=["npm", "run", "test"],
        timeout_seconds=900,
        accepts_test_paths=True,
        test_path_separator=["--"],
    )

    assert scoped_test_command(command, []) is None


@pytest.mark.asyncio
async def test_a_failing_narrowed_run_spares_the_whole_repository_suite(tmp_path: Path) -> None:
    """The saving that motivated this, measured as the effect rather than the intent.

    AB-Feature-108's attempts 3 to 7 each failed on a defect in the one test file the
    attempt had just written -- an eslint rule, a bad `jest.mock` factory, an unmocked hook
    -- and each learned about it only after fifteen minutes of running the entire suite,
    which then exhausted the heap before it ever reached that file. Once the change's own
    suites have rejected it, the whole-repository run can only agree, far more slowly.
    """
    # Contains `jest`, so the runner is recognised as one that takes file arguments; it is
    # really a script this test controls, which records how it was invoked.
    write_file(tmp_path, "package.json", '{"scripts":{"test":"node ./jest-stub.js"}}')
    write_file(tmp_path, "package-lock.json", '{"lockfileVersion":3}')
    write_file(
        tmp_path,
        "jest-stub.js",
        "const fs = require('fs');\n"
        "const args = process.argv.slice(2);\n"
        "fs.appendFileSync('invocations.txt', JSON.stringify(args) + '\\n');\n"
        "process.exit(args.length ? 1 : 0);\n",
    )
    write_file(tmp_path, "src/App.test.js", "test('x', () => {});\n")

    results = await WorkspaceValidationTools(
        tmp_path, changed_test_paths=["src/App.test.js"]
    ).run_validations_async(timeout_seconds=60)

    invocations = [line for line in (tmp_path / "invocations.txt").read_text().splitlines() if line]
    # Exactly one run, and it was the narrowed one. The full suite was never started.
    assert invocations == ['["src/App.test.js"]']
    test_results = [item for item in results if item.validation_type == "test"]
    assert len(test_results) == 1
    assert test_results[0].status is ValidationStatus.FAILED
    # Promoted to required: the repository's own runner rejected the source, and however few
    # files it was given that is a real failure rather than advisory detail.
    assert test_results[0].required is True


@pytest.mark.asyncio
async def test_a_passing_narrowed_run_still_defers_to_the_configured_command(
    tmp_path: Path,
) -> None:
    """Narrowing speeds up rejection; it must never stand in for the repository's own gate."""
    write_file(tmp_path, "package.json", '{"scripts":{"test":"node ./jest-stub.js"}}')
    write_file(tmp_path, "package-lock.json", '{"lockfileVersion":3}')
    write_file(
        tmp_path,
        "jest-stub.js",
        "const fs = require('fs');\n"
        "fs.appendFileSync('invocations.txt', JSON.stringify(process.argv.slice(2)) + '\\n');\n"
        "process.exit(0);\n",
    )
    write_file(tmp_path, "src/App.test.js", "test('x', () => {});\n")

    results = await WorkspaceValidationTools(
        tmp_path, changed_test_paths=["src/App.test.js"]
    ).run_validations_async(timeout_seconds=60)

    invocations = [line for line in (tmp_path / "invocations.txt").read_text().splitlines() if line]
    assert invocations == ['["src/App.test.js"]', "[]"], "the full suite must still run"
    required = [item for item in results if item.validation_type == "test" and item.required]
    assert len(required) == 1, "only the configured command carries the verdict"


def test_validation_plan_detects_typescript_and_configured_scripts(tmp_path: Path) -> None:
    """A TypeScript frontend retains its lint, typecheck, test, and build scripts."""
    write_file(tmp_path, "src/App.tsx", "export const App = () => null;\n")
    write_file(tmp_path, "tsconfig.json", "{}")
    write_file(
        tmp_path,
        "package.json",
        (
            '{"scripts":{"lint":"eslint .","typecheck":"tsc --noEmit",'
            '"test":"vitest run","build":"vite build"}}'
        ),
    )

    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)

    assert "TypeScript" in plan.technology_profile.languages
    assert [item.validation_type for item in plan.commands] == [
        "lint",
        "typecheck",
        "test",
        "build",
    ]


def test_validation_plan_uses_configured_python_tools_only(tmp_path: Path) -> None:
    """Python validation follows explicit project tool configuration."""
    write_file(
        tmp_path,
        "pyproject.toml",
        "[tool.ruff]\n[tool.mypy]\n[tool.pytest.ini_options]\n",
    )
    write_file(tmp_path, "app.py", "VALUE = 1\n")
    write_file(tmp_path, "test_app.py", "def test_value():\n    assert True\n")

    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)

    assert plan.technology_profile.primary_language == "Python"
    assert [item.command for item in plan.commands] == [
        ["ruff", "check", "."],
        ["mypy", "."],
        ["pytest", "-q", "."],
    ]


def test_validation_plan_uses_the_checked_in_uv_environment_when_available(tmp_path: Path) -> None:
    """A locked Python project runs its own declared environment rather than host tooling."""
    write_file(tmp_path, "pyproject.toml", "[tool.ruff]\n[tool.pytest.ini_options]\n")
    write_file(tmp_path, "uv.lock", "version = 1\n")
    write_file(tmp_path, "application/status.py", "VALUE = 1\n")
    write_file(tmp_path, "checks/test_status.py", "def test_status():\n    assert True\n")

    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)

    assert [item.command for item in plan.commands] == [
        ["uv", "run", "--frozen", "ruff", "check", "."],
        ["uv", "run", "--frozen", "pytest", "-q", "."],
    ]


def test_nested_node_manifest_inherits_the_repository_pnpm_lock(tmp_path: Path) -> None:
    """A workspace package uses the root manager instead of silently invoking npm."""
    write_file(tmp_path, "package.json", '{"name":"workspace","private":true}')
    write_file(tmp_path, "pnpm-lock.yaml", "lockfileVersion: '9.0'\n")
    write_file(
        tmp_path,
        "packages/web/package.json",
        '{"name":"web","scripts":{"lint":"eslint .","test":"vitest run"}}',
    )

    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)

    nested = [item for item in plan.commands if item.working_directory == "packages/web"]
    assert [item.command for item in nested] == [
        ["pnpm", "run", "lint"],
        ["pnpm", "run", "test"],
    ]


def test_nested_node_manifest_inherits_the_nearest_package_manager_declaration(
    tmp_path: Path,
) -> None:
    """Corepack's ancestor packageManager field is authoritative even without a lockfile.

    The inherited *version* is what the command carries, which asserts the inheritance more
    precisely than the manager name alone did: a nested package with no declaration of its own
    still runs the Yarn its monorepo root pinned, not the one the image happens to ship.
    """
    write_file(
        tmp_path,
        "package.json",
        '{"name":"workspace","private":true,"packageManager":"yarn@4.6.0"}',
    )
    write_file(
        tmp_path,
        "packages/web/package.json",
        '{"name":"web","scripts":{"build":"vite build"}}',
    )

    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)

    command = next(item for item in plan.commands if item.working_directory == "packages/web")
    assert command.command == ["corepack", "yarn@4.6.0", "run", "build"]


def test_mixed_repository_validation_is_scoped_to_each_manifest_directory(tmp_path: Path) -> None:
    """Mixed checkouts retain a per-subproject working directory for native commands."""
    write_file(tmp_path, "api/pyproject.toml", "[tool.ruff]\n")
    write_file(tmp_path, "api/main.py", "VALUE = 1\n")
    write_file(tmp_path, "web/package.json", '{"scripts":{"lint":"eslint ."}}')
    write_file(tmp_path, "web/src/index.js", "console.log('ok');\n")

    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)

    assert set(plan.technology_profile.languages) >= {"Python", "JavaScript"}
    assert {(item.validation_type, item.working_directory) for item in plan.commands} == {
        ("lint", "api"),
        ("lint", "web"),
    }


def test_repository_revision_changes_for_uncommitted_source_content(tmp_path: Path) -> None:
    """A workspace without a commit still gets a content-sensitive validation revision."""
    write_file(tmp_path, "module.js", "export const version = 1;\n")
    first = calculate_repository_revision_sync(tmp_path)
    write_file(tmp_path, "module.js", "export const version = 2;\n")
    second = calculate_repository_revision_sync(tmp_path)

    assert first.combined_fingerprint != second.combined_fingerprint
    assert first.working_tree_fingerprint != second.working_tree_fingerprint


def test_pytest_exit_code_five_is_structural_no_tests_not_success() -> None:
    """No collected pytest tests remains distinct from a passed or failed suite."""
    result = ValidationResult(
        command=("pytest", "-q", "."),
        return_code=5,
        stdout="no tests ran",
        stderr="",
        timed_out=False,
        duration_seconds=0.1,
        validation_type="test",
    )

    assert result.status is ValidationStatus.NO_TESTS_FOUND
    assert not result.succeeded


def test_wrapped_pytest_exit_code_five_is_structural_no_tests_not_failure() -> None:
    """A locked uv project keeps pytest's exit semantics through the wrapper argv."""
    result = ValidationResult(
        command=("uv", "run", "--frozen", "pytest", "-q", "."),
        return_code=5,
        stdout="no tests ran",
        stderr="",
        timed_out=False,
        duration_seconds=0.1,
        validation_type="test",
    )

    assert result.status is ValidationStatus.NO_TESTS_FOUND
    assert not result.succeeded


def test_a_jest_suite_with_no_tests_is_structural_not_a_broken_checkout() -> None:
    """Jest reports an empty suite as exit 1, so only its output distinguishes the two.

    Feature -056 was refused on this: a frontend repository that had not written tests yet
    was classified as a checkout too broken to implement against, which blocks precisely the
    features that would add the first test.
    """
    result = ValidationResult(
        command=("npm", "run", "test"),
        return_code=1,
        stdout=(
            "No tests found, exiting with code 1\n"
            "Run with `--passWithNoTests` to exit with code 0\n"
        ),
        stderr="",
        timed_out=False,
        duration_seconds=0.1,
        validation_type="test",
    )

    assert result.status is ValidationStatus.NO_TESTS_FOUND
    assert not result.succeeded


def test_a_failing_jest_suite_is_still_a_failure() -> None:
    """The empty-suite signal must not swallow a suite that ran and had tests fail."""
    result = ValidationResult(
        command=("npm", "run", "test"),
        return_code=1,
        stdout="Tests:       2 failed, 5 passed, 7 total\n",
        stderr="",
        timed_out=False,
        duration_seconds=0.1,
        validation_type="test",
    )

    assert result.status is ValidationStatus.FAILED


def test_an_empty_suite_banner_does_not_excuse_a_later_runner_failure() -> None:
    """AB-Feature-225: one `test` script, two runners, and the second one is what failed.

    Jest ran first under `--passWithNoTests`, printed the empty-suite banner and declared exit
    0; a browser runner then failed and the whole command exited 1. Reading the banner alone
    recorded the failing command as a structural non-failure, whose output is discarded -- so
    the only account of the real failure was lost, three review cycles ran against a record
    that said nothing had gone wrong, and the workstream stopped with its budget unspent.
    """
    result = ValidationResult(
        command=("npm", "run", "test"),
        return_code=1,
        stdout=(
            "No tests found, exiting with code 0\n"
            "Installing dependencies...\n"
            "Switching to root user to install dependencies...\n"
            "Failed to install browsers\n"
        ),
        stderr="Password: Authentication failure\n",
        timed_out=False,
        duration_seconds=20.5,
        validation_type="test",
    )

    assert result.status is ValidationStatus.FAILED
    # And the account of it survives, which is the half of this that made the run undiagnosable.
    assert "failed to install browsers" in result.stdout_summary.lower()


def test_an_empty_suite_that_accounts_for_its_exit_code_keeps_its_output() -> None:
    """A genuine empty suite is still structural, and no longer arrives with nothing to read."""
    result = ValidationResult(
        command=("uv", "run", "--frozen", "pytest", "-q", "."),
        return_code=5,
        stdout="no tests ran in 0.01s\n",
        stderr="",
        timed_out=False,
        duration_seconds=0.1,
        validation_type="test",
    )

    assert result.status is ValidationStatus.NO_TESTS_FOUND
    assert "no tests ran" in result.stdout_summary


def test_a_passing_command_still_keeps_none_of_its_output() -> None:
    """Retention is keyed on the exit code, not widened to every command."""
    result = ValidationResult(
        command=("npm", "run", "build"),
        return_code=0,
        stdout="compiled successfully\n",
        stderr="",
        timed_out=False,
        duration_seconds=1.0,
        validation_type="build",
    )

    assert result.status is ValidationStatus.PASSED
    assert result.stdout_summary == ""


def test_a_test_runner_that_timed_out_is_never_read_as_an_empty_suite() -> None:
    """A killed runner may have printed anything; a deadline is a failure regardless."""
    result = ValidationResult(
        command=("npm", "run", "test"),
        return_code=None,
        stdout="No tests found",
        stderr="",
        timed_out=True,
        duration_seconds=900.0,
        validation_type="test",
    )

    assert result.status is ValidationStatus.FAILED


def test_validation_tools_require_positive_timeouts(tmp_path: Path) -> None:
    """Every validation invocation must have a finite deadline."""
    with pytest.raises(ValueError, match="greater than zero"):
        run_ruff(tmp_path, timeout_seconds=0)


def test_validation_tools_return_captured_timeout_results(tmp_path: Path) -> None:
    """Timed-out validation commands are terminated and reported without a shell invocation."""
    write_file(
        tmp_path,
        "test_slow.py",
        "import time\n\n\ndef test_slow() -> None:\n    time.sleep(2)\n",
    )

    result = run_pytest(tmp_path, timeout_seconds=0.01)

    assert result.timed_out
    assert result.return_code is None
    assert not result.succeeded
    assert isinstance(result.stdout, str)
    assert isinstance(result.stderr, str)


def test_a_timeout_is_reported_even_when_the_child_cannot_be_signalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failing to stop a hung command must not replace its timeout with a crash.

    `killpg` raises EPERM whenever the group has already gone or is not ours to signal,
    which is a race every timeout runs. It escaped as a PermissionError, so a repository
    whose suite hung was recorded as a crashed validation step instead of a timeout, and
    the retry lost the diagnostic that would have told it what to fix.
    """
    write_file(
        tmp_path,
        "test_slow.py",
        "import time\n\n\ndef test_slow() -> None:\n    time.sleep(2)\n",
    )

    def refuse_to_signal(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", refuse_to_signal)

    result = run_pytest(tmp_path, timeout_seconds=0.01)

    assert result.timed_out
    assert not result.succeeded


def test_node_validation_commands_run_once_instead_of_watching(tmp_path: Path) -> None:
    """A watch-mode test runner never exits, so a passing suite is recorded as a timeout.

    The Admin Console declares `react-scripts test`, which watches unless CI is set. Every
    client test run from feature -015 onward was therefore a guaranteed timeout.
    """
    (tmp_path / "package.json").write_text(
        '{"name":"web","scripts":{"test":"react-scripts test","lint":"eslint ."}}\n',
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")

    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)

    test_command = next(item for item in plan.commands if item.validation_type == "test")
    assert test_command.environment["CI"] == "true"


def test_a_formatter_that_rewrites_the_checkout_is_not_a_validation_command(
    tmp_path: Path,
) -> None:
    """A script that edits what it inspects has no verdict to give, so it is not a gate.

    AB-console-admin-2.0 declares `format` as `prettier --write` over
    `./**/*.{js,jsx,ts,tsx,css,md,json}`, and it was planned as a required validation
    command. Running it rewrote every matching file in the checkout -- including files the
    feature never touched, whose churn then reached the change fingerprint and the pull
    request -- and it could not fail, because writing is how it resolves what it finds.
    """
    (tmp_path / "package.json").write_text(
        '{"name":"web","scripts":{'
        '"format":"prettier --write \'./**/*.{js,jsx,ts,tsx,css,md,json}\'",'
        '"lint":"eslint .","test":"react-scripts test","build":"react-scripts build"}}\n',
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")

    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)

    assert not [item for item in plan.commands if item.validation_type == "format"]
    # Everything else the checkout declares is still planned; only the writer is dropped.
    assert {item.validation_type for item in plan.commands} == {"lint", "test", "build"}


def test_a_repository_that_declares_a_checking_formatter_still_gets_one(tmp_path: Path) -> None:
    """The rule is about rewriting, not about the word `format`."""
    (tmp_path / "package.json").write_text(
        '{"name":"web","scripts":{'
        '"format":"prettier --write .","format:check":"prettier --check .",'
        '"lint":"eslint . --fix"}}\n',
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")

    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)

    formatting = next(item for item in plan.commands if item.validation_type == "format")
    assert formatting.command == ["npm", "run", "format:check"]
    # A linter that also fixes keeps its place: it still exits non-zero for every rule it
    # could not fix, so dropping it would remove a check rather than a no-op.
    linting = next(item for item in plan.commands if item.validation_type == "lint")
    assert linting.command == ["npm", "run", "lint"]


def test_a_git_failure_reports_a_safe_code_not_repository_hook_output() -> None:
    """A malicious hook transcript cannot become trusted workflow diagnostics."""
    stderr = "\n".join(
        [
            "  12:5  error  'x' is assigned a value but never used  no-unused-vars",
            "",
            "✖ 1 problem (1 error, 0 warnings)",
            *[f"filler line {index}" for index in range(20)],
            "Warning: React version not specified in eslint-plugin-react settings.",
        ]
    )

    with pytest.raises(GitAdapterError) as error:
        _require_success(stderr, 1, "git commit")

    assert "GIT_COMMIT_FAILED_EXIT_1" in str(error.value)
    assert "no-unused-vars" not in str(error.value)


def test_production_work_is_not_rejected_for_landing_outside_a_planned_guess() -> None:
    """Feature -046's frontend implemented the tile and was rejected for the wrong reason.

    The plan named an unrelated existing directory as the expected area for an accessibility
    requirement, so correct production work in a different directory could never satisfy it.
    """
    workstream = RepositoryWorkstreamPlan.model_validate(
        {
            "workstream_id": "ws-web",
            "repository_id": "web",
            "role": "frontend",
            "requirement_ids": ["tile"],
            "scoped_requirements": [
                {
                    "requirement_id": "tile",
                    "acceptance_criterion_ids": ["tile:ac-1"],
                    "responsibility": "implements",
                }
            ],
            "out_of_scope_requirements": [],
            "shared_requirements": [],
            "responsibilities": ["Render the tile."],
            "task_ids": ["t1"],
            "dependency_workstream_ids": [],
            "contract_sections_consumed": [],
            "contract_sections_implemented": [],
            "acceptance_criteria": ["The tile renders."],
            "test_requirements": ["Cover the tile states."],
            "documentation_requirements": ["Describe the tile."],
            "expected_files_or_areas": ["i18n/"],
            "implementation_expectations": [
                {
                    "requirement_id": "tile",
                    "expected_change_categories": ["production"],
                    "expected_source_areas": ["i18n/"],
                    "tests_required": False,
                }
            ],
            "required": True,
        }
    )
    # Also carries the connecting edit the integration promise requires, so the subject of
    # this test stays the planned-area guess rather than the wiring.
    completion = _completion_with_files(
        ["src/components/admin/StatusTile.js"], modified=["src/pages/Home.js"]
    )

    result = validate_implementation_completeness(workstream, completion)

    assert result.findings == []
    assert result.production_files_changed == [
        "src/components/admin/StatusTile.js",
        "src/pages/Home.js",
    ]


def _completion_with_files(
    paths: list[str], modified: list[str] | None = None
) -> CodeCompletionArtifact:
    """Build a completion artifact whose changes are all production source."""
    return create_artifact(
        CodeCompletionArtifact,
        workflow_id="wf",
        artifact_id="006_code_completion.json",
        producer="engineer",
        payload={
            "completion_status": "completed",
            "summary": "Add the tile.",
            "file_changes": [
                *({"path": path, "change_type": "added", "description": "new"} for path in paths),
                *(
                    {"path": path, "change_type": "modified", "description": "wire"}
                    for path in modified or []
                ),
            ],
            "validation_results": [],
            "test_coverage_percent": None,
            "remaining_work": [],
            "commit_sha": None,
        },
        metadata={},
    )


def test_only_the_test_command_declares_ci(tmp_path: Path) -> None:
    """CI makes a test runner exit, but makes react-scripts build fail on warnings.

    Feature -047's build was rejected for pre-existing warnings the change never introduced,
    because the variable that stops the test runner watching also changes what a build means.
    """
    (tmp_path / "package.json").write_text(
        '{"name":"web","scripts":{"test":"react-scripts test","build":"react-scripts build"}}\n',
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")

    plan = DefaultValidationPlanBuilder().build_plan_sync(tmp_path)
    by_type = {item.validation_type: item for item in plan.commands}

    assert by_type["test"].environment == {"CI": "true"}
    assert by_type["build"].environment == {}


def test_a_failing_command_reports_what_it_printed() -> None:
    """The failure's own output is the only thing that says what to change.

    Feature -062 spent four review cycles on "npm run test exited with code 1" while the
    output it was denied named the defect and its line.
    """
    result = ValidationResult(
        command=("npm", "run", "test"),
        return_code=1,
        stdout=(
            "  ● HealthHistory › shows empty state\n"
            "    TypeError: Cannot read properties of undefined "
            "(reading 'mockResolvedValueOnce')\n"
            "      at src/components/__tests__/HealthHistory.test.js:34:11\n"
            "Tests:       3 failed, 3 total\n"
        ),
        stderr="",
        timed_out=False,
        duration_seconds=1.2,
        validation_type="test",
    )

    assert "mockResolvedValueOnce" in result.stdout_summary
    assert "HealthHistory.test.js:34:11" in result.stdout_summary


def test_a_passing_command_keeps_its_chatter_out_of_durable_state() -> None:
    """Output only earns its place in state when it explains a failure."""
    result = ValidationResult(
        command=("npm", "run", "test"),
        return_code=0,
        stdout="Tests:       7 passed, 7 total\n",
        stderr="",
        timed_out=False,
        duration_seconds=1.2,
        validation_type="test",
    )

    assert result.stdout_summary == ""


def test_a_failure_summary_is_bounded_and_keeps_the_end() -> None:
    """A runner prints its failures last, so a bounded excerpt must keep the tail."""
    result = ValidationResult(
        command=("npm", "run", "test"),
        return_code=1,
        stdout=("banner\n" * 5000) + "TypeError: the defect is here\n",
        stderr="",
        timed_out=False,
        duration_seconds=1.2,
        validation_type="test",
    )

    assert len(result.stdout_summary) < 5000
    assert "TypeError: the defect is here" in result.stdout_summary


def _framework_crash_output(*, frames: int = 200) -> str:
    """A Jest-shaped component crash: the throwing frame at the head, framework noise after.

    This is AB-Feature-182's failure shape. The frame that names the product file sits near
    the top; what follows is thousands of characters of package-store frames, then the test
    call site and the suite summary. A plain tail-bound keeps only the noise and the test.
    """
    noise = "\n".join(
        "      at unstable_runWithPriority "
        "(node_modules/react-dom/cjs/react-dom.development.js:11327:26)"
        for _ in range(frames)
    )
    return (
        "FAIL src/pages/AllApps.test.js\n"
        "  ● AllApps creation actions › renders the administrator bulk action\n"
        "    TypeError: Cannot read properties of undefined (reading 'publisher_name')\n"
        "      at src/pages/AllApps.js:220:34\n"
        f"{noise}\n"
        "      at Object.<anonymous> (src/pages/AllApps.test.js:63:5)\n"
        "Tests:       1 failed, 10 passed, 11 total\n"
    )


def _frontend_workspace(tmp_path: Path) -> Path:
    (tmp_path / "src" / "pages").mkdir(parents=True)
    (tmp_path / "src" / "pages" / "AllApps.js").write_text("export default 1;\n")
    (tmp_path / "src" / "pages" / "AllApps.test.js").write_text("test('x', () => {});\n")
    # A real file inside a package store, to prove exclusion is by convention, not absence.
    store = tmp_path / "node_modules" / "react-dom" / "cjs"
    store.mkdir(parents=True)
    (store / "react-dom.development.js").write_text("module.exports = {};\n")
    return tmp_path


def test_a_tail_bounded_report_still_names_the_file_the_head_saw(tmp_path: Path) -> None:
    """The reference list is resolved from the FULL output, before any bounding.

    AB-Feature-182's frontend failed three remediation attempts on one TypeError because
    the tail-bounded summary had dropped the throwing frame: the only app path left was the
    test's render call site, so every repair was pointed at the test fixture while the
    defect lived in the product file.
    """
    workspace = _frontend_workspace(tmp_path)

    summary = _summary_with_references(_framework_crash_output(), workspace)

    assert _WORKSPACE_REFERENCES_HEADER in summary
    references = summary.split(_WORKSPACE_REFERENCES_HEADER, 1)[1]
    assert "src/pages/AllApps.js:220" in references
    assert "src/pages/AllApps.test.js" in references
    # Package-store internals are not files a repair may edit, even when present on disk.
    assert "react-dom.development.js" not in references


def test_workspace_references_promote_the_product_file_into_repair_regions(
    tmp_path: Path,
) -> None:
    """The invariant this exists for: the engineer's promotion sees the product file.

    Asserting on the summary text alone would be the command-not-effect trap; what matters
    is that `_diagnostic_file_locations` -- the function that decides which files a repair
    is shown -- resolves the reference the summary now carries.
    """
    from agents.engineer.agent import _diagnostic_file_locations

    workspace = _frontend_workspace(tmp_path)
    summary = _summary_with_references(_framework_crash_output(), workspace)

    locations = _diagnostic_file_locations([summary], workspace)

    assert locations.get("src/pages/AllApps.js") == 220


def test_references_survive_the_reviewers_tail_excerpt(tmp_path: Path) -> None:
    """The reviewer re-truncates the summary to its own tail; the section must outlive it."""
    from agents.reviewer.agent import _validation_failure_excerpt
    from tools.technology_detection import RepositoryRevision

    workspace = _frontend_workspace(tmp_path)
    result = _result(
        command=("npm", "run", "test"),
        return_code=1,
        stdout="",
        stderr=_framework_crash_output(),
        timed_out=False,
        duration_seconds=1.2,
        validation_type="test",
        repository_id="frontend",
        revision=RepositoryRevision(
            index_fingerprint="test",
            working_tree_fingerprint="test",
            untracked_fingerprint="test",
            combined_fingerprint="test",
        ),
        success_exit_codes=(0,),
        required=True,
        repository_root=workspace,
    )

    excerpt = _validation_failure_excerpt(result)

    assert "src/pages/AllApps.js:220" in excerpt


def test_the_reference_list_is_bounded_deduplicated_and_first_seen(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    for index in range(30):
        (tmp_path / "src" / f"module{index:02d}.js").write_text("export default 1;\n")
    lines = [f"      at broken (src/module{index:02d}.js:{index + 1}:1)" for index in range(30)]
    # The first file again, with a different line: one entry per file, first line kept.
    lines.append("      at broken (src/module00.js:99:1)")
    output = "boom\n" + "\n".join(lines)

    summary = _summary_with_references(output, tmp_path)

    references = summary.split(_WORKSPACE_REFERENCES_HEADER, 1)[1]
    reference_lines = [line for line in references.splitlines() if line.strip()]
    assert len(reference_lines) == 20
    assert reference_lines[0] == "src/module00.js:1"
    assert "src/module00.js:99" not in references


def test_an_output_naming_no_workspace_files_gains_no_section(tmp_path: Path) -> None:
    """Absent files, escapes, and package-store internals resolve to nothing."""
    (tmp_path / "node_modules" / "left-pad").mkdir(parents=True)
    (tmp_path / "node_modules" / "left-pad" / "index.js").write_text("module.exports = 1;\n")
    output = (
        "the command exited with code 1\n"
        "      at helper (node_modules/left-pad/index.js:3:1)\n"
        "      at gone (src/missing.js:5:1)\n"
        "      at escape (../outside.js:1:1)\n"
    )

    summary = _summary_with_references(output, tmp_path)

    assert _WORKSPACE_REFERENCES_HEADER not in summary
    assert "exited with code 1" in summary


def test_a_config_directory_file_satisfies_a_configuration_requirement() -> None:
    """A repository that keeps a config directory puts configuration in it.

    Feature -063's backend created `server/config/healthHistory.js`, declared it production
    in its own file lists, and was refused three times for "required change categories:
    configuration" -- a category it had in fact changed.
    """
    assert classify_file_change("server/config/healthHistory.js") == "configuration"
    assert classify_file_change("src/settings/limits.py") == "configuration"
    # The directory has to contain the file, not merely share a name with it.
    assert classify_file_change("server/config.js") == "production"
    assert classify_file_change("server/utils/healthHistory.js") == "production"


def test_test_infrastructure_files_are_configuration_not_production() -> None:
    """conftest.py wires a runner; nothing in production imports it, and that is correct.

    `conftest.py` matched no test token (`test_` needs its underscore), classified as
    production, and AB-Feature-216's reachability gate then demanded a production referrer
    for it -- satisfied, at attempt 3, by exporting a string from an OAuth model.
    Configuration rather than test, so `changed_test_paths` never hands these files to a
    narrowed runner as suites (`pytest -q conftest.py` collects nothing and fails).
    """
    assert classify_file_change("server/conftest.py") == "configuration"
    assert classify_file_change("tests/conftest.py") == "configuration"
    assert classify_file_change("jest.setup.js") == "configuration"
    assert classify_file_change("src/setupTests.ts") == "configuration"
    assert classify_file_change("vitest.setup.ts") == "configuration"
    # Helper names the runners collect as suites by convention keep their test classification.
    assert classify_file_change("tests/test_helper.py") == "test"
    # And the prefix rows match filenames, never directories that happen to share the name.
    assert classify_file_change("server/jest.setup.d/service.js") == "production"


def test_a_conftest_beside_production_source_still_satisfies_production() -> None:
    """Moving test infrastructure out of the production bucket cannot unmake a change.

    The safety check for the reclassification: a change whose production expectation was
    satisfied before still satisfies it when a conftest.py rides along, and the conftest
    lands in the configuration bucket rather than vanishing.
    """
    workstream = RepositoryWorkstreamPlan.model_validate(
        {
            "workstream_id": "ws-api",
            "repository_id": "api",
            "role": "backend",
            "requirement_ids": ["export"],
            "scoped_requirements": [
                {
                    "requirement_id": "export",
                    "acceptance_criterion_ids": ["export:ac-1"],
                    "responsibility": "implements",
                }
            ],
            "out_of_scope_requirements": [],
            "shared_requirements": [],
            "responsibilities": ["Serve the export."],
            "task_ids": ["t1"],
            "dependency_workstream_ids": [],
            "contract_sections_consumed": [],
            "contract_sections_implemented": [],
            "acceptance_criteria": ["The export responds."],
            "test_requirements": ["Cover the export."],
            "documentation_requirements": ["Describe the export."],
            "expected_files_or_areas": ["src/"],
            "implementation_expectations": [
                {
                    "requirement_id": "export",
                    "expected_change_categories": ["production"],
                    "expected_source_areas": ["src/"],
                    "tests_required": False,
                }
            ],
            "required": True,
        }
    )
    # The modified route is the connecting edit the integration promise requires, so the
    # subject of this test stays the conftest reclassification rather than the wiring.
    completion = _completion_with_files(
        ["src/services/export.service.js", "server/conftest.py"],
        modified=["src/routes/index.js"],
    )

    result = validate_implementation_completeness(workstream, completion)

    assert result.findings == []
    assert result.production_files_changed == [
        "src/services/export.service.js",
        "src/routes/index.js",
    ]
    assert result.configuration_files_changed == ["server/conftest.py"]


def test_a_verbose_failing_command_keeps_the_end_where_it_says_what_failed() -> None:
    """Head-only capture threw away the answer.

    AB-Feature-168's Jest run reported "Tests: 13 failed, 1 passed" and named every failing
    assertion. None of it survived: capture kept the first bytes and stopped, so the durable
    record held React `act()` warnings from the middle of the run and no verdict at all --
    and that record is what the next attempt would have been given to work from.
    """
    stream = _HeadAndTail(200)

    stream.feed(b"> jest --ci\nPASS src/a.test.js\n")
    stream.feed(b"noise\n" * 400)
    stream.feed(b"Tests: 13 failed, 1 passed, 14 total\n")

    collected = stream.collected()
    assert stream.truncated is True
    # The invocation, which says what was actually run.
    assert collected.startswith(b"> jest --ci\n")
    # And the verdict, which is the whole reason to keep the output.
    assert collected.endswith(b"Tests: 13 failed, 1 passed, 14 total\n")
    # The gap is marked rather than silently joined, so nobody reads across it.
    assert b"[output truncated]" in collected
    assert len(collected) <= 200 + len(b"\n[output truncated]\n")


def test_output_that_fits_is_returned_whole_and_unmarked() -> None:
    """The common case must be byte-exact, with no truncation marker invented for it."""
    stream = _HeadAndTail(200)

    stream.feed(b"all 14 tests passed\n")

    assert stream.collected() == b"all 14 tests passed\n"
    assert stream.truncated is False


def _repo_git(root: Path, *arguments: str) -> None:
    """Run one git command for real, because every answer here comes out of a checkout."""
    subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True, timeout=30)


def _checkout_with_build_output(root: Path) -> Path:
    """A checkout shaped like AB-Feature-218's: source, and a gitignored populated `dist/`."""
    (root / "server" / "validation").mkdir(parents=True)
    (root / "dist" / "server" / "auth").mkdir(parents=True)
    (root / ".gitignore").write_text("dist/\n", encoding="utf-8")
    (root / "server" / "validation" / "app.validation.js").write_text(
        "const addAppSchema = Joi.object({});\nmodule.exports = { addAppSchema };\n",
        encoding="utf-8",
    )
    (root / "package.json").write_text('{"name": "admanager"}\n', encoding="utf-8")
    # What the platform's own baseline `npm run build` wrote, minutes before the first attempt.
    (root / "dist" / "server.js").write_text("// generated\n", encoding="utf-8")
    (root / "dist" / "server" / "auth" / "OAuthClient.js").write_text(
        "// generated\n", encoding="utf-8"
    )
    _repo_git(root, "init", "--quiet")
    _repo_git(root, "config", "user.email", "engineer@example.test")
    _repo_git(root, "config", "user.name", "Engineer")
    _repo_git(root, "add", "-A")
    _repo_git(root, "commit", "--quiet", "-m", "the repository, without its build output")
    return root


def test_the_snapshot_is_the_repository_not_its_build_output(tmp_path: Path) -> None:
    """AB-Feature-218's `dist/**` took 13% of the engineer's context budget.

    `git ls-files dist` returned nothing and `.gitignore` named `dist/`; the eight files
    existed only because the platform's own baseline build had just written them into the
    working tree this walk reads. They displaced, among other things, the source file the
    instruction told four consecutive attempts to reuse.
    """
    root = _checkout_with_build_output(tmp_path)

    scan = scan_directory(root)
    paths = [path.as_posix() for path in scan.files]

    assert [path for path in paths if path.startswith("dist/")] == []
    assert "server/validation/app.validation.js" in paths
    assert "package.json" in paths
    # The same walk feeds `context_file_paths` and the review's `channel_seam_paths`, so one
    # exclusion answers both consumers 218 recorded the pollution on.
    assert scan.total_size_bytes > 0


def test_a_checkout_with_no_git_still_gets_the_static_answer(tmp_path: Path) -> None:
    """A root Git cannot be asked about keeps exactly the behaviour it has today."""
    write_file(tmp_path, "app/module.py", "VALUE = 1\n")
    write_file(tmp_path, "app/local.py", "VALUE = 2\n")
    write_file(tmp_path, "node_modules/pkg/index.js", "module.exports = {};\n")

    scan = scan_directory(tmp_path)

    assert scan.files == (Path("app/local.py"), Path("app/module.py"))


def test_the_static_fallback_still_drops_build_output(tmp_path: Path) -> None:
    """Failing closed means the fallback is not obviously wrong either.

    Without Git there is no authority to consult, so build output is recognised by
    directory name -- worse than asking the repository, and better than admitting it.
    """
    write_file(tmp_path, "src/index.js", "export default 1;\n")
    write_file(tmp_path, "dist/index.js", "// generated\n")
    write_file(tmp_path, "coverage/lcov.info", "TN:\n")

    scan = scan_directory(tmp_path)

    assert scan.files == (Path("src/index.js"),)


def test_a_tracked_file_under_an_ignore_rule_is_still_returned(tmp_path: Path) -> None:
    """The index outranks the ignore rules, and only `git ls-files` knows that.

    A repository that commits a file has settled whether it is source, whatever a parent
    directory's pattern says. `git check-ignore` alone is not a safe authority here, and a
    repository vendoring its build output would otherwise lose it from every snapshot.
    """
    root = _checkout_with_build_output(tmp_path)
    vendored = root / "dist" / "vendored.js"
    vendored.write_text("// committed on purpose\n", encoding="utf-8")
    _repo_git(root, "add", "--force", "dist/vendored.js")
    _repo_git(root, "commit", "--quiet", "-m", "this one is source")

    paths = [path.as_posix() for path in scan_directory(root).files]

    assert "dist/vendored.js" in paths
    assert "dist/server.js" not in paths
