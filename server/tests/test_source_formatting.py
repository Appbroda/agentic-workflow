"""Formatting generated code with the repository's own tooling before it is committed.

Feature -015's frontend implemented its requirement correctly and was then rejected by
the repository's husky pre-commit hook over twenty-two Prettier violations. The platform
must normalize what it writes using the checkout's own fixers, scoped to only the files
it changed.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest

from services.process_runner import ProcessResult
from tests.fixtures import node_javascript_repository, python_fastapi_repository
from tools.source_formatting import (
    _MAX_DIAGNOSTIC_CHARACTERS,
    NullSourceFormatter,
    RepositoryToolFormatter,
    SourceValidationError,
    _failure_lines,
    _formatting_commands,
    _verification_commands,
)


class CapturingProcessRunner:
    """Record commands without invoking a real formatter or linter."""

    def __init__(self, *, fail_verification: bool = False) -> None:
        """Track commands and optionally leave a non-fixable lint violation behind."""
        self.commands: list[tuple[str, ...]] = []
        self.working_directories: list[Path] = []
        self._fail_verification = fail_verification

    async def run(self, command: Any, cwd: Path, *_args: Any, **_kwargs: Any) -> ProcessResult:
        """Return a deterministic result for fixer and verification invocations."""
        received = tuple(command)
        self.commands.append(received)
        self.working_directories.append(cwd)
        verifying = received == ("npm", "run", "lint") or (
            received[0].endswith("eslint") and "--fix" not in received
        )
        fails = self._fail_verification and verifying
        return ProcessResult(
            command=received,
            return_code=1 if fails else 0,
            stdout="src/components/Tile.js: no-unused-vars" if fails else "",
            stderr="some findings are not auto-fixable" if fails else "",
            duration_seconds=0.01,
        )


@pytest.mark.asyncio
async def test_generated_javascript_is_fixed_with_the_repositories_own_binaries(
    tmp_path: Path,
) -> None:
    """Only the checkout's installed fixers run, and only over the generated paths."""
    root = node_javascript_repository(tmp_path / "web")
    _install_node_binary(root, "prettier")
    _install_node_binary(root, "eslint")
    runner = CapturingProcessRunner()

    applied = await RepositoryToolFormatter(process_runner=runner).format_paths(
        root, ["src/components/Tile.js", "src/hooks/useStatus.js"]
    )

    assert len(runner.commands) == 3
    prettier, eslint_fix, eslint_verify = runner.commands
    assert prettier[0].endswith("node_modules/.bin/prettier")
    assert prettier[1] == "--write"
    assert eslint_fix[0].endswith("node_modules/.bin/eslint")
    assert eslint_fix[1] == "--fix"
    assert eslint_verify[0].endswith("node_modules/.bin/eslint")
    assert "--fix" not in eslint_verify
    # Exactly the generated paths: an unrelated repository file is never reformatted.
    assert set(prettier[2:]) == {"src/components/Tile.js", "src/hooks/useStatus.js"}
    assert set(eslint_verify[1:]) == {"src/components/Tile.js", "src/hooks/useStatus.js"}
    assert len(applied) == 3


@pytest.mark.asyncio
async def test_unfixable_lint_is_rejected_before_git_can_commit(tmp_path: Path) -> None:
    """An ESLint error becomes actionable retry feedback instead of a Husky failure.

    "Actionable" means the rule and the file it fired on. Feature -057 proved what the code
    alone is worth: both attempts were handed the same bare
    `code=SOURCE_VALIDATION_FAILED_EXIT_1` and the second reproduced the first's violation,
    because nothing in the feedback distinguished them.
    """
    root = node_javascript_repository(tmp_path / "web")
    _install_node_binary(root, "prettier")
    _install_node_binary(root, "eslint")
    runner = CapturingProcessRunner(fail_verification=True)

    with pytest.raises(SourceValidationError) as error:
        await RepositoryToolFormatter(process_runner=runner).format_paths(
            root, ["src/components/Tile.js"]
        )

    diagnostic = error.value.diagnostics[0]
    assert "SOURCE_VALIDATION_FAILED_EXIT_1" in diagnostic
    assert "no-unused-vars" in diagnostic
    assert "Tile.js" in diagnostic
    assert runner.commands[-1][-1] == "src/components/Tile.js"


@pytest.mark.asyncio
async def test_a_repository_without_installed_fixers_is_left_untouched(tmp_path: Path) -> None:
    """Nothing is invoked when the checkout has no local formatter to run."""
    root = node_javascript_repository(tmp_path / "web")
    runner = CapturingProcessRunner()

    applied = await RepositoryToolFormatter(process_runner=runner).format_paths(
        root, ["src/components/Tile.js"]
    )

    assert runner.commands == []
    assert applied == ()


def test_python_sources_are_formatted_only_when_ruff_is_configured(tmp_path: Path) -> None:
    """Ruff runs for a configured Python project and never for an unconfigured one."""
    configured = python_fastapi_repository(tmp_path / "service")
    unconfigured = tmp_path / "plain"
    (unconfigured / "app").mkdir(parents=True)
    (unconfigured / "app" / "main.py").write_text("x = 1\n", encoding="utf-8")

    configured_commands = _formatting_commands(configured, ["app/status.py"])
    unconfigured_commands = _formatting_commands(unconfigured, ["app/main.py"])

    assert [command[:2] for command in configured_commands] == [
        ("ruff", "format"),
        ("ruff", "check"),
    ]
    assert unconfigured_commands == []


def test_a_subdirectory_package_is_formatted_with_its_own_installed_binary(
    tmp_path: Path,
) -> None:
    """The binary a nested package's Husky hook would run is the one that must be used."""
    root = node_javascript_repository(tmp_path / "monorepo")
    _install_node_binary(root, "prettier")
    web = root / "web"
    web.mkdir(parents=True, exist_ok=True)
    (web / "package.json").write_text('{"name": "web"}\n', encoding="utf-8")
    _install_node_binary(web, "prettier")

    commands = _formatting_commands(root, ["src/a.js", "web/src/Tile.jsx"])

    binaries = {command[0]: set(command[2:]) for command in commands}
    assert binaries == {
        str(root / "node_modules" / ".bin" / "prettier"): {"src/a.js"},
        str(web / "node_modules" / ".bin" / "prettier"): {"web/src/Tile.jsx"},
    }


def test_a_nested_path_falls_back_to_the_repository_root_installation(tmp_path: Path) -> None:
    """A package without its own install still uses the checkout's hoisted binary."""
    root = node_javascript_repository(tmp_path / "monorepo")
    _install_node_binary(root, "prettier")

    commands = _formatting_commands(root, ["packages/ui/src/Tile.jsx"])

    assert len(commands) == 1
    assert commands[0][0] == str(root / "node_modules" / ".bin" / "prettier")
    assert commands[0][2:] == ("packages/ui/src/Tile.jsx",)


def test_json_is_formatted_but_never_linted_as_a_program(tmp_path: Path) -> None:
    """Feature -018 failed because ESLint was handed `.eslintrc.json` and parsed it as code."""
    root = node_javascript_repository(tmp_path / "web")
    _install_node_binary(root, "prettier")
    _install_node_binary(root, "eslint")
    paths = [".eslintrc.json", "src/config/serverStatusConfig.js"]

    fixers = _formatting_commands(root, paths)
    verifiers = _verification_commands(root, paths)

    for command in (*fixers, *verifiers):
        if command[0].endswith("eslint"):
            assert ".eslintrc.json" not in command
    prettier = next(command for command in fixers if command[0].endswith("prettier"))
    assert ".eslintrc.json" in prettier
    assert verifiers[0][1:] == ("src/config/serverStatusConfig.js",)


def test_non_source_paths_are_never_sent_to_a_formatter(tmp_path: Path) -> None:
    """A lockfile or binary asset must not be handed to a code formatter."""
    root = node_javascript_repository(tmp_path / "web")
    _install_node_binary(root, "prettier")

    commands = _formatting_commands(root, ["package-lock.json", "assets/logo.png", "src/a.js"])

    assert len(commands) == 1
    # JSON is a Prettier target; the binary asset is not.
    assert set(commands[0][2:]) == {"package-lock.json", "src/a.js"}


@pytest.mark.asyncio
async def test_the_repositories_own_gate_replaces_the_private_approximation(
    tmp_path: Path,
) -> None:
    """Both target repositories' Husky hooks run `npm run lint`, so verification must too."""
    root = node_javascript_repository(tmp_path / "web")
    _install_node_binary(root, "prettier")
    _install_node_binary(root, "eslint")
    runner = CapturingProcessRunner()

    await RepositoryToolFormatter(
        process_runner=runner, gate=(("npm", "run", "lint"), ".")
    ).format_paths(root, ["src/components/Tile.js"])

    assert runner.commands[-1] == ("npm", "run", "lint")
    # The fixers still run per path; only the verification defers to the gate.
    assert not any(
        command[0].endswith("eslint") and "--fix" not in command for command in runner.commands
    )


@pytest.mark.asyncio
async def test_a_nested_packages_gate_runs_in_its_own_directory(tmp_path: Path) -> None:
    """A package's lint script is declared in its own manifest and scoped to its own tree."""
    root = node_javascript_repository(tmp_path / "monorepo")
    _install_node_binary(root, "prettier")
    admin = root / "apps" / "admin"
    admin.mkdir(parents=True, exist_ok=True)
    runner = CapturingProcessRunner()

    await RepositoryToolFormatter(
        process_runner=runner, gate=(("npm", "run", "lint"), "apps/admin")
    ).format_paths(root, ["apps/admin/src/Tile.js"])

    assert runner.commands[-1] == ("npm", "run", "lint")
    assert runner.working_directories[-1] == admin
    # The fixers still run from the checkout root, where their paths are resolved.
    assert runner.working_directories[0] == root


@pytest.mark.asyncio
async def test_a_gate_directory_outside_the_checkout_falls_back_to_the_root(
    tmp_path: Path,
) -> None:
    """A traversing working directory must not run a command outside the given checkout."""
    root = node_javascript_repository(tmp_path / "web")
    _install_node_binary(root, "prettier")
    runner = CapturingProcessRunner()

    await RepositoryToolFormatter(
        process_runner=runner, gate=(("npm", "run", "lint"), "../elsewhere")
    ).format_paths(root, ["src/a.js"])

    assert runner.working_directories[-1] == root


@pytest.mark.asyncio
async def test_an_already_failing_gate_is_not_blamed_on_the_generated_change(
    tmp_path: Path,
) -> None:
    """A dirty default branch is a repository defect no generated change can clear."""
    root = node_javascript_repository(tmp_path / "web")
    _install_node_binary(root, "prettier")
    runner = CapturingProcessRunner(fail_verification=True)

    with pytest.raises(SourceValidationError) as error:
        await RepositoryToolFormatter(
            process_runner=runner,
            gate=(("npm", "run", "lint"), "."),
            gate_already_failing=True,
        ).format_paths(root, ["src/components/Tile.js"])

    assert "already failing" in error.value.diagnostics[-1]


@pytest.mark.asyncio
async def test_the_default_formatter_makes_no_changes(tmp_path: Path) -> None:
    """Mock execution must not depend on any repository tooling being present."""
    assert await NullSourceFormatter().format_paths(tmp_path, ["src/a.js"]) == ()


def _install_node_binary(root: Path, tool: str) -> None:
    """Create an executable stub where a package manager would install the real tool."""
    binary = root / "node_modules" / ".bin" / tool
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    manifest = root / "package.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.setdefault("devDependencies", {})[tool] = "1.0.0"
    manifest.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def test_the_diagnostic_keeps_the_errors_not_the_first_hundred_warnings() -> None:
    """Feature -048's retry was handed a transcript that never showed the defect.

    A project-wide lint prints hundreds of warnings and one error, and truncating at a fixed
    length kept the warnings and cut the error away.
    """
    noise = "\n".join(
        [
            f"/repo/src/components/Noise{index}.js\n  1:1  warning  Something advisory  some-rule"
            for index in range(200)
        ]
    )
    output = (
        f"{noise}\n"
        "/repo/src/components/home/__tests__/Tiles.test.js\n"
        "   7:52  error  Component definition is missing display name  react/display-name\n"
        "\n✖ 1 problem (1 error, 0 warnings)\n"
    )

    kept = _failure_lines(output)

    assert "react/display-name" in kept
    assert "Tiles.test.js" in kept
    assert "1 problem (1 error" in kept
    assert "Noise0.js" not in kept
    assert len(kept) < _MAX_DIAGNOSTIC_CHARACTERS


class OneUnfixableViolation:
    """Fail both the fixer and the gate over a single non-fixable rule, as ESLint does.

    ``eslint --fix`` exits 1 for exactly the violations it could not fix, and the gate then
    reports the same ones. This is the ordinary shape of a rejected attempt, not an edge case.
    """

    _REPORT = (
        "/repo/src/components/apps/BulkAppImportModal.test.js\n"
        "  7:33  error  Component definition is missing display name  react/display-name\n"
        "✖ 1 problem (1 error, 0 warnings)"
    )

    def __init__(self, *, gate_succeeds: bool = False) -> None:
        self.commands: list[tuple[str, ...]] = []
        self._gate_succeeds = gate_succeeds

    async def run(self, command: Any, cwd: Path, *_args: Any, **_kwargs: Any) -> ProcessResult:
        """Report the same violation from the fixer and, unless told otherwise, the gate."""
        del cwd
        received = tuple(command)
        self.commands.append(received)
        gate = received == ("npm", "run", "lint")
        fixer = received[0].endswith("eslint") and "--fix" in received
        fails = fixer or (gate and not self._gate_succeeds)
        return ProcessResult(
            command=received,
            return_code=1 if fails else 0,
            stdout=self._REPORT if fails else "",
            stderr="",
            duration_seconds=0.01,
        )


@pytest.mark.asyncio
async def test_one_unfixable_violation_produces_one_diagnostic(tmp_path: Path) -> None:
    """The fixer and the gate were both reporting it, and neither report was redundant text.

    AB-Feature-121's console carried the same `react/display-name` error twice into its retry
    feedback -- once as `tool=eslint`, once as `tool=npm` -- because the fixer's own exit code
    was recorded beside the gate's verdict on the same violation. The `tool=` token differs,
    so the existing de-duplication could not see them as one. Seventeen attempts across the
    last fifteen features were fed a doubled diagnostic this way.
    """
    root = node_javascript_repository(tmp_path / "web")
    _install_node_binary(root, "prettier")
    _install_node_binary(root, "eslint")
    runner = OneUnfixableViolation()

    with pytest.raises(SourceValidationError) as rejected:
        await RepositoryToolFormatter(
            process_runner=runner, gate=(("npm", "run", "lint"), ".")
        ).format_paths(root, ["src/components/Tile.js"])

    assert len(rejected.value.diagnostics) == 1
    diagnostic = rejected.value.diagnostics[0]
    assert "tool=npm" in diagnostic
    assert "react/display-name" in diagnostic


@pytest.mark.asyncio
async def test_a_fixer_that_fails_alone_is_still_reported(tmp_path: Path) -> None:
    """A fixer can fail for reasons the gate never sees: a broken config, a parser crash.

    Suppressing its diagnostic whenever the gate is quiet would leave a rejected attempt
    holding no explanation at all, which is the failure mode the excerpt exists to prevent.
    """
    root = node_javascript_repository(tmp_path / "web")
    _install_node_binary(root, "prettier")
    _install_node_binary(root, "eslint")
    runner = OneUnfixableViolation(gate_succeeds=True)

    with pytest.raises(SourceValidationError) as rejected:
        await RepositoryToolFormatter(
            process_runner=runner, gate=(("npm", "run", "lint"), ".")
        ).format_paths(root, ["src/components/Tile.js"])

    assert len(rejected.value.diagnostics) == 1
    assert "tool=eslint" in rejected.value.diagnostics[0]
