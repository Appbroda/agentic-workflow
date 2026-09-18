"""Reporting the lint vocabulary a checkout actually resolves."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest

from services.process_runner import ProcessResult
from tests.fixtures import node_javascript_repository
from tools.lint_capabilities import (
    NullLintCapabilityProbe,
    RepositoryLintCapabilities,
    _eslint_capabilities,
)

_PRINTED_CONFIG = json.dumps(
    {
        "rules": {
            "no-unused-vars": ["error"],
            "no-console": ["warn"],
            "eqeqeq": "off",
            "react/jsx-uses-vars": [2],
            "import/no-unresolved": ["error"],
            "jsx-a11y/alt-text": [0],
        }
    }
)


class CapturingProcessRunner:
    """Return one canned resolved configuration and record how it was requested."""

    def __init__(self, *, stdout: str = _PRINTED_CONFIG, return_code: int = 0) -> None:
        """Configure the probe response without invoking a real linter."""
        self.commands: list[tuple[str, ...]] = []
        self._stdout = stdout
        self._return_code = return_code

    async def run(self, command: Any, cwd: Path, *_args: Any, **_kwargs: Any) -> ProcessResult:
        """Record the probe invocation and return the configured result."""
        received = tuple(command)
        self.commands.append(received)
        return ProcessResult(
            command=received,
            return_code=self._return_code,
            stdout=self._stdout,
            stderr="",
            duration_seconds=0.01,
        )


def test_only_enabled_rules_and_their_namespaces_are_reported() -> None:
    """A rule switched off is not available vocabulary, whichever way it was disabled."""
    capabilities = _eslint_capabilities(_PRINTED_CONFIG, "src/a.js")["eslint"]

    assert capabilities["available_plugin_namespaces"] == ["import", "react"]
    assert "eqeqeq" not in capabilities["enabled_rules"]
    assert "jsx-a11y/alt-text" not in capabilities["enabled_rules"]
    assert capabilities["enabled_rule_count"] == 4
    # react-hooks is the namespace three live runs failed on; it must be absent here.
    assert "react-hooks" not in capabilities["available_plugin_namespaces"]


def test_a_late_alphabet_style_rule_still_reaches_the_model() -> None:
    """A count-based cut hid exactly the rules that fail a commit.

    Feature -022's backend failed on `line-comment-position`. Sorted alphabetically among a
    few hundred rules, it sat far past a 60-entry cut and was never reported.
    """
    rules: dict[str, Any] = {f"aaa-rule-{index:03d}": "error" for index in range(250)}
    rules["line-comment-position"] = ["error", "above"]

    capabilities = _eslint_capabilities(json.dumps({"rules": rules}), "src/a.js")["eslint"]

    # Reported with its option, because the option is what says how to satisfy it.
    assert 'line-comment-position: ["above"]' in capabilities["enabled_rules"]
    assert capabilities["enabled_rules_truncated"] is False
    assert capabilities["enabled_rule_count"] == 251


def test_a_rule_without_options_is_reported_as_a_bare_name() -> None:
    """Only rules that carry configuration spend prompt space on it."""
    config = json.dumps({"rules": {"no-debugger": "error", "quotes": ["error", "single"]}})

    enabled = _eslint_capabilities(config, "src/a.js")["eslint"]["enabled_rules"]

    assert "no-debugger" in enabled
    assert 'quotes: ["single"]' in enabled


@pytest.mark.asyncio
async def test_the_checkouts_own_eslint_resolves_its_configuration(tmp_path: Path) -> None:
    """The tool resolves the config, because only it knows which format won."""
    root = node_javascript_repository(tmp_path / "web")
    _install_node_binary(root, "eslint")
    runner = CapturingProcessRunner()

    capabilities = await RepositoryLintCapabilities(process_runner=runner).describe(
        root, ["README.md", "src/a.js", "src/b.js"]
    )

    assert len(runner.commands) == 1
    command = runner.commands[0]
    assert command[0].endswith("node_modules/.bin/eslint")
    assert command[1] == "--print-config"
    # Resolved against a lintable source file, never a Markdown document.
    assert command[2] == "src/a.js"
    assert capabilities["eslint"]["resolved_for"] == "src/a.js"


@pytest.mark.asyncio
async def test_a_probe_failure_is_advisory_and_never_blocks_coding(tmp_path: Path) -> None:
    """The commit gate stays the authority; a failed probe just reports nothing."""
    root = node_javascript_repository(tmp_path / "web")
    _install_node_binary(root, "eslint")
    runner = CapturingProcessRunner(stdout="not json at all", return_code=1)

    assert (
        await RepositoryLintCapabilities(process_runner=runner).describe(root, ["src/a.js"]) == {}
    )


@pytest.mark.asyncio
async def test_a_checkout_without_eslint_reports_nothing(tmp_path: Path) -> None:
    """Nothing is invoked when the repository has no linter of its own."""
    root = node_javascript_repository(tmp_path / "web")
    runner = CapturingProcessRunner()

    assert (
        await RepositoryLintCapabilities(process_runner=runner).describe(root, ["src/a.js"]) == {}
    )
    assert runner.commands == []
    assert await NullLintCapabilityProbe().describe(root, ["src/a.js"]) == {}


def _install_node_binary(root: Path, tool: str) -> None:
    """Create an executable stub where a package manager would install the real tool."""
    binary = root / "node_modules" / ".bin" / tool
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)


_PRODUCTION_CONFIG = json.dumps(
    {
        "rules": {
            "no-unused-vars": ["error"],
            "react/jsx-uses-vars": [2],
        }
    }
)
_TEST_FILE_CONFIG = json.dumps(
    {
        "rules": {
            "no-unused-vars": ["error"],
            "react/jsx-uses-vars": [2],
            "testing-library/no-node-access": ["error"],
            "testing-library/render-result-naming-convention": ["error"],
        }
    }
)


class PerSubjectProcessRunner:
    """Answer `--print-config` per subject, the way ESLint's own override matching does."""

    def __init__(self, configs: dict[str, str]) -> None:
        self.commands: list[tuple[str, ...]] = []
        self._configs = configs

    async def run(self, command: Any, cwd: Path, *_args: Any, **_kwargs: Any) -> ProcessResult:
        received = tuple(command)
        self.commands.append(received)
        subject = received[-1]
        stdout = self._configs.get(subject, "")
        return ProcessResult(
            command=received,
            return_code=0 if stdout else 1,
            stdout=stdout,
            stderr="",
            duration_seconds=0.01,
        )


@pytest.mark.asyncio
async def test_the_vocabulary_is_resolved_per_file_class_the_attempt_can_write(
    tmp_path: Path,
) -> None:
    """One subject per class, because overrides bind whole plugins to test files only.

    AB-Feature-171's checkout resolves 0 of 19 `testing-library/*` rules for `src/App.js`
    and all 19 for a test file. Every lint-gate death in both live features was in a
    `*.test.js` file, on rules the platform had authoritatively said do not exist here.
    """
    root = node_javascript_repository(tmp_path / "web")
    _install_node_binary(root, "eslint")
    runner = PerSubjectProcessRunner(
        {"src/a.js": _PRODUCTION_CONFIG, "test/a.test.js": _TEST_FILE_CONFIG}
    )

    capabilities = await RepositoryLintCapabilities(process_runner=runner).describe(
        root, ["README.md", "src/a.js", "test/a.test.js"]
    )

    # Two invocations, one representative per class; the tool applied its own overrides.
    assert [command[-1] for command in runner.commands] == ["src/a.js", "test/a.test.js"]
    eslint = capabilities["eslint"]
    assert eslint["resolved_for"] == "src/a.js"
    assert "testing-library/no-node-access" not in eslint["enabled_rules"]
    test_files = eslint["test_files"]
    assert test_files["resolved_for"] == "test/a.test.js"
    # Reported with per-class provenance: only what test files add, labeled as theirs.
    assert test_files["additional_enabled_rules"] == [
        "testing-library/no-node-access",
        "testing-library/render-result-naming-convention",
    ]
    assert "testing-library" in test_files["available_plugin_namespaces"]
    assert test_files["enabled_rule_count"] == 4


@pytest.mark.asyncio
async def test_a_class_whose_subject_cannot_resolve_is_absent_not_asserted(
    tmp_path: Path,
) -> None:
    """A probe blind spot is reported as absence, never converted into a false claim."""
    root = node_javascript_repository(tmp_path / "web")
    _install_node_binary(root, "eslint")
    runner = PerSubjectProcessRunner({"src/a.js": _PRODUCTION_CONFIG})

    capabilities = await RepositoryLintCapabilities(process_runner=runner).describe(
        root, ["src/a.js", "test/a.test.js"]
    )

    eslint = capabilities["eslint"]
    assert eslint["resolved_for"] == "src/a.js"
    assert "test_files" not in eslint
    # And the inverse: only a test file resolvable still reports that class alone.
    runner = PerSubjectProcessRunner({"test/a.test.js": _TEST_FILE_CONFIG})
    capabilities = await RepositoryLintCapabilities(process_runner=runner).describe(
        root, ["src/a.js", "test/a.test.js"]
    )
    assert capabilities["eslint"]["resolved_for"] == "test/a.test.js"
