"""Tell the coding model which lint rules a checkout actually has.

Generated code keeps failing on rules the repository never configured: a disable comment
naming a plugin that is not installed is itself an error, and so is a rule reference the
resolver cannot find. Instructing a model not to do that has not held, because the
instruction carries no information about what *is* available.

This asks the linter to resolve its own configuration and reports what came back. Nothing
is parsed by hand: a repository may declare ESLint through flat config, a legacy rc file,
JSON, YAML, or a package.json key, and only the tool itself knows which one won.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from services.cancellation import CancellationToken, MockCancellationToken
from services.process_runner import (
    AsyncioProcessRunner,
    ProcessRunner,
    repository_subprocess_environment,
)
from tools.file_tools import PathLike, resolve_workspace_root
from tools.source_formatting import _eslint_paths, _node_binary

_PROBE_TIMEOUT_SECONDS = 120.0
# Bounded by size rather than by count. A fixed count truncates an alphabetically sorted
# configuration at whatever letter it reaches, which silently hides most of the vocabulary:
# a 60-rule cut over a 300-rule config ends around "c", so a rule like
# `line-comment-position` never reaches the model at all. Rule names are short, and this
# budget carries a full configuration for the repositories seen so far.
_MAX_RULE_CHARACTERS = 16_000
# One rule's options must not crowd out every rule that sorts after it.
_MAX_OPTION_CHARACTERS = 160


class LintCapabilityProbe(Protocol):
    """Describe the lint vocabulary a checkout resolves for the files being changed."""

    async def describe(self, repository_root: PathLike, paths: Sequence[str]) -> dict[str, Any]:
        """Return a bounded, JSON-serializable description, empty when nothing is known."""


class NullLintCapabilityProbe:
    """Report nothing; the default for mock execution."""

    async def describe(self, repository_root: PathLike, paths: Sequence[str]) -> dict[str, Any]:
        """Probe nothing and report an empty capability set."""
        del repository_root, paths
        return {}


class RepositoryLintCapabilities:
    """Resolve the checkout's own lint configuration, one representative file per class."""

    def __init__(
        self,
        *,
        process_runner: ProcessRunner | None = None,
        cancellation_token: CancellationToken | None = None,
        timeout_seconds: float = _PROBE_TIMEOUT_SECONDS,
    ) -> None:
        """Bind bounded execution; a probe never blocks the coding attempt it informs."""
        self._process_runner = process_runner or AsyncioProcessRunner()
        self._cancellation_token = cancellation_token or MockCancellationToken()
        self._timeout = timeout_seconds

    async def describe(self, repository_root: PathLike, paths: Sequence[str]) -> dict[str, Any]:
        """Return the vocabulary ESLint resolves, per file class the attempt can write.

        One representative per class -- a production source file and a test file, when the
        checkout contains both -- because a shared configuration routinely applies whole
        plugins through overrides that match only test files. `react-app/jest` binds all of
        `testing-library/*` that way, so a probe that resolved only the first sorted `.js`
        file reported 0 of the 19 testing-library rules the gate then failed the attempt
        on, while telling the model those rules do not exist in this checkout.

        Class membership comes from the checkout's own naming; which rules bind to each
        subject is the tool's answer, which is the point of `--print-config` per subject:
        ESLint applies its own override matching, never reimplemented here.
        """
        root = resolve_workspace_root(repository_root)
        candidates = _eslint_paths(paths)
        if not candidates:
            return {}
        production_subject = next((path for path in candidates if not _is_test_path(path)), None)
        test_subject = next((path for path in candidates if _is_test_path(path)), None)
        production = (
            await self._resolved_for_subject(root, production_subject)
            if production_subject is not None
            else None
        )
        test = (
            await self._resolved_for_subject(root, test_subject)
            if test_subject is not None
            else None
        )
        # Fail-soft per class: a subject that cannot resolve drops its class, never the
        # attempt -- and the prompt's authority claim is scoped to the classes listed, so
        # a blind spot is reported as absence rather than converted into a false assertion.
        if production is not None and test is not None:
            return _merged_eslint_capabilities(production, test)
        resolved = production if production is not None else test
        return {"eslint": resolved} if resolved is not None else {}

    async def _resolved_for_subject(self, root: Path, subject: str) -> dict[str, Any] | None:
        """Resolve one subject's configuration, advisory and bounded like the whole probe."""
        binary = _node_binary(root, "eslint", subject)
        if binary is None:
            return None
        command = (str(binary), "--print-config", subject)
        await self._cancellation_token.raise_if_cancelled()
        result = await self._process_runner.run(
            command,
            root,
            self._timeout,
            self._cancellation_token,
            repository_subprocess_environment(),
        )
        if not result.succeeded:
            # A probe is advisory. Its failure must never fail the coding attempt; the
            # commit gate remains the authority on whether the change is acceptable.
            return None
        capabilities = _eslint_capabilities(result.stdout, subject)
        eslint = capabilities.get("eslint")
        return eslint if isinstance(eslint, dict) else None


# Path segments and stem shapes this platform reads as "a test file". Naming only, and
# deliberately generic: which rules actually bind is ESLint's own answer per subject, so a
# repository with unusual conventions loses nothing but the second probe invocation.
_TEST_PATH_MARKERS = frozenset({"test", "tests", "__tests__", "spec", "specs"})


def _is_test_path(path: str) -> bool:
    """Say whether the checkout's own naming marks this path as a test file."""
    posix = PurePosixPath(path)
    if any(part.lower() in _TEST_PATH_MARKERS for part in posix.parts[:-1]):
        return True
    stem = posix.stem.lower()
    return stem.endswith((".test", ".spec", "_test", "_spec")) or stem.startswith("test_")


def _merged_eslint_capabilities(production: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    """Combine the two class resolutions: the main list, plus what test files add.

    The main list keeps its historical shape, so everything that already reads
    `resolved_for` and `enabled_rules` keeps working. The test class is reported as the
    rules it adds on top of production -- per-class provenance rather than a second full
    list, which is what keeps both classes inside the existing character budget: the real
    pilot checkout adds 29 rules over a 145-rule base, all 19 `testing-library/*` rules
    among them, where a second full list would nearly double the section.
    """
    production_rules = set(production.get("enabled_rules", ()))
    additional = [rule for rule in test.get("enabled_rules", ()) if rule not in production_rules]
    return {
        "eslint": {
            **production,
            "test_files": {
                "resolved_for": test.get("resolved_for"),
                "available_plugin_namespaces": test.get("available_plugin_namespaces", []),
                "enabled_rule_count": test.get("enabled_rule_count", 0),
                "additional_enabled_rules": additional,
                "enabled_rules_truncated": bool(test.get("enabled_rules_truncated")),
            },
        }
    }


def _eslint_capabilities(printed_config: str, subject: str) -> dict[str, Any]:
    """Summarize a resolved ESLint configuration without reproducing the whole document."""
    try:
        config = json.loads(printed_config)
    except json.JSONDecodeError:
        return {}
    rules = config.get("rules")
    if not isinstance(rules, dict):
        return {}
    enabled = sorted(name for name, value in rules.items() if _is_enabled(value))
    namespaces = sorted({name.rsplit("/", 1)[0] for name in enabled if "/" in name})
    reported = _within_budget([_described(name, rules[name]) for name in enabled])
    return {
        "eslint": {
            "resolved_for": subject,
            # A rule outside these namespaces does not exist in this checkout, so naming
            # it in a disable comment is itself a lint error.
            "available_plugin_namespaces": namespaces,
            "enabled_rule_count": len(enabled),
            "enabled_rules": reported,
            "enabled_rules_truncated": len(reported) < len(enabled),
        }
    }


def _described(name: str, value: object) -> str:
    """Render one rule as its name plus the options that say what it actually requires.

    A name alone does not tell a model what to write. ``line-comment-position`` is satisfied
    only by knowing its option is ``above``, and that option is already in the resolved
    configuration; reporting the name and dropping it withholds the actionable half.
    """
    options = value[1:] if isinstance(value, list) and len(value) > 1 else []
    if not options:
        return name
    rendered = json.dumps(options, separators=(",", ":"), sort_keys=True)
    if len(rendered) > _MAX_OPTION_CHARACTERS:
        rendered = f"{rendered[:_MAX_OPTION_CHARACTERS]}…"
    return f"{name}: {rendered}"


def _within_budget(rules: Sequence[str]) -> list[str]:
    """Return as many rule names as fit the character budget, in resolved order."""
    reported: list[str] = []
    used = 0
    for name in rules:
        used += len(name) + 2
        if used > _MAX_RULE_CHARACTERS:
            break
        reported.append(name)
    return reported


def _is_enabled(value: object) -> bool:
    """Return whether one resolved rule entry is anything other than switched off."""
    severity = value[0] if isinstance(value, list) and value else value
    if isinstance(severity, str):
        return severity != "off"
    if isinstance(severity, bool):
        return severity
    return isinstance(severity, int) and severity > 0


__all__ = [
    "LintCapabilityProbe",
    "NullLintCapabilityProbe",
    "RepositoryLintCapabilities",
    "_eslint_capabilities",
]
