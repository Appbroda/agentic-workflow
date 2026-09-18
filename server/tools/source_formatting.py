"""Apply a repository's own formatters to the files the platform just wrote.

A repository that enforces formatting in a pre-commit hook will reject generated code
that is functionally correct but not styled to its conventions. Rather than bypassing the
hook or spending a retry asking a model to reproduce exact formatter output, the platform
runs the repository's own installed tools over exactly the paths it changed.

Only tools already present in the checkout are invoked, always with explicit file paths
and never through a shell, so this cannot reformat unrelated files or fetch anything.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from services.cancellation import CancellationToken, MockCancellationToken
from services.process_runner import (
    AsyncioProcessRunner,
    ProcessResult,
    ProcessRunner,
    repository_subprocess_environment,
    subprocess_result_code,
)
from tools.file_tools import PathLike, resolve_workspace_root

# Prettier reformats any of these, but ESLint parses only JavaScript. Handing it a config
# file such as `.eslintrc.json` makes it read JSON as a program and report a parsing error
# against a file the engineer never got wrong.
_ESLINT_SUFFIXES = frozenset({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"})
_PRETTIER_SUFFIXES = _ESLINT_SUFFIXES | {".css", ".scss", ".json", ".md"}
_PYTHON_SUFFIXES = frozenset({".py", ".pyi"})
_FORMAT_TIMEOUT_SECONDS = 120.0
_GATE_TIMEOUT_SECONDS = 900.0
_MAX_DIAGNOSTIC_CHARACTERS = 4_000
_ERROR_LINE = re.compile(r"\berror\b", re.IGNORECASE)
_SUMMARY_LINE = re.compile(r"\bproblems?\b|\bfailed\b", re.IGNORECASE)


class SourceValidationError(RuntimeError):
    """Report safe, actionable source diagnostics before Git is allowed to stage files."""

    # The completion the Engineer produced before the gate rejected it, attached by the
    # Engineer on its way out. For `validation_source_failure` -- the dominant failure class --
    # there used to be no record at all of what the model was shown or wrote: AB-Feature-171's
    # attempts 0, 2 and 4 each discarded a full implementation with nothing durable but the
    # linter's sentence. Typed loosely because this module sits below the artifact schemas.
    code_completion: Any | None = None
    # Every workspace-relative path written by the attempt, including repair passes that ran
    # after the first rejection. The caller that builds the failed completion needs them; the
    # exception is the only channel that survives the raise.
    modified_paths: tuple[str, ...] = ()
    # The named reason the attempt stopped, when something other than the gate's own verdict
    # decided it -- today, an in-attempt repair the assertion guard refused. Recorded on the
    # completion so the escape is answerable from durable state rather than from reading a
    # raw lint or test line and inferring what happened.
    terminal_outcome: str | None = None
    # The repair passes that actually ran before the gate rejected the attempt. Carried here
    # for the same reason `modified_paths` is: the loop accumulates them in a local, and the
    # raise is the only channel out. Without this, a gate-rejected attempt recorded no
    # `source_repair_*` block at all, so "the repair loop never engaged" and "it ran four
    # passes on the scoped-fix model and lost" were byte-identical in the artifact record --
    # which is what made the anthropic scoped-fix loop unauditable across runs 183 and 186.
    # Typed loosely for the same reason as `code_completion`: this module sits below the
    # Engineer's own `_SourceRepair`.
    source_repairs: tuple[Any, ...] = ()
    # The self-review record for the attempt this rejection ends, when a review ran before
    # or caused it. Carried for the same reason `source_repairs` is: the raise is the only
    # channel out, and the attempts a review stops are exactly the ones whose measurement
    # matters most. None means no review ran, never that one ran and found nothing.
    self_review: dict[str, Any] | None = None
    # The structured results of the gate's own runs that completed and rejected the source,
    # in the shape `current_validation_results` is persisted in. Carried here for the reason
    # `source_repairs` is -- the raise is the only channel out -- and only for the checks
    # that produce one: a lint or typecheck rejection contributes nothing, so the record of
    # those attempts is exactly what it was. What this closes is 49-C going blind on a
    # gate-rejected attempt: it reads `current_validation_results`, that field was empty for
    # every commit-gate rejection, and "empty" is correctly read as *unknown*, which breaks
    # the walk-back. AB-Feature-218's four consecutive failures on one suite counted as two.
    validation_results: tuple[dict[str, Any], ...] = ()

    def __init__(self, diagnostics: Sequence[str]) -> None:
        """Keep platform-owned result codes available to the retry planner and operator."""
        self.diagnostics = tuple(diagnostics)
        super().__init__("generated source did not pass the repository's pre-commit validation")


class SourceFormatter(Protocol):
    """Normalize generated files using tooling the repository already declares."""

    async def format_paths(
        self, repository_root: PathLike, paths: Sequence[str]
    ) -> tuple[str, ...]:
        """Return the commands that were applied, which may be empty."""


class NullSourceFormatter:
    """Leave generated files exactly as written; the default for mock execution."""

    async def format_paths(
        self, repository_root: PathLike, paths: Sequence[str]
    ) -> tuple[str, ...]:
        """Perform no formatting and report that nothing ran."""
        del repository_root, paths
        return ()


class RepositoryToolFormatter:
    """Format and verify only the generated paths before they can be committed."""

    def __init__(
        self,
        *,
        process_runner: ProcessRunner | None = None,
        cancellation_token: CancellationToken | None = None,
        timeout_seconds: float = _FORMAT_TIMEOUT_SECONDS,
        gate: tuple[Sequence[str], str] | None = None,
        gate_timeout_seconds: float = _GATE_TIMEOUT_SECONDS,
        gate_already_failing: bool = False,
    ) -> None:
        """Bind bounded execution; formatting never blocks on an unbounded process.

        ``gate`` is the repository's own commit gate and the directory it runs in, when the
        checkout declares one. Verifying with the exact command the hook runs is the only way
        to agree with it: a private approximation is both stricter, flagging files the
        repository's own script excludes, and weaker, missing the flags that decide whether a
        warning fails the build. The directory is part of that command, since a package in a
        monorepo lints from its own manifest rather than from the checkout root.
        """
        self._process_runner = process_runner or AsyncioProcessRunner()
        self._cancellation_token = cancellation_token or MockCancellationToken()
        self._timeout = timeout_seconds
        self._gate_command = tuple(gate[0]) if gate else None
        self._gate_directory = gate[1] if gate else "."
        self._gate_timeout = gate_timeout_seconds
        self._gate_already_failing = gate_already_failing

    async def format_paths(
        self, repository_root: PathLike, paths: Sequence[str]
    ) -> tuple[str, ...]:
        """Format generated paths and reject remaining lint defects before Git staging."""
        root = resolve_workspace_root(repository_root)
        applied: list[str] = []
        diagnostics: list[str] = []
        # A fixer's own non-zero exit is held back rather than reported beside the gate's.
        # `eslint --fix` exits 1 for exactly the violations it could not fix, which is what
        # the verification below is about to report authoritatively -- so both stages
        # described the same defect and only the `tool=` token differed, which is why
        # `dict.fromkeys` could not collapse them. Seventeen attempts in the last fifteen
        # features carried the same eslint error twice into the engineer's feedback.
        fixer_diagnostics: list[str] = []
        for command in _formatting_commands(root, paths):
            await self._cancellation_token.raise_if_cancelled()
            result = await self._process_runner.run(
                command,
                root,
                self._timeout,
                self._cancellation_token,
                repository_subprocess_environment(),
            )
            if result.cancelled:
                await self._cancellation_token.raise_if_cancelled()
            if not result.cancelled:
                applied.append(" ".join(command))
            if not result.succeeded:
                fixer_diagnostics.append(_validation_diagnostic(command, result))
        # A fixer may exit successfully while still leaving non-fixable rule violations.
        # Verify now, before ``git add`` gives the commit hook the first opportunity to find
        # them. The repository's own gate decides when it declares one; only a checkout that
        # declares none falls back to checking the generated paths directly.
        verification = (
            [self._gate_command] if self._gate_command else _verification_commands(root, paths)
        )
        using_gate = self._gate_command is not None
        gate_directory = _gate_directory(root, self._gate_directory) if using_gate else root
        # The gate lints the whole project; the per-path fallback does not.
        verification_timeout = self._gate_timeout if using_gate else self._timeout
        for command in verification:
            await self._cancellation_token.raise_if_cancelled()
            result = await self._process_runner.run(
                command,
                gate_directory,
                verification_timeout,
                self._cancellation_token,
                repository_subprocess_environment(),
            )
            if result.cancelled:
                await self._cancellation_token.raise_if_cancelled()
            if not result.cancelled:
                applied.append(" ".join(command))
            if not result.succeeded:
                diagnostics.append(_validation_diagnostic(command, result))
        # Only when verification found nothing to say. A fixer that failed for its own
        # reasons -- a broken config, a parser crash, a missing binary -- reports something
        # the verification stage never sees, and losing that would leave a rejected attempt
        # with no diagnostic at all.
        if not diagnostics:
            diagnostics = fixer_diagnostics
        if diagnostics:
            if self._gate_already_failing:
                # The gate rejected this checkout before the engineer touched it, so the
                # findings are not necessarily its work and rewriting the change cannot
                # clear them. Say so, rather than sending it to fix somebody else's defect.
                diagnostics.append(
                    "This repository's commit gate was already failing on its own checkout "
                    "before this change. Its default branch has to be repaired first; no "
                    "generated change can pass this gate until then."
                )
            raise SourceValidationError(tuple(dict.fromkeys(diagnostics)))
        return tuple(applied)


def _formatting_commands(root: Path, paths: Sequence[str]) -> list[tuple[str, ...]]:
    """Select fixer invocations evidenced by this checkout for these exact paths."""
    python_paths = [path for path in paths if Path(path).suffix in _PYTHON_SUFFIXES]
    commands: list[tuple[str, ...]] = []
    # Prettier owns layout; ESLint --fix then resolves rule-level auto-fixes. Running them
    # in this order matches how these repositories are configured to lint.
    commands.extend(_node_commands(root, _prettier_paths(paths), "prettier", "--write"))
    commands.extend(_node_commands(root, _eslint_paths(paths), "eslint", "--fix"))
    if python_paths and _ruff_configured(root):
        commands.append(("ruff", "format", *python_paths))
        commands.append(("ruff", "check", "--fix", *python_paths))
    return commands


def _verification_commands(root: Path, paths: Sequence[str]) -> list[tuple[str, ...]]:
    """Select non-mutating lint checks matching the auto-fixers used above."""
    python_paths = [path for path in paths if Path(path).suffix in _PYTHON_SUFFIXES]
    commands: list[tuple[str, ...]] = list(_node_commands(root, _eslint_paths(paths), "eslint"))
    if python_paths and _ruff_configured(root):
        commands.append(("ruff", "check", *python_paths))
    return commands


def _gate_directory(root: Path, working_directory: str) -> Path:
    """Resolve the gate's directory, refusing one that leaves the checkout."""
    candidate = (root / working_directory).resolve()
    if not (candidate == root or root in candidate.parents) or not candidate.is_dir():
        return root
    return candidate


def _eslint_paths(paths: Sequence[str]) -> list[str]:
    """Return only the paths ESLint can parse as a program."""
    return [path for path in paths if Path(path).suffix in _ESLINT_SUFFIXES]


def _prettier_paths(paths: Sequence[str]) -> list[str]:
    """Return the paths Prettier reformats, including stylesheets, JSON, and Markdown."""
    return [path for path in paths if Path(path).suffix in _PRETTIER_SUFFIXES]


def _node_commands(
    root: Path, paths: Sequence[str], tool: str, *flags: str
) -> list[tuple[str, ...]]:
    """Group paths under the installation that actually owns them.

    A frontend held in a subdirectory installs its own ``node_modules``, and that package's
    binary is the one its Husky hook runs. Resolving only at the repository root would find
    no formatter there and leave exactly the violations this module exists to prevent.
    """
    grouped: dict[Path, list[str]] = {}
    for path in paths:
        binary = _node_binary(root, tool, path)
        if binary is not None:
            grouped.setdefault(binary, []).append(path)
    return [(str(binary), *flags, *group) for binary, group in grouped.items()]


def _validation_diagnostic(command: tuple[str, ...], result: ProcessResult) -> str:
    """Return a stable failure code together with the lines that say what to repair.

    The code alone classifies the failure but cannot resolve it. This diagnostic becomes the
    retry feedback the coding model is given, and feature -057 showed what happens without
    content: two attempts received the identical `code=SOURCE_VALIDATION_FAILED_EXIT_1`, so
    the second had nothing the first did not and reproduced the same violation exactly.

    The excerpt is safe to carry: `ProcessRunner` redacts stdout and stderr before building
    the result, so what arrives here has already had known secrets removed.
    """
    tool = Path(command[0]).name if command else "unknown"
    code = subprocess_result_code(
        return_code=result.return_code,
        timed_out=result.timed_out,
        cancelled=result.cancelled,
        prefix="source_validation",
    )
    headline = f"Pre-commit source validation failed (tool={tool}; code={code})."
    excerpt = _failure_lines(f"{result.stdout}\n{result.stderr}".strip())
    if not excerpt.strip():
        return headline
    if len(excerpt) > _MAX_DIAGNOSTIC_CHARACTERS:
        excerpt = f"{excerpt[:_MAX_DIAGNOSTIC_CHARACTERS]}\n[truncated]"
    return f"{headline}\n{excerpt}"


def _failure_lines(output: str) -> str:
    """Keep the lines that name a failure, with the file each one belongs to.

    A project-wide lint prints hundreds of warnings and a handful of errors. Truncating the
    transcript at a fixed length kept the warnings and cut the errors away, so a retry was
    handed a diagnostic that never showed the defect it was asked to repair.
    """
    if not output:
        return output
    lines = output.splitlines()
    if not any(_ERROR_LINE.search(line) for line in lines):
        return output
    kept: list[str] = []
    current_file: str | None = None
    emitted_file: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("/") or stripped.startswith("./"):
            current_file = line
            continue
        if _ERROR_LINE.search(line):
            if current_file is not None and current_file != emitted_file:
                kept.append(current_file)
                emitted_file = current_file
            kept.append(line)
        elif _SUMMARY_LINE.search(line):
            kept.append(line)
    return "\n".join(kept) if kept else output


def _node_binary(root: Path, tool: str, relative_path: str) -> Path | None:
    """Return the nearest checkout-local executable, never one resolved from the host PATH."""
    # Resolved before the walk so a traversing path cannot reach an installation that
    # lives outside the checkout the platform was given.
    directory = (root / relative_path).resolve().parent
    while directory == root or root in directory.parents:
        candidate = directory / "node_modules" / ".bin" / tool
        if candidate.is_file():
            return candidate
        if directory == root:
            return None
        directory = directory.parent
    return None


def _ruff_configured(root: Path) -> bool:
    """Detect an explicit Ruff configuration before reformatting Python sources."""
    if (root / "ruff.toml").is_file() or (root / ".ruff.toml").is_file():
        return True
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        return "[tool.ruff" in pyproject.read_text(encoding="utf-8")
    except OSError:
        return False


__all__ = [
    "NullSourceFormatter",
    "RepositoryToolFormatter",
    "SourceFormatter",
    "SourceValidationError",
]
