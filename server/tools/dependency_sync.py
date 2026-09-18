"""Install what generated code declares, before anything tries to resolve it.

Generated code that imports a package the checkout does not have fails at the pre-commit
gate with an unresolved import, and no amount of rewriting the source fixes it. When the
engineer declares a new dependency in a manifest, the platform installs it with the
repository's own package manager so the import resolves and the lockfile stays consistent.

This is deliberately narrower than preflight bootstrapping. Preflight installs only frozen,
already-declared dependencies and never resolves new ones; this runs a resolving install
*only* when the engineer actually changed a manifest, which is the one moment the lockfile
is expected to move.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from shutil import which
from typing import Protocol

from services.cancellation import CancellationToken, MockCancellationToken
from services.process_runner import (
    AsyncioProcessRunner,
    ProcessResult,
    ProcessRunner,
    repository_subprocess_environment,
    subprocess_result_code,
)
from tools.file_tools import PathLike, resolve_workspace_root
from tools.source_formatting import (
    _MAX_DIAGNOSTIC_CHARACTERS,
    SourceValidationError,
    _failure_lines,
)

_NODE_MANIFEST = "package.json"
_PYTHON_MANIFEST = "pyproject.toml"
_INSTALL_TIMEOUT_SECONDS = 600.0

# A resolving install, unlike preflight's frozen one: the manifest just changed, so the
# lockfile is expected to move with it. `--no-audit` for the same reason preflight's
# `npm ci` carries it: the advisory report is a live POST to registry endpoints that have
# been observed to accept connections and never answer, it verifies nothing (integrity
# hashes are checked regardless), and nothing here reads it. pnpm and yarn do not audit
# on install, so only npm needs saying.
_NODE_INSTALL_COMMANDS = {
    "pnpm": ("pnpm", "install"),
    "yarn": ("yarn", "install"),
    "npm": ("npm", "install", "--no-audit"),
}
_NODE_LOCKFILES = {
    "pnpm-lock.yaml": "pnpm",
    "yarn.lock": "yarn",
    "package-lock.json": "npm",
}
_PYTHON_LOCKFILE = "uv.lock"
# Which manifest each lockfile is regenerated from, derived from the registry above rather
# than restated. This module is the only place that knows a lockfile is machine-generated
# output rather than authored source, and other layers have to be able to ask: the reviewer's
# evidence policy treats an unreadably large lockfile differently from an unreadably large
# source file, and it must resolve that identity here so a lockfile added to `_NODE_LOCKFILES`
# is known everywhere at once instead of in one list that drifts from another.
_LOCKFILE_MANIFESTS: dict[str, str] = {
    **dict.fromkeys(_NODE_LOCKFILES, _NODE_MANIFEST),
    _PYTHON_LOCKFILE: _PYTHON_MANIFEST,
}
# Which manager wrote each lockfile, on the same terms and for the same reason: a caller that
# has to read a lockfile's contents needs to know whose syntax it is in, and asking here means
# a manager added to `_NODE_LOCKFILES` is understood by that caller too. The Node half is the
# registry itself -- it already maps name to manager -- so only `uv` is stated.
_LOCKFILE_MANAGERS: dict[str, str] = {**_NODE_LOCKFILES, _PYTHON_LOCKFILE: "uv"}


class DependencySynchronizer(Protocol):
    """Reconcile the checkout with the dependencies generated code declares."""

    async def sync(self, repository_root: PathLike, paths: Sequence[str]) -> tuple[str, ...]:
        """Return the install commands that were applied, which may be empty."""


class NullDependencySynchronizer:
    """Install nothing; the default for mock execution."""

    async def sync(self, repository_root: PathLike, paths: Sequence[str]) -> tuple[str, ...]:
        """Perform no installation and report that nothing ran."""
        del repository_root, paths
        return ()


class RepositoryToolSynchronizer:
    """Run the checkout's own package manager for each manifest the engineer changed."""

    def __init__(
        self,
        *,
        process_runner: ProcessRunner | None = None,
        cancellation_token: CancellationToken | None = None,
        timeout_seconds: float = _INSTALL_TIMEOUT_SECONDS,
    ) -> None:
        """Bind bounded execution; an install never blocks on an unbounded process."""
        self._process_runner = process_runner or AsyncioProcessRunner()
        self._cancellation_token = cancellation_token or MockCancellationToken()
        self._timeout = timeout_seconds

    async def sync(self, repository_root: PathLike, paths: Sequence[str]) -> tuple[str, ...]:
        """Install declared dependencies for every changed manifest, or explain the failure."""
        root = resolve_workspace_root(repository_root)
        applied: list[str] = []
        diagnostics: list[str] = []
        for command, working_directory in _install_commands(root, paths):
            if which(command[0]) is None:
                # The lockfile picks the manager, so the platform can select one its own
                # image lacks. Say which tool is missing instead of surfacing a bare
                # "No such file", which reads like a defect in the generated change.
                diagnostics.append(
                    f"Dependency installation requires `{command[0]}`, which this "
                    f"repository's lockfile selects but the platform image does not provide."
                )
                continue
            await self._cancellation_token.raise_if_cancelled()
            result = await self._process_runner.run(
                command,
                working_directory,
                self._timeout,
                self._cancellation_token,
                repository_subprocess_environment(),
            )
            if result.cancelled:
                await self._cancellation_token.raise_if_cancelled()
                continue
            applied.append(" ".join(command))
            if not result.succeeded:
                diagnostics.append(_install_diagnostic(command, result))
        if diagnostics:
            raise SourceValidationError(tuple(diagnostics))
        return tuple(applied)


def installed_lockfiles(repository_root: PathLike, paths: Sequence[str]) -> list[str]:
    """Return the lockfiles an install for these manifests may have rewritten.

    A lockfile the platform regenerated has to be committed beside the manifest that caused
    it. Leaving it behind publishes a dependency declaration no frozen install can satisfy,
    which breaks every later checkout rather than only this change.
    """
    root = resolve_workspace_root(repository_root)
    found: dict[str, None] = {}
    for _command, directory in _install_commands(root, paths):
        for name in _LOCKFILE_MANIFESTS:
            candidate = directory / name
            if candidate.is_file():
                found[candidate.relative_to(root).as_posix()] = None
    return list(found)


def generated_lockfile_manifest(relative_path: PathLike) -> str | None:
    """Return the manifest this path is a generated lockfile of, or ``None`` if it is not one.

    The inverse of what `installed_lockfiles` answers, and deliberately the same registry: a
    caller asking "is this machine-generated dependency output, and what declares it?" must
    get an answer that changes when this module's own understanding of package managers
    changes. The pairing is positional -- a lockfile is regenerated from the manifest in its
    own directory -- which is what `_install_commands` already assumes when it runs an install
    in the manifest's directory.

    Purely a name-and-directory question: nothing is read, nothing is stat'd, and no
    repository has to exist. The path is interpreted as workspace-relative and POSIX, which is
    how every reviewed path in this platform is spelled.
    """
    path = PurePosixPath(str(relative_path))
    manifest = _LOCKFILE_MANIFESTS.get(path.name)
    if manifest is None:
        return None
    parent = path.parent
    return manifest if parent in {PurePosixPath("."), PurePosixPath("")} else f"{parent}/{manifest}"


def generated_lockfile_manager(relative_path: PathLike) -> str | None:
    """Return the package manager whose output this path is, or ``None`` if it is not one.

    The same name-and-directory question `generated_lockfile_manifest` answers, against the
    same registry, differing only in which half of the pair is wanted. The reviewer's evidence
    needs both: the manifest to say what a lockfile change should be explained by, and the
    manager to know whose syntax the file's own bytes are in when something has to read them
    -- npm's resolutions are JSON, yarn's are not, and a scan that guessed would report a
    lockfile it could not parse as resolving from nothing.

    Nothing is read and no repository has to exist; the path is interpreted as
    workspace-relative and POSIX, the way every reviewed path in this platform is spelled.
    """
    return _LOCKFILE_MANAGERS.get(PurePosixPath(str(relative_path)).name)


def _install_commands(root: Path, paths: Sequence[str]) -> list[tuple[tuple[str, ...], Path]]:
    """Select one resolving install per changed manifest, in a deterministic order."""
    commands: dict[tuple[tuple[str, ...], Path], None] = {}
    for path in paths:
        candidate = (root / path).resolve()
        if not (candidate == root or root in candidate.parents):
            continue
        directory = candidate.parent
        if candidate.name == _NODE_MANIFEST:
            manager = _node_package_manager(directory, root)
            if manager is not None:
                commands[(_NODE_INSTALL_COMMANDS[manager], directory)] = None
        elif candidate.name == _PYTHON_MANIFEST and (directory / _PYTHON_LOCKFILE).is_file():
            commands[(("uv", "sync"), directory)] = None
    return list(commands)


def _node_package_manager(directory: Path, root: Path) -> str | None:
    """Identify the manager from the nearest checked-in lockfile, never from the host."""
    current = directory
    while current == root or root in current.parents:
        for lockfile, manager in _NODE_LOCKFILES.items():
            if (current / lockfile).is_file():
                return manager
        if current == root:
            return None
        current = current.parent
    return None


def _install_diagnostic(command: tuple[str, ...], result: ProcessResult) -> str:
    """Return the failure code together with what the package manager actually reported.

    This diagnostic is the retry feedback for a manifest the coding model wrote itself, so
    the unresolvable requirement is the one fact that lets it correct the declaration.
    Directing it to read a command log it has no access to left it guessing instead.

    `ProcessRunner` redacts stdout and stderr before building the result, so a registry URL
    carrying a token does not survive into this text.
    """
    del command
    code = subprocess_result_code(
        return_code=result.return_code,
        timed_out=result.timed_out,
        cancelled=result.cancelled,
        prefix="dependency_installation",
    )
    headline = f"Dependency installation failed ({code})."
    excerpt = _failure_lines(f"{result.stdout}\n{result.stderr}".strip())
    if not excerpt.strip():
        return headline
    if len(excerpt) > _MAX_DIAGNOSTIC_CHARACTERS:
        excerpt = f"{excerpt[:_MAX_DIAGNOSTIC_CHARACTERS]}\n[truncated]"
    return f"{headline}\n{excerpt}"


__all__ = [
    "DependencySynchronizer",
    "NullDependencySynchronizer",
    "RepositoryToolSynchronizer",
    "generated_lockfile_manager",
    "generated_lockfile_manifest",
    "installed_lockfiles",
]
