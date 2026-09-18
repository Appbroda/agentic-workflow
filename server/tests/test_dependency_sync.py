"""Installing what generated code declares, before the linter tries to resolve it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from services.process_runner import ProcessResult
from tests.fixtures import node_javascript_repository, python_fastapi_repository
from tools.dependency_sync import (
    NullDependencySynchronizer,
    RepositoryToolSynchronizer,
    _install_commands,
    generated_lockfile_manifest,
    installed_lockfiles,
)
from tools.source_formatting import SourceValidationError


class CapturingProcessRunner:
    """Record install commands without running a package manager."""

    def __init__(self, *, fail: bool = False) -> None:
        """Track invocations and optionally simulate an install that cannot resolve."""
        self.invocations: list[tuple[tuple[str, ...], Path]] = []
        self._fail = fail

    async def run(self, command: Any, cwd: Path, *_args: Any, **_kwargs: Any) -> ProcessResult:
        """Return a deterministic result for one install invocation."""
        received = tuple(command)
        self.invocations.append((received, cwd))
        return ProcessResult(
            command=received,
            return_code=1 if self._fail else 0,
            stdout="",
            stderr="npm ERR! notarget No matching version found for supertest@^99"
            if self._fail
            else "",
            duration_seconds=0.01,
        )


@pytest.mark.asyncio
async def test_a_changed_manifest_is_installed_with_the_repositories_manager(
    tmp_path: Path,
) -> None:
    """A newly declared package must exist in the checkout before anything resolves it."""
    root = node_javascript_repository(tmp_path / "web")
    (root / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    runner = CapturingProcessRunner()

    applied = await RepositoryToolSynchronizer(process_runner=runner).sync(
        root, ["package.json", "src/a.js"]
    )

    assert len(runner.invocations) == 1
    command, cwd = runner.invocations[0]
    # A resolving install, not preflight's frozen one: the manifest just changed. `--no-audit`
    # because the advisory report is the call that hung every install on 2026-09-03, and a
    # sync inside an attempt dies on it just as dead as preflight did.
    assert command == ("npm", "install", "--no-audit")
    assert cwd == root
    assert applied == ("npm install --no-audit",)


@pytest.mark.asyncio
async def test_source_only_changes_never_trigger_an_install(tmp_path: Path) -> None:
    """Nothing is installed when the engineer did not touch a dependency manifest."""
    root = node_javascript_repository(tmp_path / "web")
    (root / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    runner = CapturingProcessRunner()

    applied = await RepositoryToolSynchronizer(process_runner=runner).sync(
        root, ["src/a.js", "src/b.js"]
    )

    assert runner.invocations == []
    assert applied == ()


@pytest.mark.asyncio
async def test_a_failed_install_is_reported_before_the_commit_is_attempted(
    tmp_path: Path,
) -> None:
    """An unresolvable dependency becomes retry feedback, not a Husky failure.

    The feedback has to name the requirement that could not be resolved. The model wrote the
    manifest, so the package and range are the only facts that let it correct the
    declaration; a bare exit code leaves it to guess, which is how -057 burned both of its
    attempts on one unchanged lint failure.
    """
    root = node_javascript_repository(tmp_path / "web")
    (root / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    runner = CapturingProcessRunner(fail=True)

    with pytest.raises(SourceValidationError) as error:
        await RepositoryToolSynchronizer(process_runner=runner).sync(root, ["package.json"])

    assert "DEPENDENCY_INSTALLATION_FAILED_EXIT_1" in error.value.diagnostics[0]
    assert "supertest@^99" in error.value.diagnostics[0]


def test_a_subdirectory_manifest_installs_in_its_own_package(tmp_path: Path) -> None:
    """A monorepo package is installed where its lockfile lives, not at the checkout root."""
    root = node_javascript_repository(tmp_path / "monorepo")
    (root / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    web = root / "web"
    web.mkdir(parents=True, exist_ok=True)
    (web / "package.json").write_text('{"name": "web"}\n', encoding="utf-8")
    (web / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")

    commands = _install_commands(root, ["web/package.json"])

    assert commands == [(("pnpm", "install"), web)]


def test_a_python_manifest_is_installed_only_with_a_frozen_lock(tmp_path: Path) -> None:
    """Without a lockfile the platform would be guessing at a resolution the repo never chose."""
    locked = python_fastapi_repository(tmp_path / "locked")
    (locked / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    unlocked = python_fastapi_repository(tmp_path / "unlocked")

    assert _install_commands(locked, ["pyproject.toml"]) == [(("uv", "sync"), locked)]
    assert _install_commands(unlocked, ["pyproject.toml"]) == []


def test_a_manifest_outside_the_checkout_is_never_installed(tmp_path: Path) -> None:
    """A traversing path must not reach a package directory the platform was not given."""
    root = node_javascript_repository(tmp_path / "web")
    (root / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    (tmp_path / "package.json").write_text('{"name": "outside"}\n', encoding="utf-8")

    assert _install_commands(root, ["../package.json"]) == []


def test_a_regenerated_lockfile_is_reported_for_commit(tmp_path: Path) -> None:
    """A manifest committed without its lockfile breaks every later frozen install."""
    root = node_javascript_repository(tmp_path / "web")
    (root / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")

    assert installed_lockfiles(root, ["package.json"]) == ["package-lock.json"]
    # Nothing was installed, so nothing was regenerated.
    assert installed_lockfiles(root, ["src/a.js"]) == []


def test_a_generated_lockfile_resolves_to_the_manifest_beside_it() -> None:
    """This module is where "is that machine-generated output?" is answered for the platform.

    The reviewer's evidence policy asks it, because a lockfile too large to read is a very
    different thing from an unreadable source file, and it must not carry a second filename
    list of its own to answer with -- one that would drift the first time a package manager
    was added here. Every name in the install registry answers, and the pairing is positional,
    because that is what `_install_commands` already assumes when it runs an install in the
    manifest's own directory.
    """
    assert generated_lockfile_manifest("package-lock.json") == "package.json"
    assert generated_lockfile_manifest("yarn.lock") == "package.json"
    assert generated_lockfile_manifest("pnpm-lock.yaml") == "package.json"
    assert generated_lockfile_manifest("uv.lock") == "pyproject.toml"
    # A workspace package in a monorepo pairs with its own manifest, not the root's.
    assert generated_lockfile_manifest("packages/web/package-lock.json") == (
        "packages/web/package.json"
    )
    # And nothing else is a lockfile, however much it looks like one.
    assert generated_lockfile_manifest("package.json") is None
    assert generated_lockfile_manifest("src/package-lock.json.bak") is None
    assert generated_lockfile_manifest("composer.lock") is None
    assert generated_lockfile_manifest("src/index.js") is None


@pytest.mark.asyncio
async def test_the_default_synchronizer_installs_nothing(tmp_path: Path) -> None:
    """Mock execution must not depend on any package manager being present."""
    assert await NullDependencySynchronizer().sync(tmp_path, ["package.json"]) == ()
