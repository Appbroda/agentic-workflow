"""Each repository runs the package-manager version it declares.

`packageManager` is a Corepack provisioning directive -- *use this exact version* -- and
Corepack ships with Node. The platform used to install one pnpm and one Yarn globally and
demand every repository match them, which serves exactly the repositories that happen to
agree. AB-Feature-222 was refused for declaring `npm@9.8.1` against an image carrying npm
10.9.8, having written no code after 537 seconds of planning.

Two repositories pinning different versions must both work without rebuilding the platform,
which is what these tests are about.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import node_toolchain
from tools.node_toolchain import declared_package_manager, package_manager_argv


@pytest.fixture(autouse=True)
def _corepack_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert against the platform's own logic, not against whoever's machine runs the suite.

    Without this the answers below depend on whether `corepack` happens to be on PATH, so the
    suite would pass on a developer's laptop and quietly assert the opposite in a container
    that lacks it. The absent case is its own test.
    """
    monkeypatch.setattr(node_toolchain, "corepack_available", lambda: True)


def _project(root: Path, declaration: str | None, lockfile: str = "package-lock.json") -> Path:
    manifest: dict[str, object] = {"name": root.name}
    if declaration is not None:
        manifest["packageManager"] = declaration
    root.mkdir(parents=True, exist_ok=True)
    (root / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / lockfile).write_text("{}", encoding="utf-8")
    return root


def test_two_repositories_pinning_different_versions_each_get_their_own(tmp_path: Path) -> None:
    """The property the whole change exists for, stated once.

    No shared worker version appears in either command. This is what "it should work for
    different repos, each with different versions" means when written down.
    """
    first = _project(tmp_path / "admin", "npm@9.8.1")
    second = _project(tmp_path / "user", "npm@10.9.8")

    first_argv, first_version = package_manager_argv("npm", first, first)
    second_argv, second_version = package_manager_argv("npm", second, second)

    assert first_argv == ["corepack", "npm@9.8.1"]
    assert second_argv == ["corepack", "npm@10.9.8"]
    assert (first_version, second_version) == ("9.8.1", "10.9.8")


@pytest.mark.parametrize(
    ("declaration", "manager", "expected"),
    [
        ("npm@9.8.1", "npm", ["corepack", "npm@9.8.1"]),
        ("yarn@4.6.0", "yarn", ["corepack", "yarn@4.6.0"]),
        ("pnpm@8.15.9", "pnpm", ["corepack", "pnpm@8.15.9"]),
        # A Berry declaration on a worker whose global Yarn is 1.22 was previously terminal.
        ("yarn@4.6.0-rc.1", "yarn", ["corepack", "yarn@4.6.0-rc.1"]),
    ],
)
def test_every_corepack_manager_is_honoured(
    tmp_path: Path, declaration: str, manager: str, expected: list[str]
) -> None:
    """All three managers, because the old check blocked all three the same way."""
    lockfiles = {"npm": "package-lock.json", "yarn": "yarn.lock", "pnpm": "pnpm-lock.yaml"}
    project = _project(tmp_path / "repo", declaration, lockfiles[manager])

    argv, version = package_manager_argv(manager, project, project)

    assert argv == expected
    assert version == declaration.partition("@")[2]


def test_a_repository_that_declares_nothing_keeps_the_image_toolchain(tmp_path: Path) -> None:
    """The unchanged case, and why Corepack is invoked explicitly rather than enabled globally.

    `corepack enable` replaces `npm` process-wide, and a project declaring nothing then
    resolves to whatever Corepack considers current -- measured as npm 12.0.2 on the worker
    whose image provides 10.9.8. Every repository the platform already builds declares
    nothing, so that would have been a silent toolchain change for all of them.
    """
    project = _project(tmp_path / "repo", None)

    assert package_manager_argv("npm", project, project) == (["npm"], None)


def test_an_inexact_declaration_is_not_guessed_at(tmp_path: Path) -> None:
    """A range is not a version, and provisioning a guess would run something unasked for.

    Preflight reports the unparseable declaration separately; this only declines to invent a
    version for it.
    """
    for declaration in ("npm@^9.8.1", "npm@latest", "npm", "npm@9", "yarn@stable"):
        project = _project(tmp_path / declaration.replace("@", "-").replace("^", ""), declaration)
        assert package_manager_argv("npm", project, project) == (["npm"], None), declaration


def test_a_declaration_conflicting_with_the_lockfile_is_left_to_preflight(tmp_path: Path) -> None:
    """pnpm declared, an npm lockfile committed: a real conflict with its own diagnosis.

    Running the declared manager against a lockfile another one wrote would replace
    `PACKAGE_MANAGER_LOCKFILE_MISMATCH` -- which names both sides -- with a confusing install
    failure.
    """
    project = _project(tmp_path / "repo", "pnpm@9.1.0", "package-lock.json")

    assert package_manager_argv("npm", project, project) == (["npm"], None)


def test_a_workspace_package_inherits_the_monorepo_declaration(tmp_path: Path) -> None:
    """Corepack resolves the nearest declaration, and a monorepo declares once at its root."""
    root = _project(tmp_path / "repo", "pnpm@9.1.0", "pnpm-lock.yaml")
    package = root / "packages" / "web"
    package.mkdir(parents=True)
    (package / "package.json").write_text(json.dumps({"name": "web"}), encoding="utf-8")

    argv, version = package_manager_argv("pnpm", package, root)

    assert argv == ["corepack", "pnpm@9.1.0"]
    assert version == "9.1.0"
    # And the nearer declaration wins when there is one.
    (package / "package.json").write_text(
        json.dumps({"name": "web", "packageManager": "pnpm@8.15.9"}), encoding="utf-8"
    )
    assert declared_package_manager(package, root) == ("pnpm", "8.15.9")


def test_the_search_never_escapes_the_repository(tmp_path: Path) -> None:
    """A declaration outside the checkout is not this repository's, whoever wrote it.

    The worker's workspace root holds every feature's clones side by side, so walking past the
    repository root would read a sibling repository's pin -- or the platform's own.
    """
    outside = _project(tmp_path, "npm@9.8.1")
    inner = outside / "repo"
    inner.mkdir()
    (inner / "package.json").write_text(json.dumps({"name": "inner"}), encoding="utf-8")

    assert declared_package_manager(inner, inner) is None
    assert package_manager_argv("npm", inner, inner) == (["npm"], None)


def test_an_unreadable_manifest_does_not_raise(tmp_path: Path) -> None:
    """Preflight has its own diagnosis for a broken manifest; this must not pre-empt it."""
    project = tmp_path / "repo"
    project.mkdir()
    (project / "package.json").write_text("{ not json", encoding="utf-8")

    assert package_manager_argv("npm", project, project) == (["npm"], None)


def test_a_manager_corepack_does_not_manage_is_untouched(tmp_path: Path) -> None:
    """uv is a Python installer and reaches this code through the same install plumbing."""
    project = _project(tmp_path / "repo", "npm@9.8.1")

    assert package_manager_argv("uv", project, project) == (["uv"], None)


def test_an_image_without_corepack_falls_back_instead_of_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker with no Corepack keeps working on the manager it has.

    Preflight then judges whether that manager can read the committed lockfile, which is a
    weaker guarantee than provisioning the declared version but is not a refusal to run.
    """
    monkeypatch.setattr(node_toolchain, "corepack_available", lambda: False)
    project = _project(tmp_path / "repo", "npm@9.8.1")

    assert package_manager_argv("npm", project, project) == (["npm"], None)
