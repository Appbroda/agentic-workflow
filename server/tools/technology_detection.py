"""Deterministic repository technology and revision inspection.

This module deliberately derives its answers from checked-out repository files.  It is
used before validation and review so neither command selection nor review scope depends
on a model guessing a repository's implementation language.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from pathlib import Path

from pydantic import Field

from services.process_runner import repository_subprocess_environment
from state.models import StateModel
from tools.file_tools import PathLike, resolve_workspace_root


class RepositoryTechnologyProfile(StateModel):
    """Evidence-backed technology summary for one repository checkout."""

    repository_id: str = Field(min_length=1)
    primary_language: str = Field(min_length=1)
    languages: list[str] = Field(default_factory=list)
    package_managers: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    test_frameworks: list[str] = Field(default_factory=list)
    linters: list[str] = Field(default_factory=list)
    formatters: list[str] = Field(default_factory=list)
    typecheckers: list[str] = Field(default_factory=list)
    build_tools: list[str] = Field(default_factory=list)
    detected_files: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class RepositoryRevision(StateModel):
    """A content-sensitive revision for committed and uncommitted workspace state."""

    head_sha: str | None = None
    index_fingerprint: str = Field(min_length=1)
    working_tree_fingerprint: str = Field(min_length=1)
    untracked_fingerprint: str = Field(min_length=1)
    combined_fingerprint: str = Field(min_length=1)


def inspect_repository_technology(
    repository_root: PathLike, *, repository_id: str | None = None
) -> RepositoryTechnologyProfile:
    """Inspect standard manifests/configuration files without executing repository code."""
    root = resolve_workspace_root(repository_root)
    identifier = repository_id or root.name or "repository"
    files = _detected_files(root)
    file_names = {path.name for path in files}
    suffixes = {path.suffix.lower() for path in _source_files(root)}
    package = _read_json(root / "package.json")
    package_text = json.dumps(package, sort_keys=True).lower() if package else ""
    pyproject_text = _read_text(root / "pyproject.toml").lower()
    requirements_text = "\n".join(
        _read_text(root / name).lower()
        for name in ("requirements.txt", "setup.py", "setup.cfg", "tox.ini", "pytest.ini")
    )
    all_text = f"{pyproject_text}\n{requirements_text}\n{package_text}"

    languages: set[str] = set()
    if ({".py", ".pyi"} & suffixes) or any(
        name in file_names
        for name in ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg", "tox.ini")
    ):
        languages.add("Python")
    if ".ts" in suffixes or ".tsx" in suffixes or (root / "tsconfig.json").is_file():
        languages.add("TypeScript")
    if ({".js", ".jsx", ".mjs", ".cjs"} & suffixes) or (root / "package.json").is_file():
        languages.add("JavaScript")
    if ".java" in suffixes or (root / "pom.xml").is_file() or (root / "build.gradle").is_file():
        languages.add("Java")
    if ".go" in suffixes or (root / "go.mod").is_file():
        languages.add("Go")
    if ".rs" in suffixes or (root / "Cargo.toml").is_file():
        languages.add("Rust")

    package_managers: set[str] = set()
    if (root / "package-lock.json").is_file():
        package_managers.add("npm")
    if (root / "pnpm-lock.yaml").is_file():
        package_managers.add("pnpm")
    if (root / "yarn.lock").is_file():
        package_managers.add("yarn")
    if (root / "uv.lock").is_file():
        package_managers.add("uv")
    if "Python" in languages:
        package_managers.add("pip")
    if (root / "pom.xml").is_file():
        package_managers.add("maven")
    if (root / "build.gradle").is_file() or (root / "build.gradle.kts").is_file():
        package_managers.add("gradle")
    if (root / "go.mod").is_file():
        package_managers.add("go")
    if (root / "Cargo.toml").is_file():
        package_managers.add("cargo")

    frameworks: set[str] = set()
    for needle, name in (
        ("next", "Next.js"),
        ("vite", "Vite"),
        ("react", "React"),
        ("vue", "Vue"),
        ("angular", "Angular"),
        ("express", "Express"),
        ("fastapi", "FastAPI"),
        ("django", "Django"),
        ("flask", "Flask"),
        ("spring", "Spring"),
    ):
        if needle in all_text or any(path.name.startswith(needle) for path in files):
            frameworks.add(name)

    test_frameworks: set[str] = set()
    for needle, name in (
        ("pytest", "pytest"),
        ("jest", "Jest"),
        ("vitest", "Vitest"),
        ("mocha", "Mocha"),
        ("junit", "JUnit"),
        ("cargo test", "cargo test"),
        ("go test", "go test"),
    ):
        if needle in all_text or any(path.name.startswith(needle) for path in files):
            test_frameworks.add(name)
    if any(path.name.startswith("test_") or path.name.endswith("_test.py") for path in files):
        test_frameworks.add("pytest")

    linters = _tool_names(
        all_text,
        files,
        (
            ("ruff", "ruff"),
            ("eslint", "ESLint"),
            ("pylint", "pylint"),
            ("flake8", "flake8"),
            ("golangci", "golangci-lint"),
            ("clippy", "clippy"),
        ),
    )
    formatters = _tool_names(
        all_text,
        files,
        (
            ("black", "black"),
            ("isort", "isort"),
            ("prettier", "Prettier"),
            ("ruff format", "ruff format"),
            ("gofmt", "gofmt"),
            ("rustfmt", "rustfmt"),
        ),
    )
    typecheckers = _tool_names(
        all_text,
        files,
        (
            ("mypy", "mypy"),
            ("pyright", "pyright"),
            ("typescript", "TypeScript"),
            ("tsc", "tsc"),
            ("maven", "maven"),
        ),
    )
    build_tools = _tool_names(
        all_text,
        files,
        (
            ("vite", "Vite"),
            ("next", "Next.js"),
            ("webpack", "webpack"),
            ("maven", "Maven"),
            ("gradle", "Gradle"),
            ("cargo", "Cargo"),
            ("go build", "go"),
        ),
    )

    ordered_languages = [
        name
        for name in ("Python", "TypeScript", "JavaScript", "Java", "Go", "Rust")
        if name in languages
    ]
    primary = ordered_languages[0] if ordered_languages else "Unknown"
    return RepositoryTechnologyProfile(
        repository_id=identifier,
        primary_language=primary,
        languages=ordered_languages or ["Unknown"],
        package_managers=sorted(package_managers),
        frameworks=sorted(frameworks),
        test_frameworks=sorted(test_frameworks),
        linters=sorted(linters),
        formatters=sorted(formatters),
        typecheckers=sorted(typecheckers),
        build_tools=sorted(build_tools),
        detected_files=[path.as_posix() for path in files],
        confidence=1.0 if ordered_languages else 0.2,
    )


async def calculate_repository_revision(repository_root: PathLike) -> RepositoryRevision:
    """Return a revision that changes for HEAD, index, tracked, and untracked changes."""
    root = resolve_workspace_root(repository_root)
    return await asyncio.to_thread(_calculate_repository_revision, root)


def calculate_repository_revision_sync(repository_root: PathLike) -> RepositoryRevision:
    """Synchronous equivalent used by non-live validation helpers and unit tests."""
    return _calculate_repository_revision(resolve_workspace_root(repository_root))


def _calculate_repository_revision(root: Path) -> RepositoryRevision:
    if (root / ".git").exists():
        head = _git(root, "rev-parse", "HEAD")
        index = _git(root, "diff", "--cached", "--binary", "--no-ext-diff")
        worktree = _git(root, "diff", "--binary", "--no-ext-diff")
        untracked_names = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
        if (
            head is not None
            and index is not None
            and worktree is not None
            and untracked_names is not None
        ):
            untracked = _untracked_manifest(root, untracked_names)
            return _revision(
                head.strip() or None,
                index.encode(),
                worktree.encode(),
                untracked,
            )
    # A workspace without a Git directory is still safe: fingerprint all regular files.
    return _revision(None, b"", _workspace_manifest(root), b"")


def _revision(
    head: str | None, index: bytes, worktree: bytes, untracked: bytes
) -> RepositoryRevision:
    index_hash = _sha(index)
    worktree_hash = _sha(worktree)
    untracked_hash = _sha(untracked)
    combined = _sha(
        "|".join((head or "no-head", index_hash, worktree_hash, untracked_hash)).encode()
    )
    return RepositoryRevision(
        head_sha=head,
        index_fingerprint=index_hash,
        working_tree_fingerprint=worktree_hash,
        untracked_fingerprint=untracked_hash,
        combined_fingerprint=combined,
    )


def _git(root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=repository_subprocess_environment(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _untracked_manifest(root: Path, names: str) -> bytes:
    entries: list[bytes] = []
    for raw_name in names.split("\0"):
        if not raw_name:
            continue
        path = root / raw_name
        if path.is_file():
            entries.append(raw_name.encode("utf-8", errors="surrogateescape"))
            entries.append(b"\0")
            entries.append(_sha(path.read_bytes()).encode())
            entries.append(b"\0")
    return b"".join(entries)


def _workspace_manifest(root: Path) -> bytes:
    entries: list[bytes] = []
    for path in _source_files(root):
        entries.extend(
            (path.as_posix().encode(), b"\0", _sha((root / path).read_bytes()).encode(), b"\0")
        )
    return b"".join(entries)


def _detected_files(root: Path) -> list[Path]:
    patterns = (
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
        "setup.cfg",
        "tox.ini",
        "pytest.ini",
        "uv.lock",
        "poetry.lock",
        "Pipfile.lock",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "tsconfig.json",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "go.mod",
        "Cargo.toml",
        "Makefile",
        "eslint.config.*",
        ".eslintrc*",
        "prettier.config.*",
        "vite.config.*",
        "next.config.*",
        "jest.config.*",
        "vitest.config.*",
    )
    ignored = {".git", ".venv", "node_modules", "__pycache__"}
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(
            path.relative_to(root)
            for path in root.rglob(pattern)
            if path.is_file() and not any(part in ignored for part in path.relative_to(root).parts)
        )
    return sorted(paths)


def _source_files(root: Path) -> list[Path]:
    ignored = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache"}
    return sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and not any(part in ignored for part in path.relative_to(root).parts)
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _tool_names(text: str, files: list[Path], candidates: tuple[tuple[str, str], ...]) -> set[str]:
    names: set[str] = set()
    lowered_files = " ".join(path.name.lower() for path in files)
    for needle, name in candidates:
        if needle in text or needle in lowered_files:
            names.add(name)
    return names


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "RepositoryRevision",
    "RepositoryTechnologyProfile",
    "calculate_repository_revision",
    "calculate_repository_revision_sync",
    "inspect_repository_technology",
]
