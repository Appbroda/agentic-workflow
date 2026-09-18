"""Checkout-derived repository layout evidence.

Feature plans are produced before a repository is cloned, so they must never rely on
``src``/``routes`` or a framework-specific directory convention.  This module resolves
only an unambiguous stale path against the files that actually exist in one checkout.
It deliberately has no language, framework, or repository-name allowlist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from artifacts.schemas import ImplementationExpectation, RepositoryWorkstreamPlan
from tools.file_tools import PathLike, resolve_workspace_root

_IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
}
_NON_SOURCE_SUFFIXES = {
    ".lock",
    ".map",
    ".md",
    ".rst",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
}
_TEST_DIRECTORY_NAMES = {"test", "tests", "spec", "specs", "__tests__"}


@dataclass(frozen=True, slots=True)
class RepositoryLayoutEvidence:
    """Safe layout facts and any exact expectation rebindings made from them."""

    source_directories: tuple[str, ...]
    test_directories: tuple[str, ...]
    remapped_areas: tuple[tuple[str, str], ...]
    unresolved_areas: tuple[str, ...]
    ambiguous_areas: tuple[str, ...]


def inspect_repository_layout(repository_root: PathLike) -> RepositoryLayoutEvidence:
    """Return layout evidence without imposing a source-root naming convention."""
    root = resolve_workspace_root(repository_root)
    source_directories: set[str] = set()
    test_directories: set[str] = set()
    for path in _workspace_files(root):
        relative = path.relative_to(root)
        if _is_test_path(relative):
            if relative.parent != Path("."):
                test_directories.add(relative.parent.as_posix())
            continue
        if _is_probable_source_file(relative) and relative.parent != Path("."):
            source_directories.add(relative.parent.as_posix())
    return RepositoryLayoutEvidence(
        source_directories=tuple(sorted(source_directories)),
        test_directories=tuple(sorted(test_directories)),
        remapped_areas=(),
        unresolved_areas=(),
        ambiguous_areas=(),
    )


def reconcile_workstream_layout(
    workstream: RepositoryWorkstreamPlan, repository_root: PathLike
) -> tuple[RepositoryWorkstreamPlan, RepositoryLayoutEvidence]:
    """Rebind only missing plan areas that have one exact checkout-derived match.

    A path already present in the checkout is preserved. For a missing path, matching
    tries its complete suffix and then progressively shorter suffixes. A replacement is
    permitted only when exactly one non-test checkout path has that suffix. When there
    is no unique replacement, the stale area is recorded as evidence but removed as a
    completion constraint, so a pre-clone model guess cannot reject valid work in an
    unknown repository layout.
    """
    root = resolve_workspace_root(repository_root)
    evidence = inspect_repository_layout(root)
    remapped: list[tuple[str, str]] = []
    unresolved: list[str] = []
    ambiguous: list[str] = []
    expectations: list[ImplementationExpectation] = []
    for expectation in workstream.implementation_expectations:
        updated_areas: list[str] = []
        for area in expectation.expected_source_areas:
            resolved, status = _resolve_area(root, area)
            if status in {"exact", "remapped"}:
                updated_areas.append(resolved)
            if status == "remapped":
                remapped.append((area, resolved))
            elif status == "unresolved":
                unresolved.append(area)
            elif status == "ambiguous":
                ambiguous.append(area)
        expectations.append(
            expectation.model_copy(
                update={"expected_source_areas": list(dict.fromkeys(updated_areas))}
            )
        )
    return (
        workstream.model_copy(update={"implementation_expectations": expectations}),
        RepositoryLayoutEvidence(
            source_directories=evidence.source_directories,
            test_directories=evidence.test_directories,
            remapped_areas=tuple(remapped),
            unresolved_areas=tuple(sorted(set(unresolved))),
            ambiguous_areas=tuple(sorted(set(ambiguous))),
        ),
    )


def _resolve_area(root: Path, area: str) -> tuple[str, str]:
    relative = _safe_relative_path(area)
    if relative is None:
        return area, "unresolved"
    if (root / relative).exists():
        return relative.as_posix(), "exact"
    parts = relative.parts
    for offset in range(len(parts)):
        suffix = parts[offset:]
        candidates = _matching_paths(root, suffix)
        if len(candidates) == 1:
            return candidates[0], "remapped"
        if len(candidates) > 1:
            # A more specific suffix may still be unique.  Do not report ambiguity
            # until every allowed suffix has been examined.
            continue
    has_candidate = any(_matching_paths(root, parts[offset:]) for offset in range(len(parts)))
    return area, "ambiguous" if has_candidate else "unresolved"


def _matching_paths(root: Path, suffix: tuple[str, ...]) -> list[str]:
    if not suffix:
        return []
    candidates: list[str] = []
    for path in root.rglob(suffix[-1]):
        if _ignored(path, root):
            continue
        relative = path.relative_to(root)
        if _is_test_path(relative) or len(relative.parts) < len(suffix):
            continue
        if relative.parts[-len(suffix) :] != suffix:
            continue
        candidates.append(relative.as_posix())
    return sorted(set(candidates))


def _workspace_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [path for path in root.rglob("*") if path.is_file() and not _ignored(path, root)]


def _safe_relative_path(value: str) -> PurePosixPath | None:
    candidate = PurePosixPath(value.strip("/"))
    if not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    return candidate


def _ignored(path: Path, root: Path) -> bool:
    return any(part.lower() in _IGNORED_DIRECTORIES for part in path.relative_to(root).parts)


def _is_test_path(path: Path) -> bool:
    parts = tuple(part.lower() for part in path.parts)
    name = path.name.lower()
    return (
        any(part in _TEST_DIRECTORY_NAMES for part in parts)
        or ".test." in name
        or ".spec." in name
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _is_probable_source_file(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.suffix.lower() not in _NON_SOURCE_SUFFIXES
        and not name.startswith(".")
        and name
        not in {
            "makefile",
            "dockerfile",
            "license",
            "readme",
            "readme.md",
        }
    )


__all__ = ["RepositoryLayoutEvidence", "inspect_repository_layout", "reconcile_workstream_layout"]
