"""Run each repository's own declared package manager instead of the image's one copy.

`packageManager` in package.json is a Corepack provisioning directive -- *use this exact
version* -- and Corepack ships with Node. A platform that installs one pnpm and one Yarn
globally and then asks every repository to match them can serve exactly the repositories
that happen to agree with it. That is what AB-Feature-222 hit: `npm@9.8.1` declared,
npm 10.9.8 in the image, and a terminal "repository defect" for a repository that installs
perfectly.

So the version is not compared here, it is *honoured*. A declaring project's commands are
prefixed with `corepack <manager>@<version>`, which provisions that version on first use and
caches it. Two repositories pinning different versions both work, and neither needs the
platform rebuilt.

Invoked explicitly rather than through `corepack enable`'s global shims, deliberately. The
shims replace `npm` process-wide, and a project that declares nothing then resolves to
whatever Corepack considers current -- measured as npm 12.0.2 on the worker whose image
provides 10.9.8. Explicit invocation leaves a non-declaring repository on the image's
toolchain, unchanged, and puts the version that ran in the command the journal records.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

# Corepack manages these three and nothing else.
COREPACK_MANAGERS = frozenset({"npm", "pnpm", "yarn"})

_DECLARATION = re.compile(r"^(npm|pnpm|yarn)@(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)$", re.IGNORECASE)


def _package_manifest(root: Path) -> dict[str, object]:
    """Read one directory's package.json, treating anything unreadable as absent."""
    try:
        value = json.loads((root / "package.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def declared_package_manager(project_root: Path, repository_root: Path) -> tuple[str, str] | None:
    """Read the nearest exact `packageManager` declaration, or None when there is not one.

    Nearest rather than per-directory because a monorepo declares once at its root and every
    workspace package inherits it, which is the same resolution Corepack itself performs.

    A declaration this cannot parse exactly -- a range, a URL, a hash-pinned string -- returns
    None so the caller keeps the image's manager. Preflight reports the unparseable
    declaration separately; guessing a version here would run something nobody asked for.
    """
    if repository_root not in project_root.parents and project_root != repository_root:
        return None
    current = project_root
    while True:
        declared = _package_manifest(current).get("packageManager")
        if isinstance(declared, str):
            match = _DECLARATION.fullmatch(declared.strip())
            if match is not None:
                return match.group(1).lower(), match.group(2)
        if current == repository_root:
            return None
        current = current.parent


def corepack_available() -> bool:
    """Whether this worker has Corepack at all.

    Separated so an image without it keeps working: every caller falls back to the image's
    own manager, and preflight then judges lockfile compatibility rather than refusing.
    """
    return shutil.which("corepack") is not None


def package_manager_argv(
    manager: str, project_root: Path, repository_root: Path
) -> tuple[list[str], str | None]:
    """Return the argv prefix that invokes this project's package manager, and the version.

    The version is None when the image's own manager is being used, which is the answer for a
    project that declares nothing, declares something unparseable, or declares a manager other
    than the one its lockfile selects. That last case is a real conflict and preflight reports
    it as `PACKAGE_MANAGER_LOCKFILE_MISMATCH`; running the declared manager against a lockfile
    written by a different one would turn a clear diagnosis into a confusing install failure.
    """
    if manager not in COREPACK_MANAGERS or not corepack_available():
        return [manager], None
    declaration = declared_package_manager(project_root, repository_root)
    if declaration is None:
        return [manager], None
    declared_manager, version = declaration
    if declared_manager != manager:
        return [manager], None
    return ["corepack", f"{manager}@{version}"], version


__all__ = [
    "COREPACK_MANAGERS",
    "corepack_available",
    "declared_package_manager",
    "package_manager_argv",
]
