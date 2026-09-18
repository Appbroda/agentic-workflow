"""Bounded, read-only checkout evidence gathered before a feature is planned.

Planning currently happens against a PRD and a repository URL, so every statement the plan
makes about a checkout is a guess. The four premises that cost the -066..-075 live runs were
all of this kind: an endpoint said to hold state that is a one-line literal, per-route auth in
a repository that applies auth globally, a console section that does not exist, and a shared
time formatter that was never shared. Each became a requirement no attempt could satisfy.

This module answers only what can be established by reading files. It makes no judgement --
that is the reconnaissance agent's job -- and it never runs repository code.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path, PurePosixPath

from pydantic import Field

from state.models import StateModel
from tools.file_tools import (
    PathLike,
    carries_key_material,
    is_model_safe_context_path,
    resolve_workspace_root,
)
from tools.repository_layout import inspect_repository_layout
from tools.technology_detection import (
    RepositoryTechnologyProfile,
    calculate_repository_revision_sync,
    inspect_repository_technology,
)

_SOURCE_SUFFIXES = frozenset(
    {".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".go", ".rb", ".java", ".rs"}
)
# A registry is the file a new route or component must be added to in order to be reachable
# at all, and it is the fact a plan most often gets wrong. It is found structurally, by how
# many of the repository's own modules a file pulls in: a file that references many siblings
# is assembling them. Naming candidates by filename instead encodes one project's taste --
# the pilot backend's registry is `server/config/express.js`, which no stem list would guess,
# while plenty of repositories have an `index.js` that assembles nothing.
_REFERENCE_PATTERNS = (
    re.compile(r"""(?:from|require)\s*\(?\s*['"]([^'"]+)['"]"""),
    re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", re.MULTILINE),
)
# Below this a file is an ordinary consumer of a couple of helpers, not an assembly point.
_MIN_WIRING_REFERENCES = 2
# Path words that say a file mounts modules rather than merely importing some. Reference
# counting alone cannot tell an assembly point from a large consumer: this repository's
# `server/auth/models.js` and `services/IDRefresh/ironsource.service.js` both pull in several
# siblings and register nothing, and they outranked the route index that does.
# Tokens that identify a registry only as a whole filename. `App.js` is a React root;
# `AddAppModal.js` is a modal. Matching "app" as one camel-case fragment made every file
# whose name mentions an app a registry, which in an ad-management console is very nearly
# all of them: AB-Feature-112's console was told twelve times to mount bulk deletion in
# `src/components/apps/AddAppModal.js`, and burned every review cycle on it.
_REGISTRAR_STEM_TOKENS = frozenset({"app", "index", "main", "root"})
# Tokens that identify a registry wherever they appear in the name, because they describe
# the job rather than the subject: a sidebar or a router mounts things whatever it is called.
_REGISTRAR_FRAGMENT_TOKENS = frozenset(
    {
        "menu",
        "nav",
        "navigation",
        "registry",
        "route",
        "router",
        "routes",
        "sidebar",
    }
)
# A directory says where the repository puts this kind of file, so either sort counts there.
_REGISTRAR_TOKENS = _REGISTRAR_STEM_TOKENS | _REGISTRAR_FRAGMENT_TOKENS
_SYMBOL_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:export\s+)?(?:default\s+)?"
    r"(?:async\s+)?(?:class|def|function|const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)"
)
_MAX_INVENTORY_FILES = 600
_MAX_READ_FILES = 400
_MAX_WIRING_FILES = 12
_MAX_SYMBOL_FILES = 60
_MAX_SYMBOLS_PER_FILE = 24
_MAX_FILE_BYTES = 200_000


class ModuleSymbols(StateModel):
    """The top-level names one source file defines, as a shape a plan can be checked against."""

    path: str = Field(min_length=1)
    symbols: list[str] = Field(default_factory=list)


class RepositoryReconnaissanceEvidence(StateModel):
    """Read-only facts about one checkout, gathered before any plan exists."""

    repository_id: str = Field(min_length=1)
    repository_revision: str = Field(min_length=1)
    head_sha: str | None = None
    technology: RepositoryTechnologyProfile
    source_directories: list[str] = Field(default_factory=list)
    test_directories: list[str] = Field(default_factory=list)
    declared_scripts: list[str] = Field(default_factory=list)
    wiring_files: list[ModuleSymbols] = Field(default_factory=list)
    module_symbols: list[ModuleSymbols] = Field(default_factory=list)
    file_inventory: list[str] = Field(default_factory=list)
    omitted_inventory_file_count: int = 0


def inspect_repository_for_planning(
    repository_root: PathLike, *, repository_id: str
) -> RepositoryReconnaissanceEvidence:
    """Collect the checkout facts a planner needs, without executing anything in it."""
    root = resolve_workspace_root(repository_root)
    layout = inspect_repository_layout(root)
    revision = calculate_repository_revision_sync(root)
    sources = _source_files(root)
    inventory = [path.relative_to(root).as_posix() for path in sources]
    contents = {path: _read_source(path) for path in sources[:_MAX_READ_FILES]}
    own_module_names = {path.stem.lower() for path in sources}
    # Registrars first, then by how much each assembles. The list is truncated before anyone
    # reads it, so ordering decides what a plan and a coding attempt ever see: ranked by count
    # alone, the pilot backend offered `server/auth/models.js` and an ID-refresh service ahead
    # of anything that mounts a route, and the attempt that had to register one was shown
    # neither `server/routes/index.route.js` nor a file resembling it.
    wiring = sorted(
        (
            (path, count)
            for path, content in contents.items()
            if (count := _own_reference_count(content, own_module_names, path))
            >= _MIN_WIRING_REFERENCES
        ),
        key=lambda item: (
            not registers_modules(item[0].relative_to(root).as_posix()),
            -item[1],
            item[0].relative_to(root).as_posix(),
        ),
    )
    return RepositoryReconnaissanceEvidence(
        repository_id=repository_id,
        repository_revision=revision.combined_fingerprint,
        head_sha=revision.head_sha,
        technology=inspect_repository_technology(root, repository_id=repository_id),
        source_directories=list(layout.source_directories),
        test_directories=list(layout.test_directories),
        declared_scripts=_declared_scripts(root),
        wiring_files=[
            _module_symbols(root, path, contents.get(path))
            for path, _ in wiring[:_MAX_WIRING_FILES]
        ],
        module_symbols=[
            _module_symbols(root, path, contents.get(path)) for path in sources[:_MAX_SYMBOL_FILES]
        ],
        file_inventory=inventory[:_MAX_INVENTORY_FILES],
        omitted_inventory_file_count=max(0, len(inventory) - _MAX_INVENTORY_FILES),
    )


def _own_reference_count(content: str | None, module_names: set[str], path: Path) -> int:
    """Count how many of this repository's own modules a file pulls in.

    Only the repository's own modules count: a file importing a dozen third-party packages is
    a consumer of libraries, while one importing a dozen siblings is assembling them.
    """
    if content is None:
        return 0
    referenced: set[str] = set()
    for pattern in _REFERENCE_PATTERNS:
        for match in pattern.findall(content):
            stem = _referenced_module_name(str(match))
            if stem and stem in module_names and stem != path.stem.lower():
                referenced.add(stem)
    return len(referenced)


def _referenced_module_name(reference: str) -> str:
    """Return the module stem a reference names, for path and dotted-import styles alike.

    Replacing every dot with a separator resolves a Python `package.module` import, and
    destroys any path whose file name contains a dot: `./app/app.route` became `//app/app/route`
    and resolved to `route`, `./utils/helpers.js` resolved to `js`. Both matched nothing, so in
    a repository naming files `app.route.js` and `app.service.js` -- the dominant convention in
    the pilot backend -- practically every relative import went uncounted. Its route index
    imports two dozen sibling routes and scored zero, which is why reconnaissance offered
    consumers as that repository's registries.
    """
    text = reference.strip()
    # A dotted module path has no separator; anything with one is a filesystem path, whose
    # own dots belong to the file name.
    name = PurePosixPath(text).name if "/" in text else text.rsplit(".", maxsplit=1)[-1]
    suffix = PurePosixPath(name).suffix
    if suffix and suffix.lower() in _SOURCE_SUFFIXES:
        name = name[: -len(suffix)]
    return name.lower()


def registers_modules(path: str) -> bool:
    """Judge whether a file mounts modules, rather than merely mentioning one.

    Derived from the path the repository already chose, so it stays a question about the
    checkout rather than an assumption about a framework.
    """
    posix = PurePosixPath(path)
    if {part.lower() for part in posix.parts[:-1]} & _REGISTRAR_TOKENS:
        return True
    if posix.stem.lower() in _REGISTRAR_STEM_TOKENS:
        return True
    fragments = {token.lower() for token in re.findall(r"[A-Z]?[a-z]+", posix.stem)}
    return bool(fragments & _REGISTRAR_FRAGMENT_TOKENS)


def _read_source(path: Path) -> str | None:
    """Return one bounded source file's text, or None when it cannot be safely read as text.

    Unreadable and key-material-carrying files answer the same way, because both mean this
    file cannot contribute to something a model will be shown. Reconnaissance only ever
    emits symbol names, but a name lifted out of a key file is still a name from a key file.
    """
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return None
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return None if carries_key_material(source) else source


def _source_files(root: Path) -> list[Path]:
    """Return every readable source file this checkout may safely describe itself with."""
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in _SOURCE_SUFFIXES
        and is_model_safe_context_path(PurePosixPath(path.relative_to(root).as_posix()))
    ]
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    return files


def _declared_scripts(root: Path) -> list[str]:
    """Report the task names this repository declares, whatever ecosystem declares them.

    Only names, never their command bodies: a script line commonly carries a host, a port, or
    a token, and the name is what a plan needs in order to avoid inventing a test command the
    repository does not have.
    """
    names: set[str] = set()
    package = root / "package.json"
    if package.is_file():
        try:
            payload = json.loads(package.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        scripts = payload.get("scripts") if isinstance(payload, dict) else None
        if isinstance(scripts, dict):
            names.update(str(key) for key in scripts)
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
            data = {}
        scripts = data.get("project", {}).get("scripts") if isinstance(data, dict) else None
        if isinstance(scripts, dict):
            names.update(str(key) for key in scripts)
    return sorted(names)


def _module_symbols(root: Path, path: Path, content: str | None = None) -> ModuleSymbols:
    """Extract the top-level names one file defines, or none when it cannot be read."""
    relative = path.relative_to(root).as_posix()
    source = content if content is not None else _read_source(path)
    if source is None:
        return ModuleSymbols(path=relative, symbols=[])
    found = list(dict.fromkeys(_SYMBOL_PATTERN.findall(source)))
    return ModuleSymbols(path=relative, symbols=found[:_MAX_SYMBOLS_PER_FILE])


__all__ = [
    "ModuleSymbols",
    "RepositoryReconnaissanceEvidence",
    "inspect_repository_for_planning",
]
