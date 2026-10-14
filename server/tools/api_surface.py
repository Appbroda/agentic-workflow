"""Extract the method+path routes a repository's own code actually registers.

Deliberately a heuristic source scanner, following the same narrow-by-design precedent already
used elsewhere in this codebase (`tools/reachability.py`, `tools/channel_packages.py`): it reads
real files with targeted regular expressions rather than a full per-language AST, and documents
exactly what it does and does not resolve. Consumed by `tools/repository_reconnaissance.py` to
ground a plan against what a repository already has wired, before any code is written.

One engine, one row per framework. Unlike `tools/channel_packages.py`'s ecosystem tables -- where
the *algorithm* (does an import name a package, does a bound name reach a configuration call) is
identical across ecosystems and only the vocabulary differs -- route-registration syntax differs
in *shape*, not just vocabulary: Express needs a two-step path-then-method scan across lines,
while FastAPI and Flask capture a method and a path together on one decorator line. Each
framework below therefore owns its own small matching function; what is genuinely shared --
combining a bare registered path with every observed mount prefix -- is factored into one helper
every framework's function calls.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_METHOD_NAMES = ("get", "post", "put", "patch", "delete", "head", "options")

# An Express-style chained route registration: `.route('/x')` followed, on the same or a later
# line, by `.get(`/`.post(`/etc. Narrow to this one framework shape by design -- a repository
# using a different router style simply yields no candidates here, which fails open (no
# finding) rather than closed (a false block), consistent with every other heuristic tool in
# this package.
_ROUTE_PATH_PATTERN = re.compile(r"""\.route\(\s*['"](/[A-Za-z0-9_\-/:{}.]*)['"]\s*\)""")
_METHOD_CALL_PATTERN = re.compile(rf"""\.({"|".join(_METHOD_NAMES)})\(""")

# A router mount: `router.use('/service-requests', serviceRequestRoutes)`,
# `app.use('/api', router)`. Collected platform-wide (not file-scoped) because the mounting
# file and the route-defining file are conventionally different (e.g. `index.route.js` mounts;
# `serviceRequest.route.js` defines).
_MOUNT_PATTERN = re.compile(r"""\.use\(\s*['"](/[A-Za-z0-9_\-/:{}.]*)['"]""")

# The same mount call, but also naming what it mounts -- `.use('/prefix', someRouter)` -- paired
# with a local import of that same name, so a route combines only with the prefix that actually
# carries it rather than with every prefix observed anywhere in the repository. A repository
# with many route modules each mounted under its own prefix turns "every bare route combined
# with every prefix" into a combinatorial explosion that buries the one correct combination
# past any cap reconnaissance can afford (AB-204: 36 wrong combinations of one bare route ahead
# of its one real mounted form, in a repository with ~30 distinct prefixes). A file this cannot
# resolve -- a dynamic import, a destructured require, a name this regex does not recognize --
# simply contributes nothing here, and `extract_registered_routes` falls back to the
# platform-wide set for it, same as before this existed.
_IMPORT_PATTERN = re.compile(r"""import\s+(\w+)\s+from\s+['"](\.[^'"]*)['"]""")
_MOUNT_WITH_TARGET_PATTERN = re.compile(
    r"""\.use\(\s*['"](/[A-Za-z0-9_\-/:{}.]*)['"]\s*,\s*(\w+)"""
)

# A FastAPI (or any `@<name>.<method>(...)` decorator-shaped) route registration: method and
# path are captured together on one line, unlike Express's two-step chain.
_FASTAPI_ROUTE_PATTERN = re.compile(
    rf"""@\w+\.({"|".join(_METHOD_NAMES)})\(\s*['"](/[A-Za-z0-9_\-/:{{}}.]*)['"]"""
)
# `app.include_router(router, prefix="/api")`. Bounded (not `re.DOTALL` unbounded) so a call
# whose `prefix=` kwarg is absent cannot make the scan search arbitrarily far into the file --
# the same fail-open-on-absence discipline as the rest of this module.
_FASTAPI_MOUNT_PATTERN = re.compile(
    r"""include_router\(.{0,200}?prefix\s*=\s*['"](/[A-Za-z0-9_\-/:{}.]*)['"]""", re.DOTALL
)

# Flask's classic form, `@app.route('/x', methods=['POST'])`, where the path and its optional
# method list are both inside one call; absent `methods=`, Flask itself defaults to GET, which
# this mirrors. Modern Flask also accepts the FastAPI-shaped `@app.get('/x')` shorthand, so the
# Flask extractor below also tries `_FASTAPI_ROUTE_PATTERN`.
_FLASK_ROUTE_PATTERN = re.compile(
    r"""@\w+\.route\(\s*['"](/[A-Za-z0-9_\-/:{}.]*)['"](.{0,200}?)\)""", re.DOTALL
)
_FLASK_METHODS_KWARG_PATTERN = re.compile(r"""methods\s*=\s*\[([^\]]*)\]""")
# `app.register_blueprint(bp, url_prefix="/api")`, the Flask analogue of `include_router`.
_FLASK_MOUNT_PATTERN = re.compile(
    r"""register_blueprint\(.{0,200}?url_prefix\s*=\s*['"](/[A-Za-z0-9_\-/:{}.]*)['"]""",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class RegisteredRoute:
    """One method+path a repository's own router registration actually exposes."""

    method: str
    path: str
    source_path: str


@dataclass(frozen=True, slots=True)
class RouteFramework:
    """One framework's own route-registration shape.

    `extract` takes one file's own `(path, source)` and returns that file's own routes --
    already combined with any mount prefix the SAME file declares. A cross-file mount (the
    mounting file and the route-defining file are conventionally different, as in Express) is
    handled by `extract_registered_routes` itself, platform-wide, after every file's bare routes
    and mount prefixes are collected.
    """

    name: str
    source_suffixes: frozenset[str]
    bare_routes: Callable[[str, str], list[RegisteredRoute]]
    mount_prefixes: Callable[[str], frozenset[str]]
    # Optional: this file's own mount calls, resolved to the specific route module (relative
    # path, suffix-less) each one mounts -- the precise alternative to `mount_prefixes`'s
    # platform-wide set, used when a route's own source file is a key in the combined map.
    # `None` for a framework where this has not been written yet; falls back to the flat set.
    mount_map: Callable[[str, str], dict[str, frozenset[str]]] | None = None


def _express_bare_routes(relative_path: str, text: str) -> list[RegisteredRoute]:
    """Resolve a chained `.route('/x').get(...)` registration to its local path."""
    routes: list[RegisteredRoute] = []
    current_route_path = ""
    for line in text.splitlines():
        route_match = _ROUTE_PATH_PATTERN.search(line)
        if route_match:
            current_route_path = route_match.group(1)
        if not current_route_path:
            continue
        for method_match in _METHOD_CALL_PATTERN.finditer(line):
            routes.append(
                RegisteredRoute(
                    method=method_match.group(1).upper(),
                    path=current_route_path,
                    source_path=relative_path,
                )
            )
    return routes


def _express_mount_prefixes(text: str) -> frozenset[str]:
    return frozenset(match.group(1) for match in _MOUNT_PATTERN.finditer(text))


def _express_mount_map(relative_path: str, text: str) -> dict[str, frozenset[str]]:
    """Map each route module this file imports and mounts to the prefix it mounts it under.

    `router.use('/service-requests', serviceRequestRoutes)` only names what it mounts, not
    where that module lives -- resolved here from this same file's own
    `import serviceRequestRoutes from './serviceRequest.route'`, a local import a mounting file
    conventionally carries for exactly the module it mounts (`repository_reconnaissance.py`'s
    own `_registered_routes` docstring: "the mounting file and the route-defining file are
    conventionally different").
    """
    imports = {match.group(1): match.group(2) for match in _IMPORT_PATTERN.finditer(text)}
    if not imports:
        return {}
    base_dir = str(PurePosixPath(relative_path).parent)
    mounts: dict[str, set[str]] = {}
    for match in _MOUNT_WITH_TARGET_PATTERN.finditer(text):
        prefix, name = match.group(1), match.group(2)
        module_spec = imports.get(name)
        if module_spec is None:
            continue
        resolved = posixpath.normpath(posixpath.join(base_dir, module_spec))
        mounts.setdefault(resolved, set()).add(prefix)
    return {module_path: frozenset(prefixes) for module_path, prefixes in mounts.items()}


def _fastapi_bare_routes(relative_path: str, text: str) -> list[RegisteredRoute]:
    """Resolve a `@router.get('/x')`-shaped decorator to its method and path together."""
    return [
        RegisteredRoute(
            method=match.group(1).upper(), path=match.group(2), source_path=relative_path
        )
        for match in _FASTAPI_ROUTE_PATTERN.finditer(text)
    ]


def _fastapi_mount_prefixes(text: str) -> frozenset[str]:
    return frozenset(match.group(1) for match in _FASTAPI_MOUNT_PATTERN.finditer(text))


def _flask_bare_routes(relative_path: str, text: str) -> list[RegisteredRoute]:
    """Resolve `@app.route('/x', methods=[...])`.

    The modern `@app.get('/x')` shorthand Flask also accepts is the same shape as FastAPI's own
    decorator, and the dispatcher in `extract_registered_routes` already tries
    `_FASTAPI_FRAMEWORK` against every `.py` file separately -- matching it again here would
    double-count every route written in that shape.
    """
    routes: list[RegisteredRoute] = []
    for match in _FLASK_ROUTE_PATTERN.finditer(text):
        path, rest = match.group(1), match.group(2)
        methods_match = _FLASK_METHODS_KWARG_PATTERN.search(rest)
        methods = (
            [
                name.strip(" '\"").upper()
                for name in methods_match.group(1).split(",")
                if name.strip()
            ]
            if methods_match
            else ["GET"]
        )
        routes.extend(
            RegisteredRoute(method=method, path=path, source_path=relative_path)
            for method in methods
        )
    return routes


def _flask_mount_prefixes(text: str) -> frozenset[str]:
    return frozenset(match.group(1) for match in _FLASK_MOUNT_PATTERN.finditer(text))


_EXPRESS_FRAMEWORK = RouteFramework(
    name="express",
    source_suffixes=frozenset({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}),
    bare_routes=_express_bare_routes,
    mount_prefixes=_express_mount_prefixes,
    mount_map=_express_mount_map,
)
_FASTAPI_FRAMEWORK = RouteFramework(
    name="fastapi",
    source_suffixes=frozenset({".py"}),
    bare_routes=_fastapi_bare_routes,
    mount_prefixes=_fastapi_mount_prefixes,
)
_FLASK_FRAMEWORK = RouteFramework(
    name="flask",
    source_suffixes=frozenset({".py"}),
    bare_routes=_flask_bare_routes,
    mount_prefixes=_flask_mount_prefixes,
)
# A `.py` file is tried against both Python frameworks; a file written for one simply yields
# nothing from the other's patterns, the same fail-open discipline as an unrecognized framework
# entirely. Add a new framework here, with its own `extract`-shaped functions, rather than
# bending an existing one to fit a second syntax.
_ROUTE_FRAMEWORKS: tuple[RouteFramework, ...] = (
    _EXPRESS_FRAMEWORK,
    _FASTAPI_FRAMEWORK,
    _FLASK_FRAMEWORK,
)


def extract_registered_routes(
    workspace: Path, changed_files: Sequence[str]
) -> list[RegisteredRoute]:
    """Return every method+path a providing repository actually registers.

    Dispatches each file to every framework whose `source_suffixes` match its extension
    (mirroring `tools/channel_packages.py`'s own suffix-based ecosystem dispatch), resolves each
    framework's own bare routes, mount prefixes and (where written) precise mount map, then
    combines each bare route with the prefix(es) that actually mount its own source file when
    that is known, falling back to every prefix observed anywhere in the changed files when it
    is not -- conservative rather than precise only where precision could not be resolved.
    """
    bare_routes: list[RegisteredRoute] = []
    mount_prefixes: set[str] = set()
    mount_map: dict[str, set[str]] = {}
    for relative_path in changed_files:
        text = _read_text(workspace / relative_path)
        if text is None:
            continue
        suffix = PurePosixPath(relative_path).suffix
        for framework in _ROUTE_FRAMEWORKS:
            if suffix not in framework.source_suffixes:
                continue
            bare_routes.extend(framework.bare_routes(relative_path, text))
            mount_prefixes.update(framework.mount_prefixes(text))
            if framework.mount_map is not None:
                for module_path, mounted_under in framework.mount_map(relative_path, text).items():
                    mount_map.setdefault(module_path, set()).update(mounted_under)
    if not mount_prefixes:
        return bare_routes
    # Each route's own combinations stay grouped with it, mounted forms first -- not two
    # giant blocks (every bare path, then every mounted one). AB-204 is why: a repository with
    # more bare routes than `_MAX_REGISTERED_ROUTES` (`repository_reconnaissance.py`) truncates
    # the list before the second block is ever reached, so every mounted path -- the complete,
    # externally reachable one a contract is supposed to copy -- was silently dropped, and only
    # the bare fragment survived to ground the plan. Interleaved and mounted-first, a cap still
    # loses routes scanned late, but never loses a route's complete form while keeping its bare
    # one.
    combined: list[RegisteredRoute] = []
    for route in bare_routes:
        source_without_suffix = posixpath.splitext(route.source_path)[0]
        # The precise set when this route's own file was resolved to a mount; the flat,
        # platform-wide set otherwise. A route combined with its one real prefix, instead of
        # with every prefix anything in the repository happens to mount something under, is
        # what keeps a combinatorial repository from burying its one correct combination.
        prefixes: AbstractSet[str] = mount_map.get(source_without_suffix) or mount_prefixes
        combined.extend(
            RegisteredRoute(
                method=route.method,
                path=f"{prefix.rstrip('/')}{route.path}",
                source_path=route.source_path,
            )
            for prefix in prefixes
        )
        combined.append(route)
    return combined


def _read_text(path: Path) -> str | None:
    """Return a changed file's text, or `None` for anything not readable as source."""
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


__all__ = [
    "RegisteredRoute",
    "RouteFramework",
    "extract_registered_routes",
]
