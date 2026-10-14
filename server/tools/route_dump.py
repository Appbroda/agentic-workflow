"""Execution-grounded route evidence: what a framework itself registers, not a regex's guess.

Deliberately narrow and safe by construction: requires only the one route-definition module a
caller names -- never the application's own entry point, which can carry real side effects (a
database connection, an outbound call, a bound port) at module-load time. A route module built
the conventional way (`module.exports = router`) carries none of that; what mounts it under a
prefix lives in a different file and is resolved separately, by reading source text
(`tools/api_surface.py`'s own precise mount-mapping), never by requiring it.

This exists because every bug `tools/api_surface.py`'s static extraction hit this session --
the truncation, the combinatorial explosion that buried the one correct mounted path under
thousands of wrong ones -- was a symptom of the same root choice: re-deriving, by regex, what a
framework already knows with total certainty at runtime. Asking it directly cannot be truncated
and cannot mis-pair a route with the wrong prefix; it answers exactly one question, "what does
this file itself register", with no ambiguity about which of several registration styles a
repository happens to use.

One engine, one row per framework -- the same shape `tools/api_surface.py`'s own
`RouteFramework`/`_ROUTE_FRAMEWORKS` dispatch already uses, and for the same reason: a route
module's *registration syntax* differs by framework, but "resolve an interpreter, run the one
script that knows this framework's router shape, parse its JSON" does not. A caller never names
a framework; it names a file, and the file's own suffix decides which row answers it. A
repository in a framework with no row yet is not a false finding -- it fails open, the same
"narrow by design, documented, extend when needed" discipline as every heuristic here -- and
gets one when a dump script is written for it, the same way `tools/api_surface.py` grew
FastAPI/Flask rows without touching Express's.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from services.cancellation import CancellationToken
from services.process_runner import ProcessRunner
from tools.api_surface import RegisteredRoute

_DUMP_SCRIPT_DIR = Path(__file__).parent / "route_dump"
_DEFAULT_TIMEOUT_SECONDS = 20.0
_MAX_STDERR_TAIL = 500


@dataclass(frozen=True, slots=True)
class RouteDumpResult:
    """One route module's real, execution-grounded routes, or why it could not be dumped."""

    source_path: str
    routes: tuple[RegisteredRoute, ...]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True)
class RouteDumpFramework:
    """One framework's own way to require a route module and report what it registers.

    `interpreter` resolves the executable (`shutil.which`-shaped: a path, or `None` when it is
    not installed) rather than a fixed string, because the one thing that varies host to host
    is whether the interpreter is on `PATH` at all -- the script itself is a fixed file, found
    the same way regardless.
    """

    name: str
    source_suffixes: frozenset[str]
    interpreter: Callable[[], str | None]
    script: Path


_EXPRESS_ROUTE_DUMP = RouteDumpFramework(
    name="express",
    source_suffixes=frozenset({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}),
    interpreter=lambda: shutil.which("node"),
    script=_DUMP_SCRIPT_DIR / "express_route_module.js",
)
# A `.py` route module has no row yet: FastAPI's `APIRouter` and Flask's `Blueprint` need their
# own dump scripts (a bare Blueprint has no routes until registered on *some* Flask app, unlike
# an Express Router or a FastAPI APIRouter, which are both already route-complete on their own)
# -- add them here, each its own row, when one exists, rather than bending Express's script to
# guess at either shape.
_ROUTE_DUMP_FRAMEWORKS: tuple[RouteDumpFramework, ...] = (_EXPRESS_ROUTE_DUMP,)


def _framework_for(relative_path: str) -> RouteDumpFramework | None:
    suffix = PurePosixPath(relative_path).suffix
    for framework in _ROUTE_DUMP_FRAMEWORKS:
        if suffix in framework.source_suffixes:
            return framework
    return None


async def dump_route_module(
    workspace: Path,
    relative_path: str,
    *,
    process_runner: ProcessRunner,
    cancellation_token: CancellationToken,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> RouteDumpResult:
    """Require one route module and report exactly what it registers.

    Dispatches on the file's own suffix (mirroring `tools/api_surface.py`'s own suffix-based
    dispatch) to the framework row that owns it; a suffix no row claims fails open with its own
    error rather than guessing at a shape nothing here was written to read. Every other failure
    -- no interpreter installed, a module this cannot require, a malformed result -- fails open
    the same way, the discipline every heuristic in this package already follows: a
    `RouteDumpResult` with `.error` set and an empty route tuple, never an exception, so a
    caller falls back to static evidence without a try/except of its own.
    """
    framework = _framework_for(relative_path)
    if framework is None:
        suffix = PurePosixPath(relative_path).suffix or "no suffix"
        return RouteDumpResult(
            source_path=relative_path,
            routes=(),
            error=f"no route dump is implemented for this file's framework ({suffix})",
        )
    interpreter = framework.interpreter()
    if interpreter is None:
        return RouteDumpResult(
            source_path=relative_path,
            routes=(),
            error=f"{framework.name}'s interpreter is not available",
        )
    absolute_path = workspace / relative_path
    if not absolute_path.is_file():
        return RouteDumpResult(
            source_path=relative_path, routes=(), error=f"{relative_path} does not exist"
        )
    result = await process_runner.run(
        (interpreter, str(framework.script), str(absolute_path)),
        workspace,
        timeout_seconds,
        cancellation_token,
    )
    if result.timed_out:
        return RouteDumpResult(source_path=relative_path, routes=(), error="route dump timed out")
    if result.cancelled:
        return RouteDumpResult(source_path=relative_path, routes=(), error="route dump cancelled")
    if result.return_code != 0:
        return RouteDumpResult(
            source_path=relative_path, routes=(), error=_error_from_stderr(result.stderr)
        )
    return _parse_routes(relative_path, result.stdout)


def _parse_routes(relative_path: str, stdout: str) -> RouteDumpResult:
    try:
        entries = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return RouteDumpResult(
            source_path=relative_path, routes=(), error="route dump did not print valid JSON"
        )
    if not isinstance(entries, list):
        return RouteDumpResult(
            source_path=relative_path, routes=(), error="route dump did not print a JSON array"
        )
    routes = tuple(
        RegisteredRoute(method=entry["method"], path=entry["path"], source_path=relative_path)
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("method"), str)
        and isinstance(entry.get("path"), str)
    )
    return RouteDumpResult(source_path=relative_path, routes=routes)


def _error_from_stderr(stderr: str) -> str:
    """Prefer the script's own `{"error": "..."}` marker; fall back to the raw tail otherwise."""
    try:
        payload = json.loads(stderr)
    except (json.JSONDecodeError, TypeError):
        return stderr[-_MAX_STDERR_TAIL:] or "route dump exited non-zero with no output"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, str):
            return error
    return stderr[-_MAX_STDERR_TAIL:] or "route dump exited non-zero with no output"


__all__ = ["RouteDumpFramework", "RouteDumpResult", "dump_route_module"]
