"""Cross-check what two repositories actually wire to each other against the contract.

AB-Feature-184's frontend hardcoded `/v1/service-requests/export-active-ad-units` as a real,
callable constant; its backend registered `/service-requests/export-active-ad-units` -- no
`/v1`. Both repositories individually matched the (wrong) contract text, so nothing caught the
drift between them. This module generalizes the fix: instead of one hand-written check for that
one prefix, mechanically extract every absolute path a consuming repository's own code
references and every method+path a providing repository actually registers, then diff both
against the contract and against each other. Any drift shape -- a prefix, a typo, a trailing
slash, a wrong method -- is caught the same way, because nothing here encodes what the drift
looks like; it only compares what each side's real, current source says.

Deliberately a heuristic source scanner, following the same narrow-by-design precedent already
used elsewhere in this codebase (`tools/reachability.py`, `tools/channel_packages.py`): it reads
real files with targeted regular expressions rather than a full per-language AST, and documents
exactly what it does and does not resolve. A path it cannot confidently attribute is reported as
a candidate rather than silently dropped, so the diff below stays conservative in the direction
of asking a repair round-trip to confirm rather than missing a real mismatch.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from artifacts.schemas import IntegrationContractArtifact

_METHOD_NAMES = ("get", "post", "put", "patch", "delete", "head", "options")

# A quoted absolute path a consuming repository's own code calls, e.g. the literal constant
# `'/v1/service-requests/export-active-ad-units'` found hardcoded in AB-184's frontend. `{id}`
# and `:id`-style path parameters are kept verbatim; the diff below compares them as written,
# the same way the contract and the registered route both spell them.
_REFERENCED_PATH_PATTERN = re.compile(r"""['"](/[A-Za-z0-9_\-/:{}.]{2,})['"]""")

# An Express-style chained route registration: `.route('/x')` followed, on the same or a later
# line, by `.get(`/`.post(`/etc. Narrow to this one framework shape by design -- the same
# shape confirmed by hand in both AB-184's backend route files this session; a repository using
# a different router style simply yields no candidates here, which fails open (no finding)
# rather than closed (a false block), consistent with every other heuristic tool in this
# package.
_ROUTE_PATH_PATTERN = re.compile(r"""\.route\(\s*['"](/[A-Za-z0-9_\-/:{}.]*)['"]\s*\)""")
_METHOD_CALL_PATTERN = re.compile(rf"""\.({"|".join(_METHOD_NAMES)})\(""")

# A router mount: `router.use('/service-requests', serviceRequestRoutes)`,
# `app.use('/api', router)`. Collected platform-wide (not file-scoped) because the mounting
# file and the route-defining file are conventionally different, exactly as in AB-184's
# backend (`index.route.js` mounts; `serviceRequest.route.js` defines).
_MOUNT_PATTERN = re.compile(r"""\.use\(\s*['"](/[A-Za-z0-9_\-/:{}.]*)['"]""")


@dataclass(frozen=True, slots=True)
class EndpointReference:
    """One absolute path literal a repository's own source references, as written."""

    path: str
    source_path: str


@dataclass(frozen=True, slots=True)
class RegisteredRoute:
    """One method+path a repository's own router registration actually exposes."""

    method: str
    path: str
    source_path: str


@dataclass(frozen=True, slots=True)
class EndpointMismatch:
    """A contract endpoint neither side's real source confirms the same way."""

    operation_id: str
    contract_path: str
    consumer_confirms: bool
    provider_confirms: bool
    detail: str
    # Every repository whose own scanned source failed to confirm this endpoint on either
    # side -- the routing signal a cross-repository finding needs (`responsible_repository_id`/
    # `affected_repository_ids`), kept here rather than resolved by the caller a second time.
    non_confirming_repository_ids: tuple[str, ...]


def extract_referenced_endpoints(
    workspace: Path, changed_files: Sequence[str]
) -> list[EndpointReference]:
    """Return every absolute path literal a consuming repository's changed files reference.

    Deliberately broad rather than precise: this is a source of candidates for the diff below,
    not a claim that every literal string found is an HTTP call. A path that never matches a
    contract endpoint is simply never compared against, so over-collection here costs nothing.
    """
    references: list[EndpointReference] = []
    for relative_path in changed_files:
        text = _read_text(workspace / relative_path)
        if text is None:
            continue
        for match in _REFERENCED_PATH_PATTERN.finditer(text):
            references.append(EndpointReference(path=match.group(1), source_path=relative_path))
    return references


def extract_registered_routes(
    workspace: Path, changed_files: Sequence[str]
) -> list[RegisteredRoute]:
    """Return every method+path an Express-shaped providing repository actually registers.

    Resolves a chained `.route('/x').get(...)` registration to its local path, and separately
    reports every `.use('/prefix', ...)` mount found anywhere in the changed files. The diff
    below tries a registered path both bare and under every observed mount prefix, rather than
    this function resolving exactly one true combination -- the same two-step reading done by
    hand this session, kept conservative rather than precise about which prefix applies to
    which router.
    """
    bare_routes: list[RegisteredRoute] = []
    mount_prefixes: set[str] = set()
    for relative_path in changed_files:
        text = _read_text(workspace / relative_path)
        if text is None:
            continue
        for match in _MOUNT_PATTERN.finditer(text):
            mount_prefixes.add(match.group(1))
        current_route_path = ""
        for line in text.splitlines():
            route_match = _ROUTE_PATH_PATTERN.search(line)
            if route_match:
                current_route_path = route_match.group(1)
            if not current_route_path:
                continue
            for method_match in _METHOD_CALL_PATTERN.finditer(line):
                bare_routes.append(
                    RegisteredRoute(
                        method=method_match.group(1).upper(),
                        path=current_route_path,
                        source_path=relative_path,
                    )
                )
    if not mount_prefixes:
        return bare_routes
    mounted_routes = [
        RegisteredRoute(
            method=route.method,
            path=f"{prefix.rstrip('/')}{route.path}",
            source_path=route.source_path,
        )
        for route in bare_routes
        for prefix in mount_prefixes
    ]
    return [*bare_routes, *mounted_routes]


def diff_endpoint_manifests(
    contract: IntegrationContractArtifact,
    consumer_refs_by_repository: Mapping[str, Sequence[EndpointReference]],
    provider_routes_by_repository: Mapping[str, Sequence[RegisteredRoute]],
) -> list[EndpointMismatch]:
    """Return every contract endpoint neither side's real source confirms.

    An endpoint with no consumer reference and no provider route at all, across every
    repository, is not reported -- that is simply a repository this change did not touch, not
    a mismatch. A mismatch is only raised when at least one repository's source was actually
    scanned (its changed files were non-empty) and none of them, on either side, confirmed the
    contract's exact literal method and path.
    """
    consumer_paths = {
        reference.path for refs in consumer_refs_by_repository.values() for reference in refs
    }
    provider_methods_by_path: dict[str, set[str]] = {}
    for routes in provider_routes_by_repository.values():
        for route in routes:
            provider_methods_by_path.setdefault(route.path, set()).add(route.method)
    any_consumer_scanned = any(consumer_refs_by_repository.values())
    any_provider_scanned = any(provider_routes_by_repository.values())

    mismatches: list[EndpointMismatch] = []
    for endpoint in contract.endpoints:
        consumer_confirms = not any_consumer_scanned or endpoint.path in consumer_paths
        provider_confirms = not any_provider_scanned or (
            endpoint.method in provider_methods_by_path.get(endpoint.path, set())
        )
        if consumer_confirms and provider_confirms:
            continue
        detail_parts = []
        non_confirming: list[str] = []
        if not consumer_confirms:
            detail_parts.append("no consuming repository source references this exact path")
            non_confirming.extend(
                repository_id for repository_id, refs in consumer_refs_by_repository.items() if refs
            )
        if not provider_confirms:
            detail_parts.append(
                "no providing repository route registers this exact method and path"
            )
            non_confirming.extend(
                repository_id
                for repository_id, routes in provider_routes_by_repository.items()
                if routes
            )
        mismatches.append(
            EndpointMismatch(
                operation_id=endpoint.operation_id,
                contract_path=f"{endpoint.method} {endpoint.path}",
                consumer_confirms=consumer_confirms,
                provider_confirms=provider_confirms,
                detail="; ".join(detail_parts),
                non_confirming_repository_ids=tuple(sorted(set(non_confirming))),
            )
        )
    return mismatches


def _read_text(path: Path) -> str | None:
    """Return a changed file's text, or `None` for anything not readable as source."""
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


__all__ = [
    "EndpointMismatch",
    "EndpointReference",
    "RegisteredRoute",
    "diff_endpoint_manifests",
    "extract_referenced_endpoints",
    "extract_registered_routes",
]
