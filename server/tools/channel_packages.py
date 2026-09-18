"""Name the unchanged modules that share a side-effect channel with this change.

AB-Feature-206's frontend change imported a channel package bare -- the package itself,
instead of the instance one repository module configures with a base URL and a credential
interceptor -- and every gate passed: lint, the scoped tests, the full suite, the build. The
configuration module was unchanged, so it never entered review evidence, and a reviewer
cannot flag a mismatch with a file it never read. The engineer's own imported-modules tier
(AB-Feature-128's fix) is structurally blind here too: importing a bare *package* surfaces no
repository file at all, so the module that configures the package is, by construction, absent
from the importer's import graph.

The question this module answers is narrow and derivable from the diff at the moment it
matters: which packages that a change's own sources import imply a configured side-effect
channel, and which other repository modules import the same package. That co-importer set is
the seam the change stands on -- in 206 it was exactly two files, the configuration module
and the new code itself -- and it is derived deterministically with the one import scanner
this platform has (`tools.reachability.module_specifiers`), never from reconnaissance prose,
which is allowed to fail and may cite a file for a different reason.

Repository-agnostic, per the standing rule: nothing here names a target repository or reads
any checkout's conventions. What makes a package a *channel* package -- an HTTP client, a
database driver or ORM, a queue or cache client -- is a per-ecosystem table owned by this
module the way `lockfile_hosts` owns lockfile names. The detection code is parameterized
over the table; adding an ecosystem is adding rows, not code, and no ecosystem branch exists
outside the table.

87- Part A: an allow-list of package names cannot keep up with the wrappers a repository
actually imports, and the thing worth reading was never the package. AB-Feature-218's
frontend imported `axios-hooks`, the table held only `axios`, the scan found no channel, and
the review approved a change whose two production blockers both lived in the module that
configures the client -- a `baseURL` already ending in `/api`, and a request interceptor
forcing `Content-Type: application/json` over a multipart body. So this module now answers a
second question beside "who else imports this package": which co-importer is the one the
repository *configured*. `channel_configuration_calls` decides that structurally -- a
constructor or configuration entry point reached through a name the channel package's own
import bound -- because a filename rule (`config/axios.js`, `lib/api.js`) is exactly the
per-repository encoding the paragraph above forbids. A channel whose configuration module
cannot be found is recorded as unlocated, never guessed at.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tools.reachability import import_bindings, module_specifiers, resolve_specifier

# How many co-importing modules the scan reports. Everything reported here is force-included
# somewhere -- the engineer's required set, the review's evidence -- so an unbounded list
# would let one popular package spend a whole context budget. Configuring a channel twice is
# rare, so the ordinary set is one or two files; a set past this ceiling is itself a fact a
# reader should see, which is why the scan says the list was cut rather than presenting a
# truncated list as the whole answer.
MAX_SEAM_CO_IMPORTERS = 8
# How many of the change's own imported modules are read to ask whether *they* open a
# channel. One hop, bounded: deeper is a graph walk with no budget here, and in a repository
# with a configured client nearly everything reaches it at depth two.
_MAX_TRANSITIVE_MODULES = 12
# How strong the evidence that a module configures a channel is, and therefore which module
# a bounded seam list shows first. Not a severity and not a threshold -- nothing here is
# filtered by rank, only ordered.
_RANK_ENTRY_POINT = 0
_RANK_CONSTRUCTOR = 1
_RANK_CALL_SITE = 2
# Below this a bound name matched against a whole file says nothing: a one- or two-letter
# alias appears inside no identifier but beside far too many.
_MIN_BINDING_LENGTH = 3


@dataclass(frozen=True, slots=True)
class ChannelEcosystem:
    """One ecosystem's channel-package rows, and the syntax facts needed to match them.

    Everything ecosystem-specific lives on this value: which packages imply a channel, which
    file suffixes its sources use, how a subpath of a package is spelled, and how its test
    frameworks replace a module with a double. The functions below iterate ecosystems and
    read these fields; none of them knows an ecosystem by name.
    """

    name: str
    packages: frozenset[str]
    source_suffixes: frozenset[str]
    # What follows a package name in a specifier that still names that package: `/` joins a
    # subpath in Node (`axios/lib/adapters`), `.` joins a submodule in Python
    # (`requests.adapters`).
    package_separator: str
    # How this ecosystem's test frameworks replace a module with a double, each pattern
    # capturing the replaced target as group 1. Deliberately the well-known spellings and no
    # more: a mock written another way is missed, which under-reports a note that only ever
    # informs -- the direction 47-D makes safe.
    mock_patterns: tuple[re.Pattern[str], ...]
    # The calls that *construct or configure* a channel rather than use one. Library API, the
    # same kind of ecosystem vocabulary the package rows are, and emphatically not a filename
    # or directory convention: `config/axios.js` and `lib/api.js` are two repositories' habits
    # and this platform is not allowed to carry either. A name here only counts when it is
    # reached through a binding the channel package's own import made, so the surface these
    # words are matched against is a handful of names per file, never the whole source.
    configuration_calls: frozenset[str] = frozenset()


# The two tables. A package earns a row when importing it *is* opening a side-effect channel
# -- HTTP clients, database drivers and ORMs, queue and cache clients -- because those are the
# packages a repository configures once (base URL, credentials, pool) and a change that
# imports one bare has stepped around that configuration. Utility packages do not belong
# here: a lodash import implies no channel and would make every change grow seam context.
#
# A *wrapper* over one of those clients earns a row on the same rule and for the same reason.
# AB-Feature-218's frontend imported `axios-hooks`, never `axios`, so the table matched
# nothing, the scan reported no channel, and the two blockers that shipped both lived in the
# module the wrapper is configured from: a `baseURL` already ending in `/api` and a request
# interceptor forcing `Content-Type: application/json` over a multipart body. The wrapper is
# configured once, exactly like the client under it -- `configure({ axios })`, a global
# fetcher, a QueryClient -- so importing it bare is the same step around the same
# configuration, and the row is what makes that configuration reachable.
_NODE_CHANNEL_PACKAGES = frozenset(
    {
        "@apollo/client",
        "@aws-sdk/client-s3",
        "@elastic/elasticsearch",
        "@prisma/client",
        "@tanstack/react-query",
        "amqplib",
        "axios",
        "axios-hooks",
        "better-sqlite3",
        "bull",
        "bullmq",
        "got",
        "graphql-request",
        "ioredis",
        "kafkajs",
        "knex",
        "ky",
        "ky-universal",
        "memcached",
        "mongodb",
        "mongoose",
        "mysql",
        "mysql2",
        "nats",
        "node-fetch",
        "pg",
        "react-query",
        "redis",
        "sequelize",
        "socket.io-client",
        "sqlite3",
        "superagent",
        "swr",
        "typeorm",
        "undici",
        "ws",
    }
)
# Import names, not distribution names: the scanner sees `import psycopg2`, never
# `pip install psycopg2-binary`. The standard-library clients are deliberately absent --
# `urllib` would match every `urllib.parse` and make ordinary string handling grow seam
# context for a channel it never opens.
_PYTHON_CHANNEL_PACKAGES = frozenset(
    {
        "aio_pika",
        "aiohttp",
        "aiomysql",
        "asyncpg",
        "boto3",
        "celery",
        "confluent_kafka",
        "elasticsearch",
        "grpc",
        "httpx",
        "kafka",
        "motor",
        "peewee",
        "pika",
        "psycopg",
        "psycopg2",
        "pymemcache",
        "pymongo",
        "pymysql",
        "redis",
        "requests",
        "sqlalchemy",
        "urllib3",
        "websocket",
        "websockets",
    }
)

CHANNEL_ECOSYSTEMS: tuple[ChannelEcosystem, ...] = (
    ChannelEcosystem(
        name="node",
        packages=_NODE_CHANNEL_PACKAGES,
        source_suffixes=frozenset({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}),
        package_separator="/",
        mock_patterns=(
            # jest.mock('axios') / vi.mock('axios'): the whole package becomes a double.
            re.compile(r"""\b(?:jest|vi)\s*\.\s*mock\s*\(\s*['"]([^'"]+)['"]"""),
        ),
        configuration_calls=frozenset(
            {
                "configure",
                "connect",
                "create",
                "createClient",
                "createConnection",
                "createInstance",
                "createPool",
                "extend",
                "setDefaults",
            }
        ),
    ),
    ChannelEcosystem(
        name="python",
        packages=_PYTHON_CHANNEL_PACKAGES,
        source_suffixes=frozenset({".py"}),
        package_separator=".",
        mock_patterns=(
            # unittest.mock.patch('requests.post'), mock.patch(...), mocker.patch(...): the
            # dotted target's root names the replaced package.
            re.compile(r"""\bpatch\s*\(\s*['"]([^'"]+)['"]"""),
            # monkeypatch.setattr('requests.post', ...) and the unquoted module form,
            # monkeypatch.setattr(requests, 'post', ...).
            re.compile(
                r"""\bmonkeypatch\s*\.\s*(?:setattr|setitem|delattr)\s*\(\s*['"]([^'"]+)['"]"""
            ),
            re.compile(r"""\bmonkeypatch\s*\.\s*setattr\s*\(\s*([A-Za-z_][\w.]*)\s*,"""),
        ),
        configuration_calls=frozenset(
            {
                "AsyncClient",
                "Client",
                "ClientSession",
                "MongoClient",
                "Redis",
                "Session",
                "client",
                "connect",
                "connect_robust",
                "create_engine",
                "from_url",
                "resource",
                "sessionmaker",
            }
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class SeamCoImporter:
    """One unchanged repository module that imports a channel package this change imports.

    ``configuration_calls`` is what makes this module the seam rather than one more call
    site: the constructor or configuration entry points it reaches through the channel
    package's own import, spelled as they appear (`Axios.create`, `configure`,
    `mongoose.connect`). Empty means this module only *uses* the channel.
    """

    path: str
    channel_packages: tuple[str, ...]
    configuration_calls: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChannelSeamScan:
    """What the change's sources import over a channel, and who else imports the same thing.

    ``co_importer_count`` counts the whole set before the ceiling, so a reader can tell "two
    modules share this channel" from "eight were shown and more exist" -- a large set is
    itself something a reviewer should see, and a cut list presented without saying so would
    read as the whole answer.

    ``configuration_unlocated`` names the channels this scan could not find a configuration
    module for. It is the recorded limitation A2 was told to prefer over a filename rule: a
    repository that configures its client somewhere this detection cannot see says so, and a
    reader knows the difference between "checked, nothing configures it" and silence.
    """

    packages: tuple[str, ...]
    co_importers: tuple[SeamCoImporter, ...]
    co_importer_count: int
    truncated: bool
    configuration_unlocated: tuple[str, ...] = ()

    @property
    def configuration_paths(self) -> tuple[str, ...]:
        """The reported co-importers that configure a channel, strongest evidence first."""
        return tuple(item.path for item in self.co_importers if item.configuration_calls)


def specifier_names_package(specifier: str, package: str, separator: str) -> bool:
    """Say whether one import specifier names this package or a subpath of it."""
    return specifier == package or specifier.startswith(package + separator)


def _ecosystems_for(path: str) -> tuple[ChannelEcosystem, ...]:
    """Return the ecosystems whose sources use this file's suffix."""
    suffix = PurePosixPath(path).suffix
    return tuple(item for item in CHANNEL_ECOSYSTEMS if suffix in item.source_suffixes)


def channel_imports(path: str, source: str) -> tuple[str, ...]:
    """Return the channel packages this one source file imports, sorted.

    The file's suffix decides which ecosystems' tables are consulted, so a package name two
    ecosystems share -- `redis` is in both tables -- is matched under the syntax the file is
    actually written in.
    """
    found: set[str] = set()
    ecosystems = _ecosystems_for(path)
    if not ecosystems:
        return ()
    for specifier in module_specifiers(source):
        for ecosystem in ecosystems:
            for package in ecosystem.packages:
                if specifier_names_package(specifier, package, ecosystem.package_separator):
                    found.add(package)
    return tuple(sorted(found))


def mocked_channel_packages(path: str, source: str) -> tuple[str, ...]:
    """Return the channel packages this file replaces with a test double, sorted.

    Matching goes through each ecosystem's own mock spellings and the same
    specifier-names-package rule imports use, so `patch('requests.post')` names `requests`
    and `jest.mock('./api/client')` names nothing -- a repository's own module is not a
    channel package, whatever it wraps.
    """
    found: set[str] = set()
    for ecosystem in _ecosystems_for(path):
        for pattern in ecosystem.mock_patterns:
            for match in pattern.finditer(source):
                target = match.group(1)
                for package in ecosystem.packages:
                    if specifier_names_package(target, package, ecosystem.package_separator):
                        found.add(package)
    return tuple(sorted(found))


def channel_configuration_calls(
    path: str, source: str, packages: Collection[str]
) -> tuple[str, ...]:
    """Return the configuration entry points this file reaches through a channel import.

    The question A2 exists to answer, and the reason it is asked this way. AB-Feature-218's
    two blockers both lived in the module that ran `Axios.create({ baseURL })` and
    `configure({ axios })` -- a base URL already ending in `/api`, and a request interceptor
    forcing a JSON content type over a multipart body. Neither is inferable from the diff, and
    the file appears in no diff, so the only way a reviewer sees them is if this platform can
    point at the module. Pointing at it by name (`config/axios.js`) would encode one
    repository's convention; pointing at what it *does* does not.

    Two shapes count, and both require the callee to be a name this file's own import of the
    channel package bound:

    * an entry point from the ecosystem's table, called as the binding (`configure(`) or
      through it (`Axios.create(`, `mongoose.connect(`, `psycopg2.connect(`); and
    * a constructor -- `new Pool(`, or a capitalised member (`httpx.Client(`,
      `redis.Redis(`) -- because a repository configuring a client through its class has no
      verb for this table to hold.

    What is deliberately *not* here: any judgement about the path, and any call whose callee
    is not bound by the channel import. `useAxios(...)` is a hook, twenty sibling modules
    called it in 218, and if a call site read as configuration the ordering this feeds would
    be worth nothing. Read line by line through the one import scanner this platform has, so
    an import split across lines is missed -- under-detection, which records an omission
    rather than a wrong file, and is the direction 47-D makes safe.
    """
    hits = _configuration_hits(path, source, packages)
    return tuple(sorted({name for ranked in hits.values() for _rank, name in ranked}))


def _configuration_hits(
    path: str, source: str, packages: Collection[str]
) -> dict[str, tuple[tuple[int, str], ...]]:
    """Return, per channel package, the ranked configuration calls this file makes.

    Rank orders evidence and nothing else. An entry point the ecosystem table names outranks
    a bare constructor shape, because the constructor shape is the looser of the two: a
    Mongoose repository's every model file calls `mongoose.Schema(`, and a reader asking
    which module holds the connection must not have to read past twenty of them to find the
    one that calls `mongoose.connect(`.
    """
    found: dict[str, set[tuple[int, str]]] = {}
    for ecosystem in _ecosystems_for(path):
        for line in source.splitlines():
            specifiers = module_specifiers(line)
            if not specifiers:
                continue
            named = [
                package
                for package in packages
                if package in ecosystem.packages
                and any(
                    specifier_names_package(specifier, package, ecosystem.package_separator)
                    for specifier in specifiers
                )
            ]
            if not named:
                continue
            for binding in sorted(import_bindings(line)):
                for rank, call in _binding_configuration_calls(binding, source, ecosystem):
                    for package in named:
                        found.setdefault(package, set()).add((rank, call))
    return {package: tuple(sorted(ranked)) for package, ranked in found.items()}


def _binding_configuration_calls(
    binding: str, source: str, ecosystem: ChannelEcosystem
) -> tuple[tuple[int, str], ...]:
    """Return the ranked configuration calls made through one bound name."""
    if len(binding) < _MIN_BINDING_LENGTH:
        # A one-letter alias matched against a whole file is noise, not evidence.
        return ()
    escaped = re.escape(binding)
    found: set[tuple[int, str]] = set()
    if binding in ecosystem.configuration_calls and re.search(rf"\b{escaped}\s*\(", source):
        found.add((_RANK_ENTRY_POINT, binding))
    if re.search(rf"\bnew\s+{escaped}\s*\(", source):
        found.add((_RANK_CONSTRUCTOR, binding))
    for match in re.finditer(rf"\b{escaped}\s*\.\s*(\w+)\s*\(", source):
        member = match.group(1)
        if member in ecosystem.configuration_calls:
            found.add((_RANK_ENTRY_POINT, f"{binding}.{member}"))
        elif member[:1].isupper():
            found.add((_RANK_CONSTRUCTOR, f"{binding}.{member}"))
    return tuple(sorted(found))


def _transitive_channel_imports(
    sources: Sequence[tuple[str, str]],
    repository_paths: Collection[str],
    read: Callable[[str], str | None],
) -> dict[str, set[str]]:
    """Return the channels the change reaches through its own modules, one hop out.

    A change that imports the repository's own client wrapper opens the same channel as one
    that imports the package -- and it is the more common shape of the two, because it is the
    shape a repository with a configured client is *supposed* to produce. 218's frontend
    happened to import the wrapper package directly; the next one will import
    `./api/client`, and a scan that only reads the change's own text would find no channel
    at all.

    One hop, and bounded at ``_MAX_TRANSITIVE_MODULES`` reads. Deeper is a graph walk this
    module has no budget for, and the second hop is where the false positives live: nearly
    everything in a repository transitively reaches its HTTP client.
    """
    existing = {Path(path) for path in repository_paths}
    changed = {path for path, _ in sources}
    hits: dict[str, set[str]] = {}
    read_count = 0
    for path, content in sources:
        directory = PurePosixPath(path).parent
        for specifier in module_specifiers(content):
            if read_count >= _MAX_TRANSITIVE_MODULES:
                return hits
            target = resolve_specifier(specifier, directory, existing)
            if target is None or target in changed:
                continue
            read_count += 1
            neighbour = read(target)
            if neighbour is None:
                continue
            for ecosystem in _ecosystems_for(target):
                for package in channel_imports(target, neighbour):
                    if package in ecosystem.packages:
                        hits.setdefault(ecosystem.name, set()).add(package)
    return hits


def channel_seam_scan(
    sources: Sequence[tuple[str, str]],
    repository_paths: Iterable[str],
    read: Callable[[str], str | None],
    *,
    exclude: Collection[str] = (),
) -> ChannelSeamScan:
    """Return the repository modules that co-import a channel package this change imports.

    ``sources`` are the change's own files as ``(path, content)`` -- whatever the caller
    holds as "this change": the reviewer's accumulated change set, the engineer's assigned
    files and prior-attempt diff. Their paths are excluded from the co-importer set
    automatically, because the change is one side of the seam and the answer wanted is the
    other side.

    The common case still pays almost nothing. A change with no channel import spends no
    checkout walk, and the only reads it can cause are ``_MAX_TRANSITIVE_MODULES`` of its own
    directly imported modules -- files a caller assembling change context is holding anyway.
    A specifier that names a dependency rather than a checkout file resolves to nothing and
    reads nothing, so a change importing only utilities (`lodash/merge`) reads no file at
    all, which is the regression 72- named and this part had to keep.

    ``read`` returning ``None`` skips the file: an unreadable module cannot be shown as seam
    context anyway, and this scan reports evidence, never the absence of it.
    """
    # Materialised once: the caller may pass a generator, and this is read twice -- by the
    # one-hop resolution below and by the checkout walk.
    checkout = sorted(set(repository_paths))
    hits: dict[str, set[str]] = {}
    for path, content in sources:
        for ecosystem in _ecosystems_for(path):
            for specifier in module_specifiers(content):
                for package in ecosystem.packages:
                    if specifier_names_package(specifier, package, ecosystem.package_separator):
                        hits.setdefault(ecosystem.name, set()).add(package)
    # One hop through the change's own modules: importing this repository's client wrapper
    # opens the same channel as importing the package, and is the shape a repository with a
    # configured client is supposed to produce.
    for name, names in _transitive_channel_imports(sources, checkout, read).items():
        hits.setdefault(name, set()).update(names)
    packages = tuple(sorted({package for names in hits.values() for package in names}))
    if not hits:
        return ChannelSeamScan(packages=(), co_importers=(), co_importer_count=0, truncated=False)
    excluded = {path for path, _ in sources} | set(exclude)
    by_name = {item.name: item for item in CHANNEL_ECOSYSTEMS}
    co_importing: dict[str, set[str]] = {}
    configuring: dict[str, dict[str, tuple[tuple[int, str], ...]]] = {}
    located: set[str] = set()
    for path in checkout:
        if path in excluded:
            continue
        wanted = [
            (by_name[name], names)
            for name, names in sorted(hits.items())
            if PurePosixPath(path).suffix in by_name[name].source_suffixes
        ]
        if not wanted:
            continue
        candidate = read(path)
        # A file that never spells any hit package cannot import one; the substring probe
        # keeps the specifier scan off the overwhelming majority of the checkout.
        if candidate is None or not any(
            package in candidate for _, names in wanted for package in names
        ):
            continue
        for specifier in module_specifiers(candidate):
            for ecosystem, names in wanted:
                for package in names:
                    if specifier_names_package(specifier, package, ecosystem.package_separator):
                        co_importing.setdefault(path, set()).add(package)
        if path not in co_importing:
            continue
        # Asked over every channel package *this file* imports, not only the ones the change
        # shares with it. 218's client module creates the instance from `axios` and registers
        # it with `axios-hooks`; the change imported only the second. Narrowing the question
        # to the shared package would have found `configure` and missed `Axios.create` -- the
        # call that carries the base URL, which is the whole point of reading the file.
        found = _configuration_hits(path, candidate, channel_imports(path, candidate))
        if found:
            configuring[path] = found
            # The channels this module is the configuration *for*: the ones it shares with
            # the change. That is the claim `configuration_files` makes, so it is the claim
            # `configuration_not_located` must be the complement of.
            located |= co_importing[path]
    # A change that configures the channel itself needs nothing pointed out to it -- the
    # reviewer is reading those lines in the diff already.
    for path, content in sources:
        if _configuration_hits(path, content, channel_imports(path, content)):
            located |= {
                package
                for ecosystem in _ecosystems_for(path)
                for package in channel_imports(path, content)
                if package in ecosystem.packages
            }
    # Configuration modules first, strongest evidence first within that, then path order.
    # This is the whole of A2's ordering claim and it is what makes the bound survivable: in
    # 218 twenty sibling call sites shared the channel and every one of them sorted ahead of
    # `src/config/...`, so a path-ordered list cut at its ceiling would have shown the
    # reviewer twenty hooks and not the base URL.
    ordered = sorted(
        co_importing,
        key=lambda path: (*_seam_rank(configuring.get(path), co_importing[path]), path),
    )
    return ChannelSeamScan(
        packages=packages,
        co_importers=tuple(
            SeamCoImporter(
                path=path,
                channel_packages=tuple(sorted(co_importing[path])),
                configuration_calls=tuple(
                    sorted(
                        {
                            call
                            for ranked in configuring.get(path, {}).values()
                            for _rank, call in ranked
                        }
                    )
                ),
            )
            for path in ordered[:MAX_SEAM_CO_IMPORTERS]
        ),
        co_importer_count=len(ordered),
        truncated=len(ordered) > MAX_SEAM_CO_IMPORTERS,
        configuration_unlocated=tuple(item for item in packages if item not in located),
    )


def _seam_rank(
    found: dict[str, tuple[tuple[int, str], ...]] | None, shared: Collection[str]
) -> tuple[int, int]:
    """Rank one co-importer: whose channel it configures first, then how strongly.

    Two keys, because a co-importer can configure a channel that has nothing to do with this
    change. `admanager_console-2.0` has one utility module that imports Mongoose and
    configures an HTTP client with `axios.create`: a genuine configuration module, and not
    the one a Mongoose change needs to read. So the change's *own* channel comes first, and
    only then does an entry point (`mongoose.connect`) beat a constructor
    (`mongoose.Schema`, which every one of that repository's thirty-five model files calls).
    """
    if not found:
        return (1, _RANK_CALL_SITE)
    relevant = [
        rank for package, ranked in found.items() if package in shared for rank, _call in ranked
    ]
    if relevant:
        return (0, min(relevant))
    return (1, min(rank for ranked in found.values() for rank, _call in ranked))


__all__ = [
    "CHANNEL_ECOSYSTEMS",
    "MAX_SEAM_CO_IMPORTERS",
    "ChannelEcosystem",
    "ChannelSeamScan",
    "SeamCoImporter",
    "channel_configuration_calls",
    "channel_imports",
    "channel_seam_scan",
    "mocked_channel_packages",
    "specifier_names_package",
]
