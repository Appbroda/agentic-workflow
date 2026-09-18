"""Resolve a name something named to the file in this checkout that defines it.

Two prompts in this platform are assembled from evidence that names a *symbol* and cannot
name a *file*, and both of them fail the same way when the file is not already in front of
the model:

* the in-attempt repair pass, whose commonest diagnostic is a missing import. The linter says
  ``src/controller.js:88  'permissionUtils' is not defined``, the repair prompt quotes thirty
  lines around line 88 -- which contain the *use* and never the definition -- and the model
  has no way to look. So it guesses a specifier, and a guessed specifier either fails the next
  pass or fails at review.
* the remediation retry after a Reviewer rejection, whose finding is frequently about
  something nothing imports yet -- "this duplicates the existing retry helper" -- which is
  often the whole point of the finding. Selection then falls back to lexical relevance, and
  lexical relevance scores the file at zero against the finding's wording.

This module answers the one question both need: **given an identifier, where does this
checkout export or define it?** It is the reverse of `_diagnostic_symbol_paths`, which
resolves only *through an import that already exists* and therefore returns nothing in exactly
the case that costs the most attempts -- for a missing import there is no binding to follow.

Deterministic, and deliberately so. There is no tool loop, no adapter change and no second
model call: this runs in Python before the call, exactly as the diagnostic region quoting
already does. "Named by evidence that exists" is the boundary. Pulling in an arbitrary file
for an arbitrary reason would need the model to ask for it, which this design does not have.

Three properties keep it honest:

* **A name that resolves to nothing costs nothing and quotes nothing.** Resolution is the
  filter, not the extraction: a diagnostic quoting a dependency's API or an ordinary English
  sentence produces candidates that no file in this checkout defines, so it produces no sites.
* **An ambiguous name resolves to several candidates and they are all offered**, oldest-first
  by path, up to the caller's bound. Several quoted candidates beat a blind guess at a
  specifier, and nothing here needs to know what any name means.
* **`is_model_safe_context_path` decides what may be scanned at all.** A file the path rule
  rejects is never read and never named. `carries_key_material` is deliberately *not* applied
  here: it is a decision about what may be sent, the two callers apply different policies to
  it -- the repair quotes nothing, the context snapshot withholds the bytes and reports the
  path through `required_omitted_paths` -- and answering it here would collapse both into
  silence, so "it never saw the file" would stop being answerable from the artifact.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import PurePosixPath

from tools.file_tools import is_model_safe_context_path
from tools.implementation_completeness import classify_file_change
from tools.reachability import GENERIC_SYMBOL_NAMES, exported_names

_LOGGER = logging.getLogger(__name__)

# Below this a name is too short for a definition-site match to be evidence of anything,
# which is the same floor the reachability inspection applies for the same reason.
_MIN_SYMBOL_LENGTH = 4
# How many distinct names one set of diagnostics may be searched for. A review with findings
# across several modules names a handful; a bound because each one is a pass over the scanned
# sources, and a diagnostic that names twenty identifiers is describing something this lookup
# was never going to resolve.
_MAX_SYMBOLS = 6
# How many files are read to answer. Generous, because the scan is local reads and the
# alternative is a wrong import specifier costing a whole attempt -- and bounded, because an
# unbounded walk of a monorepo is a cost nobody asked for. Truncation is logged, never silent.
_MAX_SCANNED_FILES = 2_000
# How many definition sites one name may contribute, so an ambiguous name cannot spend a
# caller's whole budget while a second name resolves to nothing quoted at all.
_MAX_SITES_PER_SYMBOL = 3

# A data file names a module and never defines one, so reading it can only produce a false
# site. Kept as suffixes rather than as a language decision: this says "generated or declared
# data", not "not source".
_DECLARES_WITHOUT_DEFINING = frozenset({".json", ".lock", ".toml", ".yaml", ".yml"})

# A name a diagnostic put in quotes or backticks. Every linter that reports an unresolved
# identifier writes it this way, which makes it the strongest evidence available here -- so a
# quoted name is taken whatever it looks like, and resolution decides whether it meant
# anything.
_QUOTED_NAME = re.compile(r"""['"`]([A-Za-z_$][\w$]*)['"`]""")
# Any bare word. What survives is decided by `_is_candidate` below, never by this pattern.
_BARE_NAME = re.compile(r"[A-Za-z_$][\w$]*")
# A filename with an extension, blanked before bare names are read. `AllApps.js` would
# otherwise contribute `AllApps`, and a diagnostic naming a *file* is already resolved by
# `_diagnostic_file_locations`, which knows the line as well. Two dotted segments only, and an
# all-lowercase tail of at most four characters: `appValidation.addAppSchema` keeps both of
# its names, and `envelope.pageBody` keeps both of its.
_FILENAME_TOKEN = re.compile(r"\b[A-Za-z_$][\w$-]*\.[a-z0-9]{1,4}\b")
# Anything with a path separator in it: a path, a lint rule id (`react-hooks/exhaustive-deps`),
# a URL. None of them is a name this checkout defines.
_PATH_TOKEN = re.compile(r"\S*[/\\]\S*")
# What separates an identifier from an English word: an internal capital after a lowercase
# letter, or an underscore. `permissionUtils`, `AllApps` and `retry_helper` pass; `This`,
# `Reviewer` and `delete` do not, which is what keeps prose about behaviour from resolving to
# whatever file happens to declare a common word. Sentence-initial capitals are the reason
# PascalCase alone is not enough.
_IDENTIFIER_SHAPE = re.compile(r"[a-z][A-Z]|_")


@dataclass(frozen=True, slots=True)
class DefinitionSite:
    """One place this checkout declares one name, and the line it declares it on."""

    symbol: str
    path: str
    line: int


@dataclass(frozen=True, slots=True)
class DefinitionIndex:
    """The files this lookup may read, and how to read one.

    A value rather than a service: both call sites already hold a checkout listing and a
    workspace-bound reader, and handing those over keeps this deterministic, synchronous and
    testable against an ordinary dictionary. ``read`` returns ``None`` for anything it cannot
    read as text, which is the same best-effort contract every other read on the repair path
    has -- a file that cannot be read contributes no site rather than raising.
    """

    paths: tuple[str, ...]
    read: Callable[[str], str | None]

    def including(self, paths: Iterable[str]) -> DefinitionIndex:
        """Return this index widened by files that did not exist when it was built.

        The repair loop needs it: a name may be defined in a module this very attempt wrote,
        and the checkout listing was taken before the coding call.
        """
        return DefinitionIndex(paths=tuple(dict.fromkeys([*self.paths, *paths])), read=self.read)

    def sites(self, symbols: Sequence[str], *, limit: int) -> tuple[DefinitionSite, ...]:
        """Return where this checkout defines these names, best candidates first."""
        return definition_sites(symbols, self.paths, self.read, limit=limit)


def definition_symbols(diagnostics: Sequence[str]) -> tuple[str, ...]:
    """Return the identifiers these diagnostics name, in first-seen order.

    Quoted names first and unconditionally: a linter writes the name it could not resolve in
    quotes, and that is the case this whole module exists for. Bare words are admitted only
    when they are shaped like an identifier rather than like English, because a Reviewer
    finding is mostly prose and a lookup that followed every word in it would resolve
    whichever file happens to declare a common one.

    Nothing here decides whether a name *means* anything. It decides only what is worth
    asking the checkout about, and the checkout's answer is what filters the rest.
    """
    named: list[str] = []
    for text in diagnostics:
        for match in _QUOTED_NAME.finditer(text):
            _admit(named, match.group(1))
        readable = _PATH_TOKEN.sub(" ", text)
        readable = _FILENAME_TOKEN.sub(" ", readable)
        for match in _BARE_NAME.finditer(readable):
            name = match.group(0)
            if _IDENTIFIER_SHAPE.search(name):
                _admit(named, name)
    return tuple(named[:_MAX_SYMBOLS])


def definition_sites(
    symbols: Sequence[str],
    paths: Sequence[str],
    read: Callable[[str], str | None],
    *,
    limit: int,
) -> tuple[DefinitionSite, ...]:
    """Return the places this checkout defines these names, one name at a time.

    Interleaved rather than concatenated: every name that resolves contributes its first site
    before any name contributes its second. An ambiguous name is worth quoting more than once
    -- several candidates beat a guess -- but never at the price of a second name resolving to
    nothing shown at all.

    Files are read in path order so the answer is the same on every run, and only production
    source is read: a test file declaring a fixture of the same name is not where the change
    should import from, and a data file declares nothing that can be imported anywhere.
    """
    wanted = [symbol for symbol in dict.fromkeys(symbols) if symbol]
    if not wanted or limit <= 0:
        return ()
    candidates = [path for path in sorted(dict.fromkeys(paths)) if _may_be_scanned(path)]
    scanned = candidates[:_MAX_SCANNED_FILES]
    if len(candidates) > len(scanned):
        # Named as a count rather than silently applied: a bound that nobody can see is
        # indistinguishable from "this checkout defines none of these names".
        _LOGGER.warning(
            "the definition-site lookup read %d of %d eligible files and never asked the "
            "remaining %d about %s [outcome=definition_scan_truncated]",
            len(scanned),
            len(candidates),
            len(candidates) - len(scanned),
            ", ".join(wanted),
        )
    found: dict[str, list[DefinitionSite]] = {symbol: [] for symbol in wanted}
    for path in scanned:
        outstanding = [symbol for symbol in wanted if len(found[symbol]) < _MAX_SITES_PER_SYMBOL]
        if not outstanding:
            break
        source = read(path)
        if source is None:
            continue
        for symbol in outstanding:
            if symbol not in source:
                continue
            line = definition_line(source, symbol)
            if line is not None:
                found[symbol].append(DefinitionSite(symbol=symbol, path=path, line=line))
    ordered: list[DefinitionSite] = []
    for rank in range(_MAX_SITES_PER_SYMBOL):
        for symbol in wanted:
            if rank < len(found[symbol]):
                ordered.append(found[symbol][rank])
    return tuple(ordered[:limit])


def definition_line(source: str, name: str) -> int | None:
    """Return the one-based line where this source declares ``name`` publicly, or None.

    "Publicly" is the whole test, and it is why a local `const permissionUtils` in an
    unrelated file is not an answer: what a repair needs is somewhere it can import from.
    The forms are the ones mainstream JavaScript, TypeScript and Python actually write, and
    they are anchored so a nested helper is never mistaken for a module's surface.

    `exported_names` is asked last and only as a membership test, because it is the shared
    parser of a file's public surface and it already handles the brace-list forms --
    `export { a, b }`, `module.exports = { pageBody }` -- whose declaration sits elsewhere in
    the file. Where it answers, the declaration line is looked for; where no declaration is
    found the first mention will do, since the point is to quote a window that contains the
    definition rather than to point at a character.

    One form here is deliberately absent from that shared parser: a Python module-level
    binding, `TIMEOUT = 30`. The reachability inspection must not treat one as an export --
    every extra name it accepts makes its gate quieter about genuinely dead code -- and this
    lookup must, because a module-level constant is exactly the kind of thing an attempt
    forgets to import.
    """
    for pattern in _declaration_patterns(name):
        match = pattern.search(source)
        if match is not None:
            return _line_of(source, match, name)
    if name not in exported_names(source):
        return None
    for pattern in (_local_declaration_pattern(name), _word_pattern(name)):
        match = pattern.search(source)
        if match is not None:
            return _line_of(source, match, name)
    return 1


def _line_of(source: str, match: re.Match[str], name: str) -> int:
    """Return the one-based line the name itself sits on inside this match.

    The name's own offset rather than the match's start, because several of the patterns
    deliberately consume the character before the declaration and one of them consumes the
    newline that ends the previous line -- which would report the declaration one line early.
    """
    offset = match.group(0).find(name)
    return source.count("\n", 0, match.start() + max(offset, 0)) + 1


def _admit(named: list[str], name: str) -> None:
    """Add one candidate name, if it is one worth asking the checkout about."""
    if name not in named and _is_candidate(name):
        named.append(name)


def _is_candidate(name: str) -> bool:
    """Say whether a definition-site match on this name would be evidence of anything.

    A private name is skipped for the reason the reachability inspection skips one: it is not
    something another module imports. The generic list is shared with that inspection because
    it is the same judgement -- a name common enough that finding it proves nothing -- and one
    list is the only way the two stay the same judgement.
    """
    return (
        len(name) >= _MIN_SYMBOL_LENGTH
        and not name.startswith("_")
        and name.lower() not in GENERIC_SYMBOL_NAMES
    )


def _may_be_scanned(path: str) -> bool:
    """Say whether this checkout path may be read at all in service of a model request."""
    return (
        is_model_safe_context_path(PurePosixPath(path))
        and classify_file_change(path) == "production"
        and PurePosixPath(path).suffix.lower() not in _DECLARES_WITHOUT_DEFINING
    )


@lru_cache(maxsize=512)
def _declaration_patterns(name: str) -> tuple[re.Pattern[str], ...]:
    """Return the anchored forms that declare ``name`` as this module's own public surface."""
    escaped = re.escape(name)
    return (
        # `export function x`, `export default class X`, `export const x`, `export type X`.
        re.compile(
            rf"^\s*export\s+(?:default\s+)?(?:async\s+)?"
            rf"(?:function\s*\*?|class|const|let|var|type|interface|enum)\s+{escaped}\b",
            re.MULTILINE,
        ),
        # `exports.x = ...` and `module.exports.x = ...`.
        re.compile(rf"^\s*(?:module\.)?exports\.{escaped}\s*=", re.MULTILINE),
        # `module.exports = x`, where the module's whole surface is one already-named thing.
        re.compile(rf"^\s*module\.exports\s*=\s*{escaped}\s*;?\s*$", re.MULTILINE),
        # Python module-level definitions. Column zero is what keeps a method out.
        re.compile(rf"^(?:async\s+)?def\s+{escaped}\s*\(", re.MULTILINE),
        re.compile(rf"^class\s+{escaped}\b", re.MULTILINE),
        # A Python module-level binding, annotated or not, and never a comparison.
        re.compile(rf"^{escaped}\s*(?::[^=\n]+)?=[^=]", re.MULTILINE),
    )


@lru_cache(maxsize=512)
def _local_declaration_pattern(name: str) -> re.Pattern[str]:
    """Return the declaration form of a name a brace list exports from elsewhere in the file."""
    return re.compile(rf"(?:^|[^\w$.])(?:const|let|var|function|class|def)\s+{re.escape(name)}\b")


@lru_cache(maxsize=512)
def _word_pattern(name: str) -> re.Pattern[str]:
    """Return a word-boundary match on one name, the last resort for locating it."""
    return re.compile(rf"(?:^|[^\w$]){re.escape(name)}\b")


__all__ = [
    "DefinitionIndex",
    "DefinitionSite",
    "definition_line",
    "definition_sites",
    "definition_symbols",
]
