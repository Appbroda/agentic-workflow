"""Ask the checkout whether the code a change adds can be reached when it runs.

A component nobody renders and a route nobody registers are finished files and unfinished
work. This check is the single largest source of wasted attempts in this platform's history
-- 15 of 62 attempt-0s on live runs 66 through 106, 24%, both pilot repositories, both
languages -- and until now it existed in exactly one place: the runtime, *after* the Engineer
had returned and declared itself `completed`. So a wiring miss could not be repaired where it
was made. It always cost a full outer attempt and a fifteen-to-forty-minute
re-implementation.

The logic lives here so it can run in **two layers**, which is the point of the move:

* inside the attempt, as fast feedback the Engineer's own bounded repair loop can clear; and
* in the runtime, unchanged, as the **authoritative** gate.

Those two are not interchangeable and the second is not optional. The Engineer clearing its
own reachability check must never be what decides the question -- an agent that both writes
the code and rules on it has no gate at all -- so the runtime still asks, from scratch, over
the workspace as it finally stands.

Nothing here knows any framework. Whether an unreferenced file is a defect depends on the
repository, so the checkout is asked instead of assumed: if the new file's existing
neighbours are reached from elsewhere, this directory is wired by hand and an unreferenced
newcomer is dead code; if they are not, the directory is discovered by convention and nothing
is reported. The same reasoning is applied one level down for a new export in an existing
file -- see ``_modified_export_candidates``.

Three blind spots the single-layer version had, all closed here so both layers benefit:

1. **Added files only.** A new export added to an *existing* file that nothing calls was
   invisible. ``_modified_export_candidates`` now asks HEAD what that file used to export and
   treats the difference as a candidate.
2. **A silent cap.** The candidate list was truncated to eight and nothing said so, which
   made a change adding nine modules read as "wiring passed". The drop is now logged and
   returned, so no reader can mistake a truncated run for a clean one.
3. **The symbol was the filename stem.** A file whose exports do not share its filename was
   checked against the wrong token. ``candidate_symbols`` parses the file's actual exported
   names and keeps the stem as a fallback -- and keeps ``-w`` and case sensitivity, because a
   substring of a longer identifier is not a reference and the -072/-073 consoles shipped
   three unreachable pages proving it.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from services.cancellation import CancellationToken, MockCancellationToken
from services.process_runner import ProcessRunner, repository_subprocess_environment
from tools.file_tools import PathLike
from tools.implementation_completeness import classify_file_change
from tools.repository_reconnaissance import registers_modules

_LOGGER = logging.getLogger(__name__)

# How every diagnostic the *in-attempt* layer composes begins. The Engineer's repair loop
# matches on this constant rather than on a copy of a sentence, so the producer of a
# diagnostic and the gate that admits it cannot drift apart. The runtime's own blocking
# issues are deliberately left untagged: they are read by people and classified by the retry
# policy, and this move must not change a single one of their bytes.
REACHABILITY_DIAGNOSTIC_PREFIX = "This change added code the running application cannot reach"

# How many candidates and neighbours are inspected, because each answer costs a `git grep`
# over the whole checkout. Added files are examined before modified-file exports, so the
# higher-confidence case can never be squeezed out by the one added here.
_MAX_CANDIDATES = 8
_MAX_NEIGHBOURS = 6
# How many modified production files are asked what they used to export. Bounded for the same
# reason: one `git show` each, and an attempt that edits thirty files is a different problem.
_MAX_MODIFIED_FILES = 12
# Below this a name is too generic for a textual reference to mean anything.
_MIN_SYMBOL_LENGTH = 4
# The largest file this reads to parse its exports. Well above any source file; a bound
# because this read bypasses the workspace file tools.
_MAX_SOURCE_BYTES = 2_000_000
# A data file names a module; it never calls one. Kept separate from the documentation rule
# because a generated contract is neither documentation nor a directory anyone can exclude.
_DESCRIBES_WITHOUT_REACHING = frozenset({".json", ".yaml", ".yml", ".toml", ".lock"})
_IMPORT_KEYWORDS = frozenset({"as", "const", "default", "from", "import", "let", "require", "var"})
# The filename shapes a module specifier may resolve to in this checkout. The empty suffix
# comes first so a specifier that already spells its extension resolves to itself.
_MODULE_SUFFIXES = ("", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".py")
# A JavaScript/TypeScript `require('x')` or `from 'x'`, and a Python `import x` / `from x`.
# Only the specifier is captured; what a specifier means -- a checkout file, a dependency, a
# channel package -- is the caller's question. This is the platform's one import scanner:
# the engineer's context selector and the channel-package detection both read imports through
# it, because two scanners drift apart silently.
MODULE_SPECIFIER = re.compile(
    r"""require\(\s*['"](?P<req>[^'"]+)['"]\s*\)"""
    r"""|from\s+['"](?P<js>[^'"]+)['"]"""
    r"""|^\s*(?:from|import)\s+(?P<py>[A-Za-z_][\w.]*)""",
    re.MULTILINE,
)

# Exported names too common for a word-boundary grep to be evidence of anything. Filtering
# one is the *safe* direction here and that asymmetry is the whole justification: the stem is
# always a candidate too, and the stem is what catches a path-based reference
# (`require('./routes/statusFormatter')`), so dropping a generic export name can only make
# this check stricter. Keeping one would let `module.exports = router` match every router in
# the checkout and report a genuinely dead module as wired, which is the failure mode that
# produced -072 and -073.
#
# Public, because the definition-site lookup makes the same judgement -- a name common enough
# that finding it proves nothing -- and two copies of a judgement drift apart silently.
GENERIC_SYMBOL_NAMES = frozenset(
    {
        "app",
        "client",
        "config",
        "constants",
        "context",
        "data",
        "default",
        "error",
        "exports",
        "handler",
        "handlers",
        "index",
        "instance",
        "item",
        "items",
        "logger",
        "main",
        "middleware",
        "model",
        "models",
        "module",
        "name",
        "options",
        "params",
        "path",
        "payload",
        "props",
        "request",
        "response",
        "result",
        "root",
        "route",
        "router",
        "routes",
        "schema",
        "server",
        "service",
        "settings",
        "setup",
        "state",
        "store",
        "type",
        "types",
        "utils",
        "value",
    }
)

# What a file declares as its public surface, in the forms mainstream JavaScript, TypeScript
# and Python actually write. Deliberately shallow -- no parser, no language detection -- and
# deliberately anchored: the Python patterns require column zero so a nested helper is not
# mistaken for a module export, which is the difference between a public name and a local one.
_EXPORT_PATTERNS = (
    # `export function x`, `export default class X`, `export const x`, `export type X`.
    re.compile(
        r"^\s*export\s+(?:default\s+)?(?:async\s+)?"
        r"(?:function\s*\*?|class|const|let|var|type|interface|enum)\s+"
        r"([A-Za-z_$][\w$]*)",
        re.MULTILINE,
    ),
    # `exports.x = ...` and `module.exports.x = ...`, the CommonJS single-name forms.
    re.compile(r"^\s*(?:module\.)?exports\.([A-Za-z_$][\w$]*)\s*=", re.MULTILINE),
    # `module.exports = x`, where the module's whole surface is one already-named thing.
    re.compile(r"^\s*module\.exports\s*=\s*([A-Za-z_$][\w$]*)\s*;?\s*$", re.MULTILINE),
    # Python module-level definitions. Column zero is the anchor that keeps a method out.
    re.compile(r"^(?:async\s+)?def\s+([A-Za-z_]\w*)", re.MULTILINE),
    re.compile(r"^class\s+([A-Za-z_]\w*)", re.MULTILINE),
)
# `export { a, b as c }` and `module.exports = { a, b: c }`: a brace list whose entries name
# the public surface. Both sides of an alias are taken -- the public name and the local one --
# because a union of candidates can only make this check laxer, and a laxer answer here is a
# quieter gate rather than a wrong instruction.
_EXPORT_LIST_PATTERN = re.compile(
    r"^\s*(?:export\s*\{|module\.exports\s*=\s*\{)([^}]*)\}", re.MULTILINE
)
_NAME_PATTERN = re.compile(r"[A-Za-z_$][\w$]*")
# A line that is part of a file's export surface rather than a use of what it exports, and
# the subset of those that open a brace list continuing over the lines that follow.
_EXPORT_SURFACE_LINE = re.compile(r"^(?:export\b|module\.exports\b|exports\.|__all__\b)")
_EXPORT_SURFACE_OPENS = re.compile(r"^(?:export\s*\{|module\.exports\s*=\s*\{|__all__\s*=)")


@dataclass(frozen=True, slots=True)
class ChangedPaths:
    """What one uncommitted change did to the checkout, read from `git status` once.

    ``added`` and ``modified`` are the production files this inspection reasons about, and
    they are what this module has always computed. ``touched`` is every path the status
    named, whatever its class and whatever happened to it -- deletions and both sides of a
    rename included -- and it exists for the callers that need to answer "did this change
    leave that file alone?" rather than "what did this change add?". A deletion is not a
    file left alone, and a class filter is not the right instrument for that question.
    """

    added: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    touched: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ReachabilityOutcome:
    """Everything one reachability inspection found, including what it did not look at.

    ``dropped_candidates`` exists because a silent cap is indistinguishable from a clean
    result. A change adding more new modules than ``_MAX_CANDIDATES`` used to leave the rest
    unchecked and log nothing, so the run read as "wiring passed"; anything recording this
    outcome can now say how much of the change was actually asked about.
    """

    issues: tuple[str, ...] = ()
    repairs: tuple[dict[str, str], ...] = ()
    dropped_candidates: tuple[str, ...] = ()
    examined_candidates: tuple[str, ...] = ()
    # Added files the gate passed whose every reference is textual -- quoted, or a name that
    # is declared and exported but never run. The gate still passes (a path in a route table
    # or a settings module is legitimate wiring in many repositories); the fact is recorded so
    # it reaches the operator instead of being invisible. AB-Feature-216's `server/conftest.py`
    # was "wired" by `const pytest_collect_file = 'server/conftest.py::pytest_collect_file'`
    # in the OAuth model -- a declaration and a string, executing nothing. Before this could
    # ever block, it needs a measurement of how often textual is the *correct* wiring in real
    # repositories.
    textual_wirings: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One thing this change added, and the names by which the checkout would refer to it.

    ``origin`` distinguishes a whole new file from a new export in an existing one. They get
    different diagnostics because they have different remedies: a new file needs a host, and
    a new export needs a caller in a file that is already reached.
    """

    path: str
    symbols: tuple[str, ...]
    origin: str
    display_symbol: str = ""
    added_names: tuple[str, ...] = ()


class ReachabilityChecker(Protocol):
    """Ask one workspace whether the code a change added to it can be reached."""

    async def issues(
        self, workspace_root: PathLike, *, assigned_paths: Sequence[str] = ()
    ) -> ReachabilityOutcome:
        """Return what cannot be reached; never raise for an ordinary inspection failure."""


class NullReachabilityChecker:
    """Inspect nothing, which is what a composition with no configured checker must do."""

    async def issues(
        self, workspace_root: PathLike, *, assigned_paths: Sequence[str] = ()
    ) -> ReachabilityOutcome:
        """Report that no reachability inspection ran here."""
        del workspace_root, assigned_paths
        return ReachabilityOutcome()


class RepositoryReachabilityChecker:
    """Run the deterministic reachability inspection against a real checkout.

    Costs no model call at all, which is what makes this the cheapest part of the Engineer's
    in-attempt verification and the one with the best value-to-risk ratio: `git status`,
    `git grep` and `git ls-files` over a local checkout, and nothing that can hallucinate.
    """

    def __init__(
        self,
        *,
        process_runner: ProcessRunner,
        cancellation_token: CancellationToken | None = None,
        timeout_seconds: float,
        excluded_paths: Sequence[str] = (),
    ) -> None:
        """Bind the execution boundaries; nothing here reaches beyond the workspace."""
        self._process_runner = process_runner
        self._cancellation_token = cancellation_token or MockCancellationToken()
        self._timeout = timeout_seconds
        self._excluded_paths = tuple(excluded_paths)

    async def issues(
        self, workspace_root: PathLike, *, assigned_paths: Sequence[str] = ()
    ) -> ReachabilityOutcome:
        """Inspect this workspace and report what the running application cannot reach."""
        return await unreachable_addition_issues(
            Path(workspace_root),
            runner=self._process_runner,
            timeout=self._timeout,
            cancellation_token=self._cancellation_token,
            assigned_paths=tuple(assigned_paths),
            excluded_paths=self._excluded_paths,
        )


async def unreachable_addition_issues(
    workspace: Path,
    *,
    runner: ProcessRunner,
    timeout: float,
    cancellation_token: CancellationToken,
    assigned_paths: Sequence[str] = (),
    excluded_paths: Sequence[str] = (),
) -> ReachabilityOutcome:
    """Report the production code this change added that nothing in the repository reaches.

    ``assigned_paths`` is the plan's own ``expected_files_or_areas`` for this workstream. It
    is consulted first, because it is the only input here that knows what the change is
    *for*. Everything else reasons structurally -- directory adjacency and reference counting
    -- which answers "what could host this" and not "what should". AB-Feature-166's console
    was assigned `src/pages/Profile.js` nine times over in its plan and told three times to
    wire into `src/components/Header.js`, a file the plan never mentions, because the
    structural answer was the only one this gate had.

    Feature -048's frontend added its tile and tests six times and never edited the homepage
    renderer, so every attempt shipped code the application could not reach.

    Never raises for an inspection that could not run. A failed `git` command returns nothing
    to report, because blocking a change on the inspection's own failure would replace a
    wiring defect with a platform one -- and the reviewer and the repository's own validation
    still stand between this and a commit.
    """
    changed = await changed_paths(runner, workspace, timeout, cancellation_token, excluded_paths)
    added, modified = list(changed.added), list(changed.modified)
    candidates = [
        *(_added_candidate(workspace, path) for path in added),
        *await _modified_export_candidates(
            runner, workspace, timeout, cancellation_token, modified, added
        ),
    ]
    examined = candidates[:_MAX_CANDIDATES]
    dropped = candidates[_MAX_CANDIDATES:]
    if dropped:
        # Logged in the shared implementation so both layers say it, and named rather than
        # counted: "three candidates were dropped" cannot be acted on and "these three
        # files were never asked about" can. 46- calls a silent cap out by name and this is
        # exactly the kind it means.
        _LOGGER.warning(
            "the reachability inspection examined %d of %d candidates and never asked about "
            "%s [outcome=reachability_candidates_dropped]",
            len(examined),
            len(candidates),
            ", ".join(item.path for item in dropped),
        )
    issues: list[str] = []
    repairs: list[dict[str, str]] = []
    textual_wirings: list[dict[str, str]] = []
    for candidate in examined:
        await cancellation_token.raise_if_cancelled()
        if candidate.origin == "added_file":
            await _report_added_file(
                runner,
                workspace,
                timeout,
                cancellation_token,
                candidate,
                added,
                assigned_paths,
                issues,
                repairs,
                textual_wirings,
            )
        else:
            await _report_added_exports(
                runner,
                workspace,
                timeout,
                cancellation_token,
                candidate,
                assigned_paths,
                issues,
                repairs,
            )
    return ReachabilityOutcome(
        issues=tuple(issues),
        repairs=tuple(repairs),
        dropped_candidates=tuple(item.path for item in dropped),
        examined_candidates=tuple(item.path for item in examined),
        textual_wirings=tuple(textual_wirings),
    )


def reachability_diagnostic(issue: str) -> str:
    """Tag one issue so the Engineer's repair loop can recognise it as this check speaking.

    The tag is prepended rather than woven in, so the sentence the runtime's authoritative
    gate reports and the sentence the in-attempt repair is shown are the same words. A repair
    told what rejected its work looks for the defect in the right place.
    """
    return f"{REACHABILITY_DIAGNOSTIC_PREFIX}.\n{issue}"


def is_reachability_diagnostic(diagnostic: str) -> bool:
    """Say whether one diagnostic is this module's, and therefore about wiring."""
    return diagnostic.startswith(REACHABILITY_DIAGNOSTIC_PREFIX)


# --------------------------------------------------------------------------------------
# What this change added
# --------------------------------------------------------------------------------------


async def changed_paths(
    runner: ProcessRunner,
    workspace: Path,
    timeout: float,
    cancellation_token: CancellationToken,
    excluded_paths: Sequence[str] = (),
) -> ChangedPaths:
    """Return what this uncommitted change did to the checkout, in one `git status`.

    Porcelain rather than `git diff`, which never lists an untracked file: nothing is staged
    at this point, so a newly written module appears only as `??`. `-uall` so a wholly new
    directory is listed as its files -- the default collapses it to `server/handlers/`, which
    is not a module and has no neighbours.

    Public because a second in-attempt check now needs the same answer, and "what did this
    change touch?" is one judgement about one `git status` output. Two parsers of the same
    porcelain would drift apart the first time either is corrected.
    """
    captured = await runner.run(
        (
            "git",
            "status",
            "--porcelain",
            "-uall",
            "--",
            ".",
            *(f":(exclude){path}" for path in excluded_paths),
        ),
        workspace,
        timeout,
        cancellation_token,
        repository_subprocess_environment(),
    )
    if captured.return_code != 0:
        return ChangedPaths()
    added: list[str] = []
    modified: list[str] = []
    touched: set[str] = set()
    for line in captured.stdout.splitlines():
        status, path = line[:2], line[3:].strip()
        if not path:
            continue
        # Both sides of a rename, and no class filter at all: this set answers "was that file
        # left alone", where a renamed-away file and a deleted one are both answers of "no".
        touched.update(part.strip() for part in path.split(" -> ") if part.strip())
        if classify_file_change(path) != "production":
            continue
        if status in {"??", "A ", "AM", " A"}:
            # A file with no extension is a LICENSE or a script, not an importable module.
            if Path(path).suffix and len(Path(path).stem) >= _MIN_SYMBOL_LENGTH:
                added.append(path)
        elif "M" in status or "R" in status:
            # Every shape of edit, staged or not. A deletion is deliberately neither: a file
            # this change removed has no surface left to be unreachable.
            modified.append(path)
    return ChangedPaths(added=tuple(added), modified=tuple(modified), touched=frozenset(touched))


def _added_candidate(workspace: Path, path: str) -> _Candidate:
    """Describe one added file by every name the checkout could refer to it under."""
    symbols = candidate_symbols(workspace, path)
    return _Candidate(path=path, symbols=symbols, origin="added_file", display_symbol=symbols[0])


async def _modified_export_candidates(
    runner: ProcessRunner,
    workspace: Path,
    timeout: float,
    cancellation_token: CancellationToken,
    modified: Sequence[str],
    added: Sequence[str],
) -> list[_Candidate]:
    """Return the existing files this change gave a new public name that nothing calls.

    The blind spot this closes: everything above reasons about *files*, so an export added to
    a file that already existed was invisible to the whole gate. A helper exported for a
    caller that was never written is the same unfinished work as a component nobody renders,
    and it reads as "wiring passed".

    HEAD is asked what the file used to export rather than the diff being parsed, because a
    declaration wrapped across lines defeats a line-oriented reading of `+` lines and the
    comparison of two whole surfaces does not care how either was formatted.

    **Only a file something else in the checkout imports is asked.** This is the same "ask
    the checkout" reasoning the neighbour search uses one level up, and it is what keeps this
    from firing on the two shapes where an unreferenced export is perfectly correct:

    * a **package entry point**. `src/index.js` is imported by whatever consumes this
      repository, which is outside it, so its export surface is reachable by definition and
      no amount of grepping can establish otherwise. Nothing here names it, so it is skipped.
    * a **convention-loaded module** -- a Flask blueprint, a framework page directory. It runs
      without anything importing it, so its surface is not what makes its code run.

    The test is the file's own stem, because a reference to a *file* is written as a path. A
    stem too short to mean anything is treated as no evidence, which skips the file.

    One deliberate laxness, stated so it is not mistaken for an oversight: a name this file
    merely *re-exports* from another module is found in the module that declares it, so it is
    not reported. That is the right answer -- a re-export adds no code that could be dead --
    and it is why the check is about names rather than about lines.
    """
    candidates: list[_Candidate] = []
    for path in list(modified)[:_MAX_MODIFIED_FILES]:
        await cancellation_token.raise_if_cancelled()
        current = _read_source(workspace / path)
        if current is None:
            continue
        previous = await _committed_source(runner, workspace, timeout, cancellation_token, path)
        if previous is None:
            continue
        before = set(exported_names(previous))
        after = exported_names(current)
        new_names = tuple(name for name in after if name not in before)
        if not new_names:
            continue
        stem = PurePosixPath(path).stem
        if len(stem) < _MIN_SYMBOL_LENGTH:
            continue
        importers = await _referencing_paths(
            runner, workspace, timeout, cancellation_token, path, (stem,)
        )
        if not importers:
            continue
        usable = tuple(name for name in new_names if _is_usable_symbol(name))
        if not usable:
            continue
        candidates.append(
            _Candidate(
                path=path,
                symbols=usable,
                origin="added_export",
                display_symbol=usable[0],
                added_names=usable,
            )
        )
    return candidates


# --------------------------------------------------------------------------------------
# The names a checkout would use
# --------------------------------------------------------------------------------------


def candidate_symbols(workspace: Path, path: str) -> tuple[str, ...]:
    """Return the names a reference to this module could plausibly be written as.

    The file's own exported names first, then its filename stem. Both are needed and neither
    is sufficient: a reference is written either as a symbol (`import { formatCurrency }`) or
    as a path (`require('./utils/helpers')`), and only the stem appears in the second. The
    stem alone is what checked `src/pages/ServerHealthHistory.js` against a token that
    `getServerHealthHistory` happened to contain, so -072 and -073 both shipped a page
    nothing rendered while this check stayed silent.

    The stem is always kept, so this can only ever add names. That direction is chosen
    deliberately: an extra name makes the check quieter, and a wrong instruction to wire
    something already wired costs a whole attempt.
    """
    stem = PurePosixPath(path).stem
    source = _read_source(workspace / path)
    names = [name for name in exported_names(source or "") if _is_usable_symbol(name)]
    ordered = [*names, stem] if stem not in names else [stem, *(n for n in names if n != stem)]
    return tuple(dict.fromkeys(ordered))


def module_specifiers(source: str) -> tuple[str, ...]:
    """Return every module specifier this source imports, in first-seen order.

    Textual and language-neutral like `exported_names`, and with the same contract: it does
    not need to be complete, and it must not be wrong, so nothing is inferred from what a
    specifier looks like -- resolving one is the caller's question.
    """
    found: list[str] = []
    for match in MODULE_SPECIFIER.finditer(source):
        specifier = match.group("req") or match.group("js") or match.group("py")
        if specifier and specifier not in found:
            found.append(specifier)
    return tuple(found)


def exported_names(source: str) -> tuple[str, ...]:
    """Return the public names this source declares, in first-seen order.

    Textual and language-neutral on purpose. This does not need to be complete -- an export
    it misses falls back to the filename stem, which is where this check stood before -- and
    it must not be wrong, so every pattern is anchored to the start of a line and nothing is
    inferred from what a name looks like.
    """
    found: list[str] = []
    for pattern in _EXPORT_PATTERNS:
        found.extend(match.group(1) for match in pattern.finditer(source))
    for match in _EXPORT_LIST_PATTERN.finditer(source):
        # Quoted text first: `module.exports = { greet: () => 'hello world' }` would
        # otherwise contribute `hello` and `world` as names of this module's public surface,
        # and every extra name makes this check quieter about a genuinely dead module.
        found.extend(_NAME_PATTERN.findall(_without_quoted_text(match.group(1))))
    return tuple(
        name for name in dict.fromkeys(found) if name not in _IMPORT_KEYWORDS and name != "exports"
    )


def _is_usable_symbol(name: str) -> bool:
    """Say whether a word-boundary grep for this name would be evidence of anything."""
    if len(name) < _MIN_SYMBOL_LENGTH or name.startswith("_"):
        return False
    return name.lower() not in GENERIC_SYMBOL_NAMES


# --------------------------------------------------------------------------------------
# The two shapes of diagnostic
# --------------------------------------------------------------------------------------


def _wired_only_textually(workspace: Path, referrer: str, symbols: Sequence[str]) -> bool:
    """Say whether this referrer reaches the candidate only in text, never in execution.

    A symbol mention is textual when it sits inside a string literal, or when the file
    declares or exports a name and never runs it -- `_uses_name_beyond_its_declaration`'s
    exact question, reused rather than re-derived. Both shapes satisfied the gate in
    AB-Feature-216: quote-stripping alone would not have seen the second, because the const
    the agent wrote was itself named after the symbol.

    Unreadable answers False -- "not established as textual" -- so an unreadable referrer
    records nothing rather than a claim about evidence nobody saw.
    """
    source = _read_source(workspace / referrer)
    if source is None:
        return False
    unquoted = _without_quoted_text(source)
    for name in symbols:
        if not re.search(rf"\b{re.escape(name)}\b", unquoted):
            continue
        if _uses_name_beyond_its_declaration(workspace, referrer, name):
            return False
    return True


async def _report_added_file(
    runner: ProcessRunner,
    workspace: Path,
    timeout: float,
    cancellation_token: CancellationToken,
    candidate: _Candidate,
    added: Sequence[str],
    assigned_paths: Sequence[str],
    issues: list[str],
    repairs: list[dict[str, str]],
    textual_wirings: list[dict[str, str]],
) -> None:
    """Report one added module nothing reaches, naming the file that has to reach it."""
    path, symbol = candidate.path, candidate.display_symbol
    referrers = await _referencing_paths(
        runner, workspace, timeout, cancellation_token, path, candidate.symbols
    )
    if referrers:
        # Named somewhere, which the earliest version of this check accepted as wired. An
        # import satisfies that and renders nothing: chk-1 imported its tile into the
        # homepage renderer and never put it in the returned markup, and the gate passed a
        # change the reviewer then rejected for exactly that.
        satisfied = [
            referrer
            for referrer in referrers
            if file_uses_symbol(workspace, referrer, candidate.symbols)
        ]
        if satisfied:
            # Passed -- and when every satisfying reference turns out to be textual, that
            # fact is recorded rather than swallowed. See ReachabilityOutcome.
            for referrer in satisfied:
                if _wired_only_textually(workspace, referrer, candidate.symbols):
                    textual_wirings.append(
                        {
                            "added_path": path,
                            "referrer": referrer,
                            "symbol": symbol,
                            "wiring_reference_kind": "textual_only",
                        }
                    )
            return
        target = referrers[0]
        issues.append(
            f"{target} imports {symbol} from {path} but never uses it, so the "
            "running application still cannot reach it. An import is a declaration, "
            f"not a use: {symbol} has to appear in what {target} actually renders, "
            "registers or returns."
        )
        repairs.append(
            {
                "target_path": target,
                "symbol": symbol,
                "added_path": path,
                "problem": "imported_but_unused",
            }
        )
        return
    assigned = await _assigned_wiring_target(
        runner, workspace, timeout, cancellation_token, path, added, assigned_paths
    )
    if assigned is not None:
        # The plan already answered this. Naming its file is not a stronger guess than the
        # structural one -- it is a different kind of answer, from the only input that knows
        # what the change is for.
        issues.append(
            f"{path} was added but no production file refers to it, so the running "
            f"application cannot reach it. This workstream's plan assigns it "
            f"{assigned}, so reference {path} from there, using an `edits` entry so "
            "the rest of that file is untouched. If that file is genuinely the wrong "
            "host, reference it from whichever existing file this change is actually "
            "for -- but leaving it unreachable is the one outcome that cannot be "
            "accepted."
        )
        repairs.append(
            {
                "target_path": assigned,
                "symbol": symbol,
                "added_path": path,
                "problem": "never_referenced",
                "target_source": "workstream_plan",
            }
        )
        return
    wiring = await _neighbour_wiring_point(
        runner, workspace, timeout, cancellation_token, path, added
    )
    if wiring is None:
        return
    neighbour, referrer, referrer_registers = wiring
    # Names one file to edit. Naming two, with the sibling first, is how -070's console both
    # registered its route correctly and rendered the new page inside the neighbouring one --
    # reachable by anyone holding the neighbour's permission, which the reviewer caught as a
    # privilege bypass.
    if referrer_registers:
        issues.append(
            f"{path} was added but no production file refers to it, so the running "
            f"application cannot reach it. Edit {referrer}, which is where this "
            f"repository registers modules of this kind: it already reaches the "
            f"neighbouring {neighbour}. Add the same kind of reference for {path} "
            "there, using an `edits` entry so the rest of that file is untouched. Do "
            f"not add the reference to {neighbour} itself, which would make {path} "
            "reachable only through an unrelated module."
        )
    else:
        # No registry exists here, so there is no file this repository mounts modules in and
        # none can be named as one. Saying so, and offering the consumer as an example rather
        # than an instruction, leaves the decision where the evidence is: with whatever screen
        # or flow the change is actually for.
        issues.append(
            f"{path} was added but no production file refers to it, so the running "
            "application cannot reach it. This repository has no module registry to "
            f"add it to, so there is no single correct answer: the one file reaching "
            f"the neighbouring {neighbour} is {referrer}, which is the strongest "
            f"candidate if this change belongs to that screen or flow. Reference "
            f"{path} from there, or from whichever existing file this change is "
            "actually for, using an `edits` entry so the rest of that file is "
            "untouched. Adding a reference nowhere leaves the code unreachable, "
            "which is the one outcome that cannot be accepted."
        )
    repairs.append(
        {
            "target_path": referrer,
            "example_path": neighbour,
            "symbol": symbol,
            "added_path": path,
            "problem": "never_referenced",
        }
    )


async def _report_added_exports(
    runner: ProcessRunner,
    workspace: Path,
    timeout: float,
    cancellation_token: CancellationToken,
    candidate: _Candidate,
    assigned_paths: Sequence[str],
    issues: list[str],
    repairs: list[dict[str, str]],
) -> None:
    """Report new public names in an existing file that nothing in production calls.

    No host is guessed at structurally here. The file already exists and is already reached,
    so the neighbour search's premise -- a brand-new module in a hand-wired directory -- does
    not hold, and naming an arbitrary consumer as "where this belongs" is what told
    AB-Feature-112's console twelve times to mount bulk deletion in a modal for adding an
    app. The plan's own assignment is used when it named a file, and otherwise the diagnostic
    says plainly that it has no single correct answer.
    """
    path = candidate.path
    unreferenced: list[str] = []
    for name in candidate.added_names:
        await cancellation_token.raise_if_cancelled()
        referrers = await _referencing_paths(
            runner, workspace, timeout, cancellation_token, path, (name,)
        )
        if referrers:
            continue
        if _uses_name_beyond_its_declaration(workspace, path, name):
            # Exported and also called inside its own already-reached file. The export
            # keyword may be pointless but the code runs, and this check reports code that
            # cannot run.
            continue
        unreferenced.append(name)
    if not unreferenced:
        return
    named = ", ".join(unreferenced)
    assigned = await _assigned_wiring_target(
        runner, workspace, timeout, cancellation_token, path, (), assigned_paths
    )
    where = (
        f"This workstream's plan assigns it {assigned}, so call {named} from there"
        if assigned is not None
        else f"Call {named} from whichever existing file this change is actually for"
    )
    issues.append(
        f"{path} now exports {named}, and no production file in this repository calls "
        f"{'them' if len(unreferenced) > 1 else 'it'}, so the running application cannot "
        f"reach the code this change added there. {where}, using an `edits` entry so the "
        "rest of that file is untouched. A name exported for a caller that was never "
        "written is unfinished work, not a finished file."
    )
    repairs.append(
        {
            # No `target_path` unless the plan named one. Every other repair shape here has a
            # file it can honestly name; this one frequently does not, and an empty or guessed
            # value would be force-included into the next attempt's snapshot as though it were
            # the answer -- the AB-Feature-112 failure, where a guess stated as fact cost
            # twelve review cycles. The file holding the export is force-included anyway, as a
            # file a prior attempt changed.
            **({"target_path": assigned, "target_source": "workstream_plan"} if assigned else {}),
            "symbol": unreferenced[0],
            "added_path": path,
            "problem": "exported_but_unreferenced",
        }
    )


# --------------------------------------------------------------------------------------
# The checkout's own answers
# --------------------------------------------------------------------------------------


async def _referencing_paths(
    runner: ProcessRunner,
    workspace: Path,
    timeout: float,
    cancellation_token: CancellationToken,
    path: str,
    symbols: Sequence[str],
) -> list[str]:
    """Return the production files that mention any of these names.

    Naming them matters: a diagnostic that says only "nothing refers to it" leaves the model
    to guess where the wiring belongs, and -052b answered that guess by repeating the same
    change until the identical-diagnostic rule ended the workstream.

    Only production source counts. A component whose readers are the spec written beside it
    and a page of documentation describing it still cannot be reached when the application
    runs, and that is exactly the shape a workstream produces when it builds a component and
    never mounts it: -050 added the tile, its test and `docs/`, and the documentation alone
    was enough to make the module look wired.

    Data files are excluded for the same reason, because the same shape returned in a form
    the documentation rule does not cover. `openapi.yaml` is generated by the platform from
    the approved contract -- the engineer is forbidden to change it -- and it classifies as
    production source, so naming the new component there was enough to make -072's console
    look reachable. This check stayed silent, its registry hint was never produced, and three
    attempts were told to register a page without being told where.

    Every name is asked in one `git grep`, because the alternative is one process per symbol
    over the whole checkout.
    """
    patterns = [symbol for symbol in dict.fromkeys(symbols) if symbol]
    if not patterns:
        return []
    await cancellation_token.raise_if_cancelled()
    captured = await runner.run(
        # `-w`, because a substring of a longer identifier is not a reference to this module.
        # `src/apiUtils/home.apiUtils.js` defines `getServerHealthHistory`, which contains the
        # stem of the page `ServerHealthHistory.js`, so the page looked referenced and this
        # check stayed silent through -072 and -073 while nothing rendered it. -070 escaped
        # only because its helper happened to start lowercase and this match is case-sensitive.
        # Both properties are load-bearing; neither may be relaxed to catch more.
        (
            "git",
            "grep",
            "--untracked",
            "-l",
            "-F",
            "-w",
            *(argument for symbol in patterns for argument in ("-e", symbol)),
            "--",
        ),
        workspace,
        timeout,
        cancellation_token,
        repository_subprocess_environment(),
    )
    # git grep exits non-zero when nothing matched, which is the answer, not a failure.
    return [
        candidate
        for line in captured.stdout.splitlines()
        if (candidate := line.strip())
        and candidate not in {"", path}
        and classify_file_change(candidate) == "production"
        and PurePosixPath(candidate).suffix.lower() not in _DESCRIBES_WITHOUT_REACHING
    ]


async def _assigned_wiring_target(
    runner: ProcessRunner,
    workspace: Path,
    timeout: float,
    cancellation_token: CancellationToken,
    path: str,
    added: Sequence[str],
    assigned_paths: Sequence[str],
) -> str | None:
    """Return the production file this workstream's plan assigns, where it named one.

    Only a real file in the checkout counts. ``expected_files_or_areas`` mixes files with
    directories -- `src/apiUtils`, `src` -- and a directory is not somewhere a reference can
    be added. The path under inspection is excluded for the reason the neighbour search
    excludes it: referencing a module from itself cannot make it reachable.
    """
    if not assigned_paths:
        return None
    tracked = await tracked_paths(runner, workspace, timeout, cancellation_token)
    for candidate in assigned_paths:
        normalized = candidate.strip().lstrip("./")
        if (
            normalized
            and normalized not in added
            and normalized != path
            and normalized in tracked
            and classify_file_change(normalized) == "production"
        ):
            return normalized
    return None


async def tracked_paths(
    runner: ProcessRunner, workspace: Path, timeout: float, cancellation_token: CancellationToken
) -> frozenset[str]:
    """Every file the checkout tracks, so an assigned path can be confirmed to exist.

    Public for the same reason ``changed_paths`` is: an assigned path is only an assigned
    *file* if the checkout holds it, and both in-attempt checks that read the plan's
    assignment have to resolve it the same way.
    """
    captured = await runner.run(
        ("git", "ls-files"),
        workspace,
        timeout,
        cancellation_token,
        repository_subprocess_environment(),
    )
    if captured.return_code != 0:
        return frozenset()
    return frozenset(line.strip() for line in captured.stdout.splitlines() if line.strip())


async def _neighbour_wiring_point(
    runner: ProcessRunner,
    workspace: Path,
    timeout: float,
    cancellation_token: CancellationToken,
    path: str,
    added: Sequence[str],
) -> tuple[str, str, bool] | None:
    """Return a wired-in neighbour, the file reaching it, and whether that file is a registry.

    The third element is what stops the caller stating a guess as a fact: only a genuine
    registrar is "where this repository registers modules of this kind".
    """
    directory = PurePosixPath(path).parent.as_posix()
    captured = await runner.run(
        ("git", "ls-files", "--", directory),
        workspace,
        timeout,
        cancellation_token,
        repository_subprocess_environment(),
    )
    if captured.return_code != 0:
        return None
    neighbours = [
        candidate
        for line in captured.stdout.splitlines()
        if (candidate := line.strip())
        and candidate not in added
        and candidate != path
        and classify_file_change(candidate) == "production"
        and Path(candidate).suffix
        and len(Path(candidate).stem) >= _MIN_SYMBOL_LENGTH
    ]
    # Prefer a referrer that is where modules get mounted over the first file that merely
    # mentions the neighbour. Taking the first match told -069's console that a page is wired
    # in by `src/apiUtils/activities.apiUtils.js`, so it rendered the new page inside an
    # unrelated one; the reviewer rejected that, the attempt reverted, and the same hint sent
    # it back. Two attempts alternated between those states. The registry it needed --
    # `src/utils/routeUtils.js` -- was in the same list, further down.
    # The third element says whether the referrer actually registers modules. Without it the
    # caller described every answer as "where this repository registers modules of this kind"
    # -- true of a registrar, and false of the fallback below. AB-Feature-112's console was
    # told twelve times to wire bulk deletion into `AddAppModal.js`, a modal for adding an
    # app, which qualified only by importing a neighbouring utility. It spent every review
    # cycle following an instruction stated as fact.
    fallback: tuple[str, str, bool] | None = None
    for neighbour in neighbours[:_MAX_NEIGHBOURS]:
        # This attempt's own new files are excluded from the referrers, as they already are
        # from the neighbours. Both halves of the answer are claims about what the repository
        # *already* does -- "where this repository registers modules of this kind", "the one
        # file reaching the neighbouring X" -- and a file that did not exist before this
        # attempt cannot be either.
        #
        # Only the neighbours were filtered once, so the added file could be returned as the
        # place to wire itself in. A Python service was told "Edit app/status_feed.py, which
        # is where this repository registers modules of this kind: it already reaches the
        # neighbouring app/status.py" -- about the file the same sentence had just called
        # unreachable, which had imported nothing. It qualified only by containing the token
        # `status`. Four attempts followed that instruction and the module was no more
        # reachable at the end than at the start, because referencing a module from itself
        # cannot make it reachable.
        referrers = [
            item
            for item in await _referencing_paths(
                runner,
                workspace,
                timeout,
                cancellation_token,
                neighbour,
                candidate_symbols(workspace, neighbour),
            )
            if item not in added
        ]
        if not referrers:
            continue
        registrar = next((item for item in referrers if registers_modules(item)), None)
        if registrar is not None:
            return neighbour, registrar, True
        fallback = fallback or (neighbour, referrers[0], False)
    return fallback


async def _committed_source(
    runner: ProcessRunner,
    workspace: Path,
    timeout: float,
    cancellation_token: CancellationToken,
    path: str,
) -> str | None:
    """Return this file's content at HEAD, or None when HEAD does not hold it as text."""
    captured = await runner.run(
        ("git", "show", f"HEAD:{path}"),
        workspace,
        timeout,
        cancellation_token,
        repository_subprocess_environment(),
    )
    if captured.return_code != 0:
        return None
    return captured.stdout


def file_uses_symbol(workspace: Path, path: str, symbols: Sequence[str]) -> bool:
    """Return whether a file uses a module somewhere other than the line importing it.

    A module can be imported and never rendered, registered or called, which reads as wired
    to anything that only greps for its name.

    What a file calls the module is rarely its file name. A default import binds a local name
    of the importer's choosing, so the module's own stem appears only inside the quoted path
    -- fix-1's route imported `admin.serverStatus.controller` and used it as
    `serverStatusController`, and comparing stems alone reported a correctly wired route as
    dead six times. The bindings introduced by the importing line are what must be looked for
    in the body.
    """
    try:
        content = (workspace / path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        # Unreadable is not evidence of a defect; treat it as used so nothing is blocked.
        return True
    import_lines: list[str] = []
    body_lines: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) or "require(" in stripped:
            import_lines.append(stripped)
        else:
            body_lines.append(line)
    body = "\n".join(body_lines)
    if any(symbol in body for symbol in symbols):
        return True
    bindings: set[str] = set()
    for line in import_lines:
        if any(symbol in line for symbol in symbols):
            bindings |= import_bindings(line)
    return any(re.search(rf"\b{re.escape(name)}\b", body) for name in bindings)


def _uses_name_beyond_its_declaration(workspace: Path, path: str, name: str) -> bool:
    """Say whether an already-reached file runs the code behind a name it newly exports.

    Counting occurrences will not do it. `function formatStatus() {}` beside
    `module.exports = { statusRoute, formatStatus }` mentions the name twice and calls it
    never, so a naive count reports every added export as used and this whole blind spot
    stays open. What counts is a mention that is neither the declaration nor part of the
    export surface -- `router.get('/x', formatStatus)`, a call, a registration.

    A decorated declaration is treated as used, whatever else the file says. `@app.route`
    above a handler *is* the wiring in every framework that offers it, and reporting one
    would be a false positive that costs a real attempt to disprove. This is the one place
    here that knows a convention exists, and it only ever makes this check quieter.

    Unreadable answers "used", like every other read in this module: the check reports on
    evidence and never on the absence of it.
    """
    source = _read_source(workspace / path)
    if source is None:
        return True
    word = re.compile(rf"\b{re.escape(name)}\b")
    declares = re.compile(
        rf"(?:function\s*\*?|class|const|let|var|def|interface|type|enum)\s+{re.escape(name)}\b"
        rf"|^\s*{re.escape(name)}\s*[:=]"
    )
    surface_depth = 0
    previous = ""
    for line in source.splitlines():
        stripped = line.strip()
        inside_surface = surface_depth > 0
        if inside_surface:
            surface_depth += line.count("{") - line.count("}")
        elif _EXPORT_SURFACE_OPENS.match(stripped):
            surface_depth = max(0, line.count("{") - line.count("}"))
        if not word.search(line):
            previous = stripped or previous
            continue
        if declares.search(line):
            if previous.startswith("@"):
                return True
            previous = stripped
            continue
        if inside_surface or _EXPORT_SURFACE_LINE.match(stripped):
            previous = stripped
            continue
        return True
    return False


def import_bindings(line: str) -> set[str]:
    """Return the local names an import line introduces, ignoring the quoted module path.

    Public for the same reason `module_specifiers` is: the seam-mock detector has to know
    what a test file calls the channel package it mocked, and that judgement already lives
    here -- a second copy of it would drift apart silently.
    """
    return set(_NAME_PATTERN.findall(_without_quoted_text(line))) - _IMPORT_KEYWORDS


def resolve_specifier(
    specifier: str, directory: PurePosixPath, existing_files: set[Path]
) -> str | None:
    """Resolve one module specifier against the checkout, or return nothing.

    Public and here rather than private to one agent, for the reason `module_specifiers` is:
    this is the other half of the platform's one import scanner. The engineer's context
    selector reads a change's imports and follows them to files; 87- Part A2 reads the same
    imports to ask which of the change's own modules opens a side-effect channel. Two copies
    of "what file is this specifier" would drift apart silently, and the answer is not
    obvious -- a bare package name resolves to nothing here on purpose, because a dependency
    is not a file in this checkout.
    """
    if specifier.startswith("."):
        base = PurePosixPath(os.path.normpath(str(directory / specifier)))
    elif "." in specifier and "/" not in specifier:
        # A dotted Python module: `app.status_feed` is `app/status_feed.py` in this checkout.
        base = PurePosixPath(specifier.replace(".", "/"))
    else:
        return None
    for suffix in _MODULE_SUFFIXES:
        for candidate in (
            PurePosixPath(f"{base}{suffix}"),
            base / f"index{suffix}" if suffix else base / "index.js",
        ):
            if Path(candidate) in existing_files:
                return candidate.as_posix()
    return None


def _without_quoted_text(text: str) -> str:
    """Blank out single- and double-quoted runs, so a string literal contributes no names."""
    return re.sub(r"""['"][^'"]*['"]""", " ", text)


def _read_source(path: Path) -> str | None:
    """Return one bounded source file's text, or None when it cannot be read as text."""
    try:
        if path.stat().st_size > _MAX_SOURCE_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        return None


__all__ = [
    "GENERIC_SYMBOL_NAMES",
    "MODULE_SPECIFIER",
    "REACHABILITY_DIAGNOSTIC_PREFIX",
    "ChangedPaths",
    "NullReachabilityChecker",
    "ReachabilityChecker",
    "ReachabilityOutcome",
    "RepositoryReachabilityChecker",
    "candidate_symbols",
    "changed_paths",
    "exported_names",
    "file_uses_symbol",
    "import_bindings",
    "is_reachability_diagnostic",
    "module_specifiers",
    "reachability_diagnostic",
    "resolve_specifier",
    "tracked_paths",
    "unreachable_addition_issues",
]
