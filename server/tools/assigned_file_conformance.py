"""Ask whether a change put its code beside the file its plan named instead of in it.

The cheapest wasted attempt in live run 183. The frontend's scoped acceptance criteria named
the exact file -- `bulkCreateApps` exported from `src/apiUtils/allapps.apiUtils.js`, "declared
alongside the existing route constants in that module" -- and the implementation created a
sibling module in the same directory and never opened the file the plan named. A reviewer
cycle, which is a whole outer attempt, to state a fact `git status` already knew.

**Three conditions, all of them, or nothing.** This is deliberately the narrowest check in
this platform, because it is also the only one whose subject is a *judgement someone else
made* -- the plan's -- rather than a fact the checkout states:

1. the plan names an existing file by exact path, resolved against the checkout;
2. the change never touches that file; and
3. the change *adds* a new module of the same kind in the same directory.

Condition 2 alone is routine and must never fire: a plan names files a change legitimately
does not modify all the time. What made 183 wrong was the *sibling creation*, and condition 3
is what pins it. There is no fuzzy matching anywhere here: if `expected_files_or_areas` holds
area strings (`src`, `src/apiUtils`) rather than exact paths, this check simply never fires.

It costs no model call to ask, it decides nothing, and it never fails an attempt -- it emits
one diagnostic into the Engineer's own in-attempt repair channel, the same one the wiring
findings use, and the reviewer remains the authority on whether the placement is actually
wrong. It is the one check in the 49- family with real false-positive risk, so it is
deliberately confined to this module and one seam: dropping it is deleting this file and the
three references to it.

Two `git` invocations of its own, duplicating the ones the reachability inspection runs a
moment earlier. That is chosen: the two checks answer different questions and share no state,
and a local `git status` costs milliseconds against an attempt that costs minutes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Protocol

from services.cancellation import CancellationToken, MockCancellationToken
from services.process_runner import ProcessRunner
from tools.file_tools import PathLike
from tools.implementation_completeness import classify_file_change
from tools.reachability import changed_paths, tracked_paths

_LOGGER = logging.getLogger(__name__)

# How every diagnostic this check composes begins, so the repair prompt can recognise it as
# this check speaking rather than the wiring inspection. The distinction is not cosmetic: a
# wiring finding is told to leave the added code exactly where it is and edit a *host*, and
# this one is about the added code being in the wrong place. A repair given the wrong sentence
# looks for the defect in the wrong file.
ASSIGNED_FILE_DIAGNOSTIC_PREFIX = (
    "This change added a new module beside the file its plan assigned, without touching that file"
)

# How many findings one inspection reports, and how many added siblings one finding names.
# Bounded because a diagnostic is only useful if it can be read, and an attempt that added a
# dozen modules beside a dozen assigned files is a different problem from this one.
_MAX_ISSUES = 2
_MAX_NAMED_SIBLINGS = 3


class AssignedFileChecker(Protocol):
    """Ask one workspace whether a change was written beside its assigned file."""

    async def issues(
        self, workspace_root: PathLike, *, assigned_paths: Sequence[str] = ()
    ) -> tuple[str, ...]:
        """Return what to say about placement; never raise for an inspection failure."""


class NullAssignedFileChecker:
    """Inspect nothing, which is what a composition with no configured checker must do."""

    async def issues(
        self, workspace_root: PathLike, *, assigned_paths: Sequence[str] = ()
    ) -> tuple[str, ...]:
        """Report that no placement inspection ran here."""
        del workspace_root, assigned_paths
        return ()


class RepositoryAssignedFileChecker:
    """Run the deterministic placement inspection against a real checkout.

    Costs no model call: `git status` and `git ls-files` over a local checkout, and nothing
    that can hallucinate. Wired only into the Engineer's in-attempt loop -- deliberately not
    into the runtime's authoritative gate, which decides whether a change may be published
    and must not start refusing one on a placement opinion.
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
    ) -> tuple[str, ...]:
        """Inspect this workspace and report a change written beside its assigned file."""
        return await assigned_file_conformance_issues(
            Path(workspace_root),
            runner=self._process_runner,
            timeout=self._timeout,
            cancellation_token=self._cancellation_token,
            assigned_paths=tuple(assigned_paths),
            excluded_paths=self._excluded_paths,
        )


async def assigned_file_conformance_issues(
    workspace: Path,
    *,
    runner: ProcessRunner,
    timeout: float,
    cancellation_token: CancellationToken,
    assigned_paths: Sequence[str] = (),
    excluded_paths: Sequence[str] = (),
) -> tuple[str, ...]:
    """Report an assigned file this change never opened, beside a module it added instead.

    Never raises for an inspection that could not run, like every other check of this kind
    here: a failed `git` command has nothing to report, and turning the inspection's own
    failure into a finding would replace a placement question with a platform one.
    """
    exact = _exact_assigned_paths(assigned_paths)
    if not exact:
        # Areas only, or nothing at all. There is no path membership to test, and guessing
        # what an area string meant is exactly the fuzzy matching this check refuses.
        return ()
    tracked = await tracked_paths(runner, workspace, timeout, cancellation_token)
    if not tracked:
        return ()
    changed = await changed_paths(runner, workspace, timeout, cancellation_token, excluded_paths)
    if not changed.added:
        # Condition 3 cannot hold: this change added no module anywhere.
        return ()
    await cancellation_token.raise_if_cancelled()
    qualifying = [
        (assigned, siblings)
        for assigned in exact
        if _is_untouched_assigned_file(assigned, tracked, changed.touched)
        and (siblings := _added_siblings(assigned, changed.added))
    ]
    if len(qualifying) > _MAX_ISSUES:
        # Named rather than counted, and logged rather than dropped in silence -- the same
        # discipline the reachability inspection's candidate cap follows.
        _LOGGER.warning(
            "the assigned-file check reported %d of %d findings and never mentioned %s "
            "[outcome=assigned_file_findings_dropped]",
            _MAX_ISSUES,
            len(qualifying),
            ", ".join(assigned for assigned, _siblings in qualifying[_MAX_ISSUES:]),
        )
    return tuple(_issue(assigned, siblings) for assigned, siblings in qualifying[:_MAX_ISSUES])


def assigned_file_diagnostic(issue: str) -> str:
    """Tag one issue so the Engineer's repair loop can recognise which check spoke."""
    return f"{ASSIGNED_FILE_DIAGNOSTIC_PREFIX}.\n{issue}"


def is_assigned_file_diagnostic(diagnostic: str) -> bool:
    """Say whether one diagnostic is this module's, and therefore about placement."""
    return diagnostic.startswith(ASSIGNED_FILE_DIAGNOSTIC_PREFIX)


# --------------------------------------------------------------------------------------
# The three conditions
# --------------------------------------------------------------------------------------


def _exact_assigned_paths(assigned_paths: Sequence[str]) -> tuple[str, ...]:
    """Return the plan's assignments that are exact path tokens, in the plan's own order.

    ``expected_files_or_areas`` mixes files with areas -- `src/pages/Profile.js` beside
    `src/apiUtils` and `src` -- and only a token that spells a file can be tested for
    membership. A suffix is the same discriminator ``_assigned_context_paths`` uses for the
    same list, and nothing here tries to interpret an area: an area string means this check
    does not fire, which is the documented outcome and not a gap.
    """
    exact: list[str] = []
    for candidate in assigned_paths:
        if not isinstance(candidate, str):
            continue
        normalized = candidate.strip().lstrip("./")
        if normalized and PurePosixPath(normalized).suffix:
            exact.append(normalized)
    return tuple(dict.fromkeys(exact))


def _is_untouched_assigned_file(
    assigned: str, tracked: frozenset[str], touched: frozenset[str]
) -> bool:
    """Say whether the plan named this existing production file and the change left it alone.

    Conditions 1 and 2. Membership in the checkout is `git ls-files`, so a plan naming a file
    that does not exist -- a file it expected the change to *create* -- says nothing here.
    Production only, because a plan naming a test file and a change adding a test beside it is
    ordinary suite organisation and not the defect this check is about.
    """
    return (
        assigned in tracked
        and classify_file_change(assigned) == "production"
        and assigned not in touched
    )


def _added_siblings(assigned: str, added: Sequence[str]) -> tuple[str, ...]:
    """Return the modules this change added in the assigned file's own directory.

    Condition 3, and the one that makes this check narrow enough to ship. Same directory, and
    the same file extension: a plan that names a JavaScript module and a change that adds a
    JSON document beside it are not the shape 183 produced, and requiring the suffix to match
    keeps a generated data file or a fixture from ever standing in for a sibling module. Both
    tests are path membership; neither reads a byte of either file.
    """
    directory = PurePosixPath(assigned).parent
    suffix = PurePosixPath(assigned).suffix.lower()
    return tuple(
        path
        for path in added
        if PurePosixPath(path).parent == directory and PurePosixPath(path).suffix.lower() == suffix
    )[:_MAX_NAMED_SIBLINGS]


def _issue(assigned: str, siblings: Sequence[str]) -> str:
    """Compose the finding, naming the assigned file and what was added beside it.

    Deliberately not phrased as a verdict. The plan can be wrong about where code belongs, and
    this check cannot tell -- so the sentence states what it observed, asks for the file the
    plan named to be used if that is where the code belongs, and offers the completion summary
    as the answer if it is not. A pass that answers by explaining itself costs one repair call
    and nothing else, which is what makes a false positive here survivable.

    Nothing asks for the new module to be deleted: a coding response can write files and edit
    them and has no way to remove one, so an instruction to remove it could not be followed.
    """
    named = ", ".join(siblings)
    return (
        f"This workstream's plan assigns {assigned}, a file that already exists in this "
        f"checkout and that this change does not touch. The change adds {named} in the same "
        f"directory instead. If the code the plan asked for belongs in {assigned}, declare it "
        f"there -- as an `edits` entry, so the rest of that file is untouched -- and have the "
        f"rest of this change use it from there. If a separate module beside it is genuinely "
        f"the right shape, leave both files as they stand and say why in your completion "
        f"summary: an unexplained sibling is what review sends back, and the plan named "
        f"{assigned} for a reason someone can read."
    )


__all__ = [
    "ASSIGNED_FILE_DIAGNOSTIC_PREFIX",
    "AssignedFileChecker",
    "NullAssignedFileChecker",
    "RepositoryAssignedFileChecker",
    "assigned_file_conformance_issues",
    "assigned_file_diagnostic",
    "is_assigned_file_diagnostic",
]
