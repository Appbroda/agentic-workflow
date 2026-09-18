"""Report the design's own colours and type families that a change never mentions.

The only deterministic thing this platform can say about "does it match the design". Every
other gate it runs -- build, lint, format, test -- passes identically whether the engineer used
the design's `#0a0a0a` or a colour it invented, because no command any repository ships checks
a colour. A reviewer reading a diff cannot check it either: it has the design as JSON and the
change as text, and matching one against the other by eye across a hundred nodes is exactly the
task models are worst at. So this counts.

**Colours and families only, deliberately.** The rendered design also carries sizes, radii and
spacing, and almost all of them are scaled fractions: measured on AB-Feature-229's login frame,
a real type size is `7.309706687927246` and a real corner radius is `3.654853343963623`. Nobody
writes those in a stylesheet, so requiring them would report every implementation as wrong. A
hex code and a font family are exact, are written literally, and mean the same thing in Figma
and in CSS.

**A miss is a question, not a verdict.** A repository with a design system should express
`#0a0a0a` as a token, and a change that writes `bg-surface-raised` instead of the literal is
*better* than one that hardcodes it -- so a value reported here may be correctly absent. That is
why this returns sentences for the attempt to read rather than a pass/fail, and why it is on the
same footing as the in-attempt reachability check: informative to the role that can act on it,
never able to fail the attempt on its own.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from services.cancellation import CancellationToken
from services.process_runner import ProcessRunner, repository_subprocess_environment
from tools.file_tools import PathLike

# How many absent values are named before the rest are counted. The sentence is read by a model
# with a finite prompt, and forty colours it did not use is not forty pieces of information.
_MAX_REPORTED = 8

# A colour worth asking about. Pure black, pure white and fully transparent are in every design
# and every stylesheet ever written, so their presence proves nothing and their absence means
# nothing -- reporting them is noise that buries the palette that actually identifies this
# design.
_UNREMARKABLE_COLOURS = frozenset({"#000000", "#ffffff", "#00000000", "#ffffffff"})

# Families the design names that no stylesheet has to mention, because they are what a browser
# falls back to anyway.
_UNREMARKABLE_FAMILIES = frozenset({"inherit", "sans-serif", "serif", "monospace"})

_HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")

# How every diagnostic this check composes begins, so the repair prompt can recognise it as
# this check speaking rather than the wiring or placement inspections. Those two are about
# *where* code went; this one is about what it is made of, and a repair told the wrong
# sentence looks for the defect in the wrong file.
DESIGN_VALUE_DIAGNOSTIC_PREFIX = (
    "A deterministic check compared the design this workstream was given against the lines "
    "this change added, and the change does not use some of the design's own values"
)


def design_value_diagnostic(issue: str) -> str:
    """Tag one issue so the Engineer's repair loop can recognise which check spoke."""
    return f"{DESIGN_VALUE_DIAGNOSTIC_PREFIX}.\n{issue}"


def is_design_value_diagnostic(diagnostic: str) -> bool:
    """Say whether one diagnostic is this module's, and therefore about design values."""
    return diagnostic.startswith(DESIGN_VALUE_DIAGNOSTIC_PREFIX)


def design_values(content: Mapping[str, Any]) -> tuple[frozenset[str], frozenset[str]]:
    """Return the distinct colours and type families one rendered frame specifies.

    Walked over the rendering the Engineer was handed rather than over raw Figma, so what is
    counted is exactly what the model could have read: a value cut off at the depth bound is
    not in the content and is therefore never asked about.
    """
    colours: set[str] = set()
    families: set[str] = set()
    for node in _walk(content):
        for paint in _paints(node):
            value = paint.get("hex")
            if isinstance(value, str) and value.strip():
                colours.add(value.strip().lower())
        typography = node.get("typography")
        if isinstance(typography, Mapping):
            family = typography.get("family")
            if isinstance(family, str) and family.strip():
                families.add(family.strip())
    return (
        frozenset(c for c in colours if c not in _UNREMARKABLE_COLOURS),
        frozenset(f for f in families if f.lower() not in _UNREMARKABLE_FAMILIES),
    )


def design_value_issues(
    contents: Sequence[Mapping[str, Any]], *, changed_text: str
) -> tuple[str, ...]:
    """Report the design's colours and families this change mentions nowhere.

    ``changed_text`` is every added line of the change, concatenated -- not the whole file and
    not the whole repository. A colour that was already in the stylesheet before this feature
    began is not evidence that this change used it, and counting it would let an attempt that
    restyled nothing report full coverage.

    Empty for a change that mentions everything, for a design that specifies nothing
    distinctive, and for an empty diff -- a step with nothing to say says nothing, so a feature
    that cites no design produces prompts byte-identical to what they were before this existed.
    """
    colours: set[str] = set()
    families: set[str] = set()
    for content in contents:
        found_colours, found_families = design_values(content)
        colours |= found_colours
        families |= found_families
    if not colours and not families:
        return ()

    present_hex = {value.lower() for value in _HEX.findall(changed_text)}
    lowered = changed_text.lower()
    missing_colours = sorted(c for c in colours if not _colour_present(c, present_hex))
    missing_families = sorted(f for f in families if f.lower() not in lowered)

    issues: list[str] = []
    if missing_colours:
        issues.append(_sentence("colour", missing_colours, len(colours)))
    if missing_families:
        issues.append(_sentence("type family", missing_families, len(families)))
    return tuple(issues)


def _sentence(kind: str, missing: Sequence[str], total: int) -> str:
    """One reportable sentence, naming a bounded sample and counting the rest."""
    named = list(missing[:_MAX_REPORTED])
    remainder = len(missing) - len(named)
    tail = f", and {remainder} more" if remainder > 0 else ""
    return (
        f"The design specifies {total} distinct {kind} value(s) and this change mentions "
        f"{total - len(missing)} of them. Absent: {', '.join(named)}{tail}. If the repository "
        f"already has a token for one of these, use the token and this is nothing to fix; if "
        f"it does not, the implementation is not using the design's own {kind}."
    )


def _colour_present(colour: str, present: Iterable[str]) -> bool:
    """Whether a design colour appears in the change, in any spelling a stylesheet uses.

    `#0a0a0a` and `#0A0A0A` are one colour, and so is the three-digit `#aaa` form of `#aaaaaa`:
    a check that reported a correctly-written shorthand as absent would teach its reader to
    ignore it, which is worse than not running.
    """
    candidates = {colour}
    body = colour.lstrip("#")
    if len(body) == 6 and body[0] == body[1] and body[2] == body[3] and body[4] == body[5]:
        candidates.add(f"#{body[0]}{body[2]}{body[4]}")
    if len(body) == 8:
        candidates.add(f"#{body[:6]}")
    return any(candidate in present for candidate in candidates)


def _paints(node: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Yield every paint on one node: what fills it and what strokes it."""
    for key in ("fills", "strokes"):
        value = node.get(key)
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            continue
        for paint in value:
            if isinstance(paint, Mapping):
                yield paint


def _walk(node: Any) -> Iterator[Mapping[str, Any]]:
    """Yield every rendered node in one frame, the frame itself included."""
    if not isinstance(node, Mapping):
        return
    yield node
    children = node.get("children")
    if isinstance(children, Sequence) and not isinstance(children, str | bytes):
        for child in children:
            yield from _walk(child)


class DesignValueChecker:
    """The design-value inspection, configured once and asked per attempt.

    Shaped like `ReachabilityChecker` and `AssignedFileChecker` deliberately: the Engineer
    injects configured inspections rather than a process runner, so a composition that cannot
    run one simply does not supply it and the attempt proceeds exactly as it did before.
    """

    def __init__(
        self,
        *,
        process_runner: ProcessRunner,
        cancellation_token: CancellationToken | None = None,
        timeout_seconds: float,
    ) -> None:
        """Bind the execution boundaries; nothing here reaches beyond the workspace."""
        from services.cancellation import MockCancellationToken

        self._process_runner = process_runner
        self._cancellation_token = cancellation_token or MockCancellationToken()
        self._timeout = timeout_seconds

    async def issues(
        self, workspace_root: PathLike, *, contents: Sequence[Mapping[str, Any]] = ()
    ) -> tuple[str, ...]:
        """Report the design values this change never mentions, or nothing."""
        if not contents:
            return ()
        added = await added_change_text(
            Path(workspace_root),
            runner=self._process_runner,
            timeout=self._timeout,
            cancellation_token=self._cancellation_token,
        )
        return design_value_issues(contents, changed_text=added)


async def added_change_text(
    workspace: Path,
    *,
    runner: ProcessRunner,
    timeout: float,
    cancellation_token: CancellationToken,
) -> str:
    """Return only the lines this change *added*, across tracked and untracked files.

    The added lines and not the files: a colour already in the stylesheet before this feature
    began is not evidence that this change used it, and counting the whole file would let an
    attempt that restyled nothing report full coverage.

    **Read-only, and that is load-bearing.** Every inspection of this kind here reads the
    workspace and never writes it, because the Engineer's commit gate stages and commits in
    the same checkout: an earlier draft of this function called `git add --intent-to-add` to
    make untracked files visible to `git diff`, which mutates the index and can change what a
    later commit picks up. An inspection that can alter the change it is inspecting is worse
    than no inspection. So tracked edits come from `git diff -U0` -- `-U0` because a context
    line is by definition a line the change did not write -- and untracked files are read
    whole from disk, which is exact rather than approximate: every line of a file this change
    created is a line this change added.

    Empty string for an inspection that could not run, like every other check of this kind: a
    failed `git` command has nothing to report, and turning the inspection's own failure into
    a finding would replace a design question with a platform one.
    """
    added: list[str] = []
    await cancellation_token.raise_if_cancelled()
    try:
        diff = await runner.run(
            ("git", "diff", "-U0"),
            cwd=workspace,
            timeout_seconds=timeout,
            cancellation_token=cancellation_token,
            environment=repository_subprocess_environment(),
        )
        listed = await runner.run(
            ("git", "ls-files", "--others", "--exclude-standard"),
            cwd=workspace,
            timeout_seconds=timeout,
            cancellation_token=cancellation_token,
            environment=repository_subprocess_environment(),
        )
    except Exception:  # noqa: BLE001 - a best-effort inspection reports nothing on failure
        return ""
    if diff.return_code != 0 or listed.return_code != 0:
        return ""
    added.extend(
        line[1:]
        for line in diff.stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    for name in listed.stdout.splitlines():
        candidate = (workspace / name.strip()).resolve()
        if not name.strip() or not _inside(candidate, workspace):
            continue
        try:
            added.append(candidate.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            # A file that cannot be read contributes nothing, exactly as an unreadable file
            # contributes nothing to every other inspection here.
            continue
    return "\n".join(added)


def _inside(candidate: Path, workspace: Path) -> bool:
    """Whether a listed path really is under the workspace, before anything is read.

    `git ls-files` answers about this checkout, so this should never bite -- which is the
    reason to check it rather than the reason to skip it. Nothing here reads a path the
    workspace does not contain.
    """
    try:
        candidate.relative_to(workspace.resolve())
    except ValueError:
        return False
    return True


__all__ = [
    "DESIGN_VALUE_DIAGNOSTIC_PREFIX",
    "DesignValueChecker",
    "added_change_text",
    "design_value_diagnostic",
    "design_value_issues",
    "design_values",
    "is_design_value_diagnostic",
]
