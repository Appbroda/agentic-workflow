"""Recognize the failure that has not moved, when its wording has moved every attempt.

``diagnostic_signature`` reduces a set of blocking issues to the defect tokens they name --
locations, exception names, result codes, linter rules -- and its repeat rule ends a
workstream whose second attempt names the same ones. That is the right identity for a linter
message and the wrong one for a suite:

AB-Feature-183's backend failed ``bulk-create-apps.test.js`` on attempts 1, 2, 3 and 4 with
engineer-authored fixtures omitting columns the product schema derives as required. Sixteen
tests failed, then ten, with different excerpts and different result codes each round, so the
token set changed every attempt while the defect did not. Every guard the loop had read that
as progress. The run was cancelled mid-attempt-4 against a ceiling of eight.

What did not move across those attempts is what this module counts:

    a workspace FILE that the failing evidence keeps naming, attempt after attempt

The file set is ``_workspace_reference_lines``' output -- resolved against the workspace
before any bound was applied, so it survives truncation, which is the property that makes it
usable as an identity at all. Nothing here parses runner output: test names, counts and
failure blocks are runner-specific, and the file set is neither.

The unit is the file and not the pair ``(command, files)``, and that is this module's second
lesson rather than its first. The pair was the original key, and AB-Feature-194 walked
straight through it: its backend failed ``bulk-create-apps.test.js`` on attempts 0, 1, 3, 4
and 5 (attempt 2 was stopped by the assertion guard), under **three different command
strings** -- because ``scoped_test_command``'s file list follows each attempt's changed files,
so the command is rewritten most rounds. Consecutive equality of the pair therefore never
held, neither the x2 line nor the x3 stop ever armed, and the validation ceiling of eight was
the only bound on the loop. AB-Feature-188's stop fired only because its command happened to
stabilize. Counting per file is blind to that variation by construction; the commands are
still carried, because they are what a reader can run, but they no longer gate the count.

Counting per file is deliberately lossier than counting the pair, in the one direction that
matters: any single file named by three consecutive failing attempts stops the workstream,
even if everything else about those failures differed. That is accepted for the reason the
threshold is three and not two -- the x2 line comes first, so the loop always gets one
attempt that has been told in so many words which file is not moving and what the previous
repair touched, before anything is stopped.

This is a second, coarser measure beside ``diagnostic_signature``, not a replacement. The
token signature and its counters keep their behaviour exactly; a failure that is unchanged by
this measure and changing by that one is precisely the case 183 spent four attempts on.

The module reports; it decides nothing and calls no model. Its caller consults the
resolved-issue ledger first, because a design decision being re-litigated is not a repair that
is stuck, and exempts a deterministic gate for the reason the existing repeat rule exempts it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import cast

from pydantic import Field

from artifacts.schemas import ChildWorkflowResultArtifact
from state.models import StateModel
from tools.scoped_tests import ASSERTION_GUARD_OUTCOME, is_scoped_test_diagnostic
from tools.validation_tools import workspace_references_in_summary

# How many consecutive attempts may name the same failing file before the next remediation is
# told in so many words that its strategy did not work, and before the loop stops.
#
# Two and three, matching `_ALLOWED_REPEATED_DIAGNOSTIC_SIGNATURES` and set for the same
# reason: this reduction is lossier than that one, not tighter, so it may not stop a
# workstream any sooner than the finer key would. The first occurrence of any failure buys
# nothing at all -- a repair that has been attempted once has produced no evidence about
# whether repeating it works.
_STRATEGY_CHANGE_ATTEMPTS = 2
_STOP_ATTEMPTS = 3

# How many file and command names a single generated line may carry. A required command's
# argument list and a broad suite's reference set are both unbounded in principle, and these
# lines are spliced into a task description that has a budget.
_MAX_NAMED_ITEMS = 8
_MAX_COMMAND_CHARACTERS = 300

# How the assertion guard's rejection begins, and where a scoped-test diagnostic names the
# command that failed. Both shapes are the producers' own -- `assertion_guard_diagnostic`
# and `scoped_test_diagnostic` -- matched by their imported constants rather than by copies.
_GUARD_PREFIX = f"{ASSERTION_GUARD_OUTCOME}:"
_SCOPED_TEST_COMMAND = re.compile(r"\(command=(.+?); code=")


class UnchangedFailure(StateModel):
    """One or more workspace files the failing evidence has named for several attempts."""

    # The files whose consecutive run is the longest one measured, and the required commands
    # that named them in the attempt being judged. The files are the identity; the commands
    # are context, kept because they are the one thing a reader can go and run -- and kept
    # per attempt rather than compared, since 194's varied every round while the file did not.
    files: tuple[str, ...] = Field(min_length=1)
    commands: tuple[str, ...] = Field(min_length=1)
    # How many consecutive attempts, ending with the one being judged, named those files.
    attempts: int = Field(ge=_STRATEGY_CHANGE_ATTEMPTS)
    # What the attempt that just failed actually changed. The point of the strategy-change
    # line is that these edits were made and made no difference, so a next attempt that reads
    # only "try something else" has been told the smaller half of it.
    changed_files: tuple[str, ...] = ()

    @property
    def stops_the_loop(self) -> bool:
        """Report whether repeating this has stopped being worth an attempt."""
        return self.attempts >= _STOP_ATTEMPTS


def _named(items: Sequence[str]) -> str:
    """Render a bounded list of names for a sentence, saying so when it left some out."""
    shown = list(items[:_MAX_NAMED_ITEMS])
    remainder = len(items) - len(shown)
    listed = ", ".join(shown)
    return f"{listed} and {remainder} more" if remainder > 0 else listed


def _named_commands(commands: Sequence[str]) -> str:
    """Render the failing commands, bounding each one rather than the joined list.

    Bounding the join would cut the last command in half, and a half-quoted command is one a
    reader cannot run -- which is the only thing this text is for.
    """
    return _named(
        [
            command
            if len(command) <= _MAX_COMMAND_CHARACTERS
            else f"{command[:_MAX_COMMAND_CHARACTERS]} [truncated]"
            for command in commands
        ]
    )


def _command_text(command: object) -> str:
    """Return a recorded command as one string, whichever of its two shapes it was stored in.

    The reviewer's summary joins the argument vector before persisting it; other writers of
    this same list have stored the vector itself. Both are in the durable record, and a
    reader that read only one of them would silently never see the other.
    """
    if isinstance(command, str):
        return command.strip()
    if isinstance(command, list) and all(isinstance(item, str) for item in command):
        return " ".join(cast("list[str]", command)).strip()
    return ""


def _failing_required_evidence(
    validations: Sequence[object],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Reduce one attempt's recorded validation to the required commands that rejected it.

    Only required commands, because an optional one's verdict never decides an attempt, and
    only failing ones: a command that passed says nothing about what is unchanged. Sorted by
    command so two attempts that ran the same checks in a different order compare equal.
    """
    evidence: dict[str, tuple[str, ...]] = {}
    for item in validations:
        if not isinstance(item, dict) or not item.get("required") or item.get("passed"):
            continue
        command = _command_text(item.get("command"))
        if not command:
            continue
        files: list[str] = []
        for stream in ("stderr_summary", "stdout_summary"):
            summary = item.get(stream)
            if not isinstance(summary, str):
                continue
            files.extend(
                path for path in workspace_references_in_summary(summary) if path not in files
            )
        merged = evidence.get(command, ())
        evidence[command] = tuple(sorted({*merged, *files}))
    return tuple(sorted(evidence.items()))


def _assertion_guard_evidence(
    result: ChildWorkflowResultArtifact,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Recover the failure identity a refused assertion repair was trying to clear.

    The assertion guard ends its attempt *before* review, so the result carries no validation
    evidence at all -- and no evidence in the middle of a run restarts the consecutive
    counters. 194's sequence was same -> same -> guard -> same: the guard attempt broke the
    chain, the fourth attempt was read as a run of one, and both the x2 strategy line and the
    x3 stop moved out by an attempt -- extra budget for exactly the defect the guard had just
    refused to paper over.

    The underlying failure is still in the durable record. The guard's rejection travels with
    the scoped-test diagnostics it was asked to clear, and each of those names its command in
    its own headline and the workspace files its bounded summary references. So a
    guard-stopped attempt inherits those files instead of resetting their counts.

    Only when the guard actually spoke, and only from the diagnostics' own shapes: an
    ordinary rejection derives nothing here, and a diagnostic whose command or file section
    cannot be read contributes nothing -- no evidence still means unknown, never unchanged.
    """
    issues = tuple(result.blocking_issues)
    if not any(issue.startswith(_GUARD_PREFIX) for issue in issues):
        return ()
    evidence: dict[str, tuple[str, ...]] = {}
    for issue in issues:
        if not is_scoped_test_diagnostic(issue):
            continue
        matched = _SCOPED_TEST_COMMAND.search(issue.splitlines()[0])
        if matched is None:
            continue
        command = matched.group(1).strip()
        if not command:
            continue
        merged = evidence.get(command, ())
        evidence[command] = tuple(sorted({*merged, *workspace_references_in_summary(issue)}))
    return tuple(sorted(evidence.items()))


def coarse_failure_evidence(
    result: ChildWorkflowResultArtifact,
) -> tuple[tuple[str, tuple[str, ...]], ...] | None:
    """Return one attempt's failing required commands and the workspace files they named.

    None means "unknown", never "unchanged" -- the rule ``diagnostic_signature`` already
    states for an empty token set, and it matters more here. An attempt that failed before any
    required command ran, or one whose failing output named no workspace file at all, has
    produced no evidence about which files are stuck, and reading it as agreement with the
    previous attempt would make every such pair a repeat.

    One deliberate exception to "no validation evidence means unknown": an attempt the
    assertion guard stopped inherits the files of the test failure the refused repair was
    asked to clear -- see ``_assertion_guard_evidence`` for why a guard outcome must not
    reset the consecutive counts.
    """
    evidence = _failing_required_evidence(result.current_validation_results)
    if not evidence or not any(files for _command, files in evidence):
        evidence = _assertion_guard_evidence(result)
    if not evidence or not any(files for _command, files in evidence):
        return None
    return evidence


def failing_evidence_files(result: ChildWorkflowResultArtifact) -> frozenset[str] | None:
    """Return the workspace files one attempt's failing evidence named, or None if unknown.

    The counted unit, projected off the recorded evidence. Which command named a file is
    dropped here on purpose: ``scoped_test_command`` rewrites its file list from each
    attempt's changed files, so the command string is the part that moves while the failing
    file is the part that does not.
    """
    evidence = coarse_failure_evidence(result)
    if evidence is None:
        return None
    return frozenset(file for _command, files in evidence for file in files)


def _attempt_changed_files(result: ChildWorkflowResultArtifact) -> tuple[str, ...]:
    """Return the files one attempt actually changed, as the workspace boundary recorded it."""
    categorized = [
        *result.production_files_changed,
        *result.test_files_changed,
        *result.configuration_files_changed,
    ]
    paths = categorized or [change.path for change in result.changed_files]
    return tuple(dict.fromkeys(paths))


def unchanged_failure_run(
    artifacts: Sequence[object],
    *,
    child_workflow_id: str,
    current: ChildWorkflowResultArtifact,
) -> UnchangedFailure | None:
    """Return the longest run of consecutive attempts that kept failing on one file.

    ``artifacts`` is this feature's lineage as it stands when an attempt is being judged, which
    holds every earlier attempt of this workstream and not this one -- so ``current`` is passed
    separately rather than read back out. Derived from persisted artifacts alone: no new state,
    and a crash, a resume or an integration-fix re-entry does not restart the count.

    Each file carries its own count, so a file that recovers resets only itself and leaves a
    sibling that is still failing at whatever run it has earned. The walk back ends at the
    first attempt with no evidence at all, because unknown is not agreement.

    None when the current attempt named no failing file, and None on a first occurrence: a run
    of one is what every failure starts as.
    """
    evidence = coarse_failure_evidence(current)
    if evidence is None:
        return None
    counts = {file: 1 for _command, files in evidence for file in files}
    # Every file the current attempt named starts a run of one; a file drops out of `alive`
    # the moment an earlier attempt did not name it, and stops accumulating from there.
    alive = set(counts)
    earlier = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, ChildWorkflowResultArtifact)
        and artifact.child_workflow_id == child_workflow_id
    ]
    for artifact in reversed(earlier):
        if not alive:
            break
        previous = failing_evidence_files(artifact)
        if previous is None:
            break
        alive &= previous
        for file in alive:
            counts[file] += 1
    attempts = max(counts.values())
    if attempts < _STRATEGY_CHANGE_ATTEMPTS:
        return None
    # Only the files at the longest run are named. A file with a shorter run is not what
    # stopped the workstream, and naming it in the same sentence would send a reader to a
    # file the evidence says less about than the sentence claims.
    stuck = tuple(sorted(file for file, count in counts.items() if count == attempts))
    return UnchangedFailure(
        files=stuck,
        # Only the commands that named a stuck file. A sibling command that failed on
        # something else is not what a reader should be sent to run about these files.
        commands=tuple(
            command for command, files in evidence if any(file in stuck for file in files)
        ),
        attempts=attempts,
        changed_files=_attempt_changed_files(current),
    )


def strategy_change_line(run: UnchangedFailure) -> str:
    """Tell the next attempt that the last repair was made, was real, and changed nothing.

    One line that stands alone, carrying its own status, for the reason every other retry
    feedback line does: these are spliced into individual task descriptions, and a heading
    above the group is a heading some tasks never receive.

    It names the files the previous repair touched rather than only asking for a different
    approach. "Try something else" against an unnamed previous attempt is advice a model can
    satisfy by rewriting the same file differently, which is what the run this line describes
    already consists of.
    """
    changed = _named(run.changed_files) if run.changed_files else "no files this platform recorded"
    return (
        f"UNCHANGED FAILURE -- the previous repair changed {changed} and the same workspace "
        f"files kept failing: {_named(run.files)} appeared in the failing evidence of "
        f"{run.attempts} consecutive attempts, most recently under "
        f"{_named_commands(run.commands)}. A different approach is required -- do not repeat "
        "the previous strategy."
    )


def unchanged_failure_narrative(run: UnchangedFailure) -> tuple[str, list[str]]:
    """State what a person now owns, and the lineage facts that establish it.

    The files are named because they are what the evidence actually pins -- the same ones kept
    failing across attempts that changed the source every round -- and the command because it
    is the one thing a reader can run themselves. The command is quoted as the last attempt's,
    not as a constant: 194's was rewritten most rounds while the failing file was not, which
    is why the count is per file in the first place.

    Deliberately not phrased as a repository defect: nothing here establishes that the
    repository is broken, and an operator sent to repair one finds nothing. What the evidence
    establishes is that further attempts of this kind produce no new information.
    """
    question = (
        f"{_named(run.files)} failed on {run.attempts} consecutive "
        "attempts while the source changed every time. The last of those attempts ran "
        f"{_named_commands(run.commands)}; earlier ones may have named the same files under a "
        "different command. Repeating the repair has stopped producing new information, so "
        "the remaining attempts were not spent. Run that command against those files: is the "
        "requirement achievable here as written, or does the code the attempts keep rewriting "
        "need a person?"
    )
    evidence = [
        f"{run.attempts} consecutive attempts named the same workspace files in their failing "
        f"evidence: {_named(run.files)}.",
        f"The last of them failed on: {_named_commands(run.commands)}.",
    ]
    if run.changed_files:
        evidence.append(f"The last of those attempts changed: {_named(run.changed_files)}.")
    return question, evidence


__all__ = [
    "UnchangedFailure",
    "coarse_failure_evidence",
    "failing_evidence_files",
    "strategy_change_line",
    "unchanged_failure_narrative",
    "unchanged_failure_run",
]
