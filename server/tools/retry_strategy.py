"""Failure-aware child-workstream retry planning and bounded progress checks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from pydantic import Field

from state.models import StateModel

# Re-exported: the extraction moved to its own module so tools.validation_tools (which
# this module imports from) can use it without a cycle. Existing importers keep this path.
from tools.diagnostic_references import diagnostic_file_references as diagnostic_file_references
from tools.implementation_completeness import ImplementationCompletenessResult
from tools.repository_preflight import RepositoryPreflightResult
from tools.validation_tools import (
    CAPACITY_FAILURE_CLASSIFICATIONS,
    ValidationResult,
    ValidationStatus,
)


class FailureClassification(StrEnum):
    """Mutually exclusive causes that consume separate retry budgets."""

    IMPLEMENTATION_MISSING = "implementation_missing"
    VALIDATION_SOURCE_FAILURE = "validation_source_failure"
    VALIDATION_CAPACITY_FAILURE = "validation_capacity_failure"
    VALIDATION_CONFIGURATION_FAILURE = "validation_configuration_failure"
    DEPENDENCY_INSTALLATION_FAILURE = "dependency_installation_failure"
    TEST_INFRASTRUCTURE_MISSING = "test_infrastructure_missing"
    REVIEW_SCOPE_FAILURE = "review_scope_failure"
    CONTRACT_MISMATCH = "contract_mismatch"


class RemediationOwner(StrEnum):
    """Who a classified failure belongs to, decided before any model is selected.

    Model routing is only ever consulted for ``REQUIRES_FEATURE_ENGINEER``. That boundary is
    what stops a coding model being handed work that is not code: a repository whose own lint
    configuration references an uninstalled dependency needs a repository repair, and asking a
    remediation model to fix it repeatedly is the failure mode this enum exists to prevent.
    """

    REQUIRES_FEATURE_ENGINEER = "requires_feature_engineer"
    AUTO_REPAIRABLE_REPOSITORY = "auto_repairable_repository"
    INFRASTRUCTURE = "infrastructure"
    REQUIRES_HUMAN = "requires_human"


# Read from the classifications the platform already produces, rather than from a second
# opinion about the same failure. Each entry says what kind of problem that class *is*:
#
#   A missing implementation, a source failure the repository's own tools reported, an
#   unaddressed review finding and a contract mismatch are all defects in the code that was
#   written -- an Engineer's to correct.
#
#   Checked-in validation configuration and absent test infrastructure are properties of the
#   repository. They belong to the repair flow, which asks a person before it changes them.
#
#   A dependency install, and a command that exhausted its time or memory, are the machine
#   rather than the code; a coding attempt cannot make either succeed.
_REMEDIATION_OWNERS: dict[FailureClassification, RemediationOwner] = {
    FailureClassification.IMPLEMENTATION_MISSING: RemediationOwner.REQUIRES_FEATURE_ENGINEER,
    FailureClassification.VALIDATION_SOURCE_FAILURE: RemediationOwner.REQUIRES_FEATURE_ENGINEER,
    FailureClassification.REVIEW_SCOPE_FAILURE: RemediationOwner.REQUIRES_FEATURE_ENGINEER,
    FailureClassification.CONTRACT_MISMATCH: RemediationOwner.REQUIRES_FEATURE_ENGINEER,
    FailureClassification.VALIDATION_CONFIGURATION_FAILURE: (
        RemediationOwner.AUTO_REPAIRABLE_REPOSITORY
    ),
    FailureClassification.TEST_INFRASTRUCTURE_MISSING: RemediationOwner.AUTO_REPAIRABLE_REPOSITORY,
    FailureClassification.DEPENDENCY_INSTALLATION_FAILURE: RemediationOwner.INFRASTRUCTURE,
    FailureClassification.VALIDATION_CAPACITY_FAILURE: RemediationOwner.INFRASTRUCTURE,
}


def remediation_owner(classification: FailureClassification) -> RemediationOwner:
    """Return who a classified failure belongs to, before any model is selected for it."""
    return _REMEDIATION_OWNERS.get(classification, RemediationOwner.REQUIRES_HUMAN)


def requires_feature_engineer(classification: FailureClassification) -> bool:
    """Return whether Engineer remediation is the appropriate response to this failure."""
    return remediation_owner(classification) is RemediationOwner.REQUIRES_FEATURE_ENGINEER


class RetryPlan(StateModel):
    """The changed approach passed to an Engineer Agent on a retry."""

    failure_classification: FailureClassification
    root_cause: str = Field(min_length=1)
    required_strategy_change: str = Field(min_length=1)
    files_or_areas_to_inspect: list[str] = Field(min_length=1)
    previous_approach_to_avoid: str = Field(min_length=1)
    validation_to_rerun: list[str] = Field(min_length=1)
    # What the same review said and did not block on. Carried separately from `root_cause`
    # because these are not why the attempt failed and must not be read as the work order --
    # but they are findings the platform already holds, and dropping them is how a defect
    # gets raised, withheld, and then blocked on one attempt later.
    #
    # AB-Feature-215's backend: its first review raised a `high` authorization defect and a
    # `medium` test-coverage defect. `_blocking_findings` returns only the criticals and highs
    # whenever any exist, so the remediation was handed one finding of the two, fixed it, and
    # was failed by the second on the very next attempt -- a full retry spent on a defect the
    # platform had already written down. Advisory findings ride along from here so the next
    # attempt can settle them while it is in the file anyway.
    advisory_findings: list[str] = Field(default_factory=list)


def classify_failure(
    *,
    preflight: RepositoryPreflightResult | None = None,
    completeness: ImplementationCompletenessResult | None = None,
    validation_results: tuple[ValidationResult, ...] = (),
    review_findings: list[dict[str, Any]] | None = None,
) -> FailureClassification:
    """Identify the smallest responsible retry class, preferring unsafe setup blocks."""
    if preflight is not None and preflight.validation_readiness == "blocked":
        if preflight.dependency_install_status == "failed":
            return FailureClassification.DEPENDENCY_INSTALLATION_FAILURE
        return FailureClassification.VALIDATION_CONFIGURATION_FAILURE
    if completeness is not None and not completeness.passed:
        return FailureClassification.IMPLEMENTATION_MISSING
    if any(
        result.validation_type == "test"
        and result.status in {ValidationStatus.NOT_CONFIGURED, ValidationStatus.NO_TESTS_FOUND}
        for result in validation_results
    ):
        return FailureClassification.TEST_INFRASTRUCTURE_MISSING
    if any(
        result.validation_type == "lint"
        and result.failure_classification
        in {"lint_configuration_error", "missing_dependency", "unsupported_runtime"}
        for result in validation_results
    ):
        return FailureClassification.VALIDATION_CONFIGURATION_FAILURE
    # Ordered before the generic source rule: a command that timed out or exhausted the
    # machine's memory never produced a verdict on the source, so reading it as a source
    # failure spends a coding budget on feedback that does not exist. AB-Feature-108 lost
    # its last attempts to exactly that, against a suite whose full run cannot fit here.
    if any(
        result.status is ValidationStatus.FAILED
        and result.failure_classification in CAPACITY_FAILURE_CLASSIFICATIONS
        for result in validation_results
    ):
        return FailureClassification.VALIDATION_CAPACITY_FAILURE
    if any(result.status is ValidationStatus.FAILED for result in validation_results):
        return FailureClassification.VALIDATION_SOURCE_FAILURE
    findings = review_findings or []
    if any(item.get("finding_category") == "contract" for item in findings):
        return FailureClassification.CONTRACT_MISMATCH
    return FailureClassification.REVIEW_SCOPE_FAILURE


def a_required_command_rejected_the_source(results: Sequence[dict[str, Any]]) -> bool:
    """Whether any required command ran and returned a verdict against this change's source.

    The question `build_retry_plan` needs to tell its two `validation_source_failure` shapes
    apart. The classification covers both a command that failed and a review finding that
    merely reports one, so the plan's advice was written for the first and handed to the
    second: AB-Feature-225's attempts 1, 2 and 3 were each told to "inspect the exact failing
    source diagnostics" when format, lint and build were green and nothing had failed at all.
    """
    return any(
        isinstance(item, dict)
        and item.get("required", True)
        and item.get("status") == ValidationStatus.FAILED.value
        for item in results
    )


def build_retry_plan(
    classification: FailureClassification,
    *,
    expected_source_areas: list[str],
    previous_findings: list[str],
    current_revision: str | None,
    advisory_findings: Sequence[str] = (),
    source_verdict_available: bool = True,
) -> RetryPlan:
    """Turn a failure into a materially different, auditable Engineer strategy.

    ``source_verdict_available`` says whether a required command actually rejected the source.
    It defaults to True, which is the historical reading and the right one wherever the caller
    cannot tell.
    """
    revision_note = current_revision or "the current repository checkout"
    areas = expected_source_areas or ["repository source layout and configured scripts"]
    root = "; ".join(previous_findings[:3]) or "the prior child attempt did not satisfy its gate"
    plans: dict[FailureClassification, tuple[str, str, list[str]]] = {
        FailureClassification.IMPLEMENTATION_MISSING: (
            (
                "Inspect the repository's established source layout and implement the missing "
                "production behavior before changing tests."
            ),
            "Repeating test-only edits without production implementation.",
            ["repository-native test command", "lint", "typecheck", "build"],
        ),
        FailureClassification.VALIDATION_SOURCE_FAILURE: (
            (
                "Inspect the exact failing source diagnostics and make a targeted source fix "
                "before rerunning the same command."
            ),
            "Re-running unchanged validation against the same source failure.",
            ["the failed repository-native validation command"],
        ),
        FailureClassification.VALIDATION_CAPACITY_FAILURE: (
            (
                "Reduce what the required command has to do -- scope it to the suites this "
                "change touches, or raise its limit -- rather than editing the source it "
                "never finished judging."
            ),
            "Rewriting application source in response to a command that never completed.",
            ["the required validation command, scoped to the changed suites"],
        ),
        FailureClassification.VALIDATION_CONFIGURATION_FAILURE: (
            (
                "Inspect checked-in lint/runtime configuration and lockfile declarations; do "
                "not treat bootstrap failure as a source lint violation."
            ),
            "Changing application code before resolving the checked-in validation configuration.",
            ["deterministic dependency installation", "lint"],
        ),
        FailureClassification.DEPENDENCY_INSTALLATION_FAILURE: (
            (
                "Inspect the repository's selected dependency lockfile and installation error; "
                "automatic coding retries are unsafe until setup is resolved."
            ),
            "Retrying the Engineer Agent while deterministic dependency installation still fails.",
            ["deterministic dependency installation"],
        ),
        FailureClassification.TEST_INFRASTRUCTURE_MISSING: (
            (
                "Inspect existing repository test conventions and dependencies, then add tests "
                "only through an explicitly justified framework/configuration change."
            ),
            "Claiming unconfigured or empty tests passed.",
            ["configured test command", "lint"],
        ),
        FailureClassification.REVIEW_SCOPE_FAILURE: (
            (
                "Re-read the scoped requirement evidence and previous reviewer findings; "
                "address only this child repository's assigned scope."
            ),
            "Reusing a generic retry prompt without the reviewer findings.",
            ["repository review", "configured validation commands"],
        ),
        FailureClassification.CONTRACT_MISMATCH: (
            (
                "Compare the implementation against the approved contract projection and "
                "correct the child without changing the contract."
            ),
            "Changing shared contract artifacts from a child workstream.",
            ["contract projection check", "repository validation"],
        ),
    }
    strategy, avoid, validations = plans[classification]
    # The same classification, the other shape. `validation_source_failure` is what a reviewer's
    # own findings are called whenever one of them reports a failing command, so it arrives here
    # for a review rejection just as it does for a command that exited non-zero -- and the advice
    # above is written for the second. Given to the first it names a diagnostic that does not
    # exist, and an attempt that goes looking for one finds the only failing thing in reach: the
    # repository's own validation configuration. AB-Feature-225 spent attempts 1, 2 and 3
    # rewriting its checkout's `test` script while every required command was green.
    if classification is FailureClassification.VALIDATION_SOURCE_FAILURE and (
        not source_verdict_available
    ):
        strategy = (
            "No required validation command failed, so there is no failing source diagnostic "
            "to inspect. What blocks this attempt is the review's findings: read them and "
            "address what each one asks for. If a finding asks for a command to be RUN, you "
            "cannot satisfy it -- the platform decides which commands run, and changing the "
            "repository's own scripts to make one of them run something else is a change "
            "nobody asked for. Say so in your completion summary and leave the scripts alone."
        )
        avoid = (
            "Editing application source, test configuration or the repository's package "
            "scripts in response to a review finding that reports no failing command."
        )
        validations = ["the repository's configured validation commands, unchanged"]
    return RetryPlan(
        failure_classification=classification,
        root_cause=f"At {revision_note}: {root}",
        required_strategy_change=strategy,
        files_or_areas_to_inspect=areas,
        previous_approach_to_avoid=avoid,
        validation_to_rerun=validations,
        # Deduplicated against the blocking set: a finding that already leads the work order
        # must not also appear as something nobody has judged.
        advisory_findings=[
            item for item in dict.fromkeys(advisory_findings) if item not in set(previous_findings)
        ],
    )


# The tokens that identify *which* defect a diagnostic is about, as opposed to how the
# model chose to word it this time. A path with a line number, an exception or error-code
# identifier, and a linter rule name are all stable across rewordings; prose is not.
_LOCATION_PATTERN = re.compile(r"\b[\w./-]+\.[A-Za-z]{1,5}:\d+\b")
_ERROR_NAME_PATTERN = re.compile(r"\b[A-Z][A-Za-z]*(?:Error|Exception|Warning)\b")
_RESULT_CODE_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+){2,}\b")
# Bounded on both sides so a path segment pair -- `src/pages` inside a longer file path --
# cannot be mistaken for a scoped linter rule such as `react/display-name`.
_LINT_RULE_PATTERN = re.compile(
    r"(?<![\w/.-])[a-z][a-z0-9]*(?:-[a-z0-9]+)*/[a-z][a-z0-9-]+(?![\w/.])"
)
_EXIT_CODE_PATTERN = re.compile(r"\bexit(?:ed with)?\s+code\s+(\d+)\b", re.IGNORECASE)


def diagnostic_signature(findings: Sequence[str]) -> tuple[str, ...]:
    """Reduce a set of blocking issues to the defects they name, ignoring their wording.

    Two attempts that fail on the same exception at the same line have not responded to
    their feedback, however differently the sentence around it is phrased. Comparing the
    prose instead missed exactly that: AB-Feature-108's attempts 5, 6 and 7 all died on
    `TypeError: undefined is not iterable` at `AllApps.js:117`, described three ways, and
    the loop read three different strings and allowed all three.

    Returns an empty tuple when nothing identifying can be extracted, which callers must
    treat as "unknown" rather than "unchanged": a signature that captured nothing would
    otherwise compare equal to every other one that captured nothing.
    """
    tokens: set[str] = set()
    for finding in findings:
        for match in _LOCATION_PATTERN.findall(finding):
            tokens.add(f"at:{match}")
        for match in _ERROR_NAME_PATTERN.findall(finding):
            tokens.add(f"error:{match}")
        for match in _RESULT_CODE_PATTERN.findall(finding):
            tokens.add(f"code:{match}")
        for match in _LINT_RULE_PATTERN.findall(finding):
            tokens.add(f"rule:{match}")
        for match in _EXIT_CODE_PATTERN.findall(finding):
            tokens.add(f"exit:{match}")
    return tuple(sorted(tokens))


def retry_counter_field(classification: FailureClassification) -> str:
    """Map a failure to its isolated persisted retry counter."""
    return {
        FailureClassification.IMPLEMENTATION_MISSING: "implementation_retry_count",
        FailureClassification.VALIDATION_SOURCE_FAILURE: "validation_retry_count",
        # Drawn from the repository-setup budget, not the coding one. Whether the required
        # command fits in this sandbox is a property of the checkout and the machine, so an
        # allowance spent on it must not reduce the attempts left for writing source.
        FailureClassification.VALIDATION_CAPACITY_FAILURE: "repository_setup_retry_count",
        FailureClassification.VALIDATION_CONFIGURATION_FAILURE: "repository_setup_retry_count",
        FailureClassification.DEPENDENCY_INSTALLATION_FAILURE: "repository_setup_retry_count",
        FailureClassification.TEST_INFRASTRUCTURE_MISSING: "repository_setup_retry_count",
        FailureClassification.REVIEW_SCOPE_FAILURE: "implementation_retry_count",
        FailureClassification.CONTRACT_MISMATCH: "integration_retry_count",
    }[classification]


class RetryDecision(StateModel):
    """The single authoritative verdict on whether one more attempt may run."""

    should_retry: bool
    reason: str = Field(min_length=1)
    counter_field: str = Field(min_length=1)
    # What to put to a human when the answer is no. A refusal reason states what the platform
    # observed; it does not tell whoever now owns this workstream what decision is theirs to
    # make. Ending a run with "the implementation_retry_count budget of 4 is exhausted" and
    # nothing else leaves an operator to reconstruct, from artifacts, what was tried and
    # whether it was ever achievable. The question is empty only when work may continue.
    operator_question: str = ""
    # What this decision concluded about the attempt it judged, so the caller can persist it
    # without re-deciding it. The convergence rules below can overturn the raw
    # `meaningful_progress` verdict -- an attempt that returned to a state this workstream
    # already submitted is not progress however new it looked against the attempt immediately
    # before -- and the durable child row must record what was decided rather than what was
    # observed. Previously the caller computed both, which is how one loop came to hold two
    # opinions about whether the same attempt had made progress.
    meaningful_change: bool = True
    meaningful_change_reason: str = ""
    # The convergence counters carried into the next attempt. Returned rather than updated by
    # the caller because the conditions that increment them are the same conditions this
    # function decides on, and a caller that re-evaluated them would be reaching a verdict.
    identical_resubmissions: int = 0
    returns_to_earlier_states: int = 0
    repeated_diagnostic_signatures: int = 0
    # Whether the blocker in front of this workstream stopped responding to the source it
    # was handed. Read by `triage_stopped_workstream`, which needs it to tell a repository
    # that kept hitting one wall from one that was converging and ran out of room.
    blocker_repeated: bool = False
    # True only for the refusal the convergence rules produced. The caller annotates that
    # attempt differently -- it says the remaining budget was deliberately not spent -- and
    # this is what tells it which refusal it is holding without inspecting the reason text.
    non_converging: bool = False


# A structurally broken repository cannot be fixed by coding again. A failed dependency
# install is different: it is frequently transient infrastructure, so it alone consumes
# the repository-setup budget rather than terminating the attempt outright.
_TERMINAL_SETUP_CLASSIFICATIONS = frozenset(
    {
        FailureClassification.VALIDATION_CONFIGURATION_FAILURE,
        FailureClassification.TEST_INFRASTRUCTURE_MISSING,
    }
)


# A command that cannot complete here is bounded separately from one that completes and
# rejects the source. One allowance is deliberate rather than zero: a runaway test the
# attempt itself just wrote is worth one scoped rerun, while a suite that does not fit in
# this sandbox will not start fitting because the source changed again.
_CAPACITY_CLASSIFICATIONS = frozenset({FailureClassification.VALIDATION_CAPACITY_FAILURE})

_CAPACITY_OPERATOR_QUESTION = (
    "The required validation command never finished here -- it hit its time limit or "
    "exhausted the machine's memory -- so it never returned a verdict on the code that was "
    "written. Should this command be scoped to the suites this change touches, given a "
    "higher limit or more memory, or should this repository run without it?"
)


class TerminalCause(StrEnum):
    """Why a stopped workstream is unreachable, which decides who has to act next."""

    REPOSITORY_SETUP = "repository_setup"
    VALIDATION_CAPACITY = "validation_capacity"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    NEVER_IMPLEMENTED = "never_implemented"
    BLOCKER_UNRESPONSIVE = "blocker_unresponsive"
    BUDGET_EXHAUSTED = "budget_exhausted"
    # A review is demanding what an earlier attempt of this workstream already delivered and a
    # later one removed. Deliberately not `BLOCKER_UNRESPONSIVE`: that cause means the model
    # could not act on its feedback, and it sends its reader to the repository or to a stronger
    # tier. Here the model acted on the feedback perfectly well -- twice, in opposite
    # directions -- because two demands disagree. The owner is the person who decides which one
    # holds, and no amount of retrying reaches them.
    DESIGN_CONFLICT = "design_conflict"


class TreeResubmission(StateModel):
    """The measured fact behind an identical-resubmission stop: which tree came back.

    Carried into the triage so its evidence can say what the firing rule compared --
    ``production_diff_fingerprint`` equality -- instead of asserting a diagnostic comparison
    the rule never made. AB-Feature-207's stop printed "the blocking diagnostics did not
    change between attempts" over a resubmission whose diagnostics demonstrably changed.
    """

    # The attempt whose tree this attempt reproduced, where the lineage identifies it.
    attempt: int | None = None
    # This attempt's production diff fingerprint, and the matched earlier artifact's own.
    # Equal by construction -- recorded from both records so the metadata proves the
    # measurement instead of restating it.
    production_diff_fingerprint: str = ""
    matched_production_diff_fingerprint: str = ""


class TerminalTriage(StateModel):
    """The cause a stopped workstream is attributed to, and what that makes someone's job."""

    cause: TerminalCause
    question: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    # Measured values behind the evidence sentences -- fingerprints, attempt numbers --
    # merged into the result metadata as `terminal_measurements` so a report can read the
    # numbers without parsing prose.
    metadata: dict[str, str] = Field(default_factory=dict)


# How many consecutive attempts one substantive review finding must survive -- with the
# source changing underneath it -- before the terminal narrative reads it as a capability
# limit. Three, matching the repeated-signature allowance above and for the same reason:
# twice can be a model that read its feedback badly once; a third round of changed source
# against an unchanged requirement is a model that cannot do this at this tier.
_CAPABILITY_FINDING_SURVIVED_ATTEMPTS = 3


def _feature_authored_diagnostic_paths(
    diagnostics: Sequence[str], authored_paths: Sequence[str]
) -> list[str]:
    """Return the authored files the diagnostics name, when they name nothing else.

    Empty means "not established", never "established false": the caller falls back to the
    open question. A token with a directory in it that resolves to no authored path is
    positive evidence the blocker involves a file this feature never wrote, so the whole
    attribution is withdrawn. A bare dotted token -- as likely a version number as a
    filename -- counts only when exactly one authored file carries that name, and is
    otherwise ignored rather than read as an outside file.
    """
    authored = [path for path in authored_paths if path]
    matched: list[str] = []
    for token, _line in diagnostic_file_references(diagnostics):
        resolved = _authored_path_for(token, authored)
        if resolved is None:
            if "/" in token:
                return []
            continue
        if resolved not in matched:
            matched.append(resolved)
    return matched


def _authored_path_for(token: str, authored_paths: Sequence[str]) -> str | None:
    """Resolve one diagnostic token to the attempt-lineage path it names, if any.

    A diagnostic spells the same file three ways: the lineage's own repo-relative path, an
    absolute workspace path ending in it, or a path relative to the directory the tool ran
    in -- a suffix of the lineage path. A bare filename resolves only when it is unambiguous.
    """
    if "/" in token:
        for path in authored_paths:
            if token == path or token.endswith(f"/{path}") or path.endswith(f"/{token}"):
                return path
        return None
    candidates = [path for path in authored_paths if PurePosixPath(path).name == token]
    return candidates[0] if len(candidates) == 1 else None


def triage_stopped_workstream(
    *,
    classification: FailureClassification,
    attempts: int,
    production_source_written: bool,
    diagnostics_repeated: bool = False,
    resubmission: TreeResubmission | None = None,
    fallback_question: str,
    final_diagnostics: Sequence[str] = (),
    attempt_authored_paths: Sequence[str] = (),
    surviving_finding: str | None = None,
    surviving_finding_attempts: int = 0,
    tier_label: str = "",
) -> TerminalTriage:
    """Attribute a stopped workstream to a cause, from evidence the attempts already produced.

    This decides nothing about retrying -- by the time it runs, that answer is already no. It
    exists because "N attempts were spent, decide whether the requirement, the repository or
    the budget should change" hands a human three possibilities and the job of working out
    which. The attempts themselves usually say which: a workstream that never wrote a line of
    production source is not the same as one that wrote plenty and kept hitting the same wall,
    and neither is the same as one that was converging and ran out of room.

    An unresponsive blocker splits further on evidence already in hand, because one sentence
    covered three different owners and sent two of them to the wrong place. AB-Feature-176's
    backend was told "a defect outside the implementation's reach" about a syntax error in a
    file the feature itself wrote, and AB-Feature-177 got the same sentence for a capability
    limit. ``final_diagnostics`` against ``attempt_authored_paths`` decides the first: when
    every file the last blocker names is the feature's own work, the defect is inside the
    implementation, not the repository. ``surviving_finding`` decides the second: a
    substantive review finding that outlived several attempts of changing source is what a
    model at this tier cannot do, not what this repository forbids.

    Deliberately conservative. Where the evidence does not distinguish, it falls back to the
    open question rather than guessing a cause, because a confidently wrong attribution sends
    someone to repair a repository that was never broken.

    ``diagnostics_repeated`` and ``resubmission`` are the two measurements the old
    ``blocker_repeated`` flag merged into one bit (77-, item 33). The first is a diagnostic
    comparison -- repeated findings text or repeated diagnostic signatures -- and only it may
    put "the blocking diagnostics did not change" on the record. The second is
    ``production_diff_fingerprint`` equality: an identical resubmission of the tree, which
    says nothing about the diagnostics around it. AB-Feature-207 stopped on the second while
    its diagnostics changed every round, and the fixed sentence written for the first misled
    the failure summary and a forensic pass. Each fact contributes only the sentence it
    actually establishes.
    """
    # Checked before the production-source rule below: source almost certainly was written
    # here, and attributing this to "nothing was ever built" would send someone to look for
    # a missing implementation instead of at the command that could not run.
    if classification in _CAPACITY_CLASSIFICATIONS:
        return TerminalTriage(
            cause=TerminalCause.VALIDATION_CAPACITY,
            question=_CAPACITY_OPERATOR_QUESTION,
            evidence=[
                f"{attempts} attempts ended with a required command that never completed.",
                "The command hit its time limit or exhausted memory, so it judged nothing.",
            ],
        )
    if classification in _TERMINAL_SETUP_CLASSIFICATIONS:
        return TerminalTriage(
            cause=TerminalCause.REPOSITORY_SETUP,
            question=(
                "This repository cannot run its own checks on an untouched checkout, so no "
                "implementation could have been validated here. Should the repository's "
                "configuration be repaired first, or should this feature run without the "
                "validation it is missing?"
            ),
            evidence=["The checkout was blocked before any code was written."],
        )
    if not production_source_written:
        return TerminalTriage(
            cause=TerminalCause.NEVER_IMPLEMENTED,
            question=(
                f"After {attempts} attempts this repository has no production source change "
                "at all, so nothing was ever built to review. Does what the requirement asks "
                "for exist in this repository -- and is it expressed in terms this repository "
                "can implement?"
            ),
            evidence=[
                f"{attempts} attempts produced no production source change.",
                "Tests or configuration alone cannot satisfy a production requirement.",
            ],
        )
    if diagnostics_repeated or resubmission is not None:
        # Only the sentences the firing rules established. The old fixed line asserted a
        # diagnostic comparison whichever rule fired; for 207 the rule was tree-fingerprint
        # equality and the diagnostics had changed.
        repeat_evidence: list[str] = []
        measurements: dict[str, str] = {}
        if diagnostics_repeated:
            repeat_evidence.append("The blocking diagnostics did not change between attempts.")
        if resubmission is not None:
            matched = (
                f"attempt {resubmission.attempt}"
                if resubmission.attempt is not None
                else "an earlier attempt"
            )
            repeat_evidence.append(f"This attempt resubmitted the production tree of {matched}.")
            if resubmission.production_diff_fingerprint:
                measurements["resubmitted_production_diff_fingerprint"] = (
                    resubmission.production_diff_fingerprint
                )
            if resubmission.matched_production_diff_fingerprint:
                measurements["matched_production_diff_fingerprint"] = (
                    resubmission.matched_production_diff_fingerprint
                )
            if resubmission.attempt is not None:
                measurements["resubmitted_attempt"] = str(resubmission.attempt)
        authored = _feature_authored_diagnostic_paths(final_diagnostics, attempt_authored_paths)
        if authored:
            reintroduced = (
                "the implementation repeatedly reintroduced the same defect into its own files"
                if diagnostics_repeated
                else "the implementation arrived back at a tree it had already submitted "
                "while the diagnostics named its own files"
            )
            return TerminalTriage(
                cause=TerminalCause.BLOCKER_UNRESPONSIVE,
                question=(
                    "Every file the final blocking diagnostics name was written by this "
                    f"feature's own attempts: {reintroduced}. The repository is not what has "
                    "to be fixed -- the diagnostics name the exact file and location to "
                    "correct. Should a person make that edit, or should this run be retried?"
                ),
                evidence=[
                    f"{attempts} attempts changed production source.",
                    *repeat_evidence,
                    "Every file the final diagnostics name is the feature's own work: "
                    f"{', '.join(authored)}.",
                ],
                metadata=measurements,
            )
        if (
            surviving_finding
            and surviving_finding_attempts >= _CAPABILITY_FINDING_SURVIVED_ATTEMPTS
        ):
            tier = f" at the {tier_label} tier" if tier_label else ""
            return TerminalTriage(
                cause=TerminalCause.BLOCKER_UNRESPONSIVE,
                question=(
                    "The same substantive review finding survived "
                    f"{surviving_finding_attempts} attempts while the source kept changing"
                    f"{tier}, which is what a model capability limit looks like -- not a "
                    "repository defect. Should this work be retried on a stronger tier, or "
                    "implemented by a person?"
                ),
                evidence=[
                    f"{attempts} attempts changed production source.",
                    f"The finding that survived every attempt: {surviving_finding}",
                ],
                metadata=measurements,
            )
        if diagnostics_repeated:
            return TerminalTriage(
                cause=TerminalCause.BLOCKER_UNRESPONSIVE,
                question=(
                    "Production source was written and rewritten, and the same blocker came "
                    "back unchanged each time -- which is what a defect outside the "
                    "implementation's reach looks like: an undeclared dependency, an "
                    "unconfigured rule, or a pre-existing fault. What in this repository has "
                    "to be fixed before this work can land?"
                ),
                evidence=[
                    f"{attempts} attempts changed production source.",
                    *repeat_evidence,
                ],
                metadata=measurements,
            )
        # Resubmission without a diagnostic repeat: the diagnostics moved, the tree did not.
        # That is a circling implementation, not a defect beyond its reach, and the question
        # must not send anyone to repair the repository.
        return TerminalTriage(
            cause=TerminalCause.BLOCKER_UNRESPONSIVE,
            question=(
                "Production source was written and rewritten, but this attempt arrived back "
                "at a tree the workstream had already submitted while the blocking "
                "diagnostics kept moving -- the implementation is circling rather than "
                "converging. Is the requirement achievable in this repository as written -- "
                "and if so, what does the implementation need to be told that it has not "
                "been?"
            ),
            evidence=[
                f"{attempts} attempts changed production source.",
                *repeat_evidence,
            ],
            metadata=measurements,
        )
    return TerminalTriage(
        cause=TerminalCause.BUDGET_EXHAUSTED,
        question=fallback_question,
        evidence=[f"{attempts} attempts changed production source and the blocker moved."],
    )


# ---------------------------------------------------------------------------------------
# Convergence bounds
#
# These lived beside the child loop in `workflows/feature_workflow.py` and were applied
# *after* `decide_child_retry` had already returned a verdict, which meant a workstream could
# be told it might proceed and then stopped anyway, with its refusal reason rewritten
# underneath it. Six production rows record their own stop as "Attempt N may proceed with a
# changed strategy" for exactly that reason. The bounds now sit with the function that owns
# the verdict; each comment names the feature the number was set for, because that is the
# only record of why it is what it is.
# ---------------------------------------------------------------------------------------

# How many times an attempt may resend the previous change before the loop gives up.
# Repeating once is a model that did not read its diagnostic; repeating twice is one that
# cannot act on it. Ending the workstream on the first repeat spent two attempts of eight:
# r-2's backend had registered its route and was one documentation edit from approval when
# it resent the same five files and was stopped. The second repeat still ends it, so ser-1's
# ten wasted attempts cannot come back.
_ALLOWED_IDENTICAL_RESUBMISSIONS = 2
# How many times an attempt may arrive back at a state this workstream already submitted,
# consecutively or not. Separate from the counter above because that one asks only about the
# attempt immediately before, so a workstream alternating A -> B -> A -> C -> A clears it
# every other attempt and is never stopped. Two, the same allowance: coming back once can be
# a reviewer reversing itself, twice is a circle.
_ALLOWED_RETURNS_TO_EARLIER_STATES = 2
# How many consecutive attempts may name the same defect -- the same exception at the same
# line, however the sentence around it is worded -- before the loop treats the blocker as
# one the model cannot act on. Three, not two, because the reduction to a signature is
# lossy: normalization can merge two genuinely different defects, and the documented cost of
# stopping a converging workstream too early is higher than one extra attempt. rep-a's
# frontend was ended three attempts in while its source was being rewritten behind an
# unchanging linter message.
_ALLOWED_REPEATED_DIAGNOSTIC_SIGNATURES = 3

# The two `meaningful_progress` verdicts this function compares against by name. The first is
# produced by `tools.implementation_completeness`; the second is produced here, for an attempt
# that arrived back at a state the workstream had already submitted.
REPRODUCED_PREVIOUS_CHANGE = "attempt_reproduced_the_previous_production_change"
RETURNED_TO_EARLIER_STATE = "attempt_returned_to_an_earlier_submitted_state"

_NON_CONVERGING_REFUSAL = (
    "The same diagnostics came back unchanged, so this workstream is not converging and its "
    "remaining attempts were not spent."
)


def _clamped_progress(
    *,
    meaningful_change: bool,
    meaningful_change_reason: str,
    production_change_present: bool,
    revisited_earlier_state: bool,
    source_rejected_before_commit: bool,
    identical_resubmissions: int,
    returns_to_earlier_states: int,
) -> tuple[bool, str, int, int]:
    """Re-judge one attempt's progress against everything this workstream has submitted.

    ``meaningful_progress`` compares an attempt with the one immediately before it, which is
    the wrong window for a workstream going round in circles: -069's console cycled unwired ->
    wired into the wrong page -> unwired, its third attempt byte-identical to its first, and
    every one of them looked new to a rule that only ever looked back one step.

    A pre-commit rejection resets the workspace, so the fingerprint taken afterwards describes
    the state the attempt started from rather than the source it wrote. -076's console
    produced two attempts with byte-identical fingerprints that failed at different gates,
    which cannot both be true of the same source. Reading that as a circle would undo the very
    allowance that lets a lint rejection be retried at all, so a rejected attempt is neither
    counted as a repeat nor allowed to clear the count of earlier repeats: clearing it is how
    a circling workstream became immortal. -112's console needed two consecutive repeats to be
    stopped, reached one on three separate occasions, and had the count wiped each time by the
    attempt in between. It spent all twelve review cycles submitting five copies of one
    production diff.
    """
    circling_is_unknowable = source_rejected_before_commit
    if (
        meaningful_change_reason == REPRODUCED_PREVIOUS_CHANGE
        or (revisited_earlier_state and not circling_is_unknowable)
    ) and production_change_present:
        # Only where production work exists to build on. A tests-only attempt that repeats
        # has still implemented nothing, and is refused at once as before.
        if revisited_earlier_state and not circling_is_unknowable:
            meaningful_change_reason = RETURNED_TO_EARLIER_STATE
            # Counted separately from consecutive resubmission, and never cleared. Returning
            # to a state this workstream already submitted is evidence of a circle whatever
            # it did on the way round, and the consecutive counter cannot see one with any
            # variety in it. AB-Feature-124's console went 04199e3a -> 53188603 -> 04199e3a
            # -> ac8cbe16 -> 04199e3a: three arrivals at the same state, each correctly
            # detected, and each forgotten by the time the next one came.
            returns_to_earlier_states += 1
        identical_resubmissions += 1
        meaningful_change = (
            identical_resubmissions < _ALLOWED_IDENTICAL_RESUBMISSIONS
            and returns_to_earlier_states < _ALLOWED_RETURNS_TO_EARLIER_STATES
        )
    elif not circling_is_unknowable:
        identical_resubmissions = 0
    return (
        meaningful_change,
        meaningful_change_reason,
        identical_resubmissions,
        returns_to_earlier_states,
    )


def decide_child_retry(
    *,
    classification: FailureClassification,
    attempt_count: int,
    budget: int,
    retry_count: int,
    max_child_review_cycles: int,
    meaningful_change: bool,
    # -- Evidence the caller gathered, not verdicts it reached ----------------------------
    # Everything below describes what was observed about the attempt just finished. Three of
    # these rules used to run in `_run_one_child` *after* this function had returned
    # `should_retry=True`, stop the workstream anyway, and overwrite its reason. The caller
    # now gathers facts and applies one verdict; it decides nothing.
    #
    # Every one defaults to the reading of "this attempt repeated nothing", so a caller with
    # no convergence history -- the integration remediation edge, and every test that asks
    # only about a budget -- gets exactly the verdict it got before.
    meaningful_change_reason: str = "",
    production_change_present: bool = False,
    revisited_earlier_state: bool = False,
    source_rejected_before_commit: bool = False,
    findings_repeat_previous: bool = False,
    diagnostic_signature_repeated: bool = False,
    deterministic_gate: bool = False,
    identical_resubmissions: int = 0,
    returns_to_earlier_states: int = 0,
    repeated_diagnostic_signatures: int = 0,
) -> RetryDecision:
    """Decide whether a child workstream may attempt again.

    This is the only place that answer is produced, and now the only place its reason is.
    Callers supply evidence and apply the verdict; they must not add their own retry
    conditions, which is how the platform ended up with several counters disagreeing about
    the same loop.

    The order of the rules is the order the platform applied them when they were spread over
    two files: the repeat clamp re-judges progress, then the four refusals, then the
    convergence rules that used to run only once a verdict of "yes" had already been written
    into durable state.
    """
    counter_field = retry_counter_field(classification)
    (
        meaningful_change,
        meaningful_change_reason,
        identical_resubmissions,
        returns_to_earlier_states,
    ) = _clamped_progress(
        meaningful_change=meaningful_change,
        meaningful_change_reason=meaningful_change_reason,
        production_change_present=production_change_present,
        revisited_earlier_state=revisited_earlier_state,
        source_rejected_before_commit=source_rejected_before_commit,
        identical_resubmissions=identical_resubmissions,
        returns_to_earlier_states=returns_to_earlier_states,
    )
    # The same rule as the exact-text comparison below, applied to what the diagnostics
    # identify rather than how they are written, so a model that rephrases its report every
    # attempt cannot defeat the comparison.
    repeated_diagnostic_signatures = (
        repeated_diagnostic_signatures + 1 if diagnostic_signature_repeated else 0
    )

    def decided(
        *,
        should_retry: bool,
        reason: str,
        operator_question: str = "",
        non_converging: bool = False,
    ) -> RetryDecision:
        """Attach the evidence this call resolved to whichever verdict it reached."""
        return RetryDecision(
            should_retry=should_retry,
            reason=reason,
            counter_field=counter_field,
            operator_question=operator_question,
            meaningful_change=meaningful_change,
            meaningful_change_reason=meaningful_change_reason,
            identical_resubmissions=identical_resubmissions,
            returns_to_earlier_states=returns_to_earlier_states,
            repeated_diagnostic_signatures=repeated_diagnostic_signatures,
            # An attempt counted as an identical resubmission wrote source the gate could
            # not tell apart from the last one, which is the same evidence as unchanged
            # diagnostics: the blocker is not responding to source. A non-converging refusal
            # is that evidence by definition.
            blocker_repeated=non_converging or identical_resubmissions > 0,
            non_converging=non_converging,
        )

    if classification in _TERMINAL_SETUP_CLASSIFICATIONS:
        return decided(
            should_retry=False,
            reason=(
                "The repository's checked-in setup is invalid; an approved repository repair "
                "is required before coding can be retried."
            ),
            operator_question=(
                "This repository cannot run its own checks on an untouched checkout, so no "
                "implementation could have been validated here. Should the repository's "
                "configuration be repaired first, or should this feature run without the "
                "validation it is missing?"
            ),
        )
    # The loop-prevention invariant: no attempt may repeat without a changed input. It
    # governs coding retries only. A failed dependency install changes no source by
    # definition, so it is bounded by its own budget rather than by this guard.
    # A capacity failure is exempt for the same reason a failed install is: the change it
    # asks for is to the command or its limits, not to application source, so requiring a
    # production diff would refuse the one retry that could scope the command down.
    if (
        classification is not FailureClassification.DEPENDENCY_INSTALLATION_FAILURE
        and classification not in _CAPACITY_CLASSIFICATIONS
        and not meaningful_change
    ):
        return decided(
            should_retry=False,
            reason=(
                "The previous attempt produced no meaningful production change, so repeating "
                "it would submit identical inputs."
            ),
            operator_question=(
                "The last attempt submitted the same source as the one before it, so this "
                "workstream is not converging on its own. Is the requirement achievable in "
                "this repository as written -- and if so, what does the implementation need "
                "to be told that it has not been?"
            ),
        )
    if attempt_count >= budget:
        return decided(
            should_retry=False,
            reason=f"The {counter_field} budget of {budget} is exhausted.",
            # A capacity failure exhausts its budget for a different reason than a coding
            # one, and asking "should the requirement change?" about a suite that ran out of
            # memory sends the reader to the wrong decision entirely.
            operator_question=(
                _CAPACITY_OPERATOR_QUESTION
                if classification in _CAPACITY_CLASSIFICATIONS
                else (
                    f"{budget} attempts were spent on this repository without reaching an "
                    "approved review. Read the blocking issues above: should the requirement "
                    "change, should the repository be repaired, or is this work simply larger "
                    "than the attempt budget it was given?"
                )
            ),
        )
    if retry_count + 1 >= max_child_review_cycles:
        return decided(
            should_retry=False,
            reason=f"The child review cycle limit of {max_child_review_cycles} is reached.",
            operator_question=(
                f"This repository reached its limit of {max_child_review_cycles} review "
                "cycles without an approval. Reading the blocking issues above: is it the "
                "requirement, the repository, or the cycle limit that should change?"
            ),
        )
    # An attempt that reproduced the previous diagnostics exactly did not respond to the
    # feedback at all, which is what a defect outside the model's reach looks like: an
    # undeclared dependency or an unconfigured lint rule cannot be fixed by writing the file
    # again. Only this attempt's own findings are compared, because inherited feedback is
    # constant by construction and would stop the first retry.
    #
    # A deterministic gate is exempt from both readings. This rule treats repeated text as a
    # model that ignored its feedback, which holds for a reviewer writing prose and not for a
    # check that emits the same sentence for as long as its condition holds. Feature -052b
    # lost both workstreams two attempts in because the reachability gate said the same true
    # thing twice while the attempts around it were changing.
    repeated_findings = findings_repeat_previous or repeated_diagnostic_signatures >= (
        _ALLOWED_REPEATED_DIAGNOSTIC_SIGNATURES - 1
    )
    if repeated_findings and not deterministic_gate:
        return decided(
            should_retry=False,
            reason=_NON_CONVERGING_REFUSAL,
            # Deliberately empty, exactly as it was when this rule lived in the caller and
            # ran against a positive verdict whose question is empty by construction. The
            # triage this feeds is handed `non_converging` as its diagnostics-repeated fact,
            # true here by definition, so it answers from its own text and never falls back
            # to this.
            operator_question="",
            non_converging=True,
        )
    return decided(
        should_retry=True,
        reason=f"Attempt {retry_count + 1} may proceed with a changed strategy.",
    )


__all__ = [
    "FailureClassification",
    "RemediationOwner",
    "RetryDecision",
    "REPRODUCED_PREVIOUS_CHANGE",
    "RETURNED_TO_EARLIER_STATE",
    "RetryPlan",
    "TerminalCause",
    "TerminalTriage",
    "TreeResubmission",
    "a_required_command_rejected_the_source",
    "build_retry_plan",
    "classify_failure",
    "decide_child_retry",
    "diagnostic_file_references",
    "diagnostic_signature",
    "remediation_owner",
    "requires_feature_engineer",
    "retry_counter_field",
    "triage_stopped_workstream",
]
