"""Deterministic completion checks for repository workstreams.

The coding model reports files; this module verifies that those files satisfy the
production and test categories explicitly promised by the execution plan.

Two bucket sets, and they must never be crossed. The attempt's own declared change is what
this module *reports* -- the lists and fingerprints the runtime copies onto the completion
and the child, and that `meaningful_progress` reads to decide whether an attempt repeated
itself. The workstream's branch is what it *judges against*, because a category promised by
a plan is a property of the branch the workstream builds, not of whichever one-line
remediation happens to be in flight. AB-Feature-201's frontend was rejected six times
running for "requires tests but no test file changed" while both of its test files sat
committed on its own branch, written by the attempt this one was correcting.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field

from artifacts.schemas import CodeCompletionArtifact, RepositoryWorkstreamPlan
from state.models import StateModel


class CompletionFinding(StateModel):
    """A deterministic implementation-completeness failure."""

    code: Literal[
        "tests_only_change",
        "documentation_only_change",
        "configuration_only_change",
        "missing_required_source_change",
        "missing_required_test_change",
        "unexpected_scope_change",
    ]
    requirement_id: str | None = None
    description: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)


CategoryEvidenceSource = Literal["branch_diff", "lineage_completions", "attempt_declared_paths"]


class ImplementationCompletenessResult(StateModel):
    """Completion evidence that must pass before repository review can approve work."""

    passed: bool
    production_files_changed: list[str] = Field(default_factory=list)
    test_files_changed: list[str] = Field(default_factory=list)
    configuration_files_changed: list[str] = Field(default_factory=list)
    requirements_implemented: list[str] = Field(default_factory=list)
    requirements_not_implemented: list[str] = Field(default_factory=list)
    implementation_expectations_satisfied: list[str] = Field(default_factory=list)
    findings: list[CompletionFinding] = Field(default_factory=list)
    production_diff_fingerprint: str = Field(min_length=1)
    test_diff_fingerprint: str | None = None
    # Which evidence decided category satisfaction, recorded because a silent fall-through to
    # the attempt's own declared paths is the defect this parameter exists to fix. A run that
    # regressed to `attempt_declared_paths` on a remediation attempt has the old behaviour
    # whatever its blocking-issue count happens to be, and that has to be legible on the
    # record rather than inferred from an absence of findings.
    category_evidence_source: CategoryEvidenceSource = "attempt_declared_paths"


_PRODUCTION_CATEGORIES = {
    "production",
    "route",
    "controller",
    "service",
    "model",
    "migration",
    "frontend_component",
    "client",
}
# Categories a plan may name that the checkout can turn out not to need. Documentation is
# deliberately absent: a repository always has somewhere to document a change.
_ADVISORY_CATEGORIES = frozenset({"configuration"})
_CONFIGURATION_NAMES = {
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "tsconfig.json",
    "pyproject.toml",
    "setup.cfg",
    "eslint.config.js",
    ".eslintrc",
    ".eslintrc.js",
}
_CONFIGURATION_DIRECTORIES = {"config", "configs", "configuration", "settings"}
# Test-infrastructure filenames the ecosystems reserve for wiring their runners -- a naming
# convention, exactly like the configuration names above. `conftest.py` matches no test token
# (`test_` needs its underscore), so it classified as production and the reachability gate
# demanded production wiring for it: AB-Feature-216's attempt 3 satisfied that demand by
# exporting a string from an OAuth model. Classified as configuration rather than test,
# deliberately: a `test` classification would hand these files to the narrowed runners as
# suites, and `pytest -q conftest.py` collects nothing and fails. Helper names that runners
# collect as suites by convention (`test_helper.py`, `tests_helper.js`) are deliberately
# absent.
_TEST_INFRASTRUCTURE_NAMES = {"conftest.py"}
_TEST_INFRASTRUCTURE_PREFIXES = (
    "jest.setup.",
    "jest.teardown.",
    "vitest.setup.",
    "setuptests.",
    "setup_tests.",
)


def classify_file_change(
    path: str,
) -> Literal["production", "test", "configuration", "documentation"]:
    """Classify a workspace-relative path without accepting path traversal."""
    normalized = PurePosixPath(path)
    parts = tuple(part.lower() for part in normalized.parts)
    name = normalized.name.lower()
    if name in _CONFIGURATION_NAMES or name.startswith(".eslintrc") or name.endswith(".config.js"):
        return "configuration"
    if name in _TEST_INFRASTRUCTURE_NAMES or name.startswith(_TEST_INFRASTRUCTURE_PREFIXES):
        return "configuration"
    # A directory convention, read the same way `docs/` is below. A repository that keeps a
    # `config` directory puts configuration in it, and the file inside carries a name of the
    # repository's choosing that no filename list can anticipate.
    if any(part in _CONFIGURATION_DIRECTORIES for part in parts[:-1]):
        return "configuration"
    if parts and parts[0] in {"docs", "documentation"} or name.endswith(".md"):
        return "documentation"
    if any(part in {"test", "tests", "__tests__", "spec", "specs"} for part in parts) or any(
        token in name for token in (".test.", ".spec.", "_test.", "test_")
    ):
        return "test"
    return "production"


def validate_implementation_completeness(
    workstream: RepositoryWorkstreamPlan,
    completion: CodeCompletionArtifact,
    *,
    branch_paths: Sequence[str] | None = None,
    branch_modified_paths: Sequence[str] | None = None,
    branch_evidence_source: CategoryEvidenceSource | None = None,
) -> ImplementationCompletenessResult:
    """Compare the workstream's branch to its explicit implementation contract.

    ``branch_paths`` is every path the workstream's branch has changed, with
    ``branch_modified_paths`` naming those that already existed at its baseline; the caller
    runs Git because it holds the workspace, and this stays a synchronous function over data.
    They feed category satisfaction and the tests-required question and nothing else -- the
    reported lists and both fingerprints keep describing this attempt alone, because they are
    copied onto the completion artifact and read by ``meaningful_progress``, whose -105 and
    ser-1 churn guards a union would disarm.

    Passing nothing is exactly today's behaviour: the attempt's own declared buckets judge it,
    and the result says so.
    """
    paths = [change.path for change in completion.file_changes]
    buckets = _buckets(paths, completion)
    contract_buckets, evidence_source = _contract_buckets(
        buckets,
        branch_paths=branch_paths,
        branch_modified_paths=branch_modified_paths,
        branch_evidence_source=branch_evidence_source,
    )
    findings: list[CompletionFinding] = []
    implemented: list[str] = []
    missing: list[str] = []
    satisfied: list[str] = []
    expectations = workstream.implementation_expectations
    for expectation in expectations:
        categories = set(expectation.expected_change_categories)
        source_required = bool(categories & _PRODUCTION_CATEGORIES)
        # Judged against the branch, not this attempt's delta. A targeted remediation owes the
        # one-line fix it was asked for; the categories the plan promised are the workstream's
        # to satisfy, and an attempt that cannot satisfy them however correct its change is an
        # attempt the gate can only repeat itself at.
        missing_categories = sorted(
            category
            for category in categories
            if not _category_satisfied(
                category, contract_buckets, expectation.expected_source_areas
            )
        )
        # `configuration` is promised before the repository is checked out, and a change can
        # be genuinely complete without it: mounting an existing component onto an existing
        # page needs no configuration file. Enforcing it absolutely made every attempt
        # unsatisfiable, and the workstream stopped for repeating the one diagnostic it could
        # never clear. Report it as an advisory the reviewer weighs, not a completion gate.
        missing_categories = [
            category for category in missing_categories if category not in _ADVISORY_CATEGORIES
        ]
        category_matches = not missing_categories
        # The same question as the categories above, and the sentence it produces was 201's
        # unclearable one: does this workstream have tests, not did this attempt write one.
        tests_present = bool(contract_buckets["test"])
        # Deliberately the attempt's own buckets. "This change is tests only" and "this change
        # touched no production source" are statements about what was just written, and the
        # tail checks below say the same thing for the same reason.
        if source_required and not buckets["production"]:
            findings.append(
                _finding(
                    "tests_only_change" if buckets["test"] else "missing_required_source_change",
                    expectation.requirement_id,
                    (
                        "The requirement expects production implementation but no production "
                        "source file changed."
                    ),
                    (
                        "Implement the required production behavior in the repository's "
                        "established source layout before relying on tests."
                    ),
                )
            )
        # A generic planned source area is only a hint because planning happens before
        # checkout. Explicit categories such as route/controller/service are completion
        # promises, however, and cannot be satisfied by an unrelated production edit.
        elif missing_categories:
            findings.append(
                _finding(
                    "missing_required_source_change",
                    expectation.requirement_id,
                    (
                        "The implementation does not satisfy required change categories: "
                        f"{', '.join(missing_categories)}."
                        + (
                            " Every file in this change is new, so nothing that already runs "
                            "refers to any of it. Edit the existing file that has to use the "
                            "new code -- the module that registers the route, or the view "
                            "that renders the component."
                            if "integration" in missing_categories
                            else ""
                        )
                    ),
                    (
                        "Implement every scoped category using the repository's established "
                        "layout, or obtain an explicit plan correction."
                    ),
                )
            )
        if expectation.tests_required and not tests_present:
            findings.append(
                _finding(
                    "missing_required_test_change",
                    expectation.requirement_id,
                    "The workstream requires tests but no test file changed.",
                    (
                        "Add executable tests using the repository's existing framework, or "
                        "justify an approved infrastructure change."
                    ),
                )
            )
        if category_matches and (not expectation.tests_required or tests_present):
            implemented.append(expectation.requirement_id)
            satisfied.append(expectation.requirement_id)
        else:
            missing.append(expectation.requirement_id)
    if (
        paths
        and not buckets["production"]
        and buckets["test"]
        and any(
            set(item.expected_change_categories) & _PRODUCTION_CATEGORIES for item in expectations
        )
        and not any(item.code == "tests_only_change" for item in findings)
    ):
        findings.append(
            _finding(
                "tests_only_change",
                None,
                (
                    "Only test files changed even though the workstream expects production "
                    "implementation."
                ),
                "Implement the expected production source changes before revising tests again.",
            )
        )
    if paths and not buckets["production"] and buckets["documentation"] and not buckets["test"]:
        findings.append(
            _finding(
                "documentation_only_change",
                None,
                "Only documentation changed for this implementation workstream.",
                "Implement the required production source changes before reporting completion.",
            )
        )
    if paths and not buckets["production"] and buckets["configuration"] and not buckets["test"]:
        findings.append(
            _finding(
                "configuration_only_change",
                None,
                "Only configuration changed for this implementation workstream.",
                "Implement the required production source changes before reporting completion.",
            )
        )
    return ImplementationCompletenessResult(
        passed=not findings and not missing,
        production_files_changed=buckets["production"],
        test_files_changed=buckets["test"],
        configuration_files_changed=buckets["configuration"],
        requirements_implemented=list(dict.fromkeys(implemented)),
        requirements_not_implemented=list(dict.fromkeys(missing)),
        implementation_expectations_satisfied=list(dict.fromkeys(satisfied)),
        findings=findings,
        production_diff_fingerprint=(
            completion.production_diff_fingerprint or _fingerprint(buckets["production"])
        ),
        test_diff_fingerprint=_fingerprint(buckets["test"]),
        category_evidence_source=evidence_source,
    )


def meaningful_progress(
    result: ImplementationCompletenessResult,
    *,
    previous_fingerprint: str | None,
    previous_test_fingerprint: str | None = None,
    blocking_configuration_resolved: bool = False,
    validation_outcome_improved: bool = False,
    source_rejected_before_commit: bool = False,
    diagnostics_changed: bool = True,
) -> tuple[bool, str]:
    """Return whether an attempt materially progressed rather than repeated a prior edit."""
    if blocking_configuration_resolved:
        return True, "resolved_blocking_configuration_issue"
    reproduced_previous_change = (
        previous_fingerprint is not None
        and result.production_diff_fingerprint == previous_fingerprint
    )
    # A reviewer that rejects an attempt over its tests is answered by editing those tests,
    # and production bytes are then correctly unchanged. Asking only about production read
    # that as a repeat: -068's backend was ended holding two test-scoped findings, having
    # rewritten its endpoint test to drive a real request through the assembled application.
    rewrote_test_source = (
        result.test_diff_fingerprint is not None
        and previous_test_fingerprint is not None
        and result.test_diff_fingerprint != previous_test_fingerprint
    )
    # A pre-commit gate rejects the change before anything is committed, so the attempt
    # reports no changed files even though source was genuinely written. Reading that as
    # "no material change" refuses the single retry that could clear the lint error the
    # gate reported, and rep-a's frontend was stopped that way three attempts in while its
    # source was being rewritten between attempts.
    #
    # But it must not be unconditional, which is what it was. AB-Feature-105 spent eleven
    # attempts on one missing import: byte-identical production source, byte-identical
    # eslint output -- `'BulkDeleteApps' is not defined` -- and this branch called every one
    # of them progress. So it now asks for one of the two things that distinguishes rep-a
    # from that: either the source changed, or the gate is saying something new.
    if source_rejected_before_commit and (diagnostics_changed or not reproduced_previous_change):
        return True, "source_rejected_before_commit_with_changed_input"
    if result.production_files_changed and not reproduced_previous_change:
        return True, "new_or_material_production_source_implementation"
    # Rewriting tests answers a reviewer that rejected an attempt over its tests, and
    # production bytes are then correctly unchanged. It does not answer a production defect:
    # with the same production source and the same diagnostic, editing the test is not
    # progress towards the thing that is broken, and -105 bought four attempts that way
    # after its validation budget was spent.
    if reproduced_previous_change and rewrote_test_source and diagnostics_changed:
        return True, "new_test_source_for_unchanged_production_implementation"
    # Checked before the test branch below, which asked only whether tests and production
    # files were both present. An attempt that returned byte-identical files therefore
    # counted as progress for as long as it carried a test, and ser-1 spent ten attempts --
    # its entire budget, in both repositories -- re-sending the same change to the same
    # gate. Reproducing the previous attempt is the definition of no progress.
    if reproduced_previous_change:
        return False, "attempt_reproduced_the_previous_production_change"
    if result.test_files_changed and result.production_files_changed:
        return True, "new_executable_tests_for_implemented_behavior"
    if validation_outcome_improved:
        return True, "validation_outcome_improved_after_relevant_change"
    if result.test_files_changed and not result.production_files_changed:
        return False, "test_only_change_while_required_production_implementation_is_missing"
    return False, "no_material_change_detected"


def _buckets(paths: list[str], completion: CodeCompletionArtifact) -> dict[str, list[str]]:
    modified_production = [
        change.path
        for change in completion.file_changes
        if change.change_type == "modified" and classify_file_change(change.path) == "production"
    ]
    values = {
        "production": list(completion.production_files_changed),
        "test": list(completion.test_files_changed),
        "configuration": list(completion.configuration_files_changed),
        "documentation": [],
        "modified_production": modified_production,
    }
    if any(values[key] for key in ("production", "test", "configuration")):
        # The model's own bucketing is a claim about its change; the path is evidence. Where
        # they disagree the claim loses, because a category the model mislabelled is one it
        # cannot be told to fix: -063's backend created `server/config/healthHistory.js`,
        # declared it production, and was refused three times for changing no configuration.
        for category in ("documentation", "configuration"):
            values[category] = [
                *values[category],
                *(path for path in paths if classify_file_change(path) == category),
            ]
        return {key: list(dict.fromkeys(value)) for key, value in values.items()}
    for path in paths:
        values[classify_file_change(path)].append(path)
    return {key: list(dict.fromkeys(value)) for key, value in values.items()}


def _contract_buckets(
    attempt_buckets: dict[str, list[str]],
    *,
    branch_paths: Sequence[str] | None,
    branch_modified_paths: Sequence[str] | None,
    branch_evidence_source: CategoryEvidenceSource | None,
) -> tuple[dict[str, list[str]], CategoryEvidenceSource]:
    """Bucket the branch's paths for satisfaction only, never for what the result reports.

    Additive over the attempt's own buckets rather than a replacement, so this can only ever
    add evidence: a first attempt, an attempt whose branch Git could not describe, and an
    attempt whose completion already carries the cumulative lineage union all answer exactly
    as they do today. The classification of a branch path is `classify_file_change`'s, not any
    model's claim, because nothing declared a bucket for a file some earlier attempt wrote.
    """
    if branch_paths is None:
        return attempt_buckets, "attempt_declared_paths"
    modified = set(branch_modified_paths or ())
    values = {key: list(value) for key, value in attempt_buckets.items()}
    for path in branch_paths:
        category = classify_file_change(path)
        values[category].append(path)
        if category == "production" and path in modified:
            values["modified_production"].append(path)
    buckets = {key: list(dict.fromkeys(value)) for key, value in values.items()}
    # The label travels with the paths. The one caller passes both together; anything that
    # passed paths without saying where they came from would be recorded as the branch read
    # it did not make, so the pair is documented here rather than defaulted apart.
    return buckets, branch_evidence_source or "branch_diff"


def _category_satisfied(category: str, buckets: dict[str, list[str]], areas: list[str]) -> bool:
    if category == "integration":
        # A file that already existed and was changed. Adding modules only leaves them
        # unreachable: a route nobody registers and a component nobody renders are the two
        # ways this platform has repeatedly shipped, or failed to ship, unfinished work.
        return bool(buckets["modified_production"])
    if category == "production":
        return bool(buckets["production"])
    if category == "test":
        return bool(buckets["test"])
    if category == "documentation":
        return bool(buckets["documentation"])
    if category == "configuration":
        return bool(buckets["configuration"])
    tokens = {
        "route": ("route", "routes", "router"),
        "controller": ("controller", "controllers"),
        "service": ("service", "services"),
        "model": ("model", "models", "schema", "schemas"),
        "migration": ("migration", "migrations"),
        "frontend_component": ("component", "components", "view", "views", "page", "pages"),
        "client": ("client", "api", "sdk"),
    }.get(category, ())
    return any(
        any(token in path.lower() for token in tokens) for path in buckets["production"]
    ) or (bool(buckets["production"]) and not areas)


def _fingerprint(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(paths)).encode()).hexdigest()


def _finding(
    code: Literal[
        "tests_only_change",
        "documentation_only_change",
        "configuration_only_change",
        "missing_required_source_change",
        "missing_required_test_change",
        "unexpected_scope_change",
    ],
    requirement_id: str | None,
    description: str,
    recommended_action: str,
) -> CompletionFinding:
    return CompletionFinding(
        code=code,
        requirement_id=requirement_id,
        description=description,
        recommended_action=recommended_action,
    )


__all__ = [
    "CategoryEvidenceSource",
    "CompletionFinding",
    "ImplementationCompletenessResult",
    "classify_file_change",
    "meaningful_progress",
    "validate_implementation_completeness",
]
