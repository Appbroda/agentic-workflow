"""The completeness gate judges the workstream's branch, not one attempt's delta.

66-. AB-Feature-201's frontend (2026-09-03) completed its first cycle properly -- eight
files including `src/utils/bulkAppCsvPrecheck.test.js` and
`src/components/apps/BulkAddAppsModal.test.js`, validation 4/4, review approved, published.
Integration review then asked for a Content-Type boundary header and an error-message style.
Each remediation attempt made its one-file production fix, and each was refused by the same
deterministic sentence -- *"The workstream requires tests but no test file changed"* -- six
times running, until the workstream wrote a test file it did not owe just to clear a gate.

The tests were on the branch the whole time. Two things put them out of the gate's reach:
`_buckets` reads the completion's own declared buckets, so a targeted attempt's empty test
list *is* the evidence; and the completion holding the tests carries
`published_after_approval`, which every lineage reader on this platform deliberately
excludes. So the branch itself has to be the evidence, and a union of lineage completions
would have found nothing -- which is what the tier ordering here is about.

What must not move, and is asserted below: the reported lists and both fingerprints keep
describing the attempt (T8), `configuration` stays the only advisory category (T6), a
workstream with no tests anywhere still fails every attempt (T2, T3), and the evidence
source is always on the record (T9).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import HttpUrl

from agents.shared.contracts import (
    ARTIFACT_FILENAMES,
    artifact_id_matches_lineage,
    create_artifact,
)
from artifacts.schemas import CodeCompletionArtifact, RepositoryWorkstreamPlan, ReviewArtifact
from services.cancellation import MockCancellationToken
from services.feature_runtime import (
    _previous_child_review,
    _prior_child_completions,
    _prior_child_completions_including_published,
)
from services.process_runner import AsyncioProcessRunner, ProcessResult, ProcessRunner
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.feature_models import (
    ChildWorkflowReference,
    FeatureWorkflowSnapshot,
    RepositorySpec,
)
from tests.fixtures import commit_all, node_javascript_repository, start_working_branch
from tools.implementation_completeness import (
    classify_file_change,
    meaningful_progress,
    validate_implementation_completeness,
)
from tools.lineage_base import LineageBaseRevision, branch_change_evidence
from workflow_schema import WORKFLOW_SCHEMA_VERSION

_AT = datetime(2026, 9, 3, 8, 35, 3, tzinfo=UTC)

# 201's frontend branch, read from the checkout on 2026-09-03: eight paths, two of them test
# files, all written by the approved-and-published attempt 0.
_ATTEMPT_ZERO_BRANCH = (
    "README.md",
    "public/bulk-add-apps-template.csv",
    "src/apiUtils/allapps.apiUtils.js",
    "src/apiUtils/allapps.apiUtils.test.js",
    "src/components/apps/BulkAddAppsModal.js",
    "src/components/apps/BulkAddAppsModal.test.js",
    "src/pages/AllApps.js",
    "src/utils/bulkAppCsvPrecheck.js",
    "src/utils/bulkAppCsvPrecheck.test.js",
)
# The one file every remediation attempt from 1 onwards touched, and nothing else.
_TARGETED_FIX = "src/apiUtils/allapps.apiUtils.js"
_TESTS_REQUIRED_SENTENCE = "The workstream requires tests but no test file changed."


def test_a_targeted_fix_passes_the_test_category_the_branch_already_satisfies() -> None:
    """T1 -- the 201 replay. The gate raises nothing, and the blind version raises the defect."""
    workstream = _workstream(categories=["client", "test"], tests_required=True)
    completion = _completion([(_TARGETED_FIX, "modified")], production=[_TARGETED_FIX])

    result = validate_implementation_completeness(
        workstream,
        completion,
        branch_paths=_ATTEMPT_ZERO_BRANCH,
        branch_modified_paths=(_TARGETED_FIX,),
        branch_evidence_source="branch_diff",
    )

    assert result.passed
    assert result.findings == []
    assert result.requirements_not_implemented == []
    assert result.category_evidence_source == "branch_diff"

    # The same attempt, judged the way it was judged six times in a row.
    blind = validate_implementation_completeness(workstream, completion)
    assert not blind.passed
    assert _TESTS_REQUIRED_SENTENCE in [finding.description for finding in blind.findings]
    assert blind.category_evidence_source == "attempt_declared_paths"


def test_the_published_completion_holding_the_tests_is_only_visible_to_the_new_reader() -> None:
    """T1 -- the exclusion that makes a lineage-union fix find nothing on 201's shape."""
    feature = _feature(
        [
            _persisted_completion(
                attempt=0,
                paths=["src/utils/bulkAppCsvPrecheck.js", "src/utils/bulkAppCsvPrecheck.test.js"],
                published=True,
            ),
            _persisted_completion(attempt=1, paths=[_TARGETED_FIX], published=False),
        ]
    )
    child = _child(retry_count=2)
    published = next(
        artifact
        for artifact in feature.artifacts
        if isinstance(artifact, CodeCompletionArtifact)
        and artifact.metadata.get("child_attempt") == 0
    )

    # 201's shape, asserted rather than assumed: the tests live in a published completion.
    assert published.metadata["published_after_approval"] is True
    assert [
        item.metadata["child_attempt"] for item in _prior_child_completions(feature, child)
    ] == [1]
    assert [
        item.metadata["child_attempt"]
        for item in _prior_child_completions_including_published(feature, child)
    ] == [0, 1]


def test_a_prior_review_is_renamed_back_to_child_scoped_lineage_before_the_next_attempt() -> None:
    """Root cause of Fix 5 (seam-evidence escalation) never firing in production.

    Parent persistence namespaces a review by repository -- `_with_attempt_identity`
    (`workflows/feature_workflow.py`) -- exactly as it does a code completion, but nothing
    renamed it back before `_previous_child_review` handed it to the next attempt. The
    reviewer's own lineage matcher, `artifact_id_matches_lineage`, then rejected every prior
    review on every attempt after the first, for every repository, on every feature this
    platform has ever run -- confirmed directly against AB-Feature-171's own persisted
    artifacts, whose real `artifact_id` is exactly the qualified shape built here.
    """
    feature = _feature(
        [_persisted_review(attempt=0, omitted_path="server/utils/service.util.js")]
    )

    renamed = _previous_child_review(feature, "feature-66:frontend")

    assert renamed is not None
    assert artifact_id_matches_lineage(renamed.artifact_id, ARTIFACT_FILENAMES["review"])
    assert renamed.metadata["seam_evidence_budget_omitted_paths"] == [
        "server/utils/service.util.js"
    ]


def test_a_prior_review_with_no_parseable_attempt_number_is_still_returned_unrenamed() -> None:
    """A malformed or already-plain id degrades to returning the artifact as-is, never a crash."""
    feature = _feature(
        [
            create_artifact(
                ReviewArtifact,
                workflow_id="feature-66:frontend",
                artifact_id="007_review.json",
                producer="reviewer",
                metadata={},
                payload={
                    "verdict": "changes_requested",
                    "summary": "First attempt, no attempt suffix at all.",
                    "requirement_checks": [],
                    "findings": [],
                    "architecture_assessment": "Not assessed.",
                    "security_assessment": "Not assessed.",
                    "test_coverage_assessment": "Not assessed.",
                },
            )
        ]
    )

    renamed = _previous_child_review(feature, "feature-66:frontend")

    assert renamed is not None
    assert renamed.artifact_id == "007_review.json"


def test_a_first_attempt_without_tests_still_fails_with_the_same_sentence() -> None:
    """T2 -- this change adds evidence, never leniency.

    On a first attempt the branch diff *is* the attempt's own change, so there is nothing for
    branch evidence to add and the sentence is the one it has always been.
    """
    workstream = _workstream(categories=["client"], tests_required=True)
    completion = _completion([(_TARGETED_FIX, "modified")], production=[_TARGETED_FIX])

    result = validate_implementation_completeness(
        workstream,
        completion,
        branch_paths=(_TARGETED_FIX,),
        branch_modified_paths=(_TARGETED_FIX,),
        branch_evidence_source="branch_diff",
    )

    assert not result.passed
    assert [finding.description for finding in result.findings] == [_TESTS_REQUIRED_SENTENCE]
    assert [finding.code for finding in result.findings] == ["missing_required_test_change"]
    assert result.requirements_not_implemented == ["bulk-add-apps"]


def test_a_workstream_that_never_wrote_a_test_fails_on_every_attempt() -> None:
    """T3 -- a growing branch of production-only work never satisfies `test`."""
    workstream = _workstream(categories=["client", "test"], tests_required=True)
    branch = [_TARGETED_FIX]
    for attempt in range(3):
        path = f"src/apiUtils/attempt{attempt}.js"
        branch.append(path)
        result = validate_implementation_completeness(
            workstream,
            _completion([(path, "modified")], production=[path]),
            branch_paths=tuple(branch),
            branch_modified_paths=tuple(branch),
            branch_evidence_source="branch_diff",
        )

        assert not result.passed, f"attempt {attempt} was let through"
        assert _TESTS_REQUIRED_SENTENCE in [finding.description for finding in result.findings]
        assert any(
            "required change categories: test." in finding.description
            for finding in result.findings
        )


def test_documentation_already_on_the_branch_clears_its_missing_category_finding() -> None:
    """T4 -- documentation moves with every other promised category, and stays enforced."""
    workstream = _workstream(categories=["client", "documentation"], tests_required=False)
    completion = _completion([(_TARGETED_FIX, "modified")], production=[_TARGETED_FIX])

    documented = validate_implementation_completeness(
        workstream,
        completion,
        branch_paths=(_TARGETED_FIX, "docs/bulk-add-apps.md"),
        branch_modified_paths=(_TARGETED_FIX,),
        branch_evidence_source="branch_diff",
    )
    undocumented = validate_implementation_completeness(
        workstream,
        completion,
        branch_paths=(_TARGETED_FIX, "src/pages/AllApps.js"),
        branch_modified_paths=(_TARGETED_FIX,),
        branch_evidence_source="branch_diff",
    )

    assert documented.passed
    assert documented.findings == []
    # The converse: nothing on the branch documents this, so the demand is honest rather
    # than impossible -- an attempt can satisfy it by writing one.
    assert not undocumented.passed
    assert any(
        "required change categories: documentation" in finding.description
        for finding in undocumented.findings
    )


@pytest.mark.realrepo
@pytest.mark.asyncio
async def test_the_evidence_stops_at_the_lineage_base(tmp_path: Path) -> None:
    """T5 -- a workstream inherits its own branch's history and nothing else.

    The fixture's own `test/status.test.js` is committed on the default branch *before* this
    workstream exists, so it is on the baseline side of the diff and satisfies nothing. The
    test file an attempt commits on the branch is what the gate may read -- which is the
    whole boundary, and the reason this needs a real checkout rather than a doubled runner.
    """
    workspace = tmp_path / "checkout"
    node_javascript_repository(workspace)
    start_working_branch(workspace, "ai/feature-66/frontend/bulk-add-apps")
    workstream = _workstream(categories=["client", "test"], tests_required=True)
    completion = _completion([(_TARGETED_FIX, "modified")], production=[_TARGETED_FIX])
    runner = AsyncioProcessRunner()

    inherited = await branch_change_evidence(
        workspace=workspace,
        baseline=_baseline(workspace, runner),
        process_runner=runner,
        cancellation_token=MockCancellationToken(),
    )
    assert inherited is not None
    assert inherited.paths == ()
    refused = validate_implementation_completeness(
        workstream,
        completion,
        branch_paths=inherited.paths,
        branch_modified_paths=inherited.modified_paths,
        branch_evidence_source="branch_diff",
    )

    # Now this workstream's own approved attempt commits a test on the branch.
    (workspace / "src" / "apiUtils").mkdir(parents=True, exist_ok=True)
    (workspace / "src" / "apiUtils" / "allapps.apiUtils.js").write_text(
        "module.exports = { bulkCreateApps: () => null };\n", encoding="utf-8"
    )
    (workspace / "test" / "bulk.test.js").write_text(
        "const { test } = require('node:test');\ntest('bulk', () => {});\n", encoding="utf-8"
    )
    commit_all(workspace, "attempt 0, approved and published")

    written = await branch_change_evidence(
        workspace=workspace,
        baseline=_baseline(workspace, runner),
        process_runner=runner,
        cancellation_token=MockCancellationToken(),
    )
    assert written is not None
    accepted = validate_implementation_completeness(
        workstream,
        completion,
        branch_paths=written.paths,
        branch_modified_paths=written.modified_paths,
        branch_evidence_source="branch_diff",
    )

    assert set(written.paths) == {"src/apiUtils/allapps.apiUtils.js", "test/bulk.test.js"}
    assert "test/status.test.js" not in written.paths
    assert not refused.passed
    assert _TESTS_REQUIRED_SENTENCE in [finding.description for finding in refused.findings]
    assert accepted.passed


def test_the_fallback_reader_is_scoped_to_this_child(tmp_path: Path) -> None:
    """T5 -- a sibling repository's completion describes a different checkout entirely."""
    del tmp_path
    feature = _feature(
        [
            _persisted_completion(attempt=0, paths=["tests/test_status.py"], published=True),
            _persisted_completion(
                attempt=0,
                paths=["server/tests/test_bulk.py"],
                published=True,
                workflow_id="feature-66:backend",
                repository="backend",
            ),
        ]
    )

    completions = _prior_child_completions_including_published(feature, _child(retry_count=1))

    assert [change.path for item in completions for change in item.file_changes] == [
        "tests/test_status.py"
    ]


def test_configuration_stays_advisory_with_branch_evidence_in_hand() -> None:
    """T6 -- the fix is better evidence, not a weaker demand, and no new demotion."""
    workstream = _workstream(categories=["client", "configuration"], tests_required=False)
    completion = _completion([(_TARGETED_FIX, "modified")], production=[_TARGETED_FIX])

    result = validate_implementation_completeness(
        workstream,
        completion,
        branch_paths=(_TARGETED_FIX,),
        branch_modified_paths=(_TARGETED_FIX,),
        branch_evidence_source="branch_diff",
    )

    assert result.passed
    assert result.findings == []
    assert result.configuration_files_changed == []


def test_the_declared_bucket_path_is_the_one_that_reads_branch_evidence() -> None:
    """T7 -- a completion that declares nothing takes `_buckets`'s fallback branch instead.

    201's remediation attempts declared their one production file, so the declared branch
    wins and `buckets["test"]` is that attempt's empty list. A fixture that declares nothing
    would exercise the other branch and pass while 201 stayed broken.
    """
    workstream = _workstream(categories=["client", "test"], tests_required=True)
    completion = _completion(
        [(_TARGETED_FIX, "modified"), ("src/pages/AllApps.js", "modified")],
        production=[_TARGETED_FIX],
    )

    result = validate_implementation_completeness(
        workstream,
        completion,
        branch_paths=_ATTEMPT_ZERO_BRANCH,
        branch_modified_paths=(_TARGETED_FIX,),
        branch_evidence_source="branch_diff",
    )

    # The declared branch was taken: the undeclared production path in `file_changes` is
    # absent from what the gate reports, exactly as it is today.
    assert result.production_files_changed == [_TARGETED_FIX]
    assert "src/pages/AllApps.js" not in result.production_files_changed
    assert result.passed


def test_the_reported_lists_and_fingerprints_still_describe_the_attempt() -> None:
    """T8 -- branch evidence feeds satisfaction and nothing else.

    A union in these fields would disarm the one guard that stops a workstream burning its
    budget on a repeat: -105 spent eleven attempts on one missing import, and ser-1 its
    entire budget in both repositories, re-sending the same change to the same gate.
    """
    workstream = _workstream(categories=["client", "test"], tests_required=True)
    completion = _completion([(_TARGETED_FIX, "modified")], production=[_TARGETED_FIX])

    result = validate_implementation_completeness(
        workstream,
        completion,
        branch_paths=_ATTEMPT_ZERO_BRANCH,
        branch_modified_paths=(_TARGETED_FIX,),
        branch_evidence_source="branch_diff",
    )
    blind = validate_implementation_completeness(workstream, completion)

    assert result.production_files_changed == [_TARGETED_FIX]
    assert result.test_files_changed == []
    assert result.configuration_files_changed == []
    assert result.production_diff_fingerprint == blind.production_diff_fingerprint
    assert result.test_diff_fingerprint == blind.test_diff_fingerprint
    # And the churn guard therefore behaves identically: re-sending the previous production
    # change is still not progress, however much the branch around it holds.
    assert meaningful_progress(
        result,
        previous_fingerprint=result.production_diff_fingerprint,
        previous_test_fingerprint=result.test_diff_fingerprint,
    ) == (False, "attempt_reproduced_the_previous_production_change")


def test_no_evidence_at_all_behaves_exactly_as_before_and_says_so() -> None:
    """T9 -- the attempt-only fall-through is legible rather than silent."""
    workstream = _workstream(categories=["client", "test"], tests_required=True)
    completion = _completion(
        [(_TARGETED_FIX, "modified"), ("src/utils/bulkAppCsvPrecheck.test.js", "added")],
        production=[_TARGETED_FIX],
        tests=["src/utils/bulkAppCsvPrecheck.test.js"],
    )

    result = validate_implementation_completeness(workstream, completion)

    assert result.passed
    assert result.category_evidence_source == "attempt_declared_paths"
    assert result.test_files_changed == ["src/utils/bulkAppCsvPrecheck.test.js"]


@pytest.mark.asyncio
async def test_an_unresolvable_baseline_answers_nothing_rather_than_head() -> None:
    """T9 -- `is_branch_point` false means unanswerable, so the caller must fall to tier 2.

    A diff against the `HEAD` fallback would report an approved attempt's own committed work
    as absent, which is the failure this whole change exists to stop -- so the branch read is
    refused rather than answered wrongly.
    """
    evidence = await branch_change_evidence(
        workspace=Path("/nonexistent-checkout"),
        baseline=_baseline(Path("/nonexistent-checkout"), _EmptyGitRunner()),
        process_runner=_EmptyGitRunner(),
        cancellation_token=MockCancellationToken(),
    )

    assert evidence is None


def test_the_documentation_half_of_201_is_answered_by_the_branch() -> None:
    """66-'s open question, answered from the checkout: `README.md` is on that branch.

    The spec suspected otherwise, because none of the five completions read from the
    database names a `.md` path. The branch does: `git diff --name-status <merge-base>` in
    201 FE's workspace reports `M README.md` alongside the eight source paths, and
    `classify_file_change` calls that documentation. So the documentation rejection is fixed
    by the same evidence as the test one, with no demotion of the category.
    """
    assert classify_file_change("README.md") == "documentation"
    workstream = _workstream(categories=["client", "documentation"], tests_required=False)

    result = validate_implementation_completeness(
        workstream,
        _completion([(_TARGETED_FIX, "modified")], production=[_TARGETED_FIX]),
        branch_paths=_ATTEMPT_ZERO_BRANCH,
        branch_modified_paths=(_TARGETED_FIX, "README.md"),
        branch_evidence_source="branch_diff",
    )

    assert result.passed


class _EmptyGitRunner:
    """A runner whose Git says nothing at all: no repository, or a checkout with no ancestor."""

    async def run(self, command: Any, cwd: Path, *_args: Any, **_kwargs: Any) -> ProcessResult:
        """Return a successful, empty result so only `is_branch_point` can decide."""
        del cwd
        return ProcessResult(
            command=tuple(command),
            return_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.01,
        )


def _baseline(workspace: Path, runner: ProcessRunner) -> LineageBaseRevision:
    return LineageBaseRevision(
        workspace=workspace,
        default_branch="main",
        process_runner=runner,
        cancellation_token=MockCancellationToken(),
    )


def _workstream(*, categories: list[str], tests_required: bool) -> RepositoryWorkstreamPlan:
    """201's frontend workstream, scoped to the requirement its remediation was about."""
    return RepositoryWorkstreamPlan.model_validate(
        {
            "workstream_id": "frontend",
            "repository_id": "frontend",
            "role": "frontend",
            "requirement_ids": ["bulk-add-apps"],
            "scoped_requirements": [
                {
                    "requirement_id": "bulk-add-apps",
                    "acceptance_criterion_ids": ["bulk-add-apps:ac-1"],
                    "responsibility": "implements",
                }
            ],
            "out_of_scope_requirements": [],
            "shared_requirements": [],
            "responsibilities": ["Let an admin bulk add apps from a CSV."],
            "task_ids": ["bulk-add-apps-task"],
            "dependency_workstream_ids": [],
            "contract_sections_consumed": [],
            "contract_sections_implemented": [],
            "acceptance_criteria": ["A CSV of apps can be uploaded."],
            "test_requirements": ["Run the configured test command."],
            "documentation_requirements": [],
            "expected_files_or_areas": ["src/apiUtils", "src/components/apps"],
            "required": True,
            "implementation_expectations": [
                {
                    "requirement_id": "bulk-add-apps",
                    "expected_change_categories": categories,
                    "expected_source_areas": ["src/apiUtils"],
                    "tests_required": tests_required,
                }
            ],
        }
    )


def _completion(
    changes: list[tuple[str, str]],
    *,
    production: list[str] | None = None,
    tests: list[str] | None = None,
) -> CodeCompletionArtifact:
    """A completion shaped like a targeted remediation's: its own change, declared."""
    return create_artifact(
        CodeCompletionArtifact,
        workflow_id="feature-66:frontend",
        artifact_id="006_code_completion.json",
        producer="engineer",
        metadata={},
        payload={
            "completion_status": "completed",
            "summary": "Send the multipart boundary the endpoint expects.",
            "file_changes": [
                {
                    "path": path,
                    "change_type": change_type,
                    "description": "Updated by the configured coding executor.",
                }
                for path, change_type in changes
            ],
            "validation_results": [],
            "test_coverage_percent": None,
            "remaining_work": [],
            "commit_sha": None,
            "production_files_changed": list(production or []),
            "test_files_changed": list(tests or []),
        },
    )


def _persisted_completion(
    *,
    attempt: int,
    paths: list[str],
    published: bool,
    workflow_id: str = "feature-66:frontend",
    repository: str = "frontend",
) -> CodeCompletionArtifact:
    """A prior attempt's completion, shaped the way parent persistence records one."""
    metadata: dict[str, object] = {"child_attempt": attempt}
    if published:
        metadata["published_after_approval"] = True
    return create_artifact(
        CodeCompletionArtifact,
        workflow_id=workflow_id,
        artifact_id=f"006_code_completion.{repository}.attempt-{attempt}.json",
        producer="engineer",
        metadata=metadata,
        payload={
            "completion_status": "completed",
            "summary": "The approved attempt that built this branch.",
            "file_changes": [
                {
                    "path": path,
                    "change_type": "added",
                    "description": "Updated by the configured coding executor.",
                }
                for path in paths
            ],
            "validation_results": [],
            "test_coverage_percent": None,
            "remaining_work": [],
            "commit_sha": None,
        },
    )


def _persisted_review(
    *,
    attempt: int,
    omitted_path: str,
    workflow_id: str = "feature-66:frontend",
    repository: str = "frontend",
) -> ReviewArtifact:
    """A prior attempt's review, shaped the way parent persistence records one.

    `_with_attempt_identity` (`workflows/feature_workflow.py`) rewrites the reviewer's own
    plain `007_review.attempt-N.json` id to this repository-qualified form before folding it
    into the feature's own live state -- the exact shape confirmed directly against a real
    deployment's persisted artifacts (AB-Feature-171).
    """
    return create_artifact(
        ReviewArtifact,
        workflow_id=workflow_id,
        artifact_id=f"007_review.{repository}.attempt-{attempt}.json",
        producer="reviewer",
        metadata={"seam_evidence_budget_omitted_paths": [omitted_path]},
        payload={
            "verdict": "changes_requested",
            "summary": "Blocked on an evidence-budget omission.",
            "requirement_checks": [],
            "findings": [],
            "architecture_assessment": "Not assessed.",
            "security_assessment": "Not assessed.",
            "test_coverage_assessment": "Not assessed.",
        },
    )


def _feature(
    artifacts: Sequence[CodeCompletionArtifact | ReviewArtifact],
) -> FeatureWorkflowSnapshot:
    """The parent snapshot, carrying only what a lineage reader looks at."""
    return FeatureWorkflowSnapshot.model_validate(
        {
            "feature_id": "feature-66",
            "workflow_id": "feature-66",
            "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
            "created_by_build_revision": "rev",
            "last_executor_build_revision": "rev",
            "status": FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
            "title": "Allow an admin to bulk add apps",
            "repository_specs": [
                RepositorySpec(
                    repository_id="frontend",
                    name="AB-console-admin-2.0",
                    role="frontend",
                    repository_url=HttpUrl("https://github.com/cryn3t/AB-console-admin-2.0"),
                    default_branch="master",
                )
            ],
            "artifacts": list(artifacts),
            "max_integration_review_cycles": 5,
            "max_child_review_cycles": 12,
            "max_implementation_retries": 8,
            "max_validation_retries": 4,
            "max_repository_setup_retries": 2,
            "max_contract_revision_cycles": 3,
            "execution_mode": "live",
            "created_at": _AT,
            "updated_at": _AT,
        }
    )


def _child(*, retry_count: int) -> ChildWorkflowReference:
    return ChildWorkflowReference(
        child_workflow_id="feature-66:frontend",
        repository_id="frontend",
        workstream_id="frontend",
        branch_name="ai/feature-66/frontend/bulk-add-apps",
        workspace_path="/workspaces/feature-66/frontend",
        status=ChildWorkflowStatus.RUNNING,
        retry_count=retry_count,
    )
