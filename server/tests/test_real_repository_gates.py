"""Every rejection path of the child executor, against checkouts that really run.

Audit risk P1-8, third tier (C4). The full child path -- preflight, dependency install,
baseline lint, contract projection, Engineer, the deterministic gates, the reviewer, the
commit gate and the push -- had never run end to end against a real checkout with real
subprocesses. Selection was covered; behaviour was not.

Marked `realrepo` and deselected by default, because these tests execute `npm`, `node`,
`ruff` and `pytest` for real and a default tier people skip is worse than one that does not
exist. `uv run pytest -m realrepo` runs them; CI sets `REQUIRE_REAL_REPOSITORY_TESTS=1`,
which turns a missing toolchain into a failure rather than a silent pass.

Three things every scenario asserts, because each is a contract a previous task established
and none of them was ever checked against a live gate:

* the **failure classification**, so a stopped attempt says what stopped it (`25-`);
* `current_revision`, so an attempt stopped at any gate still names the checkout it ran
  against (`26-`);
* a **non-empty diagnostic**, so what stopped it is legible to whoever reads it next.

And, per overview section 4.4, the effect on the worktree or the remote -- never that a
command was issued.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from adapters.git_adapter import GitSafetyError
from adapters.interruptible_git import InterruptibleGitService, reviewed_content_fingerprint
from adapters.llm_adapter import ImageInput, LLMResponse, MockCodingExecutor
from agents.engineer.agent import EngineerAgent, RequiredContextRefusal
from agents.engineer.publisher import require_reviewed_workspace_match
from agents.reviewer.agent import ReviewerAgent
from agents.shared.contracts import AgentArtifactError, create_artifact
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import (
    ChildWorkflowResultArtifact,
    CodeCompletionArtifact,
    IntegrationReviewArtifact,
    RepositoryExecutionPlanArtifact,
    ReviewArtifact,
    ReviewFinding,
)
from prompts.prompt_loader import PromptLoader
from services.cancellation import MockCancellationToken
from services.feature_runtime import LiveChildWorkstreamExecutor
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.external_operations import ExternalOperationType, WorkflowCheckpointBoundary
from state.feature_models import ChildWorkflowReference
from tests import real_repository_support as real_support
from tests.fixtures import commit_all, start_working_branch
from tests.fixtures.real_repositories import (
    BINARY_ASSET,
    BINARY_ASSET_BYTES,
    CREDENTIAL_DOCUMENTATION,
    CREDENTIAL_ENVIRONMENT_FILE,
    CREDENTIAL_ENVIRONMENT_TEMPLATE,
    CREDENTIAL_PEM_FILE,
    CREDENTIAL_SECRET,
    CREDENTIAL_SERVICE_ACCOUNT,
    CROSS_MODULE_DEFINITION,
    CROSS_MODULE_SYMBOL,
    DECLARED_TYPES,
    KEY_BEARING_DEFINITION,
    KEY_BEARING_SYMBOL,
    LARGE_CONSTANTS_MODULE,
    MISSING_IMPORT_DEFINITION,
    OVERSIZED_LOCKFILE,
    OVERSIZED_LOCKFILE_MANIFEST,
    UNDECLARED_SHARED_CONFIG,
    broken_lint_config_repository,
    credential_bearing_repository,
    cross_module_dependency_repository,
    key_bearing_module_repository,
    large_constants_module_source,
    large_constants_repository,
    missing_executables,
    missing_import_repository,
    no_tests_repository,
    node_npm_repository,
    node_pnpm_repository,
    oversized_lockfile_repository,
    oversized_lockfile_source,
    python_uv_repository,
    requires_real_repository_tier,
    slow_suite_repository,
    two_package_monorepo_repository,
    typechecked_repository,
    unavailable_reason,
    unnarrowable_test_repository,
)
from tests.real_repository_support import (
    REPOSITORY_ID,
    build_harness,
    contract_artifact,
    review_payload,
    run_child,
    seed_completed_clone,
    workstream_plan,
)
from tests.test_agents import (
    StaticLLMClient,
    StaticValidationTool,
    _RecordingCodingExecutor,
    agent_state,
    domain_payload,
    review_artifact,
    task_plan_artifact,
    technical_prd_artifact,
    validation_result,
)
from tests.test_live_child_executor import ScriptedLLMClient
from tools.file_tools import DEFAULT_MAX_FILE_BYTES
from tools.lint_capabilities import RepositoryLintCapabilities
from tools.reachability import NullReachabilityChecker
from tools.retry_strategy import FailureClassification
from tools.scoped_tests import ASSERTION_GUARD_OUTCOME
from tools.self_review import SELF_REVIEW_DIAGNOSTIC_PREFIX, SELF_REVIEW_SUBSTANTIVE_OUTCOME
from tools.unchanged_failure import failing_evidence_files
from tools.validation_tools import VALIDATION_TIMED_OUT
from workflows.feature_workflow import (
    FeatureWorkflowOrchestrator,
    _append_artifacts,
    _execution_artifacts,
    _with_attempt_identity,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.realrepo]


@pytest.fixture(autouse=True)
def _toolchain() -> None:
    """Refuse to certify anything on a machine that cannot run these checkouts."""
    missing = missing_executables("git", "node", "npm")
    if not missing:
        return
    if requires_real_repository_tier():
        pytest.fail(unavailable_reason(missing), pytrace=False)
    pytest.skip(unavailable_reason(missing))


# --------------------------------------------------------------------------------------
# Approval: the path everything else is a deviation from
# --------------------------------------------------------------------------------------


async def test_an_approved_change_is_committed_and_pushed_to_a_real_remote(
    tmp_path: Path,
) -> None:
    """The whole child path, with nothing about the repository doubled.

    The assertion that matters is the remote: a branch object that exists in a bare
    repository on disk is the effect a push has, and it is the one twelve consecutive live
    runs never produced.
    """
    harness = await build_harness(tmp_path, node_npm_repository)
    try:
        execution = await run_child(
            harness,
            engineer_payload=_route_change(),
            review_payload=review_payload(verdict="approved"),
        )

        assert execution.result.status == "approved"
        assert execution.result.current_revision
        # The repository's own commands ran, from the repository's own scripts.
        assert harness.runner.ran("npm", "ci") == 1
        assert harness.runner.ran("npm", "run", "lint") >= 1
        # The change is in the worktree and on the remote, not merely in a recorded argv.
        assert "recovered: true" in (harness.worktree_file("src/routes/status.js") or "")
        assert harness.branch in harness.remote_branches()
        assert execution.code_completion is not None
        assert execution.code_completion.commit_sha
    finally:
        await harness.dispose()


# --------------------------------------------------------------------------------------
# The three causes that share the preflight exit
# --------------------------------------------------------------------------------------


async def test_a_checkout_with_no_lockfile_is_blocked_before_the_engineer_is_called(
    tmp_path: Path,
) -> None:
    """Preflight blocked, from preflight's own evidence: nothing can be installed here."""
    harness = await build_harness(
        tmp_path, lambda root: node_npm_repository(root, with_lockfile=False)
    )
    try:
        execution = await run_child(harness, engineer_payload=_route_change())

        result = execution.result
        assert result.status == "failed"
        assert (
            result.failure_classification
            == FailureClassification.VALIDATION_CONFIGURATION_FAILURE.value
        )
        assert result.blocking_issues
        assert result.current_revision
        assert "NODE_LOCKFILE_NOT_FOUND" in _issue_ids(result.preflight_result)
        # Blocked before coding: no install was attempted and the worktree is untouched.
        assert harness.runner.ran("npm", "ci") == 0
        assert execution.code_completion is None
        assert harness.worktree_file("src/routes/status.js") == _committed_route()
    finally:
        await harness.dispose()


async def test_a_lint_configuration_the_manifest_never_declares_stops_the_attempt(
    tmp_path: Path,
) -> None:
    """The repository's own linter cannot start, and the engineer is not asked to fix it.

    Feature -007 spent every attempt it had rewriting source over a shared config the
    checkout never declared. The classification has to name the dependency, not the code.
    """
    harness = await build_harness(tmp_path, broken_lint_config_repository)
    try:
        execution = await run_child(harness, engineer_payload=_route_change())

        result = execution.result
        assert result.status == "failed"
        assert (
            result.failure_classification
            == FailureClassification.VALIDATION_CONFIGURATION_FAILURE.value
        )
        assert "LINT_CONFIGURATION_REQUIRES_HUMAN" in _issue_ids(result.preflight_result)
        assert result.current_revision
        assert any(UNDECLARED_SHARED_CONFIG in issue for issue in _undeclared(result))
        # The install succeeded; it is the configuration that did not resolve.
        assert harness.runner.ran("npm", "ci") == 1
        assert execution.code_completion is None
    finally:
        await harness.dispose()


async def test_a_test_command_that_cannot_finish_is_a_capacity_failure_not_a_defect(
    tmp_path: Path,
) -> None:
    """A suite that never returns has no verdict, and no rewrite of the source supplies one.

    `retry_count=0`, because only a first attempt measures a baseline -- a retry runs against
    a workspace that still holds the previous attempt's files and would report the
    engineer's defect as the repository's. Provisioning is what that field also controls, so
    the clone is journalled as already completed and the platform's own reuse path supplies
    the checkout.
    """
    harness = await build_harness(
        tmp_path,
        slow_suite_repository,
        settings_overrides={"validation_command_timeout_seconds": 1},
    )
    try:
        await seed_completed_clone(harness)

        execution = await run_child(harness, engineer_payload=_route_change(), retry_count=0)

        result = execution.result
        assert result.status == "failed"
        assert (
            result.failure_classification
            == FailureClassification.VALIDATION_CONFIGURATION_FAILURE.value
        )
        assert "BASELINE_REQUIRED_VALIDATION_FAILED_1" in _issue_ids(result.preflight_result)
        assert result.current_revision
        # The distinction the classification exists for: the command did not fail, it never
        # finished, and that is recorded as capacity rather than as a source defect.
        assert VALIDATION_TIMED_OUT in str(result.preflight_result)
        assert execution.code_completion is None
    finally:
        await harness.dispose()


# --------------------------------------------------------------------------------------
# The gates between the Engineer and the reviewer
# --------------------------------------------------------------------------------------


async def test_the_commit_gate_rejects_a_change_the_repositorys_own_linter_refuses(
    tmp_path: Path,
) -> None:
    """The repository's lint script is the gate, and it runs for real against the change."""
    harness = await build_harness(tmp_path, node_npm_repository)
    try:
        execution = await run_child(harness, engineer_payload=_route_change(with_var=True))

        result = execution.result
        assert result.status == "failed"
        assert (
            result.failure_classification == FailureClassification.VALIDATION_SOURCE_FAILURE.value
        )
        assert result.metadata["source_validation_rejected"] is True
        assert result.current_revision
        assert any("no-var" in issue for issue in result.blocking_issues)
        # Rejected before the commit: nothing reached the remote, and the branch is unmoved.
        assert harness.branch not in harness.remote_branches()
        assert execution.review is None
    finally:
        await harness.dispose()


async def test_a_module_nothing_refers_to_is_rejected_before_review(tmp_path: Path) -> None:
    """The wiring gate, against a real checkout whose other files really do not import it."""
    harness = await build_harness(tmp_path, node_npm_repository)
    try:
        execution = await run_child(harness, engineer_payload=_unwired_module())

        result = execution.result
        assert result.status == "failed"
        assert result.failure_classification == FailureClassification.IMPLEMENTATION_MISSING.value
        assert result.metadata["deterministic_gate"] is True
        assert result.metadata["wiring_repair"]["added_path"] == ("src/routes/status_formatter.js")
        assert result.metadata["wiring_repair"]["target_path"]
        assert result.current_revision
        assert result.blocking_issues
        assert harness.branch not in harness.remote_branches()
    finally:
        await harness.dispose()


async def test_replacing_a_file_wholesale_is_rejected_before_review(tmp_path: Path) -> None:
    """The rewrite gate reads the checkout's own diff, so the checkout has to be real."""
    harness = await build_harness(tmp_path, node_npm_repository)
    try:
        execution = await run_child(harness, engineer_payload=_wholesale_rewrite())

        result = execution.result
        assert result.status == "failed"
        assert result.failure_classification == FailureClassification.IMPLEMENTATION_MISSING.value
        assert result.metadata["deterministic_gate"] is True
        assert result.current_revision
        assert any(
            "rewrit" in issue.lower() or "removed" in issue.lower()
            for issue in result.blocking_issues
        )
        assert harness.branch not in harness.remote_branches()
    finally:
        await harness.dispose()


async def test_an_attempt_that_answers_with_tests_alone_fails_the_completeness_check(
    tmp_path: Path,
) -> None:
    """A production expectation is not satisfied by a test file, however good the test."""
    harness = await build_harness(tmp_path, node_npm_repository)
    try:
        execution = await run_child(harness, engineer_payload=_tests_only())

        result = execution.result
        assert result.status == "failed"
        assert result.failure_classification == FailureClassification.IMPLEMENTATION_MISSING.value
        assert result.metadata["deterministic_gate"] is True
        assert result.current_revision
        assert result.blocking_issues
        assert result.production_files_changed == []
        assert harness.branch not in harness.remote_branches()
    finally:
        await harness.dispose()


async def test_editing_the_generated_contract_becomes_a_change_request_not_a_rejection(
    tmp_path: Path,
) -> None:
    """The contract is read-only to a child, and disagreeing with it is a decision to make."""
    harness = await build_harness(tmp_path, node_npm_repository)
    try:
        execution = await run_child(
            harness,
            engineer_payload=_route_change(
                extra_files=[{"path": "openapi.yaml", "content": "openapi: 3.1.0\npaths: {}\n"}]
            ),
            contract=contract_artifact(openapi_document=_openapi_document()),
        )

        result = execution.result
        assert result.status == "waiting_for_contract_change"
        assert execution.contract_change_request is not None
        assert result.current_revision
        assert result.blocking_issues
        assert harness.branch not in harness.remote_branches()
    finally:
        await harness.dispose()


async def test_a_reviewer_that_requests_changes_stops_the_attempt_with_what_to_fix(
    tmp_path: Path,
) -> None:
    """The reviewer's verdict decides, and its findings are what the next attempt is given."""
    harness = await build_harness(tmp_path, node_npm_repository)
    try:
        execution = await run_child(
            harness,
            engineer_payload=_route_change(),
            review_payload=review_payload(
                verdict="changes_requested",
                findings=[
                    {
                        "finding_id": "review-1",
                        "severity": "high",
                        "title": "The status payload omits the uptime.",
                        "description": "The status route does not report the uptime.",
                        "recommendation": "Include uptime in the status payload.",
                        "file_path": "src/routes/status.js",
                        "line_number": 2,
                        "finding_category": "code_quality",
                    }
                ],
            ),
        )

        result = execution.result
        assert result.status == "failed"
        assert result.failure_classification
        assert result.current_revision
        assert any("uptime" in issue for issue in result.blocking_issues)
        assert harness.branch not in harness.remote_branches()
    finally:
        await harness.dispose()


# --------------------------------------------------------------------------------------
# Bounded review scope, measured against the same checkouts every other scenario uses
# --------------------------------------------------------------------------------------


def _unscoped_finding() -> dict[str, Any]:
    """A correct senior-engineer critique that names nothing this workstream was asked for.

    Taken from the production shape of `review_scope_failure`: a real defect in code the
    attempt touched, deriving from no scoped requirement, no contract section, no failed
    command and no implementation expectation.
    """
    return {
        "finding_id": "review-scope-1",
        "severity": "high",
        "title": "The upsert does not reread the winner on a duplicate key",
        "description": (
            "ensureRetryState performs findOneAndUpdate with upsert but does not catch a "
            "duplicate-key result and reread the winner."
        ),
        "recommendation": "Catch the duplicate-key result and reread the winning document.",
        "file_path": "src/routes/status.js",
        "line_number": 2,
        "finding_category": "code_quality",
    }


async def test_an_unscoped_rejection_is_published_for_a_person_when_scope_is_bounded(
    tmp_path: Path,
) -> None:
    """The behaviour change, as an effect on a real remote.

    With the switch off this is `test_a_reviewer_that_requests_changes_stops_the_attempt_
    with_what_to_fix` above, byte for byte: the finding blocks and the bare remote holds
    nothing. With it on, the same finding is published in the pull-request body of work that
    is otherwise complete, and a person answers it instead of a fourth rewrite of the file.
    """
    harness = await build_harness(
        tmp_path, node_npm_repository, settings_overrides={"bounded_review_scope": True}
    )
    try:
        execution = await run_child(
            harness,
            engineer_payload=_route_change(),
            review_payload=review_payload(
                verdict="changes_requested", findings=[_unscoped_finding()]
            ),
        )

        result = execution.result
        assert result.status == "approved"
        assert result.blocking_issues == []
        assert result.advisory_findings == [_unscoped_finding()["description"]]
        assert result.advisory_review_verdict == "changes_requested"
        assert result.review_finding_counts is not None
        assert result.review_finding_counts.findings_untraceable == 1
        assert result.current_revision
        # The effect: the change is in the worktree and the branch exists on the remote.
        assert "recovered: true" in (harness.worktree_file("src/routes/status.js") or "")
        assert harness.branch in harness.remote_branches()
    finally:
        await harness.dispose()


async def test_a_scoped_rejection_still_stops_the_attempt_when_scope_is_bounded(
    tmp_path: Path,
) -> None:
    """Bounding what blocks must not narrow the requirement the workstream was given."""
    harness = await build_harness(
        tmp_path, node_npm_repository, settings_overrides={"bounded_review_scope": True}
    )
    try:
        scoped = {
            **_unscoped_finding(),
            "finding_category": "requirement",
            "repository_id": "backend",
            "requirement_id": "backend-status-api",
            "responsibility": "implements",
            "validated_revision": "unvalidated",
            "evidence": "The scoped route does not reread the winner.",
        }
        execution = await run_child(
            harness,
            engineer_payload=_route_change(),
            review_payload=review_payload(verdict="changes_requested", findings=[scoped]),
        )

        result = execution.result
        assert result.status == "failed"
        assert result.blocking_issues == [scoped["description"]]
        assert result.advisory_findings == []
        assert result.failure_classification
        assert result.current_revision
        assert harness.branch not in harness.remote_branches()
    finally:
        await harness.dispose()


async def test_a_deterministic_gate_still_blocks_an_approval_when_scope_is_bounded(
    tmp_path: Path,
) -> None:
    """A finding the platform injected names no requirement and must still block.

    The same repository and the same approving review as
    `test_a_repository_with_no_test_script_says_so_rather_than_inventing_one`. What could not
    be checked at all is not an opinion a person can be handed instead.
    """
    harness = await build_harness(
        tmp_path, no_tests_repository, settings_overrides={"bounded_review_scope": True}
    )
    try:
        execution = await run_child(
            harness,
            engineer_payload=_route_change(),
            review_payload=review_payload(verdict="approved"),
        )

        result = execution.result
        assert result.status == "failed"
        assert (
            result.failure_classification == FailureClassification.TEST_INFRASTRUCTURE_MISSING.value
        )
        assert any("TEST_COMMAND_NOT_CONFIGURED" in issue for issue in result.blocking_issues)
        assert result.advisory_review_verdict is None
        assert harness.branch not in harness.remote_branches()
    finally:
        await harness.dispose()


async def test_reproducing_the_committed_revision_exactly_is_refusable_not_a_fault(
    tmp_path: Path,
) -> None:
    """An empty commit is an outcome the retry policy already knows how to handle."""
    harness = await build_harness(tmp_path, node_npm_repository)
    try:
        execution = await run_child(
            harness,
            engineer_payload={
                "summary": "Confirm the existing status route.",
                "files": [{"path": "src/routes/status.js", "content": _committed_route()}],
            },
            review_payload=review_payload(verdict="approved"),
        )

        result = execution.result
        assert result.status == "failed"
        assert result.failure_classification == FailureClassification.IMPLEMENTATION_MISSING.value
        assert result.blocking_issues
        assert result.current_revision
        assert harness.branch not in harness.remote_branches()
    finally:
        await harness.dispose()


# --------------------------------------------------------------------------------------
# Detection, not assumption
# --------------------------------------------------------------------------------------


async def test_a_pnpm_lockfile_selects_pnpm_and_nothing_else_does(tmp_path: Path) -> None:
    """The only difference from the npm checkout is the lockfile, and it decides everything."""
    if missing := missing_executables("pnpm"):
        _skip_or_fail(missing)
    harness = await build_harness(tmp_path, node_pnpm_repository)
    try:
        execution = await run_child(
            harness,
            engineer_payload=_route_change(),
            review_payload=review_payload(verdict="approved"),
        )

        assert execution.result.status == "approved"
        assert _preflight(execution.result)["package_manager"] == "pnpm"
        # Chosen from the checkout and then actually executed: no npm process ever ran.
        assert harness.runner.ran("pnpm", "install", "--frozen-lockfile") == 1
        assert harness.runner.ran("npm", "ci") == 0
        assert harness.runner.ran("pnpm", "run", "lint") >= 1
    finally:
        await harness.dispose()


async def test_a_python_checkout_runs_python_tooling_and_never_a_package_manager(
    tmp_path: Path,
) -> None:
    """A pyproject declaring ruff and pytest plans and runs exactly those."""
    if missing := missing_executables("ruff", "pytest"):
        _skip_or_fail(missing)
    harness = await build_harness(tmp_path, python_uv_repository)
    try:
        execution = await run_child(
            harness,
            engineer_payload=_python_change(),
            review_payload=review_payload(verdict="approved"),
            workstream=workstream_plan(
                expected_source_areas=["app"], expected_change_categories=["production"]
            ),
        )

        assert execution.result.status == "approved"
        assert _preflight(execution.result)["package_manager"] is None
        assert harness.runner.ran("ruff", "check", ".") >= 1
        assert harness.runner.ran("npm", "ci") == 0
        assert "app/status_feed.py" in harness.tracked_paths()
    finally:
        await harness.dispose()


async def test_each_monorepo_package_is_validated_in_its_own_directory(tmp_path: Path) -> None:
    """A correct command in the wrong directory is the monorepo failure, so assert the cwd.

    Asserted from the directories the process runner actually used, not from the plan: the
    plan naming `web` and the command running at the checkout root are indistinguishable to
    a test that reads the plan.
    """
    if missing := missing_executables("ruff", "pytest"):
        _skip_or_fail(missing)
    harness = await build_harness(
        tmp_path,
        two_package_monorepo_repository,
        settings_overrides={"validation_command_timeout_seconds": 120},
    )
    try:
        await seed_completed_clone(harness)

        execution = await run_child(
            harness,
            engineer_payload=_monorepo_change(),
            review_payload=review_payload(verdict="approved"),
            workstream=workstream_plan(
                expected_source_areas=["service/app"], expected_change_categories=["production"]
            ),
            retry_count=0,
        )

        assert execution.result.status == "approved"
        # Every npm process ran in the Node package; every Python tool in the Python one.
        assert harness.runner.directories_for("npm", relative_to=harness.workspace) == {"web"}
        assert harness.runner.directories_for("ruff", relative_to=harness.workspace) <= {
            "service",
            ".",
        }
        assert "service" in harness.runner.directories_for("ruff", relative_to=harness.workspace)
        assert harness.runner.directories_for("pytest", relative_to=harness.workspace) == {
            "service"
        }
    finally:
        await harness.dispose()


async def test_a_repository_with_no_test_script_says_so_rather_than_inventing_one(
    tmp_path: Path,
) -> None:
    """Missing test infrastructure is named as such, and no test command is invented for it.

    Not a repository defect and not an engineer defect: a change here cannot be validated,
    and the platform says so in its own classification rather than reporting the absence as
    a failing suite. An approving review does not override it, which is the point -- the
    reviewer's verdict is about the code, and this is about what could be checked at all.
    """
    harness = await build_harness(tmp_path, no_tests_repository)
    try:
        execution = await run_child(
            harness,
            engineer_payload=_route_change(),
            review_payload=review_payload(verdict="approved"),
        )

        result = execution.result
        assert result.status == "failed"
        assert (
            result.failure_classification == FailureClassification.TEST_INFRASTRUCTURE_MISSING.value
        )
        assert result.current_revision
        assert any("TEST_COMMAND_NOT_CONFIGURED" in issue for issue in result.blocking_issues)
        warnings = {item["issue_id"] for item in _preflight(result)["warnings"]}
        assert "TEST_COMMAND_NOT_CONFIGURED" in warnings
        # Nothing invented a test runner the repository never configured.
        assert harness.runner.ran("npm", "run", "test") == 0
    finally:
        await harness.dispose()


# --------------------------------------------------------------------------------------
# What the Engineer is shown: the module its own code imports
#
# From AB-Feature-170, not from the audit. Four attempts wrote against
# `appValidation.addAppSchema` without ever being shown the file defining it -- thirty-five
# files delivered, two hundred and sixty-one omitted, and the definition one resolved
# specifier away in every one of them.
#
# Every assertion below is on the text the coding model was handed. The defect stayed
# invisible for as long as it did because what a selector returned in isolation was checked
# and what was delivered never was.
# --------------------------------------------------------------------------------------


async def test_the_module_an_assigned_file_imports_is_delivered_to_the_model(
    tmp_path: Path,
) -> None:
    """The definition of a symbol the assigned file imports must reach the prompt.

    The previous attempt's diff cannot supply this and never could. `_diff_added_source`
    reads added lines only -- deliberately, so a removed import does not spend the budget --
    and the import here is committed, so an attempt that starts *using* the module adds no
    import line for it. AB-Feature-170's route file had imported that module since April.

    So the source that has to answer is the checkout copy of the file the plan assigned,
    read ahead of the six-module bootstrap the attempt also touched.
    """
    root = cross_module_dependency_repository(tmp_path / "checkout")
    executor = _RecordingCodingExecutor({"src/routes/listing.js": _listing_using_envelope()})

    await _run_engineer(
        root,
        executor,
        assigned=["src/routes/listing.js", "src"],
        # A review that rejected the attempt without naming any module symbol, so the only
        # thing that can deliver the definition here is what the assigned file imports.
        review=_blocking_review("The listing endpoint returns no cursor for an empty page."),
        previous_attempt_diff=_bootstrap_diff(),
        retry_count=1,
    )

    assert CROSS_MODULE_DEFINITION in _delivered_paths(executor)
    assert "pageBody" in _delivered_text(executor), (
        "the definition has to arrive as content, not as an inventory path"
    )


async def test_a_diagnostic_naming_a_symbol_puts_its_module_in_the_next_context(
    tmp_path: Path,
) -> None:
    """`X.Y is undefined` is the sharpest signal there is about what to show next.

    Isolated from the import ordering on purpose: the bootstrap module is assigned first
    here, so its six imports take the whole budget and nothing the change depends on is
    reached by that route. The only thing that can rescue the definition is the reviewer
    having named `envelope.pageBody`, resolved through the import that binds `envelope`.
    """
    root = cross_module_dependency_repository(tmp_path / "checkout")
    executor = _RecordingCodingExecutor({"src/routes/listing.js": _listing_using_envelope()})

    await _run_engineer(
        root,
        executor,
        assigned=["src/app.js", "src/routes/listing.js"],
        review=_blocking_review(
            f"listing.js reads {CROSS_MODULE_SYMBOL}.pageBody and calls .describe() on it. "
            "The required test command shows that value is undefined at module load."
        ),
        retry_count=1,
    )

    assert CROSS_MODULE_DEFINITION in _delivered_paths(executor)


async def test_a_first_attempt_can_know_what_its_assigned_file_imports(
    tmp_path: Path,
) -> None:
    """The boundary section 2.1 asks about, encoded rather than left folkloric.

    A first attempt has no previous diff and no review, and for a *new* module it writes
    there is genuinely nothing to go on. But the files the plan assigns already exist in the
    checkout, and their imports are readable before anything is written -- so the answer to
    "can attempt 1 know?" is yes for every dependency the assigned files already have, and
    no only for dependencies the attempt itself introduces.

    This matters because attempt 1 sets the shape every later attempt inherits.
    """
    root = cross_module_dependency_repository(tmp_path / "checkout")
    executor = _RecordingCodingExecutor({"src/routes/listing.js": _listing_using_envelope()})

    await _run_engineer(
        root,
        executor,
        assigned=["src/routes/listing.js", "src"],
        retry_count=0,
    )

    assert CROSS_MODULE_DEFINITION in _delivered_paths(executor)


async def test_a_change_with_no_cross_module_dependency_is_shown_no_more_files(
    tmp_path: Path,
) -> None:
    """A targeted selector that quietly became a bigger one is not a fix.

    The assigned file here imports nothing, and the diagnostic names a dotted filename that
    binds to no import in this checkout. Both new signals must therefore contribute exactly
    nothing: the same number of files as the control run, and not one module forced in
    behind them.

    `src/platform/` is what proves the second half. Those six modules are reachable only by
    being imported -- nothing about a listing endpoint ranks them, and ninety subject-named
    modules sit ahead of them in the budget -- so if any appears here, the import signal
    collected for a change that has no cross-module dependency at all.
    """
    root = cross_module_dependency_repository(tmp_path / "checkout")
    control = _RecordingCodingExecutor({"src/support/envelope.js": _leaf_module()})
    await _run_engineer(root, control, assigned=[CROSS_MODULE_DEFINITION], retry_count=0)

    root = cross_module_dependency_repository(tmp_path / "second")
    measured = _RecordingCodingExecutor({"src/support/envelope.js": _leaf_module()})
    await _run_engineer(
        root,
        measured,
        assigned=[CROSS_MODULE_DEFINITION],
        review=_blocking_review(
            "helpers.spec.js fails at line 4 and package.json declares no runner for it."
        ),
        retry_count=1,
    )

    delivered = _delivered_paths(measured)
    assert len(delivered) == len(_delivered_paths(control))
    assert [path for path in delivered if path.startswith("src/platform/")] == []


# --------------------------------------------------------------------------------------
# Key material, and the prompt that carried it
#
# AB-Feature-170 sent four credential files from its checkout to a provider as content, on
# five engineer calls across two features. Path-based tests did not catch a path-based bug
# for as long as it existed, because they asserted against the rule and never against what
# was delivered. Every assertion here is on the text the coding model was handed.
# --------------------------------------------------------------------------------------


async def test_no_key_material_reaches_the_engineers_prompt_whatever_the_file_is_called(
    tmp_path: Path,
) -> None:
    """The security fix, asserted where the leak happened: in the delivered prompt.

    Two failures in one checkout. `mailer.env` ends with `.env` rather than beginning with
    it, which is how `sendgrid.env`'s live key passed a rule that matched only the prefix.
    The service-account key and the PEM are named so that no filename rule could ever see
    them, and they are withheld by their bytes instead.
    """
    root = credential_bearing_repository(tmp_path / "checkout")
    executor = _RecordingCodingExecutor({"src/routes/status.js": _committed_route()})

    await _run_engineer(root, executor, assigned=["src/routes/status.js"], retry_count=0)

    delivered = _delivered_paths(executor)
    assert CREDENTIAL_SECRET not in _delivered_text(executor), (
        "no key material may reach a model request, by any route"
    )
    assert CREDENTIAL_ENVIRONMENT_FILE not in delivered
    assert CREDENTIAL_SERVICE_ACCOUNT not in delivered
    assert CREDENTIAL_PEM_FILE not in delivered


async def test_the_template_and_the_ordinary_configuration_still_reach_the_engineer(
    tmp_path: Path,
) -> None:
    """The other half, and the one that is easy to lose.

    A scanner with false positives silently starves the Engineer of context, which is the
    defect the previous task spent itself fixing, so the files that must survive are asserted
    as specifically as the files that must not. `.env.example` carries the same variable
    names as the file next to it and holds no key; `tsconfig.json` is ordinary configuration
    that a blunter content rule would have no reason to touch and a blunter path rule would.
    """
    root = credential_bearing_repository(tmp_path / "checkout")
    executor = _RecordingCodingExecutor({"src/routes/status.js": _committed_route()})

    await _run_engineer(root, executor, assigned=["src/routes/status.js"], retry_count=0)

    delivered = _delivered_paths(executor)
    assert CREDENTIAL_ENVIRONMENT_TEMPLATE in delivered
    assert "tsconfig.json" in delivered
    assert "package.json" in delivered


async def test_documentation_quoting_a_pem_header_is_withheld_and_that_is_the_choice(
    tmp_path: Path,
) -> None:
    """The known false positive, encoded rather than left to be rediscovered.

    A file explaining how to rotate a key quotes the PEM header in a fenced block and is
    withheld for it. That is deliberate: withholding a document is a much smaller harm than
    sending a key, and the marker is kept unambiguous rather than made clever. This test
    exists so the cost is visible and a later reader can weigh it, not because it is desired.
    """
    root = credential_bearing_repository(tmp_path / "checkout")
    executor = _RecordingCodingExecutor({"src/routes/status.js": _committed_route()})

    await _run_engineer(root, executor, assigned=["src/routes/status.js"], retry_count=0)

    assert CREDENTIAL_DOCUMENTATION not in _delivered_paths(executor)


async def test_a_content_caught_key_file_is_still_named_in_the_path_inventory(
    tmp_path: Path,
) -> None:
    """The deliberate decision about the path-only inventory, encoded so it stays deliberate.

    A filename discloses something -- a service-account key is named for its project -- so
    this is a real cost and not a free choice. It is taken anyway, for two reasons. Catching
    it here means reading every one of a thousand inventory candidates where the snapshot
    reads sixty, which is the second pass over the checkout the design rules out. And the
    previous task established that telling an attempt a file exists is most of what stops it
    inventing one.

    What is never traded is the content. The name is listed; the bytes are not sent.
    """
    root = credential_bearing_repository(tmp_path / "checkout")
    executor = _RecordingCodingExecutor({"src/routes/status.js": _committed_route()})

    await _run_engineer(root, executor, assigned=["src/routes/status.js"], retry_count=0)

    snapshot = json.loads(executor.calls[0]["input_text"])["repository_context"]
    assert CREDENTIAL_SERVICE_ACCOUNT in snapshot["file_inventory"]
    assert CREDENTIAL_SERVICE_ACCOUNT not in _delivered_paths(executor)
    # A name the path rule does catch is gone from the inventory too, which is the existing
    # behaviour rather than a second decision: the rule that screens the snapshot screens it.
    assert CREDENTIAL_ENVIRONMENT_FILE not in snapshot["file_inventory"]


async def test_a_touched_service_account_key_is_redacted_for_the_reviewer_not_dropped(
    tmp_path: Path,
) -> None:
    """The reviewer's policy is deliberately not the engineer's, and it has to survive.

    A credential-shaped file the change itself touched is evidence: the reviewer must be able
    to see that the attempt wrote it, and a person must be told the review did not read it.
    So the path stays, the bytes do not, and the limitation is recorded -- which is what makes
    the verdict a human decision rather than another retry.

    The file here is caught by content, not by name, so this also proves the reviewer's own
    content check runs and does not merely inherit the engineer's.
    """
    root = credential_bearing_repository(tmp_path / "checkout")
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={
                    CREDENTIAL_SERVICE_ACCOUNT: (
                        '{"type": "service_account", "private_key": '
                        f'"-----BEGIN PRIVATE KEY-----\\n{CREDENTIAL_SECRET}\\n"}}\n'
                    )
                }
            ),
        ).run(agent_state(root, [task_plan_artifact()]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))

    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    ).run(agent_state(root, [technical_prd_artifact(), task_plan_artifact(), completion]))

    assert CREDENTIAL_SECRET not in client.calls[0][1]
    evidence = json.loads(client.calls[0][1])["workspace_change_evidence"]
    entry = next(item for item in evidence["files"] if item["path"] == CREDENTIAL_SERVICE_ACCOUNT)
    assert entry["content"] == "[CONTENT WITHHELD]"
    assert "sensitive_content" in evidence["limitations"]
    assert update["artifacts"][0].metadata["manual_review_required"] is True


async def test_the_live_commit_service_refuses_to_commit_key_material(tmp_path: Path) -> None:
    """The same rule on the path that writes to a repository, driven through the real service.

    Nothing has been committed in production -- this is preventive -- but the gate is worth
    proving against real Git rather than the validator alone, because `InterruptibleGitService`
    is what the live executor uses and a guard that holds on only one of the two commit
    services would read as coverage without being it.

    Asserted on the effect: `HEAD` does not move and the file stays uncommitted. `mailer.env`
    is refused by name, `analytics-...json` and `release-signing.txt` by their bytes.
    """
    root = credential_bearing_repository(tmp_path / "checkout")
    service = InterruptibleGitService(
        default_branch="main",
        environment={},
        cancellation_token=MockCancellationToken(),
        workspace_root=tmp_path,
    )
    refused = (CREDENTIAL_ENVIRONMENT_FILE, CREDENTIAL_SERVICE_ACCOUNT, CREDENTIAL_PEM_FILE)
    for name in refused:
        (root / name).write_text(
            (root / name).read_text(encoding="utf-8") + "\n# rotated\n", encoding="utf-8"
        )
    head = _git_head(root)

    for name in refused:
        with pytest.raises(GitSafetyError):
            await service.commit(root, f"Update {name}", files=[name])

    assert _git_head(root) == head, "a refused commit must not advance the branch"
    assert set(refused) <= set(_git_uncommitted(root)), (
        "the refused files must remain uncommitted rather than be staged by a later commit"
    )


async def test_the_live_commit_service_still_commits_a_template_and_ordinary_source(
    tmp_path: Path,
) -> None:
    """The half that is easy to lose: a gate that refuses everything has stopped no leak.

    `.env.example` holds the same variable names as the file next to it and no key, and it is
    committed deliberately by the repositories that carry it. If this stops being committable
    the platform can no longer make the change a person asked for.
    """
    root = credential_bearing_repository(tmp_path / "checkout")
    service = InterruptibleGitService(
        default_branch="main",
        environment={},
        cancellation_token=MockCancellationToken(),
        workspace_root=tmp_path,
    )
    committable = [CREDENTIAL_ENVIRONMENT_TEMPLATE, "tsconfig.json"]
    for name in committable:
        (root / name).write_text((root / name).read_text(encoding="utf-8") + "\n", encoding="utf-8")
    head = _git_head(root)

    await service.commit(root, "Refresh the environment template", files=committable)

    assert _git_head(root) != head
    assert not set(committable) & set(_git_uncommitted(root))


async def test_the_live_commit_service_commits_a_binary_asset(tmp_path: Path) -> None:
    """A file that is not text is not a file that cannot be read, and it has to commit.

    Nothing in any tier ever committed a binary, which is how the gate came to refuse every
    one of them with every gate green. A frontend feature that adds a logo, an icon or a
    font stages bytes no decoder accepts, and the refusal it got claimed the file could not
    be read -- untrue, and unactionable for whoever read it. Both key-material markers are
    ASCII, so a PNG carries neither.

    Asserted on the effect, per overview section 4.4: `HEAD` moves and the bytes are in the
    commit, read back out of the object store rather than off the worktree, so this cannot
    pass on a file that was merely left lying next to the repository.
    """
    root = credential_bearing_repository(tmp_path / "checkout")
    asset = root / BINARY_ASSET
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(BINARY_ASSET_BYTES)
    service = InterruptibleGitService(
        default_branch="main",
        environment={},
        cancellation_token=MockCancellationToken(),
        workspace_root=tmp_path,
    )
    head = _git_head(root)

    await service.commit(root, "Add the service logo", files=[BINARY_ASSET])

    assert _git_head(root) != head, "a binary asset must be committable"
    assert BINARY_ASSET not in set(_git_uncommitted(root))
    assert _git_committed_bytes(root, BINARY_ASSET) == BINARY_ASSET_BYTES


async def test_a_binary_is_refused_only_when_it_genuinely_cannot_be_read(
    tmp_path: Path,
) -> None:
    """The pair that proves the distinction, since letting it blur is what broke the gate.

    `release-signing.txt` is a PEM behind an innocuous name: it decodes, it carries a marker,
    and it is refused by its bytes. A directory staged as a file is a genuine `OSError` and
    is refused too. What is between them -- bytes that are simply not text -- is not a
    refusal, and the same PNG that commits above is committed here after both refusals to
    show the gate did not simply stop refusing.

    A directory rather than a permissions bit, because a suite that happens to run as root
    would quietly stop testing anything.
    """
    root = credential_bearing_repository(tmp_path / "checkout")
    (root / BINARY_ASSET).parent.mkdir(parents=True, exist_ok=True)
    (root / BINARY_ASSET).write_bytes(BINARY_ASSET_BYTES)
    unreadable = "public/fonts"
    (root / unreadable).mkdir(parents=True)
    service = InterruptibleGitService(
        default_branch="main",
        environment={},
        cancellation_token=MockCancellationToken(),
        workspace_root=tmp_path,
    )
    head = _git_head(root)

    with pytest.raises(GitSafetyError, match="contents are key material"):
        await service.commit(root, "Rotate the signing key", files=[CREDENTIAL_PEM_FILE])
    with pytest.raises(GitSafetyError, match="cannot be read"):
        await service.commit(root, "Add the font directory", files=[unreadable])

    assert _git_head(root) == head, "a refused commit must not advance the branch"

    await service.commit(root, "Add the service logo", files=[BINARY_ASSET])

    assert _git_head(root) != head


def _git_committed_bytes(root: Path, path: str) -> bytes:
    """Read a path's bytes out of the commit itself, not off the worktree next to it."""
    return subprocess.run(
        ("git", "show", f"HEAD:{path}"), cwd=root, capture_output=True, check=True
    ).stdout


def _git_head(root: Path) -> str:
    """Return the current commit, so a refusal can be asserted against the branch not moving."""
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _git_uncommitted(root: Path) -> list[str]:
    """Return the paths Git still reports as changed, read from the worktree itself."""
    status = subprocess.run(
        ("git", "status", "--porcelain"), cwd=root, capture_output=True, text=True, check=True
    ).stdout
    return [line[3:].strip() for line in status.splitlines() if line.strip()]


# --------------------------------------------------------------------------------------
# Driving the Engineer against a real checkout, and reading what it sent
# --------------------------------------------------------------------------------------


async def _run_engineer(
    root: Path,
    executor: _RecordingCodingExecutor,
    *,
    assigned: list[str],
    review: ReviewArtifact | None = None,
    previous_attempt_diff: str = "",
    retry_count: int = 0,
) -> None:
    """Run the real Engineer over a real checkout, doubling only the coding model."""
    plan = task_plan_artifact()
    scoped = plan.model_copy(
        update={
            "metadata": {
                **plan.metadata,
                "review_scope": {"expected_files_or_areas": assigned},
            }
        }
    )
    state = agent_state(root, [scoped, *([review] if review is not None else [])])
    state["retry_count"] = retry_count
    await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        previous_attempt_diff=previous_attempt_diff or None,
    ).run(state)


def _delivered_text(executor: _RecordingCodingExecutor) -> str:
    """Return everything the coding model was handed on its first call."""
    call = executor.calls[0]
    return f"{call['instructions']}\n{call['input_text']}"


def _delivered_paths(executor: _RecordingCodingExecutor) -> list[str]:
    """Return the files whose contents the model actually received, read back from the prompt.

    Parsed out of the delivered text rather than read from the artifact's metadata, so this
    asserts on what was sent and never on the platform's own account of it.
    """
    call = executor.calls[0]
    snapshot = json.loads(call["input_text"])["repository_context"]
    return [str(item["path"]) for item in snapshot["files"]]


def _listing_using_envelope() -> str:
    """The change the attempt makes: it starts using a module it already imported."""
    return (
        "const envelope = require('../support/envelope');\n"
        "const { statusRoute } = require('./status');\n\n"
        "function listingRoute(request, response) {\n"
        "  const shape = envelope.pageBody.describe();\n"
        "  response.json({ items: [], keys: shape.keys });\n"
        "}\n\n"
        "module.exports = { listingRoute, statusRoute };\n"
    )


def _leaf_module() -> str:
    """A change to a module that imports nothing."""
    return (
        "const pageBody = { describe: () => ({ keys: {} }) };\n\nmodule.exports = { pageBody };\n"
    )


def _bootstrap_diff() -> str:
    """A previous attempt whose added lines touch the broad module and never the narrow one.

    Exactly the shape AB-Feature-170's attempt 1 left behind: the bootstrap is edited to
    mount something new, and the route file's added lines use a module whose import line was
    already committed and is therefore nowhere in the diff.
    """
    return (
        "diff --git a/src/app.js b/src/app.js\n"
        "--- a/src/app.js\n"
        "+++ b/src/app.js\n"
        "@@ -1,1 +1,7 @@\n"
        "+const { config } = require('./platform/config');\n"
        "+const { logger } = require('./platform/logger');\n"
        "+const { errors } = require('./platform/errors');\n"
        "+const { sockets } = require('./platform/sockets');\n"
        "+const { auth } = require('./platform/auth');\n"
        "+const { metrics } = require('./platform/metrics');\n"
        "diff --git a/src/routes/listing.js b/src/routes/listing.js\n"
        "--- a/src/routes/listing.js\n"
        "+++ b/src/routes/listing.js\n"
        "@@ -4,3 +4,4 @@\n"
        "+  const shape = envelope.pageBody.describe();\n"
    )


def _blocking_review(description: str) -> ReviewArtifact:
    """A review that rejected the previous attempt, carrying one blocking finding."""
    approved = review_artifact()
    return approved.model_copy(
        update={
            "verdict": "changes_requested",
            "findings": [
                ReviewFinding(
                    finding_id="finding-1",
                    severity="high",
                    title="The previous attempt used a module shape it was never shown.",
                    description=description,
                    recommendation="Read the module's definition before writing against it.",
                    file_path="src/routes/listing.js",
                    line_number=4,
                    repository_id=REPOSITORY_ID,
                    requirement_id="requirement-1",
                    responsibility="implements",
                    finding_category="validation_failure",
                )
            ],
        }
    )


# --------------------------------------------------------------------------------------
# Engineer payloads, each shaped to reach exactly one gate
# --------------------------------------------------------------------------------------


def _committed_route() -> str:
    """Return the route the fixtures commit, byte for byte."""
    return (
        "function statusRoute(request, response) {\n"
        "  response.json({ status: 'ok' });\n"
        "}\n\n"
        "module.exports = { statusRoute };\n"
    )


def _route_change(
    *, with_var: bool = False, extra_files: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    """Return an ordinary in-place edit of the assigned route, plus the test it changes.

    The committed test asserts the exact payload, so a change that only edits the route
    leaves the repository's own suite failing -- and the reviewer, which runs that suite for
    real here, is right to reject it. Sending the test alongside is what an engineer would
    do, and it is what keeps this the *approval* scenario rather than another rejection.
    """
    body = (
        "  var payload = { status: 'ok', recovered: true };\n  response.json(payload);\n"
        if with_var
        else "  response.json({ status: 'ok', recovered: true });\n"
    )
    return {
        "summary": "Report recovery state from the status route.",
        "files": [
            {
                "path": "src/routes/status.js",
                "content": (
                    f"function statusRoute(request, response) {{\n{body}}}\n\n"
                    "module.exports = { statusRoute };\n"
                ),
            },
            {
                "path": "test/status.test.js",
                "content": (
                    "const { test } = require('node:test');\n"
                    "const assert = require('node:assert');\n"
                    "const { statusRoute } = require('../src/routes/status');\n\n"
                    "test('the status route answers', () => {\n"
                    "  let body = null;\n"
                    "  statusRoute({}, { json: (value) => { body = value; } });\n"
                    "  assert.deepStrictEqual(body, { status: 'ok', recovered: true });\n"
                    "});\n"
                ),
            },
            *(extra_files or []),
        ],
    }


def _unwired_module() -> dict[str, Any]:
    """Return a new module in the assigned area that nothing in the checkout refers to."""
    return {
        "summary": "Add a status formatter.",
        "files": [
            {
                "path": "src/routes/status_formatter.js",
                "content": (
                    "function formatStatus(status) {\n  return { status };\n}\n\n"
                    "module.exports = { formatStatus };\n"
                ),
            }
        ],
    }


def _wholesale_rewrite() -> dict[str, Any]:
    """Return a change that answers its requirement by discarding an existing module.

    The catalog module the fixture commits is the behaviour being thrown away. That is the
    production shape: an attempt asked for one small addition kept the thing it was asked
    for and replaced everything around it with a comment saying the rest would normally be
    there.
    """
    return {
        "summary": "Rewrite the catalog route.",
        "files": [
            {
                "path": "src/routes/catalog.js",
                "content": (
                    "function catalogRoute(request, response) {\n"
                    "  response.json({ entries: [] });\n"
                    "}\n\n"
                    "module.exports = { catalogRoute };\n"
                ),
            }
        ],
    }


def _tests_only() -> dict[str, Any]:
    """Return a change that touches no production file at all."""
    return {
        "summary": "Cover the status route.",
        "files": [
            {
                "path": "test/status.test.js",
                "content": (
                    "const { test } = require('node:test');\n"
                    "test('the status route is covered', () => {});\n"
                ),
            }
        ],
    }


def _python_change() -> dict[str, Any]:
    """Return a Python module plus its test, in the area the workstream assigned."""
    return {
        "summary": "Add the status feed.",
        "files": [
            {
                "path": "app/status_feed.py",
                "content": (
                    '"""Server status feed."""\n\n\n'
                    "def status_feed() -> dict[str, str]:\n"
                    '    """Return the current status feed."""\n'
                    '    return {"status": "ok"}\n'
                ),
            },
            {
                # Edited, not added: a new module nothing refers to is unreachable, and the
                # completeness contract is satisfied by the file that starts using it.
                "path": "app/status.py",
                "content": (
                    '"""Service status."""\n\n'
                    "from app.status_feed import status_feed\n\n\n"
                    "def server_status() -> dict[str, str]:\n"
                    '    """Return the current service status."""\n'
                    "    return status_feed()\n"
                ),
            },
            {
                "path": "tests/test_status_feed.py",
                "content": (
                    '"""Status feed tests."""\n\n'
                    "from app.status_feed import status_feed\n\n\n"
                    "def test_status_feed_reports_the_service_status() -> None:\n"
                    '    """The feed reports what the service reports."""\n'
                    '    assert status_feed() == {"status": "ok"}\n'
                ),
            },
        ],
    }


def _monorepo_change() -> dict[str, Any]:
    """Return a change inside the monorepo's Python package only, with its test."""
    return {
        "summary": "Report the service status.",
        "files": [
            {
                "path": "service/app/handler.py",
                "content": (
                    '"""Monorepo service entry point."""\n\n\n'
                    "def handler() -> str:\n"
                    '    """Return the service name."""\n'
                    '    return "service-ok"\n'
                ),
            },
            {
                "path": "service/tests/test_handler.py",
                "content": (
                    '"""Monorepo service tests."""\n\n'
                    "from app.handler import handler\n\n\n"
                    "def test_handler_names_the_service() -> None:\n"
                    '    """The handler names its own service."""\n'
                    '    assert handler() == "service-ok"\n'
                ),
            },
        ],
    }


def _openapi_document() -> dict[str, Any]:
    """Return the contract's own OpenAPI document, which the child may not change."""
    return {
        "openapi": "3.1.0",
        "info": {"title": "Server status", "version": "1.0.0"},
        "paths": {
            "/status": {
                "get": {
                    "operationId": "getStatus",
                    "responses": {"200": {"description": "The current status."}},
                }
            }
        },
    }


# --------------------------------------------------------------------------------------
# Reading what the result recorded
# --------------------------------------------------------------------------------------


def _preflight(result: Any) -> dict[str, Any]:
    """Return the preflight record every result carries, whatever stopped the attempt."""
    preflight = result.preflight_result
    assert isinstance(preflight, dict), "every attempt records the checkout it ran against"
    return preflight


def _issue_ids(preflight: Any) -> set[str]:
    """Return the identifiers of the issues that blocked this checkout."""
    if not isinstance(preflight, dict):
        return set()
    return {str(item.get("issue_id")) for item in preflight.get("blocking_issues", [])}


def _undeclared(result: Any) -> list[str]:
    """Return every shared configuration the checkout referenced but never declared."""
    return [
        str(reference)
        for item in _preflight(result).get("blocking_issues", [])
        for reference in item.get("undeclared_references", [])
    ]


def _skip_or_fail(missing: list[str]) -> None:
    """Skip locally and fail in CI, so an absent toolchain never reads as coverage."""
    if requires_real_repository_tier():
        pytest.fail(unavailable_reason(missing), pytrace=False)
    pytest.skip(unavailable_reason(missing))


# --------------------------------------------------------------------------------------
# The retry edits the attempt in place -- against a real checkout, real gate, real remote
# --------------------------------------------------------------------------------------


async def test_a_retry_after_a_gate_rejection_edits_the_attempt_in_place(tmp_path: Path) -> None:
    """AB-Feature-171's attempt-4 shape: a complete implementation with one lint error.

    Attempt one writes a route the repository's own linter rejects (`no-var`). The retry
    finds that file still in the workspace, corrects it with a single `edits` entry, and
    reaches review -- which approves -- in one attempt. The reflog is the witness that no
    reset ran, and the pushed commit is the union the reset used to be needed to protect.
    """
    harness = await build_harness(tmp_path, node_npm_repository)
    try:
        first = await run_child(
            harness,
            engineer_payload={
                "summary": "Add the metrics route and register it.",
                "files": [
                    {
                        "path": "src/routes/metrics.js",
                        "content": (
                            "var metrics = { count: 1 };\n\n"
                            "function metricsRoute(request, response) {\n"
                            "  response.json(metrics);\n"
                            "}\n\nmodule.exports = { metricsRoute };\n"
                        ),
                    },
                    {
                        "path": "src/index.js",
                        "content": (
                            "const { statusRoute } = require('./routes/status');\n"
                            "const { catalogRoute } = require('./routes/catalog');\n"
                            "const { metricsRoute } = require('./routes/metrics');\n\n"
                            "module.exports = { statusRoute, catalogRoute, metricsRoute };\n"
                        ),
                    },
                ],
            },
        )
        assert first.result.status == "failed"
        assert first.result.metadata["source_validation_rejected"] is True
        assert first.code_completion is not None, "the rejection must keep the record"
        assert any("no-var" in issue for issue in first.result.blocking_issues)
        identified = _with_attempt_identity(first, repository_id=REPOSITORY_ID, attempt=1)

        second = await run_child(
            harness,
            engineer_payload={
                "summary": "Replace var with const, exactly as the diagnostic asks.",
                "files": [
                    {
                        "path": "src/routes/metrics.js",
                        "edits": [
                            {
                                "find": "var metrics = { count: 1 };",
                                "replace": "const metrics = { count: 1 };",
                            }
                        ],
                    }
                ],
            },
            review_payload=review_payload(verdict="approved"),
            retry_count=2,
            prior_artifacts=_execution_artifacts(identified),
        )

        assert second.result.status == "approved", second.result.blocking_issues
        # One attempt from rejection to review: the reviewer was consulted exactly once,
        # about a change whose file was edited where it stood.
        reflog = subprocess.run(
            ("git", "reflog"),
            cwd=harness.workspace,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        moves = [line for line in reflog.splitlines() if "reset: moving to HEAD" in line]
        assert moves == [], f"an ordinary retry must not reset the workspace: {moves}"
        # The union lineage is what the remote holds: the retry returned one edit and the
        # committed tree still carries the whole implementation.
        assert harness.branch in harness.remote_branches()
        assert second.code_completion is not None
        committed = subprocess.run(
            ("git", "show", "--name-only", "--format=", "HEAD"),
            cwd=harness.workspace,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.split()
        assert "src/routes/metrics.js" in committed
        assert "const metrics" in (harness.worktree_file("src/routes/metrics.js") or "")
        inputs = second.result.metadata["attempt_inputs"]
        assert inputs["workspace"] == "preserved"
    finally:
        await harness.dispose()


async def test_a_wholesale_rewrite_rejection_resets_and_the_capture_names_withheld_files(
    tmp_path: Path,
) -> None:
    """The reset survives only as §3.3's explicit path, with an honest capture.

    The retry after a rewrite rejection gets the reset, and what it is shown is bounded at
    whole-file boundaries with every withheld file named -- asserted on the real reflog and
    the delivered prompt.
    """
    harness = await build_harness(tmp_path, node_npm_repository)
    try:
        (harness.workspace / "src" / "drift.js").write_text(
            "module.exports = { drifted: true };\n", encoding="utf-8"
        )
        engineer = ScriptedLLMClient(
            [
                {
                    "summary": "Re-implement the change.",
                    "files": [
                        {
                            "path": "src/routes/metrics.js",
                            "content": (
                                "function metricsRoute(request, response) {\n"
                                "  response.json({ count: 1 });\n"
                                "}\n\nmodule.exports = { metricsRoute };\n"
                            ),
                        }
                    ],
                }
            ]
        )
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            review_payload=review_payload(verdict="approved"),
            retry_count=1,
            retry_strategy={"workspace_reset_required": "wholesale_rewrite_rejected"},
        )
        reflog = subprocess.run(
            ("git", "reflog"),
            cwd=harness.workspace,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        assert any("reset: moving to HEAD" in line for line in reflog.splitlines()), (
            "the rewrite rejection is the one path that must still reset"
        )
        assert not (harness.workspace / "src" / "drift.js").exists()
        inputs = execution.result.metadata["attempt_inputs"]
        assert inputs["workspace"] == "reset"
        assert inputs["reset_trigger"] == "wholesale_rewrite_rejected"
        # The capture reached the prompt whole: the drifted file appears end to end.
        instructions = engineer.calls[0][0]
        assert "diff --git a/src/drift.js b/src/drift.js" in instructions
        assert "{ drifted: true }" in instructions
    finally:
        await harness.dispose()


async def test_a_retry_with_no_checkout_provisions_one_and_runs_a_normal_attempt(
    tmp_path: Path,
) -> None:
    """AB-Feature-190: the attempt-0 clone never completed, so the retry had no workspace.

    Live, the retry assumed the previous attempt had left a checkout to preserve or capture.
    `_capture_attempt`'s first `git add` was spawned with a working directory that did not
    exist, `FileNotFoundError` came out of the spawn itself, nothing anticipated it, and a
    diagnosable condition -- "this attempt has no checkout" -- was filed as a platform
    defect with no safe diagnostics.

    Here the workspace is removed outright and the retry is asked to run anyway. The clone
    is real, into the real bare origin this harness pushes to; what is asserted is the
    effect, not the call: a checkout on disk that Git answers for, on the working branch,
    and an ordinary attempt that reaches review, commit and the remote.
    """
    harness = await build_harness(tmp_path, node_npm_repository, local_remote_clone=True)
    try:
        shutil.rmtree(harness.workspace)
        assert not harness.workspace.exists()

        execution = await run_child(
            harness,
            engineer_payload={
                "summary": "Add the metrics route and register it.",
                "files": [
                    {
                        "path": "src/routes/metrics.js",
                        "content": (
                            "const metrics = { count: 1 };\n\n"
                            "function metricsRoute(request, response) {\n"
                            "  response.json(metrics);\n"
                            "}\n\nmodule.exports = { metricsRoute };\n"
                        ),
                    },
                    {
                        "path": "src/index.js",
                        "content": (
                            "const { statusRoute } = require('./routes/status');\n"
                            "const { catalogRoute } = require('./routes/catalog');\n"
                            "const { metricsRoute } = require('./routes/metrics');\n\n"
                            "module.exports = { statusRoute, catalogRoute, metricsRoute };\n"
                        ),
                    },
                ],
            },
            review_payload=review_payload(verdict="approved"),
            retry_count=1,
        )

        # A checkout exists again, and it is a real one: Git answers for it, and it is on
        # the working branch the provisioner was asked to create.
        assert (harness.workspace / ".git").is_dir()
        assert _git_head(harness.workspace)
        assert harness.runner.clone_targets() == [harness.workspace]
        branch = subprocess.run(
            ("git", "rev-parse", "--abbrev-ref", "HEAD"),
            cwd=harness.workspace,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        assert branch == harness.branch

        # And the attempt was ordinary. No unanticipated error, a review verdict, and the
        # work on the remote -- which is what run 190's operator grant was asking for.
        assert execution.result.status == "approved", execution.result.blocking_issues
        assert not any("did not anticipate" in issue for issue in execution.result.blocking_issues)
        assert execution.result.failure_classification is None
        assert harness.branch in harness.remote_branches()
        # The record says why this attempt started from a clean tree, rather than leaving a
        # reader to wonder whether the previous attempt was discarded on purpose.
        inputs = execution.result.metadata["attempt_inputs"]
        assert inputs["workspace"] == "fresh_checkout"
        assert inputs["reset_trigger"] == "workspace_unusable"
    finally:
        await harness.dispose()


async def test_a_retry_whose_checkout_lost_its_git_directory_is_replaced(
    tmp_path: Path,
) -> None:
    """The other half of the sanity check: a directory Git can no longer answer for.

    A tree with no `.git` is not a checkout, and every Git command the retry path runs
    against it fails -- the capture, and then the reset that the capture's failure is
    supposed to trigger, so the reset path could not rescue itself. It is replaced instead,
    and the stale files it held are gone.
    """
    harness = await build_harness(tmp_path, node_npm_repository, local_remote_clone=True)
    try:
        shutil.rmtree(harness.workspace / ".git")
        (harness.workspace / "src" / "stale.js").write_text(
            "module.exports = { stale: true };\n", encoding="utf-8"
        )

        execution = await run_child(
            harness,
            engineer_payload={
                "summary": "Add the metrics route.",
                "files": [
                    {
                        "path": "src/routes/metrics.js",
                        "content": (
                            "const metrics = { count: 1 };\n\n"
                            "function metricsRoute(request, response) {\n"
                            "  response.json(metrics);\n"
                            "}\n\nmodule.exports = { metricsRoute };\n"
                        ),
                    },
                    {
                        "path": "src/index.js",
                        "content": (
                            "const { statusRoute } = require('./routes/status');\n"
                            "const { catalogRoute } = require('./routes/catalog');\n"
                            "const { metricsRoute } = require('./routes/metrics');\n\n"
                            "module.exports = { statusRoute, catalogRoute, metricsRoute };\n"
                        ),
                    },
                ],
            },
            review_payload=review_payload(verdict="approved"),
            retry_count=1,
        )

        assert (harness.workspace / ".git").is_dir()
        assert not (harness.workspace / "src" / "stale.js").exists()
        assert execution.result.status == "approved", execution.result.blocking_issues
        assert execution.result.metadata["attempt_inputs"]["workspace"] == "fresh_checkout"
    finally:
        await harness.dispose()


# --------------------------------------------------------------------------------------
# The in-attempt repair, against the repository's real gate
# --------------------------------------------------------------------------------------


async def test_the_repair_performs_the_rename_the_diagnostic_asks_for_and_the_gate_clears(
    tmp_path: Path,
) -> None:
    """A lint rule only a rename can clear is cleared by the repair, inside the attempt.

    The old repair instruction ended "do not restructure it, do not rename anything" and ran
    twice on AB-Feature-171 against diagnostics that demanded exactly that, clearing nothing.
    Here the fixture's own linter demands a rename (`camelcase`), the repair performs it, and
    the real gate says yes -- one attempt, no retry spent.
    """
    harness = await build_harness(tmp_path, node_npm_repository)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the metrics route.",
                "files": [
                    {
                        "path": "src/routes/metrics.js",
                        "content": (
                            "const metrics_payload = { count: 1 };\n\n"
                            "function metricsRoute(request, response) {\n"
                            "  response.json(metrics_payload);\n"
                            "}\n\nmodule.exports = { metricsRoute };\n"
                        ),
                    },
                    {
                        "path": "src/index.js",
                        "content": (
                            "const { statusRoute } = require('./routes/status');\n"
                            "const { catalogRoute } = require('./routes/catalog');\n"
                            "const { metricsRoute } = require('./routes/metrics');\n\n"
                            "module.exports = { statusRoute, catalogRoute, metricsRoute };\n"
                        ),
                    },
                ],
            },
            {
                "summary": "Rename metrics_payload to metricsPayload, exactly as asked.",
                "files": [
                    {
                        "path": "src/routes/metrics.js",
                        "content": (
                            "const metricsPayload = { count: 1 };\n\n"
                            "function metricsRoute(request, response) {\n"
                            "  response.json(metricsPayload);\n"
                            "}\n\nmodule.exports = { metricsRoute };\n"
                        ),
                    }
                ],
            },
        ]
    )
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            review_payload=review_payload(verdict="approved"),
        )

        assert execution.result.status == "approved", execution.result.blocking_issues
        completion = execution.code_completion
        assert completion is not None
        # The repair happened inside the attempt and is recorded as such.
        assert completion.metadata["source_repair_passes"] == 1
        assert completion.metadata["source_repair_paths"] == ["src/routes/metrics.js"]
        # The rename is in the worktree and the repository's own gate accepted it.
        assert "metricsPayload" in (harness.worktree_file("src/routes/metrics.js") or "")
        assert harness.runner.ran("npm", "run", "lint") >= 2
        # The delivered repair instruction permits the asked change and forbids the rest.
        repair_instructions = engineer.calls[1][0]
        assert "camelcase" in repair_instructions
        assert (
            "when a diagnostic demands a rename, rename precisely what it names"
            in repair_instructions
        )
        assert (
            "do not rename or restructure anything the diagnostics did not name"
            in repair_instructions
        )
        # Its counterpart: nothing the diagnostics did not name was touched. The other file
        # this attempt wrote is byte-identical to what the first call produced.
        assert (harness.worktree_file("src/index.js") or "").endswith(
            "module.exports = { statusRoute, catalogRoute, metricsRoute };\n"
        )
        assert "metrics_payload" not in (harness.worktree_file("src/index.js") or "")
    finally:
        await harness.dispose()


# --------------------------------------------------------------------------------------
# The AB-Feature-176 replay: remediation sees what it repairs (44-)
# --------------------------------------------------------------------------------------

_BROKEN_DECLARATION = "  const nestedDocument;"


def _bulk_import_source(*, broken: bool) -> str:
    """An ~10 KB service module with 176's one-token defect at a known deep line.

    `const nestedDocument;` -- a `const` declared for deferred assignment -- is invalid
    JavaScript, and it sat at line 179 of an 11.5 KB file the remediation was never shown.
    The bulk of the module is inert helpers, so the file is large enough that a greedy
    budget walk drops it while smaller filler still fits.
    """
    lines = ["const rows = [];", ""]
    for index in range(180):
        lines.extend(
            [
                f"function bulkHelper{index:03d}(value) {{",
                f"  return value + {index};",
                "}",
                "",
            ]
        )
    declaration = _BROKEN_DECLARATION if broken else "  let nestedDocument;"
    lines.extend(
        [
            "function bulkImport(rowsToImport) {",
            declaration,
            "  nestedDocument = { imported: rowsToImport.length };",
            "  return nestedDocument;",
            "}",
            "",
            "module.exports = { bulkImport, rows };",
        ]
    )
    return "\n".join(lines) + "\n"


_BROKEN_BULK_PAYLOAD: dict[str, Any] = {
    "summary": "Implement the bulk import service and register it.",
    "files": [
        {"path": "src/routes/bulk.js", "content": _bulk_import_source(broken=True)},
        {
            "path": "src/index.js",
            "content": (
                "const { statusRoute } = require('./routes/status');\n"
                "const { catalogRoute } = require('./routes/catalog');\n"
                "const { bulkImport } = require('./routes/bulk');\n\n"
                "module.exports = { statusRoute, catalogRoute, bulkImport };\n"
            ),
        },
    ],
}


def _write_ranked_filler(workspace: Path) -> None:
    """Commit filler the task's own wording ranks highly, totalling more than the budget.

    176's condition, generically: the checkout must not fit the snapshot, or the target file
    is delivered whatever the selection does and the test proves nothing. The filler is
    named for the requirement's subject ("status"), so relevance ranking prefers it, and the
    bulk module -- named after none of the task's words -- is exactly what falls off the end.
    """
    for index in range(55):
        filler = workspace / "src" / "status" / f"statusFlow{index:02d}.js"
        filler.parent.mkdir(parents=True, exist_ok=True)
        filler.write_text(f"// status flow filler line {index:02d}\n" * 50, encoding="utf-8")
    subprocess.run(("git", "add", "-A"), cwd=workspace, check=True, capture_output=True, timeout=30)
    subprocess.run(
        ("git", "commit", "-m", "Add status flow filler"),
        cwd=workspace,
        check=True,
        capture_output=True,
        timeout=30,
    )


class _NamedModelClient(ScriptedLLMClient):
    """A scripted client whose responses carry a distinguishable model name."""

    def __init__(self, payloads: list[dict[str, Any]], *, model: str) -> None:
        super().__init__(payloads)
        self._model = model

    @property
    def vision_capable(self) -> bool:
        """No image reaches this double, so the boundary answers False."""
        return False

    async def respond(
        self,
        *,
        instructions: str,
        input_text: str,
        images: Sequence[ImageInput] = (),
    ) -> LLMResponse:
        response = await super().respond(instructions=instructions, input_text=input_text)
        return LLMResponse(
            output_text=response.output_text,
            model=self._model,
            response_id=response.response_id,
            input_tokens=0,
            output_tokens=0,
        )


async def test_a_remediation_is_shown_the_file_it_is_repairing(tmp_path: Path) -> None:
    """The 176 replay, part one: the next attempt's snapshot contains the rejected file.

    Attempt one writes `const nestedDocument;` deep in a large service file the real gate
    rejects by absolute path with `line:col`. The recorded run then rewrote that file from
    memory three times, because its context held 46 files and not this one. Here the retry's
    snapshot must contain the file's current bytes -- defect included -- and the one-line
    `find`/`replace` correction lands, clears the gate, and reaches an approved review.
    """
    harness = await build_harness(tmp_path, node_npm_repository)
    try:
        _write_ranked_filler(harness.workspace)
        first_engineer = ScriptedLLMClient([_BROKEN_BULK_PAYLOAD])
        first = await run_child(harness, engineer_payload={}, engineer_client=first_engineer)
        assert first.result.status == "failed"
        assert first.result.metadata["source_validation_rejected"] is True
        assert first.code_completion is not None
        rejected = "\n".join(first.result.blocking_issues)
        assert "Parsing error: Unexpected token ;" in rejected
        # The gate names the file the way ESLint's stylish report does: the absolute path on
        # its own line, the `line:col` indented beneath it. Neither spelling used to survive
        # the required-context membership test.
        broken_line = _bulk_import_source(broken=True).splitlines().index(_BROKEN_DECLARATION) + 1
        assert "/src/routes/bulk.js\n" in rejected
        assert f"{broken_line}:23  error" in rejected
        identified = _with_attempt_identity(first, repository_id=REPOSITORY_ID, attempt=1)

        second_engineer = ScriptedLLMClient(
            [
                {
                    "summary": "Make the deferred declaration mutable, as the gate demands.",
                    "files": [
                        {
                            "path": "src/routes/bulk.js",
                            "edits": [
                                {
                                    "find": "const nestedDocument;",
                                    "replace": "let nestedDocument;",
                                }
                            ],
                        }
                    ],
                }
            ]
        )
        second = await run_child(
            harness,
            engineer_payload={},
            engineer_client=second_engineer,
            review_payload=review_payload(verdict="approved"),
            retry_count=2,
            prior_artifacts=_execution_artifacts(identified),
            retry_strategy={
                "failure_classification": "validation_source_failure",
                "root_cause": first.result.blocking_issues[0],
                "required_strategy_change": "Correct the exact line the gate rejected.",
                "files_or_areas_to_inspect": ["src/routes"],
                "previous_approach_to_avoid": "Rewriting the file from memory.",
                "validation_to_rerun": ["npm run lint"],
            },
        )

        assert second.result.status == "approved", second.result.blocking_issues
        completion = second.code_completion
        assert completion is not None
        # (a) The file under repair is in the retry's snapshot -- whole, with the defect --
        # while the ranked filler is what the budget sheds.
        assert "src/routes/bulk.js" in completion.metadata["context_file_paths"]
        assert completion.metadata["context_required_omitted"] == []
        assert completion.metadata["context_required_excerpted"] == []
        assert completion.metadata["context_required_dropped"] == []
        delivered = second_engineer.calls[0][1]
        assert "const nestedDocument;" in delivered
        # The one-line edit is in the worktree and on the remote.
        assert "let nestedDocument;" in (harness.worktree_file("src/routes/bulk.js") or "")
        assert harness.branch in harness.remote_branches()
    finally:
        await harness.dispose()


async def test_the_repair_pass_runs_on_the_scoped_fix_role_and_sees_the_region(
    tmp_path: Path,
) -> None:
    """The 176 replay, part two: the in-attempt repair is sighted and scoped-fix-routed.

    The recorded run burned three Engineer attempts and three blind repair passes on a
    one-keyword fix. Here the repair pass receives the workspace bytes around `179:23`,
    executes on the SCOPED_FIX boundary rather than the primary coding client, returns the
    one-line `find`/`replace`, and the gate clears inside the same attempt -- review reached
    in one attempt, with the repairing model recorded on the artifact.
    """
    harness = await build_harness(tmp_path, node_npm_repository)
    engineer = ScriptedLLMClient([_BROKEN_BULK_PAYLOAD])
    scoped = _NamedModelClient(
        [
            {
                "summary": "Make the deferred declaration mutable, as the gate demands.",
                "files": [
                    {
                        "path": "src/routes/bulk.js",
                        "edits": [
                            {
                                "find": "const nestedDocument;",
                                "replace": "let nestedDocument;",
                            }
                        ],
                    }
                ],
            }
        ],
        model="scoped-fix-model",
    )
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            review_payload=review_payload(verdict="approved"),
            scoped_fix_client=scoped,
        )

        # (c) The gate passed within the same attempt: one Engineer call, one repair pass,
        # an approved review -- where the recorded run spent three full attempts.
        assert execution.result.status == "approved", execution.result.blocking_issues
        assert len(engineer.calls) == 1
        completion = execution.code_completion
        assert completion is not None
        assert completion.metadata["source_repair_passes"] == 1
        assert completion.metadata["source_repair_paths"] == ["src/routes/bulk.js"]
        # (b) The repair prompt quotes the offending line's text beside the diagnostic.
        broken_line = _bulk_import_source(broken=True).splitlines().index(_BROKEN_DECLARATION) + 1
        repair_instructions = scoped.calls[0][0]
        assert "const nestedDocument;" in repair_instructions
        assert f"src/routes/bulk.js, around line {broken_line}" in repair_instructions
        assert "Parsing error: Unexpected token ;" in repair_instructions
        # The scoped-fix routing is recorded on the artifact, model and all.
        recorded = completion.metadata["source_repair_execution"]
        assert recorded["model"] == "scoped-fix-model"
        assert recorded["scoped_fix_role_resolved"] is True
        # The fix is real: the repository's own gate accepted the rewritten bytes.
        assert "let nestedDocument;" in (harness.worktree_file("src/routes/bulk.js") or "")
        assert harness.runner.ran("npm", "run", "lint") >= 2
    finally:
        await harness.dispose()


# --------------------------------------------------------------------------------------
# The lint vocabulary, resolved per file class through a real subprocess
# --------------------------------------------------------------------------------------

_ESLINT_STUB = """#!/usr/bin/env node
// A minimal eslint stand-in implementing `--print-config <file>` over the checkout's own
// .eslintrc.json: the `extends` chain is folded in and `overrides` globs are applied to
// the subject, which is exactly the resolution shape `react-app/jest` uses to bind
// `testing-library/*` to test files only.
const { readFileSync } = require('node:fs');
const path = require('node:path');

function load(file) {
  const config = JSON.parse(readFileSync(file, 'utf8'));
  let rules = {};
  for (const parent of config.extends || []) {
    rules = Object.assign(rules, load(path.join(path.dirname(file), parent)).rules);
  }
  return Object.assign({}, config, { rules: Object.assign(rules, config.rules || {}) });
}

const subject = process.argv[process.argv.indexOf('--print-config') + 1];
const checkout = path.join(__dirname, '..', '..');
const resolved = load(path.join(checkout, '.eslintrc.json'));
let rules = Object.assign({}, resolved.rules);
for (const override of resolved.overrides || []) {
  const patterns = [].concat(override.files || []);
  const matched = patterns.some((pattern) => subject.endsWith(pattern.split('*').pop()));
  if (matched) rules = Object.assign(rules, override.rules || {});
}
process.stdout.write(JSON.stringify({ rules: rules }));
"""


async def test_test_file_rules_behind_an_extends_chain_reach_the_delivered_vocabulary(
    tmp_path: Path,
) -> None:
    """The probe's subjects decide what the model is told exists; the tool decides what binds.

    The configuration here is the live failure's shape: a base config reached through
    `extends`, and an `overrides` block that binds `testing-library/*` to test files only.
    Resolving one production subject reported none of those rules; the second, test-class
    subject is what carries them -- and the resolution is the (stub) tool's own, never a
    reimplementation of override globs in the platform.
    """
    root = node_npm_repository(tmp_path / "checkout")
    (root / "eslint-base.json").write_text(
        json.dumps({"rules": {"no-unused-vars": ["error"]}}), encoding="utf-8"
    )
    (root / ".eslintrc.json").write_text(
        json.dumps(
            {
                "extends": ["./eslint-base.json"],
                "overrides": [
                    {
                        "files": ["**/*.test.js"],
                        "rules": {
                            "testing-library/no-node-access": ["error"],
                            "testing-library/render-result-naming-convention": ["error"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    binary = root / "node_modules" / ".bin" / "eslint"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(_ESLINT_STUB, encoding="utf-8")
    binary.chmod(0o755)

    probe = RepositoryLintCapabilities()
    capabilities = await probe.describe(
        root, ["src/index.js", "src/routes/status.js", "test/status.test.js"]
    )

    eslint = capabilities["eslint"]
    assert eslint["resolved_for"] == "src/index.js"
    assert "testing-library/no-node-access" not in eslint["enabled_rules"]
    test_files = eslint["test_files"]
    assert test_files["resolved_for"] == "test/status.test.js"
    assert any(
        rule.startswith("testing-library/no-node-access")
        for rule in test_files["additional_enabled_rules"]
    )
    assert any(
        rule.startswith("testing-library/render-result-naming-convention")
        for rule in test_files["additional_enabled_rules"]
    )


# --------------------------------------------------------------------------------------
# The integration fix attempt runs coding instead of replaying the settled publication
# --------------------------------------------------------------------------------------


def _atomic_fix() -> dict[str, Any]:
    """Return the fix attempt's edit: the change the integration review demanded."""
    return {
        "summary": "Make the bulk creation atomic, as the integration review requires.",
        "files": [
            {
                "path": "src/routes/status.js",
                "content": (
                    "function statusRoute(request, response) {\n"
                    "  response.json({ status: 'ok', recovered: true, atomic: true });\n"
                    "}\n\nmodule.exports = { statusRoute };\n"
                ),
            },
            {
                "path": "test/status.test.js",
                "content": (
                    "const { test } = require('node:test');\n"
                    "const assert = require('node:assert');\n"
                    "const { statusRoute } = require('../src/routes/status');\n\n"
                    "test('the status route answers atomically', () => {\n"
                    "  let body = null;\n"
                    "  statusRoute({}, { json: (value) => { body = value; } });\n"
                    "  assert.deepStrictEqual(\n"
                    "    body,\n"
                    "    { status: 'ok', recovered: true, atomic: true }\n"
                    "  );\n"
                    "});\n"
                ),
            },
        ],
    }


async def test_an_integration_fix_attempt_codes_against_the_finding_instead_of_replaying(
    tmp_path: Path,
) -> None:
    """AB-Feature-184's replay, against a real checkout and the real scheduling path.

    A child with a SUCCEEDED, checkpointed publication receives an integration
    changes-requested routing. The next wave must decline recovery, run a real coding
    attempt whose prompt carries the review's required fix, and land a new commit on top
    of the settled one -- with exactly one journaled coding effect per attempt. This is the
    sequence that ended 184 in under a second.
    """
    harness = await build_harness(tmp_path, node_npm_repository)
    required_fix = "Make the bulk app creation atomic across the import path."
    engineer = ScriptedLLMClient([_route_change(), _atomic_fix()])
    reviewer = ScriptedLLMClient(
        [review_payload(verdict="approved"), review_payload(verdict="approved")]
    )
    executor = LiveChildWorkstreamExecutor(
        settings=harness.settings,
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness.journal,
        cancellation_token=MockCancellationToken(),
        process_runner=harness.runner,
    )
    request = StartFeatureRequest.model_validate(real_support.feature_payload())
    state = _initial_feature_state(real_support.FEATURE_ID, request)
    technical_prd = real_support.technical_prd_artifact()
    contract = real_support.contract_artifact()
    workstream = workstream_plan()
    plan = create_artifact(
        RepositoryExecutionPlanArtifact,
        workflow_id=real_support.FEATURE_ID,
        artifact_id="010_repository_execution_plan.json",
        producer="feature_planner",
        metadata={},
        payload={
            "feature_id": real_support.FEATURE_ID,
            "contract_artifact_id": contract.artifact_id,
            "workstreams": [workstream.model_dump(mode="json")],
            "execution_order": [REPOSITORY_ID],
            "parallel_groups": [[REPOSITORY_ID]],
            "integration_test_plan": ["Validate the shared contract."],
            "merge_strategy": "independent",
            "deployment_strategy": "independent",
            "feature_flag_strategy": [],
            "rollback_strategy": ["Revert the pull request."],
        },
    )
    state.artifacts.extend([technical_prd, contract, plan])
    repository = next(
        item for item in state.repository_specs if item.repository_id == REPOSITORY_ID
    )
    child = ChildWorkflowReference(
        child_workflow_id=f"{real_support.FEATURE_ID}:{REPOSITORY_ID}",
        repository_id=REPOSITORY_ID,
        workstream_id=REPOSITORY_ID,
        status=ChildWorkflowStatus.RUNNING,
        branch_name=harness.branch,
        workspace_path=str(harness.workspace),
        retry_count=1,
    )
    state.child_workflows[REPOSITORY_ID] = child
    try:
        published = await executor.run(
            feature=state,
            repository=repository,
            workstream=workstream,
            child=child,
            technical_prd=technical_prd,
            contract=contract,
            feedback=[],
            credentials=real_support.CREDENTIALS,
        )
        assert published.result.status == "approved"
        settled_sha = harness.remote_branches()[harness.branch]

        # What the fan-out records once the wave settles this attempt: the artifacts under
        # their repository-and-attempt scoped identities, and the child row naming them.
        identified = _with_attempt_identity(
            published, repository_id=REPOSITORY_ID, attempt=child.retry_count
        )
        _append_artifacts(state, _execution_artifacts(identified))
        state.child_workflows[REPOSITORY_ID] = child.model_copy(
            update={
                "status": ChildWorkflowStatus.APPROVED,
                "code_completion_artifact_id": identified.result.code_completion_artifact_id,
                "review_artifact_id": identified.result.review_artifact_id,
                "checkpoint_boundary": WorkflowCheckpointBoundary.AFTER_VALIDATION,
            }
        )

        # The verdict that asks this repository for one precise fix.
        integration_review = create_artifact(
            IntegrationReviewArtifact,
            workflow_id=real_support.FEATURE_ID,
            artifact_id="012_integration_review.json",
            producer="integration_reviewer",
            metadata={"source_artifact_ids": [contract.artifact_id]},
            payload={
                "feature_id": real_support.FEATURE_ID,
                "contract_artifact_id": contract.artifact_id,
                "review_status": "changes_requested",
                "repository_results": [
                    {
                        "repository_id": REPOSITORY_ID,
                        "child_workflow_id": child.child_workflow_id,
                        "status": "approved",
                        "child_result_artifact_id": identified.result.artifact_id,
                    }
                ],
                "contract_checks": ["The import path must not partially create apps."],
                "cross_repository_findings": [
                    {
                        "finding_id": "finding-atomic-bulk-creation",
                        "severity": "high",
                        "responsible_repository_id": REPOSITORY_ID,
                        "affected_repository_ids": [REPOSITORY_ID],
                        "contract_reference": "contract",
                        "description": "Bulk creation is not atomic across the import path.",
                        "evidence": "The route commits each app independently.",
                        "recommended_fix": required_fix,
                    }
                ],
                "compatibility_assessment": "One repository must correct its behaviour.",
                "security_assessment": "No security assessment was performed.",
                "deployment_assessment": "No deployment assessment was performed.",
                "merge_order": [REPOSITORY_ID],
                "required_fixes": [required_fix],
            },
        )
        _append_artifacts(state, [integration_review])
        state.status = FeatureWorkflowStatus.INTEGRATION_REVIEW

        orchestrator = FeatureWorkflowOrchestrator(
            child_executor=executor, workspace_root=harness.workspace_root
        )
        orchestrator._route_integration_remediation(  # noqa: SLF001
            state, plan=plan, review=integration_review, eligible={REPOSITORY_ID}
        )
        routed = state.child_workflows[REPOSITORY_ID]
        assert routed.status is ChildWorkflowStatus.PENDING
        assert routed.retry_count == child.retry_count + 1
        # The routed attempt has not started coding, and the settled attempt's artifact
        # references stay as the lineage the fix attempt builds on.
        assert routed.checkpoint_boundary is WorkflowCheckpointBoundary.BEFORE_CODING
        assert routed.code_completion_artifact_id == (identified.result.code_completion_artifact_id)
        assert routed.review_artifact_id == identified.result.review_artifact_id

        state = await orchestrator._step_execute_workstreams(  # noqa: SLF001
            state, credentials=real_support.CREDENTIALS
        )
        operations = await harness.journal.list_operations_for_workflow(state.workflow_id)
    finally:
        await harness.dispose()

    # The fix attempt ran real coding against the finding rather than replaying attempt 1.
    assert len(engineer.calls) == 2, "the wave must ask the Engineer for the fix attempt"
    assert required_fix in engineer.calls[1][1], (
        "the fix attempt's prompt must carry the integration review's required fix"
    )
    fixed = state.child_workflows[REPOSITORY_ID]
    assert fixed.status in {ChildWorkflowStatus.APPROVED, ChildWorkflowStatus.COMPLETED}
    results = [
        item
        for item in state.artifacts
        if isinstance(item, ChildWorkflowResultArtifact)
        and item.metadata.get("child_retry_count") == routed.retry_count
    ]
    assert len(results) == 1 and results[0].status == "approved"
    completions = [
        item
        for item in state.artifacts
        if isinstance(item, CodeCompletionArtifact)
        and item.metadata.get("child_attempt") == routed.retry_count
    ]
    assert completions and all(
        item.metadata.get("publication_recovered") is not True for item in completions
    )
    # A new commit on top of the settled one, on the real remote.
    remote_sha = harness.remote_branches()[harness.branch]
    assert remote_sha != settled_sha
    parent = subprocess.run(
        ("git", "rev-parse", f"{remote_sha}~1"),
        cwd=harness.remote,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    assert parent == settled_sha
    assert "atomic: true" in (harness.worktree_file("src/routes/status.js") or "")
    # One journaled coding effect per attempt; the settled operations were not re-executed.
    kinds = [item.operation_type for item in operations]
    assert kinds.count(ExternalOperationType.WRITE_FILE_CHANGES) == 2
    assert kinds.count(ExternalOperationType.CREATE_COMMIT) == 2
    assert kinds.count(ExternalOperationType.PUSH_BRANCH) == 2


# --------------------------------------------------------------------------------------
# 47- Part C: the change's own tests run inside the attempt, and no repair may weaken them
# --------------------------------------------------------------------------------------

_METRICS_INDEX = (
    "const { statusRoute } = require('./routes/status');\n"
    "const { catalogRoute } = require('./routes/catalog');\n"
    "const { metricsRoute } = require('./routes/metrics');\n\n"
    "module.exports = { statusRoute, catalogRoute, metricsRoute };\n"
)


def _metrics_route(*, with_total: bool) -> str:
    """The module under test, either satisfying its requirement or falling short of it.

    `with_total=False` is C2's defect and it is a defect in *production* code: the suite the
    same attempt writes is correct about what the requirement asked for, and the route is
    what disagrees with it.
    """
    payload = "{ count: 1, total: 3 }" if with_total else "{ count: 1 }"
    return (
        f"const metricsPayload = {payload};\n\n"
        "function metricsRoute(request, response) {\n"
        "  response.json(metricsPayload);\n"
        "}\n\nmodule.exports = { metricsRoute };\n"
    )


def _metrics_suite(*, module: str, expected: str = "{ count: 1, total: 3 }") -> str:
    """The change's own suite, parameterized by what it requires and what it asserts.

    Two knobs, because the two scenarios differ in exactly one each. C1 gets the module path
    wrong -- broken scaffolding, no assertion involved -- and its repair corrects that line
    while every line the guard inspects stays byte-identical. C2's suite is correct in both
    respects and its repair tries to move `expected`, which is the thing that must not work.
    """
    return (
        "const { test } = require('node:test');\n"
        "const assert = require('node:assert');\n"
        f"const {{ metricsRoute }} = require('{module}');\n\n"
        "test('the metrics route reports a total', () => {\n"
        "  let body = null;\n"
        "  metricsRoute({}, { json: (value) => { body = value; } });\n"
        f"  assert.deepStrictEqual(body, {expected});\n"
        "});\n"
    )


def _metrics_attempt(*, module: str, expected: str, with_total: bool) -> dict[str, Any]:
    """One complete engineer attempt: the route, its registration, and its suite."""
    return {
        "summary": "Add the metrics route and the suite that covers it.",
        "files": [
            {"path": "src/routes/metrics.js", "content": _metrics_route(with_total=with_total)},
            {"path": "src/index.js", "content": _METRICS_INDEX},
            {
                "path": "test/metrics.test.js",
                "content": _metrics_suite(module=module, expected=expected),
            },
        ],
    }


def _narrowed_test_runs(harness: real_support.RealRepositoryHarness) -> list[tuple[str, ...]]:
    """Return every test invocation this run actually executed, narrowed or whole."""
    return [
        command
        for command, _cwd in harness.runner.executed
        if command[:3] == ("npm", "run", "test")
    ]


def _npm_test(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the checkout's own test command for real, as the platform would have run it."""
    return subprocess.run(
        ("npm", "run", "test", *arguments),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "CI": "true"},
        check=False,
    )


async def test_a_broken_fixture_is_repaired_inside_the_attempt_and_the_suite_then_passes(
    tmp_path: Path,
) -> None:
    """C1. The narrowed runner rejects broken scaffolding; the repair clears it in place.

    Before this, the first thing to run a repository's test suite was the reviewer -- so a
    suite that could not even resolve the module it tests was structurally guaranteed to
    cost a full attempt and a 15-40 minute re-implementation. Here the attempt asks the
    repository's own runner about the one file it just wrote, repairs the `require` it got
    wrong, and reaches review in the same attempt.

    The repair touches only a test file, which is exactly the shape the assertion guard is
    watching for -- and it is allowed, because the assertion, the test declaration and the
    absence of any skip are byte-identical across it. That is the line between repairing
    scaffolding and rewriting a claim.
    """
    harness = await build_harness(tmp_path, node_npm_repository)
    engineer = ScriptedLLMClient(
        [
            # `../src/routes/metric` -- singular, and therefore unresolvable.
            _metrics_attempt(
                module="../src/routes/metric",
                expected="{ count: 1, total: 3 }",
                with_total=True,
            ),
            {
                "summary": "Require the module by the name it actually has.",
                "files": [
                    {
                        "path": "test/metrics.test.js",
                        "content": _metrics_suite(module="../src/routes/metrics"),
                    }
                ],
            },
        ]
    )
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            review_payload=review_payload(verdict="approved"),
        )

        assert execution.result.status == "approved", execution.result.blocking_issues
        # The effect, not the invocation: the repository's own narrowed command, run again
        # here against the worktree the attempt left behind, exits zero.
        rerun = _npm_test(harness.workspace, "--", "test/metrics.test.js")
        assert rerun.returncode == 0, f"{rerun.stdout}\n{rerun.stderr}"
        # And it did so because of a repair inside this attempt, not a second one: the child
        # executor ran once, and the Engineer was asked twice -- the implementation and the
        # repair -- with no outer attempt in between.
        assert len(engineer.calls) == 2
        completion = execution.code_completion
        assert completion is not None
        assert completion.metadata["source_repair_passes"] == 1
        assert completion.metadata["source_repair_paths"] == ["test/metrics.test.js"]
        assert completion.metadata["scoped_test_commands"] == [
            "npm run test -- test/metrics.test.js"
        ]
        # The repair was told what it was allowed to do, and what it was not.
        repair_instructions = engineer.calls[1][0]
        assert "rejected by its own test runner" in repair_instructions
        assert "Fix the code under test, never the test itself." in repair_instructions
        # It obeyed: what the suite asserts is byte-identical to what the attempt wrote.
        assert harness.worktree_file("test/metrics.test.js") == _metrics_suite(
            module="../src/routes/metrics"
        )
        # The narrowed command was narrowed to the one suite this change touched, and the
        # committed tree carries both files.
        assert ("npm", "run", "test", "--", "test/metrics.test.js") in _narrowed_test_runs(harness)
        assert harness.branch in harness.remote_branches()
        committed = subprocess.run(
            ("git", "show", "--name-only", "--format=", "HEAD"),
            cwd=harness.workspace,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.split()
        assert "test/metrics.test.js" in committed
        assert "src/routes/metrics.js" in committed
    finally:
        await harness.dispose()


async def test_a_correct_test_failing_on_wrong_product_code_is_not_resolved_by_editing_it(
    tmp_path: Path,
) -> None:
    """C2. The guard that decides whether Part C ships at all.

    The suite is right and the route is wrong: the requirement asks for a total and the
    route never reports one. A repair told to make the diagnostics go away can satisfy that
    failure by moving the assertion, and no later gate would ever catch it -- the suite would
    be green and smaller. AB-Feature-182 is the precedent, where three remediation attempts
    edited a test fixture while the defect sat in the product file.

    So the repair here does exactly that, and the assertion the attempt wrote is what the
    worktree still holds afterwards. The pass is discarded, the bytes are put back, the loop
    stops rather than asking again, and the attempt leaves with a named outcome instead of a
    raw test line.
    """
    harness = await build_harness(tmp_path, node_npm_repository)
    engineer = ScriptedLLMClient(
        [
            _metrics_attempt(
                module="../src/routes/metrics",
                expected="{ count: 1, total: 3 }",
                with_total=False,
            ),
            {
                "summary": "Make the suite agree with what the route returns.",
                "files": [
                    {
                        "path": "test/metrics.test.js",
                        "content": _metrics_suite(
                            module="../src/routes/metrics", expected="{ count: 1 }"
                        ),
                    }
                ],
            },
        ]
    )
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            review_payload=review_payload(verdict="approved"),
        )

        assert execution.result.status == "failed"
        # The effect that decides this: the weakened assertion is not in the workspace. The
        # next attempt, and the reviewer after it, see the claim the attempt actually made.
        suite = harness.worktree_file("test/metrics.test.js")
        assert suite == _metrics_suite(module="../src/routes/metrics")
        assert "total: 3" in (suite or "")
        # And the product defect is untouched, so what is wrong is still there to be fixed.
        assert "total" not in (harness.worktree_file("src/routes/metrics.js") or "")
        # The loop stopped on the spot rather than buying another pass with the refusal.
        assert len(engineer.calls) == 2
        # The escape is a named outcome, in the diagnostics and on the record.
        assert any(
            ASSERTION_GUARD_OUTCOME in issue for issue in execution.result.blocking_issues
        ), execution.result.blocking_issues
        completion = execution.code_completion
        assert completion is not None
        assert completion.metadata["terminal_outcome"] == ASSERTION_GUARD_OUTCOME
        assert completion.completion_status == "failed"
        # The refused pass is not recorded as a repair, because it was thrown away -- said as
        # a count of zero rather than as an absent key. A gate-rejected attempt now always
        # states its repair history, and "no pass survived" has to be readable as such: while
        # the block was simply omitted, this outcome and four spent passes were the same bytes.
        assert completion.metadata["source_repair_engaged"] is False
        assert completion.metadata["source_repair_passes"] == 0
        assert "source_repair_execution" not in completion.metadata
        # And the record says what the gate ran to reach that outcome. The guard builds a
        # fresh rejection around its own sentence, and rebuilding it used to discard the
        # structured results of the run the refused repair was asked to clear -- so this
        # attempt recorded `[]` and the only readable trace of the failing suite was a
        # sentence. The coarse failure counter reads this field, and derives the same file
        # set from the diagnostics when it is empty; the derivation was never the problem,
        # the silent record was. Asserted here, against a real checkout and a real runner,
        # because a fixture asserting it would only be asserting itself.
        recorded = execution.result.current_validation_results
        assert len(recorded) == 1, recorded
        assert recorded[0]["command"] == "npm run test -- test/metrics.test.js"
        assert recorded[0]["required"] is True
        assert recorded[0]["passed"] is False
        assert failing_evidence_files(execution.result) == frozenset({"test/metrics.test.js"})
        # Nothing reached review and nothing was published.
        assert harness.branch not in harness.remote_branches()
        # The only test command this run executed was the Engineer's own narrowed one.
        assert _narrowed_test_runs(harness) == [
            ("npm", "run", "test", "--", "test/metrics.test.js")
        ]
    finally:
        await harness.dispose()


async def test_a_test_script_that_cannot_be_narrowed_runs_nothing_inside_the_attempt(
    tmp_path: Path,
) -> None:
    """A composed test script is not narrowed, and the attempt behaves exactly as before.

    `node scripts/pretest.js && node --test` would give trailing arguments to whichever half
    ran last, so narrowing would change the question rather than answer it faster.
    `scoped_test_command` says None, and the in-attempt check has to run nothing at all --
    running the whole suite here would cost the fifteen minutes the narrowing exists to
    avoid, twice per attempt, for an answer the reviewer is about to produce anyway.
    """
    harness = await build_harness(tmp_path, unnarrowable_test_repository)
    engineer = ScriptedLLMClient(
        [
            _metrics_attempt(
                module="../src/routes/metrics",
                expected="{ count: 1, total: 3 }",
                with_total=True,
            )
        ]
    )
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            review_payload=review_payload(verdict="approved"),
        )

        assert execution.result.status == "approved", execution.result.blocking_issues
        completion = execution.code_completion
        assert completion is not None
        # Nothing ran in the attempt, and the record says so rather than implying the tests
        # passed there.
        assert completion.metadata["scoped_test_commands"] == []
        assert "source_repair_passes" not in completion.metadata
        assert len(engineer.calls) == 1
        # Every test invocation in the whole run is the repository's own whole command.
        runs = _narrowed_test_runs(harness)
        assert runs and all(command == ("npm", "run", "test") for command in runs)
    finally:
        await harness.dispose()


# --------------------------------------------------------------------------------------
# 47- Part D: reachability, checked inside the attempt and decided outside it
# --------------------------------------------------------------------------------------


def _unwired_metrics_attempt() -> dict[str, Any]:
    """The route and its suite, correct as source, and mounted nowhere.

    Lint passes, the narrowed suite passes, and the running application still cannot reach
    `src/routes/metrics.js` -- which is the shape 24% of live attempt-0s produced. `src/index.js`
    is deliberately absent from this payload; the repair is what has to add it.
    """
    return {
        "summary": "Add the metrics route and the suite that covers it.",
        "files": [
            {"path": "src/routes/metrics.js", "content": _metrics_route(with_total=True)},
            {
                "path": "test/metrics.test.js",
                "content": _metrics_suite(module="../src/routes/metrics"),
            },
        ],
    }


async def test_an_unreachable_module_is_wired_inside_the_attempt_and_then_approved(
    tmp_path: Path,
) -> None:
    """D1. The test that proves the 24% figure moves.

    A change adds a production module nothing references. Until now that was found only by
    the runtime, after the Engineer had returned `completed`, so it always cost a full outer
    attempt and a 15-40 minute re-implementation -- and it was the single largest cause of
    wasted attempts across runs 66 to 106.

    Here the Engineer asks the checkout itself, in the same attempt, after lint and the
    change's own tests are green. The repair adds the reference to the registry the finding
    named, the authoritative gate then agrees, and the change reaches review without the
    retry policy granting anything.
    """
    harness = await build_harness(tmp_path, node_npm_repository)
    engineer = ScriptedLLMClient(
        [
            _unwired_metrics_attempt(),
            {
                "summary": "Register the metrics route where this repository mounts modules.",
                "files": [{"path": "src/index.js", "content": _METRICS_INDEX}],
            },
        ]
    )
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            review_payload=review_payload(verdict="approved"),
        )

        assert execution.result.status == "approved", execution.result.blocking_issues
        # The effect, not the invocation: the registry in the worktree really does reach the
        # new module, and the module is really in the pushed commit.
        index = harness.worktree_file("src/index.js") or ""
        assert "require('./routes/metrics')" in index
        assert "metricsRoute" in index
        assert harness.branch in harness.remote_branches()
        committed = subprocess.run(
            ("git", "show", "--name-only", "--format=", "HEAD"),
            cwd=harness.workspace,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.split()
        assert {"src/index.js", "src/routes/metrics.js"} <= set(committed)
        # And it happened inside one attempt: the child executor ran once and the Engineer was
        # asked twice, the implementation and the repair, with no outer attempt between them.
        assert len(engineer.calls) == 2
        completion = execution.code_completion
        assert completion is not None
        assert completion.completion_status == "completed"
        assert completion.metadata["source_repair_passes"] == 1
        assert completion.metadata["source_repair_paths"] == ["src/index.js"]
        # Nothing unrepaired is recorded, because nothing was left unrepaired.
        assert "reachability_issues_unrepaired" not in completion.metadata
        assert "reachability_candidates_dropped" not in completion.metadata
        # The attempt counter is what the retry policy grants, and this repair asked for
        # nothing: the runtime's own wiring gate reported no defect to spend one on.
        assert execution.result.metadata.get("wiring_repair") is None
        assert execution.result.failure_classification is None
        # The repair was told what rejected its work, and told not to touch the module.
        repair_instructions = engineer.calls[1][0]
        assert "still cannot be reached" in repair_instructions
        assert "no production file refers to it" in repair_instructions
        assert "Do not rewrite, rename, move or delete the code that cannot be reached" in (
            repair_instructions
        )
        # It obeyed: the module the finding named is byte-identical to what the attempt wrote.
        assert harness.worktree_file("src/routes/metrics.js") == _metrics_route(with_total=True)
    finally:
        await harness.dispose()


async def test_the_engineers_own_reachability_check_cannot_short_circuit_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D4. The authoritative gate still runs, and still decides.

    An agent that both writes the code and rules on whether it is reachable has no gate at
    all, so the two layers must be independent -- not one layer with a cache. The way to prove
    independence is to make them disagree: the Engineer's own check is replaced with one that
    inspects nothing, so as far as the attempt is concerned reachability passed, and the
    runtime must still reject the very same workspace on its own reading of it.

    That the runtime consults nothing the Engineer decided is the point. It is also why an
    unrepaired finding inside the attempt is deliberately *not* fatal there: the gate below is
    what carries `wiring_repair` into the next attempt's snapshot, and short-circuiting to a
    source-validation failure would throw that hint away.
    """
    monkeypatch.setattr(
        "services.feature_runtime.RepositoryReachabilityChecker",
        lambda **_kwargs: NullReachabilityChecker(),
    )
    harness = await build_harness(tmp_path, node_npm_repository)
    engineer = ScriptedLLMClient([_unwired_metrics_attempt()])
    reviewer = ScriptedLLMClient([review_payload(verdict="approved")])
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            reviewer_client=reviewer,
        )

        # The in-attempt check said nothing, so the Engineer declared itself done in one call
        # and recorded no finding of its own.
        assert len(engineer.calls) == 1
        completion = execution.code_completion
        assert completion is not None
        assert completion.completion_status == "completed"
        assert "reachability_issues_unrepaired" not in completion.metadata
        # And the runtime rejected the attempt anyway, on its own reading of the checkout.
        result = execution.result
        assert result.status == "failed"
        assert result.failure_classification == FailureClassification.IMPLEMENTATION_MISSING.value
        assert result.metadata["deterministic_gate"] is True
        assert result.metadata["wiring_repair"]["added_path"] == "src/routes/metrics.js"
        assert result.metadata["wiring_repair"]["target_path"] == "src/index.js"
        assert any("cannot reach it" in issue for issue in result.blocking_issues)
        # Nothing reached review and nothing was published, which is what the gate is for.
        assert reviewer.calls == []
        assert harness.branch not in harness.remote_branches()
    finally:
        await harness.dispose()


# --------------------------------------------------------------------------------------
# 49- Part D: the diff touches the file the plan names
#
# The negatives -- the shapes this check must stay silent on -- are in
# `test_assigned_file_conformance.py` against real `git status` output, because they are
# properties of the predicate and not of the attempt. What is here is the one thing only an
# attempt can show: the diagnostic reaching the repair pass, and the repair using the file the
# plan named, inside a single attempt.
# --------------------------------------------------------------------------------------

_STATUS_WITH_METRICS = (
    "const metricsPayload = { count: 1, total: 3 };\n\n"
    "function statusRoute(request, response) {\n"
    "  response.json({ status: 'ok' });\n"
    "}\n\n"
    "function metricsRoute(request, response) {\n"
    "  response.json(metricsPayload);\n"
    "}\n\n"
    "module.exports = { statusRoute, metricsRoute };\n"
)


def _sibling_metrics_attempt() -> dict[str, Any]:
    """The 183 shape: a new module beside the assigned file, correct and wired.

    Everything a check could object to is in order. Lint passes, the narrowed suite passes,
    and `src/index.js` really does reach the new module -- so the wiring inspection has
    nothing to say. The single defect is that the plan named `src/routes/status.js` and this
    change never opens it.
    """
    return {
        "summary": "Add the metrics route beside the status route.",
        "files": [
            {"path": "src/routes/metrics.js", "content": _metrics_route(with_total=True)},
            {"path": "src/index.js", "content": _METRICS_INDEX},
            {
                "path": "test/metrics.test.js",
                "content": _metrics_suite(module="../src/routes/metrics"),
            },
        ],
    }


async def test_a_module_written_beside_its_assigned_file_is_moved_inside_the_attempt(
    tmp_path: Path,
) -> None:
    """D1. The 183 frontend replay, and the cheapest attempt this platform ever wasted.

    Run 183's frontend was rejected on attempt 0 for exactly this: the scoped acceptance
    criteria named the exact file, the implementation wrote a sibling module in the same
    directory, and a reviewer cycle -- a whole outer attempt -- was spent saying so. Every
    input the answer needed was already on disk.

    Three conditions have to hold together for this to fire, and this fixture is the only
    arrangement in the tier where they do: the plan names `src/routes/status.js`, the change
    never touches it, and the change adds a module beside it. Everything else about the
    attempt is deliberately clean, so the finding cannot be the wiring inspection's in
    disguise.
    """
    harness = await build_harness(tmp_path, node_npm_repository)
    engineer = ScriptedLLMClient(
        [
            _sibling_metrics_attempt(),
            {
                "summary": "Declare the metrics route in the module the plan assigned.",
                "files": [{"path": "src/routes/status.js", "content": _STATUS_WITH_METRICS}],
            },
        ]
    )
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            review_payload=review_payload(verdict="approved"),
            workstream=real_support.workstream_plan(
                expected_source_areas=["src/routes", "src/routes/status.js"]
            ),
        )

        assert execution.result.status == "approved", execution.result.blocking_issues
        # The effect: the file the plan named really does hold the new route now, and it took
        # one attempt -- the implementation and the repair, with no outer attempt between them.
        assert "metricsRoute" in (harness.worktree_file("src/routes/status.js") or "")
        assert len(engineer.calls) == 2
        completion = execution.code_completion
        assert completion is not None
        assert completion.completion_status == "completed"
        assert completion.metadata["source_repair_passes"] == 1
        assert completion.metadata["source_repair_paths"] == ["src/routes/status.js"]
        # Nothing left on the record, because the repair used the assigned file.
        assert "assigned_file_issues_unrepaired" not in completion.metadata
        assert "reachability_issues_unrepaired" not in completion.metadata
        # The diagnostic named *both* paths and was attributed to the check that produced it,
        # not to the linter and not to the wiring inspection -- whose instruction is the
        # opposite one, and whose sentence would send the repair to a host file instead.
        repair_instructions = engineer.calls[1][0]
        assert "beside the file this workstream's plan named" in repair_instructions
        assert "src/routes/status.js" in repair_instructions
        assert "src/routes/metrics.js" in repair_instructions
        assert "Do not rewrite, rename, move or delete" not in repair_instructions
        # And the attempt counter is untouched: this check informs a prompt and grants nothing.
        assert execution.result.failure_classification is None
        assert execution.result.metadata.get("wiring_repair") is None
    finally:
        await harness.dispose()


# --------------------------------------------------------------------------------------
# 47- Part A: a named symbol resolves to the file that defines it
#
# One deterministic definition-site lookup, at the two places that need it: the in-attempt
# repair pass, whose commonest diagnostic is a missing import (A1), and the remediation retry
# after a reviewer rejection, whose finding is frequently about something nothing imports yet
# (A3). Both are context-assembly failures and both were previously answered by guessing.
#
# The lookup as a function -- what it extracts, what it refuses, where it says a name is
# defined -- is covered in the default tier, in `test_definition_sites.py`. What is here is
# what was never checked before the defect it addresses existed for as long as it did: the
# text the model was actually handed.
# --------------------------------------------------------------------------------------


def _publish_route(*, specifier: str | None) -> str:
    """The route the attempt writes, either importing the module it uses or not.

    `specifier=None` is A1's defect and it is the defect that costs this platform the most
    attempts: the name is used, the import is absent, and the thirty lines the repair prompt
    quotes around the linter's line contain the use and nothing else.
    """
    imported = f"const {{ permissionUtils }} = require('{specifier}');\n\n" if specifier else ""
    return (
        f"{imported}"
        "function publishRoute(request, response) {\n"
        "  response.json({ allowed: permissionUtils.canPublish(request.user) });\n"
        "}\n"
        "\n"
        "module.exports = { publishRoute };\n"
    )


def _index_with(route: str, symbol: str) -> str:
    """Return the checkout's registry, extended to reach one more route."""
    return (
        "const { statusRoute } = require('./routes/status');\n"
        "const { catalogRoute } = require('./routes/catalog');\n"
        f"const {{ {symbol} }} = require('{route}');\n\n"
        f"module.exports = {{ statusRoute, catalogRoute, {symbol} }};\n"
    )


def _npm_lint(root: Path) -> subprocess.CompletedProcess[str]:
    """Run the checkout's own lint command for real, as the platform would have run it."""
    return subprocess.run(
        ("npm", "run", "lint"),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "CI": "true"},
        check=False,
    )


async def test_a_missing_import_is_repaired_from_the_definition_the_prompt_was_shown(
    tmp_path: Path,
) -> None:
    """A1. The definition reaches the repair prompt, and the real linter then exits zero.

    `permissionUtils` is exported by `src/support/permissions.js` -- a filename that shares
    nothing with the symbol, which is the ordinary case. Before this the repair pass could not
    find that file by any route: `_diagnostic_symbol_paths` resolves only *through an import
    that already exists*, and the whole defect is that no import exists. So the pass guessed a
    specifier, and the linter's second rule is what tells a guess from a reading -- a module
    path that does not resolve is a diagnostic of its own.

    The assertions are the four the acceptance list asks for: the definition site is in the
    text the repair model was handed, the specifier the repair wrote is the correct one, the
    repository's own lint command exits zero against the worktree afterwards, and no outer
    attempt was spent on any of it.
    """
    harness = await build_harness(tmp_path, missing_import_repository)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the publish route and register it.",
                "files": [
                    {"path": "src/routes/publish.js", "content": _publish_route(specifier=None)},
                    {
                        "path": "src/index.js",
                        "content": _index_with("./routes/publish", "publishRoute"),
                    },
                ],
            },
            {
                "summary": "Import the permission helper from the module that exports it.",
                "files": [
                    {
                        "path": "src/routes/publish.js",
                        "content": _publish_route(specifier="../support/permissions"),
                    }
                ],
            },
        ]
    )
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            review_payload=review_payload(verdict="approved"),
        )

        assert execution.result.status == "approved", execution.result.blocking_issues
        # The effect, not the invocation: the repository's own linter, run again here against
        # the worktree this attempt left behind, exits zero.
        rerun = _npm_lint(harness.workspace)
        assert rerun.returncode == 0, f"{rerun.stdout}\n{rerun.stderr}"
        # It was repaired inside this attempt: the child executor ran once and the Engineer
        # was asked twice, with no outer attempt between them.
        assert len(engineer.calls) == 2
        assert execution.result.failure_classification is None
        completion = execution.code_completion
        assert completion is not None
        assert completion.metadata["source_repair_passes"] == 1
        assert completion.metadata["source_repair_paths"] == ["src/routes/publish.js"]
        # And this is why it could be: the pass was shown where the checkout defines the name
        # the diagnostic said was undefined, as content rather than as a path.
        repair_instructions = engineer.calls[1][0]
        assert "'permissionUtils' is not defined" in repair_instructions
        assert "Where this checkout defines the names" in repair_instructions
        assert f"===== {MISSING_IMPORT_DEFINITION}, around line" in repair_instructions
        assert "const permissionUtils = { canPublish };" in repair_instructions
        # The specifier in the worktree is the one that resolves, not one shaped like the
        # symbol -- which is exactly what a guess produces and what the linter refuses.
        published = harness.worktree_file("src/routes/publish.js") or ""
        assert "require('../support/permissions')" in published
    finally:
        await harness.dispose()


async def test_a_diagnostic_that_names_no_resolvable_symbol_quotes_no_definition(
    tmp_path: Path,
) -> None:
    """A2. No false resolution, and no displacing of the diagnostics that do resolve.

    The diagnostic here names a module path that does not resolve, which is a dotted filename
    inside quotes -- the shape most likely to be mistaken for a symbol. Nothing about it is a
    name this checkout defines, so the repair prompt carries the diagnostic's own source
    window and no definition block at all: a name resolving to nothing costs nothing.

    The shape-by-shape precision -- a dotted filename, a dependency's API, an ordinary English
    sentence -- is asserted in the default tier, where each can be stated in isolation. What
    this adds is that the decision survives all the way into the delivered prompt.
    """
    harness = await build_harness(tmp_path, missing_import_repository)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the publish route and register it.",
                "files": [
                    {
                        "path": "src/routes/publish.js",
                        # A specifier shaped like the symbol rather than like the module: the
                        # guess this whole part exists to make unnecessary.
                        "content": _publish_route(specifier="../support/permissionUtils"),
                    },
                    {
                        "path": "src/index.js",
                        "content": _index_with("./routes/publish", "publishRoute"),
                    },
                ],
            },
            {
                "summary": "Import the permission helper from the module that exports it.",
                "files": [
                    {
                        "path": "src/routes/publish.js",
                        "content": _publish_route(specifier="../support/permissions"),
                    }
                ],
            },
        ]
    )
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            review_payload=review_payload(verdict="approved"),
        )

        assert execution.result.status == "approved", execution.result.blocking_issues
        repair_instructions = engineer.calls[1][0]
        assert "Cannot find module '../support/permissionUtils'" in repair_instructions
        # Nothing resolved, so nothing was quoted as a definition...
        assert "Where this checkout defines the names" not in repair_instructions
        # ...and the diagnostic's own window, which is what the budget is for, is still there.
        assert "===== src/routes/publish.js, around line" in repair_instructions
    finally:
        await harness.dispose()


async def test_a_finding_naming_a_symbol_nothing_imports_still_reaches_the_next_attempt(
    tmp_path: Path,
) -> None:
    """A3. The test that proves A.2 does something A.1 does not.

    The finding's subject here is reachable by none of the routes that existed: nothing in
    the change imports `pageBody`, no finding names its file by path, the plan assigns
    `src/app.js` instead, and no prior attempt touched it. So selection fell to lexical
    ranking, which scores a module named for none of the task's wording at zero -- behind
    ninety filler modules named for all of it.

    The control is the same run with a finding that names no symbol, and it is what makes
    this a statement about the lookup rather than about the checkout: the definition is
    absent there and present here, and the only difference is a name in a sentence.
    """
    root = cross_module_dependency_repository(tmp_path / "control")
    control = _RecordingCodingExecutor({"src/app.js": _leaf_module()})
    await _run_engineer(
        root,
        control,
        assigned=["src/app.js"],
        review=_blocking_review("The listing endpoint returns no cursor for an empty page."),
        retry_count=1,
    )
    assert CROSS_MODULE_DEFINITION not in _delivered_paths(control), (
        "the control must not deliver the definition, or this test proves nothing"
    )

    root = cross_module_dependency_repository(tmp_path / "measured")
    measured = _RecordingCodingExecutor({"src/app.js": _leaf_module()})
    await _run_engineer(
        root,
        measured,
        assigned=["src/app.js"],
        review=_blocking_review(
            "This duplicates work that already exists: pageBody already describes the keys "
            "an envelope carries, and this change declares them a second time."
        ),
        retry_count=1,
    )

    delivered = _delivered_paths(measured)
    assert CROSS_MODULE_DEFINITION in delivered
    assert "pageBody" in _delivered_text(measured), (
        "the definition has to arrive as content, not as an inventory path"
    )
    snapshot = json.loads(measured.calls[0]["input_text"])["repository_context"]
    assert CROSS_MODULE_DEFINITION not in snapshot["required_omitted_paths"]
    assert CROSS_MODULE_DEFINITION not in snapshot["required_dropped_paths"]


async def test_a_definition_withheld_for_key_material_is_reported_rather_than_dropped(
    tmp_path: Path,
) -> None:
    """A4. Withheld is still reported, so the artifact says why the finding is not fixable.

    `src/support/signingKeys.js` defines the symbol the finding names and also holds a private
    key. Being required is not permission to leak, so its bytes are withheld -- and being
    withheld silently would leave "the attempt never saw the file it was told about" to be
    re-derived from budget arithmetic, which is the question this channel exists to answer.

    So the file is absent from what was delivered, the key material is absent from the whole
    prompt, and the path is named in `required_omitted_paths`. The finding is simply not
    fixable by that attempt, and the record says so.
    """
    root = key_bearing_module_repository(tmp_path / "checkout")
    executor = _RecordingCodingExecutor({"src/routes/status.js": _committed_route()})

    await _run_engineer(
        root,
        executor,
        assigned=["src/routes/status.js"],
        review=_blocking_review(
            f"The route builds its own token instead of using {KEY_BEARING_SYMBOL}, which "
            "already carries the algorithm this endpoint is required to sign with."
        ),
        retry_count=1,
    )

    delivered = _delivered_paths(executor)
    assert CREDENTIAL_SECRET not in _delivered_text(executor), (
        "no key material may reach a model request, by any route"
    )
    assert KEY_BEARING_DEFINITION not in delivered
    snapshot = json.loads(executor.calls[0]["input_text"])["repository_context"]
    # A key-material withholding is a DROP -- the bytes reached no prompt -- and it says so
    # under its own reason, never as an excerpt. The union field keeps naming it for one
    # release so existing readers of `required_omitted_paths` see what they always saw.
    assert KEY_BEARING_DEFINITION in snapshot["required_dropped_paths"], (
        "a required path that was withheld must be named, or the withholding is silent"
    )
    assert snapshot["required_dropped_reasons"][KEY_BEARING_DEFINITION] == "key_material"
    assert KEY_BEARING_DEFINITION not in snapshot["required_excerpted_paths"]
    assert KEY_BEARING_DEFINITION in snapshot["required_omitted_paths"], (
        "a required path that was withheld must be named, or the withholding is silent"
    )


async def test_an_assigned_key_bearing_file_refuses_the_attempt_naming_redaction(
    tmp_path: Path,
) -> None:
    """A4's sharper case: when the withheld file is the assignment itself, nothing can run.

    A4's finding is merely *informed by* the key-bearing file, so the attempt proceeds and
    the record says why the finding is unfixable. Here the plan assigns that file, so every
    line the attempt could write about it would be a guess -- the attempt is refused before
    the model is called, and the refusal names the redaction policy, not the context budget:
    the drop is correct and permanent, and no budget change would ever include the file.
    """
    root = key_bearing_module_repository(tmp_path / "checkout")
    executor = _RecordingCodingExecutor({"src/routes/status.js": _committed_route()})

    with pytest.raises(RequiredContextRefusal) as raised:
        await _run_engineer(
            root,
            executor,
            assigned=[KEY_BEARING_DEFINITION, "src/routes/status.js"],
        )

    # Before the model call: nothing was delivered, so no key material could leak and no
    # attempt budget was spent.
    assert executor.calls == []
    text = " ".join(raised.value.diagnostics)
    assert KEY_BEARING_DEFINITION in text
    assert "redaction" in text
    assert "key material" in text
    assert CREDENTIAL_SECRET not in text, "a refusal must not quote what it refused to show"


async def test_no_definition_region_ever_carries_key_material_into_a_repair(
    tmp_path: Path,
) -> None:
    """F1. The repair half of the same rule, where the policy is different and the answer is not.

    The context snapshot withholds and reports; a repair prompt has nowhere to report to and
    simply quotes nothing. Both halves are the same predicate over the same bytes, and neither
    is relaxed because the file was named by evidence: the diagnostic asked for a definition,
    the definition is in a file holding a private key, and the pass is left to proceed on the
    diagnostics alone exactly as it did before this lookup existed.

    Scoped to what this change introduces -- the definition-site regions. The redaction of
    validation output and of reviewer evidence is established elsewhere and unchanged here.
    """
    harness = await build_harness(tmp_path, key_bearing_module_repository)
    signing_route = (
        "function signRoute(request, response) {\n"
        "  response.json({ algorithm: signingCredentials.algorithm });\n"
        "}\n"
        "\n"
        "module.exports = { signRoute };\n"
    )
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the signing route and register it.",
                "files": [
                    {"path": "src/routes/sign.js", "content": signing_route},
                    {
                        "path": "src/index.js",
                        "content": _index_with("./routes/sign", "signRoute"),
                    },
                ],
            },
            {
                "summary": "Import the signing credentials from the module that exports them.",
                "files": [
                    {
                        "path": "src/routes/sign.js",
                        "content": (
                            "const { signingCredentials } = "
                            "require('../support/signingKeys');\n\n" + signing_route
                        ),
                    }
                ],
            },
        ]
    )
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            review_payload=review_payload(verdict="approved"),
        )

        assert execution.result.status == "approved", execution.result.blocking_issues
        assert len(engineer.calls) == 2
        repair_instructions, repair_input = engineer.calls[1]
        # The diagnostic named the symbol, the lookup found where it is defined, and the
        # region was dropped rather than quoted -- so there is no definition block at all.
        assert "'signingCredentials' is not defined" in repair_instructions
        assert "Where this checkout defines the names" not in repair_instructions
        for delivered in (repair_instructions, repair_input, *engineer.calls[0]):
            assert CREDENTIAL_SECRET not in delivered
            assert "BEGIN RSA PRIVATE KEY" not in delivered
        # And withholding it cost the attempt nothing else: the repair still landed and the
        # repository's own linter accepts the worktree.
        rerun = _npm_lint(harness.workspace)
        assert rerun.returncode == 0, f"{rerun.stdout}\n{rerun.stderr}"
    finally:
        await harness.dispose()


# --------------------------------------------------------------------------------------
# 47- Part B: the repository's own typecheck runs inside the attempt
#
# The Engineer's exit condition was *lint passed*. So a type error -- however localized,
# however visible in the file the attempt had just written -- was first seen by the reviewer,
# a full outer attempt and a 15-40 minute re-implementation later. This is the clean half of
# that problem: a type error is unambiguous in a way a failing test is not, so it needs no
# guard of the kind Part C required, only a place to be repaired.
# --------------------------------------------------------------------------------------


def _typed_metrics_route(*, total: str) -> str:
    """The module under test, with its total either agreeing with its declaration or not.

    `total="'3'"` is B1's defect and it is exactly the shape a repair pass can clear: one
    value in the file the attempt just wrote, disagreeing with the type that same change
    declared for it.
    """
    return (
        f"const metricsTotal = {total};\n\n"
        "function metricsRoute(request, response) {\n"
        "  response.json({ count: 1, total: metricsTotal });\n"
        "}\n\n"
        "module.exports = { metricsRoute, metricsTotal };\n"
    )


def _declared_types_with_metrics() -> str:
    """The checkout's type declarations, extended with the module this change adds."""
    declared = {
        "src/routes/status.js": {"statusRoute": "function"},
        "src/routes/catalog.js": {
            "CATEGORIES": "object",
            "buildEntry": "function",
            "catalogRoute": "function",
            "normalizeCategory": "function",
            "paginate": "function",
        },
        "src/routes/metrics.js": {"metricsRoute": "function", "metricsTotal": "number"},
    }
    return f"{json.dumps(declared, indent=2)}\n"


def _typed_metrics_attempt(*, total: str) -> dict[str, Any]:
    """One complete engineer attempt: the module, its declared shape, and its registration."""
    return {
        "summary": "Add the metrics route and declare what it exports.",
        "files": [
            {"path": "src/routes/metrics.js", "content": _typed_metrics_route(total=total)},
            {"path": DECLARED_TYPES, "content": _declared_types_with_metrics()},
            {"path": "src/index.js", "content": _METRICS_INDEX},
        ],
    }


def _npm_typecheck(root: Path) -> subprocess.CompletedProcess[str]:
    """Run the checkout's own typecheck command for real, as the platform would have run it."""
    return subprocess.run(
        ("npm", "run", "typecheck"),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "CI": "true"},
        check=False,
    )


async def test_a_type_error_is_repaired_inside_the_attempt_and_the_types_then_agree(
    tmp_path: Path,
) -> None:
    """B1. The repository's own typechecker rejects the change, and the repair clears it here.

    Before this the Engineer declared itself `completed` on lint alone, so this defect --
    a value that disagrees with the type the same change declared for it, in the file the
    model had just written -- was found by the reviewer and cost a whole outer attempt.

    The assertions are on effects: the repository's own typecheck command, run again against
    the worktree this attempt left behind, exits zero; the repair was told the typechecker
    rejected it rather than the linter; and the commands it ran are on the record, so a reader
    can tell an attempt whose types were checked from one whose were not.
    """
    harness = await build_harness(tmp_path, typechecked_repository)
    engineer = ScriptedLLMClient(
        [
            _typed_metrics_attempt(total="'3'"),
            {
                "summary": "Report the total as the number its declaration promises.",
                "files": [
                    {"path": "src/routes/metrics.js", "content": _typed_metrics_route(total="3")}
                ],
            },
        ]
    )
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            review_payload=review_payload(verdict="approved"),
        )

        assert execution.result.status == "approved", execution.result.blocking_issues
        rerun = _npm_typecheck(harness.workspace)
        assert rerun.returncode == 0, f"{rerun.stdout}\n{rerun.stderr}"
        # Inside one attempt: the child executor ran once, the Engineer was asked twice.
        assert len(engineer.calls) == 2
        assert execution.result.failure_classification is None
        completion = execution.code_completion
        assert completion is not None
        assert completion.completion_status == "completed"
        assert completion.metadata["source_repair_passes"] == 1
        assert completion.metadata["source_repair_paths"] == ["src/routes/metrics.js"]
        assert completion.metadata["typecheck_commands"] == ["npm run typecheck"]
        # The repair was told what actually rejected its work. A pass that thinks the linter
        # rejected a type error looks for a style defect that is not there.
        repair_instructions = engineer.calls[1][0]
        assert "rejected by its own typechecker" in repair_instructions
        assert "TS2322" in repair_instructions
        # And told the one way this check could be silenced rather than answered.
        assert "not to widen a type to `any`" in repair_instructions
        # And the worktree holds the fix, in the committed tree.
        assert "const metricsTotal = 3;" in (harness.worktree_file("src/routes/metrics.js") or "")
        assert harness.branch in harness.remote_branches()
    finally:
        await harness.dispose()


async def test_a_type_error_the_loop_cannot_clear_still_reaches_the_retry_policy(
    tmp_path: Path,
) -> None:
    """B2. A typecheck failure the repair cannot resolve leaves the attempt, exactly as today.

    Two readings of "not mechanical" are worth separating, and both are covered. A run that
    could not produce a verdict at all -- a timeout, an exhausted machine -- is never admitted
    to this loop in the first place, which is asserted in the default tier where a timeout can
    be stated directly. The reading asserted here is the one that reaches an operator: a
    genuine type error that a repair pass does not fix.

    The stop is the same one that has always governed lint. The pass rewrote the file and the
    diagnostic did not change, so another identical telling would produce another identical
    answer; the loop stops on the spot rather than buying a second pass with the same
    question, and the attempt fails with the typechecker's own words in its blocking issues.
    """
    harness = await build_harness(tmp_path, typechecked_repository)
    engineer = ScriptedLLMClient(
        [
            _typed_metrics_attempt(total="'3'"),
            {
                "summary": "Rewrite the total.",
                # Still a string, so the typechecker says exactly what it said before.
                "files": [
                    {"path": "src/routes/metrics.js", "content": _typed_metrics_route(total="'4'")}
                ],
            },
        ]
    )
    reviewer = ScriptedLLMClient([review_payload(verdict="approved")])
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            reviewer_client=reviewer,
        )

        result = execution.result
        assert result.status == "failed"
        assert (
            result.failure_classification == FailureClassification.VALIDATION_SOURCE_FAILURE.value
        )
        assert result.current_revision
        assert any("TS2322" in issue for issue in result.blocking_issues), result.blocking_issues
        # The loop stopped on the repeat rather than spending its whole ceiling on it.
        assert len(engineer.calls) == 2
        # Nothing reached review and nothing was published, which is the point of the escape.
        assert reviewer.calls == []
        assert harness.branch not in harness.remote_branches()
        # The rejected attempt is still recorded, so the next one can edit these files rather
        # than regenerate them -- and it is `failed`, so nothing can publish it.
        completion = execution.code_completion
        assert completion is not None
        assert completion.completion_status == "failed"
        assert completion.metadata["source_validation_rejected"] is True
    finally:
        await harness.dispose()


# --------------------------------------------------------------------------------------
# Review evidence for a file too large to show whole (51- Part A)
# --------------------------------------------------------------------------------------


def _large_constants_lineage(root: Path, *, added: Sequence[str]) -> None:
    """Commit these entries into the oversized module as an earlier approved attempt would.

    Run 186's shape, reproduced against a checkout whose own commands run: the workflow
    branches from the default branch, an attempt adds a handful of lines to a module far too
    long to be quoted whole, and that attempt is approved and committed. Every later attempt in
    the lineage is then reviewed against a `HEAD` that already contains the change.
    """
    start_working_branch(root, "workflow/workflow-1")
    (root / LARGE_CONSTANTS_MODULE).write_text(
        large_constants_module_source(extra_entries=added), encoding="utf-8"
    )
    commit_all(root, "an earlier approved attempt")


async def _review_large_constants(root: Path) -> tuple[dict[str, Any], Any, list[str]]:
    """Review a lineage carrying the committed oversized module, plus one live change.

    Returns the evidence the model was sent, the published review, and the reviewed paths --
    the last so the publication gate can be asked the same question the reviewer answered.
    """
    first = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={
                    LARGE_CONSTANTS_MODULE: (root / LARGE_CONSTANTS_MODULE).read_text(
                        encoding="utf-8"
                    )
                }
            ),
        ).run(agent_state(root, [task_plan_artifact()]))
    )["artifacts"][0]
    second = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={
                    "src/routes/metrics.js": (
                        "const { CONSTANTS } = require('../support/constants');\n\n"
                        "function metricsRoute(request, response) {\n"
                        "  response.json({ entries: Object.keys(CONSTANTS).length });\n"
                        "}\n\nmodule.exports = { metricsRoute };\n"
                    )
                }
            ),
        ).run(agent_state(root, [task_plan_artifact(), first]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))
    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    ).run(agent_state(root, [technical_prd_artifact(), task_plan_artifact(), second]))
    evidence = json.loads(client.calls[0][1])["workspace_change_evidence"]
    review = update["artifacts"][0]
    return evidence, review, [change.path for change in second.file_changes]


async def test_a_small_change_to_an_oversized_file_is_reviewed_on_its_merits(
    tmp_path: Path,
) -> None:
    """The run 186 replay: four committed lines in a 37 KB module reach the review.

    This is the defect that single-handedly cost the first anthropic delivery. The frontend's
    integration-fix attempts 2, 3 and 4 were each refused with the same platform-authored
    finding -- the constants module was head-truncated, `truncated_content` was recorded, and a
    deterministic gate is required to refuse on it. No retry could clear it: the instruction was
    to split a shared constants module, and the module stays large however many attempts try.
    """
    root = large_constants_repository(tmp_path / "checkout")
    added = [f"  BULK_IMPORT_LIMIT_{index}: {index * 10}," for index in range(4)]
    _large_constants_lineage(root, added=added)
    # Asserted rather than assumed, because it is the precondition for the whole defect: by the
    # time this attempt is reviewed, `git diff HEAD` reports nothing at all for the file.
    against_head = subprocess.run(
        ("git", "diff", "HEAD", "--", LARGE_CONSTANTS_MODULE),
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert against_head.stdout == ""

    evidence, review, _paths = await _review_large_constants(root)

    entry = next(item for item in evidence["files"] if item["path"] == LARGE_CONSTANTS_MODULE)
    assert entry["content_kind"] == "changed_regions"
    # Every added line is inside a quoted region, and the regions say where they are.
    for line in added:
        assert line in entry["content"]
    body = (root / LARGE_CONSTANTS_MODULE).read_text(encoding="utf-8").splitlines()
    changed_numbers = [body.index(line) + 1 for line in added]
    covered = [
        number
        for number in changed_numbers
        if any(
            region["excerpt_lines"]["start"] <= number <= region["excerpt_lines"]["end"]
            for region in entry["changed_regions"]
        )
    ]
    assert covered == changed_numbers
    assert entry["hunks_omitted"] == 0
    # No limitation, so the deterministic gate never fires, so the verdict is the model's.
    assert evidence["limitations"] == []
    assert review.verdict == "approved"
    assert [item.finding_id for item in review.findings] != ["REVIEW_EVIDENCE_INCOMPLETE"]


async def test_a_change_the_reviewer_cannot_see_still_refuses_against_a_real_checkout(
    tmp_path: Path,
) -> None:
    """The discipline is an invariant of the fix, not a casualty of it.

    Excerpting exists so a small change to a large file is reviewable. It must not become a way
    for an unreviewable change to pass, and a single contiguous block bigger than the whole
    per-file budget is exactly that. The verdict is still forced, and the sentence now names
    the lines that were hidden.
    """
    root = large_constants_repository(tmp_path / "checkout")
    giant = [
        f"  GENERATED_{index:05d}: 'generated-payload-value-{index:05d}'," for index in range(900)
    ]
    _large_constants_lineage(root, added=giant)

    evidence, review, _paths = await _review_large_constants(root)

    entry = next(item for item in evidence["files"] if item["path"] == LARGE_CONSTANTS_MODULE)
    assert entry["hunks_omitted"] == 1
    assert entry["changed_regions"] == []
    assert "truncated_content" in evidence["limitations"]
    assert review.verdict == "changes_requested"
    finding = next(
        item for item in review.findings if item.finding_id == "REVIEW_EVIDENCE_INCOMPLETE"
    )
    body = (root / LARGE_CONSTANTS_MODULE).read_text(encoding="utf-8").splitlines()
    first_line = body.index(giant[0]) + 1
    last_line = body.index(giant[-1]) + 1
    assert f"{LARGE_CONSTANTS_MODULE} lines {first_line}-{last_line}" in finding.description
    assert "into smaller modules" not in finding.recommendation


async def test_the_publication_binding_is_untouched_by_excerpted_evidence(
    tmp_path: Path,
) -> None:
    """The fingerprint is still every reviewed byte on disk, and excerpting did not weaken it.

    `reviewed_content_fingerprint` never sees `model_input`, and it must not start to: it is
    what the publication gate binds, and 48- had just finished fixing that gate. So the same
    approved review that was reached on an excerpt still refuses a single byte changed
    afterwards -- including one in the part of the large file the reviewer was never shown.
    """
    root = large_constants_repository(tmp_path / "checkout")
    _large_constants_lineage(root, added=["  AUDIT_RETENTION_DAYS: 30,"])

    _evidence, review, paths = await _review_large_constants(root)

    assert review.verdict == "approved"
    require_reviewed_workspace_match(
        review,
        reviewed_paths=paths,
        content_fingerprint=reviewed_content_fingerprint(root, paths),
        evidence_required=True,
    )
    # One byte, in the elided remainder rather than in any quoted region, and the gate refuses.
    module = root / LARGE_CONSTANTS_MODULE
    module.write_text(module.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(AgentArtifactError, match="no longer matches"):
        require_reviewed_workspace_match(
            review,
            reviewed_paths=paths,
            content_fingerprint=reviewed_content_fingerprint(root, paths),
            evidence_required=True,
        )


async def test_excerpting_a_changed_region_is_not_a_way_around_redaction(
    tmp_path: Path,
) -> None:
    """A quoted region is sliced out of source already redacted whole, so it cannot leak.

    The check that matters is the negative one: the literal must be absent from the delivered
    request. A credential beside a changed line is exactly the case an excerpting path could
    have got wrong, because the whole-file redaction it used to rely on no longer decides what
    is sent.
    """
    root = large_constants_repository(tmp_path / "checkout")
    _large_constants_lineage(
        root,
        added=[
            "  ANALYTICS_TOKEN: 'sk-proj-EXAMPLESECRET123456789',",
            "  ANALYTICS_ENABLED: true,",
        ],
    )

    evidence, _review, _paths = await _review_large_constants(root)

    entry = next(item for item in evidence["files"] if item["path"] == LARGE_CONSTANTS_MODULE)
    assert entry["content_kind"] == "changed_regions"
    # The changed line reached the reviewer; the secret on it did not.
    assert "sk-proj-EXAMPLESECRET123456789" not in json.dumps(evidence)
    assert "ANALYTICS_TOKEN" in entry["content"]
    assert "[REDACTED]" in entry["content"]
    assert entry["redacted"] is True
    # And the neighbouring changed line is intact, so redaction cost the review nothing else.
    assert "ANALYTICS_ENABLED: true," in entry["content"]


# --------------------------------------------------------------------------------------
# A lockfile is not a secret (53-)
# --------------------------------------------------------------------------------------


_LOCKFILE_ADDED_PACKAGES = tuple(f"declared-dependency-{index:03d}" for index in range(120))


def _lockfile_lineage(root: Path) -> None:
    """Branch off the default branch, as every workstream's attempts do.

    The branch point is what the numstat below is measured against, and it has to be a real
    one: an unresolvable baseline is the case `_LineageBaseRevision.is_branch_point` fences,
    and it reports no line count at all rather than a wrong one.
    """
    start_working_branch(root, "workflow/workflow-1")


def _churn_lockfile(root: Path, *, packages: Sequence[str]) -> None:
    """Rewrite the lockfile on disk as the repository's own install would have.

    Written directly rather than through the coding executor, because that is where a churned
    lockfile actually comes from and the difference is the whole defect. `npm install` runs
    when the engineer changed a manifest, rewrites the lockfile itself, and `installed_lockfiles`
    then adds the path to the change set -- which is also the only way a file this size can get
    there at all: every coding executor validates its own updates against the same 1 MiB limit
    and would refuse to write one.
    """
    (root / OVERSIZED_LOCKFILE).write_text(
        oversized_lockfile_source("lockfile-carrying-service", extra_packages=packages),
        encoding="utf-8",
    )


async def _review_lockfile_change(
    root: Path, updates: dict[str, str]
) -> tuple[dict[str, Any], Any, list[str]]:
    """Run one attempt writing these files and review what the workspace then holds.

    The engineer is the real one, so a manifest in `updates` is what pulls the lockfile beside
    it into the change set, through `installed_lockfiles` and for the reason that function
    exists. Returns the evidence the model was actually sent, the published review, and the
    reviewed paths -- the last so the publication gate can be asked the same question the
    review answered.
    """
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(file_updates=updates),
        ).run(agent_state(root, [task_plan_artifact()]))
    )["artifacts"][0]
    return (*await _review_completion(root, completion), [c.path for c in completion.file_changes])


async def _review_completion(root: Path, completion: Any) -> tuple[dict[str, Any], Any]:
    """Review this completion against the checkout, returning its evidence and its verdict."""
    client = StaticLLMClient(domain_payload(review_artifact()))
    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    ).run(agent_state(root, [technical_prd_artifact(), task_plan_artifact(), completion]))
    return json.loads(client.calls[0][1])["workspace_change_evidence"], update["artifacts"][0]


def _completion_naming(*paths: str) -> CodeCompletionArtifact:
    """Return a completion whose change set is exactly these paths, and nothing else.

    Used where the change set under test is one the engineer cannot currently produce, so that
    the reviewer is asked the question directly rather than through a path that would quietly
    answer it first. The reviewer's contract is to handle whatever change set it is given.
    """
    return create_artifact(
        CodeCompletionArtifact,
        workflow_id="workflow-1",
        artifact_id="006_code_completion.json",
        producer="engineer",
        payload={
            "completion_status": "completed",
            "summary": "An attempt whose change set is stated directly.",
            "file_changes": [
                {"path": path, "change_type": "modified", "description": "changed"}
                for path in paths
            ],
            "validation_results": [],
            "test_coverage_percent": None,
            "remaining_work": [],
            "commit_sha": None,
        },
        metadata={"source": "test"},
    )


def _manifest_with_test_flag(root: Path) -> str:
    """Return the manifest with a flag added to its test script, as run 189's attempt did."""
    manifest = json.loads((root / OVERSIZED_LOCKFILE_MANIFEST).read_text(encoding="utf-8"))
    scripts = manifest["scripts"]
    scripts["test"] = f"{scripts['test']} --test-concurrency=1"
    return f"{json.dumps(manifest, indent=2)}\n"


async def test_a_churned_lockfile_no_longer_ends_the_workstream(tmp_path: Path) -> None:
    """The run 189 replay: a manifest edit and the lockfile churn it caused both survive.

    Run 189's frontend edited `package.json` -- a flag on its test script -- the platform's own
    dependency install churned `package-lock.json` exactly as designed, and the workstream
    terminated. 1,290,320 bytes is past the reviewer's read cap, an unreadable file joins the
    class reserved for a human decision, and the retry policy then refuses every attempt. One
    machine-generated file, treated with the severity reserved for a credential, and every
    dependency-affecting change on a Node repository died the same way.

    The lockfile now arrives as what can be known about it without reading it. No limitation,
    so no deterministic gate fires, so the verdict is the review model's own.
    """
    missing = missing_executables("git")
    if missing:
        assert not requires_real_repository_tier(), unavailable_reason(missing)
        pytest.skip(unavailable_reason(missing))
    root = oversized_lockfile_repository(tmp_path / "checkout")
    _lockfile_lineage(root)
    # The precondition, asserted rather than assumed: this is the file the read cap refuses.
    assert (root / OVERSIZED_LOCKFILE).stat().st_size > DEFAULT_MAX_FILE_BYTES

    manifest = _manifest_with_test_flag(root)
    _churn_lockfile(root, packages=_LOCKFILE_ADDED_PACKAGES)
    evidence, review, paths = await _review_lockfile_change(
        root, {OVERSIZED_LOCKFILE_MANIFEST: manifest}
    )

    # The lockfile reached the change set the way it does in a deployment: the manifest moved,
    # so the install ran, so `installed_lockfiles` named it. Nothing in the test put it there.
    assert paths == [OVERSIZED_LOCKFILE_MANIFEST, OVERSIZED_LOCKFILE]
    entry = next(item for item in evidence["files"] if item["path"] == OVERSIZED_LOCKFILE)
    assert entry["content_kind"] == "generated_lockfile"
    assert entry["manifest_changed"] is True
    assert entry["manifest_path"] == OVERSIZED_LOCKFILE_MANIFEST
    assert entry["size_bytes"] == (root / OVERSIZED_LOCKFILE).stat().st_size
    assert entry["sha256"] == hashlib.sha256((root / OVERSIZED_LOCKFILE).read_bytes()).hexdigest()
    # The numstat is Git's own answer, asked the same way against the same branch point.
    expected = subprocess.run(
        ("git", "diff", "--numstat", "main", "--", OVERSIZED_LOCKFILE),
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.split("\t")
    assert [entry["insertions"], entry["deletions"]] == [int(expected[0]), int(expected[1])]
    assert entry["insertions"] > 0
    # Item 19: the install resolved 120 more packages from the registry the repository already
    # used, which is what an install does. The comparison was made -- an empty list, not a
    # null -- and it found nothing, so the entry says nothing happened rather than saying
    # nothing.
    assert entry["resolved_host_scan"] == {
        "format": "npm",
        "scanned": True,
        "reason": None,
        "baseline_present": True,
        "distinct_hosts": 1,
        "new_hosts": [],
        "hosts_truncated": False,
    }
    # Neither class the old chain ran through is present.
    assert "unavailable_content" not in evidence["limitations"]
    assert "truncated_content" not in evidence["limitations"]
    assert evidence["limitations"] == []
    assert evidence["manual_review_required"] is False
    # The verdict is the review model's, and the metadata the retry policy reads does not
    # refuse: `manual_review_required` is the only thing that turns an attempt terminal there.
    assert review.verdict == "approved"
    assert review.metadata["manual_review_required"] is False
    assert review.metadata["retryable"] is True
    # And the manifest edit beside it was reviewed on its merits, whole.
    manifest_entry = next(
        item for item in evidence["files"] if item["path"] == OVERSIZED_LOCKFILE_MANIFEST
    )
    assert manifest_entry["content_kind"] == "utf8_source"
    assert "--test-concurrency=1" in manifest_entry["content"]


async def test_a_lockfile_that_moved_without_its_manifest_is_asked_about_not_waved_through(
    tmp_path: Path,
) -> None:
    """Orphan lockfile churn stays suspicious -- and stays retryable.

    The carve-out is not "lockfiles are exempt". A lockfile that moves with the manifest that
    explains it is ordinary; one that moves alone is a dependency change no declaration in this
    change accounts for, and the reviewer should say so. What it must not be is the unreadable
    class: this is a finding a retry can resolve, by making the declaration or by reverting the
    churn, and the verdict therefore goes no further than `changes_requested`.

    The change set is stated directly because today's engineer cannot produce this one: a
    lockfile this size reaches the reviewer only through `installed_lockfiles`, which fires
    only for a changed manifest, and every coding executor refuses to write a file past the
    same 1 MiB limit. That makes this the fail-safe rather than a live production shape, and
    it is the reason `manifest_changed` is on the entry at all -- without it the carve-out
    would exempt any oversized lockfile from review regardless of what put it there, and the
    first upstream change that widens one of those two constraints would be silent.
    """
    missing = missing_executables("git")
    if missing:
        assert not requires_real_repository_tier(), unavailable_reason(missing)
        pytest.skip(unavailable_reason(missing))
    root = oversized_lockfile_repository(tmp_path / "checkout")
    _lockfile_lineage(root)

    _churn_lockfile(root, packages=_LOCKFILE_ADDED_PACKAGES)
    evidence, review = await _review_completion(root, _completion_naming(OVERSIZED_LOCKFILE))

    entry = next(item for item in evidence["files"] if item["path"] == OVERSIZED_LOCKFILE)
    assert entry["content_kind"] == "generated_lockfile"
    assert entry["manifest_changed"] is False
    assert evidence["limitations"] == ["orphan_generated_lockfile"]
    # Retryable, not terminal: the verdict is forced no further than `changes_requested`, and
    # the metadata the retry policy reads leaves the attempt eligible for another try.
    assert evidence["manual_review_required"] is False
    assert review.verdict == "changes_requested"
    assert review.metadata["manual_review_required"] is False
    assert review.metadata["retryable"] is True
    finding = next(
        item for item in review.findings if item.finding_id == "REVIEW_EVIDENCE_INCOMPLETE"
    )
    # The finding names the file and asks the question the change actually raises. It must not
    # be the budget sentence: nothing about this evidence was cut short.
    assert finding.file_path == OVERSIZED_LOCKFILE
    assert OVERSIZED_LOCKFILE in finding.description
    assert "manifest" in finding.recommendation
    assert "smaller, separately reviewable" not in finding.recommendation


async def test_the_lockfile_carve_out_does_not_leak_to_anything_else(tmp_path: Path) -> None:
    """The unreadable class stays terminal for everything that is not this exact carve-out.

    Both halves of the fence, because both are ways it could have leaked. A file that is merely
    large is not a lockfile and gets no help from the registry; and a file that *is* a
    registry lockfile, but whose bytes cannot be had for a reason other than its size, must
    take the unreadable path exactly as it does today. The second is why size is decided by
    `stat` before any read is attempted and why the streaming pass is allowed to fail: an
    exception message is never parsed to work out what went wrong.
    """
    missing = missing_executables("git")
    if missing:
        assert not requires_real_repository_tier(), unavailable_reason(missing)
        pytest.skip(unavailable_reason(missing))
    root = oversized_lockfile_repository(tmp_path / "checkout")
    _lockfile_lineage(root)

    # (a) Past the cap, and not a lockfile. Exactly today's terminal behaviour.
    bulk = (
        "const BULK = [\n"
        + "\n".join(f"  'row-{index:07d}'," for index in range(90_000))
        + "\n];\n"
    )
    assert len(bulk.encode("utf-8")) > DEFAULT_MAX_FILE_BYTES
    (root / "src" / "support").mkdir(parents=True, exist_ok=True)
    (root / "src" / "support" / "bulk.js").write_text(bulk, encoding="utf-8")
    evidence, review = await _review_completion(root, _completion_naming("src/support/bulk.js"))

    entry = next(item for item in evidence["files"] if item["path"] == "src/support/bulk.js")
    assert entry["content_kind"] == "unavailable"
    assert entry["content"] == "[CONTENT UNAVAILABLE]"
    assert "unavailable_content" in evidence["limitations"]
    assert evidence["manual_review_required"] is True
    assert review.verdict == "rejected"
    assert review.metadata["retryable"] is False


async def test_a_lockfile_unreadable_for_any_reason_but_size_stays_terminal(
    tmp_path: Path,
) -> None:
    """(b) A registry lockfile whose read fails for something other than size.

    Split from the case above because it is the sharper one: the size condition holds, the
    registry condition holds, and the carve-out must still decline -- the bytes genuinely
    cannot be had, and a permissions failure on a dependency file is exactly the kind of broken
    checkout a person should look at. What proves the difference is the streaming pass itself
    failing, not a string read out of an exception.
    """
    missing = missing_executables("git")
    if missing:
        assert not requires_real_repository_tier(), unavailable_reason(missing)
        pytest.skip(unavailable_reason(missing))
    if os.geteuid() == 0:
        pytest.skip("running as root, where a mode of 000 does not prevent a read")
    root = oversized_lockfile_repository(tmp_path / "checkout")
    _lockfile_lineage(root)

    _churn_lockfile(root, packages=_LOCKFILE_ADDED_PACKAGES)
    lockfile = root / OVERSIZED_LOCKFILE
    # Past the cap, so the size condition holds; and unreadable, so the carve-out must not.
    assert lockfile.stat().st_size > DEFAULT_MAX_FILE_BYTES
    lockfile.chmod(0o000)
    try:
        evidence, review = await _review_completion(
            root, _completion_naming(OVERSIZED_LOCKFILE_MANIFEST, OVERSIZED_LOCKFILE)
        )
    finally:
        lockfile.chmod(0o644)

    entry = next(item for item in evidence["files"] if item["path"] == OVERSIZED_LOCKFILE)
    assert entry["content_kind"] == "unavailable"
    assert "unavailable_content" in evidence["limitations"]
    assert "orphan_generated_lockfile" not in evidence["limitations"]
    assert evidence["manual_review_required"] is True
    assert review.verdict == "rejected"
    assert review.metadata["retryable"] is False


async def test_a_lockfile_small_enough_to_read_takes_the_ordinary_ladder(
    tmp_path: Path,
) -> None:
    """The carve-out activates only where the read cap would have ended the workstream.

    A lockfile a repository can actually show is source like any other, and showing it is
    strictly better evidence than describing it. Asserted because the cheap implementation of
    this spec -- keying on the filename alone -- would silently stop quoting every lockfile in
    every repository small enough to have one, and nothing else would notice.
    """
    missing = missing_executables("git")
    if missing:
        assert not requires_real_repository_tier(), unavailable_reason(missing)
        pytest.skip(unavailable_reason(missing))
    root = node_npm_repository(tmp_path / "checkout")
    _lockfile_lineage(root)

    lockfile = json.loads((root / OVERSIZED_LOCKFILE).read_text(encoding="utf-8"))
    lockfile["packages"]["node_modules/declared-dependency"] = {"version": "1.0.0"}
    evidence, _review, _paths = await _review_lockfile_change(
        root, {OVERSIZED_LOCKFILE: f"{json.dumps(lockfile, indent=2)}\n"}
    )

    entry = next(item for item in evidence["files"] if item["path"] == OVERSIZED_LOCKFILE)
    assert (root / OVERSIZED_LOCKFILE).stat().st_size <= DEFAULT_MAX_FILE_BYTES
    assert entry["content_kind"] == "utf8_source"
    assert "declared-dependency" in entry["content"]
    assert not any(item.get("content_kind") == "generated_lockfile" for item in evidence["files"])


async def test_a_registry_host_substituted_in_an_unread_lockfile_is_named_in_the_evidence(
    tmp_path: Path,
) -> None:
    """Item 19, end to end: the 189 shape with one resolution repointed at another registry.

    Same real checkout, same real engineer, same install-appended path, and the only difference
    from the replay above is a single `resolved` URL out of four thousand three hundred. 53-
    recorded exactly this as the gap it was not closing: the entry's facts -- size, numstat,
    digest -- move identically whether the lockfile was regenerated or tampered with, because
    the digest of a changed file changes either way.

    What must hold is both halves. The substituted host is named, and *only* it: the registry
    the repository already resolves from is not reported, because a scanner that restated the
    status quo as a finding would fire on every dependency change and be learned to be ignored.
    And it changes nothing else -- no limitation, so no deterministic gate, so the verdict is
    still the review model's own and retries are not refused. Whether a new registry is a
    mirror migration or an attack is a judgement, and this platform states the fact and leaves
    the judgement to the reviewer and the person reading the review.
    """
    missing = missing_executables("git")
    if missing:
        assert not requires_real_repository_tier(), unavailable_reason(missing)
        pytest.skip(unavailable_reason(missing))
    root = oversized_lockfile_repository(tmp_path / "checkout")
    _lockfile_lineage(root)

    manifest = _manifest_with_test_flag(root)
    _churn_lockfile(root, packages=_LOCKFILE_ADDED_PACKAGES)
    lockfile = root / OVERSIZED_LOCKFILE
    substituted = lockfile.read_text(encoding="utf-8").replace(
        f"https://registry.example.test/{_LOCKFILE_ADDED_PACKAGES[0]}/",
        f"https://registry.substituted.test/{_LOCKFILE_ADDED_PACKAGES[0]}/",
        1,
    )
    lockfile.write_text(substituted, encoding="utf-8")
    assert lockfile.stat().st_size > DEFAULT_MAX_FILE_BYTES

    evidence, review, _paths = await _review_lockfile_change(
        root, {OVERSIZED_LOCKFILE_MANIFEST: manifest}
    )

    entry = next(item for item in evidence["files"] if item["path"] == OVERSIZED_LOCKFILE)
    assert entry["content_kind"] == "generated_lockfile"
    assert entry["resolved_host_scan"] == {
        "format": "npm",
        "scanned": True,
        "reason": None,
        "baseline_present": True,
        "distinct_hosts": 2,
        "new_hosts": ["registry.substituted.test"],
        "hosts_truncated": False,
    }
    assert "registry.substituted.test" in entry["content"]
    # A fact, not a verdict: nothing about the platform's own gates moved.
    assert evidence["limitations"] == []
    assert evidence["manual_review_required"] is False
    assert review.verdict == "approved"
    assert review.metadata["retryable"] is True


async def test_the_publication_binding_still_speaks_for_a_lockfile_never_quoted(
    tmp_path: Path,
) -> None:
    """A file the review deliberately did not read is still bound byte for byte.

    This is what makes the carve-out safe to ship. `reviewed_content_fingerprint` streams every
    reviewed file's exact bytes and always did -- it has no cap and never had one -- so the
    lockfile the reviewer described rather than quoted is the same lockfile the publication
    gate refuses to see changed. The entry's own hash is that same value, which is why the two
    cannot drift apart.
    """
    missing = missing_executables("git")
    if missing:
        assert not requires_real_repository_tier(), unavailable_reason(missing)
        pytest.skip(unavailable_reason(missing))
    root = oversized_lockfile_repository(tmp_path / "checkout")
    _lockfile_lineage(root)

    manifest = _manifest_with_test_flag(root)
    _churn_lockfile(root, packages=_LOCKFILE_ADDED_PACKAGES)
    evidence, review, paths = await _review_lockfile_change(
        root, {OVERSIZED_LOCKFILE_MANIFEST: manifest}
    )

    entry = next(item for item in evidence["files"] if item["path"] == OVERSIZED_LOCKFILE)
    assert entry["content_kind"] == "generated_lockfile"
    assert review.verdict == "approved"
    require_reviewed_workspace_match(
        review,
        reviewed_paths=paths,
        content_fingerprint=reviewed_content_fingerprint(root, paths),
        evidence_required=True,
    )
    # One byte, inside the 1.3 MB the reviewer was never shown, and the gate refuses.
    lockfile = root / OVERSIZED_LOCKFILE
    lockfile.write_bytes(lockfile.read_bytes() + b"\n")
    with pytest.raises(AgentArtifactError, match="no longer matches"):
        require_reviewed_workspace_match(
            review,
            reviewed_paths=paths,
            content_fingerprint=reviewed_content_fingerprint(root, paths),
            evidence_required=True,
        )


# --------------------------------------------------------------------------------------
# A failed attempt keeps its repair record (51- Part B)
# --------------------------------------------------------------------------------------


def _metrics_module(*declarations: str, count: int = 1) -> str:
    """Return the metrics route with these leading declarations, for the fixture's linter.

    The declarations occupy the first lines, so which line a violation lands on is the test's
    to choose -- and it has to be, because the diagnostic signature that bounds the repair loop
    identifies a defect by where it is as well as by what it is.
    """
    return (
        "\n".join(declarations)
        + "\n\nfunction metricsRoute(request, response) {\n"
        + f"  response.json({{ count: {count} }});\n"
        + "}\n\nmodule.exports = { metricsRoute };\n"
    )


async def test_a_gate_rejected_attempt_records_the_repair_passes_that_ran(
    tmp_path: Path,
) -> None:
    """A rejected attempt is the one whose repair history matters most, and it used to lose it.

    `source_repairs=()` was hard-coded on the `SourceValidationError` completion path, so every
    gate-rejected attempt omitted the whole `source_repair_*` block -- including which model
    repaired and whether the scoped-fix role resolved. That is what made "does the anthropic
    scoped-fix loop engage at all?" unanswerable from runs 183 and 186: openai attempts showed
    the metadata because they *passed* after repairing, and anthropic's gate failures could not
    show it by construction.

    Here the loop genuinely runs twice on the scoped-fix boundary and genuinely loses -- the
    second pass reproduces the rule the first one introduced, so the signature repeats and the
    attempt fails on the gate's own diagnostics, exactly as before. What is new is the record.
    """
    harness = await build_harness(tmp_path, node_npm_repository)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the metrics route.",
                "files": [
                    {
                        "path": "src/routes/metrics.js",
                        "content": _metrics_module("const metrics_payload = { count: 1 };"),
                    },
                    {
                        "path": "src/index.js",
                        "content": (
                            "const { statusRoute } = require('./routes/status');\n"
                            "const { catalogRoute } = require('./routes/catalog');\n"
                            "const { metricsRoute } = require('./routes/metrics');\n\n"
                            "module.exports = { statusRoute, catalogRoute, metricsRoute };\n"
                        ),
                    },
                ],
            }
        ]
    )
    scoped = _NamedModelClient(
        [
            # Clears `camelcase` on line one and introduces `no-var` on line two, so the
            # diagnostics name a different defect in a different place and the loop is granted
            # another pass.
            {
                "summary": "Rename the payload and hoist the total.",
                "files": [
                    {
                        "path": "src/routes/metrics.js",
                        "content": _metrics_module(
                            "const metricsPayload = { count: 1 };", "var total = 3;"
                        ),
                    }
                ],
            },
            # Rewrites the file and leaves `no-var` exactly where it was, so the signature
            # repeats and the loop stops on the spot rather than buying a third pass.
            {
                "summary": "Rewrite the response, leaving the declaration alone.",
                "files": [
                    {
                        "path": "src/routes/metrics.js",
                        "content": _metrics_module(
                            "const metricsPayload = { count: 1 };", "var total = 3;", count=2
                        ),
                    }
                ],
            },
        ],
        model="scoped-fix-model",
    )
    reviewer = ScriptedLLMClient([])
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            reviewer_client=reviewer,
            scoped_fix_client=scoped,
        )

        # Nothing about the outcome changed: the gate rejected the attempt on its own
        # diagnostics, nothing reached review, and nothing was published.
        assert execution.result.status == "failed"
        assert (
            execution.result.failure_classification
            == FailureClassification.VALIDATION_SOURCE_FAILURE.value
        )
        assert any("no-var" in issue for issue in execution.result.blocking_issues), (
            execution.result.blocking_issues
        )
        assert reviewer.calls == []
        assert harness.branch not in harness.remote_branches()

        completion = execution.code_completion
        assert completion is not None
        assert completion.completion_status == "failed"
        assert completion.metadata["source_validation_rejected"] is True
        # The record the failure path used to discard: both passes, the model that ran them,
        # and the fact that the scoped-fix role resolved rather than the primary standing in.
        assert completion.metadata["source_repair_engaged"] is True
        assert completion.metadata["source_repair_passes"] == 2
        assert completion.metadata["source_repair_paths"] == ["src/routes/metrics.js"]
        assert completion.metadata["source_repair_execution"]["model"] == "scoped-fix-model"
        assert completion.metadata["source_repair_execution"]["scoped_fix_role_resolved"] is True
        assert completion.metadata["source_repair_scoped_fix_role_resolved"] is True
        # The passes were real: they ran on the scoped-fix boundary and the worktree holds
        # what the last one wrote.
        assert len(scoped.calls) == 2
        assert "count: 2" in (harness.worktree_file("src/routes/metrics.js") or "")
    finally:
        await harness.dispose()


# --------------------------------------------------------------------------------------
# 47- Part E: the Engineer reviews its own finished work before saying ready
# --------------------------------------------------------------------------------------


def _self_review_verdict(findings: list[dict[str, Any]]) -> dict[str, Any]:
    """One schema-valid self-review response carrying exactly these findings."""
    return {
        "summary": "Reviewed the finished implementation against its assignment.",
        "requirement_coverage": [
            {
                "requirement_id": "backend-status-api",
                "status": "partial" if findings else "implemented",
                "evidence": "src/routes/metrics.js implements the metrics payload.",
            }
        ],
        "findings": findings,
    }


async def test_a_missing_requirement_branch_is_corrected_inside_the_attempt(
    tmp_path: Path,
) -> None:
    """E1. Everything deterministic is green and the work is still short of its requirement.

    The route reports a count and no total, and the suite the same attempt wrote asserts
    exactly that -- so lint, format, the narrowed tests and the wiring inspection all accept
    an implementation that does not do what the requirement asks. That gap was structurally
    invisible until review, a full attempt later. Here the self-review names it, one bounded
    correction closes it, the re-run validation accepts the corrected files, and the same
    attempt reaches review and approval: the outer counter never moves.
    """
    harness = await build_harness(tmp_path, node_npm_repository)
    engineer = ScriptedLLMClient(
        [
            # Internally consistent and green -- and short of the requirement.
            _metrics_attempt(
                module="../src/routes/metrics", expected="{ count: 1 }", with_total=False
            ),
            {
                "summary": "Add the total the requirement asks for, and assert it.",
                "files": [
                    {"path": "src/routes/metrics.js", "content": _metrics_route(with_total=True)},
                    {
                        "path": "test/metrics.test.js",
                        "content": _metrics_suite(module="../src/routes/metrics"),
                    },
                ],
            },
        ]
    )
    self_review = ScriptedLLMClient(
        [
            _self_review_verdict(
                [
                    {
                        "description": (
                            "The metrics payload omits the total the requirement asks "
                            "for; src/routes/metrics.js reports only the count."
                        ),
                        "classification": "localized",
                        "paths": ["src/routes/metrics.js", "test/metrics.test.js"],
                        "requirement_id": "backend-status-api",
                    }
                ]
            )
        ]
    )
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            self_review_client=self_review,
            review_payload=review_payload(verdict="approved"),
        )

        assert execution.result.status == "approved", execution.result.blocking_issues
        # The effect: the corrected route and suite are what the worktree holds, and the
        # repository's own narrowed command accepts them.
        assert "total: 3" in (harness.worktree_file("src/routes/metrics.js") or "")
        assert "total: 3" in (harness.worktree_file("test/metrics.test.js") or "")
        rerun = _npm_test(harness.workspace, "--", "test/metrics.test.js")
        assert rerun.returncode == 0, f"{rerun.stdout}\n{rerun.stderr}"
        # One outer attempt: the implementation and its correction, no retry in between.
        assert execution.result.metadata["child_retry_count"] == 1
        assert len(engineer.calls) == 2
        # Exactly one review pass, corrections notwithstanding: no review-of-a-review.
        assert len(self_review.calls) == 1
        completion = execution.code_completion
        assert completion is not None
        assert completion.completion_status == "completed"
        record = completion.metadata["self_review"]
        assert record["outcome"] == "corrected"
        assert sorted(record["corrections_applied"]) == [
            "src/routes/metrics.js",
            "test/metrics.test.js",
        ]
        assert len(record["findings"]) == 1
        # The correction pass was told exactly what the review found, and no more.
        assert SELF_REVIEW_DIAGNOSTIC_PREFIX in engineer.calls[1][0]
        # The corrected work was published like any other approved attempt.
        assert harness.branch in harness.remote_branches()
    finally:
        await harness.dispose()


async def test_a_passing_self_review_approves_nothing(tmp_path: Path) -> None:
    """E2. Self-review gates `completion_status` and nothing further.

    A review pass that finds nothing must change nothing downstream: the commit gate still
    stages and commits through the publisher, and the independent Reviewer still reads the
    diff and renders the verdict. An Engineer that both writes the code and clears its own
    publication would be no gate at all.
    """
    harness = await build_harness(tmp_path, node_npm_repository)
    engineer = ScriptedLLMClient(
        [
            _metrics_attempt(
                module="../src/routes/metrics",
                expected="{ count: 1, total: 3 }",
                with_total=True,
            )
        ]
    )
    self_review = ScriptedLLMClient([_self_review_verdict([])])
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            self_review_client=self_review,
            review_payload=review_payload(verdict="approved"),
        )

        assert execution.result.status == "approved", execution.result.blocking_issues
        completion = execution.code_completion
        assert completion is not None
        assert completion.metadata["self_review"]["outcome"] == "clean"
        assert len(self_review.calls) == 1
        # The independent Reviewer still ran and its verdict is what approved this work.
        assert execution.result.review_artifact_id is not None
        # The commit gate still ran and the publication is real: the branch is on the
        # remote with the attempt's files committed.
        assert harness.branch in harness.remote_branches()
        committed = subprocess.run(
            ("git", "show", "--name-only", "--format=", "HEAD"),
            cwd=harness.workspace,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.split()
        assert "src/routes/metrics.js" in committed
    finally:
        await harness.dispose()


async def test_a_substantive_finding_leaves_the_attempt_with_the_named_outcome(
    tmp_path: Path,
) -> None:
    """E3. A problem no bounded correction can answer ends the attempt, explained by name.

    No correction loop is entered, no second review pass runs, nothing reaches the reviewer
    and nothing is published. The findings travel as the attempt's blocking issues so the
    retry that follows can see what its predecessor's own review concluded -- and the
    failure is classified as implementation work, not as a commit-gate stop, so it has no
    claim on the pre-commit allowance and no exemption from the repeat rules.
    """
    harness = await build_harness(tmp_path, node_npm_repository)
    engineer = ScriptedLLMClient(
        [
            _metrics_attempt(
                module="../src/routes/metrics",
                expected="{ count: 1, total: 3 }",
                with_total=True,
            )
        ]
    )
    self_review = ScriptedLLMClient(
        [
            _self_review_verdict(
                [
                    {
                        "description": (
                            "The contract requires the metrics payload to be served per "
                            "tenant; this route is a single shared payload and the storage "
                            "model has no tenant dimension to add one to."
                        ),
                        "classification": "substantive",
                        "paths": ["src/routes/metrics.js"],
                        "requirement_id": "backend-status-api",
                    }
                ]
            )
        ]
    )
    try:
        execution = await run_child(
            harness,
            engineer_payload={},
            engineer_client=engineer,
            self_review_client=self_review,
        )

        assert execution.result.status == "failed"
        # Implementation work, not a deterministic gate: no repeat-rule exemption and no
        # pre-commit allowance.
        assert (
            execution.result.failure_classification
            == FailureClassification.IMPLEMENTATION_MISSING.value
        )
        assert execution.result.metadata.get("self_review_rejected") is True
        assert not execution.result.metadata.get("deterministic_gate")
        assert not execution.result.metadata.get("source_validation_rejected")
        # The escape is named, and the finding travels as the attempt's blocking issues.
        assert any(
            issue.startswith(SELF_REVIEW_DIAGNOSTIC_PREFIX)
            for issue in execution.result.blocking_issues
        ), execution.result.blocking_issues
        # One review pass, one coding call: no correction was attempted and no second
        # review ran.
        assert len(self_review.calls) == 1
        assert len(engineer.calls) == 1
        completion = execution.code_completion
        assert completion is not None
        assert completion.completion_status == "failed"
        assert completion.metadata["terminal_outcome"] == SELF_REVIEW_SUBSTANTIVE_OUTCOME
        assert completion.metadata["self_review"]["outcome"] == "substantive_problem"
        assert completion.metadata["source_validation_rejected"] is False
        # Nothing was published, and the workspace keeps the attempt's files so the retry
        # that follows can edit them in place rather than regenerate them.
        assert harness.branch not in harness.remote_branches()
        assert harness.worktree_file("src/routes/metrics.js") is not None
    finally:
        await harness.dispose()
