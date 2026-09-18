"""Where each finished attempt ended, and the four ways an assembler gets that wrong.

AB-Feature-201's backend attempt 0 is the case: 57 minutes, every journaled row succeeded,
stopped by the implementation self-review before the reviewer was called, and drawn as a wall
of green ticks. The records that answer it already existed; nothing read them.

Four of these tests exist because the shape they pin would otherwise ship broken while
everything else passed:

* the ✗ has to land on `review` from a self-review ending with *no review artifact at all*
  (A1) -- the case that has no operation row to hang anything on;
* an `approved` attempt that still carries a `failure_classification` must take no ✗
  anywhere (A1c) -- 201 has four of them;
* the ending is assembled once per finished attempt and never for the in-flight one (A11),
  which is the cost invariant the whole design rests on;
* a failed command under an environment classification is a `fault`, not a
  `validation_failed` (A10) -- the 20- split, decided by the record and not by the exit code.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import HttpUrl

from artifacts.schemas import (
    ChildWorkflowResultArtifact,
    CodeCompletionArtifact,
    ReviewArtifact,
)
from services.attempt_endings import AttemptEndingCache
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.feature_models import (
    ChildWorkflowReference,
    FeatureWorkflowSnapshot,
    RepositorySpec,
)
from workflow_schema import WORKFLOW_SCHEMA_VERSION

_AT = datetime(2026, 9, 3, 6, 27, 2, tzinfo=UTC)
_REPOSITORY = "admanager_console-2.0"


def _envelope(artifact_id: str, producer: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "workflow_id": "wf-201",
        "artifact_id": artifact_id,
        "producer": producer,
        "timestamp": _AT,
        "metadata": metadata,
        "validation_status": "valid",
    }


def _completion(attempt: int, **metadata: Any) -> CodeCompletionArtifact:
    return CodeCompletionArtifact.model_validate(
        {
            **_envelope(
                f"006_code_completion.{_REPOSITORY}.attempt-{attempt}.json",
                "engineer",
                {"child_attempt": attempt, **metadata},
            ),
            "completion_status": "completed",
            "summary": f"Attempt {attempt}.",
            "file_changes": [],
            "validation_results": [],
            "test_coverage_percent": None,
            "remaining_work": [],
            "commit_sha": None,
        }
    )


def _review(attempt: int, *, verdict: str, findings: list[dict[str, Any]]) -> ReviewArtifact:
    return ReviewArtifact.model_validate(
        {
            **_envelope(
                f"007_review.{_REPOSITORY}.attempt-{attempt}.json",
                "reviewer",
                {"child_attempt": attempt},
            ),
            "verdict": verdict,
            "summary": "A review.",
            "requirement_checks": [],
            "findings": findings,
            "architecture_assessment": "Fine.",
            "security_assessment": "Fine.",
            "test_coverage_assessment": "Fine.",
        }
    )


def _finding(finding_id: str, severity: str = "high") -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "severity": severity,
        "title": "The bulk orchestrator is still exported.",
        "description": "The superseded export is still reachable.",
        "recommendation": "Remove it.",
        "file_path": None,
        "line_number": None,
        "repository_id": _REPOSITORY,
    }


def _result(
    attempt: int,
    *,
    status: str = "failed",
    reviewed: bool = True,
    workspace: str | None = "preserved",
    classification: str | None = None,
    validation: list[dict[str, Any]] | None = None,
    blocking: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ChildWorkflowResultArtifact:
    inputs = {"workspace": workspace} if workspace is not None else None
    return ChildWorkflowResultArtifact.model_validate(
        {
            **_envelope(
                f"011_child_workflow_result.{_REPOSITORY}.attempt-{attempt}.json",
                "child_workflow",
                {
                    "child_retry_count": attempt,
                    **({"attempt_inputs": inputs} if inputs else {}),
                    **(metadata or {}),
                },
            ),
            "feature_id": "f-201",
            "parent_workflow_id": "wf-201",
            "child_workflow_id": f"f-201:{_REPOSITORY}",
            "repository_id": _REPOSITORY,
            "workstream_id": "ws-admanager-api",
            "branch_name": "ai/f-201/admanager",
            "workspace_path": f"/w/{_REPOSITORY}",
            "code_completion_artifact_id": (
                f"006_code_completion.{_REPOSITORY}.attempt-{attempt}.json"
            ),
            "review_artifact_id": (
                f"007_review.{_REPOSITORY}.attempt-{attempt}.json" if reviewed else None
            ),
            "changed_files": [],
            "validation_results": [],
            "status": status,
            "blocking_issues": blocking or [],
            "pull_request_readiness": status == "approved",
            "contract_sections_consumed": [],
            "contract_sections_implemented": [],
            "failure_classification": classification,
            "current_validation_results": validation or [],
        }
    )


def _child(**overrides: Any) -> ChildWorkflowReference:
    return ChildWorkflowReference.model_validate(
        {
            "child_workflow_id": f"f-201:{_REPOSITORY}",
            "repository_id": _REPOSITORY,
            "workstream_id": "ws-admanager-api",
            "status": ChildWorkflowStatus.RUNNING,
            "branch_name": "ai/f-201/admanager",
            "workspace_path": f"/w/{_REPOSITORY}",
            "retry_count": 0,
            **overrides,
        }
    )


def _feature(artifacts: list[Any], child: ChildWorkflowReference) -> FeatureWorkflowSnapshot:
    return FeatureWorkflowSnapshot.model_validate(
        {
            "feature_id": "f-201",
            "workflow_id": "wf-201",
            "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
            "created_by_build_revision": "rev",
            "last_executor_build_revision": "rev",
            "status": FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
            "title": "Allow an admin to bulk add apps",
            "repository_specs": [
                RepositorySpec(
                    repository_id=_REPOSITORY,
                    name=_REPOSITORY,
                    role="backend",
                    repository_url=HttpUrl("https://github.com/cryn3t/admanager_console-2.0"),
                    default_branch="main",
                )
            ],
            "artifacts": artifacts,
            "child_workflows": {_REPOSITORY: child},
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


def _endings(artifacts: list[Any], child: ChildWorkflowReference) -> dict[int, Any]:
    cache = AttemptEndingCache()
    served = cache.endings_for(_feature(artifacts, child), _REPOSITORY)
    return {item.attempt: item for item in served}


def test_a1_a_self_review_ending_lands_on_review_with_no_review_artifact_anywhere() -> None:
    """A1: 201 a0 replayed -- `corrections_failed`, one finding, and no 007 for the attempt.

    The whole difficulty is that there is nothing to read the ending off. No `run_reviewer`
    row ran, no review artifact exists, and every record the attempt *did* write says
    `succeeded`. The ending comes from the completion's own self-review record, and the stage
    it names is `review` -- which is what puts a ✗ where a person is looking.
    """
    artifacts = [
        _completion(
            0,
            self_review={
                "ran": True,
                "outcome": "corrections_failed",
                "corrections_applied": [],
                "findings": [{"requirement_id": "FR-009", "classification": "localized"}],
            },
            source_repair_passes=0,
        ),
        _result(
            0,
            status="failed",
            reviewed=False,
            workspace="fresh_checkout",
            classification="implementation_missing",
            blocking=["The implementation self-review found (localized): the error codes are…"],
        ),
    ]
    ending = _endings(artifacts, _child(retry_count=1))[0]

    assert ending.ended_by == "self_review"
    assert ending.stage == "review"
    # The finding is named by what the self-review record actually carries: a requirement and
    # a classification, not a finding id, which self-review findings do not have.
    assert ending.detail == "self-review corrections failed against 1 finding (FR-009, localized)"
    assert ending.workspace == "fresh_checkout"
    assert ending.self_review_outcome == "corrections_failed"
    assert ending.source_repair_passes == 0


def test_a1b_a_review_rejection_names_its_verdict_and_its_blocking_finding() -> None:
    """A1b: the reviewer answered, and the answer was not an approval."""
    artifacts = [
        _completion(1, self_review={"ran": True, "outcome": "clean", "corrections_applied": []}),
        _review(
            1,
            verdict="changes_requested",
            findings=[
                _finding("bulk-partial-error-reporting", severity="low"),
                _finding("superseded-bulk-orchestrator-still-exported"),
            ],
        ),
        _result(1, status="failed", classification="review_scope_failure"),
    ]
    ending = _endings(artifacts, _child(retry_count=2))[1]

    assert ending.ended_by == "review_rejected"
    assert ending.stage == "review"
    # The severity ordering matters: the low-severity remark recorded for a person is not why
    # the attempt was sent back.
    assert ending.detail == (
        "review returned changes requested with 2 findings, "
        "blocking on superseded-bulk-orchestrator-still-exported"
    )


def test_a1c_an_approved_attempt_that_still_carries_a_classification_is_not_a_failure() -> None:
    """A1c: 201's backend attempts 2, 6, 7 and 8, and the trap they exposed.

    Each is `approved` and each still holds a `failure_classification`. An assembler that
    read the classification first reported four delivered attempts as failures -- which is
    exactly what the first draft of this design did.
    """
    approved = [
        _completion(2, self_review={"ran": True, "outcome": "clean", "corrections_applied": []}),
        _review(2, verdict="approved", findings=[_finding("advisory-note", severity="low")]),
        _result(2, status="approved", classification="review_scope_failure"),
    ]
    superseded = _endings([*approved, _result(3, status="failed")], _child(retry_count=4))[2]
    final = _endings(approved, _child(retry_count=2, status=ChildWorkflowStatus.APPROVED))[2]

    # Reopened by a later attempt: `superseded`, and the publication stage owns the wording.
    assert superseded.ended_by == "superseded"
    assert superseded.stage == "publication"
    assert "review_scope_failure" not in (superseded.detail or "")
    # The last attempt of a settled child: `approved`, never the classification it holds.
    assert final.ended_by == "approved"
    assert final.detail == "approved by review"


def test_a1c_an_approved_attempt_that_published_says_so() -> None:
    """A1c/A10: `published` is the completion's own recorded fact, not an inference."""
    artifacts = [
        _completion(2, published_after_approval=True),
        _review(2, verdict="approved", findings=[]),
        _result(2, status="approved"),
    ]
    ending = _endings(artifacts, _child(retry_count=2, status=ChildWorkflowStatus.APPROVED))[2]

    assert ending.ended_by == "published"
    assert ending.detail == "approved by review and published to the branch"


def test_a2_a_validation_ending_quotes_the_command_that_failed() -> None:
    """A2: the failed required command, from the revision-bound set the policy read."""
    artifacts = [
        _completion(5),
        _result(
            5,
            status="failed",
            reviewed=False,
            classification="validation_source_failure",
            validation=[
                {"validation_type": "lint", "command": "npm run lint", "passed": True},
                {
                    "validation_type": "test",
                    "command": "npm run test -- server/tests/bulk-create-apps.test.js",
                    "passed": False,
                    "exit_code": 1,
                },
            ],
        ),
    ]
    ending = _endings(artifacts, _child(retry_count=6))[5]

    assert ending.ended_by == "validation_failed"
    assert ending.stage == "validation"
    assert ending.detail == (
        "the required test command failed: "
        "npm run test -- server/tests/bulk-create-apps.test.js (exit 1)"
    )


def test_a2_a_validation_ending_with_nothing_journaled_still_names_its_class() -> None:
    """A2: 201's backend attempt 4 -- the gate that rejected it journals nothing.

    Its `current_validation_results` is empty because the Engineer's own in-attempt source
    validation is what refused the attempt. The classification is still the record, so the
    ending is still `validation_failed` and the detail says what the platform called it
    rather than inventing a command nobody ran.
    """
    artifacts = [
        _completion(4, source_validation_rejected=True, source_repair_passes=0),
        _result(4, status="failed", reviewed=False, classification="validation_source_failure"),
    ]
    ending = _endings(artifacts, _child(retry_count=5))[4]

    assert ending.ended_by == "validation_failed"
    assert ending.detail == "classified as validation source failure"
    # No self-review key at all -- the seventh state the code does not enumerate. Absent, not
    # clean: a cell that renders nothing here would read as a clean pass.
    assert ending.self_review_outcome is None


def test_a3_a_refusal_ending_quotes_one_sentence_of_the_recorded_reason() -> None:
    """A3: the refusal outranks whatever the work would otherwise have been judged on."""
    reason = (
        "The Engineer refused the attempt: two required context files were dropped. "
        "Nothing was spent on the call."
    )
    artifacts = [
        _completion(3),
        _result(3, status="failed", reviewed=False, metadata={"retry_refusal_reason": reason}),
    ]
    ending = _endings(artifacts, _child(retry_count=4))[3]

    assert ending.ended_by == "refusal"
    assert ending.stage == "coding"
    # One sentence: the whole reason is a paragraph, and this clause sits on a dropdown line.
    assert ending.detail == (
        "The Engineer refused the attempt: two required context files were dropped."
    )


def test_a10_a_failed_command_under_an_environment_class_is_a_fault_not_a_validation_failure() -> (
    None
):
    """A10: the 20- split, decided by the record's own classification and never by the exit code.

    The same red command means two different things. A repository whose toolchain cannot
    install its dependencies did not fail validation, and rendering it as though it had sends
    a reader to look at a diff that is fine.
    """
    validation = [
        {"validation_type": "test", "command": "npm run test", "passed": False, "exit_code": 1}
    ]
    faulted = _endings(
        [
            _completion(6),
            _result(
                6,
                status="failed",
                reviewed=False,
                classification="dependency_installation_failure",
                validation=validation,
            ),
        ],
        _child(retry_count=7),
    )[6]
    sourced = _endings(
        [
            _completion(6),
            _result(
                6,
                status="failed",
                reviewed=False,
                classification="validation_source_failure",
                validation=validation,
            ),
        ],
        _child(retry_count=7),
    )[6]

    assert (faulted.ended_by, faulted.stage) == ("fault", "setup")
    assert faulted.detail == "classified as dependency installation failure"
    assert (sourced.ended_by, sourced.stage) == ("validation_failed", "validation")


def test_a11_an_ending_is_assembled_once_per_finished_attempt_and_never_for_the_live_one() -> None:
    """A11: the caching contract, which is the cost invariant the whole design rests on.

    The operations endpoint is timer-polled per repository. Assembling "wherever nothing is
    cached yet" would scan artifacts on every poll for the one attempt anybody is watching,
    which is precisely the cost the cache exists to prevent. Without this test the rule is a
    comment.
    """
    artifacts = [
        _completion(0, self_review={"ran": True, "outcome": "clean", "corrections_applied": []}),
        _review(0, verdict="changes_requested", findings=[_finding("F-1")]),
        _result(0, status="failed", classification="review_scope_failure"),
        # The in-flight attempt: its records exist, and it has no ending.
        _completion(1),
        _result(1, status="failed"),
    ]
    child = _child(retry_count=1, status=ChildWorkflowStatus.RUNNING)
    state = _feature(artifacts, child)
    cache = AttemptEndingCache()

    first = cache.endings_for(state, _REPOSITORY)
    after_first = cache.assemblies
    second = cache.endings_for(state, _REPOSITORY)

    # The finished attempt, and only it.
    assert [item.attempt for item in first] == [0]
    assert [item.attempt for item in second] == [0]
    # One assembly across two polls: the finished attempt was assembled once, and the
    # in-flight attempt was never assembled at all.
    assert after_first == 1
    assert cache.assemblies == 1


def test_a11_the_final_attempt_of_a_stopped_child_does_get_an_ending() -> None:
    """A11: "moved past" includes the last attempt of a child that stopped.

    That attempt is the marker case this whole item opens with, so exempting it -- which
    "only attempts before `retry_count`" would do -- would leave the defect in place for
    every failed workstream.
    """
    artifacts = [
        _completion(0, self_review={"ran": True, "outcome": "clean", "corrections_applied": []}),
        _result(0, status="failed", reviewed=False, classification="implementation_missing"),
    ]
    endings = _endings(artifacts, _child(retry_count=0, status=ChildWorkflowStatus.FAILED))

    assert [*endings] == [0]
    assert endings[0].ended_by == "fault"
    assert endings[0].stage == "coding"


def test_a5_a_status_that_names_no_ending_is_served_as_no_ending() -> None:
    """A5: an attempt held at a human gate has no ending, and the block says nothing.

    A contract-change wait and a cancellation are real states and neither judges the attempt.
    Naming one would mean minting a vocabulary value for "we do not know", which is the lie
    this whole part exists to remove.
    """
    artifacts = [
        _completion(0),
        _result(0, status="waiting_for_contract_change", reviewed=False),
    ]
    endings = _endings(
        artifacts, _child(retry_count=0, status=ChildWorkflowStatus.WAITING_FOR_CONTRACT_CHANGE)
    )

    assert endings == {}


def test_a9_the_recorded_workspace_travels_verbatim_rather_than_as_a_boolean() -> None:
    """A9: `reset` and `fresh_checkout` are different reasons for the same absence.

    `preserved` and `recovered_coding_output` are the two that mean work was inherited, and
    only they license the drawer's "carried over" wording. Serving a boolean would collapse
    four recorded facts into one and make that distinction unavailable.
    """
    recorded = {}
    for attempt, workspace in enumerate(
        ("fresh_checkout", "preserved", "reset", "recovered_coding_output")
    ):
        artifacts = [
            _completion(attempt),
            _result(attempt, status="failed", reviewed=False, workspace=workspace),
        ]
        ending = _endings(artifacts, _child(retry_count=attempt + 1))[attempt]
        recorded[workspace] = ending.workspace

    assert recorded == {
        "fresh_checkout": "fresh_checkout",
        "preserved": "preserved",
        "reset": "reset",
        "recovered_coding_output": "recovered_coding_output",
    }
