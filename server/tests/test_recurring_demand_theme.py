"""A same-theme demand asks its question by round three (77-, item 35).

AB-Feature-208's backend died at r8 after attempts 2-7 all rejected on one demand, reworded
every round: `fingerprint_for_text` matched none of the wordings, the resolved-issue ledger
saw only novel demands, and the design-conflict question never fired -- a human was never
asked. The theme key tested here groups a demand by the anchors a reviewer keeps stable
while it rewords its prose (requirement + file), decides only *when to ask*, is derived
fresh from the lineage every cycle (49-: derived, never stored), and never participates in
verdict binding (73-: nothing looser than the exact fingerprint).

Fixtures are the REAL payloads, verbatim, per the standing rule: 208's four wordings from
`007_review.admanager_console-2.0.attempt-3..6.json` (feature-4e7e34ae) MUST group; 207's
distinct per-attempt findings (feature-29f561ce) MUST NOT. The path and requirement strings
inside them are test data quoted from preserved artifacts, not knowledge of any repository.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from agents.shared.contracts import create_artifact
from artifacts.schemas import (
    ChildWorkflowResultArtifact,
    DesignConflictArtifact,
    ReviewArtifact,
    ReviewFinding,
)
from services.design_conflict import open_design_conflicts, recurring_demand_payload
from services.feature_runtime import _resolve_review_outcome
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from tests.test_bounded_review_scope import _finding, _review
from tests.test_overruled_review_demand import _settled
from tests.test_resolved_issue_ledger import _backend_triage, _child_result, _run
from tools.resolved_issue_ledger import (
    demand_theme,
    recurring_demand_narrative,
    recurring_demand_theme,
)
from tools.review_fix_classification import fingerprint_for_text
from workflows.feature_workflow import ChildExecution, MockChildWorkstreamExecutor

CLEANUP_DEMAND_ATTEMPT_3 = (
    "The default bulkCreateAppsV2 path tracks GAM-affected rows but neither its GAM-failure "
    "path nor its post-GAM persistence-failure path calls an adapter cleanup capability. The "
    "failure builders default to cleanupStatus: unsupported whenever rows exist, so the "
    "response cannot honestly distinguish supported successful or failed cleanup from an "
    "unavailable operation."
)
CLEANUP_DEMAND_ATTEMPT_4 = (
    "The default bulkCreateAppsV2 path tracks GAM-affected rows but does not call an adapter "
    "cleanup capability in either its GAM-failure or post-GAM persistence-failure paths. "
    "Failure builders therefore default nonempty affected rows to cleanupStatus unsupported, "
    "which cannot honestly distinguish unavailable cleanup from successful or failed "
    "supported cleanup."
)
CLEANUP_DEMAND_ATTEMPT_5 = (
    "The default bulkCreateAppsV2 flow records GAM-affected rows but never invokes an adapter "
    "cleanup capability after a GAM failure or a later local persistence failure. Its failure "
    "constructors therefore report unsupported by default whenever affected rows exist, "
    "rather than an assessed cleanup outcome."
)
CLEANUP_DEMAND_ATTEMPT_6 = (
    "The default V2 path records GAM-affected rows but does not invoke an adapter cleanup "
    "capability after GAM failure or local persistence failure. Failure builders therefore "
    "report unsupported by default whenever effects exist rather than reporting an assessed "
    "cleanup outcome."
)
CLEANUP_WORDINGS = (
    CLEANUP_DEMAND_ATTEMPT_3,
    CLEANUP_DEMAND_ATTEMPT_4,
    CLEANUP_DEMAND_ATTEMPT_5,
    CLEANUP_DEMAND_ATTEMPT_6,
)
CLEANUP_REQUIREMENT = "FR-007"
CLEANUP_FILE = "server/services/app/app.service.js"

# 207's attempt-2 review: four findings, four different defects. They MUST NOT group.
DISTINCT_207_FINDINGS = (
    (
        "FR-005",
        "server/validation/bulkCreateApps.validation.js",
        "parseAndValidateBulkCsv computes countedRecords by filtering isExcludedRecord before "
        "the loop that emits CSV_RECORD_STRUCTURE_INVALID. A delimiter-only row whose all "
        "fields are empty but whose delimiter count produces the wrong number of columns is "
        "excluded and never reported.",
    ),
    (
        "FR-009",
        "server/services/app/app.service.js",
        "The bulk parser accepts any non-empty store because it delegates to the existing Joi "
        "schema, but getAppStore supports only the three known GAM store values and returns "
        "null for every other string.",
    ),
    (
        "FR-007",
        "server/gamUtils/app/create_apps.py",
        "The Python adapter accesses each GAM result with mapping indexing for displayName "
        "and id but uses getattr for applicationCode. For mapping results, getattr does not "
        "retrieve the applicationCode key, so the adapter returns applicationCode null.",
    ),
    (
        "NFR-004",
        "server/services/app/app.service.js",
        "After a complete GAM application batch succeeds, createBulkAppsInGAM invokes "
        "orderService.createAndApproveOrder once per app before persistBulkApps opens its "
        "Mongoose session, leaving already-created order side effects outside the abort path.",
    ),
)

# Item 30's trap, verbatim shape: an infrastructure sentence that once entered the ledger as
# a review demand (AB-Feature-204). A theme built from it would raise a conflict about a
# measurement, which 73- correctly forbids overruling.
INSTALL_GATE_SENTENCE = "Deterministic dependency installation failed before coding began."

_CHILD = "feature-theme:backend"


def _themed_finding(
    finding_id: str, description: str, *, requirement_id: str, file_path: str
) -> ReviewFinding:
    """One blocking review-judgement finding, shaped as the reviewer produces it."""
    return ReviewFinding(
        finding_id=finding_id,
        severity="high",
        title=f"Blocking demand {finding_id}",
        description=description,
        recommendation="Address the demand as worded.",
        file_path=file_path,
        line_number=None,
        repository_id="backend",
        requirement_id=requirement_id,
        responsibility="implements",
        validated_revision="rev-under-review",
        evidence="Quoted from the change under review.",
        finding_category="requirement",
    )


def _reviewed_cycle(
    attempt: int,
    findings: Sequence[ReviewFinding],
    *,
    deterministic_ids: Sequence[str] = (),
    deterministic_gate: bool = False,
) -> tuple[ReviewArtifact, ChildWorkflowResultArtifact]:
    """One review verdict and the child result it failed, linked the way the loop links them."""
    review = create_artifact(
        ReviewArtifact,
        workflow_id=_CHILD,
        artifact_id=f"artifact-review-backend-attempt-{attempt}",
        producer="reviewer",
        payload={
            "verdict": "changes_requested",
            "summary": f"Attempt {attempt} declined.",
            "requirement_checks": [],
            "findings": [item.model_dump(mode="json") for item in findings],
            "architecture_assessment": "Assessed.",
            "security_assessment": "Assessed.",
            "test_coverage_assessment": "Assessed.",
        },
        metadata={"deterministic_finding_ids": list(deterministic_ids)},
    )
    result = create_artifact(
        ChildWorkflowResultArtifact,
        workflow_id="feature-theme",
        artifact_id=f"artifact-result-backend-attempt-{attempt}",
        producer="child_workflow",
        payload={
            "feature_id": "feature-theme",
            "parent_workflow_id": "feature-theme",
            "child_workflow_id": _CHILD,
            "repository_id": "backend",
            "workstream_id": "workstream-backend",
            "branch_name": "feature/backend",
            "workspace_path": "/tmp/backend",
            "code_completion_artifact_id": None,
            "review_artifact_id": review.artifact_id,
            "changed_files": [],
            "validation_results": [],
            "status": "failed",
            "blocking_issues": [item.description for item in findings],
            "pull_request_readiness": False,
            "contract_sections_consumed": [],
            "contract_sections_implemented": [],
        },
        metadata={
            "child_retry_count": attempt,
            **({"deterministic_gate": True} if deterministic_gate else {}),
        },
    )
    return review, result


def _cleanup_cycles(count: int) -> list[tuple[ReviewArtifact, ChildWorkflowResultArtifact]]:
    """The 208 lineage: the same demand, reworded each round, one review cycle per wording."""
    return [
        _reviewed_cycle(
            attempt,
            [
                _themed_finding(
                    "bulk-gam-cleanup-never-attempted",
                    wording,
                    requirement_id=CLEANUP_REQUIREMENT,
                    file_path=CLEANUP_FILE,
                )
            ],
        )
        for attempt, wording in enumerate(CLEANUP_WORDINGS[:count])
    ]


def test_the_four_real_wordings_share_a_theme_and_no_exact_key() -> None:
    """The 208 premise, asserted from the preserved payloads: one theme, four fingerprints."""
    findings = [
        _themed_finding(
            "bulk-gam-cleanup-never-attempted",
            wording,
            requirement_id=CLEANUP_REQUIREMENT,
            file_path=CLEANUP_FILE,
        )
        for wording in CLEANUP_WORDINGS
    ]
    themes = {demand_theme(item) for item in findings}
    fingerprints = {fingerprint_for_text(item.description) for item in findings}

    assert len(themes) == 1
    # This is exactly why the ledger never fired: every wording is a novel exact key.
    assert len(fingerprints) == len(CLEANUP_WORDINGS)


def test_207s_distinct_findings_carry_distinct_themes_and_raise_nothing() -> None:
    """The MUST-NOT fixture: distinct defects stay distinct, and one cycle asks nothing."""
    findings = [
        _themed_finding(f"defect-{index}", text, requirement_id=req, file_path=path)
        for index, (req, path, text) in enumerate(DISTINCT_207_FINDINGS)
    ]
    assert len({demand_theme(item) for item in findings}) == len(findings)

    review, result = _reviewed_cycle(2, findings)
    assert (
        recurring_demand_theme(
            [],
            child_workflow_id=_CHILD,
            current_review=review,
            current_result=result,
            current_attempt=2,
        )
        is None
    )


def test_a_demand_blocking_its_third_cycle_raises_the_theme_with_every_wording() -> None:
    """Three same-theme blocking rejections ask; two do not. 208 burned six before r8."""
    cycles = _cleanup_cycles(3)
    lineage = [artifact for pair in cycles[:2] for artifact in pair]
    current_review, current_result = cycles[2]

    early = recurring_demand_theme(
        list(cycles[0]),
        child_workflow_id=_CHILD,
        current_review=current_review,
        current_result=current_result,
        current_attempt=2,
    )
    assert early is None  # two cycles: one prior plus the current one

    theme = recurring_demand_theme(
        lineage,
        child_workflow_id=_CHILD,
        current_review=current_review,
        current_result=current_result,
        current_attempt=2,
    )
    assert theme is not None
    assert theme.issue == CLEANUP_DEMAND_ATTEMPT_5
    assert [item.issue for item in theme.wordings] == list(CLEANUP_WORDINGS[:3])
    assert [item.attempt for item in theme.wordings] == [0, 1, 2]

    question, evidence = recurring_demand_narrative(theme)
    assert question.endswith("?")
    for wording in CLEANUP_WORDINGS[:3]:
        assert any(wording in line for line in evidence)
    # Not the vocabulary of the stops this one was split out of: nobody is sent to repair a
    # repository, buy a stronger model, or read a reversal that never happened.
    assert "already made and then removed" not in question
    assert "capability limit" not in question
    assert "what has to be fixed" not in question


def test_infrastructure_and_deterministic_sentences_never_enter_a_theme() -> None:
    """Item 30's trap, fenced three ways: preflight, deterministic gate, injected findings."""
    # Five preflight-blocked cycles carrying the identical install sentence: no review
    # artifact exists, so they contribute nothing however often the sentence repeats.
    preflight_results = [
        _child_result("backend", _CHILD, attempt, [INSTALL_GATE_SENTENCE]) for attempt in range(5)
    ]
    review, result = _reviewed_cycle(
        5,
        [
            _themed_finding(
                "real-demand",
                CLEANUP_DEMAND_ATTEMPT_3,
                requirement_id=CLEANUP_REQUIREMENT,
                file_path=CLEANUP_FILE,
            )
        ],
    )
    assert (
        recurring_demand_theme(
            list(preflight_results),
            child_workflow_id=_CHILD,
            current_review=review,
            current_result=result,
            current_attempt=5,
        )
        is None
    )

    # Three reviewed cycles whose only blocking finding is platform-injected -- named by the
    # review's own deterministic census: no theme, however exactly it repeats.
    injected_cycles = [
        _reviewed_cycle(
            attempt,
            [
                _themed_finding(
                    "VALIDATION-TEST-FAILED",
                    "The required test command failed against this revision.",
                    requirement_id=CLEANUP_REQUIREMENT,
                    file_path=CLEANUP_FILE,
                )
            ],
            deterministic_ids=["VALIDATION-TEST-FAILED"],
        )
        for attempt in range(3)
    ]
    lineage = [artifact for pair in injected_cycles[:2] for artifact in pair]
    current_review, current_result = injected_cycles[2]
    assert (
        recurring_demand_theme(
            lineage,
            child_workflow_id=_CHILD,
            current_review=current_review,
            current_result=current_result,
            current_attempt=2,
        )
        is None
    )

    # A deterministic-gate attempt contributes no cycle even with a review attached.
    gate_cycles = [
        _reviewed_cycle(
            attempt,
            [
                _themed_finding(
                    "bulk-gam-cleanup-never-attempted",
                    wording,
                    requirement_id=CLEANUP_REQUIREMENT,
                    file_path=CLEANUP_FILE,
                )
            ],
            deterministic_gate=attempt < 2,
        )
        for attempt, wording in enumerate(CLEANUP_WORDINGS[:3])
    ]
    lineage = [artifact for pair in gate_cycles[:2] for artifact in pair]
    current_review, current_result = gate_cycles[2]
    assert (
        recurring_demand_theme(
            lineage,
            child_workflow_id=_CHILD,
            current_review=current_review,
            current_result=current_result,
            current_attempt=2,
        )
        is None
    )


def test_alternating_themes_still_count_their_third_occurrence() -> None:
    """The 49-C alternation lesson re-asserted here: distinct cycles, not consecutive ones."""
    other = _themed_finding(
        "other-demand",
        "The response omits the pagination cursor on the last page of results.",
        requirement_id="FR-002",
        file_path="src/api/pagination.js",
    )
    script: list[tuple[ReviewArtifact, ChildWorkflowResultArtifact]] = []
    for attempt in range(5):
        finding = (
            _themed_finding(
                "bulk-gam-cleanup-never-attempted",
                CLEANUP_WORDINGS[attempt // 2],
                requirement_id=CLEANUP_REQUIREMENT,
                file_path=CLEANUP_FILE,
            )
            if attempt % 2 == 0
            else other
        )
        script.append(_reviewed_cycle(attempt, [finding]))
    lineage = [artifact for pair in script[:4] for artifact in pair]
    current_review, current_result = script[4]

    theme = recurring_demand_theme(
        lineage,
        child_workflow_id=_CHILD,
        current_review=current_review,
        current_result=current_result,
        current_attempt=4,
    )
    assert theme is not None
    assert [item.attempt for item in theme.wordings] == [0, 2, 4]


def test_a_settled_wording_suppresses_the_ask_by_its_exact_key_only() -> None:
    """An answered theme is not re-asked -- excluded by a real wording's exact fingerprint."""
    cycles = _cleanup_cycles(4)
    lineage = [artifact for pair in cycles[:3] for artifact in pair]
    current_review, current_result = cycles[3]

    settled_first = recurring_demand_theme(
        lineage,
        child_workflow_id=_CHILD,
        current_review=current_review,
        current_result=current_result,
        current_attempt=3,
        settled={fingerprint_for_text(CLEANUP_DEMAND_ATTEMPT_3)},
    )
    assert settled_first is None

    unrelated_settled = recurring_demand_theme(
        lineage,
        child_workflow_id=_CHILD,
        current_review=current_review,
        current_result=current_result,
        current_attempt=3,
        settled={fingerprint_for_text("Some other decided question entirely.")},
    )
    assert unrelated_settled is not None


def test_the_conflict_artifact_binds_the_earliest_wording_and_stores_no_theme() -> None:
    """49-: derived, never stored. 73-: the binding key is an exact wording's fingerprint."""
    assert "theme" not in DesignConflictArtifact.model_fields
    cycles = _cleanup_cycles(3)
    lineage = [artifact for pair in cycles[:2] for artifact in pair]
    current_review, current_result = cycles[2]
    theme = recurring_demand_theme(
        lineage,
        child_workflow_id=_CHILD,
        current_review=current_review,
        current_result=current_result,
        current_attempt=2,
    )
    assert theme is not None
    question, evidence = recurring_demand_narrative(theme)

    payload = recurring_demand_payload(
        theme,
        feature_id="feature-theme",
        repository_id="backend",
        child_workflow_id=_CHILD,
        question=question,
        evidence=evidence,
        attempts_spent=3,
    )
    conflict = create_artifact(
        DesignConflictArtifact,
        workflow_id="feature-theme",
        artifact_id="017_design_conflict.test.json",
        producer="feature_workflow",
        payload=payload,
        metadata={"repository_id": "backend", "source": "recurring_demand_theme"},
    )
    assert conflict.kind == "recurring_demand"
    assert conflict.fingerprint == fingerprint_for_text(CLEANUP_DEMAND_ATTEMPT_3)
    assert conflict.demand == CLEANUP_DEMAND_ATTEMPT_5
    assert conflict.satisfied_under is None
    assert conflict.satisfied_demand is None
    assert conflict.removals == 0
    # A later reworded round derives the same identity, so the idempotent record finds its
    # question already on file instead of putting a second form in front of the same person.
    assert conflict.conflict_id.endswith(fingerprint_for_text(CLEANUP_DEMAND_ATTEMPT_3))


def test_each_conflict_kind_refuses_the_other_kinds_positions() -> None:
    """The pairing validator: a reversal proves satisfaction, a recurrence has none."""
    import pytest

    base = {
        "conflict_id": "conflict-backend-test",
        "feature_id": "feature-theme",
        "repository_id": "backend",
        "child_workflow_id": _CHILD,
        "fingerprint": fingerprint_for_text(CLEANUP_DEMAND_ATTEMPT_3),
        "question": "Which position holds?",
        "demanded_by": "repository_review",
        "demand": CLEANUP_DEMAND_ATTEMPT_5,
        "status": "open",
    }

    def build(payload: dict[str, object]) -> DesignConflictArtifact:
        return create_artifact(
            DesignConflictArtifact,
            workflow_id="feature-theme",
            artifact_id="017_design_conflict.kinds.json",
            producer="feature_workflow",
            payload=payload,
            metadata={},
        )

    with pytest.raises(Exception, match="satisfied"):
        build({**base, "kind": "reversal"})
    with pytest.raises(Exception, match="satisfied"):
        build(
            {
                **base,
                "kind": "recurring_demand",
                "removals": 0,
                "satisfied_under": "repository_review",
                "satisfied_demand": CLEANUP_DEMAND_ATTEMPT_3,
            }
        )
    with pytest.raises(Exception, match="removals"):
        build({**base, "kind": "recurring_demand", "removals": 2})


class RewordingReviewExecutor:
    """Reject the backend with one structured demand, reworded every attempt -- 208's shape.

    Every attempt reports genuinely different production source and a genuinely different
    wording, so no exact-text guard, no diff fingerprint, and no ledger recurrence can stop
    the loop -- exactly the lineage that burned to r8 live. The review artifact travels with
    the result the way the live executor's does, so the theme derivation reads real linked
    artifacts rather than a shortcut.
    """

    def __init__(self, wordings: Sequence[str]) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self._wordings = list(wordings)
        self.attempts = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Return the next reworded rejection for the backend, approving every sibling."""
        execution = await self._delegate.run(**kwargs)
        if kwargs["repository"].repository_id != "backend":
            return execution
        self.attempts += 1
        wording = self._wordings[min(self.attempts, len(self._wordings)) - 1]
        finding = _themed_finding(
            "bulk-gam-cleanup-never-attempted",
            wording,
            requirement_id=CLEANUP_REQUIREMENT,
            file_path=CLEANUP_FILE,
        )
        assert execution.review is not None
        review = execution.review.model_copy(
            update={
                "verdict": "changes_requested",
                "findings": [finding],
                "metadata": {**execution.review.metadata, "deterministic_finding_ids": []},
            }
        )
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    "blocking_issues": [wording],
                    "pull_request_readiness": False,
                    "production_files_changed": [f"src/attempt{self.attempts}.service.js"],
                    "failure_classification": "validation_source_failure",
                    "production_diff_fingerprint": f"production-{self.attempts}",
                    "test_diff_fingerprint": f"tests-{self.attempts}",
                }
            ),
            code_completion=execution.code_completion,
            review=review,
        )


@pytest.mark.asyncio
async def test_the_reworded_demand_asks_its_question_by_round_three_in_the_loop() -> None:
    """208's backend lineage replayed: the stop that never fired live fires at round three.

    Live, attempts 2-7 burned to the identical-resubmission stop at r8 without a human ever
    being asked. With the theme check the loop stops on the third same-theme rejection as a
    design conflict, quotes every wording, and files the answerable question.
    """
    executor = RewordingReviewExecutor(CLEANUP_WORDINGS)

    result = await _run("feature-theme-loop", executor)

    assert executor.attempts == 3
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.FAILED
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    triage = _backend_triage(result)
    assert triage["terminal_cause"] == "design_conflict"
    question = triage["operator_question"]
    assert question.endswith("requiring it?")
    for wording in CLEANUP_WORDINGS[:3]:
        assert any(wording in line for line in triage["terminal_evidence"])
    # The stop is answerable: the question is on file as an open conflict of the new kind,
    # bound by the earliest wording's exact fingerprint.
    conflicts = open_design_conflicts(result.artifacts, repository_id="backend")
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.kind == "recurring_demand"
    assert conflict.fingerprint == fingerprint_for_text(CLEANUP_DEMAND_ATTEMPT_3)
    assert conflict.demand == CLEANUP_DEMAND_ATTEMPT_5
    assert conflict.question == question
    # And the workstream's own account of why it stopped describes THIS stop. It used to
    # borrow the reversal stop's sentence -- "a change this workstream already made and a
    # later attempt removed" -- which is a story the evidence contradicts: nothing was made
    # and taken out, the demand was simply never met. That field is what an escalation and the
    # assistant's context read as the cause, so AB-Feature-225's operator was told the wrong
    # thing about the only stop the run produced.
    refusal = result.child_workflows["backend"].retry_refusal_reason or ""
    assert "demanded the same thing in every round" in refusal
    assert "already made" not in refusal


def test_the_verdict_binds_byte_identical_wordings_and_nothing_looser() -> None:
    """73-'s nothing-looser rule, asserted byte-identically across the theme boundary.

    A `removal_holds` verdict bound to one wording demotes a finding whose description is
    byte-identical to it -- and does NOT demote the same demand reworded, same theme or not.
    The theme's only post-verdict effect is that the question is not asked twice.
    """
    bound = _finding("SEAM-1").model_copy(update={"description": CLEANUP_DEMAND_ATTEMPT_3})
    reworded = _finding("SEAM-2").model_copy(update={"description": CLEANUP_DEMAND_ATTEMPT_4})
    settled = [_settled(demand=CLEANUP_DEMAND_ATTEMPT_3)]

    demoted = _resolve_review_outcome(
        _review(findings=[bound]), bounded=False, publishable=True, settled=settled
    )
    assert demoted.blocking_issues == []
    assert [item.finding_id for item in demoted.overruled_findings] == ["SEAM-1"]

    kept = _resolve_review_outcome(
        _review(findings=[reworded]), bounded=False, publishable=True, settled=settled
    )
    assert kept.blocking_issues == [CLEANUP_DEMAND_ATTEMPT_4]
    assert kept.overruled_findings == []
