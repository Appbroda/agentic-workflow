"""Coverage for the resolved-issue ledger and the design-conflict stop it feeds.

AB-Feature-183 and -184 both ended by spending attempts on a question no attempt could
answer. 183's backend failed the same suite four times while the wording moved underneath it;
184's frontend was rejected twice for one finding, and at the integration seam the reviewer
demanded compensating cleanup for a design this repository's own review had just approved
without. Nothing in the loop could tell "still broken" from "already fixed, and now demanded
again", because every guard it had compares an attempt to the one immediately before it.

These tests assert effects rather than calls: what the next engineer was actually told, and
what a person reading the stopped workstream is actually sent to decide.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest

from agents.shared.contracts import FEATURE_ARTIFACT_FILENAMES, create_artifact
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import (
    ChildWorkflowResultArtifact,
    IntegrationContractArtifact,
    IntegrationReviewArtifact,
)
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.feature_models import FeatureWorkflowSnapshot
from tests.test_feature_workflow import (
    RewordsTheSameDefectExecutor,
    feature_payload,
)
from tools.resolved_issue_ledger import (
    IssueAuthority,
    RecurringIssue,
    ResolvedIssue,
    design_conflict_narrative,
    recurring_resolved_issues,
    resolved_issue_ledger,
    satisfied_invariant_lines,
)
from tools.review_fix_classification import fingerprint_for_text
from workflows.feature_workflow import (
    ChildExecution,
    FeatureWorkflowOrchestrator,
    MockChildWorkstreamExecutor,
)

# One defect, worded three ways, plus one that is genuinely something else. Every wording of
# the first reduces to the same location token, which is the whole basis of the ledger: a
# reviewer rewords its complaint between cycles and a prose comparison reads that as new.
COMPENSATION_RAISED = (
    "Bulk creation at server/services/bulkAppImport.service.js:88 must compensate: delete the "
    "apps it created when a later row fails."
)
COMPENSATION_REWORDED = (
    "At server/services/bulkAppImport.service.js:88 the bulk create has no compensating "
    "cleanup; created apps survive a later row failure."
)
COMPENSATION_AS_INTEGRATION_FIX = (
    "Add compensating cleanup to the bulk creation in "
    "server/services/bulkAppImport.service.js:88 so a later row failure removes what it created."
)
UNRELATED_DEFECT = (
    "The CSV parser at server/util/csv.util.js:41 raises RangeError on an empty header row."
)


def test_three_wordings_of_one_defect_share_an_identity() -> None:
    """The premise every test below rests on, asserted once rather than assumed four times."""
    assert fingerprint_for_text(COMPENSATION_RAISED) == fingerprint_for_text(COMPENSATION_REWORDED)
    assert fingerprint_for_text(COMPENSATION_RAISED) == fingerprint_for_text(
        COMPENSATION_AS_INTEGRATION_FIX
    )
    assert fingerprint_for_text(COMPENSATION_RAISED) != fingerprint_for_text(UNRELATED_DEFECT)


class ScriptedFindingsExecutor:
    """Reject the backend with a scripted list of blocking issues, one entry per attempt.

    Every attempt reports genuinely different production source, so the existing convergence
    guards permit the next one and nothing but the scripted findings can stop the loop. An
    empty entry means the attempt was approved, which is how a lineage records that a review
    stopped asking for something.
    """

    def __init__(self, script: Sequence[Sequence[str]]) -> None:
        """Record the per-attempt findings and count the attempts actually spent."""
        self._delegate = MockChildWorkstreamExecutor()
        self._script = [list(entry) for entry in script]
        self.attempts = 0
        self.feedback: list[list[str]] = []

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Return the scripted verdict for this backend attempt, approving every sibling."""
        execution = await self._delegate.run(**kwargs)
        if kwargs["repository"].repository_id != "backend":
            return execution
        self.feedback.append(list(kwargs["feedback"]))
        self.attempts += 1
        findings = (
            self._script[self.attempts - 1]
            if self.attempts <= len(self._script)
            else self._script[-1]
        )
        if not findings:
            return execution
        touched = [f"server/services/bulkAppImport{self.attempts}.service.js"]
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    "blocking_issues": findings,
                    "pull_request_readiness": False,
                    "production_files_changed": touched,
                    "failure_classification": "validation_source_failure",
                    "production_diff_fingerprint": f"production-{self.attempts}",
                    "test_diff_fingerprint": f"tests-{self.attempts}",
                }
            ),
            code_completion=execution.code_completion,
            review=execution.review,
        )


class DemandingIntegrationReviewer:
    """Ask one repository for one named fix, for ever, so the seam authority is scriptable."""

    def __init__(self, *, repository_id: str, recommended_fix: str) -> None:
        """Record which repository is asked, and in whose exact words."""
        self._repository_id = repository_id
        self._recommended_fix = recommended_fix

    async def review(
        self,
        *,
        feature_id: str,
        contract: IntegrationContractArtifact,
        child_results: Sequence[ChildWorkflowResultArtifact],
        merge_order: Sequence[str],
    ) -> IntegrationReviewArtifact:
        """Return a changes-requested review naming this repository and this fix."""
        return create_artifact(
            IntegrationReviewArtifact,
            workflow_id=feature_id,
            artifact_id=FEATURE_ARTIFACT_FILENAMES["integration_review"],
            producer="integration_reviewer",
            payload={
                "feature_id": feature_id,
                "contract_artifact_id": contract.artifact_id,
                "review_status": "changes_requested",
                "repository_results": [
                    {
                        "repository_id": item.repository_id,
                        "child_workflow_id": item.child_workflow_id,
                        "status": item.status,
                        "child_result_artifact_id": item.artifact_id,
                    }
                    for item in child_results
                ],
                "contract_checks": ["Simulated contract check."],
                "cross_repository_findings": [
                    {
                        "finding_id": "finding-seam-compensation",
                        "severity": "high",
                        "responsible_repository_id": self._repository_id,
                        "affected_repository_ids": [self._repository_id],
                        "contract_reference": "contract",
                        "description": "The seam requires compensating cleanup.",
                        "evidence": "Simulated cross-repository finding.",
                        "recommended_fix": self._recommended_fix,
                    }
                ],
                "compatibility_assessment": "Simulated cross-repository incompatibility.",
                "security_assessment": "No security assessment was performed.",
                "deployment_assessment": "No deployment assessment was performed.",
                "merge_order": list(merge_order),
                "required_fixes": [self._recommended_fix],
            },
            metadata={"source_artifact_ids": [contract.artifact_id]},
        )


async def _run(
    feature_id: str,
    executor: object,
    *,
    integration_reviewer: object | None = None,
    validation_retries: int = 8,
) -> FeatureWorkflowSnapshot:
    """Run one feature to its end with a scripted backend and a generous retry budget."""
    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state(feature_id, request)
    # Generous on purpose. Every assertion below is that a workstream stopped well short of
    # this, so a stop caused by exhaustion would not be mistaken for the stop under test.
    state.max_validation_retries = validation_retries
    orchestrator = FeatureWorkflowOrchestrator(
        child_executor=cast(Any, executor),
        **(
            {"integration_reviewer": cast(Any, integration_reviewer)}
            if integration_reviewer is not None
            else {}
        ),
    )
    return await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )


def _backend_triage(result: Any) -> dict[str, Any]:
    """Return the terminal triage the backend's final attempt recorded."""
    newest: dict[str, Any] = {}
    for artifact in result.artifacts:
        if (
            isinstance(artifact, ChildWorkflowResultArtifact)
            and artifact.repository_id == "backend"
        ):
            newest = dict(artifact.metadata)
    return newest


@pytest.mark.asyncio
async def test_a_resolved_finding_that_comes_back_stops_the_loop_as_a_design_conflict() -> None:
    """B1. Raised, satisfied, demanded again -- which is a decision being reversed, not a bug.

    The lineage is the one no existing guard can see: the finding's wording changes, the
    production source changes every attempt, and one whole attempt sits between the two
    appearances. `diagnostic_signature` compares against the previous attempt only,
    `_revisits_an_earlier_attempt` compares diff bytes, and `unresolved_since` deleted its
    counter the moment the finding disappeared -- so all three read the third attempt as
    ordinary new work and would have bought five more.
    """
    executor = ScriptedFindingsExecutor(
        [
            [COMPENSATION_RAISED],
            [UNRELATED_DEFECT],
            [COMPENSATION_REWORDED],
        ]
    )

    result = await _run("feature-conflict-recurs", executor)

    # Stopped on the attempt that re-raised it, not after another coding call.
    assert executor.attempts == 3
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.FAILED
    triage = _backend_triage(result)
    assert triage["terminal_cause"] == "design_conflict"
    # The narrative quotes both appearances, so a reader can see it is one defect and two
    # sentences rather than two defects.
    question = triage["operator_question"]
    assert "already made and then removed" in question
    assert any(COMPENSATION_REWORDED in item for item in triage["terminal_evidence"])
    assert any(COMPENSATION_RAISED in item for item in triage["terminal_evidence"])
    # Not the vocabulary of the causes this one was split out of: nobody is being sent to
    # repair a repository or to buy a stronger model over a design disagreement.
    assert "what has to be fixed" not in question
    assert "capability limit" not in question
    assert "did not anticipate" not in question
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN


@pytest.mark.asyncio
async def test_a_finding_that_stays_resolved_does_not_stop_and_becomes_an_invariant() -> None:
    """B2. The same lineage without the recurrence: no stop, and the next attempt is told.

    This is the half that makes B1 safe. A satisfied demand is not a reason to end anything;
    it is a reason for the next engineer to be told not to undo it.
    """
    executor = ScriptedFindingsExecutor(
        [
            [COMPENSATION_RAISED],
            [UNRELATED_DEFECT],
            [],
        ]
    )

    result = await _run("feature-conflict-absent", executor)

    assert executor.attempts == 3
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.COMPLETED
    assert _backend_triage(result).get("terminal_cause") != "design_conflict"
    # The third attempt was told what the second one had delivered, whole and marked as
    # satisfied rather than as something to verify.
    invariants = [line for line in executor.feedback[2] if line.startswith("ALREADY SATISFIED")]
    assert len(invariants) == 1
    assert COMPENSATION_RAISED in invariants[0]
    assert "this repository's review" in invariants[0]
    assert "Do not undo it" in invariants[0]
    # And the thing that is still open is not in that list.
    assert not any(UNRELATED_DEFECT in line for line in invariants)


@pytest.mark.asyncio
async def test_a_finding_that_was_never_resolved_is_not_a_design_conflict() -> None:
    """B3. Ordinary non-convergence stays where it was -- 45- section 4c's third case.

    One defect, reworded every attempt, never once absent. There is no reversal here and no
    disagreement between demands: the model simply cannot fix it, and the existing machinery
    already says so correctly.
    """
    executor = RewordsTheSameDefectExecutor("backend")

    result = await _run("feature-never-resolved", executor)

    child = result.child_workflows["backend"]
    assert child.status is ChildWorkflowStatus.FAILED
    assert _backend_triage(result)["terminal_cause"] != "design_conflict"
    assert child.retry_refusal_reason is not None
    assert "not converging" in child.retry_refusal_reason


@pytest.mark.asyncio
async def test_two_authorities_demanding_opposite_things_is_named_as_a_disagreement() -> None:
    """B4. The 184 seam: this repository's review let it go, the integration review requires it.

    Same terminal state as B1 and a different sentence, because it sends its reader somewhere
    else. B1 asks whether a change should stand; this asks which review holds. Retrying cannot
    answer either, but only this one is a conflict between two reviewers rather than inside
    one.
    """
    executor = ScriptedFindingsExecutor(
        [
            [COMPENSATION_RAISED],
            [],
            [UNRELATED_DEFECT],
        ]
    )

    result = await _run(
        "feature-conflict-cross-authority",
        executor,
        integration_reviewer=DemandingIntegrationReviewer(
            repository_id="backend", recommended_fix=COMPENSATION_AS_INTEGRATION_FIX
        ),
    )

    # Approved on attempt 2, sent back by the seam, and stopped on the fix attempt rather
    # than spending the rest of the budget re-arguing it.
    assert executor.attempts == 3
    triage = _backend_triage(result)
    assert triage["terminal_cause"] == "design_conflict"
    question = triage["operator_question"]
    assert "the two authorities disagree and retrying cannot settle it" in question
    assert "the integration review" in question
    assert "Which authority's position holds" in question
    # Neither of the causes this was split out of, in either vocabulary.
    assert "what has to be fixed" not in question
    assert "capability limit" not in question
    assert "did not anticipate" not in question
    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN


def _resolved(index: int, issue: str) -> ResolvedIssue:
    """Build one ledger entry directly, for the bounding rules that need many of them."""
    return ResolvedIssue(
        fingerprint=fingerprint_for_text(issue),
        issue=issue,
        authority=(
            IssueAuthority.INTEGRATION_REVIEW if index % 2 else IssueAuthority.REPOSITORY_REVIEW
        ),
        resolved_at=index,
        grounds="",
    )


def test_the_invariant_section_quotes_whole_issues_and_states_what_it_left_out() -> None:
    """A1. The capture discipline: bounded, never sliced, and never silently halved.

    A satisfied requirement handed back with its object missing is an instruction nobody can
    follow, and a list that quietly stops at the budget reads as the complete set -- so an
    engineer told about four of nine invariants believes there are four.
    """
    issues = [
        f"Requirement {index} at server/services/module{index}.js:{index + 10} must be "
        "preserved exactly as the earlier attempt delivered it, including the behaviour its "
        "review asked for and the surrounding contract it was written against."
        for index in range(20)
    ]
    ledger = [_resolved(index, issue) for index, issue in enumerate(issues)]

    lines = satisfied_invariant_lines(ledger)

    quoted = [line for line in lines if line.startswith("ALREADY SATISFIED")]
    omission = [line for line in lines if "further requirement(s)" in line]
    assert 0 < len(quoted) < len(ledger), "the cap has to actually bite for this to test it"
    # Whole issues, every one of them. Nothing is truncated to fit.
    for line, entry in zip(quoted, ledger, strict=False):
        assert entry.issue in line
    # The omission is stated, and it accounts for every entry that did not fit.
    assert len(omission) == 1
    assert f"{len(ledger) - len(quoted)} further requirement(s)" in omission[0]
    # Every entry carries the authority that demanded it, because which review it was is what
    # a cross-authority conflict is later established from.
    assert any("this repository's review" in line for line in quoted)
    assert any("the integration review" in line for line in quoted)


def test_an_outstanding_demand_is_never_also_listed_as_satisfied() -> None:
    """One defect cannot be both "fix this" and "do not undo this" in the same prompt."""
    ledger = [_resolved(0, COMPENSATION_RAISED), _resolved(2, UNRELATED_DEFECT)]

    lines = satisfied_invariant_lines(ledger, demanded=[COMPENSATION_REWORDED])

    assert not any(COMPENSATION_RAISED in line for line in lines)
    assert any(UNRELATED_DEFECT in line for line in lines)


def test_the_ledger_reads_both_authorities_out_of_one_artifact_lineage() -> None:
    """A demand is satisfied when the review that made it makes its next verdict without it.

    Walked per authority, deliberately: an integration review that says nothing about a
    repository finding is silent about it, not in agreement with dropping it.
    """
    contract_id = "artifact-contract"
    lineage: list[object] = [
        _child_result("backend", "child-backend", 1, [COMPENSATION_RAISED]),
        _integration_review(contract_id, "backend", [UNRELATED_DEFECT]),
        _child_result("backend", "child-backend", 2, []),
        _integration_review(contract_id, "backend", []),
    ]

    ledger = resolved_issue_ledger(
        lineage, child_workflow_id="child-backend", repository_id="backend"
    )

    assert [(entry.issue, entry.authority) for entry in ledger] == [
        (UNRELATED_DEFECT, IssueAuthority.INTEGRATION_REVIEW),
        (COMPENSATION_RAISED, IssueAuthority.REPOSITORY_REVIEW),
    ]
    # And a sibling's lineage is not this one's: a finding resolved in the frontend says
    # nothing about what the backend may undo.
    assert (
        resolved_issue_ledger(lineage, child_workflow_id="child-frontend", repository_id="frontend")
        == []
    )


# AB-Feature-216's two blocking findings, verbatim from the persisted artifacts. They are two
# different demands -- missing second-page test coverage, and a failure path serializing a 200
# CSV -- that reduce to the single token `code:GAM_AD_UNIT_EXPORT_FAILED` and therefore to one
# fingerprint. The run was ended on that collision, arbitrated by a cycle in which no review ran.
FEATURE_216_ATTEMPT_1_FINDING = (
    "The scoped test requirements explicitly require a failed second-page retrieval to produce "
    "GAM_AD_UNIT_EXPORT_FAILED and no successful CSV response. "
    "server/gamUtils/ad_unit/test_get_ad_units.py tests successful multi-page traversal only. "
    "The Jest GAM-failure test rejects the single mocked Python invocation directly, so it does "
    "not exercise or establish failure during a later page retrieval in fetch_ad_units."
)
FEATURE_216_ATTEMPT_3_FINDING = (
    "The Python executable catches every exception, including a second-page "
    "getAdUnitsByStatement failure, prints a JSON array containing success: false, and exits "
    "without re-raising. fetchActiveAdUnitsFromGAM only rejects when runPythonScript rejects or "
    "its returned value is not an array. If the executor returns the printed array, the service "
    "filters it to an empty array and serializes a 200 CSV attachment, violating the "
    "requirement that incomplete GAM retrieval returns GAM_AD_UNIT_EXPORT_FAILED and never a "
    "successful partial export."
)
FEATURE_216_GATE_FINDING = (
    "server/conftest.py was added but no production file refers to it, so the running "
    "application cannot reach it."
)


def test_a_review_less_gate_cycle_neither_resolves_nor_recurs_a_demand() -> None:
    """The AB-Feature-216 shape: a demand 'resolved' by an attempt in which no review ran.

    Attempt 1's review demanded test coverage; attempt 2 died at the reachability gate before
    any review, carrying `deterministic_gate` and no review artifact; attempt 3's review raised
    a different finding that fingerprints identically. The gate's silence was read as the
    demand being satisfied, and the collision as its reversal -- a stop built from two events
    that never happened.
    """
    assert fingerprint_for_text(FEATURE_216_ATTEMPT_1_FINDING) == fingerprint_for_text(
        FEATURE_216_ATTEMPT_3_FINDING
    )
    lineage: list[object] = [
        _child_result("backend", "child-backend", 1, [FEATURE_216_ATTEMPT_1_FINDING]),
        _child_result(
            "backend",
            "child-backend",
            2,
            [FEATURE_216_GATE_FINDING],
            reviewed=False,
            metadata={"deterministic_gate": True},
        ),
    ]

    ledger = resolved_issue_ledger(
        lineage, child_workflow_id="child-backend", repository_id="backend"
    )

    assert ledger == []
    assert (
        recurring_resolved_issues(
            ledger,
            demanded={IssueAuthority.REPOSITORY_REVIEW: [FEATURE_216_ATTEMPT_3_FINDING]},
        )
        == []
    )


def test_a_result_with_a_review_but_no_gate_marker_still_counts() -> None:
    """The fence cannot silently disable the ledger: real review cycles keep resolving."""
    lineage: list[object] = [
        _child_result("backend", "child-backend", 1, [COMPENSATION_RAISED]),
        _child_result("backend", "child-backend", 2, [UNRELATED_DEFECT]),
    ]

    ledger = resolved_issue_ledger(
        lineage, child_workflow_id="child-backend", repository_id="backend"
    )

    assert [entry.issue for entry in ledger] == [COMPENSATION_RAISED]


def test_a_gate_marker_alone_excludes_a_cycle_even_when_a_review_id_is_present() -> None:
    """Both halves of the fence hold on their own: the marker, and the missing review."""
    marked: list[object] = [
        _child_result("backend", "child-backend", 1, [COMPENSATION_RAISED]),
        _child_result(
            "backend",
            "child-backend",
            2,
            [UNRELATED_DEFECT],
            metadata={"deterministic_gate": True},
        ),
    ]
    unreviewed: list[object] = [
        _child_result("backend", "child-backend", 1, [COMPENSATION_RAISED]),
        _child_result("backend", "child-backend", 2, [UNRELATED_DEFECT], reviewed=False),
    ]

    for lineage in (marked, unreviewed):
        assert (
            resolved_issue_ledger(
                lineage, child_workflow_id="child-backend", repository_id="backend"
            )
            == []
        )


def test_a_lone_code_token_match_needs_the_prose_to_agree() -> None:
    """216's collision: two different demands, one fingerprint, no recurrence.

    Both sentences reduce to the single token `code:GAM_AD_UNIT_EXPORT_FAILED`, and in a
    contract-driven feature every finding about an endpoint names its error codes -- so a lone
    code token is the normal collision, not a rare one. A genuine repeat of the same demand
    still recurs, because repeats agree in prose as well.
    """
    ledger = [
        ResolvedIssue(
            fingerprint=fingerprint_for_text(FEATURE_216_ATTEMPT_1_FINDING),
            issue=FEATURE_216_ATTEMPT_1_FINDING,
            authority=IssueAuthority.REPOSITORY_REVIEW,
            resolved_at=2,
            grounds="",
        )
    ]

    collided = recurring_resolved_issues(
        ledger, demanded={IssueAuthority.REPOSITORY_REVIEW: [FEATURE_216_ATTEMPT_3_FINDING]}
    )
    # The same demand back again, lightly reworded: the prose agrees, so it still stops.
    repeat = FEATURE_216_ATTEMPT_1_FINDING.replace(
        "The scoped test requirements explicitly require",
        "The scoped test requirements require",
    )
    assert fingerprint_for_text(repeat) == ledger[0].fingerprint
    repeated = recurring_resolved_issues(
        ledger, demanded={IssueAuthority.REPOSITORY_REVIEW: [repeat]}
    )

    assert collided == []
    assert [item.issue for item in repeated] == [repeat]


def test_a_code_token_with_a_location_anchor_still_recurs() -> None:
    """A multi-token identity already says which defect it is; the prose test never runs."""
    raised = (
        "The export at server/services/adunit.export.service.js:42 returns "
        "GAM_AD_UNIT_EXPORT_FAILED for a complete retrieval."
    )
    reworded = (
        "A complete retrieval is answered with GAM_AD_UNIT_EXPORT_FAILED by "
        "server/services/adunit.export.service.js:42, which is wrong."
    )
    assert fingerprint_for_text(raised) == fingerprint_for_text(reworded)
    ledger = [
        ResolvedIssue(
            fingerprint=fingerprint_for_text(raised),
            issue=raised,
            authority=IssueAuthority.REPOSITORY_REVIEW,
            resolved_at=2,
            grounds="",
        )
    ]

    recurrences = recurring_resolved_issues(
        ledger, demanded={IssueAuthority.REPOSITORY_REVIEW: [reworded]}
    )

    assert [item.issue for item in recurrences] == [reworded]


def test_a_prose_only_identity_is_unchanged_by_the_corroboration_rule() -> None:
    """An identity made of normalized prose is not a code token, so it behaves as before."""
    issue = "The export button stays disabled after a successful download completes."
    ledger = [
        ResolvedIssue(
            fingerprint=fingerprint_for_text(issue),
            issue=issue,
            authority=IssueAuthority.REPOSITORY_REVIEW,
            resolved_at=2,
            grounds="",
        )
    ]

    recurrences = recurring_resolved_issues(
        ledger, demanded={IssueAuthority.REPOSITORY_REVIEW: [issue]}
    )

    assert [item.issue for item in recurrences] == [issue]


def test_the_216_shape_produces_no_reversal_to_narrate() -> None:
    """Composition: ledger fence plus corroboration leave nothing for the narrative to say.

    The exact artifact shape that ended AB-Feature-216 -- a reviewed demand, a review-less
    gate cycle, and a colliding demand at the next review -- must produce no recurrence at
    either layer, so no design-conflict narrative can exist to mislead an operator.
    """
    lineage: list[object] = [
        _child_result("backend", "child-backend", 1, [FEATURE_216_ATTEMPT_1_FINDING]),
        _child_result(
            "backend",
            "child-backend",
            2,
            [FEATURE_216_GATE_FINDING],
            reviewed=False,
            metadata={"deterministic_gate": True},
        ),
        _child_result("backend", "child-backend", 3, [FEATURE_216_ATTEMPT_3_FINDING]),
    ]

    ledger = resolved_issue_ledger(
        lineage, child_workflow_id="child-backend", repository_id="backend"
    )

    assert (
        recurring_resolved_issues(
            ledger,
            demanded={IssueAuthority.REPOSITORY_REVIEW: [FEATURE_216_ATTEMPT_3_FINDING]},
        )
        == []
    )


def test_the_grounds_clause_is_quoted_only_when_it_reads_as_a_removal() -> None:
    """A summary with no removal-shaped wording is not quoted as the grounds for one."""
    additive_grounds = RecurringIssue(
        issue=COMPENSATION_REWORDED,
        authority=IssueAuthority.REPOSITORY_REVIEW,
        resolved=ResolvedIssue(
            fingerprint=fingerprint_for_text(COMPENSATION_RAISED),
            issue=COMPENSATION_RAISED,
            authority=IssueAuthority.REPOSITORY_REVIEW,
            resolved_at=2,
            grounds="Applied targeted updates by adding explicit unittest coverage.",
        ),
    )
    removal_grounds = RecurringIssue(
        issue=COMPENSATION_REWORDED,
        authority=IssueAuthority.REPOSITORY_REVIEW,
        resolved=additive_grounds.resolved.model_copy(
            update={"grounds": "Removed the compensating cleanup the review had asked for."}
        ),
    )

    additive_question, additive_evidence = design_conflict_narrative(additive_grounds)
    removal_question, _ = design_conflict_narrative(removal_grounds)

    assert "most recently on the grounds" not in additive_question
    assert "most recently on the grounds" in removal_question
    assert "Removed the compensating cleanup" in removal_question
    # The evidence still quotes what the attempt reported, because that is a fact either way.
    assert any("The attempt that removed it reported" in item for item in additive_evidence)
    """The tag is what separates B1's narrative from B4's, so it is asserted on its own."""
    ledger = [_resolved(0, COMPENSATION_RAISED)]

    same = recurring_resolved_issues(
        ledger, demanded={IssueAuthority.REPOSITORY_REVIEW: [COMPENSATION_REWORDED]}
    )
    crossed = recurring_resolved_issues(
        ledger, demanded={IssueAuthority.INTEGRATION_REVIEW: [COMPENSATION_AS_INTEGRATION_FIX]}
    )
    none = recurring_resolved_issues(
        ledger, demanded={IssueAuthority.REPOSITORY_REVIEW: [UNRELATED_DEFECT]}
    )

    assert [item.cross_authority for item in same] == [False]
    assert [item.cross_authority for item in crossed] == [True]
    assert none == []


def _child_result(
    repository_id: str,
    child_workflow_id: str,
    attempt: int,
    blocking_issues: Sequence[str],
    *,
    reviewed: bool = True,
    metadata: dict[str, Any] | None = None,
) -> ChildWorkflowResultArtifact:
    """Build one persisted attempt result, with only the fields the ledger reads.

    ``reviewed`` says whether a review actually spoke in this cycle. The ledger counts only
    those; a gate-stopped attempt is built with ``reviewed=False`` and, where the live executor
    would stamp it, ``metadata={"deterministic_gate": True}``.
    """
    return create_artifact(
        ChildWorkflowResultArtifact,
        workflow_id="feature-ledger",
        artifact_id=f"artifact-result-{repository_id}-{attempt}",
        producer="child_workflow",
        payload={
            "feature_id": "feature-ledger",
            "parent_workflow_id": "feature-ledger",
            "child_workflow_id": child_workflow_id,
            "repository_id": repository_id,
            "workstream_id": f"workstream-{repository_id}",
            "branch_name": f"feature/{repository_id}",
            "workspace_path": f"/tmp/{repository_id}",
            "code_completion_artifact_id": None,
            "review_artifact_id": (
                f"artifact-review-{repository_id}-{attempt}" if reviewed else None
            ),
            "changed_files": [],
            "validation_results": [],
            "status": "approved" if not blocking_issues else "failed",
            "blocking_issues": list(blocking_issues),
            "pull_request_readiness": False,
            "contract_sections_consumed": [],
            "contract_sections_implemented": [],
        },
        metadata=dict(metadata or {}),
    )


def _integration_review(
    contract_artifact_id: str, repository_id: str, fixes: Sequence[str]
) -> IntegrationReviewArtifact:
    """Build one persisted integration verdict; no fixes means it approved."""
    return create_artifact(
        IntegrationReviewArtifact,
        workflow_id="feature-ledger",
        artifact_id=f"artifact-integration-{repository_id}-{len(fixes)}-{fixes!r:.16}",
        producer="integration_reviewer",
        payload={
            "feature_id": "feature-ledger",
            "contract_artifact_id": contract_artifact_id,
            "review_status": "changes_requested" if fixes else "approved",
            "repository_results": [],
            "contract_checks": [],
            "cross_repository_findings": [
                {
                    "finding_id": f"finding-{index}",
                    "severity": "high",
                    "responsible_repository_id": repository_id,
                    "affected_repository_ids": [repository_id],
                    "contract_reference": "contract",
                    "description": "A cross-repository finding.",
                    "evidence": "Evidence.",
                    "recommended_fix": fix,
                }
                for index, fix in enumerate(fixes)
            ],
            "compatibility_assessment": "Assessed.",
            "security_assessment": "Assessed.",
            "deployment_assessment": "Assessed.",
            "merge_order": [],
            "required_fixes": list(fixes),
        },
        metadata={},
    )
