"""An overruled review demand cannot fail the attempt (73-, backlog item 27).

A ``removal_holds`` verdict binds the Engineer (the settled line it may not argue) and the
retry authority (the recurrence stop excludes the settled fingerprint) -- but until 73- it
never bound the Reviewer. A review that re-raised the overruled demand failed the attempt on
every retry, and the one guard built for re-litigation was deliberately blind to exactly that
fingerprint, so the workstream burned its whole budget re-arguing a question a person had
already answered.

What is asserted here, in the spec's order: the demotion and its identity rule, the
measurement exemption, the fenced flip to publication, the audit trace on both artifacts and
in the pull-request body, the ledger interplay, the reviewer prompt clause, and the live
executor path end to end. Effects, not commands: the load-bearing assertions are on what an
attempt's result says and what a person can read, not on which helper filtered what.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from adapters.llm_adapter import MockCodingExecutor
from agents.engineer.agent import EngineerAgent
from agents.reviewer.agent import ReviewerAgent
from agents.shared.contracts import create_artifact
from api.control_plane import RequestScopedCredentials
from artifacts.schemas import DesignConflictArtifact, ReviewFinding
from prompts.prompt_loader import PromptLoader
from services.cancellation import MockCancellationToken
from services.design_conflict import settled_questions
from services.feature_runtime import (
    LiveChildWorkstreamExecutor,
    _RemediationReviewScope,
    _resolve_review_outcome,
)
from tests.test_agents import (
    StaticValidationTool,
    agent_state,
    task_plan_artifact,
    technical_prd_artifact,
    validation_result,
)
from tests.test_bounded_review_scope import _body, _finding, _publishable_result, _review
from tests.test_live_child_executor import (
    RecordingProcessRunner,
    ScriptedLLMClient,
    _changed_status_payload,
    _harness,
    _live_run_context,
    _review_payload,
    _workstream,
)
from tools.resolved_issue_ledger import (
    IssueAuthority,
    ResolvedIssue,
    SettledQuestion,
    recurring_resolved_issues,
    satisfied_invariant_lines,
)
from tools.review_fix_classification import fingerprint_for_text

DEMAND = (
    "The bulk import endpoint must delete every previously imported row before applying a "
    "new upload, so a re-upload cannot leave stale rows behind."
)
# The same defect identity under `defect_identity`'s normalization -- casing and whitespace
# moved, words unmoved. A rewording that changes the word sequence is a different identity to
# the recurrence stop too, and 73- deliberately matches nothing the stop would not.
DEMAND_REWORDED = (
    "The bulk import endpoint MUST delete   every previously imported row before applying a "
    "new upload, so a re-upload cannot leave stale rows behind."
)
OVERRULE = (
    "The review is wrong to require it: imports are append-only by design, and reconciliation "
    "handles duplicates downstream. Do not implement the delete."
)
DECIDED_AT = datetime(2026, 9, 4, 12, 0, 0, tzinfo=UTC)


def _settled(
    demand: str = DEMAND,
    *,
    verdict: str = "removal_holds",
    conflict_id: str = "conflict-backend-test",
) -> SettledQuestion:
    """One decided question, carrying the audit fields the demotion stamps onto the record."""
    return SettledQuestion(
        fingerprint=fingerprint_for_text(demand),
        verdict=verdict,
        decision=OVERRULE,
        demand=demand,
        authority=IssueAuthority.REPOSITORY_REVIEW,
        decided_by="akhilesh (via platform key)",
        conflict_id=conflict_id,
        decided_at=DECIDED_AT,
    )


def _demanding(finding_id: str, description: str, **kwargs: Any) -> ReviewFinding:
    """A blocking finding whose description is the demand under test."""
    return _finding(finding_id, **kwargs).model_copy(update={"description": description})


# --------------------------------------------------------------------------------------
# The demotion and its identity rule
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bounded", [False, True], ids=["unbounded", "bounded"])
def test_a_removal_holds_demand_is_demoted_and_the_work_publishes(bounded: bool) -> None:
    """The core of 73-, in both modes: the verdict is a human authority, not an A/B arm."""
    review = _review(
        findings=[
            _demanding("REQ-1", DEMAND, category="requirement", requirement_id="backend-status-api")
        ]
    )

    outcome = _resolve_review_outcome(
        review, bounded=bounded, publishable=True, settled=[_settled()]
    )

    assert outcome.blocking_issues == []
    assert outcome.advisory_verdict == "changes_requested"
    assert [item.finding_id for item in outcome.overruled_findings] == ["REQ-1"]
    record = outcome.overruled_findings[0]
    assert record.fingerprint == fingerprint_for_text(DEMAND)
    assert record.conflict_id == "conflict-backend-test"
    assert record.verdict == "removal_holds"
    assert record.decision == OVERRULE
    assert record.decided_by == "akhilesh (via platform key)"
    assert record.decided_at == DECIDED_AT


@pytest.mark.parametrize("bounded", [False, True], ids=["unbounded", "bounded"])
def test_requirement_holds_demotes_nothing(bounded: bool) -> None:
    """The demand stands, so a review re-raising it is the review doing its job."""
    review = _review(
        findings=[
            _demanding("REQ-1", DEMAND, category="requirement", requirement_id="backend-status-api")
        ]
    )

    outcome = _resolve_review_outcome(
        review,
        bounded=bounded,
        publishable=True,
        settled=[_settled(verdict="requirement_holds")],
    )

    assert outcome.blocking_issues == [DEMAND]
    assert outcome.overruled_findings == []
    assert outcome.advisory_verdict is None


def test_the_verdict_removes_one_question_and_blinds_the_review_to_nothing_else() -> None:
    """A genuinely new defect in the same file keeps full blocking authority."""
    new_defect = "The upload handler never closes the temporary file on the error path."
    review = _review(
        findings=[
            _demanding(
                "REQ-1", DEMAND, category="requirement", requirement_id="backend-status-api"
            ),
            _demanding(
                "REQ-2", new_defect, category="requirement", requirement_id="backend-status-api"
            ),
        ]
    )

    outcome = _resolve_review_outcome(review, bounded=True, publishable=True, settled=[_settled()])

    # The attempt still fails -- over the new defect and only the new defect.
    assert outcome.blocking_issues == [new_defect]
    assert outcome.advisory_verdict is None
    assert [item.finding_id for item in outcome.overruled_findings] == ["REQ-1"]


def test_the_match_is_the_ledger_fingerprint_and_nothing_looser() -> None:
    """A reworded demand with the same defect identity demotes; severity never matters."""
    review = _review(
        findings=[
            _demanding(
                "REQ-1",
                DEMAND_REWORDED,
                severity="critical",
                category="requirement",
                requirement_id="backend-status-api",
            )
        ]
    )
    assert fingerprint_for_text(DEMAND_REWORDED) == fingerprint_for_text(DEMAND)

    outcome = _resolve_review_outcome(review, bounded=False, publishable=True, settled=[_settled()])

    assert outcome.blocking_issues == []
    assert [item.description for item in outcome.overruled_findings] == [DEMAND_REWORDED]


def test_a_deterministic_finding_is_never_overruled_whatever_its_fingerprint() -> None:
    """The verdict binds judgement, not measurement -- 204's install sentence is the proof.

    A settled fingerprint can name a failing command (AB-Feature-204's conflict does exactly
    that), and no verdict makes a failing command pass. The finding keeps blocking.
    """
    gate_sentence = "Deterministic dependency installation failed before coding began."
    review = _review(
        findings=[_demanding("validation-install", gate_sentence, category="validation_failure")],
        deterministic=["validation-install"],
    )

    outcome = _resolve_review_outcome(
        review, bounded=False, publishable=True, settled=[_settled(gate_sentence)]
    )

    assert outcome.blocking_issues == [gate_sentence]
    assert outcome.overruled_findings == []
    assert outcome.advisory_verdict is None


# --------------------------------------------------------------------------------------
# The fenced flip
# --------------------------------------------------------------------------------------


def test_the_flip_keeps_every_publication_fence() -> None:
    """The verdict answers a design question; it cannot stand in for the other fences."""
    findings = [
        _demanding("REQ-1", DEMAND, category="requirement", requirement_id="backend-status-api")
    ]

    unpublishable = _resolve_review_outcome(
        _review(findings=findings), bounded=False, publishable=False, settled=[_settled()]
    )
    assert unpublishable.advisory_verdict is None
    # The record of why the attempt failed never names the demand a person ruled out --
    # here nothing else blocked, so the record is empty rather than re-litigating.
    assert DEMAND not in unpublishable.blocking_issues
    # The suppression trace survives the refusal: the demotion happened either way.
    assert [item.finding_id for item in unpublishable.overruled_findings] == ["REQ-1"]

    manual = _resolve_review_outcome(
        _review(findings=findings, manual_review_required=True),
        bounded=False,
        publishable=True,
        settled=[_settled()],
    )
    assert manual.advisory_verdict is None


def test_the_unbounded_path_accepts_only_on_the_verdicts_authority() -> None:
    """Without an overruled demand the unbounded path never flips -- pre-73 behaviour."""
    review = _review(
        findings=[
            _demanding("REQ-1", DEMAND, category="requirement", requirement_id="backend-status-api")
        ]
    )

    outcome = _resolve_review_outcome(review, bounded=False, publishable=True, settled=[])

    assert outcome.blocking_issues == [DEMAND]
    assert outcome.advisory_verdict is None
    assert outcome.overruled_findings == []


def test_remaining_findings_stay_visible_on_an_unbounded_acceptance() -> None:
    """A publication resting on a verdict must not make the review's other remarks vanish."""
    remark = "The handler would read better split into validation and persistence phases."
    review = _review(
        findings=[
            _demanding(
                "REQ-1", DEMAND, category="requirement", requirement_id="backend-status-api"
            ),
            _demanding("LOW-1", remark, severity="low"),
        ]
    )

    outcome = _resolve_review_outcome(review, bounded=False, publishable=True, settled=[_settled()])

    assert outcome.advisory_verdict == "changes_requested"
    assert outcome.advisory_findings == [remark]


def test_the_counts_record_the_demotion() -> None:
    """The bounded census names how many findings the verdict demoted."""
    review = _review(
        findings=[
            _demanding("REQ-1", DEMAND, category="requirement", requirement_id="backend-status-api")
        ]
    )

    outcome = _resolve_review_outcome(review, bounded=True, publishable=True, settled=[_settled()])

    assert outcome.counts is not None
    assert outcome.counts.findings_overruled == 1
    assert outcome.counts.findings_blocking == 0


def test_the_demotion_runs_after_the_remediation_filter_not_instead_of_it() -> None:
    """The two authorities compose: remediation scope first, then the verdict."""
    review = _review(
        findings=[
            _demanding("REQ-1", DEMAND, category="requirement", requirement_id="backend-status-api")
        ]
    )
    remediation = _RemediationReviewScope(
        prior_finding_ids=frozenset({"REQ-1"}),
        prior_fingerprints=frozenset(),
        delta_paths=frozenset(),
    )

    outcome = _resolve_review_outcome(
        review, bounded=True, publishable=True, remediation=remediation, settled=[_settled()]
    )

    assert outcome.blocking_issues == []
    assert [item.finding_id for item in outcome.overruled_findings] == ["REQ-1"]


# --------------------------------------------------------------------------------------
# The trace a person reads
# --------------------------------------------------------------------------------------


def _overruled_record(**overrides: Any) -> dict[str, Any]:
    """The serialized audit record as the result artifact carries it."""
    record: dict[str, Any] = {
        "finding_id": "REQ-1",
        "description": DEMAND,
        "fingerprint": fingerprint_for_text(DEMAND),
        "conflict_id": "conflict-backend-test",
        "verdict": "removal_holds",
        "decision": OVERRULE,
        "decided_by": "akhilesh (via platform key)",
        "decided_at": DECIDED_AT,
    }
    record.update(overrides)
    return record


def test_the_pull_request_body_shows_the_overruled_finding_as_judged() -> None:
    """Which finding, whose decision, when -- readable where the person reads the change."""
    body = _body(
        _publishable_result(
            overruled_findings=[_overruled_record()],
            advisory_review_verdict="changes_requested",
        )
    )

    assert "Overruled review findings (a person ruled these demands do not stand):" in body
    assert f"- {DEMAND}" in body
    assert "Overruled by akhilesh (via platform key) on 2026-09-04" in body
    assert OVERRULE in body
    # The acceptance sentence states the verdict's authority instead of the untraceable
    # sentence, which would be false: this finding named its requirement perfectly well.
    assert "overruled by design verdict" in body
    assert "and every finding it raised named no scoped requirement" not in body
    # And the body never claims an approval the reviewer did not give.
    assert "Repository review passed" not in body


def test_an_overruled_finding_is_never_presented_as_unjudged() -> None:
    """The advisory heading says nobody judged these; an overruled finding was judged."""
    body = _body(
        _publishable_result(
            overruled_findings=[_overruled_record()],
            advisory_review_verdict="changes_requested",
        )
    )

    advisory_heading = "Advisory review findings (not blocking; nobody has judged these yet):"
    assert advisory_heading not in body


# --------------------------------------------------------------------------------------
# The ledger interplay, and the settled projection the demotion reads
# --------------------------------------------------------------------------------------


def test_the_settled_projection_carries_the_audit_fields() -> None:
    """`settled_questions` hands the demotion its conflict id and timestamp in one read."""
    conflict = create_artifact(
        DesignConflictArtifact,
        workflow_id="feature-live",
        artifact_id="017_design_conflict.conflict-backend-test.json",
        producer="feature_workflow",
        payload={
            "conflict_id": "conflict-backend-test",
            "feature_id": "feature-live",
            "repository_id": "backend",
            "child_workflow_id": "feature-live:backend",
            "fingerprint": fingerprint_for_text(DEMAND),
            "question": "Should the change stand, or is the review wrong to require it?",
            "evidence": [],
            "demanded_by": "repository_review",
            "demand": DEMAND,
            "satisfied_under": "repository_review",
            "satisfied_demand": DEMAND,
            "status": "answered",
            "verdict": "removal_holds",
            "decision": OVERRULE,
            "decided_by": "akhilesh (via platform key)",
            "decided_at": DECIDED_AT,
        },
        metadata={},
    )

    settled = settled_questions([conflict], repository_id="backend")

    assert len(settled) == 1
    assert settled[0].conflict_id == "conflict-backend-test"
    assert settled[0].decided_at == DECIDED_AT


def test_the_ledger_reads_a_demoted_cycle_through_the_existing_exclusions() -> None:
    """Design question (c): no second stop and no contradicting invariant, by the same key.

    After the demotion the demand leaves `blocking_issues`, so the ledger reads it as
    dropped and re-derives a resolved entry for the settled fingerprint. Both readers must
    keep excluding it -- one fingerprint, three authorities, no drift.
    """
    settled = _settled()
    ledger = [
        ResolvedIssue(
            fingerprint=settled.fingerprint,
            issue=DEMAND,
            authority=IssueAuthority.REPOSITORY_REVIEW,
            resolved_at=4,
            removals=2,
        )
    ]
    demanded = {IssueAuthority.REPOSITORY_REVIEW: [DEMAND_REWORDED]}

    assert recurring_resolved_issues(ledger, demanded=demanded, settled=[settled.fingerprint]) == []
    assert satisfied_invariant_lines(ledger, demanded=[], settled=[settled.fingerprint]) == []


# --------------------------------------------------------------------------------------
# The reviewer is told
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_reviewer_prompt_states_the_overruled_demand_verbatim(tmp_path: Path) -> None:
    """Told to the model here, enforced by the platform's verdict handling either way."""
    client = ScriptedLLMClient(
        [
            {
                "verdict": "approved",
                "summary": "The work satisfies its scoped requirements.",
                "requirement_checks": [],
                "findings": [],
                "architecture_assessment": "Consistent with the repository layout.",
                "security_assessment": "No credential handling changed.",
                "test_coverage_assessment": "Covered by the configured commands.",
            }
        ]
    )
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(file_updates={"src/service.py": "value = 1\n"}),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]

    await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
        settled_questions=[_settled(), _settled(verdict="requirement_holds")],
    ).run(agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion]))

    instructions = client.calls[0][0]
    assert f"- Overruled demand: {DEMAND}" in instructions
    assert OVERRULE in instructions
    assert "akhilesh (via platform key)" in instructions
    # Exactly one rendering: the requirement_holds entry needs no reviewer clause -- the
    # demand stands, and re-raising it is the review doing its job.
    assert instructions.count(f"- Overruled demand: {DEMAND}") == 1


@pytest.mark.asyncio
async def test_the_reviewer_prompt_is_untouched_with_nothing_overruled(tmp_path: Path) -> None:
    """The common review pays nothing for a verdict nobody has given."""
    client = ScriptedLLMClient(
        [
            {
                "verdict": "approved",
                "summary": "The work satisfies its scoped requirements.",
                "requirement_checks": [],
                "findings": [],
                "architecture_assessment": "Consistent with the repository layout.",
                "security_assessment": "No credential handling changed.",
                "test_coverage_assessment": "Covered by the configured commands.",
            }
        ]
    )
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(file_updates={"src/service.py": "value = 1\n"}),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]

    await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    ).run(agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion]))

    assert "Overruled demand" not in client.calls[0][0]


# --------------------------------------------------------------------------------------
# The live executor, end to end
# --------------------------------------------------------------------------------------


def _declining_review_payload(*descriptions: str) -> dict[str, Any]:
    """A reviewer response declining the work over the given demand(s), schema-complete."""
    payload = _review_payload(verdict="changes_requested")
    payload["findings"] = [
        {
            "finding_id": f"finding-{index}",
            "severity": "high",
            "title": "Demanded change",
            "description": description,
            "recommendation": "Make the demanded change.",
            "file_path": "src/routes/status.js",
            "line_number": 2,
            "repository_id": "backend",
            "requirement_id": "backend-status-api",
            "contract_reference": None,
            "responsibility": "implements",
            "validated_revision": "rev-1",
            "evidence": "The scoped source area was read.",
            "recommended_fix": "Make the demanded change.",
            "finding_category": "requirement",
        }
        for index, description in enumerate(descriptions, start=1)
    ]
    return payload


def _answered_conflict() -> DesignConflictArtifact:
    """The answered question as the verdict path persists it, for the executor to read."""
    return create_artifact(
        DesignConflictArtifact,
        workflow_id="feature-live",
        artifact_id="017_design_conflict.conflict-backend-test.json",
        producer="feature_workflow",
        payload={
            "conflict_id": "conflict-backend-test",
            "feature_id": "feature-live",
            "repository_id": "backend",
            "child_workflow_id": "feature-live:backend",
            "fingerprint": fingerprint_for_text(DEMAND),
            "question": "Should the change stand, or is the review wrong to require it?",
            "evidence": [],
            "demanded_by": "repository_review",
            "demand": DEMAND,
            "satisfied_under": "repository_review",
            "satisfied_demand": DEMAND,
            "status": "answered",
            "verdict": "removal_holds",
            "decision": OVERRULE,
            "decided_by": "akhilesh (via platform key)",
            "decided_at": DECIDED_AT,
        },
        metadata={"repository_id": "backend", "source": "resolved_issue_ledger"},
    )


async def _run_with_verdict(harness: dict[str, Any], *, reviewer: ScriptedLLMClient) -> Any:
    """Execute one live child attempt on a feature that carries the answered conflict."""
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=ScriptedLLMClient([_changed_status_payload()]),
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=RecordingProcessRunner(),
    )
    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    feature.artifacts.append(_answered_conflict())
    try:
        return await executor.run(
            feature=feature,
            repository=repository,
            workstream=_workstream(),
            child=child,
            technical_prd=technical_prd,
            contract=contract,
            feedback=[],
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()


@pytest.mark.asyncio
async def test_a_review_re_raising_the_overruled_demand_cannot_fail_the_attempt(
    tmp_path: Path,
) -> None:
    """The whole loop, on the live executor and the deployed configuration.

    The reviewer declines the work over exactly the demand a person overruled. Before 73-
    this attempt failed and every retry after it failed the same way; now it publishes, and
    both records -- the review artifact and the result -- say what was suppressed, on whose
    decision, and when.
    """
    harness = await _harness(tmp_path)
    reviewer = ScriptedLLMClient([_declining_review_payload(DEMAND)])

    execution = await _run_with_verdict(harness, reviewer=reviewer)

    result = execution.result
    assert result.status == "approved"
    assert result.blocking_issues == []
    assert result.advisory_review_verdict == "changes_requested"
    assert [item.description for item in result.overruled_findings] == [DEMAND]
    assert result.overruled_findings[0].decided_by == "akhilesh (via platform key)"
    assert result.overruled_findings[0].conflict_id == "conflict-backend-test"
    # The audit trace is on the review artifact itself, not only on the result.
    trace = execution.review.metadata["overruled_findings"]
    assert [item["description"] for item in trace] == [DEMAND]
    assert trace[0]["decision"] == OVERRULE
    # And the reviewer was told, in the decider's own words.
    assert OVERRULE in reviewer.calls[0][0]


@pytest.mark.asyncio
async def test_a_genuinely_new_defect_still_fails_the_attempt_beside_the_verdict(
    tmp_path: Path,
) -> None:
    """Binding the verdict must not blind the review: the other finding still blocks."""
    new_defect = "The status route returns 200 for a payload the schema rejects."
    harness = await _harness(tmp_path)
    reviewer = ScriptedLLMClient([_declining_review_payload(new_defect, DEMAND)])

    execution = await _run_with_verdict(harness, reviewer=reviewer)

    result = execution.result
    assert result.status == "failed"
    assert result.blocking_issues == [new_defect]
    assert DEMAND not in result.blocking_issues
    assert [item.description for item in result.overruled_findings] == [DEMAND]
