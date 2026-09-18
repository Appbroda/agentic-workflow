"""What a review finding must name before it may spend one of a workstream's attempts.

Audit risk P2-9. `review_scope_failure` was 34 of 281 recorded workstreams -- the third
largest failure class -- and none of those reviews was wrong. They were unbounded: nothing
required a finding to derive from the requirement, the contract, a failed command or an
implementation expectation, so a competent reviewer could always find one more thing and
each one cost an attempt against a ceiling of five.

The change bounds what *blocks*, never what the reviewer is allowed to notice. A finding
that names nothing becomes advisory: recorded on the result, carried into the pull-request
body, judged by a person. That is a real trade-off -- some weaker work reaches a human
instead of being rewritten -- which is why it lands behind a setting that defaults off and
why the switch-off equivalence below is asserted rather than assumed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from adapters.llm_adapter import ImageInput, LLMResponse, MockCodingExecutor
from agents.engineer.agent import EngineerAgent
from agents.reviewer.agent import ReviewerAgent
from agents.shared.contracts import create_artifact
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import (
    ChildWorkflowResultArtifact,
    ReviewArtifact,
    ReviewFinding,
)
from prompts.prompt_loader import PromptLoader
from services.feature_runtime import _RemediationReviewScope, _resolve_review_outcome
from state.enums import ChildWorkflowStatus
from state.feature_models import ChildWorkflowReference
from tests.test_agents import (
    StaticValidationTool,
    agent_state,
    domain_payload,
    review_artifact,
    task_plan_artifact,
    technical_prd_artifact,
    validation_result,
)
from tests.test_live_child_executor import (
    ScriptedLLMClient,
    _contract,
    _feature_payload,
    _harness,
    _review_payload,
    _run,
)
from tools.review_fix_classification import finding_fingerprint
from workflows.feature_workflow import _pull_request_body

# --------------------------------------------------------------------------------------
# Findings, by what they name
# --------------------------------------------------------------------------------------


def _finding(
    finding_id: str,
    *,
    severity: str = "high",
    category: str = "code_quality",
    requirement_id: str | None = None,
    contract_reference: str | None = None,
) -> ReviewFinding:
    """Build one finding whose only interesting property is what it derives from."""
    scoped = category in {"requirement", "contract"}
    return ReviewFinding(
        finding_id=finding_id,
        severity=severity,  # type: ignore[arg-type]
        title=f"Finding {finding_id}",
        description=f"Description of {finding_id}",
        recommendation=f"Fix {finding_id}",
        file_path="src/routes/status.js",
        line_number=2,
        repository_id="backend" if scoped else None,
        requirement_id=requirement_id,
        contract_reference=contract_reference,
        responsibility="implements" if scoped else None,
        validated_revision="rev-1" if scoped else None,
        evidence="The scoped source area was read." if scoped else None,
        finding_category=category,  # type: ignore[arg-type]
    )


def _review(
    *,
    verdict: str = "changes_requested",
    findings: list[ReviewFinding],
    deterministic: list[str] | None = None,
    manual_review_required: bool = False,
) -> ReviewArtifact:
    """Build a review carrying the platform metadata the runtime reads back."""
    return review_artifact().model_copy(
        update={
            "verdict": verdict,
            "findings": findings,
            "metadata": {
                "deterministic_finding_ids": deterministic or [],
                "manual_review_required": manual_review_required,
            },
        }
    )


# --------------------------------------------------------------------------------------
# 5.1 Traceability: what a blocking finding must name
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "finding"),
    [
        (
            "a scoped requirement",
            _finding("REQ-1", category="requirement", requirement_id="backend-status-api"),
        ),
        (
            "a contract section",
            _finding("CON-1", category="contract", contract_reference="paths./status.get"),
        ),
        (
            "a failed required validation command",
            _finding("validation-lint-npm", category="validation_failure"),
        ),
        (
            "a validation command that could not finish",
            _finding("VALIDATION_CAPACITY_TEST", category="validation_capacity"),
        ),
        (
            "an implementation expectation",
            _finding(
                "ROUTE_NOT_IMPLEMENTED",
                category="requirement",
                requirement_id="backend-status-api",
            ),
        ),
    ],
)
def test_a_finding_that_names_its_source_still_blocks(name: str, finding: ReviewFinding) -> None:
    """Each of the four sources a bounded review may block on, and both validation kinds."""
    outcome = _resolve_review_outcome(_review(findings=[finding]), bounded=True, publishable=True)

    assert outcome.blocking_issues == [finding.description], name
    assert outcome.advisory_findings == []
    assert outcome.advisory_verdict is None
    assert outcome.counts is not None
    assert outcome.counts.findings_untraceable == 0


@pytest.mark.parametrize("bounded", [False, True], ids=["unbounded", "bounded"])
def test_a_medium_beside_a_high_is_recorded_instead_of_dropped(bounded: bool) -> None:
    """AB-Feature-215's exact shape, in both modes, on the path the platform actually runs.

    `_blocking_findings` returns only the criticals and highs whenever any exist, which is the
    right work order. The medium then has to go somewhere: it is a defect the reviewer found
    and the platform is holding. Recorded on the unbounded path it was not -- that branch
    filled `advisory_findings` only when it was accepting a declining verdict, so on an
    ordinary rejection the finding vanished from the result entirely, and the next attempt was
    failed by a demand nobody had been shown.

    `bounded` is a measurement arm and this is not measurement: the same finding must survive
    in both.
    """
    blocker = _finding("AUTH-1", severity="high", category="requirement", requirement_id="FR-002")
    quieter = _finding(
        "TESTS-1", severity="medium", category="requirement", requirement_id="FR-002"
    )

    outcome = _resolve_review_outcome(
        _review(findings=[blocker, quieter]), bounded=bounded, publishable=True
    )

    assert outcome.blocking_issues == [blocker.description]
    assert outcome.advisory_findings == [quieter.description]


@pytest.mark.parametrize("bounded", [False, True], ids=["unbounded", "bounded"])
def test_a_finding_that_blocks_is_not_also_advisory(bounded: bool) -> None:
    """The two lists partition the review; a finding in both reads as two separate demands."""
    blocker = _finding("AUTH-1", severity="high", category="requirement", requirement_id="FR-002")

    outcome = _resolve_review_outcome(
        _review(findings=[blocker]), bounded=bounded, publishable=True
    )

    assert outcome.blocking_issues == [blocker.description]
    assert outcome.advisory_findings == []


def test_a_finding_that_names_nothing_is_advisory_and_recorded() -> None:
    """The production shape: a good senior-engineer critique of something nobody asked for."""
    unscoped = _finding("SCOPE-1")

    outcome = _resolve_review_outcome(_review(findings=[unscoped]), bounded=True, publishable=True)

    assert outcome.blocking_issues == []
    assert outcome.advisory_findings == [unscoped.description]
    # The verdict is preserved rather than rewritten: the reviewer said what it said.
    assert outcome.advisory_verdict == "changes_requested"
    assert outcome.counts is not None
    assert outcome.counts.findings_total == 1
    assert outcome.counts.findings_blocking == 0
    assert outcome.counts.findings_advisory == 1
    assert outcome.counts.findings_untraceable == 1


def test_one_traceable_finding_keeps_the_untraceable_ones_advisory() -> None:
    """Mixed reviews block on what was asked for and advise on the rest, in one attempt."""
    scoped = _finding("REQ-1", category="requirement", requirement_id="backend-status-api")
    unscoped = _finding("SCOPE-1")

    outcome = _resolve_review_outcome(
        _review(findings=[scoped, unscoped]), bounded=True, publishable=True
    )

    assert outcome.blocking_issues == [scoped.description]
    assert outcome.advisory_findings == [unscoped.description]
    assert outcome.advisory_verdict is None
    assert outcome.counts is not None
    assert outcome.counts.findings_untraceable == 1


# --------------------------------------------------------------------------------------
# 5.2 Severity
# --------------------------------------------------------------------------------------


def test_a_rejection_over_medium_findings_alone_still_blocks() -> None:
    """The existing promotion is preserved: a blocked attempt must carry its reason."""
    medium = _finding(
        "MED-1", severity="medium", category="requirement", requirement_id="backend-status-api"
    )

    outcome = _resolve_review_outcome(_review(findings=[medium]), bounded=True, publishable=True)

    assert outcome.blocking_issues == [medium.description]


def test_low_findings_are_never_promoted_to_blocking() -> None:
    """`low` is wording for the plan to correct, explicitly not a change to make.

    The unbounded rule's final fallback -- every finding, including `low` -- contradicted the
    reserved meaning of `low` two lines above it, and is gone.
    """
    low = _finding(
        "LOW-1", severity="low", category="requirement", requirement_id="backend-status-api"
    )

    outcome = _resolve_review_outcome(_review(findings=[low]), bounded=True, publishable=True)

    assert outcome.blocking_issues == []
    assert outcome.advisory_findings == [low.description]
    assert outcome.advisory_verdict == "changes_requested"


def test_critical_and_high_traceable_findings_are_unchanged() -> None:
    """Nothing about the ordinary rejection path moves."""
    critical = _finding(
        "CRIT-1",
        severity="critical",
        category="requirement",
        requirement_id="backend-status-api",
    )

    for bounded in (False, True):
        outcome = _resolve_review_outcome(
            _review(findings=[critical]), bounded=bounded, publishable=True
        )
        assert outcome.blocking_issues == [critical.description]


# --------------------------------------------------------------------------------------
# 5.3 Deterministic findings are unaffected
# --------------------------------------------------------------------------------------


def test_a_platform_injected_finding_blocks_even_though_it_names_no_requirement() -> None:
    """`REVIEW_EVIDENCE_INCOMPLETE` names no requirement and must never become advisory.

    It reports that the review could not read the source, which is not an opinion about the
    code and cannot be published for a person to judge instead.
    """
    evidence = _finding("REVIEW_EVIDENCE_INCOMPLETE")

    outcome = _resolve_review_outcome(
        _review(findings=[evidence], deterministic=["REVIEW_EVIDENCE_INCOMPLETE"]),
        bounded=True,
        publishable=True,
    )

    assert outcome.blocking_issues == [evidence.description]
    assert outcome.advisory_verdict is None


def test_a_review_that_asked_for_a_person_is_never_published_without_one() -> None:
    """`manual_review_required` is a decision no bounded scope may take from a person.

    The runtime branch that records `retry_refusal_reason` had no coverage anywhere when
    `32-` looked for it. It is fenced twice now: the evidence gate's own finding is
    platform-injected and therefore blocking, and acceptance refuses the flag outright. The
    finding here is untraceable and the flag alone is what stops publication.
    """
    outcome = _resolve_review_outcome(
        _review(verdict="rejected", findings=[_finding("SCOPE-1")], manual_review_required=True),
        bounded=True,
        publishable=True,
    )

    assert outcome.advisory_verdict is None
    assert outcome.blocking_issues == ["Description of SCOPE-1"]


def test_a_rejection_that_names_nothing_at_all_is_not_published_either() -> None:
    """ "A person judges this instead" is not an outcome for a review that judged nothing.

    The reviewer is asked again when it declines while raising no finding, and if it says
    the same thing the workstream keeps the behaviour it always had: it is not published,
    because there is nothing to publish for anyone to read.
    """
    outcome = _resolve_review_outcome(_review(findings=[]), bounded=True, publishable=True)

    assert outcome.advisory_verdict is None
    assert outcome.blocking_issues == []
    assert outcome.advisory_findings == []


def test_work_an_approved_result_may_not_carry_is_not_published_without_a_reason() -> None:
    """Nothing blocks and the change cannot be published: the record still says why.

    An attempt whose result carries no blocking issue at all is exactly what this task
    exists to stop producing, so the unbounded list is used to state the reason rather than
    leaving the record empty.
    """
    unscoped = _finding("SCOPE-1")

    outcome = _resolve_review_outcome(_review(findings=[unscoped]), bounded=True, publishable=False)

    assert outcome.advisory_verdict is None
    assert outcome.blocking_issues == [unscoped.description]


# --------------------------------------------------------------------------------------
# 5.5 The switch: off is the behaviour that already shipped
# --------------------------------------------------------------------------------------


def _unbounded_reference(review: ReviewArtifact) -> list[str]:
    """The rule as it stood before this task, quoted from `feature_runtime` at bd46f97."""
    blocking_issues = [
        item.description for item in review.findings if item.severity in {"critical", "high"}
    ]
    if review.verdict != "approved" and not blocking_issues:
        blocking_issues = [
            item.description for item in review.findings if item.severity != "low"
        ] or [item.description for item in review.findings]
    return blocking_issues


_SEVERITIES = ("critical", "high", "medium", "low", "info")
_SHAPES = (
    ("code_quality", None, None),
    ("requirement", "backend-status-api", None),
    ("contract", None, "paths./status.get"),
    ("validation_failure", None, None),
)


# Every combination the artifact schema permits. An approved review may not carry a critical
# or high finding at all -- `approved_reviews_cannot_contain_blocking_findings` rejects it --
# so those combinations are not generated rather than generated and skipped.
_EQUIVALENCE_CASES = [
    (verdict, severity, shape, deterministic)
    for verdict in ("approved", "changes_requested", "rejected")
    for severity in _SEVERITIES
    for shape in _SHAPES
    for deterministic in (False, True)
    if not (verdict == "approved" and severity in {"critical", "high"})
]


@pytest.mark.parametrize(("verdict", "severity", "shape", "deterministic"), _EQUIVALENCE_CASES)
def test_the_switch_off_reduces_exactly_to_the_rule_that_already_shipped(
    verdict: str,
    severity: str,
    shape: tuple[str, str | None, str | None],
    deterministic: bool,
) -> None:
    """Every combination of verdict, severity, traceability and injection, off.

    This is the property that makes the rollout safe. A behaviour change nobody can turn off
    is not reversible, and one whose off state is merely believed to be inert is not either.

    What the switch governs is **what blocks** -- `blocking_issues` below is quoted from the
    rule as it stood at bd46f97 and must reduce to it exactly. It does not govern whether the
    platform keeps a record of what the same review said and did not block on. That used to be
    the same question only by accident: the unbounded branch filled `advisory_findings` solely
    when accepting a declining verdict, so on an ordinary rejection a `medium` raised beside a
    `high` was dropped by the severity filter and then lost entirely, and AB-Feature-215's
    backend spent a retry being failed by a demand it had never been shown. Recording it is
    deliberately outside this arm, for the same reason 73-'s overrule demotion is: a defect the
    reviewer found and the platform is holding is not an experimental variable.
    """
    category, requirement_id, contract_reference = shape
    findings = [
        _finding(
            "F-1",
            severity=severity,
            category=category,
            requirement_id=requirement_id,
            contract_reference=contract_reference,
        )
    ]
    review = _review(
        verdict=verdict,
        findings=findings,
        deterministic=["F-1"] if deterministic else [],
    )

    outcome = _resolve_review_outcome(review, bounded=False, publishable=True)

    assert outcome.blocking_issues == _unbounded_reference(review)
    assert outcome.counts is None
    assert outcome.advisory_verdict is None
    # The record, not the gate: exactly the findings this outcome is not already carrying,
    # and never a finding that is also blocking.
    assert outcome.advisory_findings == [
        item.description
        for item in review.findings
        if item.description not in outcome.blocking_issues
    ]


def test_the_switch_off_records_no_advisory_state_on_an_empty_rejection() -> None:
    """A declining review with no findings at all keeps its empty issue list, either way."""
    review = _review(findings=[])

    off = _resolve_review_outcome(review, bounded=False, publishable=True)
    on = _resolve_review_outcome(review, bounded=True, publishable=True)

    assert off.blocking_issues == [] == _unbounded_reference(review)
    assert off.advisory_verdict is None
    assert on.blocking_issues == off.blocking_issues
    assert on.advisory_verdict is None


# --------------------------------------------------------------------------------------
# 5.4 Advisory findings survive, all the way to the pull request
# --------------------------------------------------------------------------------------


def _publishable_result(**overrides: Any) -> ChildWorkflowResultArtifact:
    """Build the approved child result a pull-request body is composed from."""
    payload: dict[str, Any] = {
        "feature_id": "feature-live",
        "parent_workflow_id": "feature-live",
        "child_workflow_id": "feature-live:backend",
        "repository_id": "backend",
        "workstream_id": "backend",
        "branch_name": "ai/feature-live/backend/server-status",
        "workspace_path": "/workspaces/feature-live/backend",
        "code_completion_artifact_id": "005_code_completion.backend.json",
        "review_artifact_id": "007_review.backend.json",
        "changed_files": [],
        "validation_results": [],
        "status": "approved",
        "blocking_issues": [],
        "pull_request_readiness": True,
        "contract_sections_consumed": [],
        "contract_sections_implemented": [],
    }
    payload.update(overrides)
    return create_artifact(
        ChildWorkflowResultArtifact,
        workflow_id="feature-live",
        artifact_id="011_child_workflow_result.backend.json",
        producer="child_workflow",
        payload=payload,
        metadata={},
    )


def _body(result: ChildWorkflowResultArtifact) -> str:
    """Compose the pull-request body the publisher writes for this result."""
    feature = _initial_feature_state(
        "feature-live", StartFeatureRequest.model_validate(_feature_payload())
    )
    child = ChildWorkflowReference(
        child_workflow_id="feature-live:backend",
        repository_id="backend",
        workstream_id="backend",
        branch_name=result.branch_name,
        workspace_path=result.workspace_path,
        status=ChildWorkflowStatus.APPROVED,
        retry_count=0,
    )
    return _pull_request_body(feature, child, result, _contract(), draft=True)


def test_an_advisory_finding_reaches_the_pull_request_body() -> None:
    """The substantive argument for the change: the observation is not discarded."""
    upsert = (
        "ensureRetryState performs findOneAndUpdate with upsert but does not catch a "
        "duplicate-key result and reread the winner"
    )

    body = _body(
        _publishable_result(
            advisory_findings=[upsert],
            advisory_review_verdict="changes_requested",
            review_finding_counts={
                "findings_total": 1,
                "findings_blocking": 0,
                "findings_advisory": 1,
                "findings_untraceable": 1,
            },
        )
    )

    assert f"- {upsert}" in body
    # And the body never claims an approval the reviewer did not give.
    assert "Repository review passed" not in body
    assert (
        "Validation summary: The repository review returned 'changes_requested', and every "
        "finding it raised named no scoped requirement, contract section, failed required "
        "validation command or implementation expectation, so each is listed below for "
        "review here rather than blocking the change; the feature's integration gate "
        "approved."
    ) in body


def test_a_pull_request_body_is_unchanged_when_nothing_was_advisory() -> None:
    """With the switch off no result carries advisory state, so no body moves."""
    body = _body(_publishable_result())

    assert "Validation summary: Repository review passed; the feature's integration gate" in body
    assert "Advisory review findings" not in body


# --------------------------------------------------------------------------------------
# The reviewer agent: which findings this platform put on the review itself
# --------------------------------------------------------------------------------------


class _SequencedLLMClient:
    """Return one scripted response per call, recording the instructions each time."""

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = list(payloads)
        self.calls: list[tuple[str, str]] = []

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
        """Answer with the next scripted payload, repeating the last one if exhausted."""
        self.calls.append((instructions, input_text))
        payload = self._payloads[min(len(self.calls) - 1, len(self._payloads) - 1)]
        return LLMResponse(
            response_id=f"mock-review-{len(self.calls)}",
            model="mock-reasoning-model",
            output_text=json.dumps(payload),
            input_tokens=10,
            output_tokens=5,
        )


def _reviewer_payload(*, verdict: str, findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a schema-valid reviewer response for the shared agent fixtures."""
    payload = domain_payload(review_artifact())
    payload["verdict"] = verdict
    payload["findings"] = findings
    return payload


async def _review_once(
    tmp_path: Path, client: _SequencedLLMClient, *, bounded: bool
) -> ReviewArtifact:
    """Run the reviewer agent over the shared fixtures with the switch in one position."""
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(file_updates={"src/service.py": "value = 1\n"}),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]
    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
        bounded_review_scope=bounded,
    ).run(agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion]))
    review: ReviewArtifact = update["artifacts"][0]
    return review


@pytest.mark.asyncio
async def test_the_reviewer_names_only_the_findings_the_platform_injected(
    tmp_path: Path,
) -> None:
    """The census is a difference, so a model cannot write itself onto it."""
    forged = {
        "finding_id": "validation-lint-npm",
        "severity": "high",
        "title": "A finding the model wrote under a platform id",
        "description": "The model supplied this finding itself.",
        "recommendation": "None.",
        "file_path": None,
        "line_number": None,
        "finding_category": "code_quality",
    }
    client = _SequencedLLMClient(
        [_reviewer_payload(verdict="changes_requested", findings=[forged])]
    )

    review = await _review_once(tmp_path, client, bounded=True)

    assert review.metadata["deterministic_finding_ids"] == []


@pytest.mark.asyncio
async def test_the_reviewer_records_the_census_with_the_switch_off(tmp_path: Path) -> None:
    """Off, the census is still recorded -- 73-'s measurement exemption reads it -- and
    nothing else moves: no verdict changes, no second model call is made."""
    client = _SequencedLLMClient([_reviewer_payload(verdict="approved", findings=[])])

    review = await _review_once(tmp_path, client, bounded=False)

    assert review.metadata["deterministic_finding_ids"] == []
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_a_rejection_whose_findings_are_all_low_is_asked_again(tmp_path: Path) -> None:
    """A declining verdict over `low` findings alone contradicts what `low` means."""
    low = {
        "finding_id": "LOW-1",
        "severity": "low",
        "title": "The requirement's wording does not match this runtime",
        "description": "The requirement asks for a lock this runtime does not have.",
        "recommendation": "Correct the requirement wording in the plan.",
        "file_path": None,
        "line_number": None,
        "finding_category": "code_quality",
    }
    high = {**low, "finding_id": "HIGH-1", "severity": "high", "requirement_id": "requirement-1"}
    client = _SequencedLLMClient(
        [
            _reviewer_payload(verdict="changes_requested", findings=[low]),
            _reviewer_payload(verdict="changes_requested", findings=[high]),
        ]
    )

    review = await _review_once(tmp_path, client, bounded=True)

    assert len(client.calls) == 2
    assert "every finding is severity 'low'" in client.calls[1][0]
    assert [item.finding_id for item in review.findings] == ["HIGH-1"]


@pytest.mark.asyncio
async def test_a_rejection_that_names_nothing_at_all_is_asked_again(tmp_path: Path) -> None:
    """A declining review with no findings names nothing for the next attempt to change."""
    client = _SequencedLLMClient([_reviewer_payload(verdict="changes_requested", findings=[])])

    review = await _review_once(tmp_path, client, bounded=True)

    assert len(client.calls) == 2
    assert "raised no findings at all" in client.calls[1][0]
    # Asking twice is the bound. The first review stands when the second says the same.
    assert review.verdict == "changes_requested"
    assert review.findings == []


@pytest.mark.asyncio
async def test_the_contradiction_repair_never_runs_with_the_switch_off(tmp_path: Path) -> None:
    """Off, a low-only rejection costs exactly the one model call it always did."""
    low = {
        "finding_id": "LOW-1",
        "severity": "low",
        "title": "Wording",
        "description": "Wording for the plan to correct.",
        "recommendation": "Correct the plan.",
        "file_path": None,
        "line_number": None,
        "finding_category": "code_quality",
    }
    client = _SequencedLLMClient([_reviewer_payload(verdict="changes_requested", findings=[low])])

    await _review_once(tmp_path, client, bounded=False)

    assert len(client.calls) == 1


# --------------------------------------------------------------------------------------
# The live child path: the effect on the worktree, the remote and the next attempt
# --------------------------------------------------------------------------------------


def _status_route_change() -> dict[str, Any]:
    """One material route change in the workstream's own scoped source area."""
    return {
        "summary": "Add the server status route.",
        "files": [
            {
                "path": "src/routes/status.js",
                "content": (
                    "function statusRoute(req, res) {\n"
                    "  res.json({ status: 'ok' });\n"
                    "}\n\nmodule.exports = { statusRoute };\n"
                ),
            },
            {
                "path": "test/status.test.js",
                "content": ("const { test } = require('node:test');\ntest('status', () => {});\n"),
            },
        ],
    }


def _unscoped_review() -> dict[str, Any]:
    """A good critique of something this workstream's requirement never asked for."""
    review = _review_payload(verdict="changes_requested")
    review["findings"] = [
        {
            "finding_id": "SCOPE-1",
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
    ]
    return review


@pytest.mark.asyncio
async def test_an_unscoped_rejection_still_spends_an_attempt_with_the_switch_off(
    tmp_path: Path,
) -> None:
    """The default behaviour, unchanged: the finding blocks and nothing is published.

    The switch is set explicitly rather than left at its default, so this stays a test of
    the off behaviour even when the whole suite is run with the environment overriding it.
    """
    harness = await _harness(tmp_path)
    harness["settings"] = harness["settings"].model_copy(update={"bounded_review_scope": False})
    review = _unscoped_review()

    execution = await _run(
        harness,
        engineer=ScriptedLLMClient([_status_route_change()]),
        reviewer=ScriptedLLMClient([review, review, review]),
    )

    assert execution.result.status == "failed"
    assert execution.result.blocking_issues == [review["findings"][0]["description"]]
    assert execution.result.advisory_findings == []
    assert execution.result.review_finding_counts is None
    assert execution.result.advisory_review_verdict is None
    assert execution.result.pull_request_readiness is False


@pytest.mark.asyncio
async def test_an_unscoped_rejection_is_published_for_a_person_with_the_switch_on(
    tmp_path: Path,
) -> None:
    """The behaviour change, as an effect: the branch exists on the remote and a human judges.

    This is the trade-off stated plainly. The reviewer's observation is correct and is kept;
    what changes is who answers it. It stops being a fourth rewrite of a file and becomes a
    line in the pull-request body of work that is otherwise complete.
    """
    harness = await _harness(tmp_path)
    harness["settings"] = harness["settings"].model_copy(update={"bounded_review_scope": True})
    review = _unscoped_review()

    execution = await _run(
        harness,
        engineer=ScriptedLLMClient([_status_route_change()]),
        reviewer=ScriptedLLMClient([review, review, review]),
    )

    result = execution.result
    assert result.status == "approved"
    assert result.blocking_issues == []
    assert result.advisory_findings == [review["findings"][0]["description"]]
    assert result.advisory_review_verdict == "changes_requested"
    assert result.pull_request_readiness is True
    assert result.review_finding_counts is not None
    assert result.review_finding_counts.findings_untraceable == 1
    # The commit is the effect, not the verdict: an accepted result must be publishable.
    assert execution.code_completion is not None
    assert execution.code_completion.commit_sha
    assert review["findings"][0]["description"] in _body(result)


@pytest.mark.asyncio
async def test_a_scoped_rejection_still_stops_the_attempt_with_the_switch_on(
    tmp_path: Path,
) -> None:
    """Narrowing scope must not narrow the requirement itself."""
    harness = await _harness(tmp_path)
    harness["settings"] = harness["settings"].model_copy(update={"bounded_review_scope": True})
    review = _unscoped_review()
    review["findings"][0]["requirement_id"] = "backend-status-api"
    review["findings"][0]["finding_category"] = "requirement"
    review["findings"][0]["repository_id"] = "backend"
    review["findings"][0]["responsibility"] = "implements"
    review["findings"][0]["validated_revision"] = "unvalidated"
    review["findings"][0]["evidence"] = "The scoped route does not reread the winner."

    execution = await _run(
        harness,
        engineer=ScriptedLLMClient([_status_route_change()]),
        reviewer=ScriptedLLMClient([review, review, review]),
    )

    assert execution.result.status == "failed"
    assert execution.result.blocking_issues == [review["findings"][0]["description"]]
    assert execution.result.retry_strategy is not None


# --------------------------------------------------------------------------------------
# A remediation review may not raise the bar the first round set
# --------------------------------------------------------------------------------------


def _remediation(
    *,
    prior: list[ReviewFinding],
    delta: set[str],
) -> _RemediationReviewScope:
    """Build the remediation scope the executor computes from the prior review and delta."""
    return _RemediationReviewScope(
        prior_finding_ids=frozenset(item.finding_id for item in prior),
        prior_fingerprints=frozenset(finding_fingerprint(item) for item in prior),
        delta_paths=frozenset(delta),
    )


def test_a_remediation_review_cannot_block_on_a_brand_new_demand() -> None:
    """Round N+1's fresh demand is recorded as advisory, and the work is publishable.

    AB-Feature-170's backend: round 4 blocked on five coverage demands none of which round
    2 raised, and each cost a 15-40-minute regeneration cycle. A bar that grows per round
    never converges, whatever the remediation does.
    """
    prior = [
        _finding("prior-1", category="requirement", requirement_id="backend-status-api"),
    ]
    fresh_demand = _finding(
        "new-coverage-demand",
        category="requirement",
        requirement_id="backend-status-api",
    ).model_copy(
        update={
            "file_path": "src/untouched/Elsewhere.js",
            "description": "Add multipart rejection coverage.",
        }
    )
    review = _review(verdict="changes_requested", findings=[fresh_demand])

    outcome = _resolve_review_outcome(
        review,
        bounded=True,
        publishable=True,
        remediation=_remediation(prior=prior, delta={"src/routes/status.js"}),
    )

    assert outcome.blocking_issues == []
    assert any("multipart" in item for item in outcome.advisory_findings)
    # Nothing blocks and the change is publishable, so the declining verdict is accepted
    # as advisory: the workstream does not fail over a demand no earlier round made.
    assert outcome.advisory_verdict == "changes_requested"


def test_an_unresolved_prior_finding_still_blocks_a_remediation() -> None:
    """The bar the first round set is exactly what a remediation is held to."""
    prior_finding = _finding("prior-1", category="requirement", requirement_id="backend-status-api")
    review = _review(verdict="changes_requested", findings=[prior_finding])

    outcome = _resolve_review_outcome(
        review,
        bounded=True,
        publishable=True,
        remediation=_remediation(prior=[prior_finding], delta=set()),
    )

    assert outcome.blocking_issues == [prior_finding.description]
    assert outcome.advisory_verdict is None


def test_a_reworded_repeat_of_a_prior_demand_still_blocks() -> None:
    """The fingerprint reduces a finding to the defect it names, so rewording cannot hide it."""
    prior_finding = _finding(
        "prior-1", category="requirement", requirement_id="backend-status-api"
    ).model_copy(
        update={"description": "TypeError: undefined is not iterable at src/pages/AllApps.js:117"}
    )
    reworded = prior_finding.model_copy(
        update={
            "finding_id": "different-id-this-round",
            "description": (
                "The listing page still crashes with a TypeError: undefined is not iterable "
                "(src/pages/AllApps.js:117)."
            ),
        }
    )
    review = _review(verdict="changes_requested", findings=[reworded])

    outcome = _resolve_review_outcome(
        review,
        bounded=True,
        publishable=True,
        remediation=_remediation(prior=[prior_finding], delta=set()),
    )

    assert outcome.blocking_issues, "a reworded unresolved prior demand must still block"


def test_a_regression_in_the_delta_still_blocks_a_remediation() -> None:
    """What the remediation itself changed is fully in scope, including what it broke."""
    regression = _finding(
        "regression-1", category="requirement", requirement_id="backend-status-api"
    ).model_copy(update={"file_path": "src/routes/status.js"})
    review = _review(verdict="changes_requested", findings=[regression])

    outcome = _resolve_review_outcome(
        review,
        bounded=True,
        publishable=True,
        remediation=_remediation(prior=[], delta={"src/routes/status.js"}),
    )

    assert outcome.blocking_issues == [regression.description]


def test_a_platform_injected_finding_always_blocks_a_remediation() -> None:
    """A failed required command ran against exactly this revision; it is never advisory."""
    injected = _finding("validation-test-npm", category="validation_failure")
    review = _review(
        verdict="changes_requested",
        findings=[injected],
        deterministic=["validation-test-npm"],
    )

    outcome = _resolve_review_outcome(
        review,
        bounded=True,
        publishable=True,
        remediation=_remediation(prior=[], delta=set()),
    )

    assert outcome.blocking_issues == [injected.description]


def test_the_first_review_keeps_its_full_blocking_authority() -> None:
    """Only a remediation is bounded; the round that sets the bar is not."""
    fresh_demand = _finding(
        "first-round-demand", category="requirement", requirement_id="backend-status-api"
    )
    review = _review(verdict="changes_requested", findings=[fresh_demand])

    outcome = _resolve_review_outcome(review, bounded=True, publishable=True, remediation=None)

    assert outcome.blocking_issues == [fresh_demand.description]


@pytest.mark.asyncio
async def test_a_remediation_reviews_new_demand_is_durably_advisory_end_to_end(
    tmp_path: Path,
) -> None:
    """Through the live executor: the demand is on the record, and the work still lands.

    The advisory has to survive into the result artifact -- that is what reaches the parent,
    the API and the pull-request body -- and the workstream has to reach publication rather
    than spending another 15-40-minute regeneration cycle answering a bar that grew.
    """
    from api.control_plane import RequestScopedCredentials
    from configs.settings import load_settings
    from services.cancellation import MockCancellationToken
    from services.feature_runtime import LiveChildWorkstreamExecutor
    from tests.test_live_child_executor import (
        RecordingProcessRunner,
        _live_run_context,
        _workstream,
    )

    harness = await _harness(tmp_path)
    harness["settings"] = load_settings(
        workspace_root=tmp_path / "workspaces", bounded_review_scope=True
    )
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Resolve the prior finding.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": (
                            "function statusRoute(req, res) {\n"
                            "  res.json({ status: 'ok', uptime: 1 });\n"
                            "}\n\nmodule.exports = { statusRoute };\n"
                        ),
                    },
                    {
                        "path": "test/status.remediation.test.js",
                        "content": (
                            "const { test } = require('node:test');\n"
                            "const assert = require('node:assert');\n"
                            "const { statusRoute } = require('../src/routes/status');\n\n"
                            "test('status route responds', () => {\n"
                            "  assert.ok(typeof statusRoute === 'function');\n"
                            "});\n"
                        ),
                    },
                ],
            }
        ]
    )
    new_demand = {
        "finding_id": "brand-new-coverage-demand",
        "severity": "high",
        "title": "Multipart rejection coverage is missing",
        "description": "Add multipart rejection coverage for the upload flow.",
        "recommendation": "Cover multipart rejections.",
        "file_path": "src/pages/Uploads.js",
        "line_number": None,
        "repository_id": "backend",
        "requirement_id": "backend-status-api",
        "contract_reference": None,
        "responsibility": "implements",
        "validated_revision": "rev-1",
        "evidence": "No multipart rejection test exists.",
        "recommended_fix": "Add the coverage.",
        "finding_category": "requirement",
    }
    reviewer = ScriptedLLMClient(
        [
            {
                **_review_payload(verdict="changes_requested"),
                "findings": [new_demand],
            }
        ]
    )
    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    prior_review = review_artifact().model_copy(
        update={
            "workflow_id": "feature-live:backend",
            "verdict": "changes_requested",
            "findings": [
                _finding(
                    "prior-1", category="requirement", requirement_id="backend-status-api"
                ).model_copy(update={"description": "The status payload lacks the uptime field."})
            ],
        }
    )
    feature.artifacts.append(prior_review)
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=RecordingProcessRunner(),
    )
    try:
        execution = await executor.run(
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

    # The reviewer was told which round this is and what the bar already contains.
    review_instructions = reviewer.calls[0][0]
    assert "blocking authority" in review_instructions
    assert "prior-1" in review_instructions
    # The workstream did not fail over the new demand; the demand is durably advisory.
    assert execution.result.status == "approved"
    assert execution.result.advisory_review_verdict == "changes_requested"
    assert any("multipart" in item.lower() for item in execution.result.advisory_findings), (
        "the demand must stay visible on the durable result"
    )
    assert execution.result.blocking_issues == []


@pytest.mark.asyncio
async def test_the_first_review_is_told_it_sets_the_whole_bar(tmp_path: Path) -> None:
    """Coverage expectations are to be stated completely in round one, and the prompt says so."""
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Implement the route.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": (
                            "function statusRoute(req, res) {\n"
                            "  res.json({ status: 'ok' });\n"
                            "}\n\nmodule.exports = { statusRoute };\n"
                        ),
                    },
                    {
                        "path": "test/status.first.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('x', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    await _run(harness, engineer=engineer, reviewer=reviewer)

    review_instructions = reviewer.calls[0][0]
    assert "sets the whole bar" in review_instructions
    assert "blocking authority" not in review_instructions
