"""The step decision: what a feature owes next, read from its persisted state alone.

Every case here is table-driven and free of I/O, which is the property that makes the
decision cheap enough to ask on every claim. The states are not hand-written dictionaries:
one deterministic feature is run end to end with a checkpoint writer recording every durable
snapshot it produced, and the table asserts against those. A fixture invented by hand agrees
with whatever the person writing it believed; a captured checkpoint is what the platform
actually persists.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from agents.shared.contracts import create_artifact
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import (
    ChildWorkflowResultArtifact,
    ClarificationQuestion,
    ContractChangeRequestArtifact,
    IntegrationContractArtifact,
    IntegrationReviewArtifact,
    RepositoryExecutionPlanArtifact,
    TechnicalPRDArtifact,
)
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.external_operations import WorkflowCheckpointBoundary
from state.feature_models import FeatureWorkflowSnapshot
from tests.test_feature_api import feature_payload
from workflows.feature_workflow import (
    FeatureStep,
    FeatureWorkflowOrchestrator,
    NoStep,
    StepDecision,
    next_step,
)

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)


def _fresh(feature_id: str = "feature-steps") -> FeatureWorkflowSnapshot:
    """Return the state a feature has the moment it is accepted, before anything runs."""
    return _initial_feature_state(feature_id, StartFeatureRequest.model_validate(feature_payload()))


async def _captured_run() -> tuple[FeatureWorkflowSnapshot, list[FeatureWorkflowSnapshot]]:
    """Run one deterministic feature to completion, keeping every durable snapshot it wrote."""
    captured: list[FeatureWorkflowSnapshot] = []

    async def _record(
        state: FeatureWorkflowSnapshot,
        boundary: WorkflowCheckpointBoundary,
        repository_id: str | None,
    ) -> None:
        del boundary, repository_id
        captured.append(state)

    orchestrator = FeatureWorkflowOrchestrator(checkpoint_writer=_record)
    final = await orchestrator.start(_fresh(), credentials=CREDENTIALS)
    return final, captured


def _rewound(
    state: FeatureWorkflowSnapshot,
    *,
    without: type[Any] | None = None,
    status: FeatureWorkflowStatus = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
) -> FeatureWorkflowSnapshot:
    """Return a completed run's state as it looked before one artifact type was written.

    The status moves back with the artifacts. A completed feature is terminal whatever else
    it holds, so leaving `completed` in place would make every one of these cases assert the
    terminal answer instead of the one being examined.
    """
    rewound = state.model_copy(deep=True)
    rewound.status = status
    if without is not None:
        rewound.artifacts = [
            artifact for artifact in rewound.artifacts if not isinstance(artifact, without)
        ]
    return rewound


@pytest.mark.asyncio
async def test_a_feature_with_no_technical_prd_analyzes_its_prd_first() -> None:
    """The first step is the only one whose evidence is the absence of every artifact."""
    decision = next_step(_fresh())

    assert decision.step is FeatureStep.ANALYZE_PRD
    assert decision.no_step_reason is None


@pytest.mark.asyncio
async def test_an_unanswered_question_is_a_reason_not_to_step_rather_than_a_step() -> None:
    """A feature resting on a person must not be advanced by the platform on its own."""
    final, _ = await _captured_run()
    asking = _rewound(final)
    technical_prd = _latest(asking, TechnicalPRDArtifact)
    asking.artifacts = [
        technical_prd.model_copy(
            update={
                "unresolved_questions": [
                    ClarificationQuestion(
                        question_id="premise-1",
                        question="Does this repository already have an audit log?",
                        rationale="The requirement assumes one exists.",
                        required=True,
                    )
                ]
            }
        )
        if artifact.artifact_id == technical_prd.artifact_id
        else artifact
        for artifact in asking.artifacts
    ]
    asking.status = FeatureWorkflowStatus.WAITING_FOR_HUMAN

    decision = next_step(asking)

    assert decision.step is None
    assert decision.no_step_reason is NoStep.AWAITING_CLARIFICATION_ANSWERS


@pytest.mark.asyncio
async def test_a_question_nobody_has_been_asked_yet_is_planned_around_not_waited_on() -> None:
    """The product manager's own questions must not stop a feature before the checkouts run.

    Planning is what reads every repository, answers the questions the checkouts settle and
    raises the ones only a repository can. A feature that stopped on the product manager's
    questions alone asked its author everything except those.
    """
    final, _ = await _captured_run()
    asking = _rewound(
        final,
        without=IntegrationContractArtifact,
        status=FeatureWorkflowStatus.ANALYZING_PRD,
    )
    technical_prd = _latest(asking, TechnicalPRDArtifact)
    asking.artifacts = [
        technical_prd.model_copy(
            update={
                "unresolved_questions": [
                    ClarificationQuestion(
                        question_id="premise-1",
                        question="Does this repository already have an audit log?",
                        rationale="The requirement assumes one exists.",
                        required=True,
                    )
                ]
            }
        )
        if artifact.artifact_id == technical_prd.artifact_id
        else artifact
        for artifact in asking.artifacts
    ]

    decision = next_step(asking)

    assert decision.step is FeatureStep.PLAN


@pytest.mark.asyncio
async def test_an_analyzed_feature_with_no_contract_plans_next() -> None:
    """Reconnaissance and planning are one step, so the evidence for both is the contract."""
    final, _ = await _captured_run()

    assert next_step(_rewound(final, without=IntegrationContractArtifact)).step is FeatureStep.PLAN
    assert (
        next_step(_rewound(final, without=RepositoryExecutionPlanArtifact)).step is FeatureStep.PLAN
    )


@pytest.mark.asyncio
async def test_a_planned_feature_with_pending_children_executes_workstreams() -> None:
    """A child that has not settled is work an attempt could still do.

    Rewound past the child results as well as the statuses. A repository whose attempt has
    already written its immutable result has spent that attempt whatever its row says, and
    scheduling it again would collide with the artifact it produced -- so a state that claims
    to be pending while holding attempt 0's results is not a state this platform can reach.
    """
    final, _ = await _captured_run()
    pending = _rewound(
        final,
        without=ChildWorkflowResultArtifact,
        status=FeatureWorkflowStatus.CONTRACT_READY,
    )
    for repository_id, child in pending.child_workflows.items():
        pending.child_workflows[repository_id] = child.model_copy(
            update={"status": ChildWorkflowStatus.PENDING}
        )

    decision = next_step(pending)

    assert decision.step is FeatureStep.EXECUTE_WORKSTREAMS
    assert "2 repository workstream(s)" in decision.explanation


@pytest.mark.asyncio
async def test_settled_children_with_no_verdict_go_to_integration_review() -> None:
    """Every repository has finished, so what is left is reviewing them together."""
    final, _ = await _captured_run()
    unreviewed = _rewound(final, without=IntegrationReviewArtifact)

    assert next_step(unreviewed).step is FeatureStep.INTEGRATION_REVIEW


@pytest.mark.asyncio
async def test_an_approved_verdict_with_an_unpublished_repository_publishes() -> None:
    """The publication step is owed until every approved repository is open for review."""
    final, _ = await _captured_run()
    unpublished = final.model_copy(deep=True)
    unpublished.status = FeatureWorkflowStatus.READY_FOR_PULL_REQUESTS
    unpublished.child_workflows["frontend"] = unpublished.child_workflows["frontend"].model_copy(
        update={"pull_request_artifact_id": None}
    )

    decision = next_step(unpublished)

    assert decision.step is FeatureStep.PUBLISH


@pytest.mark.asyncio
async def test_a_completed_feature_owes_nothing() -> None:
    """The end of a run is a reason, not an empty answer."""
    final, _ = await _captured_run()

    assert final.status is FeatureWorkflowStatus.COMPLETED
    decision = next_step(final)

    assert decision.step is None
    assert decision.no_step_reason is NoStep.FEATURE_IS_TERMINAL


@pytest.mark.asyncio
async def test_every_step_done_is_distinguishable_from_a_terminal_feature() -> None:
    """A feature whose work is finished but whose status has not caught up says so."""
    final, _ = await _captured_run()
    settled = final.model_copy(deep=True)
    settled.status = FeatureWorkflowStatus.CREATING_PULL_REQUESTS

    decision = next_step(settled)

    assert decision.step is None
    assert decision.no_step_reason is NoStep.EVERY_STEP_IS_DONE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        FeatureWorkflowStatus.CANCELLED,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    ],
)
async def test_a_cancelled_feature_is_terminal(status: FeatureWorkflowStatus) -> None:
    """Neither cancelled status is ever advanced, whatever its artifacts say."""
    cancelled = _fresh().model_copy(update={"status": status})

    assert next_step(cancelled).no_step_reason is NoStep.FEATURE_IS_TERMINAL


@pytest.mark.asyncio
async def test_a_cancellation_in_progress_schedules_no_further_step() -> None:
    """A cancellation records its own cleanup; ordinary work on top would discard it."""
    cancelling = _fresh().model_copy(
        update={"status": FeatureWorkflowStatus.CANCELLING, "cancellation_requested": True}
    )

    assert next_step(cancelling).no_step_reason is NoStep.CANCELLATION_IN_PROGRESS

    # The request alone is enough: a feature can be asked to stop from any running status.
    requested = _fresh().model_copy(
        update={
            "status": FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
            "cancellation_requested": True,
        }
    )

    assert next_step(requested).no_step_reason is NoStep.CANCELLATION_IN_PROGRESS


@pytest.mark.asyncio
async def test_a_pending_contract_change_request_stops_every_step() -> None:
    """The contract is an input to every remaining step, so none may run while it is disputed."""
    final, _ = await _captured_run()
    disputed = _rewound(final, without=IntegrationReviewArtifact)
    disputed.artifacts.append(_contract_change_request(disputed, status="pending"))

    assert next_step(disputed).no_step_reason is NoStep.AWAITING_CONTRACT_DECISION

    # The newest revision of the request is what counts: a decision unblocks the feature.
    resolved = _rewound(final, without=IntegrationReviewArtifact)
    resolved.artifacts.append(_contract_change_request(resolved, status="pending"))
    resolved.artifacts.append(_contract_change_request(resolved, status="approved"))

    assert next_step(resolved).step is FeatureStep.INTEGRATION_REVIEW


@pytest.mark.asyncio
async def test_a_failed_feature_names_the_step_a_resume_would_run() -> None:
    """Failure is a stop, not an end.

    The two failed statuses are in `TERMINAL_FEATURE_STATUSES` because this platform will not
    act on them unasked -- which is the queue's rule, enforced by `_CONTINUABLE_STATUSES`, not
    a claim that the work is over. `_recover_from_checkpoint` has always resumed both, and it
    now asks this function what to resume. Returning "terminal" here would make every resume
    of a failed feature a no-op.
    """
    final, _ = await _captured_run()
    stopped = _rewound(
        final,
        without=IntegrationReviewArtifact,
        status=FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
    )

    assert next_step(stopped).step is FeatureStep.INTEGRATION_REVIEW


@pytest.mark.asyncio
async def test_a_workstream_whose_retry_decision_is_terminal_is_not_rescheduled() -> None:
    """A repository that stopped for a recorded reason is not work another attempt could do."""
    final, _ = await _captured_run()
    refused = final.model_copy(deep=True)
    refused.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
    for repository_id, child in refused.child_workflows.items():
        refused.child_workflows[repository_id] = child.model_copy(
            update={
                "status": ChildWorkflowStatus.FAILED,
                "retry_refusal_reason": "This workstream stopped converging.",
            }
        )

    decision = next_step(refused)

    assert decision.step is not FeatureStep.EXECUTE_WORKSTREAMS


@pytest.mark.asyncio
async def test_the_decision_is_total_for_every_status_the_platform_can_persist() -> None:
    """Totality, asserted rather than asserted about: every status yields a usable answer."""
    final, _ = await _captured_run()
    for status in FeatureWorkflowStatus:
        for state in (_fresh(), final):
            decision = next_step(state.model_copy(update={"status": status}, deep=True))

            assert (decision.step is None) != (decision.no_step_reason is None)
            assert decision.explanation.strip()


@pytest.mark.asyncio
async def test_the_decision_is_total_for_every_checkpoint_a_real_run_persisted() -> None:
    """Each durable boundary a feature actually wrote is a state a claim can land on."""
    _, captured = await _captured_run()

    assert captured, "the deterministic run wrote no checkpoints"
    for state in captured:
        decision = next_step(state)

        assert (decision.step is None) != (decision.no_step_reason is None)


@pytest.mark.asyncio
async def test_the_decision_is_pure() -> None:
    """Asking twice gives the same answer and leaves the state it read untouched."""
    final, captured = await _captured_run()
    for state in [final, *captured]:
        before = state.model_dump(mode="json")

        first = next_step(state)
        second = next_step(state)

        assert first == second
        assert state.model_dump(mode="json") == before


def test_a_decision_must_name_exactly_one_of_a_step_and_a_reason() -> None:
    """The dataclass refuses the ambiguity that would make the caller guess."""
    with pytest.raises(ValueError, match="exactly one"):
        StepDecision(
            step=FeatureStep.PLAN,
            no_step_reason=NoStep.EVERY_STEP_IS_DONE,
            explanation="",
        )
    with pytest.raises(ValueError, match="exactly one"):
        StepDecision(step=None, no_step_reason=None, explanation="")


def test_every_no_step_reason_is_reachable() -> None:
    """A reason nothing produces is a reason nobody can act on."""
    assert {reason for reason in NoStep} == {
        NoStep.FEATURE_IS_TERMINAL,
        NoStep.CANCELLATION_IN_PROGRESS,
        NoStep.AWAITING_CLARIFICATION_ANSWERS,
        NoStep.AWAITING_CONTRACT_DECISION,
        NoStep.EVERY_STEP_IS_DONE,
    }
    assert dataclasses.is_dataclass(StepDecision)


def _has(state: FeatureWorkflowSnapshot, artifact_class: type[Any]) -> bool:
    """Report whether this state holds any artifact of one type."""
    return any(isinstance(artifact, artifact_class) for artifact in state.artifacts)


def _latest(state: FeatureWorkflowSnapshot, artifact_class: type[Any]) -> Any:
    """Return the newest artifact of one type, which the caller has already established exists."""
    return next(
        artifact for artifact in reversed(state.artifacts) if isinstance(artifact, artifact_class)
    )


def _contract_change_request(
    state: FeatureWorkflowSnapshot, *, status: str
) -> ContractChangeRequestArtifact:
    """Build one child-owned request against this feature's current contract."""
    contract = _latest(state, IntegrationContractArtifact)
    return create_artifact(
        ContractChangeRequestArtifact,
        workflow_id=state.feature_id,
        artifact_id=f"contract_change_request.{status}.json",
        producer="child_workflow",
        payload={
            "change_request_id": "change-1",
            "feature_id": state.feature_id,
            "current_contract_version": contract.contract_version,
            "requested_by_repository_id": "backend",
            "requested_changes": ["Add a nullable field to the audit event."],
            "reason": "The contract cannot express the audit payload.",
            "affected_workstreams": ["workstream-backend"],
            "compatibility_impact": "Additive; existing consumers are unaffected.",
            "migration_requirements": [],
            "status": status,
            "resolution": None,
            "new_contract_artifact_id": None,
        },
        metadata={"source_artifact_ids": [contract.artifact_id]},
    )
