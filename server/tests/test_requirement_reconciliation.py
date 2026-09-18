"""The answers rewrite the requirement text, or the feature stops saying so.

AB-Feature-200 is the run these tests replay. A person answered that the target database is a
standalone Mongo where `session.withTransaction` fails at runtime, and that bulk creation must
compensate with application-level deletes. The answer reached the PRD's metadata and the
execution plan; FR-006 went on depending on "Transactional app persistence" and demanding that
"a failure rolls back all app records created by that request", with "repository tests cover
rollback behavior". The engineer, handed both, built transaction wrappers *and* a compensating
`deleteMany`; the self-review correctly proved the hybrid incoherent on four consecutive
attempts whose blocking diagnostics never changed, and the no-meaningful-change guard stopped
the workstream with nothing to show.

Every test here drives the real orchestrator with a scripted model, exactly as the planner
suites do: the reconciliation call is a real `ProductManagerAgent` call against a client that
returns prepared text, so the prompt, the parse, the structural refusal and the one bounded
repair are all in the path rather than mocked past.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from adapters.llm_adapter import ImageInput, LLMAdapterError, LLMResponse
from agents.planner.feature_planner import (
    DeterministicFeaturePlanner,
    _ensure_plan_covers_technical_requirements,
)
from agents.product_manager.agent import ProductManagerAgent
from agents.reviewer.agent import _review_scope
from agents.shared.contracts import ARTIFACT_FILENAMES, AgentArtifactError, create_artifact
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from api.schemas import ClarificationAnswer
from artifacts.schemas import (
    ClarificationQuestion,
    IntegrationContractArtifact,
    RepositoryExecutionPlanArtifact,
    TechnicalPRDArtifact,
)
from prompts.prompt_loader import PromptLoader
from services.cancellation import MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.feature_runtime import _child_task_plan
from state.enums import FeatureWorkflowStatus
from state.external_operations import ExternalOperationType
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from tools.requirement_reconciliation import (
    AnsweredClarification,
    ReconciliationRejected,
    parse_reconciliation,
)
from workflows import feature_workflow as feature_workflow_module
from workflows.feature_workflow import (
    _ALLOWED_FEATURE_STAGE_INFRASTRUCTURE_FAULTS,
    FeatureWorkflowOrchestrator,
)

pytestmark = pytest.mark.asyncio

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)

# The answer that settled 200, quoted as a person would have written it.
COMPENSATION_ANSWER = (
    "The cluster is a standalone Mongo, so session.withTransaction fails at runtime. Use "
    "application-level compensating deletes instead of transactions."
)
TRANSACTION_QUESTION_ID = "recon-backend-1"


def _feature_payload() -> dict[str, object]:
    """One repository, so the assertions are about requirement text and not about fan-out."""
    return {
        "feature_id": "feature-reconcile",
        "prd": {
            "title": "Bulk app upload",
            "problem_statement": "Operators upload many apps from one CSV.",
            "goals": ["Create many apps in one request."],
            "user_stories": [],
            "requirements": [],
            "constraints": [],
            "out_of_scope": [],
            "stakeholders": ["Ad operations"],
        },
        "repositories": [
            {
                "repository_id": "backend",
                "name": "Backend",
                "role": "backend",
                "repository_url": "https://github.com/example/backend.git",
                "default_branch": "main",
            }
        ],
    }


def _technical_prd(feature_id: str) -> TechnicalPRDArtifact:
    """200's draft PRD: one requirement the answer overrules, two it does not touch."""
    return create_artifact(
        TechnicalPRDArtifact,
        workflow_id=feature_id,
        artifact_id=ARTIFACT_FILENAMES["technical_prd"],
        producer="product_manager",
        payload={
            "title": "Bulk app upload technical plan",
            "solution_summary": "Accept a CSV of apps and create each one.",
            "functional_requirements": [
                {
                    "requirement_id": "FR-005",
                    "description": "The upload endpoint validates every CSV row before "
                    "creating anything.",
                    "priority": "must",
                    "acceptance_criteria": [
                        "A malformed row is reported with its line number and nothing is created."
                    ],
                    "dependencies": ["CSV parsing"],
                },
                {
                    "requirement_id": "FR-006",
                    "description": "Bulk app creation is atomic: either every app in the "
                    "request is created or none is.",
                    "priority": "must",
                    "acceptance_criteria": [
                        "A failure part-way through a request rolls back all app records "
                        "created by that request.",
                        "Repository tests cover rollback behavior.",
                    ],
                    "dependencies": ["Transactional app persistence"],
                },
            ],
            "non_functional_requirements": [
                {
                    "requirement_id": "NFR-001",
                    "description": "The endpoint reports progress for large uploads.",
                    "priority": "should",
                    "acceptance_criteria": ["The response names how many rows were accepted."],
                    "dependencies": [],
                }
            ],
            "data_requirements": ["App documents are stored in Mongo."],
            "integration_requirements": ["Use the approved integration contract."],
            "security_requirements": ["Do not persist provider credentials."],
            "assumptions": ["The CSV is uploaded whole."],
            "unresolved_questions": [
                {
                    "question_id": TRANSACTION_QUESTION_ID,
                    "question": "Does the target cluster support multi-document transactions?",
                    "rationale": "Atomic bulk creation needs them, and the checkout does not "
                    "say which deployment it runs against.",
                    "required": True,
                }
            ],
        },
        metadata={"source_artifact_ids": ["001_prd.json"]},
    )


def _reconciled_payload(driver: str = TRANSACTION_QUESTION_ID) -> dict[str, Any]:
    """What a reconciliation returns for 200: FR-006 restated, everything else untouched.

    The driver is a parameter because a later round's accepted answers are that round's own:
    `apply_clarification_answers` records the answers of the round it applies, so a rewrite
    can only ever cite an answer that round actually carried.
    """
    return {
        "requirement_ids": ["FR-005", "FR-006", "NFR-001"],
        "changed_requirements": [
            {
                "requirement_id": "FR-006",
                "driving_answer_ids": [driver],
                "description": (
                    "Bulk app creation is compensated rather than transactional: a failed "
                    "request removes the app documents it created."
                ),
                "dependencies": ["Compensating cleanup of created app documents"],
                "acceptance_criteria": [
                    "A failure part-way through a request deletes the app documents that "
                    "request created (compensating cleanup).",
                    "Objects already created in the external ad server may remain and are "
                    "reported in the response as external residue.",
                    "Repository tests cover the compensation path and the reported external "
                    "residue.",
                ],
            }
        ],
        "unsettled_contradictions": [],
    }


RESIDUE_QUESTION_ID = "recon-backend-2"
# What `contradiction_questions` names the conflict between those two answers over FR-006.
# Stable by construction: the same contradiction reported again produces the same id, which
# is what lets the already-asked filter recognise it.
RECONCILE_QUESTION_ID = "reconcile-fr-006-recon-backend-1-recon-backend-2"
RESIDUE_ANSWER = "Keep every app the request already created and report the failed rows."


def _contradicting_payload() -> dict[str, Any]:
    """What a reconciliation returns when two answers cannot both be written down."""
    return {
        "requirement_ids": ["FR-005", "FR-006", "NFR-001"],
        "changed_requirements": [],
        "unsettled_contradictions": [
            {
                "answer_ids": [TRANSACTION_QUESTION_ID, RESIDUE_QUESTION_ID],
                "requirement_id": "FR-006",
                "conflict": (
                    "One answer removes what a failed request created and the other keeps it."
                ),
            }
        ],
    }


def _confirming_payload() -> dict[str, Any]:
    """What a reconciliation returns when the answers contradict nothing."""
    return {
        "requirement_ids": ["FR-005", "FR-006", "NFR-001"],
        "changed_requirements": [],
        "unsettled_contradictions": [],
    }


class _ScriptedClient:
    """Answer like the real adapters do, from a queue of prepared responses.

    The emptiness check is the adapters' own: both refuse an empty `instructions` or
    `input_text` before any request is made, which is how the grounding call once failed in
    zero seconds on six consecutive runs while its fallback quietly asked a human instead.
    """

    def __init__(self, *responses: str | Exception) -> None:
        self._responses = list(responses)
        self.instructions: list[str] = []
        self.inputs: list[str] = []

    @property
    def calls(self) -> int:
        """How many times a provider was actually reached."""
        return len(self.inputs)

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
        """Return the next prepared response, or raise the next prepared failure."""
        if not instructions.strip() or not input_text.strip():
            msg = "instructions and input_text must not be empty"
            raise LLMAdapterError(msg, diagnostics=(msg,), failure_classification="empty_request")
        self.instructions.append(instructions)
        self.inputs.append(input_text)
        nxt = self._responses[0] if len(self._responses) == 1 else self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return LLMResponse(
            response_id=f"resp-reconcile-{len(self.inputs)}",
            model="stub-reasoning",
            output_text=nxt,
            input_tokens=None,
            output_tokens=None,
        )


class _ScriptedProductManager:
    """200's draft PRD, and a real reconciliation call against a scripted model."""

    def __init__(self, *responses: str | Exception) -> None:
        self.client = _ScriptedClient(*responses)
        self._agent = ProductManagerAgent(
            prompt_loader=PromptLoader(), llm_client=cast(Any, self.client)
        )

    async def create_technical_prd(
        self, *, feature_id: str, prd: Any, design_snapshot: Any = None
    ) -> TechnicalPRDArtifact:
        """Return the draft PRD without reaching a provider: this suite is about the answers."""
        del prd
        return _technical_prd(feature_id)

    async def reconcile_requirements(self, **kwargs: Any) -> Any:
        """Run the agent's own reconciliation, prompt, validation, repair and all."""
        return await self._agent.reconcile_requirements(
            workflow_id=kwargs["feature_id"],
            technical_prd=kwargs["technical_prd"],
            answers=kwargs["answers"],
        )


class _CriteriaCopyingPlanner:
    """Plan deterministically, but scope the criteria a live planner would: the PRD's own.

    The deterministic planner writes one canned workstream criterion, which is fine for every
    suite that is not about requirement text. This one copies the requirement criteria into
    the workstream the way a real planner does, because that copy is the link `_review_scope`
    reads -- and it records the PRD it was handed, which is the single input every downstream
    judge derives from.
    """

    def __init__(self) -> None:
        self._delegate = DeterministicFeaturePlanner()
        self.received: list[TechnicalPRDArtifact] = []

    async def plan(self, **kwargs: Any) -> Any:
        """Record the technical PRD, then plan with the criteria it actually carries."""
        technical_prd = kwargs["technical_prd"]
        self.received.append(technical_prd)
        architecture, contract, plan = await self._delegate.plan(**kwargs)
        criteria = [
            criterion
            for requirement in technical_prd.functional_requirements
            for criterion in requirement.acceptance_criteria
        ]
        return (
            architecture,
            contract,
            plan.model_copy(
                update={
                    "workstreams": [
                        workstream.model_copy(update={"acceptance_criteria": criteria})
                        for workstream in plan.workstreams
                    ]
                }
            ),
        )


def _newest_prd(state: Any) -> TechnicalPRDArtifact:
    """The revision every PRD reader sees: newest wins."""
    return next(
        artifact
        for artifact in reversed(state.artifacts)
        if isinstance(artifact, TechnicalPRDArtifact)
    )


def _prd_revisions(state: Any) -> list[TechnicalPRDArtifact]:
    """Every technical-PRD revision this feature holds, oldest first."""
    return [artifact for artifact in state.artifacts if isinstance(artifact, TechnicalPRDArtifact)]


def _requirement(technical_prd: TechnicalPRDArtifact, requirement_id: str) -> Any:
    """One requirement by id, from either list."""
    return next(
        item
        for item in [
            *technical_prd.functional_requirements,
            *technical_prd.non_functional_requirements,
        ]
        if item.requirement_id == requirement_id
    )


async def _waiting_feature(
    product_manager: Any, *, planner: Any = None, feature_id: str = "feature-reconcile"
) -> tuple[FeatureWorkflowOrchestrator, Any]:
    """Drive a fresh feature to the clarification gate, where a person answers."""
    request = StartFeatureRequest.model_validate(_feature_payload())
    state = _initial_feature_state(feature_id, request)
    orchestrator = FeatureWorkflowOrchestrator(
        product_manager=cast(Any, product_manager),
        planner=cast(Any, planner) if planner is not None else None,
    )
    waiting = await orchestrator.start(state, credentials=CREDENTIALS)
    assert waiting.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
    return orchestrator, waiting


def _answers() -> list[ClarificationAnswer]:
    """The one answer 200's operator gave."""
    return [ClarificationAnswer(question_id=TRANSACTION_QUESTION_ID, answer=COMPENSATION_ANSWER)]


# --- A1 -------------------------------------------------------------------------------------


async def test_the_answer_rewrites_the_requirement_it_contradicts() -> None:
    """200's replay: the settled decision reaches the requirement text, and only that text."""
    product_manager = _ScriptedProductManager(json.dumps(_reconciled_payload()))
    orchestrator, waiting = await _waiting_feature(product_manager)
    before = _newest_prd(waiting)

    result = await orchestrator.resume(waiting, answers=_answers(), credentials=CREDENTIALS)

    reconciled = _newest_prd(result)
    requirement = _requirement(reconciled, "FR-006")
    # The dependency the answer overruled is gone, and the mechanism it chose is named.
    assert not any("Transactional" in item for item in requirement.dependencies)
    assert requirement.dependencies == ["Compensating cleanup of created app documents"]
    criteria = " ".join(requirement.acceptance_criteria)
    assert "compensating cleanup" in criteria
    assert "external residue" in criteria
    assert "rolls back all app records" not in criteria
    assert "rollback behavior" not in criteria
    # The change list names exactly that requirement and exactly the answer that drove it.
    assert reconciled.metadata["requirement_reconciliation"] == {
        "FR-006": [TRANSACTION_QUESTION_ID]
    }
    assert reconciled.metadata["reconciled_answer_ids"] == [TRANSACTION_QUESTION_ID]
    # Every requirement no answer addressed is byte-identical, and identity is unchanged.
    for requirement_id in ("FR-005", "NFR-001"):
        assert _requirement(reconciled, requirement_id) == _requirement(before, requirement_id)
    assert [item.requirement_id for item in reconciled.functional_requirements] == [
        "FR-005",
        "FR-006",
    ]
    assert _requirement(reconciled, "FR-006").priority == "must"
    # The human's words remain the authority, verbatim, beside the text they drove.
    assert reconciled.metadata["clarification_answers"] == {
        TRANSACTION_QUESTION_ID: COMPENSATION_ANSWER
    }
    # Revisions stay flattened: `.revision-2.revision-3.json` is what broke every child.
    assert reconciled.artifact_id == "002_technical_prd.revision-3.json"
    assert result.status is FeatureWorkflowStatus.COMPLETED


async def test_the_reconciliation_call_is_journaled_as_a_planning_call(tmp_path: Path) -> None:
    """One row, with its own logical step, on the wrapper every pre-coding call goes through."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'reconcile-journal.db'}")
    await database.create_schema()
    try:
        journal = ExternalOperationJournal(database)
        product_manager = _ScriptedProductManager(json.dumps(_reconciled_payload()))
        request = StartFeatureRequest.model_validate(_feature_payload())
        state = _initial_feature_state("feature-reconcile-journal", request)
        orchestrator = FeatureWorkflowOrchestrator(
            product_manager=cast(Any, product_manager),
            operation_executor_factory=lambda snapshot: ExternalOperationExecutor(
                journal=journal,
                cancellation_token=MockCancellationToken(),
                scope=ExternalOperationScope(
                    workflow_id=snapshot.workflow_id, feature_id=snapshot.feature_id
                ),
            ),
        )
        waiting = await orchestrator.start(state, credentials=CREDENTIALS)
        result = await orchestrator.resume(waiting, answers=_answers(), credentials=CREDENTIALS)

        rows = await journal.list_operations_for_feature(
            result.feature_id, operation_types=[ExternalOperationType.RUN_PRODUCT_MANAGER]
        )
        steps = sorted(str(row.safe_metadata["logical_step"]) for row in rows)
        assert steps == ["product_manager", "requirement_reconciliation"]
    finally:
        await database.drop_schema()
        await database.dispose()


# --- A2 -------------------------------------------------------------------------------------


async def test_a_pure_confirmation_writes_no_revision() -> None:
    """Answers that contradict nothing cost one call and change nothing at all."""
    product_manager = _ScriptedProductManager(json.dumps(_confirming_payload()))
    orchestrator, waiting = await _waiting_feature(product_manager)
    revisions_before = len(_prd_revisions(waiting))

    result = await orchestrator.resume(waiting, answers=_answers(), credentials=CREDENTIALS)

    # One revision for the answers themselves, and none for a reconciliation with nothing to
    # say. A revision differing from its predecessor in nothing but a timestamp is noise a
    # reader has to diff to dismiss.
    assert len(_prd_revisions(result)) == revisions_before + 1
    reconciled = _newest_prd(result)
    assert "requirement_reconciliation" not in reconciled.metadata
    assert _requirement(reconciled, "FR-006").dependencies == ["Transactional app persistence"]
    assert product_manager.client.calls == 1
    assert result.status is FeatureWorkflowStatus.COMPLETED


async def test_a_feature_nobody_answered_never_reaches_the_reconciliation_call() -> None:
    """No clarification round, no reconciliation: the negative check on the added cost."""

    class _NeverAsking(_ScriptedProductManager):
        async def create_technical_prd(
            self, *, feature_id: str, prd: Any, design_snapshot: Any = None
        ) -> TechnicalPRDArtifact:
            """The same PRD with nothing open, which is the ordinary case."""
            draft = await super().create_technical_prd(feature_id=feature_id, prd=prd)
            return draft.model_copy(update={"unresolved_questions": []})

    product_manager = _NeverAsking(json.dumps(_confirming_payload()))
    request = StartFeatureRequest.model_validate(_feature_payload())
    state = _initial_feature_state("feature-reconcile-unasked", request)
    orchestrator = FeatureWorkflowOrchestrator(product_manager=cast(Any, product_manager))

    result = await orchestrator.start(state, credentials=CREDENTIALS)

    assert result.status is FeatureWorkflowStatus.COMPLETED
    assert product_manager.client.calls == 0


# --- A4 -------------------------------------------------------------------------------------


async def test_reconciliation_failure_stops_the_feature_at_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the fault allowance it is a planning-stage failure, never a quiet skip."""
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)
    fault = LLMAdapterError(
        "provider timeout",
        diagnostics=("The model provider did not answer.",),
        failure_classification="APITimeoutError",
    )
    product_manager = _ScriptedProductManager(fault)
    planner = _CriteriaCopyingPlanner()
    orchestrator, waiting = await _waiting_feature(product_manager, planner=planner)
    answered = await orchestrator.begin_resume(waiting, answers=_answers())
    revisions_before = len(_prd_revisions(answered))

    with pytest.raises(LLMAdapterError):
        await orchestrator._step_plan(answered, credentials=CREDENTIALS)

    # Nothing was planned against the unreconciled PRD -- proceeding anyway is the defect
    # with extra steps.
    assert planner.received == []
    assert len(_prd_revisions(answered)) == revisions_before
    assert _requirement(_newest_prd(answered), "FR-006").dependencies == [
        "Transactional app persistence"
    ]
    # The allowance was spent in place, and the time it took is charged to the fault clock
    # rather than to the feature's wall-clock ceiling.
    assert product_manager.client.calls == _ALLOWED_FEATURE_STAGE_INFRASTRUCTURE_FAULTS + 1
    assert (answered.planning_provider_fault_seconds or 0.0) > 0.0
    # And the answers survive: they were persisted before the call that failed, so recovery
    # is a resume rather than a second submission.
    assert _newest_prd(answered).metadata["clarification_answers"] == {
        TRANSACTION_QUESTION_ID: COMPENSATION_ANSWER
    }


# --- A6 -------------------------------------------------------------------------------------


async def test_a_renamed_requirement_is_refused_and_never_replaces_the_prd() -> None:
    """The structural refusal: reconciliation restates requirements, it cannot rename one."""
    renamed = _reconciled_payload()
    renamed["changed_requirements"][0]["requirement_id"] = "FR-006-compensating"
    product_manager = _ScriptedProductManager(json.dumps(renamed), json.dumps(renamed))
    planner = _CriteriaCopyingPlanner()
    orchestrator, waiting = await _waiting_feature(product_manager, planner=planner)
    answered = await orchestrator.begin_resume(waiting, answers=_answers())
    revisions_before = len(_prd_revisions(answered))

    with pytest.raises(AgentArtifactError, match="reconcile the requirements"):
        await orchestrator._step_plan(answered, credentials=CREDENTIALS)

    # Rejected, restated to the provider once, and rejected again: two calls, no revision.
    assert product_manager.client.calls == 2
    assert "unknown requirement" in product_manager.client.instructions[1]
    assert len(_prd_revisions(answered)) == revisions_before
    assert planner.received == []


async def test_a_dropped_requirement_is_refused() -> None:
    """A response proposing a smaller PRD is proposing to redesign approved scope."""
    dropped = _reconciled_payload()
    dropped["requirement_ids"] = ["FR-005", "FR-006"]
    product_manager = _ScriptedProductManager(json.dumps(dropped), json.dumps(dropped))
    orchestrator, waiting = await _waiting_feature(product_manager)
    answered = await orchestrator.begin_resume(waiting, answers=_answers())

    with pytest.raises(AgentArtifactError):
        await orchestrator._step_plan(answered, credentials=CREDENTIALS)

    assert len(_prd_revisions(answered)) == 2
    assert _requirement(_newest_prd(answered), "FR-006").dependencies == [
        "Transactional app persistence"
    ]


async def test_a_rejected_response_that_is_repaired_lands_the_revision() -> None:
    """The refusal is a correction first: one repair, on the planner's own terms."""
    renamed = _reconciled_payload()
    renamed["changed_requirements"][0]["requirement_id"] = "FR-006-compensating"
    product_manager = _ScriptedProductManager(
        json.dumps(renamed), json.dumps(_reconciled_payload())
    )
    orchestrator, waiting = await _waiting_feature(product_manager)

    result = await orchestrator.resume(waiting, answers=_answers(), credentials=CREDENTIALS)

    assert product_manager.client.calls == 2
    assert _requirement(_newest_prd(result), "FR-006").dependencies == [
        "Compensating cleanup of created app documents"
    ]
    assert result.status is FeatureWorkflowStatus.COMPLETED


async def test_an_emptied_requirement_and_an_unnamed_driver_are_refused() -> None:
    """A requirement may be restated, never emptied, and never anonymously."""
    technical_prd = _technical_prd("feature-parse")
    requirements = [
        *technical_prd.functional_requirements,
        *technical_prd.non_functional_requirements,
    ]
    answers = [
        AnsweredClarification(
            question_id=TRANSACTION_QUESTION_ID,
            question="Does the target cluster support multi-document transactions?",
            answer=COMPENSATION_ANSWER,
        )
    ]

    emptied = _reconciled_payload()
    emptied["changed_requirements"][0]["acceptance_criteria"] = []
    with pytest.raises(ReconciliationRejected, match="never emptied"):
        parse_reconciliation(emptied, requirements=requirements, answers=answers)

    unattributed = _reconciled_payload()
    unattributed["changed_requirements"][0]["driving_answer_ids"] = ["recon-backend-9"]
    with pytest.raises(ReconciliationRejected, match="must name the answers"):
        parse_reconciliation(unattributed, requirements=requirements, answers=answers)

    reordered = _reconciled_payload()
    reordered["requirement_ids"] = ["NFR-001", "FR-005", "FR-006"]
    with pytest.raises(ReconciliationRejected, match="in the same order"):
        parse_reconciliation(reordered, requirements=requirements, answers=answers)


# --- A5 -------------------------------------------------------------------------------------


async def test_the_plan_and_the_review_scope_inherit_the_reconciled_text() -> None:
    """End to end: what the planner scopes and the reviewer enforces is the reconciled text.

    `feature_planner` derives every acceptance criterion id from
    `requirement.acceptance_criteria`, and the reviewer validates its requirement checks
    against a scope built from the criteria the plan carries. Neither needed a change for this
    task, which is exactly why it is asserted: the whole fix is that they now read one story.
    """
    product_manager = _ScriptedProductManager(json.dumps(_reconciled_payload()))
    planner = _CriteriaCopyingPlanner()
    orchestrator, waiting = await _waiting_feature(product_manager, planner=planner)

    result = await orchestrator.resume(waiting, answers=_answers(), credentials=CREDENTIALS)

    # The planner was handed the reconciled PRD, and only that.
    assert len(planner.received) == 1
    handed = _requirement(planner.received[0], "FR-006")
    assert handed.dependencies == ["Compensating cleanup of created app documents"]
    # The criterion ids the plan may scope are the reconciled ones: FR-006 now has three
    # criteria, so the stale two-criterion projection no longer validates.
    reconciled = _newest_prd(result)
    plan = next(
        artifact
        for artifact in reversed(result.artifacts)
        if isinstance(artifact, RepositoryExecutionPlanArtifact)
    )
    _ensure_plan_covers_technical_requirements(plan, reconciled)
    stale = plan.model_copy(
        update={
            "workstreams": [
                workstream.model_copy(
                    update={
                        "scoped_requirements": [
                            reference.model_copy(
                                update={
                                    "acceptance_criterion_ids": [
                                        *reference.acceptance_criterion_ids,
                                        "FR-006:ac-4",
                                    ]
                                }
                            )
                            if reference.requirement_id == "FR-006"
                            else reference
                            for reference in workstream.scoped_requirements
                        ]
                    }
                )
                for workstream in plan.workstreams
            ]
        }
    )
    with pytest.raises(AgentArtifactError, match="unknown acceptance criteria"):
        _ensure_plan_covers_technical_requirements(stale, reconciled)
    # And the criteria the reviewer is given are the reconciled sentences themselves.
    contract = next(
        artifact
        for artifact in reversed(result.artifacts)
        if isinstance(artifact, IntegrationContractArtifact)
    )
    task_plan = _child_task_plan(
        child=result.child_workflows["backend"],
        technical_prd=reconciled,
        contract=contract,
        workstream=plan.workstreams[0],
        feedback=[],
    )
    scope_criteria = " ".join(_review_scope(task_plan)["acceptance_criteria"])
    assert "compensating cleanup" in scope_criteria
    assert "external residue" in scope_criteria
    assert "rolls back all app records" not in scope_criteria


# --- A3 -------------------------------------------------------------------------------------


class _TwoQuestionProductManager(_ScriptedProductManager):
    """The same draft PRD, asking the second question whose answer contradicts the first."""

    async def create_technical_prd(
        self, *, feature_id: str, prd: Any, design_snapshot: Any = None
    ) -> TechnicalPRDArtifact:
        """Add the residue question, so a person can answer two things that disagree."""
        draft = await super().create_technical_prd(feature_id=feature_id, prd=prd)
        return draft.model_copy(
            update={
                "unresolved_questions": [
                    *draft.unresolved_questions,
                    ClarificationQuestion(
                        question_id=RESIDUE_QUESTION_ID,
                        question="Should a partly failed upload keep the apps it created?",
                        rationale="The requirement is silent about a partial failure.",
                        required=True,
                    ),
                ]
            }
        )


def _contradicting_answers() -> list[ClarificationAnswer]:
    """Two answers that cannot both be written into one requirement."""
    return [
        ClarificationAnswer(question_id=TRANSACTION_QUESTION_ID, answer=COMPENSATION_ANSWER),
        ClarificationAnswer(question_id=RESIDUE_QUESTION_ID, answer=RESIDUE_ANSWER),
    ]


async def test_an_unsettled_contradiction_asks_instead_of_planning() -> None:
    """The platform refuses to guess: both sides are quoted back and a person decides."""
    product_manager = _TwoQuestionProductManager(json.dumps(_contradicting_payload()))
    planner = _CriteriaCopyingPlanner()
    orchestrator, waiting = await _waiting_feature(product_manager, planner=planner)

    result = await orchestrator.resume(
        waiting, answers=_contradicting_answers(), credentials=CREDENTIALS
    )

    assert result.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
    # Nothing was planned against a requirement whose text nobody has settled.
    assert planner.received == []
    asked = _newest_prd(result).unresolved_questions
    assert [item.question_id for item in asked] == [RECONCILE_QUESTION_ID]
    question = asked[0]
    # Both sides, verbatim, and the platform's own words around them.
    assert COMPENSATION_ANSWER in question.question
    assert RESIDUE_ANSWER in question.question
    assert "Should a partly failed upload keep the apps it created?" in question.question
    assert "Which one holds" in question.question
    assert "One answer removes what a failed request created" in question.rationale
    # No suggestion: this is the one decision the platform has just said it cannot make.
    assert question.suggested_answer == ""
    # The round the answers spent is the round the cap counts, and no new cap exists.
    assert result.clarification_rounds == 1
    assert result.max_clarification_rounds == waiting.max_clarification_rounds
    # Already-settled answers are not re-litigated on the next round.
    assert _newest_prd(result).metadata["reconciled_answer_ids"] == [
        TRANSACTION_QUESTION_ID,
        RESIDUE_QUESTION_ID,
    ]


async def test_the_answer_to_a_reconcile_question_reconciles_like_the_first_round() -> None:
    """A second round is an ordinary round: it reconciles, it plans, it finishes."""
    product_manager = _TwoQuestionProductManager(
        json.dumps(_contradicting_payload()),
        json.dumps(_reconciled_payload(driver=RECONCILE_QUESTION_ID)),
    )
    orchestrator, waiting = await _waiting_feature(product_manager)
    asked_again = await orchestrator.resume(
        waiting, answers=_contradicting_answers(), credentials=CREDENTIALS
    )
    question_id = _newest_prd(asked_again).unresolved_questions[0].question_id
    assert question_id == RECONCILE_QUESTION_ID

    result = await orchestrator.resume(
        asked_again,
        answers=[
            ClarificationAnswer(
                question_id=question_id,
                answer="Compensating deletes hold; nothing created by a failed request stays.",
            )
        ],
        credentials=CREDENTIALS,
    )

    assert result.status is FeatureWorkflowStatus.COMPLETED
    assert result.clarification_rounds == 2
    assert _requirement(_newest_prd(result), "FR-006").dependencies == [
        "Compensating cleanup of created app documents"
    ]


async def test_a_repeated_planning_step_neither_re_asks_nor_pays_for_a_second_call() -> None:
    """The answers a revision states are settled: a step that runs twice reconciles once.

    This is the crash-resume shape. The planning step is not checkpointed inside itself, so a
    process that dies after the reconciliation revision runs the step again -- and a second
    reconciliation there would spend another call and put the same question in front of its
    author for a second time.
    """
    product_manager = _TwoQuestionProductManager(json.dumps(_contradicting_payload()))
    orchestrator, waiting = await _waiting_feature(product_manager)
    asked_again = await orchestrator.resume(
        waiting, answers=_contradicting_answers(), credentials=CREDENTIALS
    )
    calls_before = product_manager.client.calls
    revisions_before = len(_prd_revisions(asked_again))

    repeated = await orchestrator._step_plan(asked_again, credentials=CREDENTIALS)

    assert product_manager.client.calls == calls_before
    assert len(_prd_revisions(repeated)) == revisions_before
    assert [item.question_id for item in _newest_prd(repeated).unresolved_questions] == [
        RECONCILE_QUESTION_ID
    ]
    assert repeated.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN


async def test_at_the_rounds_cap_the_existing_terminal_fires_unchanged() -> None:
    """Reconciliation asks; it never gets a bound of its own, and never vetoes."""
    product_manager = _TwoQuestionProductManager(json.dumps(_contradicting_payload()))
    orchestrator, waiting = await _waiting_feature(product_manager)
    asked_again = await orchestrator.resume(
        waiting, answers=_contradicting_answers(), credentials=CREDENTIALS
    )
    question_id = _newest_prd(asked_again).unresolved_questions[0].question_id
    at_the_cap = asked_again.model_copy(
        update={"clarification_rounds": asked_again.max_clarification_rounds}
    )

    result = await orchestrator.resume(
        at_the_cap,
        answers=[ClarificationAnswer(question_id=question_id, answer="Compensating deletes.")],
        credentials=CREDENTIALS,
    )

    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert result.failure_summary is not None
    assert result.failure_summary.root_classification == "clarification_unresolved"
