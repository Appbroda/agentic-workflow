"""What the platform offers as an answer, and what it refuses to make up.

A clarification question that reads "does this endpoint use the existing admin middleware?" is
asking somebody to go and read a repository the platform has just finished reading. Where the
answer is a fact about a checkout, the platform states it and prefills the field. Where it is
not, all three suggestion fields stay empty and the question is asked plainly -- because an
ungrounded suggestion looks exactly like a grounded one and gets accepted.

Answering remains an explicit act either way: a suggestion is never submitted on somebody's
behalf, which is the property the last test here holds.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from adapters.llm_adapter import ImageInput
from agents.recon.agent import _grounded_answers
from agents.shared.contracts import ARTIFACT_FILENAMES, create_artifact
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import FeatureRecord, _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from api.identity import WorkspaceScope
from api.schemas import ClarificationAnswer
from artifacts.schemas import ClarificationQuestion, TechnicalPRDArtifact
from main import create_app
from state.enums import FeatureWorkflowStatus
from state.external_operations import WorkflowCheckpointBoundary
from state.feature_models import FeatureFailureSummary, FeatureWorkflowSnapshot
from storage.db import Database
from storage.feature_store import SqlAlchemyFeatureControlPlane
from tests.support import settle
from tests.test_feature_api import feature_payload
from tests.test_feature_workflow import RecordingReconnaissance, _reconnaissance_artifact
from workflows.feature_workflow import FeatureWorkflowOrchestrator, validate_clarification_answers

KEY = "clarify-key"
AUTH = {"Authorization": f"Bearer {KEY}"}

_GLOBAL_AUTH_PREMISE = {
    "premise": "Authentication can be applied per route.",
    "contradicted_by": "Authentication is applied once, globally, in server/config/express.js.",
    "evidence_paths": ["server/config/express.js"],
    "question": "Rely on the global middleware, or introduce per-route auth?",
}


class BlockingResumeOrchestrator(FeatureWorkflowOrchestrator):
    """Keep an accepted clarification resume live while its public read model is inspected."""

    def __init__(self) -> None:
        super().__init__(
            reconnaissance=cast(
                Any,
                RecordingReconnaissance(contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]}),
            )
        )
        self.resume_started = asyncio.Event()
        self.release_resume = asyncio.Event()

    async def advance_one_step(
        self,
        state: Any,
        *,
        credentials: RequestScopedCredentials,
    ) -> Any:
        """Prove the queue owns the accepted answers before allowing planning to finish.

        Blocked in the step rather than in `resume`, because a claim is one step now: it
        settles what the resume carried and then runs the step this holds open. Nothing about
        what the test asserts changes -- the accepted answers are the queue's, and the
        durable checkpoint is still the waiting one until the step ends.

        Only the step that follows a resume is held. Every claim runs through here, including
        the ones that carry the feature to its clarification pause in the first place, and a
        recorded clarification round is what tells the two apart.
        """
        if state.clarification_rounds == 0:
            return await super().advance_one_step(state, credentials=credentials)
        self.resume_started.set()
        await self.release_resume.wait()
        return await super().advance_one_step(state, credentials=credentials)


@pytest.mark.asyncio
async def test_a_repository_premise_question_arrives_with_the_answer_the_checkout_gives() -> None:
    """The suggestion is what the repository does, quoted, with the file it was read from."""
    state = _initial_feature_state(
        "feature-suggested", StartFeatureRequest.model_validate(feature_payload())
    )
    orchestrator = FeatureWorkflowOrchestrator(
        reconnaissance=cast(
            Any, RecordingReconnaissance(contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]})
        )
    )

    waiting = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert waiting.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
    technical_prd = next(
        item for item in reversed(waiting.artifacts) if isinstance(item, TechnicalPRDArtifact)
    )
    question = next(
        item for item in technical_prd.unresolved_questions if item.question_id == "recon-backend-1"
    )
    # The answer the platform would give, in the words the checkout justified.
    assert "applied once, globally" in question.suggested_answer
    assert question.suggested_answer.startswith("Follow what backend already does:")
    # And where it read it, so the suggestion can be checked rather than trusted.
    assert "repository analysis of backend" in question.suggestion_source
    assert "server/config/express.js" in question.suggestion_source
    assert question.suggestion_confidence == "high"


@pytest.mark.asyncio
async def test_a_question_the_repositories_cannot_answer_carries_no_suggestion() -> None:
    """An unverifiable acceptance criterion is a decision only the author can make.

    Nothing in a checkout says whether a latency figure is measured elsewhere or should be
    dropped, so the platform asks and leaves the field empty rather than proposing something.
    """
    payload = feature_payload()
    payload["prd"]["requirements"][0]["acceptance_criteria"] = [
        "A load test shows p99 latency under 100ms in production."
    ]
    state = _initial_feature_state(
        "feature-unverifiable", StartFeatureRequest.model_validate(payload)
    )
    orchestrator = FeatureWorkflowOrchestrator(reconnaissance=cast(Any, RecordingReconnaissance()))

    waiting = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    technical_prd = next(
        item for item in reversed(waiting.artifacts) if isinstance(item, TechnicalPRDArtifact)
    )
    criteria_questions = [
        item
        for item in technical_prd.unresolved_questions
        if item.question_id.startswith("criteria-")
    ]
    assert criteria_questions, "an unverifiable criterion should still be asked about"
    for question in criteria_questions:
        assert question.suggested_answer == ""
        assert question.suggestion_source == ""
        assert question.suggestion_confidence is None


class GroundingReconnaissance(RecordingReconnaissance):
    """Reconnaissance that also answers the questions its evidence settles.

    A scripted stand-in for the model call: `answers` is what the model would have returned,
    so the test exercises the provenance rules the platform applies to it rather than the
    model's judgement.
    """

    def __init__(self, *, answers: list[dict[str, Any]], **kwargs: Any) -> None:
        """Record the scripted answers and how many times grounding was asked for."""
        super().__init__(**kwargs)
        self._answers = answers
        self.asked: list[str] = []

    async def suggest_answers(
        self, *, feature_id: str, questions: Any, reconnaissance: Any
    ) -> list[Any]:
        """Apply the scripted answers through the real provenance checks."""
        del feature_id
        self.asked = [item.question_id for item in questions]
        grounded = _grounded_answers(self._answers, list(reconnaissance))
        return [
            item.model_copy(update=grounded[item.question_id])
            if item.question_id in grounded
            else item
            for item in questions
        ]


@pytest.mark.asyncio
async def test_the_product_managers_own_questions_are_answered_from_the_checkouts() -> None:
    """The gap this closes: a question written before any repository was read.

    The product manager asks about existing behaviour it has no way to know, and by the time
    the repositories have actually been read those questions are already fixed. Asking
    somebody to go and find what the platform just recorded is asking them to do it twice.
    """
    state = _initial_feature_state(
        "feature-grounded", StartFeatureRequest.model_validate(feature_payload())
    )
    reconnaissance = GroundingReconnaissance(
        contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]},
        answers=[
            {
                "question_id": "UQ-001",
                "answer": "Deletion is a hard delete; there is no retention path.",
                "repository_id": "backend",
                "evidence_paths": ["backend/src"],
                "confidence": "high",
            }
        ],
    )
    orchestrator = FeatureWorkflowOrchestrator(
        reconnaissance=cast(Any, reconnaissance),
        product_manager=cast(
            Any, AskingProductManager("UQ-001", "Does deleting an app remove it permanently?")
        ),
    )

    waiting = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    technical_prd = next(
        item for item in reversed(waiting.artifacts) if isinstance(item, TechnicalPRDArtifact)
    )
    question = next(
        item for item in technical_prd.unresolved_questions if item.question_id == "UQ-001"
    )
    assert question.suggested_answer == "Deletion is a hard delete; there is no retention path."
    assert "repository analysis of backend" in question.suggestion_source
    assert "backend/src" in question.suggestion_source
    assert question.suggestion_confidence == "high"
    # Recorded on the artifact, so it is visible afterwards which answers the platform
    # supplied rather than the author.
    assert technical_prd.metadata["repository_grounded_answer_ids"] == ["UQ-001"]


@pytest.mark.asyncio
async def test_a_criteria_question_is_never_put_to_the_repositories() -> None:
    """It asks the author to restate a criterion. No checkout has an opinion about that."""
    payload = feature_payload()
    payload["prd"]["requirements"][0]["acceptance_criteria"] = [
        "A load test shows p99 latency under 100ms in production."
    ]
    state = _initial_feature_state(
        "feature-criteria-excluded", StartFeatureRequest.model_validate(payload)
    )
    reconnaissance = GroundingReconnaissance(
        contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]}, answers=[]
    )
    orchestrator = FeatureWorkflowOrchestrator(reconnaissance=cast(Any, reconnaissance))

    await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )

    assert reconnaissance.asked, "grounding should have been asked for"
    assert not any(item.startswith("criteria-") for item in reconnaissance.asked)


def test_an_answer_naming_a_repository_this_feature_never_read_is_dropped() -> None:
    """An identifier the model invented is not evidence, whatever it claims to have read."""
    artifact = _reconnaissance_artifact("feature-x", "backend")

    kept = _grounded_answers(
        [
            {
                "question_id": "UQ-1",
                "answer": "It is done this way.",
                "repository_id": "a-repository-nobody-inspected",
                "evidence_paths": ["backend/src"],
                "confidence": "high",
            }
        ],
        [artifact],
    )

    assert kept == {}


def test_an_answer_citing_a_path_the_scan_never_saw_is_dropped() -> None:
    """A suggestion nobody can trace back reads exactly like one they can."""
    artifact = _reconnaissance_artifact("feature-x", "backend")

    kept = _grounded_answers(
        [
            {
                "question_id": "UQ-1",
                "answer": "Authorisation lives in the middleware.",
                "repository_id": "backend",
                "evidence_paths": ["backend/src/middleware/invented.js"],
                "confidence": "high",
            },
            {
                "question_id": "UQ-2",
                "answer": "The tests live beside the source.",
                "repository_id": "backend",
                # A path the scan actually recorded, quoted exactly.
                "evidence_paths": ["backend/tests"],
                "confidence": "medium",
            },
        ],
        [artifact],
    )

    assert set(kept) == {"UQ-2"}
    assert kept["UQ-2"]["suggestion_confidence"] == "medium"


def test_an_answer_with_no_text_is_dropped_rather_than_shown_as_a_blank_suggestion() -> None:
    """Omitting a question is the documented way to say the evidence does not answer it."""
    artifact = _reconnaissance_artifact("feature-x", "backend")

    kept = _grounded_answers(
        [
            {
                "question_id": "UQ-1",
                "answer": "   ",
                "repository_id": "backend",
                "evidence_paths": ["backend/src"],
                "confidence": "low",
            }
        ],
        [artifact],
    )

    assert kept == {}


class AskingProductManager:
    """A product manager that leaves exactly one question open, and answers none of it.

    Which is the real situation: it writes its questions before any repository has been read,
    so it cannot know what the checkouts contain.
    """

    def __init__(self, question_id: str, question: str) -> None:
        """Record the one question this product manager will leave open."""
        self._question_id = question_id
        self._question = question

    async def create_technical_prd(
        self, *, feature_id: str, prd: Any, design_snapshot: Any = None
    ) -> TechnicalPRDArtifact:
        """Return a technical PRD carrying that question, with no suggestion attached."""
        del prd, design_snapshot
        return _technical_prd_asking(feature_id, self._question_id, self._question)


def _technical_prd_asking(feature_id: str, question_id: str, question: str) -> TechnicalPRDArtifact:
    """A technical PRD carrying one open question the product manager could not answer."""
    return create_artifact(
        TechnicalPRDArtifact,
        workflow_id=feature_id,
        artifact_id=ARTIFACT_FILENAMES["technical_prd"],
        producer="product_manager",
        metadata={},
        payload={
            "title": "Bulk deletion",
            "solution_summary": "Delete several apps at once.",
            "functional_requirements": [
                {
                    "requirement_id": "REQ-1",
                    "description": "Delete several apps in one action.",
                    "priority": "must",
                    "acceptance_criteria": ["The selected apps are deleted."],
                    "dependencies": [],
                }
            ],
            "non_functional_requirements": [],
            "data_requirements": [],
            "integration_requirements": [],
            "security_requirements": [],
            "assumptions": [],
            "unresolved_questions": [
                {
                    "question_id": question_id,
                    "question": question,
                    "rationale": "The product document does not say.",
                    "required": True,
                }
            ],
        },
    )


def test_a_suggestion_without_provenance_is_rejected() -> None:
    """A prefilled answer whose origin is unstated is indistinguishable from the author's own."""
    with pytest.raises(ValidationError, match="where it came from"):
        ClarificationQuestion(
            question_id="Q-1",
            question="Should this require authentication?",
            rationale="Because it exposes data.",
            required=True,
            suggested_answer="Yes.",
        )


@pytest.mark.asyncio
async def test_the_clarification_endpoint_serves_the_suggestion_and_never_submits_it() -> None:
    """The suggestion reaches the browser, and the feature stays waiting until somebody answers.

    Both halves matter. Without the first the platform knows the answer and does not say so;
    without the second a prefilled field becomes a decision nobody made.
    """
    app = create_app(platform_api_key=KEY)
    app.state.feature_control_plane._mock_runner = FeatureWorkflowOrchestrator(  # noqa: SLF001
        reconnaissance=cast(
            Any, RecordingReconnaissance(contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]})
        )
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=AUTH, json=feature_payload())
        await settle(app)
        clarification = await client.get("/features/feature-login/clarification", headers=AUTH)
        feature = await client.get("/features/feature-login", headers=AUTH)

    body = clarification.json()
    assert body["awaiting_answers"] is True
    question = next(item for item in body["questions"] if item["question_id"] == "recon-backend-1")
    assert "applied once, globally" in question["suggested_answer"]
    assert "repository analysis of backend" in question["suggestion_source"]
    assert question["suggestion_confidence"] == "high"
    # Nothing was answered on the author's behalf: the feature is still waiting.
    assert feature.json()["status"] == "waiting_for_human"
    assert body["previous_answers"] == {}


@pytest.mark.asyncio
async def test_provisional_questions_are_published_as_investigating_not_awaiting() -> None:
    """A checkpointed PRD's questions are visible as the platform's own open items.

    This test used to assert the opposite -- `questions: []` while analysis is active --
    which encoded exactly the defect AB-Feature-173's operator watched for 35 minutes: the
    questions the platform was busy answering sat unrendered while the panel read as
    "waiting for you". What must still hold is that nothing is *actionable*: submitting
    against a provisional list stays refused (`awaiting_answers: false`, and the answer
    validator requires the human gate). What changed, deliberately, is that the list itself
    is served, labelled `investigating`.
    """
    state = _initial_feature_state(
        "feature-provisional", StartFeatureRequest.model_validate(feature_payload())
    )
    waiting = await FeatureWorkflowOrchestrator(
        reconnaissance=cast(
            Any, RecordingReconnaissance(contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]})
        )
    ).start(state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None))
    assert waiting.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
    assert any(
        isinstance(item, TechnicalPRDArtifact) and item.unresolved_questions
        for item in waiting.artifacts
    )
    provisional = waiting.model_copy(
        update={
            "status": FeatureWorkflowStatus.ANALYZING_PRD,
            "current_agent": "product_manager",
        }
    )

    class ProvisionalControlPlane:
        # `scope` is accepted and unused: the route reaches this through a
        # `ScopedFeatureControlPlane`, which supplies the request's workspace on every call.
        async def get_record(
            self, feature_id: str, *, scope: WorkspaceScope | None = None
        ) -> FeatureRecord:
            assert feature_id == provisional.feature_id
            return FeatureRecord(
                state=provisional,
                created_at=provisional.created_at,
                updated_at=provisional.updated_at,
            )

    app = create_app(
        platform_api_key=KEY,
        feature_control_plane=cast(Any, ProvisionalControlPlane()),
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        clarification = await client.get(
            "/features/feature-provisional/clarification", headers=AUTH
        )

    assert clarification.status_code == 200
    body = clarification.json()
    # Nothing to submit: the human gate has not been reached.
    assert body["awaiting_answers"] is False
    # But the questions are rendered, as the platform's open items rather than the user's.
    assert body["clarification_state"] == "investigating"
    assert body["questions"], "the platform's open items must be visible while it works"
    assert body["technical_prd_artifact_id"]


@pytest.mark.asyncio
async def test_answers_are_checkpointed_before_planning_and_not_requested_after_failure() -> None:
    """A failed contract call resumes from accepted answers instead of asking for them again."""
    checkpoints: list[tuple[FeatureWorkflowSnapshot, WorkflowCheckpointBoundary]] = []

    async def checkpoint(
        state: FeatureWorkflowSnapshot,
        boundary: WorkflowCheckpointBoundary,
        _repository_id: str | None,
    ) -> None:
        checkpoints.append((state, boundary))

    class FailingPlanner:
        async def plan(self, **_kwargs: Any) -> Any:
            msg = "simulated provider disconnect during shared-contract planning"
            raise RuntimeError(msg)

    state = _initial_feature_state(
        "feature-answer-checkpoint", StartFeatureRequest.model_validate(feature_payload())
    )
    orchestrator = FeatureWorkflowOrchestrator(
        reconnaissance=cast(
            Any, RecordingReconnaissance(contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]})
        ),
        planner=cast(Any, FailingPlanner()),
        checkpoint_writer=checkpoint,
    )
    waiting = await orchestrator.start(
        state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
    )
    technical_prd = next(
        item for item in reversed(waiting.artifacts) if isinstance(item, TechnicalPRDArtifact)
    )
    answers = [
        ClarificationAnswer(question_id=item.question_id, answer="Use the recorded decision.")
        for item in technical_prd.unresolved_questions
    ]

    with pytest.raises(RuntimeError, match="provider disconnect"):
        await orchestrator.resume(
            waiting,
            answers=answers,
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )

    accepted, boundary = next(
        (snapshot, checkpoint_boundary)
        for snapshot, checkpoint_boundary in reversed(checkpoints)
        if snapshot.clarification_rounds == 1
    )
    accepted_prd = next(
        item for item in reversed(accepted.artifacts) if isinstance(item, TechnicalPRDArtifact)
    )
    assert boundary is WorkflowCheckpointBoundary.BEFORE_CLONE
    assert accepted.status is FeatureWorkflowStatus.PLANNING
    assert accepted.current_agent == "planner"
    assert accepted_prd.unresolved_questions == []
    assert len(accepted_prd.metadata["clarification_answers"]) == len(answers)

    recoverable = accepted.model_copy(
        update={
            "status": FeatureWorkflowStatus.WAITING_FOR_HUMAN,
            "current_agent": "feature_runtime",
            "failure_summary": FeatureFailureSummary(
                stage="feature_runtime",
                agent="feature_runtime",
                root_classification="APIConnectionError",
                diagnostics=["the model provider call failed"],
                retryable=True,
                next_action="Resume from the accepted-answer checkpoint.",
                recorded_at=datetime.now(UTC),
            ),
        }
    )
    validate_clarification_answers(recoverable, [])

    class RecoverableControlPlane:
        async def get_record(
            self, feature_id: str, *, scope: WorkspaceScope | None = None
        ) -> FeatureRecord:
            assert feature_id == recoverable.feature_id
            return FeatureRecord(
                state=recoverable,
                created_at=recoverable.created_at,
                updated_at=recoverable.updated_at,
            )

    app = create_app(
        platform_api_key=KEY,
        feature_control_plane=cast(Any, RecoverableControlPlane()),
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        feature = await client.get(f"/features/{recoverable.feature_id}", headers=AUTH)
        clarification = await client.get(
            f"/features/{recoverable.feature_id}/clarification", headers=AUTH
        )

    assert "RESUME_WORKFLOW" in feature.json()["available_actions"]
    assert "ANSWER_CLARIFICATION" not in feature.json()["available_actions"]
    assert clarification.json()["awaiting_answers"] is False
    assert clarification.json()["questions"] == []


@pytest.mark.asyncio
async def test_an_accepted_resume_is_not_still_advertised_as_waiting_for_answers(
    tmp_path: Path,
) -> None:
    """The queue and checkpoint together are the live state, not either one in isolation.

    A resume deliberately leaves the waiting checkpoint intact until its background worker
    reaches the next safe boundary. Returning that checkpoint by itself re-rendered the same
    form after a successful submission, so every further click spent ten seconds colliding
    with the worker's run lock and was recorded as a failed action. Hiding only the action flag
    is insufficient for an older client, so the non-actionable question list is empty too.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'active-clarification.db'}")
    await database.create_schema()
    runner = BlockingResumeOrchestrator()
    store = SqlAlchemyFeatureControlPlane(database, mock_runner=runner)
    app = create_app(platform_api_key=KEY, feature_control_plane=store)

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            await client.post("/features/start", headers=AUTH, json=feature_payload())
            await settle(app)
            open_questions = await client.get("/features/feature-login/clarification", headers=AUTH)
            answers = [
                {"question_id": item["question_id"], "answer": "Use the existing behavior."}
                for item in open_questions.json()["questions"]
            ]

            accepted = await client.post(
                "/features/feature-login/resume",
                headers={**AUTH, "Idempotency-Key": "accepted-clarification-resume"},
                json={"answers": answers},
            )
            # An injected test control plane has no production notifier wired into it. Start
            # exactly one claim explicitly so the read below observes a running worker.
            app.state.feature_dispatcher.schedule()
            await asyncio.wait_for(runner.resume_started.wait(), timeout=3)
            feature = await client.get("/features/feature-login", headers=AUTH)
            clarification = await client.get("/features/feature-login/clarification", headers=AUTH)

        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["effective_status"] == "resuming"
        assert feature.json()["status"] == "waiting_for_human", "the checkpoint stays durable"
        assert feature.json()["effective_status"] == "resuming"
        assert "ANSWER_CLARIFICATION" not in feature.json()["available_actions"]
        assert clarification.json()["awaiting_answers"] is False
        assert clarification.json()["questions"] == []
        assert clarification.json()["technical_prd_artifact_id"], (
            "the artifact remains available as evidence"
        )
    finally:
        runner.release_resume.set()
        await settle(app)
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_suggestion_survives_being_persisted_and_read_back() -> None:
    """The structured suggestion is part of the artifact, not a value computed for one response."""
    app = create_app(platform_api_key=KEY)
    app.state.feature_control_plane._mock_runner = FeatureWorkflowOrchestrator(  # noqa: SLF001
        reconnaissance=cast(
            Any, RecordingReconnaissance(contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]})
        )
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=AUTH, json=feature_payload())
        await settle(app)
        artifacts = await client.get(
            "/features/feature-login/artifacts",
            headers=AUTH,
            params={"artifact_type": "technical_prd"},
        )
        # Read twice, so a value that only existed on the first read would fail here.
        again = await client.get("/features/feature-login/clarification", headers=AUTH)

    stored = artifacts.json()["artifacts"][-1]["payload"]["unresolved_questions"]
    question = next(item for item in stored if item["question_id"] == "recon-backend-1")
    assert "applied once, globally" in question["suggested_answer"]
    assert question["suggestion_confidence"] == "high"
    assert any(
        "applied once, globally" in item["suggested_answer"] for item in again.json()["questions"]
    )


@pytest.mark.asyncio
async def test_an_edited_answer_is_what_the_platform_plans_from() -> None:
    """A suggestion is a starting point. The answer that is submitted is the one that counts."""
    app = create_app(platform_api_key=KEY)
    app.state.feature_control_plane._mock_runner = FeatureWorkflowOrchestrator(  # noqa: SLF001
        reconnaissance=cast(
            Any, RecordingReconnaissance(contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]})
        )
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=AUTH, json=feature_payload())
        await settle(app)
        resumed = await client.post(
            "/features/feature-login/resume",
            headers=AUTH,
            json={
                "answers": [
                    {
                        "question_id": "recon-backend-1",
                        "answer": "No. Add per-route auth for this endpoint only.",
                    }
                ]
            },
        )
        # A resume is accepted and queued, so the answer it carries is applied by the worker
        # rather than inside the request that submitted it.
        await settle(app)
        artifacts = await client.get(
            "/features/feature-login/artifacts",
            headers=AUTH,
            params={"artifact_type": "technical_prd"},
        )

    assert resumed.status_code == 200, resumed.text
    answers = artifacts.json()["artifacts"][-1]["metadata"]["clarification_answers"]
    assert answers["recon-backend-1"] == "No. Add per-route auth for this endpoint only."


class GroundingFailsReconnaissance(RecordingReconnaissance):
    """Reconnaissance whose answer-suggestion call fails, as it did twice on 2026-08-31."""

    async def suggest_answers(self, **_kwargs: Any) -> list[Any]:
        """Fail the way a real grounding model call does."""
        msg = "the model provider call failed (ReadTimeout)"
        raise RuntimeError(msg)


@pytest.mark.asyncio
async def test_a_grounded_ask_and_a_fallback_ask_are_served_as_different_states() -> None:
    """The API distinguishes "we need you" from "we tried, failed, and now need you".

    `clarification_grounding_failed` fired twice across AB-Feature-173 and -174 on one
    morning, and both renderings were identical to a grounded ask -- the fallback was
    invisible. The state is durable and serialized, never re-derived by the client.
    """
    for reconnaissance, expected_state in (
        (
            RecordingReconnaissance(contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]}),
            "awaiting_answers",
        ),
        (
            GroundingFailsReconnaissance(contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]}),
            "asked_after_grounding_failure",
        ),
    ):
        app = create_app(platform_api_key=KEY)
        app.state.feature_control_plane._mock_runner = FeatureWorkflowOrchestrator(  # noqa: SLF001
            reconnaissance=cast(Any, reconnaissance)
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            await client.post("/features/start", headers=AUTH, json=feature_payload())
            await settle(app)
            clarification = await client.get("/features/feature-login/clarification", headers=AUTH)

        body = clarification.json()
        assert body["awaiting_answers"] is True, expected_state
        assert body["clarification_state"] == expected_state
        assert body["questions"], "both asks still put the questions in front of the human"


@pytest.mark.asyncio
async def test_answering_after_a_grounding_failure_clears_the_fallback_state() -> None:
    """The flag describes one ask; the answered feature does not stay marked forever."""
    app = create_app(platform_api_key=KEY)
    app.state.feature_control_plane._mock_runner = FeatureWorkflowOrchestrator(  # noqa: SLF001
        reconnaissance=cast(
            Any,
            GroundingFailsReconnaissance(contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]}),
        )
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=AUTH, json=feature_payload())
        await settle(app)
        before = await client.get("/features/feature-login/clarification", headers=AUTH)
        assert before.json()["clarification_state"] == "asked_after_grounding_failure"
        question_ids = [item["question_id"] for item in before.json()["questions"]]
        resumed = await client.post(
            "/features/feature-login/resume",
            headers=AUTH,
            json={
                "answers": [
                    {"question_id": item, "answer": "Follow the global middleware."}
                    for item in question_ids
                ]
            },
        )
        assert resumed.status_code == 200, resumed.text
        await settle(app)
        after = await client.get("/features/feature-login/clarification", headers=AUTH)

    body = after.json()
    assert body["awaiting_answers"] is False
    assert body["clarification_state"] == "idle"
    assert body["questions"] == []


class _EmptyInputRefusingClient:
    """Answer like the real adapters: an empty instructions or input is refused before any call.

    Both production clients validate exactly this way, and the workflow-level stubs above do
    not -- which is how `suggest_answers` shipped sending `input_text=""` and failed in zero
    seconds on every feature that ever reached grounding, six for six in the 175-180 matrix,
    while its best-effort fallback quietly asked the human instead.
    """

    def __init__(self, output_text: str) -> None:
        self._output_text = output_text
        self.inputs: list[str] = []

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
    ) -> Any:
        from adapters.llm_adapter import LLMAdapterError, LLMResponse

        if not instructions.strip() or not input_text.strip():
            msg = "instructions and input_text must not be empty"
            raise LLMAdapterError(msg, diagnostics=(msg,), failure_classification="empty_request")
        self.inputs.append(input_text)
        return LLMResponse(
            response_id="resp-grounding",
            model="stub-reasoning",
            output_text=self._output_text,
            input_tokens=None,
            output_tokens=None,
        )


@pytest.mark.asyncio
async def test_grounding_sends_its_evidence_as_input_and_survives_the_adapters_contract() -> None:
    """The questions and findings travel as the user message, and a grounded answer lands."""
    import json as _json

    from agents.recon.agent import RepositoryReconAgent
    from prompts.prompt_loader import PromptLoader

    recon = _reconnaissance_artifact("feature-grounded", "backend")
    llm_client = _EmptyInputRefusingClient(
        _json.dumps(
            {
                "answers": [
                    {
                        "question_id": "recon-backend-1",
                        "answer": "Authentication is applied once, globally.",
                        "repository_id": "backend",
                        "evidence_paths": ["backend/src"],
                        "confidence": "high",
                    }
                ]
            }
        )
    )
    agent = RepositoryReconAgent(prompt_loader=PromptLoader(), llm_client=cast(Any, llm_client))

    grounded = await agent.suggest_answers(
        feature_id="feature-grounded",
        questions=[
            ClarificationQuestion(
                question_id="recon-backend-1",
                question="Rely on the global middleware, or introduce per-route auth?",
                rationale="The premise contradicts the checkout.",
                required=True,
            )
        ],
        reconnaissance=[recon],
    )

    # The call was made -- the adapter contract did not refuse it -- and the payload it
    # carried is the evidence, not an empty string.
    assert len(llm_client.inputs) == 1
    payload = _json.loads(llm_client.inputs[0])
    assert [item["question_id"] for item in payload["open_questions"]] == ["recon-backend-1"]
    assert payload["reconnaissance"][0]["repository_id"] == "backend"
    # And the repository-grounded answer round-tripped onto the question.
    assert grounded[0].suggested_answer
    assert "globally" in grounded[0].suggested_answer
