"""An unconfirmed effect gets its retry budget, and the terminal story is the last true one.

D3, verified against the deployed database on 2026-09-03: the cross-link comment failed 16
of 17 times window-wide, every row ``attempt=1, max_attempts=1`` -- ``add_comment`` was the
one sibling in its file with no reconcile and no budget, and nothing re-drove a failed row
anyway. ``run_clarification_grounding`` failed in all six of runs 175-180 to one two-minute
provider blip, because the planning wrapper journals one attempt per call and its per-call
nonce means a journal-level budget could never retry it.

D8, verified the same day: run 199's persisted story still read "The model provider did not
answer" for a run that ended 87 minutes and six attempts later in a design-conflict stop,
because a retry grant -- unlike a resume -- never cleared the write-once failure summary.

The assertions here are about effects: what the provider ends up holding, what the journal
row says, and what a human is or is not asked -- never which calls were issued.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

import services.journaled_github as journaled_github_module
import workflows.feature_workflow as feature_workflow_module
from adapters.github_adapter import GitHubAdapterError, MockGitHubService
from adapters.llm_adapter import LLMAdapterError
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import TechnicalPRDArtifact
from services.cancellation import MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.journaled_github import JournaledGitHubService
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.external_operations import ExternalOperationStatus
from state.failure_diagnosis import FailureStage, FeatureFailureClassification
from state.feature_models import FeatureFailureSummary, ensure_feature_failure_summary
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from storage.feature_store import SqlAlchemyFeatureControlPlane
from tests.support import drain_feature_queue
from tests.test_clarification_suggestions import (
    _GLOBAL_AUTH_PREMISE,
    AskingProductManager,
    GroundingReconnaissance,
)
from tests.test_feature_workflow import feature_payload
from tests.test_repository_preflight_and_retry import ControlledChildExecutor
from workflows.feature_workflow import FeatureWorkflowOrchestrator

pytestmark = pytest.mark.asyncio

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)
REPOSITORY = "example/platform"
COMMENT = "Related pull requests:\n- example/platform: https://github.invalid/pr/1"


def _comment_status_fault(status: int) -> GitHubAdapterError:
    """The fault the live adapter raises when GitHub reports one HTTP status."""
    message = f"pull-request comment could not be posted (status {status})"
    return GitHubAdapterError(
        message,
        diagnostics=(message,),
        failure_classification=f"pull_request_comment_status_{status}",
        provider_status=status,
    )


class _AnswerNeverArrives(MockGitHubService):
    """The first post dies on a 502: the effect never landed on the provider."""

    def __init__(self) -> None:
        super().__init__()
        self.failures_left = 1
        self.post_calls = 0

    def add_comment(self, repository: str, pull_request_number: int, body: str) -> None:
        self.post_calls += 1
        if self.failures_left:
            self.failures_left -= 1
            raise _comment_status_fault(502)
        super().add_comment(repository, pull_request_number, body)


class _AnswerLostAfterAcceptance(MockGitHubService):
    """The provider applies the comment; the answer is lost on the way back."""

    def __init__(self) -> None:
        super().__init__()
        self.failures_left = 1
        self.post_calls = 0

    def add_comment(self, repository: str, pull_request_number: int, body: str) -> None:
        self.post_calls += 1
        super().add_comment(repository, pull_request_number, body)
        if self.failures_left:
            self.failures_left -= 1
            raise _comment_status_fault(502)


class _RefusesEveryPost(MockGitHubService):
    """Refuse the way the sixteen recorded failures were refused: a 403 permission answer."""

    def __init__(self) -> None:
        super().__init__()
        self.post_calls = 0

    def add_comment(self, repository: str, pull_request_number: int, body: str) -> None:
        self.post_calls += 1
        raise _comment_status_fault(403)


def _with_pull_request[ProviderT: MockGitHubService](provider: ProviderT) -> ProviderT:
    """A provider already holding the pull request the comment belongs to."""
    provider.create_pull_request(
        REPOSITORY,
        title="[AI] feature",
        body="Feature ID: feature-comment",
        source_branch="ai/feature",
        target_branch="main",
    )
    return provider


async def _journaled(
    tmp_path: Path, provider: Any, name: str
) -> tuple[Database, ExternalOperationJournal, JournaledGitHubService]:
    """The journaled service exactly as the deployment composes it."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / f'{name}.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    service = JournaledGitHubService(
        provider,
        operation_executor=ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(workflow_id=name, feature_id=name),
        ),
    )
    return database, journal, service


async def test_a_transient_comment_fault_earns_the_retry_and_posts_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One 502, one retry, one comment -- and the operation record says attempt two."""
    monkeypatch.setattr(journaled_github_module, "_COMMENT_FAULT_BACKOFF_SECONDS", 0.0)
    provider = _with_pull_request(_AnswerNeverArrives())
    database, journal, service = await _journaled(tmp_path, provider, "comment-weather")
    try:
        await service.add_comment(REPOSITORY, 1, COMMENT)

        assert provider.comments[(REPOSITORY, 1)] == [COMMENT]
        (operation,) = await journal.list_operations_for_workflow("comment-weather")
        assert operation.status is ExternalOperationStatus.SUCCEEDED
        assert operation.attempt == 2
    finally:
        await database.dispose()


async def test_a_comment_that_landed_with_its_answer_lost_is_never_posted_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reconciliation proves the effect landed and records success without a second post."""
    monkeypatch.setattr(journaled_github_module, "_COMMENT_FAULT_BACKOFF_SECONDS", 0.0)
    provider = _with_pull_request(_AnswerLostAfterAcceptance())
    database, journal, service = await _journaled(tmp_path, provider, "comment-lost-answer")
    try:
        await service.add_comment(REPOSITORY, 1, COMMENT)

        assert provider.comments[(REPOSITORY, 1)] == [COMMENT]
        assert provider.post_calls == 1
        (operation,) = await journal.list_operations_for_workflow("comment-lost-answer")
        assert operation.status is ExternalOperationStatus.SUCCEEDED
        assert (operation.result_payload or {})["recovery_method"] == "comment_present"
    finally:
        await database.dispose()


async def test_a_deterministic_refusal_spends_nothing(tmp_path: Path) -> None:
    """A 403 answers the same way every time, so it fails once, terminally, and says why."""
    provider = _with_pull_request(_RefusesEveryPost())
    database, journal, service = await _journaled(tmp_path, provider, "comment-refused")
    try:
        with pytest.raises(GitHubAdapterError):
            await service.add_comment(REPOSITORY, 1, COMMENT)

        assert provider.post_calls == 1, "the predicate refused it; no retry may happen"
        assert provider.comments[(REPOSITORY, 1)] == []
        (operation,) = await journal.list_operations_for_workflow("comment-refused")
        assert operation.status is ExternalOperationStatus.FAILED_TERMINAL
        assert operation.attempt == 1
        assert "GitHubAdapterError/pull_request_comment_status_403" in (
            operation.error_message or ""
        )
    finally:
        await database.dispose()


class _GroundingFaultsThenAnswers(GroundingReconnaissance):
    """Drop the connection a fixed number of times, then answer as the scripted model does."""

    def __init__(self, *, faults: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._faults = faults
        self.calls = 0

    async def suggest_answers(self, **kwargs: Any) -> list[Any]:
        self.calls += 1
        if self._faults > 0:
            self._faults -= 1
            msg = "the model provider call failed (APIConnectionError)"
            raise LLMAdapterError(msg, diagnostics=(msg,))
        return await super().suggest_answers(**kwargs)


_GROUNDED_ANSWER = {
    "question_id": "UQ-001",
    "answer": "Deletion is a hard delete; there is no retention path.",
    "repository_id": "backend",
    "evidence_paths": ["backend/src"],
    "confidence": "high",
}


def _grounding_orchestrator(reconnaissance: Any) -> FeatureWorkflowOrchestrator:
    """A feature whose product manager asks one question the checkouts can answer."""
    return FeatureWorkflowOrchestrator(
        reconnaissance=cast(Any, reconnaissance),
        product_manager=cast(
            Any, AskingProductManager("UQ-001", "Does deleting an app remove it permanently?")
        ),
    )


def _grounded_question(waiting: Any) -> Any:
    """The product manager's question, as the newest Technical PRD will ask it."""
    technical_prd = next(
        item for item in reversed(waiting.artifacts) if isinstance(item, TechnicalPRDArtifact)
    )
    return next(item for item in technical_prd.unresolved_questions if item.question_id == "UQ-001")


async def test_grounding_survives_one_provider_blip(monkeypatch: pytest.MonkeyPatch) -> None:
    """The six runs of 175-180 were one retry away from answering their own questions."""
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)
    reconnaissance = _GroundingFaultsThenAnswers(
        faults=1,
        contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]},
        answers=[_GROUNDED_ANSWER],
    )
    state = _initial_feature_state(
        "feature-grounding-blip", StartFeatureRequest.model_validate(feature_payload())
    )

    waiting = await _grounding_orchestrator(reconnaissance).start(state, credentials=CREDENTIALS)

    assert reconnaissance.calls == 2
    assert waiting.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
    assert waiting.clarification_grounding_failed is False, "this ask is a grounded ask"
    assert _grounded_question(waiting).suggested_answer == _GROUNDED_ANSWER["answer"]


async def test_grounding_exhaustion_falls_back_to_the_human_exactly_as_today(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback is unchanged, just later: the flag still means "we tried and could not"."""
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)
    reconnaissance = _GroundingFaultsThenAnswers(
        faults=99,
        contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]},
        answers=[_GROUNDED_ANSWER],
    )
    state = _initial_feature_state(
        "feature-grounding-down", StartFeatureRequest.model_validate(feature_payload())
    )

    waiting = await _grounding_orchestrator(reconnaissance).start(state, credentials=CREDENTIALS)

    allowed = 1 + feature_workflow_module._ALLOWED_FEATURE_STAGE_INFRASTRUCTURE_FAULTS  # noqa: SLF001
    assert reconnaissance.calls == allowed, "the retries are spent before anyone is asked"
    assert waiting.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN, "never gated"
    assert waiting.clarification_grounding_failed is True
    assert _grounded_question(waiting).suggested_answer == "", "the ask is exactly as it was"


async def test_both_retry_paths_consult_the_one_retryability_predicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seam-level: the GitHub driver and the planning wrapper ask the same single function."""
    consulted: list[str] = []
    genuine = feature_workflow_module.is_transient_provider_fault

    def recording(error: BaseException) -> bool:
        consulted.append(type(error).__name__)
        return genuine(error)

    monkeypatch.setattr(feature_workflow_module, "is_transient_provider_fault", recording)
    monkeypatch.setattr(journaled_github_module, "is_transient_provider_fault", recording)
    monkeypatch.setattr(journaled_github_module, "_COMMENT_FAULT_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)

    provider = _with_pull_request(_AnswerNeverArrives())
    database, _journal, service = await _journaled(tmp_path, provider, "one-predicate")
    try:
        await service.add_comment(REPOSITORY, 1, COMMENT)
    finally:
        await database.dispose()
    assert "GitHubAdapterError" in consulted

    reconnaissance = _GroundingFaultsThenAnswers(
        faults=1,
        contradicted_premises={"backend": [_GLOBAL_AUTH_PREMISE]},
        answers=[_GROUNDED_ANSWER],
    )
    state = _initial_feature_state(
        "feature-one-predicate", StartFeatureRequest.model_validate(feature_payload())
    )
    await _grounding_orchestrator(reconnaissance).start(state, credentials=CREDENTIALS)
    assert "LLMAdapterError" in consulted


# --------------------------------------------------------------------------------------
# Part B -- the terminal story is the last true one
# --------------------------------------------------------------------------------------


def _fossil_summary() -> FeatureFailureSummary:
    """The diagnosis run 199 carried into a stop it never described."""
    return FeatureFailureSummary(
        stage=FailureStage.CHILD_WORKFLOWS.value,
        agent="child_workstream",
        repository_id="backend",
        root_classification="provider_unavailable",
        diagnostics=[
            "The model provider did not answer (LLMAdapterError) while this repository was running."
        ],
        retryable=True,
        next_action="Confirm the model provider is answering, then resume this feature.",
        recorded_at=datetime.now(UTC) - timedelta(hours=2),
    )


async def test_the_199_replay_leaves_the_second_stop_as_the_persisted_story(
    tmp_path: Path,
) -> None:
    """Terminal with a provider summary, retry granted, second terminal: the story moves on.

    Run 199's exact shape: the fossil said "the model provider did not answer", the retry
    ran four more attempts, and the write-once contract preserved the fossil into a stop it
    never described. The grant now clears the snapshot's summary the way resume always has,
    so the second terminal records its own cause -- while the first stop stays in the
    append-only timeline.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'terminal-story.db'}")
    await database.create_schema()
    try:
        request = StartFeatureRequest.model_validate(feature_payload()).model_copy(
            update={"feature_id": "terminal-story"}
        )
        store = SqlAlchemyFeatureControlPlane(
            database,
            mock_runner=FeatureWorkflowOrchestrator(
                child_executor=cast(Any, ControlledChildExecutor())
            ),
        )
        await store.start(
            request,
            idempotency_key="terminal-story",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        record = await store.get_record("terminal-story")
        first = record.state.failure_summary
        assert first is not None, "the first stop must have recorded its own diagnosis"
        second_stop_classification = first.root_classification

        # Model the fossil exactly as 199 held it: a provider-outage diagnosis recorded long
        # before the retry this test grants. Precondition-building, not behaviour.
        fossil_recorded_at = datetime.now(UTC) - timedelta(hours=2)
        state = record.state
        state.failure_summary = first.model_copy(
            update={
                "root_classification": "provider_unavailable",
                "retryable": True,
                "recorded_at": fossil_recorded_at,
                "diagnostics": [
                    "The model provider did not answer (LLMAdapterError) while this "
                    "repository was running."
                ],
            }
        )
        await store._replace_state(  # noqa: SLF001 - building the precondition, not the behaviour
            "terminal-story", state, "feature_failed_requires_human"
        )

        granted_at = datetime.now(UTC)
        await store.retry_workstream(
            "terminal-story",
            "backend",
            additional_attempts=1,
            requested_by="akhilesh",
            reason="The provider is healthy again; the repository never was the problem.",
            credentials=CREDENTIALS,
        )
        await drain_feature_queue(store)

        record = await store.get_record("terminal-story")
        summary = record.state.failure_summary
        assert summary is not None, "the second stop must record a diagnosis of its own"
        assert summary.root_classification == second_stop_classification
        assert summary.root_classification != "provider_unavailable"
        assert "provider did not answer" not in " ".join(summary.diagnostics).lower()
        assert summary.recorded_at >= granted_at, "recorded at the second terminal, not the first"
        # The previous stop is retained where history lives; only the active snapshot moved on.
        assert any(
            item.event == "feature_failed_requires_human" for item in record.lifecycle_events
        )
    finally:
        await database.drop_schema()
        await database.dispose()


async def test_write_once_within_one_termination_episode_survives() -> None:
    """A specific diagnosis recorded first is untouched by the generic caller after it."""
    state = _initial_feature_state(
        "feature-write-once", StartFeatureRequest.model_validate(feature_payload())
    )
    state.status = FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    specific = ensure_feature_failure_summary(
        state,
        stage=FailureStage.INTEGRATION_REVIEW,
        classification=FeatureFailureClassification.CONTRACT_MISMATCH,
        diagnostics=["A satisfied demand was demanded again; the disagreement is the demands'."],
    )
    recorded = specific.failure_summary
    assert recorded is not None
    assert recorded.root_classification == "contract_mismatch"

    generic = ensure_feature_failure_summary(specific)

    assert generic.failure_summary is not None
    assert generic.failure_summary.root_classification == "contract_mismatch"
    assert generic.failure_summary.recorded_at == recorded.recorded_at
    assert generic.failure_summary.diagnostics == recorded.diagnostics


async def test_the_retryable_resume_branch_still_reads_the_new_summary_after_a_retry() -> None:
    """Retry-then-fail writes a fresh diagnosis, and the empty-answer recovery still works.

    The clear must not change which resume branch a later operator lands in: after a granted
    retry fails again, whatever summary the operator's resume reads is the *second* stop's,
    and a retryable one still routes an empty-answer resume through checkpoint recovery.
    """
    executor = ControlledChildExecutor()
    state = _initial_feature_state(
        "feature-retry-resume", StartFeatureRequest.model_validate(feature_payload())
    )
    orchestrator = FeatureWorkflowOrchestrator(child_executor=cast(Any, executor))
    stopped = await orchestrator.start(state, credentials=CREDENTIALS)
    assert stopped.child_workflows["backend"].status is ChildWorkflowStatus.FAILED
    # The snapshot's summary is written at persistence; model the fossil the deployed record
    # holds -- the first stop's diagnosis, recorded well before the grant below.
    stopped.failure_summary = _fossil_summary()

    granted_at = datetime.now(UTC)
    granted = await orchestrator.retry_workstream(
        stopped,
        repository_id="backend",
        additional_attempts=1,
        requested_by="akhilesh",
        reason="Repaired the runner; the stop itself looked transient.",
        credentials=CREDENTIALS,
    )

    # The grant cleared the fossil: whatever the snapshot now carries describes the second
    # stop (or nothing, for the persistence layer to fill), never the pre-grant diagnosis.
    assert granted.child_workflows["backend"].status is ChildWorkflowStatus.FAILED
    second = granted.failure_summary
    assert second is None or second.recorded_at >= granted_at

    # A later provider fault parks the feature waiting with a retryable summary; the
    # empty-answer recovery branch must read that fresh diagnosis and clear it for the next.
    waiting = granted.model_copy(deep=True)
    waiting.status = FeatureWorkflowStatus.WAITING_FOR_HUMAN
    waiting.failure_summary = _fossil_summary().model_copy(
        update={"recorded_at": datetime.now(UTC)}
    )

    resumed = await orchestrator.begin_resume(waiting, answers=[])

    assert resumed.failure_summary is None, "a second failure must be free to record its cause"
