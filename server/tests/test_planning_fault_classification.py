"""A deterministic provider answer fails once, with its reason -- and the clocks now exist.

AB-Feature-172 died on an HTTP 400 that returns the identical answer every time, spent
2 in-stage retries x 3 queue attempts proving it, and recorded only "the model provider
call failed (BadRequestError)" while the actual fix -- ``anthropic-workspace-id is
required`` -- was discarded at the adapter boundary. Hotfix a3e8119 stopped the retries;
these tests pin what 41- Part B added on top: the provider's own sentence carried redacted
and bounded into the diagnostics, and wall-versus-charged clocks on the pre-coding stages
so 173's 37 minutes of ReadTimeout can never again pass unattributed.

The existing adapter tests assert transient faults carry no provider text at all; that
stays exactly as it was, and is re-asserted here beside the deterministic exception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import select

import workflows.feature_workflow as feature_workflow_module
from adapters.llm_adapter import LLMAdapterError, _provider_faults_declared
from agents.shared.contracts import safe_error_diagnostics
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from state.enums import FeatureWorkflowStatus
from storage.db import Database
from storage.feature_store import SqlAlchemyFeatureControlPlane
from storage.models import FeatureExecutionQueueModel, FeatureWorkflowEventModel
from tests.support import drain_feature_queue
from tests.test_feature_api import feature_payload
from tests.test_feature_workflow import ProviderFaultThenPlanPlanner
from tests.test_feature_workflow import feature_payload as workflow_feature_payload
from workflows.feature_workflow import FeatureWorkflowOrchestrator

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)

# The provider sentence 172's operator had to learn from a manual curl.
_PROVIDER_SENTENCE = (
    "anthropic-workspace-id is required when authenticating with an identity-linked key"
)


class BadRequestError(Exception):
    """The SDK's own 4xx type name, which is what the adapter classifies by."""


class APIConnectionError(Exception):
    """The SDK's transient shape, whose text must stay uncarried."""


def _adapter_fault(error: Exception) -> LLMAdapterError:
    """Raise one SDK error through the adapter's declaration boundary, as a real call does."""
    with pytest.raises(LLMAdapterError) as raised, _provider_faults_declared():
        raise error
    return raised.value


def test_a_deterministic_400_carries_the_providers_sentence_redacted() -> None:
    """The sentence that names the fix survives; the token beside it does not."""
    fault = _adapter_fault(
        BadRequestError(f"{_PROVIDER_SENTENCE} (request used sk-live-a1b2c3d4e5f6)")
    )

    diagnostics = safe_error_diagnostics(fault)
    assert diagnostics[0] == "the model provider call failed (BadRequestError)"
    assert len(diagnostics) == 2
    assert _PROVIDER_SENTENCE in diagnostics[1]
    assert "sk-live-a1b2c3d4e5f6" not in " ".join(diagnostics)
    assert "[REDACTED]" in diagnostics[1]
    assert fault.failure_classification == "BadRequestError"


def test_a_deterministic_400_detail_is_bounded_and_flattened() -> None:
    """A provider message that embeds a payload cannot flood the durable record."""
    fault = _adapter_fault(BadRequestError("first line\nsecond line " + "x" * 10_000))

    detail = safe_error_diagnostics(fault)[1]
    assert "\n" not in detail
    assert "first line second line" in detail
    assert len(detail) < 700
    assert "[truncated]" in detail


def test_a_transient_fault_still_carries_no_provider_text() -> None:
    """172's retry machinery was right; so is the discard for faults a retry can outlive."""
    fault = _adapter_fault(APIConnectionError("connection reset by https://internal.host/api/x"))

    assert safe_error_diagnostics(fault) == ("the model provider call failed (APIConnectionError)",)
    assert "internal.host" not in str(fault)


@pytest.mark.asyncio
async def test_a_scripted_400_fails_the_feature_on_the_first_call_with_the_reason(
    tmp_path: Path,
) -> None:
    """One queue attempt, seconds not minutes, and the fix named in the durable record.

    172's measured baseline: nine identical calls across three queue attempts in 83
    seconds, then a record whose only content was the classification token.
    """

    class RefusedProductManager:
        calls = 0

        async def create_technical_prd(self, **_kwargs: Any) -> Any:
            type(self).calls += 1
            raise _adapter_fault(BadRequestError(_PROVIDER_SENTENCE))

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'deterministic.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(
            database,
            mock_runner=FeatureWorkflowOrchestrator(
                product_manager=cast(Any, RefusedProductManager())
            ),
        )
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="deterministic-400",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)

        assert RefusedProductManager.calls == 1, (
            "a deterministic answer must not be asked for again on any layer"
        )
        async with database.session() as session:
            queue_row = (
                await session.scalars(
                    select(FeatureExecutionQueueModel).where(
                        FeatureExecutionQueueModel.feature_id == "feature-login"
                    )
                )
            ).one()
            events = list(
                await session.scalars(
                    select(FeatureWorkflowEventModel)
                    .where(FeatureWorkflowEventModel.feature_id == "feature-login")
                    .order_by(FeatureWorkflowEventModel.timestamp)
                )
            )
        assert queue_row.attempt == 1, "the queue's remaining attempts must stay unspent"

        record = await store.get_record("feature-login")
        summary = record.state.failure_summary
        assert summary is not None
        assert any(_PROVIDER_SENTENCE in item for item in summary.diagnostics), (
            f"the provider's sentence must reach the summary, got: {summary.diagnostics}"
        )
        failure_events = [item for item in events if item.event == "feature_execution_failed"]
        assert failure_events, "the failure must be on the timeline"
        details = failure_events[-1].details
        assert any(_PROVIDER_SENTENCE in item for item in details.get("diagnostics", [])), (
            f"the provider's sentence must reach the timeline event, got: {details}"
        )
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_transient_fault_time_lands_on_the_wall_vs_charged_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retries behave exactly as before, and their cost stops being invisible."""
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.01)
    state = _initial_feature_state(
        "feature-clocked-faults", StartFeatureRequest.model_validate(workflow_feature_payload())
    )
    planner = ProviderFaultThenPlanPlanner(faults=2)
    orchestrator = FeatureWorkflowOrchestrator(planner=cast(Any, planner))

    result = await orchestrator.start(state, credentials=CREDENTIALS)

    assert result.status is FeatureWorkflowStatus.COMPLETED
    assert planner.calls == 3, "the retry behaviour itself is unchanged"
    assert result.planning_wall_seconds is not None
    assert result.planning_provider_fault_seconds is not None
    # Two faults at 0.01 and 0.02 seconds of backoff plus their stalls.
    assert result.planning_provider_fault_seconds >= 0.03
    assert result.planning_wall_seconds >= result.planning_provider_fault_seconds
    charged = result.planning_wall_seconds - result.planning_provider_fault_seconds
    assert charged >= 0.0


@pytest.mark.asyncio
async def test_a_clean_planning_run_records_wall_time_and_no_fault_time() -> None:
    """The clocks exist on every run, so absence of faults is a measurement too."""
    state = _initial_feature_state(
        "feature-clocked-clean", StartFeatureRequest.model_validate(workflow_feature_payload())
    )
    orchestrator = FeatureWorkflowOrchestrator()

    result = await orchestrator.start(state, credentials=CREDENTIALS)

    assert result.status is FeatureWorkflowStatus.COMPLETED
    assert result.planning_wall_seconds is not None
    assert result.planning_wall_seconds > 0.0
    assert result.planning_provider_fault_seconds == 0.0
