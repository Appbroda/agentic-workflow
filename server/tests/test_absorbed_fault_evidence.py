"""Absorbed provider faults reach the attempt's result artifact.

AB-Feature-186's attempt-0 record carried only `runtime_wall_seconds` versus
`runtime_charged_seconds`; the fault class, the count and the degraded-retry marker existed
nowhere in the artifact, so the wall-vs-charged difference could not be explained without
the server logs. These tests pin todo item 1's fix: the child loop folds its own fault
history into the result metadata at build time -- and a fault-free attempt records zeros
and false explicitly, because absence must keep meaning "recorded before this shipped".
"""

from __future__ import annotations

from typing import Any

import pytest

import workflows.feature_workflow as feature_workflow_module
from adapters.llm_adapter import LLMAdapterError
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import ChildWorkflowResultArtifact
from state.enums import ChildWorkflowStatus
from storage.external_operation_store import ExternalOperationError
from tests.test_model_routing import feature_payload, no_credentials, router
from workflows.feature_workflow import (
    ChildExecution,
    FeatureWorkflowOrchestrator,
    MockChildWorkstreamExecutor,
)

pytestmark = pytest.mark.asyncio


class TruncationThenFaultThenApproveExecutor:
    """186's shape: one wrapped truncation, one transient ReadError fault, then approval.

    The truncation arrives exactly as the live coding call delivers it -- an
    `LLMAdapterError` carrying `response_truncated`, wrapped in `ExternalOperationError` --
    and is answered by the degraded retry, never counted as a fault. The ReadError is the
    classified transient fault the loop absorbs and excludes from the charged clock.
    """

    def __init__(self) -> None:
        self._delegate = MockChildWorkstreamExecutor()
        self.calls = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        if kwargs["repository"].repository_id != "backend":
            return await self._delegate.run(**kwargs)
        self.calls += 1
        if self.calls == 1:
            msg = "the model response reached its configured max_tokens bound and is truncated"
            truncated = LLMAdapterError(
                msg, diagnostics=(msg,), failure_classification="response_truncated"
            )
            wrapper_msg = "external operation failed before a confirmed result"
            raise ExternalOperationError(wrapper_msg) from truncated
        if self.calls == 2:
            fault_msg = "provider connection dropped mid-response"
            raise LLMAdapterError(
                fault_msg, diagnostics=(fault_msg,), failure_classification="ReadError"
            )
        return await self._delegate.run(**kwargs)


def _backend_result_metadata(artifacts: list[Any]) -> dict[str, Any]:
    """The final backend result artifact's metadata, where the evidence has to land."""
    results = [
        item
        for item in artifacts
        if isinstance(item, ChildWorkflowResultArtifact) and item.repository_id == "backend"
    ]
    assert results, "the backend recorded no child result artifact"
    return dict(results[-1].metadata)


async def test_an_attempts_fault_history_lands_on_its_result_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3a: one degraded truncation retry and one absorbed ReadError are all recorded."""
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.05)
    request = StartFeatureRequest.model_validate(feature_payload("feature-fault-evidence"))
    state = _initial_feature_state("feature-fault-evidence", request)
    executor = TruncationThenFaultThenApproveExecutor()

    result = await FeatureWorkflowOrchestrator(
        child_executor=executor, model_router=router()
    ).start(state, credentials=no_credentials())

    assert executor.calls == 3
    assert result.child_workflows["backend"].status in {
        ChildWorkflowStatus.APPROVED,
        ChildWorkflowStatus.COMPLETED,
    }
    metadata = _backend_result_metadata(list(result.artifacts))
    # The ReadError is a fault; the truncation is a degraded retry and is deliberately not.
    assert metadata["fault_count"] == 1
    assert metadata["fault_classes"] == ["LLMAdapterError/ReadError"]
    # The stalled attempt plus its backoff were excluded from the charged clock, and the
    # exclusion is now a recorded fact rather than a reconstruction from journal timestamps.
    assert metadata["fault_seconds_excluded"] >= 0.05
    assert metadata["truncated_degraded_retry"] is True
    wall = metadata["runtime_wall_seconds"]
    charged = metadata["runtime_charged_seconds"]
    assert wall is not None and charged is not None
    assert wall - charged == pytest.approx(metadata["fault_seconds_excluded"], abs=0.05)


async def test_a_fault_free_attempt_records_zeros_and_false_explicitly() -> None:
    """3b: zeros are written, not implied -- absence stays 'recorded before this shipped'."""
    request = StartFeatureRequest.model_validate(feature_payload("feature-no-faults"))
    state = _initial_feature_state("feature-no-faults", request)

    result = await FeatureWorkflowOrchestrator(
        child_executor=MockChildWorkstreamExecutor(), model_router=router()
    ).start(state, credentials=no_credentials())

    assert result.child_workflows["backend"].status in {
        ChildWorkflowStatus.APPROVED,
        ChildWorkflowStatus.COMPLETED,
    }
    metadata = _backend_result_metadata(list(result.artifacts))
    assert metadata["fault_count"] == 0
    assert metadata["fault_seconds_excluded"] == 0.0
    assert metadata["fault_classes"] == []
    assert metadata["truncated_degraded_retry"] is False


# --------------------------------------------------------------------------------------
# 85- F1: the two clocks describe the attempt they are recorded on
# --------------------------------------------------------------------------------------


class _ClockedRejectingExecutor:
    """Reject the backend twice, charging a different, known duration to each attempt.

    Time is advanced by a fake monotonic clock rather than by sleeping, the precedent
    `test_runtime_limits` sets: these are wall-clock assertions, and a test that waited for
    them is not a test anybody runs.
    """

    def __init__(self, seconds_per_attempt: list[float]) -> None:
        """Charge one duration per backend attempt, in order."""
        self._inner = MockChildWorkstreamExecutor()
        self._seconds = list(seconds_per_attempt)
        self.elapsed = 0.0
        self.attempts = 0

    def monotonic(self) -> float:
        """Stand in for `time.monotonic`, advancing only when the backend runs."""
        return self.elapsed

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Charge this attempt's duration, then reject it with a genuinely changed input."""
        if kwargs["repository"].repository_id != "backend":
            return await self._inner.run(**kwargs)
        self.attempts += 1
        index = self.attempts
        if index <= len(self._seconds):
            self.elapsed += self._seconds[index - 1]
        execution = await self._inner.run(**kwargs)
        if index > len(self._seconds):
            return execution
        return ChildExecution(
            result=execution.result.model_copy(
                update={
                    "status": "failed",
                    "pull_request_readiness": False,
                    "current_revision": f"backend-revision-{index}",
                    "production_files_changed": [f"src/module_{index}.py"],
                    "production_diff_fingerprint": f"backend-fingerprint-{index}",
                    "blocking_issues": [f"The reviewer asked for change {index}."],
                }
            ),
            code_completion=execution.code_completion,
            review=execution.review,
        )


def _backend_result_walls(artifacts: list[Any]) -> list[tuple[float, float]]:
    """Every backend attempt's `(wall, charged)` pair, in the order they were recorded."""
    return [
        (item.metadata["runtime_wall_seconds"], item.metadata["runtime_charged_seconds"])
        for item in artifacts
        if isinstance(item, ChildWorkflowResultArtifact) and item.repository_id == "backend"
    ]


async def test_each_attempts_clocks_describe_that_attempt_and_not_the_workstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`started_at` was set before the retry loop, so attempt N's clock was cumulative.

    Summing AB-Feature-218's six recorded backend values gave 9,543 s for a workstream that
    ran 4,952 s -- because every reader sums these as attempt durations, and they were not.
    """
    executor = _ClockedRejectingExecutor([100.0, 250.0])
    monkeypatch.setattr("workflows.feature_workflow.time.monotonic", executor.monotonic)
    request = StartFeatureRequest.model_validate(feature_payload("feature-attempt-clocks"))
    state = _initial_feature_state("feature-attempt-clocks", request)

    result = await FeatureWorkflowOrchestrator(
        child_executor=executor, model_router=router()
    ).start(state, credentials=no_credentials())

    walls = _backend_result_walls(list(result.artifacts))
    assert executor.attempts >= 3
    assert len(walls) >= 2
    # Each attempt's own duration. Cumulative would have made the second 350.
    assert walls[0][0] == pytest.approx(100.0)
    assert walls[1][0] == pytest.approx(250.0)
    # Fault-free, so the two clocks agree; what matters is that neither accumulated.
    assert walls[0][1] == pytest.approx(100.0)
    assert walls[1][1] == pytest.approx(250.0)


class _RejectOnceThenFaultForever:
    """218's terminal shape: one completed attempt, then a provider outage that never ends.

    The completed attempt is what makes the defect reproducible. It writes clocks onto the
    child row, and the terminal record was then built from that row -- so the fault-terminal
    attempt reported the earlier attempt's numbers while claiming to have excluded an
    outage of its own.
    """

    def __init__(self, *, first_attempt_seconds: float, seconds_per_fault: float) -> None:
        """Charge one duration to the completed attempt and one to each faulting try."""
        self._inner = MockChildWorkstreamExecutor()
        self._first = first_attempt_seconds
        self._per_fault = seconds_per_fault
        self.elapsed = 0.0
        self.calls = 0

    def monotonic(self) -> float:
        """Stand in for `time.monotonic`, advancing only when the backend runs."""
        return self.elapsed

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Reject attempt one, then fault every later try past the consecutive ceiling."""
        if kwargs["repository"].repository_id != "backend":
            return await self._inner.run(**kwargs)
        self.calls += 1
        if self.calls == 1:
            self.elapsed += self._first
            execution = await self._inner.run(**kwargs)
            return ChildExecution(
                result=execution.result.model_copy(
                    update={
                        "status": "failed",
                        "pull_request_readiness": False,
                        "current_revision": "backend-revision-1",
                        "production_files_changed": ["src/module_1.py"],
                        "production_diff_fingerprint": "backend-fingerprint-1",
                        "blocking_issues": ["The reviewer asked for change 1."],
                    }
                ),
                code_completion=execution.code_completion,
                review=execution.review,
            )
        self.elapsed += self._per_fault
        message = "provider connection dropped mid-response"
        raise LLMAdapterError(message, diagnostics=(message,), failure_classification="ReadError")


async def test_a_fault_terminal_attempt_records_its_own_measured_clocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """218's attempt 6 recorded attempt 5's numbers, byte for byte.

    `runtime_wall_seconds == runtime_charged_seconds == 2391.89`, byte-identical to attempt
    5's, while reporting `fault_seconds_excluded: 2523.222`. `_with_fault_evidence` filled
    the clocks "only where the result does not already carry a value", and on this path
    there is no later measurement to defer to -- the terminal result is built from a child
    row that already carries the PREVIOUS attempt's values. So the record simultaneously
    claimed it had excluded a 42-minute outage and that its charged clock equalled its wall
    clock, with both numbers describing a different, earlier attempt.
    """
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 0.0)
    executor = _RejectOnceThenFaultForever(first_attempt_seconds=100.0, seconds_per_fault=400.0)
    monkeypatch.setattr("workflows.feature_workflow.time.monotonic", executor.monotonic)
    request = StartFeatureRequest.model_validate(feature_payload("feature-fault-terminal"))
    state = _initial_feature_state("feature-fault-terminal", request)

    result = await FeatureWorkflowOrchestrator(
        child_executor=executor, model_router=router()
    ).start(state, credentials=no_credentials())

    metadata = _backend_result_metadata(list(result.artifacts))
    wall = metadata["runtime_wall_seconds"]
    charged = metadata["runtime_charged_seconds"]
    excluded = metadata["fault_seconds_excluded"]
    assert metadata["fault_count"] > feature_workflow_module._ALLOWED_INFRASTRUCTURE_FAULTS
    # Not the completed attempt's 100 s, which is what the deferred fill inherited.
    assert wall != pytest.approx(100.0)
    # The invariant the record used to contradict, on the one path where nothing else
    # measures afterwards.
    assert wall - excluded == pytest.approx(charged, abs=0.01)
    assert charged >= 0.0
