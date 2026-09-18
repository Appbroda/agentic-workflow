"""Blind planning is a state, not a log line.

AB-Feature-173's admanager repository was planned blind after a 37-minute ReadTimeout, and
the only record was one warning in `docker logs`. The feature's timeline, the workstream
record, and the UI said nothing; the operator found out by asking why an artifact was
missing. These tests pin 41- Part C.1: when the reconnaissance fail-soft fires, the fact is
recorded on the feature snapshot, emitted to the timeline, copied onto the repository's
workstream record, and survives the database round-trip the API reads from.

The fail-soft itself is untouched -- every test here ends with the feature still running,
because blind planning remains better than no feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest, WorkstreamResponse
from configs.settings import load_settings
from services.cancellation import MockCancellationToken
from services.feature_runtime import LiveChildWorkstreamExecutor
from state.enums import FeatureWorkflowStatus
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from storage.feature_store import SqlAlchemyFeatureControlPlane
from tests.test_feature_workflow import RecordingReconnaissance, feature_payload
from tests.test_repository_reconnaissance import build_repository
from workflows.feature_workflow import (
    BlindPlanningRecord,
    FeatureWorkflowOrchestrator,
    ReconnaissanceReport,
)

pytestmark = pytest.mark.asyncio

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)

_REASON = "the model provider call failed (ReadTimeout)"


class PartiallyBlindReconnaissance(RecordingReconnaissance):
    """Read every repository except one, exactly as 173's reconnaissance ended up doing."""

    def __init__(self, blind_repository: str) -> None:
        super().__init__()
        self._blind = blind_repository

    async def inspect(self, **kwargs: Any) -> ReconnaissanceReport:
        readable = [item for item in kwargs["repositories"] if item.repository_id != self._blind]
        report = await super().inspect(**{**kwargs, "repositories": readable})
        return ReconnaissanceReport(
            artifacts=report.artifacts,
            blind=[
                BlindPlanningRecord(
                    repository_id=self._blind, error_type="LLMAdapterError", reason=_REASON
                )
            ],
        )


class RecordingEventWriter:
    """Capture every timeline event the orchestrator writes mid-run."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    async def __call__(self, feature_id: str, event: str, details: dict[str, Any]) -> None:
        self.events.append((feature_id, event, details))


async def test_a_blind_repository_is_recorded_on_snapshot_timeline_and_workstream() -> None:
    """Three writes, one fact -- and the feature still runs to the end."""
    state = _initial_feature_state(
        "feature-blind", StartFeatureRequest.model_validate(feature_payload())
    )
    writer = RecordingEventWriter()
    orchestrator = FeatureWorkflowOrchestrator(
        reconnaissance=cast(Any, PartiallyBlindReconnaissance("backend")),
        event_writer=writer,
    )

    result = await orchestrator.start(state, credentials=CREDENTIALS)

    # The fail-soft is untouched: the feature ran through, blind repository included.
    assert result.status is FeatureWorkflowStatus.COMPLETED
    assert result.planned_blind_repositories == {"backend": _REASON}
    assert [(event, details["repository_id"]) for _, event, details in writer.events] == [
        ("repository_planned_blind", "backend")
    ]
    details = writer.events[0][2]
    assert details["reason"] == _REASON
    assert details["error_type"] == "LLMAdapterError"
    assert details["occurrence"] == "reconnaissance"

    backend = result.child_workflows["backend"]
    assert backend.planned_blind is True
    assert backend.planned_blind_reason == _REASON
    assert result.child_workflows["frontend"].planned_blind is False
    assert result.child_workflows["frontend"].planned_blind_reason is None


async def test_the_flag_survives_the_store_round_trip_and_the_workstreams_read(
    tmp_path: Path,
) -> None:
    """What the API serves comes back from rows, so the rows must carry the flag."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'blind.db'}")
    await database.create_schema()
    try:
        # The store binds writers onto the live runner; a test orchestrator receives one
        # through its constructor, so the store's durable writer is bound late here.
        bound: dict[str, SqlAlchemyFeatureControlPlane] = {}

        async def durable_writer(feature_id: str, event: str, details: dict[str, Any]) -> None:
            await bound["store"].record_feature_event(feature_id, event, details)

        store = SqlAlchemyFeatureControlPlane(
            database,
            mock_runner=FeatureWorkflowOrchestrator(
                reconnaissance=cast(Any, PartiallyBlindReconnaissance("backend")),
                event_writer=durable_writer,
            ),
        )
        bound["store"] = store
        started = await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="blind-round-trip",
            credentials=CREDENTIALS,
            owner_id="platform-admin",
        )
        feature_id = started.record.state.feature_id
        from tests.support import drain_feature_queue

        await drain_feature_queue(store)

        rehydrated = (await store.get_record(feature_id)).state
        assert rehydrated.planned_blind_repositories == {"backend": _REASON}
        assert rehydrated.child_workflows["backend"].planned_blind is True
        assert rehydrated.child_workflows["backend"].planned_blind_reason == _REASON

        workstreams = await store.workstreams(feature_id)
        by_repository = {item.repository_id: item for item in workstreams}
        serialized = WorkstreamResponse.model_validate(
            {
                **by_repository["backend"].model_dump(mode="python", exclude={"preflight_result"}),
                "available_actions": [],
            }
        )
        assert serialized.planned_blind is True
        assert serialized.planned_blind_reason == _REASON

        # The timeline has the event, durably, where an operator watches.
        events = await store.events_after(feature_id, after_id=None, limit=200)
        blind_events = [item for item in events if item.event == "repository_planned_blind"]
        assert len(blind_events) == 1
        assert blind_events[0].details["repository_id"] == "backend"
    finally:
        await database.dispose()


async def test_the_late_probe_recovers_registry_paths_from_a_checkout(tmp_path: Path) -> None:
    """The structural half of reconnaissance, run against the checkout the child now has."""
    checkout = tmp_path / "workspaces" / "checkout"
    checkout.parent.mkdir(parents=True)
    checkout.mkdir()
    build_repository(checkout)
    executor = LiveChildWorkstreamExecutor(
        settings=load_settings(workspace_root=tmp_path / "workspaces"),
        git_environment={},
        engineer_client=cast(Any, None),
        reviewer_client=cast(Any, None),
        journal=ExternalOperationJournal(Database(f"sqlite+aiosqlite:///{tmp_path / 'probe.db'}")),
        cancellation_token=MockCancellationToken(),
    )

    paths = await executor._late_reconnaissance_paths(  # noqa: SLF001
        feature_id="feature-blind", repository_id="backend", workspace=checkout
    )

    # The fixture's registry is the file that assembles its modules -- the exact evidence
    # the wiring gate exists to deliver and a blind plan lacks.
    assert "server/config/express.js" in paths


async def test_a_probe_failure_never_fails_the_attempt_and_is_recorded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second failure leaves the workstream exactly as blind as it was, recorded again."""
    import services.feature_runtime as feature_runtime_module

    def refuse(workspace: Path, *, repository_id: str) -> Any:
        raise OSError(f"unreadable checkout for {repository_id}: {workspace}")

    monkeypatch.setattr(feature_runtime_module, "inspect_repository_for_planning", refuse)
    writer = RecordingEventWriter()
    executor = LiveChildWorkstreamExecutor(
        settings=load_settings(workspace_root=tmp_path / "workspaces"),
        git_environment={},
        engineer_client=cast(Any, None),
        reviewer_client=cast(Any, None),
        journal=ExternalOperationJournal(
            Database(f"sqlite+aiosqlite:///{tmp_path / 'probe-fail.db'}")
        ),
        cancellation_token=MockCancellationToken(),
        event_writer=writer,
    )

    paths = await executor._late_reconnaissance_paths(  # noqa: SLF001
        feature_id="feature-blind", repository_id="backend", workspace=tmp_path / "missing"
    )

    assert paths == []
    assert [(event, details["occurrence"]) for _, event, details in writer.events] == [
        ("repository_planned_blind", "late_reconnaissance")
    ]
