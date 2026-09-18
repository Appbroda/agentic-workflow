"""A blind-planned repository's first attempt gives itself its evidence back.

By the time a child workstream starts, the checkout exists and is already being read -- so
when the plan for its repository was written blind, the structural reconnaissance probe
runs against that checkout before the Engineer is called, and its wiring files flow where
recon results already flow: `registry_paths`, into the Engineer's delivered prompt. The
wiring-gate history is the justification -- 24% of first attempts shipped unreachable code
before 35-, and `registry_paths` is precisely what a blind repository's Engineer lacks.

Real tier: a real checkout, the real executor, the real prompt bytes. The assertions read
the paths back out of what the coding model was actually handed (the `_delivered_paths`
discipline from 35-), never out of the platform's own account of it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.real_repositories import (
    missing_executables,
    node_npm_repository,
    requires_real_repository_tier,
    unavailable_reason,
)
from tests.real_repository_support import build_harness, review_payload, run_child
from tests.test_live_child_executor import ScriptedLLMClient
from tests.test_real_repository_gates import _route_change
from tools.repository_reconnaissance import inspect_repository_for_planning

pytestmark = [pytest.mark.asyncio, pytest.mark.realrepo]


@pytest.fixture(autouse=True)
def _toolchain() -> None:
    """Refuse to certify anything on a machine that cannot run these checkouts."""
    missing = missing_executables()
    if not missing:
        return
    if requires_real_repository_tier():
        pytest.fail(unavailable_reason(missing), pytrace=False)
    pytest.skip(unavailable_reason(missing))


def _delivered_paths(engineer_client: ScriptedLLMClient) -> list[str]:
    """The files whose contents the coding model actually received, read from the prompt."""
    _instructions, input_text = engineer_client.calls[0]
    snapshot = json.loads(input_text)["repository_context"]
    return [str(item["path"]) for item in snapshot["files"]]


class _RecordingEventWriter:
    """Capture every timeline event the executor writes."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    async def __call__(self, feature_id: str, event: str, details: dict[str, Any]) -> None:
        self.events.append((feature_id, event, details))


async def test_a_blind_repositorys_registry_paths_reach_the_delivered_prompt(
    tmp_path: Path,
) -> None:
    """The late probe's wiring files arrive as prompt content on the first attempt."""
    harness = await build_harness(tmp_path, node_npm_repository)
    engineer_client = ScriptedLLMClient([_route_change()])
    try:
        expected = [
            module.path
            for module in inspect_repository_for_planning(
                harness.workspace, repository_id="backend"
            ).wiring_files
        ]
        assert expected, "the fixture must have wiring evidence for this test to mean anything"

        execution = await run_child(
            harness,
            engineer_payload=_route_change(),
            review_payload=review_payload(verdict="approved"),
            engineer_client=engineer_client,
            child_overrides={
                "planned_blind": True,
                "planned_blind_reason": "the model provider call failed (ReadTimeout)",
            },
        )

        assert execution.result.status == "approved"
        delivered = _delivered_paths(engineer_client)
        for path in expected:
            assert path in delivered, (
                f"the late probe's registry path {path} never reached the Engineer: {delivered}"
            )
    finally:
        await harness.dispose()


async def test_a_repository_that_was_not_blind_runs_no_late_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe belongs to blind plans alone; an evidenced plan keeps its own paths."""
    import services.feature_runtime as feature_runtime_module

    harness = await build_harness(tmp_path, node_npm_repository)
    engineer_client = ScriptedLLMClient([_route_change()])
    probes = 0
    original = inspect_repository_for_planning

    def counting(workspace: Path, *, repository_id: str) -> Any:
        nonlocal probes
        probes += 1
        return original(workspace, repository_id=repository_id)

    monkeypatch.setattr(feature_runtime_module, "inspect_repository_for_planning", counting)
    try:
        execution = await run_child(
            harness,
            engineer_payload=_route_change(),
            review_payload=review_payload(verdict="approved"),
            engineer_client=engineer_client,
        )
        assert execution.result.status == "approved"
        assert probes == 0, "an evidenced plan must not spend attempt time re-probing"
    finally:
        await harness.dispose()


async def test_a_probe_failure_leaves_the_attempt_running_and_records_blind_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second failure costs the evidence, never the attempt -- and it is recorded again."""
    import services.feature_runtime as feature_runtime_module

    harness = await build_harness(tmp_path, node_npm_repository)
    engineer_client = ScriptedLLMClient([_route_change()])
    writer = _RecordingEventWriter()

    def refuse(workspace: Path, *, repository_id: str) -> Any:
        raise OSError(f"unreadable checkout for {repository_id}: {workspace}")

    monkeypatch.setattr(feature_runtime_module, "inspect_repository_for_planning", refuse)
    try:
        execution = await run_child(
            harness,
            engineer_payload=_route_change(),
            review_payload=review_payload(verdict="approved"),
            engineer_client=engineer_client,
            event_writer=writer,
            child_overrides={
                "planned_blind": True,
                "planned_blind_reason": "the model provider call failed (ReadTimeout)",
            },
        )

        # The attempt ran to its ordinary end: the probe informs, it never gates.
        assert execution.result.status == "approved"
        assert engineer_client.calls, "the Engineer must still have been called"
        occurrences = [
            details["occurrence"]
            for _feature, event, details in writer.events
            if event == "repository_planned_blind"
        ]
        assert occurrences == ["late_reconnaissance"], (
            "the second failure must be recorded the way the first was"
        )
    finally:
        await harness.dispose()
