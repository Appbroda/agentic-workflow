"""A terminal operation row belongs to the attempt that failed it, not to every attempt after.

69-'s defect, found trying to recover AB-Feature-203. Its `AB-console-admin-2.0` install was
left `FAILED_TERMINAL` by attempt 0. An operator bought attempt 2, which computed the same
idempotency key -- the attempt is deliberately not key material, which is right, and is what
lets a preserved-workspace retry reuse a clone it already has -- found the terminal row, and
refused. The refusal was then retried as though it were weather: four backoffs, 207 seconds
of fault allowance, zero operations journaled, and a spent grant.

Two halves, and both are tested here. A row that *succeeded* is still reused across attempts
(T3, which is the test that proves the 23- rule survived); a row that *ended without a
result* is inherited only by the attempt that ended it. And the refusal itself is a
diagnosis rather than a fault, so it stops the attempt at once instead of buying backoffs to
reach the same sentence.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

import workflows.feature_workflow as feature_workflow_module
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import ChildWorkflowResultArtifact
from services.cancellation import CancellationRequested, MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from state.enums import ChildWorkflowStatus
from state.external_operations import (
    CHILD_ATTEMPT_METADATA_KEY,
    ExternalOperationStatus,
    ExternalOperationType,
)
from storage.db import Database
from storage.external_operation_store import (
    ExternalOperationJournal,
    OperationReplayRefused,
    OperationResult,
)
from tests.test_model_routing import feature_payload, no_credentials, router
from workflows.feature_workflow import (
    ChildExecution,
    FeatureWorkflowOrchestrator,
    is_transient_provider_fault,
)

pytestmark = pytest.mark.asyncio

WORKFLOW = "feature-203"
CHILD = "child-ab-console-admin"
REPOSITORY = "AB-console-admin-2.0"
INSTALL_STEP = "install_dependencies"
# What the install's idempotency is computed from. Identical across attempts on purpose: the
# revision has not moved, so this is the same work, and that sameness is exactly what made
# attempt 2 land on attempt 0's row.
INSTALL_INPUT = {"repository_revision": "abc123", "command_fingerprint": "npm-ci"}


class _Command:
    """A side effect that records every time it actually ran, and can be made to fail.

    The counter is the subject of most of these assertions. "No exception was raised" does
    not distinguish work that ran from work that was skipped, and the defect being fixed
    produced a granted attempt that raised nothing anybody saw and installed nothing.
    """

    def __init__(self, *, fails: bool = False) -> None:
        self.runs = 0
        self._fails = fails

    async def __call__(self) -> tuple[str, OperationResult]:
        """Run once, recording the run, and fail the way a command over its budget fails."""
        self.runs += 1
        if self._fails:
            msg = "npm ci exceeded its 120s budget"
            raise TimeoutError(msg)
        return "installed", OperationResult(payload={"ran": self.runs})


async def _journal(tmp_path: Path, name: str) -> tuple[Database, ExternalOperationJournal]:
    """Open one test's own journal on its own database file."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / f'{name}.db'}")
    await database.create_schema()
    return database, ExternalOperationJournal(database, stale_after_seconds=300.0)


def _executor(journal: ExternalOperationJournal, *, attempt: int) -> ExternalOperationExecutor:
    """Build the executor exactly as a child attempt's runtime builds it."""
    return ExternalOperationExecutor(
        journal=journal,
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(
            workflow_id=WORKFLOW,
            feature_id=WORKFLOW,
            child_workflow_id=CHILD,
            repository_id=REPOSITORY,
            child_attempt=attempt,
        ),
    )


async def _install(
    journal: ExternalOperationJournal,
    *,
    attempt: int,
    command: Callable[[], Awaitable[tuple[str, OperationResult]]],
    operation_type: ExternalOperationType = ExternalOperationType.INSTALL_DEPENDENCIES,
    logical_step: str = INSTALL_STEP,
    safe_input: dict[str, Any] | None = None,
) -> Any:
    """Run one attempt's install through the executor, as the validation step does."""
    return await _executor(journal, attempt=attempt).run(
        operation_type=operation_type,
        logical_step=logical_step,
        safe_input=dict(safe_input if safe_input is not None else INSTALL_INPUT),
        action=command,
    )


async def _terminal_install(journal: ExternalOperationJournal, *, attempt: int) -> Any:
    """Leave a real `FAILED_TERMINAL` install row behind, the way attempt 0 left one.

    Produced by actually failing a single-attempt operation rather than by writing the status
    in: the row under test has to be one the executor itself would create, including its
    attempt stamp and its spent budget.
    """
    failing = _Command(fails=True)
    with pytest.raises(TimeoutError):
        await _install(journal, attempt=attempt, command=failing)
    assert failing.runs == 1
    (row,) = await journal.list_operations_for_workflow(WORKFLOW)
    assert row.status is ExternalOperationStatus.FAILED_TERMINAL
    return row


async def test_t1_a_later_attempt_runs_work_an_earlier_attempt_ended(tmp_path: Path) -> None:
    """T1, the 203 replay: attempt 2 installs, and the record shows attempt 2 installing."""
    database, journal = await _journal(tmp_path, "t1-replay")
    try:
        ended = await _terminal_install(journal, attempt=0)

        granted = _Command()
        result = await _install(journal, attempt=2, command=granted)

        # The effect, not the absence of an exception: the command ran under the new attempt.
        assert granted.runs == 1
        assert result.reused is False
        assert result.value == "installed"

        rows = await journal.list_operations_for_workflow(WORKFLOW)
        assert len(rows) == 2, "attempt 2 must get its own row, not overwrite attempt 0's"
        successor = next(row for row in rows if row.operation_id != ended.operation_id)
        assert successor.status is ExternalOperationStatus.SUCCEEDED
        assert successor.safe_metadata[CHILD_ATTEMPT_METADATA_KEY] == 2

        # Attempt 0's row survives as the history of the attempt that failed, and now points
        # at what happened next.
        replaced = await journal.get(ended.operation_id)
        assert replaced.status is ExternalOperationStatus.FAILED_TERMINAL
        assert replaced.safe_metadata[CHILD_ATTEMPT_METADATA_KEY] == 0
        assert replaced.superseded_by_operation_id == successor.operation_id
        assert replaced.is_current is False
    finally:
        await database.dispose()


async def test_t2_the_same_attempt_is_still_refused(tmp_path: Path) -> None:
    """T2: repeating inside one attempt is what the terminal status exists to stop."""
    database, journal = await _journal(tmp_path, "t2-same-attempt")
    try:
        await _terminal_install(journal, attempt=0)

        repeat = _Command()
        with pytest.raises(OperationReplayRefused, match="terminal failure and cannot be replayed"):
            await _install(journal, attempt=0, command=repeat)

        assert repeat.runs == 0, "the refusal must stop the command, not merely be raised after it"
        rows = await journal.list_operations_for_workflow(WORKFLOW)
        assert len(rows) == 1, "a refused replay must not open a row"
    finally:
        await database.dispose()


async def test_t2b_an_earlier_attempt_cannot_reopen_a_later_attempts_row(tmp_path: Path) -> None:
    """T2, the other direction: only a strictly newer attempt starts fresh work.

    Not in the spec's list, and it is the same rule read backwards -- the guard is `newer`,
    not `different`, and a `!=` would let a stale caller reopen work a later attempt had
    already finished with.
    """
    database, journal = await _journal(tmp_path, "t2b-older-caller")
    try:
        await _terminal_install(journal, attempt=3)

        stale = _Command()
        with pytest.raises(OperationReplayRefused):
            await _install(journal, attempt=1, command=stale)

        assert stale.runs == 0
        assert len(await journal.list_operations_for_workflow(WORKFLOW)) == 1
    finally:
        await database.dispose()


async def test_t3_a_succeeded_row_is_still_reused_across_attempts(tmp_path: Path) -> None:
    """T3: the 23- rule. A retry that already has a clone must not clone again."""
    database, journal = await _journal(tmp_path, "t3-reuse")
    try:
        clone = _Command()
        first = await _install(
            journal,
            attempt=0,
            command=clone,
            operation_type=ExternalOperationType.CLONE_REPOSITORY,
            logical_step="clone_repository",
            safe_input={"repository_revision": "abc123"},
        )
        assert clone.runs == 1
        assert first.reused is False

        second = await _install(
            journal,
            attempt=1,
            command=clone,
            operation_type=ExternalOperationType.CLONE_REPOSITORY,
            logical_step="clone_repository",
            safe_input={"repository_revision": "abc123"},
        )

        assert clone.runs == 1, "attempt 1 re-cloned a checkout it already had"
        assert second.reused is True
        assert second.operation.operation_id == first.operation.operation_id
        assert len(await journal.list_operations_for_workflow(WORKFLOW)) == 1

        # And the succeeded row keeps attempt 0's stamp: reuse is not re-stamping, so the
        # journal still says which attempt actually did the work.
        assert second.operation.safe_metadata[CHILD_ATTEMPT_METADATA_KEY] == 0
        assert second.operation.superseded_by_operation_id is None
    except Exception:
        raise
    finally:
        await database.dispose()


async def test_t3b_an_unchanged_install_is_reused_and_a_moved_one_is_not(tmp_path: Path) -> None:
    """T3, the install half: unchanged revision reuses; a moved revision is different work."""
    database, journal = await _journal(tmp_path, "t3b-install-reuse")
    try:
        install = _Command()
        await _install(journal, attempt=0, command=install)
        assert install.runs == 1

        await _install(journal, attempt=1, command=install)
        assert install.runs == 1, "an install at an unchanged revision must not re-run"

        moved = await _install(
            journal,
            attempt=1,
            command=install,
            safe_input={"repository_revision": "def456", "command_fingerprint": "npm-ci"},
        )
        assert install.runs == 2, "a moved revision is different work and must run"
        assert moved.reused is False
    finally:
        await database.dispose()


async def test_t5_a_cancelled_row_does_not_veto_the_next_attempt(tmp_path: Path) -> None:
    """T5: a cancelled attempt's row is that attempt's, on exactly the same rule."""
    database, journal = await _journal(tmp_path, "t5-cancelled")
    try:
        # Cancelled the way the executor itself cancels: the action is interrupted while it
        # runs, and the `CancellationRequested` branch writes the row.
        stopped = _Command()

        async def interrupted() -> tuple[str, OperationResult]:
            """Stand in for a command stopped mid-flight by a cancellation request."""
            await stopped()
            raise CancellationRequested

        with pytest.raises(CancellationRequested):
            await _install(journal, attempt=0, command=interrupted)
        (cancelled_row,) = await journal.list_operations_for_workflow(WORKFLOW)
        assert cancelled_row.status is ExternalOperationStatus.CANCELLED

        # Same attempt: refused, with the cancelled sentence rather than the terminal one.
        blocked = _Command()
        with pytest.raises(OperationReplayRefused, match="cancelled operation cannot be replayed"):
            await _install(journal, attempt=0, command=blocked)
        assert blocked.runs == 0

        # A later attempt runs it.
        granted = _Command()
        result = await _install(journal, attempt=1, command=granted)

        assert granted.runs == 1
        assert result.reused is False
        rows = await journal.list_operations_for_workflow(WORKFLOW)
        assert len(rows) == 2
        successor = next(row for row in rows if row.operation_id != cancelled_row.operation_id)
        assert successor.safe_metadata[CHILD_ATTEMPT_METADATA_KEY] == 1
        assert successor.status is ExternalOperationStatus.SUCCEEDED
    finally:
        await database.dispose()


async def test_a_successor_that_ends_is_not_chased_further(tmp_path: Path) -> None:
    """One successor per attempt: the new row is that attempt's, and it too can end.

    The bound matters. Without it the second failure inside the granted attempt would open a
    third row, and a fourth, and the loop the terminal status exists to stop would be back --
    reached by a different route.
    """
    database, journal = await _journal(tmp_path, "successor-bound")
    try:
        await _terminal_install(journal, attempt=0)

        failing = _Command(fails=True)
        with pytest.raises(TimeoutError):
            await _install(journal, attempt=2, command=failing)
        assert failing.runs == 1
        assert len(await journal.list_operations_for_workflow(WORKFLOW)) == 2

        again = _Command()
        with pytest.raises(OperationReplayRefused):
            await _install(journal, attempt=2, command=again)

        assert again.runs == 0
        assert len(await journal.list_operations_for_workflow(WORKFLOW)) == 2
    finally:
        await database.dispose()


async def test_an_unstamped_row_is_never_called_older(tmp_path: Path) -> None:
    """A scope with no attempt belongs to no attempt, so it can outrank nothing.

    Reconnaissance and the parent's publication calls are feature-scoped and stamp no
    attempt. Treating a missing stamp as attempt -1 would let any caller reopen their rows.
    """
    database, journal = await _journal(tmp_path, "unstamped")
    try:
        unscoped = ExternalOperationExecutor(
            journal=journal,
            cancellation_token=MockCancellationToken(),
            scope=ExternalOperationScope(
                workflow_id=WORKFLOW, feature_id=WORKFLOW, repository_id=REPOSITORY
            ),
        )
        failing = _Command(fails=True)
        with pytest.raises(TimeoutError):
            await unscoped.run(
                operation_type=ExternalOperationType.RUN_REPOSITORY_RECON,
                logical_step="recon",
                safe_input={"scope": "feature"},
                action=failing,
            )

        blocked = _Command()
        with pytest.raises(OperationReplayRefused):
            await unscoped.run(
                operation_type=ExternalOperationType.RUN_REPOSITORY_RECON,
                logical_step="recon",
                safe_input={"scope": "feature"},
                action=blocked,
            )

        assert blocked.runs == 0
        assert len(await journal.list_operations_for_workflow(WORKFLOW)) == 1
    finally:
        await database.dispose()


# -- The classification half ---------------------------------------------------------------


class _RefusesTheReplay:
    """A child attempt that hits the replay refusal, counting how many times it is asked."""

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, **kwargs: Any) -> ChildExecution:
        """Refuse the backend the way a granted attempt on an ended row refuses."""
        if kwargs["repository"].repository_id != "backend":
            msg = "only the backend is under test here"
            raise AssertionError(msg)
        self.calls += 1
        msg = "operation has a terminal failure and cannot be replayed: operation-203-install"
        raise OperationReplayRefused(msg)


def _backend_result(artifacts: list[Any]) -> ChildWorkflowResultArtifact:
    """The backend's final result artifact, where the stop has to be legible."""
    results = [
        item
        for item in artifacts
        if isinstance(item, ChildWorkflowResultArtifact) and item.repository_id == "backend"
    ]
    assert results, "the backend recorded no child result artifact"
    return results[-1]


async def test_t4_the_predicate_refuses_to_retry_a_replay_refusal() -> None:
    """T4, the unit half: this is platform state, not weather."""
    refusal = OperationReplayRefused("operation has a terminal failure and cannot be replayed: x")

    assert is_transient_provider_fault(refusal) is False
    # And the base class it inherits from is still weather, because AB-Feature-111 is why.
    # Narrowing the subclass must not narrow the wrap it is a subclass of.
    from storage.external_operation_store import ExternalOperationError

    assert is_transient_provider_fault(ExternalOperationError("failed before a confirmed result"))


async def test_t4_the_refusal_stops_the_attempt_without_buying_a_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T4, the effect: one call, no backoff slept, and no fault seconds accounted.

    The backoff base is set high on purpose. Before this change the refusal was a transient
    fault, so this run would have slept 30s, then 60s, then 120s, then 240s before ending on
    "the provider did not answer" -- 203's granted attempt spent 207 seconds doing exactly
    that. Wall-clock is therefore the decisive assertion: a run that finishes in under a
    second cannot have slept a 30-second backoff.
    """
    monkeypatch.setattr(feature_workflow_module, "_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS", 30.0)
    request = StartFeatureRequest.model_validate(feature_payload("feature-replay-refused"))
    state = _initial_feature_state("feature-replay-refused", request)
    executor = _RefusesTheReplay()

    started = time.monotonic()
    result = await FeatureWorkflowOrchestrator(
        child_executor=executor, model_router=router()
    ).start(state, credentials=no_credentials())
    elapsed = time.monotonic() - started

    assert executor.calls == 1, "the refusal was retried; it returns the same answer every time"
    assert elapsed < 5.0, f"a backoff was slept: {elapsed:.1f}s"
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.FAILED

    artifact = _backend_result(list(result.artifacts))
    # No fault was ever counted, so no fault seconds were excluded from the charged clock.
    # The key is absent rather than zero on this route: the refusal leaves the retry loop
    # before its fault accounting runs at all, which is stronger than being counted as zero.
    metadata = dict(artifact.metadata)
    assert metadata.get("fault_seconds_excluded", 0.0) == 0.0
    assert metadata.get("fault_count", 0) == 0
    assert metadata.get("fault_classes", []) == []


async def test_t6_a_grant_that_cannot_run_says_so_in_the_blocking_issues() -> None:
    """T6: a grant spent with nothing journaled and nothing said is the forbidden outcome.

    203's granted attempt ended with "the provider did not answer" -- a sentence about a
    remote that was never contacted -- and an operator reading it had no way to reach the
    real reason. The refusal's own words have to survive to the record.
    """
    request = StartFeatureRequest.model_validate(feature_payload("feature-grant-refused"))
    state = _initial_feature_state("feature-grant-refused", request)

    result = await FeatureWorkflowOrchestrator(
        child_executor=_RefusesTheReplay(), model_router=router()
    ).start(state, credentials=no_credentials())

    artifact = _backend_result(list(result.artifacts))
    blocking = list(artifact.blocking_issues)

    assert any("cannot be replayed" in issue for issue in blocking), blocking
    assert not any("did not answer" in issue for issue in blocking), (
        "the stop was attributed to a provider that was never contacted"
    )
    assert not any("did not anticipate" in issue for issue in blocking), (
        "the refusal is anticipated; recording it as a surprise sends the reader elsewhere"
    )
    assert artifact.failure_classification == "platform_defect"
