"""A claim advances a feature by one step and returns.

Audit risk P0-2, second half. A claim used to mean *run this whole feature*: one coroutine
held it for a mean of 86 minutes and up to 571, under one Redis lease and one queue lease
renewed underneath it, and everything it had not checkpointed died with the process. A crash
cost the feature; a deploy was indistinguishable from a crash.

What is asserted here is the difference that makes: the same feature, the same artifacts and
the same external effects, reached through many small claims instead of one long one -- and
the properties that only matter once the loop between claims is durable. Progress must not
spend the budget reserved for crashes, two features must interleave, two workers must not run
one feature's step at once, and a cancellation between steps must stop the next one.

The crash half lives in `test_crash_windows.py`, where a real process is killed inside a real
step. Nothing here simulates a crash with an exception.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import update

from api.control_plane import RequestScopedCredentials
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import Artifact, PullRequestArtifact
from services.feature_queue import (
    CONTINUE,
    QUEUED,
    RUNNING,
    START,
    SUCCEEDED,
    DatabaseFeatureExecutionQueue,
)
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.feature_models import FeatureWorkflowSnapshot
from storage.db import Database
from storage.feature_store import ClaimDisposition, SqlAlchemyFeatureControlPlane
from storage.models import FeatureExecutionQueueModel
from tests.support import drain_feature_queue
from tests.test_feature_api import feature_payload
from workflows.feature_workflow import (
    FeatureStep,
    FeatureWorkflowOrchestrator,
    feature_is_at_rest,
    next_step,
)

pytestmark = pytest.mark.asyncio

CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)


class StepRecordingOrchestrator(FeatureWorkflowOrchestrator):
    """The deterministic orchestrator, plus a record of every step it was asked to run.

    Recording the step rather than counting calls to a collaborator, because the step *is*
    the unit under test: how many there were, in what order, and which of them a second claim
    repeated are the assertions this whole file is made of.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Start with nothing run."""
        super().__init__(**kwargs)
        self.steps: list[str] = []

    async def _run_step(
        self, state: FeatureWorkflowSnapshot, step: FeatureStep, *, credentials: Any
    ) -> FeatureWorkflowSnapshot:
        """Record this step, then run it exactly as the orchestrator would."""
        self.steps.append(step.value)
        return await super()._run_step(state, step, credentials=credentials)


async def _store(
    tmp_path: Path, name: str, runner: FeatureWorkflowOrchestrator
) -> tuple[Database, SqlAlchemyFeatureControlPlane]:
    """A durable control plane on disk, wired to the runner a test wants to observe."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / name}")
    await database.create_schema()
    return database, SqlAlchemyFeatureControlPlane(database, mock_runner=runner)


async def _accept(store: SqlAlchemyFeatureControlPlane, feature_id: str) -> str:
    """Accept one feature through the real path, so its queue row is the real one."""
    request = StartFeatureRequest.model_validate({**feature_payload(), "feature_id": feature_id})
    result = await store.start(
        request, idempotency_key=feature_id, credentials=CREDENTIALS, owner_id="platform-admin"
    )
    return result.record.state.feature_id


async def _entry(database: Database, feature_id: str) -> FeatureExecutionQueueModel:
    """Read one queue row as a later claim sees it."""
    async with database.session() as session:
        row = await session.get(FeatureExecutionQueueModel, feature_id)
        assert row is not None
        return row


def _comparable(artifacts: list[Artifact]) -> list[dict[str, Any]]:
    """Reduce artifacts to what a run must reproduce, dropping only when it happened.

    Timestamps and the artifact's own recorded time differ between two runs of the same
    feature for reasons that say nothing about the work, and comparing them would make this
    assertion a clock comparison.
    """
    reduced: list[dict[str, Any]] = []
    for artifact in sorted(artifacts, key=lambda item: item.artifact_id):
        payload = artifact.model_dump(mode="json")
        for key in ("created_at", "recorded_at", "completed_at", "timestamp", "approved_at"):
            payload.pop(key, None)
        metadata = payload.get("metadata", {})
        metadata.pop("executor_build_revision", None)
        # The attempt's clocks are measurements of when and how long, not of what was
        # produced -- two identical runs cannot share them any more than they share
        # timestamps. The fault evidence beside them (count, classes, the degraded-retry
        # flag) is behavior and stays compared.
        for key in ("runtime_wall_seconds", "runtime_charged_seconds", "fault_seconds_excluded"):
            metadata.pop(key, None)
        reduced.append(payload)
    return reduced


# --------------------------------------------------------------------------------------
# 8.2 -- a feature completes across many claims, and produces what one claim produced
# --------------------------------------------------------------------------------------


async def test_a_multi_repository_feature_completes_across_many_claims(tmp_path: Path) -> None:
    """The same feature, the same artifacts, reached one step at a time.

    The comparison is against a feature-scoped run of the identical composition -- the
    in-process driver, which is the code path this change replaced for the queue and kept for
    everything else. If stepping changed what a feature produces, this is where it shows.
    """
    stepped_runner = StepRecordingOrchestrator()
    database, store = await _store(tmp_path, "many-claims.db", stepped_runner)
    try:
        await _accept(store, "feature-one-claim")
        claims = await drain_feature_queue(store)
        stepped = (await store.get_record("feature-one-claim")).state

        # The same feature run the old way: one call, one process, no queue. Given the
        # reference the durable run was allocated, because a pull request is titled with it
        # and the reference comes from the database rather than from the work.
        feature_scoped_runner = StepRecordingOrchestrator()
        origin = _initial_snapshot("feature-one-claim")
        origin.reference = stepped.reference
        feature_scoped = await feature_scoped_runner.start(origin, credentials=CREDENTIALS)

        assert claims > 1, "the queue ran the whole feature inside a single claim"
        assert stepped_runner.steps == feature_scoped_runner.steps
        assert stepped.status is feature_scoped.status is FeatureWorkflowStatus.COMPLETED
        assert {key: child.status for key, child in stepped.child_workflows.items()} == {
            key: child.status for key, child in feature_scoped.child_workflows.items()
        }
        assert _comparable(stepped.artifacts) == _comparable(feature_scoped.artifacts)
        # One pull request per repository, counted on what the run produced rather than on
        # how many times the publisher was called.
        published = [item for item in stepped.artifacts if isinstance(item, PullRequestArtifact)]
        assert len(published) == len(stepped.child_workflows)
        assert sorted(
            str(child.pull_request_artifact_id) for child in stepped.child_workflows.values()
        ) == sorted(item.artifact_id for item in published)
        # And the entry is closed once, on the feature's real outcome.
        entry = await _entry(database, "feature-one-claim")
        assert entry.status == SUCCEEDED
        assert entry.steps == len(stepped_runner.steps) - 1
        assert claims == len(stepped_runner.steps), "a claim ran more or less than one step"
    finally:
        await database.dispose()


def _initial_snapshot(feature_id: str) -> FeatureWorkflowSnapshot:
    """The state acceptance persists, without needing a database to produce it."""
    from api.feature_control_plane import _initial_feature_state

    return _initial_feature_state(
        feature_id,
        StartFeatureRequest.model_validate({**feature_payload(), "feature_id": feature_id}),
    )


# --------------------------------------------------------------------------------------
# 8.4 -- progress does not spend the crash budget
# --------------------------------------------------------------------------------------


async def test_progress_does_not_consume_the_attempt_budget(tmp_path: Path) -> None:
    """`max_attempts` bounds crashes. A claim is one step, so it must not bound steps.

    Left alone, a feature of more than three steps would have exhausted its entry on its
    fourth -- every feature this platform runs, failed as a platform defect at the point it
    started working.
    """
    runner = StepRecordingOrchestrator()
    database, store = await _store(tmp_path, "budget.db", runner)
    try:
        await _accept(store, "feature-budget")
        await drain_feature_queue(store)

        entry = await _entry(database, "feature-budget")
        assert len(runner.steps) > entry.max_attempts, (
            "this feature is too short to prove anything about a three-attempt budget"
        )
        assert entry.steps == len(runner.steps) - 1, "the entry did not count what it ran"
        # One, not one per step: every claim that made progress gave its attempt back, so
        # what is left on the closed entry is the single attempt its last claim spent.
        assert entry.attempt == 1
        assert (await store.get_record("feature-budget")).state.status is (
            FeatureWorkflowStatus.COMPLETED
        )
    finally:
        await database.dispose()


async def test_a_step_that_keeps_crashing_still_reaches_the_end_of_its_budget(
    tmp_path: Path,
) -> None:
    """The other half: returning the attempt on progress must not make a crash free.

    Reproduced the way the queue sees a crash -- a claim that recorded nothing and a lease
    that lapsed -- three times over. The fourth claim is refused, and the abandoned-run sweep
    is what gives the feature its terminal status, exactly as it did before stepping.
    """
    runner = StepRecordingOrchestrator()
    database, store = await _store(tmp_path, "crash-budget.db", runner)
    try:
        await _accept(store, "feature-crashing")
        entry = await _entry(database, "feature-crashing")
        state = (await store.get_record("feature-crashing")).state.model_copy(deep=True)
        state.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
        await store._replace_state(  # noqa: SLF001 - building the precondition, not the behaviour
            "feature-crashing", state, "child_workflows_started"
        )

        for attempt in range(1, entry.max_attempts + 1):
            await _lapsed_claim(database, "feature-crashing", attempt=attempt)
            assert (
                await store._claim_disposition(  # noqa: SLF001
                    (await store.get_record("feature-crashing")).state, START
                )
                is ClaimDisposition.CONTINUE
            ), f"attempt {attempt} was refused too early"

        await _lapsed_claim(database, "feature-crashing", attempt=entry.max_attempts + 1)
        assert (
            await store._claim_disposition(  # noqa: SLF001
                (await store.get_record("feature-crashing")).state, START
            )
            is ClaimDisposition.DROP
        )
        assert await store.feature_owes_another_step("feature-crashing", intent=START) is False
    finally:
        await database.dispose()


async def _lapsed_claim(database: Database, feature_id: str, *, attempt: int) -> None:
    """Leave the entry in the shape a killed worker leaves: claimed, lease expired."""
    lapsed = datetime.now(UTC) - timedelta(minutes=5)
    async with database.session() as session:
        await session.execute(
            update(FeatureExecutionQueueModel)
            .where(FeatureExecutionQueueModel.feature_id == feature_id)
            .values(
                status=RUNNING,
                attempt=attempt,
                lease_owner="worker-that-died",
                lease_expires_at=lapsed,
                started_at=lapsed,
            )
        )
        await session.commit()


async def test_the_feature_runtime_ceiling_is_not_restarted_by_every_step(
    tmp_path: Path,
) -> None:
    """The six-hour backstop is clocked from `started_at`, and re-queueing must not clear it.

    `requeue` -- the one a person's resume uses -- deliberately does clear it, because that is
    a new run. A step is not, and a re-queue that reset the clock would silently retire the
    only ceiling on a feature that goes on making small amounts of progress for ever.
    """
    runner = StepRecordingOrchestrator()
    database, store = await _store(tmp_path, "runtime-clock.db", runner)
    try:
        await _accept(store, "feature-clock")
        queue = DatabaseFeatureExecutionQueue(database)
        assert await queue.claim(owner="worker", lease_seconds=60) is not None
        claimed_at = (await _entry(database, "feature-clock")).started_at
        assert claimed_at is not None

        await queue.advance_to_next_step("feature-clock")

        after = await _entry(database, "feature-clock")
        assert after.started_at == claimed_at, "stepping restarted the feature runtime clock"
        assert after.status == QUEUED
        assert after.intent == CONTINUE, "a step re-queued itself as somebody's resume"
        assert after.attempt == 0
        assert after.steps == 1
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# 8.5 -- fairness
# --------------------------------------------------------------------------------------


async def test_two_features_interleave_and_neither_is_starved(tmp_path: Path) -> None:
    """A forty-step feature must not hold the workers against a feature queued behind it.

    `advance_to_next_step` moves `queued_at` forward, so a re-queued entry takes its turn
    behind whatever is already waiting. Without that the first feature accepted would run to
    completion before the second one was looked at, which is the behaviour stepping was
    supposed to remove.
    """
    order: list[str] = []

    class _Ordering(FeatureWorkflowOrchestrator):
        async def _run_step(
            self, state: FeatureWorkflowSnapshot, step: FeatureStep, *, credentials: Any
        ) -> FeatureWorkflowSnapshot:
            order.append(state.feature_id)
            return await super()._run_step(state, step, credentials=credentials)

    database, store = await _store(tmp_path, "fairness.db", _Ordering())
    try:
        await _accept(store, "feature-first")
        await _accept(store, "feature-second")

        await drain_feature_queue(store)

        assert set(order) == {"feature-first", "feature-second"}
        # Interleaved, not sequential: the second feature took a step before the first one
        # had taken all of its.
        first_finished = len(order) - 1 - order[::-1].index("feature-first")
        assert "feature-second" in order[:first_finished]
        for feature_id in ("feature-first", "feature-second"):
            assert (await store.get_record(feature_id)).state.status is (
                FeatureWorkflowStatus.COMPLETED
            ), f"{feature_id} was starved"
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# 8.6 -- cancellation
# --------------------------------------------------------------------------------------


async def test_a_cancellation_between_steps_schedules_no_further_step(tmp_path: Path) -> None:
    """`next_step` returns nothing for a cancelled feature, so the loop simply stops.

    This is the half stepping makes easier: there is a decision point between every step, and
    a cancellation recorded at any of them is read at the next one. Interrupting a step
    already in flight is unchanged and is the durable cancellation token's job.
    """
    runner = StepRecordingOrchestrator()
    database, store = await _store(tmp_path, "cancel-between.db", runner)
    try:
        await _accept(store, "feature-cancelled")
        queue = DatabaseFeatureExecutionQueue(database)
        claimed = await queue.claim(owner="worker", lease_seconds=60)
        assert claimed is not None
        await store.execute_queued(
            "feature-cancelled", credentials=CREDENTIALS, intent=claimed.intent
        )
        after_one_step = list(runner.steps)
        assert after_one_step, "the first claim ran no step at all"

        await store.cancel(
            "feature-cancelled",
            reason="the author changed their mind",
            requested_by="platform-admin",
        )

        cancelled = (await store.get_record("feature-cancelled")).state
        decision = next_step(cancelled)
        assert decision.step is None, "a cancelled feature was still owed work"
        assert feature_is_at_rest(cancelled)
        assert await store.feature_owes_another_step("feature-cancelled", intent=CONTINUE) is False
        # And a claim that arrives anyway does nothing, rather than restarting the feature.
        await drain_feature_queue(store)
        assert runner.steps == after_one_step
    finally:
        await database.dispose()


async def test_a_stale_step_result_cannot_erase_a_cancellation(tmp_path: Path) -> None:
    """A step in flight when the cancel lands must not write the cancellation away.

    `_preserve_cancellation` is what stops it, and stepping multiplies the number of writes
    that pass through it: every step ends in one, where a feature used to end in one.
    """
    runner = StepRecordingOrchestrator()
    database, store = await _store(tmp_path, "cancel-stale.db", runner)
    try:
        await _accept(store, "feature-stale")
        queue = DatabaseFeatureExecutionQueue(database)
        claimed = await queue.claim(owner="worker", lease_seconds=60)
        assert claimed is not None
        await store.execute_queued("feature-stale", credentials=CREDENTIALS, intent=claimed.intent)
        mid_run = (await store.get_record("feature-stale")).state.model_copy(deep=True)

        await store.cancel("feature-stale", reason="stop this", requested_by="platform-admin")
        cancelled = (await store.get_record("feature-stale")).state
        assert cancelled.cancellation_requested is True

        # The write a step already in flight would make: it knows nothing about the cancel.
        stale = mid_run.model_copy(deep=True)
        stale.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
        stale.cancellation_requested = False
        await store._replace_state(  # noqa: SLF001 - reproducing the concurrent write itself
            "feature-stale", stale, "child_workflows_started"
        )

        preserved = (await store.get_record("feature-stale")).state
        assert preserved.cancellation_requested is True
        assert preserved.cancellation_reason == cancelled.cancellation_reason
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# 8.7 -- locks
#
# "Two workers cannot run the same feature's step at once" is asserted in
# `test_durable_postgres_tier.py` and not here. It rests on `FOR UPDATE SKIP LOCKED`, which
# SQLite has neither of, so on this database two dispatchers really do claim the same row --
# and a test that passed here would be certifying a guarantee this engine does not offer.
# --------------------------------------------------------------------------------------


async def test_a_step_that_outlives_its_lease_is_detected_and_costs_only_that_step(
    tmp_path: Path,
) -> None:
    """A lapsed lease hands over one step, not a feature.

    The renewal loop exists so this is rare; what matters is what it costs when it happens.
    The second claim reads the same persisted state the first one had reached, so the step it
    redoes is the interrupted one and nothing before it.
    """
    runner = StepRecordingOrchestrator()
    database, store = await _store(tmp_path, "lease-lapse.db", runner)
    try:
        await _accept(store, "feature-lapsed")
        queue = DatabaseFeatureExecutionQueue(database)
        first = await queue.claim(owner="worker-slow", lease_seconds=60)
        assert first is not None
        await store.execute_queued("feature-lapsed", credentials=CREDENTIALS, intent=first.intent)
        completed_before_the_lapse = list(runner.steps)

        # The lease lapses under the slow worker, and another one takes the entry.
        await _lapsed_claim(database, "feature-lapsed", attempt=first.attempt)
        second = await queue.claim(owner="worker-taking-over", lease_seconds=60)
        assert second is not None
        assert second.attempt == first.attempt + 1, "a lapse must be visible as a spent attempt"

        await store.execute_queued("feature-lapsed", credentials=CREDENTIALS, intent=second.intent)

        redone = runner.steps[len(completed_before_the_lapse) :]
        assert len(redone) == 1, "the takeover ran more than one step"
        assert redone[0] not in completed_before_the_lapse[:-1], (
            "the takeover repeated a step the first worker had already finished"
        )
    finally:
        await database.dispose()


# --------------------------------------------------------------------------------------
# The claim decision, under stepping
# --------------------------------------------------------------------------------------


async def test_a_claim_arriving_on_a_resting_feature_closes_its_entry(tmp_path: Path) -> None:
    """The fence that stops a re-queue loop, asked at the point the entry would be reopened.

    A feature resting where a person owns it refuses a `start` claim, and it has to refuse the
    re-queue that would follow just as firmly. Asking under any other intent would answer that
    the entry should be reopened, and reopening it runs exactly the work the drop refused.
    """
    runner = StepRecordingOrchestrator()
    database, store = await _store(tmp_path, "resting-entry.db", runner)
    try:
        await _accept(store, "feature-resting")
        resting = (await store.get_record("feature-resting")).state.model_copy(deep=True)
        resting.status = FeatureWorkflowStatus.CONTRACT_READY
        resting.child_workflows = {}
        await store._replace_state(  # noqa: SLF001 - building the precondition
            "feature-resting", resting, "contract_ready"
        )

        assert await store.feature_owes_another_step("feature-resting", intent=START) is False
        assert await drain_feature_queue(store) == 1
        assert runner.steps == [], "a dropped claim ran a step anyway"
        assert (await store.get_record("feature-resting")).state.status is (
            FeatureWorkflowStatus.CONTRACT_READY
        )
        assert (await _entry(database, "feature-resting")).status == SUCCEEDED
    finally:
        await database.dispose()


async def test_a_feature_whose_every_workstream_stopped_is_not_re_queued_for_ever(
    tmp_path: Path,
) -> None:
    """The evidence still names a step; no claim may run it, so the entry has to close.

    Without this the entry is reopened into a claim that refuses it, over and over, and the
    only thing that stops it is the step ceiling -- two hundred claims later, having written
    two hundred identical refusals to the timeline.
    """
    runner = StepRecordingOrchestrator()
    database, store = await _store(tmp_path, "no-eligible.db", runner)
    try:
        await _accept(store, "feature-stopped")
        queue = DatabaseFeatureExecutionQueue(database)
        for _ in range(2):
            claimed = await queue.claim(owner="worker", lease_seconds=60)
            assert claimed is not None
            await store.execute_queued(
                "feature-stopped", credentials=CREDENTIALS, intent=claimed.intent
            )
            await queue.advance_to_next_step("feature-stopped")
        stopped = (await store.get_record("feature-stopped")).state.model_copy(deep=True)
        assert stopped.child_workflows, "the planning steps produced no repositories to stop"
        stopped.status = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
        for repository_id, child in stopped.child_workflows.items():
            stopped.child_workflows[repository_id] = child.model_copy(
                update={
                    "status": ChildWorkflowStatus.FAILED,
                    "retry_refusal_reason": "this workstream is not converging",
                    "retry_count": stopped.max_child_review_cycles - 1,
                }
            )
        await store._replace_state(  # noqa: SLF001 - building the precondition
            "feature-stopped", stopped, "child_workflows_started"
        )

        assert next_step(stopped).step is not None, "this state is meant to still name a step"
        assert await store.feature_owes_another_step("feature-stopped", intent=CONTINUE) is False
        assert (await _entry(database, "feature-stopped")).steps < 5
    finally:
        await database.dispose()
