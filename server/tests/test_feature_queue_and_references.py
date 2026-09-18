"""Acceptance, identity, and setup: what a submission is guaranteed before anything runs.

Three properties this platform did not previously have, each of which was a real complaint:

* a submission is durably accepted and answered *before* planning, so it is visible instead of
  invisible for minutes;
* it is answered with an identity a person can quote, allocated by the database rather than
  counted by the application;
* and the credentials the work will need are established up front, because the work now
  outlives the request that asked for it.

The tests are written against the effects -- what is persisted, what a later reader sees, what
a pull request ends up called -- rather than against which method was invoked.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from adapters.llm_adapter import LLMAdapterError
from api.control_plane import RequestScopedCredentials, WorkflowConflictError
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import PullRequestArtifact, TechnicalPRDArtifact
from main import create_app
from services.feature_queue import (
    FeatureExecutionBusy,
    FeatureProviderFault,
    FeatureQueueDispatcher,
    InMemoryFeatureExecutionQueue,
)
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.feature_models import FeatureWorkflowSnapshot
from storage.db import Database
from storage.feature_store import SqlAlchemyFeatureControlPlane
from storage.repository_configuration_store import (
    DatabaseRepositoryConfigurationDirectory,
    InMemoryRepositoryConfigurationDirectory,
    RepositoryConfigurationError,
)
from tests.support import drain_feature_queue, settle
from tests.test_feature_api import feature_payload
from workflows.feature_workflow import FeatureWorkflowOrchestrator

KEY = "queue-test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


class BlockingRunner(FeatureWorkflowOrchestrator):
    """A platform whose analysis never finishes until it is let go.

    The point of the whole change is that a caller does not wait for this, so the test needs
    something that provably has not finished when the response arrives.
    """

    def __init__(self) -> None:
        """Start held, and record whether anything ever reached the runner."""
        super().__init__()
        self.release = asyncio.Event()
        self.started = asyncio.Event()

    async def advance_one_step(
        self, state: FeatureWorkflowSnapshot, *, credentials: RequestScopedCredentials
    ) -> FeatureWorkflowSnapshot:
        """Announce that analysis began, then wait to be released before doing any of it.

        Held at the step rather than at `start`, because a claim runs one step now. The first
        step of this feature is still the product manager reading the requirement, which is
        the thing the caller provably has not waited for.
        """
        self.started.set()
        await self.release.wait()
        return await super().advance_one_step(state, credentials=credentials)


@pytest.mark.asyncio
async def test_a_submission_is_answered_queued_before_any_analysis_runs(tmp_path: Path) -> None:
    """The response comes back while the product manager has not read a word.

    This is the behaviour the whole queue exists for. Previously the caller waited for
    planning, reconnaissance and every clone, so a submission was indistinguishable from a
    failure for minutes and was lost outright if the request timed out.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'queued.db'}")
    await database.create_schema()
    runner = BlockingRunner()
    try:
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=runner)
        # The queue is taken from the injected control plane. Passing one explicitly would
        # hide the case that matters: an application given a plane and nothing else must
        # still watch the queue that plane writes to.
        app = create_app(platform_api_key=KEY, feature_control_plane=store)
        assert app.state.feature_queue is store.queue
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            started = await client.post("/features/start", headers=AUTH, json=feature_payload())
            # Read back through a separate request, so the assertion is about what is
            # persisted rather than about the object the write returned.
            fetched = await client.get("/features/feature-login", headers=AUTH)
            listed = await client.get("/features", headers=AUTH)

            assert started.status_code == 201, started.text
            assert started.json()["status"] == "pending"
            assert started.json()["reference"] == "AB-Feature-1"
            # Persisted immediately: the feature is readable by id and present in the list
            # before anything has executed.
            assert fetched.status_code == 200
            assert fetched.json()["reference"] == "AB-Feature-1"
            summary = next(
                item for item in listed.json()["features"] if item["feature_id"] == "feature-login"
            )
            assert summary["dashboard_group"] == "queued"
            assert summary["reference"] == "AB-Feature-1"
            # And analysis had not finished -- indeed the runner is still holding.
            dispatch = asyncio.create_task(drain_feature_queue(store))
            await asyncio.wait_for(runner.started.wait(), timeout=5)
            assert (await store.get_record("feature-login")).state.status is (
                FeatureWorkflowStatus.PENDING
            )
            runner.release.set()
            await asyncio.wait_for(dispatch, timeout=30)

            executed = await client.get("/features/feature-login", headers=AUTH)
            assert executed.json()["status"] == "completed"
            # The identity it was given at acceptance is the identity it keeps.
            assert executed.json()["reference"] == "AB-Feature-1"
    finally:
        runner.release.set()
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_queued_feature_offers_cancelling_but_not_resuming() -> None:
    """There is nothing to resume before a worker has claimed it.

    Resume is published as an available action, and the console renders what the server
    publishes. Offering it on a queued feature would let somebody race the worker that is
    about to pick it up. Cancelling stays available: deciding against a feature before it runs
    is a real thing to want.
    """
    app = create_app(platform_api_key=KEY)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        queued = await client.post("/features/start", headers=AUTH, json=feature_payload())
        await settle(app)
        executed = await client.get("/features/feature-login", headers=AUTH)

    assert queued.json()["status"] == "pending"
    assert "RESUME_WORKFLOW" not in queued.json()["available_actions"]
    assert "CANCEL_WORKFLOW" in queued.json()["available_actions"]
    # And a feature that has run is unchanged by this: it is terminal here, so it offers
    # nothing, which is the behaviour that already existed.
    assert executed.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_concurrent_submissions_never_share_a_feature_reference(tmp_path: Path) -> None:
    """Ten at once get ten different numbers, because the database allocates them.

    `count(features) + 1` passes a sequential test and fails exactly here, which is why the
    allocation is an insert into a table with an autoincrementing key.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'concurrent.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)

        async def submit(index: int) -> str | None:
            request = StartFeatureRequest.model_validate(feature_payload()).model_copy(
                update={"feature_id": f"concurrent-{index}"}
            )
            result = await store.start(
                request,
                idempotency_key=f"concurrent-{index}",
                credentials=credentials,
                owner_id="platform-admin",
            )
            return result.record.state.reference

        references = await asyncio.gather(*(submit(index) for index in range(10)))

        assert len(set(references)) == 10
        assert set(references) == {f"AB-Feature-{number}" for number in range(1, 11)}
    finally:
        await database.drop_schema()
        await database.dispose()


class ClarifyingRunner(FeatureWorkflowOrchestrator):
    """Park a feature awaiting clarification, and record what actually resumed it."""

    def __init__(self) -> None:
        """Start with nothing resumed."""
        super().__init__()
        self.resumed_with: list[list[str]] = []

    async def begin_resume(
        self, state: FeatureWorkflowSnapshot, *, answers: Any
    ) -> FeatureWorkflowSnapshot:
        """Record the answers this resumption carried, and schedule nothing."""
        self.resumed_with.append([item.question_id for item in answers])
        return state

    async def advance_one_step(
        self, state: FeatureWorkflowSnapshot, *, credentials: RequestScopedCredentials
    ) -> FeatureWorkflowSnapshot:
        """Leave the feature waiting the way an unanswered question does, until it is answered.

        Written against the step surface because that is what a claim runs. Which claim this
        is comes from the same evidence the assertions read: nothing has been resumed until
        `begin_resume` has recorded a resumption.
        """
        del credentials
        if not self.resumed_with:
            return state.model_copy(
                update={
                    "status": FeatureWorkflowStatus.WAITING_FOR_HUMAN,
                    "current_agent": "human_clarification",
                }
            )
        return state.model_copy(update={"status": FeatureWorkflowStatus.COMPLETED})


@pytest.mark.asyncio
async def test_resuming_a_feature_is_queued_work_not_request_work(tmp_path: Path) -> None:
    """AB-Feature-108's shape: the whole feature ran inside one `POST /resume`.

    Answering a clarification opened a request that then ran contract planning, two
    workstreams and nine coding attempts inside itself for seventy minutes. One lease renewal
    failed, the request was cancelled, and the run died mid-attempt with no terminal status --
    because the only thing that knew the run existed was a socket.

    So a resume must be durable before it executes, exactly as a submission is.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'queued-resume.db'}")
    await database.create_schema()
    try:
        runner = ClarifyingRunner()
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=runner)
        credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="queued-resume-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        assert (
            await store.get_record("feature-login")
        ).state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN

        await store.resume("feature-login", answers=[], credentials=credentials)

        # Nothing has run yet, and the intent to run is already durable. That is the whole
        # property: the work now survives the request that asked for it.
        assert runner.resumed_with == []
        assert await store.queue.pending_count() == 1
        assert await store.queue.active_intent("feature-login") == "resume"

        assert await drain_feature_queue(store) == 1
        assert runner.resumed_with == [[]]
        assert await store.queue.active_intent("feature-login") is None
        assert (
            await store.get_record("feature-login")
        ).state.status is FeatureWorkflowStatus.COMPLETED
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_resume_a_dead_worker_dropped_is_claimed_by_another(tmp_path: Path) -> None:
    """The reason queueing it matters: a lost worker no longer loses the work.

    -108's run was unrecoverable because it lived in a request. A queued resumption is
    claimed under a lease, so a process that dies mid-run leaves an entry the next process
    picks up -- which is the difference between a queue and a background task.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'resume-lease.db'}")
    await database.create_schema()
    try:
        runner = ClarifyingRunner()
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=runner)
        credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="resume-lease-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        await store.resume("feature-login", answers=[], credentials=credentials)

        # A worker claims the resumption with a short lease, and then dies without finishing.
        claimed = await store.queue.claim(owner="worker-that-dies", lease_seconds=1)
        assert claimed is not None
        assert claimed.intent == "resume"
        assert claimed.requested_by == "platform-admin", (
            "the feature owner's identity resolves credentials, not the resumer's"
        )
        assert runner.resumed_with == []

        await asyncio.sleep(1.1)
        reclaimed = await store.queue.claim(owner="worker-that-lives", lease_seconds=60)

        assert reclaimed is not None
        assert reclaimed.intent == "resume"
        assert reclaimed.attempt == 2, "the lapsed claim is retried, not abandoned"
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_queued_retry_grant_carries_the_repository_it_was_granted_for(
    tmp_path: Path,
) -> None:
    """A retry does everything a first attempt does, so it must not run on a request thread.

    Its arguments have to survive the hand-off intact: a grant that reached a worker without
    its repository would either do nothing or retry the wrong one.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'queued-retry.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="queued-retry-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        # Put a stopped repository where a person would find one before granting a retry.
        state = (await store.get_record("feature-login")).state
        state.status = FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        child = state.child_workflows["backend"]
        state.child_workflows["backend"] = child.model_copy(
            update={"status": ChildWorkflowStatus.FAILED}
        )
        await store._replace_state(  # noqa: SLF001 - building the precondition
            "feature-login", state, "feature_failed_requires_human"
        )

        await store.retry_workstream(
            "feature-login",
            "backend",
            additional_attempts=1,
            # What the route actually composes: a sentence naming the authenticated identity
            # and the label somebody typed. It is an audit answer, not an account.
            requested_by="akhilesh (via platform key)",
            reason="The repository was repaired out of band.",
            credentials=credentials,
        )

        claimed = await store.queue.claim(owner="worker", lease_seconds=60)
        assert claimed is not None
        assert claimed.intent == "retry_workstream"
        assert claimed.payload == {
            "repository_id": "backend",
            "additional_attempts": 1,
            "requested_by": "akhilesh (via platform key)",
            "reason": "The repository was repaired out of band.",
        }
        # The identity the worker resolves stored credentials against stays the one the
        # submission established. AB-Feature-111's granted retry failed outright because the
        # audit sentence was used here instead, so the worker went looking for provider
        # credentials belonging to "akhilesh (via platform key)".
        assert claimed.requested_by == "platform-admin"
    finally:
        await database.drop_schema()
        await database.dispose()


class _FlakyRenewalQueue:
    """Wrap a queue so renewal fails a fixed number of times, as a busy pool does."""

    def __init__(self, inner: Any, *, failures: int) -> None:
        self._inner = inner
        self.remaining_failures = failures
        self.renewals = 0

    def __getattr__(self, name: str) -> Any:
        """Delegate everything this wrapper does not override."""
        return getattr(self._inner, name)

    async def renew(self, feature_id: str, *, owner: str, lease_seconds: int) -> None:
        """Raise while failures remain, then renew normally."""
        self.renewals += 1
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            msg = "QueuePool limit reached, connection timed out"
            raise TimeoutError(msg)
        await self._inner.renew(feature_id, owner=owner, lease_seconds=lease_seconds)


@pytest.mark.asyncio
async def test_a_transient_renewal_error_does_not_drop_a_running_claim(tmp_path: Path) -> None:
    """The queue lease now carries every long run, so it must survive a database blink.

    A renewal loop that ended on one error would let the entry lapse under work that was
    progressing, and another worker would claim it and burn the entry's attempts colliding
    with a run that was fine. This is the same defect that cost AB-Feature-108 seventy
    minutes on the action lease, in the place that now matters most.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'flaky-renew.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="flaky-renew-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        queue = _FlakyRenewalQueue(store.queue, failures=3)

        async def credentials_for(_owner_id: str) -> RequestScopedCredentials:
            return credentials

        class SlowExecutor:
            """Run long enough that several renewals are attempted."""

            async def execute_queued(self, feature_id: str, **kwargs: Any) -> None:
                del kwargs
                await asyncio.sleep(0.7)
                await store.execute_queued(feature_id, credentials=credentials)

            async def fail_queued(self, feature_id: str, **kwargs: Any) -> None:
                del feature_id, kwargs

            async def fail_exhausted_fault(self, feature_id: str, **kwargs: Any) -> None:
                del feature_id, kwargs

            async def feature_owes_another_step(self, feature_id: str, **kwargs: Any) -> bool:
                return await store.feature_owes_another_step(feature_id, **kwargs)

            async def stop_unprogressing_run(self, feature_id: str, **kwargs: Any) -> None:
                await store.stop_unprogressing_run(feature_id, **kwargs)

            async def feature_run_succeeded(self, feature_id: str) -> bool:
                return await store.feature_run_succeeded(feature_id)

        dispatcher = FeatureQueueDispatcher(
            queue=queue,
            executor=SlowExecutor(),
            credentials_for=credentials_for,
            lease_seconds=1,
            # The production floor is five seconds, which no fast test can observe. Lowering
            # it is what makes the retry behaviour itself testable.
            min_renewal_seconds=0.1,
        )

        assert await dispatcher.run_once() is True
        assert queue.remaining_failures == 0, "renewal must be retried rather than given up on"
        # The step completed under its own claim rather than being taken over mid-flight. One
        # claim is one step now, so what this holds a lease over -- and what the renewals were
        # protecting -- is the product manager reading the requirement.
        stepped = (await store.get_record("feature-login")).state
        assert stepped.status is FeatureWorkflowStatus.ANALYZING_PRD
        assert any(isinstance(item, TechnicalPRDArtifact) for item in stepped.artifacts)
        # And the entry it was still holding carries the rest of the feature to completion.
        await drain_feature_queue(store, credentials=credentials)
        assert (
            await store.get_record("feature-login")
        ).state.status is FeatureWorkflowStatus.COMPLETED
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_claim_a_dead_process_never_finished_is_picked_up_again(tmp_path: Path) -> None:
    """The property that makes this a queue rather than a background task.

    A worker that stops mid-run -- deployed over, killed, out of memory -- leaves its entry
    claimed. Without a lease that entry is lost and its feature sits queued forever, which is
    exactly the failure the whole change exists to avoid.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'lease.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="lease-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        queue = store.queue

        # A worker claims it with a lease measured in seconds, and then dies.
        claimed = await queue.claim(owner="worker-that-dies", lease_seconds=1)
        assert claimed is not None
        assert claimed.attempt == 1
        # Nothing else may take it while the lease holds.
        assert await queue.claim(owner="worker-two", lease_seconds=60) is None

        await asyncio.sleep(1.2)
        reclaimed = await queue.claim(owner="worker-two", lease_seconds=60)

        assert reclaimed is not None
        assert reclaimed.feature_id == claimed.feature_id
        # The attempt count carries across, so a feature that kills every worker it touches
        # eventually stops rather than cycling.
        assert reclaimed.attempt == 2
        await queue.finish(reclaimed.feature_id, succeeded=True)
        assert await queue.claim(owner="worker-three", lease_seconds=60) is None
        assert await queue.pending_count() == 0
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_claim_landing_on_a_running_feature_costs_it_nothing(tmp_path: Path) -> None:
    """The defect that retired AB-Feature-104 while its first attempt was still working.

    A live feature can outlive its own lease -- the renewal is an asyncio task and this
    platform's work blocks the loop for minutes at a time -- so a second worker takes the
    entry. It then cannot execute, because the run lock is held by the attempt that is still
    going. Counted as a failure, three of those in thirty seconds spend the whole budget and
    mark a working feature as needing a human, with no questions and no explanation.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'busy.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(
            database, mock_runner=FeatureWorkflowOrchestrator(), lock=_BusyLock()
        )
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="busy-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        queue = store.queue

        with pytest.raises(FeatureExecutionBusy):
            await store.execute_queued(
                "feature-login",
                credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            )

        # And the dispatcher answers that by putting the entry back, attempt returned.
        claimed = await queue.claim(owner="worker-one", lease_seconds=60)
        assert claimed is not None and claimed.attempt == 1
        await queue.defer(claimed.feature_id, error="another worker holds this feature")
        again = await queue.claim(owner="worker-two", lease_seconds=60)
        assert again is not None
        assert again.attempt == 1, "a collision must not spend one of the feature's attempts"
        # The feature itself is untouched: still queued, still answerable.
        assert (await store.get_record("feature-login")).state.status is (
            FeatureWorkflowStatus.PENDING
        )
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_queue_refusal_says_what_went_wrong(tmp_path: Path) -> None:
    """A feature that says it is waiting on you must say what it is waiting for.

    AB-Feature-104 reached a person as `waiting_for_human` with no questions, no failure
    summary, and a reason naming only a number of attempts. The summary was silently dropped
    because it is only ever written for a failed state, and the status had already been set
    to waiting before it was asked for.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'refused.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="refused-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )

        await store.fail_queued(
            "feature-login",
            event="feature_execution_failed",
            reason="The platform could not start this feature after 3 attempts.",
            error_type="WorkflowConflictError",
        )
        record = await store.get_record("feature-login")

        # Answerable, because every refusal the queue makes is something somebody can fix.
        assert record.state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
        summary = record.state.failure_summary
        assert summary is not None, "a refusal with no diagnosis is unreadable"
        # The classification the platform owns, not the exception type it happened to hit.
        # `WorkflowConflictError` said nothing about whose problem this was, and recording an
        # exception type here is how `DBAPIError` came to be a feature's stated root cause.
        assert summary.root_classification == "platform_defect"
        assert any("3 attempts" in item for item in summary.diagnostics)
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_stale_refusal_cannot_overwrite_a_feature_that_progressed(
    tmp_path: Path,
) -> None:
    """A claim that lost the race must not report failure over the run that won it."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'stale.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="stale-refusal-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)
        assert (await store.get_record("feature-login")).state.status is (
            FeatureWorkflowStatus.COMPLETED
        )

        await store.fail_queued(
            "feature-login",
            event="feature_execution_failed",
            reason="The platform could not start this feature after 3 attempts.",
        )

        record = await store.get_record("feature-login")
        assert record.state.status is FeatureWorkflowStatus.COMPLETED
        assert record.state.failure_summary is None
    finally:
        await database.drop_schema()
        await database.dispose()


class _BusyLock:
    """A run lock somebody else is always holding, as the Redis lock reports it.

    Only the run lock: creating the feature takes `feature:start:` first, and refusing that
    too would fail the test in setup rather than at the collision it is about.
    """

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        """Refuse a run acquisition the way `RedisWorkflowLock` does once it times out."""
        if key.startswith("feature:run:"):
            msg = "workflow operation is already in progress; retry shortly"
            raise WorkflowConflictError(msg)
        yield


@pytest.mark.asyncio
async def test_a_reference_survives_execution_and_a_restart(tmp_path: Path) -> None:
    """Immutable: nothing the workflow does may reassign the identity people are quoting."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'immutable.db'}")
    await database.create_schema()
    try:
        credentials = RequestScopedCredentials(openai_api_key=None, github_token=None)
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        created = await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="immutable-001",
            credentials=credentials,
            owner_id="platform-admin",
        )
        reference = created.record.state.reference
        await drain_feature_queue(store)

        # A second control plane over the same database is what a restarted process sees.
        restarted = SqlAlchemyFeatureControlPlane(
            database, mock_runner=FeatureWorkflowOrchestrator()
        )
        record = await restarted.get_record("feature-login")
        page = await restarted.list_features(limit=10, cursor=None)

        assert reference == "AB-Feature-1"
        assert record.state.reference == reference
        assert record.state.status is FeatureWorkflowStatus.COMPLETED
        assert [item.reference for item in page.features] == [reference]
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_every_pull_request_a_feature_opens_is_titled_with_its_reference(
    tmp_path: Path,
) -> None:
    """A reviewer looking at a list of pull requests can tell which feature each belongs to.

    The prefix used to be the internal id, which in the pilot runs produced
    `[AI Feature adunit-deactivate-live-086]`. Unique, and meaningless to the person reading it.
    """
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'titles.db'}")
    await database.create_schema()
    try:
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        await store.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="titles-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)

        pull_requests = await store.pull_requests("feature-login")

        assert len(pull_requests) == 2
        assert all(isinstance(item, PullRequestArtifact) for item in pull_requests)
        for artifact in pull_requests:
            assert artifact.title.startswith("[AB-Feature-1] "), artifact.title
        # Two repositories, so each title is qualified -- they would otherwise be identical.
        # By repository name rather than by role: neither was given one, and two titles both
        # reading "other:" say less than the names do.
        assert {item.title for item in pull_requests} == {
            "[AB-Feature-1] backend: Login audit trail",
            "[AB-Feature-1] frontend: Login audit trail",
        }
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_single_repository_feature_gets_an_unqualified_pull_request_title(
    tmp_path: Path,
) -> None:
    """One repository needs no qualifier; the title reads as somebody would have written it."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'single-title.db'}")
    await database.create_schema()
    try:
        payload = feature_payload()
        payload["repositories"] = [payload["repositories"][0]]
        store = SqlAlchemyFeatureControlPlane(database, mock_runner=FeatureWorkflowOrchestrator())
        await store.start(
            StartFeatureRequest.model_validate(payload),
            idempotency_key="single-title-001",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        await drain_feature_queue(store)

        [pull_request] = await store.pull_requests("feature-login")

        assert pull_request.title == "[AB-Feature-1] Login audit trail"
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_queued_live_feature_without_stored_credentials_says_so() -> None:
    """A live submission is refused up front rather than queued to fail on a worker.

    Execution outlives the request, so a header this request happens to carry is no use to the
    worker that will do the work -- and copying it into the queue row would mean persisting a
    provider secret in a table that is not built to hold one. So the credentials have to be in
    the profile, and the refusal has to say which one is missing.
    """
    app = create_app(platform_api_key=KEY, secret_store=_EmptySecretStore())
    payload = {**feature_payload(), "execution_mode": "live"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        refused = await client.post("/features/start", headers=AUTH, json=payload)
        listed = await client.get("/features", headers=AUTH)

    assert refused.status_code == 422
    assert "OpenAI" in refused.json()["detail"]
    assert "GitHub" in refused.json()["detail"]
    # Refused before persistence: nothing half-created is left behind.
    assert listed.json()["features"] == []


@pytest.mark.asyncio
async def test_setup_state_reports_each_missing_provider_separately() -> None:
    """Partial setup is shown as partial: one configured, one not."""
    store = _PartialSecretStore(configured={"openai"})
    app = create_app(platform_api_key=KEY, secret_store=store)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        partial = await client.get("/setup", headers=AUTH)
        store.configured.add("github")
        complete = await client.get("/setup", headers=AUTH)

    assert partial.json()["credentials_ready"] is False
    assert {item["label"]: item["configured"] for item in partial.json()["providers"]} == {
        "OpenAI": True,
        "GitHub": False,
    }
    assert complete.json()["credentials_ready"] is True
    assert complete.json()["credential_storage_available"] is True


@pytest.mark.asyncio
async def test_setup_state_reports_repositories_before_any_are_saved() -> None:
    """The second prerequisite: saved repositories, so a feature is not retyped every time."""
    app = create_app(platform_api_key=KEY)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        empty = await client.get("/setup", headers=AUTH)
        await client.post(
            "/repositories",
            headers=AUTH,
            json={
                "repository_url": "https://github.com/Appbroda/admanager_console-2.0",
                "default_branch": "master",
                "repository_type": "Backend",
            },
        )
        ready = await client.get("/setup", headers=AUTH)

    assert empty.json()["repositories_ready"] is False
    assert empty.json()["saved_repository_count"] == 0
    assert ready.json()["repositories_ready"] is True
    assert ready.json()["saved_repository_count"] == 1


@pytest.mark.asyncio
async def test_saved_repositories_are_created_edited_and_removed() -> None:
    """The whole lifecycle, including the fields the person does not supply."""
    app = create_app(platform_api_key=KEY)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        created = await client.post(
            "/repositories",
            headers=AUTH,
            json={
                # Deliberately with a `.git` suffix and a trailing slash: the same repository
                # saved in two spellings must be one row.
                "repository_url": "https://github.com/Appbroda/admanager_console-2.0.git",
                "repository_type": "Backend",
            },
        )
        duplicate = await client.post(
            "/repositories",
            headers=AUTH,
            json={"repository_url": "https://github.com/Appbroda/admanager_console-2.0"},
        )
        sibling = await client.post(
            "/repositories",
            headers=AUTH,
            json={
                "repository_url": "https://github.com/Appbroda/reporting-api",
                "repository_type": "Backend",
            },
        )
        listed = await client.get("/repositories", headers=AUTH)
        configuration_id = created.json()["configuration_id"]
        edited = await client.put(
            f"/repositories/{configuration_id}",
            headers=AUTH,
            json={
                "repository_url": "https://github.com/Appbroda/admanager_console-2.0",
                "default_branch": "develop",
                "repository_type": "Service",
            },
        )
        removed = await client.delete(f"/repositories/{configuration_id}", headers=AUTH)
        after = await client.get("/repositories", headers=AUTH)

    assert created.status_code == 201, created.text
    # Derived, never asked for. And the identifier is the server's.
    assert created.json()["name"] == "admanager_console-2.0"
    assert created.json()["repository_url"] == ("https://github.com/Appbroda/admanager_console-2.0")
    assert created.json()["default_branch"] == "master"
    assert created.json()["configuration_id"].startswith("repo-")
    assert duplicate.status_code == 422
    assert sibling.status_code == 201
    # Two repositories may share a type. Nothing about a saved label is exclusive.
    assert [item["repository_type"] for item in listed.json()["repositories"]] == [
        "Backend",
        "Backend",
    ]
    assert "Backend" in listed.json()["suggested_types"]
    assert edited.json()["default_branch"] == "develop"
    assert edited.json()["repository_type"] == "Service"
    assert edited.json()["configuration_id"] == configuration_id
    assert removed.status_code == 204
    assert [item["name"] for item in after.json()["repositories"]] == ["reporting-api"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "not-a-url",
        "ftp://github.com/owner/name",
        "https://token@github.com/owner/name",
        "https://github.com/owner/name?token=abc",
        "https://github.com",
    ],
)
async def test_a_saved_repository_url_is_validated_when_it_is_saved(url: str) -> None:
    """Rejected here rather than when a feature built from it fails to clone."""
    app = create_app(platform_api_key=KEY)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post("/repositories", headers=AUTH, json={"repository_url": url})

    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_saved_repositories_belong_to_the_identity_that_saved_them(tmp_path: Path) -> None:
    """An identifier is not an authorisation: another owner's id reaches nothing."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'repo-owners.db'}")
    await database.create_schema()
    try:
        directory = DatabaseRepositoryConfigurationDirectory(database)
        mine = await directory.create(
            owner_id="me",
            repository_url="https://github.com/owner/mine",
            default_branch="master",
            repository_type="Backend",
        )

        assert [item.name for item in await directory.list_for_owner("me")] == ["mine"]
        assert await directory.list_for_owner("somebody-else") == []
        assert await directory.delete(mine.configuration_id, owner_id="somebody-else") is False
        with pytest.raises(RepositoryConfigurationError):
            await directory.update(
                mine.configuration_id,
                owner_id="somebody-else",
                repository_url="https://github.com/owner/hijacked",
                default_branch="master",
                repository_type="Backend",
            )
        assert [item.name for item in await directory.list_for_owner("me")] == ["mine"]
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_feature_can_be_submitted_from_saved_repositories_of_the_same_type() -> None:
    """Saved repositories must not have introduced fixed roles: two Backends is a valid feature."""
    directory = InMemoryRepositoryConfigurationDirectory()
    for url in ("https://github.com/owner/payments-api", "https://github.com/owner/reporting-api"):
        await directory.create(
            owner_id="platform-admin",
            repository_url=url,
            default_branch="master",
            repository_type="Backend",
        )
    app = create_app(platform_api_key=KEY, repository_configurations=directory)
    saved = [item for item in await directory.list_for_owner("platform-admin")]
    payload = {
        **feature_payload(),
        "feature_id": "two-backends",
        "repositories": [
            {
                "repository_url": item.repository_url,
                "default_branch": item.default_branch,
                "required": True,
            }
            for item in saved
        ],
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        started = await client.post("/features/start", headers=AUTH, json=payload)
        await settle(app)
        feature = await client.get("/features/two-backends", headers=AUTH)

    assert started.status_code == 201, started.text
    # Identity is still derived from the URL, not from the saved label.
    assert {item["repository_id"] for item in feature.json()["repositories"]} == {
        "payments-api",
        "reporting-api",
    }
    assert feature.json()["repository_count"] == 2


class _EmptySecretStore:
    """A credential store that holds nothing, so a setup gate can be observed."""

    async def resolve(self, *, owner_id: str, provider: str) -> str | None:
        """Report every provider as unconfigured."""
        del owner_id, provider
        return None

    async def describe_all(self, *, owner_id: str) -> list[Any]:
        """Nothing configured."""
        del owner_id
        return []


class _PartialSecretStore(_EmptySecretStore):
    """A credential store where some providers are configured and others are not."""

    def __init__(self, *, configured: set[str]) -> None:
        """Record which providers this identity has set up."""
        self.configured = configured

    async def resolve(self, *, owner_id: str, provider: str) -> str | None:
        """Return an opaque value for a configured provider, and nothing otherwise."""
        del owner_id
        return "stored" if provider in self.configured else None


class ProviderFaultExecutor:
    """An executor whose feature always ends in a provider fault, never in its own defect."""

    def __init__(self) -> None:
        """Count the attempts, and record every terminal verdict reached about the feature."""
        self.attempts = 0
        self.recorded: list[str] = []

    async def execute_queued(self, feature_id: str, **kwargs: Any) -> None:
        """Fail the way the model provider does: unrelated to the feature or its repository."""
        del feature_id, kwargs
        self.attempts += 1
        raise FeatureProviderFault(LLMAdapterError("the model provider call failed"))

    async def fail_queued(self, feature_id: str, **kwargs: Any) -> None:
        """Refusals the queue decides without running anything do not belong to this test."""
        del feature_id, kwargs
        self.recorded.append("fail_queued")

    async def fail_exhausted_fault(self, feature_id: str, **kwargs: Any) -> None:
        """Record that the platform gave up on this feature and told a human."""
        del feature_id, kwargs
        self.recorded.append("fail_exhausted_fault")

    async def feature_owes_another_step(self, feature_id: str, **kwargs: Any) -> bool:
        """Never reached: every claim against this executor raises before it can be asked."""
        del feature_id, kwargs
        return False

    async def stop_unprogressing_run(self, feature_id: str, **kwargs: Any) -> None:
        """Never reached: this executor never completes a step, so it never takes many."""
        del feature_id, kwargs

    async def feature_run_succeeded(self, feature_id: str) -> bool:
        """Report the feature as unfinished; this double never completes one."""
        del feature_id
        return False


@pytest.mark.asyncio
async def test_a_provider_fault_spends_the_attempts_the_entry_reserved_for_it() -> None:
    """A dropped provider call is what the queue's attempt budget is for.

    AB-Feature-168 never spent it. The executor caught the fault, recorded the feature as
    waiting on its author, and returned normally, so the dispatcher closed the entry
    `succeeded` at attempt 1 of 3 with `last_error` empty -- while the console asked a person
    to press resume, three times in one morning, to do what the queue already had the budget
    to do by itself.
    """
    queue = InMemoryFeatureExecutionQueue()
    executor = ProviderFaultExecutor()
    await queue.enqueue(
        feature_id="feature-bulk-add",
        execution_mode="live",
        agent_platform="openai",
        requested_by="platform-admin",
    )

    async def credentials_for(owner_id: str) -> RequestScopedCredentials:
        del owner_id
        return RequestScopedCredentials(openai_api_key="key", github_token="token")

    dispatcher = FeatureQueueDispatcher(
        queue=queue,
        executor=executor,
        credentials_for=credentials_for,
        lease_seconds=30,
    )

    # Three claims, because that is the entry's whole budget. Each returns True: the
    # dispatcher found work, and the first two put it back rather than retiring it.
    assert [await dispatcher.run_once() for _ in range(3)] == [True, True, True]

    assert executor.attempts == 3, "every reserved attempt must reach the provider"
    # The feature is told about exactly once, and only after there is nothing left to try.
    # Before this it was told on the first fault, which is why an operator saw a stopped
    # feature while two thirds of its budget sat unused.
    assert executor.recorded == ["fail_exhausted_fault"]
    assert await queue.pending_count() == 0, "the entry must be terminal once its budget is gone"
    assert await dispatcher.run_once() is False, "an exhausted entry must not be claimed again"
