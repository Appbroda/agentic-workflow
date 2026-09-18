"""One Slack thread per feature: the dispatcher, the ledger, the guards, the failure table.

The spec's T1-T10 run here against SQLite with a fake Slack client; T11 and T12 need the row
locking and partial indexes only PostgreSQL has and live in `test_slack_postgres_tier.py`.
No test in this repository reaches Slack.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sqlalchemy import select, update
from structlog.testing import capture_logs

from adapters.slack_adapter import (
    PostedMessage,
    SlackAuthIdentity,
    SlackChannelInfo,
    SlackClient,
    SlackClientError,
    SlackFailureMode,
)
from services.logbook import LOGBOOK_TEMPLATES, LogbookEvent, feature_logbook
from services.slack_notifications import (
    SLACK_DETAILED_TEMPLATES,
    SLACK_EXCLUDED_TEMPLATES,
    SLACK_HUMAN_INTERACTION_TEMPLATES,
    SLACK_MILESTONE_TEMPLATES,
    SlackNotificationDispatcher,
    compose_reply,
    compose_root,
)
from state.enums import (
    HUMAN_INTERACTION_FEATURE_STATUSES,
    IN_FLIGHT_FEATURE_STATUSES,
    TERMINAL_FEATURE_STATUSES,
    FeatureWorkflowStatus,
)
from state.external_operations import ExternalOperationType
from storage.db import Database
from storage.models import FeatureWorkflowModel, SlackNotificationModel
from storage.slack_store import (
    InMemorySlackConfigurationDirectory,
    InMemorySlackUserLinkDirectory,
    SlackNotificationStore,
)
from tests.test_logbook import _artifact, _at, _feature_184, _operation, _state

SERVER_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------------------------------


async def _database(tmp_path: Path, name: str = "slack.db") -> Database:
    """One SQLite-backed schema, as the durable-state tests build theirs."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / name}")
    await database.create_schema()
    return database


async def _seed_feature(
    database: Database,
    feature_id: str,
    *,
    status: FeatureWorkflowStatus = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
    title: str = "Bulk add apps",
    reference: str | None = "AB-Feature-184",
    owner_id: str = "platform-admin",
) -> None:
    """One `feature_workflows` row: the anchor columns and the candidate scan read these."""
    now = datetime.now(UTC)
    async with database.session() as session:
        session.add(
            FeatureWorkflowModel(
                owner_id=owner_id,
                feature_id=feature_id,
                status=status,
                title=title,
                reference=reference,
                execution_mode="mock",
                agent_platform="openai",
                performance_tier="high",
                state_json={},
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()


async def _touch_feature(database: Database, feature_id: str) -> None:
    """Bump `updated_at`, the way any real state write does, so the scan re-selects it."""
    async with database.session() as session:
        await session.execute(
            update(FeatureWorkflowModel)
            .where(FeatureWorkflowModel.feature_id == feature_id)
            .values(updated_at=datetime.now(UTC))
        )
        await session.commit()


async def _age_root_claim(database: Database, feature_id: str, *, seconds: float) -> None:
    """Back-date the root claim, standing in for a worker that died holding it."""
    async with database.session() as session:
        await session.execute(
            update(FeatureWorkflowModel)
            .where(FeatureWorkflowModel.feature_id == feature_id)
            .values(
                slack_root_claimed_at=datetime.now(UTC) - timedelta(seconds=seconds),
                updated_at=FeatureWorkflowModel.updated_at,
            )
        )
        await session.commit()


class FakeSlackClient(SlackClient):
    """Records every call, returns synthetic ts values, can be primed with refusals.

    Records *before* raising, because the crash window under test is exactly "Slack has the
    message and the caller never learned it" -- a fake that raised without recording would
    test a window that does not exist.
    """

    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []
        self._queued_errors: list[SlackClientError | None] = []
        self.always_raise: SlackClientError | None = None
        self._counter = 0

    def prime(self, *errors: SlackClientError | None) -> None:
        """Queue per-call outcomes: an error to raise, or None for one success."""
        self._queued_errors.extend(errors)

    async def post_message(
        self,
        channel: str,
        text: str,
        *,
        blocks: list[dict[str, Any]] | None = None,
        thread_ts: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PostedMessage:
        """Record the call, then answer or refuse as primed."""
        self._counter += 1
        self.posts.append(
            {
                "channel": channel,
                "text": text,
                "blocks": blocks or [],
                "thread_ts": thread_ts,
                "metadata": metadata,
            }
        )
        if self.always_raise is not None:
            raise self.always_raise
        if self._queued_errors:
            queued = self._queued_errors.pop(0)
            if queued is not None:
                raise queued
        return PostedMessage(channel=channel, ts=f"1717000000.{self._counter:06d}")

    async def auth_test(self) -> SlackAuthIdentity:
        """A fixed workspace identity."""
        return SlackAuthIdentity(workspace_id="T123", workspace_name="example")

    async def channel_info(self, channel: str) -> SlackChannelInfo:
        """A live channel."""
        return SlackChannelInfo(channel_id=channel, channel_name="deploys", is_archived=False)

    @property
    def roots(self) -> list[dict[str, Any]]:
        """Every post made without a thread_ts: the root messages."""
        return [item for item in self.posts if item["thread_ts"] is None]

    @property
    def replies(self) -> list[dict[str, Any]]:
        """Every threaded post."""
        return [item for item in self.posts if item["thread_ts"] is not None]


def _refusal(
    mode: SlackFailureMode, code: str, *, retry_after: float | None = None
) -> SlackClientError:
    """One classified refusal, as the real adapter raises them."""
    return SlackClientError(
        f"slack call refused ({code})",
        mode=mode,
        error_code=code,
        endpoint="chat.postMessage",
        retry_after_seconds=retry_after,
    )


class FakeRecordSource:
    """`get_record` and `events_after`, from fixtures instead of a control plane."""

    def __init__(self) -> None:
        self.states: dict[str, Any] = {}
        self.events: dict[str, list[LogbookEvent]] = {}

    async def get_record(self, feature_id: str, *, scope: Any = None) -> Any:
        """The record shape the dispatcher reads: `.state` and nothing else."""
        return SimpleNamespace(state=self.states[feature_id])

    async def events_after(
        self, feature_id: str, *, after_id: int | None, limit: int, scope: Any = None
    ) -> list[Any]:
        """The persisted lifecycle events, in id order."""
        return [
            SimpleNamespace(
                id=item.id, timestamp=item.timestamp, event=item.event, details=item.details
            )
            for item in self.events.get(feature_id, [])
        ]


class FakeJournal:
    """`list_operations_for_feature`, from a plain list."""

    def __init__(self) -> None:
        self.operations: dict[str, list[Any]] = {}

    async def list_operations_for_feature(
        self, feature_id: str, *, operation_types: Any = None, limit: int = 200
    ) -> list[Any]:
        """This feature's journal rows."""
        return list(self.operations.get(feature_id, []))


class FakeSecretStore:
    """Resolves one token, or nothing."""

    def __init__(self, token: str | None = "xoxb-test-token") -> None:
        self.token = token

    async def resolve(self, *, owner_id: str, provider: str) -> str | None:
        """The stored secret for this pass only."""
        assert provider == "slack"
        return self.token


async def _configuration() -> InMemorySlackConfigurationDirectory:
    """An enabled configuration pointing at channel C-CONFIG."""
    directory = InMemorySlackConfigurationDirectory()
    await directory.save(
        enabled=True,
        channel_id="C-CONFIG",
        channel_name="deploys",
        token_owner_id="platform-admin",
        verbosity="milestones",
        console_base_url="https://console.example.com",
        updated_by="platform-admin",
    )
    return directory


def _dispatcher(
    database: Database,
    source: FakeRecordSource,
    fake: FakeSlackClient,
    configuration: InMemorySlackConfigurationDirectory,
    *,
    journal: FakeJournal | None = None,
    links: InMemorySlackUserLinkDirectory | None = None,
    secret_store: FakeSecretStore | None = None,
    user_directory: Any = None,
) -> SlackNotificationDispatcher:
    """The dispatcher exactly as the deployment composes it, minus the network.

    `user_directory` is absent by default, which means nothing is withheld -- an application
    assembled without identity has no second workspace for a thread to leak into. The tests
    that are about the withholding rule pass one.
    """
    return SlackNotificationDispatcher(
        store=SlackNotificationStore(database),
        configuration=configuration,
        user_links=links or InMemorySlackUserLinkDirectory(),
        control_plane=source,
        journal=journal,
        secret_store=secret_store or FakeSecretStore(),
        client_factory=lambda _token: fake,
        user_directory=user_directory,
    )


def _simple_state() -> Any:
    """One approved review: a single milestone bubble with a quote."""
    review = _artifact(
        "review",
        "007_review.backend.json",
        20,
        verdict="approved",
        summary="The endpoint validates every row before writing.",
        requirement_checks=[],
        findings=[],
        architecture_assessment="Fits.",
        security_assessment="Fine.",
        test_coverage_assessment="Covered.",
    )
    return _state(artifacts=[review], status=FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS)


async def _ledger_rows(database: Database) -> list[Any]:
    """Every ledger row, oldest claim first."""
    async with database.session() as session:
        rows = (
            (
                await session.execute(
                    select(SlackNotificationModel).order_by(SlackNotificationModel.claimed_at)
                )
            )
            .scalars()
            .all()
        )
        return [
            SimpleNamespace(
                dedup_key=row.dedup_key,
                status=row.status,
                attempts=row.attempts,
                template=row.template,
                error_code=row.error_code,
                retry_after_at=row.retry_after_at,
            )
            for row in rows
        ]


async def _anchor(database: Database, feature_id: str) -> tuple[str | None, str | None]:
    """The persisted anchor columns."""
    store = SlackNotificationStore(database)
    anchor = await store.get_anchor(feature_id)
    assert anchor is not None
    return anchor.channel_id, anchor.thread_ts


# ----------------------------------------------------------------------------------------------
# T9 — registry integrity, and T7 — the state family
# ----------------------------------------------------------------------------------------------


def test_every_delivered_key_exists_in_the_registry() -> None:
    """A renamed template must fail here, not silently match nothing in a channel."""
    named = (
        SLACK_MILESTONE_TEMPLATES
        | SLACK_HUMAN_INTERACTION_TEMPLATES
        | SLACK_DETAILED_TEMPLATES
        | SLACK_EXCLUDED_TEMPLATES
    )
    unknown = named - set(LOGBOOK_TEMPLATES)
    assert unknown == set(), f"slack sets name templates the registry does not define: {unknown}"


def test_every_registry_key_is_classified_exactly_once() -> None:
    """Adding a logbook template forces a Slack decision rather than a silence.

    The deliberate inversion of the tab's never-dropped rule: an unrecognized record in the
    tab renders generically because dropping it would hide something; an unrecognized
    template in Slack would be a message nobody chose to send. So every key must be placed,
    and placed once.
    """
    sets = (
        SLACK_MILESTONE_TEMPLATES,
        SLACK_HUMAN_INTERACTION_TEMPLATES,
        SLACK_DETAILED_TEMPLATES,
        SLACK_EXCLUDED_TEMPLATES,
    )
    for key in LOGBOOK_TEMPLATES:
        placements = sum(key in group for group in sets)
        assert placements == 1, f"{key} is classified {placements} times; it must be exactly 1"


def _family_memberships(member: FeatureWorkflowStatus) -> int:
    """How many of the three named families claim one status.

    `FAILED_REQUIRES_HUMAN` is the one documented dual membership: it is a terminal outcome
    *and* a question addressed to a person, and the human-interaction ping keys on the second
    reading. Everything else must be in exactly one family.
    """
    families = (
        TERMINAL_FEATURE_STATUSES,
        HUMAN_INTERACTION_FEATURE_STATUSES,
        IN_FLIGHT_FEATURE_STATUSES,
    )
    return sum(member in family for family in families)


def test_every_feature_status_falls_in_exactly_one_named_family() -> None:
    """Item 4 adds states in parallel; an unclassified one fails here, not by pinging nobody."""
    for member in FeatureWorkflowStatus:
        if member is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN:
            assert member in TERMINAL_FEATURE_STATUSES
            assert member in HUMAN_INTERACTION_FEATURE_STATUSES
            assert member not in IN_FLIGHT_FEATURE_STATUSES
            continue
        assert _family_memberships(member) == 1, (
            f"{member.value} must be classified into exactly one of TERMINAL / "
            "HUMAN_INTERACTION / IN_FLIGHT before the Slack CC can be trusted"
        )
    # `CHANGES_REQUESTED` is deliberately in-flight: the integration review asks for fixes
    # and the platform makes them. Nobody is waiting on a person.
    assert FeatureWorkflowStatus.CHANGES_REQUESTED in IN_FLIGHT_FEATURE_STATUSES


def test_the_family_assertion_actually_fires_for_an_unclassified_member() -> None:
    """Prove the exhaustiveness check bites, with a member no family claims."""

    class _NotAStatus:
        value = "a_state_item_4_added"

    assert _family_memberships(_NotAStatus()) == 0  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------------
# T1 — one root, ever; replies carry the feature's own anchor
# ----------------------------------------------------------------------------------------------


async def test_one_root_ever_and_replies_thread_to_the_features_own_anchor(
    tmp_path: Path,
) -> None:
    """Two passes post one root; a config re-point moves new features, never this thread."""
    database = await _database(tmp_path)
    try:
        await _seed_feature(database, "feature-184")
        source = FakeRecordSource()
        source.states["feature-184"] = _simple_state()
        fake = FakeSlackClient()
        configuration = await _configuration()
        dispatcher = _dispatcher(database, source, fake, configuration)

        first = await dispatcher.run_once()
        second = await dispatcher.run_once()

        assert first.roots_posted == 1 and first.replies_sent == 1
        assert second.roots_posted == 0 and second.replies_sent == 0
        assert len(fake.roots) == 1
        channel, thread = await _anchor(database, "feature-184")
        assert channel == "C-CONFIG" and thread is not None

        # Re-point the configuration and produce something new: the thread must not move.
        await configuration.save(
            enabled=True,
            channel_id="C-ELSEWHERE",
            channel_name="other",
            token_owner_id="platform-admin",
            verbosity="milestones",
            console_base_url=None,
            updated_by="platform-admin",
        )
        source.events["feature-184"] = [
            LogbookEvent(
                id=9,
                timestamp=_at(50),
                event="feature_waiting_for_human",
                details={"reason": "somebody has to decide"},
            )
        ]
        await _touch_feature(database, "feature-184")
        third = await dispatcher.run_once()

        assert third.replies_sent == 1 and third.roots_posted == 0
        assert len(fake.roots) == 1, "a config edit must never re-root an anchored feature"
        for reply in fake.replies:
            assert reply["thread_ts"] == thread
            assert reply["channel"] == "C-CONFIG", (
                "a config edit mid-run must not scatter the thread"
            )
    finally:
        await database.drop_schema()
        await database.dispose()


# ----------------------------------------------------------------------------------------------
# T2 — the root-post crash window
# ----------------------------------------------------------------------------------------------


async def test_crash_between_root_post_and_anchor_write_reposts_once(tmp_path: Path) -> None:
    """The shipped disposition: at most one duplicate root, one warning naming the feature."""
    database = await _database(tmp_path)
    try:
        await _seed_feature(database, "feature-184")
        source = FakeRecordSource()
        source.states["feature-184"] = _simple_state()
        fake = FakeSlackClient()
        # Slack receives the root; the answer is lost. The claim is now held by a dead pass.
        fake.prime(_refusal(SlackFailureMode.TRANSPORT, "slack_transport_503"))
        dispatcher = _dispatcher(database, source, fake, await _configuration())

        first = await dispatcher.run_once()
        assert first.roots_posted == 0 and len(fake.roots) == 1
        _, thread = await _anchor(database, "feature-184")
        assert thread is None, "an unacknowledged post must not be anchored"

        # Within the stale window nothing may take the claim over.
        held = await dispatcher.run_once()
        assert held.roots_posted == 0 and len(fake.roots) == 1

        await _age_root_claim(database, "feature-184", seconds=3600)
        with capture_logs() as logs:
            third = await dispatcher.run_once()
        assert third.roots_posted == 1
        assert len(fake.roots) == 2, "at most one duplicate root per crash window"
        _, thread = await _anchor(database, "feature-184")
        assert thread is not None, "the anchor settles"
        reposts = [item for item in logs if item["event"] == "slack_root_reposted_after_crash"]
        assert len(reposts) == 1 and reposts[0]["feature_id"] == "feature-184"
        # The metadata that keeps the channels:history reconciliation path open later.
        for root in fake.roots:
            assert root["metadata"]["event_payload"]["feature_id"] == "feature-184"
    finally:
        await database.drop_schema()
        await database.dispose()


async def test_crash_after_the_anchor_write_never_posts_a_second_root(tmp_path: Path) -> None:
    """An anchored feature is rooted, whatever happened to the pass that anchored it."""
    database = await _database(tmp_path)
    try:
        await _seed_feature(database, "feature-184")
        store = SlackNotificationStore(database)
        assert await store.claim_root(
            "feature-184", stale_claim_cutoff=datetime.now(UTC) - timedelta(seconds=300)
        )
        assert await store.write_anchor(
            "feature-184", channel_id="C-CONFIG", thread_ts="1717000000.000001"
        )
        source = FakeRecordSource()
        source.states["feature-184"] = _simple_state()
        fake = FakeSlackClient()
        dispatcher = _dispatcher(database, source, fake, await _configuration())

        summary = await dispatcher.run_once()

        assert summary.roots_posted == 0 and len(fake.roots) == 0
        assert all(item["thread_ts"] == "1717000000.000001" for item in fake.posts)
    finally:
        await database.drop_schema()
        await database.dispose()


# ----------------------------------------------------------------------------------------------
# T3 — idempotency across resume, and sequence immunity
# ----------------------------------------------------------------------------------------------


async def test_a_late_arriving_operation_shifts_sequences_and_reposts_nothing(
    tmp_path: Path,
) -> None:
    """The test that fails if anyone keys the ledger on `sequence`.

    A journal operation with an earlier timestamp is inserted after the first pass -- which
    shifts every later entry's render-time sequence -- and the re-run must send nothing:
    the identity `(record.kind, record.id, emission)` did not change.
    """
    database = await _database(tmp_path)
    try:
        await _seed_feature(database, "feature-184")
        state = _simple_state()
        events = [
            LogbookEvent(
                id=1,
                timestamp=_at(30),
                event="feature_waiting_for_human",
                details={"reason": "a person has to answer"},
            )
        ]
        source = FakeRecordSource()
        source.states["feature-184"] = state
        source.events["feature-184"] = events
        journal = FakeJournal()
        fake = FakeSlackClient()
        dispatcher = _dispatcher(database, source, fake, await _configuration(), journal=journal)

        first = await dispatcher.run_once()
        assert first.replies_sent == 2  # the review approval and the waiting-for-human stop
        sent_before = len(fake.posts)

        # The late arrival: journaled earlier than everything already narrated.
        late = _operation(
            ExternalOperationType.CLONE_REPOSITORY, "op-late", 1, repository_id="backend"
        )
        journal.operations["feature-184"] = [late]
        before = feature_logbook(state, lifecycle_events=events, operations=[])
        after = feature_logbook(state, lifecycle_events=events, operations=[late])
        shifted = {
            (str(item.record.kind), item.record.id, item.emission): item.sequence for item in before
        }
        assert any(
            shifted[(str(item.record.kind), item.record.id, item.emission)] != item.sequence
            for item in after
            if (str(item.record.kind), item.record.id, item.emission) in shifted
        ), "the fixture must actually shift sequences, or this test proves nothing"

        await _touch_feature(database, "feature-184")
        second = await dispatcher.run_once()

        assert second.replies_sent == 0 and second.roots_posted == 0
        assert len(fake.posts) == sent_before, "no dedup_key is ever sent twice"
        rows = await _ledger_rows(database)
        assert len({row.dedup_key for row in rows}) == len(rows)
    finally:
        await database.drop_schema()
        await database.dispose()


# ----------------------------------------------------------------------------------------------
# T4 — a send cannot fail or delay a run
# ----------------------------------------------------------------------------------------------


def test_nothing_on_the_run_path_can_reach_slack() -> None:
    """The structural half of non-negotiable 1, in the 57- technique.

    Reads the AST of every module under `server/workflows/` and every
    `server/services/feature_*.py` and asserts none imports the Slack module, the adapter,
    the ledger store, an SDK, or the HTTP client they would need to send without them. A
    future inline send fails here instead of shipping a run a Slack outage can hang.
    """
    forbidden = (
        "adapters.slack_adapter",
        "services.slack_notifications",
        "storage.slack_store",
        "slack_sdk",
        "httpx",
    )
    run_path = sorted((SERVER_ROOT / "workflows").rglob("*.py")) + sorted(
        (SERVER_ROOT / "services").glob("feature_*.py")
    )
    assert run_path, "the guard found no modules; the paths it protects moved"
    offenders: list[str] = []
    for module_path in run_path:
        imported: set[str] = set()
        for node in ast.walk(ast.parse(module_path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for name in imported:
            if name in forbidden or any(name.startswith(f"{item}.") for item in forbidden):
                offenders.append(f"{module_path.relative_to(SERVER_ROOT)} imports {name}")
    assert offenders == [], (
        f"a run must never be able to wait on Slack; move the send into the dispatcher: {offenders}"
    )


async def test_a_slack_client_that_always_raises_leaves_the_run_untouched(
    tmp_path: Path,
) -> None:
    """Effects, not commands: the durable feature state is identical before and after a
    dispatcher pass whose every Slack call raises -- not merely "the error was swallowed".

    A whole mock feature runs end to end through the real control plane first, so what is
    being compared is the state a deployment would actually hold.
    """
    from api.control_plane import RequestScopedCredentials
    from api.feature_schemas import StartFeatureRequest
    from services.feature_queue import DatabaseFeatureExecutionQueue
    from storage.feature_store import SqlAlchemyFeatureControlPlane
    from tests.support import drain_feature_queue
    from tests.test_feature_api import feature_payload
    from workflows.feature_workflow import FeatureWorkflowOrchestrator

    database = await _database(tmp_path)
    try:
        control_plane = SqlAlchemyFeatureControlPlane(
            database,
            mock_runner=FeatureWorkflowOrchestrator(),
            queue=DatabaseFeatureExecutionQueue(database),
        )
        await control_plane.start(
            StartFeatureRequest.model_validate(feature_payload()),
            idempotency_key="slack-t4b",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id="platform-admin",
        )
        await drain_feature_queue(control_plane)
        settled = await control_plane.get_record("feature-login")
        assert settled.state.status is FeatureWorkflowStatus.COMPLETED

        fake = FakeSlackClient()
        fake.always_raise = _refusal(SlackFailureMode.TRANSPORT, "slack_transport_503")
        dispatcher = SlackNotificationDispatcher(
            store=SlackNotificationStore(database),
            configuration=await _configuration(),
            user_links=InMemorySlackUserLinkDirectory(),
            control_plane=control_plane,
            journal=None,
            secret_store=FakeSecretStore(),
            client_factory=lambda _token: fake,
        )
        summary = await dispatcher.run_once()
        assert fake.posts, "the pass must actually have attempted sends for this to prove anything"
        assert summary.replies_sent == 0

        after = await control_plane.get_record("feature-login")
        assert after.state.model_dump() == settled.state.model_dump(), (
            "a Slack failure left a mark on the run's durable state"
        )
        events_before = await control_plane.events_after("feature-login", after_id=None, limit=2000)
        assert all(not item.event.startswith("slack") for item in events_before), (
            "nothing on a notification path may write a feature event"
        )
    finally:
        await database.drop_schema()
        await database.dispose()


# ----------------------------------------------------------------------------------------------
# T5 — the three failure modes
# ----------------------------------------------------------------------------------------------


async def test_transport_failures_retry_bounded_then_skip(tmp_path: Path) -> None:
    """Slack down: `failed`, retried next sweep to the attempt budget, then skipped for good."""
    database = await _database(tmp_path)
    try:
        await _seed_feature(database, "feature-184")
        source = FakeRecordSource()
        source.states["feature-184"] = _simple_state()
        fake = FakeSlackClient()
        dispatcher = _dispatcher(database, source, fake, await _configuration())
        # Root the feature first, cleanly, so the failing send is a reply with a ledger row.
        await dispatcher.run_once()
        posts_after_clean_pass = len(fake.posts)

        source.events["feature-184"] = [
            LogbookEvent(id=7, timestamp=_at(40), event="feature_cancelled", details={})
        ]
        await _touch_feature(database, "feature-184")
        fake.always_raise = _refusal(SlackFailureMode.TRANSPORT, "slack_transport_503")
        with capture_logs() as logs:
            for _ in range(6):
                await dispatcher.run_once()
        fake.always_raise = None

        rows = [row for row in await _ledger_rows(database) if row.status != "sent"]
        assert len(rows) == 1
        assert rows[0].status == "skipped"
        assert rows[0].attempts == 5, "the retry budget is five attempts, not a storm"
        assert rows[0].error_code == "slack_transport_503"
        assert len(fake.posts) == posts_after_clean_pass + 5
        failures = [item for item in logs if item["event"] == "slack_notification_failed"]
        assert len(failures) == 5, "one warning per attempt"
        assert all(item["error_code"] == "slack_transport_503" for item in failures)
        # Skipped is permanent: a healthy Slack later never resurrects the message.
        final = await dispatcher.run_once()
        assert final.replies_sent == 0
    finally:
        await database.drop_schema()
        await database.dispose()


async def test_a_rate_limit_is_not_retried_before_its_instant(tmp_path: Path) -> None:
    """A 429's Retry-After is honoured, and the warning is once per pass."""
    database = await _database(tmp_path)
    try:
        await _seed_feature(database, "feature-184")
        source = FakeRecordSource()
        source.states["feature-184"] = _simple_state()
        fake = FakeSlackClient()
        dispatcher = _dispatcher(database, source, fake, await _configuration())
        fake.prime(None)  # the root succeeds
        fake.prime(_refusal(SlackFailureMode.RATE_LIMITED, "slack_rate_limited", retry_after=3600))
        with capture_logs() as logs:
            await dispatcher.run_once()
        attempts_after_limit = len(fake.posts)

        immediately = await dispatcher.run_once()

        assert immediately.replies_sent == 0
        assert len(fake.posts) == attempts_after_limit, "nothing is retried before the instant"
        rows = [row for row in await _ledger_rows(database) if row.status == "failed"]
        assert len(rows) == 1 and rows[0].retry_after_at is not None
        warned = [item for item in logs if item["event"] == "slack_notification_rate_limited"]
        assert len(warned) == 1
    finally:
        await database.drop_schema()
        await database.dispose()


async def test_a_revoked_token_degrades_the_configuration_once(tmp_path: Path) -> None:
    """`invalid_auth`: not retryable, all sending stops, one warning on the transition."""
    database = await _database(tmp_path)
    try:
        await _seed_feature(database, "feature-184")
        source = FakeRecordSource()
        source.states["feature-184"] = _simple_state()
        fake = FakeSlackClient()
        fake.always_raise = _refusal(SlackFailureMode.TOKEN_REVOKED, "slack_api_invalid_auth")
        configuration = await _configuration()
        dispatcher = _dispatcher(database, source, fake, configuration)

        with capture_logs() as logs:
            await dispatcher.run_once()
            await dispatcher.run_once()

        stored = await configuration.get()
        assert stored is not None and stored.status == "degraded"
        assert stored.status_reason is not None and "re-save" in stored.status_reason
        disabled = [item for item in logs if item["event"] == "slack_delivery_disabled"]
        assert len(disabled) == 1, "once, on the transition -- not per message, not per pass"
        assert len(fake.posts) == 1, "a dead token is not a retry storm"
    finally:
        await database.drop_schema()
        await database.dispose()


async def test_an_archived_channel_degrades_and_the_anchor_is_untouched(tmp_path: Path) -> None:
    """`channel_not_found`: the config degrades; an anchored feature keeps its anchor."""
    database = await _database(tmp_path)
    try:
        await _seed_feature(database, "feature-184")
        store = SlackNotificationStore(database)
        await store.claim_root(
            "feature-184", stale_claim_cutoff=datetime.now(UTC) - timedelta(seconds=300)
        )
        await store.write_anchor("feature-184", channel_id="C-OLD", thread_ts="1717000000.000042")
        source = FakeRecordSource()
        source.states["feature-184"] = _simple_state()
        fake = FakeSlackClient()
        fake.always_raise = _refusal(
            SlackFailureMode.CHANNEL_UNAVAILABLE, "slack_api_channel_not_found"
        )
        configuration = await _configuration()
        dispatcher = _dispatcher(database, source, fake, configuration)

        with capture_logs() as logs:
            await dispatcher.run_once()

        stored = await configuration.get()
        assert stored is not None and stored.status == "degraded"
        disabled = [item for item in logs if item["event"] == "slack_delivery_disabled"]
        assert len(disabled) == 1
        assert await _anchor(database, "feature-184") == ("C-OLD", "1717000000.000042")
    finally:
        await database.drop_schema()
        await database.dispose()


# ----------------------------------------------------------------------------------------------
# T6 — CC
# ----------------------------------------------------------------------------------------------


async def test_cc_mentions_follow_the_scope_each_person_chose(tmp_path: Path) -> None:
    """Human-interaction mentions everyone opted in; a milestone mentions only `all`."""
    database = await _database(tmp_path)
    try:
        await _seed_feature(database, "feature-184")
        links = InMemorySlackUserLinkDirectory()
        await links.save("user-all", slack_user_id="U0AAAAAAA", notify_scope="all")
        await links.save("user-hi", slack_user_id="U0BBBBBBB", notify_scope="human_interaction")
        await links.save("user-none", slack_user_id="U0CCCCCCC", notify_scope="none")
        await links.save("user-no-id", slack_user_id=None, notify_scope="all")
        source = FakeRecordSource()
        source.states["feature-184"] = _simple_state()
        source.events["feature-184"] = [
            LogbookEvent(
                id=1,
                timestamp=_at(30),
                event="feature_waiting_for_human",
                details={"reason": "a person has to answer"},
            )
        ]
        fake = FakeSlackClient()
        dispatcher = _dispatcher(database, source, fake, await _configuration(), links=links)

        await dispatcher.run_once()

        def _mentions(post: dict[str, Any]) -> str:
            fragments = []
            for block in post["blocks"]:
                for element in block.get("elements", []) or []:
                    if isinstance(element, dict) and element.get("text", "").startswith("cc "):
                        fragments.append(element["text"])
            return " ".join(fragments)

        milestone = next(item for item in fake.replies if "reviewer approved" in item["text"])
        ping = next(item for item in fake.replies if "waiting for a person" in item["text"])
        assert "U0AAAAAAA" in _mentions(milestone)
        assert "U0BBBBBBB" not in _mentions(milestone)
        assert "U0AAAAAAA" in _mentions(ping) and "U0BBBBBBB" in _mentions(ping)
        for post in fake.posts:
            assert "U0CCCCCCC" not in _mentions(post), "scope none is never mentioned"
            assert "None" not in _mentions(post)
        assert all("U0CCCCCCC" not in str(post["blocks"]) for post in fake.posts)
    finally:
        await database.drop_schema()
        await database.dispose()


# ----------------------------------------------------------------------------------------------
# T8 — redaction
# ----------------------------------------------------------------------------------------------


async def test_the_slack_quote_is_the_tabs_quote(tmp_path: Path) -> None:
    """The body is the logbook's body: the quoted fragment matches what the tab renders."""
    state = _simple_state()
    entries = feature_logbook(state)
    approved = next(item for item in entries if item.template == "artifact.review.approved")
    assert approved.quote is not None

    body = compose_reply(approved)

    quoted_blocks = [
        block["text"]["text"]
        for block in body.blocks
        if block["type"] == "section" and block["text"]["text"].startswith(">")
    ]
    assert quoted_blocks == [f"> {approved.quote}"], (
        "the Slack quote must be byte-identical to the logbook entry's own quote"
    )


async def test_a_body_carrying_key_material_is_withheld_not_sent(tmp_path: Path) -> None:
    """The outbound backstop: `skipped`, one warning, and the body appears in no log."""
    pem = "-----BEGIN RSA PRIVATE KEY----- MIIEow..."
    database = await _database(tmp_path)
    try:
        await _seed_feature(database, "feature-184")
        stopped = _artifact(
            "child_workflow_result",
            "008_child_result.backend.json",
            24,
            feature_id="feature-184",
            parent_workflow_id="feature-184",
            child_workflow_id="feature-184:backend",
            repository_id="backend",
            workstream_id="ws-backend",
            branch_name="feature/184-backend",
            workspace_path="/workspaces/backend",
            code_completion_artifact_id=None,
            review_artifact_id=None,
            changed_files=[],
            validation_results=[],
            status="failed",
            blocking_issues=[pem],
            pull_request_readiness=False,
            contract_sections_consumed=[],
            contract_sections_implemented=[],
        )
        source = FakeRecordSource()
        source.states["feature-184"] = _state(
            artifacts=[stopped], status=FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS
        )
        fake = FakeSlackClient()
        dispatcher = _dispatcher(database, source, fake, await _configuration())

        with capture_logs() as logs:
            await dispatcher.run_once()

        withheld_rows = [row for row in await _ledger_rows(database) if row.status == "skipped"]
        assert len(withheld_rows) == 1
        assert withheld_rows[0].error_code == "slack_body_withheld_key_material"
        assert all(pem not in str(post) for post in fake.posts), "the body was sent anyway"
        warnings = [item for item in logs if item["event"] == "slack_body_withheld_key_material"]
        assert len(warnings) == 1
        assert warnings[0]["template"] == "artifact.child_workflow_result.failed"
        assert all("PRIVATE KEY" not in str(item) for item in logs), (
            "the withheld body leaked into a log line"
        )
    finally:
        await database.drop_schema()
        await database.dispose()


def test_the_composer_reads_only_the_published_entry_fields() -> None:
    """Redaction is structural: the composer's inputs are the four screened fields, proven
    by watching what it actually reads rather than promised in a docstring."""

    class _Probe:
        accessed: set[str]

        def __init__(self) -> None:
            object.__setattr__(self, "accessed", set())

        def __getattr__(self, name: str) -> str:
            self.accessed.add(name)
            return ""

    probe = _Probe()
    compose_reply(probe, mention_ids=["U0AAAAAAA"])  # type: ignore[arg-type]
    assert probe.accessed <= {"text", "detail", "quote", "quote_source"}, (
        f"the composer read {probe.accessed - {'text', 'detail', 'quote', 'quote_source'}}; "
        "every new input must already have passed the logbook's screen"
    )


def test_the_root_is_a_container_not_a_bubble() -> None:
    """The root authors no narrative: identity, facts, a link, nothing composed."""
    body = compose_root(
        reference="AB-Feature-184",
        title="Bulk add apps",
        repositories=["Backend API", "Web client"],
        agent_platform="anthropic",
        performance_tier="high",
        requested_by="akhilesh",
        console_url="https://console.example.com/ui/features/feature-184",
    )
    assert "AB-Feature-184" in body.text
    flattened = str(body.blocks)
    assert "Backend API, Web client" in flattened
    assert "anthropic" in flattened and "high" in flattened
    assert "https://console.example.com/ui/features/feature-184" in flattened


# ----------------------------------------------------------------------------------------------
# T10 — the 184 thread
# ----------------------------------------------------------------------------------------------


async def test_the_184_thread_tells_the_tabs_story_minus_the_journal(tmp_path: Path) -> None:
    """The 57- A1 acceptance, delivered: the integration refusal, the required fix, the fix
    attempt's platform failure, the salvage PRs -- and none of the per-command chatter."""
    database = await _database(tmp_path)
    try:
        await _seed_feature(
            database, "feature-184", status=FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
        )
        state, events = _feature_184()
        source = FakeRecordSource()
        source.states["feature-184"] = state
        source.events["feature-184"] = events
        journal = FakeJournal()
        journal.operations["feature-184"] = [
            _operation(ExternalOperationType.CLONE_REPOSITORY, "op-1", 2, repository_id="backend"),
            _operation(ExternalOperationType.RUN_TESTS, "op-2", 18, repository_id="backend"),
        ]
        fake = FakeSlackClient()
        dispatcher = _dispatcher(database, source, fake, await _configuration(), journal=journal)

        summary = await dispatcher.run_once()
        assert summary.roots_posted == 1

        rows = await _ledger_rows(database)
        sent_templates = [row.template for row in rows if row.status == "sent"]
        assert sent_templates.count("artifact.review.approved") == 2
        assert "artifact.integration_review.changes_requested" in sent_templates
        assert "artifact.integration_review.required_fix" in sent_templates
        assert "artifact.child_workflow_result.failed" in sent_templates
        assert sent_templates.count("artifact.pull_request") == 2
        assert "event.feature_failed" in sent_templates
        assert "event.feature_failed.next_action" in sent_templates
        assert not any(template.startswith("operation.") for template in sent_templates), (
            "the journal is per-command chatter and stays in the tab"
        )
        texts = "\n".join(item["text"] for item in fake.replies)
        blocks = "\n".join(str(item["blocks"]) for item in fake.replies)
        assert "Make POST /apps/bulk atomic" in blocks, "the required fix is quoted verbatim"
        assert "Pull request #318" in texts and "Pull request #204" in texts
        assert "Review the two pull requests" in blocks, "the next action reaches the thread"
    finally:
        await database.drop_schema()
        await database.dispose()


# ----------------------------------------------------------------------------------------------
# The configuration and link API surface
# ----------------------------------------------------------------------------------------------


async def test_the_slack_surface_is_configured_saved_and_checked() -> None:
    """The console round trip: save, read back, check -- and `slack` is a credentials row."""
    from httpx import ASGITransport, AsyncClient

    from main import create_app
    from services.secrets import InMemorySecretStore

    fake = FakeSlackClient()
    app = create_app(
        platform_api_key="slack-test-key",
        secret_store=InMemorySecretStore(),
        slack_client_factory=lambda _token: fake,
    )
    headers = {"Authorization": "Bearer slack-test-key"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        empty = await client.get("/slack-configuration", headers=headers)
        assert empty.status_code == 200 and empty.json()["configured"] is False

        saved = await client.put(
            "/slack-configuration",
            headers=headers,
            json={
                "enabled": True,
                "channel_id": "C12345",
                "channel_name": "deploys",
                "console_base_url": "https://console.example.com",
            },
        )
        assert saved.status_code == 200, saved.text
        body = saved.json()
        assert body["enabled"] is True and body["status"] == "active"
        assert body["credential_configured"] is False

        # The bot token is a credential: stored through the one existing surface, and it
        # appears in the credentials list because `describe_all` iterates the tuple.
        stored = await client.put(
            "/credentials/slack", headers=headers, json={"secret": "xoxb-secret-value"}
        )
        assert stored.status_code == 200, stored.text
        assert stored.json()["hint"] == "alue"
        providers = {
            item["provider"]
            for item in (await client.get("/credentials", headers=headers)).json()["credentials"]
        }
        assert "slack" in providers

        check = await client.post("/slack-configuration/check", headers=headers)
        assert check.status_code == 200, check.text
        verdict = check.json()
        assert verdict["provider"] == "slack" and verdict["verified"] == "accepted"

        refreshed = await client.get("/slack-configuration", headers=headers)
        assert refreshed.json()["credential_configured"] is True
        assert refreshed.json()["workspace_name"] == "example"

        link = await client.put(
            "/account/slack-link",
            headers=headers,
            json={"slack_user_id": "U0123ABCDEF", "notify_scope": "human_interaction"},
        )
        assert link.status_code == 200, link.text
        assert link.json()["notify_scope"] == "human_interaction"
        read_back = await client.get("/account/slack-link", headers=headers)
        assert read_back.json()["slack_user_id"] == "U0123ABCDEF"

        malformed = await client.put(
            "/account/slack-link",
            headers=headers,
            json={"slack_user_id": "not-a-member-id", "notify_scope": "all"},
        )
        assert malformed.status_code == 422
        assert "member ID" in malformed.json()["detail"]


async def test_a_missing_token_sends_nothing_and_says_so_once(tmp_path: Path) -> None:
    """An enabled configuration with no stored `slack` credential warns once and idles."""
    database = await _database(tmp_path)
    try:
        await _seed_feature(database, "feature-184")
        source = FakeRecordSource()
        source.states["feature-184"] = _simple_state()
        fake = FakeSlackClient()
        dispatcher = _dispatcher(
            database, source, fake, await _configuration(), secret_store=FakeSecretStore(None)
        )
        with capture_logs() as logs:
            await dispatcher.run_once()
            await dispatcher.run_once()
        assert fake.posts == []
        missing = [item for item in logs if item["event"] == "slack_token_missing"]
        assert len(missing) == 1
    finally:
        await database.drop_schema()
        await database.dispose()


async def test_a_non_admin_workspace_gets_no_thread_and_an_administrator_s_does(
    tmp_path: Path,
) -> None:
    """The §4.10 decision, as an effect on what Slack received.

    There is one enabled Slack configuration for the whole deployment, so every feature's
    thread lands in the same channel -- where other people read titles, statuses and whatever
    the summary quotes. That defeats workspace isolation through a side channel the API never
    sees, and the safe default is to withhold rather than to leak: a feature owned by a
    non-admin account gets no root and no replies.

    Both features are seeded and both are candidates. What separates them is whose workspace
    they are in, which is exactly the property under test.
    """
    database = await _database(tmp_path)
    try:
        await _seed_feature(database, "feature-admins", owner_id="platform-admin")
        await _seed_feature(database, "feature-theirs", owner_id="user-sam")
        source = FakeRecordSource()
        source.states["feature-admins"] = _simple_state()
        source.states["feature-theirs"] = _simple_state()
        fake = FakeSlackClient()
        # `platform-admin` holds `admin`; `user-sam` does not. Stated directly rather than
        # built through a directory, because what the dispatcher reads is exactly this one
        # answer and a real directory would only be a longer way of writing it.
        dispatcher = _dispatcher(
            database,
            source,
            fake,
            await _configuration(),
            user_directory=_Administrators({"platform-admin"}),
        )

        summary = await dispatcher.run_once()

        # One feature delivered, one withheld, and the summary counts only what it walked.
        assert summary.roots_posted == 1
        assert len(fake.roots) == 1
        # The withheld feature has no anchor at all: not a claimed root, not a thread.
        channel, thread = await _anchor(database, "feature-theirs")
        assert channel is None and thread is None
        admin_channel, admin_thread = await _anchor(database, "feature-admins")
        assert admin_channel == "C-CONFIG" and admin_thread is not None
    finally:
        await database.drop_schema()
        await database.dispose()


async def test_without_a_user_directory_nothing_is_withheld(tmp_path: Path) -> None:
    """An application assembled without identity delivers everything, as it always did.

    There is no second workspace in one of those for a thread to leak into, and a mock
    deployment that silently stopped notifying would be a confusing regression rather than a
    safety measure.
    """
    database = await _database(tmp_path)
    try:
        await _seed_feature(database, "feature-theirs", owner_id="user-sam")
        source = FakeRecordSource()
        source.states["feature-theirs"] = _simple_state()
        fake = FakeSlackClient()
        dispatcher = _dispatcher(database, source, fake, await _configuration())

        summary = await dispatcher.run_once()

        assert summary.roots_posted == 1
    finally:
        await database.drop_schema()
        await database.dispose()


class _Administrators:
    """The one read the dispatcher makes of the user directory."""

    def __init__(self, ids: set[str]) -> None:
        """Hold the set of accounts that hold `admin`."""
        self._ids = frozenset(ids)

    async def administrator_ids(self) -> frozenset[str]:
        """Return every account id holding `admin`."""
        return self._ids
