"""Durable state for Slack delivery: the configuration, the links, the ledger, the anchor.

Four concerns, one module, because they are read and written by the same two callers -- the
configuration API and the notification dispatcher -- and none of them is feature state. The
one thing here that touches `feature_workflows` is the thread anchor, and every write to it
is a conditional UPDATE that deliberately leaves `updated_at` alone: a Slack anchor write
must not make a stalled feature look alive to the abandoned-run sweep, and must not write a
feature event or touch feature state in any way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol, cast
from uuid import uuid4

from sqlalchemy import func, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError

from state.enums import TERMINAL_FEATURE_STATUSES
from storage.db import Database
from storage.models import (
    FeatureExecutionQueueModel,
    FeatureWorkflowModel,
    SlackNotificationModel,
    SlackUserLinkModel,
    SlackWorkspaceConfigurationModel,
)

# The opt-in vocabulary, closed like `SUPPORTED_PROVIDERS` is and for the same reason: an
# unknown scope would be stored, never matched, and quietly notify nobody.
NOTIFY_SCOPES = ("none", "human_interaction", "all")

# The configuration's own lifecycle vocabulary.
CONFIGURATION_STATUSES = ("active", "degraded", "disabled")

# `milestones` is the only verbosity v1 accepts; `detailed` is a defined template set whose
# delivery is deliberately deferred until somebody asks for it.
VERBOSITY_LEVELS = ("milestones",)

# Comparisons against "the newest send for this feature" need a floor for features that have
# never been sent anything. Timezone-aware epoch, so the comparison never mixes naive and
# aware datetimes.
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def dedup_key_for(feature_id: str, kind: str, record_id: str, emission: int) -> str:
    """The one spelling of a ledger key: the logbook entry's own identity, readable.

    Never `LogbookEntry.sequence` -- a render-time index that any late-arriving record
    shifts, which would re-post the whole tail of a thread.
    """
    return f"{feature_id}|{kind}|{record_id}|{emission}"


@dataclass(frozen=True, slots=True)
class SlackWorkspaceConfiguration:
    """The stored configuration, safe to show: it has no field a token could occupy."""

    configuration_id: str
    enabled: bool
    workspace_id: str | None
    workspace_name: str | None
    channel_id: str
    channel_name: str | None
    token_owner_id: str
    verbosity: str
    status: str
    status_reason: str | None
    console_base_url: str | None
    updated_by: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class SlackUserLink:
    """One person's self-entered Slack identity and how much they asked to hear."""

    user_id: str
    slack_user_id: str | None
    notify_scope: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class SlackNotificationRecord:
    """One ledger row, as the dispatcher reads it back."""

    notification_id: str
    feature_id: str
    dedup_key: str
    entry_kind: str
    entry_record_id: str
    entry_emission: int
    template: str
    status: str
    slack_ts: str | None
    attempts: int
    claimed_at: datetime
    sent_at: datetime | None
    retry_after_at: datetime | None
    error_code: str | None


@dataclass(frozen=True, slots=True)
class FeatureSlackAnchor:
    """One feature's thread anchor and just enough identity to root or reply."""

    feature_id: str
    reference: str | None
    title: str
    status: str
    agent_platform: str
    performance_tier: str
    channel_id: str | None
    thread_ts: str | None
    root_claimed_at: datetime | None


class SlackConfigurationDirectory(Protocol):
    """Read and write the deployment's one Slack configuration."""

    async def get(self) -> SlackWorkspaceConfiguration | None:
        """Return the configuration, or nothing when none was ever saved."""

    async def save(
        self,
        *,
        enabled: bool,
        channel_id: str,
        channel_name: str | None,
        token_owner_id: str,
        verbosity: str,
        console_base_url: str | None,
        updated_by: str,
        workspace_id: str | None = None,
        workspace_name: str | None = None,
    ) -> SlackWorkspaceConfiguration:
        """Create or replace the single configuration row. A save re-activates it."""

    async def record_workspace_identity(
        self, configuration_id: str, *, workspace_id: str, workspace_name: str
    ) -> None:
        """Store what `auth.test` said, for display."""

    async def mark_degraded(self, configuration_id: str, *, reason: str) -> bool:
        """Degrade the configuration, reporting whether this call made the transition.

        The report is what makes "one warning per transition" implementable: only the caller
        whose write actually changed the status logs the warning.
        """


class SlackUserLinkDirectory(Protocol):
    """Each person's own Slack link and opt-in scope."""

    async def get(self, user_id: str) -> SlackUserLink | None:
        """Return one person's link, or nothing when they never saved one."""

    async def save(
        self, user_id: str, *, slack_user_id: str | None, notify_scope: str
    ) -> SlackUserLink:
        """Create or replace one person's link."""

    async def list_links(self) -> list[SlackUserLink]:
        """Every stored link, for the dispatcher to decide mentions from in one read."""


class DatabaseSlackConfigurationDirectory:
    """The deployment's configuration row, kept in PostgreSQL."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def get(self) -> SlackWorkspaceConfiguration | None:
        """Return the configuration. The table is effectively singleton; read the newest."""
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(SlackWorkspaceConfigurationModel)
                    .order_by(SlackWorkspaceConfigurationModel.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        return _configuration_from(row) if row is not None else None

    async def save(
        self,
        *,
        enabled: bool,
        channel_id: str,
        channel_name: str | None,
        token_owner_id: str,
        verbosity: str,
        console_base_url: str | None,
        updated_by: str,
        workspace_id: str | None = None,
        workspace_name: str | None = None,
    ) -> SlackWorkspaceConfiguration:
        """Create or replace the single row. A save is an operator statement, so it clears
        any degraded status: re-saving the configuration is exactly the remedy the status
        banner asks for."""
        _require_scope_values(verbosity=verbosity)
        now = datetime.now(UTC)
        async with self._database.session() as session:
            existing = (
                await session.execute(
                    select(SlackWorkspaceConfigurationModel)
                    .order_by(SlackWorkspaceConfigurationModel.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if existing is None:
                existing = SlackWorkspaceConfigurationModel(
                    configuration_id=f"slack-config-{uuid4()}",
                    created_at=now,
                )
                session.add(existing)
            existing.enabled = enabled
            existing.channel_id = channel_id
            existing.channel_name = channel_name
            existing.token_owner_id = token_owner_id
            existing.verbosity = verbosity
            existing.console_base_url = console_base_url
            existing.updated_by = updated_by
            if workspace_id is not None:
                existing.workspace_id = workspace_id
            if workspace_name is not None:
                existing.workspace_name = workspace_name
            existing.status = "active" if enabled else "disabled"
            existing.status_reason = None
            existing.updated_at = now
            await session.commit()
            configuration = _configuration_from(existing)
        return configuration

    async def record_workspace_identity(
        self, configuration_id: str, *, workspace_id: str, workspace_name: str
    ) -> None:
        """Store what `auth.test` said. Display only; never part of any decision."""
        async with self._database.session() as session:
            await session.execute(
                update(SlackWorkspaceConfigurationModel)
                .where(SlackWorkspaceConfigurationModel.configuration_id == configuration_id)
                .values(workspace_id=workspace_id, workspace_name=workspace_name)
            )
            await session.commit()

    async def mark_degraded(self, configuration_id: str, *, reason: str) -> bool:
        """Degrade once: only the write that made the transition reports True."""
        async with self._database.session() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(SlackWorkspaceConfigurationModel)
                    .where(
                        SlackWorkspaceConfigurationModel.configuration_id == configuration_id,
                        SlackWorkspaceConfigurationModel.status != "degraded",
                    )
                    .values(status="degraded", status_reason=reason)
                ),
            )
            await session.commit()
            return bool(result.rowcount)


class InMemorySlackConfigurationDirectory:
    """The same contract without a database, for isolated applications and tests."""

    def __init__(self) -> None:
        self._row: SlackWorkspaceConfiguration | None = None

    async def get(self) -> SlackWorkspaceConfiguration | None:
        """Return the configuration, or nothing when none was ever saved."""
        return self._row

    async def save(
        self,
        *,
        enabled: bool,
        channel_id: str,
        channel_name: str | None,
        token_owner_id: str,
        verbosity: str,
        console_base_url: str | None,
        updated_by: str,
        workspace_id: str | None = None,
        workspace_name: str | None = None,
    ) -> SlackWorkspaceConfiguration:
        """Create or replace the single configuration."""
        _require_scope_values(verbosity=verbosity)
        now = datetime.now(UTC)
        previous = self._row
        self._row = SlackWorkspaceConfiguration(
            configuration_id=(previous.configuration_id if previous else f"slack-config-{uuid4()}"),
            enabled=enabled,
            workspace_id=workspace_id or (previous.workspace_id if previous else None),
            workspace_name=workspace_name or (previous.workspace_name if previous else None),
            channel_id=channel_id,
            channel_name=channel_name,
            token_owner_id=token_owner_id,
            verbosity=verbosity,
            status="active" if enabled else "disabled",
            status_reason=None,
            console_base_url=console_base_url,
            updated_by=updated_by,
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        return self._row

    async def record_workspace_identity(
        self, configuration_id: str, *, workspace_id: str, workspace_name: str
    ) -> None:
        """Store what `auth.test` said."""
        row = self._row
        if row is None or row.configuration_id != configuration_id:
            return
        self._row = replace(row, workspace_id=workspace_id, workspace_name=workspace_name)

    async def mark_degraded(self, configuration_id: str, *, reason: str) -> bool:
        """Degrade once."""
        row = self._row
        if row is None or row.configuration_id != configuration_id or row.status == "degraded":
            return False
        self._row = replace(row, status="degraded", status_reason=reason)
        return True


class DatabaseSlackUserLinkDirectory:
    """Per-user Slack links, kept beside the users they belong to."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def get(self, user_id: str) -> SlackUserLink | None:
        """Return one person's link."""
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(SlackUserLinkModel).where(SlackUserLinkModel.user_id == user_id)
                )
            ).scalar_one_or_none()
        return _link_from(row) if row is not None else None

    async def save(
        self, user_id: str, *, slack_user_id: str | None, notify_scope: str
    ) -> SlackUserLink:
        """Create or replace one person's link."""
        _require_scope_values(notify_scope=notify_scope)
        now = datetime.now(UTC)
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(SlackUserLinkModel).where(SlackUserLinkModel.user_id == user_id)
                )
            ).scalar_one_or_none()
            if row is None:
                row = SlackUserLinkModel(
                    link_id=f"slack-link-{uuid4()}", user_id=user_id, created_at=now
                )
                session.add(row)
            row.slack_user_id = slack_user_id
            row.notify_scope = notify_scope
            row.updated_at = now
            await session.commit()
            link = _link_from(row)
        return link

    async def list_links(self) -> list[SlackUserLink]:
        """Every stored link."""
        async with self._database.session() as session:
            rows = (await session.execute(select(SlackUserLinkModel))).scalars().all()
        return [_link_from(row) for row in rows]


class InMemorySlackUserLinkDirectory:
    """The same contract without a database."""

    def __init__(self) -> None:
        self._rows: dict[str, SlackUserLink] = {}

    async def get(self, user_id: str) -> SlackUserLink | None:
        """Return one person's link."""
        return self._rows.get(user_id)

    async def save(
        self, user_id: str, *, slack_user_id: str | None, notify_scope: str
    ) -> SlackUserLink:
        """Create or replace one person's link."""
        _require_scope_values(notify_scope=notify_scope)
        now = datetime.now(UTC)
        previous = self._rows.get(user_id)
        link = SlackUserLink(
            user_id=user_id,
            slack_user_id=slack_user_id,
            notify_scope=notify_scope,
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        self._rows[user_id] = link
        return link

    async def list_links(self) -> list[SlackUserLink]:
        """Every stored link."""
        return list(self._rows.values())


class SlackNotificationStore:
    """The ledger and the anchor: everything the dispatcher persists.

    Database-only on purpose. The dispatcher exists only where a durable read model exists,
    and an in-memory ledger would let a test pass while proving nothing about the unique
    constraint that is the actual idempotency mechanism.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    # -- The ledger ---------------------------------------------------------------------------

    async def ledger_for_feature(self, feature_id: str) -> dict[str, SlackNotificationRecord]:
        """Every ledger row for one feature, keyed for the dispatcher's one pass over it."""
        async with self._database.session() as session:
            rows = (
                (
                    await session.execute(
                        select(SlackNotificationModel).where(
                            SlackNotificationModel.feature_id == feature_id
                        )
                    )
                )
                .scalars()
                .all()
            )
        return {row.dedup_key: _notification_from(row) for row in rows}

    async def insert_claimed(
        self,
        *,
        feature_id: str,
        dedup_key: str,
        entry_kind: str,
        entry_record_id: str,
        entry_emission: int,
        template: str,
    ) -> SlackNotificationRecord | None:
        """Insert the claim that precedes every send. A duplicate insert loses, and losing
        is the point: the unique constraint is the idempotency mechanism, not a
        check-then-send."""
        row = SlackNotificationModel(
            notification_id=f"slack-notification-{uuid4()}",
            feature_id=feature_id,
            dedup_key=dedup_key,
            entry_kind=entry_kind,
            entry_record_id=entry_record_id,
            entry_emission=entry_emission,
            template=template,
            status="claimed",
            attempts=1,
            claimed_at=datetime.now(UTC),
        )
        try:
            async with self._database.session() as session:
                session.add(row)
                await session.commit()
                record = _notification_from(row)
        except IntegrityError:
            return None
        return record

    async def claim_retry(
        self,
        notification_id: str,
        *,
        expected_attempts: int,
        stale_claim_cutoff: datetime,
    ) -> bool:
        """Take one bounded retry, or lose to whoever already took it.

        A `failed` row is claimable outright; a `claimed` row only when its claim is stale --
        a worker that died between the claim and the send. The attempts guard is the
        compare-and-swap that keeps two sweeps from both taking the same retry.
        """
        async with self._database.session() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(SlackNotificationModel)
                    .where(
                        SlackNotificationModel.notification_id == notification_id,
                        SlackNotificationModel.attempts == expected_attempts,
                        or_(
                            SlackNotificationModel.status == "failed",
                            (SlackNotificationModel.status == "claimed")
                            & (SlackNotificationModel.claimed_at < stale_claim_cutoff),
                        ),
                    )
                    .values(
                        status="claimed",
                        attempts=expected_attempts + 1,
                        claimed_at=datetime.now(UTC),
                    )
                ),
            )
            await session.commit()
            return bool(result.rowcount)

    async def mark_sent(self, notification_id: str, *, slack_ts: str) -> None:
        """Record the proof of delivery."""
        async with self._database.session() as session:
            await session.execute(
                update(SlackNotificationModel)
                .where(SlackNotificationModel.notification_id == notification_id)
                .values(status="sent", slack_ts=slack_ts, sent_at=datetime.now(UTC))
            )
            await session.commit()

    async def mark_failed(
        self,
        notification_id: str,
        *,
        error_code: str,
        retry_after_at: datetime | None = None,
    ) -> None:
        """Record a retryable refusal and, for a rate limit, the instant Slack named."""
        async with self._database.session() as session:
            await session.execute(
                update(SlackNotificationModel)
                .where(SlackNotificationModel.notification_id == notification_id)
                .values(status="failed", error_code=error_code, retry_after_at=retry_after_at)
            )
            await session.commit()

    async def mark_skipped(self, notification_id: str, *, error_code: str) -> None:
        """Record the one permanent non-delivery outcome, with the platform's own reason."""
        async with self._database.session() as session:
            await session.execute(
                update(SlackNotificationModel)
                .where(SlackNotificationModel.notification_id == notification_id)
                .values(status="skipped", error_code=error_code)
            )
            await session.commit()

    # -- The anchor ---------------------------------------------------------------------------

    async def get_anchor(self, feature_id: str) -> FeatureSlackAnchor | None:
        """One feature's anchor and root-message identity, from indexed columns only."""
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(
                        FeatureWorkflowModel.feature_id,
                        FeatureWorkflowModel.reference,
                        FeatureWorkflowModel.title,
                        FeatureWorkflowModel.status,
                        FeatureWorkflowModel.agent_platform,
                        FeatureWorkflowModel.performance_tier,
                        FeatureWorkflowModel.slack_channel_id,
                        FeatureWorkflowModel.slack_thread_ts,
                        FeatureWorkflowModel.slack_root_claimed_at,
                    ).where(FeatureWorkflowModel.feature_id == feature_id)
                )
            ).one_or_none()
        if row is None:
            return None
        return FeatureSlackAnchor(
            feature_id=row.feature_id,
            reference=row.reference,
            title=row.title,
            status=str(row.status),
            agent_platform=row.agent_platform,
            performance_tier=row.performance_tier,
            channel_id=row.slack_channel_id,
            thread_ts=row.slack_thread_ts,
            root_claimed_at=_as_utc(row.slack_root_claimed_at),
        )

    async def claim_root(self, feature_id: str, *, stale_claim_cutoff: datetime) -> bool:
        """Claim the right to post this feature's root. Exactly one winner, decided by the
        database: only the caller whose rowcount is 1 posts.

        `updated_at` is explicitly written back to itself so its `onupdate` does not fire:
        an anchor write must not make a stalled feature look alive to the abandoned-run
        sweep, and must not change what the candidate scan reads.
        """
        async with self._database.session() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(FeatureWorkflowModel)
                    .where(
                        FeatureWorkflowModel.feature_id == feature_id,
                        FeatureWorkflowModel.slack_thread_ts.is_(None),
                        or_(
                            FeatureWorkflowModel.slack_root_claimed_at.is_(None),
                            FeatureWorkflowModel.slack_root_claimed_at < stale_claim_cutoff,
                        ),
                    )
                    .values(
                        slack_root_claimed_at=datetime.now(UTC),
                        updated_at=FeatureWorkflowModel.updated_at,
                    )
                ),
            )
            await session.commit()
            return bool(result.rowcount)

    async def write_anchor(self, feature_id: str, *, channel_id: str, thread_ts: str) -> bool:
        """Persist where the root landed. Conditional on the anchor still being empty, so a
        duplicate root from the crash window can never overwrite the one already recorded."""
        async with self._database.session() as session:
            result = cast(
                CursorResult[Any],
                await session.execute(
                    update(FeatureWorkflowModel)
                    .where(
                        FeatureWorkflowModel.feature_id == feature_id,
                        FeatureWorkflowModel.slack_thread_ts.is_(None),
                    )
                    .values(
                        slack_channel_id=channel_id,
                        slack_thread_ts=thread_ts,
                        updated_at=FeatureWorkflowModel.updated_at,
                    )
                ),
            )
            await session.commit()
            return bool(result.rowcount)

    # -- Candidate selection ------------------------------------------------------------------

    async def list_candidate_feature_ids(
        self, *, limit: int, terminal_tail_cutoff: datetime
    ) -> tuple[list[str], bool]:
        """Features that may need a root or a reply, bounded, oldest activity first.

        A feature qualifies when it is anchored, still in flight, or recently terminal (the
        bounded tail that keeps a final message from being missed) -- and either its
        `updated_at` is newer than the ledger's newest claim for it, or its ledger holds an
        unsettled row (`failed`, or a `claimed` whose worker may have died). The second arm
        is what makes "retried next sweep" true: without it a failed send would wait for the
        feature to change before being retried. Returns the page and whether it was
        truncated, because a silent cap reads as "everything was sent".
        """
        newest_claim = (
            select(func.max(SlackNotificationModel.claimed_at))
            .where(SlackNotificationModel.feature_id == FeatureWorkflowModel.feature_id)
            .scalar_subquery()
        )
        has_unsettled_rows = (
            select(SlackNotificationModel.notification_id)
            .where(
                SlackNotificationModel.feature_id == FeatureWorkflowModel.feature_id,
                SlackNotificationModel.status.in_(("failed", "claimed")),
            )
            .exists()
        )
        terminal = [item.value for item in TERMINAL_FEATURE_STATUSES]
        async with self._database.session() as session:
            rows = (
                await session.execute(
                    select(FeatureWorkflowModel.feature_id)
                    .where(
                        or_(
                            FeatureWorkflowModel.slack_thread_ts.is_not(None),
                            FeatureWorkflowModel.status.not_in(terminal),
                            FeatureWorkflowModel.updated_at > terminal_tail_cutoff,
                        ),
                        or_(
                            FeatureWorkflowModel.updated_at > func.coalesce(newest_claim, _EPOCH),
                            has_unsettled_rows,
                        ),
                    )
                    .order_by(FeatureWorkflowModel.updated_at)
                    .limit(limit + 1)
                )
            ).all()
        feature_ids = [row.feature_id for row in rows]
        return feature_ids[:limit], len(feature_ids) > limit

    async def owners_of(self, feature_ids: Sequence[str]) -> dict[str, str]:
        """Return whose workspace each of these features is in.

        One query for the page the sweep is about to walk, so deciding which features may be
        delivered costs one read rather than one per feature. Read here rather than joined
        into `list_candidate_feature_ids` because the delivery rule is the dispatcher's and
        this store should not be the place that knows which roles exist.
        """
        if not feature_ids:
            return {}
        async with self._database.session() as session:
            rows = (
                await session.execute(
                    select(FeatureWorkflowModel.feature_id, FeatureWorkflowModel.owner_id).where(
                        FeatureWorkflowModel.feature_id.in_(list(feature_ids))
                    )
                )
            ).all()
        return {row.feature_id: row.owner_id for row in rows}

    async def requested_by(self, feature_id: str) -> str | None:
        """Who asked for this feature, from its queue entry -- display only, may be gone."""
        async with self._database.session() as session:
            value = (
                await session.execute(
                    select(FeatureExecutionQueueModel.requested_by).where(
                        FeatureExecutionQueueModel.feature_id == feature_id
                    )
                )
            ).scalar_one_or_none()
        return value


def _configuration_from(row: SlackWorkspaceConfigurationModel) -> SlackWorkspaceConfiguration:
    """Read one configuration row into the frozen shape callers hold."""
    return SlackWorkspaceConfiguration(
        configuration_id=row.configuration_id,
        enabled=row.enabled,
        workspace_id=row.workspace_id,
        workspace_name=row.workspace_name,
        channel_id=row.channel_id,
        channel_name=row.channel_name,
        token_owner_id=row.token_owner_id,
        verbosity=row.verbosity,
        status=row.status,
        status_reason=row.status_reason,
        console_base_url=row.console_base_url,
        updated_by=row.updated_by,
        created_at=_require_utc(row.created_at),
        updated_at=_require_utc(row.updated_at),
    )


def _link_from(row: SlackUserLinkModel) -> SlackUserLink:
    """Read one link row into the frozen shape callers hold."""
    return SlackUserLink(
        user_id=row.user_id,
        slack_user_id=row.slack_user_id,
        notify_scope=row.notify_scope,
        created_at=_require_utc(row.created_at),
        updated_at=_require_utc(row.updated_at),
    )


def _notification_from(row: SlackNotificationModel) -> SlackNotificationRecord:
    """Read one ledger row into the frozen shape the dispatcher holds."""
    return SlackNotificationRecord(
        notification_id=row.notification_id,
        feature_id=row.feature_id,
        dedup_key=row.dedup_key,
        entry_kind=row.entry_kind,
        entry_record_id=row.entry_record_id,
        entry_emission=row.entry_emission,
        template=row.template,
        status=row.status,
        slack_ts=row.slack_ts,
        attempts=row.attempts,
        claimed_at=_require_utc(row.claimed_at),
        sent_at=_as_utc(row.sent_at),
        retry_after_at=_as_utc(row.retry_after_at),
        error_code=row.error_code,
    )


def _require_scope_values(*, notify_scope: str | None = None, verbosity: str | None = None) -> None:
    """Refuse a vocabulary value nothing would ever match."""
    if notify_scope is not None and notify_scope not in NOTIFY_SCOPES:
        msg = f"unknown notify scope: {notify_scope}"
        raise ValueError(msg)
    if verbosity is not None and verbosity not in VERBOSITY_LEVELS:
        msg = f"unknown verbosity: {verbosity}"
        raise ValueError(msg)


def _as_utc(value: datetime | None) -> datetime | None:
    """Treat a naive timestamp from a database without timezone support as UTC."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _require_utc(value: datetime) -> datetime:
    """As `_as_utc`, for a column that is NOT NULL."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


__all__ = [
    "CONFIGURATION_STATUSES",
    "NOTIFY_SCOPES",
    "VERBOSITY_LEVELS",
    "DatabaseSlackConfigurationDirectory",
    "DatabaseSlackUserLinkDirectory",
    "FeatureSlackAnchor",
    "InMemorySlackConfigurationDirectory",
    "InMemorySlackUserLinkDirectory",
    "SlackConfigurationDirectory",
    "SlackNotificationRecord",
    "SlackNotificationStore",
    "SlackUserLink",
    "SlackUserLinkDirectory",
    "SlackWorkspaceConfiguration",
    "dedup_key_for",
]
