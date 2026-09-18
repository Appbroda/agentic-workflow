"""One Slack thread per feature: selection, composition, and the dispatching sweep.

Item 9 (the logbook) already wrote the sentences; this module is transport, configuration
and idempotency, and composes no prose. Three rules carry the whole design:

**A send can never fail or delay a run.** Nothing in `workflows/` or the feature runtime
imports this module or the adapter -- an AST guard in the suite makes that a property of the
code. Delivery is a periodic sweep over the durable read model, so its latency is the sweep
interval: a ping up to 30 seconds late is the price of a run no Slack outage can touch.

**Copy comes from the logbook registry, verbatim.** The template sets below select from
`LOGBOOK_TEMPLATES` by key, and every body is a `LogbookEntry`'s own fields reformatted into
Slack blocks. An unknown template is NOT delivered -- the deliberate opposite of the tab's
never-dropped rule: an unrecognized record in the tab renders generically because dropping it
would hide something; an unrecognized template here would be a message nobody chose to send,
to a channel with people in it. Two registry-integrity tests keep that a decision rather
than a silence.

**Sends are idempotent across crash recovery.** Every reply is claimed in the
`slack_notifications` ledger before it is sent, keyed on the entry's own identity --
`(record.kind, record.id, emission)`, never `sequence`, which any late-arriving record
shifts. The unique constraint is the mechanism; a duplicate insert loses and the send is
skipped.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import structlog

from adapters.slack_adapter import SlackClient, SlackClientError, SlackFailureMode
from services.logbook import LOGBOOK_TEMPLATES, LogbookEntry, LogbookEvent, feature_logbook
from state.enums import HUMAN_INTERACTION_FEATURE_STATUSES
from storage.slack_store import (
    FeatureSlackAnchor,
    SlackConfigurationDirectory,
    SlackNotificationStore,
    SlackUserLinkDirectory,
    SlackWorkspaceConfiguration,
    dedup_key_for,
)
from tools.file_tools import carries_key_material

# ----------------------------------------------------------------------------------------------
# Event -> template selection
#
# Published sets of registry keys, in the registry's own table-like style. Every key below must
# exist in `LOGBOOK_TEMPLATES`, and every `LOGBOOK_TEMPLATES` key must appear in exactly one of
# the delivered or excluded sets -- both enforced by tests, so a renamed template is caught and
# a new template forces a decision rather than a silence.
# ----------------------------------------------------------------------------------------------

# The default verbosity: the moments a person watching a channel actually wants.
SLACK_MILESTONE_TEMPLATES: frozenset[str] = frozenset(
    {
        # Intake and planning.
        "artifact.prd",
        "artifact.technical_prd",
        "artifact.execution_graph",
        "artifact.integration_contract.approved",
        # A workstream settling.
        "artifact.review.approved",
        "artifact.review.changes_requested",
        "artifact.review.rejected",
        "artifact.child_workflow_result.approved",
        "artifact.child_workflow_result.failed",
        # Integration.
        "artifact.integration_review.approved",
        "artifact.integration_review.changes_requested",
        "artifact.integration_review.required_fix",
        # Delivery.
        "artifact.pull_request",
        "artifact.feature_completion.completed",
        "artifact.feature_completion.partial_failure",
        "artifact.feature_completion.failed",
        "artifact.feature_completion.cancelled",
        # Terminal and lifecycle.
        "event.feature_failed",
        "event.feature_failed.next_action",
        "event.feature_runtime_limit_reached",
        "event.feature_step_budget_exhausted",
        "event.feature_cancelled",
        "event.feature_retired_by_operator",
        "event.feature_published_by_person",
        "event.feature_run_abandoned",
        "event.unconfirmed_external_effect",
    }
)

# Always sent, always CC'd: the messages that exist because a person has to act.
SLACK_HUMAN_INTERACTION_TEMPLATES: frozenset[str] = frozenset(
    {
        "event.feature_waiting_for_human",
        "event.feature_credentials_missing",
        "artifact.technical_prd.questions",
        # The 49-B design-conflict stop reaches Slack through this key; its quote is
        # `metadata.operator_question`, where the terminal-triage, ledger and design-conflict
        # narratives actually live. The dispatcher takes the rendered bubble and re-reads
        # neither field.
        "artifact.child_workflow_result.operator_question",
        "artifact.child_workflow_result.waiting_for_contract_change",
        "artifact.integration_review.failed_requires_human",
        "artifact.contract_change_request",
        "artifact.repository_repair_proposal",
        "artifact.repository_reconnaissance.contradiction",
        # A feature that did not land publishes nothing on its own. This is the message that
        # replaces the pull-request link a partial feature used to post -- without it the
        # held state is silent, which is the 2026-08 defect exactly.
        "event.feature_publication_held",
    }
)

# The attempt-level feed. Defined so the registry partition is total and the classification
# is recorded, but v1 delivers milestones only: this is the setting that turns a readable
# thread into an unreadable one, and nobody has asked for it yet.
SLACK_DETAILED_TEMPLATES: frozenset[str] = frozenset(
    {
        "artifact.code_completion.completed",
        "artifact.code_completion.partially_completed",
        "artifact.code_completion.failed",
        "artifact.code_completion.repairs",
        "artifact.code_completion.repairs_declined",
        "artifact.code_completion.repairs_unavailable",
        "artifact.code_completion.assertion_guard",
        "artifact.code_completion.unreachable",
        "artifact.code_completion.reachability_bounded",
        "artifact.code_completion.misplaced",
        "artifact.code_completion.self_review",
        "artifact.code_completion.self_review_unavailable",
        "artifact.review.finding",
        "artifact.child_workflow_result.truncated",
        "artifact.child_workflow_result.degraded_retry",
        "artifact.child_workflow_result.provider_fault",
        "artifact.child_workflow_result.retry_strategy",
        "artifact.child_workflow_result.routing",
        "artifact.child_workflow_result.retry_refused",
        "workstream.retry_granted",
        "event.repository_planned_blind",
    }
)

# Never delivered at any verbosity. Every `operation.*` key is here by design: the journal is
# per-command chatter -- clone, install, lint, tests, build, push -- and a Slack thread that
# carries it is a thread nobody reads. It stays in the tab, one click from the root's link.
# The rest are either duplicates of a richer bubble the thread already carries, bookkeeping,
# or the unknown fallbacks -- which are excluded on purpose: a message nobody chose is worse
# in a channel than a generic bubble is in the tab.
SLACK_EXCLUDED_TEMPLATES: frozenset[str] = frozenset(
    {
        "operation.succeeded",
        "operation.running",
        "operation.failed",
        "operation.cancelled",
        "operation.unresolved",
        "operation.recorded",
        "operation.unknown_type",
        "artifact.technical_prd.answers",
        "artifact.repository_reconnaissance",
        "artifact.architecture",
        "artifact.task_plan",
        "artifact.integration_contract.draft",
        "artifact.integration_contract.superseded",
        "artifact.repository_execution_plan",
        "artifact.child_workflow_result.cancelled",
        "artifact.child_workflow_result.runtime",
        "artifact.child_workflow_result.advisory_verdict",
        "artifact.integration_review.failed",
        "artifact.unknown",
        "event.feature_started",
        "event.feature_queued",
        "event.repository_repair_completed",
        "event.repository_repair_rejected",
        "event.repository_repair_superseded",
        "event.feature_run_continued",
        "event.feature_resume_refused",
        "event.feature_resume_found_no_eligible_workstreams",
        "event.feature_cancellation_requested",
        "event.feature_cancellation_updated",
        "event.feature_completed",
        "event.unknown",
    }
)

# What v1 actually delivers. `detailed` stays out until an operator asks for it; there is no
# toggle to reach it yet, and that is a decision recorded in spec 60's Decisions section.
SLACK_DELIVERED_TEMPLATES: frozenset[str] = (
    SLACK_MILESTONE_TEMPLATES | SLACK_HUMAN_INTERACTION_TEMPLATES
)

# The read bounds, mirroring the logbook endpoint's own (`_LOGBOOK_EVENT_BOUND` /
# `_LOGBOOK_OPERATION_BOUND` in `api/feature_routes.py`): the dispatcher recomposes exactly
# what the endpoint serves, so it reads exactly as much.
_EVENT_BOUND = 2000
_OPERATION_BOUND = 1000

# How stale a root-post claim must be before another worker may take it over. Comfortably
# longer than one HTTP call, because taking over a live claim is what mints duplicate roots.
_ROOT_CLAIM_STALE_SECONDS = 300.0

# How long after a feature goes terminal it is still swept, so a final message is not missed.
_TERMINAL_TAIL_SECONDS = 3600.0

# Bounded retries for a retryable send, after which the row is skipped permanently.
_MAX_SEND_ATTEMPTS = 5

# A 429 with no Retry-After still must not be hammered; one sweep interval is the floor.
_DEFAULT_RETRY_AFTER_SECONDS = 60.0

# One pass reads at most this many features; the truncation is logged, never silent.
_CANDIDATE_LIMIT = 50

_METADATA_EVENT_TYPE = "feature_thread_rooted"


@dataclass(frozen=True, slots=True)
class SlackMessageBody:
    """One assembled outbound body: the fallback text and the blocks."""

    text: str
    blocks: list[dict[str, Any]]


@dataclass(slots=True)
class SweepSummary:
    """What one pass did, for tests and for the log line that says it did anything."""

    features_seen: int = 0
    roots_posted: int = 0
    replies_sent: int = 0
    sends_failed: int = 0
    withheld: int = 0
    truncated: bool = False


class _DeliveryDisabled(Exception):
    """Raised inside a pass when the configuration degraded; ends the pass, never escapes."""


def _escape(text: str) -> str:
    """Escape the three characters Slack's mrkdwn treats as control characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def compose_reply(entry: LogbookEntry, *, mention_ids: Sequence[str] = ()) -> SlackMessageBody:
    """One thread reply from one logbook entry, and from nothing else.

    The inputs are exactly `text`, `detail`, `quote` and `quote_source` -- every one already
    screened when the logbook rendered it, which is what makes redaction structural here
    rather than a second predicate. The mentions are Slack member IDs people entered
    themselves, appended as a context line so the sentence stays the registry's.
    """
    lines = [_escape(entry.text)]
    if entry.detail:
        lines.append(_escape(entry.detail))
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}}
    ]
    if entry.quote:
        quoted = "\n".join(f"> {line}" for line in _escape(entry.quote).splitlines() or [""])
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": quoted}})
        if entry.quote_source:
            blocks.append(
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": f"from {_escape(entry.quote_source)}"}],
                }
            )
    if mention_ids:
        mentions = " ".join(f"<@{member}>" for member in mention_ids)
        blocks.append(
            {"type": "context", "elements": [{"type": "mrkdwn", "text": f"cc {mentions}"}]}
        )
    return SlackMessageBody(text=entry.text, blocks=blocks)


def compose_root(
    *,
    reference: str | None,
    title: str,
    repositories: Sequence[str],
    agent_platform: str,
    performance_tier: str,
    requested_by: str | None,
    console_url: str | None,
) -> SlackMessageBody:
    """The root message: the only body this item authors, and it authors no narrative.

    A container, not a bubble -- the feature's identity and a way back to the console.
    Every field is one an existing UI surface already shows.
    """
    heading = f"*{_escape(reference or title)}*"
    if reference:
        heading = f"*{_escape(reference)} — {_escape(title)}*"
    facts = [
        f"Repositories: {_escape(', '.join(repositories) or 'none recorded')}",
        f"Platform: {_escape(agent_platform)} · Tier: {_escape(performance_tier)}",
    ]
    if requested_by:
        facts.append(f"Requested by: {_escape(requested_by)}")
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": heading}},
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": line} for line in facts],
        },
    ]
    if console_url:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"<{console_url}|Open in the console>"},
            }
        )
    fallback = f"{reference or title} — {title}" if reference else title
    return SlackMessageBody(text=fallback, blocks=blocks)


def console_feature_url(base_url: str | None, feature_id: str) -> str | None:
    """The console deep link, when the configuration knows where the console lives."""
    if not base_url or not base_url.strip():
        return None
    return f"{base_url.strip().rstrip('/')}/ui/features/{feature_id}"


def body_carries_key_material(body: SlackMessageBody) -> bool:
    """The one outbound backstop, asked of the assembled body through the one existing rule.

    Slack is an outbound channel to a chat history that is searchable forever, so the body is
    screened once more before the send -- with `carries_key_material`, never a new predicate:
    the reviewer's and git adapter's separately-maintained copies of this question had each
    drifted with the same bug before they were collapsed onto it. If the structural markers
    are ever insufficient here, the fix is a marker in `_KEY_MATERIAL_PATTERNS`, shared by
    every caller.
    """
    fragments = [body.text]
    for block in body.blocks:
        text = block.get("text")
        if isinstance(text, dict):
            fragments.append(str(text.get("text", "")))
        for element in block.get("elements", []) or []:
            if isinstance(element, dict):
                fragments.append(str(element.get("text", "")))
    return carries_key_material("\n".join(fragments))


class _RecordSource(Protocol):
    """The two reads the dispatcher makes of the control plane, and nothing more."""

    async def get_record(self, feature_id: str) -> Any:
        """Return one durable parent state."""

    async def events_after(self, feature_id: str, *, after_id: int | None, limit: int) -> list[Any]:
        """Return durable lifecycle events after a cursor, cheaply."""


class _OperationSource(Protocol):
    """The one read the dispatcher makes of the journal."""

    async def list_operations_for_feature(
        self, feature_id: str, *, operation_types: Any = None, limit: int = 200
    ) -> list[Any]:
        """Return this feature's journaled operations."""


class _SecretSource(Protocol):
    """The one read the dispatcher makes of the secret store."""

    async def resolve(self, *, owner_id: str, provider: str) -> str | None:
        """Return the secret for use in this pass only."""


class _AdministratorSource(Protocol):
    """The one read the dispatcher makes of the user directory.

    Whose features may be delivered to a channel the whole deployment can read. Narrow on
    purpose: the dispatcher has no business knowing anything else about an account, and a
    wider dependency would invite it to.
    """

    async def administrator_ids(self) -> frozenset[str]:
        """Return every account id holding `admin`."""


@dataclass(slots=True)
class _PassState:
    """Per-pass bookkeeping for the warnings that are once-per-pass, not once-per-message."""

    rate_limit_warned: bool = False
    summary: SweepSummary = field(default_factory=SweepSummary)


class SlackNotificationDispatcher:
    """The sweep that turns the durable read model into thread replies.

    Runs as its own periodic task, deliberately not a fourth `RecoveryService` sweep: the
    recovery sweeps guard against duplicated external effects, and a notification loop does
    not belong in that budget. Its own body is isolated the same way those sweeps are --
    `CancelledError` propagates, everything else is one structured error and a return -- so
    no pass can take the process down.
    """

    def __init__(
        self,
        *,
        store: SlackNotificationStore,
        configuration: SlackConfigurationDirectory,
        user_links: SlackUserLinkDirectory,
        control_plane: _RecordSource,
        journal: _OperationSource | None,
        secret_store: _SecretSource | None,
        client_factory: Callable[[str], SlackClient],
        user_directory: _AdministratorSource | None = None,
        candidate_limit: int = _CANDIDATE_LIMIT,
        max_send_attempts: int = _MAX_SEND_ATTEMPTS,
        root_claim_stale_seconds: float = _ROOT_CLAIM_STALE_SECONDS,
        terminal_tail_seconds: float = _TERMINAL_TAIL_SECONDS,
    ) -> None:
        self._store = store
        self._configuration = configuration
        self._user_links = user_links
        self._control_plane = control_plane
        self._journal = journal
        self._secret_store = secret_store
        self._client_factory = client_factory
        self._user_directory = user_directory
        self._candidate_limit = candidate_limit
        self._max_send_attempts = max_send_attempts
        self._root_claim_stale_seconds = root_claim_stale_seconds
        self._terminal_tail_seconds = terminal_tail_seconds
        # For the token-missing warning, which is once per transition rather than per pass.
        self._token_missing_warned = False
        # For the withheld-features warning, once per transition for the reason above.
        self._withholding_warned = False

    async def run_periodic(self, *, interval_seconds: float = 30.0) -> None:
        """Sweep on a fixed cadence. Delivery latency is this interval, and that is the
        price of the guarantee that no run ever waits on Slack -- nothing here should be
        "optimized" by adding an inline send for the urgent cases."""
        if interval_seconds <= 0:
            msg = "slack notification sweep interval must be positive"
            raise ValueError(msg)
        logger = structlog.get_logger("runtime.slack")
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # Provider text may carry channel names or fragments of bodies; keep only
                # the platform-owned category, and try again on the next bounded tick.
                logger.error("slack_notification_sweep_failed", error_type=type(error).__name__)
                continue

    async def run_once(self) -> SweepSummary:
        """One bounded pass: read the configuration, walk the candidates, send what is new."""
        logger = structlog.get_logger("runtime.slack")
        state = _PassState()
        configuration = await self._configuration.get()
        if configuration is None or not configuration.enabled or configuration.status != "active":
            return state.summary
        token = await self._resolve_token(configuration)
        if token is None:
            return state.summary
        client = self._client_factory(token)
        now = datetime.now(UTC)
        candidates, truncated = await self._store.list_candidate_feature_ids(
            limit=self._candidate_limit,
            terminal_tail_cutoff=now - timedelta(seconds=self._terminal_tail_seconds),
        )
        state.summary.truncated = truncated
        if truncated:
            # A silent cap reads as "everything was sent"; the rest is next pass's work.
            logger.warning("slack_candidate_scan_truncated", limit=self._candidate_limit)
        candidates = await self._deliverable(candidates, logger=logger)
        links = await self._user_links.list_links()
        for feature_id in candidates:
            state.summary.features_seen += 1
            try:
                await self._deliver_feature(
                    feature_id,
                    configuration=configuration,
                    client=client,
                    links=links,
                    state=state,
                )
            except _DeliveryDisabled:
                # The configuration just degraded; the rest of the pass would only repeat
                # the same refusal per feature and bury the one line that says what to do.
                break
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error(
                    "slack_feature_delivery_failed",
                    feature_id=feature_id,
                    error_type=type(error).__name__,
                )
                continue
        return state.summary

    async def _deliverable(self, candidates: list[str], *, logger: Any) -> list[str]:
        """Drop the features whose thread would leak one workspace's work into another.

        There is one enabled Slack configuration for the whole deployment, so every feature's
        thread posts into the same channel -- where other people read titles, statuses and
        whatever the summary quotes. That defeats workspace isolation through a side channel
        the API never sees, and no amount of care inside the API closes it.

        The safe default, and the one implemented here: **a feature owned by a non-admin
        account gets no Slack anchor and no replies.** An administrator's features still post,
        because the deployment's own work is what the channel was configured for.

        Two other answers exist and are the product owner's to choose. Accept the leak and
        tell people their feature titles are visible to the channel; or scope Slack
        configuration per user, which means dropping the partial unique index that makes it a
        singleton and re-deriving `token_owner_id` per feature -- a follow-up of comparable
        size to this change. `docs/AUTHENTICATION_AND_WORKSPACES.md` records the choice.

        With no user directory bound, nothing is withheld. That is an application assembled
        without identity -- an isolated test app, a mock deployment -- and in one of those
        there is no second workspace for anything to leak into.

        Withholding is logged once per transition, not per pass and not silently: a
        notification that never arrives is indistinguishable from one nothing had to say.
        """
        if self._user_directory is None or not candidates:
            return candidates
        administrators = await self._user_directory.administrator_ids()
        owners = await self._store.owners_of(candidates)
        # A feature whose owner is not in the map is one this store could not read an owner
        # for, which cannot happen for a row it just selected -- but if it ever does, the
        # answer is to withhold rather than to guess.
        deliverable = [
            feature_id for feature_id in candidates if owners.get(feature_id) in administrators
        ]
        withheld = len(candidates) - len(deliverable)
        if withheld and not self._withholding_warned:
            logger.warning(
                "slack_delivery_withheld_for_non_admin_workspaces",
                withheld=withheld,
                reason=(
                    "one Slack channel serves the whole deployment, so a non-admin's feature "
                    "thread would show their work to everybody in it"
                ),
            )
            self._withholding_warned = True
        elif not withheld:
            self._withholding_warned = False
        return deliverable

    async def _resolve_token(self, configuration: SlackWorkspaceConfiguration) -> str | None:
        """Resolve the bot token for this pass only; it is held nowhere else."""
        logger = structlog.get_logger("runtime.slack")
        if self._secret_store is None:
            return None
        token = await self._secret_store.resolve(
            owner_id=configuration.token_owner_id, provider="slack"
        )
        if token is None:
            if not self._token_missing_warned:
                logger.warning("slack_token_missing", token_owner_id=configuration.token_owner_id)
                self._token_missing_warned = True
            return None
        self._token_missing_warned = False
        return token

    async def _deliver_feature(
        self,
        feature_id: str,
        *,
        configuration: SlackWorkspaceConfiguration,
        client: SlackClient,
        links: Sequence[Any],
        state: _PassState,
    ) -> None:
        """Root the feature if it has no thread yet, then reply everything new, in order."""
        anchor = await self._store.get_anchor(feature_id)
        if anchor is None:
            return
        if anchor.thread_ts is None:
            rooted = await self._root_feature(
                anchor, configuration=configuration, client=client, state=state
            )
            if not rooted:
                return
            anchor = await self._store.get_anchor(feature_id)
            if anchor is None or anchor.thread_ts is None:
                return
        await self._reply_new_entries(
            anchor, configuration=configuration, client=client, links=links, state=state
        )

    async def _root_feature(
        self,
        anchor: FeatureSlackAnchor,
        *,
        configuration: SlackWorkspaceConfiguration,
        client: SlackClient,
        state: _PassState,
    ) -> bool:
        """Post one root, claimed through the database so exactly one worker wins.

        The crash window, honestly: the sequence is claim -> post -> write the anchor, and a
        crash between the post and the write leaves a root Slack has and the database does
        not. `chat.postMessage` has no idempotency key, so the window cannot be closed by
        the API. The shipped disposition accepts at most one duplicate root per
        crash-inside-one-HTTP-call and logs `slack_root_reposted_after_crash` once, naming
        the feature -- and every root carries the feature id in its message metadata, so a
        deployment that later grants `channels:history` can reconcile instead of re-posting.
        """
        logger = structlog.get_logger("runtime.slack")
        now = datetime.now(UTC)
        stale_cutoff = now - timedelta(seconds=self._root_claim_stale_seconds)
        reposting_after_crash = (
            anchor.root_claimed_at is not None and anchor.root_claimed_at < stale_cutoff
        )
        claimed = await self._store.claim_root(anchor.feature_id, stale_claim_cutoff=stale_cutoff)
        if not claimed:
            # Another worker holds a live claim, or the anchor was written since we read it.
            return False
        record = await self._control_plane.get_record(anchor.feature_id)
        repositories = [item.name for item in record.state.repository_specs]
        requested_by = await self._store.requested_by(anchor.feature_id)
        body = compose_root(
            reference=anchor.reference,
            title=anchor.title,
            repositories=repositories,
            agent_platform=anchor.agent_platform,
            performance_tier=anchor.performance_tier,
            requested_by=requested_by,
            console_url=console_feature_url(configuration.console_base_url, anchor.feature_id),
        )
        if body_carries_key_material(body):
            state.summary.withheld += 1
            logger.warning(
                "slack_body_withheld_key_material",
                feature_id=anchor.feature_id,
                template="root",
            )
            return False
        try:
            posted = await client.post_message(
                configuration.channel_id,
                body.text,
                blocks=body.blocks,
                metadata={
                    "event_type": _METADATA_EVENT_TYPE,
                    "event_payload": {"feature_id": anchor.feature_id},
                },
            )
        except SlackClientError as error:
            await self._record_refusal(
                error, configuration=configuration, feature_id=anchor.feature_id, state=state
            )
            return False
        state.summary.roots_posted += 1
        if reposting_after_crash:
            logger.warning("slack_root_reposted_after_crash", feature_id=anchor.feature_id)
        wrote = await self._store.write_anchor(
            anchor.feature_id, channel_id=posted.channel, thread_ts=posted.ts
        )
        if not wrote:
            # Another worker's root was anchored first; ours is the duplicate the shipped
            # disposition accepts. The recorded anchor wins, and this post is left behind.
            logger.warning("slack_duplicate_root_not_anchored", feature_id=anchor.feature_id)
        return wrote

    async def _reply_new_entries(
        self,
        anchor: FeatureSlackAnchor,
        *,
        configuration: SlackWorkspaceConfiguration,
        client: SlackClient,
        links: Sequence[Any],
        state: _PassState,
    ) -> None:
        """Recompose the logbook exactly as the endpoint does, diff against the ledger, and
        post what is new -- in thread order, stopping at the first refusal so the thread
        never tells its story out of order."""
        logger = structlog.get_logger("runtime.slack")
        feature_id = anchor.feature_id
        record = await self._control_plane.get_record(feature_id)
        events = await self._control_plane.events_after(
            feature_id, after_id=None, limit=_EVENT_BOUND
        )
        operations: list[Any] = []
        if self._journal is not None:
            operations = await self._journal.list_operations_for_feature(
                feature_id, limit=_OPERATION_BOUND
            )
        entries = feature_logbook(
            record.state,
            lifecycle_events=[
                LogbookEvent(
                    id=item.id, timestamp=item.timestamp, event=item.event, details=item.details
                )
                for item in events
            ],
            operations=operations,
        )
        ledger = await self._store.ledger_for_feature(feature_id)
        status_needs_person = record.state.status in HUMAN_INTERACTION_FEATURE_STATUSES
        now = datetime.now(UTC)
        stale_claim_cutoff = now - timedelta(seconds=self._root_claim_stale_seconds)
        for entry in entries:
            if entry.template not in SLACK_DELIVERED_TEMPLATES:
                continue
            key = dedup_key_for(feature_id, str(entry.record.kind), entry.record.id, entry.emission)
            existing = ledger.get(key)
            if existing is not None and existing.status in ("sent", "skipped"):
                continue
            if existing is not None:
                # A failed or claimed row. Retry it here, bounded, or stop the feature's
                # batch: sending anything later while this one is unsettled would put the
                # thread out of order.
                if existing.attempts >= self._max_send_attempts:
                    await self._store.mark_skipped(
                        existing.notification_id,
                        error_code=existing.error_code or "slack_retry_budget_exhausted",
                    )
                    logger.warning(
                        "slack_notification_abandoned",
                        feature_id=feature_id,
                        template=entry.template,
                        attempts=existing.attempts,
                        error_code=existing.error_code,
                    )
                    continue
                if existing.retry_after_at is not None and existing.retry_after_at > now:
                    return
                taken = await self._store.claim_retry(
                    existing.notification_id,
                    expected_attempts=existing.attempts,
                    stale_claim_cutoff=stale_claim_cutoff,
                )
                if not taken:
                    return
                notification_id = existing.notification_id
                attempt = existing.attempts + 1
            else:
                inserted = await self._store.insert_claimed(
                    feature_id=feature_id,
                    dedup_key=key,
                    entry_kind=str(entry.record.kind),
                    entry_record_id=entry.record.id,
                    entry_emission=entry.emission,
                    template=entry.template,
                )
                if inserted is None:
                    # Another worker claimed this entry; its send may be in flight, and
                    # posting anything after it from here would race the thread's order.
                    return
                notification_id = inserted.notification_id
                attempt = 1
            is_human_interaction = (
                entry.template in SLACK_HUMAN_INTERACTION_TEMPLATES or status_needs_person
            )
            mention_ids = _mention_ids(links, human_interaction=is_human_interaction)
            body = compose_reply(entry, mention_ids=mention_ids)
            if body_carries_key_material(body):
                state.summary.withheld += 1
                await self._store.mark_skipped(
                    notification_id, error_code="slack_body_withheld_key_material"
                )
                logger.warning(
                    "slack_body_withheld_key_material",
                    feature_id=feature_id,
                    template=entry.template,
                )
                continue
            try:
                posted = await client.post_message(
                    # The feature's own anchor, never the configuration's current channel:
                    # a config edit mid-run must not scatter a thread.
                    anchor.channel_id or configuration.channel_id,
                    body.text,
                    blocks=body.blocks,
                    thread_ts=anchor.thread_ts,
                )
            except SlackClientError as error:
                await self._record_refusal(
                    error,
                    configuration=configuration,
                    feature_id=feature_id,
                    notification_id=notification_id,
                    attempt=attempt,
                    state=state,
                )
                return
            await self._store.mark_sent(notification_id, slack_ts=posted.ts)
            state.summary.replies_sent += 1

    async def _record_refusal(
        self,
        error: SlackClientError,
        *,
        configuration: SlackWorkspaceConfiguration,
        feature_id: str,
        state: _PassState,
        notification_id: str | None = None,
        attempt: int = 1,
    ) -> None:
        """The three-mode failure table. Every path returns normally; a Slack failure leaves
        no mark on the run's story -- it marks the configuration and the ledger."""
        logger = structlog.get_logger("runtime.slack")
        state.summary.sends_failed += 1
        if error.mode is SlackFailureMode.RATE_LIMITED:
            retry_after = error.retry_after_seconds or _DEFAULT_RETRY_AFTER_SECONDS
            if notification_id is not None:
                await self._store.mark_failed(
                    notification_id,
                    error_code=error.error_code,
                    retry_after_at=datetime.now(UTC) + timedelta(seconds=retry_after),
                )
            if not state.rate_limit_warned:
                # Once per pass: a limit that logs per message buries its own remedy.
                logger.warning("slack_notification_rate_limited", error_code=error.error_code)
                state.rate_limit_warned = True
            return
        if error.mode is SlackFailureMode.TRANSPORT:
            if notification_id is not None:
                await self._store.mark_failed(notification_id, error_code=error.error_code)
            logger.warning(
                "slack_notification_failed",
                feature_id=feature_id,
                error_code=error.error_code,
                attempt=attempt,
            )
            return
        if error.disables_delivery:
            reason = (
                "the bot token was rejected — re-save it in Settings"
                if error.mode is SlackFailureMode.TOKEN_REVOKED
                else "the channel cannot be delivered to (archived, deleted, or the bot was "
                "removed) — re-point the configuration; anchored features keep their anchor"
            )
            if notification_id is not None:
                # Left `failed` rather than skipped: once an operator re-saves the token or
                # re-points the channel, these rows are exactly what should go out.
                await self._store.mark_failed(notification_id, error_code=error.error_code)
            transitioned = await self._configuration.mark_degraded(
                configuration.configuration_id, reason=reason
            )
            if transitioned:
                # Once, on the transition -- a warning per event per sweep buries the one
                # line that says what to do.
                logger.warning(
                    "slack_delivery_disabled",
                    error_code=error.error_code,
                    status_reason=reason,
                )
            raise _DeliveryDisabled
        # A refusal about this one message. Permanent for the message, silent for the rest.
        if notification_id is not None:
            await self._store.mark_skipped(notification_id, error_code=error.error_code)
        logger.warning(
            "slack_notification_failed",
            feature_id=feature_id,
            error_code=error.error_code,
            attempt=attempt,
        )


def _mention_ids(links: Sequence[Any], *, human_interaction: bool) -> list[str]:
    """Who to CC: opted-in people only, by the scope they chose themselves.

    A human-interaction message mentions everyone with a scope; a milestone mentions only
    `all`. A row with no Slack member ID is never mentioned, whatever its scope says.
    """
    wanted = ("human_interaction", "all") if human_interaction else ("all",)
    return [
        link.slack_user_id for link in links if link.slack_user_id and link.notify_scope in wanted
    ]


def slack_template_partition() -> Mapping[str, frozenset[str]]:
    """The published selection sets, for the registry-integrity tests to read as data."""
    return {
        "milestones": SLACK_MILESTONE_TEMPLATES,
        "human_interaction": SLACK_HUMAN_INTERACTION_TEMPLATES,
        "detailed": SLACK_DETAILED_TEMPLATES,
        "excluded": SLACK_EXCLUDED_TEMPLATES,
    }


# Checked here as well as in the suite, because a selection that references a renamed
# template would otherwise ship a set that silently matches nothing.
_UNKNOWN_KEYS = (
    SLACK_MILESTONE_TEMPLATES
    | SLACK_HUMAN_INTERACTION_TEMPLATES
    | SLACK_DETAILED_TEMPLATES
    | SLACK_EXCLUDED_TEMPLATES
) - set(LOGBOOK_TEMPLATES)
if _UNKNOWN_KEYS:
    msg = f"slack template sets name keys the logbook registry does not define: {_UNKNOWN_KEYS}"
    raise RuntimeError(msg)


__all__ = [
    "SLACK_DELIVERED_TEMPLATES",
    "SLACK_DETAILED_TEMPLATES",
    "SLACK_EXCLUDED_TEMPLATES",
    "SLACK_HUMAN_INTERACTION_TEMPLATES",
    "SLACK_MILESTONE_TEMPLATES",
    "SlackMessageBody",
    "SlackNotificationDispatcher",
    "SweepSummary",
    "body_carries_key_material",
    "compose_reply",
    "compose_root",
    "console_feature_url",
    "slack_template_partition",
]
