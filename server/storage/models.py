"""SQLAlchemy persistence models for workflows, artifacts, and execution logs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from state.enums import (
    ApprovalState,
    ChildWorkflowStatus,
    ContractChangeRequestStatus,
    FeatureActionStatus,
    FeatureWorkflowStatus,
    IntegrationReviewStatus,
    LogLevel,
    RepositoryRepairStatus,
    WorkflowStatus,
)
from state.external_operations import (
    CompensationStatus,
    ExternalOperationStatus,
    ExternalOperationType,
)


def utc_now() -> datetime:
    """Return a timezone-aware timestamp suitable for persisted lifecycle events."""
    return datetime.now(UTC)


# How many steps one run of a feature may take before it is stopped whatever it is doing.
#
# A claim advances a feature by one step and re-queues it, so the loop that carries a feature
# from its PRD to its pull requests is durable rather than a coroutine -- and a durable loop
# that nothing bounds is a new failure mode, not a fixed one. A step that went on naming
# itself would re-queue for ever, spending a real provider call each time.
#
# Two hundred, which is far above anything a healthy feature reaches: a repository is one
# execution step per integration cycle, and the review cycles are already bounded well below
# ten. This is a backstop against a step that cannot make progress, not a scheduling policy.
# It lives here, beside the column that carries it, because the queue and this schema are the
# two things that have to agree about it, and a limit with two homes has none.
DEFAULT_FEATURE_STEP_LIMIT = 200

# The two kinds of credential `platform_api_tokens` holds. A `session` is what a password
# login mints -- bounded, and revoked when the password changes. An `api` token is what an
# administrator hands to a script, and a password change deliberately leaves it working.
# Named here rather than in the API layer because the column's server default is one of them
# and the two must not be able to disagree about the spelling.
TOKEN_KIND_SESSION = "session"
TOKEN_KIND_API = "api"


class Base(DeclarativeBase):
    """The declarative base shared by all persisted platform models."""


class WorkflowModel(Base):
    """The persisted lifecycle state and recovery data for one workflow."""

    __tablename__ = "workflows"

    workflow_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    workflow_schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_by_build_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    last_executor_build_revision: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[WorkflowStatus] = mapped_column(
        SqlEnum(
            WorkflowStatus,
            name="workflow_status",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
        default=WorkflowStatus.PENDING,
        index=True,
    )
    current_agent: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    approval_state: Mapped[ApprovalState] = mapped_column(
        SqlEnum(
            ApprovalState,
            name="approval_state",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
        default=ApprovalState.NOT_REQUIRED,
    )
    conversation_history: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    workspace_descriptor: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    checkpoints: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    artifacts: Mapped[list[ArtifactModel]] = relationship(
        back_populates="workflow", cascade="all, delete-orphan"
    )
    execution_logs: Mapped[list[ExecutionLogModel]] = relationship(
        back_populates="workflow", cascade="all, delete-orphan"
    )


class ArtifactModel(Base):
    """A serialized, validated artifact emitted by a workflow agent."""

    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    artifact_id: Mapped[str] = mapped_column(String(128), nullable=False)
    workflow_id: Mapped[str] = mapped_column(
        ForeignKey("workflows.workflow_id", ondelete="CASCADE"), nullable=False, index=True
    )
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    producer: Mapped[str] = mapped_column(String(128), nullable=False)
    validation_status: Mapped[str] = mapped_column(String(16), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    produced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    workflow: Mapped[WorkflowModel] = relationship(back_populates="artifacts")


class WorkflowRequestModel(Base):
    """A durable idempotency record containing only a request fingerprint and workflow link."""

    __tablename__ = "workflow_requests"

    idempotency_key: Mapped[str] = mapped_column(String(512), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    workflow_id: Mapped[str] = mapped_column(
        ForeignKey("workflows.workflow_id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class WorkflowEventModel(Base):
    """A durable lifecycle event emitted by the HTTP control plane or worker runtime."""

    __tablename__ = "workflow_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workflow_id: Mapped[str] = mapped_column(
        ForeignKey("workflows.workflow_id", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    event: Mapped[str] = mapped_column(String(128), nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class ExecutionLogModel(Base):
    """A durable, queryable log event produced during workflow execution."""

    __tablename__ = "execution_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    log_id: Mapped[str] = mapped_column(String(128), nullable=False)
    workflow_id: Mapped[str] = mapped_column(
        ForeignKey("workflows.workflow_id", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    level: Mapped[LogLevel] = mapped_column(
        SqlEnum(
            LogLevel,
            name="log_level",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
    )
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    event: Mapped[str] = mapped_column(String(128), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    workflow: Mapped[WorkflowModel] = relationship(back_populates="execution_logs")


class FeatureWorkflowModel(Base):
    """Durable parent feature lifecycle and serialized non-secret orchestration state.

    The root of the feature graph, and the one place ownership is recorded. Thirteen tables
    cascade from `feature_id`, and every route that reads one of them loads this row first,
    so the owner here decides who may reach all of them.
    """

    __tablename__ = "feature_workflows"
    __table_args__ = (
        Index("ix_feature_workflows_owner_id", "owner_id"),
        # Matches `list_features`' ordering exactly -- `(created_at DESC, feature_id DESC)`
        # -- so "my features, newest first" is one index read rather than an owner filter
        # over a scan of everybody's.
        Index(
            "ix_feature_workflows_owner_created",
            "owner_id",
            text("created_at DESC"),
            text("feature_id DESC"),
        ),
    )

    feature_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    # Whose workspace this feature is in. Set from the submitting actor at creation and never
    # reassigned: it decides who may see and act on the feature, and a feature that changes
    # hands would change who may spend money on it.
    #
    # Deliberately not `feature_execution_queue.requested_by`, which answers a different
    # question -- whose stored provider credentials the worker resolves. The two are equal
    # for a submission and must not be crossed for a granted retry; see §4.4 of
    # `docs/AUTHENTICATION_AND_WORKSPACES.md` and the comments at the `requeue` call sites.
    #
    # No foreign key to `platform_users.user_id`, for `ProviderCredentialModel.owner_id`'s
    # reason: the shared platform key resolves to an identity that need not be a row.
    # NOT NULL with no default, so a code path that forgets to set an owner fails at insert
    # time instead of quietly filing somebody's work under the administrator.
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[FeatureWorkflowStatus] = mapped_column(
        SqlEnum(
            FeatureWorkflowStatus,
            name="feature_workflow_status",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    # The identity people use for this feature: `AB-Feature-42`. Nullable only because a
    # snapshot written before references existed has none until the migration backfills it;
    # every feature created since carries one, and it never changes afterwards.
    reference: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    reference_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    execution_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    # Which model provider this feature runs on. Indexed nowhere and queried by nothing yet;
    # it is a column rather than a field inside `state_json` so a list view can report it
    # without hydrating every feature's artifacts, exactly as `execution_mode` is.
    agent_platform: Mapped[str] = mapped_column(String(16), nullable=False, server_default="openai")
    # How expensively that provider's model roles resolve for this feature. A column beside
    # the platform for the reason the platform is one: a list view reports a feature's cost
    # basis without hydrating its artifacts. `high` is what every pre-tier feature ran at.
    performance_tier: Mapped[str] = mapped_column(String(16), nullable=False, server_default="high")
    # The pinned role map a custom feature resolves through, copied at creation (G3). NULL
    # means this feature runs a tier, which is what every pre-setup row does -- no backfill.
    model_setup_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # Provenance only: which setup the snapshot came from, so a person can see it and the UI
    # can say "edited since". Never resolved through -- the snapshot is the execution
    # authority, and this id may point at an edited or deleted row.
    model_setup_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    merge_strategy: Mapped[str | None] = mapped_column(String(32), nullable=True)
    deployment_strategy: Mapped[str | None] = mapped_column(String(32), nullable=True)
    state_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    # The Slack thread anchor: where this feature's story is told, fixed for its life. The
    # channel is snapshotted here rather than read from the configuration, because a thread
    # lives in the channel it was rooted in -- re-pointing the config moves new features only.
    # Columns rather than `state_json` fields for two learned reasons: the dispatcher must
    # find features needing a root without hydrating every feature's artifacts, and a field
    # inside serialized state silently vanishes at `_replace_state` unless projected by name
    # (47- trap 1); an anchor that vanishes re-posts a root. All nullable: a feature with no
    # Slack configuration has no anchor, and that is not a missing value.
    slack_channel_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    slack_thread_ts: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # The root-post claim instant. Set by the conditional UPDATE that decides which of two
    # sweeping workers posts the root -- exactly one winner, decided by the database.
    slack_root_claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class SlackWorkspaceConfigurationModel(Base):
    """Where feature threads go: one workspace and channel per deployment.

    Effectively singleton: at most one row may be enabled, enforced by a partial unique
    index over a constant expression -- the database decides, not application care. The bot
    token is deliberately not a column; it lives in `provider_credentials` under the `slack`
    provider for `token_owner_id`, sealed like every other credential.
    """

    __tablename__ = "slack_workspace_configurations"
    __table_args__ = (
        Index(
            "uq_slack_workspace_configurations_enabled",
            text("(enabled)"),
            unique=True,
            postgresql_where=text("enabled"),
            sqlite_where=text("enabled"),
        ),
    )

    configuration_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # As returned by `auth.test`, stored for display. The name may go stale; the id is truth.
    workspace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    workspace_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # The id is what messages are keyed on; the name is display only and may go stale.
    channel_id: Mapped[str] = mapped_column(String(64), nullable=False)
    channel_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # Which identity's `slack` credential the dispatcher resolves for its sends.
    token_owner_id: Mapped[str] = mapped_column(String(128), nullable=False)
    # `milestones` is the only value v1 writes; `detailed` is a defined set awaiting demand.
    verbosity: Mapped[str] = mapped_column(String(16), nullable=False, default="milestones")
    # active | degraded | disabled. `degraded` is written by the dispatcher on a revoked
    # token or a dead channel, with the reason in the platform's own words beside it.
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    status_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Where the console lives, for the deep link in each thread's root message. Stored here
    # because no server setting knows the deployment's public origin, and the dispatcher has
    # no request to derive one from. Optional: without it the root simply carries no link.
    console_base_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    updated_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class DesignSourceConfigurationModel(Base):
    """Where this deployment's designs come from: one design account per deployment.

    Effectively singleton, and enforced the way `slack_workspace_configurations` is: a partial
    unique index over a constant expression, so two operators saving concurrently cannot both
    leave an enabled row behind. The design account's personal access token is deliberately
    not a column -- it lives in `provider_credentials` under the `figma` provider for
    `token_owner_id`, sealed like every other credential.

    `file_allowlist` is JSON rather than a child table because it is a short list of opaque
    keys read whole on every resolution and never joined against anything.
    """

    __tablename__ = "design_source_configurations"
    __table_args__ = (
        Index(
            "uq_design_source_configurations_enabled",
            text("(enabled)"),
            unique=True,
            postgresql_where=text("enabled"),
            sqlite_where=text("enabled"),
        ),
    )

    configuration_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    # Off is a state, not an absent row, for the reason the Slack configuration has one: an
    # operator turning designs off keeps the allowlist and the token owner they configured.
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Which identity's `figma` credential the resolver opens. Defaults to whoever saved.
    token_owner_id: Mapped[str] = mapped_column(String(128), nullable=False)
    # The file keys citations are permitted against. Empty means any file that token can
    # read, which is the right default for a single-team deployment and the wrong one for a
    # shared token -- so the control exists, and the response states its emptiness.
    file_allowlist: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    # active | degraded | disabled. Unlike the Slack precedent, the vocabulary is enforced on
    # every write in `design_source_store`; a declared-and-unchecked vocabulary is a comment.
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    status_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_by: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class SlackUserLinkModel(Base):
    """One person's own mapping to their Slack identity, and how much they asked to hear.

    Opt-in is explicit and self-service: default scope is `none`, the person enters their own
    Slack member ID, and the platform never looks one up by email. `user_id` is a plain
    string rather than a foreign key for `ProviderCredentialModel.owner_id`'s reason: the
    shared platform key resolves to an identity that is not a `platform_users` row.
    """

    __tablename__ = "slack_user_links"
    __table_args__ = (UniqueConstraint("user_id", name="uq_slack_user_links_user"),)

    link_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    # The member ID the person entered themselves (`U0123ABCDEF`). A row without one can
    # never be mentioned, whatever its scope says.
    slack_user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # none | human_interaction | all
    notify_scope: Mapped[str] = mapped_column(String(32), nullable=False, default="none")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class SlackNotificationModel(Base):
    """The delivery ledger: one row per Slack message the dispatcher decided to send.

    Deliberately not an `external_operations` row -- that journal's rows carry escalation
    power (sweeps, UNKNOWN_EXTERNAL_STATE, manual-cleanup requirements) and a chat message
    must never be able to stop a feature. The idempotency mechanism is the UNIQUE
    `dedup_key`, inserted *before* the send: a duplicate insert loses and the send is
    skipped, which is what makes two workers, or a resumed run re-walking its own history,
    safe without a lock. The key is the logbook entry's own identity -- record kind, record
    id and per-record emission -- never `LogbookEntry.sequence`, which any late-arriving
    record shifts.
    """

    __tablename__ = "slack_notifications"

    notification_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    feature_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    # `feature_id|record.kind|record.id|emission`, readable and unique.
    dedup_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True, index=True)
    entry_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    entry_record_id: Mapped[str] = mapped_column(String(256), nullable=False)
    entry_emission: Mapped[int] = mapped_column(Integer, nullable=False)
    template: Mapped[str] = mapped_column(String(128), nullable=False)
    # claimed | sent | failed | skipped
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    slack_ts: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    claimed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retry_after_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # The platform's classification (`slack_transport_503`). Never a provider message.
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)


class FeatureReferenceModel(Base):
    """One allocated user-facing feature number, and nothing else.

    The table exists so the database assigns the number rather than the application counting
    rows: `count(features) + 1` gives two simultaneous submissions the same answer, and no
    amount of application-side care fixes that. An insert here is an atomic allocation on
    both PostgreSQL and SQLite, and the row is never updated or deleted, which is what makes
    a reference immutable once a feature has one.

    Gaps are expected and harmless. A creation that rolls back after allocating leaves its
    number unused; a reference is an identifier, not a count.
    """

    __tablename__ = "feature_references"
    # Without this SQLite reuses the highest freed rowid. Nothing deletes from this table, so
    # it is belt and braces -- but the whole value of the table is that a number is never
    # handed out twice.
    __table_args__ = (
        UniqueConstraint("feature_id", name="uq_feature_references_feature"),
        {"sqlite_autoincrement": True},
    )

    reference_number: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_id: Mapped[str] = mapped_column(String(128), nullable=False)
    allocated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class FeatureExecutionQueueModel(Base):
    """A feature that has been accepted and is waiting to be executed.

    This is what makes "queued" a durable fact rather than a promise: `POST /features/start`
    commits the feature and this row in one transaction and answers, and execution begins
    afterwards. A dispatcher claims a row with a lease, so a process dying mid-run leaves
    work that another process picks up when the lease expires -- which is the difference
    between a queue and a background task.

    Provider credentials are deliberately absent. The row records who asked, and the
    dispatcher resolves that identity's stored credentials when it runs; a secret in a queue
    row would be a secret at rest in a table that is not built to hold one.
    """

    __tablename__ = "feature_execution_queue"

    feature_id: Mapped[str] = mapped_column(
        ForeignKey("feature_workflows.feature_id", ondelete="CASCADE"), primary_key=True
    )
    execution_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    # Which provider's stored credential the worker must resolve before it can run this
    # entry. Here for the same reason `execution_mode` is: the worker that runs the feature
    # is not the request that accepted it, and this decides which key it needs before it has
    # loaded any feature state. A provider *name*, never a key -- see the class docstring.
    agent_platform: Mapped[str] = mapped_column(String(16), nullable=False, server_default="openai")
    # The tier the worker resolves models at, beside the platform whose credential it needs,
    # and for the same reason: the worker that runs this entry is not the request that
    # accepted it.
    performance_tier: Mapped[str] = mapped_column(String(16), nullable=False, server_default="high")
    # The custom feature's pinned role map, beside the tier and for the same reason: the
    # worker that runs this entry is not the request that accepted it, and it must know which
    # platforms' credentials the entry needs before it has loaded any feature state.
    model_setup_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    model_setup_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    requested_by: Mapped[str] = mapped_column(String(128), nullable=False)
    # start | resume | retry_workstream. What the worker should do with the feature, so that
    # resuming and granting retries are queued work like starting is, rather than seventy
    # minutes of execution inside an HTTP request that nothing durable knows about.
    intent: Mapped[str] = mapped_column(String(32), nullable=False, default="start")
    # The arguments that intent needs: clarification answers, or a retry grant. Never
    # credentials -- the row records who asked, and the worker resolves their stored
    # credentials when it runs.
    intent_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    # queued | running | succeeded | failed
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    # How many times this entry has been *claimed without making progress*, not how many
    # times it has been claimed. A claim advances the feature by one step, so a feature with
    # forty steps takes forty claims; counting those against a budget of three would fail
    # every feature on its fourth step. Reset to zero whenever a step completes, so what is
    # left is consecutive crashes on one step -- which is the thing the budget was always for.
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    # How many steps this run has completed, and the ceiling on them. Not a record of where
    # the feature is -- `next_step` derives that from the feature's own artifacts and there is
    # deliberately no second copy of it -- but a bound on a loop that is now durable. The
    # ceiling is `DEFAULT_FEATURE_STEP_LIMIT`, defined once above and read by the queue from
    # here; it used to be the literal 200 in three separate places, which is the shape a limit
    # takes just before the three of them start disagreeing.
    steps: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    max_steps: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=DEFAULT_FEATURE_STEP_LIMIT,
        server_default=str(DEFAULT_FEATURE_STEP_LIMIT),
    )
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    queued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class RepositoryConfigurationModel(Base):
    """A repository somebody saved so they do not retype it for every feature.

    Organisational metadata only. The stable identity a workflow uses is still the
    `repository_id` derived from the URL at submission time, and the `type` here is a label
    the person chose -- it does not constrain what reconnaissance or the planner conclude
    about the repository's actual role.
    """

    __tablename__ = "repository_configurations"
    __table_args__ = (
        UniqueConstraint(
            "owner_id", "repository_url", name="uq_repository_configurations_owner_url"
        ),
    )

    configuration_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    # Derived from the URL rather than asked for, and recomputed whenever the URL changes.
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    repository_url: Mapped[str] = mapped_column(Text, nullable=False)
    default_branch: Mapped[str] = mapped_column(String(256), nullable=False)
    repository_type: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class ModelSetupModel(Base):
    """A user-authored model setup: per role, the platform, model, effort and output bound.

    Organisational data with execution consequences, which is why it is validated at save and
    again at feature start rather than trusted. A feature never resolves through this row --
    it snapshots the role map at creation (`feature_workflows.model_setup_snapshot`), so
    editing or deleting a setup never changes what a running or historical feature meant.
    """

    __tablename__ = "model_setups"
    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_model_setups_owner_name"),)

    setup_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    # The person's own label, shown in the selector and quoted in provenance sentences.
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    # The four-role map, keyed by ModelRole value. Exactly what a role resolution needs and
    # nothing that could go stale; never a credential, a variable name, or a base URL.
    roles: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class PRDAttachmentModel(Base):
    """One image somebody attached to a submission, with its bytes.

    In Postgres rather than on a volume or in an object store, deliberately. `workspace_data`
    is per-run scratch that clone and install churn, and a submission's evidence has to
    outlive the workspace of the run that read it; a new object store is infrastructure for a
    handful of megabytes per feature in a deployment that already runs a durable database.

    `media_type` is the *sniffed* type, never the declared one: the name and the browser's
    guess are both the uploader's, and this column is what the bytes are served back as.

    `content` goes null when a retired feature's attachments are purged, and the row stays.
    The submitted PRD artifact quotes `attachment_id` and `sha256`, so a deleted blob must
    still leave something for that reference to resolve to -- the record of what was
    submitted outliving the thing submitted is the whole point of keeping the row.
    """

    __tablename__ = "prd_attachments"

    attachment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    # Who uploaded it, under the name every other per-person store here uses. The only
    # reader while `feature_id` is null: an unbound attachment is nobody's but theirs.
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    # Null until the submission that references it is accepted. Deliberately not a foreign
    # key with a cascade: retiring a feature purges the content and keeps this row, and a
    # cascade would delete the evidence that something was submitted at all.
    feature_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    # As given, for the person reading it back. Never used as a path, never as an id.
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    media_type: Mapped[str] = mapped_column(String(64), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    # The identity of the bytes, and what the artifact quotes.
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    bound_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    purged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RepositorySpecModel(Base):
    """The persisted repository input selected for a parent feature workflow."""

    __tablename__ = "feature_repository_specs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_id: Mapped[str] = mapped_column(
        ForeignKey("feature_workflows.feature_id", ondelete="CASCADE"), nullable=False, index=True
    )
    repository_id: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    repository_url: Mapped[str] = mapped_column(Text, nullable=False)
    default_branch: Mapped[str] = mapped_column(String(256), nullable=False)
    local_workspace_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    required: Mapped[bool] = mapped_column(nullable=False)
    implementation_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )


class ChildWorkflowModel(Base):
    """Independently persisted child workstream status and artifact references."""

    __tablename__ = "feature_child_workflows"

    child_workflow_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    feature_id: Mapped[str] = mapped_column(
        ForeignKey("feature_workflows.feature_id", ondelete="CASCADE"), nullable=False, index=True
    )
    repository_id: Mapped[str] = mapped_column(String(128), nullable=False)
    workstream_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[ChildWorkflowStatus] = mapped_column(
        SqlEnum(
            ChildWorkflowStatus,
            name="feature_child_workflow_status",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
    )
    branch_name: Mapped[str] = mapped_column(String(256), nullable=False)
    # The branch the checkout is created from. `NULL` means the repository's configured
    # default, which is what every child written before revisions existed did, so the base
    # it decides needs no data migration. A revision writes the superseded branch here.
    base_branch: Mapped[str | None] = mapped_column(String(256), nullable=True)
    workspace_path: Mapped[str] = mapped_column(Text, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    code_completion_artifact_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    review_artifact_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    pull_request_artifact_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    blocking_issues: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    checkpoint_boundary: Mapped[str | None] = mapped_column(String(64), nullable=True)
    technology_profile: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    validation_plan: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    current_revision: Mapped[str | None] = mapped_column(String(128), nullable=True)
    current_validation_results: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    superseded_validation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    scoped_requirements: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    out_of_scope_requirements: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    preflight_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    preflight_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    blocking_setup_issues: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    selected_package_manager: Mapped[str | None] = mapped_column(String(32), nullable=True)
    layout_evidence: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    configured_validation_commands: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    test_availability: Mapped[str | None] = mapped_column(String(64), nullable=True)
    implementation_expectations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    production_files_changed: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    test_files_changed: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    configuration_files_changed: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    requirements_implemented: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    requirements_not_implemented: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    failure_classification: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retry_strategy: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # The configured model role selected for the next attempt. The `model_routing_state`
    # column beside it in a migrated database is deliberately not mapped: it held the retired
    # ladder's per-finding ledger, which nothing decides anything from any more.
    model_routing: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    implementation_retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    validation_retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    repository_setup_retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    integration_retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Which authority demanded the attempt this child is on, stamped where the counters above
    # are. Nullable-with-no-default rather than backfilled: `NULL` means a workstream retry,
    # which is what every attempt written before this column existed is, so the origin it
    # decides needs no data migration.
    targeted_attempt_kind: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Attempts a person granted after the platform stopped this repository, and the record of
    # each grant. Stored rather than derived: the counters alone cannot distinguish a
    # repository that was given more room from one whose configured limits were changed.
    granted_extra_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retry_grants: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    meaningful_change: Mapped[bool | None] = mapped_column(nullable=True)
    meaningful_change_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_refusal_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    production_diff_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    previous_attempt_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    test_diff_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    previous_test_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # The two clocks of the workstream's run: wall time, and wall time minus what was spent
    # inside classified provider faults. The runtime ceiling judges the charged clock.
    runtime_wall_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    runtime_charged_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Reconnaissance failed for this repository and the plan was written without its
    # evidence. Nullable-with-default rather than backfilled: a row from before the flag
    # existed genuinely does not know.
    planned_blind: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    planned_blind_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class FeatureArtifactModel(Base):
    """A serialized parent or child handoff artifact owned by a feature workflow."""

    __tablename__ = "feature_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_id: Mapped[str] = mapped_column(
        ForeignKey("feature_workflows.feature_id", ondelete="CASCADE"), nullable=False, index=True
    )
    artifact_id: Mapped[str] = mapped_column(String(256), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    produced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class IntegrationContractModel(Base):
    """One immutable contract version, indexed separately for audit and restart recovery."""

    __tablename__ = "feature_integration_contracts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_id: Mapped[str] = mapped_column(
        ForeignKey("feature_workflows.feature_id", ondelete="CASCADE"), nullable=False, index=True
    )
    artifact_id: Mapped[str] = mapped_column(String(256), nullable=False)
    contract_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class ContractChangeRequestModel(Base):
    """A durable request to change a parent-approved integration contract."""

    __tablename__ = "feature_contract_change_requests"

    change_request_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    feature_id: Mapped[str] = mapped_column(
        ForeignKey("feature_workflows.feature_id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[ContractChangeRequestStatus] = mapped_column(
        SqlEnum(
            ContractChangeRequestStatus,
            name="feature_contract_change_status",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
    )
    requested_by_repository_id: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class IntegrationReviewModel(Base):
    """A cross-repository review outcome, persisted separately from generic artifacts."""

    __tablename__ = "feature_integration_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_id: Mapped[str] = mapped_column(
        ForeignKey("feature_workflows.feature_id", ondelete="CASCADE"), nullable=False, index=True
    )
    artifact_id: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[IntegrationReviewStatus] = mapped_column(
        SqlEnum(
            IntegrationReviewStatus,
            name="feature_integration_review_status",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class FeaturePullRequestModel(Base):
    """A created coordinated PR, retained even when a later sibling PR fails."""

    __tablename__ = "feature_pull_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_id: Mapped[str] = mapped_column(
        ForeignKey("feature_workflows.feature_id", ondelete="CASCADE"), nullable=False, index=True
    )
    repository_id: Mapped[str] = mapped_column(String(128), nullable=False)
    artifact_id: Mapped[str] = mapped_column(String(256), nullable=False)
    pull_request_url: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class FeatureWorkflowRequestModel(Base):
    """Durable idempotency link for a parent feature start request."""

    __tablename__ = "feature_workflow_requests"

    idempotency_key: Mapped[str] = mapped_column(String(512), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_id: Mapped[str] = mapped_column(
        ForeignKey("feature_workflows.feature_id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class FeatureWorkflowEventModel(Base):
    """Credential-free structured events for parent observability and audit timelines."""

    __tablename__ = "feature_workflow_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_id: Mapped[str] = mapped_column(
        ForeignKey("feature_workflows.feature_id", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    event: Mapped[str] = mapped_column(String(128), nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)


class FeatureChatMessageModel(Base):
    """One durable message in a feature's assistant conversation.

    A proposed action is a column rather than something parsed back out of the message text:
    executing an action must never depend on re-reading prose the model wrote.
    """

    __tablename__ = "feature_chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_id: Mapped[str] = mapped_column(
        ForeignKey("feature_workflows.feature_id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    proposed_action: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    action_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    action_result: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The durable action this proposal became once somebody confirmed it. Nullable because a
    # pending or rejected proposal never becomes one, and because every message written
    # before durable actions existed has none.
    action_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class FeatureActionModel(Base):
    """Durable intent, ownership and outcome for one workflow-changing action.

    This is not a second external-operation journal: the effects an action causes are still
    journaled by ``ExternalOperationModel``, which is what makes them idempotent. This row
    answers the question that journal cannot -- *was this thing a person asked for carried
    out, and by whom* -- which previously lived in one nullable string column on a chat
    message and so could not survive the executor dying.
    """

    __tablename__ = "feature_actions"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_feature_actions_key"),)

    action_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    feature_id: Mapped[str] = mapped_column(
        ForeignKey("feature_workflows.feature_id", ondelete="CASCADE"), nullable=False, index=True
    )
    repository_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Who asked. Recorded on the action rather than derived from the transcript so an audit
    # can answer "who retried this repository" without reading chat prose.
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    actor_display_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    origin: Mapped[str] = mapped_column(String(16), nullable=False)
    origin_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    # The identity that makes a repeat safe. Covers the action type, its normalized payload
    # and the state it was decided against -- repository revision, contract version, repair
    # proposal -- so the same confirmed intent replays and a materially different one does not.
    input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[FeatureActionStatus] = mapped_column(
        SqlEnum(
            FeatureActionStatus,
            name="feature_action_status",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
        index=True,
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # Who is executing it, and until when. A crashed executor stops renewing; the lease then
    # expires and recovery can reason about the action instead of it sitting in EXECUTING
    # forever, which is exactly what the chat message column did.
    lease_owner: Mapped[str | None] = mapped_column(String(256), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Written in its own transaction after the domain service returns but before the action
    # is finalized. Recovery may only call an interrupted action successful when this proof
    # exists; successful provider calls alone do not prove the workflow state committed.
    domain_result_committed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    pending_result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    reconciled_by: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    reconciliation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reconciled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Recorded on success for audit. Recovery reads the exact action-operation link instead,
    # because an executor that crashed never got to write this final summary.
    external_operation_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)


class RepositoryRepairModel(Base):
    """A durable proposal to repair one repository's checked-in setup, and its decision.

    Indexed separately from the artifact history for the same reason contracts are: an
    operator asks "what is waiting on me for this repository", and that must not require
    deserializing every artifact the feature has produced.
    """

    __tablename__ = "feature_repository_repairs"

    repair_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    feature_id: Mapped[str] = mapped_column(
        ForeignKey("feature_workflows.feature_id", ondelete="CASCADE"), nullable=False, index=True
    )
    repository_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    artifact_id: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[RepositoryRepairStatus] = mapped_column(
        SqlEnum(
            RepositoryRepairStatus,
            name="feature_repository_repair_status",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
        index=True,
    )
    failure_classification: Mapped[str] = mapped_column(String(64), nullable=False)
    # The revision the diagnosis was written against. An approval is refused when the
    # repository has moved on, because the repair may no longer describe the checkout.
    proposed_at_revision: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resulting_revision: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class PlatformUserModel(Base):
    """One person the platform can tell apart from another, and how they prove it.

    It holds a password now, and it is still not a password database in the sense the
    original docstring meant: `password_hash` is nullable, and `NULL` means this account
    proves itself some other way. An identity provider's assertion can be added beside it
    without any of this changing, and an account that only ever holds tokens never gains one.

    `subject` is the login identifier as well as the field an external identity provider
    would map onto. It is stored lowercased and trimmed and looked up the same way -- see
    `normalize_subject` in `storage/user_store.py` -- because two rows differing only in case
    are two accounts for one person.
    """

    __tablename__ = "platform_users"
    __table_args__ = (UniqueConstraint("subject", name="uq_platform_users_subject"),)

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    # Stable across renames, and the field an external identity provider would map onto.
    subject: Mapped[str] = mapped_column(String(256), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    roles: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    # Disabled rather than deleted: an actor named in a year of audit records must still
    # resolve to a person, and deleting the row would leave those records pointing at nothing.
    disabled: Mapped[bool] = mapped_column(nullable=False, default=False)
    # The encoded scrypt string -- `scrypt$n$r$p$salt$hash` -- parameters and salt included,
    # so the work factor can be raised by rewriting rows rather than by a second migration.
    # Never a plaintext password, and never returned by any endpoint. NULL means this account
    # cannot password-login, which is the state of a federated account and of the
    # administrator before the bootstrap runs.
    password_hash: Mapped[str | None] = mapped_column(String(512), nullable=True)
    password_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Set when an administrator hands out a password, including the bootstrap's. It is what
    # makes forgetting to remove `BOOTSTRAP_ADMIN_PASSWORD` from the environment survivable.
    must_change_password: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class PlatformApiTokenModel(Base):
    """A credential that proves a request is one particular user.

    The token itself is never stored -- only its digest, so a database that leaks does not
    hand somebody the ability to act as its users. It cannot be shown again after issue for
    the same reason.
    """

    __tablename__ = "platform_api_tokens"
    __table_args__ = (UniqueConstraint("token_hash", name="uq_platform_api_tokens_hash"),)

    token_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("platform_users.user_id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # What somebody calls this token when deciding whether to revoke it.
    label: Mapped[str] = mapped_column(String(256), nullable=False)
    # `session` (minted by a password login, bounded) or `api` (minted by an administrator,
    # long-lived). A column rather than a `label` convention because two operations need it
    # and cannot be expressed without it: changing a password revokes this user's sessions
    # and must leave their automation tokens working, and so must "log out everywhere".
    # Defaulted to `api` so every token issued before this existed keeps its meaning.
    kind: Mapped[str] = mapped_column(
        String(16), nullable=False, default=TOKEN_KIND_API, server_default=TOKEN_KIND_API
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ProviderCredentialModel(Base):
    """One provider secret a user has asked the platform to keep, encrypted at rest.

    The ciphertext is bound to its owner and provider: a row copied into somebody else's
    name does not decrypt, because those values are the additional authenticated data. The
    plaintext is never a column, never returned by any endpoint, and never logged.
    """

    __tablename__ = "provider_credentials"
    __table_args__ = (
        UniqueConstraint("owner_id", "provider", name="uq_provider_credentials_owner_provider"),
    )

    credential_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    nonce: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    # Which encryption key sealed this, so a rotation can re-seal without guessing.
    key_version: Mapped[str] = mapped_column(String(32), nullable=False)
    # The last four characters, so somebody can tell which of their keys is configured
    # without the platform showing them a secret they already have.
    hint: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ExternalOperationModel(Base):
    """Durable pre-side-effect journal record shared by single and feature workflows."""

    __tablename__ = "external_operations"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_external_operations_key"),)

    operation_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    feature_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    child_workflow_id: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    repository_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    operation_type: Mapped[ExternalOperationType] = mapped_column(
        SqlEnum(
            ExternalOperationType,
            name="external_operation_type",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
        index=True,
    )
    idempotency_key: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[ExternalOperationStatus] = mapped_column(
        SqlEnum(
            ExternalOperationStatus,
            name="external_operation_status",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
        index=True,
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    repository_revision: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    command_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_current: Mapped[bool] = mapped_column(nullable=False, default=True, index=True)
    superseded_by_operation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    external_reference: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    compensation_status: Mapped[CompensationStatus | None] = mapped_column(
        SqlEnum(
            CompensationStatus,
            name="compensation_status",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=True,
    )
    safe_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    attempts: Mapped[list[ExternalOperationAttemptModel]] = relationship(
        back_populates="operation", cascade="all, delete-orphan"
    )
    events: Mapped[list[ExternalOperationEventModel]] = relationship(
        back_populates="operation", cascade="all, delete-orphan"
    )


class FeatureActionExternalOperationModel(Base):
    """Exact many-to-many evidence linking an action to each journaled effect it used."""

    __tablename__ = "feature_action_external_operations"

    action_id: Mapped[str] = mapped_column(
        ForeignKey("feature_actions.action_id", ondelete="CASCADE"), primary_key=True
    )
    operation_id: Mapped[str] = mapped_column(
        ForeignKey("external_operations.operation_id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    linked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class ExternalOperationAttemptModel(Base):
    """Execution-attempt evidence, including local process and remote provider identifiers."""

    __tablename__ = "external_operation_attempts"

    attempt_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    operation_id: Mapped[str] = mapped_column(
        ForeignKey("external_operations.operation_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    provider: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    workspace_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_plan_artifact_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    contract_artifact_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    process_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    external_run_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    status: Mapped[ExternalOperationStatus] = mapped_column(
        SqlEnum(
            ExternalOperationStatus,
            name="external_operation_attempt_status",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    safe_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    operation: Mapped[ExternalOperationModel] = relationship(back_populates="attempts")


class ExternalOperationEventModel(Base):
    """Append-only state transition history for external-operation recovery and auditing."""

    __tablename__ = "external_operation_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    operation_id: Mapped[str] = mapped_column(
        ForeignKey("external_operations.operation_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    previous_status: Mapped[ExternalOperationStatus | None] = mapped_column(
        SqlEnum(
            ExternalOperationStatus,
            name="external_operation_previous_status",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=True,
    )
    new_status: Mapped[ExternalOperationStatus] = mapped_column(
        SqlEnum(
            ExternalOperationStatus,
            name="external_operation_event_status",
            native_enum=False,
            create_constraint=True,
            values_callable=lambda enum_type: [member.value for member in enum_type],
        ),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    workflow_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    feature_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    child_workflow_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    repository_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    safe_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    operation: Mapped[ExternalOperationModel] = relationship(back_populates="events")
