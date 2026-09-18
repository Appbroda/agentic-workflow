"""Strongly typed state and input models for multi-repository feature workflows."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, HttpUrl, field_validator, model_validator
from typing_extensions import TypedDict

from artifacts.schemas import Artifact
from state.enums import (
    CancellationLifecycleStatus,
    ChildWorkflowStatus,
    DeploymentStrategy,
    FeatureWorkflowStatus,
    MergeStrategy,
    TargetedAttempt,
)
from state.external_operations import ManualCleanupRequirement, WorkflowCheckpointBoundary
from state.failure_diagnosis import (
    FailureStage,
    FeatureFailureClassification,
    fallback_diagnostic,
    is_retryable_classification,
    normalize_classification,
)
from state.models import StateModel
from workflow_schema import WORKFLOW_SCHEMA_VERSION

type NonEmptyString = Annotated[str, Field(min_length=1)]
type RepositoryRole = Annotated[str, Field(min_length=1, max_length=32)]

_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,126}$")


class RepositorySpec(StateModel):
    """A repository selected for one independently reviewed feature workstream."""

    repository_id: NonEmptyString
    name: NonEmptyString
    # Roles are descriptive labels, never repository identity. Keep this open-ended so a
    # future role (mobile, documentation, data-pipeline, and so on) does not require a schema
    # deployment before the repository can participate in the existing generic workstream.
    role: RepositoryRole
    repository_url: HttpUrl
    default_branch: NonEmptyString
    local_workspace_path: str | None = None
    required: bool = True
    implementation_order: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("repository_id")
    @classmethod
    def repository_id_must_be_safe(cls, value: str) -> str:
        """Keep IDs safe for child workflow IDs, paths, and Git branch segments."""
        if not _IDENTIFIER.fullmatch(value):
            msg = "repository_id may contain only letters, numbers, '.', '_', and '-'"
            raise ValueError(msg)
        return value

    @field_validator("repository_url")
    @classmethod
    def repository_url_must_not_embed_credentials(cls, value: HttpUrl) -> HttpUrl:
        """Keep provider credentials in request headers and out of durable feature state."""
        parsed = urlsplit(str(value))
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            msg = "repository_url must not contain credentials, query parameters, or fragments"
            raise ValueError(msg)
        return value

    @field_validator("local_workspace_path")
    @classmethod
    def workspace_path_must_be_absolute(cls, value: str | None) -> str | None:
        """Avoid a process-relative workspace request in durable state."""
        if value is not None and not Path(value).is_absolute():
            msg = "local_workspace_path must be absolute when supplied"
            raise ValueError(msg)
        return value


class ChildWorkflowReference(StateModel):
    """The durable status and artifact references for one repository child workflow."""

    child_workflow_id: NonEmptyString
    repository_id: NonEmptyString
    workstream_id: NonEmptyString
    status: ChildWorkflowStatus
    branch_name: NonEmptyString
    # The branch this child's checkout is created from. `None` means the repository's
    # configured default, which is what every child written before revisions existed did.
    # A revision sets it to the superseded branch so the new attempt starts from the
    # published work instead of regenerating it.
    #
    # Projected explicitly in `storage/feature_store.py` (`_add_state_children` /
    # `_state_from_model`) and as a `feature_child_workflows` column: a child field the
    # projection does not name silently never persists (the `40-` lesson).
    base_branch: str | None = None
    workspace_path: NonEmptyString
    retry_count: int = Field(ge=0)
    code_completion_artifact_id: str | None = None
    review_artifact_id: str | None = None
    pull_request_artifact_id: str | None = None
    blocking_issues: list[str] = Field(default_factory=list)
    checkpoint_boundary: WorkflowCheckpointBoundary | None = None
    technology_profile: dict[str, Any] | None = None
    validation_plan: dict[str, Any] | None = None
    current_revision: str | None = None
    current_validation_results: list[dict[str, Any]] = Field(default_factory=list)
    superseded_validation_count: int = Field(default=0, ge=0)
    scoped_requirements: list[dict[str, Any]] = Field(default_factory=list)
    out_of_scope_requirements: list[str] = Field(default_factory=list)
    preflight_result: dict[str, Any] | None = None
    preflight_status: str | None = None
    blocking_setup_issues: list[dict[str, Any]] = Field(default_factory=list)
    selected_package_manager: str | None = None
    layout_evidence: dict[str, Any] | None = None
    configured_validation_commands: list[dict[str, Any]] = Field(default_factory=list)
    test_availability: str | None = None
    implementation_expectations: list[dict[str, Any]] = Field(default_factory=list)
    production_files_changed: list[str] = Field(default_factory=list)
    test_files_changed: list[str] = Field(default_factory=list)
    configuration_files_changed: list[str] = Field(default_factory=list)
    requirements_implemented: list[str] = Field(default_factory=list)
    requirements_not_implemented: list[str] = Field(default_factory=list)
    failure_classification: str | None = None
    retry_strategy: dict[str, Any] | None = None
    # Which configured model role executes this repository's next attempt, and why. Decided
    # only after the retry policy has already allowed the attempt, so this records a selection
    # and never an allowance. Repository scoped like every other field here: a remediation in
    # one repository cannot change what another repository's next attempt uses.
    model_routing: dict[str, Any] | None = None
    implementation_retry_count: int = Field(default=0, ge=0)
    validation_retry_count: int = Field(default=0, ge=0)
    repository_setup_retry_count: int = Field(default=0, ge=0)
    integration_retry_count: int = Field(default=0, ge=0)
    # Which authority demanded the attempt this child is currently on, stamped where the
    # counters above are stamped. It exists because nothing else can answer it: the true fact
    # is an in-flight parameter the platform never persisted, the previous result's
    # classification fails in both directions -- an approved attempt carries none, and a
    # repository review with a `contract` finding sets `contract_mismatch` too -- and
    # `integration_retry_count` is cumulative, so it cannot say whether attempt K in
    # particular was a remediation.
    #
    # `None` means a workstream retry, which is also what every attempt written before this
    # field existed reads as. Not "unknown": the graph draws it from the lane's own review
    # loop exactly as it always did, which is the compatible answer.
    targeted_attempt_kind: TargetedAttempt | None = None
    # The persisted plan of a context-partition wave (80-): the clusters a refused retry's
    # blocking demands were split into, each `{"diagnostics": [...], "files": [...],
    # "status": "pending"|"done"}`. Non-None exactly while a wave is in flight, so a crashed
    # run resumes the wave through the existing claim machinery instead of re-deciding --
    # a partition must survive the sweep like any other step.
    pending_context_partition: list[dict[str, Any]] | None = None
    # How many scoped attempts beyond the wave's first this workstream has run. The wave's
    # first scoped attempt carries the `retry_count` increment the refused attempt was
    # already granted; later clusters increment only this, so partitioning never spends a
    # second retry.
    context_partition_count: int = Field(default=0, ge=0)
    meaningful_change: bool | None = None
    meaningful_change_reason: str | None = None
    # Why the retry loop stopped, so an escalation states its cause instead of leaving
    # an operator to infer it from counter values.
    retry_refusal_reason: str | None = None
    # Attempts a person explicitly granted this repository after the platform stopped it.
    # The platform never raises its own budget: exhausting it is a finding, and resetting the
    # counter silently would repeat coding and validation side effects on the strength of no
    # new information. A grant is an override, so it is recorded as one -- who, when and how
    # many -- and it applies to this repository alone rather than to the feature's limits.
    granted_extra_attempts: int = Field(default=0, ge=0)
    retry_grants: list[dict[str, Any]] = Field(default_factory=list)
    production_diff_fingerprint: str | None = None
    previous_attempt_fingerprint: str | None = None
    # Tracked separately from production source because a review finding about a test is
    # answered by editing a test. Judging that attempt on production bytes alone reported no
    # progress and ended the workstream, which is how -068's backend finished.
    test_diff_fingerprint: str | None = None
    previous_test_fingerprint: str | None = None
    # The two clocks of this workstream's current run. Wall is everything since its loop
    # started; charged excludes time spent inside external operations that ended in a
    # classified provider fault, including their retries and stall time -- 54 minutes of a
    # terminally failing coding call is the weather, not the work, and the runtime ceiling
    # judges the charged clock. Recorded so the difference is legible afterwards.
    runtime_wall_seconds: float | None = Field(default=None, ge=0)
    runtime_charged_seconds: float | None = Field(default=None, ge=0)
    # Reconnaissance failed for this repository and the feature was planned without its
    # evidence -- the workstream most likely to fail, and the one whose first attempt runs
    # the late reconnaissance probe against the checkout it by then has. The flag never
    # clears: late reconnaissance restores evidence, not the judgment the planner already
    # exercised without it.
    planned_blind: bool = False
    planned_blind_reason: str | None = None

    @field_validator("status", mode="before")
    @classmethod
    def status_must_be_known(cls, value: ChildWorkflowStatus | str) -> ChildWorkflowStatus:
        """Rehydrate enum status safely from persisted JSON."""
        return value if isinstance(value, ChildWorkflowStatus) else ChildWorkflowStatus(value)

    @field_validator("targeted_attempt_kind", mode="before")
    @classmethod
    def targeted_attempt_kind_reads_as_a_workstream_retry_when_absent(
        cls, value: TargetedAttempt | str | None
    ) -> TargetedAttempt | None:
        """Rehydrate the recorded attempt kind, treating anything unrecognised as absent.

        Absent means a workstream retry, which is what every attempt written before this
        field existed is. A value a later deployment wrote and this one has never heard of
        reads the same way rather than refusing to load the whole child: the origin it
        decides is a drawing, and a repository that will not hydrate is an outage.
        """
        if value is None or isinstance(value, TargetedAttempt):
            return value
        try:
            return TargetedAttempt(value)
        except ValueError:
            return None

    @field_validator("checkpoint_boundary", mode="before")
    @classmethod
    def checkpoint_boundary_must_be_known(
        cls, value: WorkflowCheckpointBoundary | str | None
    ) -> WorkflowCheckpointBoundary | None:
        """Rehydrate the durable safe-resume boundary from its database string."""
        if value is None or isinstance(value, WorkflowCheckpointBoundary):
            return value
        return WorkflowCheckpointBoundary(value)


class FeatureFailureSummary(StateModel):
    """Credential-free terminal diagnosis sufficient to decide the next operator action."""

    stage: NonEmptyString
    agent: NonEmptyString
    repository_id: str | None = None
    root_classification: NonEmptyString
    # What the workstream as a whole is attributed to, from ``triage_stopped_workstream``.
    # Distinct from ``root_classification``, which classifies the *last* attempt's blocker:
    # a workstream can end on `implementation_missing` after eleven attempts that each wrote
    # production source, and reporting only the classification sends a reader to look for a
    # missing implementation that was there an attempt ago. `None` for a feature that stopped
    # without reaching a workstream triage at all.
    terminal_cause: str | None = None
    attempt: int | None = Field(default=None, ge=0)
    command: list[str] = Field(default_factory=list)
    exit_code: int | None = None
    diagnostics: list[str] = Field(default_factory=list)
    retryable: bool
    next_action: NonEmptyString
    recorded_at: datetime


class FeatureWorkflowSnapshot(StateModel):
    """Persistable parent state; artifacts contain no request-scoped credentials."""

    feature_id: NonEmptyString
    workflow_id: NonEmptyString
    workflow_schema_version: NonEmptyString
    created_by_build_revision: NonEmptyString
    last_executor_build_revision: NonEmptyString
    status: FeatureWorkflowStatus
    # Why the feature is at the status above, written by `state.feature_transitions` and by
    # nothing else. Distinct from `failure_summary`, which explains a feature that *stopped*:
    # this explains every status, so an operator reading a live feature can see which of the
    # forty-odd paths that write a status is the one that wrote this one. Optional because a
    # snapshot persisted before transitions had an owner carries no reason, and reading an
    # old feature has to keep working.
    transition_reason: str | None = None
    title: NonEmptyString
    # The identity people use, allocated by the database at creation: `AB-Feature-42`. It
    # travels in the snapshot because the parts of the workflow that need it -- pull-request
    # titles most of all -- are handed state rather than a database session. Optional so a
    # snapshot written before references existed still validates.
    reference: str | None = None
    repository_specs: list[RepositorySpec] = Field(min_length=1)
    artifacts: list[Artifact] = Field(default_factory=list)
    child_workflows: dict[str, ChildWorkflowReference] = Field(default_factory=dict)
    clarification_rounds: int = Field(default=0, ge=0)
    max_clarification_rounds: int = Field(default=10, ge=1)
    integration_review_cycles: int = Field(default=0, ge=0)
    max_integration_review_cycles: int = Field(ge=1)
    max_child_review_cycles: int = Field(ge=1)
    max_implementation_retries: int = Field(default=4, ge=0)
    max_validation_retries: int = Field(default=2, ge=0)
    max_repository_setup_retries: int = Field(default=1, ge=0)
    max_contract_revision_cycles: int = Field(ge=1)
    contract_revision_cycles: int = Field(default=0, ge=0)
    # How many times a person has revised this feature after it completed. 0 is the
    # original run, which is what every snapshot written before revisions existed reads
    # as. Artifacts made by a revision run carry the counter in their metadata as
    # `feature_revision`, so the step machinery can tell a revision's plan and completion
    # from the superseded run's without any artifact being deleted.
    revision: int = Field(default=0, ge=0)
    merge_strategy: MergeStrategy | None = None
    deployment_strategy: DeploymentStrategy | None = None
    cancellation_requested: bool = False
    cancellation_status: CancellationLifecycleStatus = CancellationLifecycleStatus.NOT_REQUESTED
    cancellation_requested_at: datetime | None = None
    cancellation_requested_by: str | None = None
    cancellation_reason: str | None = None
    cancellation_current_operation_id: str | None = None
    cancellation_completed_at: datetime | None = None
    cleanup_requirements: list[ManualCleanupRequirement] = Field(default_factory=list)
    checkpoint_boundaries: dict[str, WorkflowCheckpointBoundary] = Field(default_factory=dict)
    # The two clocks of the stages that run before any repository does, accumulated across
    # every pre-coding agent call this feature made. Wall is what the calls took; the fault
    # clock is what classified provider faults and their backoff consumed inside that --
    # AB-Feature-173 spent 37 of its 41 pre-coding minutes inside one ReadTimeout, and until
    # these existed that time was unattributed anywhere. Record-only: the pre-coding runtime
    # ceiling is deliberately deferred until these measurements justify one (41- §6). None on
    # a snapshot written before the fields existed, which is itself the record that nothing
    # was measured.
    planning_wall_seconds: float | None = Field(default=None, ge=0)
    planning_provider_fault_seconds: float | None = Field(default=None, ge=0)
    # The repositories whose reconnaissance fail-soft fired, with the latest sanitized
    # reason. Written the moment planning goes blind -- before any child workstream exists --
    # and copied onto each repository's workstream record when the children are initialized,
    # so the flag survives a crash between reconnaissance and planning. Empty means either
    # nothing failed or nothing was attempted; only the fail-soft writes here.
    planned_blind_repositories: dict[str, str] = Field(default_factory=dict)
    # The last clarification-grounding attempt failed, so the human being asked is the
    # fallback rather than the plan. Reset at the start of every grounding attempt and when
    # answers are applied; read by the clarification endpoint to distinguish "we need you"
    # from "we tried to answer these ourselves, failed, and now need you" -- two states
    # AB-Feature-173's operator could not tell apart because this was a log-only warning.
    clarification_grounding_failed: bool = False
    failure_summary: FeatureFailureSummary | None = None
    current_agent: str | None = None
    execution_mode: Literal["mock", "live"] = "mock"
    # Which model provider this feature's agents run on, chosen at submission and fixed for
    # its life. Nothing re-resolves it from configuration: a recovered attempt that switched
    # SDK mid-workstream would make its own execution and routing records false.
    #
    # `openai` for a snapshot written before the field existed, which is what those features
    # actually ran on -- not a default standing in for the current deployment's choice.
    agent_platform: Literal["openai", "anthropic"] = "openai"
    # How expensively that provider's model roles resolve for this feature, pinned beside the
    # platform at submission and never re-read from configuration. `high` for a snapshot
    # written before the field existed: the unsuffixed variables those features resolved
    # against are the high tier, so this is what they actually ran at -- not a default
    # standing in for the current deployment's choice. `custom` means the feature resolves
    # through `model_setup_snapshot` below rather than through any environment preset.
    performance_tier: Literal["low", "medium", "high", "ultra", "custom"] = "high"
    # The pinned role map of a custom feature -- `{"setup_id", "name", "roles"}` -- copied at
    # creation so editing or deleting the setup never changes what this feature means. The
    # snapshot is the only execution authority; `model_setup_id` beside it is provenance for
    # a person, never resolved through. Both are None for a tier feature.
    #
    # Projected explicitly at `_apply_state` in `storage/feature_store.py`: a state field
    # that projection does not name silently never persists (the `40-` lesson).
    model_setup_snapshot: dict[str, Any] | None = None
    model_setup_id: str | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator("status", mode="before")
    @classmethod
    def status_must_be_known(cls, value: FeatureWorkflowStatus | str) -> FeatureWorkflowStatus:
        """Rehydrate the strict parent enum from database JSON."""
        return value if isinstance(value, FeatureWorkflowStatus) else FeatureWorkflowStatus(value)

    @field_validator("workflow_schema_version")
    @classmethod
    def workflow_schema_version_must_be_compatible(cls, value: str) -> str:
        """Reject an explicitly incompatible durable snapshot before it can be resumed."""
        if value != WORKFLOW_SCHEMA_VERSION:
            msg = (
                f"workflow state schema {value!r} is incompatible with executor schema "
                f"{WORKFLOW_SCHEMA_VERSION!r}"
            )
            raise ValueError(msg)
        return value

    @field_validator("merge_strategy", mode="before")
    @classmethod
    def merge_strategy_must_be_known(
        cls, value: MergeStrategy | str | None
    ) -> MergeStrategy | None:
        """Rehydrate an optional human merge recommendation from persisted JSON."""
        if value is None or isinstance(value, MergeStrategy):
            return value
        return MergeStrategy(value)

    @field_validator("deployment_strategy", mode="before")
    @classmethod
    def deployment_strategy_must_be_known(
        cls, value: DeploymentStrategy | str | None
    ) -> DeploymentStrategy | None:
        """Rehydrate an optional human deployment recommendation from persisted JSON."""
        if value is None or isinstance(value, DeploymentStrategy):
            return value
        return DeploymentStrategy(value)

    @model_validator(mode="after")
    def repository_identifiers_must_be_unique(self) -> FeatureWorkflowSnapshot:
        """Prevent one child from overwriting another repository's durable state."""
        repository_ids = [spec.repository_id for spec in self.repository_specs]
        if len(repository_ids) != len(set(repository_ids)):
            msg = "repository_ids must be unique"
            raise ValueError(msg)
        return self


# How many sentences a terminal record carries. Enough for the triage, the path's own
# explanation and the failing repository's blockers; short enough to sit beside the snapshot.
_MAX_SUMMARY_DIAGNOSTICS = 5

_TERMINAL_STATUSES = frozenset(
    {
        FeatureWorkflowStatus.FAILED,
        FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
    }
)


# What a person is told to do about a failure a resume can clear, per service. Both say the
# same thing about the journal and differ only in which service to check first -- and that is
# the whole point of the split: `retryable` is one property, and "resume it" was being paired
# with "the provider call failed transiently" for a Git outage that never reached a provider.
#
# Both now open by naming the service to go and look at, which the Git line already did and
# the provider line did not. Every route that records one of these has already spent an
# allowance -- the child loop's four faults, or the queue entry's attempts -- so a person who
# reads "resume this feature" and does exactly that is being sent back into the same outage.
_RETRYABLE_NEXT_ACTIONS: dict[FeatureFailureClassification, str] = {
    FeatureFailureClassification.PROVIDER_UNAVAILABLE: (
        "Check that the model provider is answering, then resume this feature; the provider "
        "call failed transiently and completed work is reused from the operation journal."
    ),
    FeatureFailureClassification.GIT_REMOTE_UNAVAILABLE: (
        "Check that the Git remote is reachable, then resume this feature; the Git operation "
        "failed transiently and completed work is reused from the operation journal."
    ),
    FeatureFailureClassification.DESIGN_SOURCE_UNAVAILABLE: (
        "Check that the design source is answering for the cited file, then resume this "
        "feature; the design fetch failed transiently and completed work is reused from the "
        "operation journal."
    ),
}
# Only reachable through a durable row that predates the vocabulary -- one whose recorded name
# is a provider SDK exception the substring rule still reads as transient.
_DEFAULT_RETRYABLE_NEXT_ACTION = _RETRYABLE_NEXT_ACTIONS[
    FeatureFailureClassification.PROVIDER_UNAVAILABLE
]


def _summary_diagnostics(
    *,
    triage_evidence: list[str],
    supplied: list[str],
    child_diagnostics: list[str],
    classification: FeatureFailureClassification,
) -> list[str]:
    """Order a terminal record's sentences, and never return none of them.

    Three sources, most-explanatory first. The triage leads because it is the only line that
    describes the workstream rather than its final attempt, and truncating it away is how a
    feature that spent eleven attempts writing production source came to be reported as
    never having written any. What the terminal path itself supplied comes next: it knows
    why the feature is stopping *now*, which the child's blockers do not. Previously the
    child's blockers displaced it entirely, so an abandoned run's careful explanation of what
    the executor had reached was dropped whenever any repository had also failed.
    """
    ordered: list[str] = []
    for item in [*triage_evidence, *supplied, *child_diagnostics]:
        text = str(item).strip()
        if text and text not in ordered:
            ordered.append(text)
    if not ordered:
        # A caller that supplies nothing and has no failed child is a bug in the caller. In a
        # test that is loud; here it falls back to the sentence the classification is worth,
        # because a record that says nothing is worse than one that says only its category.
        ordered.append(fallback_diagnostic(classification))
    return ordered[:_MAX_SUMMARY_DIAGNOSTICS]


def ensure_feature_failure_summary(
    state: FeatureWorkflowSnapshot,
    *,
    stage: FailureStage | str | None = None,
    classification: FeatureFailureClassification | None = None,
    error_type: str | None = None,
    diagnostics: list[str] | None = None,
) -> FeatureWorkflowSnapshot:
    """Attach one structured terminal diagnosis without exposing arbitrary exception text.

    ``stage`` is where the failure happened, supplied by the path that recorded it. Nothing
    is derived from ``current_agent`` any more: that names whichever agent the orchestrator
    last assigned, so a workstream that failed its own validation was filed under the stage
    that ran after it. A caller that does not know falls back to the failing repository, and
    only then to the runtime.

    ``classification`` is authoritative and overrides the failing child's, for a path that
    stopped the feature for a reason of its own. ``error_type`` is the weaker form: a name to
    fall back to when no child carries a classification. Both are normalised into
    ``FeatureFailureClassification``, so a bare exception type can only ever be recorded as
    ``PLATFORM_DEFECT`` -- never as a statement about the repository.
    """
    if state.status not in _TERMINAL_STATUSES:
        return state
    if state.failure_summary is not None:
        # An existing diagnosis is left exactly as it is, except for the one thing that makes
        # it useless: an empty diagnostics list, which five recorded failures carry. Repaired
        # on the way past rather than asserted on, so reading an old feature still works.
        if state.failure_summary.diagnostics:
            return state
        repaired = state.failure_summary.model_copy(
            update={
                "diagnostics": [
                    fallback_diagnostic(
                        normalize_classification(state.failure_summary.root_classification)
                    )
                ]
            }
        )
        return state.model_copy(update={"failure_summary": repaired})
    failed_children = [
        child
        for child in state.child_workflows.values()
        if child.status
        in {
            ChildWorkflowStatus.FAILED,
            ChildWorkflowStatus.BLOCKED,
            ChildWorkflowStatus.REVIEW_REJECTED,
        }
    ]
    child = (
        sorted(failed_children, key=lambda item: item.repository_id)[0] if failed_children else None
    )
    validation = next(
        (
            item
            for item in (child.current_validation_results if child is not None else [])
            if item.get("status") not in {"passed", "not_configured", "no_tests_found"}
        ),
        {},
    )
    raw_command = validation.get("command")
    command = (
        [str(item) for item in raw_command]
        if isinstance(raw_command, list)
        else ([raw_command] if isinstance(raw_command, str) and raw_command else [])
    )
    raw_exit_code = validation.get("exit_code")
    exit_code = raw_exit_code if isinstance(raw_exit_code, int) else None
    child_diagnostics = list(child.blocking_issues) if child is not None else []
    refusal = child.retry_refusal_reason if child is not None else None
    resolved = classification or normalize_classification(
        child.failure_classification
        if child is not None and child.failure_classification
        else error_type
    )
    retryable = is_retryable_classification(resolved.value)
    triage = _recorded_triage(state, child.repository_id) if child is not None else _NO_TRIAGE
    # Where it happened, in preference to who was assigned when it was written. A caller that
    # states the stage wins; otherwise a failed repository is the stage, because that is where
    # the work stopped even when publication or review ran afterwards.
    resolved_stage = stage or (
        FailureStage.CHILD_WORKFLOWS if child is not None else FailureStage.FEATURE_RUNTIME
    )
    summary = FeatureFailureSummary(
        stage=str(resolved_stage),
        agent=state.current_agent or "feature_runtime",
        repository_id=child.repository_id if child is not None else None,
        root_classification=resolved.value,
        terminal_cause=triage.cause,
        attempt=child.retry_count if child is not None else None,
        command=command,
        exit_code=exit_code,
        diagnostics=_summary_diagnostics(
            triage_evidence=triage.evidence,
            supplied=list(diagnostics or []),
            child_diagnostics=child_diagnostics,
            classification=resolved,
        ),
        retryable=retryable,
        # The question a person now owns, in preference to the reason no further attempt was
        # scheduled. Both are true; only one of them says what to do next.
        next_action=(
            triage.question
            or refusal
            or (
                _RETRYABLE_NEXT_ACTIONS.get(resolved, _DEFAULT_RETRYABLE_NEXT_ACTION)
                if retryable
                else "Inspect the persisted diagnostics and repository evidence before "
                "starting a new run."
            )
        ),
        recorded_at=datetime.now(UTC),
    )
    return state.model_copy(update={"failure_summary": summary})


class _RecordedTriage(StateModel):
    """What a stopped workstream's own triage concluded, where one was recorded."""

    cause: str | None = None
    question: str | None = None
    evidence: list[str] = Field(default_factory=list)


_NO_TRIAGE = _RecordedTriage()


def _recorded_triage(state: FeatureWorkflowSnapshot, repository_id: str) -> _RecordedTriage:
    """Read the terminal triage this repository's newest attempt recorded.

    ``triage_stopped_workstream`` already attributes a stopped workstream to a cause from the
    evidence its attempts produced, and writes that onto the attempt's own result artifact.
    Nothing carried it up to the feature, so the summary an operator reads first was assembled
    from the last attempt's symptoms alone and could contradict the triage underneath it.
    """
    newest: dict[str, Any] | None = None
    for artifact in state.artifacts:
        metadata = getattr(artifact, "metadata", None)
        if (
            getattr(artifact, "artifact_type", "") != "child_workflow_result"
            or getattr(artifact, "repository_id", None) != repository_id
            or not isinstance(metadata, dict)
        ):
            continue
        # Artifacts are appended in attempt order, so the last match is the final attempt.
        newest = metadata
    if newest is None:
        return _NO_TRIAGE
    cause = newest.get("terminal_cause")
    question = newest.get("operator_question")
    evidence = newest.get("terminal_evidence")
    return _RecordedTriage(
        cause=cause if isinstance(cause, str) and cause else None,
        question=question if isinstance(question, str) and question else None,
        evidence=[item for item in evidence if isinstance(item, str)]
        if isinstance(evidence, list)
        else [],
    )


class FeatureWorkflowState(TypedDict):
    """LangGraph-compatible shape retained for integrations that need parent graph state."""

    feature_id: str
    workflow_id: str
    status: FeatureWorkflowStatus
    prd_artifact_id: str
    technical_prd_artifact_id: str | None
    architecture_artifact_id: str | None
    integration_contract_artifact_id: str | None
    repository_plan_artifact_id: str | None
    repository_specs: list[RepositorySpec]
    child_workflows: dict[str, ChildWorkflowReference]
    clarification_rounds: int
    integration_review_cycles: int
    max_integration_review_cycles: int
    merge_strategy: MergeStrategy | None
    deployment_strategy: DeploymentStrategy | None
    cancellation_requested: bool
    created_at: datetime
    updated_at: datetime
