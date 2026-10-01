"""A feature's run, told as a conversation, from the records it already wrote.

During the 185-194 verification cycle three separate "what is actually going on?" questions
were each answered the same way: by opening Postgres, reading `feature_workflow_events`,
`feature_artifacts` and `external_operations` side by side, and translating them into
sentences by hand. This module is that translation, done deterministically.

Five rules run through all of it, and each is a rule because breaking it would make the
logbook worse than nothing:

**No narrator, ever.** There is no model call on this path and no code here that could make
one. Every bubble is a template from the table below, filled with values read off one durable
record. A logbook that can say "I fixed it" when nothing was fixed is worse than no logbook,
and the way to be sure it cannot is to have nothing here capable of composing a novel
sentence.

**One bubble, one record.** Every entry carries a `LogbookRecord` naming the durable row it
was read from -- a lifecycle event, an artifact, a journal operation, or the workstream row.
One record may produce several bubbles (a settled attempt's fault history is four), but a
bubble that cannot name its record does not exist: `feature_logbook` refuses to emit one, and
a test asserts it.

**The agents' own words are the voice.** Where the platform persisted prose -- a review
summary, a retry plan's root cause, a routing reason, a terminal diagnostic, a blocking issue,
a self-review finding -- the bubble quotes it, bounded, with the full text behind the record
link. The tone is model-authored; the sentence around it is not.

**Honest granularity.** Repair passes, assertion-guard refusals, wiring findings and
self-review outcomes are deliberately unjournaled while an attempt runs: one journaled coding
effect per attempt is a property the engineer's gate depends on. They appear when the attempt
settles, read from the completion artifact's metadata. Deferred, never invented, never
omitted.

**Nothing new is disclosed.** Every quoted field is one an existing UI surface already shows,
and the operation projection is exactly the one the workstream-operations endpoint publishes:
no `safe_metadata`, no `error_message`, no raw command output. The logbook is a second reading
of screened records, never a second channel.

The registry is importable and callable on its own -- `LOGBOOK_TEMPLATES` and `render_bubble`
depend on nothing in this package -- because Slack delivery is meant to reuse it verbatim
rather than grow a second implementation of the same sentences.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from artifacts.schemas import (
    ArchitectureArtifact,
    Artifact,
    ChildWorkflowResultArtifact,
    CodeCompletionArtifact,
    ContractChangeRequestArtifact,
    ExecutionGraphArtifact,
    FeatureCompletionArtifact,
    IntegrationContractArtifact,
    IntegrationReviewArtifact,
    PRDArtifact,
    PullRequestArtifact,
    RepositoryExecutionPlanArtifact,
    RepositoryReconnaissanceArtifact,
    RepositoryRepairProposalArtifact,
    ReviewArtifact,
    TaskPlanArtifact,
    TechnicalPRDArtifact,
)
from state.external_operations import (
    CONTRACT_PROJECTION_LOGICAL_STEP,
    ExternalOperation,
    ExternalOperationStatus,
    ExternalOperationType,
    logical_step_of,
)
from state.failure_diagnosis import FeatureFailureClassification, normalize_classification
from state.feature_models import (
    ChildWorkflowReference,
    FeatureFailureSummary,
    FeatureWorkflowSnapshot,
)


class LogbookAgent(StrEnum):
    """Who a bubble is attributed to.

    The chips are the platform's own cast, plus one that is not an agent at all. A
    clarification answer, a granted retry and an approved repository repair are decisions a
    person made, and attributing them to `Platform` would credit this codebase with somebody
    else's judgement -- so they say `Person`. Where a record names the actor, the sentence
    quotes the name; the chip only says which kind of party acted.
    """

    ORCHESTRATOR = "Orchestrator"
    PRODUCT_MANAGER = "Product manager"
    PLANNER = "Planner"
    ENGINEER = "Engineer"
    REVIEWER = "Reviewer"
    INTEGRATION_REVIEWER = "Integration reviewer"
    PUBLISHER = "Publisher"
    PLATFORM = "Platform"
    PERSON = "Person"


class LogbookTone(StrEnum):
    """How a bubble reads at a glance. The same four the timeline already uses."""

    DONE = "done"
    STOPPED = "stopped"
    ATTENTION = "attention"
    WORKING = "working"


class LogbookRecordKind(StrEnum):
    """The kinds of durable record a bubble may be read from.

    The first three are the ones the whole story is normally told from. `WORKSTREAM` exists
    for one thing only: a granted retry is recorded on the durable child row
    (`feature_child_workflows.retry_grants`) and nowhere else -- no event, no artifact, no
    journal row carries it. The choice was to anchor those bubbles to the row that actually
    holds them or to leave a person's override out of the story, and leaving it out was
    worse. It is still a durable record with a detail page of its own.
    """

    EVENT = "event"
    ARTIFACT = "artifact"
    OPERATION = "operation"
    WORKSTREAM = "workstream"


@dataclass(frozen=True, slots=True)
class LogbookTemplate:
    """One row of the registry: a record shape, and the sentence it becomes.

    ``sentence`` and ``detail`` are ``str.format`` templates over fields the caller supplies.
    A missing field raises rather than rendering a half sentence -- the table and its callers
    are meant to be checked against each other by the test suite, not at a reader's expense.

    ``quote_field`` documents which persisted field a bubble of this kind quotes. It is
    published with the entry so a reader (and Slack, later) can say where the words came from
    without this module having to describe it twice.
    """

    key: str
    agent: LogbookAgent
    tone: LogbookTone
    sentence: str
    detail: str | None = None
    quote_field: str | None = None


@dataclass(frozen=True, slots=True)
class LogbookRecord:
    """The durable row one bubble was read from."""

    kind: LogbookRecordKind
    id: str
    repository_id: str | None = None


@dataclass(frozen=True, slots=True)
class LogbookBubble:
    """One rendered bubble, before it is anchored and placed in time.

    ``repository_id`` is set only where the bubble is about a different repository from the
    record it was read from -- the integration review is one artifact whose required fix
    belongs to whichever repository owns it, and badging that bubble with nothing (or with
    every repository) is what made 184's terminal question hard to place.
    """

    template: str
    agent: LogbookAgent
    tone: LogbookTone
    text: str
    detail: str | None
    quote: str | None
    quote_source: str | None
    repository_id: str | None = None


@dataclass(frozen=True, slots=True)
class LogbookEntry:
    """One placed, anchored bubble: the thing a reader sees.

    ``emission`` is the per-record ordinal: which of its record's bubbles this one is. It is
    published because ``(record.kind, record.id, emission)`` is the only identity of a bubble
    that is stable across recompositions -- ``sequence`` is a render-time index that any
    late-arriving record shifts -- and Slack delivery keys its idempotency ledger on it.
    Re-deriving the ordinal downstream would be a second implementation of the composer's own
    ordering, so it travels with the entry instead. Like the template keys, it is a published
    contract, not internal spelling.
    """

    sequence: int
    emission: int
    timestamp: datetime
    agent: LogbookAgent
    tone: LogbookTone
    template: str
    text: str
    detail: str | None
    quote: str | None
    quote_source: str | None
    record: LogbookRecord
    repository_id: str | None = None


@dataclass(frozen=True, slots=True)
class LogbookEvent:
    """One lifecycle event, with the durable id that makes it referenceable.

    Declared here rather than reused from the API layer so the composer can be called (and
    tested) without one. `services` sits below `api`, and the alternative was an import edge
    pointing the wrong way for the sake of four fields.
    """

    id: int
    timestamp: datetime
    event: str
    details: Mapping[str, Any] = field(default_factory=dict)


# How long a quoted fragment may be before it is clipped. A sentence-ish bound: long enough
# for a reviewer's verdict sentence or a root cause, short enough that a bubble stays a
# bubble. The full text is always one link away, which is what makes clipping safe.
QUOTE_CHARACTER_BOUND = 240


def clip_quote(text: str, *, bound: int = QUOTE_CHARACTER_BOUND) -> str:
    """Clip persisted prose at a sentence-ish boundary, never mid-word.

    Prefers the last sentence end inside the bound, so a quote reads as a finished thought.
    Failing that it falls back to the last whole word and marks the clip -- never a bare
    character cut, which is how a path or an identifier becomes a different path or a
    different identifier.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= bound:
        return collapsed
    window = collapsed[:bound]
    for terminator in (". ", "! ", "? ", "; "):
        cut = window.rfind(terminator)
        if cut > bound // 2:
            return window[: cut + 1]
    spaced = window.rsplit(" ", 1)[0] if " " in window else window
    return f"{spaced.rstrip(',;:')}…"


# --------------------------------------------------------------------------------------------
# The registry
#
# Record kind -> template key -> sentence. One table, so item 11's Slack delivery reads the
# same sentences this tab does rather than growing its own. Keys are namespaced by the record
# they are read from and are a published contract: renaming one changes what a Slack message
# says, so they are treated as data and not as internal spelling.
# --------------------------------------------------------------------------------------------

_TEMPLATE_ROWS: tuple[LogbookTemplate, ...] = (
    # -- The request, and the planning that turned it into work ------------------------------
    LogbookTemplate(
        key="artifact.prd",
        agent=LogbookAgent.PERSON,
        tone=LogbookTone.WORKING,
        sentence="Somebody asked for this feature: {title}.",
        quote_field="problem_statement",
    ),
    LogbookTemplate(
        key="artifact.technical_prd",
        agent=LogbookAgent.PRODUCT_MANAGER,
        tone=LogbookTone.WORKING,
        sentence="The product manager worked the request into {requirements}.",
        quote_field="solution_summary",
    ),
    LogbookTemplate(
        key="artifact.technical_prd.questions",
        agent=LogbookAgent.PRODUCT_MANAGER,
        tone=LogbookTone.ATTENTION,
        sentence="{questions} had to be answered before planning could go on.",
        quote_field="unresolved_questions[0].question",
    ),
    LogbookTemplate(
        key="artifact.technical_prd.answers",
        agent=LogbookAgent.PERSON,
        tone=LogbookTone.DONE,
        sentence="A person answered {answers} of those.",
        quote_field="metadata.clarification_answers",
    ),
    LogbookTemplate(
        key="artifact.repository_reconnaissance",
        agent=LogbookAgent.PLANNER,
        tone=LogbookTone.WORKING,
        sentence="The platform read {repository} before assuming anything about it.",
        quote_field="summary",
    ),
    LogbookTemplate(
        key="artifact.repository_reconnaissance.contradiction",
        agent=LogbookAgent.PLANNER,
        tone=LogbookTone.ATTENTION,
        sentence="Reading {repository} contradicted something the request took for granted.",
        detail="{question}",
        quote_field="contradicted_premises[].premise",
    ),
    LogbookTemplate(
        key="artifact.architecture",
        agent=LogbookAgent.PLANNER,
        tone=LogbookTone.WORKING,
        sentence="The shape of the change was settled across {components}.",
        quote_field="system_overview",
    ),
    LogbookTemplate(
        key="artifact.execution_graph",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.WORKING,
        sentence="The platform fixed the order its agents would run in.",
    ),
    LogbookTemplate(
        key="artifact.task_plan",
        agent=LogbookAgent.PLANNER,
        tone=LogbookTone.WORKING,
        sentence="The plan breaks the work into {tasks}.",
        quote_field="summary",
    ),
    LogbookTemplate(
        key="artifact.integration_contract.draft",
        agent=LogbookAgent.PLANNER,
        tone=LogbookTone.WORKING,
        sentence="A shared contract was drafted for the repositories to build against "
        "(version {contract_version}).",
    ),
    LogbookTemplate(
        key="artifact.integration_contract.approved",
        agent=LogbookAgent.PLANNER,
        tone=LogbookTone.DONE,
        sentence="The shared contract the repositories build against was approved "
        "(version {contract_version}).",
    ),
    LogbookTemplate(
        key="artifact.integration_contract.superseded",
        agent=LogbookAgent.PLANNER,
        tone=LogbookTone.WORKING,
        sentence="Version {contract_version} of the shared contract was replaced.",
    ),
    LogbookTemplate(
        key="artifact.repository_execution_plan",
        agent=LogbookAgent.PLANNER,
        tone=LogbookTone.DONE,
        sentence="The work was split across {repositories}, merging {merge_strategy}.",
    ),
    # -- What the engineer wrote, and what its own checks made of it -------------------------
    LogbookTemplate(
        key="artifact.code_completion.completed",
        agent=LogbookAgent.ENGINEER,
        tone=LogbookTone.DONE,
        sentence="The engineer finished writing code for {repository}, changing {files}.",
        quote_field="summary",
    ),
    LogbookTemplate(
        key="artifact.code_completion.partially_completed",
        agent=LogbookAgent.ENGINEER,
        tone=LogbookTone.ATTENTION,
        sentence="The engineer got part of the way through {repository} and stopped.",
        quote_field="summary",
    ),
    LogbookTemplate(
        key="artifact.code_completion.failed",
        agent=LogbookAgent.ENGINEER,
        tone=LogbookTone.STOPPED,
        sentence="What the engineer wrote for {repository} did not pass its own checks.",
        quote_field="summary",
    ),
    LogbookTemplate(
        key="artifact.code_completion.repairs",
        agent=LogbookAgent.ENGINEER,
        tone=LogbookTone.WORKING,
        sentence="Its own checks rejected the first draft, so it corrected the code in "
        "{passes} inside the same attempt, using {model}.",
        quote_field="metadata.source_repair_diagnostics",
    ),
    LogbookTemplate(
        key="artifact.code_completion.repairs_declined",
        agent=LogbookAgent.ENGINEER,
        tone=LogbookTone.ATTENTION,
        sentence="No repair pass was attempted: nothing the checks reported was "
        "mechanically repairable.",
    ),
    LogbookTemplate(
        key="artifact.code_completion.repairs_unavailable",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.ATTENTION,
        sentence="No repair pass was attempted: this deployment has no scoped-fix model "
        "configured for one.",
    ),
    LogbookTemplate(
        key="artifact.code_completion.assertion_guard",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.STOPPED,
        sentence="A repair tried to make the change's own tests pass by weakening what they "
        "assert, and the platform refused it.",
    ),
    LogbookTemplate(
        key="artifact.code_completion.unreachable",
        agent=LogbookAgent.ENGINEER,
        tone=LogbookTone.ATTENTION,
        sentence="The attempt left {issues} unreachable -- nothing that runs refers to them.",
        quote_field="metadata.reachability_issues_unrepaired",
    ),
    LogbookTemplate(
        key="artifact.code_completion.reachability_bounded",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.ATTENTION,
        sentence="The wiring check did not examine {candidates}, so it says nothing about "
        "them either way.",
        quote_field="metadata.reachability_candidates_dropped",
    ),
    LogbookTemplate(
        key="artifact.code_completion.misplaced",
        agent=LogbookAgent.ENGINEER,
        tone=LogbookTone.ATTENTION,
        sentence="The change went into a different file from the one the plan assigned, in "
        "{places}.",
        quote_field="metadata.assigned_file_issues_unrepaired",
    ),
    LogbookTemplate(
        key="artifact.code_completion.self_review",
        agent=LogbookAgent.ENGINEER,
        tone=LogbookTone.WORKING,
        sentence="Before declaring itself done the engineer read its own change back: "
        "{outcome_sentence}",
        quote_field="metadata.self_review.summary",
    ),
    LogbookTemplate(
        key="artifact.code_completion.self_review_unavailable",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.ATTENTION,
        sentence="The engineer's own read-back of its change could not run for this attempt.",
    ),
    # -- What the reviewers said -------------------------------------------------------------
    LogbookTemplate(
        key="artifact.review.approved",
        agent=LogbookAgent.REVIEWER,
        tone=LogbookTone.DONE,
        sentence="The reviewer approved {repository}.",
        quote_field="summary",
    ),
    LogbookTemplate(
        key="artifact.review.changes_requested",
        agent=LogbookAgent.REVIEWER,
        tone=LogbookTone.ATTENTION,
        sentence="The reviewer asked for changes to {repository}, raising {findings}.",
        quote_field="summary",
    ),
    LogbookTemplate(
        key="artifact.review.rejected",
        agent=LogbookAgent.REVIEWER,
        tone=LogbookTone.STOPPED,
        sentence="The reviewer rejected {repository}, raising {findings}.",
        quote_field="summary",
    ),
    LogbookTemplate(
        key="artifact.review.finding",
        agent=LogbookAgent.REVIEWER,
        tone=LogbookTone.ATTENTION,
        sentence="Its most serious finding was {severity}, about {location}.",
        quote_field="findings[].description",
    ),
    LogbookTemplate(
        key="artifact.integration_review.approved",
        agent=LogbookAgent.INTEGRATION_REVIEWER,
        tone=LogbookTone.DONE,
        sentence="The integration reviewer checked the repositories against each other "
        "and approved them.",
        quote_field="compatibility_assessment",
    ),
    LogbookTemplate(
        key="artifact.integration_review.changes_requested",
        agent=LogbookAgent.INTEGRATION_REVIEWER,
        tone=LogbookTone.ATTENTION,
        sentence="The integration reviewer would not approve the repositories together, and "
        "asked for {fixes}.",
        quote_field="compatibility_assessment",
    ),
    LogbookTemplate(
        key="artifact.integration_review.failed",
        agent=LogbookAgent.INTEGRATION_REVIEWER,
        tone=LogbookTone.STOPPED,
        sentence="The integration review did not complete.",
        quote_field="compatibility_assessment",
    ),
    LogbookTemplate(
        key="artifact.integration_review.failed_requires_human",
        agent=LogbookAgent.INTEGRATION_REVIEWER,
        tone=LogbookTone.STOPPED,
        sentence="The integration review stopped and needs a person to decide.",
        quote_field="compatibility_assessment",
    ),
    LogbookTemplate(
        key="artifact.integration_review.required_fix",
        agent=LogbookAgent.INTEGRATION_REVIEWER,
        tone=LogbookTone.ATTENTION,
        sentence="The fix it required of {repository} was this.",
        quote_field="required_fixes[]",
    ),
    # -- How one repository's attempt ended --------------------------------------------------
    LogbookTemplate(
        key="artifact.child_workflow_result.approved",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.DONE,
        sentence="{repository} finished: reviewed, approved and ready to publish.",
    ),
    LogbookTemplate(
        key="artifact.child_workflow_result.failed",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.STOPPED,
        sentence="{repository} stopped without a publishable result.",
        quote_field="blocking_issues[0]",
    ),
    LogbookTemplate(
        key="artifact.child_workflow_result.waiting_for_contract_change",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.ATTENTION,
        sentence="{repository} stopped to ask for a change to the shared contract.",
        quote_field="blocking_issues[0]",
    ),
    LogbookTemplate(
        key="artifact.child_workflow_result.cancelled",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.STOPPED,
        sentence="{repository} was stopped because the feature was cancelled.",
    ),
    LogbookTemplate(
        # The terminal triage question -- including the 49- ledger and design-conflict
        # narratives, which arrive through the same field. It is the sentence a stopped
        # workstream exists to produce, and quoting it is the whole point of the tab.
        key="artifact.child_workflow_result.operator_question",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.ATTENTION,
        sentence="This is the decision {repository} leaves to a person.",
        quote_field="metadata.operator_question",
    ),
    LogbookTemplate(
        key="artifact.child_workflow_result.truncated",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.ATTENTION,
        sentence="The model's answer for {repository} was cut off by its own output limit.",
    ),
    LogbookTemplate(
        key="artifact.child_workflow_result.degraded_retry",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.WORKING,
        sentence="So the platform asked again at a lower reasoning effort, once.",
        quote_field="model_routing.routing_reason",
    ),
    LogbookTemplate(
        key="artifact.child_workflow_result.provider_fault",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.ATTENTION,
        sentence="The provider failed to answer {faults} and each attempt was given back; "
        "{excluded_seconds}s of waiting is not counted against this repository's time.",
        detail="{fault_classes}",
    ),
    LogbookTemplate(
        key="artifact.child_workflow_result.runtime",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.WORKING,
        sentence="{repository} ran for {wall_seconds}s, of which {charged_seconds}s counted "
        "as work.",
    ),
    LogbookTemplate(
        key="artifact.child_workflow_result.retry_strategy",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.WORKING,
        sentence="The next attempt for {repository} was told what to change.",
        quote_field="retry_strategy.root_cause",
    ),
    LogbookTemplate(
        key="artifact.child_workflow_result.routing",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.WORKING,
        sentence="{repository}'s attempt ran on {model}.",
        quote_field="model_routing.routing_reason",
    ),
    LogbookTemplate(
        key="artifact.child_workflow_result.retry_refused",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.STOPPED,
        sentence="No further attempt was scheduled for {repository}.",
        quote_field="metadata.retry_refusal_reason",
    ),
    LogbookTemplate(
        key="artifact.child_workflow_result.advisory_verdict",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.ATTENTION,
        sentence="The reviewer said {advisory_verdict} for {repository}, but every finding "
        "it raised was outside what this repository had been asked for, so the work was "
        "published for a person to read.",
    ),
    # -- Contract deviations, repairs, publication and the end of the run --------------------
    LogbookTemplate(
        key="artifact.contract_change_request",
        agent=LogbookAgent.ENGINEER,
        tone=LogbookTone.ATTENTION,
        sentence="{repository} asked for {changes} to the shared contract rather than "
        "working around it.",
        quote_field="reason",
    ),
    LogbookTemplate(
        key="artifact.repository_repair_proposal",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.ATTENTION,
        sentence="{repository} cannot run its own checks on an untouched checkout, so the "
        "platform proposed a fix for a person to approve.",
        detail="{proposed_repair}",
        quote_field="detected_problem",
    ),
    LogbookTemplate(
        key="artifact.pull_request",
        agent=LogbookAgent.PUBLISHER,
        tone=LogbookTone.DONE,
        sentence="Pull request #{number} was opened against {repository}: {title}.",
    ),
    LogbookTemplate(
        key="artifact.feature_completion.completed",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.DONE,
        sentence="The feature finished with {pull_requests} to merge, {merge_order}.",
    ),
    LogbookTemplate(
        key="artifact.feature_completion.partial_failure",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.ATTENTION,
        sentence="Part of this feature was delivered: {pull_requests} opened, and the rest "
        "of the work was not finished.",
    ),
    LogbookTemplate(
        key="artifact.feature_completion.failed",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.STOPPED,
        sentence="The feature was closed out as failed, with {pull_requests} opened.",
    ),
    LogbookTemplate(
        key="artifact.feature_completion.cancelled",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.STOPPED,
        sentence="The feature was cancelled, with {pull_requests} already opened.",
    ),
    LogbookTemplate(
        key="artifact.unknown",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.WORKING,
        sentence="The platform recorded a {artifact_type} record.",
    ),
    # -- Lifecycle events --------------------------------------------------------------------
    LogbookTemplate(
        key="event.feature_started",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.WORKING,
        sentence="Work on this feature started.",
    ),
    LogbookTemplate(
        key="event.feature_queued",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.WORKING,
        sentence="The feature was accepted and queued to run.",
    ),
    LogbookTemplate(
        key="event.feature_credentials_missing",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.ATTENTION,
        sentence="The feature could not start: the model provider credentials it needs are "
        "not configured.",
        quote_field="details.reason",
    ),
    LogbookTemplate(
        key="event.repository_planned_blind",
        agent=LogbookAgent.PLANNER,
        tone=LogbookTone.ATTENTION,
        sentence="{repository} could not be read, so the plan for it was written without "
        "seeing the code.",
        quote_field="details.reason",
    ),
    LogbookTemplate(
        key="event.repository_repair_completed",
        agent=LogbookAgent.PERSON,
        tone=LogbookTone.DONE,
        sentence="{actor} approved the repository fix, and the platform applied it.",
    ),
    LogbookTemplate(
        key="event.repository_repair_rejected",
        agent=LogbookAgent.PERSON,
        tone=LogbookTone.STOPPED,
        sentence="{actor} declined the repository fix the platform proposed.",
    ),
    LogbookTemplate(
        key="event.repository_repair_superseded",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.ATTENTION,
        sentence="The repository moved on since the fix was diagnosed, so the proposal was "
        "withdrawn rather than applied.",
    ),
    LogbookTemplate(
        key="event.feature_run_continued",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.WORKING,
        sentence="A previous run of this feature stopped without finishing, so the platform "
        "picked it up where it left off.",
    ),
    LogbookTemplate(
        key="event.feature_run_abandoned",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.STOPPED,
        sentence="The worker holding this feature stopped without recording an outcome.",
    ),
    LogbookTemplate(
        key="event.feature_resume_refused",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.ATTENTION,
        sentence="The platform declined to resume this feature.",
        quote_field="details.reason",
    ),
    LogbookTemplate(
        key="event.feature_resume_found_no_eligible_workstreams",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.ATTENTION,
        sentence="A resume was asked for, but no repository had work left that the platform "
        "may do unasked.",
        quote_field="details.reason",
    ),
    LogbookTemplate(
        key="event.feature_runtime_limit_reached",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.STOPPED,
        sentence="The feature ran for {elapsed_minutes} minutes, longer than the "
        "{limit_minutes} this deployment allows, so it was stopped.",
    ),
    LogbookTemplate(
        key="event.feature_step_budget_exhausted",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.STOPPED,
        sentence="The feature took {steps} steps without settling, so the platform stopped "
        "scheduling more.",
    ),
    LogbookTemplate(
        key="event.unconfirmed_external_effect",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.STOPPED,
        sentence="A {operation_type} against the provider was interrupted and its effect "
        "could not be confirmed, so a person has to check the repository.",
    ),
    LogbookTemplate(
        key="event.feature_cancellation_requested",
        agent=LogbookAgent.PERSON,
        tone=LogbookTone.ATTENTION,
        sentence="Somebody asked for this feature to be cancelled.",
    ),
    LogbookTemplate(
        key="event.feature_cancellation_updated",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.STOPPED,
        sentence="The cancellation finished working through what was already running.",
    ),
    LogbookTemplate(
        key="event.feature_retired_by_operator",
        agent=LogbookAgent.PERSON,
        tone=LogbookTone.STOPPED,
        sentence="{actor} retired this feature.",
        quote_field="details.reason",
    ),
    LogbookTemplate(
        key="event.feature_waiting_for_human",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.ATTENTION,
        sentence="The feature stopped and is waiting for a person.",
        quote_field="details.reason",
    ),
    LogbookTemplate(
        key="event.feature_publication_held",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.ATTENTION,
        sentence=(
            "This feature did not land, so nothing was published. Publishing it is a "
            "person's decision now."
        ),
        quote_field="details.reason",
    ),
    LogbookTemplate(
        key="event.feature_published_by_person",
        agent=LogbookAgent.PERSON,
        tone=LogbookTone.ATTENTION,
        sentence="{actor} published this feature's finished work.",
        quote_field="details.reason",
    ),
    LogbookTemplate(
        key="event.feature_completed",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.DONE,
        sentence="The feature finished.",
    ),
    LogbookTemplate(
        key="event.feature_cancelled",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.STOPPED,
        sentence="The feature was cancelled.",
    ),
    LogbookTemplate(
        key="event.feature_failed",
        agent=LogbookAgent.ORCHESTRATOR,
        tone=LogbookTone.STOPPED,
        sentence="The feature stopped: {classification_sentence}",
        quote_field="details.diagnostics[0]",
    ),
    LogbookTemplate(
        key="event.feature_failed.next_action",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.ATTENTION,
        sentence="What the platform says to do about it:",
        quote_field="failure_summary.next_action",
    ),
    LogbookTemplate(
        key="event.unknown",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.WORKING,
        sentence="The platform recorded {event}.",
    ),
    # -- Journal operations ------------------------------------------------------------------
    #
    # One template per outcome rather than per operation type: the type supplies the gerund
    # (`OPERATION_VOICES` below) and the outcome supplies the sentence around it. A type the
    # journal adds later still renders, through `operation.unknown_type`.
    LogbookTemplate(
        key="operation.succeeded",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.DONE,
        sentence="Finished {doing}.",
        detail="{attempt_note}",
    ),
    LogbookTemplate(
        key="operation.running",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.WORKING,
        sentence="Started {doing}.",
        detail="{attempt_note}",
    ),
    LogbookTemplate(
        key="operation.failed",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.STOPPED,
        sentence="Failed while {doing}.",
        detail="{error_note}{attempt_note}",
    ),
    LogbookTemplate(
        key="operation.cancelled",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.STOPPED,
        sentence="Stopped while {doing}, because this feature was being cancelled.",
        detail="{attempt_note}",
    ),
    LogbookTemplate(
        key="operation.unresolved",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.ATTENTION,
        sentence="Left off {doing} without an answer a person has not yet confirmed.",
        detail="{error_note}{attempt_note}",
    ),
    LogbookTemplate(
        key="operation.recorded",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.WORKING,
        sentence="Recorded that it was about to start {doing}.",
        detail="{attempt_note}",
    ),
    LogbookTemplate(
        key="operation.unknown_type",
        agent=LogbookAgent.PLATFORM,
        tone=LogbookTone.WORKING,
        sentence="The platform recorded a {operation_type} operation, which ended {status}.",
        detail="{attempt_note}",
    ),
    # -- The durable workstream row ----------------------------------------------------------
    LogbookTemplate(
        key="workstream.retry_granted",
        agent=LogbookAgent.PERSON,
        tone=LogbookTone.ATTENTION,
        sentence="{actor} granted {repository} {attempts} after the platform had stopped it.",
        detail="{stopped_because}",
        quote_field="retry_grants[].reason",
    ),
)

LOGBOOK_TEMPLATES: Mapping[str, LogbookTemplate] = {row.key: row for row in _TEMPLATE_ROWS}


@dataclass(frozen=True, slots=True)
class OperationVoice:
    """What one journaled operation type is, said the way a person would say it.

    The agent owns the operation and the gerund goes inside the outcome sentence: "Finished
    cloning the repository", "Failed while running the repository's tests". One form per type,
    so a new operation type is one row here rather than six sentences.
    """

    agent: LogbookAgent
    doing: str


OPERATION_VOICES: Mapping[ExternalOperationType, OperationVoice] = {
    ExternalOperationType.CLONE_REPOSITORY: OperationVoice(
        LogbookAgent.ENGINEER, "taking a fresh copy of {repository}"
    ),
    ExternalOperationType.CREATE_BRANCH: OperationVoice(
        LogbookAgent.ENGINEER, "opening a branch to work on"
    ),
    ExternalOperationType.INSTALL_DEPENDENCIES: OperationVoice(
        LogbookAgent.ENGINEER, "installing what {repository} needs to run its checks"
    ),
    ExternalOperationType.RUN_CODING_EXECUTOR: OperationVoice(
        LogbookAgent.ENGINEER, "asking the model to write the code for {repository}"
    ),
    ExternalOperationType.WRITE_FILE_CHANGES: OperationVoice(
        LogbookAgent.ENGINEER, "writing the change into the files"
    ),
    ExternalOperationType.RUN_FORMATTER: OperationVoice(
        LogbookAgent.ENGINEER, "running {repository}'s own code formatter"
    ),
    ExternalOperationType.RUN_LINTER: OperationVoice(
        LogbookAgent.ENGINEER, "running {repository}'s own lint checks"
    ),
    ExternalOperationType.RUN_TYPECHECK: OperationVoice(
        LogbookAgent.ENGINEER, "running {repository}'s own type checks"
    ),
    ExternalOperationType.RUN_TESTS: OperationVoice(
        LogbookAgent.ENGINEER, "running {repository}'s own tests"
    ),
    ExternalOperationType.RUN_BUILD: OperationVoice(LogbookAgent.ENGINEER, "building {repository}"),
    ExternalOperationType.RUN_REVIEWER: OperationVoice(
        LogbookAgent.REVIEWER, "reading the change to {repository} as a reviewer"
    ),
    ExternalOperationType.CREATE_COMMIT: OperationVoice(
        LogbookAgent.PUBLISHER, "committing the change to {repository}"
    ),
    ExternalOperationType.PUSH_BRANCH: OperationVoice(
        LogbookAgent.PUBLISHER, "pushing the branch to {repository}"
    ),
    ExternalOperationType.CREATE_PULL_REQUEST: OperationVoice(
        LogbookAgent.PUBLISHER, "opening a pull request against {repository}"
    ),
    ExternalOperationType.UPDATE_PULL_REQUEST: OperationVoice(
        LogbookAgent.PUBLISHER, "updating the pull request for {repository}"
    ),
    ExternalOperationType.CLOSE_PULL_REQUEST: OperationVoice(
        LogbookAgent.PUBLISHER, "closing the superseded pull request on {repository}"
    ),
    ExternalOperationType.ADD_LABELS: OperationVoice(
        LogbookAgent.PUBLISHER, "labelling the pull request for {repository}"
    ),
    ExternalOperationType.ADD_REVIEWERS: OperationVoice(
        LogbookAgent.PUBLISHER, "asking for reviewers on the pull request for {repository}"
    ),
    ExternalOperationType.RUN_PRODUCT_MANAGER: OperationVoice(
        LogbookAgent.PRODUCT_MANAGER, "reading the request and writing it up as requirements"
    ),
    ExternalOperationType.RUN_REPOSITORY_RECON: OperationVoice(
        LogbookAgent.PLANNER, "reading {repository} to see what it already contains"
    ),
    ExternalOperationType.RUN_CLARIFICATION_GROUNDING: OperationVoice(
        LogbookAgent.PLANNER, "trying to answer the open questions from the code itself"
    ),
    ExternalOperationType.RUN_FEATURE_PLANNER: OperationVoice(
        LogbookAgent.PLANNER, "planning how the repositories divide the work"
    ),
    # Attributed to the orchestrator rather than to any agent: no model reads a design at this
    # point. This is the platform fetching a document the author attached, before the product
    # manager is asked to read the request at all.
    ExternalOperationType.FETCH_DESIGN_REFERENCE: OperationVoice(
        LogbookAgent.ORCHESTRATOR, "reading the designs attached to this request"
    ),
    ExternalOperationType.RUN_INTEGRATION_REVIEW: OperationVoice(
        LogbookAgent.INTEGRATION_REVIEWER,
        "comparing the repositories' changes against the shared contract",
    ),
}


# The voices that belong to a step rather than to a type, keyed by both.
#
# One operation type, two callers: `write_file_changes` is journaled by the coding call for the
# change it wrote, and by `feature_runtime.py::_write_contract_projection` for the approved
# contract's projection -- written into the checkout at the start of every attempt, before the
# Engineer is called at all. The type's own voice says "writing the change into the files",
# which is the implementation's sentence and, on the projection row, a false claim about what
# the platform had done. Found while fixing the same false claim on the drawer (item 21); this
# is the logbook's copy of it.
#
# Keyed on the journal's `logical_step`, which the executor stamps on every row, so the
# distinction comes from the record rather than from a guess about row order. A step this table
# does not name falls through to the type's voice, which is the right default: the
# implementation write is the ordinary case and every other type has exactly one caller.
OPERATION_STEP_VOICES: Mapping[tuple[ExternalOperationType, str], OperationVoice] = {
    (ExternalOperationType.WRITE_FILE_CHANGES, CONTRACT_PROJECTION_LOGICAL_STEP): OperationVoice(
        LogbookAgent.ENGINEER,
        "preparing {repository}'s workspace with the approved contract",
    ),
}


def operation_voice(operation: ExternalOperation) -> OperationVoice | None:
    """The voice for one row: its step's, where the step has one, otherwise its type's."""
    step = logical_step_of(operation)
    if step is not None:
        stepped = OPERATION_STEP_VOICES.get((operation.operation_type, step))
        if stepped is not None:
            return stepped
    return OPERATION_VOICES.get(operation.operation_type)


class UnknownLogbookTemplate(KeyError):
    """A caller asked for a template the registry does not define."""


def render_bubble(
    key: str,
    fields: Mapping[str, Any] | None = None,
    *,
    agent: LogbookAgent | None = None,
    quote: str | None = None,
    repository_id: str | None = None,
) -> LogbookBubble:
    """Render one bubble from the registry. The only place a sentence is composed.

    ``agent`` overrides the template's own chip, and exactly one family of templates needs
    that: a journal operation's outcome template is shared across every operation type, and
    the agent belongs to the type (`OPERATION_VOICES`) rather than to the outcome.

    A quote is clipped here rather than by the caller, so no path can publish an unbounded
    fragment by forgetting to.
    """
    template = LOGBOOK_TEMPLATES.get(key)
    if template is None:
        raise UnknownLogbookTemplate(key)
    values = dict(fields or {})
    detail = template.detail.format(**values).strip() if template.detail else None
    return LogbookBubble(
        template=template.key,
        agent=agent or template.agent,
        tone=template.tone,
        text=template.sentence.format(**values),
        detail=detail or None,
        quote=clip_quote(quote) if quote and quote.strip() else None,
        quote_source=template.quote_field if quote and quote.strip() else None,
        repository_id=repository_id,
    )


# --------------------------------------------------------------------------------------------
# The composer
# --------------------------------------------------------------------------------------------

# Lifecycle events that are bookkeeping rather than story. Checkpoints say a durable boundary
# was written -- a fact about persistence, several times per attempt -- and `feature_executed`
# is the generic name every non-terminal state write carries. Both are excluded by name, and
# that is different from dropping something unrecognised: an event this table has never heard
# of renders through `event.unknown` with its own name, precisely so the newest thing in a run
# is never the invisible one.
_BOOKKEEPING_EVENTS = frozenset({"feature_executed", "feature_artifact_rejected"})
_BOOKKEEPING_EVENT_PREFIX = "feature_checkpoint_"

# Events whose whole content is "this artifact now exists". The artifact itself is a richer
# bubble carrying the agent's own prose, so rendering both would say the same thing twice --
# suppressed only when the artifact they name is actually in the thread, so an event about
# something this read cannot see still renders.
# A sentinel in the "artifacts this thread renders" set, standing for "a completion artifact
# is in the thread". Not an artifact id: no artifact is called this, so nothing else can
# match it.
_COMPLETION_EVENT_COVERED = "\x00feature_completion"

_ARTIFACT_ANNOUNCEMENT_EVENTS = frozenset(
    {
        "contract_created",
        "contract_approved",
        "child_workflow_started",
        "child_workflow_completed",
        "child_workflow_failed",
        "integration_review_started",
        "integration_review_completed",
        "repository_fix_requested",
        "contract_change_requested",
        "pull_request_created",
        "feature_completed",
    }
)

# Which terminal event a failure classification is worth a sentence of its own for. Read from
# the durable summary rather than reworded: `fallback_diagnostic` already owns the wording a
# classification is worth, and this bubble quotes the event's own diagnostic beside it.
_STOP_SENTENCES: Mapping[str, str] = {
    "platform_defect": "the platform itself failed while running it, rather than anything "
    "being wrong with the repository or the request.",
    "platform_capacity_failure": "the worker could not provide the workspace it needed.",
    "provider_unavailable": "the model provider did not answer.",
    "git_remote_unavailable": "the Git remote did not answer, or the platform could not reach it.",
    "design_source_unavailable": "the design source did not answer, so the designs it cites "
    "could not be read.",
    "provider_credentials_missing": "the model provider credentials it needs are not "
    "configured for the account that asked.",
    "publication_failure": "publishing the pull requests did not finish.",
    "pull_request_unverified": "a pull request it reported opening could not be read back.",
    "unconfirmed_external_effect": "an operation against the provider was interrupted and "
    "nobody can yet say what it did.",
    "clarification_unresolved": "its open questions were never resolved.",
    "integration_review_unsatisfied": "the integration review never approved it.",
    "workstream_unfinished": "a repository it needed produced nothing publishable.",
    "executor_stopped": "the worker holding it stopped.",
    "feature_runtime_limit_reached": "it ran for longer than this deployment allows.",
    "repository_runtime_limit_reached": "one repository ran for longer than this deployment "
    "allows.",
}
# What a stop is worth when nothing recognisable was recorded about it. Deliberately not a
# claim: an unmapped classification is a gap in this table, not evidence that the platform
# failed, and saying the latter would send somebody to debug this codebase over a value
# nobody has taught the logbook yet.
_UNRECORDED_STOP = "the platform recorded it as stopped and kept its evidence."


def _stop_sentence(classification: str | None) -> str:
    """Say what a recorded classification is worth, reading a provider subtype as one.

    Two vocabularies reach this field. A terminal event whose classification came off the
    feature-level enum is looked up directly. A terminal event whose classification came off
    an adapter is a provider SDK exception class -- `_root_failure_classification` retains
    those verbatim, and retaining them is the point, because the subtype is the evidence -- so
    `APITimeoutError` used to fall through to the sentence that says nothing. The Git half of
    the same stop reads correctly, which made the pair asymmetric: an operator whose GitHub
    outage exhausted the queue got a sentence naming Git, and the same operator whose provider
    timed out got "kept its evidence".

    `normalize_classification` is the platform's one reading of those names, and it is
    consulted only where it recognises one: everything it cannot read lands on
    `PLATFORM_DEFECT`, which as a fallback would be an accusation rather than a gap.
    """
    if not classification:
        return _UNRECORDED_STOP
    sentence = _STOP_SENTENCES.get(classification)
    if sentence is not None:
        return sentence
    normalized = normalize_classification(classification)
    if normalized is FeatureFailureClassification.PLATFORM_DEFECT:
        return _UNRECORDED_STOP
    return _STOP_SENTENCES.get(normalized.value, _UNRECORDED_STOP)


_SELF_REVIEW_SENTENCES: Mapping[str, str] = {
    "clean": "it found nothing to correct.",
    "corrected": "it found something, corrected it, and the checks accepted the correction.",
    "substantive_problem": "it found a problem too large to correct inside the attempt, so "
    "the attempt ended rather than claiming to be done.",
    "corrections_failed": "it found something and the correction did not happen, so the "
    "attempt ended with the finding unresolved.",
    "correction_rejected": "it found something, corrected it, and the re-run checks rejected "
    "the correction.",
}

_OPERATION_OUTCOME_TEMPLATES: Mapping[ExternalOperationStatus, str] = {
    ExternalOperationStatus.SUCCEEDED: "operation.succeeded",
    ExternalOperationStatus.RUNNING: "operation.running",
    ExternalOperationStatus.STARTING: "operation.running",
    ExternalOperationStatus.FAILED_RETRYABLE: "operation.failed",
    ExternalOperationStatus.FAILED_TERMINAL: "operation.failed",
    ExternalOperationStatus.CANCELLED: "operation.cancelled",
    ExternalOperationStatus.CANCELLATION_REQUESTED: "operation.cancelled",
    ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE: "operation.unresolved",
    ExternalOperationStatus.AWAITING_RECONCILIATION: "operation.unresolved",
    ExternalOperationStatus.PENDING: "operation.recorded",
}

# Where a record kind sits when two records carry the same instant. Events are written by the
# transaction that made the thing true, so they lead; the artifact is the thing itself; the
# journal row is the effect underneath it. Only a tiebreak -- the timestamps decide first.
_KIND_ORDER: Mapping[LogbookRecordKind, int] = {
    LogbookRecordKind.EVENT: 0,
    LogbookRecordKind.ARTIFACT: 1,
    LogbookRecordKind.OPERATION: 2,
    LogbookRecordKind.WORKSTREAM: 3,
}


@dataclass(frozen=True, slots=True)
class _Draft:
    """One bubble with everything needed to place it, before sequencing."""

    timestamp: datetime
    kind: LogbookRecordKind
    record_id: str
    emission: int
    bubble: LogbookBubble
    repository_id: str | None


def feature_logbook(
    state: FeatureWorkflowSnapshot,
    *,
    lifecycle_events: Sequence[LogbookEvent] = (),
    operations: Sequence[ExternalOperation] = (),
) -> list[LogbookEntry]:
    """Read one feature's whole run as a chronological thread of attributed bubbles.

    Pure: given the same durable records it returns the same thread, in the same order, with
    the same sentences. It reads no configuration, opens no session and makes no call of any
    kind -- which is what makes "this cannot hallucinate" a property of the code rather than
    a promise about it.

    Backfillable by construction. Nothing here reads live state: a feature that finished
    months ago renders its whole story from the rows it left behind.
    """
    names = {item.repository_id: item.name for item in state.repository_specs}
    artifacts = list(state.artifacts)
    known_artifact_ids = {item.artifact_id for item in artifacts}
    # `feature_completed` is written twice for a feature that finishes: once from the
    # completion artifact, once from the status transition -- and only the first carries an
    # artifact id. The completion artifact's own bubble says strictly more than either, so
    # the event is covered whichever of the two writes reached this read.
    if any(isinstance(item, FeatureCompletionArtifact) for item in artifacts):
        known_artifact_ids.add(_COMPLETION_EVENT_COVERED)
    owners = _repository_index(state)

    # `failure_summary` describes the stop this feature is currently at, and a feature that
    # was resumed and failed again has more than one terminal event. Only the last of them may
    # borrow that summary's `next_action`: pairing today's advice with a stop from two days ago
    # would tell a reader to act on a state the run has since left.
    latest_stop = next(
        (item.id for item in reversed(lifecycle_events) if item.event == "feature_failed"), None
    )

    drafts: list[_Draft] = []
    for event in lifecycle_events:
        drafts.extend(
            _event_drafts(
                event,
                state=state,
                names=names,
                artifacts=known_artifact_ids,
                latest_stop=latest_stop,
            )
        )
    for artifact in artifacts:
        drafts.extend(_artifact_drafts(artifact, names=names, owners=owners))
    fallback = state.created_at
    for operation in operations:
        stamp = operation.started_at or operation.completed_at or operation.heartbeat_at
        fallback = stamp or fallback
        drafts.extend(_operation_drafts(operation, timestamp=fallback, names=names))
    for child in state.child_workflows.values():
        drafts.extend(_workstream_drafts(child, names=names))

    drafts.sort(
        key=lambda item: (item.timestamp, _KIND_ORDER[item.kind], item.record_id, item.emission)
    )
    return [
        LogbookEntry(
            sequence=index,
            emission=draft.emission,
            timestamp=draft.timestamp,
            agent=draft.bubble.agent,
            tone=draft.bubble.tone,
            template=draft.bubble.template,
            text=draft.bubble.text,
            detail=draft.bubble.detail,
            quote=draft.bubble.quote,
            quote_source=draft.bubble.quote_source,
            record=LogbookRecord(
                kind=draft.kind, id=draft.record_id, repository_id=draft.repository_id
            ),
            repository_id=draft.repository_id,
        )
        for index, draft in enumerate(drafts)
    ]


def _event_drafts(
    event: LogbookEvent,
    *,
    state: FeatureWorkflowSnapshot,
    names: Mapping[str, str],
    artifacts: frozenset[str] | set[str],
    latest_stop: int | None = None,
) -> list[_Draft]:
    """Turn one lifecycle event into its bubbles, or into none where it is bookkeeping."""
    if event.event in _BOOKKEEPING_EVENTS or event.event.startswith(_BOOKKEEPING_EVENT_PREFIX):
        return []
    details = dict(event.details)
    artifact_id = _text(details.get("artifact_id"))
    covered = artifact_id in artifacts or (
        event.event == "feature_completed" and _COMPLETION_EVENT_COVERED in artifacts
    )
    if event.event in _ARTIFACT_ANNOUNCEMENT_EVENTS and covered:
        return []
    repository_id = _text(details.get("repository_id"))
    record_id = str(event.id)
    key = f"event.{event.event}"

    def draft(bubble: LogbookBubble, emission: int = 0) -> _Draft:
        return _Draft(
            timestamp=event.timestamp,
            kind=LogbookRecordKind.EVENT,
            record_id=record_id,
            emission=emission,
            bubble=bubble,
            repository_id=repository_id,
        )

    if key == "event.feature_failed":
        return _stop_drafts(event, state=state, draft=draft, is_latest_stop=event.id == latest_stop)
    if key not in LOGBOOK_TEMPLATES:
        return [draft(render_bubble("event.unknown", {"event": event.event}))]
    fields: dict[str, Any] = {
        "repository": _repository_name(repository_id, names),
        "actor": (
            _text(details.get("actor_id"))
            or _text(details.get("operator"))
            or _text(details.get("requested_by"))
            or "Somebody"
        ),
        "elapsed_minutes": _number(details.get("elapsed_minutes")),
        "limit_minutes": _number(details.get("limit_minutes")),
        "steps": _number(details.get("steps")),
        "operation_type": _text(details.get("operation_type")) or "provider operation",
    }
    quote_field = LOGBOOK_TEMPLATES[key].quote_field
    quote = _text(details.get("reason")) if quote_field == "details.reason" else None
    return [draft(render_bubble(key, fields, quote=quote))]


def _stop_drafts(
    event: LogbookEvent,
    *,
    state: FeatureWorkflowSnapshot,
    draft: Callable[..., _Draft],
    is_latest_stop: bool,
) -> list[_Draft]:
    """Say why a feature stopped, and what the platform says to do about it.

    Two bubbles, both anchored to the terminal event. The first quotes the event's own
    diagnostic -- the F7 narrative, already bounded and screened when it was written. The
    second quotes ``next_action`` from the durable failure summary, which is the sentence the
    overview already shows under "What to do": a reader who gets to the end of the thread
    should not have to go and find it.

    Only the newest stop gets the second bubble, because the summary is singular and describes
    where the feature is now. An earlier stop keeps its own diagnostic, which is a record of
    that moment, and borrows nothing from a later one.
    """
    details = dict(event.details)
    summary: FeatureFailureSummary | None = state.failure_summary
    classification = _text(details.get("failure_classification")) or (
        summary.root_classification if summary is not None else None
    )
    diagnostics = [item for item in _strings(details.get("diagnostics")) if item] or (
        list(summary.diagnostics) if summary is not None else []
    )
    bubbles = [
        draft(
            render_bubble(
                "event.feature_failed",
                {"classification_sentence": _stop_sentence(classification)},
                quote=diagnostics[0] if diagnostics else None,
            )
        )
    ]
    if is_latest_stop and summary is not None and summary.next_action.strip():
        bubbles.append(
            draft(
                render_bubble("event.feature_failed.next_action", quote=summary.next_action),
                emission=1,
            )
        )
    return bubbles


def _artifact_drafts(
    artifact: Artifact, *, names: Mapping[str, str], owners: Mapping[str, str]
) -> list[_Draft]:
    """Turn one immutable artifact into its bubbles, deferred content included."""
    repository_id = _artifact_repository(artifact, owners)
    bubbles = _artifact_bubbles(artifact, names=names, repository_id=repository_id)
    return [
        _Draft(
            timestamp=artifact.timestamp,
            kind=LogbookRecordKind.ARTIFACT,
            record_id=artifact.artifact_id,
            emission=index,
            bubble=bubble,
            # The bubble's own repository wins where it named one: a cross-repository
            # artifact says different things about different repositories.
            repository_id=bubble.repository_id or repository_id,
        )
        for index, bubble in enumerate(bubbles)
    ]


def _artifact_bubbles(  # noqa: C901, PLR0911, PLR0912 - one branch per artifact kind
    artifact: Artifact, *, names: Mapping[str, str], repository_id: str | None
) -> list[LogbookBubble]:
    """Read one artifact as the bubbles it is worth, in the order they happened."""
    repository = _repository_name(repository_id, names)
    if isinstance(artifact, PRDArtifact):
        return [
            render_bubble(
                "artifact.prd", {"title": artifact.title}, quote=artifact.problem_statement
            )
        ]
    if isinstance(artifact, TechnicalPRDArtifact):
        bubbles = [
            render_bubble(
                "artifact.technical_prd",
                {"requirements": _plural(len(artifact.functional_requirements), "requirement")},
                quote=artifact.solution_summary,
            )
        ]
        if artifact.unresolved_questions:
            bubbles.append(
                render_bubble(
                    "artifact.technical_prd.questions",
                    {"questions": _plural(len(artifact.unresolved_questions), "question")},
                    quote=artifact.unresolved_questions[0].question,
                )
            )
        answers = artifact.metadata.get("clarification_answers")
        if isinstance(answers, dict) and answers:
            bubbles.append(
                render_bubble(
                    "artifact.technical_prd.answers",
                    {"answers": _plural(len(answers), "question")},
                    quote="; ".join(f"{key}: {value}" for key, value in sorted(answers.items())),
                )
            )
        return bubbles
    if isinstance(artifact, RepositoryReconnaissanceArtifact):
        bubbles = [
            render_bubble(
                "artifact.repository_reconnaissance",
                {"repository": repository},
                quote=artifact.summary,
            )
        ]
        bubbles.extend(
            render_bubble(
                "artifact.repository_reconnaissance.contradiction",
                {"repository": repository, "question": item.question},
                quote=item.premise,
            )
            for item in artifact.contradicted_premises
        )
        return bubbles
    if isinstance(artifact, ArchitectureArtifact):
        return [
            render_bubble(
                "artifact.architecture",
                {"components": _plural(len(artifact.components), "component")},
                quote=artifact.system_overview,
            )
        ]
    if isinstance(artifact, ExecutionGraphArtifact):
        return [render_bubble("artifact.execution_graph")]
    if isinstance(artifact, TaskPlanArtifact):
        return [
            render_bubble(
                "artifact.task_plan",
                {"tasks": _plural(len(artifact.tasks), "task")},
                quote=artifact.summary,
            )
        ]
    if isinstance(artifact, IntegrationContractArtifact):
        return [
            render_bubble(
                f"artifact.integration_contract.{artifact.status}",
                {"contract_version": artifact.contract_version},
            )
        ]
    if isinstance(artifact, RepositoryExecutionPlanArtifact):
        return [
            render_bubble(
                "artifact.repository_execution_plan",
                {
                    "repositories": _plural(len(artifact.workstreams), "repository"),
                    "merge_strategy": artifact.merge_strategy.replace("_", " "),
                },
            )
        ]
    if isinstance(artifact, CodeCompletionArtifact):
        return _code_completion_bubbles(artifact, repository=repository)
    if isinstance(artifact, ReviewArtifact):
        return _review_bubbles(artifact, repository=repository)
    if isinstance(artifact, ChildWorkflowResultArtifact):
        return _child_result_bubbles(artifact, repository=repository)
    if isinstance(artifact, IntegrationReviewArtifact):
        return _integration_review_bubbles(artifact, names=names)
    if isinstance(artifact, ContractChangeRequestArtifact):
        return [
            render_bubble(
                "artifact.contract_change_request",
                {
                    "repository": _repository_name(artifact.requested_by_repository_id, names),
                    "changes": _plural(len(artifact.requested_changes), "change"),
                },
                quote=artifact.reason,
                repository_id=artifact.requested_by_repository_id,
            )
        ]
    if isinstance(artifact, RepositoryRepairProposalArtifact):
        return [
            render_bubble(
                "artifact.repository_repair_proposal",
                {"repository": repository, "proposed_repair": artifact.proposed_repair},
                quote=artifact.detected_problem,
            )
        ]
    if isinstance(artifact, PullRequestArtifact):
        return [
            render_bubble(
                "artifact.pull_request",
                {
                    "number": artifact.pull_request_number,
                    "repository": artifact.repository,
                    "title": artifact.title,
                },
            )
        ]
    if isinstance(artifact, FeatureCompletionArtifact):
        return [
            render_bubble(
                f"artifact.feature_completion.{artifact.status}",
                {
                    "pull_requests": _plural(len(artifact.pull_request_urls), "pull request"),
                    "merge_order": ", ".join(artifact.merge_order) or "in any order",
                },
            )
        ]
    return [render_bubble("artifact.unknown", {"artifact_type": artifact.artifact_type})]


def _code_completion_bubbles(
    artifact: CodeCompletionArtifact, *, repository: str
) -> list[LogbookBubble]:
    """The attempt's own account of itself, plus everything it deferred until it settled.

    The deferred half is the point. Repair passes, the assertion guard, the wiring and
    placement inspections and the self-review all run *inside* one attempt and are
    deliberately unjournaled while they do -- the engineer's gate depends on there being one
    journaled coding effect per attempt. So none of this can be shown while the attempt is
    running, and all of it is here the moment the attempt records a completion.
    """
    metadata = artifact.metadata
    bubbles = [
        render_bubble(
            f"artifact.code_completion.{artifact.completion_status}",
            {"repository": repository, "files": _plural(len(artifact.file_changes), "file")},
            quote=artifact.summary,
        )
    ]
    passes = _integer(metadata.get("source_repair_passes"))
    if passes:
        execution = metadata.get("source_repair_execution")
        model = (
            _text(execution.get("model")) if isinstance(execution, dict) else None
        ) or "the configured scoped-fix model"
        bubbles.append(
            render_bubble(
                "artifact.code_completion.repairs",
                {"passes": _plural(passes, "further pass", "further passes"), "model": model},
                quote="; ".join(_strings(metadata.get("source_repair_diagnostics"))) or None,
            )
        )
    elif metadata.get("source_repair_engaged") is False:
        # Two different facts, and the record distinguishes them: the loop declined because
        # nothing was mechanically repairable, or there was no scoped-fix boundary to decline
        # with. Reporting the first when it was the second is how a deployment gap reads as a
        # judgement about the code.
        resolved = metadata.get("source_repair_scoped_fix_role_resolved")
        bubbles.append(
            render_bubble(
                "artifact.code_completion.repairs_declined"
                if resolved is not False
                else "artifact.code_completion.repairs_unavailable"
            )
        )
    if _text(metadata.get("terminal_outcome")) == "SCOPED_TEST_ASSERTION_WEAKENED":
        bubbles.append(render_bubble("artifact.code_completion.assertion_guard"))
    wiring = _strings(metadata.get("reachability_issues_unrepaired"))
    if wiring:
        bubbles.append(
            render_bubble(
                "artifact.code_completion.unreachable",
                {"issues": _plural(len(wiring), "new thing", "new things")},
                quote="; ".join(wiring),
            )
        )
    dropped = _strings(metadata.get("reachability_candidates_dropped"))
    if dropped:
        bubbles.append(
            render_bubble(
                "artifact.code_completion.reachability_bounded",
                {"candidates": _plural(len(dropped), "candidate")},
                quote="; ".join(dropped),
            )
        )
    misplaced = _strings(metadata.get("assigned_file_issues_unrepaired"))
    if misplaced:
        bubbles.append(
            render_bubble(
                "artifact.code_completion.misplaced",
                {"places": _plural(len(misplaced), "place")},
                quote="; ".join(misplaced),
            )
        )
    review = metadata.get("self_review")
    if isinstance(review, dict):
        if review.get("ran") is False:
            bubbles.append(render_bubble("artifact.code_completion.self_review_unavailable"))
        else:
            outcome = _text(review.get("outcome")) or ""
            bubbles.append(
                render_bubble(
                    "artifact.code_completion.self_review",
                    {
                        "outcome_sentence": _SELF_REVIEW_SENTENCES.get(
                            outcome, f"the platform recorded the outcome {outcome or 'unstated'}."
                        )
                    },
                    quote=_text(review.get("summary")),
                )
            )
    return bubbles


def _review_bubbles(artifact: ReviewArtifact, *, repository: str) -> list[LogbookBubble]:
    """The reviewer's verdict, and the worst thing it found."""
    bubbles = [
        render_bubble(
            f"artifact.review.{artifact.verdict}",
            {"repository": repository, "findings": _plural(len(artifact.findings), "finding")},
            quote=artifact.summary,
        )
    ]
    ranking = ["critical", "high", "medium", "low", "info"]
    findings = sorted(
        artifact.findings,
        key=lambda item: ranking.index(item.severity) if item.severity in ranking else len(ranking),
    )
    if findings and artifact.verdict != "approved":
        worst = findings[0]
        location = (
            worst.file_path or worst.requirement_id or worst.contract_reference or "the change"
        )
        bubbles.append(
            render_bubble(
                "artifact.review.finding",
                {"severity": worst.severity, "location": location},
                quote=worst.recommended_fix or worst.description,
            )
        )
    return bubbles


def _child_result_bubbles(
    artifact: ChildWorkflowResultArtifact, *, repository: str
) -> list[LogbookBubble]:
    """One repository attempt, settled -- including everything absorbed while it ran.

    The fault history comes first and in the order it happened: a truncated answer, the one
    degraded re-ask it earns, then the provider faults that were given back. None of it is
    journaled as it happens (a provider that fails to answer writes no artifact and no
    lifecycle event), so this is the first moment any of it can be said, and 186's attempt-0
    is exactly this sequence.
    """
    metadata = artifact.metadata
    bubbles: list[LogbookBubble] = []
    if metadata.get("truncated_degraded_retry") is True:
        bubbles.append(
            render_bubble("artifact.child_workflow_result.truncated", {"repository": repository})
        )
        bubbles.append(
            render_bubble(
                "artifact.child_workflow_result.degraded_retry",
                quote=_routing_reason(artifact.model_routing),
            )
        )
    faults = _integer(metadata.get("fault_count"))
    if faults:
        classes = _strings(metadata.get("fault_classes"))
        bubbles.append(
            render_bubble(
                "artifact.child_workflow_result.provider_fault",
                {
                    "faults": _times(faults),
                    "excluded_seconds": _number(metadata.get("fault_seconds_excluded")),
                    "fault_classes": (f"Recorded as {', '.join(classes)}." if classes else ""),
                },
            )
        )
    bubbles.append(
        render_bubble(
            f"artifact.child_workflow_result.{artifact.status}",
            {"repository": repository},
            quote=artifact.blocking_issues[0] if artifact.blocking_issues else None,
        )
    )
    # The triage question, when the stop produced one. `_with_triage` appends it to
    # `blocking_issues` *and* records it here, so quoting the first blocking issue can miss it
    # entirely on a workstream that stopped with several -- and it is the one sentence a
    # person reading a dead workstream actually needs. Skipped when it is already the quote
    # above, so the same words are never in the thread twice.
    question = _text(metadata.get("operator_question"))
    if question and question != (artifact.blocking_issues[0] if artifact.blocking_issues else None):
        bubbles.append(
            render_bubble(
                "artifact.child_workflow_result.operator_question",
                {"repository": repository},
                quote=question,
            )
        )
    if artifact.advisory_review_verdict is not None:
        bubbles.append(
            render_bubble(
                "artifact.child_workflow_result.advisory_verdict",
                {
                    "repository": repository,
                    "advisory_verdict": artifact.advisory_review_verdict.replace("_", " "),
                },
            )
        )
    wall = _optional_number(metadata.get("runtime_wall_seconds"))
    charged = _optional_number(metadata.get("runtime_charged_seconds"))
    # "ran for 0s, of which 0s counted as work" is what a mock attempt measures, and it is
    # not a sentence worth a bubble. A real attempt is never under a second.
    if wall is not None and charged is not None and wall != "0":
        bubbles.append(
            render_bubble(
                "artifact.child_workflow_result.runtime",
                {"repository": repository, "wall_seconds": wall, "charged_seconds": charged},
            )
        )
    routing_reason = _routing_reason(artifact.model_routing)
    model = _routing_model(artifact.model_routing)
    if model is not None:
        bubbles.append(
            render_bubble(
                "artifact.child_workflow_result.routing",
                {"repository": repository, "model": model},
                quote=routing_reason,
            )
        )
    root_cause = (
        _text(artifact.retry_strategy.get("root_cause"))
        if isinstance(artifact.retry_strategy, dict)
        else None
    )
    if root_cause:
        bubbles.append(
            render_bubble(
                "artifact.child_workflow_result.retry_strategy",
                {"repository": repository},
                quote=root_cause,
            )
        )
    refusal = _text(metadata.get("retry_refusal_reason"))
    if refusal:
        bubbles.append(
            render_bubble(
                "artifact.child_workflow_result.retry_refused",
                {"repository": repository},
                quote=refusal,
            )
        )
    return bubbles


def _integration_review_bubbles(
    artifact: IntegrationReviewArtifact, *, names: Mapping[str, str]
) -> list[LogbookBubble]:
    """The cross-repository verdict, and the fix it required of whichever repository owns it."""
    bubbles = [
        render_bubble(
            f"artifact.integration_review.{artifact.review_status}",
            {"fixes": _plural(len(artifact.required_fixes), "fix", "fixes")},
            quote=artifact.compatibility_assessment,
        )
    ]
    if artifact.required_fixes:
        responsible = next(
            (item.responsible_repository_id for item in artifact.cross_repository_findings), None
        )
        bubbles.append(
            render_bubble(
                "artifact.integration_review.required_fix",
                {"repository": _repository_name(responsible, names)},
                quote=artifact.required_fixes[0],
                repository_id=responsible,
            )
        )
    return bubbles


def _operation_drafts(
    operation: ExternalOperation, *, timestamp: datetime, names: Mapping[str, str]
) -> list[_Draft]:
    """Turn one journal row into its bubble, using only the fields the journal endpoint serves.

    Deliberately narrow: the operation's type, status, attempt ratio, error *code*, and its
    ``logical_step`` -- the platform's own name for which step journaled the row, which is what
    tells two callers of one operation type apart and is never rendered as text, only used to
    choose a registry row. Never ``error_message``, never ``result_payload``, and nothing else
    out of ``safe_metadata`` -- the workstream-operations endpoint is held to the same line, and
    the logbook must not become the second, wider channel onto the same rows.
    """
    voice = operation_voice(operation)
    repository = _repository_name(operation.repository_id, names)
    attempt_note = (
        f"Attempt {operation.attempt} of {operation.max_attempts}."
        if operation.attempt > 1 or operation.max_attempts > 1
        else ""
    )
    error_note = (
        f"The platform recorded this as {operation.error_code}. " if operation.error_code else ""
    )
    if voice is None:
        bubble = render_bubble(
            "operation.unknown_type",
            {
                "operation_type": str(operation.operation_type),
                "status": str(operation.status).replace("_", " "),
                "attempt_note": attempt_note,
            },
        )
    else:
        key = _OPERATION_OUTCOME_TEMPLATES.get(operation.status, "operation.recorded")
        if operation.started_at is None and key == "operation.recorded":
            # Journaled and never started: the row exists because the journal is written
            # before the effect, which is the whole point of it.
            attempt_note = f"Nothing was started. {attempt_note}".strip()
        bubble = render_bubble(
            key,
            {
                "doing": voice.doing.format(repository=repository),
                "attempt_note": attempt_note,
                "error_note": error_note,
            },
            agent=voice.agent,
        )
    return [
        _Draft(
            timestamp=timestamp,
            kind=LogbookRecordKind.OPERATION,
            record_id=operation.operation_id,
            emission=0,
            bubble=bubble,
            repository_id=operation.repository_id,
        )
    ]


def _workstream_drafts(child: ChildWorkflowReference, *, names: Mapping[str, str]) -> list[_Draft]:
    """Every attempt a person bought this repository after the platform had stopped it.

    The only thing read from the child row rather than from an event, an artifact or a journal
    row, because a grant is recorded here and nowhere else.

    ``granted_at`` is what places it, and a grant whose timestamp cannot be read is skipped
    rather than placed at a guessed instant. That is the one omission in this module, and it
    is deliberate: the thread is chronological, so an invented time would put a person's
    override at the wrong point in the story -- next to work it did not authorise. Every grant
    the platform has ever written carries the field; the guard is for a row that does not.
    """
    drafts: list[_Draft] = []
    for index, grant in enumerate(child.retry_grants):
        if not isinstance(grant, dict):
            continue
        granted_at = _timestamp(grant.get("granted_at"))
        if granted_at is None:
            continue
        stopped = _text(grant.get("stopped_because"))
        drafts.append(
            _Draft(
                timestamp=granted_at,
                kind=LogbookRecordKind.WORKSTREAM,
                record_id=child.child_workflow_id,
                emission=index,
                bubble=render_bubble(
                    "workstream.retry_granted",
                    {
                        "actor": _text(grant.get("granted_by")) or "Somebody",
                        "repository": _repository_name(child.repository_id, names),
                        "attempts": _plural(
                            _integer(grant.get("attempts")) or 1,
                            "more attempt",
                        ),
                        "stopped_because": (
                            f"It had stopped because: {stopped}" if stopped else ""
                        ),
                    },
                    quote=_text(grant.get("reason")),
                ),
                repository_id=child.repository_id,
            )
        )
    return drafts


def _repository_index(state: FeatureWorkflowSnapshot) -> dict[str, str]:
    """Map every artifact id this feature owns to the repository it belongs to.

    Needed because the artifacts that matter most say nothing about their own repository: a
    ``code_completion`` and a ``review`` carry no ``repository_id`` at all. What does carry
    it is the link -- the child result names the completion and review it was built from, and
    the durable child row names the completion, review and pull request it is currently
    holding. Reading the ownership off those links is how a bubble gets its repository badge
    without the composer guessing from an artifact filename.
    """
    index: dict[str, str] = {}
    for artifact in state.artifacts:
        if isinstance(artifact, ChildWorkflowResultArtifact):
            for linked in (
                artifact.artifact_id,
                artifact.code_completion_artifact_id,
                artifact.review_artifact_id,
            ):
                if linked:
                    index.setdefault(linked, artifact.repository_id)
    for child in state.child_workflows.values():
        for linked in (
            child.code_completion_artifact_id,
            child.review_artifact_id,
            child.pull_request_artifact_id,
        ):
            if linked:
                index.setdefault(linked, child.repository_id)
    return index


def _artifact_repository(artifact: Artifact, index: Mapping[str, str]) -> str | None:
    """Name the repository an artifact belongs to, from the record and never from a filename."""
    declared = getattr(artifact, "repository_id", None)
    if isinstance(declared, str) and declared:
        return declared
    return _text(artifact.metadata.get("repository_id")) or index.get(artifact.artifact_id)


def _repository_name(repository_id: str | None, names: Mapping[str, str]) -> str:
    """Prefer the repository's own name, fall back to its id, and never invent one."""
    if repository_id is None:
        return "this feature"
    return names.get(repository_id, repository_id)


def _routing_reason(routing: Mapping[str, Any] | None) -> str | None:
    """Read the recorded reason a model was selected, never a reason composed here."""
    if not isinstance(routing, dict):
        return None
    return _text(routing.get("routing_reason"))


def _routing_model(routing: Mapping[str, Any] | None) -> str | None:
    """Read the model an attempt actually ran on, as its own record named it."""
    if not isinstance(routing, dict):
        return None
    return _text(routing.get("model"))


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    """Count something the way a person writes it: "1 finding", "3 findings".

    A field rather than a sentence, which is what keeps it out of the registry's remit. The
    templates used to carry "{n} finding(s)" and read like a form: this tab has to be legible
    to somebody who does not work on the platform, and "1 requirement(s)" is the first thing
    that says it was not written for them.
    """
    return f"{count} {singular if count == 1 else (plural or f'{singular}s')}"


def _times(count: int) -> str:
    """How many times something happened: "once", "twice", "5 times"."""
    return {1: "once", 2: "twice"}.get(count, f"{count} times")


def _text(value: Any) -> str | None:
    """Return a non-empty string, or nothing at all."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _strings(value: Any) -> list[str]:
    """Read a persisted list of strings, ignoring anything that is not one."""
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _integer(value: Any) -> int:
    """Read a persisted count, treating anything else as none recorded."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _number(value: Any) -> str:
    """Format a persisted measurement for a sentence, without inventing precision."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "an unrecorded number of"
    return f"{value:g}" if isinstance(value, float) else str(value)


def _optional_number(value: Any) -> str | None:
    """Format a measurement, or report that the record carries none."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return f"{round(value):d}"


def _timestamp(value: Any) -> datetime | None:
    """Read a persisted ISO instant without letting a malformed one break the read.

    A value with no zone is read as UTC rather than left naive. Not cosmetic: everything else
    the thread is sorted against is timezone-aware, and one naive instant among them makes the
    sort raise -- so a single old row written without a zone would take the whole endpoint
    down rather than render one bubble oddly.
    """
    parsed: datetime | None = None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    if parsed is None:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def logbook_agents() -> Iterable[str]:
    """Every chip the registry can attribute a bubble to."""
    return [str(item) for item in LogbookAgent]


__all__ = [
    "LOGBOOK_TEMPLATES",
    "OPERATION_STEP_VOICES",
    "OPERATION_VOICES",
    "QUOTE_CHARACTER_BOUND",
    "LogbookAgent",
    "LogbookBubble",
    "LogbookEntry",
    "LogbookEvent",
    "LogbookRecord",
    "LogbookRecordKind",
    "LogbookTemplate",
    "LogbookTone",
    "OperationVoice",
    "UnknownLogbookTemplate",
    "clip_quote",
    "feature_logbook",
    "logbook_agents",
    "operation_voice",
    "render_bubble",
]
