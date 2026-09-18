"""Parent orchestration for one feature delivered through many repository workstreams."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

import structlog
from pydantic import ValidationError

from adapters.figma_adapter import FigmaClientError
from adapters.git_adapter import (
    GitAdapterError,
    GitAuthenticationError,
    GitBranchNotFoundError,
)
from adapters.github_adapter import GitHubAdapterError, MockGitHubService
from adapters.llm_adapter import DETERMINISTIC_PROVIDER_CLASSIFICATIONS, LLMAdapterError
from agents.engineer.agent import RequiredContextRefusal
from agents.integration_reviewer.agent import (
    IntegrationReviewerAgent,
    assessment_coverage_limitations,
)
from agents.planner.feature_planner import DeterministicFeaturePlanner, FeaturePlanner
from agents.shared.contracts import (
    ARTIFACT_FILENAMES,
    FEATURE_ARTIFACT_FILENAMES,
    attempt_artifact_id,
    create_artifact,
    safe_error_diagnostics,
)
from api.control_plane import RequestScopedCredentials
from api.schemas import ClarificationAnswer
from artifacts.schemas import (
    ArchitectureArtifact,
    Artifact,
    BaseArtifact,
    ChildWorkflowResultArtifact,
    ClarificationQuestion,
    CodeCompletionArtifact,
    ContractChangeRequestArtifact,
    DesignConflictArtifact,
    DesignDetailArtifact,
    DesignSnapshotArtifact,
    FeatureCompletionArtifact,
    FeatureRevisionRequestArtifact,
    IntegrationContractArtifact,
    IntegrationReviewArtifact,
    IntegrationReviewFinding,
    OverruledReviewFinding,
    PRDArtifact,
    PullRequestArtifact,
    RepositoryExecutionPlanArtifact,
    RepositoryReconnaissanceArtifact,
    RepositoryRepairProposalArtifact,
    RepositoryWorkstreamPlan,
    Requirement,
    ReviewArtifact,
    ReviewFindingCounts,
    TechnicalPRDArtifact,
)
from configs.model_roles import TIER_LABELS, PerformanceTier
from runtime_identity import load_runtime_identity
from services.cancellation import CancellationRequested, CancellationToken
from services.design_conflict import (
    VERDICTS,
    answered_payload,
    conflict_payload,
    current_design_conflicts,
    find_design_conflict,
    recurring_demand_payload,
    restated_for_the_next_attempt,
    settled_questions,
)
from services.design_resolution import (
    DesignReferenceResolver,
    DeterministicDesignResolver,
    design_detail_artifact_id,
    design_snapshot_artifact_id,
    unreachable_design_detail,
)
from services.external_operations import ExternalOperationExecutor
from services.repository_repair import (
    build_repair_payload,
    find_repair,
    open_repair_for,
    repair_is_actionable,
    repair_is_stale,
    repairable_issues,
)
from state.enums import (
    ChildWorkflowStatus,
    DeploymentStrategy,
    FeatureWorkflowStatus,
    MergeStrategy,
    TargetedAttempt,
    WorkstreamRole,
)
from state.external_operations import ExternalOperationType, WorkflowCheckpointBoundary
from state.failure_diagnosis import (
    DiagnosedFailure,
    FailureStage,
    FeatureFailureClassification,
    classification_of,
)
from state.feature_models import (
    ChildWorkflowReference,
    FeatureWorkflowSnapshot,
    RepositorySpec,
    ensure_feature_failure_summary,
)
from state.feature_transitions import transition_feature
from storage.external_operation_store import (
    ExternalOperationError,
    OperationInProgressError,
    OperationLeaseLostError,
    OperationReconciliationRequired,
    OperationReplayRefused,
    OperationResult,
    OperationTransitionConflictError,
)
from tools.acceptance_criteria import unverifiable_criteria
from tools.implementation_completeness import ImplementationCompletenessResult, meaningful_progress
from tools.llm_call_record import last_llm_call_payload, reset_llm_call_record
from tools.model_routing import (
    ModelExecutionMode,
    ModelRouter,
    ModelRoutingDecision,
    ModelRoutingInputs,
    degraded_truncation_retry,
)
from tools.repository_preflight import PreflightIssue
from tools.requirement_reconciliation import (
    QUESTION_ID_PREFIX,
    AnsweredClarification,
    RequirementReconciliation,
    contradiction_questions,
    reconciled_requirements,
    reconciliation_change_list,
)
from tools.resolved_issue_ledger import (
    IssueAuthority,
    RecurringDemandTheme,
    RecurringIssue,
    design_conflict_narrative,
    recurring_demand_narrative,
    recurring_demand_theme,
    recurring_resolved_issues,
    resolved_issue_ledger,
    satisfied_invariant_lines,
    settled_question_lines,
)
from tools.retry_strategy import (
    REPRODUCED_PREVIOUS_CHANGE,
    RETURNED_TO_EARLIER_STATE,
    FailureClassification,
    RetryDecision,
    TerminalCause,
    TerminalTriage,
    TreeResubmission,
    a_required_command_rejected_the_source,
    build_retry_plan,
    classify_failure,
    decide_child_retry,
    diagnostic_file_references,
    diagnostic_signature,
    requires_feature_engineer,
    retry_counter_field,
    triage_stopped_workstream,
)
from tools.review_fix_classification import (
    ReviewFixClassification,
    classify_findings,
    fingerprint_for_text,
)
from tools.scoped_tests import is_scoped_test_diagnostic
from tools.self_review import SELF_REVIEW_SUBSTANTIVE_OUTCOME
from tools.source_formatting import SourceValidationError
from tools.unchanged_failure import (
    strategy_change_line,
    unchanged_failure_narrative,
    unchanged_failure_run,
)
from workflow_schema import workflow_mutation_identity_error

type FeatureCheckpointWriter = Callable[
    [FeatureWorkflowSnapshot, WorkflowCheckpointBoundary, str | None], Awaitable[None]
]
# Writes one timeline event mid-run, outside any state replacement: (feature_id, event,
# details). The checkpoint writer is the only other mid-run durable channel, and it can only
# say "a boundary passed" -- `repository_planned_blind` is a fact about a repository, not a
# boundary, which is why it gets a writer of its own shape.
type FeatureEventWriter = Callable[[str, str, dict[str, Any]], Awaitable[None]]


# The three convergence bounds this loop used to hold -- how many identical resubmissions,
# returns to an earlier state, and repeated diagnostic signatures a workstream is allowed --
# now live with the verdict they belong to, in `tools/retry_strategy.py`. They were applied
# here, after `decide_child_retry` had already answered, which meant this loop could stop a
# workstream the retry authority had just permitted and rewrite its reason underneath it.
#
# A provider fault says nothing about the repository's code, so it must not end a workstream
# the way a rejected attempt does. Bounded, because a provider that is down stays down and the
# feature has to reach a human rather than retry until its deadline.
# Four, and a wait that grows. Two flat five-second retries did not outlast a real blip:
# -077's backend took three provider failures inside four minutes on one attempt, exhausted
# the allowance, and ended holding two test-coverage findings it was close to clearing.
_ALLOWED_INFRASTRUCTURE_FAULTS = 4
_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS = 5.0
# What the same allowance is worth when the fault cost no model spend. A model call that
# failed is a stalled attempt an operator is waiting on, so its tail stays short: 5, 10, 20,
# 40 -- about seventy-five seconds -- and then a person decides. A clone or a fetch is
# different in the one way that matters here: there is nothing to protect. Waiting longer
# spends no tokens, holds no attempt a model could have used, and is excluded from the
# runtime ceiling exactly as the short tail already is.
#
# So the same four retries wait four times as long: 20, 40, 80, 160 -- five minutes rather
# than seventy-five seconds. AB-Feature-190 died inside a forty-second GitHub outage that
# this tail outlasts twice over, and 54- Part 2's note that seventy-five seconds is too short
# for routine weather is what this answers.
#
# A multiplier rather than a second constant on purpose: thirteen tests set the base to zero
# to keep the fault paths instant, and a separate constant would have left every Git-fault
# test sleeping for real until somebody remembered to patch it too.
_NO_MODEL_COST_FAULT_BACKOFF_MULTIPLIER = 4
# How long one uninterrupted sleep may last. The backoff is served in slices with a
# cancellation check between them: the loop only checks cancellation between attempts, so a
# single 160-second sleep would make a cancelled feature take almost three minutes to notice
# -- which is worse than the forty-second tail this change replaces, not better.
_FAULT_BACKOFF_SLICE_SECONDS = 5.0
# Why no further attempt was scheduled when a review re-demanded work an earlier attempt had
# already delivered. Phrased as a fact about the demands rather than about the model: this
# workstream did what it was told, twice, and the two instructions disagree.
_DESIGN_CONFLICT_REFUSAL = (
    "A review is demanding a change this workstream already made and a later attempt removed, "
    "so the disagreement is between the demands and not inside the code. The remaining "
    "attempts were not spent, because another one would only reverse the decision again."
)
# The other stop that opens a design conflict, and it is not a reversal: nothing here was made
# and then taken out -- the demand was never satisfied at all, in any round. Sharing the
# sentence above put a story on the record that the evidence contradicts, and this field is
# what an escalation and the assistant's own context read as the cause of the stop:
# AB-Feature-225's workstream reported that a change had been "made and removed" when its three
# rejections were one requirement re-asserted in three wordings and never met once.
# `recurring_demand_narrative` says as much in its own docstring; only this constant did not.
_RECURRING_DEMAND_REFUSAL = (
    "A review has demanded the same thing in every round and no attempt has satisfied it, so "
    "the disagreement is about what is being required and not about what the code does. The "
    "remaining attempts were not spent, because another one would be refused the same way."
)
# The same field, for the other stop this loop learned to make. Phrased as a fact about the
# evidence rather than about the model: the required command and the files it named were
# identical every round, so another repair of the same shape has nothing left to tell anyone.
_UNCHANGED_FAILURE_REFUSAL = (
    "The same workspace files appeared in the failing evidence of every one of the last "
    "attempts, so repeating the repair has stopped producing new information. The remaining "
    "attempts were not spent."
)
# The same allowance for the stages that run before any repository does -- product manager,
# reconnaissance, planner -- because until now they had none at all. The catch below sits
# inside the child loop, so a provider fault in planning left `resume` entirely and the
# feature was put back in front of a human with two of its three queue attempts unspent.
# AB-Feature-168 died that way three times in one morning, each run losing twenty to
# forty-five minutes of planning to a dropped connection.
#
# Two rather than four, because the unit of work is not the same size. A child attempt is
# minutes, so four retries there costs an operator little; one maximum-effort planning call
# runs long enough that four would put a feature hours away from the person who has to
# decide about it. Two covers a blip and still reaches a human inside roughly one more call.
_ALLOWED_FEATURE_STAGE_INFRASTRUCTURE_FAULTS = 2

# How many transient faults one repository's detail resolution absorbs before it degrades to
# "the design was unreachable" and lets the workstream run anyway.
#
# Two, matching the planning allowance rather than the child loop's four, and for a stronger
# reason than either: the thing being retried is a single read that costs seconds, and what
# happens when the allowance runs out is not a failure. It is an honest gap the prompts already
# know how to read. So the allowance only has to cover a blip -- there is nothing to protect a
# person from by trying harder, and a design source that is genuinely down should not hold a
# repository's whole workstream while it is retried.
#
# AB-Feature-227 is why the number is small and the outcome is a degrade: its design step spent
# every retry the queue reserves inside 60 seconds against a Figma rate limit whose window is
# measured in hours. No allowance sized for a blip will ever clear that, and the only useful
# answer is to say so and carry on.
_ALLOWED_DESIGN_DETAIL_INFRASTRUCTURE_FAULTS = 2
# The provider and the clone, because neither is the repository's code. A clone that fails is
# journaled `failed_retryable` with attempts still reserved, and -075's backend died at
# retry_count 0 holding two of them unspent. GitSafetyError is deliberately absent: a refused
# path is a decision, not a fault, and repeating it would only refuse again.
#
# `GitAuthenticationError` is a `GitAdapterError` and so matches this tuple, and is excluded
# a line earlier in `is_transient_provider_fault` for exactly GitSafetyError's reason: a
# credential the remote refused is a decision too, and it refuses again every time.
# `FigmaClientError` joins them for the same reason a clone failure is here: a design fetch
# that never got an answer is weather, and it says exactly as little about the requirement as a
# dropped model connection does. Which of its refusals is weather is the adapter's own
# `retryable`, read through `_figma_design_fault` below -- never re-derived here.
_INFRASTRUCTURE_FAULTS = (
    LLMAdapterError,
    GitAdapterError,
    ExternalOperationError,
    FigmaClientError,
)
# Provider answers that do not change on a retry. The adapter records what the provider
# said in `failure_classification` -- its own verdicts (`model_refusal`,
# `response_truncated`) and the SDK's 4xx type names -- and retrying any of them re-asks a
# question whose answer is already known. AB-Feature-174's backend spent three identical
# twelve-minute coding calls and its whole operation budget on one such answer, then
# reported "the provider did not answer" when the provider had answered every time.
# Timeouts, connection drops, 429s and 5xx stay faults: those are the weather this
# allowance exists for.
#
# One narrow exception lives in the child loop, and it does not weaken this rule: a
# `response_truncated` answer may be re-asked exactly once with the request *changed* --
# the effort stepped down one level, which is the remedy the truncation diagnostic itself
# prescribes (see `degraded_truncation_retry`). Verbatim, every one of these stays
# terminal.
#
# Defined with the adapter that raises every one of these, and imported here rather than
# restated: the Engineer's in-attempt repair loop needs the same answer, and it cannot import
# this module. Three readers, one definition.
_DETERMINISTIC_PROVIDER_CLASSIFICATIONS = DETERMINISTIC_PROVIDER_CLASSIFICATIONS
# The GitHub REST statuses that are weather rather than an answer. A 408 is the provider
# timing out and a 429 is it asking for exactly a later retry; every other 4xx is GitHub
# answering "no", and it answers the same way every time. A 5xx, or no status at all, is the
# dropped connection the fault allowance exists for.
_GITHUB_WEATHER_HTTP_STATUSES = frozenset({408, 429})
# How far the cause chain is walked for a classification. The coding call's adapter error
# reaches the child loop wrapped in `ExternalOperationError`, so the top-level type never
# carries it; bounded because __context__ chains can be arbitrarily long.
_FAULT_CAUSE_WALK_LIMIT = 8
# For rendering a service's own `Retry-After` as a sentence. Named rather than inline because
# `_duration_phrase` reads them in a chain where a bare 3600 is indistinguishable from a bound.
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 60 * 60
_SECONDS_PER_DAY = 24 * 60 * 60


def _deterministic_provider_error(error: BaseException) -> BaseException | None:
    """Return the first error in the chain whose classification says a retry cannot help."""
    current: BaseException | None = error
    for _ in range(_FAULT_CAUSE_WALK_LIMIT):
        if current is None:
            return None
        classification = getattr(current, "failure_classification", None)
        if (
            isinstance(classification, str)
            and classification in _DETERMINISTIC_PROVIDER_CLASSIFICATIONS
        ):
            return current
        current = current.__cause__ or current.__context__
    return None


def _refused_credential_error(error: BaseException) -> BaseException | None:
    """Return the refused-credential failure in this chain, when that is what happened.

    The same bounded walk `_deterministic_provider_error` does, and needed for the same
    reason: a clone reaches this loop wrapped in `ExternalOperationError`.

    Separate from that function rather than folded into it, because the two answer different
    questions about different services. That one asks "did the model provider answer
    something a retry cannot change"; this one asks "did the Git remote refuse the credential
    it was given" -- and the remedy is not another attempt at any interval, it is a person
    replacing a credential.
    """
    current: BaseException | None = error
    for _ in range(_FAULT_CAUSE_WALK_LIMIT):
        if current is None:
            return None
        if isinstance(current, GitAuthenticationError):
            return current
        current = current.__cause__ or current.__context__
    return None


def _missing_branch_error(error: BaseException) -> BaseException | None:
    """Return the absent-base-branch failure in this chain, when that is what happened.

    The same bounded walk the two above do, and separate from both for the reason they are
    separate from each other: this asks "was the repository configured to build from a branch
    that is not there". The remedy is neither another attempt nor a new credential -- it is
    one field corrected in Settings.

    AB-Feature-221 spent three fault retries and 140 seconds of backoff per repository on a
    ref that could not appear between attempts, and was recorded as a platform defect. The
    clone pins the configured branch now, so the answer arrives at the clone and arrives
    classified; this is what stops it being treated as weather on the way out.
    """
    current: BaseException | None = error
    for _ in range(_FAULT_CAUSE_WALK_LIMIT):
        if current is None:
            return None
        if isinstance(current, GitBranchNotFoundError):
            return current
        current = current.__cause__ or current.__context__
    return None


def _figma_design_fault(error: BaseException) -> FigmaClientError | None:
    """Return the classified Figma failure in this chain, when that is what happened.

    The same bounded walk the two seams below use: a resolution's fault reaches the step
    wrapped in `ExternalOperationError`, so the top-level type never carries it.
    """
    current: BaseException | None = error
    for _ in range(_FAULT_CAUSE_WALK_LIMIT):
        if current is None:
            return None
        if isinstance(current, FigmaClientError):
            return current
        current = current.__cause__ or current.__context__
    return None


def _github_rest_fault(error: BaseException) -> GitHubAdapterError | None:
    """Return the classified GitHub REST failure in this chain, when that is what happened.

    The same bounded walk as above. Only failures the adapter classified count: a
    `GitHubAdapterError` without a classification is a local decision -- empty input, a
    missing mock pull request -- and repeating a decision only decides the same thing again,
    so those keep falling through to the final refusal below.
    """
    current: BaseException | None = error
    for _ in range(_FAULT_CAUSE_WALK_LIMIT):
        if current is None:
            return None
        if isinstance(current, GitHubAdapterError) and current.failure_classification:
            return current
        current = current.__cause__ or current.__context__
    return None


def _truncated_response_error(error: BaseException) -> BaseException | None:
    """Return the truncation in this chain, when that is what the provider answered.

    Narrower than `_deterministic_provider_error` on purpose: of the deterministic
    classifications, only `response_truncated` carries its own remedy inside the request
    (a smaller thinking spend against the same bound). A refusal or a 4xx prescribes
    nothing a re-send could change, so neither is ever eligible for the degraded retry.
    """
    inner = _deterministic_provider_error(error)
    classification = getattr(inner, "failure_classification", None)
    return inner if classification == "response_truncated" else None


def _is_git_fault(error: BaseException) -> bool:
    """Whether the failure that spent the fault allowance came from Git and not from a model.

    Walked over the same bounded cause chain `_deterministic_provider_error` walks, and for
    the same reason: a clone reaches the child loop wrapped in `ExternalOperationError`, so
    the top-level type never says Git.

    `_INFRASTRUCTURE_FAULTS` holds three unrelated services precisely because none of them is
    the repository's code -- but a narrative that then names only one of them sends its reader
    to the wrong service. AB-Feature-190's four clone retries died on a forty-second GitHub
    outage and were reported as "The model provider did not answer (GitAdapterError)"; the
    operator correlated timestamps by hand because nothing in the record mentioned Git.
    """
    current: BaseException | None = error
    for _ in range(_FAULT_CAUSE_WALK_LIMIT):
        if current is None:
            return False
        if isinstance(current, GitAdapterError):
            return True
        current = current.__cause__ or current.__context__
    return False


def _fault_source_name(error: BaseException, *, subject: bool = True) -> str:
    """Name the external service whose repeated failure ended a workstream.

    Three services, because there are three. The design branch was missing until 2026-09-12
    and a `FigmaClientError` fell through to the model provider, so AB-Feature-227 and -228
    both sent their operator to confirm that OpenAI was answering when Figma was the one
    refusing. Read through `_figma_design_fault` and `_is_git_fault` rather than the top-level
    type, because both arrive wrapped in `ExternalOperationError`.
    """
    if _is_git_fault(error):
        return "The Git remote" if subject else "the Git remote"
    if _figma_design_fault(error) is not None:
        return "The design source" if subject else "the design source"
    return "The model provider" if subject else "the model provider"


def _fault_classification(error: BaseException) -> FeatureFailureClassification:
    """Name the external service a spent fault allowance is recorded against.

    Only ever called where `is_transient_provider_fault` has already admitted the error, so
    the effect is established not to have landed and both values are retryable. What the
    split buys is the surfaces that read the classification rather than the sentence below:
    the feature-level summary, its next-action line, and the logbook's one-liner all said
    "the model provider did not answer" for AB-Feature-190's GitHub outage, because 54-
    Part 2 reworded the narrative and left the classification alone.

    The design branch is here for that same reason and not only in `_fault_source_name`: the
    sentence and the classification are read by different surfaces, and fixing one of them is
    how this bug has now been shipped twice.
    """
    if _is_git_fault(error):
        return FeatureFailureClassification.GIT_REMOTE_UNAVAILABLE
    if _figma_design_fault(error) is not None:
        return FeatureFailureClassification.DESIGN_SOURCE_UNAVAILABLE
    return FeatureFailureClassification.PROVIDER_UNAVAILABLE


def transient_fault_classification(error: BaseException) -> FeatureFailureClassification:
    """Classify one failure, naming a service only where the effect is known not to have landed.

    The platform's single answer to "what stopped this, and may it be attempted again". Two
    callers reach the same question from opposite ends of the run: the child loop, when a
    repository's fault allowance is spent, and the queue's `fail_exhausted_fault`, when the
    entry has no attempt left. Both had the expression inline; only one of them had it right.

    `is_transient_provider_fault` is asked first and is what makes this safe. A service value
    is recorded from the *confirmed effect* -- that admission test is the only thing in the
    platform that establishes the operation did not land -- and never from an exception's type
    name, which is why `normalize_classification` still maps no type to
    `GIT_REMOTE_UNAVAILABLE`. A `GitAdapterError` escaping publication may well have pushed,
    and presenting that as retryable is the guess the operation journal exists to prevent.

    Everything the admission test rejects falls through to `classification_of`, so a
    deterministic provider answer, a refused credential, an unconfirmed external effect and
    the worker volume all keep the classification they already declare.
    """
    if is_transient_provider_fault(error):
        return _fault_classification(error)
    return classification_of(error)


def exhausted_fault_diagnostics(error: BaseException, *, attempts: int) -> tuple[str, ...]:
    """Say which service failed and that the platform's own allowance for it is spent.

    The sentences the queue-exhaustion record leads with, and the place the honest distinction
    lives. `git_remote_unavailable` and `provider_unavailable` say an external service did not
    answer, which is true whether one attempt failed or every one did; that a person is now
    the only thing that will move this feature is a fact about *this* route, so it is stated
    here rather than folded into a classification the child loop shares.

    Empty for anything the fault admission test rejects. The workspace-capacity exhaustion
    comes through the same executor method and already carries the sentence its own
    classification is worth.

    Composed from platform constants and a platform-owned counter only -- no model output, no
    checkout, no process environment -- because `safe_error_diagnostics` publishes what it is
    handed into durable state.
    """
    if not is_transient_provider_fault(error):
        return ()
    service = _fault_source_name(error)
    object_form = _fault_source_name(error, subject=False)
    sentences = [
        f"{service} did not answer, and this feature spent all {attempts} of the attempts its "
        "queue entry reserves for that before stopping. Nothing here is a finding about the "
        "target repository or the requirement.",
        f"The platform will not try again on its own. Confirm {object_form} is answering, then "
        "resume this feature: work that already completed is replayed from the operation "
        "journal rather than repeated.",
    ]
    waiting = _retry_window_sentence(error, service=service)
    if waiting is not None:
        # Third and last, because it qualifies the instruction above rather than replacing it:
        # a resume is still the action, and this says when it can succeed.
        sentences.append(waiting)
    return tuple(sentences)


def _retry_window_sentence(error: BaseException, *, service: str) -> str | None:
    """Say how long the service asked to be left alone, where it said so.

    The record's most actionable line when it exists, and it did exist for AB-Feature-228: the
    429 that ended it carried `Retry-After: 261044`, so the honest instruction was "come back
    in about three days" and the one a person got was "resume this feature". Only the design
    source supplies this today -- `_figma_design_fault` is the only seam that carries a parsed
    window -- so this stays silent for every other fault rather than inventing one.
    """
    fault = _figma_design_fault(error)
    seconds = getattr(fault, "retry_after_seconds", None)
    if not isinstance(seconds, int) or seconds <= 0:
        return None
    return (
        f"{service} asked for {_duration_phrase(seconds)} before the next request, so a resume "
        "before then will spend this feature's attempts against the same refusal."
    )


def _duration_phrase(seconds: int) -> str:
    """Render a wait in the largest unit that does not overstate its precision."""
    if seconds < _SECONDS_PER_MINUTE:
        return f"another {seconds} seconds"
    if seconds < _SECONDS_PER_HOUR:
        return f"about another {round(seconds / _SECONDS_PER_MINUTE)} minutes"
    if seconds < _SECONDS_PER_DAY:
        return f"about another {round(seconds / _SECONDS_PER_HOUR)} hours"
    return f"about another {round(seconds / _SECONDS_PER_DAY, 1)} days"


def _fault_backoff_seconds(error: BaseException, *, fault_count: int) -> float:
    """Return how long to wait before re-attempting after one absorbed fault.

    Doubling from the base, as it has since -077, and four times the base for a fault with no
    model spend behind it -- see `_NO_MODEL_COST_FAULT_BACKOFF_MULTIPLIER`. The discriminator
    is `_is_git_fault` and not the top-level type, for the reason that function documents: a
    clone reaches these loops wrapped in `ExternalOperationError`.
    """
    base = float(_INFRASTRUCTURE_FAULT_BACKOFF_SECONDS)
    if _is_git_fault(error):
        base *= _NO_MODEL_COST_FAULT_BACKOFF_MULTIPLIER
    return base * float(2 ** (fault_count - 1))


def _fault_did_not_answer(error: BaseException, *, error_name: str) -> str:
    """Say which external service failed, in the sentence a stopped workstream carries.

    Two whole sentences rather than one with a substituted noun. The classification now
    names the service too (`_fault_classification`), so this is no longer the only place a
    reader learns it -- but it stays the place that says it in full, because a classification
    is a category and this is the line an operator reads first.
    """
    if _is_git_fault(error):
        return (
            f"The Git operation did not complete ({error_name}) while this repository was "
            "running. This is the Git remote or the platform's access to it, not the model "
            "provider, and nothing here is a finding about this repository."
        )
    return (
        f"The model provider did not answer ({error_name}) while this repository was "
        "running. Nothing here is a finding about this repository."
    )


def _fault_label(error: BaseException) -> str:
    """Name a fault for the log: the type, plus the classification when one is declared.

    AB-Feature-174's three fault-retry warnings carried no error identity at all, so the
    difference between a dropped connection and a deterministic provider answer was
    unknowable from the record. Both tokens are platform- or library-owned symbols, never
    provider message text.
    """
    classification = getattr(error, "failure_classification", None)
    inner = _deterministic_provider_error(error)
    if inner is not None:
        classification = inner.failure_classification  # type: ignore[attr-defined]
    name = type(error).__name__
    return f"{name}/{classification}" if isinstance(classification, str) else name


# The named subclasses of `ExternalOperationError` all mean the same thing: what happened
# outside this process could not be confirmed, so a person has to look. Retrying those would
# be the platform guessing about an effect it deliberately refuses to guess about, which is
# the one thing the operation journal exists to prevent. Only the bare base class -- the
# generic "failed before a confirmed result" wrap -- is a fault worth another attempt.
#
# AB-Feature-111's backend is why this exists at all. Its coding call failed, the journal
# recorded `failed_retryable` with two of three attempts still reserved, and the workstream
# ended at retry_count 0 having spent none of its own -- because the exception arrived as
# `ExternalOperationError` and the tuple above only named the adapter errors.
_UNCONFIRMED_OPERATION_FAULTS = (
    OperationInProgressError,
    OperationLeaseLostError,
    OperationReconciliationRequired,
    OperationTransitionConflictError,
)


def is_transient_provider_fault(error: BaseException) -> bool:
    """Whether one failure is the provider's and worth the attempt again.

    The single answer to that question, so the queue and the workflow cannot drift into
    disagreeing about it. Both tuples below already encode the distinction the platform
    cares about -- infrastructure failed, versus an external effect nobody could confirm --
    and the second must win, because attempting an unconfirmed effect again is the one guess
    the operation journal exists to prevent.
    """
    if isinstance(error, _UNCONFIRMED_OPERATION_FAULTS):
        return False
    if isinstance(error, OperationReplayRefused):
        # The platform's own record answering, not the remote. A row that already ended under
        # this attempt returns this refusal identically on every retry, so the allowance buys
        # nothing: AB-Feature-203's granted attempt spent four backoffs and 207 seconds to
        # reach the same sentence, journaling no operations at all. Deliberately its own
        # branch rather than a fifth entry in the tuple above -- nothing was attempted here,
        # so calling it an unconfirmed effect would put the wrong reason on the right answer.
        return False
    if _refused_credential_error(error) is not None:
        # The remote answered, and the answer was no. AB-Feature-190 and -191 each spent four
        # fault retries per repository against a PAT that had expired at midnight, and every
        # one of them was refused in exactly the same way -- the allowance exists for weather,
        # and a credential a person has to replace is not weather.
        return False
    if _missing_branch_error(error) is not None:
        # The remote answered, and it has no such branch. AB-Feature-221's two repositories
        # each spent three fault retries and 140 seconds waiting for a ref to appear that no
        # amount of waiting produces -- a branch name is a field somebody corrects, and a
        # field is not weather either.
        return False
    if _deterministic_provider_error(error) is not None:
        # The provider answered; the answer is just not the one anybody wanted. A refusal,
        # a truncated response, or a 4xx returns identically on every retry, so spending
        # the fault allowance on it converts a diagnosable stop into "the provider did not
        # answer" -- which is how AB-Feature-174's backend ended.
        return False
    figma = _figma_design_fault(error)
    if figma is not None:
        # The design provider's analogue of the two refusals above, behind this same single
        # seam. The adapter already classified it -- a 401 or 403 is a credential a person
        # replaces, a 404 is a file this token cannot open, and both answer identically every
        # time; a 429, a 408, a 5xx and a dropped connection are the weather this allowance is
        # for. The predicate is the adapter's, so the workflow and the adapter cannot drift
        # into disagreeing about which is which.
        return figma.retryable
    github = _github_rest_fault(error)
    if github is not None:
        # The GitHub REST analogue of the two refusals above, behind this same single seam.
        # A 4xx is the provider answering no -- sixteen of the seventeen recorded cross-link
        # comments died on one 403 whose remedy was a permission grant a person had to make,
        # and no interval of retrying changes a grant. Everything else the adapter classified
        # (a 5xx, a connection that never got an answer, the two statuses that themselves ask
        # for a retry) is weather.
        status = github.provider_status
        return not (
            status is not None
            and 400 <= status < 500  # noqa: PLR2004 - the HTTP 4xx range, not a magic bound
            and status not in _GITHUB_WEATHER_HTTP_STATUSES
        )
    return isinstance(error, _INFRASTRUCTURE_FAULTS)


_LOGGER = structlog.get_logger(__name__)
# Strips a trailing revision qualifier so a revision of a revision stays one level deep.
_REVISION_QUALIFIER = re.compile(r"\.revision-\d+$")


# A repository that stopped and can be sent back through the child loop. RUNNING would race
# its own attempt; APPROVED and COMPLETED would discard work that already passed review;
# PENDING has not run at all, which ordinary resume already covers. BLOCKED is excluded on
# purpose: it stopped because something it depends on failed, so the dependency is what needs
# the attempt.
_RETRYABLE_CHILD_STATUSES = frozenset(
    {ChildWorkflowStatus.FAILED, ChildWorkflowStatus.REVIEW_REJECTED}
)


class FeatureResumeNotEligible(RuntimeError):
    """Every workstream has reached a terminal retry decision, so a resume has nothing to run.

    Not a failure of the feature and not a reason to move it: its repositories already carry
    the verdicts and refusals that say why they stopped, and re-running them on the strength
    of no new information is exactly what resume must not do.

    It is an exception rather than a returned snapshot because of what returning did. The
    guard this replaces returned the state unchanged, the caller saw a normal return, and the
    queue closed the entry `succeeded`. A feature disappeared with no event anywhere saying it
    had been asked and had declined -- which is indistinguishable, to every reader, from a
    feature nothing ever tried.
    """


class ClarificationAnswerError(ValueError):
    """Raised when supplied clarification answers do not match the pending questions.

    A caller mistake, not a workflow failure. Marking the feature failed for it stranded a
    run permanently: the retry with correct answers was then refused because the feature was
    no longer waiting for human input, so one malformed request destroyed the whole feature.
    """

    def __init__(self, *args: object) -> None:
        """Expose the explanation so the caller can correct the request."""
        self.diagnostics = tuple(str(item) for item in args if str(item))
        super().__init__(*args)


class FeatureWorkflowError(DiagnosedFailure, RuntimeError):
    """Raised when a parent-feature invariant rejects an operation.

    The detailed exception remains useful to the immediate caller, but identifiers inside it
    may originate in a model response, so the message is never what reaches durable state.
    ``diagnostic`` is: one sentence, written at the raise site, composed from platform
    constants alone.

    Each raise site states its own. Until this task the class derived one by matching the
    message against a table of sixteen substrings, with a fallback -- "feature workflow
    invariant rejected the operation" -- that named neither the invariant nor the operation.
    A table cannot be kept in step with the raise sites it reads, and a raise site with no
    sentence worth writing is an invariant with no name worth having.

    ``classification`` defaults to ``PLATFORM_DEFECT`` because that is what an invariant
    rejection is: an operation this platform allowed to be attempted and then refused. A
    subclass that means something else says so.
    """

    def __init__(
        self,
        *args: object,
        diagnostic: str,
        classification: FeatureFailureClassification = (
            FeatureFailureClassification.PLATFORM_DEFECT
        ),
    ) -> None:
        """Record one platform-owned sentence and the classification it belongs to."""
        super().__init__(*args, classification=classification, diagnostics=(diagnostic,))


class RepairSupersededError(FeatureWorkflowError):
    """Refuse a stale repair while carrying the state that records it as stale.

    The refusal and the state change are one thing. Approving a repair works on a copy of the
    snapshot, so raising an ordinary error would discard the supersession along with it, and
    the same stale proposal would be offered again forever.
    """

    def __init__(self, *args: object, state: FeatureWorkflowSnapshot, diagnostic: str) -> None:
        """Carry the snapshot the caller must persist before reporting the refusal."""
        super().__init__(*args, diagnostic=diagnostic)
        self.state = state


class PartialPullRequestError(FeatureWorkflowError):
    """Retain already-created PR artifacts when a later coordinated publication fails."""

    def __init__(
        self,
        pull_requests: Sequence[PullRequestArtifact],
        failed_repository_ids: Sequence[str] = (),
    ) -> None:
        """Carry only safe persisted PR artifacts, never a provider exception message."""
        # The repositories are named because they are platform-owned identifiers the operator
        # has to act on: without them a half-published feature said only that publication was
        # partial, leaving a person to work out which repository still needed a pull request.
        failed = ", ".join(sorted(failed_repository_ids))
        super().__init__(
            "coordinated pull-request creation completed only partially",
            classification=FeatureFailureClassification.PUBLICATION_FAILURE,
            diagnostic=(
                "Pull-request creation did not succeed for every repository this feature "
                + (f"approved; it failed for: {failed}." if failed else "approved.")
            ),
        )
        self.pull_requests = list(pull_requests)
        self.failed_repository_ids = list(failed_repository_ids)


class FeatureProductManager(Protocol):
    """Create one technical PRD before repository workstreams can be planned."""

    async def create_technical_prd(
        self,
        *,
        feature_id: str,
        prd: PRDArtifact,
        design_snapshot: DesignSnapshotArtifact | None = None,
    ) -> TechnicalPRDArtifact:
        """Return a technical PRD with any human clarification requirements.

        The snapshot is passed rather than looked up because this port's implementations build
        their own agent state, and the one that matters built it from the PRD alone -- so the
        role whose entire job is deriving requirements *from* the design was the only role that
        never saw one. Optional, so a composition that cites nothing is unchanged.
        """

    async def reconcile_requirements(
        self,
        *,
        feature_id: str,
        technical_prd: TechnicalPRDArtifact,
        answers: Sequence[AnsweredClarification],
    ) -> RequirementReconciliation:
        """Restate the requirements a person's answers settled, and report what they cannot.

        On this port rather than the planner's because the PRD is the product manager's
        authorship. Read by capability: a composition whose product manager predates this
        reconciles nothing and says so in the log, which is not the same thing as a
        reconciliation that failed.
        """


@dataclass(frozen=True, slots=True)
class BlindPlanningRecord:
    """One repository reconnaissance tried to read and could not, with the sanitized reason.

    Only the fail-soft produces these. A composition with no reconnaissance capability plans
    without evidence *by design* and records nothing -- what this names is the live case
    where the evidence was asked for and lost, because that repository's workstream is the
    one most likely to fail and nobody watching AB-Feature-173 knew to expect it.
    """

    repository_id: str
    error_type: str
    reason: str


@dataclass(frozen=True, slots=True)
class ReconnaissanceReport:
    """What reconnaissance read, and which repositories it had to leave unread."""

    artifacts: list[RepositoryReconnaissanceArtifact]
    blind: list[BlindPlanningRecord] = dataclass_field(default_factory=list)


class RepositoryReconnaissance(Protocol):
    """Read every target checkout before the plan is allowed to assume anything about it."""

    async def inspect(
        self,
        *,
        feature: FeatureWorkflowSnapshot,
        repositories: Sequence[RepositorySpec],
        technical_prd: TechnicalPRDArtifact,
        credentials: RequestScopedCredentials,
    ) -> ReconnaissanceReport:
        """Return one grounded artifact per readable repository, and name the unreadable."""


class NullRepositoryReconnaissance:
    """Plan without checkout evidence, preserving the behaviour mock features already had.

    Deliberately not a fabricated artifact. A plausible-looking reconnaissance result that no
    checkout produced is the exact failure this pipeline stage exists to remove, so the mock
    path reports nothing and the planner is told it is planning blind. It reports no blind
    repositories either: nothing was attempted, so nothing failed.
    """

    async def inspect(
        self,
        *,
        feature: FeatureWorkflowSnapshot,
        repositories: Sequence[RepositorySpec],
        technical_prd: TechnicalPRDArtifact,
        credentials: RequestScopedCredentials,
    ) -> ReconnaissanceReport:
        """Report no evidence rather than inventing any."""
        del feature, repositories, technical_prd, credentials
        return ReconnaissanceReport(artifacts=[])


@dataclass(frozen=True, slots=True)
class ChildExecution:
    """One isolated workstream attempt and its immutable artifacts."""

    result: ChildWorkflowResultArtifact
    code_completion: CodeCompletionArtifact | None = None
    review: ReviewArtifact | None = None
    contract_change_request: ContractChangeRequestArtifact | None = None


class ChildWorkstreamExecutor(Protocol):
    """Run one repository's engineer/reviewer loop without owning parent coordination."""

    async def run(
        self,
        *,
        feature: FeatureWorkflowSnapshot,
        repository: RepositorySpec,
        workstream: RepositoryWorkstreamPlan,
        child: ChildWorkflowReference,
        technical_prd: TechnicalPRDArtifact,
        contract: IntegrationContractArtifact,
        feedback: Sequence[str],
        credentials: RequestScopedCredentials,
    ) -> ChildExecution:
        """Return an isolated result; never mutate the parent contract."""


class CoordinatedPullRequestPublisher(Protocol):
    """Publish per-repository PRs only after the parent integration gate approves."""

    async def publish(
        self,
        *,
        feature: FeatureWorkflowSnapshot,
        repositories: Sequence[RepositorySpec],
        children: Sequence[ChildWorkflowReference],
        child_results: Sequence[ChildWorkflowResultArtifact],
        contract: IntegrationContractArtifact,
        integration_approved: bool = True,
        unreviewed_commit_shas: Mapping[str, str] | None = None,
    ) -> list[PullRequestArtifact]:
        """Create one PR per repository and cross-link the completed set.

        ``unreviewed_commit_shas`` names the repositories a person chose to publish despite
        their review rejecting them, each mapped to the commit the publication action has just
        made for it. Absent for every automatic caller, which is why it defaults to nothing:
        the approved-only refusal below is the rule, and this is its one named exception.
        """

    def find_pull_request(
        self,
        repository: str,
        *,
        source_branch: str,
        target_branch: str,
        title: str,
    ) -> Any:
        """Fetch the provider record required before a completion claim is durable."""

    async def comment_on_pull_request(
        self, repository: str, pull_request_number: int, body: str
    ) -> None:
        """Post one comment -- the cross-link a superseded pull request carries."""

    async def close_pull_request(self, repository: str, pull_request_number: int) -> None:
        """Close a superseded pull request; already closed or merged is a no-op."""


class FeatureWorkflowRunner(Protocol):
    """Execute a parent feature state without taking ownership of HTTP credentials."""

    async def start(
        self, state: FeatureWorkflowSnapshot, *, credentials: RequestScopedCredentials
    ) -> FeatureWorkflowSnapshot:
        """Run PM/planning/workstreams until a terminal or human-waiting state."""

    async def advance_one_step(
        self, state: FeatureWorkflowSnapshot, *, credentials: RequestScopedCredentials
    ) -> FeatureWorkflowSnapshot:
        """Advance this feature by exactly one step and return, whatever remains after it."""

    async def begin_resume(
        self, state: FeatureWorkflowSnapshot, *, answers: Sequence[ClarificationAnswer]
    ) -> FeatureWorkflowSnapshot:
        """Apply and refuse everything a resume decides, scheduling no work."""

    async def grant_and_run_one_retry(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repository_id: str,
        additional_attempts: int,
        requested_by: str,
        reason: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Record a granted retry and run the one repository it bought, then stop."""

    async def answer_design_conflict(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        conflict_id: str,
        verdict: str,
        decision: str,
        decided_by: str,
        additional_attempts: int,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Record which position holds in one design conflict and run that repository once."""

    async def resume(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        answers: Sequence[ClarificationAnswer],
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Resolve parent clarification answers and continue its durable workflow."""

    async def retry_workstream(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repository_id: str,
        additional_attempts: int,
        requested_by: str,
        reason: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Grant one stopped repository more attempts and run it again."""

    async def publish_feature(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        requested_by: str,
        reason: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Open the pull requests a person decided a feature that did not land should have."""

    async def approve_contract_change(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        request_id: str,
        updated_contract: IntegrationContractArtifact,
        resolution: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Replace the approved contract only through an explicit human-approved revision."""

    async def reject_contract_change(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        request_id: str,
        resolution: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Record a human rejection and end safely for manual resolution."""

    async def approve_repository_repair(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repair_id: str,
        actor_id: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Apply one approved repository repair and give its workstream another attempt."""

    async def reject_repository_repair(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repair_id: str,
        actor_id: str,
        reason: str,
    ) -> FeatureWorkflowSnapshot:
        """Record that a person declined a repair, leaving the repository stopped."""


def _derived_requirement(prd: PRDArtifact) -> Requirement:
    """The one requirement a mock run needs when the submission carried none.

    It restates the submitted problem rather than inventing scope of its own: everything
    downstream traces to a requirement id, so mock mode needs exactly one to exist, and the
    honest content for it is what the author actually wrote.
    """
    return Requirement(
        requirement_id="REQ-1",
        description=prd.problem_statement,
        priority="must",
        acceptance_criteria=[f"The submitted problem is addressed: {prd.title}"],
        dependencies=[],
    )


class DeterministicFeatureProductManager:
    """Turn a structured PRD into a deterministic mock technical PRD without network calls."""

    async def create_technical_prd(
        self,
        *,
        feature_id: str,
        prd: PRDArtifact,
        design_snapshot: DesignSnapshotArtifact | None = None,
    ) -> TechnicalPRDArtifact:
        """Preserve submitted requirements and require no artificial mock clarification."""
        del design_snapshot  # A mock derives nothing from a design; the port's shape is shared.
        return create_artifact(
            TechnicalPRDArtifact,
            workflow_id=feature_id,
            artifact_id=ARTIFACT_FILENAMES["technical_prd"],
            producer="product_manager",
            payload={
                "title": f"{prd.title} technical plan",
                "solution_summary": (
                    "Implement the approved feature across isolated repository workstreams."
                ),
                # A submission may carry no structured requirements at all -- deriving them
                # is what the real product-manager agent is for. This is that agent's mock
                # stand-in, so it derives the one requirement the rest of the workflow needs
                # to have something to trace against, from the problem statement it was
                # given. Everything it produces is already marked `"mode": "mock"`.
                "functional_requirements": [
                    item.model_dump(mode="python")
                    for item in (prd.requirements or [_derived_requirement(prd)])
                ],
                "non_functional_requirements": [],
                "data_requirements": [],
                "integration_requirements": ["Use the approved integration contract."],
                "security_requirements": ["Do not persist provider credentials."],
                "assumptions": ["Mock mode does not contact repository providers."],
                "unresolved_questions": [],
            },
            metadata={"source_artifact_ids": [prd.artifact_id], "mode": "mock"},
        )

    async def reconcile_requirements(
        self,
        *,
        feature_id: str,
        technical_prd: TechnicalPRDArtifact,
        answers: Sequence[AnsweredClarification],
    ) -> RequirementReconciliation:
        """Change nothing, and deliberately not by having nothing to change it with.

        Reconciliation is semantic work: deciding that "use compensating deletes" contradicts
        "rolls back all app records" is the reading a model does. A deterministic stand-in
        that guessed at it from phrase lists would be the fabricated planning input this
        composition exists to avoid, so it settles nothing and reports nothing unsettled --
        the honest answer for a run that reaches no provider. It implements the method rather
        than omitting it so that a *live* composition missing the capability is the only thing
        the absence can mean.
        """
        del feature_id, technical_prd, answers
        return RequirementReconciliation()


class MockChildWorkstreamExecutor:
    """Produce deterministic reviewed child results without cloning or invoking providers."""

    async def run(
        self,
        *,
        feature: FeatureWorkflowSnapshot,
        repository: RepositorySpec,
        workstream: RepositoryWorkstreamPlan,
        child: ChildWorkflowReference,
        technical_prd: TechnicalPRDArtifact,
        contract: IntegrationContractArtifact,
        feedback: Sequence[str],
        credentials: RequestScopedCredentials,
    ) -> ChildExecution:
        """Simulate the engineer/reviewer boundary while proving parent isolation in tests."""
        del technical_prd, credentials
        code_completion = create_artifact(
            CodeCompletionArtifact,
            workflow_id=child.child_workflow_id,
            artifact_id=ARTIFACT_FILENAMES["code_completion"],
            producer="engineer",
            payload={
                "completion_status": "completed",
                "summary": f"Mock implementation completed for {repository.name}.",
                "file_changes": [],
                "validation_results": [],
                "test_coverage_percent": None,
                "remaining_work": [],
                "commit_sha": f"mock-{_slug(child.child_workflow_id)}",
            },
            metadata={
                "contract_artifact_id": contract.artifact_id,
                "contract_version": contract.contract_version,
                "feedback": list(feedback),
            },
        )
        review = create_artifact(
            ReviewArtifact,
            workflow_id=child.child_workflow_id,
            artifact_id=ARTIFACT_FILENAMES["review"],
            producer="reviewer",
            payload={
                "verdict": "approved",
                "summary": "Mock repository review approved the assigned contract sections.",
                "requirement_checks": [],
                "findings": [],
                "architecture_assessment": "Implementation is scoped to the shared contract.",
                "security_assessment": "No credentials are present in child state.",
                "test_coverage_assessment": "Mock mode has no repository validation commands.",
            },
            metadata={"source_artifact_ids": [code_completion.artifact_id, contract.artifact_id]},
        )
        result = _child_result(
            feature=feature,
            repository=repository,
            workstream=workstream,
            child=child,
            code_completion=code_completion,
            review=review,
            status="approved",
            blocking_issues=[],
        )
        return ChildExecution(result=result, code_completion=code_completion, review=review)


class GitHubPullRequestPublisher:
    """Create draft-by-default PRs and cross-link the final set through GitHub comments."""

    def __init__(
        self,
        *,
        github_service: Any | None = None,
        draft_pull_requests: bool = True,
        cancellation_token: CancellationToken | None = None,
        max_publication_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        """Inject a mock by default, preserving no-network behavior for standard test runs."""
        self._github_service: Any = github_service or MockGitHubService()
        self._draft_pull_requests = draft_pull_requests
        self._cancellation_token = cancellation_token
        if max_publication_attempts < 1:
            msg = "max_publication_attempts must be at least 1"
            raise ValueError(msg)
        if retry_backoff_seconds < 0:
            msg = "retry_backoff_seconds must not be negative"
            raise ValueError(msg)
        self._max_publication_attempts = max_publication_attempts
        self._retry_backoff_seconds = retry_backoff_seconds

    def find_pull_request(
        self,
        repository: str,
        *,
        source_branch: str,
        target_branch: str,
        title: str,
    ) -> Any:
        """Fetch a published pull request back so completion can be verified, not assumed."""
        return self._github_service.find_pull_request(
            repository,
            source_branch=source_branch,
            target_branch=target_branch,
            title=title,
        )

    async def publish(
        self,
        *,
        feature: FeatureWorkflowSnapshot,
        repositories: Sequence[RepositorySpec],
        children: Sequence[ChildWorkflowReference],
        child_results: Sequence[ChildWorkflowResultArtifact],
        contract: IntegrationContractArtifact,
        integration_approved: bool = True,
        unreviewed_commit_shas: Mapping[str, str] | None = None,
    ) -> list[PullRequestArtifact]:
        """Publish every repository on its own merits, then cross-link whatever reached GitHub."""
        child_by_repository = {item.repository_id: item for item in children}
        result_by_repository = {item.repository_id: item for item in child_results}
        unreviewed = dict(unreviewed_commit_shas or {})
        artifacts: list[PullRequestArtifact] = []
        failed_repository_ids: list[str] = []
        for repository in repositories:
            await self._raise_if_cancelled()
            try:
                artifacts.append(
                    await self._publish_one(
                        feature=feature,
                        repository=repository,
                        child=child_by_repository[repository.repository_id],
                        result=result_by_repository[repository.repository_id],
                        contract=contract,
                        integration_approved=integration_approved,
                        unreviewed_commit_sha=unreviewed.get(repository.repository_id),
                    )
                )
            except CancellationRequested:
                raise
            except Exception as error:  # noqa: BLE001 - one repository's fault is its own
                # Aborting the loop here denied every later repository an attempt it would
                # have passed: one transient provider fault on the first repository left the
                # rest unpublished despite pushed branches and approved reviews. Each
                # repository is created independently, so each one fails independently.
                _LOGGER.error(
                    "pull_request_publication_failed_for_repository",
                    feature_id=feature.feature_id,
                    stage="pull_request_publication",
                    agent="github",
                    repository_id=repository.repository_id,
                    error_type=type(error).__name__,
                )
                failed_repository_ids.append(repository.repository_id)
        await self._cross_link(artifacts)
        if failed_repository_ids:
            raise PartialPullRequestError(artifacts, failed_repository_ids)
        return artifacts

    async def _publish_one(
        self,
        *,
        feature: FeatureWorkflowSnapshot,
        repository: RepositorySpec,
        child: ChildWorkflowReference,
        result: ChildWorkflowResultArtifact,
        contract: IntegrationContractArtifact,
        integration_approved: bool = True,
        unreviewed_commit_sha: str | None = None,
    ) -> PullRequestArtifact:
        """Turn one repository into exactly one pull-request artifact.

        ``unreviewed_commit_sha`` relaxes the approved-only refusal below, and nothing else
        does. Its presence *is* the discriminator: a rejected attempt commits nothing, so the
        only way this repository has a revision to open is that the publication action has
        just committed one, and there is no representable state where the work is unreviewed
        and no such commit exists. A separate boolean beside the sha could disagree with it;
        one parameter cannot.
        """
        unreviewed = unreviewed_commit_sha is not None
        if not unreviewed and (result.status != "approved" or not result.pull_request_readiness):
            msg = f"repository is not ready for a pull request: {repository.repository_id}"
            raise FeatureWorkflowError(
                msg,
                diagnostic=(
                    "A repository that has not passed review and been marked pull-request ready "
                    "cannot be published."
                ),
            )
        commit_sha = (
            unreviewed_commit_sha if unreviewed_commit_sha is not None else _commit_sha(result)
        )
        repository_name = _repository_name(str(repository.repository_url))
        title = _pull_request_title(feature, repository, unreviewed=unreviewed)
        body = _pull_request_body(
            feature,
            child,
            result,
            contract,
            draft=self._draft_pull_requests,
            integration_approved=integration_approved,
        )
        details: Any = await self._create_or_adopt(
            repository_name,
            title=title,
            body=body,
            source_branch=child.branch_name,
            target_branch=repository.default_branch,
            commit_sha=commit_sha,
        )
        await self._raise_if_cancelled()
        return create_artifact(
            PullRequestArtifact,
            workflow_id=feature.feature_id,
            # The base id belongs to the original run; a revision's pull request is a new
            # immutable record beside it, never a rewrite of it.
            artifact_id=(
                f"008_pull_request.{repository.repository_id}.json"
                if not feature.revision
                else f"008_pull_request.{repository.repository_id}.v{feature.revision + 1}.json"
            ),
            producer="github",
            payload={
                "repository": details.repository,
                "pull_request_number": details.number,
                "url": details.url,
                "title": details.title,
                "body": body,
                "source_branch": details.source_branch,
                "target_branch": details.target_branch,
                "commit_sha": commit_sha,
                "labels": [
                    "automated",
                    "ai-feature",
                    "draft" if self._draft_pull_requests else "ready-for-review",
                    # Beside the title marker rather than instead of it: a reviewer scanning a
                    # list of pull requests reads titles, and a label alone is missable.
                    *(["review-rejected"] if unreviewed else []),
                ],
                "reviewers": [],
                "state": "open",
            },
            metadata={
                **_revision_metadata(feature),
                "feature_id": feature.feature_id,
                "child_workflow_id": child.child_workflow_id,
                "contract_version": contract.contract_version,
                "draft_requested": self._draft_pull_requests,
                "published_without_review_approval": unreviewed,
            },
        )

    async def _create_or_adopt(
        self,
        repository_name: str,
        *,
        title: str,
        body: str,
        source_branch: str,
        target_branch: str,
        commit_sha: str,
    ) -> Any:
        """Retry a failed create, but only after checking whether it already reached GitHub."""
        last_error: Exception | None = None
        for attempt in range(1, self._max_publication_attempts + 1):
            await self._raise_if_cancelled()
            try:
                return await _await_if_needed(
                    self._github_service.create_pull_request(
                        repository_name,
                        title=title,
                        body=body,
                        source_branch=source_branch,
                        target_branch=target_branch,
                        draft=self._draft_pull_requests,
                        expected_head_sha=commit_sha,
                    )
                )
            except CancellationRequested:
                raise
            except Exception as error:  # noqa: BLE001 - retried below once reconciled
                last_error = error
                # A create that raised may still have reached GitHub, and a provider that
                # already holds the pull request rejects the next create for the same branch
                # forever. Look it up before retrying so a retry adopts the existing pull
                # request instead of turning a transient fault into a permanent failure.
                existing = await self._existing_pull_request(
                    repository_name,
                    source_branch=source_branch,
                    target_branch=target_branch,
                    title=title,
                )
                if existing is not None:
                    return existing
                if attempt < self._max_publication_attempts:
                    _LOGGER.warning(
                        "pull_request_creation_retried",
                        repository=repository_name,
                        attempt=attempt,
                        error_type=type(error).__name__,
                    )
                    await asyncio.sleep(self._retry_backoff_seconds * attempt)
        if last_error is not None:
            raise last_error
        msg = "pull-request creation produced no result"
        raise FeatureWorkflowError(
            msg,
            diagnostic=("Pull-request creation returned no result and reported no error."),
        )

    async def _existing_pull_request(
        self, repository_name: str, *, source_branch: str, target_branch: str, title: str
    ) -> Any:
        """Look a pull request back up, treating an unusable lookup as 'not found'."""
        finder = getattr(self._github_service, "find_pull_request", None)
        if finder is None:
            return None
        try:
            return await _await_if_needed(
                finder(
                    repository_name,
                    source_branch=source_branch,
                    target_branch=target_branch,
                    title=title,
                )
            )
        except CancellationRequested:
            raise
        except Exception:  # noqa: BLE001 - a failed lookup only means 'cannot confirm'
            _LOGGER.warning("pull_request_lookup_failed", repository=repository_name)
            return None

    async def comment_on_pull_request(
        self, repository: str, pull_request_number: int, body: str
    ) -> None:
        """Post one comment through whichever service this publisher was built with."""
        await _await_if_needed(
            self._github_service.add_comment(repository, pull_request_number, body)
        )

    async def close_pull_request(self, repository: str, pull_request_number: int) -> None:
        """Close one superseded pull request; already closed or merged is a no-op."""
        close = getattr(self._github_service, "close_pull_request", None)
        if close is None:
            msg = "this GitHub service cannot close pull requests"
            raise FeatureWorkflowError(
                msg,
                diagnostic=(
                    "The configured GitHub service has no close capability, so the "
                    "superseded pull request has to be closed by hand."
                ),
            )
        await _await_if_needed(close(repository, pull_request_number))

    async def _cross_link(self, artifacts: Sequence[PullRequestArtifact]) -> None:
        """Point each published pull request at the others without gating on the comment.

        The body is built from the pull requests this feature actually published -- each
        one's own repository and its own provider-returned URL. No repository name, owner,
        or URL shape is written here, so a comment stays correct for any checkout the
        platform is pointed at.
        """
        if not artifacts:
            return
        related = "\n".join(
            [
                "Related pull requests:",
                *[f"- {item.repository}: {item.url}" for item in artifacts],
            ]
        )
        for artifact in artifacts:
            await self._raise_if_cancelled()
            try:
                await _await_if_needed(
                    self._github_service.add_comment(
                        artifact.repository, artifact.pull_request_number, related
                    )
                )
            except CancellationRequested:
                raise
            except Exception as error:  # noqa: BLE001 - a convenience comment is not a gate
                # The cross-link is a convenience for a human reader. Every pull request
                # already exists and the integration review already approved them, so
                # failing the feature here reported a finished piece of work as failed
                # and left an operator hunting for a problem that was not there.
                #
                # Not gating it is right; saying nothing about it was not. Every one of these
                # comments has failed -- ten for ten across the platform's lifetime,
                # including every feature that completed -- and the record of each was a
                # warning naming no cause and a journal row reading "failed before a
                # confirmed result". A hundred-per-cent failure stayed invisible for want of
                # the error type this line already had in hand.
                #
                # The adapter now classifies the refusal, so `diagnostics` carries the HTTP
                # status, the endpoint, and the likely cause -- one warning per skipped
                # comment, no stack trace, and enough to act on. The status is what points at
                # the remedy: a 403 here is very probably a credential that cannot write
                # pull-request comments, which is a permission grant and not a code change.
                # Only the type and this repository's own safe diagnostics are logged; a
                # provider message is not quoted.
                _LOGGER.warning(
                    "pull_request_cross_link_failed",
                    repository=artifact.repository,
                    pull_request_number=artifact.pull_request_number,
                    error_type=type(error).__name__,
                    diagnostics=list(_causal_diagnostics(error)),
                )

    async def _raise_if_cancelled(self) -> None:
        """Prevent a cancellation request from starting another PR mutation."""
        if self._cancellation_token is not None:
            await self._cancellation_token.raise_if_cancelled()


class FeatureWorkflowOrchestrator:
    """Run one PM/planner pass, fan out child workstreams, then gate coordinated PRs."""

    def __init__(
        self,
        *,
        product_manager: FeatureProductManager | None = None,
        # Turns the citations on the PRD into one design snapshot. Defaults to the
        # deterministic resolver for `DeterministicFeatureProductManager`'s reason: mock mode
        # runs with no adapters at all, and the step's place in the sequence has to be
        # exercisable without Figma existing.
        design_resolver: DesignReferenceResolver | None = None,
        reconnaissance: RepositoryReconnaissance | None = None,
        planner: FeaturePlanner | None = None,
        child_executor: ChildWorkstreamExecutor | None = None,
        integration_reviewer: IntegrationReviewerAgent | None = None,
        pull_request_publisher: CoordinatedPullRequestPublisher | None = None,
        workspace_root: Path | str = "/workspaces",
        cancellation_token: CancellationToken | None = None,
        checkpoint_writer: FeatureCheckpointWriter | None = None,
        # Writes `repository_planned_blind` to the feature timeline the moment the fail-soft
        # fires, so an operator deciding whether to trust a plan does not need `docker logs`.
        # Optional for the same reason the checkpoint writer is: deterministic compositions
        # have no store to write to, and the snapshot field still records the fact.
        event_writer: FeatureEventWriter | None = None,
        # Records the pre-coding model calls -- product manager, grounding, planner -- as
        # journal rows, so forty-one minutes inside one hung call is a row with a widening
        # heartbeat gap instead of nothing (AB-Feature-173). A factory rather than an
        # executor because an operation's scope names the feature it belongs to, and one
        # orchestrator instance can serve more than one feature. Optional because the
        # deterministic composition reaches no provider; None journals nothing.
        operation_executor_factory: Callable[[FeatureWorkflowSnapshot], ExternalOperationExecutor]
        | None = None,
        max_parallel_workstreams: int = 4,
        # Selects which configured model role executes an attempt this orchestrator has
        # already decided may run. Never asked whether an attempt may run: that verdict is
        # `decide_child_retry`'s alone, and the router is consulted only after it says yes.
        # The default routes and records without naming a model, which is what the
        # deterministic composition needs -- it reaches no provider at all.
        model_router: ModelRouter | None = None,
        # A backstop on how long one repository's workstream may go on being given attempts.
        # Zero disables it, which is what every deterministic composition uses: a ceiling is
        # only meaningful against wall-clock time a real run spends.
        repository_runtime_limit_seconds: float = 0.0,
        # Builds the Git boundary for one repository, for the one path that commits and
        # pushes outside a child attempt: publishing a workstream whose checks all passed and
        # whose review rejected it. Per repository because the adapter is bound to a default
        # branch. None for every composition that never reaches that path, which is every
        # deterministic one -- the publication refuses with a sentence rather than pretending.
        git_service_factory: Callable[[RepositorySpec, ChildWorkflowReference], Any] | None = None,
    ) -> None:
        """Use deterministic mock dependencies unless live runtime composition injects real ones."""
        self._product_manager = product_manager or DeterministicFeatureProductManager()
        self._design_resolver: DesignReferenceResolver = (
            design_resolver or DeterministicDesignResolver()
        )
        self._reconnaissance = reconnaissance or NullRepositoryReconnaissance()
        self._planner = planner or DeterministicFeaturePlanner()
        self._child_executor = child_executor or MockChildWorkstreamExecutor()
        self._integration_reviewer = integration_reviewer or IntegrationReviewerAgent()
        self._pull_request_publisher = pull_request_publisher or GitHubPullRequestPublisher()
        self._workspace_root = Path(workspace_root)
        self._cancellation_token = cancellation_token
        self._checkpoint_writer = checkpoint_writer
        self._event_writer = event_writer
        self._operation_executor_factory = operation_executor_factory
        self._max_parallel_workstreams = max_parallel_workstreams
        self._model_router = model_router or ModelRouter()
        self._repository_runtime_limit_seconds = repository_runtime_limit_seconds
        self._git_service_factory = git_service_factory

    async def start(
        self, state: FeatureWorkflowSnapshot, *, credentials: RequestScopedCredentials
    ) -> FeatureWorkflowSnapshot:
        """Advance a fresh feature until it needs somebody or has nothing left to do."""
        state = state.model_copy(deep=True)
        _record_executor_identity(state)
        await self._raise_if_cancelled(state)
        # Refused here rather than inside the first step, so a feature with no requirement is
        # rejected before a status is written or a provider is reached.
        _require_artifact(state.artifacts, PRDArtifact)
        return await self._run_to_rest(state, credentials=credentials)

    async def advance_one_step(
        self, state: FeatureWorkflowSnapshot, *, credentials: RequestScopedCredentials
    ) -> FeatureWorkflowSnapshot:
        """Do the one thing this feature's persisted state calls for next, then return.

        The unit a queue claim runs. `start`, `resume` and every other entry point below are
        loops over this method, so there is one implementation of what a step does and two
        drivers over it: a claim, which runs one and re-queues, and an in-process caller,
        which runs them until the feature comes to rest.

        Which step is `next_step`'s answer and nothing else's. There is deliberately no
        `feature_is_at_rest` guard here: whether the platform may act on a resting feature is
        the claim's question, not the step's, and the two answers differ in exactly the case
        that matters. A feature waiting for a person is at rest, and a resume of it is a
        person asking -- so a guard here would refuse the request it was made to carry out,
        which is how a provider timeout that stopped before it could write anything would
        never be resumable at all. The queue makes that decision in `_claim_disposition`,
        which drops a `start` claim on a resting feature and runs a `resume`.
        """
        state = state.model_copy(deep=True)
        _record_executor_identity(state)
        await self._raise_if_cancelled(state)
        decision = next_step(state)
        if decision.step is None:
            return self._settle_without_a_step(state, decision)
        return await self._run_decided_step(state, decision.step, credentials=credentials)

    async def _run_to_rest(
        self, state: FeatureWorkflowSnapshot, *, credentials: RequestScopedCredentials
    ) -> FeatureWorkflowSnapshot:
        """Run step after step until this feature is waiting on somebody or is finished.

        The in-process driver. Nothing durable distinguishes a feature advanced by this loop
        from one advanced by a sequence of claims: both ask `next_step` between every step and
        both stop at `feature_is_at_rest`. What differs is only that this one keeps the
        process, which is what an API path that must answer with a finished feature needs.
        """
        while not feature_is_at_rest(state):
            decision = next_step(state)
            if decision.step is None:
                return self._settle_without_a_step(state, decision)
            state = await self._run_decided_step(state, decision.step, credentials=credentials)
        return state

    async def _run_decided_step(
        self,
        state: FeatureWorkflowSnapshot,
        step: FeatureStep,
        *,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Run one already-decided step, containing a cancellation that arrives during it."""
        try:
            state = await self._run_step(state, step, credentials=credentials)
        except CancellationRequested:
            # Every step already stops at its own boundaries; this is the one that arrives
            # between them. The feature keeps whatever the interrupted step made durable and
            # the cancellation records its own cleanup requirements.
            await self._mark_cancelled(state)
            return state
        state.updated_at = datetime.now(UTC)
        return state

    async def _run_step(
        self,
        state: FeatureWorkflowSnapshot,
        step: FeatureStep,
        *,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Dispatch one named step to the method that owns it."""
        if step is FeatureStep.RESOLVE_DESIGN:
            return await self._step_resolve_design(state)
        if step is FeatureStep.ANALYZE_PRD:
            return await self._step_analyze_prd(state)
        if step is FeatureStep.PLAN:
            return await self._step_plan(state, credentials=credentials)
        if step is FeatureStep.EXECUTE_WORKSTREAMS:
            return await self._step_execute_workstreams(state, credentials=credentials)
        if step is FeatureStep.INTEGRATION_REVIEW:
            return await self._step_integration_review(state)
        return await self._step_publish(state)

    def _settle_without_a_step(
        self, state: FeatureWorkflowSnapshot, decision: StepDecision
    ) -> FeatureWorkflowSnapshot:
        """Record what a feature with nothing to do is actually waiting for.

        Only one of `next_step`'s answers implies a status the feature does not already
        carry: a run that reaches its open questions from a checkpoint is waiting for a
        person, and saying so is what puts the recovery action in front of them. The rest --
        terminal, cancelling, waiting on a contract owner, finished -- already say what they
        are, and rewriting them would lose the distinction.
        """
        if decision.no_step_reason is NoStep.AWAITING_CLARIFICATION_ANSWERS:
            transition_feature(
                state,
                FeatureWorkflowStatus.WAITING_FOR_HUMAN,
                reason=decision.explanation,
                agent="human_clarification",
            )
        state.updated_at = datetime.now(UTC)
        return state

    async def resume(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        answers: Sequence[ClarificationAnswer],
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Resolve clarification or continue an interrupted feature from its safe checkpoint."""
        state = await self.begin_resume(state, answers=answers)
        return await self._run_to_rest(state, credentials=credentials)

    async def begin_resume(
        self, state: FeatureWorkflowSnapshot, *, answers: Sequence[ClarificationAnswer]
    ) -> FeatureWorkflowSnapshot:
        """Apply and refuse everything a resume decides, and schedule no work.

        Split out of `resume` so a queue claim can settle the request and then run exactly one
        step. Everything here is a property of the *request*: what it may not do, and what it
        carries. None of it reaches a provider or a repository, so a claim that dies between
        this and its step loses nothing a person has to supply twice.
        """
        state = state.model_copy(deep=True)
        _record_executor_identity(state)
        await self._raise_if_cancelled(state)
        if state.status != FeatureWorkflowStatus.WAITING_FOR_HUMAN:
            validate_clarification_answers(state, answers)
            # The timeline retains the previous stop. The active snapshot describes the
            # current attempt, so a successful recovery must not keep displaying a stale
            # terminal diagnosis, and a second failure must be free to record its own cause.
            state.failure_summary = None
            require_resumable_feature(state)
            return state
        if state.failure_summary is not None and state.failure_summary.retryable and not answers:
            # A transient provider failure uses WAITING_FOR_HUMAN so the recovery action is
            # available, but it has no clarification questions. Treat an empty-answer resume
            # as checkpoint recovery rather than requiring a Technical PRD that the timed-out
            # call never produced.
            validate_clarification_answers(state, answers)
            state.failure_summary = None
            require_resumable_feature(state)
            return state
        if state.clarification_rounds >= state.max_clarification_rounds:
            transition_feature(
                state,
                FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
                reason=(
                    f"This feature used all {state.max_clarification_rounds} clarification "
                    "rounds it is allowed without its open questions being resolved."
                ),
                agent="human_clarification",
            )
            state.updated_at = datetime.now(UTC)
            return ensure_feature_failure_summary(
                state,
                stage=FailureStage.HUMAN_CLARIFICATION,
                classification=FeatureFailureClassification.CLARIFICATION_UNRESOLVED,
                diagnostics=[
                    f"This feature used all {state.max_clarification_rounds} of its "
                    "clarification rounds and its open questions are still unanswered.",
                    "Start a fresh feature with the ambiguity resolved in the requirement "
                    "itself, rather than answering another round here.",
                ],
            )
        return await self.apply_clarification_answers(state, answers=answers)

    async def apply_clarification_answers(
        self, state: FeatureWorkflowSnapshot, *, answers: Sequence[ClarificationAnswer]
    ) -> FeatureWorkflowSnapshot:
        """Write accepted answers into the Technical PRD, and schedule no work.

        Split out of `resume` so a queue claim can apply the answers it carries and then run
        exactly one step. They are a durable human decision rather than input to the planner
        call that used to follow them immediately, and this is the write that says so: it
        reaches no provider, so nothing between here and the next step can put the same
        questions back in front of their author or ask for a second submission.
        """
        state = state.model_copy(deep=True)
        _record_executor_identity(state)
        await self._raise_if_cancelled(state)
        validate_clarification_answers(state, answers)
        technical_prd = _require_artifact(state.artifacts, TechnicalPRDArtifact)
        resolved = technical_prd.model_copy(
            update={
                "unresolved_questions": [],
                "metadata": {
                    **technical_prd.metadata,
                    "clarification_answers": {item.question_id: item.answer for item in answers},
                },
            }
        )
        _replace_artifact(state, technical_prd, resolved)
        state.clarification_rounds += 1
        state.failure_summary = None
        # The ask this flag described is over: the human answered it.
        state.clarification_grounding_failed = False
        transition_feature(
            state,
            FeatureWorkflowStatus.PLANNING,
            reason="A person answered this feature's open questions, so planning resumes.",
            agent="planner",
        )
        await self._checkpoint(state, WorkflowCheckpointBoundary.BEFORE_CLONE)
        return state

    async def retry_workstream(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repository_id: str,
        additional_attempts: int,
        requested_by: str,
        reason: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Grant one stopped repository more attempts and run it again.

        This is the one path that may raise a retry budget, and it exists because the platform
        must not raise its own. An exhausted budget is a finding about the work; ordinary
        resume deliberately refuses to reset it, because repeating coding and validation on
        the strength of no new information spends money to reach the same place. A person who
        has read the blocking issues -- and possibly repaired the repository -- knows something
        the platform does not, and that knowledge is what the grant records.

        The grant buys attempts, not an outcome. The re-run goes through the same child loop,
        the same review and the same publication as any other attempt, and ``decide_child_retry``
        still decides afterwards whether yet another one may follow.
        """
        state = await self.grant_and_run_one_retry(
            state,
            repository_id=repository_id,
            additional_attempts=additional_attempts,
            requested_by=requested_by,
            reason=reason,
            credentials=credentials,
        )
        return await self._run_to_rest(state, credentials=credentials)

    async def publish_feature(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        requested_by: str,
        reason: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Open pull requests for a feature that did not land, because a person said to.

        Automatic publication belongs to the feature that fully landed. Everything else waits
        here, and what this opens is strictly more than the automatic path ever could: work
        that passed review and has not been published, and work whose every required check
        passed but whose review said no -- each labelled in its title and its body for what it
        is.

        No ordinary step follows it. The feature failed and still has, so this deliberately
        does not run to rest: a publication is not a resume, and nothing about it makes the
        repository that never landed any more finished than it was.
        """
        del credentials
        state = state.model_copy(deep=True)
        _record_executor_identity(state)
        await self._raise_if_cancelled(state)
        require_publishable_feature(state)
        return await self._publish_awaiting_repositories(
            state, requested_by=requested_by, reason=reason
        )

    async def grant_and_run_one_retry(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repository_id: str,
        additional_attempts: int,
        requested_by: str,
        reason: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Record the grant and run the one repository it bought, then stop.

        The grant's own step, split out of `retry_workstream` so a queue claim runs it and
        returns. It has to stay targeted: a grant names a repository, and an untargeted pass
        would also pick up a sibling that merely still had attempts left -- work nobody asked
        for, and which the person granting this retry did not agree to pay for. Every step
        after this one is the ordinary untargeted kind, chosen by `next_step`.
        """
        state = state.model_copy(deep=True)
        _record_executor_identity(state)
        await self._raise_if_cancelled(state)
        require_retryable_workstream(
            state, repository_id=repository_id, additional_attempts=additional_attempts
        )
        # The same clearing symmetry resume has had since its own comment said why: the
        # timeline retains the previous stop, and the active snapshot must describe the
        # current attempt. `ensure_feature_failure_summary` is deliberately write-once, so a
        # summary left here outlives the stop it described -- run 199 went terminal on a
        # design conflict 87 minutes and six attempts after its retry grant, and its
        # persisted story still read "the model provider did not answer".
        state.failure_summary = None
        child = state.child_workflows[repository_id]

        state.child_workflows[repository_id] = child.model_copy(
            update={
                "granted_extra_attempts": child.granted_extra_attempts + additional_attempts,
                # The refusal described the previous stop. Leaving it in place would make the
                # escalation for a still-unresolved workstream quote a decision that has since
                # been overridden.
                "retry_refusal_reason": None,
                # Same clearing symmetry for a partition wave: the plan describes an attempt
                # that has ended, and the granted attempt must start unpartitioned, or the
                # resume door would replay a stale wave instead of the run the grant bought.
                "pending_context_partition": None,
                "retry_grants": [
                    *child.retry_grants,
                    {
                        "granted_at": datetime.now(UTC).isoformat(),
                        "granted_by": requested_by,
                        "attempts": additional_attempts,
                        "reason": reason,
                        "stopped_because": child.retry_refusal_reason or "",
                    },
                ],
            }
        )
        return await self._run_one_targeted_attempt(
            state, repository_id=repository_id, credentials=credentials
        )

    async def answer_design_conflict(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        conflict_id: str,
        verdict: str,
        decision: str,
        decided_by: str,
        additional_attempts: int,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Record which position holds in one design conflict, then run the repository once.

        The stop this answers is unchanged: 49- Part B still ends the workstream synchronously
        the moment a satisfied demand is demanded again, and still spends nothing doing it. What
        this adds is the other way out of that stop. Cancelling the feature and buying attempts
        in the hope the argument comes out differently were the only two before, and neither
        answers the question the platform asked.

        The verdict is carried as an invariant rather than as advice. It goes into the artifact
        lineage, which is where ``resolved_issue_ledger`` and ``settled_questions`` read from,
        so the next attempt is handed the decision in the decider's own words and the recurrence
        check may no longer stop on that fingerprint. Both halves matter: without the first the
        engineer would reverse it again, and without the second the loop would stop before the
        engineer was ever called.

        Attempts are not granted here by default, and never implicitly. The stop spends nothing
        -- it is checked before the retry verdict -- so a workstream may reach it with budget
        left or with none, depending on which attempt the reversal was recognised on. Where
        budget remains, a verdict buys nothing and records no grant. Where it does not, the
        caller has to say how many attempts to grant and it is recorded in ``retry_grants``
        exactly as an operator's retry grant is. What is deliberately absent is the middle
        option: topping the budget up on the caller's behalf would make an answered question
        look like a refund of the attempts the stop declined to spend.
        """
        state = state.model_copy(deep=True)
        _record_executor_identity(state)
        await self._raise_if_cancelled(state)
        conflict = find_design_conflict(state.artifacts, conflict_id)
        # The request's own shape first, before anything about the feature. A caller who sent a
        # verdict this platform does not have should be told that, not told about a budget --
        # and each of these is a refusal rather than a validation error, because letting one
        # reach the artifact would mark the feature failed for somebody's typo.
        if verdict not in VERDICTS:
            msg = f"{verdict!r} is not a design verdict; expected one of {list(VERDICTS)}"
            raise FeatureWorkflowError(
                msg,
                diagnostic=("A design decision must say which of the two positions holds."),
            )
        if not decision.strip():
            msg = "a design decision must record the reasoning in the decider's words"
            raise FeatureWorkflowError(
                msg,
                diagnostic=(
                    "A design decision must be recorded with its reasoning: the next attempt is "
                    "handed it as an instruction, and an unexplained verdict is not one."
                ),
            )
        if not decided_by.strip():
            msg = "a design decision must record who made it"
            raise FeatureWorkflowError(
                msg,
                diagnostic=("A design decision must record who made it."),
            )
        require_answerable_design_conflict(
            state, conflict=conflict, additional_attempts=additional_attempts
        )
        child = state.child_workflows[conflict.repository_id]
        # Recorded before the attempt, and this is the ordering the whole feature rests on: the
        # attempt reads the verdict out of the lineage, so writing it afterwards would run the
        # attempt the decision authorised without the decision in it.
        revise_design_conflict(
            state,
            conflict,
            **answered_payload(
                verdict=verdict,
                decision=decision,
                decided_by=decided_by,
                decided_at=datetime.now(UTC),
            ),
        )
        state.child_workflows[conflict.repository_id] = child.model_copy(
            update={
                "granted_extra_attempts": child.granted_extra_attempts + additional_attempts,
                # The refusal quoted the stop this answer settles.
                "retry_refusal_reason": None,
                # What the next attempt inherits, minus what this decision has answered. The
                # stop wrote its question into `blocking_issues` because that list is what
                # reaches the console -- and the same list is what a resumed attempt is handed
                # as outstanding work, so leaving it would ask the engineer to fix the
                # platform's question to an operator. An overruled demand goes with it: an
                # attempt told both "implement this" and "a person decided this must not be
                # implemented" is the contradiction the verdict exists to remove.
                "blocking_issues": [
                    item
                    for item in child.blocking_issues
                    if item != conflict.question
                    and not (
                        verdict == "removal_holds"
                        # Two exact keys, both of real recorded sentences: the conflict's
                        # binding fingerprint, and the demand as currently worded. For a
                        # reversal they are the same key; for a recurring demand (77-) the
                        # binding key is the earliest wording while the inherited issue is
                        # the current one. Nothing looser than byte-exact prose keys (73-).
                        and fingerprint_for_text(item)
                        in {conflict.fingerprint, fingerprint_for_text(conflict.demand)}
                    )
                ],
                "retry_grants": (
                    [
                        *child.retry_grants,
                        {
                            "granted_at": datetime.now(UTC).isoformat(),
                            "granted_by": decided_by,
                            "attempts": additional_attempts,
                            "reason": (
                                f"design conflict {conflict_id} decided ({verdict}): {decision}"
                            ),
                            "stopped_because": child.retry_refusal_reason or "",
                        },
                    ]
                    # No grant, no grant record. A row saying nobody bought anything reads in an
                    # audit as an override that happened, and the commonest answer to a design
                    # conflict buys nothing -- the attempts were never spent.
                    if additional_attempts
                    else list(child.retry_grants)
                ),
            }
        )
        return await self._run_one_targeted_attempt(
            state, repository_id=conflict.repository_id, credentials=credentials
        )

    async def _run_one_targeted_attempt(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repository_id: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Run the one repository a person acted on, and nothing else.

        Shared by the retry grant and the design verdict because it is the same step: an
        untargeted pass would also pick up a sibling that merely still had attempts left --
        work nobody asked for, and which the person who acted did not agree to pay for. Every
        step after this one is the ordinary untargeted kind, chosen by `next_step`.
        """
        technical_prd = _require_artifact(state.artifacts, TechnicalPRDArtifact)
        contract = _require_artifact(state.artifacts, IntegrationContractArtifact)
        plan = _require_artifact(state.artifacts, RepositoryExecutionPlanArtifact)
        # A grant buys attempts; it does not decide what the attempt should do. Where
        # integration review has already named a fix for this repository, that instruction is
        # what the engineer needs -- the grant's own reason is a sentence a person typed for
        # the audit record. Carrying only the latter sent AB-Feature-111's console in blind:
        # it re-read the requirement, found the work already implemented and published, and
        # changed nothing, then failed as though it were not converging.
        outstanding = _outstanding_integration_fixes(state, repository_id)
        state = await self._step_execute_workstreams(
            state,
            technical_prd=technical_prd,
            contract=contract,
            plan=plan,
            target_workstreams={repository_id},
            targeted_attempt=TargetedAttempt.OPERATOR_RETRY,
            credentials=credentials,
            feedback={repository_id: outstanding} if outstanding else None,
        )
        state.updated_at = datetime.now(UTC)
        return state

    async def approve_repository_repair(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repair_id: str,
        actor_id: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Apply one approved repair, then let the repository try again.

        The repair is applied by the attempt itself rather than by a separate execution
        path: the granted attempt re-clones the repository, and the approved commands run
        there through the same journal that makes every other side effect idempotent. That
        is why approving twice cannot install twice -- the second approval finds the repair
        already succeeded and never reaches an attempt at all.
        """
        state = state.model_copy(deep=True)
        _record_executor_identity(state)
        await self._raise_if_cancelled(state)
        repair = find_repair(state.artifacts, repair_id)
        child = state.child_workflows.get(repair.repository_id)

        if repair.status in {"succeeded", "executing"}:
            # Already carried out, or being carried out. Returning the state unchanged is the
            # idempotent answer; re-running the commands is the thing this must never do.
            state.updated_at = datetime.now(UTC)
            return state
        if repair.status != "proposed":
            msg = f"repair {repair_id} is {repair.status} and cannot be approved"
            raise FeatureWorkflowError(
                msg,
                diagnostic=("A repository repair can only be approved while it is still proposed."),
            )
        if child is None:
            msg = f"repair {repair_id} names a repository this feature does not have"
            raise FeatureWorkflowError(
                msg,
                diagnostic=("A repository repair named a repository this feature does not have."),
            )
        if repair_is_stale(repair, current_revision=child.current_revision):
            # The checkout moved after the diagnosis was written, so the commands were chosen
            # against code that is no longer there. Superseding it forces a fresh look rather
            # than applying a fix to a problem that may already be gone.
            #
            # The supersession travels with the refusal rather than being written and thrown
            # away. This method works on a copy, so raising a plain error discarded the very
            # state change that stops the same stale repair being offered again -- every
            # later approval would meet the same wall with nothing recorded.
            revise_repository_repair(
                state,
                repair,
                status="superseded",
                execution_result=(
                    "The repository changed after this repair was proposed, so it was not "
                    "applied. The next attempt will diagnose the current checkout."
                ),
            )
            state.updated_at = datetime.now(UTC)
            msg = (
                f"repair {repair_id} was written against revision "
                f"{repair.proposed_at_revision} and the repository is now at "
                f"{child.current_revision}; it has been superseded and must be re-evaluated"
            )
            raise RepairSupersededError(
                msg,
                state=state,
                diagnostic=(
                    "The repository moved on from the revision this repair was written "
                    "against, so the repair was superseded and has to be re-evaluated."
                ),
            )

        revise_repository_repair(
            state,
            repair,
            status="executing",
            approved_by=actor_id,
            approved_at=datetime.now(UTC),
        )
        try:
            state = await self.retry_workstream(
                state,
                repository_id=repair.repository_id,
                additional_attempts=1,
                requested_by=actor_id,
                reason=f"approved repository repair {repair_id}: {repair.proposed_repair}",
                credentials=credentials,
            )
        except Exception:
            current = find_repair(state.artifacts, repair_id)
            revise_repository_repair(
                state,
                current,
                status="failed",
                execution_result=(
                    "The repository could not be retried after this repair was approved."
                ),
            )
            raise

        # Whether the repair worked is judged by what the repository can do afterwards, not
        # by whether commands exited zero: the point of the repair was to let this repository
        # run its own checks, so the evidence is that it got past the setup block.
        retried = state.child_workflows.get(repair.repository_id)
        still_blocked = bool(retried.blocking_setup_issues) if retried is not None else True
        current = find_repair(state.artifacts, repair_id)
        revise_repository_repair(
            state,
            current,
            status="failed" if still_blocked else "succeeded",
            execution_result=(
                "The repository still cannot run its own checks after the repair."
                if still_blocked
                else "The repository ran its own checks after the repair was applied."
            ),
            resulting_revision=retried.current_revision if retried is not None else None,
        )
        state.updated_at = datetime.now(UTC)
        return state

    async def reject_repository_repair(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repair_id: str,
        actor_id: str,
        reason: str,
    ) -> FeatureWorkflowSnapshot:
        """Record that a person declined a repair, so the stop stays explained."""
        state = state.model_copy(deep=True)
        _record_executor_identity(state)
        repair = find_repair(state.artifacts, repair_id)
        if repair.status == "rejected":
            state.updated_at = datetime.now(UTC)
            return state
        if repair.status != "proposed":
            msg = f"repair {repair_id} is {repair.status} and cannot be rejected"
            raise FeatureWorkflowError(
                msg,
                diagnostic=("A repository repair can only be rejected while it is still proposed."),
            )
        if not reason.strip():
            msg = "rejecting a repair requires a reason"
            raise FeatureWorkflowError(
                msg,
                diagnostic=("Rejecting a repository repair requires a stated reason."),
            )
        revise_repository_repair(
            state,
            repair,
            status="rejected",
            rejected_by=actor_id,
            rejection_reason=reason,
        )
        state.updated_at = datetime.now(UTC)
        return state

    async def approve_contract_change(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        request_id: str,
        updated_contract: IntegrationContractArtifact,
        resolution: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Adopt a replacement contract and rerun every consumer against that exact revision."""
        state = state.model_copy(deep=True)
        _record_executor_identity(state)
        request = _contract_change_request(state, request_id)
        if request.status != "pending":
            msg = "contract change request is not pending"
            raise FeatureWorkflowError(
                msg,
                diagnostic=(
                    "The contract change request is not pending, so it cannot be decided again."
                ),
            )
        if state.contract_revision_cycles >= state.max_contract_revision_cycles:
            transition_feature(
                state,
                FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
                reason=(
                    f"This feature reached its limit of {state.max_contract_revision_cycles} "
                    "shared-contract revisions."
                ),
                agent="feature_workflow",
            )
            state.updated_at = datetime.now(UTC)
            return ensure_feature_failure_summary(
                state,
                stage=FailureStage.CONTRACT_REVISION,
                classification=(FeatureFailureClassification.CONTRACT_REVISION_LIMIT_REACHED),
                diagnostics=[
                    f"This feature used all {state.max_contract_revision_cycles} of its "
                    "shared-contract revisions and the repositories still disagree about it.",
                    "The contract itself is what has to change; another revision cycle here "
                    "would rerun every consumer against the same disagreement.",
                ],
            )
        current = _require_artifact(state.artifacts, IntegrationContractArtifact)
        if updated_contract.feature_id != state.feature_id or updated_contract.status != "approved":
            msg = "updated contract must be an approved contract for this feature"
            raise FeatureWorkflowError(
                msg,
                diagnostic=(
                    "A contract revision must be an approved contract belonging to this feature."
                ),
            )
        if _contract_version_key(updated_contract.contract_version) <= _contract_version_key(
            current.contract_version
        ):
            msg = "updated contract must use a higher contract_version"
            raise FeatureWorkflowError(
                msg,
                diagnostic=(
                    "A contract revision must carry a higher contract version than the one it "
                    "replaces."
                ),
            )
        unknown_owners = set(updated_contract.owning_workstreams) - {
            item.repository_id for item in state.repository_specs
        }
        if unknown_owners:
            msg = "updated contract cannot name an unknown repository workstream"
            raise FeatureWorkflowError(
                msg,
                diagnostic=(
                    "A contract revision named a repository workstream this feature does not have."
                ),
            )
        if _has_durable_child_commit(state):
            # An already-committed branch contains bytes reviewed against the superseded
            # contract. A delta-only rerun cannot prove that the whole branch satisfies the
            # replacement contract, and a no-op rerun has no new commit to bind. Requiring
            # fresh branches is safer than silently relabelling v1 history as v2-approved.
            _replace_artifact(
                state,
                request,
                request.model_copy(
                    update={
                        "status": "rejected",
                        "resolution": (
                            "Automated in-place revision is unsafe after a child commit; start "
                            "a new feature run on fresh branches."
                        ),
                        "new_contract_artifact_id": None,
                    }
                ),
            )
            transition_feature(
                state,
                FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
                reason=(
                    "A shared-contract revision was asked for after a child had already "
                    "committed, which needs a new run on fresh branches."
                ),
                agent="human_contract_owner",
            )
            state.updated_at = datetime.now(UTC)
            return ensure_feature_failure_summary(
                state,
                stage=FailureStage.CONTRACT_REVISION,
                classification=(
                    FeatureFailureClassification.CONTRACT_REVISION_REQUIRES_FRESH_BRANCHES
                ),
                diagnostics=[
                    "A shared-contract revision after a child commit requires a new feature "
                    "run on fresh branches.",
                    "The branches already hold bytes reviewed against the contract this "
                    "revision replaces, and no rerun can prove the whole branch satisfies "
                    "the replacement.",
                ],
            )
        approved_request = request.model_copy(
            update={
                "status": "approved",
                "resolution": resolution,
                "new_contract_artifact_id": updated_contract.artifact_id,
            }
        )
        _replace_artifact(state, request, approved_request)
        _append_artifacts(state, [updated_contract])
        state.contract_revision_cycles += 1
        plan = _require_artifact(state.artifacts, RepositoryExecutionPlanArtifact)
        revised_plan = plan.model_copy(
            update={
                # Count-based like the contract's own id: equal to the old
                # `contract_revision_cycles + 1` numbering on every never-revised feature,
                # and collision-free once a feature revision has appended plans of its own.
                "artifact_id": _versioned_artifact_id(
                    state,
                    RepositoryExecutionPlanArtifact,
                    "010_repository_execution_plan.json",
                ),
                "contract_artifact_id": updated_contract.artifact_id,
                "metadata": {
                    **plan.metadata,
                    **_revision_metadata(state),
                    "supersedes": plan.artifact_id,
                    "contract_version": updated_contract.contract_version,
                },
            }
        )
        _append_artifacts(state, [revised_plan])
        # A shared contract revision invalidates every earlier child approval, including
        # consumers not named by the repository that requested the change. Without a formal
        # compatibility proof, selectively retaining a sibling would publish code reviewed
        # against a superseded schema.
        affected = {item.workstream_id for item in revised_plan.workstreams}
        state = await self._step_execute_workstreams(
            state,
            technical_prd=_require_artifact(state.artifacts, TechnicalPRDArtifact),
            contract=updated_contract,
            plan=revised_plan,
            target_workstreams=affected,
            credentials=credentials,
        )
        state.updated_at = datetime.now(UTC)
        return await self._run_to_rest(state, credentials=credentials)

    async def reject_contract_change(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        request_id: str,
        resolution: str,
        credentials: RequestScopedCredentials,
    ) -> FeatureWorkflowSnapshot:
        """Record an explicit rejection and require a human rather than guessing a safe fix."""
        del credentials
        state = state.model_copy(deep=True)
        _record_executor_identity(state)
        request = _contract_change_request(state, request_id)
        _replace_artifact(
            state,
            request,
            request.model_copy(
                update={
                    "status": "rejected",
                    "resolution": resolution,
                }
            ),
        )
        transition_feature(
            state,
            FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
            reason=(
                "A person declined the shared-contract change this feature asked for, so the "
                "workstreams that depend on it cannot continue."
            ),
            agent="human_contract_owner",
        )
        state.updated_at = datetime.now(UTC)
        return ensure_feature_failure_summary(
            state,
            stage=FailureStage.CONTRACT_REVISION,
            classification=FeatureFailureClassification.CONTRACT_REVISION_REJECTED,
            diagnostics=[
                "A person declined the shared-contract change this feature asked for.",
                "The repositories that asked for it cannot proceed against the contract as "
                "it stands; the requirement or the contract has to change before a new run.",
            ],
        )

    async def _despite_provider_faults[T](
        self,
        state: FeatureWorkflowSnapshot,
        *,
        stage: str,
        call: Callable[[], Awaitable[T]],
    ) -> T:
        """Run one feature-level agent call again when the provider, not the work, failed.

        The child workstream loop has granted this allowance since -072; the stages that run
        before any repository does never had it, so the same dropped connection that costs a
        workstream one attempt cost the whole feature its planning stage. Nothing about the
        two situations differs from the provider's side, and a fault here says exactly as
        little about the requirement as it does there.

        Bounded, and deliberately smaller than the child loop's: see the constant. Exhausting
        it re-raises the original fault so the failure is classified, summarised, and queued
        for another attempt by the paths that already know how -- this retries a call, it does
        not decide what a feature does when the provider is genuinely down.

        Both clocks are kept, the way the child loop has kept them since 40- Part F: wall is
        everything a stage's calls took, and the fault clock is what classified provider
        faults and their backoff consumed inside that. AB-Feature-173 spent 37 of its 41
        dark minutes inside one ReadTimeout and nothing recorded the split; the deliberately
        deferred pre-coding ceiling can only ever be justified by these measurements
        existing. No ceiling reads them in this task -- they record.
        """
        faults = 0
        stage_started = time.monotonic()
        fault_seconds = 0.0

        def record_clocks() -> None:
            wall = time.monotonic() - stage_started
            state.planning_wall_seconds = (state.planning_wall_seconds or 0.0) + wall
            state.planning_provider_fault_seconds = (
                state.planning_provider_fault_seconds or 0.0
            ) + fault_seconds

        while True:
            await self._raise_if_cancelled(state)
            attempt_started = time.monotonic()
            try:
                value = await call()
            except _UNCONFIRMED_OPERATION_FAULTS:
                # Ordered first, as in the child loop. An effect that could not be confirmed
                # is a question for a person, never something to attempt again.
                record_clocks()
                raise
            except _INFRASTRUCTURE_FAULTS as error:
                if not is_transient_provider_fault(error):
                    # A deterministic provider answer returns identically on every retry;
                    # AB-Feature-172 spent nine calls proving that on one 400.
                    record_clocks()
                    raise
                faults += 1
                # The stall is the provider's, not the requirement's -- the same
                # accounting the child loop applies.
                fault_seconds += time.monotonic() - attempt_started
                if faults > _ALLOWED_FEATURE_STAGE_INFRASTRUCTURE_FAULTS:
                    record_clocks()
                    raise
                backoff = _fault_backoff_seconds(error, fault_count=faults)
                fault_seconds += backoff
                _LOGGER.warning(
                    "feature_stage_provider_fault_retried",
                    feature_id=state.feature_id,
                    stage=stage,
                    fault_count=faults,
                    fault=_fault_label(error),
                    backoff_seconds=round(backoff, 3),
                )
                await self._sleep_between_faults(state, backoff)
            else:
                record_clocks()
                return value

    async def _journaled_planning_call[T](
        self,
        state: FeatureWorkflowSnapshot,
        *,
        operation_type: ExternalOperationType,
        stage: str,
        call: Callable[[], Awaitable[T]],
        safe_input: dict[str, Any] | None = None,
    ) -> T:
        """Record one pre-coding model call as a journal row, and decide nothing from it.

        Observability, not recovery semantics. The stages that make these calls already
        resume from their artifacts (`_existing_reconnaissance`, `next_step`), so a row here
        must never become a second reconciliation input: every call writes a fresh row --
        the idempotency input carries a per-call nonce -- and no result is ever replayed
        from one. What the row buys is the part AB-Feature-173 had nothing of: a
        `started_at` that keeps its distance from `heartbeat_at` while a provider call
        hangs, and a sanitized error code when it fails.

        The provider-fault allowance lives here, once, and not per call site. The nonce
        means a journal-level budget can never retry these operations -- a second `run()`
        is a new row -- so the retry is in-place, around the journaled call, and each try
        the allowance buys writes its own row: the record then shows the retries rather
        than hiding them. Runs 175-180 are why the placement matters: the grounding call
        had no wrap of its own, so one two-minute provider blip took all six runs' first
        model call, and every one asked its operator questions the platform was one retry
        away from answering itself. Wall and fault clocks are kept by the loop exactly as
        before -- `dec7191`'s rule that provider-fault seconds are excluded from runtime
        ceilings holds for the retries too.
        """
        if self._operation_executor_factory is None:
            return await self._despite_provider_faults(state, stage=stage, call=call)
        executor = self._operation_executor_factory(state)

        async def journaled_call() -> T:
            async def action() -> tuple[T, OperationResult]:
                # Reset-then-read, inside the action's own task: the row says which model
                # answered *this* call, read at the only moment that is provably true. A call
                # that never reached a provider -- a deterministic composition, a mock run --
                # reads nothing and the row honestly records no model. See
                # `tools.llm_call_record` for why this cannot cross concurrent calls.
                reset_llm_call_record()
                value = await call()
                return value, OperationResult(payload=last_llm_call_payload())

            journaled = await executor.run(
                operation_type=operation_type,
                logical_step=stage,
                safe_input={"stage": stage, **(safe_input or {})},
                idempotency_input={"stage": stage, "call_nonce": uuid4().hex},
                action=action,
                max_attempts=1,
                attempt_metadata={"stage": stage},
            )
            return cast(T, journaled.value)

        return await self._despite_provider_faults(state, stage=stage, call=journaled_call)

    async def _record_planned_blind(
        self,
        state: FeatureWorkflowSnapshot,
        blind: BlindPlanningRecord,
        *,
        occurrence: str,
    ) -> None:
        """Make one blind-planned repository a durable state, not a log line.

        Three writes, so each reader has it where that reader already looks: the snapshot
        field (which `_initialize_children` copies onto the workstream record), the feature
        timeline (where an operator watches), and the log (where the fail-soft always said
        it, kept because the deployment's log line is what §1.2 was reconstructed from).
        The fail-soft itself is untouched -- blind planning remains better than no feature,
        and recording it twice is the designed answer to it happening twice.
        """
        state.planned_blind_repositories[blind.repository_id] = blind.reason
        _LOGGER.warning(
            "repository_planned_blind",
            feature_id=state.feature_id,
            repository_id=blind.repository_id,
            error_type=blind.error_type,
            reason=blind.reason,
            occurrence=occurrence,
        )
        if self._event_writer is None:
            return
        await self._event_writer(
            state.feature_id,
            "repository_planned_blind",
            {
                "repository_id": blind.repository_id,
                "error_type": blind.error_type,
                "reason": blind.reason,
                "occurrence": occurrence,
            },
        )

    async def _inspect_repositories(
        self, state: FeatureWorkflowSnapshot, *, credentials: RequestScopedCredentials
    ) -> TechnicalPRDArtifact:
        """Read every checkout once, and record what it contradicts as a question for a human.

        Everything the plan freezes -- requirement scope, acceptance criteria, expected source
        areas, the contract -- was previously decided from a PRD and a repository URL, so a
        requirement resting on a convention the repository does not have became a workstream
        no attempt could finish. This is the last point at which that answer can change.
        """
        technical_prd = _require_artifact(state.artifacts, TechnicalPRDArtifact)
        if _existing_reconnaissance(state):
            return technical_prd
        transition_feature(
            state,
            FeatureWorkflowStatus.INSPECTING_REPOSITORIES,
            reason="Every repository this feature names is being read before anything is planned.",
            agent="repository_recon",
        )
        await self._raise_if_cancelled(state)
        report = await self._despite_provider_faults(
            state,
            stage="repository_recon",
            call=lambda: self._reconnaissance.inspect(
                feature=state,
                repositories=state.repository_specs,
                technical_prd=technical_prd,
                credentials=credentials,
            ),
        )
        reconnaissance = report.artifacts
        for blind in report.blind:
            await self._record_planned_blind(state, blind, occurrence="reconnaissance")
        _append_artifacts(state, reconnaissance)
        await self._checkpoint(state, WorkflowCheckpointBoundary.BEFORE_CLONE)
        asked = _asked_question_ids(technical_prd)
        questions = [
            item
            for item in [
                *_premise_questions(reconnaissance),
                *_unverifiable_criteria_questions(technical_prd),
            ]
            if item.question_id not in asked
        ]
        # The product manager wrote its own questions before any repository had been read, so
        # it asked about existing behaviour and conventions it had no way to know. The
        # checkouts have now been read; answering what they settle is the difference between
        # asking somebody a question and asking them to go and find what was just recorded.
        grounded = await self._ground_open_questions(
            state,
            reconnaissance,
            [*technical_prd.unresolved_questions, *questions],
        )
        if not questions and grounded == technical_prd.unresolved_questions:
            return technical_prd
        existing = len(technical_prd.unresolved_questions)
        revised = technical_prd.model_copy(
            update={
                "unresolved_questions": grounded,
                "metadata": {
                    **technical_prd.metadata,
                    "reconnaissance_question_ids": [item.question_id for item in questions],
                    "repository_grounded_answer_ids": [
                        item.question_id
                        for item in grounded[:existing]
                        if item.suggested_answer.strip()
                    ],
                },
            }
        )
        _replace_artifact(state, technical_prd, revised)
        return _require_artifact(state.artifacts, TechnicalPRDArtifact)

    async def _ground_open_questions(
        self,
        state: FeatureWorkflowSnapshot,
        reconnaissance: Sequence[RepositoryReconnaissanceArtifact],
        questions: Sequence[ClarificationQuestion],
    ) -> list[ClarificationQuestion]:
        """Attach the answers the checkouts already give, and change nothing else.

        Deliberately best-effort. This runs between a feature being planned and a person being
        asked, and it improves the question rather than deciding anything -- so a runner
        without the capability, or a model call that fails, costs the suggestions and leaves
        every question exactly as it was.

        Criteria questions are excluded on the way in. They ask an author to restate an
        acceptance criterion no repository change could demonstrate, which is a decision only
        they can make; a repository has nothing to say about it.
        """
        suggest = getattr(self._reconnaissance, "suggest_answers", None)
        if suggest is None or not reconnaissance:
            return list(questions)
        answerable = [item for item in questions if not item.question_id.startswith("criteria-")]
        if not answerable:
            return list(questions)
        # Reset per attempt: the flag describes this ask, not the feature's history. A
        # grounding run that succeeds after an earlier failure asks the human as itself.
        state.clarification_grounding_failed = False
        try:
            grounded = await self._journaled_planning_call(
                state,
                operation_type=ExternalOperationType.RUN_CLARIFICATION_GROUNDING,
                stage="clarification_grounding",
                safe_input={"question_count": len(answerable)},
                call=lambda: suggest(
                    feature_id=state.feature_id,
                    questions=answerable,
                    reconnaissance=reconnaissance,
                ),
            )
        except CancellationRequested:
            raise
        except Exception:  # noqa: BLE001 - a suggestion is an improvement, never a gate
            # Durable as well as logged: falling back to the human is the designed
            # behaviour, and an operator who knows the platform *tried* to self-answer
            # reads the questions differently. The read model serializes this.
            state.clarification_grounding_failed = True
            _LOGGER.warning(
                "clarification_grounding_failed",
                feature_id=state.feature_id,
                stage="repository_reconnaissance",
                agent="repository_recon",
            )
            return list(questions)
        by_id = {item.question_id: item for item in grounded}
        return [by_id.get(item.question_id, item) for item in questions]

    async def _reconcile_answered_requirements(
        self, state: FeatureWorkflowSnapshot, technical_prd: TechnicalPRDArtifact
    ) -> TechnicalPRDArtifact:
        """Rewrite the requirement text a person's answers settled, before anything is planned.

        The answers reached the PRD's metadata and the plan; the requirement descriptions,
        dependencies and acceptance criteria did not. `feature_planner` scopes tasks and
        acceptance criterion ids from `requirement.acceptance_criteria`, and the reviewer
        validates its requirement checks against a scope built from the same text -- so a
        requirement still describing what an answer overruled is a second specification, and
        every judge downstream enforces it. AB-Feature-200 ran that way: a PRD demanding
        transactional rollback, a plan compensating without transactions, an engineer that
        built both, and four attempts rejected for the hybrid before the no-meaningful-change
        guard stopped the workstream.

        It runs here rather than inside `apply_clarification_answers` on purpose, and the
        difference is durability, not taste. That method reaches no provider by design: the
        answers are persisted, and their revision checkpointed, *before* any call that can
        fail -- which is what lets a provider fault after accepted answers recover as an
        empty-answer resume instead of asking their author to type them again. Reconciling
        inside it would put a model call between a person's submission and its persistence.
        Here, the answers are already durable, this is the first call of the step that plans
        against them, and a failure is a planning-stage failure a resume retries.

        Charged, journaled and retried exactly as the other pre-coding calls: the fault
        allowance and the planning clocks belong to `_journaled_planning_call`, so this is one
        more caller of it and nothing bespoke. It borrows the product manager's operation type
        because it is that agent's call, and its own `logical_step` is what tells the two apart
        in the journal.

        A contradiction the call could not settle is asked, not planned around: the question
        goes into `unresolved_questions` and `_step_plan`'s existing gate returns the feature
        to a person. That spends a clarification round and therefore inherits the existing
        `max_clarification_rounds` terminal bound -- no new cap, no new authority, and the
        platform refusing only to *guess*.
        """
        answers = _unreconciled_answers(state.artifacts, technical_prd)
        if not answers:
            return technical_prd
        reconcile = getattr(self._product_manager, "reconcile_requirements", None)
        if reconcile is None:
            # Not a silent skip: the deterministic composition implements this and answers
            # "nothing to change", so the only thing an absence can mean is a composition
            # that was assembled without the capability, and that is worth a line.
            _LOGGER.warning(
                "requirement_reconciliation_unavailable",
                feature_id=state.feature_id,
                stage="requirement_reconciliation",
                answer_count=len(answers),
            )
            return technical_prd
        reconciliation = await self._journaled_planning_call(
            state,
            operation_type=ExternalOperationType.RUN_PRODUCT_MANAGER,
            stage="requirement_reconciliation",
            safe_input={"answer_count": len(answers)},
            call=lambda: reconcile(
                feature_id=state.feature_id, technical_prd=technical_prd, answers=answers
            ),
        )
        questions = _new_contradiction_questions(technical_prd, reconciliation, answers)
        if reconciliation.changes_nothing or not (reconciliation.rewrites or questions):
            # A pure confirmation, or a contradiction this feature has already asked about.
            # No revision either way, so the history does not gain an artifact that differs
            # from its predecessor in nothing but a timestamp.
            return technical_prd
        revised = technical_prd.model_copy(
            update={
                "functional_requirements": reconciled_requirements(
                    technical_prd.functional_requirements, reconciliation.rewrites
                ),
                "non_functional_requirements": reconciled_requirements(
                    technical_prd.non_functional_requirements, reconciliation.rewrites
                ),
                # Every rewrite the call *did* settle is applied alongside the question, so a
                # second round reconciles from the settled text rather than re-deciding it.
                # `_step_plan`'s existing gate reads this list and stops the feature: the
                # question travels through the machinery every other clarification uses, and
                # the round it spends is bounded by the same `max_clarification_rounds`.
                "unresolved_questions": [*technical_prd.unresolved_questions, *questions],
                "metadata": {
                    **technical_prd.metadata,
                    # requirement_id -> the answers that drove its rewrite. The human's words
                    # stay verbatim in `clarification_answers`; this is the trace from a
                    # changed acceptance criterion back to the sentence they wrote.
                    "requirement_reconciliation": reconciliation_change_list(
                        reconciliation.rewrites
                    ),
                    "reconciled_answer_ids": sorted(
                        _reconciled_answer_ids(technical_prd)
                        | {answer.question_id for answer in answers}
                    ),
                },
            }
        )
        _replace_artifact(state, technical_prd, revised)
        return _require_artifact(state.artifacts, TechnicalPRDArtifact)

    async def _step_resolve_design(self, state: FeatureWorkflowSnapshot) -> FeatureWorkflowSnapshot:
        """Resolve the cited designs into one snapshot, and stop at that boundary.

        Reached only by a feature that cited a design. It resumes from its artifact like every
        other pre-coding stage, which is what keeps its journal row a record and never a
        recovery input: re-running it after a crash costs one fetch, and it never re-runs once
        the snapshot exists because that artifact is what `next_step` reads to know it is done.

        Safety rule 4 lives in the artifact id. A second resolution is a new *revision*, never
        an edit of the first -- so the snapshot attempt 1 was built against and the review that
        judged it stay readable forever, and a design edited mid-feature cannot silently change
        the work order.

        No status of its own. `analyzing_prd` is what this is: the design is part of the
        request being read, and inventing a lifecycle state for it would add one to every
        client's vocabulary for the sake of a step most features never take.
        """
        prd = _require_artifact(state.artifacts, PRDArtifact)
        transition_feature(
            state,
            FeatureWorkflowStatus.ANALYZING_PRD,
            reason="The designs this feature cites are being resolved into a snapshot.",
            agent="design_resolver",
        )
        existing = sum(isinstance(item, DesignSnapshotArtifact) for item in state.artifacts)
        artifact_id = design_snapshot_artifact_id(existing)
        # The provider-fault allowance is inside `_journaled_planning_call`, once for every
        # pre-coding call; a second wrap here would square the budget. A refusal the person has
        # to fix is not in `_INFRASTRUCTURE_FAULTS` at all, so it propagates on the first try
        # rather than spending retries on an answer that will not change.
        snapshot = await self._journaled_planning_call(
            state,
            operation_type=ExternalOperationType.FETCH_DESIGN_REFERENCE,
            stage="design_snapshot",
            call=lambda: self._design_resolver.resolve(
                feature_id=state.feature_id,
                references=prd.design_references,
                artifact_id=artifact_id,
            ),
            # File keys and frame ids only. A citation's URL is the person's and the file key
            # is enough to identify what was read; nothing here is a credential.
            safe_input={
                "file_keys": sorted({item.file_key for item in prd.design_references}),
                "citations": len(prd.design_references),
            },
        )
        _append_artifacts(state, [snapshot])
        await self._checkpoint(state, WorkflowCheckpointBoundary.BEFORE_CODING)
        return state

    async def _resolve_design_detail(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repository: RepositorySpec,
        workstream: RepositoryWorkstreamPlan,
    ) -> DesignDetailArtifact | None:
        """Resolve one repository's assigned frames at build fidelity, once per workstream.

        Returns the artifact and appends it to the feature, which is how the child executor
        finds it -- the same route the snapshot itself travels. `None` for the two cases that
        must be byte-identical to a feature with no design at all: no snapshot, and a
        workstream whose plan assigns no frame.

        **Resolved once, and the artifact is the pin.** A workstream's second attempt reuses
        the artifact its first attempt resolved, matched on this repository and on the snapshot
        revision it was resolved against. Keying on the artifact rather than on
        `retry_count == 0` is deliberate: a resume after a crash re-enters this loop with the
        counter it had, and an attempt-counter gate would then read the design again and could
        hand attempt 2 a different work order than attempt 1 built against.

        **A provider fault degrades rather than kills.** There is no fault allowance at this
        seam -- an exception raised here leaves `_run_one_child` before its loop, and the
        fan-out records the repository as failed having spent none of its retries, which is
        exactly the -072 shape. So a transient fault buys a small local allowance and then, if
        the provider is still not answering, the assigned frames are recorded unreachable and
        the workstream builds knowing it was not shown them. The prompts already carry the
        "you were not shown these, do not attest and do not claim a violation" rule.

        Deliberately *not* wrapped in `_despite_provider_faults`: its fault counter is per call
        and would be safe, but it also writes `planning_wall_seconds` and
        `planning_provider_fault_seconds`, and this is not planning. Charging coding-time
        seconds to the pre-coding clocks would corrupt the only measurements the deferred
        pre-coding ceiling could ever be justified by.
        """
        snapshot = _latest_artifact(state.artifacts, DesignSnapshotArtifact)
        if snapshot is None:
            return None
        node_ids = [item for item in dict.fromkeys(workstream.design_nodes) if item]
        if not node_ids:
            return None
        existing = [
            item
            for item in state.artifacts
            if isinstance(item, DesignDetailArtifact)
            and item.repository_id == repository.repository_id
            and item.snapshot_artifact_id == snapshot.artifact_id
        ]
        if existing:
            return existing[-1]
        prd = _require_artifact(state.artifacts, PRDArtifact)
        artifact_id = design_detail_artifact_id(
            repository.repository_id,
            sum(
                isinstance(item, DesignDetailArtifact)
                and item.repository_id == repository.repository_id
                for item in state.artifacts
            ),
        )

        async def resolve_once() -> DesignDetailArtifact:
            return await self._design_resolver.resolve_detail(
                feature_id=state.feature_id,
                repository_id=repository.repository_id,
                workstream_id=workstream.workstream_id,
                snapshot_artifact_id=snapshot.artifact_id,
                references=prd.design_references,
                node_ids=node_ids,
                artifact_id=artifact_id,
            )

        faults = 0
        while True:
            try:
                detail = await self._journaled_design_detail_call(
                    state,
                    repository_id=repository.repository_id,
                    node_ids=node_ids,
                    call=resolve_once,
                )
            except _UNCONFIRMED_OPERATION_FAULTS:
                # A design read has no effect to be unconfirmed about, so this cannot come
                # from the resolution itself -- it came from the journal. Ordered first for
                # the reason every other loop orders it first, and never absorbed.
                raise
            except _INFRASTRUCTURE_FAULTS as error:
                faults += 1
                if not is_transient_provider_fault(error) or (
                    faults > _ALLOWED_DESIGN_DETAIL_INFRASTRUCTURE_FAULTS
                ):
                    detail = self._design_detail_unreachable(
                        state,
                        repository=repository,
                        workstream=workstream,
                        snapshot=snapshot,
                        node_ids=node_ids,
                        artifact_id=artifact_id,
                        error=error,
                    )
                    break
                backoff = _fault_backoff_seconds(error, fault_count=faults)
                _LOGGER.warning(
                    "design_detail_provider_fault_retried",
                    feature_id=state.feature_id,
                    repository_id=repository.repository_id,
                    stage="design_detail",
                    fault_count=faults,
                    fault=_fault_label(error),
                    backoff_seconds=round(backoff, 3),
                )
                await self._sleep_between_faults(state, backoff)
            else:
                break
        _append_artifacts(state, [detail])
        await self._checkpoint(
            state, WorkflowCheckpointBoundary.BEFORE_CODING, repository.repository_id
        )
        return detail

    def _design_detail_unreachable(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repository: RepositorySpec,
        workstream: RepositoryWorkstreamPlan,
        snapshot: DesignSnapshotArtifact,
        node_ids: Sequence[str],
        artifact_id: str,
        error: BaseException,
    ) -> DesignDetailArtifact:
        """Record every assigned frame as unreachable, rather than ending the workstream.

        The honest degradation this seam needs. A workstream that cannot be shown its design
        can still be *told* it was not shown its design, and every prompt already renders
        `design_nodes_unreachable` with the rule that it must not attest to what it did not
        see. Ending the workstream instead would spend a repository on a provider blip.
        """
        _LOGGER.warning(
            "design_detail_unreachable",
            feature_id=state.feature_id,
            repository_id=repository.repository_id,
            stage="design_detail",
            error_type=type(error).__name__,
            frames=len(node_ids),
        )
        return unreachable_design_detail(
            feature_id=state.feature_id,
            repository_id=repository.repository_id,
            workstream_id=workstream.workstream_id,
            snapshot_artifact_id=snapshot.artifact_id,
            snapshot=snapshot,
            node_ids=node_ids,
            artifact_id=artifact_id,
        )

    async def _journaled_design_detail_call(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repository_id: str,
        node_ids: Sequence[str],
        call: Callable[[], Awaitable[DesignDetailArtifact]],
    ) -> DesignDetailArtifact:
        """Record one detail resolution as a journal row, deciding nothing from it.

        The shape reconnaissance uses: one row per resolution, `max_attempts=1`, and a nonce in
        the idempotency input so nothing is ever replayed from a row. Recovery is by the
        artifact, not by the journal -- which is what the reuse check above reads.

        Observability is the whole point. AB-Feature-227's design step failed three times and
        the journal recorded `{"stage": "design_snapshot"}` with no provider status, so "why
        did the design fail" was answerable only from a log line. The safe input here names the
        repository and how many frames were asked for; the file keys are already on the
        snapshot.
        """
        if self._operation_executor_factory is None:
            return await call()
        executor = self._operation_executor_factory(state)

        async def resolve_and_record() -> tuple[DesignDetailArtifact, OperationResult]:
            # No model answers a design read, so the row honestly records no model -- the same
            # thing `_journaled_planning_call` writes for a deterministic composition.
            return await call(), OperationResult()

        journaled = await executor.run(
            operation_type=ExternalOperationType.FETCH_DESIGN_REFERENCE,
            logical_step="design_detail",
            safe_input={
                "stage": "design_detail",
                "repository_id": repository_id,
                "frames": len(node_ids),
            },
            idempotency_input={"stage": "design_detail", "call_nonce": uuid4().hex},
            action=resolve_and_record,
            max_attempts=1,
            attempt_metadata={"stage": "design_detail", "repository_id": repository_id},
        )
        return cast("DesignDetailArtifact", journaled.value)

    async def _step_analyze_prd(self, state: FeatureWorkflowSnapshot) -> FeatureWorkflowSnapshot:
        """Turn the requirement into a technical PRD, and stop at that boundary.

        The first step. It reaches one provider and writes one artifact, so re-running it
        after a crash costs a single product-manager call -- and it never re-runs once the
        artifact exists, because that artifact is what `next_step` reads to know it is done.
        """
        prd = _require_artifact(state.artifacts, PRDArtifact)
        transition_feature(
            state,
            FeatureWorkflowStatus.ANALYZING_PRD,
            reason="The requirement is being turned into a technical PRD.",
            agent="product_manager",
        )
        # The provider-fault allowance is inside `_journaled_planning_call`, once for every
        # pre-coding model call; a second wrap here would square the budget.
        technical_prd = await self._journaled_planning_call(
            state,
            operation_type=ExternalOperationType.RUN_PRODUCT_MANAGER,
            stage="product_manager",
            call=lambda: self._product_manager.create_technical_prd(
                feature_id=state.feature_id,
                prd=prd,
                # Read from the same place the planner's snapshot is read from, one step
                # below. This step runs after `_resolve_design_snapshot`, so the artifact is
                # already on the state; `None` for the features that cited nothing.
                design_snapshot=_latest_artifact(state.artifacts, DesignSnapshotArtifact),
            ),
        )
        _append_artifacts(state, [technical_prd])
        await self._checkpoint(state, WorkflowCheckpointBoundary.BEFORE_CODING)
        return state

    async def _step_plan(
        self, state: FeatureWorkflowSnapshot, *, credentials: RequestScopedCredentials
    ) -> FeatureWorkflowSnapshot:
        """Read the checkouts, ask what they contradict, and otherwise freeze the plan.

        Reconnaissance belongs to this step rather than to a step of its own. The only durable
        evidence it leaves is its artifacts, and a composition without a reconnaissance
        capability produces none -- so "has it run?" would have to be answered from something
        written for the purpose, which is the second source of truth this whole decision
        exists to avoid. It re-reads nothing on a repeat: `_inspect_repositories` returns
        immediately once its artifacts exist, so a crash between the checkouts and the planner
        costs the planner call alone.

        The clarification gate is here, after the checkouts have been read, and not in the
        step before. Pausing on the product manager's questions first asked the human
        everything except the questions worth asking: the ones only a repository can raise.
        """
        technical_prd = await self._inspect_repositories(state, credentials=credentials)
        technical_prd = await self._reconcile_answered_requirements(state, technical_prd)
        if technical_prd.unresolved_questions:
            # Every question this feature has -- the product manager's, the ones the checkouts
            # raised, and the contradictions its own answers created -- is asked in one round,
            # before anything is planned against an answer. This is the write that makes
            # `next_step` say so on the next claim.
            transition_feature(
                state,
                FeatureWorkflowStatus.WAITING_FOR_HUMAN,
                reason=_open_questions_reason(technical_prd.unresolved_questions),
                agent="human_clarification",
            )
            return state
        reconnaissance = _existing_reconnaissance(state)
        transition_feature(
            state,
            FeatureWorkflowStatus.PLANNING,
            reason="Nothing is unresolved, so the contract and execution plan are being written.",
            agent="planner",
        )
        # As at the product-manager site: the fault allowance is the wrapper's, not this
        # call site's, so the planner is not double-budgeted.
        architecture, contract, plan = await self._journaled_planning_call(
            state,
            operation_type=ExternalOperationType.RUN_FEATURE_PLANNER,
            stage="feature_planner",
            call=lambda: self._planner.plan(
                feature_id=state.feature_id,
                technical_prd=technical_prd,
                repositories=state.repository_specs,
                reconnaissance=reconnaissance,
                # The whole design, because this is the role that decides which repository
                # each frame belongs to. `None` for every feature that cited nothing.
                design_snapshot=_latest_artifact(state.artifacts, DesignSnapshotArtifact),
            ),
        )
        if state.revision:
            # A revision's planning pass is the second producer of these three types, and the
            # planner mints fixed base ids. Re-identify each one before it joins the history
            # (count-based, composing with the contract-change path's `.v{N}` numbering) and
            # stamp which revision produced it, which is what the `next_step` gates read.
            architecture = architecture.model_copy(
                update={
                    "artifact_id": _versioned_artifact_id(
                        state, ArchitectureArtifact, architecture.artifact_id
                    ),
                    "metadata": {**architecture.metadata, **_revision_metadata(state)},
                }
            )
            contract = contract.model_copy(
                update={
                    "artifact_id": _versioned_artifact_id(
                        state, IntegrationContractArtifact, contract.artifact_id
                    ),
                    "metadata": {**contract.metadata, **_revision_metadata(state)},
                }
            )
            plan = plan.model_copy(
                update={
                    "artifact_id": _versioned_artifact_id(
                        state, RepositoryExecutionPlanArtifact, plan.artifact_id
                    ),
                    "contract_artifact_id": contract.artifact_id,
                    "metadata": {**plan.metadata, **_revision_metadata(state)},
                }
            )
        _append_artifacts(state, [architecture, contract, plan])
        state.merge_strategy = MergeStrategy(plan.merge_strategy)
        state.deployment_strategy = DeploymentStrategy(plan.deployment_strategy)
        transition_feature(
            state,
            FeatureWorkflowStatus.CONTRACT_READY,
            reason="The integration contract and repository execution plan are written.",
            agent="feature_planner",
        )
        _initialize_children(state, plan, workspace_root=self._workspace_root)
        _resync_children_with_plan(state, plan)
        await self._checkpoint(state, WorkflowCheckpointBoundary.BEFORE_CLONE)
        return state

    async def _step_execute_workstreams(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        credentials: RequestScopedCredentials,
        technical_prd: TechnicalPRDArtifact | None = None,
        contract: IntegrationContractArtifact | None = None,
        plan: RepositoryExecutionPlanArtifact | None = None,
        target_workstreams: set[str] | None = None,
        targeted_attempt: TargetedAttempt = TargetedAttempt.INTEGRATION_REMEDIATION,
        feedback: dict[str, list[str]] | None = None,
    ) -> FeatureWorkflowSnapshot:
        """Run one wave of repository work, and stop at the boundary where it settles.

        The dependency-ready set is the unit, not a single repository attempt. Keeping the
        wave whole preserves the semaphore, the `FIRST_COMPLETED` result recording and the
        blocking propagation exactly as they are, and it keeps one attempt in one process in
        exclusive possession of its workspace -- which is what makes `26-`'s stale
        `index.lock` removal safe. Splitting each attempt into its own claim would bound the
        loss more tightly and make that removal capable of deleting a lock a live sibling
        holds.

        The artifacts default to the persisted ones, which is what an ordinary step reads.
        The three human-initiated paths -- a granted retry, an approved contract revision --
        pass their own, because they run a set somebody named rather than the set the
        feature's own evidence calls for.
        """
        await self._execute_workstreams(
            state,
            technical_prd=technical_prd or _require_artifact(state.artifacts, TechnicalPRDArtifact),
            contract=contract or _require_artifact(state.artifacts, IntegrationContractArtifact),
            plan=plan or _require_artifact(state.artifacts, RepositoryExecutionPlanArtifact),
            target_workstreams=target_workstreams,
            targeted_attempt=targeted_attempt,
            credentials=credentials,
            feedback=feedback if feedback is not None else _integration_remediation_feedback(state),
        )
        # Every path that runs a repository comes through here -- the first attempt, an
        # operator's granted retry, an integration remediation -- so this is the one place a
        # repository that stopped on its own setup has to be recognised.
        _propose_repository_repairs(state)
        return state

    async def _step_integration_review(
        self, state: FeatureWorkflowSnapshot
    ) -> FeatureWorkflowSnapshot:
        """Judge the repositories that passed their own review against the shared contract.

        One verdict, and what that verdict costs: the cycle it spends and the repositories it
        sends back. It deliberately does not run those repositories. Routing is expressed the
        only way a later claim can read it -- the routed children go back to `pending` with
        their attempt counters spent -- so the wave that answers the findings is the ordinary
        execution step, chosen by `next_step` from that evidence rather than handed along in
        memory by a coroutine that has to survive for it to arrive.

        Publication is likewise a separate step. This one stops as soon as the verdict is
        durable and charged, so a crash between judging and publishing re-enters publication
        instead of asking the reviewer the same question again.
        """
        await self._raise_if_cancelled(state)
        contract = _require_artifact(state.artifacts, IntegrationContractArtifact)
        plan = _require_artifact(state.artifacts, RepositoryExecutionPlanArtifact)
        transition_feature(
            state,
            FeatureWorkflowStatus.INTEGRATION_REVIEW,
            reason=(
                "The repositories that passed their own review are being reviewed together "
                "against the contract."
            ),
            agent="integration_reviewer",
        )
        # The newest *approved* result per repository, not the newest result if it
        # happens to be approved. A repository that passed review and then failed an
        # integration remediation still passed review, and its approved revision is the
        # one publication uses -- so reading only the last artifact told this gate that a
        # repository which had produced three results had produced none.
        #
        # AB-Feature-152 is the case. Its console was approved at attempt 1, the gate
        # asked for a change, attempt 2 failed, and the next gate pass reported
        # "No child workflow result was produced for a contract owner" as a critical
        # finding against it -- while both pull requests were being opened from the very
        # approval it could no longer see. A false critical finding is worse than a
        # missing one: it blocks a feature and sends its reader to look for work that was
        # never missing.
        child_results = _latest_approved_child_results(state)
        ready_repository_ids = {item.repository_id for item in child_results}
        missing_required = [
            item.repository_id
            for item in plan.workstreams
            if item.required and item.repository_id not in ready_repository_ids
        ]
        for repository_id in missing_required:
            child = state.child_workflows[repository_id]
            state.child_workflows[repository_id] = child.model_copy(
                update={
                    "blocking_issues": [
                        *child.blocking_issues,
                        "Required repository has no approved, PR-ready child result.",
                    ]
                }
            )
        if missing_required and not child_results:
            # Nothing survived review, so there is nothing to publish and no integration
            # to check.
            transition_feature(
                state,
                FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
                reason=(
                    "No repository produced a review-approved result, so there is nothing to "
                    "review together and nothing to publish."
                ),
                agent="integration_reviewer",
            )
            state.updated_at = datetime.now(UTC)
            # No stage is supplied on purpose. Nothing failed in the integration review
            # -- it never ran. The repositories are where this stopped, and the failing
            # one's own classification is a better answer than "unfinished".
            return ensure_feature_failure_summary(
                state,
                error_type=(FeatureFailureClassification.WORKSTREAM_UNFINISHED.value),
                diagnostics=[
                    "No repository produced a review-approved, publishable result, so "
                    "there was nothing to review together and nothing to publish.",
                ],
            )
        integration_merge_order = _integration_merge_order(plan, ready_repository_ids)
        # An integration review is a function of the contract, the child results it reads,
        # and the order they merge in. When all three are unchanged, recomputing it can
        # only produce the same verdict at the cost of another provider call -- and a
        # resume after a crash between "verdict persisted" and "verdict routed" is exactly
        # that case. Reuse routes the durable verdict once instead of asking again.
        input_signature = _integration_review_input_signature(
            contract=contract,
            child_results=child_results,
            merge_order=integration_merge_order,
        )
        review = _persisted_integration_review(state, input_signature=input_signature)
        if review is None:
            review = await self._integration_reviewer.review(
                feature_id=state.feature_id,
                contract=contract,
                child_results=child_results,
                merge_order=integration_merge_order,
            )
            review_attempt = sum(
                isinstance(artifact, IntegrationReviewArtifact) for artifact in state.artifacts
            )
            review = review.model_copy(
                update={
                    "artifact_id": attempt_artifact_id(
                        FEATURE_ARTIFACT_FILENAMES["integration_review"],
                        review_attempt,
                    ),
                    "metadata": {
                        **review.metadata,
                        "integration_review_attempt": review_attempt,
                        "input_signature": input_signature,
                    },
                }
            )
            _append_artifacts(state, [review])
            await self._checkpoint(state, WorkflowCheckpointBoundary.AFTER_VALIDATION)
        # An integration review cannot approve a contract only half of whose
        # repositories are implemented, so with a required one unfinished it would loop
        # to its cycle limit and publish nothing. The review is still recorded above;
        # publication then opens pull requests for the repositories that did pass their
        # own review, and the feature still ends needing a human.
        if review.review_status == "approved" or missing_required:
            return state
        # Charged with the review that asked for the change, and only once. A reused
        # verdict whose cycle was already spent must not be charged again on the resume
        # that routes it, or a crash between the two would shorten the budget by one
        # every time the feature was picked back up.
        if _integration_cycle_uncharged(state):
            state.integration_review_cycles += 1
        responsible = {item.responsible_repository_id for item in review.cross_repository_findings}
        if state.integration_review_cycles >= state.max_integration_review_cycles or (
            not responsible
        ):
            # The integration review will not approve and no further child retry can
            # change that: either the cycle budget is spent or no finding names a
            # repository to send the work back to. Every repository reaching here passed
            # its own review with a pushed branch, so each still opens its draft pull
            # request. Returning empty-handed left the human who now has to intervene
            # with nothing to read but orphaned branches, and threw away finished work
            # for the second time. The feature stays FAILED_REQUIRES_HUMAN below.
            return state
        # `changes_requested` alone is not an instruction to code again. This edge used to
        # be unconditional, so a repository that had already satisfied the finding -- or
        # could not act on it -- was sent back for every remaining cycle: AB-Feature-111's
        # console ran twelve Engineer/Review passes against one frozen revision and one
        # unchanging recommended fix, right past the five review cycles it was configured
        # for. The one authoritative retry decision is asked here too, per repository.
        eligible: set[str] = set()
        for repository_id in sorted(responsible):
            if repository_id not in state.child_workflows:
                # Not a repository of this feature. Left in the eligible set so the
                # workstream resolution below rejects it exactly as it does today,
                # rather than being silently dropped here.
                eligible.add(repository_id)
                continue
            decision = _integration_remediation_decision(state, review, repository_id)
            if decision.should_retry:
                eligible.add(repository_id)
                continue
            state.child_workflows[repository_id] = _with_remediation_refusal(
                state.child_workflows[repository_id], decision
            )
            _LOGGER.warning(
                "integration_remediation_denied",
                feature_id=state.feature_id,
                repository_id=repository_id,
                stage="integration_review",
                attempt=state.child_workflows[repository_id].retry_count,
                integration_review_cycles=state.integration_review_cycles,
                revision=_reviewed_revision(state, review, repository_id),
                reason=decision.reason,
            )
        if not eligible:
            # Nothing is left that another attempt could change, so this feature stops
            # here instead of spending its remaining cycles reproducing this verdict.
            # Whatever passed its own review is still published by the next step, and the
            # feature still needs a human -- each refused repository now carrying the
            # reason it stopped.
            return state
        self._route_integration_remediation(state, plan=plan, review=review, eligible=eligible)
        return state

    def _route_integration_remediation(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        plan: RepositoryExecutionPlanArtifact,
        review: IntegrationReviewArtifact,
        eligible: set[str],
    ) -> None:
        """Send an integration finding back to the repositories that must answer it.

        The whole of what routing is, written durably: each repository the verdict names
        gets the attempt it is about to spend and goes back to `pending`, which is what makes
        `next_step` schedule the execution step that answers the finding. Held only in a
        coroutine's locals -- which is where it lived -- the routing died with the process,
        and the claim that picked the feature up published the unfixed work instead.

        A repository the plan does not have is rejected here, by the same resolution the
        targeted paths use, rather than being quietly dropped.
        """
        selected = _resolve_target_workstream_ids(plan, eligible) or set()
        workstreams = {item.workstream_id: item for item in plan.workstreams}
        feedback = _integration_remediation_feedback(state)
        for workstream_id in sorted(selected):
            repository_id = workstreams[workstream_id].repository_id
            child = state.child_workflows[repository_id]
            _LOGGER.info(
                "integration_remediation_scheduled",
                feature_id=state.feature_id,
                repository_id=repository_id,
                stage="integration_review",
                attempt=child.retry_count + 1,
                max_attempts=state.max_child_review_cycles,
                integration_review_cycles=state.integration_review_cycles,
                revision=_reviewed_revision(state, review, repository_id),
                findings=len(feedback.get(repository_id, [])),
            )
            # Parent integration remediation is a distinct immutable child attempt.
            # Reusing the previous retry_count would collide with its completion, review,
            # and result artifact IDs and make the new contract approval indistinguishable.
            #
            # `pending` is the routing. It is the status that already means "work still to
            # come" for this repository, so no new field is introduced and no second source
            # of truth is created: the next claim reads the same child row every other part
            # of this workflow reads and finds a repository owing an attempt.
            state.child_workflows[repository_id] = child.model_copy(
                update={
                    "retry_count": child.retry_count + 1,
                    "integration_retry_count": child.integration_retry_count + 1,
                    # Which authority demanded this attempt, recorded beside the counters it
                    # spends. Without it the retry's own record says "Review →
                    # Implementation" and the graph draws the loop out of the lane's own
                    # reviewer, attributing to it a rework the integration review demanded.
                    "targeted_attempt_kind": TargetedAttempt.INTEGRATION_REMEDIATION,
                    "status": ChildWorkflowStatus.PENDING,
                    # The attempt this routing grants has not started coding, and "what
                    # stage is this child in" must mean the same thing on every route: the
                    # review-retry loop resets its boundary the same way. The artifact
                    # references deliberately stay -- they are the settled lineage the next
                    # attempt builds on, and the evidence recovery reads as settledness.
                    "checkpoint_boundary": WorkflowCheckpointBoundary.BEFORE_CODING,
                }
            )
        transition_feature(
            state,
            FeatureWorkflowStatus.CHANGES_REQUESTED,
            reason=(
                "The integration review asked for changes, and the repositories it named are "
                "routed back for another attempt."
            ),
            agent="integration_reviewer",
        )

    async def _step_publish(self, state: FeatureWorkflowSnapshot) -> FeatureWorkflowSnapshot:
        """Open a pull request for every repository whose work passed review.

        Whether the integration review approved the work is read from the durable verdict
        rather than carried here by the caller, which is what lets publication be a step a
        separate claim can run. The three points that used to call this each passed the same
        answer: the latest review's own status.
        """
        contract = _require_artifact(state.artifacts, IntegrationContractArtifact)
        plan = _require_artifact(state.artifacts, RepositoryExecutionPlanArtifact)
        review = _latest_artifact(state.artifacts, IntegrationReviewArtifact)
        return await self._publish_pull_requests(
            state,
            contract=contract,
            plan=plan,
            child_results=_latest_approved_child_results(state),
            integration_approved=review is not None and review.review_status == "approved",
        )

    async def _execute_workstreams(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        technical_prd: TechnicalPRDArtifact,
        contract: IntegrationContractArtifact,
        plan: RepositoryExecutionPlanArtifact,
        target_workstreams: set[str] | None,
        credentials: RequestScopedCredentials,
        feedback: dict[str, list[str]] | None = None,
        targeted_attempt: TargetedAttempt = TargetedAttempt.INTEGRATION_REMEDIATION,
    ) -> None:
        """Run every dependency-ready workstream concurrently and isolate failures to dependents.

        A repository that fails must never starve an unrelated sibling. Only workstreams
        that transitively depend on a failure are blocked; everything else still runs, and
        the parent escalates once no runnable work remains.
        """
        transition_feature(
            state,
            FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
            reason="Every dependency-ready repository workstream is running.",
            agent="child_workflows",
        )
        workstreams = {item.workstream_id: item for item in plan.workstreams}
        repository_by_id = {item.repository_id: item for item in state.repository_specs}
        _require_acyclic_workstreams(workstreams)
        selected_workstream_ids = _resolve_target_workstream_ids(plan, target_workstreams)
        if selected_workstream_ids is not None:
            # Parent integration/contract remediation is a distinct immutable child attempt.
            # Reusing the previous retry_count would collide with its completion, review, and
            # result artifact IDs and make the new contract approval indistinguishable.
            for workstream_id in selected_workstream_ids:
                selected_workstream = workstreams[workstream_id]
                child = state.child_workflows[selected_workstream.repository_id]
                update: dict[str, Any] = {
                    "retry_count": child.retry_count + 1,
                    # The kind, recorded in the same dict as the counters it decides. It is
                    # read back to say which node the retry's arrow starts at, and it is the
                    # only thing that can: the classification fails in both directions, and
                    # `integration_retry_count` is cumulative.
                    "targeted_attempt_kind": targeted_attempt,
                }
                if targeted_attempt is TargetedAttempt.INTEGRATION_REMEDIATION:
                    # An attempt a person granted is not a contract mismatch, and charging it
                    # to the shared integration allowance would take the parent's remaining
                    # review cycles away from the loop that actually needs them.
                    update["integration_retry_count"] = child.integration_retry_count + 1
                state.child_workflows[selected_workstream.repository_id] = child.model_copy(
                    update=update
                )
        limit = asyncio.Semaphore(self._max_parallel_workstreams)
        # What an untargeted pass schedules is `_schedulable_workstream_ids`, which is the
        # same function `next_step` consults to decide that this step should run at all. Two
        # copies of that rule -- which is what these were -- can disagree, and under stepping
        # a disagreement is not a subtle bug: `next_step` names an execution step, the wave
        # finds nothing to do, and the claim re-queues a feature that never moves.
        #
        # The exclusions apply only to an untargeted pass. A named workstream is a decision
        # somebody made deliberately, and `retry_workstream` clears the refusal before it asks.
        remaining = (
            set(selected_workstream_ids)
            if selected_workstream_ids is not None
            else _schedulable_workstream_ids(state, plan)
        )
        failed: set[str] = set()
        blocked: dict[str, list[str]] = {}

        async def _run_with_limit(item: RepositoryWorkstreamPlan) -> ChildExecution:
            async with limit:
                if state.child_workflows[item.repository_id].pending_context_partition:
                    # A crashed run left a context-partition wave in flight; resume the
                    # persisted plan through this same claim path rather than re-deciding
                    # anything -- a partition survives the sweep like any other step (80-).
                    return await self._run_context_partition_wave(
                        state,
                        repository=repository_by_id[item.repository_id],
                        workstream=item,
                        technical_prd=technical_prd,
                        contract=contract,
                        credentials=credentials,
                    )
                return await self._run_one_child(
                    state,
                    repository=repository_by_id[item.repository_id],
                    workstream=item,
                    technical_prd=technical_prd,
                    contract=contract,
                    feedback=(feedback or {}).get(item.repository_id, []),
                    credentials=credentials,
                )

        while remaining:
            try:
                await self._raise_if_cancelled(state)
            except CancellationRequested:
                return
            ready: list[str] = []
            newly_blocked: dict[str, list[str]] = {}
            for workstream_id in sorted(remaining):
                dependencies = set(workstreams[workstream_id].dependency_workstream_ids)
                broken = sorted(dependencies & (failed | set(blocked)))
                if broken:
                    newly_blocked[workstream_id] = broken
                elif not dependencies & remaining:
                    ready.append(workstream_id)
            if newly_blocked:
                # Re-evaluate rather than run: blocking propagates to their dependents too.
                blocked.update(newly_blocked)
                remaining.difference_update(newly_blocked)
                for workstream_id in newly_blocked:
                    await self._mark_child_blocked(
                        state, workstreams[workstream_id], newly_blocked[workstream_id]
                    )
                continue
            if not ready:
                break
            remaining.difference_update(ready)
            selected = [workstreams[workstream_id] for workstream_id in ready]
            for item in selected:
                child = state.child_workflows[item.repository_id]
                state.child_workflows[item.repository_id] = child.model_copy(
                    update={
                        "status": ChildWorkflowStatus.RUNNING,
                        "checkpoint_boundary": WorkflowCheckpointBoundary.BEFORE_CODING,
                    }
                )
                await self._checkpoint(
                    state, WorkflowCheckpointBoundary.BEFORE_CODING, item.repository_id
                )
            # Each result is recorded as its own child settles, not after the slowest one.
            # `asyncio.gather` withheld every result until the last sibling finished, so a
            # repository that failed in its first minute was still displayed as running half
            # an hour later: -105's backend failed on a refused clone at 10:35 and said so at
            # 11:06, one second before the feature did. Nothing about the scheduling changes
            # -- the children still run concurrently under the same semaphore -- only when
            # what they produced becomes visible.
            running = {asyncio.ensure_future(_run_with_limit(item)): item for item in selected}
            contract_change_requested = False
            pending_tasks = set(running)
            while pending_tasks:
                finished, pending_tasks = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in finished:
                    workstream = running[task]
                    outcome = _task_outcome(task)
                    if isinstance(outcome, CancellationRequested):
                        execution = _cancelled_child_execution(
                            state,
                            repository=repository_by_id[workstream.repository_id],
                            workstream=workstream,
                            child=state.child_workflows[workstream.repository_id],
                        )
                    elif isinstance(outcome, RequiredContextRefusal) and (
                        clusters := _context_partition_clusters(
                            state.child_workflows[workstream.repository_id]
                        )
                    ):
                        # A refused retry partitions before it dies (80- Part 3): the
                        # blocking demands split into scoped attempts that each fit, on the
                        # one retry already granted. The plan is persisted first so a crash
                        # resumes the wave through the claim machinery, and the wave runs as
                        # its own task so sibling results keep recording as they settle.
                        refused = state.child_workflows[workstream.repository_id]
                        state.child_workflows[workstream.repository_id] = refused.model_copy(
                            update={
                                "pending_context_partition": clusters,
                                "targeted_attempt_kind": TargetedAttempt.CONTEXT_PARTITION,
                            }
                        )
                        await self._checkpoint(
                            state,
                            WorkflowCheckpointBoundary.BEFORE_CODING,
                            workstream.repository_id,
                        )
                        _LOGGER.info(
                            "child_workstream_context_partition_started",
                            feature_id=state.feature_id,
                            repository_id=workstream.repository_id,
                            workstream_id=workstream.workstream_id,
                            stage="child_workstream",
                            attempt=refused.retry_count,
                            clusters=len(clusters),
                        )
                        wave_task = asyncio.ensure_future(_run_with_limit(workstream))
                        running[wave_task] = workstream
                        pending_tasks.add(wave_task)
                        continue
                    elif isinstance(outcome, BaseException):
                        # The exception text and traceback can contain rejected model values or
                        # repository-controlled paths. Persist only platform-owned context and
                        # error type; typed diagnostics below retain safe classification detail.
                        _LOGGER.error(
                            "child_workstream_failed",
                            feature_id=state.feature_id,
                            repository_id=workstream.repository_id,
                            workstream_id=workstream.workstream_id,
                            stage="child_workstream",
                            attempt=state.child_workflows[workstream.repository_id].retry_count,
                            error_type=type(outcome).__name__,
                            # Already-redacted diagnostics only. AB-Feature-174's ValueError
                            # was diagnosable solely by timing forensics because this record
                            # carried a type name and nothing else.
                            error_detail="; ".join(safe_error_diagnostics(outcome))
                            or "no safe diagnostics",
                        )
                        execution = _failed_child_execution(
                            state,
                            repository=repository_by_id[workstream.repository_id],
                            workstream=workstream,
                            child=state.child_workflows[workstream.repository_id],
                            error=outcome,
                        )
                    else:
                        execution = outcome
                    execution = _with_attempt_identity(
                        execution,
                        repository_id=workstream.repository_id,
                        attempt=state.child_workflows[workstream.repository_id].retry_count,
                    )
                    _append_artifacts(state, _execution_artifacts(execution))
                    child = state.child_workflows[execution.result.repository_id]
                    state.child_workflows[execution.result.repository_id] = child.model_copy(
                        update={
                            "status": _child_status(execution.result.status),
                            "retry_count": int(execution.result.metadata["child_retry_count"]),
                            "code_completion_artifact_id": (
                                execution.result.code_completion_artifact_id
                            ),
                            "review_artifact_id": execution.result.review_artifact_id,
                            "blocking_issues": execution.result.blocking_issues,
                            "checkpoint_boundary": WorkflowCheckpointBoundary.AFTER_VALIDATION,
                            "technology_profile": execution.result.technology_profile,
                            # The same guard `current_revision` has below, for the same
                            # reason: this write persists the *last* attempt, and a result
                            # produced without a plan in hand must not erase the pinned one
                            # a retry will be resumed from (83-).
                            "validation_plan": (
                                execution.result.validation_plan or child.validation_plan
                            ),
                            # The guard, not the mechanism. Every executor path now measures
                            # the revision, so this should rarely have anything to fall back
                            # to -- but this is the write that persists the *last* attempt,
                            # usually a failed one, and without the fallback it erased a
                            # revision the loop already knew. The in-loop write below has
                            # always had it; this one did not.
                            "current_revision": (
                                execution.result.current_revision or child.current_revision
                            ),
                            "current_validation_results": (
                                execution.result.current_validation_results
                            ),
                            "superseded_validation_count": (
                                execution.result.superseded_validation_count
                            ),
                            "scoped_requirements": [
                                item.model_dump(mode="json")
                                for item in execution.result.scoped_requirements
                            ],
                            "out_of_scope_requirements": execution.result.out_of_scope_requirements,
                            "preflight_result": execution.result.preflight_result,
                            "preflight_status": (
                                execution.result.preflight_result.get("validation_readiness")
                                if isinstance(execution.result.preflight_result, dict)
                                else None
                            ),
                            "blocking_setup_issues": (
                                list(execution.result.preflight_result.get("blocking_issues", []))
                                if isinstance(execution.result.preflight_result, dict)
                                else []
                            ),
                            "selected_package_manager": (
                                execution.result.preflight_result.get("package_manager")
                                if isinstance(execution.result.preflight_result, dict)
                                else None
                            ),
                            "layout_evidence": execution.result.layout_evidence,
                            "configured_validation_commands": (
                                list(execution.result.validation_plan.get("commands", []))
                                if isinstance(execution.result.validation_plan, dict)
                                else child.configured_validation_commands
                            ),
                            "test_availability": _test_availability(execution.result),
                            "implementation_expectations": [
                                item.model_dump(mode="json")
                                for item in workstream.implementation_expectations
                            ],
                            "production_files_changed": execution.result.production_files_changed,
                            "test_files_changed": execution.result.test_files_changed,
                            "configuration_files_changed": (
                                execution.result.configuration_files_changed
                            ),
                            "requirements_implemented": execution.result.requirements_implemented,
                            "requirements_not_implemented": (
                                execution.result.requirements_not_implemented
                            ),
                            "failure_classification": execution.result.failure_classification,
                            "retry_strategy": execution.result.retry_strategy,
                            "implementation_retry_count": int(
                                execution.result.metadata.get("implementation_retry_count", 0)
                            ),
                            "validation_retry_count": int(
                                execution.result.metadata.get("validation_retry_count", 0)
                            ),
                            "repository_setup_retry_count": int(
                                execution.result.metadata.get("repository_setup_retry_count", 0)
                            ),
                            "integration_retry_count": int(
                                execution.result.metadata.get("integration_retry_count", 0)
                            ),
                            "meaningful_change": execution.result.meaningful_change,
                            "meaningful_change_reason": execution.result.meaningful_change_reason,
                            "retry_refusal_reason": execution.result.metadata.get(
                                "retry_refusal_reason"
                            ),
                            "production_diff_fingerprint": (
                                execution.result.production_diff_fingerprint
                            ),
                            "previous_attempt_fingerprint": (
                                execution.result.previous_attempt_fingerprint
                            ),
                            "runtime_wall_seconds": (
                                _clock_seconds(
                                    execution.result.metadata.get("runtime_wall_seconds")
                                )
                                or child.runtime_wall_seconds
                            ),
                            "runtime_charged_seconds": (
                                _clock_seconds(
                                    execution.result.metadata.get("runtime_charged_seconds")
                                )
                                or child.runtime_charged_seconds
                            ),
                        }
                    )
                    await self._checkpoint(
                        state,
                        WorkflowCheckpointBoundary.AFTER_VALIDATION,
                        execution.result.repository_id,
                    )
                    if execution.contract_change_request is not None:
                        _append_artifacts(state, [execution.contract_change_request])
                        contract_change_requested = True
                    elif execution.result.status != "approved":
                        # Record the failure and keep scheduling: only this workstream's
                        # dependents are affected, never an independent sibling.
                        failed.add(workstream.workstream_id)
            if contract_change_requested:
                transition_feature(
                    state,
                    FeatureWorkflowStatus.WAITING_FOR_HUMAN,
                    reason=(
                        "A repository asked for a change to the shared contract, which only a "
                        "person may decide."
                    ),
                    agent="human_contract_owner",
                )
                return
            if await self._is_cancelled():
                await self._mark_cancelled(state)
                return
        unfinished = [
            workstreams[workstream_id]
            for workstream_id in sorted(failed | set(blocked) | remaining)
            if repository_by_id[workstreams[workstream_id].repository_id].required
        ]
        if unfinished:
            transition_feature(
                state,
                FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
                reason=(
                    f"{len(unfinished)} required repository workstream(s) did not finish, so "
                    "this feature is unfinished whatever its siblings reached."
                ),
                agent="child_workflows",
            )
            # Recorded here, but the caller still runs integration review and publication
            # over whatever passed review. Stopping the feature at this point discarded
            # finished repositories because a sibling was unfinished: r-2 approved its
            # frontend and opened nothing, r-3 approved its backend and opened nothing.

    def _repository_runtime_overrun(
        self, started_at: float, *, attempts: int, excluded_seconds: float = 0.0
    ) -> float | None:
        """Return this workstream's charged runtime, once that exceeds its ceiling.

        ``None`` while the ceiling is disabled, before the first attempt has been spent, or
        while the workstream is still inside its budget -- so the caller reads a single
        answer rather than assembling one from three conditions at the point of decision.

        ``excluded_seconds`` is time spent inside external operations that ended in a
        classified provider fault, retries and stall included. The ceiling exists to bound
        what this deployment spends on a repository's work; a provider stall is not the
        repository's work, and charging it ended AB-Feature-171's backend after one call
        consumed 54 of its 90 minutes and then failed terminally. The ceiling's value does
        not change; what it measures does.
        """
        limit = self._repository_runtime_limit_seconds
        if limit <= 0 or attempts <= 0:
            return None
        elapsed = max(0.0, time.monotonic() - started_at - excluded_seconds)
        return elapsed if elapsed > limit else None

    async def _mark_child_blocked(
        self,
        state: FeatureWorkflowSnapshot,
        workstream: RepositoryWorkstreamPlan,
        blocking_workstream_ids: list[str],
    ) -> None:
        """Record a dependent that cannot run so a starved repository is never silent."""
        child = state.child_workflows[workstream.repository_id]
        state.child_workflows[workstream.repository_id] = child.model_copy(
            update={
                "status": ChildWorkflowStatus.BLOCKED,
                "blocking_issues": [
                    "Blocked by an unfinished build-artifact dependency: "
                    f"{', '.join(blocking_workstream_ids)}."
                ],
            }
        )
        await self._checkpoint(
            state, WorkflowCheckpointBoundary.BEFORE_CODING, workstream.repository_id
        )

    async def _run_one_child(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repository: RepositorySpec,
        workstream: RepositoryWorkstreamPlan,
        technical_prd: TechnicalPRDArtifact,
        contract: IntegrationContractArtifact,
        feedback: Sequence[str],
        credentials: RequestScopedCredentials,
    ) -> ChildExecution:
        """Retry only this child repository review loop, never any sibling workstream."""
        child = state.child_workflows[repository.repository_id]
        # What this attempt inherits from the one before it, with any design-conflict stop
        # restated as a fact rather than repeated as a question. This is the one place the
        # inheritance is built, so it is the one place the restatement belongs: a retry grant, a
        # resume and the first attempt of an integration-fix re-entry all arrive here. So does
        # the verdict path, which has nothing left to restate -- an answered question is dropped
        # from the list outright, because the decision it carries has replaced it.
        #
        # Nothing here shifts a settled identity. The three identities that matter are computed
        # elsewhere and from other inputs: `finding_fingerprint` and the ledger's
        # `fingerprint_for_text` read this attempt's own result and the review artifacts behind
        # it, and the unchanged-failure coarse key reads recorded validation evidence. What this
        # list decides is what a prompt says.
        inherited_feedback = restated_for_the_next_attempt(
            list(feedback) or list(child.blocking_issues),
            state.artifacts,
            repository_id=repository.repository_id,
        )
        # The design questions a person has decided about this repository. Read once, here,
        # rather than per attempt: the only thing that adds one is a human answering a stopped
        # workstream's question, which happens between runs of this loop and never inside one.
        #
        # Used for two things that must not drift apart -- which recurrences may no longer stop
        # the loop, and what the engineer is told was decided.
        settled = settled_questions(state.artifacts, repository_id=repository.repository_id)
        settled_fingerprints = {item.fingerprint for item in settled}
        # This repository's own frames, at build fidelity, resolved once here for the same
        # reason `settled` is read once here: it must not change between this workstream's
        # attempts. A design re-read per attempt is a work order that can move under a retry,
        # which is the failure 67- was about -- and the reason AB-Feature-216's validation plan
        # is pinned rather than re-derived. Reuse is keyed on the artifact already existing for
        # this repository against this snapshot revision, which survives a crash and a resume
        # in a way an in-memory flag or an attempt counter does not.
        await self._resolve_design_detail(state, repository=repository, workstream=workstream)
        # What review has already demanded and been given, carried into the very first attempt
        # of this loop as well as into every retry. The first attempt matters most: an
        # integration-fix re-entry starts a fresh loop whose in-memory bookkeeping is empty
        # while the lineage behind it is not, and that is the attempt AB-Feature-184 spent
        # undoing its own approved work. It is also the attempt a verdict authorises, which is
        # why the decided questions lead the list.
        retry_feedback = [
            *inherited_feedback,
            *settled_question_lines(settled),
            *satisfied_invariant_lines(
                resolved_issue_ledger(
                    state.artifacts,
                    child_workflow_id=child.child_workflow_id,
                    repository_id=repository.repository_id,
                ),
                demanded=inherited_feedback,
                settled=settled_fingerprints,
            ),
        ]
        # How many consecutive attempts each finding has survived, and everything reported
        # earlier that the last attempt did not repeat. Both exist so the next attempt can be
        # told what is still broken separately from what merely used to be.
        unresolved_since: dict[str, int] = {}
        superseded_findings: list[str] = []
        last_findings: tuple[str, ...] | None = None
        # The same two facts about the previous attempt's diagnostics, reduced to the defects
        # they name rather than their wording, plus how many attempts running have named
        # exactly those. Tracked separately from `last_findings` because a model that
        # rephrases its report every attempt defeats the string comparison entirely.
        last_signature: tuple[str, ...] = ()
        # The three convergence counters, carried between attempts. Every one of them is
        # incremented, reset and read by `decide_child_retry`: this loop hands them back in
        # and takes what the verdict returns, so nothing here has an opinion about what they
        # mean. Holding them is not the same as deciding on them.
        repeated_signature_attempts = 0
        identical_resubmissions = 0
        returns_to_earlier_states = 0
        infrastructure_faults = 0
        # Whether this workstream has already answered a truncated response by stepping the
        # effort down. Exactly one such step: a truncation is a deterministic provider
        # answer (`is_transient_provider_fault` refuses it, unchanged), and the degraded
        # retry is permitted only because it changes the request -- the one remedy the
        # truncation diagnostic itself prescribes. A second truncation, at the lower
        # effort, is terminal exactly as before.
        truncation_effort_step_spent = False
        # Held across attempts, not read from the last one. A workstream that implemented
        # something and then had a later attempt rejected before it committed still wrote
        # production source, and attributing its failure to "nothing was ever built" would
        # send an operator to re-examine a requirement that was implemented fine.
        production_source_written = False
        # Every path any attempt of this workstream wrote, held across attempts for the same
        # reason. The terminal triage compares the final blocking diagnostics against this
        # lineage: a blocker that names only files the feature itself authored is a defect
        # inside the implementation, and its narrative must not send anyone to repair the
        # repository -- which is exactly where 176's terminal text sent its reader.
        attempt_authored_paths: list[str] = []
        # An earlier attempt can commit and push before a later one fails. The result artifact
        # describes only the final attempt, so without this the branch already on the remote
        # disappears from the record entirely and nobody knows to review or delete it.
        stranded: tuple[str, str] | None = None
        # Which model role the next attempt runs with, and what it is being asked to correct.
        # Decided here rather than inside the executor because the decision belongs beside the
        # retry verdict it depends on: an attempt is routed only once the loop has established
        # that it may happen at all.
        routing = _initial_routing_request(child, inherited_feedback)
        child = self._route_attempt(
            state, repository=repository, child=child, contract=contract, request=routing
        )
        state.child_workflows[repository.repository_id] = child
        started_at = time.monotonic()
        # Two clocks, at two scales, and conflating them is what made AB-Feature-218's
        # runtime record unreadable. `started_at` and `provider_fault_seconds` are the
        # WORKSTREAM's, and only the runtime ceiling reads them -- the ceiling bounds what
        # this deployment spends on a repository, so cumulative is the question it asks.
        # Everything recorded ON AN ATTEMPT uses the pair below, reset when a retry is
        # granted at the bottom of this loop. Summing 218's six recorded backend values gave
        # 9,543 s for a workstream that ran 4,952 s, because attempt N's clock described the
        # whole workstream to that point and every reader summed them as attempt durations.
        provider_fault_seconds = 0.0
        attempt_clock_started = started_at
        attempt_fault_seconds = 0.0
        # The distinct labels of the faults absorbed above, in first-seen order. 186's
        # attempt-0 record carried only the two clocks; the fault class, count and the
        # degraded-retry marker existed nowhere in the artifact, so the wall-vs-charged
        # difference could not be explained without the server logs.
        fault_classes: list[str] = []
        while True:
            await self._raise_if_cancelled(state)
            # Between attempts, never during one. A ceiling that could fire mid-attempt
            # could fire during a push, and a breaker that interrupts an external effect
            # leaves exactly the unconfirmed state the rest of this platform is built to
            # avoid. This stops scheduling; whatever is in flight finishes.
            #
            # Never before the first attempt either: `child.retry_count` is what says one has
            # been spent. A backstop that can end a workstream before it has run once is a
            # scheduling policy, and this is not allowed to be one.
            overrun = self._repository_runtime_overrun(
                started_at, attempts=child.retry_count, excluded_seconds=provider_fault_seconds
            )
            if overrun is not None:
                # The one record on this loop that is deliberately the WORKSTREAM's and not
                # this attempt's: the ceiling fired on cumulative charged runtime, so the
                # cumulative pair is what explains the stop it is recorded beside.
                wall_seconds = time.monotonic() - started_at
                return _with_fault_evidence(
                    _with_attempt_identity(
                        _repository_runtime_limit_execution(
                            state,
                            repository=repository,
                            workstream=workstream,
                            child=child.model_copy(
                                update={
                                    "runtime_wall_seconds": wall_seconds,
                                    "runtime_charged_seconds": overrun,
                                }
                            ),
                            elapsed_seconds=overrun,
                            wall_seconds=wall_seconds,
                            limit_seconds=self._repository_runtime_limit_seconds,
                        ),
                        repository_id=repository.repository_id,
                        attempt=child.retry_count,
                    ),
                    fault_count=infrastructure_faults,
                    fault_seconds_excluded=provider_fault_seconds,
                    fault_classes=fault_classes,
                    truncated_degraded_retry=truncation_effort_step_spent,
                    wall_seconds=wall_seconds,
                    charged_seconds=overrun,
                )
            attempt_started = time.monotonic()
            try:
                execution = await self._child_executor.run(
                    feature=state,
                    repository=repository,
                    workstream=workstream,
                    child=child,
                    technical_prd=technical_prd,
                    contract=contract,
                    feedback=retry_feedback,
                    credentials=credentials,
                )
            except _UNCONFIRMED_OPERATION_FAULTS:
                # Ordered first so it wins over the fault clause below. An effect that could
                # not be confirmed is a question for a person, and spending attempts on it
                # would be the platform guessing about exactly what it refuses to guess.
                raise
            except _INFRASTRUCTURE_FAULTS as error:
                # This left the child loop entirely and was caught by the fan-out, so the
                # repository was recorded as failed having spent none of its retries: -072's
                # backend ended before it wrote a line, and -070's ended one finding from
                # approval. The provider is not the repository, and a fault here earns the
                # attempt again rather than the workstream's end.
                if _truncated_response_error(error) is not None and (
                    not truncation_effort_step_spent
                ):
                    # A truncated response is deterministic for the request that was sent,
                    # and its own diagnostic names the remedy: thinking shares the response
                    # bound, so lower the effort. AB-Feature-181's coding call spent its
                    # entire 128000-token bound -- the model's output ceiling, so no larger
                    # bound existed -- at `xhigh` and the workstream ended at attempt 0
                    # with the fix printed in its own failure record. Apply that remedy
                    # once: re-run the same attempt with the persisted selection stepped
                    # down one effort level. No step available (effort absent, `low`, or
                    # disabled thinking) falls through to the deterministic stop below.
                    degraded = degraded_truncation_retry(
                        ModelRoutingDecision.from_persisted(child.model_routing)
                    )
                    if degraded is not None:
                        truncation_effort_step_spent = True
                        child = child.model_copy(
                            update={"model_routing": degraded.model_dump(mode="json")}
                        )
                        # Persisted before the re-run for the same reason `_route_attempt`
                        # persists: recovery after a crash must rebuild the degraded
                        # client, not the one that already truncated.
                        state.child_workflows[repository.repository_id] = child
                        await self._checkpoint(
                            state,
                            WorkflowCheckpointBoundary.BEFORE_CODING,
                            repository.repository_id,
                        )
                        _LOGGER.warning(
                            "child_workstream_truncated_response_degraded_retry",
                            feature_id=state.feature_id,
                            repository_id=repository.repository_id,
                            stage="child_workstream",
                            attempt=child.retry_count,
                            reasoning=degraded.reasoning,
                            model=degraded.model,
                            fault=_fault_label(error),
                            reason=degraded.routing_reason,
                        )
                        continue
                if not is_transient_provider_fault(error):
                    # The provider answered deterministically -- a refusal, a truncation,
                    # a 4xx. Retrying re-asks a settled question; let the ordinary failure
                    # path record the declared classification and its diagnostics instead.
                    raise
                infrastructure_faults += 1
                fault = _fault_label(error)
                if fault not in fault_classes:
                    fault_classes.append(fault)
                if infrastructure_faults > _ALLOWED_INFRASTRUCTURE_FAULTS:
                    # Triaged like any other stop. This path skipped it, so -083's backend
                    # ended with a bare exception type, no cause and no question -- while its
                    # sibling, which stopped through the ordinary route, told its operator
                    # what had been tried and what to decide.
                    ceiling_wall_seconds = time.monotonic() - attempt_clock_started
                    return _with_fault_evidence(
                        _with_attempt_identity(
                            _with_triage(
                                _failed_child_execution(
                                    state,
                                    repository=repository,
                                    workstream=workstream,
                                    child=child,
                                    error=error,
                                ),
                                TerminalTriage(
                                    # The triage cause stays `PROVIDER_UNAVAILABLE` even
                                    # though the classification beside it now names Git.
                                    # `TerminalCause` answers a narrower question -- who has
                                    # to act next -- and the answer is the same person for
                                    # either service: somebody who checks that it is healthy
                                    # and re-runs this repository. Which service it was is in
                                    # the classification and in the question below, both of
                                    # which this record carries.
                                    cause=TerminalCause.PROVIDER_UNAVAILABLE,
                                    question=(
                                        f"{_fault_source_name(error)} failed "
                                        f"{infrastructure_faults} times in a row for this "
                                        "repository, so it was not the repository or the "
                                        "requirement that stopped this work. Is "
                                        f"{_fault_source_name(error, subject=False)} healthy, "
                                        "and should this repository be run again once it is?"
                                    ),
                                    evidence=[
                                        f"{infrastructure_faults} consecutive "
                                        f"{_fault_source_name(error, subject=False)} faults.",
                                        "No attempt was completed after the final fault.",
                                    ],
                                ),
                            ),
                            repository_id=repository.repository_id,
                            attempt=child.retry_count,
                        ),
                        fault_count=infrastructure_faults,
                        fault_seconds_excluded=attempt_fault_seconds,
                        fault_classes=fault_classes,
                        truncated_degraded_retry=truncation_effort_step_spent,
                        wall_seconds=ceiling_wall_seconds,
                        charged_seconds=max(0.0, ceiling_wall_seconds - attempt_fault_seconds),
                    )
                # The whole stalled attempt ended in a classified provider fault
                # (`is_transient_provider_fault` is the platform's single answer to that,
                # and it is what admitted the error to this branch), so its time -- stall
                # and the backoff about to be slept -- is the weather, not the work. It is
                # excluded from the runtime-ceiling accounting and kept on the wall clock.
                backoff = _fault_backoff_seconds(error, fault_count=infrastructure_faults)
                stalled = (time.monotonic() - attempt_started) + backoff
                provider_fault_seconds += stalled
                attempt_fault_seconds += stalled
                _LOGGER.warning(
                    "child_workstream_provider_fault_retried",
                    feature_id=state.feature_id,
                    repository_id=repository.repository_id,
                    stage="child_workstream",
                    attempt=child.retry_count,
                    fault_count=infrastructure_faults,
                    fault_seconds_excluded=round(provider_fault_seconds, 3),
                    fault=fault,
                    backoff_seconds=round(backoff, 3),
                )
                await self._sleep_between_faults(state, backoff)
                continue
            except SourceValidationError as error:
                execution = _failed_child_execution(
                    state,
                    repository=repository,
                    workstream=workstream,
                    child=child,
                    error=error,
                )
            attempt_wall_seconds = time.monotonic() - attempt_clock_started
            execution = _with_fault_evidence(
                _with_attempt_identity(
                    execution,
                    repository_id=repository.repository_id,
                    attempt=child.retry_count,
                ),
                fault_count=infrastructure_faults,
                fault_seconds_excluded=attempt_fault_seconds,
                fault_classes=fault_classes,
                truncated_degraded_retry=truncation_effort_step_spent,
                wall_seconds=attempt_wall_seconds,
                charged_seconds=max(0.0, attempt_wall_seconds - attempt_fault_seconds),
            )
            if execution.code_completion is not None and execution.code_completion.commit_sha:
                stranded = (child.branch_name, execution.code_completion.commit_sha)
            if execution.result.status in {"approved", "waiting_for_contract_change", "cancelled"}:
                return execution
            if execution.result.metadata.get("retry_refusal_reason") == "manual_review_required":
                return execution
            classification = _failure_classification(execution)
            production_source_written = production_source_written or bool(
                execution.result.production_files_changed
            )
            for authored_path in (
                *execution.result.production_files_changed,
                *execution.result.test_files_changed,
                *execution.result.configuration_files_changed,
            ):
                if authored_path not in attempt_authored_paths:
                    attempt_authored_paths.append(authored_path)
            previous_findings = execution.result.blocking_issues or list(retry_feedback)
            retry_plan = build_retry_plan(
                classification,
                expected_source_areas=_effective_source_areas(
                    workstream, execution.result.layout_evidence
                ),
                previous_findings=previous_findings,
                current_revision=execution.result.current_revision,
                # Everything the same review said that did not block. The severity filter in
                # `_blocking_findings` keeps these out of the work order, which is correct --
                # but it used to drop them entirely, so a `medium` raised alongside a `high`
                # was withheld from the remediation and then blocked the attempt after it.
                advisory_findings=execution.result.advisory_findings,
                # Whether anything actually rejected the source, which decides which of the
                # two `validation_source_failure` shapes this is -- see `build_retry_plan`.
                source_verdict_available=a_required_command_rejected_the_source(
                    execution.result.current_validation_results
                ),
            )
            retry_plan = retry_plan.model_copy(
                update={
                    "previous_approach_to_avoid": (
                        f"Attempt {child.retry_count}: {retry_plan.previous_approach_to_avoid}"
                    )
                }
            )
            completeness = ImplementationCompletenessResult(
                passed=not execution.result.requirements_not_implemented,
                production_files_changed=execution.result.production_files_changed,
                test_files_changed=execution.result.test_files_changed,
                configuration_files_changed=execution.result.configuration_files_changed,
                requirements_implemented=execution.result.requirements_implemented,
                requirements_not_implemented=execution.result.requirements_not_implemented,
                implementation_expectations_satisfied=[],
                findings=[],
                production_diff_fingerprint=(
                    execution.result.production_diff_fingerprint
                    or _production_fingerprint(execution.result.production_files_changed)
                ),
                test_diff_fingerprint=(
                    execution.result.test_diff_fingerprint
                    or _production_fingerprint(execution.result.test_files_changed)
                ),
            )
            # The commit gate runs the repository's own linters before anything lands, so a
            # rejected attempt commits nothing and reports no changed files. That is not the
            # same as an attempt that wrote nothing, and refusing to retry it stranded a
            # workstream one lint fix short of approval.
            # Not conditioned on the diagnostics differing. A linter repeats itself for as
            # long as a rule is still broken, and rep-a's frontend was stopped after three
            # attempts -- of eight it was allowed -- because its eslint output was identical
            # each time, while the source behind it was being rewritten between attempts.
            # The validation budget is what bounds this loop.
            # Asked of the commit gate itself, which is the only thing that knows. Inferring
            # it from the classification and an absent commit SHA identified the wrong
            # attempts in both directions, and every rule below inherited the error:
            #
            #   Nothing commits until review approves, so `not commit_sha` holds for every
            #   failed attempt -- 82 of the 93 code completions in the last fifteen features.
            #   That disjunct never excluded anything.
            #
            #   And `validation_source_failure` is what a reviewer's own findings are called
            #   whenever one of them reports a failing command, so an ordinary review
            #   rejection arrived here wearing the pre-commit label.
            #
            # Together those gave the allowance to 72 of 117 attempts when 24 had earned it.
            # Twenty-six of the 72 reproduced the previous production diff byte for byte:
            # AB-Feature-112's console submitted fingerprint 04e11d7a five separate times and
            # spent all twelve of its review cycles, because this flag suppressed the repeat
            # rules that exist to stop exactly that.
            source_rejected_before_commit = bool(
                execution.result.metadata.get("source_validation_rejected")
            ) and bool(execution.result.blocking_issues)
            # Whether the gate is saying something new. `meaningful_progress` needs this to
            # tell a workstream that is narrowing its findings from one that is resubmitting
            # the same file to the same linter: -105 did the latter eleven times, and every
            # exemption above read it as progress because neither asked this question.
            diagnostics_changed = last_findings is None or (
                tuple(execution.result.blocking_issues) != last_findings
            )
            meaningful, reason = meaningful_progress(
                completeness,
                previous_fingerprint=child.production_diff_fingerprint,
                previous_test_fingerprint=child.test_diff_fingerprint,
                blocking_configuration_resolved=False,
                source_rejected_before_commit=source_rejected_before_commit,
                diagnostics_changed=diagnostics_changed,
            )
            # The repeat rule inside `meaningful_progress` compares against the attempt
            # immediately before, so alternating between two states always looks like
            # progress. -069's console cycled unwired -> wired into the wrong page ->
            # unwired, its third attempt byte-identical to its first, each one "new" to a
            # rule that only ever looked back one step. Whether that is a circle is
            # `decide_child_retry`'s to say; this only establishes that it happened.
            revisited_earlier_state = _revisits_an_earlier_attempt(
                state,
                child_workflow_id=child.child_workflow_id,
                production_fingerprint=completeness.production_diff_fingerprint,
                test_fingerprint=completeness.test_diff_fingerprint,
            )
            # This attempt's own findings, and whether they say anything the last attempt did
            # not -- as text, and as the defects the text names. Only this attempt's findings
            # are compared, because inherited feedback is constant by construction and would
            # stop the first retry.
            findings = tuple(execution.result.blocking_issues)
            deterministic_gate = bool(execution.result.metadata.get("deterministic_gate"))
            signature = diagnostic_signature(findings)
            # What both authorities have demanded and been given across this workstream's whole
            # lineage, and whether either of them is demanding one of those things again. Built
            # here, before the retry verdict, because a demand that has already been satisfied
            # once is not evidence about convergence at all -- and read from the artifacts as
            # they stand, which is every cycle but this one: this attempt's result is appended
            # only after the decision below, so a finding cannot recur against itself.
            ledger = resolved_issue_ledger(
                state.artifacts,
                child_workflow_id=child.child_workflow_id,
                repository_id=repository.repository_id,
            )
            recurrences = (
                recurring_resolved_issues(
                    ledger,
                    demanded={
                        IssueAuthority.REPOSITORY_REVIEW: findings,
                        IssueAuthority.INTEGRATION_REVIEW: _outstanding_integration_fixes(
                            state, repository.repository_id
                        ),
                    },
                    settled=settled_fingerprints,
                )
                # A deterministic gate is exempt for the reason the repeat rule already exempts
                # it: it emits the same sentence for as long as its condition holds, so its
                # reappearance is the check being consistent, not a decision being reversed.
                if not deterministic_gate
                else []
            )
            # The reworded sibling of the ledger check (77-, item 35): a demand blocking its
            # third review cycle under wordings the exact fingerprint could not match. Runs
            # only where the exact ledger found nothing -- the reversal path outranks it --
            # never for a deterministic gate, and only against a real review verdict, so an
            # infrastructure or self-review sentence can never contribute a theme (item 30's
            # trap). The theme is derived fresh from the lineage here and stored nowhere.
            theme_recurrence = (
                recurring_demand_theme(
                    state.artifacts,
                    child_workflow_id=child.child_workflow_id,
                    current_review=execution.review,
                    current_result=execution.result,
                    current_attempt=child.retry_count,
                    settled=settled_fingerprints,
                )
                if not recurrences and not deterministic_gate and execution.review is not None
                else None
            )
            # The second, coarser measure beside `diagnostic_signature`: which workspace files
            # the failing evidence of consecutive attempts keeps naming. 183's attempts moved
            # every token the fine key reads -- sixteen failing tests, then ten, with different
            # excerpts and result codes -- while the failing file did not move at all, and
            # 194's command string moved every round while the file still did not. Same
            # lineage, same exemption, and read before this attempt's own result is appended,
            # so `execution.result` is passed rather than found in `state.artifacts`.
            unchanged_failure = (
                unchanged_failure_run(
                    state.artifacts,
                    child_workflow_id=child.child_workflow_id,
                    current=execution.result,
                )
                if not deterministic_gate
                else None
            )
            counter_field = retry_counter_field(classification)
            counter_value = getattr(child, counter_field)
            decision = decide_child_retry(
                classification=classification,
                attempt_count=counter_value,
                # A grant raises this repository's ceiling and nobody else's. The feature's
                # configured limits are left alone so a sibling that has spent nothing does
                # not inherit an allowance made for a repository somebody looked at.
                budget=_retry_budget(state, classification) + child.granted_extra_attempts,
                retry_count=child.retry_count,
                max_child_review_cycles=state.max_child_review_cycles,
                meaningful_change=meaningful,
                # Evidence, all of it. Three of these rules used to run in this loop after the
                # call above had already returned a verdict, stop the workstream anyway, and
                # rewrite its reason -- which is how six production rows came to record their
                # own stop as "Attempt N may proceed with a changed strategy". Nothing here
                # decides anything; each line reports something this attempt did.
                meaningful_change_reason=reason,
                production_change_present=bool(completeness.production_files_changed),
                revisited_earlier_state=revisited_earlier_state,
                source_rejected_before_commit=source_rejected_before_commit,
                findings_repeat_previous=bool(findings) and findings == last_findings,
                diagnostic_signature_repeated=bool(signature) and signature == last_signature,
                deterministic_gate=deterministic_gate,
                identical_resubmissions=identical_resubmissions,
                returns_to_earlier_states=returns_to_earlier_states,
                repeated_diagnostic_signatures=repeated_signature_attempts,
            )
            identical_resubmissions = decision.identical_resubmissions
            returns_to_earlier_states = decision.returns_to_earlier_states
            repeated_signature_attempts = decision.repeated_diagnostic_signatures
            last_signature = signature
            result_child = child.model_copy(
                update={
                    "preflight_result": execution.result.preflight_result,
                    "preflight_status": (
                        execution.result.preflight_result.get("validation_readiness")
                        if isinstance(execution.result.preflight_result, dict)
                        else child.preflight_status
                    ),
                    "blocking_setup_issues": (
                        list(execution.result.preflight_result.get("blocking_issues", []))
                        if isinstance(execution.result.preflight_result, dict)
                        else child.blocking_setup_issues
                    ),
                    "selected_package_manager": (
                        execution.result.preflight_result.get("package_manager")
                        if isinstance(execution.result.preflight_result, dict)
                        else child.selected_package_manager
                    ),
                    "layout_evidence": execution.result.layout_evidence,
                    # The revision this attempt actually produced, carried into the next one.
                    # Left out, the child handed to every retry still named the revision the
                    # loop started from -- `None` on a first pass -- so an attempt's own
                    # durable state disagreed with the checkout the engineer was looking at,
                    # and a restart mid-loop resumed against a revision two attempts old.
                    # Falls back rather than overwriting: an executor that returned no review
                    # reports no revision, and that is not evidence the revision is gone.
                    "current_revision": (
                        execution.result.current_revision or child.current_revision
                    ),
                    "implementation_expectations": [
                        item.model_dump(mode="json")
                        for item in workstream.implementation_expectations
                    ],
                    "failure_classification": classification.value,
                    # The wiring gate records which file and symbol the next attempt has to
                    # connect. build_retry_plan does not know about it and rebuilds this
                    # field from scratch, which dropped the hint before it ever reached the
                    # engineer: fix-1 carried `wiring_repair: null` while both repositories
                    # were failing for exactly the reason it describes.
                    "retry_strategy": _with_wiring_repair(
                        retry_plan.model_dump(mode="json"), execution.result
                    ),
                    # What the verdict concluded, not what `meaningful_progress` observed. The
                    # two differ for an attempt the convergence rules re-judged, and the
                    # durable row has to carry the decision the loop actually acted on.
                    "meaningful_change": decision.meaningful_change,
                    "meaningful_change_reason": decision.meaningful_change_reason,
                    "previous_attempt_fingerprint": child.production_diff_fingerprint,
                    "production_diff_fingerprint": completeness.production_diff_fingerprint,
                    "previous_test_fingerprint": child.test_diff_fingerprint,
                    "test_diff_fingerprint": completeness.test_diff_fingerprint,
                    "production_files_changed": execution.result.production_files_changed,
                    "test_files_changed": execution.result.test_files_changed,
                    "configuration_files_changed": execution.result.configuration_files_changed,
                    "requirements_implemented": execution.result.requirements_implemented,
                    "requirements_not_implemented": execution.result.requirements_not_implemented,
                    "repository_setup_retry_count": int(
                        execution.result.metadata.get(
                            "repository_setup_retry_count", child.repository_setup_retry_count
                        )
                    ),
                    # Both clocks, so wall != charged is a recorded fact rather than a
                    # reconstruction from journal timestamps -- and both describe THIS
                    # attempt, so a reader may sum them across a workstream.
                    "runtime_wall_seconds": time.monotonic() - attempt_clock_started,
                    "runtime_charged_seconds": max(
                        0.0, time.monotonic() - attempt_clock_started - attempt_fault_seconds
                    ),
                }
            )
            # Only a refusal is a refusal. `decide_child_retry` returns a reason for both
            # verdicts, and writing it unconditionally put the positive one -- "Attempt N may
            # proceed with a changed strategy" -- into the field an escalation reads as the
            # cause of a stop. Six production workstreams say that about themselves.
            execution = _with_child_diagnostics(
                execution,
                result_child.model_copy(
                    update={
                        "retry_refusal_reason": (
                            _DESIGN_CONFLICT_REFUSAL
                            if recurrences
                            else _RECURRING_DEMAND_REFUSAL
                            if theme_recurrence is not None
                            else _UNCHANGED_FAILURE_REFUSAL
                            if unchanged_failure is not None and unchanged_failure.stops_the_loop
                            else (None if decision.should_retry else decision.reason)
                        )
                    }
                ),
            )
            if recurrences:
                # Answered before the convergence verdict, whichever way that verdict went. A
                # re-litigated design decision and a repair that cannot converge both end the
                # workstream, so nothing here grants or spends an attempt; what differs is who
                # the narrative sends the work to. Attributing this to an unresponsive blocker
                # sends someone to repair a repository that is not broken, and attributing it
                # to a capability limit sends them to buy a stronger model to lose the same
                # argument -- which is what AB-Feature-184's operator was told twice.
                #
                # The existing guards are untouched: their counters were computed above from
                # the same evidence as before and carry the same values into the record.
                question, evidence = design_conflict_narrative(recurrences[0])
                evidence = [
                    *evidence,
                    f"{child.retry_count + 1} attempts were spent before the "
                    "reversal was recognized.",
                ]
                # The same question, written down as something a person can answer rather than
                # only as a sentence in a stopped attempt's blocking issues. Recorded here, at
                # the stop, because this is where both positions are in hand: the demand being
                # made now and the wording that was satisfied. `_record_design_conflict` is
                # idempotent on (repository, fingerprint), so a workstream stopped twice on one
                # question does not ask it twice.
                #
                # Nothing about the stop changes. It is still synchronous, still terminal, and
                # still spends no attempt; what the artifact adds is a second way out of it.
                _record_design_conflict(
                    state,
                    recurrences[0],
                    repository_id=repository.repository_id,
                    child_workflow_id=child.child_workflow_id,
                    question=question,
                    evidence=evidence,
                    attempts_spent=child.retry_count + 1,
                )
                return _with_stranded_commit(
                    _with_triage(
                        execution,
                        TerminalTriage(
                            cause=TerminalCause.DESIGN_CONFLICT,
                            question=question,
                            evidence=evidence,
                        ),
                    ),
                    stranded,
                )
            if theme_recurrence is not None:
                # The same stop, one guard later, for the demand the ledger cannot see: three
                # review cycles blocked on one theme, reworded each round, so no exact key
                # ever matched and no repeat rule ever counted it. AB-Feature-208's backend
                # burned six such rounds to the identical-resubmission stop at r8 without a
                # human ever being asked. Terminal, synchronous, spending nothing -- and the
                # question quotes every wording so the person sees the drift. The verdict a
                # person gives binds through the earliest wording's exact fingerprint only;
                # the theme itself decides nothing but the asking.
                question, evidence = recurring_demand_narrative(theme_recurrence)
                _record_recurring_demand_conflict(
                    state,
                    theme_recurrence,
                    repository_id=repository.repository_id,
                    child_workflow_id=child.child_workflow_id,
                    question=question,
                    evidence=evidence,
                    attempts_spent=child.retry_count + 1,
                )
                return _with_stranded_commit(
                    _with_triage(
                        execution,
                        TerminalTriage(
                            cause=TerminalCause.DESIGN_CONFLICT,
                            question=question,
                            evidence=evidence,
                        ),
                    ),
                    stranded,
                )
            if unchanged_failure is not None and unchanged_failure.stops_the_loop:
                # Second, after the ledger and before the convergence verdict. The order is the
                # rule: a design decision being re-litigated is not a repair that is stuck, and
                # reading this one first would send an operator to run a command about a
                # question no command answers. Placed before the verdict for the reason the
                # block above is -- both stops end the workstream, so nothing here grants or
                # spends an attempt, and what differs is only the sentence a person reads.
                #
                # Attributed to an unresponsive blocker rather than to a new cause, because
                # that is exactly what the evidence establishes: the source changed every
                # round and the required command's verdict on it did not. The question below
                # is this stop's own, naming the command and the files instead of the generic
                # three the existing triage chooses between.
                question, evidence = unchanged_failure_narrative(unchanged_failure)
                return _with_stranded_commit(
                    _with_triage(
                        # Deliberately not `_non_converging_execution`: its sentence says the
                        # diagnostics were identical, which is the one thing that was not true
                        # here, and asserts the defect is beyond what rewriting the source can
                        # fix, which this evidence does not establish. `_with_triage` puts the
                        # question below into `blocking_issues` on its own.
                        execution,
                        TerminalTriage(
                            cause=TerminalCause.BLOCKER_UNRESPONSIVE,
                            question=question,
                            evidence=evidence,
                        ),
                    ),
                    stranded,
                )
            if not decision.should_retry:
                # One refusal, one reason, one exit. The convergence stop used to arrive here
                # separately, after this loop had already written the opposite verdict into
                # durable state and then overwritten it; all that distinguishes it now is the
                # sentence it adds to the attempt saying its remaining budget went unspent.
                stopped = (
                    _non_converging_execution(execution) if decision.non_converging else execution
                )
                # How long each of this attempt's own review findings has now survived, so
                # the triage can tell a capability limit from an open question. Gate output
                # is excluded: a linter repeating its sentence is the tool being consistent,
                # not a substantive finding a model failed to answer.
                survived_findings = (
                    {item: unresolved_since.get(item, 0) + 1 for item in findings}
                    if findings and not deterministic_gate
                    else {}
                )
                longest_survivor = (
                    max(survived_findings.items(), key=lambda entry: entry[1])
                    if survived_findings
                    else ("", 0)
                )
                return _with_stranded_commit(
                    _with_triage(
                        stopped,
                        triage_stopped_workstream(
                            classification=classification,
                            attempts=child.retry_count + 1,
                            production_source_written=production_source_written,
                            # The two measurements `blocker_repeated` used to merge into one
                            # bit (77-, item 33): only a diagnostic comparison may put "the
                            # diagnostics did not change" on the record, and a resubmitted
                            # tree is reported as exactly that, fingerprints attached.
                            diagnostics_repeated=decision.non_converging,
                            resubmission=(
                                _tree_resubmission(
                                    state,
                                    child_workflow_id=child.child_workflow_id,
                                    production_fingerprint=(
                                        completeness.production_diff_fingerprint
                                    ),
                                )
                                # Only when the verdict judged THIS attempt a repeat. The
                                # counter alone survives pre-commit-rejected attempts, whose
                                # post-reset fingerprint describes the state they started
                                # from, not source they submitted.
                                if decision.meaningful_change_reason
                                in {REPRODUCED_PREVIOUS_CHANGE, RETURNED_TO_EARLIER_STATE}
                                else None
                            ),
                            fallback_question=decision.operator_question,
                            final_diagnostics=previous_findings,
                            attempt_authored_paths=attempt_authored_paths,
                            surviving_finding=longest_survivor[0] or None,
                            surviving_finding_attempts=longest_survivor[1],
                            tier_label=TIER_LABELS.get(PerformanceTier(state.performance_tier), ""),
                        ),
                    ),
                    stranded,
                )
            last_findings = findings
            # The next coding attempt must receive the exact bounded diagnostics that caused
            # this one to fail, ordered so it cannot mistake them for history. Merging every
            # attempt's findings into one deduplicated list -- which is what this did -- gave
            # -105's ninth attempt a pile of partly-contradictory instructions and no way to
            # tell which still applied.
            for item in findings:
                unresolved_since[item] = unresolved_since.get(item, 0) + 1
            superseded_findings = [
                item
                for item in dict.fromkeys([*superseded_findings, *previous_findings])
                if item not in findings
            ]
            # A finding the last attempt stopped reporting keeps no counter here, on purpose:
            # this map answers "how many attempts running has this been broken", and a demand
            # that is no longer being made has no such number. What it must not do is take the
            # history with it, which is what it used to do -- so the record that the demand was
            # once made and met now lives in the artifact lineage, where `resolved_issue_ledger`
            # reads it below and a crash, a resume or an integration-fix re-entry cannot erase
            # it. Nothing durable is being kept in this dictionary.
            for item in list(unresolved_since):
                if item not in findings:
                    del unresolved_since[item]
            # Persist this rejected attempt before starting another one. Without this append,
            # only the final loop outcome reached parent state, so a crash or later success
            # erased the code completion, review, and result that justified the retry.
            _append_artifacts(state, _execution_artifacts(execution))
            retry_feedback = _retry_feedback(
                list(findings) or list(previous_findings),
                unresolved_since=unresolved_since,
                attempt=child.retry_count + 1,
                inherited=inherited_feedback,
                superseded=superseded_findings,
                # Read after the append above, so this attempt's own result is part of the
                # lineage: what it stopped reporting is satisfied from here on, and telling the
                # next engineer to preserve it is what stops the next attempt undoing it.
                #
                # Decided questions come first and are exempt from the outstanding-demand
                # filter below them. That exemption is the point: the one case where a
                # requirement is both being demanded and must not be argued is a question a
                # person has already answered, and the answer is what settles it.
                invariants=[
                    *settled_question_lines(settled),
                    *satisfied_invariant_lines(
                        resolved_issue_ledger(
                            state.artifacts,
                            child_workflow_id=child.child_workflow_id,
                            repository_id=repository.repository_id,
                        ),
                        demanded=[*findings, *previous_findings, *inherited_feedback],
                        settled=settled_fingerprints,
                    ),
                ],
                # Reached only below the stop above, so this is the second consecutive attempt
                # to end in this exact failure and never the third. The next attempt is the
                # last one this shape gets, and it is told what the previous repair touched --
                # "try something else" against an unnamed previous attempt is advice that
                # rewriting the same file differently satisfies.
                strategy_change=(
                    strategy_change_line(unchanged_failure) if unchanged_failure is not None else ""
                ),
            )
            # A retry is granted, so the attempt clocks start here. Reset beside the
            # counter they describe, which is the only place the two cannot drift apart.
            attempt_clock_started = time.monotonic()
            attempt_fault_seconds = 0.0
            child = result_child.model_copy(
                update={
                    "retry_count": child.retry_count + 1,
                    counter_field: counter_value + 1,
                    "status": ChildWorkflowStatus.RUNNING,
                    "blocking_issues": previous_findings,
                    "checkpoint_boundary": WorkflowCheckpointBoundary.BEFORE_CODING,
                    "code_completion_artifact_id": (execution.result.code_completion_artifact_id),
                    "review_artifact_id": execution.result.review_artifact_id,
                }
            )
            # Only now, with the attempt already granted by the decision above, is the model
            # role for it selected. Deliberately after `decide_child_retry` and after the
            # convergence guards: an attempt that may not run is never routed, so another
            # configured model cannot revive a workstream the policy has stopped.
            child = self._route_attempt(
                state,
                repository=repository,
                child=child,
                contract=contract,
                request=_next_routing_request(execution, classification),
            )
            # Persist the retry decision before the next model/process call. Without this,
            # a crash between attempts reloads the previous counter and silently repeats the
            # same workspace reset, coding operation, and external side effects.
            state.child_workflows[repository.repository_id] = child
            await self._checkpoint(
                state, WorkflowCheckpointBoundary.BEFORE_CODING, repository.repository_id
            )

    async def _run_context_partition_wave(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repository: RepositorySpec,
        workstream: RepositoryWorkstreamPlan,
        technical_prd: TechnicalPRDArtifact,
        contract: IntegrationContractArtifact,
        credentials: RequestScopedCredentials,
    ) -> ChildExecution:
        """Run a refused retry as a wave of scoped attempts, one review after the last (80-).

        The wave is execution shape, not a new grant: the refused attempt's ``retry_count``
        increment was already decided when the retry was granted, the wave's first scoped
        attempt runs under it, and later clusters increment only ``context_partition_count``.
        No path in here consults the retry authority -- the one-retry-decision law applies
        verbatim; the first verdict-shaped decision after the refusal is the final cluster's
        review, judged by the ordinary loop.

        Every cluster runs on the same branch and preserved workspace. Intermediate clusters
        skip the review cycle -- the review judges the workstream, not the delta (66-), and a
        per-cluster review would feed the 77- theme counter and the 49- ledger with
        recurrences the partition itself manufactured. Their code completions are appended
        with partition-qualified artifact ids so the final attempt's lineage (and therefore
        publication) carries their files; their result records are control-flow only and are
        never persisted.

        The floor: a scoped attempt that itself raises ``RequiredContextRefusal``, or any
        intermediate cluster that fails a deterministic gate, ends the workstream through
        the existing failure path with the wave's plan cleared.
        """
        child = state.child_workflows[repository.repository_id]
        clusters = list(child.pending_context_partition or [])
        total = len(clusters)
        base_strategy = {
            key: value
            for key, value in (child.retry_strategy or {}).items()
            if key != "context_partition"
        }

        def scoped_strategy(index: int, cluster: dict[str, Any], *, final: bool) -> dict[str, Any]:
            return {
                **base_strategy,
                "context_partition": {
                    "index": index,
                    "total": total,
                    "final": final,
                    "blocking_diagnostics": list(cluster.get("diagnostics", [])),
                    "expected_files_or_areas": list(cluster.get("files", [])),
                },
            }

        async def persist(updated: ChildWorkflowReference) -> ChildWorkflowReference:
            state.child_workflows[repository.repository_id] = updated
            await self._checkpoint(
                state, WorkflowCheckpointBoundary.BEFORE_CODING, repository.repository_id
            )
            return updated

        def cleared(reference: ChildWorkflowReference) -> ChildWorkflowReference:
            return reference.model_copy(update={"pending_context_partition": None})

        for index, cluster in enumerate(clusters[:-1]):
            if cluster.get("status") == "done":
                continue
            await self._raise_if_cancelled(state)
            child = await persist(
                child.model_copy(
                    update={
                        "retry_strategy": scoped_strategy(index, cluster, final=False),
                        # Every cluster after the wave's first is a scoped attempt the one
                        # granted retry did not number; it spends this counter and nothing else.
                        "context_partition_count": child.context_partition_count
                        + (1 if index > 0 else 0),
                    }
                )
            )
            try:
                execution = await self._child_executor.run(
                    feature=state,
                    repository=repository,
                    workstream=workstream,
                    child=child,
                    technical_prd=technical_prd,
                    contract=contract,
                    feedback=list(cluster.get("diagnostics", [])),
                    credentials=credentials,
                )
            except RequiredContextRefusal as refusal:
                # The floor: even one cluster's own file set exceeds the model's window.
                # Ends exactly as an unpartitioned refusal does -- same sentences, same
                # classification, budget unspent by the refusal itself.
                child = await persist(cleared(child))
                return _failed_child_execution(
                    state,
                    repository=repository,
                    workstream=workstream,
                    child=child,
                    error=refusal,
                )
            if execution.result.metadata.get("context_partition_intermediate") is not True:
                # A deterministic gate (or any other terminal shape) rejected the scoped
                # attempt before the review stage. The wave ends and this execution is the
                # pass's recorded outcome, exactly as a failed unpartitioned attempt would be.
                child = await persist(cleared(child))
                return execution
            if execution.code_completion is not None:
                # Into the lineage under a partition-qualified identity: the final attempt's
                # completion merges these file changes, which is what publication commits.
                completion = execution.code_completion.model_copy(
                    update={
                        "artifact_id": attempt_artifact_id(
                            _repository_artifact_id(
                                ARTIFACT_FILENAMES["code_completion"],
                                f"{repository.repository_id}.partition-{index}",
                            ),
                            child.retry_count,
                        ),
                        "metadata": {
                            **execution.code_completion.metadata,
                            "child_attempt": child.retry_count,
                            "context_partition_index": index,
                        },
                    }
                )
                _append_artifacts(state, [completion])
            clusters[index] = {**cluster, "status": "done"}
            child = await persist(
                child.model_copy(update={"pending_context_partition": list(clusters)})
            )
            _LOGGER.info(
                "child_workstream_context_partition_cluster_completed",
                feature_id=state.feature_id,
                repository_id=repository.repository_id,
                stage="child_workstream",
                attempt=child.retry_count,
                cluster=index,
                clusters=total,
            )
        final_index = total - 1
        final_cluster = clusters[final_index]
        child = await persist(
            child.model_copy(
                update={
                    "retry_strategy": scoped_strategy(final_index, final_cluster, final=True),
                    # The final cluster narrows what the next engineer call inherits; the
                    # review that follows it still judges the whole workstream.
                    "blocking_issues": list(final_cluster.get("diagnostics", [])),
                    "context_partition_count": child.context_partition_count
                    + (1 if final_index > 0 else 0),
                }
            )
        )
        try:
            execution = await self._run_one_child(
                state,
                repository=repository,
                workstream=workstream,
                technical_prd=technical_prd,
                contract=contract,
                feedback=[],
                credentials=credentials,
            )
        except RequiredContextRefusal as refusal:
            child = await persist(cleared(state.child_workflows[repository.repository_id]))
            return _failed_child_execution(
                state,
                repository=repository,
                workstream=workstream,
                child=child,
                error=refusal,
            )
        await persist(cleared(state.child_workflows[repository.repository_id]))
        return execution

    def _route_attempt(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repository: RepositorySpec,
        child: ChildWorkflowReference,
        contract: IntegrationContractArtifact,
        request: _RoutingRequest,
    ) -> ChildWorkflowReference:
        """Record which configured model role executes this repository's next attempt.

        Repository scoped throughout: the durable routing decision read and written here
        belongs to one child, so a remediation in one repository cannot change another
        repository's selection and cannot cause a sibling to run at all.

        A decision already persisted for this attempt number is left exactly as it is. That is
        what makes recovery safe: after a crash between "routing decided" and "Engineer
        invoked", re-deciding could read different evidence or changed deployment configuration.
        """
        persisted = ModelRoutingDecision.from_persisted(child.model_routing)
        if persisted is not None and persisted.attempt == child.retry_count:
            return child
        outcome = self._model_router.route(
            inputs=ModelRoutingInputs(
                feature_id=state.feature_id,
                repository_id=repository.repository_id,
                execution_mode=request.execution_mode,
                attempt=child.retry_count,
                failure_classification=request.failure_classification,
                repository_revision=child.current_revision,
                finding_fingerprints=[item.fingerprint for item in request.classifications],
                validation_fingerprint=_validation_evidence_fingerprint(child),
                contract_version=contract.contract_version,
                repair_version=_applied_repair_fingerprint(child),
            ),
            classifications=request.classifications,
            persisted_decision=child.model_routing,
        )
        # Stamped from feature state rather than decided by the router: the tier is the
        # feature's pinned cost basis, and the record carries it so every attempt's cost is
        # attributable afterwards. On a reused decision this rewrites the same value -- the
        # tier never changes mid-feature. The setup id travels beside it for a custom
        # feature, as provenance; the resolved values on the decision are the record itself.
        decision = outcome.decision.model_copy(
            update={
                "performance_tier": PerformanceTier(state.performance_tier),
                "model_setup_id": state.model_setup_id,
            }
        )
        _LOGGER.info(
            "model_routing_decision",
            feature_id=state.feature_id,
            repository_id=repository.repository_id,
            stage="child_workstream",
            execution_mode=decision.execution_mode.value,
            classification=(
                decision.classification.value if decision.classification is not None else None
            ),
            failure_classification=(
                decision.failure_classification.value
                if decision.failure_classification is not None
                else None
            ),
            repair_scope=(
                decision.repair_scope.value if decision.repair_scope is not None else None
            ),
            role=decision.role.value,
            platform=decision.platform.value,
            performance_tier=decision.performance_tier.value,
            # Named, not resolved here: the model is whatever configuration binds to the role,
            # and a workflow log that quoted a model name from its own code would be the one
            # place model names leak back into orchestration.
            model=decision.model,
            reasoning=decision.reasoning,
            attempt=decision.attempt,
            findings=len(decision.finding_fingerprints),
            reason=decision.routing_reason,
        )
        return child.model_copy(update={"model_routing": decision.model_dump(mode="json")})

    async def _publish_pull_requests(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        contract: IntegrationContractArtifact,
        plan: RepositoryExecutionPlanArtifact,
        child_results: Sequence[ChildWorkflowResultArtifact],
        integration_approved: bool = True,
    ) -> FeatureWorkflowSnapshot:
        """Open a PR for every repository that passed its own review, completing only if all did."""
        transition_feature(
            state,
            FeatureWorkflowStatus.READY_FOR_PULL_REQUESTS,
            reason="Every repository that passed its own review has work ready to open.",
            agent="feature_workflow",
        )
        transition_feature(
            state,
            FeatureWorkflowStatus.CREATING_PULL_REQUESTS,
            reason="Pull requests are being opened for the repositories that passed review.",
            agent="github",
        )
        try:
            await self._raise_if_cancelled(state)
        except CancellationRequested:
            return state
        await self._checkpoint(state, WorkflowCheckpointBoundary.BEFORE_PR)
        result_by_repository = {item.repository_id: item for item in child_results}
        workstream_by_repository = {item.repository_id: item for item in plan.workstreams}
        workstream_by_id = {item.workstream_id: item for item in plan.workstreams}
        existing_by_repository = {
            repository_id: artifact
            for repository_id, child in state.child_workflows.items()
            if child.pull_request_artifact_id is not None
            for artifact in state.artifacts
            if isinstance(artifact, PullRequestArtifact)
            and artifact.artifact_id == child.pull_request_artifact_id
        }
        repositories_to_publish: list[RepositorySpec] = []
        children_to_publish: list[ChildWorkflowReference] = []
        results_to_publish: list[ChildWorkflowResultArtifact] = []
        optional_failures: list[str] = []
        required_failures: list[str] = []
        for repository in state.repository_specs:
            workstream = workstream_by_repository.get(repository.repository_id)
            if workstream is None:
                continue
            if repository.repository_id in existing_by_repository:
                continue
            child = state.child_workflows[repository.repository_id]
            result = result_by_repository.get(repository.repository_id)
            if result is not None and result.status == "approved" and result.pull_request_readiness:
                repositories_to_publish.append(repository)
                children_to_publish.append(child)
                results_to_publish.append(result)
                continue
            # The latest attempt failed, but an earlier one may have passed review and pushed
            # its commit: integration review sends an approved repository back, and the rework
            # can fail while the branch still holds the reviewed commit. Only an attempt that
            # actually passed review is eligible, so nothing unreviewed is ever published.
            superseded = _latest_approved_result(state, repository.repository_id)
            if superseded is not None:
                repositories_to_publish.append(repository)
                children_to_publish.append(child)
                results_to_publish.append(superseded)
            if repository.required:
                # Recorded, and the loop finishes before anything acts on it. Returning from
                # inside the loop is what discarded reviewed work twice -- r-2 finished and
                # approved its frontend and published nothing because the backend was still
                # failing, and r-3 did the same with the sides reversed -- because the
                # repositories after this one never reached the sorting at all.
                #
                # They still do. What the finished list feeds is now a hold rather than a
                # publication: reviewed work is still never discarded, and what it waits for
                # is a person rather than a pull request nobody asked for.
                required_failures.append(repository.repository_id)
                state.child_workflows[repository.repository_id] = child.model_copy(
                    update={
                        "blocking_issues": [
                            *child.blocking_issues,
                            (
                                "Required repository regressed after review; the commit that "
                                "passed review is on its branch and can be published on "
                                "request."
                                if superseded is not None
                                else "Required repository is not review-approved and PR-ready."
                            ),
                        ]
                    }
                )
                continue
            optional_failures.append(repository.repository_id)
        if required_failures:
            # The feature did not land, so nothing publishes itself. A frontend pull request
            # whose backend does not exist is not reviewable work -- it is a trap for whoever
            # opens it -- and the reviewed work is not thrown away either: it waits for the
            # one action that publishes it, alongside anything clean the reviewer rejected,
            # which the automatic path could never offer at all.
            #
            # The comment this replaces recorded the opposite decision and the two live runs
            # behind it (r-2 published its frontend while the backend failed; r-3 the same
            # with the sides reversed). Both are still true and neither is being undone: the
            # guarantee is that reviewed work always reaches a person, and what changes is
            # only that it reaches them through an advertised action instead of a pull
            # request nobody asked for. Section E of 81- is what makes that honest; if the
            # held state is ever quiet again, this is the line that became a defect.
            return await self._hold_publication_for_a_person(
                state,
                awaiting=[item.repository_id for item in repositories_to_publish],
                required_failures=required_failures,
            )
        try:
            newly_published = (
                await self._pull_request_publisher.publish(
                    feature=state,
                    repositories=repositories_to_publish,
                    children=children_to_publish,
                    child_results=results_to_publish,
                    contract=contract,
                    integration_approved=integration_approved,
                )
                if repositories_to_publish
                else []
            )
        except PartialPullRequestError as error:
            _append_artifacts(state, error.pull_requests)
            _mark_published_children(state, error.pull_requests)
            # Name the repositories whose publication failed. Without this the operator saw a
            # feature marked FAILED_REQUIRES_HUMAN with some pull requests open and nothing
            # saying which repository still needed one.
            for failed_repository_id in error.failed_repository_ids:
                failed_child = state.child_workflows.get(failed_repository_id)
                if failed_child is not None:
                    state.child_workflows[failed_repository_id] = failed_child.model_copy(
                        update={
                            "blocking_issues": [
                                *failed_child.blocking_issues,
                                "Pull-request creation did not succeed for this repository.",
                            ]
                        }
                    )
            transition_feature(
                state,
                FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
                reason=(
                    "Pull-request creation did not succeed for every repository that passed "
                    "its own review."
                ),
                agent="github",
            )
            state.updated_at = datetime.now(UTC)
            return ensure_feature_failure_summary(
                state,
                stage=FailureStage.PULL_REQUEST_PUBLICATION,
                classification=FeatureFailureClassification.PUBLICATION_FAILURE,
                # The error already names the repositories it could not publish, in a
                # sentence composed here rather than from a provider message. It reached
                # `blocking_issues` and stopped there; the summary an operator reads first
                # said only `unclassified_failure`.
                diagnostics=[
                    *error.diagnostics,
                    "The pull requests that were created are recorded on this feature; the "
                    "repositories named above still need one.",
                ],
            )
        except CancellationRequested:
            await self._mark_cancelled(state)
            return state
        except Exception as error:
            # Everything this handler knew went to structlog and nowhere else. Container logs
            # rotate; the database is the only durable record, and what it recorded was
            # `unclassified_failure` with an empty diagnostics array -- three of the fifty
            # failures measured for this task, each one a feature whose publication broke and
            # which could say nothing at all about it.
            attempted = sorted(item.repository_id for item in repositories_to_publish)
            classification = classification_of(error)
            _LOGGER.error(
                "pull_request_publication_failed",
                feature_id=state.feature_id,
                stage=FailureStage.PULL_REQUEST_PUBLICATION.value,
                agent="github",
                repository_ids=attempted,
                error_type=type(error).__name__,
                failure_classification=classification.value,
            )
            transition_feature(
                state,
                FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
                reason=(
                    "Pull-request publication raised before it could finish, so what reached "
                    "the provider is unconfirmed."
                ),
                agent="github",
            )
            state.updated_at = datetime.now(UTC)
            return ensure_feature_failure_summary(
                state,
                stage=FailureStage.PULL_REQUEST_PUBLICATION,
                classification=classification,
                diagnostics=[
                    (
                        "The model provider did not answer during pull-request publication."
                        if classification is FeatureFailureClassification.PROVIDER_UNAVAILABLE
                        else "The platform failed during pull-request publication and did "
                        "not anticipate the error it hit. This is a defect in the platform, "
                        "not in these repositories."
                    ),
                    (
                        f"Publication was running for: {', '.join(attempted)}."
                        if attempted
                        else "No repository had been published when this failed."
                    ),
                    # Only the same already-safe sources any other durable record reads.
                    *_causal_diagnostics(error),
                    "Check these repositories for a branch or pull request this run may have "
                    "left behind before starting the work again.",
                ],
            )
        _append_artifacts(state, newly_published)
        _mark_published_children(state, newly_published)
        pull_requests = [*existing_by_repository.values(), *newly_published]
        await self._checkpoint(state, WorkflowCheckpointBoundary.AFTER_PR)
        # A completion artifact is a claim that pull requests exist. Verify that claim against
        # the provider before making it, rather than trusting the create call's own report: a
        # publisher that failed silently, or an artifact carrying a URL nobody fetched, would
        # otherwise mark the feature complete with nothing on GitHub to show for it.
        unverified = await self._unverified_pull_requests(pull_requests)
        if unverified:
            transition_feature(
                state,
                FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
                reason=(
                    "A pull request this run reported creating could not be read back from "
                    "the provider, so no completion may claim the work is open for review."
                ),
                agent="github",
            )
            state.updated_at = datetime.now(UTC)
            for repository_name, detail in unverified:
                repository_id = next(
                    (
                        item.repository_id
                        for item in state.repository_specs
                        if _repository_name(str(item.repository_url)) == repository_name
                    ),
                    None,
                )
                verified_child = (
                    state.child_workflows.get(repository_id) if repository_id is not None else None
                )
                if verified_child is not None and repository_id is not None:
                    state.child_workflows[repository_id] = verified_child.model_copy(
                        update={"blocking_issues": [*verified_child.blocking_issues, detail]}
                    )
            return ensure_feature_failure_summary(
                state,
                stage=FailureStage.PULL_REQUEST_PUBLICATION,
                classification=FeatureFailureClassification.PULL_REQUEST_UNVERIFIED,
                diagnostics=[
                    "A pull request this run reported creating could not be read back from "
                    "the provider, so no completion may claim the work is open for review.",
                    *[detail for _, detail in unverified],
                ],
            )
        if not integration_approved:
            # Everything that passed its own review is now open as a draft pull request, but
            # the feature is not complete: the integration review never approved it, so no
            # completion artifact may claim that it did.
            transition_feature(
                state,
                FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
                reason=(
                    "Every repository that passed its own review is open as a draft pull "
                    "request, but the integration review never approved this feature."
                ),
                agent="github",
            )
            state.updated_at = datetime.now(UTC)
            return ensure_feature_failure_summary(
                state,
                stage=FailureStage.INTEGRATION_REVIEW,
                classification=(FeatureFailureClassification.INTEGRATION_REVIEW_UNSATISFIED),
                diagnostics=[
                    "Every repository that passed its own review has a draft pull request "
                    "open, but the integration review never approved them together.",
                    "Read the integration review's findings before merging any of them.",
                ],
            )
        # Strictly after every new pull request was created *and read back from the provider*,
        # and before the completion is written: the person must never be left with the old
        # pull request closed and no replacement open.
        await self._close_superseded_pull_requests(state)
        integration_review = _require_artifact(state.artifacts, IntegrationReviewArtifact)
        completion = create_artifact(
            FeatureCompletionArtifact,
            workflow_id=state.feature_id,
            # A revision writes its own completion beside the superseded run's; the
            # completion gate in `next_step` reads the revision stamp, not the id.
            artifact_id=_versioned_artifact_id(
                state,
                FeatureCompletionArtifact,
                FEATURE_ARTIFACT_FILENAMES["feature_completion"],
            ),
            producer="feature_workflow",
            payload={
                "feature_id": state.feature_id,
                "parent_workflow_id": state.workflow_id,
                "status": "completed",
                "technical_prd_artifact_id": _require_artifact(
                    state.artifacts, TechnicalPRDArtifact
                ).artifact_id,
                "integration_contract_artifact_id": contract.artifact_id,
                "repository_execution_plan_artifact_id": plan.artifact_id,
                "child_workflow_results": [
                    item.artifact_id for item in _latest_child_results(state)
                ],
                "integration_review_artifact_id": integration_review.artifact_id,
                "pull_request_artifact_ids": [item.artifact_id for item in pull_requests],
                "pull_request_urls": [item.url for item in pull_requests],
                "merge_strategy": plan.merge_strategy,
                "merge_order": [
                    item
                    for item in plan.execution_order
                    if workstream_by_id[item].repository_id
                    in {repository.repository_id for repository in repositories_to_publish}
                    | set(existing_by_repository)
                ],
                "deployment_strategy": plan.deployment_strategy,
                "feature_flags": plan.feature_flag_strategy,
                "rollback_plan": plan.rollback_strategy,
                "known_limitations": [
                    "Pull requests are not merged automatically.",
                    # A completed feature is the one artifact a non-engineer reads, and it
                    # names an approved integration review. That approval is contract
                    # conformance only, so without this the summary implies the seam between
                    # the repositories was reviewed by something. Nothing reviews it.
                    *assessment_coverage_limitations(integration_review),
                    *[
                        f"Optional repository '{repository_id}' did not produce a pull request."
                        for repository_id in optional_failures
                    ],
                ],
                "completed_at": datetime.now(UTC),
            },
            metadata={
                **_revision_metadata(state),
                "source_artifact_ids": [
                    contract.artifact_id,
                    plan.artifact_id,
                    integration_review.artifact_id,
                    *[item.artifact_id for item in pull_requests],
                ],
                "executor_build_revision": state.last_executor_build_revision,
                "workflow_schema_version": state.workflow_schema_version,
            },
        )
        _append_artifacts(state, [completion])
        transition_feature(
            state,
            FeatureWorkflowStatus.COMPLETED,
            reason=(
                "Every repository this feature needed has a pull request that was read back "
                "from the provider."
            ),
            agent="feature_workflow",
        )
        state.updated_at = datetime.now(UTC)
        return state

    async def _close_superseded_pull_requests(self, state: FeatureWorkflowSnapshot) -> None:
        """Close the pull requests this revision replaced, now that their replacements exist.

        Best-effort by design: every new pull request already exists and was read back, so a
        close that fails must not fail the feature. The failure is recorded on the repository
        (and the journaled operation's DEFER_TO_CREDENTIALED disposition lets a later
        credentialed pass settle it); the person is told to close the old one by hand.

        Idempotent across re-entry: a closed-marker artifact is appended per closed pull
        request, and an entry whose marker already exists is skipped -- so a crash between
        this and the completion write never comments or closes twice (the journal already
        guards the provider side; the marker guards the artifact side).
        """
        if not state.revision:
            return
        request = next(
            (
                artifact
                for artifact in reversed(state.artifacts)
                if isinstance(artifact, FeatureRevisionRequestArtifact)
                and artifact.revision == state.revision
            ),
            None,
        )
        if request is None:
            return
        pull_requests_by_id = {
            artifact.artifact_id: artifact
            for artifact in state.artifacts
            if isinstance(artifact, PullRequestArtifact)
        }
        known_ids = {artifact.artifact_id for artifact in state.artifacts}
        replacement_urls: dict[str, str] = {}
        for repository_id, child in state.child_workflows.items():
            if child.pull_request_artifact_id is not None:
                replacement = pull_requests_by_id.get(child.pull_request_artifact_id)
                if replacement is not None:
                    replacement_urls[repository_id] = str(replacement.url)
        for item in request.superseded:
            if item.pull_request_artifact_id is None or item.pull_request_number is None:
                continue
            superseded = pull_requests_by_id.get(item.pull_request_artifact_id)
            if superseded is None or superseded.state != "open":
                continue
            marker_id = f"{superseded.artifact_id.removesuffix('.json')}.closed.json"
            if marker_id in known_ids:
                continue
            replacement_url = replacement_urls.get(item.repository_id)
            try:
                if replacement_url is not None:
                    # The cross-link first, so whoever still has the old pull request open in
                    # a tab is pointed at its replacement. A failed comment does not stop the
                    # close: the new pull request's own description names the feature.
                    try:
                        await self._pull_request_publisher.comment_on_pull_request(
                            superseded.repository,
                            superseded.pull_request_number,
                            f"Superseded by V{state.revision + 1}: {replacement_url}",
                        )
                    except CancellationRequested:
                        raise
                    except Exception:  # noqa: BLE001 - a convenience comment is not a gate
                        _LOGGER.warning(
                            "superseded_pull_request_comment_failed",
                            feature_id=state.feature_id,
                            repository_id=item.repository_id,
                            pull_request_number=superseded.pull_request_number,
                        )
                await self._pull_request_publisher.close_pull_request(
                    superseded.repository, superseded.pull_request_number
                )
            except CancellationRequested:
                raise
            except Exception as error:  # noqa: BLE001 - a failed close must not fail the feature
                _LOGGER.error(
                    "superseded_pull_request_close_failed",
                    feature_id=state.feature_id,
                    stage="pull_request_publication",
                    agent="github",
                    repository_id=item.repository_id,
                    pull_request_number=superseded.pull_request_number,
                    error_type=type(error).__name__,
                )
                failed_child = state.child_workflows.get(item.repository_id)
                if failed_child is not None:
                    note = (
                        f"The superseded pull request #{superseded.pull_request_number} "
                        f"({superseded.url}) could not be closed automatically; close it "
                        "manually."
                    )
                    if note not in failed_child.blocking_issues:
                        state.child_workflows[item.repository_id] = failed_child.model_copy(
                            update={"blocking_issues": [*failed_child.blocking_issues, note]}
                        )
                continue
            _append_artifacts(
                state,
                [
                    superseded.model_copy(
                        update={
                            "artifact_id": marker_id,
                            "state": "closed",
                            "timestamp": datetime.now(UTC),
                            "metadata": {
                                **superseded.metadata,
                                "supersedes": superseded.artifact_id,
                                "closed_by_revision": state.revision,
                            },
                        }
                    )
                ],
            )
            known_ids.add(marker_id)

    async def _hold_publication_for_a_person(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        awaiting: Sequence[str],
        required_failures: Sequence[str],
    ) -> FeatureWorkflowSnapshot:
        """Stop before publishing anything and say, loudly, what is waiting on a person.

        The feature did not land, so publication is no longer the platform's decision. What
        this owes is therefore not silence but the opposite: a status a person is asked from,
        a reason naming both sides -- what is ready to open, and what never reached a review
        -- and a timeline event the feature's Slack thread carries. Without that last one this
        change is the 2026-08 defect wearing a different name.
        """
        ready = sorted(awaiting)
        failed = sorted(required_failures)
        # Repository ids are quoted exactly as this feature holds them -- no capitalisation,
        # no reflowing. A reader matches these against the workstream list.
        reason = (
            f"This feature did not land: {', '.join(failed)} did not reach a review it passed, "
            "so nothing was published automatically. "
            + (
                f"Ready to open on request: {', '.join(ready)}."
                if ready
                else "No repository passed review, so there is nothing ready to open."
            )
        )
        transition_feature(
            state,
            FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
            reason=reason,
            agent="feature_workflow",
        )
        state.updated_at = datetime.now(UTC)
        if self._event_writer is not None:
            await self._event_writer(
                state.feature_id,
                "feature_publication_held",
                {
                    "reason": reason,
                    "awaiting_publication": ready,
                    "required_failures": failed,
                },
            )
        # No stage, for the reason the path this replaces gave: publication did not fail, and
        # the repository that never reached it supplies both the stage and the classification.
        return ensure_feature_failure_summary(
            state,
            error_type=(FeatureFailureClassification.WORKSTREAM_UNFINISHED.value),
            diagnostics=[
                reason,
                "Nothing is published automatically for a feature that did not land. Publish "
                "this feature from the console when you have read what stopped it; that "
                "opens a pull request for everything that passed review, and for any "
                "repository whose checks all passed but whose review rejected it.",
            ],
        )

    async def _publish_awaiting_repositories(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        requested_by: str,
        reason: str,
    ) -> FeatureWorkflowSnapshot:
        """Open a pull request for everything a person has just decided should be opened.

        A sibling of `_publish_pull_requests` rather than a flag on it. That one owns
        feature-level policy -- required and optional sorting, four failure transitions, the
        completion artifact, the integration verdict -- and almost none of it applies here: a
        feature reaching this path has already failed, will still have failed afterwards, and
        is publishing work precisely because the automatic rules said not to.

        What the two share is the per-repository mechanics, and those are shared for real:
        this delegates to the same publisher, which keeps one copy of the retry-then-adopt
        logic that stops a transient provider fault becoming a permanent one.
        """
        contract = _require_artifact(state.artifacts, IntegrationContractArtifact)
        repositories: list[RepositorySpec] = []
        children: list[ChildWorkflowReference] = []
        results: list[ChildWorkflowResultArtifact] = []
        unreviewed_commit_shas: dict[str, str] = {}
        refused: list[str] = []
        for repository in state.repository_specs:
            child = state.child_workflows.get(repository.repository_id)
            if child is None:
                continue
            try:
                publication_class = require_publishable_workstream(
                    state, repository_id=repository.repository_id
                )
            except FeatureWorkflowError as error:
                refused.append(f"{repository.repository_id}: {error.diagnostics[0]}")
                continue
            result = _publishable_result(state, repository.repository_id, publication_class)
            if result is None:  # pragma: no cover - the raiser proves one exists
                continue
            if publication_class is WorkstreamPublicationClass.UNREVIEWED:
                unreviewed_commit_shas[repository.repository_id] = await self._commit_and_push(
                    state, repository=repository, child=child, result=result
                )
            repositories.append(repository)
            children.append(child)
            results.append(result)
        # One repository's provider fault must not lose another's pull request. `publish`
        # opens each repository independently and reports the set that failed by raising --
        # carrying the artifacts for the ones that did open. Letting that propagate would
        # skip the two lines below, so a draft pull request would exist on GitHub and this
        # feature's record would deny it: the exact shape the publication invariant exists
        # to prevent, reached through the path built to honour it.
        creation_failures: list[str] = []
        creation_diagnostics: list[str] = []
        try:
            published = await self._pull_request_publisher.publish(
                feature=state,
                repositories=repositories,
                children=children,
                child_results=results,
                contract=contract,
                # Never. A feature reaching this path failed, and the manual publication of
                # its parts is not an integration verdict about them.
                integration_approved=False,
                unreviewed_commit_shas=unreviewed_commit_shas,
            )
        except PartialPullRequestError as error:
            published = list(error.pull_requests)
            creation_failures = sorted(error.failed_repository_ids)
            creation_diagnostics = list(error.diagnostics)
        _append_artifacts(state, published)
        _mark_published_children(state, published)
        for failed_repository_id in creation_failures:
            failed_child = state.child_workflows.get(failed_repository_id)
            if failed_child is not None:
                state.child_workflows[failed_repository_id] = failed_child.model_copy(
                    update={
                        "blocking_issues": [
                            *failed_child.blocking_issues,
                            "Pull-request creation did not succeed for this repository when "
                            "this feature was published; pressing publish again adopts "
                            "whatever reached the provider rather than opening a second one.",
                        ]
                    }
                )
        # What actually opened, never what was requested. A sentence naming a repository whose
        # create failed would be the same false record the handler above exists to prevent.
        opened_names = {item.repository for item in published}
        opened_repository_ids = sorted(
            item.repository_id
            for item in state.repository_specs
            if _repository_name(str(item.repository_url)) in opened_names
        )
        opened = sorted(opened_names)
        outcome = (
            f"{requested_by} published this feature's finished work: "
            f"{', '.join(opened)}." + (f" Reason given: {reason}" if reason.strip() else "")
            if opened
            else f"{requested_by} asked to publish this feature and nothing was opened."
        )
        if creation_failures:
            outcome += (
                f" Pull-request creation did not succeed for {', '.join(creation_failures)}, "
                "which still need one."
            )
        transition_feature(
            state,
            FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
            reason=outcome,
            agent="github",
        )
        state.updated_at = datetime.now(UTC)
        # The summary described the hold this request has just answered, and it is cleared
        # whether the publication was whole or partial.
        #
        # Not clearing it looks like the safer half of the choice -- the hold's summary is
        # where "what is still owed" is written, and a partial publication has not answered
        # it. But `ensure_feature_failure_summary` is write-once: leaving the old one in place
        # does not preserve a sentence, it *suppresses* the new one, so the summary an
        # operator reads first would say nothing at all about the create that failed. That is
        # the defect the automatic path's own handler was written to fix. So the old summary
        # goes and the replacement carries both halves: what opened, and what is still owed.
        state.failure_summary = None
        unreviewed_opened = sorted(set(unreviewed_commit_shas) & set(opened_repository_ids))
        if self._event_writer is not None:
            await self._event_writer(
                state.feature_id,
                "feature_published_by_person",
                {
                    "reason": outcome,
                    "requested_by": requested_by,
                    "published": opened,
                    "unreviewed": unreviewed_opened,
                    "creation_failed": creation_failures,
                },
            )
        return ensure_feature_failure_summary(
            state,
            stage=(FailureStage.PULL_REQUEST_PUBLICATION if creation_failures else None),
            classification=(
                FeatureFailureClassification.PUBLICATION_FAILURE if creation_failures else None
            ),
            error_type=(FeatureFailureClassification.WORKSTREAM_UNFINISHED.value),
            diagnostics=[
                outcome,
                *(
                    [
                        "Published without an approving repository review, at a person's "
                        f"request: {', '.join(unreviewed_opened)}."
                    ]
                    if unreviewed_opened
                    else []
                ),
                *creation_diagnostics,
                *(
                    [
                        "Press publish again for the repositories that did not reach a pull "
                        "request. Nothing is opened twice: a create that already reached the "
                        "provider is adopted rather than repeated."
                    ]
                    if creation_failures
                    else []
                ),
                *(
                    ["Not offered for publication: " + "; ".join(sorted(refused))]
                    if refused
                    else []
                ),
            ],
        )

    async def _commit_and_push(
        self,
        state: FeatureWorkflowSnapshot,
        *,
        repository: RepositorySpec,
        child: ChildWorkflowReference,
        result: ChildWorkflowResultArtifact,
    ) -> str:
        """Commit a rejected attempt's worktree and push it, so there is something to open.

        A rejected attempt commits nothing, so this class of publication is not "adopt a
        branch that already exists": it is commit, then push, then open. Three journaled steps
        no existing path performs, which is why this is a helper rather than a flag on one.

        The commit stages `result.changed_files` explicitly -- the adapter's signature takes a
        file list and never `-a`, so the set has to be enumerated -- and the push is never
        forced. The branch is this workstream's own and holds no commit anybody else made, so
        there is nothing a force push would buy and `GitSafetyError` is what it would cost.
        """
        if self._git_service_factory is None:
            msg = "this composition cannot publish an unreviewed workstream"
            raise FeatureWorkflowError(
                msg,
                diagnostic=(
                    "Publishing a workstream the review rejected needs a Git service, and this "
                    "deployment's feature runner was built without one."
                ),
            )
        git = self._git_service_factory(repository, child)
        paths = sorted({item.path for item in result.changed_files})
        commit_sha = await _await_if_needed(
            git.commit(
                result.workspace_path,
                f"{state.reference or state.feature_id}: {state.title}",
                files=paths,
            )
        )
        if not isinstance(commit_sha, str) or not commit_sha.strip():
            msg = f"git returned no commit for {repository.repository_id}"
            raise FeatureWorkflowError(
                msg,
                diagnostic=(
                    "The Git service reported no commit for a workstream this platform was "
                    "publishing on a person's instruction."
                ),
            )
        await _await_if_needed(
            git.push(
                result.workspace_path,
                child.branch_name,
                force=False,
                expected_commit_sha=commit_sha,
            )
        )
        return commit_sha

    async def _unverified_pull_requests(
        self, pull_requests: Sequence[PullRequestArtifact]
    ) -> list[tuple[str, str]]:
        """Fetch each published pull request back and report every one that cannot be confirmed."""
        finder = getattr(self._pull_request_publisher, "find_pull_request", None)
        if finder is None:
            return [
                (
                    artifact.repository,
                    "The pull-request publisher cannot fetch provider state for verification; "
                    "the feature is not complete.",
                )
                for artifact in pull_requests
            ]
        unverified: list[tuple[str, str]] = []
        for artifact in pull_requests:
            try:
                found = await _await_if_needed(
                    finder(
                        artifact.repository,
                        source_branch=artifact.source_branch,
                        target_branch=artifact.target_branch,
                        title=artifact.title,
                    )
                )
            except Exception as error:  # noqa: BLE001 - provider faults are reported, not raised
                unverified.append(
                    (
                        artifact.repository,
                        f"Pull request verification for {artifact.repository} raised "
                        f"{type(error).__name__}; the feature is not complete until the pull "
                        "request can be fetched back from the provider.",
                    )
                )
                continue
            if found is None:
                unverified.append(
                    (
                        artifact.repository,
                        f"The pull request recorded for {artifact.repository} at {artifact.url} "
                        f"could not be fetched back from the provider for branch "
                        f"{artifact.source_branch} into {artifact.target_branch}.",
                    )
                )
            elif getattr(found, "head_sha", None) != artifact.commit_sha:
                unverified.append(
                    (
                        artifact.repository,
                        f"The pull request for {artifact.repository} points to head "
                        f"{getattr(found, 'head_sha', None) or 'unknown'}, not the approved "
                        f"commit {artifact.commit_sha}.",
                    )
                )
        return unverified

    async def _is_cancelled(self) -> bool:
        """Use the durable token when supplied while retaining state-only mock cancellation."""
        return (
            self._cancellation_token is not None and await self._cancellation_token.is_cancelled()
        )

    async def _raise_if_cancelled(self, state: FeatureWorkflowSnapshot) -> None:
        """Convert a durable cancellation signal into a persisted parent cancellation boundary."""
        if state.cancellation_requested or await self._is_cancelled():
            await self._mark_cancelled(state)
            raise CancellationRequested("feature cancellation requested")

    async def _sleep_between_faults(self, state: FeatureWorkflowSnapshot, seconds: float) -> None:
        """Serve a fault backoff in slices so a cancellation does not wait out the whole tail.

        Nothing is in flight while this runs -- the attempt has already failed -- so stopping
        mid-backoff destroys no external effect, which is the one thing the fault loops are
        otherwise careful never to interrupt. Both loops check cancellation only between
        attempts, so without this the Git tail's final 160-second wait would be 160 seconds a
        cancelled feature keeps running for.
        """
        remaining = seconds
        while True:
            # Always at least one `sleep`, including for the zero the tests set: it is also
            # the yield point that lets a concurrent sibling workstream make progress.
            await asyncio.sleep(max(0.0, min(remaining, _FAULT_BACKOFF_SLICE_SECONDS)))
            remaining -= _FAULT_BACKOFF_SLICE_SECONDS
            if remaining <= 0:
                return
            await self._raise_if_cancelled(state)

    async def _mark_cancelled(self, state: FeatureWorkflowSnapshot) -> None:
        """Preserve completed child work while preventing future commit, push, and PR effects."""
        state.cancellation_requested = True
        transition_feature(
            state,
            FeatureWorkflowStatus.CANCELLED,
            reason=(
                "A cancellation reached this feature between steps, so no further commit, "
                "push or pull request will be attempted for it."
            ),
            agent="feature_workflow",
        )
        settle_unfinished_children(state, ChildWorkflowStatus.CANCELLED)
        state.updated_at = datetime.now(UTC)
        await self._checkpoint(state, WorkflowCheckpointBoundary.AFTER_VALIDATION)

    async def _checkpoint(
        self,
        state: FeatureWorkflowSnapshot,
        boundary: WorkflowCheckpointBoundary,
        repository_id: str | None = None,
    ) -> None:
        """Durably persist each parent or child safe boundary before later side effects begin."""
        key = repository_id or "parent"
        state.checkpoint_boundaries[key] = boundary
        state.updated_at = datetime.now(UTC)
        if self._checkpoint_writer is not None:
            await self._checkpoint_writer(state.model_copy(deep=True), boundary, repository_id)


# The three statuses nothing resumes. Deliberately narrower than `TERMINAL_FEATURE_STATUSES`,
# which also holds the two failed ones: those *are* terminal in the sense that this platform
# will not act on them unasked, but a person may resume either, and `require_resumable_feature`
# has always refused exactly these three and no more. `next_step` answers what a feature's
# evidence still calls for; whether the platform may act on that answer without being asked is
# the queue's decision, and `_CONTINUABLE_STATUSES` is where it is made.
_UNRESUMABLE_STATUSES = frozenset(
    {
        FeatureWorkflowStatus.COMPLETED,
        FeatureWorkflowStatus.CANCELLED,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    }
)


class FeatureStep(StrEnum):
    """The unit of work that advances a feature, named after the boundary it ends at.

    Every member ends where the platform already writes a checkpoint, and every member is
    idempotent given the state it starts from: running it twice over the same persisted
    evidence produces the same artifacts and no second external effect. That is the property
    that makes a step a safe amount of work to lose to a crash, and it is why the taxonomy
    follows `WorkflowCheckpointBoundary` rather than the orchestrator's call graph.

    Reconnaissance is deliberately not a member. It reads every checkout and can be slow, so
    the loss bound would be worth having -- but the only durable evidence it leaves is its
    artifacts, and a composition without a reconnaissance capability produces none at all.
    "Has reconnaissance run?" would then have to be answered from something written for the
    purpose, which is the second source of truth this whole decision exists to avoid. It
    belongs to `PLAN`, whose checkpoint boundary it already shares.
    """

    # Before the product manager, because the design is part of the request. Reached only by
    # a feature that cited one: for every other feature `next_step` never names it, so
    # nothing about their lifecycle changes.
    RESOLVE_DESIGN = "resolve_design"
    ANALYZE_PRD = "analyze_prd"
    PLAN = "plan"
    EXECUTE_WORKSTREAMS = "execute_workstreams"
    INTEGRATION_REVIEW = "integration_review"
    PUBLISH = "publish"


class NoStep(StrEnum):
    """Why a feature has no next step, as evidence rather than as an absence.

    `next_step` is total: for any persisted state it either names work or names one of these.
    A caller that reads `None` and cannot say why would have to guess between "this feature
    finished", "somebody owns it" and "it is being cancelled", and those need opposite
    responses.
    """

    FEATURE_IS_TERMINAL = "feature_is_terminal"
    CANCELLATION_IN_PROGRESS = "cancellation_in_progress"
    AWAITING_CLARIFICATION_ANSWERS = "awaiting_clarification_answers"
    AWAITING_CONTRACT_DECISION = "awaiting_contract_decision"
    EVERY_STEP_IS_DONE = "every_step_is_done"


@dataclass(frozen=True, slots=True)
class StepDecision:
    """What to do next with one feature, or the reason there is nothing to do.

    Exactly one of `step` and `no_step_reason` is set, enforced on construction. `explanation`
    is the sentence a timeline or a log entry can carry without a reader having to know the
    enum.
    """

    step: FeatureStep | None
    no_step_reason: NoStep | None
    explanation: str

    def __post_init__(self) -> None:
        """Reject a decision that names both a step and a reason there is none, or neither."""
        if (self.step is None) == (self.no_step_reason is None):
            msg = "a step decision names exactly one of a step or a reason there is none"
            raise ValueError(msg)


def _step(step: FeatureStep, explanation: str) -> StepDecision:
    """Name the work a claim should do."""
    return StepDecision(step=step, no_step_reason=None, explanation=explanation)


def _no_step(reason: NoStep, explanation: str) -> StepDecision:
    """Name the reason this feature is not owed any work right now."""
    return StepDecision(step=None, no_step_reason=reason, explanation=explanation)


def _latest_artifact[ArtifactModel: Artifact](
    artifacts: Sequence[Artifact], artifact_class: type[ArtifactModel]
) -> ArtifactModel | None:
    """Return the newest artifact of one type, or `None` where `_require_artifact` would raise.

    `next_step` asks about artifacts that may legitimately be absent -- that absence is the
    evidence it reads -- so it cannot use the raising accessor the orchestrator's execution
    paths use, where a missing artifact really is a broken feature.
    """
    for artifact in reversed(artifacts):
        if isinstance(artifact, artifact_class):
            return artifact
    return None


def _artifact_revision(artifact: Artifact) -> int:
    """Return which feature revision produced this artifact.

    Absent reads as the original run: every artifact written before revisions existed
    carries no stamp, and 0 is what those features' `revision` counter reads as -- so a
    never-revised feature's decisions are byte-identical to what they were.
    """
    value = artifact.metadata.get("feature_revision", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _revision_metadata(state: FeatureWorkflowSnapshot) -> dict[str, Any]:
    """The stamp a revision run's artifacts carry, and nothing at all for the original run.

    Empty rather than `{"feature_revision": 0}` on purpose: the original run's artifacts
    must stay byte-identical to what every feature before revisions wrote, and the readers
    all treat an absent stamp as 0.
    """
    return {"feature_revision": state.revision} if state.revision else {}


def _versioned_artifact_id(
    state: FeatureWorkflowSnapshot, artifact_class: type[BaseArtifact], base_artifact_id: str
) -> str:
    """Return the base id for the first artifact of a type, and a `.v{N}` id afterwards.

    Count-based, counting instances of the type already in the history, so it composes with
    the contract-change path's own `.v{N}` numbering (which is count-equivalent: one original
    plus one per approved change) and can never re-mint an id `_append_artifacts` would
    refuse. A revision's planning pass is the second producer of these types; without this,
    its planner output would collide with the original run's fixed ids.
    """
    count = sum(isinstance(item, artifact_class) for item in state.artifacts)
    if count == 0:
        return base_artifact_id
    stem = base_artifact_id.removesuffix(".json")
    return f"{stem}.v{count + 1}.json"


def _awaiting_contract_decision(state: FeatureWorkflowSnapshot) -> bool:
    """Report whether a contract owner still has a change request in front of them.

    A child that asks to revise the contract stops the feature until a person approves or
    rejects the request, and no step of any kind may run in the meantime: the contract is the
    input every remaining step reads.
    """
    open_requests: dict[str, str] = {}
    for artifact in state.artifacts:
        if isinstance(artifact, ContractChangeRequestArtifact):
            open_requests[artifact.change_request_id] = artifact.status
    return any(status == "pending" for status in open_requests.values())


def _schedulable_workstream_ids(
    state: FeatureWorkflowSnapshot, plan: RepositoryExecutionPlanArtifact
) -> set[str]:
    """Return the workstreams an untargeted execution pass would still schedule.

    The same three exclusions `_execute_workstreams` applies, read from persisted state: a
    repository the plan does not name has no workstream, an approved or completed one is
    done, and one whose retry decision is already terminal stopped for a reason that another
    attempt cannot change.

    Dependency readiness is deliberately not applied here. The plan is acyclic -- enforced
    before any repository is cloned -- so a non-empty remainder always contains at least one
    workstream whose dependencies have all left it, and the scheduler is what walks the
    waves. This answers only whether there is anything left for it to walk.

    The fourth exclusion is the attempt itself. A repository whose current attempt has already
    produced its immutable result has spent it, and running it again at the same number would
    write a second `011_child_workflow_result.<repository>.attempt-N.json` -- which the
    artifact history refuses, ending the whole feature on a platform error. That was reachable
    before stepping, on any untargeted resume of a feature whose child failed through the
    fan-out's own exception handler rather than through its retry loop; a claim that asks this
    question between every step reaches it in the ordinary course of events. What buys the
    next attempt is a grant or an integration remediation, and both raise the number first.
    """
    return {
        item.workstream_id
        for item in plan.workstreams
        if item.repository_id in state.child_workflows
        and state.child_workflows[item.repository_id].status
        not in {ChildWorkflowStatus.APPROVED, ChildWorkflowStatus.COMPLETED}
        and not _child_retry_decision_is_terminal(state, state.child_workflows[item.repository_id])
        and not _attempt_already_produced_a_result(state, item.repository_id)
    }


def _design_resolution_is_owed(state: FeatureWorkflowSnapshot) -> bool:
    """Whether this feature cited a design that has not been resolved into a snapshot.

    Two questions and no third: did somebody cite a design, and is there a snapshot. A feature
    that cited nothing answers False on the first and never asks the second, which is what
    keeps every citation-free feature's lifecycle byte-identical -- no extra step, no operation
    row, no artifact.

    Read from the artifacts rather than from a stored flag, for `next_step`'s own reason: the
    evidence already exists, and a column restating it would be a second source of truth for
    where a feature is.
    """
    prd = _latest_artifact(state.artifacts, PRDArtifact)
    if prd is None or not prd.design_references:
        return False
    return _latest_artifact(state.artifacts, DesignSnapshotArtifact) is None


def _attempt_already_produced_a_result(state: FeatureWorkflowSnapshot, repository_id: str) -> bool:
    """Report whether this repository's current attempt has already recorded its outcome."""
    child = state.child_workflows[repository_id]
    return any(
        isinstance(artifact, ChildWorkflowResultArtifact)
        and artifact.repository_id == repository_id
        and artifact.metadata.get("child_retry_count") == child.retry_count
        for artifact in state.artifacts
    )


def _unpublished_publishable_repository_ids(state: FeatureWorkflowSnapshot) -> set[str]:
    """Return every repository that has review-approved work and no pull request yet."""
    return {
        result.repository_id
        for result in _latest_approved_child_results(state)
        if result.repository_id in state.child_workflows
        and state.child_workflows[result.repository_id].pull_request_artifact_id is None
    }


def next_step(state: FeatureWorkflowSnapshot) -> StepDecision:
    """Decide what one feature owes next, from its persisted state and nothing else.

    Pure and free of I/O by construction: every question it asks is answered by the snapshot's
    own artifacts, child rows and counters. That is what makes it cheap enough to ask on every
    claim and exhaustive enough to test as a table.

    There is deliberately no stored "current step". The evidence already exists -- which
    artifacts were written, which children settled, whether the integration verdict was
    charged -- and a column that restated it would be a second source of truth for where a
    feature is, which is the defect class this platform has already been bitten by four
    times. The order below is the order checkpoint recovery established: a feature can
    only be somewhere its predecessors' artifacts allow.
    """
    if state.status in _UNRESUMABLE_STATUSES:
        return _no_step(
            NoStep.FEATURE_IS_TERMINAL,
            f"This feature is {state.status.value} and will not be advanced further.",
        )
    if state.cancellation_requested or state.status is FeatureWorkflowStatus.CANCELLING:
        # A cancellation is already ending the feature and records its own cleanup
        # requirements as it goes. Scheduling ordinary work on top of it would discard them.
        return _no_step(
            NoStep.CANCELLATION_IN_PROGRESS,
            "This feature is being cancelled, so no further step will be scheduled.",
        )
    if _design_resolution_is_owed(state):
        # Before the technical-PRD check, because the requirements the product manager derives
        # should be derived from the frames too. The artifact's absence is the evidence the
        # step is owed -- there is deliberately no stored current step -- so this is total for
        # any persisted state and idempotent given it.
        return _step(
            FeatureStep.RESOLVE_DESIGN,
            "This feature cites a design that has not been resolved into a snapshot yet.",
        )
    technical_prd = _latest_artifact(state.artifacts, TechnicalPRDArtifact)
    if technical_prd is None:
        return _step(
            FeatureStep.ANALYZE_PRD,
            "This feature has no technical PRD, so the product manager runs first.",
        )
    asked_of_a_human = state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
    if technical_prd.unresolved_questions and asked_of_a_human:
        # Open questions stop this feature only once they have actually been put in front of
        # somebody, which is what `waiting_for_human` records. Before that they are the
        # product manager's questions, asked before any repository had been read, and the
        # planning step below is what reads the checkouts, answers what they settle, adds the
        # ones only a repository can raise, and *then* asks. Gating on the questions alone --
        # which is what this did while nothing but recovery consulted it -- meant a feature
        # advanced step by step stopped before reconnaissance ever ran, and asked its author
        # everything except the questions worth asking.
        return _no_step(
            NoStep.AWAITING_CLARIFICATION_ANSWERS,
            f"This feature is waiting on {len(technical_prd.unresolved_questions)} "
            "clarification answer(s).",
        )
    if _awaiting_contract_decision(state):
        return _no_step(
            NoStep.AWAITING_CONTRACT_DECISION,
            "This feature is waiting on a decision about a contract change request.",
        )
    contract = _latest_artifact(state.artifacts, IntegrationContractArtifact)
    plan = _latest_artifact(state.artifacts, RepositoryExecutionPlanArtifact)
    if (
        contract is None
        or plan is None
        # A revision plans again against its own requirement set. The superseded run's
        # contract and plan stay in the history but read as absent here, exactly as a
        # missing one would -- the stamp is what distinguishes them, and artifacts written
        # before revisions existed read as revision 0.
        or _artifact_revision(contract) != state.revision
        or _artifact_revision(plan) != state.revision
    ):
        return _step(
            FeatureStep.PLAN,
            "This feature has no integration contract and execution plan yet.",
        )
    schedulable = _schedulable_workstream_ids(state, plan)
    if schedulable:
        return _step(
            FeatureStep.EXECUTE_WORKSTREAMS,
            f"{len(schedulable)} repository workstream(s) still have work an attempt could do.",
        )
    child_results = _latest_approved_child_results(state)
    ready_repository_ids = {item.repository_id for item in child_results}
    missing_required = [
        item.repository_id
        for item in plan.workstreams
        if item.required and item.repository_id not in ready_repository_ids
    ]
    if missing_required and not child_results:
        # Nothing survived review, so there is nothing to review together and nothing to
        # publish. The integration step is still what records that, because it is the step
        # that owns the transition and the diagnosis it writes.
        return _step(
            FeatureStep.INTEGRATION_REVIEW,
            "No repository produced a review-approved result, which the integration step "
            "records as this feature's stop.",
        )
    merge_order = [
        item.workstream_id
        for workstream_id in plan.execution_order
        for item in plan.workstreams
        if item.workstream_id == workstream_id and item.repository_id in ready_repository_ids
    ]
    review = _persisted_integration_review(
        state,
        input_signature=_integration_review_input_signature(
            contract=contract, child_results=child_results, merge_order=merge_order
        ),
    )
    if review is None:
        return _step(
            FeatureStep.INTEGRATION_REVIEW,
            "The repositories that passed their own review have not been reviewed together "
            "against the current contract.",
        )
    if (
        review.review_status != "approved"
        and not missing_required
        and _integration_cycle_uncharged(state)
    ):
        # The verdict is durable but its cycle has not been spent and its findings have not
        # been routed. Both belong to the integration step, so a crash between writing the
        # verdict and acting on it re-enters there rather than skipping it.
        return _step(
            FeatureStep.INTEGRATION_REVIEW,
            "The latest integration review asked for changes that have not been routed "
            "to a repository yet.",
        )
    completion = _latest_artifact(state.artifacts, FeatureCompletionArtifact)
    if _unpublished_publishable_repository_ids(state) or (
        # A revision owes its own completion. Without the stamp check, the superseded
        # run's completion would satisfy this gate and a crash between the revision's
        # pull requests and its completion artifact would strand the feature at
        # `creating_pull_requests` with nothing left to schedule.
        completion is None or _artifact_revision(completion) != state.revision
    ):
        return _step(
            FeatureStep.PUBLISH,
            "Work that passed review is not open for review by a human yet.",
        )
    return _no_step(
        NoStep.EVERY_STEP_IS_DONE,
        "Every step this feature's persisted state calls for has already been done.",
    )


# The statuses at which the platform stops scheduling steps of its own accord. Deliberately
# not the complement of `_ACTIVELY_RUNNING_STATUSES` in the store, which answers a different
# question -- "is something executing this right now", for a sweep -- and which counts
# `contract_ready` and `changes_requested` as resting. Both of those are *mid-run*: a feature
# reaches them between two steps of a run nobody has paused, and treating them as rest would
# strand every feature the moment it finished planning.
_FEATURE_AT_REST_STATUSES = frozenset(
    {
        FeatureWorkflowStatus.COMPLETED,
        FeatureWorkflowStatus.CANCELLED,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
        FeatureWorkflowStatus.FAILED,
        FeatureWorkflowStatus.WAITING_FOR_HUMAN,
    }
)


def feature_is_at_rest(state: FeatureWorkflowSnapshot) -> bool:
    """Report whether this feature is waiting on a person rather than on a step.

    The companion to `next_step`, and deliberately a separate question. `next_step` answers
    what a feature's evidence still calls for, which a *resume* must be able to act on from a
    failed status -- that is how a person picks a stopped feature back up. This answers
    whether the platform may act on it unasked, and the two must not be collapsed: a single
    function would have to either refuse to continue a failed feature, breaking resume, or
    continue one unasked, which is how a feature that stopped for a human keeps running.

    Two conditional cases now, and they used to be one.

    The older one: a feature that failed while a repository still holds reviewed, pushed,
    unpublished work is not at rest, because publication has not happened yet. Stopping there
    is what threw finished work away twice -- a required sibling failed, and two approved
    repositories with pushed branches opened nothing. Unless publication is what failed; then
    the unpublished work is unpublished *because* the attempt to publish it did not succeed,
    and offering it publication again would re-enter the step that just broke for as long as
    anything kept asking. A person's resume clears that summary and is how the step is
    re-entered deliberately.

    The newer one, from 81-: a feature *holding publication for a person* is at rest, however
    much unpublished approved work it has. Automatic publication now narrows to the feature
    that fully landed, so for one that did not, unpublished approved work is the expected
    resting condition rather than evidence a step was skipped -- it is waiting on
    `PUBLISH_FEATURE`. Without this the queue would re-claim the feature forever, because
    `next_step` keeps answering `publish` and publication keeps declining to publish.

    Deliberately narrower than "every `failed_requires_human` feature is at rest", which
    would have been the shorter edit. A feature whose repositories all landed and whose
    *publication* broke is not held for anybody: its missing pull requests are a step's to
    open, and collapsing the two would strand it on the transient provider fault this
    platform has been bitten by more than once.

    The guarantee the older rule protected is not gone. Reviewed work still always reaches a
    human; what changed is the delivery, from a pull request nobody asked for to an
    advertised action -- and `_has_unpublished_approved_child` is still what answers whether
    that action has anything to do.
    """
    if state.status in _FEATURE_AT_REST_STATUSES:
        return True
    if state.status is not FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN:
        return False
    summary = state.failure_summary
    if summary is not None and summary.stage == FailureStage.PULL_REQUEST_PUBLICATION.value:
        return True
    if publication_is_held(state):
        return True
    return not _has_unpublished_approved_child(state)


def publication_is_held(state: FeatureWorkflowSnapshot) -> bool:
    """Report whether this feature is waiting on a person to decide about publication.

    True exactly when publication is the only thing this feature still owes and
    `_publish_pull_requests` would refuse to do it: some required repository has no pull
    request and its latest attempt did not pass review. Both halves are needed. The second
    mirrors the publication step's own sorting, deliberately, because this has to answer for
    a feature nobody is currently publishing and the step cannot be asked from here. The
    first is what keeps a feature that stopped in its workstreams from being called "held"
    before the integration review it still owes has run -- publication is decided last, and a
    feature is not waiting on a person for it until everything before it is done.
    """
    if next_step(state).step is not FeatureStep.PUBLISH:
        return False
    plan = _latest_artifact(state.artifacts, RepositoryExecutionPlanArtifact)
    if plan is None:
        return False
    planned = {item.repository_id for item in plan.workstreams}
    for repository in state.repository_specs:
        child = state.child_workflows.get(repository.repository_id)
        if (
            not repository.required
            or child is None
            or repository.repository_id not in planned
            or child.pull_request_artifact_id is not None
        ):
            continue
        result = _latest_result_for(state, repository.repository_id)
        if result is None or result.status != "approved" or not result.pull_request_readiness:
            return True
    return False


def feature_may_be_resumed(state: FeatureWorkflowSnapshot) -> bool:
    """Report whether a resume of this feature could run anything at all.

    The same two conditions `require_resumable_feature` raises on, as a question rather than
    a refusal, because the queue has to ask it without provoking an exception: a claim that
    would refuse is a claim the entry must not be reopened for. Re-queueing into one produces
    a feature that is asked and declines, over and over, until a budget stops it.
    """
    if state.status in _UNRESUMABLE_STATUSES:
        return False
    return not state.child_workflows or bool(resume_eligible_repository_ids(state))


def require_resumable_feature(state: FeatureWorkflowSnapshot) -> None:
    """Refuse a resume that has nothing it could possibly run.

    Properties of the *request* rather than of the feature, which is why they are here and
    not in `next_step`: a resume may not restart something already finished, and a resume
    whose every workstream has spent its retry decision has nothing to run even when the
    feature itself still owes an integration review.
    """
    if state.status in _UNRESUMABLE_STATUSES:
        msg = "completed or cancelled features cannot be resumed"
        raise FeatureWorkflowError(
            msg,
            diagnostic="A completed or cancelled feature cannot be resumed.",
        )
    if state.child_workflows and not resume_eligible_repository_ids(state):
        # Every repository has already reached a terminal retry decision. Each keeps its
        # status and its refusal; there is simply nothing left for this resume to run.
        # Asked feature-wide -- which is how this read until task 28- -- one terminal
        # repository stopped its healthy siblings from ever being resumed at all.
        msg = "every workstream has already reached a terminal retry decision"
        raise FeatureResumeNotEligible(msg)


def _integration_merge_order(
    plan: RepositoryExecutionPlanArtifact, ready_repository_ids: set[str]
) -> list[str]:
    """Order the repositories an integration review reads by the plan's own merge order."""
    return [
        item.workstream_id
        for workstream_id in plan.execution_order
        for item in plan.workstreams
        if item.workstream_id == workstream_id and item.repository_id in ready_repository_ids
    ]


def _integration_remediation_feedback(state: FeatureWorkflowSnapshot) -> dict[str, list[str]]:
    """Return what the latest integration review asks of each repository it names.

    Derived on the way into an execution step rather than handed to it, because under
    step-based execution the step that routes a finding and the step that answers it are two
    different claims in two different processes. The durable verdict is the only thing that
    survives between them, and it already says everything the engineer is told.
    """
    review = _latest_artifact(state.artifacts, IntegrationReviewArtifact)
    if review is None or review.review_status != "changes_requested":
        return {}
    feedback: dict[str, list[str]] = {}
    for finding in review.cross_repository_findings:
        feedback.setdefault(finding.responsible_repository_id, []).append(finding.recommended_fix)
    return feedback


def _initialize_children(
    state: FeatureWorkflowSnapshot,
    plan: RepositoryExecutionPlanArtifact,
    *,
    workspace_root: Path,
) -> None:
    """Persist deterministic child IDs, safe branches, and separate workspace paths once."""
    for workstream in plan.workstreams:
        repository = next(
            item
            for item in state.repository_specs
            if item.repository_id == workstream.repository_id
        )
        if repository.repository_id in state.child_workflows:
            continue
        workspace_path = repository.local_workspace_path or str(
            workspace_root / _feature_branch_segment(state.feature_id) / repository.repository_id
        )
        state.child_workflows[repository.repository_id] = ChildWorkflowReference(
            child_workflow_id=f"{state.feature_id}:{repository.repository_id}",
            repository_id=repository.repository_id,
            workstream_id=workstream.workstream_id,
            status=ChildWorkflowStatus.PENDING,
            branch_name=(
                f"ai/{_feature_branch_segment(state.feature_id)}/"
                f"{repository.repository_id}/{_slug(state.title)}"
            ),
            workspace_path=workspace_path,
            retry_count=0,
            # Recorded at reconnaissance time, copied here so the workstream an operator
            # reads carries the warning its plan earned.
            planned_blind=repository.repository_id in state.planned_blind_repositories,
            planned_blind_reason=state.planned_blind_repositories.get(repository.repository_id),
        )


def _resync_children_with_plan(
    state: FeatureWorkflowSnapshot, plan: RepositoryExecutionPlanArtifact
) -> None:
    """Point each existing child at the workstream the current plan gives its repository.

    `_initialize_children` deliberately skips a repository that already has a child, which is
    what keeps branch names and workspaces stable across every ordinary re-plan. A revision's
    planner runs against a fresh requirement set, so its workstream ids may differ from the
    ones the children were created with -- and the executor looks a child's workstream up in
    the plan by id, so a stale id would make the repository unschedulable.
    """
    for workstream in plan.workstreams:
        child = state.child_workflows.get(workstream.repository_id)
        if child is not None and child.workstream_id != workstream.workstream_id:
            state.child_workflows[workstream.repository_id] = child.model_copy(
                update={"workstream_id": workstream.workstream_id}
            )


# The branch and workspace suffixes one revision adds and the next one replaces. `-V3`
# supersedes `-V2` rather than nesting after it, so the stem is always the original run's name.
_BRANCH_REVISION_SUFFIX = re.compile(r"-V\d+$")
_WORKSPACE_REVISION_SUFFIX = re.compile(r"-v\d+$")


def begin_revision(
    state: FeatureWorkflowSnapshot, *, request_text: str, requested_by: str
) -> FeatureWorkflowSnapshot:
    """Re-open a completed feature so a person's change request runs as its next revision.

    Pure state mutation, no I/O: the control plane applies it synchronously inside the
    request (under the run lock, like a cancellation) and then queues the run, so no claim
    ever observes a completed feature mid-revision.

    What it does, in order: records the request and exactly which branches and pull requests
    it supersedes; replaces the technical PRD with one whose single requirement *is* the
    request text (so planning runs against the change being asked for, and the product
    manager never re-runs); resets every child to a fresh `-V{n+1}` branch based on its
    superseded branch, with `retry_count + 1` because attempt-scoped artifact ids are
    immutable; and moves the feature to `planning`, the one status `next_step` will advance
    from through plan, execution, review and publication exactly as any other run.
    """
    revision = state.revision + 1
    superseded: list[dict[str, Any]] = []
    pull_requests_by_id = {
        artifact.artifact_id: artifact
        for artifact in state.artifacts
        if isinstance(artifact, PullRequestArtifact)
    }
    for repository_id in sorted(state.child_workflows):
        child = state.child_workflows[repository_id]
        pull_request = (
            pull_requests_by_id.get(child.pull_request_artifact_id)
            if child.pull_request_artifact_id is not None
            else None
        )
        superseded.append(
            {
                "repository_id": repository_id,
                "branch_name": child.branch_name,
                "pull_request_artifact_id": child.pull_request_artifact_id,
                "pull_request_url": str(pull_request.url) if pull_request is not None else None,
                "pull_request_number": (
                    pull_request.pull_request_number if pull_request is not None else None
                ),
            }
        )
    request_artifact = create_artifact(
        FeatureRevisionRequestArtifact,
        workflow_id=state.feature_id,
        artifact_id=(
            f"{FEATURE_ARTIFACT_FILENAMES['feature_revision_request'].removesuffix('.json')}"
            f".v{revision}.json"
        ),
        producer="human_revision",
        payload={
            "feature_id": state.feature_id,
            "revision": revision,
            "request_text": request_text,
            "requested_by": requested_by,
            "superseded": superseded,
            "requested_at": datetime.now(UTC),
        },
        metadata={"feature_revision": revision},
    )
    _append_artifacts(state, [request_artifact])
    previous_prd = _require_artifact(state.artifacts, TechnicalPRDArtifact)
    revision_prd = create_artifact(
        TechnicalPRDArtifact,
        workflow_id=state.feature_id,
        # `_replace_artifact` below re-identifies this as the PRD lineage's next revision;
        # the base id here is never appended.
        artifact_id=ARTIFACT_FILENAMES["technical_prd"],
        producer="human_revision",
        payload={
            "title": state.title,
            "solution_summary": (
                f"Revision {revision} of this feature. The previously published work is the "
                "starting point -- each repository's checkout is created from its superseded "
                "branch -- and the one requirement below is the change a person asked for."
            ),
            "functional_requirements": [
                {
                    "requirement_id": f"REV{revision}-1",
                    "description": request_text,
                    "priority": "must",
                    "acceptance_criteria": [
                        "The published work is changed as the revision request describes, "
                        "and nothing it did not ask about is reworked."
                    ],
                    "dependencies": [],
                }
            ],
            "non_functional_requirements": [],
            "data_requirements": [],
            "integration_requirements": [],
            "security_requirements": [],
            "assumptions": [
                "The superseded branch already satisfies the original requirements; this "
                "revision judges only the requested change."
            ],
            # Deliberately empty: a revision never re-opens clarification. The person just
            # said, in their own words, what they want -- that text is the answer.
            "unresolved_questions": [],
        },
        metadata={
            "feature_revision": revision,
            "revision_request_artifact_id": request_artifact.artifact_id,
            # Carried forward so `_asked_question_ids` stays complete and reconciliation
            # never re-asks or re-applies what the original run already settled.
            **{
                key: previous_prd.metadata[key]
                for key in ("clarification_answers", "reconciled_answer_ids")
                if key in previous_prd.metadata
            },
        },
    )
    _replace_artifact(state, previous_prd, revision_prd)
    for repository_id in sorted(state.child_workflows):
        child = state.child_workflows[repository_id]
        branch_stem = _BRANCH_REVISION_SUFFIX.sub("", child.branch_name)
        workspace_stem = _WORKSPACE_REVISION_SUFFIX.sub("", child.workspace_path)
        state.child_workflows[repository_id] = child.model_copy(
            update={
                # The superseded branch is the base: the new attempt modifies the published
                # work rather than regenerating it from the repository default.
                "base_branch": child.branch_name,
                "branch_name": f"{branch_stem}-V{revision + 1}",
                # A fresh checkout, so the journaled clone's identity is new and nothing
                # replays the superseded run's clone.
                "workspace_path": f"{workspace_stem}-v{revision + 1}",
                "status": ChildWorkflowStatus.PENDING,
                # A revision is a distinct immutable attempt: reusing the previous
                # retry_count would collide with its completion, review and result artifact
                # ids -- the same move integration remediation makes, for the same reason.
                "retry_count": child.retry_count + 1,
                "targeted_attempt_kind": None,
                "checkpoint_boundary": WorkflowCheckpointBoundary.BEFORE_CLONE,
                "pull_request_artifact_id": None,
                "blocking_issues": [],
                "retry_refusal_reason": None,
                "failure_classification": None,
                "retry_strategy": None,
                # A fresh checkout re-runs setup and re-measures its own revision.
                "preflight_result": None,
                "preflight_status": None,
                "current_revision": None,
                "current_validation_results": [],
                "pending_context_partition": None,
                # The no-progress gate must judge the revision's first attempt on its own
                # diff, never against the superseded run's.
                "meaningful_change": None,
                "meaningful_change_reason": None,
                "production_diff_fingerprint": None,
                "previous_attempt_fingerprint": None,
                "test_diff_fingerprint": None,
                "previous_test_fingerprint": None,
            }
        )
    state.revision = revision
    transition_feature(
        state,
        FeatureWorkflowStatus.PLANNING,
        reason=f"Revision {revision}: {requested_by} asked for changes to the published work.",
        agent="human_revision",
    )
    state.updated_at = datetime.now(UTC)
    return state


def _cancelled_child_execution(
    feature: FeatureWorkflowSnapshot,
    *,
    repository: RepositorySpec,
    workstream: RepositoryWorkstreamPlan,
    child: ChildWorkflowReference,
) -> ChildExecution:
    """Retain a typed child result when cancellation stops active sibling execution."""
    return ChildExecution(
        result=_child_result(
            feature=feature,
            repository=repository,
            workstream=workstream,
            child=child,
            code_completion=None,
            review=None,
            status="cancelled",
            blocking_issues=["Cancellation requested before child workstream completed."],
        )
    )


def _with_stranded_commit(
    execution: ChildExecution, stranded: tuple[str, str] | None
) -> ChildExecution:
    """Record a branch an earlier attempt already pushed, when the child ends up failed.

    Reporting only the final attempt hides work that is already on the remote. The status
    stays ``failed`` because a later stage did reject this workstream, but the branch has to
    be named so it can be reviewed or deleted rather than accumulating unnoticed.
    """
    if stranded is None or execution.result.status != "failed":
        return execution
    branch, commit_sha = stranded
    note = (
        f"An earlier attempt already committed {commit_sha} and pushed branch {branch}. "
        "That branch is on the remote and is not part of any pull request; review or delete "
        "it. This workstream is still failed because a later attempt was rejected."
    )
    if note in execution.result.blocking_issues:
        return execution
    return ChildExecution(
        result=execution.result.model_copy(
            update={"blocking_issues": [*execution.result.blocking_issues, note]}
        ),
        code_completion=execution.code_completion,
        review=execution.review,
        contract_change_request=execution.contract_change_request,
    )


def _non_converging_execution(execution: ChildExecution) -> ChildExecution:
    """Return the attempt annotated so the operator sees why retrying stopped early."""
    return ChildExecution(
        result=execution.result.model_copy(
            update={
                "blocking_issues": [
                    *execution.result.blocking_issues,
                    "Retrying produced identical diagnostics, so the remaining attempts "
                    "were not spent. This defect is outside what rewriting the source "
                    "can fix.",
                ]
            }
        ),
        code_completion=execution.code_completion,
        review=execution.review,
        contract_change_request=execution.contract_change_request,
    )


def _with_triage(execution: ChildExecution, triage: TerminalTriage) -> ChildExecution:
    """Attach the decision a human now owns to a workstream that has stopped.

    The question is recorded in `blocking_issues` as well as metadata deliberately. That list
    is what already reaches the parent, the completion record, the `/features` API and the
    console, so a question placed only in metadata would be a question nobody is shown. The
    cause and its evidence go to metadata, where a later gate or report can read them without
    parsing prose.
    """
    if execution.result.status == "approved":
        return execution
    return ChildExecution(
        result=execution.result.model_copy(
            update={
                "blocking_issues": [*execution.result.blocking_issues, triage.question],
                "metadata": {
                    **execution.result.metadata,
                    "operator_question": triage.question,
                    "terminal_cause": triage.cause.value,
                    "terminal_evidence": list(triage.evidence),
                    # The measured values behind the evidence sentences (77-, item 33):
                    # fingerprints and attempt numbers a report reads without parsing prose.
                    **({"terminal_measurements": dict(triage.metadata)} if triage.metadata else {}),
                },
            }
        ),
        code_completion=execution.code_completion,
        review=execution.review,
        contract_change_request=execution.contract_change_request,
    )


def _repository_runtime_limit_execution(
    feature: FeatureWorkflowSnapshot,
    *,
    repository: RepositorySpec,
    workstream: RepositoryWorkstreamPlan,
    child: ChildWorkflowReference,
    elapsed_seconds: float,
    limit_seconds: float,
    wall_seconds: float | None = None,
) -> ChildExecution:
    """Stop scheduling attempts for a workstream that has run past its runtime ceiling.

    Says which ceiling and which setting, because the operator's decision here is whether
    this repository deserves more time -- and that is unanswerable from "it stopped". The
    revision this workstream reached is carried forward: the last attempt's result is what
    it was measured against, and a ceiling does not move the checkout.

    Classified as neither a platform defect nor a repository finding. The work may have been
    progressing perfectly well; what ran out was time this deployment is willing to spend.
    """
    elapsed_minutes = round(elapsed_seconds / 60)
    limit_minutes = round(limit_seconds / 60)
    wall_minutes = round(wall_seconds / 60) if wall_seconds is not None else elapsed_minutes
    excluded_note = (
        (
            f" Another {wall_minutes - elapsed_minutes} minute(s) of wall time was spent "
            "inside classified provider faults and was deliberately not charged."
        )
        if wall_minutes > elapsed_minutes
        else ""
    )
    diagnostic = (
        f"This repository's workstream ran for about {elapsed_minutes} minutes of charged "
        f"time, past the {limit_minutes}-minute ceiling this deployment sets, so no further "
        "attempt was scheduled for it. Nothing was interrupted: the attempt that was "
        f"running finished first.{excluded_note} Raise `repository_runtime_limit_seconds` "
        "if this repository legitimately needs longer, or grant it attempts again once you "
        "have read what it was doing."
    )
    result = _child_result(
        feature=feature,
        repository=repository,
        workstream=workstream,
        child=child,
        code_completion=None,
        review=None,
        status="failed",
        blocking_issues=[diagnostic],
        current_revision=child.current_revision,
    )
    return _with_triage(
        ChildExecution(
            result=result.model_copy(
                update={
                    "failure_classification": (
                        FeatureFailureClassification.REPOSITORY_RUNTIME_LIMIT_REACHED.value
                    ),
                }
            )
        ),
        TerminalTriage(
            # The nearest existing cause, and the honest one: what this workstream exhausted
            # was a budget. `TerminalCause` is `tools/retry_strategy.py`'s, which this task
            # may not change, and inventing a synonym elsewhere would split the vocabulary.
            cause=TerminalCause.BUDGET_EXHAUSTED,
            question=(
                f"This repository spent about {elapsed_minutes} minutes without reaching an "
                "approved result. Is the requirement right for this repository, and should "
                "it be given more time?"
            ),
            evidence=[
                f"{child.retry_count} attempt(s) spent before the ceiling was reached.",
                f"Runtime ceiling: {limit_minutes} minutes.",
            ],
        ),
    )


def source_rejection_disposition(error: SourceValidationError) -> tuple[str, dict[str, Any]]:
    """Say what kind of failure one Engineer-raised rejection is, and what it must carry.

    One function, consumed by both handlers that turn a ``SourceValidationError`` into a
    child result -- the runtime's and this module's -- because the two drifting apart would
    let the same attempt be classified two ways depending on which loop caught it.

    The commit gate's rejection is a deterministic tool speaking about the source, so it
    keeps its historical shape: `validation_source_failure`, the `deterministic_gate`
    exemption from the repeat rules, and the pre-commit retry allowance.

    **But the exemption belongs to the check that spoke, not to the gate it came from.** A
    linter repeating itself is the tool being consistent: it emits the same sentence for as
    long as a rule is broken, and Feature -052b lost both workstreams two attempts in because
    the reachability gate said the same true thing twice while the attempts around it were
    changing. A *test* repeating itself is the change not moving, which is the one thing these
    rules exist to catch. Both wore this marker, because the commit gate runs formatting,
    lint, typecheck **and the change's own scoped tests**, so a failing suite arrived here
    dressed as an eslint rule -- and AB-Feature-218's backend consequently failed the same
    suite on four consecutive attempts with every convergence guard the platform has switched
    off, no budget exhausted and nothing stopped.

    So the exemption now covers exactly the checks whose repetition is uninformative: it is
    withheld when any diagnostic in the rejection is the scoped test runner's, identified by
    its producer's own constant rather than by a copy of its sentence. A mixed rejection --
    one eslint error and one failing test -- is not exempt. The failing test is the part that
    says something about convergence, and treating the pair as exempt because a lint error
    rode along is how the exemption became universal in the first place.

    `source_validation_rejected` is untouched and still true in both cases. It is a fact about
    *where* the attempt was stopped, which is what the pre-commit retry allowance is actually
    about, and 83- established that the two must not be re-conflated.

    A self-review rejection is neither of those things. Every configured check *accepted*
    the source; what stopped the attempt is a model's judgement that the assignment was not
    met. That is implementation work (`implementation_missing` -- the classification the
    completeness gate already uses for the same defect found deterministically), it is not
    exempt from the repeat rules (a judgement recurring across attempts is evidence, not a
    tool being consistent), and it has no claim on the allowance that exists for attempts
    the commit gate stopped.
    """
    if error.terminal_outcome == SELF_REVIEW_SUBSTANTIVE_OUTCOME:
        return (
            FailureClassification.IMPLEMENTATION_MISSING.value,
            {
                "self_review_rejected": True,
                "terminal_outcome": error.terminal_outcome,
            },
        )
    if any(is_scoped_test_diagnostic(diagnostic) for diagnostic in error.diagnostics):
        return (
            FailureClassification.VALIDATION_SOURCE_FAILURE.value,
            {"source_validation_rejected": True},
        )
    return (
        FailureClassification.VALIDATION_SOURCE_FAILURE.value,
        {
            "deterministic_gate": True,
            "source_validation_rejected": True,
        },
    )


# The most scoped attempts one context-partition wave may run. Beyond it the smallest
# clusters merge: a wave is one granted retry's execution shape, and an unbounded number of
# engineer runs on one grant would be a budget the retry policy never priced.
_MAX_CONTEXT_PARTITION_CLUSTERS = 4


def _context_partition_clusters(child: ChildWorkflowReference) -> list[dict[str, Any]]:
    """Cluster a refused retry's blocking demands by the files they name (80- Part 3).

    Returns the wave plan -- ``{"diagnostics": [...], "files": [...], "status": "pending"}``
    per cluster -- or ``[]`` when partitioning is not allowed: a first-attempt refusal
    (attempt 0's ordered set is at most six assigned paths and cannot meaningfully overflow),
    a workstream already mid-wave, no blocking diagnostics to split, or a single cluster
    (nothing to split -- the honest floor stays terminal).

    Clustering reuses the same location-token parsing the engineer's
    ``_diagnostic_file_locations`` applies (``diagnostic_file_references``), then merges
    clusters that share a file (union-find). Diagnostics naming no file ride with the first
    cluster. Beyond the cap the smallest clusters merge first.
    """
    if child.retry_count <= 0 or child.pending_context_partition:
        return []
    diagnostics = [item for item in child.blocking_issues if isinstance(item, str) and item.strip()]
    if not diagnostics:
        return []
    parent = list(range(len(diagnostics)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        parent[find(right)] = find(left)

    file_owner: dict[str, int] = {}
    files_of: list[list[str]] = []
    for index, text in enumerate(diagnostics):
        files = [token for token, _line in diagnostic_file_references([text])]
        files_of.append(files)
        for token in files:
            if token in file_owner:
                union(file_owner[token], index)
            else:
                file_owner[token] = index
    anchored = [index for index, files in enumerate(files_of) if files]
    if not anchored:
        return []
    for index, files in enumerate(files_of):
        if not files:
            union(anchored[0], index)
    clusters: dict[int, dict[str, list[str]]] = {}
    for index, text in enumerate(diagnostics):
        group = clusters.setdefault(find(index), {"diagnostics": [], "files": []})
        group["diagnostics"].append(text)
        group["files"].extend(token for token in files_of[index] if token not in group["files"])
    ordered = list(clusters.values())
    if len(ordered) < 2:
        return []
    while len(ordered) > _MAX_CONTEXT_PARTITION_CLUSTERS:
        # Merge the two smallest, sized by how many demands each carries; stable on ties so
        # the plan is deterministic for one input.
        by_size = sorted(range(len(ordered)), key=lambda i: (len(ordered[i]["diagnostics"]), i))
        keep, absorb = sorted(by_size[:2])
        ordered[keep]["diagnostics"].extend(ordered[absorb]["diagnostics"])
        ordered[keep]["files"].extend(
            token for token in ordered[absorb]["files"] if token not in ordered[keep]["files"]
        )
        del ordered[absorb]
    return [{**cluster, "status": "pending"} for cluster in ordered]


def _failed_child_execution(
    feature: FeatureWorkflowSnapshot,
    *,
    repository: RepositorySpec,
    workstream: RepositoryWorkstreamPlan,
    child: ChildWorkflowReference,
    error: BaseException,
) -> ChildExecution:
    """Contain one child fault without hiding safe pre-commit lint diagnostics."""
    if isinstance(error, SourceValidationError):
        # The completion the Engineer produced before the gate rejected it, where the
        # raising path attached one. Recording it is what gives the next retry its lineage;
        # its status is `failed`, so no publication read can select it.
        rejected_completion = cast(CodeCompletionArtifact | None, error.code_completion)
        result = _child_result(
            feature=feature,
            repository=repository,
            workstream=workstream,
            child=child,
            code_completion=rejected_completion,
            review=None,
            status="failed",
            blocking_issues=list(error.diagnostics),
        )
        classification, rejection_metadata = source_rejection_disposition(error)
        return ChildExecution(
            result=result.model_copy(
                update={
                    "failure_classification": classification,
                    # The same markers the runtime's own handler records. Stated rather
                    # than inferred: the retry loop's pre-commit allowance must be
                    # available only where the gate really did stop the attempt.
                    "metadata": {
                        **result.metadata,
                        **rejection_metadata,
                    },
                    # And the same evidence, for the same reason the markers are the same:
                    # this handler and the runtime's must not describe one attempt two ways.
                    # See the runtime's own comment for what an empty list means to the
                    # coarse failure counter.
                    "current_validation_results": list(error.validation_results),
                }
            ),
            code_completion=rejected_completion,
        )
    # The classification may sit on a wrapped cause rather than the top-level type: the
    # coding call's adapter error reaches this function inside `ExternalOperationError`, and
    # reading only the wrapper turned AB-Feature-174's deterministic truncation into
    # "the provider did not answer". The deterministic walk is authoritative when it finds
    # one; the top-level declaration keeps its existing meaning otherwise.
    declared_source = _deterministic_provider_error(error) or error
    declared_classification = getattr(declared_source, "failure_classification", None)
    declared_diagnostics = safe_error_diagnostics(declared_source)
    if isinstance(declared_classification, str) and declared_diagnostics:
        return ChildExecution(
            result=_child_result(
                feature=feature,
                repository=repository,
                workstream=workstream,
                child=child,
                code_completion=None,
                review=None,
                status="failed",
                blocking_issues=list(declared_diagnostics),
            ).model_copy(update={"failure_classification": declared_classification})
        )
    # One repository raising must never abort the fan-out, for the same reason a failing
    # review must not: unrelated repositories still have deliverable work. Keep generic
    # exception text out of artifacts because it can carry credential-bearing detail.
    #
    # Classified rather than left blank. An exception type nothing anticipated is evidence
    # about this platform, not about the checkout: leaving the classification absent made the
    # feature's own summary read `unclassified_failure` -- ten recorded failures -- and letting
    # the type name through made a `DBAPIError` in the platform's connection pool look like a
    # verdict on somebody's repository.
    #
    # `is_transient_provider_fault` is asked first because it is already the platform's single
    # answer to "was that the provider's fault", and a bare `LLMAdapterError` carries no
    # provider subtype for the classifier to read.
    fault = is_transient_provider_fault(error)
    # The named helper rather than the expression, because the queue's exhaustion route needs
    # the same answer and reached it through `classification_of` alone -- so a GitHub outage
    # that ran the queue out of attempts was recorded as a defect in this platform.
    classification = transient_fault_classification(error)
    # The type name stays in the sentence. It is platform-owned, never model output, and it
    # is what tells whoever reads this which failure they are looking at -- what changes is
    # that it is no longer *also* the feature's domain classification.
    error_name = type(error).__name__
    return ChildExecution(
        result=_child_result(
            feature=feature,
            repository=repository,
            workstream=workstream,
            child=child,
            code_completion=None,
            review=None,
            status="failed",
            blocking_issues=[
                (
                    _fault_did_not_answer(error, error_name=error_name)
                    if fault
                    else f"The platform did not anticipate {error_name} while running this "
                    "repository and could not complete it. This is a defect in the platform, "
                    "not in this repository or the requirement."
                ),
                # Withholding an already-safe explanation left an operator with a type
                # name for a failure whose cause was known: four separate runs ended that
                # way before the text was allowed through.
                *_causal_diagnostics(error),
            ],
        ).model_copy(update={"failure_classification": classification.value})
    )


async def _await_if_needed(value: object) -> object:
    """Accept the established synchronous mock GitHub service and async live wrapper."""
    return await value if inspect.isawaitable(value) else value


def _has_unpublished_approved_child(state: FeatureWorkflowSnapshot) -> bool:
    """Return whether any repository passed review and has no pull request yet."""
    return any(
        result.status == "approved"
        and result.pull_request_readiness
        and state.child_workflows[result.repository_id].pull_request_artifact_id is None
        for result in _latest_child_results(state)
        if result.repository_id in state.child_workflows
    )


def _outstanding_integration_fixes(state: FeatureWorkflowSnapshot, repository_id: str) -> list[str]:
    """Return what the most recent integration review still asks of one repository.

    Only the newest review, and only if it asked for changes. An earlier cycle's findings
    were either answered or superseded, and replaying them would hand the engineer
    instructions about source that has since been rewritten -- the same mistake that gave
    -105's ninth attempt a pile of partly-contradictory feedback.
    """
    latest: IntegrationReviewArtifact | None = None
    for artifact in state.artifacts:
        if isinstance(artifact, IntegrationReviewArtifact):
            latest = artifact
    if latest is None or latest.review_status != "changes_requested":
        return []
    return [
        item.recommended_fix
        for item in latest.cross_repository_findings
        if item.responsible_repository_id == repository_id and item.recommended_fix
    ]


# Punctuation, casing and spacing vary between two reports of the same instruction.
# `finding_id`, severity and evidence are deliberately absent: the Engineer receives the
# recommended fix, so changing reviewer-only metadata must not authorize another coding call.
_FINDING_WORDING = re.compile(r"[^a-z0-9]+")


def _finding_key(finding: IntegrationReviewFinding) -> str:
    """Normalize the exact finding text handed back to this repository's Engineer."""
    return _FINDING_WORDING.sub(" ", finding.recommended_fix.lower()).strip()


def _reviewed_child_result(
    state: FeatureWorkflowSnapshot, review: IntegrationReviewArtifact, repository_id: str
) -> ChildWorkflowResultArtifact | None:
    """Return only the immutable child result this review explicitly cites."""
    result_id = next(
        (
            item.child_result_artifact_id
            for item in review.repository_results
            if item.repository_id == repository_id
        ),
        None,
    )
    if result_id is None:
        return None
    return next(
        (
            artifact
            for artifact in state.artifacts
            if isinstance(artifact, ChildWorkflowResultArtifact)
            and artifact.artifact_id == result_id
        ),
        None,
    )


def _reviewed_revision(
    state: FeatureWorkflowSnapshot, review: IntegrationReviewArtifact, repository_id: str
) -> str:
    """Return the exact repository revision one integration review judged.

    Read from the child result the review itself names, never from current child state. A
    verdict that judged revision A must not be compared -- or routed -- as though it had
    judged the B the repository has since moved to, which is the whole point of binding a
    review to a revision.
    """
    result = _reviewed_child_result(state, review, repository_id)
    if result is not None:
        return str(
            result.current_revision or result.metadata.get("commit_sha") or result.artifact_id
        )
    # Fail closed on a dangling citation. Falling back to the mutable current child revision
    # relabelled a verdict on A as a verdict on B after the repository moved. An empty value
    # is not a revision, and the retry decision below refuses remediation without the cited
    # immutable result.
    return ""


_STABLE_VALIDATION_FIELDS = (
    "name",
    "command",
    "passed",
    "validation_type",
    "repository_id",
    "repository_revision",
    "working_tree_fingerprint",
    "status",
    "exit_code",
    "result_code",
    "failure_classification",
    "required",
    "is_current",
)


def _reviewed_validation_signature(result: ChildWorkflowResultArtifact | None) -> str:
    """Fingerprint validation verdicts without durations, operation IDs, or timestamps."""
    if result is None:
        return ""
    evidence: Sequence[Any] = (
        result.current_validation_results
        if result.current_validation_results
        else result.validation_results
    )
    normalized: list[dict[str, Any]] = []
    for item in evidence:
        value = item if isinstance(item, dict) else item.model_dump(mode="json")
        normalized.append(
            {key: value.get(key) for key in _STABLE_VALIDATION_FIELDS if key in value}
        )
    normalized.sort(key=lambda item: json.dumps(item, sort_keys=True, default=str))
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _remediation_signature(
    state: FeatureWorkflowSnapshot, review: IntegrationReviewArtifact, repository_id: str
) -> str:
    """Fingerprint the effective Engineer input one integration remediation would carry.

    Deliberately built from the reviewed revision, approved contract, validation/repair state,
    artifact schema, and normalized findings actually handed to this repository. No timestamp
    and no attempt number take part: an idempotency key that changes on its own answers "has
    this already been done" with "no" for ever.
    """
    keys = sorted(
        {
            _finding_key(item)
            for item in review.cross_repository_findings
            if item.responsible_repository_id == repository_id
        }
    )
    result = _reviewed_child_result(state, review, repository_id)
    contract = next(
        (
            artifact
            for artifact in state.artifacts
            if isinstance(artifact, IntegrationContractArtifact)
            and artifact.artifact_id == review.contract_artifact_id
        ),
        None,
    )
    repair_state = json.dumps(
        result.retry_strategy if result is not None and result.retry_strategy is not None else {},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    parts = [
        repository_id,
        _reviewed_revision(state, review, repository_id),
        review.contract_artifact_id,
        contract.contract_version if contract is not None else "",
        result.schema_version if result is not None else "",
        _reviewed_validation_signature(result),
        repair_state,
        *keys,
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _previous_remediation_signature(
    state: FeatureWorkflowSnapshot, review: IntegrationReviewArtifact, repository_id: str
) -> str | None:
    """Return the same fingerprint for the last remediation this repository was already sent.

    ``None`` when there is no earlier one, which callers must read as "this is new work"
    rather than "unchanged": a first remediation is exactly the legitimate case.
    """
    earlier = [
        item
        for item in state.artifacts
        if isinstance(item, IntegrationReviewArtifact)
        and item.artifact_id != review.artifact_id
        and item.review_status == "changes_requested"
        and any(
            finding.responsible_repository_id == repository_id
            for finding in item.cross_repository_findings
        )
    ]
    if not earlier:
        return None
    return _remediation_signature(state, earlier[-1], repository_id)


def _integration_remediation_decision(
    state: FeatureWorkflowSnapshot, review: IntegrationReviewArtifact, repository_id: str
) -> RetryDecision:
    """Decide whether integration review may send one repository back to the Engineer Agent.

    Routed through `decide_child_retry` on purpose. That function documents itself as the
    only place the answer "may this workstream attempt again" is produced, and this edge was
    the one caller that never asked it -- which is how a repository was coded twelve times
    against a ceiling of five. The evidence it needs is supplied here and nothing else is
    decided here:

    * the attempt count is this repository's own integration counter, so two repositories
      reworking in parallel bound each other's cycles not at all;
    * the ceiling is the configured review-cycle limit, the same one an operator grant is
      refused against, so both paths now agree about what the limit means;
    * "meaningful change" is whether this remediation carries anything the previous one did
      not -- a moved revision, or a different finding.
    """
    child = state.child_workflows[repository_id]
    reviewed_result = _reviewed_child_result(state, review, repository_id)
    previous = _previous_remediation_signature(state, review, repository_id)
    current = _remediation_signature(state, review, repository_id)
    return decide_child_retry(
        classification=FailureClassification.CONTRACT_MISMATCH,
        attempt_count=child.integration_retry_count,
        # The configured limit rather than what the parent loop has left of it. The parent
        # checks its own remaining cycles one step earlier; charging them twice would halve
        # the reworks a feature is actually allowed.
        budget=state.max_integration_review_cycles,
        retry_count=child.retry_count,
        max_child_review_cycles=state.max_child_review_cycles,
        meaningful_change=(
            reviewed_result is not None and (previous is None or current != previous)
        ),
    )


def _with_remediation_refusal(
    child: ChildWorkflowReference, decision: RetryDecision
) -> ChildWorkflowReference:
    """Record why a repository was not sent back, where an operator already looks for it.

    The reason goes to `retry_refusal_reason`, which the API and the console already render,
    and the question goes to `blocking_issues` for the same reason `_with_triage` puts it
    there: a question recorded only in a field nothing displays is a question nobody is asked.
    """
    question = decision.operator_question
    issues = list(child.blocking_issues)
    if question and question not in issues:
        issues.append(question)
    return child.model_copy(
        update={"retry_refusal_reason": decision.reason, "blocking_issues": issues}
    )


def _integration_review_input_signature(
    *,
    contract: IntegrationContractArtifact,
    child_results: Sequence[ChildWorkflowResultArtifact],
    merge_order: Sequence[str],
) -> str:
    """Fingerprint everything one integration review reads, so a replay can reuse its verdict."""
    parts = [
        contract.artifact_id,
        contract.contract_version,
        *sorted(
            "@".join(
                (
                    item.repository_id,
                    item.artifact_id,
                    item.current_revision or "",
                    _reviewed_validation_signature(item),
                )
            )
            for item in child_results
        ),
        "|".join(merge_order),
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _persisted_integration_review(
    state: FeatureWorkflowSnapshot, *, input_signature: str
) -> IntegrationReviewArtifact | None:
    """Return the durable verdict already computed over exactly these inputs, if there is one.

    Only the most recent integration review is eligible. An older verdict that happens to
    match was superseded by a later one, and resurrecting it would route a decision the
    feature has already moved past.
    """
    for artifact in reversed(state.artifacts):
        if isinstance(artifact, IntegrationReviewArtifact):
            if artifact.metadata.get("input_signature") == input_signature:
                return artifact
            return None
    return None


def _integration_cycle_uncharged(state: FeatureWorkflowSnapshot) -> bool:
    """Return whether the latest blocking integration review has not yet spent its cycle.

    Derived rather than stored: the durable artifacts already say how many integration
    reviews asked for changes, and the snapshot says how many of those were charged. A crash
    between persisting a verdict and charging it therefore reconciles on its own.
    """
    requested = sum(
        isinstance(artifact, IntegrationReviewArtifact)
        and artifact.review_status == "changes_requested"
        for artifact in state.artifacts
    )
    return state.integration_review_cycles < requested


# A workstream in one of these has finished deciding, so a terminal feature leaves it alone.
# BLOCKED is here on purpose: "starved by a failed dependency" is a better answer than
# "failed", and it is the distinction `_RETRYABLE_CHILD_STATUSES` uses to send the attempt to
# the dependency instead. Only PENDING and RUNNING are left over, and both claim work that is
# still to come.
_SETTLED_CHILD_STATUSES: frozenset[ChildWorkflowStatus] = frozenset(
    {
        ChildWorkflowStatus.APPROVED,
        ChildWorkflowStatus.COMPLETED,
        ChildWorkflowStatus.FAILED,
        ChildWorkflowStatus.CANCELLED,
        ChildWorkflowStatus.BLOCKED,
        ChildWorkflowStatus.REVIEW_REJECTED,
        ChildWorkflowStatus.WAITING_FOR_CONTRACT_CHANGE,
    }
)


def settle_unfinished_children(
    state: FeatureWorkflowSnapshot, settled: ChildWorkflowStatus
) -> None:
    """Give every still-open workstream a terminal status when the feature reaches one.

    A feature that has stopped will never schedule anything again, so a workstream left
    `pending` or `running` is not waiting for a turn -- it is a row nobody will ever update.
    AB-Feature-107 was cancelled while its console workstream was mid-attempt and that row
    still said `running` a day later, which is what an operator and the workflow graph both
    read as work in flight.

    Finished work is never rewritten: an approved repository stays approved so publication
    and the completion guard still see it, exactly as cancellation already promised.
    """
    for repository_id, child in state.child_workflows.items():
        if child.status in _SETTLED_CHILD_STATUSES:
            continue
        state.child_workflows[repository_id] = child.model_copy(update={"status": settled})


def _revisits_an_earlier_attempt(
    state: FeatureWorkflowSnapshot,
    *,
    child_workflow_id: str,
    production_fingerprint: str,
    test_fingerprint: str | None,
) -> bool:
    """Report whether this child has already submitted exactly this source once before.

    Deliberately strict: both the production and the test bytes must match the same earlier
    attempt. A workstream that returns to old production source while carrying genuinely new
    tests is answering a review, not going in a circle, and must keep its budget.
    """
    if test_fingerprint is None:
        return False
    return any(
        artifact.child_workflow_id == child_workflow_id
        and artifact.production_diff_fingerprint == production_fingerprint
        and artifact.test_diff_fingerprint == test_fingerprint
        for artifact in state.artifacts
        if isinstance(artifact, ChildWorkflowResultArtifact)
    )


def _tree_resubmission(
    state: FeatureWorkflowSnapshot,
    *,
    child_workflow_id: str,
    production_fingerprint: str | None,
) -> TreeResubmission:
    """Name the earliest attempt whose production tree this attempt reproduced.

    The same lineage comparison `_revisits_an_earlier_attempt` makes, kept so the terminal
    evidence can say which tree came back -- "this attempt resubmitted the production tree
    of attempt N", with both fingerprints on the record -- instead of the fixed diagnostic
    sentence 207's stop printed over diagnostics that had changed (77-, item 33).
    """
    if not production_fingerprint:
        return TreeResubmission()
    for artifact in state.artifacts:
        if (
            isinstance(artifact, ChildWorkflowResultArtifact)
            and artifact.child_workflow_id == child_workflow_id
            and artifact.production_diff_fingerprint == production_fingerprint
        ):
            attempt = artifact.metadata.get("child_retry_count")
            return TreeResubmission(
                attempt=attempt if isinstance(attempt, int) else None,
                production_diff_fingerprint=production_fingerprint,
                matched_production_diff_fingerprint=artifact.production_diff_fingerprint or "",
            )
    # The lineage holds no matching artifact -- an immediate repeat judged against the
    # previous attempt's in-memory fingerprint before its result was appended. The fact
    # stands; the attempt number does not.
    return TreeResubmission(production_diff_fingerprint=production_fingerprint)


def _with_wiring_repair(
    retry_strategy: dict[str, Any], result: ChildWorkflowResultArtifact
) -> dict[str, Any]:
    """Carry the wiring gate's scoped repairs into the plan handed to the next attempt.

    Every repair, not only the first. `build_retry_plan` rebuilds the strategy from scratch,
    so anything this function does not copy never reaches the engineer: carrying one repair
    while the gate reported two is how a change that left two modules unreachable still
    needed one attempt per module.
    """
    carried = dict(retry_strategy)
    # The rewrite gate's verdict about the workspace travels the same way: the next attempt
    # reads it to decide whether the previous attempt is worth building on, and a strategy
    # rebuilt without it would silently preserve exactly the drift the gate rejected.
    reset_required = result.metadata.get("workspace_reset_required")
    if isinstance(reset_required, str) and reset_required:
        carried["workspace_reset_required"] = reset_required
    repair = result.metadata.get("wiring_repair")
    if not isinstance(repair, dict):
        return carried if carried != retry_strategy else retry_strategy
    carried["wiring_repair"] = repair
    repairs = result.metadata.get("wiring_repairs")
    if isinstance(repairs, list) and repairs:
        carried["wiring_repairs"] = repairs
    return carried


def _child_result(
    *,
    feature: FeatureWorkflowSnapshot,
    repository: RepositorySpec,
    workstream: RepositoryWorkstreamPlan,
    child: ChildWorkflowReference,
    code_completion: CodeCompletionArtifact | None,
    review: ReviewArtifact | None,
    status: str,
    blocking_issues: list[str],
    current_revision: str | None = None,
    advisory_findings: list[str] | None = None,
    review_finding_counts: ReviewFindingCounts | None = None,
    advisory_review_verdict: str | None = None,
    overruled_findings: Sequence[OverruledReviewFinding] = (),
) -> ChildWorkflowResultArtifact:
    """Create a parent-scoped result without leaking mutable child state or credentials.

    ``current_revision`` is the revision the caller measured against the checkout this
    attempt actually ran on. It is optional only so that callers which have no workspace in
    hand still work; every path in the live executor supplies it.

    ``advisory_review_verdict`` is the declining verdict the platform accepted, where it
    accepted one because every finding the review raised named nothing this workstream was
    asked for. It is what makes the work publishable without an approval, and it is recorded
    so neither this result nor the pull request it produces claims the reviewer said yes.

    ``overruled_findings`` are the blocking findings a person's ``removal_holds`` verdict
    demoted (73-): the suppression, carried onto the record with the decider's name so the
    result and the pull request show a judged finding rather than a vanished one.
    """
    active_contract = _require_artifact(feature.artifacts, IntegrationContractArtifact)
    return create_artifact(
        ChildWorkflowResultArtifact,
        workflow_id=feature.feature_id,
        artifact_id=attempt_artifact_id(
            _repository_artifact_id(
                FEATURE_ARTIFACT_FILENAMES["child_workflow_result"],
                repository.repository_id,
            ),
            child.retry_count,
        ),
        producer="child_workflow",
        payload={
            "feature_id": feature.feature_id,
            "parent_workflow_id": feature.workflow_id,
            "child_workflow_id": child.child_workflow_id,
            "repository_id": repository.repository_id,
            "workstream_id": workstream.workstream_id,
            "branch_name": child.branch_name,
            "workspace_path": child.workspace_path,
            "code_completion_artifact_id": (
                code_completion.artifact_id if code_completion is not None else None
            ),
            "review_artifact_id": review.artifact_id if review is not None else None,
            "changed_files": (
                [item.model_dump(mode="python") for item in code_completion.file_changes]
                if code_completion is not None
                else []
            ),
            "validation_results": (
                [item.model_dump(mode="python") for item in code_completion.validation_results]
                if code_completion is not None
                else []
            ),
            "status": status,
            "blocking_issues": blocking_issues,
            "pull_request_readiness": status == "approved"
            and review is not None
            and (review.verdict == "approved" or advisory_review_verdict is not None),
            "contract_sections_consumed": workstream.contract_sections_consumed,
            "contract_sections_implemented": workstream.contract_sections_implemented,
            "technology_profile": (
                review.metadata.get("technology_profile") if review is not None else None
            ),
            # The review's own record where one ran, otherwise the plan pinned on the child
            # (83-): a gate-stopped attempt has no review, and without the fallback the
            # per-cycle child update would erase the workstream's pinned plan on exactly
            # those attempts.
            "validation_plan": (
                review.metadata.get("validation_plan") if review is not None else None
            )
            or child.validation_plan,
            # Measured first, derived second. The derivation below survives only when a
            # review exists and every one of its validation results agrees on one revision,
            # which is why 165 of 281 recorded workstreams carry none: an attempt stopped at
            # any gate before review has no review to derive from at all.
            "current_revision": current_revision or _current_revision(review),
            "current_validation_results": (
                list(review.metadata.get("validation", [])) if review is not None else []
            ),
            "superseded_validation_count": 0,
            "scoped_requirements": [
                item.model_dump(mode="python") for item in workstream.scoped_requirements
            ],
            "out_of_scope_requirements": workstream.out_of_scope_requirements,
            "preflight_result": child.preflight_result,
            "layout_evidence": child.layout_evidence,
            "failure_classification": child.failure_classification,
            "retry_strategy": child.retry_strategy,
            # The role that produced this result, recorded on the result itself: the child
            # state has already moved on to whatever the next attempt was routed to.
            "model_routing": child.model_routing,
            "production_files_changed": (
                code_completion.production_files_changed
                if code_completion is not None
                else child.production_files_changed
            ),
            "test_files_changed": (
                code_completion.test_files_changed
                if code_completion is not None
                else child.test_files_changed
            ),
            "configuration_files_changed": (
                code_completion.configuration_files_changed
                if code_completion is not None
                else child.configuration_files_changed
            ),
            "requirements_implemented": (
                code_completion.requirements_implemented
                if code_completion is not None
                else child.requirements_implemented
            ),
            "requirements_not_implemented": (
                code_completion.requirements_not_implemented
                if code_completion is not None
                else child.requirements_not_implemented
            ),
            "meaningful_change": child.meaningful_change,
            "meaningful_change_reason": child.meaningful_change_reason,
            "production_diff_fingerprint": child.production_diff_fingerprint,
            "test_diff_fingerprint": child.test_diff_fingerprint,
            "previous_attempt_fingerprint": child.previous_attempt_fingerprint,
            "advisory_findings": advisory_findings or [],
            "overruled_findings": [item.model_dump(mode="python") for item in overruled_findings],
            "review_finding_counts": (
                review_finding_counts.model_dump(mode="python")
                if review_finding_counts is not None
                else None
            ),
            "advisory_review_verdict": advisory_review_verdict,
        },
        metadata={
            # Which feature revision this attempt belongs to. The selectors read within the
            # current revision, so an unstamped record (every result before revisions, and
            # every original-run result) reads as revision 0.
            **_revision_metadata(feature),
            "contract_artifact_id": active_contract.artifact_id,
            "contract_version": active_contract.contract_version,
            "child_retry_count": child.retry_count,
            "commit_sha": code_completion.commit_sha if code_completion is not None else None,
            "implementation_retry_count": child.implementation_retry_count,
            "validation_retry_count": child.validation_retry_count,
            "repository_setup_retry_count": child.repository_setup_retry_count,
            "integration_retry_count": child.integration_retry_count,
            # Which authority demanded this attempt. On the attempt's own immutable record
            # because the child's field is a single value that has moved on by the time
            # anybody reads the history: without this, every past retry would be drawn from
            # whichever authority happened to demand the latest one.
            "targeted_attempt_kind": (
                str(child.targeted_attempt_kind)
                if child.targeted_attempt_kind is not None
                else None
            ),
            # Both clocks travel with the attempt record, so the difference between wall
            # and charged time survives into everything built from results.
            "runtime_wall_seconds": child.runtime_wall_seconds,
            "runtime_charged_seconds": child.runtime_charged_seconds,
        },
    )


def _effective_source_areas(
    workstream: RepositoryWorkstreamPlan, layout_evidence: dict[str, Any] | None
) -> list[str]:
    """Use only persisted, unambiguous checkout rebinding in retry instructions."""
    remapped: dict[str, str] = {}
    raw_mappings = layout_evidence.get("remapped_areas", []) if layout_evidence else []
    if isinstance(raw_mappings, list):
        for item in raw_mappings:
            if (
                isinstance(item, list)
                and len(item) == 2
                and isinstance(item[0], str)
                and isinstance(item[1], str)
            ):
                remapped[item[0]] = item[1]
    return list(
        dict.fromkeys(
            remapped.get(area, area)
            for expectation in workstream.implementation_expectations
            for area in expectation.expected_source_areas
        )
    )


@dataclass(frozen=True, slots=True)
class _RoutingRequest:
    """What the next attempt is, and what it has been asked to correct."""

    execution_mode: ModelExecutionMode
    classifications: tuple[ReviewFixClassification, ...] = ()
    failure_classification: FailureClassification | None = None


def _initial_routing_request(
    child: ChildWorkflowReference, feedback: Sequence[str]
) -> _RoutingRequest:
    """Route the first attempt of a run, which may be an implementation or a remediation.

    A workstream entered with nothing to correct is an implementation. Entered with
    instructions -- integration review's recommended fixes, a granted retry's outstanding
    findings, a resumed loop's blocking issues -- it is a remediation, but only where the
    platform's own failure classification says an Engineer is the right respondent. A
    repository stopped on its checked-in lint configuration is not routed to a remediation
    model however cheap that model is; it belongs to the repair flow.
    """
    issues = [item for item in feedback if item.strip()]
    if not issues:
        return _RoutingRequest(ModelExecutionMode.INITIAL_IMPLEMENTATION)
    classification = _persisted_failure_classification(child)
    if classification is not None and not requires_feature_engineer(classification):
        return _RoutingRequest(
            ModelExecutionMode.INITIAL_IMPLEMENTATION,
            failure_classification=classification,
        )
    return _RoutingRequest(
        ModelExecutionMode.REVIEW_REMEDIATION,
        classify_findings(
            (),
            blocking_issues=issues,
            previous_attempt_intact=_previous_attempt_intact_for(child.retry_strategy),
        ),
        classification,
    )


def _next_routing_request(
    execution: ChildExecution, classification: FailureClassification
) -> _RoutingRequest:
    """Route the attempt that follows a rejected one, from what actually blocked it.

    The failure classification decides whether this is Engineer work at all -- the boundary the
    PRD puts before model routing, and the reason a broken dependency install cannot reach a
    remediation model. Where it is, the reviewer's structured findings are classified in
    preference to their flattened descriptions: a category and a path say things a sentence
    does not, and only the findings that actually blocked are considered, because a low
    severity remark the reviewer recorded for the plan's author is not work to route.
    """
    if not requires_feature_engineer(classification):
        return _RoutingRequest(
            ModelExecutionMode.INITIAL_IMPLEMENTATION,
            failure_classification=classification,
        )
    blocking = set(execution.result.blocking_issues)
    findings = (
        [item for item in execution.review.findings if item.description in blocking]
        if execution.review is not None
        else []
    )
    classifications = classify_findings(
        findings,
        blocking_issues=execution.result.blocking_issues,
        # A retry that will not have the previous attempt intact is a reconstruction, and a
        # reconstruction may not classify as a scoped fix whatever the finding looks like:
        # the scope classifier judges the finding, and the job here is the rebuild (§3.4).
        previous_attempt_intact=(
            _previous_attempt_intact_for(execution.result.retry_strategy)
            and not execution.result.metadata.get("workspace_reset_required")
        ),
    )
    if not classifications:
        return _RoutingRequest(
            ModelExecutionMode.INITIAL_IMPLEMENTATION,
            failure_classification=classification,
        )
    return _RoutingRequest(
        ModelExecutionMode.REVIEW_REMEDIATION,
        classifications,
        classification,
    )


def _previous_attempt_intact_for(retry_strategy: object) -> bool:
    """Whether the retry this strategy describes still builds on the previous attempt."""
    if not isinstance(retry_strategy, dict):
        return True
    return not retry_strategy.get("workspace_reset_required")


def _persisted_failure_classification(
    child: ChildWorkflowReference,
) -> FailureClassification | None:
    """Read the previous attempt's classification, tolerating a value this build cannot map."""
    if not child.failure_classification:
        return None
    try:
        return FailureClassification(child.failure_classification)
    except ValueError:
        return None


def _validation_evidence_fingerprint(child: ChildWorkflowReference) -> str:
    """Fingerprint the validation evidence a remediation is answering.

    Part of the effective-input identity, so an attempt answering different validation results
    is a different question even at the same revision. Empty where a workstream has produced no
    validation yet, which is honest rather than a hash of nothing.
    """
    if not child.current_validation_results:
        return ""
    canonical = json.dumps(child.current_validation_results, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _applied_repair_fingerprint(child: ChildWorkflowReference) -> str:
    """Fingerprint the repository repair this checkout has had applied, if any.

    An approved repair changes what the checkout is, so a remediation after one is answering a
    different question than the same remediation before it.
    """
    preflight = child.preflight_result if isinstance(child.preflight_result, dict) else {}
    commands = preflight.get("applied_repair_commands")
    if not commands:
        return ""
    canonical = json.dumps(commands, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _failure_classification(execution: ChildExecution) -> FailureClassification:
    """Prefer an executor's precise diagnosis, then classify a generic failed child safely."""
    raw = execution.result.failure_classification
    if isinstance(raw, str):
        try:
            return FailureClassification(raw)
        except ValueError:
            pass
    if execution.result.preflight_result:
        try:
            from tools.repository_preflight import RepositoryPreflightResult

            return classify_failure(
                preflight=RepositoryPreflightResult.model_validate(
                    execution.result.preflight_result
                )
            )
        except (TypeError, ValueError):
            pass
    if execution.result.requirements_not_implemented:
        return FailureClassification.IMPLEMENTATION_MISSING
    return FailureClassification.REVIEW_SCOPE_FAILURE


def _with_fault_evidence(
    execution: ChildExecution,
    *,
    fault_count: int,
    fault_seconds_excluded: float,
    fault_classes: Sequence[str],
    truncated_degraded_retry: bool,
    wall_seconds: float,
    charged_seconds: float,
) -> ChildExecution:
    """Fold the attempt's absorbed provider-fault history into its durable result record.

    Artifact metadata, never child-state columns: no migration, no `_replace_state`
    projection, and nothing for `extra="forbid"` to trap. Zero-fault attempts record zeros
    and false explicitly -- absence has to keep meaning "recorded before this shipped".

    **The measured clocks overwrite; they used to defer.** Filling them "only where the
    result does not already carry a value" was reasoning about the ordinary path, where the
    result is built after this and the later measurement is the better one. On the
    provider-fault terminal path there is no later measurement: the result is built from a
    child that already carries the PREVIOUS attempt's numbers, so the correctly-measured
    values computed at the ceiling were discarded in favour of them. AB-Feature-218's
    attempt 6 recorded `runtime_wall_seconds == runtime_charged_seconds == 2391.89`,
    byte-identical to attempt 5's, while reporting `fault_seconds_excluded: 2523.222` -- a
    record that simultaneously claims it excluded 2,523 s of fault and that its charged
    clock equals its wall clock, with both numbers describing a different, earlier attempt.
    The caller measures at the moment it returns and is the authority here.
    """
    metadata = dict(execution.result.metadata)
    metadata["runtime_wall_seconds"] = wall_seconds
    metadata["runtime_charged_seconds"] = charged_seconds
    result = execution.result.model_copy(
        update={
            "metadata": {
                **metadata,
                "fault_count": fault_count,
                "fault_seconds_excluded": round(fault_seconds_excluded, 3),
                "fault_classes": list(fault_classes),
                "truncated_degraded_retry": truncated_degraded_retry,
            }
        }
    )
    return ChildExecution(
        result=result,
        code_completion=execution.code_completion,
        review=execution.review,
        contract_change_request=execution.contract_change_request,
    )


def _with_child_diagnostics(
    execution: ChildExecution, child: ChildWorkflowReference
) -> ChildExecution:
    """Copy retry state into the immutable result returned to parent persistence."""
    metadata = {
        **execution.result.metadata,
        "child_retry_count": child.retry_count,
        "implementation_retry_count": child.implementation_retry_count,
        "validation_retry_count": child.validation_retry_count,
        "repository_setup_retry_count": child.repository_setup_retry_count,
        "integration_retry_count": child.integration_retry_count,
        # See the builder above: the attempt kind travels on the attempt's own record.
        "targeted_attempt_kind": (
            str(child.targeted_attempt_kind) if child.targeted_attempt_kind is not None else None
        ),
        "retry_refusal_reason": child.retry_refusal_reason,
        "runtime_wall_seconds": child.runtime_wall_seconds,
        "runtime_charged_seconds": child.runtime_charged_seconds,
    }
    result = execution.result.model_copy(
        update={
            "metadata": metadata,
            "preflight_result": child.preflight_result,
            "layout_evidence": child.layout_evidence,
            "failure_classification": child.failure_classification,
            "retry_strategy": child.retry_strategy,
            "production_files_changed": child.production_files_changed,
            "test_files_changed": child.test_files_changed,
            "configuration_files_changed": child.configuration_files_changed,
            "requirements_implemented": child.requirements_implemented,
            "requirements_not_implemented": child.requirements_not_implemented,
            "meaningful_change": child.meaningful_change,
            "meaningful_change_reason": child.meaningful_change_reason,
            "production_diff_fingerprint": child.production_diff_fingerprint,
            "test_diff_fingerprint": child.test_diff_fingerprint,
            "previous_attempt_fingerprint": child.previous_attempt_fingerprint,
            "current_revision": execution.result.current_revision or child.current_revision,
        }
    )
    return ChildExecution(
        result=result,
        code_completion=execution.code_completion,
        review=execution.review,
        contract_change_request=execution.contract_change_request,
    )


# How many superseded findings are carried forward as context. Unbounded, this list is the
# union of every complaint ever made about a workstream: -105's ninth attempt received
# "wire BulkDeleteApps into AppConfigModal" alongside "your AppConfigModal wiring can never
# enable the button", with nothing saying which was still true. A pile of contradictions is
# not feedback.
_MAX_CARRIED_FINDINGS = 4


def _causal_diagnostics(error: BaseException) -> tuple[str, ...]:
    """Return safe diagnostics from this error or from whatever actually caused it.

    A wrapper carries no explanation of its own. `ExternalOperationError` is raised around a
    journalled operation and says nothing about why the operation failed, so -105's backend
    was recorded as "raised ExternalOperationError and could not complete" -- while the
    `GitAdapterError` underneath it held `git clone failed (GIT_CLONE_FAILED_EXIT_128)`, the
    one sentence that would have identified a wrong repository URL in seconds.

    Only the same already-safe sources are read, at every level: a cause volunteers explicit
    diagnostics or it contributes nothing, so nothing new becomes quotable.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        diagnostics = safe_error_diagnostics(current)
        if diagnostics:
            return diagnostics
        current = current.__cause__ or current.__context__
    return ()


def _task_outcome(task: asyncio.Task[ChildExecution]) -> ChildExecution | BaseException:
    """Resolve one finished child task the way `gather(return_exceptions=True)` did.

    Kept identical on purpose: the caller distinguishes a cancellation from any other
    exception and from a result, and moving off `gather` must not change which branch a
    given ending takes -- only when it is reached.
    """
    if task.cancelled():
        return CancellationRequested()
    error = task.exception()
    return error if error is not None else task.result()


def _retry_feedback(
    current_findings: Sequence[str],
    *,
    unresolved_since: Mapping[str, int],
    attempt: int,
    inherited: Sequence[str] = (),
    superseded: Sequence[str] = (),
    invariants: Sequence[str] = (),
    strategy_change: str = "",
) -> list[str]:
    """Order the next attempt's feedback so the thing that just failed cannot be missed.

    Every line says its own status, because these are spliced into each task's description
    and nothing guarantees the order survives. Five kinds, most urgent first:

    * that the previous repair was made, was real, and left the failure exactly where it was;
    * what this attempt failed on, marked as the thing to fix;
    * the same thing when it has survived several attempts, which is the signal to stop
      repeating an approach rather than to try it once more;
    * what review has already demanded and been given, marked as not to be undone;
    * a bounded tail of earlier findings, marked as possibly already fixed.

    The first kind leads because it is the only line that is about the *approach* rather than
    about a defect, and a model that reads it after a list of findings has already started
    repairing them one at a time -- which is the strategy it is being told not to repeat.

    The fourth and fifth kinds are the two halves of "this is not what you are working on",
    and they say opposite things about it: an invariant was satisfied and must stay satisfied,
    a superseded finding may never have been. Merging them would tell an engineer to verify
    work that is already load-bearing, which is how it gets reverted.

    The distinction matters more than the wording. A model handed one undifferentiated list
    cannot tell a live defect from one it fixed four attempts ago, and -105 spent eleven
    attempts demonstrating that.
    """
    lines: list[str] = [strategy_change] if strategy_change else []
    for finding in current_findings:
        repeats = unresolved_since.get(finding, 1)
        if repeats > 1:
            lines.append(
                f"MUST FIX -- reported {repeats} attempts running and still failing. "
                f"Repeating the previous approach has not worked; change it: {finding}"
            )
        else:
            lines.append(f"MUST FIX -- attempt {attempt} failed on this: {finding}")
    for finding in list(inherited)[:_MAX_CARRIED_FINDINGS]:
        if finding not in current_findings:
            lines.append(f"Also required by the integration review: {finding}")
    # Already bounded and already self-describing by the time they arrive: the ledger renders
    # whole issues under a character budget and states what it left out, so this splices them
    # in without a second cap that would silently drop what the first one kept.
    lines.extend(invariants)
    carried = [item for item in superseded if item not in current_findings]
    for finding in carried[-_MAX_CARRIED_FINDINGS:]:
        lines.append(
            f"Reported by an earlier attempt and not by the last one, so it may already be "
            f"fixed -- verify before acting on it: {finding}"
        )
    return list(dict.fromkeys(lines))


def _retry_budget(state: FeatureWorkflowSnapshot, classification: FailureClassification) -> int:
    """Keep repository setup, validation, implementation, and integration budgets separate."""
    if classification is FailureClassification.VALIDATION_SOURCE_FAILURE:
        return state.max_validation_retries
    if classification in {
        FailureClassification.DEPENDENCY_INSTALLATION_FAILURE,
        FailureClassification.VALIDATION_CAPACITY_FAILURE,
        FailureClassification.VALIDATION_CONFIGURATION_FAILURE,
        FailureClassification.TEST_INFRASTRUCTURE_MISSING,
    }:
        return state.max_repository_setup_retries
    if classification is FailureClassification.CONTRACT_MISMATCH:
        # One shared allowance. A child re-attempting a contract mismatch draws down the
        # same budget as the parent integration loop that asked for the fix, instead of
        # each loop independently spending the full configured limit.
        return max(0, state.max_integration_review_cycles - state.integration_review_cycles)
    return state.max_implementation_retries


def _clock_seconds(value: object) -> float | None:
    """Read a recorded clock out of result metadata, tolerating records without one."""
    return float(value) if isinstance(value, (int, float)) and value >= 0 else None


def _production_fingerprint(paths: Sequence[str]) -> str:
    """Persist a stable non-secret fingerprint of reported production-file scope."""
    return hashlib.sha256("\n".join(sorted(paths)).encode("utf-8")).hexdigest()


def _execution_artifacts(execution: ChildExecution) -> list[Artifact]:
    """Return outcome artifacts in deterministic handoff order."""
    artifacts: list[Artifact] = []
    if execution.code_completion is not None:
        artifacts.append(execution.code_completion)
    if execution.review is not None:
        artifacts.append(execution.review)
    artifacts.append(execution.result)
    return artifacts


def _with_attempt_identity(
    execution: ChildExecution, *, repository_id: str, attempt: int
) -> ChildExecution:
    """Canonicalize one executor outcome and rewire every reference to that exact attempt."""
    completion = execution.code_completion
    review = execution.review
    original_completion_id = completion.artifact_id if completion is not None else None
    completion_id: str | None = None
    review_id: str | None = None
    if completion is not None:
        completion_id = attempt_artifact_id(
            _repository_artifact_id(ARTIFACT_FILENAMES["code_completion"], repository_id),
            attempt,
        )
        completion = completion.model_copy(
            update={
                "artifact_id": completion_id,
                "metadata": {**completion.metadata, "child_attempt": attempt},
            }
        )
    if review is not None:
        if completion is None or completion_id is None:
            msg = "a child review cannot exist without its attempt's code completion"
            raise FeatureWorkflowError(
                msg,
                diagnostic=(
                    "A child review was recorded without the code completion its attempt produced."
                ),
            )
        assert original_completion_id is not None
        review_id = attempt_artifact_id(
            _repository_artifact_id(ARTIFACT_FILENAMES["review"], repository_id),
            attempt,
        )
        source_ids = _replace_source_artifact_id(
            review.metadata.get("source_artifact_ids"),
            old_id=original_completion_id,
            new_id=completion_id,
        )
        if completion_id not in source_ids:
            source_ids.append(completion_id)
        review = review.model_copy(
            update={
                "artifact_id": review_id,
                "metadata": {
                    **review.metadata,
                    "source_artifact_ids": source_ids,
                    "child_attempt": attempt,
                },
            }
        )
    source_artifact_ids = [
        artifact_id for artifact_id in (completion_id, review_id) if artifact_id is not None
    ]
    result = execution.result.model_copy(
        update={
            "artifact_id": attempt_artifact_id(
                _repository_artifact_id(
                    FEATURE_ARTIFACT_FILENAMES["child_workflow_result"], repository_id
                ),
                attempt,
            ),
            "code_completion_artifact_id": completion_id,
            "review_artifact_id": review_id,
            "metadata": {
                **execution.result.metadata,
                "source_artifact_ids": source_artifact_ids,
                "child_retry_count": attempt,
            },
        }
    )
    return ChildExecution(
        result=result,
        code_completion=completion,
        review=review,
        contract_change_request=execution.contract_change_request,
    )


def _replace_source_artifact_id(value: object, *, old_id: str, new_id: str) -> list[str]:
    """Rewrite a metadata reference list without trusting malformed executor metadata."""
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return []
    return list(dict.fromkeys(new_id if item == old_id else item for item in value))


def _repository_artifact_id(base_artifact_id: str, repository_id: str) -> str:
    """Namespace child artifacts that share one parent feature history."""
    return f"{base_artifact_id.removesuffix('.json')}.{repository_id}.json"


def _latest_approved_result(
    state: FeatureWorkflowSnapshot, repository_id: str
) -> ChildWorkflowResultArtifact | None:
    """Return the newest reviewed, PR-ready result for a repository whose last attempt failed.

    A repository can pass review and push its commit, then be sent back by integration review
    and fail the rework. Only the final attempt reaches the publication gate, so the reviewed
    commit sat on the remote attached to nothing but a note asking a human to go find it. A
    rejected attempt commits nothing, so the branch still holds exactly this commit.
    """
    approved: ChildWorkflowResultArtifact | None = None
    for artifact in state.artifacts:
        if (
            isinstance(artifact, ChildWorkflowResultArtifact)
            and artifact.repository_id == repository_id
            and artifact.status == "approved"
            and artifact.pull_request_readiness
            # A superseded run's approval is not this revision's fallback: publishing it
            # again would open a second pull request for work a person just asked to change.
            and _artifact_revision(artifact) == state.revision
        ):
            approved = artifact
    return approved


# What a title says when the repository review rejected the work and a person published it
# anyway. Named once so the title, the tests that assert it, and anybody grepping for it read
# the same string.
_UNREVIEWED_TITLE_MARKER = "[REVIEW REJECTED]"


def _pull_request_title(
    feature: FeatureWorkflowSnapshot, repository: RepositorySpec, *, unreviewed: bool = False
) -> str:
    """Title one pull request so its feature is identifiable from a list of pull requests.

    The prefix is the feature's reference -- `[AB-Feature-42]` -- because that is what a
    person quotes, and every pull request a feature opens carries the same one. It used to be
    the internal id, which for the pilot runs produced titles like
    `[AI Feature adunit-deactivate-live-086]`: unique, and meaningless to a reviewer.

    A repository qualifier is added only when the feature has more than one, where the titles
    would otherwise be identical across repositories. A feature with a single repository gets
    the plain title, which is what somebody would have written by hand.

    The qualifier is the repository's name unless the planner assigned it a meaningful role.
    `other` is what a repository gets when nobody stated one -- which is the common case, since
    a saved organisational label is deliberately not sent as a role -- and two pull requests
    both reading "other:" say less than their repository names would.

    A feature with no reference keeps the old prefix rather than being given a fabricated one.

    ``unreviewed`` marks the one class of pull request whose code the repository review
    rejected and a person published anyway. It goes in the prefix, in capitals, because a
    draft pull request that reads like every other draft pull request but contains rejected
    code is exactly the thing that gets merged on a Friday.
    """
    label = feature.reference or f"AI Feature {feature.feature_id}"
    # A revision's pull request replaces one a reviewer may still have open in a tab, so the
    # title has to say which one they are looking at. V2 is the first revision: the original
    # run carries no marker, exactly as it did before revisions existed.
    version = f" [V{feature.revision + 1}]" if feature.revision else ""
    prefix = f"[{label}]{version}" + (f" {_UNREVIEWED_TITLE_MARKER}" if unreviewed else "")
    if len(feature.repository_specs) < 2:
        return f"{prefix} {feature.title}"
    role = str(repository.role or "").strip()
    qualifier = repository.name if role in {"", WorkstreamRole.OTHER.value} else role
    return f"{prefix} {qualifier}: {feature.title}"


def _asked_question_ids(technical_prd: TechnicalPRDArtifact) -> set[str]:
    """Return every clarification this feature has already put to a human, open or answered.

    Both halves matter. Without the open ones a question is duplicated; without the answered
    ones a feature re-asks what it was just told and never leaves the clarification gate.
    """
    answered = technical_prd.metadata.get("clarification_answers")
    return {item.question_id for item in technical_prd.unresolved_questions} | (
        set(answered) if isinstance(answered, dict) else set()
    )


# What an answer is sent for reconciliation with when no revision in this feature's history
# still carries the question. Stated rather than omitted: the answer is the human's decision
# either way, and a reconciliation told nothing about the missing half must not read the empty
# string as though the question had been blank.
_MISSING_QUESTION_TEXT = "(this feature's history no longer holds the question text)"


def _reconciled_answer_ids(technical_prd: TechnicalPRDArtifact) -> set[str]:
    """Return the answers whose decisions this PRD's requirement text already states."""
    recorded = technical_prd.metadata.get("reconciled_answer_ids")
    return {str(item) for item in recorded} if isinstance(recorded, list) else set()


def _unreconciled_answers(
    artifacts: Sequence[Artifact], technical_prd: TechnicalPRDArtifact
) -> list[AnsweredClarification]:
    """Return the accepted answers whose decisions the requirement text does not yet state.

    Empty is the answer for a feature nobody was asked about, which is most of them, and it is
    what keeps this from costing a model call on a run with no clarification round at all.

    The question text comes out of the feature's own artifact history rather than the newest
    revision: `apply_clarification_answers` empties `unresolved_questions` when it records the
    answers, so by the time this runs, the question a person answered survives only on the
    revision it superseded. An answer without its question is frequently unreadable -- "use
    compensating deletes" settles nothing on its own -- and the history is append-only, so the
    text is still there to be read.
    """
    accepted = technical_prd.metadata.get("clarification_answers")
    if not isinstance(accepted, dict) or not accepted:
        return []
    already = _reconciled_answer_ids(technical_prd)
    asked: dict[str, str] = {}
    for artifact in artifacts:
        if isinstance(artifact, TechnicalPRDArtifact):
            for question in artifact.unresolved_questions:
                asked[question.question_id] = question.question
    return [
        AnsweredClarification(
            question_id=question_id,
            question=asked.get(question_id, _MISSING_QUESTION_TEXT),
            answer=str(answer),
        )
        for question_id, answer in sorted(accepted.items())
        if question_id not in already
    ]


def _open_questions_reason(questions: Sequence[ClarificationQuestion]) -> str:
    """Say why this feature is waiting, in the words that are true of what it is waiting on.

    Two kinds of round reach the same gate. One asks about premises the checkouts contradicted
    and criteria no diff could demonstrate -- questions the repositories raised. The other
    asks which of two statements a person's own answers left standing, which no repository has
    anything to say about; telling its author the repositories could not answer it would send
    them to read a checkout about a decision only they can make.
    """
    count = len(questions)
    if any(item.question_id.startswith(QUESTION_ID_PREFIX) for item in questions):
        return (
            f"This feature has {count} open question(s): its answers cannot all be written "
            "into its requirements, so a person has to say which holds."
        )
    return (
        f"This feature has {count} open question(s) that the repositories themselves could "
        "not answer."
    )


def _new_contradiction_questions(
    technical_prd: TechnicalPRDArtifact,
    reconciliation: RequirementReconciliation,
    answers: Sequence[AnsweredClarification],
) -> list[ClarificationQuestion]:
    """Turn what reconciliation could not settle into questions this feature has not asked.

    The already-asked filter is the same one the reconnaissance and criteria questions pass
    through, and it is what makes a second round converge: a contradiction whose question is
    already answered must not be asked again, or a person is asked the identical question every
    round until the rounds cap ends the feature. What remains after the filter is a *new*
    contradiction, which is the only thing worth another round.
    """
    if reconciliation.settles_everything:
        return []
    asked = _asked_question_ids(technical_prd)
    return [
        question
        for question in contradiction_questions(
            reconciliation.contradictions,
            answers=answers,
            requirements=[
                *technical_prd.functional_requirements,
                *technical_prd.non_functional_requirements,
            ],
        )
        if question.question_id not in asked
    ]


def _unverifiable_criteria_questions(
    technical_prd: TechnicalPRDArtifact,
) -> list[ClarificationQuestion]:
    """Ask about acceptance criteria no repository change could ever demonstrate.

    The reviewer withholds these so an attempt is not failed for missing a measurement it
    cannot take, but withholding decides on the author's behalf and tells them nothing. Asking
    here costs one round and happens before any code is written; the alternative is a
    requirement quietly dropped, or -- on the paths that did enforce it -- a workstream
    rejected for it five times over.
    """
    return [
        ClarificationQuestion(
            question_id=f"criteria-{item.requirement_id}-{index}",
            question=(
                f'Requirement {item.requirement_id} is accepted on: "{item.criterion}". '
                "Nothing in a repository change can demonstrate that. Restate it as behaviour "
                "visible in the code, or confirm it is verified outside this workflow."
            ),
            rationale=(
                "A reviewer judges the diff and the output of the repository's own commands. "
                "This criterion names a measurement against a running deployment, so it would "
                "either be withheld from the review without telling you, or block work that "
                "no attempt could unblock."
            ),
            required=True,
        )
        for index, item in enumerate(
            unverifiable_criteria(
                [
                    *technical_prd.functional_requirements,
                    *technical_prd.non_functional_requirements,
                ]
            ),
            start=1,
        )
    ]


def _premise_questions(
    reconnaissance: Sequence[RepositoryReconnaissanceArtifact],
) -> list[ClarificationQuestion]:
    """Turn every premise a checkout contradicts into a question asked before planning.

    Deliberately uncapped. Each entry is a premise that would otherwise be written into a
    workstream and spend its whole retry budget failing, so dropping any to keep the round
    short trades a short round for a dead repository. A feature that raises a great many of
    these is telling the operator something true: it is not the feature these repositories
    can accept.
    """
    questions: list[ClarificationQuestion] = []
    for artifact in reconnaissance:
        for index, premise in enumerate(artifact.contradicted_premises, start=1):
            question_id = f"recon-{artifact.repository_id}-{index}"
            questions.append(
                ClarificationQuestion(
                    question_id=question_id,
                    question=premise.question,
                    # The evidence travels with the question. A human asked to choose between
                    # conventions cannot do it from the question alone, and the paths are the
                    # part they can go and check.
                    rationale=(
                        f"In {artifact.repository_id}, the requirements assume: "
                        f"{premise.premise} The checkout instead has: {premise.contradicted_by} "
                        f"Evidence: {', '.join(premise.evidence_paths)}."
                    ),
                    required=True,
                    # The suggestion is what the checkout actually contains, not a preference.
                    # Nine times in ten the right answer to "the requirements assume X, the
                    # repository has Y" is "use Y", and making somebody retype Y from the
                    # rationale is asking them to transcribe a fact the platform just read.
                    # It stays editable, because the tenth time the answer is "add X".
                    suggested_answer=(
                        f"Follow what {artifact.repository_id} already does: "
                        f"{premise.contradicted_by}"
                    ),
                    suggestion_source=(
                        f"Suggested from repository analysis of {artifact.repository_id} "
                        f"({', '.join(premise.evidence_paths)})"
                    ),
                    # High: this is a statement about a checkout that was read, not an
                    # inference about what somebody wants.
                    suggestion_confidence="high",
                )
            )
    return questions


def _existing_reconnaissance(
    state: FeatureWorkflowSnapshot,
) -> list[RepositoryReconnaissanceArtifact]:
    """Return reconnaissance this feature already holds, one artifact per repository.

    A clarification answer sends the feature back through planning, and a recovery may too.
    Re-reading the same checkouts would spend another model call per repository to restate
    facts the feature already recorded, so the existing artifacts are reused.
    """
    by_repository: dict[str, RepositoryReconnaissanceArtifact] = {}
    for artifact in state.artifacts:
        if isinstance(artifact, RepositoryReconnaissanceArtifact):
            by_repository[artifact.repository_id] = artifact
    return list(by_repository.values())


def _latest_child_results(state: FeatureWorkflowSnapshot) -> list[ChildWorkflowResultArtifact]:
    """Select the last result per repository after retries without rerunning successful siblings."""
    results: dict[str, ChildWorkflowResultArtifact] = {}
    for artifact in state.artifacts:
        if isinstance(artifact, ChildWorkflowResultArtifact) and (
            # Selectors read within the current revision: a superseded run's results are
            # history, not candidates, and artifacts written before revisions read as 0.
            _artifact_revision(artifact) == state.revision
        ):
            results[artifact.repository_id] = artifact
    return [
        results[item.repository_id]
        for item in state.repository_specs
        if item.repository_id in results
    ]


def _latest_approved_child_results(
    state: FeatureWorkflowSnapshot,
) -> list[ChildWorkflowResultArtifact]:
    """Select each repository's newest publishable result, ignoring later failed attempts.

    Publication reads the approved revision, so the integration gate has to read the same
    one. Taking the last artifact instead meant a failed integration remediation deleted the
    approval from this gate's view, and it then reported the repository as having produced
    nothing at all -- about a repository whose pull request was being opened at that moment.
    """
    approved: dict[str, ChildWorkflowResultArtifact] = {}
    for artifact in state.artifacts:
        if (
            isinstance(artifact, ChildWorkflowResultArtifact)
            and artifact.status == "approved"
            and artifact.pull_request_readiness
            # Within the current revision only: a revision must earn its own approvals,
            # or a failed revision workstream would republish the superseded run's work.
            and _artifact_revision(artifact) == state.revision
        ):
            approved[artifact.repository_id] = artifact
    return [
        approved[item.repository_id]
        for item in state.repository_specs
        if item.repository_id in approved
    ]


def _record_executor_identity(state: FeatureWorkflowSnapshot) -> None:
    """Refuse incompatible durable state and record the build executing this mutation."""
    identity = load_runtime_identity()
    error = workflow_mutation_identity_error(
        workflow_schema_version=state.workflow_schema_version,
        created_by_build_revision=state.created_by_build_revision,
        last_executor_build_revision=state.last_executor_build_revision,
    )
    if error is not None:
        raise FeatureWorkflowError(
            error,
            diagnostic=(
                "This feature's durable state was written by an executor build this one "
                "cannot safely mutate."
            ),
        )
    if state.workflow_schema_version != identity.workflow_schema_version:
        raise FeatureWorkflowError(
            f"workflow state schema {state.workflow_schema_version!r} is incompatible with "
            f"executor schema {identity.workflow_schema_version!r}",
            diagnostic=(
                "This feature's workflow state schema is incompatible with the running executor."
            ),
        )
    state.last_executor_build_revision = identity.build_revision


def _resolve_target_workstream_ids(
    plan: RepositoryExecutionPlanArtifact, targets: set[str] | None
) -> set[str] | None:
    """Resolve repository-owned findings and legacy workstream targets to one scheduler key."""
    if targets is None:
        return None
    by_workstream = {item.workstream_id: item.workstream_id for item in plan.workstreams}
    by_repository = {item.repository_id: item.workstream_id for item in plan.workstreams}
    unknown = targets - set(by_workstream) - set(by_repository)
    if unknown:
        msg = f"targeted retry references unknown repositories/workstreams: {sorted(unknown)}"
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "A targeted retry named a repository workstream this feature does not have."
            ),
        )
    return {by_workstream.get(target, by_repository.get(target, "")) for target in targets}


def validate_clarification_answers(
    state: FeatureWorkflowSnapshot, answers: Sequence[ClarificationAnswer]
) -> None:
    """Refuse answers that cannot resolve this feature's open questions.

    Pure, for the same reason `require_retryable_workstream` is: a caller who sent the wrong
    question ids has made a correctable mistake and must be told so synchronously, before any
    execution is arranged. Resuming is queued work now, so a check left inside the runner
    would surface minutes later as a failed feature rather than as a rejected request.
    """
    if state.status != FeatureWorkflowStatus.WAITING_FOR_HUMAN:
        if answers:
            msg = (
                "clarification answers require a feature waiting for human input; this "
                f"feature is {state.status.value}"
            )
            raise ClarificationAnswerError(msg)
        return
    if state.clarification_rounds >= state.max_clarification_rounds:
        # Not an error: the runner turns this into a terminal state for a person to read,
        # and rejecting it here would deny the caller the explanation.
        return
    try:
        technical_prd = _require_artifact(state.artifacts, TechnicalPRDArtifact)
    except FeatureWorkflowError:
        if not answers:
            # A resume carrying no answers is a request to continue from the last
            # checkpoint, which needs no technical PRD. Left to the runner.
            return
        msg = (
            "this feature has no analysed questions, so there is nothing these answers "
            "could resolve"
        )
        raise ClarificationAnswerError(msg) from None
    expected = {item.question_id for item in technical_prd.unresolved_questions}
    received = {item.question_id for item in answers}
    if (
        not expected
        and not answers
        and state.failure_summary is not None
        and state.failure_summary.retryable
    ):
        # A retryable provider failure after accepted answers is a checkpoint recovery. The
        # answer revision was persisted before the failed call, so there is deliberately
        # nothing to answer again.
        return
    if not expected or received != expected:
        msg = (
            "clarification answers must match unresolved question IDs exactly. "
            f"Expected: {sorted(expected)}. Received: {sorted(received)}"
        )
        raise ClarificationAnswerError(msg)


def require_retryable_workstream(
    state: FeatureWorkflowSnapshot, *, repository_id: str, additional_attempts: int
) -> None:
    """Refuse a retry grant the platform could never carry out, before anything is spent.

    Every check here reads persisted state and nothing else. That is the point: a caller who
    typed the wrong repository id must be told so, and it must cost nothing -- no workspace
    maintenance, no provider client, no credentials. Ordering these after the live runner was
    built meant a typo reached the generic failure handler, which marked the feature
    ``failed_requires_human`` and answered 200, so the request that broke the feature looked
    like the one that fixed it.
    """
    if state.status in {
        FeatureWorkflowStatus.COMPLETED,
        FeatureWorkflowStatus.CANCELLED,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    }:
        msg = f"a {state.status.value} feature cannot retry a workstream"
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "A feature that has finished or been cancelled cannot be granted another "
                "workstream attempt."
            ),
        )
    if additional_attempts < 1:
        msg = "a retry grant must buy at least one attempt"
        raise FeatureWorkflowError(
            msg,
            diagnostic=("A retry grant must buy at least one attempt."),
        )
    child = state.child_workflows.get(repository_id)
    if child is None:
        known = sorted(state.child_workflows)
        msg = f"repository {repository_id!r} is not part of this feature; known: {known}"
        raise FeatureWorkflowError(
            msg,
            diagnostic=("A retry grant named a repository this feature does not have."),
        )
    if child.status not in _RETRYABLE_CHILD_STATUSES:
        # Retrying a repository that is running would race its own attempt, and retrying one
        # that was approved would discard work that already passed review.
        msg = (
            f"repository {repository_id!r} is {child.status.value} and has nothing to retry; "
            "only a stopped repository can be granted more attempts"
        )
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "Only a stopped repository can be granted more attempts; a running or approved one "
                "has nothing left to retry."
            ),
        )
    if child.retry_count + 1 >= state.max_child_review_cycles:
        # A grant raises this repository's classification budget; it does not raise the
        # review-cycle ceiling, and `decide_child_retry` checks that ceiling too. So a grant
        # here was accepted, cost a real clone, install and coding call, and was then refused
        # before a second attempt -- an override the platform took payment for and could not
        # honour. AB-Feature-111's console reached the ceiling and accepted three more.
        msg = (
            f"repository {repository_id!r} has used all {state.max_child_review_cycles} of its "
            "review cycles, which a grant does not raise; start a fresh feature rather than "
            "buying attempts this one cannot spend"
        )
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "This repository has used every review cycle it has, which a retry grant does not "
                "raise."
            ),
        )
    technical_prd = _require_artifact(state.artifacts, TechnicalPRDArtifact)
    if technical_prd.unresolved_questions:
        # The questions are what stopped this feature. Spending a granted attempt without
        # their answers reproduces the guess that failed.
        msg = "answer the open clarification questions before retrying a repository"
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "A granted attempt cannot run while this feature's clarification questions are "
                "unanswered."
            ),
        )


class WorkstreamPublicationClass(StrEnum):
    """Which of the two ways one repository is publishable on a person's instruction.

    Two classes, not a boolean, because what the publication does differs entirely. A
    `REVIEWED` repository already has its commit on the branch and only needs a pull request
    opened over it. An `UNREVIEWED` one has nothing on the branch at all -- a rejected attempt
    commits nothing -- so publishing it is commit, then push, then open, and the pull request
    it produces has to say in its title and its body that no review approved it.
    """

    REVIEWED = "reviewed"
    UNREVIEWED = "unreviewed"


def require_publishable_workstream(
    state: FeatureWorkflowSnapshot, *, repository_id: str
) -> WorkstreamPublicationClass:
    """Refuse a publication this platform could not carry out, and say which kind it would be.

    Shaped exactly like `require_retryable_workstream`: it raises `FeatureWorkflowError` with
    a diagnostic, so the refusal carries the sentence a disabled control shows, and
    `_workstream_actions` advertises the action by try/excepting it. It reads persisted state
    and one `stat`, and nothing else -- no workspace maintenance, no credentials, no provider.

    It returns the class rather than answering yes, because every caller needs both answers
    and deriving the second one separately is how the two drift apart.
    """
    child = state.child_workflows.get(repository_id)
    if child is None:
        known = sorted(state.child_workflows)
        msg = f"repository {repository_id!r} is not part of this feature; known: {known}"
        raise FeatureWorkflowError(
            msg,
            diagnostic="A publication named a repository this feature does not have.",
        )
    if child.pull_request_artifact_id is not None:
        msg = f"repository {repository_id!r} already has a pull request"
        raise FeatureWorkflowError(
            msg,
            diagnostic="This repository already has a pull request; it is not published twice.",
        )
    approved = _latest_approved_result(state, repository_id)
    if approved is not None:
        # Class 1. There is a commit on the branch already, put there by the attempt that
        # passed review, so nothing new is written and the ordinary publication path applies.
        return WorkstreamPublicationClass.REVIEWED
    result = _latest_result_for(state, repository_id)
    if result is None:
        msg = f"repository {repository_id!r} produced no result to publish"
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "This repository never produced a result, so there is nothing to publish for it."
            ),
        )
    if result.status != "failed":
        msg = f"repository {repository_id!r} is {result.status} and is not publishable"
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                f"This repository's last attempt ended as '{result.status}' rather than being "
                "rejected by a review, so there is no rejected work here to overrule."
            ),
        )
    if not result.changed_files:
        msg = f"repository {repository_id!r} changed nothing"
        raise FeatureWorkflowError(
            msg,
            diagnostic="This repository's last attempt changed no files, so it has nothing to "
            "publish.",
        )
    failing = _failing_required_checks(result)
    if failing is None:
        msg = f"repository {repository_id!r} recorded no validation evidence"
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "This repository's last attempt recorded no validation at all, so nothing "
                "shows its checks passed. A person may overrule a review's judgement; there "
                "is no judgement here to overrule."
            ),
        )
    if failing:
        msg = f"repository {repository_id!r} has failing required checks"
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "This repository's last attempt did not pass every check it requires "
                f"({', '.join(failing)}), so it is not offered for publication. A person may "
                "overrule a review's judgement; nobody overrules a failing command."
            ),
        )
    if not _is_git_worktree(result.workspace_path):
        # The shelf life of this action, said out loud -- and it is not the retention setting
        # the obvious guess names. A held feature is `failed_requires_human`, and
        # `_maintain_feature_workspaces` runs two policies, one per kind of ended feature:
        # `workspace_retention_features` (5) governs *completed and cancelled* features and
        # never applies here, while a failed one is bounded by
        # `workspace_failed_retention_hours` (72) *or* `workspace_failed_retention_features`
        # (10), either being enough to reclaim. So this action lasts roughly three days or ten
        # failed features, whichever reclaims first.
        #
        # A rejected attempt lives only in its worktree, so a feature left past that has
        # nothing left to commit. That is a real answer about this feature and it must arrive
        # as one, not as a `FileNotFoundError` from somewhere inside Git.
        msg = f"repository {repository_id!r} no longer has a checkout"
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "The checkout holding this repository's rejected work has been reclaimed, so "
                "the code no longer exists anywhere this platform can reach. Work a review "
                "rejected is never committed, so it exists only in that checkout -- kept by "
                "default for about three days after a feature ends needing a human, or for "
                "the ten most recent features that ended that way, whichever reclaims first."
            ),
        )
    return WorkstreamPublicationClass.UNREVIEWED


def feature_has_publishable_work(state: FeatureWorkflowSnapshot) -> bool:
    """Report whether anything on this feature would be published if somebody asked.

    `require_publishable_feature` as a question rather than a refusal, for the callers that
    have to ask it without provoking an exception. The capacity floor is the one that matters:
    it lets a feature past the free-space refusal precisely when publication is what the
    feature owes, and asking with the narrower "has approved unpublished work" predicate meant
    a feature whose only publishable work was clean-but-rejected was refused for disk space
    instead of published.
    """
    try:
        require_publishable_feature(state)
    except FeatureWorkflowError:
        return False
    return True


def require_publishable_feature(state: FeatureWorkflowSnapshot) -> None:
    """Refuse a publication request no repository on this feature could act on.

    Synchronous, and before anything is queued. A person who presses this on a feature with
    nothing to publish is told so in the response, rather than given a queue entry that fails
    on a worker minutes later with nobody watching.
    """
    if state.status in _UNRESUMABLE_STATUSES:
        msg = f"a {state.status.value} feature cannot be published"
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "A completed or cancelled feature cannot be published; whatever it opened is "
                "already open."
            ),
        )
    refusals: list[str] = []
    for repository_id in sorted(state.child_workflows):
        try:
            require_publishable_workstream(state, repository_id=repository_id)
        except FeatureWorkflowError as error:
            refusals.append(f"{repository_id}: {error.diagnostics[0]}")
        else:
            return
    msg = "no repository on this feature can be published"
    raise FeatureWorkflowError(
        msg,
        diagnostic=(
            "Nothing on this feature can be published. " + " ".join(refusals)
            if refusals
            else "This feature has no repository workstreams to publish."
        ),
    )


def _latest_result_for(
    state: FeatureWorkflowSnapshot, repository_id: str
) -> ChildWorkflowResultArtifact | None:
    """Return one repository's newest result, whatever it says.

    The last attempt's, and deliberately not the best by any measure: retries edit in place,
    so the worktree at `workspace_path` holds the final attempt's edits and no earlier one
    exists on disk to publish instead.
    """
    latest: ChildWorkflowResultArtifact | None = None
    for artifact in state.artifacts:
        if (
            isinstance(artifact, ChildWorkflowResultArtifact)
            and artifact.repository_id == repository_id
            and _artifact_revision(artifact) == state.revision
        ):
            latest = artifact
    return latest


def _publishable_result(
    state: FeatureWorkflowSnapshot,
    repository_id: str,
    publication_class: WorkstreamPublicationClass,
) -> ChildWorkflowResultArtifact | None:
    """Return the result each publication class publishes, which is not the same artifact."""
    if publication_class is WorkstreamPublicationClass.REVIEWED:
        return _latest_approved_result(state, repository_id)
    return _latest_result_for(state, repository_id)


def _failing_required_checks(result: ChildWorkflowResultArtifact) -> list[str] | None:
    """Name every required check this attempt did not pass, or None if it recorded none.

    `current_validation_results` first: it is the revision-bound set the attempt was actually
    judged on, and it carries `required`, which decides whether a verdict counted at all.
    `validation_results` is the fallback for a result written before that set existed; every
    entry in it is required by construction.

    A check whose record does not say whether it was required is counted as required. The
    question this answers is whether a person may overrule a *judgement*, and an unidentified
    failing command is not a judgement.

    None -- no evidence at all -- is distinct from an empty list and must stay so. A result
    that never reached validation cannot show its checks passed, and reporting it as clean
    would publish work nothing has ever run.
    """
    current = [item for item in result.current_validation_results if isinstance(item, Mapping)]
    if current:
        return [
            str(item.get("name") or item.get("command") or item.get("validation_type") or "a check")
            for item in current
            if item.get("required", True) and item.get("passed") is not True
        ]
    if result.validation_results:
        return [item.name for item in result.validation_results if not item.passed]
    return None


def _is_git_worktree(workspace_path: str) -> bool:
    """Report whether a recorded workspace path still holds a Git checkout.

    `.git` rather than the directory alone, because a reclaimed workspace is sometimes
    replaced by an empty one, and a directory with no repository in it is not a tree anything
    can be committed from. Both a directory (an ordinary clone) and a file (a linked
    worktree) count.
    """
    try:
        return (Path(workspace_path) / ".git").exists()
    except OSError:  # pragma: no cover - an unreadable path is an absent one here
        return False


def design_conflict_attempts_remaining(
    state: FeatureWorkflowSnapshot, child: ChildWorkflowReference
) -> int:
    """Return how many attempts this stopped workstream still has, on its own classification.

    Read rather than assumed, because it is what decides whether answering a design conflict
    has to buy anything. 49- Part B stops the loop without incrementing the counter -- the
    remaining budget is returned unspent, and the refusal says so -- so the ordinary case is a
    workstream that can act on a verdict immediately. A workstream whose budget really is gone
    is a different request, and the caller has to say so out loud.

    An unrecognised or absent classification answers zero rather than guessing a budget: the
    honest answer to "how many attempts does this have" when nothing recorded the kind of
    failure is none, and it makes the caller state the grant.
    """
    if child.failure_classification not in {item.value for item in FailureClassification}:
        return 0
    classification = FailureClassification(child.failure_classification)
    spent = int(getattr(child, retry_counter_field(classification), 0))
    budget = _retry_budget(state, classification) + child.granted_extra_attempts
    return max(0, budget - spent)


def require_answerable_design_conflict(
    state: FeatureWorkflowSnapshot,
    *,
    conflict: DesignConflictArtifact,
    additional_attempts: int,
) -> None:
    """Refuse a verdict the platform could not act on, before anything is recorded.

    Deliberately parallel to ``require_retryable_workstream``, including its ceiling rule: a
    verdict, like a grant, does not raise the review-cycle limit, and accepting one the loop
    could not honour would take a decision and then decline to act on it.

    One check is inverted rather than shared. A retry grant must buy at least one attempt --
    that is what a grant is. A verdict must not have to: the stop it answers spent nothing, so
    zero is the ordinary answer, and only a workstream with no attempts left needs a purchase.
    """
    if state.status in {
        FeatureWorkflowStatus.COMPLETED,
        FeatureWorkflowStatus.CANCELLED,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    }:
        msg = f"a {state.status.value} feature cannot act on a design decision"
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "A feature that has finished or been cancelled cannot act on a design decision."
            ),
        )
    if conflict.status != "open":
        # Answering twice is not idempotent the way approving a completed repair is: the second
        # answer would be a different decision recorded over the first, and the attempt the
        # first one authorised has already run on it.
        msg = f"design conflict {conflict.conflict_id} has already been decided"
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "This design question has already been decided and cannot be answered again."
            ),
        )
    if additional_attempts < 0:
        msg = "a design decision cannot buy a negative number of attempts"
        raise FeatureWorkflowError(
            msg,
            diagnostic=("A design decision cannot buy a negative number of attempts."),
        )
    child = state.child_workflows.get(conflict.repository_id)
    if child is None:
        known = sorted(state.child_workflows)
        msg = (
            f"design conflict {conflict.conflict_id} names repository "
            f"{conflict.repository_id!r}, which is not part of this feature; known: {known}"
        )
        raise FeatureWorkflowError(
            msg,
            diagnostic=("This design question names a repository this feature does not have."),
        )
    if child.status not in _RETRYABLE_CHILD_STATUSES:
        msg = (
            f"repository {conflict.repository_id!r} is {child.status.value}, so there is no "
            "stopped attempt for this decision to resume"
        )
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "Only a stopped repository can act on a design decision; a running or approved "
                "one has nothing to resume."
            ),
        )
    if child.retry_count + 1 >= state.max_child_review_cycles:
        msg = (
            f"repository {conflict.repository_id!r} has used all "
            f"{state.max_child_review_cycles} of its review cycles, which a design decision "
            "does not raise; start a fresh feature rather than deciding a question this one "
            "cannot act on"
        )
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "This repository has used every review cycle it has, which a design decision "
                "does not raise."
            ),
        )
    if not additional_attempts and not design_conflict_attempts_remaining(state, child):
        # The one place a verdict has to buy something. Saying so is the alternative to
        # silently topping the budget up, which would make an answered question look like a
        # refund of the attempts the stop declined to spend.
        msg = (
            f"repository {conflict.repository_id!r} has no attempts left to act on this "
            "decision; state how many to grant"
        )
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "This repository has no attempts left, so acting on this decision requires "
                "granting at least one."
            ),
        )
    technical_prd = _require_artifact(state.artifacts, TechnicalPRDArtifact)
    if technical_prd.unresolved_questions:
        msg = "answer the open clarification questions before deciding a design conflict"
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "An attempt cannot run while this feature's clarification questions are unanswered."
            ),
        )


def _child_retry_decision_is_terminal(
    state: FeatureWorkflowSnapshot, child: ChildWorkflowReference
) -> bool:
    """Report whether one repository's retry decision has already been made and spent.

    The intent is the one the feature-wide guard this replaces was written for, and it is
    correct: an empty resume must not silently reset a persisted terminal retry decision and
    repeat the coding and validation side effects that reached it. Only the scope changes.
    Asked feature-wide, it meant one terminal repository could permanently prevent its two
    healthy siblings from ever being resumed -- the resume returned the state unchanged, with
    no event, and the queue recorded success.

    Nothing here decides a *new* retry. `decide_child_retry` owns that and is untouched; this
    reads what it already wrote.
    """
    if child.status is not ChildWorkflowStatus.FAILED:
        return False
    if child.retry_refusal_reason or child.retry_count >= state.max_child_review_cycles - 1:
        return True
    if child.failure_classification is None:
        return False
    try:
        classification = FailureClassification(child.failure_classification)
    except ValueError:
        return False
    counter = getattr(child, retry_counter_field(classification))
    return bool(counter >= _retry_budget(state, classification))


def resume_eligible_repository_ids(state: FeatureWorkflowSnapshot) -> set[str]:
    """Return the repositories a resume may still run, which may be all or none of them."""
    return {
        repository_id
        for repository_id, child in state.child_workflows.items()
        if not _child_retry_decision_is_terminal(state, child)
    }


def _require_artifact[ArtifactModel: Artifact](
    artifacts: Sequence[Artifact], artifact_class: type[ArtifactModel]
) -> ArtifactModel:
    """Return the latest matching parent artifact, including replacement contract revisions."""
    for artifact in reversed(artifacts):
        if isinstance(artifact, artifact_class):
            return artifact
    msg = f"feature artifact is missing: {artifact_class.__name__}"
    raise FeatureWorkflowError(
        msg,
        diagnostic=(
            "A feature artifact this operation requires is missing from the feature's history."
        ),
    )


def _repair_artifact_id(repository_id: str, repair_id: str) -> str:
    """Name a proposal after the repository and the repair, so several can be open at once."""
    stem = FEATURE_ARTIFACT_FILENAMES["repository_repair_proposal"].removesuffix(".json")
    return f"{stem}.{repository_id}.{repair_id}.json"


def _design_conflict_artifact_id(conflict_id: str) -> str:
    """Name a conflict after itself, since the identifier already carries the repository."""
    stem = FEATURE_ARTIFACT_FILENAMES["design_conflict"].removesuffix(".json")
    return f"{stem}.{conflict_id}.json"


def _record_design_conflict(
    state: FeatureWorkflowSnapshot,
    recurrence: RecurringIssue,
    *,
    repository_id: str,
    child_workflow_id: str,
    question: str,
    evidence: Sequence[str],
    attempts_spent: int,
) -> None:
    """Write the stop's question down as something a person can answer.

    Idempotent on the conflict's identity, which is the repository plus the demand's
    fingerprint: a workstream can reach this stop again after an unrelated grant, and asking
    the same question twice would put two forms in front of the same person for one decision.
    An already-answered conflict is left alone too -- re-opening a decided question is exactly
    what the verdict exists to prevent.
    """
    payload = conflict_payload(
        recurrence,
        feature_id=state.feature_id,
        repository_id=repository_id,
        child_workflow_id=child_workflow_id,
        question=question,
        evidence=evidence,
        attempts_spent=attempts_spent,
    )
    conflict_id = str(payload["conflict_id"])
    if conflict_id in current_design_conflicts(state.artifacts):
        return
    _append_artifacts(
        state,
        [
            create_artifact(
                DesignConflictArtifact,
                workflow_id=state.feature_id,
                artifact_id=_design_conflict_artifact_id(conflict_id),
                producer="feature_workflow",
                payload=payload,
                metadata={"repository_id": repository_id, "source": "resolved_issue_ledger"},
            )
        ],
    )


def _record_recurring_demand_conflict(
    state: FeatureWorkflowSnapshot,
    theme: RecurringDemandTheme,
    *,
    repository_id: str,
    child_workflow_id: str,
    question: str,
    evidence: Sequence[str],
    attempts_spent: int,
) -> None:
    """Write a recurring demand down as something a person can answer (77-, item 35).

    Idempotent the way `_record_design_conflict` is, on the repository plus the theme's
    earliest wording's exact fingerprint -- stable across later rewordings, so a workstream
    stopped again after another reworded round finds its question already on file instead of
    putting a second form in front of the same person.
    """
    payload = recurring_demand_payload(
        theme,
        feature_id=state.feature_id,
        repository_id=repository_id,
        child_workflow_id=child_workflow_id,
        question=question,
        evidence=evidence,
        attempts_spent=attempts_spent,
    )
    conflict_id = str(payload["conflict_id"])
    if conflict_id in current_design_conflicts(state.artifacts):
        return
    _append_artifacts(
        state,
        [
            create_artifact(
                DesignConflictArtifact,
                workflow_id=state.feature_id,
                artifact_id=_design_conflict_artifact_id(conflict_id),
                producer="feature_workflow",
                payload=payload,
                metadata={"repository_id": repository_id, "source": "recurring_demand_theme"},
            )
        ],
    )


def revise_design_conflict(
    state: FeatureWorkflowSnapshot,
    conflict: DesignConflictArtifact,
    **updates: Any,
) -> DesignConflictArtifact:
    """Append a new state for one conflict, preserving the question that was answered.

    The same append-only rule the repairs follow, and for a sharper reason here: the two
    positions somebody read are the evidence for the verdict they gave, and overwriting them
    would leave a decision with nothing behind it.

    Revalidated rather than copied, unlike the repairs beside it. This artifact carries a
    validator that refuses a decision with no verdict, no words or nobody behind it, and
    ``model_copy`` does not run one -- so a copy would let a malformed verdict into the lineage
    the next attempt reads as binding.
    """
    revised = DesignConflictArtifact.model_validate(
        {**conflict.model_dump(mode="python"), **updates}
    )
    _replace_artifact(state, conflict, revised)
    return cast(DesignConflictArtifact, state.artifacts[-1])


def _propose_repository_repairs(state: FeatureWorkflowSnapshot) -> None:
    """Write down an approvable diagnosis for every repository stopped by its own setup.

    Only for repositories whose stop was about the repository rather than about the work:
    a lint configuration that cannot load, a lockfile that will not resolve. An ordinary
    failing test produces no proposal, because there is nothing about the repository for
    anybody to approve.

    Idempotent by construction -- a repository that already has an open proposal gets no
    second one -- so running it after every attempt costs nothing and cannot accumulate
    duplicates across retries.
    """
    for repository_id, child in state.child_workflows.items():
        if child.status not in {ChildWorkflowStatus.FAILED, ChildWorkflowStatus.REVIEW_REJECTED}:
            continue
        if open_repair_for(state.artifacts, repository_id) is not None:
            continue
        if not child.blocking_setup_issues:
            continue
        try:
            issues = [PreflightIssue.model_validate(item) for item in child.blocking_setup_issues]
        except ValidationError:
            # Setup findings persisted by an older build may not satisfy the current model.
            # A proposal that cannot be described precisely is not one worth approving.
            continue
        classification = (
            FailureClassification(child.failure_classification)
            if child.failure_classification in {item.value for item in FailureClassification}
            else None
        )
        selected = repairable_issues(issues, classification=classification)
        if not selected:
            continue
        # Only propose what the platform could actually carry out. A dependency the
        # repository never declared is fixed by declaring it, and without a command that
        # does so there is nothing here to approve -- an Approve button that changes nothing
        # is worse than saying plainly that a person has to make the change.
        if not repair_is_actionable(
            selected,
            classification=classification,
            package_manager=child.selected_package_manager,
        ):
            continue
        payload = build_repair_payload(
            feature_id=state.feature_id,
            repository_id=repository_id,
            child_workflow_id=child.child_workflow_id,
            # The stage that produced the diagnosis, not the readiness it produced. These
            # were conflated, so a card promising "found by" displayed "blocked" -- which
            # is a state the repository is in and not a place the finding came from.
            originating_stage="repository_preflight",
            classification=classification or "repository_setup",
            issues=selected,
            current_revision=child.current_revision,
            package_manager=child.selected_package_manager,
        )
        _append_artifacts(
            state,
            [
                create_artifact(
                    RepositoryRepairProposalArtifact,
                    workflow_id=state.feature_id,
                    artifact_id=_repair_artifact_id(repository_id, str(payload["repair_id"])),
                    producer="repository_preflight",
                    payload=payload,
                    metadata={"repository_id": repository_id, "source": "repository_preflight"},
                )
            ],
        )


def revise_repository_repair(
    state: FeatureWorkflowSnapshot,
    repair: RepositoryRepairProposalArtifact,
    **updates: Any,
) -> RepositoryRepairProposalArtifact:
    """Append a new state for one repair, preserving what it said before.

    A decision is recorded by appending, never by editing: the proposal somebody read is the
    evidence for the approval they gave, and overwriting it would erase what they agreed to.
    """
    revised = repair.model_copy(update=updates)
    _replace_artifact(state, repair, revised)
    # ``_replace_artifact`` assigns the appended revision its own identifier, so read back
    # what was actually stored rather than returning the copy that lacks it.
    return cast(RepositoryRepairProposalArtifact, state.artifacts[-1])


def _contract_change_request(
    state: FeatureWorkflowSnapshot, request_id: str
) -> ContractChangeRequestArtifact:
    """Find one pending request by its explicit external identifier."""
    for artifact in reversed(state.artifacts):
        if (
            isinstance(artifact, ContractChangeRequestArtifact)
            and artifact.change_request_id == request_id
        ):
            return artifact
    msg = f"contract change request not found: {request_id}"
    raise FeatureWorkflowError(
        msg,
        diagnostic=(
            "The contract change request this operation names is not in the feature's history."
        ),
    )


def _has_durable_child_commit(state: FeatureWorkflowSnapshot) -> bool:
    """Return whether any immutable child evidence proves repository history was committed."""
    return any(
        (isinstance(artifact, CodeCompletionArtifact) and artifact.commit_sha is not None)
        or (
            isinstance(artifact, ChildWorkflowResultArtifact)
            and isinstance(artifact.metadata.get("commit_sha"), str)
            and bool(artifact.metadata["commit_sha"])
        )
        for artifact in state.artifacts
    )


def _append_artifacts(state: FeatureWorkflowSnapshot, artifacts: Sequence[Artifact]) -> None:
    """Advance append-only history while refusing a newly reused artifact identifier."""
    known_ids = {artifact.artifact_id for artifact in state.artifacts}
    for artifact in artifacts:
        if artifact.artifact_id in known_ids:
            msg = f"artifact identifier already exists in feature history: {artifact.artifact_id}"
            raise FeatureWorkflowError(
                msg,
                diagnostic=(
                    "The feature's history already contains an artifact with this identifier."
                ),
            )
        known_ids.add(artifact.artifact_id)
    state.artifacts.extend(artifacts)
    state.updated_at = datetime.now(UTC)


def _replace_artifact(state: FeatureWorkflowSnapshot, old: Artifact, new: Artifact) -> None:
    """Append a uniquely identified revision while preserving the superseded evidence."""
    if old not in state.artifacts:
        msg = f"cannot revise unknown artifact: {old.artifact_id}"
        raise FeatureWorkflowError(
            msg,
            diagnostic=("The feature's history cannot revise an artifact it does not contain."),
        )
    revision_number = sum(isinstance(item, type(old)) for item in state.artifacts) + 1
    # Flattened, never nested. Revising a revision built `002_technical_prd.revision-2.
    # revision-3.json`, and `artifact_id_matches_lineage` accepts one qualifier by design, so
    # every child then failed with "required artifact is missing: 002_technical_prd.json".
    # Only one thing revised the technical PRD before reconnaissance existed; now that it adds
    # its questions first, answering a clarification is the second revision.
    stem = _REVISION_QUALIFIER.sub("", old.artifact_id.removesuffix(".json"))
    revision = new.model_copy(
        update={
            "artifact_id": f"{stem}.revision-{revision_number}.json",
            "timestamp": datetime.now(UTC),
            "metadata": {**new.metadata, "supersedes": old.artifact_id},
        }
    )
    state.artifacts.append(revision)
    state.updated_at = datetime.now(UTC)


def _require_acyclic_workstreams(workstreams: dict[str, RepositoryWorkstreamPlan]) -> None:
    """Reject a dependency cycle before any repository is cloned or modified.

    Artifact validation only enforces that declared orderings agree with each other, so a
    cycle reaching the scheduler would otherwise deadlock the fan-out loop silently.
    """
    scheduled: set[str] = set()
    pending = set(workstreams)
    while pending:
        ready = {
            workstream_id
            for workstream_id in pending
            if set(workstreams[workstream_id].dependency_workstream_ids) & pending == set()
        }
        if not ready:
            msg = (
                "repository execution plan contains a workstream dependency cycle: "
                f"{sorted(pending)}"
            )
            raise FeatureWorkflowError(
                msg,
                diagnostic=(
                    "The repository execution plan contains a workstream dependency cycle."
                ),
            )
        scheduled.update(ready)
        pending.difference_update(ready)


def _child_status(status: str) -> ChildWorkflowStatus:
    """Map serialized result statuses to independently persisted child lifecycle states."""
    return {
        "approved": ChildWorkflowStatus.APPROVED,
        "failed": ChildWorkflowStatus.FAILED,
        "waiting_for_contract_change": ChildWorkflowStatus.WAITING_FOR_CONTRACT_CHANGE,
        "cancelled": ChildWorkflowStatus.CANCELLED,
    }[status]


def _current_revision(review: ReviewArtifact | None) -> str | None:
    """Read the revision out of review metadata. The compatibility path, not the source.

    Kept because durable rows written before the executor measured the revision directly
    still depend on it, and reading one of those must not start returning ``None``. Nothing
    new should reach here: ``_child_result`` prefers the measured value, and every live
    executor path supplies one.

    Its limits are exactly why it stopped being primary. It needs a review to exist, so an
    attempt rejected at preflight, the commit gate, the wiring gate or completeness has
    nothing to read; and it needs unanimity, so an attempt whose lint and test commands
    observed different worktree states reports none either.
    """
    if review is None:
        return None
    validations = review.metadata.get("validation", [])
    if not isinstance(validations, list):
        return None
    revisions = {
        item.get("repository_revision")
        for item in validations
        if isinstance(item, dict) and isinstance(item.get("repository_revision"), str)
    }
    return next(iter(revisions)) if len(revisions) == 1 else None


def _test_availability(result: ChildWorkflowResultArtifact) -> str | None:
    """Expose truthful test configuration state without interpreting it as a passing suite."""
    validations = result.current_validation_results
    if any(
        isinstance(item, dict) and item.get("result_code") == "TEST_COMMAND_NOT_CONFIGURED"
        for item in validations
    ):
        return "TEST_COMMAND_NOT_CONFIGURED"
    if isinstance(result.preflight_result, dict):
        scripts = result.preflight_result.get("configured_scripts", {})
        if isinstance(scripts, dict) and ("test" in scripts or "test:ci" in scripts):
            return "tests_configured"
    return None


def _commit_sha(result: ChildWorkflowResultArtifact) -> str:
    """Read the child code artifact indirectly only after PR-readiness has been proven."""
    if result.code_completion_artifact_id is None:
        msg = f"child result has no code completion: {result.repository_id}"
        raise FeatureWorkflowError(
            msg,
            diagnostic=(
                "A child result claimed publishable work without the code completion that produced "
                "it."
            ),
        )
    # Mock and live child executors record the commit in the artifact metadata for the PR publisher.
    # The deterministic synthetic SHA remains non-sensitive and is only used when no live artifact
    # store is bound to the publisher.
    return str(result.metadata.get("commit_sha") or f"feature-{_slug(result.child_workflow_id)}")


def _contract_version_key(version: str) -> tuple[int, int, int]:
    """Order schema-validated semantic contract versions before accepting a revision."""
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def _mark_published_children(
    state: FeatureWorkflowSnapshot, pull_requests: Sequence[PullRequestArtifact]
) -> None:
    """Expose every successfully created PR even if a later repository publication fails."""
    for artifact in pull_requests:
        repository_id = next(
            item.repository_id
            for item in state.repository_specs
            if _repository_name(str(item.repository_url)) == artifact.repository
        )
        child = state.child_workflows[repository_id]
        state.child_workflows[repository_id] = child.model_copy(
            update={
                "status": ChildWorkflowStatus.COMPLETED,
                "pull_request_artifact_id": artifact.artifact_id,
            }
        )


def _repository_name(repository_url: str) -> str:
    """Extract ``owner/repository`` from a credential-free HTTPS GitHub repository URL."""
    path = repository_url.split("//", maxsplit=1)[-1].split("/", maxsplit=1)
    if len(path) != 2:
        msg = "repository URL must contain an owner and repository"
        raise FeatureWorkflowError(
            msg,
            diagnostic=("A repository URL must name an owner and a repository."),
        )
    parts = [item for item in path[1].removesuffix(".git").split("/") if item]
    if len(parts) < 2:
        msg = "repository URL must contain an owner and repository"
        raise FeatureWorkflowError(
            msg,
            diagnostic=("A repository URL must name an owner and a repository."),
        )
    return "/".join(parts[-2:])


def _pull_request_body(
    feature: FeatureWorkflowSnapshot,
    child: ChildWorkflowReference,
    result: ChildWorkflowResultArtifact,
    contract: IntegrationContractArtifact,
    *,
    draft: bool,
    integration_approved: bool = True,
) -> str:
    """Build a reviewable body while full cross-links are added after all PRs exist.

    This text is read by a human on GitHub, so it states only what happened. It previously
    asserted that the integration gate passed on every pull request it wrote, including the
    two paths that publish precisely because that gate did not approve: a required repository
    that never finished, and an integration review that exhausted its cycle budget.

    The same rule governs the repository review. Where the work was published without an
    approving verdict, the body says which verdict it carried and lists every finding that
    verdict rested on, because those findings are now a person's to judge and this is where
    that person reads them.

    Three voices now, dispatched here rather than selected by the caller: approved,
    accepted-with-advisory-findings, and -- for a result the review rejected outright, which
    only reaches publication because a person decided it should -- the unreviewed voice
    below. This stays the single producer of a body, because ``_cross_link`` and the
    artifact's own ``body`` field both assume there is only one.
    """
    if result.status != "approved":
        return _unreviewed_pull_request_body(
            feature,
            child,
            result,
            contract,
            draft=draft,
            integration_approved=integration_approved,
        )
    advisory_verdict = result.advisory_review_verdict
    return "\n".join(
        [
            f"Feature ID: {feature.feature_id}",
            f"Child workflow ID: {child.child_workflow_id}",
            f"Shared contract version: {contract.contract_version}",
            f"Implementation summary: {result.repository_id} workstream approved.",
            (
                (
                    "Validation summary: Repository review passed"
                    if advisory_verdict is None
                    # Both acceptance shapes are declared for what they were. The
                    # untraceable sentence would be false here: an overruled demand
                    # named its requirement perfectly well -- a person ruled it out.
                    else (
                        f"Validation summary: The repository review returned "
                        f"'{advisory_verdict}'; every finding that would have blocked "
                        "either re-raised a demand a person had already overruled by "
                        "design verdict (listed below with the decision) or named no "
                        "scoped requirement, contract section, failed required validation "
                        "command or implementation expectation, so the work is published "
                        "with every finding on the record here"
                        if result.overruled_findings
                        else f"Validation summary: The repository review returned "
                        f"'{advisory_verdict}', and every finding it raised named no scoped "
                        "requirement, contract section, failed required validation command or "
                        "implementation expectation, so each is listed below for review here "
                        "rather than blocking the change"
                    )
                )
                + "; the feature's integration gate "
                + ("approved." if integration_approved else "did NOT approve this feature.")
            ),
            *_review_finding_lines(result),
            f"Deployment order: {feature.merge_strategy or 'manual'}.",
            "Feature flag instructions: Follow the repository execution plan.",
            "Rollback instructions: Follow the feature rollback plan.",
            (
                "Known limitations: Pull requests are never auto-merged. The integration gate "
                "checks contract conformance only and reads no repository diff, so behaviour "
                "across repositories is not reviewed by it."
            ),
            "Status: draft" if draft else "Status: ready for review",
            "Related pull requests: added after all coordinated PRs are created.",
        ]
    )


def _review_finding_lines(result: ChildWorkflowResultArtifact) -> list[str]:
    """Render the findings a review left behind, for whichever voice is writing the body.

    Extracted rather than duplicated: an overruled finding carries a person's decision and an
    advisory one does not, and that distinction is the whole reason they are separate fields.
    A second copy of this rendering would be a second place for the distinction to be lost.
    """
    return [
        *(
            [
                "Overruled review findings (a person ruled these demands do not stand):",
                *(
                    f"- {item.description}\n  Overruled by "
                    f"{item.decided_by or 'an unrecorded decider'}"
                    + (
                        f" on {item.decided_at.date().isoformat()}"
                        if item.decided_at is not None
                        else ""
                    )
                    + (f": {item.decision}" if item.decision else ".")
                    for item in result.overruled_findings
                ),
            ]
            if result.overruled_findings
            else []
        ),
        *(
            [
                "Advisory review findings (not blocking; nobody has judged these yet):",
                *(f"- {item}" for item in result.advisory_findings),
            ]
            if result.advisory_findings
            else []
        ),
    ]


def _unreviewed_pull_request_body(
    feature: FeatureWorkflowSnapshot,
    child: ChildWorkflowReference,
    result: ChildWorkflowResultArtifact,
    contract: IntegrationContractArtifact,
    *,
    draft: bool,
    integration_approved: bool = True,
) -> str:
    """Say, without hedging, that the review rejected this work and a person published it.

    The reader of this body is a human on GitHub who is about to decide whether to merge code
    that the platform's own reviewer said no to. Every other body this module writes opens by
    saying the workstream was approved; this one has to open by saying the opposite, in the
    first line, before anything softens it.

    What the reviewer asked for is then listed rather than summarised. ``blocking_issues``
    holds its final blockers -- plus any stranded-branch or non-convergence note the workflow
    appended -- and the overruled and advisory findings beside them are rendered by the same
    helper the approved voice uses.
    """
    return "\n".join(
        [
            f"Feature ID: {feature.feature_id}",
            f"Child workflow ID: {child.child_workflow_id}",
            f"Shared contract version: {contract.contract_version}",
            (
                f"Implementation summary: the repository review REJECTED the "
                f"{result.repository_id} workstream. A person published it anyway, and this "
                "pull request holds that work."
            ),
            (
                "Validation summary: every check this repository requires passed, which is why "
                "publishing it was offered at all -- the reviewer's objection was a judgement "
                "about the change, not a failing command. Nobody has overruled that judgement; "
                "publishing it moves the decision to you."
            ),
            (
                "The feature's integration gate "
                + ("approved." if integration_approved else "did NOT approve this feature.")
            ),
            *(
                [
                    "What the review blocked on (these were not fixed):",
                    *(f"- {item}" for item in result.blocking_issues),
                ]
                if result.blocking_issues
                else ["What the review blocked on: the result recorded no blocking issue."]
            ),
            *(
                [f"Recorded failure classification: {result.failure_classification}."]
                if result.failure_classification
                else []
            ),
            *_review_finding_lines(result),
            f"Deployment order: {feature.merge_strategy or 'manual'}.",
            "Feature flag instructions: Follow the repository execution plan.",
            "Rollback instructions: Follow the feature rollback plan.",
            (
                "Known limitations: Pull requests are never auto-merged. The integration gate "
                "checks contract conformance only and reads no repository diff, so behaviour "
                "across repositories is not reviewed by it. This change additionally carries no "
                "approving repository review at all."
            ),
            "Status: draft" if draft else "Status: ready for review",
            "Related pull requests: added after all coordinated PRs are created.",
        ]
    )


def _slug(value: str) -> str:
    """Produce a deterministic Git-safe branch segment without accepting history syntax."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48] or "feature"


def _feature_branch_segment(feature_id: str) -> str:
    """Include a stable digest so differently punctuated IDs cannot collide in a repository."""
    digest = hashlib.sha256(feature_id.encode("utf-8")).hexdigest()[:10]
    return f"{_slug(feature_id)}-{digest}"


__all__ = [
    "ChildExecution",
    "ChildWorkstreamExecutor",
    "CoordinatedPullRequestPublisher",
    "DeterministicFeatureProductManager",
    "FeatureProductManager",
    "FeatureResumeNotEligible",
    "FeatureWorkflowError",
    "FeatureWorkflowOrchestrator",
    "FeatureWorkflowRunner",
    "GitHubPullRequestPublisher",
    "MockChildWorkstreamExecutor",
    "NullRepositoryReconnaissance",
    "RepositoryReconnaissance",
    "WorkstreamPublicationClass",
    "begin_revision",
    "feature_has_publishable_work",
    "feature_is_at_rest",
    "feature_may_be_resumed",
    "publication_is_held",
    "require_publishable_feature",
    "require_publishable_workstream",
    "require_resumable_feature",
    "require_retryable_workstream",
    "resume_eligible_repository_ids",
]
