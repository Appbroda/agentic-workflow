"""The logbook tells a feature's run from its records, and cannot tell it any other way.

Todo item 9. Every test here pins one of the acceptance conditions the item was filed with:

* A1 -- AB-Feature-184's shape renders from history alone, such that a non-technical reader
  can see where and why it died: the repositories' approvals, the integration review's
  refusal with the fix it required quoted, the fix attempt's platform failure with its
  diagnostic, and the pull requests that were salvaged anyway.
* A2 -- AB-Feature-186's attempt-0 sequence: truncation, the one degraded re-ask it earns,
  the absorbed provider fault with its excluded seconds, then the completion.
* A3 -- deferred honesty: an in-attempt repair pass is not a bubble while the attempt runs,
  and is one the moment it settles.
* A4 -- 1:1 anchoring: a bubble whose record cannot be resolved is a test failure.
* A5 -- redaction: quoted prose is what the source surface already shows, and the fields the
  journal endpoint withholds appear nowhere.
* A6 -- the registry is importable and callable without the tab or the endpoint, because
  Slack delivery is meant to reuse it rather than reimplement it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from artifacts.schemas import (
    BaseArtifact,
    ChildWorkflowResultArtifact,
    CodeCompletionArtifact,
    FeatureCompletionArtifact,
    IntegrationContractArtifact,
    IntegrationReviewArtifact,
    PRDArtifact,
    PullRequestArtifact,
    RepositoryExecutionPlanArtifact,
    ReviewArtifact,
    TechnicalPRDArtifact,
)
from main import create_app
from services.logbook import (
    LOGBOOK_TEMPLATES,
    OPERATION_STEP_VOICES,
    OPERATION_VOICES,
    LogbookAgent,
    LogbookEntry,
    LogbookEvent,
    LogbookRecordKind,
    UnknownLogbookTemplate,
    clip_quote,
    feature_logbook,
    render_bubble,
)
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.external_operations import (
    CONTRACT_PROJECTION_LOGICAL_STEP,
    ExternalOperation,
    ExternalOperationStatus,
    ExternalOperationType,
)
from state.feature_models import (
    ChildWorkflowReference,
    FeatureFailureSummary,
    FeatureWorkflowSnapshot,
)
from tests.support import settle
from tests.test_feature_api import feature_payload
from workflow_schema import WORKFLOW_SCHEMA_VERSION

START = datetime(2026, 9, 1, 4, 10, tzinfo=UTC)

# The artifact kinds these fixtures build, keyed by the type name the platform uses for them.
_ARTIFACT_CLASSES: dict[str, type[BaseArtifact]] = {
    "prd": PRDArtifact,
    "technical_prd": TechnicalPRDArtifact,
    "integration_contract": IntegrationContractArtifact,
    "repository_execution_plan": RepositoryExecutionPlanArtifact,
    "code_completion": CodeCompletionArtifact,
    "review": ReviewArtifact,
    "child_workflow_result": ChildWorkflowResultArtifact,
    "integration_review": IntegrationReviewArtifact,
    "pull_request": PullRequestArtifact,
    "feature_completion": FeatureCompletionArtifact,
}


def _at(minutes: int) -> datetime:
    """A deterministic instant, so the thread's order is a property of the fixture."""
    return START + timedelta(minutes=minutes)


def _artifact(artifact_type: str, artifact_id: str, minutes: int, **payload: Any) -> Any:
    """Build one durable artifact of the given kind, envelope included.

    Validated through the real union rather than faked, so a test cannot assert a field the
    persisted artifact could not carry -- which is how hand-written fixtures come to describe
    a record shape the platform never writes.
    """
    artifact_class: type[BaseArtifact] = _ARTIFACT_CLASSES[artifact_type]
    metadata = payload.pop("metadata", {})
    return artifact_class.model_validate(
        {
            "schema_version": "1.0.0",
            "workflow_id": "feature-184",
            "artifact_id": artifact_id,
            "producer": "test",
            "timestamp": _at(minutes),
            "metadata": metadata,
            "validation_status": "valid",
            **payload,
        }
    )


def _child(repository_id: str, *, status: ChildWorkflowStatus, **overrides: Any) -> Any:
    """Build one durable child workstream row."""
    values: dict[str, Any] = {
        "child_workflow_id": f"feature-184:{repository_id}",
        "repository_id": repository_id,
        "workstream_id": f"ws-{repository_id}",
        "status": status,
        "branch_name": f"feature/184-{repository_id}",
        "workspace_path": f"/workspaces/{repository_id}",
        "retry_count": 0,
    }
    values.update(overrides)
    return ChildWorkflowReference.model_validate(values)


def _state(
    *,
    artifacts: list[Any],
    children: dict[str, Any] | None = None,
    status: FeatureWorkflowStatus = FeatureWorkflowStatus.COMPLETED,
    failure_summary: FeatureFailureSummary | None = None,
) -> FeatureWorkflowSnapshot:
    """Build the durable parent snapshot the composer reads, and nothing more."""
    return FeatureWorkflowSnapshot.model_validate(
        {
            "feature_id": "feature-184",
            "workflow_id": "feature-184",
            "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
            "created_by_build_revision": "test",
            "last_executor_build_revision": "test",
            "status": status,
            "title": "Bulk add apps",
            "reference": "AB-Feature-184",
            "repository_specs": [
                {
                    "repository_id": "backend",
                    "name": "admanager-server",
                    "role": "backend",
                    "repository_url": "https://github.com/example/admanager-server",
                    "default_branch": "main",
                },
                {
                    "repository_id": "frontend",
                    "name": "admanager-client",
                    "role": "frontend",
                    "repository_url": "https://github.com/example/admanager-client",
                    "default_branch": "main",
                },
            ],
            "artifacts": artifacts,
            "child_workflows": children or {},
            "max_integration_review_cycles": 5,
            "max_child_review_cycles": 5,
            "max_contract_revision_cycles": 3,
            "failure_summary": failure_summary,
            "created_at": START,
            "updated_at": _at(120),
        }
    )


def _operation(
    operation_type: ExternalOperationType,
    operation_id: str,
    minutes: int,
    *,
    status: ExternalOperationStatus = ExternalOperationStatus.SUCCEEDED,
    repository_id: str | None = "backend",
    **overrides: Any,
) -> ExternalOperation:
    """Build one journal row as the journal itself returns it."""
    values: dict[str, Any] = {
        "operation_id": operation_id,
        "workflow_id": "feature-184",
        "feature_id": "feature-184",
        "repository_id": repository_id,
        "operation_type": operation_type,
        "idempotency_key": f"key-{operation_id}",
        "status": status,
        "attempt": 1,
        "max_attempts": 1,
        "input_fingerprint": "fingerprint",
        "started_at": _at(minutes),
        "completed_at": _at(minutes),
    }
    values.update(overrides)
    return ExternalOperation.model_validate(values)


def _templates(entries: list[LogbookEntry], *, record_id: str | None = None) -> list[str]:
    """The registry keys the thread used, in order -- optionally for one record only."""
    return [item.template for item in entries if record_id is None or item.record.id == record_id]


def _texts(entries: list[LogbookEntry]) -> str:
    """Everything the thread says, as one string, for "is this visible at all" assertions."""
    return "\n".join(
        " ".join(filter(None, (item.text, item.detail, item.quote))) for item in entries
    )


# --------------------------------------------------------------------------------------------
# A1 -- the 184 story
# --------------------------------------------------------------------------------------------


def _feature_184() -> tuple[FeatureWorkflowSnapshot, list[LogbookEvent]]:
    """AB-Feature-184's recorded shape: approved twice, refused at integration, dead on a defect.

    The run of 2026-09-01: both repositories were implemented, reviewed and approved; the
    integration review demanded atomicity of the backend; the fix attempt reached the
    publisher with nothing coded and tripped the reviewed-workspace guard, which surfaced as
    an unanticipated error and was classified `platform_defect`; the two pull requests the
    approved work had already produced survived.
    """
    backend_review = _artifact(
        "review",
        "007_review.backend.json",
        20,
        verdict="approved",
        summary="The bulk create endpoint validates every row before writing and returns "
        "per-row errors, which is what the requirement asked for.",
        requirement_checks=[],
        findings=[],
        architecture_assessment="Fits the existing controller/service split.",
        security_assessment="Admin middleware is applied.",
        test_coverage_assessment="Covered by the new suite.",
    )
    frontend_review = _artifact(
        "review",
        "007_review.frontend.json",
        22,
        verdict="approved",
        summary="The bulk form posts the whole batch and renders the per-row errors the "
        "endpoint returns.",
        requirement_checks=[],
        findings=[],
        architecture_assessment="Uses the existing API utility module.",
        security_assessment="No untrusted text is rendered.",
        test_coverage_assessment="Covered.",
    )
    backend_result = _artifact(
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
        code_completion_artifact_id="006_code_completion.backend.json",
        review_artifact_id="007_review.backend.json",
        changed_files=[],
        validation_results=[],
        status="approved",
        blocking_issues=[],
        pull_request_readiness=True,
        contract_sections_consumed=[],
        contract_sections_implemented=["POST /apps/bulk"],
    )
    frontend_result = _artifact(
        "child_workflow_result",
        "008_child_result.frontend.json",
        25,
        feature_id="feature-184",
        parent_workflow_id="feature-184",
        child_workflow_id="feature-184:frontend",
        repository_id="frontend",
        workstream_id="ws-frontend",
        branch_name="feature/184-frontend",
        workspace_path="/workspaces/frontend",
        code_completion_artifact_id="006_code_completion.frontend.json",
        review_artifact_id="007_review.frontend.json",
        changed_files=[],
        validation_results=[],
        status="approved",
        blocking_issues=[],
        pull_request_readiness=True,
        contract_sections_consumed=["POST /apps/bulk"],
        contract_sections_implemented=[],
    )
    integration_review = _artifact(
        "integration_review",
        "010_integration_review.json",
        30,
        feature_id="feature-184",
        contract_artifact_id="009_integration_contract.json",
        review_status="changes_requested",
        repository_results=[],
        contract_checks=["POST /apps/bulk matches the contract"],
        cross_repository_findings=[
            {
                "finding_id": "INT-001",
                "severity": "high",
                "responsible_repository_id": "backend",
                "affected_repository_ids": ["backend", "frontend"],
                "contract_reference": "POST /apps/bulk",
                "description": "A partially applied batch leaves the client unable to say "
                "which apps exist.",
                "evidence": "The endpoint writes each row in its own transaction.",
                "recommended_fix": "Wrap the batch in one transaction and compensate on failure.",
            }
        ],
        compatibility_assessment="The two repositories agree on the request and response "
        "shape, but not on what happens when half the batch fails.",
        security_assessment="No cross-repository security concern.",
        deployment_assessment="Backend must deploy first.",
        merge_order=["backend", "frontend"],
        required_fixes=[
            "Make POST /apps/bulk atomic: either every app in the batch is created or none "
            "is, and the response says which."
        ],
    )
    fix_attempt = _artifact(
        "child_workflow_result",
        "008_child_result.backend.fix.json",
        44,
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
        blocking_issues=[
            "The platform could not publish this repository: the workspace content no "
            "longer matches the source evidence the reviewer approved."
        ],
        pull_request_readiness=False,
        contract_sections_consumed=[],
        contract_sections_implemented=[],
        failure_classification="platform_defect",
    )
    backend_pr = _artifact(
        "pull_request",
        "011_pull_request.backend.json",
        40,
        repository="example/admanager-server",
        pull_request_number=318,
        url="https://github.com/example/admanager-server/pull/318",
        title="Bulk add apps (backend)",
        body="Adds POST /apps/bulk.",
        source_branch="feature/184-backend",
        target_branch="main",
        commit_sha="a1b2c3d",
        labels=[],
        reviewers=[],
        state="open",
    )
    frontend_pr = _artifact(
        "pull_request",
        "011_pull_request.frontend.json",
        41,
        repository="example/admanager-client",
        pull_request_number=204,
        url="https://github.com/example/admanager-client/pull/204",
        title="Bulk add apps (frontend)",
        body="Adds the bulk form.",
        source_branch="feature/184-frontend",
        target_branch="main",
        commit_sha="d4e5f6a",
        labels=[],
        reviewers=[],
        state="open",
    )
    summary = FeatureFailureSummary.model_validate(
        {
            "stage": "pull_request_publication",
            "agent": "publisher",
            "repository_id": "backend",
            "root_classification": "platform_defect",
            "attempt": 3,
            "diagnostics": [
                "Publication was refused because the workspace content no longer matches "
                "the source evidence approved by the reviewer."
            ],
            "retryable": False,
            "next_action": "Review the two pull requests this run already opened, then run "
            "the atomicity change as a fresh feature.",
            "recorded_at": _at(45),
        }
    )
    state = _state(
        artifacts=[
            backend_review,
            frontend_review,
            backend_result,
            frontend_result,
            integration_review,
            backend_pr,
            frontend_pr,
            fix_attempt,
        ],
        children={
            "backend": _child(
                "backend",
                status=ChildWorkflowStatus.FAILED,
                pull_request_artifact_id="011_pull_request.backend.json",
            ),
            "frontend": _child(
                "frontend",
                status=ChildWorkflowStatus.COMPLETED,
                pull_request_artifact_id="011_pull_request.frontend.json",
            ),
        },
        status=FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
        failure_summary=summary,
    )
    events = [
        LogbookEvent(id=1, timestamp=START, event="feature_started"),
        LogbookEvent(
            id=2,
            timestamp=START,
            event="feature_queued",
            details={"reference": "AB-Feature-184", "requested_by": "akhilesh"},
        ),
        # The artifact-announcement events the store writes beside every artifact. They must
        # not double the thread: the artifact itself is the richer bubble.
        LogbookEvent(
            id=3,
            timestamp=_at(30),
            event="integration_review_completed",
            details={
                "artifact_id": "010_integration_review.json",
                "review_status": "changes_requested",
            },
        ),
        LogbookEvent(
            id=4,
            timestamp=_at(30),
            event="repository_fix_requested",
            details={"artifact_id": "010_integration_review.json"},
        ),
        # Bookkeeping. Four of these fire per attempt and none of them is the story.
        LogbookEvent(
            id=5,
            timestamp=_at(43),
            event="feature_checkpoint_before_commit",
            details={"repository_id": "backend"},
        ),
        LogbookEvent(
            id=6,
            timestamp=_at(45),
            event="feature_failed",
            details={
                "error_type": "RuntimeConfigurationError",
                "failure_classification": "platform_defect",
                "diagnostics": [
                    "Publication was refused because the workspace content no longer "
                    "matches the source evidence approved by the reviewer."
                ],
            },
        ),
    ]
    return state, events


def test_the_184_story_says_where_and_why_it_died() -> None:
    """A1: the whole run reads as a story from history alone, ending where it actually ended."""
    state, events = _feature_184()
    entries = feature_logbook(state, lifecycle_events=events)
    keys = _templates(entries)

    # Both repositories were approved, and the thread says so before anything went wrong.
    assert keys.count("artifact.review.approved") == 2
    assert keys.count("artifact.child_workflow_result.approved") == 2
    # The integration review refused, and the fix it required is quoted rather than summarised.
    refusal = next(
        item for item in entries if item.template == "artifact.integration_review.changes_requested"
    )
    required = next(
        item for item in entries if item.template == "artifact.integration_review.required_fix"
    )
    assert "would not approve" in refusal.text
    assert required.quote is not None
    assert required.quote.startswith("Make POST /apps/bulk atomic")
    assert required.repository_id == "backend"
    # The fix attempt's platform failure, with its own diagnostic.
    fix = next(
        item
        for item in entries
        if item.template == "artifact.child_workflow_result.failed"
        and item.record.id == "008_child_result.backend.fix.json"
    )
    assert fix.quote is not None
    assert "no longer matches the source evidence" in fix.quote
    # The pull requests the approved work produced survived, and are in the thread.
    numbers = [item.text for item in entries if item.template == "artifact.pull_request"]
    assert any("#318" in item for item in numbers)
    assert any("#204" in item for item in numbers)
    # And the end says it was the platform, not the repository -- with what to do about it.
    stop = next(item for item in entries if item.template == "event.feature_failed")
    assert "the platform itself failed while running it" in stop.text
    action = next(item for item in entries if item.template == "event.feature_failed.next_action")
    assert action.quote is not None
    assert action.quote.startswith("Review the two pull requests")
    # Order: approvals, then the refusal, then the fix attempt, then the stop.
    order = [
        keys.index("artifact.review.approved"),
        keys.index("artifact.integration_review.changes_requested"),
        keys.index("artifact.child_workflow_result.failed"),
        keys.index("event.feature_failed"),
    ]
    assert order == sorted(order)


def test_the_184_story_is_told_once() -> None:
    """A1: an event that only announces an artifact in the thread is not a second bubble."""
    state, events = _feature_184()
    entries = feature_logbook(state, lifecycle_events=events)
    announcements = [item for item in entries if item.record.kind is LogbookRecordKind.EVENT]
    assert [item.record.id for item in announcements] == ["1", "2", "6", "6"]
    assert "event.unknown" not in _templates(entries)


def test_only_the_newest_stop_borrows_the_summarys_advice() -> None:
    """A1: a run that failed, was resumed and failed again has one live "what to do"."""
    state, events = _feature_184()
    earlier = LogbookEvent(
        id=0,
        timestamp=_at(5),
        event="feature_failed",
        details={
            "error_type": "APITimeoutError",
            "failure_classification": "provider_unavailable",
            "diagnostics": ["The model provider did not answer within the configured timeout."],
        },
    )
    entries = feature_logbook(state, lifecycle_events=[earlier, *events])
    stops = [item for item in entries if item.template == "event.feature_failed"]
    actions = [item for item in entries if item.template == "event.feature_failed.next_action"]
    assert len(stops) == 2
    # Each stop keeps its own diagnostic; only the last one carries the standing advice.
    assert stops[0].quote is not None
    assert "did not answer" in stops[0].quote
    assert stops[0].text == "The feature stopped: the model provider did not answer."
    assert len(actions) == 1
    assert actions[0].record.id == "6"


def test_a_git_outage_stops_the_feature_in_gits_name() -> None:
    """The logbook one-liner reads the classification, so 54- Part 2 never reached it.

    That change reworded the workstream's own narrative and left `provider_unavailable` in
    the field this sentence is keyed on -- so the logbook went on telling an operator that
    AB-Feature-190's GitHub outage was the model provider not answering.
    """
    state, events = _feature_184()
    outage = LogbookEvent(
        id=0,
        timestamp=_at(5),
        event="feature_failed",
        details={
            "error_type": "GitAdapterError",
            "failure_classification": "git_remote_unavailable",
            "diagnostics": ["The Git operation did not complete (GitAdapterError)."],
        },
    )
    entries = feature_logbook(state, lifecycle_events=[outage, *events])
    stop = next(item for item in entries if item.template == "event.feature_failed")
    assert stop.text == (
        "The feature stopped: the Git remote did not answer, or the platform could not reach it."
    )
    assert "model provider" not in stop.text


def test_a_provider_subtype_stops_the_feature_in_the_providers_name() -> None:
    """The Git half of the pair read correctly and the provider half did not.

    A terminal event's `failure_classification` retains the adapter's declared value verbatim,
    which for the model provider is the SDK exception class -- `APITimeoutError` -- and that
    is not a key in this table. So the outage that named Git got a sentence naming Git, and
    the identical outage against the provider got "kept its evidence". `normalize_classification`
    is the platform's one reading of those names, and it is now consulted for exactly the
    names it recognises.
    """
    state, events = _feature_184()
    outage = LogbookEvent(
        id=0,
        timestamp=_at(5),
        event="feature_failed",
        details={
            "error_type": "LLMAdapterError",
            "failure_classification": "APITimeoutError",
            "diagnostics": ["The model provider did not answer (LLMAdapterError)."],
        },
    )
    entries = feature_logbook(state, lifecycle_events=[outage, *events])
    stop = next(item for item in entries if item.template == "event.feature_failed")
    assert stop.text == "The feature stopped: the model provider did not answer."


def test_a_classification_this_table_has_never_heard_of_is_not_called_a_defect() -> None:
    """The fallback stays a gap and never becomes an accusation.

    `normalize_classification` answers `platform_defect` for everything it cannot read, so
    consulting it unguarded would turn every unmapped value into "the platform itself failed
    while running it" -- sending somebody to debug this codebase over a word nobody has
    taught the logbook yet.
    """
    state, events = _feature_184()
    unknown = LogbookEvent(
        id=0,
        timestamp=_at(5),
        event="feature_failed",
        details={"error_type": "SomethingNew", "failure_classification": "something_new"},
    )
    entries = feature_logbook(state, lifecycle_events=[unknown, *events])
    stop = next(item for item in entries if item.template == "event.feature_failed")
    assert stop.text == (
        "The feature stopped: the platform recorded it as stopped and kept its evidence."
    )


def test_an_unrecognised_event_still_renders() -> None:
    """A1's companion: a vocabulary this table has never heard of is never the silent one."""
    state, _ = _feature_184()
    entries = feature_logbook(
        state,
        lifecycle_events=[
            LogbookEvent(id=9, timestamp=_at(1), event="feature_teleported_sideways")
        ],
    )
    unknown = next(item for item in entries if item.template == "event.unknown")
    assert unknown.text == "The platform recorded feature_teleported_sideways."
    assert unknown.record.id == "9"


def test_a_held_publication_reaches_the_thread_saying_whose_decision_it_is() -> None:
    """The message that replaces the pull request a partial feature used to open.

    Task 81- stopped a feature that did not land from publishing itself. If the held state is
    quiet, that is the 2026-08 defect under a new name -- so the event carries its own
    sentence naming what is waiting, and it is classified as a human-interaction template so
    the thread CCs somebody rather than logging into the tab.
    """
    from services.slack_notifications import SLACK_HUMAN_INTERACTION_TEMPLATES

    state, _ = _feature_184()
    reason = (
        "This feature did not land: backend did not reach a review it passed, so nothing was "
        "published automatically. Ready to open on request: frontend."
    )
    entries = feature_logbook(
        state,
        lifecycle_events=[
            LogbookEvent(
                id=9,
                timestamp=_at(1),
                event="feature_publication_held",
                details={"reason": reason},
            )
        ],
    )

    held = next(item for item in entries if item.template == "event.feature_publication_held")
    assert "a person's decision" in held.text
    assert held.quote == reason
    assert "event.feature_publication_held" in SLACK_HUMAN_INTERACTION_TEMPLATES


def test_a_publication_a_person_asked_for_is_attributed_to_them() -> None:
    """A person's override is recorded as one, with their name on it rather than the agent's."""
    state, _ = _feature_184()
    entries = feature_logbook(
        state,
        lifecycle_events=[
            LogbookEvent(
                id=10,
                timestamp=_at(2),
                event="feature_published_by_person",
                details={
                    "requested_by": "akhilesh (via platform key)",
                    "reason": "akhilesh published this feature's finished work: example/frontend.",
                },
            )
        ],
    )

    published = next(
        item for item in entries if item.template == "event.feature_published_by_person"
    )
    assert published.text.startswith("akhilesh (via platform key) published")


# --------------------------------------------------------------------------------------------
# A2 -- the 186 attempt-0 sequence
# --------------------------------------------------------------------------------------------


def _feature_186_attempt_zero() -> Any:
    """186's attempt-0 result artifact: truncated, re-asked once degraded, then two faults."""
    return _artifact(
        "child_workflow_result",
        "008_child_result.backend.json",
        51,
        feature_id="feature-184",
        parent_workflow_id="feature-184",
        child_workflow_id="feature-184:backend",
        repository_id="backend",
        workstream_id="ws-backend",
        branch_name="feature/186-backend",
        workspace_path="/workspaces/backend",
        code_completion_artifact_id="006_code_completion.backend.json",
        review_artifact_id="007_review.backend.json",
        changed_files=[],
        validation_results=[],
        status="approved",
        blocking_issues=[],
        pull_request_readiness=True,
        contract_sections_consumed=[],
        contract_sections_implemented=[],
        model_routing={
            "model": "gpt-5.3-codex",
            "routing_reason": "The previous call reached its output limit at xhigh effort, "
            "so the same attempt is re-asked once at high.",
        },
        metadata={
            "truncated_degraded_retry": True,
            "fault_count": 1,
            "fault_seconds_excluded": 844.0,
            "fault_classes": ["ReadError"],
            "runtime_wall_seconds": 3060.0,
            "runtime_charged_seconds": 2216.0,
        },
    )


def test_the_186_attempt_zero_sequence_reads_in_order() -> None:
    """A2: truncation, degraded retry, provider fault, completion -- four bubbles, in order."""
    artifact = _feature_186_attempt_zero()
    entries = feature_logbook(_state(artifacts=[artifact]))
    keys = _templates(entries, record_id=artifact.artifact_id)

    assert keys[:4] == [
        "artifact.child_workflow_result.truncated",
        "artifact.child_workflow_result.degraded_retry",
        "artifact.child_workflow_result.provider_fault",
        "artifact.child_workflow_result.approved",
    ]
    degraded = next(
        item for item in entries if item.template == "artifact.child_workflow_result.degraded_retry"
    )
    # The recorded reason sentence, quoted -- not a sentence about the recorded reason.
    assert degraded.quote == (
        "The previous call reached its output limit at xhigh effort, so the same attempt is "
        "re-asked once at high."
    )
    assert degraded.quote_source == "model_routing.routing_reason"
    fault = next(
        item for item in entries if item.template == "artifact.child_workflow_result.provider_fault"
    )
    assert "844s of waiting is not counted" in fault.text
    assert fault.detail == "Recorded as ReadError."
    runtime = next(
        item for item in entries if item.template == "artifact.child_workflow_result.runtime"
    )
    assert "3060s" in runtime.text
    assert "2216s" in runtime.text


def test_a_fault_free_attempt_says_nothing_about_faults() -> None:
    """A2's other half: explicit zeros are not a story, and must not be told as one."""
    artifact = _feature_186_attempt_zero().model_copy(
        update={
            "metadata": {
                "truncated_degraded_retry": False,
                "fault_count": 0,
                "fault_seconds_excluded": 0.0,
                "fault_classes": [],
            }
        }
    )
    keys = _templates(feature_logbook(_state(artifacts=[artifact])))
    assert "artifact.child_workflow_result.truncated" not in keys
    assert "artifact.child_workflow_result.provider_fault" not in keys
    assert "artifact.child_workflow_result.approved" in keys


# --------------------------------------------------------------------------------------------
# A3 -- deferred honesty
# --------------------------------------------------------------------------------------------


def _completion_with_repairs() -> Any:
    """A settled attempt that needed two repair passes on the scoped-fix model."""
    return _artifact(
        "code_completion",
        "006_code_completion.backend.json",
        18,
        completion_status="completed",
        summary="Adds POST /apps/bulk with per-row validation.",
        file_changes=[
            {
                "path": "src/controllers/apps.js",
                "change_type": "modified",
                "description": "Adds the bulk handler.",
            }
        ],
        validation_results=[],
        test_coverage_percent=None,
        remaining_work=[],
        commit_sha=None,
        metadata={
            "source_repair_passes": 2,
            "source_repair_execution": {"model": "gpt-5.3-codex", "model_role": "scoped_fix"},
            "source_repair_diagnostics": [
                "src/controllers/apps.js:41  'next' is defined but never used"
            ],
        },
    )


def test_a_repair_pass_is_not_a_bubble_while_the_attempt_runs() -> None:
    """A3: mid-attempt there is no completion artifact, so there is nothing to say yet."""
    running = _state(
        artifacts=[],
        children={"backend": _child("backend", status=ChildWorkflowStatus.RUNNING)},
        status=FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
    )
    entries = feature_logbook(
        running,
        operations=[
            _operation(
                ExternalOperationType.RUN_CODING_EXECUTOR,
                "op-coding",
                10,
                status=ExternalOperationStatus.RUNNING,
                completed_at=None,
            )
        ],
    )
    keys = _templates(entries)
    assert keys == ["operation.running"]
    assert "artifact.code_completion.repairs" not in keys
    assert "repair" not in _texts(entries)


def test_a_repair_pass_appears_the_moment_the_attempt_settles() -> None:
    """A3: from completion metadata -- the count and the model, deferred and never invented."""
    entries = feature_logbook(_state(artifacts=[_completion_with_repairs()]))
    repairs = next(item for item in entries if item.template == "artifact.code_completion.repairs")
    assert repairs.text == (
        "Its own checks rejected the first draft, so it corrected the code in 2 further "
        "passes inside the same attempt, using gpt-5.3-codex."
    )
    assert repairs.quote is not None
    assert repairs.quote.startswith("src/controllers/apps.js:41")
    assert repairs.record.kind is LogbookRecordKind.ARTIFACT
    assert repairs.record.id == "006_code_completion.backend.json"


def test_a_declined_repair_says_which_kind_of_decline_it_was() -> None:
    """A3: "nothing was repairable" and "no scoped-fix model is configured" are not the same."""
    failed = _artifact(
        "code_completion",
        "006_code_completion.backend.json",
        18,
        completion_status="failed",
        summary="The lint run rejected what was written.",
        file_changes=[],
        validation_results=[],
        test_coverage_percent=None,
        remaining_work=["Correct the reported lint failures."],
        commit_sha=None,
        metadata={
            "source_repair_engaged": False,
            "source_repair_passes": 0,
            "source_repair_scoped_fix_role_resolved": True,
        },
    )
    keys = _templates(feature_logbook(_state(artifacts=[failed])))
    assert "artifact.code_completion.repairs_declined" in keys

    unavailable = failed.model_copy(
        update={
            "metadata": {
                "source_repair_engaged": False,
                "source_repair_passes": 0,
                "source_repair_scoped_fix_role_resolved": False,
            }
        }
    )
    keys = _templates(feature_logbook(_state(artifacts=[unavailable])))
    assert "artifact.code_completion.repairs_unavailable" in keys
    assert "artifact.code_completion.repairs_declined" not in keys


def test_the_deferred_in_attempt_checks_all_reach_the_thread() -> None:
    """A3: the assertion guard, the two wiring findings and the self-review are all deferred."""
    artifact = _artifact(
        "code_completion",
        "006_code_completion.backend.json",
        18,
        completion_status="failed",
        summary="The change was written but its own checks did not accept it.",
        file_changes=[],
        validation_results=[],
        test_coverage_percent=None,
        remaining_work=["Rewrite the assertion the guard refused to weaken."],
        commit_sha=None,
        metadata={
            "terminal_outcome": "SCOPED_TEST_ASSERTION_WEAKENED",
            "reachability_issues_unrepaired": [
                "src/services/bulkApps.js is not referred to by anything that runs"
            ],
            "reachability_candidates_dropped": ["src/services/appsIndex.js"],
            "assigned_file_issues_unrepaired": [
                "the plan assigned src/apiUtils/allapps.apiUtils.js and the change added a "
                "sibling module instead"
            ],
            "self_review": {
                "ran": True,
                "outcome": "substantive_problem",
                "summary": "The bulk handler writes each row in its own transaction.",
            },
        },
    )
    keys = _templates(feature_logbook(_state(artifacts=[artifact])))
    assert keys == [
        "artifact.code_completion.failed",
        "artifact.code_completion.assertion_guard",
        "artifact.code_completion.unreachable",
        "artifact.code_completion.reachability_bounded",
        "artifact.code_completion.misplaced",
        "artifact.code_completion.self_review",
    ]


def test_the_question_a_stopped_workstream_leaves_is_quoted() -> None:
    """A3: the terminal triage question -- 49-'s ledger and conflict narratives included."""
    conflict = (
        "The repository reviewer approved what the integration reviewer rejects; the two "
        "authorities disagree and retrying cannot settle it. Which position holds?"
    )
    artifact = _artifact(
        "child_workflow_result",
        "008_child_result.backend.json",
        44,
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
        # `_with_triage` appends the question to a list that already had one entry, which is
        # exactly the case quoting `blocking_issues[0]` alone would miss.
        blocking_issues=["The bulk endpoint is still not atomic.", conflict],
        pull_request_readiness=False,
        contract_sections_consumed=[],
        contract_sections_implemented=[],
        metadata={"operator_question": conflict, "terminal_cause": "design_conflict"},
    )
    entries = feature_logbook(_state(artifacts=[artifact]))
    keys = _templates(entries)
    assert keys == [
        "artifact.child_workflow_result.failed",
        "artifact.child_workflow_result.operator_question",
    ]
    assert entries[0].quote == "The bulk endpoint is still not atomic."
    assert entries[1].quote == conflict
    assert entries[1].quote_source == "metadata.operator_question"


def test_a_question_that_is_already_the_first_blocking_issue_is_said_once() -> None:
    """A3: the same words never appear twice; a repeated quote reads as two findings."""
    question = "Should the change stand, or is the review wrong to require it?"
    artifact = _artifact(
        "child_workflow_result",
        "008_child_result.backend.json",
        44,
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
        blocking_issues=[question],
        pull_request_readiness=False,
        contract_sections_consumed=[],
        contract_sections_implemented=[],
        metadata={"operator_question": question},
    )
    keys = _templates(feature_logbook(_state(artifacts=[artifact])))
    assert keys == ["artifact.child_workflow_result.failed"]


# --------------------------------------------------------------------------------------------
# A4 -- 1:1 anchoring
# --------------------------------------------------------------------------------------------


def _assert_every_bubble_is_anchored(
    entries: list[LogbookEntry],
    *,
    state: FeatureWorkflowSnapshot,
    events: list[LogbookEvent],
    operations: list[ExternalOperation],
) -> None:
    """Resolve every bubble's record against the rows it claims to have been read from."""
    resolvable = {
        LogbookRecordKind.EVENT: {str(item.id) for item in events},
        LogbookRecordKind.ARTIFACT: {item.artifact_id for item in state.artifacts},
        LogbookRecordKind.OPERATION: {item.operation_id for item in operations},
        LogbookRecordKind.WORKSTREAM: {
            item.child_workflow_id for item in state.child_workflows.values()
        },
    }
    assert entries, "a feature with records must produce a thread"
    for entry in entries:
        assert entry.record.id, f"{entry.template} rendered without a record"
        assert entry.record.id in resolvable[entry.record.kind], (
            f"{entry.template} points at {entry.record.kind} {entry.record.id}, "
            "which is not in the records this thread was composed from"
        )
        assert entry.template in LOGBOOK_TEMPLATES


def test_every_bubble_carries_a_resolvable_record() -> None:
    """A4: an unanchored bubble is a failure here, never a rendering fallback."""
    state, events = _feature_184()
    operations = [
        _operation(ExternalOperationType.CLONE_REPOSITORY, "op-clone", 2),
        _operation(ExternalOperationType.RUN_TESTS, "op-tests", 16),
        _operation(
            ExternalOperationType.CREATE_PULL_REQUEST,
            "op-pr",
            40,
            status=ExternalOperationStatus.FAILED_TERMINAL,
            error_code="pull_request_unverified",
        ),
    ]
    entries = feature_logbook(state, lifecycle_events=events, operations=operations)
    _assert_every_bubble_is_anchored(entries, state=state, events=events, operations=operations)


def test_a_granted_retry_is_anchored_to_the_row_that_records_it() -> None:
    """A4: the one bubble read from the child row, because a grant is recorded nowhere else."""
    state = _state(
        artifacts=[],
        children={
            "backend": _child(
                "backend",
                status=ChildWorkflowStatus.RUNNING,
                granted_extra_attempts=2,
                retry_grants=[
                    {
                        "granted_at": _at(60).isoformat(),
                        "granted_by": "akhilesh",
                        "attempts": 2,
                        "reason": "The suite it kept failing was fixed in main; give it two "
                        "more attempts.",
                        "stopped_because": "the implementation_retry_count budget of 4 is "
                        "exhausted",
                    }
                ],
            )
        },
        status=FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
    )
    entry = next(
        item for item in feature_logbook(state) if item.template == "workstream.retry_granted"
    )
    assert entry.record.kind is LogbookRecordKind.WORKSTREAM
    assert entry.record.id == "feature-184:backend"
    assert entry.agent is LogbookAgent.PERSON
    assert entry.text == (
        "akhilesh granted admanager-server 2 more attempts after the platform had stopped it."
    )
    assert entry.detail is not None
    assert "implementation_retry_count budget of 4 is exhausted" in entry.detail
    assert entry.quote is not None
    assert entry.quote.startswith("The suite it kept failing")


def test_a_grant_recorded_without_a_zone_still_places_in_the_thread() -> None:
    """A4's crash mode: one naive instant among aware ones would take the whole read down."""
    state = _state(
        artifacts=[_completion_with_repairs()],
        children={
            "backend": _child(
                "backend",
                status=ChildWorkflowStatus.RUNNING,
                retry_grants=[
                    {
                        "granted_at": _at(60).replace(tzinfo=None).isoformat(),
                        "granted_by": "akhilesh",
                        "attempts": 1,
                        "reason": "One more attempt.",
                    },
                    # And one whose timestamp is not a timestamp at all: dropped rather than
                    # placed at an invented instant, because the thread is chronological and
                    # a guess would put a human decision in the wrong place in the story.
                    {"granted_at": "not a date", "granted_by": "akhilesh", "attempts": 1},
                ],
            )
        },
        status=FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
    )
    entries = feature_logbook(state)
    grants = [item for item in entries if item.template == "workstream.retry_granted"]
    assert len(grants) == 1
    assert grants[0].timestamp == _at(60)
    # Read as UTC, so it sorts after the completion artifact rather than raising.
    assert [item.sequence for item in entries] == list(range(len(entries)))
    assert entries[-1].template == "workstream.retry_granted"


# --------------------------------------------------------------------------------------------
# A5 -- redaction
# --------------------------------------------------------------------------------------------


def test_a_quoted_blocking_issue_reads_exactly_as_its_source_does() -> None:
    """A5: already-screened prose is quoted, not re-screened and not re-worded."""
    screened = (
        "Validation failed: config/credentials.example.json declares GITHUB_TOKEN=[redacted] "
        "and the loader rejects the placeholder."
    )
    artifact = _artifact(
        "child_workflow_result",
        "008_child_result.backend.json",
        30,
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
        blocking_issues=[screened],
        pull_request_readiness=False,
        contract_sections_consumed=[],
        contract_sections_implemented=[],
    )
    entry = next(
        item
        for item in feature_logbook(_state(artifacts=[artifact]))
        if item.template == "artifact.child_workflow_result.failed"
    )
    assert entry.quote == screened
    assert entry.quote_source == "blocking_issues[0]"


def test_the_logbook_publishes_nothing_the_journal_endpoint_withholds() -> None:
    """A5: no `error_message`, no `safe_metadata`, no `result_payload` -- not one channel wider."""
    operation = _operation(
        ExternalOperationType.RUN_TESTS,
        "op-tests",
        16,
        status=ExternalOperationStatus.FAILED_TERMINAL,
        error_code="validation_source_failure",
        error_message="npm ERR! could not resolve https://user:hunter2@registry.internal",
        safe_metadata={"workspace_path": "/workspaces/backend", "exit_code": 1},
        result_payload={"stdout": "16 failing"},
    )
    state, events = _feature_184()
    said = _texts(feature_logbook(state, lifecycle_events=events, operations=[operation]))
    assert "hunter2" not in said
    assert "registry.internal" not in said
    assert "/workspaces/backend" not in said
    assert "16 failing" not in said
    # The error *code* is a platform constant and the journal endpoint already serves it.
    assert "validation_source_failure" in said


# --------------------------------------------------------------------------------------------
# The first non-negotiable: no narrator, ever
# --------------------------------------------------------------------------------------------


def test_nothing_on_this_path_can_reach_a_model() -> None:
    """The rule the whole item rests on, checked structurally rather than promised.

    A logbook that can say "I fixed it" when nothing was fixed is worse than no logbook, and
    a comment saying "no model calls here" is not what makes that true -- the module having
    nothing it could call is. So this reads the composer's own imports: an adapter, an agent,
    a prompt loader or a model router appearing among them is what a narrator would arrive
    as, and it fails here rather than in a run.
    """
    import ast
    import pathlib

    import services.logbook as module

    source = pathlib.Path(module.__file__ or "").read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = ("adapters", "agents", "tools.model_routing", "configs", "prompts", "httpx")
    offenders = sorted(
        name for name in imported if any(name.startswith(prefix) for prefix in forbidden)
    )
    assert not offenders, (
        f"the logbook composer imports {offenders}; every bubble must be a template filled "
        "from a durable record, and nothing on this path may be able to ask a model anything"
    )
    # And the registry itself reaches for nothing at all: no session, no request, no app.
    assert not any(name.startswith(("fastapi", "api.", "storage.")) for name in imported)


# --------------------------------------------------------------------------------------------
# A6 -- registry reuse
# --------------------------------------------------------------------------------------------


def test_the_registry_renders_a_bubble_without_a_feature() -> None:
    """A6: importable and callable on its own, which is what item 11 needs of it."""
    bubble = render_bubble(
        "artifact.review.changes_requested",
        {"repository": "admanager-server", "findings": "2 findings"},
        quote="The endpoint writes each row in its own transaction.",
    )
    assert bubble.agent is LogbookAgent.REVIEWER
    assert bubble.tone == "attention"
    assert bubble.text == "The reviewer asked for changes to admanager-server, raising 2 findings."
    assert bubble.quote == "The endpoint writes each row in its own transaction."
    assert bubble.quote_source == "summary"


def test_the_registry_refuses_a_key_it_does_not_define() -> None:
    """A6: a sentence composed outside the table is not available, even by accident."""
    with pytest.raises(UnknownLogbookTemplate):
        render_bubble("artifact.invented_by_a_caller")


def test_every_registry_row_is_reachable_from_a_record() -> None:
    """A6: the table and its callers are checked against each other, not merely documented."""
    used: set[str] = set()

    state, events = _feature_184()
    operations = [
        _operation(kind, f"op-{index}", index, status=status)
        for index, kind in enumerate(OPERATION_VOICES)
        for status in [ExternalOperationStatus.SUCCEEDED]
    ]
    used.update(_templates(feature_logbook(state, lifecycle_events=events, operations=operations)))
    used.update(_templates(feature_logbook(_state(artifacts=[_feature_186_attempt_zero()]))))
    used.update(_templates(feature_logbook(_state(artifacts=[_completion_with_repairs()]))))

    # Whatever the fixtures above do not reach, the registry must still be able to render:
    # a row nothing can fill is a sentence that would break the first time it was needed.
    for key, template in LOGBOOK_TEMPLATES.items():
        if key in used:
            continue
        fields = {name: "x" for name in _placeholders(template.sentence)}
        fields.update({name: "x" for name in _placeholders(template.detail or "")})
        assert render_bubble(key, fields).text


def _placeholders(text: str) -> set[str]:
    """The field names one template asks for, read from the template itself."""
    import string

    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_a_quote_is_clipped_at_a_whole_word() -> None:
    """A6: bounded quoting, never mid-token -- a clipped path is a different path."""
    sentence = "The endpoint at src/controllers/appsBulkCreate.js:118 writes each row of the "
    long = sentence + "batch inside its own transaction, so a partial failure is invisible."
    clipped = clip_quote(long, bound=60)
    # Every character before the clip mark is the source's own, in order: nothing was
    # reworded, and no word was cut in half on the way out.
    assert clipped.endswith("…")
    assert clipped[:-1] in long
    assert clipped[:-1].split()[-1] in long.split()
    assert "src/controllers/appsBulkCreate.js:118" in clipped
    assert clip_quote("Short enough.", bound=60) == "Short enough."
    # A sentence boundary in the second half of the bound is preferred to a word boundary,
    # so a quote reads as a finished thought. One in the first half is not: clipping a
    # 240-character quote down to four words because it opened with "Yes." tells nobody
    # anything, and the fuller fragment is the more useful one.
    assert clip_quote("One thing happened here and then. Another thing did too.", bound=45) == (
        "One thing happened here and then."
    )
    assert clip_quote("Yes. And then a great deal else happened afterwards.", bound=30) == (
        "Yes. And then a great deal…"
    )


def test_an_unmapped_operation_type_still_renders() -> None:
    """A6: every journaled type has a voice, and a type added later is never dropped."""
    assert set(OPERATION_VOICES) == set(ExternalOperationType), (
        "a journaled operation type with no voice renders through the generic template; "
        "give it one rather than leaving the thread to say 'a ... operation'"
    )
    entries = feature_logbook(
        _state(artifacts=[]),
        operations=[_operation(ExternalOperationType.RUN_TESTS, "op-tests", 5)],
    )
    assert _templates(entries) == ["operation.succeeded"]
    assert entries[0].agent is LogbookAgent.ENGINEER
    assert entries[0].text == "Finished running admanager-server's own tests."


def test_the_contract_projection_write_narrates_as_workspace_preparation() -> None:
    """24: one operation type, two callers -- the step decides which sentence the row gets.

    The projection write happens at the start of every attempt, before the Engineer is called.
    Narrating it as "writing the change into the files" is the same false claim item 21 removed
    from the drawer, on the logbook's surface: the thread said the change had been written into
    the files a minute before the model was even asked for it.
    """
    projection = _operation(
        ExternalOperationType.WRITE_FILE_CHANGES,
        "op-projection",
        1,
        safe_metadata={
            "logical_step": CONTRACT_PROJECTION_LOGICAL_STEP,
            "path": "openapi.yaml",
            "contract_version": "1.0.0",
        },
    )
    implementation = _operation(
        ExternalOperationType.WRITE_FILE_CHANGES,
        "op-implementation",
        2,
        safe_metadata={"logical_step": "apply_coding_updates"},
    )
    entries = feature_logbook(_state(artifacts=[]), operations=[projection, implementation])

    assert [item.text for item in entries] == [
        "Finished preparing admanager-server's workspace with the approved contract.",
        "Finished writing the change into the files.",
    ]
    # The step decided the sentence, so the projection's own metadata never reaches the thread:
    # neither the path it wrote nor the contract version it wrote appears anywhere in it.
    rendered = " ".join(f"{item.text} {item.detail or ''} {item.quote or ''}" for item in entries)
    assert "openapi.yaml" not in rendered
    assert "1.0.0" not in rendered


def test_a_step_the_registry_does_not_name_keeps_its_types_voice() -> None:
    """24: the fallback is the type's voice -- a step table miss is never a missing sentence.

    Both cases that must not go quiet: a `write_file_changes` row from a step nobody has
    written a voice for, and a row from before the executor stamped a step at all.
    """
    for metadata in ({"logical_step": "a_step_added_next_year"}, {}):
        entries = feature_logbook(
            _state(artifacts=[]),
            operations=[
                _operation(
                    ExternalOperationType.WRITE_FILE_CHANGES,
                    "op-write",
                    3,
                    safe_metadata=metadata,
                )
            ],
        )
        assert [item.text for item in entries] == ["Finished writing the change into the files."]
        assert entries[0].agent is LogbookAgent.ENGINEER


def test_every_step_voice_names_a_type_that_has_one() -> None:
    """24: a step voice is a refinement of a type's voice, never a replacement for a missing one.

    A step keyed on a type with no voice of its own would render some rows of that type and
    leave the rest falling through to `operation.unknown_type` -- one type speaking with two
    unrelated registrations.
    """
    for operation_type, step in OPERATION_STEP_VOICES:
        assert operation_type in OPERATION_VOICES, f"{operation_type} has no type-level voice"
        assert step, f"{operation_type} has a step voice keyed on an empty step"


# --------------------------------------------------------------------------------------------
# The endpoint
# --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_endpoint_tells_a_real_run_and_pages_it() -> None:
    """A run that actually happened renders from what it persisted, with no logbook stored."""
    app = create_app(platform_api_key="logbook-test-key")
    headers = {"Authorization": "Bearer logbook-test-key", "Idempotency-Key": "logbook-001"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        await settle(app)
        logbook = await client.get("/features/feature-login/logbook", headers=headers)
        first = await client.get("/features/feature-login/logbook?limit=3", headers=headers)
        artifacts = await client.get("/features/feature-login/artifacts", headers=headers)
        events = await client.get("/features/feature-login/events", headers=headers)

    assert logbook.status_code == 200, logbook.text
    body = logbook.json()
    entries = body["entries"]
    assert entries, "a completed run must have a story"
    assert [item["sequence"] for item in entries] == list(range(len(entries)))
    assert body["next_cursor"] is None
    assert "Orchestrator" in body["agents"]
    # Every bubble names a record the API itself will serve back.
    artifact_ids = {item["artifact_id"] for item in artifacts.json()["artifacts"]}
    event_ids = {str(item["id"]) for item in events.json()["events"]}
    for entry in entries:
        record = entry["record"]
        assert record["id"]
        if record["kind"] == "artifact":
            assert record["id"] in artifact_ids
        elif record["kind"] == "event":
            assert record["id"] in event_ids
    # The story a product manager reads: it was asked for, planned, built, reviewed, published.
    templates = [item["template"] for item in entries]
    assert "artifact.prd" in templates
    assert "artifact.integration_contract.approved" in templates
    assert "artifact.pull_request" in templates
    assert any(item.startswith("artifact.feature_completion") for item in templates)
    # Paging is by entry and resumes exactly where it stopped.
    page = first.json()
    assert len(page["entries"]) == 3
    assert page["next_cursor"] == page["entries"][-1]["sequence"]
    assert page["entries"] == entries[:3]


@pytest.mark.asyncio
async def test_the_endpoint_refuses_an_unknown_feature() -> None:
    """A read model still answers 404 for something that was never started."""
    app = create_app(platform_api_key="logbook-test-key")
    headers = {"Authorization": "Bearer logbook-test-key"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        missing = await client.get("/features/feature-nope/logbook", headers=headers)
    assert missing.status_code == 404
