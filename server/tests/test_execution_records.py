"""What the published execution records claim, and what they must never claim.

These are about truthfulness rather than shape. The graph draws arrows from this data, and an
arrow that names a model no model ran, or names today's configured model on last week's
attempt, is worse than an arrow with nothing on it: it looks the same as a correct one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import HttpUrl

from artifacts.schemas import (
    ChildWorkflowResultArtifact,
    CodeCompletionArtifact,
    IntegrationContractArtifact,
    IntegrationReviewArtifact,
    RepositoryExecutionPlanArtifact,
    RepositoryRepairProposalArtifact,
    ReviewArtifact,
    TechnicalPRDArtifact,
)
from services.execution_records import (
    ExecutionHandlerType,
    ExecutionRecord,
    ExecutionStage,
    ExecutionStatus,
    HumanAction,
    feature_executions,
)
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from state.feature_models import (
    ChildWorkflowReference,
    FeatureWorkflowSnapshot,
    RepositorySpec,
    ensure_feature_failure_summary,
)
from workflow_schema import WORKFLOW_SCHEMA_VERSION

_AT = datetime(2026, 8, 28, 10, 42, 16, tzinfo=UTC)


def _envelope(artifact_id: str, producer: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "workflow_id": "wf-1",
        "artifact_id": artifact_id,
        "producer": producer,
        "timestamp": _AT,
        "metadata": metadata,
        "validation_status": "valid",
    }


def _technical_prd(
    *, metadata: dict[str, Any], questions: list[dict[str, Any]] | None = None
) -> TechnicalPRDArtifact:
    return TechnicalPRDArtifact.model_validate(
        {
            **_envelope("002_technical_prd.json", "product_manager", metadata),
            "title": "A plan",
            "solution_summary": "Do the thing.",
            "functional_requirements": [
                {
                    "requirement_id": "REQ-1",
                    "description": "The thing happens.",
                    "priority": "must",
                    "acceptance_criteria": ["It happened."],
                    "dependencies": [],
                }
            ],
            "non_functional_requirements": [],
            "data_requirements": [],
            "integration_requirements": [],
            "security_requirements": [],
            "assumptions": [],
            "unresolved_questions": questions or [],
        }
    )


def _contract() -> IntegrationContractArtifact:
    return IntegrationContractArtifact.model_validate(
        {
            **_envelope(
                "009_integration_contract.json",
                "feature_planner",
                {"model": "gpt-5.6-sol", "contract_version": "1.0.0"},
            ),
            "feature_id": "f-1",
            "contract_version": "1.0.0",
            "api_style": "rest",
            "endpoints": [],
            "shared_schemas": [],
            "authentication_contract": None,
            "authorization_rules": [],
            "error_contracts": [],
            "event_contracts": [],
            "environment_variables": [],
            "status": "approved",
            "compatibility_policy": {
                "policy": "Additive changes only.",
                "breaking_change_allowed": False,
                "migration_requirements": [],
                "rollback_requirements": [],
            },
            "owning_workstreams": ["ws-1"],
            "approved_at": _AT,
        }
    )


def _plan(metadata: dict[str, Any]) -> RepositoryExecutionPlanArtifact:
    return RepositoryExecutionPlanArtifact.model_validate(
        {
            **_envelope("010_repository_execution_plan.json", "feature_planner", metadata),
            "feature_id": "f-1",
            "contract_artifact_id": "009_integration_contract.json",
            "workstreams": [
                {
                    "workstream_id": "ws-1",
                    "repository_id": "api",
                    "role": "backend",
                    "requirement_ids": ["REQ-1"],
                    "responsibilities": ["Do the thing."],
                    "task_ids": ["T-1"],
                    "dependency_workstream_ids": [],
                    "contract_sections_consumed": [],
                    "contract_sections_implemented": [],
                    "acceptance_criteria": ["It happened."],
                    "test_requirements": ["It is tested."],
                    "documentation_requirements": [],
                    "expected_files_or_areas": ["src"],
                    "required": True,
                }
            ],
            "execution_order": ["ws-1"],
            "parallel_groups": [["ws-1"]],
            "integration_test_plan": ["End to end."],
            "merge_strategy": "independent",
            "deployment_strategy": "independent",
            "feature_flag_strategy": [],
            "rollback_strategy": ["Revert the pull request."],
        }
    )


def _code_completion(attempt: int, metadata: dict[str, Any]) -> CodeCompletionArtifact:
    return CodeCompletionArtifact.model_validate(
        {
            **_envelope(
                f"006_code_completion.api.attempt-{attempt}.json",
                "engineer",
                {"child_attempt": attempt, **metadata},
            ),
            "completion_status": "completed",
            "summary": f"Attempt {attempt} implementation.",
            "file_changes": [],
            "validation_results": [],
            "test_coverage_percent": None,
            "remaining_work": [],
            "commit_sha": None,
        }
    )


def _review(
    attempt: int,
    *,
    verdict: str,
    metadata: dict[str, Any],
    findings: list[Any] | None = None,
) -> ReviewArtifact:
    return ReviewArtifact.model_validate(
        {
            **_envelope(
                f"007_review.api.attempt-{attempt}.json",
                "reviewer",
                {"child_attempt": attempt, **metadata},
            ),
            "verdict": verdict,
            "summary": "A review.",
            "requirement_checks": [],
            "findings": findings or [],
            "architecture_assessment": "Fine.",
            "security_assessment": "Fine.",
            "test_coverage_assessment": "Fine.",
        }
    )


def _finding(**overrides: Any) -> dict[str, Any]:
    return {
        "finding_id": "F-1",
        "severity": "high",
        "title": "Authentication middleware is not applied to the new endpoint.",
        "description": "The route is registered without requireAdminAuth.",
        "recommendation": "Apply the middleware.",
        "file_path": None,
        "line_number": None,
        "repository_id": "api",
        **overrides,
    }


def _result(
    attempt: int,
    *,
    status: str = "failed",
    metadata: dict[str, Any] | None = None,
    **overrides: Any,
) -> ChildWorkflowResultArtifact:
    return ChildWorkflowResultArtifact.model_validate(
        {
            **_envelope(
                f"011_child_workflow_result.api.attempt-{attempt}.json",
                "child_workflow",
                {"child_retry_count": attempt, **(metadata or {})},
            ),
            "feature_id": "f-1",
            "parent_workflow_id": "wf-1",
            "child_workflow_id": "f-1:api",
            "repository_id": "api",
            "workstream_id": "ws-1",
            "branch_name": "ai/f-1/api",
            "workspace_path": "/w/api",
            "code_completion_artifact_id": f"006_code_completion.api.attempt-{attempt}.json",
            "review_artifact_id": f"007_review.api.attempt-{attempt}.json",
            "changed_files": [],
            "validation_results": [],
            "status": status,
            "blocking_issues": ["Authentication middleware is not applied."],
            "pull_request_readiness": False,
            "contract_sections_consumed": [],
            "contract_sections_implemented": [],
            **overrides,
        }
    )


def _integration_review(metadata: dict[str, Any]) -> IntegrationReviewArtifact:
    return IntegrationReviewArtifact.model_validate(
        {
            **_envelope("012_integration_review.attempt-0.json", "integration_reviewer", metadata),
            "feature_id": "f-1",
            "contract_artifact_id": "009_integration_contract.json",
            "review_status": "approved",
            "repository_results": [],
            "contract_checks": ["The contract projection matches."],
            "cross_repository_findings": [],
            "compatibility_assessment": "Compatible.",
            "security_assessment": "Fine.",
            "deployment_assessment": "Independent.",
            "merge_order": ["api"],
            "required_fixes": [],
        }
    )


def _child(**overrides: Any) -> ChildWorkflowReference:
    return ChildWorkflowReference.model_validate(
        {
            "child_workflow_id": "f-1:api",
            "repository_id": "api",
            "workstream_id": "ws-1",
            "status": ChildWorkflowStatus.RUNNING,
            "branch_name": "ai/f-1/api",
            "workspace_path": "/w/api",
            "retry_count": 0,
            **overrides,
        }
    )


def _feature(
    *,
    artifacts: list[Any] | None = None,
    children: dict[str, ChildWorkflowReference] | None = None,
    status: FeatureWorkflowStatus = FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS,
    execution_mode: str = "live",
    **overrides: Any,
) -> FeatureWorkflowSnapshot:
    return FeatureWorkflowSnapshot.model_validate(
        {
            "feature_id": "f-1",
            "workflow_id": "wf-1",
            "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
            "created_by_build_revision": "rev",
            "last_executor_build_revision": "rev",
            "status": status,
            "title": "A feature",
            "repository_specs": [
                RepositorySpec(
                    repository_id="api",
                    name="api",
                    role="backend",
                    repository_url=HttpUrl("https://github.com/acme/api"),
                    default_branch="main",
                )
            ],
            "artifacts": artifacts or [],
            "child_workflows": children or {},
            "max_integration_review_cycles": 5,
            "max_child_review_cycles": 8,
            "max_implementation_retries": 4,
            "max_validation_retries": 2,
            "max_repository_setup_retries": 1,
            "max_contract_revision_cycles": 3,
            "execution_mode": execution_mode,
            "created_at": _AT,
            "updated_at": _AT,
            **overrides,
        }
    )


def _by_id(records: list[ExecutionRecord]) -> dict[str, ExecutionRecord]:
    return {item.execution_id: item for item in records}


# --------------------------------------------------------------------------- resolved models


def test_a_completed_transition_names_the_model_its_own_artifact_recorded() -> None:
    """The resolved model, effort and provider come from the execution, not configuration."""
    state = _feature(
        artifacts=[
            _technical_prd(
                metadata={
                    "execution": {
                        "agent_type": "Product manager",
                        "provider": "openai",
                        "model": "gpt-5.6-sol",
                        "reasoning_effort": "xhigh",
                    }
                }
            )
        ]
    )
    record = _by_id(feature_executions(state))["technical_prd:002_technical_prd.json"]
    assert record.handler_type is ExecutionHandlerType.MODEL
    assert record.model == "gpt-5.6-sol"
    assert record.reasoning_effort == "xhigh"
    assert record.provider == "openai"
    assert record.model_resolved is True
    assert record.status is ExecutionStatus.COMPLETED
    assert record.agent_type == "Product manager"


def test_an_artifact_written_before_the_canonical_block_still_names_its_model() -> None:
    """Artifacts are immutable, so the older per-agent keys stay readable.

    Without this a feature that ran before the execution block existed would show "model not
    recorded" on every arrow, which is a regression dressed as honesty: the model *is*
    recorded, under the key that agent used at the time.
    """
    state = _feature(artifacts=[_technical_prd(metadata={"model": "gpt-5.3-codex"})])
    record = _by_id(feature_executions(state))["technical_prd:002_technical_prd.json"]
    assert record.model == "gpt-5.3-codex"
    assert record.reasoning_effort is None
    assert record.model_resolved is True


def test_the_planner_is_named_on_both_transitions_it_performs() -> None:
    """The contract and the plan come out of the same agent and each names its own execution."""
    state = _feature(
        artifacts=[
            _technical_prd(metadata={"model": "gpt-5.6-sol"}),
            _contract(),
            _plan(
                {
                    "execution": {
                        "agent_type": "Technical planner",
                        "model": "gpt-5.6-sol",
                        "reasoning_effort": "max",
                    }
                }
            ),
        ]
    )
    records = _by_id(feature_executions(state))
    contract = records["integration_contract:009_integration_contract.json"]
    plan = records["execution_plan:010_repository_execution_plan.json"]
    assert contract.agent_type == "Technical planner"
    assert contract.model == "gpt-5.6-sol"
    assert contract.contract_version == "1.0.0"
    assert plan.reasoning_effort == "max"
    assert plan.from_stage is ExecutionStage.INTEGRATION_CONTRACT


def test_a_pending_transition_does_not_invent_a_model() -> None:
    """Nothing has run, so nothing is named. A client must say so rather than predict."""
    state = _feature(artifacts=[_technical_prd(metadata={"model": "gpt-5.6-sol"})])
    record = _by_id(feature_executions(state))["execution_plan:pending"]
    assert record.model is None
    assert record.model_resolved is False
    assert record.status is ExecutionStatus.PENDING


def test_a_historical_attempt_keeps_its_own_model_when_a_later_one_used_another() -> None:
    """Two attempts, two models, and neither is allowed to overwrite the other."""
    artifacts = [
        _technical_prd(metadata={"model": "gpt-5.6-sol"}),
        _code_completion(0, {"execution": {"agent_type": "Engineer", "model": "gpt-5.6-sol"}}),
        _review(
            0,
            verdict="changes_requested",
            metadata={"model": "gpt-5.6-sol"},
            findings=[_finding()],
        ),
        _result(0, failure_classification="review_scope_failure"),
        _code_completion(
            1,
            {
                "execution": {
                    "agent_type": "Engineer",
                    "model": "gpt-5.3-codex",
                    "reasoning_effort": "high",
                }
            },
        ),
        _review(1, verdict="approved", metadata={"model": "gpt-5.6-sol"}),
        _result(1, status="approved", pull_request_readiness=True),
    ]
    state = _feature(
        artifacts=artifacts,
        children={"api": _child(retry_count=1, status=ChildWorkflowStatus.COMPLETED)},
    )
    records = _by_id(feature_executions(state))
    assert records["implementation:api:0"].model == "gpt-5.6-sol"
    assert records["implementation:api:1"].model == "gpt-5.3-codex"
    assert records["implementation:api:1"].reasoning_effort == "high"
    # The retry arrow itself carries the model that answered it, not the one that failed.
    assert records["retry:api:1"].model == "gpt-5.3-codex"


# --------------------------------------------------------------------------- deterministic


def test_validation_is_a_validator_and_never_a_model() -> None:
    """A subprocess ran the commands, and the record says the command and the exit code."""
    artifacts = [
        _code_completion(0, {"coding_model": "gpt-5.6-sol"}),
        _review(0, verdict="changes_requested", metadata={"model": "gpt-5.6-sol"}),
        _result(
            0,
            current_validation_results=[
                {
                    "command": ["npm", "run", "lint"],
                    "passed": False,
                    "exit_code": 1,
                    "duration_seconds": 4.5,
                    "validation_type": "lint",
                },
                {
                    "command": ["npm", "run", "test"],
                    "passed": True,
                    "exit_code": 0,
                    "validation_type": "test",
                },
            ],
        ),
    ]
    state = _feature(artifacts=artifacts, children={"api": _child()})
    record = _by_id(feature_executions(state))["validation:api:0"]
    assert record.handler_type is ExecutionHandlerType.DETERMINISTIC
    assert record.handler == "Validator"
    assert record.model is None
    assert record.command == ["npm", "run", "lint"]
    assert record.exit_code == 1
    assert (record.validation_passed, record.validation_total) == (1, 2)
    assert record.duration_seconds == pytest.approx(4.5)
    assert record.status is ExecutionStatus.FAILED
    # The failure class on a deterministic validation record is the gate that failed, as the
    # failing result itself recorded it -- the one word an arrow can carry.
    assert record.failure_classification == "lint"


def test_a_passing_validation_states_no_failure_class() -> None:
    """A class of failure is a claim about a failure; a green run makes none."""
    artifacts = [
        _code_completion(0, {"coding_model": "gpt-5.6-sol"}),
        _review(0, verdict="approved", metadata={"model": "gpt-5.6-sol"}),
        _result(
            0,
            status="approved",
            current_validation_results=[
                {
                    "command": ["npm", "run", "test"],
                    "passed": True,
                    "exit_code": 0,
                    "validation_type": "test",
                }
            ],
        ),
    ]
    state = _feature(artifacts=artifacts, children={"api": _child()})
    record = _by_id(feature_executions(state))["validation:api:0"]
    assert record.status is ExecutionStatus.COMPLETED
    assert record.failure_classification is None


def test_the_fan_out_and_the_pull_requests_name_their_real_handlers() -> None:
    """Queue dispatch is this orchestrator; a pull request is a provider API call."""
    state = _feature(children={"api": _child()})
    records = _by_id(feature_executions(state))
    assert records["dispatch:api"].handler == "Orchestrator"
    assert records["dispatch:api"].handler_type is ExecutionHandlerType.DETERMINISTIC
    assert records["pull_requests"].handler == "GitHub"
    assert records["pull_requests"].model is None


def test_a_mock_run_that_invoked_no_provider_is_not_called_a_model_execution() -> None:
    """A simulated feature reaches no model, so no arrow on it may say one did."""
    state = _feature(artifacts=[_technical_prd(metadata={"mode": "mock"})], execution_mode="mock")
    record = _by_id(feature_executions(state))["technical_prd:002_technical_prd.json"]
    assert record.handler_type is ExecutionHandlerType.DETERMINISTIC
    assert record.handler == "Simulation"
    # The agent role is still recorded; what is denied is that a model performed it.
    assert record.agent_type == "Product manager"
    assert record.model is None
    assert record.execution_mode == "mock"


# --------------------------------------------------------------------------- retries


def test_a_retry_carries_the_attempt_the_reason_and_the_remediation() -> None:
    """Why the previous attempt failed and what the next must change are separate answers."""
    artifacts = [
        _code_completion(0, {"coding_model": "gpt-5.6-sol"}),
        _review(
            0,
            verdict="changes_requested",
            metadata={"model": "gpt-5.6-sol"},
            findings=[_finding()],
        ),
        _result(
            0,
            failure_classification="validation_configuration_failure",
            current_revision="abc123",
            retry_strategy={
                "failure_classification": "validation_configuration_failure",
                "required_strategy_change": (
                    "Repository configuration must be repaired before feature implementation "
                    "can continue."
                ),
            },
            metadata={"repository_setup_retry_count": 1},
        ),
        _code_completion(1, {"execution": {"agent_type": "Engineer", "model": "gpt-5.3-codex"}}),
        _review(1, verdict="approved", metadata={"model": "gpt-5.6-sol"}),
        _result(1, status="approved", pull_request_readiness=True, current_revision="def456"),
    ]
    state = _feature(
        artifacts=artifacts,
        children={"api": _child(retry_count=1, status=ChildWorkflowStatus.COMPLETED)},
    )
    retry = _by_id(feature_executions(state))["retry:api:1"]
    assert retry.is_retry is True
    assert (retry.attempt, retry.max_attempts) == (2, 8)
    assert "maximum of 8" in (retry.attempt_meaning or "")
    assert retry.failure_summary == _finding()["title"]
    assert retry.failure_severity == "high"
    assert retry.remediation_summary is not None
    assert "repaired" in retry.remediation_summary
    assert retry.failure_classification == "validation_configuration_failure"
    assert retry.revision_before == "abc123"
    assert retry.revision_after == "def456"
    assert retry.review_artifact_id == "007_review.api.attempt-0.json"
    assert retry.previous_execution_id == "review:api:0"
    # The classification budget it was granted against, beside the attempt ratio and never
    # confused with it.
    assert retry.counter_label == "Repository setup retries"
    assert (retry.counter_value, retry.counter_limit) == (1, 1)


def test_every_retry_attempt_keeps_its_own_execution() -> None:
    """Three attempts produce three implementations and two retry transitions."""
    artifacts: list[Any] = []
    for attempt in range(3):
        artifacts.append(_code_completion(attempt, {"coding_model": f"model-{attempt}"}))
        artifacts.append(
            _review(attempt, verdict="changes_requested", metadata={"model": "reviewer-model"})
        )
        artifacts.append(_result(attempt, failure_classification="implementation_missing"))
    state = _feature(
        artifacts=artifacts,
        children={"api": _child(retry_count=2, status=ChildWorkflowStatus.FAILED)},
    )
    records = _by_id(feature_executions(state))
    assert [records[f"implementation:api:{index}"].model for index in range(3)] == [
        "model-0",
        "model-1",
        "model-2",
    ]
    assert [records[f"retry:api:{index}"].attempt for index in (1, 2)] == [2, 3]


def test_a_queued_retry_names_the_model_the_backend_already_resolved() -> None:
    """A resolved decision for this exact attempt is published; nothing else is."""
    artifacts = [
        _code_completion(0, {"coding_model": "gpt-5.6-sol"}),
        _review(0, verdict="changes_requested", metadata={"model": "gpt-5.6-sol"}),
        _result(0, failure_classification="review_scope_failure"),
    ]
    child = _child(
        retry_count=1,
        status=ChildWorkflowStatus.RUNNING,
        model_routing={
            "execution_mode": "REVIEW_REMEDIATION",
            "role": "scoped_fix",
            "model": "gpt-5.3-codex",
            "reasoning": "high",
            "attempt": 1,
            "routing_reason": "STANDARD review remediation.",
            "input_fingerprint": "sha256:abc",
        },
    )
    retry = _by_id(feature_executions(_feature(artifacts=artifacts, children={"api": child})))[
        "retry:api:1"
    ]
    assert retry.status is ExecutionStatus.RUNNING
    assert retry.model == "gpt-5.3-codex"
    assert retry.reasoning_effort == "high"
    assert retry.model_role == "scoped_fix"
    assert retry.model_resolved is True
    assert retry.routing_reason == "STANDARD review remediation."


def test_graph_edges_use_each_execution_roles_persisted_resolved_model() -> None:
    """Coding, review and scoped retry arrows are backend facts, never frontend guesses."""
    artifacts = [
        _code_completion(
            0,
            {
                "execution": {
                    "agent_type": "Engineer",
                    "model_role": "coding",
                    "model": "gpt-5.6-sol",
                    "reasoning_effort": "max",
                    "routing_reason": "Initial implementation.",
                }
            },
        ),
        _review(
            0,
            verdict="changes_requested",
            metadata={
                "execution": {
                    "agent_type": "Code reviewer",
                    "model_role": "review",
                    "model": "gpt-5.6-terra",
                    "reasoning_effort": "high",
                    "routing_reason": "Independent review.",
                }
            },
            findings=[_finding()],
        ),
        _result(0, failure_classification="review_scope_failure"),
    ]
    child = _child(
        retry_count=1,
        status=ChildWorkflowStatus.RUNNING,
        model_routing={
            "execution_mode": "REVIEW_REMEDIATION",
            "role": "scoped_fix",
            "model": "gpt-5.3-codex",
            "reasoning": "high",
            "attempt": 1,
            "routing_reason": "Localized import and lint repair.",
        },
    )

    records = _by_id(feature_executions(_feature(artifacts=artifacts, children={"api": child})))
    coding = records["implementation:api:0"]
    review = records["review:api:0"]
    scoped = records["retry:api:1"]
    assert (coding.model_role, coding.model, coding.reasoning_effort) == (
        "coding",
        "gpt-5.6-sol",
        "max",
    )
    assert (review.model_role, review.model) == ("review", "gpt-5.6-terra")
    assert (scoped.model_role, scoped.model) == ("scoped_fix", "gpt-5.3-codex")
    assert scoped.routing_reason == "Localized import and lint repair."


def test_a_stale_routing_decision_is_not_published_as_the_next_model() -> None:
    """A decision recorded for a previous attempt is not a prediction about this one."""
    artifacts = [
        _code_completion(0, {"coding_model": "gpt-5.6-sol"}),
        _review(0, verdict="changes_requested", metadata={"model": "gpt-5.6-sol"}),
        _result(0, failure_classification="review_scope_failure"),
    ]
    child = _child(
        retry_count=1,
        model_routing={"role": "fix", "model": "gpt-5.3-codex", "attempt": 0},
    )
    retry = _by_id(feature_executions(_feature(artifacts=artifacts, children={"api": child})))[
        "retry:api:1"
    ]
    assert retry.model is None
    assert retry.model_resolved is False


def test_a_refused_retry_is_published_as_needing_a_human_and_not_as_a_next_attempt() -> None:
    """Remaining capacity is not permission: no arrow may promise an attempt that will not run."""
    artifacts = [
        _code_completion(0, {"coding_model": "gpt-5.6-sol"}),
        _review(0, verdict="changes_requested", metadata={"model": "gpt-5.6-sol"}),
        _result(
            0,
            failure_classification="implementation_missing",
            model_routing={"role": "fix", "model": "gpt-5.3-codex"},
        ),
    ]
    child = _child(
        retry_count=1,
        status=ChildWorkflowStatus.FAILED,
        retry_refusal_reason=(
            "The same diagnostics came back unchanged, so this workstream is not converging."
        ),
    )
    records = _by_id(feature_executions(_feature(artifacts=artifacts, children={"api": child})))
    # Attempts 0 and 1 were spent; the attempt after them is the one that will not run, and it
    # is published as a handoff rather than as "Retry 3 / 8".
    assert "retry:api:2" not in records
    refused = records["retry-refused:api:2"]
    assert refused.status is ExecutionStatus.NEEDS_HUMAN
    assert refused.handler_type is ExecutionHandlerType.HUMAN
    assert refused.human_action is HumanAction.RETRY_REFUSED
    assert "not converging" in (refused.human_requirement or "")
    # The last model that ran is named, and explicitly not as a resolved next model.
    assert refused.model == "gpt-5.3-codex"
    assert refused.model_resolved is False
    assert refused.attempt == 2


# --------------------------------------------------------------------------- human gates


def test_an_unanswered_clarification_is_a_human_transition_with_no_model() -> None:
    """Answering a question is not something a model did."""
    prd = _technical_prd(
        metadata={"model": "gpt-5.6-sol"},
        questions=[
            {
                "question_id": "Q-1",
                "question": "Does this endpoint use the existing admin middleware?",
                "rationale": "It decides the route's guard.",
                "required": True,
            }
        ],
    )
    state = _feature(artifacts=[prd], status=FeatureWorkflowStatus.WAITING_FOR_HUMAN)
    record = _by_id(feature_executions(state))["human:clarification:002_technical_prd.json"]
    assert record.handler_type is ExecutionHandlerType.HUMAN
    assert record.model is None
    assert record.status is ExecutionStatus.NEEDS_HUMAN
    assert record.to_stage is ExecutionStage.INTEGRATION_CONTRACT
    assert "1 question" in (record.human_requirement or "")


def test_a_proposed_repository_repair_holds_the_implementation_transition() -> None:
    """The platform will not change somebody's repository, and the arrow says who must."""
    repair = RepositoryRepairProposalArtifact.model_validate(
        {
            **_envelope("016_repository_repair_proposal.api.r-1.json", "repository_repair", {}),
            "repair_id": "r-1",
            "feature_id": "f-1",
            "repository_id": "api",
            "originating_stage": "repository_preflight",
            "failure_classification": "validation_configuration_failure",
            "detected_problem": "ESLint could not load eslint-config-airbnb-base.",
            "evidence": ["npm run lint exited with code 1"],
            "proposed_repair": "Declare the missing shared config as a dev dependency.",
            "expected_impact": "Lint can bootstrap.",
            "risk": "low",
            "changes_source_logic": False,
            "proposed_at_revision": "abc123",
            "status": "proposed",
        }
    )
    state = _feature(artifacts=[repair], children={"api": _child()})
    record = _by_id(feature_executions(state))["human:repair:r-1"]
    assert record.handler_type is ExecutionHandlerType.HUMAN
    assert record.repair_id == "r-1"
    assert record.failure_summary == "ESLint could not load eslint-config-airbnb-base."
    assert record.remediation_summary == "Declare the missing shared config as a dev dependency."
    assert record.model is None


# --------------------------------------------------------------------------- shape and safety


def test_records_are_uniquely_identified_so_two_attempts_are_two_arrows() -> None:
    """A source and target pair is not an identity when retries reuse it."""
    artifacts: list[Any] = []
    for attempt in range(4):
        artifacts.append(_code_completion(attempt, {"coding_model": "m"}))
        artifacts.append(_review(attempt, verdict="changes_requested", metadata={"model": "m"}))
        artifacts.append(_result(attempt, failure_classification="implementation_missing"))
    records = feature_executions(
        _feature(artifacts=artifacts, children={"api": _child(retry_count=3)})
    )
    identifiers = [item.execution_id for item in records]
    assert len(identifiers) == len(set(identifiers))


def test_no_record_carries_a_credential_or_a_prompt() -> None:
    """The serialized record is read by a browser, so its whole surface is checked."""
    artifacts = [
        _technical_prd(
            metadata={
                "execution": {"agent_type": "Product manager", "model": "gpt-5.6-sol"},
                "prompt_template": "product_manager/v1.jinja2",
                "response_id": "resp_123",
            }
        ),
        _code_completion(0, {"coding_model": "gpt-5.6-sol", "coding_response_id": "resp_456"}),
        _review(0, verdict="approved", metadata={"model": "gpt-5.6-sol"}),
        _result(0, status="approved", pull_request_readiness=True),
    ]
    records = feature_executions(
        _feature(
            artifacts=artifacts,
            children={"api": _child(status=ChildWorkflowStatus.COMPLETED)},
        )
    )
    serialized = "\n".join(item.model_dump_json() for item in records)
    for forbidden in ("api_key", "apiKey", "sk-", "token", "prompt_template", "jinja2", "resp_"):
        assert forbidden not in serialized


def test_several_repositories_each_get_their_own_lane_of_executions() -> None:
    """Nothing here is shared between repositories, including the retry history."""
    state = _feature(
        repository_specs=[
            RepositorySpec(
                repository_id=name,
                name=name,
                role="backend",
                repository_url=HttpUrl(f"https://github.com/acme/{name}"),
                default_branch="main",
            )
            for name in ("api", "web")
        ],
        children={"api": _child(), "web": _child(child_workflow_id="f-1:web", repository_id="web")},
    )
    records = feature_executions(state)
    lanes = {item.repository_id for item in records if item.repository_id is not None}
    assert lanes == {"api", "web"}
    assert {item.execution_id for item in records} >= {"dispatch:api", "dispatch:web"}


def test_a_later_routing_decision_does_not_rewrite_an_earlier_attempt() -> None:
    """Configuration moving on is the whole risk, and the child reference is where it shows.

    The child carries the decision for the attempt in flight. If historical arrows read from
    it -- rather than from each attempt's own artifact -- every completed attempt would be
    relabelled with whatever the deployment configured most recently.
    """
    artifacts = [
        _code_completion(0, {"coding_model": "gpt-5.3-codex"}),
        _review(0, verdict="changes_requested", metadata={"model": "gpt-5.6-sol"}),
        _result(0, failure_classification="review_scope_failure"),
    ]
    child = _child(
        retry_count=1,
        model_routing={
            "role": "escalation",
            "model": "a-model-configured-today",
            "reasoning": "max",
            "attempt": 1,
        },
    )
    records = _by_id(feature_executions(_feature(artifacts=artifacts, children={"api": child})))
    assert records["implementation:api:0"].model == "gpt-5.3-codex"
    assert records["review:api:0"].model == "gpt-5.6-sol"
    assert records["retry:api:1"].model == "a-model-configured-today"


def test_a_workstream_stopped_on_a_completed_attempt_still_says_no_retry_is_coming() -> None:
    """The ordinary terminal case: the last attempt finished and its successor was declined.

    Keyed off the refusal itself rather than off an attempt being in flight. Read the other way
    -- which it once was -- exactly this shape, the common one, produced no arrow at all and a
    reader saw a graph that simply stopped.
    """
    artifacts = [
        _code_completion(0, {"coding_model": "gpt-5.6-sol"}),
        _review(0, verdict="changes_requested", metadata={"model": "gpt-5.6-sol"}),
        _result(0, failure_classification="implementation_missing"),
    ]
    child = _child(
        retry_count=0,
        status=ChildWorkflowStatus.FAILED,
        retry_refusal_reason=(
            "The previous attempt produced no meaningful production change, so repeating it "
            "would submit identical inputs."
        ),
    )
    records = _by_id(feature_executions(_feature(artifacts=artifacts, children={"api": child})))
    refused = records["retry-refused:api:1"]
    assert refused.attempt == 1
    assert refused.status is ExecutionStatus.NEEDS_HUMAN
    assert "identical inputs" in (refused.human_requirement or "")


def test_an_attempt_that_never_reached_validation_publishes_no_validation_execution() -> None:
    """A subprocess that never ran is not an execution, and must not be drawn as a pending one."""
    artifacts = [
        _code_completion(0, {"coding_model": "gpt-5.6-sol"}),
        _review(0, verdict="changes_requested", metadata={"model": "gpt-5.6-sol"}),
        _result(0, failure_classification="implementation_missing"),
    ]
    records = _by_id(feature_executions(_feature(artifacts=artifacts, children={"api": _child()})))
    assert "validation:api:0" not in records
    assert "implementation:api:0" in records


def test_a_contract_only_integration_review_names_its_validator_and_not_a_model() -> None:
    """The seam review is the model half. Where it did not run, no model performed this."""
    integration = _integration_review(
        {"contract_validator": "openapi", "contract_version": "1.0.0"}
    )
    state = _feature(
        artifacts=[integration],
        children={"api": _child(status=ChildWorkflowStatus.COMPLETED)},
    )
    record = _by_id(feature_executions(state))[f"integration_review:{integration.artifact_id}"]
    assert record.handler_type is ExecutionHandlerType.DETERMINISTIC
    assert record.handler == "Contract validator"
    assert record.model is None


def test_a_seam_review_that_did_run_names_the_model_that_read_the_seams() -> None:
    """And where it did, the model it recorded is the handler."""
    integration = _integration_review(
        {"contract_validator": "openapi", "seam_review_model": "gpt-5.6-sol"}
    )
    state = _feature(
        artifacts=[integration],
        children={"api": _child(status=ChildWorkflowStatus.COMPLETED)},
    )
    record = _by_id(feature_executions(state))[f"integration_review:{integration.artifact_id}"]
    assert record.handler_type is ExecutionHandlerType.MODEL
    assert record.model == "gpt-5.6-sol"
    assert record.agent_type == "Integration reviewer"


def _stopped_feature(*, triage: dict[str, Any] | None) -> FeatureWorkflowSnapshot:
    """A workstream that wrote production source for attempts and then stopped on nothing.

    Modelled on AB-Feature-158's backend: eleven attempts changed production source, the final
    one changed none, and the reliability logic refused another.
    """
    result = _result(
        1,
        failure_classification="implementation_missing",
        metadata=triage or {},
        production_files_changed=[],
    )
    child = _child(
        retry_count=11,
        status=ChildWorkflowStatus.FAILED,
        failure_classification="implementation_missing",
        blocking_issues=[
            "The requirement expects production implementation but no production source "
            "file changed.",
            "The workstream requires tests but no test file changed.",
            "The requirement expects production implementation but no production source "
            "file changed.",
            "The workstream requires tests but no test file changed.",
            "The requirement expects production implementation but no production source "
            "file changed.",
        ],
        retry_refusal_reason=(
            "The previous attempt produced no meaningful production change, so repeating it "
            "would submit identical inputs."
        ),
    )
    return _feature(
        artifacts=[result],
        children={"api": child},
        status=FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN,
        current_agent="github",
    )


def test_a_feature_summary_reports_the_workstream_triage_not_the_last_attempt_alone() -> None:
    """The first thing an operator reads must not contradict the triage underneath it.

    AB-Feature-158 reported `implementation_missing` with "no production source file changed"
    five times over, while its own triage had already concluded `budget_exhausted` from
    "12 attempts changed production source and the blocker moved". A reader was sent to look
    for a missing implementation that had been there one attempt earlier.
    """
    state = _stopped_feature(
        triage={
            "terminal_cause": "budget_exhausted",
            "terminal_evidence": ["12 attempts changed production source and the blocker moved."],
            "operator_question": (
                "The last attempt submitted the same source as the one before it, so this "
                "workstream is not converging on its own. Is the requirement achievable in "
                "this repository as written?"
            ),
        }
    )
    summary = ensure_feature_failure_summary(state).failure_summary
    assert summary is not None
    assert summary.terminal_cause == "budget_exhausted"
    # The evidence about the whole workstream leads, rather than being truncated away by the
    # final attempt's five identical blockers.
    assert summary.diagnostics[0] == (
        "12 attempts changed production source and the blocker moved."
    )
    assert "achievable in this repository" in summary.next_action
    # The classification of what blocked the last attempt is still reported as itself.
    assert summary.root_classification == "implementation_missing"


def test_a_feature_that_recorded_no_triage_keeps_the_summary_it_always_had() -> None:
    """Nothing is invented for a feature that stopped before a workstream was triaged."""
    summary = ensure_feature_failure_summary(_stopped_feature(triage=None)).failure_summary
    assert summary is not None
    assert summary.terminal_cause is None
    assert summary.diagnostics[0].startswith("The requirement expects production")
    assert "identical inputs" in summary.next_action


def _targeted(attempt: int, kind: str | None, **overrides: Any) -> ChildWorkflowResultArtifact:
    """One attempt's result, carrying the authority that demanded it on its own record."""
    return _result(
        attempt,
        metadata={"targeted_attempt_kind": kind} if kind is not None else {},
        **overrides,
    )


def test_d1_an_integration_remediation_retry_starts_at_the_integration_review() -> None:
    """65 D1: the arrow starts at the authority that demanded the rework.

    Observed live on 201: both children finished, the integration review requested changes,
    and the remediation attempts drew as orange loops out of each lane's own Review node --
    attributing to the repository reviewer a rework the integration review had demanded.

    The discriminator did not exist before this change. Three candidates were all wrong: the
    true fact was an in-flight parameter, the previous result's classification fails in both
    directions, and `integration_retry_count` is cumulative. So the kind is stamped, and this
    is the record reading it.
    """
    records = _by_id(
        feature_executions(
            _feature(
                artifacts=[
                    _code_completion(0, {}),
                    _review(0, verdict="approved", metadata={}),
                    _result(0, status="approved"),
                    _code_completion(1, {}),
                    _targeted(1, "integration_remediation", status="approved"),
                ],
                children={"api": _child(retry_count=1, status=ChildWorkflowStatus.APPROVED)},
            )
        )
    )

    retry = records["retry:api:1"]
    assert retry.from_stage is ExecutionStage.INTEGRATION_REVIEW
    assert retry.to_stage is ExecutionStage.IMPLEMENTATION
    # The execution id shape is untouched, so nothing keyed on it moves.
    assert retry.is_retry is True


def test_d2_a_workstream_retry_is_unchanged() -> None:
    """65 D2: a review-driven retry keeps `review → implementation`, exactly as today."""
    records = _by_id(
        feature_executions(
            _feature(
                artifacts=[
                    _code_completion(0, {}),
                    _review(0, verdict="changes_requested", metadata={}, findings=[_finding()]),
                    _result(0, status="failed", failure_classification="review_scope_failure"),
                    _code_completion(1, {}),
                    _targeted(1, "operator_retry", status="approved"),
                ],
                children={"api": _child(retry_count=1, status=ChildWorkflowStatus.APPROVED)},
            )
        )
    )

    assert records["retry:api:1"].from_stage is ExecutionStage.REVIEW


def test_d3_a_record_written_before_the_stamp_existed_still_draws_from_the_lane() -> None:
    """65 D3: "no stamp" is a workstream retry, not an unknown.

    Every attempt of every run that predates this change carries no kind. Treating that as
    unknown -- or as a remediation -- would either move records that have always drawn from
    the lane loop or make them vanish from the graph entirely.
    """
    records = _by_id(
        feature_executions(
            _feature(
                artifacts=[
                    _code_completion(0, {}),
                    _review(0, verdict="changes_requested", metadata={}, findings=[_finding()]),
                    _result(0, status="failed"),
                    _code_completion(1, {}),
                    _targeted(1, None, status="approved"),
                ],
                children={"api": _child(retry_count=1, status=ChildWorkflowStatus.APPROVED)},
            )
        )
    )

    assert records["retry:api:1"].from_stage is ExecutionStage.REVIEW


def test_d4_each_attempt_reads_its_own_stamp_and_never_the_childs() -> None:
    """65 D4: two workstream retries and two remediations, none counted on both arrows.

    The trap this exists for: the stamp lives on the child, which has moved on by the time
    anybody reads the history. Reading it there would draw every past retry from whichever
    authority demanded the latest one -- and D1 and D2 each test one arrow in isolation and
    would both pass with that shipped.
    """
    artifacts: list[Any] = [_code_completion(0, {}), _result(0, status="failed")]
    remediation = "integration_remediation"
    kinds = ["operator_retry", remediation, "operator_retry", remediation]
    for attempt, kind in enumerate(kinds, start=1):
        artifacts.extend([_code_completion(attempt, {}), _targeted(attempt, kind, status="failed")])
    records = _by_id(
        feature_executions(
            _feature(
                artifacts=artifacts,
                # The child is on the last attempt, whose kind is a remediation. Every earlier
                # retry must still read its own.
                children={
                    "api": _child(
                        retry_count=4,
                        integration_retry_count=2,
                        targeted_attempt_kind="integration_remediation",
                        status=ChildWorkflowStatus.FAILED,
                    )
                },
            )
        )
    )

    assert [records[f"retry:api:{index}"].from_stage for index in range(1, 5)] == [
        ExecutionStage.REVIEW,
        ExecutionStage.INTEGRATION_REVIEW,
        ExecutionStage.REVIEW,
        ExecutionStage.INTEGRATION_REVIEW,
    ]


def test_d4_the_integration_counter_is_reported_against_the_limit_it_is_refused_against() -> None:
    """65 D4: one arrow, one number.

    `_counter` mapped `integration_retry_count` to `max_child_review_cycles`, while
    `_integration_remediation_decision` refuses the attempt against
    `max_integration_review_cycles` and adds no grant. Two different numbers for one arrow is
    how "2 / 8" came to mean nothing in particular.
    """
    records = _by_id(
        feature_executions(
            _feature(
                artifacts=[
                    _code_completion(0, {}),
                    _result(
                        0,
                        status="failed",
                        failure_classification="contract_mismatch",
                        metadata={"integration_retry_count": 2},
                    ),
                    _code_completion(1, {}),
                    _targeted(1, "integration_remediation", status="approved"),
                ],
                children={
                    "api": _child(
                        retry_count=1,
                        integration_retry_count=2,
                        granted_extra_attempts=3,
                        status=ChildWorkflowStatus.APPROVED,
                    )
                },
                max_integration_review_cycles=5,
                max_child_review_cycles=8,
            )
        )
    )

    retry = records["retry:api:1"]
    assert retry.counter_label == "Integration retries"
    assert (retry.counter_value, retry.counter_limit) == (2, 5)


def test_d5_a_refusal_recorded_during_remediation_moves_too() -> None:
    """65 D5: the refusal record is derived exactly as the granted retry's is.

    A refusal recorded while the child was in integration remediation is a refusal of the
    integration review's request. Drawing it from the lane's own reviewer misattributes it in
    the same way, and "same for the refusal record" is otherwise untested.
    """
    records = _by_id(
        feature_executions(
            _feature(
                artifacts=[
                    _code_completion(0, {}),
                    _review(0, verdict="changes_requested", metadata={}, findings=[_finding()]),
                    _targeted(0, "integration_remediation", status="failed"),
                ],
                children={
                    "api": _child(
                        retry_count=0,
                        status=ChildWorkflowStatus.FAILED,
                        targeted_attempt_kind="integration_remediation",
                        retry_refusal_reason="The integration allowance of 5 is exhausted.",
                    )
                },
            )
        )
    )

    refused = records["retry-refused:api:1"]
    assert refused.from_stage is ExecutionStage.INTEGRATION_REVIEW
    assert refused.human_action is HumanAction.RETRY_REFUSED


def test_the_two_correct_review_origins_are_untouched() -> None:
    """65 D: only the two retry builders move.

    Two more `from_stage=REVIEW` sites exist and are right -- the human contract-change record
    and the integration-review record itself, both `review → integration_review`. A
    grep-and-replace over the origin string would have broken both.
    """
    records = feature_executions(
        _feature(
            artifacts=[
                _code_completion(0, {}),
                _review(0, verdict="approved", metadata={}),
                _result(0, status="approved"),
                _integration_review({}),
            ],
            children={"api": _child(retry_count=0, status=ChildWorkflowStatus.APPROVED)},
        )
    )

    integration = next(
        item for item in records if item.to_stage is ExecutionStage.INTEGRATION_REVIEW
    )
    assert integration.from_stage is ExecutionStage.REVIEW
