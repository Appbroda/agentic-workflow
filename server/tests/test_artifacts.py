"""Tests for serialization and validation of workflow artifact schemas."""

from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

import pytest
from pydantic import HttpUrl, ValidationError

from artifacts.schemas import (
    APIContract,
    ArchitectureArtifact,
    ArchitectureComponent,
    ArchitectureDecision,
    ArchitectureRisk,
    BaseArtifact,
    CodeCompletionArtifact,
    DataEntity,
    DataField,
    ExecutionEdge,
    ExecutionGraphArtifact,
    ExecutionNode,
    FileChange,
    Milestone,
    PlannedTask,
    PRDArtifact,
    PullRequestArtifact,
    Requirement,
    RequirementCheck,
    ReviewArtifact,
    ReviewFinding,
    TaskPlanArtifact,
    TechnicalPRDArtifact,
    UserStory,
    ValidationResult,
)


class CommonArtifactFields(TypedDict):
    """Typed common fields accepted by all concrete artifact constructors."""

    schema_version: str
    workflow_id: str
    artifact_id: str
    producer: str
    timestamp: datetime
    metadata: dict[str, Any]
    validation_status: Literal["valid", "invalid", "pending"]


def base_fields() -> CommonArtifactFields:
    """Return the required common envelope for a valid test artifact."""
    return {
        "schema_version": "1.0",
        "workflow_id": "workflow-123",
        "artifact_id": "artifact-123",
        "producer": "test-agent",
        "timestamp": datetime(2026, 8, 1, 10, 30, tzinfo=UTC),
        "metadata": {"source": "unit-test"},
        "validation_status": "valid",
    }


def requirement(requirement_id: str) -> Requirement:
    """Build a traceable requirement used by several artifact fixtures."""
    return Requirement(
        requirement_id=requirement_id,
        description="The platform must provide a testable capability.",
        priority="must",
        acceptance_criteria=["A valid request returns a structured result."],
        dependencies=[],
    )


def artifact_cases() -> list[tuple[str, BaseArtifact]]:
    """Build one complete instance of every supported workflow artifact."""
    prd = PRDArtifact(
        **base_fields(),
        title="Artifact-driven platform",
        problem_statement="Teams need a reliable way to automate bounded engineering workflows.",
        goals=["Produce auditable software changes."],
        user_stories=[
            UserStory(
                story_id="story-1",
                persona="Engineering manager",
                need="Review each generated change",
                benefit="I can safely supervise automated delivery",
                acceptance_criteria=["A review artifact is generated before a pull request."],
            )
        ],
        requirements=[requirement("prd-req-1")],
        constraints=["Use request-scoped credentials."],
        out_of_scope=["Direct pushes to the default branch."],
        stakeholders=["Engineering"],
    )
    technical_prd = TechnicalPRDArtifact(
        **base_fields(),
        title="Artifact pipeline technical PRD",
        solution_summary="Use strict Pydantic schemas as workflow contracts.",
        functional_requirements=[requirement("tech-functional-1")],
        non_functional_requirements=[requirement("tech-quality-1")],
        data_requirements=["Persist artifacts with versioned metadata."],
        integration_requirements=["Expose artifacts through the workflow API."],
        security_requirements=["Reject unknown request fields."],
        assumptions=["All workflow agents run inside a trusted workspace."],
        unresolved_questions=[],
    )
    architecture = ArchitectureArtifact(
        **base_fields(),
        system_overview="FastAPI orchestrates artifact-driven LangGraph workflows.",
        technology_stack={"api": "FastAPI", "workflow": "LangGraph"},
        repository_structure=["artifacts/schemas.py", "workflows/"],
        components=[
            ArchitectureComponent(
                component_id="api",
                name="API",
                responsibility="Accept and expose workflow requests.",
                technology="FastAPI",
                dependencies=["workflow"],
            ),
            ArchitectureComponent(
                component_id="workflow",
                name="Workflow engine",
                responsibility="Coordinate agent state transitions.",
                technology="LangGraph",
                dependencies=[],
            ),
        ],
        api_contracts=[
            APIContract(
                method="POST",
                path="/workflow/start",
                summary="Start a workflow.",
                request_schema="StartWorkflowRequest",
                response_schema="WorkflowResponse",
                authentication_required=True,
            )
        ],
        data_entities=[
            DataEntity(
                entity_name="Artifact",
                description="A versioned workflow output.",
                fields=[
                    DataField(
                        name="artifact_id",
                        data_type="string",
                        nullable=False,
                        description="Stable artifact identifier.",
                    )
                ],
            )
        ],
        decisions=[
            ArchitectureDecision(
                decision_id="adr-1",
                title="Use Pydantic",
                decision="Use strict Pydantic models for artifact contracts.",
                rationale="Runtime validation protects agent boundaries.",
                consequences=["Unknown fields are rejected."],
            )
        ],
        risks=[
            ArchitectureRisk(
                risk_id="risk-1",
                description="An invalid artifact could stop the workflow.",
                likelihood="medium",
                impact="high",
                mitigation="Validate every artifact before storage.",
            )
        ],
        deployment_strategy="Deploy the stateless API behind a managed load balancer.",
    )
    execution_graph = ExecutionGraphArtifact(
        **base_fields(),
        entry_node_id="product-manager",
        terminal_node_ids=["github"],
        nodes=[
            ExecutionNode(
                node_id="product-manager",
                agent="product_manager",
                action="create_technical_prd",
                input_artifact_types=["prd"],
                output_artifact_type="technical_prd",
                requires_human_approval=True,
            ),
            ExecutionNode(
                node_id="github",
                agent="github",
                action="create_pull_request",
                input_artifact_types=["review"],
                output_artifact_type="pull_request",
                requires_human_approval=False,
            ),
        ],
        edges=[
            ExecutionEdge(
                source_node_id="product-manager",
                target_node_id="github",
                condition="review approved",
            )
        ],
    )
    task_plan = TaskPlanArtifact(
        **base_fields(),
        summary="Implement the typed artifact contracts before adding workflow persistence.",
        tasks=[
            PlannedTask(
                task_id="task-1",
                title="Define artifact schemas",
                description="Implement strict Pydantic models.",
                owner="engineer",
                priority="high",
                dependencies=[],
                acceptance_criteria=["All artifact types serialize successfully."],
                estimated_effort_points=3,
            ),
            PlannedTask(
                task_id="task-2",
                title="Validate artifact schemas",
                description="Add unit tests for artifact contracts.",
                owner="reviewer",
                priority="high",
                dependencies=["task-1"],
                acceptance_criteria=["Invalid fields are rejected."],
                estimated_effort_points=2,
            ),
        ],
        milestones=[
            Milestone(
                milestone_id="milestone-1",
                title="Artifact system",
                objective="Deliver tested artifact schemas.",
                task_ids=["task-1", "task-2"],
            )
        ],
        implementation_order=["task-1", "task-2"],
        test_strategy=["Run unit tests for serialization and validation."],
        risk_management_plan=["Block storage of invalid artifacts."],
    )
    code_completion = CodeCompletionArtifact(
        **base_fields(),
        completion_status="completed",
        summary="Implemented strict artifact schemas and unit tests.",
        file_changes=[
            FileChange(
                path="artifacts/schemas.py",
                change_type="added",
                description="Added versioned artifact models.",
            )
        ],
        validation_results=[
            ValidationResult(
                name="pytest",
                command="uv run pytest",
                passed=True,
                output_summary="All tests passed.",
                duration_seconds=1.2,
            )
        ],
        test_coverage_percent=95.0,
        remaining_work=[],
        commit_sha="0123456789abcdef",
    )
    review = ReviewArtifact(
        **base_fields(),
        verdict="approved",
        summary="The artifact contracts meet the Phase 2 acceptance criteria.",
        requirement_checks=[
            RequirementCheck(
                requirement_id="tech-functional-1",
                passed=True,
                evidence="Serialization tests pass for every artifact type.",
            )
        ],
        findings=[
            ReviewFinding(
                finding_id="finding-1",
                severity="info",
                title="Documentation opportunity",
                description=(
                    "Artifact contract examples can be expanded in a later documentation phase."
                ),
                recommendation="Add examples when API documentation is introduced.",
                file_path="artifacts/schemas.py",
                line_number=1,
            )
        ],
        architecture_assessment="The schema hierarchy keeps shared fields consistent.",
        security_assessment="Strict validation rejects undeclared input fields.",
        test_coverage_assessment="All eight artifact schemas have serialization coverage.",
    )
    pull_request = PullRequestArtifact(
        **base_fields(),
        repository="example/ai-software-engineering-platform",
        pull_request_number=42,
        url=HttpUrl("https://github.com/example/ai-software-engineering-platform/pull/42"),
        title="Implement artifact schemas",
        body="Adds validated, versioned workflow artifact contracts.",
        source_branch="feature/artifact-schemas",
        target_branch="main",
        commit_sha="0123456789abcdef",
        labels=["enhancement", "artifacts"],
        reviewers=["platform-team"],
        state="open",
    )
    return [
        ("prd", prd),
        ("technical_prd", technical_prd),
        ("architecture", architecture),
        ("execution_graph", execution_graph),
        ("task_plan", task_plan),
        ("code_completion", code_completion),
        ("review", review),
        ("pull_request", pull_request),
    ]


@pytest.mark.parametrize(("artifact_kind", "artifact"), artifact_cases())
def test_artifacts_round_trip_through_json(artifact_kind: str, artifact: BaseArtifact) -> None:
    """Every artifact must preserve its type and fields through JSON serialization."""
    serialized = artifact.model_dump_json()

    restored = type(artifact).model_validate_json(serialized)

    assert restored == artifact
    assert restored.model_dump(mode="json")["artifact_type"] == artifact_kind


@pytest.mark.parametrize(("_artifact_kind", "artifact"), artifact_cases())
def test_artifacts_reject_unknown_and_missing_common_fields(
    _artifact_kind: str, artifact: BaseArtifact
) -> None:
    """Every artifact should reject uncontracted input and incomplete envelopes."""
    payload_with_unknown_field = artifact.model_dump(mode="python")
    payload_with_unknown_field["uncontracted"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(artifact).model_validate(payload_with_unknown_field)

    payload_without_workflow_id = artifact.model_dump(mode="python")
    del payload_without_workflow_id["workflow_id"]
    with pytest.raises(ValidationError, match="Field required"):
        type(artifact).model_validate(payload_without_workflow_id)


def test_base_artifact_rejects_naive_timestamp() -> None:
    """Artifact timestamps must carry a timezone."""
    payload = base_fields()
    payload["timestamp"] = datetime(2026, 8, 1, 10, 30)

    with pytest.raises(ValidationError, match="timezone-aware"):
        BaseArtifact.model_validate(payload)


def test_artifacts_are_immutable_after_publication() -> None:
    """Attempt history must be revised with a new artifact, never rewritten in place."""
    _, artifact = artifact_cases()[0]

    with pytest.raises(ValidationError, match="Instance is frozen"):
        artifact.validation_status = "pending"


def test_execution_graph_rejects_dangling_edges() -> None:
    """Workflow graph transitions must point to declared nodes."""
    _, artifact = next(case for case in artifact_cases() if case[0] == "execution_graph")
    payload = artifact.model_dump(mode="python")
    payload["edges"] = [
        {
            "source_node_id": "product-manager",
            "target_node_id": "unknown-node",
            "condition": "always",
        }
    ]

    with pytest.raises(ValidationError, match="must reference nodes"):
        ExecutionGraphArtifact.model_validate(payload)


def test_task_plan_rejects_cyclic_dependencies() -> None:
    """Implementation plans must not make tasks mutually dependent."""
    _, artifact = next(case for case in artifact_cases() if case[0] == "task_plan")
    payload = artifact.model_dump(mode="python")
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    first_task = tasks[0]
    assert isinstance(first_task, dict)
    first_task["dependencies"] = ["task-2"]

    with pytest.raises(ValidationError, match="must not contain a cycle"):
        TaskPlanArtifact.model_validate(payload)


def test_pull_request_rejects_equal_source_and_target_branches() -> None:
    """The GitHub artifact must not describe a self-targeting pull request."""
    _, artifact = next(case for case in artifact_cases() if case[0] == "pull_request")
    payload = artifact.model_dump(mode="python")
    payload["target_branch"] = "feature/artifact-schemas"

    with pytest.raises(ValidationError, match="must differ"):
        PullRequestArtifact.model_validate(payload)
