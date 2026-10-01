"""Schema and generator tests for the versioned multi-repository artifact family."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from artifacts.schemas import (
    CompatibilityPolicy,
    EndpointContract,
    IntegrationContractArtifact,
    RepositoryExecutionPlanArtifact,
    RepositoryWorkstreamPlan,
    SharedSchemaDefinition,
)
from tools.contract_tools import OpenAPIContractCodeGenerator


@pytest.mark.asyncio
async def test_rest_contract_has_stable_operation_ids_and_generates_openapi() -> None:
    """The integration contract is validated once and projected into a reusable OpenAPI document."""
    contract = contract_artifact()

    openapi = await OpenAPIContractCodeGenerator().generate_openapi(contract)

    assert openapi is not None
    assert openapi["paths"]["/v1/login"]["post"]["operationId"] == "createLogin"
    assert openapi["components"]["schemas"]["LoginRequest"]["type"] == "object"


def test_contract_rejects_duplicate_operation_ids() -> None:
    """A planner cannot publish ambiguous generated-client method names."""
    payload = contract_artifact().model_dump(mode="python")
    payload["endpoints"] = [*payload["endpoints"], payload["endpoints"][0]]

    with pytest.raises(ValidationError, match="operation identifiers"):
        IntegrationContractArtifact.model_validate(payload)


def test_contract_rejects_an_unordered_revision_label() -> None:
    """Contract revisions must be comparable before a human approval can replace a version."""
    payload = contract_artifact().model_dump(mode="python")
    payload["contract_version"] = "next"

    with pytest.raises(ValidationError, match="semantic versioning"):
        IntegrationContractArtifact.model_validate(payload)


def test_repository_plan_rejects_dependencies_scheduled_in_the_same_parallel_group() -> None:
    """Untrusted planner output cannot start a contract consumer before its producer is ready."""
    with pytest.raises(ValidationError, match="dependencies in an earlier group"):
        RepositoryExecutionPlanArtifact.model_validate(
            {
                **artifact_envelope("010_repository_execution_plan.json"),
                "feature_id": "feature-contract",
                "contract_artifact_id": "009_integration_contract.json",
                "workstreams": [
                    workstream("backend", dependencies=[]),
                    workstream("frontend", dependencies=["backend"]),
                ],
                "execution_order": ["backend", "frontend"],
                "parallel_groups": [["backend", "frontend"]],
                "integration_test_plan": ["Verify the shared contract."],
                "merge_strategy": "backend_first",
                "deployment_strategy": "backend_first",
                "feature_flag_strategy": [],
                "rollback_strategy": ["Revert backend first."],
            }
        )


def test_repository_plan_accepts_legacy_artifacts_without_requirement_ownership() -> None:
    """Deploying scoped reviews must not make previously persisted plans unreadable."""
    legacy_workstream = workstream("backend", dependencies=[])
    del legacy_workstream["requirement_ids"]

    plan = RepositoryExecutionPlanArtifact.model_validate(
        {
            **artifact_envelope("010_repository_execution_plan.json"),
            "feature_id": "feature-contract",
            "contract_artifact_id": "009_integration_contract.json",
            "workstreams": [legacy_workstream],
            "execution_order": ["backend"],
            "parallel_groups": [["backend"]],
            "integration_test_plan": ["Verify the shared contract."],
            "merge_strategy": "backend_first",
            "deployment_strategy": "backend_first",
            "feature_flag_strategy": [],
            "rollback_strategy": ["Revert backend first."],
        }
    )

    assert plan.workstreams[0].requirement_ids == []


def test_a_plan_written_before_task_ordering_existed_still_reads() -> None:
    """A workstream from before task_dependencies existed has no ordering, not a corrupt row."""
    plan = RepositoryWorkstreamPlan.model_validate(workstream("backend", dependencies=[]))

    assert plan.task_dependencies == []


def test_a_workstream_cannot_order_a_task_it_does_not_declare() -> None:
    """A dependency naming a task outside this workstream's own task_ids is not a real edge."""
    payload = {
        **workstream("backend", dependencies=[]),
        "task_dependencies": [{"task_id": "backend-task", "depends_on": ["nope"]}],
    }

    with pytest.raises(ValidationError, match="does not declare"):
        RepositoryWorkstreamPlan.model_validate(payload)


def test_a_workstream_task_cannot_depend_on_itself() -> None:
    """A task named as its own dependency is refused rather than silently accepted."""
    payload = {
        **workstream("backend", dependencies=[]),
        "task_dependencies": [{"task_id": "backend-task", "depends_on": ["backend-task"]}],
    }

    with pytest.raises(ValidationError, match="cannot depend on itself"):
        RepositoryWorkstreamPlan.model_validate(payload)


def artifact_envelope(artifact_id: str) -> dict[str, object]:
    """Return the common strict envelope for a feature-level artifact fixture."""
    return {
        "schema_version": "1.0",
        "workflow_id": "feature-contract",
        "artifact_id": artifact_id,
        "producer": "feature_planner",
        "timestamp": datetime(2026, 8, 2, tzinfo=UTC),
        "metadata": {"source": "test"},
        "validation_status": "valid",
    }


def workstream(repository_id: str, *, dependencies: list[str]) -> dict[str, object]:
    """Return one minimal repository workstream for schedule validation."""
    return {
        "workstream_id": repository_id,
        "repository_id": repository_id,
        "role": repository_id,
        "requirement_ids": ["requirement-login"],
        "responsibilities": [f"Implement {repository_id}."],
        "task_ids": [f"{repository_id}-task"],
        "dependency_workstream_ids": dependencies,
        "contract_sections_consumed": [],
        "contract_sections_implemented": [],
        "acceptance_criteria": ["Review passes."],
        "test_requirements": ["Run tests."],
        "documentation_requirements": [],
        "expected_files_or_areas": [],
        "required": True,
    }


def contract_artifact() -> IntegrationContractArtifact:
    """Return a small approved REST contract shared by frontend and backend workstreams."""
    return IntegrationContractArtifact(
        schema_version="1.0",
        workflow_id="feature-contract",
        artifact_id="009_integration_contract.json",
        producer="feature_planner",
        timestamp=datetime(2026, 8, 2, tzinfo=UTC),
        metadata={"source": "test"},
        validation_status="valid",
        feature_id="feature-contract",
        contract_version="1.0.0",
        status="approved",
        api_style="rest",
        endpoints=[
            EndpointContract(
                operation_id="createLogin",
                method="POST",
                path="/v1/login",
                summary="Create a login session.",
                request_schema={"$ref": "#/components/schemas/LoginRequest"},
                response_schema={"$ref": "#/components/schemas/LoginResponse"},
                query_parameters=[],
                path_parameters=[],
                headers=[],
                authentication_required=False,
                authorization_requirements=[],
                success_status_codes=[200],
                error_codes=["invalid_credentials"],
                version="v1",
            )
        ],
        shared_schemas=[
            SharedSchemaDefinition(
                name="LoginRequest",
                description="Credentials submitted by the web client.",
                json_schema={"type": "object", "properties": {"email": {"type": "string"}}},
            )
        ],
        authentication_contract=None,
        authorization_rules=[],
        error_contracts=[],
        event_contracts=[],
        environment_variables=[],
        compatibility_policy=CompatibilityPolicy(
            policy="Additive changes only.",
            breaking_change_allowed=False,
            migration_requirements=[],
            rollback_requirements=["Revert coordinated changes."],
        ),
        owning_workstreams=["backend", "frontend"],
        approved_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
