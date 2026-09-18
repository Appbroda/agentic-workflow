"""Regression coverage for live multi-repository requirement ownership."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import HttpUrl

from adapters.llm_adapter import ImageInput, LLMResponse
from agents.planner.feature_planner import DeterministicFeaturePlanner, FeaturePlannerAgent
from agents.shared.contracts import AgentArtifactError
from artifacts.schemas import Requirement, TechnicalPRDArtifact
from prompts.prompt_loader import PromptLoader
from state.enums import WorkstreamRole
from state.feature_models import RepositorySpec
from tools.contract_tools import MockContractCodeGenerator


class QueuedLLMClient:
    """Return deterministic planning responses while recording each model call."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, str]] = []

    @property
    def vision_capable(self) -> bool:
        """No image reaches this double, so the boundary answers False."""
        return False

    async def respond(
        self,
        *,
        instructions: str,
        input_text: str,
        images: Sequence[ImageInput] = (),
    ) -> LLMResponse:
        self.calls.append((instructions, input_text))
        return LLMResponse(
            response_id=f"response-{len(self.calls)}",
            model="test-model",
            output_text=json.dumps(next(self._responses)),
            input_tokens=1,
            output_tokens=1,
        )


@pytest.mark.asyncio
async def test_feature_planner_allows_shared_requirement_implementation_in_each_repository() -> (
    None
):
    """Backend and frontend may each implement their side of one shared API contract."""
    shared_plan = _execution_plan(
        frontend_reference={
            "requirement_id": "server-status-contract",
            "acceptance_criterion_ids": ["server-status-contract:ac-1"],
            "responsibility": "implements",
        }
    )
    client = QueuedLLMClient([_planning_response(shared_plan)])
    planner = FeaturePlannerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        contract_generator=MockContractCodeGenerator(),
    )

    _, _, plan = await planner.plan(
        feature_id="feature-plan-repair",
        technical_prd=_technical_prd(),
        repositories=_repositories(),
    )

    assert len(client.calls) == 1
    frontend = next(item for item in plan.workstreams if item.repository_id == "frontend")
    assert frontend.scoped_requirements[0].responsibility == "implements"


@pytest.mark.asyncio
async def test_feature_planner_repairs_a_plan_that_drops_requirement_ownership() -> None:
    """A model losing requirement ownership is repaired inside planning, never as a 500."""
    invalid_plan = _execution_plan(
        frontend_reference={
            "requirement_id": "server-status-contract",
            "acceptance_criterion_ids": ["server-status-contract:ac-1"],
            "responsibility": "implements",
        }
    )
    for workstream in invalid_plan["workstreams"]:
        workstream["scoped_requirements"] = []
        workstream["shared_requirements"] = []
        workstream["requirement_ids"] = []
        workstream["implementation_expectations"] = []
    fixed_plan = _execution_plan(
        frontend_reference={
            "requirement_id": "server-status-contract",
            "acceptance_criterion_ids": ["server-status-contract:ac-1"],
            "responsibility": "implements",
        }
    )
    client = QueuedLLMClient([_planning_response(invalid_plan), _planning_response(fixed_plan)])
    planner = FeaturePlannerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        contract_generator=MockContractCodeGenerator(),
    )

    _, _, plan = await planner.plan(
        feature_id="feature-plan-order-repair",
        technical_prd=_technical_prd(),
        repositories=_repositories(),
    )

    assert len(client.calls) == 2
    assert plan.metadata["repair_of_response_id"] == "response-1"
    assert {item.workstream_id for item in plan.workstreams} == {"backend", "frontend"}
    # Feature -024 was rejected for two field errors and its repair came back with sixteen,
    # having rewritten fields nothing complained about. The repair must anchor the model to
    # its own previous response and to the exact errors, not restate the task alone.
    repair_instructions = client.calls[1][0]
    assert "Here is your previous response:" in repair_instructions
    # The rejected response itself, so the model corrects rather than re-plans from scratch.
    assert '"workstream_id": "backend"' in repair_instructions
    assert "Here is exactly what was wrong with it:" in repair_instructions
    assert "byte-identical to your previous response" in repair_instructions
    assert "do not introduce empty strings" in repair_instructions


@pytest.mark.asyncio
async def test_feature_planner_does_not_persist_rejected_model_values() -> None:
    """Repair metadata and terminal diagnostics contain platform categories, not payload data."""
    secret = "PLATFORM_API_KEY=malicious-model-value"
    invalid_plan = _execution_plan(
        frontend_reference={
            "requirement_id": secret,
            "acceptance_criterion_ids": [f"{secret}:ac-1"],
            "responsibility": "implements",
        }
    )
    fixed_plan = _execution_plan(
        frontend_reference={
            "requirement_id": "server-status-contract",
            "acceptance_criterion_ids": ["server-status-contract:ac-1"],
            "responsibility": "implements",
        }
    )
    repaired_client = QueuedLLMClient(
        [_planning_response(invalid_plan), _planning_response(fixed_plan)]
    )
    repaired_planner = FeaturePlannerAgent(
        prompt_loader=PromptLoader(),
        llm_client=repaired_client,
        contract_generator=MockContractCodeGenerator(),
    )

    _, _, repaired = await repaired_planner.plan(
        feature_id="feature-plan-safe-error",
        technical_prd=_technical_prd(),
        repositories=_repositories(),
    )

    assert repaired.metadata["repair_reason"] == (
        "workstream assigns unknown technical requirements"
    )
    assert secret not in json.dumps(repaired.metadata)

    failing_client = QueuedLLMClient(
        [_planning_response(invalid_plan), _planning_response(invalid_plan)]
    )
    failing_planner = FeaturePlannerAgent(
        prompt_loader=PromptLoader(),
        llm_client=failing_client,
        contract_generator=MockContractCodeGenerator(),
    )

    with pytest.raises(AgentArtifactError) as raised:
        await failing_planner.plan(
            feature_id="feature-plan-safe-terminal-error",
            technical_prd=_technical_prd(),
            repositories=_repositories(),
        )

    assert raised.value.diagnostics == (
        "first attempt rejected: workstream assigns unknown technical requirements",
        "repair attempt rejected: workstream assigns unknown technical requirements",
    )
    assert secret not in " ".join(raised.value.diagnostics)


@pytest.mark.asyncio
async def test_feature_planner_does_not_persist_validation_error_values() -> None:
    """Custom Pydantic validator messages cannot smuggle rejected model text into state."""
    secret = "github_pat_malicious_model_value"
    plan = _execution_plan(
        frontend_reference={
            "requirement_id": "server-status-contract",
            "acceptance_criterion_ids": ["server-status-contract:ac-1"],
            "responsibility": "implements",
        }
    )
    invalid_response = _planning_response(plan)
    invalid_response["architecture"]["components"][0]["dependencies"] = [secret]
    repaired_client = QueuedLLMClient([invalid_response, _planning_response(plan)])
    repaired_planner = FeaturePlannerAgent(
        prompt_loader=PromptLoader(),
        llm_client=repaired_client,
        contract_generator=MockContractCodeGenerator(),
    )

    _, _, repaired = await repaired_planner.plan(
        feature_id="feature-plan-safe-schema-error",
        technical_prd=_technical_prd(),
        repositories=_repositories(),
    )

    # The rejected model's own name is kept, because it is declared in this repository's
    # schemas and it is what tells an operator which agent could not produce its artifact.
    # `value_error` carries no context here on purpose: its `ctx["error"]` is whatever a
    # custom validator raised, and this validator rejected the secret by quoting it.
    assert repaired.metadata["repair_reason"] == (
        "ArchitectureArtifact: schema validation rejected a value [value_error]"
    )
    assert secret not in json.dumps(repaired.metadata)

    failing_client = QueuedLLMClient([invalid_response, invalid_response])
    failing_planner = FeaturePlannerAgent(
        prompt_loader=PromptLoader(),
        llm_client=failing_client,
        contract_generator=MockContractCodeGenerator(),
    )

    with pytest.raises(AgentArtifactError) as raised:
        await failing_planner.plan(
            feature_id="feature-plan-safe-schema-terminal-error",
            technical_prd=_technical_prd(),
            repositories=_repositories(),
        )

    assert all(secret not in item for item in raised.value.diagnostics)
    assert all("[value_error]" in item for item in raised.value.diagnostics)


@pytest.mark.asyncio
async def test_feature_planner_discards_model_proposed_serial_execution() -> None:
    """A repository consuming the approved contract never waits for the one serving it.

    The submitted plan declares frontend-after-backend, exactly the shape that starved the
    frontend in production. Scheduling is recomputed rather than repaired, so one model
    round trip still yields a single parallel group.
    """
    serial_plan = _execution_plan(
        frontend_reference={
            "requirement_id": "server-status-contract",
            "acceptance_criterion_ids": ["server-status-contract:ac-1"],
            "responsibility": "implements",
        }
    )
    client = QueuedLLMClient([_planning_response(serial_plan)])
    planner = FeaturePlannerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        contract_generator=MockContractCodeGenerator(),
    )

    _, _, plan = await planner.plan(
        feature_id="feature-plan-parallel",
        technical_prd=_technical_prd(),
        repositories=_repositories(),
    )

    assert len(client.calls) == 1
    assert plan.parallel_groups == [["backend", "frontend"]]
    assert all(not item.dependency_workstream_ids for item in plan.workstreams)
    assert plan.metadata["dropped_dependency_workstream_ids"] == {"frontend": ["backend"]}
    # Operator sequencing survives as merge guidance rather than an execution barrier.
    assert plan.recommended_merge_order == ["backend", "frontend"]


@pytest.mark.asyncio
async def test_deterministic_feature_planner_does_not_invent_language_or_layout_areas() -> None:
    """Planning repository links before clone uses generic production evidence only."""
    _, _, plan = await DeterministicFeaturePlanner().plan(
        feature_id="portable-plan", technical_prd=_technical_prd(), repositories=_repositories()
    )

    for workstream in plan.workstreams:
        expectation = workstream.implementation_expectations[0]
        # `integration` is added by the workstream contract: production code that nothing
        # already running refers to is not a finished requirement.
        assert expectation.expected_change_categories == ["production", "test", "integration"]
        assert expectation.expected_source_areas == []


def _technical_prd() -> TechnicalPRDArtifact:
    """Create the one shared requirement needed for plan-ownership validation."""
    return TechnicalPRDArtifact(
        schema_version="1.0",
        workflow_id="feature-plan-repair",
        artifact_id="002_technical_prd.json",
        producer="product_manager",
        timestamp=datetime.now(UTC),
        metadata={},
        validation_status="valid",
        title="Shared API contract",
        solution_summary="A backend API is consumed by a frontend.",
        functional_requirements=[
            Requirement(
                requirement_id="server-status-contract",
                description="Both repositories use one server-status contract.",
                priority="must",
                acceptance_criteria=["Both sides use the same contract."],
                dependencies=[],
            )
        ],
        non_functional_requirements=[],
        data_requirements=[],
        integration_requirements=[],
        security_requirements=[],
        assumptions=[],
        unresolved_questions=[],
    )


def _repositories() -> list[RepositorySpec]:
    """Return the ordered backend-first workstreams used by the repair case."""
    return [
        RepositorySpec(
            repository_id="backend",
            name="Backend",
            role=WorkstreamRole.BACKEND,
            repository_url=HttpUrl("https://example.com/backend.git"),
            default_branch="main",
            implementation_order=0,
        ),
        RepositorySpec(
            repository_id="frontend",
            name="Frontend",
            role=WorkstreamRole.FRONTEND,
            repository_url=HttpUrl("https://example.com/frontend.git"),
            default_branch="main",
            implementation_order=1,
        ),
    ]


def _planning_response(plan: dict[str, Any]) -> dict[str, Any]:
    """Return the complete shape the live planner requires from the model."""
    return {
        "architecture": {
            "system_overview": "A backend API is consumed by a frontend.",
            "technology_stack": {"backend": "Node.js", "frontend": "React"},
            "repository_structure": ["backend", "frontend"],
            "components": [
                {
                    "component_id": "api",
                    "name": "Server status API",
                    "responsibility": "Return status.",
                    "technology": "Node.js",
                    "dependencies": [],
                }
            ],
            "api_contracts": [],
            "data_entities": [],
            "decisions": [
                {
                    "decision_id": "backend-first",
                    "title": "Implement the provider first",
                    "decision": "The backend owns the shared API implementation.",
                    "rationale": "The frontend consumes the backend contract.",
                    "consequences": ["The frontend waits for the backend workstream."],
                }
            ],
            "risks": [],
            "deployment_strategy": "Deploy backend before frontend.",
        },
        "integration_contract": {
            "contract_version": "1.0.0",
            "api_style": "none",
            "endpoints": [],
            "shared_schemas": [],
            "authentication_contract": None,
            "authorization_rules": [],
            "error_contracts": [],
            "event_contracts": [],
            "environment_variables": [],
            "compatibility_policy": {
                "policy": "Additive changes only.",
                "breaking_change_allowed": False,
                "migration_requirements": [],
                "rollback_requirements": ["Revert both sides together."],
            },
            "owning_workstreams": ["backend"],
        },
        "repository_execution_plan": plan,
    }


def _execution_plan(frontend_reference: dict[str, str | list[str]]) -> dict[str, Any]:
    """Build a plan where the frontend reference can be malformed or repaired."""
    backend_reference: dict[str, str | list[str]] = {
        "requirement_id": "server-status-contract",
        "acceptance_criterion_ids": ["server-status-contract:ac-1"],
        "responsibility": "implements",
    }
    return {
        "workstreams": [
            _workstream(
                workstream_id="backend",
                repository_id="backend",
                requirement_ids=["server-status-contract"],
                scoped_requirements=[backend_reference],
                shared_requirements=[],
                dependencies=[],
            ),
            _workstream(
                workstream_id="frontend",
                repository_id="frontend",
                requirement_ids=[str(frontend_reference["requirement_id"])],
                scoped_requirements=[frontend_reference],
                shared_requirements=[frontend_reference],
                dependencies=["backend"],
            ),
        ],
        "execution_order": ["backend", "frontend"],
        "parallel_groups": [["backend"], ["frontend"]],
        "integration_test_plan": ["Exercise the shared API contract."],
        "merge_strategy": "backend_first",
        "deployment_strategy": "backend_first",
        "feature_flag_strategy": [],
        "rollback_strategy": ["Revert the feature."],
    }


def _workstream(
    *,
    workstream_id: str,
    repository_id: str,
    requirement_ids: list[str],
    scoped_requirements: list[dict[str, str | list[str]]],
    shared_requirements: list[dict[str, str | list[str]]],
    dependencies: list[str],
) -> dict[str, Any]:
    """Return a schema-valid workstream with the supplied requirement references."""
    return {
        "workstream_id": workstream_id,
        "repository_id": repository_id,
        "role": repository_id,
        "requirement_ids": requirement_ids,
        "scoped_requirements": scoped_requirements,
        "out_of_scope_requirements": [],
        "shared_requirements": shared_requirements,
        "responsibilities": [f"Deliver {repository_id} work."],
        "task_ids": [f"{repository_id}-task"],
        "dependency_workstream_ids": dependencies,
        "contract_sections_consumed": [],
        "contract_sections_implemented": [],
        "acceptance_criteria": [f"{repository_id} is reviewed."],
        "test_requirements": [f"Run {repository_id} tests."],
        "documentation_requirements": [],
        "expected_files_or_areas": [],
        "required": True,
    }


def _contract_section_plan(*, implemented: list[str], consumed: list[str]) -> dict[str, Any]:
    """Build a two-workstream plan whose contract references the test controls."""
    reference: dict[str, str | list[str]] = {
        "requirement_id": "server-status-contract",
        "acceptance_criterion_ids": ["server-status-contract:ac-1"],
        "responsibility": "implements",
    }
    plan = _execution_plan(reference)
    plan["workstreams"][0]["contract_sections_implemented"] = implemented
    plan["workstreams"][1]["contract_sections_consumed"] = consumed
    return plan


def _planning_response_with_contract(
    plan: dict[str, Any], *, operation_ids: list[str]
) -> dict[str, Any]:
    """Return a planning response whose contract actually defines the named endpoints."""
    response = _planning_response(plan)
    response["integration_contract"]["api_style"] = "rest"
    response["integration_contract"]["endpoints"] = [
        {
            "operation_id": operation_id,
            "method": "GET",
            "path": f"/{operation_id}",
            "summary": "Return status.",
            "request_schema": None,
            "response_schema": {"type": "object"},
            "query_parameters": [],
            "path_parameters": [],
            "headers": [],
            "authentication_required": False,
            "authorization_requirements": [],
            "success_status_codes": [200],
            "error_codes": [],
            "version": "1.0.0",
        }
        for operation_id in operation_ids
    ]
    return response


@pytest.mark.asyncio
async def test_a_workstream_may_not_be_told_to_build_a_contract_section_that_does_not_exist() -> (
    None
):
    """These two fields were the one part of the plan nothing checked.

    They do not stay in the artifact. The runtime puts them verbatim into the engineer's task
    -- "Implement sections: X. Consume sections: Y." -- and into the scope the reviewer judges
    against. Written in the same response that invents the contract, an operation ID that
    exists nowhere became an instruction to build an endpoint that is not in the contract, and
    an acceptance criterion no diff could ever satisfy.
    """
    invalid = _contract_section_plan(
        implemented=["getServerStatus", "getSomethingNobodyDefined"], consumed=["getServerStatus"]
    )
    repaired = _contract_section_plan(implemented=["getServerStatus"], consumed=["getServerStatus"])
    client = QueuedLLMClient(
        [
            _planning_response_with_contract(invalid, operation_ids=["getServerStatus"]),
            _planning_response_with_contract(repaired, operation_ids=["getServerStatus"]),
        ]
    )
    planner = FeaturePlannerAgent(
        llm_client=client,
        prompt_loader=PromptLoader(),
        contract_generator=MockContractCodeGenerator(),
    )

    _architecture, contract, plan = await planner.plan(
        feature_id="feature-plan-repair",
        technical_prd=_technical_prd(),
        repositories=_repositories(),
    )

    # Repaired rather than shipped, through the bounded path that already exists.
    assert len(client.calls) == 2
    defined = {endpoint.operation_id for endpoint in contract.endpoints}
    named = {
        section
        for workstream in plan.workstreams
        for section in [
            *workstream.contract_sections_implemented,
            *workstream.contract_sections_consumed,
        ]
    }
    assert named <= defined, "the engineer must only be sent sections the contract defines"
    # The repair was told exactly which name was wrong, not merely that something was.
    assert "getSomethingNobodyDefined" in client.calls[1][0]


@pytest.mark.asyncio
async def test_contract_sections_naming_schemas_events_and_errors_are_accepted() -> None:
    """A section is not only an endpoint: schemas, events, error codes and env vars count."""
    plan = _contract_section_plan(implemented=["StatusPayload"], consumed=["STATUS_UNAVAILABLE"])
    response = _planning_response_with_contract(plan, operation_ids=["getServerStatus"])
    response["integration_contract"]["shared_schemas"] = [
        {
            "name": "StatusPayload",
            "description": "The server status body.",
            "json_schema": {"type": "object"},
        }
    ]
    response["integration_contract"]["error_contracts"] = [
        {
            "code": "STATUS_UNAVAILABLE",
            "status_code": 503,
            "description": "The status could not be read.",
            "schema_definition": None,
        }
    ]
    client = QueuedLLMClient([response])
    planner = FeaturePlannerAgent(
        llm_client=client,
        prompt_loader=PromptLoader(),
        contract_generator=MockContractCodeGenerator(),
    )

    _architecture, _contract, plan_artifact = await planner.plan(
        feature_id="feature-plan-repair",
        technical_prd=_technical_prd(),
        repositories=_repositories(),
    )

    assert len(client.calls) == 1, "a valid plan must not be sent through the repair path"
    assert plan_artifact.workstreams[0].contract_sections_implemented == ["StatusPayload"]
    assert plan_artifact.workstreams[1].contract_sections_consumed == ["STATUS_UNAVAILABLE"]
