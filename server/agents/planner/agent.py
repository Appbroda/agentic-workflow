"""Artifact-producing technical-planning node."""

from __future__ import annotations

import json
from typing import Any

from adapters.llm_adapter import LLMClient
from agents.shared.contracts import (
    ARTIFACT_FILENAMES,
    AgentArtifactError,
    artifact_json,
    artifact_list_json,
    artifact_update,
    create_artifact,
    execution_metadata,
    parse_model_json,
    require_artifact,
)
from artifacts.schemas import (
    ArchitectureArtifact,
    ExecutionGraphArtifact,
    TaskPlanArtifact,
    TechnicalPRDArtifact,
)
from prompts.prompt_loader import PromptLoader
from state.models import AgentState


class PlannerAgent:
    """Create traceable architecture, execution-graph, and task-plan artifacts."""

    def __init__(self, *, prompt_loader: PromptLoader, llm_client: LLMClient) -> None:
        """Inject the versioned planner prompt and configured reasoning-model boundary."""
        self._prompt_loader = prompt_loader
        self._llm_client = llm_client

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Publish planning artifacts without exchanging natural-language agent messages."""
        technical_prd = require_artifact(
            state,
            TechnicalPRDArtifact,
            artifact_id=ARTIFACT_FILENAMES["technical_prd"],
        )
        clarification_answers = technical_prd.metadata.get("clarification_answers", {})
        if not isinstance(clarification_answers, dict):
            msg = "technical PRD clarification_answers metadata must be an object"
            raise AgentArtifactError(msg)
        instructions = self._prompt_loader.render(
            "planner/v1.jinja2",
            workflow_id=state["workflow_id"],
            technical_prd=artifact_json(technical_prd),
            clarification_answers=json.dumps(clarification_answers, indent=2, sort_keys=True),
            available_artifacts=artifact_list_json(state["artifacts"]),
        )
        response = await self._llm_client.respond(
            instructions=instructions,
            input_text=json.dumps(
                {
                    "technical_prd": technical_prd.model_dump(mode="json"),
                    "clarification_answers": clarification_answers,
                },
                indent=2,
                sort_keys=True,
            ),
        )
        payload = parse_model_json(
            response.output_text,
            expected_keys=("architecture", "execution_graph", "task_plan"),
        )
        planning_metadata = {
            "source_artifact_ids": [technical_prd.artifact_id],
            "model": response.model,
            "response_id": response.response_id,
            "prompt_template": "planner/v1.jinja2",
            **execution_metadata(
                agent_type="Technical planner",
                provider=response.provider,
                model=response.model,
                reasoning_effort=response.reasoning_effort,
                model_role=response.model_role,
                model_variable=response.model_variable,
                routing_reason=response.routing_reason,
            ),
        }
        architecture = create_artifact(
            ArchitectureArtifact,
            workflow_id=state["workflow_id"],
            artifact_id=ARTIFACT_FILENAMES["architecture"],
            producer="planner",
            payload=_artifact_payload(payload, "architecture"),
            metadata=planning_metadata,
        )
        execution_graph = create_artifact(
            ExecutionGraphArtifact,
            workflow_id=state["workflow_id"],
            artifact_id=ARTIFACT_FILENAMES["execution_graph"],
            producer="planner",
            payload=_artifact_payload(payload, "execution_graph"),
            metadata=planning_metadata,
        )
        task_plan = create_artifact(
            TaskPlanArtifact,
            workflow_id=state["workflow_id"],
            artifact_id=ARTIFACT_FILENAMES["task_plan"],
            producer="planner",
            payload=_artifact_payload(payload, "task_plan"),
            metadata=planning_metadata,
        )
        return artifact_update("planner", [architecture, execution_graph, task_plan])


async def planner_node(
    state: AgentState,
    *,
    prompt_loader: PromptLoader,
    llm_client: LLMClient,
) -> dict[str, Any]:
    """Run a technical-planning node with explicitly injected dependencies."""
    return await PlannerAgent(prompt_loader=prompt_loader, llm_client=llm_client).run(state)


def _artifact_payload(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Return one planner artifact payload after proving it is a JSON object."""
    artifact_payload = payload[key]
    if not isinstance(artifact_payload, dict):
        msg = f"planner response field '{key}' must be a JSON object"
        raise AgentArtifactError(msg)
    return artifact_payload
