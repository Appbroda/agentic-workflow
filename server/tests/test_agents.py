"""Independent, mock-only tests for each artifact-driven workflow agent node."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from adapters.github_adapter import MockGitHubService, PullRequestDetails
from adapters.interruptible_git import reviewed_content_fingerprint
from adapters.llm_adapter import (
    CodingExecutionResult,
    ImageInput,
    LLMAdapterError,
    LLMResponse,
    MockCodingExecutor,
    ResponsesCodingExecutor,
)
from agents.engineer.agent import (
    _MAX_ASSIGNED_CONTEXT_PATHS,
    _MAX_IMPORTED_CONTEXT_PATHS,
    _MAX_RECORDED_IMPORT_DISCARDS,
    _MAX_SOURCE_REPAIR_PASSES,
    _REPOSITORY_CONTEXT_MAX_CHARACTERS,
    _REPOSITORY_CONTEXT_MAX_FILES,
    _REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS,
    EngineerAgent,
    RequiredContextRefusal,
    _assigned_context_paths,
    _assigned_scan_sources,
    _blocking_diagnostics,
    _diagnostic_file_locations,
    _imported_repository_paths,
    _prior_attempt_required_paths,
    _refuse_if_required_context_dropped,
    _relevance_terms,
    _repository_context,
    _required_context_paths,
    repository_snapshot_budget,
)
from agents.engineer.publisher import ApprovedChangePublisher
from agents.github.agent import GitHubAgent
from agents.planner.agent import PlannerAgent
from agents.product_manager.agent import ProductManagerAgent
from agents.reviewer.agent import (
    _REVIEW_EVIDENCE_MAX_CHARACTERS,
    _REVIEW_EVIDENCE_PER_FILE_MAX_CHARACTERS,
    ReviewerAgent,
    _apply_unreviewable_criteria_findings,
    _apply_validation_findings,
    _apply_workspace_evidence_findings,
    _review_input_signature,
    _review_scope,
    _review_technical_prd,
)
from agents.shared.contracts import (
    ARTIFACT_FILENAMES,
    AgentArtifactError,
    artifact_id_matches_lineage,
    attempt_artifact_id,
    safe_error_diagnostics,
)
from artifacts.schemas import (
    ArchitectureArtifact,
    ArchitectureComponent,
    ArchitectureDecision,
    ArchitectureRisk,
    CodeCompletionArtifact,
    ExecutionGraphArtifact,
    ExecutionNode,
    FileChange,
    Milestone,
    PlannedTask,
    PRDArtifact,
    Requirement,
    RequirementCheck,
    ReviewArtifact,
    ReviewFinding,
    TaskPlanArtifact,
    TechnicalPRDArtifact,
    UserStory,
)
from configs.model_roles import ModelRole
from prompts.prompt_loader import PromptLoader
from services.external_operations import ExternalOperationExecutor
from state.failure_diagnosis import FeatureFailureClassification
from state.models import AgentState, WorkspaceDescriptor, create_initial_agent_state
from storage.feature_store import _rejected_stage
from tests.fixtures import commit_all, init_git_repository, start_working_branch
from tools.acceptance_criteria import UnverifiableCriterion
from tools.file_tools import PathLike, WorkspaceFileTools
from tools.model_routing import ModelExecutionMode, ModelRoutingDecision
from tools.reachability import ReachabilityOutcome
from tools.review_fix_classification import ReviewFixComplexity
from tools.source_formatting import SourceValidationError
from tools.validation_tools import (
    ValidationResult,
    ValidationStatus,
    ValidationTool,
)


class StaticLLMClient:
    """An asynchronous LLM protocol double with a JSON response chosen by each test."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._response = LLMResponse(
            response_id="mock-agent-response",
            model="mock-reasoning-model",
            output_text=json.dumps(payload),
            input_tokens=10,
            output_tokens=5,
        )
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
        """Record the model inputs and return a deterministic structured response."""
        self.calls.append((instructions, input_text))
        return self._response


class StaticValidationTool(ValidationTool):
    """A validation protocol double that does not spawn subprocesses in agent unit tests."""

    def __init__(self, *, ruff_result: ValidationResult, pytest_result: ValidationResult) -> None:
        self._ruff_result = ruff_result
        self._pytest_result = pytest_result
        self.calls: list[tuple[str, float]] = []

    def run_ruff(self, *, timeout_seconds: float) -> ValidationResult:
        """Return the configured Ruff outcome."""
        self.calls.append(("ruff", timeout_seconds))
        return self._ruff_result

    def run_pytest(self, *, timeout_seconds: float) -> ValidationResult:
        """Return the configured pytest outcome."""
        self.calls.append(("pytest", timeout_seconds))
        return self._pytest_result


@pytest.mark.asyncio
async def test_product_manager_node_publishes_a_technical_prd(tmp_path: Path) -> None:
    """The product manager consumes only the PRD artifact and returns the next artifact update."""
    state = agent_state(tmp_path, [prd_artifact()])
    llm_client = StaticLLMClient(domain_payload(technical_prd_artifact()))

    update = await ProductManagerAgent(prompt_loader=PromptLoader(), llm_client=llm_client).run(
        state
    )

    technical_prd = update["artifacts"][0]
    assert isinstance(technical_prd, TechnicalPRDArtifact)
    assert technical_prd.artifact_id == ARTIFACT_FILENAMES["technical_prd"]
    assert technical_prd.metadata["source_artifact_ids"] == [ARTIFACT_FILENAMES["prd"]]
    assert update["current_agent"] == "product_manager"
    assert len(llm_client.calls) == 1
    instructions, _ = llm_client.calls[0]
    assert "Return only the TechnicalPRD payload" in instructions
    assert "Do not include platform-managed artifact envelope fields" in instructions
    assert "must contain only plain strings, never objects" in instructions


@pytest.mark.asyncio
async def test_planner_node_publishes_all_three_planning_artifacts(tmp_path: Path) -> None:
    """The planner emits architecture, execution graph, and task plan from the technical PRD."""
    planning_payload = {
        "architecture": domain_payload(architecture_artifact()),
        "execution_graph": domain_payload(execution_graph_artifact()),
        "task_plan": domain_payload(task_plan_artifact()),
    }
    llm_client = StaticLLMClient(planning_payload)

    technical_prd = technical_prd_artifact().model_copy(
        update={
            "metadata": {
                "source": "test",
                "clarification_answers": {"question-1": "Use PostgreSQL."},
            }
        }
    )
    update = await PlannerAgent(prompt_loader=PromptLoader(), llm_client=llm_client).run(
        agent_state(tmp_path, [technical_prd])
    )

    assert [artifact.artifact_id for artifact in update["artifacts"]] == [
        ARTIFACT_FILENAMES["architecture"],
        ARTIFACT_FILENAMES["execution_graph"],
        ARTIFACT_FILENAMES["task_plan"],
    ]
    assert all(artifact.producer == "planner" for artifact in update["artifacts"])
    assert len(llm_client.calls) == 1
    instructions, input_text = llm_client.calls[0]
    assert '"question-1": "Use PostgreSQL."' in instructions
    assert json.loads(input_text)["clarification_answers"] == {"question-1": "Use PostgreSQL."}


def test_feature_planner_prompt_declares_its_strict_artifact_shapes() -> None:
    """The live multi-repository planner must not infer a different JSON vocabulary."""
    instructions = PromptLoader().render(
        "planner/feature_v1.jinja2",
        feature_id="feature-1",
        repositories=[],
        technical_prd="{}",
        has_reconnaissance=False,
        reconnaissance="[]",
    )

    assert "system_overview" in instructions
    assert "repository_execution_plan" in instructions
    assert "Do not rename fields" in instructions
    assert "platform-managed artifact envelope fields" in instructions
    assert "must be a JSON object, never a plain string" in instructions


def test_feature_planner_prompt_only_permits_named_source_areas_once_checkouts_are_read() -> None:
    """The instruction to plan blind must not survive the evidence that replaces it.

    Without reconnaissance the planner is told to leave `expected_source_areas` empty, because
    anything it wrote there would be a guess. With reconnaissance the opposite is required: a
    guess is now the failure, and the evidence is what the plan must be bound to.
    """
    blind = PromptLoader().render(
        "planner/feature_v1.jinja2",
        feature_id="feature-1",
        repositories=[],
        technical_prd="{}",
        has_reconnaissance=False,
        reconnaissance="[]",
    )
    grounded = PromptLoader().render(
        "planner/feature_v1.jinja2",
        feature_id="feature-1",
        repositories=[],
        technical_prd="{}",
        has_reconnaissance=True,
        reconnaissance='[{"repository_id": "backend", "source_areas": ["server/routes"]}]',
    )

    assert "before their contents are checked out" in blind
    assert "contradicted_premises" not in blind
    assert "server/routes" not in blind

    assert "before their contents are checked out" not in grounded
    assert "must name directories that" in grounded
    assert "contradicted_premises" in grounded
    assert "server/routes" in grounded


def test_engineer_prompt_declares_the_coding_executor_response_shape() -> None:
    """The coding model must return files in the adapter's strict JSON format."""
    instructions = PromptLoader().render(
        "engineer/v1.jinja2",
        workflow_id="workflow-1",
        workspace_descriptor={},
        task_plan="{}",
        repository_context="{}",
        execution_context="{}",
        lint_capabilities="{}",
        previous_attempt_diff="",
    )

    assert "top-level keys must be exactly `summary` and `files`" in instructions
    assert "`files` must be a JSON array of objects" in instructions
    assert "complete replacement UTF-8 file content" in instructions


@pytest.mark.asyncio
async def test_repository_reviewer_receives_only_its_assigned_requirement_scope(
    tmp_path: Path,
) -> None:
    """A backend child cannot be rejected for a frontend workstream it does not own."""
    technical_prd = technical_prd_artifact().model_copy(
        update={
            "functional_requirements": [
                requirement("requirement-1"),
                Requirement(
                    requirement_id="frontend-tile",
                    description="Show an Admin Console status tile.",
                    priority="must",
                    acceptance_criteria=["The tile polls and cleans up its timer."],
                    dependencies=[],
                ),
            ]
        }
    )
    task_plan = task_plan_artifact().model_copy(
        update={
            "metadata": {
                "review_scope": {
                    "repository_id": "backend",
                    "workstream_id": "backend",
                    "role": "backend",
                    "requirement_ids": ["requirement-1"],
                    "scoped_requirements": [
                        {
                            "requirement_id": "requirement-1",
                            "acceptance_criterion_ids": ["requirement-1:ac-1"],
                            "responsibility": "implements",
                        }
                    ],
                    "out_of_scope_requirements": ["frontend-tile"],
                    "shared_requirements": [],
                    "responsibilities": ["Implement the status endpoint."],
                    "acceptance_criteria": ["Return the status response."],
                    "test_requirements": ["Run backend tests."],
                    "contract_sections_consumed": [],
                    "contract_sections_implemented": ["getServerStatus"],
                    "expected_files_or_areas": ["app/server_status.py"],
                }
            }
        }
    )
    validation_tool = StaticValidationTool(
        ruff_result=validation_result("ruff", return_code=0),
        pytest_result=validation_result("pytest", return_code=0),
    )
    llm_client = StaticLLMClient(domain_payload(review_artifact()))

    await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=llm_client,
        validation_tool=validation_tool,
    ).run(agent_state(tmp_path, [technical_prd, task_plan, code_completion_artifact()]))

    instructions, input_text = llm_client.calls[0]
    assert '"repository_id": "backend"' in instructions
    assert '"out_of_scope_requirements": [' in input_text
    assert '"frontend-tile"' in input_text
    assert "Admin Console status tile" not in instructions
    assert "Admin Console status tile" not in input_text


@pytest.mark.asyncio
async def test_engineer_node_uses_coding_executor_and_requires_review_on_retries(
    tmp_path: Path,
) -> None:
    """The engineer writes through the executor and refuses a retry with no review artifact."""
    retry_state = agent_state(tmp_path, [task_plan_artifact()])
    retry_state["retry_count"] = 1
    executor = MockCodingExecutor(file_updates={"src/service.py": "value = 1\n"})
    agent = EngineerAgent(prompt_loader=PromptLoader(), coding_executor=executor)

    with pytest.raises(AgentArtifactError, match="prior 007_review.json"):
        await agent.run(retry_state)

    update = await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][0]
    assert isinstance(completion, CodeCompletionArtifact)
    assert completion.artifact_id == attempt_artifact_id(ARTIFACT_FILENAMES["code_completion"], 0)
    assert completion.file_changes[0].path == "src/service.py"
    assert completion.file_changes[0].change_type == "added"
    assert (tmp_path / "src/service.py").read_text(encoding="utf-8") == "value = 1\n"


@pytest.mark.asyncio
async def test_engineer_and_reviewer_artifacts_preserve_attempt_lineage(tmp_path: Path) -> None:
    """A retry emits new IDs and every handoff names the exact attempt it consumed."""
    technical_prd = technical_prd_artifact()
    task_plan = task_plan_artifact()
    passing_validation = StaticValidationTool(
        ruff_result=validation_result("ruff", return_code=0),
        pytest_result=validation_result("pytest", return_code=0),
    )
    review_client = StaticLLMClient(domain_payload(review_artifact()))

    first_state = agent_state(tmp_path, [technical_prd, task_plan])
    first_completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(file_updates={"src/service.py": "value = 1\n"}),
        ).run(first_state)
    )["artifacts"][0]
    first_review = (
        await ReviewerAgent(
            prompt_loader=PromptLoader(),
            llm_client=review_client,
            validation_tool=passing_validation,
        ).run(agent_state(tmp_path, [technical_prd, task_plan, first_completion]))
    )["artifacts"][0]

    retry_state = agent_state(tmp_path, [technical_prd, task_plan, first_completion, first_review])
    retry_state["retry_count"] = 1
    second_completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(file_updates={"src/service.py": "value = 2\n"}),
        ).run(retry_state)
    )["artifacts"][0]
    retry_state["artifacts"].append(second_completion)
    second_review = (
        await ReviewerAgent(
            prompt_loader=PromptLoader(),
            llm_client=review_client,
            validation_tool=passing_validation,
        ).run(retry_state)
    )["artifacts"][0]

    assert first_completion.artifact_id == "006_code_completion.attempt-0.json"
    assert second_completion.artifact_id == "006_code_completion.attempt-1.json"
    assert first_review.artifact_id == "007_review.attempt-0.json"
    assert second_review.artifact_id == "007_review.attempt-1.json"
    assert first_completion.artifact_id != second_completion.artifact_id
    assert first_review.artifact_id != second_review.artifact_id
    assert first_review.artifact_id in second_completion.metadata["source_artifact_ids"]
    assert second_completion.artifact_id in second_review.metadata["source_artifact_ids"]
    assert first_completion.artifact_id not in second_review.metadata["source_artifact_ids"]


@pytest.mark.asyncio
async def test_reviewer_model_receives_bounded_current_workspace_source(tmp_path: Path) -> None:
    """A model cannot approve implementation code that it was never shown."""
    marker = "REVIEWED_IMPLEMENTATION_MARKER"
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={"src/service.py": f"value = '{marker}'\n"}
            ),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))
    validation = StaticValidationTool(
        ruff_result=validation_result("ruff", return_code=0),
        pytest_result=validation_result("pytest", return_code=0),
    )

    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=validation,
    ).run(agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion]))

    model_input = json.loads(client.calls[0][1])
    evidence = model_input["workspace_change_evidence"]
    assert evidence["limitations"] == []
    assert evidence["files"][0]["path"] == "src/service.py"
    assert marker in evidence["files"][0]["content"]
    review = update["artifacts"][0]
    assert review.metadata["reviewed_file_paths"] == ["src/service.py"]
    assert review.metadata["reviewed_content_fingerprint"] == reviewed_content_fingerprint(
        tmp_path, ["src/service.py"]
    )

    class RejectChangedBytes:
        def commit(self, *_args: Any, **_kwargs: Any) -> str:
            raise AssertionError("changed bytes must be rejected before commit")

        def push(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("changed bytes must be rejected before push")

    (tmp_path / "src/service.py").write_text("value = 'changed-after-review'\n", encoding="utf-8")
    with pytest.raises(AgentArtifactError, match="no longer matches"):
        await ApprovedChangePublisher(git_service=RejectChangedBytes()).run(
            agent_state(tmp_path, [completion, review])
        )


@pytest.mark.asyncio
async def test_reviewer_source_evidence_preserves_ordinary_authentication_code(
    tmp_path: Path,
) -> None:
    """A dynamic token reference is source code, not a credential to hide from review."""
    source = (
        "def authenticate(request):\n"
        '    token = request.headers["Authorization"]\n'
        '    return token.removeprefix("Bearer ")\n'
    )
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(file_updates={"src/auth.py": source}),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))

    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    ).run(agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion]))

    evidence = json.loads(client.calls[0][1])["workspace_change_evidence"]
    assert evidence["limitations"] == []
    assert evidence["files"][0]["content"] == source
    assert update["artifacts"][0].verdict == "approved"


@pytest.mark.asyncio
async def test_reviewer_can_inspect_environment_template_source(tmp_path: Path) -> None:
    """A documented `.env.example` is source evidence, not an actual credential file."""
    source = (
        "OPENAI_API_KEY=your-api-key-here\n"
        "DATABASE_URL=postgresql://user:password@localhost/example\n"
    )
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(file_updates={".env.example": source}),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))

    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    ).run(agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion]))

    evidence = json.loads(client.calls[0][1])["workspace_change_evidence"]
    assert evidence["limitations"] == []
    assert evidence["files"][0]["content"] == source
    assert update["artifacts"][0].verdict == "approved"


@pytest.mark.asyncio
async def test_reviewer_can_inspect_explicit_credential_data_template(tmp_path: Path) -> None:
    """Explicit JSON credential templates remain reviewable while actual files are withheld."""
    source = '{"accessToken":"your-token-here"}\n'
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(file_updates={"credentials.example.json": source}),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))

    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    ).run(agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion]))

    evidence = json.loads(client.calls[0][1])["workspace_change_evidence"]
    assert evidence["limitations"] == []
    assert evidence["files"][0]["content"] == source
    assert update["artifacts"][0].verdict == "approved"


@pytest.mark.asyncio
async def test_reviewer_routes_actual_environment_file_to_manual_review(tmp_path: Path) -> None:
    """A permanently withheld deployed env file is terminal rather than repeatedly retried."""
    secret = "sk-proj-ACTUALDEPLOYMENTSECRET123456789"
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={".env.production": f"OPENAI_API_KEY={secret}\n"}
            ),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))

    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    ).run(agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion]))

    assert secret not in client.calls[0][1]
    evidence = json.loads(client.calls[0][1])["workspace_change_evidence"]
    assert evidence["files"][0]["content"] == "[CONTENT WITHHELD]"
    assert "sensitive_path" in evidence["limitations"]
    review = update["artifacts"][0]
    assert review.verdict == "rejected"
    assert review.metadata["manual_review_required"] is True
    assert review.metadata["retryable"] is False


_OVERSIZED_MODULE = "src/catalog/constants.py"
# Every line the same width, so a test can convert a line number into a character count and
# reason about the budget without depending on the shape of the padding.
_OVERSIZED_LINE_COUNT = 900


def _oversized_module_source(changes: dict[int, str] | None = None) -> str:
    """Return a module far past the per-file evidence budget, with these lines replaced.

    Keyed by one-based line number so a test states where its change is and then asserts the
    reviewer was shown exactly there. The baseline is deliberately uniform: a changed line is
    then the only thing in the file that distinguishes one region from another.
    """
    edits = changes or {}
    lines = [
        edits.get(number, f"ENTRY_{number:04d} = 'catalog-entry-value-{number:04d}'")
        for number in range(1, _OVERSIZED_LINE_COUNT + 1)
    ]
    return "\n".join(lines) + "\n"


def _oversized_module_checkout(root: Path, *, committed: dict[int, str] | None = None) -> Path:
    """Build a checkout on a working branch whose baseline holds the oversized module.

    The branch point is what the reviewer's evidence is measured against, so the fixture makes
    one: `main` holds the module as the repository has it, and the attempts happen on a branch
    off that. Committing an attempt's change on the branch afterwards is what reproduces the
    case `git diff HEAD` cannot see.
    """
    init_git_repository(root)
    module = root / _OVERSIZED_MODULE
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(_oversized_module_source(), encoding="utf-8")
    commit_all(root, "checkout baseline")
    start_working_branch(root, "workflow/workflow-1")
    if committed:
        module.write_text(_oversized_module_source(committed), encoding="utf-8")
        commit_all(root, "an earlier approved attempt")
    return root


async def _oversized_module_lineage(root: Path) -> Any:
    """Return the completion of an attempt whose lineage includes the oversized module.

    Two Engineer runs, because one cannot express the case: the first attempt writes the
    module and its completion is what puts the path into the lineage, and the second attempt
    carries that path forward while touching something else entirely. That is the shape run
    186's fix attempts had -- a reviewed file whose change belongs to an earlier attempt.
    """
    first = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={
                    _OVERSIZED_MODULE: (root / _OVERSIZED_MODULE).read_text(encoding="utf-8")
                }
            ),
        ).run(agent_state(root, [task_plan_artifact()]))
    )["artifacts"][0]
    return (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={"src/catalog/report.py": "REPORT = 'catalog report'\n"}
            ),
        ).run(agent_state(root, [task_plan_artifact(), first]))
    )["artifacts"][0]


async def _review_oversized_module(root: Path, completion: Any) -> tuple[dict[str, Any], Any]:
    """Review this completion against the checkout, returning its evidence and its verdict."""
    client = StaticLLMClient(domain_payload(review_artifact()))
    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    ).run(agent_state(root, [technical_prd_artifact(), task_plan_artifact(), completion]))
    evidence = json.loads(client.calls[0][1])["workspace_change_evidence"]
    return evidence, update["artifacts"][0]


async def _oversized_module_evidence(root: Path) -> tuple[dict[str, Any], Any]:
    """Build the lineage and review it in one step, for tests that need no seam between."""
    return await _review_oversized_module(root, await _oversized_module_lineage(root))


def _entry(evidence: dict[str, Any], path: str) -> dict[str, Any]:
    """Return the one evidence entry for this path, failing loudly when it is absent."""
    return next(item for item in evidence["files"] if item["path"] == path)


# Enough resolved packages to put the lockfile past the reviewer's read cap, in the shape npm
# writes: a registry URL and an integrity hash per package is what makes a real one this size.
_LOCKFILE_PACKAGE_COUNT = 4_200


def _oversized_lockfile_source(extra_packages: Sequence[str] = ()) -> str:
    """Return a `package-lock.json` larger than a reviewer is allowed to read."""
    packages: dict[str, Any] = {"": {"name": "service", "version": "1.0.0"}}
    for entry in (
        *(f"generated-package-{index:05d}" for index in range(_LOCKFILE_PACKAGE_COUNT)),
        *extra_packages,
    ):
        packages[f"node_modules/{entry}"] = {
            "version": "1.0.0",
            "resolved": f"https://registry.example.test/{entry}/-/{entry}-1.0.0.tgz",
            "integrity": f"sha512-{'0' * 86}==",
            "license": "MIT",
        }
    return (
        f"{json.dumps({'name': 'service', 'lockfileVersion': 3, 'packages': packages}, indent=2)}\n"
    )


def _oversized_lockfile_checkout(root: Path) -> Path:
    """Build a checkout on a working branch whose baseline holds an unreadable lockfile."""
    init_git_repository(root)
    (root / "package.json").write_text(
        json.dumps({"name": "service", "version": "1.0.0"}, indent=2) + "\n", encoding="utf-8"
    )
    (root / "package-lock.json").write_text(_oversized_lockfile_source(), encoding="utf-8")
    commit_all(root, "checkout baseline")
    start_working_branch(root, "workflow/workflow-1")
    return root


async def _lockfile_attempt(root: Path, *, packages: Sequence[str]) -> Any:
    """Return the completion of an attempt that declared these packages.

    The manifest is what the attempt writes; the lockfile is written straight to disk, because
    that is where a churned lockfile comes from -- the platform's own install runs when a
    manifest changed and rewrites it -- and because no coding executor would accept a file this
    size anyway. `installed_lockfiles` is what then puts the path into the change set.
    """
    (root / "package-lock.json").write_text(_oversized_lockfile_source(packages), encoding="utf-8")
    return (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={
                    "package.json": json.dumps(
                        {
                            "name": "service",
                            "version": "1.0.0",
                            "dependencies": dict.fromkeys(packages, "^1.0.0"),
                        },
                        indent=2,
                    )
                    + "\n"
                }
            ),
        ).run(agent_state(root, [task_plan_artifact()]))
    )["artifacts"][0]


@pytest.mark.asyncio
async def test_reviewer_sees_the_change_in_a_file_too_large_to_show_whole(
    tmp_path: Path,
) -> None:
    """A large file reaches the model as the regions the change touched, not as its head.

    The middle rung of the evidence ladder used to be the whole unified diff against `HEAD`,
    and it could not express this case at all: the change was committed by an earlier approved
    attempt, so `git diff HEAD` reports nothing for the file and the reviewer was handed the
    first 24,000 characters of it with `truncated: true` -- which a deterministic gate is
    required to refuse, on a file no retry can shrink.
    """
    root = _oversized_module_checkout(
        tmp_path / "checkout", committed={410: "CHANGED_BY_THE_EARLIER_ATTEMPT = True"}
    )
    # The precondition, asserted rather than assumed: against `HEAD` this change is invisible,
    # which is what sent run 186 to head truncation three times over.
    against_head = subprocess.run(
        ("git", "diff", "HEAD", "--", _OVERSIZED_MODULE),
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert against_head.stdout == ""

    evidence, review = await _oversized_module_evidence(root)

    entry = _entry(evidence, _OVERSIZED_MODULE)
    assert entry["content_kind"] == "changed_regions"
    assert "CHANGED_BY_THE_EARLIER_ATTEMPT = True" in entry["content"]
    covering = [
        region
        for region in entry["changed_regions"]
        if region["excerpt_lines"]["start"] <= 410 <= region["excerpt_lines"]["end"]
    ]
    assert len(covering) == 1, entry["changed_regions"]
    assert entry["hunks_omitted"] == 0
    # The hash still names the whole file the excerpt came from, not the excerpt.
    assert entry["sha256"] == hashlib.sha256((root / _OVERSIZED_MODULE).read_bytes()).hexdigest()
    # No limitation, so the deterministic gate never runs, so the verdict is the model's.
    assert evidence["limitations"] == []
    assert review.verdict == "approved"
    assert [item.finding_id for item in review.findings] != ["REVIEW_EVIDENCE_INCOMPLETE"]


@pytest.mark.asyncio
async def test_the_review_is_shown_a_file_the_attempt_wrote_and_never_declared(
    tmp_path: Path,
) -> None:
    """AB-Feature-225 was told coverage was missing for a suite sitting in the worktree.

    The review's file universe was the coding executor's self-reported `file_changes` and
    nothing else, so a file the executor wrote and omitted from its own report was invisible to
    every reviewer on this platform -- and the review then reasoned from its absence. 225's
    attempt 0 wrote a second browser suite covering the pending-submit behaviour, declared
    only the first, and was handed a finding saying that behaviour was untested. 216 lost an
    edited authentication middleware the same way, and was blocked for a defect the unread
    file ruled out.

    Git already answers this, and this platform already asks it elsewhere (66-). The fix is to
    ask it here too and union the answer in.
    """
    root = tmp_path / "checkout"
    init_git_repository(root)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "sign_in.py").write_text("FORM = 'baseline'\n", encoding="utf-8")
    commit_all(root, "checkout baseline")
    start_working_branch(root, "workflow/workflow-1")
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={"src/sign_in.py": "FORM = 'themed'\n"}
            ),
        ).run(agent_state(root, [task_plan_artifact()]))
    )["artifacts"][0]
    # Written into the worktree and absent from the completion's `file_changes`, which is the
    # shape an executor's under-report takes: the file is real, untracked, and unclaimed.
    undeclared = root / "src" / "test_sign_in_pending.py"
    undeclared.write_text("PENDING_SUBMIT_IS_COVERED = True\n", encoding="utf-8")
    assert "src/test_sign_in_pending.py" not in [item.path for item in completion.file_changes]

    evidence, review = await _review_oversized_module(root, completion)

    assert review.metadata["paths_the_attempt_did_not_declare"] == ["src/test_sign_in_pending.py"]
    # The model is shown its contents, not merely its name. This is the whole fix.
    assert "PENDING_SUBMIT_IS_COVERED" in _entry(evidence, "src/test_sign_in_pending.py")["content"]
    # And what the platform ATTESTS to is untouched. `reviewed_file_paths` and the fingerprint
    # beside it are the publication contract -- `require_reviewed_workspace_match` recomputes
    # both from the same declared `file_changes`, and the publisher stages exactly that list --
    # so widening what the review reads must not widen what an approved commit contains. Git
    # lists whatever an attempt left in the worktree; committing that unasked would be a worse
    # failure than the one being fixed here.
    assert review.metadata["reviewed_file_paths"] == ["src/sign_in.py"]
    assert review.metadata["reviewed_content_fingerprint"] == reviewed_content_fingerprint(
        root, ["src/sign_in.py"]
    )


@pytest.mark.asyncio
async def test_a_discovered_path_can_never_raise_a_limitation(tmp_path: Path) -> None:
    """Widening what a review reads must not become a new way to fail a change that was fine.

    Every limitation forbids approval, and a discovered path is evidence the review did not
    have at all before -- so anything that cannot be shown cleanly is dropped rather than
    reported. The real-repository tier found the permissive version: a Python checkout with no
    ignore rule for `__pycache__` offered five `.pyc` files, none of which decode, and the
    resulting `REVIEW_EVIDENCE_INCOMPLETE` blocked an approved production-only change.
    """
    root = tmp_path / "checkout"
    init_git_repository(root)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "sign_in.py").write_text("FORM = 'baseline'\n", encoding="utf-8")
    commit_all(root, "checkout baseline")
    start_working_branch(root, "workflow/workflow-1")
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={"src/sign_in.py": "FORM = 'themed'\n"}
            ),
        ).run(agent_state(root, [task_plan_artifact()]))
    )["artifacts"][0]
    # Generated bytecode in a directory the checkout never ignored, and undecodable bytes at a
    # path that is otherwise ordinary source. Neither is something a review can attest to.
    (root / "src" / "__pycache__").mkdir(parents=True, exist_ok=True)
    (root / "src" / "__pycache__" / "sign_in.cpython-312.pyc").write_bytes(b"\x00\x01\x02\xff")
    (root / "src" / "logo.py").write_bytes(b"\xff\xfe\x00binary\x00")

    evidence, review = await _review_oversized_module(root, completion)

    assert evidence["limitations"] == []
    assert review.verdict == "approved"
    assert review.metadata["paths_the_attempt_did_not_declare"] == []
    assert "src/__pycache__/sign_in.cpython-312.pyc" not in [
        item["path"] for item in evidence["files"]
    ]
    assert "src/logo.py" not in [item["path"] for item in evidence["files"]]


@pytest.mark.asyncio
async def test_the_contract_projection_is_never_offered_as_an_undeclared_change(
    tmp_path: Path,
) -> None:
    """The platform writes it into every workspace, so it is untracked in all of them.

    Offering it would hand every review a permanently-undeclared production file the engineer
    is forbidden to touch -- a finding that can never be resolved, which is the exact defect
    the prompt's own openapi rule exists to prevent.
    """
    root = tmp_path / "checkout"
    init_git_repository(root)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "sign_in.py").write_text("FORM = 'baseline'\n", encoding="utf-8")
    commit_all(root, "checkout baseline")
    start_working_branch(root, "workflow/workflow-1")
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={"src/sign_in.py": "FORM = 'themed'\n"}
            ),
        ).run(agent_state(root, [task_plan_artifact()]))
    )["artifacts"][0]
    (root / "openapi.yaml").write_text("openapi: 3.1.0\n", encoding="utf-8")

    _evidence, review = await _review_oversized_module(root, completion)

    assert review.metadata["paths_the_attempt_did_not_declare"] == []
    assert "openapi.yaml" not in review.metadata["reviewed_file_paths"]


@pytest.mark.asyncio
async def test_a_change_no_baseline_can_locate_is_refused_rather_than_assumed_absent(
    tmp_path: Path,
) -> None:
    """ "Nothing changed" is only ever believed from a real branch point.

    The one way this whole path could approve source nobody saw. With no default branch to
    branch from, the baseline falls back to `HEAD` -- and against `HEAD`, a change an earlier
    approved attempt committed produces no hunks at all. Read as "unchanged", that would
    approve run 186's constants module having quoted none of it. It has to be read as
    unanswerable instead: the evidence falls back to truncation and the gate refuses, which is
    exactly the behaviour this replaced and is the safe direction to fail in.
    """
    root = _oversized_module_checkout(
        tmp_path / "checkout", committed={410: "CHANGED_BY_THE_EARLIER_ATTEMPT = True"}
    )
    # The branch point becomes unreachable, so `merge-base` has nothing to resolve.
    subprocess.run(
        ("git", "branch", "-D", "main"), cwd=root, capture_output=True, check=True, timeout=30
    )

    evidence, review = await _oversized_module_evidence(root)

    entry = _entry(evidence, _OVERSIZED_MODULE)
    assert entry["content_kind"] == "utf8_source"
    assert entry["truncated"] is True
    assert "truncated_content" in evidence["limitations"]
    assert review.verdict == "changes_requested"


@pytest.mark.asyncio
async def test_a_changed_region_too_large_to_quote_still_refuses_and_names_itself(
    tmp_path: Path,
) -> None:
    """The discipline survives the fix: changed content the review never saw still blocks.

    This is the half of the change that keeps it honest. Excerpting exists so a small change
    to a large file is reviewable, not so an unreviewable change is waved through -- and a
    single contiguous block larger than the whole per-file budget is exactly that. What is
    different from before is the sentence: it names the lines that were hidden instead of
    telling the engineer to split the file they added four lines to.
    """
    giant = _REVIEW_EVIDENCE_PER_FILE_MAX_CHARACTERS + 4_000
    root = _oversized_module_checkout(
        tmp_path / "checkout", committed={410: "GIANT = '" + ("x" * giant) + "'"}
    )

    evidence, review = await _oversized_module_evidence(root)

    entry = _entry(evidence, _OVERSIZED_MODULE)
    assert entry["hunks_omitted"] == 1
    assert entry["changed_regions"] == []
    assert "truncated_content" in evidence["limitations"]
    assert review.verdict == "changes_requested"
    finding = next(
        item for item in review.findings if item.finding_id == "REVIEW_EVIDENCE_INCOMPLETE"
    )
    # The hidden lines, named. This is the sentence the next attempt is handed.
    assert f"{_OVERSIZED_MODULE} lines 410-410" in finding.description
    # The unmeetable instruction is gone. It asked run 186 three times to break up a shared
    # constants module because four lines had been added to it.
    assert "into smaller modules" not in finding.recommendation
    assert "smaller" in finding.recommendation


@pytest.mark.asyncio
async def test_review_evidence_is_a_pure_function_of_the_workspace_bytes(
    tmp_path: Path,
) -> None:
    """Two builds over identical bytes are byte-identical; one changed line is not.

    `_review_input_signature` is the idempotency key for the journaled `RUN_REVIEWER`
    operation. Evidence that varied between builds over the same checkout would make crash
    recovery re-call the review model instead of reusing the journalled answer, so hunk
    ordering, window bounds and the rendered headings all have to be deterministic.
    """
    root = _oversized_module_checkout(tmp_path / "checkout", committed={410: "FIRST_CHANGE = True"})
    completion = await _oversized_module_lineage(root)

    first, _review = await _review_oversized_module(root, completion)
    second, _again = await _review_oversized_module(root, completion)

    assert _review_input_signature(first) == _review_input_signature(second)
    assert first == second

    (root / _OVERSIZED_MODULE).write_text(
        _oversized_module_source({410: "FIRST_CHANGE = True", 411: "SECOND_CHANGE = True"}),
        encoding="utf-8",
    )
    changed, _third = await _review_oversized_module(root, completion)

    assert _review_input_signature(changed) != _review_input_signature(first)


@pytest.mark.asyncio
async def test_generated_lockfile_evidence_is_a_pure_function_of_the_workspace_bytes(
    tmp_path: Path,
) -> None:
    """53- A5. Two builds of a lockfile entry over identical bytes are byte-identical.

    Same rule as the excerpting path above, and the same reason: `_review_input_signature` is
    the idempotency key for the journaled `RUN_REVIEWER` operation, so evidence that varied
    between builds over one checkout would make crash recovery re-call the review model
    instead of reusing the journalled answer. The lockfile entry is a natural place to have
    got this wrong -- a size and a hash are stable, but a timestamp, an install ordering or a
    `mtime` would not have been, and none of them are on the entry.
    """
    root = _oversized_lockfile_checkout(tmp_path / "checkout")
    completion = await _lockfile_attempt(root, packages=("declared-dependency",))

    first, _review = await _review_oversized_module(root, completion)
    second, _again = await _review_oversized_module(root, completion)

    entry = _entry(first, "package-lock.json")
    assert entry["content_kind"] == "generated_lockfile"
    assert _review_input_signature(first) == _review_input_signature(second)
    assert first == second

    # And it is a function of the *bytes*: one more resolved package changes the signature,
    # which is what stops a stale journalled review being reused for a different lockfile.
    (root / "package-lock.json").write_text(
        _oversized_lockfile_source(("declared-dependency", "second-declared-dependency")),
        encoding="utf-8",
    )
    changed, _third = await _review_oversized_module(root, completion)

    assert _review_input_signature(changed) != _review_input_signature(first)
    assert _entry(changed, "package-lock.json")["sha256"] != entry["sha256"]


@pytest.mark.asyncio
async def test_a_substituted_registry_host_in_an_unread_lockfile_reaches_the_review(
    tmp_path: Path,
) -> None:
    """Item 19. One resolution repointed at another registry, in a file nobody reads.

    53- made this lockfile reviewable as a size, a line count and a digest, and said plainly
    that those facts are not a defence: a substituted registry host changes the digest exactly
    as a legitimate install does, so the entry was equally consistent with either. The scan is
    the fact that separates them, and this is the shape it exists for -- no manifest edit
    beyond the one that legitimately declared a dependency, one line changed out of thousands,
    and a file past the read cap.

    It arrives as a fact and only a fact. No limitation is raised, so no deterministic gate
    fires and the verdict stays the review model's own: whether a new registry is a mirror
    migration or an attack is a judgement, and the platform is not the one making it.
    """
    root = _oversized_lockfile_checkout(tmp_path / "checkout")
    completion = await _lockfile_attempt(root, packages=("declared-dependency",))
    lockfile = root / "package-lock.json"
    lockfile.write_text(
        lockfile.read_text(encoding="utf-8").replace(
            "https://registry.example.test/declared-dependency/",
            "https://registry.substituted.test/declared-dependency/",
            1,
        ),
        encoding="utf-8",
    )

    evidence, review = await _review_oversized_module(root, completion)

    entry = _entry(evidence, "package-lock.json")
    assert entry["content_kind"] == "generated_lockfile"
    assert entry["resolved_host_scan"] == {
        "format": "npm",
        "scanned": True,
        "reason": None,
        "baseline_present": True,
        "distinct_hosts": 2,
        "new_hosts": ["registry.substituted.test"],
        "hosts_truncated": False,
    }
    # And it is in the narrative the model reads beside the entry, not only in a field.
    assert "registry.substituted.test" in entry["content"]
    assert evidence["limitations"] == []
    assert evidence["manual_review_required"] is False
    assert review.verdict == "approved"


@pytest.mark.asyncio
async def test_a_lockfile_that_grew_on_the_registry_it_already_used_reports_no_new_host(
    tmp_path: Path,
) -> None:
    """The false-alarm case: an ordinary install adds packages, not hosts.

    Asserted because it is the failure that would make the field worthless. If adding declared
    dependencies read as a new-host finding, every dependency-affecting change in this
    platform's history would carry one, and a reviewer would learn to skip the field.
    """
    root = _oversized_lockfile_checkout(tmp_path / "checkout")
    completion = await _lockfile_attempt(root, packages=("declared-dependency",))

    evidence, _review = await _review_oversized_module(root, completion)

    scan = _entry(evidence, "package-lock.json")["resolved_host_scan"]
    assert scan["scanned"] is True
    assert scan["distinct_hosts"] == 1
    assert scan["new_hosts"] == []
    assert evidence["limitations"] == []


@pytest.mark.asyncio
async def test_more_changed_regions_than_the_budget_holds_are_counted_not_dropped(
    tmp_path: Path,
) -> None:
    """A budget that cannot hold every region says so, and spends itself on the changed lines.

    Two orderings are being asserted, and both are the entry's stated policy rather than an
    accident of arithmetic: earlier regions before later ones, and -- the one that matters --
    a changed line being visible at all before any context around an earlier change. A
    selector that gave the first few hunks their full window and hid the rest would be a
    smaller version of the failure this replaced.
    """
    # Spaced wider than the window radius so no two of them merge, and each changed line long
    # enough that the changed lines *alone* overrun the per-file budget -- derived from that
    # budget rather than restated, so raising it cannot quietly stop this test reaching the
    # over-budget path it exists for.
    changed_lines = list(range(20, _OVERSIZED_LINE_COUNT - 20, 70))
    padding = "y" * (_REVIEW_EVIDENCE_PER_FILE_MAX_CHARACTERS // 8)
    root = _oversized_module_checkout(
        tmp_path / "checkout",
        committed={number: f"CHANGED_AT_{number:04d} = '{padding}'" for number in changed_lines},
    )

    evidence, review = await _oversized_module_evidence(root)

    entry = _entry(evidence, _OVERSIZED_MODULE)
    shown = [region["excerpt_lines"] for region in entry["changed_regions"]]
    assert entry["region_order"] == "changed-hunks-in-file-order-with-surrounding-context"
    assert shown == sorted(shown, key=lambda item: item["start"])
    # Every changed line the budget could hold is quoted, and the ones it could not are
    # counted rather than silently absent.
    quoted = [
        number
        for number in changed_lines
        if any(item["start"] <= number <= item["end"] for item in shown)
    ]
    assert entry["hunks_omitted"] == len(changed_lines) - len(quoted)
    assert entry["hunks_omitted"] > 0
    assert quoted == changed_lines[: len(quoted)]
    # Context was what got shed, and it was shed from the later regions first: the earliest
    # region carries its surroundings and the last one is down to the changed line itself.
    assert shown[0]["end"] - shown[0]["start"] > shown[-1]["end"] - shown[-1]["start"]
    # And it still refuses, because some changed lines genuinely were not shown.
    assert "truncated_content" in evidence["limitations"]
    assert review.verdict == "changes_requested"


@pytest.mark.asyncio
async def test_total_source_evidence_over_budget_earns_a_retry_that_can_fix_it(
    tmp_path: Path,
) -> None:
    """A change too large for the evidence budget is a fact about our budget, not the code.

    This test previously asserted the opposite -- that an over-budget review is terminal and
    must not consume the coding retry budget. AB-Feature-168 is why that is wrong. Two files
    2% and 5% over the per-file limit stopped both repositories at `retry_count` 0 with all
    eight retries unspent, on an attempt whose real defect was thirteen failing tests.

    Nothing about the change is approved: the verdict is `changes_requested` and the review
    genuinely did not see all the source. But the attempt is told which files were too long
    and to split them, which is something the next attempt can actually do. The retry budget
    and the repeated-diagnostic ceiling still bound the loop if it never manages it.
    """
    # Each file comfortably under the per-file limit, and enough of them to exhaust the total
    # budget. Both sizes are derived from the limits rather than restated, so raising either
    # cannot quietly stop this test exercising the over-budget path it exists for.
    _each = _REVIEW_EVIDENCE_PER_FILE_MAX_CHARACTERS - 4_000
    _count = (_REVIEW_EVIDENCE_MAX_CHARACTERS // _each) + 2
    files = {
        f"src/module_{index}.py": f"VALUE_{index} = '{'x' * _each}'\n" for index in range(_count)
    }
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(file_updates=files),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))

    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    ).run(agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion]))

    review = update["artifacts"][0]
    assert review.verdict == "changes_requested"
    assert review.metadata["manual_review_required"] is False
    assert review.metadata["retryable"] is True
    assert (
        "truncated_content"
        in json.loads(client.calls[0][1])["workspace_change_evidence"]["limitations"]
    )
    # Back to the engineer, which is the budget it was reserved spent on the defect it was
    # reserved for -- rather than to a person, which is where this used to go with all of it
    # still unspent. The two assertions above are that decision: `manual_review_required`
    # is what turns the verdict terminal (`rejected`), and it is what the child loop reads
    # as a retry refusal. Nothing else routes on a review.
    assert review.verdict != "rejected"


@pytest.mark.asyncio
async def test_reviewer_withholds_secret_shaped_source_and_cannot_approve_it(
    tmp_path: Path,
) -> None:
    """Credential-like changed source is neither sent to the model nor approved silently."""
    secret = "sk-proj-EXAMPLESECRET123456789"
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={"src/client.py": f'client = OpenAI(api_key="{secret}")\n'}
            ),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))
    validation = StaticValidationTool(
        ruff_result=validation_result("ruff", return_code=0),
        pytest_result=validation_result("pytest", return_code=0),
    )

    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=validation,
    ).run(agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion]))

    assert secret not in client.calls[0][1]
    review = update["artifacts"][0]
    assert review.verdict == "rejected"
    assert review.metadata["manual_review_required"] is True
    assert review.metadata["retryable"] is False
    assert any(item.finding_id == "REVIEW_EVIDENCE_INCOMPLETE" for item in review.findings)


@pytest.mark.asyncio
async def test_reviewer_withholds_aws_credentials_but_preserves_placeholders(
    tmp_path: Path,
) -> None:
    """AWS identifiers and secret keys cannot enter review while obvious examples can."""
    access_key = "AKIA1234567890ABCDEF"
    temporary_access_key = "ASIAFEDCBA0987654321"
    secret_key = "q1W2e3R4t5Y6u7I8o9P0a1S2d3F4g5H6j7K8l9Z0"
    source = (
        f'AWS_ACCESS_KEY_ID = "{access_key}"\n'
        f'aws_access_key_id = "{temporary_access_key}"\n'
        f'aws_secret_access_key = "{secret_key}"\n'
        'example = {"awsAccessKeyId": "AKIAEXAMPLE", '
        '"awsSecretAccessKey": "your-secret-access-key"}\n'
    )
    completion = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(file_updates={"src/aws_client.py": source}),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]
    client = StaticLLMClient(domain_payload(review_artifact()))

    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
    ).run(agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion]))

    model_input = client.calls[0][1]
    assert access_key not in model_input
    assert temporary_access_key not in model_input
    assert secret_key not in model_input
    evidence = json.loads(model_input)["workspace_change_evidence"]
    assert "AKIAEXAMPLE" in evidence["files"][0]["content"]
    assert "your-secret-access-key" in evidence["files"][0]["content"]
    assert "redacted_content" in evidence["limitations"]
    review = update["artifacts"][0]
    assert review.verdict == "rejected"
    assert review.metadata["manual_review_required"] is True
    assert review.metadata["retryable"] is False


@pytest.mark.asyncio
async def test_retry_review_and_publication_keep_all_prior_dirty_workspace_files(
    tmp_path: Path,
) -> None:
    """Attempt 1 repairing A must not make attempt 0's still-dirty B disappear from commit."""

    class CommitSpy:
        def __init__(self) -> None:
            self.committed_files: list[str] = []

        def commit(self, *_args: Any, **kwargs: Any) -> str:
            self.committed_files = list(kwargs["files"])
            return "cumulative-commit"

        def push(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    first = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(
                file_updates={"src/a.py": "A = 0\n", "src/b.py": "B = 0\n"}
            ),
        ).run(agent_state(tmp_path, [task_plan_artifact()]))
    )["artifacts"][0]
    rejected = review_artifact().model_copy(
        update={"artifact_id": "007_review.attempt-0.json", "verdict": "changes_requested"}
    )
    retry_state = agent_state(tmp_path, [task_plan_artifact(), first, rejected])
    retry_state["retry_count"] = 1
    second = (
        await EngineerAgent(
            prompt_loader=PromptLoader(),
            coding_executor=MockCodingExecutor(file_updates={"src/a.py": "A = 1\n"}),
        ).run(retry_state)
    )["artifacts"][0]
    assert [item.path for item in second.file_changes] == ["src/a.py", "src/b.py"]

    approved = review_artifact().model_copy(
        update={"artifact_id": "007_review.attempt-1.json", "verdict": "approved"}
    )
    git = CommitSpy()
    published = await ApprovedChangePublisher(git_service=git).run(
        agent_state(tmp_path, [first, rejected, second, approved])
    )

    assert git.committed_files == ["src/a.py", "src/b.py"]
    assert [item.path for item in published["artifacts"][0].file_changes] == [
        "src/a.py",
        "src/b.py",
    ]


@pytest.mark.asyncio
async def test_engineer_does_not_commit_when_precommit_source_validation_fails(
    tmp_path: Path,
) -> None:
    """A lint defect must be retried before Git can hand it to Husky."""

    class RejectingFormatter:
        async def format_paths(
            self, repository_root: PathLike, paths: Sequence[str]
        ) -> tuple[str, ...]:
            del repository_root, paths
            raise SourceValidationError(
                ["Pre-commit source validation `eslint src/service.py` exited with code 1"]
            )

    class CommitSpy:
        def __init__(self) -> None:
            self.commit_calls = 0

        def commit(self, *_args: Any, **_kwargs: Any) -> str:
            self.commit_calls += 1
            return "should-not-be-created"

        def push(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("push must not run after failed source validation")

    git = CommitSpy()
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(file_updates={"src/service.py": "value = 1\n"}),
        git_service_factory=lambda _default_branch: git,
        source_formatter=RejectingFormatter(),
    )

    with pytest.raises(SourceValidationError, match="pre-commit validation"):
        await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    assert git.commit_calls == 0


@pytest.mark.asyncio
async def test_a_declared_dependency_lockfile_is_committed_only_after_approval(
    tmp_path: Path,
) -> None:
    """Declare, install, review, then commit the manifest and generated lockfile together.

    Each stage was verified alone, which is exactly how the lockfile came to be regenerated
    and then left unstaged: the seam between installing and committing had no test at all.
    """
    (tmp_path / "package.json").write_text('{"name": "web"}\n', encoding="utf-8")
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")

    class InstallingSynchronizer:
        """Stand in for the package manager, rewriting the lockfile as a real install does."""

        def __init__(self) -> None:
            self.synced_paths: list[str] = []

        async def sync(self, repository_root: PathLike, paths: Sequence[str]) -> tuple[str, ...]:
            self.synced_paths = list(paths)
            lockfile = Path(repository_root) / "package-lock.json"
            lockfile.write_text(
                '{"lockfileVersion": 3, "packages": {"node_modules/supertest": {}}}\n',
                encoding="utf-8",
            )
            return ("npm install",)

    class CommitSpy:
        def __init__(self) -> None:
            self.committed_files: list[str] = []
            self.expected_content_fingerprint: str | None = None
            self.expected_push_sha: str | None = None

        def commit(self, *_args: Any, **kwargs: Any) -> str:
            self.committed_files = list(kwargs["files"])
            self.expected_content_fingerprint = kwargs["expected_content_fingerprint"]
            return "commit-sha"

        def push(self, *_args: Any, **kwargs: Any) -> None:
            self.expected_push_sha = kwargs["expected_commit_sha"]
            return None

    synchronizer = InstallingSynchronizer()
    git = CommitSpy()
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(
            file_updates={
                "package.json": '{"name": "web", "devDependencies": {"supertest": "^6.3.4"}}\n',
                "src/service.test.js": "require('supertest');\n",
            }
        ),
        git_service_factory=lambda _default_branch: git,
        dependency_synchronizer=synchronizer,
    )

    update = await agent.run(agent_state(tmp_path, [task_plan_artifact()]))
    completion = update["artifacts"][0]

    # Coding and dependency synchronization are reviewable workspace effects only.
    assert "package.json" in synchronizer.synced_paths
    assert "package-lock.json" not in synchronizer.synced_paths
    assert git.committed_files == []
    assert isinstance(completion, CodeCompletionArtifact)
    assert completion.commit_sha is None

    published = await ApprovedChangePublisher(git_service=git).run(
        agent_state(tmp_path, [completion, review_artifact()])
    )

    assert set(git.committed_files) == {
        "package.json",
        "package-lock.json",
        "src/service.test.js",
    }
    published_completion = published["artifacts"][0]
    assert isinstance(published_completion, CodeCompletionArtifact)
    assert published_completion.artifact_id == (
        f"{completion.artifact_id.removesuffix('.json')}.published.json"
    )
    assert published_completion.commit_sha == "commit-sha"
    assert published_completion.metadata["published_after_approval"] is True
    assert git.expected_content_fingerprint == reviewed_content_fingerprint(
        tmp_path, git.committed_files
    )
    assert git.expected_push_sha == "commit-sha"
    assert completion.metadata["applied_dependency_commands"] == ["npm install"]
    assert "supertest" in (tmp_path / "package-lock.json").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_retry_receives_the_rejected_change_it_is_asked_to_repair(tmp_path: Path) -> None:
    """Diagnostics name files and lines, so the code they describe has to travel with them.

    The workspace is reset before a retry, so the rejected files are gone from the snapshot.
    Without the diff the model is asked to fix code it cannot see and regenerates instead,
    which re-rolls every rule that is not auto-fixable.
    """

    class RecordingCodingExecutor(MockCodingExecutor):
        def __init__(self) -> None:
            super().__init__(file_updates={"src/service.py": "value = 1\n"})
            self.instructions = ""

        async def execute(self, **kwargs: Any) -> Any:
            self.instructions = str(kwargs["instructions"])
            return await super().execute(**kwargs)

    executor = RecordingCodingExecutor()
    diff = (
        "diff --git a/src/service.py b/src/service.py\n"
        "+++ b/src/service.py\n"
        "+value = 1  # trailing comment the linter rejected\n"
    )
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        previous_attempt_diff=diff,
    )

    await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    assert diff in executor.instructions
    assert "Repair it." in executor.instructions
    assert "byte-identical" in executor.instructions
    # Only the files it returns are committed, which is how -023 lost its production file.
    assert "every file that change contained" in executor.instructions


@pytest.mark.asyncio
async def test_a_first_attempt_is_shown_the_files_this_repository_registers_modules_in(
    tmp_path: Path,
) -> None:
    """A registry the task's own words do not name still has to reach the snapshot.

    The snapshot is ranked by how many of the task's terms a path matches, so a registry
    whose name has nothing to do with the feature loses the budget to files that merely
    sound relevant. That is how a bulk-app-delete task never saw the route file its
    validators are mounted in, and the wiring gate spent an attempt delivering a path
    reconnaissance had already established before coding began.
    """

    class RecordingCodingExecutor(MockCodingExecutor):
        def __init__(self) -> None:
            super().__init__(file_updates={"src/service.py": "value = 1\n"})
            self.input_text = ""

        async def execute(self, **kwargs: Any) -> Any:
            self.input_text = str(kwargs["input_text"])
            return await super().execute(**kwargs)

    registry = tmp_path / "src" / "routes" / "adunit.route.py"
    registry.parent.mkdir(parents=True, exist_ok=True)
    # Nothing in this file's name matches the task fixture's wording, which is exactly the
    # case ranking alone gets wrong.
    registry.write_text("ROUTES = ['adunit']\n", encoding="utf-8")
    # Enough better-ranked files to exhaust the snapshot's file budget on their own, so the
    # registry can only appear by being required. Without that the assertion below passes on
    # an empty checkout and proves nothing.
    crowd = tmp_path / "src" / "agents"
    crowd.mkdir(parents=True, exist_ok=True)
    for index in range(_REPOSITORY_CONTEXT_MAX_FILES + 10):
        (crowd / f"agents_workflow_artifact_{index}.py").write_text(
            f"VALUE = {index}\n", encoding="utf-8"
        )
    executor = RecordingCodingExecutor()
    agent = EngineerAgent(prompt_loader=PromptLoader(), coding_executor=executor)
    plan = task_plan_artifact()
    plan = plan.model_copy(
        update={"metadata": {**plan.metadata, "registry_paths": ["src/routes/adunit.route.py"]}}
    )

    await agent.run(agent_state(tmp_path, [plan]))

    snapshot = json.loads(executor.input_text)
    shown = {item["path"] for item in snapshot["repository_context"]["files"]}
    assert "src/routes/adunit.route.py" in shown
    # It is offered as evidence for where to wire, not as a file to change unconditionally.
    assert snapshot["execution_context"]["registry_paths"] == ["src/routes/adunit.route.py"]


@pytest.mark.asyncio
async def test_a_retry_can_see_every_file_its_wiring_repairs_name(tmp_path: Path) -> None:
    """Two repairs put two target files in the snapshot, not just the first.

    Force-inclusion is what makes a repair actionable: the engineer must copy `find` text out
    of the target file verbatim, which it can only do for a file it was shown. Carrying one
    target meant the second diagnostic asked for an edit to an unseen file, and it came back
    identically on the following attempt.
    """

    class RecordingCodingExecutor(MockCodingExecutor):
        def __init__(self) -> None:
            super().__init__(file_updates={"src/service.py": "value = 1\n"})
            self.input_text = ""

        async def execute(self, **kwargs: Any) -> Any:
            self.input_text = str(kwargs["input_text"])
            return await super().execute(**kwargs)

    for path in ("src/routes/adunit.route.py", "src/services/galaxy.service.py"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("REGISTERED = []\n", encoding="utf-8")
    # Crowd the budget so neither target can arrive by ranking alone.
    crowd = tmp_path / "src" / "agents"
    crowd.mkdir(parents=True, exist_ok=True)
    for index in range(_REPOSITORY_CONTEXT_MAX_FILES + 10):
        (crowd / f"agents_workflow_artifact_{index}.py").write_text(
            f"VALUE = {index}\n", encoding="utf-8"
        )
    executor = RecordingCodingExecutor()
    agent = EngineerAgent(prompt_loader=PromptLoader(), coding_executor=executor)
    plan = task_plan_artifact()
    plan = plan.model_copy(
        update={
            "metadata": {
                **plan.metadata,
                "retry_plan": {
                    "attempt": 1,
                    "wiring_repair": {"target_path": "src/routes/adunit.route.py"},
                    "wiring_repairs": [
                        {"target_path": "src/routes/adunit.route.py"},
                        {"target_path": "src/services/galaxy.service.py"},
                    ],
                },
            }
        }
    )

    await agent.run(agent_state(tmp_path, [plan]))

    shown = {
        item["path"] for item in json.loads(executor.input_text)["repository_context"]["files"]
    }
    assert {"src/routes/adunit.route.py", "src/services/galaxy.service.py"} <= shown


@pytest.mark.asyncio
async def test_a_first_attempt_is_not_told_to_repair_anything(tmp_path: Path) -> None:
    """Repair instructions must not leak into an attempt that has nothing to repair."""

    class RecordingCodingExecutor(MockCodingExecutor):
        def __init__(self) -> None:
            super().__init__(file_updates={"src/service.py": "value = 1\n"})
            self.instructions = ""

        async def execute(self, **kwargs: Any) -> Any:
            self.instructions = str(kwargs["instructions"])
            return await super().execute(**kwargs)

    executor = RecordingCodingExecutor()
    agent = EngineerAgent(prompt_loader=PromptLoader(), coding_executor=executor)

    await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    assert "Repair it." not in executor.instructions


def test_repository_context_includes_project_files_but_excludes_credentials(tmp_path: Path) -> None:
    """The coding model receives enough local context without receiving common secret files."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("def status() -> str:\n    return 'ok'\n", encoding="utf-8")
    (tmp_path / ".env").write_text("API_TOKEN=not-for-models\n", encoding="utf-8")

    context = _repository_context(
        {Path("pyproject.toml"), Path("app.py"), Path(".env")}, WorkspaceFileTools(tmp_path)
    )

    assert context["file_inventory"] == ["app.py", "pyproject.toml"]
    assert context["files"] == [
        {"path": "pyproject.toml", "content": "[project]\nname = 'example'\n"},
        {"path": "app.py", "content": "def status() -> str:\n    return 'ok'\n"},
    ]


def test_a_criterion_needing_a_deployed_measurement_is_withheld_from_the_reviewer() -> None:
    """A reviewer reading one diff cannot run a load test, so it must not be asked to."""
    latency = Requirement(
        requirement_id="nfr-response-time",
        description="The status endpoint must respond quickly.",
        priority="must",
        acceptance_criteria=[
            "The endpoint performs only cheap read-only checks.",
            "A staging or production-like load test measures the 95th percentile below 500ms.",
        ],
        dependencies=[],
    )
    technical_prd = technical_prd_artifact().model_copy(
        update={"non_functional_requirements": [latency]}
    )
    scope = {"scope_kind": "repository_workstream", "requirement_ids": ["nfr-response-time"]}

    projected = _review_technical_prd(technical_prd, scope)

    [reviewed] = projected["non_functional_requirements"]
    # The requirement survives and is still judged; only its unreachable proof is withheld.
    assert reviewed["acceptance_criteria"] == ["The endpoint performs only cheap read-only checks."]
    assert reviewed["acceptance_criteria_not_reviewable"] == [
        "A staging or production-like load test measures the 95th percentile below 500ms."
    ]


def test_an_unscoped_review_withholds_the_same_criteria_a_scoped_one_does() -> None:
    """The same criterion must not be unmeetable in one review and enforced in another.

    The full-PRD path handed every criterion straight to the model, so a workflow without a
    repository workstream scope was judged against a load test the scoped path had already
    established no attempt could run.
    """
    latency = Requirement(
        requirement_id="nfr-response-time",
        description="The status endpoint must respond quickly.",
        priority="must",
        acceptance_criteria=[
            "The endpoint performs only cheap read-only checks.",
            "A staging or production-like load test measures the 95th percentile below 500ms.",
        ],
        dependencies=[],
    )
    technical_prd = technical_prd_artifact().model_copy(
        update={"non_functional_requirements": [latency]}
    )

    projected = _review_technical_prd(technical_prd, {"scope_kind": "full_technical_prd"})

    [reviewed] = projected["non_functional_requirements"]
    assert reviewed["acceptance_criteria"] == ["The endpoint performs only cheap read-only checks."]
    assert reviewed["acceptance_criteria_not_reviewable"] == [
        "A staging or production-like load test measures the 95th percentile below 500ms."
    ]


def test_a_withheld_criterion_is_named_in_the_review_without_blocking_it() -> None:
    """Withholding is right; withholding silently is not -- and blocking would be worse.

    Feature -047 spent five attempts rejected for a criterion no attempt could satisfy, which
    is why this finding is `low` and leaves the verdict alone. Its job is to tell whoever
    wrote the requirement that a criterion they wrote was not part of the judgement.
    """
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}

    _apply_unreviewable_criteria_findings(
        payload,
        [
            UnverifiableCriterion(
                requirement_id="nfr-response-time",
                criterion="A staging load test measures the 95th percentile below 500ms.",
            )
        ],
    )

    assert payload["verdict"] == "approved"
    [finding] = payload["findings"]
    assert finding["finding_id"] == "ACCEPTANCE_CRITERIA_NOT_REVIEWED-nfr-response-time"
    assert finding["severity"] == "low"
    assert finding["requirement_id"] == "nfr-response-time"
    assert "95th percentile" in finding["description"]


def test_a_review_with_nothing_withheld_gains_no_finding() -> None:
    """The common case must stay silent, or every review grows noise it did not earn."""
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}

    _apply_unreviewable_criteria_findings(payload, [])

    assert payload["findings"] == []


def test_an_ordinary_criterion_is_never_withheld_from_the_reviewer() -> None:
    """Only the measurement vocabulary is withheld; real scoped work keeps every criterion."""
    monitoring = Requirement(
        requirement_id="feature-monitoring-tile",
        description="Build the monitoring dashboard tile the operators asked for.",
        priority="must",
        acceptance_criteria=[
            "The dashboard tile renders each service's reported state.",
            "An alert banner appears when a service reports unavailable.",
        ],
        dependencies=[],
    )
    technical_prd = technical_prd_artifact().model_copy(
        update={"functional_requirements": [monitoring]}
    )
    scope = {"scope_kind": "repository_workstream", "requirement_ids": ["feature-monitoring-tile"]}

    projected = _review_technical_prd(technical_prd, scope)

    [reviewed] = projected["functional_requirements"]
    assert reviewed["acceptance_criteria"] == monitoring.acceptance_criteria
    assert "acceptance_criteria_not_reviewable" not in reviewed


def test_a_scoped_repair_always_sees_the_file_it_must_edit(tmp_path: Path) -> None:
    """A repair has to quote its target back exactly, so that file cannot miss the budget."""
    # Sorts after the filler below, so ranking alone would leave it out of the budget.
    target = Path("src") / "zwidgets" / "Tiles.js"
    (tmp_path / "src" / "zwidgets").mkdir(parents=True)
    (tmp_path / target).write_text("export default Tiles;\n", encoding="utf-8")
    source_files = {target}
    # Enough unrelated source to fill the budget several times over.
    (tmp_path / "src" / "controllers").mkdir(parents=True)
    for index in range(90):
        other = Path("src") / "controllers" / f"controller_{index:02d}.js"
        (tmp_path / other).write_text(f"// controller {index}\n", encoding="utf-8")
        source_files.add(other)

    unranked = _repository_context(source_files, WorkspaceFileTools(tmp_path))
    required = _repository_context(
        source_files, WorkspaceFileTools(tmp_path), frozenset(), (target.as_posix(),)
    )

    assert target.as_posix() not in [item["path"] for item in unranked["files"]]
    assert target.as_posix() in [item["path"] for item in required["files"]]


def test_repository_context_shows_the_module_the_task_names_over_alphabetical_neighbours(
    tmp_path: Path,
) -> None:
    """The file a task depends on is shown even when the path budget is full of earlier ones."""
    middleware = Path("server") / "middlewares" / "authenticate.js"
    (tmp_path / "server" / "middlewares").mkdir(parents=True)
    (tmp_path / middleware).write_text("module.exports = { requireAdmin };\n", encoding="utf-8")
    # Enough alphabetically earlier source to consume the whole file budget on its own.
    source_files = {middleware}
    (tmp_path / "server" / "controllers").mkdir(parents=True)
    for index in range(90):
        earlier = Path("server") / "controllers" / f"controller_{index:02d}.js"
        (tmp_path / earlier).write_text(f"// controller {index}\n", encoding="utf-8")
        source_files.add(earlier)

    relevance = _relevance_terms(
        '{"summary": "Add an admin route that reuses the existing authenticate middleware."}'
    )
    context = _repository_context(source_files, WorkspaceFileTools(tmp_path), relevance)

    shown = [item["path"] for item in context["files"]]
    assert middleware.as_posix() in shown
    # Without relevance the same budget shows only the alphabetically earlier controllers.
    unranked = _repository_context(source_files, WorkspaceFileTools(tmp_path))
    assert middleware.as_posix() not in [item["path"] for item in unranked["files"]]


def test_repository_context_omits_dependency_trees_and_bounds_its_inventory(tmp_path: Path) -> None:
    """A dependency tree cannot make the coding prompt exceed the provider size limit."""
    source_files = {Path("node_modules") / "package" / "index.js"}
    source_files.update(Path("src") / f"module_{index}.py" for index in range(1_025))

    context = _repository_context(source_files, WorkspaceFileTools(tmp_path))

    assert all("node_modules" not in path for path in context["file_inventory"])
    assert len(context["file_inventory"]) == 1_000
    assert context["omitted_inventory_file_count"] == 25


@pytest.mark.asyncio
async def test_reviewer_node_turns_failed_validation_into_structured_findings(
    tmp_path: Path,
) -> None:
    """Ruff or pytest failures block approval even when the model approves the code."""
    validation_tool = StaticValidationTool(
        ruff_result=validation_result("ruff", return_code=1),
        pytest_result=validation_result("pytest", return_code=0),
    )
    llm_client = StaticLLMClient(domain_payload(review_artifact()))

    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=llm_client,
        validation_tool=validation_tool,
        validation_timeout_seconds=15.0,
    ).run(
        agent_state(
            tmp_path,
            [technical_prd_artifact(), task_plan_artifact(), code_completion_artifact()],
        )
    )

    review = update["artifacts"][0]
    assert isinstance(review, ReviewArtifact)
    assert review.artifact_id == attempt_artifact_id(ARTIFACT_FILENAMES["review"], 0)
    assert review.verdict == "changes_requested"
    # Keyed by validation type as well as executable so that lint, test and build
    # commands sharing one package manager cannot collide into a single finding id.
    assert review.findings[0].finding_id == "validation-custom-ruff"
    assert validation_tool.calls == [("ruff", 15.0), ("pytest", 15.0)]


@pytest.mark.asyncio
async def test_a_rejected_review_is_repaired_instead_of_failing_the_workstream(
    tmp_path: Path,
) -> None:
    """One malformed field must not discard work that already passed every other gate.

    Feature -031's client cleared formatting, linting, tests and the build, then lost the
    whole workstream because the review named a finding category outside the allowed set.
    """

    class RejectThenRepairClient:
        """Return an invalid category first, then a valid review."""

        def __init__(self) -> None:
            self.calls: list[str] = []

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
            del input_text
            self.calls.append(instructions)
            payload = domain_payload(review_artifact())
            if len(self.calls) == 1:
                payload["findings"] = [
                    {
                        "finding_id": "review-1",
                        "severity": "low",
                        "title": "Naming could be clearer.",
                        "description": "The helper name is ambiguous.",
                        "recommendation": "Rename the helper.",
                        "file_path": "src/service.py",
                        "line_number": 1,
                        "requirement_id": "requirement-1",
                        # Not one of the allowed categories: the exact rejection seen live.
                        "finding_category": "performance",
                    }
                ]
            return LLMResponse(
                response_id=f"review-{len(self.calls)}",
                model="mock-reasoning-model",
                output_text=json.dumps(payload),
                input_tokens=1,
                output_tokens=1,
            )

    client = RejectThenRepairClient()

    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=client,
        validation_tool=StaticValidationTool(
            ruff_result=validation_result("ruff", return_code=0),
            pytest_result=validation_result("pytest", return_code=0),
        ),
        validation_timeout_seconds=15.0,
    ).run(
        agent_state(
            tmp_path,
            [technical_prd_artifact(), task_plan_artifact(), code_completion_artifact()],
        )
    )

    assert isinstance(update["artifacts"][0], ReviewArtifact)
    assert len(client.calls) == 2
    repair = client.calls[1]
    assert "Here is your previous response:" in repair
    assert "finding_category" in repair
    assert "byte-identical to your previous response" in repair


def test_reviewer_keeps_baseline_validation_configuration_failure_blocking() -> None:
    """A red baseline is repository health evidence, never a passing required gate."""
    lint_failure = ValidationResult(
        command=("lint-tool", "check", "."),
        return_code=1,
        stdout="",
        stderr="configuration could not resolve shared plugin",
        timed_out=False,
        duration_seconds=0.1,
        validation_type="lint",
        failure_classification="missing_dependency",
    )

    payload: dict[str, Any] = {"verdict": "approved", "findings": []}

    _apply_validation_findings(payload, (lint_failure,))

    assert lint_failure.required is True
    assert payload["verdict"] == "changes_requested"
    assert payload["findings"][0]["finding_category"] == "validation_failure"


@pytest.mark.asyncio
async def test_github_node_uses_mock_service_by_default_and_publishes_pull_request(
    tmp_path: Path,
) -> None:
    """Pull-request publication is mock-backed unless a service is explicitly injected."""
    update = await GitHubAgent(prompt_loader=PromptLoader()).run(
        agent_state(tmp_path, [code_completion_artifact(), review_artifact()])
    )

    pull_request = update["artifacts"][0]
    assert pull_request.artifact_id == ARTIFACT_FILENAMES["pull_request"]
    assert pull_request.repository == "example/platform"
    assert pull_request.labels == ["automated"]
    assert pull_request.metadata["provider"] == MockGitHubService.__name__


@pytest.mark.asyncio
async def test_github_node_refuses_a_pr_whose_fetched_head_is_not_the_approved_commit(
    tmp_path: Path,
) -> None:
    """A successful create response is insufficient when GitHub reports another head SHA."""

    class MismatchedHeadService(MockGitHubService):
        def find_pull_request(self, *args: Any, **kwargs: Any) -> PullRequestDetails | None:
            details = super().find_pull_request(*args, **kwargs)
            assert details is not None
            return PullRequestDetails(
                repository=details.repository,
                number=details.number,
                url=details.url,
                title=details.title,
                source_branch=details.source_branch,
                target_branch=details.target_branch,
                head_sha="different-commit",
            )

    service = MismatchedHeadService()
    agent = GitHubAgent(prompt_loader=PromptLoader(), github_service=service)

    with pytest.raises(AgentArtifactError, match="head does not match"):
        await agent.run(agent_state(tmp_path, [code_completion_artifact(), review_artifact()]))

    assert len(service.pull_requests) == 1
    assert service.labels[("example/platform", 1)] == []
    assert service.reviewers[("example/platform", 1)] == []
    assert service.comments[("example/platform", 1)] == []


@pytest.mark.asyncio
async def test_rejected_review_cannot_reach_commit_or_push(tmp_path: Path) -> None:
    """The publication boundary checks approval before beginning either Git mutation."""

    class RejectSideEffects:
        def commit(self, *_args: Any, **_kwargs: Any) -> str:
            raise AssertionError("commit must not run for rejected work")

        def push(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("push must not run for rejected work")

    completion = code_completion_artifact().model_copy(update={"commit_sha": None})
    rejected = review_artifact().model_copy(update={"verdict": "rejected"})

    with pytest.raises(AgentArtifactError, match="approved review"):
        await ApprovedChangePublisher(git_service=RejectSideEffects()).run(
            agent_state(tmp_path, [completion, rejected])
        )


def agent_state(workspace_root: Path, artifacts: Sequence[Any]) -> AgentState:
    """Create a complete state fixture whose only inter-agent inputs are artifacts."""
    descriptor = WorkspaceDescriptor(
        workspace_id="workspace-1",
        root_path=str(workspace_root.resolve()),
        source_repo_url="https://github.com/example/platform.git",
        default_branch="main",
        working_branch="workflow/workflow-1",
    )
    state = create_initial_agent_state(workflow_id="workflow-1", workspace_descriptor=descriptor)
    state["artifacts"] = list(artifacts)
    return state


def envelope(artifact_id: str, producer: str = "fixture") -> dict[str, Any]:
    """Return the shared fields required by each strict artifact fixture."""
    return {
        "schema_version": "1.0",
        "workflow_id": "workflow-1",
        "artifact_id": artifact_id,
        "producer": producer,
        "timestamp": datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
        "metadata": {"source": "test"},
        "validation_status": "valid",
    }


def requirement(requirement_id: str = "requirement-1") -> Requirement:
    """Create one traceable requirement for artifact fixtures."""
    return Requirement(
        requirement_id=requirement_id,
        description="The platform must provide a tested artifact-driven workflow.",
        priority="must",
        acceptance_criteria=["The workflow publishes a structured artifact."],
        dependencies=[],
    )


def prd_artifact() -> PRDArtifact:
    """Create the workflow's input PRD artifact."""
    return PRDArtifact(
        **envelope(ARTIFACT_FILENAMES["prd"]),
        title="Artifact workflow",
        problem_statement="Teams need reliable automation for engineering changes.",
        goals=["Produce reviewed changes."],
        user_stories=[
            UserStory(
                story_id="story-1",
                persona="Engineering manager",
                need="Review generated changes",
                benefit="I can supervise delivery safely",
                acceptance_criteria=["A review happens before pull-request creation."],
            )
        ],
        requirements=[requirement()],
        constraints=["Use artifacts for agent handoffs."],
        out_of_scope=["Direct pushes to main."],
        stakeholders=["Engineering"],
    )


def technical_prd_artifact() -> TechnicalPRDArtifact:
    """Create a minimal valid technical PRD fixture."""
    return TechnicalPRDArtifact(
        **envelope(ARTIFACT_FILENAMES["technical_prd"]),
        title="Artifact workflow technical PRD",
        solution_summary="Use strict artifacts between bounded specialized agents.",
        functional_requirements=[requirement()],
        non_functional_requirements=[],
        data_requirements=["Store all agent outputs as artifacts."],
        integration_requirements=[],
        security_requirements=["Reject path traversal."],
        assumptions=["The workspace descriptor is trusted."],
        unresolved_questions=[],
    )


def architecture_artifact() -> ArchitectureArtifact:
    """Create one valid architecture artifact fixture."""
    return ArchitectureArtifact(
        **envelope(ARTIFACT_FILENAMES["architecture"]),
        system_overview="Agents create validated artifacts in a shared workflow state.",
        technology_stack={"workflow": "LangGraph"},
        repository_structure=["agents/", "artifacts/"],
        components=[
            ArchitectureComponent(
                component_id="workflow",
                name="Workflow engine",
                responsibility="Coordinates artifact-producing agent nodes.",
                technology="LangGraph",
                dependencies=[],
            )
        ],
        api_contracts=[],
        data_entities=[],
        decisions=[
            ArchitectureDecision(
                decision_id="decision-1",
                title="Use artifacts",
                decision="Use strict artifact schemas for every handoff.",
                rationale="Validated handoffs keep agents decoupled.",
                consequences=["Agents do not exchange natural-language messages."],
            )
        ],
        risks=[
            ArchitectureRisk(
                risk_id="risk-1",
                description="Invalid model output can stop a workflow.",
                likelihood="medium",
                impact="high",
                mitigation="Validate every model response against artifact schemas.",
            )
        ],
        deployment_strategy="Run the API and workers as separate services.",
    )


def execution_graph_artifact() -> ExecutionGraphArtifact:
    """Create a minimal valid execution-graph artifact fixture."""
    return ExecutionGraphArtifact(
        **envelope(ARTIFACT_FILENAMES["execution_graph"]),
        entry_node_id="planner",
        terminal_node_ids=["planner"],
        nodes=[
            ExecutionNode(
                node_id="planner",
                agent="planner",
                action="create_plan",
                input_artifact_types=["technical_prd"],
                output_artifact_type="architecture",
                requires_human_approval=False,
            )
        ],
        edges=[],
    )


def task_plan_artifact() -> TaskPlanArtifact:
    """Create a minimal valid task-plan artifact fixture."""
    return TaskPlanArtifact(
        **envelope(ARTIFACT_FILENAMES["task_plan"]),
        summary="Implement the validated agent nodes.",
        tasks=[
            PlannedTask(
                task_id="task-1",
                title="Implement agents",
                description="Add artifact-producing workflow nodes.",
                owner="engineer",
                priority="high",
                dependencies=[],
                acceptance_criteria=["Every agent can run independently with mocks."],
                estimated_effort_points=3,
            )
        ],
        milestones=[
            Milestone(
                milestone_id="milestone-1",
                title="Agent layer",
                objective="Deliver the five core workflow agents.",
                task_ids=["task-1"],
            )
        ],
        implementation_order=["task-1"],
        test_strategy=["Run node unit tests with mock adapters."],
        risk_management_plan=["Reject invalid model output."],
    )


def code_completion_artifact() -> CodeCompletionArtifact:
    """Create a completed implementation report suitable for review and PR publication."""
    return CodeCompletionArtifact(
        **envelope(ARTIFACT_FILENAMES["code_completion"]),
        completion_status="completed",
        summary="Implement artifact-driven agent nodes.",
        file_changes=[],
        validation_results=[],
        test_coverage_percent=None,
        remaining_work=[],
        commit_sha="0123456789abcdef",
    )


def review_artifact() -> ReviewArtifact:
    """Create an approved review fixture with a passing requirement check."""
    return ReviewArtifact(
        **envelope(ARTIFACT_FILENAMES["review"]),
        verdict="approved",
        summary="The implementation satisfies the technical PRD.",
        requirement_checks=[
            RequirementCheck(
                requirement_id="requirement-1",
                passed=True,
                evidence="Mocked node tests cover the artifact handoff.",
            )
        ],
        findings=[],
        architecture_assessment="The nodes depend only on injected boundaries.",
        security_assessment="The executor retains workspace file boundaries.",
        test_coverage_assessment="Each node has independent mock-based tests.",
    )


def domain_payload(artifact: Any) -> dict[str, Any]:
    """Strip platform-controlled envelope fields to form an allowed model response payload."""
    payload = cast(dict[str, Any], artifact.model_dump(mode="json"))
    for field in (
        "schema_version",
        "workflow_id",
        "artifact_id",
        "producer",
        "timestamp",
        "metadata",
        "validation_status",
        "artifact_type",
    ):
        del payload[field]
    return payload


def validation_result(command: str, *, return_code: int) -> ValidationResult:
    """Create one completed validation command result without running any process."""
    return ValidationResult(
        command=(command, "check", "."),
        return_code=return_code,
        stdout="",
        stderr="",
        timed_out=False,
        duration_seconds=0.1,
    )


def test_a_test_suite_that_already_failed_remains_a_required_gate() -> None:
    """Existing repository debt is diagnosed before coding rather than hidden at review."""
    failing = ValidationResult(
        command=("npm", "run", "test"),
        return_code=1,
        stdout="Cannot find module '../../utils/qaUtils'",
        stderr="",
        timed_out=False,
        duration_seconds=1.0,
        validation_type="test",
        repository_id="frontend",
        status=ValidationStatus.FAILED,
    )
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}

    _apply_validation_findings(payload, (failing,))

    assert failing.required is True
    assert payload["verdict"] == "changes_requested"


@pytest.mark.asyncio
async def test_a_malformed_coding_response_is_repaired_not_abandoned(tmp_path: Path) -> None:
    """Feature -047's backend died before writing anything, on one unparseable response.

    The planner and the reviewer both repair a rejected response; the coding executor had no
    such attempt, so a single malformed reply ended the workstream.
    """

    class MalformedThenValidClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

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
            del input_text
            self.calls.append(instructions)
            body = (
                "here is your code: {oops"
                if len(self.calls) == 1
                else json.dumps(
                    {
                        "summary": "Add the service module.",
                        "files": [{"path": "src/service.py", "content": "value = 1\n"}],
                    }
                )
            )
            return LLMResponse(
                response_id=f"coding-{len(self.calls)}",
                model="mock-coding-model",
                output_text=body,
                input_tokens=1,
                output_tokens=1,
            )

    client = MalformedThenValidClient()

    result = await ResponsesCodingExecutor(client).execute(
        workspace_root=tmp_path,
        instructions="Implement the service.",
        input_text="{}",
    )

    assert len(client.calls) == 2
    assert "rejected before any file was written" in client.calls[1]
    assert (tmp_path / "src" / "service.py").read_text(encoding="utf-8") == "value = 1\n"
    assert result.summary == "Add the service module."


def test_a_clarification_revision_stays_in_its_artifacts_lineage() -> None:
    """A resolved technical PRD must still satisfy a lookup for its base artifact ID.

    Clarification answers are folded in as an immutable replacement rather than a mutation,
    which publishes `002_technical_prd.revision-2.json`. Until the matcher recognised that
    suffix, every feature that answered a question failed with "required artifact is
    missing: 002_technical_prd.json" -- it cost features -057, -058 and -059.
    """
    base = ARTIFACT_FILENAMES["technical_prd"]

    assert artifact_id_matches_lineage(f"{base.removesuffix('.json')}.revision-2.json", base)
    # Retries and publication markers keep the behaviour they already had.
    assert artifact_id_matches_lineage(f"{base.removesuffix('.json')}.attempt-1.json", base)
    assert artifact_id_matches_lineage(base, base)


def test_an_unrelated_artifact_never_matches_another_lineage() -> None:
    """Matching must stay anchored to the base ID, not to any similarly prefixed name."""
    base = ARTIFACT_FILENAMES["technical_prd"]
    stem = base.removesuffix(".json")

    assert not artifact_id_matches_lineage(f"{stem}.revision-two.json", base)
    assert not artifact_id_matches_lineage(f"{stem}.summary.json", base)
    assert not artifact_id_matches_lineage(ARTIFACT_FILENAMES["prd"], base)


def test_a_scoped_criterion_demanding_a_person_is_withheld_like_any_other() -> None:
    """The reviewer enforces the workstream's criteria, so those must be filtered too.

    Withholding applied only to PRD requirements, but a plan writes its own scoped criteria and
    those are what the reviewer holds the repository to. Live feature -083 spent ten frontend
    attempts against "requires manual verification of ... rendering", which no attempt could
    satisfy and which the requirement-level projection never saw.
    """
    task_plan = task_plan_artifact().model_copy(
        update={
            "metadata": {
                "review_scope": {
                    "repository_id": "frontend",
                    "workstream_id": "frontend-ws",
                    "role": "frontend",
                    "requirement_ids": ["REQ-2"],
                    "acceptance_criteria": [
                        "The view renders one entry per returned sample.",
                        "Requires manual verification of healthy and unhealthy rendering.",
                    ],
                }
            }
        }
    )

    scope = _review_scope(task_plan)

    assert scope["acceptance_criteria"] == ["The view renders one entry per returned sample."]
    assert scope["acceptance_criteria_not_reviewable"] == [
        "Requires manual verification of healthy and unhealthy rendering."
    ]


class QueuedProductManagerClient:
    """Serve one scripted response per call, so a repair attempt can be observed directly."""

    def __init__(self, payloads: Sequence[dict[str, Any]]) -> None:
        """Queue the raw payloads the model will return, in order."""
        self._payloads = list(payloads)
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
        """Return the next queued payload, refusing to be called more times than scripted."""
        self.calls.append((instructions, input_text))
        if not self._payloads:
            msg = "the product manager called the model more times than the test scripted"
            raise AssertionError(msg)
        payload = self._payloads.pop(0)
        return LLMResponse(
            response_id=f"pm-response-{len(self.calls)}",
            model="mock-reasoning-model",
            output_text=json.dumps(payload),
            input_tokens=10,
            output_tokens=5,
        )


def _technical_prd_payload_with_priority(priority: str) -> dict[str, Any]:
    """Return a technical-PRD payload whose single requirement carries the given priority."""
    payload = domain_payload(technical_prd_artifact())
    payload["functional_requirements"][0]["priority"] = priority
    return payload


@pytest.mark.asyncio
async def test_a_rejected_technical_prd_is_repaired_instead_of_ending_the_feature(
    tmp_path: Path,
) -> None:
    """AB-Feature-120 died twenty-four seconds in, on four values outside one enum.

    This is the first model call of a feature and it had no repair path at all, unlike the
    planner immediately after it, so one bad enum value ended the run before a repository was
    read. The prompt states the allowed values as `must|should|could|wont` -- which is also
    what a placeholder looks like, so a model copying its shape produces exactly this.
    """
    state = agent_state(tmp_path, [prd_artifact()])
    llm_client = QueuedProductManagerClient(
        [
            _technical_prd_payload_with_priority("must|should|could|wont"),
            _technical_prd_payload_with_priority("must"),
        ]
    )

    update = await ProductManagerAgent(prompt_loader=PromptLoader(), llm_client=llm_client).run(
        state
    )

    technical_prd = update["artifacts"][0]
    assert isinstance(technical_prd, TechnicalPRDArtifact)
    assert technical_prd.functional_requirements[0].priority == "must"
    # Exactly one repair, and it is recorded as one: a repaired plan must be distinguishable
    # from one the model got right first time.
    assert len(llm_client.calls) == 2
    assert technical_prd.metadata["repair_of_response_id"] == "pm-response-1"
    assert technical_prd.metadata["repair_reason"] == (
        "TechnicalPRDArtifact: schema validation rejected a value "
        "[literal_error] (expected: 'must', 'should', 'could' or 'wont')"
    )
    # The repair is anchored to the response that was rejected, and is told which values the
    # schema will accept -- both of which the first attempt provably did not act on.
    repair_instructions = llm_client.calls[1][0]
    assert "Your previous response was rejected" in repair_instructions
    assert "must|should|could|wont" in repair_instructions
    assert "is never itself a valid value" in repair_instructions


@pytest.mark.asyncio
async def test_a_technical_prd_rejected_twice_names_the_stage_that_could_not_produce_it(
    tmp_path: Path,
) -> None:
    """One repair, not an unbounded loop -- and a failure that says whose it was."""
    state = agent_state(tmp_path, [prd_artifact()])
    llm_client = QueuedProductManagerClient(
        [
            _technical_prd_payload_with_priority("high"),
            _technical_prd_payload_with_priority("critical"),
        ]
    )

    with pytest.raises(AgentArtifactError) as rejection:
        await ProductManagerAgent(prompt_loader=PromptLoader(), llm_client=llm_client).run(state)

    # Bounded: the model is asked twice and never a third time.
    assert len(llm_client.calls) == 2
    error = rejection.value
    assert error.stage == "product_manager"
    assert len(error.diagnostics) == 2
    assert error.diagnostics[0].startswith("first attempt rejected: ")
    assert error.diagnostics[1].startswith("repair attempt rejected: ")
    # Both rejections say which field the schema refused, which four identical lines of
    # "schema validation rejected a value [literal_error]" never could.
    for diagnostic in error.diagnostics:
        assert "TechnicalPRDArtifact" in diagnostic
        assert "'must', 'should', 'could' or 'wont'" in diagnostic


def test_a_schema_rejection_names_the_constraint_but_never_the_rejected_value() -> None:
    """The allowed set comes from this repository's schema; the input never contributes.

    ``loc`` stays excluded deliberately. An `extra_forbidden` rejection puts the offending
    mapping key there verbatim, and a mapping key is input.
    """
    secret = "sk-not-a-real-key-value"
    with pytest.raises(ValidationError) as rejection:
        TechnicalPRDArtifact.model_validate(
            {
                **envelope(ARTIFACT_FILENAMES["technical_prd"]),
                "title": "t",
                "solution_summary": "s",
                "functional_requirements": [
                    {
                        "requirement_id": "req-1",
                        "description": "d",
                        "priority": secret,
                        "acceptance_criteria": ["a"],
                        "dependencies": [],
                    }
                ],
                "non_functional_requirements": [],
                "data_requirements": [],
                "integration_requirements": [],
                "security_requirements": [],
                "assumptions": [],
                "unresolved_questions": [],
                secret: secret,
            }
        )

    diagnostics = safe_error_diagnostics(rejection.value)
    joined = " ".join(diagnostics)
    assert secret not in joined
    assert any(
        item
        == (
            "TechnicalPRDArtifact: schema validation rejected a value "
            "[literal_error] (expected: 'must', 'should', 'could' or 'wont')"
        )
        for item in diagnostics
    ), diagnostics
    # The forbidden extra key is reported by type alone, because naming it would quote input.
    assert any("[extra_forbidden]" in item for item in diagnostics), diagnostics


def test_a_rejected_artifact_is_attributed_to_the_agent_that_could_not_produce_it() -> None:
    """A planning failure label sent AB-Feature-120's reader to the one agent that never ran."""
    with pytest.raises(ValidationError) as rejection:
        TechnicalPRDArtifact.model_validate({"title": "t"})

    assert _rejected_stage(rejection.value) == ("feature_analysis_failed", "product_manager")
    # A raiser that names its own stage is believed over any inference.
    assert _rejected_stage(AgentArtifactError("rejected", stage="product_manager")) == (
        "feature_analysis_failed",
        "product_manager",
    )
    assert _rejected_stage(AgentArtifactError("rejected", stage="feature_planner")) == (
        "feature_planning_failed",
        "feature_planner",
    )
    # And one that names nothing, about a model this mapping does not cover, is reported as
    # unattributed rather than assigned to a component that may never have run.
    with pytest.raises(ValidationError) as unmapped:
        ReviewArtifact.model_validate({"verdict": "nope"})
    assert _rejected_stage(unmapped.value) == ("feature_artifact_rejected", "feature_runtime")
    assert _rejected_stage(AgentArtifactError("rejected")) == (
        "feature_artifact_rejected",
        "feature_runtime",
    )


def test_a_failing_command_tells_the_next_attempt_what_it_actually_reported() -> None:
    """The finding is how a failing command reaches the engineer, and it said nothing.

    A validation finding becomes a blocking issue, which becomes the root cause in the retry
    plan the engineer is handed. For a plain non-zero exit that whole path carried only
    "exited with code 1". AB-Feature-123's console spent six attempts on that sentence while
    the runner's own report -- captured, bounded and redacted, sitting in `stderr_summary` --
    named the failing test and the exact defect.
    """
    jest_output = (
        "  ● AllApps bulk deletion integration › passes the established full-access value\n"
        "\n"
        "    mockConstructor(...): Nothing was returned from render. This usually means a "
        "return statement is missing. Or, to render nothing, return null.\n"
        "\n"
        "      31 |   it('passes the established All Apps full-access value', () => {\n"
        "\n"
        "Tests: 1 failed, 4 passed, 5 total\n"
    )
    failing = ValidationResult(
        command=("npm", "run", "test", "--", "src/pages/AllApps.test.js"),
        return_code=1,
        stdout="> user-fe@0.1.0 test\n",
        stderr=jest_output,
        timed_out=False,
        duration_seconds=1.0,
        validation_type="test",
    )

    payload: dict[str, Any] = {"verdict": "approved", "findings": []}
    _apply_validation_findings(payload, (failing,))

    description = payload["findings"][0]["description"]
    # Still says which command failed and how.
    assert "npm run test -- src/pages/AllApps.test.js exited with code 1." in description
    # And now says what it reported, which is the part that can be acted on.
    assert "Nothing was returned from render" in description
    assert "AllApps bulk deletion integration" in description


def test_a_timeout_still_reports_the_cause_rather_than_a_transcript() -> None:
    """A command that never finished has no verdict to quote, and its tail is not one.

    Carrying an excerpt here would hand the engineer a runner's progress chatter as though
    it were a diagnosis, which is the misreading the capacity classification exists to stop.
    """
    timed_out = ValidationResult(
        command=("npm", "run", "test"),
        return_code=None,
        stdout="RUNS src/a.test.js\nRUNS src/b.test.js\n",
        stderr="",
        timed_out=True,
        duration_seconds=900.0,
        validation_type="test",
    )

    payload: dict[str, Any] = {"verdict": "approved", "findings": []}
    _apply_validation_findings(payload, (timed_out,))

    description = payload["findings"][0]["description"]
    assert description == "npm run test exceeded its configured timeout."


def test_the_modules_an_attempt_imports_are_shown_to_the_next_one(tmp_path: Path) -> None:
    """Nothing ranks a file higher than the change depending on it, and ranking missed it.

    AB-Feature-128's backend wrote `appValidation.addAppSchema.body.validate(row)` and spent
    five of its eleven attempts on `TypeError: addAppSchema.body is undefined`. The reviewer
    said so every time; the engineer still could not see `app.validation.js` to learn the
    real shape, because the snapshot is relevance-scored from the task's wording and a module
    the task never names loses the budget to files nobody needs.
    """
    (tmp_path / "server" / "services" / "app").mkdir(parents=True)
    (tmp_path / "server" / "validations").mkdir(parents=True)
    written = Path("server") / "services" / "app" / "bulkImport.service.js"
    (tmp_path / written).write_text(
        "const appValidation = require('../../validations/app.validation');\n"
        "module.exports = { run: (row) => appValidation.addAppSchema.validate(row) };\n",
        encoding="utf-8",
    )
    # The file whose exports the attempt is guessing at. Sorts last, so ranking loses it.
    imported = Path("server") / "validations" / "app.validation.js"
    (tmp_path / imported).write_text(
        "const addAppSchema = { body: undefined };\nmodule.exports = { addAppSchema };\n",
        encoding="utf-8",
    )
    source_files = {written, imported}
    (tmp_path / "server" / "controllers").mkdir(parents=True)
    for index in range(90):
        other = Path("server") / "controllers" / f"controller_{index:02d}.js"
        (tmp_path / other).write_text(f"// controller {index}\n", encoding="utf-8")
        source_files.add(other)

    tools = WorkspaceFileTools(tmp_path)
    resolved = _imported_repository_paths([written.as_posix()], source_files, tools)

    assert resolved.paths == (imported.as_posix(),)

    unranked = _repository_context(source_files, tools)
    shown = _repository_context(source_files, tools, frozenset(), resolved.paths)

    assert imported.as_posix() not in [item["path"] for item in unranked["files"]]
    assert imported.as_posix() in [item["path"] for item in shown["files"]]


def test_a_dependency_is_not_mistaken_for_a_file_this_repository_owns(tmp_path: Path) -> None:
    """A package specifier resolves to nothing here, and its source is not ours to show."""
    (tmp_path / "server").mkdir(parents=True)
    written = Path("server") / "handler.js"
    (tmp_path / written).write_text(
        "const express = require('express');\nconst mongoose = require('mongoose');\n"
        "import Joi from 'joi';\n",
        encoding="utf-8",
    )

    resolved = _imported_repository_paths(
        [written.as_posix()], {written}, WorkspaceFileTools(tmp_path)
    )

    assert resolved.paths == ()


def test_a_dotted_python_import_resolves_to_the_module_in_this_checkout(tmp_path: Path) -> None:
    """Python names its modules with dots; the checkout stores them as directories."""
    (tmp_path / "app").mkdir(parents=True)
    written = Path("app") / "status.py"
    (tmp_path / written).write_text("from app.status_feed import status_feed\n", encoding="utf-8")
    imported = Path("app") / "status_feed.py"
    (tmp_path / imported).write_text(
        "def status_feed() -> dict:\n    return {}\n", encoding="utf-8"
    )

    resolved = _imported_repository_paths(
        [written.as_posix()], {written, imported}, WorkspaceFileTools(tmp_path)
    )

    assert resolved.paths == (imported.as_posix(),)


def test_imports_are_found_in_the_diff_when_the_workspace_has_been_reset(tmp_path: Path) -> None:
    """A rejected attempt is reset before the next one runs, so its files are already gone.

    Reading the workspace to learn what the previous attempt imported finds nothing on
    exactly the retry this exists for: the checkout is restored to its committed state
    between attempts, and `git status` shows a clean tree. What the attempt wrote survives
    only in the captured diff, which the engineer is already handed.
    """
    (tmp_path / "server" / "validation").mkdir(parents=True)
    imported = Path("server") / "validation" / "app.validation.js"
    (tmp_path / imported).write_text(
        "const addAppSchema = { body: undefined };\nmodule.exports = { addAppSchema };\n",
        encoding="utf-8",
    )
    # The service the attempt wrote is absent from the workspace, exactly as after a reset.
    written = "server/services/app/bulkImport.service.js"
    diff = (
        "diff --git a/server/services/app/bulkImport.service.js "
        "b/server/services/app/bulkImport.service.js\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{written}\n"
        "@@ -0,0 +1,2 @@\n"
        "+const appValidation = require('../../validation/app.validation');\n"
        "+module.exports = { run: (row) => appValidation.addAppSchema.body.validate(row) };\n"
    )

    resolved = _imported_repository_paths([written], {imported}, WorkspaceFileTools(tmp_path), diff)

    assert resolved.paths == (imported.as_posix(),)


def test_a_module_the_attempt_stopped_importing_is_not_forced_into_context(
    tmp_path: Path,
) -> None:
    """A removed require is a dependency the change is getting rid of, not one to study."""
    (tmp_path / "server").mkdir(parents=True)
    abandoned = Path("server") / "legacy.js"
    (tmp_path / abandoned).write_text("module.exports = {};\n", encoding="utf-8")
    diff = (
        "--- a/server/handler.js\n"
        "+++ b/server/handler.js\n"
        "@@ -1,2 +1,1 @@\n"
        "-const legacy = require('./legacy');\n"
        "+module.exports = {};\n"
    )

    resolved = _imported_repository_paths(
        ["server/handler.js"], {abandoned}, WorkspaceFileTools(tmp_path), diff
    )

    assert resolved.paths == ()


def _drains_the_import_cap(directory: Path, count: int) -> str:
    """Write ``count`` resolvable neighbours and return a module importing all of them."""
    for index in range(count):
        (directory / f"neighbour_{index:02d}.js").write_text(
            f"module.exports = {{ id: {index} }};\n", encoding="utf-8"
        )
    return "".join(
        f"const n{index} = require('./neighbour_{index:02d}');\n" for index in range(count)
    )


def test_an_import_the_cap_refuses_is_recorded_and_not_left_to_vanish(tmp_path: Path) -> None:
    """AB-Feature-218's defect: the cap is global, so the first source read can spend it all.

    Its `_change_sources` order led with the plan's two assigned files -- `package.json` and
    `package-lock.json`, which resolve no specifiers, because `_assigned_context_paths` had
    dropped the plan's nine directories -- and then the lineage's first-seen order.
    `app.service.js` resolved six imports on its own, the cap was reached, and the function
    returned before it read `app.route.js`, whose third resolved import is
    `server/validation/app.validation.js`: the file four attempts were told to reuse, and the
    one none of them was shown.

    What is asserted here is the half this function owns. The file is refused -- ordering is
    what stops that, and it is fixed where the order is decided -- but the refusal is now on
    the record with a reason, where before it appeared in no field at all: not
    `context_file_paths`, not `context_required_dropped`, not `context_required_omitted`.
    """
    (tmp_path / "server" / "services").mkdir(parents=True)
    (tmp_path / "server" / "validation").mkdir(parents=True)
    services = tmp_path / "server" / "services"
    # The lineage's first file, and on its own it resolves the whole category budget.
    greedy = Path("server") / "services" / "app.service.js"
    (tmp_path / greedy).write_text(
        _drains_the_import_cap(services, _MAX_IMPORTED_CONTEXT_PATHS), encoding="utf-8"
    )
    # The file the change is actually about, read second and therefore too late.
    route = Path("server") / "services" / "app.route.js"
    (tmp_path / route).write_text(
        "const appValidation = require('../validation/app.validation');\n"
        "module.exports = { run: (row) => appValidation.addAppSchema.body.validate(row) };\n",
        encoding="utf-8",
    )
    starved = Path("server") / "validation" / "app.validation.js"
    (tmp_path / starved).write_text(
        "const addAppSchema = Joi.object({});\nmodule.exports = { addAppSchema };\n",
        encoding="utf-8",
    )
    existing = {
        starved,
        greedy,
        route,
        *(Path("server") / "services" / f"neighbour_{index:02d}.js" for index in range(6)),
    }
    # Enough checkout for the ranked walk to genuinely lose the file, as 218's did: the
    # refused module is absent from the prompt, not merely absent from the required set.
    (tmp_path / "server" / "controllers").mkdir(parents=True)
    for index in range(90):
        other = Path("server") / "controllers" / f"controller_{index:02d}.js"
        (tmp_path / other).write_text(f"// controller {index}\n", encoding="utf-8")
        existing.add(other)
    tools = WorkspaceFileTools(tmp_path)

    resolved = _imported_repository_paths([greedy.as_posix(), route.as_posix()], existing, tools)

    assert len(resolved.paths) == _MAX_IMPORTED_CONTEXT_PATHS
    assert starved.as_posix() not in resolved.paths
    # The point of the whole part: refused, and answerable from the record.
    assert resolved.discarded[starved.as_posix()] == "imported_module_cap"

    context = _repository_context(
        existing, tools, frozenset(), resolved.paths, candidates_discarded=resolved.discarded
    )

    assert starved.as_posix() in context["candidate_discarded_paths"]
    assert context["candidate_discarded_reasons"][starved.as_posix()] == "imported_module_cap"


def test_a_refused_import_the_ranked_walk_showed_anyway_is_not_reported_as_discarded(
    tmp_path: Path,
) -> None:
    """The field says what the prompt did not get, and a false entry in it misleads a triage.

    A candidate the category cap refused can still reach the snapshot through lexical
    ranking. Reporting it as discarded would put a file that IS in the prompt into the one
    field a forensic pass consults to ask whether the attempt ever saw it.
    """
    (tmp_path / "server").mkdir(parents=True)
    shown = Path("server") / "app.validation.js"
    (tmp_path / shown).write_text("module.exports = {};\n", encoding="utf-8")
    tools = WorkspaceFileTools(tmp_path)

    context = _repository_context(
        {shown},
        tools,
        frozenset(),
        (),
        candidates_discarded={shown.as_posix(): "imported_module_cap"},
    )

    assert shown.as_posix() in [item["path"] for item in context["files"]]
    assert context["candidate_discarded_paths"] == []


def test_a_discarded_candidate_cannot_refuse_an_attempt(tmp_path: Path) -> None:
    """The safety invariant: no attempt that passes today fails tomorrow.

    `_refuse_if_required_context_dropped` reads `required_dropped_paths`. A cap-refused
    candidate is recorded beside that list and never in it, so even when the path is one the
    plan also assigned, the new record cannot turn a running attempt into a stop.
    """
    (tmp_path / "server").mkdir(parents=True)
    assigned = Path("server") / "app.validation.js"
    (tmp_path / assigned).write_text("module.exports = {};\n", encoding="utf-8")
    tools = WorkspaceFileTools(tmp_path)

    context = _repository_context(
        {assigned},
        tools,
        frozenset(),
        (assigned.as_posix(),),
        candidates_discarded={assigned.as_posix(): "imported_module_cap"},
    )

    assert context["required_dropped_paths"] == []
    # Would raise RequiredContextRefusal if the discard had been merged into the drop list.
    _refuse_if_required_context_dropped(
        context, assigned_paths=(assigned.as_posix(),), diagnostic_file_paths=()
    )


def test_the_discard_ledger_is_bounded(tmp_path: Path) -> None:
    """A ledger of refusals is a record, not an inventory of the checkout's import graph."""
    (tmp_path / "server").mkdir(parents=True)
    server = tmp_path / "server"
    over = _MAX_IMPORTED_CONTEXT_PATHS + _MAX_RECORDED_IMPORT_DISCARDS + 10
    written = Path("server") / "handler.js"
    (tmp_path / written).write_text(_drains_the_import_cap(server, over), encoding="utf-8")
    existing = {written, *(Path("server") / f"neighbour_{index:02d}.js" for index in range(over))}

    resolved = _imported_repository_paths(
        [written.as_posix()], existing, WorkspaceFileTools(tmp_path)
    )

    assert len(resolved.paths) == _MAX_IMPORTED_CONTEXT_PATHS
    assert len(resolved.discarded) == _MAX_RECORDED_IMPORT_DISCARDS


@pytest.mark.asyncio
async def test_a_review_records_the_role_that_performed_it(tmp_path: Path) -> None:
    """Role separation is logical, so the artifact has to say which role judged the work.

    Two roles may legitimately be configured with the same model. What makes a review
    independent is that it was performed as a review -- it never writes code, and it is the
    only thing that may mark a finding resolved -- so the role belongs on the artifact beside
    the model name.
    """
    completion = cast(
        CodeCompletionArtifact,
        (
            await EngineerAgent(
                prompt_loader=PromptLoader(),
                coding_executor=MockCodingExecutor(
                    file_updates={"src/service.py": "value = 'reviewed'\n"}
                ),
            ).run(agent_state(tmp_path, [task_plan_artifact()]))
        )["artifacts"][0],
    )
    validation = StaticValidationTool(
        ruff_result=validation_result("ruff", return_code=0),
        pytest_result=validation_result("pytest", return_code=0),
    )

    update = await ReviewerAgent(
        prompt_loader=PromptLoader(),
        llm_client=StaticLLMClient(domain_payload(review_artifact())),
        validation_tool=validation,
        model_role=ModelRole.REVIEW,
    ).run(agent_state(tmp_path, [technical_prd_artifact(), task_plan_artifact(), completion]))

    review = update["artifacts"][0]
    assert review.metadata["model_role"] == ModelRole.REVIEW.value
    # The reviewer changed no code: its artifact reports a verdict and findings only.
    assert not hasattr(review, "file_changes")


@pytest.mark.asyncio
async def test_an_engineer_execution_records_the_role_that_wrote_the_change(
    tmp_path: Path,
) -> None:
    """A completed attempt has to name the model role that produced it, and its mode."""
    routing = ModelRoutingDecision(
        execution_mode=ModelExecutionMode.REVIEW_REMEDIATION,
        role=ModelRole.SCOPED_FIX,
        model="configured-fix-model",
        classification=ReviewFixComplexity.STANDARD,
        attempt=1,
        routing_reason="A localized lint correction.",
    )

    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(file_updates={"src/service.py": "value = 1\n"}),
        model_routing=routing,
    ).run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][0]
    assert completion.metadata["execution_mode"] == "REVIEW_REMEDIATION"
    assert completion.metadata["model_routing"]["role"] == ModelRole.SCOPED_FIX.value
    assert completion.metadata["model_routing"]["classification"] == "STANDARD"
    # The engineer never records a verdict about the finding it was asked to correct.
    assert "resolved" not in completion.metadata


class _RecordingCodingExecutor:
    """A coding executor that records each call and can write different files per call.

    Records the ``operation_executor`` it was handed, because whether a call is journaled is
    the property that matters most about the in-attempt source repair.
    """

    def __init__(self, *updates: dict[str, str], model: str = "test-model") -> None:
        self._updates = list(updates)
        self._model = model
        self.calls: list[dict[str, Any]] = []

    async def execute(
        self,
        *,
        workspace_root: PathLike,
        instructions: str,
        input_text: str,
        cancellation_token: Any = None,
        operation_executor: Any = None,
        images: Any = (),
    ) -> Any:
        del cancellation_token
        self.calls.append(
            {
                "instructions": instructions,
                "input_text": input_text,
                "journaled": operation_executor is not None,
            }
        )
        updates = self._updates[min(len(self.calls) - 1, len(self._updates) - 1)]
        root = Path(str(workspace_root))
        written: list[Path] = []
        for relative, content in updates.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(target)
        return CodingExecutionResult(
            response_id=f"response-{len(self.calls)}",
            model=self._model,
            summary="Applied the requested change.",
            modified_files=tuple(written),
        )


class _FormatterFailingUntil:
    """Reject the first ``failures`` verifications, then accept, recording each call."""

    def __init__(self, failures: int, diagnostics: Sequence[str]) -> None:
        self._failures = failures
        self._diagnostics = tuple(diagnostics)
        self.calls: list[tuple[str, ...]] = []

    async def format_paths(
        self, repository_root: PathLike, paths: Sequence[str]
    ) -> tuple[str, ...]:
        del repository_root
        self.calls.append(tuple(paths))
        if len(self.calls) <= self._failures:
            raise SourceValidationError(self._diagnostics)
        return ("eslint --fix",)


_LINT_DIAGNOSTIC = (
    "Pre-commit source validation failed (tool=npm; code=SOURCE_VALIDATION_FAILED_EXIT_1).\n"
    "src/service.test.js\n  74:36  error  Avoid using multiple assertions within `waitFor` "
    "callback  testing-library/no-wait-for-multiple-assertions"
)


@pytest.mark.asyncio
async def test_engineer_repairs_its_own_lint_defect_inside_the_attempt(tmp_path: Path) -> None:
    """A rule no fixer can rewrite must not cost the whole attempt.

    AB-Feature-158 lost four of twelve attempts to exactly this: the repository's own
    verification rejected the engineer's new files, the workspace was reset, and a retry was
    spent re-implementing the feature from nothing to clear a lint error.
    """
    executor = _RecordingCodingExecutor(
        {"src/service.js": "export const a = 1;\n"},
        {"src/service.test.js": "test('a', () => {});\n"},
    )
    formatter = _FormatterFailingUntil(1, [_LINT_DIAGNOSTIC])
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        source_formatter=formatter,
    )

    update = await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    # The attempt survived rather than being discarded.
    completion = update["artifacts"][-1]
    assert completion.completion_status == "completed"
    assert len(executor.calls) == 2
    # The correction was told exactly what the repository reported, and nothing else.
    assert "no-wait-for-multiple-assertions" in executor.calls[1]["instructions"]
    assert "pre-commit verification" in executor.calls[1]["instructions"]
    # It is recorded, so an attempt that took two calls is distinguishable from one that did not.
    assert completion.metadata["source_repair_response_id"] == "response-2"
    assert completion.metadata["source_repair_paths"] == ["src/service.test.js"]


@pytest.mark.asyncio
async def test_the_in_attempt_source_repair_is_never_journaled(tmp_path: Path) -> None:
    """One journaled coding effect per attempt, exactly as before.

    Crash recovery finds a completed coding operation by ``child_attempt`` and decides from it
    whether the engineer still has to run. Two completed records under one attempt would give
    that reconciliation a choice it has never had to make, and it is the most re-broken
    invariant in this codebase.
    """
    executor = _RecordingCodingExecutor(
        {"src/service.js": "export const a = 1;\n"},
        {"src/service.js": "export const a = 2;\n"},
    )
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        source_formatter=_FormatterFailingUntil(1, [_LINT_DIAGNOSTIC]),
        # Any non-None executor: the assertion is about which call is handed one.
        operation_executor=cast(ExternalOperationExecutor, object()),
    )

    await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    assert [call["journaled"] for call in executor.calls] == [True, False]


@pytest.mark.asyncio
async def test_a_repair_that_changes_nothing_does_not_re_ask_the_same_question(
    tmp_path: Path,
) -> None:
    """Re-verifying unchanged bytes gets the same answer, so the attempt fails as it did."""
    executor = _RecordingCodingExecutor({"src/service.js": "export const a = 1;\n"}, {})
    formatter = _FormatterFailingUntil(9, [_LINT_DIAGNOSTIC])
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        source_formatter=formatter,
    )

    with pytest.raises(SourceValidationError, match="pre-commit validation"):
        await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    assert len(executor.calls) == 2
    # Verified once for the original attempt; not a second time over identical bytes.
    assert len(formatter.calls) == 1


@pytest.mark.asyncio
async def test_a_repair_the_model_cannot_answer_fails_exactly_as_it_did_before(
    tmp_path: Path,
) -> None:
    """Every failure in the repair path degrades to the behaviour without it.

    The unclassified case, which is the fallback the two tests below it are the exceptions
    to: this adapter fault names no classification, so nothing can say whether the provider
    answered or the transport failed, and an attempt is not charged to a guess.
    """

    class FailingOnRepair(_RecordingCodingExecutor):
        async def execute(self, **kwargs: Any) -> Any:
            if self.calls:
                self.calls.append({"instructions": "", "input_text": "", "journaled": False})
                msg = "the model provider call failed (APIConnectionError)"
                raise LLMAdapterError(msg)
            return await super().execute(**kwargs)

    executor = FailingOnRepair({"src/service.js": "export const a = 1;\n"})
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        source_formatter=_FormatterFailingUntil(1, [_LINT_DIAGNOSTIC]),
    )

    with pytest.raises(SourceValidationError, match="pre-commit validation"):
        await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    assert len(executor.calls) == 2


@pytest.mark.asyncio
async def test_a_repair_the_transport_never_delivered_is_not_a_source_verdict(
    tmp_path: Path,
) -> None:
    """A repair whose call died in transport raises the fault, not the gate's rejection.

    AB-Feature-215 is the whole reason. A repair call was accepted by the provider and sent
    nothing for thirty minutes; this loop re-raised the gate's `SourceValidationError` in its
    place, so a dead socket was recorded as `validation_source_failure`, charged a full
    attempt, and sent the next attempt to fix source that had never been the problem. The
    same fault on the same feature's other repository -- a *journaled* call, so it reached
    the child loop as itself -- was classified as weather, had its 1919 seconds excluded from
    the budget, and cost no attempt at all. The only difference between the two was which
    call happened to have a journal row, and that is what this test removes.
    """

    class SilentOnRepair(_RecordingCodingExecutor):
        async def execute(self, **kwargs: Any) -> Any:
            if self.calls:
                self.calls.append({"instructions": "", "input_text": "", "journaled": False})
                msg = "Responses API accepted the request and sent no event"
                raise LLMAdapterError(msg, failure_classification="stream_silent")
            return await super().execute(**kwargs)

    executor = SilentOnRepair({"src/service.js": "export const a = 1;\n"})
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        source_formatter=_FormatterFailingUntil(1, [_LINT_DIAGNOSTIC]),
    )

    with pytest.raises(LLMAdapterError) as raised:
        await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    # Raised as itself: the classification is what the child loop's fault predicate reads,
    # and a `SourceValidationError` here would carry no classification at all.
    assert raised.value.failure_classification == "stream_silent"
    assert len(executor.calls) == 2


@pytest.mark.asyncio
async def test_a_repair_the_provider_refused_still_fails_on_the_gates_diagnostics(
    tmp_path: Path,
) -> None:
    """A provider that answered is settled, so the attempt keeps the verdict it already had.

    The other half of the rule, and the half that keeps the fix narrow: a refusal returns the
    same refusal to every retry, so spending the fault allowance on it would convert a
    diagnosable stop into "the provider did not answer" -- which is how AB-Feature-174's
    backend ended. Only the transport failing earns the allowance.
    """

    class RefusedOnRepair(_RecordingCodingExecutor):
        async def execute(self, **kwargs: Any) -> Any:
            if self.calls:
                self.calls.append({"instructions": "", "input_text": "", "journaled": False})
                msg = "the model declined to answer this request"
                raise LLMAdapterError(msg, failure_classification="model_refusal")
            return await super().execute(**kwargs)

    executor = RefusedOnRepair({"src/service.js": "export const a = 1;\n"})
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        source_formatter=_FormatterFailingUntil(1, [_LINT_DIAGNOSTIC]),
    )

    with pytest.raises(SourceValidationError, match="pre-commit validation"):
        await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    assert len(executor.calls) == 2


@pytest.mark.asyncio
async def test_a_repair_that_breaks_something_else_is_still_caught(tmp_path: Path) -> None:
    """The second verification covers everything the attempt now touches, not just the fix."""
    executor = _RecordingCodingExecutor(
        {"src/service.js": "export const a = 1;\n"},
        {"src/other.js": "export const b = 2;\n"},
    )
    formatter = _FormatterFailingUntil(2, [_LINT_DIAGNOSTIC])
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        source_formatter=formatter,
    )

    with pytest.raises(SourceValidationError, match="pre-commit validation"):
        await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    # The re-verification saw both the original file and the repair's.
    assert set(formatter.calls[1]) == {"src/service.js", "src/other.js"}


# --------------------------------------------------------------------------------------
# Remediation sees what it repairs (44-: F2 and F4)
# --------------------------------------------------------------------------------------


def _prior_attempt_completion(paths: Sequence[str], attempt: int = 0) -> CodeCompletionArtifact:
    """Record one earlier attempt's own writes, the way the engineer persists them."""
    fields = envelope(attempt_artifact_id(ARTIFACT_FILENAMES["code_completion"], attempt))
    fields["metadata"] = {"attempt_changed_paths": list(paths)}
    return CodeCompletionArtifact(
        **fields,
        completion_status="completed",
        summary="The prior attempt's implementation.",
        file_changes=[
            FileChange(path=path, change_type="added", description="Written by a prior attempt.")
            for path in paths
        ],
        validation_results=[],
        test_coverage_percent=None,
        remaining_work=[],
        commit_sha=None,
    )


def test_diagnostic_paths_are_normalized_into_the_workspace(tmp_path: Path) -> None:
    """An absolute workspace path with a `line:col` suffix becomes a usable relative path.

    176's gate named `/workspaces/feature-…/server/services/app/appBulkImport.service.js:
    179:23` three times, and the required-context membership test could match none of those
    spellings -- so the one file the attempt had to edit was never in its snapshot. A path
    outside the workspace must resolve to nothing: another feature's checkout is not this
    attempt's to read.
    """
    target = tmp_path / "server" / "x.js"
    target.parent.mkdir(parents=True)
    target.write_text("let value;\n" * 200, encoding="utf-8")

    locations = _diagnostic_file_locations(
        [
            "Pre-commit source validation failed (tool=npm; code=SOURCE_VALIDATION_FAILED).\n"
            f"{tmp_path}/server/x.js:179:23  error  Parsing error: Unexpected token ;",
            "/somewhere/else/entirely/other.js:5:1  error  no-undef",
        ],
        tmp_path,
    )

    assert locations == {"server/x.js": 179}

    # The repo-relative spelling with a bare line number resolves identically.
    relative = _diagnostic_file_locations(["server/x.js:42 is where the defect sits"], tmp_path)
    assert relative == {"server/x.js": 42}


def _reviewed_finding(**overrides: Any) -> ReviewFinding:
    """A blocking code-quality finding, with any field overridden by the test."""
    fields: dict[str, Any] = {
        "finding_id": "finding-1",
        "severity": "high",
        "title": "The bulk import drops rows whose owner is missing",
        "description": "The mapper silently skips a row when the owner lookup returns null.",
        "recommendation": "Surface the skipped row as a validation error.",
        "file_path": None,
        "line_number": None,
    }
    fields.update(overrides)
    return ReviewFinding(**fields)


def _blocking_review(*findings: ReviewFinding) -> ReviewArtifact:
    return ReviewArtifact(
        **envelope(ARTIFACT_FILENAMES["review"]),
        verdict="changes_requested",
        summary="Blocking findings remain.",
        requirement_checks=[],
        findings=list(findings),
        architecture_assessment="n/a",
        security_assessment="n/a",
        test_coverage_assessment="n/a",
    )


def test_a_findings_structured_citation_reaches_the_retry_context(tmp_path: Path) -> None:
    """A finding's `file_path` steers context even when its prose never repeats the path.

    45- F-B's shape: the prose names only the test file, but the reviewer cited the product
    file in the structured field. Before this, `_blocking_diagnostics` dropped `file_path`
    entirely, so the one file the fix had to edit competed for context by wording alone
    (the gap 47- Part A recorded). Non-blocking findings stay out, exactly as their text does.
    """
    product = tmp_path / "src" / "pages" / "AllApps.js"
    product.parent.mkdir(parents=True)
    product.write_text("export const AllApps = () => null;\n" * 30, encoding="utf-8")

    review = _blocking_review(
        _reviewed_finding(
            description="The render asserted in src/pages/AllApps.test.js throws a TypeError.",
            file_path="src/pages/AllApps.js",
            line_number=220,
        ),
        _reviewed_finding(
            finding_id="finding-2",
            severity="low",
            title="Advisory only",
            description="A non-blocking observation.",
            recommendation="Take it or leave it.",
            file_path="src/pages/AllApps.js",
            line_number=3,
        ),
    )

    diagnostics = _blocking_diagnostics(review, {})

    assert "The review locates this finding at src/pages/AllApps.js:220" in diagnostics
    # The low finding contributes nothing, citation included: one citation, not two.
    assert sum("locates this finding" in text for text in diagnostics) == 1
    # The cited product file is promoted with its line, though no prose spelled it.
    locations = _diagnostic_file_locations(diagnostics, tmp_path)
    assert locations["src/pages/AllApps.js"] == 220


def test_a_citation_naming_no_real_file_is_a_claim_not_a_fact(tmp_path: Path) -> None:
    """A cited path that resolves to nothing in this workspace is never promoted.

    The citation is the reviewer's claim; promotion requires the file to exist here. A
    finding without a citation adds no location line at all, and a citation without a line
    number still promotes the file, just with no excerpt anchor.
    """
    real = tmp_path / "src" / "service.js"
    real.parent.mkdir(parents=True)
    real.write_text("module.exports = {};\n", encoding="utf-8")

    review = _blocking_review(
        _reviewed_finding(file_path="src/phantom.js", line_number=12),
        _reviewed_finding(finding_id="finding-2", file_path="src/service.js"),
        _reviewed_finding(finding_id="finding-3"),
    )

    diagnostics = _blocking_diagnostics(review, {})

    assert "The review locates this finding at src/phantom.js:12" in diagnostics
    assert "The review locates this finding at src/service.js" in diagnostics
    assert sum("locates this finding" in text for text in diagnostics) == 2
    locations = _diagnostic_file_locations(diagnostics, tmp_path)
    assert "src/phantom.js" not in locations
    assert locations["src/service.js"] is None


@pytest.mark.asyncio
async def test_a_retry_is_shown_the_files_its_own_prior_attempts_wrote(tmp_path: Path) -> None:
    """The file a remediation exists to fix must not lose the snapshot budget to filler.

    176's remediation context held 46 files -- 2022 log files among them -- and none of the
    ten the feature had written. The models rewrote an 11.5 KB service file from memory three
    times and deterministically reproduced their own typo. A prior attempt's files are the
    remediation's target: they are required context, with first claim on the budget.
    """
    target = "src/pages/deep/service.js"
    written = tmp_path / target
    written.parent.mkdir(parents=True)
    # Large enough that a greedy budget walk would skip it once filler has eaten the budget,
    # and named after nothing the task says, so ranking cannot rescue it.
    written.write_text("function compute() {\n  return 1;\n}\n" * 320, encoding="utf-8")
    # Filler the task's own wording ranks highly, totalling more than the whole budget.
    for index in range(55):
        filler = tmp_path / "src" / "flows" / f"agentNodes{index:02d}.js"
        filler.parent.mkdir(parents=True, exist_ok=True)
        filler.write_text(f"// agent nodes workflow filler {index}\n" * 42, encoding="utf-8")

    state = agent_state(
        tmp_path,
        [task_plan_artifact(), _prior_attempt_completion([target]), review_artifact()],
    )
    state["retry_count"] = 1
    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(file_updates={target: "function compute() {}\n"}),
    ).run(state)

    completion = update["artifacts"][-1]
    assert target in completion.metadata["context_file_paths"]
    # Included whole: every channel of the omission record exists and stays empty.
    assert completion.metadata["context_required_excerpted"] == []
    assert completion.metadata["context_required_dropped"] == []
    assert completion.metadata["context_required_dropped_reasons"] == {}
    assert completion.metadata["context_required_omitted"] == []


def test_a_required_file_over_the_per_file_bound_is_excerpted_around_the_diagnostic_line(
    tmp_path: Path,
) -> None:
    """The inverse: too large to show whole means excerpted at the defect, never dropped.

    The partial view is recorded in ``required_excerpted_paths`` -- the file IS in the
    prompt, as a window -- and never in ``required_dropped_paths``, which is reserved for
    files whose bytes reached no prompt at all. The 61- analysis showed the old single field
    conflated the two, so "excerpted" was read as "never saw it" and vice versa. The union
    field stays for one release so existing readers keep their meaning.
    """
    # Sized from the constant, not a literal: enough ~40-character lines to exceed the
    # per-file bound whatever its value, with the defect line at the middle.
    line_count = _REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS // 20
    defect_index = line_count // 2
    lines = [f"function helper{index:03d}() {{ return {index}; }}" for index in range(line_count)]
    lines[defect_index] = "  const nestedDocument;"
    big = tmp_path / "src" / "big.js"
    big.parent.mkdir(parents=True)
    big.write_text("\n".join(lines) + "\n", encoding="utf-8")
    source_files = {Path("src/big.js")}

    context = _repository_context(
        source_files,
        WorkspaceFileTools(tmp_path),
        frozenset(),
        ("src/big.js",),
        {"src/big.js": defect_index + 1},
    )

    (entry,) = context["files"]
    assert entry["path"] == "src/big.js"
    assert "const nestedDocument;" in entry["content"]
    assert len(entry["content"]) <= _REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS
    assert entry["excerpt_lines"]["start"] <= defect_index + 1 <= entry["excerpt_lines"]["end"]
    # The prompt states what the window is a window of, so a model reading the excerpt can
    # tell "the file ends here" apart from "I was shown this much of it".
    assert entry["total_lines"] == line_count
    assert context["required_excerpted_paths"] == ["src/big.js"]
    assert context["required_dropped_paths"] == []
    assert context["required_omitted_paths"] == ["src/big.js"]


def test_a_required_file_with_no_anchor_is_excerpted_to_the_character_budget(
    tmp_path: Path,
) -> None:
    """The D1 shape from the 61- analysis: a big file, no diagnostic line, no symbol match.

    199's `app.service.js` -- 17,905 characters, 643 lines -- was shown as lines 1-61 on
    eight straight attempts, because the excerpt's binding constraint was the thirty-line
    radius rather than the 12,000-character budget the file was entitled to. With no anchor
    at all the window must still fill the character budget from the head of the file:
    roughly two-thirds of this file, not a tenth of it.
    """
    lines = [f"const helper{index:03d} = 'row {index:03d}';" for index in range(643)]
    big = tmp_path / "src" / "service.js"
    big.parent.mkdir(parents=True)
    big.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert len(big.read_text(encoding="utf-8")) > _REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS

    context = _repository_context(
        {Path("src/service.js")},
        WorkspaceFileTools(tmp_path),
        frozenset(),
        ("src/service.js",),
        {},
    )

    (entry,) = context["files"]
    # The window is head-of-file, sized by the character budget rather than a line radius:
    # far past the 61 lines the fixed radius produced, and stated as a range plus the file's
    # true length so the model can say "I cannot see it" about the rest.
    assert entry["excerpt_lines"]["start"] == 1
    assert entry["excerpt_lines"]["end"] >= 400
    assert "helper400" in entry["content"]
    assert (
        _REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS - 1_000
        <= len(entry["content"])
        <= _REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS
    )
    assert entry["total_lines"] == 643
    assert context["required_excerpted_paths"] == ["src/service.js"]
    assert context["required_dropped_paths"] == []


def test_each_dropped_required_path_records_which_branch_dropped_it(tmp_path: Path) -> None:
    """A drop is not one thing: unreadable, key-bearing and budget-evicted read identically
    in the old union field, and all four branches were indistinguishable in the record.

    Three of the four reasons in one selection: an undecodable file, a key-bearing file, and
    a file evicted because the required set exceeded the whole character budget. Each lands
    in ``required_dropped_paths`` with its reason, none in ``required_excerpted_paths``, and
    the union field carries excerpted and dropped alike.
    """
    unreadable = tmp_path / "src" / "binary.js"
    unreadable.parent.mkdir(parents=True)
    unreadable.write_bytes(b"\xff\xfe\x00 not utf-8")
    keyed = tmp_path / "src" / "signing.js"
    keyed.write_text(
        "const key = '-----BEGIN RSA PRIVATE KEY-----\\nabc\\n-----END RSA PRIVATE KEY-----';\n",
        encoding="utf-8",
    )
    # Sized from the constants, not literals: enough files each just under the per-file
    # bound that their sum exceeds the total budget, so the last one is excerpt-eligible
    # content that no longer fits and is dropped, not shown.
    filler_line = "const filler = 'x';\n"
    repeats = (_REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS - 200) // len(filler_line)
    wide_count = _REPOSITORY_CONTEXT_MAX_CHARACTERS // (repeats * len(filler_line)) + 2
    for index in range(wide_count):
        (tmp_path / "src" / f"wide{index}.js").write_text(filler_line * repeats, encoding="utf-8")

    required = (
        "src/binary.js",
        "src/signing.js",
        *(f"src/wide{index}.js" for index in range(wide_count)),
    )
    context = _repository_context(
        {Path(item) for item in required},
        WorkspaceFileTools(tmp_path),
        frozenset(),
        required,
        {},
    )

    reasons = context["required_dropped_reasons"]
    assert reasons["src/binary.js"] == "unreadable"
    assert reasons["src/signing.js"] == "key_material"
    assert "required_budget_exhausted" in reasons.values()
    assert set(context["required_dropped_paths"]) == set(reasons)
    assert not set(context["required_dropped_paths"]) & set(context["required_excerpted_paths"])
    assert set(context["required_omitted_paths"]) == (
        set(context["required_dropped_paths"]) | set(context["required_excerpted_paths"])
    )


@pytest.mark.asyncio
async def test_a_finding_naming_a_symbol_anchors_the_oversized_files_excerpt_on_it(
    tmp_path: Path,
) -> None:
    """Now-1(b): with no line number, the excerpt centres on the finding's subject.

    88% of review findings carry no line number, so the anchor of last resort was the head
    of the file -- which put 199's ordered block 367 lines below the bottom of its own
    evidence on four straight attempts. The finding here names `bulkOrchestrator` and no
    line; the checkout defines it at line 600 of a 700-line file, past what even the widened
    head-of-file window reaches. The definition-site lookup that already force-includes the
    file must also hand its line to the excerpt.
    """
    target = "src/services/order.js"
    lines = [f"const helper{index:03d} = 'row {index:03d}';" for index in range(700)]
    lines[599] = "module.exports.bulkOrchestrator = { run() { return 'orchestrated'; } };"
    written = tmp_path / target
    written.parent.mkdir(parents=True)
    written.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert len(written.read_text(encoding="utf-8")) > _REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS

    review = _blocking_review(
        _reviewed_finding(
            title="The superseded bulk orchestrator is still exported",
            description=(
                "The change adds a new flow while `bulkOrchestrator` is still exported; "
                "the superseded orchestrator must be deleted, not wrapped."
            ),
            recommendation="Delete the superseded export.",
        )
    )
    state = agent_state(
        tmp_path, [task_plan_artifact(), _prior_attempt_completion([target]), review]
    )
    state["retry_count"] = 1
    executor = _RecordingCodingExecutor({target: "module.exports = {};\n"})

    await EngineerAgent(prompt_loader=PromptLoader(), coding_executor=executor).run(state)

    snapshot = json.loads(executor.calls[0]["input_text"])["repository_context"]
    (entry,) = [item for item in snapshot["files"] if item["path"] == target]
    # Anchored at the definition, not at the head of the file: the window contains the
    # export the finding is about, and says which slice of the 700 lines it is.
    assert entry["excerpt_lines"]["start"] > 1
    assert entry["excerpt_lines"]["start"] <= 600 <= entry["excerpt_lines"]["end"]
    assert "bulkOrchestrator" in entry["content"]
    assert entry["total_lines"] == 700
    assert snapshot["required_excerpted_paths"] == [target]


@pytest.mark.asyncio
async def test_a_dropped_assigned_file_refuses_the_attempt_before_the_model_call(
    tmp_path: Path,
) -> None:
    """Now-2's refusal: the attempt whose own assignment is missing must not run.

    D2's shape: a file the plan assigns cannot be shown -- here it cannot even be decoded --
    and until now the attempt ran anyway, recorded the drop in a field nothing read, and
    spent its budget editing blind. The refusal must arrive before any model call, name the
    file and the reason, and carry its own classification so it never reads as "the platform
    did not anticipate" an exception type.
    """
    broken = tmp_path / "src" / "assigned.js"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"\xff\xfe\x00 not decodable as utf-8")
    plan = task_plan_artifact()
    scoped = plan.model_copy(
        update={
            "metadata": {
                **plan.metadata,
                "review_scope": {"expected_files_or_areas": ["src/assigned.js"]},
            }
        }
    )
    state = agent_state(tmp_path, [scoped])
    executor = _RecordingCodingExecutor({"src/assigned.js": "let repaired;\n"})

    with pytest.raises(RequiredContextRefusal) as raised:
        await EngineerAgent(prompt_loader=PromptLoader(), coding_executor=executor).run(state)

    # Refused before the model was called: the executor never ran, so no budget of any kind
    # was spent answering with guesses.
    assert executor.calls == []
    text = " ".join(raised.value.diagnostics)
    assert "src/assigned.js" in text
    assert "could not be read or decoded" in text
    assert "refused before the model was called" in text
    assert raised.value.failure_classification is FeatureFailureClassification.PLATFORM_DEFECT
    # The self-built sentences are what `safe_error_diagnostics` records verbatim.
    assert safe_error_diagnostics(raised.value) == raised.value.diagnostics


@pytest.mark.asyncio
async def test_a_benign_required_drop_does_not_refuse_and_is_recorded(tmp_path: Path) -> None:
    """The refusal's scope, from the other side: required is not the same as ordered.

    Required paths routinely include force-includes the plan never names -- here a module the
    assigned file imports -- and their absence is survivable. Dropping one must not stop the
    attempt: the model runs, and the drop is recorded with its reason so the record still
    answers "was that file in the snapshot" without re-deriving budget arithmetic. Only a
    dropped path the plan assigns or a blocking diagnostic names refuses (the test above).
    """
    importer = "const broken = require('./broken');\nmodule.exports = { run: () => broken };\n"
    page = tmp_path / "src" / "page.js"
    page.parent.mkdir(parents=True)
    page.write_text(importer, encoding="utf-8")
    # Required via the import trace, named by nothing: undecodable, so it drops.
    (tmp_path / "src" / "broken.js").write_bytes(b"\xff\xfe\x00 not decodable as utf-8")
    plan = task_plan_artifact()
    scoped = plan.model_copy(
        update={
            "metadata": {
                **plan.metadata,
                "review_scope": {"expected_files_or_areas": ["src/page.js"]},
            }
        }
    )
    state = agent_state(tmp_path, [scoped])
    executor = _RecordingCodingExecutor({"src/page.js": importer})

    update = await EngineerAgent(prompt_loader=PromptLoader(), coding_executor=executor).run(state)

    # The attempt ran: a benign drop is a recorded fact, not a stop.
    assert executor.calls
    completion = update["artifacts"][-1]
    assert completion.metadata["context_required_dropped"] == ["src/broken.js"]
    assert completion.metadata["context_required_dropped_reasons"] == {
        "src/broken.js": "unreadable"
    }
    assert "src/broken.js" in completion.metadata["context_required_omitted"]
    assert "src/broken.js" not in completion.metadata["context_required_excerpted"]


def test_the_snapshot_budget_derives_from_the_declared_window_with_the_identity_property() -> None:
    """T2: a declared 200k window is byte-identical to no declaration at all.

    The identity property is the safety proof: the default constants ARE what a 200_000-token
    window derives, so declaring a current-generation model changes nothing, and an undeclared
    model falls back to exactly those defaults -- absence of a declaration is never a
    punishment. The literals below are the one place outside the constant definitions they
    are allowed to appear (the T12 pin enforces that).
    """
    declared = repository_snapshot_budget("m", {"m": 200_000})
    default = repository_snapshot_budget("anything", {})

    assert (declared.max_characters, declared.per_file_max_characters) == (160_000, 16_000)
    assert (default.max_characters, default.per_file_max_characters) == (
        declared.max_characters,
        declared.per_file_max_characters,
    )
    assert (default.max_characters, default.per_file_max_characters) == (
        _REPOSITORY_CONTEXT_MAX_CHARACTERS,
        _REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS,
    )
    assert default.source == "default"
    wide = repository_snapshot_budget("gpt-6-astra", {"gpt-6-astra": 400_000})
    assert (wide.max_characters, wide.per_file_max_characters) == (320_000, 32_000)
    # The source names the model and the window, so a completion record says where the
    # budget came from without re-deriving it.
    assert wide.source == "declared:gpt-6-astra:400000"
    # The file-count cap is deliberately not part of the budget: its rationale (a repository
    # of tiny files becoming a long list) is model-independent and it is never scaled.
    assert not hasattr(wide, "max_files")


def test_budget_literals_appear_only_at_the_definitions_and_the_derivation_test() -> None:
    """T12: no test re-hardcodes the budget -- the 9baf758 lesson, pinned at grep level.

    Fixture sizes derive from the constants in force, so retuning the budget retunes every
    fixture with it. The literals may appear exactly at the constant definitions in the
    engineer and in the derivation test above (whose whole point is the identity property);
    anywhere else is a fixture that would silently stop meaning anything on the next retune.
    """
    server_root = Path(__file__).resolve().parents[1]
    total_literal = "160" + "_000"
    per_file_literal = "16" + "_000"
    agent_source = (server_root / "agents" / "engineer" / "agent.py").read_text(encoding="utf-8")
    assert agent_source.count(total_literal) == 1
    assert agent_source.count(per_file_literal) == 1
    # The tests 9baf758 re-derived: only the derivation test above may spell the numbers.
    own_source = Path(__file__).read_text(encoding="utf-8")
    assert own_source.count(total_literal) == 1
    assert own_source.count(per_file_literal) == 1
    seam_source = (server_root / "tests" / "test_channel_seam.py").read_text(encoding="utf-8")
    assert total_literal not in seam_source
    assert per_file_literal not in seam_source


@pytest.mark.asyncio
async def test_the_attempt_records_the_budget_it_ran_under(tmp_path: Path) -> None:
    """T3: the run-to-build-attribution instrument for budgets.

    A forensic pass must be able to tell "this attempt ran under the declared 400k budget"
    from "the declaration never crossed the compose boundary" by reading the completion --
    never by re-deriving arithmetic. A default run and a declared-window run record
    different sources and different numbers.
    """
    target = "src/page.js"
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / target).write_text("export const page = 1;\n", encoding="utf-8")

    state = agent_state(tmp_path, [task_plan_artifact()])
    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(file_updates={target: "export const page = 2;\n"}),
    ).run(state)
    default_metadata = update["artifacts"][-1].metadata
    assert default_metadata["context_budget_characters"] == _REPOSITORY_CONTEXT_MAX_CHARACTERS
    assert (
        default_metadata["context_budget_per_file_characters"]
        == _REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS
    )
    assert default_metadata["context_budget_source"] == "default"

    declared = repository_snapshot_budget("gpt-6-astra", {"gpt-6-astra": 400_000})
    state = agent_state(tmp_path, [task_plan_artifact()])
    update = await EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(file_updates={target: "export const page = 3;\n"}),
        snapshot_budget=declared,
    ).run(state)
    declared_metadata = update["artifacts"][-1].metadata
    assert declared_metadata["context_budget_characters"] == declared.max_characters
    assert (
        declared_metadata["context_budget_per_file_characters"] == declared.per_file_max_characters
    )
    assert declared_metadata["context_budget_source"] == "declared:gpt-6-astra:400000"
    assert declared_metadata["context_budget_source"] != default_metadata["context_budget_source"]


# AB-Feature-212's frontend attempt-0 change set, verbatim from the preserved artifacts
# (feature-d029f60d-e1a4-4b8b-b515-e175ba94fa67): the twelve files whose prior-attempt
# union evicted the two plan-assigned files below and killed the workstream at r1.
_212_PRIOR_CHANGED_PATHS = (
    "src/pages/AllApps.js",
    "src/apiUtils/allapps.apiUtils.js",
    "src/config/axios.js",
    "src/utils/allappsUtils.js",
    "src/utils/appCsvUtils.js",
    "src/components/apps/BulkAddApps.js",
    "src/utils/bulkAppsUtils.test.js",
    "src/apiUtils/allapps.apiUtils.test.js",
    "src/components/apps/BulkAddApps.test.js",
    "src/pages/AllApps.test.js",
    "src/components/apps/README.md",
    "src/utils/accessUtils.js",
)
# The two files the 61- refusal named when they lost the budget race, verbatim.
_212_ASSIGNED_PATHS = ("src/pages/AllApps.js", "src/components/apps/AddAppModal.js")


@pytest.mark.asyncio
async def test_the_212_shape_keeps_the_assigned_files_and_drops_only_advisory_tail(
    tmp_path: Path,
) -> None:
    """T4: the regression pin for the run that died at r1.

    Twelve prior-changed files plus two assigned files, each sized near the per-file bound,
    under the default budget: the ordered set (the plan's assignment) packs first, so both
    assigned files are IN the snapshot, every dropped path is advisory (a prior-attempt
    file), the refusal does not fire, and the executor is called. Before the ordered-first
    packing, the prior-changed union spent the budget and the refusal fired on the files it
    had evicted -- AB-Feature-212, both workstreams, platform_defect at r1.
    """
    filler_line = "const filler = 'x';\n"
    repeats = (_REPOSITORY_CONTEXT_PER_FILE_MAX_CHARACTERS - 200) // len(filler_line)
    for path in {*_212_PRIOR_CHANGED_PATHS, *_212_ASSIGNED_PATHS}:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(filler_line * repeats, encoding="utf-8")
    # The shape only reproduces if the union genuinely exceeds the budget.
    union = {*_212_PRIOR_CHANGED_PATHS, *_212_ASSIGNED_PATHS}
    assert len(union) * repeats * len(filler_line) > _REPOSITORY_CONTEXT_MAX_CHARACTERS

    plan = task_plan_artifact()
    scoped = plan.model_copy(
        update={
            "metadata": {
                **plan.metadata,
                "review_scope": {"expected_files_or_areas": list(_212_ASSIGNED_PATHS)},
            }
        }
    )
    review = _blocking_review(
        _reviewed_finding(
            title="The apps page must not render rows whose owner is missing.",
            description="Rows with a null owner render as undefined.",
            recommendation="Filter or annotate the ownerless rows.",
        )
    )
    state = agent_state(
        tmp_path,
        [scoped, _prior_attempt_completion(list(_212_PRIOR_CHANGED_PATHS)), review],
    )
    state["retry_count"] = 1
    executor = _RecordingCodingExecutor(
        {"src/pages/AllApps.js": "export const AllApps = () => null;\n"}
    )

    update = await EngineerAgent(prompt_loader=PromptLoader(), coding_executor=executor).run(state)

    # The attempt that died at r1 now runs: the model was called.
    assert executor.calls
    completion = update["artifacts"][-1]
    metadata = completion.metadata
    for assigned in _212_ASSIGNED_PATHS:
        assert assigned in metadata["context_file_paths"]
        assert assigned not in metadata["context_required_dropped"]
    # The budget still binds -- the tail of the prior-changed union is recorded as dropped,
    # and every drop is advisory: a prior-attempt file, never an ordered one.
    assert metadata["context_required_dropped"]
    assert set(metadata["context_required_dropped"]) <= (
        set(_212_PRIOR_CHANGED_PATHS) - set(_212_ASSIGNED_PATHS)
    )
    assert set(metadata["context_required_dropped_reasons"].values()) == {
        "required_budget_exhausted"
    }


@pytest.mark.asyncio
async def test_a_dropped_wiring_target_refuses_and_a_dropped_example_does_not(
    tmp_path: Path,
) -> None:
    """T5: a wiring repair's target is an order; its example is a file to imitate.

    An attempt that cannot see the file it was ordered to rewire answers with guesses, so a
    dropped `target_path` refuses before the model call exactly as a dropped assignment does.
    The `example_path` is advisory: its drop is recorded and the attempt runs -- the split
    citizenship 80- Part 2 introduced.
    """
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "host.js").write_bytes(b"\xff\xfe\x00 not decodable as utf-8")
    (tmp_path / "src" / "sibling.js").write_text("export const sibling = 1;\n", encoding="utf-8")
    plan = task_plan_artifact()
    wired = plan.model_copy(
        update={
            "metadata": {
                **plan.metadata,
                "retry_plan": {
                    "wiring_repairs": [
                        {"target_path": "src/host.js", "example_path": "src/sibling.js"}
                    ]
                },
            }
        }
    )
    state = agent_state(tmp_path, [wired])
    executor = _RecordingCodingExecutor({"src/host.js": "let repaired;\n"})

    with pytest.raises(RequiredContextRefusal) as raised:
        await EngineerAgent(prompt_loader=PromptLoader(), coding_executor=executor).run(state)

    assert executor.calls == []
    text = " ".join(raised.value.diagnostics)
    assert "src/host.js" in text
    assert "could not be read or decoded" in text
    assert "refused before the model was called" in text
    assert raised.value.failure_classification is FeatureFailureClassification.PLATFORM_DEFECT
    # The structured facts the partition wave clusters on ride the exception itself.
    assert raised.value.dropped_paths == ("src/host.js",)
    assert raised.value.budget_source == "default"

    # The inverse: an undecodable example file is recorded and survived.
    (tmp_path / "src" / "host.js").write_text("export const host = 1;\n", encoding="utf-8")
    (tmp_path / "src" / "example.js").write_bytes(b"\xff\xfe\x00 not decodable as utf-8")
    exampled = plan.model_copy(
        update={
            "metadata": {
                **plan.metadata,
                "retry_plan": {
                    "wiring_repairs": [
                        {"target_path": "src/host.js", "example_path": "src/example.js"}
                    ]
                },
            }
        }
    )
    state = agent_state(tmp_path, [exampled])
    executor = _RecordingCodingExecutor({"src/host.js": "let rewired;\n"})

    update = await EngineerAgent(prompt_loader=PromptLoader(), coding_executor=executor).run(state)

    assert executor.calls
    completion = update["artifacts"][-1]
    assert completion.metadata["context_required_dropped"] == ["src/example.js"]
    assert completion.metadata["context_required_dropped_reasons"] == {
        "src/example.js": "unreadable"
    }


def test_diagnostic_named_files_outrank_recency_inside_the_prior_attempt_cap() -> None:
    """Inside the bounded lineage list, the file the gate just rejected comes first."""
    recent_first = [f"src/module{index}.js" for index in range(20)]
    ordered = _prior_attempt_required_paths(recent_first, ("src/module19.js",))

    assert ordered[0] == "src/module19.js"
    assert len(ordered) == 12
    # Recency fills the rest: the oldest files are what the cap sheds.
    assert "src/module0.js" in ordered
    assert "src/module18.js" not in ordered


_PARSE_ERROR_DIAGNOSTIC = (
    "Pre-commit source validation failed (tool=npm; code=SOURCE_VALIDATION_FAILED_EXIT_1).\n"
    "src/service.js:40:23  error  Parsing error: Unexpected token ;"
)


@pytest.mark.asyncio
async def test_the_repair_prompt_quotes_the_region_the_diagnostic_points_at(
    tmp_path: Path,
) -> None:
    """A repair that can read `const nestedDocument;` beside the eslint line is one edit.

    176's three repair passes carried the diagnostic and nothing of the source it named, so
    every pass was a blind full-file regeneration that reproduced the same one-token defect.
    """
    lines = [f"function helper{index:02d}() {{ return {index}; }}" for index in range(80)]
    lines[39] = "  const nestedDocument;"
    broken = "\n".join(lines) + "\n"
    executor = _RecordingCodingExecutor(
        {"src/service.js": broken},
        {"src/service.js": broken.replace("const nestedDocument;", "let nestedDocument;")},
    )
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        source_formatter=_FormatterFailingUntil(1, [_PARSE_ERROR_DIAGNOSTIC]),
    )

    update = await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    assert update["artifacts"][-1].completion_status == "completed"
    repair_instructions = executor.calls[1]["instructions"]
    assert "const nestedDocument;" in repair_instructions
    assert "src/service.js, around line 40" in repair_instructions


@pytest.mark.asyncio
async def test_the_scoped_fix_role_repairs_when_configured_and_the_record_names_it(
    tmp_path: Path,
) -> None:
    """The repair pass is the SCOPED_FIX role's job, and the artifact says which model ran.

    Every per-tier `*_SCOPED_FIX_*` selection was dead configuration on this path -- the
    175-180 audit found the loop hard-wired to the primary coding client, and could not even
    establish that from the artifacts. The repair executes on the injected scoped-fix
    boundary, unjournaled exactly as before, and `source_repair_execution` records it.
    """
    primary = _RecordingCodingExecutor(
        {"src/service.js": "export const a = 1;\n"}, model="primary-coding-model"
    )
    scoped = _RecordingCodingExecutor(
        {"src/service.js": "export const a = 2;\n"}, model="scoped-fix-model"
    )
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=primary,
        scoped_fix_executor=scoped,
        source_formatter=_FormatterFailingUntil(1, [_LINT_DIAGNOSTIC]),
        operation_executor=cast(ExternalOperationExecutor, object()),
    )

    update = await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][-1]
    assert [len(primary.calls), len(scoped.calls)] == [1, 1]
    assert scoped.calls[0]["journaled"] is False
    recorded = completion.metadata["source_repair_execution"]
    assert recorded["model"] == "scoped-fix-model"
    assert recorded["scoped_fix_role_resolved"] is True


@pytest.mark.asyncio
async def test_an_unresolvable_scoped_fix_role_falls_back_to_the_primary_executor(
    tmp_path: Path,
) -> None:
    """No deployment loses the repair loop to a role that does not resolve -- and it says so."""
    primary = _RecordingCodingExecutor(
        {"src/service.js": "export const a = 1;\n"},
        {"src/service.js": "export const a = 2;\n"},
        model="primary-coding-model",
    )
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=primary,
        scoped_fix_executor=None,
        source_formatter=_FormatterFailingUntil(1, [_LINT_DIAGNOSTIC]),
    )

    update = await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][-1]
    assert len(primary.calls) == 2
    recorded = completion.metadata["source_repair_execution"]
    assert recorded["model"] == "primary-coding-model"
    assert recorded["scoped_fix_role_resolved"] is False


class _ScriptedReachabilityChecker:
    """Answer each inspection with one prepared outcome, and record what it was asked.

    A double rather than a real checkout, because the propositions below are about what the
    Engineer *does* with an answer -- repair it, record it, proceed past it -- and the answers
    themselves are established against real git in `tests/test_reachability.py`.
    """

    def __init__(self, *outcomes: ReachabilityOutcome) -> None:
        """Queue one outcome per expected inspection; the last one repeats."""
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, ...]] = []

    async def issues(
        self, workspace_root: Any, *, assigned_paths: Sequence[str] = ()
    ) -> ReachabilityOutcome:
        """Return the next queued outcome, recording the plan's assignment it was given."""
        del workspace_root
        self.calls.append(tuple(assigned_paths))
        return self._outcomes[min(len(self.calls) - 1, len(self._outcomes) - 1)]


_WIRING_ISSUE = (
    "src/added.js was added but no production file refers to it, so the running application "
    "cannot reach it. Edit src/index.js, which is where this repository registers modules of "
    "this kind."
)


@pytest.mark.asyncio
async def test_an_unreachable_addition_is_wired_by_a_repair_pass_inside_the_attempt(
    tmp_path: Path,
) -> None:
    """The in-attempt half of Part D: one finding, one unjournaled repair, one journaled write.

    The repair runs on the scoped-fix boundary, is told what actually rejected its work rather
    than being told the linter did, and is told not to touch the code the finding named -- that
    last one matters because this is the only repair in the agent whose fix is in a *different*
    file from the one the diagnostic is about.
    """
    primary = _RecordingCodingExecutor({"src/added.js": "export const b = 2;\n"})
    scoped = _RecordingCodingExecutor({"src/index.js": "export const c = 3;\n"})
    checker = _ScriptedReachabilityChecker(
        ReachabilityOutcome(issues=(_WIRING_ISSUE,)), ReachabilityOutcome()
    )
    plan = task_plan_artifact()
    assigned = plan.model_copy(
        update={
            "metadata": {
                **plan.metadata,
                "review_scope": {"expected_files_or_areas": ["src", "src/index.js"]},
            }
        }
    )
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=primary,
        scoped_fix_executor=scoped,
        reachability_checker=cast(Any, checker),
        operation_executor=cast(ExternalOperationExecutor, object()),
    )

    update = await agent.run(agent_state(tmp_path, [assigned]))

    completion = update["artifacts"][-1]
    assert completion.completion_status == "completed"
    # One implementation call, journaled; one wiring repair, not.
    assert primary.calls[0]["journaled"] is True
    assert [len(primary.calls), len(scoped.calls)] == [1, 1]
    assert scoped.calls[0]["journaled"] is False
    # Asked twice: once to find the defect and once to confirm the repair cleared it.
    assert len(checker.calls) == 2
    # Both layers are asked the identical question, so the plan's own list goes through whole
    # -- directories included, uncapped -- and not the file-only subset the snapshot uses.
    assert checker.calls[0] == ("src", "src/index.js")
    repair_instructions = scoped.calls[0]["instructions"]
    assert "cannot be reached when the application runs" in repair_instructions
    assert _WIRING_ISSUE in repair_instructions
    assert "Do not rewrite, rename, move or delete the code that cannot be reached" in (
        repair_instructions
    )
    assert "rejected by this repository's own pre-commit verification" not in repair_instructions
    # The host file is part of the attempt's change set, and nothing is left on the record as
    # unrepaired.
    assert "src/index.js" in completion.metadata["source_repair_paths"]
    assert "reachability_issues_unrepaired" not in completion.metadata


@pytest.mark.asyncio
async def test_a_wiring_finding_the_repair_cannot_clear_does_not_fail_the_attempt(
    tmp_path: Path,
) -> None:
    """The decision that keeps Part D free of behavioural risk, asserted rather than assumed.

    An unrepaired wiring finding must leave this agent exactly as it left it before Part D:
    `completed`, with the runtime's authoritative gate still to come. Failing here instead
    would classify it as a source-validation rejection and throw away `wiring_repairs` -- the
    hint that force-includes the host file in the next attempt's snapshot, and the fix for the
    third of the three gaps behind the 24% figure. The next attempt would be told to edit a
    file it had never been shown.

    It must also be *recorded*, or a reader comparing this `completed` against the gate's
    rejection cannot tell an in-attempt check that failed from one that never ran.
    """
    executor = _RecordingCodingExecutor(
        {"src/added.js": "export const b = 2;\n"}, {"src/added.js": "export const b = 3;\n"}
    )
    # The same finding every time: the repair rewrote something and changed nothing about the
    # answer, which is where this loop stops.
    checker = _ScriptedReachabilityChecker(ReachabilityOutcome(issues=(_WIRING_ISSUE,)))
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        reachability_checker=cast(Any, checker),
    )

    update = await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][-1]
    assert completion.completion_status == "completed"
    assert completion.metadata["reachability_issues_unrepaired"] == [_WIRING_ISSUE]
    # Stopped on the repeated finding rather than spending its whole ceiling on it.
    assert len(executor.calls) == 2


@pytest.mark.asyncio
async def test_the_candidates_the_inspection_never_examined_are_on_the_record(
    tmp_path: Path,
) -> None:
    """D3's second channel. A log line is not something anybody can look up later.

    The cap used to truncate silently, so a change adding more new modules than the bound read
    as "wiring passed". The Engineer records what was never asked about on the attempt's own
    artifact, so the absence of a finding can be told apart from the absence of a question.
    """
    checker = _ScriptedReachabilityChecker(
        ReachabilityOutcome(
            examined_candidates=("src/a.js",), dropped_candidates=("src/b.js", "src/c.js")
        )
    )
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=_RecordingCodingExecutor({"src/added.js": "export const b = 2;\n"}),
        reachability_checker=cast(Any, checker),
    )

    update = await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][-1]
    assert completion.completion_status == "completed"
    assert completion.metadata["reachability_candidates_dropped"] == ["src/b.js", "src/c.js"]
    assert "reachability_issues_unrepaired" not in completion.metadata


@pytest.mark.asyncio
async def test_an_inspection_that_raises_leaves_the_attempt_exactly_as_it_was(
    tmp_path: Path,
) -> None:
    """This check exists to make a known failure cheaper, never to invent a new way to fail."""

    class _BrokenChecker:
        async def issues(
            self, workspace_root: Any, *, assigned_paths: Sequence[str] = ()
        ) -> ReachabilityOutcome:
            del workspace_root, assigned_paths
            msg = "git is not available in this workspace"
            raise RuntimeError(msg)

    executor = _RecordingCodingExecutor({"src/added.js": "export const b = 2;\n"})
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        reachability_checker=cast(Any, _BrokenChecker()),
    )

    update = await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][-1]
    assert completion.completion_status == "completed"
    assert len(executor.calls) == 1
    assert "reachability_issues_unrepaired" not in completion.metadata
    assert "reachability_candidates_dropped" not in completion.metadata


class _ScriptedAssignedFileChecker:
    """Answer each placement inspection with one prepared finding set, and record the plan.

    A double for the same reason the reachability one is: what is at stake here is what the
    Engineer *does* with a placement finding, and the finding itself is established against
    real `git status` output in `tests/test_assigned_file_conformance.py`.
    """

    def __init__(self, *answers: tuple[str, ...]) -> None:
        """Queue one answer per expected inspection; the last one repeats."""
        self._answers = list(answers)
        self.calls: list[tuple[str, ...]] = []

    async def issues(
        self, workspace_root: Any, *, assigned_paths: Sequence[str] = ()
    ) -> tuple[str, ...]:
        """Return the next queued answer, recording the plan's assignment it was given."""
        del workspace_root
        self.calls.append(tuple(assigned_paths))
        return self._answers[min(len(self.calls) - 1, len(self._answers) - 1)]


_PLACEMENT_ISSUE = (
    "This workstream's plan assigns src/index.js, a file that already exists in this checkout "
    "and that this change does not touch. The change adds src/added.js in the same directory "
    "instead."
)


@pytest.mark.asyncio
async def test_a_module_written_beside_its_assigned_file_is_repaired_inside_the_attempt(
    tmp_path: Path,
) -> None:
    """49- Part D at this seam: the finding reaches the repair pass, with its own sentence.

    The cheapest wasted attempt in run 183 was a reviewer cycle spent saying the plan named a
    file and the implementation built a sibling. The repair is told that, in the words of the
    check that established it -- not the linter's, and not the wiring inspection's, whose
    instruction is the opposite one ("leave the added code where it is").
    """
    primary = _RecordingCodingExecutor({"src/added.js": "export const b = 2;\n"})
    scoped = _RecordingCodingExecutor({"src/index.js": "export const c = 3;\n"})
    placement = _ScriptedAssignedFileChecker((_PLACEMENT_ISSUE,), ())
    plan = task_plan_artifact()
    assigned = plan.model_copy(
        update={
            "metadata": {
                **plan.metadata,
                "review_scope": {"expected_files_or_areas": ["src", "src/index.js"]},
            }
        }
    )
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=primary,
        scoped_fix_executor=scoped,
        assigned_file_checker=cast(Any, placement),
        operation_executor=cast(ExternalOperationExecutor, object()),
    )

    update = await agent.run(agent_state(tmp_path, [assigned]))

    completion = update["artifacts"][-1]
    assert completion.completion_status == "completed"
    assert [len(primary.calls), len(scoped.calls)] == [1, 1]
    assert scoped.calls[0]["journaled"] is False
    # Asked the plan's own list, uncapped and directories included, exactly as the wiring
    # inspection is: one question, one answer, no second normalization idiom.
    assert placement.calls[0] == ("src", "src/index.js")
    repair_instructions = scoped.calls[0]["instructions"]
    assert "beside the file this workstream's plan named" in repair_instructions
    assert _PLACEMENT_ISSUE in repair_instructions
    # Not the wiring sentence, and not the wiring clause: this repair is told to move the
    # code, and that one is told never to.
    assert "cannot be reached when the application runs" not in repair_instructions
    assert "Do not rewrite, rename, move or delete" not in repair_instructions
    assert "rejected by this repository's own pre-commit verification" not in repair_instructions
    assert "assigned_file_issues_unrepaired" not in completion.metadata


@pytest.mark.asyncio
async def test_a_placement_finding_the_repair_cannot_clear_does_not_fail_the_attempt(
    tmp_path: Path,
) -> None:
    """The property that makes this part droppable: it informs, and it decides nothing.

    Nothing downstream of here treats a placement finding as a verdict -- no gate asks the
    question, and the reviewer remains the authority on where code belongs. What must survive
    is the *record*, because "this check fired and review approved the change anyway" is the
    evidence that would retire it, and a log line is not something anybody can count later.
    """
    executor = _RecordingCodingExecutor(
        {"src/added.js": "export const b = 2;\n"}, {"src/added.js": "export const b = 3;\n"}
    )
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        assigned_file_checker=cast(Any, _ScriptedAssignedFileChecker((_PLACEMENT_ISSUE,))),
    )

    update = await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][-1]
    assert completion.completion_status == "completed"
    assert completion.metadata["assigned_file_issues_unrepaired"] == [_PLACEMENT_ISSUE]
    # Stopped on the repeated finding rather than spending its whole ceiling on it.
    assert len(executor.calls) == 2


@pytest.mark.asyncio
async def test_an_unreachable_module_is_answered_before_a_placement_question(
    tmp_path: Path,
) -> None:
    """Two findings, contradictory instructions, so one of them has to go first.

    A wiring finding says the added module is finished and a host file needs the reference;
    a placement finding says the code belongs in a file the change never opened. Handed both,
    a pass can satisfy neither. Unreachable code is what the runtime's authoritative gate is
    waiting to reject, so it is answered first and the placement check is not even asked.
    """
    executor = _RecordingCodingExecutor(
        {"src/added.js": "export const b = 2;\n"}, {"src/index.js": "export const c = 3;\n"}
    )
    wiring = _ScriptedReachabilityChecker(
        ReachabilityOutcome(issues=(_WIRING_ISSUE,)), ReachabilityOutcome()
    )
    placement = _ScriptedAssignedFileChecker((_PLACEMENT_ISSUE,))
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        reachability_checker=cast(Any, wiring),
        assigned_file_checker=cast(Any, placement),
    )

    update = await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][-1]
    repair_instructions = executor.calls[1]["instructions"]
    assert _WIRING_ISSUE in repair_instructions
    assert _PLACEMENT_ISSUE not in repair_instructions
    assert "Do not rewrite, rename, move or delete the code that cannot be reached" in (
        repair_instructions
    )
    # Two wiring inspections and one placement inspection: the skipped one is the inspection
    # where something was unreachable, and this double was ready to answer it. Precedence is
    # "not asked", not "asked and discarded".
    assert len(wiring.calls) == 2
    assert len(placement.calls) == 1
    # The wiring finding is gone, and the placement one it uncovered rides out on the record
    # rather than buying a pass of its own. Neither issue set carries a `file:line`, so
    # `diagnostic_signature` extracts nothing and the loop reads that as unknown rather than
    # as changed -- the existing conservative rule, which this part does not touch.
    assert "reachability_issues_unrepaired" not in completion.metadata
    assert completion.metadata["assigned_file_issues_unrepaired"] == [_PLACEMENT_ISSUE]


@pytest.mark.asyncio
async def test_a_placement_inspection_that_raises_leaves_the_attempt_exactly_as_it_was(
    tmp_path: Path,
) -> None:
    """The narrowest check in the platform must not become a new way for an attempt to fail."""

    class _BrokenChecker:
        async def issues(
            self, workspace_root: Any, *, assigned_paths: Sequence[str] = ()
        ) -> tuple[str, ...]:
            del workspace_root, assigned_paths
            msg = "git is not available in this workspace"
            raise RuntimeError(msg)

    executor = _RecordingCodingExecutor({"src/added.js": "export const b = 2;\n"})
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        assigned_file_checker=cast(Any, _BrokenChecker()),
    )

    update = await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][-1]
    assert completion.completion_status == "completed"
    assert len(executor.calls) == 1
    assert "assigned_file_issues_unrepaired" not in completion.metadata


@pytest.mark.asyncio
async def test_the_plan_s_assigned_file_is_in_the_snapshot_however_the_task_is_worded(
    tmp_path: Path,
) -> None:
    """A file the plan assigns must not have to win a lexical ranking to be seen.

    Ranking scores the checkout against the task's own wording, so `src/pages/Profile.js`
    scores nothing against a task about build information. AB-Feature-166's console was
    assigned that page nine times over in its plan, wrote only the API helper it had thought
    of itself, and did the same thing twice more.
    """
    (tmp_path / "src" / "pages").mkdir(parents=True)
    (tmp_path / "src" / "pages" / "Profile.js").write_text(
        "export default function Profile() { return null; }\n", encoding="utf-8"
    )
    # Plenty of files whose names match the task's wording, so the ranked snapshot has
    # something to prefer over the assigned page.
    for index in range(12):
        (tmp_path / "src" / f"agents{index}.js").write_text(
            "export const agents = [];\n", encoding="utf-8"
        )

    plan = task_plan_artifact()
    assigned = plan.model_copy(
        update={
            "metadata": {
                **plan.metadata,
                "review_scope": {
                    "expected_files_or_areas": ["src/apiUtils", "src/pages/Profile.js", "src"],
                },
            }
        }
    )
    executor = _RecordingCodingExecutor({"src/apiUtils/health.js": "export const a = 1;\n"})
    agent = EngineerAgent(prompt_loader=PromptLoader(), coding_executor=executor)

    update = await agent.run(agent_state(tmp_path, [assigned]))

    seen = update["artifacts"][-1].metadata["context_file_paths"]
    assert "src/pages/Profile.js" in seen
    # The area entries are not files and are left to the ranked snapshot rather than
    # force-included, which for "src" would be the whole checkout.
    assert "src" not in seen
    assert "src/apiUtils" not in seen


@pytest.mark.asyncio
async def test_an_attempt_records_which_files_it_could_actually_read(tmp_path: Path) -> None:
    """ "Was the file it was told to edit in its snapshot?" must be answerable afterwards.

    It was the first question worth asking of a failed coding attempt and the one thing no
    record could answer: the snapshot was built, sent to the model, and never written down.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "existing.js").write_text("export const a = 1;\n", encoding="utf-8")
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=_RecordingCodingExecutor({"src/added.js": "export const b = 2;\n"}),
    )

    update = await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    metadata = update["artifacts"][-1].metadata
    assert "src/existing.js" in metadata["context_file_paths"]
    assert isinstance(metadata["context_omitted_file_count"], int)


def _evidence(**overrides: Any) -> dict[str, Any]:
    """Build one workspace-evidence summary of the shape the reviewer produces."""
    evidence: dict[str, Any] = {
        "limitations": [],
        "manual_review_required": False,
        "evidence_budget_exceeded": False,
        "oversized_paths": [],
        "unshown_changed_regions": [],
    }
    evidence.update(overrides)
    return evidence


def test_a_file_over_the_evidence_budget_earns_another_attempt_not_a_dead_feature() -> None:
    """A file too long for our own budget is not a decision somebody has to make.

    AB-Feature-168 ended here. Two generated files came in at 16,346 and 16,828 characters
    against a 16,000 limit -- over by 2% and 5% -- and truncation was bucketed with a leaked
    credential and a never-publish path. Both repositories stopped at `retry_count` 0 with all
    eight implementation retries unspent, on an attempt whose real defect was thirteen failing
    tests: exactly what those retries exist to fix.
    """
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}

    _apply_workspace_evidence_findings(
        payload,
        _evidence(
            limitations=["truncated_content"],
            evidence_budget_exceeded=True,
            oversized_paths=["src/components/apps/BulkAppImportModal.js"],
        ),
    )

    # Not approved -- the review genuinely did not see all of it -- but not terminal either.
    assert payload["verdict"] == "changes_requested"
    finding = payload["findings"][0]
    assert finding["finding_id"] == "REVIEW_EVIDENCE_INCOMPLETE"
    # Actionable: it names the file and says what to do, rather than reporting that this
    # platform has a limit. The engineer receives this as its instructions.
    assert finding["file_path"] == "src/components/apps/BulkAppImportModal.js"
    assert "BulkAppImportModal.js" in finding["description"]
    assert "smaller" in finding["recommendation"].lower()
    assert "manual review" not in finding["recommendation"].lower()
    assert finding["finding_category"] == "code_quality"
    # And it no longer asks for the one thing no attempt could do. Run 186 was told three
    # times to break up a 31 KB shared constants module it had added four lines to.
    assert "into smaller modules" not in finding["recommendation"]


def test_the_budget_finding_names_the_changed_lines_it_could_not_show() -> None:
    """A refusal on the change is only useful if it says which part of the change was hidden.

    The old sentence described the *file* -- too long, cut off before its end -- and the file's
    length was never the actionable fact. This one names the hunk, which is what the reviewer
    actually failed to see and the only thing the next attempt can do anything about.
    """
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}

    _apply_workspace_evidence_findings(
        payload,
        _evidence(
            limitations=["truncated_content"],
            evidence_budget_exceeded=True,
            oversized_paths=["src/const.js"],
            unshown_changed_regions=[{"path": "src/const.js", "start_line": 210, "end_line": 740}],
        ),
    )

    finding = payload["findings"][0]
    assert payload["verdict"] == "changes_requested"
    assert "src/const.js lines 210-740" in finding["description"]
    assert "into smaller modules" not in finding["recommendation"]


def test_withheld_or_unreadable_source_still_stops_for_a_person() -> None:
    """The distinction is the point: these are decisions no retry can make."""
    for limitation in ("sensitive_path", "redacted_content", "unavailable_content"):
        payload: dict[str, Any] = {"verdict": "approved", "findings": []}

        _apply_workspace_evidence_findings(
            payload, _evidence(limitations=[limitation], manual_review_required=True)
        )

        assert payload["verdict"] == "rejected", limitation
        finding = payload["findings"][0]
        assert "manual review" in finding["recommendation"].lower(), limitation
        assert finding["file_path"] is None, limitation


def test_a_change_with_too_many_files_is_also_actionable() -> None:
    """The file-count ceiling is the same kind of bound as the per-file one."""
    payload: dict[str, Any] = {"verdict": "approved", "findings": []}

    _apply_workspace_evidence_findings(
        payload, _evidence(limitations=["file_limit_exceeded"], evidence_budget_exceeded=True)
    )

    assert payload["verdict"] == "changes_requested"
    finding = payload["findings"][0]
    assert "fewer files" in finding["recommendation"]
    assert "manual review" not in finding["recommendation"].lower()


# Part C additions for tests/test_agents.py (appended after existing repair tests)


class _RuleChasingFormatter:
    """Judge the workspace's own bytes: rule one while `props` survives, then rule two.

    A state-holding double rather than a call counter, per overview section 4.4: what the
    loop must react to is the workspace, and a formatter that fails N times regardless of
    the bytes would pass a loop that never repaired anything.
    """

    def __init__(self, root: Path, path: str) -> None:
        self._root = root
        self._path = path
        self.calls: list[tuple[str, ...]] = []

    async def format_paths(
        self, repository_root: PathLike, paths: Sequence[str]
    ) -> tuple[str, ...]:
        del repository_root
        self.calls.append(tuple(paths))
        text = (self._root / self._path).read_text(encoding="utf-8")
        if "props" in text:
            raise SourceValidationError(
                (
                    "Pre-commit source validation failed (tool=eslint; "
                    "code=SOURCE_VALIDATION_FAILED_EXIT_1).\n"
                    f"{self._path}\n  12:9  error  use a render-result name  "
                    "testing-library/render-result-naming-convention",
                )
            )
        if "container.querySelector" in text:
            raise SourceValidationError(
                (
                    "Pre-commit source validation failed (tool=eslint; "
                    "code=SOURCE_VALIDATION_FAILED_EXIT_1).\n"
                    f"{self._path}\n  20:3  error  Avoid direct Node access  "
                    "testing-library/no-node-access",
                )
            )
        return ("eslint",)


@pytest.mark.asyncio
async def test_the_repair_loop_clears_two_rules_in_one_attempt(tmp_path: Path) -> None:
    """Clearing rule one surfaces rule two, and both are cleared inside the attempt.

    AB-Feature-171's frontend went `render-result-naming` -> `no-node-access` ->
    `no-unnecessary-act`, and with one repair per attempt each remaining rule cost a full
    re-implementation cycle. The loop is bounded by changed diagnostics, not by count.
    """
    executor = _RecordingCodingExecutor(
        {"src/view.test.js": "const props = render();\ncontainer.querySelector('x');\n"},
        {"src/view.test.js": "const view = render();\ncontainer.querySelector('x');\n"},
        {"src/view.test.js": "const view = render();\nview.getByRole('button');\n"},
    )
    formatter = _RuleChasingFormatter(tmp_path, "src/view.test.js")
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        source_formatter=formatter,
    )

    update = await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    completion = update["artifacts"][-1]
    assert completion.completion_status == "completed"
    # One implementation call, then exactly one repair pass per surfaced rule.
    assert len(executor.calls) == 3
    assert "render-result-naming-convention" in executor.calls[1]["instructions"]
    assert "no-node-access" in executor.calls[2]["instructions"]
    assert completion.metadata["source_repair_passes"] == 2
    # The workspace holds the doubly-repaired bytes; the gate itself said yes.
    assert "view.getByRole" in (tmp_path / "src" / "view.test.js").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_repeating_diagnostic_signature_stops_the_repair_loop_on_the_spot(
    tmp_path: Path,
) -> None:
    """A signature that does not change is a defect the loop cannot act on."""

    class _UnmovedFormatter:
        def __init__(self) -> None:
            self.calls = 0

        async def format_paths(
            self, repository_root: PathLike, paths: Sequence[str]
        ) -> tuple[str, ...]:
            del repository_root, paths
            self.calls += 1
            raise SourceValidationError((_LINT_DIAGNOSTIC,))

    executor = _RecordingCodingExecutor(
        {"src/service.js": "export const a = 1;\n"},
        {"src/service.js": "export const a = 2;\n"},
        {"src/service.js": "export const a = 3;\n"},
    )
    formatter = _UnmovedFormatter()
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        source_formatter=formatter,
    )

    with pytest.raises(SourceValidationError, match="pre-commit validation") as rejected:
        await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    # The pass that ran is on the rejected attempt's own record. It used to be discarded with
    # the rejection: `source_repairs=()` was hard-coded on this path, so a gate-rejected
    # attempt that had repaired four times looked exactly like one whose loop never ran.
    failed = rejected.value.code_completion
    assert failed is not None
    metadata = failed.metadata
    assert metadata["source_validation_rejected"] is True
    assert metadata["source_repair_engaged"] is True
    assert metadata["source_repair_passes"] == 1
    assert metadata["source_repair_paths"] == ["src/service.js"]
    assert metadata["source_repair_execution"]["scoped_fix_role_resolved"] is False
    assert metadata["source_repair_scoped_fix_role_resolved"] is False

    # One implementation call and exactly one repair pass: the second verification
    # reported the same signature, so no further pass was granted.
    assert len(executor.calls) == 2
    assert formatter.calls == 2


@pytest.mark.asyncio
async def test_the_repair_loop_never_consumes_a_non_mechanical_diagnostic(
    tmp_path: Path,
) -> None:
    """A test failure belongs to the retry policy and the reviewer, never to this loop."""

    class _TestFailureFormatter:
        async def format_paths(
            self, repository_root: PathLike, paths: Sequence[str]
        ) -> tuple[str, ...]:
            del repository_root, paths
            raise SourceValidationError(
                ("npm test exited with code 1.\nAssertionError: expected the route to answer",)
            )

    executor = _RecordingCodingExecutor(
        {"src/service.js": "export const a = 1;\n"},
        {"src/service.js": "export const a = 2;\n"},
    )
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        source_formatter=_TestFailureFormatter(),
    )

    with pytest.raises(SourceValidationError) as rejected:
        await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    # The implementation call only. No repair pass was spent on a diagnostic that is not
    # the gate's own lint output.
    assert len(executor.calls) == 1
    # And the record says the loop did not engage, which is a different fact from having no
    # record: this is the shape a reader has to be able to tell apart from four spent passes.
    failed = rejected.value.code_completion
    assert failed is not None
    metadata = failed.metadata
    assert metadata["source_validation_rejected"] is True
    assert metadata["source_repair_engaged"] is False
    assert metadata["source_repair_passes"] == 0
    # The scoped-fix wiring is reported even with nothing to attribute a pass to -- "was the
    # boundary resolved at all?" is exactly the question three runs' artifacts could not answer.
    assert metadata["source_repair_scoped_fix_role_resolved"] is False
    assert "source_repair_execution" not in metadata


@pytest.mark.asyncio
async def test_the_repair_loop_has_a_hard_pass_ceiling(tmp_path: Path) -> None:
    """A pathological linter whose output never stabilises cannot hold the attempt forever."""

    class _EndlesslyNovelFormatter:
        def __init__(self) -> None:
            self.calls = 0

        async def format_paths(
            self, repository_root: PathLike, paths: Sequence[str]
        ) -> tuple[str, ...]:
            del repository_root, paths
            self.calls += 1
            raise SourceValidationError(
                (
                    "Pre-commit source validation failed (tool=eslint; "
                    "code=SOURCE_VALIDATION_FAILED_EXIT_1).\n"
                    f"src/service.js\n  {self.calls}:1  error  rule number {self.calls}  "
                    f"plugin/rule-{self.calls}",
                )
            )

    updates = [{"src/service.js": f"export const a = {index};\n"} for index in range(9)]
    executor = _RecordingCodingExecutor(*updates)
    formatter = _EndlesslyNovelFormatter()
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        source_formatter=formatter,
    )

    with pytest.raises(SourceValidationError):
        await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    # One implementation call plus at most the ceiling's worth of repair passes.
    assert len(executor.calls) == 1 + _MAX_SOURCE_REPAIR_PASSES


@pytest.mark.asyncio
async def test_the_repair_instruction_permits_the_asked_change_and_forbids_the_rest(
    tmp_path: Path,
) -> None:
    """Both halves of the §4.1 contract, asserted on the delivered instruction."""
    executor = _RecordingCodingExecutor(
        {"src/service.js": "export const a = 1;\n"},
        {"src/service.js": "export const a = 2;\n"},
    )
    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=executor,
        source_formatter=_FormatterFailingUntil(1, [_LINT_DIAGNOSTIC]),
    )

    await agent.run(agent_state(tmp_path, [task_plan_artifact()]))

    delivered = executor.calls[1]["instructions"]
    assert "when a diagnostic demands a rename, rename precisely what it names" in delivered
    assert "do not rename or restructure anything the diagnostics did not name" in delivered
    assert "do not restructure it, do not rename anything" not in delivered


@pytest.mark.asyncio
async def test_a_regenerated_lockfile_larger_than_the_text_limit_still_completes(
    tmp_path: Path,
) -> None:
    """The completion's write-verification checks presence, never text-size readability.

    AB-Feature-174's console died as `platform_defect` here: its attempt changed
    package.json, the dependency sync regenerated a 1.3 MB package-lock.json, the platform
    appended that lockfile to the change set, and the verification read rejected it for
    exceeding `max_workspace_file_bytes`. A lockfile is platform-appended generated output;
    the commit stages bytes, not decoded text.
    """
    (tmp_path / "package.json").write_text('{"name": "web"}\n', encoding="utf-8")
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")

    class OversizedLockfileSynchronizer:
        async def sync(self, repository_root: PathLike, paths: Sequence[str]) -> tuple[str, ...]:
            del paths
            lockfile = Path(repository_root) / "package-lock.json"
            padding = '"x": "' + "y" * 1_200_000 + '"'
            lockfile.write_text('{"lockfileVersion": 3, ' + padding + "}\n", encoding="utf-8")
            return ("npm install",)

    agent = EngineerAgent(
        prompt_loader=PromptLoader(),
        coding_executor=MockCodingExecutor(
            file_updates={
                "package.json": '{"name": "web", "devDependencies": {"jest": "^29.0.0"}}\n'
            }
        ),
        dependency_synchronizer=OversizedLockfileSynchronizer(),
    )

    update = await agent.run(agent_state(tmp_path, [task_plan_artifact()]))
    completion = update["artifacts"][0]

    assert isinstance(completion, CodeCompletionArtifact)
    changed = {change.path for change in completion.file_changes}
    assert "package-lock.json" in changed, "the oversized lockfile is still part of the change"


@pytest.mark.asyncio
async def test_a_claimed_write_that_is_not_in_the_workspace_still_fails_loudly(
    tmp_path: Path,
) -> None:
    """Presence is still enforced: an executor claiming a phantom write is a typed failure."""

    class PhantomWriteExecutor(MockCodingExecutor):
        def __init__(self) -> None:
            super().__init__(file_updates={"src/real.py": "value = 1\n"})

        async def execute(self, **kwargs: Any) -> Any:
            result = await super().execute(**kwargs)
            (Path(kwargs["workspace_root"]) / "src" / "real.py").unlink()
            return result

    agent = EngineerAgent(prompt_loader=PromptLoader(), coding_executor=PhantomWriteExecutor())

    with pytest.raises(AgentArtifactError):
        await agent.run(agent_state(tmp_path, [task_plan_artifact()]))


# ------------------------------------------------------------------- model JSON envelopes


def test_a_fenced_json_response_parses_to_the_same_object_the_bare_form_would() -> None:
    """One surrounding Markdown fence changes nothing about the answer, so it costs nothing.

    AB-Feature-179 died in planning twice over exactly this class of envelope: "planner
    response is not valid JSON", first attempt and repair alike, with nothing recorded of
    what was rejected.
    """
    from agents.shared.contracts import parse_model_json

    bare = parse_model_json('{"answers": []}', expected_keys=("answers",))
    fenced = parse_model_json('```json\n{"answers": []}\n```', expected_keys=("answers",))
    bare_fence = parse_model_json('```\n{"answers": []}\n```', expected_keys=("answers",))

    assert bare == fenced == bare_fence == {"answers": []}


def test_prose_around_one_json_object_is_tolerated_and_the_key_contract_still_holds() -> None:
    """A sentence of preamble is recoverable; prose quoting the wrong shape is still refused."""
    from agents.shared.contracts import parse_model_json

    recovered = parse_model_json(
        'Here is the plan you asked for:\n{"answers": [{"question_id": "q-1"}]}\nHope it helps.',
        expected_keys=("answers",),
    )
    assert recovered["answers"] == [{"question_id": "q-1"}]

    # The slice passes json.loads but not the caller's key contract: still an error.
    with pytest.raises(AgentArtifactError, match="exactly"):
        parse_model_json('Note: {"unrelated": true}', expected_keys=("answers",))


def test_a_rejected_response_failure_carries_a_redacted_head_of_what_was_rejected() -> None:
    """The failure names what actually came back, so a dead planning run is diagnosable.

    Whether 179's rejected planner text was a fence or garbage was unknowable afterwards
    because nothing of it was persisted; the head in the error is what removes that hole.
    """
    from agents.shared.contracts import parse_model_json

    with pytest.raises(AgentArtifactError) as rejected:
        parse_model_json("I could not produce the plan you asked for.")

    assert "I could not produce the plan" in str(rejected.value)


def test_genuinely_broken_text_is_still_rejected() -> None:
    """Tolerance stops at recovery: text with no complete JSON object in it stays an error."""
    from agents.shared.contracts import parse_model_json

    with pytest.raises(AgentArtifactError, match="must be valid JSON"):
        parse_model_json('{"answers": [truncated mid tok')


_218_PLAN = ["server/services/app", "package.json"]


def _bulk_app_checkout(root: Path) -> tuple[set[Path], str]:
    """AB-Feature-218's shape: a service under an assigned AREA, importing the file it reuses."""
    (root / "server" / "services" / "app").mkdir(parents=True)
    (root / "server" / "validation").mkdir(parents=True)
    changed = "server/services/app/bulkApp.service.js"
    (root / changed).write_text(
        "const appValidation = require('../../validation/app.validation');\n"
        "const resolve = () => appValidation.addAppSchema.body;\n",
        encoding="utf-8",
    )
    reused = Path("server") / "validation" / "app.validation.js"
    (root / reused).write_text(
        "const addAppSchema = Joi.object({ appDetails: Joi.object({}) });\n"
        "module.exports = { addAppSchema };\n",
        encoding="utf-8",
    )
    (root / "package.json").write_text('{"name": "admanager"}\n', encoding="utf-8")
    return {Path(changed), reused, Path("package.json")}, changed


def test_an_assigned_area_shows_the_file_128_and_218_were_told_to_reuse(tmp_path: Path) -> None:
    """One failure, twice, in the same repository and on the same expression.

    AB-Feature-128's backend wrote `appValidation.addAppSchema.body.validate(row)` and spent
    five of eleven attempts on `TypeError: addAppSchema.body is undefined`. AB-Feature-218's
    backend wrote the same expression and spent four attempts on the same defect, and
    `server/validation/app.validation.js` was in `context_file_paths` on none of them.

    The tier meant to prevent it was wired and correctly ordered; it was handed nothing to
    scan. 218's plan named nine directories and two JSON files, `_assigned_context_paths`
    dropped every directory, and the import scan's highest-priority source was
    `package.json` and `package-lock.json` -- two files with no module specifiers between
    them. An assigned AREA now contributes the change's own files under it, so the scan
    reads the service the change is about instead of falling through to the lineage's
    incidental first-seen order.
    """
    existing, changed = _bulk_app_checkout(tmp_path)
    reused = "server/validation/app.validation.js"

    sources = _assigned_scan_sources(_218_PLAN, [changed])
    imported = _imported_repository_paths(
        [changed], existing, WorkspaceFileTools(tmp_path), "", sources
    )
    required = _required_context_paths({"expected_files_or_areas": _218_PLAN}, imported.paths)

    assert changed in sources
    assert reused in imported.paths
    assert reused in required


def test_an_assigned_area_the_change_never_touched_contributes_nothing(tmp_path: Path) -> None:
    """An area is not a directory walk. "Every file under `server/utils`" is the checkout."""
    existing, changed = _bulk_app_checkout(tmp_path)
    (tmp_path / "server" / "helpers").mkdir(parents=True)
    untouched = Path("server") / "helpers" / "unrelated.js"
    (tmp_path / untouched).write_text("module.exports = {};\n", encoding="utf-8")
    existing.add(untouched)
    plan = ["server/services/app", "server/helpers", "server/utils"]

    sources = _assigned_scan_sources(plan, [changed])
    required = _required_context_paths({"expected_files_or_areas": plan}, ())

    assert sources == (changed,)
    assert untouched.as_posix() not in sources
    # No directory reaches the required set, on this path or any other.
    assert required == ()


def test_a_plan_that_names_files_scans_exactly_what_it_scanned_before(tmp_path: Path) -> None:
    """The byte-identical clause: an area-free plan must not notice this change at all."""
    _existing, changed = _bulk_app_checkout(tmp_path)
    plan = ["server/services/app/bulkApp.service.js", "package.json", "./server/app.js"]

    assert _assigned_scan_sources(plan, [changed]) == _assigned_context_paths(plan)
    assert _assigned_scan_sources(plan, []) == _assigned_context_paths(plan)


def test_an_assigned_area_never_adds_a_path_to_the_ordered_set(tmp_path: Path) -> None:
    """The safety invariant, asserted directly: no attempt that passes today fails tomorrow.

    `_refuse_if_required_context_dropped` refuses when an ORDERED file is dropped, and
    ordered files come from `_assigned_context_paths`. An area contributes scan sources and
    never requirements, so this part cannot newly trip that refusal however many files the
    change has under the plan's directories.
    """
    _existing, changed = _bulk_app_checkout(tmp_path)
    plan = ["server/services/app", "server/validation", "package.json"]

    before = _assigned_context_paths(plan)
    sources = _assigned_scan_sources(plan, [changed, "server/validation/other.validation.js"])
    required = _required_context_paths({"expected_files_or_areas": plan}, ())

    assert before == ("package.json",)
    assert required == ("package.json",)
    # Sources genuinely grew; the ordered set did not.
    assert len(sources) > len(before)
    assert not (set(sources) - {"package.json"}) & set(required)


def test_an_assigned_area_is_bounded_like_the_assignment_it_came_from() -> None:
    """The expansion spends the assignment's existing claim on the budget, not a new one."""
    plan = ["server/services/app"]
    changed = [f"server/services/app/file_{index:02d}.js" for index in range(20)]

    sources = _assigned_scan_sources(plan, changed)

    assert len(sources) == _MAX_ASSIGNED_CONTEXT_PATHS
