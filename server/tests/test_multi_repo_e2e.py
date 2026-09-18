"""End-to-end multi-repository features driven through the live child executor.

The existing feature API test substitutes a stub child executor, so it verifies parent
plumbing only. These tests run the real per-repository path across mixed technology
stacks, which is what actually failed in production: twelve consecutive live runs, a
frontend that never executed once, and not a single pull request created.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from agents.shared.contracts import create_artifact
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import (
    ChildWorkflowResultArtifact,
    IntegrationContractArtifact,
    PullRequestArtifact,
    TechnicalPRDArtifact,
)
from configs.settings import load_settings
from services.cancellation import MockCancellationToken
from services.feature_runtime import LiveChildWorkstreamExecutor
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from tests.fixtures import node_javascript_repository, python_fastapi_repository
from tests.test_live_child_executor import RecordingProcessRunner, ScriptedLLMClient
from workflows.feature_workflow import (
    FeatureWorkflowOrchestrator,
    WorkstreamPublicationClass,
    feature_is_at_rest,
    publication_is_held,
    require_publishable_workstream,
)


@pytest.mark.asyncio
async def test_a_node_backend_and_python_service_are_validated_with_their_own_toolchains(
    tmp_path: Path,
) -> None:
    """Test 2: each repository runs its own stack's commands, never the other's.

    Both children execute in one parallel group and both reach a pull request, which is
    the outcome twelve live runs never produced.
    """
    harness = await _prepare(tmp_path, {"backend": "node", "service": "python"})
    runner = RecordingProcessRunner()

    result = await _run_feature(harness, process_runner=runner, engineer_files=_engineer_files())

    assert result.status is FeatureWorkflowStatus.COMPLETED
    assert {child.status for child in result.child_workflows.values()} == {
        ChildWorkflowStatus.COMPLETED
    }
    pull_requests = [
        artifact for artifact in result.artifacts if isinstance(artifact, PullRequestArtifact)
    ]
    assert len(pull_requests) == 2
    _assert_every_reviewed_repository_published(result)

    executed = {" ".join(command) for command in runner.commands}
    node_commands = {command for command in executed if command.startswith("npm")}
    python_commands = {
        command for command in executed if command.split(" ")[0] in {"ruff", "pytest", "mypy", "uv"}
    }
    assert node_commands, "the Node repository must run its configured npm scripts"
    assert python_commands, "the Python repository must run its configured Python tooling"
    # The cross-stack mistake: neither toolchain may leak into the other's workspace.
    assert not any("ruff" in command or "pytest" in command for command in node_commands)
    assert not any(command.startswith("npm") for command in python_commands)


@pytest.mark.asyncio
async def test_a_failing_backend_still_lets_an_independent_service_finish(
    tmp_path: Path,
) -> None:
    """The production failure, reproduced end to end through the real child executor.

    The backend never satisfies its implementation expectation, but the unrelated service
    must still be implemented, reviewed and completed rather than left pending.
    """
    harness = await _prepare(tmp_path, {"backend": "node", "service": "python"})

    result = await _run_feature(
        harness,
        process_runner=RecordingProcessRunner(),
        # The backend only ever edits tests, so its expectation is never satisfied.
        engineer_files={"backend": _tests_only_files(), "service": _python_files()},
    )

    assert result.status is FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN
    assert result.child_workflows["backend"].status is ChildWorkflowStatus.FAILED
    # Nothing opens while a required repository is unfinished: a service pull request whose
    # backend does not exist is not reviewable work. The finished service is held instead,
    # and the branch that carries it is already on the remote either way.
    assert not [
        artifact for artifact in result.artifacts if isinstance(artifact, PullRequestArtifact)
    ]
    assert publication_is_held(result)
    assert feature_is_at_rest(result)
    assert require_publishable_workstream(result, repository_id="service") is (
        WorkstreamPublicationClass.REVIEWED
    )
    # The reviewed commit really is on the remote, which is what makes the offer honest --
    # judged by what the bare repository holds, not by a call having returned.
    remote_head = _git_output(
        harness["workspaces"]["service"], "ls-remote", "origin", harness["branches"]["service"]
    )
    assert remote_head, "the service's reviewed commit must already be pushed"

    published = await FeatureWorkflowOrchestrator().publish_feature(
        result,
        requested_by="an operator (via platform key)",
        reason="the backend is not going to land and this work should be reviewed",
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
    )

    assert published.child_workflows["service"].status is ChildWorkflowStatus.COMPLETED
    opened = [
        artifact for artifact in published.artifacts if isinstance(artifact, PullRequestArtifact)
    ]
    assert [artifact.repository for artifact in opened] == ["example/service"]
    _assert_every_reviewed_repository_published(published)


def _assert_every_reviewed_repository_published(result: Any) -> None:
    """Hold the real path to the rule the parent tests state over mock dependencies.

    Twelve consecutive live runs ended with no pull request at all, so the guarantee is
    worth asserting where the child executor, the workspaces and Git are real.
    """
    reviewed = {
        artifact.repository_id
        for artifact in result.artifacts
        if isinstance(artifact, ChildWorkflowResultArtifact)
        and artifact.status == "approved"
        and artifact.pull_request_readiness
    }
    published = {
        repository_id
        for repository_id, child in result.child_workflows.items()
        if child.pull_request_artifact_id is not None
    }
    assert reviewed <= published, (
        "reviewed work was left on the remote without a pull request: "
        f"{sorted(reviewed - published)}"
    )


async def _prepare(tmp_path: Path, repositories: dict[str, str]) -> dict[str, Any]:
    """Create one real checkout per repository plus the durable journal and settings."""
    workspace_root = tmp_path / "workspaces"
    builders = {"node": node_javascript_repository, "python": python_fastapi_repository}
    branches: dict[str, str] = {}
    workspaces: dict[str, Path] = {}
    for repository_id, technology in repositories.items():
        workspace = workspace_root / "feature-multi" / repository_id
        builders[technology](workspace)
        remote = tmp_path / f"{repository_id}.git"
        subprocess.run(("git", "init", "--bare", str(remote)), check=True, capture_output=True)
        _git(workspace, "remote", "add", "origin", str(remote))
        _git(workspace, "push", "origin", "main")
        branch = f"ai/feature-multi/{repository_id}/server-status"
        _git(workspace, "checkout", "-b", branch)
        branches[repository_id] = branch
        workspaces[repository_id] = workspace

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'multi.db'}")
    await database.create_schema()
    return {
        "database": database,
        "journal": ExternalOperationJournal(database),
        "settings": load_settings(workspace_root=workspace_root),
        "branches": branches,
        "workspaces": workspaces,
        "repositories": repositories,
    }


async def _run_feature(
    harness: dict[str, Any],
    *,
    process_runner: RecordingProcessRunner,
    engineer_files: dict[str, list[dict[str, str]]],
) -> Any:
    """Drive the real orchestrator over the real child executor for every repository."""
    executor = _PerRepositoryExecutor(harness, engineer_files=engineer_files, runner=process_runner)
    request = StartFeatureRequest.model_validate(_payload(harness["repositories"]))
    state = _initial_feature_state("feature-multi", request)
    orchestrator = FeatureWorkflowOrchestrator(child_executor=executor)
    try:
        return await orchestrator.start(
            state, credentials=RequestScopedCredentials(openai_api_key=None, github_token=None)
        )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()


class _PerRepositoryExecutor:
    """Route each repository to the real live executor with its own scripted provider."""

    def __init__(
        self,
        harness: dict[str, Any],
        *,
        engineer_files: dict[str, list[dict[str, str]]],
        runner: RecordingProcessRunner,
    ) -> None:
        """Bind the shared harness and the per-repository engineer output."""
        self._harness = harness
        self._engineer_files = engineer_files
        self._runner = runner

    async def run(self, **kwargs: Any) -> Any:
        """Execute one repository through LiveChildWorkstreamExecutor against its checkout."""
        repository_id = kwargs["repository"].repository_id
        child = kwargs["child"].model_copy(
            update={
                "workspace_path": str(self._harness["workspaces"][repository_id]),
                "branch_name": self._harness["branches"][repository_id],
                # Provisioning is skipped so the prepared checkout is used instead of a
                # network clone; everything after it is the genuine live path.
                "retry_count": max(kwargs["child"].retry_count, 1),
            }
        )
        engineer = ScriptedLLMClient(
            [
                {
                    "summary": f"Implement the scoped work for {repository_id}.",
                    "files": self._engineer_files[repository_id],
                }
            ]
        )
        reviewer = ScriptedLLMClient([_review_payload(repository_id, kwargs["workstream"])])
        executor = LiveChildWorkstreamExecutor(
            settings=self._harness["settings"],
            git_environment={},
            engineer_client=engineer,
            reviewer_client=reviewer,
            journal=self._harness["journal"],
            cancellation_token=MockCancellationToken(),
            process_runner=self._runner,
        )
        return await executor.run(**{**kwargs, "child": child})


def _engineer_files() -> dict[str, list[dict[str, str]]]:
    """Return a complete implementation for every repository in the feature."""
    return {"backend": _node_files(), "service": _python_files()}


def _node_files() -> list[dict[str, str]]:
    """Return a route plus its test, satisfying the Node workstream expectation."""
    return [
        {
            "path": "src/routes/status.js",
            "content": "module.exports = { statusRoute: () => ({ status: 'ok' }) };\n",
        },
        {
            "path": "test/status.test.js",
            "content": "const { test } = require('node:test');\ntest('ok', () => {});\n",
        },
    ]


def _python_files() -> list[dict[str, str]]:
    """Return a module plus its test, satisfying the Python workstream expectation."""
    return [
        {
            "path": "app/status_feed.py",
            "content": (
                '"""Server status feed."""\n\n\n'
                "def status_feed() -> dict[str, str]:\n"
                '    """Return the current status feed."""\n'
                '    return {"status": "ok"}\n'
            ),
        },
        {
            "path": "tests/test_status_feed.py",
            "content": (
                '"""Status feed tests."""\n\n'
                "from app.status_feed import status_feed\n\n\n"
                "def test_status_feed() -> None:\n"
                '    """The feed reports a healthy service."""\n'
                '    assert status_feed() == {"status": "ok"}\n'
            ),
        },
    ]


def _tests_only_files() -> list[dict[str, str]]:
    """Return a tests-only change, which cannot satisfy a production expectation."""
    return [
        {
            "path": "test/status.test.js",
            "content": "const { test } = require('node:test');\ntest('todo', () => {});\n",
        }
    ]


def _review_payload(repository_id: str, workstream: Any) -> dict[str, Any]:
    """Return an approving review covering exactly the requirements this child was assigned."""
    return {
        "verdict": "approved",
        "summary": f"{repository_id} implements its scoped requirements.",
        # The reviewer must answer for its assigned scope exactly: no more, no fewer.
        "requirement_checks": [
            {
                "requirement_id": reference.requirement_id,
                "passed": True,
                "evidence": "The scoped source area contains the implementation.",
            }
            for reference in workstream.scoped_requirements
        ],
        "findings": [],
        "architecture_assessment": "The change follows the existing repository layout.",
        "security_assessment": "No credential handling changed.",
        "test_coverage_assessment": "The repository's configured tests cover the change.",
    }


def _payload(repositories: dict[str, str]) -> dict[str, Any]:
    """Return one PRD assigning every repository exactly one scoped requirement."""
    return {
        "feature_id": "feature-multi",
        "prd": {
            "title": "Server status",
            "problem_statement": "Operators cannot see server status across services.",
            "goals": ["Expose server status."],
            "user_stories": [
                {
                    "story_id": "status",
                    "persona": "Operator",
                    "need": "see server status",
                    "benefit": "faster triage",
                    "acceptance_criteria": ["Status is visible."],
                }
            ],
            "requirements": [
                {
                    "requirement_id": f"{repository_id}-status",
                    "description": f"Implement the {repository_id} side of server status.",
                    "priority": "must",
                    "acceptance_criteria": [f"The {repository_id} status work is complete."],
                    "dependencies": [],
                }
                for repository_id in repositories
            ],
            "constraints": [],
            "out_of_scope": [],
            "stakeholders": ["Platform"],
        },
        "repositories": [
            {
                "repository_id": repository_id,
                "name": repository_id,
                "role": "backend" if repository_id == "backend" else "service",
                "repository_url": f"https://github.com/example/{repository_id}.git",
                "default_branch": "main",
            }
            for repository_id in repositories
        ],
    }


def _technical_prd(repositories: dict[str, str]) -> TechnicalPRDArtifact:
    """Return a technical PRD mirroring the submitted per-repository requirements."""
    return create_artifact(
        TechnicalPRDArtifact,
        workflow_id="feature-multi",
        artifact_id="002_technical_prd.json",
        producer="product_manager",
        metadata={},
        payload={
            "title": "Server status",
            "solution_summary": "Expose server status from every participating service.",
            "functional_requirements": [
                {
                    "requirement_id": f"{repository_id}-status",
                    "description": f"Implement the {repository_id} side of server status.",
                    "priority": "must",
                    "acceptance_criteria": [f"The {repository_id} status work is complete."],
                    "dependencies": [],
                }
                for repository_id in repositories
            ],
            "non_functional_requirements": [],
            "data_requirements": [],
            "integration_requirements": [],
            "security_requirements": [],
            "assumptions": [],
            "unresolved_questions": [],
        },
    )


def _contract(repositories: dict[str, str]) -> IntegrationContractArtifact:
    """Return an approved contract owned by every participating repository."""
    return create_artifact(
        IntegrationContractArtifact,
        workflow_id="feature-multi",
        artifact_id="009_integration_contract.json",
        producer="feature_planner",
        metadata={},
        payload={
            "feature_id": "feature-multi",
            "contract_version": "1.0.0",
            "status": "approved",
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
                "rollback_requirements": ["Revert the coordinated pull requests."],
            },
            "owning_workstreams": sorted(repositories),
            "approved_at": datetime(2026, 8, 4, tzinfo=UTC),
            "openapi_document": None,
        },
    )


def _git(root: Path, *arguments: str) -> None:
    """Run one git command against a fixture checkout with no shell."""
    subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True, timeout=30)


def _git_output(root: Path, *arguments: str) -> str:
    """Return one git command's stdout, so an assertion can read what Git actually holds."""
    finished = subprocess.run(
        ("git", *arguments), cwd=root, check=True, capture_output=True, timeout=30, text=True
    )
    return finished.stdout.strip()
