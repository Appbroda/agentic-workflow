"""Driving `LiveChildWorkstreamExecutor.run` against a real checkout with real subprocesses.

Audit risk P1-8, third tier. `test_live_child_executor.py` already runs the child path
against on-disk fixtures, but every command it plans is answered by a double that returns
exit zero. Selection was therefore proven and *behaviour* was not: a repository whose lint
script cannot run, whose test suite never returns, or whose install refuses its own lockfile
all looked identical to a healthy one.

Here nothing about the checkout is doubled. `npm ci`, `npm run lint`, `node --test`, `ruff`,
`pytest` and every `git` invocation are executed by `AsyncioProcessRunner`, which is the
production runner. Only the provider boundary is scripted, and only because a deterministic
Engineer is what makes a specific gate reachable on purpose.

Three things this module exists to make cheap:

* **A first attempt without a network clone.** `run()` provisions only when
  `retry_count == 0`, and provisioning insists on an HTTPS github.com source. That is the
  same field the baseline validation measurement keys off, so a suite that always used a
  retry could never reach the baseline gate at all. `seed_completed_clone` journals a
  *completed* clone for the exact operation the provisioner is about to ask for, so the
  reuse path the platform already has for a worker that died after cloning is what supplies
  the checkout. Nothing is stubbed: the reuse is the deployment's own.
* **A clone that actually runs.** The reuse above needs the destination to already hold a
  checkout, so it cannot serve a scenario whose whole subject is provisioning one --
  AB-Feature-190's retry, which had no workspace at all. `LocalRemoteCloneRunner` rewrites
  that one declared URL to the bare repository this harness already pushes to, and the
  clone is then real in every other respect.
* **Working directories as an effect.** `RecordedRealProcessRunner` records the directory
  each command actually ran in, not the one a plan named. Overview section 4.4 is explicit
  that a test asserting the argument vector and not the worktree is worse than no test, and
  a monorepo's whole failure mode is a correct command in the wrong directory.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agents.shared.contracts import create_artifact
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import (
    IntegrationContractArtifact,
    RepositoryWorkstreamPlan,
    TechnicalPRDArtifact,
)
from configs.settings import Settings, load_settings
from services.cancellation import CancellationToken, MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.feature_runtime import LiveChildWorkstreamExecutor
from services.process_runner import AsyncioProcessRunner, ProcessResult
from state.enums import ChildWorkflowStatus
from state.external_operations import ExternalOperationType
from state.feature_models import ChildWorkflowReference
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal, OperationResult
from tests.test_live_child_executor import ScriptedLLMClient
from workflows.feature_workflow import ChildExecution

FEATURE_ID = "feature-real"
REPOSITORY_ID = "backend"
BRANCH = "ai/feature-real/backend/server-status"
# The branch these repositories are configured to build from, and the one the fixture's
# remote serves as HEAD. Named once because three places have to agree about it: the
# repository spec below, the bare remote's `symbolic-ref`, and the clone `seed_completed_clone`
# journals -- and when they drifted, the seeded key stopped matching and the reuse path
# silently gave way to a real clone into a directory the fixture had already built.
DEFAULT_BRANCH = "main"
CREDENTIALS = RequestScopedCredentials(openai_api_key=None, github_token=None)

# What the feature declares as this repository's origin, and therefore what the provisioner
# asks Git for. Not reached by default -- it exists so the seeded clone below carries the same
# idempotency key the provisioner will compute -- and rewritten to the on-disk origin by
# `LocalRemoteCloneRunner` for the scenarios that need the clone itself to run.
REPOSITORY_URL = "https://github.com/example/backend.git"
_CLONE_URL = "https://x-access-token@github.com/example/backend.git"


class RecordedRealProcessRunner:
    """The production process runner, plus what each command was and where it ran.

    Not a double. Every command executes; the record exists because "the monorepo's Python
    tooling never ran in the Node package" is a claim about a working directory, and the
    plan alone cannot distinguish a directory that was chosen from one that was reached.
    """

    def __init__(self) -> None:
        """Start with an empty record and the production runner underneath."""
        self._runner = AsyncioProcessRunner()
        self.executed: list[tuple[tuple[str, ...], Path]] = []

    async def run(
        self,
        command: Sequence[str],
        cwd: Path,
        timeout_seconds: float,
        cancellation_token: CancellationToken,
        environment: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        """Record the command and its directory, then run it for real."""
        self.executed.append((tuple(command), Path(cwd)))
        return await self._runner.run(
            command, cwd, timeout_seconds, cancellation_token, environment
        )

    def clone_targets(self) -> list[Path]:
        """Return the destination of every `git clone` that actually ran."""
        return [
            Path(command[-1])
            for command, _cwd in self.executed
            if command[:2] == ("git", "clone") and len(command) >= 4
        ]

    def directories_for(self, executable: str, *, relative_to: Path) -> set[str]:
        """Return every directory one executable actually ran in, relative to the checkout."""
        return {
            _relative(directory, relative_to)
            for command, directory in self.executed
            if command and Path(command[0]).name == executable
        }

    def ran(self, *prefix: str) -> int:
        """How many recorded commands start with these arguments."""
        return len([command for command, _cwd in self.executed if command[: len(prefix)] == prefix])


class LocalRemoteCloneRunner(RecordedRealProcessRunner):
    """The production runner, with the provisioner's remote pointed at the local one.

    Provisioning insists on an HTTPS github.com source, which is the right guard and is not
    negotiable -- so a scenario that needs a *real* fresh clone has nowhere to clone from,
    and `seed_completed_clone` cannot help because the reuse path requires the destination
    to already hold a checkout.

    The one substitution is the URL, on exactly the declared origin, replaced by the bare
    repository this harness already pushes to. Everything about the clone stays real: real
    `git clone`, real objects, real working tree, real branch creation on top of it. Any
    command that is not that clone is passed straight through.
    """

    def __init__(self, remote: Path) -> None:
        """Bind the on-disk origin the declared remote is rewritten to."""
        super().__init__()
        self._remote = remote

    async def run(
        self,
        command: Sequence[str],
        cwd: Path,
        timeout_seconds: float,
        cancellation_token: CancellationToken,
        environment: Mapping[str, str] | None = None,
    ) -> ProcessResult:
        """Run the command, substituting the local origin for the declared clone source."""
        arguments = tuple(command)
        if arguments[:2] == ("git", "clone"):
            arguments = tuple(
                str(self._remote) if item == _CLONE_URL else item for item in arguments
            )
        return await super().run(arguments, cwd, timeout_seconds, cancellation_token, environment)


@dataclass(slots=True)
class RealRepositoryHarness:
    """One real checkout, one real bare remote, and the durable journal beside them."""

    workspace: Path
    workspace_root: Path
    remote: Path
    branch: str
    database: Database
    journal: ExternalOperationJournal
    settings: Settings
    runner: RecordedRealProcessRunner = field(default_factory=RecordedRealProcessRunner)

    async def dispose(self) -> None:
        """Drop the journal this harness owned, whatever the test did with it."""
        await self.database.drop_schema()
        await self.database.dispose()

    def remote_branches(self) -> dict[str, str]:
        """Return what the bare remote actually holds, which is the effect of a push."""
        listing = subprocess.run(
            ("git", "for-each-ref", "--format=%(refname:short) %(objectname)", "refs/heads"),
            cwd=self.remote,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        entries = (line.split(" ", 1) for line in listing.splitlines() if line.strip())
        return {name: sha for name, sha in entries}

    def worktree_file(self, path: str) -> str | None:
        """Read one file out of the checkout, or None when this attempt never wrote it."""
        candidate = self.workspace / path
        return candidate.read_text(encoding="utf-8") if candidate.is_file() else None

    def tracked_paths(self) -> set[str]:
        """Return every path Git currently tracks on the working branch."""
        listing = subprocess.run(
            ("git", "ls-files"),
            cwd=self.workspace,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        return {line for line in listing.splitlines() if line.strip()}


async def build_harness(
    tmp_path: Path,
    build: Any,
    *,
    settings_overrides: dict[str, Any] | None = None,
    checkout_branch: bool = True,
    local_remote_clone: bool = False,
) -> RealRepositoryHarness:
    """Build the checkout, give it a real local origin, and open its journal.

    The remote is a bare repository on disk. Push and pull-request publication therefore
    have somewhere real to land, and nothing in this tier can reach a hosting provider even
    if a credential were somehow present -- which is overview section 4.8's requirement, not
    a convenience.
    """
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / FEATURE_ID / REPOSITORY_ID
    build(workspace)
    remote = tmp_path / "origin.git"
    subprocess.run(
        ("git", "init", "--bare", str(remote)), check=True, capture_output=True, timeout=30
    )
    _git(workspace, "remote", "add", "origin", str(remote))
    _git(workspace, "push", "origin", DEFAULT_BRANCH)
    # `git init --bare` points HEAD at whatever this Git's default branch name is, which is
    # not the branch the fixture pushed -- so a clone of this remote warns that HEAD refers
    # to a nonexistent ref and checks nothing out. A hosted remote never looks like that,
    # and a scenario that clones from here is entitled to the same shape it would get live.
    subprocess.run(
        ("git", "--git-dir", str(remote), "symbolic-ref", "HEAD", f"refs/heads/{DEFAULT_BRANCH}"),
        check=True,
        capture_output=True,
        timeout=30,
    )
    if checkout_branch:
        _git(workspace, "checkout", "-b", BRANCH)

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'journal.db'}")
    await database.create_schema()
    return RealRepositoryHarness(
        workspace=workspace,
        workspace_root=workspace_root,
        remote=remote,
        branch=BRANCH,
        database=database,
        journal=ExternalOperationJournal(database),
        settings=load_settings(workspace_root=workspace_root, **(settings_overrides or {})),
        runner=(
            LocalRemoteCloneRunner(remote) if local_remote_clone else RecordedRealProcessRunner()
        ),
    )


async def seed_completed_clone(harness: RealRepositoryHarness) -> None:
    """Journal the clone the provisioner is about to ask for as already done.

    Written through the real `ExternalOperationExecutor`, with the operation type, logical
    step and safe input the provisioner supplies, so the key is computed the same way rather
    than reproduced here. The action does nothing because the checkout is already present;
    what matters is that the journal then holds a `SUCCEEDED` clone, and the executor's
    first-attempt path reuses it exactly as it does for a worker that died after cloning.
    """

    async def already_cloned() -> tuple[Path, OperationResult]:
        return harness.workspace, OperationResult(
            external_reference=str(harness.workspace),
            payload={
                "workspace_path": str(harness.workspace),
                "source_url": _CLONE_URL,
                "base_branch": DEFAULT_BRANCH,
            },
        )

    await ExternalOperationExecutor(
        journal=harness.journal,
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(
            workflow_id=FEATURE_ID,
            feature_id=FEATURE_ID,
            child_workflow_id=f"{FEATURE_ID}:{REPOSITORY_ID}",
            repository_id=REPOSITORY_ID,
        ),
    ).run(
        operation_type=ExternalOperationType.CLONE_REPOSITORY,
        logical_step="clone_repository",
        # Every field the provisioner puts in, because the key is computed from all of
        # them: a seed missing one is a seed for a different operation, and the reuse it
        # exists to exercise quietly does not happen.
        safe_input={
            "workspace_path": str(harness.workspace),
            "source_url": _CLONE_URL,
            "base_branch": DEFAULT_BRANCH,
        },
        action=already_cloned,
        max_attempts=3,
    )


async def run_child(
    harness: RealRepositoryHarness,
    *,
    engineer_payload: dict[str, Any],
    review_payload: dict[str, Any] | None = None,
    workstream: RepositoryWorkstreamPlan | None = None,
    contract: IntegrationContractArtifact | None = None,
    retry_count: int = 1,
    prior_artifacts: Sequence[Any] = (),
    retry_strategy: dict[str, Any] | None = None,
    engineer_client: ScriptedLLMClient | None = None,
    reviewer_client: ScriptedLLMClient | None = None,
    # The boundary the in-attempt source repair runs on, standing in for the SCOPED_FIX
    # role's resolved client. Omitted, the repair falls back to the engineer client exactly
    # as a deployment whose scoped-fix role does not resolve.
    scoped_fix_client: Any | None = None,
    # The boundary the one in-attempt self-review runs on, standing in for the CODING
    # role's resolved client. Omitted, no self-review runs -- both what every pre-Part-E
    # scenario encodes and what a deployment whose coding role cannot resolve for it does.
    self_review_client: Any | None = None,
    child_overrides: dict[str, Any] | None = None,
    event_writer: Any | None = None,
) -> ChildExecution:
    """Run one child workstream end to end against this harness's real checkout."""
    executor = LiveChildWorkstreamExecutor(
        settings=harness.settings,
        git_environment={},
        engineer_client=engineer_client or ScriptedLLMClient([engineer_payload]),
        reviewer_client=reviewer_client
        or ScriptedLLMClient([review_payload] if review_payload else []),
        journal=harness.journal,
        cancellation_token=MockCancellationToken(),
        process_runner=harness.runner,
        # The realrepo tier's recording runner executes every command for real, so handing
        # it to the Git adapter as well changes nothing about what runs -- and it is what
        # lets `LocalRemoteCloneRunner` point one clone at the on-disk origin.
        git_process_runner=harness.runner,
        scoped_fix_client_factory=(
            (lambda _decision: scoped_fix_client) if scoped_fix_client is not None else None
        ),
        self_review_client_factory=(
            (lambda _decision: self_review_client) if self_review_client is not None else None
        ),
        event_writer=event_writer,
    )
    request = StartFeatureRequest.model_validate(feature_payload())
    feature = _initial_feature_state(FEATURE_ID, request)
    technical_prd = technical_prd_artifact()
    active_contract = contract or contract_artifact()
    feature.artifacts.extend([technical_prd, active_contract])
    feature.artifacts.extend(prior_artifacts)
    repository = next(
        item for item in feature.repository_specs if item.repository_id == REPOSITORY_ID
    )
    child = ChildWorkflowReference(
        child_workflow_id=f"{FEATURE_ID}:{REPOSITORY_ID}",
        repository_id=REPOSITORY_ID,
        workstream_id=REPOSITORY_ID,
        branch_name=harness.branch,
        workspace_path=str(harness.workspace),
        status=ChildWorkflowStatus.RUNNING,
        retry_count=retry_count,
        retry_strategy=retry_strategy,
        **(child_overrides or {}),
    )
    return await executor.run(
        feature=feature,
        repository=repository,
        workstream=workstream or workstream_plan(),
        child=child,
        technical_prd=technical_prd,
        contract=active_contract,
        feedback=[],
        credentials=CREDENTIALS,
    )


# --------------------------------------------------------------------------------------
# The stable planning inputs every scenario shares
# --------------------------------------------------------------------------------------


def workstream_plan(
    *,
    expected_source_areas: list[str] | None = None,
    expected_change_categories: list[str] | None = None,
) -> RepositoryWorkstreamPlan:
    """Scope this repository to one requirement it is expected to satisfy in `src/routes`."""
    return RepositoryWorkstreamPlan.model_validate(
        {
            "workstream_id": REPOSITORY_ID,
            "repository_id": REPOSITORY_ID,
            "role": "backend",
            "requirement_ids": ["backend-status-api"],
            "scoped_requirements": [
                {
                    "requirement_id": "backend-status-api",
                    "acceptance_criterion_ids": ["backend-status-api:ac-1"],
                    "responsibility": "implements",
                }
            ],
            "out_of_scope_requirements": [],
            "shared_requirements": [],
            "responsibilities": ["Serve the server status payload."],
            "task_ids": ["backend-status-task"],
            "dependency_workstream_ids": [],
            "contract_sections_consumed": [],
            "contract_sections_implemented": [],
            "acceptance_criteria": ["The status route responds."],
            "test_requirements": ["Run the configured test command."],
            "documentation_requirements": [],
            "expected_files_or_areas": expected_source_areas or ["src/routes"],
            "required": True,
            "implementation_expectations": [
                {
                    "requirement_id": "backend-status-api",
                    "expected_change_categories": expected_change_categories or ["route"],
                    "expected_source_areas": expected_source_areas or ["src/routes"],
                    "tests_required": False,
                }
            ],
        }
    )


def technical_prd_artifact() -> TechnicalPRDArtifact:
    """Return the technical PRD the child is given as read-only context."""
    return create_artifact(
        TechnicalPRDArtifact,
        workflow_id=FEATURE_ID,
        artifact_id="002_technical_prd.json",
        producer="product_manager",
        metadata={},
        payload={
            "title": "Server status",
            "solution_summary": "Expose the server status.",
            "functional_requirements": [
                {
                    "requirement_id": "backend-status-api",
                    "description": "Serve a server status payload.",
                    "priority": "must",
                    "acceptance_criteria": ["The status route responds."],
                    "dependencies": [],
                }
            ],
            "non_functional_requirements": [],
            "data_requirements": [],
            "integration_requirements": [],
            "security_requirements": [],
            "assumptions": [],
            "unresolved_questions": [],
        },
    )


def contract_artifact(
    *, openapi_document: dict[str, Any] | None = None
) -> IntegrationContractArtifact:
    """Return the approved contract, optionally one that projects an OpenAPI document."""
    return create_artifact(
        IntegrationContractArtifact,
        workflow_id=FEATURE_ID,
        artifact_id="009_integration_contract.json",
        producer="feature_planner",
        metadata={},
        payload={
            "feature_id": FEATURE_ID,
            "contract_version": "1.0.0",
            "status": "approved",
            "api_style": "rest" if openapi_document else "none",
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
                "rollback_requirements": ["Revert the pull request."],
            },
            "owning_workstreams": [REPOSITORY_ID],
            "approved_at": datetime(2026, 8, 4, tzinfo=UTC),
            "openapi_document": openapi_document,
        },
    )


def feature_payload() -> dict[str, object]:
    """Return a single-repository feature request matching the checkout this tier builds."""
    return {
        "feature_id": FEATURE_ID,
        "prd": {
            "title": "Server status",
            "problem_statement": "Operators cannot see server status.",
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
                    "requirement_id": "backend-status-api",
                    "description": "Serve a server status payload.",
                    "priority": "must",
                    "acceptance_criteria": ["The status route responds."],
                    "dependencies": [],
                }
            ],
            "constraints": [],
            "out_of_scope": [],
            "stakeholders": ["Platform"],
        },
        "repositories": [
            {
                "repository_id": REPOSITORY_ID,
                "name": "Backend",
                "role": "backend",
                "repository_url": REPOSITORY_URL,
                "default_branch": DEFAULT_BRANCH,
            }
        ],
    }


def review_payload(*, verdict: str, findings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return a schema-valid reviewer response with the verdict a scenario needs."""
    return {
        "verdict": verdict,
        "summary": "The repository implements its scoped status requirement.",
        "requirement_checks": [
            {
                "requirement_id": "backend-status-api",
                "passed": verdict == "approved",
                "evidence": "src/routes/status.js serves the status payload.",
            }
        ],
        "findings": findings or [],
        "architecture_assessment": "The route follows the existing repository layout.",
        "security_assessment": "No credential or input handling changed.",
        "test_coverage_assessment": "The configured test command covers the route.",
    }


def _git(root: Path, *arguments: str) -> None:
    """Run one git command against a checkout with no shell."""
    subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True, timeout=30)


def _relative(directory: Path, root: Path) -> str:
    """Describe one working directory relative to the checkout, or absolutely if outside it."""
    try:
        return directory.resolve().relative_to(root.resolve()).as_posix() or "."
    except ValueError:
        return str(directory)


__all__ = [
    "BRANCH",
    "CREDENTIALS",
    "FEATURE_ID",
    "REPOSITORY_ID",
    "REPOSITORY_URL",
    "RealRepositoryHarness",
    "RecordedRealProcessRunner",
    "build_harness",
    "contract_artifact",
    "feature_payload",
    "review_payload",
    "run_child",
    "seed_completed_clone",
    "technical_prd_artifact",
    "workstream_plan",
]
