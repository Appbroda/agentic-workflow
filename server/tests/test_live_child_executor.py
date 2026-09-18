"""Coverage for the live child workstream executor against real repository checkouts.

This path previously had no tests at all: every child-level test substituted a stub
executor, so the code that actually runs during a live feature was exercised only in
production. Each phase consequently fixed one symptom and revealed the next. Only the
OpenAI boundary is doubled here; git, preflight, validation planning and completeness all
run for real against on-disk fixtures.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import update as sqlalchemy_update

from adapters.git_adapter import EmptyCommitError
from adapters.interruptible_git import InterruptibleGitService, reviewed_content_fingerprint
from adapters.llm_adapter import ImageInput, LLMResponse
from agents.engineer.agent import EngineerAgent
from agents.engineer.publisher import require_reviewed_workspace_match
from agents.reviewer.agent import _redaction_is_preexisting
from agents.shared.contracts import AgentArtifactError, create_artifact
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from artifacts.schemas import (
    ChildWorkflowResultArtifact,
    CodeCompletionArtifact,
    IntegrationContractArtifact,
    RepositoryWorkstreamPlan,
    ReviewArtifact,
    TechnicalPRDArtifact,
)
from configs.model_roles import AgentPlatform, ModelRole
from configs.settings import load_settings
from services.cancellation import MockCancellationToken
from services.external_operations import (
    ExternalOperationExecutor,
    ExternalOperationScope,
    UnknownExternalOperation,
)
from services.feature_runtime import (
    AmbiguousPublicationReceiptError,
    LiveChildWorkstreamExecutor,
    LiveCrossRepositoryDiffs,
    RepositoryWorkspaceError,
    _preserved_on_reset,
)
from services.process_runner import ProcessResult
from state.enums import ChildWorkflowStatus
from state.external_operations import ExternalOperationStatus, ExternalOperationType
from state.failure_diagnosis import FeatureFailureClassification
from state.feature_models import ChildWorkflowReference
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from storage.models import ExternalOperationModel
from tests.fixtures import init_git_repository, node_javascript_repository
from tests.fixtures.real_repositories import python_uv_repository
from tools.model_routing import ModelExecutionMode, ModelRoutingDecision
from tools.repository_preflight import (
    PreflightIssue,
    RepositoryPreflight,
    RepositoryPreflightResult,
    _dependency_installers,
)
from tools.review_fix_classification import ReviewFixComplexity
from tools.scoped_tests import scoped_test_diagnostic
from tools.source_formatting import SourceValidationError
from tools.technology_detection import calculate_repository_revision_sync
from tools.unchanged_failure import failing_evidence_files
from tools.validation_tools import (
    ValidationResult,
    _summary_with_references,
    validation_summary,
)
from workflows.feature_workflow import (
    _execution_artifacts,
    _failed_child_execution,
    _has_unpublished_approved_child,
    _latest_approved_result,
    _with_attempt_identity,
)


class ScriptedLLMClient:
    """Return queued provider payloads without any network access."""

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        """Queue one JSON payload per expected model call."""
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
        """Pop the next scripted payload and record what the agent actually asked for."""
        self.calls.append((instructions, input_text))
        payload = self._payloads.pop(0) if self._payloads else {}
        return LLMResponse(
            output_text=json.dumps(payload),
            model="test-model",
            response_id=f"response-{len(self.calls)}",
            input_tokens=0,
            output_tokens=0,
        )


class CrashOnceLLMClient(ScriptedLLMClient):
    """Lose the first provider result before an artifact can be persisted."""

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
        """Raise once at the reviewer boundary, then return the queued response."""
        if not self.calls:
            self.calls.append((instructions, input_text))
            raise RuntimeError("simulated worker crash before review persistence")
        return await super().respond(instructions=instructions, input_text=input_text)


class RecordingProcessRunner:
    """Succeed deterministically while recording every command the platform chose.

    Real package managers are not available or network-reachable in the default suite, so
    the runner is doubled. Command *selection* is still the platform's real decision, which
    is exactly what must be asserted.
    """

    def __init__(self) -> None:
        """Track each executed command and its working directory."""
        self.commands: list[tuple[str, ...]] = []

    async def run(self, command: Any, cwd: Path, *_args: Any, **_kwargs: Any) -> ProcessResult:
        """Return a successful result without invoking the underlying tool."""
        self.commands.append(tuple(command))
        return ProcessResult(
            command=tuple(command),
            return_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.01,
        )


class RealGitProcessRunner(RecordingProcessRunner):
    """Run `git` for real while still doubling package managers and build tools.

    Git is local, fast, and deterministic, so a gate that reads the checkout's own diff has
    to see the actual repository to be tested at all. Everything else stays doubled for the
    reasons the base runner documents.
    """

    async def run(self, command: Any, cwd: Path, *args: Any, **kwargs: Any) -> ProcessResult:
        result = await super().run(command, cwd, *args, **kwargs)
        if tuple(command)[:1] != ("git",):
            return result
        completed = subprocess.run(tuple(command), cwd=cwd, capture_output=True, text=True)
        return ProcessResult(
            command=tuple(command),
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=0.01,
        )


class FailedResetProcessRunner(RecordingProcessRunner):
    """Fail the destructive reset so the executor must stop on a dirty checkout."""

    async def run(self, command: Any, cwd: Path, *_args: Any, **_kwargs: Any) -> ProcessResult:
        result = await super().run(command, cwd)
        if tuple(command) == ("git", "reset", "--hard", "HEAD"):
            return ProcessResult(
                command=tuple(command),
                return_code=128,
                stdout="",
                stderr="fatal: cannot reset the index",
                duration_seconds=0.01,
            )
        return result


class SecretDiffProcessRunner(RecordingProcessRunner):
    """Return credential-shaped rejected source from the retry diff boundary."""

    async def run(self, command: Any, cwd: Path, *_args: Any, **_kwargs: Any) -> ProcessResult:
        result = await super().run(command, cwd)
        if tuple(command) == ("git", "diff", "--cached"):
            return ProcessResult(
                command=tuple(command),
                return_code=0,
                stdout=(
                    "+client = OpenAI(api_key='sk-proj-RESETSECRET123456789')\n"
                    "+token = 'gho_RESETSECRET123456789'\n"
                    "+options = {'clientSecret': 'V7mQ2xL9pR4sT8wY3kN6dF1z'}\n"
                ),
                stderr="",
                duration_seconds=0.01,
            )
        return result


@pytest.mark.asyncio
async def test_live_child_implements_a_javascript_repository_and_is_approved(
    tmp_path: Path,
) -> None:
    """The full live child path succeeds against a real Node checkout."""
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the server status route and controller.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": (
                            "function statusRoute(req, res) {\n"
                            "  res.json({ status: 'ok', uptime: process.uptime() });\n"
                            "}\n\nmodule.exports = { statusRoute };\n"
                        ),
                    },
                    {
                        "path": "src/controllers/status.js",
                        "content": (
                            "function statusController() {\n  return { status: 'ok' };\n}\n\n"
                            "module.exports = { statusController };\n"
                        ),
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\n"
                            "const assert = require('node:assert');\n"
                            "const { statusRoute } = require('../src/routes/status');\n\n"
                            "test('status route responds', () => {\n"
                            "  assert.ok(typeof statusRoute === 'function');\n"
                            "});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    execution = await _run(harness, engineer=engineer, reviewer=reviewer)

    assert execution.result.status == "approved"
    assert execution.code_completion is not None
    assert "src/routes/status.js" in execution.result.production_files_changed
    # Validation was planned from the checkout, so no Python tool can appear.
    commands = {
        tuple(item.get("command", []))[:1]
        for item in (execution.result.validation_plan or {}).get("commands", [])
    }
    assert commands and all(command in {("npm",)} for command in commands)


@pytest.mark.asyncio
async def test_feature_resume_after_push_recovers_before_coding_or_review(tmp_path: Path) -> None:
    """A lost child checkpoint reuses its receipt without another model invocation."""
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Update the server status route.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": (
                            "function statusRoute(req, res) {\n"
                            "  res.json({ status: 'ok', uptime: process.uptime() });\n"
                            "}\n\nmodule.exports = { statusRoute };\n"
                        ),
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\n"
                            "const assert = require('node:assert');\n"
                            "const { statusRoute } = require('../src/routes/status');\n\n"
                            "test('status route responds', () => {\n"
                            "  assert.ok(typeof statusRoute === 'function');\n"
                            "});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=RecordingProcessRunner(),
    )
    request = StartFeatureRequest.model_validate(_feature_payload())
    feature = _initial_feature_state("feature-live", request)
    technical_prd = _technical_prd()
    contract = _contract()
    feature.artifacts.extend([technical_prd, contract])
    repository = next(item for item in feature.repository_specs if item.repository_id == "backend")
    child = ChildWorkflowReference(
        child_workflow_id="feature-live:backend",
        repository_id="backend",
        workstream_id="backend",
        branch_name=harness["branch"],
        workspace_path=str(harness["workspace"]),
        status=ChildWorkflowStatus.RUNNING,
        retry_count=1,
    )
    arguments: dict[str, Any] = {
        "feature": feature,
        "repository": repository,
        "workstream": _workstream(),
        "child": child,
        "technical_prd": technical_prd,
        "contract": contract,
        "feedback": [],
        "credentials": RequestScopedCredentials(openai_api_key=None, github_token=None),
    }
    try:
        first = await executor.run(**arguments)
        first_operations = await harness["journal"].list_operations_for_workflow("feature-live")
        assert first.result.status == "approved"
        assert any(
            item.operation_type is ExternalOperationType.CREATE_COMMIT
            and bool(item.safe_metadata.get("publication_receipt"))
            for item in first_operations
        )
        # The first return value is intentionally discarded from durable feature state.
        replay = await executor.run(**arguments)
        recorded = await harness["journal"].list_operations_for_workflow("feature-live")
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert first.result.status == replay.result.status == "approved"
    assert replay.code_completion is not None
    assert replay.code_completion.metadata["publication_recovered"] is True
    assert len(engineer.calls) == 1
    assert len(reviewer.calls) == 1
    assert [item.operation_type for item in recorded].count(
        ExternalOperationType.RUN_CODING_EXECUTOR
    ) == 1
    assert [item.operation_type for item in recorded].count(
        ExternalOperationType.CREATE_COMMIT
    ) == 1
    assert [item.operation_type for item in recorded].count(ExternalOperationType.PUSH_BRANCH) == 1


@pytest.mark.asyncio
async def test_a_published_repository_can_still_be_asked_to_change(tmp_path: Path) -> None:
    """AB-Feature-111: approved and published, then unable to act on an integration fix.

    A publication receipt exists so a worker can die between commit and checkpoint without
    losing the commit. That window closes the moment durable child state records the pull
    request the publication produced -- after which the receipt is settled history.

    Treating it as recoverable regardless made every later attempt on that repository a
    replay of the old publication instead of a new attempt. -111's console was published at
    attempt 7, integration review asked it for one precise fix, and each granted attempt
    died in under a second against evidence from the publication that had already succeeded.
    A repository could be published, or corrected, but never both.
    """
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status route.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": (
                            "function statusRoute(req, res) {\n"
                            "  res.json({ status: 'ok' });\n"
                            "}\n\nmodule.exports = { statusRoute };\n"
                        ),
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\n"
                            "const assert = require('node:assert');\n"
                            "const { statusRoute } = require('../src/routes/status');\n\n"
                            "test('status route responds', () => {\n"
                            "  assert.ok(typeof statusRoute === 'function');\n"
                            "});\n"
                        ),
                    },
                ],
            },
            {
                "summary": "Correct the route after integration review.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": (
                            "function statusRoute(req, res) {\n"
                            "  res.json({ status: 'corrected' });\n"
                            "}\n\nmodule.exports = { statusRoute };\n"
                        ),
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\n"
                            "const assert = require('node:assert');\n"
                            "const { statusRoute } = require('../src/routes/status');\n\n"
                            "test('status route responds', () => {\n"
                            "  assert.ok(typeof statusRoute === 'function');\n"
                            "});\n"
                        ),
                    },
                ],
            },
        ]
    )
    reviewer = ScriptedLLMClient(
        [_review_payload(verdict="approved"), _review_payload(verdict="approved")]
    )
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=RecordingProcessRunner(),
    )
    request = StartFeatureRequest.model_validate(_feature_payload())
    feature = _initial_feature_state("feature-published-fix", request)
    technical_prd = _technical_prd()
    contract = _contract()
    feature.artifacts.extend([technical_prd, contract])
    repository = next(item for item in feature.repository_specs if item.repository_id == "backend")
    child = ChildWorkflowReference(
        child_workflow_id="feature-published-fix:backend",
        repository_id="backend",
        workstream_id="backend",
        branch_name=harness["branch"],
        workspace_path=str(harness["workspace"]),
        status=ChildWorkflowStatus.RUNNING,
        retry_count=1,
    )
    arguments: dict[str, Any] = {
        "feature": feature,
        "repository": repository,
        "workstream": _workstream(),
        "child": child,
        "technical_prd": technical_prd,
        "contract": contract,
        "feedback": ["Integration review: submit the persisted identifier."],
        "credentials": RequestScopedCredentials(openai_api_key=None, github_token=None),
    }
    try:
        published = await executor.run(**arguments)
        assert published.result.status == "approved"

        # What the parent records once publication settles: the pull request it produced.
        arguments["child"] = child.model_copy(
            update={"pull_request_artifact_id": "008_pull_request.backend.json"}
        )
        corrected = await executor.run(**arguments)
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    # A real second attempt, not a replay of the settled one.
    assert len(engineer.calls) == 2
    assert corrected.code_completion is not None
    assert corrected.code_completion.metadata.get("publication_recovered") is not True


@pytest.mark.asyncio
async def test_live_child_rejects_a_tests_only_change_before_review(tmp_path: Path) -> None:
    """A workstream expecting production code cannot be satisfied by editing tests.

    The engineer never reaches the reviewer, so no provider call is wasted and the
    failure is classified as missing implementation rather than a review rejection.
    """
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add a test for the status route.",
                "files": [
                    {
                        "path": "test/status.test.js",
                        "content": "const { test } = require('node:test');\ntest('x', () => {});\n",
                    }
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    execution = await _run(harness, engineer=engineer, reviewer=reviewer)

    assert execution.result.status == "failed"
    assert execution.result.failure_classification == "implementation_missing"
    assert execution.result.production_files_changed == []
    assert execution.result.test_files_changed == ["test/status.test.js"]
    # The reviewer must not be consulted about an incomplete implementation.
    assert reviewer.calls == []


@pytest.mark.asyncio
async def test_an_ordinary_retry_keeps_the_previous_attempts_files_in_place(
    tmp_path: Path,
) -> None:
    """An ordinary retry edits the previous attempt where it stands; nothing is reset.

    Deliberately the inverse of the rule this suite used to assert. The old test encoded
    the destroy-and-regenerate design: every retry reset the workspace and was handed a
    24,000-character slice of its own previous diff, which is the mechanism behind most of
    AB-Feature-170's and -171's failures. What the reset protected -- a fix-only retry
    silently dropping the implementation from the commit -- is now protected by the
    recorded lineage: the completion's `file_changes` are the union across attempts.
    """
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    (workspace / "src" / "prior.js").write_text(
        "module.exports = { prior: true };\n", encoding="utf-8"
    )
    runner = RecordingProcessRunner()
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Fix only the reported lint finding.",
                "files": [{"path": "src/fix.js", "content": "module.exports = {};\n"}],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])
    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    feature.artifacts.append(_persisted_prior_completion(paths=["src/prior.js"], attempt=0))
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=runner,
    )
    try:
        execution = await executor.run(
            feature=feature,
            repository=repository,
            workstream=_workstream(),
            child=child,
            technical_prd=technical_prd,
            contract=contract,
            feedback=[],
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    # The workspace enters the attempt still holding the previous attempt's file, and no
    # destructive command was chosen. The capture is still taken, for the durable record.
    assert ("git", "reset", "--hard", "HEAD") not in runner.commands
    assert all(command[:2] != ("git", "clean") for command in runner.commands)
    assert ("git", "add", "-A") in runner.commands
    assert ("git", "diff", "--cached") in runner.commands
    assert (workspace / "src" / "prior.js").is_file()
    # The prompt states the files are present, named from the recorded lineage, and asks
    # for correction via edits rather than byte-identical regeneration.
    instructions = engineer.calls[0][0]
    assert "still in this workspace" in instructions
    assert "src/prior.js" in instructions
    assert "returned to its committed state" not in instructions
    # The union lineage: the retry returned only its fix, and the completion carries both.
    assert execution.code_completion is not None
    changed = {change.path for change in execution.code_completion.file_changes}
    assert {"src/prior.js", "src/fix.js"} <= changed
    # The durable record says what this attempt ran against (§2.4).
    assert execution.result.metadata["attempt_inputs"]["workspace"] == "preserved"


@pytest.mark.asyncio
async def test_a_targeted_fix_is_judged_against_the_branch_that_holds_the_tests(
    tmp_path: Path,
) -> None:
    """66- T1: AB-Feature-201's frontend, replayed with the tests where they actually were.

    201's first cycle completed properly -- eight files including two test files, review
    approved, published. Integration review then asked for a Content-Type boundary header,
    and six consecutive one-file remediation attempts were refused by the same deterministic
    sentence, "The workstream requires tests but no test file changed", while both test files
    sat committed on the branch the attempt was correcting.

    Two things had put them out of reach, and this fixture reproduces both: the attempt's own
    completion declares one production file, so `_buckets` reads its empty test list as the
    evidence; and the completion holding the tests carries `published_after_approval`, which
    every lineage reader on this platform deliberately excludes. Only the branch can answer
    it, which is why Git runs for real here.
    """
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    # The approved-and-published first cycle, committed on the branch exactly as publication
    # leaves it. A *new* test file: the fixture's own `test/status.test.js` is on the default
    # branch, which is the baseline side of the diff and satisfies nothing.
    (workspace / "test" / "status.boundary.test.js").write_text(
        "const { test } = require('node:test');\ntest('boundary', () => {});\n", encoding="utf-8"
    )
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Attempt 0: the approved implementation and its tests")
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Send the Content-Type boundary the integration review asked for.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": (
                            "function statusRoute(req, res) {\n"
                            "  res.set('Content-Type', 'application/json; charset=utf-8');\n"
                            "  res.json({ status: 'ok' });\n"
                            "}\n\nmodule.exports = { statusRoute };\n"
                        ),
                    }
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])
    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    published = _persisted_prior_completion(
        paths=["src/routes/status.js", "test/status.boundary.test.js"],
        attempt=0,
        published=True,
    )
    feature.artifacts.append(published)
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=RealGitProcessRunner(),
    )
    try:
        execution = await executor.run(
            feature=feature,
            repository=repository,
            workstream=_tests_required_workstream(),
            child=child.model_copy(update={"pull_request_artifact_id": "008_pull_request.json"}),
            technical_prd=technical_prd,
            contract=contract,
            feedback=["Integration review: send the multipart boundary."],
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    # 201's shape, asserted rather than assumed: an unpublished prior completion would prove
    # nothing, because the reader that skips this one is the reason the naive fix finds
    # nothing.
    assert published.metadata["published_after_approval"] is True
    # The attempt reached the reviewer and was approved, rather than being refused by the
    # deterministic gate for tests it did not owe.
    assert execution.result.status == "approved"
    assert reviewer.calls
    assert execution.result.failure_classification is None
    assert not [item for item in execution.result.blocking_issues if "requires tests" in item], (
        execution.result.blocking_issues
    )
    # Which evidence judged it, on the durable record of the attempt.
    assert execution.result.metadata["attempt_inputs"]["category_evidence_source"] == "branch_diff"
    # 66- T8: the attempt's own claim is unchanged. These fields are copied onto the
    # completion artifact and the child, and `meaningful_progress` reads them to decide
    # whether an attempt repeated itself -- a union here would disarm that guard.
    assert execution.result.production_files_changed == ["src/routes/status.js"]
    assert execution.result.test_files_changed == []
    assert execution.result.configuration_files_changed == []
    assert execution.code_completion is not None
    assert execution.code_completion.test_files_changed == []
    assert execution.code_completion.production_files_changed == ["src/routes/status.js"]


@pytest.mark.asyncio
async def test_a_doubled_git_falls_back_to_the_lineage_and_says_which_it_used(
    tmp_path: Path,
) -> None:
    """66- T9: no silent fall-through, in either direction.

    With the lineage base unresolvable -- Git answering nothing, which is a checkout with no
    shared ancestor as far as this gate is concerned -- evidence comes from the lineage's
    completions *including the published ones*, and the record says so. With no lineage
    either, the gate reads the attempt's own declared buckets exactly as it did before, and
    the record says that instead: a remediation attempt showing `attempt_declared_paths` has
    the old behaviour whatever its blocking-issue count happens to be.
    """
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Send the Content-Type boundary the integration review asked for.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": (
                            "function statusRoute(req, res) {\n"
                            "  res.json({ status: 'ok', boundary: true });\n"
                            "}\n\nmodule.exports = { statusRoute };\n"
                        ),
                    }
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])
    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    feature.artifacts.append(
        _persisted_prior_completion(
            paths=["src/routes/status.js", "test/status.boundary.test.js"],
            attempt=0,
            published=True,
        )
    )
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        # Git is doubled here, so `merge-base` answers nothing and the branch read is refused
        # rather than measured against `HEAD`.
        process_runner=RecordingProcessRunner(),
    )
    try:
        recovered = await executor.run(
            feature=feature,
            repository=repository,
            workstream=_tests_required_workstream(),
            child=child.model_copy(update={"pull_request_artifact_id": "008_pull_request.json"}),
            technical_prd=technical_prd,
            contract=contract,
            feedback=["Integration review: send the multipart boundary."],
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert (
        recovered.result.metadata["attempt_inputs"]["category_evidence_source"]
        == "lineage_completions"
    )
    assert recovered.result.status == "approved"
    assert recovered.result.test_files_changed == []

    # The same attempt with nothing to read at all: the pre-66 behaviour, recorded as such.
    blind_harness = await _harness(tmp_path / "blind")
    blind_engineer = ScriptedLLMClient(
        [
            {
                "summary": "Send the Content-Type boundary the integration review asked for.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": (
                            "function statusRoute(req, res) {\n"
                            "  res.json({ status: 'ok', boundary: true });\n"
                            "}\n\nmodule.exports = { statusRoute };\n"
                        ),
                    }
                ],
            }
        ]
    )
    # `_run` disposes this harness's database itself.
    blind = await _run(
        blind_harness,
        engineer=blind_engineer,
        reviewer=ScriptedLLMClient([_review_payload(verdict="approved")]),
        workstream=_tests_required_workstream(),
    )

    assert (
        blind.result.metadata["attempt_inputs"]["category_evidence_source"]
        == "attempt_declared_paths"
    )
    assert blind.result.status == "failed"
    assert any("requires tests" in item for item in blind.result.blocking_issues)
    assert blind.result.metadata["category_evidence_source"] == "attempt_declared_paths"


def _tests_required_workstream() -> RepositoryWorkstreamPlan:
    """201's shape: the workstream owes tests, which its first cycle already wrote."""
    return _workstream().model_copy(
        update={
            "implementation_expectations": [
                expectation.model_copy(update={"tests_required": True})
                for expectation in _workstream().implementation_expectations
            ]
        }
    )


def _persisted_prior_completion(
    *,
    paths: list[str],
    attempt: int,
    workflow_id: str = "feature-live:backend",
    published: bool = False,
) -> CodeCompletionArtifact:
    """A prior attempt's completion, shaped the way parent persistence records one.

    ``published`` is what an approved-and-committed attempt carries: its files are already in
    `HEAD`, so the lineage reader that feeds the next attempt's prompt skips it -- and the
    evidence reader added for 66- does not.
    """
    metadata: dict[str, Any] = {"child_attempt": attempt}
    if published:
        metadata["published_after_approval"] = True
    return create_artifact(
        CodeCompletionArtifact,
        workflow_id=workflow_id,
        artifact_id=f"006_code_completion.backend.attempt-{attempt}.json",
        producer="engineer",
        payload={
            "completion_status": "completed" if published else "failed",
            "summary": (
                "The approved attempt that built this branch."
                if published
                else "The gate rejected this attempt."
            ),
            "file_changes": [
                {
                    "path": path,
                    "change_type": "added",
                    "description": "Updated by the configured coding executor.",
                }
                for path in paths
            ],
            "validation_results": [],
            "test_coverage_percent": None,
            "remaining_work": [],
            "commit_sha": None,
        },
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_a_wholesale_rewrite_rejection_still_resets_the_workspace(tmp_path: Path) -> None:
    """The one policy trigger that still discards a workspace, recorded with its cause.

    Drift is exactly what must not be edited into shape, so a retry after the rewrite gate
    fired gets the reset path -- and its record says so, with the trigger, rather than
    leaving the reflog as the only witness.
    """
    harness = await _harness(tmp_path)
    runner = RecordingProcessRunner()
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Re-implement from the capture.",
                "files": [{"path": "src/status.js", "content": "module.exports = {};\n"}],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])
    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    child = child.model_copy(
        update={"retry_strategy": {"workspace_reset_required": "wholesale_rewrite_rejected"}}
    )
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=runner,
    )
    try:
        execution = await executor.run(
            feature=feature,
            repository=repository,
            workstream=_workstream(),
            child=child,
            technical_prd=technical_prd,
            contract=contract,
            feedback=[],
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert ("git", "reset", "--hard", "HEAD") in runner.commands
    # No -x exclusions by pattern: cleanable paths are named, and the dependency tree and
    # hook bootstrap are preserved by never being named.
    clean = next(command for command in runner.commands if command[:2] == ("git", "clean"))
    assert clean[:6] == ("git", "clean", "--force", "-d", "-x", "--")
    assert "node_modules" not in clean[6:]
    staged = runner.commands.index(("git", "add", "-A"))
    captured = runner.commands.index(("git", "diff", "--cached"))
    reset = runner.commands.index(("git", "reset", "--hard", "HEAD"))
    assert staged < captured < reset
    inputs = execution.result.metadata["attempt_inputs"]
    assert inputs["workspace"] == "reset"
    assert inputs["reset_trigger"] == "wholesale_rewrite_rejected"


class LargeCaptureProcessRunner(RecordingProcessRunner):
    """Return a multi-file capture larger than the previous-attempt prompt budget."""

    def __init__(self, diff: str) -> None:
        super().__init__()
        self._diff = diff

    async def run(self, command: Any, cwd: Path, *_args: Any, **_kwargs: Any) -> ProcessResult:
        result = await super().run(command, cwd)
        if tuple(command) == ("git", "diff", "--cached"):
            return ProcessResult(
                command=tuple(command),
                return_code=0,
                stdout=self._diff,
                stderr="",
                duration_seconds=0.01,
            )
        return result


def _fabricated_capture(files: dict[str, int]) -> str:
    """A unified diff with one new file per entry, each of the requested content size."""
    segments = []
    for path, size in files.items():
        body = "+".join(f"line {index}\n" for index in range(size // 8))
        segments.append(
            f"diff --git a/{path} b/{path}\n"
            f"new file mode 100644\n"
            f"--- /dev/null\n"
            f"+++ b/{path}\n"
            f"@@ -0,0 +1 @@\n+{body}\n"
        )
    return "".join(segments)


@pytest.mark.asyncio
async def test_a_reset_path_capture_is_bounded_at_whole_file_boundaries(tmp_path: Path) -> None:
    """The capture shown to the model never cuts a file mid-hunk, and names what it withheld.

    Asserted on the delivered prompt, per the spec's own rule: AB-Feature-170's capture was
    50,477 characters, the blind 24,000-character slice landed inside its fourth file, and
    the prompt then demanded byte-identical reproduction of five files it never showed.
    """
    harness = await _harness(tmp_path)
    diff = _fabricated_capture(
        {
            "src/first.js": 9_000,
            "src/second.js": 9_000,
            "src/third.js": 9_000,
            "src/fourth.js": 9_000,
            "src/fifth.js": 9_000,
        }
    )
    runner = LargeCaptureProcessRunner(diff)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Re-create the change.",
                "files": [{"path": "src/first.js", "content": "module.exports = {};\n"}],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])
    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    child = child.model_copy(
        update={"retry_strategy": {"workspace_reset_required": "wholesale_rewrite_rejected"}}
    )
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=runner,
    )
    try:
        execution = await executor.run(
            feature=feature,
            repository=repository,
            workstream=_workstream(),
            child=child,
            technical_prd=technical_prd,
            contract=contract,
            feedback=[],
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    instructions = engineer.calls[0][0]
    # Every included file arrives intact end-to-end: its header and its final line.
    for included in ("src/first.js", "src/second.js"):
        assert f"diff --git a/{included} b/{included}" in instructions
    shown = instructions[instructions.index("diff --git a/src/first.js") :]
    included_segment = shown[: shown.index("That capture is incomplete")]
    assert included_segment.rstrip().endswith("line 1124"), (
        "the last included file must end exactly where the capture ends it"
    )
    # Every excluded file is named, and the mid-file cut never happens.
    assert "src/fifth.js" in instructions
    assert "NOT shown" in instructions
    withheld = execution.result.metadata["attempt_inputs"]["capture_withheld_paths"]
    assert withheld and withheld[-1] == "src/fifth.js"
    for path in withheld:
        assert f"diff --git a/{path}" not in instructions, "a withheld file must not be shown"


@pytest.mark.asyncio
async def test_live_child_reviewer_receives_only_repository_scoped_requirements(
    tmp_path: Path,
) -> None:
    """Scenario C: a backend must never be judged against a frontend-owned requirement."""
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status route.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": "module.exports = { statusRoute: () => ({ status: 'ok' }) };\n",
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('ok', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    await _run(harness, engineer=engineer, reviewer=reviewer)

    assert reviewer.calls, "the reviewer should have been consulted"
    review_prompt = "\n".join(reviewer.calls[0])
    assert '"requirement_id": "backend-status-api"' in review_prompt
    # The frontend requirement may appear as non-actionable background, but never as a
    # scoped reference the backend is obliged to satisfy.
    assert '"requirement_id": "frontend-status-tile"' not in review_prompt
    assert "frontend-status-tile" in review_prompt


@pytest.mark.asyncio
async def test_the_reset_removes_generated_output_but_keeps_what_is_expensive(
    tmp_path: Path,
) -> None:
    """Run the reset for real, because every recent production bug was an effect bug.

    The other tests here double the process runner, so they prove which commands were
    chosen and nothing about what those commands did. Feature -027 timed out because a
    gitignored ``build/`` survived a reset that recorded exactly the right arguments.
    """
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    (workspace / ".gitignore").write_text("node_modules\n/build\n", encoding="utf-8")
    _git(workspace, "add", ".gitignore")
    _git(workspace, "commit", "-m", "ignore generated output")

    # What a rejected attempt leaves behind, plus what must not be paid for twice.
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "src" / "Tile.js").write_text(
        "export const Tile = () => null;\n", encoding="utf-8"
    )
    (workspace / "build").mkdir(parents=True, exist_ok=True)
    (workspace / "build" / "bundle.js").write_text("//" + "x" * 2048 + "\n", encoding="utf-8")
    (workspace / "node_modules" / ".bin").mkdir(parents=True, exist_ok=True)
    (workspace / "node_modules" / "installed.txt").write_text("kept\n", encoding="utf-8")
    (workspace / "openapi.yaml").write_text("openapi: 3.0.0\n", encoding="utf-8")
    committed = (workspace / "package.json").read_text(encoding="utf-8")

    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=ScriptedLLMClient([]),
        reviewer_client=ScriptedLLMClient([]),
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        # No double: this test exists to observe what the commands actually do.
    )
    try:
        captured = await executor._reset_workspace(workspace)
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert not (workspace / "src" / "Tile.js").exists(), "the rejected attempt must be discarded"
    assert not (workspace / "build").exists(), "gitignored build output must not survive"
    assert (workspace / "node_modules" / "installed.txt").exists(), "installs must be reused"
    # Not preserved: it is rewritten from the approved contract on every attempt, so a copy
    # the previous attempt edited must not be carried forward and blamed on the next one.
    assert not (workspace / "openapi.yaml").exists()
    assert (workspace / "package.json").read_text(encoding="utf-8") == committed
    # The diff is the only record of the rejected attempt once the files are gone.
    assert captured is not None
    assert "Tile.js" in captured
    assert "export const Tile" in captured
    assert "openapi" not in captured, "platform scaffolding is not the model's work"


@pytest.mark.asyncio
async def test_failed_workspace_reset_stops_before_another_coding_attempt(tmp_path: Path) -> None:
    """Continuing after `git reset --hard` fails would mix rejected attempts together."""
    harness = await _harness(tmp_path)
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=ScriptedLLMClient([]),
        reviewer_client=ScriptedLLMClient([]),
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=FailedResetProcessRunner(),
    )
    try:
        with pytest.raises(RepositoryWorkspaceError) as caught:
            await executor._reset_workspace(harness["workspace"])
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert caught.value.failure_classification == "validation_configuration_failure"
    assert "git reset --hard HEAD" in caught.value.diagnostics[0]
    assert "WORKSPACE_COMMAND_FAILED_EXIT_128" in caught.value.diagnostics[0]


@pytest.mark.asyncio
async def test_retry_diff_is_redacted_before_entering_next_engineer_prompt(tmp_path: Path) -> None:
    """Injected process runners cannot bypass credential filtering at the prompt handoff."""
    harness = await _harness(tmp_path)
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=ScriptedLLMClient([]),
        reviewer_client=ScriptedLLMClient([]),
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=SecretDiffProcessRunner(),
    )
    try:
        previous_attempt = await executor._reset_workspace(harness["workspace"])
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert previous_attempt is not None
    assert "sk-proj-RESETSECRET123456789" not in previous_attempt
    assert "gho_RESETSECRET123456789" not in previous_attempt
    assert "V7mQ2xL9pR4sT8wY3kN6dF1z" not in previous_attempt
    assert previous_attempt.count("[REDACTED]") == 3


@pytest.mark.asyncio
async def test_publishing_a_change_that_matches_the_committed_revision_is_refusable(
    tmp_path: Path,
) -> None:
    """Nothing is published until review approves, so the no-op is detected at that point.

    The repair instruction asks for a minimal change, which makes returning content identical
    to the committed revision more likely. Git exits non-zero for that, and reading it as a
    hard adapter fault abandoned the whole workstream over an ordinary outcome.
    """
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=ScriptedLLMClient([]),
        reviewer_client=ScriptedLLMClient([]),
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
    )
    git_service = InterruptibleGitService(
        default_branch="main",
        environment={},
        cancellation_token=MockCancellationToken(),
        workspace_root=harness["settings"].workspace_root,
    )
    reviewed_paths = ["package.json"]
    completion = _code_completion(reviewed_paths)
    review = _approved_review(
        reviewed_paths,
        reviewed_content_fingerprint(workspace, reviewed_paths),
    )

    try:
        with pytest.raises(EmptyCommitError):
            await executor._publish_approved_change(
                git_service,
                workspace=workspace,
                branch_name=harness["branch"],
                summary="no net change",
                code_completion=completion,
                review=review,
                workflow_id="feature-live:backend",
            )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()


@pytest.mark.asyncio
async def test_publication_reuses_commit_and_push_after_a_lost_feature_checkpoint(
    tmp_path: Path,
) -> None:
    """A crash after push can replay publication without a second commit or remote write."""
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    changed_path = workspace / "src" / "routes" / "status.js"
    changed_path.write_text(
        "function statusRoute(req, res) {\n"
        "  res.json({ status: 'ok', recovered: true });\n"
        "}\n\nmodule.exports = { statusRoute };\n",
        encoding="utf-8",
    )
    reviewed_paths = ["src/routes/status.js"]
    completion = _code_completion(reviewed_paths)
    review = _approved_review(
        reviewed_paths,
        reviewed_content_fingerprint(workspace, reviewed_paths),
    )
    operations = ExternalOperationExecutor(
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(
            workflow_id="feature-live",
            feature_id="feature-live",
            child_workflow_id="feature-live:backend",
            repository_id="backend",
        ),
    )
    git_service = InterruptibleGitService(
        default_branch="main",
        cancellation_token=MockCancellationToken(),
        operation_executor=operations,
        workspace_root=harness["settings"].workspace_root,
    )
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=ScriptedLLMClient([]),
        reviewer_client=ScriptedLLMClient([]),
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
    )
    try:
        first = await executor._publish_approved_change(
            git_service,
            workspace=workspace,
            branch_name=harness["branch"],
            summary="durable publication",
            code_completion=completion,
            review=review,
            workflow_id="feature-live:backend",
        )
        # Discard `first`, exactly as a process crash would before the child checkpoint.
        replay = await executor._publish_approved_change(
            git_service,
            workspace=workspace,
            branch_name=harness["branch"],
            summary="durable publication",
            code_completion=completion,
            review=review,
            workflow_id="feature-live:backend",
        )
        recorded = await harness["journal"].list_operations_for_workflow("feature-live")
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert first.commit_sha == replay.commit_sha
    assert _git_output(workspace, "rev-list", "--count", "main..HEAD") == "1"
    assert (
        _git_output(
            workspace,
            "ls-remote",
            "origin",
            f"refs/heads/{harness['branch']}",
        ).split()[0]
        == first.commit_sha
    )
    assert [item.operation_type for item in recorded].count(
        ExternalOperationType.CREATE_COMMIT
    ) == 1
    assert [item.operation_type for item in recorded].count(ExternalOperationType.PUSH_BRANCH) == 1
    assert all(item.status is ExternalOperationStatus.SUCCEEDED for item in recorded)


@pytest.mark.asyncio
async def test_crash_after_engineer_reconciles_coding_before_review(tmp_path: Path) -> None:
    """A lost review boundary must not invoke Engineer twice for one durable attempt."""
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient([_changed_status_payload()])
    reviewer = CrashOnceLLMClient([_review_payload(verdict="approved")])
    feature, repository, child, technical_prd, contract = _live_run_context(harness)

    def executor() -> LiveChildWorkstreamExecutor:
        return LiveChildWorkstreamExecutor(
            settings=harness["settings"],
            git_environment={},
            engineer_client=engineer,
            reviewer_client=reviewer,
            journal=harness["journal"],
            cancellation_token=MockCancellationToken(),
            process_runner=RecordingProcessRunner(),
        )

    arguments = {
        "feature": feature,
        "repository": repository,
        "workstream": _workstream(),
        "child": child,
        "technical_prd": technical_prd,
        "contract": contract,
        "feedback": [],
        "credentials": RequestScopedCredentials(openai_api_key=None, github_token=None),
    }
    try:
        with pytest.raises(RuntimeError, match="simulated worker crash"):
            await executor().run(**arguments)

        execution = await executor().run(**arguments)
        operations = await harness["journal"].list_operations_for_workflow("feature-live")
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert execution.result.status == "approved"
    assert len(engineer.calls) == 1
    assert len(reviewer.calls) == 2
    coding_operations = [
        item
        for item in operations
        if item.operation_type is ExternalOperationType.RUN_CODING_EXECUTOR
    ]
    assert len(coding_operations) == 1
    assert coding_operations[0].status is ExternalOperationStatus.SUCCEEDED
    assert coding_operations[0].safe_metadata["child_attempt"] == child.retry_count


@pytest.mark.asyncio
async def test_crash_after_repository_review_reuses_the_durable_verdict(tmp_path: Path) -> None:
    """A validated review receipt routes after restart without a second reviewer call."""

    class CrashBeforePublication(LiveChildWorkstreamExecutor):
        async def _publish_approved_change(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("simulated crash after review persistence")

    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient([_changed_status_payload()])
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])
    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    common = {
        "settings": harness["settings"],
        "git_environment": {},
        "engineer_client": engineer,
        "reviewer_client": reviewer,
        "journal": harness["journal"],
        "cancellation_token": MockCancellationToken(),
        "process_runner": RecordingProcessRunner(),
    }
    arguments = {
        "feature": feature,
        "repository": repository,
        "workstream": _workstream(),
        "child": child,
        "technical_prd": technical_prd,
        "contract": contract,
        "feedback": [],
        "credentials": RequestScopedCredentials(openai_api_key=None, github_token=None),
    }
    try:
        with pytest.raises(RuntimeError, match="after review persistence"):
            await CrashBeforePublication(**common).run(**arguments)

        execution = await LiveChildWorkstreamExecutor(**common).run(**arguments)
        operations = await harness["journal"].list_operations_for_workflow("feature-live")
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert execution.result.status == "approved"
    assert len(engineer.calls) == 1
    assert len(reviewer.calls) == 1
    reviewer_operations = [
        item for item in operations if item.operation_type is ExternalOperationType.RUN_REVIEWER
    ]
    assert len(reviewer_operations) == 1
    assert reviewer_operations[0].status is ExternalOperationStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_no_change_result_preserves_the_exact_preflight_revision(tmp_path: Path) -> None:
    """A no-op remediation stops on R1 without erasing R1 from durable child state."""
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient([_unchanged_status_payload()])
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])
    feature, repository, child, technical_prd, contract = _live_run_context(
        harness, current_revision="persisted-R1"
    )
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=RecordingProcessRunner(),
    )
    try:
        execution = await executor.run(
            feature=feature,
            repository=repository,
            workstream=_workstream(),
            child=child,
            technical_prd=technical_prd,
            contract=contract,
            feedback=[],
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert execution.result.status == "failed"
    assert execution.result.current_revision
    assert execution.result.preflight_result is not None
    assert execution.result.current_revision == execution.result.preflight_result["revision"]
    assert "no change" in " ".join(execution.result.blocking_issues).lower()
    assert len(engineer.calls) == 1
    assert len(reviewer.calls) == 1


@pytest.mark.asyncio
async def test_post_review_cleanup_mutation_is_rejected_before_commit(tmp_path: Path) -> None:
    """A prepare/cleanup mutation cannot silently replace bytes the reviewer approved."""

    class MutatingCleanupExecutor(LiveChildWorkstreamExecutor):
        async def _remove_generated_output(self, workspace: Path) -> None:
            (workspace / "src" / "routes" / "status.js").write_text(
                "module.exports = { statusRoute: () => 'mutated after review' };\n",
                encoding="utf-8",
            )

    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    baseline = _git_output(workspace, "rev-parse", "HEAD")
    (workspace / "src" / "routes" / "status.js").write_text(
        "module.exports = { statusRoute: () => 'reviewed' };\n", encoding="utf-8"
    )
    executor = MutatingCleanupExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=ScriptedLLMClient([]),
        reviewer_client=ScriptedLLMClient([]),
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
    )
    git_service = InterruptibleGitService(
        default_branch="main",
        cancellation_token=MockCancellationToken(),
        workspace_root=harness["settings"].workspace_root,
    )
    reviewed_paths = ["src/routes/status.js"]
    review = _approved_review(
        reviewed_paths,
        reviewed_content_fingerprint(workspace, reviewed_paths),
    )
    try:
        with pytest.raises(AgentArtifactError, match="no longer matches"):
            await executor._publish_approved_change(
                git_service,
                workspace=workspace,
                branch_name=harness["branch"],
                summary="reviewed publication",
                code_completion=_code_completion(reviewed_paths),
                review=review,
                workflow_id="feature-live:backend",
            )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert _git_output(workspace, "rev-parse", "HEAD") == baseline


@pytest.mark.asyncio
async def test_git_hook_mutation_is_detected_and_never_pushed(tmp_path: Path) -> None:
    """A hook that stages different bytes leaves an unknown local commit, not a remote one."""
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    changed_path = workspace / "src" / "routes" / "status.js"
    changed_path.write_text(
        "module.exports = { statusRoute: () => 'reviewed' };\n", encoding="utf-8"
    )
    hook = workspace / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"module.exports = { statusRoute: () => 'hook-mutated' };\" "
        "> src/routes/status.js\n"
        "git add -- src/routes/status.js\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)
    operations = ExternalOperationExecutor(
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(
            workflow_id="feature-live",
            feature_id="feature-live",
            child_workflow_id="feature-live:backend",
            repository_id="backend",
        ),
    )
    git_service = InterruptibleGitService(
        default_branch="main",
        cancellation_token=MockCancellationToken(),
        operation_executor=operations,
        workspace_root=harness["settings"].workspace_root,
    )
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=ScriptedLLMClient([]),
        reviewer_client=ScriptedLLMClient([]),
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
    )
    reviewed_paths = ["src/routes/status.js"]
    review = _approved_review(
        reviewed_paths,
        reviewed_content_fingerprint(workspace, reviewed_paths),
    )
    try:
        with pytest.raises(UnknownExternalOperation, match="changed the approved commit"):
            await executor._publish_approved_change(
                git_service,
                workspace=workspace,
                branch_name=harness["branch"],
                summary="reviewed publication",
                code_completion=_code_completion(reviewed_paths),
                review=review,
                workflow_id="feature-live:backend",
            )
        recorded = await harness["journal"].list_operations_for_workflow("feature-live")
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    commit_operations = [
        item for item in recorded if item.operation_type is ExternalOperationType.CREATE_COMMIT
    ]
    assert len(commit_operations) == 1
    assert commit_operations[0].status is ExternalOperationStatus.UNKNOWN_EXTERNAL_STATE
    assert not any(item.operation_type is ExternalOperationType.PUSH_BRANCH for item in recorded)
    assert _git_output(workspace, "ls-remote", "origin", f"refs/heads/{harness['branch']}") == ""


def _code_completion(paths: list[str]) -> Any:
    """Build the minimal completion artifact the publish step consumes."""
    return create_artifact(
        CodeCompletionArtifact,
        workflow_id="feature-live:backend",
        artifact_id="006_code_completion.json",
        producer="engineer",
        payload={
            "completion_status": "completed",
            "summary": "unchanged",
            "file_changes": [
                {"path": path, "change_type": "modified", "description": "unchanged"}
                for path in paths
            ],
            "validation_results": [],
            "test_coverage_percent": None,
            "remaining_work": [],
            "commit_sha": None,
        },
        metadata={},
    )


def _approved_review(paths: list[str], content_fingerprint: str) -> ReviewArtifact:
    """Build an approval bound to the exact workspace evidence ReviewerAgent inspected."""
    return create_artifact(
        ReviewArtifact,
        workflow_id="feature-live:backend",
        artifact_id="007_review.json",
        producer="reviewer",
        payload={
            "verdict": "approved",
            "summary": "The exact changed source is approved.",
            "requirement_checks": [],
            "findings": [],
            "architecture_assessment": "The change preserves repository structure.",
            "security_assessment": "The changed source contains no blocking issue.",
            "test_coverage_assessment": "Repository validation passed.",
        },
        metadata={
            "review_evidence_version": "workspace-source-v1",
            "reviewed_file_paths": paths,
            "reviewed_content_fingerprint": content_fingerprint,
            "manual_review_required": False,
            "retryable": True,
        },
    )


@pytest.mark.asyncio
async def test_the_pre_commit_cleanup_keeps_the_toolchain_it_needs(tmp_path: Path) -> None:
    """Run the publish-time cleanup for real against a checkout that has a toolchain.

    This is the test that was missing. The reset was covered by a real-effect test, but the
    cleanup that runs just before a commit was covered only by asserting which flags it
    chose -- and those flags were wrong: `git clean --exclude` adds a pattern to the ignore
    rules, so under -X it deleted the dependency tree and the hook bootstrap it was believed
    to protect. The commit then failed because its pre-commit hook could not load.
    """
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    (workspace / ".gitignore").write_text("node_modules\n/build\n/coverage\n", encoding="utf-8")
    _git(workspace, "add", ".gitignore")
    _git(workspace, "commit", "-m", "ignore generated output")

    (workspace / "node_modules" / ".bin").mkdir(parents=True, exist_ok=True)
    (workspace / "node_modules" / ".bin" / "husky").write_text("#!/bin/sh\n", encoding="utf-8")
    (workspace / ".husky" / "_").mkdir(parents=True, exist_ok=True)
    (workspace / ".husky" / "_" / "husky.sh").write_text("# bootstrap\n", encoding="utf-8")
    (workspace / ".husky" / "pre-commit").write_text("npm run lint\n", encoding="utf-8")
    (workspace / "build").mkdir(parents=True, exist_ok=True)
    (workspace / "build" / "bundle.js").write_text("//generated\n", encoding="utf-8")
    (workspace / "coverage").mkdir(parents=True, exist_ok=True)
    (workspace / "coverage" / "lcov.info").write_text("TN:\n", encoding="utf-8")

    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=ScriptedLLMClient([]),
        reviewer_client=ScriptedLLMClient([]),
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
    )
    try:
        await executor._remove_generated_output(workspace)
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    # The commit runs the repository's own hook, which needs both of these to exist.
    assert (workspace / "node_modules" / ".bin" / "husky").is_file()
    assert (workspace / ".husky" / "_" / "husky.sh").is_file()
    assert (workspace / ".husky" / "pre-commit").is_file()
    # Generated output is what this step exists to remove.
    assert not (workspace / "build").exists()
    assert not (workspace / "coverage").exists()


async def _harness(tmp_path: Path) -> dict[str, Any]:
    """Build a real checkout, a local remote, and the durable journal the executor needs."""
    workspace_root = tmp_path / "workspaces"
    workspace = workspace_root / "feature-live" / "backend"
    node_javascript_repository(workspace)
    remote = tmp_path / "remote.git"
    subprocess.run(("git", "init", "--bare", str(remote)), check=True, capture_output=True)
    _git(workspace, "remote", "add", "origin", str(remote))
    _git(workspace, "push", "origin", "main")
    branch = "ai/feature-live/backend/server-status"
    _git(workspace, "checkout", "-b", branch)

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'journal.db'}")
    await database.create_schema()
    return {
        "database": database,
        "journal": ExternalOperationJournal(database),
        "settings": load_settings(workspace_root=workspace_root),
        "workspace": workspace,
        "branch": branch,
        "remote": remote,
    }


async def _run(
    harness: dict[str, Any],
    *,
    engineer: ScriptedLLMClient,
    reviewer: ScriptedLLMClient,
    process_runner: RecordingProcessRunner | None = None,
    workstream: RepositoryWorkstreamPlan | None = None,
) -> Any:
    """Execute one live child workstream and always dispose of its database."""
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=process_runner or RecordingProcessRunner(),
    )
    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    try:
        return await executor.run(
            feature=feature,
            repository=repository,
            workstream=workstream or _workstream(),
            child=child,
            technical_prd=technical_prd,
            contract=contract,
            feedback=[],
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()


def _live_run_context(
    harness: dict[str, Any], *, current_revision: str | None = None
) -> tuple[
    Any,
    Any,
    ChildWorkflowReference,
    TechnicalPRDArtifact,
    IntegrationContractArtifact,
]:
    """Build the stable live inputs reused when a test simulates process restart."""
    request = StartFeatureRequest.model_validate(_feature_payload())
    feature = _initial_feature_state("feature-live", request)
    technical_prd = _technical_prd()
    contract = _contract()
    # The child result records parent lineage, so the planning artifacts must already be
    # on the feature exactly as they are when the real orchestrator fans out.
    feature.artifacts.extend([technical_prd, contract])
    repository = next(item for item in feature.repository_specs if item.repository_id == "backend")
    child = ChildWorkflowReference(
        child_workflow_id="feature-live:backend",
        repository_id="backend",
        workstream_id="backend",
        branch_name=harness["branch"],
        workspace_path=str(harness["workspace"]),
        status=ChildWorkflowStatus.RUNNING,
        # A retry skips provisioning, so the executor uses the checkout prepared above
        # instead of cloning from GitHub during a unit test.
        retry_count=1,
        current_revision=current_revision,
    )
    return feature, repository, child, technical_prd, contract


@pytest.mark.asyncio
async def test_replacing_an_existing_file_instead_of_editing_it_is_rejected(
    tmp_path: Path,
) -> None:
    """Answering a small requirement by rewriting a file must not reach the reviewer.

    A frontend workstream once kept the tile it was asked to add, dropped the tiles already
    on the page, and left a comment saying the rest would normally be rendered. Every gate
    passed because each asked whether the new behavior existed, not whether the old survived.
    """
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    existing = workspace / "src" / "routes" / "tiles.js"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(
        "".join(f"// existing behaviour line {index}\n" for index in range(120)),
        encoding="utf-8",
    )
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Add the tiles already on the page")
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status tile.",
                "files": [
                    {
                        "path": "src/routes/tiles.js",
                        "content": "// Other tiles would normally be rendered here.\nstatus();\n",
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('status', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    execution = await _run(
        harness, engineer=engineer, reviewer=reviewer, process_runner=RealGitProcessRunner()
    )

    assert execution.result.status == "failed"
    assert any("rewritten rather than edited" in item for item in execution.result.blocking_issues)
    # The reviewer must never be asked to bless a change that deleted existing behavior.
    assert reviewer.calls == []


@pytest.mark.asyncio
async def test_a_new_module_nothing_refers_to_is_rejected_as_unfinished(tmp_path: Path) -> None:
    """A module added beside hand-wired neighbours, that nothing reaches, cannot ship."""
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    routes = workspace / "src" / "routes"
    routes.mkdir(parents=True, exist_ok=True)
    # An existing neighbour the application does reach, so this directory is wired by hand.
    (routes / "boardroute.js").write_text("module.exports = {};\n", encoding="utf-8")
    (workspace / "src" / "application.js").write_text(
        "const board = require('./routes/boardroute');\nmodule.exports = { board };\n",
        encoding="utf-8",
    )
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Add the route the application registers")
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status route.",
                "files": [
                    {
                        "path": "src/routes/statusroute.js",
                        "content": "module.exports = { handler: () => 'ok' };\n",
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('status', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    execution = await _run(
        harness, engineer=engineer, reviewer=reviewer, process_runner=RealGitProcessRunner()
    )

    assert execution.result.status == "failed"
    # The diagnostic names one file to edit, and it is the module registry rather than the
    # first file that happens to mention a neighbour. `src/application.js` also reaches this
    # directory, and naming a consumer as the way in is what sent -069's console to render a
    # page inside an unrelated one. Naming the sibling first is what made -070's console
    # render its page inside that sibling as well as registering it.
    issue = next(
        item
        for item in execution.result.blocking_issues
        if "no production file refers to it" in item
    )
    assert "Edit src/index.js, which is where this repository registers modules" in issue
    assert "it already reaches the neighbouring src/routes/status.js" in issue
    assert "Do not add the reference to src/routes/status.js itself" in issue
    # The scoped repair handed to the next attempt agrees with the sentence: the registry is
    # the file to change, and the sibling is only there to be read.
    repair = execution.result.metadata.get("wiring_repair")
    assert repair is not None
    assert repair["target_path"] == "src/index.js"
    assert repair["example_path"] == "src/routes/status.js"
    assert repair["problem"] == "never_referenced"
    # The reviewer is never asked to bless code the application cannot reach.
    assert reviewer.calls == []


@pytest.mark.asyncio
async def test_the_wiring_gate_names_the_file_the_plan_assigned(tmp_path: Path) -> None:
    """The plan already answered where the change goes; the gate must not answer differently.

    Everything else in the gate reasons structurally -- directory adjacency, reference
    counting -- which finds what *could* host a module, not what *should*. AB-Feature-166's
    console was assigned `src/pages/Profile.js` and told three times to wire into
    `src/components/Header.js`, a file its plan never mentions, because the structural answer
    was the only one the gate had. Three attempts followed that pointer and the workstream
    ended on the identical-diagnostic rule.
    """
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    routes = workspace / "src" / "routes"
    routes.mkdir(parents=True, exist_ok=True)
    (routes / "boardroute.js").write_text("module.exports = {};\n", encoding="utf-8")
    (workspace / "src" / "application.js").write_text(
        "const board = require('./routes/boardroute');\nmodule.exports = { board };\n",
        encoding="utf-8",
    )
    # The file the plan assigns, which nothing in this directory references -- so the
    # structural search would never reach it.
    pages = workspace / "src" / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    (pages / "Profile.js").write_text("module.exports = { Profile: () => null };\n", "utf-8")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Add the assigned page and a wired route")

    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status route.",
                "files": [
                    {
                        "path": "src/routes/statusroute.js",
                        "content": "module.exports = { handler: () => 'ok' };\n",
                    }
                ],
            }
        ]
    )
    assigned = _workstream().model_copy(
        update={"expected_files_or_areas": ["src", "src/pages/Profile.js"]}
    )

    execution = await _run(
        harness,
        engineer=engineer,
        reviewer=ScriptedLLMClient([_review_payload(verdict="approved")]),
        process_runner=RealGitProcessRunner(),
        workstream=assigned,
    )

    assert execution.result.status == "failed"
    issue = next(
        item
        for item in execution.result.blocking_issues
        if "no production file refers to it" in item
    )
    assert "This workstream's plan assigns it src/pages/Profile.js" in issue
    # And the scoped repair the next attempt is given agrees with the sentence.
    repair = execution.result.metadata.get("wiring_repair")
    assert repair is not None
    assert repair["target_path"] == "src/pages/Profile.js"
    assert repair["target_source"] == "workstream_plan"


@pytest.mark.asyncio
async def test_a_plan_that_names_only_areas_keeps_the_structural_answer(tmp_path: Path) -> None:
    """A directory is not somewhere a reference can be added, so nothing changes for it."""
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    routes = workspace / "src" / "routes"
    routes.mkdir(parents=True, exist_ok=True)
    (routes / "boardroute.js").write_text("module.exports = {};\n", encoding="utf-8")
    (workspace / "src" / "index.js").write_text(
        "const board = require('./routes/boardroute');\nmodule.exports = { board };\n",
        encoding="utf-8",
    )
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Add a registry and a wired route")

    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status route.",
                "files": [
                    {
                        "path": "src/routes/statusroute.js",
                        "content": "module.exports = { handler: () => 'ok' };\n",
                    }
                ],
            }
        ]
    )

    execution = await _run(
        harness,
        engineer=engineer,
        reviewer=ScriptedLLMClient([_review_payload(verdict="approved")]),
        process_runner=RealGitProcessRunner(),
        # The harness plan names `src/routes`, a directory and not a file.
    )

    repair = execution.result.metadata.get("wiring_repair")
    assert repair is not None
    assert repair["target_path"] == "src/index.js"
    assert "target_source" not in repair


@pytest.mark.asyncio
async def test_every_unreachable_module_gets_its_own_scoped_repair(tmp_path: Path) -> None:
    """Two unreachable modules produce two repairs, not just one.

    The engineer force-includes a repair's `target_path` in its snapshot, and can only quote
    a file back exactly when it is there. Reporting one repair for two defects therefore cost
    an attempt per extra file: -106's backend wired the service whose registry it was shown
    and left the validator whose registry it never saw, and the identical diagnostic came
    back the next attempt.
    """
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    routes = workspace / "src" / "routes"
    helpers = workspace / "src" / "helpers"
    routes.mkdir(parents=True, exist_ok=True)
    helpers.mkdir(parents=True, exist_ok=True)
    # Each directory is wired by hand through a different registry, which is the shape that
    # makes one repair insufficient: the two targets are not the same file.
    (routes / "boardroute.js").write_text("module.exports = {};\n", encoding="utf-8")
    (helpers / "boardhelper.js").write_text("module.exports = {};\n", encoding="utf-8")
    (workspace / "src" / "application.js").write_text(
        "const board = require('./routes/boardroute');\nmodule.exports = { board };\n",
        encoding="utf-8",
    )
    (workspace / "src" / "support.js").write_text(
        "const helper = require('./helpers/boardhelper');\nmodule.exports = { helper };\n",
        encoding="utf-8",
    )
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Add two hand-wired directories")
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status route and its helper.",
                "files": [
                    {
                        "path": "src/routes/statusroute.js",
                        "content": "module.exports = { handler: () => 'ok' };\n",
                    },
                    {
                        "path": "src/helpers/statushelper.js",
                        "content": "module.exports = { format: () => 'ok' };\n",
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('status', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    execution = await _run(
        harness, engineer=engineer, reviewer=reviewer, process_runner=RealGitProcessRunner()
    )

    assert execution.result.status == "failed"
    repairs = execution.result.metadata.get("wiring_repairs")
    assert repairs is not None
    # Both unreachable modules are described, each against the file that has to reach it.
    assert {item["added_path"] for item in repairs} == {
        "src/routes/statusroute.js",
        "src/helpers/statushelper.js",
    }
    assert all(item["problem"] == "never_referenced" for item in repairs)
    # Two distinct registries, so an attempt shown only the first could never fix the second.
    assert len({item["target_path"] for item in repairs}) == 2
    # The single-repair key still names the first, for anything reading one scoped fix.
    assert execution.result.metadata["wiring_repair"] == repairs[0]
    assert reviewer.calls == []


@pytest.mark.asyncio
async def test_a_module_only_its_own_test_imports_is_still_unreachable(tmp_path: Path) -> None:
    """A module read only by its own spec and its documentation is still not mounted."""
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status route and its test.",
                "files": [
                    {
                        "path": "src/routes/statusroute.js",
                        "content": "module.exports = { handler: () => 'ok' };\n",
                    },
                    {
                        "path": "test/statusroute.test.js",
                        "content": (
                            "const { test } = require('node:test');\n"
                            "const mod = require('../src/routes/statusroute');\n"
                            "test('status', () => { mod.handler(); });\n"
                        ),
                    },
                    {
                        # Documentation describing a module does not make it reachable either.
                        "path": "docs/statusroute.md",
                        "content": "# statusroute\n\nDescribes the statusroute handler.\n",
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    execution = await _run(
        harness, engineer=engineer, reviewer=reviewer, process_runner=RealGitProcessRunner()
    )

    assert execution.result.status == "failed"
    assert any(
        "no production file refers to it" in item for item in execution.result.blocking_issues
    )


@pytest.mark.asyncio
async def test_a_file_without_an_extension_is_not_treated_as_a_wired_neighbour(
    tmp_path: Path,
) -> None:
    """A LICENSE beside a module is not evidence that the directory is wired by hand."""
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    handlers = workspace / "src" / "handlers"
    handlers.mkdir(parents=True, exist_ok=True)
    # Extensionless, and named in prose elsewhere, which once made it look referenced.
    (handlers / "LICENSE").write_text("All rights reserved.\n", encoding="utf-8")
    (workspace / "src" / "notice.js").write_text(
        "// See the LICENSE file for terms.\nmodule.exports = {};\n", encoding="utf-8"
    )
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Add the licence notice")
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status route handler.",
                "files": [
                    {
                        "path": "src/handlers/statusroute.js",
                        "content": "module.exports = { handler: () => 'ok' };\n",
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('status', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    execution = await _run(
        harness, engineer=engineer, reviewer=reviewer, process_runner=RealGitProcessRunner()
    )

    # The subject here is the reachability gate staying silent. The workstream itself is
    # now also held to the integration promise, which a convention-scanned addition cannot
    # meet; that trade is recorded in the schema, and this test does not assert approval.
    assert not any(
        "no production file refers to it" in item for item in execution.result.blocking_issues
    )


@pytest.mark.asyncio
async def test_a_module_that_is_imported_but_never_used_is_still_unreachable(
    tmp_path: Path,
) -> None:
    """An import is a declaration. chk-1 imported its tile and rendered nothing."""
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    routes = workspace / "src" / "routes"
    routes.mkdir(parents=True, exist_ok=True)
    (routes / "boardroute.js").write_text("module.exports = {};\n", encoding="utf-8")
    (workspace / "src" / "application.js").write_text(
        "const board = require('./routes/boardroute');\nmodule.exports = { board };\n",
        encoding="utf-8",
    )
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Add the route the application registers")
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status route.",
                "files": [
                    {
                        "path": "src/routes/statusroute.js",
                        "content": "module.exports = { handler: () => 'ok' };\n",
                    },
                    {
                        # Imported at the top and never referred to again.
                        "path": "src/application.js",
                        "content": (
                            "const board = require('./routes/boardroute');\n"
                            "const statusroute = require('./routes/statusroute');\n"
                            "module.exports = { board };\n"
                        ),
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('status', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    execution = await _run(
        harness, engineer=engineer, reviewer=reviewer, process_runner=RealGitProcessRunner()
    )

    assert execution.result.status == "failed"
    assert any("never uses it" in item for item in execution.result.blocking_issues)
    # The next attempt is handed the file and symbol, not the whole task again.
    repair = execution.result.metadata.get("wiring_repair")
    assert repair is not None
    assert repair["target_path"] == "src/application.js"
    assert repair["problem"] == "imported_but_unused"
    # The reviewer must not be asked to bless a component nothing renders.
    assert reviewer.calls == []


@pytest.mark.asyncio
async def test_a_default_import_used_under_a_local_name_is_not_called_dead(
    tmp_path: Path,
) -> None:
    """What a file calls a module is rarely its file name.

    fix-1's route imported `admin.serverStatus.controller` and used it as
    `serverStatusController`. Comparing the module's own stem against the body reported a
    correctly wired route as unreachable on six consecutive attempts.
    """
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    routes = workspace / "src" / "routes"
    routes.mkdir(parents=True, exist_ok=True)
    (routes / "boardroute.js").write_text("module.exports = {};\n", encoding="utf-8")
    (workspace / "src" / "application.js").write_text(
        "const board = require('./routes/boardroute');\nmodule.exports = { board };\n",
        encoding="utf-8",
    )
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Add the route the application registers")
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status controller and use it from the route.",
                "files": [
                    {
                        "path": "src/routes/status.controller.js",
                        "content": "module.exports = { getStatus: () => 'ok' };\n",
                    },
                    {
                        # Bound to a local name that shares nothing with the file stem.
                        "path": "src/routes/statusroute.js",
                        "content": (
                            "const statusController = require('./status.controller');\n"
                            "module.exports = { handler: statusController.getStatus };\n"
                        ),
                    },
                    {
                        "path": "src/application.js",
                        "content": (
                            "const board = require('./routes/boardroute');\n"
                            "const statusRoute = require('./routes/statusroute');\n"
                            "module.exports = { board, statusRoute };\n"
                        ),
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('status', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    execution = await _run(
        harness, engineer=engineer, reviewer=reviewer, process_runner=RealGitProcessRunner()
    )

    assert not any("never uses it" in item for item in execution.result.blocking_issues)
    assert execution.result.status == "approved"


@pytest.mark.asyncio
async def test_a_convention_scanned_directory_does_not_report_its_new_file(tmp_path: Path) -> None:
    """Where no neighbour is referenced either, the checkout discovers files by convention."""
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    # A directory of its own: the fixture's src/routes is wired by hand from src/index.js
    # and would correctly be read as hand-wired.
    routers = workspace / "src" / "routers"
    routers.mkdir(parents=True, exist_ok=True)
    # Nothing imports these: the framework reaches them by their location alone.
    (routers / "contactrouter.js").write_text(
        "module.exports = () => 'contact';\n", encoding="utf-8"
    )
    (routers / "aboutrouter.js").write_text("module.exports = () => 'about';\n", encoding="utf-8")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Add the routed pages")
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status route.",
                "files": [
                    {
                        "path": "src/routers/statusrouter.js",
                        "content": "module.exports = () => 'status';\n",
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('status', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    execution = await _run(
        harness, engineer=engineer, reviewer=reviewer, process_runner=RealGitProcessRunner()
    )

    # The subject here is the reachability gate staying silent. The workstream itself is
    # now also held to the integration promise, which a convention-scanned addition cannot
    # meet; that trade is recorded in the schema, and this test does not assert approval.
    assert not any(
        "no production file refers to it" in item for item in execution.result.blocking_issues
    )


@pytest.mark.asyncio
async def test_adding_to_an_existing_file_is_not_mistaken_for_a_rewrite(tmp_path: Path) -> None:
    """The rewrite gate must not stop an ordinary edit that keeps what is already there."""
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    existing = workspace / "src" / "routes" / "tiles.js"
    existing.parent.mkdir(parents=True, exist_ok=True)
    kept = "".join(f"// existing behaviour line {index}\n" for index in range(120))
    existing.write_text(kept, encoding="utf-8")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Add the tiles already on the page")
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status tile beside the existing ones.",
                "files": [
                    {"path": "src/routes/tiles.js", "content": f"{kept}status();\n"},
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('status', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    execution = await _run(
        harness, engineer=engineer, reviewer=reviewer, process_runner=RealGitProcessRunner()
    )

    assert execution.result.status == "approved"
    assert not any(
        "rewritten rather than edited" in item for item in execution.result.blocking_issues
    )


@pytest.mark.asyncio
async def test_a_rejection_over_medium_findings_still_reports_what_to_fix(tmp_path: Path) -> None:
    """A blocked attempt must carry its reason, whatever severity the reviewer assigned.

    Only critical and high findings became blocking issues, so a review that requested
    changes over medium ones failed the workstream with an empty issue list. Every retry
    was then told only that a gate was not satisfied and repeated the same code.
    """
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the server status route.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": (
                            "function statusRoute(req, res) {\n"
                            "  res.json({ status: 'ok' });\n"
                            "}\n\nmodule.exports = { statusRoute };\n"
                        ),
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('status', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    review = _review_payload(verdict="changes_requested")
    review["findings"] = [
        {
            "finding_id": "OBS-001",
            "severity": "medium",
            "title": "Status logging bypasses the diagnostics pipeline",
            "description": "The route logs with console.error instead of the diagnostics API.",
            "recommendation": "Emit a structured diagnostics event instead.",
            "file_path": "src/routes/status.js",
            "line_number": 2,
            "repository_id": "backend",
            "requirement_id": "backend-status-api",
            "responsibility": "implements",
        }
    ]
    reviewer = ScriptedLLMClient([review, review, review])

    execution = await _run(harness, engineer=engineer, reviewer=reviewer)

    assert execution.result.status == "failed"
    assert execution.result.blocking_issues == [
        "The route logs with console.error instead of the diagnostics API."
    ]


@pytest.mark.asyncio
async def test_advisory_wording_findings_are_not_handed_back_as_work(tmp_path: Path) -> None:
    """A `low` finding is plan feedback, not a change the engineer is meant to make.

    The reviewer is instructed to record a requirement its runtime cannot express -- or
    already satisfies -- as a passed check plus a low-severity `requirement` finding. The
    fallback that keeps the issue list from being empty took every finding, so -068's backend
    was handed "add explicit synchronization" by the same review that called the safety intent
    already met, and spent attempts on it.
    """
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the server status route.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": (
                            "function statusRoute(req, res) {\n"
                            "  res.json({ status: 'ok' });\n"
                            "}\n\nmodule.exports = { statusRoute };\n"
                        ),
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('status', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    review = _review_payload(verdict="changes_requested")
    review["findings"] = [
        {
            "finding_id": "OBS-001",
            "severity": "medium",
            "title": "Capacity limit is not tested",
            "description": "The endpoint test never exercises the configured capacity.",
            "recommendation": "Record more samples than the capacity and assert the cap.",
            "file_path": "test/status.test.js",
            "line_number": 2,
            "repository_id": "backend",
            "requirement_id": "backend-status-api",
            "responsibility": "implements",
        },
        {
            "finding_id": "OBS-002",
            "severity": "low",
            "title": "Synchronization wording does not match the runtime model",
            "description": "The requirement asks for a lock this single-threaded runtime lacks.",
            "recommendation": "Correct the requirement wording in the plan.",
            "file_path": None,
            "line_number": None,
            "repository_id": "backend",
            "requirement_id": "backend-status-api",
            "responsibility": "implements",
        },
    ]
    reviewer = ScriptedLLMClient([review, review, review])

    execution = await _run(harness, engineer=engineer, reviewer=reviewer)

    assert execution.result.status == "failed"
    assert execution.result.blocking_issues == [
        "The endpoint test never exercises the configured capacity."
    ]


def _review_payload(*, verdict: str) -> dict[str, Any]:
    """Return a schema-valid reviewer response for the backend workstream."""
    return {
        "verdict": verdict,
        "summary": "The backend implements its scoped status requirement.",
        "requirement_checks": [
            {
                "requirement_id": "backend-status-api",
                "passed": verdict == "approved",
                "evidence": "src/routes/status.js serves the status payload.",
            }
        ],
        "findings": [],
        "architecture_assessment": "The route follows the existing repository layout.",
        "security_assessment": "No credential or input-handling change was introduced.",
        "test_coverage_assessment": "The configured node test command covers the route.",
    }


def _changed_status_payload() -> dict[str, Any]:
    """Return one material route-and-test change for crash-recovery coverage."""
    return {
        "summary": "Update the server status route.",
        "files": [
            {
                "path": "src/routes/status.js",
                "content": (
                    "function statusRoute(req, res) {\n"
                    "  res.json({ status: 'ok', recovered: true });\n"
                    "}\n\nmodule.exports = { statusRoute };\n"
                ),
            },
            {
                "path": "test/status.test.js",
                "content": (
                    "const { test } = require('node:test');\n"
                    "const assert = require('node:assert');\n"
                    "const { statusRoute } = require('../src/routes/status');\n\n"
                    "test('status route responds', () => {\n"
                    "  assert.ok(typeof statusRoute === 'function');\n"
                    "});\n"
                ),
            },
        ],
    }


def _unchanged_status_payload() -> dict[str, Any]:
    """Return the fixture's already-committed route and test byte-for-byte."""
    return {
        "summary": "Confirm the existing server status route.",
        "files": [
            {
                "path": "src/routes/status.js",
                "content": (
                    "function statusRoute(req, res) {\n"
                    "  res.json({ status: 'ok' });\n"
                    "}\n\nmodule.exports = { statusRoute };\n"
                ),
            },
            {
                "path": "test/status.test.js",
                "content": (
                    "const { test } = require('node:test');\n"
                    "const assert = require('node:assert');\n"
                    "const { statusRoute } = require('../src/routes/status');\n\n"
                    "test('status route responds', () => {\n"
                    "  let body = null;\n"
                    "  statusRoute({}, { json: (value) => { body = value; } });\n"
                    "  assert.deepStrictEqual(body, { status: 'ok' });\n"
                    "});\n"
                ),
            },
        ],
    }


def _workstream() -> RepositoryWorkstreamPlan:
    """Scope the backend to its own requirement, with the frontend's explicitly excluded."""
    return RepositoryWorkstreamPlan.model_validate(
        {
            "workstream_id": "backend",
            "repository_id": "backend",
            "role": "backend",
            "requirement_ids": ["backend-status-api"],
            "scoped_requirements": [
                {
                    "requirement_id": "backend-status-api",
                    "acceptance_criterion_ids": ["backend-status-api:ac-1"],
                    "responsibility": "implements",
                }
            ],
            "out_of_scope_requirements": ["frontend-status-tile"],
            "shared_requirements": [],
            "responsibilities": ["Serve the server status payload."],
            "task_ids": ["backend-status-task"],
            "dependency_workstream_ids": [],
            "contract_sections_consumed": [],
            "contract_sections_implemented": [],
            "acceptance_criteria": ["The status route responds."],
            "test_requirements": ["Run the configured node test command."],
            "documentation_requirements": [],
            "expected_files_or_areas": ["src/routes"],
            "required": True,
            "implementation_expectations": [
                {
                    "requirement_id": "backend-status-api",
                    "expected_change_categories": ["route", "test"],
                    "expected_source_areas": ["src/routes"],
                    "tests_required": False,
                }
            ],
        }
    )


def _technical_prd() -> TechnicalPRDArtifact:
    """Return a PRD containing one backend and one frontend requirement."""
    return create_artifact(
        TechnicalPRDArtifact,
        workflow_id="feature-live",
        artifact_id="002_technical_prd.json",
        producer="product_manager",
        metadata={},
        payload={
            "title": "Server status",
            "solution_summary": "Expose and display server status.",
            "functional_requirements": [
                {
                    "requirement_id": "backend-status-api",
                    "description": "Serve a server status payload.",
                    "priority": "must",
                    "acceptance_criteria": ["The status route responds."],
                    "dependencies": [],
                },
                {
                    "requirement_id": "frontend-status-tile",
                    "description": "Display the server status tile.",
                    "priority": "must",
                    "acceptance_criteria": ["The tile renders the status."],
                    "dependencies": [],
                },
            ],
            "non_functional_requirements": [],
            "data_requirements": [],
            "integration_requirements": [],
            "security_requirements": [],
            "assumptions": [],
            "unresolved_questions": [],
        },
    )


def _contract() -> IntegrationContractArtifact:
    """Return an approved, contract-free integration contract for a single repository."""
    return create_artifact(
        IntegrationContractArtifact,
        workflow_id="feature-live",
        artifact_id="009_integration_contract.json",
        producer="feature_planner",
        metadata={},
        payload={
            "feature_id": "feature-live",
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
                "rollback_requirements": ["Revert the pull request."],
            },
            "owning_workstreams": ["backend"],
            "approved_at": datetime(2026, 8, 4, tzinfo=UTC),
            "openapi_document": None,
        },
    )


def _feature_payload() -> dict[str, object]:
    """Return a single-repository feature request matching the live checkout."""
    return {
        "feature_id": "feature-live",
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
                "repository_id": "backend",
                "name": "Backend",
                "role": "backend",
                "repository_url": "https://github.com/example/backend.git",
                "default_branch": "main",
            }
        ],
    }


def _git(root: Path, *arguments: str) -> None:
    """Run one git command against the fixture checkout with no shell."""
    subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True, timeout=30)


def _git_output(root: Path, *arguments: str) -> str:
    """Return bounded Git stdout for a fixture assertion."""
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _git_repo_with_committed_secret(root: Path) -> None:
    """Create a checkout whose config module already holds credential-shaped literals."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "--initial-branch=main")
    _git(root, "config", "user.email", "fixture@example.com")
    _git(root, "config", "user.name", "Fixture")
    (root / "config.js").write_text(
        "module.exports = {\n"
        "  githubToken: 'ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',\n"
        "  port: 3000,\n"
        "};\n",
        encoding="utf-8",
    )
    _git(root, "add", "config.js")
    _git(root, "commit", "-m", "initial")


@pytest.mark.asyncio
async def test_a_secret_committed_before_the_change_does_not_block_review(tmp_path: Path) -> None:
    """Editing a file whose secrets predate the change must stay reviewable.

    -067's backend added two keys to a config module that already carried nine
    credential-shaped literals. Redaction fired, the reviewer recorded
    `manual_review_required`, and the verdict was terminal -- while that same review said the
    visible feature source and required validations appeared compliant. Any repository keeping
    its settings in one such file could never have a change approved in it.
    """
    workspace = tmp_path / "repo"
    _git_repo_with_committed_secret(workspace)
    source = (workspace / "config.js").read_text(encoding="utf-8")
    # The change a workstream actually makes: a new setting beside the existing ones.
    changed = source.replace("  port: 3000,\n", "  port: 3000,\n  maxSamples: 50,\n")
    (workspace / "config.js").write_text(changed, encoding="utf-8")

    preexisting = await _redaction_is_preexisting(
        workspace=workspace,
        relative_path="config.js",
        source=changed,
        process_runner=RealGitProcessRunner(),
        cancellation_token=MockCancellationToken(),
    )

    assert preexisting is True


@pytest.mark.asyncio
async def test_a_secret_the_change_introduces_still_blocks_review(tmp_path: Path) -> None:
    """The gate must still fire when the attempt is what added the credential."""
    workspace = tmp_path / "repo"
    _git_repo_with_committed_secret(workspace)
    source = (workspace / "config.js").read_text(encoding="utf-8")
    changed = source.replace(
        "  port: 3000,\n",
        "  port: 3000,\n  awsKey: 'AKIAIOSFODNN7EXAMPLE',\n"
        "  awsSecret: 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',\n",
    )
    (workspace / "config.js").write_text(changed, encoding="utf-8")

    preexisting = await _redaction_is_preexisting(
        workspace=workspace,
        relative_path="config.js",
        source=changed,
        process_runner=RealGitProcessRunner(),
        cancellation_token=MockCancellationToken(),
    )

    assert preexisting is False


@pytest.mark.asyncio
async def test_a_secret_in_a_file_the_change_adds_still_blocks_review(tmp_path: Path) -> None:
    """A brand-new file has no committed version, so nothing about it is pre-existing."""
    workspace = tmp_path / "repo"
    _git_repo_with_committed_secret(workspace)
    added = "module.exports = { token: 'ghp_ZYXWVUTSRQPONMLKJIHGFEDCBA9876543210' };\n"
    (workspace / "added.js").write_text(added, encoding="utf-8")

    preexisting = await _redaction_is_preexisting(
        workspace=workspace,
        relative_path="added.js",
        source=added,
        process_runner=RealGitProcessRunner(),
        cancellation_token=MockCancellationToken(),
    )

    assert preexisting is False


@pytest.mark.asyncio
async def test_a_generated_contract_naming_a_module_does_not_make_it_reachable(
    tmp_path: Path,
) -> None:
    """`openapi.yaml` describes the component; it never renders one.

    The platform writes that file from the approved contract and forbids the engineer to
    change it, and it classifies as production source. Naming the new page there was enough
    for -072's console to look wired in: this gate stayed silent, so its registry hint was
    never produced, and three attempts were told to register a page without being told where.
    """
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    routes = workspace / "src" / "routes"
    routes.mkdir(parents=True, exist_ok=True)
    (routes / "boardroute.js").write_text("module.exports = {};\n", encoding="utf-8")
    (workspace / "src" / "index.js").write_text(
        "const board = require('./routes/boardroute');\nmodule.exports = { board };\n",
        encoding="utf-8",
    )
    # The generated contract already mentions the module the attempt is about to add.
    (workspace / "openapi.yaml").write_text(
        "paths:\n  /status:\n    get:\n      operationId: statusroute\n", encoding="utf-8"
    )
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Add the route the application registers")
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status route.",
                "files": [
                    {
                        "path": "src/routes/statusroute.js",
                        "content": "module.exports = { handler: () => 'ok' };\n",
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('status', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    execution = await _run(
        harness, engineer=engineer, reviewer=reviewer, process_runner=RealGitProcessRunner()
    )

    assert execution.result.status == "failed"
    issue = next(
        item
        for item in execution.result.blocking_issues
        if "no production file refers to it" in item
    )
    assert "Edit src/index.js, which is where this repository registers modules" in issue
    assert reviewer.calls == []


@pytest.mark.asyncio
async def test_a_helper_named_after_the_module_does_not_prove_it_is_reachable(
    tmp_path: Path,
) -> None:
    """`getStatusRoute` contains `statusroute`'s name and reaches nothing.

    The reference check grepped for the module's stem as a substring, so a helper whose
    identifier merely contains that name counted as a reference. -072 and -073 both added a
    page beside an API helper named after it, this gate stayed silent, and the console spent
    its attempts being told to register a page without ever being told where. -070 escaped
    only because its helper happened to start with a lowercase letter.
    """
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    routes = workspace / "src" / "routes"
    routes.mkdir(parents=True, exist_ok=True)
    (routes / "boardroute.js").write_text("module.exports = {};\n", encoding="utf-8")
    (workspace / "src" / "index.js").write_text(
        "const board = require('./routes/boardroute');\nmodule.exports = { board };\n",
        encoding="utf-8",
    )
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Add the route the application registers")
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status route and its helper.",
                "files": [
                    {
                        "path": "src/routes/statusroute.js",
                        "content": "module.exports = { handler: () => 'ok' };\n",
                    },
                    {
                        # Names the module without reaching it: the identifier merely
                        # contains the stem.
                        "path": "src/apiUtils/home.apiUtils.js",
                        "content": "export const getstatusroutehistory = () => [];\n",
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('status', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    execution = await _run(
        harness, engineer=engineer, reviewer=reviewer, process_runner=RealGitProcessRunner()
    )

    assert execution.result.status == "failed"
    issue = next(
        item
        for item in execution.result.blocking_issues
        if "no production file refers to it" in item
    )
    assert "Edit src/index.js, which is where this repository registers modules" in issue
    assert reviewer.calls == []


@pytest.mark.asyncio
async def test_cross_repository_diffs_come_from_the_results_not_a_captured_snapshot(
    tmp_path: Path,
) -> None:
    """The seam reviewer must be handed source that exists, not a pre-run view of the feature.

    Live feature -084 published two pull requests with the seam unreviewed. The provider had
    been bound to the state the orchestrator was constructed with -- the state before any
    child ran -- so every repository still had an empty changed-file list and the gate
    correctly reported that it had nothing to read. The results carry their own workspace and
    changed files and are current by construction.
    """
    settings = load_settings(workspace_root=tmp_path)
    workspace = tmp_path / "feature" / "backend"
    (workspace / "server" / "routes").mkdir(parents=True)
    (workspace / "server" / "routes" / "health.js").write_text(
        "router.get('/health-history', ctrl.list);\n", encoding="utf-8"
    )
    result = ChildWorkflowResultArtifact(
        schema_version="1.0",
        workflow_id="feature",
        artifact_id="011_child_workflow_result.backend.json",
        producer="child_workflow",
        timestamp=datetime(2026, 8, 25, tzinfo=UTC),
        metadata={},
        validation_status="valid",
        feature_id="feature",
        parent_workflow_id="feature",
        child_workflow_id="feature:backend",
        repository_id="backend",
        workstream_id="backend",
        branch_name="ai/feature/backend/health",
        workspace_path=str(workspace),
        code_completion_artifact_id="006_code_completion.json",
        review_artifact_id="007_review.json",
        changed_files=[],
        validation_results=[],
        status="approved",
        blocking_issues=[],
        pull_request_readiness=True,
        contract_sections_consumed=[],
        contract_sections_implemented=["healthHistory"],
        production_files_changed=["server/routes/health.js"],
    )

    changes = await LiveCrossRepositoryDiffs(settings=settings).collect(
        feature_id="feature", child_results=[result]
    )

    [change] = changes
    assert change.repository_id == "backend"
    assert change.contract_sections_implemented == ["healthHistory"]
    assert [item["path"] for item in change.files] == ["server/routes/health.js"]
    assert "router.get('/health-history', ctrl.list);" in change.files[0]["content"]


@pytest.mark.asyncio
async def test_the_file_this_attempt_added_is_never_named_as_the_place_to_wire_it_in(
    tmp_path: Path,
) -> None:
    """A file that did not exist before this attempt cannot be where the repository mounts things.

    Both halves of the wiring answer are claims about what the repository *already* does --
    "where this repository registers modules of this kind", "the one file reaching the
    neighbouring X". The neighbours were filtered against the attempt's own additions and the
    referrers were not, so the added file could be returned as the place to wire itself in.

    A canary run of a Python service was told: "Edit app/status_feed.py, which is where this
    repository registers modules of this kind: it already reaches the neighbouring
    app/status.py" -- about the file the same sentence had just called unreachable, which
    imported nothing at all. It qualified only by containing the token `status`. Four attempts
    followed the instruction and the module was no more reachable at the end than at the
    start: referencing a module from itself cannot mount it.
    """
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    modules = workspace / "src" / "modules"
    modules.mkdir(parents=True, exist_ok=True)
    # A neighbour whose only referrer will be the file this attempt adds. Nothing else in the
    # checkout mentions it, so the added file is the sole candidate the search can return.
    (modules / "statusboard.js").write_text("module.exports = {};\n", encoding="utf-8")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Add a neighbour nothing refers to")
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add a module that names its neighbour but is mounted nowhere.",
                "files": [
                    {
                        "path": "src/modules/statusfeed.js",
                        "content": (
                            "const board = require('./statusboard');\n"
                            "module.exports = { feed: () => board };\n"
                        ),
                    },
                    {
                        "path": "test/statusfeed.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('feed', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])

    execution = await _run(
        harness, engineer=engineer, reviewer=reviewer, process_runner=RealGitProcessRunner()
    )

    assert execution.result.status == "failed"
    added = "src/modules/statusfeed.js"
    repairs = execution.result.metadata.get("wiring_repairs") or []
    assert all(item["target_path"] != item["added_path"] for item in repairs), repairs
    assert all(item["target_path"] != added for item in repairs), repairs
    # And the sentence handed to the engineer must not instruct it to edit that file either.
    guidance = " ".join(execution.result.blocking_issues)
    assert f"Edit {added}" not in guidance, guidance


@pytest.mark.asyncio
async def test_a_routed_remediation_calls_the_model_configured_for_that_role(
    tmp_path: Path,
) -> None:
    """The routing decision has to change which model is actually called, or it is bookkeeping.

    The Engineer abstraction is the same one an initial implementation uses -- deliberately, so
    there is no second fix agent -- and what differs is the model boundary behind it. This
    asserts that boundary is selected from the persisted decision, and that the role's own
    configured model is the one that runs.
    """
    harness = await _harness(tmp_path)
    settings = harness["settings"]
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Remove the unused import the reviewer reported.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": "module.exports = { statusRoute: () => ({ status: 'ok' }) };\n",
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('status', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])
    requested_roles: list[ModelRole] = []

    def engineer_client_for(decision: ModelRoutingDecision | None) -> Any:
        """Record the persisted role the executor used and return the scripted boundary."""
        assert decision is not None
        assert isinstance(decision.role, ModelRole)
        requested_roles.append(decision.role)
        return engineer

    executor = LiveChildWorkstreamExecutor(
        settings=settings,
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=RecordingProcessRunner(),
        engineer_client_factory=engineer_client_for,
    )
    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    coding = settings.model_config_for_role(ModelRole.CODING, platform=AgentPlatform.OPENAI)
    routed = ModelRoutingDecision(
        execution_mode=ModelExecutionMode.REVIEW_REMEDIATION,
        role=ModelRole.CODING,
        model=coding.model,
        reasoning=coding.reasoning,
        classification=ReviewFixComplexity.COMPLEX,
        attempt=child.retry_count,
        routing_reason="The finding concerns a state transition.",
    )
    child = child.model_copy(update={"model_routing": routed.model_dump(mode="json")})

    try:
        execution = await executor.run(
            feature=feature,
            repository=repository,
            workstream=_workstream(),
            child=child,
            technical_prd=technical_prd,
            contract=contract,
            feedback=["Remove the unused import."],
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert requested_roles == [ModelRole.CODING]
    assert execution.code_completion is not None
    recorded = execution.code_completion.metadata["model_routing"]
    assert recorded["role"] == ModelRole.CODING.value
    assert recorded["model"] == coding.model
    assert execution.code_completion.metadata["execution_mode"] == "REVIEW_REMEDIATION"
    # The engineer was told what kind of job this is, without being told which model it is.
    instructions, input_text = engineer.calls[0]
    assert "REVIEW_REMEDIATION" in input_text
    assert coding.model not in input_text
    assert coding.model not in instructions


@pytest.mark.asyncio
async def test_a_child_written_before_routing_existed_uses_the_coding_role(
    tmp_path: Path,
) -> None:
    """A durable child with no routing decision must not be treated as an escalation."""
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Implement the status route.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": "module.exports = { statusRoute: () => ({ status: 'ok' }) };\n",
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\ntest('status', () => {});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])
    requested_roles: list[ModelRole] = []

    def record_role(decision: ModelRoutingDecision | None) -> Any:
        """Record the absent legacy decision as the conservative coding fallback."""
        requested_roles.append(ModelRole.CODING if decision is None else decision.role)
        return engineer

    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=RecordingProcessRunner(),
        engineer_client_factory=record_role,
    )
    feature, repository, child, technical_prd, contract = _live_run_context(harness)

    try:
        await executor.run(
            feature=feature,
            repository=repository,
            workstream=_workstream(),
            child=child,
            technical_prd=technical_prd,
            contract=contract,
            feedback=[],
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert requested_roles == [ModelRole.CODING]


# ---------------------------------------------------------------------------------------
# The workspace reset, asserted by its effect on a real checkout
#
# Everything below runs `_capture_and_reset` with the real `AsyncioProcessRunner` against
# real Git. That is the point. The command it used to issue --
# `git add -A -- :(exclude)openapi.yaml` -- had two failure modes, and the test that
# covered it asserted the argv and passed through both of them:
#
#   exit 1   in a repository that gitignores the contract projection. Naming a path in an
#            `add` pathspec triggers Git's ignored-path refusal, and excluding it counts as
#            naming it.
#   exit 128 when a `.git/index.lock` is left behind by a killed Git process, which is what
#            this platform's process-group termination produces.
#
# Both were reproduced against real Git before this code was changed. Each fixture below is
# one of the shapes that reproduction covered.
# ---------------------------------------------------------------------------------------


def _reset_fixture(root: Path, *, projection: str) -> None:
    """Build a checkout whose relationship to the contract projection is `projection`.

    ``tracked``   the repository contains and tracks `openapi.yaml`.
    ``absent``    the repository has no `openapi.yaml` at all.
    ``ignored``   the repository gitignores `openapi.yaml`.
    ``empty``     a base commit and nothing else, so there is nothing to stage.
    """
    node_javascript_repository(root)
    # Every real Node checkout ignores its dependency directory, and the reset's behaviour
    # towards that directory is one of the things being asserted, so the fixture has to be
    # honest about it: without this, `git add -A` stages the tree and `git reset --hard`
    # then deletes it as a staged path that is not in HEAD.
    ignored = ["node_modules/"] + (["openapi.yaml"] if projection == "ignored" else [])
    (root / ".gitignore").write_text("\n".join([*ignored, ""]), encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "Record what this repository ignores")
    if projection == "tracked":
        (root / "openapi.yaml").write_text("openapi: 3.0.0\n", encoding="utf-8")
        _git(root, "add", "-A")
        _git(root, "commit", "-m", "Track the contract projection")


async def _capture_and_reset(harness: dict[str, Any], workspace: Path) -> str | None:
    """Drive the real reset, with the real process runner, against one fixture."""
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=ScriptedLLMClient([]),
        reviewer_client=ScriptedLLMClient([]),
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        # No process_runner: the real AsyncioProcessRunner, so Git actually runs.
    )
    return await executor._capture_and_reset(workspace)


@pytest.mark.parametrize("projection", ["tracked", "absent", "ignored", "empty"])
@pytest.mark.asyncio
async def test_capture_and_reset_restores_the_tree_whatever_the_projection_is(
    tmp_path: Path, projection: str
) -> None:
    """The reset succeeds, returns the tree to HEAD, and keeps what it must, on every shape.

    `ignored` is the fixture that made the old command exit 1 and end the feature on every
    retry, and `absent` is the one the task requires: a repository with no `openapi.yaml`.
    """
    harness = await _harness(tmp_path)
    workspace = tmp_path / "fixtures" / projection
    _reset_fixture(workspace, projection=projection)
    head = _git_output(workspace, "rev-parse", "HEAD")
    dependencies = workspace / "node_modules" / "left-pad"
    dependencies.mkdir(parents=True)
    (dependencies / "index.js").write_text("module.exports = () => {};\n", encoding="utf-8")
    (workspace / "openapi.yaml").write_text("openapi: 3.0.0\ninfo: {}\n", encoding="utf-8")
    if projection != "empty":
        (workspace / "src" / "written.js").write_text(
            "module.exports = { written: true };\n", encoding="utf-8"
        )

    try:
        captured = await _capture_and_reset(harness, workspace)
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    # The command succeeded: a failure would have raised RepositoryWorkspaceError, which is
    # exactly what ended AB-Feature-132.
    assert _git_output(workspace, "rev-parse", "HEAD") == head
    assert _git_output(workspace, "status", "--porcelain", "--untracked-files=no") == ""
    # A repository that tracks the projection gets its committed version back, like any
    # other tracked file. Where it is untracked the reset removes it on purpose: the next
    # attempt rewrites it from the approved contract, and carrying forward a copy the
    # previous attempt edited would accuse that attempt of a contract change it never made.
    assert (workspace / "openapi.yaml").is_file() is (projection == "tracked")
    if projection == "tracked":
        assert (workspace / "openapi.yaml").read_text(encoding="utf-8") == "openapi: 3.0.0\n"
    # An installed dependency tree costs minutes and holds nothing the model wrote.
    assert (dependencies / "index.js").is_file()
    if projection == "empty":
        assert captured is None
    else:
        assert captured is not None
        assert "written.js" in captured
        # The projection is the platform's; presenting it as the previous attempt's work
        # would ask the next attempt to explain a change it never made.
        assert "openapi.yaml" not in captured


@pytest.mark.asyncio
async def test_capture_and_reset_clears_an_index_lock_a_killed_attempt_left(
    tmp_path: Path,
) -> None:
    """A lock left by a terminated process group must not end the workstream.

    This is the shape that produced exit 128 in production. The platform kills process
    groups on cancellation and on worker death, so an attempt killed mid-`git` leaves
    `.git/index.lock`, and this method's staging command is the next attempt's first Git
    write. Reproduced here by leaving the lock exactly as a killed process would.
    """
    harness = await _harness(tmp_path)
    workspace = tmp_path / "fixtures" / "locked"
    _reset_fixture(workspace, projection="absent")
    (workspace / "src" / "written.js").write_text(
        "module.exports = { written: true };\n", encoding="utf-8"
    )
    lock = workspace / ".git" / "index.lock"
    lock.write_text("", encoding="utf-8")

    try:
        captured = await _capture_and_reset(harness, workspace)
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert not lock.exists()
    assert captured is not None
    assert "written.js" in captured
    assert _git_output(workspace, "status", "--porcelain", "--untracked-files=no") == ""


@pytest.mark.asyncio
async def test_a_failing_workspace_command_says_what_git_actually_refused(
    tmp_path: Path,
) -> None:
    """A workspace failure names the condition, not just the exit code.

    The production record of AB-Feature-132 kept `WORKSPACE_COMMAND_FAILED_EXIT_128` and
    nothing else, so recovering what it meant took a reproduction hunt across eight fixture
    repositories months later. The cause is now recognised from Git's own stderr and stated
    in this platform's words -- never by quoting the checkout, which the diagnosed-failure
    contract forbids putting into durable state.
    """
    result = ProcessResult(
        command=("git", "add", "-A"),
        return_code=128,
        stdout="",
        stderr=(
            "fatal: Unable to create '/workspaces/feature-1/backend/.git/index.lock': File exists."
        ),
        duration_seconds=0.01,
    )

    error = RepositoryWorkspaceError(stage="retry staging", result=result)

    assert error.diagnostics[0].startswith("Workspace retry staging command")
    assert "WORKSPACE_COMMAND_FAILED_EXIT_128" in error.diagnostics[0]
    assert len(error.diagnostics) == 2
    assert "index lock" in error.diagnostics[1]
    assert "terminates a cancelled or abandoned attempt" in error.diagnostics[1]
    # The checkout's own words never reach a durable diagnostic.
    assert not any("/workspaces/" in item for item in error.diagnostics)


# ---------------------------------------------------------------------------------------
# The revision every attempt ran against
#
# 165 of 281 recorded workstreams carry no `current_revision`, 79 of them after at least one
# attempt, because the value was derived from review metadata: it needed a review to exist,
# and needed every one of that review's validation results to agree on one revision. An
# attempt stopped at any earlier gate had neither. Each case below is one of those gates.
# ---------------------------------------------------------------------------------------


def _workspace_revision(workspace: Path) -> str:
    """Measure the checkout the same way the executor does, for an equality assertion."""
    return calculate_repository_revision_sync(workspace).combined_fingerprint


def _tests_only_payload() -> dict[str, Any]:
    """Return a change the completeness gate rejects: tests, and no production source."""
    return {
        "summary": "Add a test for the status route.",
        "files": [
            {
                "path": "test/status.test.js",
                "content": "const { test } = require('node:test');\ntest('x', () => {});\n",
            }
        ],
    }


def _unwired_module_payload() -> dict[str, Any]:
    """Return a new production module nothing in the checkout refers to."""
    return {
        "summary": "Add the status reporter.",
        "files": [
            {
                "path": "src/routes/statusreporter.js",
                "content": "module.exports = { statusReporter: () => ({ status: 'ok' }) };\n",
            },
            {
                "path": "test/statusreporter.test.js",
                "content": "const { test } = require('node:test');\ntest('x', () => {});\n",
            },
        ],
    }


@pytest.mark.asyncio
async def test_an_attempt_blocked_at_preflight_still_records_its_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The earliest gate there is, and the one furthest from a review to derive from."""
    harness = await _harness(tmp_path)
    blocked = RepositoryPreflightResult(
        repository_id="backend",
        revision=_workspace_revision(harness["workspace"]),
        dependency_install_status="blocked_unsupported_runtime",
        validation_readiness="blocked",
        blocking_issues=[
            PreflightIssue(
                issue_id="PACKAGE_MANAGER_LOCKFILE_MISMATCH",
                category="package_manager_lockfile",
                severity="critical",
                description="The declared package manager does not match the committed lockfile.",
                evidence="package.json selects `yarn`; the lockfile selects `pnpm`.",
                recommended_action="Commit the lockfile the declared package manager produces.",
                automatically_repairable=False,
            )
        ],
    )

    async def _blocked(self: Any, *_args: Any, **_kwargs: Any) -> RepositoryPreflightResult:
        del self
        return blocked

    monkeypatch.setattr(RepositoryPreflight, "run", _blocked)
    engineer = ScriptedLLMClient([])
    reviewer = ScriptedLLMClient([])

    execution = await _run(harness, engineer=engineer, reviewer=reviewer)

    assert execution.result.status == "failed"
    # No model was consulted at all, so there is no review anywhere to derive a revision
    # from -- which is exactly the case the old derivation could never answer.
    assert engineer.calls == []
    assert reviewer.calls == []
    assert execution.result.current_revision == blocked.revision


@pytest.mark.asyncio
async def test_an_attempt_stopped_by_the_commit_gate_still_records_its_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repository's own linter rejected the source before anything was committed."""
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]

    async def _rejected(self: Any, state: Any) -> Any:
        del self, state
        # Written before the gate rejects it, as a real attempt's files are: the recorded
        # revision has to describe the checkout including what this attempt wrote.
        (workspace / "src" / "routes" / "rejected.js").write_text(
            "module.exports = {};\n", encoding="utf-8"
        )
        raise SourceValidationError(["eslint reported 3 problems in src/routes/rejected.js."])

    monkeypatch.setattr(EngineerAgent, "run", _rejected)
    reviewer = ScriptedLLMClient([])

    execution = await _run(harness, engineer=ScriptedLLMClient([]), reviewer=reviewer)

    assert execution.result.status == "failed"
    assert execution.result.failure_classification == "validation_source_failure"
    assert reviewer.calls == []
    assert execution.result.current_revision == _workspace_revision(workspace)
    # A lint rejection contributes no structured evidence, because the lint gate keeps none.
    # Stated here rather than left implicit: it is the half of 86- Part B that does *not*
    # change, and it is what keeps the -052b class of lineage reading as "unknown".
    assert execution.result.current_validation_results == []


@pytest.mark.asyncio
async def test_an_attempt_the_commit_gate_stopped_records_the_tests_that_stopped_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """86- Part B. The scoped run that ended the attempt reaches the durable record.

    The commit gate runs the change's own tests, and until now their verdict left the attempt
    as prose only: `current_validation_results` is filled from the review, a gate-rejected
    attempt has no review, and the field was empty on every one of them. The coarse failure
    counter reads exactly that field and reads empty as *unknown* -- correctly, since unknown
    is not agreement -- so its walk-back broke at every commit-gate rejection.
    AB-Feature-218's backend failed one suite on four consecutive attempts; the counter never
    read more than two.

    Asserted as the effect the counter actually consumes, not as the presence of a field:
    `failing_evidence_files` names the suite.
    """
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    suite = workspace / "test" / "bulk.test.js"
    suite.parent.mkdir(parents=True, exist_ok=True)
    suite.write_text("// the suite the gate ran\n", encoding="utf-8")
    failing = ValidationResult(
        command=("npm", "run", "test", "--", "test/bulk.test.js"),
        return_code=1,
        stdout="",
        stderr="",
        timed_out=False,
        duration_seconds=1.5,
        validation_type="test",
        stderr_summary=_summary_with_references(
            "FAIL test/bulk.test.js\n  BULK_APP_TRANSACTION_FAILED (500), expected 400\n",
            workspace,
        ),
        result_code="TEST_VALIDATION_FAILED_EXIT_1",
        required=True,
    )

    async def _rejected(self: Any, state: Any) -> Any:
        del self, state
        rejection = SourceValidationError([scoped_test_diagnostic(failing)])
        rejection.validation_results = (validation_summary(failing),)
        raise rejection

    monkeypatch.setattr(EngineerAgent, "run", _rejected)

    execution = await _run(harness, engineer=ScriptedLLMClient([]), reviewer=ScriptedLLMClient([]))

    assert execution.result.status == "failed"
    assert execution.result.failure_classification == "validation_source_failure"
    recorded = execution.result.current_validation_results
    assert len(recorded) == 1
    assert recorded[0]["command"] == "npm run test -- test/bulk.test.js"
    assert recorded[0]["required"] is True
    assert recorded[0]["passed"] is False
    # The effect: the counter can now see which file this attempt failed on, where before
    # this it saw nothing at all and stopped counting.
    assert failing_evidence_files(execution.result) == frozenset({"test/bulk.test.js"})


@pytest.mark.asyncio
async def test_an_attempt_rejected_by_the_wiring_gate_still_records_its_revision(
    tmp_path: Path,
) -> None:
    """A module nothing refers to is rejected before review, so there is none to read."""
    harness = await _harness(tmp_path)
    reviewer = ScriptedLLMClient([])

    execution = await _run(
        harness,
        engineer=ScriptedLLMClient([_unwired_module_payload()]),
        reviewer=reviewer,
        process_runner=RealGitProcessRunner(),
    )

    assert execution.result.status == "failed"
    assert execution.result.failure_classification == "implementation_missing"
    assert reviewer.calls == []
    assert execution.result.current_revision == _workspace_revision(harness["workspace"])


@pytest.mark.asyncio
async def test_an_attempt_rejected_by_completeness_still_records_its_revision(
    tmp_path: Path,
) -> None:
    """A tests-only change never reaches the reviewer, and used to record no revision."""
    harness = await _harness(tmp_path)
    reviewer = ScriptedLLMClient([])

    execution = await _run(
        harness, engineer=ScriptedLLMClient([_tests_only_payload()]), reviewer=reviewer
    )

    assert execution.result.status == "failed"
    assert execution.result.failure_classification == "implementation_missing"
    assert reviewer.calls == []
    assert execution.result.current_revision == _workspace_revision(harness["workspace"])


@pytest.mark.asyncio
async def test_an_attempt_rejected_by_review_records_the_revision_that_was_reviewed(
    tmp_path: Path,
) -> None:
    """A review exists here, so the derivation could answer -- but only under unanimity."""
    harness = await _harness(tmp_path)

    execution = await _run(
        harness,
        engineer=ScriptedLLMClient([_changed_status_payload()]),
        reviewer=ScriptedLLMClient([_review_payload(verdict="changes_requested")]),
    )

    assert execution.result.status == "failed"
    # Nothing was committed, so the checkout is still exactly what the reviewer judged.
    assert execution.result.current_revision == _workspace_revision(harness["workspace"])


@pytest.mark.asyncio
async def test_an_approved_attempt_records_the_revision_the_reviewer_validated(
    tmp_path: Path,
) -> None:
    """Approval publishes a commit, and must still name the revision that was reviewed.

    The measurement is taken before publication for exactly this reason. Afterwards HEAD has
    moved and the worktree is clean, so a revision measured then describes a checkout the
    reviewer never saw -- the same content under a different fingerprint.
    """
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    before_publication = _workspace_revision(workspace)

    execution = await _run(
        harness,
        engineer=ScriptedLLMClient([_changed_status_payload()]),
        reviewer=ScriptedLLMClient([_review_payload(verdict="approved")]),
        process_runner=RealGitProcessRunner(),
    )

    assert execution.result.status == "approved"
    recorded = execution.result.current_revision
    assert recorded
    # The commit happened, so the checkout has moved on from what was recorded.
    assert _workspace_revision(workspace) != recorded
    # And what was recorded is not the pre-attempt state either: it is the state the
    # reviewer actually judged, with the engineer's files in the worktree.
    assert recorded != before_publication


# ---------------------------------------------------------------------------------------
# The dependency tree the reset must not take, whatever ecosystem produced it
#
# `install_dependencies` is an `ExternalOperation`. Once it is journaled successful a retry
# replays the recorded result instead of running the installer again, so a tree the reset
# deletes is not rebuilt at a cost -- it is gone for the rest of the workstream, and every
# validation command afterwards runs against a repository with no dependencies.
#
# That reasoning is why `node_modules` was preserved. It applies identically everywhere else
# and only Node had been written down, so a Python checkout lost `.venv` on its first retry.
# Both pilot repositories are Node, so this never fired in production; it is latent, and the
# failure mode is silent -- nothing errors, validation simply starts failing for a reason
# nothing reports.
#
# Every assertion below is on the tree after a real reset with real Git, never on what
# `_cleanable_paths` returned. The record on this repository is that path-list assertions
# hide the defect they were written to catch.
# ---------------------------------------------------------------------------------------

# The fixture, the directory that ecosystem's installer fills, and one gitignored build
# directory that must still be removed -- because preserving the tree by dropping `-x` would
# leave the compiled output that walked a project-wide lint into a timeout.
_RESET_ECOSYSTEMS: dict[str, tuple[Callable[[Path], Path], str, str]] = {
    "node": (node_javascript_repository, "node_modules", "dist"),
    "python": (python_uv_repository, ".venv", "build"),
}


@pytest.mark.parametrize("ecosystem", sorted(_RESET_ECOSYSTEMS))
@pytest.mark.asyncio
async def test_a_reset_keeps_whatever_dependency_tree_this_checkout_actually_has(
    tmp_path: Path, ecosystem: str
) -> None:
    """A retry reset preserves the installed tree of the ecosystem it is actually run in.

    `python` is the row that fails before the preserved set is derived from the checkout:
    `.venv` was named as cleanable and `-x` removed it. `node` is the row that must keep
    passing, because a fix that protects one ecosystem by forgetting another has moved the
    defect rather than closed it.

    The reset still has to reset. The rejected attempt's own file goes, and so does the
    gitignored build output, which is the whole reason `-x` is there.
    """
    build_repository, dependencies, build_output = _RESET_ECOSYSTEMS[ecosystem]
    harness = await _harness(tmp_path)
    workspace = tmp_path / "fixtures" / ecosystem
    build_repository(workspace)
    # Real checkouts ignore both of these, and the reset's behaviour towards them is what is
    # being asserted, so the fixture has to be honest about it.
    (workspace / ".gitignore").write_text(f"{dependencies}/\n{build_output}/\n", encoding="utf-8")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Record what this repository ignores")
    installed = workspace / dependencies / "some-package" / "installed.txt"
    installed.parent.mkdir(parents=True)
    installed.write_text("installed once, replayed thereafter\n", encoding="utf-8")
    generated = workspace / build_output / "bundle.js"
    generated.parent.mkdir(parents=True)
    generated.write_text("// tens of megabytes of this\n", encoding="utf-8")
    rejected = workspace / "rejected.txt"
    rejected.write_text("what the previous attempt wrote\n", encoding="utf-8")

    try:
        await _capture_and_reset(harness, workspace)
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert installed.is_file(), (
        f"a {ecosystem} checkout lost the dependency tree no retry will reinstall"
    )
    assert not generated.exists(), "-x must still remove the build output that timed out a lint"
    assert not rejected.exists(), "the rejected attempt's own files must not survive the reset"


@pytest.mark.asyncio
async def test_a_checkout_no_runtime_is_detected_in_is_cleaned_completely(
    tmp_path: Path,
) -> None:
    """The unclassifiable case, decided deliberately rather than left to fall out.

    Detection is what selects the dependency install in the first place, so a checkout no
    runtime is recognised in had no journaled install step and holds no result a replay
    could fail to reproduce. Preserving ignored directories here would keep exactly the
    build output `-x` exists to remove, to protect a tree nothing created.

    This is a real cost if the guess is ever wrong -- a slower attempt while something is
    rebuilt -- and it is encoded so a later reader can weigh it against the alternative,
    which is a workstream that silently validates a repository with no dependencies.
    """
    harness = await _harness(tmp_path)
    workspace = tmp_path / "fixtures" / "unclassified"
    init_git_repository(workspace)
    (workspace / "README.md").write_text("# A checkout with no manifest\n", encoding="utf-8")
    (workspace / ".gitignore").write_text("cache/\n", encoding="utf-8")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-m", "Fixture baseline")
    cached = workspace / "cache" / "artifact.bin"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"\x00\x01\x02")

    try:
        await _capture_and_reset(harness, workspace)
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert not cached.exists()
    assert (workspace / "README.md").is_file(), "a reset never touches what HEAD tracks"


def test_a_checkout_the_platform_installs_into_always_has_a_detected_runtime(
    tmp_path: Path,
) -> None:
    """The premise the unclassified decision rests on, asserted instead of argued.

    Preserving nothing when no runtime is detected is only safe because detection is
    strictly wider than installation. `_dependency_installers` bootstraps a Node lockfile or
    a frozen uv lock, and those manifests are exactly what makes the runtime detectable, so
    a checkout that had a journaled install always resolves to something preserved and a
    checkout resolving to nothing never had one. The moment an installer is added whose
    manifest detection does not see, the fallback stops being safe and this fails.

    Deliberately the one assertion here that is not on a worktree: the effect of the reset
    is proven above, and what this covers is an invariant between two platform decisions
    that no single filesystem state can show.
    """
    node = node_javascript_repository(tmp_path / "node")
    python = python_uv_repository(tmp_path / "python", with_lockfile=True)

    for root in (node, python):
        assert _dependency_installers(root), (
            "the fixture no longer models a checkout the platform installs into"
        )
        assert _preserved_on_reset(root), (
            "a checkout the platform installs into must never resolve to preserving nothing"
        )


# --------------------------------------------------------------------------------------
# A gate rejection records what the attempt was, and the next retry's state carries it
#
# `validation_source_failure` is the dominant failure class, and until now it recorded
# nothing: AB-Feature-171's attempts 0, 2 and 4 each discarded a complete implementation
# with nothing durable but the linter's sentence. Everything here asserts effects -- the
# recorded artifact, the publication reads, and the next attempt's own completion.
# --------------------------------------------------------------------------------------


class FailingLintGateProcessRunner(RecordingProcessRunner):
    """Reject the repository's own lint gate with an ESLint-shaped report."""

    async def run(self, command: Any, cwd: Path, *_args: Any, **_kwargs: Any) -> ProcessResult:
        result = await super().run(command, cwd)
        if tuple(command)[:3] == ("npm", "run", "lint"):
            return ProcessResult(
                command=tuple(command),
                return_code=1,
                stdout=(
                    "/workspace/src/status.js\n"
                    "  1:7  error  'unused' is assigned a value but never used"
                    "  no-unused-vars\n"
                    "✖ 1 problem (1 error, 0 warnings)\n"
                ),
                stderr="",
                duration_seconds=0.01,
            )
        return result


def _gate_rejected_engineer_payload() -> dict[str, Any]:
    """One implementation whose file the scripted lint gate then rejects."""
    return {
        "summary": "Add the status module.",
        "files": [
            {
                "path": "src/status.js",
                "content": "const unused = 1;\nmodule.exports = { status: 'ok' };\n",
            }
        ],
    }


@pytest.mark.asyncio
async def test_a_gate_rejected_attempt_records_the_completion_it_produced(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The rejection keeps the attempt's record, and nothing can publish that record.

    Asserted on the artifact and on the publication reads themselves -- status, readiness,
    `_latest_approved_result` and `_has_unpublished_approved_child` -- because the record is
    only safe to keep if every read that selects work for publication refuses it.
    """
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient([_gate_rejected_engineer_payload()])
    reviewer = ScriptedLLMClient([])

    with caplog.at_level("WARNING", logger="agents.engineer.agent"):
        execution = await _run(
            harness,
            engineer=engineer,
            reviewer=reviewer,
            process_runner=FailingLintGateProcessRunner(),
        )

    assert execution.result.status == "failed"
    assert execution.result.metadata["source_validation_rejected"] is True
    completion = execution.code_completion
    assert completion is not None, "the gate rejection must keep the attempt's record"
    assert completion.completion_status == "failed"
    assert "src/status.js" in [change.path for change in completion.file_changes]
    assert completion.metadata["context_file_paths"], (
        "the record must say which files the attempt was shown"
    )
    assert execution.result.code_completion_artifact_id == completion.artifact_id
    # The reviewer never saw this attempt, so nothing about it is publishable.
    assert execution.result.pull_request_readiness is False
    identified = _with_attempt_identity(execution, repository_id="backend", attempt=1)
    request = StartFeatureRequest.model_validate(_feature_payload())
    feature = _initial_feature_state("feature-live", request)
    feature.artifacts.extend(_execution_artifacts(identified))
    feature.child_workflows["backend"] = ChildWorkflowReference(
        child_workflow_id="feature-live:backend",
        repository_id="backend",
        workstream_id="backend",
        branch_name=harness["branch"],
        workspace_path=str(harness["workspace"]),
        status=ChildWorkflowStatus.FAILED,
        retry_count=1,
    )
    assert _latest_approved_result(feature, "backend") is None
    assert not _has_unpublished_approved_child(feature)
    # The repair ran and could not clear the gate; that must be legible without journal
    # forensics (the outcome marker, not the wording, is the contract).
    assert any("outcome=repair_" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_a_gate_rejected_completion_reaches_the_next_retrys_state(tmp_path: Path) -> None:
    """The next attempt's own completion carries the rejected attempt's files.

    The union is the observable effect of the retry state holding the earlier completion:
    `_prior_attempt_file_changes` can only merge what the state carries, and until now the
    state never carried anything, so the merge was dead code on the live path (§1.8).
    """
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient([_gate_rejected_engineer_payload()])

    first = await _run_without_dispose(
        harness,
        engineer=engineer,
        reviewer=ScriptedLLMClient([]),
        process_runner=FailingLintGateProcessRunner(),
    )
    assert first.result.status == "failed"
    identified = _with_attempt_identity(first, repository_id="backend", attempt=1)

    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    feature.artifacts.extend(_execution_artifacts(identified))
    retry_engineer = ScriptedLLMClient(
        [
            {
                "summary": "Fix the lint finding only.",
                "files": [
                    {
                        "path": "src/fix.js",
                        "content": "module.exports = { fixed: true };\n",
                    }
                ],
            }
        ]
    )
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=retry_engineer,
        reviewer_client=ScriptedLLMClient(
            [
                _review_payload_with_finding(
                    verdict="changes_requested",
                    description="The fix is incomplete.",
                )
            ]
        ),
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=RecordingProcessRunner(),
    )
    try:
        second = await executor.run(
            feature=feature,
            repository=repository,
            workstream=_workstream(),
            child=child.model_copy(update={"retry_count": 2}),
            technical_prd=technical_prd,
            contract=contract,
            feedback=[],
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert second.code_completion is not None
    changed = {change.path for change in second.code_completion.file_changes}
    assert {"src/status.js", "src/fix.js"} <= changed, (
        "the retry's completion must be the union across attempts, not only its own fix"
    )


async def _run_without_dispose(
    harness: dict[str, Any],
    *,
    engineer: ScriptedLLMClient,
    reviewer: ScriptedLLMClient,
    process_runner: RecordingProcessRunner | None = None,
) -> Any:
    """Run one child workstream, keeping the harness alive for a follow-up attempt."""
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=process_runner or RecordingProcessRunner(),
    )
    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    return await executor.run(
        feature=feature,
        repository=repository,
        workstream=_workstream(),
        child=child,
        technical_prd=technical_prd,
        contract=contract,
        feedback=[],
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
    )


def _review_payload_with_finding(*, verdict: str, description: str) -> dict[str, Any]:
    """A reviewer response carrying exactly one blocking finding."""
    payload = _review_payload(verdict=verdict)
    payload["findings"] = [
        {
            "finding_id": "finding-1",
            "severity": "high",
            "title": "Incomplete fix",
            "description": description,
            "recommendation": "Complete the fix.",
            "file_path": None,
            "line_number": None,
            "repository_id": "backend",
            "requirement_id": "backend-status-api",
            "contract_reference": None,
            "responsibility": "implements",
            "validated_revision": None,
            "evidence": "The change does not clear the reported diagnostics.",
            "recommended_fix": "Address the reported diagnostics.",
            "finding_category": "requirement",
        }
    ]
    return payload


@pytest.mark.asyncio
async def test_a_preserved_retrys_publication_stages_the_union_across_attempts(
    tmp_path: Path,
) -> None:
    """What publication commits is the union lineage, asserted on the committed tree.

    This is the property the workspace reset existed to protect -- a fix-only retry must not
    silently drop the implementation from the commit -- proven here on the tree a real
    `git commit` produced, not on the artifact alone.
    """
    harness = await _harness(tmp_path)
    workspace = harness["workspace"]
    (workspace / "src" / "prior.js").write_text(
        "module.exports = { prior: true };\n", encoding="utf-8"
    )
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Correct the reported findings against the preserved attempt.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": (
                            "const { prior } = require('../prior');\n\n"
                            "function statusRoute(req, res) {\n"
                            "  res.json({ status: 'ok', prior });\n"
                            "}\n\nmodule.exports = { statusRoute };\n"
                        ),
                    },
                    {
                        "path": "test/status.retry.test.js",
                        "content": (
                            "const { test } = require('node:test');\n"
                            "const assert = require('node:assert');\n"
                            "const { statusRoute } = require('../src/routes/status');\n\n"
                            "test('status route responds', () => {\n"
                            "  assert.ok(typeof statusRoute === 'function');\n"
                            "});\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])
    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    feature.artifacts.append(_persisted_prior_completion(paths=["src/prior.js"], attempt=0))
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=RecordingProcessRunner(),
    )
    try:
        execution = await executor.run(
            feature=feature,
            repository=repository,
            workstream=_workstream(),
            child=child,
            technical_prd=technical_prd,
            contract=contract,
            feedback=[],
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert execution.result.status == "approved"
    assert execution.code_completion is not None
    assert execution.code_completion.commit_sha
    committed = _git_output(workspace, "show", "--name-only", "--format=", "HEAD").split()
    assert "src/prior.js" in committed, "the previous attempt's file must be in the commit"
    assert "src/routes/status.js" in committed, "the retry's own fix must be in the commit"
    assert "test/status.retry.test.js" in committed


def test_the_evidence_gate_binds_the_path_set_and_fingerprint_not_their_ordering() -> None:
    """Order-insensitive, content-sensitive: only ordering ceases to matter.

    The reviewer records paths in review order; the operation journal stores
    `expected_staged_files` sorted. Both name the same files with the same fingerprint, and
    the gate refusing over the ordering is what killed AB-Feature-184's recovery. A set that
    differs by one path, or the same set with a different fingerprint, must still refuse.
    """
    review_order = ["test/status.test.js", "src/routes/status.js"]
    review = _approved_review(review_order, "sha256:reviewed")

    require_reviewed_workspace_match(
        review,
        reviewed_paths=sorted(review_order),
        content_fingerprint="sha256:reviewed",
        evidence_required=True,
    )

    with pytest.raises(AgentArtifactError, match="no longer matches"):
        require_reviewed_workspace_match(
            review,
            reviewed_paths=["src/routes/status.js"],
            content_fingerprint="sha256:reviewed",
            evidence_required=True,
        )
    with pytest.raises(AgentArtifactError, match="no longer matches"):
        require_reviewed_workspace_match(
            review,
            reviewed_paths=[*sorted(review_order), "src/added.js"],
            content_fingerprint="sha256:reviewed",
            evidence_required=True,
        )
    with pytest.raises(AgentArtifactError, match="no longer matches"):
        require_reviewed_workspace_match(
            review,
            reviewed_paths=sorted(review_order),
            content_fingerprint="sha256:other-bytes",
            evidence_required=True,
        )


@pytest.mark.asyncio
async def test_recovery_accepts_a_receipt_whose_review_order_differs_from_the_journal(
    tmp_path: Path,
) -> None:
    """AB-Feature-184's F-1 shape: review order vs the journal's sorted list, same content.

    The Engineer names its test file before its production file, so the review evidence is
    recorded in that order while the commit operation journals the same paths sorted. The
    crash-window replay must still complete the publication from the receipt -- same files,
    same fingerprint, different order -- with no second model call.
    """
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Update the server status route.",
                "files": [
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\n"
                            "const assert = require('node:assert');\n"
                            "const { statusRoute } = require('../src/routes/status');\n\n"
                            "test('status route responds', () => {\n"
                            "  assert.ok(typeof statusRoute === 'function');\n"
                            "});\n"
                        ),
                    },
                    {
                        "path": "src/routes/status.js",
                        "content": (
                            "function statusRoute(req, res) {\n"
                            "  res.json({ status: 'ok', ordered: false });\n"
                            "}\n\nmodule.exports = { statusRoute };\n"
                        ),
                    },
                ],
            }
        ]
    )
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=RecordingProcessRunner(),
    )
    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    arguments: dict[str, Any] = {
        "feature": feature,
        "repository": repository,
        "workstream": _workstream(),
        "child": child,
        "technical_prd": technical_prd,
        "contract": contract,
        "feedback": [],
        "credentials": RequestScopedCredentials(openai_api_key=None, github_token=None),
    }
    try:
        first = await executor.run(**arguments)
        assert first.result.status == "approved"
        commit_operation = next(
            item
            for item in await harness["journal"].list_operations_for_workflow("feature-live")
            if item.operation_type is ExternalOperationType.CREATE_COMMIT
        )
        # The premise of this test: the journal stored the paths sorted while the review
        # evidence holds them in review order. If either side changes form, the scenario
        # has silently stopped covering the mismatch.
        stored = commit_operation.safe_metadata["expected_staged_files"]
        assert stored == sorted(stored)
        receipt = commit_operation.safe_metadata["publication_receipt"]
        recorded = receipt["review"]["metadata"]["reviewed_file_paths"]
        assert recorded != stored and sorted(recorded) == stored

        # The first result is intentionally discarded, as by a worker that died before the
        # child result checkpoint; the replay must recover, not re-code.
        replay = await executor.run(**arguments)
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    assert replay.result.status == "approved"
    assert replay.code_completion is not None
    assert replay.code_completion.metadata["publication_recovered"] is True
    assert len(engineer.calls) == 1, "recovery re-ran coding over an ordering difference"


@pytest.mark.asyncio
async def test_a_settled_publication_is_not_replayed_by_the_integration_fix_attempt(
    tmp_path: Path,
) -> None:
    """One journaled coding effect per attempt, and the settled attempt is not re-executed.

    The AB-Feature-184 sequence at the executor boundary: attempt 1 publishes and its result
    is checkpointed into parent state exactly as the fan-out records it -- renamed artifacts
    and all -- then integration review routes the repository back and the next attempt runs
    with the review's required fix as feedback. Recovery must decline the settled receipt,
    the Engineer must be asked once for the new attempt, and the journal must hold exactly
    one WRITE_FILE_CHANGES, one CREATE_COMMIT and one PUSH_BRANCH per attempt.
    """
    harness = await _harness(tmp_path)
    required_fix = "Integration review: make the bulk creation atomic."
    engineer = ScriptedLLMClient(
        [
            {
                "summary": "Add the status route.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": (
                            "function statusRoute(req, res) {\n"
                            "  res.json({ status: 'ok' });\n"
                            "}\n\nmodule.exports = { statusRoute };\n"
                        ),
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\n"
                            "const assert = require('node:assert');\n"
                            "const { statusRoute } = require('../src/routes/status');\n\n"
                            "test('status route responds', () => {\n"
                            "  assert.ok(typeof statusRoute === 'function');\n"
                            "});\n"
                        ),
                    },
                ],
            },
            {
                "summary": "Make the bulk creation atomic, as integration review asked.",
                "files": [
                    {
                        "path": "src/routes/status.js",
                        "content": (
                            "function statusRoute(req, res) {\n"
                            "  res.json({ status: 'ok', atomic: true });\n"
                            "}\n\nmodule.exports = { statusRoute };\n"
                        ),
                    },
                    {
                        "path": "test/status.test.js",
                        "content": (
                            "const { test } = require('node:test');\n"
                            "const assert = require('node:assert');\n"
                            "const { statusRoute } = require('../src/routes/status');\n\n"
                            "test('status route responds atomically', () => {\n"
                            "  let body = null;\n"
                            "  statusRoute({}, { json: (value) => { body = value; } });\n"
                            "  assert.deepStrictEqual(body, { status: 'ok', atomic: true });\n"
                            "});\n"
                        ),
                    },
                ],
            },
        ]
    )
    reviewer = ScriptedLLMClient(
        [_review_payload(verdict="approved"), _review_payload(verdict="approved")]
    )
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=RecordingProcessRunner(),
    )
    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    arguments: dict[str, Any] = {
        "feature": feature,
        "repository": repository,
        "workstream": _workstream(),
        "child": child,
        "technical_prd": technical_prd,
        "contract": contract,
        "feedback": [],
        "credentials": RequestScopedCredentials(openai_api_key=None, github_token=None),
    }
    try:
        published = await executor.run(**arguments)
        assert published.result.status == "approved"
        settled_sha = _git_output(harness["workspace"], "rev-parse", "HEAD")

        # What the parent records once the wave settles this attempt: the artifacts under
        # their repository-and-attempt scoped identities, and the child row referencing them.
        identified = _with_attempt_identity(
            published, repository_id="backend", attempt=child.retry_count
        )
        feature.artifacts.extend(_execution_artifacts(identified))
        # The integration changes-requested routing: attempt spent, repository owing work,
        # artifact references still naming the settled attempt.
        arguments["child"] = child.model_copy(
            update={
                "retry_count": child.retry_count + 1,
                "integration_retry_count": child.integration_retry_count + 1,
                "code_completion_artifact_id": identified.result.code_completion_artifact_id,
                "review_artifact_id": identified.result.review_artifact_id,
            }
        )
        arguments["feedback"] = [required_fix]

        corrected = await executor.run(**arguments)
        recorded = await harness["journal"].list_operations_for_workflow("feature-live")
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    # A real second attempt ran against the integration finding, on top of the settled one.
    assert corrected.result.status == "approved"
    assert corrected.code_completion is not None
    assert corrected.code_completion.metadata.get("publication_recovered") is not True
    assert len(engineer.calls) == 2
    assert required_fix in engineer.calls[1][1], (
        "the fix attempt's prompt must carry the integration review's required fix"
    )
    corrected_sha = _git_output(harness["workspace"], "rev-parse", "HEAD")
    assert corrected_sha != settled_sha
    assert _git_output(harness["workspace"], "rev-parse", "HEAD~1") == settled_sha
    # One journaled coding effect per attempt; the settled attempt's operations stay settled.
    kinds = [item.operation_type for item in recorded]
    assert kinds.count(ExternalOperationType.WRITE_FILE_CHANGES) == 2
    assert kinds.count(ExternalOperationType.CREATE_COMMIT) == 2
    assert kinds.count(ExternalOperationType.PUSH_BRANCH) == 2


@pytest.mark.asyncio
async def test_an_unreadable_receipt_is_a_diagnosed_refusal_not_an_unanticipated_crash(
    tmp_path: Path,
) -> None:
    """The genuine ambiguous case names what it knows instead of blaming the platform.

    A receipt is present, no durable child state holds its outcome, and its evidence can no
    longer be validated -- so recovery must refuse to re-run coding (the commit may exist
    without a recorded result). The refusal used to surface as "the platform did not
    anticipate RuntimeConfigurationError" with `platform_defect`, which is how a protective
    check ended AB-Feature-184. It must reach the child result as a named classification
    whose blocking issues tell a person which operation and branch to inspect.
    """
    harness = await _harness(tmp_path)
    engineer = ScriptedLLMClient([_changed_status_payload()])
    reviewer = ScriptedLLMClient([_review_payload(verdict="approved")])
    executor = LiveChildWorkstreamExecutor(
        settings=harness["settings"],
        git_environment={},
        engineer_client=engineer,
        reviewer_client=reviewer,
        journal=harness["journal"],
        cancellation_token=MockCancellationToken(),
        process_runner=RecordingProcessRunner(),
    )
    feature, repository, child, technical_prd, contract = _live_run_context(harness)
    arguments: dict[str, Any] = {
        "feature": feature,
        "repository": repository,
        "workstream": _workstream(),
        "child": child,
        "technical_prd": technical_prd,
        "contract": contract,
        "feedback": [],
        "credentials": RequestScopedCredentials(openai_api_key=None, github_token=None),
    }
    try:
        first = await executor.run(**arguments)
        assert first.result.status == "approved"
        commit_operation = next(
            item
            for item in await harness["journal"].list_operations_for_workflow("feature-live")
            if item.operation_type is ExternalOperationType.CREATE_COMMIT
        )
        # The result was never checkpointed (the worker-died shape), and the receipt's
        # evidence is no longer readable: the settled question cannot be answered.
        stripped = {
            key: value
            for key, value in commit_operation.safe_metadata.items()
            if key != "reviewed_content_fingerprint"
        }
        async with harness["database"].session() as session:
            await session.execute(
                sqlalchemy_update(ExternalOperationModel)
                .where(ExternalOperationModel.operation_id == commit_operation.operation_id)
                .values(safe_metadata=stripped)
            )
            await session.commit()

        with pytest.raises(AmbiguousPublicationReceiptError) as raised:
            await executor.run(**arguments)
        triaged = _failed_child_execution(
            feature,
            repository=repository,
            workstream=_workstream(),
            child=child,
            error=raised.value,
        )
    finally:
        await harness["database"].drop_schema()
        await harness["database"].dispose()

    # The refusal protected the work: coding was never re-run over the ambiguity.
    assert len(engineer.calls) == 1
    assert (
        raised.value.failure_classification
        is FeatureFailureClassification.UNCONFIRMED_EXTERNAL_EFFECT
    )
    issues = " ".join(triaged.result.blocking_issues)
    assert commit_operation.operation_id in issues
    assert harness["branch"] in issues
    assert "did not anticipate" not in issues
    assert triaged.result.status == "failed"
    assert (
        triaged.result.failure_classification
        == FeatureFailureClassification.UNCONFIRMED_EXTERNAL_EFFECT.value
    )
