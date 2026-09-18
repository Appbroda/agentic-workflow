"""Tests for protocol-backed external adapters using mocked network-capable clients."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from adapters import llm_adapter
from adapters.git_adapter import (
    GitPythonService,
    GitSafetyError,
    MockGitService,
    _validate_commit_files,
)
from adapters.github_adapter import GitHubAdapterError, MockGitHubService, PyGithubService
from adapters.llm_adapter import (
    ImageInput,
    LLMAdapterError,
    LLMResponse,
    MockCodingExecutor,
    OpenAILLMClient,
    ResponsesCodingExecutor,
    is_transport_fault,
)
from agents.shared.contracts import safe_error_diagnostics
from configs.model_roles import AgentPlatform, ModelRole
from configs.settings import Settings, load_settings
from services.cancellation import CancellationRequested
from tools.file_tools import WorkspacePathError
from tools.llm_call_record import last_llm_call_payload


class FakePullRequest:
    """A network-free substitute for the subset of a PyGithub pull request we use."""

    def __init__(self, comment_error: Exception | None = None) -> None:
        self.number = 42
        self.html_url = "https://github.com/example/platform/pull/42"
        self.title = "Implement adapters"
        self.labels: tuple[str, ...] = ()
        self.reviewers: list[str] = []
        self.issue_comments: list[str] = []
        self._comment_error = comment_error

    def add_to_labels(self, *labels: str) -> None:
        """Record labels that the live adapter would send to GitHub."""
        self.labels = labels

    def create_review_request(self, *, reviewers: list[str]) -> None:
        """Record reviewers that the live adapter would send to GitHub."""
        self.reviewers = reviewers

    def create_issue_comment(self, body: str) -> None:
        """Record the comment sent by the live adapter, or refuse it like the provider can."""
        if self._comment_error is not None:
            raise self._comment_error
        self.issue_comments.append(body)


class FakeProviderError(Exception):
    """A stand-in for a PyGithub HTTP error, carrying a status and an unsafe message."""

    def __init__(self, status: int) -> None:
        super().__init__(f"Resource not accessible by personal access token ghp_{status}secret")
        self.status = status


class FakeRepository:
    """A network-free PyGithub repository test double."""

    def __init__(
        self,
        comment_error: Exception | None = None,
        pull_read_error: Exception | None = None,
    ) -> None:
        self.pull_request = FakePullRequest(comment_error)
        self.create_pull_arguments: dict[str, str] | None = None
        self.issue_reads: list[int] = []
        self._pull_read_error = pull_read_error

    def create_pull(self, **kwargs: str) -> FakePullRequest:
        """Record pull-request creation arguments and return a test double."""
        self.create_pull_arguments = kwargs
        return self.pull_request

    def get_pull(self, pull_request_number: int) -> FakePullRequest:
        """Return the configured pull request, or refuse the read like the provider can."""
        assert pull_request_number == 42
        if self._pull_read_error is not None:
            raise self._pull_read_error
        return self.pull_request

    def get_issue(self, pull_request_number: int) -> None:
        """Refuse the issue read exactly as the pilot credential does, and record the attempt.

        The live token is refused on `/issues/{n}` with a 403 while pull-request reads and
        writes succeed. Failing here keeps that fact in the double: an adapter that reaches
        for the issue in order to comment cannot pass this test.
        """
        self.issue_reads.append(pull_request_number)
        raise FakeProviderError(403)


class FakeGitHubClient:
    """A network-free PyGithub root client test double."""

    def __init__(self, repository: FakeRepository) -> None:
        self.repository = repository
        self.requested_repositories: list[str] = []

    def get_repo(self, repository: str) -> FakeRepository:
        """Record repository lookup without network access."""
        self.requested_repositories.append(repository)
        return self.repository


def test_mock_and_pygithub_services_record_pull_request_operations() -> None:
    """Both GitHub implementations expose equivalent PR, label, reviewer, and comment behavior."""
    mock_service = MockGitHubService()
    mock_pull_request = mock_service.create_pull_request(
        "example/platform",
        title="Implement adapters",
        body="Adds protocol-backed external adapters.",
        source_branch="feature/adapters",
        target_branch="main",
    )
    mock_service.add_labels("example/platform", mock_pull_request.number, ["adapters", "adapters"])
    mock_service.request_reviewers("example/platform", mock_pull_request.number, ["reviewer"])
    mock_service.add_comment("example/platform", mock_pull_request.number, "Ready for review.")

    repository = FakeRepository()
    service = PyGithubService("request-scoped-token", client=FakeGitHubClient(repository))
    live_pull_request = service.create_pull_request(
        "example/platform",
        title="Implement adapters",
        body="Adds protocol-backed external adapters.",
        source_branch="feature/adapters",
        target_branch="main",
    )
    service.add_labels("example/platform", live_pull_request.number, ["adapters"])
    service.request_reviewers("example/platform", live_pull_request.number, ["reviewer"])
    service.add_comment("example/platform", live_pull_request.number, "Ready for review.")

    assert mock_service.labels[("example/platform", mock_pull_request.number)] == ["adapters"]
    assert mock_service.reviewers[("example/platform", mock_pull_request.number)] == ["reviewer"]
    assert mock_service.comments[("example/platform", mock_pull_request.number)] == [
        "Ready for review."
    ]
    assert live_pull_request.number == 42
    assert repository.create_pull_arguments == {
        "title": "Implement adapters",
        "body": "Adds protocol-backed external adapters.",
        "head": "feature/adapters",
        "base": "main",
    }
    assert repository.pull_request.labels == ("adapters",)
    assert repository.pull_request.reviewers == ["reviewer"]
    assert repository.pull_request.issue_comments == ["Ready for review."]
    # The comment must not depend on an issue read the credential may be refused.
    assert repository.issue_reads == []


def test_a_refused_pull_request_comment_reports_status_endpoint_and_likely_cause() -> None:
    """Every cross-link comment this platform ever posted failed, and no record said why.

    The journal row read `GithubException` with empty diagnostics, because the boundary
    dropped the only detail that identifies the failure: the HTTP status. It is an integer
    and the endpoint is the adapter's own choice, so both are recorded -- while the provider
    message, which can carry a credential, is not.
    """
    repository = FakeRepository(comment_error=FakeProviderError(403))
    service = PyGithubService("request-scoped-token", client=FakeGitHubClient(repository))

    with pytest.raises(GitHubAdapterError) as failure:
        service.add_comment("example/platform", 42, "Related pull requests:\n- one")

    diagnostics = safe_error_diagnostics(failure.value)
    assert failure.value.failure_classification == "pull_request_comment_status_403"
    assert any("status=403" in line for line in diagnostics)
    assert any("POST /repos/example/platform/issues/42/comments" in line for line in diagnostics)
    assert any("Pull requests: write" in line for line in diagnostics)
    # A cause the response did not actually state must not read as a finding.
    assert any("not confirmed" in line for line in diagnostics)
    # The provider's own message is never quoted, at any level of the chain.
    recorded = "\n".join((*diagnostics, str(failure.value)))
    assert "personal access token" not in recorded
    assert "ghp_" not in recorded


def test_a_refused_pull_request_read_is_not_reported_as_a_refused_write() -> None:
    """The failing call has to be the one named; blaming the write cost ten features.

    The historical 403 was on a *read* (`GET /issues/{n}`) and every record described a
    failed comment, which sent the diagnosis after a write permission the token already had.
    """
    repository = FakeRepository(pull_read_error=FakeProviderError(403))
    service = PyGithubService("request-scoped-token", client=FakeGitHubClient(repository))

    with pytest.raises(GitHubAdapterError) as failure:
        service.add_comment("example/platform", 42, "Related pull requests:\n- one")

    diagnostics = safe_error_diagnostics(failure.value)
    assert any("GET /repos/example/platform/pulls/42" in line for line in diagnostics)
    assert not any("POST" in line for line in diagnostics)
    assert any("Pull requests: read" in line for line in diagnostics)
    assert repository.pull_request.issue_comments == []


def test_a_comment_refused_without_a_status_still_names_the_failing_endpoint() -> None:
    """A provider error carrying no status must not silently produce an empty record."""
    repository = FakeRepository(comment_error=RuntimeError("connection reset"))
    service = PyGithubService("request-scoped-token", client=FakeGitHubClient(repository))

    with pytest.raises(GitHubAdapterError) as failure:
        service.add_comment("example/platform", 42, "Related pull requests:\n- one")

    diagnostics = safe_error_diagnostics(failure.value)
    assert failure.value.failure_classification == "pull_request_comment_failed"
    assert any("RuntimeError status=unreported" in line for line in diagnostics)
    assert any("POST /repos/example/platform/issues/42/comments" in line for line in diagnostics)
    assert "connection reset" not in "\n".join((*diagnostics, str(failure.value)))


def test_git_services_enforce_default_branch_and_force_push_safety(tmp_path: Path) -> None:
    """Safety rules reject a default-branch or force push before remote methods run."""
    mock_service = MockGitService()
    repository_path = mock_service.clone(
        "https://example.invalid/platform.git", tmp_path / "platform"
    )
    mock_service.create_branch(repository_path, "feature/adapters", base_branch="main")
    (repository_path / "adapter.py").write_text("VALUE = 1\n", encoding="utf-8")
    mock_service.commit(repository_path, "Implement adapters", files=["adapter.py"])

    result = mock_service.push(repository_path, "feature/adapters")

    assert result.branch == "feature/adapters"
    with pytest.raises(GitSafetyError, match="default branch"):
        mock_service.push(repository_path, "main")
    with pytest.raises(GitSafetyError, match="force pushes"):
        mock_service.push(repository_path, "feature/adapters", force=True)

    gitpython_service = GitPythonService(
        repository_factory=lambda _path: pytest.fail("unsafe pushes must not open a repository")
    )
    with pytest.raises(GitSafetyError, match="default branch"):
        gitpython_service.push(tmp_path, "main")
    with pytest.raises(GitSafetyError, match="force pushes"):
        gitpython_service.push(tmp_path, "feature/adapters", force=True)


def test_git_services_reject_unsafe_ref_names_and_secret_staging(tmp_path: Path) -> None:
    """Autonomous Git writes allow templates but reject refs, broad staging, and real envs."""
    service = MockGitService()
    repository_path = service.clone("https://example.invalid/platform.git", tmp_path / "platform")
    (repository_path / ".env.example").write_text(
        "OPENAI_API_KEY=your-api-key-here\n",
        encoding="utf-8",
    )
    (repository_path / "credentials.example.json").write_text(
        '{"accessToken":"your-token-here"}\n',
        encoding="utf-8",
    )

    with pytest.raises(GitSafetyError, match="safe Git ref"):
        service.create_branch(repository_path, "--upload-pack=evil", base_branch="main")
    with pytest.raises(GitSafetyError, match="explicit intended"):
        service.commit(repository_path, "Unsafe broad commit")
    with pytest.raises(GitSafetyError, match="secret-like"):
        service.commit(repository_path, "Unsafe secret commit", files=[".env"])
    with pytest.raises(GitSafetyError, match="secret-like"):
        service.commit(repository_path, "Unsafe deployed env commit", files=[".env.production"])
    with pytest.raises(GitSafetyError, match="secret-like"):
        service.commit(repository_path, "Unsafe secret data commit", files=["secrets.yaml"])

    committed = service.commit(
        repository_path,
        "Document environment template",
        files=[".env.example"],
    )

    assert committed
    assert service.commit(
        repository_path,
        "Document credential template",
        files=["credentials.example.json"],
    )


def test_the_commit_gate_refuses_key_material_whatever_the_file_is_called(
    tmp_path: Path,
) -> None:
    """The commit half of the rule `36-` unified, asserted on the commit being refused.

    Nothing has been committed -- this is preventive -- but the two gaps here are the two
    that did fire on the context side. `sendgrid.env` ends with `.env` rather than beginning
    with it, and a service-account key is indistinguishable from `package.json` by name.

    Every assertion drives the validator that both commit services call and checks the
    refusal, not `_is_sensitive_path`'s return value: the private predicate answering
    correctly is not what keeps a key out of a branch.
    """
    (tmp_path / "config").mkdir()
    (tmp_path / "sendgrid.env").write_text("SENDGRID_API_KEY=SG.live\n", encoding="utf-8")
    (tmp_path / "config" / "prod.env").write_text("DATABASE_URL=postgres://\n", encoding="utf-8")
    (tmp_path / "imls-1566290608647-c1739c763650.json").write_text(
        '{\n  "type": "service_account",\n  "project_id": "imls"\n}\n', encoding="utf-8"
    )
    (tmp_path / "release-signing.txt").write_text(
        "-----BEGIN RSA PRIVATE KEY-----\nAAAA\n-----END RSA PRIVATE KEY-----\n", encoding="utf-8"
    )
    (tmp_path / ".env.example").write_text("SENDGRID_API_KEY=\n", encoding="utf-8")
    (tmp_path / ".env.dist").write_text("SENDGRID_API_KEY=\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"name": "mailer"}\n', encoding="utf-8")
    (tmp_path / "tsconfig.json").write_text('{"compilerOptions": {}}\n', encoding="utf-8")

    for named in ("sendgrid.env", "config/prod.env"):
        with pytest.raises(GitSafetyError, match="secret-like"):
            _validate_commit_files(tmp_path, [named])
    for by_content in ("imls-1566290608647-c1739c763650.json", "release-signing.txt"):
        with pytest.raises(GitSafetyError, match="contents are key material"):
            _validate_commit_files(tmp_path, [by_content])

    committable = [".env.example", ".env.dist", "package.json", "tsconfig.json"]

    assert _validate_commit_files(tmp_path, committable) == committable


def test_the_commit_gate_refuses_git_metadata_but_not_a_file_that_is_merely_binary(
    tmp_path: Path,
) -> None:
    """The two rules that are not the credential rule, and the line between them.

    `.git` metadata has nothing to do with key material -- an autonomous commit rewriting a
    repository's own Git state is wrong on its face -- so it keeps its own clause and is
    asserted separately from anything the shared predicate answers.

    The other rule is fail-closed on a file that cannot be *read*, and it briefly meant
    something wider than that. The prefix read opened as UTF-8 and let `UnicodeDecodeError`
    escape, this gate caught it, and the platform could no longer commit any binary a
    repository contains -- no image, no font, no icon, no compiled asset -- while reporting
    that a file which reads perfectly could not be read. Both key-material markers are
    ASCII, so a file that is not text carries neither and commits.

    **This assertion is the reverse of the one it replaces**, deliberately: the previous
    version asserted the refusal, which is how the regression shipped with a green tier.

    A genuine read failure still refuses. It is provoked here with a directory rather than a
    permissions bit, because a suite that happens to run as root would quietly stop testing
    anything at all.
    """
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\xfd")
    (tmp_path / "assets").mkdir()

    with pytest.raises(GitSafetyError, match="secret-like"):
        _validate_commit_files(tmp_path, [".git/config"])
    with pytest.raises(GitSafetyError, match="cannot be read"):
        _validate_commit_files(tmp_path, ["assets"])

    assert _validate_commit_files(tmp_path, ["logo.png"]) == ["logo.png"]
    # A path with no file behind it is a staged deletion, which has no bytes to leak. Fail-
    # closed applies to a file that is there and unreadable, not to one the change removed.
    assert _validate_commit_files(tmp_path, ["src/removed.js"]) == ["src/removed.js"]


@dataclass(frozen=True, slots=True)
class FakeStreamEvent:
    """One Responses stream event, carrying a response only on the completion envelope."""

    type: str
    response: Any = None


class FakeResponseStream:
    """The event iterator a streamed Responses call returns."""

    def __init__(self, response: Any) -> None:
        self._events = iter(
            (
                FakeStreamEvent("response.created"),
                FakeStreamEvent("response.in_progress"),
                FakeStreamEvent("response.completed", response),
            )
        )

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> FakeStreamEvent:
        """Hand back the next event, ending the stream the way the SDK's iterator does."""
        try:
            return next(self._events)
        except StopIteration as end:
            raise StopAsyncIteration from end


class FakeResponsesAPI:
    """An asynchronous OpenAI Responses API test double that records all call arguments.

    Serves both transports, because the adapter picks one from the call's deadline: a long
    deadline is issued over a stream so the connection is not left idle for tens of minutes.
    A double that only answered plain POSTs would make the long-deadline agents -- the ones
    the choice exists for -- untestable.
    """

    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        """Return a preconfigured response, streamed or not, without network access."""
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return FakeResponseStream(self.response)
        return self.response


class FakeOpenAIClient:
    """A network-free OpenAI root client test double."""

    def __init__(self, response: Any) -> None:
        self.responses = FakeResponsesAPI(response)


@dataclass(frozen=True, slots=True)
class FakeUsage:
    """A minimal Responses API usage object."""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class FakeOpenAIResponse:
    """A minimal successful Responses API response object."""

    id: str
    model: str
    output_text: str | None
    usage: FakeUsage


class StaticLLMClient:
    """A configured LLM protocol test double with a fixed normalized response."""

    def __init__(self, response: LLMResponse) -> None:
        self.response = response

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
        """Return the configured response after confirming the executor supplied both inputs."""
        assert instructions
        assert input_text
        return self.response


def configure_model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure environment-backed settings without introducing real credentials into tests."""
    monkeypatch.setenv("OPENAI_REASONING_MODEL", "reasoning-model")
    monkeypatch.setenv("OPENAI_CODING_MODEL", "coding-model")
    monkeypatch.setenv("OPENAI_CODING_REASONING_EFFORT", "")
    monkeypatch.setenv("OPENAI_REVIEW_REASONING_EFFORT", "")
    monkeypatch.setenv("OPENAI_SCOPED_FIX_REASONING_EFFORT", "")
    monkeypatch.setenv("PLATFORM_API_KEY", "platform-key")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///adapter-test.db")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")


async def test_openai_and_coding_executors_use_mocks_and_workspace_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """OpenAI calls are configuration-bound and coding file writes remain within the workspace."""
    configure_model_environment(monkeypatch)
    response_payload = FakeOpenAIResponse(
        id="response-1",
        model="coding-model",
        output_text=json.dumps(
            {
                "summary": "Updated module.",
                "files": [{"path": "src/module.py", "content": "x = 1\n"}],
            }
        ),
        usage=FakeUsage(input_tokens=10, output_tokens=5),
    )
    fake_openai_client = FakeOpenAIClient(response_payload)
    llm_client = OpenAILLMClient(load_settings(), "engineer", client=fake_openai_client)

    response = await llm_client.respond(
        instructions="Follow the plan.", input_text="Implement module."
    )
    execution_result = await ResponsesCodingExecutor(StaticLLMClient(response)).execute(
        workspace_root=tmp_path,
        instructions="Follow the plan.",
        input_text="Implement module.",
    )

    assert fake_openai_client.responses.calls == [
        {
            "model": "coding-model",
            "instructions": "Follow the plan.",
            "input": "Implement module.",
            "temperature": 0.0,
            "store": False,
            # Read from policy, not restated: the deadline is a tuning decision and a test
            # that hardcodes it fails for the configuration changing rather than the adapter.
            "timeout": load_settings().agents["engineer"].timeout_seconds,
            # A coding call is one of the long ones, so it goes over a stream rather than
            # sitting on an idle connection for the length of its deadline.
            "stream": True,
        }
    ]
    assert response.input_tokens == 10
    # The adapter contract behind the planning-call journal: model facts are recorded
    # task-locally the moment the response exists, for the journal row wrapping the call.
    recorded = last_llm_call_payload()
    assert recorded is not None
    assert recorded["execution"]["model"] == "coding-model"
    assert recorded["execution"]["provider"] == "openai"
    assert execution_result.modified_files == (Path("src/module.py"),)
    assert (tmp_path / "src/module.py").read_text(encoding="utf-8") == "x = 1\n"

    mock_result = await MockCodingExecutor(file_updates={"mock.py": "value = 1\n"}).execute(
        workspace_root=tmp_path,
        instructions="Apply mock change.",
        input_text="Mock task.",
    )
    assert mock_result.modified_files == (Path("mock.py"),)

    traversal_response = LLMResponse(
        response_id="response-2",
        model="coding-model",
        output_text=json.dumps(
            {
                "summary": "Unsafe update.",
                "files": [{"path": "../outside.py", "content": "x = 1"}],
            }
        ),
        input_tokens=None,
        output_tokens=None,
    )
    with pytest.raises(WorkspacePathError, match="escapes workspace"):
        await ResponsesCodingExecutor(StaticLLMClient(traversal_response)).execute(
            workspace_root=tmp_path,
            instructions="Follow the plan.",
            input_text="Attempt unsafe write.",
        )

    oversized_response = LLMResponse(
        response_id="response-3",
        model="coding-model",
        output_text=json.dumps(
            {
                "summary": "Oversized update.",
                "files": [
                    {"path": "first.py", "content": "x = 1\n"},
                    {"path": "second.py", "content": "x" * 32},
                ],
            }
        ),
        input_tokens=None,
        output_tokens=None,
    )
    with pytest.raises(LLMAdapterError, match="size limit"):
        await ResponsesCodingExecutor(
            StaticLLMClient(oversized_response), max_file_bytes=16
        ).execute(
            workspace_root=tmp_path,
            instructions="Follow the plan.",
            input_text="Reject oversized updates before writing.",
        )
    assert not (tmp_path / "first.py").exists()


@pytest.mark.asyncio
async def test_reasoning_role_sends_the_deployment_timeout_to_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured deadline must reach each reasoning-role Responses request."""
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_REASONING_TIMEOUT_SECONDS", "901")
    fake_openai_client = FakeOpenAIClient(
        FakeOpenAIResponse(
            id="response-reasoning",
            model="reasoning-model",
            output_text="Technical PRD",
            usage=FakeUsage(input_tokens=2, output_tokens=3),
        )
    )
    client = OpenAILLMClient(load_settings(), "product_manager", client=fake_openai_client)

    await client.respond(instructions="Analyze the feature.", input_text="Build it.")

    assert fake_openai_client.responses.calls[0]["timeout"] == 901


def _coding_response(payload: dict[str, Any]) -> LLMResponse:
    """Wrap a coding-response body in the normalized LLM response the executor consumes."""
    return LLMResponse(
        response_id="response-edit",
        model="coding-model",
        output_text=json.dumps(payload),
        input_tokens=None,
        output_tokens=None,
    )


@pytest.mark.asyncio
async def test_a_targeted_edit_changes_one_place_and_keeps_the_rest_of_the_file(
    tmp_path: Path,
) -> None:
    """An edit names what it replaces, so the lines it does not name cannot be dropped."""
    existing = "".join(f"const tile{index} = render();\n" for index in range(200))
    (tmp_path / "tiles.js").write_text(f"{existing}export default tiles;\n", encoding="utf-8")
    response = _coding_response(
        {
            "summary": "Add the status tile beside the existing ones.",
            "files": [
                {
                    "path": "tiles.js",
                    "edits": [
                        {
                            "find": "export default tiles;",
                            "replace": "const statusTile = render();\nexport default tiles;",
                        }
                    ],
                }
            ],
        }
    )

    await ResponsesCodingExecutor(StaticLLMClient(response)).execute(
        workspace_root=tmp_path,
        instructions="Follow the plan.",
        input_text="Add the tile.",
    )

    written = (tmp_path / "tiles.js").read_text(encoding="utf-8")
    assert "const statusTile = render();" in written
    # Every line the edit did not name survived, which whole-file replacement could not promise.
    assert existing in written
    assert written.count("const tile") == 200


@pytest.mark.asyncio
async def test_an_edit_that_matches_nothing_writes_nothing(tmp_path: Path) -> None:
    """A `find` the file does not contain is rejected rather than guessed at."""
    (tmp_path / "tiles.js").write_text("const kept = 1;\n", encoding="utf-8")
    response = _coding_response(
        {
            "summary": "Edit a snippet that is not there.",
            "files": [{"path": "tiles.js", "edits": [{"find": "not in the file", "replace": "x"}]}],
        }
    )

    with pytest.raises(LLMAdapterError, match="const kept = 1;") as failure:
        await ResponsesCodingExecutor(StaticLLMClient(response)).execute(
            workspace_root=tmp_path,
            instructions="Follow the plan.",
            input_text="Edit the tile.",
        )

    assert (tmp_path / "tiles.js").read_text(encoding="utf-8") == "const kept = 1;\n"
    # The repair prompt is given the real text, so the next attempt can copy a snippet that
    # matches instead of guessing again and losing the whole attempt.
    assert "does not match the file" in str(failure.value)


@pytest.mark.asyncio
async def test_an_ambiguous_edit_is_rejected_instead_of_changing_the_first_match(
    tmp_path: Path,
) -> None:
    """A snippet appearing twice cannot be resolved, so the executor refuses to choose."""
    (tmp_path / "tiles.js").write_text("render();\nrender();\n", encoding="utf-8")
    response = _coding_response(
        {
            "summary": "Edit an ambiguous snippet.",
            "files": [{"path": "tiles.js", "edits": [{"find": "render();", "replace": "x();"}]}],
        }
    )

    with pytest.raises(LLMAdapterError, match="matches 2 places"):
        await ResponsesCodingExecutor(StaticLLMClient(response)).execute(
            workspace_root=tmp_path,
            instructions="Follow the plan.",
            input_text="Edit the tile.",
        )

    assert (tmp_path / "tiles.js").read_text(encoding="utf-8") == "render();\nrender();\n"


@pytest.mark.asyncio
async def test_openai_client_omits_temperature_for_gpt5_responses_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPT-5 Responses models reject temperature, so use their configured default instead."""
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_CODING_MODEL", "gpt-5-mini")
    response_payload = FakeOpenAIResponse(
        id="response-gpt5",
        model="gpt-5-mini",
        output_text="Implementation plan.",
        usage=FakeUsage(input_tokens=10, output_tokens=5),
    )
    fake_openai_client = FakeOpenAIClient(response_payload)
    llm_client = OpenAILLMClient(load_settings(), "engineer", client=fake_openai_client)

    await llm_client.respond(instructions="Follow the plan.", input_text="Implement module.")

    assert fake_openai_client.responses.calls == [
        {
            "model": "gpt-5-mini",
            "instructions": "Follow the plan.",
            "input": "Implement module.",
            "store": False,
            "timeout": load_settings().agents["engineer"].timeout_seconds,
            "stream": True,
        }
    ]


@pytest.mark.asyncio
async def test_openai_client_omits_temperature_for_declared_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model the deployment declares in MODEL_TEMPERATURE_UNSUPPORTED never receives it.

    AB-Feature-210: the adapter's gpt-5 prefix test guessed wrong for gpt-6 and the run died
    on its first provider call with a deterministic 400. The declaration, not another
    prefix, is how a new family says no.
    """
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_CODING_MODEL", "gpt-6-astra")
    monkeypatch.setenv("MODEL_TEMPERATURE_UNSUPPORTED", '["gpt-6-astra"]')
    response_payload = FakeOpenAIResponse(
        id="response-gpt6",
        model="gpt-6-astra",
        output_text="Implementation plan.",
        usage=FakeUsage(input_tokens=10, output_tokens=5),
    )
    fake_openai_client = FakeOpenAIClient(response_payload)
    llm_client = OpenAILLMClient(load_settings(), "engineer", client=fake_openai_client)

    await llm_client.respond(instructions="Follow the plan.", input_text="Implement module.")

    assert fake_openai_client.responses.calls == [
        {
            "model": "gpt-6-astra",
            "instructions": "Follow the plan.",
            "input": "Implement module.",
            "store": False,
            "timeout": load_settings().agents["engineer"].timeout_seconds,
            "stream": True,
        }
    ]


@pytest.mark.asyncio
async def test_openai_client_keeps_temperature_for_undeclared_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An undeclared non-gpt-5 model keeps the configured temperature: the provider stays
    the authority for everything the deployment has not declared."""
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_CODING_MODEL", "gpt-6-astra")
    monkeypatch.setenv("MODEL_TEMPERATURE_UNSUPPORTED", '["some-other-model"]')
    response_payload = FakeOpenAIResponse(
        id="response-gpt6-undeclared",
        model="gpt-6-astra",
        output_text="Implementation plan.",
        usage=FakeUsage(input_tokens=10, output_tokens=5),
    )
    fake_openai_client = FakeOpenAIClient(response_payload)
    settings = load_settings()
    llm_client = OpenAILLMClient(settings, "engineer", client=fake_openai_client)

    await llm_client.respond(instructions="Follow the plan.", input_text="Implement module.")

    call = fake_openai_client.responses.calls[0]
    assert call["temperature"] == settings.agents["engineer"].temperature


def test_declared_temperature_unsupported_parses_and_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declaration parses to a set, and a malformed one refuses Settings construction —
    startup, not the first live provider call, is where a bad declaration dies."""
    import pydantic

    configure_model_environment(monkeypatch)
    monkeypatch.setenv("MODEL_TEMPERATURE_UNSUPPORTED", '["gpt-6-astra", " padded "]')
    assert load_settings().declared_temperature_unsupported() == frozenset(
        {"gpt-6-astra", "padded"}
    )

    monkeypatch.setenv("MODEL_TEMPERATURE_UNSUPPORTED", '{"gpt-6-astra": true}')
    with pytest.raises(pydantic.ValidationError, match="MODEL_TEMPERATURE_UNSUPPORTED"):
        load_settings()

    monkeypatch.setenv("MODEL_TEMPERATURE_UNSUPPORTED", "not json")
    with pytest.raises(pydantic.ValidationError, match="MODEL_TEMPERATURE_UNSUPPORTED"):
        load_settings()


def test_declared_context_window_tokens_parses_and_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1: the context-window declaration parses like its siblings and dies at startup.

    Empty means nothing is declared; a valid object yields trimmed keys and int values
    through the same normalize pass the runtime applies; and every malformed shape -- a
    list, a string value, non-JSON -- refuses Settings construction with the one sentence,
    exactly the posture of MODEL_TEMPERATURE_UNSUPPORTED above and the ceiling parse.
    """
    import pydantic

    from configs.model_roles import normalize_context_window_tokens

    configure_model_environment(monkeypatch)
    monkeypatch.setenv("MODEL_CONTEXT_WINDOW_TOKENS", "")
    assert load_settings().declared_context_window_tokens() == {}

    monkeypatch.setenv("MODEL_CONTEXT_WINDOW_TOKENS", '{"gpt-6-astra": 400000, " padded ": 200000}')
    settings = load_settings()
    assert normalize_context_window_tokens(settings.declared_context_window_tokens()) == {
        "gpt-6-astra": 400000,
        "padded": 200000,
    }

    sentence = (
        "MODEL_CONTEXT_WINDOW_TOKENS must be a JSON object of model to context window in tokens"
    )
    for malformed in ('["gpt-6-astra"]', '{"m": "400k"}', "not json"):
        monkeypatch.setenv("MODEL_CONTEXT_WINDOW_TOKENS", malformed)
        with pytest.raises(pydantic.ValidationError, match=sentence):
            load_settings()


@pytest.mark.asyncio
async def test_openai_client_rejects_missing_output_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Responses without final text fail at the adapter boundary."""
    configure_model_environment(monkeypatch)
    response_payload = FakeOpenAIResponse(
        id="response-without-text",
        model="gpt-5-mini",
        output_text=None,
        usage=FakeUsage(input_tokens=10, output_tokens=5),
    )
    llm_client = OpenAILLMClient(
        load_settings(), "reviewer", client=FakeOpenAIClient(response_payload)
    )

    with pytest.raises(LLMAdapterError, match="no output text"):
        await llm_client.respond(instructions="Return JSON.", input_text="Review code.")


@pytest.mark.asyncio
async def test_an_adapter_failure_tells_an_operator_what_actually_went_wrong(
    tmp_path: Path,
) -> None:
    """A provider fault reaches the operator; a rejected edit's file content never does.

    Live feature -083 lost a repository to four consecutive provider faults and recorded only
    "The child workstream raised LLMAdapterError and could not complete", hiding whether the
    model returned nothing or the key was missing. But this same error type also quotes the
    workspace file a rejected edit failed to match -- deliberately, so the next attempt can
    copy real text -- and a repository's configuration file is exactly where a credential
    lives. So diagnostics are opt-in per raise site, never declared for the type.
    """
    empty = FakeOpenAIResponse(
        id="response-1",
        model="m",
        output_text="   ",
        usage=FakeUsage(input_tokens=1, output_tokens=0),
    )
    client = OpenAILLMClient(load_settings(), "engineer", client=FakeOpenAIClient(empty))

    with pytest.raises(LLMAdapterError) as provider_fault:
        await client.respond(instructions="Follow the plan.", input_text="Implement module.")

    assert safe_error_diagnostics(provider_fault.value) == (
        "Responses API returned no output text",
    )

    (tmp_path / "tiles.js").write_text("const apiKey = 'live-secret';\n", encoding="utf-8")
    response = _coding_response(
        {
            "summary": "Edit a snippet that is not there.",
            "files": [{"path": "tiles.js", "edits": [{"find": "absent", "replace": "x"}]}],
        }
    )

    with pytest.raises(LLMAdapterError) as rejected_edit:
        await ResponsesCodingExecutor(StaticLLMClient(response)).execute(
            workspace_root=tmp_path,
            instructions="Follow the plan.",
            input_text="Edit the tile.",
        )

    # The message carries the file's real text for the repair prompt, so it must not become
    # durable operator diagnostics.
    assert "live-secret" in str(rejected_edit.value)
    assert safe_error_diagnostics(rejected_edit.value) == ()


class ProviderOutageAPI:
    """A Responses API that fails the way a provider outage does: with its own exception."""

    class APIConnectionError(Exception):
        """Named for the SDK class AB-Feature-121 actually saw, and unrelated to it by type."""

    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **kwargs: Any) -> Any:
        """Raise a provider-owned exception carrying request material in its message."""
        del kwargs
        self.calls += 1
        raise self.APIConnectionError("Connection error while POSTing to https://api/x?key=sk-live")


class ProviderOutageClient:
    """A root client whose Responses API is down."""

    def __init__(self) -> None:
        self.responses = ProviderOutageAPI()


@pytest.mark.asyncio
async def test_a_provider_outage_arrives_as_this_adapter_s_own_fault_type() -> None:
    """Every retry allowance in the platform is keyed on ``LLMAdapterError``.

    AB-Feature-121's console called the reviewer, the SDK raised ``APIConnectionError``, and
    nothing recognised it: the child loop grants four attempts with backoff for an
    ``LLMAdapterError`` and the journal had already reserved two more against ``run_reviewer``
    -- attempt 1 of 3, ``failed_retryable``. The workstream was closed at ``retry_count`` 1
    with all of that unspent, and filed under the previous attempt's
    ``validation_source_failure``, which sends its reader to read source code for a dropped
    connection. AB-Feature-119's backend ended the same way.
    """
    client = OpenAILLMClient(load_settings(), "engineer", client=ProviderOutageClient())

    with pytest.raises(LLMAdapterError) as fault:
        await client.respond(instructions="Follow the plan.", input_text="Implement module.")

    # The type name identifies the fault and is a public symbol of an installed library.
    assert safe_error_diagnostics(fault.value) == (
        "the model provider call failed (APIConnectionError)",
    )
    assert fault.value.failure_classification == "APIConnectionError"
    # The provider builds its message from request material, so none of it is carried.
    assert "sk-live" not in str(fault.value)
    assert "api/x" not in str(fault.value)
    # Still reachable for anyone debugging the process rather than reading the record.
    assert isinstance(fault.value.__cause__, ProviderOutageAPI.APIConnectionError)


@pytest.mark.asyncio
async def test_a_streamed_provider_outage_is_wrapped_at_both_of_its_network_reads() -> None:
    """A stream fails either opening the connection or reading the next event off it."""

    class FailsMidStream:
        """Opens successfully, then drops while the platform is reading events."""

        class APIError(Exception):
            """A provider-owned exception raised from the event iterator."""

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            raise self.APIError("stream reset")

    class MidStreamAPI:
        async def create(self, **kwargs: Any) -> Any:
            del kwargs
            return FailsMidStream()

    class MidStreamClient:
        def __init__(self) -> None:
            self.responses = MidStreamAPI()

    client = OpenAILLMClient(load_settings(), "engineer", client=MidStreamClient())

    with pytest.raises(LLMAdapterError) as fault:
        async for _ in client.stream(instructions="Explain.", input_text="the plan"):
            pass

    assert safe_error_diagnostics(fault.value) == ("the model provider call failed (APIError)",)
    assert "stream reset" not in str(fault.value)


@pytest.mark.asyncio
async def test_wrapping_a_provider_fault_never_absorbs_a_cancellation() -> None:
    """Cancellation is this platform's decision, not a provider's fault, and is routed by type."""

    class CancelsInsteadOfAnswering:
        async def create(self, **kwargs: Any) -> Any:
            del kwargs
            raise CancellationRequested("feature cancellation requested")

    class CancellingClient:
        def __init__(self) -> None:
            self.responses = CancelsInsteadOfAnswering()

    client = OpenAILLMClient(load_settings(), "engineer", client=CancellingClient())

    with pytest.raises(CancellationRequested):
        await client.respond(instructions="Follow the plan.", input_text="Implement module.")


@pytest.mark.asyncio
async def test_a_routed_role_selects_the_model_and_effort_the_request_carries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The role decides the model and the effort; the named agent still bounds the call.

    This is the whole mechanism behind "one Engineer abstraction, routed roles": the timeout
    a coding call gets is a property of the engineer agent, and the model is a property of the
    role the router picked.
    """
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_SCOPED_FIX_MODEL", "fix-model")
    monkeypatch.setenv("OPENAI_SCOPED_FIX_REASONING_EFFORT", "high")
    response_payload = FakeOpenAIResponse(
        id="response-fix",
        model="fix-model",
        output_text="{}",
        usage=FakeUsage(input_tokens=1, output_tokens=1),
    )
    fake_openai_client = FakeOpenAIClient(response_payload)
    settings = load_settings()
    llm_client = OpenAILLMClient(
        settings, "engineer", client=fake_openai_client, model_role=ModelRole.SCOPED_FIX
    )

    await llm_client.respond(instructions="Fix the finding.", input_text="Remove the import.")

    assert (
        llm_client.model
        == settings.model_config_for_role(ModelRole.SCOPED_FIX, platform=AgentPlatform.OPENAI).model
    )
    assert llm_client.model_role is ModelRole.SCOPED_FIX
    assert fake_openai_client.responses.calls == [
        {
            "model": "fix-model",
            "instructions": "Fix the finding.",
            "input": "Remove the import.",
            "store": False,
            "timeout": settings.agents["engineer"].timeout_seconds,
            # A model whose name is not a GPT-5 identifier still takes the configured
            # temperature; the role changed the model, not the agent's operational policy.
            "temperature": settings.agents["engineer"].temperature,
            "reasoning": {"effort": "high"},
            "stream": True,
        }
    ]


@pytest.mark.asyncio
async def test_a_recovered_routing_decision_executes_its_persisted_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restart or env edit cannot change an attempt after its route was persisted."""
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_CODING_MODEL", "current-coding-model")
    monkeypatch.setenv("OPENAI_CODING_REASONING_EFFORT", "max")
    fake_openai_client = FakeOpenAIClient(
        FakeOpenAIResponse(
            id="response-recovered",
            model="persisted-coding-model",
            output_text="{}",
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
    )
    llm_client = OpenAILLMClient(
        load_settings(),
        "engineer",
        client=fake_openai_client,
        model_role=ModelRole.CODING,
        resolved_model="persisted-coding-model",
        resolved_reasoning_effort="high",
        resolved_model_variable="OPENAI_CODING_MODEL",
        resolved_routing_reason="Recovered the selection persisted before execution.",
    )

    response = await llm_client.respond(instructions="Continue safely.", input_text="Retry.")

    assert fake_openai_client.responses.calls[0]["model"] == "persisted-coding-model"
    assert fake_openai_client.responses.calls[0]["reasoning"] == {"effort": "high"}
    assert response.model == "persisted-coding-model"
    assert response.reasoning_effort == "high"
    assert response.model_variable == "OPENAI_CODING_MODEL"
    assert response.routing_reason == "Recovered the selection persisted before execution."


@pytest.mark.asyncio
async def test_an_effort_the_deployment_declares_unsupported_never_reaches_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capability normalization is central, so the request simply omits the level.

    Omitting leaves the provider's own default, which is what a deployment configuring nothing
    has always had. The model is never swapped for a different one.
    """
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_SCOPED_FIX_MODEL", "codex-style-model")
    monkeypatch.setenv("OPENAI_SCOPED_FIX_REASONING_EFFORT", "max")
    monkeypatch.setenv("MODEL_REASONING_UNSUPPORTED", '{"codex-style-model": ["max"]}')
    fake_openai_client = FakeOpenAIClient(
        FakeOpenAIResponse(
            id="response-normalized",
            model="codex-style-model",
            output_text="{}",
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
    )
    settings = load_settings()
    llm_client = OpenAILLMClient(
        settings, "engineer", client=fake_openai_client, model_role=ModelRole.SCOPED_FIX
    )

    await llm_client.respond(instructions="Fix the finding.", input_text="Remove the import.")

    assert llm_client.model == "codex-style-model"
    assert llm_client.reasoning_effort is None
    assert "reasoning" not in fake_openai_client.responses.calls[0]


@pytest.mark.asyncio
async def test_a_response_records_the_effort_it_actually_asked_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The effort travels with the response, so an artifact can state its own execution.

    Read back from configuration instead, an artifact written today would report whatever the
    deployment happens to be set to whenever somebody opens it -- which for a historical record
    is simply a different number.
    """
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_SCOPED_FIX_MODEL", "fix-model")
    monkeypatch.setenv("OPENAI_SCOPED_FIX_REASONING_EFFORT", "high")
    fake_openai_client = FakeOpenAIClient(
        FakeOpenAIResponse(
            id="response-effort",
            model="fix-model",
            output_text=json.dumps(
                {"summary": "Fixed.", "files": [{"path": "src/a.py", "content": "x = 1\n"}]}
            ),
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
    )
    llm_client = OpenAILLMClient(
        load_settings(), "engineer", client=fake_openai_client, model_role=ModelRole.SCOPED_FIX
    )

    response = await llm_client.respond(instructions="Fix it.", input_text="Here.")
    coding = await ResponsesCodingExecutor(StaticLLMClient(response)).execute(
        workspace_root=tmp_path, instructions="Fix it.", input_text="Here."
    )

    assert (response.provider, response.reasoning_effort) == ("openai", "high")
    assert response.model_role == "scoped_fix"
    assert response.model_variable == "OPENAI_SCOPED_FIX_MODEL"
    assert response.routing_reason
    # And through the coding boundary, which is what the engineer's artifact records.
    assert (coding.provider, coding.reasoning_effort) == ("openai", "high")
    assert coding.model_role == "scoped_fix"
    assert coding.model_variable == "OPENAI_SCOPED_FIX_MODEL"
    assert coding.routing_reason == response.routing_reason
    assert coding.model == "fix-model"


@pytest.mark.asyncio
async def test_a_declared_unsupported_effort_is_recorded_as_absent_rather_than_as_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record says what was sent. A level the request omitted was not asked for."""
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_SCOPED_FIX_MODEL", "codex-style-model")
    monkeypatch.setenv("OPENAI_SCOPED_FIX_REASONING_EFFORT", "max")
    monkeypatch.setenv("MODEL_REASONING_UNSUPPORTED", '{"codex-style-model": ["max"]}')
    fake_openai_client = FakeOpenAIClient(
        FakeOpenAIResponse(
            id="response-omitted",
            model="codex-style-model",
            output_text="{}",
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
    )
    llm_client = OpenAILLMClient(
        load_settings(), "engineer", client=fake_openai_client, model_role=ModelRole.SCOPED_FIX
    )

    response = await llm_client.respond(instructions="Fix it.", input_text="Here.")

    assert response.reasoning_effort is None


@pytest.mark.asyncio
async def test_a_long_deadline_call_is_issued_over_a_stream_and_answers_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call that may think for tens of minutes must not sit on an idle connection.

    AB-Feature-168's planning call was a plain POST at maximum effort. It ran twenty to
    forty-five minutes with no bytes moving in either direction -- exactly what an idle
    timeout, a proxy or a load balancer reaps -- and was killed three times in one morning,
    twice by APIConnectionError and once by APITimeoutError, having produced nothing.

    What the caller receives is unchanged. Only the transport is.
    """
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_REASONING_TIMEOUT_SECONDS", "1800")
    fake_openai_client = FakeOpenAIClient(
        FakeOpenAIResponse(
            id="response-planned",
            model="reasoning-model",
            output_text="The plan.",
            usage=FakeUsage(input_tokens=7, output_tokens=11),
        )
    )
    client = OpenAILLMClient(load_settings(), "planner", client=fake_openai_client)

    response = await client.respond(instructions="Plan the feature.", input_text="Build it.")

    assert fake_openai_client.responses.calls[0]["stream"] is True
    # Read off the completion envelope, so the response is whole rather than partial.
    assert response.output_text == "The plan."
    assert response.response_id == "response-planned"
    assert response.input_tokens == 7
    assert response.output_tokens == 11


@pytest.mark.asyncio
async def test_a_short_deadline_call_keeps_the_plain_transport_it_always_had(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The choice is made from the deadline, so short-deadline agents are untouched."""
    configure_model_environment(monkeypatch)
    fake_openai_client = FakeOpenAIClient(
        FakeOpenAIResponse(
            id="response-reviewer",
            model="review-model",
            output_text="Done.",
            usage=FakeUsage(input_tokens=1, output_tokens=1),
        )
    )
    # `reviewer` is configured at three minutes and routes the review role, not the reasoning
    # one, so it keeps the deadline the agent declares rather than the deployment's reasoning
    # one -- which leaves it below the threshold that changes transport.
    client = OpenAILLMClient(load_settings(), "reviewer", client=fake_openai_client)

    await client.respond(instructions="Review the change.", input_text="the diff")

    assert "stream" not in fake_openai_client.responses.calls[0]


@pytest.mark.asyncio
async def test_a_stream_that_never_completes_is_a_declared_fault_not_a_partial_answer() -> None:
    """Half a plan that happens to validate is worse than no plan at all."""

    class NeverCompletes:
        """Emits progress and then ends, without the provider ever finishing the response."""

        def __init__(self) -> None:
            self._events = iter((FakeStreamEvent("response.created"),))

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> FakeStreamEvent:
            try:
                return next(self._events)
            except StopIteration as end:
                raise StopAsyncIteration from end

    class TruncatingAPI:
        async def create(self, **kwargs: Any) -> Any:
            del kwargs
            return NeverCompletes()

    class TruncatingClient:
        def __init__(self) -> None:
            self.responses = TruncatingAPI()

    client = OpenAILLMClient(load_settings(), "engineer", client=TruncatingClient())

    with pytest.raises(LLMAdapterError) as fault:
        await client.respond(instructions="Follow the plan.", input_text="Implement module.")

    assert safe_error_diagnostics(fault.value) == (
        "Responses API stream ended without a completed response",
    )


@pytest.mark.asyncio
async def test_two_agents_sharing_a_role_can_be_asked_to_think_differently_hard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An agent that declares an effort has decided about its own job; the role is the default.

    The product manager and the planner are both the `reasoning` role, so the only lever over
    either was `OPENAI_REASONING_EFFORT` and it moved both at once. They are not the same job:
    the planner emits the architecture, the integration contract and the execution plan in one
    response, holding two dozen cross-references consistent, and AB-Feature-168 spent twenty to
    forty-five minutes per attempt on it at `max`.
    """
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "max")
    monkeypatch.setenv("OPENAI_PLANNER_REASONING_EFFORT", "")
    settings = load_settings()

    planner = OpenAILLMClient(settings, "planner", client=FakeOpenAIClient(None))
    product_manager = OpenAILLMClient(settings, "product_manager", client=FakeOpenAIClient(None))

    # The checked-in policy file eases the planner off the role's level, and only the planner.
    assert planner.reasoning_effort == "high"
    assert product_manager.reasoning_effort == "max"
    # The role still decides the model. This is a statement about effort, nothing else.
    assert planner.model == product_manager.model == "reasoning-model"


@pytest.mark.asyncio
async def test_a_deliberately_routed_role_keeps_its_own_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routing is a caller's decision and must not be overridden by the agent it reused."""
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "max")
    monkeypatch.setenv("OPENAI_PLANNER_REASONING_EFFORT", "")
    monkeypatch.setenv("OPENAI_SCOPED_FIX_MODEL", "fix-model")
    monkeypatch.setenv("OPENAI_SCOPED_FIX_REASONING_EFFORT", "low")
    settings = load_settings()

    routed = OpenAILLMClient(
        settings, "planner", client=FakeOpenAIClient(None), model_role=ModelRole.SCOPED_FIX
    )

    # The planner's declared `high` does not follow the call onto a role it was routed to.
    assert routed.reasoning_effort == "low"
    assert routed.model == "fix-model"


@pytest.mark.asyncio
async def test_the_deployment_overrides_the_checked_in_effort_without_a_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configuration an operator cannot change on a running deployment is not configuration.

    The policy file travels inside the image and the build refuses a dirty tree, so editing it
    costs a commit and a rebuild. The per-agent variable is read from the deployment at
    start-up, so an operator comparing effort levels restarts instead.
    """
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "max")
    monkeypatch.setenv("OPENAI_PLANNER_REASONING_EFFORT", "medium")
    settings = load_settings()

    planner = OpenAILLMClient(settings, "planner", client=FakeOpenAIClient(None))

    # `medium` from the deployment, not `high` from the policy file, nor `max` from the role.
    assert planner.reasoning_effort == "medium"


@pytest.mark.asyncio
async def test_a_per_agent_override_moves_only_the_agent_it_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The variable exists precisely so one of two agents sharing a role can be moved alone."""
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "max")
    monkeypatch.setenv("OPENAI_PLANNER_REASONING_EFFORT", "")
    monkeypatch.setenv("OPENAI_PRODUCT_MANAGER_REASONING_EFFORT", "low")
    settings = load_settings()

    product_manager = OpenAILLMClient(settings, "product_manager", client=FakeOpenAIClient(None))
    planner = OpenAILLMClient(settings, "planner", client=FakeOpenAIClient(None))

    assert product_manager.reasoning_effort == "low"
    # Untouched: the planner still takes its own checked-in default.
    assert planner.reasoning_effort == "high"


@pytest.mark.asyncio
async def test_an_unsupported_level_is_still_refused_however_it_was_chosen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one capability check is central, so a new way to choose a level cannot bypass it."""
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_PLANNER_REASONING_EFFORT", "max")
    monkeypatch.setenv("MODEL_REASONING_UNSUPPORTED", '{"reasoning-model": ["max"]}')
    settings = load_settings()

    planner = OpenAILLMClient(settings, "planner", client=FakeOpenAIClient(None))

    assert planner.reasoning_effort != "max"


@pytest.mark.asyncio
async def test_reconnaissance_is_not_moved_by_a_decision_about_the_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recon shared the planner's policy, so easing the planner off `max` eased it down too.

    Two different jobs moved by one decision, which is the thing per-agent policy exists to
    prevent. Recon is what everything after it is grounded in -- a plan built on a convention
    the repository does not have becomes a workstream no attempt can finish -- so it keeps the
    reasoning role's own level while the planner sits below it.
    """
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "max")
    monkeypatch.setenv("OPENAI_PLANNER_REASONING_EFFORT", "")
    monkeypatch.setenv("OPENAI_RECON_REASONING_EFFORT", "")
    settings = load_settings()

    recon = OpenAILLMClient(settings, "recon", client=FakeOpenAIClient(None))
    planner = OpenAILLMClient(settings, "planner", client=FakeOpenAIClient(None))

    assert recon.reasoning_effort == "max"
    assert planner.reasoning_effort == "high"
    # Same role, so the same model and the same deadline. Only the effort differs.
    assert recon.model == planner.model
    assert settings.timeout_seconds_for_agent("recon") == settings.timeout_seconds_for_agent(
        "planner"
    )


@pytest.mark.asyncio
async def test_reconnaissance_effort_is_overridable_from_the_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It gets the same operator knob as the other two, for the same reason."""
    configure_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "max")
    monkeypatch.setenv("OPENAI_RECON_REASONING_EFFORT", "medium")
    settings = load_settings()

    assert (
        OpenAILLMClient(settings, "recon", client=FakeOpenAIClient(None)).reasoning_effort
        == "medium"
    )


# The stream that never speaks, and what it costs. AB-Feature-215: two calls a second apart
# were accepted and sent no byte at all, each burning the engineer's full 1800-second
# deadline before failing. The replacement work took 34 seconds. These tests are about the
# budget that notices the silence and the re-issue that answers it -- not about how long a
# model may think, which is a different deadline and is deliberately left alone below.


@dataclass(frozen=True, slots=True)
class FakeCompletedResponse:
    """The response object a Responses stream carries on its completion envelope."""

    id: str = "response-streamed"
    model: str = "coding-model"
    output_text: str = "the implementation"
    usage: Any = None


class SilentResponseStream:
    """A stream that is opened and then says nothing, for as long as anyone will wait."""

    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        """Never answer. The caller's own budget is the only thing that ends this."""
        await asyncio.sleep(3_600)
        raise AssertionError("the silent stream was waited out, which cannot happen")

    async def close(self) -> None:
        """Record that the adapter released this stream before abandoning it."""
        self.closed = True


class StallingResponseStream:
    """A stream that speaks immediately and then thinks for a while, like a real one."""

    def __init__(self, response: Any, *, gap_seconds: float) -> None:
        self._events = [
            FakeStreamEvent("response.created"),
            FakeStreamEvent("response.completed", response),
        ]
        self._gap_seconds = gap_seconds
        self._spoken = 0

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> FakeStreamEvent:
        """First event at once; every later one after a pause the first-event budget forbids."""
        if self._spoken >= len(self._events):
            raise StopAsyncIteration
        if self._spoken:
            await asyncio.sleep(self._gap_seconds)
        event = self._events[self._spoken]
        self._spoken += 1
        return event


class SilentThenAnsweringAPI:
    """A Responses API that hands back silent streams, then answers on a later issue."""

    def __init__(self, response: Any, *, silent_issues: int) -> None:
        self.response = response
        self._silent_remaining = silent_issues
        self.calls: list[dict[str, Any]] = []
        self.silent_streams: list[SilentResponseStream] = []

    async def create(self, **kwargs: Any) -> Any:
        """Return a silent stream while the budget of them lasts, then a real one."""
        self.calls.append(kwargs)
        if self._silent_remaining > 0:
            self._silent_remaining -= 1
            stream = SilentResponseStream()
            self.silent_streams.append(stream)
            return stream
        return FakeResponseStream(self.response)


class FakeOpenAIClientWithAPI:
    """A root client double that carries an already-built Responses API double."""

    def __init__(self, responses: Any) -> None:
        self.responses = responses


@pytest.fixture
def _fast_first_event(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Shrink the budget and the backoff so the real waits do not run in a test.

    The budget arrives through settings now, so it is overridden where an operator would
    override it rather than by patching a constant -- which also keeps this test honest about
    the field being configurable at all.
    """
    monkeypatch.setattr(llm_adapter, "_STREAM_REISSUE_BACKOFF_SECONDS", (0.0, 0.0))
    return load_settings(first_event_timeout_seconds=1)


@pytest.mark.asyncio
async def test_silent_stream_is_issued_again_and_the_call_succeeds(
    _fast_first_event: Settings,
) -> None:
    """A stream that never speaks costs one more request, not the attempt.

    The property, stated as the fix intends it: the caller sees an ordinary successful
    response and never learns that the first issue was dead. Anything else -- an error
    reaching the workstream, a partial answer -- is the failure this exists to prevent.
    """
    api = SilentThenAnsweringAPI(FakeCompletedResponse(), silent_issues=1)
    client = OpenAILLMClient(_fast_first_event, "engineer", client=FakeOpenAIClientWithAPI(api))

    response = await client.respond(instructions="Follow the plan.", input_text="Implement.")

    assert response.output_text == "the implementation"
    assert len(api.calls) == 2
    # Counted, and carried out on the response. This is the only trace a successful re-issue
    # leaves anywhere: it writes no journal row of its own and the caller sees an ordinary
    # answer, so without this the signal that the budget is close to binding does not exist.
    assert response.stream_reissues == 1
    # And it reaches the journal seam, so a planning row can be queried for it per model
    # rather than grepped for in container logs.
    recorded = last_llm_call_payload() or {}
    assert recorded["execution"]["stream_reissues"] == 1
    # Re-issued verbatim: a request that changed between issues would make the second call a
    # different question, and the whole basis for asking it again is that nothing happened.
    assert api.calls[0] == api.calls[1]
    # And the dead stream was released rather than left open behind the new one.
    assert api.silent_streams[0].closed is True


@pytest.mark.asyncio
async def test_a_provider_that_stays_silent_fails_with_the_count_it_took(
    _fast_first_event: Settings,
) -> None:
    """Three silent issues end the call, classified as transport and saying how hard it tried.

    The count is in the message on purpose. A fault record that reads "the provider did not
    answer" without it sends its reader looking for a timeout that is not the one that fired
    -- which is exactly the half-hour of forensics AB-Feature-215 cost.
    """
    api = SilentThenAnsweringAPI(FakeCompletedResponse(), silent_issues=99)
    client = OpenAILLMClient(_fast_first_event, "engineer", client=FakeOpenAIClientWithAPI(api))

    with pytest.raises(LLMAdapterError) as raised:
        await client.respond(instructions="Follow the plan.", input_text="Implement.")

    assert raised.value.failure_classification == "stream_silent"
    assert "on 3 successive issues" in str(raised.value)
    assert len(api.calls) == 3
    # Classified as weather, so the platform's fault allowance -- not an attempt -- pays for it.
    assert is_transport_fault(raised.value) is True


@pytest.mark.asyncio
async def test_the_first_event_budget_does_not_govern_a_model_that_is_thinking(
    _fast_first_event: Settings,
) -> None:
    """Once a stream has spoken, it may take as long as its own deadline allows.

    The regression this guards is the tempting one: capping the whole call at the short
    budget would fail every genuine long-reasoning call -- an ultra-tier coding call was
    measured at 3652 seconds, and it was correct. The gap here is three times the
    first-event budget and must pass unremarked.
    """
    api = FakeResponsesAPI(FakeCompletedResponse())
    stalling = StallingResponseStream(FakeCompletedResponse(), gap_seconds=3.0)

    async def create(**kwargs: Any) -> Any:
        api.calls.append(kwargs)
        return stalling

    api.create = create  # type: ignore[method-assign]
    client = OpenAILLMClient(_fast_first_event, "engineer", client=FakeOpenAIClientWithAPI(api))

    response = await client.respond(instructions="Follow the plan.", input_text="Implement.")

    assert response.output_text == "the implementation"
    # One issue, not two: nothing about the pause after the first event is a fault.
    assert len(api.calls) == 1
    # Zero, not absent. A stream that spoke first time is a measurement of the budget, and
    # the reassuring answer has to be distinguishable from nobody having taken one.
    assert response.stream_reissues == 0


def test_is_transport_fault_separates_weather_from_an_answer() -> None:
    """The single predicate three callers share, stated as its truth table.

    A provider that answered -- a refusal, a truncation, a 4xx -- returns the same answer to
    every retry, and spending a fault allowance on it converts a diagnosable stop into "the
    provider did not answer". The transport failing is the opposite case in every respect.
    """
    assert is_transport_fault(LLMAdapterError("t", failure_classification="ReadTimeout"))
    assert is_transport_fault(LLMAdapterError("s", failure_classification="stream_silent"))
    assert is_transport_fault(LLMAdapterError("c", failure_classification="APIConnectionError"))
    assert not is_transport_fault(LLMAdapterError("r", failure_classification="model_refusal"))
    assert not is_transport_fault(LLMAdapterError("t", failure_classification="response_truncated"))
    assert not is_transport_fault(LLMAdapterError("b", failure_classification="BadRequestError"))
    # Bytes arrived and this adapter judged them. Every one of these is worth asking again --
    # and none of them is the transport, which is the distinction the first draft of the
    # predicate collapsed: it called a malformed coding response a transport fault, and two
    # live-executor tests caught it by demanding the gate's verdict survive.
    assert not is_transport_fault(
        LLMAdapterError("m", failure_classification="malformed_coding_response")
    )
    assert not is_transport_fault(
        LLMAdapterError("e", failure_classification="coding_edit_mismatch")
    )
    assert not is_transport_fault(LLMAdapterError("n", failure_classification="empty_response"))
    assert not is_transport_fault(LLMAdapterError("i", failure_classification="incomplete_stream"))
    # Nothing was sent at all here -- this adapter refused to show an image to a model the
    # deployment never declared able to read one -- and it is still not the transport. A
    # retry decides the same thing again, which is the whole point of the classification.
    assert not is_transport_fault(
        LLMAdapterError("v", failure_classification="model_not_vision_capable")
    )
    # This adapter's own local decisions carry no classification, and a decision is not
    # weather: repeating one only decides the same thing again.
    assert not is_transport_fault(LLMAdapterError("no classification"))
    assert not is_transport_fault(RuntimeError("not an adapter fault at all"))


def test_every_classification_this_adapter_raises_is_accounted_for() -> None:
    """No classification may reach the predicate without a decision having been made about it.

    The drift guard for the bug above. `is_transport_fault` treats an unrecognised value as an
    SDK exception name, which is the right default for names this module never writes -- and
    exactly the wrong one for a new classification added here and nowhere else. A new raise
    site must now either be judged from delivered bytes or be declared a transport verdict.
    """
    source = Path(llm_adapter.__file__).read_text(encoding="utf-8")
    raised = set(re.findall(r'failure_classification="([a-z_]+)"', source))
    declared = llm_adapter._ADAPTER_JUDGED_CLASSIFICATIONS | {"stream_silent"}

    assert raised, "the scan found no classifications at all, so it is no longer a guard"
    assert raised <= declared, (
        f"these classifications are raised here but classified nowhere: {sorted(raised - declared)}"
    )


def test_the_first_event_budget_declares_the_value_it_ships_with() -> None:
    """The default is pinned, because it was chosen by judgement and not by measurement.

    Three minutes rather than the sixty seconds first drafted: the journal can bound
    time-to-first-event only where a fast complete call exists, and `gpt-6-astra`'s fastest
    observed call is 114 seconds, which bounds nothing below a one-minute budget. Three
    minutes clears every such bound in the record while still noticing a dead stream ten
    times sooner than the 1800-second deadline that used to be the only thing that could.

    Changing it is fine and is exactly what the setting is for -- changing it *silently* is
    not, which is what this assertion prevents.
    """
    assert load_settings().first_event_timeout_seconds == 180


@pytest.mark.asyncio
async def test_a_plain_post_reports_no_re_issue_measurement_at_all() -> None:
    """A transport with no stream answers `None`, never `0`.

    The reviewer runs on a 180-second deadline, below the streaming threshold, so its calls
    are a single POST. There is no first event to wait for and no budget in play, so there is
    nothing to have measured -- and claiming `0` would put a reassuring number on a question
    that was never asked. The distinction is the whole reason the field is nullable.
    """
    payload = FakeOpenAIResponse(
        id="response-review", model="review-model", output_text="reviewed", usage=FakeUsage(11, 3)
    )
    client = OpenAILLMClient(load_settings(), "reviewer", client=FakeOpenAIClient(payload))

    response = await client.respond(instructions="Review it.", input_text="The diff.")

    assert response.stream_reissues is None
    assert "stream_reissues" not in (last_llm_call_payload() or {})["execution"]
