"""Adversarial coverage for repository-controlled subprocess trust boundaries."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from adapters.git_adapter import GitAdapterError, _gitpython_environment
from adapters.interruptible_git import InterruptibleGitService
from main import create_app
from services.cancellation import MockCancellationToken
from services.external_operations import ExternalOperationExecutor, ExternalOperationScope
from services.process_runner import (
    AsyncioProcessRunner,
    redact_output,
    redact_source_credentials,
    repository_subprocess_environment,
)
from storage.db import Database
from storage.external_operation_store import ExternalOperationJournal
from tools.dependency_sync import RepositoryToolSynchronizer
from tools.technology_detection import inspect_repository_technology
from tools.validation_tools import (
    MockValidationPlanBuilder,
    RepositoryValidationPlan,
    ValidationCommand,
    WorkspaceValidationTools,
)

_PLATFORM_SECRETS = {
    "OPENAI_API_KEY": "openai-platform-secret-value",
    "GITHUB_TOKEN": "github-platform-secret-value",
    "DATABASE_URL": "postgresql://platform:database-secret@db/platform",
    "REDIS_URL": "redis://:redis-secret@redis/0",
}


def test_source_redaction_preserves_dynamic_authentication_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary auth plumbing remains reviewable while actual credential values do not."""
    known_secret = "github_pat_runtime_value_that_must_not_leave"
    url_credential = "mnbvcxzlkjhgfdsapoiuytre"
    high_entropy = "V7mQ2xL9pR4sT8wY3kN6dF1z"
    monkeypatch.setenv("GITHUB_TOKEN", known_secret)
    source = (
        'token = request.headers["Authorization"]\n'
        "password = credentials.current_password()\n"
        'password = "test"\n'
        'token = "example-token"\n'
        'docs_url = "https://example.com/reference"\n'
        'placeholder_url = "https://your-token@github.com/example/repository"\n'
        f'credential_url = "https://{url_credential}@github.com/private/repository"\n'
        f'client_secret = "{high_entropy}"\n'
        f'access_token = "{high_entropy}"\n'
        f'credential = "{high_entropy}"\n'
        f'private_key = "{high_entropy}"\n'
        f'config["refresh-token"] = "{high_entropy}"\n'
        "options = {"
        f'"clientSecret": "{high_entropy}", '
        f'"accessToken": "{high_entropy}", '
        f'"apiKey": "{high_entropy}"'
        "}\n"
        'client = OpenAI(api_key="sk-proj-EXAMPLESECRET123456789")\n'
        'literal_token = "ghp_project_literal_value"\n'
        f'copied_runtime_secret = "{known_secret}"\n'
    )

    redacted = redact_source_credentials(source)

    assert 'token = request.headers["Authorization"]' in redacted
    assert "password = credentials.current_password()" in redacted
    assert 'password = "test"' in redacted
    assert 'token = "example-token"' in redacted
    assert 'docs_url = "https://example.com/reference"' in redacted
    assert "https://your-token@github.com/example/repository" in redacted
    assert url_credential not in redacted
    assert high_entropy not in redacted
    assert "sk-proj-EXAMPLESECRET123456789" not in redacted
    assert "ghp_project_literal_value" not in redacted
    assert known_secret not in redacted
    assert redacted.count("[REDACTED]") == 12


def test_process_output_aggressively_redacts_provider_token_families() -> None:
    """Repository stdout cannot carry newer OpenAI or GitHub token shapes into prompts."""
    openai_token = "sk-proj-ACTUALOUTPUTSECRET123456789"
    github_token = "gho_ACTUALOUTPUTSECRET123456789"
    aws_access_key = "AKIA1234567890ABCDEF"
    aws_secret_key = "q1W2e3R4t5Y6u7I8o9P0a1S2d3F4g5H6j7K8l9Z0"

    redacted = redact_output(
        f"reset output {openai_token} and {github_token}; "
        f"aws_access_key_id={aws_access_key}; aws_secret_access_key={aws_secret_key}"
    )

    assert openai_token not in redacted
    assert github_token not in redacted
    assert aws_access_key not in redacted
    assert aws_secret_key not in redacted
    assert redacted == (
        "reset output [REDACTED] and [REDACTED]; "
        "aws_access_key_id=[REDACTED] aws_secret_access_key=[REDACTED]"
    )


@pytest.mark.asyncio
async def test_http_failure_log_does_not_copy_arbitrary_exception_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A provider or repository exception cannot turn retained API logs into a secret sink."""
    secret = "github_pat_http_exception_secret"
    app = create_app(platform_api_key="test-key")

    @app.get("/security-test-failure/{resource_id}")
    async def fail_request(resource_id: str) -> None:
        del resource_id
        raise RuntimeError(secret)

    # The default transport re-raises an unconsumed ASGI exception. A response therefore
    # proves the application boundary did not hand the original message to a server logger.
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get(f"/security-test-failure/{secret}")

    captured = capsys.readouterr()
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert secret not in captured.out
    assert secret not in captured.err


@pytest.mark.asyncio
async def test_default_repository_process_environment_does_not_inherit_platform_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A caller that omits ``env`` gets an allowlist, never the API process environment."""
    for key, value in _PLATFORM_SECRETS.items():
        monkeypatch.setenv(key, value)
    script = (
        "import json, os; "
        f"print(json.dumps({{key: os.environ.get(key) for key in {list(_PLATFORM_SECRETS)!r}}}))"
    )

    result = await AsyncioProcessRunner().run(
        (sys.executable, "-c", script),
        tmp_path,
        10,
        MockCancellationToken(),
    )

    assert result.succeeded
    assert json.loads(result.stdout) == {key: None for key in _PLATFORM_SECRETS}


def test_repository_environment_rejects_credential_overrides() -> None:
    """A validation plan cannot opt a platform credential back into checkout code."""
    with pytest.raises(ValueError, match="unsupported keys"):
        repository_subprocess_environment({"OPENAI_API_KEY": "do-not-pass"})


def test_legacy_gitpython_environment_masks_inherited_secrets_and_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GitPython's internal os.environ merge is neutralized for its legacy remote path."""
    for key, value in _PLATFORM_SECRETS.items():
        monkeypatch.setenv(key, value)

    environment = _gitpython_environment(
        {"PLATFORM_GIT_TOKEN": "request-scoped-github-secret"}, remote=True
    )

    assert all(environment[key] == "" for key in _PLATFORM_SECRETS)
    assert environment["PLATFORM_GIT_TOKEN"] == "request-scoped-github-secret"
    assert environment["GIT_CONFIG_KEY_1"] == "core.hooksPath"
    assert environment["GIT_CONFIG_VALUE_1"] == os.devnull


@pytest.mark.asyncio
async def test_malicious_dependency_lifecycle_process_receives_no_platform_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Install-time repository code cannot read API, database, Redis, or GitHub secrets."""
    root = tmp_path / "repository"
    root.mkdir()
    (root / "package.json").write_text('{"name":"malicious"}\n', encoding="utf-8")
    (root / "package-lock.json").write_text('{"lockfileVersion":3}\n', encoding="utf-8")
    tool_bin = tmp_path / "bin"
    tool_bin.mkdir()
    fake_npm = tool_bin / "npm"
    fake_npm.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "${OPENAI_API_KEY-absent}" "${GITHUB_TOKEN-absent}" '
        '"${DATABASE_URL-absent}" "${REDIS_URL-absent}" > lifecycle-env.txt\n',
        encoding="utf-8",
    )
    fake_npm.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tool_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    for key, value in _PLATFORM_SECRETS.items():
        monkeypatch.setenv(key, value)

    applied = await RepositoryToolSynchronizer().sync(root, ["package.json"])

    assert applied == ("npm install --no-audit",)
    assert (root / "lifecycle-env.txt").read_text(encoding="utf-8").splitlines() == [
        "absent",
        "absent",
        "absent",
        "absent",
    ]


@pytest.mark.asyncio
async def test_malicious_precommit_hook_is_sanitized_and_its_output_is_not_trusted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The checked-out hook sees no request token and cannot persist its stderr as a diagnostic."""
    repository = tmp_path / "repository"
    _git("init", str(repository))
    _git("-C", str(repository), "config", "user.email", "tests@example.invalid")
    _git("-C", str(repository), "config", "user.name", "Test User")
    tracked = repository / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    _git("-C", str(repository), "add", "tracked.txt")
    _git("-C", str(repository), "commit", "-m", "initial")
    hook = repository / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "${OPENAI_API_KEY-absent}" "${GITHUB_TOKEN-absent}" '
        '"${DATABASE_URL-absent}" "${REDIS_URL-absent}" '
        '"${PLATFORM_GIT_TOKEN-absent}" > hook-env.txt\n'
        "printf '%s\\n' 'ARBITRARY_CHILD_TRANSCRIPT_DO_NOT_PERSIST' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    hook.chmod(0o700)
    tracked.write_text("after\n", encoding="utf-8")
    for key, value in _PLATFORM_SECRETS.items():
        monkeypatch.setenv(key, value)
    service = InterruptibleGitService(
        cancellation_token=MockCancellationToken(),
        environment={
            "GIT_ASKPASS": "/bin/false",
            "GIT_TERMINAL_PROMPT": "0",
            "PLATFORM_GIT_TOKEN": "request-scoped-github-secret",
        },
    )

    with pytest.raises(GitAdapterError) as captured:
        await service.commit(repository, "malicious hook", files=["tracked.txt"])

    assert (repository / "hook-env.txt").read_text(encoding="utf-8").splitlines() == [
        "absent",
        "absent",
        "absent",
        "absent",
        "absent",
    ]
    assert "ARBITRARY_CHILD_TRANSCRIPT_DO_NOT_PERSIST" not in str(captured.value)
    assert all(
        "ARBITRARY_CHILD_TRANSCRIPT_DO_NOT_PERSIST" not in item
        for item in captured.value.diagnostics
    )


@pytest.mark.asyncio
async def test_credentialed_push_disables_repository_prepush_hook(tmp_path: Path) -> None:
    """A pre-push hook cannot inspect the request token needed by the remote Git process."""
    remote = tmp_path / "remote.git"
    repository = tmp_path / "repository"
    _git("init", "--bare", str(remote))
    _git("init", str(repository))
    _git("-C", str(repository), "config", "user.email", "tests@example.invalid")
    _git("-C", str(repository), "config", "user.name", "Test User")
    tracked = repository / "tracked.txt"
    tracked.write_text("initial\n", encoding="utf-8")
    _git("-C", str(repository), "add", "tracked.txt")
    _git("-C", str(repository), "commit", "-m", "initial")
    _git("-C", str(repository), "branch", "-M", "main")
    _git("-C", str(repository), "remote", "add", "origin", str(remote))
    _git("-C", str(repository), "push", "origin", "main:main")
    _git("-C", str(repository), "switch", "-c", "feature/security")
    tracked.write_text("feature\n", encoding="utf-8")
    _git("-C", str(repository), "add", "tracked.txt")
    _git("-C", str(repository), "commit", "-m", "feature")
    hook_marker = repository / "pre-push-ran.txt"
    hook = repository / ".git" / "hooks" / "pre-push"
    hook.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"${{PLATFORM_GIT_TOKEN-absent}}\" > {hook_marker!s}\nexit 1\n",
        encoding="utf-8",
    )
    hook.chmod(0o700)
    head = _git_output("-C", str(repository), "rev-parse", "HEAD")
    service = InterruptibleGitService(
        cancellation_token=MockCancellationToken(),
        environment={
            "GIT_ASKPASS": "/bin/false",
            "GIT_TERMINAL_PROMPT": "0",
            "PLATFORM_GIT_TOKEN": "request-scoped-github-secret",
        },
    )

    result = await service.push(repository, "feature/security", expected_commit_sha=head)

    assert result.summaries == ("GIT_PUSH_SUCCEEDED",)
    assert not hook_marker.exists()
    assert _git_output("--git-dir", str(remote), "rev-parse", "refs/heads/feature/security") == head


@pytest.mark.asyncio
async def test_validation_journal_persists_safe_codes_not_arbitrary_child_output(
    tmp_path: Path,
) -> None:
    """A malicious validator transcript stays ephemeral while its stable failure code survives."""
    repository = tmp_path / "repository"
    repository.mkdir()
    script = repository / "malicious_validation.py"
    transcript = "ARBITRARY_VALIDATOR_TRANSCRIPT_DO_NOT_PERSIST"
    script.write_text(
        f"import sys\nprint({transcript!r}, file=sys.stderr)\nraise SystemExit(7)\n",
        encoding="utf-8",
    )
    plan = RepositoryValidationPlan(
        repository_id="malicious",
        technology_profile=inspect_repository_technology(repository),
        commands=[
            ValidationCommand(
                validation_type="test",
                command=[sys.executable, script.name],
                timeout_seconds=10,
            )
        ],
        source="configured",
    )
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'journal.db'}")
    await database.create_schema()
    journal = ExternalOperationJournal(database)
    operations = ExternalOperationExecutor(
        journal=journal,
        cancellation_token=MockCancellationToken(),
        scope=ExternalOperationScope(
            workflow_id="feature-security",
            feature_id="feature-security",
            child_workflow_id="feature-security:malicious",
            repository_id="malicious",
        ),
    )
    tools = WorkspaceValidationTools(
        repository,
        repository_id="malicious",
        plan_builder=MockValidationPlanBuilder(plan),
        operation_executor=operations,
    )
    try:
        (result,) = await tools.run_validations_async(timeout_seconds=10)
        recorded = await journal.list_operations_for_workflow("feature-security")
    finally:
        await database.dispose()

    assert transcript in result.stderr
    # The summary is review evidence, and a failing command's own report is the only thing
    # that tells the coding model what to change: -062 spent four review cycles on "exited
    # with code 1" while the output naming the defect was withheld. It is bounded and
    # redacted rather than suppressed.
    assert transcript in result.stderr_summary
    assert result.result_code == "TEST_VALIDATION_FAILED_EXIT_7"
    # The durable journal is the line that holds. It is replayed and audited, so it carries
    # the stable code and never the child's transcript.
    assert len(recorded) == 1
    payload = recorded[0].result_payload or {}
    assert transcript not in json.dumps(payload, sort_keys=True)
    assert payload["result_code"] == "TEST_VALIDATION_FAILED_EXIT_7"


def _git(*arguments: str) -> None:
    subprocess.run(
        ("git", *arguments),
        check=True,
        capture_output=True,
        text=True,
        env=repository_subprocess_environment(),
    )


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        check=True,
        capture_output=True,
        text=True,
        env=repository_subprocess_environment(),
    ).stdout.strip()
