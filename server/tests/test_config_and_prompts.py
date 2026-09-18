"""Tests for environment-backed settings and strict versioned prompt rendering."""

import json
from pathlib import Path
from typing import Any

import pytest
from jinja2 import UndefinedError

from adapters.llm_adapter import _apply_reasoning_effort
from configs.model_roles import AgentPlatform, ModelRole
from configs.settings import AGENT_CONFIG_FILE, load_settings
from prompts.prompt_loader import PromptLoader, PromptTemplateError


def configure_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the required runtime secrets and model configuration for settings tests."""
    monkeypatch.setenv("OPENAI_REASONING_MODEL", "reasoning-model-from-environment")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "max")
    monkeypatch.setenv("OPENAI_CODING_MODEL", "coding-model-from-environment")
    monkeypatch.setenv("OPENAI_CODING_REASONING_EFFORT", "max")
    monkeypatch.setenv("OPENAI_REVIEW_MODEL", "review-model-from-environment")
    monkeypatch.setenv("OPENAI_REVIEW_REASONING_EFFORT", "high")
    monkeypatch.setenv("OPENAI_SCOPED_FIX_MODEL", "scoped-fix-model-from-environment")
    monkeypatch.setenv("OPENAI_SCOPED_FIX_REASONING_EFFORT", "high")
    monkeypatch.setenv("PLATFORM_API_KEY", "test-platform-api-key")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///test-settings.db")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


def test_settings_load_environment_models_and_yaml_agent_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model IDs are environment-only while operational agent policy comes from YAML."""
    configure_environment(monkeypatch)

    settings = load_settings()

    assert settings.openai_reasoning_model == "reasoning-model-from-environment"
    assert settings.openai_coding_model == "coding-model-from-environment"
    assert settings.platform_api_key is not None
    assert settings.platform_api_key.get_secret_value() == "test-platform-api-key"
    assert settings.log_level == "DEBUG"
    assert settings.max_clarification_rounds == 10
    assert settings.max_engineer_review_cycles == 5
    # An agent names a role, never a provider's environment variable. Which provider that
    # role is read from is the submitting feature's choice.
    assert settings.agents["engineer"].model_role is ModelRole.CODING
    openai = AgentPlatform.OPENAI
    assert settings.model_for_agent("engineer", platform=openai) == (
        "coding-model-from-environment"
    )
    assert settings.model_for_agent("product_manager", platform=openai) == (
        "reasoning-model-from-environment"
    )
    assert settings.model_for_agent("reviewer", platform=openai) == (
        "review-model-from-environment"
    )


def test_load_settings_accepts_an_explicit_policy_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests and deployments can select a version-controlled configuration directory explicitly."""
    configure_environment(monkeypatch)
    (tmp_path / "config.yaml").write_text(
        "max_clarification_rounds: 4\n"
        "max_engineer_review_cycles: 2\n"
        "default_agent_timeout_seconds: 45\n"
        "default_agent_max_retries: 1\n"
        "confidence_threshold: 0.9\n",
        encoding="utf-8",
    )
    (tmp_path / "agent.yaml").write_text(
        AGENT_CONFIG_FILE.read_text(encoding="utf-8"), encoding="utf-8"
    )

    settings = load_settings(tmp_path)

    assert settings.max_clarification_rounds == 4
    assert settings.max_engineer_review_cycles == 2
    assert settings.default_agent_timeout_seconds == 45
    assert settings.confidence_threshold == 0.9


def test_deployment_environment_outranks_version_controlled_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A deployment's own environment overrides the defaults named in the policy files."""
    configure_environment(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "deployment-workspaces"))
    monkeypatch.setenv("ALLOWED_HOSTS", '["api.internal.example"]')

    settings = load_settings()

    # config.yaml names `development` and `/workspaces`; the container must win, otherwise
    # the production-only identity and allowed-hosts gates silently never run.
    assert settings.environment == "production"
    assert settings.workspace_root == tmp_path / "deployment-workspaces"
    assert settings.allowed_hosts == ["api.internal.example"]
    # Values the deployment does not set still come from the policy files.
    assert settings.max_clarification_rounds == 10


def test_reasoning_timeout_is_environment_configurable_without_changing_other_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Maximum-effort planning gets deployment headroom without widening every model call."""
    configure_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_REASONING_TIMEOUT_SECONDS", "901")

    settings = load_settings()

    assert settings.timeout_seconds_for_agent("product_manager") == 901
    assert settings.timeout_seconds_for_agent("planner") == 901
    assert (
        settings.timeout_seconds_for_agent("engineer")
        == settings.agents["engineer"].timeout_seconds
    )


def test_explicit_overrides_outrank_the_deployment_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A caller-supplied override remains authoritative over an environment variable."""
    configure_environment(monkeypatch)
    monkeypatch.setenv("WORKSPACE_ROOT", "/workspaces-from-environment")

    settings = load_settings(workspace_root=tmp_path / "explicit")

    assert settings.workspace_root == tmp_path / "explicit"


@pytest.mark.parametrize(
    ("template_name", "context", "expected_text"),
    [
        (
            "product_manager/v1.jinja2",
            {"workflow_id": "workflow-1", "prd": "Build an artifact platform."},
            "Build an artifact platform.",
        ),
        (
            "planner/v1.jinja2",
            {
                "workflow_id": "workflow-1",
                "technical_prd": "Validated requirements.",
                "available_artifacts": "001_prd.json",
            },
            "Validated requirements.",
        ),
        (
            "engineer/v1.jinja2",
            {
                "workflow_id": "workflow-1",
                "workspace_descriptor": "workspace-1",
                "task_plan": "Implement schemas.",
                "repository_context": "Repository files.",
                "execution_context": "No retry context.",
                "lint_capabilities": "{}",
                "previous_attempt_diff": "",
            },
            "Implement schemas.",
        ),
        (
            "reviewer/v1.jinja2",
            {
                "workflow_id": "workflow-1",
                "technical_prd": "Validated requirements.",
                "review_scope": "Assigned repository requirements.",
                "code_completion": "All checks passed.",
            },
            "All checks passed.",
        ),
    ],
)
def test_prompt_templates_render_versioned_context(
    template_name: str, context: dict[str, str], expected_text: str
) -> None:
    """Each initial agent template renders supplied context without hardcoded task data."""
    loader = PromptLoader()

    rendered_prompt = loader.render(template_name, **context)

    assert "workflow-1" in rendered_prompt
    assert expected_text in rendered_prompt
    assert template_name in loader.list_templates()


def test_the_review_is_told_which_commands_the_platform_will_run() -> None:
    """A demand to execute something needs the list of what can be executed to be bounded by.

    The review has always received validation *results* and never the *plan*, so it could see
    that a command produced no result and not that no such command exists. AB-Feature-225 spent
    its whole lineage demanding a result from a command the plan never contained -- a demand
    the engineer had no lever to meet, since it cannot add commands to the plan.
    """
    rendered = PromptLoader().render(
        "reviewer/v1.jinja2",
        workflow_id="workflow-1",
        technical_prd="Validated requirements.",
        review_scope="Assigned repository requirements.",
        code_completion="All checks passed.",
        planned_validation_commands=[
            {"command": "npm run lint", "validation_type": "lint", "required": True},
            {"command": "npm run test", "validation_type": "test", "required": True},
        ],
    )

    assert "npm run lint` (lint, required)" in rendered
    assert "npm run test` (test, required)" in rendered
    assert "is admissible only when that command" in rendered


def test_a_review_with_no_resolved_plan_says_so_rather_than_rendering_nothing() -> None:
    """Silence would read as "no restriction"; the honest rendering is that none resolved."""
    rendered = PromptLoader().render(
        "reviewer/v1.jinja2",
        workflow_id="workflow-1",
        technical_prd="Validated requirements.",
        review_scope="Assigned repository requirements.",
        code_completion="All checks passed.",
    )

    assert "none were resolved for this repository" in rendered


def test_the_planner_may_only_require_commands_the_platform_can_run() -> None:
    """A requirement naming a command the plan lacks is one no attempt can ever satisfy.

    The planner learns what a checkout holds from reconnaissance prose, and "a Playwright suite
    is present at the repository root" is a true sentence about a repository whose Playwright
    suite this platform does not run. AB-Feature-225 turned that sentence into a binding test
    requirement, the review enforced it, and no attempt could produce the result because
    neither the engineer nor the reviewer can add a command to the validation plan.
    """
    rendered = PromptLoader().render(
        "planner/feature_v1.jinja2",
        feature_id="feature-1",
        technical_prd="{}",
        repositories=[],
        has_reconnaissance=True,
        reconnaissance=json.dumps(
            [{"repository_id": "admin", "runnable_validation_commands": ["npm run test"]}]
        ),
        design_snapshot="",
    )

    assert "runnable_validation_commands" in rendered
    assert "may name only commands on that list" in rendered
    assert "npm run test" in rendered


def test_a_reconnaissance_that_established_no_commands_reads_as_unknown() -> None:
    """Empty must mean "not established", never "nothing runs" -- every artifact written
    before the field existed reads back empty, and forbidding every command over that would
    strip the test requirements out of plans that were fine."""
    rendered = PromptLoader().render(
        "planner/feature_v1.jinja2",
        feature_id="feature-1",
        technical_prd="{}",
        repositories=[],
        has_reconnaissance=True,
        reconnaissance=json.dumps([{"repository_id": "admin"}]),
        design_snapshot="",
    )

    assert "was not established for that repository, not" in rendered


def test_prompt_loader_rejects_path_traversal_and_missing_context() -> None:
    """Template loading cannot leave the prompt root or silently omit required values."""
    loader = PromptLoader()

    with pytest.raises(PromptTemplateError, match="stay within"):
        loader.render("../outside.jinja2")
    with pytest.raises(UndefinedError):
        loader.render("product_manager/v1.jinja2", workflow_id="workflow-1")


def test_no_reasoning_effort_is_requested_unless_one_is_configured() -> None:
    """A deployment that configures nothing keeps the provider's own default."""
    request: dict[str, Any] = {"model": "gpt-5.6-terra", "input": "x"}

    _apply_reasoning_effort(request, None)
    _apply_reasoning_effort(request, "")

    assert "reasoning" not in request


def test_a_configured_reasoning_effort_reaches_the_request() -> None:
    """The level is passed through verbatim for the provider to accept or reject.

    Deliberately not validated against a per-model table here: the accepted set differs
    between models -- `gpt-5.3-codex` takes `xhigh` but rejects `max`, while `gpt-5.6-sol`
    takes both -- and a table of that in this repository would be wrong the day it is
    written. The provider's rejection is the authoritative answer.
    """
    request: dict[str, Any] = {"model": "gpt-5.6-sol", "input": "x"}

    _apply_reasoning_effort(request, "max")

    assert request["reasoning"] == {"effort": "max"}


def test_the_environment_overrides_the_policy_file_per_model_role() -> None:
    """An operator compares effort levels without editing checked-in configuration."""
    settings = load_settings(
        openai_coding_reasoning_effort="xhigh",
        openai_review_reasoning_effort="",
        openai_review_reasoning="",
        openai_reasoning_effort="",
    )

    openai = AgentPlatform.OPENAI
    assert settings.reasoning_effort_for_agent("engineer", platform=openai) == "xhigh"
    # The independent reviewer role was not overridden.
    assert settings.reasoning_effort_for_agent("reviewer", platform=openai) is None
