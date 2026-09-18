"""Choosing a platform, and everything that has to stay true once one is chosen.

The provider used to be a deployment constant. It is now a property of a feature, so the two
things this file protects are: that a feature keeps the platform it was submitted on across
every restart and recovered attempt, and that the credential it is refused for is the one it
would actually have used.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from api.feature_schemas import StartFeatureRequest
from configs.model_roles import (
    AgentPlatform,
    ModelConfigurationError,
    ModelRole,
    PlatformModelSettings,
    build_model_config_service,
)
from configs.settings import load_settings
from main import create_app
from tests.test_feature_api import feature_payload

KEY = "queue-key"
AUTH = {"Authorization": f"Bearer {KEY}"}

_ANTHROPIC_MODELS = {
    ModelRole.REASONING: "claude-opus-5",
    ModelRole.CODING: "claude-opus-5",
    ModelRole.REVIEW: "claude-opus-5",
    ModelRole.SCOPED_FIX: "claude-sonnet-5",
}
_EFFORT: dict[ModelRole, str | None] = {
    ModelRole.REASONING: "max",
    ModelRole.CODING: "xhigh",
    ModelRole.REVIEW: "high",
    ModelRole.SCOPED_FIX: "high",
}
_MAX_TOKENS: dict[ModelRole, int | None] = {
    ModelRole.REASONING: 32_000,
    ModelRole.CODING: 64_000,
    ModelRole.REVIEW: 16_000,
    ModelRole.SCOPED_FIX: 16_000,
}


class _SecretStore:
    """A credential store where exactly the named providers are configured."""

    def __init__(self, *provider_names: str) -> None:
        """Record which providers this identity has set up."""
        self.configured = set(provider_names)

    async def resolve(self, *, owner_id: str, provider: str) -> str | None:
        """Return a usable secret only for a provider this identity has configured."""
        del owner_id
        return f"sk-{provider}" if provider in self.configured else None

    async def describe(self, *, owner_id: str, provider: str) -> Any:
        """Unused by the start gate; present so the store satisfies the same shape."""
        raise NotImplementedError


# ------------------------------------------------------------------------- startup validation


def test_one_completely_configured_platform_is_enough_to_start() -> None:
    """A deployment that only uses Claude must not need `OPENAI_REVIEW_MODEL`."""
    service = build_model_config_service(
        platforms={
            AgentPlatform.ANTHROPIC: PlatformModelSettings(
                models=_ANTHROPIC_MODELS, reasoning=_EFFORT, max_tokens=_MAX_TOKENS
            ),
            # Absent entirely, which is a preference rather than a mistake.
            AgentPlatform.OPENAI: PlatformModelSettings(
                models=dict.fromkeys(ModelRole, ""),
                reasoning=dict.fromkeys(ModelRole),
                max_tokens=dict.fromkeys(ModelRole),
            ),
        }
    )

    assert service.configured_platforms() == (AgentPlatform.ANTHROPIC,)
    assert service.is_configured(AgentPlatform.ANTHROPIC)
    assert not service.is_configured(AgentPlatform.OPENAI)
    assert (
        service.get_model_config(ModelRole.SCOPED_FIX, platform=AgentPlatform.ANTHROPIC).model
        == "claude-sonnet-5"
    )


def test_three_of_four_roles_is_refused_at_startup_by_the_missing_variable_name() -> None:
    """Half a provider is a configuration mistake, not a preference.

    Refused here rather than discovered when a feature reaches its scoped-fix path, and the
    message names the variable and never a value.
    """
    with pytest.raises(ModelConfigurationError) as refused:
        build_model_config_service(
            platforms={
                AgentPlatform.ANTHROPIC: PlatformModelSettings(
                    models={**_ANTHROPIC_MODELS, ModelRole.SCOPED_FIX: ""},
                    reasoning=_EFFORT,
                    max_tokens=_MAX_TOKENS,
                )
            }
        )

    assert "ANTHROPIC_SCOPED_FIX_MODEL" in str(refused.value)
    assert "claude" not in str(refused.value)


def test_asking_for_a_platform_this_deployment_does_not_run_names_what_is_missing() -> None:
    """The refusal points at configuration rather than at whatever failed downstream."""
    service = build_model_config_service(
        platforms={
            AgentPlatform.ANTHROPIC: PlatformModelSettings(
                models=_ANTHROPIC_MODELS, reasoning=_EFFORT, max_tokens=_MAX_TOKENS
            )
        }
    )

    with pytest.raises(ModelConfigurationError, match="OPENAI_REVIEW_MODEL"):
        service.get_model_config(ModelRole.REVIEW, platform=AgentPlatform.OPENAI)


def test_settings_resolve_both_platforms_when_both_are_configured() -> None:
    """Both may be configured at once; a feature chooses between them at submission."""
    settings = load_settings(
        openai_reasoning_model="gpt-5.6-sol",
        openai_coding_model="gpt-5.6-sol",
        openai_review_model="gpt-5.6-terra",
        openai_scoped_fix_model="gpt-5.3-codex",
        anthropic_reasoning_model="claude-opus-5",
        anthropic_coding_model="claude-opus-5",
        anthropic_review_model="claude-opus-5",
        anthropic_scoped_fix_model="claude-sonnet-5",
        anthropic_coding_reasoning_effort="xhigh",
    )

    assert set(settings.model_configs.configured_platforms()) == set(AgentPlatform)
    coding = settings.model_config_for_role(ModelRole.CODING, platform=AgentPlatform.ANTHROPIC)
    assert (coding.model, coding.reasoning) == ("claude-opus-5", "xhigh")
    # The same role on the other platform is a different model, resolved independently.
    assert (
        settings.model_config_for_role(ModelRole.CODING, platform=AgentPlatform.OPENAI).model
        == "gpt-5.6-sol"
    )
    # And the output bound only one of them needs is carried beside it.
    assert coding.max_tokens == settings.anthropic_coding_max_tokens
    assert (
        settings.model_config_for_role(ModelRole.CODING, platform=AgentPlatform.OPENAI).max_tokens
        is None
    )


# ------------------------------------------------------------------------- the credential gate


@pytest.mark.asyncio
async def test_an_anthropic_feature_with_only_an_openai_key_is_refused_naming_anthropic() -> None:
    """The message is the whole feature here: it has to name the key that is missing.

    A feature submitted on Claude and refused for a missing OpenAI key would be refused over a
    credential none of its agents would ever reach for.
    """
    app = create_app(platform_api_key=KEY, secret_store=_SecretStore("openai", "github"))
    payload = {**feature_payload(), "execution_mode": "live", "agent_platform": "anthropic"}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        refused = await client.post("/features/start", headers=AUTH, json=payload)
        listed = await client.get("/features", headers=AUTH)

    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert "Anthropic" in detail
    assert "OpenAI" not in detail
    # The explanation of *why* stored credentials are needed is kept; only the name changed.
    assert "executes it after answering you" in detail
    assert listed.json()["features"] == []


@pytest.mark.asyncio
async def test_an_openai_feature_with_only_an_anthropic_key_is_refused_symmetrically() -> None:
    """Neither platform needs the other's key, in either direction."""
    app = create_app(platform_api_key=KEY, secret_store=_SecretStore("anthropic", "github"))
    payload = {**feature_payload(), "execution_mode": "live", "agent_platform": "openai"}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        refused = await client.post("/features/start", headers=AUTH, json=payload)

    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert "OpenAI" in detail
    assert "Anthropic" not in detail


@pytest.mark.asyncio
async def test_a_feature_is_accepted_on_the_platform_whose_key_is_stored() -> None:
    """The other half of the gate: the right key is enough, and the other one is not needed."""
    app = create_app(platform_api_key=KEY, secret_store=_SecretStore("anthropic", "github"))
    payload = {**feature_payload(), "execution_mode": "live", "agent_platform": "anthropic"}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        accepted = await client.post(
            "/features/start", headers={**AUTH, "Idempotency-Key": "platform"}, json=payload
        )

    assert accepted.status_code == 201
    assert accepted.json()["agent_platform"] == "anthropic"


# --------------------------------------------------------------------------- carrying the choice


def test_a_request_that_names_no_platform_reads_as_openai() -> None:
    """Every feature written before the choice existed ran on OpenAI.

    A default is only honest here because it states what actually happened, rather than
    resolving against whatever the deployment is configured with at the moment of reading.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    assert request.agent_platform == "openai"


@pytest.mark.asyncio
async def test_the_platform_a_feature_was_submitted_on_is_published_and_never_re_resolved() -> None:
    """It survives a read, and no endpoint offers to change it."""
    app = create_app(platform_api_key=KEY)
    payload = {**feature_payload(), "agent_platform": "anthropic"}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        created = await client.post(
            "/features/start", headers={**AUTH, "Idempotency-Key": "immutable"}, json=payload
        )
        fetched = await client.get("/features/feature-login", headers=AUTH)
        listed = await client.get("/features", headers=AUTH)

    assert created.json()["agent_platform"] == "anthropic"
    assert fetched.json()["agent_platform"] == "anthropic"
    # Published on the list read model too, which is built from indexed columns rather than
    # from parent state -- so the column and the response model have to agree.
    assert listed.json()["features"][0]["agent_platform"] == "anthropic"
    # There is no route that changes it. `execution_mode` has the same rule, and neither has
    # a mutating endpoint: the platform is fixed at submission for the feature's whole life.
    paths = {getattr(route, "path", "") for route in app.routes}
    assert not any("platform" in path for path in paths)


@pytest.mark.asyncio
async def test_the_setup_endpoint_reports_which_platforms_this_deployment_can_run() -> None:
    """A dropdown offering Claude where nothing resolves it is a submission that fails later.

    An isolated application publishes no resolved model configuration, so it reports the
    platform this API has always been able to run and no more.
    """
    app = create_app(platform_api_key=KEY)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        setup = await client.get("/setup", headers=AUTH)

    platforms = {item["platform"]: item for item in setup.json()["agent_platforms"]}
    assert set(platforms) == {"openai", "anthropic"}
    assert platforms["openai"]["configured"] is True
    # Reported and disabled rather than omitted: a control that vanishes is indistinguishable
    # from a control that never existed. The entries are (platform, tier) pairings now, and
    # an isolated application offers only the tier this API has always run.
    assert platforms["anthropic"]["configured"] is False
    assert platforms["anthropic"]["label"] == "Claude — Max"
    # A pairing nothing resolves promises nothing: no models, and no efforts either. An
    # effort published for a stage with no model would be a fact about a choice that cannot
    # be made, and the form would render it beside an empty model cell.
    assert platforms["anthropic"]["models"] == {}
    assert platforms["anthropic"]["reasoning_efforts"] == {}
    assert {item["performance_tier"] for item in setup.json()["agent_platforms"]} == {"high"}


@pytest.mark.asyncio
async def test_the_setup_endpoint_publishes_the_models_a_choice_resolves_to() -> None:
    """The form says which models a platform runs, without holding a copy of configuration."""
    settings = load_settings(
        anthropic_reasoning_model="claude-opus-5",
        anthropic_coding_model="claude-opus-5",
        anthropic_review_model="claude-opus-5",
        anthropic_scoped_fix_model="claude-sonnet-5",
    )
    app = create_app(platform_api_key=KEY)
    app.state.model_configs = settings.model_configs

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        setup = await client.get("/setup", headers=AUTH)

    # Keyed by (platform, tier): the endpoint publishes one entry per pairing, and the
    # deployment `.env` may configure any number of tiers beside the unsuffixed high one --
    # a platform-keyed dict would silently assert on whichever tier happens to come last.
    platforms = {
        (item["platform"], item["performance_tier"]): item
        for item in setup.json()["agent_platforms"]
    }
    anthropic_high = platforms[("anthropic", "high")]
    assert anthropic_high["configured"] is True
    assert anthropic_high["models"]["coding"] == "claude-opus-5"
    assert anthropic_high["models"]["scoped_fix"] == "claude-sonnet-5"
    # Never a credential, a base URL or an account.
    assert not any(
        marker in key for key in anthropic_high for marker in ("api_key", "secret", "credential")
    )
    # The effort each of those roles is sent at, published beside the model: two pairings can
    # name the same model and differ only here, and a form holding the models alone would show
    # two different selections as identical work.
    assert set(anthropic_high["reasoning_efforts"]) == set(anthropic_high["models"])
