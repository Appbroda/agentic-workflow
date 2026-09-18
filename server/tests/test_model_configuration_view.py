"""The resolved model-roles table, served instead of left in a startup log.

Which model each role runs on, how hard it is asked to think and what bounds its output was
knowable only by reading the API's first log line -- a fact about a running deployment that
nobody operating it could see. This file defends the endpoint that publishes it and the two
properties that make publishing it safe and honest: it carries no credential, and it reports
what the service every agent resolves through actually answers rather than a second reading of
the environment.

Read-only is also a property, not a documentation note: there is no write route on this path,
and the payload says so rather than leaving a client to infer it from a 405.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from configs.model_roles import (
    AgentPlatform,
    ModelRole,
    PerformanceTier,
    PlatformModelSettings,
    build_model_config_service,
)
from main import create_app
from services.secrets import InMemorySecretStore

KEY = "model-configuration-key"
AUTH = {"Authorization": f"Bearer {KEY}"}

_OPENAI_HIGH = PlatformModelSettings(
    models={
        ModelRole.REASONING: "gpt-5.6-sol",
        ModelRole.CODING: "gpt-5.6-sol",
        ModelRole.REVIEW: "gpt-5.6-terra",
        ModelRole.SCOPED_FIX: "gpt-5.3-codex",
    },
    reasoning={
        ModelRole.REASONING: "max",
        ModelRole.CODING: "max",
        ModelRole.REVIEW: "high",
        ModelRole.SCOPED_FIX: "high",
    },
    max_tokens=dict.fromkeys(ModelRole, None),
)

_ANTHROPIC_HIGH = PlatformModelSettings(
    models={
        ModelRole.REASONING: "claude-opus-5",
        ModelRole.CODING: "claude-opus-5",
        ModelRole.REVIEW: "claude-opus-5",
        ModelRole.SCOPED_FIX: "claude-sonnet-5",
    },
    reasoning={
        ModelRole.REASONING: "max",
        ModelRole.CODING: "xhigh",
        ModelRole.REVIEW: "high",
        ModelRole.SCOPED_FIX: "high",
    },
    max_tokens={
        ModelRole.REASONING: 64_000,
        ModelRole.CODING: 128_000,
        ModelRole.REVIEW: 48_000,
        ModelRole.SCOPED_FIX: 32_000,
    },
)

_ANTHROPIC_LOW = PlatformModelSettings(
    models={
        ModelRole.REASONING: "claude-sonnet-5",
        ModelRole.CODING: "claude-sonnet-5",
        ModelRole.REVIEW: "claude-sonnet-5",
        ModelRole.SCOPED_FIX: "claude-haiku-4-5",
    },
    reasoning=dict.fromkeys(ModelRole, "medium"),
    max_tokens=dict.fromkeys(ModelRole, 32_000),
)


def app_with(
    *,
    platforms: dict[AgentPlatform, PlatformModelSettings] | None = None,
    tiers: dict[tuple[AgentPlatform, PerformanceTier], PlatformModelSettings] | None = None,
    unsupported_reasoning: dict[str, list[str]] | None = None,
    secret_store: InMemorySecretStore | None = None,
) -> Any:
    """Build an application whose resolved model configuration is the one under test.

    ``model_configs`` is what the production lifespan installs after `load_settings` has
    validated it, and it is the same object agents resolve models through -- so binding it
    here tests the projection rather than a rebuilt copy of it.
    """
    app = create_app(platform_api_key=KEY, secret_store=secret_store)
    if platforms is not None:
        app.state.model_configs = build_model_config_service(
            platforms=platforms, tiers=tiers, unsupported_reasoning=unsupported_reasoning
        )
    return app


def client(app: Any) -> AsyncClient:
    """Return an HTTP client bound to the application under test."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest.mark.asyncio
async def test_every_configured_pairing_publishes_all_four_roles_in_full() -> None:
    """Platform, tier and role, each with the model, effort, bound and routing reason."""
    app = app_with(
        platforms={
            AgentPlatform.OPENAI: _OPENAI_HIGH,
            AgentPlatform.ANTHROPIC: _ANTHROPIC_HIGH,
        },
        tiers={(AgentPlatform.ANTHROPIC, PerformanceTier.LOW): _ANTHROPIC_LOW},
    )

    async with client(app) as http:
        response = await http.get("/model-configuration", headers=AUTH)

    assert response.status_code == 200
    payload = response.json()
    assert [(item["platform"], item["performance_tier"]) for item in payload["setups"]] == [
        ("openai", "high"),
        ("anthropic", "low"),
        ("anthropic", "high"),
    ]
    by_key = {(item["platform"], item["performance_tier"]): item for item in payload["setups"]}
    # Every pairing carries all four roles, in the order the platform defines them, because a
    # table missing a role reads as a role that does not run rather than one nobody published.
    for setup in payload["setups"]:
        assert [role["role"] for role in setup["roles"]] == [role.value for role in ModelRole]
    anthropic_high = {role["role"]: role for role in by_key[("anthropic", "high")]["roles"]}
    assert anthropic_high["coding"]["model"] == "claude-opus-5"
    assert anthropic_high["coding"]["reasoning_effort"] == "xhigh"
    assert anthropic_high["coding"]["max_tokens"] == 128_000
    assert anthropic_high["coding"]["routing_reason"]
    assert anthropic_high["coding"]["model_variable"] == "ANTHROPIC_CODING_MODEL"
    assert anthropic_high["coding"]["reasoning_variable"] == "ANTHROPIC_CODING_REASONING_EFFORT"
    # The cheap tier is its own table and not a discount applied to the expensive one.
    anthropic_low = {role["role"]: role for role in by_key[("anthropic", "low")]["roles"]}
    assert anthropic_low["scoped_fix"]["model"] == "claude-haiku-4-5"
    assert anthropic_low["scoped_fix"]["max_tokens"] == 32_000
    # Labels are the ones the submission form offers for the same pairing.
    assert by_key[("anthropic", "high")]["label"] == "Claude — Max"
    assert by_key[("anthropic", "low")]["label"] == "Claude — Economy"
    assert by_key[("openai", "high")]["platform_label"] == "OpenAI"
    assert by_key[("anthropic", "low")]["tier_label"] == "Economy"


@pytest.mark.asyncio
async def test_the_published_table_is_what_the_service_answers_agents() -> None:
    """Row for row, the endpoint's values are `get_model_config`'s -- not a second reading.

    The point of the role indirection is that one service decides what a role resolves to. A
    panel assembled from a parallel projection of the environment would agree on the day it
    was written and drift silently afterwards, which is the failure this asserts against.
    """
    app = app_with(
        platforms={AgentPlatform.OPENAI: _OPENAI_HIGH, AgentPlatform.ANTHROPIC: _ANTHROPIC_HIGH},
        tiers={(AgentPlatform.ANTHROPIC, PerformanceTier.LOW): _ANTHROPIC_LOW},
    )
    configs = app.state.model_configs

    async with client(app) as http:
        payload = (await http.get("/model-configuration", headers=AUTH)).json()

    published = {
        (setup["platform"], setup["performance_tier"], role["role"]): role
        for setup in payload["setups"]
        for role in setup["roles"]
    }
    assert published, "the endpoint published nothing to compare"
    for platform, tier in configs.configured_options():
        for role in ModelRole:
            resolved = configs.get_model_config(role, platform=platform, tier=tier)
            row = published[(platform.value, tier.value, role.value)]
            assert row["model"] == resolved.model
            assert row["reasoning_effort"] == resolved.reasoning
            assert row["max_tokens"] == resolved.max_tokens
            assert row["routing_reason"] == resolved.routing_reason


@pytest.mark.asyncio
async def test_the_deployment_rows_stay_read_only_and_the_table_has_no_write_route() -> None:
    """System defaults are deployment configuration, and every deployment row states that.

    The top-level `editable` now means "you may author a setup here" -- true for a caller
    holding MODEL_SETUP_MANAGE on a deployment that persists setups -- while the deployment
    rows themselves keep saying they are set in the environment (`editable: false`,
    `origin: "deployment"`), which is the property PR #33's comment reserved. Writes still
    happen on `/model-setups`, never here.
    """
    app = app_with(platforms={AgentPlatform.OPENAI: _OPENAI_HIGH})

    async with client(app) as http:
        payload = (await http.get("/model-configuration", headers=AUTH)).json()
        refused = [
            (await http.request(method, "/model-configuration", headers=AUTH, json={})).status_code
            for method in ("POST", "PUT", "PATCH", "DELETE")
        ]

    # The shared platform key is an admin, and an isolated application persists setups, so
    # authoring is offered -- at the top level only.
    assert payload["editable"] is True
    for setup in payload["setups"]:
        assert setup["origin"] == "deployment"
        assert setup["editable"] is False
        assert setup["setup_id"] is None
    assert refused == [405, 405, 405, 405]


@pytest.mark.asyncio
async def test_the_table_carries_no_credential() -> None:
    """It names models, efforts, bounds and variable names -- never a value that authenticates.

    Checked against a stored credential rather than against the model shape alone: the
    property being defended is that this response cannot leak the deployment's keys, and only
    a real one sitting in the store proves the response never reaches for it.
    """
    store = InMemorySecretStore()
    app = app_with(
        platforms={AgentPlatform.ANTHROPIC: _ANTHROPIC_HIGH},
        secret_store=store,
    )
    await store.put(owner_id="platform-admin", provider="anthropic", secret="sk-do-not-publish")
    await store.put(owner_id="platform-admin", provider="github", secret="ghp-do-not-publish")

    async with client(app) as http:
        response = await http.get("/model-configuration", headers=AUTH)

    body = response.text
    assert "do-not-publish" not in body
    assert "sk-" not in body
    assert "ghp-" not in body
    # No field a credential could occupy, either: the only names published are the variables
    # holding a model identifier, an effort level and an output bound.
    for setup in response.json()["setups"]:
        for role in setup["roles"]:
            for name in (role["model_variable"], role["reasoning_variable"] or ""):
                assert name.endswith(("_MODEL", "_REASONING_EFFORT", "_EFFORT")), name


@pytest.mark.asyncio
async def test_a_pairing_nobody_configured_is_absent_rather_than_empty() -> None:
    """A deployment offering one platform at one tier publishes exactly that one table.

    `/setup` is where a client learns which pairings exist and which are selectable, and it
    keeps an unconfigured platform visible and disabled on purpose. This table is the other
    question -- what runs -- and a row of blanks for a pairing that resolved no model would be
    an answer to neither.
    """
    app = app_with(platforms={AgentPlatform.OPENAI: _OPENAI_HIGH})

    async with client(app) as http:
        payload = (await http.get("/model-configuration", headers=AUTH)).json()

    assert [(item["platform"], item["performance_tier"]) for item in payload["setups"]] == [
        ("openai", "high")
    ]


@pytest.mark.asyncio
async def test_a_role_still_running_on_a_pre_roles_variable_says_so() -> None:
    """A deployment without independent review configuration can see that it has none.

    The startup log reports this as `roles_on_legacy_fallback`. Per row it is the more useful
    shape: the row names the variable actually being read, so the fix is visible from the same
    line as the problem.
    """
    app = app_with(
        platforms={
            AgentPlatform.OPENAI: PlatformModelSettings(
                models={
                    ModelRole.REASONING: "gpt-5.6-sol",
                    ModelRole.CODING: "gpt-5.6-sol",
                },
                reasoning={ModelRole.REASONING: "max", ModelRole.CODING: "max"},
                max_tokens=dict.fromkeys(ModelRole, None),
                legacy_models={
                    ModelRole.REVIEW: "gpt-5.6-sol",
                    ModelRole.SCOPED_FIX: "gpt-5.6-sol",
                },
            )
        }
    )

    async with client(app) as http:
        payload = (await http.get("/model-configuration", headers=AUTH)).json()

    roles = {role["role"]: role for role in payload["setups"][0]["roles"]}
    assert roles["review"]["resolved_from_legacy_variable"] is True
    assert roles["review"]["model_variable"] == "OPENAI_REASONING_MODEL"
    # A role the deployment configured explicitly is not marked, or the flag would mean
    # nothing.
    assert roles["coding"]["resolved_from_legacy_variable"] is False
    assert roles["coding"]["model_variable"] == "OPENAI_CODING_MODEL"


@pytest.mark.asyncio
async def test_an_effort_the_deployment_declared_unsupported_shows_both_values() -> None:
    """What was asked for and what is sent, because a normalization is not a preference.

    Showing only the effective value would present the provider's default as the deployment's
    choice, and a person comparing this table against their `.env` would find a disagreement
    with no explanation on the page.
    """
    app = app_with(
        platforms={AgentPlatform.OPENAI: _OPENAI_HIGH},
        unsupported_reasoning={"gpt-5.3-codex": ["high"]},
    )

    async with client(app) as http:
        payload = (await http.get("/model-configuration", headers=AUTH)).json()

    roles = {role["role"]: role for role in payload["setups"][0]["roles"]}
    # `scoped_fix` is configured `high` on a model declared not to accept it.
    assert roles["scoped_fix"]["reasoning_effort"] is None
    assert roles["scoped_fix"]["requested_reasoning_effort"] == "high"
    # Every other role asked for what it got, and says nothing twice.
    assert roles["coding"]["reasoning_effort"] == "max"
    assert roles["coding"]["requested_reasoning_effort"] is None


@pytest.mark.asyncio
async def test_a_deployment_with_no_resolved_configuration_publishes_no_table() -> None:
    """An application that resolved no models says nothing rather than inventing rows.

    `/setup` answers the availability question for an isolated application by naming the
    pairing this API has always been able to run. That is right there and wrong here: a model
    identifier is not something to guess, and a table of guesses is worse than an empty one.
    """
    app = app_with()

    async with client(app) as http:
        response = await http.get("/model-configuration", headers=AUTH)

    assert response.status_code == 200
    # No deployment rows and no custom rows -- only the authoring affordance, which is about
    # the caller and the store rather than about what resolved.
    assert response.json() == {"editable": True, "setups": []}


@pytest.mark.asyncio
async def test_an_unauthenticated_request_reaches_no_table() -> None:
    """Deployment configuration is not public, even though none of it is a secret."""
    app = app_with(platforms={AgentPlatform.OPENAI: _OPENAI_HIGH})

    async with client(app) as http:
        response = await http.get("/model-configuration")

    assert response.status_code == 401
