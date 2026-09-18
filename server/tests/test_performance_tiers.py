"""Tiers resolve from configuration, with the unsuffixed variables as the high tier.

A tier is a named preset over the four model roles, not a routing input. What this file
protects: the unsuffixed variables keep meaning exactly what they meant (so no deployment
changes behaviour by upgrading), a `HIGH_`-suffixed variable overrides its unsuffixed twin,
the completeness rule applies per (platform, tier), and a preset that names a pairing the
deployment has declared unsupported is refused at startup rather than discovered as a 400 at
attempt time.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from configs.model_roles import (
    TIER_LABELS,
    AgentPlatform,
    ModelConfigurationError,
    ModelRole,
    PerformanceTier,
    PlatformModelSettings,
    build_model_config_service,
    tier_model_role_variables,
)
from configs.settings import CONFIG_DIRECTORY, REPOSITORY_ROOT, load_settings
from main import create_app
from services.feature_queue import InMemoryFeatureExecutionQueue
from storage.db import Database
from storage.feature_store import SqlAlchemyFeatureControlPlane
from storage.models import FeatureExecutionQueueModel, FeatureWorkflowModel
from tests.test_feature_api import feature_payload
from tests.test_model_routing import (
    feature_payload as routing_feature_payload,
)
from tests.test_model_routing import (
    no_credentials,
    router,
)
from workflows.feature_workflow import FeatureWorkflowOrchestrator, MockChildWorkstreamExecutor

KEY = "tier-key"
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

_OPENAI_LOW = PlatformModelSettings(
    models={
        ModelRole.REASONING: "gpt-5.6-terra",
        ModelRole.CODING: "gpt-5.3-codex",
        ModelRole.REVIEW: "gpt-5.6-terra",
        ModelRole.SCOPED_FIX: "gpt-5.3-codex",
    },
    reasoning={
        ModelRole.REASONING: "medium",
        ModelRole.CODING: "high",
        ModelRole.REVIEW: "medium",
        ModelRole.SCOPED_FIX: "medium",
    },
    max_tokens=dict.fromkeys(ModelRole, None),
)


def _cleared_tier_fields(*live: str) -> dict[str, object]:
    """Explicitly unset every tier field except the named ones.

    Pins the deployment `.env` out of these tests: once the checked-in presets land there, a
    `load_settings` call would otherwise resolve tiers this file never configured.
    """
    cleared: dict[str, object] = {}
    for platform in AgentPlatform:
        for tier in PerformanceTier.env_backed():
            prefix = f"{platform.value}_{tier.value}_"
            for role in ModelRole:
                cleared[f"{prefix}{role.value}_model"] = ""
                effort = (
                    f"{prefix}reasoning_effort"
                    if role is ModelRole.REASONING
                    else f"{prefix}{role.value}_reasoning_effort"
                )
                cleared[effort] = ""
                if platform is AgentPlatform.ANTHROPIC:
                    cleared[f"{prefix}{role.value}_max_tokens"] = None
    for name in live:
        cleared.pop(name, None)
    return cleared


def _base_model_overrides() -> dict[str, str]:
    """The eight unsuffixed model variables, pinned so the deployment `.env` cannot leak in."""
    return {
        "openai_reasoning_model": "gpt-5.6-sol",
        "openai_coding_model": "gpt-5.6-sol",
        "openai_review_model": "gpt-5.6-terra",
        "openai_scoped_fix_model": "gpt-5.3-codex",
        "openai_reasoning_effort": "max",
        "openai_coding_reasoning_effort": "max",
        "openai_review_reasoning_effort": "high",
        "openai_scoped_fix_reasoning_effort": "high",
        "anthropic_reasoning_model": "claude-opus-5",
        "anthropic_coding_model": "claude-opus-5",
        "anthropic_review_model": "claude-opus-5",
        "anthropic_scoped_fix_model": "claude-sonnet-5",
        "anthropic_reasoning_effort": "max",
        "anthropic_coding_reasoning_effort": "xhigh",
        "anthropic_review_reasoning_effort": "high",
        "anthropic_scoped_fix_reasoning_effort": "high",
        "model_reasoning_unsupported": "",
    }


# ---------------------------------------------------------------------------------- resolution


def test_an_environment_defining_tier_presets_resolves_every_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each (platform, tier, role) resolves to its own configured model, effort and bound.

    Set through real environment variables rather than keyword overrides, so what is proven
    is the `{PROVIDER}_{TIER}_{ROLE}_...` naming itself; every tier variable this test does
    not set is explicitly cleared so the deployment `.env` cannot leak in.
    """
    live = {
        "OPENAI_LOW_REASONING_MODEL": "gpt-5.6-terra",
        "OPENAI_LOW_REASONING_EFFORT": "medium",
        "OPENAI_LOW_CODING_MODEL": "gpt-5.3-codex",
        "OPENAI_LOW_CODING_REASONING_EFFORT": "high",
        "OPENAI_LOW_REVIEW_MODEL": "gpt-5.6-terra",
        "OPENAI_LOW_REVIEW_REASONING_EFFORT": "medium",
        "OPENAI_LOW_SCOPED_FIX_MODEL": "gpt-5.3-codex",
        "OPENAI_LOW_SCOPED_FIX_REASONING_EFFORT": "medium",
        "ANTHROPIC_MEDIUM_REASONING_MODEL": "claude-opus-5",
        "ANTHROPIC_MEDIUM_REASONING_EFFORT": "high",
        "ANTHROPIC_MEDIUM_CODING_MODEL": "claude-sonnet-5",
        "ANTHROPIC_MEDIUM_CODING_REASONING_EFFORT": "xhigh",
        "ANTHROPIC_MEDIUM_CODING_MAX_TOKENS": "128000",
        "ANTHROPIC_MEDIUM_REVIEW_MODEL": "claude-sonnet-5",
        "ANTHROPIC_MEDIUM_REVIEW_REASONING_EFFORT": "high",
        "ANTHROPIC_MEDIUM_SCOPED_FIX_MODEL": "claude-sonnet-5",
        "ANTHROPIC_MEDIUM_SCOPED_FIX_REASONING_EFFORT": "high",
    }
    for variable, value in live.items():
        monkeypatch.setenv(variable, value)
    settings = load_settings(
        CONFIG_DIRECTORY,
        **_base_model_overrides(),
        **_cleared_tier_fields(*(variable.lower() for variable in live)),
    )

    low_coding = settings.model_configs.get_model_config(
        ModelRole.CODING, platform=AgentPlatform.OPENAI, tier=PerformanceTier.LOW
    )
    assert (low_coding.model, low_coding.reasoning) == ("gpt-5.3-codex", "high")
    assert low_coding.tier is PerformanceTier.LOW
    assert low_coding.model_variable == "OPENAI_LOW_CODING_MODEL"
    medium_coding = settings.model_configs.get_model_config(
        ModelRole.CODING, platform=AgentPlatform.ANTHROPIC, tier=PerformanceTier.MEDIUM
    )
    assert (medium_coding.model, medium_coding.reasoning) == ("claude-sonnet-5", "xhigh")
    assert medium_coding.max_tokens == 128_000
    # A tier that names no bound of its own keeps the platform's per-role bound.
    medium_review = settings.model_configs.get_model_config(
        ModelRole.REVIEW, platform=AgentPlatform.ANTHROPIC, tier=PerformanceTier.MEDIUM
    )
    assert medium_review.max_tokens == settings.anthropic_review_max_tokens
    # The reasoning-role effort variable is the base name with a tier segment, not a
    # doubled `REASONING_REASONING`.
    low_reasoning = settings.model_configs.get_model_config(
        ModelRole.REASONING, platform=AgentPlatform.OPENAI, tier=PerformanceTier.LOW
    )
    assert low_reasoning.reasoning == "medium"
    assert (
        tier_model_role_variables(
            AgentPlatform.OPENAI, PerformanceTier.LOW, ModelRole.REASONING
        ).reasoning_variable
        == "OPENAI_LOW_REASONING_EFFORT"
    )
    assert settings.model_configs.configured_options() == (
        (AgentPlatform.OPENAI, PerformanceTier.LOW),
        (AgentPlatform.OPENAI, PerformanceTier.HIGH),
        (AgentPlatform.ANTHROPIC, PerformanceTier.MEDIUM),
        (AgentPlatform.ANTHROPIC, PerformanceTier.HIGH),
    )


def test_an_ultra_preset_resolves_above_the_unsuffixed_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ULTRA` is a fourth environment-backed preset: same naming, same completeness rule.

    Set through real environment variables so what is proven is the
    `{PROVIDER}_ULTRA_{ROLE}_...` naming itself; every tier variable this test does not set
    is explicitly cleared so the deployment `.env` cannot leak in.
    """
    live = {
        "OPENAI_ULTRA_REASONING_MODEL": "gpt-6-astra",
        "OPENAI_ULTRA_REASONING_EFFORT": "max",
        "OPENAI_ULTRA_CODING_MODEL": "gpt-6-astra",
        "OPENAI_ULTRA_CODING_REASONING_EFFORT": "max",
        "OPENAI_ULTRA_REVIEW_MODEL": "gpt-6-astra",
        "OPENAI_ULTRA_REVIEW_REASONING_EFFORT": "xhigh",
        "OPENAI_ULTRA_SCOPED_FIX_MODEL": "gpt-6-astra",
        "OPENAI_ULTRA_SCOPED_FIX_REASONING_EFFORT": "high",
    }
    for variable, value in live.items():
        monkeypatch.setenv(variable, value)
    settings = load_settings(
        CONFIG_DIRECTORY,
        **_base_model_overrides(),
        **_cleared_tier_fields(*(variable.lower() for variable in live)),
    )

    expected_efforts = {
        ModelRole.REASONING: "max",
        ModelRole.CODING: "max",
        ModelRole.REVIEW: "xhigh",
        ModelRole.SCOPED_FIX: "high",
    }
    for role, effort in expected_efforts.items():
        config = settings.model_configs.get_model_config(
            role, platform=AgentPlatform.OPENAI, tier=PerformanceTier.ULTRA
        )
        assert config.model == "gpt-6-astra"
        assert config.reasoning == effort
        # The Responses API needs no output bound: ultra declares none, like every
        # OpenAI tier, so the provider default stands.
        assert config.max_tokens is None
        assert config.tier is PerformanceTier.ULTRA
        assert (
            config.model_variable
            == tier_model_role_variables(
                AgentPlatform.OPENAI, PerformanceTier.ULTRA, role
            ).model_variable
        )
    # Offered above Max in the order a person is shown, and absent for the provider that
    # never configured it -- the same preference the completeness rule always encoded.
    assert settings.model_configs.configured_options() == (
        (AgentPlatform.OPENAI, PerformanceTier.HIGH),
        (AgentPlatform.OPENAI, PerformanceTier.ULTRA),
        (AgentPlatform.ANTHROPIC, PerformanceTier.HIGH),
    )
    assert TIER_LABELS[PerformanceTier.ULTRA] == "Ultra"


def test_a_start_request_may_name_the_ultra_tier() -> None:
    """The request vocabulary carries the new tier; the queue and state columns are strings."""
    request = StartFeatureRequest.model_validate({**feature_payload(), "performance_tier": "ultra"})
    assert request.performance_tier == "ultra"


def test_the_unsuffixed_variables_resolve_as_the_high_tier() -> None:
    """A deployment that never heard of tiers is a high-tier deployment, unchanged."""
    service = build_model_config_service(platforms={AgentPlatform.OPENAI: _OPENAI_HIGH})

    for role in ModelRole:
        untiered = service.get_model_config(role, platform=AgentPlatform.OPENAI)
        high = service.get_model_config(
            role, platform=AgentPlatform.OPENAI, tier=PerformanceTier.HIGH
        )
        assert untiered is high
        assert untiered.tier is PerformanceTier.HIGH
        assert untiered.as_metadata()["performance_tier"] == "high"
    assert service.configured_options() == ((AgentPlatform.OPENAI, PerformanceTier.HIGH),)
    assert service.is_tier_configured(AgentPlatform.OPENAI, PerformanceTier.HIGH)
    assert not service.is_tier_configured(AgentPlatform.OPENAI, PerformanceTier.LOW)


def test_an_explicit_high_variable_overrides_its_unsuffixed_twin() -> None:
    """`OPENAI_HIGH_CODING_MODEL`, when set, wins -- for tiered and untiered callers alike."""
    service = build_model_config_service(
        platforms={AgentPlatform.OPENAI: _OPENAI_HIGH},
        tiers={
            (AgentPlatform.OPENAI, PerformanceTier.HIGH): PlatformModelSettings(
                models={ModelRole.CODING: "gpt-6-preview"},
                reasoning={ModelRole.CODING: "xhigh"},
                max_tokens=dict.fromkeys(ModelRole, None),
            )
        },
    )

    coding = service.get_model_config(ModelRole.CODING, platform=AgentPlatform.OPENAI)
    assert (coding.model, coding.reasoning) == ("gpt-6-preview", "xhigh")
    assert coding.model_variable == "OPENAI_HIGH_CODING_MODEL"
    # The roles the overlay never named keep their unsuffixed resolution.
    review = service.get_model_config(ModelRole.REVIEW, platform=AgentPlatform.OPENAI)
    assert (review.model, review.model_variable) == ("gpt-5.6-terra", "OPENAI_REVIEW_MODEL")


# -------------------------------------------------------------------------------- completeness


def test_a_partial_tier_refuses_startup_naming_the_missing_variable() -> None:
    """Three of four roles is a mistake at tier granularity, exactly as it is per platform."""
    incomplete = PlatformModelSettings(
        models={
            ModelRole.REASONING: "gpt-5.6-terra",
            ModelRole.CODING: "gpt-5.3-codex",
            ModelRole.REVIEW: "gpt-5.6-terra",
        },
        reasoning=dict.fromkeys(ModelRole),
        max_tokens=dict.fromkeys(ModelRole, None),
    )

    with pytest.raises(ModelConfigurationError) as refused:
        build_model_config_service(
            platforms={AgentPlatform.OPENAI: _OPENAI_HIGH},
            tiers={(AgentPlatform.OPENAI, PerformanceTier.LOW): incomplete},
        )

    assert "OPENAI_LOW_SCOPED_FIX_MODEL" in str(refused.value)
    assert "gpt" not in str(refused.value)


def test_a_tier_nobody_configured_is_simply_not_offered() -> None:
    """No tier variables at all is a preference: the pairing is absent, not an error."""
    service = build_model_config_service(
        platforms={AgentPlatform.OPENAI: _OPENAI_HIGH},
        tiers={(AgentPlatform.OPENAI, PerformanceTier.LOW): _OPENAI_LOW},
    )

    assert (AgentPlatform.OPENAI, PerformanceTier.MEDIUM) not in service.configured_options()
    with pytest.raises(ModelConfigurationError, match="OPENAI_MEDIUM_CODING_MODEL"):
        service.get_model_config(
            ModelRole.CODING, platform=AgentPlatform.OPENAI, tier=PerformanceTier.MEDIUM
        )


# ---------------------------------------------------------------------------------- capability


def test_a_preset_naming_an_unsupported_pairing_refuses_startup() -> None:
    """`gpt-5.3-codex` takes `xhigh` but not `max`: a preset that asks anyway is a mistake.

    Refused at startup by the tier variable's name -- not discovered as a provider 400 when a
    live feature reaches its first low-tier model call.
    """
    bad_low = PlatformModelSettings(
        models=dict(_OPENAI_LOW.models),
        reasoning={**_OPENAI_LOW.reasoning, ModelRole.CODING: "max"},
        max_tokens=dict.fromkeys(ModelRole, None),
    )

    with pytest.raises(ModelConfigurationError) as refused:
        build_model_config_service(
            platforms={AgentPlatform.OPENAI: _OPENAI_HIGH},
            tiers={(AgentPlatform.OPENAI, PerformanceTier.LOW): bad_low},
            unsupported_reasoning={"gpt-5.3-codex": ["max"]},
        )

    assert "OPENAI_LOW_CODING_REASONING_EFFORT" in str(refused.value)


def test_an_unsuffixed_pairing_keeps_its_normalizing_behaviour() -> None:
    """The strict check is scoped to tier-suffixed configuration.

    A deployment that has always run `OPENAI_SCOPED_FIX_MODEL=gpt-5.3-codex` beside a
    declared-unsupported effort keeps starting, with the level dropped and the normalization
    recorded -- refusing it here would break an existing deployment that asked for nothing.
    """
    service = build_model_config_service(
        platforms={
            AgentPlatform.OPENAI: PlatformModelSettings(
                models=dict(_OPENAI_HIGH.models),
                reasoning={**_OPENAI_HIGH.reasoning, ModelRole.SCOPED_FIX: "max"},
                max_tokens=dict.fromkeys(ModelRole, None),
            )
        },
        unsupported_reasoning={"gpt-5.3-codex": ["max"]},
    )

    scoped_fix = service.get_model_config(ModelRole.SCOPED_FIX, platform=AgentPlatform.OPENAI)
    assert scoped_fix.reasoning is None
    assert scoped_fix.requested_reasoning == "max"


# ------------------------------------------------------------------------------- the tier view


def test_a_tier_scoped_view_answers_every_existing_question_with_the_tier() -> None:
    """`for_tier` is how the tier feeds the router and the adapters without changing them."""
    service = build_model_config_service(
        platforms={AgentPlatform.OPENAI: _OPENAI_HIGH},
        tiers={(AgentPlatform.OPENAI, PerformanceTier.LOW): _OPENAI_LOW},
    )

    view = service.for_tier(PerformanceTier.LOW)
    coding = view.get_model_config(ModelRole.CODING, platform=AgentPlatform.OPENAI)
    assert (coding.model, coding.reasoning, coding.tier) == (
        "gpt-5.3-codex",
        "high",
        PerformanceTier.LOW,
    )
    assert service.for_tier(PerformanceTier.HIGH) is service
    with pytest.raises(ModelConfigurationError, match="medium performance tier"):
        service.for_tier(PerformanceTier.MEDIUM)


def test_settings_scoped_to_a_tier_resolve_agents_on_that_tier() -> None:
    """The tier-scoped settings view carries the tier into every agent-level accessor.

    The per-agent effort override keeps today's semantics on it: when the deployment sets
    one, it beats the resolved tier's role-level effort -- it is a deployment-wide tuning
    knob, not a per-tier one.
    """
    settings = load_settings(
        CONFIG_DIRECTORY,
        **_base_model_overrides(),
        **_cleared_tier_fields(
            "openai_medium_reasoning_model",
            "openai_medium_reasoning_effort",
            "openai_medium_coding_model",
            "openai_medium_coding_reasoning_effort",
            "openai_medium_review_model",
            "openai_medium_review_reasoning_effort",
            "openai_medium_scoped_fix_model",
            "openai_medium_scoped_fix_reasoning_effort",
        ),
        openai_medium_reasoning_model="gpt-5.6-sol",
        openai_medium_reasoning_effort="high",
        openai_medium_coding_model="gpt-5.6-sol",
        openai_medium_coding_reasoning_effort="high",
        openai_medium_review_model="gpt-5.6-terra",
        openai_medium_review_reasoning_effort="high",
        openai_medium_scoped_fix_model="gpt-5.3-codex",
        openai_medium_scoped_fix_reasoning_effort="high",
        openai_planner_reasoning_effort="xhigh",
        openai_product_manager_reasoning_effort="",
    )

    assert settings.for_performance_tier(PerformanceTier.HIGH) is settings
    medium = settings.for_performance_tier(PerformanceTier.MEDIUM)
    coding = medium.model_config_for_role(ModelRole.CODING, platform=AgentPlatform.OPENAI)
    assert (coding.model, coding.reasoning, coding.tier) == (
        "gpt-5.6-sol",
        "high",
        PerformanceTier.MEDIUM,
    )
    # The engineer follows its role onto the tier; the planner's deployment override still
    # outranks the tier's role-level effort.
    assert medium.reasoning_effort_for_agent("engineer", platform=AgentPlatform.OPENAI) == "high"
    assert medium.reasoning_effort_for_agent("planner", platform=AgentPlatform.OPENAI) == "xhigh"
    # Everything that is not a model resolution is untouched on the view.
    assert medium.feature_queue_lease_seconds == settings.feature_queue_lease_seconds
    # And the original settings keep answering as the high tier.
    assert (
        settings.model_config_for_role(ModelRole.CODING, platform=AgentPlatform.OPENAI).tier
        is PerformanceTier.HIGH
    )


# -------------------------------------------------------------------------------------- pinning


def test_a_request_that_names_no_tier_reads_as_high() -> None:
    """Every feature written before tiers existed resolved the unsuffixed variables.

    The API default is honest for the reason the platform's is: it states what actually
    happens for a caller that never heard of the field, not a recommendation. The submission
    form states its own default.
    """
    request = StartFeatureRequest.model_validate(feature_payload())
    assert request.performance_tier == "high"


@pytest.mark.asyncio
async def test_the_tier_a_feature_was_submitted_on_is_published_and_never_re_resolved() -> None:
    """It survives a read and a listing, and no endpoint offers to change it."""
    app = create_app(platform_api_key=KEY)
    payload = {**feature_payload(), "performance_tier": "medium"}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        created = await client.post(
            "/features/start", headers={**AUTH, "Idempotency-Key": "tier-immutable"}, json=payload
        )
        fetched = await client.get("/features/feature-login", headers=AUTH)
        listed = await client.get("/features", headers=AUTH)

    assert created.json()["performance_tier"] == "medium"
    assert fetched.json()["performance_tier"] == "medium"
    # Published on the list read model too, which is built from indexed columns rather than
    # from parent state -- so the column and the response model have to agree.
    assert listed.json()["features"][0]["performance_tier"] == "medium"
    # There is no route that changes it: like the platform and the mode, the tier is fixed at
    # submission for the feature's whole life.
    paths = {getattr(route, "path", "") for route in app.routes}
    assert not any("tier" in path for path in paths)


@pytest.mark.asyncio
async def test_the_queue_entry_carries_the_tier_to_the_worker() -> None:
    """The worker that runs a feature is not the request that accepted it."""
    queue = InMemoryFeatureExecutionQueue()
    await queue.enqueue(
        feature_id="feature-tiered",
        execution_mode="mock",
        agent_platform="openai",
        requested_by="operator",
        performance_tier="low",
    )
    await queue.enqueue(
        feature_id="feature-untiered",
        execution_mode="mock",
        agent_platform="openai",
        requested_by="operator",
    )

    first = await queue.claim(owner="worker-a", lease_seconds=30)
    second = await queue.claim(owner="worker-a", lease_seconds=30)

    assert first is not None and first.performance_tier == "low"
    # An entry enqueued by a caller that never named a tier is high-tier work, exactly as the
    # feature it describes is.
    assert second is not None and second.performance_tier == "high"


@pytest.mark.asyncio
async def test_a_requeued_entry_keeps_the_pinned_tier() -> None:
    """Resume and retry are new work on the same feature, at the same cost basis."""
    queue = InMemoryFeatureExecutionQueue()
    await queue.enqueue(
        feature_id="feature-resumed",
        execution_mode="mock",
        agent_platform="anthropic",
        requested_by="operator",
        performance_tier="medium",
    )
    await queue.finish("feature-resumed", succeeded=True)
    await queue.requeue(
        feature_id="feature-resumed",
        execution_mode="mock",
        agent_platform="anthropic",
        intent="resume",
        performance_tier="medium",
    )

    claimed = await queue.claim(owner="worker-b", lease_seconds=30)

    assert claimed is not None and claimed.performance_tier == "medium"


# ------------------------------------------------------------------------------ routing records


@pytest.mark.asyncio
async def test_every_attempts_routing_record_names_the_tier() -> None:
    """The persisted `model_routing` metadata carries the feature's pinned cost basis."""
    request = StartFeatureRequest.model_validate(
        {**routing_feature_payload("feature-tier-routing"), "performance_tier": "medium"}
    )
    state = _initial_feature_state("feature-tier-routing", request)
    assert state.performance_tier == "medium"

    result = await FeatureWorkflowOrchestrator(
        child_executor=MockChildWorkstreamExecutor(), model_router=router()
    ).start(state, credentials=no_credentials())

    routing = result.child_workflows["backend"].model_routing
    assert routing is not None
    assert routing["performance_tier"] == "medium"


@pytest.mark.asyncio
async def test_a_feature_that_names_no_tier_routes_as_high() -> None:
    """The default's routing record states what the attempt actually resolved against."""
    request = StartFeatureRequest.model_validate(routing_feature_payload("feature-untiered"))
    state = _initial_feature_state("feature-untiered", request)

    result = await FeatureWorkflowOrchestrator(
        child_executor=MockChildWorkstreamExecutor(), model_router=router()
    ).start(state, credentials=no_credentials())

    routing = result.child_workflows["backend"].model_routing
    assert routing is not None
    assert routing["performance_tier"] == "high"


# ------------------------------------------------------------------------------- shipped presets


def test_the_checked_in_presets_resolve_to_eight_selectable_options() -> None:
    """`.env.example`'s recommended presets are configuration this code can actually read.

    Nothing else executes that file, so a mistyped variable name in it -- the one place a new
    deployment copies its configuration from -- would otherwise be discovered by an operator
    wondering why options they copied never appeared.
    """
    example = REPOSITORY_ROOT / ".env.example"
    declared: dict[str, str] = {}
    for line in example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        declared[name.strip().lower()] = value.strip()
    overrides = {
        name: value
        for name, value in declared.items()
        if name.startswith(("openai_", "anthropic_")) and name != "openai_api_key"
    }
    settings = load_settings(CONFIG_DIRECTORY, **overrides)

    assert settings.model_configs.configured_options() == (
        (AgentPlatform.OPENAI, PerformanceTier.LOW),
        (AgentPlatform.OPENAI, PerformanceTier.MEDIUM),
        (AgentPlatform.OPENAI, PerformanceTier.HIGH),
        (AgentPlatform.OPENAI, PerformanceTier.ULTRA),
        (AgentPlatform.ANTHROPIC, PerformanceTier.LOW),
        (AgentPlatform.ANTHROPIC, PerformanceTier.MEDIUM),
        (AgentPlatform.ANTHROPIC, PerformanceTier.HIGH),
        (AgentPlatform.ANTHROPIC, PerformanceTier.ULTRA),
    )

    # The §5 tables' load-bearing values: effort cut before model class in the middle tier,
    # codex promoted to coder in the low one, and review left on terra at `high`.
    def resolved(
        platform: AgentPlatform, tier: PerformanceTier, role: ModelRole
    ) -> tuple[str, str | None]:
        config = settings.model_configs.get_model_config(role, platform=platform, tier=tier)
        return (config.model, config.reasoning)

    # `high`, not `xhigh`: AB-Feature-181 spent the entire 128000-token bound -- the
    # model's output ceiling -- on `xhigh` thinking plus a partial implementation.
    assert resolved(AgentPlatform.ANTHROPIC, PerformanceTier.MEDIUM, ModelRole.CODING) == (
        "claude-sonnet-5",
        "high",
    )
    assert resolved(AgentPlatform.OPENAI, PerformanceTier.LOW, ModelRole.CODING) == (
        "gpt-5.3-codex",
        "high",
    )
    assert resolved(AgentPlatform.OPENAI, PerformanceTier.MEDIUM, ModelRole.REVIEW) == (
        "gpt-5.6-terra",
        "high",
    )
    # Haiku carries no effort: it does not take the parameter.
    assert resolved(AgentPlatform.ANTHROPIC, PerformanceTier.LOW, ModelRole.SCOPED_FIX) == (
        "claude-haiku-4-5",
        None,
    )
    # Astra everywhere on the OpenAI ultra tier: reasoning and coding at `max`, review at
    # `xhigh`, scoped fixes at `high`, no output bounds -- the provider default stands.
    assert resolved(AgentPlatform.OPENAI, PerformanceTier.ULTRA, ModelRole.CODING) == (
        "gpt-6-astra",
        "max",
    )
    assert resolved(AgentPlatform.OPENAI, PerformanceTier.ULTRA, ModelRole.REVIEW) == (
        "gpt-6-astra",
        "xhigh",
    )
    # Fable on the Anthropic ultra tier's judgment-heavy roles. Coding stays at `high`: a
    # higher effort with the bound flush at Fable's declared 128000 ceiling is the exact
    # AB-Feature-181 exhaustion shape the ceiling judge names. The review bound is raised to
    # 64000 -- the 48000 platform bound is the `xhigh` floor with zero headroom.
    assert resolved(AgentPlatform.ANTHROPIC, PerformanceTier.ULTRA, ModelRole.CODING) == (
        "claude-fable-5",
        "high",
    )
    assert resolved(AgentPlatform.ANTHROPIC, PerformanceTier.ULTRA, ModelRole.REASONING) == (
        "claude-fable-5",
        "max",
    )
    review = settings.model_configs.get_model_config(
        ModelRole.REVIEW, platform=AgentPlatform.ANTHROPIC, tier=PerformanceTier.ULTRA
    )
    assert (review.model, review.reasoning, review.max_tokens) == (
        "claude-fable-5",
        "xhigh",
        64_000,
    )
    assert resolved(AgentPlatform.ANTHROPIC, PerformanceTier.ULTRA, ModelRole.SCOPED_FIX) == (
        "claude-sonnet-5",
        "high",
    )
    # 128000 coding max_tokens on every configured Anthropic tier, because a truncation costs
    # a whole attempt and no tier saves money on that. Unconfigured pairings are not offered,
    # so they hold no bound to check.
    for tier in PerformanceTier.env_backed():
        if not settings.model_configs.is_tier_configured(AgentPlatform.ANTHROPIC, tier):
            continue
        coding = settings.model_configs.get_model_config(
            ModelRole.CODING, platform=AgentPlatform.ANTHROPIC, tier=tier
        )
        assert coding.max_tokens == 128_000, f"the {tier.value} tier's coding bound is not 128000"


def test_the_container_starts_when_every_tier_variable_is_present_but_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deployment that configures no tier still starts, with the tiers simply absent.

    This is the shape `docker-compose.yml` actually produces: interpolation cannot express
    absence, so an unset variable arrives as `NAME=`. The optional integer bounds are the
    only settings for which "" is not already the same as unset, and the API refused to start
    with twelve parse errors the first time these were passed through.
    """
    for platform in AgentPlatform:
        for tier in PerformanceTier.env_backed():
            for role in ModelRole:
                variables = tier_model_role_variables(platform, tier, role)
                monkeypatch.setenv(variables.model_variable, "")
                monkeypatch.setenv(variables.reasoning_variable, "")
                monkeypatch.setenv(variables.max_tokens_variable, "")

    settings = load_settings(CONFIG_DIRECTORY, **_base_model_overrides())

    # It starts, and offers exactly the tier the unsuffixed variables have always been.
    assert settings.model_configs.configured_options() == (
        (AgentPlatform.OPENAI, PerformanceTier.HIGH),
        (AgentPlatform.ANTHROPIC, PerformanceTier.HIGH),
    )
    assert settings.anthropic_low_coding_max_tokens is None


def test_every_tier_variable_is_passed_into_the_container() -> None:
    """A variable `docker-compose.yml` does not name never reaches the deployment.

    `.env` is read by Compose for interpolation; only the variables each service's
    `environment:` block names are actually set inside the container. Adding a preset to
    `.env` and `.env.example` while forgetting this file produces a deployment that offers a
    single tier while its configuration says six -- which is exactly what happened, and was
    found by an operator looking at the form rather than by this suite.

    Derived from the code that builds the names, so the two cannot drift.
    """
    compose = (REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    declared = {
        line.strip().split(":", 1)[0]
        for line in compose.splitlines()
        if line.strip() and not line.strip().startswith("#") and ":" in line
    }
    expected: set[str] = set()
    # Environment-backed tiers only: `custom` names no variables, so Compose has nothing to
    # pass for it.
    for platform in AgentPlatform:
        for tier in PerformanceTier.env_backed():
            for role in ModelRole:
                variables = tier_model_role_variables(platform, tier, role)
                expected.add(variables.model_variable)
                expected.add(variables.reasoning_variable)
                if platform is AgentPlatform.ANTHROPIC:
                    # Only the Messages API needs an output bound, so only this platform's
                    # tiers carry one.
                    expected.add(variables.max_tokens_variable)

    missing = sorted(expected - declared)
    assert not missing, f"docker-compose.yml never passes these into the container: {missing}"


# --------------------------------------------------------------------------------------- /setup


@pytest.mark.asyncio
async def test_the_setup_endpoint_serves_six_labeled_options_when_both_providers_are_full() -> None:
    """Two platforms times three tiers, each honestly labeled with its resolved models."""
    settings = load_settings(
        CONFIG_DIRECTORY,
        **_base_model_overrides(),
        **_cleared_tier_fields(
            "openai_low_reasoning_model",
            "openai_low_coding_model",
            "openai_low_review_model",
            "openai_low_scoped_fix_model",
            "openai_medium_reasoning_model",
            "openai_medium_coding_model",
            "openai_medium_review_model",
            "openai_medium_scoped_fix_model",
            "anthropic_low_reasoning_model",
            "anthropic_low_coding_model",
            "anthropic_low_review_model",
            "anthropic_low_scoped_fix_model",
            "anthropic_medium_reasoning_model",
            "anthropic_medium_coding_model",
            "anthropic_medium_review_model",
            "anthropic_medium_scoped_fix_model",
        ),
        openai_low_reasoning_model="gpt-5.6-terra",
        openai_low_coding_model="gpt-5.3-codex",
        openai_low_review_model="gpt-5.6-terra",
        openai_low_scoped_fix_model="gpt-5.3-codex",
        openai_medium_reasoning_model="gpt-5.6-sol",
        openai_medium_coding_model="gpt-5.6-sol",
        openai_medium_review_model="gpt-5.6-terra",
        openai_medium_scoped_fix_model="gpt-5.3-codex",
        anthropic_low_reasoning_model="claude-sonnet-5",
        anthropic_low_coding_model="claude-sonnet-5",
        anthropic_low_review_model="claude-sonnet-5",
        anthropic_low_scoped_fix_model="claude-haiku-4-5",
        anthropic_medium_reasoning_model="claude-opus-5",
        anthropic_medium_coding_model="claude-sonnet-5",
        anthropic_medium_review_model="claude-sonnet-5",
        anthropic_medium_scoped_fix_model="claude-sonnet-5",
    )
    app = create_app(platform_api_key=KEY)
    app.state.model_configs = settings.model_configs

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        setup = await client.get("/setup", headers=AUTH)

    entries = setup.json()["agent_platforms"]
    assert [(item["platform"], item["performance_tier"]) for item in entries] == [
        ("openai", "low"),
        ("openai", "medium"),
        ("openai", "high"),
        ("anthropic", "low"),
        ("anthropic", "medium"),
        ("anthropic", "high"),
    ]
    by_key = {(item["platform"], item["performance_tier"]): item for item in entries}
    assert all(item["configured"] for item in entries)
    assert by_key[("openai", "low")]["label"] == "OpenAI — Economy"
    assert by_key[("anthropic", "medium")]["label"] == "Claude — Standard"
    assert by_key[("anthropic", "high")]["label"] == "Claude — Max"
    assert by_key[("openai", "low")]["models"]["coding"] == "gpt-5.3-codex"
    assert by_key[("anthropic", "medium")]["models"]["coding"] == "claude-sonnet-5"
    # The high entries are the unsuffixed configuration, exactly as before tiers existed.
    assert by_key[("anthropic", "high")]["models"]["coding"] == "claude-opus-5"
    # The effort each role is sent at, published beside the model and keyed identically. Two
    # tiers can name the same model and differ only here, so a form holding the models alone
    # would show two different selections as the same work.
    assert by_key[("openai", "high")]["reasoning_efforts"] == {
        "reasoning": "max",
        "coding": "max",
        "review": "high",
        "scoped_fix": "high",
    }
    # These tiers name models but no effort of their own, and a tier does not inherit the
    # unsuffixed one: `null` says the provider's default applies, which is what runs.
    assert by_key[("openai", "low")]["reasoning_efforts"] == dict.fromkeys(
        ("reasoning", "coding", "review", "scoped_fix"), None
    )
    # Keyed identically to the models at every entry, so the two are read as one answer.
    for item in entries:
        assert set(item["reasoning_efforts"]) == set(item["models"])


@pytest.mark.asyncio
async def test_a_tier_nobody_configured_is_absent_from_setup() -> None:
    """A pairing the deployment does not offer is not shown disabled -- it is not shown."""
    settings = load_settings(
        CONFIG_DIRECTORY,
        **_base_model_overrides(),
        **_cleared_tier_fields(
            "openai_medium_reasoning_model",
            "openai_medium_coding_model",
            "openai_medium_review_model",
            "openai_medium_scoped_fix_model",
        ),
        openai_medium_reasoning_model="gpt-5.6-sol",
        openai_medium_coding_model="gpt-5.6-sol",
        openai_medium_review_model="gpt-5.6-terra",
        openai_medium_scoped_fix_model="gpt-5.3-codex",
    )
    app = create_app(platform_api_key=KEY)
    app.state.model_configs = settings.model_configs

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        setup = await client.get("/setup", headers=AUTH)

    pairs = [
        (item["platform"], item["performance_tier"]) for item in setup.json()["agent_platforms"]
    ]
    assert ("openai", "medium") in pairs
    assert ("openai", "low") not in pairs
    assert ("anthropic", "medium") not in pairs
    # The high pairings stay listed -- configured or visibly not.
    assert ("openai", "high") in pairs
    assert ("anthropic", "high") in pairs


# ------------------------------------------------------------------------ the durable round trip


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_the_pinned_tier_round_trips_through_the_durable_store(
    postgres_database: Database,
) -> None:
    """The tier survives acceptance, a state flush, and a restarted control plane.

    The flush goes through `_replace_state`, whose column-by-column projection is the `40-`
    regression this guards: a state field that projection does not name silently never
    persists, and the runtime clocks were lost to exactly that.
    """
    store = SqlAlchemyFeatureControlPlane(
        postgres_database, mock_runner=FeatureWorkflowOrchestrator()
    )
    request = StartFeatureRequest.model_validate(
        {**feature_payload(), "feature_id": "feature-tier-trip", "performance_tier": "medium"}
    )
    result = await store.start(
        request,
        idempotency_key="tier-round-trip",
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        owner_id="platform-admin",
    )
    feature_id = result.record.state.feature_id
    assert result.record.state.performance_tier == "medium"

    # The accepted queue row carries the tier for the worker that will claim it.
    async with postgres_database.session() as session:
        queue_row = await session.get(FeatureExecutionQueueModel, feature_id)
        assert queue_row is not None and queue_row.performance_tier == "medium"

    # A later state flush keeps the column true rather than silently dropping the field.
    flushed = (await store.get_record(feature_id)).state.model_copy(deep=True)
    flushed.current_agent = "tier-writer"
    await store._replace_state(feature_id, flushed, "tier_round_trip")  # noqa: SLF001
    async with postgres_database.session() as session:
        column = await session.scalar(
            select(FeatureWorkflowModel.performance_tier).where(
                FeatureWorkflowModel.feature_id == feature_id
            )
        )
    assert column == "medium"

    # A restarted process -- a fresh control plane over the same database -- reads it back.
    restarted = SqlAlchemyFeatureControlPlane(
        postgres_database, mock_runner=FeatureWorkflowOrchestrator()
    )
    assert (await restarted.get_record(feature_id)).state.performance_tier == "medium"
