"""A feature pinned to a model setup: resolution, the snapshot, and the start-path gates.

The spec's T2 (refused at start when the declarations changed underneath a saved setup, with
nothing queued), T4/T5 (a mixed setup resolves per role and never authenticates against the
wrong provider), T7 (the snapshot is a copy -- editing or deleting the setup changes nothing),
T8 (the postgres round trip through `_replace_state`), T9 (the routing record names the tier,
the setup and each role's own platform), and T10 (credentials refused at creation naming the
role). Plus the decisions the spec left open: a setup's effort outranks the per-agent
deployment overrides, and `for_performance_tier(CUSTOM)` refuses rather than resolving.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from adapters.llm_adapter import AnthropicLLMClient, OpenAILLMClient
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import _initial_feature_state
from api.feature_schemas import StartFeatureRequest
from configs.model_roles import (
    AgentPlatform,
    ModelConfigurationError,
    ModelRole,
    PerformanceTier,
    model_config_service_for_setup,
)
from configs.settings import CONFIG_DIRECTORY, load_settings
from services.feature_queue import InMemoryFeatureExecutionQueue
from storage.db import Database
from storage.feature_store import SqlAlchemyFeatureControlPlane
from storage.models import FeatureExecutionQueueModel, FeatureWorkflowModel
from tests.support import settle
from tests.test_feature_api import feature_payload
from tests.test_model_routing import feature_payload as routing_feature_payload
from tests.test_model_routing import no_credentials
from tests.test_model_setups import (
    AUTH,
    _Declarations,
    app_with,
    client,
    setup_payload,
    stocked_store,
)
from tests.test_performance_tiers import _base_model_overrides, _cleared_tier_fields
from tools.model_routing import ModelExecutionMode, ModelRouter, ModelRoutingInputs
from workflows.feature_workflow import FeatureWorkflowOrchestrator, MockChildWorkstreamExecutor


def snapshot(name: str = "Mixed pilot setup", setup_id: str = "setup-t") -> dict[str, Any]:
    """The pinned copy a feature carries: the T4 mixed shape, coding on OpenAI."""
    return {"setup_id": setup_id, "name": name, "roles": setup_payload()["roles"]}


def custom_settings(**overrides: Any) -> Any:
    """Deployment settings with the pilot base models and no tier presets leaking in."""
    return load_settings(
        CONFIG_DIRECTORY, **_base_model_overrides(), **_cleared_tier_fields(), **overrides
    )


# --------------------------------------------------------------- T4/T5: per-role resolution


def test_a_mixed_setup_resolves_each_role_on_its_own_platform() -> None:
    """T4's map half: the role-addressed service answers per role, on the pinned platform."""
    settings = custom_settings()
    clone = settings.for_model_setup(snapshot())
    service = clone.model_configs

    assert service.platform_for_role(ModelRole.CODING) is AgentPlatform.OPENAI
    assert service.platform_for_role(ModelRole.REASONING) is AgentPlatform.ANTHROPIC
    reasoning = service.get_model_config(ModelRole.REASONING, platform=AgentPlatform.ANTHROPIC)
    assert reasoning.model == "claude-opus-5"
    assert reasoning.reasoning == "max"
    assert reasoning.max_tokens == 64_000
    assert reasoning.tier is PerformanceTier.CUSTOM
    coding = service.get_model_config(ModelRole.CODING, platform=AgentPlatform.OPENAI)
    assert coding.model == "gpt-5.6-sol"
    # No bound for the Responses API, whatever the person typed elsewhere.
    assert coding.max_tokens is None
    # Provenance names the setup: there is no environment variable to point anybody at.
    assert coding.model_variable == 'model setup "Mixed pilot setup"'


def test_the_constructed_boundaries_are_each_platforms_client() -> None:
    """T4's boundary half: the adapters built from the clone carry the pinned selections.

    Constructed with a stub transport so the assertion is about the boundary -- which client
    class, which model, which bound, which effort -- rather than about a network call.
    """
    clone = custom_settings().for_model_setup(snapshot())

    engineer = OpenAILLMClient(clone, "engineer", client=object(), model_role=ModelRole.CODING)
    assert engineer.provider == "openai"
    assert engineer.model == "gpt-5.6-sol"
    assert engineer.reasoning_effort == "high"

    planner = AnthropicLLMClient(clone, "planner", client=object(), model_role=ModelRole.REASONING)
    assert planner.provider == "anthropic"
    assert planner.model == "claude-opus-5"
    assert planner.reasoning_effort == "max"
    assert planner.max_tokens == 64_000


def test_the_wrong_platform_never_answers_a_pinned_role() -> None:
    """T5: a role pinned to Claude cannot be constructed onto the Responses client.

    This is the guard that keeps a mixed setup from authenticating a call with the other
    provider's key: the mismatch is refused at construction, by name, before any request.
    """
    clone = custom_settings().for_model_setup(snapshot())

    with pytest.raises(ModelConfigurationError, match="pinned to anthropic"):
        OpenAILLMClient(clone, "engineer", client=object(), model_role=ModelRole.SCOPED_FIX)


def test_the_router_records_each_roles_own_platform() -> None:
    """The routing decision names the platform the role actually resolves on.

    Constructed with the *other* provider as the feature platform on purpose: the
    role-addressed service outranks it, because a scoped fix pinned to Claude recorded as
    `openai` would rebuild its client against the wrong API on recovery. For every
    environment-backed service `platform_for_role` answers nothing and the feature's platform
    stands, which is the router's unchanged behaviour.
    """
    service = model_config_service_for_setup(snapshot(), unsupported={}, ceilings={})
    router = ModelRouter(service, platform=AgentPlatform.ANTHROPIC)

    outcome = router.route(
        inputs=ModelRoutingInputs(
            feature_id="feature-router",
            repository_id="backend",
            execution_mode=ModelExecutionMode.INITIAL_IMPLEMENTATION,
            attempt=0,
        )
    )

    assert outcome.decision.platform is AgentPlatform.OPENAI
    assert outcome.decision.model == "gpt-5.6-sol"


# ------------------------------------------------------- the decisions the spec left open


def test_a_setups_effort_outranks_the_per_agent_deployment_override() -> None:
    """Decision 12.4: a pinned, recorded choice is not silently defeated by a deployment knob."""
    settings = custom_settings(openai_planner_reasoning_effort="low")
    clone = settings.for_model_setup(snapshot())

    # The base configuration keeps the operator's knob for tier features...
    assert settings.reasoning_effort_for_agent("planner", platform=AgentPlatform.OPENAI) == "low"
    # ...and the setup-scoped clone follows the person's pinned effort.
    assert clone.reasoning_effort_for_agent("planner", platform=AgentPlatform.ANTHROPIC) == "max"


def test_the_tier_path_refuses_custom_rather_than_resolving_something_else() -> None:
    """A caller that reached the tier path with CUSTOM has lost the snapshot; that is a bug."""
    settings = custom_settings()

    with pytest.raises(ModelConfigurationError, match="pinned model setup snapshot"):
        settings.for_performance_tier(PerformanceTier.CUSTOM)


def test_the_ceiling_declaration_parses_like_the_capability_declaration() -> None:
    """MODEL_MAX_OUTPUT_TOKENS: same JSON-object shape, same refusal posture.

    Since 74-, malformed JSON refuses at startup rather than at the first custom-setup
    use -- the tier ceiling checks parse the declaration during validation, the same
    moment MODEL_REASONING_UNSUPPORTED has always been parsed. A ceiling declaration
    that silently failed to apply was exactly the blindspot item 29 closed.
    """
    settings = custom_settings(model_max_output_tokens='{"claude-sonnet-5": 128000}')
    assert settings.declared_max_output_tokens() == {"claude-sonnet-5": 128000}

    with pytest.raises(ValidationError, match="MODEL_MAX_OUTPUT_TOKENS must be"):
        custom_settings(model_max_output_tokens='["claude-sonnet-5"]')


def test_a_stale_snapshot_is_refused_when_it_is_resolved_again() -> None:
    """Belt and braces: the snapshot is re-validated at resolution, never clamped."""
    with pytest.raises(ModelConfigurationError, match="exceeds"):
        model_config_service_for_setup(
            snapshot(), unsupported={}, ceilings={"claude-sonnet-5": 40_000}
        )


# ----------------------------------------------------------------- T2/T10: the start gates


@pytest.mark.asyncio
async def test_a_setup_invalidated_by_new_declarations_is_refused_at_start() -> None:
    """T2: valid when saved, invalid now -- refused with the same sentence, nothing queued."""
    app = app_with(secret_store=await stocked_store("anthropic", "openai", "github"))

    async with client(app) as http:
        created = await http.post("/model-setups", headers=AUTH, json=setup_payload())
        assert created.status_code == 201
        setup_id = created.json()["setup_id"]

        # The deployment's declarations change underneath the saved setup: claude-sonnet-5's
        # ceiling is now below the review role's authored bound.
        app.state.settings = _Declarations(ceilings={"claude-sonnet-5": 40_000})

        refused = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "t2"},
            json={
                **feature_payload(),
                "execution_mode": "live",
                "model_setup_id": setup_id,
            },
        )
        listed = await http.get("/features", headers=AUTH)

    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert "roles.review.max_tokens=48000" in detail
    assert "40000" in detail
    # The refusal is before acceptance: no feature and no queue row exist afterwards.
    assert listed.json()["features"] == []
    assert await app.state.feature_queue.pending_count() == 0


@pytest.mark.asyncio
async def test_a_missing_pinned_credential_refuses_the_start_naming_the_role() -> None:
    """T10: the refusal names the platform and the roles that pinned it, and queues nothing."""
    store = await stocked_store("anthropic", "openai", "github")
    app = app_with(secret_store=store)

    async with client(app) as http:
        created = await http.post("/model-setups", headers=AUTH, json=setup_payload())
        assert created.status_code == 201
        setup_id = created.json()["setup_id"]

        # The credential is deleted between creation and submission -- G4's residual case.
        await store.delete(owner_id="platform-admin", provider="openai")

        refused = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "t10"},
            json={
                **feature_payload(),
                "execution_mode": "live",
                "model_setup_id": setup_id,
            },
        )
        listed = await http.get("/features", headers=AUTH)

    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert "OpenAI" in detail
    assert "coding is pinned to it" in detail
    assert listed.json()["features"] == []


@pytest.mark.asyncio
async def test_a_setup_and_an_explicit_tier_are_two_answers_to_one_question() -> None:
    """Not a precedence rule: the request is refused, exactly as the 173 lesson demands."""
    app = app_with()

    async with client(app) as http:
        created = await http.post("/model-setups", headers=AUTH, json=setup_payload())
        setup_id = created.json()["setup_id"]
        refused = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "both"},
            json={
                **feature_payload(),
                "model_setup_id": setup_id,
                "performance_tier": "medium",
            },
        )

    assert refused.status_code == 422
    assert "two answers to one question" in refused.text


@pytest.mark.asyncio
async def test_an_unknown_or_foreign_setup_id_answers_404() -> None:
    """T13's start half: "does not exist" and "is not yours" are the same answer."""
    app = app_with()

    async with client(app) as http:
        refused = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "missing"},
            json={**feature_payload(), "model_setup_id": "setup-nobody"},
        )

    assert refused.status_code == 404


# --------------------------------------------------------------- T7: the snapshot is a copy


@pytest.mark.asyncio
async def test_editing_and_deleting_the_setup_never_changes_the_running_feature() -> None:
    """T7 and G3: the feature keeps its snapshot through an edit and through a delete."""
    app = app_with()

    async with client(app) as http:
        created = await http.post("/model-setups", headers=AUTH, json=setup_payload())
        setup_id = created.json()["setup_id"]

        started = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "t7"},
            json={**feature_payload(), "model_setup_id": setup_id},
        )
        assert started.status_code == 201
        body = started.json()
        feature_id = body["feature_id"]
        assert body["performance_tier"] == "custom"
        # Decision 12.5: the NOT NULL platform column records the coding role's platform.
        assert body["agent_platform"] == "openai"
        await settle(app)

        # Change every role, then check the feature; then delete, then check again.
        edited = setup_payload()
        edited["roles"] = {
            role: {"platform": "openai", "model": f"gpt-other-{role}"}
            for role in ("reasoning", "coding", "review", "scoped_fix")
        }
        assert (
            await http.put(f"/model-setups/{setup_id}", headers=AUTH, json=edited)
        ).status_code == 200
        after_edit = (await http.get(f"/features/{feature_id}", headers=AUTH)).json()

        assert (await http.delete(f"/model-setups/{setup_id}", headers=AUTH)).status_code == 204
        after_delete = (await http.get(f"/features/{feature_id}", headers=AUTH)).json()

    pin = after_edit["model_setup"]
    assert pin["setup_state"] == "edited"
    by_role = {item["role"]: item for item in pin["roles"]}
    # Still the authored values, in full -- not the edit.
    assert by_role["coding"]["model"] == "gpt-5.6-sol"
    assert by_role["reasoning"]["model"] == "claude-opus-5"
    assert by_role["reasoning"]["max_tokens"] == 64_000
    assert after_edit["performance_tier"] == "custom"

    pin = after_delete["model_setup"]
    assert pin["setup_state"] == "deleted"
    assert {item["role"] for item in pin["roles"]} == {
        "reasoning",
        "coding",
        "review",
        "scoped_fix",
    }


@pytest.mark.asyncio
async def test_the_queue_entry_carries_the_snapshot_to_the_worker() -> None:
    """The worker decides which credentials an entry needs before it loads any state."""
    queue = InMemoryFeatureExecutionQueue()
    await queue.enqueue(
        feature_id="feature-pinned",
        execution_mode="mock",
        agent_platform="openai",
        requested_by="operator",
        performance_tier="custom",
        model_setup_snapshot=snapshot(),
        model_setup_id="setup-t",
    )

    claimed = await queue.claim(owner="worker-a", lease_seconds=30)

    assert claimed is not None
    assert claimed.model_setup_id == "setup-t"
    assert claimed.model_setup_snapshot is not None
    assert claimed.model_setup_snapshot["roles"]["coding"]["platform"] == "openai"


# ------------------------------------------------------------------- T9: the routing record


@pytest.mark.asyncio
async def test_the_routing_record_names_the_setup_the_tier_and_the_roles_platform() -> None:
    """T9: `custom`, the setup id, and the role's own platform, on the persisted decision."""
    request = StartFeatureRequest.model_validate(
        {**routing_feature_payload("feature-custom-routing"), "model_setup_id": "setup-t"}
    )
    state = _initial_feature_state("feature-custom-routing", request, model_setup=snapshot())
    assert state.performance_tier == "custom"
    assert state.model_setup_id == "setup-t"
    service = model_config_service_for_setup(snapshot(), unsupported={}, ceilings={})

    result = await FeatureWorkflowOrchestrator(
        child_executor=MockChildWorkstreamExecutor(),
        model_router=ModelRouter(service, platform=AgentPlatform.OPENAI),
    ).start(state, credentials=no_credentials())

    routing = result.child_workflows["backend"].model_routing
    assert routing is not None
    assert routing["performance_tier"] == "custom"
    assert routing["model_setup_id"] == "setup-t"
    assert routing["platform"] == "openai"
    assert routing["model"] == "gpt-5.6-sol"
    assert routing["reasoning"] == "high"
    # And the feature carries the whole four-role snapshot -- the record in full.
    assert result.model_setup_snapshot == snapshot()


# ----------------------------------------------------------------- T8: the postgres round trip


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_the_pinned_snapshot_round_trips_through_the_durable_store(
    postgres_database: Database,
) -> None:
    """T8: the snapshot survives acceptance, a `_replace_state` flush, and a restart.

    The flush goes through the column-by-column projection the `40-` regression documented:
    a state field that projection does not name silently never persists, and only this tier
    catches it -- the default tier reads its own writes.
    """
    store = SqlAlchemyFeatureControlPlane(
        postgres_database, mock_runner=FeatureWorkflowOrchestrator()
    )
    request = StartFeatureRequest.model_validate(
        {
            **feature_payload(),
            "feature_id": "feature-setup-trip",
            "model_setup_id": "setup-t",
        }
    )
    result = await store.start(
        request,
        idempotency_key="setup-round-trip",
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        model_setup=snapshot(),
        owner_id="platform-admin",
    )
    feature_id = result.record.state.feature_id
    assert result.record.state.performance_tier == "custom"
    assert result.record.state.model_setup_snapshot == snapshot()

    # The accepted queue row carries the snapshot for the worker that will claim it.
    async with postgres_database.session() as session:
        queue_row = await session.get(FeatureExecutionQueueModel, feature_id)
        assert queue_row is not None
        assert queue_row.model_setup_id == "setup-t"
        assert queue_row.model_setup_snapshot is not None
        assert queue_row.model_setup_snapshot["roles"]["coding"]["model"] == "gpt-5.6-sol"

    # A later state flush keeps the columns true rather than silently dropping the fields.
    flushed = (await store.get_record(feature_id)).state.model_copy(deep=True)
    flushed.current_agent = "setup-writer"
    await store._replace_state(feature_id, flushed, "setup_round_trip")  # noqa: SLF001
    async with postgres_database.session() as session:
        row = (
            await session.execute(
                select(
                    FeatureWorkflowModel.model_setup_id,
                    FeatureWorkflowModel.model_setup_snapshot,
                    FeatureWorkflowModel.performance_tier,
                ).where(FeatureWorkflowModel.feature_id == feature_id)
            )
        ).one()
    assert row.model_setup_id == "setup-t"
    assert row.model_setup_snapshot == snapshot()
    assert row.performance_tier == "custom"

    # A restarted process -- a fresh control plane over the same database -- reads it back.
    restarted = SqlAlchemyFeatureControlPlane(
        postgres_database, mock_runner=FeatureWorkflowOrchestrator()
    )
    recovered = (await restarted.get_record(feature_id)).state
    assert recovered.model_setup_snapshot == snapshot()
    assert recovered.model_setup_id == "setup-t"
