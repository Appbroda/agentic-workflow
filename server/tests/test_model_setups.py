"""Authoring model setups: the 181 footgun refused at save, and the published table's flip.

What this file protects, by spec 59- section: the validation predicate's three ceiling rules
and the floor rule, refused with the honest sentence and never a clamp (T1, T3); credentials
at creation naming the role (T11); the resolved table serving the caller's setups beside the
deployment rows without changing what those rows mean (T12); identity scoping (T13);
permission (T14); and `CUSTOM` staying out of every environment-backed loop (T6).
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from api.identity import Role
from configs.model_roles import (
    AgentPlatform,
    ModelConfigurationError,
    ModelRole,
    PerformanceTier,
    PlatformModelSettings,
    build_model_config_service,
    parse_model_setup_roles,
    validate_model_setup,
)
from main import create_app
from services.secrets import InMemorySecretStore
from storage.model_setup_store import (
    MAX_SETUPS_PER_OWNER,
    InMemoryModelSetupDirectory,
    ModelSetupError,
)
from storage.user_store import InMemoryUserDirectory

KEY = "model-setups-key"
AUTH = {"Authorization": f"Bearer {KEY}"}

_ANTHROPIC_HIGH = PlatformModelSettings(
    models={
        ModelRole.REASONING: "claude-opus-5",
        ModelRole.CODING: "claude-sonnet-5",
        ModelRole.REVIEW: "claude-sonnet-5",
        ModelRole.SCOPED_FIX: "claude-haiku-4-5",
    },
    reasoning={
        ModelRole.REASONING: "max",
        ModelRole.CODING: "high",
        ModelRole.REVIEW: "high",
        ModelRole.SCOPED_FIX: None,
    },
    max_tokens={
        ModelRole.REASONING: 64_000,
        ModelRole.CODING: 128_000,
        ModelRole.REVIEW: 48_000,
        ModelRole.SCOPED_FIX: 32_000,
    },
)


class _Declarations:
    """The deployment declarations these routes read, and nothing else.

    Three now, not two: the published setup and pairing rows carry whether their reasoning
    model can be shown an image, so `vision` joins the pair the validation predicate reads.
    """

    def __init__(
        self,
        *,
        unsupported: dict[str, list[str]] | None = None,
        ceilings: dict[str, int] | None = None,
        vision: list[str] | None = None,
    ) -> None:
        self._unsupported = unsupported or {}
        self._ceilings = ceilings or {}
        self._vision = vision or []

    def declared_unsupported_reasoning(self) -> dict[str, list[str]]:
        return dict(self._unsupported)

    def declared_max_output_tokens(self) -> dict[str, int]:
        return dict(self._ceilings)

    def declared_vision_capable(self) -> frozenset[str]:
        return frozenset(self._vision)


def setup_payload(**overrides: Any) -> dict[str, Any]:
    """A valid mixed setup: three roles on Claude, coding on OpenAI (the spec's T4 shape)."""
    payload: dict[str, Any] = {
        "name": "Mixed pilot setup",
        "roles": {
            "reasoning": {
                "platform": "anthropic",
                "model": "claude-opus-5",
                "reasoning_effort": "max",
                "max_tokens": 64_000,
            },
            "coding": {
                "platform": "openai",
                "model": "gpt-5.6-sol",
                "reasoning_effort": "high",
            },
            "review": {
                "platform": "anthropic",
                "model": "claude-sonnet-5",
                "reasoning_effort": "high",
                "max_tokens": 48_000,
            },
            "scoped_fix": {
                "platform": "anthropic",
                "model": "claude-haiku-4-5",
                "max_tokens": 32_000,
            },
        },
    }
    payload.update(overrides)
    return payload


def app_with(
    *,
    secret_store: Any = None,
    user_directory: Any = None,
    ceilings: dict[str, int] | None = None,
    unsupported: dict[str, list[str]] | None = None,
) -> Any:
    """Build an application whose declarations are the ones under test."""
    app = create_app(platform_api_key=KEY, secret_store=secret_store, user_directory=user_directory)
    app.state.settings = _Declarations(unsupported=unsupported, ceilings=ceilings)
    app.state.model_configs = build_model_config_service(
        platforms={AgentPlatform.ANTHROPIC: _ANTHROPIC_HIGH},
        unsupported_reasoning=unsupported,
    )
    return app


def client(app: Any) -> AsyncClient:
    """Return an HTTP client bound to the application under test."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def stocked_store(*providers: str) -> InMemorySecretStore:
    """A secret store where the platform admin has the named providers configured."""
    store = InMemorySecretStore()
    for provider in providers:
        await store.put(owner_id="platform-admin", provider=provider, secret=f"sk-{provider}")
    return store


# ------------------------------------------------------------------ T1: refused at save


@pytest.mark.asyncio
async def test_a_bound_over_the_declared_ceiling_is_refused_at_save() -> None:
    """Rule 1: max_tokens above a declared ceiling is refused, naming both numbers."""
    app = app_with(ceilings={"claude-sonnet-5": 128_000})
    payload = setup_payload()
    payload["roles"]["review"]["max_tokens"] = 200_000

    async with client(app) as http:
        response = await http.post("/model-setups", headers=AUTH, json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "roles.review" in detail
    assert "200000" in detail and "128000" in detail
    assert "MODEL_MAX_OUTPUT_TOKENS" in detail


@pytest.mark.asyncio
async def test_a_bound_under_the_efforts_floor_is_refused_at_save() -> None:
    """Rule 2: the existing floor sentence, verbatim -- value, role, why, minimum, both outs."""
    app = app_with()
    payload = setup_payload()
    payload["roles"]["review"]["max_tokens"] = 16_000

    async with client(app) as http:
        response = await http.post("/model-setups", headers=AUTH, json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "roles.review.max_tokens=16000" in detail
    assert "review" in detail and "'high'" in detail
    assert "thinking spends the same max_tokens bound" in detail
    assert "at least 32000" in detail
    assert "lower roles.review.reasoning_effort" in detail


@pytest.mark.asyncio
async def test_an_unsatisfiable_effort_ceiling_pairing_is_refused_at_save() -> None:
    """Rule 3: when the floor exceeds the ceiling, no bound exists, and the refusal says so.

    This is AB-Feature-181 refused at authoring time: the alternative was telling somebody to
    raise a bound past the model's ceiling, which is advice nobody can follow.
    """
    app = app_with(ceilings={"claude-sonnet-5": 32_000})
    payload = setup_payload()
    payload["roles"]["review"]["reasoning_effort"] = "xhigh"

    async with client(app) as http:
        response = await http.post("/model-setups", headers=AUTH, json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "'xhigh'" in detail and "claude-sonnet-5" in detail
    assert "48000" in detail and "32000" in detail
    assert "choose a lower effort or a different model" in detail


@pytest.mark.asyncio
async def test_a_declared_unsupported_pairing_is_refused_never_normalized() -> None:
    """A setup is always strict: the person asked for that pairing explicitly (spec §6.2)."""
    app = app_with(unsupported={"gpt-5.6-sol": ["high"]})

    async with client(app) as http:
        response = await http.post("/model-setups", headers=AUTH, json=setup_payload())

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "roles.coding" in detail
    assert "MODEL_REASONING_UNSUPPORTED" in detail


@pytest.mark.asyncio
async def test_the_181_shape_is_accepted_with_a_warning() -> None:
    """A coding role at xhigh+ whose bound sits exactly at the ceiling is valid -- and named.

    Decision 12.2(a): the shipped preset must stay acceptable, so this is a warning rather
    than a refusal, surfaced where the setup is authored.
    """
    app = app_with(ceilings={"claude-sonnet-5": 128_000})
    payload = setup_payload()
    payload["roles"]["coding"] = {
        "platform": "anthropic",
        "model": "claude-sonnet-5",
        "reasoning_effort": "xhigh",
        "max_tokens": 128_000,
    }

    async with client(app) as http:
        response = await http.post("/model-setups", headers=AUTH, json=payload)

    assert response.status_code == 201
    body = response.json()
    assert any("AB-Feature-181" in item for item in body["warnings"])


@pytest.mark.asyncio
async def test_an_openai_role_with_a_bound_is_refused_rather_than_ignored() -> None:
    """Silently dropping an authored value is a silent rewrite; the refusal says why."""
    app = app_with()
    payload = setup_payload()
    payload["roles"]["coding"]["max_tokens"] = 64_000

    async with client(app) as http:
        response = await http.post("/model-setups", headers=AUTH, json=payload)

    assert response.status_code == 422
    assert "Responses API" in response.json()["detail"]


@pytest.mark.asyncio
async def test_a_partial_setup_is_refused_like_a_partial_tier() -> None:
    """All four roles or nothing -- the same contract a partial tier meets at startup."""
    app = app_with()
    payload = setup_payload()
    del payload["roles"]["scoped_fix"]

    async with client(app) as http:
        response = await http.post("/model-setups", headers=AUTH, json=payload)

    assert response.status_code == 422
    assert "scoped_fix" in response.json()["detail"]


# ------------------------------------------------------------------ T3: never a clamp


@pytest.mark.asyncio
async def test_a_refused_edit_leaves_the_stored_setup_exactly_as_authored() -> None:
    """T3: after a refusal, nothing anywhere recorded a lowered effort or a raised bound."""
    app = app_with(ceilings={"claude-sonnet-5": 128_000})
    payload = setup_payload()

    async with client(app) as http:
        created = await http.post("/model-setups", headers=AUTH, json=payload)
        assert created.status_code == 201
        setup_id = created.json()["setup_id"]

        broken = setup_payload()
        broken["roles"]["review"]["max_tokens"] = 200_000
        refused = await http.put(f"/model-setups/{setup_id}", headers=AUTH, json=broken)
        assert refused.status_code == 422

        listed = (await http.get("/model-setups", headers=AUTH)).json()["setups"]

    assert len(listed) == 1
    roles = listed[0]["roles"]
    for role, expected in payload["roles"].items():
        assert roles[role]["platform"] == expected["platform"]
        assert roles[role]["model"] == expected["model"]
        assert roles[role]["reasoning_effort"] == expected.get("reasoning_effort")
        assert roles[role]["max_tokens"] == expected.get("max_tokens")


# ------------------------------------------------------------------ T11: credentials at save


@pytest.mark.asyncio
async def test_a_setup_naming_a_keyless_platform_is_refused_naming_the_role() -> None:
    """G4 at authoring: the refusal names the role and the platform, not just the platform."""
    app = app_with(secret_store=await stocked_store("anthropic", "github"))

    async with client(app) as http:
        response = await http.post("/model-setups", headers=AUTH, json=setup_payload())

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "coding is pinned to OpenAI" in detail
    assert "no OpenAI credential is stored for you" in detail


@pytest.mark.asyncio
async def test_a_deployment_with_no_secret_store_saves_without_the_credential_gate() -> None:
    """The header-credentials precedent: no store has always meant no stored-key requirement."""
    app = app_with(secret_store=None)

    async with client(app) as http:
        response = await http.post("/model-setups", headers=AUTH, json=setup_payload())

    assert response.status_code == 201


# ------------------------------------------------------------------ T12: the published table


@pytest.mark.asyncio
async def test_editable_flips_and_the_deployment_rows_do_not_change() -> None:
    """T12: the caller's setup joins the table as its own row; deployment rows keep meaning."""
    app = app_with(secret_store=await stocked_store("anthropic", "openai", "github"))

    async with client(app) as http:
        created = await http.post("/model-setups", headers=AUTH, json=setup_payload())
        assert created.status_code == 201
        setup_id = created.json()["setup_id"]
        payload = (await http.get("/model-configuration", headers=AUTH)).json()

    assert payload["editable"] is True
    deployment_rows = [item for item in payload["setups"] if item["origin"] == "deployment"]
    custom_rows = [item for item in payload["setups"] if item["origin"] == "custom"]
    assert deployment_rows, "the deployment's own pairings must still be published"
    for row in deployment_rows:
        assert row["editable"] is False
        assert row["setup_id"] is None
    assert len(custom_rows) == 1
    row = custom_rows[0]
    assert row["setup_id"] == setup_id
    assert row["editable"] is True
    assert row["performance_tier"] == "custom"
    assert row["label"] == "Mixed pilot setup"
    # A mixed setup names each role's own platform, because the pairing-level platform -- the
    # coding role's -- can only name one of them.
    by_role = {item["role"]: item for item in row["roles"]}
    assert by_role["coding"]["platform"] == "openai"
    assert by_role["reasoning"]["platform"] == "anthropic"
    # Provenance names the setup, not an environment variable nobody can grep for.
    assert by_role["coding"]["model_variable"] == 'model setup "Mixed pilot setup"'


@pytest.mark.asyncio
async def test_without_the_permission_no_custom_row_appears_and_editable_is_false() -> None:
    """T12's other half, and T14: a viewer reads the table and authors nothing."""
    directory = InMemoryUserDirectory()
    app = app_with(user_directory=directory)
    viewer = await directory.create_user(
        subject="viewer", display_name="A viewer", roles=(Role.VIEWER.value,)
    )
    token = (await directory.issue_token(viewer.user_id, label="test")).token
    viewer_auth = {"Authorization": f"Bearer {token}"}

    async with client(app) as http:
        table = (await http.get("/model-configuration", headers=viewer_auth)).json()
        payload = setup_payload()
        refused = [
            (await http.get("/model-setups", headers=viewer_auth)).status_code,
            (await http.post("/model-setups", headers=viewer_auth, json=payload)).status_code,
            (await http.put("/model-setups/x", headers=viewer_auth, json=payload)).status_code,
            (await http.delete("/model-setups/x", headers=viewer_auth)).status_code,
        ]

    assert table["editable"] is False
    assert all(item["origin"] == "deployment" for item in table["setups"])
    assert refused == [403, 403, 403, 403]


# ------------------------------------------------------------------ T13: identity scoping


@pytest.mark.asyncio
async def test_another_identitys_setup_never_appears_in_this_identitys_table() -> None:
    """T13: setups are per-identity, and neither the list nor the table crosses owners."""
    directory = InMemoryUserDirectory()
    app = app_with(user_directory=directory)

    async def operator_auth(subject: str) -> dict[str, str]:
        user = await directory.create_user(
            subject=subject, display_name=subject, roles=(Role.OPERATOR.value,)
        )
        issued = await directory.issue_token(user.user_id, label="t")
        return {"Authorization": f"Bearer {issued.token}"}

    first = await operator_auth("first-person")
    second = await operator_auth("second-person")

    async with client(app) as http:
        created = await http.post("/model-setups", headers=first, json=setup_payload())
        assert created.status_code == 201
        setup_id = created.json()["setup_id"]

        own_list = (await http.get("/model-setups", headers=second)).json()["setups"]
        own_table = (await http.get("/model-configuration", headers=second)).json()["setups"]
        foreign_delete = await http.delete(f"/model-setups/{setup_id}", headers=second)

    assert own_list == []
    assert all(item["origin"] == "deployment" for item in own_table)
    # "Does not exist" and "is not yours" are the same answer, so ids cannot be enumerated.
    assert foreign_delete.status_code == 404


# ------------------------------------------------------------------ limits


@pytest.mark.asyncio
async def test_a_duplicate_name_is_refused_per_owner() -> None:
    """The name is the label a person selects by; two of them would be one choice twice."""
    app = app_with()

    async with client(app) as http:
        first = await http.post("/model-setups", headers=AUTH, json=setup_payload())
        second = await http.post("/model-setups", headers=AUTH, json=setup_payload())

    assert first.status_code == 201
    assert second.status_code == 422
    assert "already exists" in second.json()["detail"]


@pytest.mark.asyncio
async def test_the_per_identity_cap_refuses_the_twenty_first_setup() -> None:
    """The cap exists to stop unbounded growth in a user-writable table, not to ration."""
    directory = InMemoryModelSetupDirectory()
    roles = setup_payload()["roles"]
    for index in range(MAX_SETUPS_PER_OWNER):
        await directory.create(owner_id="someone", name=f"setup {index}", roles=roles)

    with pytest.raises(ModelSetupError, match="delete one"):
        await directory.create(owner_id="someone", name="one too many", roles=roles)
    # Another identity is not rationed by this one's collection.
    other = await directory.create(owner_id="someone-else", name="fresh", roles=roles)
    assert other.setup_id


# ------------------------------------------------------------------ T6: CUSTOM stays out


def test_custom_is_never_an_environment_backed_option() -> None:
    """T6: no loop that derives configuration from tier names ever sees `custom`."""
    assert PerformanceTier.CUSTOM not in PerformanceTier.env_backed()
    service = build_model_config_service(
        platforms={AgentPlatform.ANTHROPIC: _ANTHROPIC_HIGH},
    )
    assert all(tier is not PerformanceTier.CUSTOM for _, tier in service.configured_options())


@pytest.mark.asyncio
async def test_setup_never_publishes_a_custom_pairing() -> None:
    """`/setup` offers environment-backed pairings only; a setup is chosen elsewhere."""
    app = app_with()

    async with client(app) as http:
        payload = (await http.get("/setup", headers=AUTH)).json()

    assert all(item["performance_tier"] != "custom" for item in payload["agent_platforms"])


# ------------------------------------------------------------------ the predicate, directly


def test_the_predicate_reports_an_unconfigured_platform_as_a_named_warning() -> None:
    """Decision 12.7: allowed -- the setup supplies its own model -- but said out loud."""
    roles = parse_model_setup_roles(setup_payload()["roles"])

    warnings = validate_model_setup(
        roles,
        unsupported={},
        ceilings={},
        configured_platforms=(AgentPlatform.ANTHROPIC,),
    )

    assert any("coding" in item and "openai" in item for item in warnings)


def test_the_predicate_refuses_an_unknown_effort_and_an_unknown_platform_by_name() -> None:
    """A misspelling is a mistake to refuse, not a value to guess at."""
    payload = setup_payload()["roles"]
    payload["review"]["reasoning_effort"] = "extreme"
    with pytest.raises(ModelConfigurationError, match="must be one of"):
        validate_model_setup(parse_model_setup_roles(payload), unsupported={}, ceilings={})

    payload = setup_payload()["roles"]
    payload["coding"]["platform"] = "gemini"
    with pytest.raises(ModelConfigurationError, match="roles.coding.platform"):
        parse_model_setup_roles(payload)


def test_a_missing_anthropic_bound_is_refused_at_save_not_at_client_construction() -> None:
    """The Messages API rejects a request without max_tokens; the refusal happens here."""
    payload = setup_payload()["roles"]
    del payload["review"]["max_tokens"]
    with pytest.raises(ModelConfigurationError, match="roles.review.max_tokens is required"):
        validate_model_setup(parse_model_setup_roles(payload), unsupported={}, ceilings={})
