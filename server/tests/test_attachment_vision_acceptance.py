"""The refusal a submission with images gets when nothing will read them (89-, Part E5).

Refused at acceptance rather than accepted and answered from the prose. That is the stronger
form of 61-'s rule: a drop is a drop, and this drop would leave a technical PRD derived from a
description of a picture with nobody aware the picture was ignored.

Two sitings, because the platform resolves models in two places. A custom setup already has a
resolved selection at POST time; a (platform, tier) pairing had none, and this item builds one.
"""

from __future__ import annotations

import json
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
from configs.settings import CONFIG_DIRECTORY, load_settings
from main import create_app
from storage.attachment_store import InMemoryAttachmentStore
from tests import attachment_fixtures as fixtures
from tests.test_performance_tiers import _base_model_overrides, _cleared_tier_fields

KEY = "vision-acceptance-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
OWNER = "platform-admin"

_ROLES = {
    ModelRole.REASONING: "gpt-6-astra",
    ModelRole.CODING: "gpt-6-astra",
    ModelRole.REVIEW: "gpt-6-astra",
    ModelRole.SCOPED_FIX: "gpt-6-astra",
}


class _Declarations:
    """The declarations the acceptance checks read, and nothing else.

    Duck-typed exactly as `test_model_setups` does it: the route asks settings four
    questions, and a test that had to build a whole `Settings` to answer one of them would
    be a test about configuration loading.
    """

    def __init__(self, *, vision: list[str] | None = None) -> None:
        """Declare which models this deployment says can read an image."""
        self._vision = vision or []
        self._configs = build_model_config_service(
            platforms={
                AgentPlatform.OPENAI: PlatformModelSettings(
                    models=dict(_ROLES),
                    reasoning=dict.fromkeys(_ROLES, "high"),
                    max_tokens=dict.fromkeys(_ROLES, 64_000),
                )
            }
        )

    def declared_unsupported_reasoning(self) -> dict[str, list[str]]:
        """No pairing is unsupported on this deployment."""
        return {}

    def declared_max_output_tokens(self) -> dict[str, int]:
        """No ceiling is known on this deployment."""
        return {}

    def declared_vision_capable(self) -> frozenset[str]:
        """Which models this deployment says can be shown an image."""
        return frozenset(self._vision)

    def for_performance_tier(self, tier: PerformanceTier) -> Any:
        """Return this configuration viewed through one tier, shaped as the real clone is.

        `Settings.for_performance_tier` returns a view whose `ModelConfigService` is
        *already* tier-scoped -- asking that view for a tier again raises "not configured".
        The fake keeps that shape rather than returning `self` for every tier, so a caller
        that double-applies the tier fails here exactly as it fails on a real deployment.
        """
        if tier is PerformanceTier.HIGH:
            return self
        view = _Declarations(vision=list(self._vision))
        view._configs = self._configs.for_tier(tier)
        return view

    @property
    def model_configs(self) -> Any:
        """The role resolution the tier check reads the reasoning model out of."""
        return self._configs


def app_with(*, vision: list[str] | None = None) -> Any:
    """An application whose vision declaration is the one under test."""
    app = create_app(platform_api_key=KEY, attachments=InMemoryAttachmentStore())
    app.state.settings = _Declarations(vision=vision)
    return app


def client(app: Any) -> AsyncClient:
    """Return an HTTP client bound to the application under test."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def uploaded(app: Any) -> str:
    """Put one committed fixture in the application's store and return its id."""
    record = await app.state.attachments.create(
        owner_id=OWNER,
        filename="login.png",
        media_type="image/png",
        content=fixtures.content(fixtures.PNG),
    )
    return str(record.attachment_id)


def payload(attachment_id: str | None, *, execution_mode: str = "live", **overrides: Any) -> Any:
    """One submission, with or without an image reference."""
    body: dict[str, Any] = {
        "feature_id": "feature-login",
        "prd": {
            "title": "Login audit trail",
            "problem_statement": "Login fails with no explanation [image:login-error].",
            "attachments": (
                [{"attachment_id": attachment_id, "marker": "login-error"}] if attachment_id else []
            ),
        },
        "repositories": [
            {
                "repository_url": "https://github.com/example/backend",
                "default_branch": "main",
                "required": True,
            }
        ],
        "execution_mode": execution_mode,
        "agent_platform": "openai",
        "performance_tier": "high",
    }
    if not attachment_id:
        body["prd"]["problem_statement"] = "Login fails with no explanation."
    body.update(overrides)
    return body


@pytest.mark.asyncio
async def test_an_undeclared_tier_model_refuses_the_submission_and_names_it() -> None:
    """The tier resolution this item had to build: no model resolution existed at POST time."""
    app = app_with(vision=[])
    attachment_id = await uploaded(app)

    async with client(app) as http:
        refused = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "e1"},
            json=payload(attachment_id),
        )
        listed = await http.get("/features", headers=AUTH)

    assert refused.status_code == 422
    detail = refused.json()["detail"]
    # The model is named, because "your model cannot read images" sends nobody anywhere.
    assert "gpt-6-astra" in detail
    assert "openai at the high tier" in detail
    # And both ways out are stated.
    assert "Remove the images" in detail
    assert "reasoning model reads them" in detail
    # Nothing accepted and nothing bound.
    assert listed.json()["features"] == []
    record = await app.state.attachments.get_metadata(attachment_id)
    assert record is not None
    assert record.feature_id is None


@pytest.mark.asyncio
async def test_a_declared_tier_model_accepts_the_same_submission() -> None:
    """The declaration is the only difference between the two outcomes."""
    app = app_with(vision=["gpt-6-astra"])
    attachment_id = await uploaded(app)

    async with client(app) as http:
        accepted = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "e2"},
            json=payload(attachment_id),
        )

    assert accepted.status_code == 201, accepted.text
    record = await app.state.attachments.get_metadata(attachment_id)
    assert record is not None
    assert record.feature_id == "feature-login"


def app_with_real_settings(*, vision: list[str]) -> Any:
    """An application whose settings are a real `Settings` with a configured low tier.

    The duck-typed `_Declarations` answers the route's four questions but cannot prove the
    tier resolution against the object a deployment actually runs. This one can: the low
    tier's reasoning model is `gpt-5.6-terra`, resolved through the genuine
    `for_performance_tier` clone -- the path that once double-applied the tier and refused
    every non-high submission as "not configured".
    """
    low = {
        "openai_low_reasoning_model": "gpt-5.6-terra",
        "openai_low_reasoning_effort": "medium",
        "openai_low_coding_model": "gpt-5.3-codex",
        "openai_low_coding_reasoning_effort": "high",
        "openai_low_review_model": "gpt-5.6-terra",
        "openai_low_review_reasoning_effort": "medium",
        "openai_low_scoped_fix_model": "gpt-5.3-codex",
        "openai_low_scoped_fix_reasoning_effort": "medium",
    }
    settings = load_settings(
        CONFIG_DIRECTORY,
        **_base_model_overrides(),
        **_cleared_tier_fields(*low),
        **low,
        model_vision_capable=json.dumps(vision),
    )
    app = create_app(platform_api_key=KEY, attachments=InMemoryAttachmentStore())
    app.state.settings = settings
    return app


@pytest.mark.asyncio
async def test_a_low_tier_submission_is_accepted_through_a_real_settings_clone() -> None:
    """The tier is applied once: the clone is already scoped, and the reader must not re-scope.

    Regression for the double application: `for_performance_tier(low)` followed by
    `get_model_config(..., tier=low)` asked the already-scoped view for a tier it does not
    hold, and every non-high submission with images 422ed as "not configured".
    """
    app = app_with_real_settings(vision=["gpt-5.6-terra"])
    attachment_id = await uploaded(app)

    async with client(app) as http:
        accepted = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "e-low-1"},
            json=payload(attachment_id, performance_tier="low"),
        )

    assert accepted.status_code == 201, accepted.text
    record = await app.state.attachments.get_metadata(attachment_id)
    assert record is not None
    assert record.feature_id == "feature-login"


@pytest.mark.asyncio
async def test_an_undeclared_low_tier_model_is_refused_naming_the_low_tiers_own_model() -> None:
    """The refusal judges and names the tier's model, not the high tier's."""
    # The high tier's reasoning model is declared; the low tier's is not. A resolution that
    # leaked the unsuffixed configuration would accept this submission.
    app = app_with_real_settings(vision=["gpt-5.6-sol"])
    attachment_id = await uploaded(app)

    async with client(app) as http:
        refused = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "e-low-2"},
            json=payload(attachment_id, performance_tier="low"),
        )

    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert "gpt-5.6-terra" in detail
    assert "openai at the low tier" in detail


@pytest.mark.asyncio
async def test_a_submission_with_no_images_is_unaffected_by_the_declaration() -> None:
    """The check is about images. A deployment declaring nothing still runs every feature."""
    app = app_with(vision=[])

    async with client(app) as http:
        accepted = await http.post(
            "/features/start", headers={**AUTH, "Idempotency-Key": "e3"}, json=payload(None)
        )

    assert accepted.status_code == 201, accepted.text


@pytest.mark.asyncio
async def test_mock_mode_accepts_images_without_asking_the_capability_question() -> None:
    """It selects no model and reaches no provider, so there is nothing to be wrong about."""
    app = app_with(vision=[])
    attachment_id = await uploaded(app)

    async with client(app) as http:
        accepted = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "e4"},
            json=payload(attachment_id, execution_mode="mock"),
        )

    assert accepted.status_code == 201, accepted.text


@pytest.mark.asyncio
async def test_a_custom_setup_is_judged_on_its_own_reasoning_role() -> None:
    """The setup path reads the snapshot the route already took and validated."""
    app = app_with(vision=["gpt-6-astra"])
    attachment_id = await uploaded(app)
    setup = await app.state.model_setups.create(
        owner_id=OWNER,
        name="Blind reasoning",
        roles={
            # A model nothing declared, on the one role that reads the PRD.
            "reasoning": {"platform": "openai", "model": "gpt-5.6-sol"},
            "coding": {"platform": "openai", "model": "gpt-6-astra"},
            "review": {"platform": "openai", "model": "gpt-6-astra"},
            "scoped_fix": {"platform": "openai", "model": "gpt-6-astra"},
        },
    )

    async with client(app) as http:
        refused = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "e5"},
            # A setup id alone: the server refuses two answers to one question, so the
            # pairing keys are omitted rather than nulled.
            json={
                key: value
                for key, value in {
                    **payload(attachment_id),
                    "model_setup_id": setup.setup_id,
                }.items()
                if key not in ("agent_platform", "performance_tier")
            },
        )

    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert "gpt-5.6-sol" in detail
    assert "Blind reasoning" in detail


@pytest.mark.asyncio
async def test_a_replay_is_not_re_judged_against_a_declaration_that_has_since_changed() -> None:
    """A replay asks for an answer already given; re-running the check would revoke it."""
    app = app_with(vision=["gpt-6-astra"])
    attachment_id = await uploaded(app)
    body = payload(attachment_id)

    async with client(app) as http:
        created = await http.post(
            "/features/start", headers={**AUTH, "Idempotency-Key": "e6"}, json=body
        )
        # The deployment stops declaring the model between the two identical requests.
        app.state.settings = _Declarations(vision=[])
        replay = await http.post(
            "/features/start", headers={**AUTH, "Idempotency-Key": "e6"}, json=body
        )

    assert created.status_code == 201, created.text
    assert replay.status_code == 200
    assert replay.json()["created"] is False


# ----------------------------------------------- Part G's server half: the published answer


@pytest.mark.asyncio
async def test_the_setup_state_publishes_which_pairings_read_images() -> None:
    """The form needs the answer before it can say it at the selector.

    Published rather than derived in the browser, for the reason `configured` is: the client
    cannot know a deployment's declarations, and the alternative is a form that offers a
    selection the start endpoint will refuse.
    """
    app = app_with(vision=["gpt-6-astra"])
    app.state.model_configs = _Declarations(vision=["gpt-6-astra"]).model_configs

    async with client(app) as http:
        state = await http.get("/setup", headers=AUTH)

    rows = {
        (item["platform"], item["performance_tier"]): item
        for item in state.json()["agent_platforms"]
    }
    openai_high = rows[("openai", "high")]
    assert openai_high["configured"] is True
    assert openai_high["models"]["reasoning"] == "gpt-6-astra"
    assert openai_high["vision_capable"] is True
    # An unconfigured pairing resolves no model, so it answers the fail-closed default rather
    # than a guess about a model nobody selected.
    assert rows[("anthropic", "high")]["configured"] is False
    assert rows[("anthropic", "high")]["vision_capable"] is False


@pytest.mark.asyncio
async def test_a_deployment_declaring_nothing_publishes_false_for_every_pairing() -> None:
    """The same answer the start endpoint would give the same submission."""
    app = app_with(vision=[])
    app.state.model_configs = _Declarations(vision=[]).model_configs

    async with client(app) as http:
        state = await http.get("/setup", headers=AUTH)

    assert all(not item["vision_capable"] for item in state.json()["agent_platforms"])


@pytest.mark.asyncio
async def test_a_saved_setup_publishes_whether_its_reasoning_role_reads_images() -> None:
    """Only the reasoning role: it is the only call a submission's images travel on."""
    app = app_with(vision=["gpt-6-astra"])
    blind = await app.state.model_setups.create(
        owner_id=OWNER,
        name="Blind reasoning",
        roles={
            "reasoning": {"platform": "openai", "model": "gpt-5.6-sol"},
            "coding": {"platform": "openai", "model": "gpt-6-astra"},
            "review": {"platform": "openai", "model": "gpt-6-astra"},
            "scoped_fix": {"platform": "openai", "model": "gpt-6-astra"},
        },
    )
    seeing = await app.state.model_setups.create(
        owner_id=OWNER,
        name="Seeing reasoning",
        roles={
            "reasoning": {"platform": "openai", "model": "gpt-6-astra"},
            # A coding model nothing declared, which must not change the answer: images do
            # not travel on the coding call.
            "coding": {"platform": "openai", "model": "gpt-5.6-sol"},
            "review": {"platform": "openai", "model": "gpt-6-astra"},
            "scoped_fix": {"platform": "openai", "model": "gpt-6-astra"},
        },
    )

    async with client(app) as http:
        listed = await http.get("/model-setups", headers=AUTH)

    rows = {item["setup_id"]: item for item in listed.json()["setups"]}
    assert rows[blind.setup_id]["vision_capable"] is False
    assert rows[seeing.setup_id]["vision_capable"] is True
