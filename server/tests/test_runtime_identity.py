"""Runtime identity and fail-closed request-boundary regression tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

import main
from main import create_app
from runtime_identity import (
    RuntimeIdentity,
    RuntimeIdentityError,
    load_runtime_identity,
)
from workflow_schema import WORKFLOW_SCHEMA_VERSION


class UnavailableInfrastructureProbe:
    """Report the one thing readiness is still allowed to fail on."""

    async def is_ready(self) -> bool:
        return False

    async def infrastructure_is_ready(self) -> bool:
        return False


def test_production_identity_rejects_unknown_and_mismatched_versions() -> None:
    """A controller cannot bless an unknown image or a different build/schema by accident."""
    unknown = load_runtime_identity(
        {
            "ENVIRONMENT": "production",
            "BUILD_REVISION": "unknown",
            "EXPECTED_BUILD_REVISION": "unknown",
            "EXPECTED_WORKFLOW_SCHEMA_VERSION": WORKFLOW_SCHEMA_VERSION,
        }
    )
    mismatched = load_runtime_identity(
        {
            "ENVIRONMENT": "production",
            "BUILD_REVISION": "a" * 40,
            "EXPECTED_BUILD_REVISION": "b" * 40,
            "EXPECTED_WORKFLOW_SCHEMA_VERSION": "obsolete",
        }
    )

    assert unknown.compatibility_errors == ("unknown_build_revision",)
    assert mismatched.compatibility_errors == (
        "build_revision_mismatch",
        "workflow_schema_version_mismatch",
    )
    with pytest.raises(RuntimeIdentityError, match="build_revision_mismatch"):
        mismatched.require_compatible()


@pytest.mark.asyncio
async def test_production_lifespan_refuses_an_incompatible_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup stops before connecting to infrastructure when deployment identity is stale."""
    monkeypatch.setattr(
        main,
        "load_settings",
        lambda: SimpleNamespace(
            build_revision="a" * 40,
            expected_build_revision="b" * 40,
            expected_workflow_schema_version=WORKFLOW_SCHEMA_VERSION,
            environment="production",
        ),
    )
    app = main.create_production_app()

    with pytest.raises(RuntimeIdentityError, match="build_revision_mismatch"):
        async with app.router.lifespan_context(app):
            pytest.fail("an incompatible runtime must not enter its serving lifespan")


@pytest.mark.asyncio
async def test_identity_is_observable_and_blocks_only_mutating_requests() -> None:
    """Stale workers remain diagnosable but cannot start, resume, cancel, or approve work."""
    identity = RuntimeIdentity(
        build_revision="a" * 40,
        workflow_schema_version=WORKFLOW_SCHEMA_VERSION,
        expected_build_revision="b" * 40,
        expected_workflow_schema_version=WORKFLOW_SCHEMA_VERSION,
        environment="production",
    )
    app = create_app(platform_api_key="identity-test-key", runtime_identity=identity)
    headers = {"Authorization": "Bearer identity-test-key"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        health = await client.get("/healthz")
        readiness = await client.get("/readyz")
        read_only = await client.get("/workflow/missing", headers=headers)
        mutation = await client.post("/workflow/start", headers=headers, json={})

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "build_revision": "a" * 40,
        "workflow_schema_version": WORKFLOW_SCHEMA_VERSION,
        "runtime_compatible": False,
    }
    assert readiness.status_code == 503
    assert readiness.json()["runtime_compatible"] is False
    assert read_only.status_code == 404
    assert mutation.status_code == 503
    assert mutation.json() == {"detail": "Runtime is not ready to process mutating requests."}


@pytest.mark.asyncio
async def test_unreachable_infrastructure_refuses_every_mutation_including_recovery() -> None:
    """With the unresolved-operation coupling gone, an unready runtime has no escape hatch.

    The middleware used to let `/workflow/resume` through an unready probe, because the only
    way to clear a global unresolved-operation outage was to make a credentialed request and
    readiness itself was blocking them. Readiness no longer knows about unresolved
    operations, so the remaining reason to be unready is infrastructure -- and a resume
    against an unreachable database is not something to wave through.
    """
    app = create_app(
        platform_api_key="identity-test-key",
        readiness_probe=UnavailableInfrastructureProbe(),
    )
    headers = {"Authorization": "Bearer identity-test-key"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        start = await client.post("/workflow/start", headers=headers, json={})
        resume = await client.post(
            "/workflow/resume",
            headers=headers,
            json={"workflow_id": "missing-workflow", "answers": []},
        )

    assert start.status_code == 503
    assert resume.status_code == 503
