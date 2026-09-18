"""Tests for the foundational API scaffold."""

from httpx import ASGITransport, AsyncClient

from main import app


async def test_healthz_reports_liveness() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    # Reported so a running container can be matched to the build it was made from.
    assert payload["build_revision"] == "local"


async def test_readyz_reports_readiness() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/readyz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    # Reported so a running container can be matched to the build it was made from.
    assert payload["build_revision"] == "local"
