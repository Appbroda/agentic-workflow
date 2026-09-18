"""The API serves the built web client from its own origin, when a build is present."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from main import create_app


@pytest.fixture
def built_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a minimal build in the shape Vite produces."""
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>App</title>", encoding="utf-8")
    (root / "assets" / "index-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (root / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    monkeypatch.setenv("WEB_CLIENT_ROOT", str(root))
    return root


@pytest.mark.asyncio
async def test_a_client_route_reloads_into_the_application_not_a_404(built_client: Path) -> None:
    """The client routes in the browser, so a reload of one of its pages must reach it."""
    del built_client
    app = create_app(platform_api_key="serving-key")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        bare = await client.get("/")
        home = await client.get("/ui/")
        deep = await client.get("/ui/features/some-feature/chat")
        asset = await client.get("/ui/assets/index-abc123.js")

    # Somebody who opens the bare origin is sent to the application.
    assert bare.status_code == 307
    assert bare.headers["location"] == "/ui/"
    assert home.status_code == 200
    # The page a person reloads is a client route, not a file on disk.
    assert deep.status_code == 200
    assert "<title>App</title>" in deep.text
    assert asset.status_code == 200
    assert "console.log(1)" in asset.text
    assert home.headers["x-content-type-options"] == "nosniff"
    assert home.headers["x-frame-options"] == "DENY"
    assert home.headers["referrer-policy"] == "no-referrer"
    assert "default-src 'self'" in home.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in home.headers["content-security-policy"]
    # Design previews are `URL.createObjectURL` blobs -- the bytes come through the API with a
    # bearer token an `<img src>` cannot carry -- so `img-src` must allow `blob:`. jsdom does
    # not enforce CSP, which makes this served header the only place the directive is provable.
    assert "img-src 'self' data: blob:" in home.headers["content-security-policy"]


@pytest.mark.asyncio
async def test_serving_the_client_never_shadows_the_api_or_the_health_checks(
    built_client: Path,
) -> None:
    """A catch-all registered before the routes would swallow the platform."""
    del built_client
    app = create_app(platform_api_key="serving-key")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        health = await client.get("/healthz")
        prefixed_health = await client.get("/api/readyz")
        api = await client.get("/api/features")
        unauthenticated = await client.get("/features")

    assert health.status_code == 200
    assert health.json()["workflow_schema_version"]
    # The client's base URL is `/api`, so health has to answer there too. It did not, and the
    # settings page reported the running platform as "no longer exists".
    assert prefixed_health.status_code == 200
    assert prefixed_health.json()["workflow_schema_version"]
    # Still the API's, and still authenticated -- not the application shell.
    assert api.status_code == 401
    assert unauthenticated.status_code == 401


@pytest.mark.asyncio
async def test_a_traversing_path_cannot_read_outside_the_build(built_client: Path) -> None:
    """`..` in a URL must not reach the server's filesystem."""
    secret = built_client.parent / "secret.txt"
    secret.write_text("PLATFORM_API_KEY=hunter2", encoding="utf-8")
    app = create_app(platform_api_key="serving-key")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        escaped = await client.get("/ui/..%2fsecret.txt")

    # Refused by falling back to the shell rather than by reading the file.
    assert "hunter2" not in escaped.text


@pytest.mark.asyncio
async def test_the_api_runs_alone_when_no_build_is_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A deployment that ships only the API must start and serve it."""
    monkeypatch.setenv("WEB_CLIENT_ROOT", str(tmp_path / "absent"))
    app = create_app(platform_api_key="serving-key")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        health = await client.get("/healthz")
        missing = await client.get("/ui/")

    assert health.status_code == 200
    # No application to serve, and the API does not pretend otherwise.
    assert missing.status_code == 404
