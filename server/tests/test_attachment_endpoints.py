"""Uploading, fetching and deleting an attachment over HTTP (item 89-, Part B).

The refusals get a case each, because each one is a different sentence somebody reads: the
per-file cap, a type the platform does not serve, an SVG (which has its own way out), and a
file whose name and contents disagree. The security headers are asserted on the response and
not on the handler -- a header set in code and stripped by a middleware is a header nobody
receives.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from api.identity import Role
from main import create_app
from storage.attachment_store import (
    MAX_ATTACHMENT_BYTES,
    InMemoryAttachmentStore,
)
from storage.user_store import InMemoryUserDirectory
from tests import attachment_fixtures as fixtures

KEY = "attachments-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


def app_with_attachments(**overrides: Any) -> Any:
    """An application with a working attachment store, like a deployment has."""
    return create_app(platform_api_key=KEY, attachments=InMemoryAttachmentStore(), **overrides)


def client(app: Any) -> AsyncClient:
    """Return an HTTP client bound to the application under test."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def token_for(directory: InMemoryUserDirectory, *, roles: tuple[str, ...]) -> str:
    """Register somebody with the given roles and return a usable token for them."""
    user = await directory.create_user(
        subject=f"person-{roles[0]}-{len(await directory.list_users())}",
        display_name=f"A {roles[0]}",
        roles=roles,
    )
    issued = await directory.issue_token(user.user_id, label="test")
    return issued.token


def upload(
    path: Any, *, name: str | None = None, content_type: str | None = None
) -> dict[str, Any]:
    """One multipart body carrying one committed fixture."""
    return {
        "files": {
            "file": (
                name or path.name,
                fixtures.content(path),
                content_type if content_type is not None else "application/octet-stream",
            )
        }
    }


@pytest.mark.parametrize(("path", "media_type"), fixtures.ACCEPTED)
@pytest.mark.asyncio
async def test_each_accepted_type_uploads_and_serves_back_its_own_bytes(
    path: Any, media_type: str
) -> None:
    """Round trip over HTTP, with the sniffed type on the way out."""
    app = app_with_attachments()

    async with client(app) as http:
        created = await http.post("/attachments", headers=AUTH, **upload(path))
        body = created.json()
        fetched = await http.get(f"/attachments/{body['attachment_id']}", headers=AUTH)

    assert created.status_code == 201
    assert body["media_type"] == media_type
    assert body["sha256"] == fixtures.digest(path)
    assert body["byte_size"] == len(fixtures.content(path))
    # No URL: the client fetches by id with its own token, and a link a browser could follow
    # unauthenticated is exactly what this response must not contain.
    assert "url" not in body
    assert fetched.status_code == 200
    assert fetched.content == fixtures.content(path)
    assert fetched.headers["content-type"] == media_type


@pytest.mark.asyncio
async def test_the_security_headers_are_on_the_response() -> None:
    """Asserted where a browser would read them, not where they were written."""
    app = app_with_attachments()

    async with client(app) as http:
        created = await http.post("/attachments", headers=AUTH, **upload(fixtures.PNG))
        fetched = await http.get(f"/attachments/{created.json()['attachment_id']}", headers=AUTH)

    assert fetched.headers["x-content-type-options"] == "nosniff"
    assert fetched.headers["content-security-policy"] == "default-src 'none'; sandbox"
    assert fetched.headers["cache-control"] == "private, max-age=3600"


@pytest.mark.asyncio
async def test_a_png_named_jpg_is_accepted_as_png_and_served_as_png() -> None:
    """The bytes decide. A name is a label somebody typed."""
    app = app_with_attachments()

    async with client(app) as http:
        created = await http.post(
            "/attachments", headers=AUTH, **upload(fixtures.PNG, name="screenshot.jpg")
        )
        body = created.json()
        fetched = await http.get(f"/attachments/{body['attachment_id']}", headers=AUTH)

    assert created.status_code == 201
    assert body["media_type"] == "image/png"
    # The name is kept as given, for the person; it decides nothing.
    assert body["filename"] == "screenshot.jpg"
    assert fetched.headers["content-type"] == "image/png"


@pytest.mark.asyncio
async def test_a_declared_image_type_that_disagrees_with_the_bytes_is_refused() -> None:
    """A declared type is a claim, and one that contradicts the contents is a lie."""
    app = app_with_attachments()

    async with client(app) as http:
        refused = await http.post(
            "/attachments",
            headers=AUTH,
            **upload(fixtures.PNG, name="a.jpg", content_type="image/jpeg"),
        )

    assert refused.status_code == 422
    assert "image/jpeg" in refused.json()["detail"]
    assert "image/png" in refused.json()["detail"]


@pytest.mark.asyncio
async def test_an_svg_is_refused_with_its_own_message() -> None:
    """ "Unsupported type" would send somebody away to export another SVG."""
    app = app_with_attachments()

    async with client(app) as http:
        by_name = await http.post("/attachments", headers=AUTH, **upload(fixtures.SVG))
        renamed = await http.post(
            "/attachments", headers=AUTH, **upload(fixtures.SVG, name="mock.png")
        )

    for refused in (by_name, renamed):
        assert refused.status_code == 422
        detail = refused.json()["detail"]
        assert "SVG is markup, not an image; export a PNG" in detail


@pytest.mark.asyncio
async def test_a_png_whose_bytes_are_a_zip_is_refused() -> None:
    """Sniffed, not believed. The extension is the only thing that said "image" here."""
    app = app_with_attachments()

    async with client(app) as http:
        refused = await http.post("/attachments", headers=AUTH, **upload(fixtures.ZIP_NAMED_PNG))

    assert refused.status_code == 422
    assert "not an image the platform accepts" in refused.json()["detail"]


@pytest.mark.asyncio
async def test_an_oversized_upload_is_refused_before_it_is_stored() -> None:
    """The cap quoted in the message, and nothing kept."""
    app = app_with_attachments()
    # A real PNG header followed by padding, so the refusal is the size rule and not the
    # sniffer: this is the case where the bytes are fine and there are too many of them.
    oversized = fixtures.content(fixtures.PNG) + b"\x00" * (MAX_ATTACHMENT_BYTES + 1)

    async with client(app) as http:
        refused = await http.post(
            "/attachments",
            headers=AUTH,
            files={"file": ("huge.png", oversized, "application/octet-stream")},
        )

    assert refused.status_code == 413
    assert "5 MiB" in refused.json()["detail"]
    assert await app.state.attachments.list_for_owner("platform-admin") == []


@pytest.mark.asyncio
async def test_an_empty_upload_is_refused() -> None:
    """Zero bytes is not an image, and the message says so rather than naming a type."""
    app = app_with_attachments()

    async with client(app) as http:
        refused = await http.post(
            "/attachments",
            headers=AUTH,
            files={"file": ("empty.png", b"", "application/octet-stream")},
        )

    assert refused.status_code == 422
    assert "no bytes" in refused.json()["detail"]


@pytest.mark.asyncio
async def test_an_unauthenticated_fetch_is_401_and_a_wrong_owner_fetch_is_404() -> None:
    """Not 403: an id somebody guessed must not be confirmed to exist."""
    directory = InMemoryUserDirectory()
    app = app_with_attachments(user_directory=directory)
    mine = await token_for(directory, roles=(Role.OPERATOR.value,))
    theirs = await token_for(directory, roles=(Role.OPERATOR.value,))

    async with client(app) as http:
        created = await http.post(
            "/attachments", headers={"Authorization": f"Bearer {mine}"}, **upload(fixtures.PNG)
        )
        attachment_id = created.json()["attachment_id"]
        anonymous = await http.get(f"/attachments/{attachment_id}")
        somebody_else = await http.get(
            f"/attachments/{attachment_id}", headers={"Authorization": f"Bearer {theirs}"}
        )
        owner = await http.get(
            f"/attachments/{attachment_id}", headers={"Authorization": f"Bearer {mine}"}
        )

    assert anonymous.status_code == 401
    assert somebody_else.status_code == 404
    assert owner.status_code == 200


@pytest.mark.asyncio
async def test_a_bound_attachment_is_readable_by_anybody_who_may_read_the_feature() -> None:
    """Once submitted it is the feature's evidence, on the permission that shows the PRD."""
    directory = InMemoryUserDirectory()
    app = app_with_attachments(user_directory=directory)
    mine = await token_for(directory, roles=(Role.OPERATOR.value,))
    viewer = await token_for(directory, roles=(Role.VIEWER.value,))

    async with client(app) as http:
        created = await http.post(
            "/attachments", headers={"Authorization": f"Bearer {mine}"}, **upload(fixtures.PNG)
        )
        attachment_id = created.json()["attachment_id"]
        before = await http.get(
            f"/attachments/{attachment_id}", headers={"Authorization": f"Bearer {viewer}"}
        )
        await app.state.attachments.bind([attachment_id], feature_id="feature-1")
        after = await http.get(
            f"/attachments/{attachment_id}", headers={"Authorization": f"Bearer {viewer}"}
        )

    # Unbound: nobody's but the uploader's, and not an existence oracle for anybody else.
    assert before.status_code == 404
    # Bound: a viewer already sees this feature's PRD, so they see what it points at.
    assert after.status_code == 200


@pytest.mark.asyncio
async def test_a_purged_attachment_answers_410_and_not_404() -> None:
    """It existed. Saying so is the difference between "gone" and "never was"."""
    app = app_with_attachments()

    async with client(app) as http:
        created = await http.post("/attachments", headers=AUTH, **upload(fixtures.PNG))
        attachment_id = created.json()["attachment_id"]
        await app.state.attachments.bind([attachment_id], feature_id="feature-1")
        await app.state.attachments.purge_content_for_feature("feature-1")
        gone = await http.get(f"/attachments/{attachment_id}", headers=AUTH)

    assert gone.status_code == 410
    assert "purged" in gone.json()["detail"]


@pytest.mark.asyncio
async def test_delete_removes_an_unbound_upload_and_refuses_a_bound_one() -> None:
    """A bound attachment is part of a submitted record; retiring the feature removes it."""
    app = app_with_attachments()

    async with client(app) as http:
        loose = (await http.post("/attachments", headers=AUTH, **upload(fixtures.PNG))).json()
        bound = (await http.post("/attachments", headers=AUTH, **upload(fixtures.JPEG))).json()
        await app.state.attachments.bind([bound["attachment_id"]], feature_id="feature-1")

        removed = await http.delete(f"/attachments/{loose['attachment_id']}", headers=AUTH)
        again = await http.delete(f"/attachments/{loose['attachment_id']}", headers=AUTH)
        refused = await http.delete(f"/attachments/{bound['attachment_id']}", headers=AUTH)

    assert removed.status_code == 204
    assert again.status_code == 404
    assert refused.status_code == 404


@pytest.mark.asyncio
async def test_uploading_needs_the_permission_that_creates_features() -> None:
    """An upload is the first half of a submission, and is guarded as one."""
    directory = InMemoryUserDirectory()
    app = app_with_attachments(user_directory=directory)
    viewer = await token_for(directory, roles=(Role.VIEWER.value,))

    async with client(app) as http:
        refused = await http.post(
            "/attachments",
            headers={"Authorization": f"Bearer {viewer}"},
            **upload(fixtures.PNG),
        )

    assert refused.status_code == 403


@pytest.mark.asyncio
async def test_a_deployment_with_no_attachment_store_says_so() -> None:
    """503 rather than a fault: "this deployment does not do that" is not an error."""
    app = create_app(platform_api_key=KEY)

    async with client(app) as http:
        refused = await http.post("/attachments", headers=AUTH, **upload(fixtures.PNG))

    assert refused.status_code == 503


@pytest.mark.asyncio
async def test_the_published_limits_are_the_constants_the_server_enforces() -> None:
    """The interface's helper text reads these, so they cannot drift from the enforcement."""
    app = app_with_attachments()

    async with client(app) as http:
        limits = await http.get("/attachments/limits", headers=AUTH)

    body = limits.json()
    assert body["max_attachment_bytes"] == MAX_ATTACHMENT_BYTES
    assert body["max_attachments_per_feature"] == 8
    assert body["max_attachment_bytes_per_feature"] == 20 * 1024 * 1024
    assert body["accepted_media_types"] == ["image/png", "image/jpeg", "image/webp"]


@pytest.mark.asyncio
async def test_the_upload_route_is_exempt_from_the_json_body_cap() -> None:
    """The one route whose body is legitimately larger than any JSON request.

    Without this the platform's own 1 MiB request cap answers 413 before the handler is
    reached, and a 5 MiB image is refused by a rule that exists to bound JSON payloads.
    """
    app = app_with_attachments()
    # Comfortably over the JSON cap and comfortably under the attachment cap.
    payload = fixtures.content(fixtures.PNG) + b"\x00" * (2 * 1024 * 1024)

    async with client(app) as http:
        created = await http.post(
            "/attachments",
            headers=AUTH,
            files={"file": ("big.png", payload, "application/octet-stream")},
        )
        # And the cap still applies to everything else on the same application.
        oversized_json = await http.post(
            "/features/start",
            headers={**AUTH, "Content-Length": str(2 * 1024 * 1024)},
            content=b"{}" + b" " * (2 * 1024 * 1024),
        )

    assert created.status_code == 201
    assert created.json()["byte_size"] == len(payload)
    assert oversized_json.status_code == 413
