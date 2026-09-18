"""A submission that carries image references, and every way one can be wrong (89-, Part C).

The rule these cases circle is a single one: an image the model will not see must not be
silently dropped. So a marker the prose names and no attachment declares is a rejection, an
id that is not the submitter's is a rejection, and a re-used id is a rejection -- while an
attachment nothing references is *accepted*, because somebody who attached three screenshots
has shown the platform three screenshots whether or not they wrote three markers.

The last case is the one worth keeping honest: when acceptance fails for any other reason,
nothing is bound and the images are still uploaded and still usable.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from api.identity import Role
from main import create_app
from storage.attachment_store import InMemoryAttachmentStore
from storage.user_store import InMemoryUserDirectory
from tests import attachment_fixtures as fixtures
from tests.support import settle

KEY = "attachment-submission-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
OWNER = "platform-admin"


def app_with_attachments(**overrides: Any) -> Any:
    """An application with a working attachment store, like a deployment has."""
    return create_app(platform_api_key=KEY, attachments=InMemoryAttachmentStore(), **overrides)


def client(app: Any) -> AsyncClient:
    """Return an HTTP client bound to the application under test."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def uploaded(
    store: InMemoryAttachmentStore, *, owner_id: str = OWNER, name: str = "login.png"
) -> str:
    """Put one committed fixture in the store and return its id."""
    record = await store.create(
        owner_id=owner_id,
        filename=name,
        media_type="image/png",
        content=fixtures.content(fixtures.PNG),
    )
    return record.attachment_id


def payload(
    *,
    attachments: list[dict[str, Any]] | None = None,
    problem_statement: str = "Support staff need to trace failed login attempts.",
    feature_id: str = "feature-login",
) -> dict[str, Any]:
    """One parent submission, with whatever image references the case is about."""
    return {
        "feature_id": feature_id,
        "prd": {
            "title": "Login audit trail",
            "problem_statement": problem_statement,
            "goals": ["Record failed attempts."],
            "attachments": attachments or [],
        },
        "repositories": [
            {
                "repository_url": "https://github.com/example/backend",
                "default_branch": "main",
                "required": True,
            }
        ],
        "execution_mode": "mock",
    }


async def prd_payload(http: AsyncClient, feature_id: str) -> dict[str, Any]:
    """Read back the submitted PRD artifact as the artifacts API returns it."""
    listed = await http.get(
        f"/features/{feature_id}/artifacts", headers=AUTH, params={"artifact_type": "prd"}
    )
    artifact_id = listed.json()["artifacts"][-1]["artifact_id"]
    fetched = await http.get(f"/features/{feature_id}/artifacts/{artifact_id}", headers=AUTH)
    body: dict[str, Any] = fetched.json()["payload"]
    return body


@pytest.mark.asyncio
async def test_two_images_with_one_marker_each_are_accepted_and_bound() -> None:
    """The ordinary case: two screenshots, two references, one submitted record."""
    app = app_with_attachments()
    store = app.state.attachments
    first = await uploaded(store, name="login-error.png")
    second = await uploaded(store, name="empty-state.png")

    async with client(app) as http:
        created = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "c1"},
            json=payload(
                problem_statement=(
                    "Login fails with no explanation [image:login-error] and the list "
                    "behind it is blank [image:empty-state]."
                ),
                attachments=[
                    {"attachment_id": first, "marker": "login-error", "caption": "The error"},
                    {"attachment_id": second, "marker": "empty-state", "caption": ""},
                ],
            ),
        )
        await settle(app)
        prd = await prd_payload(http, "feature-login")

    assert created.status_code == 201, created.text
    # Bound to this feature, inside the transaction that wrote it.
    assert [item.feature_id for item in await store.list_for_feature("feature-login")] == [
        "feature-login",
        "feature-login",
    ]
    # The artifact records what was attached: the marker and caption the submitter chose, and
    # the filename, type, size and hash the store resolved.
    recorded = prd["attachments"]
    assert [item["marker"] for item in recorded] == ["login-error", "empty-state"]
    assert [item["filename"] for item in recorded] == ["login-error.png", "empty-state.png"]
    assert {item["media_type"] for item in recorded} == {"image/png"}
    assert {item["sha256"] for item in recorded} == {fixtures.digest(fixtures.PNG)}
    assert recorded[0]["caption"] == "The error"
    # And no bytes, ever.
    assert all("data" not in item and "content" not in item for item in recorded)


@pytest.mark.asyncio
async def test_a_marker_the_prose_names_and_nothing_declares_is_refused() -> None:
    """A dangling pointer, refused by name. Accepting it renders four literal words."""
    app = app_with_attachments()
    first = await uploaded(app.state.attachments)

    async with client(app) as http:
        refused = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "c2"},
            json=payload(
                problem_statement="The error looks like this [image:login-error].",
                attachments=[{"attachment_id": first, "marker": "something-else"}],
            ),
        )
        listed = await http.get("/features", headers=AUTH)

    assert refused.status_code == 422
    assert "[image:login-error]" in refused.text
    # Refused before acceptance: nothing exists and nothing is bound.
    assert listed.json()["features"] == []
    record = await app.state.attachments.get_metadata(first)
    assert record is not None
    assert record.feature_id is None


@pytest.mark.asyncio
async def test_a_marker_referenced_from_a_requirement_is_found_too() -> None:
    """A reference is allowed anywhere somebody writes, not only in the problem statement."""
    app = app_with_attachments()
    first = await uploaded(app.state.attachments)
    body = payload(attachments=[{"attachment_id": first, "marker": "shown"}])
    body["prd"]["requirements"] = [
        {
            "requirement_id": "r1",
            "description": "Match the layout in [image:missing].",
            "priority": "must",
            "acceptance_criteria": ["The layout matches."],
            "dependencies": [],
        }
    ]

    async with client(app) as http:
        refused = await http.post(
            "/features/start", headers={**AUTH, "Idempotency-Key": "c3"}, json=body
        )

    assert refused.status_code == 422
    assert "[image:missing]" in refused.text


@pytest.mark.asyncio
async def test_an_attachment_no_prose_references_is_accepted_and_still_reaches_the_model() -> None:
    """Somebody who attached three screenshots has shown the platform three screenshots."""
    app = app_with_attachments()
    store = app.state.attachments
    quiet = await uploaded(store, name="quiet.png")
    named = await uploaded(store, name="named.png")

    async with client(app) as http:
        created = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "c4"},
            json=payload(
                problem_statement="Only one of these is referenced [image:named].",
                attachments=[
                    {"attachment_id": quiet, "marker": "quiet"},
                    {"attachment_id": named, "marker": "named"},
                ],
            ),
        )
        await settle(app)
        prd = await prd_payload(http, "feature-login")

    assert created.status_code == 201, created.text
    # Referenced first, in the order the prose references them; unreferenced afterwards, in
    # declaration order. The order is the order the model is shown the images.
    assert [item["marker"] for item in prd["attachments"]] == ["named", "quiet"]


@pytest.mark.asyncio
async def test_an_attachment_owned_by_somebody_else_is_refused() -> None:
    """An id is not an authorisation, and the refusal does not say whose it is."""
    directory = InMemoryUserDirectory()
    app = app_with_attachments(user_directory=directory)
    theirs = await uploaded(app.state.attachments, owner_id="somebody-else")
    user = await directory.create_user(
        subject="submitter", display_name="Submitter", roles=(Role.OPERATOR.value,)
    )
    token = (await directory.issue_token(user.user_id, label="test")).token

    async with client(app) as http:
        refused = await http.post(
            "/features/start",
            headers={"Authorization": f"Bearer {token}", "Idempotency-Key": "c5"},
            json=payload(attachments=[{"attachment_id": theirs, "marker": "theirs"}]),
        )

    assert refused.status_code == 422
    assert "does not exist or is not yours to submit" in refused.text
    assert "somebody-else" not in refused.text


@pytest.mark.asyncio
async def test_an_attachment_already_submitted_with_another_feature_is_refused() -> None:
    """An attachment binds once. A second submission naming it is not a quiet re-use."""
    app = app_with_attachments()
    store = app.state.attachments
    shared = await uploaded(store)

    async with client(app) as http:
        first = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "c6a"},
            json=payload(
                feature_id="feature-one", attachments=[{"attachment_id": shared, "marker": "m"}]
            ),
        )
        second = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "c6b"},
            json=payload(
                feature_id="feature-two", attachments=[{"attachment_id": shared, "marker": "m"}]
            ),
        )

    assert first.status_code == 201, first.text
    assert second.status_code == 422
    assert "already been submitted with another feature" in second.text
    record = await store.get_metadata(shared)
    assert record is not None
    assert record.feature_id == "feature-one"


@pytest.mark.asyncio
async def test_a_replay_of_the_same_accepted_request_is_not_a_binding_conflict() -> None:
    """The idempotent arm finds the attachments bound to this feature and treats it as done."""
    app = app_with_attachments()
    store = app.state.attachments
    first = await uploaded(store)
    body = payload(attachments=[{"attachment_id": first, "marker": "m"}])

    async with client(app) as http:
        created = await http.post(
            "/features/start", headers={**AUTH, "Idempotency-Key": "c7"}, json=body
        )
        replay = await http.post(
            "/features/start", headers={**AUTH, "Idempotency-Key": "c7"}, json=body
        )

    assert created.status_code == 201, created.text
    assert replay.status_code == 200
    assert replay.json()["created"] is False
    record = await store.get_metadata(first)
    assert record is not None
    assert record.feature_id == "feature-login"


@pytest.mark.asyncio
async def test_a_ninth_image_is_refused() -> None:
    """The per-feature count cap, checked across the whole set rather than per upload."""
    app = app_with_attachments()
    store = app.state.attachments
    references = [
        {"attachment_id": await uploaded(store, name=f"{index}.png"), "marker": f"m{index}"}
        for index in range(9)
    ]

    async with client(app) as http:
        refused = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "c8"},
            json=payload(attachments=references),
        )

    assert refused.status_code == 422
    assert "at most 8 images" in refused.text
    assert "names 9" in refused.text


@pytest.mark.asyncio
async def test_a_set_that_fits_the_count_cap_and_busts_the_byte_cap_is_refused() -> None:
    """Eight images are allowed; eight large images are not, and the cap that fires says so."""
    app = app_with_attachments()
    store = app.state.attachments
    # Three megabytes each: within the per-file cap, over 20 MiB across eight files.
    big = fixtures.content(fixtures.PNG) + b"\x00" * (3 * 1024 * 1024)
    references = []
    for index in range(8):
        record = await store.create(
            owner_id=OWNER, filename=f"{index}.png", media_type="image/png", content=big
        )
        references.append({"attachment_id": record.attachment_id, "marker": f"m{index}"})

    async with client(app) as http:
        refused = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "c9"},
            json=payload(attachments=references),
        )

    assert refused.status_code == 422
    assert "at most 20 MiB" in refused.text


@pytest.mark.asyncio
async def test_a_duplicate_marker_or_a_duplicate_id_is_refused() -> None:
    """Two attachments answering to one marker means the reference points at neither."""
    app = app_with_attachments()
    store = app.state.attachments
    first = await uploaded(store, name="a.png")
    second = await uploaded(store, name="b.png")

    async with client(app) as http:
        same_marker = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "c10a"},
            json=payload(
                attachments=[
                    {"attachment_id": first, "marker": "same"},
                    {"attachment_id": second, "marker": "same"},
                ]
            ),
        )
        same_id = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "c10b"},
            json=payload(
                attachments=[
                    {"attachment_id": first, "marker": "one"},
                    {"attachment_id": first, "marker": "two"},
                ]
            ),
        )

    assert same_marker.status_code == 422
    assert "markers must be unique" in same_marker.text
    assert same_id.status_code == 422
    assert "referenced only once" in same_id.text


@pytest.mark.asyncio
async def test_a_malformed_marker_is_refused_by_the_one_pattern() -> None:
    """One regular expression, and it is the field validator as well as the prose scanner."""
    app = app_with_attachments()
    first = await uploaded(app.state.attachments)

    async with client(app) as http:
        for marker in ("Login Error", "-leading-hyphen", "", "x" * 41, "UPPER"):
            refused = await http.post(
                "/features/start",
                headers={**AUTH, "Idempotency-Key": f"c11-{marker[:4]}"},
                json=payload(attachments=[{"attachment_id": first, "marker": marker}]),
            )
            assert refused.status_code == 422, marker


@pytest.mark.asyncio
async def test_acceptance_failing_for_another_reason_leaves_feature_id_null() -> None:
    """Nothing is bound unless the whole transaction commits.

    Driven through the feature-id collision arm, which raises inside the acceptance
    transaction after the attachment resolution has already succeeded -- the shape of every
    "accepted for some other reason" failure.
    """
    app = app_with_attachments()
    store = app.state.attachments
    taken = await uploaded(store, name="first.png")
    loose = await uploaded(store, name="second.png")

    async with client(app) as http:
        first = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "c12a"},
            json=payload(
                feature_id="feature-collide",
                attachments=[{"attachment_id": taken, "marker": "m"}],
            ),
        )
        # Same feature id, different idempotency key: a conflict raised inside the single
        # transaction, after `_resolve_attachments` has already approved this submission's
        # own, still-unbound image.
        conflict = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "c12b"},
            json=payload(
                feature_id="feature-collide",
                attachments=[{"attachment_id": loose, "marker": "m"}],
            ),
        )

    assert first.status_code == 201, first.text
    assert conflict.status_code == 409
    still_loose = await store.get_metadata(loose)
    assert still_loose is not None
    assert still_loose.feature_id is None
    assert still_loose.content_present is True


@pytest.mark.asyncio
async def test_the_workflow_surface_refuses_attachments_rather_than_dropping_them() -> None:
    """It has nowhere to resolve them, and a dropped image is exactly what must not happen."""
    app = app_with_attachments()
    first = await uploaded(app.state.attachments)

    async with client(app) as http:
        refused = await http.post(
            "/workflow/start",
            headers={**AUTH, "Idempotency-Key": "c13"},
            json={
                "workspace_descriptor": {
                    "workspace_id": "workspace-1",
                    "root_path": "/workspace/backend",
                    "source_repo_url": "https://github.com/example/backend",
                    "default_branch": "main",
                    "working_branch": "feature/login-audit",
                },
                "prd": {
                    "title": "Login audit trail",
                    "problem_statement": "Trace failed logins.",
                    "attachments": [{"attachment_id": first, "marker": "m"}],
                },
            },
        )

    assert refused.status_code == 422
    assert "POST /features/start" in refused.text


@pytest.mark.asyncio
async def test_a_deployment_with_no_attachment_store_refuses_a_submission_naming_one() -> None:
    """503, not a dropped reference: the submission asked for something this deployment lacks."""
    app = create_app(platform_api_key=KEY)

    async with client(app) as http:
        refused = await http.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "c14"},
            json=payload(attachments=[{"attachment_id": "attachment-x", "marker": "m"}]),
        )

    assert refused.status_code == 503


@pytest.mark.asyncio
async def test_a_submission_with_no_attachments_serializes_no_attachments_key() -> None:
    """The unchanged path: no images, and the artifact reads as it did before this item.

    Absent rather than present-and-empty. `{{ prd }}` in the product-manager prompt is this
    payload, so `"attachments": []` would put an instruction to look at pictures in front of
    the model on every feature that showed none -- and would change the idempotency
    fingerprint of every submission that predates the field. The strict schema is unaffected:
    `default_factory=list` means a reader that finds no key still reads an empty list --
    asserted at the schema in `test_prd_attachment_artifact.py`.

    The other empty sections are asserted *present*, because that is what makes this a
    statement about the evidence fields rather than about empty lists in general:
    `constraints: []` was in this payload before either 89- item and must still be.
    """
    app = app_with_attachments()

    async with client(app) as http:
        created = await http.post(
            "/features/start", headers={**AUTH, "Idempotency-Key": "c15"}, json=payload()
        )
        await settle(app)
        prd = await prd_payload(http, "feature-login")

    assert created.status_code == 201, created.text
    assert "attachments" not in prd
    assert "design_references" not in prd
    assert prd["constraints"] == []
    assert prd["stakeholders"] == []
