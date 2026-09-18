"""One deployment, from an empty database to two people working in it and not seeing each other.

The whole change, driven through the HTTP surface in the order an operator would: the
administrator's account gets a password from the environment, they log in, they are made to
choose their own, they create somebody, that person signs in and does their own work, and
neither of them can see the other's.

It exists because the parts are individually tested and the *sequence* is where the seams
are. `must_change_password` gating every route, the query cache being cleared between two
identities in one tab, a disabled account's historical work still resolving for an
administrator -- none of those is visible from inside one endpoint's test.

Every password is generated inline. Nothing in this repository holds one.
"""

from __future__ import annotations

import secrets
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from api.identity import Role
from main import bootstrap_administrator_password, create_app
from services.secrets import InMemorySecretStore
from storage.user_store import InMemoryUserDirectory
from tests.test_feature_api import feature_payload

ADMIN_EMAIL = "akhilesh@appbroda.com"


def a_password() -> str:
    """Return a fresh acceptable password."""
    return f"pw-{secrets.token_urlsafe(16)}"


def client(app: Any) -> AsyncClient:
    """Return an HTTP client bound to the application under test."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


def auth(token: str) -> dict[str, str]:
    """Return the header that authenticates as one token."""
    return {"Authorization": f"Bearer {token}"}


async def deployment() -> tuple[Any, InMemoryUserDirectory]:
    """Build an application in the state migrations 0029-0031 leave a fresh deployment in.

    Which is: the administrator's row exists, has no password, and already carries
    `must_change_password`. `PLATFORM_API_KEY` is deliberately unset -- the target state --
    so per-user credentials are the only way in and nothing below can accidentally
    authenticate with a shared key.
    """
    directory = InMemoryUserDirectory()
    await directory.create_user(
        subject=ADMIN_EMAIL,
        display_name="Akhilesh Kumar Pandey",
        roles=(Role.ADMIN.value,),
        must_change_password=True,
    )
    app = create_app(
        platform_api_key=None,
        user_directory=directory,
        secret_store=InMemorySecretStore(),
    )
    return app, directory


@pytest.mark.asyncio
async def test_a_deployment_goes_from_one_account_to_two_isolated_workspaces() -> None:
    """The whole sequence, in order, with the assertions each step earns.

    Long on purpose. Splitting it into eight tests would mean eight fixtures reconstructing
    the state the previous step reached, and the reconstruction is exactly where a test stops
    describing the deployment and starts describing itself.
    """
    app, directory = await deployment()
    bootstrap = a_password()
    chosen = a_password()
    theirs = a_password()

    # 1. The deploy sets the administrator's first password from the environment.
    assert (
        await bootstrap_administrator_password(directory, email=ADMIN_EMAIL, password=bootstrap)
        == "set"
    )

    async with client(app) as http:
        # 2. They log in with it.
        login = await http.post("/auth/login", json={"email": ADMIN_EMAIL, "password": bootstrap})
        assert login.status_code == 200
        admin_token = login.json()["token"]
        # The response says the password must be replaced, which is what the console gates on.
        assert login.json()["must_change_password"] is True
        assert login.json()["actor"]["subject"] == ADMIN_EMAIL

        # 3. They choose their own. The bootstrap value stops working, which is what makes
        #    leaving `BOOTSTRAP_ADMIN_PASSWORD` in the environment survivable.
        changed = await http.post(
            "/auth/password",
            headers=auth(admin_token),
            json={"current_password": bootstrap, "new_password": chosen},
        )
        assert changed.status_code == 200
        assert changed.json()["must_change_password"] is False
        assert (
            await http.post("/auth/login", json={"email": ADMIN_EMAIL, "password": bootstrap})
        ).status_code == 401
        # The session that made the change survives it; every other one would not.
        assert (await http.get("/me", headers=auth(admin_token))).status_code == 200

        # 4. They create somebody, from the console, with no database or environment edit.
        created = await http.post(
            "/users",
            headers=auth(admin_token),
            json={
                "subject": "Sam@Example.com",
                "display_name": "Sam",
                "roles": ["operator"],
                "password": theirs,
            },
        )
        assert created.status_code == 201
        # Normalised on write, so `Sam@Example.com` and `sam@example.com` are one person.
        assert created.json()["subject"] == "sam@example.com"
        assert created.json()["has_password"] is True
        assert created.json()["must_change_password"] is True
        # Nothing password-shaped in the response. There is no field for one.
        assert "password" not in created.json()
        assert "password_hash" not in created.json()

        # 5. The administrator logs out. Their token stops working immediately -- a session
        #    token is revocable, which is the whole reason login mints one rather than
        #    signing a stateless assertion.
        assert (await http.post("/auth/logout", headers=auth(admin_token))).status_code == 204
        assert (await http.get("/me", headers=auth(admin_token))).status_code == 401

        # 6. Sam logs in, typed with the capitals they used, and replaces the handover
        #    password an administrator chose for them.
        sam_login = await http.post(
            "/auth/login", json={"email": "SAM@example.com", "password": theirs}
        )
        assert sam_login.status_code == 200
        sam_token = sam_login.json()["token"]
        assert sam_login.json()["must_change_password"] is True
        sam_own = a_password()
        assert (
            await http.post(
                "/auth/password",
                headers=auth(sam_token),
                json={"current_password": theirs, "new_password": sam_own},
            )
        ).status_code == 200

        # 7. Sam stores their own provider keys and submits their own feature.
        for provider in ("openai", "github"):
            stored = await http.put(
                f"/credentials/{provider}",
                headers=auth(sam_token),
                json={"secret": f"{provider}-{secrets.token_hex(8)}"},
            )
            assert stored.status_code == 200
            assert stored.json()["configured"] is True

        submitted = await http.post(
            "/features/start",
            headers=auth(sam_token),
            json={**feature_payload(), "feature_id": "feature-sams"},
        )
        assert submitted.status_code == 201
        sams_feature = submitted.json()["feature_id"]

        # 8. Sam sees exactly their own feature.
        sams_list = await http.get("/features", headers=auth(sam_token))
        assert [item["feature_id"] for item in sams_list.json()["features"]] == [sams_feature]

        # 9. A third account, freshly created, sees nothing at all -- not an error, and not
        #    the deployment.
        admin_again = await http.post(
            "/auth/login", json={"email": ADMIN_EMAIL, "password": chosen}
        )
        admin_token = admin_again.json()["token"]
        third = a_password()
        await http.post(
            "/users",
            headers=auth(admin_token),
            json={
                "subject": "alex@example.com",
                "display_name": "Alex",
                "roles": ["operator"],
                "password": third,
            },
        )
        alex_login = await http.post(
            "/auth/login", json={"email": "alex@example.com", "password": third}
        )
        alex_token = alex_login.json()["token"]
        alex_list = await http.get("/features", headers=auth(alex_token))
        assert alex_list.json()["features"] == []
        # And cannot reach Sam's by name. Byte-identical to a feature that never existed.
        foreign = await http.get(f"/features/{sams_feature}", headers=auth(alex_token))
        absent = await http.get("/features/feature-never-existed", headers=auth(alex_token))
        assert foreign.status_code == 404
        assert foreign.text == absent.text.replace("feature-never-existed", sams_feature)

        # 10. The administrator sees both workspaces, because the disaster-recovery runbook
        #     needs them to -- and cannot act in either, because acting spends that person's
        #     money and pushes with their token.
        admin_list = await http.get("/features", headers=auth(admin_token))
        assert sams_feature in {item["feature_id"] for item in admin_list.json()["features"]}
        assert (
            await http.get(f"/features/{sams_feature}", headers=auth(admin_token))
        ).status_code == 200
        assert (
            await http.post(
                f"/features/{sams_feature}/retire",
                headers=auth(admin_token),
                json={"operator": "the admin", "reason": "tidying up"},
            )
        ).status_code == 404

        # 11. Sam is disabled. They cannot log in, and the session they were holding stops
        #     working on the next request rather than at its expiry.
        sam = await directory.find_by_subject("sam@example.com")
        assert sam is not None
        disabled = await http.patch(
            f"/users/{sam.user_id}", headers=auth(admin_token), json={"disabled": True}
        )
        assert disabled.status_code == 200
        assert disabled.json()["disabled"] is True
        assert (
            await http.post("/auth/login", json={"email": "sam@example.com", "password": sam_own})
        ).status_code == 401
        assert (await http.get("/me", headers=auth(sam_token))).status_code == 401

        # 12. And their historical work still resolves for the administrator. Accounts are
        #     disabled rather than deleted precisely so a year of audit records keeps
        #     resolving to a person.
        still_there = await http.get(f"/features/{sams_feature}", headers=auth(admin_token))
        assert still_there.status_code == 200
        assert still_there.json()["feature_id"] == sams_feature


@pytest.mark.asyncio
async def test_the_last_administrator_cannot_lock_the_deployment_out() -> None:
    """Refused, because nobody could undo it without editing the database.

    Asserted as an effect: after the refusal the account still holds `admin` and still works.
    And it is a refusal about *this* deployment's population rather than a rule about the
    administrator -- granting somebody else `admin` first makes the same change succeed.
    """
    app, directory = await deployment()
    password = a_password()
    await bootstrap_administrator_password(directory, email=ADMIN_EMAIL, password=password)

    async with client(app) as http:
        login = await http.post("/auth/login", json={"email": ADMIN_EMAIL, "password": password})
        token = login.json()["token"]
        replacement = a_password()
        await http.post(
            "/auth/password",
            headers=auth(token),
            json={"current_password": password, "new_password": replacement},
        )

        # The administrator's own id, rather than the literal `platform-admin`. Migration
        # 0031 assigns that string, and `InMemoryUserDirectory.create_user` cannot -- it
        # mints a uuid like every other account. Reading it back is what keeps this test
        # about the last-administrator rule rather than about the id.
        administrator = await directory.find_by_subject(ADMIN_EMAIL)
        assert administrator is not None
        demoted = await http.patch(
            f"/users/{administrator.user_id}", headers=auth(token), json={"roles": ["operator"]}
        )
        turned_off = await http.patch(
            f"/users/{administrator.user_id}", headers=auth(token), json={"disabled": True}
        )
        assert demoted.status_code == 409
        assert turned_off.status_code == 409
        # Still an administrator, and still able to administer.
        assert (await http.get("/users", headers=auth(token))).status_code == 200

        # With a second administrator, the same change is ordinary.
        second = a_password()
        await http.post(
            "/users",
            headers=auth(token),
            json={
                "subject": "second@example.com",
                "display_name": "Second",
                "roles": ["admin"],
                "password": second,
            },
        )
        assert (
            await http.patch(
                f"/users/{administrator.user_id}",
                headers=auth(token),
                json={"roles": ["operator"]},
            )
        ).status_code == 200


@pytest.mark.asyncio
async def test_a_duplicate_account_for_one_person_is_a_conflict() -> None:
    """`409`, from the unique constraint rather than from a read before the insert.

    Two administrators creating the same person at the same moment both pass a pre-check;
    only the database can refuse the second. The normalised subject is what makes it one
    person: the second attempt below differs only in capitalisation.
    """
    app, directory = await deployment()
    password = a_password()
    await bootstrap_administrator_password(directory, email=ADMIN_EMAIL, password=password)

    async with client(app) as http:
        login = await http.post("/auth/login", json={"email": ADMIN_EMAIL, "password": password})
        token = login.json()["token"]
        replacement = a_password()
        await http.post(
            "/auth/password",
            headers=auth(token),
            json={"current_password": password, "new_password": replacement},
        )
        body = {"subject": "sam@example.com", "display_name": "Sam", "roles": ["operator"]}

        first = await http.post("/users", headers=auth(token), json=body)
        again = await http.post(
            "/users", headers=auth(token), json={**body, "subject": "SAM@Example.com"}
        )

    assert first.status_code == 201
    assert again.status_code == 409
    assert len(await directory.list_users()) == 2


@pytest.mark.asyncio
async def test_an_administrator_reset_ends_the_account_s_sessions() -> None:
    """The usual reason for one is that the account may be compromised.

    So every session it had stops working, including one being used right now -- and the
    account is made to choose its own password again.
    """
    app, directory = await deployment()
    admin_password = a_password()
    await bootstrap_administrator_password(directory, email=ADMIN_EMAIL, password=admin_password)

    async with client(app) as http:
        login = await http.post(
            "/auth/login", json={"email": ADMIN_EMAIL, "password": admin_password}
        )
        admin_token = login.json()["token"]
        replacement = a_password()
        await http.post(
            "/auth/password",
            headers=auth(admin_token),
            json={"current_password": admin_password, "new_password": replacement},
        )
        theirs = a_password()
        await http.post(
            "/users",
            headers=auth(admin_token),
            json={
                "subject": "sam@example.com",
                "display_name": "Sam",
                "roles": ["operator"],
                "password": theirs,
            },
        )
        sam_login = await http.post(
            "/auth/login", json={"email": "sam@example.com", "password": theirs}
        )
        sam_token = sam_login.json()["token"]
        sam = await directory.find_by_subject("sam@example.com")
        assert sam is not None
        forced = a_password()

        reset = await http.post(
            f"/users/{sam.user_id}/password",
            headers=auth(admin_token),
            json={"password": forced},
        )
        after = await http.get("/me", headers=auth(sam_token))
        with_new = await http.post(
            "/auth/login", json={"email": "sam@example.com", "password": forced}
        )

    assert reset.status_code == 200
    assert reset.json()["must_change_password"] is True
    # No password and no token in the response, and no field either could occupy.
    assert set(reset.json()) == {
        "user_id",
        "subject",
        "display_name",
        "roles",
        "disabled",
        "created_at",
        "has_password",
        "must_change_password",
        "last_login_at",
    }
    assert after.status_code == 401
    assert with_new.status_code == 200
    assert with_new.json()["must_change_password"] is True


@pytest.mark.asyncio
async def test_the_legacy_single_workflow_surface_is_administrators_only() -> None:
    """The §6.1 decision, asserted at the router.

    `/workflow/*` predates features and workspaces and has no owner anywhere in its request
    or response schemas, so it cannot be scoped to its caller. It is confined to
    `platform-admin`'s workspace by the control plane behind it and gated here on
    `WORKSPACE_READ_ANY` -- an administrator grant. An ordinary operator is refused before the
    route resolves anything, which matters because these routes unseal stored credentials as
    a dependency.
    """
    app, directory = await deployment()
    password = a_password()
    await bootstrap_administrator_password(directory, email=ADMIN_EMAIL, password=password)
    operator = await directory.create_user(
        subject="operator@example.com", display_name="An operator", roles=(Role.OPERATOR.value,)
    )
    operator_token = (await directory.issue_token(operator.user_id, label="test")).token

    async with client(app) as http:
        login = await http.post("/auth/login", json={"email": ADMIN_EMAIL, "password": password})
        admin_token = login.json()["token"]
        replacement = a_password()
        await http.post(
            "/auth/password",
            headers=auth(admin_token),
            json={"current_password": password, "new_password": replacement},
        )

        refused = await http.get("/workflow/anything", headers=auth(operator_token))
        refused_start = await http.post("/workflow/start", headers=auth(operator_token), json={})
        allowed = await http.get("/workflow/anything", headers=auth(admin_token))

    assert refused.status_code == 403
    # Refused before the body is even validated, which is what a route-level dependency
    # buys: an unauthorized caller is not told which fields their body was missing, and --
    # more to the point -- their stored keys are not unsealed on the way to refusing them.
    assert refused_start.status_code == 403
    # The administrator reaches it; there is simply no such workflow.
    assert allowed.status_code == 404
