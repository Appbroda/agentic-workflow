"""Who may do what, and what the platform will admit about the keys it holds.

Two properties are being defended here. The first is that authorization is the server's job:
a client that never draws a button must still be refused if it calls the endpoint. The second
is that a stored provider credential is never readable back out of the platform -- not from an
endpoint, not from a response body, not from a descriptor that something later serializes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pytest
from httpx import ASGITransport, AsyncClient

from api.control_plane import RequestScopedCredentials, WorkflowNotFoundError
from api.feature_schemas import StartFeatureRequest
from api.identity import Actor, Permission, Role, hash_token, platform_admin
from api.routes import CREDENTIAL_PROVIDERS, required_providers_for_platforms, stored_credentials
from main import create_app
from services.credential_verification import (
    CompositeCredentialVerifier,
    CredentialVerdict,
    FigmaCredentialVerifier,
)
from services.secrets import (
    SUPPORTED_PROVIDERS,
    EncryptedDatabaseSecretStore,
    InMemorySecretStore,
    SecretStoreError,
    SecretStoreUnavailableError,
    generate_encryption_key,
    load_encryption_key,
)
from state.enums import FeatureActionStatus
from storage.db import Database
from storage.user_store import InMemoryUserDirectory, UserDirectoryError
from tests.test_feature_api import feature_payload

KEY = "identity-key"


async def app_with_identity() -> tuple[Any, InMemoryUserDirectory, InMemorySecretStore]:
    """Build an application that knows individual users and can keep their credentials."""
    directory = InMemoryUserDirectory()
    secrets_store = InMemorySecretStore()
    app = create_app(platform_api_key=KEY, user_directory=directory, secret_store=secrets_store)
    return app, directory, secrets_store


async def token_for(directory: InMemoryUserDirectory, *, roles: tuple[str, ...]) -> str:
    """Register somebody with the given roles and return a usable token for them."""
    user = await directory.create_user(
        subject=f"person-{roles[0]}-{len(await directory.list_users())}",
        display_name=f"A {roles[0]}",
        roles=roles,
    )
    issued = await directory.issue_token(user.user_id, label="test")
    return issued.token


def client(app: Any) -> AsyncClient:
    """Return an HTTP client bound to the application under test."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest.mark.asyncio
async def test_the_shared_key_resolves_to_a_named_administrative_identity() -> None:
    """The key used to be an absence of identity. It is now an identity with a name."""
    app, _, _ = await app_with_identity()

    async with client(app) as http:
        response = await http.get("/me", headers={"Authorization": f"Bearer {KEY}"})

    body = response.json()
    assert response.status_code == 200
    assert body["actor_id"] == "platform-admin"
    assert body["authentication"] == "platform_key"
    assert "repair:approve" in body["permissions"]


@pytest.mark.asyncio
async def test_a_user_token_resolves_to_that_user() -> None:
    """The point of the whole exercise: the platform can tell one person from another."""
    app, directory, _ = await app_with_identity()
    token = await token_for(directory, roles=(Role.OPERATOR.value,))

    async with client(app) as http:
        response = await http.get("/me", headers={"Authorization": f"Bearer {token}"})

    body = response.json()
    assert body["authentication"] == "user_token"
    assert body["actor_id"].startswith("user-")
    assert body["roles"] == [Role.OPERATOR.value]


@pytest.mark.asyncio
async def test_an_unauthenticated_request_reaches_nothing() -> None:
    """Every new route is behind the same authentication as everything else."""
    app, _, _ = await app_with_identity()

    async with client(app) as http:
        responses = [
            await http.get("/me"),
            await http.get("/credentials"),
            await http.put("/credentials/openai", json={"secret": "sk-test"}),
            await http.get("/users"),
        ]

    assert [item.status_code for item in responses] == [401, 401, 401, 401]


@pytest.mark.asyncio
async def test_a_revoked_token_stops_working() -> None:
    """Revocation has to take effect on the next request, not on the next restart."""
    app, directory, _ = await app_with_identity()
    user = await directory.create_user(
        subject="leaver", display_name="A leaver", roles=(Role.OPERATOR.value,)
    )
    issued = await directory.issue_token(user.user_id, label="laptop")
    headers = {"Authorization": f"Bearer {issued.token}"}

    async with client(app) as http:
        before = await http.get("/me", headers=headers)
        await directory.revoke_token(issued.token_id)
        after = await http.get("/me", headers=headers)

    assert before.status_code == 200
    assert after.status_code == 401


@pytest.mark.asyncio
async def test_an_expired_token_stops_working() -> None:
    """An expiry nobody enforces is a note, not a control."""
    app, directory, _ = await app_with_identity()
    user = await directory.create_user(
        subject="temporary", display_name="Temporary", roles=(Role.OPERATOR.value,)
    )
    issued = await directory.issue_token(
        user.user_id, label="short", expires_at=datetime.now(UTC) - timedelta(minutes=1)
    )

    async with client(app) as http:
        response = await http.get("/me", headers={"Authorization": f"Bearer {issued.token}"})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_a_disabled_user_cannot_authenticate_with_a_token_they_still_hold() -> None:
    """Turning somebody off must not depend on collecting their tokens back in."""
    app, directory, _ = await app_with_identity()
    user = await directory.create_user(
        subject="disabled", display_name="Disabled", roles=(Role.OPERATOR.value,)
    )
    issued = await directory.issue_token(user.user_id, label="laptop")
    await directory.set_disabled(user.user_id, disabled=True)

    async with client(app) as http:
        response = await http.get("/me", headers={"Authorization": f"Bearer {issued.token}"})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_a_viewer_is_refused_every_action_that_changes_a_feature() -> None:
    """The server refuses, whatever the client chose to render.

    Each of these has a button somewhere. A client that hides them is being helpful; it is
    not the control, and this is the control.

    Every refusal below is the permission refusing, not the workspace filter, and that is
    load-bearing: the permission is resolved before anything looks the feature up -- as a
    route dependency on most of these and as the first statement in the body on the rest --
    so a viewer is told they may not do this rather than told the feature does not exist.
    The feature here is owned by `platform-admin` and the viewer is somebody else, so if the
    order were the other way round these would all be 404 and this test would defend nothing.

    The read at the end is therefore against a feature in the viewer's *own* workspace,
    because reading is what a viewer is for and that has to keep working.
    """
    app, directory, _ = await app_with_identity()
    viewer_user = await directory.create_user(
        subject="viewer@example.com", display_name="A viewer", roles=(Role.VIEWER.value,)
    )
    viewer = (await directory.issue_token(viewer_user.user_id, label="test")).token
    admin_headers = {"Authorization": f"Bearer {KEY}", "Idempotency-Key": "viewer-check"}
    headers = {"Authorization": f"Bearer {viewer}"}

    async with client(app) as http:
        await http.post("/features/start", headers=admin_headers, json=feature_payload())
        own = await app.state.feature_control_plane.start(
            StartFeatureRequest.model_validate(
                {**feature_payload(), "feature_id": "feature-viewers-own"}
            ),
            idempotency_key="viewer-own",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            owner_id=viewer_user.user_id,
        )
        refusals = {
            "create": await http.post("/features/start", headers=headers, json=feature_payload()),
            "resume": await http.post(
                "/features/feature-login/resume", headers=headers, json={"answers": []}
            ),
            "retry": await http.post(
                "/features/feature-login/workstreams/backend/retry",
                headers=headers,
                json={"additional_attempts": 1, "requested_by": "me", "reason": "because"},
            ),
            "cancel": await http.post("/features/feature-login/cancel", headers=headers, json={}),
            "repair_approve": await http.post(
                "/features/feature-login/repairs/repair-1/approve",
                headers=headers,
                json={"acknowledge_repository_change": True},
            ),
            "repair_reject": await http.post(
                "/features/feature-login/repairs/repair-1/reject",
                headers=headers,
                json={"reason": "no"},
            ),
            "retire": await http.post(
                "/features/feature-login/retire",
                headers=headers,
                json={"operator": "me", "reason": "done"},
            ),
        }
        # Reading is what a viewer is for, and must still work -- in their own workspace.
        readable = await http.get(f"/features/{own.record.state.feature_id}", headers=headers)

    assert {name: item.status_code for name, item in refusals.items()} == {
        "create": 403,
        "resume": 403,
        "retry": 403,
        "cancel": 403,
        "repair_approve": 403,
        "repair_reject": 403,
        "retire": 403,
    }
    assert readable.status_code == 200


@pytest.mark.asyncio
async def test_a_viewer_is_refused_before_the_request_body_is_even_examined() -> None:
    """Authorization must not wait behind body validation, or behind a stored credential.

    Checked inside the function, the permission is the last thing to run: FastAPI validates
    the body and resolves every other dependency first. So an identity that may not create a
    feature was told which fields its body was missing, and -- because these routes resolve
    the caller's stored provider credentials to build their arguments -- had its own keys
    unsealed and their last-used timestamp moved, on the way to being refused.
    """
    app, directory, secrets_store = await app_with_identity()
    viewer = await token_for(directory, roles=(Role.VIEWER.value,))
    headers = {"Authorization": f"Bearer {viewer}"}
    actor = await directory.resolve(viewer)
    assert actor is not None
    await secrets_store.put(owner_id=actor.actor_id, provider="openai", secret="sk-viewer-key")

    async with client(app) as http:
        refusals = {
            "create": await http.post("/features/start", headers=headers, json={"prd": "nonsense"}),
            "resume": await http.post(
                "/features/feature-login/resume", headers=headers, json={"answers": "nonsense"}
            ),
            "retry": await http.post(
                "/features/feature-login/workstreams/backend/retry", headers=headers, json={}
            ),
            "repair_approve": await http.post(
                "/features/feature-login/repairs/repair-1/approve", headers=headers, json={}
            ),
        }

    assert {name: item.status_code for name, item in refusals.items()} == {
        "create": 403,
        "resume": 403,
        "retry": 403,
        "repair_approve": 403,
    }
    # Refused before anything read the key, so nothing claims it was used.
    descriptor = await secrets_store.describe(owner_id=actor.actor_id, provider="openai")
    assert descriptor.last_used_at is None


@pytest.mark.asyncio
async def test_chat_cannot_be_used_to_reach_a_mutation_the_caller_is_refused() -> None:
    """The assistant is an interpretation layer, not a second way past authorization."""
    app, directory, _ = await app_with_identity()
    viewer = await token_for(directory, roles=(Role.VIEWER.value,))
    admin_headers = {"Authorization": f"Bearer {KEY}", "Idempotency-Key": "chat-authz"}

    async with client(app) as http:
        await http.post("/features/start", headers=admin_headers, json=feature_payload())
        response = await http.post(
            "/features/feature-login/chat/1/confirm",
            headers={"Authorization": f"Bearer {viewer}"},
        )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_an_operator_cannot_grant_themselves_more() -> None:
    """Somebody who can retry a repository must not be able to hand themselves user admin."""
    app, directory, _ = await app_with_identity()
    operator = await token_for(directory, roles=(Role.OPERATOR.value,))

    async with client(app) as http:
        listed = await http.get("/users", headers={"Authorization": f"Bearer {operator}"})
        created = await http.post(
            "/users",
            headers={"Authorization": f"Bearer {operator}"},
            json={"subject": "me-again", "display_name": "Me", "roles": [Role.ADMIN.value]},
        )

    assert listed.status_code == 403
    assert created.status_code == 403


@pytest.mark.asyncio
async def test_only_an_admin_can_reconcile_an_uncertain_action() -> None:
    """An operator may execute work but may not assert an unprovable outcome after a crash."""
    app, directory, _ = await app_with_identity()
    operator = await token_for(directory, roles=(Role.OPERATOR.value,))
    admin_headers = {"Authorization": f"Bearer {KEY}", "Idempotency-Key": "reconcile-authz"}
    async with client(app) as http:
        await http.post("/features/start", headers=admin_headers, json=feature_payload())
        actions = app.state.feature_actions
        action, _ = await actions.submit(
            feature_id="feature-login",
            action_type="CANCEL_WORKFLOW",
            actor_id="original-actor",
            payload={"reason": "stop"},
            context_version=None,
        )
        store = actions.store
        store._actions[action.action_id] = action.model_copy(  # noqa: SLF001 - crash fixture
            update={"status": FeatureActionStatus.REQUIRES_RECONCILIATION}
        )
        path = f"/features/feature-login/actions/{action.action_id}/reconcile"
        refused = await http.post(
            path,
            headers={"Authorization": f"Bearer {operator}"},
            json={"outcome": "failed", "reason": "Checked the feature state."},
        )
        accepted = await http.post(
            path,
            headers={"Authorization": f"Bearer {KEY}"},
            json={"outcome": "failed", "reason": "Checked the feature and provider state."},
        )
        repeated = await http.post(
            path,
            headers={"Authorization": f"Bearer {KEY}"},
            json={"outcome": "succeeded", "reason": "Try to overwrite the first decision."},
        )

    assert refused.status_code == 403
    assert accepted.status_code == 200
    assert accepted.json()["reconciled_by"] == "platform-admin"
    assert repeated.status_code == 409


@pytest.mark.asyncio
async def test_a_token_is_returned_once_and_never_again() -> None:
    """The platform stores a digest, so the issue response is the only copy that exists."""
    app, _, _ = await app_with_identity()
    headers = {"Authorization": f"Bearer {KEY}"}

    async with client(app) as http:
        created = await http.post(
            "/users",
            headers=headers,
            json={
                "subject": "new-person",
                "display_name": "New Person",
                "roles": [Role.OPERATOR.value],
            },
        )
        user_id = created.json()["user_id"]
        issued = await http.post(
            f"/users/{user_id}/tokens", headers=headers, json={"label": "laptop"}
        )
        listed = await http.get(f"/users/{user_id}/tokens", headers=headers)

    assert issued.status_code == 200
    minted = issued.json()["token"]
    assert minted
    # The listing knows the token exists and cannot reproduce it.
    assert minted not in listed.text
    assert hash_token(minted) not in listed.text
    assert [item["label"] for item in listed.json()["tokens"]] == ["laptop"]


@pytest.mark.asyncio
async def test_the_credential_api_never_returns_the_credential() -> None:
    """Asserted against the raw response body, not a parsed field.

    A model with no secret field can still leak one if something puts it there; checking the
    bytes that actually leave the process is the assertion that cannot be satisfied by
    accident.
    """
    app, _, _ = await app_with_identity()
    headers = {"Authorization": f"Bearer {KEY}"}
    secret = "sk-live-do-not-echo-this-value"

    async with client(app) as http:
        stored = await http.put("/credentials/openai", headers=headers, json={"secret": secret})
        listed = await http.get("/credentials", headers=headers)
        checked = await http.post("/credentials/openai/check", headers=headers)

    for response in (stored, listed, checked):
        assert secret not in response.text
    assert stored.json()["configured"] is True
    # The hint identifies the key to whoever owns it and nobody else.
    assert stored.json()["hint"] == secret[-4:]


@pytest.mark.asyncio
async def test_a_stored_credential_is_used_when_a_request_brings_none() -> None:
    """This is the point of storing one: a feature can run without somebody retyping a key."""
    app, directory, store = await app_with_identity()
    token = await token_for(directory, roles=(Role.OPERATOR.value,))

    async with client(app) as http:
        me = await http.get("/me", headers={"Authorization": f"Bearer {token}"})
        await http.put(
            "/credentials/openai",
            headers={"Authorization": f"Bearer {token}"},
            json={"secret": "sk-stored-value"},
        )

    resolved = await store.resolve(owner_id=me.json()["actor_id"], provider="openai")
    assert resolved == "sk-stored-value"


@pytest.mark.asyncio
async def test_a_request_header_wins_over_anything_stored() -> None:
    """Supplying a key for one request is a deliberate choice about which key does the work."""
    from api.identity import Actor as ActorType
    from api.routes import resolve_credentials

    store = InMemorySecretStore()
    actor = ActorType(actor_id="user-1", display_name="Someone", roles=frozenset())
    await store.put(owner_id="user-1", provider="openai", secret="sk-stored")
    await store.put(owner_id="user-1", provider="github", secret="ghp-stored")

    class _Request:
        app = type("_App", (), {"state": type("_State", (), {"secret_store": store})()})()

    resolved = await resolve_credentials(
        _Request(),  # type: ignore[arg-type]
        actor,
        openai_api_key="sk-from-header",
        github_token=None,
    )

    assert resolved.openai_api_key == "sk-from-header"
    # Only what the request did not supply falls back to the store.
    assert resolved.github_token == "ghp-stored"


@pytest.mark.asyncio
async def test_one_persons_credential_does_not_open_for_another() -> None:
    """The seal is bound to its owner, so a row moved into another name is unreadable."""
    store = InMemorySecretStore()
    await store.put(owner_id="user-1", provider="openai", secret="sk-one")

    import copy

    stolen = copy.copy(store._rows[("user-1", "openai")])  # noqa: SLF001 - a moved row
    store._rows[("user-2", "openai")] = stolen  # noqa: SLF001

    with pytest.raises(SecretStoreError, match="re-entered"):
        await store.resolve(owner_id="user-2", provider="openai")


@pytest.mark.asyncio
async def test_an_expired_credential_is_not_used() -> None:
    """A key its owner has retired must stop being presented to a provider."""
    store = InMemorySecretStore()
    await store.put(
        owner_id="user-1",
        provider="openai",
        secret="sk-old",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    assert await store.resolve(owner_id="user-1", provider="openai") is None
    # Still reported as configured, so somebody can see why nothing is working.
    assert (await store.describe(owner_id="user-1", provider="openai")).configured is True


@pytest.mark.asyncio
async def test_removing_a_credential_removes_it() -> None:
    """Deletion is deletion, not a flag that something later ignores."""
    store = InMemorySecretStore()
    await store.put(owner_id="user-1", provider="github", secret="ghp-value")

    assert await store.delete(owner_id="user-1", provider="github") is True
    assert await store.resolve(owner_id="user-1", provider="github") is None
    assert (await store.describe(owner_id="user-1", provider="github")).configured is False
    assert await store.delete(owner_id="user-1", provider="github") is False


@pytest.mark.asyncio
async def test_replacing_a_credential_leaves_no_older_one_behind() -> None:
    """An old key lingering behind a new one could be resolved by accident."""
    store = InMemorySecretStore()
    await store.put(owner_id="user-1", provider="openai", secret="sk-first")
    await store.put(owner_id="user-1", provider="openai", secret="sk-second")

    assert await store.resolve(owner_id="user-1", provider="openai") == "sk-second"


def test_a_descriptor_has_nowhere_to_put_a_secret() -> None:
    """The type that describes a credential must not be able to carry one.

    Asserted structurally rather than by inspecting a value, because the risk is a field
    added later that something serializes without thinking about it.
    """
    from dataclasses import fields

    from services.secrets import SecretDescriptor

    names = {item.name for item in fields(SecretDescriptor)}
    assert not names & {"secret", "value", "token", "api_key", "plaintext", "ciphertext"}


def test_an_unconfigured_or_weak_encryption_key_is_refused() -> None:
    """A store that appears to work and encrypts weakly is worse than one that refuses."""
    import base64

    with pytest.raises(SecretStoreUnavailableError):
        load_encryption_key(None)
    with pytest.raises(SecretStoreUnavailableError):
        load_encryption_key("   ")
    with pytest.raises(SecretStoreUnavailableError, match="base64"):
        load_encryption_key("not-base64-!!")
    with pytest.raises(SecretStoreUnavailableError, match="32 bytes"):
        load_encryption_key(base64.b64encode(b"too-short").decode("ascii"))

    key, version = load_encryption_key(generate_encryption_key())
    assert len(key) == 32
    assert version == "v1"


@pytest.mark.asyncio
async def test_a_deployment_without_an_encryption_key_says_so_rather_than_failing() -> None:
    """Not storing credentials is the platform's original behaviour, not a fault."""
    app = create_app(platform_api_key=KEY)
    headers = {"Authorization": f"Bearer {KEY}"}

    async with client(app) as http:
        listed = await http.get("/credentials", headers=headers)
        stored = await http.put("/credentials/openai", headers=headers, json={"secret": "sk-test"})

    assert listed.status_code == 503
    assert stored.status_code == 503


@pytest.mark.asyncio
async def test_an_unknown_provider_is_refused() -> None:
    """A credential nothing would ever resolve should not be silently accepted."""
    app, _, _ = await app_with_identity()

    async with client(app) as http:
        # A provider this platform has no client for. `anthropic` was this test's example
        # until the platform gained a second model provider, at which point it stopped being
        # an unknown name and started being one of the two.
        response = await http.put(
            "/credentials/mistral",
            headers={"Authorization": f"Bearer {KEY}"},
            json={"secret": "value"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_credentials_survive_a_restart_of_the_process(tmp_path: Path) -> None:
    """Written against a real database, because that is the property being claimed."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'secrets.db'}")
    await database.create_schema()
    key = generate_encryption_key()
    try:
        first = EncryptedDatabaseSecretStore(database, encryption_key=key)
        await first.put(owner_id="user-1", provider="github", secret="ghp-persisted")

        # A new process, same key.
        second = EncryptedDatabaseSecretStore(database, encryption_key=key)
        assert await second.resolve(owner_id="user-1", provider="github") == "ghp-persisted"

        described = await second.describe(owner_id="user-1", provider="github")
        assert described.configured is True
        assert described.hint == "sted"

        # A different key cannot open what this one sealed.
        third = EncryptedDatabaseSecretStore(database, encryption_key=generate_encryption_key())
        with pytest.raises(SecretStoreError, match="re-entered"):
            await third.resolve(owner_id="user-1", provider="github")
    finally:
        await database.dispose()


@pytest.mark.asyncio
async def test_a_previous_key_is_lazily_rotated_to_the_current_version(tmp_path: Path) -> None:
    """Rotation must open old rows and persist a fresh seal before the old key is removed."""
    from sqlalchemy import select

    from storage.models import ProviderCredentialModel

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'rotation.db'}")
    old_key = generate_encryption_key()
    new_key = f"v2:{generate_encryption_key().partition(':')[2]}"
    try:
        old_store = EncryptedDatabaseSecretStore(database, encryption_key=old_key)
        await database.create_schema()
        await old_store.put(owner_id="user-1", provider="openai", secret="sk-rotate-me")

        rotating = EncryptedDatabaseSecretStore(
            database,
            encryption_key=new_key,
            previous_encryption_keys=[old_key],
        )
        assert await rotating.resolve(owner_id="user-1", provider="openai") == "sk-rotate-me"

        async with database.session() as session:
            row = (await session.execute(select(ProviderCredentialModel))).scalar_one()
            assert row.key_version == "v2"

        # The old key is no longer needed after the successful read re-sealed the row.
        current_only = EncryptedDatabaseSecretStore(database, encryption_key=new_key)
        assert await current_only.resolve(owner_id="user-1", provider="openai") == "sk-rotate-me"
    finally:
        await database.dispose()


def test_duplicate_encryption_key_versions_are_refused(tmp_path: Path) -> None:
    """A version label must identify exactly one key or decryption becomes ambiguous."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'duplicate-versions.db'}")
    first = generate_encryption_key()
    second = generate_encryption_key()

    with pytest.raises(SecretStoreUnavailableError, match="more than once"):
        EncryptedDatabaseSecretStore(
            database,
            encryption_key=first,
            previous_encryption_keys=[second],
        )


@pytest.mark.asyncio
async def test_the_plaintext_is_never_a_column(tmp_path: Path) -> None:
    """Reading the row directly must not produce the secret."""
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'columns.db'}")
    await database.create_schema()
    try:
        store = EncryptedDatabaseSecretStore(database, encryption_key=generate_encryption_key())
        await store.put(owner_id="user-1", provider="openai", secret="sk-secret-value")

        from sqlalchemy import select

        from storage.models import ProviderCredentialModel

        async with database.session() as session:
            row = (await session.execute(select(ProviderCredentialModel))).scalar_one()
            stored = {
                column.name: getattr(row, column.name)
                for column in ProviderCredentialModel.__table__.columns
            }

        assert b"sk-secret-value" not in bytes(stored["ciphertext"])
        assert "sk-secret-value" not in json.dumps(stored, default=str)
    finally:
        await database.dispose()


def test_roles_grant_what_they_say_and_nothing_more() -> None:
    """An unrecognised role must grant nothing rather than everything."""
    viewer = Actor(actor_id="a", display_name="A", roles=frozenset({Role.VIEWER.value}))
    operator = Actor(actor_id="b", display_name="B", roles=frozenset({Role.OPERATOR.value}))
    admin = Actor(actor_id="c", display_name="C", roles=frozenset({Role.ADMIN.value}))
    invented = Actor(actor_id="d", display_name="D", roles=frozenset({"superuser"}))

    assert viewer.may(Permission.FEATURE_READ)
    assert not viewer.may(Permission.FEATURE_RETRY)
    assert operator.may(Permission.REPAIR_APPROVE)
    assert not operator.may(Permission.USER_MANAGE)
    assert admin.may(Permission.USER_MANAGE)
    assert invented.permissions == frozenset()
    assert platform_admin().may(Permission.USER_MANAGE)


@pytest.mark.asyncio
async def test_a_user_cannot_be_registered_with_a_role_nobody_defined() -> None:
    """A deployment that thinks it granted something it did not is worse than an error."""
    directory = InMemoryUserDirectory()

    with pytest.raises(UserDirectoryError, match="unknown roles"):
        await directory.create_user(subject="someone", display_name="Someone", roles=("superuser",))


@pytest.mark.asyncio
async def test_the_same_subject_cannot_be_registered_twice() -> None:
    """Two rows for one person would make an audit record ambiguous about who acted."""
    directory = InMemoryUserDirectory()
    await directory.create_user(subject="alex", display_name="Alex", roles=(Role.OPERATOR.value,))

    with pytest.raises(UserDirectoryError, match="already exists"):
        await directory.create_user(
            subject="alex", display_name="Alex Again", roles=(Role.VIEWER.value,)
        )


def test_a_token_is_stored_only_as_a_digest() -> None:
    """The stored form must not be reversible to the credential."""
    from api.identity import issue_token

    token, digest = issue_token()

    assert digest == hash_token(token)
    assert token not in digest
    assert len(digest) == 64


@pytest.mark.asyncio
async def test_a_retry_grant_records_the_authenticated_identity() -> None:
    """Who overrode the platform's own decision is not a name somebody typed.

    The request body still carries one, and it is kept when the shared administrative key
    was used -- that is the only attribution available then, and it is labelled as
    unverified rather than presented as though the platform checked it.

    Each grant is made on a feature its own grantor owns. Since workspaces became isolated a
    grant on somebody else's feature is a 404, so the two calls need two features -- and this
    test is about the recorded name, not about who may reach what.
    """
    app, directory, _ = await app_with_identity()
    operator = await directory.create_user(
        subject="granter@example.com", display_name="A granter", roles=(Role.OPERATOR.value,)
    )
    token = (await directory.issue_token(operator.user_id, label="test")).token
    granted: list[str] = []

    async def spy(feature_id: str, repository_id: str, **kwargs: Any) -> Any:
        granted.append(str(kwargs["requested_by"]))
        raise WorkflowNotFoundError("stop here; the grant argument is what is under test")

    theirs = await app.state.feature_control_plane.start(
        StartFeatureRequest.model_validate(
            {**feature_payload(), "feature_id": "feature-granters-own"}
        ),
        idempotency_key="granter-own",
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        owner_id=operator.user_id,
    )
    app.state.feature_control_plane.retry_workstream = spy
    body = {"additional_attempts": 1, "requested_by": "typed-by-hand", "reason": "read them"}

    async with client(app) as http:
        await http.post(
            "/features/start",
            headers={"Authorization": f"Bearer {KEY}"},
            json=feature_payload(),
        )
        me = await http.get("/me", headers={"Authorization": f"Bearer {token}"})
        await http.post(
            f"/features/{theirs.record.state.feature_id}/workstreams/backend/retry",
            headers={"Authorization": f"Bearer {token}"},
            json=body,
        )
        await http.post(
            "/features/feature-login/workstreams/backend/retry",
            headers={"Authorization": f"Bearer {KEY}"},
            json={**body, "reason": "read them again"},
        )

    assert granted[0] == me.json()["actor_id"], "a named person is recorded as themselves"
    assert granted[1] == "typed-by-hand (via platform key)", (
        "the shared key cannot say who held it, so the typed name is kept and labelled"
    )


@pytest.mark.asyncio
async def test_a_repair_decision_names_who_made_it_in_the_timeline() -> None:
    """An audit answers 'who' from the timeline, which is where somebody looks first.

    The decision is made on a feature the decider owns: a repair decision on somebody else's
    feature is a 404 now, and what is under test here is the recorded name.
    """
    app, directory, _ = await app_with_identity()
    operator = await directory.create_user(
        subject="decider@example.com", display_name="A decider", roles=(Role.OPERATOR.value,)
    )
    token = (await directory.issue_token(operator.user_id, label="test")).token
    recorded: list[dict[str, Any]] = []

    async def spy(feature_id: str, **kwargs: Any) -> Any:
        recorded.append(dict(kwargs))
        raise WorkflowNotFoundError("stop here; the actor argument is what is under test")

    theirs = await app.state.feature_control_plane.start(
        StartFeatureRequest.model_validate(
            {**feature_payload(), "feature_id": "feature-deciders-own"}
        ),
        idempotency_key="decider-own",
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        owner_id=operator.user_id,
    )
    app.state.feature_control_plane.reject_repair = spy

    async with client(app) as http:
        me = await http.get("/me", headers={"Authorization": f"Bearer {token}"})
        await http.post(
            f"/features/{theirs.record.state.feature_id}/repairs/repair-1/reject",
            headers={"Authorization": f"Bearer {token}"},
            json={"reason": "We are removing the rule instead."},
        )

    assert recorded[0]["actor_id"] == me.json()["actor_id"]
    assert recorded[0]["repair_id"] == "repair-1"


@pytest.mark.asyncio
async def test_a_chat_action_needs_the_permission_that_action_needs() -> None:
    """Confirming in a sentence must meet the same bar as pressing the button that does it.

    A single "may execute chat actions" permission let anybody allowed to answer a
    clarification also approve a repository repair -- a change to somebody's repository --
    purely for having asked for it in prose.
    """
    from typing import cast

    from api.identity import Permission as PermissionType
    from api.identity import permission_for_action

    class _Clarifier:
        """Somebody who may answer questions and nothing else."""

        def may(self, permission: PermissionType) -> bool:
            return permission is PermissionType.FEATURE_ANSWER_CLARIFICATION

    actor = cast(Actor, _Clarifier())

    assert actor.may(permission_for_action("ANSWER_CLARIFICATION"))
    assert actor.may(permission_for_action("RESUME_WORKFLOW"))
    # Each of these is a different decision, and none of them is theirs to make.
    assert not actor.may(permission_for_action("RETRY_WORKSTREAM"))
    assert not actor.may(permission_for_action("CANCEL_WORKFLOW"))
    assert not actor.may(permission_for_action("APPROVE_REPOSITORY_REPAIR"))
    assert not actor.may(permission_for_action("REJECT_REPOSITORY_REPAIR"))


def test_an_unknown_action_type_is_refused_rather_than_allowed() -> None:
    """A build that does not know what an action does must not conclude anybody may do it."""
    from api.identity import Permission as PermissionType
    from api.identity import permission_for_action

    operator = Actor(actor_id="a", display_name="A", roles=frozenset({Role.OPERATOR.value}))

    assert permission_for_action("SOMETHING_A_NEWER_BUILD_ADDED") is PermissionType.USER_MANAGE
    assert not operator.may(permission_for_action("SOMETHING_A_NEWER_BUILD_ADDED"))


def test_every_action_the_assistant_can_propose_has_a_permission() -> None:
    """A new action type with no mapping would fall back to admin and silently be unusable."""
    from agents.assistant.agent import SUPPORTED_ACTIONS
    from api.identity import ACTION_PERMISSIONS

    proposable = {item.type for item in SUPPORTED_ACTIONS}

    assert proposable <= set(ACTION_PERMISSIONS), (
        f"unmapped actions: {sorted(proposable - set(ACTION_PERMISSIONS))}"
    )


# --------------------------------------------------------------------------------------
# Readable is not usable: verifying a stored credential against its provider
# --------------------------------------------------------------------------------------


class _ScriptedVerifier:
    """Answer with a fixed verdict and record what it was asked about, never the secret."""

    def __init__(self, verdict: CredentialVerdict) -> None:
        """Bind the verdict every call returns."""
        self._verdict = verdict
        self.providers: list[str] = []

    async def verify(self, *, provider: str, secret: str) -> CredentialVerdict:
        """Record the provider asked about and return the scripted verdict."""
        assert secret, "a verifier is never asked about an empty credential"
        self.providers.append(provider)
        return self._verdict


@pytest.mark.asyncio
async def test_the_check_endpoint_distinguishes_readable_from_provider_accepted() -> None:
    """The question run 190 needed, beside the one this endpoint already answered.

    The GitHub PAT stored on 2026-08-26 expired at 2026-09-02 00:00 UTC. This endpoint went
    on reporting it usable for the whole day that followed -- correctly, because the bytes
    decrypted fine -- while every authenticated clone the platform attempted was refused.
    Both answers are now available and they are separate fields: `usable` is still about this
    deployment's own encryption, and `verified` is the provider's verdict.
    """
    app, _, _ = await app_with_identity()
    verifier = _ScriptedVerifier(CredentialVerdict.REFUSED)
    app.state.credential_verifier = verifier
    headers = {"Authorization": f"Bearer {KEY}"}
    secret = "ghp-expired-at-midnight"

    async with client(app) as http:
        await http.put("/credentials/github", headers=headers, json={"secret": secret})
        local = await http.post("/credentials/github/check", headers=headers)
        verified = await http.post("/credentials/github/check?verify=true", headers=headers)

    # Not asked unless asked for: the default is still the local answer, and no provider was
    # dialled on a button press.
    assert local.json()["usable"] is True
    assert local.json()["verified"] is None
    assert verifier.providers == ["github"]
    # Asked, and the two answers disagree -- which is precisely the state a dead PAT is in.
    body = verified.json()
    assert body["usable"] is True
    assert body["verified"] == "refused"
    assert "refused it" in body["detail"]
    assert "replaced" in body["detail"]
    # And neither response is a way to read the credential back out.
    for response in (local, verified):
        assert secret not in response.text


@pytest.mark.asyncio
async def test_an_unreachable_provider_is_not_reported_as_a_bad_credential() -> None:
    """No answer is not a negative answer, and this endpoint must not blur them.

    A check that reported "refused" whenever GitHub was briefly unreachable would train its
    reader to ignore it, which costs exactly the diagnosis the verification exists to add.
    """
    app, _, _ = await app_with_identity()
    app.state.credential_verifier = _ScriptedVerifier(CredentialVerdict.UNKNOWN)
    headers = {"Authorization": f"Bearer {KEY}"}

    async with client(app) as http:
        await http.put("/credentials/github", headers=headers, json={"secret": "ghp-fine"})
        verified = await http.post("/credentials/github/check?verify=true", headers=headers)

    body = verified.json()
    assert body["usable"] is True
    assert body["verified"] == "unknown"
    assert "could not be asked" in body["detail"]


@pytest.mark.asyncio
async def test_a_deployment_that_verifies_nothing_answers_the_local_question_only() -> None:
    """Every application built without a verifier keeps exactly the behaviour it had."""
    app, _, _ = await app_with_identity()
    headers = {"Authorization": f"Bearer {KEY}"}

    async with client(app) as http:
        await http.put("/credentials/github", headers=headers, json={"secret": "ghp-fine"})
        verified = await http.post("/credentials/github/check?verify=true", headers=headers)

    assert verified.json()["usable"] is True
    assert verified.json()["verified"] == "unknown"


@pytest.mark.asyncio
async def test_a_live_feature_is_refused_when_the_provider_refuses_its_git_credential() -> None:
    """The same refusal shape a missing credential gets, before anything is queued.

    Runs 190 and 191 were both accepted against an expired PAT. Each then spent its whole
    fault allowance on clones that could not succeed and was reported as a model-provider
    outage. A credential the provider has stopped accepting is, for a feature that has not
    started, the same condition as one that was never configured.
    """
    app, directory, _ = await app_with_identity()
    token = await token_for(directory, roles=(Role.OPERATOR.value,))
    headers = {"Authorization": f"Bearer {token}"}
    app.state.credential_verifier = _ScriptedVerifier(CredentialVerdict.REFUSED)

    async with client(app) as http:
        for provider in ("openai", "github"):
            await http.put(
                f"/credentials/{provider}", headers=headers, json={"secret": f"{provider}-key"}
            )
        refused = await http.post(
            "/features/start",
            headers=headers,
            json={**feature_payload(), "execution_mode": "live"},
        )
        # The same submission goes through once the provider accepts the credential, so the
        # refusal above is about the verdict and not about anything else in the payload.
        app.state.credential_verifier = _ScriptedVerifier(CredentialVerdict.ACCEPTED)
        accepted = await http.post(
            "/features/start",
            headers=headers,
            json={**feature_payload(), "execution_mode": "live"},
        )

    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert "GitHub refused the credential" in detail
    assert "Replace it in Settings" in detail
    assert accepted.status_code == 201


@pytest.mark.asyncio
async def test_a_stored_git_credential_carries_the_date_it_was_stored() -> None:
    """The one fact a refused clone needs and cannot look up for itself.

    "Authentication failed" names nothing. "The stored github credential (stored 2026-08-26)
    was refused" names the credential a person has to go and replace -- and the place that
    discovers the refusal is a `git clone` inside an adapter, which has no secret store to
    ask. So the date travels beside the value, and only for the stored one: a token supplied
    as a request header has no age this platform knows, and claiming the stored credential's
    date for it would name the wrong credential in the diagnosis.
    """
    _, _, store = await app_with_identity()
    await store.put(owner_id="someone", provider="github", secret="ghp-token")
    descriptor = await store.describe(owner_id="someone", provider="github")

    resolved = await stored_credentials(store, "someone")

    assert resolved.github_token == "ghp-token"
    assert resolved.github_token_stored_at == descriptor.created_at
    # A deployment that stores nothing reports no age rather than a wrong one.
    empty = await stored_credentials(None, "someone")
    assert empty.github_token is None
    assert empty.github_token_stored_at is None


# --------------------------------------------------------------------------------------
# Figma is a stored credential, on the terms every other credential has
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        # Verified against the live API on 2026-09-07 with a read-only personal access token:
        # `GET /v1/me` answers 200, a missing token answers 401 and an invalid one answers
        # 401. A scope the token does not hold answers 403 -- which is how the platform learns
        # a scope is missing, rather than by checking a hardcoded list of scope names Figma
        # has already renamed once.
        (200, CredentialVerdict.ACCEPTED),
        (401, CredentialVerdict.REFUSED),
        (403, CredentialVerdict.REFUSED),
        # Everything that is not the provider deciding. A 429 in particular: the live API rate
        # limits, and a rate limit read as "this token is dead" would refuse somebody's
        # submission over a credential that is fine.
        (404, CredentialVerdict.UNKNOWN),
        (429, CredentialVerdict.UNKNOWN),
        (500, CredentialVerdict.UNKNOWN),
        (503, CredentialVerdict.UNKNOWN),
    ],
)
@pytest.mark.asyncio
async def test_the_figma_verifier_reads_the_verdict_out_of_the_status(
    status: int, expected: CredentialVerdict
) -> None:
    """Three verdicts, decided by the number and never by the message.

    The provider's message is provider-owned text that may quote a request; the status is a
    number. `verdict_for_error` is the one classifier, and this asserts the verifier reaches
    it rather than inventing a second reading of the same statuses.
    """
    asked: list[str] = []

    async def probe(token: str) -> int:
        asked.append(token)
        return status

    verifier = FigmaCredentialVerifier(probe=probe)

    assert await verifier.verify(provider="figma", secret="figd-token") is expected
    assert asked == ["figd-token"]


@pytest.mark.asyncio
async def test_the_figma_verifier_answers_unknown_without_asking_anybody() -> None:
    """A foreign provider, a blank secret, and a transport that never answered.

    None of the three is a negative verdict, and each has to reach UNKNOWN without a network
    call being blamed for it: nothing in this platform refuses on UNKNOWN, and a verifier that
    turned "not asked" into "refused" would refuse submissions over credentials that work.
    """
    asked: list[str] = []

    async def probe(token: str) -> int:
        asked.append(token)
        raise TimeoutError("the provider never answered")

    verifier = FigmaCredentialVerifier(probe=probe)

    # Not its provider: the composite dispatches, but the guard survives because this class is
    # still installable on its own.
    assert await verifier.verify(provider="github", secret="ghp-token") is CredentialVerdict.UNKNOWN
    assert await verifier.verify(provider="figma", secret="   ") is CredentialVerdict.UNKNOWN
    assert asked == [], "nothing was asked about a foreign provider or a blank secret"
    # And a transport failure carries no status, so it classifies as no answer at all.
    assert await verifier.verify(provider="figma", secret="figd-token") is CredentialVerdict.UNKNOWN
    assert asked == ["figd-token"]


@pytest.mark.asyncio
async def test_the_composite_verifier_dispatches_and_answers_unknown_for_the_rest() -> None:
    """One slot, one verifier per provider, and no route-level special case.

    The app-state slot holds exactly one verifier, so before the composite existed every
    provider except GitHub answered UNKNOWN by construction. This asserts the dispatch: each
    provider reaches its own verifier and no other verifier is asked about it.
    """
    github = FigmaCredentialVerifier(probe=_never_called)
    figma_calls: list[str] = []

    async def figma_probe(token: str) -> int:
        figma_calls.append(token)
        return 200

    class _GitHubShaped:
        provider: ClassVar[str] = "github"

        def __init__(self) -> None:
            self.calls: list[str] = []

        async def verify(self, *, provider: str, secret: str) -> CredentialVerdict:
            self.calls.append(provider)
            return CredentialVerdict.REFUSED

    github_verifier = _GitHubShaped()
    composite = CompositeCredentialVerifier(
        github_verifier, FigmaCredentialVerifier(probe=figma_probe)
    )

    assert composite.providers == ("github", "figma")
    assert await composite.verify(provider="figma", secret="figd") is CredentialVerdict.ACCEPTED
    assert await composite.verify(provider="github", secret="ghp") is CredentialVerdict.REFUSED
    # A provider nobody registered is the same answer a deployment with no verifier gives.
    assert await composite.verify(provider="slack", secret="xoxb") is CredentialVerdict.UNKNOWN
    assert figma_calls == ["figd"]
    assert github_verifier.calls == ["github"]
    del github


async def _never_called(token: str) -> int:
    """Fail loudly if a verifier dials a provider it was not asked about."""
    raise AssertionError("this probe must never be called")


def test_two_verifiers_for_one_provider_is_a_construction_error() -> None:
    """Registering twice is a mistake with a silent failure mode, so it is refused loudly.

    Whichever one lost would never be asked, and "why does this check say UNKNOWN" would be
    answered by reading a constructor argument list.
    """
    with pytest.raises(ValueError, match="two verifiers registered"):
        CompositeCredentialVerifier(FigmaCredentialVerifier(), FigmaCredentialVerifier())


@pytest.mark.asyncio
async def test_figma_is_a_settings_row_and_never_a_feature_prerequisite() -> None:
    """Safety rule 2, asserted against the catalogue itself rather than against a symptom.

    `figma` is in `SUPPORTED_PROVIDERS` so it gets the storage path and the settings row
    `describe_all` renders for free -- and it is deliberately absent from
    `api.routes.CREDENTIAL_PROVIDERS`, which is the catalogue of what a feature *requires*. A
    deployment that has never heard of Figma must report no unmet prerequisite and refuse no
    submission, which is the trap the `slack` comment stated in advance.
    """
    assert "figma" in SUPPORTED_PROVIDERS
    assert "figma" not in dict(CREDENTIAL_PROVIDERS), (
        "a design citation is one author's attachment to one submission; requiring a Figma "
        "credential would refuse every submission on a deployment that cites no designs"
    )
    assert "figma" not in dict(required_providers_for_platforms(("openai", "anthropic"))), (
        "no feature requires a figma credential, whatever platform it runs on"
    )

    app, directory, _ = await app_with_identity()
    token = await token_for(directory, roles=(Role.OPERATOR.value,))
    headers = {"Authorization": f"Bearer {token}"}
    async with client(app) as http:
        for provider in ("openai", "github"):
            await http.put(
                f"/credentials/{provider}", headers=headers, json={"secret": f"{provider}-key"}
            )
        setup = await http.get("/setup", headers=headers)
        # The submission a deployment with no Figma credential sends is accepted, exactly as
        # it is today.
        started = await http.post("/features/start", headers=headers, json=feature_payload())

    body = setup.json()
    assert [item["provider"] for item in body["providers"] if item["provider"] == "figma"] == []
    assert body["credentials_ready"] is True
    assert started.status_code == 201


@pytest.mark.asyncio
async def test_a_fresh_owner_has_a_figma_row_that_says_it_is_unconfigured() -> None:
    """`describe_all` iterates `SUPPORTED_PROVIDERS`, so the settings page is complete.

    The row is what makes the credential enterable at all, and "not configured" is the honest
    state of a deployment that has never pasted a Figma token.
    """
    _, _, store = await app_with_identity()

    descriptors = {item.provider: item for item in await store.describe_all(owner_id="fresh")}

    assert "figma" in descriptors
    figma = descriptors["figma"]
    assert figma.configured is False
    assert figma.hint == ""
    # And it stores and describes on the same terms as every other provider, hint only.
    await store.put(owner_id="fresh", provider="figma", secret="figd-abcdefgh")
    stored = await store.describe(owner_id="fresh", provider="figma")
    assert stored.configured is True
    assert stored.hint == "efgh"
