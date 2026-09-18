"""Proving who you are with a password, and what stops working when you change it.

Every password in this file is generated inline. Nothing here, and nothing anywhere else in
the repository, contains a password, a hash, or a placeholder shaped like one -- the
deployment's bootstrap password arrives from the environment and appears in no file.

The properties being defended: a refused login says the same thing whatever was wrong with
it, a session ends when its holder says so, and changing a password stops the sessions it
opened without stopping the automation tokens an administrator issued separately.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from api.auth_routes import LoginPolicy
from api.identity import Role
from api.passwords import (
    MIN_PASSWORD_LENGTH,
    PasswordPolicyError,
    dummy_hash,
    hash_password,
    verify_password,
)
from main import create_app
from services.login_rate_limit import LoginRateLimiter
from storage.models import TOKEN_KIND_API, TOKEN_KIND_SESSION
from storage.user_store import InMemoryUserDirectory, SubjectAlreadyRegisteredError


def a_password() -> str:
    """Return a fresh acceptable password, so no literal exists in this repository."""
    return f"pw-{secrets.token_urlsafe(16)}"


async def app_with_directory() -> tuple[Any, InMemoryUserDirectory]:
    """Build an application whose only credential is a per-user token."""
    directory = InMemoryUserDirectory()
    return create_app(platform_api_key="login-tests-key", user_directory=directory), directory


def client(app: Any) -> AsyncClient:
    """Return an HTTP client bound to the application under test."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


class _CountingLimiter:
    """A rate-limit backend that counts in a dict, so the window can be exercised."""

    def __init__(self) -> None:
        """Start with nothing counted."""
        self.counts: dict[str, int] = {}
        self.expiries: dict[str, int] = {}

    async def incr(self, name: str) -> int:
        """Increment one counter."""
        self.counts[name] = self.counts.get(name, 0) + 1
        return self.counts[name]

    async def expire(self, name: str, seconds: int) -> bool:
        """Record the lifetime the limiter asked for."""
        self.expiries[name] = seconds
        return True


# --- the hashing itself -------------------------------------------------------------------


def test_a_password_verifies_against_its_own_hash() -> None:
    """The round trip, which everything else depends on."""
    password = a_password()

    assert verify_password(password, hash_password(password)) is True


def test_a_different_password_does_not_verify() -> None:
    """The other half of the round trip."""
    assert verify_password(a_password(), hash_password(a_password())) is False


def test_two_hashes_of_one_password_differ() -> None:
    """A fresh salt per password, so identical passwords are not identical rows.

    Without this, the database says which accounts share a password -- which is a map of
    which accounts to attack next once one of them falls.
    """
    password = a_password()

    assert hash_password(password) != hash_password(password)


def test_a_stored_hash_carries_its_own_parameters() -> None:
    """The work factor can be raised later by rewriting rows, not by a second migration."""
    encoded = hash_password(a_password())

    label, n, r, p, _salt, _digest = encoded.split("$")
    assert label == "scrypt"
    assert (int(n), int(r), int(p)) == (2**15, 8, 1)


def test_a_short_password_is_refused_before_any_work_is_done() -> None:
    """The length rule lives in `hash_password`, so every path that sets one obeys it."""
    with pytest.raises(PasswordPolicyError):
        hash_password("x" * (MIN_PASSWORD_LENGTH - 1))


def test_an_enormous_password_is_refused() -> None:
    """A caller must not be able to choose how much scrypt this process runs."""
    with pytest.raises(PasswordPolicyError):
        hash_password("x" * 1025)


def test_an_absent_hash_matches_nothing() -> None:
    """An account that cannot password-login must not match an empty password."""
    assert verify_password("", None) is False
    assert verify_password(a_password(), None) is False


def test_an_unreadable_hash_matches_nothing() -> None:
    """A row this build cannot parse is a refusal, not a crash and not a match."""
    assert verify_password(a_password(), "not-an-encoded-hash") is False
    assert verify_password(a_password(), "argon2$v=19$m=1,t=1,p=1$c2FsdA$aGFzaA") is False


def test_the_dummy_hash_matches_nothing_a_caller_can_send() -> None:
    """What an unknown email is verified against must not be guessable."""
    assert verify_password("", dummy_hash()) is False


# --- the directory ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_subject_is_stored_and_matched_lowercased() -> None:
    """Two spellings of one email are one account, or they are two people's workspaces."""
    directory = InMemoryUserDirectory()
    password = a_password()
    created = await directory.create_user(
        subject="  Person@Example.COM ",
        display_name="A person",
        roles=(Role.OPERATOR.value,),
        password=password,
    )

    assert created.subject == "person@example.com"
    assert await directory.find_by_subject("PERSON@example.com") is not None
    authenticated = await directory.authenticate(subject="Person@Example.com", password=password)
    assert authenticated is not None
    assert authenticated.user_id == created.user_id


@pytest.mark.asyncio
async def test_a_duplicate_subject_is_its_own_refusal() -> None:
    """`409` and `400` are different answers, so they are different exception types."""
    directory = InMemoryUserDirectory()
    await directory.create_user(
        subject="taken@example.com", display_name="First", roles=(Role.OPERATOR.value,)
    )

    with pytest.raises(SubjectAlreadyRegisteredError):
        await directory.create_user(
            subject="Taken@example.com", display_name="Second", roles=(Role.OPERATOR.value,)
        )


@pytest.mark.asyncio
async def test_an_account_with_no_password_cannot_password_login() -> None:
    """A federated account, and the administrator before the bootstrap runs."""
    directory = InMemoryUserDirectory()
    await directory.create_user(
        subject="federated@example.com", display_name="Asserted", roles=(Role.OPERATOR.value,)
    )

    assert await directory.authenticate(subject="federated@example.com", password="") is None


@pytest.mark.asyncio
async def test_a_disabled_account_cannot_password_login() -> None:
    """Disabling somebody has to stop them getting in, not just stop their old tokens."""
    directory = InMemoryUserDirectory()
    password = a_password()
    user = await directory.create_user(
        subject="leaver@example.com",
        display_name="A leaver",
        roles=(Role.OPERATOR.value,),
        password=password,
    )
    await directory.set_disabled(user.user_id, disabled=True)

    assert await directory.authenticate(subject="leaver@example.com", password=password) is None


@pytest.mark.asyncio
async def test_changing_a_password_ends_sessions_and_spares_api_tokens() -> None:
    """The one thing the `kind` column exists for, asserted as an effect on each token."""
    directory = InMemoryUserDirectory()
    user = await directory.create_user(
        subject="rotator@example.com",
        display_name="A rotator",
        roles=(Role.OPERATOR.value,),
        password=a_password(),
    )
    session = await directory.issue_token(user.user_id, label="s", kind=TOKEN_KIND_SESSION)
    automation = await directory.issue_token(user.user_id, label="a", kind=TOKEN_KIND_API)

    await directory.set_password(user.user_id, password=a_password())

    assert await directory.resolve(session.token) is None
    assert await directory.resolve(automation.token) is not None


@pytest.mark.asyncio
async def test_a_password_change_can_spare_the_session_making_it() -> None:
    """Securing your own account must not log you out of it."""
    directory = InMemoryUserDirectory()
    user = await directory.create_user(
        subject="keeper@example.com",
        display_name="A keeper",
        roles=(Role.OPERATOR.value,),
        password=a_password(),
    )
    keeping = await directory.issue_token(user.user_id, label="k", kind=TOKEN_KIND_SESSION)
    other = await directory.issue_token(user.user_id, label="o", kind=TOKEN_KIND_SESSION)

    await directory.set_password(
        user.user_id, password=a_password(), keep_token_id=keeping.token_id
    )

    assert await directory.resolve(keeping.token) is not None
    assert await directory.resolve(other.token) is None


@pytest.mark.asyncio
async def test_the_last_enabled_administrator_is_counted() -> None:
    """What the lockout refusal reads. A disabled admin does not hold the deployment open."""
    directory = InMemoryUserDirectory()
    first = await directory.create_user(
        subject="one@example.com", display_name="One", roles=(Role.ADMIN.value,)
    )
    await directory.create_user(
        subject="two@example.com", display_name="Two", roles=(Role.OPERATOR.value,)
    )

    assert await directory.enabled_administrator_count() == 1
    await directory.set_disabled(first.user_id, disabled=True)
    assert await directory.enabled_administrator_count() == 0


# --- the endpoints -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_login_returns_a_token_that_authenticates() -> None:
    """The whole point: an administrator created by hand can now get in from a screen."""
    app, directory = await app_with_directory()
    password = a_password()
    await directory.create_user(
        subject="operator@example.com",
        display_name="An operator",
        roles=(Role.OPERATOR.value,),
        password=password,
    )

    async with client(app) as http:
        login = await http.post(
            "/auth/login", json={"email": "operator@example.com", "password": password}
        )
        token = login.json()["token"]
        me = await http.get("/me", headers={"Authorization": f"Bearer {token}"})

    assert login.status_code == 200
    assert login.json()["actor"]["subject"] == "operator@example.com"
    assert me.status_code == 200
    assert me.json()["display_name"] == "An operator"


@pytest.mark.asyncio
async def test_a_login_records_when_it_happened() -> None:
    """Requested for the administrator's user list, and it must survive the login path."""
    app, directory = await app_with_directory()
    password = a_password()
    user = await directory.create_user(
        subject="seen@example.com",
        display_name="Seen",
        roles=(Role.OPERATOR.value,),
        password=password,
    )

    async with client(app) as http:
        await http.post("/auth/login", json={"email": "seen@example.com", "password": password})

    refreshed = await directory.get(user.user_id)
    assert refreshed is not None
    assert refreshed.last_login_at is not None


@pytest.mark.asyncio
async def test_every_kind_of_failed_login_is_the_same_answer() -> None:
    """An unknown email, a wrong password and a disabled account are indistinguishable.

    Asserted on the body as well as the status: a distinct sentence is an enumeration oracle
    just as surely as a distinct status code is.
    """
    app, directory = await app_with_directory()
    password = a_password()
    await directory.create_user(
        subject="known@example.com",
        display_name="Known",
        roles=(Role.OPERATOR.value,),
        password=password,
    )
    without = await directory.create_user(
        subject="nopassword@example.com", display_name="No password", roles=(Role.OPERATOR.value,)
    )
    off = await directory.create_user(
        subject="off@example.com",
        display_name="Off",
        roles=(Role.OPERATOR.value,),
        password=password,
    )
    await directory.set_disabled(off.user_id, disabled=True)
    assert without.has_password is False

    async with client(app) as http:
        answers = [
            await http.post(
                "/auth/login", json={"email": "absent@example.com", "password": password}
            ),
            await http.post(
                "/auth/login", json={"email": "known@example.com", "password": a_password()}
            ),
            await http.post(
                "/auth/login", json={"email": "nopassword@example.com", "password": password}
            ),
            await http.post("/auth/login", json={"email": "off@example.com", "password": password}),
        ]

    assert [item.status_code for item in answers] == [401, 401, 401, 401]
    assert len({item.text for item in answers}) == 1


@pytest.mark.asyncio
async def test_a_session_token_expires() -> None:
    """A browser tab somebody walked away from stops being a credential."""
    app, directory = await app_with_directory()
    user = await directory.create_user(
        subject="expiring@example.com",
        display_name="Expiring",
        roles=(Role.OPERATOR.value,),
        password=a_password(),
    )
    expired = await directory.issue_token(
        user.user_id,
        label="stale",
        kind=TOKEN_KIND_SESSION,
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )

    async with client(app) as http:
        response = await http.get("/me", headers={"Authorization": f"Bearer {expired.token}"})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_logging_out_stops_the_token_that_asked() -> None:
    """Asserted as an effect on the credential, not as the status of the logout call."""
    app, directory = await app_with_directory()
    password = a_password()
    await directory.create_user(
        subject="goer@example.com",
        display_name="A goer",
        roles=(Role.OPERATOR.value,),
        password=password,
    )

    async with client(app) as http:
        login = await http.post(
            "/auth/login", json={"email": "goer@example.com", "password": password}
        )
        token = login.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        first = await http.post("/auth/logout", headers=headers)
        second = await http.post("/auth/logout", headers=headers)
        after = await http.get("/me", headers=headers)

    assert first.status_code == 204
    # Idempotent: the second call presents a token that no longer authenticates, so the
    # router refuses it. What matters is that the credential is dead, which `after` proves.
    assert second.status_code in {204, 401}
    assert after.status_code == 401


@pytest.mark.asyncio
async def test_changing_a_password_over_http_keeps_this_session_and_kills_the_others() -> None:
    """The endpoint's effect on three credentials at once."""
    app, directory = await app_with_directory()
    password = a_password()
    user = await directory.create_user(
        subject="changer@example.com",
        display_name="A changer",
        roles=(Role.OPERATOR.value,),
        password=password,
    )
    other = await directory.issue_token(user.user_id, label="other", kind=TOKEN_KIND_SESSION)
    automation = await directory.issue_token(user.user_id, label="ci", kind=TOKEN_KIND_API)
    replacement = a_password()

    async with client(app) as http:
        login = await http.post(
            "/auth/login", json={"email": "changer@example.com", "password": password}
        )
        mine = login.json()["token"]
        changed = await http.post(
            "/auth/password",
            headers={"Authorization": f"Bearer {mine}"},
            json={"current_password": password, "new_password": replacement},
        )
        still_me = await http.get("/me", headers={"Authorization": f"Bearer {mine}"})
        old_session = await http.get("/me", headers={"Authorization": f"Bearer {other.token}"})
        script = await http.get("/me", headers={"Authorization": f"Bearer {automation.token}"})
        with_new = await http.post(
            "/auth/login", json={"email": "changer@example.com", "password": replacement}
        )
        with_old = await http.post(
            "/auth/login", json={"email": "changer@example.com", "password": password}
        )

    assert changed.status_code == 200
    assert still_me.status_code == 200
    assert old_session.status_code == 401
    assert script.status_code == 200
    assert with_new.status_code == 200
    assert with_old.status_code == 401


@pytest.mark.asyncio
async def test_a_password_change_needs_the_current_password() -> None:
    """A stolen session must not be enough to take the account over permanently."""
    app, directory = await app_with_directory()
    password = a_password()
    await directory.create_user(
        subject="guarded@example.com",
        display_name="Guarded",
        roles=(Role.OPERATOR.value,),
        password=password,
    )

    async with client(app) as http:
        login = await http.post(
            "/auth/login", json={"email": "guarded@example.com", "password": password}
        )
        token = login.json()["token"]
        refused = await http.post(
            "/auth/password",
            headers={"Authorization": f"Bearer {token}"},
            json={"current_password": a_password(), "new_password": a_password()},
        )
        # The original password still works, which is the effect that matters.
        again = await http.post(
            "/auth/login", json={"email": "guarded@example.com", "password": password}
        )

    assert refused.status_code == 403
    assert again.status_code == 200


@pytest.mark.asyncio
async def test_a_password_change_enforces_the_length_rule() -> None:
    """The policy is not something the login route restates and can get wrong."""
    app, directory = await app_with_directory()
    password = a_password()
    await directory.create_user(
        subject="brief@example.com",
        display_name="Brief",
        roles=(Role.OPERATOR.value,),
        password=password,
    )

    async with client(app) as http:
        login = await http.post(
            "/auth/login", json={"email": "brief@example.com", "password": password}
        )
        response = await http.post(
            "/auth/password",
            headers={"Authorization": f"Bearer {login.json()['token']}"},
            json={"current_password": password, "new_password": "x" * 4},
        )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_login_is_reachable_without_authentication() -> None:
    """Its own router exists for this. A body it cannot parse is 422, never 401."""
    app, _ = await app_with_directory()

    async with client(app) as http:
        response = await http.post("/auth/login", json={"email": "", "password": ""})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_login_is_mounted_under_the_api_prefix_too() -> None:
    """The browser calls `/api`. A router mounted once works in development only."""
    app, directory = await app_with_directory()
    password = a_password()
    await directory.create_user(
        subject="prefixed@example.com",
        display_name="Prefixed",
        roles=(Role.OPERATOR.value,),
        password=password,
    )

    async with client(app) as http:
        response = await http.post(
            "/api/auth/login", json={"email": "prefixed@example.com", "password": password}
        )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_the_rate_limiter_refuses_and_says_when_to_come_back() -> None:
    """Counted before the password is checked, so a refused attempt is not free."""
    app, directory = await app_with_directory()
    backend = _CountingLimiter()
    app.state.login_policy = LoginPolicy(
        rate_limiter=LoginRateLimiter(backend, attempts=3, window_seconds=60)
    )
    password = a_password()
    await directory.create_user(
        subject="sprayed@example.com",
        display_name="Sprayed",
        roles=(Role.OPERATOR.value,),
        password=password,
    )

    async with client(app) as http:
        for _ in range(3):
            await http.post(
                "/auth/login", json={"email": "sprayed@example.com", "password": a_password()}
            )
        # The fourth attempt is over budget, and a correct password does not rescue it:
        # that is what makes this a limit rather than a hint.
        limited = await http.post(
            "/auth/login", json={"email": "sprayed@example.com", "password": password}
        )

    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "60"
    assert backend.expiries["login:subject:sprayed@example.com"] == 60


@pytest.mark.asyncio
async def test_the_rate_limit_window_is_fixed_rather_than_sliding() -> None:
    """Refreshing the lifetime on every attempt locks an address out permanently."""
    backend = _CountingLimiter()
    limiter = LoginRateLimiter(backend, attempts=10, window_seconds=30)

    await limiter.check(source="10.0.0.1", subject="a@example.com")
    backend.expiries.clear()
    await limiter.check(source="10.0.0.1", subject="a@example.com")

    assert backend.expiries == {}


@pytest.mark.asyncio
async def test_a_broken_rate_limit_backend_does_not_lock_the_deployment_out() -> None:
    """A counter that cannot be reached must not become an outage for everybody."""

    class _Broken:
        async def incr(self, name: str) -> int:
            raise RuntimeError("redis is unreachable")

        async def expire(self, name: str, seconds: int) -> bool:  # pragma: no cover
            raise RuntimeError("redis is unreachable")

    verdict = await LoginRateLimiter(_Broken(), attempts=1, window_seconds=60).check(
        source="10.0.0.1", subject="a@example.com"
    )

    assert verdict.allowed is True


@pytest.mark.asyncio
async def test_a_deployment_with_no_directory_says_it_cannot_check() -> None:
    """`503` rather than `401`: "cannot check" is a different problem from "wrong"."""
    app = create_app(platform_api_key="no-directory")

    async with client(app) as http:
        response = await http.post(
            "/auth/login", json={"email": "anyone@example.com", "password": a_password()}
        )

    assert response.status_code == 503
