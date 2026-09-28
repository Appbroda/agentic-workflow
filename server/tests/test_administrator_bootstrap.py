"""How the first account gets a password, and what the deployment does without a shared key.

The bootstrap is the one place a secret crosses from the environment into the database, so
its rules are the ones worth defending: it never overwrites an existing hash, it does nothing
at all when the account is not there, and it forces a change so that forgetting to remove the
variable is survivable rather than a permanent backdoor.

No password appears in this file as a literal. Each one is generated inline.
"""

from __future__ import annotations

import secrets
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from api.identity import Role
from main import bootstrap_administrator_password, create_app
from storage.user_store import InMemoryUserDirectory

ADMIN_EMAIL = "akhilesh@appbroda.com"


def a_password() -> str:
    """Return a fresh acceptable password, so no literal exists in this repository."""
    return f"pw-{secrets.token_urlsafe(16)}"


def client(app: Any) -> AsyncClient:
    """Return an HTTP client bound to the application under test."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def directory_with_admin() -> tuple[InMemoryUserDirectory, str]:
    """Return a directory holding the administrator's account with no password.

    Which is the state migration 0031 leaves: the row exists, `password_hash` is NULL, and
    `must_change_password` is already true.
    """
    directory = InMemoryUserDirectory()
    user = await directory.create_user(
        subject=ADMIN_EMAIL,
        display_name="Akhilesh Kumar Pandey",
        roles=(Role.ADMIN.value,),
        must_change_password=True,
    )
    return directory, user.user_id


@pytest.mark.asyncio
async def test_the_bootstrap_sets_a_password_on_an_account_that_has_none() -> None:
    """The first boot, and the only one that should change anything."""
    directory, user_id = await directory_with_admin()
    password = a_password()

    outcome = await bootstrap_administrator_password(
        directory, email=ADMIN_EMAIL, password=password
    )

    assert outcome == "set"
    authenticated = await directory.authenticate(subject=ADMIN_EMAIL, password=password)
    assert authenticated is not None
    assert authenticated.user_id == user_id


@pytest.mark.asyncio
async def test_the_bootstrap_never_overwrites_an_existing_password() -> None:
    """A container restart with a stale variable must not reset the administrator.

    Without this, `BOOTSTRAP_ADMIN_PASSWORD` left in the environment would be a working
    credential for ever, restored on every boot -- a permanent backdoor that looks exactly
    like a feature. Asserted as an effect: the password the administrator chose still works
    and the bootstrap's does not.
    """
    directory, user_id = await directory_with_admin()
    chosen = a_password()
    stale = a_password()
    await directory.set_password(user_id, password=chosen)

    outcome = await bootstrap_administrator_password(directory, email=ADMIN_EMAIL, password=stale)

    assert outcome == "already_set"
    assert await directory.authenticate(subject=ADMIN_EMAIL, password=chosen) is not None
    assert await directory.authenticate(subject=ADMIN_EMAIL, password=stale) is None


@pytest.mark.asyncio
async def test_the_bootstrap_forces_a_change() -> None:
    """A password the deployment's environment knows is a handover credential."""
    directory, user_id = await directory_with_admin()

    await bootstrap_administrator_password(directory, email=ADMIN_EMAIL, password=a_password())

    user = await directory.get(user_id)
    assert user is not None
    assert user.must_change_password is True


@pytest.mark.asyncio
async def test_the_bootstrap_does_nothing_without_a_configured_password() -> None:
    """The ordinary case after the first boot: the variable has been removed."""
    directory, user_id = await directory_with_admin()

    outcome = await bootstrap_administrator_password(directory, email=ADMIN_EMAIL, password=None)

    user = await directory.get(user_id)
    assert outcome == "not_configured"
    assert user is not None
    assert user.has_password is False


@pytest.mark.asyncio
async def test_the_bootstrap_does_not_create_the_account_it_cannot_find() -> None:
    """Migration 0031 owns the row; this owns the password, and only the password.

    A deployment whose migrations have not run yet, or whose `BOOTSTRAP_ADMIN_EMAIL` names
    somebody who is not the administrator, gets a warning and nothing else. Refusing to start
    over a recoverable misconfiguration would be worse than logging it.
    """
    directory = InMemoryUserDirectory()

    outcome = await bootstrap_administrator_password(
        directory, email=ADMIN_EMAIL, password=a_password()
    )

    assert outcome == "no_such_account"
    assert await directory.list_users() == []


@pytest.mark.asyncio
async def test_the_bootstrap_password_is_subject_to_the_length_rule() -> None:
    """One policy, enforced in `hash_password`, so no path around it exists."""
    from api.passwords import PasswordPolicyError

    directory, _ = await directory_with_admin()

    with pytest.raises(PasswordPolicyError):
        await bootstrap_administrator_password(directory, email=ADMIN_EMAIL, password="short")


# --- the deployment with no shared key ---------------------------------------------------


@pytest.mark.asyncio
async def test_the_api_serves_with_no_shared_platform_key() -> None:
    """The target state: `PLATFORM_API_KEY` unset, and per-user tokens are the only credential.

    Both halves matter. A per-user token authenticates; and a caller presenting *any* other
    bearer token is refused rather than accepted -- which is the failure an empty-string key
    would have caused, since `compare_digest("", "")` is true.
    """
    directory = InMemoryUserDirectory()
    user = await directory.create_user(
        subject="somebody@example.com", display_name="Somebody", roles=(Role.OPERATOR.value,)
    )
    issued = await directory.issue_token(user.user_id, label="laptop")
    app = create_app(platform_api_key=None, user_directory=directory)

    async with client(app) as http:
        health = await http.get("/healthz")
        mine = await http.get("/me", headers={"Authorization": f"Bearer {issued.token}"})
        empty = await http.get("/me", headers={"Authorization": "Bearer "})
        guessed = await http.get("/me", headers={"Authorization": "Bearer anything-at-all"})

    assert health.status_code == 200
    assert mine.status_code == 200
    assert mine.json()["actor_id"] == user.user_id
    assert empty.status_code == 401
    # 401, not 503. A directory is bound and it said no, so the deployment can check and did
    # -- and this is the ordinary answer for an unknown, expired or revoked credential.
    #
    # It was 503 while `PLATFORM_API_KEY` was required, because the 503 branch was only
    # reachable by a deployment that had neither a key nor a directory. With the key optional
    # and unset by design, that branch became every rejected credential's answer: it claimed
    # authentication was not configured when it was, and it never fired the client's 401
    # handling, so an expired session showed error boxes instead of the login page.
    assert guessed.status_code == 401


@pytest.mark.asyncio
async def test_a_deployment_with_neither_a_directory_nor_a_key_says_it_cannot_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one case 503 is still the right answer to.

    Nothing to check a credential against: no user directory, and no shared key. That is a
    different problem for whoever is looking at it than a credential that was wrong, and the
    distinction is the reason the status exists -- which is why it is asserted here rather
    than left to be inferred from the 401 above.

    `platform_api_key=None` alone does not guarantee that: `create_app` falls back to
    `os.environ["PLATFORM_API_KEY"]` when the parameter is unset, so a CI job (or any shell)
    that happens to export one would silently give this test a configured key and turn the
    503 under test into a 401 -- the failure this deployment's own CI hit.
    """
    monkeypatch.delenv("PLATFORM_API_KEY", raising=False)
    app = create_app(platform_api_key=None)

    async with client(app) as http:
        response = await http.get("/me", headers={"Authorization": "Bearer anything-at-all"})

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_a_blank_shared_key_leaves_the_deployment_with_no_shared_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace is absent, not a configured credential.

    Compose passes variables through as `NAME=${NAME:-}`, so an unset `PLATFORM_API_KEY`
    arrives as an empty string. Asserted against the authenticator's own state rather than
    through a request, because there is no bearer token a caller can send that would match an
    empty key -- so a request-level assertion would pass whether or not the normalisation
    happened, and prove nothing.
    """
    monkeypatch.setenv("PLATFORM_API_KEY", "   ")
    app = create_app(platform_api_key="")

    assert app.state.authenticator._platform_api_key is None  # noqa: SLF001


def test_a_blank_bootstrap_email_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose passes an absent variable through as an empty string, and it must not win.

    `docker-compose.yml` writes `BOOTSTRAP_ADMIN_EMAIL: ${BOOTSTRAP_ADMIN_EMAIL:-}`, so a
    deployment that does not set it in `.env` sends `""`. On a plain `str` field pydantic
    accepts that as a valid value and it *overrides* the declared default -- unlike a
    `SecretStr | None`, which at least ends up `None`.

    This is a regression test for a live incident, not a hypothetical. The first deploy of
    multi-user login resolved this field to `''`, so the bootstrap looked the administrator
    up by the empty string, found nobody, logged `no_such_account`, and set no password --
    on a deployment whose login page no longer accepts the shared key that was the only
    other way in.
    """
    from tests.test_config_and_prompts import configure_environment

    configure_environment(monkeypatch)
    monkeypatch.setenv("BOOTSTRAP_ADMIN_EMAIL", "")

    from configs.settings import load_settings

    assert load_settings().bootstrap_admin_email == "akhilesh@appbroda.com"


def test_a_configured_bootstrap_email_is_used_as_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback must not swallow a deployment whose administrator is somebody else."""
    from tests.test_config_and_prompts import configure_environment

    configure_environment(monkeypatch)
    monkeypatch.setenv("BOOTSTRAP_ADMIN_EMAIL", "someone.else@example.com")

    from configs.settings import load_settings

    assert load_settings().bootstrap_admin_email == "someone.else@example.com"


@pytest.mark.asyncio
async def test_settings_treat_a_blank_shared_key_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The validator, at the layer a deployment actually configures."""
    from tests.test_config_and_prompts import configure_environment

    configure_environment(monkeypatch)
    monkeypatch.setenv("PLATFORM_API_KEY", "   ")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", "")

    from configs.settings import load_settings

    settings = load_settings()

    assert settings.platform_api_key is None
    assert settings.bootstrap_admin_password is None
