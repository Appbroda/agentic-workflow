"""Logging in, logging out, and changing a password.

Its own router because `create_account_router` declares `Depends(authenticator)` at the
router level, and login has to be reachable by somebody who cannot yet authenticate. Adding
one unauthenticated route to that router would mean removing the router-level dependency and
restating it on nineteen endpoints -- which is the shape that lets a new endpoint forget.

A successful login mints a row in `platform_api_tokens` through the existing `issue_token`
and returns it once, exactly as `POST /users/{id}/tokens` already does. There is no JWT, no
session middleware and no cookie store: the authentication path stays
`PlatformAuthenticator` -> `ActorDirectory.resolve(token)` -> `Actor`, which already handles
expiry, revocation, disabled accounts and `last_used_at`, and already answers one
indistinguishable way for every kind of failure. A stateless token would duplicate all of
that and lose immediate revocation, which matters more here than it usually does: these
credentials authorise spending money on model calls and opening pull requests on other
people's repositories.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from api.account_routes import ActorResponse, actor_response
from api.auth import PlatformAuthenticator, current_actor
from api.identity import Actor
from api.passwords import MAX_PASSWORD_LENGTH, PasswordPolicyError
from services.login_rate_limit import LoginRateLimiter
from storage.models import TOKEN_KIND_SESSION

_logger = structlog.get_logger("api.auth")

# What every failed login says, whatever failed. One sentence for an unknown email, a wrong
# password, an account with no password and a disabled account: anything finer tells somebody
# which of their guesses was closest, which is the same discipline `resolve` keeps for tokens.
_LOGIN_REFUSED = "Invalid email or password."


class _PasswordModel(BaseModel):
    """A request body carrying a password, with whitespace left exactly as it was typed.

    `APIModel` strips leading and trailing whitespace from every string, which is right for
    a repository URL and wrong for a password: a password is an opaque byte sequence, and
    silently trimming one means the platform stores something the person did not type.
    Everything else `APIModel` does -- forbidding unexpected fields, refusing type coercion
    -- is kept.
    """

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=False)


class LoginRequest(_PasswordModel):
    """An email and a password."""

    email: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


class LoginResponse(BaseModel):
    """A session token, returned exactly once, and who it belongs to.

    The token is the only thing here that is a secret, and this is the only response that
    carries one. The platform stores its digest, so there is no endpoint that can show it
    again -- the same property `IssuedTokenResponse` has.
    """

    model_config = ConfigDict(extra="forbid")

    token: str
    expires_at: datetime | None
    actor: ActorResponse
    # Repeated outside `actor` because it is what the client branches on before it renders
    # anything, and reaching into a nested object for a routing decision reads as incidental.
    must_change_password: bool


class ChangePasswordRequest(_PasswordModel):
    """The password somebody has, and the one they want."""

    current_password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)
    new_password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


@dataclass(frozen=True, slots=True)
class LoginPolicy:
    """How long a session lasts and how fast somebody may guess.

    One object on application state rather than two, so an application assembled without a
    lifespan -- an isolated test app, a mock deployment -- has a complete, working policy
    rather than a scatter of `getattr` defaults at the call sites.
    """

    rate_limiter: LoginRateLimiter
    session_ttl_hours: int = 12


def create_auth_router(*, authenticator: PlatformAuthenticator) -> APIRouter:
    """Build `/auth/*`: one unauthenticated route and two that need a bearer token.

    Authentication is per-route here rather than router-wide, which is the whole reason this
    router exists separately. `login` takes no `authenticator` dependency; `logout` and
    `password` take it explicitly.
    """
    router = APIRouter(prefix="/auth", tags=["authentication"])

    @router.post("/login", response_model=LoginResponse)
    async def login(request_body: LoginRequest, request: Request) -> LoginResponse:
        """Exchange an email and a password for a bounded session token.

        The rate limit is checked and counted *before* the password is verified, so a refused
        attempt still spends part of the attacker's budget rather than being free.

        Nothing here logs the email beside the outcome, and nothing logs the password. A
        shared log line saying "login failed for alice@example.com" is an account
        enumeration oracle for anybody who can read logs, which is more people than can read
        the database.
        """
        directory = _require_directory(request)
        policy = _policy(request)
        verdict = await policy.rate_limiter.check(
            source=_source_address(request), subject=request_body.email.strip().lower()
        )
        if not verdict.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts. Try again later.",
                headers={"Retry-After": str(verdict.retry_after_seconds)},
            )
        user = await directory.authenticate(
            subject=request_body.email, password=request_body.password
        )
        if user is None:
            # One answer for every kind of failure. The KDF has already run against a
            # throwaway hash if the account did not exist, so absence is not a fast path.
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=_LOGIN_REFUSED)
        issued = await directory.issue_token(
            user.user_id,
            label="console session",
            expires_at=datetime.now(UTC) + timedelta(hours=policy.session_ttl_hours),
            kind=TOKEN_KIND_SESSION,
        )
        await directory.record_login(user.user_id)
        # The account id, never the email: an id in a log is a join key, an email is personal
        # data that then lives wherever logs go.
        _logger.info("login_succeeded", actor_id=user.user_id)
        actor = Actor(
            actor_id=user.user_id,
            display_name=user.display_name,
            authentication="user_token",
            roles=frozenset(user.roles),
        )
        return LoginResponse(
            token=issued.token,
            expires_at=issued.expires_at,
            actor=await actor_response(request, actor),
            must_change_password=bool(user.must_change_password),
        )

    @router.post(
        "/logout",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[Depends(authenticator)],
    )
    async def logout(request: Request) -> Response:
        """Revoke the token this request presented.

        Idempotent, and always `204`. A token that was already revoked, or that belongs to a
        directory this deployment does not have, is answered the same way: the caller asked
        for their credential to stop working, and afterwards it does not work. Telling them
        which of those happened would be a fact about the deployment they cannot act on.
        """
        directory = getattr(request.app.state, "user_directory", None)
        token = _presented_token(request)
        if directory is not None and token is not None:
            token_id = await directory.token_id_for(token)
            if token_id is not None:
                await directory.revoke_token(token_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.post(
        "/password",
        response_model=ActorResponse,
        dependencies=[Depends(authenticator)],
    )
    async def change_password(
        request_body: ChangePasswordRequest,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> ActorResponse:
        """Replace this identity's own password, having proved the current one.

        The current password is verified through the same `authenticate` a login uses, so
        there is one answer to "is this the password" and it cannot drift. Every other
        session this account had stops working; the one making the request does not, because
        being logged out by securing your own account is a surprise nobody benefits from.
        `kind='api'` tokens are untouched -- see `set_password`.
        """
        directory = _require_directory(request)
        user = await directory.get(actor.actor_id)
        if user is None:
            # The shared platform key, or an account that has since been deleted. Neither has
            # a password to change, and neither is a fault worth a 500.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This identity does not have a password.",
            )
        confirmed = await directory.authenticate(
            subject=user.subject, password=request_body.current_password
        )
        if confirmed is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="The current password is incorrect."
            )
        token = _presented_token(request)
        keep = None if token is None else await directory.token_id_for(token)
        try:
            await directory.set_password(
                user.user_id,
                password=request_body.new_password,
                must_change=False,
                keep_token_id=keep,
            )
        except PasswordPolicyError as error:
            # The message names the rule, never the value -- see `PasswordPolicyError`.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
            ) from error
        _logger.info("password_changed", actor_id=user.user_id)
        return await actor_response(request, actor)

    return router


def _policy(request: Request) -> LoginPolicy:
    """Return the deployment's login policy, or a permissive one built on nothing.

    An application assembled without a lifespan has no Redis to count in. It gets a limiter
    with no backend, which allows every attempt -- honest for an isolated test app, and never
    what a deployment runs, because `production_lifespan` installs a real one.
    """
    policy = getattr(request.app.state, "login_policy", None)
    if isinstance(policy, LoginPolicy):
        return policy
    return LoginPolicy(rate_limiter=LoginRateLimiter(None, attempts=10, window_seconds=900))


def _require_directory(request: Request) -> Any:
    """Return the user directory, or refuse the request as unavailable.

    `503` rather than `401`, matching `PlatformAuthenticator`: a deployment with no directory
    cannot check a password, which is a different problem for whoever is looking at it than a
    password that was wrong.
    """
    directory = getattr(request.app.state, "user_directory", None)
    if directory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This deployment does not have a user directory.",
        )
    return directory


def _presented_token(request: Request) -> str | None:
    """Return the bearer token this request carried, parsed from its own header.

    Read here rather than stashed on `request.state` by the authenticator: a raw credential
    on request state is a value every downstream dependency, error handler and middleware can
    reach, and only these two routes have any use for it.
    """
    header = request.headers.get("authorization") or ""
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def _source_address(request: Request) -> str | None:
    """Return the address to rate-limit against.

    `request.client.host` and nothing else. `X-Forwarded-For` is a client-supplied header,
    and trusting it would let an attacker choose their own rate-limit bucket per attempt --
    which is not a limit. A deployment behind a proxy that rewrites the peer address gets a
    per-proxy bucket, which is coarse but sound; the per-email counter is what carries the
    defence in that case.
    """
    client = request.client
    return None if client is None else client.host


__all__ = [
    "ChangePasswordRequest",
    "LoginPolicy",
    "LoginRequest",
    "LoginResponse",
    "create_auth_router",
]
