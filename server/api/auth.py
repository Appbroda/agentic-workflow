"""Bearer-token authentication for the HTTP control plane.

Two credentials are accepted, in one order. A token issued to a person resolves to that
person; the deployment's shared platform key resolves to an administrative identity. The
shared key is retained deliberately -- it is what the operator console, the browser flows and
existing deployments authenticate with, and removing it before per-user tokens are actually
issued would take the platform offline for no gain. It is now an identity with a name rather
than an absence of one.

Whatever authenticated, the request carries an ``Actor`` from here on, so an audit record and
an authorization check never have to ask how somebody proved who they were.
"""

from __future__ import annotations

import secrets
from typing import Annotated, Any, Protocol, cast

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.identity import Actor, Permission, WorkspaceScope, platform_admin

_bearer_scheme = HTTPBearer(auto_error=False)


class ActorDirectory(Protocol):
    """The one lookup authentication needs from the user directory."""

    async def resolve(self, token: str) -> Actor | None:
        """Return the identity a token proves, or nothing."""


class PlatformAuthenticator:
    """Resolve a request to an actor, or refuse it.

    Named as it always was because it is still the router's authentication dependency and
    every caller passes it the same way. What changed is what it produces: an identity,
    attached to the request, instead of a decision it kept to itself.
    """

    def __init__(
        self,
        platform_api_key: str | None,
        *,
        directory: ActorDirectory | None = None,
    ) -> None:
        """Store the expected shared key and the directory of individual identities."""
        self._platform_api_key = platform_api_key
        self._directory = directory

    def with_directory(self, directory: ActorDirectory) -> PlatformAuthenticator:
        """Return the same authenticator bound to a user directory.

        The routers are built before the database exists, so the directory arrives later.
        Rebinding rather than mutating keeps the constructed dependency honest about what it
        was given.
        """
        return PlatformAuthenticator(self._platform_api_key, directory=directory)

    def bind(self, directory: ActorDirectory) -> None:
        """Attach a directory to this instance after the routers were constructed.

        The router holds a reference to this object, so a replacement built later would
        never be reached. This is the seam where per-user identity becomes available to
        routes that were wired at import time.
        """
        self._directory = directory

    async def __call__(
        self,
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    ) -> Actor:
        """Reject missing, malformed, unavailable, or incorrect Bearer credentials."""
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise _authentication_error()

        if self._directory is not None:
            actor = await self._directory.resolve(credentials.credentials)
            if actor is not None:
                request.state.actor = actor
                return actor

        if self._platform_api_key is None:
            if self._directory is not None:
                # A directory is bound and it said no. The deployment *can* check, and did:
                # this is an unknown, expired, revoked or disabled credential, which is 401.
                #
                # This branch exists because `PLATFORM_API_KEY` became optional, and unset is
                # the target state. Without it, the 503 below was every rejected credential's
                # answer in a normal deployment -- claiming authentication was not configured
                # when it was, and never firing the client's 401 handling, so an expired
                # session showed error boxes for ever instead of a login page.
                raise _authentication_error()
            # Nothing to check against: no directory, and no shared key. Answering
            # "unavailable" rather than "unauthorized" says the deployment cannot check,
            # which is a different problem for whoever is looking at it.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Workflow API authentication is not configured.",
            )
        if not secrets.compare_digest(credentials.credentials, self._platform_api_key):
            raise _authentication_error()
        actor = platform_admin()
        request.state.actor = actor
        return actor


def current_actor(request: Request) -> Actor:
    """Return the identity the router's authentication dependency already established.

    Router-level dependencies run before endpoint ones, so by the time a route asks, this is
    set. It is read rather than re-derived so authentication happens exactly once per
    request and cannot answer differently in two places.
    """
    actor = getattr(request.state, "actor", None)
    if actor is None:  # pragma: no cover - unreachable behind the router dependency
        raise _authentication_error()
    return cast(Actor, actor)


def current_scope(request: Request) -> WorkspaceScope:
    """Return whose features this request may reach.

    Beside `current_actor` and derived from it, so ownership comes from what authentication
    established and never from a path parameter, a body field or a query string. There is no
    variant of this that takes an owner as an argument.
    """
    return WorkspaceScope.of(current_actor(request))


def requires(permission: Permission) -> Any:
    """Return a route dependency that refuses an actor lacking one permission.

    Route-level dependencies are solved before the endpoint's own, which is what this is
    for. A permission checked in the function body runs after every other dependency has
    been resolved, so a route whose collaborator answers "not configured" told an
    unauthorized caller that before telling them they were unauthorized.
    """

    async def guard(actor: Annotated[Actor, Depends(current_actor)]) -> None:
        """Refuse before anything else the route needs is even looked up."""
        require(actor, permission)

    return guard


def require(actor: Actor, permission: Permission) -> None:
    """Refuse an actor who does not hold a permission.

    Called in the route rather than trusted to the client. Which buttons a browser draws is
    a convenience for the person using it; it is not a security boundary, and treating it as
    one is the mistake this exists to prevent.
    """
    if not actor.may(permission):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"This identity is not permitted to {permission.value.replace(':', ' ')}.",
        )


def _authentication_error() -> HTTPException:
    """Return a uniform challenge response without revealing authentication details."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )


__all__ = [
    "ActorDirectory",
    "PlatformAuthenticator",
    "current_actor",
    "current_scope",
    "require",
    "requires",
]
