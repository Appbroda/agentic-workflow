"""Ask a provider whether a stored credential is still accepted, and store nothing.

``/credentials/{provider}/check`` proves a stored credential can be *opened*: it resolves the
value through the secret store and reports whether the bytes came back. That is the right
answer to "did this deployment's encryption key change", and it was the wrong answer to the
question AB-Feature-190 needed. The GitHub PAT stored on 2026-08-26 expired at 2026-09-02
00:00 UTC; the check reported it usable for every hour of the day that followed, while every
authenticated clone was being refused, and two features died before a person correlated clone
timestamps by hand.

So verification is a second question, asked explicitly, and it is separate from the first for
the two reasons that endpoint's own docstring gives: it sends a credential over the network,
and it can fail for reasons that say nothing about the credential.

That second reason is why the verdict has three values rather than two. ``UNKNOWN`` is what a
provider this platform could not reach returns, and nothing is ever refused on an ``UNKNOWN``
-- a submission blocked because GitHub was slow would be a worse outage than the one this
exists to diagnose. Only ``REFUSED``, which means the provider answered and said no, is acted
on.

Nothing here logs, persists or returns the credential, or anything derived from it. The
verdict is the whole output.

One verifier per provider, and one composite that dispatches between them. The app-state slot
holds a single verifier, so before ``CompositeCredentialVerifier`` existed every provider
except GitHub answered ``UNKNOWN`` by construction -- safe, and also not an answer. The
composite is the dispatch, and it is here rather than in the route so that adding a provider
is registering a verifier and nothing else.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable
from enum import StrEnum
from typing import Any, ClassVar, Protocol

import structlog

_LOGGER = structlog.get_logger(__name__)

# The HTTP statuses that mean the provider decided, rather than failed. Conservative for the
# reason `_AUTHENTICATION_REFUSAL_MARKERS` is: a missed refusal costs a diagnosis, and a false
# refusal costs somebody a submission over a credential that is fine.
_REFUSAL_STATUSES = frozenset({401, 403})


class CredentialVerdict(StrEnum):
    """What a provider said about one credential, or that it did not say."""

    # The provider answered, and the credential is live.
    ACCEPTED = "accepted"
    # The provider answered, and the answer was no. This is the only verdict anything acts on.
    REFUSED = "refused"
    # Nobody asked, the provider could not be reached, or its answer could not be read. Never
    # a reason to refuse anything: it is the absence of an answer, not a negative one.
    UNKNOWN = "unknown"


class CredentialVerifier(Protocol):
    """Ask one provider whether it still accepts a credential."""

    async def verify(self, *, provider: str, secret: str) -> CredentialVerdict:
        """Return the provider's verdict on this credential, or UNKNOWN."""


class SingleProviderCredentialVerifier(Protocol):
    """A verifier that names the one provider it answers for, so it can be dispatched to.

    Deliberately a second protocol rather than a field on ``CredentialVerifier``. The
    app-state slot holds *a* verifier, and the composite below is one; a caller installing a
    single verifier directly -- which every test that scripts a verdict does -- keeps working
    against the narrower contract it always had.
    """

    provider: ClassVar[str]

    async def verify(self, *, provider: str, secret: str) -> CredentialVerdict:
        """Return the provider's verdict on this credential, or UNKNOWN."""


def verdict_for_error(error: BaseException) -> CredentialVerdict:
    """Classify a provider client's exception as a refusal or as no answer at all.

    Read from the status the client attached, never from its message. A provider SDK's
    exception text is provider-owned and may quote a request; the status is a number.
    """
    status = getattr(error, "status", None)
    if isinstance(status, int) and status in _REFUSAL_STATUSES:
        return CredentialVerdict.REFUSED
    return CredentialVerdict.UNKNOWN


class GitHubCredentialVerifier:
    """Ask GitHub for the identity a token authenticates as, and read only the outcome.

    ``provider`` is what the composite dispatches on; the guard inside ``verify`` stays,
    because this class is still installable on its own and must answer UNKNOWN rather than
    dial GitHub about somebody else's credential.

    ``GET /user`` deliberately: it is the cheapest authenticated call GitHub has, it needs no
    scope beyond the token existing, and it names no repository -- so this cannot become a
    check that passes or fails depending on which repositories a feature happens to target.

    Synchronous underneath, because the client this platform already depends on is. Run on a
    worker thread so a slow provider cannot block the event loop that is answering everybody
    else's requests.
    """

    provider: ClassVar[str] = "github"

    def __init__(self, *, client_factory: Callable[[str], Any] | None = None) -> None:
        """Bind the client builder, defaulting to the lazily-imported live one."""
        self._client_factory = client_factory or _github_client

    async def verify(self, *, provider: str, secret: str) -> CredentialVerdict:
        """Return GitHub's verdict on this token, or UNKNOWN for anything else."""
        if provider != "github" or not secret.strip():
            return CredentialVerdict.UNKNOWN
        return await asyncio.to_thread(self._verify, secret)

    def _verify(self, secret: str) -> CredentialVerdict:
        """Make the one call, and turn whatever came back into a verdict."""
        try:
            login = self._client_factory(secret).get_user().login
        except Exception as error:  # noqa: BLE001 - every failure is a verdict, never a raise
            verdict = verdict_for_error(error)
            # The type name and the status only. Both are library-owned symbols; the
            # provider's message is not, and neither is the token.
            _LOGGER.info(
                "credential_verification_failed",
                provider="github",
                verdict=verdict.value,
                error_type=type(error).__name__,
                status=getattr(error, "status", None),
            )
            return verdict
        # An answer with no identity in it is an answer this platform cannot read, which is
        # not the same as a refusal.
        return CredentialVerdict.ACCEPTED if str(login).strip() else CredentialVerdict.UNKNOWN


class FigmaCredentialVerifier:
    """Ask Figma whether it still accepts one personal access token, and read only the status.

    ``GET /v1/me`` deliberately, for ``GET /user``'s reason one provider over: it is the
    cheapest authenticated call Figma has and it **names no file**, so this cannot become a
    check that passes or fails depending on which design a citation happens to point at.

    Verified against the live API on 2026-09-07 with a read-only personal access token:
    ``GET /v1/me`` answered ``200``, a missing token answered ``401 Missing credentials`` and
    an invalid one ``401 Invalid token``. Both refusals therefore land on
    ``verdict_for_error``'s ``_REFUSAL_STATUSES``. Nothing about the token's *scopes* is
    checked here, and deliberately so: Figma's scope names have changed at least once, a
    hardcoded list would refuse a token that works, and the platform's answer to "is this
    scope missing" is the provider's own 403 through the ``REFUSED`` path.

    Stricter than the GitHub verifier about what it reads: only the HTTP status, never the
    response body. ``/v1/me`` answers with an account's id, handle, email and avatar URL, and
    none of that is any of this platform's business -- a verdict is the whole output.

    ``httpx`` rather than an SDK, in ``adapters.slack_adapter``'s style: client created per
    call, import deferred to the first call so a run that verifies nothing never initializes
    network-capable code, and one timeout constant.
    """

    provider: ClassVar[str] = "figma"

    def __init__(self, *, probe: Callable[[str], Any] | None = None) -> None:
        """Bind the account probe, defaulting to the lazily-imported live one."""
        self._probe = probe or _figma_account_status

    async def verify(self, *, provider: str, secret: str) -> CredentialVerdict:
        """Return Figma's verdict on this token, or UNKNOWN for anything else."""
        if provider != self.provider or not secret.strip():
            return CredentialVerdict.UNKNOWN
        try:
            status = int(await self._probe(secret))
        except Exception as error:  # noqa: BLE001 - every failure is a verdict, never a raise
            return self._log_and_return(verdict_for_error(error), error=error)
        if 200 <= status < 300:
            return CredentialVerdict.ACCEPTED
        # Classified from the status through the one classifier, so a 401 or 403 is a refusal
        # and a 404, a 429 or a 5xx is the absence of an answer rather than a negative one.
        refusal = _FigmaStatus(status)
        return self._log_and_return(verdict_for_error(refusal), error=refusal)

    def _log_and_return(
        self, verdict: CredentialVerdict, *, error: BaseException
    ) -> CredentialVerdict:
        """Record the provider, the verdict, the error type and the status -- nothing else."""
        _LOGGER.info(
            "credential_verification_failed",
            provider=self.provider,
            verdict=verdict.value,
            error_type=type(error).__name__,
            status=getattr(error, "status", None),
        )
        return verdict


class CompositeCredentialVerifier:
    """Dispatch one credential to the verifier that owns its provider, or answer UNKNOWN.

    The app-state slot holds exactly one verifier, and until this existed that one was
    ``GitHubCredentialVerifier`` -- so every other provider's ``/check?verify=true`` answered
    UNKNOWN by construction, which is safe but is also not an answer. This is the dispatch,
    built once so that adding a provider is registering a verifier and nothing else: no
    route-level special case, and no second place that knows which providers can be verified.

    A provider nobody registered answers UNKNOWN, which is the same answer a deployment with
    no verifier at all gives. That is what keeps this an added diagnosis rather than a new
    prerequisite: nothing anywhere refuses on UNKNOWN.
    """

    def __init__(self, *verifiers: SingleProviderCredentialVerifier) -> None:
        """Index the given verifiers by the provider each one names."""
        self._by_provider: dict[str, SingleProviderCredentialVerifier] = {}
        for verifier in verifiers:
            if verifier.provider in self._by_provider:
                msg = f"two verifiers registered for provider: {verifier.provider}"
                raise ValueError(msg)
            self._by_provider[verifier.provider] = verifier

    @property
    def providers(self) -> tuple[str, ...]:
        """The providers this deployment can actually ask, in registration order."""
        return tuple(self._by_provider)

    async def verify(self, *, provider: str, secret: str) -> CredentialVerdict:
        """Ask the one verifier that owns this provider, or report that nobody was asked."""
        verifier = self._by_provider.get(provider)
        if verifier is None:
            return CredentialVerdict.UNKNOWN
        return await verifier.verify(provider=provider, secret=secret)


class _FigmaStatus(Exception):
    """One non-success HTTP status, carried where ``verdict_for_error`` reads it from.

    An exception rather than a plain integer because the classification must have exactly one
    implementation. ``verdict_for_error`` reads ``.status`` and never a message, so this type
    deliberately has no message worth reading.
    """

    def __init__(self, status: int) -> None:
        """Carry the status and nothing the provider authored."""
        super().__init__(f"figma answered {status}")
        self.status = status


# Figma's API root, and the one timeout every call this module makes is bounded by.
_FIGMA_API_BASE = "https://api.figma.com"
_REQUEST_TIMEOUT_SECONDS = 10.0


async def _figma_account_status(token: str) -> int:
    """Ask ``GET /v1/me`` and return its HTTP status, reading none of the response body."""
    httpx = importlib.import_module("httpx")
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.get(
            f"{_FIGMA_API_BASE}/v1/me",
            headers={"X-Figma-Token": token},
        )
    return int(response.status_code)


def _github_client(token: str) -> Any:
    """Build the live client lazily, so importing this module reaches no network code."""
    github_module = importlib.import_module("github")
    return github_module.Github(auth=github_module.Auth.Token(token))


__all__ = [
    "CompositeCredentialVerifier",
    "CredentialVerdict",
    "CredentialVerifier",
    "FigmaCredentialVerifier",
    "GitHubCredentialVerifier",
    "SingleProviderCredentialVerifier",
    "verdict_for_error",
]
