"""Ask GitHub what one token can actually reach, and answer only from what GitHub said.

``GitHubCredentialVerifier`` answers one question -- does GitHub still accept this token -- and
deliberately answers it with ``GET /user``, which names no repository and needs no scope. That
is the right shape for "has this PAT expired", and it is silent on the failure people actually
hit: a token GitHub accepts, that cannot push to anything the platform was pointed at. The
first evidence of that was a run dying at push time, hours after somebody pasted the token.

So this is the second question, and it is asked from the same place at the same moment: what
does this token reach, and is any of it writable. Two callers need the answer -- the credential
form, which refuses a token GitHub has answered "no" about, and the repository picker, which
offers the repositories a token can see instead of asking somebody to paste a URL and find out
later whether it was reachable.

**What is trustworthy here, and what is not.** Every field is copied from a GitHub response;
nothing is inferred from the token's text. But the two token kinds do not answer equally well:

* A **classic** PAT reports its scopes in the ``X-OAuth-Scopes`` response header, and that
  header is authoritative. Without ``repo`` or ``public_repo``, GitHub will refuse the pushes
  this platform makes -- and private repositories are not even listed, so the listing itself
  narrows correctly. This is the one scope rule enforced below, and it is enforced because the
  provider states it, not because a scope name was guessed at.
* A **fine-grained** PAT sends no scope header at all, and the ``permissions`` block on each
  listed repository is *the account's role in that repository*, not the token's grant. A token
  with Contents:read on a repository its owner administers still reports ``push: true``. That
  is why ``can_push`` is documented as the account's role rather than as a promise, and why a
  false there is trusted (the account cannot push, so neither can any token it issues) while a
  true is not treated as proof of anything. GitHub answers the token's own write grant on the
  first push and there is no read-only call that asks it earlier.

``refusal_reason`` is therefore built from the three things GitHub *states*: it refused the
token, the classic token carries no repository scope, or it listed no repositories at all. An
``UNKNOWN`` -- GitHub unreachable, rate-limited, or an answer that could not be read -- is
never a reason, for ``credential_verification``'s reason: refusing somebody's setup because a
provider was briefly slow is a worse outage than the one this exists to prevent.

Nothing here logs or returns the token. The report is the whole output.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog

from services.credential_verification import CredentialVerdict, verdict_for_error

_LOGGER = structlog.get_logger(__name__)

# How many repositories one listing will read before it stops and says it stopped. A token on a
# large organisation can reach thousands, and a request that pages through all of them is a
# request that times out -- so the cap is a bound on this endpoint's cost, and `truncated` is
# how the caller learns the list it was given is not the whole one.
MAX_LISTED_REPOSITORIES = 500

# The classic-PAT scopes that permit the pushes this platform makes. `repo` covers private
# repositories, `public_repo` only public ones; either is enough to not be refused here,
# because which one is needed depends on the repository and GitHub decides that per push.
_REPOSITORY_SCOPES = frozenset({"repo", "public_repo"})

# Not required, and worth saying out loud: a push that touches `.github/workflows/**` is
# rejected by GitHub for a classic token without this scope, whatever its `repo` scope says.
_WORKFLOW_SCOPE = "workflow"


class GitHubTokenKind(StrEnum):
    """Which kind of credential answered, inferred only from whether scopes were reported."""

    # GitHub returned an `X-OAuth-Scopes` header, which only classic PATs carry.
    CLASSIC = "classic"
    # No scope header. Fine-grained PATs and GitHub App installation tokens both look like
    # this, and neither publishes its grants anywhere this can read them.
    FINE_GRAINED = "fine_grained"
    # Nobody answered, so there is nothing to infer from.
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class GitHubRepositoryAccess:
    """One repository a token listed, as GitHub described it."""

    full_name: str
    url: str
    default_branch: str
    private: bool
    archived: bool
    # The *account's* role in this repository, copied from GitHub's `permissions.push`. A
    # false is conclusive -- an account that cannot push cannot issue a token that can. A true
    # is not a promise for a fine-grained token, for the reason the module docstring gives.
    can_push: bool


@dataclass(frozen=True, slots=True)
class GitHubAccessReport:
    """What one token reached, and whether anything GitHub said justifies refusing it."""

    verdict: CredentialVerdict
    token_kind: GitHubTokenKind = GitHubTokenKind.UNKNOWN
    scopes: tuple[str, ...] = ()
    repositories: tuple[GitHubRepositoryAccess, ...] = ()
    # False when the listing itself failed. Separate from an empty tuple on purpose: "GitHub
    # said you can reach nothing" is a finding, and "the listing call failed" is the absence
    # of one, and only the first may refuse a token.
    repositories_listed: bool = False
    truncated: bool = False
    # Things worth telling somebody that are not reasons to refuse anything.
    advisories: tuple[str, ...] = field(default_factory=tuple)

    @property
    def writable(self) -> tuple[GitHubRepositoryAccess, ...]:
        """The listed repositories this account can push to and that still accept pushes."""
        return tuple(
            repository
            for repository in self.repositories
            if repository.can_push and not repository.archived
        )

    @property
    def refusal_reason(self) -> str | None:
        """Why GitHub's answer disqualifies this token, or ``None`` if nothing does.

        Only ever built from something GitHub stated. Read the module docstring before adding
        a fourth reason: every condition here has to be one the provider answered, because
        this string is what stops somebody saving a credential.
        """
        if self.verdict is CredentialVerdict.REFUSED:
            return (
                "GitHub refused this token. It has expired, been revoked, or was mistyped. "
                "Issue a new one and paste it again."
            )
        if self.verdict is not CredentialVerdict.ACCEPTED:
            return None
        if self.token_kind is GitHubTokenKind.CLASSIC and not (
            _REPOSITORY_SCOPES & set(self.scopes)
        ):
            listed = ", ".join(scope for scope in self.scopes if scope) or "none"
            return (
                "This token carries no repository scope, so GitHub will refuse every clone "
                f"and push the platform makes. Scopes on it: {listed}. Re-issue it with the "
                "'repo' scope, or 'public_repo' if you only build in public repositories."
            )
        if self.repositories_listed and not self.repositories:
            return (
                "GitHub accepted this token but it can reach no repositories. A fine-grained "
                "token has to name the repositories it may use; grant it the ones you build "
                "in, with Contents and Pull requests set to read and write."
            )
        return None


class GitHubAccessProbe:
    """Make the two calls one token's report needs, off the event loop.

    Synchronous underneath because the client this platform already depends on is, and run on
    a worker thread so a slow provider cannot block the loop answering everybody else.
    """

    def __init__(
        self,
        *,
        client_factory: Callable[[str], Any] | None = None,
        repository_limit: int = MAX_LISTED_REPOSITORIES,
    ) -> None:
        """Bind the client builder and the listing bound."""
        self._client_factory = client_factory or _github_client
        self._repository_limit = max(1, repository_limit)

    async def inspect(self, token: str) -> GitHubAccessReport:
        """Return what this token reaches, or that GitHub did not say."""
        if not token.strip():
            return GitHubAccessReport(verdict=CredentialVerdict.UNKNOWN)
        return await asyncio.to_thread(self._inspect, token)

    def _inspect(self, token: str) -> GitHubAccessReport:
        """Identify the token, then list what it reaches, and never raise doing either."""
        client = self._client_factory(token)
        try:
            # One `get_user`, held rather than repeated: it is both the identity call and the
            # handle the listing hangs off, and asking twice is a second round trip for an
            # answer already in hand.
            user = client.get_user()
            login = user.login
        except Exception as error:  # noqa: BLE001 - every failure is a verdict, never a raise
            return self._log(GitHubAccessReport(verdict=verdict_for_error(error)), error=error)
        if not str(login).strip():
            # An answer with no identity in it is one this platform cannot read, which is not
            # the same as a refusal.
            return GitHubAccessReport(verdict=CredentialVerdict.UNKNOWN)

        scopes, kind = _scopes(client)
        try:
            repositories, truncated = self._repositories(user)
        except Exception as error:  # noqa: BLE001 - a failed listing is not a failed request
            # Deliberately not a refusal even on a 403. The token authenticated a moment ago,
            # so a refusal here is about the listing -- SSO enforcement, a rate limit -- and
            # reporting "reaches nothing" would refuse a credential over it.
            return self._log(
                GitHubAccessReport(
                    verdict=CredentialVerdict.ACCEPTED,
                    token_kind=kind,
                    scopes=scopes,
                    repositories_listed=False,
                    advisories=(
                        "GitHub accepted this token but would not list its repositories, so "
                        "the picker has nothing to offer. Add them once GitHub answers.",
                    ),
                ),
                error=error,
            )

        return GitHubAccessReport(
            verdict=CredentialVerdict.ACCEPTED,
            token_kind=kind,
            scopes=scopes,
            repositories=repositories,
            repositories_listed=True,
            truncated=truncated,
            advisories=_advisories(kind, scopes, repositories, truncated=truncated),
        )

    def _repositories(self, user: Any) -> tuple[tuple[GitHubRepositoryAccess, ...], bool]:
        """Read up to the limit, and report whether GitHub had more to give.

        One past the limit is read rather than counted, because the count GitHub publishes on
        an account is not the count this listing returns: affiliation and the token's own
        visibility both narrow it.
        """
        listed: list[GitHubRepositoryAccess] = []
        truncated = False
        for repository in _iterate(user):
            if len(listed) >= self._repository_limit:
                truncated = True
                break
            described = _describe(repository)
            if described is not None:
                listed.append(described)
        return tuple(listed), truncated

    def _log(self, report: GitHubAccessReport, *, error: BaseException) -> GitHubAccessReport:
        """Record the verdict, the exception's type and its status, and nothing else.

        The type name and the status are library-owned symbols. GitHub's message is not, and
        neither is the token.
        """
        _LOGGER.info(
            "github_access_probe_incomplete",
            verdict=report.verdict.value,
            error_type=type(error).__name__,
            status=getattr(error, "status", None),
        )
        return report


def _iterate(user: Any) -> Iterator[Any]:
    """Yield the repositories this token is affiliated with, most recently pushed first.

    ``affiliation`` is stated rather than defaulted so the listing includes repositories
    somebody collaborates on and their organisations', not only the ones they own -- which is
    where most of the work this platform is pointed at actually lives.
    """
    return iter(user.get_repos(affiliation="owner,collaborator,organization_member", sort="pushed"))


def _describe(repository: Any) -> GitHubRepositoryAccess | None:
    """Copy the fields the picker needs, or skip a row that cannot be identified.

    A repository with no full name or no URL is one nothing downstream can act on, and
    dropping it is better than offering a menu entry that cannot be saved.
    """
    full_name = str(getattr(repository, "full_name", "") or "").strip()
    url = str(getattr(repository, "html_url", "") or "").strip()
    if not full_name or not url:
        return None
    permissions = getattr(repository, "permissions", None)
    return GitHubRepositoryAccess(
        full_name=full_name,
        url=url,
        # GitHub always names one; the fallback is for a client stub that does not.
        default_branch=str(getattr(repository, "default_branch", "") or "").strip() or "main",
        private=bool(getattr(repository, "private", False)),
        archived=bool(getattr(repository, "archived", False)),
        can_push=bool(getattr(permissions, "push", False)),
    )


def _scopes(client: Any) -> tuple[tuple[str, ...], GitHubTokenKind]:
    """Read the scopes GitHub reported on the last response, and what their absence means.

    ``None`` is the whole signal for a fine-grained token: GitHub omits ``X-OAuth-Scopes``
    entirely rather than sending it empty, so a missing header and an empty one are different
    answers and must not be collapsed.
    """
    reported = getattr(client, "oauth_scopes", None)
    if reported is None:
        return (), GitHubTokenKind.FINE_GRAINED
    scopes = tuple(str(scope).strip() for scope in reported if str(scope).strip())
    return scopes, GitHubTokenKind.CLASSIC


def _advisories(
    kind: GitHubTokenKind,
    scopes: tuple[str, ...],
    repositories: tuple[GitHubRepositoryAccess, ...],
    *,
    truncated: bool,
) -> tuple[str, ...]:
    """Say what is worth knowing and is not a reason to refuse the token."""
    notes: list[str] = []
    if kind is GitHubTokenKind.CLASSIC and _WORKFLOW_SCOPE not in scopes:
        notes.append(
            "This token has no 'workflow' scope. GitHub will reject any push that changes a "
            "file under .github/workflows/, even though the rest of the push is permitted."
        )
    if kind is GitHubTokenKind.FINE_GRAINED:
        notes.append(
            "A fine-grained token does not publish its own permissions, so this lists what "
            "your account can do rather than what the token may do. GitHub answers that on "
            "the first push; the token needs Contents and Pull requests set to read and write."
        )
    if repositories and not any(
        repository.can_push and not repository.archived for repository in repositories
    ):
        notes.append(
            "Your account cannot push to any repository this token lists, so none of them "
            "can be built in yet."
        )
    if truncated:
        notes.append(
            f"Only the {MAX_LISTED_REPOSITORIES} most recently pushed repositories were read. "
            "A repository you expected may be below that line."
        )
    return tuple(notes)


def _github_client(token: str) -> Any:
    """Build the live client lazily, so importing this module reaches no network code.

    ``per_page`` is raised from GitHub's default of 30 so a listing costs pages rather than
    round trips: the cap above is 500, which is five calls at this size and seventeen at
    the default.
    """
    github_module = importlib.import_module("github")
    return github_module.Github(auth=github_module.Auth.Token(token), per_page=100)


__all__ = [
    "MAX_LISTED_REPOSITORIES",
    "GitHubAccessProbe",
    "GitHubAccessReport",
    "GitHubRepositoryAccess",
    "GitHubTokenKind",
]
