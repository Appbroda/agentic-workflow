"""What GitHub said about a token, and what the platform is allowed to conclude from it.

Two properties are being defended. The first is that nothing here refuses on the absence of an
answer: a deployment with no probe, a GitHub that timed out and a listing that failed all leave
somebody able to save their token and add their repositories, because the alternative is that a
provider blip locks an account out of its own setup. The second is the converse -- when GitHub
*does* state something disqualifying, it is stated back at the moment the token is pasted,
rather than discovered by a run dying at push time hours later.

The client is a stub throughout, because the interesting cases are GitHub's answers and there
is no way to make the live API produce a scopeless classic token on demand.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from main import create_app
from services.credential_verification import CredentialVerdict
from services.github_access import (
    GitHubAccessProbe,
    GitHubAccessReport,
    GitHubRepositoryAccess,
    GitHubTokenKind,
)
from services.secrets import InMemorySecretStore
from storage.user_store import InMemoryUserDirectory

KEY = "github-access-key"


class StubPermissions:
    """GitHub's `permissions` block, which reports the account's role in a repository."""

    def __init__(self, *, push: bool) -> None:
        self.push = push


class StubRepository:
    """One row of `GET /user/repos`, carrying only the fields the probe copies."""

    def __init__(
        self,
        full_name: str,
        *,
        push: bool = True,
        archived: bool = False,
        private: bool = True,
        default_branch: str = "main",
    ) -> None:
        self.full_name = full_name
        self.html_url = f"https://github.com/{full_name}"
        self.default_branch = default_branch
        self.private = private
        self.archived = archived
        self.permissions = StubPermissions(push=push)


class StubUser:
    """The authenticated identity, and the handle the listing hangs off."""

    def __init__(self, repositories: list[StubRepository], *, listing_error: Exception | None):
        self.login = "someone"
        self._repositories = repositories
        self._listing_error = listing_error
        self.listing_calls = 0

    def get_repos(self, **_: Any) -> list[StubRepository]:
        """Return the listing, or fail the way GitHub does when it will not answer one."""
        self.listing_calls += 1
        if self._listing_error is not None:
            raise self._listing_error
        return list(self._repositories)


class StubClient:
    """A PyGithub client stub, scripted with an identity, a listing, and a scope header."""

    def __init__(
        self,
        *,
        repositories: list[StubRepository] | None = None,
        scopes: list[str] | None,
        identity_error: Exception | None = None,
        listing_error: Exception | None = None,
    ) -> None:
        # `None` is the fine-grained signal: GitHub omits `X-OAuth-Scopes` rather than
        # sending it empty, and PyGithub surfaces that omission as `None`.
        self.oauth_scopes = scopes
        self._identity_error = identity_error
        self.user = StubUser(repositories or [], listing_error=listing_error)
        self.identity_calls = 0

    def get_user(self) -> StubUser:
        """Return the authenticated user, or fail the way an unusable token does."""
        self.identity_calls += 1
        if self._identity_error is not None:
            raise self._identity_error
        return self.user


class RefusedByGitHub(Exception):
    """A provider exception carrying the status `verdict_for_error` classifies from."""

    def __init__(self, status: int) -> None:
        super().__init__(f"github answered {status}")
        self.status = status


def probe_for(client: StubClient, *, repository_limit: int = 500) -> GitHubAccessProbe:
    """Build a probe bound to one scripted client."""
    return GitHubAccessProbe(
        client_factory=lambda _token: client, repository_limit=repository_limit
    )


@pytest.mark.asyncio
async def test_a_classic_token_with_repository_scope_and_a_writable_repository_is_usable() -> None:
    """The ordinary case: GitHub answered, and nothing it said disqualifies the token."""
    client = StubClient(repositories=[StubRepository("acme/api")], scopes=["repo", "workflow"])

    report = await probe_for(client).inspect("ghp_token")

    assert report.verdict is CredentialVerdict.ACCEPTED
    assert report.token_kind is GitHubTokenKind.CLASSIC
    assert report.refusal_reason is None
    assert [repository.full_name for repository in report.writable] == ["acme/api"]
    assert report.advisories == ()


@pytest.mark.asyncio
async def test_a_classic_token_without_a_repository_scope_is_refused() -> None:
    """The scope header is authoritative for a classic token, so this is a stated no."""
    client = StubClient(repositories=[StubRepository("acme/api")], scopes=["gist", "read:user"])

    report = await probe_for(client).inspect("ghp_token")

    assert report.verdict is CredentialVerdict.ACCEPTED
    reason = report.refusal_reason
    assert reason is not None
    # The remedy names the scope to add and the scopes it found, so somebody can act on it
    # without going to look up what they granted.
    assert "'repo' scope" in reason
    assert "gist" in reason


@pytest.mark.asyncio
async def test_a_token_that_reaches_no_repository_is_refused() -> None:
    """A fine-grained token granted nothing authenticates fine and can build nothing."""
    client = StubClient(repositories=[], scopes=None)

    report = await probe_for(client).inspect("github_pat_token")

    assert report.token_kind is GitHubTokenKind.FINE_GRAINED
    assert report.repositories_listed is True
    assert report.refusal_reason is not None


@pytest.mark.asyncio
async def test_a_token_github_refuses_is_refused_here() -> None:
    """A 401 on the identity call is the provider answering, and the answer is no."""
    client = StubClient(scopes=None, identity_error=RefusedByGitHub(401))

    report = await probe_for(client).inspect("ghp_expired")

    assert report.verdict is CredentialVerdict.REFUSED
    assert report.refusal_reason is not None
    assert client.user.listing_calls == 0


@pytest.mark.asyncio
async def test_a_provider_that_will_not_answer_disqualifies_nothing() -> None:
    """A 503 is the absence of an answer, and nothing is ever refused on one."""
    client = StubClient(scopes=None, identity_error=RefusedByGitHub(503))

    report = await probe_for(client).inspect("ghp_token")

    assert report.verdict is CredentialVerdict.UNKNOWN
    assert report.refusal_reason is None


@pytest.mark.asyncio
async def test_a_listing_that_fails_after_the_token_authenticated_refuses_nothing() -> None:
    """ "Reaches nothing" and "the listing failed" must not become the same sentence.

    The token authenticated a moment earlier, so a refusal on the listing is about the
    listing -- SSO enforcement, a rate limit -- and treating it as "you can reach no
    repositories" would refuse a working credential over it.
    """
    client = StubClient(scopes=["repo"], listing_error=RefusedByGitHub(403))

    report = await probe_for(client).inspect("ghp_token")

    assert report.verdict is CredentialVerdict.ACCEPTED
    assert report.repositories_listed is False
    assert report.repositories == ()
    assert report.refusal_reason is None
    assert report.advisories  # it says the picker will have nothing to offer


@pytest.mark.asyncio
async def test_an_account_with_no_push_anywhere_is_told_so_without_being_refused() -> None:
    """Read-only access to everything is worth saying, and is not GitHub refusing the token."""
    client = StubClient(repositories=[StubRepository("acme/api", push=False)], scopes=["repo"])

    report = await probe_for(client).inspect("ghp_token")

    assert report.writable == ()
    assert report.refusal_reason is None
    assert any("cannot push" in advisory for advisory in report.advisories)


@pytest.mark.asyncio
async def test_an_archived_repository_is_listed_but_never_writable() -> None:
    """GitHub reports push on an archived repository and then rejects the push."""
    client = StubClient(
        repositories=[StubRepository("acme/frozen", archived=True)], scopes=["repo", "workflow"]
    )

    report = await probe_for(client).inspect("ghp_token")

    assert len(report.repositories) == 1
    assert report.writable == ()


@pytest.mark.asyncio
async def test_a_long_listing_stops_at_the_limit_and_says_it_did() -> None:
    """A truncated menu that does not admit it is truncated is worse than a short one."""
    client = StubClient(
        repositories=[StubRepository(f"acme/repo-{index}") for index in range(10)],
        scopes=["repo", "workflow"],
    )

    report = await probe_for(client, repository_limit=4).inspect("ghp_token")

    assert len(report.repositories) == 4
    assert report.truncated is True
    assert any("most recently pushed" in advisory for advisory in report.advisories)


@pytest.mark.asyncio
async def test_a_classic_token_without_the_workflow_scope_is_warned_not_refused() -> None:
    """It only breaks pushes that touch `.github/workflows`, which is an advisory."""
    client = StubClient(repositories=[StubRepository("acme/api")], scopes=["repo"])

    report = await probe_for(client).inspect("ghp_token")

    assert report.refusal_reason is None
    assert any("workflow" in advisory for advisory in report.advisories)


@pytest.mark.asyncio
async def test_an_empty_token_asks_github_nothing() -> None:
    """There is no question to ask about a blank string, and no client is built for one."""
    built: list[str] = []

    def factory(token: str) -> StubClient:
        built.append(token)
        return StubClient(scopes=None)

    report = await GitHubAccessProbe(client_factory=factory).inspect("   ")

    assert report.verdict is CredentialVerdict.UNKNOWN
    assert built == []


# --- The routes that act on a report -------------------------------------------------------


async def app_with_probe(client: StubClient | None) -> tuple[Any, InMemoryUserDirectory]:
    """Build an application whose GitHub answers are scripted, or that has no probe at all."""
    directory = InMemoryUserDirectory()
    app = create_app(
        platform_api_key=KEY, user_directory=directory, secret_store=InMemorySecretStore()
    )
    app.state.github_access_probe = None if client is None else probe_for(client)
    return app, directory


def http(app: Any) -> AsyncClient:
    """Return an HTTP client bound to the application under test."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


HEADERS = {"Authorization": f"Bearer {KEY}"}


@pytest.mark.asyncio
async def test_a_token_github_answered_no_about_is_never_stored() -> None:
    """The whole point: the refusal arrives at the paste, not at the push."""
    app, _ = await app_with_probe(StubClient(scopes=["gist"], repositories=[]))

    async with http(app) as client:
        stored = await client.put(
            "/credentials/github", headers=HEADERS, json={"secret": "ghp_scopeless"}
        )
        listed = await client.get("/credentials", headers=HEADERS)

    assert stored.status_code == 422
    assert "'repo' scope" in stored.json()["detail"]
    # And it is genuinely not there. A refusal that still stored the token would leave the
    # account in the state the refusal exists to prevent.
    github = next(row for row in listed.json()["credentials"] if row["provider"] == "github")
    assert github["configured"] is False


@pytest.mark.asyncio
async def test_a_usable_token_is_stored_with_what_github_said_about_it() -> None:
    """Saving is the one moment somebody is looking, so the answer rides back on the save."""
    app, _ = await app_with_probe(
        StubClient(repositories=[StubRepository("acme/api")], scopes=["repo"])
    )

    async with http(app) as client:
        stored = await client.put(
            "/credentials/github", headers=HEADERS, json={"secret": "ghp_good"}
        )

    assert stored.status_code == 200
    access = stored.json()["access"]
    assert access["verified"] == "accepted"
    assert access["writable_count"] == 1
    assert any("workflow" in advisory for advisory in access["advisories"])


@pytest.mark.asyncio
async def test_a_deployment_with_no_probe_stores_exactly_as_it_did_before() -> None:
    """Gaining this capability must be the only thing that changes behaviour."""
    app, _ = await app_with_probe(None)

    async with http(app) as client:
        stored = await client.put(
            "/credentials/github", headers=HEADERS, json={"secret": "ghp_unasked"}
        )

    assert stored.status_code == 200
    assert stored.json()["access"] is None


@pytest.mark.asyncio
async def test_an_unreachable_github_does_not_stop_somebody_saving_their_token() -> None:
    """Refusing on UNKNOWN would lock an account out of setup over a provider blip."""
    app, _ = await app_with_probe(StubClient(scopes=None, identity_error=RefusedByGitHub(503)))

    async with http(app) as client:
        stored = await client.put(
            "/credentials/github", headers=HEADERS, json={"secret": "ghp_maybe"}
        )

    assert stored.status_code == 200
    assert stored.json()["access"] is None


@pytest.mark.asyncio
async def test_another_provider_is_not_asked_about_at_all() -> None:
    """The probe is GitHub's; storing an OpenAI key must not reach it."""
    stub = StubClient(scopes=["gist"], repositories=[])
    app, _ = await app_with_probe(stub)

    async with http(app) as client:
        stored = await client.put(
            "/credentials/openai", headers=HEADERS, json={"secret": "sk-test"}
        )

    assert stored.status_code == 200
    assert stub.identity_calls == 0


@pytest.mark.asyncio
async def test_the_picker_offers_what_the_token_reaches_and_marks_what_is_already_saved() -> None:
    """The answer that replaces asking somebody to paste a URL."""
    app, _ = await app_with_probe(
        StubClient(
            repositories=[
                StubRepository("acme/api", default_branch="master"),
                StubRepository("acme/web"),
            ],
            scopes=["repo", "workflow"],
        )
    )

    async with http(app) as client:
        await client.put("/credentials/github", headers=HEADERS, json={"secret": "ghp_good"})
        # Saved in the `.git` spelling, which is the same repository under the store's
        # normalisation and must be recognised as one.
        await client.post(
            "/repositories",
            headers=HEADERS,
            json={
                "repository_url": "https://github.com/acme/api.git",
                "default_branch": "master",
                "repository_type": "Backend",
            },
        )
        listed = await client.get("/credentials/github/repositories", headers=HEADERS)

    body = listed.json()
    assert body["available"] is True
    offered = {row["full_name"]: row for row in body["repositories"]}
    assert offered["acme/api"]["already_saved"] is True
    assert offered["acme/web"]["already_saved"] is False
    assert offered["acme/api"]["default_branch"] == "master"


@pytest.mark.asyncio
async def test_the_picker_says_why_it_is_empty_rather_than_looking_empty() -> None:
    """No token stored is a different answer from "your token reaches nothing"."""
    app, _ = await app_with_probe(StubClient(scopes=["repo"]))

    async with http(app) as client:
        listed = await client.get("/credentials/github/repositories", headers=HEADERS)

    body = listed.json()
    assert listed.status_code == 200
    assert body["available"] is False
    assert body["repositories"] == []
    assert "No GitHub token is stored" in body["detail"]


@pytest.mark.asyncio
async def test_a_repository_the_token_cannot_reach_is_refused_by_the_api_not_only_the_form() -> (
    None
):
    """The picker is the guided path; the guarantee has to hold at the endpoint."""
    app, _ = await app_with_probe(
        StubClient(repositories=[StubRepository("acme/api")], scopes=["repo"])
    )

    async with http(app) as client:
        await client.put("/credentials/github", headers=HEADERS, json={"secret": "ghp_good"})
        saved = await client.post(
            "/repositories",
            headers=HEADERS,
            json={
                "repository_url": "https://github.com/someone-else/private",
                "default_branch": "main",
                "repository_type": "Backend",
            },
        )

    assert saved.status_code == 422
    assert "cannot reach" in saved.json()["detail"]


@pytest.mark.asyncio
async def test_a_read_only_repository_is_refused_with_the_remedy_that_matches() -> None:
    """Seeing a repository and being able to build in it have different fixes."""
    app, _ = await app_with_probe(
        StubClient(repositories=[StubRepository("acme/api", push=False)], scopes=["repo"])
    )

    async with http(app) as client:
        await client.put("/credentials/github", headers=HEADERS, json={"secret": "ghp_good"})
        saved = await client.post(
            "/repositories",
            headers=HEADERS,
            json={
                "repository_url": "https://github.com/acme/api",
                "default_branch": "main",
                "repository_type": "Backend",
            },
        )

    assert saved.status_code == 422
    assert "cannot push to it" in saved.json()["detail"]


@pytest.mark.asyncio
async def test_a_repository_on_another_host_is_not_judged_by_a_github_token() -> None:
    """The probe speaks to github.com, so it says nothing about anywhere else."""
    app, _ = await app_with_probe(
        StubClient(repositories=[StubRepository("acme/api")], scopes=["repo"])
    )

    async with http(app) as client:
        await client.put("/credentials/github", headers=HEADERS, json={"secret": "ghp_good"})
        saved = await client.post(
            "/repositories",
            headers=HEADERS,
            json={
                "repository_url": "https://gitlab.example.com/acme/api",
                "default_branch": "main",
                "repository_type": "Backend",
            },
        )

    assert saved.status_code == 201


@pytest.mark.asyncio
async def test_saving_a_repository_with_no_token_stored_is_left_alone() -> None:
    """This platform kept repositories before it kept credentials, and still can."""
    app, _ = await app_with_probe(StubClient(repositories=[], scopes=["repo"]))

    async with http(app) as client:
        saved = await client.post(
            "/repositories",
            headers=HEADERS,
            json={
                "repository_url": "https://github.com/acme/api",
                "default_branch": "main",
                "repository_type": "Backend",
            },
        )

    assert saved.status_code == 201


def test_a_report_with_no_answer_disqualifies_nothing() -> None:
    """The shape every "nobody could ask" path returns, asserted once on the type itself."""
    assert GitHubAccessReport(verdict=CredentialVerdict.UNKNOWN).refusal_reason is None


def test_a_row_github_could_not_identify_is_dropped_rather_than_offered() -> None:
    """A menu entry with no URL is one nothing downstream could act on."""

    class Nameless:
        full_name = ""
        html_url = ""

    from services.github_access import _describe

    assert _describe(Nameless()) is None
    assert isinstance(_describe(StubRepository("acme/api")), GitHubRepositoryAccess)
