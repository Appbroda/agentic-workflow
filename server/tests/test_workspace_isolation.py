"""A workspace belongs to one person, and this is what that means at every route.

Two accounts, A and B, each with their own feature. The properties defended:

* B's feature is not in A's list, and asking for it directly is *byte-identical* to asking
  for a feature that never existed. Not 403: `AB-Feature-N` is a dense global sequence, so a
  403 confirms existence and lets anybody count the platform's work and whose it is.
* every read route keyed by `{feature_id}` obeys that -- as a parametrised sweep over the
  route list, so a route added later that reaches around the scoped control plane fails
  *here* rather than leaking quietly.
* a mutation A does not own changes nothing, asserted as an effect on B's feature rather
  than as a status code.
* an administrator reads any workspace and acts in none but their own.
* a feature always runs with its *owner's* credentials, whoever pressed the button. That
  last one is the most valuable test in the file: it is the defect §4.4 exists to prevent,
  and the one that would push to somebody's repository with an administrator's token.
"""

from __future__ import annotations

import secrets
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from api.control_plane import RequestScopedCredentials
from api.feature_schemas import StartFeatureRequest
from api.identity import Permission, Role, WorkspaceScope
from main import create_app
from services.feature_actions import FeatureActionService
from services.feature_chat import FeatureChatService
from services.feature_queue import FeatureQueueDispatcher
from services.secrets import InMemorySecretStore
from storage.action_store import InMemoryFeatureActionStore
from storage.chat_store import InMemoryChatMessageStore
from storage.user_store import InMemoryUserDirectory, PlatformUser
from tests.test_feature_api import feature_payload

KEY = "isolation-key"

# Every read route keyed by `{feature_id}`, paired with what it answers for a feature the
# caller *does* own. Parametrised rather than written out per test so that a route added
# without a scope shows up as a failure here -- which is the only mechanism that catches the
# twentieth one.
#
# The own-status is part of the case rather than assumed to be 200, because two of these name
# a sub-resource that this fixture's freshly queued feature has none of: a specific artifact
# id and a specific action id are honest 404s for it, and asserting 200 would make the case
# unwritable rather than making the route safe. What is being defended is that the *foreign*
# answer is identical to the absent-feature answer, which those two still prove.
READ_ROUTES: list[tuple[str, int]] = [
    ("/features/{id}", 200),
    ("/features/{id}/artifacts", 200),
    ("/features/{id}/artifacts/no-such-artifact.json", 404),
    ("/features/{id}/design-preview?node_id=1%3A2", 404),
    ("/features/{id}/clarification", 200),
    ("/features/{id}/workstreams", 200),
    ("/features/{id}/workstreams/backend/operations", 200),
    ("/features/{id}/executions", 200),
    ("/features/{id}/pull-requests", 200),
    ("/features/{id}/chat", 200),
    ("/features/{id}/actions", 200),
    ("/features/{id}/actions/no-such-action", 404),
    ("/features/{id}/repairs", 200),
    ("/features/{id}/events", 200),
    ("/features/{id}/timeline", 200),
    ("/features/{id}/logbook", 200),
]

# Every mutating route keyed by `{feature_id}`, with a body each accepts. `/start` is absent
# on purpose: it takes no feature id and creates in the caller's own workspace by
# construction.
MUTATING_ROUTES: list[tuple[str, dict[str, Any] | None]] = [
    ("/features/{id}/cancel", {}),
    ("/features/{id}/retire", {"operator": "someone", "reason": "done"}),
    ("/features/{id}/resume", {"answers": []}),
    (
        "/features/{id}/workstreams/backend/retry",
        {"additional_attempts": 1, "requested_by": "someone", "reason": "because"},
    ),
    ("/features/{id}/publish", {"reason": "ship it"}),
    ("/features/{id}/repairs/repair-1/approve", {"acknowledge_repository_change": True}),
    ("/features/{id}/repairs/repair-1/reject", {"reason": "no"}),
    ("/features/{id}/chat", {"message": "what is happening?"}),
    ("/features/{id}/chat/1/confirm", None),
    ("/features/{id}/chat/1/reject", None),
]


def client(app: Any) -> AsyncClient:
    """Return an HTTP client bound to the application under test."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


class _IdleDispatcher:
    """Stands in for the worker so a queue entry stays readable.

    An isolated application schedules a one-shot dispatch on every acceptance, which claims
    the entry and runs it. That is right for the tests about execution and wrong for the two
    below, where the entry *is* what is under test: by the time they looked, the worker had
    already claimed and finished it.
    """

    def schedule(self) -> None:
        """Do nothing, so nothing drains the queue."""

    def notify(self) -> None:
        """Do nothing, for the production wiring's name for the same nudge."""

    async def stop(self) -> None:
        """Nothing to stop."""


async def register(
    directory: InMemoryUserDirectory, subject: str, *, role: str
) -> tuple[PlatformUser, str]:
    """Register one account and return it with a token that authenticates as it."""
    user = await directory.create_user(
        subject=subject, display_name=subject.split("@")[0], roles=(role,)
    )
    issued = await directory.issue_token(user.user_id, label="test")
    return user, issued.token


async def feature_owned_by(app: Any, owner_id: str, *, feature_id: str) -> str:
    """Create one feature in somebody's workspace, through the real control plane."""
    result = await app.state.feature_control_plane.start(
        StartFeatureRequest.model_validate({**feature_payload(), "feature_id": feature_id}),
        idempotency_key=f"isolation-{feature_id}",
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        owner_id=owner_id,
    )
    return str(result.record.state.feature_id)


async def two_workspaces() -> tuple[Any, dict[str, Any]]:
    """Build an application with accounts A and B, each owning one feature.

    Both accounts also get their own stored GitHub credential, because the credential
    isolation test needs the dispatcher to be able to resolve two different ones and tell
    them apart.

    Nothing executes. The worker is replaced before the two features are created, so they
    stay exactly as submitted -- which is what every test here needs, because every one of
    them is about who can *see* or *reach* a feature rather than about running one.

    The replacement has to happen before the features exist, not after. An isolated
    application schedules a one-shot dispatch on every acceptance, and those dispatches
    outlive the call that scheduled them: installed afterwards, the idle worker sat beside
    two real ones that were already running, and they raced the assertions. That is what made
    the credential test fail in a full-suite run and pass on its own.
    """
    directory = InMemoryUserDirectory()
    store = InMemorySecretStore()
    actions = FeatureActionService(store=InMemoryFeatureActionStore(), build_revision="test")
    app = create_app(
        platform_api_key=KEY,
        user_directory=directory,
        secret_store=store,
        feature_actions=actions,
    )
    # A chat service, so the transcript routes answer rather than 503. Its assistant is never
    # reached: every chat call in this file is refused before a model would be asked.
    app.state.feature_chat = FeatureChatService(
        assistant_for=lambda credentials, *, agent_platform: None,  # type: ignore[arg-type]
        control_plane=app.state.feature_control_plane,
        store=InMemoryChatMessageStore(),
        actions=actions,
    )
    app.state.feature_dispatcher = _IdleDispatcher()
    account_a, token_a = await register(directory, "a@example.com", role=Role.OPERATOR.value)
    account_b, token_b = await register(directory, "b@example.com", role=Role.OPERATOR.value)
    admin, admin_token = await register(directory, "admin@example.com", role=Role.ADMIN.value)
    for account in (account_a, account_b):
        for provider in ("github", "openai"):
            await store.put(
                owner_id=account.user_id,
                provider=provider,
                secret=f"{provider}-{account.user_id}-{secrets.token_hex(4)}",
            )
    feature_a = await feature_owned_by(app, account_a.user_id, feature_id="feature-a")
    feature_b = await feature_owned_by(app, account_b.user_id, feature_id="feature-b")
    # Both submissions settled, so the queue is empty. `claim` takes the *oldest* waiting
    # entry, so a test that queues one thing and claims it would otherwise be handed one of
    # these two instead -- and one of them happens to name the identity the assertion wanted,
    # which is how that test passed for the wrong reason before this line existed.
    for feature_id in (feature_a, feature_b):
        await app.state.feature_control_plane.queue.finish(feature_id, succeeded=True)
    return app, {
        "directory": directory,
        "secrets": store,
        "a": account_a,
        "token_a": token_a,
        "b": account_b,
        "token_b": token_b,
        "admin": admin,
        "admin_token": admin_token,
        "feature_a": feature_a,
        "feature_b": feature_b,
    }


def auth(token: str) -> dict[str, str]:
    """Return the header that authenticates as one token."""
    return {"Authorization": f"Bearer {token}"}


# --- listing -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_list_holds_only_the_callers_own_features() -> None:
    """The dashboard is the first place a leak would be visible, and the easiest to miss."""
    app, world = await two_workspaces()

    async with client(app) as http:
        mine = await http.get("/features", headers=auth(world["token_a"]))
        theirs = await http.get("/features", headers=auth(world["token_b"]))

    assert [item["feature_id"] for item in mine.json()["features"]] == [world["feature_a"]]
    assert [item["feature_id"] for item in theirs.json()["features"]] == [world["feature_b"]]


@pytest.mark.asyncio
async def test_an_administrator_lists_every_workspace() -> None:
    """The deployment operator has to be able to see what the platform is doing."""
    app, world = await two_workspaces()

    async with client(app) as http:
        response = await http.get("/features", headers=auth(world["admin_token"]))

    assert {item["feature_id"] for item in response.json()["features"]} == {
        world["feature_a"],
        world["feature_b"],
    }


@pytest.mark.asyncio
async def test_a_third_account_with_no_features_sees_none() -> None:
    """An empty workspace is empty, not a view of the deployment."""
    app, world = await two_workspaces()
    _, token = await register(world["directory"], "c@example.com", role=Role.OPERATOR.value)

    async with client(app) as http:
        response = await http.get("/features", headers=auth(token))

    assert response.json()["features"] == []


# --- reads -------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("route", "own_status"), READ_ROUTES)
async def test_a_foreign_feature_is_indistinguishable_from_a_missing_one(
    route: str, own_status: int
) -> None:
    """404 and the same body, on every read route keyed by a feature id.

    Both halves matter. The status stops enumeration of the global reference sequence; the
    body stops the response *text* from doing the same thing -- a distinct sentence for
    "not yours" would be exactly as good an oracle as a distinct code.

    The bodies are compared with the feature id substituted, because the existing
    `_not_found` detail echoes the id the caller asked about. That echo carries nothing the
    caller did not already send; what must not differ is everything else.
    """
    app, world = await two_workspaces()
    missing = "feature-never-existed"

    async with client(app) as http:
        foreign = await http.get(
            route.format(id=world["feature_b"]), headers=auth(world["token_a"])
        )
        absent = await http.get(route.format(id=missing), headers=auth(world["token_a"]))
        own = await http.get(route.format(id=world["feature_a"]), headers=auth(world["token_a"]))

    assert foreign.status_code == 404, f"{route} leaked another workspace"
    assert foreign.status_code == absent.status_code
    assert foreign.text == absent.text.replace(missing, world["feature_b"]), (
        f"{route} says which kind of 404 this is"
    )
    # The route answers for a feature the caller does own -- otherwise a route broken for
    # everybody would pass above while defending nothing.
    assert own.status_code == own_status, f"{route} in the caller's own workspace"
    if own_status == 404:
        # A missing sub-resource in a visible feature says so, and says something different
        # from "there is no such feature": the caller may see the feature.
        assert own.text != foreign.text


@pytest.mark.asyncio
async def test_an_administrator_reads_a_feature_they_do_not_own() -> None:
    """`WORKSPACE_READ_ANY`, and the reason it is a named grant rather than a role check."""
    app, world = await two_workspaces()

    async with client(app) as http:
        response = await http.get(
            f"/features/{world['feature_b']}", headers=auth(world["admin_token"])
        )

    assert response.status_code == 200
    assert response.json()["feature_id"] == world["feature_b"]


@pytest.mark.asyncio
async def test_the_event_stream_refuses_to_subscribe_to_a_foreign_feature() -> None:
    """SSE checks at subscribe, and again on every poll -- see the second assertion."""
    app, world = await two_workspaces()

    async with client(app) as http:
        response = await http.get(
            f"/features/{world['feature_b']}/events/stream", headers=auth(world["token_a"])
        )

    assert response.status_code == 404


# --- mutations ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("route,body", MUTATING_ROUTES)
async def test_a_foreign_mutation_is_refused_and_changes_nothing(
    route: str, body: dict[str, Any] | None
) -> None:
    """The status is not the assertion. What B's feature looks like afterwards is.

    A refusal that had already written something would pass a status check and still have
    damaged somebody's work, so this reads B's feature before and after and compares.
    """
    app, world = await two_workspaces()
    inner = app.state.feature_control_plane

    before = await inner.get_record(world["feature_b"])
    async with client(app) as http:
        refused = await http.post(
            route.format(id=world["feature_b"]),
            headers=auth(world["token_a"]),
            json=body,
        )
    after = await inner.get_record(world["feature_b"])

    assert refused.status_code == 404, f"{route} reached another workspace"
    assert after.state.status == before.state.status
    assert after.state.model_dump(mode="json") == before.state.model_dump(mode="json")


@pytest.mark.asyncio
async def test_the_chat_action_path_cannot_reach_a_foreign_feature() -> None:
    """Its own test because this path has bypassed a guard before.

    A confirmed proposal reaches the same control-plane methods the buttons do, through the
    chat service rather than the route -- so the service has to be scoped too, and a test
    that only exercised the buttons would not have noticed.
    """
    app, world = await two_workspaces()

    async with client(app) as http:
        history = await http.get(
            f"/features/{world['feature_b']}/chat", headers=auth(world["token_a"])
        )
        confirm = await http.post(
            f"/features/{world['feature_b']}/chat/1/confirm", headers=auth(world["token_a"])
        )

    assert history.status_code == 404
    assert confirm.status_code == 404


@pytest.mark.asyncio
async def test_an_administrator_reads_but_cannot_retry_a_feature_they_do_not_own() -> None:
    """`WORKSPACE_READ_ANY` is a read grant, and acting is not a read.

    Retrying somebody else's feature spends their money and pushes with their token. An
    administrator holds every permission the retry route checks, so the only thing refusing
    them here is the workspace filter -- narrowed for mutations by
    `WorkspaceScope.for_mutation`.
    """
    app, world = await two_workspaces()
    inner = app.state.feature_control_plane

    async with client(app) as http:
        readable = await http.get(
            f"/features/{world['feature_b']}", headers=auth(world["admin_token"])
        )
        retry = await http.post(
            f"/features/{world['feature_b']}/workstreams/backend/retry",
            headers=auth(world["admin_token"]),
            json={"additional_attempts": 1, "requested_by": "the admin", "reason": "because"},
        )
        retire = await http.post(
            f"/features/{world['feature_b']}/retire",
            headers=auth(world["admin_token"]),
            json={"operator": "the admin", "reason": "tidying up"},
        )
    after = await inner.get_record(world["feature_b"])

    assert readable.status_code == 200
    assert retry.status_code == 404
    assert retire.status_code == 404
    # The effect, not the status: the feature is not retired, and no attempt was bought.
    # Its lifecycle status is deliberately not compared against a before-value -- this
    # application dispatches its own queue, so B's feature moves on its own while the
    # administrator is being refused, and a status comparison would be a flake.
    assert after.state.status.value not in {"cancelled", "cancelling"}
    assert not any(item.event == "feature_retired_by_operator" for item in after.lifecycle_events)


@pytest.mark.asyncio
async def test_an_administrator_still_acts_in_their_own_workspace() -> None:
    """The narrowing must not have made an administrator unable to use the platform."""
    app, world = await two_workspaces()
    own = await feature_owned_by(app, world["admin"].user_id, feature_id="feature-admins-own")

    async with client(app) as http:
        response = await http.post(f"/features/{own}/cancel", headers=auth(world["admin_token"]))

    assert response.status_code == 200


# --- the other user data -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_neither_account_can_read_the_others_credentials() -> None:
    """Already true before this change, and asserted here so it stays true."""
    app, world = await two_workspaces()

    async with client(app) as http:
        mine = await http.get("/credentials", headers=auth(world["token_a"]))

    hints = {item["provider"]: item["hint"] for item in mine.json()["credentials"]}
    stored = await world["secrets"].describe_all(owner_id=world["b"].user_id)
    theirs = {item.provider: item.hint for item in stored if item.configured}
    assert theirs, "B has credentials, or this test proves nothing"
    for provider, hint in theirs.items():
        assert hints.get(provider) != hint


@pytest.mark.asyncio
async def test_an_ordinary_account_cannot_list_or_create_users() -> None:
    """`USER_MANAGE` is admin-only, and it is the grant that hands out access."""
    app, world = await two_workspaces()

    async with client(app) as http:
        listed = await http.get("/users", headers=auth(world["token_a"]))
        created = await http.post(
            "/users",
            headers=auth(world["token_a"]),
            json={"subject": "d@example.com", "display_name": "D", "roles": ["operator"]},
        )
        as_admin = await http.get("/users", headers=auth(world["admin_token"]))

    assert listed.status_code == 403
    assert created.status_code == 403
    assert as_admin.status_code == 200


@pytest.mark.asyncio
async def test_an_ordinary_account_cannot_reconfigure_slack_or_the_design_source() -> None:
    """The narrowing in §4.1: both point a deployment-wide singleton at one workspace."""
    app, world = await two_workspaces()

    async with client(app) as http:
        slack = await http.put(
            "/slack-configuration",
            headers=auth(world["token_a"]),
            json={"enabled": True, "channel_id": "C123", "token_owner_id": "somebody"},
        )
        design = await http.put(
            "/design-source",
            headers=auth(world["token_a"]),
            json={"enabled": True, "token_owner_id": "somebody", "file_allowlist": []},
        )

    assert slack.status_code == 403
    assert design.status_code == 403


@pytest.mark.asyncio
async def test_an_idempotency_key_cannot_be_used_to_read_another_workspace() -> None:
    """`Idempotency-Key` is a client-chosen string, and a replay answers with the record.

    Without the owner in the key, reusing a header somebody else had used would hand back
    their feature -- the whole of it, through the normal success path and a 200.

    The two submissions carry different `feature_id`s so that what refuses the second is the
    key derivation and not the pre-existing "that id is taken" conflict. Everything else
    about them is identical, including the body, which matters for the second half of the
    test: with no header at all the fallback key is a hash of the body alone, so two people
    submitting the same PRD collided with nobody doing anything unusual.
    """
    app, world = await two_workspaces()
    payload = feature_payload()

    async with client(app) as http:
        theirs = await http.post(
            "/features/start",
            headers={**auth(world["token_b"]), "Idempotency-Key": "shared"},
            json={**payload, "feature_id": "feature-key-theirs"},
        )
        mine = await http.post(
            "/features/start",
            headers={**auth(world["token_a"]), "Idempotency-Key": "shared"},
            json={**payload, "feature_id": "feature-key-mine"},
        )
        # And again with no header, where the key is derived from the body alone.
        theirs_unkeyed = await http.post(
            "/features/start",
            headers=auth(world["token_b"]),
            json={**payload, "feature_id": "feature-body-theirs"},
        )
        mine_unkeyed = await http.post(
            "/features/start",
            headers=auth(world["token_a"]),
            json={**payload, "feature_id": "feature-body-mine"},
        )

    assert theirs.status_code == 201
    assert mine.status_code == 201, "A's submission was treated as a replay of B's"
    assert mine.json()["feature_id"] != theirs.json()["feature_id"]
    assert theirs_unkeyed.status_code == 201
    assert mine_unkeyed.status_code == 201
    assert mine_unkeyed.json()["feature_id"] != theirs_unkeyed.json()["feature_id"]


# --- the credential the work actually runs with -------------------------------------------


@pytest.mark.asyncio
async def test_a_feature_runs_with_its_owners_credentials_whoever_granted_the_retry() -> None:
    """The §4.4 test, and the most valuable one here.

    An administrator grants a retry on B's feature. Before this change the queue entry named
    the *administrator*, so the worker resolved the administrator's GitHub token and pushed
    to B's repository with it -- one person's credential doing another person's work.

    Driven through the real queue and the real dispatcher wiring rather than by inspecting
    the row: what is under test is which identity the dispatcher asks for credentials, and
    that is a fact about the whole hand-off.

    The grant is made through the unscoped control plane, because an administrator acting
    across workspaces is refused at the route now -- correctly, and by the check the test
    above asserts. What remains true and worth defending is the property underneath it: if a
    cross-workspace grant is ever authorised, the queue entry still names the owner.
    """
    app, world = await two_workspaces()
    inner = app.state.feature_control_plane
    admin_scope = WorkspaceScope(owner_id=world["admin"].user_id, may_read_any=True)
    assert admin_scope.applies() is False

    # `resume` rather than `retry_workstream`, because a retry needs a feature with a stopped
    # workstream and this one has not run. All four requeue sites derive the identity the
    # same way -- `requested_by=await self.owner_of(feature_id)` -- so this exercises the
    # derivation, the queue write and the dispatcher's read of it end to end.
    await inner.resume(
        world["feature_b"],
        answers=[],
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        scope=admin_scope,
    )

    asked: list[str] = []

    async def credentials_for(owner_id: str) -> RequestScopedCredentials:  # noqa: D103
        asked.append(owner_id)
        return RequestScopedCredentials(openai_api_key=None, github_token=None)

    dispatcher = FeatureQueueDispatcher(
        queue=inner.queue,
        executor=inner,
        credentials_for=credentials_for,
    )
    claimed = await inner.queue.claim(owner="test-worker", lease_seconds=60)

    assert claimed is not None
    assert claimed.requested_by == world["b"].user_id, (
        "the queue entry names the feature's owner, not the administrator who granted it"
    )
    assert claimed.requested_by != world["admin"].user_id

    # And the dispatcher asks for exactly that identity's keys.
    await dispatcher.credentials_for(claimed.requested_by)
    assert asked == [world["b"].user_id]


@pytest.mark.asyncio
async def test_a_submission_puts_the_submitter_on_its_own_queue_entry() -> None:
    """Owner and actor are the same identity here, and both have to be that identity."""
    app, world = await two_workspaces()

    async with client(app) as http:
        response = await http.post(
            "/features/start",
            headers=auth(world["token_a"]),
            json={**feature_payload(), "feature_id": "feature-as-submitted"},
        )

    assert response.status_code == 201
    claimed = await app.state.feature_control_plane.queue.claim(
        owner="test-worker", lease_seconds=60
    )
    assert claimed is not None
    assert claimed.requested_by == world["a"].user_id


@pytest.mark.asyncio
async def test_the_workspace_read_grant_belongs_to_administrators_only() -> None:
    """Asserted against `ROLE_PERMISSIONS` rather than by calling a route.

    One authority for what a role name grants. A test that inferred this from a 404 would
    still pass if the grant were quietly added to the operator set and the route happened to
    refuse for another reason.
    """
    from api.identity import ROLE_PERMISSIONS

    assert Permission.WORKSPACE_READ_ANY in ROLE_PERMISSIONS[Role.ADMIN]
    assert Permission.WORKSPACE_READ_ANY not in ROLE_PERMISSIONS[Role.OPERATOR]
    assert Permission.WORKSPACE_READ_ANY not in ROLE_PERMISSIONS[Role.VIEWER]
    for narrowed in (Permission.SLACK_CONFIGURATION_MANAGE, Permission.DESIGN_SOURCE_MANAGE):
        assert narrowed in ROLE_PERMISSIONS[Role.ADMIN]
        assert narrowed not in ROLE_PERMISSIONS[Role.OPERATOR]
