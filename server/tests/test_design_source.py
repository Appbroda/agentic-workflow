"""Where designs come from, as a configuration a deployment saves and a resolver reads.

Two properties this file exists to defend.

**A deployment that cites no designs behaves exactly as it does today.** No new required
field, no unmet prerequisite, no refused submission. Every test that touches submission here
is checking an absence.

**A control that cannot work says so instead of appearing configured.** An allowlist holding a
pasted URL would be compared against a normalized file key, match nothing, and refuse every
citation while reading as "on". A `token_owner_id` with no stored credential is the opposite
case and must *not* be refused: configuring the account before pasting the key is a normal
order to do things in.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from api.identity import Role
from api.schemas import PRDSubmission
from artifacts.design_references import (
    DesignReference,
    DesignReferenceError,
    parse_design_url,
)
from artifacts.prd_evidence import EMPTY_WHEN_UNSUPPLIED
from artifacts.schemas import PRDArtifact
from main import create_app
from services.credential_verification import CredentialVerdict
from services.secrets import InMemorySecretStore
from storage.design_source_store import (
    DESIGN_SOURCE_STATUSES,
    DesignSourceConfigurationError,
    InMemoryDesignSourceConfigurationDirectory,
)
from storage.user_store import InMemoryUserDirectory
from tests.test_feature_api import feature_payload

KEY = "design-source-key"

# A real Figma file key, from the file this item's extraction was verified against on
# 2026-09-07. Used rather than an invented string so an allowlist test cannot pass against a
# shape the live API would never produce.
REAL_FILE_KEY = "28gd2JrZO28FCN9PCKM4qK"
OTHER_REAL_FILE_KEY = "VGULlnz44R0Ooe4FZKDxlhh4"


class _ScriptedVerifier:
    """Answer with a fixed verdict and record the providers it was asked about."""

    def __init__(self, verdict: CredentialVerdict) -> None:
        """Bind the verdict every call returns."""
        self._verdict = verdict
        self.providers: list[str] = []

    async def verify(self, *, provider: str, secret: str) -> CredentialVerdict:
        """Record the provider and return the scripted verdict, never the secret."""
        assert secret, "a verifier is never asked about an empty credential"
        self.providers.append(provider)
        return self._verdict


async def app_with_design_source() -> tuple[Any, InMemoryUserDirectory, InMemorySecretStore]:
    """Build an application that knows users, keeps credentials and saves a design source."""
    directory = InMemoryUserDirectory()
    secrets = InMemorySecretStore()
    app = create_app(
        platform_api_key=KEY,
        user_directory=directory,
        secret_store=secrets,
        design_source_configuration=InMemoryDesignSourceConfigurationDirectory(),
    )
    return app, directory, secrets


async def operator_headers(directory: InMemoryUserDirectory) -> dict[str, str]:
    """Register an administrator and return the headers that authenticate as them.

    An administrator rather than an operator, since workspaces became isolated:
    `DESIGN_SOURCE_MANAGE` moved to `_ADMIN_PERMISSIONS`, because the design source is a
    deployment-wide singleton and an operator holding it was an ordinary user re-pointing
    everybody's design citations at their own Figma account. The name is kept so the tests
    below still read as "somebody who may configure this"; that the narrowing happened is
    asserted in `test_design_source_configuration_is_an_administrators_decision`.
    """
    user = await directory.create_user(
        subject=f"operator-{len(await directory.list_users())}@example.com",
        display_name="An operator",
        roles=(Role.ADMIN.value,),
    )
    issued = await directory.issue_token(user.user_id, label="test")
    return {"Authorization": f"Bearer {issued.token}"}


async def non_admin_headers(directory: InMemoryUserDirectory) -> dict[str, str]:
    """Register an ordinary operator and return headers that authenticate as them."""
    user = await directory.create_user(
        subject=f"ordinary-{len(await directory.list_users())}@example.com",
        display_name="An ordinary operator",
        roles=(Role.OPERATOR.value,),
    )
    issued = await directory.issue_token(user.user_id, label="test")
    return {"Authorization": f"Bearer {issued.token}"}


@pytest.mark.asyncio
async def test_design_source_configuration_is_an_administrators_decision() -> None:
    """An operator may no longer re-point the deployment's design source.

    The narrowing that came with isolated workspaces. `design_source_configurations` permits
    one enabled row for the whole deployment, so an operator holding
    `DESIGN_SOURCE_MANAGE` could point everybody's citations at their own Figma account and
    choose which files anybody may cite. Reading it is still an operator's business -- a
    submission has to know whether designs are available -- and only writing moved.
    """
    app, directory, _ = await app_with_design_source()
    headers = await non_admin_headers(directory)

    async with client(app) as http:
        read = await http.get("/design-source", headers=headers)
        write = await http.put(
            "/design-source",
            headers=headers,
            json={"enabled": True, "token_owner_id": "somebody", "file_allowlist": []},
        )
        check = await http.post("/design-source/check", headers=headers)

    assert read.status_code == 200
    assert write.status_code == 403
    assert check.status_code == 403


def client(app: Any) -> AsyncClient:
    """Return an HTTP client bound to the application under test."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


# --------------------------------------------------------------------------------------
# The configuration round-trips through the real router
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unconfigured_deployment_answers_not_configured_rather_than_failing() -> None:
    """Never having pasted a Figma token is a state, not a fault.

    A 500 here would make "this deployment does not use designs" indistinguishable from "the
    design source is broken", and the first is the ordinary case.
    """
    app, directory, _ = await app_with_design_source()
    headers = await operator_headers(directory)

    async with client(app) as http:
        response = await http.get("/design-source", headers=headers)

    body = response.json()
    assert response.status_code == 200
    assert body["configured"] is False
    assert body["enabled"] is False
    assert body["status"] == "disabled"
    assert body["credential_configured"] is False
    assert body["credential_hint"] == ""


@pytest.mark.asyncio
async def test_the_configuration_round_trips_and_reports_the_credential_by_hint() -> None:
    """Saved, read back, and the token reported the way every credential is: four characters."""
    app, directory, secrets = await app_with_design_source()
    headers = await operator_headers(directory)
    secret = "figd-abcdefgh"

    async with client(app) as http:
        me = (await http.get("/me", headers=headers)).json()
        await http.put("/credentials/figma", headers=headers, json={"secret": secret})
        saved = await http.put(
            "/design-source",
            headers=headers,
            json={"enabled": True, "file_allowlist": [REAL_FILE_KEY]},
        )
        read = await http.get("/design-source", headers=headers)

    body = read.json()
    assert saved.status_code == 200
    assert body == saved.json(), "the PUT answers with exactly what the GET reads back"
    assert body["configured"] is True
    assert body["enabled"] is True
    assert body["status"] == "active"
    assert body["token_owner_id"] == me["actor_id"]
    assert body["file_allowlist"] == [REAL_FILE_KEY]
    assert body["file_allowlist_permits_any_file"] is False
    assert body["credential_configured"] is True
    assert body["credential_hint"] == "efgh"
    # The response is safe to show: no field a token could occupy, and never the value.
    assert secret not in read.text
    assert await secrets.resolve(owner_id=me["actor_id"], provider="figma") == secret


@pytest.mark.asyncio
async def test_an_empty_allowlist_states_that_it_permits_any_readable_file() -> None:
    """An empty array is ambiguous to a reader and consequential to a resolver.

    Empty means *any file the configured token can read* -- the right default for a
    single-team deployment and the wrong one for a shared token -- so the response says which
    of the two this is rather than leaving somebody to infer it from `[]`.
    """
    app, directory, _ = await app_with_design_source()
    headers = await operator_headers(directory)

    async with client(app) as http:
        response = await http.put(
            "/design-source", headers=headers, json={"enabled": True, "file_allowlist": []}
        )

    body = response.json()
    assert body["file_allowlist"] == []
    assert body["file_allowlist_permits_any_file"] is True


@pytest.mark.asyncio
async def test_a_token_owner_with_no_credential_saves_and_says_so() -> None:
    """Configuring the account before pasting the key is a normal order to do things in.

    Refusing this would make the order of two settings panels load-bearing, and there is no
    reading of the state in which "saved, no key yet" is worse than "not saved".
    """
    app, directory, _ = await app_with_design_source()
    headers = await operator_headers(directory)

    async with client(app) as http:
        response = await http.put(
            "/design-source",
            headers=headers,
            json={"enabled": True, "token_owner_id": "somebody-else"},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["configured"] is True
    assert body["token_owner_id"] == "somebody-else"
    assert body["credential_configured"] is False
    assert body["credential_hint"] == ""


@pytest.mark.asyncio
async def test_a_pasted_url_in_the_allowlist_is_refused_with_the_field_named() -> None:
    """The allowlist holds file keys, and a stored URL would be a control that lies.

    An entry is compared against the `file_key` a citation is normalized to, so a URL would
    match nothing, ever: the control would read as configured and refuse every citation.
    """
    app, directory, _ = await app_with_design_source()
    headers = await operator_headers(directory)

    async with client(app) as http:
        refused = await http.put(
            "/design-source",
            headers=headers,
            json={
                "enabled": True,
                "file_allowlist": [f"https://www.figma.com/design/{REAL_FILE_KEY}/OpenCRM"],
            },
        )
        # And the configuration was not half-saved by the refusal.
        after = await http.get("/design-source", headers=headers)

    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert "not a Figma file key" in detail
    assert "not the whole URL" in detail
    assert after.json()["configured"] is False


@pytest.mark.asyncio
async def test_the_allowlist_is_deduplicated_and_blank_entries_are_dropped() -> None:
    """Two spellings of one permission are one permission, and a blank row is not an entry."""
    app, directory, _ = await app_with_design_source()
    headers = await operator_headers(directory)

    async with client(app) as http:
        response = await http.put(
            "/design-source",
            headers=headers,
            json={
                "enabled": True,
                "file_allowlist": [REAL_FILE_KEY, "  ", OTHER_REAL_FILE_KEY, REAL_FILE_KEY],
            },
        )

    assert response.json()["file_allowlist"] == [REAL_FILE_KEY, OTHER_REAL_FILE_KEY]


@pytest.mark.asyncio
async def test_a_viewer_may_read_the_configuration_and_not_write_it() -> None:
    """The server refuses, whatever the client chose to render."""
    app, directory, _ = await app_with_design_source()
    user = await directory.create_user(
        subject="a-viewer", display_name="A viewer", roles=(Role.VIEWER.value,)
    )
    issued = await directory.issue_token(user.user_id, label="test")
    headers = {"Authorization": f"Bearer {issued.token}"}

    async with client(app) as http:
        read = await http.get("/design-source", headers=headers)
        write = await http.put("/design-source", headers=headers, json={"enabled": True})
        check = await http.post("/design-source/check", headers=headers)

    assert read.status_code == 200
    assert write.status_code == 403
    assert check.status_code == 403


@pytest.mark.asyncio
async def test_a_deployment_that_keeps_no_design_source_says_so() -> None:
    """A 503 with a sentence, not a crash: the capability is optional by construction."""
    app = create_app(platform_api_key=KEY, secret_store=InMemorySecretStore())
    app.state.design_source_configuration = None

    async with client(app) as http:
        response = await http.get("/design-source", headers={"Authorization": f"Bearer {KEY}"})

    assert response.status_code == 503
    assert "does not resolve design references" in response.json()["detail"]


# --------------------------------------------------------------------------------------
# The check acts on one verdict of three
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_check_degrades_the_source_when_figma_refuses_the_token() -> None:
    """The provider answered and said no, which is the only verdict anything acts on."""
    app, directory, _ = await app_with_design_source()
    headers = await operator_headers(directory)
    verifier = _ScriptedVerifier(CredentialVerdict.REFUSED)
    app.state.credential_verifier = verifier

    async with client(app) as http:
        await http.put("/credentials/figma", headers=headers, json={"secret": "figd-dead"})
        await http.put("/design-source", headers=headers, json={"enabled": True})
        checked = await http.post("/design-source/check", headers=headers)
        after = await http.get("/design-source", headers=headers)

    body = checked.json()
    assert body["provider"] == "figma"
    assert body["usable"] is True
    assert body["verified"] == "refused"
    assert verifier.providers == ["figma"]
    assert after.json()["status"] == "degraded"
    assert "expired or been revoked" in (after.json()["status_reason"] or "")


@pytest.mark.asyncio
async def test_the_check_writes_nothing_when_nobody_answered() -> None:
    """No answer is not a negative answer, and a network blip must not pause resolution.

    An UNKNOWN that degraded the source would stop every design citation on this deployment
    because Figma was briefly slow, which is a worse outage than the one the check diagnoses.
    """
    app, directory, _ = await app_with_design_source()
    headers = await operator_headers(directory)
    app.state.credential_verifier = _ScriptedVerifier(CredentialVerdict.UNKNOWN)

    async with client(app) as http:
        await http.put("/credentials/figma", headers=headers, json={"secret": "figd-fine"})
        await http.put("/design-source", headers=headers, json={"enabled": True})
        checked = await http.post("/design-source/check", headers=headers)
        after = await http.get("/design-source", headers=headers)

    assert checked.json()["verified"] == "unknown"
    assert "could not be asked" in checked.json()["detail"]
    assert after.json()["status"] == "active"
    assert after.json()["status_reason"] is None


@pytest.mark.asyncio
async def test_re_saving_the_configuration_is_what_clears_a_degraded_status() -> None:
    """One writer for "it is fine again", and it is the remedy the banner asks for."""
    app, directory, _ = await app_with_design_source()
    headers = await operator_headers(directory)
    app.state.credential_verifier = _ScriptedVerifier(CredentialVerdict.REFUSED)

    async with client(app) as http:
        await http.put("/credentials/figma", headers=headers, json={"secret": "figd-dead"})
        await http.put("/design-source", headers=headers, json={"enabled": True})
        await http.post("/design-source/check", headers=headers)
        degraded = (await http.get("/design-source", headers=headers)).json()
        resaved = await http.put("/design-source", headers=headers, json={"enabled": True})

    assert degraded["status"] == "degraded"
    assert resaved.json()["status"] == "active"
    assert resaved.json()["status_reason"] is None


@pytest.mark.asyncio
async def test_the_check_says_what_to_do_when_there_is_no_configuration_or_no_key() -> None:
    """Two different absences, two different sentences, and neither is a verdict."""
    app, directory, _ = await app_with_design_source()
    headers = await operator_headers(directory)
    app.state.credential_verifier = _ScriptedVerifier(CredentialVerdict.ACCEPTED)

    async with client(app) as http:
        unsaved = await http.post("/design-source/check", headers=headers)
        await http.put("/design-source", headers=headers, json={"enabled": True})
        unkeyed = await http.post("/design-source/check", headers=headers)

    assert unsaved.json()["configured"] is False
    assert "Save the design source configuration first" in unsaved.json()["detail"]
    assert unkeyed.json()["configured"] is False
    assert "No figma credential is stored" in unkeyed.json()["detail"]
    assert unkeyed.json()["verified"] is None


# --------------------------------------------------------------------------------------
# The status vocabulary is enforced, which is the precedent's declared-and-unchecked gap
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_status_vocabulary_is_enforced_on_every_write_path() -> None:
    """`slack_store.CONFIGURATION_STATUSES` is exported and then never checked.

    Four bare-literal writes bypass it, so a typo would reach the column and the console
    filter reading it would silently match nothing. Here every write goes through one
    predicate, and this asserts the values those writes actually produce are in it.
    """
    store = InMemoryDesignSourceConfigurationDirectory()

    enabled = await store.save(
        enabled=True, token_owner_id="somebody", file_allowlist=(), updated_by="somebody"
    )
    disabled = await store.save(
        enabled=False, token_owner_id="somebody", file_allowlist=(), updated_by="somebody"
    )
    assert await store.mark_degraded(disabled.configuration_id, reason="Figma said no") is True
    degraded = await store.get()

    assert enabled.status == "active"
    assert disabled.status == "disabled"
    assert degraded is not None and degraded.status == "degraded"
    assert {enabled.status, disabled.status, degraded.status} <= set(DESIGN_SOURCE_STATUSES)
    # Degrading twice reports False, so "one warning per transition" is implementable.
    assert await store.mark_degraded(disabled.configuration_id, reason="again") is False


@pytest.mark.asyncio
async def test_the_store_refuses_an_allowlist_entry_that_is_not_a_file_key() -> None:
    """Refused in the store as well as at the boundary: one predicate, both callers."""
    store = InMemoryDesignSourceConfigurationDirectory()

    with pytest.raises(DesignSourceConfigurationError, match="not a Figma file key"):
        await store.save(
            enabled=True,
            token_owner_id="somebody",
            file_allowlist=("figd_a-token-shaped-value",),
            updated_by="somebody",
        )

    assert await store.get() is None, "a refused save leaves no row behind"


# --------------------------------------------------------------------------------------
# A citation is read once, at the boundary, and checked before anything is fetched
# --------------------------------------------------------------------------------------

# The two URL forms Figma has shipped, both node-id spellings, and the shapes that are not a
# citation at all. Every accepted case here was checked against the live API on 2026-09-07:
# `2:303` is a real frame of the real file, and the API refuses the `2-303` spelling with
# `400 ID not-an-id is not a valid node_id`, which is why the translation happens at the edge.
_URL_TABLE: list[tuple[str, str, str, list[str]]] = [
    (
        "the current /design/ form",
        f"https://www.figma.com/design/{REAL_FILE_KEY}/OpenCRM?node-id=2-303",
        REAL_FILE_KEY,
        ["2:303"],
    ),
    (
        "the older /file/ form, which every link shared before the rename still uses",
        f"https://www.figma.com/file/{REAL_FILE_KEY}/OpenCRM?node-id=2-303",
        REAL_FILE_KEY,
        ["2:303"],
    ),
    (
        "the API's own colon spelling, percent-encoded as a browser would send it",
        f"https://www.figma.com/design/{REAL_FILE_KEY}/OpenCRM?node-id=2%3A303",
        REAL_FILE_KEY,
        ["2:303"],
    ),
    (
        "an unencoded colon, which is what a hand-edited link carries",
        f"https://www.figma.com/design/{REAL_FILE_KEY}/OpenCRM?node-id=2:303",
        REAL_FILE_KEY,
        ["2:303"],
    ),
    (
        "a node inside a component instance, whose id carries the instance path",
        f"https://www.figma.com/design/{REAL_FILE_KEY}/OpenCRM?node-id=I13-13%3B12-57",
        REAL_FILE_KEY,
        ["I13:13;12:57"],
    ),
    (
        "several frames in one link, in the API's comma-separated spelling",
        f"https://www.figma.com/design/{REAL_FILE_KEY}/OpenCRM?node-id=2-303,11-114",
        REAL_FILE_KEY,
        ["2:303", "11:114"],
    ),
    (
        "the share link's tracking parameter, which is not a node id",
        f"https://www.figma.com/design/{REAL_FILE_KEY}/OpenCRM?node-id=2-303&t=abc-1&m=dev",
        REAL_FILE_KEY,
        ["2:303"],
    ),
    (
        "no node id at all: a whole-file citation, which is legitimate",
        f"https://www.figma.com/design/{REAL_FILE_KEY}/OpenCRM",
        REAL_FILE_KEY,
        [],
    ),
    (
        "no slug either, which is what a copied file URL looks like",
        f"https://figma.com/design/{REAL_FILE_KEY}",
        REAL_FILE_KEY,
        [],
    ),
]


@pytest.mark.parametrize(("why", "url", "file_key", "node_ids"), _URL_TABLE)
def test_a_pasted_design_url_is_normalized_once_at_the_boundary(
    why: str, url: str, file_key: str, node_ids: list[str]
) -> None:
    """Both URL forms and both node-id spellings become the one spelling the API accepts."""
    parsed = parse_design_url(url)

    assert parsed.file_key == file_key, why
    assert list(parsed.node_ids) == node_ids, why
    # And the same derivation happens when the citation itself is constructed, so nothing
    # downstream has to parse a URL a second time.
    reference = DesignReference(url=url)
    assert reference.file_key == file_key
    assert reference.node_ids == node_ids
    assert reference.cites_whole_file is (node_ids == [])
    # The pasted text is kept verbatim, so the console links back to what was looked at.
    assert reference.url == url


_REFUSED_URLS: list[tuple[str, str, str]] = [
    (
        "a host that is not Figma",
        f"https://figma.example.com/design/{REAL_FILE_KEY}/X",
        "must be on figma.com",
    ),
    (
        "a lookalike host that merely ends in the right letters",
        f"https://notfigma.com/design/{REAL_FILE_KEY}/X",
        "must be on figma.com",
    ),
    (
        "credentials in the URL, refused for the reason a repository URL refuses them",
        f"https://user:secret@www.figma.com/design/{REAL_FILE_KEY}/X",
        "do not put credentials",
    ),
    (
        "a prototype link, which is a different kind of document",
        f"https://www.figma.com/proto/{REAL_FILE_KEY}/X?node-id=2-303",
        "does not name a design file",
    ),
    (
        "a FigJam board, which the extraction would resolve to confident nonsense",
        f"https://www.figma.com/board/{REAL_FILE_KEY}/X",
        "does not name a design file",
    ),
    (
        "a Figma URL that names no file at all",
        "https://www.figma.com/files/recent",
        "does not name a design file",
    ),
    (
        "a file key with punctuation in it, which no real key has",
        "https://www.figma.com/design/not-a-key/X",
        "is not a Figma file key",
    ),
    ("something that is not a URL", "the mock is in slack", "must be an HTTP"),
    (
        "a non-HTTP scheme",
        f"figma://design/{REAL_FILE_KEY}/X",
        "must be an HTTP",
    ),
    (
        "a branch link, whose segment 1 is the parent file's key: resolving it would "
        "silently snapshot the wrong design",
        f"https://www.figma.com/design/{REAL_FILE_KEY}/branch/{OTHER_REAL_FILE_KEY}/X"
        "?node-id=2-303",
        "branches are not supported yet",
    ),
    (
        "the pre-rename /file/ spelling of a branch link",
        f"https://www.figma.com/file/{REAL_FILE_KEY}/branch/{OTHER_REAL_FILE_KEY}/X",
        "branches are not supported yet",
    ),
]


@pytest.mark.parametrize(("why", "url", "message"), _REFUSED_URLS)
def test_a_url_that_is_not_a_design_citation_is_refused_by_name(
    why: str, url: str, message: str
) -> None:
    """Refused with the part that is wrong named, the way a repository URL is.

    Being told which field is wrong beats reading a 422 that names a JSON path, and the client
    mirrors these same rules for the same reason.
    """
    with pytest.raises(DesignReferenceError, match=message):
        parse_design_url(url)
    with pytest.raises(ValidationError):
        DesignReference(url=url)
    assert why


def test_a_malformed_node_id_is_refused_before_the_api_can_refuse_it() -> None:
    """The live API answers `400 ID ... is not a valid node_id`, so this saves a round trip."""
    with pytest.raises(ValidationError, match="is not a Figma node id"):
        DesignReference(
            url=f"https://www.figma.com/design/{REAL_FILE_KEY}/X",
            file_key=REAL_FILE_KEY,
            node_ids=["not-an-id"],
        )


def test_a_file_literally_named_branch_is_an_ordinary_citation() -> None:
    """The branch refusal keys on the segment *count*, not the word.

    `/design/<key>/branch` is what Figma produces for a file whose name is "branch": three
    segments and no branch key, which is a citation of that file and nothing else. A branch
    link always carries a fourth segment -- the branch's own key.
    """
    parsed = parse_design_url(f"https://www.figma.com/design/{REAL_FILE_KEY}/branch?node-id=2-303")

    assert parsed.file_key == REAL_FILE_KEY
    assert list(parsed.node_ids) == ["2:303"]


def test_supplied_values_that_restate_the_url_are_accepted_which_is_the_reload_path() -> None:
    """A persisted citation carries values derived from its own URL, so a reload is a match."""
    url = f"https://www.figma.com/design/{REAL_FILE_KEY}/OpenCRM?node-id=2-303,11-114"
    first = DesignReference(url=url)

    # The round trip a stored artifact takes: its own dump, re-validated.
    reloaded = DesignReference.model_validate(first.model_dump())

    assert reloaded == first
    assert reloaded.file_key == REAL_FILE_KEY
    assert reloaded.node_ids == ["2:303", "11:114"]
    # And the URL's own dash spelling restates the same frames, so it is a match too.
    respelled = DesignReference(url=url, file_key=REAL_FILE_KEY, node_ids=["11-114", "2-303"])
    assert set(respelled.node_ids) == {"2:303", "11:114"}


def test_a_supplied_file_key_that_disagrees_with_the_url_is_refused_naming_both() -> None:
    """The URL is the citation; a wire caller must not override which file it names.

    The before-validator used to `setdefault`, so supplied values won over the URL's own
    derivation -- and everything downstream (the resolver, the console's link back, the
    duplicate check) was then about two designs at once.
    """
    with pytest.raises(ValidationError, match=f"'{OTHER_REAL_FILE_KEY}'.*'{REAL_FILE_KEY}'"):
        DesignReference(
            url=f"https://www.figma.com/design/{REAL_FILE_KEY}/OpenCRM?node-id=2-303",
            file_key=OTHER_REAL_FILE_KEY,
        )


def test_supplied_node_ids_that_disagree_with_the_url_are_refused_naming_both() -> None:
    """Same rule as the file key, for the frames: the URL's derivation is the citation."""
    with pytest.raises(ValidationError, match=r"11:114.*2:303"):
        DesignReference(
            url=f"https://www.figma.com/design/{REAL_FILE_KEY}/OpenCRM?node-id=2-303",
            node_ids=["11:114"],
        )


def test_the_same_frame_cited_twice_is_refused_however_it_was_spelled() -> None:
    """Ambiguous for the reason a repeated requirement id is: one label would silently win."""
    one = f"https://www.figma.com/design/{REAL_FILE_KEY}/X?node-id=2-303"
    # The same frame, in the API's spelling, with a different label. Same citation.
    again = f"https://www.figma.com/file/{REAL_FILE_KEY}/X?node-id=2%3A303"

    with pytest.raises(ValidationError, match="cited twice"):
        PRDSubmission(
            title="A screen",
            problem_statement="It needs building.",
            design_references=[
                DesignReference(url=one, label="empty state"),
                DesignReference(url=again, label="loading state"),
            ],
        )
    # A whole-file citation twice is the same kind of ambiguity.
    whole = f"https://www.figma.com/design/{REAL_FILE_KEY}/X"
    with pytest.raises(ValidationError, match="whole file"):
        PRDSubmission(
            title="A screen",
            problem_statement="It needs building.",
            design_references=[DesignReference(url=whole), DesignReference(url=whole)],
        )
    # Two different frames of one file are not duplicates.
    accepted = PRDSubmission(
        title="A screen",
        problem_statement="It needs building.",
        design_references=[
            DesignReference(url=one),
            DesignReference(url=f"https://www.figma.com/design/{REAL_FILE_KEY}/X?node-id=11-114"),
        ],
    )
    assert [item.node_ids for item in accepted.design_references] == [["2:303"], ["11:114"]]


def test_a_persisted_prd_written_before_this_item_still_loads() -> None:
    """`default_factory=list` for the reason every added artifact field gets one."""
    stored = {
        "schema_version": "1.0.0",
        "workflow_id": "feature-old",
        "artifact_id": "001_prd.json",
        "producer": "api",
        "timestamp": datetime(2026, 8, 1, tzinfo=UTC),
        "metadata": {},
        "validation_status": "valid",
        "title": "Something from before",
        "problem_statement": "It was submitted before designs existed.",
    }

    artifact = PRDArtifact.model_validate(stored)

    assert artifact.design_references == []


# --------------------------------------------------------------------------------------
# Safety rule 3: a feature that cites no design behaves exactly as it does today
# --------------------------------------------------------------------------------------


def test_a_prd_that_supplied_no_evidence_serializes_exactly_as_it_did_before_both_items() -> None:
    """The bytes, not the intention -- for both 89- items' evidence fields at once.

    `{{ prd }}` in the product-manager prompt is the whole PRD artifact serialized, so a field
    added here reaches every feature's first model call -- and `design_references: []` in front
    of a model is an instruction to think about designs on a feature that has none, exactly as
    `attachments: []` is an instruction to look at pictures on a feature that shows none. The
    fingerprint `_feature_fingerprint` hashes is the other reader of the same bytes.

    Both fields, and all three serializer entry points, because `EMPTY_WHEN_UNSUPPLIED` is one
    rule for both and a `model_serializer(mode="wrap")` that only fired in `mode="json"` would
    leave the persisted row and the fingerprint carrying the key it claims to omit.
    """
    envelope = {
        "schema_version": "1.0.0",
        "workflow_id": "feature-plain",
        "artifact_id": "001_prd.json",
        "producer": "api",
        "timestamp": datetime(2026, 9, 7, tzinfo=UTC),
        "metadata": {},
        "validation_status": "valid",
        "title": "No design here",
        "problem_statement": "Prose only.",
    }

    plain = PRDArtifact.model_validate(envelope)
    submission = PRDSubmission(title="No design here", problem_statement="Prose only.")

    for field in EMPTY_WHEN_UNSUPPLIED:
        assert field not in plain.model_dump(mode="json")
        assert field not in plain.model_dump()
        assert field not in plain.model_dump_json()
        assert field not in submission.model_dump(mode="json")
        assert field not in submission.model_dump()
        assert field not in submission.model_dump_json()
    # Both fields, so the assertions above are about two absent keys and not one.
    assert set(EMPTY_WHEN_UNSUPPLIED) == {"attachments", "design_references"}
    # A citation, on the other hand, is part of the request and appears.
    cited = PRDArtifact.model_validate(
        {
            **envelope,
            "design_references": [
                {"url": f"https://www.figma.com/design/{REAL_FILE_KEY}/X?node-id=2-303"}
            ],
        }
    )
    payload = cited.model_dump(mode="json")
    assert payload["design_references"][0]["file_key"] == REAL_FILE_KEY
    assert payload["design_references"][0]["node_ids"] == ["2:303"]
    # And a citation does not drag the other item's key back in with it.
    assert "attachments" not in payload


def test_one_kind_of_evidence_does_not_serialize_the_other_kinds_empty_key() -> None:
    """The trap a single `if not payload.get(...)` per field is there to avoid.

    A submission that attached a screenshot and cited no design must send `attachments` and
    not `design_references`, and the reverse. Written against the submission because that is
    the shape `_feature_fingerprint` hashes.
    """
    with_picture = PRDSubmission(
        title="Shown, not cited",
        problem_statement="See [image:login-error].",
        attachments=[{"attachment_id": "attachment-1", "marker": "login-error"}],  # type: ignore[list-item]
    )
    payload = with_picture.model_dump(mode="json")
    assert payload["attachments"][0]["marker"] == "login-error"
    assert "design_references" not in payload

    with_citation = PRDSubmission(
        title="Cited, not shown",
        problem_statement="Prose only.",
        design_references=[
            DesignReference(url=f"https://www.figma.com/design/{REAL_FILE_KEY}/X?node-id=2-303")
        ],
    )
    cited_payload = with_citation.model_dump(mode="json")
    assert cited_payload["design_references"][0]["file_key"] == REAL_FILE_KEY
    assert "attachments" not in cited_payload


@pytest.mark.asyncio
async def test_a_citation_free_submission_is_accepted_by_a_deployment_with_no_design_source() -> (
    None
):
    """The whole mechanism is reachable only from a citation somebody made.

    A deployment that has never heard of Figma must submit features exactly as it does today:
    no unmet prerequisite, no refusal, and the design source is not even consulted.
    """
    app, directory, _ = await app_with_design_source()
    headers = await operator_headers(directory)
    asked: list[str] = []

    class _RefusingDirectory:
        async def get(self) -> None:
            asked.append("get")
            return None

    app.state.design_source_configuration = _RefusingDirectory()

    async with client(app) as http:
        response = await http.post("/features/start", headers=headers, json=feature_payload())

    assert response.status_code == 201
    assert asked == [], "a citation-free submission never asks whether a design source exists"


# --------------------------------------------------------------------------------------
# A citation nothing will ever resolve is refused before anything is queued
# --------------------------------------------------------------------------------------


def _payload_citing(url: str, **overrides: Any) -> dict[str, Any]:
    """One submission that cites one design, otherwise identical to the ordinary payload."""
    payload = feature_payload()
    payload["feature_id"] = f"feature-design-{abs(hash(url)) % 100000}"
    payload["prd"] = {**payload["prd"], "design_references": [{"url": url, **overrides}]}
    return payload


@pytest.mark.asyncio
async def test_a_citation_is_refused_when_no_design_source_is_configured() -> None:
    """Accepting a citation nothing will resolve is the control that lies.

    The console would show a design attached to a feature that was planned, built and reviewed
    against prose alone -- which is the class of lie this whole item exists to stop. Refused
    before anything is queued, so it costs only the person's correction.
    """
    app, directory, _ = await app_with_design_source()
    headers = await operator_headers(directory)
    url = f"https://www.figma.com/design/{REAL_FILE_KEY}/OpenCRM?node-id=2-303"

    async with client(app) as http:
        unconfigured = await http.post(
            "/features/start", headers=headers, json=_payload_citing(url)
        )
        # Saved but switched off is the same condition: nothing would resolve it.
        await http.put("/design-source", headers=headers, json={"enabled": False})
        disabled = await http.post("/features/start", headers=headers, json=_payload_citing(url))

    for response in (unconfigured, disabled):
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "no enabled design source" in detail
        assert "Settings" in detail


@pytest.mark.asyncio
async def test_a_file_outside_the_allowlist_is_refused_naming_the_configuration() -> None:
    """The refusal says where the decision lives, so it is actionable without guesswork."""
    app, directory, _ = await app_with_design_source()
    headers = await operator_headers(directory)

    async with client(app) as http:
        await http.put(
            "/design-source",
            headers=headers,
            json={"enabled": True, "file_allowlist": [OTHER_REAL_FILE_KEY]},
        )
        refused = await http.post(
            "/features/start",
            headers=headers,
            json=_payload_citing(f"https://www.figma.com/design/{REAL_FILE_KEY}/X?node-id=2-303"),
        )
        # The permitted file goes through, so the refusal is about the allowlist and nothing else.
        accepted = await http.post(
            "/features/start",
            headers=headers,
            json=_payload_citing(
                f"https://www.figma.com/design/{OTHER_REAL_FILE_KEY}/X?node-id=10-11"
            ),
        )

    assert refused.status_code == 422
    detail = refused.json()["detail"]
    assert REAL_FILE_KEY in detail
    assert "allowlist" in detail
    assert accepted.status_code == 201


@pytest.mark.asyncio
async def test_a_degraded_design_source_refuses_a_citation_with_the_reason() -> None:
    """A source whose token Figma refused cannot resolve anything, and says so."""
    app, directory, _ = await app_with_design_source()
    headers = await operator_headers(directory)
    app.state.credential_verifier = _ScriptedVerifier(CredentialVerdict.REFUSED)

    async with client(app) as http:
        await http.put("/credentials/figma", headers=headers, json={"secret": "figd-dead"})
        await http.put("/design-source", headers=headers, json={"enabled": True})
        await http.post("/design-source/check", headers=headers)
        refused = await http.post(
            "/features/start",
            headers=headers,
            json=_payload_citing(f"https://www.figma.com/design/{REAL_FILE_KEY}/X?node-id=2-303"),
        )

    assert refused.status_code == 422
    assert "design source is degraded" in refused.json()["detail"]


@pytest.mark.asyncio
async def test_a_citation_scoped_to_a_repository_this_feature_lacks_is_refused() -> None:
    """A citation that applies to nothing would silently reach no workstream at all.

    Said rather than dropped, and refused here because this is the one place that knows which
    repositories the submission actually names.
    """
    app, directory, _ = await app_with_design_source()
    headers = await operator_headers(directory)

    async with client(app) as http:
        await http.put("/design-source", headers=headers, json={"enabled": True})
        refused = await http.post(
            "/features/start",
            headers=headers,
            json=_payload_citing(
                f"https://www.figma.com/design/{REAL_FILE_KEY}/X?node-id=2-303",
                applies_to=["mobile"],
            ),
        )
        # A repository the feature does include is accepted, so the refusal is about the scope.
        accepted = await http.post(
            "/features/start",
            headers=headers,
            json=_payload_citing(
                f"https://www.figma.com/design/{REAL_FILE_KEY}/X?node-id=11-114",
                applies_to=["frontend"],
            ),
        )

    assert refused.status_code == 422
    assert "mobile" in refused.json()["detail"]
    assert "does not include" in refused.json()["detail"]
    assert accepted.status_code == 201


@pytest.mark.asyncio
async def test_an_accepted_citation_is_carried_onto_the_prd_artifact_normalized() -> None:
    """The citation is a record of what somebody submitted, and it is stored as read."""
    app, directory, _ = await app_with_design_source()
    headers = await operator_headers(directory)
    url = f"https://www.figma.com/file/{REAL_FILE_KEY}/OpenCRM?node-id=2-303&t=xyz"

    async with client(app) as http:
        await http.put("/design-source", headers=headers, json={"enabled": True})
        started = await http.post(
            "/features/start",
            headers=headers,
            json=_payload_citing(url, label="empty state"),
        )
        feature_id = started.json()["feature_id"]
        artifacts = await http.get(
            f"/features/{feature_id}/artifacts?artifact_type=prd", headers=headers
        )
        artifact_id = artifacts.json()["artifacts"][0]["artifact_id"]
        stored = await http.get(f"/features/{feature_id}/artifacts/{artifact_id}", headers=headers)

    cited = stored.json()["payload"]["design_references"]
    assert len(cited) == 1
    assert cited[0]["url"] == url, "the pasted link is kept, so the console links back to it"
    assert cited[0]["file_key"] == REAL_FILE_KEY
    assert cited[0]["node_ids"] == ["2:303"]
    assert cited[0]["label"] == "empty state"
