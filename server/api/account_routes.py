"""Who the caller is, what keys they have configured, and who the deployment knows about.

Three things that were previously unanswerable. The platform authenticated one shared key and
recorded nothing about who held it; provider keys were retyped on every request; and there was
no notion of a user to administer. These routes are the surface of the identity and secret
work, and they are deliberately small: identity that can later come from a provider, and
credentials that are described rather than shown.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import Field

from adapters.slack_adapter import SlackClientError, SlackFailureMode
from api.auth import PlatformAuthenticator, current_actor, require
from api.identity import Actor, Permission, Role
from api.passwords import MAX_PASSWORD_LENGTH, PasswordPolicyError
from api.routes import (
    CREDENTIAL_PROVIDERS,
    deployment_model_declarations,
    get_secret_store,
    inspect_github_access,
    stored_secret,
    verify_stored_credential,
)
from api.schemas import APIModel
from configs.model_roles import (
    PLATFORM_LABELS,
    TIER_LABELS,
    AgentPlatform,
    ModelConfigurationError,
    ModelRole,
    ModelSetupRole,
    PerformanceTier,
    model_config_service_for_setup,
    parse_model_setup_roles,
    validate_model_setup,
)
from services.credential_verification import CredentialVerdict
from services.github_access import GitHubAccessReport
from services.secrets import (
    SUPPORTED_PROVIDERS,
    SecretStoreError,
    SecretStoreUnavailableError,
)
from storage.design_source_store import (
    MAX_ALLOWLIST_ENTRIES,
    DesignSourceConfigurationError,
)
from storage.model_setup_store import ModelSetup, ModelSetupError
from storage.repository_configuration_store import (
    SUGGESTED_REPOSITORY_TYPES,
    RepositoryConfiguration,
    RepositoryConfigurationError,
    normalise_repository_url,
)
from storage.slack_store import NOTIFY_SCOPES
from storage.user_store import SubjectAlreadyRegisteredError, UserDirectoryError


class ActorResponse(APIModel):
    """The identity a request is acting as."""

    actor_id: str
    display_name: str
    # How they proved it. A named person and somebody holding the shared administrative key
    # are both authenticated, and an interface should be able to say which.
    authentication: str
    roles: list[str]
    permissions: list[str]
    # The login identifier, so the console can show whose account this is rather than an
    # opaque id. Empty for the shared platform key, which is not an account with an email.
    subject: str = ""
    # Whether this session is holding a password somebody else chose for it.
    #
    # The console gates its whole application on this -- `AppLayout` redirects every route to
    # the change-password screen while it is true. **No route refuses on it**, deliberately:
    # the flag exists so a password the deployment's *environment* knows cannot persist
    # unnoticed, not as an authorization boundary. An administrator who set a handover
    # password could equally have issued that account an API token, so refusing API requests
    # would defend nothing they could not do anyway.
    #
    # If that judgement is ever revisited, the check belongs in a route dependency beside
    # `requires`, not in each route body -- see `api/auth.py` for why the distinction is
    # load-bearing.
    must_change_password: bool = False


class GitHubAccessResponse(APIModel):
    """What GitHub said one token reaches, in counts and notes rather than in repositories.

    Deliberately not the repository list: this rides on the credential form's answer, which is
    about the token, and `GET /credentials/github/repositories` is where the menu comes from.
    Keeping them apart means saving a key does not pay for a listing nobody is looking at.
    """

    # One of `CredentialVerdict`'s three values, in the shape every other check answers in.
    verified: str
    token_kind: str
    # Classic tokens only. A fine-grained one publishes none, which is not an empty set.
    scopes: list[str] = Field(default_factory=list)
    repositories_listed: bool = False
    repository_count: int = 0
    # How many of those this account can push to and that still accept pushes. What the
    # repository picker can actually offer, and never larger than `repository_count`.
    writable_count: int = 0
    truncated: bool = False
    # Worth reading, and never a reason anything was refused.
    advisories: list[str] = Field(default_factory=list)


class CredentialResponse(APIModel):
    """Whether a provider credential is configured, and never what it is.

    There is no field here a secret could occupy. That is the point: this model cannot leak
    a credential by being serialized, logged, or embedded in something else.
    """

    provider: str
    configured: bool
    # The last four characters only, which lets somebody recognise their own key and tells
    # anybody else nothing.
    hint: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    # Only ever set by `PUT /credentials/github`, and only when a probe answered. Storing a
    # GitHub token is the one moment somebody is looking at the answer, so it rides back on
    # that response instead of costing a second request; listing and deleting never carry it.
    access: GitHubAccessResponse | None = None


class CredentialsResponse(APIModel):
    """One entry per supported provider, configured or not."""

    credentials: list[CredentialResponse]


class GitHubRepositoryOptionResponse(APIModel):
    """One repository a token can reach, as the picker offers it."""

    full_name: str
    repository_url: str
    default_branch: str
    private: bool
    archived: bool
    # The account's role, not the token's grant, for the reason `services.github_access`
    # states at length: a fine-grained token does not publish its own permissions.
    can_push: bool
    # Already in this identity's saved repositories, so the picker can leave it out rather
    # than offer a choice that would be refused as a duplicate.
    already_saved: bool


class GitHubRepositoriesResponse(APIModel):
    """The repositories this identity's stored GitHub token can reach.

    `available` is false whenever nothing could be asked -- no token stored, no probe in this
    deployment, GitHub unreachable -- and `detail` says which. The picker reads the pair and
    says why, rather than showing an empty menu, which is indistinguishable from an answer of
    "you can reach nothing".
    """

    available: bool
    detail: str
    access: GitHubAccessResponse
    repositories: list[GitHubRepositoryOptionResponse] = Field(default_factory=list)


class StoreCredentialRequest(APIModel):
    """Supply a provider credential for the platform to keep."""

    secret: str = Field(min_length=1, max_length=4_096)
    expires_at: datetime | None = None


class CredentialCheckResponse(APIModel):
    """Whether a stored credential is currently usable."""

    provider: str
    configured: bool
    usable: bool
    detail: str
    # What the provider itself said, when the caller asked. Three-valued rather than boolean,
    # and absent by default: `usable` answers "can this deployment open it", which is a
    # different question and the only one this endpoint used to ask. A dead PAT is `usable`
    # and `refused` at the same time, which is exactly how run 190 was misread.
    verified: str | None = None


# What each verdict is worth saying, in the field a person reads. `UNKNOWN` deliberately does
# not read as reassurance: nobody got an answer, and a check that reported "fine" for that is
# how a dead credential stayed invisible for a day.
_VERIFICATION_DETAIL: Mapping[CredentialVerdict, str] = {
    CredentialVerdict.ACCEPTED: (
        "The stored credential is readable, and the provider accepted it just now."
    ),
    CredentialVerdict.REFUSED: (
        "The stored credential is readable, but the provider refused it. It has expired or "
        "been revoked, and every operation using it will fail until it is replaced."
    ),
    CredentialVerdict.UNKNOWN: (
        "The stored credential is readable. The provider could not be asked whether it still "
        "accepts it, so this says nothing about whether it works."
    ),
}


class SlackConfigurationResponse(APIModel):
    """The deployment's Slack delivery configuration, with no field a token could occupy.

    ``credential_configured`` and ``credential_hint`` describe the `slack` credential stored
    for ``token_owner_id`` -- by hint, never by value, exactly as the credentials panel does.
    """

    configured: bool
    enabled: bool = False
    workspace_id: str | None = None
    workspace_name: str | None = None
    channel_id: str | None = None
    channel_name: str | None = None
    token_owner_id: str | None = None
    verbosity: str = "milestones"
    status: str = "disabled"
    status_reason: str | None = None
    console_base_url: str | None = None
    credential_configured: bool = False
    credential_hint: str = ""
    updated_by: str | None = None
    updated_at: datetime | None = None


class SlackConfigurationRequest(APIModel):
    """Point the deployment's notifications at one workspace channel.

    ``verbosity`` accepts only ``milestones``: the detailed per-operation feed is a defined
    template set whose delivery is deliberately deferred, and offering the value before it
    delivers anything would be a control that lies.
    """

    enabled: bool
    channel_id: str = Field(min_length=1, max_length=64)
    channel_name: str | None = Field(default=None, max_length=256)
    verbosity: Literal["milestones"] = "milestones"
    console_base_url: str | None = Field(default=None, max_length=512)
    # Which identity's `slack` credential the dispatcher resolves. Defaults to the caller.
    token_owner_id: str | None = Field(default=None, max_length=128)


class DesignSourceConfigurationResponse(APIModel):
    """Where this deployment's designs come from, with no field a token could occupy.

    Shaped on `SlackConfigurationResponse`: the credential is reported by
    `credential_configured` and a four-character `credential_hint` and by nothing else.

    `file_allowlist_permits_any_file` exists because an empty array is ambiguous to a reader
    and consequential to a resolver. An empty allowlist means *any file the configured token
    can read*, which is the right default for a single-team deployment and the wrong one for a
    shared token, so the response says which of the two this is rather than leaving somebody
    to infer it.
    """

    configured: bool
    enabled: bool = False
    token_owner_id: str | None = None
    file_allowlist: list[str] = Field(default_factory=list)
    file_allowlist_permits_any_file: bool = True
    status: str = "disabled"
    status_reason: str | None = None
    credential_configured: bool = False
    credential_hint: str = ""
    updated_by: str | None = None
    updated_at: datetime | None = None


class DesignSourceConfigurationRequest(APIModel):
    """Point the deployment's design citations at one Figma account.

    There is deliberately no token field: the personal access token is a `figma` row in the
    provider credentials panel, one credential UI and one storage path, exactly as the Slack
    bot token is.
    """

    enabled: bool
    # File *keys*, not URLs. A stored URL would be compared against the key a citation was
    # normalized to and match nothing, ever -- the control would read as configured and refuse
    # every citation, which is the "control that lies" `SlackConfigurationRequest` refuses.
    # The store validates the shape and says which one was pasted.
    file_allowlist: list[str] = Field(default_factory=list, max_length=MAX_ALLOWLIST_ENTRIES)
    # Which identity's `figma` credential the resolver opens. Defaults to the caller.
    token_owner_id: str | None = Field(default=None, max_length=128)


class SlackLinkResponse(APIModel):
    """One person's own Slack link and opt-in scope."""

    user_id: str
    slack_user_id: str | None = None
    notify_scope: str = "none"


class SlackLinkRequest(APIModel):
    """Self-service: the person enters their own Slack member ID and chooses a scope.

    The platform never looks a person up by email -- that needs `users:read.email` and would
    map somebody by an attribute they never confirmed.
    """

    slack_user_id: str | None = Field(default=None, max_length=64)
    notify_scope: Literal["none", "human_interaction", "all"] = "none"


# What a Slack member ID looks like (`U0123ABCDEF`, historically also `W...`). Validated
# because a mistyped id does not error anywhere later -- it silently mentions nobody.
_SLACK_MEMBER_ID = re.compile(r"^[UW][A-Z0-9]{2,32}$")


class ProviderRequirementResponse(APIModel):
    """One provider the platform needs, and whether this identity has configured it."""

    provider: str
    label: str
    configured: bool


class AgentPlatformResponse(APIModel):
    """One (platform, tier) a feature could be submitted on, and whether it can be.

    `configured` is a deployment fact the browser cannot work out for itself: it is true only
    when all four of that pairing's model roles resolve. The models and the efforts they run
    at are published with it so a submission form can say what a choice actually runs on
    without holding a copy of the deployment's configuration -- and so that copy cannot drift.
    """

    platform: str
    # Which performance tier this entry offers the platform at. `high` is today's behaviour,
    # so a client that predates tiers reading this list still names a pairing that runs.
    performance_tier: str = "high"
    label: str
    configured: bool
    # Role name to model identifier, for the pairings that have one. Empty for a pairing
    # this deployment has not configured. Never a credential, a base URL or an account.
    models: dict[str, str] = Field(default_factory=dict)
    # Role name to the reasoning effort that role is actually *sent* at -- the effective
    # level, not the configured one, because a level this deployment declares the model does
    # not accept is never sent and naming it here would state configuration as behaviour.
    # A role present with `null` runs at the provider's own default; a role absent entirely
    # means this response carries no answer for it, which is what a client reading an older
    # server sees. Keyed identically to `models` so the two are read together.
    reasoning_efforts: dict[str, str | None] = Field(default_factory=dict)
    # Whether this pairing's *reasoning* role -- the one role that reads the submitted PRD --
    # is declared able to be shown an image. Published for the same reason `configured` is:
    # the browser cannot work this out, and the alternative is a submission form that offers
    # a selection the start endpoint will refuse. Only the reasoning role, because that is
    # the only call images travel on.
    #
    # `false` for an unconfigured pairing and for a deployment that declares nothing, which
    # is the same fail-closed reading `MODEL_VISION_CAPABLE` has everywhere else.
    vision_capable: bool = False


class ModelRoleConfigurationResponse(APIModel):
    """What one role of one (platform, tier) pairing resolved to.

    The same four values the startup `model_roles_resolved` log names, plus the provenance a
    person needs to change one: this table is read-only, so it has to say where each value
    came from or it is a dead end. Every field here is deployment configuration -- a model
    identifier, an effort name, a bound, and the *name* of a variable. Never a value read from
    the environment other than these, and never a credential.
    """

    role: str
    # Which provider answers this role. Absent on a deployment row, whose whole pairing is one
    # platform; present on a custom setup's rows, where a mixed setup pins each role to its
    # own provider and the pairing-level platform can only name one of them.
    platform: str | None = None
    model: str
    # Absent when the deployment configured no effort, or when it declared this model does not
    # accept the one configured -- in which case `requested_reasoning_effort` says what was
    # asked for. A normalization that showed only the outcome would look like a preference.
    reasoning_effort: str | None = None
    requested_reasoning_effort: str | None = None
    max_tokens: int | None = None
    routing_reason: str
    # The variable this selection reads. The panel cannot edit it, so naming it is the
    # difference between "here is what runs" and "here is what runs, and where to change it".
    model_variable: str
    reasoning_variable: str | None = None
    # True when the role is still resolving through a pre-roles variable, which means the
    # deployment does not have independent configuration for it yet. The startup log reports
    # the same fact as `roles_on_legacy_fallback`.
    resolved_from_legacy_variable: bool = False


class ModelSetupResponse(APIModel):
    """One row of the resolved table: a deployment (platform, tier) pairing, or one of the
    caller's own setups, with everything its four roles resolved to."""

    platform: str
    platform_label: str
    performance_tier: str
    tier_label: str
    # The same composed label `/setup` publishes for the same pairing, so a feature's chosen
    # setup and this table spell it identically.
    label: str
    roles: list[ModelRoleConfigurationResponse]
    # Where this row came from. Deployment rows do not change meaning -- the property PR #33's
    # comment reserved -- and a custom row is the caller's own authored setup.
    origin: Literal["deployment", "custom"] = "deployment"
    # The setup's id for a custom row, null for deployment rows.
    setup_id: str | None = None
    # False on deployment rows -- set in the environment, not editable here -- and true on the
    # caller's own rows, which the authoring routes can change.
    editable: bool = False


class ModelConfigurationResponse(APIModel):
    """The resolved model-roles table this deployment runs on, as a read-only view.

    This is the `model_roles_resolved` structure the API logs at startup, served rather than
    left in a log -- extended across tiers, because the log names the unsuffixed configuration
    only and a deployment offering three tiers resolves three tables.

    Only the pairings this deployment can actually run are listed. An unconfigured pairing is
    absent rather than empty: `/setup` already publishes which pairings exist and which are
    selectable, and inventing rows for one that resolved no model would put models in front of
    somebody that nothing would ever call.
    """

    # Deliberately false and deliberately present. System defaults are configuration, changed
    # by changing the deployment's environment, and a client must not have to infer that from
    # the absence of a write route. When a user-authored setup exists it will be a different
    # object with this true, and nothing about these rows changes meaning.
    editable: bool = False
    setups: list[ModelSetupResponse] = Field(default_factory=list)


class ModelSetupRoleInput(APIModel):
    """One role of a setup as a person authors it: platform, model, effort, output bound."""

    platform: str = Field(min_length=1, max_length=32)
    model: str = Field(min_length=1, max_length=256)
    # Absent means "send nothing, keep the provider's default" -- the same meaning it has in
    # the environment. It is a value, not an omission: the Anthropic floor still applies.
    reasoning_effort: str | None = Field(default=None, max_length=32)
    # Required for an Anthropic-platform role, refused for an OpenAI one -- the Messages API
    # rejects a request without it and the Responses API has never been sent one.
    max_tokens: int | None = Field(default=None, gt=0)


class SaveModelSetupRequest(APIModel):
    """Author or replace one model setup. All four roles, or the request is refused."""

    name: str = Field(min_length=1, max_length=256)
    roles: dict[str, ModelSetupRoleInput]


class SavedModelSetupResponse(APIModel):
    """One saved setup, raw as entered, with whether it can currently run.

    Raw rather than resolved: a form re-populating itself needs the values as authored, and
    the resolved view drops what it normalized. `usable` folds the validation state and the
    credential state together; the two detail fields say which one is the problem, in the
    server's own sentence, because this is where a person edits the setup.
    """

    setup_id: str
    name: str
    roles: dict[str, ModelSetupRoleInput]
    created_at: datetime
    updated_at: datetime
    # The distinct platforms the roles name, in role order -- what the credential panel and
    # the Verify buttons are offered for.
    platforms: list[str] = Field(default_factory=list)
    usable: bool = True
    # The validation predicate's own sentence when the setup no longer passes it (the
    # deployment's declarations can change under a saved setup). Never a paraphrase.
    validation_error: str | None = None
    # Non-blocking authoring warnings -- the 181 shape, an unconfigured platform.
    warnings: list[str] = Field(default_factory=list)
    # Display names of providers the setup needs and this identity has not stored a key for.
    missing_credentials: list[str] = Field(default_factory=list)
    # Whether this setup's reasoning role is declared able to read an image, on the same
    # terms and for the same reason as a pairing's.
    vision_capable: bool = False


class ModelSetupsResponse(APIModel):
    """This identity's setups, raw, with usability."""

    setups: list[SavedModelSetupResponse]


class SetupStateResponse(APIModel):
    """What this identity still has to configure before the platform can build for them.

    Served rather than assembled in the browser because both halves are server decisions:
    which providers are required, and whether a deployment stores credentials at all. A
    deployment that stores none has always taken them as request headers, and it reports
    itself ready rather than blocking every feature.
    """

    credentials_ready: bool
    providers: list[ProviderRequirementResponse]
    # Which platforms this deployment can run a feature on. The form offers only the
    # configured ones, and keeps an unconfigured one visible and disabled rather than hiding
    # it: a control that vanishes is indistinguishable from a control that never existed.
    agent_platforms: list[AgentPlatformResponse] = Field(default_factory=list)
    repositories_ready: bool
    saved_repository_count: int
    # False when this deployment keeps no credential store, so the interface can explain that
    # rather than offering a setup step that cannot be completed.
    credential_storage_available: bool


class SavedRepositoryRequest(APIModel):
    """Save or replace one repository.

    Three fields, because the rest is derived: the name comes from the URL, and the stable
    identifier is the server's. Asking somebody to invent an id was one of the questions this
    whole feature exists to remove.
    """

    repository_url: str = Field(min_length=1, max_length=2_048)
    default_branch: str = Field(default="master", min_length=1, max_length=256)
    repository_type: str = Field(default="Other", max_length=64)


class SavedRepositoryResponse(APIModel):
    """One saved repository, as it is shown back."""

    configuration_id: str
    name: str
    repository_url: str
    default_branch: str
    repository_type: str
    created_at: datetime
    updated_at: datetime


class SavedRepositoriesResponse(APIModel):
    """This identity's saved repositories, and the labels the interface offers."""

    repositories: list[SavedRepositoryResponse]
    suggested_types: list[str]


class CreateUserRequest(APIModel):
    """Register one identity the platform can tell apart from another."""

    subject: str = Field(min_length=1, max_length=256)
    display_name: str = Field(min_length=1, max_length=256)
    roles: list[str] = Field(default_factory=lambda: [Role.OPERATOR.value], min_length=1)
    # Optional: an account created without one cannot password-login until an administrator
    # sets one, which is the correct state for an identity a provider will assert. When it is
    # supplied the account is created with `must_change_password`, because a password one
    # person chose for another is a handover credential, not that person's password.
    password: str | None = Field(default=None, min_length=1, max_length=MAX_PASSWORD_LENGTH)


class UpdateUserRequest(APIModel):
    """Change what one identity is called, what it may do, or whether it works.

    Every field optional, and only what is named is changed: an administrator adjusting a
    role does not have to restate a display name and risk clobbering a rename somebody else
    made. `subject` is deliberately absent -- it is the login identifier, and moving it
    re-points an account at a different person.
    """

    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    roles: list[str] | None = Field(default=None, min_length=1)
    disabled: bool | None = None


class SetUserPasswordRequest(APIModel):
    """An administrator handing one identity a password."""

    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


class UserResponse(APIModel):
    """One identity the deployment knows about.

    There is no field a password could occupy, by construction: `has_password` is the whole
    of what this surface says about one. Same property `CredentialResponse` has for provider
    secrets, and for the same reason -- a model that cannot carry a secret cannot leak one by
    being serialized, logged, or embedded in something else.
    """

    user_id: str
    subject: str
    display_name: str
    roles: list[str]
    disabled: bool
    created_at: datetime
    has_password: bool = False
    must_change_password: bool = False
    last_login_at: datetime | None = None


class UsersResponse(APIModel):
    """Every identity the deployment knows about."""

    users: list[UserResponse]


class IssueTokenRequest(APIModel):
    """Mint a token that proves a request is one particular user."""

    label: str = Field(default="unnamed", max_length=256)
    expires_at: datetime | None = None


class IssuedTokenResponse(APIModel):
    """A token, returned exactly once.

    The platform stores only a digest, so this response is the only time the value exists
    outside the caller's hands. There is no endpoint that can show it again.
    """

    token_id: str
    user_id: str
    label: str
    token: str
    expires_at: datetime | None = None


class TokenResponse(APIModel):
    """What can safely be shown about a token that already exists."""

    token_id: str
    user_id: str
    label: str
    created_at: datetime
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    last_used_at: datetime | None = None


class TokensResponse(APIModel):
    """One identity's tokens, newest first."""

    tokens: list[TokenResponse]


def create_account_router(*, authenticator: PlatformAuthenticator) -> APIRouter:
    """Build the identity and credential routes behind the same authentication."""
    router = APIRouter(tags=["identity"], dependencies=[Depends(authenticator)])

    @router.get("/me", response_model=ActorResponse)
    async def whoami(
        request: Request, actor: Annotated[Actor, Depends(current_actor)]
    ) -> ActorResponse:
        """Return the identity this request resolved to, and what it may do.

        The permissions are published so an interface can hide what it cannot do. That is a
        convenience for whoever is looking at it; every one of them is checked again on the
        route that performs the action.

        `subject` and `must_change_password` come from the directory rather than the `Actor`,
        because the actor carries what authorization needs and these two are account facts.
        An identity that is not a row -- the shared platform key -- has neither, and gets the
        empty and false answers rather than a lookup that would fail.
        """
        return await actor_response(request, actor)

    @router.get("/credentials", response_model=CredentialsResponse)
    async def list_credentials(
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> CredentialsResponse:
        """Say which provider credentials this identity has configured."""
        store = _require_store(request)
        descriptors = await store.describe_all(owner_id=actor.actor_id)
        return CredentialsResponse(credentials=[_credential_response(item) for item in descriptors])

    @router.put("/credentials/{provider}", response_model=CredentialResponse)
    async def store_credential(
        provider: str,
        request_body: StoreCredentialRequest,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> CredentialResponse:
        """Keep one provider credential for this identity, sealed at rest.

        Replacing is the same call. There is no separate update, because a credential that
        could be partially changed is a credential that could be left in a state nobody
        intended.

        A GitHub token is asked about before it is kept, and a stated "no" refuses the save.
        Every other provider, and every GitHub answer that is not a stated no, stores exactly
        as before -- see `_refuse_unusable_github_token` for which is which and why.
        """
        require(actor, Permission.CREDENTIAL_MANAGE)
        store = _require_store(request)
        access = (
            await _refuse_unusable_github_token(request, token=request_body.secret)
            if provider == "github"
            else None
        )
        try:
            descriptor = await store.put(
                owner_id=actor.actor_id,
                provider=provider,
                secret=request_body.secret,
                expires_at=request_body.expires_at,
            )
        except SecretStoreError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error
        return _credential_response(descriptor, access=access)

    @router.delete("/credentials/{provider}", response_model=CredentialResponse)
    async def remove_credential(
        provider: str,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> CredentialResponse:
        """Forget one stored provider credential."""
        require(actor, Permission.CREDENTIAL_MANAGE)
        store = _require_store(request)
        try:
            await store.delete(owner_id=actor.actor_id, provider=provider)
            return _credential_response(
                await store.describe(owner_id=actor.actor_id, provider=provider)
            )
        except SecretStoreError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error

    @router.post("/credentials/{provider}/check", response_model=CredentialCheckResponse)
    async def check_credential(
        provider: str,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
        verify: Annotated[bool, Query()] = False,
    ) -> CredentialCheckResponse:
        """Report whether a stored credential can still be opened, and optionally used.

        Local by default. Calling the provider to prove a key works sends somebody's
        credential over the network on a button press, and the failure this check was built
        for -- a deployment whose encryption key changed, leaving stored values unreadable --
        is visible without leaving the process.

        But readability is not usability, and reporting only the first is how run 190 was
        misdiagnosed: the GitHub PAT had expired at midnight, this endpoint said "usable" all
        day because the bytes decrypted fine, and every clone the platform attempted was
        refused. `verify=true` asks the provider the other question. Opt-in, because the
        network call is the cost the paragraph above is about; and it never turns `usable`
        false, because the two answers are about different things.
        """
        require(actor, Permission.CREDENTIAL_MANAGE)
        store = _require_store(request)
        descriptor = await store.describe(owner_id=actor.actor_id, provider=provider)
        if not descriptor.configured:
            return CredentialCheckResponse(
                provider=provider,
                configured=False,
                usable=False,
                detail="No credential is stored for this provider.",
            )
        try:
            secret = await store.resolve(owner_id=actor.actor_id, provider=provider)
        except SecretStoreError as error:
            return CredentialCheckResponse(
                provider=provider, configured=True, usable=False, detail=str(error)
            )
        usable = secret is not None
        if not usable:
            return CredentialCheckResponse(
                provider=provider,
                configured=True,
                usable=False,
                detail="The stored credential has expired and will not be used.",
            )
        if not verify:
            return CredentialCheckResponse(
                provider=provider,
                configured=True,
                usable=True,
                detail="The stored credential is readable.",
            )
        verdict = await verify_stored_credential(request, provider=provider, secret=secret or "")
        return CredentialCheckResponse(
            provider=provider,
            configured=True,
            usable=True,
            verified=verdict.value,
            detail=_VERIFICATION_DETAIL[verdict],
        )

    @router.get("/setup", response_model=SetupStateResponse)
    async def setup_state(
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> SetupStateResponse:
        """Say what this identity still has to configure before creating a feature.

        One call rather than three, because the interface needs the whole answer before it
        decides what to show: a form, a credential prompt, or a repository prompt.
        """
        store = get_secret_store(request)
        platforms = _agent_platforms(request)
        # Only the credentials a feature could actually need here. Both model providers are
        # listed when both are configured, because which one a feature needs is decided when
        # it is submitted -- but a deployment that offers only one must not report the other
        # as an unmet prerequisite and block every submission on a key nothing would use.
        offered = {item.platform for item in platforms if item.configured}
        required = tuple(
            (provider, label)
            for provider, label in CREDENTIAL_PROVIDERS
            if provider == "github" or provider in offered
        )
        providers: list[ProviderRequirementResponse] = []
        for provider, label in required:
            configured = (
                True
                if store is None
                else await stored_secret(store, actor.actor_id, provider) is not None
            )
            providers.append(
                ProviderRequirementResponse(
                    provider=provider, label=label, configured=bool(configured)
                )
            )
        directory = _repository_directory(request)
        saved = [] if directory is None else await directory.list_for_owner(actor.actor_id)
        return SetupStateResponse(
            # At least one model provider, plus GitHub. Requiring every offered provider's key
            # would make configuring a second platform a prerequisite for using the first.
            credentials_ready=(
                any(item.configured for item in providers if item.provider != "github")
                and all(item.configured for item in providers if item.provider == "github")
            ),
            providers=providers,
            agent_platforms=platforms,
            # A deployment with no directory cannot save repositories, so requiring saved ones
            # would block every feature. It reports ready and the one-off entry path is used.
            repositories_ready=directory is None or bool(saved),
            saved_repository_count=len(saved),
            credential_storage_available=store is not None,
        )

    @router.get("/model-configuration", response_model=ModelConfigurationResponse)
    async def model_configuration(
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> ModelConfigurationResponse:
        """Serve the resolved model-roles table: per platform, tier and role, what runs.

        The deployment rows are read-only and say so per row. Which model a role resolves to
        there is deployment configuration -- the whole point of the role indirection is that
        business logic names a role and a deployment names the model -- so this endpoint
        reports them and the environment remains the authority. The caller's own model setups
        join the same list as `origin: "custom"` rows, resolved through the same
        role-addressed service a feature pinned to them runs on, so one page reads one table.

        The identity is read for exactly one thing: which custom rows are the caller's, and
        whether the top-level `editable` -- "you may author a setup here", not "these rows are
        editable" -- is true. The safety argument is unchanged: the response carries models,
        effort names, bounds and provenance names, and there is still no field a credential
        could occupy.
        """
        directory = _model_setup_directory(request)
        editable = directory is not None and actor.may(Permission.MODEL_SETUP_MANAGE)
        setups = _model_setups(request)
        if directory is not None and actor.may(Permission.MODEL_SETUP_MANAGE):
            unsupported, ceilings = _deployment_declarations(request)
            for saved in await directory.list_for_owner(actor.actor_id):
                row = _custom_setup_row(saved, unsupported=unsupported, ceilings=ceilings)
                if row is not None:
                    setups.append(row)
        return ModelConfigurationResponse(editable=editable, setups=setups)

    @router.get("/model-setups", response_model=ModelSetupsResponse)
    async def list_model_setups(
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> ModelSetupsResponse:
        """List this identity's setups, raw as entered, each with whether it can run now.

        Raw, not resolved: a form re-populating itself needs the values as authored. Each row
        carries its validation state and its per-platform credential state, because a setup
        that cannot currently run must say so where it is edited.
        """
        require(actor, Permission.MODEL_SETUP_MANAGE)
        directory = _require_model_setups(request)
        rows = await directory.list_for_owner(actor.actor_id)
        return ModelSetupsResponse(
            setups=[await _saved_setup_response(request, actor, item) for item in rows]
        )

    @router.post(
        "/model-setups",
        response_model=SavedModelSetupResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_model_setup(
        request_body: SaveModelSetupRequest,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> SavedModelSetupResponse:
        """Author one setup, holding it to the checks an environment-backed tier passes.

        Refused with the predicate's own sentence -- this is the one place a user meets the
        AB-Feature-181 footgun, and a paraphrase would soften it. A role pinned to a platform
        with no stored credential is refused here too, naming the role: saved-but-unusable is
        a failure forty minutes from now wearing a success response.
        """
        require(actor, Permission.MODEL_SETUP_MANAGE)
        directory = _require_model_setups(request)
        roles_raw = _roles_payload(request_body)
        await _refuse_invalid_setup(request, actor, roles_raw)
        try:
            saved = await directory.create(
                owner_id=actor.actor_id, name=request_body.name, roles=roles_raw
            )
        except ModelSetupError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error
        return await _saved_setup_response(request, actor, saved)

    @router.put("/model-setups/{setup_id}", response_model=SavedModelSetupResponse)
    async def update_model_setup(
        setup_id: str,
        request_body: SaveModelSetupRequest,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> SavedModelSetupResponse:
        """Replace one setup. Features already pinned to it keep their snapshots (G3)."""
        require(actor, Permission.MODEL_SETUP_MANAGE)
        directory = _require_model_setups(request)
        roles_raw = _roles_payload(request_body)
        await _refuse_invalid_setup(request, actor, roles_raw)
        try:
            saved = await directory.update(
                setup_id, owner_id=actor.actor_id, name=request_body.name, roles=roles_raw
            )
        except ModelSetupError as error:
            detail = str(error)
            if "not found" in detail:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail) from error
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail
            ) from error
        return await _saved_setup_response(request, actor, saved)

    @router.delete("/model-setups/{setup_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_model_setup(
        setup_id: str,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> None:
        """Forget one setup. Features already created from it keep running on their snapshots."""
        require(actor, Permission.MODEL_SETUP_MANAGE)
        directory = _require_model_setups(request)
        if not await directory.delete(setup_id, owner_id=actor.actor_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"model setup not found: {setup_id}",
            )

    @router.get("/credentials/github/repositories", response_model=GitHubRepositoriesResponse)
    async def list_github_repositories(
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> GitHubRepositoriesResponse:
        """List the repositories this identity's stored GitHub token can reach.

        The whole point of the repository form no longer asking for a URL. A URL somebody
        typed is a guess about two things at once -- that the repository exists at that
        spelling, and that their token can reach it -- and both were previously answered by a
        run failing hours later. GitHub already knows both, so it is asked.

        Live on every call rather than cached. A token's reach changes when somebody is
        granted a repository or a fine-grained grant is edited, which is exactly the moment
        they come here to add it; a cache would serve the answer from before that change,
        which is the one answer that is never useful.

        Never refuses. A deployment with no probe, an identity with no token, and a GitHub
        that would not answer all come back as `available: false` with a `detail` saying so,
        because "nobody could ask" and "you can reach nothing" must not look the same.
        """
        directory = _repository_directory(request)
        saved = await directory.list_for_owner(actor.actor_id) if directory is not None else []
        # Compared in the store's own normalised spelling, so a repository saved as
        # `.../name.git` and offered by GitHub as `.../name` is recognised as the same one --
        # which is the comparison the store's uniqueness already uses.
        already_saved = {
            key
            for key in (_normalised_or_none(item.repository_url) for item in saved)
            if key is not None
        }

        store = get_secret_store(request)
        token = await stored_secret(store, actor.actor_id, "github") if store is not None else None
        if not token:
            return GitHubRepositoriesResponse(
                available=False,
                detail=(
                    "No GitHub token is stored for this account. Add one in Provider "
                    "credentials and the repositories it can reach will be listed here."
                ),
                access=_github_access(GitHubAccessReport(verdict=CredentialVerdict.UNKNOWN)),
            )

        report = await inspect_github_access(request, token=token)
        if not report.repositories_listed:
            return GitHubRepositoriesResponse(
                available=False,
                detail=(
                    report.refusal_reason
                    or "GitHub did not answer, so its repositories could not be listed."
                ),
                access=_github_access(report),
            )
        return GitHubRepositoriesResponse(
            available=True,
            detail=report.refusal_reason or "",
            access=_github_access(report),
            repositories=[
                GitHubRepositoryOptionResponse(
                    full_name=repository.full_name,
                    repository_url=repository.url,
                    default_branch=repository.default_branch,
                    private=repository.private,
                    archived=repository.archived,
                    can_push=repository.can_push,
                    already_saved=_normalised_or_none(repository.url) in already_saved,
                )
                for repository in report.repositories
            ],
        )

    @router.get("/repositories", response_model=SavedRepositoriesResponse)
    async def list_saved_repositories(
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> SavedRepositoriesResponse:
        """List the repositories this identity has saved for reuse."""
        directory = _require_repository_directory(request)
        rows = await directory.list_for_owner(actor.actor_id)
        return SavedRepositoriesResponse(
            repositories=[_saved_repository(item) for item in rows],
            suggested_types=list(SUGGESTED_REPOSITORY_TYPES),
        )

    @router.post(
        "/repositories",
        response_model=SavedRepositoryResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_saved_repository(
        request_body: SavedRepositoryRequest,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> SavedRepositoryResponse:
        """Save one repository against this identity, deriving its name from its URL.

        A GitHub repository is checked against what this identity's token reaches before it
        is kept, so the console's picker and the API agree about what may be saved rather
        than the picker being the only thing enforcing it.
        """
        directory = _require_repository_directory(request)
        await _refuse_unreachable_repository(
            request, actor=actor, repository_url=request_body.repository_url
        )
        try:
            saved = await directory.create(
                owner_id=actor.actor_id,
                repository_url=request_body.repository_url,
                default_branch=request_body.default_branch,
                repository_type=request_body.repository_type,
            )
        except RepositoryConfigurationError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error
        return _saved_repository(saved)

    @router.put("/repositories/{configuration_id}", response_model=SavedRepositoryResponse)
    async def update_saved_repository(
        configuration_id: str,
        request_body: SavedRepositoryRequest,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> SavedRepositoryResponse:
        """Replace one saved repository's URL, branch, or label."""
        directory = _require_repository_directory(request)
        await _refuse_unreachable_repository(
            request, actor=actor, repository_url=request_body.repository_url
        )
        try:
            saved = await directory.update(
                configuration_id,
                owner_id=actor.actor_id,
                repository_url=request_body.repository_url,
                default_branch=request_body.default_branch,
                repository_type=request_body.repository_type,
            )
        except RepositoryConfigurationError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error
        return _saved_repository(saved)

    @router.delete("/repositories/{configuration_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_saved_repository(
        configuration_id: str,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> None:
        """Forget one saved repository. Features already created from it are untouched."""
        directory = _require_repository_directory(request)
        if not await directory.delete(configuration_id, owner_id=actor.actor_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"saved repository not found: {configuration_id}",
            )

    @router.get("/users", response_model=UsersResponse)
    async def list_users(
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> UsersResponse:
        """List the identities this deployment knows about."""
        require(actor, Permission.USER_MANAGE)
        directory = _require_directory(request)
        return UsersResponse(users=[_user_response(item) for item in await directory.list_users()])

    @router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
    async def create_user(
        request_body: CreateUserRequest,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> UserResponse:
        """Register one identity, optionally with a password they must then change.

        A duplicate subject is `409` and everything else the directory refuses is `400`. The
        duplicate is decided by the unique constraint inside `create_user` rather than by a
        read here: two administrators creating the same person at the same moment both pass a
        pre-check, and only the database can refuse the second.
        """
        require(actor, Permission.USER_MANAGE)
        directory = _require_directory(request)
        try:
            user = await directory.create_user(
                subject=request_body.subject,
                display_name=request_body.display_name,
                roles=tuple(request_body.roles),
                password=request_body.password,
                # A password one person chose for another is a handover credential, not that
                # person's password. They change it on first use.
                must_change_password=request_body.password is not None,
            )
        except SubjectAlreadyRegisteredError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except (UserDirectoryError, PasswordPolicyError) as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
            ) from error
        return _user_response(user)

    @router.patch("/users/{user_id}", response_model=UserResponse)
    async def update_user(
        user_id: str,
        request_body: UpdateUserRequest,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> UserResponse:
        """Change one identity's name, roles, or whether it works at all.

        Refuses a change that would leave the deployment with nobody who can hand out an
        account. The check counts the enabled administrators *other than this one* and asks
        whether this change removes the last: an administrator demoting themselves while
        another exists is ordinary, and demoting themselves while alone is a lockout nobody
        can undo without a database edit.

        The count is read here and acted on immediately afterwards, which is a race in
        principle: two administrators demoting each other at the same instant could both see
        a count of two. It is not closed with a lock because the recovery is an
        administrator's own break-glass token -- see `docs/DISASTER_RECOVERY.md` -- and a
        lock held across a role change would be a new failure mode for a scenario that
        requires two people acting in the same millisecond.
        """
        require(actor, Permission.USER_MANAGE)
        directory = _require_directory(request)
        target = await directory.get(user_id)
        if target is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown user: {user_id}"
            )
        losing_admin = Role.ADMIN.value in target.roles and (
            (request_body.roles is not None and Role.ADMIN.value not in request_body.roles)
            or request_body.disabled is True
        )
        if losing_admin and not target.disabled:
            remaining = await directory.enabled_administrator_count()
            if remaining <= 1:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        "This is the deployment's last enabled administrator. Grant another "
                        "account the admin role before changing this one."
                    ),
                )
        try:
            user = await directory.update_user(
                user_id,
                display_name=request_body.display_name,
                roles=None if request_body.roles is None else tuple(request_body.roles),
                disabled=request_body.disabled,
            )
        except UserDirectoryError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
            ) from error
        if request_body.disabled is True:
            # A disabled account's tokens must stop working now rather than at expiry.
            # `resolve` already refuses them, so this is belt and braces -- but it also means
            # re-enabling the account does not silently restore somebody's old browser tab.
            await directory.revoke_sessions(user_id)
        return _user_response(user)

    @router.post("/users/{user_id}/password", response_model=UserResponse)
    async def set_user_password(
        user_id: str,
        request_body: SetUserPasswordRequest,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> UserResponse:
        """Hand one identity a new password, and stop every session it had.

        Returns no password and no token. `must_change_password` is set unconditionally: a
        password an administrator typed is known to two people, and the account holder
        replaces it before doing anything else.

        Every one of that account's sessions is revoked, including one they may be using
        right now. That is the point of an administrator resetting a password -- the usual
        reason is that the account may be compromised.
        """
        require(actor, Permission.USER_MANAGE)
        directory = _require_directory(request)
        try:
            user = await directory.set_password(
                user_id, password=request_body.password, must_change=True
            )
        except PasswordPolicyError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
            ) from error
        except UserDirectoryError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        return _user_response(user)

    @router.post("/users/{user_id}/tokens", response_model=IssuedTokenResponse)
    async def issue_user_token(
        user_id: str,
        request_body: IssueTokenRequest,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> IssuedTokenResponse:
        """Mint a token for one identity, and return it the only time it will be readable."""
        require(actor, Permission.USER_MANAGE)
        directory = _require_directory(request)
        try:
            issued = await directory.issue_token(
                user_id, label=request_body.label, expires_at=request_body.expires_at
            )
        except UserDirectoryError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        return IssuedTokenResponse(
            token_id=issued.token_id,
            user_id=issued.user_id,
            label=issued.label,
            token=issued.token,
            expires_at=issued.expires_at,
        )

    @router.get("/users/{user_id}/tokens", response_model=TokensResponse)
    async def list_user_tokens(
        user_id: str,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> TokensResponse:
        """List one identity's tokens, without anything that could authenticate."""
        require(actor, Permission.USER_MANAGE)
        directory = _require_directory(request)
        tokens = await directory.list_tokens(user_id)
        return TokensResponse(
            tokens=[
                TokenResponse(
                    token_id=item.token_id,
                    user_id=item.user_id,
                    label=item.label,
                    created_at=item.created_at,
                    expires_at=item.expires_at,
                    revoked_at=item.revoked_at,
                    last_used_at=item.last_used_at,
                )
                for item in tokens
            ]
        )

    @router.delete("/users/tokens/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def revoke_user_token(
        token_id: str,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> None:
        """Stop a token authenticating anything, keeping the record that it existed."""
        require(actor, Permission.USER_MANAGE)
        directory = _require_directory(request)
        if not await directory.revoke_token(token_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"token not found or already revoked: {token_id}",
            )

    @router.get("/slack-configuration", response_model=SlackConfigurationResponse)
    async def get_slack_configuration(
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> SlackConfigurationResponse:
        """Report where feature threads go, and whether the bot token is stored -- by hint,
        never by value."""
        directory = _require_slack_configuration(request)
        configuration = await directory.get()
        if configuration is None:
            return SlackConfigurationResponse(configured=False)
        credential_configured = False
        credential_hint = ""
        store = get_secret_store(request)
        if store is not None:
            try:
                descriptor = await store.describe(
                    owner_id=configuration.token_owner_id, provider="slack"
                )
            except SecretStoreError:
                descriptor = None
            if descriptor is not None:
                credential_configured = descriptor.configured
                credential_hint = descriptor.hint
        return SlackConfigurationResponse(
            configured=True,
            enabled=configuration.enabled,
            workspace_id=configuration.workspace_id,
            workspace_name=configuration.workspace_name,
            channel_id=configuration.channel_id,
            channel_name=configuration.channel_name,
            token_owner_id=configuration.token_owner_id,
            verbosity=configuration.verbosity,
            status=configuration.status,
            status_reason=configuration.status_reason,
            console_base_url=configuration.console_base_url,
            credential_configured=credential_configured,
            credential_hint=credential_hint,
            updated_by=configuration.updated_by,
            updated_at=configuration.updated_at,
        )

    @router.put("/slack-configuration", response_model=SlackConfigurationResponse)
    async def put_slack_configuration(
        request_body: SlackConfigurationRequest,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> SlackConfigurationResponse:
        """Save the deployment's one Slack configuration. Re-saving clears a degraded status,
        because re-saving is exactly the remedy the status banner asks for.

        The channel is snapshotted onto each feature when its thread is rooted, so editing
        this moves *new* features only -- a feature already anchored keeps replying where its
        root is, forever.
        """
        require(actor, Permission.SLACK_CONFIGURATION_MANAGE)
        directory = _require_slack_configuration(request)
        await directory.save(
            enabled=request_body.enabled,
            channel_id=request_body.channel_id.strip(),
            channel_name=(request_body.channel_name or "").strip() or None,
            token_owner_id=(request_body.token_owner_id or "").strip() or actor.actor_id,
            verbosity=request_body.verbosity,
            console_base_url=(request_body.console_base_url or "").strip() or None,
            updated_by=actor.actor_id,
        )
        return await get_slack_configuration(request, actor)

    @router.post("/slack-configuration/check", response_model=CredentialCheckResponse)
    async def check_slack_configuration(
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> CredentialCheckResponse:
        """Ask Slack whose token this is and whether the channel is reachable.

        The one place a live Slack call is made from a request, and it is operator-initiated,
        so it may fail loudly: a check is not a notification. Answered in the verdict shape
        `POST /credentials/{provider}/check` already uses.
        """
        require(actor, Permission.SLACK_CONFIGURATION_MANAGE)
        directory = _require_slack_configuration(request)
        configuration = await directory.get()
        if configuration is None:
            return CredentialCheckResponse(
                provider="slack",
                configured=False,
                usable=False,
                detail="Save the Slack configuration first.",
            )
        store = _require_store(request)
        descriptor = await store.describe(owner_id=configuration.token_owner_id, provider="slack")
        if not descriptor.configured:
            return CredentialCheckResponse(
                provider="slack",
                configured=False,
                usable=False,
                detail="No slack credential is stored for the configured token owner.",
            )
        try:
            secret = await store.resolve(owner_id=configuration.token_owner_id, provider="slack")
        except SecretStoreError as error:
            return CredentialCheckResponse(
                provider="slack", configured=True, usable=False, detail=str(error)
            )
        if secret is None:
            return CredentialCheckResponse(
                provider="slack",
                configured=True,
                usable=False,
                detail="The stored credential has expired and will not be used.",
            )
        client_factory = getattr(request.app.state, "slack_client_factory", None)
        if client_factory is None:
            return CredentialCheckResponse(
                provider="slack",
                configured=True,
                usable=True,
                verified=CredentialVerdict.UNKNOWN.value,
                detail=_VERIFICATION_DETAIL[CredentialVerdict.UNKNOWN],
            )
        client = client_factory(secret)
        try:
            identity = await client.auth_test()
            channel = await client.channel_info(configuration.channel_id)
        except SlackClientError as error:
            if error.mode is SlackFailureMode.TRANSPORT:
                return CredentialCheckResponse(
                    provider="slack",
                    configured=True,
                    usable=True,
                    verified=CredentialVerdict.UNKNOWN.value,
                    detail=_VERIFICATION_DETAIL[CredentialVerdict.UNKNOWN],
                )
            return CredentialCheckResponse(
                provider="slack",
                configured=True,
                usable=True,
                verified=CredentialVerdict.REFUSED.value,
                detail=(
                    "Slack refused the check just now (recorded as "
                    f"{error.error_code}). "
                    + (
                        "The bot token has been revoked or is wrong — re-save it."
                        if error.mode is SlackFailureMode.TOKEN_REVOKED
                        else "The channel could not be read — check the id and that the "
                        "bot has been invited to it."
                    )
                ),
            )
        if channel.is_archived:
            return CredentialCheckResponse(
                provider="slack",
                configured=True,
                usable=True,
                verified=CredentialVerdict.REFUSED.value,
                detail="Slack accepted the token, but the configured channel is archived.",
            )
        await directory.record_workspace_identity(
            configuration.configuration_id,
            workspace_id=identity.workspace_id,
            workspace_name=identity.workspace_name,
        )
        return CredentialCheckResponse(
            provider="slack",
            configured=True,
            usable=True,
            verified=CredentialVerdict.ACCEPTED.value,
            detail=("Slack accepted the token just now, and the configured channel is reachable."),
        )

    @router.get("/design-source", response_model=DesignSourceConfigurationResponse)
    async def get_design_source(
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> DesignSourceConfigurationResponse:
        """Report where designs come from, and whether a Figma token is stored -- by hint.

        An unconfigured deployment answers "not configured" rather than 500: never having
        pasted a Figma token is the ordinary state of a deployment that cites no designs, and
        it must read as a state rather than as a fault.
        """
        directory = _require_design_source(request)
        configuration = await directory.get()
        if configuration is None:
            return DesignSourceConfigurationResponse(configured=False)
        credential_configured = False
        credential_hint = ""
        store = get_secret_store(request)
        if store is not None:
            try:
                descriptor = await store.describe(
                    owner_id=configuration.token_owner_id, provider="figma"
                )
            except SecretStoreError:
                descriptor = None
            if descriptor is not None:
                credential_configured = descriptor.configured
                credential_hint = descriptor.hint
        return DesignSourceConfigurationResponse(
            configured=True,
            enabled=configuration.enabled,
            token_owner_id=configuration.token_owner_id,
            file_allowlist=list(configuration.file_allowlist),
            file_allowlist_permits_any_file=configuration.permits_any_file,
            status=configuration.status,
            status_reason=configuration.status_reason,
            credential_configured=credential_configured,
            credential_hint=credential_hint,
            updated_by=configuration.updated_by,
            updated_at=configuration.updated_at,
        )

    @router.put("/design-source", response_model=DesignSourceConfigurationResponse)
    async def put_design_source(
        request_body: DesignSourceConfigurationRequest,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> DesignSourceConfigurationResponse:
        """Save the deployment's one design source. Re-saving clears a degraded status.

        A `token_owner_id` with no `figma` credential stored is saved rather than refused, and
        reported back as `credential_configured: false`. Configuring the account before
        pasting the key is a normal order to do things in, and refusing it would make the two
        panels' order load-bearing.
        """
        require(actor, Permission.DESIGN_SOURCE_MANAGE)
        directory = _require_design_source(request)
        try:
            await directory.save(
                enabled=request_body.enabled,
                token_owner_id=(request_body.token_owner_id or "").strip() or actor.actor_id,
                file_allowlist=tuple(request_body.file_allowlist),
                updated_by=actor.actor_id,
            )
        except DesignSourceConfigurationError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error
        return await get_design_source(request, actor)

    @router.post("/design-source/check", response_model=CredentialCheckResponse)
    async def check_design_source(
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> CredentialCheckResponse:
        """Ask Figma whether the configured owner's token is still accepted.

        Answered in the verdict shape `POST /credentials/{provider}/check` already uses, and
        it acts on exactly one of the three verdicts. A `REFUSED` degrades the configuration,
        because the provider answered and said no. An `UNKNOWN` writes nothing at all: nobody
        got an answer, and degrading a working source because Figma was briefly unreachable
        would pause design resolution over a network blip. An `ACCEPTED` writes nothing
        either -- re-saving the configuration is the one place a degraded status clears, which
        is the remedy the banner asks for and keeps one writer for "it is fine again".
        """
        require(actor, Permission.DESIGN_SOURCE_MANAGE)
        directory = _require_design_source(request)
        configuration = await directory.get()
        if configuration is None:
            return CredentialCheckResponse(
                provider="figma",
                configured=False,
                usable=False,
                detail="Save the design source configuration first.",
            )
        store = _require_store(request)
        descriptor = await store.describe(owner_id=configuration.token_owner_id, provider="figma")
        if not descriptor.configured:
            return CredentialCheckResponse(
                provider="figma",
                configured=False,
                usable=False,
                detail=(
                    "No figma credential is stored for the configured token owner. Enter it "
                    "in the Provider credentials panel, where it appears as the Figma row."
                ),
            )
        try:
            secret = await store.resolve(owner_id=configuration.token_owner_id, provider="figma")
        except SecretStoreError as error:
            return CredentialCheckResponse(
                provider="figma", configured=True, usable=False, detail=str(error)
            )
        if secret is None:
            return CredentialCheckResponse(
                provider="figma",
                configured=True,
                usable=False,
                detail="The stored credential has expired and will not be used.",
            )
        verdict = await verify_stored_credential(request, provider="figma", secret=secret)
        if verdict is CredentialVerdict.REFUSED:
            await directory.mark_degraded(
                configuration.configuration_id,
                reason=(
                    "Figma refused the stored token. It has expired or been revoked, and "
                    "every design citation will be refused until it is replaced."
                ),
            )
        return CredentialCheckResponse(
            provider="figma",
            configured=True,
            usable=True,
            verified=verdict.value,
            detail=_VERIFICATION_DETAIL[verdict],
        )

    @router.get("/account/slack-link", response_model=SlackLinkResponse)
    async def get_slack_link(
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> SlackLinkResponse:
        """Return this identity's own Slack link. Nobody reads anybody else's."""
        directory = _require_slack_links(request)
        link = await directory.get(actor.actor_id)
        if link is None:
            return SlackLinkResponse(user_id=actor.actor_id)
        return SlackLinkResponse(
            user_id=link.user_id,
            slack_user_id=link.slack_user_id,
            notify_scope=link.notify_scope,
        )

    @router.put("/account/slack-link", response_model=SlackLinkResponse)
    async def put_slack_link(
        request_body: SlackLinkRequest,
        request: Request,
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> SlackLinkResponse:
        """Save this identity's own Slack member ID and opt-in scope.

        Self-service and explicit: default scope is `none`, and no permission beyond being
        authenticated -- opting into being mentioned is nobody's decision but the person's.
        """
        directory = _require_slack_links(request)
        member_id = (request_body.slack_user_id or "").strip().upper() or None
        if member_id is not None and not _SLACK_MEMBER_ID.match(member_id):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    "That does not look like a Slack member ID (it looks like U0123ABCDEF, "
                    "from your Slack profile). A wrong id would silently mention nobody."
                ),
            )
        if request_body.notify_scope not in NOTIFY_SCOPES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"unknown notify scope: {request_body.notify_scope}",
            )
        link = await directory.save(
            actor.actor_id,
            slack_user_id=member_id,
            notify_scope=request_body.notify_scope,
        )
        return SlackLinkResponse(
            user_id=link.user_id,
            slack_user_id=link.slack_user_id,
            notify_scope=link.notify_scope,
        )

    return router


def _require_design_source(request: Request) -> Any:
    """Return the deployment's design source directory, or say it keeps none."""
    directory = getattr(request.app.state, "design_source_configuration", None)
    if directory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This deployment does not resolve design references.",
        )
    return directory


def _require_slack_configuration(request: Request) -> Any:
    """Return the deployment's Slack configuration directory, or say it has none."""
    directory = getattr(request.app.state, "slack_configuration", None)
    if directory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This deployment does not deliver Slack notifications.",
        )
    return directory


def _require_slack_links(request: Request) -> Any:
    """Return the per-user Slack link directory, or say the deployment has none."""
    directory = getattr(request.app.state, "slack_user_links", None)
    if directory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This deployment does not deliver Slack notifications.",
        )
    return directory


def _require_store(request: Request) -> Any:
    """Return the deployment's secret store, or say it has none configured."""
    store = get_secret_store(request)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "This deployment does not store provider credentials. Supply them as "
                "request headers instead."
            ),
        )
    return store


def _option_label(platform: AgentPlatform, tier: PerformanceTier) -> str:
    """Name one (platform, tier) pairing the way a person is shown it, in one place.

    Two surfaces publish this label -- the submission form's options and the resolved-model
    table -- and a feature's chosen setup must be spelled identically in both or they read as
    different things.
    """
    return f"{PLATFORM_LABELS[platform]} — {TIER_LABELS[tier]}"


def _agent_platforms(request: Request) -> list[AgentPlatformResponse]:
    """Return every (platform, tier) a feature could name, and whether this deployment runs it.

    An application assembled without resolved model configuration -- an isolated test
    application -- reports the platform this API has always been able to run, at the tier it
    has always run, and no more. Claiming a pairing is available where nothing has resolved a
    model for it would put an option in front of somebody that fails at its first model call,
    minutes later, on a worker, which is the exact failure this field exists to prevent.

    The high entries stay visible and disabled when their platform is unconfigured -- a
    control that vanishes is indistinguishable from a control that never existed. A low or
    medium tier nobody configured is different: the deployment has not chosen to offer it,
    so it is absent rather than disabled, the same preference the completeness rule encodes.
    """
    configs = getattr(request.app.state, "model_configs", None)
    entries: list[AgentPlatformResponse] = []
    for platform in AgentPlatform:
        # Environment-backed tiers only: `custom` is a user's authored setup, and publishing a
        # disabled (platform, custom) pairing here would offer an option that resolves nothing.
        for tier in PerformanceTier.env_backed():
            if configs is None:
                configured = platform is AgentPlatform.OPENAI and tier is PerformanceTier.HIGH
            else:
                configured = configs.is_tier_configured(platform, tier)
            if not configured and tier is not PerformanceTier.HIGH:
                continue
            # One resolution per role, read for both fields: the model a stage runs on and
            # the effort it is sent at are two facts about the same decision, and asking
            # twice would let them be answered by two different ones.
            resolved = (
                {
                    role: configs.get_model_config(role, platform=platform, tier=tier)
                    for role in ModelRole
                }
                if configs is not None and configured
                else {}
            )
            reasoning = resolved.get(ModelRole.REASONING)
            entries.append(
                AgentPlatformResponse(
                    platform=platform.value,
                    performance_tier=tier.value,
                    label=_option_label(platform, tier),
                    configured=configured,
                    models={role.value: config.model for role, config in resolved.items()},
                    reasoning_efforts={
                        role.value: config.reasoning for role, config in resolved.items()
                    },
                    vision_capable=(
                        reasoning is not None and reasoning.model in _declared_vision(request)
                    ),
                )
            )
    return entries


def _declared_vision(request: Request) -> frozenset[str]:
    """Return the models this deployment declared able to read an image.

    Read from the same settings object the ceiling and effort declarations are read from, and
    empty for an application that has none -- an isolated test application, where every
    pairing therefore publishes `vision_capable: false`. That is the fail-closed answer and
    it matches what the start endpoint would do with the same submission.
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return frozenset()
    declared = settings.declared_vision_capable()
    return frozenset(declared)


def _model_setups(request: Request) -> list[ModelSetupResponse]:
    """Project every configured (platform, tier) onto its four resolved roles.

    Read from the same ``ModelConfigService`` every agent resolves a model through, and asked
    the same question -- ``get_model_config`` for a role, a platform and a tier -- so the table
    a person reads here cannot disagree with what a feature actually runs on. A second
    projection assembled from the environment would be a copy, and the first one to drift would
    drift silently.

    An application assembled without resolved model configuration publishes nothing rather
    than a guess. ``/setup`` reports which pairings *exist* and whether they are selectable,
    which is what a submission form needs; a table of models nobody configured is not a
    weaker version of this answer, it is a false one.
    """
    configs = getattr(request.app.state, "model_configs", None)
    if configs is None:
        return []
    setups: list[ModelSetupResponse] = []
    for platform, tier in configs.configured_options():
        roles: list[ModelRoleConfigurationResponse] = []
        for role in ModelRole:
            config = configs.get_model_config(role, platform=platform, tier=tier)
            roles.append(
                ModelRoleConfigurationResponse(
                    role=role.value,
                    model=config.model,
                    reasoning_effort=config.reasoning,
                    # Reported only when the deployment's capability declaration dropped the
                    # configured level, which is the one case the two differ. Equal values
                    # would read as two independent facts about one setting.
                    requested_reasoning_effort=(
                        config.requested_reasoning
                        if config.requested_reasoning != config.reasoning
                        else None
                    ),
                    max_tokens=config.max_tokens,
                    routing_reason=config.routing_reason,
                    model_variable=config.model_variable,
                    reasoning_variable=config.reasoning_variable,
                    resolved_from_legacy_variable=config.resolved_from_legacy_model_variable,
                )
            )
        setups.append(
            ModelSetupResponse(
                platform=platform.value,
                platform_label=PLATFORM_LABELS[platform],
                performance_tier=tier.value,
                tier_label=TIER_LABELS[tier],
                label=_option_label(platform, tier),
                roles=roles,
            )
        )
    return setups


def _model_setup_directory(request: Request) -> Any:
    """Return the deployment's model-setup directory, or nothing when it has none."""
    return getattr(request.app.state, "model_setups", None)


def _require_model_setups(request: Request) -> Any:
    """Return the model-setup directory, or say this deployment does not keep them."""
    directory = _model_setup_directory(request)
    if directory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This deployment does not persist model setups.",
        )
    return directory


def _deployment_declarations(
    request: Request,
) -> tuple[Mapping[str, frozenset[str]], Mapping[str, int]]:
    """Return the deployment's declarations; the shared helper, named locally for brevity."""
    return deployment_model_declarations(request)


def _configured_platform_set(request: Request) -> tuple[AgentPlatform, ...]:
    """Return the platforms this deployment resolves models for, for the warning only."""
    configs = getattr(request.app.state, "model_configs", None)
    if configs is None:
        return tuple(AgentPlatform)
    return tuple(configs.configured_platforms())


def _roles_payload(request_body: SaveModelSetupRequest) -> dict[str, Any]:
    """Serialize the authored roles exactly as entered, for validation and storage."""
    return {key: value.model_dump(mode="json") for key, value in sorted(request_body.roles.items())}


async def _refuse_invalid_setup(
    request: Request, actor: Actor, roles_raw: Mapping[str, Any]
) -> None:
    """Refuse a setup an environment-backed tier would be refused for, at save time.

    Two gates, in order: the validation predicate with its own sentence, then the stored
    credentials for every platform the roles name -- with the roles that pinned each missing
    one, because a platform name alone sends somebody looking for which of four rows caused
    it. A deployment with no secret store skips the second gate: it has always taken
    credentials as request headers, and refusing every setup there would be a regression.
    """
    unsupported, ceilings = _deployment_declarations(request)
    try:
        roles = parse_model_setup_roles(roles_raw)
        validate_model_setup(
            roles,
            unsupported=unsupported,
            ceilings=ceilings,
            configured_platforms=_configured_platform_set(request),
        )
    except ModelConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    store = get_secret_store(request)
    if store is None:
        return
    sentences: list[str] = []
    for platform in dict.fromkeys(entry.platform for entry in roles.values()):
        if await stored_secret(store, actor.actor_id, platform.value) is not None:
            continue
        pinned = [role.value for role in ModelRole if roles[role].platform is platform]
        label = PLATFORM_LABELS[platform]
        verb = "is" if len(pinned) == 1 else "are"
        sentences.append(
            f"{', '.join(pinned)} {verb} pinned to {label} and no {label} credential is "
            "stored for you"
        )
    if sentences:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{'; '.join(sentences)}. Configure it in Settings first.",
        )


async def _saved_setup_response(
    request: Request, actor: Actor, saved: ModelSetup
) -> SavedModelSetupResponse:
    """Project one saved setup, raw, with whether it can currently run and why not."""
    unsupported, ceilings = _deployment_declarations(request)
    validation_error: str | None = None
    warnings: tuple[str, ...] = ()
    roles: dict[ModelRole, ModelSetupRole] = {}
    try:
        roles = parse_model_setup_roles(saved.roles)
        warnings = validate_model_setup(
            roles,
            unsupported=unsupported,
            ceilings=ceilings,
            configured_platforms=_configured_platform_set(request),
        )
    except ModelConfigurationError as error:
        validation_error = str(error)
    platforms = list(
        dict.fromkeys(roles[role].platform.value for role in ModelRole if role in roles)
    )
    store = get_secret_store(request)
    missing: list[str] = []
    if store is not None:
        for platform in platforms:
            if await stored_secret(store, actor.actor_id, platform) is None:
                missing.append(dict(CREDENTIAL_PROVIDERS).get(platform, platform))
    return SavedModelSetupResponse(
        setup_id=saved.setup_id,
        name=saved.name,
        roles={
            key: ModelSetupRoleInput.model_validate(value) for key, value in saved.roles.items()
        },
        created_at=saved.created_at,
        updated_at=saved.updated_at,
        platforms=platforms,
        usable=validation_error is None and not missing,
        validation_error=validation_error,
        warnings=list(warnings),
        missing_credentials=missing,
        # Only the reasoning role, which is the only call a submission's images travel on.
        # A malformed setup that never parsed has no reasoning model, and answers false.
        vision_capable=(
            ModelRole.REASONING in roles
            and roles[ModelRole.REASONING].model in _declared_vision(request)
        ),
    )


def _custom_setup_row(
    saved: ModelSetup,
    *,
    unsupported: Mapping[str, frozenset[str]],
    ceilings: Mapping[str, int],
) -> ModelSetupResponse | None:
    """Project one setup onto the resolved table, or nothing when it no longer resolves.

    Resolved through the same role-addressed service a feature pinned to it runs on, so this
    row cannot disagree with what that feature actually resolves. A setup the declarations
    have invalidated is absent here rather than rendered as runnable -- the authoring list
    still shows it, with the refusal sentence, which is where it can be fixed.
    """
    try:
        service = model_config_service_for_setup(
            {"setup_id": saved.setup_id, "name": saved.name, "roles": saved.roles},
            unsupported=unsupported,
            ceilings=ceilings,
        )
    except ModelConfigurationError:
        return None
    roles: list[ModelRoleConfigurationResponse] = []
    for role in ModelRole:
        platform = service.platform_for_role(role)
        if platform is None:
            return None
        config = service.get_model_config(role, platform=platform)
        roles.append(
            ModelRoleConfigurationResponse(
                role=role.value,
                platform=platform.value,
                model=config.model,
                reasoning_effort=config.reasoning,
                max_tokens=config.max_tokens,
                routing_reason=config.routing_reason,
                # Names the setup, not an environment variable: there is no variable to edit,
                # and the client renders this provenance as "authored in this setup".
                model_variable=config.model_variable,
                reasoning_variable=config.reasoning_variable,
            )
        )
    coding = service.platform_for_role(ModelRole.CODING)
    coding_platform = coding if coding is not None else AgentPlatform.OPENAI
    return ModelSetupResponse(
        platform=coding_platform.value,
        platform_label=PLATFORM_LABELS[coding_platform],
        performance_tier=PerformanceTier.CUSTOM.value,
        tier_label=TIER_LABELS[PerformanceTier.CUSTOM],
        label=saved.name,
        roles=roles,
        origin="custom",
        setup_id=saved.setup_id,
        editable=True,
    )


def _repository_directory(request: Request) -> Any:
    """Return the deployment's saved-repository directory, or nothing when it has none."""
    return getattr(request.app.state, "repository_configurations", None)


def _require_repository_directory(request: Request) -> Any:
    """Return the saved-repository directory, or say this deployment does not keep them."""
    directory = _repository_directory(request)
    if directory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "This deployment does not save repository configurations. Name the "
                "repositories on the feature itself instead."
            ),
        )
    return directory


def _saved_repository(saved: RepositoryConfiguration) -> SavedRepositoryResponse:
    """Project one saved repository onto its public shape."""
    return SavedRepositoryResponse(
        configuration_id=saved.configuration_id,
        name=saved.name,
        repository_url=saved.repository_url,
        default_branch=saved.default_branch,
        repository_type=saved.repository_type,
        created_at=saved.created_at,
        updated_at=saved.updated_at,
    )


def _require_directory(request: Request) -> Any:
    """Return the deployment's user directory, or say it has none configured."""
    directory = getattr(request.app.state, "user_directory", None)
    if directory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This deployment does not manage individual users.",
        )
    return directory


def _credential_response(
    descriptor: Any, *, access: GitHubAccessResponse | None = None
) -> CredentialResponse:
    """Project a descriptor onto its public shape. Neither type can carry a secret."""
    return CredentialResponse(
        provider=descriptor.provider,
        configured=descriptor.configured,
        hint=descriptor.hint,
        created_at=descriptor.created_at,
        updated_at=descriptor.updated_at,
        expires_at=descriptor.expires_at,
        last_used_at=descriptor.last_used_at,
        access=access,
    )


# The hosts the probe actually speaks to. A GitHub Enterprise URL is deliberately not one of
# them: the probe is built against github.com, so it can say nothing about a repository
# elsewhere, and refusing one on its silence would be refusing on the absence of an answer.
_GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})


def _normalised_or_none(url: str) -> str | None:
    """Return the store's spelling of a URL, or nothing when it has no valid one.

    The store's own normaliser, so "already saved", "reachable" and "unique" are decided by
    one function. A value it rejects is not an error here -- both callers are comparing, and
    a URL that cannot be normalised simply matches nothing.
    """
    try:
        return normalise_repository_url(url)
    except RepositoryConfigurationError:
        return None


def _github_access(report: GitHubAccessReport) -> GitHubAccessResponse:
    """Project one probe report onto its public shape, in counts rather than repositories."""
    return GitHubAccessResponse(
        verified=report.verdict.value,
        token_kind=report.token_kind.value,
        scopes=list(report.scopes),
        repositories_listed=report.repositories_listed,
        repository_count=len(report.repositories),
        writable_count=len(report.writable),
        truncated=report.truncated,
        advisories=list(report.advisories),
    )


async def _refuse_unusable_github_token(
    request: Request, *, token: str
) -> GitHubAccessResponse | None:
    """Ask GitHub about a token before it is stored, and refuse one GitHub answered no about.

    Three things stop a save, and every one of them is something GitHub *stated*: it refused
    the token, the classic token it described carries no repository scope, or it listed no
    repositories at all. `GitHubAccessReport.refusal_reason` is where that judgement lives,
    and it is one function so the credential form and any later caller cannot drift apart on
    what "unusable" means.

    Everything else stores. A deployment with no probe, a GitHub that timed out, a listing
    that failed after the token authenticated -- all of them produce a report with no refusal
    reason, and the token is kept. That is the deliberate asymmetry: refusing a real token
    because GitHub was slow would lock somebody out of their own account setup with no way
    forward, and it is a worse outage than the one this exists to prevent.

    Returns what was learned, so the answer rides back on the save rather than costing a
    second call, or `None` when nobody could be asked.
    """
    report = await inspect_github_access(request, token=token)
    reason = report.refusal_reason
    if reason is not None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=reason)
    if report.verdict is CredentialVerdict.UNKNOWN:
        return None
    return _github_access(report)


async def _refuse_unreachable_repository(
    request: Request, *, actor: Actor, repository_url: str
) -> None:
    """Refuse a github.com repository this identity's token cannot build in.

    The same rule the picker offers by, enforced where saving happens, so the guarantee is a
    property of the platform rather than of one form. Only github.com is judged: an
    enterprise host or any other forge is outside what the probe can see, and this says
    nothing about those rather than refusing them.

    Silent whenever nobody could be asked -- no store, no token, no probe, a listing GitHub
    would not answer -- for `_refuse_unusable_github_token`'s reason.
    """
    key = _normalised_or_none(repository_url)
    if key is None or urlsplit(key).netloc.lower() not in _GITHUB_HOSTS:
        return
    store = get_secret_store(request)
    token = await stored_secret(store, actor.actor_id, "github") if store is not None else None
    if not token:
        return
    report = await inspect_github_access(request, token=token)
    if not report.repositories_listed:
        return
    if key in {_normalised_or_none(item.url) for item in report.writable}:
        return
    # Named separately from "cannot reach", because the two have different remedies: one is
    # a grant on the repository, the other is a grant on the token.
    reachable = key in {_normalised_or_none(item.url) for item in report.repositories}
    detail = (
        (
            "Your GitHub token can see this repository but cannot push to it, so the "
            "platform could not open a pull request. Grant write access, then add it."
        )
        if reachable
        else (
            "Your GitHub token cannot reach this repository. Check the URL, or grant the "
            "token access to it, and it will appear in the repository picker."
        )
    )
    raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail)


def _user_response(user: Any) -> UserResponse:
    """Project one identity onto its public shape.

    `has_password` is a boolean the directory already derived; there is no field here a hash
    could occupy, which is the same property `CredentialResponse` has for secrets.
    """
    return UserResponse(
        user_id=user.user_id,
        subject=user.subject,
        display_name=user.display_name,
        roles=list(user.roles),
        disabled=user.disabled,
        has_password=bool(getattr(user, "has_password", False)),
        must_change_password=bool(getattr(user, "must_change_password", False)),
        last_login_at=getattr(user, "last_login_at", None),
        created_at=user.created_at,
    )


async def actor_response(request: Request, actor: Actor) -> ActorResponse:
    """Compose `/me` from the authorization facts and the two account facts beside them.

    Shared with `POST /auth/login`, which answers with the same shape: one composition so a
    field added for the console cannot appear on one of them and not the other.

    The directory lookup is skipped for an identity that is not a row -- the shared platform
    key resolves to `platform-admin`, which is an account once migration 0031 has run and is
    not one before it. A missing row is answered with the defaults rather than an error,
    because "who am I" must keep working during that window.
    """
    directory = getattr(request.app.state, "user_directory", None)
    user = None
    if directory is not None:
        user = await directory.get(actor.actor_id)
    return ActorResponse(
        actor_id=actor.actor_id,
        display_name=actor.display_name,
        authentication=actor.authentication,
        roles=sorted(actor.roles),
        permissions=sorted(item.value for item in actor.permissions),
        subject="" if user is None else user.subject,
        must_change_password=False if user is None else bool(user.must_change_password),
    )


__all__ = [
    "SUPPORTED_PROVIDERS",
    "ActorResponse",
    "SecretStoreUnavailableError",
    "actor_response",
    "create_account_router",
]
