"""Authenticated FastAPI routes for workflow lifecycle control and read models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status

from api.auth import PlatformAuthenticator, current_actor, require, requires
from api.control_plane import (
    ClarificationValidationError,
    RequestScopedCredentials,
    WorkflowConflictError,
    WorkflowNotFoundError,
    artifact_payload,
    log_payload,
)
from api.identity import Actor, Permission
from api.schemas import (
    ArtifactResponse,
    ArtifactsResponse,
    CancelWorkflowRequest,
    ClarificationRequest,
    LogResponse,
    LogsResponse,
    StartWorkflowRequest,
    StartWorkflowResponse,
    TimelineEventResponse,
    TimelineResponse,
    WorkflowResponse,
)
from api.workflow_control_plane import WorkflowControlPlane, WorkflowRecord
from configs.model_roles import normalize_max_output_tokens, normalize_unsupported_reasoning
from services.credential_verification import CredentialVerdict, CredentialVerifier
from services.github_access import GitHubAccessProbe, GitHubAccessReport


def create_workflow_router(*, authenticator: PlatformAuthenticator) -> APIRouter:
    """Build the secured workflow router with one injected authentication dependency.

    **Deprecated, and administrators only.** This is the original single-repository surface.
    It predates both features and workspaces, and its request and response schemas have no
    owner anywhere in them -- so it cannot be scoped to its caller the way `/features/*` is,
    because it has no way to express whose work a workflow is.

    It is therefore confined to `platform-admin`'s workspace by the control plane behind it
    (see `FeatureBackedWorkflowControlPlane`), and gated here on `WORKSPACE_READ_ANY` -- the
    grant that permits reading beyond one's own workspace, which is exactly what this surface
    does. A named permission rather than a role string, for the reason `identity.py` gives:
    `actor.roles` holds raw strings and a scattered `"admin" in actor.roles` would be a
    second authorization authority.

    Route-level rather than a check in each body, and route-level for the reason `/start`
    already was: these routes resolve stored provider credentials as a dependency, and a body
    check would unseal a refused caller's keys on the way to refusing them.

    Deprecated rather than removed. `docs/USER_GUIDE.md`, `docs/DEPLOYMENT.md` and the canary
    all address it, and break-glass access to a single-repository run is a real requirement.
    New work goes through `/features/*`.
    """
    router = APIRouter(
        prefix="/workflow",
        tags=["workflow"],
        dependencies=[
            Depends(authenticator),
            Depends(requires(Permission.WORKSPACE_READ_ANY)),
        ],
        deprecated=True,
    )

    @router.post(
        "/start", response_model=StartWorkflowResponse, status_code=status.HTTP_201_CREATED
    )
    async def start_workflow(
        request_body: StartWorkflowRequest,
        response: Response,
        control_plane: Annotated[WorkflowControlPlane, Depends(get_control_plane)],
        credentials: Annotated[RequestScopedCredentials, Depends(resolve_credentials)],
        actor: Annotated[Actor, Depends(current_actor)],
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> StartWorkflowResponse:
        """Create or replay an idempotent workflow start without accepting tokens in JSON."""
        require(actor, Permission.FEATURE_CREATE)
        # `PRDSubmission` can carry image references and this surface has nowhere to resolve
        # them: it translates a single-repository request into a feature and never sees the
        # attachment store. Refused rather than dropped -- an image that never reaches the
        # model must not be silently discarded -- and the message names the endpoint that
        # does support them instead of describing the limitation.
        if request_body.prd.attachments:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    "this endpoint does not accept image attachments; submit through "
                    "POST /features/start, which resolves and binds them"
                ),
            )
        try:
            result = await control_plane.start(
                request_body,
                idempotency_key=idempotency_key,
                credentials=credentials,
            )
        except WorkflowConflictError as error:
            raise _conflict(error) from error
        if not result.created:
            response.status_code = status.HTTP_200_OK
        return StartWorkflowResponse.model_validate(
            {**workflow_response_values(result.record), "created": result.created}
        )

    @router.post("/resume", response_model=WorkflowResponse)
    async def resume_workflow(
        request_body: ClarificationRequest,
        control_plane: Annotated[WorkflowControlPlane, Depends(get_control_plane)],
        credentials: Annotated[RequestScopedCredentials, Depends(resolve_credentials)],
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> WorkflowResponse:
        """Validate answers against unresolved question IDs and resume the selected workflow."""
        require(actor, Permission.FEATURE_ANSWER_CLARIFICATION)
        try:
            record = await control_plane.resume(
                request_body.workflow_id,
                answers=request_body.answers,
                credentials=credentials,
            )
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        except ClarificationValidationError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error
        except WorkflowConflictError as error:
            raise _conflict(error) from error
        return WorkflowResponse.model_validate(workflow_response_values(record))

    @router.post("/cancel", response_model=WorkflowResponse)
    async def cancel_workflow(
        request_body: CancelWorkflowRequest,
        control_plane: Annotated[WorkflowControlPlane, Depends(get_control_plane)],
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> WorkflowResponse:
        """Cancel an active workflow while leaving its existing artifacts and history queryable."""
        require(actor, Permission.FEATURE_CANCEL)
        try:
            record = await control_plane.cancel(request_body.workflow_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        except WorkflowConflictError as error:
            raise _conflict(error) from error
        return WorkflowResponse.model_validate(workflow_response_values(record))

    @router.get("/{workflow_id}", response_model=WorkflowResponse)
    async def get_workflow(
        workflow_id: str,
        control_plane: Annotated[WorkflowControlPlane, Depends(get_control_plane)],
    ) -> WorkflowResponse:
        """Return the current lifecycle snapshot for one workflow."""
        try:
            record = await control_plane.get_record(workflow_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        return WorkflowResponse.model_validate(workflow_response_values(record))

    @router.get("/{workflow_id}/artifacts", response_model=ArtifactsResponse)
    async def get_artifacts(
        workflow_id: str,
        control_plane: Annotated[WorkflowControlPlane, Depends(get_control_plane)],
    ) -> ArtifactsResponse:
        """List the current artifact handoffs without exposing internal checkpoint structures."""
        try:
            artifacts = await control_plane.artifacts(workflow_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        return ArtifactsResponse(
            workflow_id=workflow_id,
            artifacts=[
                ArtifactResponse.model_validate(artifact_payload(item)) for item in artifacts
            ],
        )

    @router.get("/{workflow_id}/logs", response_model=LogsResponse)
    async def get_logs(
        workflow_id: str,
        control_plane: Annotated[WorkflowControlPlane, Depends(get_control_plane)],
    ) -> LogsResponse:
        """List structured execution logs for one workflow."""
        try:
            logs = await control_plane.logs(workflow_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        return LogsResponse(
            workflow_id=workflow_id,
            logs=[LogResponse.model_validate(log_payload(item)) for item in logs],
        )

    @router.get("/{workflow_id}/timeline", response_model=TimelineResponse)
    async def get_timeline(
        workflow_id: str,
        control_plane: Annotated[WorkflowControlPlane, Depends(get_control_plane)],
    ) -> TimelineResponse:
        """Return chronological lifecycle, artifact, and log events for one workflow."""
        try:
            timeline = await control_plane.timeline(workflow_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        return TimelineResponse(
            workflow_id=workflow_id,
            events=[
                TimelineEventResponse.model_validate(
                    {
                        "timestamp": timestamp,
                        "event_type": event_type,
                        "source": source,
                        "event": event,
                        "details": details,
                    }
                )
                for timestamp, event_type, source, event, details in timeline
            ],
        )

    return router


def get_control_plane(request: Request) -> WorkflowControlPlane:
    """Read the app-scoped injected control plane without treating it as request body data."""
    return cast(WorkflowControlPlane, request.app.state.workflow_control_plane)


async def request_scoped_credentials(
    openai_api_key: Annotated[str | None, Header(alias="X-OpenAI-Api-Key")] = None,
    anthropic_api_key: Annotated[str | None, Header(alias="X-Anthropic-Api-Key")] = None,
    github_token: Annotated[str | None, Header(alias="X-GitHub-Token")] = None,
) -> RequestScopedCredentials:
    """Keep optional provider credentials in memory for one request only, never in state.

    One header per provider, deliberately. A single generic `X-Model-Api-Key` would let a
    caller send a key for the wrong provider and get back an authentication failure that
    names nothing useful.
    """
    return RequestScopedCredentials(
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        github_token=github_token,
    )


def get_secret_store(request: Request) -> Any:
    """Read the app-scoped secret store, or nothing when the deployment stores no secrets."""
    return getattr(request.app.state, "secret_store", None)


async def resolve_credentials(
    request: Request,
    actor: Annotated[Actor, Depends(current_actor)],
    openai_api_key: Annotated[str | None, Header(alias="X-OpenAI-Api-Key")] = None,
    anthropic_api_key: Annotated[str | None, Header(alias="X-Anthropic-Api-Key")] = None,
    github_token: Annotated[str | None, Header(alias="X-GitHub-Token")] = None,
) -> RequestScopedCredentials:
    """Resolve this request's provider credentials, preferring what it brought itself.

    A header always wins. Somebody supplying a key for one request is making a deliberate
    choice about which key does this work, and a stored default silently overriding it would
    be the wrong answer -- so the fallback is only consulted for what the request did not
    supply.

    The return type is unchanged on purpose. Every runner, orchestrator, adapter and agent
    below this point receives exactly what it always did, so persisting credentials adds a
    source without adding a path: nothing downstream learns that a store exists.
    """
    store = get_secret_store(request)
    if store is None:
        return RequestScopedCredentials(
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            github_token=github_token,
        )
    stored_github = None if github_token else await _stored(store, actor, "github")
    return RequestScopedCredentials(
        openai_api_key=openai_api_key or await _stored(store, actor, "openai"),
        anthropic_api_key=anthropic_api_key or await _stored(store, actor, "anthropic"),
        github_token=github_token or stored_github,
        # Only for the stored one. A header-supplied token has no age this platform knows,
        # and claiming the stored credential's date for it would name the wrong credential in
        # a refusal diagnosis -- which is the defect, not the fix.
        github_token_stored_at=(
            None if stored_github is None else await _stored_at(store, actor.actor_id, "github")
        ),
    )


async def _stored_at(store: Any, owner_id: str, provider: str) -> datetime | None:
    """Read when one owner's stored credential was created, or nothing if that cannot be told.

    Best-effort by construction. A missing date makes a refusal diagnosis shorter; a raise
    here would make a feature that has perfectly good credentials fail to start, which is a
    much worse trade than the sentence it protects.
    """
    try:
        descriptor = await store.describe(owner_id=owner_id, provider=provider)
    except Exception:  # noqa: BLE001 - an unreadable date is a shorter sentence, not a failure
        return None
    created = getattr(descriptor, "created_at", None)
    return created if isinstance(created, datetime) else None


async def _stored(store: Any, actor: Actor, provider: str) -> str | None:
    """Read one owner's stored credential, treating an unreadable one as absent."""
    return await stored_secret(store, actor.actor_id, provider)


async def stored_secret(store: Any, owner_id: str, provider: str) -> str | None:
    """Read one owner's stored credential, treating an unreadable one as absent.

    A credential sealed with a key this deployment no longer has cannot be used, and raising
    here would fail an operation that may not even need it. The settings page reports the
    same credential as needing to be re-entered, which is where somebody can act on it.
    """
    try:
        return cast("str | None", await store.resolve(owner_id=owner_id, provider=provider))
    except Exception:
        return None


def get_credential_verifier(request: Request) -> CredentialVerifier | None:
    """Read the app-scoped provider verifier, absent in a deployment that configures none."""
    return cast(
        "CredentialVerifier | None", getattr(request.app.state, "credential_verifier", None)
    )


async def verify_stored_credential(
    request: Request, *, provider: str, secret: str
) -> CredentialVerdict:
    """Ask the provider what it thinks of one credential, and never raise doing it.

    Every way of not getting an answer -- a deployment with no verifier, a provider nobody
    verifies, an empty value, a client that threw -- returns ``UNKNOWN``, and nothing refuses
    on ``UNKNOWN``. That is what keeps this an added diagnosis rather than a new prerequisite:
    a deployment that gains this capability cannot lose the ability to submit a feature
    because a provider was briefly unreachable.
    """
    verifier = get_credential_verifier(request)
    if verifier is None or not secret.strip():
        return CredentialVerdict.UNKNOWN
    try:
        return await verifier.verify(provider=provider, secret=secret)
    except Exception:  # noqa: BLE001 - not getting an answer is a verdict, never a failure
        return CredentialVerdict.UNKNOWN


def get_github_access_probe(request: Request) -> GitHubAccessProbe | None:
    """Read the app-scoped GitHub probe, absent in a deployment that configures none."""
    return cast("GitHubAccessProbe | None", getattr(request.app.state, "github_access_probe", None))


async def inspect_github_access(request: Request, *, token: str) -> GitHubAccessReport:
    """Ask GitHub what one token reaches, and never raise doing it.

    ``verify_stored_credential``'s contract, one question over: every way of not getting an
    answer -- no probe installed, an empty token, a client that threw -- comes back as an
    ``UNKNOWN`` report, whose ``refusal_reason`` is ``None``. A deployment that gains this
    capability therefore cannot lose the ability to save a credential because GitHub was
    briefly unreachable, and one that never installs a probe behaves exactly as it did.
    """
    probe = get_github_access_probe(request)
    if probe is None or not token.strip():
        return GitHubAccessReport(verdict=CredentialVerdict.UNKNOWN)
    try:
        return await probe.inspect(token)
    except Exception:  # noqa: BLE001 - not getting an answer is a report, never a failure
        return GitHubAccessReport(verdict=CredentialVerdict.UNKNOWN)


# Every provider a credential can be stored for, and the name it is shown as. Both are
# needed: `openai` is what the store is keyed on, "OpenAI" is what a setup screen says.
#
# Which of these a given feature actually *requires* is no longer a constant -- it depends on
# the platform that feature was submitted on -- so this is the catalogue and
# `missing_stored_providers` below is the question.
CREDENTIAL_PROVIDERS: tuple[tuple[str, str], ...] = (
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
    ("github", "GitHub"),
)

# What GitHub work needs, whichever platform runs the agents. Every repository operation goes
# through it, so it is required by every live feature.
REPOSITORY_PROVIDER: tuple[str, str] = ("github", "GitHub")


async def stored_credentials(store: Any, owner_id: str) -> RequestScopedCredentials:
    """Build one identity's credentials from what the platform holds for them.

    This is how work that outlives its request gets its keys. A queued feature is executed
    minutes after the caller was answered, so there is no header left to read -- and copying
    one into a queue row would put a provider secret in a table that is not built to hold one.

    Both model providers are resolved because this is built before the feature is known.
    Which one the feature needs is decided where the feature is; handing back only one would
    move that decision here, to the one place that cannot see it.
    """
    if store is None:
        return RequestScopedCredentials(
            openai_api_key=None, anthropic_api_key=None, github_token=None
        )
    github_token = await stored_secret(store, owner_id, "github")
    return RequestScopedCredentials(
        openai_api_key=await stored_secret(store, owner_id, "openai"),
        anthropic_api_key=await stored_secret(store, owner_id, "anthropic"),
        github_token=github_token,
        github_token_stored_at=(
            None if github_token is None else await _stored_at(store, owner_id, "github")
        ),
    )


def deployment_model_declarations(
    request: Request,
) -> tuple[Mapping[str, frozenset[str]], Mapping[str, int]]:
    """Return the deployment's capability and ceiling declarations, normalized.

    An application assembled without settings -- an isolated test application -- has declared
    nothing, which means no pairing is unsupported and no ceiling is known: the same meaning
    an empty declaration has in production, where the deployment is the authority and it has
    said nothing. Read here, beside the credential helpers, because both the setup-authoring
    routes and the feature-start preflight ask the same question of the same state.
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return {}, {}
    return (
        normalize_unsupported_reasoning(settings.declared_unsupported_reasoning()),
        normalize_max_output_tokens(settings.declared_max_output_tokens()),
    )


def required_providers(agent_platform: str) -> tuple[tuple[str, str], ...]:
    """Return the providers one feature needs: its own model platform, and GitHub.

    Neither model provider needs the other's key. A feature submitted on Claude that is
    refused for a missing OpenAI key is being refused for a credential it would never use.
    """
    return required_providers_for_platforms((agent_platform,))


def required_providers_for_platforms(platforms: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """Return the providers a feature pinned to a set of platforms needs, plus GitHub.

    The set-shaped question a custom model setup asks: a mixed setup requires every platform
    its roles name, and still exactly one GitHub credential. A tier feature is the
    single-element case, byte for byte the answer it always got.
    """
    labels = dict(CREDENTIAL_PROVIDERS)
    return (
        *((platform, labels.get(platform, platform)) for platform in dict.fromkeys(platforms)),
        REPOSITORY_PROVIDER,
    )


async def missing_stored_providers(store: Any, owner_id: str, *, agent_platform: str) -> list[str]:
    """Return the display names of providers this identity has not configured for a feature.

    A deployment with no secret store returns none of them: it has always taken credentials
    as request headers, and refusing every feature would be a regression rather than a gate.
    """
    return await missing_stored_providers_for_platforms(
        store, owner_id, platforms=(agent_platform,)
    )


async def missing_stored_providers_for_platforms(
    store: Any, owner_id: str, *, platforms: Iterable[str]
) -> list[str]:
    """Return the providers a platform *set* needs and this identity has not configured.

    Same no-store exemption as the single-platform question: a deployment with no secret
    store has always taken credentials as request headers, and refusing every setup there
    would be a regression, not a gate.
    """
    if store is None:
        return []
    return [
        label
        for provider, label in required_providers_for_platforms(platforms)
        if await stored_secret(store, owner_id, provider) is None
    ]


def workflow_response_values(record: WorkflowRecord) -> dict[str, Any]:
    """Convert the private record shape into the stable public workflow response fields."""
    return {
        "workflow_id": record.workflow_id,
        "status": record.status,
        "current_agent": record.current_agent,
        "confidence": record.confidence,
        "retry_count": record.retry_count,
        "approval_state": record.approval_state,
        "workspace_id": record.workspace_id,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _not_found(error: WorkflowNotFoundError) -> HTTPException:
    """Map a control-plane lookup failure to the normal HTTP not-found representation."""
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))


def _conflict(error: WorkflowConflictError) -> HTTPException:
    """Map invalid lifecycle transitions and idempotency reuse to HTTP conflict responses."""
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error))
