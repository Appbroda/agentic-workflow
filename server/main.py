"""ASGI application entry point for the AI software engineering platform."""

import asyncio
import os
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import structlog
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)
from pydantic import BaseModel
from sqlalchemy import func, select, text

from adapters.figma_adapter import FigmaDesignClient, HttpxFigmaDesignClient
from adapters.llm_adapter import llm_client_for
from adapters.slack_adapter import HttpxSlackClient
from agents.assistant.agent import FeatureAssistant
from api.account_routes import create_account_router
from api.attachment_routes import (
    ATTACHMENT_REQUEST_BYTES,
    ATTACHMENT_UPLOAD_PATH,
    create_attachment_router,
)
from api.auth import PlatformAuthenticator
from api.auth_routes import LoginPolicy, create_auth_router
from api.console import create_console_router
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import FeatureControlPlane
from api.feature_routes import create_feature_router
from api.routes import create_workflow_router, stored_credentials
from api.workflow_control_plane import (
    FeatureBackedWorkflowControlPlane,
    WorkflowControlPlane,
)
from configs.model_roles import AgentPlatform, ModelRole
from configs.settings import load_settings
from prompts.prompt_loader import PromptLoader
from runtime_identity import RuntimeIdentity, load_runtime_identity
from services.action_recovery import ActionRecoveryService
from services.attempt_endings import AttemptEndingCache
from services.cancellation import signal_redis_cancellation
from services.credential_verification import (
    CompositeCredentialVerifier,
    FigmaCredentialVerifier,
    GitHubCredentialVerifier,
)
from services.design_resolution import DesignAssetSink, FigmaDesignResolver
from services.feature_actions import FeatureActionService
from services.feature_chat import FeatureChatService
from services.feature_queue import (
    DatabaseFeatureExecutionQueue,
    FeatureQueueDispatcher,
    InMemoryFeatureExecutionQueue,
    QueuedFeatureExecutor,
)
from services.feature_runtime import ProductionFeatureRunner
from services.github_access import GitHubAccessProbe
from services.login_rate_limit import LoginRateLimiter
from services.recovery_service import RecoveryService
from services.secrets import (
    EncryptedDatabaseSecretStore,
    SecretStoreError,
    SecretStoreUnavailableError,
)
from services.slack_notifications import SlackNotificationDispatcher
from state.failure_diagnosis import FeatureFailureClassification
from storage.action_store import DatabaseFeatureActionStore, InMemoryFeatureActionStore
from storage.attachment_store import DatabaseAttachmentStore
from storage.chat_store import DatabaseChatMessageStore
from storage.db import Database, EphemeralDatabase
from storage.design_source_store import (
    DatabaseDesignSourceConfigurationDirectory,
    InMemoryDesignSourceConfigurationDirectory,
)
from storage.external_operation_store import ExternalOperationJournal
from storage.feature_store import _QUOTA_SETTLED_STATUSES, SqlAlchemyFeatureControlPlane
from storage.model_setup_store import DatabaseModelSetupDirectory, InMemoryModelSetupDirectory
from storage.models import FeatureWorkflowModel
from storage.repository_configuration_store import (
    DatabaseRepositoryConfigurationDirectory,
    InMemoryRepositoryConfigurationDirectory,
)
from storage.slack_store import (
    DatabaseSlackConfigurationDirectory,
    DatabaseSlackUserLinkDirectory,
    InMemorySlackConfigurationDirectory,
    InMemorySlackUserLinkDirectory,
    SlackNotificationStore,
)
from storage.user_store import DatabaseUserDirectory
from storage.workflow_lock import RedisWorkflowLock
from workflows.feature_workflow import FeatureWorkflowOrchestrator


async def bootstrap_administrator_password(
    directory: Any,
    *,
    email: str,
    password: str | None,
) -> str:
    """Give the administrator's account a password, once, from the deployment environment.

    Separate from migration 0031 on purpose. The migration is deterministic SQL that belongs
    in version control and runs identically everywhere; a password is a secret that must not
    be either of those things. It also needs the application's KDF and the process
    environment, neither of which an Alembic revision should reach for.

    **It never overwrites an existing hash.** A container restart with a stale
    `BOOTSTRAP_ADMIN_PASSWORD` would otherwise silently reset the administrator's password to
    the bootstrap value on every boot -- a permanent backdoor that looks like a feature. The
    account is created with `must_change_password`, which is what makes forgetting to remove
    the variable survivable rather than dangerous.

    Returns what it did, as one of four words, so the caller can log it and a test can assert
    on the decision rather than on a log line.
    """
    if password is None:
        return "not_configured"
    user = await directory.find_by_subject(email)
    if user is None:
        # Migration 0031 has not been applied, or the email in the environment is not the
        # administrator's. Either way there is nothing to set a password on, and refusing to
        # start over it would take the deployment down for a recoverable misconfiguration.
        return "no_such_account"
    if user.has_password:
        return "already_set"
    await directory.set_password(user.user_id, password=password, must_change=True)
    return "set"


def _figma_client_factory(
    application: FastAPI,
) -> Callable[[], Awaitable[FigmaDesignClient | None]]:
    """Return a builder that opens the configured design source's credential, or nothing.

    Every absence answers `None` rather than raising, and each is an ordinary state rather
    than a fault: a deployment with no secret store, one that never saved a design source, one
    that saved it and switched it off, and one that configured an owner who has not pasted a
    key yet. A citation could not have been accepted in any of them -- Part C refuses one at
    submission -- so `None` is what "there was nothing to open" looks like to a resolver that
    will not be asked.

    The token exists for the life of the calls that need it and is never returned, logged, or
    put anywhere durable.
    """

    async def build() -> FigmaDesignClient | None:
        directory = getattr(application.state, "design_source_configuration", None)
        store = getattr(application.state, "secret_store", None)
        if directory is None or store is None:
            return None
        configuration = await directory.get()
        if configuration is None or not configuration.enabled:
            return None
        try:
            secret = await store.resolve(owner_id=configuration.token_owner_id, provider="figma")
        except SecretStoreError:
            return None
        if not secret:
            return None
        return HttpxFigmaDesignClient(secret)

    return build


def _design_asset_sink(application: FastAPI) -> DesignAssetSink | None:
    """Return the sink that stores one exported design image, or nothing.

    The submission attachment store, deliberately reused rather than duplicated: it already
    owns image bytes, their media-type sniffing, their caps and their lifecycle, and a second
    table holding the same kind of object would be a second place to get purging wrong.

    The owner is the feature's, not the design source's: these bytes belong to the run that
    exported them and are purged with it.
    """
    store = getattr(application.state, "attachments", None)
    if store is None:
        return None

    async def store_asset(feature_id: str, filename: str, content: bytes) -> str:
        try:
            attachment = await store.create(
                owner_id=feature_id,
                filename=filename,
                media_type="image/png",
                content=content,
            )
            await store.bind([attachment.attachment_id], feature_id=feature_id)
        except Exception:  # noqa: BLE001 - an export that cannot be stored is still exported
            # The bytes still reach the workspace this run; only the durable copy is missing,
            # so a resume re-renders rather than replays. Losing the whole asset over its
            # bookkeeping would be the worse trade.
            return ""
        return str(attachment.attachment_id)

    return store_asset


class HealthStatus(BaseModel):
    """A standard health endpoint response."""

    status: Literal["ok", "unavailable"]
    # Reported so the running implementation can be identified without shelling into the
    # container. A workspace fix that was never rebuilt is indistinguishable from a fix that
    # did not work, and several investigations were spent on exactly that ambiguity.
    build_revision: str = "unknown"
    workflow_schema_version: str
    runtime_compatible: bool


@lru_cache(maxsize=1)
def _build_revision() -> str:
    """Return the build identifier baked into the image, or a local-development marker."""
    return os.environ.get("BUILD_REVISION", "").strip() or "local"


def build_feature_chat(
    settings: Any,
    *,
    control_plane: FeatureControlPlane,
    store: Any,
    actions: Any = None,
) -> FeatureChatService:
    """Wire feature chat, taking each request's provider key the way the agents do.

    Nothing here needs a credential. The platform keeps provider keys request-scoped and the
    deployment holds none -- `docker-compose.yml` passes the model names and deliberately not
    `OPENAI_API_KEY` -- so an assistant built once at startup had no key to use. Doing it
    anyway restart-looped the whole platform on `an OpenAI API key must be supplied`, taking
    down every feature that had nothing to do with chat.
    """
    prompt_loader = PromptLoader()

    def assistant_for(
        credentials: RequestScopedCredentials, *, agent_platform: str
    ) -> FeatureAssistant:
        """Build an assistant on the feature's own platform, from this request's key.

        Scoped to the feature rather than to a deployment default, because there is no
        deployment default any more: a feature planned, implemented and reviewed on Claude is
        explained by Claude.
        """
        platform = AgentPlatform(agent_platform)
        return FeatureAssistant(
            prompt_loader=prompt_loader,
            # Feature chat diagnoses workflow state and explains recovery choices; it is a
            # reasoning task, not an implementation review, even though it shares the
            # reviewer's non-writing operational limits.
            llm_client=llm_client_for(
                platform,
                settings,
                "reviewer",
                api_key=(
                    credentials.anthropic_api_key
                    if platform is AgentPlatform.ANTHROPIC
                    else credentials.openai_api_key
                ),
                model_role=ModelRole.REASONING,
            ),
        )

    return FeatureChatService(
        assistant_for=assistant_for,
        control_plane=control_plane,
        store=store,
        actions=actions,
    )


class ReadinessProbe(Protocol):
    """Check the dependencies required before the API can safely receive workflow traffic."""

    async def is_ready(self) -> bool:
        """Return whether all configured runtime dependencies are reachable."""


class AlwaysReadyProbe:
    """Keep injected unit-test applications independent from external infrastructure."""

    async def is_ready(self) -> bool:
        """Report ready for deliberately isolated application instances."""
        return True


DEFAULT_MAX_REQUEST_BODY_BYTES = 1_048_576

# Where a browser sharing this origin addresses the API. See the router registration for why
# the same routes are served twice.
API_PREFIX = "/api"

# Where the built web client is served. It is not the root: the client's own page for a feature
# is `/features/{id}`, which is this API's URL for the same feature, and sharing an origin one
# of them has to move. It is not the API, whose paths are the established contract.
WEB_CLIENT_PREFIX = "/ui"

# Health is reachable at the root and under the browser's prefix, so anything that treats these
# paths specially has to know about both.
_HEALTH_PATHS = frozenset({"/healthz", "/readyz", f"{API_PREFIX}/healthz", f"{API_PREFIX}/readyz"})


class InfrastructureReadinessProbe:
    """Check PostgreSQL and Redis without exposing connection data in HTTP responses."""

    def __init__(
        self,
        database: Database,
        redis_client: Any,
        *,
        expected_migration_revision: str | None = None,
        recovery_service: RecoveryService | None = None,
    ) -> None:
        """Bind initialized process-wide dependency clients."""
        self._database = database
        self._redis = redis_client
        self._expected_migration_revision = expected_migration_revision
        self._recovery_service = recovery_service

    async def is_ready(self) -> bool:
        """Report only on infrastructure this whole process shares.

        Deliberately says nothing about unresolved external operations. It used to: one
        feature whose push was interrupted made the count non-zero, readiness went
        unavailable, and the middleware below then refused every mutating request on the
        deployment -- for every other feature and every other account -- until somebody
        edited the database. One feature's ambiguity is that feature's problem, and is
        surfaced on the feature that owns it.
        """
        return await self.infrastructure_is_ready()

    async def infrastructure_is_ready(self) -> bool:
        """Check the database, its schema revision, Redis, and this process's startup pass."""
        try:
            async with self._database.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
                if self._expected_migration_revision is not None:
                    revision = await connection.scalar(
                        text("SELECT version_num FROM alembic_version")
                    )
                    if revision != self._expected_migration_revision:
                        return False
            await self._redis.ping()
        except Exception:
            return False
        return self._recovery_service is None or self._recovery_service.completed


def _health_status(
    application: FastAPI, *, status_value: Literal["ok", "unavailable"]
) -> HealthStatus:
    """Return the same observable identity from liveness and readiness endpoints."""
    identity = cast(RuntimeIdentity, application.state.runtime_identity)
    return HealthStatus(
        status=status_value,
        build_revision=identity.build_revision,
        workflow_schema_version=identity.workflow_schema_version,
        runtime_compatible=identity.compatible,
    )


def create_app(
    *,
    platform_api_key: str | None = None,
    control_plane: WorkflowControlPlane | None = None,
    feature_control_plane: FeatureControlPlane | None = None,
    readiness_probe: ReadinessProbe | None = None,
    runtime_identity: RuntimeIdentity | None = None,
    operation_journal: ExternalOperationJournal | None = None,
    feature_chat: FeatureChatService | None = None,
    feature_actions: Any = None,
    feature_queue: Any = None,
    user_directory: Any = None,
    secret_store: Any = None,
    repository_configurations: Any = None,
    model_setups: Any = None,
    attachments: Any = None,
    slack_configuration: Any = None,
    slack_user_links: Any = None,
    design_source_configuration: Any = None,
    slack_client_factory: Any = None,
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
) -> FastAPI:
    """Create the FastAPI application without starting external services during import."""
    if max_request_body_bytes < 1_024:
        msg = "max_request_body_bytes must be at least 1024"
        raise ValueError(msg)
    configure_structlog()
    application = FastAPI(
        title="AI Software Engineering Platform",
        version="0.1.0",
        description="Artifact-driven workflows for AI-assisted software engineering.",
        lifespan=lifespan,
    )

    # Health is also served under the browser's prefix. The routers were moved there and these
    # were not, so the client -- whose base URL is `/api` -- asked for `/api/readyz` and got a
    # 404, and the settings page reported the platform as "no longer exists".
    @application.get("/healthz", response_model=HealthStatus, tags=["health"])
    @application.get(f"{API_PREFIX}/healthz", response_model=HealthStatus, include_in_schema=False)
    async def healthz() -> HealthStatus:
        """Report that the API process is live, and which build is serving."""
        return _health_status(application, status_value="ok")

    @application.get("/readyz", response_model=HealthStatus, tags=["health"])
    @application.get(f"{API_PREFIX}/readyz", response_model=HealthStatus, include_in_schema=False)
    async def readyz(response: Response) -> HealthStatus:
        """Report unavailable when PostgreSQL or Redis cannot support workflow operations."""
        probe = application.state.readiness_probe
        identity = application.state.runtime_identity
        if not identity.compatible or not await probe.is_ready():
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return _health_status(application, status_value="unavailable")
        return _health_status(application, status_value="ok")

    @application.get("/metrics", response_class=PlainTextResponse, tags=["health"])
    async def metrics() -> str:
        """Publish the counters an operator watches, in Prometheus text format.

        Container logs rotate away and the database is the only durable record, so the
        numbers that decide whether to intervene -- how many features are in flight against
        the quota, how much the workspace volume has left, what is waiting to be reconciled --
        have to be readable without opening a psql session.
        """
        return await _render_metrics(application)

    # A feature is accepted, queued, and then executed -- in an isolated application exactly
    # as in a deployment. There is no worker fleet here because there is no lifespan to own
    # one, so each acceptance schedules its own one-shot dispatch; `wait_for_idle` is how a
    # caller observes the result deterministically instead of sleeping.
    # An injected control plane owns its own queue, so take that one rather than watching an
    # unrelated empty one: a dispatcher pointed at the wrong queue never runs anything and
    # looks exactly like a queue that is simply empty.
    if feature_control_plane is None:
        # The durable control plane on a private in-memory database, not a second
        # implementation of the feature lifecycle. What an isolated application exercises is
        # then the merge, the revocation fence, `settle_unfinished_children` and the locking
        # that a deployment actually runs, rather than a parallel one with none of them.
        isolated = EphemeralDatabase()
        feature_control_plane = SqlAlchemyFeatureControlPlane(
            isolated,
            mock_runner=FeatureWorkflowOrchestrator(),
            queue=feature_queue or DatabaseFeatureExecutionQueue(isolated),
            on_queued=lambda _feature_id: application.state.feature_dispatcher.schedule(),
            # The same store the routes resolve against, not a second one: acceptance binds
            # the very rows the upload endpoint wrote, and two stores would make every
            # reference look dangling.
            attachments=attachments,
        )
    application.state.feature_queue = (
        feature_queue
        or getattr(feature_control_plane, "queue", None)
        or InMemoryFeatureExecutionQueue()
    )
    application.state.feature_control_plane = feature_control_plane
    # `/workflow/*` is one repository's worth of the same engine. There is no second
    # orchestrator behind it and no second store: the surface is preserved, the vocabulary is
    # translated at this boundary, and the work is a feature.
    application.state.workflow_control_plane = control_plane or FeatureBackedWorkflowControlPlane(
        feature_control_plane
    )
    application.state.feature_dispatcher = FeatureQueueDispatcher(
        queue=application.state.feature_queue,
        executor=cast(QueuedFeatureExecutor, application.state.feature_control_plane),
        credentials_for=lambda owner_id: stored_credentials(
            getattr(application.state, "secret_store", None), owner_id
        ),
    )
    application.state.operation_journal = operation_journal
    # Where each finished attempt ended, assembled once per attempt and kept. Composed here
    # rather than inside the route because the whole point is that it outlives one request:
    # the operations endpoint is timer-polled, and an ending cannot change once its attempt
    # is over. Derived state, so a restart rebuilds it rather than losing anything.
    application.state.attempt_endings = AttemptEndingCache()
    # No assistant in the default app: it needs a model, and the mock app deliberately reaches
    # no provider. The chat routes answer 503 rather than pretending, and a test injects one.
    application.state.feature_chat = feature_chat
    # Every mutation uses the same action path in mock and production modes. Keeping an
    # in-memory implementation for isolated apps preserves their lightweight setup without
    # creating a second, unjournaled REST execution path.
    application.state.feature_actions = feature_actions or FeatureActionService(
        store=InMemoryFeatureActionStore(), build_revision="local"
    )
    # Absent in an isolated test application and in a deployment that has configured no
    # encryption key. Both are answered as "this deployment does not do that" rather than
    # as a fault, because neither is one.
    application.state.user_directory = user_directory
    application.state.secret_store = secret_store
    # Absent here on purpose. Verification is the only capability in this composition that
    # reaches a provider on a request thread, and an isolated test application must not gain
    # one by default: `verify_stored_credential` answers UNKNOWN without it, which refuses
    # nothing. The production lifespan below installs the real one.
    application.state.credential_verifier = None
    # Absent for the same reason, and answered the same way: `inspect_github_access` returns
    # an UNKNOWN report without one, which refuses no credential and offers no repository
    # menu. An isolated application installs its own when a test needs the answer scripted.
    application.state.github_access_probe = None
    # Saved repositories are ordinary user data rather than infrastructure, so an isolated
    # application gets a working one rather than a 503: the flows that read them are the ones
    # these applications exist to test.
    application.state.repository_configurations = (
        repository_configurations or InMemoryRepositoryConfigurationDirectory()
    )
    # Model setups are user data with execution consequences, and an isolated application
    # gets a working directory for the reason saved repositories do: the flows that author
    # and select them are the ones these applications exist to test.
    application.state.model_setups = model_setups or InMemoryModelSetupDirectory()
    # Attachments are absent by default rather than backed by an in-memory store, unlike
    # model setups and saved repositories above. They are the one piece of user data here
    # that is *bytes*, and an application assembled without one answers 503 -- "this
    # deployment does not persist attachments" -- instead of accepting an upload it will
    # forget. A test that exercises the flow passes one in; the production lifespan below
    # installs the durable store.
    application.state.attachments = attachments
    # The Slack configuration and per-user links are user data with delivery consequences,
    # and an isolated application gets working in-memory directories for the reason model
    # setups do. The client factory is deliberately None here: an isolated application must
    # not gain network-capable code by default -- the check endpoint answers UNKNOWN without
    # it, and no dispatcher runs because there is no lifespan to own one.
    application.state.slack_configuration = (
        slack_configuration or InMemorySlackConfigurationDirectory()
    )
    application.state.slack_user_links = slack_user_links or InMemorySlackUserLinkDirectory()
    # The design source is deployment configuration with resolution consequences, and an
    # isolated application gets a working in-memory directory for the reason the Slack
    # configuration does: the flows that read it are the ones these applications exist to
    # test, and a 503 would make every one of them untestable.
    application.state.design_source_configuration = (
        design_source_configuration or InMemoryDesignSourceConfigurationDirectory()
    )
    application.state.slack_client_factory = slack_client_factory
    application.state.readiness_probe = readiness_probe or AlwaysReadyProbe()
    # No Redis in an isolated application, so no login rate limit: a limiter with no backend
    # allows every attempt, which is the honest behaviour for an app with nowhere to count.
    # The production lifespan below replaces this with one that counts in Redis.
    application.state.login_policy = LoginPolicy(
        rate_limiter=LoginRateLimiter(None, attempts=10, window_seconds=900)
    )
    application.state.runtime_identity = runtime_identity or load_runtime_identity()
    application.state.max_request_body_bytes = max_request_body_bytes
    application.state.allowed_hosts = None
    # A blank key is absent, not a credential. Compose passes variables through as
    # `NAME=${NAME:-}`, so an unset `PLATFORM_API_KEY` arrives as an empty string -- and the
    # authenticator must answer "this deployment cannot check" for it rather than hold a key
    # that no caller can present but that is nonetheless configured. `configs/settings.py`
    # makes the same normalisation for a deployment; this is the seam every test goes through.
    configured_key = (platform_api_key or os.environ.get("PLATFORM_API_KEY") or "").strip()
    authenticator = PlatformAuthenticator(configured_key or None, directory=user_directory)
    # Held so the production lifespan can hand it a user directory built after the routers
    # exist. The routers close over this object, so a replacement made later is never
    # reached and individual identities would silently never authenticate.
    application.state.authenticator = authenticator
    application.include_router(create_workflow_router(authenticator=authenticator))
    application.include_router(create_feature_router(authenticator=authenticator))
    application.include_router(create_account_router(authenticator=authenticator))
    # Its own router because login must be reachable unauthenticated, and every other
    # identity route is behind a router-level authentication dependency.
    application.include_router(create_auth_router(authenticator=authenticator))
    application.include_router(create_attachment_router(authenticator=authenticator))
    application.include_router(create_console_router())
    # The same API again under /api, for a browser sharing this origin.
    #
    # A web client's own route for a feature is `/features/{id}` -- the same URL as the API's.
    # Served from one origin they collide, and the API wins: opening or refreshing a feature
    # page returns JSON instead of the application. Namespacing the browser's calls is the
    # only fix that does not rename one of them.
    #
    # Mounted in addition to the root paths, never instead of them: the existing clients, the
    # deployment's health checks and every test address the API where they always have.
    application.include_router(
        create_workflow_router(authenticator=authenticator), prefix=API_PREFIX
    )
    application.include_router(
        create_feature_router(authenticator=authenticator), prefix=API_PREFIX
    )
    application.include_router(
        create_account_router(authenticator=authenticator), prefix=API_PREFIX
    )
    application.include_router(create_auth_router(authenticator=authenticator), prefix=API_PREFIX)
    application.include_router(
        create_attachment_router(authenticator=authenticator), prefix=API_PREFIX
    )
    application.include_router(create_console_router(), prefix=API_PREFIX)

    @application.middleware("http")
    async def structured_request_logging(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Emit method, route, and status only; never log headers, query strings, or bodies."""
        logger = structlog.get_logger("api.request")
        allowed_hosts = request.app.state.allowed_hosts
        if allowed_hosts is not None and not _host_is_allowed(request.url.hostname, allowed_hosts):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Host is not allowed."},
            )
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            identity = request.app.state.runtime_identity
            probe = request.app.state.readiness_probe
            ready = await probe.is_ready()
            if not identity.compatible or not ready:
                logger.warning(
                    "mutation_rejected_runtime_unavailable",
                    method=request.method,
                    path=_safe_request_route(request),
                    build_revision=identity.build_revision,
                    workflow_schema_version=identity.workflow_schema_version,
                    identity_errors=identity.compatibility_errors,
                )
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"detail": "Runtime is not ready to process mutating requests."},
                )
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                exceeds_limit = int(content_length) > _body_limit_for(request)
            except ValueError:
                exceeds_limit = True
            if exceeds_limit:
                return JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={"detail": "Request body exceeds the configured size limit."},
                )
        try:
            response = await call_next(request)
        except Exception as error:
            # Exception messages and tracebacks may contain provider responses, rejected
            # model values, or repository-controlled paths. Keep request context and a
            # platform-owned type without copying those values into retained service logs.
            logger.error(
                "http_request_failed",
                method=request.method,
                path=_safe_request_route(request),
                error_type=type(error).__name__,
            )
            # Starlette's outer ServerErrorMiddleware deliberately re-raises after sending
            # a 500 so Uvicorn can log the original traceback. Consuming the exception here
            # is therefore part of the confidentiality boundary, not merely response shaping.
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error."},
            )
        if request.url.path not in _HEALTH_PATHS or response.status_code >= 400:
            logger.info(
                "http_request_completed",
                method=request.method,
                path=_safe_request_route(request),
                status_code=response.status_code,
            )
        return response

    @application.middleware("http")
    async def browser_security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Harden browser responses without breaking the API's CDN-backed documentation."""
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.url.path == WEB_CLIENT_PREFIX or request.url.path.startswith(
            f"{WEB_CLIENT_PREFIX}/"
        ):
            # `blob:` in `img-src` because design previews render through
            # `URL.createObjectURL`: the bytes come back through this API with the bearer
            # token an `<img src>` cannot carry, so the element's src is always a blob URL.
            # Without it every preview fails in production and nowhere else -- jsdom does not
            # enforce CSP, so only a served response can prove the directive.
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; "
                "form-action 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                "img-src 'self' data: blob:; font-src 'self'"
            )
        return response

    _serve_web_client(application)
    return application


def _serve_web_client(application: FastAPI) -> None:
    """Serve the built web client from this origin, under its own path.

    Same-origin is what lets the browser call the API without CORS, and without this nothing
    served the build at all -- `npm run dev` worked and a deployment had no way to put the
    application in front of anybody.

    It gets `/ui` rather than the root because the two would otherwise collide: the client's
    own page for a feature is `/features/{id}`, which is this API's URL for the same feature.
    Sharing an origin, one of them has to move, and it is not the API -- those paths are the
    established contract that the existing clients, the tests and the deployment's health
    checks all address. So the browser reads the application under `/ui` and calls the API
    under `/api`, and neither can shadow the other.

    Absent a build the API simply runs alone, which is what the tests and the current image do.
    """
    root = Path(os.environ.get("WEB_CLIENT_ROOT", "/app/client/dist"))
    index = root / "index.html"
    if not index.is_file():
        return

    @application.get("/", include_in_schema=False)
    async def web_client_root() -> Response:
        """Send somebody who opened the bare origin to the application."""
        return RedirectResponse(url=f"{WEB_CLIENT_PREFIX}/")

    @application.get(f"{WEB_CLIENT_PREFIX}/{{path:path}}", include_in_schema=False)
    async def web_client(path: str) -> Response:
        """Return a static file when one matches, and the application shell otherwise.

        The shell is returned for unmatched paths because the client routes in the browser:
        reloading one of its pages has to reach the application rather than a 404. Traversal
        is prevented by resolving and requiring containment -- `..` in a URL must not be able
        to read the server's filesystem.
        """
        candidate = (root / path).resolve()
        if path and root.resolve() in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)


def create_production_app() -> FastAPI:
    """Create the ASGI application that assembles durable storage and live workflow execution."""

    @asynccontextmanager
    async def production_lifespan(application: FastAPI) -> AsyncIterator[None]:
        settings = load_settings()
        identity = load_runtime_identity(
            build_revision=settings.build_revision,
            expected_build_revision=settings.expected_build_revision,
            expected_workflow_schema_version=settings.expected_workflow_schema_version,
            deployment_environment=settings.environment,
        )
        identity.require_compatible()
        application.state.runtime_identity = identity
        structlog.get_logger("runtime.identity").info(
            "runtime_identity_verified",
            build_revision=identity.build_revision,
            workflow_schema_version=identity.workflow_schema_version,
            expected_build_revision=identity.expected_build_revision,
            expected_workflow_schema_version=identity.expected_workflow_schema_version,
        )
        # `load_settings` has already refused a half-configured platform, so reaching here
        # means at least one platform has all four of its roles. Names resolved models, effort
        # levels and safe routing reasons only; never a credential.
        structlog.get_logger("runtime.models").info(
            "model_roles_resolved",
            platforms=[
                platform.value for platform in settings.model_configs.configured_platforms()
            ],
            roles=settings.model_configs.describe(),
            roles_on_legacy_fallback=[
                f"{platform.value}.{role.value}"
                for platform, role in (
                    settings.model_configs.roles_resolved_from_legacy_configuration()
                )
            ],
            minimum_classification_confidence=settings.model_router_min_confidence,
        )
        # The fixable ceiling findings over the environment-backed tiers. Warnings by design
        # (00-todo item 29): an unsatisfiable pairing already refused startup in load_settings,
        # and what remains is legal-but-worth-saying, so it must reach the log without
        # stopping a deployment whose configuration predates the ceiling declaration.
        for detail in settings.model_configuration_warnings:
            structlog.get_logger("runtime.models").warning(
                "model_configuration_warning", detail=detail
            )
        database = Database(
            settings.database_url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            iam_auth_region=settings.database_iam_auth_region,
        )
        redis_client = _create_redis_client(settings.redis_url)
        recovery_task: asyncio.Task[None] | None = None
        slack_task: asyncio.Task[None] | None = None
        try:
            readiness_probe = InfrastructureReadinessProbe(
                database,
                redis_client,
                expected_migration_revision=_migration_head(),
            )
            if not await readiness_probe.is_ready():
                msg = "PostgreSQL and Redis must be reachable before the API starts"
                raise RuntimeError(msg)
            operation_journal = ExternalOperationJournal(database)
            action_store = DatabaseFeatureActionStore(database)
            action_recovery = ActionRecoveryService(action_store, operation_journal)
            recovery_service = RecoveryService(
                operation_journal,
                workspace_root=settings.workspace_root,
                # Actions are reconciled by the same sweep, after the operations their
                # verdicts are read from have settled.
                action_recovery=action_recovery,
                deferred_operation_settle_after_seconds=(
                    settings.deferred_operation_settle_after_seconds
                ),
            )
            await recovery_service.recover_incomplete_operations()
            readiness_probe = InfrastructureReadinessProbe(
                database,
                redis_client,
                expected_migration_revision=_migration_head(),
                recovery_service=recovery_service,
            )
            live_feature_runner = ProductionFeatureRunner(
                settings,
                database=database,
                redis_client=redis_client,
                operation_journal=operation_journal,
            )
            # Where a submitted image's bytes live. Installed only here: absence answers 503
            # and accepts nothing, which is the honest answer for an application that has no
            # durable store to keep bytes in.
            attachment_store = DatabaseAttachmentStore(database)
            application.state.attachments = attachment_store
            feature_queue = DatabaseFeatureExecutionQueue(database)
            application.state.feature_queue = feature_queue
            application.state.feature_control_plane = SqlAlchemyFeatureControlPlane(
                database,
                mock_runner=FeatureWorkflowOrchestrator(
                    workspace_root=settings.workspace_root,
                    max_parallel_workstreams=settings.max_parallel_workstreams,
                ),
                live_runner=live_feature_runner,
                lock=RedisWorkflowLock(redis_client),
                operation_journal=operation_journal,
                max_concurrent_live_features=settings.max_concurrent_live_features,
                # Checked when a live feature is accepted, so a submission the volume cannot
                # house is refused in the response rather than an hour into its run.
                workspace_root=settings.workspace_root,
                workspace_minimum_free_bytes=settings.workspace_minimum_free_bytes,
                cancellation_signaler=lambda feature_id: signal_redis_cancellation(
                    redis_client, scope="feature", identifier=feature_id
                ),
                queue=feature_queue,
                # Nudged rather than polled into life, so a submission starts being worked on
                # in milliseconds instead of at the next tick.
                on_queued=lambda _feature_id: application.state.feature_dispatcher.notify(),
                # The submission's images bind in the same transaction that writes the
                # feature, so this is the same store the upload endpoint wrote them to.
                attachments=attachment_store,
            )
            # A single-repository request is a live one-repository feature. The surface keeps
            # its own request and response schemas; what it no longer keeps is an engine.
            application.state.workflow_control_plane = FeatureBackedWorkflowControlPlane(
                application.state.feature_control_plane, execution_mode="live"
            )
            application.state.operation_journal = operation_journal
            # Individual identity, handed to the authenticator the routers already hold. A
            # freshly constructed authenticator would not be reached: the routers were built
            # during `create_app`, before this database existed.
            user_directory = DatabaseUserDirectory(database)
            application.state.user_directory = user_directory
            application.state.repository_configurations = DatabaseRepositoryConfigurationDirectory(
                database
            )
            application.state.model_setups = DatabaseModelSetupDirectory(database)
            # Slack delivery: durable configuration, per-user links, and the real adapter
            # for the operator-initiated check endpoint. Nothing in the feature runtime
            # touches any of this -- delivery is the periodic dispatcher task below.
            slack_configuration = DatabaseSlackConfigurationDirectory(database)
            slack_user_links = DatabaseSlackUserLinkDirectory(database)
            application.state.slack_configuration = slack_configuration
            application.state.slack_user_links = slack_user_links
            application.state.design_source_configuration = (
                DatabaseDesignSourceConfigurationDirectory(database)
            )
            application.state.slack_client_factory = HttpxSlackClient
            application.state.authenticator.bind(user_directory)
            # After the directory exists and before any worker starts. One line when it acts,
            # so an operator can see it happened; never the password, and never the email
            # beside a failure.
            bootstrap = await bootstrap_administrator_password(
                user_directory,
                email=settings.bootstrap_admin_email,
                password=(
                    settings.bootstrap_admin_password.get_secret_value()
                    if settings.bootstrap_admin_password is not None
                    else None
                ),
            )
            if bootstrap != "not_configured":
                structlog.get_logger("runtime.bootstrap").info(
                    "administrator_password_bootstrap", outcome=bootstrap
                )
            # Counted in Redis rather than in this process: `FEATURE_QUEUE_WORKERS` is two by
            # default, and a per-process counter would make the real limit the configured one
            # times however many processes happen to be serving.
            application.state.login_policy = LoginPolicy(
                rate_limiter=LoginRateLimiter(
                    redis_client,
                    attempts=settings.login_rate_limit_attempts,
                    window_seconds=settings.login_rate_limit_window_seconds,
                ),
                session_ttl_hours=settings.session_token_ttl_hours,
            )
            try:
                application.state.secret_store = EncryptedDatabaseSecretStore(
                    database,
                    encryption_key=(
                        settings.secret_encryption_key.get_secret_value()
                        if settings.secret_encryption_key is not None
                        else None
                    ),
                    previous_encryption_keys=[
                        item.get_secret_value() for item in settings.secret_encryption_previous_keys
                    ],
                )
            except SecretStoreUnavailableError:
                # A deployment that configured no encryption key keeps the behaviour it has
                # always had: provider credentials arrive as request headers and nothing is
                # persisted. Refusing to start over an optional capability would be worse.
                application.state.secret_store = None
                structlog.get_logger("runtime.secrets").info(
                    "provider_credential_store_disabled",
                    reason="no SECRET_ENCRYPTION_KEY configured",
                )
            # Only a real deployment gets one: this is the one capability that dials a
            # provider from a request thread. Its absence is answered as UNKNOWN, which
            # refuses nothing, so installing it can only add a diagnosis.
            # One composite holding one verifier per provider, so `/credentials/{provider}/
            # check?verify=true` works for each of them with no route-level special case and
            # the next provider is a one-line registration here. A provider nobody registered
            # still answers UNKNOWN, which refuses nothing.
            application.state.credential_verifier = CompositeCredentialVerifier(
                GitHubCredentialVerifier(),
                FigmaCredentialVerifier(),
            )
            # The second GitHub question, installed beside the first for its reason: this one
            # also dials a provider from a request thread, and only a real deployment gets to.
            # It is what turns "GitHub accepts this token" into "and here is what it reaches",
            # which is both what the credential form refuses on and what the repository
            # picker offers.
            application.state.github_access_probe = GitHubAccessProbe()
            # Built here, where both halves of the answer live: the design source says whose
            # credential to open, and the secret store opens it. The factory rather than a
            # client because a token must be resolved at the moment it is used and held no
            # longer -- and because the design token deliberately does not travel in
            # `RequestScopedCredentials`, which every layer of a feature can read.
            #
            # On app state under its own name for `slack_client_factory`'s reason: the
            # console's preview endpoint needs exactly this, and one composition point serving
            # both is one place where "which account do designs come from" is decided.
            application.state.figma_client_factory = _figma_client_factory(application)
            live_feature_runner.bind_design_resolver(
                FigmaDesignResolver(
                    client_factory=application.state.figma_client_factory,
                    # Where an exported design image is kept durably, so a resume places the
                    # same bytes without spending another Figma call against a rate limit
                    # measured in requests per month. A deployment with no attachment storage
                    # supplies nothing here and exports nothing, which is the behaviour this
                    # platform had before design assets existed.
                    asset_sink=_design_asset_sink(application),
                )
            )
            feature_actions = FeatureActionService(
                store=action_store,
                journal=operation_journal,
                build_revision=identity.build_revision,
            )
            application.state.feature_actions = feature_actions
            application.state.feature_chat = build_feature_chat(
                settings,
                control_plane=application.state.feature_control_plane,
                store=DatabaseChatMessageStore(database),
                actions=feature_actions,
            )
            application.state.readiness_probe = readiness_probe
            application.state.settings = settings
            # Which platforms a feature may actually be submitted on, and what each role
            # resolves to on them. Published so the submission form offers only what this
            # deployment can run: a dropdown offering Claude where no Anthropic configuration
            # exists is a submission that fails at its first model call, minutes later, on a
            # worker. Model names only -- never a credential.
            application.state.model_configs = settings.model_configs
            application.state.max_request_body_bytes = settings.max_request_body_bytes
            application.state.allowed_hosts = settings.allowed_hosts
            # The workers that turn queued features into running ones. They are started after
            # the secret store exists, because resolving the requesting identity's stored
            # credentials is the first thing a claim does.
            dispatcher = FeatureQueueDispatcher(
                queue=feature_queue,
                executor=cast(QueuedFeatureExecutor, application.state.feature_control_plane),
                credentials_for=lambda owner_id: stored_credentials(
                    application.state.secret_store, owner_id
                ),
                workers=settings.feature_queue_workers,
                lease_seconds=settings.feature_queue_lease_seconds,
            )
            application.state.feature_dispatcher = dispatcher
            dispatcher.start()
            # Bound now rather than at construction: the sweep needs the feature control
            # plane, which needs the secret store and the runners, which are built after the
            # startup recovery pass has already run. Nothing swept features before this, and
            # a feature whose executor died stayed "running" until somebody retired it.
            recovery_service.bind_feature_run_recovery(
                application.state.feature_control_plane,
                stale_after_seconds=settings.feature_run_stale_after_seconds,
                runtime_limit_seconds=settings.feature_runtime_limit_seconds,
            )
            recovery_service.bind_unconfirmed_effect_escalation(
                application.state.feature_control_plane
            )
            # Forgotten uploads are swept by the same periodic pass, rather than by a new
            # scheduler: a second periodic task is a second thing to start, supervise and
            # notice the death of. Bound here for the same startup-ordering reason as the two
            # above -- the recovery service is built before the store exists.
            recovery_service.bind_attachment_sweep(attachment_store)
            recovery_task = asyncio.create_task(
                recovery_service.run_periodic_recovery(),
                name="external-operation-recovery",
            )
            # Its own task, deliberately not a fourth recovery sweep: the recovery sweeps
            # guard against duplicated external effects, and a notification loop does not
            # belong in that budget. Delivery latency is the sweep interval -- a ping can
            # arrive up to 30 seconds late, and that is the price of a run no Slack outage
            # can fail or delay.
            slack_dispatcher = SlackNotificationDispatcher(
                store=SlackNotificationStore(database),
                configuration=slack_configuration,
                user_links=slack_user_links,
                control_plane=application.state.feature_control_plane,
                journal=operation_journal,
                secret_store=application.state.secret_store,
                client_factory=HttpxSlackClient,
                # Whose features may be delivered to a channel the whole deployment reads.
                # One Slack configuration serves everybody, so a non-admin's thread would
                # show their work to everybody in it -- see
                # `docs/AUTHENTICATION_AND_WORKSPACES.md` for the decision and the two
                # alternatives the product owner may still choose.
                user_directory=user_directory,
            )
            slack_task = asyncio.create_task(
                slack_dispatcher.run_periodic(interval_seconds=30.0),
                name="slack-notification-dispatch",
            )
            yield
        finally:
            await application.state.feature_dispatcher.stop()
            if slack_task is not None:
                slack_task.cancel()
                await asyncio.gather(slack_task, return_exceptions=True)
            if recovery_task is not None:
                recovery_task.cancel()
                await asyncio.gather(recovery_task, return_exceptions=True)
            await redis_client.aclose()
            await database.dispose()

    return create_app(lifespan=production_lifespan)


async def _render_metrics(application: FastAPI) -> str:
    """Collect gauges from durable state, degrading to what is available."""
    lines: list[str] = []

    def gauge(name: str, help_text: str, value: float, labels: str = "") -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name}{labels} {value}")

    identity = cast(RuntimeIdentity, application.state.runtime_identity)
    gauge(
        "platform_runtime_compatible",
        "Whether the running build matches the identity the controller expects.",
        1.0 if identity.compatible else 0.0,
    )
    settings = getattr(application.state, "settings", None)
    control_plane = getattr(application.state, "feature_control_plane", None)
    database = getattr(control_plane, "_database", None)
    if database is not None:
        async with database.session() as session:
            rows = await session.execute(
                select(FeatureWorkflowModel.status, func.count()).group_by(
                    FeatureWorkflowModel.status
                )
            )
            for status_value, count in rows.all():
                lines.append(f'platform_features_total{{status="{status_value}"}} {int(count)}')
            in_flight = await session.scalar(
                select(func.count())
                .select_from(FeatureWorkflowModel)
                .where(
                    FeatureWorkflowModel.execution_mode == "live",
                    FeatureWorkflowModel.status.notin_(_QUOTA_SETTLED_STATUSES),
                )
            )
            # How often the platform, rather than the model or the repository, is the reason
            # a run ended. Seven of the last fifty failures were this and nothing reported
            # it, so every fix so far has started from somebody noticing. Deliberately
            # undimensioned: a feature ID or a repository name here would make the series
            # unbounded, and the number that matters is the total.
            platform_defects = await session.scalar(
                select(func.count())
                .select_from(FeatureWorkflowModel)
                .where(
                    FeatureWorkflowModel.state_json["failure_summary"][
                        "root_classification"
                    ].as_string()
                    == FeatureFailureClassification.PLATFORM_DEFECT.value
                )
            )
        gauge(
            "platform_live_features_in_flight",
            "Live features occupying a worker, workspace and provider quota.",
            float(int(in_flight or 0)),
        )
        gauge(
            "platform_defect_failures_total",
            "Features whose terminal record blames this platform rather than the repository.",
            float(int(platform_defects or 0)),
        )
    queue = getattr(application.state, "feature_queue", None)
    if queue is not None:
        gauge(
            "platform_features_awaiting_execution",
            "Features accepted and durably queued that no worker has finished yet.",
            float(await queue.pending_count()),
        )
    journal = getattr(application.state, "operation_journal", None)
    if journal is not None:
        gauge(
            "platform_unresolved_operations",
            "External operations awaiting an operator decision.",
            float(await journal.unresolved_critical_count()),
        )
    if settings is not None:
        gauge(
            "platform_live_feature_quota",
            "Configured concurrent live feature limit; 0 means unbounded.",
            float(settings.max_concurrent_live_features),
        )
        try:
            usage = shutil.disk_usage(settings.workspace_root)
        except OSError:
            usage = None
        if usage is not None:
            gauge(
                "platform_workspace_free_bytes",
                "Free space on the workspace volume, which gates every new clone.",
                float(usage.free),
            )
    return "\n".join(lines) + "\n"


def _create_redis_client(redis_url: str) -> Any:
    """Load Redis lazily so imports and isolated tests never open network clients."""
    import redis.asyncio as redis

    return cast(Any, redis.from_url(redis_url, decode_responses=True))  # type: ignore[no-untyped-call]


def _migration_head() -> str:
    """Read the packaged Alembic head so readiness rejects an un-migrated deployment."""
    configuration = AlembicConfig()
    configuration.set_main_option("script_location", str(Path(__file__).with_name("migrations")))
    head = ScriptDirectory.from_config(configuration).get_current_head()
    if head is None:
        msg = "Alembic migration head is not configured"
        raise RuntimeError(msg)
    return head


def _host_is_allowed(host: str | None, allowed_hosts: list[str]) -> bool:
    """Match an explicit exact host or wildcard without trusting arbitrary Host headers."""
    return host is not None and any(fnmatchcase(host, pattern) for pattern in allowed_hosts)


def _body_limit_for(request: Request) -> int:
    """Return how large this request's body may be, before a byte of it is accepted.

    One exception to the deployment's configured cap, and it is the attachment upload. That
    cap exists to bound JSON payloads and is a megabyte by default; an image is legitimately
    five, and this is also the *only* place a size can be refused before the body is read --
    FastAPI parses a `File()` parameter's multipart body before the handler is entered, so
    the handler's own checks necessarily run on bytes that have already arrived.

    Deliberately not "raise the cap for everything": a five-megabyte JSON body is still not
    something this API has any reason to accept. The allowance is one method on one path,
    under both the bare and the browser-prefixed mounts of it.
    """
    configured = int(request.app.state.max_request_body_bytes)
    if request.method != "POST":
        return configured
    path = request.url.path.rstrip("/")
    if path in (ATTACHMENT_UPLOAD_PATH, f"{API_PREFIX}{ATTACHMENT_UPLOAD_PATH}"):
        return max(configured, ATTACHMENT_REQUEST_BYTES)
    return configured


def _safe_request_route(request: Request) -> str:
    """Return a static route label without logging user-controlled path identifiers."""
    route = request.scope.get("route")
    route_template = getattr(route, "path", None)
    if isinstance(route_template, str) and route_template.startswith("/"):
        return route_template
    path = request.url.path
    if path in _HEALTH_PATHS:
        return path
    if path == "/workflow" or path.startswith("/workflow/"):
        return "/workflow/{route}"
    if path == "/features" or path.startswith("/features/"):
        return "/features/{route}"
    return "/{unmatched}"


def configure_structlog() -> None:
    """Configure structured process logging without adding sensitive HTTP fields to events."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),
        cache_logger_on_first_use=True,
    )


app = create_production_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, proxy_headers=True)
