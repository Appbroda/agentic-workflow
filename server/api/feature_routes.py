"""Authenticated additive API routes for multi-repository parent feature workflows."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, cast

from fastapi import (
    APIRouter,
    Body,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import StreamingResponse

from adapters.figma_adapter import FigmaClientError
from adapters.llm_adapter import LLMAdapterError
from api.auth import (
    PlatformAuthenticator,
    current_actor,
    current_scope,
    require,
    requires,
)
from api.control_plane import (
    RequestScopedCredentials,
    WorkflowBusyError,
    WorkflowConflictError,
    WorkflowNotFoundError,
    artifact_payload,
)
from api.feature_control_plane import (
    FeatureControlPlane,
    FeatureRecord,
    ScopedFeatureControlPlane,
)
from api.feature_schemas import (
    DECIDING_EFFECTIVE_STATUS,
    PUBLISHING_EFFECTIVE_STATUS,
    RESUMING_EFFECTIVE_STATUS,
    RETRYING_EFFECTIVE_STATUS,
    REVISING_EFFECTIVE_STATUS,
    AnswerDesignConflictRequest,
    ApproveContractChangeRequest,
    ApproveRepairRequest,
    CancelFeatureRequest,
    ChatHistoryResponse,
    ChatMessageResponse,
    ClarificationQuestionResponse,
    ClarificationResponse,
    ClarificationState,
    DesignConflictPositionResponse,
    DesignConflictResponse,
    FeatureActionResponse,
    FeatureActionsResponse,
    FeatureArtifactResponse,
    FeatureArtifactsResponse,
    FeatureEventResponse,
    FeatureEventsResponse,
    FeatureExecutionsResponse,
    FeatureListResponse,
    FeatureResponse,
    FeatureSummaryResponse,
    FeatureTimelineEventResponse,
    FeatureTimelineResponse,
    LogbookEntryResponse,
    LogbookRecordResponse,
    LogbookResponse,
    ProposedActionResponse,
    PublishFeatureRequest,
    PullRequestsResponse,
    ReconcileFeatureActionRequest,
    RejectContractChangeRequest,
    RejectRepairRequest,
    RepositoryRepairCommandResponse,
    RepositoryRepairResponse,
    RepositoryRepairsResponse,
    ResumeFeatureRequest,
    RetireFeatureRequest,
    RetryWorkstreamRequest,
    ReviseFeatureRequest,
    SendChatMessageRequest,
    StartFeatureRequest,
    StartFeatureResponse,
    UnresolvedOperationResponse,
    UnresolvedOperationsResponse,
    WorkstreamAttemptResponse,
    WorkstreamOperationRepeatResponse,
    WorkstreamOperationResponse,
    WorkstreamOperationsResponse,
    WorkstreamResponse,
    WorkstreamsResponse,
)
from api.identity import Actor, Permission, WorkspaceScope, permission_for_action
from api.prd_markers import markers_in_all
from api.routes import (
    CREDENTIAL_PROVIDERS,
    REPOSITORY_PROVIDER,
    deployment_model_declarations,
    get_secret_store,
    missing_stored_providers,
    missing_stored_providers_for_platforms,
    resolve_credentials,
    stored_secret,
    verify_stored_credential,
)
from artifacts.design_references import MAX_DESIGN_NODES
from artifacts.schemas import (
    DesignConflictArtifact,
    DesignSnapshotArtifact,
    PRDAttachment,
    RepositoryRepairProposalArtifact,
    TechnicalPRDArtifact,
)
from configs.model_roles import (
    AgentPlatform,
    ModelConfigurationError,
    ModelRole,
    PerformanceTier,
    parse_model_setup_roles,
    validate_model_setup,
)
from services.credential_verification import CredentialVerdict
from services.design_conflict import open_design_conflicts
from services.execution_records import PLANNING_CALL_OPERATION_TYPES, feature_executions
from services.feature_actions import FeatureActionService, action_context_version
from services.feature_chat import ChatActionError, ChatMessage, FeatureChatService
from services.feature_queue import (
    ANSWER_DESIGN_VERDICT,
    PUBLISH,
    RESUME,
    RETRY_WORKSTREAM,
    REVISE,
    FeatureExecutionQueue,
)
from services.logbook import LogbookEvent, feature_logbook, logbook_agents
from services.repository_repair import repair_is_stale
from state.enums import FeatureActionStatus, FeatureWorkflowStatus, RepositoryRepairStatus
from state.external_operations import (
    ExternalOperation,
    ExternalOperationType,
    child_attempt_of,
    is_baseline_validation,
    operation_repeats,
    stream_reissues_of,
)
from state.feature_actions import FeatureAction
from state.feature_models import FeatureWorkflowSnapshot, RepositorySpec
from storage.action_store import ActionConflictError, ActionInProgressError
from storage.attachment_store import (
    MAX_ATTACHMENT_BYTES_PER_FEATURE,
    MAX_ATTACHMENTS_PER_FEATURE,
)
from workflows.feature_workflow import (
    FeatureWorkflowError,
    design_conflict_attempts_remaining,
    require_answerable_design_conflict,
    require_publishable_feature,
    require_publishable_workstream,
    require_retryable_workstream,
)

# The statuses in which unresolved questions mean the platform is still answering them
# itself -- reconnaissance and grounding run inside these. Everything later either has no
# questions left or is waiting on the human, and everything terminal is neither.
_INVESTIGATING_STATUSES = frozenset(
    {
        FeatureWorkflowStatus.ANALYZING_PRD,
        FeatureWorkflowStatus.INSPECTING_REPOSITORIES,
        FeatureWorkflowStatus.PLANNING,
    }
)


# Every journaled operation type, mapped to the human stage name a client renders. The
# mapping lives beside the endpoint that serves it, and it is data rather than derivation:
# a client must never work a stage out from an operation type's spelling. A type added to
# the enum without a row here is served as "other" rather than failing the read.
_OPERATION_STAGES: dict[ExternalOperationType, str] = {
    ExternalOperationType.CLONE_REPOSITORY: "setup",
    ExternalOperationType.CREATE_BRANCH: "setup",
    ExternalOperationType.INSTALL_DEPENDENCIES: "setup",
    ExternalOperationType.RUN_CODING_EXECUTOR: "coding",
    ExternalOperationType.WRITE_FILE_CHANGES: "coding",
    ExternalOperationType.RUN_FORMATTER: "validation",
    ExternalOperationType.RUN_LINTER: "validation",
    ExternalOperationType.RUN_TYPECHECK: "validation",
    ExternalOperationType.RUN_TESTS: "validation",
    ExternalOperationType.RUN_BUILD: "validation",
    ExternalOperationType.RUN_REVIEWER: "review",
    ExternalOperationType.CREATE_COMMIT: "publication",
    ExternalOperationType.PUSH_BRANCH: "publication",
    ExternalOperationType.CREATE_PULL_REQUEST: "publication",
    ExternalOperationType.UPDATE_PULL_REQUEST: "publication",
    ExternalOperationType.CLOSE_PULL_REQUEST: "publication",
    ExternalOperationType.ADD_LABELS: "publication",
    ExternalOperationType.ADD_REVIEWERS: "publication",
    ExternalOperationType.RUN_PRODUCT_MANAGER: "planning",
    ExternalOperationType.RUN_REPOSITORY_RECON: "planning",
    ExternalOperationType.RUN_CLARIFICATION_GROUNDING: "planning",
    ExternalOperationType.RUN_FEATURE_PLANNER: "planning",
}


def operation_stage(operation: ExternalOperation) -> str:
    """Name the human stage one journal row belongs to.

    A function of the row rather than of its type alone, because one type serves two phases.
    The checks the platform runs on an untouched checkout use the same commands, the same
    tool and the same `run_linter`/`run_tests`/`run_build` types as the attempt's own
    validation, so a type-only mapping filed the repository's measurement under the change's
    stage: 201's backend attempt 0 was stopped at coding by the self-review gate, never
    validated its change, and showed three ticked validation rows anyway.

    Those rows belong to `setup`, beside the preflight install they share a revision with --
    the platform preparing a checkout and proving it can run its own commands before the
    Engineer writes anything. Every client reads this one field, so correcting it here
    corrects the drawer, the stage stepper and the graph's Validation node together.
    """
    if is_baseline_validation(operation):
        return "setup"
    return _OPERATION_STAGES.get(operation.operation_type, "other")


# How much history one logbook read composes from. Both are far above what the longest run
# observed so far produced (AB-Feature-108's twelve attempts wrote 341 journal rows), and both
# exist so a pathological feature cannot make one request read an unbounded number of rows.
# The response itself is paged by entry, which is the bound a client actually feels.
_LOGBOOK_EVENT_BOUND = 2000
_LOGBOOK_OPERATION_BOUND = 1000


def create_feature_router(*, authenticator: PlatformAuthenticator) -> APIRouter:
    """Create `/features/*` routes without modifying the established `/workflow/*` API."""
    router = APIRouter(
        prefix="/features",
        tags=["features"],
        dependencies=[Depends(authenticator), Depends(requires(Permission.FEATURE_READ))],
    )

    @router.post(
        "/start",
        response_model=StartFeatureResponse,
        status_code=status.HTTP_201_CREATED,
        # A route dependency rather than a check in the body, because this route resolves the
        # caller's stored provider credentials to build its argument list. Checked in the body,
        # an identity that may not create features had its keys unsealed -- and their
        # last-used timestamp moved -- on the way to being refused.
        dependencies=[Depends(requires(Permission.FEATURE_CREATE))],
    )
    async def start_feature(
        request_body: StartFeatureRequest,
        response: Response,
        request: Request,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        credentials: Annotated[RequestScopedCredentials, Depends(resolve_credentials)],
        actor: Annotated[Actor, Depends(current_actor)],
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> StartFeatureResponse:
        """Accept a feature, queue it durably, and answer without waiting for analysis.

        The response is the feature, queued, with the reference the server allocated for it.
        Nothing has been planned yet -- that happens on a worker afterwards, which is why a
        live submission needs its provider credentials to be *stored* rather than supplied as
        headers here: the work outlives the request that asked for it, and a header value
        cannot be carried into a background task without persisting a secret.
        """
        # A setup is resolved, snapshotted and validated before anything is queued: the
        # snapshot is what the feature pins (G3), and a setup the declarations have since
        # invalidated is refused here with the predicate's own sentence rather than accepted
        # into a run that fails at its first model call (G1's second gate).
        # Is this a request for an answer already given? Asked first, because every check
        # below has "this submission is new" as its subject and a replay is not: the images
        # a replay names are bound to the very feature it is replaying, which is
        # indistinguishable from somebody else's submission unless the question is asked in
        # this order. `start` still owns the answer, and still owns the conflict a reused key
        # with different contents raises.
        # Mapped to 409 exactly as `start`'s own conflicts are: this lookup derives the same
        # idempotency key `start` derives, so the same refusal -- a whitespace-only header --
        # raises the same `WorkflowConflictError` here, one call earlier. Left unmapped it
        # reached the client as a 500, which reads as a platform fault instead of an answer.
        try:
            replaying = await control_plane.replayed_feature_id(
                request_body, idempotency_key=idempotency_key
            )
        except WorkflowConflictError as error:
            raise _conflict(error) from error
        model_setup = await _resolve_model_setup(request, actor, request_body)
        # A citation nothing will ever resolve is refused here, beside the credential checks
        # below and before the single acceptance transaction -- one refusal site rather than
        # two inventions of one. Deliberately outside the `live` branch: a mock feature that
        # cites a design resolves it through the deterministic resolver, so a mock deployment
        # with no design source must refuse the citation exactly as a live one does.
        unresolvable = await _unresolvable_design_reference(request, request_body)
        if unresolvable is not None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=unresolvable
            )
        # And the submission's images, resolved against the same store the upload endpoint
        # wrote them to: existence, ownership, unboundness, and the per-feature caps across
        # the whole set. Settled here, before the transaction, because an unresolvable
        # reference is this request's answer rather than a feature that fails later -- and
        # because acceptance binds these exact records.
        attachments = [] if replaying else await _resolve_attachments(request, actor, request_body)
        # A submission whose evidence will never reach a model is refused here rather than
        # accepted and answered from the prose. The check needs the resolved attachments and
        # the resolved model selection, so it sits after both -- and is skipped for a replay,
        # which was already answered when the images were first accepted.
        # Mock mode is deliberately exempt: it selects no model and reaches no provider, so
        # there is nothing whose capability could be wrong. A mock feature with three
        # screenshots runs, records them, and sends them nowhere.
        if attachments and request_body.execution_mode != "mock":
            _refuse_a_model_that_cannot_read_images(request, request_body, model_setup)
        if request_body.execution_mode == "live":
            if model_setup is not None:
                missing = await _missing_setup_providers(request, actor, model_setup)
            else:
                missing = await _missing_stored_providers(
                    request, actor, agent_platform=request_body.agent_platform
                )
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=(
                        "This feature runs against real repositories, and the platform "
                        "executes it after answering you -- so it needs credentials it can "
                        f"read later. Configure {', '.join(missing)} in Settings first."
                    ),
                )
            # A stored credential the provider has stopped accepting is, for this feature's
            # purposes, the same condition as one that was never configured -- so it is
            # refused in the same place, in the same shape, before anything is queued. Runs
            # 190 and 191 were both accepted against an expired GitHub PAT, spent their fault
            # allowance on clones that could never succeed, and reported the model provider.
            #
            # Only a provider that answered and said no. `verify_stored_credential` returns
            # UNKNOWN for a deployment that verifies nothing and for a provider it could not
            # reach, and neither refuses anything.
            refused = await _provider_refused_credentials(request, actor)
            if refused:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=(
                        "This feature runs against real repositories, and "
                        f"{', '.join(refused)} refused the credential stored for it -- it has "
                        "expired or been revoked. Replace it in Settings first."
                    ),
                )
        try:
            result = await control_plane.start(
                request_body,
                idempotency_key=idempotency_key,
                credentials=credentials,
                # The submitter's workspace, and the identity whose stored credentials the
                # worker will resolve. The same value for both because a submission is the
                # one case where "who asked" and "whose work this is" cannot differ.
                owner_id=actor.actor_id,
                model_setup=model_setup,
                attachments=attachments,
            )
        except (WorkflowConflictError, FeatureWorkflowError) as error:
            raise _conflict(error) from error
        if not result.created:
            response.status_code = status.HTTP_200_OK
        return StartFeatureResponse.model_validate(
            {
                **(await feature_response_values(result.record, request=request)),
                "created": result.created,
            }
        )

    @router.post(
        "/{feature_id}/resume",
        response_model=FeatureResponse,
        # Route-level for the same reason as `/start`: this resolves stored credentials.
        dependencies=[Depends(requires(Permission.FEATURE_ANSWER_CLARIFICATION))],
    )
    async def resume_feature(
        feature_id: str,
        request_body: ResumeFeatureRequest,
        request: Request,
        response: Response,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        credentials: Annotated[RequestScopedCredentials, Depends(resolve_credentials)],
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> FeatureResponse:
        """Resume only a parent-level clarification pause with freshly supplied headers."""
        try:
            action_type = "ANSWER_CLARIFICATION" if request_body.answers else "RESUME_WORKFLOW"
            record = await _execute_durable_mutation(
                request=request,
                response=response,
                control_plane=control_plane,
                actor=actor,
                feature_id=feature_id,
                action_type=action_type,
                payload={
                    "answers": [item.model_dump(mode="json") for item in request_body.answers]
                },
                # No identity argument. The worker resolves the *feature owner's* stored
                # credentials, read off the row inside the control plane -- so an
                # administrator answering a clarification on somebody's feature resumes it
                # with that person's keys rather than their own.
                run=lambda: control_plane.resume(
                    feature_id,
                    answers=request_body.answers,
                    credentials=credentials,
                ),
            )
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        except (
            WorkflowConflictError,
            FeatureWorkflowError,
            ActionConflictError,
            ActionInProgressError,
        ) as error:
            raise _conflict(error) from error
        return FeatureResponse.model_validate(
            await feature_response_values(record, request=request)
        )

    @router.post(
        "/{feature_id}/workstreams/{repository_id}/retry",
        response_model=FeatureResponse,
        # Route-level for the same reason as `/start`: this resolves stored credentials.
        dependencies=[Depends(requires(Permission.FEATURE_RETRY))],
    )
    async def retry_workstream(
        feature_id: str,
        repository_id: str,
        request_body: RetryWorkstreamRequest,
        request: Request,
        response: Response,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        credentials: Annotated[RequestScopedCredentials, Depends(resolve_credentials)],
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> FeatureResponse:
        """Grant one stopped repository more attempts and run it again.

        The platform stops a repository when its retry budget is spent, and ordinary resume
        will not reset that: repeating the same attempt on the same inputs costs money to
        reach the same place. This endpoint is how a person who has read the blocking issues
        overrides that decision, so it requires them to say how many attempts they are buying
        and why -- both of which are recorded against the repository.

        Who granted it is taken from the authenticated identity, not from the request body.
        A client-supplied name is a label somebody typed; it cannot be the audit answer to
        "who overrode the platform's decision", and it was the only answer available before.
        """
        try:
            requested_by = _granted_by(actor, request_body.requested_by)
            payload = {
                "repository_id": repository_id,
                "additional_attempts": request_body.additional_attempts,
                "requested_by": requested_by,
                "reason": request_body.reason,
            }
            record = await _execute_durable_mutation(
                request=request,
                response=response,
                control_plane=control_plane,
                actor=actor,
                feature_id=feature_id,
                repository_id=repository_id,
                action_type="RETRY_WORKSTREAM",
                payload=payload,
                # `requested_by` is the audit sentence naming who granted the attempt, and
                # it travels in the queue entry's payload. Whose stored credentials the
                # worker resolves is a different question with a different answer -- the
                # feature's owner -- and the control plane reads that off the row rather than
                # taking it from here. An administrator granting a retry on somebody's
                # feature buys the attempt with that person's key, not their own.
                run=lambda: control_plane.retry_workstream(
                    feature_id,
                    repository_id,
                    additional_attempts=request_body.additional_attempts,
                    requested_by=requested_by,
                    reason=request_body.reason,
                    credentials=credentials,
                ),
            )
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        except (
            WorkflowConflictError,
            FeatureWorkflowError,
            ActionConflictError,
            ActionInProgressError,
        ) as error:
            raise _conflict(error) from error
        return FeatureResponse.model_validate(
            await feature_response_values(record, request=request)
        )

    @router.post(
        "/{feature_id}/publish",
        response_model=FeatureResponse,
        # Route-level for the same reason as `/start`: this resolves stored credentials. Its
        # own permission rather than the retry one, because its effect is larger: it opens
        # pull requests, and for a rejected workstream the code in one of them passed no
        # review at all.
        dependencies=[Depends(requires(Permission.FEATURE_PUBLISH))],
    )
    async def publish_feature(
        feature_id: str,
        request_body: PublishFeatureRequest,
        request: Request,
        response: Response,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        credentials: Annotated[RequestScopedCredentials, Depends(resolve_credentials)],
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> FeatureResponse:
        """Open the pull requests a feature that did not land is holding.

        A feature that fully landed publishes itself. One that did not publishes nothing until
        somebody says so, because a pull request whose sibling repository does not exist is
        not reviewable work. This is that decision, and it opens strictly more than the
        automatic path ever could: everything that passed review and has no pull request, plus
        every repository whose required checks all passed and whose review rejected it -- each
        labelled in its title and its body for exactly what it is.

        Who asked is taken from the authenticated identity, not from the request body: a
        client-supplied name is a label somebody typed, and this override is the one an audit
        has to be able to attribute.

        Accepted and queued. A push and a provider call per repository run on a worker, so
        nothing about them happens inside this request; what arrives synchronously is the
        refusal, for a feature with nothing publishable.
        """
        try:
            requested_by = _granted_by(actor, request_body.requested_by)
            payload = {"requested_by": requested_by, "reason": request_body.reason}
            record = await _execute_durable_mutation(
                request=request,
                response=response,
                control_plane=control_plane,
                actor=actor,
                feature_id=feature_id,
                action_type="PUBLISH_FEATURE",
                payload=payload,
                # `requested_by` is the audit sentence; whose credentials the worker resolves
                # is the feature's owner, read off the row inside. Distinct questions.
                run=lambda: control_plane.publish_feature(
                    feature_id,
                    requested_by=requested_by,
                    reason=request_body.reason,
                    credentials=credentials,
                ),
            )
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        except (
            WorkflowConflictError,
            FeatureWorkflowError,
            ActionConflictError,
            ActionInProgressError,
        ) as error:
            raise _conflict(error) from error
        return FeatureResponse.model_validate(
            await feature_response_values(record, request=request)
        )

    @router.post(
        "/{feature_id}/revise",
        response_model=FeatureResponse,
        # Route-level for the same reason as `/start`: this causes a run that resolves stored
        # credentials. The permission is the create one rather than the retry one, because a
        # revision is a new run in everything but identity -- it plans, codes, pushes and
        # opens pull requests.
        dependencies=[Depends(requires(Permission.FEATURE_CREATE))],
    )
    async def revise_feature(
        feature_id: str,
        request_body: ReviseFeatureRequest,
        request: Request,
        response: Response,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        credentials: Annotated[RequestScopedCredentials, Depends(resolve_credentials)],
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> FeatureResponse:
        """Ask for changes to a completed feature's published work.

        The revision is applied synchronously -- the request text becomes the run's
        requirement set, each repository moves to a `-V{n}` branch based on the branch it
        published, and the feature leaves `completed` -- and the run itself is queued. Once
        the run's own pull requests are created and read back from the provider, the
        superseded ones are closed with a cross-link comment.

        Who asked is taken from the authenticated identity, not from the request body, for
        the publish route's reason: this re-opens finished work, and an audit has to be able
        to attribute it.
        """
        try:
            requested_by = _granted_by(actor, request_body.requested_by)
            payload = {"requested_by": requested_by, "request": request_body.request}
            record = await _execute_durable_mutation(
                request=request,
                response=response,
                control_plane=control_plane,
                actor=actor,
                feature_id=feature_id,
                action_type="REVISE_FEATURE",
                payload=payload,
                # `requested_by` is the audit sentence; whose credentials the worker resolves
                # is the feature's owner, read off the row inside. Distinct questions.
                run=lambda: control_plane.revise_feature(
                    feature_id,
                    request=request_body.request,
                    requested_by=requested_by,
                    credentials=credentials,
                ),
            )
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        except (
            WorkflowConflictError,
            FeatureWorkflowError,
            ActionConflictError,
            ActionInProgressError,
        ) as error:
            raise _conflict(error) from error
        return FeatureResponse.model_validate(
            await feature_response_values(record, request=request)
        )

    @router.post(
        "/{feature_id}/design-conflicts/{conflict_id}/answer",
        response_model=FeatureResponse,
        # Route-level for the same reason as `/start`: this resolves stored credentials. The
        # permission is the retry one rather than the clarification one, because what this
        # request causes is an attempt on a stopped repository.
        dependencies=[Depends(requires(Permission.FEATURE_RETRY))],
    )
    async def answer_design_conflict(
        feature_id: str,
        conflict_id: str,
        request_body: AnswerDesignConflictRequest,
        request: Request,
        response: Response,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        credentials: Annotated[RequestScopedCredentials, Depends(resolve_credentials)],
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> FeatureResponse:
        """Settle one re-litigated design decision and let the repository act on it.

        49- Part B stops a workstream when a demand it already satisfied is demanded again,
        because another attempt would reverse the decision rather than answer the question. The
        stop is right and is unchanged; this is the exit it lacked. A person reads both
        positions, says which holds and why, and the verdict is carried into the resumed
        attempt as an invariant the retry cannot re-argue.

        Who decided it is taken from the authenticated identity, not from the request body: a
        client-supplied name cannot be the audit answer to "who settled this question", and the
        decision is what the next attempt is bound by.

        Accepted and queued. Everything the verdict frees -- clone, coding call, validation,
        review -- runs on a worker, so nothing about it happens inside this request.
        """
        try:
            decided_by = _granted_by(actor, actor.display_name or actor.actor_id)
            payload = {
                "conflict_id": conflict_id,
                "verdict": request_body.verdict,
                "decision": request_body.decision,
                "decided_by": decided_by,
                "additional_attempts": request_body.additional_attempts,
            }
            record = await _execute_durable_mutation(
                request=request,
                response=response,
                control_plane=control_plane,
                actor=actor,
                feature_id=feature_id,
                action_type="ANSWER_DESIGN_VERDICT",
                payload=payload,
                # `decided_by` is the audit sentence; whose credentials the worker resolves
                # is the feature's owner, read off the row inside. Distinct questions.
                run=lambda: control_plane.answer_design_conflict(
                    feature_id,
                    conflict_id,
                    verdict=request_body.verdict,
                    decision=request_body.decision,
                    decided_by=decided_by,
                    additional_attempts=request_body.additional_attempts,
                    credentials=credentials,
                ),
            )
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        except (
            WorkflowConflictError,
            FeatureWorkflowError,
            ActionConflictError,
            ActionInProgressError,
        ) as error:
            raise _conflict(error) from error
        return FeatureResponse.model_validate(
            await feature_response_values(record, request=request)
        )

    @router.post("/{feature_id}/cancel", response_model=FeatureResponse)
    async def cancel_feature(
        feature_id: str,
        request: Request,
        response: Response,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        actor: Annotated[Actor, Depends(current_actor)],
        request_body: Annotated[CancelFeatureRequest | None, Body()] = None,
    ) -> FeatureResponse:
        """Cancel a parent feature before future child scheduling; existing PRs remain auditable."""
        require(actor, Permission.FEATURE_CANCEL)
        try:
            reason = request_body.reason if request_body is not None else None
            record = await _execute_durable_mutation(
                request=request,
                response=response,
                control_plane=control_plane,
                actor=actor,
                feature_id=feature_id,
                action_type="CANCEL_WORKFLOW",
                payload={"reason": reason},
                run=lambda: control_plane.cancel(
                    feature_id, reason=reason, requested_by=actor.actor_id
                ),
            )
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        except (
            WorkflowConflictError,
            FeatureWorkflowError,
            ActionConflictError,
            ActionInProgressError,
        ) as error:
            raise _conflict(error) from error
        return FeatureResponse.model_validate(
            await feature_response_values(record, request=request)
        )

    @router.get("/operations/unresolved", response_model=UnresolvedOperationsResponse)
    async def list_unresolved_operations(
        request: Request,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        scope: Annotated[WorkspaceScope, Depends(current_scope)],
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> UnresolvedOperationsResponse:
        """Show the operations an operator has to decide about.

        Readiness publishes only a count, which says that something needs attention and
        nothing about what. Without this, reconciling meant querying the database directly.

        `external_operations` has no owner and no foreign key to a feature -- its
        `feature_id` is a nullable plain string, shared by feature and single workflows -- so
        ownership is derived by asking the feature. A caller who holds `WORKSPACE_READ_ANY`
        gets everything, including the rows whose `feature_id` is null: the
        disaster-recovery runbook reconciles across the whole deployment and those rows are
        single-workflow operations that belong to no workspace at all. Everybody else sees
        only the operations of features in their own.

        Filtered after the read rather than by a join, deliberately. The journal is a
        separate store with its own query surface, and this endpoint is bounded at 500 rows
        by the caller: paying one indexed feature lookup per distinct feature in that window
        is cheaper than teaching the journal about ownership, which would put the same
        predicate in a second place.
        """
        journal = getattr(request.app.state, "operation_journal", None)
        if journal is None:
            return UnresolvedOperationsResponse(operations=[])
        operations = await journal.list_unresolved_operations(limit=limit)
        if scope.applies():
            operations = [
                item
                for item in operations
                if item.feature_id is not None
                and await _feature_is_visible(control_plane, item.feature_id)
            ]
        return UnresolvedOperationsResponse(
            operations=[
                UnresolvedOperationResponse(
                    operation_id=item.operation_id,
                    workflow_id=item.workflow_id,
                    feature_id=item.feature_id,
                    repository_id=item.repository_id,
                    operation_type=str(item.operation_type),
                    status=str(item.status),
                    attempt=item.attempt,
                    max_attempts=item.max_attempts,
                    heartbeat_at=item.heartbeat_at,
                )
                for item in operations
            ]
        )

    @router.post("/{feature_id}/retire", response_model=FeatureResponse)
    async def retire_feature(
        feature_id: str,
        request_body: RetireFeatureRequest,
        request: Request,
        response: Response,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> FeatureResponse:
        """Let an operator close out a feature the platform itself refuses to touch.

        A snapshot migrated from before build provenance existed is audit-only, so resume and
        cancel both reject it and it stays in a running status forever. This records the
        decision and its owner instead, without executing anything or discarding evidence.
        """
        require(actor, Permission.FEATURE_RETIRE)
        try:
            operator = _granted_by(actor, request_body.operator)
            record = await _execute_durable_mutation(
                request=request,
                response=response,
                control_plane=control_plane,
                actor=actor,
                feature_id=feature_id,
                action_type="RETIRE_FEATURE",
                payload={"reason": request_body.reason, "operator": operator},
                run=lambda: control_plane.retire(
                    feature_id,
                    reason=request_body.reason,
                    operator=operator,
                ),
            )
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        except (
            WorkflowConflictError,
            FeatureWorkflowError,
            ActionConflictError,
            ActionInProgressError,
        ) as error:
            raise _conflict(error) from error
        return FeatureResponse.model_validate(
            await feature_response_values(record, request=request)
        )

    @router.get("", response_model=FeatureListResponse)
    async def list_features(
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        cursor: Annotated[str | None, Query()] = None,
    ) -> FeatureListResponse:
        """List features newest first so an operator can find one without knowing its ID.

        Bounded and cursor-paged. An unbounded list would grow without limit and eventually
        be the most expensive call the API serves.
        """
        try:
            page = await control_plane.list_features(limit=limit, cursor=cursor)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
            ) from error
        return FeatureListResponse(
            features=[
                FeatureSummaryResponse.model_validate(asdict(item)) for item in page.features
            ],
            next_cursor=page.next_cursor,
        )

    @router.get("/{feature_id}", response_model=FeatureResponse)
    async def get_feature(
        feature_id: str,
        request: Request,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
    ) -> FeatureResponse:
        """Read the durable parent lifecycle snapshot."""
        try:
            record = await control_plane.get_record(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        return FeatureResponse.model_validate(
            await feature_response_values(record, request=request)
        )

    @router.get("/{feature_id}/artifacts/{artifact_id}", response_model=FeatureArtifactResponse)
    async def get_feature_artifact(
        feature_id: str,
        artifact_id: str,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
    ) -> FeatureArtifactResponse:
        """Return one artifact, so a client can open a large payload on demand."""
        try:
            artifacts = await control_plane.artifacts(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        artifact = next((item for item in artifacts if item.artifact_id == artifact_id), None)
        if artifact is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"artifact not found: {artifact_id}",
            )
        return FeatureArtifactResponse.model_validate(artifact_payload(artifact))

    @router.get("/{feature_id}/artifacts", response_model=FeatureArtifactsResponse)
    async def get_feature_artifacts(
        feature_id: str,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        artifact_type: Annotated[str | None, Query()] = None,
        include_payload: Annotated[bool, Query()] = True,
    ) -> FeatureArtifactsResponse:
        """Expose parent and child artifact handoffs while excluding request credentials.

        `include_payload=false` returns the same envelopes without their bodies, so a client
        can list what exists and fetch one on demand. A completed two-repository feature
        currently returns 46 artifacts and roughly 400 KB when every payload is included,
        which is the most expensive read this API serves.
        """
        try:
            artifacts = await control_plane.artifacts(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        selected = [
            item
            for item in artifacts
            if artifact_type is None or item.artifact_type == artifact_type
        ]
        return FeatureArtifactsResponse(
            feature_id=feature_id,
            artifacts=[
                FeatureArtifactResponse.model_validate(
                    {**artifact_payload(item), **({} if include_payload else {"payload": {}})}
                )
                for item in selected
            ],
        )

    @router.get("/{feature_id}/design-preview")
    async def get_design_preview(
        feature_id: str,
        request: Request,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        node_id: Annotated[str, Query(min_length=1, max_length=128)],
    ) -> Response:
        """Render one cited frame as a PNG, on demand, at the version the snapshot recorded.

        **No URL is stored anywhere.** Figma's render URLs expire and the snapshot artifact is
        frozen: a stored URL would go stale inside a record that must not change, and
        re-minting the artifact to refresh one would fake the design refresh Safety rule 4
        forbids. So the render happens here, per request.

        **Pinned to the recorded version.** Figma's images endpoint renders the file's
        *current* state unless told otherwise, so an unpinned re-render would show a design
        that has changed under an unchanged snapshot -- the console quietly disagreeing with
        the text every judge was given.

        The bytes come back through this platform rather than the browser being handed a
        storage URL, because the console reads this with its own bearer token like every other
        read. `Cache-Control: private` because a design is this deployment's, and a short
        max-age because a slow preview is a preview and not a defect.

        Never an input to an agent. Safety rule 1 holds regardless of what any adapter can
        carry: the design reaches every role as text, and this endpoint has exactly one caller,
        which is a browser.
        """
        try:
            artifacts = await control_plane.artifacts(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        snapshot = next(
            (item for item in reversed(artifacts) if isinstance(item, DesignSnapshotArtifact)),
            None,
        )
        node = (
            next((item for item in snapshot.nodes if item.node_id == node_id), None)
            if snapshot is not None
            else None
        )
        if snapshot is None or node is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"this feature's design snapshot does not contain the frame {node_id}. "
                    "A frame the resolution omitted, could not find or could not read has no "
                    "preview, and the snapshot says which of those happened."
                ),
            )
        version = next(
            (item.file_version for item in snapshot.files if item.file_key == node.file_key),
            "",
        )
        factory = getattr(request.app.state, "figma_client_factory", None)
        client = None if factory is None else await factory()
        if client is None or not version:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "This deployment cannot render design previews: it has no usable Figma "
                    "credential for its configured design source. The snapshot's text is "
                    "unaffected."
                ),
            )
        try:
            url = await client.render_preview(node.file_key, node_id, version=version)
            rendered = await client.fetch_rendered_bytes(url)
        except FigmaClientError as error:
            # Classified, and never a provider message. A refusal here costs a picture and
            # nothing else: the design every judge was given is the snapshot's text.
            raise HTTPException(
                status_code=(
                    status.HTTP_504_GATEWAY_TIMEOUT
                    if error.retryable
                    else status.HTTP_502_BAD_GATEWAY
                ),
                detail=(
                    f"the design provider did not return a preview ({error.error_code}). "
                    "The snapshot's text is unaffected."
                ),
            ) from error
        return Response(
            content=rendered,
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=300"},
        )

    @router.get("/{feature_id}/clarification", response_model=ClarificationResponse)
    async def get_clarification(
        feature_id: str,
        request: Request,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
    ) -> ClarificationResponse:
        """Return the questions this feature is waiting on, resolved from the current PRD.

        The client must not work out which technical PRD is current: after reconnaissance and
        a clarification round the newest is a revision, and matching that lineage is a server
        rule with its own regression history.
        """
        try:
            record = await control_plane.get_record(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        state = record.state
        technical_prd = next(
            (item for item in reversed(state.artifacts) if isinstance(item, TechnicalPRDArtifact)),
            None,
        )
        active_intent = await _active_feature_intent(request, feature_id)
        open_questions = technical_prd.unresolved_questions if technical_prd is not None else []
        answers = (
            technical_prd.metadata.get("clarification_answers", {})
            if technical_prd is not None
            else {}
        )
        awaiting_answers = (
            state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
            and active_intent is None
            and bool(open_questions)
        )
        # The design decisions a workstream stopped on. Advertised only while no queued
        # mutation owns the feature, for the reason the actions list is: an accepted verdict is
        # already the feature's next step, and offering a second one produces a conflict.
        conflicts = (
            [
                _design_conflict_response(state, conflict)
                for conflict in open_design_conflicts(state.artifacts)
            ]
            if active_intent is None
            else []
        )
        clarification_state: ClarificationState
        if awaiting_answers:
            # Falling back to the human after a failed grounding attempt is the designed
            # behaviour, and it reads differently: the platform tried to answer these
            # itself and could not.
            clarification_state = (
                "asked_after_grounding_failure"
                if state.clarification_grounding_failed
                else "awaiting_answers"
            )
        elif (
            bool(open_questions)
            and active_intent is None
            and state.status in _INVESTIGATING_STATUSES
        ):
            # AB-Feature-173's operator watched this exact shape for 35 minutes rendered as
            # "waiting for you": questions on the PRD, the platform still reading checkouts
            # and grounding answers. The questions are the platform's open items here.
            clarification_state = "investigating"
        elif conflicts:
            # Ranked below the three requirement states rather than above them, and the order
            # is a rule about who is being asked what. A pending technical-PRD question stops
            # the whole feature before any repository runs; a design conflict stops one
            # workstream after coding. The two cannot both be pending in practice -- children
            # do not start until the questions are answered -- but if they ever were, the
            # question that gates everything is the one to render.
            clarification_state = "awaiting_design_verdict"
        else:
            clarification_state = "idle"
        return ClarificationResponse(
            feature_id=feature_id,
            awaiting_answers=awaiting_answers,
            clarification_state=clarification_state,
            technical_prd_artifact_id=(
                technical_prd.artifact_id if technical_prd is not None else None
            ),
            clarification_rounds=state.clarification_rounds,
            max_clarification_rounds=state.max_clarification_rounds,
            questions=[
                ClarificationQuestionResponse(
                    question_id=item.question_id,
                    question=item.question,
                    rationale=item.rationale,
                    required=item.required,
                    suggested_answer=item.suggested_answer,
                    suggestion_source=item.suggestion_source,
                    suggestion_confidence=item.suggestion_confidence,
                )
                # Questions are an *actionable* contract only once the worker has finished
                # assembling the final set and released the queue intent -- submitting
                # against a provisional list is still refused by `awaiting_answers` and by
                # the answer validator. But the list itself is shown while the platform
                # investigates: 173's operator stared at an empty panel for 35 minutes while
                # the questions it was answering sat unrendered in the PRD. The set may
                # still grow while `investigating`; that is what the state name says.
                for item in (open_questions if clarification_state != "idle" else [])
            ],
            previous_answers={
                str(key): str(value)
                for key, value in (answers.items() if isinstance(answers, dict) else [])
            },
            design_conflicts=conflicts,
        )

    @router.get("/{feature_id}/workstreams", response_model=WorkstreamsResponse)
    async def get_workstreams(
        feature_id: str,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
    ) -> WorkstreamsResponse:
        """List repository child statuses, branches, workspaces, and blocked work separately."""
        try:
            workstreams = await control_plane.workstreams(feature_id)
            record = await control_plane.get_record(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        specs = {item.repository_id: item for item in record.state.repository_specs}
        return WorkstreamsResponse(
            feature_id=feature_id,
            workstreams=[
                WorkstreamResponse.model_validate(
                    {
                        **item.model_dump(mode="python", exclude={"preflight_result"}),
                        **_repository_identity(specs.get(item.repository_id)),
                        "available_actions": _workstream_actions(
                            record, repository_id=item.repository_id
                        ),
                        **_workstream_publication(record, repository_id=item.repository_id),
                    }
                )
                for item in workstreams
            ],
        )

    @router.get(
        "/{feature_id}/workstreams/{repository_id}/operations",
        response_model=WorkstreamOperationsResponse,
    )
    async def get_workstream_operations(
        feature_id: str,
        repository_id: str,
        request: Request,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> WorkstreamOperationsResponse:
        """Serve one repository's journal rows: what is running, since when, and is it alive.

        Every operational question of the 185-194 verification cycle -- "is it stuck", "which
        call is this", "how old is the heartbeat" -- was answered by querying
        `external_operations` by hand while the UI said "Running". This is that query, as the
        read-only endpoint it should have been. Timestamps are returned raw and never as ages:
        the client computes "2s ago" against its own clock, so the response is deterministic.

        A repository with no operations answers an empty list rather than 404, because "this
        repository has recorded nothing yet" is an ordinary state of a queued workstream.

        The rows are joined by one per-attempt block saying where each finished attempt
        ended. It rides this response rather than a second endpoint because the drawer that
        renders it must keep costing no request of its own, and it is assembled once per
        finished attempt rather than on every poll: an ending is immutable once its attempt
        is over, and the in-flight attempt has no ending to assemble.
        """
        try:
            record = await control_plane.get_record(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        endings = getattr(request.app.state, "attempt_endings", None)
        journal = getattr(request.app.state, "operation_journal", None)
        operations = (
            await journal.list_operations_for_repository(feature_id, repository_id, limit=limit)
            if journal is not None
            else []
        )
        # Why each repeated row repeats, computed across the rows this response serves. Read
        # from the operation's own `repository_revision` and `command_fingerprint` columns, so
        # nothing is parsed out of `safe_metadata` and that channel stays as narrow as the one
        # field published from it.
        repeats = operation_repeats(operations)
        return WorkstreamOperationsResponse(
            feature_id=feature_id,
            repository_id=repository_id,
            operations=[
                WorkstreamOperationResponse(
                    operation_id=item.operation_id,
                    operation_type=str(item.operation_type),
                    stage=operation_stage(item),
                    status=str(item.status),
                    attempt=item.attempt,
                    max_attempts=item.max_attempts,
                    # The attempt the row belongs to, so a reader can group a preserved-workspace
                    # retry's rows by attempt instead of guessing from where a clone appears --
                    # a retry that edits in place never re-clones, and run 197 BE showed three
                    # attempts rendered as one because of it.
                    child_attempt=child_attempt_of(item),
                    started_at=item.started_at,
                    heartbeat_at=item.heartbeat_at,
                    completed_at=item.completed_at,
                    error_code=item.error_code,
                    repeat=(
                        WorkstreamOperationRepeatResponse(
                            kind=repeats[item.operation_id].kind,
                            detail=repeats[item.operation_id].detail,
                        )
                        if item.operation_id in repeats
                        else None
                    ),
                    stream_reissues=stream_reissues_of(item),
                )
                for item in operations
            ],
            attempts=[
                WorkstreamAttemptResponse(
                    attempt=ending.attempt,
                    ended_by=str(ending.ended_by),
                    stage=ending.stage,
                    detail=ending.detail,
                    workspace=ending.workspace,
                    self_review_outcome=ending.self_review_outcome,
                    self_review_corrected_files=ending.self_review_corrected_files,
                    source_repair_passes=ending.source_repair_passes,
                    stream_reissues=ending.stream_reissues,
                )
                for ending in (
                    endings.endings_for(record.state, repository_id) if endings is not None else []
                )
            ],
        )

    @router.get("/{feature_id}/executions", response_model=FeatureExecutionsResponse)
    async def get_feature_executions(
        feature_id: str,
        request: Request,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
    ) -> FeatureExecutionsResponse:
        """Say who performed each transition in this feature's execution.

        Served from the same parent snapshot every other read uses, in one request, because a
        graph must not cost one request per arrow. The derivation reads only durable records:
        it has no access to the model configuration, so a historical transition cannot be
        re-labelled with a model a deployment configured afterwards.

        The journal rows for the pre-coding model calls ride along, because they are the
        only record with a start time and a heartbeat while a planning call is still in
        flight -- the stage rows above appear only once an artifact exists, which is
        exactly the window AB-Feature-173 spent forty-one minutes inside.
        """
        try:
            record = await control_plane.get_record(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        journal = getattr(request.app.state, "operation_journal", None)
        operations = (
            await journal.list_operations_for_feature(
                feature_id, operation_types=PLANNING_CALL_OPERATION_TYPES
            )
            if journal is not None
            else []
        )
        return FeatureExecutionsResponse(
            feature_id=feature_id,
            executions=feature_executions(record.state, operations=operations),
        )

    @router.get("/{feature_id}/pull-requests", response_model=PullRequestsResponse)
    async def get_pull_requests(
        feature_id: str,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
    ) -> PullRequestsResponse:
        """List every pull request opened for the feature, whether or not it completed.

        Not gated on integration review. A repository that passed its own review opens its
        pull request even when the feature as a whole ended needing a human, and hiding
        those was how finished work went unnoticed.
        """
        try:
            pull_requests = await control_plane.pull_requests(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        return PullRequestsResponse(
            feature_id=feature_id,
            pull_requests=[
                FeatureArtifactResponse.model_validate(artifact_payload(item))
                for item in pull_requests
            ],
        )

    @router.get("/{feature_id}/chat", response_model=ChatHistoryResponse)
    async def get_chat(
        feature_id: str,
        chat: Annotated[FeatureChatService, Depends(get_feature_chat)],
    ) -> ChatHistoryResponse:
        """Return the conversation, so reopening a feature does not lose it."""
        try:
            messages = await chat.history(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        return ChatHistoryResponse(
            feature_id=feature_id, messages=[_chat_message(item) for item in messages]
        )

    @router.post("/{feature_id}/chat", response_model=ChatHistoryResponse)
    async def send_chat(
        feature_id: str,
        request_body: SendChatMessageRequest,
        chat: Annotated[FeatureChatService, Depends(get_feature_chat)],
        credentials: Annotated[RequestScopedCredentials, Depends(resolve_credentials)],
    ) -> ChatHistoryResponse:
        """Answer one question about this feature and return both turns.

        The assistant needs a provider key, and this platform keeps those request-scoped: the
        deployment holds none, so the key comes from this request's header exactly as it does
        for the agents that do the work.
        """
        try:
            messages = await chat.send(feature_id, request_body.message, credentials=credentials)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        except LLMAdapterError as error:
            # No usable key in the header and none in the environment. That is a fact about
            # the request, not a fault: say so rather than recording an unanswered question.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
            ) from error
        return ChatHistoryResponse(
            feature_id=feature_id, messages=[_chat_message(item) for item in messages]
        )

    @router.post(
        "/{feature_id}/chat/{message_id}/confirm",
        response_model=ChatMessageResponse,
        # A route dependency rather than a check in the body: the assistant may not be
        # configured, and an unauthorized caller must be refused before being told that.
        dependencies=[Depends(requires(Permission.ACTION_EXECUTE))],
    )
    async def confirm_chat_action(
        feature_id: str,
        message_id: int,
        chat: Annotated[FeatureChatService, Depends(get_feature_chat)],
        credentials: Annotated[RequestScopedCredentials, Depends(resolve_credentials)],
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> ChatMessageResponse:
        """Run a proposal a person confirmed, through the ordinary control plane.

        The assistant never reaches this path. It proposes; this executes, and only what the
        control plane already knows how to do -- as a durable action carrying who asked, so
        the same confirmation arriving twice produces one effect and a crash midway leaves
        something the platform can reason about afterwards.

        Authorization is a route dependency above, because chat must never become a way to
        reach a mutation the same identity would be refused at its own endpoint.
        """
        try:
            message = await chat.confirm(
                feature_id,
                message_id,
                credentials=credentials,
                actor_id=actor.actor_id,
                actor_display_name=actor.display_name,
                # The route dependency above can only ask "may this identity execute a chat
                # action" -- it has a message id and nothing else. What the action turns out
                # to be is known inside, so the specific check happens there.
                authorize=lambda action_type: require(actor, permission_for_action(action_type)),
            )
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        except ChatActionError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        except (WorkflowConflictError, FeatureWorkflowError) as error:
            raise _conflict(error) from error
        return _chat_message(message)

    @router.get("/{feature_id}/actions", response_model=FeatureActionsResponse)
    async def get_feature_actions(
        feature_id: str,
        request: Request,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> FeatureActionsResponse:
        """List what people have asked this feature to do, and what became of each.

        This is what makes a confirmed action survive a browser refresh: the client follows
        the action rather than the request that started it, so closing the tab mid-execution
        loses nothing.

        The action records live outside the feature graph -- their own table, keyed by
        `feature_id` with no cascade -- so this gates on the feature explicitly. An action
        record names who asked for what and what became of it, which is a description of
        somebody's work.
        """
        try:
            await control_plane.require_visible(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        actions = getattr(request.app.state, "feature_actions", None)
        if actions is None:
            return FeatureActionsResponse(feature_id=feature_id, actions=[])
        records = await actions.store.list_for_feature(feature_id, limit=limit)
        return FeatureActionsResponse(
            feature_id=feature_id, actions=[_action_response(item) for item in records]
        )

    @router.get("/{feature_id}/actions/{action_id}", response_model=FeatureActionResponse)
    async def get_feature_action(
        feature_id: str,
        action_id: str,
        request: Request,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
    ) -> FeatureActionResponse:
        """Return one action, so a client can watch a long one without polling everything."""
        try:
            await control_plane.require_visible(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        actions = getattr(request.app.state, "feature_actions", None)
        action = None if actions is None else await actions.store.get(action_id)
        if action is None or action.feature_id != feature_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"action not found: {action_id}"
            )
        return _action_response(action)

    @router.post(
        "/{feature_id}/actions/{action_id}/reconcile",
        response_model=FeatureActionResponse,
    )
    async def reconcile_feature_action(
        feature_id: str,
        action_id: str,
        request_body: ReconcileFeatureActionRequest,
        request: Request,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> FeatureActionResponse:
        """Close an uncertain action after an administrator verifies its actual outcome.

        This writes only the action audit record. It never invokes the stored action, mutates
        workflow state, or repeats a provider operation.

        `ACTION_RECONCILE` is an administrator grant, and an administrator holds
        `WORKSPACE_READ_ANY`, so the scope check below passes for them on any feature. It is
        still made rather than skipped: the permission and the ownership predicate answer
        different questions, and a deployment that later grants reconciliation to an operator
        would otherwise have handed them everybody's actions.
        """
        require(actor, Permission.ACTION_RECONCILE)
        try:
            await control_plane.require_visible(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        actions = cast(FeatureActionService, request.app.state.feature_actions)
        action = await actions.store.get(action_id)
        if action is None or action.feature_id != feature_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"action not found: {action_id}"
            )
        try:
            reconciled = await actions.operator_reconcile(
                action_id,
                actor_id=actor.actor_id,
                succeeded=request_body.outcome == "succeeded",
                reason=request_body.reason,
            )
        except ActionConflictError as error:
            raise _conflict(error) from error
        return _action_response(reconciled)

    @router.get("/{feature_id}/repairs", response_model=RepositoryRepairsResponse)
    async def get_repairs(
        feature_id: str,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
    ) -> RepositoryRepairsResponse:
        """List every repository repair proposed for this feature, and its current state."""
        try:
            repairs = await control_plane.repairs(feature_id)
            record = await control_plane.get_record(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        children = record.state.child_workflows
        return RepositoryRepairsResponse(
            feature_id=feature_id,
            repairs=[
                _repair_response(
                    item,
                    current_revision=(
                        children[item.repository_id].current_revision
                        if item.repository_id in children
                        else None
                    ),
                )
                for item in repairs
            ],
        )

    @router.post(
        "/{feature_id}/repairs/{repair_id}/approve",
        response_model=FeatureResponse,
        # Route-level so an identity that may not approve repairs is refused before this
        # route resolves its stored credentials, and before it is told which acknowledgement
        # the body was missing.
        dependencies=[Depends(requires(Permission.REPAIR_APPROVE))],
    )
    async def approve_repair(
        feature_id: str,
        repair_id: str,
        request_body: ApproveRepairRequest,
        request: Request,
        response: Response,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        credentials: Annotated[RequestScopedCredentials, Depends(resolve_credentials)],
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> FeatureResponse:
        """Authorize one repair, apply it, and give the repository another attempt.

        The acknowledgement is required by the server rather than only by the console. A
        repair changes files or dependencies in somebody's repository, and "the button asked
        first" is not a guarantee when the endpoint is reachable without the button.
        """
        if not request_body.acknowledge_repository_change:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    "Approving a repair changes this repository's checked-in files or "
                    "dependencies and must be acknowledged explicitly."
                ),
            )
        try:
            repairs = await control_plane.repairs(feature_id)
            proposal = next((item for item in repairs if item.repair_id == repair_id), None)
            record = await _execute_durable_mutation(
                request=request,
                response=response,
                control_plane=control_plane,
                actor=actor,
                feature_id=feature_id,
                repository_id=proposal.repository_id if proposal is not None else None,
                action_type="APPROVE_REPOSITORY_REPAIR",
                payload={
                    "repair_id": repair_id,
                    "repository_id": proposal.repository_id if proposal is not None else None,
                    "acknowledge_repository_change": True,
                },
                run=lambda: control_plane.approve_repair(
                    feature_id,
                    repair_id=repair_id,
                    actor_id=actor.actor_id,
                    credentials=credentials,
                ),
            )
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        except (
            WorkflowConflictError,
            FeatureWorkflowError,
            ActionConflictError,
            ActionInProgressError,
        ) as error:
            raise _conflict(error) from error
        return FeatureResponse.model_validate(
            await feature_response_values(record, request=request)
        )

    @router.post("/{feature_id}/repairs/{repair_id}/reject", response_model=FeatureResponse)
    async def reject_repair(
        feature_id: str,
        repair_id: str,
        request_body: RejectRepairRequest,
        request: Request,
        response: Response,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> FeatureResponse:
        """Decline one repair, recording who declined it and why."""
        require(actor, Permission.REPAIR_REJECT)
        try:
            repairs = await control_plane.repairs(feature_id)
            proposal = next((item for item in repairs if item.repair_id == repair_id), None)
            record = await _execute_durable_mutation(
                request=request,
                response=response,
                control_plane=control_plane,
                actor=actor,
                feature_id=feature_id,
                repository_id=proposal.repository_id if proposal is not None else None,
                action_type="REJECT_REPOSITORY_REPAIR",
                payload={
                    "repair_id": repair_id,
                    "repository_id": proposal.repository_id if proposal is not None else None,
                    "reason": request_body.reason,
                },
                run=lambda: control_plane.reject_repair(
                    feature_id,
                    repair_id=repair_id,
                    actor_id=actor.actor_id,
                    reason=request_body.reason,
                ),
            )
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        except (
            WorkflowConflictError,
            FeatureWorkflowError,
            ActionConflictError,
            ActionInProgressError,
        ) as error:
            raise _conflict(error) from error
        return FeatureResponse.model_validate(
            await feature_response_values(record, request=request)
        )

    @router.post("/{feature_id}/chat/{message_id}/reject", response_model=ChatMessageResponse)
    async def reject_chat_action(
        feature_id: str,
        message_id: int,
        chat: Annotated[FeatureChatService, Depends(get_feature_chat)],
    ) -> ChatMessageResponse:
        """Record that a person declined a proposal, so the transcript shows the decision."""
        try:
            message = await chat.reject(feature_id, message_id)
        except WorkflowNotFoundError as error:
            # This route never used to reach the control plane, so it never had to answer
            # for a feature that is not there. It does now: the workspace check is the first
            # thing `reject` does, and the refusal must be the ordinary 404 rather than an
            # uncaught error -- which would have made "not yours" a 500 and therefore
            # perfectly distinguishable from "not there".
            raise _not_found(error) from error
        except ChatActionError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        return _chat_message(message)

    @router.get("/{feature_id}/events", response_model=FeatureEventsResponse)
    async def get_feature_events(
        feature_id: str,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        after: Annotated[int | None, Query(ge=0)] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 200,
    ) -> FeatureEventsResponse:
        """Return lifecycle events after a cursor so a client can watch a feature cheaply.

        The timeline endpoint hydrates parent state, which for a completed feature carries
        every artifact it produced. Polling that for liveness would make watching a feature
        more expensive than running it; this reads only the indexed event table.
        """
        try:
            events = await control_plane.events_after(feature_id, after_id=after, limit=limit)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        return FeatureEventsResponse(
            feature_id=feature_id,
            events=[
                FeatureEventResponse(
                    id=item.id,
                    timestamp=item.timestamp,
                    event_type=item.event_type,
                    source=item.source,
                    event=item.event,
                    details=item.details,
                )
                for item in events
            ],
            last_event_id=events[-1].id if events else after,
        )

    @router.post("/{feature_id}/chat/stream")
    async def stream_chat(
        feature_id: str,
        request_body: SendChatMessageRequest,
        chat: Annotated[FeatureChatService, Depends(get_feature_chat)],
        credentials: Annotated[RequestScopedCredentials, Depends(resolve_credentials)],
    ) -> StreamingResponse:
        """Answer one question, sending the text as the model writes it.

        Server-sent events over an ordinary authenticated POST. Not `EventSource`: the
        browser's own implementation cannot send an `Authorization` header, and the usual way
        round that -- the token in the query string -- puts a credential in somewhere that
        gets logged, cached and shared. A client reading this with `fetch` sends headers
        exactly as it does for every other call.
        """
        return StreamingResponse(
            _chat_stream(chat, feature_id, request_body.message, credentials),
            media_type="text/event-stream",
            headers=_STREAM_HEADERS,
        )

    @router.get("/{feature_id}/events/stream")
    async def stream_feature_events(
        feature_id: str,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        after: Annotated[int | None, Query(ge=0)] = None,
        last_event_id: Annotated[int | None, Header(alias="Last-Event-ID")] = None,
    ) -> StreamingResponse:
        """Send lifecycle events as they are recorded, resuming from where a client left off.

        This reads the same indexed event table the polling endpoint does, so nothing becomes
        true here that is not true there -- the stream says *when* to look, and the REST
        endpoints remain what a client believes. Checkpoints write events during a run rather
        than only at the end of one, so this is genuinely live rather than a slower way of
        finding out the same thing at the same time.

        Every event carries its own id, so a reconnect resumes from a cursor and a duplicate
        is harmless: a client that has seen an id already discards it.
        """
        try:
            await control_plane.get_record(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        return StreamingResponse(
            _event_stream(control_plane, feature_id, after if after is not None else last_event_id),
            media_type="text/event-stream",
            headers=_STREAM_HEADERS,
        )

    @router.get("/{feature_id}/timeline", response_model=FeatureTimelineResponse)
    async def get_feature_timeline(
        feature_id: str,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
    ) -> FeatureTimelineResponse:
        """Read chronological feature lifecycle and artifact events."""
        try:
            timeline = await control_plane.timeline(feature_id)
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        return FeatureTimelineResponse(
            feature_id=feature_id,
            events=[
                FeatureTimelineEventResponse.model_validate(
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

    @router.get("/{feature_id}/logbook", response_model=LogbookResponse)
    async def get_feature_logbook(
        feature_id: str,
        request: Request,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        after: Annotated[int | None, Query(ge=0)] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 200,
    ) -> LogbookResponse:
        """Tell this feature's run as a conversation, composed from its durable records.

        Read-only and derived: nothing is stored, so a feature that finished long before this
        endpoint existed renders its whole story the first time it is asked. Every entry names
        the record it was read from, and the composition is deterministic -- the same rows
        always produce the same thread, because there is no model on this path and nothing
        here that could put one there.

        Served from three reads a person had to do by hand three separate times during the
        185-194 cycle: the lifecycle events, the artifacts each agent wrote, and the operation
        journal underneath them. It costs what the timeline endpoint costs, so it is opened
        rather than polled, and it is paginated by entry so a run with two hundred journal
        rows does not arrive as one response.
        """
        try:
            record = await control_plane.get_record(feature_id)
            events = await control_plane.events_after(
                feature_id, after_id=None, limit=_LOGBOOK_EVENT_BOUND
            )
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        journal = getattr(request.app.state, "operation_journal", None)
        operations = (
            await journal.list_operations_for_feature(feature_id, limit=_LOGBOOK_OPERATION_BOUND)
            if journal is not None
            else []
        )
        entries = feature_logbook(
            record.state,
            lifecycle_events=[
                LogbookEvent(
                    id=item.id,
                    timestamp=item.timestamp,
                    event=item.event,
                    details=item.details,
                )
                for item in events
            ],
            operations=operations,
        )
        page = [item for item in entries if after is None or item.sequence > after][:limit]
        return LogbookResponse(
            feature_id=feature_id,
            entries=[
                LogbookEntryResponse(
                    sequence=item.sequence,
                    emission=item.emission,
                    timestamp=item.timestamp,
                    agent=str(item.agent),
                    tone=cast(Any, str(item.tone)),
                    template=item.template,
                    text=item.text,
                    detail=item.detail,
                    quote=item.quote,
                    quote_source=item.quote_source,
                    record=LogbookRecordResponse(
                        kind=cast(Any, str(item.record.kind)),
                        id=item.record.id,
                        repository_id=item.record.repository_id,
                    ),
                    repository_id=item.repository_id,
                )
                for item in page
            ],
            next_cursor=(
                page[-1].sequence if page and page[-1].sequence < entries[-1].sequence else None
            ),
            agents=list(logbook_agents()),
        )

    @router.post(
        "/{feature_id}/contract-change-requests/{request_id}/approve",
        response_model=FeatureResponse,
    )
    async def approve_contract_change(
        feature_id: str,
        request_id: str,
        request_body: ApproveContractChangeRequest,
        request: Request,
        response: Response,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        credentials: Annotated[RequestScopedCredentials, Depends(resolve_credentials)],
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> FeatureResponse:
        """Apply a pre-commit revision globally or require fresh branches after any commit."""
        require(actor, Permission.CONTRACT_APPROVE)
        try:
            record = await _execute_durable_mutation(
                request=request,
                response=response,
                control_plane=control_plane,
                actor=actor,
                feature_id=feature_id,
                action_type="APPROVE_CONTRACT_CHANGE",
                payload={
                    "request_id": request_id,
                    "resolution": request_body.resolution,
                    "updated_contract": request_body.updated_contract.model_dump(mode="json"),
                },
                run=lambda: control_plane.approve_contract_change(
                    feature_id,
                    request_id=request_id,
                    revision=request_body.updated_contract,
                    resolution=request_body.resolution,
                    credentials=credentials,
                ),
            )
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        except (
            WorkflowConflictError,
            FeatureWorkflowError,
            ActionConflictError,
            ActionInProgressError,
        ) as error:
            raise _conflict(error) from error
        return FeatureResponse.model_validate(
            await feature_response_values(record, request=request)
        )

    @router.post(
        "/{feature_id}/contract-change-requests/{request_id}/reject",
        response_model=FeatureResponse,
    )
    async def reject_contract_change(
        feature_id: str,
        request_id: str,
        request_body: RejectContractChangeRequest,
        request: Request,
        response: Response,
        control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
        credentials: Annotated[RequestScopedCredentials, Depends(resolve_credentials)],
        actor: Annotated[Actor, Depends(current_actor)],
    ) -> FeatureResponse:
        """Record a human rejection rather than silently mutating contract behavior."""
        require(actor, Permission.CONTRACT_REJECT)
        try:
            record = await _execute_durable_mutation(
                request=request,
                response=response,
                control_plane=control_plane,
                actor=actor,
                feature_id=feature_id,
                action_type="REJECT_CONTRACT_CHANGE",
                payload={"request_id": request_id, "resolution": request_body.resolution},
                run=lambda: control_plane.reject_contract_change(
                    feature_id,
                    request_id=request_id,
                    resolution=request_body.resolution,
                    credentials=credentials,
                ),
            )
        except WorkflowNotFoundError as error:
            raise _not_found(error) from error
        except (
            WorkflowConflictError,
            FeatureWorkflowError,
            ActionConflictError,
            ActionInProgressError,
        ) as error:
            raise _conflict(error) from error
        return FeatureResponse.model_validate(
            await feature_response_values(record, request=request)
        )

    return router


async def _resolve_model_setup(
    request: Request, actor: Actor, request_body: StartFeatureRequest
) -> dict[str, Any] | None:
    """Resolve, snapshot and validate the caller's chosen setup, before anything is queued.

    In that order (spec 59- §4.3): the snapshot is taken first because it is what the feature
    pins; validation runs against it with the deployment's *current* declarations, because a
    setup that was valid when saved can be invalid now, and the refusal must be this request's
    answer rather than a failed feature later. An id the caller does not own answers 404, the
    same answer "does not exist" gets, so the route does not enumerate other people's setups.
    """
    if request_body.model_setup_id is None:
        return None
    directory = getattr(request.app.state, "model_setups", None)
    if directory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This deployment does not persist model setups.",
        )
    saved = await directory.get(request_body.model_setup_id, owner_id=actor.actor_id)
    if saved is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"model setup not found: {request_body.model_setup_id}",
        )
    snapshot = {"setup_id": saved.setup_id, "name": saved.name, "roles": dict(saved.roles)}
    unsupported, ceilings = deployment_model_declarations(request)
    try:
        validate_model_setup(
            parse_model_setup_roles(snapshot["roles"]),
            unsupported=unsupported,
            ceilings=ceilings,
        )
    except ModelConfigurationError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error
    return snapshot


def _refuse_a_model_that_cannot_read_images(
    request: Request, request_body: StartFeatureRequest, model_setup: dict[str, Any] | None
) -> None:
    """Refuse a submission with images whose reasoning model is not declared able to read them.

    Fail-closed, and that direction is the whole point. An undeclared model is treated as not
    vision-capable, which is the opposite of the two `*_UNSUPPORTED` declarations: guessing
    "probably fine" there costs a request the provider answers, and guessing it here costs
    either a provider 400 in the middle of a run or -- worse -- a model that accepts the
    image blocks, ignores them, and answers from the prose, which looks exactly like a model
    that read them.

    Two sitings, because the platform resolves models in two different places:

    - A **custom setup** already has a resolved selection at this point: the snapshot was
      taken and validated a few lines above, so the reasoning role's model is simply read
      out of it.
    - A **(platform, tier) pairing** has no acceptance-time resolution today; concrete models
      are resolved on the worker. So one is done here, off `request.app.state.settings` --
      the same object the declaration helpers already read.

    A deployment with no settings at all -- an isolated application -- resolves no model and
    is not asked the question. Nothing there will make a provider call either, so refusing
    would be refusing on behalf of a run that never happens; the call-site fence still
    guards any client that is actually built.
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return
    declared = settings.declared_vision_capable()
    if model_setup is not None:
        roles = model_setup.get("roles", {})
        reasoning = roles.get(ModelRole.REASONING.value, {})
        model = str(reasoning.get("model") or "").strip()
        selection = f"the model setup {model_setup.get('name') or model_setup.get('setup_id')!r}"
        remedy = "edit the setup's reasoning role, or choose a platform and tier that reads them"
    else:
        platform = AgentPlatform(request_body.agent_platform)
        tier = PerformanceTier(request_body.performance_tier)
        try:
            # The clone is already tier-scoped: `for_performance_tier` returns a view whose
            # `ModelConfigService` holds the tier's own resolutions under the default key.
            # Passing `tier=` again would ask that view for a tier it does not hold and
            # refuse every non-high submission -- every other tier-clone reader omits it.
            tier_settings = settings.for_performance_tier(tier)
            model = tier_settings.model_configs.get_model_config(
                ModelRole.REASONING, platform=platform
            ).model
        except ModelConfigurationError as error:
            # The pairing resolves nothing on this deployment. Refused in the predicate's own
            # sentence, which is what the setup path already does -- not silently accepted
            # into a run whose first model call fails on a worker.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
            ) from error
        selection = f"{platform.value} at the {tier.value} tier"
        remedy = "choose a platform, tier or setup whose reasoning model reads them"
    if model and model in declared:
        return
    named = model or "no model"
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=(
            f"this submission attaches images, and {selection} reads the product "
            f"requirements on {named}, which this deployment has not declared able to read "
            f"an image. Remove the images, or {remedy}."
        ),
    )


async def _resolve_attachments(
    request: Request, actor: Actor, request_body: StartFeatureRequest
) -> list[PRDAttachment]:
    """Resolve this submission's image references into records, or refuse the submission.

    The submission carried ids, markers and captions. What the artifact records, and what the
    product manager is told it is looking at, is the store's own answer: the filename, the
    sniffed type, the size and the hash. So every reference is read here -- and every way a
    reference can be wrong is a 422 naming which one, because "one of your images is
    unavailable" is not something anybody can act on.

    The order the records come back in is the order the model is shown the images: the ones
    the prose references, in the order the prose references them, then everything else in
    declaration order. An attachment nothing references still reaches the model -- somebody
    who attached three screenshots without writing `[image:...]` three times has still shown
    the platform three screenshots -- but the ones a sentence points at come first, so the
    narrative and the images run in the same direction.
    """
    references = list(request_body.prd.attachments)
    if not references:
        return []
    store = getattr(request.app.state, "attachments", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This deployment does not persist attachments.",
        )
    if len(references) > MAX_ATTACHMENTS_PER_FEATURE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"a feature may carry at most {MAX_ATTACHMENTS_PER_FEATURE} images; "
                f"this submission names {len(references)}"
            ),
        )
    by_marker: dict[str, PRDAttachment] = {}
    total_bytes = 0
    for item in references:
        record = await store.get_metadata(item.attachment_id)
        # "Not yours" and "does not exist" are the same answer, for the reason the fetch
        # endpoint gives: an id is not an authorisation, and distinguishing the two turns
        # this into an oracle for other people's uploads.
        if record is None or record.owner_id != actor.actor_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"the image referenced as [image:{item.marker}] does not exist or is "
                    "not yours to submit"
                ),
            )
        if record.feature_id is not None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"the image referenced as [image:{item.marker}] has already been "
                    "submitted with another feature; upload it again to attach it to this one"
                ),
            )
        if not record.content_present:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    f"the image referenced as [image:{item.marker}] no longer has its "
                    "contents; upload it again"
                ),
            )
        total_bytes += record.byte_size
        by_marker[item.marker] = PRDAttachment(
            attachment_id=record.attachment_id,
            marker=item.marker,
            caption=item.caption,
            filename=record.filename,
            media_type=cast(Literal["image/png", "image/jpeg", "image/webp"], record.media_type),
            byte_size=record.byte_size,
            sha256=record.sha256,
        )
    if total_bytes > MAX_ATTACHMENT_BYTES_PER_FEATURE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"a feature's images may total at most "
                f"{MAX_ATTACHMENT_BYTES_PER_FEATURE // (1024 * 1024)} MiB; "
                f"these total {total_bytes // 1024} KiB"
            ),
        )
    referenced = markers_in_all(request_body.prd.prose())
    ordered = [by_marker[marker] for marker in referenced if marker in by_marker]
    seen = {item.marker for item in ordered}
    ordered.extend(item for item in by_marker.values() if item.marker not in seen)
    return ordered


async def _missing_setup_providers(
    request: Request, actor: Actor, model_setup: dict[str, Any]
) -> list[str]:
    """Return what a setup-pinned live feature needs and this identity lacks, naming roles.

    Every platform the setup's roles name, plus GitHub. The role names travel in the answer
    because a platform name alone sends somebody looking for which of four rows caused it:
    "coding is pinned to OpenAI and no OpenAI credential is stored for you".
    """
    roles = model_setup.get("roles", {})
    platform_roles: dict[str, list[str]] = {}
    for role_name, entry in roles.items():
        platform = str(entry.get("platform") or "")
        if platform:
            platform_roles.setdefault(platform, []).append(role_name)
    missing = await missing_stored_providers_for_platforms(
        get_secret_store(request), actor.actor_id, platforms=tuple(platform_roles)
    )
    labels = {label: provider for provider, label in CREDENTIAL_PROVIDERS}
    named: list[str] = []
    for label in missing:
        provider = labels.get(label, label)
        pinned = platform_roles.get(provider)
        if pinned:
            verb = "is" if len(pinned) == 1 else "are"
            named.append(f"{label} ({', '.join(pinned)} {verb} pinned to it by the setup)")
        else:
            named.append(label)
    return named


async def _missing_stored_providers(
    request: Request, actor: Actor, *, agent_platform: str
) -> list[str]:
    """Return the provider credentials this feature would need and this identity lacks.

    Scoped to the feature's own platform: an `anthropic` feature needs `anthropic` and
    `github`, an `openai` feature needs `openai` and `github`, and neither needs the other's
    key.
    """
    return await missing_stored_providers(
        get_secret_store(request), actor.actor_id, agent_platform=agent_platform
    )


async def _provider_refused_credentials(request: Request, actor: Actor) -> list[str]:
    """Return the display names of providers that answered "no" about this identity's key.

    Scoped to the repository provider, which is the one every live feature uses on its first
    action and the one whose refusal killed runs 190 and 191. A model provider's key is
    verified by the call that needs it, minutes later on a worker, where a refusal is already
    a diagnosable provider answer rather than an anonymous fault -- so paying a network round
    trip for it here would buy a diagnosis that already exists.
    """
    store = get_secret_store(request)
    if store is None:
        return []
    provider, label = REPOSITORY_PROVIDER
    secret = await stored_secret(store, actor.actor_id, provider)
    if secret is None:
        return []
    verdict = await verify_stored_credential(request, provider=provider, secret=secret)
    return [label] if verdict is CredentialVerdict.REFUSED else []


async def _unresolvable_design_reference(
    request: Request, request_body: StartFeatureRequest
) -> str | None:
    """Say why this submission's design citations could never be resolved, or nothing.

    Every check here is a question about the deployment's own configuration and the request's
    own text. Nothing opens a file: a citation is resolved exactly once, later, in the
    pre-coding step that writes the snapshot, and doing it here would resolve it twice.

    Refused at acceptance rather than at resolution, because a refusal before any effect costs
    only the person's correction -- no repository is cloned, no branch exists, nothing is
    queued. Accepting a citation nothing will ever resolve is the control that lies: the
    console would show a design attached to a feature that was planned, built and reviewed
    against prose alone.

    Returns the sentence to refuse with, so it slots in beside the credential checks at the
    one acceptance refusal site rather than inventing a second one.
    """
    references = request_body.prd.design_references
    if not references:
        # Safety rule 3: a feature that cites no design reaches none of this, and the
        # deployment is not asked whether it has a design source at all.
        return None
    directory = getattr(request.app.state, "design_source_configuration", None)
    configuration = None if directory is None else await directory.get()
    if configuration is None or not configuration.enabled:
        return (
            "This feature cites a design, and this deployment has no enabled design source "
            "to resolve it with. Configure one in Settings under Design source -- or remove "
            "the design links and submit the requirement on its own."
        )
    if configuration.status == "degraded":
        return (
            "This feature cites a design, and the design source is degraded: "
            f"{configuration.status_reason or 'the stored Figma token was refused'} "
            "Re-save the design source in Settings once the token is replaced."
        )
    outside = [
        reference.file_key
        for reference in references
        if not configuration.permits(reference.file_key)
    ]
    if outside:
        return (
            f"This deployment's design source allows citations against "
            f"{len(configuration.file_allowlist)} file(s), and "
            f"{', '.join(dict.fromkeys(outside))} is not one of them. Add the file key to the "
            "allowlist in Settings under Design source, or cite a permitted file."
        )
    known = {item.repository_id for item in request_body.repositories}
    unknown = [
        repository_id
        for reference in references
        for repository_id in reference.applies_to
        if repository_id not in known
    ]
    if unknown:
        # A citation scoped to a repository this feature does not include applies to nothing,
        # and would silently reach no workstream at all. Said rather than dropped.
        return (
            f"A design citation is scoped to {', '.join(dict.fromkeys(unknown))}, which this "
            "feature does not include. Scope it to one of this feature's repositories, or "
            "leave it unscoped and let the plan decide."
        )
    if MAX_DESIGN_NODES < 1:
        # The only configuration under which a whole-file citation cannot resolve to anything
        # at all. Stated as a refusal naming the bound rather than accepted into a resolution
        # that would report every frame omitted.
        whole_file = [item.file_key for item in references if item.cites_whole_file]
        if whole_file:
            return (
                f"A citation naming no frame resolves to a file's top-level frames, and this "
                f"deployment's node bound is {MAX_DESIGN_NODES}, so it would resolve to "
                "nothing. Cite a frame directly by opening it in Figma and copying its link."
            )
    return None


def get_feature_chat(
    request: Request,
    control_plane: Annotated[FeatureControlPlane, Depends(get_feature_control_plane)],
) -> FeatureChatService:
    """Return this request's view of the chat service, reading only its own workspace.

    The service is built once at startup and therefore holds the deployment-wide control
    plane. Rebinding it to this request's scoped one is what makes the chat routes obey
    workspace isolation -- including `confirm`, which reaches the same control-plane methods
    the buttons do, and which has bypassed a guard before.
    """
    chat = getattr(request.app.state, "feature_chat", None)
    if chat is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The feature assistant is not configured for this deployment.",
        )
    return cast(FeatureChatService, chat).with_control_plane(control_plane)


# Proxies and browsers will happily buffer or reuse a streamed response, which turns a live
# stream into a page that arrives all at once when it ends. `X-Accel-Buffering` is nginx's
# opt-out and is ignored by everything else.
_STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

# How often the event stream looks for new events, and how often it says something when there
# are none. The comment is worth more than the numbers: this polls an indexed table rather
# than holding a listener open, because a listener per viewer would hold a database
# connection to buy latency this screen does not need.
_EVENT_POLL_SECONDS = 1.0
_EVENT_KEEPALIVE_SECONDS = 20.0


def _sse(event: str, data: dict[str, Any], *, event_id: int | None = None) -> str:
    """Format one server-sent event.

    The id matters more than it looks: it is what a reconnecting client resumes from, and
    what lets a duplicate be recognised and dropped rather than applied twice.
    """
    lines = [] if event_id is None else [f"id: {event_id}"]
    lines.append(f"event: {event}")
    # `json.dumps` never emits a raw newline, so the payload cannot break the framing.
    lines.append(f"data: {json.dumps(data, default=str)}")
    return "\n".join(lines) + "\n\n"


async def _chat_stream(
    chat: FeatureChatService,
    feature_id: str,
    message: str,
    credentials: RequestScopedCredentials,
) -> AsyncIterator[str]:
    """Forward one streamed answer, and say plainly when it could not be produced."""
    try:
        async for event in chat.send_stream(feature_id, message, credentials=credentials):
            if event.kind == "delta":
                yield _sse("delta", {"text": event.text})
            elif event.message is not None:
                payload = _chat_message(event.message).model_dump(mode="json")
                if event.kind == "user":
                    yield _sse("user", payload)
                else:
                    yield _sse("message", {**payload, "detail": event.detail})
    except LLMAdapterError as error:
        # No usable provider key. A fact about the request, not a fault, and the transcript
        # already holds the question so nothing is lost by saying so here.
        yield _sse("error", {"detail": str(error)})
    except WorkflowNotFoundError as error:
        yield _sse("error", {"detail": str(error)})
    except asyncio.CancelledError:
        # The reader went away. The service has already persisted whatever arrived.
        raise
    except Exception:
        # The response has already begun, so there is no status code left to change. Say
        # something the client can render rather than dropping the connection silently.
        yield _sse("error", {"detail": "The assistant did not finish answering."})
    yield _sse("done", {})


async def _event_stream(
    control_plane: FeatureControlPlane,
    feature_id: str,
    after: int | None,
) -> AsyncIterator[str]:
    """Send lifecycle events from a cursor onwards, then keep the connection honest."""
    cursor = after
    # Sent immediately so a client knows it is connected before anything has happened, which
    # is most of the time: a feature waiting on a person emits nothing for hours.
    yield _sse("open", {"feature_id": feature_id, "after": cursor})
    silent_for = 0.0
    while True:
        try:
            events = await control_plane.events_after(feature_id, after_id=cursor, limit=200)
        except WorkflowNotFoundError:
            yield _sse("error", {"detail": f"feature not found: {feature_id}"})
            return
        for item in events:
            cursor = item.id
            yield _sse(
                "event",
                {
                    "id": item.id,
                    "timestamp": item.timestamp,
                    "event_type": item.event_type,
                    "source": item.source,
                    "event": item.event,
                    "details": item.details,
                },
                event_id=item.id,
            )
        silent_for = 0.0 if events else silent_for + _EVENT_POLL_SECONDS
        if silent_for >= _EVENT_KEEPALIVE_SECONDS:
            # A comment frame. Proxies drop a connection that has said nothing for long
            # enough, and a client cannot tell that from a feature that is simply quiet.
            yield ": keep-alive\n\n"
            silent_for = 0.0
        await asyncio.sleep(_EVENT_POLL_SECONDS)


def _granted_by(actor: Actor, supplied: str) -> str:
    """Return who to record, preferring the identity that authenticated.

    A named person is recorded as themselves. The shared administrative key cannot say who
    was holding it, so what the caller typed is kept as the only available attribution --
    labelled, so an audit is not misled into thinking the platform verified it.
    """
    if not actor.is_platform_key:
        return actor.actor_id
    return f"{supplied} (via platform key)" if supplied.strip() else actor.actor_id


def _action_response(action: FeatureAction) -> FeatureActionResponse:
    """Project a durable action onto its public shape, without its lease owner."""
    return FeatureActionResponse(
        action_id=action.action_id,
        feature_id=action.feature_id,
        repository_id=action.repository_id,
        action_type=action.action_type,
        actor_id=action.actor_id,
        actor_display_name=action.actor_display_name,
        origin=action.origin,
        origin_message_id=action.origin_message_id,
        status=action.status,
        attempt=action.attempt,
        max_attempts=action.max_attempts,
        created_at=action.created_at,
        started_at=action.started_at,
        completed_at=action.completed_at,
        # Compared against server time. A client cannot decide this: its clock is not the
        # platform's, and "is somebody still working on this" must not depend on the browser.
        in_progress=action.lease_is_live(now=datetime.now(UTC)),
        result_summary=action.result_summary,
        error_code=action.error_code,
        error_message=action.error_message,
        external_operation_ids=list(action.external_operation_ids),
        reconciled_by=action.reconciled_by,
        reconciliation_reason=action.reconciliation_reason,
        reconciled_at=action.reconciled_at,
    )


def _design_conflict_response(
    state: FeatureWorkflowSnapshot, conflict: DesignConflictArtifact
) -> DesignConflictResponse:
    """Project one open design question, saying whether the platform could act on a verdict.

    ``answerable`` and ``attempts_remaining`` are asked of the workflow's own precondition
    rather than worked out here, for the reason `_workstream_actions` asks it: a control whose
    only outcome is a refusal is not a control, and a browser rule that agrees with the server
    today is a browser rule that disagrees with it after the next change.
    """
    child = state.child_workflows.get(conflict.repository_id)
    try:
        require_answerable_design_conflict(state, conflict=conflict, additional_attempts=0)
    except FeatureWorkflowError:
        # Zero attempts is the ordinary request, so a refusal here is either about the
        # repository or about a budget a grant would have to cover. Asked again with one, so a
        # question that only needs a purchase is still offered as answerable.
        try:
            require_answerable_design_conflict(state, conflict=conflict, additional_attempts=1)
            answerable = True
        except FeatureWorkflowError:
            answerable = False
    else:
        answerable = True
    return DesignConflictResponse(
        conflict_id=conflict.conflict_id,
        repository_id=conflict.repository_id,
        kind=conflict.kind,
        question=conflict.question,
        evidence=list(conflict.evidence),
        demanded=DesignConflictPositionResponse(
            authority=conflict.demanded_by,
            statement=conflict.demand,
        ),
        # A recurring demand has no satisfied position; the schema validator guarantees a
        # reversal carries both fields.
        satisfied=(
            DesignConflictPositionResponse(
                authority=conflict.satisfied_under,
                statement=conflict.satisfied_demand,
                grounds=conflict.grounds,
            )
            if conflict.satisfied_under is not None and conflict.satisfied_demand is not None
            else None
        ),
        cross_authority=conflict.cross_authority,
        removals=conflict.removals,
        attempts_spent=conflict.attempts_spent,
        attempts_remaining=(
            design_conflict_attempts_remaining(state, child) if child is not None else 0
        ),
        answerable=answerable,
    )


def _repair_response(
    repair: RepositoryRepairProposalArtifact, *, current_revision: str | None
) -> RepositoryRepairResponse:
    """Project a repair proposal, saying whether the repository has moved since."""
    return RepositoryRepairResponse(
        repair_id=repair.repair_id,
        feature_id=repair.feature_id,
        repository_id=repair.repository_id,
        originating_stage=repair.originating_stage,
        failure_classification=repair.failure_classification,
        detected_problem=repair.detected_problem,
        evidence=list(repair.evidence),
        proposed_repair=repair.proposed_repair,
        affected_files=list(repair.affected_files),
        affected_dependencies=list(repair.affected_dependencies),
        commands=[
            RepositoryRepairCommandResponse(
                command=list(item.command),
                working_directory=item.working_directory,
                purpose=item.purpose,
            )
            for item in repair.commands
        ],
        expected_impact=repair.expected_impact,
        risk=repair.risk,
        changes_source_logic=repair.changes_source_logic,
        proposed_at_revision=repair.proposed_at_revision,
        status=RepositoryRepairStatus(repair.status),
        approved_by=repair.approved_by,
        approved_at=repair.approved_at,
        rejected_by=repair.rejected_by,
        rejection_reason=repair.rejection_reason,
        execution_result=repair.execution_result,
        resulting_revision=repair.resulting_revision,
        stale=repair_is_stale(repair, current_revision=current_revision),
        created_at=repair.timestamp,
    )


def _chat_message(message: ChatMessage) -> ChatMessageResponse:
    """Project a stored message onto its public shape."""
    action = message.proposed_action
    return ChatMessageResponse(
        id=message.id or 0,
        role=message.role,
        content=message.content,
        proposed_action=(
            ProposedActionResponse(
                type=str(action.get("type", "")),
                arguments=dict(action.get("arguments") or {}),
                summary=str(action.get("summary", "")),
            )
            if action
            else None
        ),
        action_status=message.action_status,
        action_result=message.action_result,
        action_id=message.action_id,
        created_at=message.created_at,
    )


def get_feature_control_plane(
    request: Request, scope: Annotated[WorkspaceScope, Depends(current_scope)]
) -> FeatureControlPlane:
    """Return this request's view of the feature control plane, narrowed to its workspace.

    Every feature route resolves this dependency, and it never hands back the unscoped store.
    That is the whole isolation boundary: there are more than twenty read routes and a dozen
    mutating ones keyed by `{feature_id}`, plus the chat action path and the SSE stream, and a
    per-route ownership check would be thirty-odd places to forget one. A route added by
    somebody who never read this cannot obtain an unscoped read, because this function does
    not produce one.
    """
    return cast(
        FeatureControlPlane,
        ScopedFeatureControlPlane(
            cast(FeatureControlPlane, request.app.state.feature_control_plane), scope
        ),
    )


async def _execute_durable_mutation(
    *,
    request: Request,
    response: Response,
    control_plane: FeatureControlPlane,
    actor: Actor,
    feature_id: str,
    action_type: str,
    payload: dict[str, Any],
    run: Callable[[], Awaitable[FeatureRecord]],
    repository_id: str | None = None,
) -> FeatureRecord:
    """Execute every REST mutation through the same durable lease used by chat.

    The response points at the action record so a client can recover after losing the HTTP
    response. Replays read current workflow state after the action service proves the prior
    intent succeeded; they never call the domain operation twice.
    """
    actions = cast(FeatureActionService, request.app.state.feature_actions)
    before = await control_plane.get_record(feature_id)
    action, _created = await actions.submit(
        feature_id=feature_id,
        repository_id=repository_id,
        action_type=action_type,
        actor_id=actor.actor_id,
        actor_display_name=actor.display_name,
        origin="rest",
        payload=payload,
        context_version=action_context_version(before.state, action_type, payload),
        request_idempotency_key=request.headers.get("idempotency-key"),
    )
    response.headers["X-Feature-Action-ID"] = action.action_id
    result: FeatureRecord | None = None

    async def invoke() -> str:
        nonlocal result
        result = await run()
        return f"{action_type} committed; feature is {result.state.status.value}."

    executed = await actions.execute(action, invoke)
    if executed.status is not FeatureActionStatus.SUCCEEDED:
        msg = executed.error_message or f"action is {executed.status.value}"
        raise ActionConflictError(msg)
    return result if result is not None else await control_plane.get_record(feature_id)


async def feature_response_values(record: FeatureRecord, *, request: Request) -> dict[str, Any]:
    """Compose the durable checkpoint and active queue hand-off into one public read model."""
    state = record.state
    active_intent = await _active_feature_intent(request, state.feature_id)
    effective_status = state.status.value
    if active_intent == RESUME:
        effective_status = RESUMING_EFFECTIVE_STATUS
    elif active_intent == RETRY_WORKSTREAM:
        effective_status = RETRYING_EFFECTIVE_STATUS
    elif active_intent == ANSWER_DESIGN_VERDICT:
        effective_status = DECIDING_EFFECTIVE_STATUS
    elif active_intent == PUBLISH:
        # Without this, pressing publish leaves the feature reading `failed_requires_human`
        # for as long as the queue takes to drain, which is indistinguishable from the press
        # having done nothing.
        effective_status = PUBLISHING_EFFECTIVE_STATUS
    elif active_intent == REVISE:
        # For the publish case's reason: a person who just asked for changes has to see that
        # the request landed, and `planning` alone reads like the platform's own idea.
        effective_status = REVISING_EFFECTIVE_STATUS
    return {
        "feature_id": state.feature_id,
        "workflow_id": state.workflow_id,
        "status": state.status,
        "effective_status": effective_status,
        "title": state.title,
        "reference": state.reference,
        "revision": state.revision,
        "current_agent": state.current_agent,
        "repository_count": len(state.repository_specs),
        "required_repository_count": sum(item.required for item in state.repository_specs),
        "repositories": [
            {
                "repository_id": item.repository_id,
                "name": item.name,
                "role": item.role,
                "repository_url": str(item.repository_url),
                "default_branch": item.default_branch,
                "required": item.required,
                "implementation_order": item.implementation_order,
            }
            for item in state.repository_specs
        ],
        "clarification_rounds": state.clarification_rounds,
        "integration_review_cycles": state.integration_review_cycles,
        "max_clarification_rounds": state.max_clarification_rounds,
        "max_integration_review_cycles": state.max_integration_review_cycles,
        "max_child_review_cycles": state.max_child_review_cycles,
        "max_implementation_retries": state.max_implementation_retries,
        "max_validation_retries": state.max_validation_retries,
        "max_repository_setup_retries": state.max_repository_setup_retries,
        "merge_strategy": state.merge_strategy,
        "deployment_strategy": state.deployment_strategy,
        "execution_mode": state.execution_mode,
        "agent_platform": state.agent_platform,
        "performance_tier": state.performance_tier,
        "model_setup": await _pinned_model_setup(request, state),
        "cancellation_status": state.cancellation_status,
        "cancellation_requested_at": state.cancellation_requested_at,
        "cancellation_reason": state.cancellation_reason,
        "cleanup_requirements": state.cleanup_requirements,
        "failure_summary": state.failure_summary,
        "planning_wall_seconds": state.planning_wall_seconds,
        "planning_provider_fault_seconds": state.planning_provider_fault_seconds,
        "available_actions": _feature_actions(record, active_intent=active_intent),
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


async def _pinned_model_setup(
    request: Request, state: FeatureWorkflowSnapshot
) -> dict[str, Any] | None:
    """Project a custom feature's pinned setup, in full, with what became of the setup since.

    The roles come from the snapshot, never from the setup row: the snapshot is what this
    feature actually runs (G3), and this is that fact made visible. The row is consulted only
    for the `setup_state` verdict -- unchanged, edited, or deleted -- which is an equality
    against the snapshot and exposes nothing of a row this reader may not own.
    """
    snapshot = state.model_setup_snapshot
    if snapshot is None:
        return None
    raw_roles = snapshot.get("roles")
    roles = (
        [
            {
                "role": role_name,
                "platform": str(entry.get("platform") or ""),
                "model": str(entry.get("model") or ""),
                "reasoning_effort": entry.get("reasoning_effort"),
                "max_tokens": entry.get("max_tokens"),
            }
            for role_name, entry in raw_roles.items()
            if isinstance(entry, dict)
        ]
        if isinstance(raw_roles, dict)
        else []
    )
    setup_id = str(snapshot.get("setup_id") or state.model_setup_id or "")
    setup_state = "deleted"
    directory = getattr(request.app.state, "model_setups", None)
    if directory is not None and setup_id:
        current = await _setup_row_for_comparison(directory, setup_id)
        if current is not None:
            setup_state = "unchanged" if current == raw_roles else "edited"
    return {
        "setup_id": setup_id,
        "name": str(snapshot.get("name") or setup_id),
        "roles": roles,
        "setup_state": setup_state,
    }


async def _setup_row_for_comparison(directory: Any, setup_id: str) -> dict[str, Any] | None:
    """Read one setup's role map for the edited-since verdict, tolerating any store shape."""
    finder = getattr(directory, "find", None)
    if finder is None:
        return None
    found = await finder(setup_id)
    return None if found is None else dict(found.roles)


def _repository_identity(spec: RepositorySpec | None) -> dict[str, Any]:
    """Return what a client needs to render a workstream, or nothing when the spec is gone.

    A workstream can outlive its spec in migrated state, so this stays optional rather than
    failing a read that would otherwise succeed.
    """
    if spec is None:
        return {}
    return {
        "repository_name": spec.name,
        "repository_role": spec.role,
        "repository_url": str(spec.repository_url),
        "repository_required": spec.required,
    }


def _feature_actions(record: FeatureRecord, *, active_intent: str | None = None) -> list[str]:
    """Publish presentation capabilities from backend state, never from browser rules."""
    state = record.state
    terminal = {
        FeatureWorkflowStatus.COMPLETED,
        FeatureWorkflowStatus.CANCELLED,
        FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS,
    }
    if state.status in terminal:
        if (
            state.status is FeatureWorkflowStatus.COMPLETED
            and active_intent is None
            and any(
                child.pull_request_artifact_id is not None
                for child in state.child_workflows.values()
            )
        ):
            # The one action a finished feature still has: asking for changes to the work it
            # published. Gated on a pull request existing because that is what a revision
            # supersedes, and on no active intent because an accepted revision already owns
            # the run lock.
            return ["REVISE_FEATURE"]
        return []
    actions = ["CANCEL_WORKFLOW", "RETIRE_FEATURE"]
    if active_intent is not None:
        # The accepted queue entry is already the feature's next mutation. Advertising a
        # second answer, resume, or retry while it owns the run lock produces only a conflict.
        return actions
    if any(
        _design_conflict_response(state, conflict).answerable
        for conflict in open_design_conflicts(state.artifacts)
    ):
        # Published beside the others rather than instead of them. A feature stopped on a
        # design question can still legitimately be cancelled or retired, and a reader who
        # cannot see that the question is answerable has only those two options.
        actions.append("ANSWER_DESIGN_VERDICT")
    try:
        # A feature-level decision, so it belongs here rather than on a workstream: one press
        # publishes everything eligible. It is what replaces the pull request a partial
        # feature used to open on its own, which makes advertising it load-bearing -- a held
        # feature that does not say it is held is the defect this change exists to remove.
        require_publishable_feature(record.state)
    except FeatureWorkflowError:
        pass
    else:
        actions.append("PUBLISH_FEATURE")
    if state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN and _open_clarification_questions(
        state
    ):
        actions.append("ANSWER_CLARIFICATION")
    elif (
        state.status is FeatureWorkflowStatus.WAITING_FOR_HUMAN
        and state.failure_summary is not None
        and state.failure_summary.retryable
    ):
        actions.append("RESUME_WORKFLOW")
    elif state.status is not FeatureWorkflowStatus.PENDING:
        # A queued feature has not started, so there is nothing to resume -- and offering it
        # would let somebody race the worker that is about to claim it. Cancelling is still
        # offered, because deciding against a feature before it runs is a real thing to want.
        actions.append("RESUME_WORKFLOW")
    return actions


def _open_clarification_questions(state: Any) -> list[Any]:
    """Return only the newest Technical PRD's unresolved human decisions."""
    technical_prd = next(
        (item for item in reversed(state.artifacts) if isinstance(item, TechnicalPRDArtifact)),
        None,
    )
    return technical_prd.unresolved_questions if technical_prd is not None else []


async def _active_feature_intent(request: Request, feature_id: str) -> str | None:
    """Read queue coordination without exposing the queued answers or retry arguments."""
    queue = cast(FeatureExecutionQueue | None, getattr(request.app.state, "feature_queue", None))
    if queue is None:
        return None
    return await queue.active_intent(feature_id)


def _workstream_actions(record: FeatureRecord, *, repository_id: str) -> list[str]:
    """Ask the workflow's own preconditions before advertising either control.

    Two independent questions, so two independent checks. A repository stopped on a design
    conflict is usually retryable as well -- the stop returned its budget unspent -- and
    offering only the grant would send somebody to buy attempts for an argument.
    """
    actions: list[str] = []
    try:
        require_retryable_workstream(
            record.state, repository_id=repository_id, additional_attempts=1
        )
    except FeatureWorkflowError:
        pass
    else:
        actions.append("RETRY_WORKSTREAM")
    if any(
        _design_conflict_response(record.state, conflict).answerable
        for conflict in open_design_conflicts(record.state.artifacts, repository_id=repository_id)
    ):
        actions.append("ANSWER_DESIGN_VERDICT")
    return actions


def _workstream_publication(record: FeatureRecord, *, repository_id: str) -> dict[str, Any]:
    """Say whether one repository would be published, under which class, or why not.

    Asked through the same precondition the action itself runs, for the reason
    `_workstream_actions` asks its own: a client that re-derived this would be a second,
    weaker copy of the rule. The action is advertised on the feature -- one press publishes
    everything eligible -- and this is what lets a reader see which repositories that means
    and which one is being held back by a failing check.
    """
    try:
        publication_class = require_publishable_workstream(
            record.state, repository_id=repository_id
        )
    except FeatureWorkflowError as error:
        return {"publication_class": None, "publication_refusal": error.diagnostics[0]}
    return {"publication_class": publication_class.value, "publication_refusal": None}


async def _feature_is_visible(control_plane: FeatureControlPlane, feature_id: str) -> bool:
    """Return whether one feature is in the caller's workspace, as a boolean.

    For the one place that filters a list rather than answering a single request: a journal
    row naming a feature the caller cannot see is dropped, not turned into a 404, because the
    caller asked for a list and the row is simply not theirs.
    """
    try:
        await control_plane.require_visible(feature_id)
    except WorkflowNotFoundError:
        return False
    return True


def _not_found(error: WorkflowNotFoundError) -> HTTPException:
    """Map missing parent feature state to HTTP 404."""
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error))


# What a client branches on instead of the sentence. Two values, because there are two
# answers: come back, or stop and read the state. Kept narrow on purpose -- a code per
# exception type would be an API surface nobody asked for and would have to be kept true.
_CONFLICT_BUSY = "workflow_busy"
_CONFLICT_SETTLED = "action_settled"


def _conflict(
    error: WorkflowConflictError | FeatureWorkflowError | ActionConflictError,
) -> HTTPException:
    """Map an invalid parent lifecycle transition to HTTP 409.

    `FeatureWorkflowError` is the orchestrator refusing an operation -- resuming a completed
    feature, acting on a contract request that is not pending. It reached the client as a 500
    "Internal server error", which reads as a platform fault and tells the operator nothing.
    A refusal is an answer.

    Two kinds of 409 leave here and they mean opposite things. "This feature is busy; the
    same request will work in a moment" and "this request has reached an outcome; read the
    state before asking again" both arrived as a bare 409 carrying prose, so a client had to
    parse a sentence to tell wait from stop -- and the loop that made AB-Feature-203
    unrecoverable was a well-behaved client doing the reasonable thing with the one that said
    to retry. The distinction now travels in headers a client can branch on, and the body is
    unchanged so nothing that reads `detail` has to move.
    """
    busy = isinstance(error, WorkflowBusyError | ActionInProgressError)
    headers = {"X-Conflict-Reason": _CONFLICT_BUSY if busy else _CONFLICT_SETTLED}
    if busy:
        # Seconds, per RFC 9110. One: the lock is held for the length of one mutation, and a
        # client that waits longer than it needs to is the failure mode this is fixing.
        headers["Retry-After"] = "1"
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error), headers=headers)
