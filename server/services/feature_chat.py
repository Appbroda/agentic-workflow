"""Durable per-feature chat, and execution of the actions proposed inside it.

The assistant never mutates workflow state. It proposes; a person confirms; and the confirmed
action runs through the same control-plane methods the REST routes use, so there is exactly one
implementation of "cancel this feature" and one place that decides whether it is legal.

Confirmation itself is not executed here any more. It becomes a durable action -- an intent
recorded before anything runs, owned by a lease, and reconciled if the executor dies. That
matters because this module previously held the whole story in one column: a crash between
claiming a proposal and recording its outcome left the message reading ``executing`` with
nothing in the platform able to move it again, and no way to tell whether the retry it
authorized had happened.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from agents.assistant.agent import (
    FeatureAssistant,
    ProposedAction,
    validate_action_arguments,
)
from agents.assistant.context import build_context
from agents.shared.contracts import AgentArtifactError
from api.control_plane import RequestScopedCredentials
from api.feature_control_plane import FeatureControlPlane
from api.schemas import ClarificationAnswer
from services.feature_actions import action_context_version
from state.enums import FeatureActionStatus
from storage.action_store import ActionConflictError


class ChatMessageStore(Protocol):
    """Durable transcript for one feature."""

    async def append(self, feature_id: str, message: ChatMessage) -> ChatMessage:
        """Persist one message and return it with its assigned identifier."""

    async def history(self, feature_id: str, *, limit: int) -> list[ChatMessage]:
        """Return the transcript in order, oldest first."""

    async def get(self, feature_id: str, message_id: int) -> ChatMessage | None:
        """Return one message, or nothing when it does not belong to this feature."""

    async def claim_action(self, feature_id: str, message_id: int) -> ChatMessage | None:
        """Atomically move a pending action to executing, or return nothing if already claimed."""

    async def record_action_outcome(
        self, feature_id: str, message_id: int, *, status: str, result: str
    ) -> ChatMessage:
        """Mark a proposal confirmed or rejected, and record what happened."""

    async def attach_action(
        self, feature_id: str, message_id: int, *, action_id: str, status: str
    ) -> ChatMessage:
        """Bind a confirmed proposal to the durable action it became."""

    async def project_action_status(
        self, feature_id: str, message_id: int, *, status: str, result: str | None
    ) -> ChatMessage:
        """Mirror a durable action's state onto the message, whatever the message now says.

        Unconditional on purpose. The guarded outcome write it replaces could only move a
        message the current request had claimed, so a message stranded by a crash could never
        be corrected -- not by a later request, and not by recovery.
        """


@dataclass(slots=True)
class ChatMessage:
    """One turn of the conversation, and any action proposed in it."""

    role: str
    content: str
    id: int | None = None
    proposed_action: dict[str, Any] | None = None
    action_status: str | None = None
    action_result: str | None = None
    action_id: str | None = None
    created_at: datetime | None = None


@dataclass(slots=True)
class ChatStreamEvent:
    """One thing that happened while an answer was being written.

    Typed rather than left as loose dictionaries because the transport serializes these onto
    a wire a browser reads, and "what kinds of event can arrive" is exactly the thing both
    ends have to agree on.
    """

    kind: str
    text: str = ""
    message: ChatMessage | None = None
    detail: str | None = None


class ChatActionError(RuntimeError):
    """Raised when a confirmed action cannot be carried out as proposed."""


class AssistantFactory(Protocol):
    """Build an assistant for one request, on one feature's platform."""

    def __call__(
        self, credentials: RequestScopedCredentials, *, agent_platform: str
    ) -> FeatureAssistant:
        """Return an assistant bound to these credentials, or raise if none are usable."""


class FeatureChatService:
    """Answer a message about one feature, and run a confirmed proposal."""

    def __init__(
        self,
        *,
        assistant_for: AssistantFactory,
        control_plane: FeatureControlPlane,
        store: ChatMessageStore,
        actions: Any = None,
        history_limit: int = 20,
    ) -> None:
        """Bind the assistant factory, the control plane it may ask, and the transcript.

        The assistant is built per request rather than once at startup because this platform
        keeps provider credentials request-scoped: the deployment does not put an API key in
        the process environment, so an assistant constructed at boot has no key to use and a
        deployment that tried was a deployment that could not chat at all.

        ``actions`` is the durable action service. It is optional so an application assembled
        without one still answers questions; confirming a proposal without it is refused
        rather than executed unprotected, because an execution nothing recorded is the exact
        failure this service was changed to remove.
        """
        self._assistant_for = assistant_for
        self._control_plane = control_plane
        self._store = store
        self._actions = actions
        self._history_limit = history_limit

    def with_control_plane(self, control_plane: FeatureControlPlane) -> FeatureChatService:
        """Return the same service reading features through a different control plane.

        This is how chat becomes workspace-scoped. The service is built once at startup, so
        it holds the deployment-wide control plane; a route resolves this with its own
        request's `ScopedFeatureControlPlane`, and from then on every feature read this
        service makes -- including the ones inside `confirm`, which reach the same
        control-plane methods the buttons do -- is narrowed to that workspace.

        A new instance rather than a mutation, because one process serves many requests
        concurrently and a service that swapped its own control plane per request would hand
        one caller another's scope.

        The transcript store and the action service are shared deliberately: both are keyed
        by `feature_id`, and the feature itself is what decides visibility. `history` gates on
        that before reading either.
        """
        return FeatureChatService(
            assistant_for=self._assistant_for,
            control_plane=control_plane,
            store=self._store,
            actions=self._actions,
            history_limit=self._history_limit,
        )

    async def history(self, feature_id: str) -> list[ChatMessage]:
        """Return the conversation so reopening a feature does not lose it.

        Any message still claiming to be executing is re-read from its durable action first.
        Opening a feature after a restart has to show what the platform now knows -- that the
        action succeeded, or that it needs somebody -- rather than a spinner left behind by a
        process that no longer exists.

        The visibility check is `require_visible` rather than a record load: the transcript
        lives in its own table keyed by `feature_id`, and this endpoint exists so reopening a
        feature is cheap. Without it, a caller who knows a feature id would read a
        conversation about work in somebody else's workspace -- the transcript quotes
        repository names, failures and plans.
        """
        await self._control_plane.require_visible(feature_id)
        messages = await self._store.history(feature_id, limit=self._history_limit)
        if self._actions is None:
            return messages
        refreshed: list[ChatMessage] = []
        for message in messages:
            if (
                message.action_status in {"pending", "executing", "needs_attention"}
                and message.action_id is not None
                and message.id is not None
            ):
                refreshed.append(
                    await self.refresh_action_status(feature_id, message.id) or message
                )
            else:
                refreshed.append(message)
        return refreshed

    async def send(
        self, feature_id: str, message: str, *, credentials: RequestScopedCredentials
    ) -> list[ChatMessage]:
        """Record the question, answer it, and return both turns.

        The assistant is built before the question is recorded: a deployment or request with
        no usable provider key must refuse without leaving a question in the transcript that
        nothing ever answered. The feature is read first only because chat runs on the
        feature's own platform -- a feature planned on Claude and explained by GPT is a
        confusing artefact -- and reading it changes nothing.
        """
        record = await self._control_plane.get_record(feature_id)
        assistant = self._assistant_for(credentials, agent_platform=record.state.agent_platform)
        events = await self._control_plane.timeline(feature_id)
        context = build_context(
            record.state,
            query=message,
            events=[
                {"timestamp": str(item[0]), "source": item[2], "event": item[3], "details": item[4]}
                for item in events
            ],
        )
        previous = await self._store.history(feature_id, limit=self._history_limit)
        user = await self._store.append(feature_id, ChatMessage(role="user", content=message))

        reply = await assistant.respond(
            feature_id=feature_id,
            context=context,
            history=[{"role": item.role, "content": item.content} for item in previous],
            message=message,
        )
        answer = await self._store.append(
            feature_id,
            ChatMessage(
                role="assistant",
                content=reply.reply,
                proposed_action=_action_payload(reply.action),
                # A proposal starts as pending. Nothing has happened to the feature yet.
                action_status="pending" if reply.action else None,
            ),
        )
        return [user, answer]

    async def send_stream(
        self, feature_id: str, message: str, *, credentials: RequestScopedCredentials
    ) -> AsyncIterator[ChatStreamEvent]:
        """Answer one question, handing back the text as it is written.

        The turns are persisted around the stream, not by it. The question is written before
        the model is asked, so a connection that dies mid-answer leaves a transcript that
        shows what was asked; and whatever prose arrived is written when the stream ends,
        however it ended. A stream is a way of watching an answer being written, not the
        record of it -- the record is the transcript, and reopening the feature must show the
        same thing whether somebody watched it arrive or not.
        """
        record = await self._control_plane.get_record(feature_id)
        assistant = self._assistant_for(credentials, agent_platform=record.state.agent_platform)
        events = await self._control_plane.timeline(feature_id)
        context = build_context(
            record.state,
            query=message,
            events=[
                {"timestamp": str(item[0]), "source": item[2], "event": item[3], "details": item[4]}
                for item in events
            ],
        )
        previous = await self._store.history(feature_id, limit=self._history_limit)
        user = await self._store.append(feature_id, ChatMessage(role="user", content=message))
        yield ChatStreamEvent(kind="user", message=user)

        prose: list[str] = []
        action: ProposedAction | None = None
        failure: str | None = None
        try:
            async for chunk in assistant.stream(
                feature_id=feature_id,
                context=context,
                history=[{"role": item.role, "content": item.content} for item in previous],
                message=message,
            ):
                if chunk.done:
                    action = chunk.action
                elif chunk.text:
                    prose.append(chunk.text)
                    yield ChatStreamEvent(kind="delta", text=chunk.text)
        except AgentArtifactError as error:
            # The answer was written and shown; only the command after it was unreadable.
            # Keeping the prose and dropping the proposal is the honest outcome -- inventing
            # a proposal from a malformed tail would put a button in front of somebody that
            # does something other than what it says.
            failure = str(error)
        except (TimeoutError, ConnectionError, asyncio.CancelledError) as error:
            # Interrupted partway. Whatever arrived is still worth keeping: it is what the
            # person read, and losing it would make the transcript disagree with the screen.
            failure = "The answer was interrupted before it finished."
            if isinstance(error, asyncio.CancelledError):
                await self._persist_stream(feature_id, prose, action=None, failure=failure)
                raise

        answer = await self._persist_stream(feature_id, prose, action=action, failure=failure)
        yield ChatStreamEvent(kind="message", message=answer, detail=failure)

    async def _persist_stream(
        self,
        feature_id: str,
        prose: list[str],
        *,
        action: ProposedAction | None,
        failure: str | None,
    ) -> ChatMessage:
        """Write down what the assistant actually said, complete or not."""
        content = "".join(prose).strip()
        if not content:
            content = (
                "The assistant did not produce an answer."
                if failure is None
                else f"The assistant did not produce an answer. {failure}"
            )
        elif failure is not None:
            content = f"{content}\n\n_{failure}_"
        return await self._store.append(
            feature_id,
            ChatMessage(
                role="assistant",
                content=content,
                proposed_action=_action_payload(action),
                action_status="pending" if action else None,
            ),
        )

    async def confirm(
        self,
        feature_id: str,
        message_id: int,
        *,
        credentials: RequestScopedCredentials,
        actor_id: str,
        actor_display_name: str | None = None,
        authorize: Callable[[str], None] | None = None,
    ) -> ChatMessage:
        """Turn a confirmed proposal into a durable action, and run it once.

        The proposal is recorded as an intent before anything executes, and its identity
        covers the state it was decided against. Two confirmations of the same proposal --
        a double-clicked button, a retried request, the same repository proposed twice in
        one conversation -- resolve to one action and therefore one effect; the second gets
        the first's recorded result rather than a second retry of somebody's repository.

        ``authorize`` is called with the action type once it is known, so a confirmation has
        to meet the same bar as the endpoint that does the same thing. It is optional only so
        an application assembled without identity still works; the routes always pass one.

        The workspace check is first, before the transcript is even read. The control-plane
        call further down is scoped and would refuse eventually, but only after this method
        had told the caller whether a message existed and what kind of proposal it carried --
        which is a description of somebody else's work.
        """
        await self._control_plane.require_visible(feature_id)
        message = await self._store.get(feature_id, message_id)
        if message is None or message.proposed_action is None:
            msg = "no proposed action on that message"
            raise ChatActionError(msg)
        if message.action_status in {"rejected", "needs_attention"}:
            raise ChatActionError(f"that proposal is already {message.action_status}")
        if self._actions is None:
            msg = "this deployment cannot execute chat actions durably, so it will not run one"
            raise ChatActionError(msg)

        proposal = message.proposed_action
        action_type, arguments = _validated_proposal(proposal)
        # Checked here because this is the first point at which the action type is known.
        # The route can only require "may execute a chat action" -- it has a message id and
        # nothing else -- and that would let somebody allowed to answer a clarification also
        # approve a repository repair, purely for having asked in a sentence.
        if authorize is not None:
            authorize(action_type)
        record = await self._control_plane.get_record(feature_id)
        action, _created = await self._actions.submit(
            feature_id=feature_id,
            action_type=action_type,
            actor_id=actor_id,
            actor_display_name=actor_display_name,
            repository_id=_repository_for(action_type, arguments),
            payload=arguments,
            context_version=action_context_version(record.state, action_type, arguments),
            origin="chat",
            origin_message_id=message_id,
        )
        await self._store.attach_action(
            feature_id,
            message_id,
            action_id=action.action_id,
            # This request is about to execute it under a lease. A recovered CONFIRMED action
            # maps to pending so it can be confirmed again, but this first execution must not
            # briefly reopen its button to a concurrent click.
            status="executing",
        )

        try:
            executed = await self._actions.execute(
                action,
                lambda: self._execute(
                    feature_id, proposal, credentials=credentials, actor_id=actor_id
                ),
            )
        except ActionConflictError as error:
            # Already running elsewhere, out of attempts, or awaiting reconciliation. Each is
            # a refusal with a reason, not a platform fault.
            current = await self._actions.store.get(action.action_id)
            if current is not None:
                await self._store.project_action_status(
                    feature_id,
                    message_id,
                    status=_message_status(current.status),
                    result=current.result_summary or current.error_message,
                )
            raise ChatActionError(str(error)) from error
        except Exception:
            refreshed = await self._actions.store.get(action.action_id)
            await self._store.project_action_status(
                feature_id,
                message_id,
                status=_message_status(
                    refreshed.status if refreshed is not None else FeatureActionStatus.FAILED
                ),
                result=(
                    refreshed.error_message
                    if refreshed is not None
                    else "The platform did not complete this action."
                ),
            )
            raise
        return await self._store.project_action_status(
            feature_id,
            message_id,
            status=_message_status(executed.status),
            result=executed.result_summary or executed.error_message,
        )

    async def refresh_action_status(self, feature_id: str, message_id: int) -> ChatMessage | None:
        """Re-read a message's action and mirror whatever the platform now knows.

        This is how a conversation recovers after a restart: recovery decides what happened
        to the action, and the transcript stops claiming something is executing when nothing
        is. Returns nothing when the message carries no durable action.
        """
        message = await self._store.get(feature_id, message_id)
        if message is None or message.action_id is None or self._actions is None:
            return message
        action = await self._actions.store.get(message.action_id)
        if action is None:
            return message
        projected = _message_status(action.status)
        if projected == message.action_status:
            return message
        return await self._store.project_action_status(
            feature_id,
            message_id,
            status=projected,
            result=action.result_summary or action.error_message,
        )

    async def reject(self, feature_id: str, message_id: int) -> ChatMessage:
        """Record that a person declined a proposal, so the transcript shows the decision.

        This method reaches no control-plane method at all -- it writes only to the
        transcript -- so unlike `confirm` it has nothing downstream that would refuse a
        foreign feature later. Without the check here, anybody who knew a feature id could
        decline a proposal in somebody else's conversation.
        """
        await self._control_plane.require_visible(feature_id)
        message = await self._store.get(feature_id, message_id)
        if message is None or message.proposed_action is None:
            msg = "no proposed action on that message"
            raise ChatActionError(msg)
        claimed = await self._store.claim_action(feature_id, message_id)
        if claimed is None:
            current = await self._store.get(feature_id, message_id)
            state = current.action_status if current is not None else "unavailable"
            raise ChatActionError(f"that proposal is already {state}")
        return await self._store.record_action_outcome(
            feature_id, message_id, status="rejected", result="Rejected by the operator."
        )

    async def _execute(
        self,
        feature_id: str,
        action: dict[str, Any],
        *,
        credentials: RequestScopedCredentials,
        actor_id: str,
    ) -> str:
        """Run one supported action. Each call is the same method the REST route uses."""
        action_type, arguments = _validated_proposal(action)

        if action_type == "APPROVE_REPOSITORY_REPAIR":
            repair_id = str(arguments.get("repair_id", ""))
            approved = await self._control_plane.approve_repair(
                feature_id,
                repair_id=repair_id,
                actor_id=actor_id,
                credentials=credentials,
            )
            return f"Repair {repair_id} was approved; feature is now {approved.state.status.value}."

        if action_type == "REJECT_REPOSITORY_REPAIR":
            repair_id = str(arguments.get("repair_id", ""))
            await self._control_plane.reject_repair(
                feature_id,
                repair_id=repair_id,
                actor_id=actor_id,
                reason=str(arguments.get("reason", "")),
            )
            return f"Repair {repair_id} was rejected and will not be applied."

        if action_type == "CANCEL_WORKFLOW":
            reason = arguments.get("reason")
            record = await self._control_plane.cancel(
                feature_id,
                reason=str(reason) if reason else None,
                requested_by=actor_id,
            )
            return f"Feature is now {record.state.status.value}."

        if action_type == "RESUME_WORKFLOW":
            record = await self._control_plane.resume(
                feature_id, answers=[], credentials=credentials
            )
            return f"Feature is now {record.state.status.value}."

        if action_type == "ANSWER_CLARIFICATION":
            answers = arguments.get("answers")
            if not isinstance(answers, list):  # Established by validation above.
                raise ChatActionError("the proposal carried no answers")
            parsed_answers = [
                ClarificationAnswer(
                    question_id=str(item["question_id"]), answer=str(item["answer"])
                )
                for item in answers
                if isinstance(item, dict)
            ]
            record = await self._control_plane.resume(
                feature_id,
                answers=parsed_answers,
                credentials=credentials,
            )
            return f"Answers submitted; feature is now {record.state.status.value}."

        if action_type == "RETRY_WORKSTREAM":
            repository_id = arguments.get("repository_id")
            reason = arguments.get("reason")
            attempts = arguments.get("additional_attempts")
            if (
                not isinstance(repository_id, str)
                or not isinstance(reason, str)
                or not isinstance(attempts, int)
            ):
                raise ChatActionError("the retry proposal is missing its audited grant details")
            record = await self._control_plane.retry_workstream(
                feature_id,
                repository_id,
                additional_attempts=attempts,
                requested_by=actor_id,
                reason=reason,
                credentials=credentials,
            )
            return (
                f"Repository {repository_id} was granted {attempts} additional attempt"
                f"{'s' if attempts != 1 else ''}; feature is now {record.state.status.value}."
            )

        # Unreachable through the assistant, which validates the type before proposing. Kept so
        # a stored proposal from an older build cannot execute something unintended.
        msg = f"unsupported action: {action_type}"
        raise ChatActionError(msg)


def _action_payload(action: ProposedAction | None) -> dict[str, Any] | None:
    """Store a proposal as data, so confirming it never re-reads the assistant's prose."""
    if action is None:
        return None
    return {"type": action.type, "arguments": action.arguments, "summary": action.summary}


def _validated_proposal(action: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Read a stored proposal back as a checked command.

    Stored rows are untrusted input too. Revalidating on confirmation is what stops a
    proposal written by an older build, or altered in the database, from executing something
    the current command contract does not allow.
    """
    action_type = action.get("type")
    arguments = action.get("arguments")
    if not isinstance(action_type, str) or not isinstance(arguments, dict):
        raise ChatActionError("the stored proposal is malformed")
    try:
        validate_action_arguments(action_type, arguments)
    except AgentArtifactError as error:
        raise ChatActionError(str(error)) from error
    return action_type, arguments


def _repository_for(action_type: str, arguments: dict[str, Any]) -> str | None:
    """Return the repository an action is scoped to, by identifier and never by role.

    Two repositories may be retried with otherwise identical arguments; without this they
    would share an action identity and the second would silently replay the first.
    """
    if action_type in {"RETRY_WORKSTREAM", "APPROVE_REPOSITORY_REPAIR", "REJECT_REPOSITORY_REPAIR"}:
        repository_id = arguments.get("repository_id")
        return str(repository_id) if isinstance(repository_id, str) and repository_id else None
    return None


# How a durable action's state reads in the transcript. The existing vocabulary is kept so
# older messages and the client's rendering stay valid; only the unconfirmed outcome is new,
# and it is new because before durable actions there was no way to be in it.
_MESSAGE_STATUS = {
    FeatureActionStatus.PROPOSED: "pending",
    # Recovery gives an abandoned, provably unstarted/idempotent action another attempt.
    # Returning it to pending is what lets the person trigger that attempt; calling it
    # executing would leave a spinner with no executor or live lease.
    FeatureActionStatus.CONFIRMED: "pending",
    FeatureActionStatus.CLAIMED: "executing",
    FeatureActionStatus.EXECUTING: "executing",
    FeatureActionStatus.SUCCEEDED: "executed",
    FeatureActionStatus.FAILED: "failed",
    FeatureActionStatus.CANCELLED: "failed",
    FeatureActionStatus.REQUIRES_RECONCILIATION: "needs_attention",
}


def _message_status(status: FeatureActionStatus) -> str:
    """Project a durable action status onto the transcript's vocabulary."""
    return _MESSAGE_STATUS.get(status, "failed")


def utc_now() -> datetime:
    """Return an aware timestamp for an in-memory transcript."""
    return datetime.now(tz=UTC)


__all__ = [
    "ChatActionError",
    "ChatMessage",
    "ChatMessageStore",
    "ChatStreamEvent",
    "FeatureChatService",
]
