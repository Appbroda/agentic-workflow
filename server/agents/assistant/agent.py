"""The feature assistant, which explains a feature and proposes actions it cannot perform."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from adapters.llm_adapter import LLMClient
from agents.shared.contracts import AgentArtifactError, parse_model_json
from prompts.prompt_loader import PromptLoader

_HISTORY_LIMIT = 12

# Where the answer stops and the command begins. Chosen to be something no ordinary prose or
# Markdown produces: a reply that happened to contain it would have its tail read as a
# command, and the streamed text before it shown as the whole answer.
ACTION_DELIMITER = "<<<PLATFORM_ACTION>>>"


@dataclass(frozen=True, slots=True)
class ActionDefinition:
    """One thing the platform can be asked to do, and what it needs."""

    type: str
    description: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ProposedAction:
    """A structured action awaiting a person's confirmation. Never executed by the assistant."""

    type: str
    arguments: dict[str, Any]
    summary: str


@dataclass(frozen=True, slots=True)
class AssistantChunk:
    """One piece of a streamed answer, or the end of one.

    Prose and proposal arrive on the same stream but are not the same thing, so they are
    distinguished here rather than left for a caller to tell apart by inspecting text. Only
    ``text`` is ever shown; ``action`` is a command, and it exists only on the final chunk,
    after the whole output has been read and validated.
    """

    text: str = ""
    done: bool = False
    action: ProposedAction | None = None


@dataclass(frozen=True, slots=True)
class AssistantReply:
    """What the assistant said, and what it proposed if anything."""

    reply: str
    action: ProposedAction | None
    model: str
    response_id: str


# Exactly the actions the control plane exposes today, and nothing beyond them: offering an
# action with no endpoint behind it would let the assistant propose something no confirmation
# could carry out. Repository repair joined the list when it became a real endpoint.
SUPPORTED_ACTIONS: tuple[ActionDefinition, ...] = (
    ActionDefinition(
        type="ANSWER_CLARIFICATION",
        description=(
            "Answer the questions this feature is waiting on, which resumes it. Every open "
            "question needs an answer."
        ),
        arguments='{"answers": [{"question_id": "...", "answer": "..."}]}',
    ),
    ActionDefinition(
        type="RESUME_WORKFLOW",
        description=(
            "Continue an interrupted feature from its last safe checkpoint. Not a retry: the "
            "platform refuses if a child has already spent its attempt budget."
        ),
        arguments="{}",
    ),
    ActionDefinition(
        type="RETRY_WORKSTREAM",
        description=(
            "Grant one stopped repository more attempts. Propose this only when the person "
            "provided the repository id, their name, why another attempt is justified, and "
            "the number of attempts (one to five)."
        ),
        arguments=(
            '{"repository_id": "...", "additional_attempts": 1, '
            '"requested_by": "...", "reason": "..."}'
        ),
    ),
    ActionDefinition(
        type="CANCEL_WORKFLOW",
        description=(
            "Stop scheduling further work. Pushed commits and open pull requests are kept."
        ),
        arguments='{"reason": "..."}',
    ),
    ActionDefinition(
        type="APPROVE_REPOSITORY_REPAIR",
        description=(
            "Apply a proposed repair to a repository's checked-in setup and give that "
            "repository another attempt. Propose this only for a repair the context shows is "
            "proposed, and only when the person asked for that specific repair by id."
        ),
        arguments='{"repair_id": "...", "repository_id": "..."}',
    ),
    ActionDefinition(
        type="REJECT_REPOSITORY_REPAIR",
        description=(
            "Decline a proposed repository repair, leaving the repository stopped. Requires "
            "the reason the person gave for declining it."
        ),
        arguments='{"repair_id": "...", "repository_id": "...", "reason": "..."}',
    ),
)

_SUPPORTED_TYPES = frozenset(item.type for item in SUPPORTED_ACTIONS)


class FeatureAssistant:
    """Answer questions about one feature from its own record, and propose nothing else."""

    def __init__(self, *, prompt_loader: PromptLoader, llm_client: LLMClient) -> None:
        """Inject the versioned prompt and the configured model boundary."""
        self._prompt_loader = prompt_loader
        self._llm_client = llm_client

    async def respond(
        self,
        *,
        feature_id: str,
        context: dict[str, Any],
        history: list[dict[str, str]],
        message: str,
    ) -> AssistantReply:
        """Return the assistant's answer, and a proposal only when one was asked for."""
        instructions = self._prompt_loader.render(
            "assistant/v1.jinja2",
            feature_id=feature_id,
            supported_actions=[
                {"type": item.type, "description": item.description, "arguments": item.arguments}
                for item in SUPPORTED_ACTIONS
            ],
            context=json.dumps(context, indent=2, sort_keys=True, default=str),
            history=json.dumps(history[-_HISTORY_LIMIT:], indent=2),
        )
        response = await self._llm_client.respond(instructions=instructions, input_text=message)
        payload = parse_model_json(response.output_text, expected_keys=("reply", "action"))
        return AssistantReply(
            reply=_reply_text(payload),
            action=_proposed_action(payload.get("action")),
            model=response.model,
            response_id=response.response_id,
        )

    async def stream(
        self,
        *,
        feature_id: str,
        context: dict[str, Any],
        history: list[dict[str, str]],
        message: str,
    ) -> AsyncIterator[AssistantChunk]:
        """Yield the answer as it is written, then the proposal it ended with.

        Prose is streamed; the command is not. Once the delimiter arrives the rest is read
        rather than shown, because forwarding it would put JSON on somebody's screen and --
        worse -- show them a proposal before the platform had established it is one it can
        carry out. The final chunk carries the validated action, or none.
        """
        instructions = self._prompt_loader.render(
            "assistant/v2.jinja2",
            feature_id=feature_id,
            action_delimiter=ACTION_DELIMITER,
            supported_actions=[
                {"type": item.type, "description": item.description, "arguments": item.arguments}
                for item in SUPPORTED_ACTIONS
            ],
            context=json.dumps(context, indent=2, sort_keys=True, default=str),
            history=json.dumps(history[-_HISTORY_LIMIT:], indent=2),
        )
        client = self._llm_client
        if not hasattr(client, "stream"):
            msg = "this deployment's model boundary cannot stream"
            raise AgentArtifactError(msg)

        # The delimiter can be split across two deltas, so a suffix that might be the start
        # of one is withheld until the next delta settles it. Emitting optimistically would
        # put a stray "<<<PLATFORM" on the screen before the rest of it arrived.
        emitted = 0
        buffered = ""
        reached_action = False
        async for delta in client.stream(instructions=instructions, input_text=message):
            buffered += delta
            if reached_action:
                # Still reading: the command arrives across several deltas too. Nothing more
                # is shown, because everything from here is for the platform and not a reader.
                continue
            found = buffered.find(ACTION_DELIMITER)
            safe = found if found >= 0 else len(buffered) - _partial_delimiter_length(buffered)
            if safe > emitted:
                yield AssistantChunk(text=buffered[emitted:safe])
                emitted = safe
            reached_action = found >= 0

        # Raises if the tail is not a proposal this platform can carry out. The caller has
        # already shown the prose and is responsible for keeping it: an answer that was
        # correct does not stop being correct because the command after it was malformed.
        _, action = split_streamed_reply(buffered)
        yield AssistantChunk(done=True, action=action)


def _partial_delimiter_length(text: str) -> int:
    """Return how many trailing characters could still turn out to be the delimiter."""
    for length in range(min(len(ACTION_DELIMITER) - 1, len(text)), 0, -1):
        if ACTION_DELIMITER.startswith(text[-length:]):
            return length
    return 0


def split_streamed_reply(output: str) -> tuple[str, ProposedAction | None]:
    """Separate the prose a person read from the command the platform will execute.

    The delimiter exists because streaming and structure pull in opposite directions. A JSON
    envelope cannot be streamed to a reader -- they would watch quotes and braces arrive --
    and prose alone cannot carry a typed action. So the model writes the answer first and
    appends the proposal, and only the part before the delimiter is ever shown.

    A malformed tail is a failure, not a proposal to be salvaged. The prose has already been
    displayed and stands on its own; guessing at a half-written command would put a button in
    front of somebody that does something other than what it says.
    """
    prose, delimiter, tail = output.partition(ACTION_DELIMITER)
    if not delimiter:
        return output.strip(), None
    if not tail.strip():
        # The model announced a proposal and did not write one. The answer is still worth
        # showing; the proposal is simply absent.
        return prose.strip(), None
    payload = parse_model_json(tail, expected_keys=("type", "arguments", "summary"))
    return prose.strip(), _proposed_action(payload)


def _reply_text(payload: dict[str, Any]) -> str:
    """Require something for a person to read; an empty answer is not an answer."""
    reply = payload.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        msg = "assistant response must contain a non-empty reply"
        raise AgentArtifactError(msg)
    return reply


def _proposed_action(value: object) -> ProposedAction | None:
    """Accept only an action the control plane can actually carry out.

    A proposal the platform cannot execute is worse than no proposal: it puts a button in
    front of a person that fails when they press it.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        msg = "assistant action must be an object or null"
        raise AgentArtifactError(msg)
    action_type = value.get("type")
    if action_type not in _SUPPORTED_TYPES:
        msg = f"assistant proposed an unsupported action: {sorted(_SUPPORTED_TYPES)}"
        raise AgentArtifactError(msg)
    arguments = value.get("arguments")
    summary = value.get("summary")
    if not isinstance(arguments, dict):
        msg = "assistant action arguments must be an object"
        raise AgentArtifactError(msg)
    if not isinstance(summary, str) or not summary.strip():
        msg = "assistant action must carry a summary a person can confirm"
        raise AgentArtifactError(msg)
    validate_action_arguments(str(action_type), arguments)
    return ProposedAction(type=str(action_type), arguments=arguments, summary=summary)


def _require_identifiers(arguments: dict[str, Any], names: tuple[str, ...], label: str) -> None:
    """Require each named argument to be a non-empty string identifier."""
    for name in names:
        value = arguments.get(name)
        if not isinstance(value, str) or not value.strip():
            raise AgentArtifactError(f"{label} requires a non-empty {name}")


def validate_action_arguments(action_type: str, arguments: dict[str, Any]) -> None:
    """Reject incomplete or unsafe proposals before they become confirmable controls."""
    if action_type not in _SUPPORTED_TYPES:
        raise AgentArtifactError(f"unsupported assistant action: {action_type}")

    if action_type == "RESUME_WORKFLOW":
        if arguments:
            raise AgentArtifactError("resume proposal does not accept arguments")
        return

    if action_type == "CANCEL_WORKFLOW":
        unknown = set(arguments) - {"reason"}
        if unknown:
            raise AgentArtifactError(f"cancel proposal has unknown arguments: {sorted(unknown)}")
        reason = arguments.get("reason")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise AgentArtifactError("cancel proposal reason must be a non-empty string")
        return

    if action_type == "APPROVE_REPOSITORY_REPAIR":
        if set(arguments) != {"repair_id", "repository_id"}:
            raise AgentArtifactError(
                "repair approval requires exactly a repair_id and a repository_id"
            )
        _require_identifiers(arguments, ("repair_id", "repository_id"), "repair approval")
        return

    if action_type == "REJECT_REPOSITORY_REPAIR":
        if set(arguments) != {"repair_id", "repository_id", "reason"}:
            raise AgentArtifactError(
                "repair rejection requires a repair_id, a repository_id and a reason"
            )
        _require_identifiers(arguments, ("repair_id", "repository_id"), "repair rejection")
        rejection_reason = arguments.get("reason")
        if not isinstance(rejection_reason, str) or not rejection_reason.strip():
            # A stop somebody chose to leave in place, with no recorded reason, explains
            # nothing to whoever picks the feature up next.
            raise AgentArtifactError("a repair rejection must record why it was rejected")
        return

    if action_type == "ANSWER_CLARIFICATION":
        if set(arguments) != {"answers"}:
            raise AgentArtifactError("clarification proposal requires only an answers list")
        answers = arguments.get("answers")
        if not isinstance(answers, list) or not answers:
            raise AgentArtifactError("clarification proposal requires at least one answer")
        question_ids: list[str] = []
        for answer in answers:
            if not isinstance(answer, dict) or set(answer) != {"question_id", "answer"}:
                raise AgentArtifactError(
                    "each clarification answer requires only question_id and answer"
                )
            question_id = answer.get("question_id")
            answer_text = answer.get("answer")
            if not isinstance(question_id, str) or not question_id.strip():
                raise AgentArtifactError("clarification answer question_id must be non-empty")
            if not isinstance(answer_text, str) or not answer_text.strip():
                raise AgentArtifactError("clarification answer text must be non-empty")
            question_ids.append(question_id)
        if len(question_ids) != len(set(question_ids)):
            raise AgentArtifactError("clarification proposal contains duplicate question ids")
        return

    if set(arguments) != {"repository_id", "additional_attempts", "requested_by", "reason"}:
        raise AgentArtifactError("retry proposal requires exactly its four audited arguments")
    repository_id = arguments.get("repository_id")
    requested_by = arguments.get("requested_by")
    reason = arguments.get("reason")
    attempts = arguments.get("additional_attempts")
    if not isinstance(repository_id, str) or not repository_id.strip():
        raise AgentArtifactError("retry proposal requires a repository_id")
    if not isinstance(requested_by, str) or not requested_by.strip() or len(requested_by) > 200:
        raise AgentArtifactError("retry proposal requires requested_by")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
        raise AgentArtifactError("retry proposal requires a reason")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 5:
        raise AgentArtifactError("retry proposal additional_attempts must be between 1 and 5")


__all__ = [
    "ActionDefinition",
    "AssistantReply",
    "FeatureAssistant",
    "ProposedAction",
    "SUPPORTED_ACTIONS",
    "validate_action_arguments",
]
