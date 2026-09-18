"""Configured model-provider clients and workspace-bounded coding executor adapters.

Two clients live here, one per platform this deployment can be configured with. They are
deliberately not related by inheritance: they normalize two genuinely different wire formats,
and the only thing they truly share is the ``LLMResponse`` they both already produce. Every
caller above this line depends on ``LLMClient`` and ``StreamingLLMClient`` and on nothing else.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from collections.abc import AsyncIterator, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from configs.model_roles import AgentPlatform, ModelConfig, ModelRole
from configs.settings import Settings
from services.cancellation import CancellationRequested, CancellationToken, await_cancellable
from services.external_operations import ExternalOperationExecutor
from services.process_runner import redact_output, redact_source_credentials
from state.external_operations import ExternalOperation, ExternalOperationType
from storage.external_operation_store import OperationResult
from tools.file_tools import (
    DEFAULT_MAX_FILE_BYTES,
    PathLike,
    WorkspaceFileTools,
    resolve_workspace_path,
    resolve_workspace_root,
)
from tools.llm_call_record import record_llm_call

_EDIT_EXCERPT_MAX_CHARACTERS = 6_000


class LLMAdapterError(RuntimeError):
    """Raised when an LLM response or coding execution violates its adapter contract.

    Diagnostics are opt-in per raise site and must never be declared wholesale. Most messages
    here are platform constants, but the coding executor's deliberately quote the workspace
    file a rejected edit failed to match, so the next attempt can copy real text instead of
    guessing -- and a repository's own configuration file is exactly where a credential-shaped
    literal lives. Only a raiser whose message is built entirely from constants may pass
    ``diagnostics``.
    """

    def __init__(
        self,
        *args: object,
        diagnostics: Sequence[str] = (),
        failure_classification: str | None = None,
    ) -> None:
        """Record safe diagnostics and an optional provider-owned failure category.

        The adapter type remains stable for retry handlers. ``failure_classification`` retains
        the SDK exception class that was wrapped, allowing terminal-state policy to distinguish
        a transient timeout from a malformed model response without importing provider types.
        """
        super().__init__(*args)
        self.diagnostics = tuple(diagnostics)
        self.failure_classification = (
            failure_classification.strip()
            if isinstance(failure_classification, str) and failure_classification.strip()
            else None
        )


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """The normalized response data returned from a configured LLM client.

    ``provider`` and ``reasoning_effort`` describe the call that produced this text, and they
    travel with the response because that is the only moment they are certainly true. An
    artifact written now and read next month must be able to name the effort its own execution
    asked for; reading it back from configuration would report whatever the deployment happens
    to be set to at the time of reading, which for a historical record is simply wrong.

    Both default to "not recorded" so a test double or an older payload stays constructible,
    and a reader can tell an unrecorded value from a recorded one.
    """

    response_id: str
    model: str
    output_text: str
    input_tokens: int | None
    output_tokens: int | None
    # How many times this response's request had to be issued again because the stream it
    # opened never spoke. Carried for the same reason the fields below it are: this is the
    # only moment it is certainly true, and it is invisible from every layer above -- a
    # re-issue costs one request and leaves no journal row of its own.
    #
    # `None` means unrecorded -- a non-streaming transport, the Anthropic client, an older
    # payload -- and is deliberately distinct from `0`, which means a stream that spoke on
    # the first issue. "Never measured" and "measured none" are different facts, and the
    # whole point of this field is to tell whether the budget is close to binding.
    stream_reissues: int | None = None
    provider: str = ""
    reasoning_effort: str | None = None
    model_role: str | None = None
    model_variable: str | None = None
    routing_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CodingExecutionResult:
    """A structured coding outcome that is independent from LangGraph state mutation."""

    response_id: str
    model: str
    summary: str
    modified_files: tuple[Path, ...]
    # Carried for the same reason ``LLMResponse`` carries them: the record of an attempt has to
    # be able to name what executed it long after the configuration has moved on.
    provider: str = ""
    reasoning_effort: str | None = None
    model_role: str | None = None
    model_variable: str | None = None
    routing_reason: str | None = None
    # What the call cost, when the provider reported it. None means unrecorded -- an older
    # journal payload or a test double -- never zero.
    input_tokens: int | None = None
    output_tokens: int | None = None
    # Carried for the in-attempt callers, who have no journal row to read it from: the
    # Engineer's repair and self-review-correction passes execute through this type and are
    # unjournaled by construction, so this is the only way their re-issues reach the
    # attempt's own record. `None` means unmeasured, exactly as on ``LLMResponse``.
    stream_reissues: int | None = None


@dataclass(frozen=True, slots=True)
class _ResolvedUpdate:
    """A file update reduced to the exact content that will be written."""

    path: str
    content: str


@dataclass(frozen=True, slots=True)
class _FileEdit:
    """One exact snippet replacement inside a file that already exists."""

    find: str
    replace: str


@dataclass(frozen=True, slots=True)
class _FileUpdate:
    """One validated file update parsed from a coding-model response.

    Either whole-file ``content`` or a list of ``edits`` applied to the file already in the
    workspace. Edits exist because whole-file replacement asks a model to reproduce every
    line it is not changing, and a long file is answered by summarizing it away: a homepage
    lost two hundred and seventy lines of tiles to a stub that kept only the new one.
    """

    path: str
    content: str | None = None
    edits: tuple[_FileEdit, ...] = ()


@dataclass(frozen=True, slots=True)
class ImageInput:
    """One image accompanying a request: what it is, and its bytes, and nothing else.

    Deliberately not a marker, a caption, or a position in a narrative. Those are the agent's
    business, expressed in the text it already sends -- an adapter that knew about markers
    would be a second place the marker syntax lives.

    ``data`` is base64 with no data-URL prefix. Each provider wants it differently wrapped and
    each client does its own wrapping; carrying a half-formatted string here would make one of
    them strip what the other added.
    """

    media_type: str
    data: str


class LLMClient(Protocol):
    """Protocol for a configured text-response model with no request-supplied credentials."""

    async def respond(
        self, *, instructions: str, input_text: str, images: Sequence[ImageInput] = ()
    ) -> LLMResponse:
        """Return a normalized model response for the configured agent.

        ``images`` accompany the request; they are never the request. ``input_text`` must
        still be non-empty, and the empty-request refusal is unchanged: a set of pictures
        with nothing asked about them is not a question.

        With no images every implementation sends byte for byte what it sent before this
        parameter existed. That is asserted rather than assumed -- see
        ``test_attachment_model_calls``.
        """

    @property
    def vision_capable(self) -> bool:
        """Whether this deployment has declared this client's model able to read an image.

        Read here rather than decided here: the answer is `MODEL_VISION_CAPABLE`, and an
        undeclared model answers False. Published on the boundary so the call site can refuse
        rather than drop -- for a tier feature the deployment can change between acceptance
        and the run, and a redeploy that swaps a tier's reasoning model for one not declared
        vision-capable makes this the only thing standing between the feature and a silent
        drop.
        """


class StreamingLLMClient(LLMClient, Protocol):
    """A model boundary that can also hand back text as it arrives.

    Separate from ``LLMClient`` because most agents have no use for it: an agent producing an
    artifact needs the whole document validated before anything acts on it, and streaming a
    half-written artifact would only invite something to read one.
    """

    def stream(self, *, instructions: str, input_text: str) -> AsyncIterator[str]:
        """Yield output text as the model produces it."""


class CodingExecutor(Protocol):
    """Protocol for coding execution that returns results instead of mutating workflow state."""

    async def execute(
        self,
        *,
        workspace_root: PathLike,
        instructions: str,
        input_text: str,
        cancellation_token: CancellationToken | None = None,
        operation_executor: ExternalOperationExecutor | None = None,
        images: Sequence[ImageInput] = (),
    ) -> CodingExecutionResult:
        """Apply only validated workspace file updates and return a structured result.

        ``images`` accompany the request on `respond`'s terms exactly: they are never the
        request, and with none of them every implementation sends byte for byte what it sent
        before the parameter existed. They exist so the role that writes the CSS can be shown
        the design rather than only its coordinates -- AB-Feature-231 produced a screen of
        correctly-positioned empty boxes from a rendering it could read and could not see.
        """


# The provider each client talks to, named once per client. Recorded on every response so an
# execution record can say "OpenAI · GPT-5.6 Sol" or "Anthropic · Claude Opus 5" without any
# reader inferring a provider from the shape of a model identifier.
_PROVIDER = AgentPlatform.OPENAI.value
_ANTHROPIC_PROVIDER = AgentPlatform.ANTHROPIC.value


def _recorded(response: LLMResponse) -> LLMResponse:
    """Note the response's model facts for the journal row wrapping this call, and pass it on.

    Both clients return through this, so a journaled planning call can say which model
    answered it even when the value the agent returns -- a reconciliation, a list of
    questions -- carries no metadata of its own. Set at the moment the response exists,
    task-locally; see ``tools.llm_call_record`` for the semantics.
    """
    record_llm_call(
        model=response.model,
        provider=response.provider or None,
        reasoning_effort=response.reasoning_effort,
        model_role=response.model_role,
        model_variable=response.model_variable,
        routing_reason=response.routing_reason,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        stream_reissues=response.stream_reissues,
    )
    return response


class OpenAILLMClient:
    """OpenAI Responses client configured solely from platform settings and environment."""

    def __init__(
        self,
        settings: Settings,
        agent_name: str,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        model_role: ModelRole | None = None,
        resolved_model: str | None = None,
        resolved_reasoning_effort: str | None = None,
        resolved_model_variable: str | None = None,
        resolved_routing_reason: str | None = None,
    ) -> None:
        """Bind one agent's policy and an optional request-scoped provider credential.

        ``model_role`` selects the model and effort while the named agent still supplies most
        operational policy. The reasoning role has one deployment-configurable deadline because
        maximum-effort planning can be much slower than the legacy agent default; other roles
        retain the named agent's timeout and SDK retry policy.
        """
        try:
            agent_settings = settings.agents[agent_name]
        except KeyError as error:
            msg = f"unknown agent: {agent_name}"
            raise LLMAdapterError(msg) from error
        resolved_role = model_role or agent_settings.model_role
        role_config = (
            settings.model_config_for_role(resolved_role, platform=AgentPlatform.OPENAI)
            if resolved_role is not None
            else None
        )
        self._model_role = resolved_role
        if resolved_model is not None and resolved_model.strip():
            # Recovery executes the selection persisted before the model call. Reading current
            # configuration here would let a restart silently change what an already-routed
            # attempt runs on and make its routing record false.
            self._model = resolved_model.strip()
            self._reasoning_effort = resolved_reasoning_effort
            self._model_variable = resolved_model_variable
            self._routing_reason = resolved_routing_reason or (
                role_config.routing_reason
                if role_config is not None
                else "The recovered attempt uses its persisted model selection."
            )
        else:
            self._model = (
                role_config.model
                if role_config is not None
                else settings.model_for_agent(agent_name, platform=AgentPlatform.OPENAI)
            )
            # Asked of settings rather than read off the role config, because an agent that
            # declares its own effort outranks the role's default and only settings knows
            # that. `model_role` is the routed role or nothing: passing it through is what
            # keeps a deliberately routed call on its role's level.
            self._reasoning_effort = settings.reasoning_effort_for_agent(
                agent_name, platform=AgentPlatform.OPENAI, model_role=model_role
            )
            self._model_variable = role_config.model_variable if role_config is not None else None
            self._routing_reason = (
                role_config.routing_reason
                if role_config is not None
                else "The agent uses its configured model boundary."
            )
        self._temperature = agent_settings.temperature
        self._temperature_unsupported = settings.declared_temperature_unsupported()
        # Read once, against the model this client is fixed to, for the reason the effort and
        # temperature declarations are read once: a client already running must not change
        # what it believes about its model because configuration moved underneath it.
        self._vision_capable = self._model in settings.declared_vision_capable()
        self._timeout_seconds = settings.timeout_seconds_for_agent(
            agent_name, model_role=resolved_role
        )
        # Read once here rather than at each call, so one client's behaviour cannot change
        # underneath a workstream that is already running on it.
        self._first_event_timeout_seconds = settings.first_event_timeout_seconds
        self._client = (
            client
            if client is not None
            else _create_openai_client(
                api_key=api_key,
                timeout_seconds=self._timeout_seconds,
                max_retries=agent_settings.max_retries,
            )
        )

    @property
    def model(self) -> str:
        """Return the configured model this client will call."""
        return self._model

    @property
    def model_role(self) -> ModelRole | None:
        """Return the routed role this client was built for, where one selected the model."""
        return self._model_role

    @property
    def reasoning_effort(self) -> str | None:
        """Return the normalized effort this client asks for, or nothing for the default."""
        return self._reasoning_effort

    @property
    def provider(self) -> str:
        """Name the provider this client calls. Never a key, a base URL, or an account."""
        return _PROVIDER

    @property
    def vision_capable(self) -> bool:
        """Whether the deployment declared this client's model able to read an image."""
        return self._vision_capable

    async def respond(
        self, *, instructions: str, input_text: str, images: Sequence[ImageInput] = ()
    ) -> LLMResponse:
        """Call the Responses API using the model and operational limits fixed at construction."""
        if not instructions.strip() or not input_text.strip():
            msg = "instructions and input_text must not be empty"
            raise LLMAdapterError(msg, diagnostics=(msg,), failure_classification="empty_request")
        if images and not self._vision_capable:
            # Raised rather than dropped: an image the model will not see must not vanish
            # quietly. This is the second fence -- acceptance already refused a submission
            # whose reasoning model is undeclared -- and it exists because a redeploy can
            # change what a tier resolves to between acceptance and the run.
            msg = (
                f"{self._model} is not declared able to read images; add it to "
                "MODEL_VISION_CAPABLE or run this feature on a model that is"
            )
            raise LLMAdapterError(
                msg, diagnostics=(msg,), failure_classification="model_not_vision_capable"
            )
        request: dict[str, Any] = {
            "model": self._model,
            "instructions": instructions,
            # With no images this stays exactly the bare string it has always been. The
            # structured form is only built when there is something to structure, so a
            # feature submitted without pictures produces byte for byte today's request.
            "input": (_openai_input_with_images(input_text, images) if images else input_text),
            "store": False,
            "timeout": self._timeout_seconds,
        }
        # GPT-5 Responses models reject the legacy temperature parameter, and any model the
        # deployment declares in MODEL_TEMPERATURE_UNSUPPORTED rejects it too. Keep the
        # configured temperature for compatible models while allowing modern
        # reasoning and coding models to use their API defaults.
        if _supports_temperature(self._model, self._temperature_unsupported):
            request["temperature"] = self._temperature
        _apply_reasoning_effort(request, self._reasoning_effort)
        # `None` while the transport is a plain POST: there is no stream to have gone silent,
        # so no count exists to report, and reporting zero would claim a measurement nobody
        # took. Only the streamed path below can answer this question at all.
        reissues: int | None = None
        if self._timeout_seconds > _STREAM_TRANSPORT_ABOVE_SECONDS:
            response, reissues = await self._completed_response(request)
        else:
            with _provider_faults_declared():
                response = await self._client.responses.create(**request)
        output_value = getattr(response, "output_text", None)
        if not isinstance(output_value, str) or not output_value.strip():
            msg = "Responses API returned no output text"
            raise LLMAdapterError(msg, diagnostics=(msg,), failure_classification="empty_response")
        output_text = output_value
        usage = getattr(response, "usage", None)
        return _recorded(
            LLMResponse(
                response_id=str(getattr(response, "id", "")),
                model=str(getattr(response, "model", self._model)),
                output_text=output_text,
                input_tokens=_optional_integer(getattr(usage, "input_tokens", None)),
                output_tokens=_optional_integer(getattr(usage, "output_tokens", None)),
                stream_reissues=reissues,
                provider=_PROVIDER,
                # The effort this request actually asked for, after the one central capability
                # check has already dropped a level the deployment declared unsupported. So this
                # is what the provider was sent, not what configuration nominally holds.
                reasoning_effort=self._reasoning_effort,
                model_role=self._model_role.value if self._model_role is not None else None,
                model_variable=self._model_variable,
                routing_reason=self._routing_reason,
            )
        )

    async def _completed_response(self, request: dict[str, Any]) -> tuple[Any, int]:
        """Issue one Responses call as a stream; return its response and the re-issue count.

        The count travels back with the response rather than being logged here, because a
        re-issue that *succeeded* is the signal an operator needs to tell a comfortable
        first-event budget from one that is about to start failing -- and it is invisible
        everywhere else. It costs one request, writes no journal row of its own, and leaves
        only a slightly slower operation behind. Zero is a real answer and is reported.

        The result is exactly what the non-streaming call returns -- the same response object,
        read off the `response.completed` event -- so nothing above this line can tell the
        difference. Only the transport changes.

        It changes because a maximum-effort reasoning call thinks for tens of minutes, and a
        plain POST spends every one of them with no bytes moving in either direction. That is
        precisely what an idle-connection timeout, a proxy, or a load balancer reaps.
        AB-Feature-168 lost three consecutive planning runs to it in one morning -- two
        `APIConnectionError`, one `APITimeoutError`, twenty to forty-five minutes each, no
        output from any of them. A stream has events arriving while the model reasons, so the
        connection is visibly in use for as long as the work takes.

        The events themselves are deliberately not surfaced. This is the blocking call; a
        caller that wants tokens as they arrive already has ``stream``.

        A stream that never speaks is issued again rather than waited out. That is the one
        provider fault this adapter can prove had no effect -- nothing was delivered, so
        nothing was decided -- and re-issuing it here costs one request, where letting it
        reach the workstream costs an attempt. Every other fault still travels straight up.
        """
        issued = 0
        while True:
            issued += 1
            spoken = await self._stream_that_spoke(request)
            if spoken is not None:
                return await self._completed_from(*spoken), issued - 1
            if issued > _MAX_STREAM_REISSUES:
                # Says how many times, because a fault record that reads "the provider did
                # not answer" without saying how hard this tried sends its reader to look
                # for a timeout that is not the one that fired.
                msg = (
                    "Responses API accepted the request and sent no event within "
                    f"{self._first_event_timeout_seconds} seconds, on {issued} successive "
                    "issues of the same request"
                )
                raise LLMAdapterError(
                    msg, diagnostics=(msg,), failure_classification="stream_silent"
                )
            await asyncio.sleep(_STREAM_REISSUE_BACKOFF_SECONDS[issued - 1])

    async def _stream_that_spoke(self, request: dict[str, Any]) -> tuple[Any, Any] | None:
        """Open one stream and return it with its first event, or ``None`` if it never spoke.

        ``None`` is returned in exactly one situation: the request was issued and no event
        arrived inside the first-event budget. That is what makes the caller's
        re-issue safe, and the reason this returns a sentinel rather than raising -- an
        exception here would be indistinguishable from the faults that *did* deliver
        something, and re-issuing one of those would be the guess the platform forbids.

        The budget covers the request as well as the first read. A response whose headers
        never arrive has delivered exactly as much as one whose first event never arrives.
        """
        stream: Any = None
        try:
            async with asyncio.timeout(self._first_event_timeout_seconds):
                with _provider_faults_declared():
                    stream = await self._client.responses.create(**request, stream=True)
                with _provider_faults_declared():
                    try:
                        return stream, await anext(stream)
                    except StopAsyncIteration:
                        # An empty stream is an answer, not silence: the provider closed the
                        # response. Handed on so `_completed_from` gives it the
                        # incomplete-stream fault, which is terminal for this request.
                        return stream, None
        except TimeoutError:
            # Only this adapter's own budget lands here. A cancellation from above raises
            # `CancelledError`, which `_provider_faults_declared` refuses to absorb and
            # `asyncio.timeout` does not convert.
            await _closed_quietly(stream)
            return None

    async def _completed_from(self, stream: Any, first: Any) -> Any:
        """Drain a stream that has already spoken, and return the response it completes with.

        Takes the first event rather than re-reading it, because it has already been consumed
        off the wire: this is the same single pass the plain loop always was, resumed one
        event in. From here the agent's own deadline governs, applied by the SDK between
        events -- a model that thinks for forty minutes is exactly as welcome as it ever was.
        """
        completed: Any = None
        event = first
        while event is not None:
            if getattr(event, "type", "") == "response.completed":
                completed = getattr(event, "response", None)
            # Stepped by hand for the same reason ``stream`` does it: reading the next event
            # is another network read, and it has to be inside the fault wrapper.
            with _provider_faults_declared():
                try:
                    event = await anext(stream)
                except StopAsyncIteration:
                    event = None
        if completed is None:
            # The stream ended without the provider ever saying the response was finished.
            # Reported as this adapter's own fault rather than returning a partial answer:
            # half a plan that validates is worse than no plan at all.
            msg = "Responses API stream ended without a completed response"
            raise LLMAdapterError(
                msg, diagnostics=(msg,), failure_classification="incomplete_stream"
            )
        return completed

    async def stream(self, *, instructions: str, input_text: str) -> AsyncIterator[str]:
        """Yield output text as the model produces it.

        Only the text deltas are surfaced. The Responses stream carries other event types --
        reasoning, tool calls, completion envelopes -- and a caller that forwarded whatever
        arrived would be putting the provider's internal event vocabulary on somebody's
        screen. Anything unrecognised is skipped rather than guessed at.
        """
        if not instructions.strip() or not input_text.strip():
            msg = "instructions and input_text must not be empty"
            raise LLMAdapterError(msg, diagnostics=(msg,), failure_classification="empty_request")
        request: dict[str, Any] = {
            "model": self._model,
            "instructions": instructions,
            "input": input_text,
            "store": False,
            "stream": True,
        }
        if _supports_temperature(self._model, self._temperature_unsupported):
            request["temperature"] = self._temperature
        _apply_reasoning_effort(request, self._reasoning_effort)
        with _provider_faults_declared():
            stream = await self._client.responses.create(**request)
        while True:
            # Stepped by hand so the fault wrapper covers the provider's work -- reading the
            # next event is another network read -- without also covering the consumer's,
            # which runs while this generator is suspended at the `yield`.
            with _provider_faults_declared():
                try:
                    event = await anext(stream)
                except StopAsyncIteration:
                    return
            if getattr(event, "type", "") != "response.output_text.delta":
                continue
            delta = getattr(event, "delta", None)
            if isinstance(delta, str) and delta:
                yield delta


class AnthropicLLMClient:
    """Anthropic Messages client configured solely from platform settings and environment.

    A sibling of ``OpenAILLMClient`` rather than a subclass of anything shared with it. The
    two satisfy the same two protocols and produce the same ``LLMResponse``; between those
    ends they translate different wire formats, and a common base class would only be a place
    for one provider's shape to leak into the other's.
    """

    def __init__(
        self,
        settings: Settings,
        agent_name: str,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        model_role: ModelRole | None = None,
        resolved_model: str | None = None,
        resolved_reasoning_effort: str | None = None,
        resolved_model_variable: str | None = None,
        resolved_routing_reason: str | None = None,
    ) -> None:
        """Bind one agent's policy and an optional request-scoped provider credential.

        The argument list matches ``OpenAILLMClient`` exactly so one factory can build either
        from the same call. The recovery arguments mean what they mean there: an already
        routed attempt executes the selection persisted before its model call.

        Unlike the Responses client, a configured model role is required. There is no
        pre-roles Anthropic variable for an agent to fall back to, and an agent that declares
        no role -- the GitHub publisher is the only one -- makes no model call at all.
        """
        try:
            agent_settings = settings.agents[agent_name]
        except KeyError as error:
            msg = f"unknown agent: {agent_name}"
            raise LLMAdapterError(msg) from error
        resolved_role = model_role or agent_settings.model_role
        if resolved_role is None:
            msg = f"agent {agent_name} declares no model role and makes no Messages API call"
            raise LLMAdapterError(msg)
        role_config = settings.model_config_for_role(
            resolved_role, platform=AgentPlatform.ANTHROPIC
        )
        self._model_role = resolved_role
        if resolved_model is not None and resolved_model.strip():
            self._model = resolved_model.strip()
            self._reasoning_effort = resolved_reasoning_effort
            self._model_variable = resolved_model_variable
            self._routing_reason = resolved_routing_reason or role_config.routing_reason
        else:
            self._model = role_config.model
            self._reasoning_effort = settings.reasoning_effort_for_agent(
                agent_name, platform=AgentPlatform.ANTHROPIC, model_role=model_role
            )
            self._model_variable = role_config.model_variable
            self._routing_reason = role_config.routing_reason
        # Read from the role rather than from the persisted routing decision, deliberately.
        # This is an operational bound on one call -- like the deadline below -- not part of
        # the selection an attempt was routed to, so a deployment that raises it gets the new
        # bound on a recovered attempt without that attempt's record becoming false.
        self._max_tokens = _required_max_tokens(role_config)
        # `AgentSettings.temperature` is deliberately not read. It stays in `agent.yaml` and
        # stays meaningful for the Responses client; the models this platform can be
        # configured with here reject `temperature`, `top_p` and `top_k` outright, and that is
        # a fact about the provider rather than about any one model identifier. Nothing below
        # sends a sampling parameter, so `LLMResponse` records that none was requested instead
        # of implying one was honoured.
        self._timeout_seconds = settings.timeout_seconds_for_agent(
            agent_name, model_role=resolved_role
        )
        # Read once against this client's fixed model, as on the Responses side.
        self._vision_capable = self._model in settings.declared_vision_capable()
        # The same budget the Responses client reads, for the same reason: a stream that
        # has not spoken has delivered nothing, whichever provider opened it.
        self._first_event_timeout_seconds = settings.first_event_timeout_seconds
        self._client = (
            client
            if client is not None
            else _create_anthropic_client(
                api_key=api_key,
                timeout_seconds=self._timeout_seconds,
                max_retries=agent_settings.max_retries,
            )
        )

    @property
    def model(self) -> str:
        """Return the configured model this client will call."""
        return self._model

    @property
    def model_role(self) -> ModelRole | None:
        """Return the routed role this client was built for, where one selected the model."""
        return self._model_role

    @property
    def reasoning_effort(self) -> str | None:
        """Return the normalized effort this client asks for, or nothing for the default."""
        return self._reasoning_effort

    @property
    def max_tokens(self) -> int:
        """Return the configured bound on one response. Required by the Messages API."""
        return self._max_tokens

    @property
    def provider(self) -> str:
        """Name the provider this client calls. Never a key, a base URL, or an account."""
        return _ANTHROPIC_PROVIDER

    @property
    def vision_capable(self) -> bool:
        """Whether the deployment declared this client's model able to read an image."""
        return self._vision_capable

    def _request(
        self, *, instructions: str, input_text: str, images: Sequence[ImageInput] = ()
    ) -> dict[str, Any]:
        """Build one Messages request from the selection fixed at construction."""
        request: dict[str, Any] = {
            "model": self._model,
            "system": instructions,
            # With no images the content stays the bare string it has always been; see the
            # Responses client for why that matters and where it is asserted.
            "messages": [
                {
                    "role": "user",
                    "content": (
                        _anthropic_content_with_images(input_text, images) if images else input_text
                    ),
                }
            ],
            "max_tokens": self._max_tokens,
            "timeout": self._timeout_seconds,
        }
        _apply_anthropic_thinking(request, self._reasoning_effort)
        return request

    async def respond(
        self, *, instructions: str, input_text: str, images: Sequence[ImageInput] = ()
    ) -> LLMResponse:
        """Call the Messages API using the model and operational limits fixed at construction."""
        if not instructions.strip() or not input_text.strip():
            msg = "instructions and input_text must not be empty"
            raise LLMAdapterError(msg, diagnostics=(msg,), failure_classification="empty_request")
        if images and not self._vision_capable:
            # The same second fence the Responses client raises, for the same reason.
            msg = (
                f"{self._model} is not declared able to read images; add it to "
                "MODEL_VISION_CAPABLE or run this feature on a model that is"
            )
            raise LLMAdapterError(
                msg, diagnostics=(msg,), failure_classification="model_not_vision_capable"
            )
        request = self._request(instructions=instructions, input_text=input_text, images=images)
        # `None` while the transport is a plain POST: no stream existed, so nothing about a
        # first event was measured, and `0` would claim a reading nobody took.
        reissues: int | None = None
        if self._max_tokens > _ANTHROPIC_STREAM_TRANSPORT_ABOVE_MAX_TOKENS:
            message, reissues = await self._streamed_message(request)
        else:
            with _provider_faults_declared():
                message = await self._client.messages.create(**request)
        return self._normalized(message, stream_reissues=reissues)

    async def _streamed_message(self, request: dict[str, Any]) -> tuple[Any, int]:
        """Issue one blocking Messages call over a stream; return it and the re-issue count.

        The transport exists for the reason ``_completed_response`` documents: a call that
        thinks for tens of minutes spends a plain POST with no bytes moving, which is exactly
        what an idle-connection timeout, a proxy or a load balancer reaps. It is also
        *required* above roughly 16K ``max_tokens``, which is why the choice here is made by
        the output bound rather than by the deadline.

        And it has the same failure this platform met on the Responses API: a stream the
        provider accepts and never writes to is indistinguishable from a model thinking, so
        without a budget it can only be noticed when the agent's whole deadline expires. This
        client had no such budget until now -- the SDK's ``get_final_message`` consumes the
        stream in one await, leaving nowhere to observe the first event -- which is why the
        loop below is hand-rolled rather than left as the one-liner it was.

        The events are deliberately not surfaced. This is the blocking call; a caller that
        wants tokens as they arrive already has ``stream``.
        """
        issued = 0
        while True:
            issued += 1
            message = await self._message_from_one_issue(request)
            if message is not None:
                return message, issued - 1
            if issued > _MAX_STREAM_REISSUES:
                msg = (
                    "Messages API accepted the request and sent no event within "
                    f"{self._first_event_timeout_seconds} seconds, on {issued} successive "
                    "issues of the same request"
                )
                raise LLMAdapterError(
                    msg, diagnostics=(msg,), failure_classification="stream_silent"
                )
            await asyncio.sleep(_STREAM_REISSUE_BACKOFF_SECONDS[issued - 1])

    async def _message_from_one_issue(self, request: dict[str, Any]) -> Any | None:
        """Open one stream and return the message it completes, or ``None`` if it never spoke.

        The first event is pulled by hand purely to time it, and then handed back to the SDK:
        ``get_final_message`` waits on whatever *remains* of the stream and returns the
        snapshot its iterator accumulates from every event, including the one already taken.
        So the message this returns is exactly the message the one-line form returned.

        ``None`` is returned only when nothing at all arrived inside the budget -- which is
        what makes the caller's re-issue safe, on the same reasoning as the Responses client:
        nothing was delivered, so nothing was decided.
        """
        stream: Any = None
        try:
            async with asyncio.timeout(self._first_event_timeout_seconds):
                with _provider_faults_declared():
                    # The manager's `__aenter__` is what issues the request, so it belongs
                    # inside the budget: headers that never arrive have delivered exactly as
                    # much as a first event that never arrives.
                    stream = await self._client.messages.stream(**request).__aenter__()
                    try:
                        await anext(stream)
                    except StopAsyncIteration:
                        msg = "Messages API stream ended before it sent any event"
                        raise LLMAdapterError(
                            msg, diagnostics=(msg,), failure_classification="incomplete_stream"
                        ) from None
        except TimeoutError:
            # Only this adapter's own budget lands here; a cancellation from above raises
            # `CancelledError`, which neither `_provider_faults_declared` nor `asyncio.timeout`
            # converts.
            await _closed_quietly(stream)
            return None
        try:
            with _provider_faults_declared():
                return await stream.get_final_message()
        finally:
            await _closed_quietly(stream)

    def _normalized(self, message: Any, *, stream_reissues: int | None = None) -> LLMResponse:
        """Turn one complete message into ``LLMResponse``, or into a classified fault.

        ``stop_reason`` is read before ``content`` on purpose. A refused request answers HTTP
        200 with an empty content array, so a reader that indexed the first block would throw
        ``IndexError`` inside this adapter and have a provider policy decision filed as a
        platform defect.
        """
        stop_reason = str(getattr(message, "stop_reason", "") or "")
        if stop_reason == "refusal":
            # Classified and handed to the existing retry policy, which owns every decision
            # about whether anything is attempted again. Nothing here retries or substitutes.
            msg = "the model declined to answer this request"
            raise LLMAdapterError(msg, diagnostics=(msg,), failure_classification="model_refusal")
        if stop_reason == "max_tokens":
            # Never reported as "the model returned no output text", and never returned as a
            # complete answer: half a plan that validates is worse than no plan at all. The
            # configured bound was too small for the work, which is a different thing from a
            # malformed response and has to be visible as one.
            #
            # The second diagnostic names what to change. The one-line form cost the 175-180
            # audit a database reconstruction to learn *which* bound had truncated *which*
            # role's call -- the reader of the durable failure record has none of this
            # client's construction context, and the variable name is exactly what a
            # deployment edits.
            msg = "the model response reached its configured max_tokens bound and is truncated"
            usage = getattr(message, "usage", None)
            produced = _optional_integer(getattr(usage, "output_tokens", None))
            role_label = self._model_role.value if self._model_role is not None else "unknown"
            # Effort leads and the bound is conditional, in that order on purpose: thinking
            # shares this bound with the answer, and a bound already at the model's output
            # ceiling cannot be raised at all. AB-Feature-181's operator was told to raise
            # ANTHROPIC_MEDIUM_CODING_MAX_TOKENS past 128000 -- claude-sonnet-5's ceiling --
            # which is advice nobody can follow. The adapter does not know the ceiling (and
            # a fault path must not gain a network dependency to describe itself), so the
            # raise is offered only as far as the model permits it.
            detail = (
                f"model={self._model} role={role_label} produced "
                f"{produced if produced is not None else 'an unrecorded number of'} tokens "
                f"against the configured bound of {self._max_tokens}"
                + (
                    "; lower the configured reasoning effort, or raise "
                    f"{self._model_variable.replace('_MODEL', '_MAX_TOKENS', 1)} "
                    "if the model's output ceiling permits a larger bound"
                    if self._model_variable
                    else ""
                )
            )
            raise LLMAdapterError(
                msg, diagnostics=(msg, detail), failure_classification="response_truncated"
            )
        output_text = "".join(
            text
            for block in (getattr(message, "content", None) or ())
            if getattr(block, "type", "") == "text"
            and isinstance(text := getattr(block, "text", None), str)
        )
        if not output_text.strip():
            msg = "Messages API returned no output text"
            raise LLMAdapterError(msg, diagnostics=(msg,), failure_classification="empty_response")
        usage = getattr(message, "usage", None)
        return _recorded(
            LLMResponse(
                response_id=str(getattr(message, "id", "")),
                model=str(getattr(message, "model", self._model)),
                output_text=output_text,
                input_tokens=_optional_integer(getattr(usage, "input_tokens", None)),
                output_tokens=_optional_integer(getattr(usage, "output_tokens", None)),
                stream_reissues=stream_reissues,
                provider=_ANTHROPIC_PROVIDER,
                # What this request actually asked for, after the one central capability check.
                reasoning_effort=self._reasoning_effort,
                model_role=self._model_role.value if self._model_role is not None else None,
                model_variable=self._model_variable,
                routing_reason=self._routing_reason,
            )
        )

    async def stream(self, *, instructions: str, input_text: str) -> AsyncIterator[str]:
        """Yield output text as the model produces it.

        Only the text deltas are surfaced, exactly as the Responses client does. The Messages
        stream carries other event types -- thinking, message envelopes, block boundaries --
        and a caller that forwarded whatever arrived would be putting the provider's internal
        event vocabulary on somebody's screen. Anything unrecognised is skipped.
        """
        if not instructions.strip() or not input_text.strip():
            msg = "instructions and input_text must not be empty"
            raise LLMAdapterError(msg, diagnostics=(msg,), failure_classification="empty_request")
        request = self._request(instructions=instructions, input_text=input_text)
        with _provider_faults_declared():
            stream = await self._client.messages.create(**request, stream=True)
        while True:
            # Stepped by hand so the fault wrapper covers the provider's work -- reading the
            # next event is another network read -- without also covering the consumer's,
            # which runs while this generator is suspended at the `yield`.
            with _provider_faults_declared():
                try:
                    event = await anext(stream)
                except StopAsyncIteration:
                    return
            if getattr(event, "type", "") != "content_block_delta":
                continue
            delta = getattr(event, "delta", None)
            if getattr(delta, "type", "") != "text_delta":
                continue
            text = getattr(delta, "text", None)
            if isinstance(text, str) and text:
                yield text


def _required_max_tokens(role_config: ModelConfig) -> int:
    """Return the configured output bound, refusing a role that has none.

    The Messages API rejects a request without ``max_tokens``, so this is the one genuinely
    new piece of configuration the second platform needs. Refused here, at construction,
    rather than as a provider 400 on the first live call.
    """
    max_tokens = role_config.max_tokens
    if max_tokens is None or max_tokens <= 0:
        msg = f"missing required configuration: {role_config.role.value} max_tokens"
        raise LLMAdapterError(msg, diagnostics=(msg,))
    return max_tokens


def _apply_anthropic_thinking(request: dict[str, Any], effort: str | None) -> None:
    """Ask for thinking and an effort level only when one is configured.

    Absent means the request is left exactly as it was before this existed, so a deployment
    that configures nothing keeps the provider's default.

    ``none`` is the one level in this platform's vocabulary the provider does not spell the
    same way, and it is handled explicitly: it means disabled thinking with no effort level at
    all. The two are mutually exclusive in this platform's configuration anyway -- a role
    resolves to one level, and ``none`` is one of them -- which matters because disabled
    thinking is accepted only at effort ``high`` or below and is refused above it.

    Raw thinking content is never returned and this platform never wanted it, so ``display``
    is not set. Adaptive thinking is on by default for the models this can be configured
    with; stating it costs nothing and makes the request self-describing.

    The level arriving here has already been through the one central capability check in
    ``configs.model_roles``: a level the deployment declared unsupported for this model is
    absent by the time this runs.
    """
    if not effort:
        return
    if effort == "none":
        request["thinking"] = {"type": "disabled"}
        return
    request["thinking"] = {"type": "adaptive"}
    request["output_config"] = {"effort": effort}


def _apply_reasoning_effort(request: dict[str, Any], effort: str | None) -> None:
    """Ask for a reasoning effort only when one is configured.

    Absent means the request is left exactly as it was before this existed, so a deployment
    that configures nothing keeps the provider's default. The level arriving here has already
    been through the one central capability check in ``configs.model_roles``: a level the
    deployment declared unsupported for this model is absent by the time this runs, and
    anything still present is sent for the provider to accept or reject. The accepted set
    differs per model -- `gpt-5.3-codex` takes `xhigh` but not `max` -- so a table of that
    here would be wrong the day it is written, which is why the declaration is configuration.
    """
    if effort:
        request["reasoning"] = {"effort": effort}


def _supports_temperature(model: str, declared_unsupported: Collection[str] = ()) -> bool:
    """Return whether the configured Responses model accepts ``temperature``.

    Two sources answer one question. The gpt-5 Responses family is known here; everything
    this file cannot know arrives as the deployment's MODEL_TEMPERATURE_UNSUPPORTED
    declaration. AB-Feature-210 is why the second source exists: a prefix guess sent
    `temperature` to a family this code had never seen (gpt-6), and the run died on its
    first provider call with a deterministic 400. A new family is declared in
    configuration, never added as another prefix.
    """
    name = model.strip()
    if name in declared_unsupported:
        return False
    return not name.lower().startswith("gpt-5")


# Above this deadline a blocking call is issued over a stream instead of a plain POST. See
# ``_completed_response`` for why. Five minutes: comfortably above every short-deadline agent,
# so none of their transports change, and comfortably below the deadlines that really do run
# long. Expressed as a deadline rather than an agent list so a deployment that raises one
# agent's timeout gets the transport that deadline implies, without editing this file.
_STREAM_TRANSPORT_ABOVE_SECONDS = 300


# The first-event budget itself is `Settings.first_event_timeout_seconds`, read per client at
# construction. It exists because of AB-Feature-215: two calls -- one coding, one in-attempt
# repair, issued a second apart -- were accepted and sent no byte at all. Each burned the
# engineer's full 1800-second deadline before failing, and the replacement work took 34
# seconds. A deadline that can only notice silence after thirty minutes converts a dead socket
# into a lost attempt.
#
# How many times a stream that never spoke is issued again. The re-issue is safe *only*
# because nothing was delivered -- see `_stream_that_spoke` -- so this is not a general retry
# allowance and must never be widened into one. Two, because the platform's own fault
# allowance sits above this and is the right place for a provider that is properly unwell.
_MAX_STREAM_REISSUES = 2
# Paced to distinguish a lost connection from a provider under load, without becoming a wait
# worth noticing: the whole point of the budget above is that a silent call fails fast.
_STREAM_REISSUE_BACKOFF_SECONDS = (5.0, 15.0)


# Above this output bound a Messages call is issued over a stream. Deliberately expressed in
# tokens rather than reusing the deadline above, which answers a different question: streaming
# is *required* by the Messages API above roughly 16K `max_tokens`, and the coding role's bound
# exceeds it. A threshold that means "this call will be slow" cannot also mean "this call is
# too large to answer in one response".
_ANTHROPIC_STREAM_TRANSPORT_ABOVE_MAX_TOKENS = 16_000


# Cancellation is a decision this platform made, not a fault the provider had, and it is
# routed by its own type all the way up. Nothing here may absorb it.
_NOT_A_PROVIDER_FAULT = (CancellationRequested, asyncio.CancelledError, LLMAdapterError)


# The SDK exception names whose answer is the same on every retry: a 4xx that is not 408 or
# 429. These are the names both provider SDKs (OpenAI, Anthropic) give their HTTP client
# errors; timeouts, connection drops, 429s and 5xx are deliberately absent, because those
# are the weather the retry allowances exist for. The single source for the platform's
# deterministic-fault predicate -- `workflows.feature_workflow` builds its classification
# set from this rather than keeping a second list that could drift.
DETERMINISTIC_PROVIDER_SDK_ERRORS = frozenset(
    {
        "BadRequestError",
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "UnprocessableEntityError",
        "ConflictError",
    }
)

# Every classification that means the provider *answered* and would answer the same way
# again. The two names below are this adapter's own; the 4xx names come from the set above,
# so the predicate and the messages cannot drift apart. `workflows.feature_workflow` reads
# this rather than keeping its own copy, and so does the Engineer's in-attempt repair loop --
# three callers, one definition, which is the property AB-Feature-215 needed and lacked.
#
# Note what is deliberately absent: `incomplete_stream`, `empty_response`, `stream_silent`
# and the malformed-response classifications. Those are the transport or a bad roll of the
# dice, and asking again is exactly the right response to them.
DETERMINISTIC_PROVIDER_CLASSIFICATIONS = frozenset(
    {
        "model_refusal",
        "response_truncated",
        *DETERMINISTIC_PROVIDER_SDK_ERRORS,
    }
)


# Every classification this adapter raises *itself*, having read something the provider did
# send. A malformed body, an edit whose `find` matched nothing, a refusal, a truncation: bytes
# arrived and this module judged them. None of these is the transport, however retryable some
# of them are -- `is_transport_fault` is a narrower question than "is another attempt worth
# making", and conflating the two is what the first draft of this predicate got wrong.
#
# `stream_silent` is deliberately absent: it is this adapter's own verdict *about* the
# transport, reached because nothing arrived at all.
_ADAPTER_JUDGED_CLASSIFICATIONS = frozenset(
    {
        "empty_request",
        "empty_response",
        "incomplete_stream",
        "malformed_coding_response",
        "coding_edit_mismatch",
        "model_refusal",
        "response_truncated",
        # This adapter refusing to send images to a model the deployment has not declared
        # able to read them. Nothing was sent, so no bytes were judged -- but it is emphatically
        # not the transport either, and retrying it decides the same thing again. It belongs
        # here with the other refusals this module makes for itself, next to `empty_request`,
        # which is the same shape of decision: a request this adapter will not issue.
        "model_not_vision_capable",
    }
)


def is_transport_fault(error: BaseException) -> bool:
    """Whether one failure is the transport failing rather than anything being read.

    The narrow question, asked of one error and not of a cause chain: callers that need the
    chain walked (the child-workstream loop) already do that themselves and then ask this.
    It is *not* the same question as "is a retry worth it" -- a malformed response is worth
    asking again and is not this. Only a caller deciding whether an attempt has a verdict to
    report should be asking this one.

    Three answers, in order: this adapter's own transport verdict; anything it judged from
    bytes that did arrive; otherwise an SDK exception name, which is transport unless it is
    one of the deterministic 4xx.

    An unclassified ``LLMAdapterError`` answers ``False`` on purpose. Those are this adapter's
    local decisions, and a decision is not weather.
    """
    if not isinstance(error, LLMAdapterError):
        return False
    classification = error.failure_classification
    if classification is None:
        return False
    if classification == "stream_silent":
        return True
    if classification in _ADAPTER_JUDGED_CLASSIFICATIONS:
        return False
    return classification not in DETERMINISTIC_PROVIDER_SDK_ERRORS


async def _closed_quietly(stream: Any) -> None:
    """Release a stream this adapter is abandoning, without ever raising in its place.

    The fault being handled is the interesting one. A provider whose stream went silent is
    not owed the assumption that closing it works, and a failure to close must not displace
    the sentence the caller is about to write about the silence.
    """
    if stream is None:
        return
    closer = getattr(stream, "close", None)
    if closer is None:
        return
    try:
        result = closer()
        if hasattr(result, "__await__"):
            await result
    except Exception:  # noqa: BLE001 - see the docstring: never displace the fault in hand
        return


# How much of the provider's own sentence a deterministic failure may carry. AB-Feature-172's
# fix was named in the first fifty characters ("anthropic-workspace-id is required..."); the
# bound exists because provider messages can embed request payloads, not because they run long.
_PROVIDER_DETAIL_MAX_CHARACTERS = 500


def _sanitized_provider_detail(error: BaseException) -> str:
    """One flattened, redacted, bounded line of the provider's own words.

    Deterministic failures only. A 4xx names its own fix -- 172's operator learned
    ``anthropic-workspace-id is required`` from a manual curl because this text was
    discarded -- while a timeout's text is transport noise that can quote a prompt.
    The redaction is the one process output already passes through: provider-shaped
    tokens, credential URLs and every credential-bearing environment value.
    """
    text = " ".join(str(error).split())
    text = redact_output(redact_source_credentials(text))
    if len(text) > _PROVIDER_DETAIL_MAX_CHARACTERS:
        text = text[:_PROVIDER_DETAIL_MAX_CHARACTERS] + " [truncated]"
    return text.strip()


@contextmanager
def _provider_faults_declared() -> Iterator[None]:
    """Re-raise anything the provider SDK throws as this adapter's own fault type.

    Every retry allowance in the platform is keyed on ``LLMAdapterError``: the child
    workstream loop grants four attempts with growing backoff for one, and the operation
    journal reserves attempts against it. An exception class belonging to the SDK matches
    none of that, so a dropped connection did not earn the attempts that were sitting
    reserved for it -- it ended the workstream instead.

    AB-Feature-121's console is the case. Its reviewer call raised ``APIConnectionError``,
    the journal recorded ``run_reviewer`` as ``failed_retryable`` on attempt 1 of 3, and the
    repository was closed at ``retry_count`` 1 with two journal attempts and its whole child
    budget unspent -- then filed under the previous attempt's ``validation_source_failure``,
    sending its reader to look at source code for a network fault. AB-Feature-119's backend
    ended the same way.

    Wrapping at this boundary rather than naming SDK classes upstream is deliberate: the
    platform must not have to enumerate a provider's exception hierarchy, and a client
    library is free to add to it in a patch release.

    For a transient fault, the provider's own message is not carried. It is built from
    request material and can quote a prompt or a header; the type name is a public symbol
    of an installed library and is what identifies the fault. The original stays reachable
    as ``__cause__`` for the diagnostics walk that looks there.

    A deterministic 4xx is the exception, and deliberately so: its message names the exact
    fix (AB-Feature-172 died on ``anthropic-workspace-id is required...`` and recorded only
    ``BadRequestError``), it returns identically on every retry so nobody will see a better
    one, and it is carried redacted and bounded under the same contract
    ``SourceValidationError`` already defines for repository output.
    """
    try:
        yield
    except _NOT_A_PROVIDER_FAULT:
        raise
    except Exception as error:
        name = type(error).__name__
        msg = f"the model provider call failed ({name})"
        diagnostics = [msg]
        if name in DETERMINISTIC_PROVIDER_SDK_ERRORS:
            detail = _sanitized_provider_detail(error)
            if detail:
                diagnostics.append(f"The provider's answer, redacted: {detail}")
        raise LLMAdapterError(
            msg,
            diagnostics=tuple(diagnostics),
            failure_classification=name,
        ) from error


class ResponsesCodingExecutor:
    """Apply JSON-described code updates returned by an injected configured LLM client."""

    def __init__(
        self, llm_client: LLMClient, *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    ) -> None:
        """Inject the configured LLM boundary instead of coupling to LangGraph state."""
        self._llm_client = llm_client
        self._max_file_bytes = max_file_bytes

    def _resolve_updates(
        self, root: Path, updates: tuple[_FileUpdate, ...]
    ) -> tuple[_ResolvedUpdate, ...]:
        """Turn every targeted edit into the complete content it produces, before any write."""
        return tuple(
            _ResolvedUpdate(
                path=update.path,
                content=(
                    update.content
                    if update.content is not None
                    else _resolve_file_edits(root, update, max_file_bytes=self._max_file_bytes)
                ),
            )
            for update in updates
        )

    async def execute(
        self,
        *,
        workspace_root: PathLike,
        instructions: str,
        input_text: str,
        cancellation_token: CancellationToken | None = None,
        operation_executor: ExternalOperationExecutor | None = None,
        images: Sequence[ImageInput] = (),
    ) -> CodingExecutionResult:
        """Parse, validate, and atomically write a coding response inside the workspace only."""
        root = resolve_workspace_root(workspace_root)
        # Dropped for a model that cannot see, rather than sent and refused: a provider error
        # here would cost the attempt, and a coding call that reads the design as text is
        # exactly what this platform did until now.
        shown = tuple(images) if self._llm_client.vision_capable else ()

        async def execute_once() -> tuple[CodingExecutionResult, OperationResult]:
            if cancellation_token is not None:
                response = await await_cancellable(
                    self._llm_client.respond(
                        instructions=instructions, input_text=input_text, images=shown
                    ),
                    cancellation_token,
                )
                await cancellation_token.raise_if_cancelled()
            else:
                response = await self._llm_client.respond(
                    instructions=instructions, input_text=input_text, images=shown
                )
            try:
                summary, parsed = _parse_coding_response(response.output_text)
                updates = self._resolve_updates(root, parsed)
                _validate_updates(root, updates, max_file_bytes=self._max_file_bytes)
            except LLMAdapterError as error:
                # The planner and the reviewer both repair a rejected response rather than
                # ending the run. Without the same bounded attempt here, a single malformed
                # response destroyed a workstream before it had written anything at all.
                repair = await self._llm_client.respond(
                    instructions=(
                        f"{instructions}\n\n"
                        "Your previous response was rejected before any file was written. "
                        f"The reason was:\n{error}\n\n"
                        "Return the same change again as one valid JSON object with exactly "
                        "the keys `summary` and `files`, and no Markdown, prose or code "
                        "fence around it. Every file entry needs a relative `path` and "
                        "either the complete `content` of a new file or, for a file that "
                        "already exists, an `edits` list whose every `find` is copied "
                        "verbatim from the current file and matches exactly one place."
                    ),
                    input_text=input_text,
                )
                if cancellation_token is not None:
                    await cancellation_token.raise_if_cancelled()
                response = repair
                summary, parsed = _parse_coding_response(response.output_text)
                updates = self._resolve_updates(root, parsed)
                _validate_updates(root, updates, max_file_bytes=self._max_file_bytes)
            if cancellation_token is not None:
                await cancellation_token.raise_if_cancelled()

            async def apply_updates() -> tuple[tuple[Path, ...], OperationResult]:
                file_tools = WorkspaceFileTools(root, max_file_bytes=self._max_file_bytes)
                modified = tuple(
                    file_tools.write_file(update.path, update.content) for update in updates
                )
                return modified, OperationResult(
                    payload={
                        "modified_files": [path.as_posix() for path in modified],
                        "modified_file_fingerprints": _workspace_file_fingerprints(root, modified),
                    }
                )

            if operation_executor is None:
                modified_files, _ = await apply_updates()
            else:
                write_operation = await operation_executor.run(
                    operation_type=ExternalOperationType.WRITE_FILE_CHANGES,
                    logical_step="apply_coding_updates",
                    safe_input={
                        "workspace_path": str(root),
                        "file_fingerprints": {
                            update.path: _sha256(update.content) for update in updates
                        },
                    },
                    action=apply_updates,
                )
                if write_operation.reused:
                    payload = write_operation.operation.result_payload or {}
                    modified_files = _payload_paths(payload, "modified_files")
                    if not _workspace_files_match_persisted_fingerprints(
                        root, modified_files, payload
                    ):
                        # The journal exists to stop a side effect happening twice, not to
                        # assert it still exists. Writing the same content again is exactly
                        # idempotent, and refusing instead abandoned a workstream over files
                        # the platform itself had removed from the workspace.
                        modified_files, _ = await apply_updates()
                else:
                    modified_files = cast(tuple[Path, ...], write_operation.value)
            result = CodingExecutionResult(
                response_id=response.response_id,
                model=response.model,
                summary=summary,
                modified_files=modified_files,
                provider=response.provider,
                reasoning_effort=response.reasoning_effort,
                model_role=response.model_role,
                model_variable=response.model_variable,
                routing_reason=response.routing_reason,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                stream_reissues=response.stream_reissues,
            )
            return result, OperationResult(
                external_reference=response.response_id,
                payload={
                    "response_id": response.response_id,
                    "model": response.model,
                    # Journalled beside the model for the same reason the model is: a resumed
                    # attempt must report what actually executed it, not what the process that
                    # resumed it would have chosen.
                    "provider": response.provider,
                    "reasoning_effort": response.reasoning_effort,
                    "model_role": response.model_role,
                    "model_variable": response.model_variable,
                    "routing_reason": response.routing_reason,
                    # How many times this call's request had to be issued again because the
                    # stream never spoke. On the row rather than only in the response,
                    # because this is the one measurement of the first-event budget anything
                    # above the adapter can see: a re-issue that then succeeded leaves no
                    # other trace at all. Absent for a non-streamed transport, where the
                    # question does not arise.
                    "stream_reissues": response.stream_reissues,
                    # "usage", not "tokens": the journal's credential screen rejects any
                    # metadata key carrying a credential-shaped marker, and "token" is one.
                    "usage_input": response.input_tokens,
                    "usage_output": response.output_tokens,
                    "summary": summary,
                    "modified_files": [path.as_posix() for path in modified_files],
                    "modified_file_fingerprints": _workspace_file_fingerprints(
                        root, modified_files
                    ),
                },
            )

        if operation_executor is None:
            result, _ = await execute_once()
            return result
        child_attempt = _coding_child_attempt(input_text)
        effective_input_sha256 = _coding_effective_input_sha256(input_text)
        safe_input: dict[str, Any] = {
            "workspace_path": str(root),
            "instructions_sha256": _sha256(instructions),
            "input_sha256": _sha256(input_text),
        }
        if child_attempt is not None:
            safe_input["child_attempt"] = child_attempt
        if effective_input_sha256 is not None:
            safe_input["effective_input_sha256"] = effective_input_sha256
        journaled = await operation_executor.run(
            operation_type=ExternalOperationType.RUN_CODING_EXECUTOR,
            logical_step="coding_and_file_changes",
            safe_input=safe_input,
            # One durable child attempt owns one coding effect. Prompt envelopes are rebuilt
            # after a process crash and carry fresh timestamps, while the claimed attempt,
            # contract and feedback do not. Keying a completed operation on those regenerated
            # bytes made recovery call the Engineer again for work the journal already held.
            # A materially different instruction must be allocated a new child attempt; using
            # the existing attempt reconciles its completed effect instead of creating another.
            idempotency_input=(
                {
                    "workspace_path": str(root),
                    "child_attempt": child_attempt,
                    "effective_input_sha256": effective_input_sha256,
                }
                if child_attempt is not None and effective_input_sha256 is not None
                else None
            ),
            action=execute_once,
            attempt_metadata={
                "provider": "openai_responses",
                "workspace_path": str(root),
                **({"child_attempt": child_attempt} if child_attempt is not None else {}),
            },
            # Workspace-local, as the recovery service already classifies it, so an interrupted
            # run leaves nothing outside this checkout to reconcile. Without a replay budget the
            # first interruption is terminal on this run and on every resume, and coding gates
            # everything downstream: -076's backend was killed by an API restart mid-attempt and
            # the feature sat in `running_child_workflows` with nothing able to move it.
            max_attempts=3,
        )
        if not journaled.reused:
            return cast(CodingExecutionResult, journaled.value)
        payload = journaled.operation.result_payload or {}
        modified_files = _payload_paths(payload, "modified_files")
        if not _workspace_files_match_persisted_fingerprints(root, modified_files, payload):
            msg = (
                "completed coding operation cannot be reused because the workspace content "
                "does not match its persisted fingerprints"
            )
            raise LLMAdapterError(msg)
        return CodingExecutionResult(
            response_id=str(
                payload.get("response_id", journaled.operation.external_reference or "")
            ),
            model=str(payload.get("model", "recovered-coding-executor")),
            summary=str(payload.get("summary", "Recovered completed coding execution.")),
            modified_files=modified_files,
            provider=str(payload.get("provider", "")),
            reasoning_effort=(
                str(payload["reasoning_effort"])
                if isinstance(payload.get("reasoning_effort"), str)
                else None
            ),
            model_role=(
                str(payload["model_role"]) if isinstance(payload.get("model_role"), str) else None
            ),
            model_variable=(
                str(payload["model_variable"])
                if isinstance(payload.get("model_variable"), str)
                else None
            ),
            routing_reason=(
                str(payload["routing_reason"])
                if isinstance(payload.get("routing_reason"), str)
                else None
            ),
            input_tokens=_optional_integer(payload.get("usage_input")),
            output_tokens=_optional_integer(payload.get("usage_output")),
        )


class MockCodingExecutor:
    """Deterministic coding executor for tests that uses the same workspace safety boundary."""

    def __init__(
        self,
        *,
        summary: str = "Mock coding execution completed.",
        file_updates: Mapping[str, str] | None = None,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        """Configure the result and optional workspace file updates produced by the mock."""
        if not summary.strip():
            msg = "summary must not be empty"
            raise LLMAdapterError(msg)
        self._summary = summary
        self._updates = tuple(
            _ResolvedUpdate(path=path, content=content)
            for path, content in (file_updates or {}).items()
        )
        self._max_file_bytes = max_file_bytes

    async def execute(
        self,
        *,
        workspace_root: PathLike,
        instructions: str,
        input_text: str,
        cancellation_token: CancellationToken | None = None,
        operation_executor: ExternalOperationExecutor | None = None,
        images: Sequence[ImageInput] = (),
    ) -> CodingExecutionResult:
        """Apply configured deterministic updates without a network call or state mutation."""
        del operation_executor, images
        if not instructions.strip() or not input_text.strip():
            msg = "instructions and input_text must not be empty"
            raise LLMAdapterError(msg)
        root = resolve_workspace_root(workspace_root)
        if cancellation_token is not None:
            await cancellation_token.raise_if_cancelled()
        _validate_updates(root, self._updates, max_file_bytes=self._max_file_bytes)
        if cancellation_token is not None:
            await cancellation_token.raise_if_cancelled()
        file_tools = WorkspaceFileTools(root, max_file_bytes=self._max_file_bytes)
        modified_files = tuple(
            file_tools.write_file(update.path, update.content) for update in self._updates
        )
        return CodingExecutionResult(
            response_id="mock-response",
            model="mock-coding-executor",
            summary=self._summary,
            modified_files=modified_files,
        )


def _create_openai_client(*, api_key: str | None, timeout_seconds: int, max_retries: int) -> Any:
    """Build the SDK client from an ephemeral key or the process environment fallback."""
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        msg = "an OpenAI API key must be supplied in X-OpenAI-Api-Key or OPENAI_API_KEY"
        raise LLMAdapterError(msg, diagnostics=(msg,))
    openai_module = importlib.import_module("openai")
    return openai_module.AsyncOpenAI(
        api_key=resolved_api_key,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )


def _create_anthropic_client(*, api_key: str | None, timeout_seconds: int, max_retries: int) -> Any:
    """Build the SDK client from an ephemeral key or the process environment fallback.

    Imported lazily for the reason the Responses client is: a deployment or a test that never
    reaches this provider should not need the package installed, and an injected ``client=``
    bypasses this entirely.

    The message names this provider's header and variable and no other. A deployment reading
    it at three in the morning is being told exactly which of two credentials is missing.
    """
    resolved_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_api_key:
        msg = "an Anthropic API key must be supplied in X-Anthropic-Api-Key or ANTHROPIC_API_KEY"
        raise LLMAdapterError(msg, diagnostics=(msg,))
    anthropic_module = importlib.import_module("anthropic")
    # An identity-linked API key is not bound to one workspace, so the Messages API rejects
    # its requests with a 400 unless every one of them names the workspace it acts in. The
    # deployment declares that workspace once; a workspace-scoped key leaves this unset and
    # the header is never sent.
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    default_headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    return anthropic_module.AsyncAnthropic(
        api_key=resolved_api_key,
        timeout=timeout_seconds,
        max_retries=max_retries,
        default_headers=default_headers,
    )


def _openai_input_with_images(
    input_text: str, images: Sequence[ImageInput]
) -> list[dict[str, Any]]:
    """Build the Responses `input` array for a request that carries images.

    Images before the text, in both providers' builders. It matches Anthropic's published
    guidance, and it leaves the instruction the model acts on as the last thing it reads --
    which is the same reason every prompt in this repository puts its contract at the end.
    """
    return [
        {
            "role": "user",
            "content": [
                *(
                    {
                        "type": "input_image",
                        "image_url": f"data:{image.media_type};base64,{image.data}",
                    }
                    for image in images
                ),
                {"type": "input_text", "text": input_text},
            ],
        }
    ]


def _anthropic_content_with_images(
    input_text: str, images: Sequence[ImageInput]
) -> list[dict[str, Any]]:
    """Build the Messages user content for a request that carries images."""
    return [
        *(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": image.data,
                },
            }
            for image in images
        ),
        {"type": "text", "text": input_text},
    ]


def llm_client_for(
    platform: AgentPlatform | str,
    settings: Settings,
    agent_name: str,
    *,
    api_key: str | None = None,
    client: Any | None = None,
    model_role: ModelRole | None = None,
    resolved_model: str | None = None,
    resolved_reasoning_effort: str | None = None,
    resolved_model_variable: str | None = None,
    resolved_routing_reason: str | None = None,
) -> LLMClient:
    """Build the model boundary one feature's chosen platform requires.

    One factory, called everywhere a client is constructed. The alternative -- a
    ``if platform == "anthropic"`` at each of the runtime's construction sites -- is how five
    components come to disagree about one decision, which is the argument
    ``ModelConfigService`` already makes for models.
    """
    resolved_platform = platform if isinstance(platform, AgentPlatform) else AgentPlatform(platform)
    constructor = (
        AnthropicLLMClient if resolved_platform is AgentPlatform.ANTHROPIC else OpenAILLMClient
    )
    return constructor(
        settings,
        agent_name,
        api_key=api_key,
        client=client,
        model_role=model_role,
        resolved_model=resolved_model,
        resolved_reasoning_effort=resolved_reasoning_effort,
        resolved_model_variable=resolved_model_variable,
        resolved_routing_reason=resolved_routing_reason,
    )


def _parse_coding_response(output_text: str) -> tuple[str, tuple[_FileUpdate, ...]]:
    """Parse the coding response contract: summary plus complete replacement file contents."""
    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as error:
        msg = "coding response must be valid JSON"
        raise LLMAdapterError(msg, failure_classification="malformed_coding_response") from error
    if not isinstance(payload, dict):
        msg = "coding response must be a JSON object"
        raise LLMAdapterError(msg, failure_classification="malformed_coding_response")
    summary = payload.get("summary")
    files = payload.get("files")
    if not isinstance(summary, str) or not summary.strip():
        msg = "coding response summary must be a non-empty string"
        raise LLMAdapterError(msg, failure_classification="malformed_coding_response")
    if not isinstance(files, list):
        msg = "coding response files must be a list"
        raise LLMAdapterError(msg, failure_classification="malformed_coding_response")

    updates: list[_FileUpdate] = []
    for file_data in files:
        if not isinstance(file_data, dict):
            msg = "coding response file entries must be objects"
            raise LLMAdapterError(msg, failure_classification="malformed_coding_response")
        path = file_data.get("path")
        content = file_data.get("content")
        edits = file_data.get("edits")
        if not isinstance(path, str) or not path.strip():
            msg = "coding response files require a non-empty path"
            raise LLMAdapterError(msg, failure_classification="malformed_coding_response")
        if edits is not None:
            if content is not None:
                msg = f"coding response file sets both content and edits: {path}"
                raise LLMAdapterError(msg, failure_classification="malformed_coding_response")
            updates.append(_FileUpdate(path=path, edits=_parse_file_edits(path, edits)))
            continue
        if not isinstance(content, str):
            msg = "coding response files require non-empty path and string content"
            raise LLMAdapterError(msg, failure_classification="malformed_coding_response")
        updates.append(_FileUpdate(path=path, content=content))
    return summary, tuple(updates)


def _parse_file_edits(path: str, edits: Any) -> tuple[_FileEdit, ...]:
    """Parse the targeted-edit form, rejecting anything that could silently match nothing."""
    if not isinstance(edits, list) or not edits:
        msg = f"coding response edits must be a non-empty list: {path}"
        raise LLMAdapterError(msg, failure_classification="malformed_coding_response")
    parsed: list[_FileEdit] = []
    for edit in edits:
        if not isinstance(edit, dict):
            msg = f"coding response edit entries must be objects: {path}"
            raise LLMAdapterError(msg, failure_classification="malformed_coding_response")
        find = edit.get("find")
        replace = edit.get("replace")
        if not isinstance(find, str) or not find:
            msg = f"coding response edit requires a non-empty find string: {path}"
            raise LLMAdapterError(msg, failure_classification="malformed_coding_response")
        if not isinstance(replace, str):
            msg = f"coding response edit requires a string replace value: {path}"
            raise LLMAdapterError(msg, failure_classification="malformed_coding_response")
        parsed.append(_FileEdit(find=find, replace=replace))
    return tuple(parsed)


def _edit_excerpt(content: str) -> str:
    """Quote the head of a file for a repair prompt, bounded and credential-free."""
    excerpt = redact_source_credentials(content[:_EDIT_EXCERPT_MAX_CHARACTERS])
    if len(content) > _EDIT_EXCERPT_MAX_CHARACTERS:
        excerpt = f"{excerpt}\n[file truncated]"
    return excerpt


def _resolve_file_edits(workspace_root: Path, update: _FileUpdate, *, max_file_bytes: int) -> str:
    """Apply targeted edits to the file already in the workspace and return the new content.

    Every ``find`` must match exactly once. An absent or ambiguous snippet is an error rather
    than a guess, because the repair prompt can then name precisely what did not match, and
    the file on disk is left untouched until the whole response validates.
    """
    file_tools = WorkspaceFileTools(workspace_root, max_file_bytes=max_file_bytes)
    try:
        content = file_tools.read_file(update.path)
    except (FileNotFoundError, OSError, ValueError) as error:
        msg = (
            f"coding response edits a file that is not in the workspace: {update.path}. "
            "Return complete content to create a new file."
        )
        raise LLMAdapterError(msg, failure_classification="coding_edit_mismatch") from error
    for edit in update.edits:
        occurrences = content.count(edit.find)
        if occurrences == 0:
            # The current text is quoted back, because a repair told only that its snippet
            # was wrong has no way to produce a better one and loses the whole attempt.
            msg = (
                f"coding response edit does not match the file: {update.path}. The `find` text "
                "must be copied exactly from the current file, including indentation. The file "
                f"currently begins:\n{_edit_excerpt(content)}"
            )
            raise LLMAdapterError(msg, failure_classification="coding_edit_mismatch")
        if occurrences > 1:
            msg = (
                f"coding response edit matches {occurrences} places in {update.path}. Extend "
                "`find` with surrounding lines until it identifies exactly one location."
            )
            raise LLMAdapterError(msg, failure_classification="coding_edit_mismatch")
        content = content.replace(edit.find, edit.replace, 1)
    return content


def _validate_updates(
    workspace_root: Path, updates: tuple[_ResolvedUpdate, ...], *, max_file_bytes: int
) -> None:
    """Validate every target before any write so no malformed response is partially applied."""
    resolved_paths: set[Path] = set()
    for update in updates:
        if len(update.content.encode("utf-8")) > max_file_bytes:
            msg = f"coding response file exceeds configured size limit: {update.path}"
            raise LLMAdapterError(msg, failure_classification="malformed_coding_response")
        resolved_path = resolve_workspace_path(workspace_root, update.path)
        if resolved_path in resolved_paths:
            msg = f"coding response updates a file more than once: {update.path}"
            raise LLMAdapterError(msg, failure_classification="malformed_coding_response")
        resolved_paths.add(resolved_path)


def _optional_integer(value: Any) -> int | None:
    """Normalize optional SDK token counts without accepting arbitrary non-integer values."""
    return value if isinstance(value, int) else None


def _coding_child_attempt(input_text: str) -> int | None:
    """Read the durable child-attempt identity from the structured Engineer input.

    Older and non-feature callers do not carry a task plan with this metadata and retain the
    existing full-prompt fingerprint. Booleans are excluded even though they are integers in
    Python: ``true`` is not a valid attempt identity.
    """
    try:
        payload = json.loads(input_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    task_plan = payload.get("task_plan")
    if not isinstance(task_plan, dict):
        return None
    metadata = task_plan.get("metadata")
    if not isinstance(metadata, dict):
        return None
    attempt = metadata.get("attempt")
    return (
        attempt
        if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= 0
        else None
    )


def _coding_effective_input_sha256(input_text: str) -> str | None:
    """Fingerprint stable Engineer instructions without post-effect workspace context.

    A task plan is rebuilt after a crash and therefore gets a new envelope timestamp. The
    repository context is also rebuilt, but after a completed Engineer call it contains the
    files that call just wrote. Neither difference is a new instruction. The plan, previous
    review, and execution context are the semantic inputs that can legitimately authorize a
    new coding effect; their schema and contract versions remain in the canonical payload.
    """
    try:
        payload = json.loads(input_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("task_plan"), dict):
        return None
    task_plan = dict(payload["task_plan"])
    task_plan.pop("timestamp", None)
    metadata = task_plan.get("metadata")
    if isinstance(metadata, dict):
        stable_metadata = dict(metadata)
        # These describe the checkout immediately before the effect. On recovery the same
        # checkout contains that effect, so both values necessarily differ even though the
        # claimed attempt, contract, repair plan and reviewer feedback are unchanged.
        stable_metadata.pop("repository_revision", None)
        stable_metadata.pop("preflight_result", None)
        task_plan["metadata"] = stable_metadata
    execution_context = payload.get("execution_context")
    stable_execution_context: Any
    if isinstance(execution_context, dict):
        stable_execution_context = dict(execution_context)
        stable_execution_context.pop("repository_revision", None)
        stable_execution_context.pop("preflight_result", None)
    else:
        stable_execution_context = execution_context
    stable_payload = {
        "task_plan": task_plan,
        "prior_review": payload.get("prior_review"),
        "execution_context": stable_execution_context,
    }
    encoded = json.dumps(stable_payload, sort_keys=True, separators=(",", ":"), default=str)
    return _sha256(encoded)


def _sha256(value: str) -> str:
    """Fingerprint prompt material without persisting it in the operation journal."""
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _payload_paths(payload: dict[str, Any], key: str) -> tuple[Path, ...]:
    """Parse only a persisted list of relative file strings from a completed operation."""
    values = payload.get(key)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        msg = "completed operation has invalid modified-file data"
        raise LLMAdapterError(msg)
    return tuple(Path(value) for value in values)


def _workspace_file_fingerprints(workspace_root: Path, paths: tuple[Path, ...]) -> dict[str, str]:
    """Hash exact applied bytes so a later write cannot impersonate an earlier operation."""
    import hashlib

    return {
        path.as_posix(): hashlib.sha256(
            resolve_workspace_path(workspace_root, path).read_bytes()
        ).hexdigest()
        for path in paths
    }


def _workspace_files_match_persisted_fingerprints(
    workspace_root: Path,
    paths: tuple[Path, ...],
    payload: dict[str, Any],
) -> bool:
    """Require complete hash evidence before attributing current bytes to a replay."""
    expected = payload.get("modified_file_fingerprints")
    if not isinstance(expected, dict) or set(expected) != {path.as_posix() for path in paths}:
        return False
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in expected.items()):
        return False
    try:
        return _workspace_file_fingerprints(workspace_root, paths) == expected
    except (OSError, ValueError):
        return False


def completed_coding_operation_matches_workspace(
    operation: ExternalOperation, workspace_root: PathLike
) -> bool:
    """Return whether a completed coding receipt still describes the exact workspace bytes.

    The live child executor asks this before resetting a retry checkout. If a worker died after
    the coding operation succeeded, those files are the output to review, not rejected history
    to erase before invoking the Engineer again.
    """
    payload = operation.result_payload or {}
    try:
        paths = _payload_paths(payload, "modified_files")
        root = resolve_workspace_root(workspace_root)
    except (LLMAdapterError, OSError, ValueError):
        return False
    return _workspace_files_match_persisted_fingerprints(root, paths, payload)
