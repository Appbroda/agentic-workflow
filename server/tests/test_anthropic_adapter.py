"""The Messages API client, and the four ways it can be wrong without anyone noticing.

Every test here exists because it is a defect this platform could plausibly ship. Two of them
are invisible without an explicit assertion, because the failure is a provider-side 400 that
only happens against a real key: a sampling parameter Claude rejects, and a `max_tokens` that
was never sent. Two more are failures that arrive as a *successful* HTTP 200 -- a truncation
and a refusal -- and would otherwise be filed as this adapter's own defect.

The response fixture is loaded through the SDK's own ``Message`` model rather than being
hand-assembled into a stub object. A hand-written fixture agrees with whatever the code that
reads it expects, which is exactly how the web client lost a dozen fields; validating through
the vendor's model means a field this repository invented, renamed or dropped fails at load.
See ``tests/fixtures/anthropic_messages_response.json`` for what that fixture is and, just as
importantly, what it is not.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message

from adapters import llm_adapter
from adapters.llm_adapter import (
    AnthropicLLMClient,
    LLMAdapterError,
    OpenAILLMClient,
    ResponsesCodingExecutor,
    is_transport_fault,
    llm_client_for,
)
from agents.shared.contracts import safe_error_diagnostics
from configs.model_roles import AgentPlatform, ModelRole
from configs.settings import load_settings
from tools.llm_call_record import last_llm_call_payload

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def anthropic_settings(**overrides: Any) -> Any:
    """Load settings for a deployment that has both platforms completely configured.

    Tier presets are explicitly cleared first: the deployment `.env` names Anthropic tiers
    whose unset bounds fall back to the unsuffixed values these tests override, so without
    the pin a test lowering one bound would be judged against tiers it never configured.
    """
    from tests.test_performance_tiers import _cleared_tier_fields

    values: dict[str, Any] = {
        **_cleared_tier_fields(),
        "anthropic_reasoning_model": "claude-opus-5",
        "anthropic_reasoning_effort": "max",
        "anthropic_coding_model": "claude-opus-5",
        "anthropic_coding_reasoning_effort": "xhigh",
        "anthropic_review_model": "claude-opus-5",
        "anthropic_review_reasoning_effort": "high",
        "anthropic_scoped_fix_model": "claude-sonnet-5",
        "anthropic_scoped_fix_reasoning_effort": "high",
    }
    return load_settings(**{**values, **overrides})


def saved_response() -> Message:
    """Read the committed Messages API payload through the SDK's own response model."""
    payload = json.loads((FIXTURES / "anthropic_messages_response.json").read_text("utf-8"))
    return Message.model_validate(payload)


def message(**overrides: Any) -> Message:
    """Return the saved response with one field changed, still validated by the SDK."""
    payload = json.loads((FIXTURES / "anthropic_messages_response.json").read_text("utf-8"))
    payload.update(overrides)
    return Message.model_validate(payload)


class FakeMessages:
    """Record what the adapter sends, and answer with a response the SDK validated."""

    def __init__(self, response: Message | None = None) -> None:
        """Bind one scripted response and start with an empty call log."""
        self._response = response
        self.calls: list[dict[str, Any]] = []
        # Which transport the adapter reached for, recorded rather than inferred: the request
        # body is identical either way, so a test that only read the body could not tell.
        self.transports: list[str] = []

    async def create(self, **request: Any) -> Message:
        """Answer the blocking call, recording the exact request that produced it."""
        self.calls.append(request)
        self.transports.append("create")
        assert self._response is not None
        return self._response

    def stream(self, **request: Any) -> Any:
        """Return the async context manager the SDK's streaming helper returns."""
        self.calls.append(request)
        self.transports.append("stream")
        response = self._response

        class _Stream:
            """The SDK's stream, in the two ways the adapter now uses it.

            An iterator *and* a `get_final_message`, because that is what the real
            `AsyncMessageStream` is: its `until_done` consumes whatever remains, and the
            snapshot it returns accumulates from every event including ones already pulled.
            The adapter takes the first event purely to time it against the first-event
            budget, then hands the rest back -- so a double that only answered
            `get_final_message` would model a contract the SDK does not have.
            """

            def __init__(self) -> None:
                self.events = iter((_StreamEvent("message_start"),))
                self.closed = False

            def __aiter__(self) -> Any:
                return self

            async def __anext__(self) -> Any:
                try:
                    return next(self.events)
                except StopIteration:
                    raise StopAsyncIteration from None

            async def get_final_message(self) -> Message:
                assert response is not None
                return response

            async def close(self) -> None:
                self.closed = True

        class _Manager:
            async def __aenter__(self) -> Any:
                return _Stream()

            async def __aexit__(self, *exception: Any) -> None:
                return None

        return _Manager()


@dataclass(frozen=True, slots=True)
class _StreamEvent:
    """One Messages stream event. `message_start` is what the API sends before any content."""

    type: str


class FakeAnthropicClient:
    """A network-free substitute for the subset of the SDK client this adapter uses."""

    def __init__(self, response: Message | None = None) -> None:
        """Expose one recorded `messages` resource, as the real client does."""
        self.messages = FakeMessages(response)


def client(agent: str = "engineer", **kwargs: Any) -> AnthropicLLMClient:
    """Build the client under test over a recording fake."""
    return AnthropicLLMClient(anthropic_settings(), agent, **kwargs)


@pytest.mark.asyncio
async def test_a_saved_messages_response_becomes_the_platforms_normalized_response() -> None:
    """Every field an execution record keeps is read off the provider's own payload."""
    fake = FakeAnthropicClient(saved_response())
    llm_client = client("reviewer", client=fake, model_role=ModelRole.REVIEW)

    response = await llm_client.respond(instructions="Review the change.", input_text="the diff")

    assert response.response_id == "msg_01XmQ8kR2vN7pLdYw3TfBnKz"
    assert response.model == "claude-opus-5"
    assert (response.input_tokens, response.output_tokens) == (18422, 1177)
    assert response.provider == "anthropic"
    # The effort the request actually asked for, not what configuration nominally holds.
    assert response.reasoning_effort == "high"
    assert response.model_role == "review"
    # The thinking block carries no text and must not become part of the answer.
    assert response.output_text.startswith('{"summary"')
    # The adapter contract behind the planning-call journal: the response's model facts are
    # recorded task-locally at the moment they exist, so a journaled call whose return value
    # carries no metadata can still say which model answered it.
    payload = last_llm_call_payload()
    assert payload is not None
    assert payload["execution"]["model"] == "claude-opus-5"
    assert payload["execution"]["provider"] == "anthropic"
    assert payload["execution"]["reasoning_effort"] == "high"
    assert payload["execution"]["usage_input"] == 18422


@pytest.mark.asyncio
async def test_no_sampling_parameter_is_ever_sent() -> None:
    """`temperature` on these models is a 400, and the old predicate said to send it.

    `_supports_temperature` returns True for every `claude-*` identifier, so a naive port
    would have sent the configured temperature and every agent except the engineer -- the one
    configured at 0.0 -- would have failed on its first call. Sampling support is a
    provider-level fact here, so this client simply never sends one, and only an explicit
    assertion can see that: the failure is a provider-side 400.
    """
    settings = anthropic_settings()
    # The product manager is configured at 0.2, so a client that forwarded the agent's
    # temperature would put a non-default value in the request.
    assert settings.agents["product_manager"].temperature == pytest.approx(0.2)
    fake = FakeAnthropicClient(saved_response())

    await AnthropicLLMClient(settings, "product_manager", client=fake).respond(
        instructions="Analyze the feature.", input_text="Build it."
    )

    sent = fake.messages.calls[0]
    assert "temperature" not in sent
    assert "top_p" not in sent
    assert "top_k" not in sent
    # And the response says no effort was requested rather than implying one was honoured.
    assert "system" in sent and "instructions" not in sent
    assert sent["messages"] == [{"role": "user", "content": "Build it."}]
    assert "store" not in sent


@pytest.mark.asyncio
async def test_max_tokens_is_sent_and_is_the_configured_bound_for_that_role() -> None:
    """The Messages API rejects a request without one, and one number cannot serve both roles.

    The reviewer's output is a findings list; the engineer's is a set of complete file bodies
    for a multi-file feature. Both bounds come from configuration rather than a literal here.
    """
    # The review effort drops with its bound: startup now refuses a bound below its effort's
    # thinking floor, and this test is about per-role bounds reaching the request, not floors.
    settings = anthropic_settings(
        anthropic_coding_max_tokens=48_000,
        anthropic_review_max_tokens=12_000,
        anthropic_review_reasoning_effort="low",
    )
    coding = FakeAnthropicClient(saved_response())
    review = FakeAnthropicClient(saved_response())

    await AnthropicLLMClient(
        settings, "engineer", client=coding, model_role=ModelRole.CODING
    ).respond(instructions="Implement it.", input_text="the plan")
    await AnthropicLLMClient(
        settings, "reviewer", client=review, model_role=ModelRole.REVIEW
    ).respond(instructions="Review it.", input_text="the diff")

    assert coding.messages.calls[0]["max_tokens"] == 48_000
    assert review.messages.calls[0]["max_tokens"] == 12_000
    assert coding.messages.calls[0]["max_tokens"] > review.messages.calls[0]["max_tokens"]


@pytest.mark.asyncio
async def test_a_bound_above_the_streaming_threshold_uses_the_streaming_transport() -> None:
    """Above roughly 16K output tokens the Messages API requires a stream, and coding exceeds it.

    Decided by the output bound rather than by the deadline, because the deadline answers a
    different question: one says "this call will be slow", the other says "this call is too
    large to answer in one response".
    """
    # `medium` keeps 16_000 legal under the startup floor; the transport threshold under test
    # cares only about the bound's size.
    settings = anthropic_settings(
        anthropic_coding_max_tokens=64_000,
        anthropic_review_max_tokens=16_000,
        anthropic_review_reasoning_effort="medium",
    )
    coding = FakeAnthropicClient(saved_response())
    review = FakeAnthropicClient(saved_response())

    await AnthropicLLMClient(
        settings, "engineer", client=coding, model_role=ModelRole.CODING
    ).respond(instructions="Implement it.", input_text="the plan")
    await AnthropicLLMClient(
        settings, "reviewer", client=review, model_role=ModelRole.REVIEW
    ).respond(instructions="Review it.", input_text="the diff")

    # The effect, not the request body: the body is identical under either transport.
    assert coding.messages.transports == ["stream"]
    assert review.messages.transports == ["create"]


@pytest.mark.asyncio
async def test_a_truncated_response_is_a_budget_failure_and_never_a_complete_answer() -> None:
    """`stop_reason: max_tokens` is a bound that was too small, not a malformed response.

    Reported as "the model returned no output text" it would send its reader to look at the
    prompt; returned as an answer it would be half a plan that validates, which
    `_completed_response` already says is worse than no plan at all.
    """
    truncated = message(stop_reason="max_tokens")
    llm_client = client("engineer", client=FakeAnthropicClient(truncated))

    with pytest.raises(LLMAdapterError) as truncation:
        await llm_client.respond(instructions="Implement it.", input_text="the plan")

    assert truncation.value.failure_classification == "response_truncated"
    assert "max_tokens" in str(truncation.value)
    assert "no output text" not in str(truncation.value)
    # The partial body the provider did produce never reaches a caller as a result, and the
    # durable record says what to change: the 175-180 audit had to reconstruct from the
    # database which bound had truncated which role's call, because this line said neither.
    diagnostics = safe_error_diagnostics(truncation.value)
    assert diagnostics[0] == (
        "the model response reached its configured max_tokens bound and is truncated"
    )
    detail = diagnostics[1]
    assert f"bound of {llm_client.max_tokens}" in detail
    assert "ANTHROPIC_CODING_MAX_TOKENS" in detail
    assert "role=coding" in detail
    # Effort first, the bound conditional: AB-Feature-181's record told its operator to
    # raise a bound already at the model's output ceiling, which nobody can follow. The
    # possible action leads, and raising is offered only as far as the model permits it.
    assert "lower the configured reasoning effort, or raise" in detail
    assert "if the model's output ceiling permits a larger bound" in detail


@pytest.mark.asyncio
async def test_a_refusal_is_classified_rather_than_crashing_inside_the_adapter() -> None:
    """A refused request answers 200 with an empty content array.

    Read before `stop_reason` is checked, the first content block would raise `IndexError`
    inside this adapter and a provider policy decision would be filed as a platform defect.
    """
    refused = message(stop_reason="refusal", content=[])
    llm_client = client("engineer", client=FakeAnthropicClient(refused))

    with pytest.raises(LLMAdapterError) as refusal:
        await llm_client.respond(instructions="Implement it.", input_text="the plan")

    assert refusal.value.failure_classification == "model_refusal"
    assert not isinstance(refusal.value.__cause__, IndexError)


@pytest.mark.asyncio
async def test_effort_none_disables_thinking_and_a_level_asks_for_adaptive_thinking() -> None:
    """`none` and an effort level are mutually exclusive, and only one of them is a level.

    Disabled thinking is accepted only at effort `high` or below and refused above it, so the
    two must never be sent together.
    """
    disabled = FakeAnthropicClient(saved_response())
    await AnthropicLLMClient(
        anthropic_settings(anthropic_review_reasoning_effort="none"),
        "reviewer",
        client=disabled,
        model_role=ModelRole.REVIEW,
    ).respond(instructions="Review it.", input_text="the diff")

    sent = disabled.messages.calls[0]
    assert sent["thinking"] == {"type": "disabled"}
    assert "output_config" not in sent

    configured = FakeAnthropicClient(saved_response())
    await AnthropicLLMClient(
        anthropic_settings(),
        "reviewer",
        client=configured,
        model_role=ModelRole.REVIEW,
    ).respond(instructions="Review it.", input_text="the diff")

    sent = configured.messages.calls[0]
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "high"}
    # Raw thinking is never returned and this platform never wanted it.
    assert "display" not in sent["thinking"]


@pytest.mark.asyncio
async def test_a_streamed_answer_yields_text_deltas_and_nothing_else() -> None:
    """The provider's event vocabulary must never reach a screen."""

    class _Event:
        def __init__(self, type_: str, delta: Any = None) -> None:
            self.type = type_
            self.delta = delta

    class _Delta:
        def __init__(self, type_: str, text: str = "") -> None:
            self.type = type_
            self.text = text

    events = iter(
        (
            _Event("message_start"),
            _Event("content_block_start"),
            _Event("content_block_delta", _Delta("thinking_delta")),
            _Event("content_block_delta", _Delta("text_delta", "Deactivating ")),
            _Event("content_block_delta", _Delta("text_delta", "the unit.")),
            _Event("content_block_stop"),
            _Event("message_stop"),
        )
    )

    class _Stream:
        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            try:
                return next(events)
            except StopIteration:
                raise StopAsyncIteration from None

    class _Messages:
        async def create(self, **request: Any) -> Any:
            assert request["stream"] is True
            return _Stream()

    class _Client:
        messages = _Messages()

    llm_client = client("reviewer", client=_Client(), model_role=ModelRole.REVIEW)
    chunks = [
        chunk async for chunk in llm_client.stream(instructions="Review it.", input_text="the diff")
    ]

    assert chunks == ["Deactivating ", "the unit."]


@pytest.mark.asyncio
async def test_a_provider_fault_is_declared_without_naming_an_sdk_class_upstream() -> None:
    """The one fault boundary works unchanged for a second provider.

    `_provider_faults_declared` wraps whatever an SDK throws and records the exception's class
    name, without this platform ever enumerating a provider's exception hierarchy -- which is
    exactly why it needed no change for a second one.
    """

    class _RateLimited:
        class messages:  # noqa: N801 - mirrors the SDK's attribute layout
            @staticmethod
            async def create(**_request: Any) -> Any:
                class RateLimitError(Exception):
                    """The SDK's own type name is what identifies the fault."""

                raise RateLimitError

    # A create-transport bound, because the fake implements only `create`: the default review
    # bound now sits above the streaming threshold. `medium` keeps 16_000 legal at startup.
    settings = anthropic_settings(
        anthropic_review_max_tokens=16_000, anthropic_review_reasoning_effort="medium"
    )
    llm_client = AnthropicLLMClient(
        settings, "reviewer", client=_RateLimited(), model_role=ModelRole.REVIEW
    )

    with pytest.raises(LLMAdapterError) as fault:
        await llm_client.respond(instructions="Review it.", input_text="the diff")

    assert fault.value.failure_classification == "RateLimitError"


def test_the_factory_builds_the_client_the_feature_asked_for() -> None:
    """One factory, so six construction sites cannot come to disagree about one decision."""
    settings = anthropic_settings()

    anthropic_client = llm_client_for(
        AgentPlatform.ANTHROPIC,
        settings,
        "reviewer",
        client=FakeAnthropicClient(),
        model_role=ModelRole.REVIEW,
    )
    openai_client = llm_client_for(
        AgentPlatform.OPENAI,
        settings,
        "reviewer",
        client=object(),
        model_role=ModelRole.REVIEW,
    )

    # The concrete class, not a duck-typed attribute. `LLMClient` declares `respond` and
    # nothing else on purpose -- every caller above the adapter depends on exactly that -- so
    # widening the protocol to let a test read `provider` would widen what all of them see.
    assert isinstance(anthropic_client, AnthropicLLMClient)
    assert isinstance(openai_client, OpenAILLMClient)
    # A string from a persisted record resolves the same way a typed value does.
    assert isinstance(
        llm_client_for(
            "anthropic",
            settings,
            "reviewer",
            client=FakeAnthropicClient(),
            model_role=ModelRole.REVIEW,
        ),
        AnthropicLLMClient,
    )


def test_a_recovered_attempt_executes_its_persisted_selection() -> None:
    """Recovery must not silently switch what an already-routed attempt runs on."""
    llm_client = AnthropicLLMClient(
        anthropic_settings(),
        "engineer",
        client=FakeAnthropicClient(),
        model_role=ModelRole.SCOPED_FIX,
        resolved_model="claude-sonnet-5",
        resolved_reasoning_effort="medium",
        resolved_model_variable="ANTHROPIC_SCOPED_FIX_MODEL",
        resolved_routing_reason="The recovered attempt uses its persisted model selection.",
    )

    assert llm_client.model == "claude-sonnet-5"
    assert llm_client.reasoning_effort == "medium"
    assert llm_client.model_role is ModelRole.SCOPED_FIX


def test_an_agent_that_makes_no_model_call_cannot_be_given_a_messages_client() -> None:
    """`github` declares no role because it never reaches a provider."""
    with pytest.raises(LLMAdapterError, match="no model role"):
        AnthropicLLMClient(anthropic_settings(), "github", client=FakeAnthropicClient())


@pytest.mark.asyncio
async def test_a_truncated_coding_response_writes_nothing_and_is_recorded_as_a_budget_failure(
    tmp_path: Path,
) -> None:
    """The consequence, not the call: a workstream must not act on half a file body.

    The coding executor repairs a response it could not *parse*, because a single malformed
    response used to destroy a workstream before it had written anything. A truncation is a
    different thing -- the bound was too small, and asking the same model for the same work
    under the same bound truncates again -- so it ends the attempt with its own
    classification, and the workspace is untouched.
    """
    (tmp_path / "src").mkdir()
    original = "const actions = [activate];\n"
    (tmp_path / "src" / "AdUnitRow.tsx").write_text(original, encoding="utf-8")

    truncated = message(stop_reason="max_tokens")
    executor = ResponsesCodingExecutor(
        client("engineer", client=FakeAnthropicClient(truncated), model_role=ModelRole.CODING)
    )

    with pytest.raises(LLMAdapterError) as budget:
        await executor.execute(
            workspace_root=tmp_path,
            instructions="Implement it.",
            input_text="the plan",
        )

    assert budget.value.failure_classification == "response_truncated"
    # Not a malformed response, so no repair attempt was spent trying to reword it.
    assert (tmp_path / "src" / "AdUnitRow.tsx").read_text(encoding="utf-8") == original


class SilentAnthropicClient:
    """A Messages client whose streams are accepted and then say nothing.

    The fault AB-Feature-215 met on the Responses API, on this provider's transport. The
    request is taken, the stream opens, and no event ever arrives -- which is what a dead
    socket and a thinking model look like identically until something puts a clock on it.
    """

    def __init__(self, response: Message, *, silent_issues: int) -> None:
        self._response = response
        self._silent_remaining = silent_issues
        self.issues = 0
        self.closed: list[bool] = []

    @property
    def messages(self) -> Any:
        """The SDK nests the stream helper under `messages`; mirror that and nothing else."""
        outer = self

        class _Messages:
            def stream(self, **request: Any) -> Any:
                outer.issues += 1
                silent = outer._silent_remaining > 0
                if silent:
                    outer._silent_remaining -= 1
                return _Manager(silent)

        class _Manager:
            def __init__(self, silent: bool) -> None:
                self._silent = silent

            async def __aenter__(self) -> Any:
                return _Stream(self._silent)

            async def __aexit__(self, *exception: Any) -> None:
                return None

        class _Stream:
            def __init__(self, silent: bool) -> None:
                self._silent = silent
                self._events = iter((_StreamEvent("message_start"),))

            def __aiter__(self) -> Any:
                return self

            async def __anext__(self) -> Any:
                if self._silent:
                    await asyncio.sleep(3_600)
                    raise AssertionError("the silent stream was waited out, which cannot happen")
                try:
                    return next(self._events)
                except StopIteration:
                    raise StopAsyncIteration from None

            async def get_final_message(self) -> Message:
                return outer._response

            async def close(self) -> None:
                outer.closed.append(True)

        return _Messages()


def _silent_stream_settings() -> Any:
    """Anthropic settings whose coding bound streams, with a budget a test can outlast."""
    return anthropic_settings(anthropic_coding_max_tokens=64_000, first_event_timeout_seconds=1)


@pytest.mark.asyncio
async def test_a_silent_messages_stream_is_issued_again_rather_than_waited_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Responses client's protection, now on this transport too.

    Until this, `get_final_message` consumed the stream in one await and there was nowhere to
    observe the first event -- so a stream the provider accepted and never wrote to could
    only be noticed when the engineer's whole 1800-second deadline expired. The count comes
    back on the response for the same reason it does there: a re-issue that then succeeded
    leaves no other trace anywhere.
    """
    monkeypatch.setattr(llm_adapter, "_STREAM_REISSUE_BACKOFF_SECONDS", (0.0, 0.0))
    fake = SilentAnthropicClient(saved_response(), silent_issues=1)

    response = await AnthropicLLMClient(
        _silent_stream_settings(), "engineer", client=fake, model_role=ModelRole.CODING
    ).respond(instructions="Implement it.", input_text="the plan")

    assert response.output_text.startswith('{"summary"')
    assert response.stream_reissues == 1
    assert fake.issues == 2
    # The dead stream was released rather than left open behind its replacement.
    assert fake.closed


@pytest.mark.asyncio
async def test_a_messages_stream_that_never_speaks_fails_as_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three silent issues end the call, classified as weather rather than as an answer.

    `stream_silent` is what sends this to the child workstream's fault allowance instead of
    to the attempt -- the distinction that cost AB-Feature-215 a full attempt when a repair
    call's timeout was re-raised as the gate's own rejection.
    """
    monkeypatch.setattr(llm_adapter, "_STREAM_REISSUE_BACKOFF_SECONDS", (0.0, 0.0))
    fake = SilentAnthropicClient(saved_response(), silent_issues=99)

    with pytest.raises(LLMAdapterError) as raised:
        await AnthropicLLMClient(
            _silent_stream_settings(), "engineer", client=fake, model_role=ModelRole.CODING
        ).respond(instructions="Implement it.", input_text="the plan")

    assert raised.value.failure_classification == "stream_silent"
    assert "on 3 successive issues" in str(raised.value)
    assert fake.issues == 3
    assert is_transport_fault(raised.value) is True
