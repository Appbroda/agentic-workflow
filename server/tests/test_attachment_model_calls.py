"""The image reaches the model, and the text path does not move (89-, Parts E and F).

The first test in this file is the one the safety rules ask for by name: with no images, the
request body each provider receives is asserted *whole* against the body it received before
this parameter existed. That is a byte-for-byte claim about every feature submitted without
pictures, which is nearly all of them, and it is checked rather than reasoned about.

The rest is the capability rule, which is the only declaration in this platform whose empty
default is a refusal. An undeclared model is not shown an image -- at acceptance, and again at
the call site, because a redeploy can change what a tier resolves to in between.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from adapters.llm_adapter import (
    AnthropicLLMClient,
    ImageInput,
    LLMAdapterError,
    OpenAILLMClient,
)
from configs.model_roles import ModelRole
from configs.settings import load_settings
from tests.test_adapters import FakeOpenAIClient, FakeOpenAIResponse, FakeUsage
from tests.test_anthropic_adapter import FakeAnthropicClient, saved_response

FIXTURES = Path(__file__).resolve().parent / "fixtures"

PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUg=="
IMAGES = (
    ImageInput(media_type="image/png", data=PNG_BASE64),
    ImageInput(media_type="image/jpeg", data="/9j/4AAQ"),
)


# `Settings` reads the repository's own `.env`, so a declaration an operator makes for their
# deployment would otherwise decide these tests: once `MODEL_VISION_CAPABLE` named the models
# this deployment runs, the two fail-closed cases below started asserting the opposite of what
# they say. The blind default is stated here rather than borrowed, and a test that wants a
# declaration passes one as an override.
_NOTHING_DECLARED_VISION_CAPABLE = {"model_vision_capable": ""}


def openai_settings(**overrides: Any) -> Any:
    """Settings for a deployment with the Responses platform configured."""
    from tests.test_performance_tiers import _cleared_tier_fields

    values: dict[str, Any] = {
        **_cleared_tier_fields(),
        **_NOTHING_DECLARED_VISION_CAPABLE,
        "openai_reasoning_model": "gpt-6-astra",
        "openai_coding_model": "gpt-6-astra",
        "openai_review_model": "gpt-6-astra",
        "openai_fix_model": "gpt-6-astra",
    }
    return load_settings(**{**values, **overrides})


def anthropic_settings(**overrides: Any) -> Any:
    """Settings for a deployment with the Messages platform configured."""
    from tests.test_performance_tiers import _cleared_tier_fields

    values: dict[str, Any] = {
        **_cleared_tier_fields(),
        **_NOTHING_DECLARED_VISION_CAPABLE,
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


def openai_client() -> FakeOpenAIClient:
    """The suite's Responses double, answering a payload valid for these tests."""
    return FakeOpenAIClient(
        FakeOpenAIResponse(
            id="resp-1",
            model="gpt-6-astra",
            output_text='{"ok": true}',
            usage=FakeUsage(input_tokens=10, output_tokens=5),
        )
    )


# --------------------------------------------------------------- the unchanged text path


@pytest.mark.asyncio
async def test_the_responses_request_with_no_images_is_exactly_todays_request() -> None:
    """Asserted whole, field for field. `input` is the bare string it has always been."""
    fake = openai_client()
    client = OpenAILLMClient(
        openai_settings(),
        "product_manager",
        api_key="k",
        client=fake,
        model_role=ModelRole.REASONING,
    )

    await client.respond(instructions="do the thing", input_text="the document")

    settings = openai_settings()
    # Whole, not key-by-key. The tunables are read from settings rather than restated, so
    # this fails for the adapter changing and not for a deadline being retuned.
    assert fake.responses.calls == [
        {
            "model": "gpt-6-astra",
            "instructions": "do the thing",
            # A bare string, which is the entire claim.
            "input": "the document",
            "store": False,
            "timeout": settings.timeout_seconds_for_agent(
                "product_manager", model_role=ModelRole.REASONING
            ),
            "reasoning": {"effort": "max"},
            # The long-deadline transport, chosen by the deadline and unaffected by any of
            # this: a reasoning call must not sit on an idle connection for half an hour.
            "stream": True,
        }
    ]
    assert isinstance(fake.responses.calls[0]["input"], str)


@pytest.mark.asyncio
async def test_the_messages_request_with_no_images_is_exactly_todays_request() -> None:
    """The same claim on the other provider: `content` stays a bare string."""
    fake = FakeAnthropicClient(saved_response())
    client = AnthropicLLMClient(
        anthropic_settings(), "reviewer", api_key="k", client=fake, model_role=ModelRole.REVIEW
    )

    await client.respond(instructions="do the thing", input_text="the document")

    settings = anthropic_settings()
    assert fake.messages.calls == [
        {
            "model": "claude-opus-5",
            "system": "do the thing",
            # A bare string as the content, which is the entire claim.
            "messages": [{"role": "user", "content": "the document"}],
            "max_tokens": client.max_tokens,
            "timeout": settings.timeout_seconds_for_agent("reviewer", model_role=ModelRole.REVIEW),
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }
    ]
    assert isinstance(fake.messages.calls[0]["messages"][0]["content"], str)


@pytest.mark.asyncio
async def test_an_empty_input_is_still_refused_even_with_images() -> None:
    """Images accompany a request; they are not one."""
    for client in (
        OpenAILLMClient(
            openai_settings(model_vision_capable='["gpt-6-astra"]'),
            "product_manager",
            api_key="k",
            client=openai_client(),
            model_role=ModelRole.REASONING,
        ),
        AnthropicLLMClient(
            anthropic_settings(model_vision_capable='["claude-opus-5"]'),
            "reviewer",
            api_key="k",
            client=FakeAnthropicClient(saved_response()),
            model_role=ModelRole.REVIEW,
        ),
    ):
        with pytest.raises(LLMAdapterError) as raised:
            await client.respond(instructions="do the thing", input_text="   ", images=IMAGES)
        assert raised.value.failure_classification == "empty_request"


# ------------------------------------------------------------------ the image-carrying path


@pytest.mark.asyncio
async def test_the_responses_request_with_images_puts_them_before_the_text() -> None:
    """One user message, the images first, the instruction the model acts on last."""
    fake = openai_client()
    client = OpenAILLMClient(
        openai_settings(model_vision_capable='["gpt-6-astra"]'),
        "product_manager",
        api_key="k",
        client=fake,
        model_role=ModelRole.REASONING,
    )

    await client.respond(instructions="do the thing", input_text="the document", images=IMAGES)

    assert fake.responses.calls[0]["input"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{PNG_BASE64}",
                },
                {"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/4AAQ"},
                {"type": "input_text", "text": "the document"},
            ],
        }
    ]


@pytest.mark.asyncio
async def test_the_messages_request_with_images_puts_them_before_the_text() -> None:
    """The Messages shape, with base64 sources and the text block last."""
    fake = FakeAnthropicClient(saved_response())
    client = AnthropicLLMClient(
        anthropic_settings(model_vision_capable='["claude-opus-5"]'),
        "reviewer",
        api_key="k",
        client=fake,
        model_role=ModelRole.REVIEW,
    )

    await client.respond(instructions="do the thing", input_text="the document", images=IMAGES)

    assert fake.messages.calls[0]["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": PNG_BASE64,
                    },
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "/9j/4AAQ",
                    },
                },
                {"type": "text", "text": "the document"},
            ],
        }
    ]


# ---------------------------------------------------------------------- the capability rule


def test_an_undeclared_model_is_not_vision_capable() -> None:
    """The fail-closed default, and the reason it points the other way to the rest."""
    blind = OpenAILLMClient(
        openai_settings(),
        "product_manager",
        api_key="k",
        client=openai_client(),
        model_role=ModelRole.REASONING,
    )
    seeing = OpenAILLMClient(
        openai_settings(model_vision_capable='["gpt-6-astra"]'),
        "product_manager",
        api_key="k",
        client=openai_client(),
        model_role=ModelRole.REASONING,
    )

    assert blind.vision_capable is False
    assert seeing.vision_capable is True


def test_a_malformed_declaration_is_refused_rather_than_read_as_empty() -> None:
    """Reading a typo as "nothing accepts images" would refuse every submission silently."""
    from configs.model_roles import ModelConfigurationError

    with pytest.raises(ModelConfigurationError, match="MODEL_VISION_CAPABLE"):
        openai_settings(model_vision_capable="gpt-6-astra").declared_vision_capable()
    with pytest.raises(ModelConfigurationError, match="MODEL_VISION_CAPABLE"):
        openai_settings(model_vision_capable='{"a": 1}').declared_vision_capable()


@pytest.mark.asyncio
async def test_an_undeclared_client_raises_rather_than_dropping_the_images() -> None:
    """Both adapters, both refusing, and neither making the call."""
    openai_fake = openai_client()
    anthropic_fake = FakeAnthropicClient(saved_response())
    clients = (
        OpenAILLMClient(
            openai_settings(),
            "product_manager",
            api_key="k",
            client=openai_fake,
            model_role=ModelRole.REASONING,
        ),
        AnthropicLLMClient(
            anthropic_settings(),
            "reviewer",
            api_key="k",
            client=anthropic_fake,
            model_role=ModelRole.REVIEW,
        ),
    )

    for client in clients:
        with pytest.raises(LLMAdapterError) as raised:
            await client.respond(
                instructions="do the thing", input_text="the document", images=IMAGES
            )
        assert raised.value.failure_classification == "model_not_vision_capable"
        assert "MODEL_VISION_CAPABLE" in str(raised.value)

    # And nothing was sent. A refusal that had already made the call would be a drop.
    assert openai_fake.responses.calls == []
    assert anthropic_fake.messages.calls == []


def test_base64_is_not_carried_into_a_sanitized_provider_detail() -> None:
    """The bound the sanitizer already applies, checked against an image-sized payload.

    Uploaded bytes are never logged, never journaled and never sent to Slack. The one place
    provider text is retained is this line, and it is length-bounded -- so a provider that
    echoed a request back cannot put a megabyte of base64 into a journal row.
    """
    from adapters.llm_adapter import _PROVIDER_DETAIL_MAX_CHARACTERS, _sanitized_provider_detail

    echoed = RuntimeError("400: invalid image " + "QUJDRA" * 200_000)

    detail = _sanitized_provider_detail(echoed)

    assert len(detail) <= _PROVIDER_DETAIL_MAX_CHARACTERS + len(" [truncated]")
    assert detail.endswith("[truncated]")


# --------------------------------------------- what the assistant inlines, and what it does not


def test_the_assistant_inlines_the_metadata_and_never_the_bytes() -> None:
    """The feature assistant stays blind to the pictures, and that is a decision.

    It already inlines a PRD artifact whole when a message mentions the PRD, so it now
    inlines the attachments *metadata* -- markers, captions, filenames, hashes -- because
    that is what the artifact contains. It cannot inline bytes because the artifact holds
    none, which is the first safety rule doing its job at a boundary nobody changed.

    A person asking the assistant about a screenshot therefore gets an answer from its
    caption. Widening the assistant is a separate item; this test pins the current answer so
    the blindness is recorded rather than assumed.
    """
    from agents.assistant.context import build_context
    from api.feature_control_plane import _initial_feature_state
    from api.feature_schemas import StartFeatureRequest
    from artifacts.schemas import PRDAttachment

    attachment = PRDAttachment(
        attachment_id="attachment-1",
        marker="login-error",
        caption="the error a user sees",
        filename="login-error.png",
        media_type="image/png",
        byte_size=70,
        sha256="c" * 64,
    )
    # Built through the acceptance path rather than by hand, so the artifact under test is
    # the artifact acceptance actually writes.
    state = _initial_feature_state(
        "feature-1",
        StartFeatureRequest.model_validate(
            {
                "prd": {
                    "title": "Login audit trail",
                    "problem_statement": "Login fails silently [image:login-error].",
                    "attachments": [{"attachment_id": "attachment-1", "marker": "login-error"}],
                },
                "repositories": [
                    {
                        "repository_url": "https://github.com/example/backend",
                        "default_branch": "main",
                        "required": True,
                    }
                ],
            }
        ),
        attachments=[attachment],
    )

    context = build_context(state, events=[], query="what does the prd say?")

    inlined = [item for item in context["artifacts_inline"] if item["artifact_type"] == "prd"]
    assert inlined, "a question about the PRD should inline it"
    recorded = inlined[0]["attachments"][0]
    assert recorded["marker"] == "login-error"
    assert recorded["caption"] == "the error a user sees"
    assert recorded["sha256"] == "c" * 64
    # And nothing a picture could travel in.
    rendered = json.dumps(context)
    for forbidden in ('"data"', '"content"', "base64"):
        assert forbidden not in rendered
