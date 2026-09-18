"""Watching an answer being written, and watching a feature move.

Both streams here are ordinary authenticated HTTP responses read with `fetch`, not
`EventSource`. That is the whole reason they exist in this shape: the browser's own
`EventSource` cannot send an `Authorization` header, and the usual way round it -- putting the
token in the query string -- writes a credential into somewhere that gets logged and cached.

A stream is also never the record. The transcript and the event table are what a client
believes; the stream only says when to look, and these tests hold it to that.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from httpx import ASGITransport, AsyncClient

from agents.assistant.agent import (
    ACTION_DELIMITER,
    FeatureAssistant,
    split_streamed_reply,
)
from agents.shared.contracts import AgentArtifactError
from main import create_app
from prompts.prompt_loader import PromptLoader
from services.feature_actions import FeatureActionService
from services.feature_chat import FeatureChatService
from storage.action_store import InMemoryFeatureActionStore
from storage.chat_store import InMemoryChatMessageStore
from tests.support import settle
from tests.test_feature_api import feature_payload

KEY = "stream-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


class ScriptedStreamingClient:
    """Emit a fixed sequence of deltas, the way the Responses stream would."""

    def __init__(self, *deltas: str, fail_after: int | None = None) -> None:
        """Queue the deltas, optionally dying partway to imitate a dropped provider."""
        self._deltas = deltas
        self._fail_after = fail_after

    @property
    def vision_capable(self) -> bool:
        """No image reaches this double, so the boundary answers False."""
        return False

    async def respond(self, **_kwargs: Any) -> Any:
        """Unused: these tests are about the streaming path."""
        raise NotImplementedError

    async def stream(self, *, instructions: str, input_text: str) -> Any:
        """Yield each queued delta in order."""
        del instructions, input_text
        for index, delta in enumerate(self._deltas):
            if self._fail_after is not None and index == self._fail_after:
                raise TimeoutError("the provider stopped answering")
            yield delta


async def streaming_app(*deltas: str, fail_after: int | None = None) -> Any:
    """Build an application whose assistant streams a scripted answer."""
    app = create_app(platform_api_key=KEY)
    assistant = FeatureAssistant(
        prompt_loader=PromptLoader(),
        llm_client=ScriptedStreamingClient(*deltas, fail_after=fail_after),
    )
    app.state.feature_actions = FeatureActionService(
        store=InMemoryFeatureActionStore(), build_revision="test"
    )
    app.state.feature_chat = FeatureChatService(
        assistant_for=lambda credentials, *, agent_platform: assistant,
        control_plane=app.state.feature_control_plane,
        store=InMemoryChatMessageStore(),
        actions=app.state.feature_actions,
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post(
            "/features/start",
            headers={**AUTH, "Idempotency-Key": "stream-feature"},
            json=feature_payload(),
        )
        await settle(app)
    return app


def parse_events(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Read an SSE body back into the events it encodes.

    Written against the bytes rather than a client library on purpose: the framing is the
    contract with the browser, and a helper that reconstructs it leniently would not notice
    if the server stopped producing it correctly.
    """
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        name = ""
        data = "{}"
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = line.removeprefix("data: ")
        events.append((name, json.loads(data)))
    return events


@pytest.mark.asyncio
async def test_an_answer_arrives_in_pieces_and_is_then_the_transcript() -> None:
    """What was streamed and what was recorded have to be the same answer."""
    app = await streaming_app("The backend ", "stopped on its ", "own setup.")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/features/feature-login/chat/stream",
            headers=AUTH,
            json={"message": "What happened?"},
        )
        history = await client.get("/features/feature-login/chat", headers=AUTH)

    events = parse_events(response.text)
    kinds = [name for name, _ in events]
    deltas = [payload["text"] for name, payload in events if name == "delta"]

    assert response.headers["content-type"].startswith("text/event-stream")
    assert kinds[0] == "user", "the question is acknowledged before the answer starts"
    assert len(deltas) > 1, "an answer delivered in one piece is not streamed"
    assert "".join(deltas) == "The backend stopped on its own setup."
    assert kinds[-1] == "done"

    turns = history.json()["messages"]
    assert [item["role"] for item in turns] == ["user", "assistant"]
    assert turns[1]["content"] == "The backend stopped on its own setup."


@pytest.mark.asyncio
async def test_a_proposal_is_never_streamed_as_prose() -> None:
    """The command is read by the platform. Nobody should watch JSON being typed."""
    app = await streaming_app(
        "I can cancel it.",
        f"\n{ACTION_DELIMITER}\n",
        '{"type": "CANCEL_WORKFLOW", "arguments": {"reason": "no longer needed"},',
        ' "summary": "Cancel this feature."}',
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/features/feature-login/chat/stream",
            headers=AUTH,
            json={"message": "Cancel this."},
        )

    events = parse_events(response.text)
    shown = "".join(payload["text"] for name, payload in events if name == "delta")
    final = next(payload for name, payload in events if name == "message")

    assert shown.strip() == "I can cancel it."
    assert ACTION_DELIMITER not in shown
    assert "CANCEL_WORKFLOW" not in shown
    # The proposal arrives as a typed action on the persisted message, which is what a
    # confirmation acts on.
    assert final["proposed_action"]["type"] == "CANCEL_WORKFLOW"
    assert final["action_status"] == "pending"


@pytest.mark.asyncio
async def test_a_delimiter_split_across_deltas_is_not_shown() -> None:
    """A provider chunks wherever it likes, including through the delimiter."""
    app = await streaming_app(
        "Cancelling.",
        "\n<<<PLAT",
        "FORM_ACTION>>>\n",
        '{"type": "CANCEL_WORKFLOW", "arguments": {}, "summary": "Cancel."}',
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/features/feature-login/chat/stream", headers=AUTH, json={"message": "Cancel."}
        )

    shown = "".join(
        payload["text"] for name, payload in parse_events(response.text) if name == "delta"
    )
    assert shown.strip() == "Cancelling."
    assert "<<<PLAT" not in shown


@pytest.mark.asyncio
async def test_an_interrupted_answer_keeps_what_was_read() -> None:
    """The person read those words. A transcript that disagrees with the screen is worse."""
    # Dies after two deltas: the provider stopped answering partway through a sentence.
    app = await streaming_app(
        "The backend stopped because ", "its lint configuration ", "never loads", fail_after=2
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/features/feature-login/chat/stream", headers=AUTH, json={"message": "Why?"}
        )
        history = await client.get("/features/feature-login/chat", headers=AUTH)

    events = parse_events(response.text)
    final = next(payload for name, payload in events if name == "message")
    stored = history.json()["messages"][1]["content"]

    assert "The backend stopped because its lint configuration" in stored
    assert "never loads" not in stored, "only what actually arrived is kept"
    assert "interrupted" in stored
    assert final["detail"] is not None


@pytest.mark.asyncio
async def test_a_malformed_proposal_keeps_the_answer_and_drops_the_command() -> None:
    """Guessing at a half-written command would put a lying button in front of somebody."""
    app = await streaming_app(
        "Here is what happened.",
        f"\n{ACTION_DELIMITER}\n",
        '{"type": "NOT_A_REAL_ACTION", "arguments": {}, "summary": "x"}',
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/features/feature-login/chat/stream", headers=AUTH, json={"message": "What?"}
        )
        history = await client.get("/features/feature-login/chat", headers=AUTH)

    stored = history.json()["messages"][1]
    assert "Here is what happened." in stored["content"]
    assert stored["proposed_action"] is None
    assert "message" in [name for name, _ in parse_events(response.text)]


@pytest.mark.asyncio
async def test_the_question_survives_an_answer_that_never_arrives() -> None:
    """A transcript that loses the question tells the next reader nothing."""
    app = await streaming_app("anything", fail_after=0)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post(
            "/features/feature-login/chat/stream",
            headers=AUTH,
            json={"message": "Why did it stop?"},
        )
        history = await client.get("/features/feature-login/chat", headers=AUTH)

    turns = history.json()["messages"]
    assert turns[0]["content"] == "Why did it stop?"
    assert "did not produce an answer" in turns[1]["content"]


@pytest.mark.asyncio
async def test_neither_stream_is_reachable_without_authentication() -> None:
    """A stream is feature data, and is behind exactly what everything else is behind."""
    app = await streaming_app("hello")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        chat = await client.post("/features/feature-login/chat/stream", json={"message": "hello"})
        events = await client.get("/features/feature-login/events/stream")

    assert chat.status_code == 401
    assert events.status_code == 401


async def collect_stream(
    generator: AsyncGenerator[str, None], *, until: int
) -> list[tuple[str, dict[str, Any]]]:
    """Read a live stream until it has produced enough events, then close it.

    Driven directly rather than over HTTP because an event stream does not end: it waits for
    the next thing to happen. Under the in-process test transport a reader that stops reading
    never reaches the server, so the generator would run forever and the test with it. In a
    deployment the disconnect does reach it, and the server cancels the task -- which is the
    same thing this does, explicitly.
    """
    frames: list[tuple[str, dict[str, Any]]] = []
    try:
        while len([1 for name, _ in frames if name == "event"]) < until:
            chunk = await asyncio.wait_for(anext(generator), timeout=5)
            frames.extend(parse_events(chunk))
    finally:
        await generator.aclose()
    return frames


@pytest.mark.asyncio
async def test_the_event_stream_sends_what_happened_and_resumes_from_a_cursor() -> None:
    """A reconnect must not replay a feature from the beginning, nor skip its middle."""
    app = await streaming_app("hello")
    control_plane = app.state.feature_control_plane
    recorded = await control_plane.events_after("feature-login", after_id=None, limit=200)
    assert len(recorded) >= 2, "the fixture feature must have produced something to stream"

    from api.feature_routes import _event_stream

    seen = await collect_stream(
        cast(AsyncGenerator[str, None], _event_stream(control_plane, "feature-login", None)),
        until=1,
    )
    first_id = next(payload["id"] for name, payload in seen if name == "event")
    resumed = await collect_stream(
        cast(
            AsyncGenerator[str, None],
            _event_stream(control_plane, "feature-login", first_id),
        ),
        until=1,
    )

    streamed_ids = [payload["id"] for name, payload in seen if name == "event"]
    resumed_ids = [payload["id"] for name, payload in resumed if name == "event"]

    assert seen[0][0] == "open", "a client learns it is connected before anything happens"
    assert streamed_ids[0] == recorded[0].id, "the stream carries what the table carries"
    assert resumed_ids and all(item > first_id for item in resumed_ids), (
        "a cursor resumes; it does not replay what the client already has"
    )


@pytest.mark.asyncio
async def test_streamed_events_carry_the_id_a_client_deduplicates_on() -> None:
    """Duplicate delivery has to be harmless, and that needs an id on the wire itself."""
    app = await streaming_app("hello")

    from api.feature_routes import _event_stream

    generator = cast(
        AsyncGenerator[str, None],
        _event_stream(app.state.feature_control_plane, "feature-login", None),
    )
    frames = ""
    try:
        while "event: event" not in frames:
            frames += await asyncio.wait_for(anext(generator), timeout=5)
    finally:
        await generator.aclose()

    # The SSE `id:` field, which is what a browser echoes back as Last-Event-ID.
    block = next(item for item in frames.split("\n\n") if "event: event" in item)
    assert block.splitlines()[0].startswith("id: ")


@pytest.mark.asyncio
async def test_the_event_stream_refuses_a_feature_that_does_not_exist() -> None:
    """Refusing before the response begins is the only place a status code still works."""
    app = await streaming_app("hello")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/features/nothing-here/events/stream", headers=AUTH)

    assert response.status_code == 404


def test_prose_without_a_delimiter_is_the_whole_answer() -> None:
    """Most turns propose nothing, and must not need the delimiter to be complete."""
    prose, action = split_streamed_reply("Both repositories finished.")

    assert prose == "Both repositories finished."
    assert action is None


def test_an_announced_proposal_that_was_never_written_is_not_invented() -> None:
    """A delimiter with nothing after it is an absent proposal, not a malformed one."""
    prose, action = split_streamed_reply(f"Here is the answer.\n{ACTION_DELIMITER}\n   ")

    assert prose == "Here is the answer."
    assert action is None


def test_an_unsupported_action_in_the_tail_is_refused() -> None:
    """The tail is a command from a model, and is checked like every other one."""
    with pytest.raises(AgentArtifactError):
        split_streamed_reply(
            f"text\n{ACTION_DELIMITER}\n"
            '{"type": "DELETE_EVERYTHING", "arguments": {}, "summary": "x"}'
        )


@pytest.mark.asyncio
async def test_a_cancelled_stream_still_records_what_was_written() -> None:
    """Closing the tab is not a reason for the transcript to lose the answer."""
    app = await streaming_app("The backend ", "stopped.")
    chat = app.state.feature_chat

    from api.control_plane import RequestScopedCredentials

    stream = chat.send_stream(
        "feature-login", "Why?", credentials=RequestScopedCredentials(None, None)
    )
    # Read the question acknowledgement and one delta, then walk away.
    await anext(stream)
    await anext(stream)
    await stream.aclose()
    await asyncio.sleep(0)

    messages = await chat.history("feature-login")
    assert [item.role for item in messages][:1] == ["user"]
    assert messages[0].content == "Why?"
