"""The feature assistant answers from the record and never mutates workflow state itself."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from adapters.llm_adapter import ImageInput
from agents.assistant.agent import FeatureAssistant
from agents.assistant.context import build_context
from agents.shared.contracts import AgentArtifactError
from api.control_plane import RequestScopedCredentials, WorkflowNotFoundError
from main import create_app
from prompts.prompt_loader import PromptLoader
from services.feature_actions import FeatureActionService
from services.feature_chat import ChatMessage, FeatureChatService
from storage.action_store import InMemoryFeatureActionStore
from storage.chat_store import InMemoryChatMessageStore
from tests.support import settle
from tests.test_feature_api import feature_payload


class ScriptedAssistantClient:
    """Return queued assistant payloads and keep the rendered prompt for assertion."""

    def __init__(self, *payloads: dict[str, Any]) -> None:
        """Queue one response per expected turn."""
        self._payloads = list(payloads)
        self.calls: list[str] = []

    @property
    def vision_capable(self) -> bool:
        """No image reaches this double, so the boundary answers False."""
        return False

    async def respond(
        self,
        *,
        instructions: str,
        input_text: str,
        images: Sequence[ImageInput] = (),
    ) -> Any:
        """Return the next queued payload."""
        del input_text
        self.calls.append(instructions)
        payload = self._payloads.pop(0)

        class Response:
            output_text = json.dumps(payload)
            model = "test-model"
            response_id = f"assistant-{len(self.calls)}"

        return Response()


def assistant(*payloads: dict[str, Any]) -> tuple[FeatureAssistant, ScriptedAssistantClient]:
    """Build the assistant over a scripted model boundary."""
    client = ScriptedAssistantClient(*payloads)
    return FeatureAssistant(prompt_loader=PromptLoader(), llm_client=client), client


async def started_app(chat_payloads: list[dict[str, Any]]) -> tuple[Any, FeatureChatService]:
    """Create an app with a started feature and an assistant wired to scripted replies.

    A confirmed proposal becomes a durable action, so the service is given the same action
    machinery the deployment uses -- the in-memory store rather than a stub, so that the
    identity, claim and lease rules a confirmation depends on are the real ones here too.
    """
    app = create_app(platform_api_key="chat-key")
    agent, _ = assistant(*chat_payloads)
    actions = FeatureActionService(store=InMemoryFeatureActionStore(), build_revision="test")
    service = FeatureChatService(
        assistant_for=lambda credentials, *, agent_platform: agent,
        control_plane=app.state.feature_control_plane,
        store=InMemoryChatMessageStore(),
        actions=actions,
    )
    app.state.feature_chat = service
    app.state.feature_actions = actions
    headers = {"Authorization": "Bearer chat-key", "Idempotency-Key": "chat-feature"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        # The assistant answers questions about a feature's record, so these tests want one
        # that has actually run rather than one still queued.
        await settle(app)
    return app, service


@pytest.mark.asyncio
async def test_the_assistant_answers_and_the_conversation_persists() -> None:
    """Reopening a feature must not lose what was already asked and answered."""
    app, _ = await started_app(
        [{"reply": "Both repositories finished and opened pull requests.", "action": None}]
    )
    headers = {"Authorization": "Bearer chat-key"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        sent = await client.post(
            "/features/feature-login/chat", headers=headers, json={"message": "What happened?"}
        )
        history = await client.get("/features/feature-login/chat", headers=headers)

    turns = sent.json()["messages"]
    assert [item["role"] for item in turns] == ["user", "assistant"]
    assert turns[1]["content"] == "Both repositories finished and opened pull requests."
    # A plain question gets a plain answer; nothing is proposed.
    assert turns[1]["proposed_action"] is None
    assert [item["role"] for item in history.json()["messages"]] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_a_proposal_does_nothing_until_a_person_confirms_it() -> None:
    """The assistant proposes; only a confirmation reaches the control plane.

    The feature here has already completed, so the platform refuses to resume it. That is the
    point: the assistant does not decide whether an action is legal, and a refusal reaches the
    person as a refusal rather than as a silent success.
    """
    app, _ = await started_app(
        [
            {
                "reply": "I can continue it.",
                "action": {
                    "type": "RESUME_WORKFLOW",
                    "arguments": {},
                    "summary": "Continue this feature.",
                },
            }
        ]
    )
    headers = {"Authorization": "Bearer chat-key"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        sent = await client.post(
            "/features/feature-login/chat", headers=headers, json={"message": "Continue this."}
        )
        proposal = sent.json()["messages"][1]
        before = await client.get("/features/feature-login", headers=headers)
        confirmed = await client.post(
            f"/features/feature-login/chat/{proposal['id']}/confirm", headers=headers
        )
        after = await client.get("/features/feature-login", headers=headers)
        history = await client.get("/features/feature-login/chat", headers=headers)

    assert proposal["proposed_action"]["type"] == "RESUME_WORKFLOW"
    assert proposal["action_status"] == "pending"
    # Proposing changed nothing.
    assert before.json()["status"] == "completed"
    # The platform refused, and said so.
    assert confirmed.status_code == 409
    assert after.json()["status"] == "completed"
    # A failed mutation is terminal. The platform may have crossed an external checkpoint
    # before reporting a provider failure, so blindly offering the same proposal again could
    # duplicate a side effect.
    assert history.json()["messages"][1]["action_status"] == "failed"


@pytest.mark.asyncio
async def test_a_confirmed_action_goes_through_the_control_plane_and_takes_its_verdict() -> None:
    """Confirmation calls the platform's own method and reports what it decided."""
    app, service = await started_app(
        [
            {
                "reply": "I can resume it.",
                "action": {
                    "type": "RESUME_WORKFLOW",
                    "arguments": {},
                    "summary": "Resume this feature.",
                },
            }
        ]
    )
    calls: list[str] = []
    original = app.state.feature_control_plane.resume

    async def counted(feature_id: str, **kwargs: Any) -> Any:
        calls.append(feature_id)
        return await original(feature_id, **kwargs)

    app.state.feature_control_plane.resume = counted

    headers = {"Authorization": "Bearer chat-key"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        sent = await client.post(
            "/features/feature-login/chat", headers=headers, json={"message": "Resume it."}
        )
        message_id = sent.json()["messages"][1]["id"]
        first = await client.post(
            f"/features/feature-login/chat/{message_id}/confirm", headers=headers
        )
        second = await client.post(
            f"/features/feature-login/chat/{message_id}/confirm", headers=headers
        )

    del service
    # This feature has completed, so the platform refuses to resume it -- and the refusal is
    # the platform's, reported as a refusal rather than executed by the assistant.
    assert first.status_code == 409
    assert "cannot be resumed" in first.json()["detail"]
    assert second.status_code == 409
    # The proposal was atomically claimed before the first call. The second confirmation can
    # never execute it again, even when requests race across server instances.
    assert calls == ["feature-login"]


@pytest.mark.asyncio
async def test_a_rejected_proposal_is_recorded_rather_than_forgotten() -> None:
    """The transcript has to show that somebody declined, not just that nothing happened."""
    app, _ = await started_app(
        [
            {
                "reply": "I can cancel it.",
                "action": {
                    "type": "CANCEL_WORKFLOW",
                    "arguments": {},
                    "summary": "Cancel this feature.",
                },
            }
        ]
    )
    headers = {"Authorization": "Bearer chat-key"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        sent = await client.post(
            "/features/feature-login/chat", headers=headers, json={"message": "Cancel it?"}
        )
        message_id = sent.json()["messages"][1]["id"]
        rejected = await client.post(
            f"/features/feature-login/chat/{message_id}/reject", headers=headers
        )
        feature = await client.get("/features/feature-login", headers=headers)

    assert rejected.json()["action_status"] == "rejected"
    assert feature.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_an_incomplete_retry_action_is_refused_before_it_reaches_a_person() -> None:
    """A retry grant without its audit fields must not become a confirmable button."""
    agent, _ = assistant(
        [
            {
                "reply": "Retrying the backend.",
                "action": {
                    "type": "RETRY_WORKSTREAM",
                    "arguments": {"repository_id": "backend"},
                    "summary": "Retry the backend.",
                },
            }
        ][0]
    )

    with pytest.raises(AgentArtifactError, match="four audited arguments"):
        await agent.respond(
            feature_id="feature-login", context={}, history=[], message="Retry the backend."
        )


@pytest.mark.asyncio
async def test_malformed_clarification_answers_cannot_turn_into_a_generic_resume() -> None:
    """Filtering invalid answers to an empty list would execute the endpoint's other meaning."""
    agent, _ = assistant(
        {
            "reply": "I can answer it.",
            "action": {
                "type": "ANSWER_CLARIFICATION",
                "arguments": {"answers": [{}]},
                "summary": "Answer the clarification.",
            },
        }
    )

    with pytest.raises(AgentArtifactError, match="question_id and answer"):
        await agent.respond(
            feature_id="feature-login", context={}, history=[], message="Answer it."
        )


@pytest.mark.asyncio
async def test_a_complete_retry_action_is_supported_and_uses_the_control_plane() -> None:
    """Chat retry is the same audited repository grant as the ordinary REST control."""
    app, _ = await started_app(
        [
            {
                "reply": "I can grant the backend one more attempt.",
                "action": {
                    "type": "RETRY_WORKSTREAM",
                    "arguments": {
                        "repository_id": "backend",
                        "additional_attempts": 1,
                        "requested_by": "A. Operator",
                        "reason": "The repository setup was repaired.",
                    },
                    "summary": "Grant backend one more attempt.",
                },
            }
        ]
    )
    record = await app.state.feature_control_plane.get_record("feature-login")
    calls: list[dict[str, Any]] = []

    async def retry(feature_id: str, repository_id: str, **kwargs: Any) -> Any:
        calls.append({"feature_id": feature_id, "repository_id": repository_id, **kwargs})
        return record

    app.state.feature_control_plane.retry_workstream = retry

    headers = {"Authorization": "Bearer chat-key"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        sent = await client.post(
            "/features/feature-login/chat", headers=headers, json={"message": "Retry backend."}
        )
        proposal = sent.json()["messages"][1]
        confirmed = await client.post(
            f"/features/feature-login/chat/{proposal['id']}/confirm", headers=headers
        )

    assert proposal["proposed_action"]["type"] == "RETRY_WORKSTREAM"
    assert confirmed.status_code == 200, confirmed.text
    assert calls[0]["feature_id"] == "feature-login"
    assert calls[0]["repository_id"] == "backend"
    assert calls[0]["additional_attempts"] == 1
    # Model-proposed attribution is untrusted. The route's authenticated actor wins.
    assert calls[0]["requested_by"] == "platform-admin"
    assert calls[0]["reason"] == "The repository setup was repaired."


@pytest.mark.asyncio
async def test_confirming_a_message_from_another_feature_is_refused() -> None:
    """A message id must not reach across features.

    The refusal is `WorkflowNotFoundError` rather than `ChatActionError` since workspaces
    became isolated: `confirm` now checks the feature is visible *before* reading the
    transcript, so a caller naming a feature they cannot reach is told the feature is not
    there rather than told what the message on it does or does not carry.
    """
    app, service = await started_app([{"reply": "ok", "action": None}])
    del app

    with pytest.raises(WorkflowNotFoundError):
        await service.confirm(
            "some-other-feature",
            1,
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
            actor_id="platform-admin",
        )


def test_context_names_large_artifacts_instead_of_including_them() -> None:
    """A completed feature carries roughly 400 KB of artifacts; sending them all is not context.

    The relevant paragraph gets lost among forty documents, and every message pays for it.
    """
    from api.feature_control_plane import _initial_feature_state
    from api.feature_schemas import StartFeatureRequest

    request = StartFeatureRequest.model_validate(feature_payload())
    state = _initial_feature_state("feature-login", request)

    context = build_context(state, events=[])

    assert context["feature"]["feature_id"] == "feature-login"
    assert [item["repository_id"] for item in context["repositories"]] == ["backend", "frontend"]
    # Artifacts are listed by identity; only the small high-value ones are inlined.
    assert "artifacts_available" in context
    assert isinstance(context["artifacts_inline"], list)


@pytest.mark.asyncio
async def test_context_selects_the_artifact_kind_the_person_asked_to_explain() -> None:
    """Artifact explanations need the artifact content, not merely a promise that it exists."""
    app, _ = await started_app([{"reply": "ok", "action": None}])
    record = await app.state.feature_control_plane.get_record("feature-login")

    context = build_context(record.state, events=[], query="Why did the repository review fail?")

    inline_types = {item["artifact_type"] for item in context["artifacts_inline"]}
    assert "review" in inline_types
    # Validation evidence is also available without sending the entire feature history.
    assert all("current_validation_results" in item for item in context["workstreams"])


@pytest.mark.asyncio
async def test_context_keeps_an_exact_historical_artifact_when_a_type_is_also_named() -> None:
    """An exact attempt id must not be replaced by the latest artifact of the same type."""
    app, _ = await started_app([{"reply": "ok", "action": None}])
    record = await app.state.feature_control_plane.get_record("feature-login")
    requested = next(item for item in record.state.artifacts if item.artifact_type == "review")
    repository_id = requested.artifact_id.split(".")[1]
    newer = requested.model_copy(
        update={"artifact_id": f"007_review.{repository_id}.attempt-99.json"}
    )
    record.state.artifacts.append(newer)
    context = build_context(
        record.state,
        events=[],
        query=f"Explain review {requested.artifact_id}",
    )

    assert requested.artifact_id in {item["artifact_id"] for item in context["artifacts_inline"]}


@pytest.mark.asyncio
async def test_chat_is_unavailable_rather_than_faked_when_no_assistant_is_configured() -> None:
    """A deployment without a model must say so, not answer from nothing."""
    app = create_app(platform_api_key="chat-key")
    headers = {"Authorization": "Bearer chat-key", "Idempotency-Key": "chat-none"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        response = await client.get("/features/feature-login/chat", headers=headers)

    assert response.status_code == 503


def test_wiring_chat_needs_no_credential_and_the_request_supplies_it() -> None:
    """Chat must not need a key in the process, and must use the one in the request.

    This platform keeps provider credentials request-scoped, and `docker-compose.yml` passes
    the model names and deliberately not OPENAI_API_KEY. An assistant built once at startup
    therefore had no key to use: the container restart-looped on `an OpenAI API key must be
    supplied`, taking down every feature that had nothing to do with chat. Found by running
    the real image, which is also the only place the missing key existed.
    """
    from adapters.llm_adapter import LLMAdapterError, OpenAILLMClient
    from configs.model_roles import ModelRole
    from configs.settings import load_settings
    from main import build_feature_chat

    app = create_app(platform_api_key="chat-key")

    # Wiring alone must not require a credential.
    chat = build_feature_chat(
        load_settings(),
        control_plane=app.state.feature_control_plane,
        store=InMemoryChatMessageStore(),
    )

    # A request that carries a key builds an assistant from it.
    assistant = chat._assistant_for(  # noqa: SLF001 - the factory is what is under test
        RequestScopedCredentials(openai_api_key="sk-from-the-request", github_token=None),
        agent_platform="openai",
    )
    assert isinstance(assistant, FeatureAssistant)
    assistant_client = assistant._llm_client  # noqa: SLF001 - verify production role wiring
    assert isinstance(assistant_client, OpenAILLMClient)
    assert assistant_client.model_role is ModelRole.REASONING

    # A request that carries none, in a deployment that holds none, is refused -- and the
    # route turns this into a 503 before any question is recorded.
    with pytest.raises(LLMAdapterError):
        chat._assistant_for(  # noqa: SLF001
            RequestScopedCredentials(openai_api_key=None, github_token=None),
            agent_platform="openai",
        )


@pytest.mark.asyncio
async def test_a_question_with_no_usable_key_is_refused_before_it_is_recorded() -> None:
    """A transcript must not accumulate questions that nothing ever answered."""
    from adapters.llm_adapter import LLMAdapterError

    app = create_app(platform_api_key="chat-key")
    store = InMemoryChatMessageStore()

    def refuse(credentials: RequestScopedCredentials, *, agent_platform: str) -> FeatureAssistant:
        del credentials, agent_platform
        msg = "an OpenAI API key must be supplied in X-OpenAI-Api-Key or OPENAI_API_KEY"
        raise LLMAdapterError(msg)

    service = FeatureChatService(
        assistant_for=refuse,
        control_plane=app.state.feature_control_plane,
        store=store,
    )
    # A real feature, because the assistant is now built on the feature's own platform and
    # the feature is therefore read first. Against a feature that does not exist this would
    # refuse for that reason instead, and would not exercise the ordering under test.
    headers = {"Authorization": "Bearer chat-key", "Idempotency-Key": "chat-refusal"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        await client.post("/features/start", headers=headers, json=feature_payload())
        await settle(app)

    with pytest.raises(LLMAdapterError):
        await service.send(
            "feature-login",
            "What happened?",
            credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
        )

    assert await store.history("feature-login", limit=20) == []


@pytest.mark.asyncio
async def test_the_transcript_actually_survives_the_request_that_wrote_it(tmp_path: Path) -> None:
    """A message must still be there when the next request reads it.

    Every test above used the in-memory store, so nothing exercised the durable one. It
    flushed without committing: `Database.session` rolls back on error and otherwise closes
    without committing, so each message got an identifier, was returned looking complete, and
    was discarded when the request ended. Live, the assistant answered and the transcript
    stayed empty.
    """
    from storage.chat_store import DatabaseChatMessageStore
    from storage.db import Database

    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'chat.db'}")
    await database.create_schema()
    try:
        store = DatabaseChatMessageStore(database)

        question = await store.append("feature-login", ChatMessage(role="user", content="Why?"))
        answer = await store.append(
            "feature-login",
            ChatMessage(
                role="assistant",
                content="Because the backend never installed its dependencies.",
                proposed_action={"type": "CANCEL_WORKFLOW", "arguments": {}, "summary": "Cancel."},
                action_status="pending",
            ),
        )

        # A separate read, as a later request would make.
        history = await store.history("feature-login", limit=20)
        assert [item.content for item in history] == [question.content, answer.content]
        assert history[1].proposed_action is not None

        claimed = await store.claim_action("feature-login", answer.id or 0)
        assert claimed is not None
        assert claimed.action_status == "executing"
        # A second request loses the atomic claim before any workflow side effect starts.
        assert await store.claim_action("feature-login", answer.id or 0) is None

        outcome = await store.record_action_outcome(
            "feature-login", answer.id or 0, status="rejected", result="Rejected by the operator."
        )
        assert outcome.action_status == "rejected"
        # And the outcome survives too, rather than reverting to pending on the next read.
        reread = await store.get("feature-login", answer.id or 0)
        assert reread is not None
        assert reread.action_status == "rejected"
        # A message id must not reach another feature's transcript.
        assert await store.get("another-feature", answer.id or 0) is None
    finally:
        await database.drop_schema()
        await database.dispose()


@pytest.mark.asyncio
async def test_a_safely_recovered_action_returns_to_pending_in_chat() -> None:
    """A recovered action must not display as executing when no executor owns it."""
    app, service = await started_app(
        [
            {
                "reply": "I can cancel it.",
                "action": {
                    "type": "CANCEL_WORKFLOW",
                    "arguments": {"reason": "no longer needed"},
                    "summary": "Cancel this feature.",
                },
            }
        ]
    )
    turns = await service.send(
        "feature-login",
        "Cancel it.",
        credentials=RequestScopedCredentials(openai_api_key=None, github_token=None),
    )
    message = turns[1]
    assert message.id is not None
    actions = app.state.feature_actions
    action, _ = await actions.submit(
        feature_id="feature-login",
        action_type="CANCEL_WORKFLOW",
        actor_id="user-1",
        payload={"reason": "no longer needed"},
        context_version=None,
        origin="chat",
        origin_message_id=message.id,
    )
    await service._store.attach_action(  # noqa: SLF001 - persisted crash fixture
        "feature-login", message.id, action_id=action.action_id, status="executing"
    )

    refreshed = await service.history("feature-login")

    assert refreshed[-1].action_status == "pending"
    assert refreshed[-1].action_id == action.action_id
