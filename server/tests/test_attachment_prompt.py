"""What the product manager is told it is looking at, and what it sends (89-, Part F).

The block in `product_manager/v1.jinja2` is conditional, not a `v2.jinja2`: template names are
hardcoded literals, `PromptLoader` renders whatever is on disk, and nothing reads the recorded
`prompt_template` metadata back -- so a second file would need a selection mechanism that does
not exist. Being conditional also gives the last safety rule its guarantee by construction,
which the final test here asserts rather than assumes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from adapters.llm_adapter import ImageInput, LLMResponse
from agents.product_manager.agent import ProductManagerAgent
from agents.shared.contracts import AgentArtifactError, artifact_json
from artifacts.schemas import PRDArtifact, PRDAttachment
from prompts.prompt_loader import PromptLoader
from state.models import AgentState
from tests import attachment_fixtures as fixtures


class CapturingClient:
    """A model boundary that records what it was shown and answers a valid technical PRD."""

    def __init__(self, *, vision: bool = True) -> None:
        """Declare whether this boundary reads images, and start with nothing recorded."""
        self._vision = vision
        self.instructions: list[str] = []
        self.images: list[tuple[ImageInput, ...]] = []

    @property
    def vision_capable(self) -> bool:
        """What the deployment declared about this client's model."""
        return self._vision

    async def respond(
        self,
        *,
        instructions: str,
        input_text: str,
        images: Sequence[ImageInput] = (),
    ) -> LLMResponse:
        """Record the prompt and the images, and answer with a minimal valid payload."""
        del input_text
        self.instructions.append(instructions)
        self.images.append(tuple(images))
        return LLMResponse(
            response_id="resp-1",
            model="gpt-6-astra",
            input_tokens=10,
            output_tokens=5,
            output_text=json.dumps(
                {
                    "title": "Login audit trail",
                    "solution_summary": "Record failed authentication attempts.",
                    "functional_requirements": [
                        {
                            "requirement_id": "r1",
                            "description": "Emit an audit event on a failed login.",
                            "priority": "must",
                            "acceptance_criteria": ["A failed login emits an audit event."],
                            "dependencies": [],
                        }
                    ],
                    "non_functional_requirements": [],
                    "data_requirements": [],
                    "integration_requirements": [],
                    "security_requirements": [],
                    "assumptions": [],
                    "unresolved_questions": [],
                }
            ),
        )


class StaticAttachments:
    """An attachment source over a dictionary, so a purge is one deletion."""

    def __init__(self, content: dict[str, bytes]) -> None:
        """Bind the bytes each attachment id resolves to."""
        self._content = content
        self.reads: list[str] = []

    async def get_content(self, attachment_id: str) -> bytes | None:
        """Return the bytes for one id, recording that they were asked for."""
        self.reads.append(attachment_id)
        return self._content.get(attachment_id)


def attachment(marker: str, media_type: str = "image/png") -> PRDAttachment:
    """One recorded attachment, from the committed fixture of the given type."""
    path = {"image/png": fixtures.PNG, "image/jpeg": fixtures.JPEG}[media_type]
    return PRDAttachment(
        attachment_id=f"attachment-{marker}",
        marker=marker,
        caption=f"the {marker} screen",
        filename=f"{marker}.png",
        media_type=media_type,  # type: ignore[arg-type]
        byte_size=len(fixtures.content(path)),
        sha256=fixtures.digest(path),
    )


def prd_state(*attachments: PRDAttachment) -> tuple[AgentState, PRDArtifact]:
    """One agent state carrying a submitted PRD with the given attachments."""
    prd = PRDArtifact(
        schema_version="1.0.0",
        workflow_id="feature-1",
        artifact_id="001_prd.json",
        producer="api",
        timestamp=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
        metadata={"source": "api"},
        validation_status="valid",
        title="Login audit trail",
        problem_statement="Login fails silently [image:login-error].",
        attachments=list(attachments),
    )
    state: AgentState = {  # type: ignore[typeddict-item]
        "workflow_id": "feature-1",
        "artifacts": [prd],
    }
    return state, prd


@pytest.mark.asyncio
async def test_the_prompt_lists_the_markers_in_artifact_order_and_the_images_follow_it() -> None:
    """One order, stated in the text and used for the payload, so the numbering is true."""
    first, second = attachment("login-error"), attachment("empty-state", "image/jpeg")
    state, _ = prd_state(first, second)
    client = CapturingClient()
    source = StaticAttachments(
        {
            first.attachment_id: fixtures.content(fixtures.PNG),
            second.attachment_id: fixtures.content(fixtures.JPEG),
        }
    )

    await ProductManagerAgent(
        prompt_loader=PromptLoader(), llm_client=client, attachments=source
    ).run(state)

    prompt = client.instructions[0]
    assert "1. `[image:login-error]` — login-error.png — the login-error screen" in prompt
    assert "2. `[image:empty-state]` — empty-state.png — the empty-state screen" in prompt
    assert prompt.index("[image:login-error]") < prompt.index("[image:empty-state]")
    # The contract the block states, and the instruction the whole block exists for.
    assert "the image is the requirement" in prompt
    assert "must be written into the technical PRD as words" in prompt
    # And the images are supplied in that same order, read from the store in that order.
    assert [item.media_type for item in client.images[0]] == ["image/png", "image/jpeg"]
    assert source.reads == [first.attachment_id, second.attachment_id]


@pytest.mark.asyncio
async def test_the_images_are_base64_with_no_data_url_prefix() -> None:
    """Each provider wraps it differently, so the boundary carries it unwrapped."""
    from base64 import b64decode

    first = attachment("login-error")
    state, _ = prd_state(first)
    client = CapturingClient()
    source = StaticAttachments({first.attachment_id: fixtures.content(fixtures.PNG)})

    await ProductManagerAgent(
        prompt_loader=PromptLoader(), llm_client=client, attachments=source
    ).run(state)

    data = client.images[0][0].data
    assert not data.startswith("data:")
    assert b64decode(data) == fixtures.content(fixtures.PNG)


@pytest.mark.asyncio
async def test_a_purged_attachment_fails_the_step_with_the_marker_in_the_diagnostic() -> None:
    """A fetch failure is not silent: the technical PRD would be derived from prose alone."""
    first = attachment("login-error")
    state, _ = prd_state(first)
    client = CapturingClient()

    with pytest.raises(AgentArtifactError) as raised:
        await ProductManagerAgent(
            prompt_loader=PromptLoader(),
            llm_client=client,
            attachments=StaticAttachments({}),
        ).run(state)

    assert "[image:login-error]" in str(raised.value)
    # And no model call was made on the prose alone.
    assert client.instructions == []


@pytest.mark.asyncio
async def test_the_call_site_fence_raises_for_a_client_that_cannot_read_images() -> None:
    """The redeploy case: acceptance passed, and the tier's model changed underneath."""
    first = attachment("login-error")
    state, _ = prd_state(first)
    client = CapturingClient(vision=False)

    with pytest.raises(AgentArtifactError) as raised:
        await ProductManagerAgent(
            prompt_loader=PromptLoader(),
            llm_client=client,
            attachments=StaticAttachments({first.attachment_id: fixtures.content(fixtures.PNG)}),
        ).run(state)

    assert "not declared able to read" in str(raised.value)
    assert client.instructions == []


@pytest.mark.asyncio
async def test_a_composition_with_no_attachment_source_fails_rather_than_running_blind() -> None:
    """A PRD naming images and nothing able to fetch them is a failed step, not a quiet one."""
    first = attachment("login-error")
    state, _ = prd_state(first)
    client = CapturingClient()

    with pytest.raises(AgentArtifactError, match="no attachment source"):
        await ProductManagerAgent(
            prompt_loader=PromptLoader(), llm_client=client, attachments=None
        ).run(state)

    assert client.instructions == []


@pytest.mark.asyncio
async def test_the_v1_prompt_for_a_prd_with_no_attachments_renders_unchanged() -> None:
    """The conditional block gives the last safety rule its guarantee by construction."""
    state, prd = prd_state()
    client = CapturingClient()

    await ProductManagerAgent(prompt_loader=PromptLoader(), llm_client=client).run(state)

    without_the_keyword = PromptLoader().render(
        "product_manager/v1.jinja2", workflow_id="feature-1", prd=artifact_json(prd)
    )
    assert client.instructions[0] == without_the_keyword
    assert "Images attached to this submission" not in client.instructions[0]
    # And no images travelled.
    assert client.images[0] == ()
