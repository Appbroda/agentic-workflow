"""Artifact-producing product-management node."""

from __future__ import annotations

import json
from base64 import b64encode
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import ValidationError

from adapters.llm_adapter import ImageInput, LLMClient, LLMResponse
from agents.shared.contracts import (
    ARTIFACT_FILENAMES,
    AgentArtifactError,
    artifact_json,
    artifact_update,
    create_artifact,
    execution_metadata,
    parse_model_json,
    require_artifact,
    safe_error_diagnostics,
)
from agents.shared.design_snapshot import design_request_context, design_snapshot_in_state
from artifacts.schemas import PRDArtifact, Requirement, TechnicalPRDArtifact
from prompts.prompt_loader import PromptLoader
from state.models import AgentState
from tools.requirement_reconciliation import (
    RECONCILIATION_RESPONSE_KEYS,
    AnsweredClarification,
    ReconciliationRejected,
    RequirementReconciliation,
    parse_reconciliation,
)

_STAGE = "product_manager"
_RECONCILE_PROMPT = "product_manager/reconcile_v1.jinja2"


class AttachmentContentSource(Protocol):
    """The one thing this agent needs from the attachment store: bytes, by id.

    Narrowed to a single method on purpose. The agent has no business listing, binding,
    deleting or purging anything, and a dependency typed as the whole store would let it.
    """

    async def get_content(self, attachment_id: str) -> bytes | None:
        """Return one attachment's bytes, or nothing when they are gone."""


def _safe_rejection(error: AgentArtifactError | ValidationError | ReconciliationRejected) -> str:
    """Return a recordable reason for a rejected response, never the value rejected."""
    diagnostics = safe_error_diagnostics(error)
    return "; ".join(diagnostics) or "technical PRD response failed schema validation"


def _reconciliation_input(
    technical_prd: TechnicalPRDArtifact,
    requirements: Sequence[Requirement],
    answers: Sequence[AnsweredClarification],
) -> str:
    """Carry the requirement text and the human's answers, both exactly as they stand.

    The answers travel with the questions they answer: `apply_clarification_answers` empties
    `unresolved_questions` when it records them, and "use compensating deletes" settles
    nothing without the question it settles. Priority travels as context and is never asked
    for back -- a field absent from the response is a field a reconciliation cannot change.

    Everything goes in the user message rather than the instructions. Both adapters reject an
    empty input before any request is made, which is how the grounding call once failed in
    zero seconds on six consecutive runs.
    """
    return json.dumps(
        {
            "feature": {
                "title": technical_prd.title,
                "solution_summary": technical_prd.solution_summary,
            },
            "requirements": [
                {
                    "requirement_id": requirement.requirement_id,
                    "priority": requirement.priority,
                    "description": requirement.description,
                    "dependencies": list(requirement.dependencies),
                    "acceptance_criteria": list(requirement.acceptance_criteria),
                }
                for requirement in requirements
            ],
            "accepted_answers": [
                {
                    "question_id": answer.question_id,
                    "question": answer.question,
                    "answer": answer.answer,
                }
                for answer in answers
            ],
        },
        indent=2,
    )


class ProductManagerAgent:
    """Transform a submitted PRD artifact into the technical PRD workflow artifact."""

    def __init__(
        self,
        *,
        prompt_loader: PromptLoader,
        llm_client: LLMClient,
        attachments: AttachmentContentSource | None = None,
    ) -> None:
        """Inject the versioned prompt source, the model boundary, and the image source.

        ``attachments`` is absent in every composition that cannot serve bytes -- an isolated
        application, a mock run. A PRD that names images and a composition that cannot fetch
        them fails the step; it does not run without them. That is 61-'s rule with the
        volume turned up: a drop is a drop, and this one is a submission answered from prose
        the author supplied as a caption to a picture.
        """
        self._prompt_loader = prompt_loader
        self._llm_client = llm_client
        self._attachments = attachments

    async def _images(self, prd: PRDArtifact) -> tuple[ImageInput, ...]:
        """Fetch and encode the submission's images, or fail the step saying which is missing.

        Never silent. Three ways this can go wrong and all three end the step with the marker
        in the diagnostic, because the alternative is a technical PRD derived from a
        description of a picture nobody looked at:

        - the composition has no way to fetch bytes,
        - the row's content has been purged, or
        - the model this feature runs on cannot be shown an image.

        The last is the call-site fence. It is not merely defensive: for a tier feature the
        deployment can change between acceptance and the run, and a redeploy that swaps the
        tier's reasoning model for one not declared vision-capable makes this the only thing
        standing between the feature and a silent drop.
        """
        if not prd.attachments:
            return ()
        if not self._llm_client.vision_capable:
            msg = (
                "this submission attaches images and the configured reasoning model is not "
                "declared able to read them"
            )
            raise AgentArtifactError(
                msg,
                diagnostics=[
                    msg,
                    "images not shown: "
                    + ", ".join(f"[image:{item.marker}]" for item in prd.attachments),
                    "declare the model in MODEL_VISION_CAPABLE, or run this feature on one "
                    "that is already declared",
                ],
            )
        if self._attachments is None:
            msg = "this submission attaches images and no attachment source is configured"
            raise AgentArtifactError(
                msg,
                diagnostics=[
                    msg,
                    "images not read: "
                    + ", ".join(f"[image:{item.marker}]" for item in prd.attachments),
                ],
            )
        images: list[ImageInput] = []
        for attachment in prd.attachments:
            content = await self._attachments.get_content(attachment.attachment_id)
            if not content:
                msg = f"the image [image:{attachment.marker}] has no content to read"
                raise AgentArtifactError(
                    msg,
                    diagnostics=[
                        msg,
                        f"{attachment.filename} was submitted with this feature and its "
                        "stored content is gone; the technical PRD would be derived from "
                        "the prose alone",
                    ],
                )
            images.append(
                ImageInput(
                    media_type=attachment.media_type,
                    data=b64encode(content).decode("ascii"),
                )
            )
        return tuple(images)

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Publish ``002_technical_prd.json`` as the only handoff to later agents.

        One bounded repair, on the same terms the feature planner already gets. This is the
        first model call of a feature and it had no repair at all, so a single value outside
        an enum ended the whole run before a repository was read: AB-Feature-120 died
        twenty-four seconds in on four rejected `priority` values, with nothing else wrong.
        The prompt states the allowed values as `must|should|could|wont`, which is also the
        shape of a placeholder -- so a model copying it verbatim produces exactly this.
        """
        prd = require_artifact(state, PRDArtifact, artifact_id=ARTIFACT_FILENAMES["prd"])
        images = await self._images(prd)
        # The frames the author attached, read as part of the request rather than as a note
        # about it. Unscoped, because no workstream exists yet: this role is deriving the
        # requirements the design implies, and there is nothing to scope by. `None` for every
        # feature that cited nothing, and the template renders nothing for it -- so those
        # prompts are unchanged byte for byte.
        design = design_request_context(design_snapshot_in_state(state))
        instructions = self._prompt_loader.render(
            "product_manager/v1.jinja2",
            workflow_id=state["workflow_id"],
            prd=artifact_json(prd),
            # The artifact's own order, which is the order the route resolved and the order
            # the images are sent in. The template says so and the adapter is handed the same
            # list, so the numbering a model reads is the numbering it is shown.
            attachments=[item.model_dump(mode="python") for item in prd.attachments],
            design_snapshot=(
                "" if design is None else json.dumps(design, indent=2, sort_keys=True)
            ),
        )
        input_text = artifact_json(prd)
        response = await self._llm_client.respond(
            instructions=instructions, input_text=input_text, images=images
        )
        try:
            return self._technical_prd(state, prd=prd, response=response)
        except (AgentArtifactError, ValidationError) as error:
            repair_response = await self._llm_client.respond(
                instructions=self._repair_instructions(instructions, response, error),
                input_text=input_text,
                # The repair sees what the first attempt saw. A repair asked to fix a
                # schema violation while blind to the images would be asked to rewrite a
                # document it can no longer read the source of.
                images=images,
            )
            try:
                return self._technical_prd(
                    state,
                    prd=prd,
                    response=repair_response,
                    extra_metadata={
                        "repair_of_response_id": response.response_id,
                        "repair_reason": _safe_rejection(error),
                    },
                )
            except (AgentArtifactError, ValidationError) as repair_error:
                msg = "product manager could not produce a valid technical PRD after one repair"
                # Both rejections describe the violated schema and no rejected value, so both
                # are safe to persist -- and they are the only record of why a feature ended
                # before it wrote an artifact.
                raise AgentArtifactError(
                    msg,
                    diagnostics=[
                        f"first attempt rejected: {_safe_rejection(error)}",
                        f"repair attempt rejected: {_safe_rejection(repair_error)}",
                    ],
                    stage=_STAGE,
                ) from repair_error

    async def reconcile_requirements(
        self,
        *,
        workflow_id: str,
        technical_prd: TechnicalPRDArtifact,
        answers: Sequence[AnsweredClarification],
    ) -> RequirementReconciliation:
        """Restate the requirements a person's answers settled, and report what they cannot.

        The answers are recorded verbatim in the PRD's metadata and reach the plan, but the
        requirement descriptions, dependencies and acceptance criteria -- the text the planner
        scopes and the reviewer enforces -- used to keep saying whatever they said before the
        person answered. AB-Feature-200 therefore ran with two specifications: a PRD demanding
        transactional rollback and a plan, derived from the same answer, compensating without
        transactions. Its engineer built both, its self-review proved the hybrid incoherent
        four attempts running, and the workstream stopped with nothing to show. This call is
        one model call spent where four attempts were.

        This agent's, and not the planner's, because the PRD is this agent's authorship. One
        bounded repair, on the same terms `run` gets: the structural rules are stated in the
        prompt, so a response that breaks one is a response to correct rather than a feature
        to end -- but a repair that breaks them too ends the stage loudly. Proceeding with the
        unreconciled PRD as a fallback is the defect this exists to remove.
        """
        requirements = [
            *technical_prd.functional_requirements,
            *technical_prd.non_functional_requirements,
        ]
        instructions = self._prompt_loader.render(_RECONCILE_PROMPT, workflow_id=workflow_id)
        input_text = _reconciliation_input(technical_prd, requirements, answers)
        response = await self._llm_client.respond(instructions=instructions, input_text=input_text)
        try:
            return self._reconciliation(response, requirements=requirements, answers=answers)
        except (AgentArtifactError, ReconciliationRejected) as error:
            repair_response = await self._llm_client.respond(
                instructions=(
                    f"{instructions}\n\n"
                    "Your previous response was rejected by deterministic validation. Here is "
                    f"your previous response:\n{response.output_text}\n\n"
                    f"Here is exactly what was wrong with it:\n{error}\n\n"
                    "Return the complete response again with only what the error names "
                    "changed. Every requirement you had already restated correctly must be "
                    "byte-identical to your previous response. Do not add, remove or rename a "
                    "requirement, and do not restate a requirement no answer touches."
                ),
                input_text=input_text,
            )
            try:
                return self._reconciliation(
                    repair_response, requirements=requirements, answers=answers
                )
            except (AgentArtifactError, ReconciliationRejected) as repair_error:
                msg = "product manager could not reconcile the requirements after one repair"
                raise AgentArtifactError(
                    msg,
                    diagnostics=[
                        f"first attempt rejected: {_safe_rejection(error)}",
                        f"repair attempt rejected: {_safe_rejection(repair_error)}",
                    ],
                    stage=_STAGE,
                ) from repair_error

    @staticmethod
    def _reconciliation(
        response: LLMResponse,
        *,
        requirements: Sequence[Requirement],
        answers: Sequence[AnsweredClarification],
    ) -> RequirementReconciliation:
        """Validate one reconciliation response into the decision the workflow may apply."""
        return parse_reconciliation(
            parse_model_json(response.output_text, expected_keys=RECONCILIATION_RESPONSE_KEYS),
            requirements=requirements,
            answers=answers,
        )

    @staticmethod
    def _repair_instructions(
        instructions: str,
        response: LLMResponse,
        error: AgentArtifactError | ValidationError,
    ) -> str:
        """Anchor the correction to the response that was rejected, and to nothing else.

        The full error goes to the provider and never to durable state. Asking for a fresh
        response instead traded the reported defect for new ones -- the planner learned that
        when an attempt rejected for two fields came back with sixteen.
        """
        return (
            f"{instructions}\n\n"
            "Your previous response was rejected by deterministic schema validation. Here is "
            "your previous response:\n"
            f"{response.output_text}\n\n"
            f"Here is exactly what was wrong with it:\n{error}\n\n"
            "Return the complete technical PRD again with only the rejected fields changed. "
            "Every other field must be byte-identical to your previous response. Do not "
            "restructure the document, rename anything, add or remove requirements, or "
            "'improve' a field the errors did not mention, and do not introduce empty "
            "strings. Where a field accepts only specific values, use one of those values "
            "exactly -- a list of alternatives such as `must|should|could|wont` names the "
            "choices and is never itself a valid value."
        )

    @staticmethod
    def _technical_prd(
        state: AgentState,
        *,
        prd: PRDArtifact,
        response: LLMResponse,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate one model response into the single technical-PRD handoff artifact."""
        technical_prd = create_artifact(
            TechnicalPRDArtifact,
            workflow_id=state["workflow_id"],
            artifact_id=ARTIFACT_FILENAMES["technical_prd"],
            producer=_STAGE,
            payload=parse_model_json(response.output_text),
            metadata={
                "source_artifact_ids": [prd.artifact_id],
                "model": response.model,
                "response_id": response.response_id,
                "prompt_template": "product_manager/v1.jinja2",
                **execution_metadata(
                    agent_type="Product manager",
                    provider=response.provider,
                    model=response.model,
                    reasoning_effort=response.reasoning_effort,
                    model_role=response.model_role,
                    model_variable=response.model_variable,
                    routing_reason=response.routing_reason,
                ),
                **(extra_metadata or {}),
            },
        )
        return artifact_update(_STAGE, [technical_prd])


async def product_manager_node(
    state: AgentState,
    *,
    prompt_loader: PromptLoader,
    llm_client: LLMClient,
) -> dict[str, Any]:
    """Run a product-management node with explicitly injected dependencies."""
    return await ProductManagerAgent(prompt_loader=prompt_loader, llm_client=llm_client).run(state)
