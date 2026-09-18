"""Pre-planning reconnaissance: what one checkout contains, before the plan assumes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from adapters.llm_adapter import LLMClient
from agents.shared.contracts import (
    FEATURE_ARTIFACT_FILENAMES,
    AgentArtifactError,
    create_artifact,
    execution_metadata,
    parse_model_json,
    safe_error_diagnostics,
)
from artifacts.schemas import (
    ClarificationQuestion,
    RepositoryReconnaissanceArtifact,
    TechnicalPRDArtifact,
)
from prompts.prompt_loader import PromptLoader
from tools.repository_reconnaissance import (
    RepositoryReconnaissanceEvidence,
    inspect_repository_for_planning,
)

_RESPONSE_KEYS = (
    "summary",
    "source_areas",
    "test_areas",
    "conventions",
    "shared_utilities",
    "contradicted_premises",
)
# Path-bearing fields whose every value must name a file the scan actually read. A convention
# is only worth having if it points at real code: a hallucinated wiring path sends the next
# engineer to edit a file that does not exist, which is the failure mode this agent exists to
# remove rather than relocate.
_PATH_FIELDS = ("source_areas", "test_areas")


class RepositoryReconAgent:
    """Turn read-only checkout evidence into the grounded facts a feature plan must use."""

    def __init__(self, *, prompt_loader: PromptLoader, llm_client: LLMClient) -> None:
        """Inject the versioned prompt source and configured reasoning-model boundary."""
        self._prompt_loader = prompt_loader
        self._llm_client = llm_client

    async def inspect(
        self,
        *,
        feature_id: str,
        repository_id: str,
        workspace_root: str,
        technical_prd: TechnicalPRDArtifact,
    ) -> RepositoryReconnaissanceArtifact:
        """Publish one repository's reconnaissance artifact from its checkout."""
        evidence = inspect_repository_for_planning(workspace_root, repository_id=repository_id)
        instructions = self._prompt_loader.render(
            "recon/v1.jinja2",
            feature_id=feature_id,
            repository_id=repository_id,
            technical_prd=technical_prd.model_dump_json(indent=2),
            evidence=evidence.model_dump_json(indent=2),
        )
        response = await self._llm_client.respond(
            instructions=instructions,
            input_text=evidence.model_dump_json(indent=2),
        )
        metadata: dict[str, Any] = {
            "source_artifact_ids": [technical_prd.artifact_id],
            "model": response.model,
            "response_id": response.response_id,
            "prompt_template": "recon/v1.jinja2",
            "repository_id": repository_id,
            "head_sha": evidence.head_sha,
            "technology_profile": evidence.technology.model_dump(mode="json"),
            # Structural, not modelled: the files that pull in the most of this repository's
            # own modules, most first. A convention's `wiring_path` says the same thing when
            # the model fills it in, and it often does not -- -106's backend produced seven
            # conventions and no wiring path at all, for the repository that then failed the
            # wiring gate on three separate attempts. This is the floor under that.
            "wiring_files": [item.path for item in evidence.wiring_files],
            **execution_metadata(
                agent_type="Repository analyst",
                provider=response.provider,
                model=response.model,
                reasoning_effort=response.reasoning_effort,
                model_role=response.model_role,
                model_variable=response.model_variable,
                routing_reason=response.routing_reason,
            ),
        }
        try:
            return self._artifact(
                feature_id=feature_id,
                repository_id=repository_id,
                evidence=evidence,
                payload=parse_model_json(response.output_text, expected_keys=_RESPONSE_KEYS),
                metadata=metadata,
            )
        except (AgentArtifactError, ValidationError) as error:
            # One bounded repair, matching the planner and reviewer. A reconnaissance pass
            # rejected for a single unknown path must not fail the feature before it has been
            # planned: the alternative is planning blind, which is the state this replaces.
            repair = await self._llm_client.respond(
                instructions=(
                    f"{instructions}\n\n"
                    "Your previous response was rejected by deterministic validation. Here is "
                    f"your previous response:\n{response.output_text}\n\n"
                    f"Here is exactly what was wrong with it:\n{error}\n\n"
                    "Return the complete response again with only the fields named above "
                    "changed. Every other field must be byte-identical to your previous "
                    "response. Every path must appear in the evidence; remove any finding "
                    "you cannot support with a path you actually read rather than "
                    "substituting a path that looks plausible."
                ),
                input_text=evidence.model_dump_json(indent=2),
            )
            return self._artifact(
                feature_id=feature_id,
                repository_id=repository_id,
                evidence=evidence,
                payload=parse_model_json(repair.output_text, expected_keys=_RESPONSE_KEYS),
                metadata={
                    **metadata,
                    "model": repair.model,
                    "response_id": repair.response_id,
                    "repair_of_response_id": response.response_id,
                    "repair_reason": "; ".join(safe_error_diagnostics(error))
                    or "reconnaissance response failed deterministic validation",
                    **execution_metadata(
                        agent_type="Repository analyst",
                        provider=repair.provider,
                        model=repair.model,
                        reasoning_effort=repair.reasoning_effort,
                        model_role=repair.model_role,
                        model_variable=repair.model_variable,
                        routing_reason=repair.routing_reason,
                    ),
                },
            )

    async def suggest_answers(
        self,
        *,
        feature_id: str,
        questions: Sequence[ClarificationQuestion],
        reconnaissance: Sequence[RepositoryReconnaissanceArtifact],
    ) -> list[ClarificationQuestion]:
        """Answer the open questions the checkouts already settle, and leave the rest alone.

        The product manager writes its questions before any repository has been read, so it
        asks about conventions and existing behaviour it has no way to know. By this point the
        repositories *have* been read -- that is what `reconnaissance` is -- and asking a
        person to go and find what the platform just recorded is asking them to do the work
        twice.

        Every returned question is the one that came in, with a suggestion attached or
        without. A suggestion is kept only when it names a repository this feature inspected
        and cites paths that appear in that repository's findings; the model is told to omit
        what the evidence does not answer, and anything it makes up anyway is dropped here
        rather than shown as though the platform had read it.
        """
        open_questions = [item for item in questions if not item.suggested_answer.strip()]
        if not open_questions or not reconnaissance:
            return list(questions)
        instructions = self._prompt_loader.render(
            "recon/questions_v1.jinja2",
            feature_id=feature_id,
        )
        # The evidence travels as the user message, like every other reconnaissance call in
        # this module. It used to be baked into the instructions with `input_text=""` -- and
        # both adapters reject an empty input before any request is made, so grounding failed
        # in zero seconds on every feature that ever reached it (six for six in the 175-180
        # matrix) while its best-effort fallback quietly asked the human instead.
        response = await self._llm_client.respond(
            instructions=instructions,
            input_text=json.dumps(
                {
                    "open_questions": [
                        {"question_id": item.question_id, "question": item.question}
                        for item in open_questions
                    ],
                    "reconnaissance": [_findings(item) for item in reconnaissance],
                },
                indent=2,
            ),
        )
        try:
            payload = parse_model_json(response.output_text, expected_keys=("answers",))
        except AgentArtifactError:
            # An unusable response costs the suggestions and nothing else. This runs between
            # a feature being planned and a person being asked, and failing it would turn an
            # improvement to the question into a new way for the feature to stop.
            return list(questions)
        grounded = _grounded_answers(payload.get("answers"), reconnaissance)
        return [
            item.model_copy(update=grounded[item.question_id])
            if item.question_id in grounded
            else item
            for item in questions
        ]

    def _artifact(
        self,
        *,
        feature_id: str,
        repository_id: str,
        evidence: RepositoryReconnaissanceEvidence,
        payload: dict[str, Any],
        metadata: dict[str, Any],
    ) -> RepositoryReconnaissanceArtifact:
        """Bind the model's reading to paths the scan actually read, then seal the artifact."""
        _require_known_paths(payload, evidence)
        return create_artifact(
            RepositoryReconnaissanceArtifact,
            workflow_id=feature_id,
            artifact_id=_repository_artifact_id(repository_id),
            producer="repository_recon",
            payload={
                **payload,
                "feature_id": feature_id,
                "repository_id": repository_id,
                "repository_revision": evidence.repository_revision,
            },
            metadata=metadata,
        )


def _findings(artifact: RepositoryReconnaissanceArtifact) -> dict[str, Any]:
    """Show one repository's evidence, minus the premises already turned into questions."""
    return {
        "repository_id": artifact.repository_id,
        "summary": artifact.summary,
        "source_areas": list(artifact.source_areas),
        "test_areas": list(artifact.test_areas),
        "conventions": [item.model_dump(mode="json") for item in artifact.conventions],
        "shared_utilities": list(artifact.shared_utilities),
    }


def _grounded_answers(
    answers: object, reconnaissance: Sequence[RepositoryReconnaissanceArtifact]
) -> dict[str, dict[str, Any]]:
    """Keep the answers the evidence supports and silently drop the rest.

    Two checks, both about provenance rather than about the answer being right. It must name a
    repository this feature actually inspected, and it must cite at least one path that
    appears in that repository's findings -- because a suggestion the reader cannot trace back
    is indistinguishable from one they typed themselves, which is how a guess becomes a
    requirement.
    """
    known = {item.repository_id: item for item in reconnaissance}
    kept: dict[str, dict[str, Any]] = {}
    for entry in _mappings(answers):
        question_id = str(entry.get("question_id", "")).strip()
        answer = str(entry.get("answer", "")).strip()
        repository_id = str(entry.get("repository_id", "")).strip()
        artifact = known.get(repository_id)
        if not question_id or not answer or artifact is None:
            continue
        # Every path this repository's findings actually named. A convention's evidence and
        # wiring paths count: they are the part of the reconnaissance a reader would follow.
        readable = {path for field in _PATH_FIELDS for path in getattr(artifact, field)}
        readable |= set(artifact.shared_utilities)
        for convention in artifact.conventions:
            readable |= set(convention.evidence_paths)
            if convention.wiring_path:
                readable.add(convention.wiring_path)
        # Exact matches only. A path *under* a recorded source area is not proof the scan
        # read that file -- `src` is most of a repository, so a prefix rule would accept any
        # filename the model cared to invent while looking exactly as traceable.
        cited = [path for path in _strings(entry.get("evidence_paths")) if path in readable]
        if not cited:
            continue
        confidence = str(entry.get("confidence", "")).strip().lower()
        kept[question_id] = {
            "suggested_answer": answer,
            "suggestion_source": (
                f"Suggested from repository analysis of {repository_id} ({', '.join(cited)})"
            ),
            "suggestion_confidence": (
                confidence if confidence in {"high", "medium", "low"} else "medium"
            ),
        }
    return kept


def _strings(value: object) -> list[str]:
    """Read a list of strings from a model response without trusting its shape."""
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _repository_artifact_id(repository_id: str) -> str:
    """Give each repository its own reconnaissance artifact within one feature."""
    stem = FEATURE_ARTIFACT_FILENAMES["repository_reconnaissance"].removesuffix(".json")
    return f"{stem}.{repository_id}.json"


def _require_known_paths(
    payload: dict[str, Any], evidence: RepositoryReconnaissanceEvidence
) -> None:
    """Reject any path the checkout scan did not produce.

    Without this the agent can restate the same guesses it was built to replace, in an
    artifact the planner is now required to trust -- which would be worse than no
    reconnaissance at all, because the guess would arrive carrying evidence's authority.
    """
    known_files = set(evidence.file_inventory)
    known_directories = {
        *evidence.source_directories,
        *evidence.test_directories,
        *{path.rsplit("/", maxsplit=1)[0] for path in known_files if "/" in path},
    }
    known = known_files | known_directories
    unknown: set[str] = set()
    for field in _PATH_FIELDS:
        values = payload.get(field)
        if isinstance(values, list):
            unknown.update(item for item in values if isinstance(item, str) and item not in known)
    for item in _mappings(payload.get("conventions")):
        paths = item.get("evidence_paths")
        if isinstance(paths, list):
            unknown.update(
                value for value in paths if isinstance(value, str) and value not in known
            )
        wiring = item.get("wiring_path")
        if isinstance(wiring, str) and wiring not in known:
            unknown.add(wiring)
    for item in _mappings(payload.get("contradicted_premises")):
        paths = item.get("evidence_paths")
        if isinstance(paths, list):
            unknown.update(
                value for value in paths if isinstance(value, str) and value not in known
            )
    if unknown:
        msg = (
            "reconnaissance response references paths absent from the checkout: "
            f"{json.dumps(sorted(unknown))}"
        )
        raise AgentArtifactError(msg)


def _mappings(value: object) -> list[dict[str, Any]]:
    """Return only the well-formed objects in a model-supplied list."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


__all__ = ["RepositoryReconAgent"]
