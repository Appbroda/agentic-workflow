"""Shared artifact-only contracts used by specialized workflow agent nodes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from artifacts.schemas import Artifact, BaseArtifact
from services.process_runner import redact_source_credentials
from state.enums import WorkflowStatus
from state.models import AgentState
from workflow_schema import WORKFLOW_SCHEMA_VERSION

SCHEMA_VERSION = WORKFLOW_SCHEMA_VERSION

ARTIFACT_FILENAMES = {
    "prd": "001_prd.json",
    "technical_prd": "002_technical_prd.json",
    "architecture": "003_architecture.json",
    "execution_graph": "004_execution_graph.json",
    "task_plan": "005_task_plan.json",
    "code_completion": "006_code_completion.json",
    "review": "007_review.json",
    "pull_request": "008_pull_request.json",
}

FEATURE_ARTIFACT_FILENAMES = {
    "integration_contract": "009_integration_contract.json",
    "repository_execution_plan": "010_repository_execution_plan.json",
    "child_workflow_result": "011_child_workflow_result.json",
    "integration_review": "012_integration_review.json",
    "contract_change_request": "013_contract_change_request.json",
    "feature_completion": "014_feature_completion.json",
    # Produced before the planner runs, so it is numbered last but ordered first in time.
    "repository_reconnaissance": "015_repository_reconnaissance.json",
    # One per repository per diagnosis rather than one per feature, so the identifier carries
    # the repository and the repair. A feature may have several open at once.
    "repository_repair_proposal": "016_repository_repair_proposal.json",
    # One per repository per re-litigated demand, for the same reason: two repositories of one
    # feature can each be stopped on a design question, and each identifier carries both.
    "design_conflict": "017_design_conflict.json",
    # Produced before the product manager runs, so it is numbered last and ordered first in
    # time -- for `repository_reconnaissance`'s reason and one more. The design is part of the
    # *request*: requirements derived from a problem statement should be derived from the
    # frames too, and a design that arrives after the requirements are written can only ever
    # contradict them.
    "design_snapshot": "018_design_snapshot.json",
    # One per repository, written when that repository's workstream starts rather than with
    # the snapshot above -- which is why it is numbered after it despite being resolved later.
    # The snapshot is the whole request's design, indexed, and exists before the planner runs;
    # this is one workstream's own frames at build fidelity, and cannot exist until the plan
    # has said which frames those are. Keyed by repository so two workstreams never collide.
    "design_detail": "018_design_detail.json",
    # One per revision, qualified `.v{n}`: a person can ask for changes again after the
    # revision completes, and each request is its own immutable record.
    "feature_revision_request": "019_feature_revision_request.json",
}


# The one metadata key that says what executed the work an artifact records. Agents used to
# each invent their own -- `model`, `coding_model`, `seam_review_model` -- so anything reading
# "which model produced this" had to know all of them, and a new agent silently recorded
# nothing readable. Those keys are still written for the readers that already depend on them;
# this block is the one a reader should consult.
EXECUTION_METADATA_KEY = "execution"


def execution_metadata(
    *,
    agent_type: str,
    provider: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    model_role: str | None = None,
    model_variable: str | None = None,
    routing_reason: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict[str, Any]:
    """Record what actually performed this artifact's work, ready to merge into metadata.

    Written at execution time and never afterwards. An artifact is immutable, so this is the
    only chance to state the truth about its execution: reconstructing it later from the
    deployment's configuration would report today's model for yesterday's work, which is the
    one error the whole execution record exists to prevent.

    Carries no credential, no base URL, no account and no prompt. The logical role is separate
    from the model on purpose -- two roles may legitimately be configured with the same model,
    and what a review is depends on the role, not on what answered it. ``routing_reason`` is a
    short platform decision, never model reasoning or chain-of-thought.
    """
    execution: dict[str, Any] = {"agent_type": agent_type}
    if provider:
        execution["provider"] = provider
    if model:
        execution["model"] = model
    if reasoning_effort:
        execution["reasoning_effort"] = reasoning_effort
    if model_role:
        execution["model_role"] = model_role
    if model_variable:
        execution["model_variable"] = model_variable
    if routing_reason:
        execution["routing_reason"] = routing_reason
    # Usage counts, when the provider reported them. Until these were written down, "what did
    # this attempt cost" had no answer anywhere in the platform: the 175-180 audit's cost
    # comparison had to fall back to wall-clock durations because every response's token
    # counts were read from the SDK and dropped. Named "usage", not "tokens": persistence
    # screens metadata keys for credential-shaped markers, and "token" is one.
    if input_tokens is not None:
        execution["usage_input"] = input_tokens
    if output_tokens is not None:
        execution["usage_output"] = output_tokens
    return {EXECUTION_METADATA_KEY: execution}


def attempt_artifact_id(base_artifact_id: str, attempt: int) -> str:
    """Return the canonical immutable artifact identifier for one zero-based attempt."""
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        msg = "artifact attempt must be a non-negative integer"
        raise ValueError(msg)
    stem = base_artifact_id.removesuffix(".json")
    return f"{stem}.attempt-{attempt}.json"


# The qualifiers a descendant artifact ID may carry. ``attempt`` numbers a retry of the same
# work; ``revision`` numbers an immutable replacement, which is how a clarification answer is
# folded into a technical PRD without mutating the artifact the workflow already recorded.
_LINEAGE_QUALIFIERS = ("attempt-", "revision-")


def artifact_id_matches_lineage(candidate: str, base_artifact_id: str) -> bool:
    """Accept a historical base ID or one canonical qualified descendant.

    Revisions were added to keep artifacts append-only, but this matcher was not taught about
    them, so every feature that answered a clarification question published its resolved
    technical PRD as `002_technical_prd.revision-2.json` and then failed with "required
    artifact is missing: 002_technical_prd.json". Features -057, -058 and -059 all died there.
    """
    if candidate == base_artifact_id:
        return True
    stem = base_artifact_id.removesuffix(".json")
    if not candidate.startswith(f"{stem}.") or not candidate.endswith(".json"):
        return False
    qualifier = candidate[len(stem) + 1 : -len(".json")]
    for marker in _LINEAGE_QUALIFIERS:
        if not qualifier.startswith(marker):
            continue
        number = qualifier.removeprefix(marker).removesuffix(".published")
        return number.isdigit() and str(int(number)) == number
    return False


def attempt_number_from_qualified_id(candidate: str) -> int | None:
    """Extract the trailing attempt number from an artifact ID, repository-qualified or not.

    ``_with_attempt_identity`` (``workflows/feature_workflow.py``) rewrites a freshly-produced
    artifact's ID to ``{stem}.{repository_id}.attempt-{N}.json`` before folding it into the
    feature's own live state -- a shape ``artifact_id_matches_lineage`` does not accept, so a
    caller that needs to rename such an artifact back to its plain, child-scoped form (as
    ``_prior_child_completions`` already does for code completions) first needs the number back
    out. Finds the last ``.attempt-`` segment rather than assuming a fixed prefix, so it works
    whether or not a repository qualifier is present. Returns ``None`` for anything that does
    not end in exactly ``.attempt-<digits>.json``.
    """
    if not candidate.endswith(".json"):
        return None
    stem = candidate[: -len(".json")]
    marker = "attempt-"
    index = stem.rfind(f".{marker}")
    if index == -1:
        return None
    number = stem[index + 1 + len(marker) :]
    return int(number) if number.isdigit() and str(int(number)) == number else None


class AgentArtifactError(ValueError):
    """Raised when an agent cannot consume or produce a valid workflow artifact.

    Failure text is withheld from persisted state by default because an arbitrary exception
    can carry credential-bearing detail. A raiser that has built its message from data it
    knows to be safe -- identifiers it validated itself, never model output or process
    environment -- may pass ``diagnostics`` to have that text recorded for the operator.

    ``stage`` names the agent that could not produce its artifact. Without it the control
    plane has to guess, and it guessed "feature_planner" for every rejection: AB-Feature-120
    died in the product manager's first model call and was recorded as a planning failure,
    which sent its reader to the one component that had not run.
    """

    def __init__(
        self, *args: object, diagnostics: Sequence[str] = (), stage: str | None = None
    ) -> None:
        """Keep explicitly safe explanations available to the control plane."""
        self.diagnostics = tuple(diagnostics)
        self.stage = stage
        super().__init__(*args)


# Pydantic error-context keys whose values are written in this repository's own schemas
# rather than taken from the input: an allowed literal set, a declared bound, a checked-in
# pattern. Recording one cannot disclose a rejected value, and it is usually enough to
# identify the field on its own -- `expected: 'must', 'should', 'could' or 'wont'` names
# `Requirement.priority` to anyone who has read the schema.
#
# `value_error` is deliberately absent. Its `ctx["error"]` is the exception a custom
# validator raised, and a validator is free to quote exactly what it rejected.
_SCHEMA_OWNED_ERROR_CONTEXT: Mapping[str, tuple[str, ...]] = {
    "literal_error": ("expected",),
    "enum": ("expected",),
    "string_pattern_mismatch": ("pattern",),
    "string_too_short": ("min_length",),
    "string_too_long": ("max_length",),
    "too_short": ("min_length",),
    "too_long": ("max_length",),
    "greater_than": ("gt",),
    "greater_than_equal": ("ge",),
    "less_than": ("lt",),
    "less_than_equal": ("le",),
}


def _schema_rejection(title: str, item: Mapping[str, Any]) -> str:
    """Describe one rejected field without repeating what was rejected.

    Four lines reading `schema validation rejected a value [literal_error]` and nothing else
    is what AB-Feature-120 left an operator, and no amount of reading it identifies the
    field. The model being validated and the constraint it violated are both owned by this
    repository, so both are safe to keep; ``loc`` still is not -- a mapping key is input, and
    an `extra_forbidden` rejection puts it there verbatim.
    """
    error_type = str(item.get("type", "unknown"))
    detail = f"schema validation rejected a value [{error_type}]"
    context = item.get("ctx")
    if isinstance(context, Mapping):
        stated = [
            f"{key}: {context[key]}"
            for key in _SCHEMA_OWNED_ERROR_CONTEXT.get(error_type, ())
            if key in context
        ]
        if stated:
            detail = f"{detail} ({'; '.join(stated)})"
    return f"{title}: {detail}" if title else detail


def safe_error_diagnostics(error: BaseException) -> tuple[str, ...]:
    """Return explanatory text for a failure only when it is known to be safe to record.

    An arbitrary exception's message can carry credential-bearing detail, so recording only
    the type is the correct default. Three kinds opt out of that: an error carrying explicit
    ``diagnostics`` it built itself, an ``AgentArtifactError``, whose message states which
    internal contract was violated, and a schema rejection, whose platform-owned error type
    describes the violated contract. Pydantic locations and messages are deliberately excluded:
    custom validators and mapping keys can place rejected model values in both.
    """
    declared = getattr(error, "diagnostics", ())
    if isinstance(declared, tuple | list) and declared:
        return tuple(str(item) for item in declared)
    if isinstance(error, AgentArtifactError):
        # This type reports a violated internal contract, and its message is written at the
        # raise site from constants and platform-generated identifiers -- never model output
        # or process environment. Suppressing it told an operator only that "the child
        # workstream raised AgentArtifactError", which names the type and hides the cause.
        # A raiser that cannot honour that must pass explicit ``diagnostics`` instead.
        return (str(error),) if str(error) else ()
    if isinstance(error, ValidationError):
        # ``input``, ``loc``, and ``msg`` may all contain rejected model values. The rejected
        # model's name and the constraint it declared are owned by the schema, so those are
        # kept; everything the input touched is not.
        return tuple(_schema_rejection(error.title, item) for item in error.errors())
    return ()


def require_artifact[ArtifactModel: BaseArtifact](
    state: AgentState,
    artifact_class: type[ArtifactModel],
    *,
    artifact_id: str | None = None,
) -> ArtifactModel:
    """Return the latest matching same-workflow artifact or raise a boundary error."""
    workflow_id = state["workflow_id"]
    for artifact in reversed(state["artifacts"]):
        if not isinstance(artifact, artifact_class):
            continue
        if artifact_id is not None and not artifact_id_matches_lineage(
            artifact.artifact_id, artifact_id
        ):
            continue
        if artifact.workflow_id != workflow_id:
            msg = f"artifact '{artifact.artifact_id}' belongs to a different workflow"
            raise AgentArtifactError(msg)
        return artifact
    expected_name = artifact_id or artifact_class.__name__
    msg = f"required artifact is missing: {expected_name}"
    raise AgentArtifactError(msg)


# How much of a rejected response the error carries. Enough to see a fence, a prose preamble
# or a truncation mid-token; far too little to reproduce the response.
_REJECTED_RESPONSE_HEAD_CHARACTERS = 500


def parse_model_json(
    output_text: str, *, expected_keys: Sequence[str] | None = None
) -> dict[str, Any]:
    """Parse one model response as a JSON object with an optional exact key set.

    Tolerant of exactly the two envelope habits models have that change nothing about the
    answer: a surrounding Markdown code fence, and a line or two of prose around one JSON
    object. AB-Feature-179 died in planning twice over an envelope -- "planner response is
    not valid JSON", first attempt and repair alike -- and because nothing of the rejected
    text was recorded, whether it was a fence or garbage was unknowable afterwards. So two
    things at once: the recovery is attempted here, and the failure that remains carries a
    redacted head of what was actually rejected.

    Deliberately not a general repair: the object slice is taken only when the raw and
    fence-stripped forms both fail, and everything still passes the same shape and
    ``expected_keys`` checks, so prose *quoting* a fragment cannot satisfy a caller that
    named its keys.
    """
    payload: Any = None
    for candidate in _json_candidates(output_text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        break
    else:
        head = redact_source_credentials(output_text[:_REJECTED_RESPONSE_HEAD_CHARACTERS])
        msg = f"agent model response must be valid JSON; the rejected response begins:\n{head}"
        raise AgentArtifactError(msg)
    if not isinstance(payload, dict):
        msg = "agent model response must be a JSON object"
        raise AgentArtifactError(msg)
    if expected_keys is not None and set(payload) != set(expected_keys):
        msg = f"agent model response must contain exactly: {sorted(expected_keys)}"
        raise AgentArtifactError(msg)
    return payload


def _json_candidates(output_text: str) -> list[str]:
    """Return the readings of one response worth attempting, most literal first."""
    candidates = [output_text]
    stripped = output_text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        # One surrounding fence, whatever its info string says (```json, ```JSON, bare ```).
        lines = stripped.splitlines()
        candidates.append("\n".join(lines[1:-1]))
    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        candidates.append(stripped[start : end + 1])
    return candidates


def create_artifact[ArtifactModel: BaseArtifact](
    artifact_class: type[ArtifactModel],
    *,
    workflow_id: str,
    artifact_id: str,
    producer: str,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> ArtifactModel:
    """Attach the platform-controlled envelope to validated model-generated artifact data."""
    protected_fields = {
        "schema_version",
        "workflow_id",
        "artifact_id",
        "producer",
        "timestamp",
        "metadata",
        "validation_status",
        "artifact_type",
    }
    supplied_protected_fields = protected_fields.intersection(payload)
    if supplied_protected_fields:
        msg = (
            "agent model response must not supply envelope fields: "
            f"{sorted(supplied_protected_fields)}"
        )
        raise AgentArtifactError(msg)
    return artifact_class.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "workflow_id": workflow_id,
            "artifact_id": artifact_id,
            "producer": producer,
            "timestamp": datetime.now(tz=UTC),
            "metadata": dict(metadata),
            "validation_status": "valid",
            **payload,
        }
    )


def artifact_json(artifact: Artifact) -> str:
    """Serialize one artifact for a prompt without converting it into agent conversation."""
    return json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True)


def artifact_list_json(artifacts: Sequence[Artifact]) -> str:
    """Serialize a deterministic artifact inventory for a planning prompt."""
    return json.dumps(
        [artifact.model_dump(mode="json") for artifact in artifacts],
        indent=2,
        sort_keys=True,
    )


def artifact_update(agent_name: str, artifacts: Sequence[Artifact]) -> dict[str, Any]:
    """Return the partial LangGraph update through which an agent publishes artifacts."""
    return {
        "artifacts": list(artifacts),
        "current_agent": agent_name,
        "current_step": WorkflowStatus.RUNNING,
    }
