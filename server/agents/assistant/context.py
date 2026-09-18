"""Selective context for the feature assistant.

A feature's full record is large -- a completed two-repository feature carries roughly 400 KB
of artifacts -- and sending it on every message would be slow, expensive, and worse at
answering: the relevant paragraph gets lost among forty documents. This module assembles the
small set of facts an assistant needs to answer questions about *this* feature's current state,
and names the artifacts it could ask for rather than including them.
"""

from __future__ import annotations

from typing import Any

from artifacts.schemas import Artifact
from state.feature_models import FeatureWorkflowSnapshot

_RECENT_EVENT_LIMIT = 15
_BLOCKING_ISSUE_LIMIT = 6

# Artifacts small enough to be worth including whole, because almost every question about a
# feature's intent or contract is answered by one of them.
_INLINE_ARTIFACT_TYPES = frozenset({"integration_contract", "repository_reconnaissance"})
_INLINE_ARTIFACT_BUDGET = 40_000

_QUERY_ARTIFACT_TERMS: dict[str, tuple[str, ...]] = {
    "prd": ("prd", "requirement", "acceptance criterion"),
    "technical_prd": ("technical prd", "technical requirement"),
    "architecture": ("architecture", "component", "decision", "risk"),
    "repository_execution_plan": ("plan", "execution order", "merge order", "dependency"),
    "task_plan": ("task plan", "task", "milestone"),
    "integration_contract": ("contract", "api", "schema", "authentication", "event"),
    "review": ("review", "rejected", "finding"),
    "integration_review": ("integration review", "compatibility"),
    "child_workflow_result": ("failure", "failed", "blocked", "validation", "workstream"),
    "code_completion": ("changed", "implementation", "file"),
    "pull_request": ("pull request", " pr", "merge"),
    "feature_completion": ("completion", "deployment", "rollback", "merge"),
    "contract_change_request": ("contract change", "change request"),
}


def build_context(
    state: FeatureWorkflowSnapshot,
    *,
    events: list[dict[str, Any]],
    query: str = "",
) -> dict[str, Any]:
    """Return the facts an assistant may reason from, and the names of what else exists."""
    workstreams = [
        {
            "repository_id": child.repository_id,
            "status": child.status.value if hasattr(child.status, "value") else str(child.status),
            "branch": child.branch_name,
            "attempts": child.retry_count,
            "failure_classification": child.failure_classification,
            "retry_refusal_reason": child.retry_refusal_reason,
            # The end of this list is where the platform put the question it stopped on.
            "blocking_issues": list(child.blocking_issues)[-_BLOCKING_ISSUE_LIMIT:],
            "production_files_changed": list(child.production_files_changed),
            "requirements_not_implemented": list(child.requirements_not_implemented),
            "current_validation_results": list(child.current_validation_results)[-4:],
            "code_completion_artifact_id": child.code_completion_artifact_id,
            "review_artifact_id": child.review_artifact_id,
            "pull_request_artifact_id": child.pull_request_artifact_id,
        }
        for child in state.child_workflows.values()
    ]
    return {
        "feature": {
            "feature_id": state.feature_id,
            "title": state.title,
            "status": state.status.value,
            "current_agent": state.current_agent,
            "execution_mode": state.execution_mode,
            "clarification_rounds": state.clarification_rounds,
            "merge_strategy": state.merge_strategy.value if state.merge_strategy else None,
            "deployment_strategy": (
                state.deployment_strategy.value if state.deployment_strategy else None
            ),
            "failure_summary": (
                state.failure_summary.model_dump(mode="json") if state.failure_summary else None
            ),
        },
        "repositories": [
            {
                "repository_id": spec.repository_id,
                "name": spec.name,
                "role": spec.role,
                "required": spec.required,
            }
            for spec in state.repository_specs
        ],
        "workstreams": workstreams,
        "open_questions": _open_questions(state),
        "recent_events": events[-_RECENT_EVENT_LIMIT:],
        "artifacts_inline": _inline_artifacts(state.artifacts, query=query),
        # Named, not included. The assistant can tell a person what exists and they can open it.
        "artifacts_available": [
            {"artifact_id": item.artifact_id, "artifact_type": item.artifact_type}
            for item in state.artifacts
        ],
    }


def _open_questions(state: FeatureWorkflowSnapshot) -> list[dict[str, str]]:
    """Return the clarification questions the feature is currently stopped on."""
    from artifacts.schemas import TechnicalPRDArtifact

    technical_prd = next(
        (item for item in reversed(state.artifacts) if isinstance(item, TechnicalPRDArtifact)),
        None,
    )
    if technical_prd is None:
        return []
    return [
        {"question_id": item.question_id, "question": item.question, "rationale": item.rationale}
        for item in technical_prd.unresolved_questions
    ]


def _inline_artifacts(artifacts: list[Artifact], *, query: str) -> list[dict[str, Any]]:
    """Include latest high-value and query-relevant artifacts within a bounded budget."""
    normalized = query.casefold()
    requested_ids = {
        item.artifact_id for item in artifacts if item.artifact_id.casefold() in normalized
    }
    requested_types = {
        artifact_type
        for artifact_type, terms in _QUERY_ARTIFACT_TERMS.items()
        if any(term in normalized for term in terms)
    }
    latest: dict[str, Artifact] = {}
    for artifact in artifacts:
        if artifact.artifact_id in requested_ids:
            # Keep an explicitly named historical artifact under its own key. If the query
            # also contains a broad word such as "review", the latest review must not replace
            # the exact attempt the person asked about.
            latest[f"requested:{artifact.artifact_id}"] = artifact
        if (
            artifact.artifact_type in _INLINE_ARTIFACT_TYPES
            or artifact.artifact_type in requested_types
        ):
            # Latest per type and repository lineage. Child artifacts carry the repository in
            # their id, so keeping their complete id avoids one repository replacing another.
            key = (
                artifact.artifact_type
                if artifact.artifact_type in _INLINE_ARTIFACT_TYPES
                else f"{artifact.artifact_type}:{_repository_hint(artifact)}"
            )
            latest[key] = artifact
    included: list[dict[str, Any]] = []
    used = 0
    for artifact in latest.values():
        payload = artifact.model_dump(mode="json")
        size = len(str(payload))
        if used + size > _INLINE_ARTIFACT_BUDGET:
            continue
        included.append(payload)
        used += size
    return included


def _repository_hint(artifact: Artifact) -> str:
    """Return a stable repository discriminator without depending on one artifact subtype."""
    payload = artifact.model_dump(mode="python")
    repository_id = payload.get("repository_id") or payload.get("requested_by_repository_id")
    if isinstance(repository_id, str):
        return repository_id
    metadata_repository = artifact.metadata.get("repository_id")
    if isinstance(metadata_repository, str):
        return metadata_repository
    # Attempt-qualified child artifacts use `<type>.<repository>.attempt-N.json`.
    parts = artifact.artifact_id.split(".")
    if len(parts) >= 3 and parts[1]:
        return parts[1]
    return artifact.artifact_id


__all__ = ["build_context"]
