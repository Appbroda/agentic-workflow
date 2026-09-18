"""Post-review Git publication for the legacy single-repository workflow."""

from __future__ import annotations

import inspect
from typing import Any

from adapters.interruptible_git import reviewed_content_fingerprint
from agents.shared.contracts import (
    ARTIFACT_FILENAMES,
    AgentArtifactError,
    artifact_id_matches_lineage,
    artifact_update,
    create_artifact,
    require_artifact,
)
from artifacts.schemas import CodeCompletionArtifact, FileChange, ReviewArtifact
from services.cancellation import CancellationToken
from state.models import AgentState
from tools.file_tools import resolve_workspace_path


class ApprovedChangePublisher:
    """Commit and push only the exact workspace change an approved review has accepted."""

    def __init__(self, *, git_service: Any, cancellation_token: CancellationToken | None = None):
        self._git_service = git_service
        self._cancellation_token = cancellation_token

    async def run(self, state: AgentState) -> dict[str, Any]:
        """Create the approved commit, push its branch, and publish the committed revision."""
        completion = require_artifact(
            state,
            CodeCompletionArtifact,
            artifact_id=ARTIFACT_FILENAMES["code_completion"],
        )
        review = require_artifact(
            state,
            ReviewArtifact,
            artifact_id=ARTIFACT_FILENAMES["review"],
        )
        if review.verdict != "approved":
            msg = "Git publication requires an approved review artifact"
            raise AgentArtifactError(msg)
        if completion.completion_status != "completed" or completion.remaining_work:
            msg = "Git publication requires a completed implementation with no remaining work"
            raise AgentArtifactError(msg)
        if completion.commit_sha is not None:
            if (
                completion.metadata.get("published_after_approval") is True
                and completion.metadata.get("approved_review_artifact_id") == review.artifact_id
            ):
                return {
                    "current_agent": "publisher",
                    "current_step": state["current_step"],
                }
            msg = "unverified pre-review commit cannot be published as approved work"
            raise AgentArtifactError(msg)
        if not completion.file_changes:
            msg = "Git publication requires at least one reviewed file change"
            raise AgentArtifactError(msg)

        await self._raise_if_cancelled()
        workspace = resolve_workspace_path(state["workspace_descriptor"].root_path, ".")
        cumulative_changes = _cumulative_file_changes(state)
        reviewed_paths = list(cumulative_changes)
        content_fingerprint = reviewed_content_fingerprint(workspace, reviewed_paths)
        require_reviewed_workspace_match(
            review,
            reviewed_paths=reviewed_paths,
            content_fingerprint=content_fingerprint,
        )
        commit_sha = await _await_if_needed(
            self._git_service.commit(
                workspace,
                f"workflow {state['workflow_id']}: {completion.summary}",
                files=reviewed_paths,
                expected_content_fingerprint=content_fingerprint,
            )
        )
        if not isinstance(commit_sha, str) or not commit_sha.strip():
            msg = "Git service returned an invalid commit SHA"
            raise AgentArtifactError(msg)
        await self._raise_if_cancelled()
        await _await_if_needed(
            self._git_service.push(
                workspace,
                state["workspace_descriptor"].working_branch,
                expected_commit_sha=commit_sha,
            )
        )

        payload = completion.model_dump(mode="python")
        for field in {
            "schema_version",
            "workflow_id",
            "artifact_id",
            "artifact_type",
            "producer",
            "timestamp",
            "metadata",
            "validation_status",
        }:
            payload.pop(field, None)
        payload["commit_sha"] = commit_sha
        payload["file_changes"] = [
            change.model_dump(mode="python") for change in cumulative_changes.values()
        ]
        published = create_artifact(
            CodeCompletionArtifact,
            workflow_id=state["workflow_id"],
            artifact_id=f"{completion.artifact_id.removesuffix('.json')}.published.json",
            producer="publisher",
            payload=payload,
            metadata={
                **completion.metadata,
                "source_artifact_ids": [completion.artifact_id, review.artifact_id],
                "published_after_approval": True,
                "approved_review_artifact_id": review.artifact_id,
                "reviewed_content_fingerprint": content_fingerprint,
            },
        )
        return artifact_update("publisher", [published])

    async def _raise_if_cancelled(self) -> None:
        """Do not begin the next irreversible Git mutation after cancellation."""
        if self._cancellation_token is not None:
            await self._cancellation_token.raise_if_cancelled()


async def _await_if_needed(value: object) -> Any:
    """Accept synchronous test adapters and async journal-backed production adapters."""
    return await value if inspect.isawaitable(value) else value


def _cumulative_file_changes(state: AgentState) -> dict[str, FileChange]:
    """Collect every dirty attempt file so a narrow retry cannot drop earlier work."""
    changes: dict[str, FileChange] = {}
    for artifact in state["artifacts"]:
        if not isinstance(artifact, CodeCompletionArtifact):
            continue
        if not artifact_id_matches_lineage(
            artifact.artifact_id, ARTIFACT_FILENAMES["code_completion"]
        ):
            continue
        if artifact.metadata.get("published_after_approval") is True:
            continue
        for change in artifact.file_changes:
            changes[change.path] = change
    return changes


def require_reviewed_workspace_match(
    review: ReviewArtifact,
    *,
    reviewed_paths: list[str],
    content_fingerprint: str,
    evidence_required: bool = False,
) -> None:
    """Bind ReviewerAgent evidence to the bytes entering the approved commit."""
    evidence_version = review.metadata.get("review_evidence_version")
    if evidence_version is None:
        # Custom deterministic/mock reviewers predate source evidence. ProductionReviewer
        # always emits the versioned fields below; retain the narrow mock compatibility path.
        if not evidence_required:
            return
        msg = "approved review is missing its workspace source provenance"
        raise AgentArtifactError(msg)
    recorded_paths = review.metadata.get("reviewed_file_paths")
    recorded_fingerprint = review.metadata.get("reviewed_content_fingerprint")
    # The path lists are identity evidence and the fingerprint is content evidence. Identity
    # is the *set* of paths -- unique, because they come from a dict of file changes -- and
    # the two sides legitimately order it differently: the reviewer records review order,
    # the operation journal stores its `expected_staged_files` sorted. Comparing raw lists
    # here made every crash-window recovery of a committed publication refuse its own
    # receipt (AB-Feature-184). Ordering must stay irrelevant inside this predicate, not be
    # patched around at call sites: a second caller sorting its input is how that happened.
    if (
        evidence_version != "workspace-source-v1"
        or not isinstance(recorded_paths, list)
        or not all(isinstance(path, str) for path in recorded_paths)
        or sorted(recorded_paths) != sorted(reviewed_paths)
        or recorded_fingerprint != content_fingerprint
    ):
        msg = "workspace content no longer matches the source evidence approved by the reviewer"
        raise AgentArtifactError(msg)


__all__ = ["ApprovedChangePublisher", "require_reviewed_workspace_match"]
