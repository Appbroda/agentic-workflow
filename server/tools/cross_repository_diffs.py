"""The changed source of every repository in one feature, for reviewing the seam between them.

Child reviews are deliberately scoped: each judges its own repository against its own assigned
requirements and is shown the rest of the feature only as non-actionable background. That is
correct -- a backend reviewer must not reject a backend change over a frontend tile -- but it
means no reviewer ever sees two repositories at once. A consumer calling a provider endpoint
with the wrong field name, the wrong method, or the wrong auth header passes every gate the
platform has.

This module carries the material that makes such a review possible. It defines no policy: the
integration reviewer decides what to do with the diffs, and a deployment without a diff source
keeps exactly the contract-conformance gate it had before.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import Field

from artifacts.schemas import ChildWorkflowResultArtifact
from state.models import StateModel

_MAX_FILES_PER_REPOSITORY = 40
_MAX_CHARACTERS_PER_REPOSITORY = 60_000


class RepositoryChange(StateModel):
    """One repository's production change, as the other repositories would have to consume it."""

    repository_id: str = Field(min_length=1)
    role: str = ""
    contract_sections_implemented: list[str] = Field(default_factory=list)
    contract_sections_consumed: list[str] = Field(default_factory=list)
    files: list[dict[str, str]] = Field(default_factory=list)
    omitted_file_count: int = 0
    truncated: bool = False


class CrossRepositoryDiffs(Protocol):
    """Supply the production change of every repository under one feature."""

    async def collect(
        self, *, feature_id: str, child_results: Sequence[ChildWorkflowResultArtifact]
    ) -> list[RepositoryChange]:
        """Return one entry per repository whose change could be read.

        Driven by the results themselves, never by a captured feature snapshot. The first
        version took the state the orchestrator was constructed with, which is the state
        *before* any child ran: every repository still had an empty changed-file list, so
        live feature -084 published two pull requests with the seam unreviewed. A result
        carries its own workspace path and production files, and is current by construction.
        """


class NullCrossRepositoryDiffs:
    """Supply nothing, leaving the integration gate exactly as it was without a diff source.

    Not an error and not an empty diff: those are different claims. A caller that receives no
    entries must report that the seam was not reviewed, never that it was reviewed and found
    clean.
    """

    async def collect(
        self, *, feature_id: str, child_results: Sequence[ChildWorkflowResultArtifact]
    ) -> list[RepositoryChange]:
        """Report that no repository change could be read."""
        del feature_id, child_results
        return []


def bounded_repository_change(
    *,
    repository_id: str,
    role: str,
    contract_sections_implemented: Sequence[str],
    contract_sections_consumed: Sequence[str],
    files: Sequence[tuple[str, str]],
) -> RepositoryChange:
    """Fit one repository's change inside the review budget, saying so when it did not fit.

    Both bounds are reported rather than silently applied. An integration review that read
    half of a repository's change and said nothing about the other half would be the same
    false assurance the contract-only gate used to give.
    """
    kept: list[dict[str, str]] = []
    characters = 0
    truncated = False
    for path, content in files[:_MAX_FILES_PER_REPOSITORY]:
        if characters + len(content) > _MAX_CHARACTERS_PER_REPOSITORY:
            truncated = True
            continue
        kept.append({"path": path, "content": content})
        characters += len(content)
    return RepositoryChange(
        repository_id=repository_id,
        role=role,
        contract_sections_implemented=list(contract_sections_implemented),
        contract_sections_consumed=list(contract_sections_consumed),
        files=kept,
        omitted_file_count=max(0, len(files) - len(kept)),
        truncated=truncated or len(files) > _MAX_FILES_PER_REPOSITORY,
    )


__all__ = [
    "CrossRepositoryDiffs",
    "NullCrossRepositoryDiffs",
    "RepositoryChange",
    "bounded_repository_change",
]
