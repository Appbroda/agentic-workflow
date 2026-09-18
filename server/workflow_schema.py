"""Dependency-leaf policy for durable workflow and artifact compatibility."""

from __future__ import annotations

WORKFLOW_SCHEMA_VERSION = "1.0"
LEGACY_UNVERIFIED_BUILD_REVISION = "legacy-unverified"


def workflow_mutation_identity_error(
    *,
    workflow_schema_version: object,
    created_by_build_revision: object,
    last_executor_build_revision: object,
) -> str | None:
    """Explain why a durable snapshot is unsafe to mutate, or return ``None``.

    Migration-backed legacy snapshots intentionally remain readable for audit.  Their
    synthetic build marker is not execution provenance, however, so it must never be
    treated as permission to resume or otherwise rewrite the snapshot.
    """
    if workflow_schema_version != WORKFLOW_SCHEMA_VERSION:
        return (
            f"workflow state schema {workflow_schema_version!r} is incompatible with "
            f"executor schema {WORKFLOW_SCHEMA_VERSION!r}"
        )
    if not isinstance(created_by_build_revision, str) or not created_by_build_revision.strip():
        return "workflow state is missing its creator build identity"
    if created_by_build_revision == LEGACY_UNVERIFIED_BUILD_REVISION:
        return "workflow state creator build identity is legacy-unverified"
    if (
        not isinstance(last_executor_build_revision, str)
        or not last_executor_build_revision.strip()
    ):
        return "workflow state is missing its last executor build identity"
    if last_executor_build_revision == LEGACY_UNVERIFIED_BUILD_REVISION:
        return "workflow state last executor build identity is legacy-unverified"
    return None
