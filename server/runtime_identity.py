"""Fail-closed identity checks for immutable production runtimes."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from workflow_schema import WORKFLOW_SCHEMA_VERSION

_GIT_REVISION = re.compile(r"^[0-9a-f]{7,64}$", re.IGNORECASE)
_UNKNOWN_REVISIONS = frozenset({"", "unknown", "local", "development"})


class RuntimeIdentityError(RuntimeError):
    """Raised before production startup when the deployed identity is not trustworthy."""


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Actual image/schema identity and the deployment controller's expectations."""

    build_revision: str
    workflow_schema_version: str
    expected_build_revision: str | None
    expected_workflow_schema_version: str | None
    environment: str

    @property
    def compatibility_errors(self) -> tuple[str, ...]:
        """Return stable, non-secret reason codes for an unsafe runtime."""
        errors: list[str] = []
        production = self.environment == "production"
        normalized_revision = self.build_revision.lower()
        if production and (
            normalized_revision in _UNKNOWN_REVISIONS
            or _GIT_REVISION.fullmatch(self.build_revision) is None
        ):
            errors.append("unknown_build_revision")
        if production and not self.expected_build_revision:
            errors.append("missing_expected_build_revision")
        elif (
            self.expected_build_revision is not None
            and self.build_revision != self.expected_build_revision
        ):
            errors.append("build_revision_mismatch")
        if production and not self.expected_workflow_schema_version:
            errors.append("missing_expected_workflow_schema_version")
        elif (
            self.expected_workflow_schema_version is not None
            and self.workflow_schema_version != self.expected_workflow_schema_version
        ):
            errors.append("workflow_schema_version_mismatch")
        return tuple(errors)

    @property
    def compatible(self) -> bool:
        """Return whether this process may safely accept mutating workflow requests."""
        return not self.compatibility_errors

    def require_compatible(self) -> None:
        """Refuse startup rather than silently serve from an ambiguous production image."""
        if self.compatibility_errors:
            reasons = ", ".join(self.compatibility_errors)
            raise RuntimeIdentityError(f"runtime identity is incompatible: {reasons}")


def load_runtime_identity(
    environment: Mapping[str, str] | None = None,
    *,
    build_revision: str | None = None,
    expected_build_revision: str | None = None,
    expected_workflow_schema_version: str | None = None,
    deployment_environment: str | None = None,
) -> RuntimeIdentity:
    """Read identity from process configuration without mutating or caching environment state."""
    values = os.environ if environment is None else environment
    return RuntimeIdentity(
        build_revision=(build_revision or values.get("BUILD_REVISION", "")).strip() or "local",
        workflow_schema_version=WORKFLOW_SCHEMA_VERSION,
        expected_build_revision=_optional_value(
            expected_build_revision
            if expected_build_revision is not None
            else values.get("EXPECTED_BUILD_REVISION")
        ),
        expected_workflow_schema_version=_optional_value(
            expected_workflow_schema_version
            if expected_workflow_schema_version is not None
            else values.get("EXPECTED_WORKFLOW_SCHEMA_VERSION")
        ),
        environment=(deployment_environment or values.get("ENVIRONMENT", "development")).strip()
        or "development",
    )


def _optional_value(value: str | None) -> str | None:
    """Normalize whitespace-only deployment expectations to an absent value."""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None
