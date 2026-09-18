"""Errors, request-scoped credentials, and the payload shapes API responses carry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from artifacts.schemas import Artifact
from state.models import (
    ExecutionLogEntry,
)


class WorkflowControlPlaneError(RuntimeError):
    """Base error for expected control-plane failures that map to HTTP responses."""


class WorkflowNotFoundError(WorkflowControlPlaneError):
    """Raised when a requested workflow identifier does not exist."""


class WorkflowConflictError(WorkflowControlPlaneError):
    """Raised when a requested lifecycle transition is not valid for the workflow state."""


class WorkflowBusyError(WorkflowConflictError):
    """Raised when the feature is mid-operation and the same request would succeed later.

    A subclass rather than a flag, so every existing `except WorkflowConflictError` keeps
    catching it and only the code that needs to tell them apart has to know it exists.

    Distinct because "wait" and "stop and read the state" are opposite instructions and both
    arrive as a bare 409 with prose. The lock's own message says "retry shortly"; the
    ledger's says "not repeated automatically". A client cannot tell which it has been given
    without reading the sentence, and the loop that made AB-Feature-203 unrecoverable was
    doing the obviously reasonable thing with the one that said to retry.

    Deliberately not every `WorkflowConflictError`. A repository with no attempts left to
    grant also raises the base class, and no interval makes that true.
    """


class ClarificationValidationError(WorkflowControlPlaneError):
    """Raised when resume answers do not exactly match unresolved technical-PRD questions."""


class FeatureOperationFailedError(WorkflowConflictError):
    """Raised when an operation on an existing feature did not happen.

    Distinct from an ordinary conflict because the feature has been marked as needing
    attention on the way past: the caller is being told both that the thing they asked for
    was not carried out, and that the reason is recorded on the feature rather than in this
    message. The message itself stays platform-owned -- an arbitrary exception's text can
    carry a provider response or a repository path.

    It exists because returning the unchanged feature was indistinguishable from success.
    Every mutation now runs inside a durable action whose result is the audit answer to "did
    this happen?", and a normal return wrote "committed" into that record for an operation
    that had committed nothing.
    """


@dataclass(frozen=True, slots=True)
class RequestScopedCredentials:
    """Ephemeral provider credentials accepted only from HTTP headers and never persisted."""

    openai_api_key: str | None
    github_token: str | None
    # Defaulted so every existing construction -- including the positional
    # `RequestScopedCredentials(None, None)` used throughout the tests -- keeps working. A
    # feature on OpenAI never needs this one, and a feature on Anthropic never needs the
    # first: which is required is a property of the feature, not of the deployment.
    anthropic_api_key: str | None = None
    # When the GitHub credential above was stored, and nothing else about it. Carried beside
    # the value because the place that discovers a credential is refused -- a `git clone`
    # deep in an adapter -- has no way to ask the secret store anything, and "the stored
    # github credential was refused; it was stored on 2026-08-26" is the entire difference
    # between run 190's diagnosis and the one it needed. None where the credential arrived
    # as a request header, which has no age to report.
    github_token_stored_at: datetime | None = None


@dataclass(slots=True)
class TimelineEvent:
    """An in-memory lifecycle event generated directly by the HTTP control plane."""

    timestamp: datetime
    event: str
    details: dict[str, Any]


def artifact_payload(artifact: Artifact) -> dict[str, Any]:
    """Return a public envelope that keeps artifact contract fields and payload distinct."""
    serialized = artifact.model_dump(mode="json")
    envelope_fields = {
        "schema_version",
        "workflow_id",
        "artifact_id",
        "producer",
        "timestamp",
        "metadata",
        "validation_status",
        "artifact_type",
    }
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_type": artifact.artifact_type,
        "workflow_id": artifact.workflow_id,
        "schema_version": artifact.schema_version,
        "producer": artifact.producer,
        "timestamp": artifact.timestamp,
        "metadata": artifact.metadata,
        "validation_status": artifact.validation_status,
        "payload": {key: value for key, value in serialized.items() if key not in envelope_fields},
    }


def log_payload(log: ExecutionLogEntry) -> dict[str, Any]:
    """Serialize one structured execution log for its API response schema."""
    return log.model_dump(mode="json")
