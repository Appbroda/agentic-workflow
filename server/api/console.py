"""A small operator console that explains feature state without workflow vocabulary.

The API speaks in lifecycle enums. `failed_requires_human` is exact and tells a product
manager nothing, least of all that some repositories may have finished and opened pull
requests. This module owns the translation, and serves the single self-contained page that
uses it, so the wording is one testable thing rather than copy scattered through markup.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from api.feature_schemas import (
    PUBLISHING_EFFECTIVE_STATUS,
    RESUMING_EFFECTIVE_STATUS,
    RETRYING_EFFECTIVE_STATUS,
    REVISING_EFFECTIVE_STATUS,
)
from state.enums import ChildWorkflowStatus, FeatureWorkflowStatus

_CONSOLE_HTML = (Path(__file__).parent / "static" / "console.html").read_text(encoding="utf-8")

Tone = Literal["working", "attention", "done", "stopped"]


@dataclass(frozen=True, slots=True)
class StatusExplanation:
    """One lifecycle state described for somebody who did not write the workflow."""

    headline: str
    detail: str
    next_step: str
    tone: Tone


FEATURE_STATUS_VOCABULARY: dict[FeatureWorkflowStatus, StatusExplanation] = {
    FeatureWorkflowStatus.PENDING: StatusExplanation(
        headline="Queued",
        detail="The feature has been accepted and is waiting to start.",
        next_step="Nothing to do yet.",
        tone="working",
    ),
    FeatureWorkflowStatus.ANALYZING_PRD: StatusExplanation(
        headline="Reading your requirements",
        detail="Turning what you wrote into a technical plan.",
        next_step="Nothing to do yet.",
        tone="working",
    ),
    FeatureWorkflowStatus.WAITING_FOR_HUMAN: StatusExplanation(
        headline="Waiting on you",
        detail="Planning found something it cannot decide on its own.",
        next_step="Answer the open questions, then resume the feature.",
        tone="attention",
    ),
    FeatureWorkflowStatus.INSPECTING_REPOSITORIES: StatusExplanation(
        headline="Reading your repositories",
        detail="Looking at how each repository is actually built before planning against it.",
        next_step="Nothing to do yet.",
        tone="working",
    ),
    FeatureWorkflowStatus.PLANNING: StatusExplanation(
        headline="Planning the work",
        detail="Deciding what each repository has to build and in what order.",
        next_step="Nothing to do yet.",
        tone="working",
    ),
    FeatureWorkflowStatus.CONTRACT_READY: StatusExplanation(
        headline="Plan agreed",
        detail="The shared contract between the repositories is fixed. Work starts next.",
        next_step="Nothing to do yet.",
        tone="working",
    ),
    FeatureWorkflowStatus.RUNNING_CHILD_WORKFLOWS: StatusExplanation(
        headline="Building",
        detail="Each repository is being implemented, tested and reviewed on its own.",
        next_step="Nothing to do yet. Per-repository progress is below.",
        tone="working",
    ),
    FeatureWorkflowStatus.INTEGRATION_REVIEW: StatusExplanation(
        headline="Checking the pieces fit",
        detail="Reviewing the repositories against each other and against the contract.",
        next_step="Nothing to do yet.",
        tone="working",
    ),
    FeatureWorkflowStatus.CHANGES_REQUESTED: StatusExplanation(
        headline="Reworking",
        detail="The integration review sent work back to a repository to be corrected.",
        next_step="Nothing to do yet.",
        tone="working",
    ),
    FeatureWorkflowStatus.READY_FOR_PULL_REQUESTS: StatusExplanation(
        headline="Opening pull requests",
        detail="The work passed review and pull requests are being opened.",
        next_step="Nothing to do yet.",
        tone="working",
    ),
    FeatureWorkflowStatus.CREATING_PULL_REQUESTS: StatusExplanation(
        headline="Opening pull requests",
        detail="The work passed review and pull requests are being opened.",
        next_step="Nothing to do yet.",
        tone="working",
    ),
    FeatureWorkflowStatus.COMPLETED: StatusExplanation(
        headline="Finished",
        detail="Every required repository passed review and has a draft pull request open.",
        next_step="Have an engineer review and merge the pull requests. Nothing merges itself.",
        tone="done",
    ),
    FeatureWorkflowStatus.FAILED: StatusExplanation(
        headline="Stopped",
        detail="The feature stopped and cannot continue on its own.",
        next_step="Send the per-repository issues below to an engineer.",
        tone="stopped",
    ),
    FeatureWorkflowStatus.FAILED_REQUIRES_HUMAN: StatusExplanation(
        # One status covers two situations: some repositories finished, or none did. The
        # wording has to be true of both. It previously asserted that pull requests were open
        # and worth reviewing, which a feature whose every repository stopped displayed above
        # a pull-request tab reading "No pull requests yet".
        headline="Stopped — needs an engineer",
        detail=(
            "At least one repository stopped without an approved review. Any repository that "
            "did pass has a real draft pull request open; the ones that stopped are listed "
            "below with why."
        ),
        next_step=(
            "Read the blocking issues on each repository that stopped and send them to an "
            "engineer. Any pull requests listed are real and worth reviewing."
        ),
        tone="attention",
    ),
    FeatureWorkflowStatus.CANCELLING: StatusExplanation(
        headline="Cancelling",
        detail="Finishing the step in progress before stopping.",
        next_step="Nothing to do yet.",
        tone="working",
    ),
    FeatureWorkflowStatus.CANCELLED: StatusExplanation(
        headline="Cancelled",
        detail="The feature was stopped before anything was published.",
        next_step="Nothing to clean up. Start again when you are ready.",
        tone="stopped",
    ),
    FeatureWorkflowStatus.CANCELLED_WITH_EXTERNAL_SIDE_EFFECTS: StatusExplanation(
        headline="Cancelled — leftovers to check",
        detail=(
            "Cancelled after something had already reached GitHub, such as a pushed branch "
            "or an open pull request."
        ),
        next_step="Ask an engineer to check the cleanup items listed on the feature.",
        tone="attention",
    ),
}

FEATURE_EFFECTIVE_STATUS_VOCABULARY: dict[str, StatusExplanation] = {
    RESUMING_EFFECTIVE_STATUS: StatusExplanation(
        headline="Resuming",
        detail="Your answers were accepted and the workflow is continuing in the background.",
        next_step="Nothing to do. Do not submit the clarification form again.",
        tone="working",
    ),
    RETRYING_EFFECTIVE_STATUS: StatusExplanation(
        headline="Retrying",
        detail="The requested repository retry was accepted and is running in the background.",
        next_step="Nothing to do yet.",
        tone="working",
    ),
    PUBLISHING_EFFECTIVE_STATUS: StatusExplanation(
        headline="Publishing",
        detail=(
            "Your request to publish this feature's finished work was accepted and its pull "
            "requests are being opened in the background."
        ),
        next_step="Nothing to do. Do not press publish again.",
        tone="working",
    ),
    REVISING_EFFECTIVE_STATUS: StatusExplanation(
        headline="Revising",
        detail=(
            "Your change request was accepted. The published work is being revised on a new "
            "branch, and a replacement pull request will supersede the old one."
        ),
        next_step="Nothing to do. Do not submit the revision again.",
        tone="working",
    ),
}


CHILD_STATUS_VOCABULARY: dict[ChildWorkflowStatus, StatusExplanation] = {
    ChildWorkflowStatus.PENDING: StatusExplanation(
        headline="Waiting to start",
        detail="This repository has not been picked up yet.",
        next_step="Nothing to do yet.",
        tone="working",
    ),
    ChildWorkflowStatus.RUNNING: StatusExplanation(
        headline="In progress",
        detail="Being implemented, tested and reviewed.",
        next_step="Nothing to do yet.",
        tone="working",
    ),
    ChildWorkflowStatus.BLOCKED: StatusExplanation(
        headline="Blocked by another repository",
        detail="It needs something from a repository that has not finished.",
        next_step="Fixing the repository it depends on unblocks this one.",
        tone="attention",
    ),
    ChildWorkflowStatus.WAITING_FOR_CONTRACT_CHANGE: StatusExplanation(
        headline="Needs a decision about the shared contract",
        detail="This repository asked to change the agreed interface between repositories.",
        next_step="An engineer has to approve or reject the requested contract change.",
        tone="attention",
    ),
    ChildWorkflowStatus.REVIEW_REJECTED: StatusExplanation(
        headline="Review rejected the change",
        detail="The code review did not accept the work and the attempts ran out.",
        next_step="Send the issues below to an engineer.",
        tone="stopped",
    ),
    ChildWorkflowStatus.APPROVED: StatusExplanation(
        headline="Approved",
        detail="The work passed review and is ready for its pull request.",
        next_step="Nothing to do yet.",
        tone="working",
    ),
    ChildWorkflowStatus.FAILED: StatusExplanation(
        headline="Did not finish",
        detail="This repository could not be completed.",
        next_step="Send the issues below to an engineer.",
        tone="stopped",
    ),
    ChildWorkflowStatus.CANCELLED: StatusExplanation(
        headline="Cancelled",
        detail="This repository was stopped along with the feature.",
        next_step="Nothing to do.",
        tone="stopped",
    ),
    ChildWorkflowStatus.COMPLETED: StatusExplanation(
        headline="Done",
        detail="Reviewed, and its pull request is open.",
        next_step="Have an engineer review and merge it.",
        tone="done",
    ),
}


def status_vocabulary() -> dict[str, dict[str, dict[str, str]]]:
    """Return the console's wording so the page never invents its own for a new state."""
    return {
        "feature": {
            **{key.value: asdict(value) for key, value in FEATURE_STATUS_VOCABULARY.items()},
            **{key: asdict(value) for key, value in FEATURE_EFFECTIVE_STATUS_VOCABULARY.items()},
        },
        "workstream": {key.value: asdict(value) for key, value in CHILD_STATUS_VOCABULARY.items()},
    }


def create_console_router() -> APIRouter:
    """Serve the console page and its wording without requiring platform authentication.

    The page holds no secrets: it is markup that asks for credentials and keeps them in the
    tab's memory. Every request it makes carries the platform key the operator typed, so the
    data behind it stays behind `PlatformAuthenticator` exactly as before. Requiring a Bearer
    header on the document itself would only make the page unreachable from a browser.
    """
    router = APIRouter(tags=["console"])

    @router.get("/console", response_class=HTMLResponse, include_in_schema=False)
    async def console() -> HTMLResponse:
        """Return the self-contained operator console."""
        return HTMLResponse(content=_CONSOLE_HTML)

    @router.get("/console/status-vocabulary", include_in_schema=False)
    async def console_status_vocabulary() -> dict[str, dict[str, dict[str, str]]]:
        """Return the plain-language wording for every feature and workstream state."""
        return status_vocabulary()

    return router


__all__ = [
    "CHILD_STATUS_VOCABULARY",
    "FEATURE_STATUS_VOCABULARY",
    "StatusExplanation",
    "create_console_router",
    "status_vocabulary",
]
