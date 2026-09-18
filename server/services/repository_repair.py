"""Repository repair as a domain concept: what qualifies, and what may be decided about it.

A repository repair is not a coding failure. It is the specific situation where a repository
cannot run its own checks on an untouched checkout -- a lint configuration whose shared plugin
is not declared, a package manager whose lockfile does not resolve, a script the manifest
promises and does not have. Nothing written into such a repository can be validated, so the
platform stops instead of reporting an implementation it could not check.

What it must not do is fix the repository by itself. Changing somebody's checked-in setup is a
decision about their project, and the platform has exactly one thing to offer: a diagnosis
precise enough to approve or refuse. This module decides which failures qualify, writes that
diagnosis down, and answers whether a proposal still describes the checkout it was written
against.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from artifacts.schemas import (
    Artifact,
    RepositoryRepairProposalArtifact,
)
from state.enums import RepositoryRepairStatus
from tools.repository_preflight import PreflightIssue
from tools.retry_strategy import FailureClassification

# The preflight findings a person can approve a deterministic fix for. Everything outside
# this set is either a defect in the code being written -- which is the engineer's problem,
# not the repository's -- or something no fixed command can repair.
#
# Kept as categories rather than as a judgement about severity: a missing dev dependency is
# repairable whether it was reported as critical or as high, and a source-structure finding
# is not repairable however severe it looks.
REPAIRABLE_CATEGORIES = frozenset(
    {
        "dependency_configuration",
        "missing_dependency",
        "invalid_lint_configuration",
        "package_manager_lockfile",
        "missing_test_command",
        "missing_build_command",
    }
)

# The failure classifications that stop a workstream *because of the repository* rather than
# because of the work. These are the only ones that may produce a repair proposal.
REPAIRABLE_CLASSIFICATIONS = frozenset(
    {
        FailureClassification.VALIDATION_CONFIGURATION_FAILURE,
        FailureClassification.DEPENDENCY_INSTALLATION_FAILURE,
    }
)

# Statuses in which a repair is still somebody's to decide or the platform's to carry out.
OPEN_REPAIR_STATUSES = frozenset(
    {
        RepositoryRepairStatus.PROPOSED,
        RepositoryRepairStatus.APPROVED,
        RepositoryRepairStatus.EXECUTING,
    }
)


class RepairNotFoundError(LookupError):
    """Raised when a repair identifier does not belong to this feature."""


def current_repairs(
    artifacts: Sequence[Artifact],
) -> dict[str, RepositoryRepairProposalArtifact]:
    """Return the newest artifact for each repair, which is that repair's current state.

    Artifact history is append-only: a decision is recorded by appending a revision carrying
    the same ``repair_id``, never by editing what was already written. So the last one wins.
    """
    latest: dict[str, RepositoryRepairProposalArtifact] = {}
    for artifact in artifacts:
        if isinstance(artifact, RepositoryRepairProposalArtifact):
            latest[artifact.repair_id] = artifact
    return latest


def find_repair(artifacts: Sequence[Artifact], repair_id: str) -> RepositoryRepairProposalArtifact:
    """Return one repair's current state, or say which identifier was not found."""
    repair = current_repairs(artifacts).get(repair_id)
    if repair is None:
        msg = f"repository repair not found: {repair_id}"
        raise RepairNotFoundError(msg)
    return repair


def open_repair_for(
    artifacts: Sequence[Artifact], repository_id: str
) -> RepositoryRepairProposalArtifact | None:
    """Return the repair currently blocking one repository, if any.

    Identified by repository id and never by role: a feature may contain any number of
    repositories, and two of them may be stopped for the same reason at once.
    """
    open_repairs = [
        item
        for item in current_repairs(artifacts).values()
        if item.repository_id == repository_id
        and RepositoryRepairStatus(item.status) in OPEN_REPAIR_STATUSES
    ]
    return open_repairs[-1] if open_repairs else None


def approved_repair_for(
    artifacts: Sequence[Artifact], repository_id: str
) -> RepositoryRepairProposalArtifact | None:
    """Return the repair this repository has been authorized to apply, if any.

    Distinct from ``open_repair_for``: a proposal nobody has decided about must never be
    applied, and this is what the executor asks before touching a checkout.
    """
    authorized = [
        item
        for item in current_repairs(artifacts).values()
        if item.repository_id == repository_id
        and RepositoryRepairStatus(item.status)
        in {RepositoryRepairStatus.APPROVED, RepositoryRepairStatus.EXECUTING}
    ]
    return authorized[-1] if authorized else None


def repair_is_stale(
    repair: RepositoryRepairProposalArtifact, *, current_revision: str | None
) -> bool:
    """Return whether the repository moved on after this diagnosis was written.

    A stale proposal is not applied. The commands it names were chosen against a checkout
    that no longer exists, and the problem may have been fixed in the meantime -- so
    approving it could install a dependency somebody has already removed, or commit a
    configuration change on top of a different one.

    A proposal with no recorded revision is not treated as stale: it predates revision
    tracking, and refusing every one of them would strand features that are already stopped.
    """
    if repair.proposed_at_revision is None or current_revision is None:
        return False
    return repair.proposed_at_revision != current_revision


def repairable_issues(
    issues: Sequence[PreflightIssue],
    *,
    classification: FailureClassification | None,
) -> list[PreflightIssue]:
    """Select the blocking findings a person could reasonably approve a fix for.

    Returns nothing when the workstream stopped for a reason that is not about the
    repository. That restraint is the point: treating an ordinary coding failure as a repair
    request would ask somebody to approve a change to their repository for a problem that was
    never in it.
    """
    if classification is not None and classification not in REPAIRABLE_CLASSIFICATIONS:
        return []
    return [item for item in issues if item.category in REPAIRABLE_CATEGORIES]


def build_repair_payload(
    *,
    feature_id: str,
    repository_id: str,
    child_workflow_id: str | None,
    originating_stage: str,
    classification: FailureClassification | str,
    issues: Sequence[PreflightIssue],
    current_revision: str | None,
    package_manager: str | None = None,
    repair_id: str | None = None,
) -> dict[str, Any]:
    """Turn preflight findings into one proposal payload a person can act on.

    Several findings become one proposal rather than several: they are usually one broken
    setup seen from different angles, and asking somebody to approve four repairs to a
    repository that has one problem is how an approval becomes a rubber stamp.
    """
    if not issues:
        msg = "a repair proposal requires at least one repairable finding"
        raise ValueError(msg)
    # Read from the finding's own fields. The preflight classifier already extracted exactly
    # which packages would not resolve, and recovering them from the sentence describing them
    # was a guess that produced nothing useful for the commonest repair of all.
    dependencies = safe_package_names(
        [
            reference
            for item in issues
            for reference in (*item.undeclared_references, *item.unresolved_references)
        ]
    )
    undeclared = [reference for item in issues for reference in item.undeclared_references]
    commands = repair_commands(package_manager, undeclared)
    return {
        "repair_id": repair_id or f"repair-{uuid4()}",
        "feature_id": feature_id,
        "repository_id": repository_id,
        "child_workflow_id": child_workflow_id,
        "originating_stage": originating_stage,
        "failure_classification": str(
            classification.value
            if isinstance(classification, FailureClassification)
            else classification
        ),
        "detected_problem": "; ".join(item.description for item in issues),
        "evidence": [item.evidence for item in issues],
        "proposed_repair": "; ".join(item.recommended_action for item in issues),
        # The manifest that declares dependencies, when the platform knows which one this
        # repository uses. Nothing is listed when it does not: a repair that names a file it
        # is guessing at is worse than one that names none.
        "affected_files": _manifest_files(package_manager) if dependencies else [],
        "affected_dependencies": dependencies,
        "commands": commands,
        "expected_impact": (
            "The repository can run its own checks again, so an implementation written here "
            "can be validated rather than reported unchecked."
        ),
        "risk": _risk(issues, commands=commands),
        # A setup repair restores the ability to check the code; it must not change what the
        # code does. Anything that would is not a repair, and is not proposed as one.
        "changes_source_logic": False,
        "proposed_at_revision": current_revision,
        "status": RepositoryRepairStatus.PROPOSED.value,
        "approved_by": None,
        "approved_at": None,
        "rejected_by": None,
        "rejection_reason": None,
        "execution_result": None,
        "resulting_revision": None,
    }


def _risk(issues: Sequence[PreflightIssue], *, commands: Sequence[Any] = ()) -> str:
    """Rate a proposal by what applying it would touch.

    A repair that runs a command against the repository is rated above one that only causes
    a fresh attempt: installing something affects everything that repository builds, not only
    the check that failed.
    """
    if commands:
        return "medium"
    if any(item.category in {"missing_dependency", "package_manager_lockfile"} for item in issues):
        return "medium"
    if any(item.severity == "critical" for item in issues):
        return "medium"
    return "low"


# The file a dependency change is made in, per package manager. Enumerated rather than
# inferred from a finding's prose: a repair that names the wrong file sends somebody to edit
# the wrong thing, which is worse than naming none at all.
_DEPENDENCY_MANIFESTS = {
    "npm": ["package.json"],
    "pnpm": ["package.json"],
    "yarn": ["package.json"],
    "bun": ["package.json"],
    "pip": ["pyproject.toml"],
    "poetry": ["pyproject.toml"],
    "uv": ["pyproject.toml"],
}

# How each package manager declares a development dependency. `pip` is deliberately absent:
# it installs into an environment without recording anything in a manifest, so there is no
# command that would make the declaration the repository is missing. npm carries
# `--no-audit` like every other npm install on this platform: the advisory report is a live
# POST to registry endpoints that have hung before, and no repair reads it.
_DECLARE_DEV_DEPENDENCY = {
    "npm": ("npm", "install", "--save-dev", "--no-audit"),
    "pnpm": ("pnpm", "add", "--save-dev"),
    "yarn": ("yarn", "add", "--dev"),
    "bun": ("bun", "add", "--dev"),
    "uv": ("uv", "add", "--dev"),
    "poetry": ("poetry", "add", "--group", "dev"),
}

# npm scoped names and PyPI names, and nothing else. These are parsed out of a package
# manager's own error output, which is untrusted text: it is quoting a string the repository
# controls. Commands run through `create_subprocess_exec` with no shell, so an argument
# cannot become a second command -- but a name is still passed to a tool that will try to
# fetch it, and anything that does not look like a package must not get that far.
_SAFE_PACKAGE_NAME = re.compile(r"^(?:@[a-z0-9][\w.-]*/)?[a-z0-9][\w.-]*$", re.IGNORECASE)


def _manifest_files(package_manager: str | None) -> list[str]:
    """Return the manifest a dependency repair would change, when it is known."""
    if package_manager is None:
        return []
    return _DEPENDENCY_MANIFESTS.get(package_manager.strip().lower(), [])


def safe_package_names(references: Sequence[str]) -> list[str]:
    """Keep only references that actually look like package names.

    Everything else is dropped rather than escaped. A reference the platform cannot vouch for
    is one it should not hand to a package manager, and a repair that installs something
    unrecognisable is worse than one that is never offered.
    """
    return sorted({item for item in references if _SAFE_PACKAGE_NAME.fullmatch(item.strip())})


def repair_commands(package_manager: str | None, undeclared: Sequence[str]) -> list[dict[str, Any]]:
    """Return the commands that would declare the dependencies this repository is missing.

    Empty when there is nothing to declare, when the package manager is unknown, or when it
    has no command that records a declaration. An empty list is meaningful: it says the
    platform cannot carry this repair out itself.
    """
    packages = safe_package_names(undeclared)
    if not packages or package_manager is None:
        return []
    base = _DECLARE_DEV_DEPENDENCY.get(package_manager.strip().lower())
    if base is None:
        return []
    return [
        {
            "command": [*base, *packages],
            "working_directory": None,
            "purpose": (
                "Declare the shared configuration this repository's tooling requires, so its "
                "own checks can run."
            ),
        }
    ]


def repair_is_actionable(
    issues: Sequence[PreflightIssue],
    *,
    classification: FailureClassification | None,
    package_manager: str | None,
) -> bool:
    """Return whether the platform could actually carry out a repair for these findings.

    A dependency the repository never declared is only fixed by declaring it, so without a
    command that does so there is nothing to approve -- and offering an Approve button that
    changes nothing is worse than saying plainly that a person has to make the change.

    A dependency that *is* declared but missing, or an installation that failed, needs no
    command: the granted attempt clones afresh and installs deterministically, which is the
    repair.
    """
    undeclared = [
        reference for item in issues for reference in item.undeclared_references if reference
    ]
    if not undeclared:
        return classification is not None
    return bool(repair_commands(package_manager, undeclared))


__all__ = [
    "OPEN_REPAIR_STATUSES",
    "REPAIRABLE_CATEGORIES",
    "REPAIRABLE_CLASSIFICATIONS",
    "RepairNotFoundError",
    "approved_repair_for",
    "build_repair_payload",
    "current_repairs",
    "find_repair",
    "open_repair_for",
    "repair_is_stale",
    "repair_commands",
    "repair_is_actionable",
    "repairable_issues",
    "safe_package_names",
]
