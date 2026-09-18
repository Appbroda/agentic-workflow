"""Cross-repository integration-review decisions use the immutable shared contract."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal, cast

import pytest

from adapters.llm_adapter import ImageInput, LLMResponse
from agents.integration_reviewer.agent import IntegrationReviewerAgent
from agents.shared.contracts import AgentArtifactError
from artifacts.schemas import (
    ChildWorkflowResultArtifact,
    CompatibilityPolicy,
    IntegrationContractArtifact,
)
from prompts.prompt_loader import PromptLoader
from tools.cross_repository_diffs import RepositoryChange, bounded_repository_change


@pytest.mark.asyncio
async def test_integration_reviewer_routes_unready_contract_owner_to_its_repository() -> None:
    """A backend defect requests only backend correction rather than rerunning the frontend."""
    contract = contract_artifact()
    backend = child_result("backend", "failed", False)
    frontend = child_result("frontend", "approved", True)

    review = await IntegrationReviewerAgent().review(
        feature_id="feature-contract",
        contract=contract,
        child_results=[backend, frontend],
        merge_order=["backend", "frontend"],
    )

    assert review.review_status == "changes_requested"
    assert review.cross_repository_findings[0].responsible_repository_id == "backend"


@pytest.mark.asyncio
async def test_approved_integration_review_claims_no_assessment_it_did_not_perform() -> None:
    """An approval is a contract-conformance gate and must not read as cross-repository review.

    This gate never opens a repository diff, so a consumer calling a provider endpoint with the
    wrong field name passes it. Child reviews cannot catch that either -- each is scoped to its
    own repository -- so an approval implying compatibility was assessed is the only record an
    operator would have of a check nobody ran.
    """
    review = await IntegrationReviewerAgent().review(
        feature_id="feature-contract",
        contract=contract_artifact(),
        child_results=[
            child_result("backend", "approved", True),
            child_result("frontend", "approved", True),
        ],
        merge_order=["backend", "frontend"],
    )

    assert review.review_status == "approved"
    assert review.metadata["assessment_coverage"] == {
        "contract_conformance": "checked",
        "cross_repository_behaviour": "not_checked",
        "security": "not_checked",
        "deployment": "not_checked",
    }
    assert "NOT PERFORMED" in review.security_assessment
    assert "NOT PERFORMED" in review.deployment_assessment
    assert "NOT assessed" in review.compatibility_assessment
    assert any("No repository diff was read" in check for check in review.contract_checks)


class SeamLLMClient:
    """Return one queued seam-review payload and record what it was shown."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Queue the response this fake model will give."""
        self._payload = payload
        self.calls: list[str] = []

    @property
    def vision_capable(self) -> bool:
        """No image reaches this double, so the boundary answers False."""
        return False

    async def respond(
        self,
        *,
        instructions: str,
        input_text: str,
        images: Sequence[ImageInput] = (),
    ) -> LLMResponse:
        """Return the queued payload as a model response."""
        del input_text
        self.calls.append(instructions)

        return LLMResponse(
            response_id="seam-1",
            model="test-model",
            output_text=json.dumps(self._payload),
            input_tokens=None,
            output_tokens=None,
            provider="test-provider",
            reasoning_effort="max",
        )


class StubDiffs:
    """Supply a fixed set of repository changes without a workspace."""

    def __init__(self, *repository_ids: str) -> None:
        """Give each named repository one production file."""
        self._repository_ids = repository_ids

    async def collect(self, *, feature_id: str, child_results: Any) -> list[RepositoryChange]:
        """Return one change per configured repository."""
        del feature_id, child_results
        return [
            bounded_repository_change(
                repository_id=item,
                role=item,
                contract_sections_implemented=["createLogin"] if item == "backend" else [],
                contract_sections_consumed=[] if item == "backend" else ["createLogin"],
                files=[(f"{item}/src/login.js", f"// {item} login source\n")],
            )
            for item in self._repository_ids
        ]


def seam_reviewer(payload: dict[str, Any], *repository_ids: str) -> IntegrationReviewerAgent:
    """Build a reviewer with the seam capability wired to fakes."""
    return IntegrationReviewerAgent(
        llm_client=cast(Any, SeamLLMClient(payload)),
        diff_provider=StubDiffs(*repository_ids),
        prompt_loader=PromptLoader(),
    )


async def approved_pair() -> list[ChildWorkflowResultArtifact]:
    """Return two approved, PR-ready children bound to the active contract."""
    return [child_result("backend", "approved", True), child_result("frontend", "approved", True)]


@pytest.mark.asyncio
async def test_a_seam_mismatch_blocks_and_routes_to_the_repository_that_must_change() -> None:
    """The bug class multi-repository orchestration exists to catch must actually block.

    A frontend calling the backend with the wrong field name passes both child reviews --
    each is scoped to its own repository -- so this is the only gate that can see it.
    """
    reviewer = seam_reviewer(
        {
            "compatibility_assessment": "The consumer sends a field the provider never reads.",
            "findings": [
                {
                    "finding_id": "LOGIN_FIELD_NAME_MISMATCH",
                    "severity": "high",
                    "responsible_repository_id": "frontend",
                    "affected_repository_ids": ["frontend", "backend"],
                    "contract_reference": "createLogin",
                    "description": "The frontend posts `user` where the backend reads `email`.",
                    "evidence": "frontend: body.user; backend: req.body.email",
                    "recommended_fix": "Send `email` as the contract declares.",
                }
            ],
        },
        "backend",
        "frontend",
    )

    review = await reviewer.review(
        feature_id="feature-contract",
        contract=contract_artifact(),
        child_results=await approved_pair(),
        merge_order=["backend", "frontend"],
    )

    assert review.review_status == "changes_requested"
    [finding] = review.cross_repository_findings
    # Routed to the side that has to change, which is not the side the mismatch is visible on.
    assert finding.responsible_repository_id == "frontend"
    assert review.metadata["assessment_coverage"]["cross_repository_behaviour"] == "checked"


@pytest.mark.asyncio
async def test_a_clean_seam_review_approves_and_says_it_looked() -> None:
    """Reviewing and finding nothing must be distinguishable from not having reviewed."""
    reviewer = seam_reviewer(
        {"compatibility_assessment": "Both sides agree on the contract.", "findings": []},
        "backend",
        "frontend",
    )

    review = await reviewer.review(
        feature_id="feature-contract",
        contract=contract_artifact(),
        child_results=await approved_pair(),
        merge_order=["backend", "frontend"],
    )

    assert review.review_status == "approved"
    assert review.metadata["assessment_coverage"]["cross_repository_behaviour"] == "checked"
    assert "NOT assessed" not in review.compatibility_assessment
    assert "was read and compared" in " ".join(review.contract_checks)


@pytest.mark.asyncio
async def test_a_finding_against_a_repository_outside_the_feature_is_refused() -> None:
    """A blocking finding costs a repository a full rework cycle, so it must name a real one.

    Without this, a model naming `api-gateway` -- a plausible repository this feature does not
    contain -- would produce a blocking finding routed to nobody, which the parent reads as an
    integration failure with no responsible repository and cannot act on.
    """
    reviewer = seam_reviewer(
        {
            "compatibility_assessment": "Mismatch found.",
            "findings": [
                {
                    "finding_id": "GATEWAY_MISMATCH",
                    "severity": "high",
                    "responsible_repository_id": "api-gateway",
                    "affected_repository_ids": ["api-gateway"],
                    "contract_reference": "createLogin",
                    "description": "The gateway rewrites the path.",
                    "evidence": "not visible in the changed source",
                    "recommended_fix": "Stop rewriting it.",
                }
            ],
        },
        "backend",
        "frontend",
    )

    with pytest.raises(AgentArtifactError, match="must name a repository in this feature"):
        await reviewer.review(
            feature_id="feature-contract",
            contract=contract_artifact(),
            child_results=await approved_pair(),
            merge_order=["backend", "frontend"],
        )


@pytest.mark.asyncio
async def test_no_repository_change_is_not_reviewed_and_costs_no_model_call() -> None:
    """Nothing to read is still silence, and silence still says so.

    This is the half of the old `len(changes) < 2` bail that was right. 87- Part B split the
    other half out: see `test_one_repository_is_read_against_the_contract_alone`.
    """
    client = SeamLLMClient({"compatibility_assessment": "unused", "findings": []})
    reviewer = IntegrationReviewerAgent(
        llm_client=cast(Any, client),
        diff_provider=StubDiffs(),
        prompt_loader=PromptLoader(),
    )

    review = await reviewer.review(
        feature_id="feature-contract",
        contract=contract_artifact(),
        child_results=[child_result("backend", "approved", True)],
        merge_order=["backend"],
    )

    assert client.calls == []
    assert review.metadata["assessment_coverage"]["cross_repository_behaviour"] == "not_checked"
    assert "NOT assessed" in review.compatibility_assessment


def seam_finding(
    severity: str, finding_id: str = "ADMIN_EXPORT_UI_AUTHORIZATION_MISMATCH"
) -> dict[str, Any]:
    """Return AB-Feature-215's real cross-repository finding at a chosen severity."""
    return {
        "finding_id": finding_id,
        "severity": severity,
        "responsible_repository_id": "frontend",
        "affected_repository_ids": ["frontend"],
        "contract_reference": "exportActiveAdUnitsCsv",
        "description": (
            "The frontend shows the export control when the caller's Employees role permission "
            "is Edit, while the backend requires employee login plus an admin role or the "
            "dedicated export permission."
        ),
        "evidence": "frontend: rolePermissions['Employees'] === 'Edit'; backend: ADMIN_ROLES",
        "recommended_fix": (
            "Gate the control on the administrator representation the backend accepts."
        ),
    }


@pytest.mark.asyncio
async def test_a_medium_seam_finding_blocks_rather_than_shipping_as_a_required_fix() -> None:
    """The severity a cross-repository defect is usually written at must stop publication.

    AB-Feature-215 is the case: this gate found the frontend gating an admin control on a
    permission its own backend does not accept, wrote the correction into `required_fixes`,
    and approved anyway because the finding was `medium`. Both pull requests opened carrying
    a defect the platform had already diagnosed and named a repository to fix.
    """
    reviewer = seam_reviewer(
        {
            "compatibility_assessment": "The two sides disagree on who may export.",
            "findings": [seam_finding("medium")],
        },
        "backend",
        "frontend",
    )

    review = await reviewer.review(
        feature_id="feature-contract",
        contract=contract_artifact(),
        child_results=await approved_pair(),
        merge_order=["backend", "frontend"],
    )

    assert review.review_status == "changes_requested"
    # Routed home rather than published: the remediation path keys on this field.
    [finding] = review.cross_repository_findings
    assert finding.responsible_repository_id == "frontend"
    assert review.required_fixes == [
        "Gate the control on the administrator representation the backend accepts."
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("severity", ["low", "info"])
async def test_advisory_severities_still_approve_and_demand_nothing(severity: str) -> None:
    """`low` is reserved for wording a plan should fix, so promoting it would send back no work."""
    reviewer = seam_reviewer(
        {
            "compatibility_assessment": "The seam agrees.",
            "findings": [seam_finding(severity)],
        },
        "backend",
        "frontend",
    )

    review = await reviewer.review(
        feature_id="feature-contract",
        contract=contract_artifact(),
        child_results=await approved_pair(),
        merge_order=["backend", "frontend"],
    )

    assert review.review_status == "approved"
    assert review.required_fixes == []
    # Not lost, though: the recommendation still reaches a reader through the finding itself.
    assert review.cross_repository_findings[0].recommended_fix


@pytest.mark.asyncio
@pytest.mark.parametrize("severity", ["critical", "high", "medium", "low", "info"])
async def test_an_approved_review_never_carries_a_required_fix(severity: str) -> None:
    """The invariant the artifact validator cannot enforce, pinned where it is produced.

    Stored artifacts are re-validated on read, and ten approved reviews already in the
    database carry a populated `required_fixes` from before this rule existed. So the schema
    has to keep admitting that shape and this is the only place the rule can live: whatever
    the reviewer writes, "approved" and "here is what you must fix" never appear together.
    """
    reviewer = seam_reviewer(
        {
            "compatibility_assessment": "Assessed.",
            "findings": [seam_finding(severity)],
        },
        "backend",
        "frontend",
    )

    review = await reviewer.review(
        feature_id="feature-contract",
        contract=contract_artifact(),
        child_results=await approved_pair(),
        merge_order=["backend", "frontend"],
    )

    assert not (review.review_status == "approved" and review.required_fixes)


def child_result(
    repository_id: str,
    status: Literal["approved", "failed", "waiting_for_contract_change", "cancelled"],
    pull_request_readiness: bool,
) -> ChildWorkflowResultArtifact:
    """Return one parent-scoped repository result for integration review tests.

    Bound to the active contract revision, so a test about owner readiness is not answered by
    an unrelated stale-contract finding that every unbound result would raise.
    """
    contract = contract_artifact()
    return ChildWorkflowResultArtifact(
        schema_version="1.0",
        workflow_id="feature-contract",
        artifact_id=f"011_child_workflow_result.{repository_id}.json",
        producer="child_workflow",
        timestamp=contract.timestamp,
        metadata={
            "contract_artifact_id": contract.artifact_id,
            "contract_version": contract.contract_version,
        },
        validation_status="valid",
        feature_id="feature-contract",
        parent_workflow_id="feature-contract",
        child_workflow_id=f"feature-contract:{repository_id}",
        repository_id=repository_id,
        workstream_id=repository_id,
        branch_name=f"ai/feature-contract/{repository_id}/login",
        workspace_path=f"/workspaces/feature-contract/{repository_id}",
        code_completion_artifact_id="006_code_completion.json",
        review_artifact_id="007_review.json",
        changed_files=[],
        validation_results=[],
        status=status,
        blocking_issues=[] if status == "approved" else ["Repository validation failed."],
        pull_request_readiness=pull_request_readiness,
        contract_sections_consumed=["createLogin"],
        contract_sections_implemented=["createLogin"],
    )


def contract_artifact() -> IntegrationContractArtifact:
    """Return the approved contract used by the integration-review routing test."""
    from datetime import UTC, datetime

    return IntegrationContractArtifact(
        schema_version="1.0",
        workflow_id="feature-contract",
        artifact_id="009_integration_contract.json",
        producer="feature_planner",
        timestamp=datetime(2026, 8, 2, tzinfo=UTC),
        metadata={"source": "test"},
        validation_status="valid",
        feature_id="feature-contract",
        contract_version="1.0.0",
        status="approved",
        api_style="none",
        endpoints=[],
        shared_schemas=[],
        authentication_contract=None,
        authorization_rules=[],
        error_contracts=[],
        event_contracts=[],
        environment_variables=[],
        compatibility_policy=CompatibilityPolicy(
            policy="Additive changes only.",
            breaking_change_allowed=False,
            migration_requirements=[],
            rollback_requirements=["Revert coordinated changes."],
        ),
        owning_workstreams=["backend", "frontend"],
        approved_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
